// checks that a MoE model whose expert weights are read from caller-filled slots (llama_model_params.moe_slots,
//     llama_context_params.moe_slots) computes bitwise identical logits to the same model loaded with its expert
//     weights, when the test fills the slots of a context itself from the file runs that the model reports:
//   - identity: one slot per expert, slot e holds expert e; then the experts move to other slots of the same context
//   - permuted: one slot per expert and a spare, slot p holds expert perm[p], the spare holds garbage
//   - reduced:  fewer slots than experts, holding exactly the experts the router selects for a short prompt, and a
//               spare holding garbage that every other expert maps to
// each from the stock file and from a copy that stores the expert weights one tensor per expert
// and that two contexts on one model have their own slots, that the slots can be of a buffer type that the caller
//     chooses, that the model reports the geometry of the slots of every context, that a non-default alignment
//     computes the same, and that a context without the slots of a layer that its graph reads fails to initialize
// and that a context with two slot sets computes the stock logits on whichever set each graph reads, rebuilds its
//     graph on a set switch without reserving its scheduler or growing its compute buffers, and counts the memory of
//     every set
//
// the models are the MoE models written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

// the test checks that the stock expert weights have no buffer, and the memory that the slots take
#include "../src/llama-ext.h"
#include "../src/llama-model.h"

// the test implements a buffer type
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

// the alignment of the members and slots when llama_model_params.moe_slot_alignment is 0
static constexpr size_t DEFAULT_ALIGNMENT = 16384;

// the position of an expert weight in a slot, or -1
static int slot_rank(const std::string & name) {
    static const std::regex re(R"(^blk\.\d+\.ffn_(gate_up|up|gate|down)_exps\.weight$)");
    static const char * kinds[] = { "gate_up", "up", "gate", "down" };
    std::smatch m;
    if (!std::regex_match(name, m, re)) {
        return -1;
    }
    for (int r = 0; r < 4; ++r) {
        if (m[1].str() == kinds[r]) {
            return r;
        }
    }
    return -1;
}

static int layer_of(const std::string & name) {
    int il = -1;
    sscanf(name.c_str(), "blk.%d.", &il);
    return il;
}

// a GGUF file read with its tensor data
struct gguf_file {
    gguf_context_ptr gguf;
    ggml_context_ptr ctx;

    explicit gguf_file(const std::string & path) {
        ggml_context * ctx_raw = nullptr;
        gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_raw };
        gguf.reset(gguf_init_from_file(path.c_str(), params));
        if (!gguf) {
            throw std::runtime_error("failed to read " + path);
        }
        ctx.reset(ctx_raw);
    }

    int64_t n_tensors() const {
        return gguf_get_n_tensors(gguf.get());
    }

    ggml_tensor * tensor(int64_t i) const {
        return ggml_get_tensor(ctx.get(), gguf_get_tensor_name(gguf.get(), i));
    }
};

// write `src` with each expert weight stored as one tensor per expert at DEFAULT_ALIGNMENT, expert by
//     expert in slot order, so that the file holds each expert's weights with the gaps of a slot
static void write_per_expert(const gguf_file & src, const std::string & path) {
    std::map<int, std::vector<std::pair<int, ggml_tensor *>>> layers; // layer -> (rank, tensor)
    int64_t n_parts = 0;
    for (int64_t i = 0; i < src.n_tensors(); ++i) {
        ggml_tensor * t = src.tensor(i);
        const int rank = slot_rank(t->name);
        if (rank >= 0) {
            layers[layer_of(t->name)].emplace_back(rank, t);
            n_parts += t->ne[2];
        }
    }
    for (auto & [_, members] : layers) {
        std::sort(members.begin(), members.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    }

    // the alignment of a context is read from a file, so the KV pairs go through one
    gguf_context_ptr out;
    {
        const std::string path_kv = path + ".kv";
        gguf_context_ptr kv(gguf_init_empty());
        gguf_set_kv(kv.get(), src.gguf.get());
        gguf_set_val_u32(kv.get(), GGUF_KEY_GENERAL_ALIGNMENT, DEFAULT_ALIGNMENT);
        if (!gguf_write_to_file(kv.get(), path_kv.c_str(), /*only_meta =*/ true)) {
            throw std::runtime_error("failed to write " + path_kv);
        }
        gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        out.reset(gguf_init_from_file(path_kv.c_str(), params));
        std::filesystem::remove(path_kv);
        if (!out || gguf_get_alignment(out.get()) != DEFAULT_ALIGNMENT) {
            throw std::runtime_error("failed to read back " + path_kv);
        }
    }

    ggml_init_params ip = { /*.mem_size =*/ (size_t) (n_parts + 1)*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context_ptr ctx_parts(ggml_init(ip));
    std::set<int> written;
    for (int64_t i = 0; i < src.n_tensors(); ++i) {
        ggml_tensor * t = src.tensor(i);
        if (slot_rank(t->name) < 0) {
            gguf_add_tensor(out.get(), t);
            continue;
        }
        const int il = layer_of(t->name);
        if (!written.insert(il).second) {
            continue;
        }
        const auto & members = layers.at(il);
        for (int64_t e = 0; e < t->ne[2]; ++e) {
            for (const auto & [_, m] : members) {
                ggml_tensor * p = ggml_new_tensor_2d(ctx_parts.get(), m->type, m->ne[0], m->ne[1]);
                p->data = (uint8_t *) m->data + e*m->nb[2];
                const std::string stem = std::string(m->name).substr(0, strlen(m->name) - strlen(".weight"));
                ggml_set_name(p, (stem + "." + std::to_string(e) + ".weight").c_str());
                gguf_add_tensor(out.get(), p);
            }
        }
    }
    if (!gguf_write_to_file(out.get(), path.c_str(), false)) {
        throw std::runtime_error("failed to write " + path);
    }
}

// the MoE layers that the loader asks about, with their number of experts; each one reads its experts from slots
using slot_layers = std::map<int32_t, int32_t>;

static bool get_slot_layer(int32_t il, int32_t n_expert, void * ud) {
    (*(slot_layers *) ud)[il] = n_expert;
    return true;
}

static llama_model_ptr load_model(const std::string & path, std::vector<ggml_backend_dev_t> devs, slot_layers * layers,
        size_t alignment = 0, llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO, bool no_alloc = false) {
    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = load_mode;
    mparams.no_alloc  = no_alloc;
    if (layers) {
        mparams.moe_slots           = get_slot_layer;
        mparams.moe_slots_user_data = layers;
        mparams.moe_slot_alignment  = alignment;
    }
    if (devs.empty()) {
        mparams.n_gpu_layers = 0;
    } else {
        devs.push_back(nullptr);
        mparams.devices = devs.data();
    }
    g_log.clear();
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        throw std::runtime_error("failed to load " + path + ":\n" + g_log);
    }
    return model;
}

// the slot counts that a context asks for, by layer, and the buffer type of every layer's slots
struct slot_counts {
    std::map<int32_t, int32_t> n_slots; // a layer not listed gets one slot per expert and n_spare
    int32_t n_spare = 0;
    ggml_backend_buffer_type_t buft = nullptr;
    int32_t buft_set = -1; // the only set that uses buft, -1 = every set
};

static llama_moe_slot_pool_params get_slot_count(int32_t il, int32_t n_expert, int32_t /*set*/, void * ud) {
    const auto * counts = (const slot_counts *) ud;
    const auto it = counts->n_slots.find(il);
    return { it == counts->n_slots.end() ? n_expert + counts->n_spare : it->second, counts->buft };
}

// a caller's buffer type that wraps another one and recognizes the slot buffers at their allocation: it records each
// allocation and returns the wrapped type's buffer unchanged, so that the backend sees a buffer of its own type
struct wrapper_buft {
    ggml_backend_buffer_type wrapper;
    ggml_backend_buffer_type_t inner;
    size_t alignment = 0;        // 0 = the inner one
    size_t max_size  = SIZE_MAX; // the inner one, at most
    std::vector<size_t> allocs;  // the sizes allocated
    std::vector<ggml_backend_buffer_t> bufs; // the buffers returned
    bool own_type = false;       // the buffers returned are of the wrapper's type instead of the inner one's
    bool flip_host = false;      // the wrapper reports the opposite of the inner one's is_host

    wrapper_buft(ggml_backend_buffer_type_t inner, ggml_backend_dev_t dev) : inner(inner) {
        wrapper.iface = {
            /*.get_name         =*/ [](ggml_backend_buffer_type_t) { return "wrapper"; },
            /*.alloc_buffer     =*/ [](ggml_backend_buffer_type_t buft, size_t size) {
                auto * w = (wrapper_buft *) buft->context;
                w->allocs.push_back(size);
                ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(w->inner, size);
                if (buf != nullptr && w->own_type) {
                    buf->buft = buft;
                }
                w->bufs.push_back(buf);
                return buf;
            },
            /*.alloc_buffer_n   =*/ nullptr,
            /*.get_alignment    =*/ [](ggml_backend_buffer_type_t buft) {
                auto * w = (wrapper_buft *) buft->context;
                return w->alignment ? w->alignment : ggml_backend_buft_get_alignment(w->inner);
            },
            /*.get_max_size     =*/ [](ggml_backend_buffer_type_t buft) {
                auto * w = (wrapper_buft *) buft->context;
                return std::min(w->max_size, ggml_backend_buft_get_max_size(w->inner));
            },
            /*.get_alloc_size   =*/ [](ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
                return ggml_backend_buft_get_alloc_size(((wrapper_buft *) buft->context)->inner, t);
            },
            /*.get_alloc_size_n =*/ nullptr,
            /*.is_host          =*/ [](ggml_backend_buffer_type_t buft) {
                auto * w = (wrapper_buft *) buft->context;
                return ggml_backend_buft_is_host(w->inner) != w->flip_host;
            },
        };
        wrapper.device  = dev;
        wrapper.context = this;
    }

    wrapper_buft(const wrapper_buft &) = delete;
    wrapper_buft & operator=(const wrapper_buft &) = delete;

    ggml_backend_buffer_type_t get() {
        return &wrapper;
    }
};

static std::vector<llama_token> get_tokens(uint32_t n_tokens, uint32_t n_vocab, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// the experts that the router selects in each layer
using selection = std::map<int32_t, std::set<int32_t>>;

static bool record_selection(ggml_tensor * t, bool ask, void * ud) {
    const bool is_topk = strncmp(ggml_get_name(t), "ffn_moe_topk-", 13) == 0;
    if (ask || !is_topk) {
        return ask ? is_topk : true;
    }
    // a view of the first entries of each row of the argsort
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    auto & experts = (*(selection *) ud)[atoi(ggml_get_name(t) + 13)];
    for (int64_t r = 0; r < t->ne[1]; ++r) {
        const int32_t * row = (const int32_t *) (bytes.data() + r*t->nb[1]);
        experts.insert(row, row + t->ne[0]);
    }
    return true;
}

static llama_context_params context_params(const slot_counts * counts, selection * sel) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 0;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = 64;
    if (counts) {
        cparams.moe_slots           = get_slot_count;
        cparams.moe_slots_user_data = (void *) counts;
    }
    if (sel) {
        cparams.cb_eval           = record_selection;
        cparams.cb_eval_user_data = sel;
    }
    return cparams;
}

static llama_context_ptr new_context(llama_model * model, const slot_counts * counts, selection * sel = nullptr) {
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, context_params(counts, sel)));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

// the logits of `n_batch` tokens in one batch, then of the other tokens one at a time, which take the vector kernels
// starts from an empty memory, so that a context computes the same logits each time
// a set >= 0 is the MoE slot set that the batch, or each single token, reads
static std::vector<float> get_logits(llama_context * lctx, size_t n_tokens, size_t n_batch, int32_t set_batch = -1, int32_t set_one = -1) {
    llama_memory_clear(llama_get_memory(lctx), true);

    const llama_model * model = llama_get_model(lctx);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> tokens = get_tokens(n_tokens, n_vocab, 42);

    std::vector<float> ret;
    common_batch batch(lctx);
    for (size_t i = 0; i < n_batch; i++) {
        batch.add(tokens[i], i, 0, true);
    }
    if (set_batch >= 0) {
        llama_set_moe_slot_set(lctx, set_batch);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode");
    }
    for (size_t i = 0; i < n_batch; i++) {
        const float * l = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    for (size_t i = n_batch; i < tokens.size(); i++) {
        common_batch one(lctx);
        one.add(tokens[i], i, 0, true);
        if (set_one >= 0) {
            llama_set_moe_slot_set(lctx, set_one);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, one.get())) {
            throw std::runtime_error("failed to decode");
        }
        const float * l = llama_get_logits_ith(lctx, 0);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    return ret;
}

static constexpr size_t N_TOKENS_LONG  = 16;
static constexpr size_t N_BATCH_LONG   = 12;
static constexpr size_t N_TOKENS_SHORT = 3;
static constexpr size_t N_BATCH_SHORT  = 2;

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-90s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

// the expert in each slot of a layer, -1 for garbage
using assignment = std::map<int32_t, std::vector<int32_t>>;

// the member bytes of each expert, by layer, as the runs fill them
using expert_bytes = std::map<std::pair<int32_t, int32_t>, std::vector<uint8_t>>;

static std::vector<uint8_t> read_file(const std::string & path, uint64_t offset, uint64_t size) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> res(size);
    f.seekg((std::streamoff) offset);
    f.read((char *) res.data(), (std::streamsize) size);
    if (!f) {
        throw std::runtime_error("failed to read " + path);
    }
    return res;
}

// play the caller: fill each slot of a context from the runs of its expert, write the slot maps, and check the layout
// returns the slot contents of each expert, member bytes only
static expert_bytes fill_slots(llama_context * lctx, const assignment & assign, bool per_expert_file, size_t alignment,
        const std::string & label, checker & check, int32_t set = 0) {
    const llama_model & model = *llama_get_model(lctx);

    expert_bytes res;
    bool layout_ok = true;
    bool runs_ok   = true;
    bool stock_ok  = true;

    for (const auto & [il, slots] : assign) {
        llama_moe_slot_layout layout;
        if (!llama_context_moe_slot_layout(lctx, il, set, &layout)) {
            fprintf(stderr, "%s: layer %d has no slots in set %d\n", label.c_str(), il, set);
            layout_ok = false;
            continue;
        }
        const int64_t n_slots = layout.n_slots;
        const char *  base    = (const char *) layout.region->data;

        // the slot map at offset 0, zero padded to the alignment, then the slots
        layout_ok = layout_ok && n_slots == (int64_t) slots.size() && layout.alignment == alignment &&
            layout.slot_size % alignment == 0 && layout.map_size % alignment == 0 &&
            layout.map_size == GGML_PAD(layout.n_expert*sizeof(int32_t), alignment) &&
            ggml_backend_buffer_get_size(layout.buffer) >= layout.map_size + n_slots*layout.slot_size &&
            ggml_nbytes(layout.region) == layout.map_size + n_slots*layout.slot_size && base == ggml_backend_buffer_get_base(layout.buffer) &&
            layout.slot_map->type == GGML_TYPE_I32 && layout.slot_map->ne[0] == 1 && layout.slot_map->ne[1] == layout.n_expert &&
            layout.slot_map->buffer == layout.buffer && (const char *) layout.slot_map->data == base;

        size_t end = 0;
        for (int32_t k = 0; k < layout.n_members; ++k) {
            const llama_moe_slot_member & m = layout.members[k];
            const ggml_tensor * stock = model.get_tensor(m.name);
            layout_ok = layout_ok && stock != nullptr && m.offset % alignment == 0 && m.offset >= end &&
                m.tensor->type == stock->type && m.tensor->ne[0] == stock->ne[0] && m.tensor->ne[1] == stock->ne[1] &&
                m.tensor->ne[2] == n_slots && m.tensor->nb[1] == stock->nb[1] && m.tensor->nb[2] == layout.slot_size &&
                m.size == stock->nb[2] && (const char *) m.tensor->data == base + layout.map_size + m.offset;
            stock_ok = stock_ok && stock != nullptr && stock->buffer == nullptr;
            end = m.offset + GGML_PAD(m.size, alignment);
        }
        layout_ok = layout_ok && end == layout.slot_size;

        std::vector<uint8_t> region(n_slots*layout.slot_size, 0xff);
        std::vector<int32_t> slot_map(layout.n_expert, -1);
        for (int64_t p = 0; p < n_slots; ++p) {
            const int32_t e = slots[p];
            if (e < 0) {
                continue;
            }
            slot_map[e] = (int32_t) p;

            llama_moe_expert_run runs[8];
            const int32_t n_runs = llama_model_moe_expert_runs(&model, il, e, runs, 8);
            // a stock file holds each expert weight as one range, the per-expert copy holds an expert as one range
            runs_ok = runs_ok && n_runs == (per_expert_file ? 1 : layout.n_members);
            uint8_t * slot = region.data() + p*layout.slot_size;
            for (int32_t r = 0; r < n_runs && r < 8; ++r) {
                runs_ok = runs_ok && runs[r].path != nullptr && runs[r].slot_offset + runs[r].size <= layout.slot_size;
                const std::vector<uint8_t> data = read_file(runs[r].path, runs[r].offset, runs[r].size);
                memcpy(slot + runs[r].slot_offset, data.data(), data.size());
            }
            // every member is covered by a run
            for (int32_t k = 0; k < layout.n_members; ++k) {
                const llama_moe_slot_member & m = layout.members[k];
                const bool covered = std::any_of(runs, runs + std::min(n_runs, 8), [&](const llama_moe_expert_run & r) {
                    return r.slot_offset <= m.offset && m.offset + m.size <= r.slot_offset + r.size;
                });
                runs_ok = runs_ok && covered;
                std::vector<uint8_t> & bytes = res[{ il, e }];
                bytes.insert(bytes.end(), slot + m.offset, slot + m.offset + m.size);
            }
        }
        // the experts that no slot holds read the garbage slot
        const auto garbage = std::find(slots.begin(), slots.end(), -1);
        for (int32_t & s : slot_map) {
            if (s < 0) {
                s = garbage == slots.end() ? 0 : (int32_t) (garbage - slots.begin());
            }
        }
        ggml_backend_tensor_set(layout.region, region.data(), layout.map_size, region.size());
        ggml_backend_tensor_set(layout.slot_map, slot_map.data(), 0, slot_map.size()*sizeof(int32_t));
    }

    check(layout_ok, label + ": slot layout, the map at offset 0 and slot 0 at map_size");
    check(runs_ok,   label + ": the runs cover each expert's members in " + (per_expert_file ? "one read" : "one read per member"));
    check(stock_ok,  label + ": the stock expert weights have no buffer");
    return res;
}

// the slot map and its padding of each layer start out zeroed
static bool maps_zeroed(llama_context * lctx, const slot_layers & layers) {
    bool ok = true;
    for (const auto & [il, _] : layers) {
        llama_moe_slot_layout layout;
        if (!llama_context_moe_slot_layout(lctx, il, 0, &layout)) {
            return false;
        }
        std::vector<uint8_t> bytes(layout.map_size, 0xff);
        ggml_backend_tensor_get(layout.region, bytes.data(), 0, bytes.size());
        ok = ok && std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == 0; });
    }
    return ok;
}

static assignment identity(const slot_layers & layers) {
    assignment res;
    for (const auto & [il, n] : layers) {
        std::vector<int32_t> & slots = res[il];
        slots.resize(n);
        std::iota(slots.begin(), slots.end(), 0);
    }
    return res;
}

static void test_file(const std::string & path, bool per_expert_file, const std::vector<ggml_backend_dev_t> & devs,
        const std::vector<float> & ref_long, const std::vector<float> & ref_short, const selection & sel,
        const std::string & stem, expert_bytes & bytes_identity, checker & check) {
    const std::string label = stem + (per_expert_file ? " (per-expert file)" : " (stock file)");

    // identity: slot e holds expert e; then the experts move to other slots of the same context
    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers);
        check(!layers.empty(), label + ", identity: the loader asks about the slots of " + std::to_string(layers.size()) + " MoE layers");
        const slot_counts counts;
        llama_context_ptr lctx = new_context(model.get(), &counts);
        check(maps_zeroed(lctx.get(), layers), label + ", identity: the slot maps start out zeroed");
        bytes_identity = fill_slots(lctx.get(), identity(layers), per_expert_file, DEFAULT_ALIGNMENT, label + ", identity", check);
        check(same_bits(get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), label + ", identity: logits are bitwise identical");

        assignment moved = identity(layers);
        for (auto & [_, slots] : moved) {
            std::reverse(slots.begin(), slots.end());
        }
        fill_slots(lctx.get(), moved, per_expert_file, DEFAULT_ALIGNMENT, label + ", moved", check);
        check(same_bits(get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long),
            label + ", moved: the experts in reversed slots of the same context, logits are bitwise identical");
    }

    // permuted: slot p holds expert perm[p], and a spare holds garbage
    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers);
        slot_counts counts;
        counts.n_spare = 1;
        llama_context_ptr lctx = new_context(model.get(), &counts);
        assignment assign = identity(layers);
        for (auto & [_, slots] : assign) {
            // a rotation: no expert keeps its own slot
            std::rotate(slots.begin(), slots.begin() + 1, slots.end());
            slots.push_back(-1);
        }
        fill_slots(lctx.get(), assign, per_expert_file, DEFAULT_ALIGNMENT, label + ", permuted", check);
        check(same_bits(get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), label + ", permuted: logits are bitwise identical");
    }

    // reduced: only the selected experts have slots, every other expert reads a spare that holds garbage
    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers);
        slot_counts counts;
        assignment assign;
        int n_reduced = 0;
        for (const auto & [il, n] : layers) {
            const auto it = sel.find(il);
            if (it == sel.end()) {
                throw std::runtime_error("no selection recorded for layer " + std::to_string(il));
            }
            assign[il].assign(it->second.begin(), it->second.end());
            assign[il].push_back(-1);
            counts.n_slots[il] = (int32_t) assign[il].size();
            n_reduced += (int32_t) it->second.size() < n ? 1 : 0;
        }
        // unobserved, as the reference: observing the router splits fused GPU kernels, and the slot graph's lookup
        //     after the top-k fuses differently from the stock graph, so observed runs compare like-for-like only
        //     on the CPU; the recorded selection only chooses which experts the slots hold
        llama_context_ptr lctx = new_context(model.get(), &counts);
        fill_slots(lctx.get(), assign, per_expert_file, DEFAULT_ALIGNMENT, label + ", reduced", check);
        const std::vector<float> logits = get_logits(lctx.get(), N_TOKENS_SHORT, N_BATCH_SHORT);
        if (devs.empty()) {
            selection sel_slots;
            llama_context_ptr lctx_observed = new_context(model.get(), &counts, &sel_slots);
            fill_slots(lctx_observed.get(), assign, per_expert_file, DEFAULT_ALIGNMENT, label + ", reduced, observed", check);
            get_logits(lctx_observed.get(), N_TOKENS_SHORT, N_BATCH_SHORT);
            check(sel_slots == sel, label + ", reduced: the router selects the experts it selects in the stock model");
        }
        check(same_bits(logits, ref_short),
            label + ", reduced (" + std::to_string(n_reduced) + " of " + std::to_string(assign.size()) + " layers with fewer slots than experts): logits are bitwise identical");
    }
}

// two contexts on one model have their own slots: one holds the right experts, the other the wrong ones
static void test_two_contexts(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, const std::vector<float> & ref_long,
        const std::string & stem, checker & check) {
    slot_layers layers;
    llama_model_ptr model = load_model(path, devs, &layers);
    const slot_counts counts;
    llama_context_ptr ctx_a = new_context(model.get(), &counts);
    llama_context_ptr ctx_b = new_context(model.get(), &counts);

    bool separate = true;
    for (const auto & [il, _] : layers) {
        llama_moe_slot_layout la;
        llama_moe_slot_layout lb;
        separate = separate && llama_context_moe_slot_layout(ctx_a.get(), il, 0, &la) && llama_context_moe_slot_layout(ctx_b.get(), il, 0, &lb) &&
            la.buffer != lb.buffer && la.region != lb.region && la.slot_map != lb.slot_map && la.members[0].tensor != lb.members[0].tensor;
    }
    check(separate, stem + ", two contexts: each has its own slot buffers and tensors");

    fill_slots(ctx_a.get(), identity(layers), false, DEFAULT_ALIGNMENT, stem + ", two contexts, A", check);

    // B: slot p holds expert p + 1, but the map sends expert e to slot e
    assignment rotated = identity(layers);
    for (auto & [_, slots] : rotated) {
        std::rotate(slots.begin(), slots.begin() + 1, slots.end());
    }
    fill_slots(ctx_b.get(), rotated, false, DEFAULT_ALIGNMENT, stem + ", two contexts, B", check);
    for (const auto & [il, n] : layers) {
        llama_moe_slot_layout layout;
        llama_context_moe_slot_layout(ctx_b.get(), il, 0, &layout);
        std::vector<int32_t> slot_map(n);
        std::iota(slot_map.begin(), slot_map.end(), 0);
        ggml_backend_tensor_set(layout.slot_map, slot_map.data(), 0, slot_map.size()*sizeof(int32_t));
    }

    check(same_bits(get_logits(ctx_a.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), stem + ", two contexts: A computes the stock logits");
    check(!same_bits(get_logits(ctx_b.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), stem + ", two contexts: B, with the wrong experts, does not");
    check(same_bits(get_logits(ctx_a.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), stem + ", two contexts: A still computes the stock logits after B");

    // B with its map written for its slots
    fill_slots(ctx_b.get(), rotated, false, DEFAULT_ALIGNMENT, stem + ", two contexts, B fixed", check);
    check(same_bits(get_logits(ctx_b.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), stem + ", two contexts: B with its map fixed computes the stock logits");
}

// the caller chooses the buffer type of the slots: the context calls a wrapper of the layer's buffer type once per
//     layer, for the documented size, uses the buffer it returns, and computes the stock logits
// a buffer type on another device, with a greater alignment or with a max size too small for the slots is refused
static void test_pool_buft(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, const std::vector<float> & ref_long,
        const std::string & stem, checker & check) {
    slot_layers layers;
    llama_model_ptr model = load_model(path, devs, &layers);

    ggml_backend_buffer_type_t inner = model->moe_slots.layer(layers.begin()->first)->buft;
    bool one_buft = true;
    for (const auto & [il, _] : layers) {
        one_buft = one_buft && model->moe_slots.layer(il)->buft == inner;
    }
    check(one_buft, stem + ", buffer type: the slot layers share one buffer type, " + ggml_backend_buft_name(inner));
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(inner);

    {
        wrapper_buft w(inner, dev);
        slot_counts counts;
        counts.n_spare = 1;
        counts.buft    = w.get();
        llama_context_ptr lctx = new_context(model.get(), &counts);

        // the layers allocate in layer order
        std::vector<size_t> expected;
        std::vector<ggml_backend_buffer_t> buffers;
        bool layout_ok = true;
        for (const auto & [il, n] : layers) {
            size_t  slot_size = 0;
            size_t  map_size  = 0;
            int32_t n_expert  = 0;
            llama_moe_slot_layout layout = {};
            layout_ok = layout_ok && llama_model_moe_slot_geometry(model.get(), il, &slot_size, &map_size, &n_expert) &&
                llama_context_moe_slot_layout(lctx.get(), il, 0, &layout);
            expected.push_back(map_size + (size_t) (n + 1)*slot_size);
            buffers.push_back(layout.buffer);
        }
        check(layout_ok && w.allocs == expected, stem + ", buffer type: alloc_buffer is called once per slot layer (" +
            std::to_string(w.allocs.size()) + "), for map_size + n_slots*slot_size bytes");
        check(layout_ok && w.bufs == buffers, stem + ", buffer type: the buffer it returns is the layer's slot buffer, of type " +
            (buffers.empty() || !buffers[0] ? "?" : ggml_backend_buft_name(ggml_backend_buffer_get_type(buffers[0]))));

        assignment assign = identity(layers);
        for (auto & [_, slots] : assign) {
            slots.push_back(-1);
        }
        fill_slots(lctx.get(), assign, false, DEFAULT_ALIGNMENT, stem + ", buffer type", check);
        check(same_bits(get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), stem + ", buffer type: logits are bitwise identical");
        check(w.allocs.size() == expected.size(), stem + ", buffer type: the decodes allocate nothing more of it");
    }

    // a context on a model loaded with no_alloc allocates nothing
    {
        slot_layers layers_fit;
        llama_model_ptr model_fit = load_model(path, devs, &layers_fit, 0, LLAMA_LOAD_MODE_NONE, true);
        wrapper_buft w(model_fit->moe_slots.layer(layers_fit.begin()->first)->buft, dev);
        slot_counts counts;
        counts.buft = w.get();
        llama_context_ptr lctx = new_context(model_fit.get(), &counts);
        check(w.allocs.empty(), stem + ", buffer type: a context on a model loaded with no_alloc allocates nothing of it");
    }

    auto refused = [&](wrapper_buft & w, const char * error) {
        slot_counts counts;
        counts.buft = w.get();
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), context_params(&counts, nullptr)));
        return !lctx && w.allocs.empty() && g_log.find(error) != std::string::npos;
    };

    // another device: with a single device, e.g. a CPU-only build, there is none to try
    ggml_backend_dev_t other = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && other == nullptr; ++i) {
        if (ggml_backend_dev_get(i) != dev) {
            other = ggml_backend_dev_get(i);
        }
    }
    if (other != nullptr) {
        wrapper_buft w(inner, other);
        check(refused(w, "is not on the device"), stem + ", buffer type: a buffer type on " + ggml_backend_dev_name(other) + " is refused");
    } else {
        printf("  %-90s %s\n", (stem + ", buffer type: a buffer type on another device is refused").c_str(), "SKIPPED (one device)");
    }

    {
        wrapper_buft w(inner, dev);
        w.alignment = 2*DEFAULT_ALIGNMENT;
        check(refused(w, "is greater than the MoE slot alignment"), stem + ", buffer type: an alignment greater than the slot alignment is refused");
    }
    {
        wrapper_buft w(inner, dev);
        w.max_size = DEFAULT_ALIGNMENT;
        check(refused(w, "is less than the"), stem + ", buffer type: a max size less than the slots is refused");
    }
}

// the model reports the slot geometry that every context lays its slots out with
static void test_geometry(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, const std::string & stem, checker & check) {
    slot_layers layers;
    llama_model_ptr model = load_model(path, devs, &layers);
    slot_counts counts_a;
    slot_counts counts_b;
    counts_b.n_spare = 3;
    llama_context_ptr ctx_a = new_context(model.get(), &counts_a);
    llama_context_ptr ctx_b = new_context(model.get(), &counts_b);

    bool ok = true;
    for (const auto & [il, n] : layers) {
        size_t  slot_size = 0;
        size_t  map_size  = 0;
        int32_t n_expert  = 0;
        ok = ok && llama_model_moe_slot_geometry(model.get(), il, &slot_size, &map_size, &n_expert) && n_expert == n;
        for (llama_context * lctx : { ctx_a.get(), ctx_b.get() }) {
            llama_moe_slot_layout layout;
            ok = ok && llama_context_moe_slot_layout(lctx, il, 0, &layout) &&
                layout.slot_size == slot_size && layout.map_size == map_size && layout.n_expert == n_expert;
        }
    }
    check(ok, stem + ", geometry: the model's slot geometry is the layout of contexts with " + std::to_string(counts_a.n_spare) +
        " and " + std::to_string(counts_b.n_spare) + " spare slots");

    llama_model_ptr stock = load_model(path, devs, nullptr);
    size_t  slot_size = 0;
    size_t  map_size  = 0;
    int32_t n_expert  = 0;
    check(!llama_model_moe_slot_geometry(stock.get(), layers.begin()->first, &slot_size, &map_size, &n_expert),
        stem + ", geometry: a layer that loads its expert weights has none");
}

// the slot counts of each slot set of a context, by layer, and the buffer type of every layer's slots
struct set_counts {
    std::vector<std::map<int32_t, int32_t>> n_slots; // [n_sets]; a layer not listed gets one slot per expert
    ggml_backend_buffer_type_t buft = nullptr;
    int32_t buft_set = -1; // the only set that uses buft, -1 = every set
};

static llama_moe_slot_pool_params get_set_slot_count(int32_t il, int32_t n_expert, int32_t set, void * ud) {
    const auto * counts = (const set_counts *) ud;
    const auto & n_slots = counts->n_slots.at(set);
    const auto it = n_slots.find(il);
    return { it == n_slots.end() ? n_expert : it->second, counts->buft_set < 0 || counts->buft_set == set ? counts->buft : nullptr };
}

// the graphs a context prepares, and the compute buffer plans its scheduler admits
struct graph_log {
    int n_reserve = 0;
    int n_grow    = 0; // the plans that grow a compute buffer
    std::vector<std::vector<std::string>> nodes; // per graph: op, name, type and shape of each node
    std::vector<std::set<std::string>>    slots; // per graph: the slot tensors that its nodes read
};

static bool log_reserve(ggml_backend_sched_t sched, const ggml_backend_sched_buffer_reservation * res, size_t n, void * ud) {
    auto * log = (graph_log *) ud;
    log->n_reserve++;
    // each entry is max(current, planned) for one chunk, so a plan grows a buffer when its entries sum to more
    std::map<int, size_t> planned;
    for (size_t i = 0; i < n; ++i) {
        planned[res[i].backend_index] += res[i].size;
    }
    for (const auto & [b, size] : planned) {
        if (size > ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, b))) {
            log->n_grow++;
            break;
        }
    }
    return true;
}

// the compute buffer size of each backend of the context's scheduler
static std::vector<size_t> compute_sizes(llama_context * lctx) {
    ggml_backend_sched_t sched = llama_get_sched(lctx);
    std::vector<size_t> res;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        res.push_back(ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, i)));
    }
    return res;
}

static bool log_graph(ggml_backend_sched_t, ggml_cgraph * gf, void * ud) {
    auto * log = (graph_log *) ud;
    std::vector<std::string> & nodes = log->nodes.emplace_back();
    std::set<std::string>    & slots = log->slots.emplace_back();
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        char sig[256];
        snprintf(sig, sizeof(sig), "%s %s %s [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]",
            ggml_op_desc(t), t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
        nodes.emplace_back(sig);
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (t->src[k] != nullptr && (strstr(t->src[k]->name, ".slots") || strstr(t->src[k]->name, "_slot_map"))) {
                slots.insert(t->src[k]->name);
            }
        }
    }
    return true;
}

static llama_context_ptr new_set_context(llama_model * model, const set_counts & counts, graph_log * log = nullptr) {
    llama_context_params cparams = context_params(nullptr, nullptr);
    cparams.moe_slots           = get_set_slot_count;
    cparams.moe_slots_user_data = (void *) &counts;
    cparams.moe_slot_sets       = (uint32_t) counts.n_slots.size();
    if (log) {
        cparams.cb_reserve                 = log_reserve;
        cparams.cb_reserve_user_data       = log;
        cparams.cb_graph_prepare           = log_graph;
        cparams.cb_graph_prepare_user_data = log;
    }
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

// the bytes of the slot regions of a set
static size_t set_bytes(llama_context * lctx, const slot_layers & layers, int32_t set) {
    size_t res = 0;
    for (const auto & [il, _] : layers) {
        llama_moe_slot_layout layout;
        if (llama_context_moe_slot_layout(lctx, il, set, &layout)) {
            res += ggml_nbytes(layout.region);
        }
    }
    return res;
}

static size_t context_bytes(llama_context * lctx) {
    size_t res = 0;
    for (const auto & [_, mb] : llama_get_memory_breakdown(lctx)) {
        res += mb.context;
    }
    return res;
}

// a context with two slot sets: set 0 holds the experts the router selects for the short prompt, in a permuted order
//     after a garbage slot, set 1 holds every expert in identity order; each graph reads the set chosen for it
static void test_slot_sets(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, const std::vector<float> & ref_long,
        const std::vector<float> & ref_short, const selection & sel, const std::string & stem, checker & check) {
    const std::string label = stem + ", slot sets";

    slot_layers layers;
    llama_model_ptr model = load_model(path, devs, &layers);

    set_counts counts;
    counts.n_slots.resize(2);
    assignment small;
    for (const auto & [il, n] : layers) {
        const auto it = sel.find(il);
        if (it == sel.end()) {
            throw std::runtime_error("no selection recorded for layer " + std::to_string(il));
        }
        std::vector<int32_t> & slots = small[il];
        slots.assign(it->second.rbegin(), it->second.rend());
        slots.insert(slots.begin(), -1);
        counts.n_slots[0][il] = (int32_t) slots.size();
    }
    const assignment full = identity(layers);

    // bitwise: decodes that alternate between the sets compute the stock logits
    {
        llama_context_ptr lctx = new_set_context(model.get(), counts);
        bool distinct = true;
        bool sizes    = true;
        int  n_small  = 0;
        for (const auto & [il, n] : layers) {
            n_small += counts.n_slots[0].at(il) < n ? 1 : 0;
            llama_moe_slot_layout l0;
            llama_moe_slot_layout l1;
            distinct = distinct && llama_context_moe_slot_layout(lctx.get(), il, 0, &l0) && llama_context_moe_slot_layout(lctx.get(), il, 1, &l1) &&
                l0.buffer != l1.buffer && l0.slot_map != l1.slot_map && l0.members[0].tensor != l1.members[0].tensor;
            sizes = sizes && l0.n_slots == counts.n_slots[0].at(il) && l1.n_slots == n;
        }
        check(distinct, label + ": each set has its own slot buffers and tensors");
        check(sizes,    label + ": each set has its own number of slots (set 0 has fewer slots than experts in " +
            std::to_string(n_small) + " of " + std::to_string(layers.size()) + " layers)");

        fill_slots(lctx.get(), small, false, DEFAULT_ALIGNMENT, label + ", set 0", check, 0);
        fill_slots(lctx.get(), full,  false, DEFAULT_ALIGNMENT, label + ", set 1", check, 1);

        bool ok = true;
        for (int pass = 0; pass < 2; ++pass) {
            ok = ok && same_bits(get_logits(lctx.get(), N_TOKENS_LONG,  N_BATCH_LONG,  1, 1), ref_long);
            ok = ok && same_bits(get_logits(lctx.get(), N_TOKENS_SHORT, N_BATCH_SHORT, 0, 0), ref_short);
        }
        check(ok, label + ": decodes alternating between set 1 (long prompt) and set 0 (short prompt) are bitwise identical");
        check(same_bits(get_logits(lctx.get(), N_TOKENS_SHORT, N_BATCH_SHORT, 1, 0), ref_short),
            label + ": a batch on set 1, then a token on set 0, in one sequence is bitwise identical");
        check(same_bits(get_logits(lctx.get(), N_TOKENS_SHORT, N_BATCH_SHORT, 0, 1), ref_short),
            label + ": a batch on set 0, then a token on set 1, in one sequence is bitwise identical");

        // an out-of-range set is refused, and the context keeps reading its set: set 0 cannot compute the long prompt
        bool refused = llama_set_moe_slot_set(lctx.get(), 1);
        for (const int32_t set : { 2, -1 }) {
            g_log.clear();
            refused = refused && !llama_set_moe_slot_set(lctx.get(), set) && g_log.find("is out of range") != std::string::npos;
        }
        check(refused && same_bits(get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long),
            label + ": set 1 is taken, sets 2 and -1 are refused, the context keeps reading set 1");

        llama_moe_slot_layout layout;
        check(!llama_context_moe_slot_layout(lctx.get(), layers.begin()->first, 2, &layout) &&
              !llama_context_moe_slot_layout(lctx.get(), layers.begin()->first, -1, &layout),
            label + ": the layout of an out-of-range set is refused");
    }

    // graph shape: a set switch rebuilds the graph with the same nodes and splits, and needs no scheduler reserve; the
    //     compute buffers that the context reserved at initialization cover every set
    // with the slot counts of the sets equal, a switch places the graph as the previous one; with different counts, a
    //     backend that sizes an expert matmul's scratch by its slot count places it again within the same buffers
    set_counts equal;
    equal.n_slots.resize(2);
    assignment reversed = full;
    for (auto & [_, slots] : reversed) {
        std::reverse(slots.begin(), slots.end());
    }
    for (const bool same_counts : { false, true }) {
        const std::string lbl = label + (same_counts ? ", equal counts" : ", different counts");
        const set_counts & cnt = same_counts ? equal : counts;

        graph_log log;
        llama_context_ptr lctx = new_set_context(model.get(), cnt, &log);
        const std::vector<size_t> sizes_init = compute_sizes(lctx.get());
        fill_slots(lctx.get(), same_counts ? reversed : small, false, DEFAULT_ALIGNMENT, lbl + ", set 0", check, 0);
        fill_slots(lctx.get(), full, false, DEFAULT_ALIGNMENT, lbl + ", set 1", check, 1);

        llama_memory_clear(llama_get_memory(lctx.get()), true);
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const std::vector<llama_token> tokens = get_tokens(64, n_vocab, 42);
        llama_pos pos = 0;
        auto decode = [&](size_t n, int32_t set) {
            llama_set_moe_slot_set(lctx.get(), set);
            const bool pending = llama_sched_reserve_pending(lctx.get());
            common_batch batch(lctx.get());
            for (size_t i = 0; i < n; ++i, ++pos) {
                batch.add(tokens[pos % tokens.size()], pos, 0, true);
            }
            if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                throw std::runtime_error("failed to decode");
            }
            return pending;
        };

        // a first single-token graph settles the placement of that shape
        decode(1, 1);
        const size_t first = log.nodes.size();
        const int n_reserve = log.n_reserve;
        const int n_grow    = log.n_grow;
        const int n_splits  = ggml_backend_sched_get_n_splits(llama_get_sched(lctx.get()));
        const int32_t n_reused = llama_perf_context(lctx.get()).n_reused;

        const int32_t sets[] = { 0, 0, 1, 1, 0 };
        bool pending = false;
        bool splits  = true;
        std::vector<int32_t> reused;
        for (size_t i = 0; i < std::size(sets); ++i) {
            pending = decode(1, sets[i]) || pending;
            splits  = splits && ggml_backend_sched_get_n_splits(llama_get_sched(lctx.get())) == n_splits;
            reused.push_back(llama_perf_context(lctx.get()).n_reused - n_reused);
        }
        const int n_replan = log.n_reserve - n_reserve;
        check(log.nodes.size() == first + std::size(sets), lbl + ": one graph per decode");
        // decodes 0 and 2 and 4 switch sets, decodes 1 and 3 repeat theirs
        check(reused == std::vector<int32_t>({ 0, 1, 1, 2, 2 }), lbl + ": a set switch rebuilds the graph, a repeated set reuses it");
        check(!pending, lbl + ": a set switch leaves no scheduler reserve pending");
        check(splits,   lbl + ": the graph of every set has the same splits (" + std::to_string(n_splits) + ")");
        if (same_counts) {
            check(n_replan == 0, lbl + ": a set switch places the graph as the previous one (" + std::to_string(n_replan) + " placements)");
        } else {
            // 3 switches; a backend whose scratch does not depend on the slot count places nothing new
            check(n_replan <= 3 && log.n_grow == n_grow,
                lbl + ": a set switch places the graph at most once, within the buffers (" + std::to_string(n_replan) + " placements on 3 switches)");
        }

        bool same_nodes = true;
        bool own_slots  = true;
        for (size_t i = 0; i < std::size(sets); ++i) {
            const size_t g = first + i;
            same_nodes = same_nodes && log.nodes[g] == log.nodes[first];
            for (const std::string & name : log.slots[g]) {
                own_slots = own_slots && (name.find(".set1") != std::string::npos) == (sets[i] == 1);
            }
            own_slots = own_slots && log.slots[g].size() >= layers.size();
        }
        check(same_nodes, lbl + ": the node list is identical across sets (" + std::to_string(log.nodes[first].size()) + " nodes)");
        check(own_slots,  lbl + ": each graph reads the slot tensors of its set only");

        // a batch of the reserve width on each set, the widest graph a decode can build, fits the initial buffers
        llama_memory_clear(llama_get_memory(lctx.get()), true);
        pos = 0;
        decode(64, 1);
        decode(64, 0);
        check(log.n_grow == n_grow && compute_sizes(lctx.get()) == sizes_init,
            lbl + ": the compute buffers reserved at initialization hold every set, single tokens and full batches");
    }

    // a set with no slots for a layer, even the initial set 0: the context initializes, its reserve reads the set that
    //     has slots, and a graph that reads the layer on the set without them fails to build
    {
        set_counts holes;
        holes.n_slots.resize(2);
        holes.n_slots[0][layers.begin()->first] = 0;
        llama_context_ptr lctx = new_set_context(model.get(), holes);
        llama_moe_slot_layout layout;
        check(!llama_context_moe_slot_layout(lctx.get(), layers.begin()->first, 0, &layout) &&
               llama_context_moe_slot_layout(lctx.get(), layers.begin()->first, 1, &layout),
            label + ": a layer with 0 slots in set 0 has no layout in set 0, and one in set 1");
        fill_slots(lctx.get(), full, false, DEFAULT_ALIGNMENT, label + ", holes, set 1", check, 1);

        g_log.clear();
        bool failed = false;
        try {
            get_logits(lctx.get(), N_TOKENS_SHORT, N_BATCH_SHORT);
        } catch (const std::exception &) {
            failed = g_log.find("but the context has no slots for it in set 0") != std::string::npos;
        }
        check(failed, label + ": a decode on a set with 0 slots for a layer that its graph reads fails to build");

        llama_set_moe_slot_set(lctx.get(), 1);
        check(same_bits(get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), label + ": the context then decodes on set 1");
    }

    // the sets of a layer read their slots from buffers of one device and one host-ness, not of one type object: a
    // buft that returns the layer's buffers, and one that returns buffers of its own type on the same device with the
    // same host-ness, are taken for one set; one whose host-ness differs is refused
    {
        ggml_backend_buffer_type_t inner = model->moe_slots.layer(layers.begin()->first)->buft;
        struct variant { bool own_type; bool flip_host; };
        // a GPU backend recognizes its buffer types by their functions, so only the CPU takes a buft of another object
        const bool on_cpu = ggml_backend_dev_type(ggml_backend_buft_get_device(inner)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        for (const variant v : { variant { false, false }, variant { true, false }, variant { true, true } }) {
            if (v.own_type && !v.flip_host && !on_cpu) {
                fprintf(stderr, "%s: skipped, a buft of its own type on the layer's device is accepted on the CPU only\n", label.c_str());
                continue;
            }
            wrapper_buft w(inner, ggml_backend_buft_get_device(inner));
            w.own_type  = v.own_type;
            w.flip_host = v.flip_host;
            set_counts one_buft = counts;
            one_buft.buft     = w.get();
            one_buft.buft_set = 1;
            llama_context_params cparams = context_params(nullptr, nullptr);
            cparams.moe_slots           = get_set_slot_count;
            cparams.moe_slots_user_data = (void *) &one_buft;
            cparams.moe_slot_sets       = (uint32_t) one_buft.n_slots.size();
            g_log.clear();
            llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
            if (v.flip_host) {
                check(!lctx && g_log.find("one device and of one host-ness") != std::string::npos,
                    label + ": set 1 in buffers of another host-ness than set 0's is refused");
            } else if (v.own_type) {
                check(lctx != nullptr, label + ": set 1 in buffers of its own type on set 0's device and host-ness initializes");
            } else {
                check(lctx != nullptr, label + ": set 1 through a buft that returns buffers of set 0's type initializes");
            }
        }
    }

    // a context with no slot set is refused
    {
        llama_context_params cparams = context_params(nullptr, nullptr);
        cparams.moe_slots           = get_set_slot_count;
        cparams.moe_slots_user_data = (void *) &counts;
        cparams.moe_slot_sets       = 0;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        check(!lctx && g_log.find("must be at least 1") != std::string::npos, label + ": 0 slot sets are refused");
    }

    // accounting: the context memory holds the slots of every set, each set allocated through its buffer type
    {
        ggml_backend_buffer_type_t inner = model->moe_slots.layer(layers.begin()->first)->buft;
        wrapper_buft w(inner, ggml_backend_buft_get_device(inner));
        set_counts one = counts;
        one.n_slots.resize(1);
        set_counts two = counts;
        two.buft = w.get();

        llama_context_ptr ctx_one = new_set_context(model.get(), one);
        llama_context_ptr ctx_two = new_set_context(model.get(), two);
        const size_t bytes0 = set_bytes(ctx_two.get(), layers, 0);
        const size_t bytes1 = set_bytes(ctx_two.get(), layers, 1);
        check(bytes0 > 0 && bytes1 > 0 && set_bytes(ctx_one.get(), layers, 0) == bytes0 &&
              context_bytes(ctx_two.get()) == context_bytes(ctx_one.get()) + bytes1,
            label + ": the memory breakdown sums the sets (" + std::to_string(bytes0) + " + " + std::to_string(bytes1) + " bytes)");

        std::vector<size_t> expected;
        for (int32_t set = 0; set < 2; ++set) {
            for (const auto & [il, _] : layers) {
                llama_moe_slot_layout layout;
                llama_context_moe_slot_layout(ctx_two.get(), il, set, &layout);
                expected.push_back(ggml_nbytes(layout.region));
            }
        }
        check(w.allocs == expected, label + ": alloc_buffer is called once per layer and set (" + std::to_string(w.allocs.size()) + ")");

        // the memory-fit pass, which allocates nothing, counts the slots of every set
        slot_layers layers_fit;
        llama_model_ptr model_fit = load_model(path, devs, &layers_fit, 0, LLAMA_LOAD_MODE_NONE, true);
        llama_context_ptr ctx_fit = new_set_context(model_fit.get(), counts);
        llama_model_ptr model_alloc = load_model(path, devs, &layers_fit, 0, LLAMA_LOAD_MODE_NONE, false);
        llama_context_ptr ctx_alloc = new_set_context(model_alloc.get(), counts);
        check(context_bytes(ctx_fit.get()) == context_bytes(ctx_alloc.get()) && context_bytes(ctx_alloc.get()) >= bytes0 + bytes1,
            label + ": the memory-fit pass counts the slots of every set");
    }
}

// the graph a context builds has the same nodes whatever the number of outputs: the last layer of a graph runs only its
// outputs, and a node list that depended on their number would not match the scheduler's reserved allocation plan
static void test_graph_structure(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, const std::string & stem, checker & check) {
    const std::string label = stem + ", graph structure";

    slot_layers layers;
    llama_model_ptr model = load_model(path, devs, &layers);
    set_counts counts;
    counts.n_slots.resize(1);
    llama_context_ptr lctx = new_set_context(model.get(), counts);

    constexpr uint32_t n_tokens = 8;
    auto nodes = [&](uint32_t n_outputs) {
        ggml_cgraph * gf = llama_graph_reserve(lctx.get(), n_tokens, 1, n_outputs);
        std::vector<std::string> res;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const ggml_tensor * t = ggml_graph_node(gf, i);
            res.push_back(std::string(ggml_op_desc(t)) + " " + t->name + " " + ggml_type_name(t->type));
        }
        return res;
    };
    const auto one   = nodes(1);
    const auto two   = nodes(2);
    const auto some  = nodes(3);
    const auto all   = nodes(n_tokens);
    check(!one.empty() && one == two && one == some && one == all,
        label + ": the node list is the same for 1, 2, 3 and " + std::to_string(n_tokens) + " outputs (" +
        std::to_string(one.size()) + ", " + std::to_string(two.size()) + ", " + std::to_string(some.size()) + ", " + std::to_string(all.size()) + " nodes)");
}

static void test_model(const std::string & path, const std::string & dir, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string();
    printf("%s\n", stem.c_str());

    std::vector<float> ref_long;
    std::vector<float> ref_short;
    selection sel;
    {
        llama_model_ptr model = load_model(path, devs, nullptr);
        llama_context_ptr lctx = new_context(model.get(), nullptr);
        ref_long  = get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG);
        ref_short = get_logits(lctx.get(), N_TOKENS_SHORT, N_BATCH_SHORT);
        llama_context_ptr lctx_observed = new_context(model.get(), nullptr, &sel);
        const std::vector<float> routed = get_logits(lctx_observed.get(), N_TOKENS_SHORT, N_BATCH_SHORT);
        // on a GPU, observing the router's nodes splits fused kernels into separate ops that round differently,
        //     so the observed and the unobserved logits are equal only on the CPU
        if (devs.empty()) {
            check(same_bits(routed, ref_short), stem + ": recording the routing leaves the logits unchanged");
        }
    }

    const std::string path_parts = dir + "/" + stem + "-slot-order.gguf";
    write_per_expert(gguf_file(path), path_parts);

    expert_bytes bytes_stock;
    expert_bytes bytes_parts;
    test_file(path,       false, devs, ref_long, ref_short, sel, stem, bytes_stock, check);
    test_file(path_parts, true,  devs, ref_long, ref_short, sel, stem, bytes_parts, check);
    check(!bytes_stock.empty() && bytes_stock == bytes_parts, stem + ": the stock and the per-expert file fill identical slots");

    // a slot that holds the wrong expert changes the logits only if the experts differ
    bool distinct = true;
    for (const auto & [key, bytes] : bytes_stock) {
        for (const auto & [key2, bytes2] : bytes_stock) {
            distinct = distinct && (key.first != key2.first || key.second == key2.second || bytes != bytes2);
        }
    }
    check(distinct, stem + ": the experts of each layer differ");

    test_two_contexts(path, devs, ref_long, stem, check);
    test_pool_buft(path, devs, ref_long, stem, check);
    test_geometry(path, devs, stem, check);
    test_slot_sets(path, devs, ref_long, ref_short, sel, stem, check);
    test_graph_structure(path, devs, stem, check);

    // a non-default alignment pads the members, the map and the slots to it, and computes the same
    {
        const size_t alignment = 2*DEFAULT_ALIGNMENT;
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers, alignment);
        const slot_counts counts;
        llama_context_ptr lctx = new_context(model.get(), &counts);
        fill_slots(lctx.get(), identity(layers), false, alignment, stem + ", alignment " + std::to_string(alignment), check);
        check(same_bits(get_logits(lctx.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long),
            stem + ", alignment " + std::to_string(alignment) + ": logits are bitwise identical");
    }

    // an alignment that is not a power of two, or below the alignment of the buffer type, is refused
    for (const size_t alignment : { 3*DEFAULT_ALIGNMENT/2, (size_t) 8 }) {
        slot_layers layers;
        bool refused = false;
        try {
            load_model(path, devs, &layers, alignment);
        } catch (const std::exception &) {
            refused = g_log.find(alignment == 8 ? "is less than the alignment" : "is not a power of two") != std::string::npos;
        }
        check(refused, stem + ": the MoE slot alignment " + std::to_string(alignment) + " is refused");
    }

    // a context without the slots of a layer that its graph reads fails to initialize, as does a negative count
    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers);
        slot_counts counts;
        counts.n_slots[layers.begin()->first] = 0;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), context_params(&counts, nullptr)));
        check(!lctx && g_log.find("but the context has no slots for it") != std::string::npos,
            stem + ": a context with 0 slots for a layer that its graph reads fails to initialize");

        g_log.clear();
        lctx.reset(llama_init_from_model(model.get(), context_params(nullptr, nullptr)));
        check(!lctx && g_log.find("but the context has no slots for it") != std::string::npos,
            stem + ": a context with no slot counts fails to initialize");

        counts.n_slots[layers.begin()->first] = -1;
        g_log.clear();
        lctx.reset(llama_init_from_model(model.get(), context_params(&counts, nullptr)));
        check(!lctx && g_log.find("invalid number of MoE slots") != std::string::npos, stem + ": a negative slot count is refused");
    }

    // the memory-fit pass, which allocates nothing, counts the memory of the context that it would allocate
    {
        auto memory = [&](bool no_alloc, size_t * region_bytes) {
            slot_layers layers;
            llama_model_ptr model = load_model(path, devs, &layers, 0, LLAMA_LOAD_MODE_NONE, no_alloc);
            slot_counts counts;
            counts.n_spare = 1;
            llama_context_ptr lctx = new_context(model.get(), &counts);
            *region_bytes = 0;
            for (const auto & [il, _] : layers) {
                llama_moe_slot_layout layout;
                if (llama_context_moe_slot_layout(lctx.get(), il, 0, &layout)) {
                    *region_bytes += ggml_nbytes(layout.region);
                }
            }
            std::pair<size_t, size_t> res = { 0, 0 }; // model, context
            for (const auto & [_, mb] : llama_get_memory_breakdown(lctx.get())) {
                res.first  += mb.model;
                res.second += mb.context;
            }
            return res;
        };
        size_t regions_fit   = 0;
        size_t regions_alloc = 0;
        const auto fit   = memory(true,  &regions_fit);
        const auto alloc = memory(false, &regions_alloc);
        check(regions_fit > 0 && regions_fit == regions_alloc && fit == alloc && alloc.second >= regions_alloc,
            stem + ": the memory-fit pass counts the context memory with its slots (" + std::to_string(fit.second) + " bytes)");
    }

    // the MoE cache keeps its own copies of the experts, it does not combine with slots
    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers);
        const slot_counts counts;
        llama_context_params cparams = context_params(&counts, nullptr);
        cparams.moe_cache_size = 1024*1024;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        check(!lctx && g_log.find("MoE cache cannot be used") != std::string::npos, stem + ": the MoE cache is refused with slots");
    }

    std::filesystem::remove(path_parts);
}

int main(int argc, char ** argv) {
    std::string dir_models;
    bool all_devices = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            dir_models = argv[++i];
        } else if (strcmp(argv[i], "--all-devices") == 0) {
            all_devices = true;
        } else {
            dir_models.clear();
            break;
        }
    }
    if (dir_models.empty()) {
        fprintf(stderr, "usage: %s --models <dir written by test-llama-archs -o> [--all-devices]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    // the log of a load is printed when it fails
    llama_log_set([](ggml_log_level, const char * text, void *) {
        g_log += text;
    }, nullptr);

    // the CPU (no devices), or with --all-devices every GPU device
    std::vector<ggml_backend_dev_t> devs;
    if (all_devices) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                devs.push_back(dev);
            }
        }
        if (devs.empty()) {
            fprintf(stderr, "--all-devices: no GPU device\n");
            return 1;
        }
    }

    // separate gate and up experts, fused gate_up experts, and fused gate_up experts with a per-expert down scale
    const char * models[] = { "qwen3moe-moe.gguf", "qwen35moe-moe.gguf", "gemma4-moe.gguf" };

    checker check;
    for (const char * name : models) {
        const std::string path = dir_models + "/" + name;
        try {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("missing " + path);
            }
            test_model(path, dir_models, devs, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", name, e.what());
            check(false, std::string(name) + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
