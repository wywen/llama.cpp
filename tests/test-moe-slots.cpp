// checks that a MoE model whose expert weights are read from caller-filled slots (llama_model_params.moe_slots)
//     computes bitwise identical logits to the same model loaded with its expert weights, when the test fills the
//     slots itself from the file runs that the model reports:
//   - identity: one slot per expert, slot e holds expert e
//   - permuted: one slot per expert and a spare, slot p holds expert perm[p], the spare holds garbage
//   - reduced:  fewer slots than experts, holding exactly the experts the router selects for a short prompt, and a
//               spare holding garbage that every other expert maps to
// each from the stock file and from a copy that stores the expert weights one tensor per expert
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

// write `src` with each expert weight stored as one tensor per expert at LLAMA_MOE_SLOT_ALIGNMENT, expert by
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
        gguf_set_val_u32(kv.get(), GGUF_KEY_GENERAL_ALIGNMENT, LLAMA_MOE_SLOT_ALIGNMENT);
        if (!gguf_write_to_file(kv.get(), path_kv.c_str(), /*only_meta =*/ true)) {
            throw std::runtime_error("failed to write " + path_kv);
        }
        gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        out.reset(gguf_init_from_file(path_kv.c_str(), params));
        std::filesystem::remove(path_kv);
        if (!out || gguf_get_alignment(out.get()) != LLAMA_MOE_SLOT_ALIGNMENT) {
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

// the slot counts that the loader asks for, by layer
struct slot_counts {
    std::map<int32_t, int32_t> n_assigned; // a layer not listed gets one slot per expert
    int32_t n_spare = 0;
    std::map<int32_t, int32_t> n_expert; // as the loader reported them
};

static llama_moe_slot_count get_slot_count(int32_t il, int32_t n_expert, void * ud) {
    auto * counts = (slot_counts *) ud;
    counts->n_expert[il] = n_expert;
    const auto it = counts->n_assigned.find(il);
    return { it == counts->n_assigned.end() ? n_expert : it->second, counts->n_spare };
}

static llama_model_ptr load_model(const std::string & path, std::vector<ggml_backend_dev_t> devs, slot_counts * counts,
        llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO, bool no_alloc = false) {
    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = load_mode;
    mparams.no_alloc  = no_alloc;
    if (counts) {
        mparams.moe_slots           = get_slot_count;
        mparams.moe_slots_user_data = counts;
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

// the logits of `n_batch` tokens in one batch, then of the other tokens one at a time, which take the vector kernels
static std::vector<float> get_logits(llama_model * model, size_t n_tokens, size_t n_batch, selection * sel = nullptr) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 0;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = 64;
    if (sel) {
        cparams.cb_eval           = record_selection;
        cparams.cb_eval_user_data = sel;
    }
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> tokens = get_tokens(n_tokens, n_vocab, 42);

    std::vector<float> ret;
    common_batch batch(lctx.get());
    for (size_t i = 0; i < n_batch; i++) {
        batch.add(tokens[i], i, 0, true);
    }
    if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode");
    }
    for (size_t i = 0; i < n_batch; i++) {
        const float * l = llama_get_logits_ith(lctx.get(), i);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    for (size_t i = n_batch; i < tokens.size(); i++) {
        common_batch one(lctx.get());
        one.add(tokens[i], i, 0, true);
        if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, one.get())) {
            throw std::runtime_error("failed to decode");
        }
        const float * l = llama_get_logits_ith(lctx.get(), 0);
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

// play the caller: fill each slot from the runs of its expert, write the slot maps, and check the layout
// returns the slot contents of each expert, member bytes only
static expert_bytes fill_slots(const llama_model & model, const assignment & assign, bool per_expert_file, const std::string & label, checker & check) {
    expert_bytes res;
    bool layout_ok = true;
    bool runs_ok   = true;
    bool stock_ok  = true;

    for (const auto & [il, slots] : assign) {
        llama_moe_slot_layout layout;
        if (!llama_model_moe_slot_layout(&model, il, &layout)) {
            fprintf(stderr, "%s: layer %d has no slots\n", label.c_str(), il);
            layout_ok = false;
            continue;
        }
        const int64_t n_slots = (int64_t) layout.n_assigned + layout.n_spare;
        layout_ok = layout_ok && n_slots == (int64_t) slots.size() && layout.slot_size % LLAMA_MOE_SLOT_ALIGNMENT == 0 &&
            ggml_backend_buffer_get_size(layout.buffer) >= n_slots*layout.slot_size &&
            ggml_nbytes(layout.region) == n_slots*layout.slot_size && layout.region->data == ggml_backend_buffer_get_base(layout.buffer) &&
            layout.slot_map->type == GGML_TYPE_I32 && layout.slot_map->ne[0] == 1 && layout.slot_map->ne[1] == layout.n_expert;

        size_t end = 0;
        for (int32_t k = 0; k < layout.n_members; ++k) {
            const llama_moe_slot_member & m = layout.members[k];
            const ggml_tensor * stock = model.get_tensor(m.name);
            layout_ok = layout_ok && stock != nullptr && m.offset % LLAMA_MOE_SLOT_ALIGNMENT == 0 && m.offset >= end &&
                m.tensor->type == stock->type && m.tensor->ne[0] == stock->ne[0] && m.tensor->ne[1] == stock->ne[1] &&
                m.tensor->ne[2] == n_slots && m.tensor->nb[1] == stock->nb[1] && m.tensor->nb[2] == layout.slot_size &&
                m.size == stock->nb[2] && (const char *) m.tensor->data == (const char *) layout.region->data + m.offset;
            stock_ok = stock_ok && stock != nullptr && stock->buffer == nullptr;
            end = m.offset + GGML_PAD(m.size, LLAMA_MOE_SLOT_ALIGNMENT);
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
        ggml_backend_tensor_set(layout.region, region.data(), 0, region.size());
        ggml_backend_tensor_set(layout.slot_map, slot_map.data(), 0, slot_map.size()*sizeof(int32_t));
    }

    check(layout_ok, label + ": slot layout");
    check(runs_ok,   label + ": the runs cover each expert's members in " + (per_expert_file ? "one read" : "one read per member"));
    check(stock_ok,  label + ": the stock expert weights have no buffer");
    return res;
}

static assignment identity(const slot_counts & counts) {
    assignment res;
    for (const auto & [il, n] : counts.n_expert) {
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

    // identity: slot e holds expert e
    {
        slot_counts counts;
        llama_model_ptr model = load_model(path, devs, &counts);
        check(!counts.n_expert.empty(), label + ", identity: the loader asks for the slots of " + std::to_string(counts.n_expert.size()) + " MoE layers");
        bytes_identity = fill_slots(*model, identity(counts), per_expert_file, label + ", identity", check);
        check(same_bits(get_logits(model.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), label + ", identity: logits are bitwise identical");
    }

    // permuted: slot p holds expert perm[p], and a spare holds garbage
    {
        slot_counts counts;
        counts.n_spare = 1;
        llama_model_ptr model = load_model(path, devs, &counts);
        assignment assign = identity(counts);
        for (auto & [_, slots] : assign) {
            // a rotation: no expert keeps its own slot
            std::rotate(slots.begin(), slots.begin() + 1, slots.end());
            slots.push_back(-1);
        }
        fill_slots(*model, assign, per_expert_file, label + ", permuted", check);
        check(same_bits(get_logits(model.get(), N_TOKENS_LONG, N_BATCH_LONG), ref_long), label + ", permuted: logits are bitwise identical");
    }

    // reduced: only the selected experts have slots, every other expert reads a spare that holds garbage
    {
        slot_counts counts;
        counts.n_spare = 1;
        for (const auto & [il, experts] : sel) {
            counts.n_assigned[il] = (int32_t) experts.size();
        }
        llama_model_ptr model = load_model(path, devs, &counts);
        assignment assign;
        int n_reduced = 0;
        for (const auto & [il, n] : counts.n_expert) {
            const auto it = sel.find(il);
            if (it == sel.end()) {
                throw std::runtime_error("no selection recorded for layer " + std::to_string(il));
            }
            assign[il].assign(it->second.begin(), it->second.end());
            assign[il].push_back(-1);
            n_reduced += (int32_t) it->second.size() < n ? 1 : 0;
        }
        fill_slots(*model, assign, per_expert_file, label + ", reduced", check);
        // unobserved, as the reference: observing the router splits fused GPU kernels, and the slot graph's lookup
        //     after the top-k fuses differently from the stock graph, so observed runs compare like-for-like only
        //     on the CPU; the recorded selection only chooses which experts the slots hold
        const std::vector<float> logits = get_logits(model.get(), N_TOKENS_SHORT, N_BATCH_SHORT);
        if (devs.empty()) {
            selection sel_slots;
            get_logits(model.get(), N_TOKENS_SHORT, N_BATCH_SHORT, &sel_slots);
            check(sel_slots == sel, label + ", reduced: the router selects the experts it selects in the stock model");
        }
        check(same_bits(logits, ref_short),
            label + ", reduced (" + std::to_string(n_reduced) + " of " + std::to_string(assign.size()) + " layers with fewer slots than experts): logits are bitwise identical");
    }
}

static void test_model(const std::string & path, const std::string & dir, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string();
    printf("%s\n", stem.c_str());

    std::vector<float> ref_long;
    std::vector<float> ref_short;
    selection sel;
    {
        llama_model_ptr model = load_model(path, devs, nullptr);
        ref_long  = get_logits(model.get(), N_TOKENS_LONG, N_BATCH_LONG);
        ref_short = get_logits(model.get(), N_TOKENS_SHORT, N_BATCH_SHORT);
        const std::vector<float> routed = get_logits(model.get(), N_TOKENS_SHORT, N_BATCH_SHORT, &sel);
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

    // the memory-fit pass, which allocates nothing, counts the memory of the model that it would allocate
    {
        auto model_memory = [&](bool no_alloc, size_t * region_bytes) {
            slot_counts counts;
            counts.n_spare = 1;
            llama_model_ptr model = load_model(path, devs, &counts, LLAMA_LOAD_MODE_NONE, no_alloc);
            *region_bytes = 0;
            for (const auto & [il, _] : counts.n_expert) {
                llama_moe_slot_layout layout;
                if (llama_model_moe_slot_layout(model.get(), il, &layout)) {
                    *region_bytes += ggml_nbytes(layout.region);
                }
            }
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = 0;
            llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
            if (!lctx) {
                throw std::runtime_error("failed to create a context:\n" + g_log);
            }
            size_t res = 0;
            for (const auto & [_, mb] : llama_get_memory_breakdown(lctx.get())) {
                res += mb.model;
            }
            return res;
        };
        size_t regions_fit   = 0;
        size_t regions_alloc = 0;
        const size_t fit   = model_memory(true,  &regions_fit);
        const size_t alloc = model_memory(false, &regions_alloc);
        check(regions_fit > 0 && regions_fit == regions_alloc && fit == alloc && alloc >= regions_alloc,
            stem + ": the memory-fit pass counts the model memory with its slots (" + std::to_string(fit) + " bytes)");
    }

    // more assigned slots than experts
    {
        slot_counts counts;
        counts.n_assigned[sel.begin()->first] = 1 << 20;
        llama_model_params mparams = llama_model_default_params();
        mparams.moe_slots           = get_slot_count;
        mparams.moe_slots_user_data = &counts;
        mparams.n_gpu_layers        = 0;
        g_log.clear();
        llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
        check(!model && g_log.find("invalid MoE slot count") != std::string::npos, stem + ": more assigned slots than experts are refused");
    }

    // the MoE cache keeps its own copies of the experts, it does not combine with slots
    {
        slot_counts counts;
        llama_model_ptr model = load_model(path, devs, &counts);
        llama_context_params cparams = llama_context_default_params();
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
