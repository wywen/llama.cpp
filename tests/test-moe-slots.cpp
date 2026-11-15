// checks the slots of a MoE model whose expert weights are read from slots (llama_model_params.moe_slots), as the
//     test fills slots on the host from the file runs that the model reports:
//   - the stock expert weights have no buffer, and the model's slot geometry and members are laid out as documented
//   - the runs fill identical slots from the stock file and from a copy that stores the expert weights one tensor per
//     expert, and the experts of a layer differ
// and that the slot layers share one buffer type, that a layer that loads its expert weights has no slot geometry, that
//     a non-default alignment pads to it, that bad alignments are refused, and that the MoE cache is refused with slots
//
// the models are the MoE models written by test-llama-archs (-o <dir>)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

// the test checks the slot members of the model and that the stock expert weights have no buffer
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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
        size_t alignment = 0) {
    llama_model_params mparams = llama_model_default_params();
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

static llama_context_params context_params() {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 0;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = 64;
    return cparams;
}

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-90s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

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

// play the caller: check the geometry and the members of each slot layer, and fill one slot of each expert on the host
//     from the runs of the expert
// returns the slot contents of each expert, member bytes only
static expert_bytes fill_slots(const llama_model & model, const slot_layers & layers, bool per_expert_file, size_t alignment,
        const std::string & label, checker & check) {
    expert_bytes res;
    bool layout_ok = true;
    bool runs_ok   = true;
    bool stock_ok  = true;

    for (const auto & [il, n] : layers) {
        size_t  slot_size = 0;
        size_t  map_size  = 0;
        int32_t n_expert  = 0;
        const llama_moe_slot_layer * layer = model.moe_slots.layer(il);
        if (layer == nullptr || !llama_model_moe_slot_geometry(&model, il, &slot_size, &map_size, &n_expert)) {
            fprintf(stderr, "%s: layer %d has no slots\n", label.c_str(), il);
            layout_ok = false;
            continue;
        }

        layout_ok = layout_ok && n_expert == n && layer->alignment == alignment && slot_size == layer->slot_size &&
            slot_size % alignment == 0 && map_size % alignment == 0 &&
            map_size == GGML_PAD(n_expert*sizeof(int32_t), alignment) && map_size == layer->map_size() &&
            !layer->members.empty() && layer->members.size() <= LLAMA_MOE_SLOT_MAX_MEMBERS;

        size_t end = 0;
        for (const auto & m : layer->members) {
            const ggml_tensor * stock = model.get_tensor(m.name.c_str());
            layout_ok = layout_ok && stock != nullptr && stock == m.stock && m.offset % alignment == 0 && m.offset >= end &&
                stock->ne[2] == n_expert && m.size == stock->nb[2];
            stock_ok = stock_ok && stock != nullptr && stock->buffer == nullptr;
            end = m.offset + GGML_PAD(m.size, alignment);
        }
        layout_ok = layout_ok && end == slot_size;

        std::vector<uint8_t> slot(slot_size);
        for (int32_t e = 0; e < n_expert; ++e) {
            std::fill(slot.begin(), slot.end(), 0xff);
            llama_moe_expert_run runs[8];
            const int32_t n_runs = llama_model_moe_expert_runs(&model, il, e, runs, 8);
            // a stock file holds each expert weight as one range, the per-expert copy holds an expert as one range
            runs_ok = runs_ok && n_runs == (per_expert_file ? 1 : (int32_t) layer->members.size());
            for (int32_t r = 0; r < n_runs && r < 8; ++r) {
                runs_ok = runs_ok && runs[r].path != nullptr && runs[r].slot_offset + runs[r].size <= slot_size;
                const std::vector<uint8_t> data = read_file(runs[r].path, runs[r].offset, runs[r].size);
                memcpy(slot.data() + runs[r].slot_offset, data.data(), data.size());
            }
            // every member is covered by a run
            for (const auto & m : layer->members) {
                const bool covered = std::any_of(runs, runs + std::min(n_runs, 8), [&](const llama_moe_expert_run & r) {
                    return r.slot_offset <= m.offset && m.offset + m.size <= r.slot_offset + r.size;
                });
                runs_ok = runs_ok && covered;
                std::vector<uint8_t> & bytes = res[{ il, e }];
                bytes.insert(bytes.end(), slot.begin() + m.offset, slot.begin() + m.offset + m.size);
            }
        }
        runs_ok = runs_ok && llama_model_moe_expert_runs(&model, il, n_expert, nullptr, 0) == -1 &&
            llama_model_moe_expert_runs(&model, il, -1, nullptr, 0) == -1;
    }

    check(layout_ok, label + ": slot geometry, the map padded to map_size and the members aligned in slot order");
    check(runs_ok,   label + ": the runs cover each expert's members in " + (per_expert_file ? "one read" : "one read per member"));
    check(stock_ok,  label + ": the stock expert weights have no buffer");
    return res;
}

static expert_bytes test_file(const std::string & path, bool per_expert_file, const std::vector<ggml_backend_dev_t> & devs,
        const std::string & stem, checker & check) {
    const std::string label = stem + (per_expert_file ? " (per-expert file)" : " (stock file)");
    slot_layers layers;
    llama_model_ptr model = load_model(path, devs, &layers);
    check(!layers.empty(), label + ": the loader asks about the slots of " + std::to_string(layers.size()) + " MoE layers");
    return fill_slots(*model, layers, per_expert_file, DEFAULT_ALIGNMENT, label, check);
}

static void test_model(const std::string & path, const std::string & dir, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string();
    printf("%s\n", stem.c_str());

    const std::string path_parts = dir + "/" + stem + "-slot-order.gguf";
    write_per_expert(gguf_file(path), path_parts);

    const expert_bytes bytes_stock = test_file(path,       false, devs, stem, check);
    const expert_bytes bytes_parts = test_file(path_parts, true,  devs, stem, check);
    check(!bytes_stock.empty() && bytes_stock == bytes_parts, stem + ": the stock and the per-expert file fill identical slots");

    // the slots of distinct experts hold distinct bytes
    bool distinct = true;
    for (const auto & [key, bytes] : bytes_stock) {
        for (const auto & [key2, bytes2] : bytes_stock) {
            distinct = distinct && (key.first != key2.first || key.second == key2.second || bytes != bytes2);
        }
    }
    check(distinct, stem + ": the experts of each layer differ");

    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers);
        ggml_backend_buffer_type_t buft = model->moe_slots.layer(layers.begin()->first)->buft;
        bool one_buft = buft != nullptr;
        for (const auto & [il, _] : layers) {
            one_buft = one_buft && model->moe_slots.layer(il)->buft == buft;
        }
        check(one_buft, stem + ": the slot layers share one buffer type, " + (buft ? ggml_backend_buft_name(buft) : "?"));

        llama_model_ptr stock = load_model(path, devs, nullptr);
        size_t  slot_size = 0;
        size_t  map_size  = 0;
        int32_t n_expert  = 0;
        llama_moe_expert_run run;
        check(!llama_model_moe_slot_geometry(stock.get(), layers.begin()->first, &slot_size, &map_size, &n_expert) &&
              llama_model_moe_expert_runs(stock.get(), layers.begin()->first, 0, &run, 1) == -1,
            stem + ": a layer that loads its expert weights has no slot geometry and no runs");
    }

    // a non-default alignment pads the members, the map and the slots to it
    {
        const size_t alignment = 2*DEFAULT_ALIGNMENT;
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers, alignment);
        fill_slots(*model, layers, false, alignment, stem + ", alignment " + std::to_string(alignment), check);
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

    // the MoE cache keeps its own copies of the experts, it does not combine with slots
    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, devs, &layers);
        llama_context_params cparams = context_params();
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
