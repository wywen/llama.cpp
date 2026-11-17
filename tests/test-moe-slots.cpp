// checks the slots of a MoE model whose expert weights are read from slots (llama_model_params.moe_slots):
//   - the model holds the stride the callback returned and the stock expert weights, which have no buffer
//   - the slot layers share one buffer type, a stride of 0 loads the expert weights, bad strides are refused
//     and the MoE cache is refused with slots
//
// the models are the MoE models written by test-llama-archs (-o <dir>)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// the test checks the slot members of the model and that the stock expert weights have no buffer
#include "../src/llama-model.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

// the stride of the slots, a multiple of the alignment of any buffer type and more than an expert of the test models
static constexpr size_t STRIDE = 1 << 24;

// the MoE layers that the loader asks about, with their number of experts; each one reads its experts from slots
using slot_layers = std::map<int32_t, int32_t>;

struct slot_request {
    size_t      stride;
    slot_layers layers;
};

static size_t get_slot_layer(int32_t il, int32_t n_expert, void * ud) {
    auto * req = (slot_request *) ud;
    req->layers[il] = n_expert;
    return req->stride;
}

static llama_model_ptr load_model(const std::string & path, std::vector<ggml_backend_dev_t> devs, slot_request * req) {
    llama_model_params mparams = llama_model_default_params();
    if (req) {
        mparams.moe_slots           = get_slot_layer;
        mparams.moe_slots_user_data = req;
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

// play the caller: check the stride and the stock expert weights of each slot layer
static void check_layers(const llama_model & model, const slot_request & req, const std::string & label, checker & check) {
    bool layout_ok = true;
    bool stock_ok  = true;

    for (const auto & [il, n_expert] : req.layers) {
        const llama_moe_slot_layer * layer = model.moe_slots.layer(il);
        if (layer == nullptr) {
            fprintf(stderr, "%s: layer %d has no slots\n", label.c_str(), il);
            layout_ok = false;
            continue;
        }

        layout_ok = layout_ok && layer->il == il && layer->n_expert == n_expert && layer->stride == req.stride &&
            !layer->members.empty();

        for (const auto & m : layer->members) {
            const ggml_tensor * stock = model.get_tensor(m.name.c_str());
            layout_ok = layout_ok && stock != nullptr && stock == m.stock && stock->ne[2] == n_expert &&
                (size_t) stock->nb[2] <= req.stride && model.moe_slots.of(stock) == layer;
            stock_ok = stock_ok && stock != nullptr && stock->buffer == nullptr;
        }
    }

    check(layout_ok, label + ": the slot layers hold the stride and their stock expert weights");
    check(stock_ok,  label + ": the stock expert weights have no buffer");
}

static void test_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string();
    printf("%s\n", stem.c_str());

    {
        slot_request req = { STRIDE, {} };
        llama_model_ptr model = load_model(path, devs, &req);
        check(!req.layers.empty(), stem + ": the loader asks about the slots of " + std::to_string(req.layers.size()) + " MoE layers");
        check_layers(*model, req, stem, check);

        ggml_backend_buffer_type_t buft = model->moe_slots.layer(req.layers.begin()->first)->buft;
        bool one_buft = buft != nullptr;
        for (const auto & [il, _] : req.layers) {
            one_buft = one_buft && model->moe_slots.layer(il)->buft == buft;
        }
        check(one_buft, stem + ": the slot layers share one buffer type, " + (buft ? ggml_backend_buft_name(buft) : "?"));

        llama_model_ptr stock = load_model(path, devs, nullptr);
        check(stock->moe_slots.empty(), stem + ": a model that loads its expert weights has no slot layers");
    }

    // a layer for which the callback returns 0 loads its expert weights
    {
        slot_request req = { 0, {} };
        llama_model_ptr model = load_model(path, devs, &req);
        check(!req.layers.empty() && model->moe_slots.empty(), stem + ": a stride of 0 loads the expert weights");
    }

    // a stride that is not a multiple of the alignment of the buffer type, or holds less than an expert, is refused
    for (const size_t stride : { STRIDE + 1, (size_t) 1 }) {
        slot_request req = { stride, {} };
        bool refused = false;
        try {
            load_model(path, devs, &req);
        } catch (const std::exception &) {
            refused = g_log.find("MoE slot stride") != std::string::npos;
        }
        check(refused, stem + ": the MoE slot stride " + std::to_string(stride) + " is refused");
    }

    // the MoE cache keeps its own copies of the experts, it does not combine with slots
    {
        slot_request req = { STRIDE, {} };
        llama_model_ptr model = load_model(path, devs, &req);
        llama_context_params cparams = context_params();
        cparams.moe_cache_size = 1024*1024;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        check(!lctx && g_log.find("MoE cache cannot be used") != std::string::npos, stem + ": the MoE cache is refused with slots");
    }
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
            test_model(path, devs, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", name, e.what());
            check(false, std::string(name) + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
