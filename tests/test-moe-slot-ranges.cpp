// checks that a MoE slot set split into slot ranges (llama_moe_slot_pool_params.n_range_splits) is laid out as asked:
//   - the context reports the range splits of every layer, for 2 and 3 ranges
//   - bad splits, too many splits, and slot sets of a layer with different numbers of ranges are refused
//
// the models are the MoE models written by test-llama-archs (-o <dir>)

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static std::string g_log;

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-100s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        fflush(stdout);
        n_fail += ok ? 0 : 1;
    }
};

// the MoE layers that the loader asks about, with their number of experts; each one reads its experts from slots
using slot_layers = std::map<int32_t, int32_t>;

static bool get_slot_layer(int32_t il, int32_t n_expert, void * ud) {
    (*(slot_layers *) ud)[il] = n_expert;
    return true;
}

static llama_model_ptr load_model(const std::string & path, ggml_backend_dev_t dev, slot_layers * layers) {
    llama_model_params mparams = llama_model_default_params();
    if (layers) {
        mparams.moe_slots           = get_slot_layer;
        mparams.moe_slots_user_data = layers;
    }
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    if (dev == nullptr) {
        mparams.n_gpu_layers = 0;
    } else {
        mparams.devices = devs;
    }
    g_log.clear();
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        throw std::runtime_error("failed to load " + path + ":\n" + g_log);
    }
    return model;
}

// the expert in each slot of a layer, -1 for garbage
using assignment = std::map<int32_t, std::vector<int32_t>>;

// the slots a context asks for: per set, the slots of each layer as `assign` holds them, split at `splits`
struct slot_spec {
    std::vector<const assignment *> sets;      // [n_sets]
    std::vector<std::vector<int32_t>> splits;  // [n_sets] the range splits, the same for every layer
};

static llama_moe_slot_pool_params get_slots(const llama_context * /*ctx*/, int32_t il, int32_t /*n_expert*/, int32_t set, void * ud) {
    const auto * spec = (const slot_spec *) ud;
    llama_moe_slot_pool_params res = {};
    res.n_slots = (int32_t) spec->sets.at(set)->at(il).size();
    const auto & splits = spec->splits.at(set);
    res.n_range_splits = (int32_t) splits.size();
    for (size_t i = 0; i < splits.size() && i < LLAMA_MOE_SLOT_MAX_RANGES - 1; ++i) {
        res.range_splits[i] = splits[i];
    }
    return res;
}

static llama_context_ptr new_context(llama_model * model, const slot_spec & spec) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx               = 0;
    cparams.n_threads           = 4;
    cparams.n_threads_batch     = 4;
    cparams.n_ubatch            = 64;
    cparams.moe_slots           = get_slots;
    cparams.moe_slots_user_data = (void *) &spec;
    cparams.moe_slot_sets       = (uint32_t) spec.sets.size();
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

// one slot per expert, in a shuffled order
static assignment bank(const slot_layers & layers) {
    assignment res;
    std::mt19937 gen(7);
    for (const auto & [il, n] : layers) {
        std::vector<int32_t> & slots = res[il];
        slots.resize(n);
        std::iota(slots.begin(), slots.end(), 0);
        std::shuffle(slots.begin(), slots.end(), gen);
    }
    return res;
}

// the fewest slots of a layer in an assignment
static int32_t min_slots(const assignment & a) {
    int32_t res = INT32_MAX;
    for (const auto & [_, slots] : a) {
        res = std::min(res, (int32_t) slots.size());
    }
    return res;
}

// uneven splits of n slots into n_ranges ranges
static std::vector<int32_t> splits_of(int32_t n, int32_t n_ranges) {
    std::vector<int32_t> res;
    if (n_ranges == 2) {
        res = { std::max(1, n/3) };
    } else if (n_ranges == 3) {
        res = { 1, std::max(2, n/2) };
    }
    return res;
}

// the context reports the range splits it was asked for, for every layer and set
static void test_layout(const std::string & path, const std::string & stem, checker & check) {
    slot_layers layers;
    llama_model_ptr model = load_model(path, nullptr, &layers);
    const assignment a = bank(layers);
    for (const int32_t n_ranges : { 2, 3 }) {
        if (min_slots(a) < n_ranges) {
            printf("  %s: %d slots, too few for %d ranges\n", stem.c_str(), min_slots(a), n_ranges);
            continue;
        }
        slot_spec spec;
        spec.sets   = { &a, &a };
        spec.splits = { splits_of(min_slots(a), n_ranges), splits_of(min_slots(a) - 1, n_ranges) };
        llama_context_ptr lctx = new_context(model.get(), spec);
        bool layout_ok = true;
        for (int32_t set = 0; set < 2; ++set) {
            for (const auto & [il, _] : layers) {
                llama_moe_slot_layout layout;
                layout_ok = layout_ok && llama_context_moe_slot_layout(lctx.get(), il, set, &layout) &&
                    layout.n_range_splits == (int32_t) spec.splits[set].size() &&
                    std::equal(spec.splits[set].begin(), spec.splits[set].end(), layout.range_splits);
            }
        }
        check(layout_ok, stem + ", " + std::to_string(n_ranges) + " ranges: the layout of each set reports its range splits");
    }
}

// bad splits and slot sets with different numbers of ranges are refused
static void test_refusals(const std::string & path, const std::string & stem, checker & check) {
    slot_layers layers;
    llama_model_ptr model = load_model(path, nullptr, &layers);
    const assignment a = bank(layers);
    const int32_t n = min_slots(a);

    auto refused = [&](const slot_spec & spec, const char * needle) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx               = 0;
        cparams.moe_slots           = get_slots;
        cparams.moe_slots_user_data = (void *) &spec;
        cparams.moe_slot_sets       = (uint32_t) spec.sets.size();
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        return !lctx && g_log.find(needle) != std::string::npos;
    };

    const char * bad_split = "is not after the previous one and before slot";
    for (const auto & [splits, what] : std::vector<std::pair<std::vector<int32_t>, std::string>>{
            { { 0 },        "a split at slot 0" },
            { { n },        "a split at n_slots" },
            { { 2, 2 },     "a repeated split" },
            { { 3, 1 },     "decreasing splits" },
        }) {
        slot_spec spec;
        spec.sets   = { &a };
        spec.splits = { splits };
        check(refused(spec, bad_split), stem + ": " + what + " is refused");
    }
    {
        slot_spec spec;
        spec.sets   = { &a };
        spec.splits = { std::vector<int32_t>(LLAMA_MOE_SLOT_MAX_RANGES, 1) };
        check(refused(spec, "invalid number of MoE slot range splits"), stem + ": " + std::to_string(LLAMA_MOE_SLOT_MAX_RANGES) + " splits are refused");
    }
    {
        slot_spec spec;
        spec.sets   = { &a, &a };
        spec.splits = { { 1 }, {} };
        check(refused(spec, "every set of a layer must have the same number of ranges"),
            stem + ": slot sets of a layer with 2 and 1 ranges are refused");
    }
}

static void test_model(const std::string & path, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string();
    printf("%s\n", stem.c_str());

    slot_layers layers;
    load_model(path, nullptr, &layers);
    check(!layers.empty(), stem + ": the loader asks about the slots of " + std::to_string(layers.size()) + " MoE layers");
    if (layers.empty()) {
        return;
    }

    test_layout(path, stem, check);
    test_refusals(path, stem, check);
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char * text, void *) {
        g_log += text;
    }, nullptr);

    std::string dir_models;
    if (argc == 3 && strcmp(argv[1], "--models") == 0) {
        dir_models = argv[2];
    }
    if (dir_models.empty()) {
        fprintf(stderr, "usage: %s --models <dir written by test-llama-archs -o>\n", argv[0]);
        return 1;
    }

    const char * models[] = { "qwen3moe-moe.gguf", "qwen35moe-moe.gguf", "gemma4-moe.gguf", "qwen4exp-moe.gguf" };

    checker check;
    for (const char * name : models) {
        const std::string path = dir_models + "/" + name;
        try {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("missing " + path);
            }
            test_model(path, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", name, e.what());
            check(false, std::string(name) + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
