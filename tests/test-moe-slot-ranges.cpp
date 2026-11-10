// checks that a MoE slot set split into slot ranges (llama_moe_slot_pool_params.n_range_splits) computes the logits of
//     the same slots unsplit, within an NMSE of 1e-6 (printed), on the CPU, on the CPU with the expert matmuls of a
//     wide batch offloaded to a GPU, and with the slots in a Metal device buffer:
//   - bank:    one slot per expert, in a shuffled order, in 2 and 3 ranges
//   - partial: the experts the router selects for a short prompt and a garbage spare, shuffled, in 2 and 3 ranges
// and that every op of the graph that reads a member tensor reads a view of it inside one range, each expert weight
//     read by one mul_mat_id per range named with the range; that the context reports its ranges, refuses bad splits
//     and slot sets of a layer with different numbers of ranges, and that two sets split differently compute alike
// on a Metal device, also with the slot buffer split at its ranges (ggml_backend_metal_buffer_from_host_ptr_split,
//     one sub-range for the slot map and one per slot range):
//   - ranged slots on the split buffer compute the logits of unsplit slots in a device buffer, also with each
//     sub-range's residency requested on its own and after a range is unwrapped and rewrapped between graphs
//   - unranged slots on the split buffer abort at encode, since a member tensor crosses the split points
//   - with the sub-range of one slot range unwrapped, a graph aborts with the message that names the range's view and
//     sub-range (every graph reads every range, whichever experts it selects)
//
// the models are the MoE models written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

// the test implements a buffer type
#include "../ggml/src/ggml-backend-impl.h"

#ifdef TEST_MOE_SLOT_RANGES_METAL
#include "ggml-metal.h"

#include <mach-o/dyld.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
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
    ggml_backend_buffer_type_t buft = nullptr;
    bool op_offload = false; // see llama_context_params.op_offload
};

static llama_moe_slot_pool_params get_slots(int32_t il, int32_t /*n_expert*/, int32_t set, void * ud) {
    const auto * spec = (const slot_spec *) ud;
    llama_moe_slot_pool_params res = {};
    res.n_slots = (int32_t) spec->sets.at(set)->at(il).size();
    res.buft    = spec->buft;
    const auto & splits = spec->splits.at(set);
    res.n_range_splits = (int32_t) splits.size();
    for (size_t i = 0; i < splits.size() && i < LLAMA_MOE_SLOT_MAX_RANGES - 1; ++i) {
        res.range_splits[i] = splits[i];
    }
    return res;
}

// the graphs a context prepares, by node
struct graph_log {
    std::vector<ggml_cgraph *> graphs;
    std::function<void(ggml_cgraph *)> inspect;
};

static bool log_graph(ggml_backend_sched_t, ggml_cgraph * gf, void * ud) {
    auto * log = (graph_log *) ud;
    if (log->inspect) {
        log->inspect(gf);
    }
    return true;
}

static llama_context_ptr new_context(llama_model * model, const slot_spec & spec, graph_log * log = nullptr) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 0;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = 64;
    cparams.moe_slots           = get_slots;
    cparams.moe_slots_user_data = (void *) &spec;
    cparams.moe_slot_sets       = (uint32_t) spec.sets.size();
    cparams.op_offload          = spec.op_offload;
    if (log) {
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

static std::vector<llama_token> get_tokens(uint32_t n_tokens, uint32_t n_vocab, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// the logits of `n_batch` tokens in one batch, then of the other tokens one at a time, which take the vector kernels
static std::vector<float> get_logits(llama_context * lctx, size_t n_tokens, size_t n_batch, int32_t set = -1) {
    llama_memory_clear(llama_get_memory(lctx), true);
    if (set >= 0) {
        llama_set_moe_slot_set(lctx, set);
    }

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));
    const std::vector<llama_token> tokens = get_tokens(n_tokens, n_vocab, 42);

    std::vector<float> ret;
    common_batch batch(lctx);
    for (size_t i = 0; i < n_batch; i++) {
        batch.add(tokens[i], i, 0, true);
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
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, one.get())) {
            throw std::runtime_error("failed to decode");
        }
        const float * l = llama_get_logits_ith(lctx, 0);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    return ret;
}

// a prompt in a batch of 12 and 4 single tokens, one in a batch of 32 for the matrix kernels, and a short one
struct prompt {
    size_t n_tokens;
    size_t n_batch;
};
static constexpr prompt PROMPT_LONG  = { 16, 12 };
static constexpr prompt PROMPT_WIDE  = { 34, 32 };
static constexpr prompt PROMPT_SHORT = {  3,  2 };

static std::vector<float> logits_of(llama_context * lctx, const std::vector<prompt> & prompts, int32_t set = -1) {
    std::vector<float> res;
    for (const prompt & p : prompts) {
        const std::vector<float> l = get_logits(lctx, p.n_tokens, p.n_batch, set);
        res.insert(res.end(), l.begin(), l.end());
    }
    return res;
}

// sum((a - ref)^2) / sum(ref^2); infinite on a size mismatch or a non-finite value
static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    if (a.size() != ref.size() || ref.empty()) {
        return INFINITY;
    }
    double err = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) {
            return INFINITY;
        }
        const double d = (double) a[i] - (double) ref[i];
        err += d*d;
        sum += (double) ref[i]*ref[i];
    }
    return err/sum;
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// near-neutral: far below the change of a wrong or missing expert
static constexpr double NMSE_MAX = 1e-6;

static std::string fmt_nmse(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.3g", v);
    return buf;
}

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

// play the caller: fill each slot of `set` from the runs of its expert, garbage for -1, and write the slot maps; the
// experts no slot holds map to the garbage slot
static void fill_slots(llama_context * lctx, const assignment & assign, int32_t set = 0) {
    const llama_model * model = llama_get_model(lctx);
    for (const auto & [il, slots] : assign) {
        llama_moe_slot_layout layout;
        if (!llama_context_moe_slot_layout(lctx, il, set, &layout)) {
            throw std::runtime_error("layer " + std::to_string(il) + " has no slots in set " + std::to_string(set));
        }
        std::vector<uint8_t> region((size_t) layout.n_slots*layout.slot_size, 0x7f);
        std::vector<int32_t> slot_map(layout.n_expert, -1);
        for (size_t p = 0; p < slots.size(); ++p) {
            const int32_t e = slots[p];
            if (e < 0) {
                continue;
            }
            slot_map[e] = (int32_t) p;
            llama_moe_expert_run runs[8];
            const int32_t n_runs = llama_model_moe_expert_runs(model, il, e, runs, 8);
            if (n_runs < 1 || n_runs > 8) {
                throw std::runtime_error("unexpected expert runs");
            }
            for (int32_t r = 0; r < n_runs; ++r) {
                const std::vector<uint8_t> data = read_file(runs[r].path, runs[r].offset, runs[r].size);
                memcpy(region.data() + p*layout.slot_size + runs[r].slot_offset, data.data(), data.size());
            }
        }
        const auto garbage = std::find(slots.begin(), slots.end(), -1);
        for (int32_t & s : slot_map) {
            if (s < 0) {
                s = garbage == slots.end() ? 0 : (int32_t) (garbage - slots.begin());
            }
        }
        ggml_backend_tensor_set(layout.region, region.data(), layout.map_size, region.size());
        ggml_backend_tensor_set(layout.slot_map, slot_map.data(), 0, slot_map.size()*sizeof(int32_t));
    }
}

// the experts that the router selects in each layer
using selection = std::map<int32_t, std::set<int32_t>>;

static bool record_selection(ggml_tensor * t, bool ask, void * ud) {
    const bool is_topk = strncmp(ggml_get_name(t), "ffn_moe_topk-", 13) == 0;
    if (ask || !is_topk) {
        return ask ? is_topk : true;
    }
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    auto & experts = (*(selection *) ud)[atoi(ggml_get_name(t) + 13)];
    for (int64_t r = 0; r < t->ne[1]; ++r) {
        const int32_t * row = (const int32_t *) (bytes.data() + r*t->nb[1]);
        experts.insert(row, row + t->ne[0]);
    }
    return true;
}

// the experts the stock model selects for the short prompt, on the CPU
static selection select_short(const std::string & path) {
    llama_model_ptr model = load_model(path, nullptr, nullptr);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 0;
    cparams.n_ubatch          = 64;
    selection sel;
    cparams.cb_eval           = record_selection;
    cparams.cb_eval_user_data = &sel;
    llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context");
    }
    get_logits(lctx.get(), PROMPT_SHORT.n_tokens, PROMPT_SHORT.n_batch);
    return sel;
}

// a bank: one slot per expert, shuffled
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

// a partial pool: the selected experts and a garbage spare, shuffled
static assignment partial(const slot_layers & layers, const selection & sel) {
    assignment res;
    std::mt19937 gen(11);
    for (const auto & [il, _] : layers) {
        const auto it = sel.find(il);
        if (it == sel.end()) {
            throw std::runtime_error("no selection recorded for layer " + std::to_string(il));
        }
        std::vector<int32_t> & slots = res[il];
        slots.assign(it->second.begin(), it->second.end());
        slots.push_back(-1);
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

// the member tensors of a context's slots and the byte ranges of their slot ranges in the region
struct ranged_members {
    struct region {
        const char * base;
        size_t map_size;
        std::vector<std::pair<size_t, size_t>> ranges; // byte ranges [begin, end) from base
    };
    std::map<const ggml_tensor *, const region *> by_member;
    std::vector<std::unique_ptr<region>> regions;
};

static ranged_members members_of(llama_context * lctx, const slot_layers & layers, int32_t n_sets) {
    ranged_members res;
    for (int32_t set = 0; set < n_sets; ++set) {
        for (const auto & [il, _] : layers) {
            llama_moe_slot_layout layout;
            if (!llama_context_moe_slot_layout(lctx, il, set, &layout)) {
                continue;
            }
            auto r = std::make_unique<ranged_members::region>();
            r->base = (const char *) layout.region->data;
            r->map_size = layout.map_size;
            std::vector<int32_t> bounds = { 0 };
            for (int32_t i = 0; i < layout.n_range_splits; ++i) {
                bounds.push_back(layout.range_splits[i]);
            }
            bounds.push_back(layout.n_slots);
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                r->ranges.emplace_back(layout.map_size + bounds[i]*layout.slot_size, layout.map_size + bounds[i + 1]*layout.slot_size);
            }
            for (int32_t k = 0; k < layout.n_members; ++k) {
                res.by_member[layout.members[k].tensor] = r.get();
            }
            res.regions.push_back(std::move(r));
        }
    }
    return res;
}

// what a graph's nodes read of the member tensors
struct graph_reads {
    int n_graphs        = 0;
    int n_direct        = 0; // reads of a whole member tensor
    int n_outside       = 0; // reads of a view that is not inside one range
    int n_mm_ranged     = 0; // mul_mat_id nodes over a view inside one range
    int n_mm_misnamed   = 0; // of those, not named <name>_r<range>-<il>
    int n_mm_named      = 0; // mul_mat_id nodes named <name>_r<range>-<il>, whose weight may be a copy on another backend
    int n_other_readers = 0; // other ops reading a member view
};

static void inspect_reads(ggml_cgraph * gf, const ranged_members & m, graph_reads & out) {
    static const std::regex re_name(R"(^ffn_moe_(gate_up|up|gate|down)_r\d+-\d+$)");
    out.n_graphs++;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        // a view node reads no data; the ops that read it are checked
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT_ID && std::regex_match(node->name, re_name)) {
            out.n_mm_named++;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src == nullptr) {
                continue;
            }
            if (m.by_member.count(src)) {
                out.n_direct++;
                continue;
            }
            const auto it = src->view_src ? m.by_member.find(src->view_src) : m.by_member.end();
            if (it == m.by_member.end()) {
                continue;
            }
            const size_t begin = (const char *) src->data - it->second->base;
            const size_t end   = begin + ggml_nbytes(src);
            const bool inside = std::any_of(it->second->ranges.begin(), it->second->ranges.end(),
                [&](const std::pair<size_t, size_t> & r) { return r.first <= begin && end <= r.second; });
            if (!inside) {
                out.n_outside++;
            } else if (node->op == GGML_OP_MUL_MAT_ID && s == 0) {
                out.n_mm_ranged++;
                out.n_mm_misnamed += std::regex_match(node->name, re_name) ? 0 : 1;
            } else {
                out.n_other_readers++;
            }
        }
    }
}

// the number of members of the layers' slots
static int n_members_total(llama_context * lctx, const slot_layers & layers) {
    int res = 0;
    for (const auto & [il, _] : layers) {
        llama_moe_slot_layout layout;
        if (llama_context_moe_slot_layout(lctx, il, 0, &layout)) {
            res += layout.n_members;
        }
    }
    return res;
}

// ranged against unranged slots holding the same experts, on the CPU (or `dev`)
static void test_ranges(const std::string & path, const std::string & stem, const char * kind, const assignment & assign,
        const std::vector<prompt> & prompts, ggml_backend_dev_t dev, bool op_offload, checker & check) {
    slot_layers layers;
    llama_model_ptr model = load_model(path, dev, &layers);

    slot_spec spec_ref;
    spec_ref.sets       = { &assign };
    spec_ref.splits     = { {} };
    spec_ref.op_offload = op_offload;
    llama_context_ptr ref_ctx = new_context(model.get(), spec_ref);
    fill_slots(ref_ctx.get(), assign);
    const std::vector<float> ref = logits_of(ref_ctx.get(), prompts);
    ref_ctx.reset();
    const std::string where = dev ? ggml_backend_dev_name(dev) : op_offload ? "CPU, op offload" : "CPU";
    for (const int32_t n_ranges : { 2, 3 }) {
        if (min_slots(assign) < n_ranges) {
            printf("  %s, %s: %d slots, too few for %d ranges\n", stem.c_str(), kind, min_slots(assign), n_ranges);
            continue;
        }
        const std::string label = stem + ", " + kind + " (" + std::to_string(min_slots(assign)) + "+ slots), " +
            std::to_string(n_ranges) + " ranges, " + where;

        slot_spec spec;
        spec.sets       = { &assign };
        spec.splits     = { splits_of(min_slots(assign), n_ranges) };
        spec.op_offload = op_offload;

        graph_log log;
        ranged_members members;
        graph_reads reads;
        log.inspect = [&](ggml_cgraph * gf) { inspect_reads(gf, members, reads); };
        llama_context_ptr lctx = new_context(model.get(), spec, &log);
        members = members_of(lctx.get(), layers, 1);

        // the context reports its ranges
        bool layout_ok = true;
        for (const auto & [il, _] : layers) {
            llama_moe_slot_layout layout;
            layout_ok = layout_ok && llama_context_moe_slot_layout(lctx.get(), il, 0, &layout) &&
                layout.n_range_splits == (int32_t) spec.splits[0].size() &&
                std::equal(spec.splits[0].begin(), spec.splits[0].end(), layout.range_splits);
        }
        check(layout_ok, label + ": the layout reports the range splits");

        fill_slots(lctx.get(), assign);
        reads = {};
        const std::vector<float> logits = logits_of(lctx.get(), prompts);
        const double e = nmse(logits, ref);
        check(e <= NMSE_MAX, label + ": NMSE vs unranged " + fmt_nmse(e) + (same_bits(logits, ref) ? " (bitwise)" : ""));

        const int n_mm_per_graph = n_members_total(lctx.get(), layers)*n_ranges;
        // with op offload, a wide batch's matmuls read copies of the views on the GPU
        check(reads.n_graphs > 0 && reads.n_direct == 0 && reads.n_outside == 0 && reads.n_other_readers == 0 &&
                reads.n_mm_misnamed == 0 && reads.n_mm_ranged > 0 && reads.n_mm_named == reads.n_graphs*n_mm_per_graph,
            label + ": every read of a member is a mul_mat_id over a view inside one range, " +
            std::to_string(n_mm_per_graph) + " per graph, named _r<range> (" + std::to_string(reads.n_graphs) + " graphs, " +
            std::to_string(reads.n_mm_ranged) + " reading the slots in place)");
    }
}

// bad splits and slot sets with different numbers of ranges are refused; two ranged sets compute alike
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

    // two sets with 2 ranges each, split differently, compute alike
    {
        slot_spec spec_ref;
        spec_ref.sets   = { &a };
        spec_ref.splits = { {} };
        llama_context_ptr ref_ctx = new_context(model.get(), spec_ref);
        fill_slots(ref_ctx.get(), a);
        const std::vector<float> ref = logits_of(ref_ctx.get(), { PROMPT_LONG });

        slot_spec spec;
        spec.sets   = { &a, &a };
        spec.splits = { { 1 }, { n - 1 } };
        llama_context_ptr lctx = new_context(model.get(), spec);
        fill_slots(lctx.get(), a, 0);
        fill_slots(lctx.get(), a, 1);
        const double e0 = nmse(logits_of(lctx.get(), { PROMPT_LONG }, 0), ref);
        const double e1 = nmse(logits_of(lctx.get(), { PROMPT_LONG }, 1), ref);
        check(e0 <= NMSE_MAX && e1 <= NMSE_MAX, stem + ": two slot sets split at slot 1 and at slot " + std::to_string(n - 1) +
            ", NMSE vs unranged " + fmt_nmse(e0) + ", " + fmt_nmse(e1));
    }
}

#ifdef TEST_MOE_SLOT_RANGES_METAL

//
// slots in a Metal buffer split at the slot ranges
//

// a caller's buffer type that allocates page-aligned host memory and wraps it with
// ggml_backend_metal_buffer_from_host_ptr_split at the offsets given for each allocation, in allocation order
struct split_buft {
    ggml_backend_buffer_type wrapper;
    ggml_backend_dev_t dev;
    std::vector<std::vector<size_t>> offsets; // per allocation
    size_t next = 0;
    std::vector<void *> mems;
    std::vector<ggml_backend_buffer_t> bufs;

    explicit split_buft(ggml_backend_dev_t dev) : dev(dev) {
        wrapper.iface = {
            /*.get_name         =*/ [](ggml_backend_buffer_type_t) { return "split"; },
            /*.alloc_buffer     =*/ [](ggml_backend_buffer_type_t buft, size_t size) -> ggml_backend_buffer_t {
                auto * w = (split_buft *) buft->context;
                if (w->next >= w->offsets.size()) {
                    return nullptr;
                }
                const size_t page = (size_t) getpagesize();
                void * mem = nullptr;
                if (posix_memalign(&mem, page, GGML_PAD(size, page)) != 0) {
                    return nullptr;
                }
                memset(mem, 0, GGML_PAD(size, page));
                w->mems.push_back(mem);
                const std::vector<size_t> & offs = w->offsets[w->next++];
                ggml_backend_buffer_t buf = ggml_backend_metal_buffer_from_host_ptr_split(w->dev, mem, size, offs.data(), (int) offs.size());
                w->bufs.push_back(buf);
                return buf;
            },
            /*.alloc_buffer_n   =*/ nullptr,
            /*.get_alignment    =*/ [](ggml_backend_buffer_type_t) { return (size_t) 32; },
            /*.get_max_size     =*/ [](ggml_backend_buffer_type_t) { return SIZE_MAX; },
            /*.get_alloc_size   =*/ nullptr,
            /*.get_alloc_size_n =*/ nullptr,
            /*.is_host          =*/ [](ggml_backend_buffer_type_t) { return true; },
        };
        wrapper.device  = dev;
        wrapper.context = this;
    }

    ~split_buft() {
        for (void * m : mems) {
            free(m);
        }
    }

    split_buft(const split_buft &) = delete;
    split_buft & operator=(const split_buft &) = delete;

    ggml_backend_buffer_type_t get() {
        return &wrapper;
    }
};

// the split offsets of each layer's slot buffer, in layer order: the slot map, then one sub-range per slot range
// (sub-range r + 1 holds slot range r)
static std::vector<std::vector<size_t>> split_offsets(const llama_model * model, const slot_layers & layers, const std::vector<int32_t> & splits) {
    std::vector<std::vector<size_t>> res;
    for (const auto & [il, _] : layers) {
        size_t  slot_size = 0;
        size_t  map_size  = 0;
        int32_t n_expert  = 0;
        if (!llama_model_moe_slot_geometry(model, il, &slot_size, &map_size, &n_expert)) {
            throw std::runtime_error("no slot geometry");
        }
        std::vector<size_t> offs = { map_size };
        for (int32_t s : splits) {
            offs.push_back(map_size + (size_t) s*slot_size);
        }
        res.push_back(offs);
    }
    return res;
}

static ggml_backend_dev_t metal_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && strncmp(ggml_backend_dev_name(dev), "MTL", 3) == 0) {
            return dev;
        }
    }
    return nullptr;
}

// the ranges of the slots in a split buffer: 3, or 2 for a bank of 2 experts
static int32_t n_ranges_of(const assignment & a) {
    return std::min(3, min_slots(a));
}

// a context whose slots, a shuffled bank in n_ranges_of ranges, are in split buffers; ranged = false reads them unranged
struct split_context {
    slot_layers layers;
    llama_model_ptr model;
    assignment assign;
    std::unique_ptr<split_buft> buft;
    slot_spec spec;
    llama_context_ptr lctx;

    split_context(const std::string & path, ggml_backend_dev_t dev, bool ranged) {
        model  = load_model(path, dev, &layers);
        assign = bank(layers);
        const std::vector<int32_t> splits = splits_of(min_slots(assign), n_ranges_of(assign));
        buft = std::make_unique<split_buft>(dev);
        buft->offsets = split_offsets(model.get(), layers, splits);
        spec.sets   = { &assign };
        spec.splits = { ranged ? splits : std::vector<int32_t>() };
        spec.buft   = buft->get();
        lctx = new_context(model.get(), spec);
        fill_slots(lctx.get(), assign);
    }

    ~split_context() {
        lctx.reset();
    }
};

// runs this binary with `args` in a child process; true when the child was waited for
static bool run_child(const std::vector<std::string> & args, int * status, std::string * err) {
    char path[4096];
    uint32_t len = sizeof(path);
    if (_NSGetExecutablePath(path, &len) != 0) {
        return false;
    }
    std::vector<std::string> args_s = args;
    std::vector<char *> argv = { path };
    for (std::string & a : args_s) {
        argv.push_back(a.data());
    }
    argv.push_back(nullptr);

    std::vector<char *> envp;
    char no_bt[] = "GGML_NO_BACKTRACE=1";
    envp.push_back(no_bt);
    for (char ** e = environ; *e != nullptr; e++) {
        envp.push_back(*e);
    }
    envp.push_back(nullptr);

    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, fds[0]);
    pid_t pid = 0;
    const bool spawned = posix_spawn(&pid, path, &actions, nullptr, argv.data(), envp.data()) == 0;
    posix_spawn_file_actions_destroy(&actions);
    close(fds[1]);

    char bytes[4096];
    ssize_t n = 0;
    while ((n = read(fds[0], bytes, sizeof(bytes))) > 0) {
        err->append(bytes, (size_t) n);
    }
    close(fds[0]);
    return spawned && waitpid(pid, status, 0) == pid;
}

static bool aborted(int status) {
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

// child modes, which must abort: unranged slots on the split buffer, or ranged slots with the sub-range of slot range
// `range` unwrapped after a graph computed
static int child_unranged(const std::string & path, ggml_backend_dev_t dev) {
    split_context sc(path, dev, false);
    get_logits(sc.lctx.get(), PROMPT_LONG.n_tokens, PROMPT_LONG.n_batch);
    printf("unranged slots on a split buffer computed\n");
    return 0;
}

static int child_unwrapped(const std::string & path, ggml_backend_dev_t dev, int range) {
    split_context sc(path, dev, true);
    get_logits(sc.lctx.get(), PROMPT_LONG.n_tokens, PROMPT_LONG.n_batch);
    fprintf(stderr, "the wrapped ranges compute\n");
    for (ggml_backend_buffer_t buf : sc.buft->bufs) {
        if (!ggml_backend_metal_buffer_end_residency_range(buf, range + 1) || !ggml_backend_metal_buffer_unwrap_range(buf, range + 1)) {
            fprintf(stderr, "sub-range %d was not unwrapped\n", range + 1);
            return 1;
        }
    }
    get_logits(sc.lctx.get(), PROMPT_LONG.n_tokens, PROMPT_LONG.n_batch);
    printf("a graph over an unwrapped range computed\n");
    return 0;
}

static void test_metal_split(const std::string & path, const std::string & stem, ggml_backend_dev_t dev, checker & check) {
    const std::string label = stem + ", split Metal buffer";
    const std::vector<prompt> prompts = { PROMPT_LONG, PROMPT_WIDE };

    // the reference: the same slots unranged in a device buffer
    std::vector<float> ref;
    {
        slot_layers layers;
        llama_model_ptr model = load_model(path, dev, &layers);
        const assignment a = bank(layers);
        slot_spec spec;
        spec.sets   = { &a };
        spec.splits = { {} };
        llama_context_ptr lctx = new_context(model.get(), spec);
        fill_slots(lctx.get(), a);
        ref = logits_of(lctx.get(), prompts);
    }

    {
        split_context sc(path, dev, true);
        bool split_ok = !sc.buft->bufs.empty() && sc.buft->next == sc.layers.size();
        for (ggml_backend_buffer_t buf : sc.buft->bufs) {
            split_ok = split_ok && buf != nullptr;
        }
        const int32_t n_ranges = n_ranges_of(sc.assign);
        check(split_ok, label + ": each slot layer's buffer is split at the slot map and its " + std::to_string(n_ranges) + " ranges (" +
            std::to_string(sc.buft->bufs.size()) + " buffers)");

        const std::vector<float> logits = logits_of(sc.lctx.get(), prompts);
        const double e = nmse(logits, ref);
        check(e <= NMSE_MAX, label + ": " + std::to_string(n_ranges) + " ranges, NMSE vs unranged in a device buffer " + fmt_nmse(e) + (same_bits(logits, ref) ? " (bitwise)" : ""));

        // the caller holds the residency of each sub-range on its own
        bool residency_ok = true;
        for (ggml_backend_buffer_t buf : sc.buft->bufs) {
            for (int i = 0; i <= n_ranges; ++i) {
                residency_ok = residency_ok && ggml_backend_metal_buffer_request_residency_range(buf, i);
            }
        }
        const double e_res = nmse(logits_of(sc.lctx.get(), prompts), ref);
        check(residency_ok && e_res <= NMSE_MAX, label + ": with each sub-range's residency requested on its own, NMSE " + fmt_nmse(e_res));

        // a range unwrapped and rewrapped between graphs computes the same
        bool cycle_ok = true;
        for (ggml_backend_buffer_t buf : sc.buft->bufs) {
            cycle_ok = cycle_ok && ggml_backend_metal_buffer_end_residency_range(buf, 2) && ggml_backend_metal_buffer_unwrap_range(buf, 2) &&
                ggml_backend_metal_buffer_rewrap_range(buf, 2) && ggml_backend_metal_buffer_request_residency_range(buf, 2);
        }
        const std::vector<float> after = logits_of(sc.lctx.get(), prompts);
        check(cycle_ok && same_bits(after, logits), label + ": after slot range 1 is unwrapped and rewrapped, the logits are bitwise unchanged");
    }

    {
        int status = 0;
        std::string err;
        const bool waited = run_child({ "--child-unranged", path }, &status, &err);
        check(waited && aborted(status) && err.find("crosses a split point") != std::string::npos,
            label + ": unranged slots on the split buffer abort at encode, a member crosses a split point");
    }

    slot_layers layers;
    load_model(path, nullptr, &layers);
    for (int range : { 0, n_ranges_of(bank(layers)) - 1 }) {
        int status = 0;
        std::string err;
        const bool waited = run_child({ "--child-unwrapped", path, std::to_string(range) }, &status, &err);
        // the abort names the view of the range
        const std::string sub = ".slots.r" + std::to_string(range) + "' is in sub-range " + std::to_string(range + 1) + " of a split buffer";
        check(err.find("the wrapped ranges compute") != std::string::npos && waited && aborted(status) && err.find(sub) != std::string::npos,
            label + ": with slot range " + std::to_string(range) + " unwrapped, a graph aborts naming its view and sub-range " + std::to_string(range + 1));
    }
}

#endif // TEST_MOE_SLOT_RANGES_METAL

static void test_model(const std::string & path, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string();
    printf("%s\n", stem.c_str());

    slot_layers layers;
    load_model(path, nullptr, &layers);
    check(!layers.empty(), stem + ": the loader asks about the slots of " + std::to_string(layers.size()) + " MoE layers");
    if (layers.empty()) {
        return;
    }

    const selection sel = select_short(path);
    const assignment a_bank    = bank(layers);
    const assignment a_partial = partial(layers, sel);

    // the CPU alone, then with the expert matmuls of the wide batch offloaded to a GPU if there is one
    const bool has_gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
                         ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr;
    for (const bool op_offload : { false, true }) {
        if (op_offload && !has_gpu) {
            continue;
        }
        test_ranges(path, stem, "bank",    a_bank,    { PROMPT_LONG, PROMPT_WIDE }, nullptr, op_offload, check);
        test_ranges(path, stem, "partial", a_partial, { PROMPT_SHORT },             nullptr, op_offload, check);
    }
    test_refusals(path, stem, check);

#ifdef TEST_MOE_SLOT_RANGES_METAL
    ggml_backend_dev_t dev = metal_device();
    if (dev == nullptr) {
        printf("  no Metal device: skipping the Metal checks\n");
        return;
    }
    test_ranges(path, stem, "bank",    a_bank,    { PROMPT_LONG, PROMPT_WIDE }, dev, true, check);
    test_ranges(path, stem, "partial", a_partial, { PROMPT_SHORT },             dev, true, check);
    test_metal_split(path, stem, dev, check);
#endif
}

int main(int argc, char ** argv) {
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char * text, void *) {
        g_log += text;
    }, nullptr);

#ifdef TEST_MOE_SLOT_RANGES_METAL
    if (argc >= 3 && strcmp(argv[1], "--child-unranged") == 0) {
        return child_unranged(argv[2], metal_device());
    }
    if (argc >= 4 && strcmp(argv[1], "--child-unwrapped") == 0) {
        return child_unwrapped(argv[2], metal_device(), atoi(argv[3]));
    }
#endif

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
