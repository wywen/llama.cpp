// Checks that a plan context (llama_init_plan_from_model) reserves the graphs a real context reserves, and allocates
// nothing.
//
// For every generated model of the architectures below, on the CPU, a plan context with a caller allocator that
// measures and a real context of the same parameters make the same calls: creation, the extent and plain reserves of
// llama_graph_reserve and its variants, with outputs and with none, llama_sched_reserve_now once backend samplers
// attach, and a reserve of no output with them; and likewise an MTP context reading each, where the model has a NextN
// layer. A graph rewrite callback runs on both and records each graph before the scheduler splits it. The plan context
// must rewrite, node by node, the graphs the real context rewrites, in the same order: the same ops, shapes, strides,
// names and parameters, reading the same nodes, model weights, memory tensors and inputs. Each of its reserves must
// reach measure with a graph rewritten for a reserve that reads no data-owning tensor but the weights; it must never
// call alloc and must hold no compute buffer.

#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

const char * const plan_archs[] = { "qwen35", "qwen35moe", "qwen4exp", "gemma4" };

// the architecture a generated model file is of: <arch>-<config>.gguf
std::optional<std::string> file_arch(const std::string & file) {
    for (const char * arch : plan_archs) {
        const std::string prefix = std::string(arch) + "-";
        if (file.rfind(prefix, 0) == 0) {
            return std::string(arch);
        }
    }
    return std::nullopt;
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string shape_of(const ggml_tensor * t) {
    std::string out = std::string(ggml_type_name(t->type)) + "[";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        out += std::to_string(t->ne[d]) + (d + 1 < GGML_MAX_DIMS ? "," : "]");
    }
    out += "{";
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        out += std::to_string(t->nb[d]) + (d + 1 < GGML_MAX_DIMS ? "," : "}");
    }
    return out;
}

bool is_weight(const ggml_tensor * t, const llama_model * model, const llama_model * other) {
    const char * name = ggml_get_name(t);
    return (model != nullptr && model->get_tensor(name) == t) || (other != nullptr && other->get_tensor(name) == t);
}

// what a tensor read by a graph is: a node of the graph by index, a model weight by identity, or another leaf (a
// memory tensor, an input) by name and shape; a view also by what it views
class graph_identity {
public:
    graph_identity(ggml_cgraph * gf, const llama_model * model, const llama_model * other) : model(model), other(other) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            node_index[ggml_graph_node(gf, i)] = i;
        }
    }

    std::string of(const ggml_tensor * t, int depth = 0) const {
        if (t == nullptr) {
            return "-";
        }
        if (depth > 16) {
            return "<deep>";
        }
        const auto it = node_index.find(t);
        if (it != node_index.end()) {
            return "node " + std::to_string(it->second);
        }
        std::string out = "leaf '" + std::string(ggml_get_name(t)) + "' " + ggml_op_name(t->op) + " " + shape_of(t);
        if (is_weight(t, model, other)) {
            char ptr[32];
            snprintf(ptr, sizeof(ptr), " weight %p", (const void *) t);
            out += ptr;
        }
        if (t->view_src != nullptr) {
            out += " view of (" + of(t->view_src, depth + 1) + ") at " + std::to_string(t->view_offs);
        }
        return out;
    }

private:
    const llama_model * model;
    const llama_model * other;
    std::unordered_map<const ggml_tensor *, int> node_index;
};

std::string describe_node(const ggml_tensor * node, const graph_identity & ids) {
    std::string out = "'" + std::string(ggml_get_name(node)) + "' " + ggml_op_name(node->op) + " " + shape_of(node);
    char params[GGML_MAX_OP_PARAMS * 2 + 1] = {};
    for (size_t i = 0; i < GGML_MAX_OP_PARAMS; ++i) {
        snprintf(params + 2 * i, 3, "%02x", ((const uint8_t *) node->op_params)[i]);
    }
    out += " params " + std::string(params) + " flags " + std::to_string(node->flags);
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (node->src[s] != nullptr) {
            out += "\n    src" + std::to_string(s) + ": " + ids.of(node->src[s]);
        }
    }
    if (node->view_src != nullptr) {
        out += "\n    views: " + ids.of(node->view_src) + " at " + std::to_string(node->view_offs);
    }
    return out;
}

// a graph node by node, as described above
std::vector<std::string> describe_graph(ggml_cgraph * gf, const llama_model * model, const llama_model * other) {
    const graph_identity ids(gf, model, other);
    std::vector<std::string> out;
    out.reserve(ggml_graph_n_nodes(gf));
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        out.push_back(describe_node(ggml_graph_node(gf, i), ids));
    }
    return out;
}

// a rewrite of the kind a caller installs: a node added after the first norm, whose readers it rewires. A real
// context's rewrite also records the graph as rewritten, before the scheduler splits it: a split rewires the readers
// of an input from another backend to a copy, and a backend may reorder the nodes of its splits
struct rewrite_state {
    const llama_model * model = nullptr;
    const llama_model * other = nullptr;
    bool record = false;
    std::vector<std::string> rewritten;
    ggml_backend_sched_rewrite_purpose recorded = GGML_BACKEND_SCHED_REWRITE_ALLOC; // of the call that recorded
    std::vector<std::vector<std::string>> history; // every graph recorded, in order
    std::vector<ggml_context_ptr> ctxs;
};

constexpr const char * rewrite_node_name = "plan_context_test_rewrite";

bool rewrite_graph(ggml_backend_sched_t /*sched*/, ggml_cgraph * gf, ggml_backend_sched_rewrite_purpose purpose, void * user_data) {
    auto * state = static_cast<rewrite_state *>(user_data);
    const int n_nodes = ggml_graph_n_nodes(gf);
    int first_norm = -1;
    bool done = false;
    for (int i = 0; i < n_nodes && !done; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        done = strcmp(ggml_get_name(node), rewrite_node_name) == 0; // a graph llama reuses keeps its rewrite
        if (first_norm < 0 && (node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_NORM)) {
            first_norm = i;
        }
    }
    if (!done && first_norm >= 0) {
        ggml_init_params params = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        state->ctxs.emplace_back(ctx);

        ggml_tensor * norm   = ggml_graph_node(gf, first_norm);
        ggml_tensor * scaled = ggml_scale(ctx, norm, 1.0f);
        ggml_set_name(scaled, rewrite_node_name);
        for (int i = first_norm + 1; i < n_nodes; ++i) {
            for (ggml_tensor *& src : ggml_graph_node(gf, i)->src) {
                if (src == norm) {
                    src = scaled;
                }
            }
        }
        ggml_graph_add_node(gf, scaled);
        ggml_tensor ** nodes = ggml_graph_nodes(gf);
        std::rotate(nodes + first_norm + 1, nodes + n_nodes, nodes + n_nodes + 1);
    }
    if (state->record) {
        state->rewritten = describe_graph(gf, state->model, state->other);
        state->recorded  = purpose;
        state->history.push_back(state->rewritten);
    }
    return true;
}

// the first difference between a plan context's graph and a real context's, as their rewrites recorded them; empty
// when they are the same
std::string graph_difference(const std::vector<std::string> & plan, const std::vector<std::string> & real) {
    if (plan.size() != real.size()) {
        return "planned " + std::to_string(plan.size()) + " nodes, reserved " + std::to_string(real.size());
    }
    for (size_t i = 0; i < plan.size(); ++i) {
        if (plan[i] != real[i]) {
            return "node " + std::to_string(i) + ":\n  planned  " + plan[i] + "\n  reserved " + real[i];
        }
    }
    return {};
}

// whether every tensor a measured graph reads that is not a model weight owns no data: nothing was allocated
std::string allocated_tensor(ggml_cgraph * gf, const llama_model & model, const llama_model * other) {
    std::vector<const ggml_tensor *> stack;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        stack.push_back(ggml_graph_node(gf, i));
    }
    std::unordered_map<const ggml_tensor *, bool> seen;
    while (!stack.empty()) {
        const ggml_tensor * t = stack.back();
        stack.pop_back();
        if (t == nullptr || seen[t]) {
            continue;
        }
        seen[t] = true;
        const ggml_tensor * root = t;
        while (root->view_src != nullptr) {
            root = root->view_src;
        }
        const char * name = ggml_get_name(root);
        const bool weight = model.get_tensor(name) == root || (other != nullptr && other->get_tensor(name) == root);
        if (!weight && t->data != nullptr) {
            return ggml_get_name(t);
        }
        if (!weight && t->buffer != nullptr && ggml_backend_buffer_get_size(t->buffer) != 0) {
            return std::string(ggml_get_name(t)) + " (in a buffer of " + std::to_string(ggml_backend_buffer_get_size(t->buffer)) + " bytes)";
        }
        for (const ggml_tensor * src : t->src) {
            stack.push_back(src);
        }
        stack.push_back(t->view_src);
    }
    return {};
}

// the compute buffers a context's scheduler holds, which a plan context never allocates
size_t compute_bytes(const llama_context & ctx) {
    ggml_backend_sched_t sched = ctx.get_sched();
    size_t total = 0;
    for (int b = 0; b < ggml_backend_sched_get_n_backends(sched); ++b) {
        total += ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, b));
    }
    return total;
}

// a model on the CPU alone: the test caller allocator places nothing, so it names no copies between devices
llama_model_ptr load_model(const std::string & path, bool load_mtp) {
    llama_model_params params = llama_model_default_params();
    static ggml_backend_dev_t no_devices[] = { nullptr };
    params.devices  = no_devices;
    params.load_mtp = load_mtp;
    return llama_model_ptr(llama_model_load_from_file(path.c_str(), params));
}

struct context_case {
    const char * name;
    uint32_t     n_ctx;
    uint32_t     n_seq_max;
    uint32_t     n_ubatch;
    llama_flash_attn_type flash_attn_type;
    bool         drafts; // whether a draft reads the target: speculation decodes one sequence
};

const context_case context_cases[] = {
    { "one-seq-auto-fa", 512, 1, 64, LLAMA_FLASH_ATTN_TYPE_AUTO,     true  },
    { "two-seq-no-fa",   512, 2, 32, LLAMA_FLASH_ATTN_TYPE_DISABLED, false },
};

bool install_rewrite(ggml_backend_sched_t sched, llama_context * /*ctx*/, void * user_data) {
    ggml_backend_sched_set_graph_rewrite_callback(sched, rewrite_graph, user_data);
    return true;
}

llama_context_params make_params(const context_case & c, rewrite_state & rewrite) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx               = c.n_ctx;
    params.n_batch             = c.n_ctx;
    params.n_ubatch            = c.n_ubatch;
    params.n_seq_max           = c.n_seq_max;
    params.flash_attn_type     = c.flash_attn_type;
    params.n_threads           = 2;
    params.n_threads_batch     = 2;
    params.cb_sched_new               = install_rewrite;
    params.cb_sched_new_user_data     = &rewrite;
    params.n_graph_nodes_extra        = 1;
    return params;
}

struct plan_case {
    std::string label;
    const llama_model * target = nullptr;
    const llama_model * draft  = nullptr; // an MTP context reading the target, or none
    // whether the memory simulates a shorter extent faithfully: the indexer cache of qwen4exp keeps its whole size
    // under a shorter attention extent, which its graph refuses
    bool extents = true;
};

struct outcome {
    size_t graphs   = 0;
    size_t failures = 0;
};

void fail(outcome & out, const std::string & what) {
    fprintf(stderr, "FAIL %s\n", what.c_str());
    out.failures++;
}

// the backend sampler chains of one context, one per sequence, freed after the context
struct sampler_chains {
    std::vector<llama_sampler *>          chains;
    std::vector<llama_sampler_seq_config> configs;

    sampler_chains() = default;
    sampler_chains(const sampler_chains &) = delete;
    sampler_chains & operator=(const sampler_chains &) = delete;
    ~sampler_chains() {
        for (llama_sampler * chain : chains) {
            llama_sampler_free(chain);
        }
    }

    void make(uint32_t n_seqs) {
        for (uint32_t seq = 0; seq < n_seqs; ++seq) {
            chains.push_back(common_speculative_draft_backend_chain());
            configs.push_back({ (llama_seq_id) seq, chains.back() });
        }
    }
};

// a plan context's caller allocator: it places nothing, refuses to allocate, and records each graph handed to measure
// as the context's graph rewrite recorded it before the split
struct measuring_caller {
    rewrite_state rewrite;
    std::vector<std::vector<std::string>> measured;
    std::vector<std::string> problems;
    size_t n_allocs   = 0;
    ggml_backend_sched_t sched = nullptr; // the last scheduler the context created
    ggml_backend_sched_caller_alloc calls = {};
};

bool measuring_place(ggml_backend_sched_t /*sched*/, ggml_cgraph * /*graph*/, void * /*user_data*/) {
    return true;
}

bool measuring_alloc(ggml_backend_sched_t /*sched*/, ggml_cgraph * /*graph*/, bool /*reserve*/, void * user_data) {
    static_cast<measuring_caller *>(user_data)->n_allocs++;
    return false;
}

bool measuring_measure(ggml_backend_sched_t sched, ggml_cgraph * graph, void * user_data) {
    auto * c = static_cast<measuring_caller *>(user_data);
    const std::string what = "measured graph " + std::to_string(c->measured.size());
    c->measured.push_back(c->rewrite.rewritten);
    if (c->rewrite.rewritten.empty() || c->rewrite.recorded != GGML_BACKEND_SCHED_REWRITE_RESERVE) {
        c->problems.push_back(what + " was not rewritten for a reserve");
    }
    if (graph != ggml_backend_sched_get_graph(sched)) {
        c->problems.push_back(what + " is not the graph the backends compute");
    }
    if (const std::string tensor = allocated_tensor(graph, *c->rewrite.model, c->rewrite.other); !tensor.empty()) {
        c->problems.push_back(what + " reads " + tensor + ", which owns data");
    }
    c->rewrite.rewritten.clear();
    return true;
}

// installs the caller's graph rewrite and caller allocator
bool install_measuring(ggml_backend_sched_t sched, llama_context * /*ctx*/, void * user_data) {
    auto * caller = static_cast<measuring_caller *>(user_data);
    caller->sched = sched;
    ggml_backend_sched_set_graph_rewrite_callback(sched, rewrite_graph, &caller->rewrite);
    return ggml_backend_sched_set_caller_alloc(sched, &caller->calls);
}

llama_context_params make_measuring_params(const context_case & c, measuring_caller & caller) {
    caller.calls = { measuring_place, measuring_alloc, &caller, measuring_measure };
    llama_context_params params = make_params(c, caller.rewrite);
    params.cb_sched_new           = install_measuring;
    params.cb_sched_new_user_data = &caller;
    caller.rewrite.record = true;
    return params;
}

// the calls a context makes after its creation; the result names the first that failed, empty when none did
std::string reserve_after_creation(llama_context * ctx, const context_case & cc, bool extents, sampler_chains & chains) {
    const uint32_t n_seqs = cc.n_seq_max;
    if (extents) {
        if (llama_graph_reserve_extent(ctx, n_seqs, n_seqs, n_seqs, 256) == nullptr) {
            return "an extent reserve failed";
        }
        if (llama_graph_reserve_widest_extent(ctx, 256) == nullptr) {
            return "a widest extent reserve failed";
        }
        if (llama_graph_reserve_extent(ctx, n_seqs, n_seqs, 0, 256) == nullptr) {
            return "an extent reserve of no output failed";
        }
    }
    if (llama_graph_reserve(ctx, cc.n_ubatch, n_seqs, n_seqs) == nullptr) {
        return "a reserve failed";
    }
    if (llama_graph_reserve(ctx, cc.n_ubatch, n_seqs, 0) == nullptr) {
        return "a reserve of no output failed";
    }
    chains.make(n_seqs);
    for (const llama_sampler_seq_config & config : chains.configs) {
        if (!llama_set_sampler(ctx, config.seq_id, config.sampler)) {
            return "the context does not sample on the backend";
        }
    }
    if (!llama_sched_reserve_pending(ctx) || !llama_sched_reserve_now(ctx) || llama_sched_reserve_pending(ctx)) {
        return "the pending scheduler reserve did not run";
    }
    if (llama_graph_reserve(ctx, cc.n_ubatch, n_seqs, 0) == nullptr) {
        return "a reserve of no output with backend samplers failed";
    }
    return {};
}

// the plan context's checks, and its graphs against the real context's
void check_measured(const std::string & what, const measuring_caller & caller, size_t n_measures, const llama_context & ctx,
        const rewrite_state & real, outcome & out) {
    for (const std::string & problem : caller.problems) {
        fail(out, what + ": " + problem);
    }
    if (caller.n_allocs != 0) {
        fail(out, what + ": the plan context allocated " + std::to_string(caller.n_allocs) + " graphs");
    }
    if (caller.sched != ctx.get_sched()) {
        fail(out, what + ": the scheduler callback did not see the context's scheduler");
    }
    if (const size_t bytes = compute_bytes(ctx); bytes != 0) {
        fail(out, what + ": the plan context holds " + std::to_string(bytes) + " compute bytes");
    }
    if (caller.measured.size() != n_measures) {
        fail(out, what + ": measured " + std::to_string(caller.measured.size()) + " graphs, reserved " + std::to_string(n_measures));
    }
    const std::vector<std::vector<std::string>> & plan = caller.rewrite.history;
    if (plan.size() != real.history.size()) {
        fail(out, what + ": the plan context rewrote " + std::to_string(plan.size()) + " graphs, the real context " + std::to_string(real.history.size()));
        return;
    }
    for (size_t g = 0; g < plan.size(); ++g) {
        out.graphs++;
        if (const std::string diff = graph_difference(plan[g], real.history[g]); !diff.empty()) {
            fail(out, what + " graph " + std::to_string(g) + ": " + diff);
        }
    }
}

// a context's creation reserves three graphs (prompt, token generation, widest); after it, the extent reserves, a plain
// reserve and one of no output, the three again once the samplers attach, and one of no output with them
void check_plan_context(const plan_case & pc, outcome & out) {
    for (const context_case & cc : context_cases) {
        if (pc.draft != nullptr && !cc.drafts) {
            continue;
        }
        const std::string label = pc.label + " [" + cc.name + "]";
        const size_t n_target_measures = 3 + (pc.extents ? 3 : 0) + 2 + 3 + 1;
        // a context's samplers outlive it
        sampler_chains real_chains;
        sampler_chains plan_chains;

        rewrite_state real_target_rewrite;
        real_target_rewrite.model  = pc.target;
        real_target_rewrite.record = true;
        llama_context_ptr real_target(llama_init_from_model(const_cast<llama_model *>(pc.target), make_params(cc, real_target_rewrite)));
        if (!real_target) {
            fail(out, label + ": the real context failed");
            continue;
        }
        if (const std::string err = reserve_after_creation(real_target.get(), cc, pc.extents, real_chains); !err.empty()) {
            fail(out, label + ": real context: " + err);
            continue;
        }

        measuring_caller target_caller;
        target_caller.rewrite.model = pc.target;
        llama_context_ptr target(llama_init_plan_from_model(const_cast<llama_model *>(pc.target), make_measuring_params(cc, target_caller)));
        if (!target) {
            fail(out, label + ": the plan context failed");
            continue;
        }
        if (const std::string err = reserve_after_creation(target.get(), cc, pc.extents, plan_chains); !err.empty()) {
            fail(out, label + ": plan context: " + err);
        }
        check_measured(label + " target", target_caller, n_target_measures, *target, real_target_rewrite, out);

        if (pc.draft != nullptr) {
            rewrite_state real_draft_rewrite;
            real_draft_rewrite.model  = pc.draft;
            real_draft_rewrite.other  = pc.target;
            real_draft_rewrite.record = true;
            llama_context_params real_params = make_params(cc, real_draft_rewrite);
            real_params.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
            real_params.ctx_other = real_target.get();
            llama_context_ptr real_draft(llama_init_from_model(const_cast<llama_model *>(pc.draft), real_params));

            measuring_caller draft_caller;
            draft_caller.rewrite.model = pc.draft;
            draft_caller.rewrite.other = pc.target;
            llama_context_params plan_params = make_measuring_params(cc, draft_caller);
            plan_params.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
            plan_params.ctx_other = target.get();
            llama_context_ptr draft(llama_init_plan_from_model(const_cast<llama_model *>(pc.draft), plan_params));
            if (!real_draft || !draft) {
                fail(out, label + ": an MTP context failed");
            } else {
                check_measured(label + " MTP", draft_caller, 3, *draft, real_draft_rewrite, out);
            }
        }

        // the draft contexts went before the targets they read
        target.reset();
        real_target.reset();
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::string models_dir;
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else {
            fprintf(stderr, "usage: %s --models DIR [-v]\n", argv[0]);
            return 1;
        }
    }
    if (models_dir.empty() || !std::filesystem::is_directory(models_dir)) {
        fprintf(stderr, "%s: --models must name a directory of generated models\n", argv[0]);
        return 1;
    }

    if (!verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
    llama_backend_init();

    std::vector<std::string> files;
    for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
        const std::string file = entry.path().filename().string();
        if (entry.is_regular_file() && ends_with(file, ".gguf") && file_arch(file).has_value()) {
            files.push_back(file);
        }
    }
    std::sort(files.begin(), files.end());

    outcome out;
    std::map<std::string, llama_model_ptr> models;
    const auto model_of = [&](const std::string & file, bool load_mtp) -> const llama_model * {
        auto it = models.find(file);
        if (it == models.end()) {
            llama_model_ptr model = load_model((std::filesystem::path(models_dir) / file).string(), load_mtp);
            if (!model) {
                fail(out, file + ": could not load");
                return nullptr;
            }
            it = models.emplace(file, std::move(model)).first;
        }
        return it->second.get();
    };

    std::vector<plan_case> cases;
    std::map<std::string, bool> archs_seen;
    for (const std::string & file : files) {
        archs_seen[*file_arch(file)] = true;
        const bool extents = *file_arch(file) != "qwen4exp";
        if (ends_with(file, "-nextn-only.gguf")) {
            // a draft in a file of its own, reading the target of the matching NextN model
            const std::string target_file = file.substr(0, file.size() - strlen("-only.gguf")) + ".gguf";
            const llama_model * target = model_of(target_file, true);
            const llama_model * draft  = model_of(file, true);
            if (target != nullptr && draft != nullptr) {
                cases.push_back({ file + " drafting for " + target_file, target, draft, extents });
            }
            continue;
        }
        const bool nextn = ends_with(file, "-nextn.gguf");
        const llama_model * model = model_of(file, nextn);
        if (model == nullptr) {
            continue;
        }
        cases.push_back({ file, model, nullptr, extents });
        if (nextn) {
            // the NextN layer appended to the target's own file
            cases.push_back({ file + " with its NextN head", model, model, extents });
        }
    }
    for (const char * arch : plan_archs) {
        if (!archs_seen[arch]) {
            fail(out, std::string("no generated model of ") + arch + " in " + models_dir);
        }
    }

    for (const plan_case & pc : cases) {
        check_plan_context(pc, out);
    }

    printf("%s: %zu cases, %zu graphs compared, %zu failures\n", argv[0], cases.size(), out.graphs, out.failures);
    return out.failures == 0 && out.graphs > 0 ? 0 : 1;
}
