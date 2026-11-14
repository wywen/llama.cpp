// Checks llama_graph_plan_build against the graphs real context construction reserves.
//
// For every generated model of the architectures below, on one backend, the plan of a context (and of an MTP
// context reading it, where the model has a NextN layer) must build, node by node, the graph the real context
// reserves for the same shape: the same ops, shapes, strides, names and parameters, reading the same nodes, model
// weights, memory tensors and inputs. A graph rewrite callback runs on both. The plan must allocate nothing. So must
// the plan of a context sampling on the backend, and of the contexts of a speculative session, which output nextn
// embeddings and sample the draft on the backend: each context reserves its graphs before these attach and again
// after, and the plan builds each shape in the state its context reserved it in.

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

// a rewrite of the kind an engine installs: a node added after the first norm, whose readers it rewires. A real
// context's rewrite also records the graph as rewritten, before the scheduler splits it: a split rewires the readers
// of an input from another backend to a copy, and a backend may reorder the nodes of its splits
struct rewrite_state {
    const llama_model * model = nullptr;
    const llama_model * other = nullptr;
    bool record = false;
    std::vector<std::string> rewritten;
    std::vector<ggml_context_ptr> ctxs;
};

constexpr const char * rewrite_node_name = "plan_test_rewrite";

bool rewrite_graph(ggml_backend_sched_t /*sched*/, ggml_cgraph * gf, void * user_data) {
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
    }
    return true;
}

// the first difference between a planned graph and a real one as its rewrite recorded it; empty when they are the same
std::string graph_difference(ggml_cgraph * planned, const std::vector<std::string> & real, const llama_model * model, const llama_model * other) {
    const std::vector<std::string> plan = describe_graph(planned, model, other);
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

// whether every tensor a planned graph reads that is not a model weight owns no data: the plan allocated nothing
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

// the compute buffers a planned context's scheduler holds, which a plan never allocates
size_t compute_bytes(const llama_context & ctx) {
    ggml_backend_sched_t sched = ctx.get_sched();
    size_t total = 0;
    for (int b = 0; b < ggml_backend_sched_get_n_backends(sched); ++b) {
        total += ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, b));
    }
    return total;
}

enum class backend_choice {
    cpu,
    gpu,
};

llama_model_ptr load_model(const std::string & path, backend_choice backend, bool load_mtp) {
    llama_model_params params = llama_model_default_params();
    static ggml_backend_dev_t no_devices[] = { nullptr };
    if (backend == backend_choice::cpu) {
        params.devices = no_devices;
    } else {
        params.n_gpu_layers = 999;
    }
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

llama_context_params make_params(const context_case & c, rewrite_state & rewrite) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx               = c.n_ctx;
    params.n_batch             = c.n_ctx;
    params.n_ubatch            = c.n_ubatch;
    params.n_seq_max           = c.n_seq_max;
    params.flash_attn_type     = c.flash_attn_type;
    params.n_threads           = 2;
    params.n_threads_batch     = 2;
    params.cb_graph_rewrite           = rewrite_graph;
    params.cb_graph_rewrite_user_data = &rewrite;
    params.n_graph_nodes_extra        = 1;
    return params;
}

llama_graph_plan_shape ubatch_shape(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, uint32_t n_cells = 0) {
    return { LLAMA_GRAPH_PLAN_WIDTH_UBATCH, n_tokens, n_seqs, n_outputs, n_cells, 0, -1, false, false, false };
}

llama_graph_plan_shape widest_shape(uint32_t n_cells = 0) {
    return { LLAMA_GRAPH_PLAN_WIDTH_WIDEST, 0, 0, 0, n_cells, 0, -1, false, false, false };
}

// the graph a real context reserves for a planned shape, in the nextn state the shape names
ggml_cgraph * reserve(llama_context * ctx, const llama_graph_plan_shape & shape) {
    llama_set_nextn_layer_offset(ctx, shape.nextn_layer_offset);
    llama_set_embeddings_nextn(ctx, shape.embeddings_nextn, shape.embeddings_nextn_masked);
    if (shape.width == LLAMA_GRAPH_PLAN_WIDTH_WIDEST) {
        return shape.n_cells == 0 ? llama_graph_reserve_widest(ctx) : llama_graph_reserve_widest_extent(ctx, shape.n_cells);
    }
    return shape.n_cells == 0
        ? llama_graph_reserve(ctx, shape.n_tokens, shape.n_seqs, shape.n_outputs)
        : llama_graph_reserve_extent(ctx, shape.n_tokens, shape.n_seqs, shape.n_outputs, shape.n_cells);
}

// what a context does past its creation parameters, which changes the graphs it builds
enum class feature_set {
    none,
    backend_sampler, // the target samples every sequence on the backend
    mtp_session,     // as a speculative session sets them: the target outputs nextn embeddings for every row, the
                     // draft for its output rows, and the draft samples on the backend
    mtp_masked,      // the same, with the target's nextn embeddings for its output rows only, as over shared memory
};

const char * feature_name(feature_set f) {
    switch (f) {
        case feature_set::none:            return "plain";
        case feature_set::backend_sampler: return "backend-sampler";
        case feature_set::mtp_session:     return "mtp-session";
        case feature_set::mtp_masked:      return "mtp-masked";
    }
    return "?";
}

struct member_features {
    bool nextn        = false;
    bool nextn_masked = false;
    bool sampler      = false; // a backend sampler on each sequence
};

member_features features_of(feature_set f, size_t member) {
    switch (f) {
        case feature_set::none:
            return {};
        case feature_set::backend_sampler:
            return { false, false, member == 0 };
        case feature_set::mtp_session:
            return member == 0 ? member_features{ true, false, false } : member_features{ true, true, true };
        case feature_set::mtp_masked:
            return member == 0 ? member_features{ true, true, false } : member_features{ true, true, true };
    }
    return {};
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

// whether a graph samples on the backend: it gathers the rows of the top-k sampler, which names them
bool samples_on_backend(ggml_cgraph * gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (strcmp(ggml_get_name(ggml_graph_node(gf, i)), "top_k_rows") == 0) {
            return true;
        }
    }
    return false;
}

struct plan_case {
    std::string label;
    const llama_model * target = nullptr;
    const llama_model * draft  = nullptr; // an MTP member reading the target, or none
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

void check_case(const plan_case & pc, feature_set features, outcome & out) {
    for (const context_case & cc : context_cases) {
        if (pc.draft != nullptr && !cc.drafts) {
            continue;
        }
        const bool session = features == feature_set::mtp_session || features == feature_set::mtp_masked;
        if (session && pc.draft == nullptr) {
            continue;
        }
        const std::string label = pc.label + " [" + cc.name + ", " + feature_name(features) + "]";
        rewrite_state rewrite_real;
        rewrite_state rewrite_plan;

        const uint32_t n_seqs = cc.n_seq_max;
        std::vector<llama_graph_plan_member> members;
        members.push_back({ pc.target, make_params(cc, rewrite_plan), -1, {
            ubatch_shape(n_seqs, n_seqs, n_seqs),
            ubatch_shape(cc.n_ubatch, n_seqs, n_seqs),
            widest_shape(),
        } });
        if (pc.extents) {
            members[0].shapes.push_back(ubatch_shape(n_seqs, n_seqs, n_seqs, 256));
            members[0].shapes.push_back(widest_shape(256));
        }
        if (pc.draft != nullptr) {
            // a speculative round's verify graph on the target, and the draft's graphs
            members[0].shapes.push_back(ubatch_shape(cc.n_ubatch, 1, 3));
            llama_context_params draft_params = make_params(cc, rewrite_plan);
            draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            members.push_back({ pc.draft, draft_params, 0, {
                ubatch_shape(cc.n_ubatch, 1, 1),
                ubatch_shape(1, 1, 1),
            } });
            if (pc.extents) {
                members[1].shapes.push_back(ubatch_shape(2, 1, 2, 256));
            }
        }

        // a member with features reserves its shapes before they attach and again after; the plan's chains and the real
        // contexts' chains are their own: a chain initializes its backend once
        std::vector<sampler_chains> plan_chains(members.size());
        std::vector<sampler_chains> real_chains(members.size());
        for (size_t m = 0; m < members.size(); ++m) {
            const member_features f = features_of(features, m);
            if (!f.nextn && !f.sampler) {
                continue;
            }
            const size_t n_before = members[m].shapes.size();
            for (size_t s = 0; s < n_before; ++s) {
                llama_graph_plan_shape after = members[m].shapes[s];
                after.embeddings_nextn        = f.nextn;
                after.embeddings_nextn_masked = f.nextn_masked;
                after.backend_sampling        = f.sampler;
                members[m].shapes.push_back(after);
            }
            if (f.sampler) {
                plan_chains[m].make(members[m].params.n_seq_max);
                real_chains[m].make(members[m].params.n_seq_max);
                members[m].params.samplers   = plan_chains[m].configs.data();
                members[m].params.n_samplers = plan_chains[m].configs.size();
            }
        }

        llama_graph_plan * plan = nullptr;
        try {
            plan = llama_graph_plan_build(members);
        } catch (const std::exception & e) {
            fail(out, label + ": the plan failed: " + e.what());
            continue;
        }

        std::vector<llama_context_ptr> real;
        for (size_t m = 0; m < members.size(); ++m) {
            llama_context_params params = make_params(cc, rewrite_real);
            params.ctx_type  = members[m].params.ctx_type;
            params.ctx_other = members[m].other >= 0 ? real[members[m].other].get() : nullptr;
            real.emplace_back(llama_init_from_model(const_cast<llama_model *>(members[m].model), params));
            if (!real.back()) {
                fail(out, label + ": the real context of member " + std::to_string(m) + " failed");
                break;
            }
        }
        // as a session attaches them to contexts it did not create: after creation, before the first shape sampling on
        // the backend
        std::vector<bool> attached(members.size(), false);
        const auto attach = [&](size_t m) {
            for (const llama_sampler_seq_config & config : real_chains[m].configs) {
                if (!llama_set_sampler(real[m].get(), config.seq_id, config.sampler)) {
                    fail(out, label + ": member " + std::to_string(m) + " does not sample on the backend");
                }
            }
            attached[m] = true;
        };
        if (real.size() != members.size() || !real.back()) {
            llama_graph_plan_free(plan);
            continue;
        }

        for (size_t m = 0; m < members.size(); ++m) {
            const llama_context & planned_ctx = *llama_graph_plan_context(plan, m);
            if (const size_t bytes = compute_bytes(planned_ctx); bytes != 0) {
                fail(out, label + ": the planned context of member " + std::to_string(m) + " holds " + std::to_string(bytes) + " compute bytes");
            }
            const llama_model * other = m > 0 ? members[0].model : nullptr;
            for (size_t s = 0; s < members[m].shapes.size(); ++s) {
                const std::string what = label + " member " + std::to_string(m) + " shape " + std::to_string(s);
                ggml_cgraph * planned = llama_graph_plan_graph(plan, m, s);
                rewrite_real.model  = members[m].model;
                rewrite_real.other  = other;
                rewrite_real.record = true;
                rewrite_real.rewritten.clear();
                if (members[m].shapes[s].backend_sampling && !attached[m]) {
                    attach(m);
                }
                if (!members[m].shapes[s].backend_sampling && attached[m]) {
                    fail(out, what + ": a shape without backend sampling follows one with it");
                    continue;
                }
                ggml_cgraph * reserved = reserve(real[m].get(), members[m].shapes[s]);
                rewrite_real.record = false;
                out.graphs++;
                if (reserved == nullptr || rewrite_real.rewritten.empty()) {
                    fail(out, what + ": the real reserve failed, or ran no rewrite");
                    continue;
                }
                if (const std::string diff = graph_difference(planned, rewrite_real.rewritten, members[m].model, other); !diff.empty()) {
                    fail(out, what + ": " + diff);
                }
                if (const std::string tensor = allocated_tensor(planned, *members[m].model, other); !tensor.empty()) {
                    fail(out, what + ": the planned graph reads " + tensor + ", which owns data");
                }
                bool rewritten = false;
                for (int i = 0; i < ggml_graph_n_nodes(planned); ++i) {
                    rewritten = rewritten || strcmp(ggml_get_name(ggml_graph_node(planned, i)), rewrite_node_name) == 0;
                }
                if (!rewritten) {
                    fail(out, what + ": the planned graph was not rewritten");
                }
                // a sampler the plan dropped would leave both graphs without it alike
                if (members[m].shapes[s].backend_sampling != samples_on_backend(planned)) {
                    fail(out, what + (members[m].shapes[s].backend_sampling ? ": the planned graph samples nothing on the backend"
                                                                            : ": the planned graph samples before its samplers attach"));
                }
            }
            llama_set_nextn_layer_offset(real[m].get(), 0);
            llama_set_embeddings_nextn(real[m].get(), false, false);
        }
        // the draft's context reads the target's, so it goes first
        while (!real.empty()) {
            real.pop_back();
        }
        llama_graph_plan_free(plan);
    }
}

// a context outputting nextn embeddings for every row builds a graph of its own: the plan is not equal to the built
// graph only because the flag changes nothing
void check_nextn_changes_graph(const plan_case & pc, outcome & out) {
    const context_case & cc = context_cases[0];
    rewrite_state rewrite;
    // a verify graph, whose outputs are fewer than its tokens
    std::vector<std::vector<std::string>> graphs;
    for (const bool nextn : { false, true }) {
        llama_graph_plan_shape verify = ubatch_shape(cc.n_ubatch, 1, 3);
        verify.embeddings_nextn = nextn;
        std::vector<llama_graph_plan_member> members;
        members.push_back({ pc.target, make_params(cc, rewrite), -1, { verify } });
        llama_graph_plan * plan = nullptr;
        try {
            plan = llama_graph_plan_build(members);
        } catch (const std::exception & e) {
            fail(out, pc.label + ": the plan failed: " + e.what());
            return;
        }
        graphs.push_back(describe_graph(llama_graph_plan_graph(plan, 0, 0), pc.target, nullptr));
        llama_graph_plan_free(plan);
    }
    if (graphs[0] == graphs[1]) {
        fail(out, pc.label + ": the nextn embeddings of every row change no planned graph");
    }
}

void check_bad_member_throws(const llama_model * model, outcome & out) {
    rewrite_state rewrite;
    std::vector<llama_graph_plan_member> members;
    members.push_back({ model, make_params(context_cases[0], rewrite), 0, { widest_shape() } });
    try {
        llama_graph_plan_free(llama_graph_plan_build(members));
        fail(out, "a member reading itself was planned");
    } catch (const std::runtime_error &) {
    }
    members[0].model = nullptr;
    members[0].other = -1;
    try {
        llama_graph_plan_free(llama_graph_plan_build(members));
        fail(out, "a null model was planned");
    } catch (const std::runtime_error &) {
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::string models_dir;
    backend_choice backend = backend_choice::cpu;
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            const std::string name = argv[++i];
            if (name == "CPU") {
                backend = backend_choice::cpu;
            } else if (name == "GPU") {
                backend = backend_choice::gpu;
            } else {
                fprintf(stderr, "%s: -b takes CPU or GPU\n", argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else {
            fprintf(stderr, "usage: %s --models DIR [-b CPU|GPU] [-v]\n", argv[0]);
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

    if (backend == backend_choice::gpu) {
        bool has_gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            has_gpu = has_gpu || ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        if (!has_gpu) {
            fprintf(stderr, "%s: no GPU device\n", argv[0]);
            return 1;
        }
    }

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
            llama_model_ptr model = load_model((std::filesystem::path(models_dir) / file).string(), backend, load_mtp);
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

    if (!cases.empty()) {
        check_bad_member_throws(cases.front().target, out);
    }
    for (const plan_case & pc : cases) {
        for (const feature_set f : { feature_set::none, feature_set::backend_sampler, feature_set::mtp_session, feature_set::mtp_masked }) {
            check_case(pc, f, out);
        }
        if (pc.draft != nullptr) {
            check_nextn_changes_graph(pc, out);
        }
    }

    printf("%s: %zu cases, %zu graphs compared, %zu failures\n", argv[0], cases.size(), out.graphs, out.failures);
    return out.failures == 0 && out.graphs > 0 ? 0 : 1;
}
