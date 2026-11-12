// checks the router lookahead of the MoE layers (llama_context_params.moe_router_ahead):
//   - with the parameter off, the graph has no "ffn_moe_logits_ahead" tensors
//   - with it on, every MoE layer with a following MoE layer has one, it reads the following layer's router weights,
//     and the graph without the lookahead nodes is the graph with the parameter off, node for node
//   - in each decode, the 12-token batch and every one-token decode, each lookahead equals the following layer's router
//     logits recomputed on the host from the layer's router input
//   - the logits with the parameter on are bitwise identical to the logits with it off
// and the extra router lookaheads (llama_context_params.moe_router_ahead_extra), over the NextN models of qwen35moe and
// qwen4exp:
//   - at the last trunk MoE layer, the router of the NextN layer of the same model or of another model, on the layer's
//     router input, and the router of the first MoE layer on the input token embeddings of a trunk or an MTP graph:
//     each node has its name and reads the right router weight, and equals the router logits recomputed on the host
//   - without extras the graph is the graph of today, node for node; with them, the graph without the extra nodes is
//     that graph and the logits are bitwise the same
//   - extras that name no MoE layer, no router, an unknown input or a width that does not match fail the context
//
// the models are the MoE models written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// the test reserves graphs to inspect their nodes
#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

static const char * AHEAD_PREFIX = "ffn_moe_logits_ahead-";
static const char * EXTRA_PREFIX = "ffn_moe_logits_ahead_extra-";
static const char * LOGITS_PREFIX = "ffn_moe_logits-";

// the layer of a tensor named <prefix><il>, or -1
static int layer_of(const ggml_tensor * t, const char * prefix) {
    const size_t n = strlen(prefix);
    return strncmp(ggml_get_name(t), prefix, n) == 0 ? atoi(ggml_get_name(t) + n) : -1;
}

static llama_model_ptr load_model(const std::string & path, std::vector<ggml_backend_dev_t> devs) {
    llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true;
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

struct context_opts {
    bool ahead = false;
    std::vector<llama_moe_router_ahead_extra> extra;
    bool extra_null = false; // a NULL list with the count of the extras
    const llama_moe_router_ahead_extra * extra_ptr = nullptr; // the list to pass instead of extra
    llama_context_type ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    llama_context * ctx_other = nullptr;
    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data = nullptr;
};

// null if the context cannot be created, with the log in g_log
static llama_context_ptr make_context(llama_model * model, const context_opts & opts) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 0;
    cparams.n_threads         = 4;
    cparams.n_threads_batch   = 4;
    cparams.n_ubatch          = 64;
    cparams.moe_router_ahead  = opts.ahead;
    cparams.moe_router_ahead_extra   = opts.extra_null ? nullptr : opts.extra_ptr ? opts.extra_ptr : opts.extra.data();
    cparams.n_moe_router_ahead_extra = opts.extra.size();
    cparams.ctx_type          = opts.ctx_type;
    cparams.ctx_other         = opts.ctx_other;
    cparams.cb_eval           = opts.cb_eval;
    cparams.cb_eval_user_data = opts.cb_eval_user_data;
    g_log.clear();
    return llama_context_ptr(llama_init_from_model(model, cparams));
}

static llama_context_ptr new_context(llama_model * model, const context_opts & opts) {
    llama_context_ptr lctx = make_context(model, opts);
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

static llama_context_ptr new_context(llama_model * model, bool ahead, ggml_backend_sched_eval_callback cb_eval = nullptr, void * ud = nullptr) {
    context_opts opts;
    opts.ahead             = ahead;
    opts.cb_eval           = cb_eval;
    opts.cb_eval_user_data = ud;
    return new_context(model, opts);
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

static constexpr size_t N_TOKENS = 16;
static constexpr size_t N_BATCH  = 12;

struct capture;
static void begin_decode(capture * cap);

// the logits of N_BATCH tokens in one batch, then of the other tokens one at a time, from an empty memory; a capture
//     records each decode apart
//     an MTP context takes a seeded state embedding with each token
static std::vector<float> get_logits(llama_context * lctx, size_t n_batch = N_BATCH, capture * cap = nullptr, size_t n_state = 0) {
    llama_memory_clear(llama_get_memory(lctx), true);

    const llama_model * model = llama_get_model(lctx);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> tokens = get_tokens(N_TOKENS, n_vocab, 42);

    std::vector<float> state(N_TOKENS*n_state);
    std::mt19937 gen(7);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (float & v : state) {
        v = dis(gen);
    }

    std::vector<float> ret;
    common_batch batch(lctx);
    for (size_t i = 0; i < n_batch; i++) {
        const int32_t idx = batch.add(tokens[i], i, 0, true);
        if (n_state > 0) {
            batch.set_embd_state(idx, { state.data() + i*n_state, 1, n_state });
        }
    }
    begin_decode(cap);
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode");
    }
    for (size_t i = 0; i < n_batch; i++) {
        const float * l = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    for (size_t i = n_batch; i < tokens.size(); i++) {
        common_batch one(lctx);
        const int32_t idx = one.add(tokens[i], i, 0, true);
        if (n_state > 0) {
            one.set_embd_state(idx, { state.data() + i*n_state, 1, n_state });
        }
        begin_decode(cap);
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, one.get())) {
            throw std::runtime_error("failed to decode");
        }
        const float * l = llama_get_logits_ith(lctx, 0);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    return ret;
}

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-90s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

// the nodes of a graph, without the nodes that only the lookahead tensors need, each as its op, its name and its
//     sources: a node by its position in the list, any other tensor by its name; ggml names an unnamed node by its
//     position in the graph, which the lookahead nodes shift, so such a name is left out
static std::vector<std::string> main_nodes(ggml_cgraph * gf, std::set<const ggml_tensor *> & extra) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    std::set<const ggml_tensor *> nodes;
    std::set<const ggml_tensor *> consumed;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        nodes.insert(t);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s]) {
                consumed.insert(t->src[s]);
            }
        }
    }

    // walk back from every sink that is not a lookahead
    std::set<const ggml_tensor *> needed;
    std::vector<const ggml_tensor *> stack;
    for (const ggml_tensor * t : nodes) {
        if (!consumed.count(t) && layer_of(t, AHEAD_PREFIX) < 0 && layer_of(t, EXTRA_PREFIX) < 0) {
            stack.push_back(t);
        }
    }
    while (!stack.empty()) {
        const ggml_tensor * t = stack.back();
        stack.pop_back();
        if (!needed.insert(t).second) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s]) {
                stack.push_back(t->src[s]);
            }
        }
        if (t->view_src) {
            stack.push_back(t->view_src);
        }
    }

    // the name of a view or a copy of an unnamed node holds the position of that node, which the extra nodes shift
    auto name_of = [](const ggml_tensor * t) {
        const char * name = ggml_get_name(t);
        if (strncmp(name, "node_", 5) == 0 || strncmp(name, "leaf_", 5) == 0) {
            return std::string();
        }
        std::string ret;
        for (const char * c = name; *c; ++c) {
            ret += *c;
            if (c - name >= 4 && (strncmp(c - 4, "node_", 5) == 0 || strncmp(c - 4, "leaf_", 5) == 0)) {
                while (isdigit((unsigned char) c[1])) {
                    ++c;
                }
            }
        }
        return ret;
    };

    std::map<const ggml_tensor *, size_t> pos;
    std::vector<std::string> ret;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        if (!needed.count(t)) {
            extra.insert(t);
            continue;
        }
        std::string desc = std::string(ggml_op_desc(t)) + " " + name_of(t) + " <-";
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s]) {
                desc += " " + (pos.count(t->src[s]) ? "#" + std::to_string(pos.at(t->src[s])) : name_of(t->src[s]));
            }
        }
        pos[t] = ret.size();
        ret.push_back(desc);
    }
    return ret;
}

static void test_graph(llama_model * model, const std::string & stem, checker & check) {
    const int32_t n_layer = llama_model_n_layer(model);

    std::set<int> expected;
    for (int32_t il = 0; il < n_layer; ++il) {
        if (llama_model_moe_router_ahead_layer(model, il) >= 0) {
            expected.insert(il);
        }
    }
    check(!expected.empty(), stem + ": the model has MoE layers with a following MoE layer");

    for (const uint32_t n_tokens : { 1u, (uint32_t) N_BATCH }) {
        const std::string what = stem + ", " + std::to_string(n_tokens) + " tokens: ";

        llama_context_ptr off = new_context(model, false);
        llama_context_ptr on  = new_context(model, true);

        ggml_cgraph * gf_off = llama_graph_reserve(off.get(), n_tokens, 1, n_tokens);
        bool off_has_ahead = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf_off); ++i) {
            off_has_ahead |= strstr(ggml_get_name(ggml_graph_node(gf_off, i)), "_ahead") != nullptr;
        }
        check(gf_off != nullptr && !off_has_ahead, what + "off: no lookahead nodes");
        std::set<const ggml_tensor *> extra_off;
        const std::vector<std::string> nodes_off = main_nodes(gf_off, extra_off);

        ggml_cgraph * gf_on = llama_graph_reserve(on.get(), n_tokens, 1, n_tokens);
        check(gf_on != nullptr, what + "on: the graph builds");

        std::set<int> found;
        bool well_formed = true;
        for (int i = 0; i < ggml_graph_n_nodes(gf_on); ++i) {
            const ggml_tensor * t = ggml_graph_node(gf_on, i);
            const int il = layer_of(t, AHEAD_PREFIX);
            if (il < 0) {
                continue;
            }
            found.insert(il);
            const std::string router = "blk." + std::to_string(llama_model_moe_router_ahead_layer(model, il)) + ".ffn_gate_inp.weight";
            well_formed &= t->op == GGML_OP_MUL_MAT && router == ggml_get_name(t->src[0]) && t->type == GGML_TYPE_F32 &&
                (t->flags & GGML_TENSOR_FLAG_OUTPUT) && t->ne[0] == t->src[0]->ne[1] && t->ne[1] == n_tokens;
        }
        check(found == expected, what + "on: a lookahead for each MoE layer with a following MoE layer");
        check(well_formed, what + "on: each reads the following layer's router weights, an F32 output");

        std::set<const ggml_tensor *> extra_on;
        const std::vector<std::string> nodes_on = main_nodes(gf_on, extra_on);
        check(nodes_on == nodes_off, what + "on: without the lookahead nodes, the graph is the graph with it off");
        if (nodes_on != nodes_off) {
            for (size_t i = 0; i < std::max(nodes_on.size(), nodes_off.size()); ++i) {
                const std::string a = i < nodes_on.size()  ? nodes_on[i]  : "-";
                const std::string b = i < nodes_off.size() ? nodes_off[i] : "-";
                if (a != b) {
                    fprintf(stderr, "    first difference at node %zu: on %s, off %s\n", i, a.c_str(), b.c_str());
                    break;
                }
            }
        }
        // the lookahead tensors, and the scaled router inputs of an architecture that scales inside its router
        bool extra_ok = extra_off.empty();
        for (const ggml_tensor * t : extra_on) {
            extra_ok &= layer_of(t, AHEAD_PREFIX) >= 0 || t->op == GGML_OP_MUL;
        }
        check(extra_ok && extra_on.size() >= expected.size(), what + "on: the extra nodes are the lookahead and its router input");
    }
}

// a node that a decode computed, copied when the scheduler computes it: the graph of a later decode reuses the
//     memory of the tensors, so nothing is kept by pointer except the model weights
struct captured {
    ggml_op op;
    int64_t n_tokens;                      // ne[1]
    std::vector<float> data;               // empty unless the node is F32 and contiguous
    std::string src_name[2];
    std::vector<float> src1_data;          // the second source of a router node, which the name of that source may not tell apart
    bool repeated = false;                 // more nodes of the decode have this name, so the entry is ambiguous
    const ggml_tensor * src_weight[2] = {}; // a source that is a model weight, which outlives the graphs
};

// the nodes of each decode, by name; ggml names every unnamed node of a graph uniquely, but a named one may repeat
struct capture {
    std::vector<std::map<std::string, captured>> decodes;
};

static void begin_decode(capture * cap) {
    if (cap != nullptr) {
        cap->decodes.emplace_back();
    }
}

static bool capture_node(ggml_tensor * t, bool ask, void * ud) {
    if (ask) {
        return true;
    }
    auto * cap = (capture *) ud;
    if (cap->decodes.empty()) {
        return true; // the warmup decode at context creation
    }
    const auto inserted = cap->decodes.back().emplace(ggml_get_name(t), captured{});
    captured & c = inserted.first->second;
    if (!inserted.second) {
        // a graph builder may give several nodes one name (a projection and its bias); keep the first and mark the
        // name so that no check reads the data of an ambiguous node
        c.repeated = true;
        return true;
    }
    c.op       = t->op;
    c.n_tokens = t->ne[1];
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        c.data.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, c.data.data(), 0, ggml_nbytes(t));
    }
    if (strncmp(ggml_get_name(t), LOGITS_PREFIX, strlen(LOGITS_PREFIX)) == 0 && t->src[1] && t->src[1]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(t->src[1])) {
        c.src1_data.resize(ggml_nelements(t->src[1]));
        ggml_backend_tensor_get(t->src[1], c.src1_data.data(), 0, ggml_nbytes(t->src[1]));
    }
    for (int s = 0; s < 2; ++s) {
        const ggml_tensor * src = t->src[s];
        if (src == nullptr) {
            continue;
        }
        c.src_name[s] = ggml_get_name(src);
        if (src->buffer != nullptr && ggml_backend_buffer_get_usage(src->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            c.src_weight[s] = src;
        }
    }
    return true;
}

// the rows of a weight as F32
static std::vector<float> weight_rows(const ggml_tensor * w) {
    std::vector<uint8_t> bytes(ggml_nbytes(w));
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(w), bytes.data(), 0, bytes.size());
    std::vector<float> ret(ggml_nelements(w));
    if (w->type == GGML_TYPE_F32) {
        memcpy(ret.data(), bytes.data(), bytes.size());
        return ret;
    }
    const auto * traits = ggml_get_type_traits(w->type);
    if (!traits->to_float) {
        throw std::runtime_error(std::string("no to_float for ") + ggml_type_name(w->type));
    }
    for (int64_t r = 0; r < ggml_nrows(w); ++r) {
        traits->to_float(bytes.data() + r*w->nb[1], ret.data() + r*w->ne[0], w->ne[0]);
    }
    return ret;
}

// max |a - b| over max |b|
static double rel_error(const std::vector<float> & a, const std::vector<double> & b) {
    double err = 0.0;
    double mag = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
        mag = std::max(mag, std::fabs(b[i]));
    }
    return mag > 0.0 ? err / mag : err;
}

// a router node against its weight applied on the host to the rows x, [n_embd, n_tokens]; -1 if the shapes differ
static double matmul_error(const captured * a, const std::vector<float> & x) {
    const ggml_tensor * w = a->src_weight[0];
    const std::vector<float> wf = weight_rows(w);
    const int64_t n_embd   = w->ne[0];
    const int64_t n_expert = w->ne[1];
    const int64_t n_tokens = a->n_tokens;
    if ((int64_t) a->data.size() != n_expert*n_tokens || (int64_t) x.size() != n_embd*n_tokens) {
        return -1.0;
    }
    std::vector<double> ref(n_expert*n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t e = 0; e < n_expert; ++e) {
            double sum = 0.0;
            for (int64_t k = 0; k < n_embd; ++k) {
                sum += (double) wf[e*n_embd + k] * x[t*n_embd + k];
            }
            ref[t*n_expert + e] = sum;
        }
    }
    return rel_error(a->data, ref);
}

// the lookahead of layer il in one decode against the following router recomputed on the host, or -1 if the decode
//     lacks a node that the check needs
static double ahead_error(const llama_model * model, const std::map<std::string, captured> & nodes, int il) {
    const auto find = [&](const std::string & name) -> const captured * {
        const auto it = nodes.find(name);
        return it == nodes.end() || it->second.repeated ? nullptr : &it->second;
    };
    const captured * a      = find(AHEAD_PREFIX + std::to_string(il));
    const captured * logits = find(LOGITS_PREFIX + std::to_string(il));
    if (a == nullptr || logits == nullptr || a->data.empty() || a->src_weight[0] == nullptr) {
        return -1.0;
    }

    // layer il's router input; an architecture that scales inside its router (gemma4) scales the shared normalized
    // input by each layer's scale, so the lookahead scales it by the following layer's
    std::vector<float> x_next;
    if (a->src_name[1] == logits->src_name[1]) {
        const captured * x = find(logits->src_name[1]);
        if (x == nullptr || x->data.empty()) {
            return -1.0;
        }
        x_next = x->data;
    } else {
        const captured * scaled = find(a->src_name[1]);
        const captured * x      = find(logits->src_name[1]);
        const std::string scale = "blk." + std::to_string(llama_model_moe_router_ahead_layer(model, il)) + ".";
        if (scaled == nullptr || x == nullptr || scaled->op != GGML_OP_MUL || x->op != GGML_OP_MUL ||
            x->src_name[0] != scaled->src_name[0] || scaled->src_weight[1] == nullptr ||
            strncmp(scaled->src_name[1].c_str(), scale.c_str(), scale.size()) != 0) {
            fprintf(stderr, "    layer %d: the lookahead input is neither the router input nor its scaled norm\n", il);
            return -1.0;
        }
        const captured * norm = find(scaled->src_name[0]);
        if (norm == nullptr || norm->data.empty()) {
            return -1.0;
        }
        const std::vector<float> s = weight_rows(scaled->src_weight[1]);
        x_next = norm->data;
        for (size_t i = 0; i < x_next.size(); ++i) {
            x_next[i] *= s[i % s.size()];
        }
    }

    return matmul_error(a, x_next);
}

static void test_values(llama_model * model, const std::string & stem, checker & check) {
    capture cap;
    llama_context_ptr lctx = new_context(model, true, capture_node, &cap);
    get_logits(lctx.get(), N_BATCH, &cap);

    std::vector<int> layers;
    for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
        if (llama_model_moe_router_ahead_layer(model, il) >= 0) {
            layers.push_back(il);
        }
    }

    // the batch decode, then each one-token decode
    check(cap.decodes.size() == 1 + N_TOKENS - N_BATCH, stem + ": a capture for each decode");
    for (size_t d = 0; d < cap.decodes.size(); ++d) {
        const size_t n_tokens = d == 0 ? N_BATCH : 1;
        bool all_ok = !layers.empty();
        double worst = 0.0;
        for (const int il : layers) {
            const auto it = cap.decodes[d].find(AHEAD_PREFIX + std::to_string(il));
            const double err = ahead_error(model, cap.decodes[d], il);
            all_ok &= it != cap.decodes[d].end() && !it->second.repeated && it->second.n_tokens == (int64_t) n_tokens && err >= 0.0 && err < 1e-2;
            worst = std::max(worst, err);
        }
        if (d > 1 && all_ok) {
            continue; // one line for the batch and one for the first one-token decode; a failure prints its decode
        }
        char buf[64];
        snprintf(buf, sizeof(buf), " (worst %.2e)", worst);
        check(all_ok, stem + ", decode " + std::to_string(d) + " (" + std::to_string(n_tokens) + " tokens): " +
            std::to_string(layers.size()) + " lookaheads equal the following router on the router input" + buf);
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// the extra router lookaheads

// what one extra of a list must produce
struct expect {
    int il;
    int k;                             // index in the list
    const ggml_tensor * router;        // the router weight it reads
    llama_moe_router_ahead_input input;
};

static llama_moe_router_ahead_extra extra_of(int il, const llama_model * model, int il_router, llama_moe_router_ahead_input input) {
    return { il, model, il_router, input };
}

static const ggml_tensor * router_of(const llama_model * model, int il) {
    return model->layers[il].ffn_gate_inp;
}

static int first_moe_layer(const llama_model * model) {
    for (int il = 0; il < llama_model_n_layer(model); ++il) {
        if (router_of(model, il) != nullptr) {
            return il;
        }
    }
    return -1;
}

static int last_moe_layer(const llama_model * model) {
    for (int il = llama_model_n_layer(model) - 1; il >= 0; --il) {
        if (router_of(model, il) != nullptr) {
            return il;
        }
    }
    return -1;
}

// the extra nodes of a graph, by name
static std::map<std::string, std::pair<int, const ggml_tensor *>> extra_nodes(ggml_cgraph * gf) {
    std::map<std::string, std::pair<int, const ggml_tensor *>> ret;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        if (layer_of(t, EXTRA_PREFIX) >= 0) {
            ret[ggml_get_name(t)] = { i, t };
        }
    }
    return ret;
}

static std::string extra_name(int il, int k) {
    return EXTRA_PREFIX + std::to_string(il) + "-" + std::to_string(k);
}

// the graph of a context with the extras against the graph of a context without them
static void check_extra_graph(llama_context * with, llama_context * without, const std::vector<expect> & expects,
        const std::string & what, checker & check) {
    for (const uint32_t n_tokens : { 1u, (uint32_t) N_BATCH }) {
        const std::string w = what + ", " + std::to_string(n_tokens) + " tokens: ";

        ggml_cgraph * gf_off = llama_graph_reserve(without, n_tokens, 1, n_tokens);
        std::set<const ggml_tensor *> extra_off;
        const std::vector<std::string> nodes_off = main_nodes(gf_off, extra_off);
        const int n_off = ggml_graph_n_nodes(gf_off);

        ggml_cgraph * gf_on = llama_graph_reserve(with, n_tokens, 1, n_tokens);
        check(gf_on != nullptr, w + "the graph builds");
        if (gf_on == nullptr) {
            return;
        }
        const auto nodes = extra_nodes(gf_on);
        check(nodes.size() == expects.size() && ggml_graph_n_nodes(gf_on) == n_off + (int) expects.size(),
            w + "one extra node for each extra, and no other node");

        bool well_formed = true;
        bool ordered     = true;
        for (const expect & e : expects) {
            const auto it = nodes.find(extra_name(e.il, e.k));
            if (it == nodes.end()) {
                well_formed = false;
                continue;
            }
            const ggml_tensor * t = it->second.second;
            well_formed &= t->op == GGML_OP_MUL_MAT && t->src[0] == e.router && t->type == GGML_TYPE_F32 &&
                (t->flags & GGML_TENSOR_FLAG_OUTPUT) && t->ne[0] == e.router->ne[1] && t->ne[1] == n_tokens &&
                t->src[1]->ne[0] == e.router->ne[0];
            // before the router of its layer, and after the extras of the layer that precede it in the list
            for (int i = 0; i < it->second.first; ++i) {
                ordered &= layer_of(ggml_graph_node(gf_on, i), LOGITS_PREFIX) != e.il;
            }
            for (const expect & f : expects) {
                if (f.il == e.il && f.k < e.k) {
                    const auto before = nodes.find(extra_name(f.il, f.k));
                    ordered &= before != nodes.end() && before->second.first < it->second.first;
                }
            }
        }
        check(well_formed, w + "each is the router times its input, an F32 output, named by layer and index");
        check(ordered, w + "each is built before the router of its layer, in list order");

        std::set<const ggml_tensor *> extra_on;
        const std::vector<std::string> nodes_on = main_nodes(gf_on, extra_on);
        if (nodes_on != nodes_off) {
            for (size_t i = 0; i < std::max(nodes_on.size(), nodes_off.size()); ++i) {
                const std::string a = i < nodes_on.size()  ? nodes_on[i]  : "-";
                const std::string b = i < nodes_off.size() ? nodes_off[i] : "-";
                if (a != b) {
                    fprintf(stderr, "    first difference at node %zu: with %s, without %s\n", i, a.c_str(), b.c_str());
                    break;
                }
            }
        }
        check(nodes_on == nodes_off && extra_on.size() == extra_off.size() + expects.size(),
            w + "without the extra nodes, the graph is the graph without extras, node for node");
    }
}

// the node of one extra in one decode against the router applied to the host rows x
static double extra_error(const std::map<std::string, captured> & nodes, const expect & e, const std::vector<float> & x) {
    const auto it = nodes.find(extra_name(e.il, e.k));
    if (it == nodes.end() || it->second.repeated || it->second.data.empty() || it->second.src_weight[0] != e.router) {
        return -1.0;
    }
    return matmul_error(&it->second, x);
}

// the rows of the table for the tokens of decode d of get_logits
static std::vector<float> table_rows(const ggml_tensor * table, size_t d, size_t n_vocab) {
    const std::vector<llama_token> tokens = get_tokens(N_TOKENS, n_vocab, 42);
    const std::vector<float> w = weight_rows(table);
    const size_t first = d == 0 ? 0 : N_BATCH + d - 1;
    const size_t count = d == 0 ? N_BATCH : 1;
    std::vector<float> ret;
    for (size_t i = 0; i < count; ++i) {
        const float * row = w.data() + (size_t) tokens[first + i] * table->ne[0];
        ret.insert(ret.end(), row, row + table->ne[0]);
    }
    return ret;
}

// the extras in the decodes: the router input from the captured nodes, the token embeddings from the table that the
//     graph reads, which is `table` for a trunk graph and found as the source of the token embedding node of an MTP graph
static void check_extra_values(llama_model * model, const context_opts & opts, const std::vector<expect> & expects,
        const ggml_tensor * trunk_table, bool mtp, const std::string & what, checker & check) {
    capture cap;
    context_opts o = opts;
    o.cb_eval           = capture_node;
    o.cb_eval_user_data = &cap;
    llama_context_ptr lctx = new_context(model, o);
    get_logits(lctx.get(), N_BATCH, &cap, mtp ? llama_model_n_embd_out(model) : 0);

    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    check(cap.decodes.size() == 1 + N_TOKENS - N_BATCH, what + ": a capture for each decode");
    bool all_ok = true;
    double worst = 0.0;
    for (size_t d = 0; d < cap.decodes.size(); ++d) {
        const auto & nodes = cap.decodes[d];
        for (const expect & e : expects) {
            std::vector<float> x;
            if (e.input == LLAMA_MOE_ROUTER_AHEAD_INPUT_ROUTER) {
                const auto logits = nodes.find(LOGITS_PREFIX + std::to_string(e.il));
                if (logits == nodes.end() || logits->second.src1_data.empty()) {
                    fprintf(stderr, "    decode %zu, layer %d: no router input\n", d, e.il);
                    all_ok = false;
                    continue;
                }
                x = logits->second.src1_data;
            } else if (mtp) {
                const auto embd = nodes.find("mtp_tok_embd-" + std::to_string(e.il));
                if (embd == nodes.end() || embd->second.src_weight[0] == nullptr) {
                    all_ok = false;
                    continue;
                }
                x = table_rows(embd->second.src_weight[0], d, n_vocab);
            } else {
                x = table_rows(trunk_table, d, n_vocab);
            }
            const double err = extra_error(nodes, e, x);
            all_ok &= err >= 0.0 && err < 1e-2;
            worst = std::max(worst, err);
        }
    }
    char buf[64];
    snprintf(buf, sizeof(buf), " (worst %.2e)", worst);
    check(all_ok && !expects.empty(), what + ": in each decode every extra equals its router on the host-computed input" + buf);
}

struct extra_case {
    std::string what;
    llama_model * model;
    context_opts opts;
    std::vector<expect> expects;
    const ggml_tensor * trunk_table = nullptr;
    bool mtp = false;
};

static void run_extra_case(const extra_case & c, checker & check) {
    context_opts base = c.opts;
    base.extra.clear();
    llama_context_ptr without = new_context(c.model, base);
    llama_context_ptr with    = new_context(c.model, c.opts);
    check_extra_graph(with.get(), without.get(), c.expects, c.what, check);
    check_extra_values(c.model, c.opts, c.expects, c.trunk_table, c.mtp, c.what, check);

    const size_t n_state = c.mtp ? llama_model_n_embd_out(c.model) : 0;
    const std::vector<float> ref = get_logits(without.get(), N_BATCH, nullptr, n_state);
    const std::vector<float> got = get_logits(with.get(), N_BATCH, nullptr, n_state);
    check(ref.size() == got.size() && memcmp(ref.data(), got.data(), ref.size()*sizeof(float)) == 0,
        c.what + ": logits with the extras are bitwise the logits without them");
}

static void test_extra(const std::string & dir, const std::string & arch, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    llama_model_ptr mp_full  = load_model(dir + "/" + arch + "-moe-nextn.gguf", devs);
    llama_model_ptr mp_other = load_model(dir + "/" + arch + "-moe-nextn.gguf", devs);
    llama_model_ptr mp_only  = load_model(dir + "/" + arch + "-moe-nextn-only.gguf", devs);
    llama_model * full  = mp_full.get();
    llama_model * other = mp_other.get();
    llama_model * only  = mp_only.get();

    const int n_layer = llama_model_n_layer(full);
    const int last    = last_moe_layer(full);
    const int first   = first_moe_layer(full);
    check(llama_model_n_layer_nextn(full) == 1 && llama_model_n_layer(only) == n_layer && last >= 0 && first >= 0 &&
        router_of(full, n_layer) != nullptr && router_of(only, n_layer) != nullptr && router_of(only, 0) == nullptr,
        arch + ": the models have trunk MoE layers and a NextN layer with a router, the NextN-only file only the latter");

    const auto ROUTER = LLAMA_MOE_ROUTER_AHEAD_INPUT_ROUTER;
    const auto EMBD   = LLAMA_MOE_ROUTER_AHEAD_INPUT_EMBD;

    // 1, 2: the router of the NextN layer on the router input of the last trunk MoE layer, from the model itself and
    //     from another model; with the layer's own lookahead on, and with several extras for one layer
    {
        extra_case c;
        c.what  = arch + ": NextN router on the last trunk layer's router input, own model";
        c.model = full;
        c.opts.extra   = { extra_of(last, nullptr, n_layer, ROUTER) };
        c.expects      = { { last, 0, router_of(full, n_layer), ROUTER } };
        run_extra_case(c, check);

        c.what         = arch + ": the same from the model passed explicitly, with moe_router_ahead on";
        c.opts.ahead   = true;
        c.opts.extra   = { extra_of(last, full, n_layer, ROUTER) };
        run_extra_case(c, check);

        c.what         = arch + ": the same from another model, which is the file of NextN layers alone";
        c.opts.ahead   = false;
        c.opts.extra   = { extra_of(last, only, n_layer, ROUTER) };
        c.expects      = { { last, 0, router_of(only, n_layer), ROUTER } };
        run_extra_case(c, check);

        c.what         = arch + ": several extras of one layer and of two layers, in list order";
        c.opts.extra   = { extra_of(last, only, n_layer, ROUTER), extra_of(first, nullptr, last, ROUTER),
                           extra_of(last, nullptr, n_layer, ROUTER), extra_of(last, other, first, ROUTER) };
        c.expects      = { { last, 0, router_of(only, n_layer), ROUTER }, { first, 1, router_of(full, last), ROUTER },
                           { last, 2, router_of(full, n_layer), ROUTER }, { last, 3, router_of(other, first), ROUTER } };
        run_extra_case(c, check);

        // the input token embeddings of a trunk graph
        c.what         = arch + ": the first layer's router on the input token embeddings of the trunk graph";
        c.opts.extra   = { extra_of(last, nullptr, first, EMBD), extra_of(last, nullptr, n_layer, ROUTER) };
        c.expects      = { { last, 0, router_of(full, first), EMBD }, { last, 1, router_of(full, n_layer), ROUTER } };
        c.trunk_table  = full->tok_embd;
        run_extra_case(c, check);
    }

    // 3: the target's first MoE layer router on the token embeddings of the NextN head's MTP graph
    {
        context_opts trunk_opts;
        llama_context_ptr trunk = new_context(full, trunk_opts);

        extra_case c;
        c.mtp   = true;
        c.what  = arch + ": MTP context, the target's first router on the token embeddings, NextN layer of the same model";
        c.model = full;
        c.opts.ctx_type   = LLAMA_CONTEXT_TYPE_MTP;
        c.opts.ctx_other  = trunk.get();
        c.opts.extra      = { extra_of(n_layer, full, first, EMBD) };
        c.expects         = { { n_layer, 0, router_of(full, first), EMBD } };
        run_extra_case(c, check);

        c.what            = arch + ": MTP context over the file of NextN layers alone, the target's router on the token embeddings";
        c.model           = only;
        c.opts.extra      = { extra_of(n_layer, full, first, EMBD), extra_of(n_layer, nullptr, n_layer, ROUTER) };
        c.expects         = { { n_layer, 0, router_of(full, first), EMBD }, { n_layer, 1, router_of(only, n_layer), ROUTER } };
        run_extra_case(c, check);
    }

    // 4: an empty list, with and without a pointer
    {
        context_opts plain;
        llama_context_params p = llama_context_default_params();
        check(p.moe_router_ahead_extra == nullptr && p.n_moe_router_ahead_extra == 0, arch + ": the defaults are no extras");

        llama_context_ptr a = new_context(full, plain);
        const llama_moe_router_ahead_extra unused = extra_of(last, nullptr, n_layer, ROUTER);
        context_opts pointed;
        pointed.extra_ptr = &unused; // a list that is not read: the count is 0
        llama_context_ptr b = new_context(full, pointed);
        const std::vector<float> la = get_logits(a.get());
        const std::vector<float> lb = get_logits(b.get());
        ggml_cgraph * ga = llama_graph_reserve(a.get(), N_BATCH, 1, N_BATCH);
        ggml_cgraph * gb = llama_graph_reserve(b.get(), N_BATCH, 1, N_BATCH);
        std::set<const ggml_tensor *> ea, eb;
        check(ggml_graph_n_nodes(ga) == ggml_graph_n_nodes(gb) && main_nodes(ga, ea) == main_nodes(gb, eb) && ea.empty() && eb.empty() &&
            la.size() == lb.size() && memcmp(la.data(), lb.data(), la.size()*sizeof(float)) == 0,
            arch + ": an empty list is the graph and the logits of no list");
    }

    // 5: extras that the context refuses
    {
        struct bad_case {
            std::string what;
            llama_model * model;
            llama_moe_router_ahead_extra extra;
        };
        const std::vector<bad_case> bad = {
            { "a negative layer",                          full, extra_of(-1, nullptr, n_layer, ROUTER) },
            { "a layer beyond the model",                  full, extra_of(n_layer + 1, nullptr, n_layer, ROUTER) },
            { "a layer without a router in the file of NextN layers", only, extra_of(0, nullptr, n_layer, ROUTER) },
            { "a router layer beyond the model",           full, extra_of(last, nullptr, n_layer + 1, ROUTER) },
            { "a negative router layer",                   full, extra_of(last, nullptr, -1, ROUTER) },
            { "a router layer without a router",           full, extra_of(last, only, 0, ROUTER) },
            { "an unknown input",                          full, extra_of(last, nullptr, n_layer, (llama_moe_router_ahead_input) 7) },
        };
        for (const bad_case & b : bad) {
            context_opts o;
            o.extra = { b.extra };
            llama_context_ptr lctx = make_context(b.model, o);
            check(!lctx && g_log.find("moe_router_ahead_extra[0]") != std::string::npos, arch + ": the context is refused for " + b.what);
        }

        context_opts o;
        o.extra      = { extra_of(last, nullptr, n_layer, ROUTER) };
        o.extra_null = true;
        check(!make_context(full, o), arch + ": the context is refused for a count without a list");
    }
}

// a router that takes another width than the input: that of deepseek2
static void test_extra_width(const std::string & dir, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    llama_model_ptr mp_a = load_model(dir + "/qwen35moe-moe-nextn.gguf", devs);
    llama_model_ptr mp_b = load_model(dir + "/deepseek2-moe.gguf", devs);
    llama_model * a = mp_a.get();
    llama_model * b = mp_b.get();
    const int la = last_moe_layer(a);
    const int lb = last_moe_layer(b);
    const int ia = (int) a->hparams.n_embd;
    const int ib = (int) b->hparams.n_embd;
    check(ia != ib && la >= 0 && lb >= 0, "the two models differ in width and have MoE layers");
    if (ia == ib || la < 0 || lb < 0) {
        return;
    }

    for (const auto input : { LLAMA_MOE_ROUTER_AHEAD_INPUT_ROUTER, LLAMA_MOE_ROUTER_AHEAD_INPUT_EMBD }) {
        const std::string in = input == LLAMA_MOE_ROUTER_AHEAD_INPUT_ROUTER ? "router input" : "token embeddings";
        context_opts o;
        o.extra = { extra_of(la, b, lb, input) };
        llama_context_ptr lctx = make_context(a, o);
        check(!lctx && g_log.find("moe_router_ahead_extra[0]") != std::string::npos,
            "the context is refused for a router of another width on the " + in);

        // the same router from a model of the same width is fine
        o.extra = { extra_of(la, a, la, input) };
        check(make_context(a, o) != nullptr, "a router of the same width on the " + in + " is accepted");
    }

    // an MTP graph has token embeddings of the model's width, so a router of another width fails the graph build
    context_opts trunk_opts;
    llama_context_ptr trunk = new_context(a, trunk_opts);
    context_opts o;
    o.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    o.ctx_other = trunk.get();
    o.extra     = { extra_of((int) llama_model_n_layer(a), b, lb, LLAMA_MOE_ROUTER_AHEAD_INPUT_EMBD) };
    check(!make_context(a, o) && g_log.find("moe_router_ahead_extra[0]") != std::string::npos,
        "the context is refused for a router of another width on the token embeddings of an MTP graph");
}

static void test_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string();
    llama_model_ptr model = load_model(path, devs);

    test_graph(model.get(), stem, check);
    test_values(model.get(), stem, check);

    llama_context_ptr off = new_context(model.get(), false);
    llama_context_ptr on  = new_context(model.get(), true);
    const std::vector<float> ref = get_logits(off.get());
    const std::vector<float> got = get_logits(on.get());
    check(ref.size() == got.size() && memcmp(ref.data(), got.data(), ref.size()*sizeof(float)) == 0,
        stem + ": logits with the lookahead are bitwise the logits without it");
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

    // the router of the shared MoE FFN, the same with a hybrid memory, and a router that scales its input itself
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

    for (const char * arch : { "qwen35moe", "qwen4exp" }) {
        try {
            test_extra(dir_models, arch, devs, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s extras: %s\n", arch, e.what());
            check(false, std::string(arch) + ": the extras run");
        }
    }
    try {
        test_extra_width(dir_models, devs, check);
    } catch (const std::exception & e) {
        fprintf(stderr, "extras width: %s\n", e.what());
        check(false, "the width checks run");
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
