// checks the router lookahead of the MoE layers (llama_context_params.moe_router_ahead):
//   - with the parameter off, the graph has no "ffn_moe_logits_ahead" tensors
//   - with it on, every MoE layer with a following MoE layer has one, it reads the following layer's router weights,
//     and the graph without the lookahead nodes is the graph with the parameter off, node for node
//   - each lookahead equals the following layer's router logits recomputed on the host from the layer's router input
//   - the logits with the parameter on are bitwise identical to the logits with it off
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
static const char * LOGITS_PREFIX = "ffn_moe_logits-";

// the layer of a tensor named <prefix><il>, or -1
static int layer_of(const ggml_tensor * t, const char * prefix) {
    const size_t n = strlen(prefix);
    return strncmp(ggml_get_name(t), prefix, n) == 0 ? atoi(ggml_get_name(t) + n) : -1;
}

static llama_model_ptr load_model(const std::string & path, std::vector<ggml_backend_dev_t> devs) {
    llama_model_params mparams = llama_model_default_params();
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

static llama_context_ptr new_context(llama_model * model, bool ahead, ggml_backend_sched_eval_callback cb_eval = nullptr, void * ud = nullptr) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = 0;
    cparams.n_threads         = 4;
    cparams.n_threads_batch   = 4;
    cparams.n_ubatch          = 64;
    cparams.moe_router_ahead  = ahead;
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = ud;
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

static constexpr size_t N_TOKENS = 16;
static constexpr size_t N_BATCH  = 12;

// the logits of N_BATCH tokens in one batch, then of the other tokens one at a time, from an empty memory
static std::vector<float> get_logits(llama_context * lctx, size_t n_batch = N_BATCH) {
    llama_memory_clear(llama_get_memory(lctx), true);

    const llama_model * model = llama_get_model(lctx);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> tokens = get_tokens(N_TOKENS, n_vocab, 42);

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
        if (!consumed.count(t) && layer_of(t, AHEAD_PREFIX) < 0) {
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

    auto name_of = [](const ggml_tensor * t) {
        const char * name = ggml_get_name(t);
        return strncmp(name, "node_", 5) == 0 || strncmp(name, "leaf_", 5) == 0 ? std::string() : std::string(name);
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

// the data of the nodes that a decode computes, by tensor; the scheduler computes one node at a time
struct capture {
    std::map<const ggml_tensor *, std::vector<float>> data;
    std::vector<const ggml_tensor *> ahead;  // the lookahead tensors, in graph order
    std::map<int, const ggml_tensor *> logits; // the router logits by layer
};

static bool capture_node(ggml_tensor * t, bool ask, void * ud) {
    if (ask) {
        return true;
    }
    auto * cap = (capture *) ud;
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        std::vector<float> & v = cap->data[t];
        v.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    }
    if (layer_of(t, AHEAD_PREFIX) >= 0) {
        cap->ahead.push_back(t);
    }
    const int il = layer_of(t, LOGITS_PREFIX);
    if (il >= 0) {
        cap->logits[il] = t;
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

static void test_values(llama_model * model, const std::string & stem, checker & check) {
    capture cap;
    llama_context_ptr lctx = new_context(model, true, capture_node, &cap);
    get_logits(lctx.get(), N_BATCH);

    // the batch decode computes the lookahead of every layer first, then each one-token decode
    bool all_ok = !cap.ahead.empty();
    double worst = 0.0;
    for (const ggml_tensor * a : cap.ahead) {
        const int il = layer_of(a, AHEAD_PREFIX);
        const ggml_tensor * logits = cap.logits.count(il) ? cap.logits.at(il) : nullptr;
        if (logits == nullptr || !cap.data.count(a)) {
            all_ok = false;
            continue;
        }

        // layer il's router input; an architecture that scales inside its router (gemma4) scales the shared
        // normalized input by each layer's scale, so the lookahead scales it by the following layer's
        const ggml_tensor * x = logits->src[1];
        std::vector<float> x_next;
        if (a->src[1] == x) {
            if (!cap.data.count(x)) {
                all_ok = false;
                continue;
            }
            x_next = cap.data.at(x);
        } else {
            const ggml_tensor * scaled = a->src[1];
            const ggml_tensor * norm   = scaled->src[0];
            const std::string scale = "blk." + std::to_string(llama_model_moe_router_ahead_layer(model, il)) + ".";
            if (scaled->op != GGML_OP_MUL || x->op != GGML_OP_MUL || x->src[0] != norm || !cap.data.count(norm) ||
                strncmp(ggml_get_name(scaled->src[1]), scale.c_str(), scale.size()) != 0) {
                fprintf(stderr, "    layer %d: the lookahead input is neither the router input nor its scaled norm\n", il);
                all_ok = false;
                continue;
            }
            const std::vector<float> s = weight_rows(scaled->src[1]);
            x_next = cap.data.at(norm);
            for (size_t i = 0; i < x_next.size(); ++i) {
                x_next[i] *= s[i % s.size()];
            }
        }

        const ggml_tensor * w = a->src[0];
        const std::vector<float> wf = weight_rows(w);
        const int64_t n_embd = w->ne[0];
        const int64_t n_expert = w->ne[1];
        const int64_t n_tokens = a->ne[1];
        std::vector<double> ref(n_expert*n_tokens);
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t e = 0; e < n_expert; ++e) {
                double sum = 0.0;
                for (int64_t k = 0; k < n_embd; ++k) {
                    sum += (double) wf[e*n_embd + k] * x_next[t*n_embd + k];
                }
                ref[t*n_expert + e] = sum;
            }
        }
        const double err = rel_error(cap.data.at(a), ref);
        worst = std::max(worst, err);
        all_ok &= err < 1e-2;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), " (worst %.2e)", worst);
    check(all_ok, stem + ": " + std::to_string(cap.ahead.size()) + " lookaheads equal the following router on the router input" + buf);
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

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
