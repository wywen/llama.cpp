// checks that the QSA layers of qwen4exp reading their K/V rows from the caller (llama_context_params.kv_rows) compute
//     logits close to the stock path, which reads the whole K/V cache view masked to the selection, when the test
//     serves the rows by reading the context's own K/V cache tensors:
//   - a prompt in one ubatch and in several ubatches, then single-token decode steps
//   - a sequence with an EOS in the middle, which cuts the n-gram context of qwen4exp
//   - two sequences in one unified KV stream
//   - quantized K/V caches
//   - a top-k that selects a few blocks, and the stock top-k that selects them all
// and that no op of the graph reads the K/V cache of a QSA layer except the stores, while the stock graph does; that
//     the caller is asked for the K rows, then the V rows of each QSA layer, for ascending cells inside the cache,
//     at most n_sel per token and fewer than the cells in use once the context outgrows the selection; that
//     corrupted rows change the logits; that a failing caller fails the decode; and that a context with more than one
//     KV stream or without flash attention is refused
//
// the models are written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

static const char * MODEL = "qwen4exp-moe.gguf";

static constexpr int32_t KPOOL  = 4; // the generated model's block size
static constexpr int32_t TOP_K  = 8; // two blocks per token, overriding the generated model's select-all top-k

static constexpr size_t N_TOKENS = 40; // a prompt of N_BATCH tokens, then single-token decode steps
static constexpr size_t N_BATCH  = 24;

// the logits of a scenario are near-neutral to the stock path when their NMSE is below this
static constexpr double NMSE_MAX = 1e-6;

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-118s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

// the attention K/V caches of a context, cache_k_l<il> and cache_v_l<il>, as its graphs reference them
static bool cache_layer(const ggml_tensor * t, char kind, int32_t * il) {
    const char * name = ggml_get_name(t);
    const char prefix[] = { 'c', 'a', 'c', 'h', 'e', '_', kind, '_', 'l', 0 };
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return false;
    }
    char * end = nullptr;
    const long v = strtol(name + sizeof(prefix) - 1, &end, 10);
    if (end == name + sizeof(prefix) - 1 || *end != 0) {
        return false;
    }
    *il = (int32_t) v;
    return true;
}

static const ggml_tensor * view_root(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool is_view_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// one call of the row callback
struct row_call {
    int32_t              il;
    bool                 v;
    std::vector<int32_t> cells;
    size_t               row_size;
};

// serves the K/V rows from the context's own caches, which it finds in each graph before the graph computes; also
//     records which ops of the graphs read the caches
struct row_server {
    std::map<int32_t, const ggml_tensor *> k, v; // by layer

    std::vector<row_call> calls;
    bool fail    = false;
    bool corrupt = false; // zero the V rows

    int32_t n_graphs = 0;
    int32_t n_stores = 0;
    bool    rotated  = false; // a graph reads a K/V rotation matrix

    std::string backend_attn; // the backends of the attention and of the cut in the last graph
    std::string backend_cut;
    std::vector<std::string> readers; // the ops other than the stores that read a cache, "op 'name' reads 'cache'"

    void scan(ggml_backend_sched_t sched, ggml_cgraph * gf) {
        n_graphs++;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            const bool is_cut = strncmp(node->name, "qsa_cut-", 8) == 0;
            if (node->op == GGML_OP_FLASH_ATTN_EXT || is_cut) {
                ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, node);
                (is_cut ? backend_cut : backend_attn) = backend ? ggml_backend_name(backend) : "none";
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = node->src[s];
                if (src == nullptr) {
                    continue;
                }
                const ggml_tensor * root = view_root(src);
                rotated = rotated || strncmp(ggml_get_name(root), "cache_k_rot_l", 13) == 0;
                int32_t il = -1;
                const bool is_k = cache_layer(root, 'k', &il);
                if (!is_k && !cache_layer(root, 'v', &il)) {
                    continue;
                }
                (is_k ? k : v)[il] = root;
                // a store writes its src 2 (and is a view of it); a view of a cache is checked where it is read
                if (node->op == GGML_OP_SET_ROWS && s == 2) {
                    n_stores++;
                } else if (!is_view_op(node->op)) {
                    readers.push_back(std::string(ggml_op_name(node->op)) + " '" + node->name + "' reads '" + root->name + "'");
                }
            }
        }
    }

    bool read(int32_t il, bool is_v, const int32_t * cells, int32_t n_cells, void * dst, size_t row_size) {
        calls.push_back({ il, is_v, std::vector<int32_t>(cells, cells + n_cells), row_size });
        if (fail) {
            return false;
        }
        auto & caches = is_v ? v : k;
        const auto it = caches.find(il);
        if (it == caches.end() || it->second->nb[1] != row_size) {
            return false;
        }
        const ggml_tensor * cache = it->second;
        for (int32_t i = 0; i < n_cells; ++i) {
            if (cells[i] < 0 || cells[i] >= cache->ne[1]) {
                return false;
            }
            ggml_backend_tensor_get(cache, (char *) dst + i*row_size, cells[i]*row_size, row_size);
        }
        if (corrupt && is_v) {
            memset(dst, 0, n_cells*row_size);
        }
        return true;
    }
};

static bool serve_rows(int32_t il, bool v, const int32_t * cells, int32_t n_cells, void * dst, size_t row_size, void * ud) {
    return ((row_server *) ud)->read(il, v, cells, n_cells, dst, row_size);
}

static bool scan_graph(ggml_backend_sched_t sched, ggml_cgraph * gf, void * ud) {
    ((row_server *) ud)->scan(sched, gf);
    return true;
}

static llama_model_ptr load_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, int32_t top_k) {
    std::vector<llama_model_kv_override> overrides;
    if (top_k > 0) {
        llama_model_kv_override o = {};
        o.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        snprintf(o.key, sizeof(o.key), "qwen4exp.attention.indexer.top_k");
        o.val_i64 = top_k;
        overrides.push_back(o);
        overrides.push_back({});
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.kv_overrides = overrides.empty() ? nullptr : overrides.data();
    std::vector<ggml_backend_dev_t> dev_list = devs;
    if (dev_list.empty()) {
        mparams.n_gpu_layers = 0;
    } else {
        dev_list.push_back(nullptr);
        mparams.devices = dev_list.data();
    }
    g_log.clear();
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        throw std::runtime_error("failed to load " + path + ":\n" + g_log);
    }
    return model;
}

// the context shapes and cache types of the scenarios
struct context_shape {
    uint32_t  n_ubatch;
    uint32_t  n_seq_max;
    ggml_type type_kv;
};

// server serves the rows, NULL for the stock path; scan sees the graphs (server by default)
static llama_context_params context_params(row_server * server, context_shape shape, row_server * scan = nullptr) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = shape.n_ubatch;
    cparams.n_seq_max       = shape.n_seq_max;
    cparams.kv_unified      = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k          = shape.type_kv;
    cparams.type_v          = shape.type_kv;
    if (server) {
        cparams.kv_rows           = serve_rows;
        cparams.kv_rows_user_data = server;
    }
    if (scan || server) {
        cparams.cb_graph_prepare           = scan_graph;
        cparams.cb_graph_prepare_user_data = scan ? scan : server;
    }
    return cparams;
}

static llama_context_ptr new_context(llama_model * model, row_server * server, context_shape shape) {
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, context_params(server, shape)));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

// the logits of the sequences, decoded together: their first N_BATCH tokens in one batch, then one token of each per
//     batch, from an empty memory; empty if a decode fails
static std::vector<float> get_logits(llama_context * lctx, const std::vector<std::vector<llama_token>> & seqs) {
    llama_memory_clear(llama_get_memory(lctx), true);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));

    std::vector<float> ret;
    for (size_t beg = 0; beg < N_TOKENS; beg = beg == 0 ? N_BATCH : beg + 1) {
        const size_t end = beg == 0 ? N_BATCH : beg + 1;
        common_batch batch(lctx);
        for (size_t s = 0; s < seqs.size(); ++s) {
            for (size_t i = beg; i < end; i++) {
                batch.add(seqs[s][i], (llama_pos) i, (llama_seq_id) s, true);
            }
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            return {};
        }
        for (size_t i = 0; i < seqs.size()*(end - beg); i++) {
            const float * l = llama_get_logits_ith(lctx, (int32_t) i);
            ret.insert(ret.end(), l, l + n_vocab);
        }
    }
    return ret;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    if (a.empty() || a.size() != ref.size()) {
        return INFINITY;
    }
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - ref[i];
        num += d*d;
        den += (double) ref[i]*ref[i];
    }
    return den > 0.0 ? num/den : INFINITY;
}

static std::string fmt_nmse(double v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2e", v);
    return buf;
}

struct scenario {
    std::string                           what;
    context_shape                         shape;
    std::vector<std::vector<llama_token>> seqs;
};

static std::vector<scenario> make_scenarios(const llama_model * model) {
    const uint32_t    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const llama_token eos     = 0; // the generated model's n-gram EOS
    std::mt19937 gen(42);
    auto random_tokens = [&]() {
        std::vector<llama_token> tokens;
        while (tokens.size() < N_TOKENS) {
            const llama_token t = (llama_token) (gen() % n_vocab);
            if (t != eos) {
                tokens.push_back(t);
            }
        }
        return tokens;
    };

    std::vector<scenario> res;
    const std::vector<llama_token> a = random_tokens();
    std::vector<llama_token> with_eos = a;
    with_eos[9]  = eos;
    with_eos[30] = eos;

    res.push_back({ "a prompt in one ubatch and decode steps",        { 64, 1, GGML_TYPE_F16 },  { a } });
    res.push_back({ "a prompt in five ubatches and decode steps",     { 5,  1, GGML_TYPE_F16 },  { a } });
    res.push_back({ "an EOS in the prompt and in the decode steps",   { 5,  1, GGML_TYPE_F16 },  { with_eos } });
    res.push_back({ "two sequences in one KV stream",                 { 64, 2, GGML_TYPE_F16 },  { a, random_tokens() } });
    res.push_back({ "a Q8_0 K/V cache",                               { 5,  1, GGML_TYPE_Q8_0 }, { a } });
    res.push_back({ "a Q4_0 K/V cache",                               { 64, 1, GGML_TYPE_Q4_0 }, { a } });
    return res;
}

// checks the calls of one decode of a single sequence: per graph, the K rows then the V rows of each QSA layer with
//     the same cells, ascending and inside the cells in use, at most n_sel per token; returns the calls of each layer
static void check_calls(const row_server & server, const scenario & sc, int32_t n_sel, int32_t n_layers,
        const std::string & label, checker & check) {
    bool order_ok  = true;
    bool cells_ok  = true;
    bool bound_ok  = true;
    bool sparse_ok = false;

    // the graphs of the decode: the prompt's ubatches, then the decode steps
    std::vector<std::pair<size_t, size_t>> graphs; // [first token, n_tokens]
    const size_t n_seqs = sc.seqs.size();
    for (size_t beg = 0; beg < N_BATCH; beg += sc.shape.n_ubatch) {
        graphs.push_back({ beg, std::min<size_t>(sc.shape.n_ubatch/n_seqs, N_BATCH - beg) });
    }
    if (n_seqs == 1) {
        for (size_t i = N_BATCH; i < N_TOKENS; ++i) {
            graphs.push_back({ i, 1 });
        }
    }

    const size_t want_calls = n_seqs == 1 ? graphs.size()*2*n_layers : server.calls.size();
    order_ok = server.calls.size() == want_calls && want_calls % (2*n_layers) == 0;

    for (size_t c = 0; order_ok && c + 1 < server.calls.size(); c += 2) {
        const row_call & ck = server.calls[c];
        const row_call & cv = server.calls[c + 1];
        order_ok = !ck.v && cv.v && ck.il == cv.il && ck.cells == cv.cells;

        const size_t g = c/(2*n_layers);
        const size_t n_used   = n_seqs == 1 ? graphs[g].first + graphs[g].second : N_TOKENS*n_seqs;
        const size_t n_tokens = n_seqs == 1 ? graphs[g].second : sc.shape.n_ubatch;

        cells_ok = cells_ok && !ck.cells.empty() && std::is_sorted(ck.cells.begin(), ck.cells.end()) &&
            std::adjacent_find(ck.cells.begin(), ck.cells.end()) == ck.cells.end() &&
            ck.cells.front() >= 0 && (size_t) ck.cells.back() < n_used;
        bound_ok = bound_ok && ck.cells.size() <= n_tokens*n_sel;
        sparse_ok = sparse_ok || ck.cells.size() < n_used;
    }
    check(order_ok, label + ": per graph, the K rows then the V rows of each of the " + std::to_string(n_layers) +
        " QSA layers, for the same cells");
    check(cells_ok && bound_ok, label + ": the cells are ascending, distinct, in use, and at most n_sel = " +
        std::to_string(n_sel) + " per token");
    check(sparse_ok, label + ": some graphs read fewer rows than the cells in use");
}

static void test_top_k(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, int32_t top_k,
        const std::string & stem, checker & check) {
    llama_model_ptr model = load_model(path, devs, top_k);
    const std::string label = stem + (top_k > 0 ? ", top-k " + std::to_string(top_k) : ", stock top-k");
    printf("%s\n", label.c_str());

    const std::vector<scenario> scs = make_scenarios(model.get());

    for (const scenario & sc : scs) {
        // the stock path, scanned to show that the check sees its reads
        std::vector<float> ref;
        row_server stock;
        {
            g_log.clear();
            llama_context_ptr lctx(llama_init_from_model(model.get(), context_params(nullptr, sc.shape, &stock)));
            if (!lctx) {
                throw std::runtime_error("failed to create a context:\n" + g_log);
            }
            ref = get_logits(lctx.get(), sc.seqs);
        }

        row_server server;
        llama_context_ptr lctx = new_context(model.get(), &server, sc.shape);
        server.calls.clear();
        server.readers.clear();
        server.n_stores = 0;
        const std::vector<float> got = get_logits(lctx.get(), sc.seqs);
        const double e = nmse(got, ref);
        const bool bitwise = !got.empty() && got.size() == ref.size() && memcmp(got.data(), ref.data(), got.size()*sizeof(float)) == 0;

        // the QSA layers are the attention layers, whose caches the graphs reference
        const int32_t n_layers = (int32_t) server.k.size();
        check(!ref.empty() && e <= NMSE_MAX, label + ": " + sc.what + ": logits near the stock path (NMSE " + fmt_nmse(e) +
            (bitwise ? ", bitwise" : "") + (server.rotated ? ", rotated K" : "") + ")");
        check(server.backend_cut == "CPU" && (devs.empty() == (server.backend_attn == "CPU")),
            label + ": " + sc.what + ": the cut runs on " + server.backend_cut + ", the attention on " + server.backend_attn);
        if (ggml_is_quantized(sc.shape.type_kv)) {
            check(server.rotated, label + ": " + sc.what + ": the quantized cache holds rotated K rows");
        }
        check(n_layers > 0 && server.v.size() == server.k.size() && server.readers.empty() && server.n_stores > 0,
            label + ": " + sc.what + ": no op reads the K/V cache of a QSA layer but the " + std::to_string(server.n_stores) +
            " stores" + (server.readers.empty() ? "" : " (" + server.readers.front() + ")"));
        check(!stock.readers.empty(), label + ": " + sc.what + ": the stock graph reads the caches (" +
            (stock.readers.empty() ? std::string("none") : stock.readers.front()) + ")");

        const int32_t n_sel = top_k > 0 ? KPOOL*(top_k/KPOOL) + KPOOL - 1 : INT32_MAX/(int32_t) N_TOKENS;
        if (top_k > 0) {
            check_calls(server, sc, n_sel, n_layers, label + ": " + sc.what, check);
        }

        // the logits depend on the served bytes
        if (&sc == &scs.front()) {
            server.corrupt = true;
            const double ec = nmse(get_logits(lctx.get(), sc.seqs), ref);
            server.corrupt = false;
            check(ec > 1e3*NMSE_MAX, label + ": zeroed V rows change the logits (NMSE " + fmt_nmse(ec) + ")");

            // a failing caller fails the decode
            server.fail = true;
            llama_memory_clear(llama_get_memory(lctx.get()), true);
            common_batch batch(lctx.get());
            batch.add(sc.seqs[0][0], 0, 0, true);
            g_log.clear();
            const int32_t rc = llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            server.fail = false;
            check(rc != 0 && g_log.find("failed to read") != std::string::npos, label + ": a failing row reader fails the decode");
        }
    }
}

static void test_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string() + (devs.empty() ? " (CPU)" : " (GPU)");

    test_top_k(path, devs, TOP_K, stem, check);
    test_top_k(path, devs, 0, stem, check);

    printf("%s, refusals\n", stem.c_str());
    llama_model_ptr model = load_model(path, devs, TOP_K);
    row_server server;
    {
        llama_context_params cparams = context_params(&server, { 64, 2, GGML_TYPE_F16 });
        cparams.kv_unified = false;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        check(!lctx && g_log.find("one KV stream") != std::string::npos, stem + ": a context with two KV streams is refused");
    }
    {
        llama_context_params cparams = context_params(&server, { 64, 1, GGML_TYPE_F16 });
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        check(!lctx && g_log.find("flash attention") != std::string::npos, stem + ": a context without flash attention is refused");
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

    // the CPU (no devices), or with --all-devices every GPU device for the rest of the graph
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

    checker check;
    const std::string path = dir_models + "/" + MODEL;
    try {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("missing " + path);
        }
        test_model(path, devs, check);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s: %s\n", MODEL, e.what());
        check(false, std::string(MODEL) + ": runs");
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
