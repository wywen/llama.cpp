// checks that the QSA layers of qwen4exp whose K/V rows the caller stores (llama_context_params.kv_rows) compute logits
//     close to the stock path, which stores the rows in the K/V cache and reads its whole view masked to the
//     selection, when the test keeps the rows in a host array per layer:
//   - a prompt in one ubatch and in several ubatches, then single-token decode steps
//   - a sequence with an EOS in the middle, which cuts the n-gram context of qwen4exp
//   - two sequences in one unified KV stream
//   - quantized K/V caches
//   - a top-k that selects a few blocks, and the stock top-k that selects them all
// and that:
//   - the context has no K/V cache tensors for the QSA layers: no op reads or writes one, while the stock graph does
//   - after each graph the caller gets the new K then V rows of each QSA layer for exactly the ubatch's cells, and the
//     rows are byte-identical to what the stock path stores in its cache
//   - at each cut the caller is asked only for cells that it stored before and that are not the ubatch's, ascending,
//     at most n_sel per token, and fewer than the cells in use once the context outgrows the selection
//   - corrupted rows change the logits, and a failing caller fails the decode, when reading or writing
//   - shifting positions and saving the state are refused, and so is a context with more than one KV stream or
//     without flash attention
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
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

static const char * MODEL = "qwen4exp-moe.gguf";

static constexpr int32_t KPOOL = 4; // the generated model's block size
static constexpr int32_t TOP_K = 8; // two blocks per token, overriding the generated model's select-all top-k

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

// the attention K/V caches of a context, cache_k_l<il> and cache_v_l<il>
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

// what the graphs of a context reference, seen before each graph computes
struct graph_scan {
    std::map<int32_t, const ggml_tensor *> k, v; // the attention K/V caches, by layer

    int32_t n_stores = 0;               // the ops that write a cache
    std::vector<std::string> readers;   // the other ops that reference a cache, "op 'name' reads 'cache'"
    bool    rotated = false;            // a graph reads a K rotation matrix

    std::string backend_attn; // the backends of the attention and of the cut in the last graph
    std::string backend_cut;

    void clear() {
        n_stores = 0;
        readers.clear();
    }

    void scan(ggml_backend_sched_t sched, ggml_cgraph * gf) {
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
};

static bool scan_graph(ggml_backend_sched_t sched, ggml_cgraph * gf, void * ud) {
    ((graph_scan *) ud)->scan(sched, gf);
    return true;
}

// one call of the row callback
struct row_call {
    int32_t              il;
    bool                 v;
    bool                 write;
    std::vector<int32_t> cells;
};

// the caller's K/V store: one host array of rows per layer and component, filled by the writes after each graph
struct row_server {
    struct store {
        size_t               row_size = 0;
        std::vector<uint8_t> rows;   // [n_cells*row_size]
        std::vector<bool>    stored; // [n_cells]
    };
    std::map<std::pair<int32_t, bool>, store> stores; // by (layer, v)

    std::vector<row_call> calls;
    bool fail_read  = false;
    bool fail_write = false;
    bool corrupt    = false; // zero the V rows that it serves
    bool bad_read   = false; // asked for a row that it never stored

    graph_scan scan;

    void reset() {
        stores.clear();
        calls.clear();
        bad_read = false;
    }

    bool rows(int32_t il, bool v, bool write, const int32_t * cells, int32_t n_cells, void * data, size_t row_size) {
        calls.push_back({ il, v, write, std::vector<int32_t>(cells, cells + n_cells) });
        if (write ? fail_write : fail_read) {
            return false;
        }
        store & st = stores[{ il, v }];
        if (st.row_size == 0) {
            st.row_size = row_size;
        }
        if (st.row_size != row_size) {
            return false;
        }
        for (int32_t i = 0; i < n_cells; ++i) {
            const size_t c = (size_t) cells[i];
            char * row = (char *) data + i*row_size;
            if (write) {
                if (st.stored.size() <= c) {
                    st.stored.resize(c + 1, false);
                    st.rows.resize((c + 1)*row_size);
                }
                memcpy(st.rows.data() + c*row_size, row, row_size);
                st.stored[c] = true;
            } else {
                if (c >= st.stored.size() || !st.stored[c]) {
                    bad_read = true;
                    return false;
                }
                memcpy(row, st.rows.data() + c*row_size, row_size);
                if (corrupt && v) {
                    memset(row, 0, row_size);
                }
            }
        }
        return true;
    }
};

static bool serve_rows(int32_t il, bool v, bool write, const int32_t * cells, int32_t n_cells, void * rows, size_t row_size, void * ud) {
    return ((row_server *) ud)->rows(il, v, write, cells, n_cells, rows, row_size);
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

// server stores the rows, NULL for the stock path; scan sees the graphs
static llama_context_params context_params(row_server * server, context_shape shape, graph_scan * scan) {
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
    cparams.cb_graph_prepare           = scan_graph;
    cparams.cb_graph_prepare_user_data = scan;
    return cparams;
}

static llama_context_ptr new_context(llama_model * model, row_server * server, context_shape shape, graph_scan * scan) {
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, context_params(server, shape, scan)));
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

    res.push_back({ "a prompt in one ubatch and decode steps",      { 64, 1, GGML_TYPE_F16 },  { a } });
    res.push_back({ "a prompt in five ubatches and decode steps",   { 5,  1, GGML_TYPE_F16 },  { a } });
    res.push_back({ "an EOS in the prompt and in the decode steps", { 5,  1, GGML_TYPE_F16 },  { with_eos } });
    res.push_back({ "two sequences in one KV stream",               { 64, 2, GGML_TYPE_F16 },  { a, random_tokens() } });
    res.push_back({ "a Q8_0 K/V cache",                             { 5,  1, GGML_TYPE_Q8_0 }, { a } });
    res.push_back({ "a Q4_0 K/V cache",                             { 64, 1, GGML_TYPE_Q4_0 }, { a } });
    return res;
}

// the token counts of the graphs of a decode, see get_logits
static std::vector<size_t> graph_sizes(const scenario & sc) {
    const size_t n_seqs  = sc.seqs.size();
    const size_t per_seq = sc.shape.n_ubatch/n_seqs;
    std::vector<size_t> res;
    for (size_t beg = 0; beg < N_BATCH; beg += per_seq) {
        res.push_back(n_seqs*std::min(per_seq, N_BATCH - beg));
    }
    res.resize(res.size() + N_TOKENS - N_BATCH, n_seqs);
    return res;
}

// checks the calls of one decode: per graph, reads at the cuts, then the writes of the new K then V rows of each QSA
//     layer for exactly the ubatch's cells; the cells read are ones written by earlier graphs and not the ubatch's,
//     ascending, at most n_sel per token
static void check_calls(const row_server & server, const scenario & sc, int32_t n_sel, int32_t n_layers,
        const std::string & label, checker & check) {
    const std::vector<size_t> sizes = graph_sizes(sc);

    bool writes_ok = true;
    bool reads_ok  = true;
    bool bound_ok  = true;
    bool sparse_ok = n_sel <= 0;

    std::set<int32_t> written;
    size_t c = 0;
    size_t g = 0;
    for (; writes_ok && g < sizes.size(); ++g) {
        // the reads at the cuts of the graph
        std::vector<const row_call *> reads;
        while (c < server.calls.size() && !server.calls[c].write) {
            reads.push_back(&server.calls[c++]);
        }
        // then 2*n_layers writes, K then V of each layer, for the ubatch's cells
        std::vector<int32_t> cells;
        for (int32_t l = 0; writes_ok && l < n_layers; ++l) {
            writes_ok = c + 1 < server.calls.size() && server.calls[c].write && server.calls[c + 1].write &&
                !server.calls[c].v && server.calls[c + 1].v && server.calls[c].il == server.calls[c + 1].il &&
                server.calls[c].cells == server.calls[c + 1].cells && server.calls[c].cells.size() == sizes[g] &&
                (l == 0 || server.calls[c].cells == cells);
            if (writes_ok) {
                cells = server.calls[c].cells;
            }
            c += 2;
        }
        const std::set<int32_t> own(cells.begin(), cells.end());
        writes_ok = writes_ok && own.size() == cells.size();

        for (size_t r = 0; r < reads.size(); r += 2) {
            const row_call & rk = *reads[r];
            const bool pair = r + 1 < reads.size() && !rk.v && reads[r + 1]->v && reads[r + 1]->il == rk.il && reads[r + 1]->cells == rk.cells;
            const bool cells_ok = !rk.cells.empty() && std::is_sorted(rk.cells.begin(), rk.cells.end()) &&
                std::adjacent_find(rk.cells.begin(), rk.cells.end()) == rk.cells.end() &&
                std::all_of(rk.cells.begin(), rk.cells.end(), [&](int32_t x) { return written.count(x) && !own.count(x); });
            reads_ok  = reads_ok && pair && cells_ok;
            bound_ok  = bound_ok && (n_sel <= 0 || rk.cells.size() <= sizes[g]*n_sel);
            sparse_ok = sparse_ok || rk.cells.size() < written.size();
        }
        for (const int32_t x : own) {
            writes_ok = writes_ok && written.insert(x).second;
        }
    }
    writes_ok = writes_ok && c == server.calls.size() && g == sizes.size();

    // a fresh memory fills its cells from the first
    const size_t n_used = N_TOKENS*sc.seqs.size();
    writes_ok = writes_ok && written.size() == n_used && *written.begin() == 0 && *written.rbegin() == (int32_t) n_used - 1;

    check(writes_ok, label + ": after each of the " + std::to_string(sizes.size()) + " graphs, the new K then V rows of each of the " +
        std::to_string(n_layers) + " QSA layers for exactly the ubatch's cells");
    check(reads_ok && !server.bad_read, label + ": the cuts ask for stored cells only, none of the ubatch's, ascending and distinct");
    if (n_sel > 0) {
        check(bound_ok && sparse_ok, label + ": at most n_sel = " + std::to_string(n_sel) +
            " cells per token, and some cuts read fewer rows than the cells stored");
    }
}

// whether the rows that the caller stores match, byte for byte, those of the stock cache
static bool same_rows(const row_server & server, const graph_scan & stock) {
    if (server.stores.empty()) {
        return false;
    }
    for (const auto & [key, st] : server.stores) {
        const auto & caches = key.second ? stock.v : stock.k;
        const auto it = caches.find(key.first);
        if (it == caches.end() || it->second->nb[1] != st.row_size) {
            return false;
        }
        std::vector<uint8_t> row(st.row_size);
        for (size_t c = 0; c < st.stored.size(); ++c) {
            if (!st.stored[c]) {
                return false;
            }
            ggml_backend_tensor_get(it->second, row.data(), c*st.row_size, st.row_size);
            if (memcmp(row.data(), st.rows.data() + c*st.row_size, st.row_size) != 0) {
                return false;
            }
        }
    }
    return true;
}

static void test_top_k(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, int32_t top_k,
        const std::string & stem, checker & check) {
    llama_model_ptr model = load_model(path, devs, top_k);
    const std::string label = stem + (top_k > 0 ? ", top-k " + std::to_string(top_k) : ", stock top-k");
    printf("%s\n", label.c_str());

    const std::vector<scenario> scs = make_scenarios(model.get());

    for (const scenario & sc : scs) {
        const std::string what = label + ": " + sc.what;

        // the stock path, which keeps its context to compare the cache rows
        graph_scan stock;
        llama_context_ptr lctx_stock = new_context(model.get(), nullptr, sc.shape, &stock);
        const std::vector<float> ref = get_logits(lctx_stock.get(), sc.seqs);

        row_server server;
        llama_context_ptr lctx = new_context(model.get(), &server, sc.shape, &server.scan);
        server.reset();
        server.scan.clear();
        const std::vector<float> got = get_logits(lctx.get(), sc.seqs);
        const double e = nmse(got, ref);
        const bool bitwise = !got.empty() && got.size() == ref.size() && memcmp(got.data(), ref.data(), got.size()*sizeof(float)) == 0;

        // the QSA layers are the attention layers, whose caches the stock graphs reference
        const int32_t n_layers = (int32_t) stock.k.size();
        check(!ref.empty() && e <= NMSE_MAX, what + ": logits near the stock path (NMSE " + fmt_nmse(e) +
            (bitwise ? ", bitwise" : "") + ")");
        check(n_layers > 0 && server.scan.k.empty() && server.scan.v.empty() && server.scan.n_stores == 0 && server.scan.readers.empty() &&
            stock.n_stores > 0 && !stock.readers.empty(),
            what + ": no op reads or writes a QSA K/V cache, while the stock graph stores and reads them (" +
            (stock.readers.empty() ? std::string("none") : stock.readers.front()) + ")");
        check(server.scan.backend_cut == "CPU" && (devs.empty() == (server.scan.backend_attn == "CPU")),
            what + ": the cut runs on " + server.scan.backend_cut + ", the attention on " + server.scan.backend_attn);
        if (ggml_is_quantized(sc.shape.type_kv)) {
            check(server.scan.rotated, what + ": the quantized rows are rotated");
        }

        const int32_t n_sel = top_k > 0 ? KPOOL*(top_k/KPOOL) + KPOOL - 1 : 0;
        check_calls(server, sc, n_sel, n_layers, what, check);
        check(same_rows(server, stock), what + ": the rows that the caller stores are byte-identical to the stock cache's");

        if (&sc != &scs.front()) {
            continue;
        }

        // the logits depend on the served bytes
        server.reset();
        server.corrupt = true;
        const double ec = nmse(get_logits(lctx.get(), sc.seqs), ref);
        server.corrupt = false;
        check(ec > 1e3*NMSE_MAX, label + ": zeroed V rows change the logits (NMSE " + fmt_nmse(ec) + ")");

        // a failing caller fails the decode
        for (const bool write : { false, true }) {
            server.reset();
            g_log.clear();
            (write ? server.fail_write : server.fail_read) = true;
            const std::vector<float> failed = get_logits(lctx.get(), sc.seqs);
            server.fail_read = server.fail_write = false;
            check(failed.empty() && g_log.find(write ? "failed to write" : "failed to read") != std::string::npos,
                label + ": a caller that fails to " + (write ? "write" : "read") + " rows fails the decode");
        }

        // positions cannot be shifted, nor the state saved
        server.reset();
        get_logits(lctx.get(), sc.seqs);
        llama_memory_t mem = llama_get_memory(lctx.get());
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        g_log.clear();
        llama_memory_seq_add(mem, 0, 0, -1, 4);
        check(!llama_memory_can_shift(mem) && llama_memory_seq_pos_max(mem, 0) == pos_max &&
            g_log.find("refusing the shift") != std::string::npos, label + ": shifting positions is refused");

        g_log.clear();
        const size_t n_state     = llama_state_get_size(lctx.get());
        const bool   state_error = g_log.find("cannot hold them") != std::string::npos;
        g_log.clear();
        const size_t n_seq_state = llama_state_seq_get_size(lctx.get(), 0);
        check(n_state == 0 && state_error && n_seq_state == 0 && g_log.find("cannot hold them") != std::string::npos,
            label + ": saving the state is refused");
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
        llama_context_params cparams = context_params(&server, { 64, 2, GGML_TYPE_F16 }, &server.scan);
        cparams.kv_unified = false;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(model.get(), cparams));
        check(!lctx && g_log.find("one KV stream") != std::string::npos, stem + ": a context with two KV streams is refused");
    }
    {
        llama_context_params cparams = context_params(&server, { 64, 1, GGML_TYPE_F16 }, &server.scan);
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
