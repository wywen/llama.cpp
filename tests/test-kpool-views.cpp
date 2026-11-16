// checks that the k-pool indexer of qwen4exp reads only the indexer cache cells in use, as attention's get_k does:
//   - after a prompt shorter than the cache, every gather from the indexer cache and the scatter of the pooled keys
//     span the n_kv cells in use, while the store of each token's row still addresses the whole cache
//   - with a stream per sequence, a ubatch of one sequence spans the n_kv cells of its stream
//   - a sequence copied into another stream re-pools its keys when it is next decoded, not in a ubatch of another
//     stream, and computes the logits bitwise of a context that decodes it right after the copy
//
// the model is written by test-llama-archs (-o <dir>)

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-100s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

static constexpr uint32_t N_CTX = 1024;
// the KV caches pad the cells in use to 256, so a short prompt is read through 256 cells of each stream
static constexpr int64_t N_KV = 256;

// the rows of the indexer cache that the get_rows and set_rows nodes of the computed graphs address
struct span_watch {
    int64_t n_cells = 0;            // rows of the indexer cache storage, all streams
    std::vector<int64_t> get_rows;  // rows that each gather spans
    std::vector<int64_t> set_rows;  // rows that each store or scatter spans
};

static bool watch_spans(ggml_tensor * t, bool ask, void * ud) {
    if (!ask || (t->op != GGML_OP_GET_ROWS && t->op != GGML_OP_SET_ROWS)) {
        return false;
    }
    // a gather reads its source; a scatter returns a view of the tensor that it writes
    const ggml_tensor * view = t->op == GGML_OP_GET_ROWS ? t->src[0] : t;
    const ggml_tensor * base = view->view_src ? view->view_src : view;
    if (strncmp(ggml_get_name(base), "cache_idx_k_l", 13) != 0) {
        return false;
    }
    auto * w = (span_watch *) ud;
    w->n_cells = base->ne[1]*base->ne[2];
    const int64_t rows = (int64_t) ((ggml_nbytes(view) + base->nb[1] - 1)/base->nb[1]);
    (t->op == GGML_OP_GET_ROWS ? w->get_rows : w->set_rows).push_back(rows);
    return false;
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

static llama_context_ptr new_context(llama_model * model, uint32_t n_seq_max, bool unified, span_watch * watch) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = N_CTX;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = 64;
    cparams.n_seq_max       = n_seq_max;
    cparams.kv_unified      = unified;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    if (watch) {
        cparams.cb_eval           = watch_spans;
        cparams.cb_eval_user_data = watch;
    }
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

static std::vector<llama_token> make_tokens(const llama_model * model, size_t n, uint32_t seed) {
    const uint32_t n_vocab = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> res;
    for (size_t i = 0; i < n; ++i) {
        res.push_back((llama_token) ((seed*7919u + i*104729u) % (n_vocab - 8) + 4));
    }
    return res;
}

// decodes tokens of one sequence from position pos0 and returns the logits of each
static std::vector<float> decode(llama_context * lctx, const std::vector<llama_token> & tokens, llama_pos pos0, llama_seq_id seq_id) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos0 + (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = (int32_t) tokens.size();
    const int32_t rc = llama_decode(lctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("decode failed");
    }

    std::vector<float> res;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * l = llama_get_logits_ith(lctx, (int32_t) i);
        res.insert(res.end(), l, l + n_vocab);
    }
    return res;
}

static bool all_equal(const std::vector<int64_t> & v, int64_t x) {
    for (int64_t e : v) {
        if (e != x) {
            return false;
        }
    }
    return !v.empty();
}

// the gathers span n_kv cells, and the set_rows nodes either store into the whole cache or scatter into n_kv cells
static void check_spans(const span_watch & w, const std::string & label, checker & check) {
    size_t n_scatter = 0;
    bool   ok        = true;
    for (int64_t rows : w.set_rows) {
        n_scatter += rows == N_KV ? 1 : 0;
        ok = ok && (rows == N_KV || rows == w.n_cells);
    }
    check(w.n_cells > N_KV && all_equal(w.get_rows, N_KV),
        label + ": " + std::to_string(w.get_rows.size()) + " gathers span " + std::to_string(N_KV) + " of " +
        std::to_string(w.n_cells) + " cells");
    check(ok && n_scatter > 0 && n_scatter < w.set_rows.size(),
        label + ": " + std::to_string(n_scatter) + " pooled-key scatters span " + std::to_string(N_KV) +
        " cells, the stores the whole cache");
}

static void test_device(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string dev = devs.empty() ? "CPU" : "GPU";
    llama_model_ptr model = load_model(path, devs);

    const std::vector<llama_token> prompt = make_tokens(model.get(), 40, 1);

    {
        span_watch w;
        llama_context_ptr lctx = new_context(model.get(), 1, true, &w);
        decode(lctx.get(), prompt, 0, 0);
        for (llama_pos i = 0; i < 6; ++i) {
            decode(lctx.get(), make_tokens(model.get(), 1, 100 + i), (llama_pos) prompt.size() + i, 0);
        }
        check_spans(w, dev + ", one sequence", check);
    }

    {
        span_watch w;
        llama_context_ptr lctx = new_context(model.get(), 2, false, &w);
        decode(lctx.get(), prompt, 0, 0);
        decode(lctx.get(), make_tokens(model.get(), 30, 2), 0, 1);
        for (llama_pos i = 0; i < 4; ++i) {
            decode(lctx.get(), make_tokens(model.get(), 1, 200 + i), 30 + i, 1);
            decode(lctx.get(), make_tokens(model.get(), 1, 300 + i), (llama_pos) prompt.size() + i, 0);
        }
        check_spans(w, dev + ", a stream per sequence", check);
    }

    // seq 1 is a copy of seq 0 in its own stream, which every pool of it must re-pool from
    const std::vector<llama_token> next = make_tokens(model.get(), 5, 3);
    std::vector<float> ref;
    {
        llama_context_ptr lctx = new_context(model.get(), 2, false, nullptr);
        decode(lctx.get(), prompt, 0, 0);
        llama_memory_seq_cp(llama_get_memory(lctx.get()), 0, 1, -1, -1);
        ref = decode(lctx.get(), next, (llama_pos) prompt.size(), 1);
    }
    {
        llama_context_ptr lctx = new_context(model.get(), 2, false, nullptr);
        decode(lctx.get(), prompt, 0, 0);
        llama_memory_seq_cp(llama_get_memory(lctx.get()), 0, 1, -1, -1);
        // ubatches of seq 0 alone cannot reach the stream of seq 1
        for (llama_pos i = 0; i < 3; ++i) {
            decode(lctx.get(), make_tokens(model.get(), 1, 400 + i), (llama_pos) prompt.size() + i, 0);
        }
        const std::vector<float> cur = decode(lctx.get(), next, (llama_pos) prompt.size(), 1);
        check(cur.size() == ref.size() && memcmp(cur.data(), ref.data(), ref.size()*sizeof(float)) == 0,
            dev + ", a copied sequence decoded after another stream computes the logits of one decoded right away");
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

    // the CPU, and with --all-devices every GPU device
    std::vector<std::vector<ggml_backend_dev_t>> runs = { {} };
    if (all_devices) {
        std::vector<ggml_backend_dev_t> gpus;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpus.push_back(dev);
            }
        }
        if (gpus.empty()) {
            fprintf(stderr, "--all-devices: no GPU device\n");
            return 1;
        }
        runs.push_back(gpus);
    }

    checker check;
    const std::string path = dir_models + "/qwen4exp-moe.gguf";
    printf("qwen4exp-moe\n");
    for (const auto & devs : runs) {
        try {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("missing " + path);
            }
            test_device(path, devs, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s\n", e.what());
            check(false, std::string(devs.empty() ? "CPU" : "GPU") + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
