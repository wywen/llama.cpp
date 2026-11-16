// checks a context whose caller stores the K/V rows of the QSA layers of qwen4exp (llama_context_params.kv_rows_caller):
//   - the context gives those layers metadata-only K/V tensors, in no buffer and holding no data, which its graphs store
//     into and attend over as they do over a cache of the context's own
//   - a caller whose graph rewrite backs those tensors with tensors of its own computes logits bitwise identical to the
//     stock context's: a prompt in one ubatch and in several ubatches, then single-token decode steps, two sequences in
//     one KV stream, and a quantized K/V cache
//   - shifting positions and saving the state are refused
//
// the models are written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// the caller's graph rewrite replaces the graph's leafs
#include "../ggml/src/ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

static const char * MODEL = "qwen4exp-moe.gguf";

static constexpr int32_t TOP_K = 8; // two blocks per token, overriding the generated model's select-all top-k

static constexpr size_t N_TOKENS = 40; // a prompt of N_BATCH tokens, then single-token decode steps
static constexpr size_t N_BATCH  = 24;

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-110s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

// an attention K/V cache tensor of a context, cache_k_l<il> or cache_v_l<il>
static bool is_cache(const ggml_tensor * t) {
    const char * name = ggml_get_name(t);
    if (strncmp(name, "cache_k_l", 9) != 0 && strncmp(name, "cache_v_l", 9) != 0) {
        return false;
    }
    char * end = nullptr;
    strtol(name + 9, &end, 10);
    return end != name + 9 && *end == 0;
}

// plays the caller: its graph rewrite backs each metadata-only K/V tensor with a tensor of its own of the same shape and
//     type, allocated in `buft` and zeroed, which takes the metadata-only tensor's place among the graph's leafs, in its
//     readers and as the base of its views
struct row_backer {
    ggml_backend_buffer_type_t buft = nullptr;

    struct backing {
        ggml_context_ptr       ctx;
        ggml_backend_buffer_ptr buf;
        ggml_tensor *          t = nullptr;
    };
    std::map<const ggml_tensor *, backing> backed; // by metadata-only tensor

    int32_t n_meta     = 0; // the metadata-only tensors seen
    int32_t n_not_meta = 0; // the cache tensors seen with a buffer or data

    ggml_tensor * backing_of(ggml_tensor * meta) {
        backing & b = backed[meta];
        if (!b.t) {
            ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            b.ctx.reset(ggml_init(params));
            b.t = ggml_dup_tensor(b.ctx.get(), meta);
            ggml_set_name(b.t, ggml_get_name(meta));
            b.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(b.ctx.get(), buft));
            ggml_backend_buffer_clear(b.buf.get(), 0);
        }
        return b.t;
    }

    void clear() {
        for (auto & [_, b] : backed) {
            ggml_backend_buffer_clear(b.buf.get(), 0);
        }
    }

    bool rewrite(ggml_cgraph * gf) {
        std::map<ggml_tensor *, ggml_tensor *> to;
        for (int i = 0; i < gf->n_leafs; ++i) {
            ggml_tensor * leaf = gf->leafs[i];
            if (!is_cache(leaf)) {
                continue;
            }
            if (leaf->buffer != nullptr || leaf->data != nullptr) {
                n_not_meta++;
                continue;
            }
            n_meta++;
            to[leaf] = backing_of(leaf);
            gf->leafs[i] = to[leaf];
            ggml_hash_insert(&gf->visited_hash_set, to[leaf]);
        }
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const auto it = node->src[s] ? to.find(node->src[s]) : to.end();
                if (it != to.end()) {
                    node->src[s] = it->second;
                }
            }
            const auto it = node->view_src ? to.find(node->view_src) : to.end();
            if (it != to.end()) {
                node->view_src = it->second;
            }
        }
        return true;
    }
};

static bool rewrite_caches(ggml_backend_sched_t, ggml_cgraph * gf, ggml_backend_sched_rewrite_purpose, void * ud) {
    return ((row_backer *) ud)->rewrite(gf);
}

static bool install_rewrite_caches(ggml_backend_sched_t sched, llama_context *, void * ud) {
    ggml_backend_sched_set_graph_rewrite_callback(sched, rewrite_caches, ud);
    return true;
}

static llama_model_ptr load_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs) {
    std::vector<llama_model_kv_override> overrides(2);
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(overrides[0].key, sizeof(overrides[0].key), "qwen4exp.attention.indexer.top_k");
    overrides[0].val_i64 = TOP_K;

    llama_model_params mparams = llama_model_default_params();
    mparams.kv_overrides = overrides.data();
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

// backer plays the caller, NULL for the stock context
static llama_context_ptr new_context(llama_model * model, row_backer * backer, context_shape shape) {
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
    if (backer) {
        cparams.kv_rows_caller             = true;
        cparams.cb_sched_new               = install_rewrite_caches;
        cparams.cb_sched_new_user_data     = backer;
    }
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

// the logits of the sequences, decoded together: their first N_BATCH tokens in one batch, then one token of each per
//     batch, from an empty memory; empty if a decode fails
static std::vector<float> get_logits(llama_context * lctx, row_backer * backer, const std::vector<std::vector<llama_token>> & seqs) {
    llama_memory_clear(llama_get_memory(lctx), true);
    if (backer) {
        backer->clear();
    }
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

    const std::vector<llama_token> a = random_tokens();
    return {
        { "a prompt in one ubatch and decode steps",    { 64, 1, GGML_TYPE_F16 },  { a } },
        { "a prompt in five ubatches and decode steps", { 5,  1, GGML_TYPE_F16 },  { a } },
        { "two sequences in one KV stream",             { 64, 2, GGML_TYPE_F16 },  { a, random_tokens() } },
        { "a Q8_0 K/V cache",                           { 5,  1, GGML_TYPE_Q8_0 }, { a } },
    };
}

static void test_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string() + (devs.empty() ? " (CPU)" : " (GPU)");
    printf("%s\n", stem.c_str());
    llama_model_ptr model = load_model(path, devs);
    const ggml_backend_buffer_type_t buft = devs.empty() ? ggml_backend_cpu_buffer_type() : ggml_backend_dev_buffer_type(devs.front());

    const std::vector<scenario> scs = make_scenarios(model.get());
    for (const scenario & sc : scs) {
        const std::string what = stem + ": " + sc.what;

        llama_context_ptr lctx_stock = new_context(model.get(), nullptr, sc.shape);
        const std::vector<float> ref = get_logits(lctx_stock.get(), nullptr, sc.seqs);

        row_backer backer;
        backer.buft = buft;
        llama_context_ptr lctx = new_context(model.get(), &backer, sc.shape);
        const std::vector<float> got = get_logits(lctx.get(), &backer, sc.seqs);

        check(backer.n_meta > 0 && backer.n_not_meta == 0 && !backer.backed.empty(),
            what + ": the caller's layers have metadata-only K/V tensors (" + std::to_string(backer.backed.size()) + ")");
        check(!ref.empty() && got.size() == ref.size() && memcmp(got.data(), ref.data(), got.size()*sizeof(float)) == 0,
            what + ": logits backed by the caller's tensors are the stock context's, bitwise");

        if (&sc != &scs.front()) {
            continue;
        }

        llama_memory_t mem = llama_get_memory(lctx.get());
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        g_log.clear();
        llama_memory_seq_add(mem, 0, 0, -1, 4);
        check(!llama_memory_can_shift(mem) && llama_memory_seq_pos_max(mem, 0) == pos_max &&
            g_log.find("refusing the shift") != std::string::npos, stem + ": shifting positions is refused");

        g_log.clear();
        const size_t n_state     = llama_state_get_size(lctx.get());
        const bool   state_error = g_log.find("cannot hold them") != std::string::npos;
        g_log.clear();
        const size_t n_seq_state = llama_state_seq_get_size(lctx.get(), 0);
        check(n_state == 0 && state_error && n_seq_state == 0 && g_log.find("cannot hold them") != std::string::npos,
            stem + ": saving the state is refused");
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
