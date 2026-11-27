// checks llama_context_params.cb_memory_buft with a hook that places the K/V of the sparse attention layers of qwen4exp
// (llama_model_kv_rows_layer) on a placeholder buffer type, whose buffers are size 0 and leave their tensors without data:
//   - the hook is asked for the K/V of every layer of both KV caches, by kind, layer and tag, and for each rotation
//     matrix; the memory places the tensors on the types it answers
//   - the placed K/V tensors and their stream views have no data; the rotation matrices of the placed layers stay on
//     the layer's buffer type, and cb_kv_rot_set still receives them
//   - a graph rewrite that backs the placed tensors with tensors of its own computes logits bitwise identical to the
//     stock context's: a prompt in one ubatch and in several, then single-token decode steps, two sequences in one KV
//     stream, a quantized K/V cache, and a sequence copied into a second stream, which skips the placed layers
//   - shifting positions and saving the state are refused; a memory built without allocation refuses neither
//   - the memory plan records the placed tensors in placeholder buffers of size 0, and the other tensors as without
//     the hook
//
// the models are written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// the caller's graph rewrite replaces the graph's leafs, and the placeholder is a buffer type of its own
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-ext.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
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

//
// the placeholder buffer type: one per buffer type it stands in for, on the same device, under a name of its own
//

struct placeholder {
    std::string              name;
    ggml_backend_buffer_type buft;
};

// the tensors handed to placeholder buffers since the last context was created, views included
static std::vector<ggml_tensor *> g_placed;

static const char * placeholder_name(ggml_backend_buffer_type_t buft) {
    return ((const placeholder *) buft->context)->name.c_str();
}

// a buffer with storage is refused: only alloc_buffer_n backs tensors
static ggml_backend_buffer_t placeholder_alloc_buffer(ggml_backend_buffer_type_t, size_t) {
    return nullptr;
}

static ggml_backend_buffer_t placeholder_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, 0);
    for (int i = 0; i < n_tensors; ++i) {
        tensors[i]->buffer = buf;
        g_placed.push_back(tensors[i]);
    }
    return buf;
}

static size_t placeholder_alignment(ggml_backend_buffer_type_t) {
    return 1;
}

static size_t placeholder_alloc_size_n(ggml_backend_buffer_type_t, ggml_tensor **, int) {
    return 0;
}

static bool is_placeholder(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == placeholder_name;
}

static ggml_backend_buffer_type_t placeholder_of(ggml_backend_buffer_type_t buft) {
    static std::map<ggml_backend_buffer_type_t, std::unique_ptr<placeholder>> types;
    std::unique_ptr<placeholder> & p = types[buft];
    if (!p) {
        p = std::make_unique<placeholder>();
        p->name = std::string(ggml_backend_buft_name(buft)) + "_Placeholder";
        p->buft = {
            /* .iface   = */ {
                /* .get_name         = */ placeholder_name,
                /* .alloc_buffer     = */ placeholder_alloc_buffer,
                /* .alloc_buffer_n   = */ placeholder_alloc_buffer_n,
                /* .get_alignment    = */ placeholder_alignment,
                /* .get_max_size     = */ nullptr,
                /* .get_alloc_size   = */ nullptr,
                /* .get_alloc_size_n = */ placeholder_alloc_size_n,
                /* .is_host          = */ nullptr,
            },
            /* .device  = */ ggml_backend_buft_get_device(buft),
            /* .context = */ p.get(),
        };
    }
    return &p->buft;
}

static ggml_backend_buffer_type_t buft_of(const ggml_tensor * t) {
    return t->buffer ? ggml_backend_buffer_get_type(t->buffer) : nullptr;
}

//
// the hook: the sparse attention layers' K/V of the attention cache (untagged) go on the placeholder of llama's type
//

struct asked_buft {
    llama_memory_plan_buffer_kind kind;
    int32_t                       il;
    std::string                   tag;
    ggml_backend_buffer_type_t    buft;   // the type llama would use
    ggml_backend_buffer_type_t    answer; // NULL keeps it
};

struct qsa_hook {
    const llama_model *     model = nullptr;
    std::vector<asked_buft> asked;

    // the first question for a layer's K/V (kv) or rotation matrices (!kv)
    const asked_buft * find(bool kv, int32_t il, const char * tag) const {
        for (const asked_buft & a : asked) {
            const bool a_kv = a.kind == LLAMA_MEMORY_PLAN_BUFFER_KV || a.kind == LLAMA_MEMORY_PLAN_BUFFER_KV_SWA;
            if ((kv ? a_kv : a.kind == LLAMA_MEMORY_PLAN_BUFFER_KV_ROT) && a.il == il && a.tag == tag) {
                return &a;
            }
        }
        return nullptr;
    }
};

static ggml_backend_buffer_type_t place_qsa(llama_memory_plan_buffer_kind kind, int32_t il, const char * tag,
        ggml_backend_buffer_type_t buft, void * ud) {
    auto * hook = (qsa_hook *) ud;
    const bool kv    = kind == LLAMA_MEMORY_PLAN_BUFFER_KV || kind == LLAMA_MEMORY_PLAN_BUFFER_KV_SWA;
    const bool place = kv && tag[0] == 0 && il >= 0 && llama_model_kv_rows_layer(*hook->model, (uint32_t) il);
    ggml_backend_buffer_type_t answer = place ? placeholder_of(buft) : nullptr;
    hook->asked.push_back({ kind, il, tag, buft, answer });
    return answer;
}

// the layer of an attention K/V cache tensor, cache_k_l<il> or cache_v_l<il>; -1 for any other tensor
static int32_t cache_layer(const char * name) {
    if (strncmp(name, "cache_k_l", 9) != 0 && strncmp(name, "cache_v_l", 9) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + 9, &end, 10);
    return end != name + 9 && *end == 0 ? (int32_t) il : -1;
}

// plays the caller: its graph rewrite backs each placed K/V tensor with a tensor of its own of the same shape and type,
//     allocated in `buft` and zeroed, which takes the placed tensor's place among the graph's leafs, in its readers
//     and as the base of its views
struct row_backer {
    ggml_backend_buffer_type_t buft = nullptr;

    struct backing {
        ggml_context_ptr        ctx;
        ggml_backend_buffer_ptr buf;
        ggml_tensor *           t = nullptr;
    };
    std::map<const ggml_tensor *, backing> backed; // by placed tensor

    int32_t n_placed   = 0; // the placed tensors seen: no data, in a placeholder buffer
    int32_t n_unplaced = 0; // the cache tensors seen without data outside a placeholder buffer
    int32_t n_data     = 0; // the cache tensors seen with data

    ggml_tensor * backing_of(ggml_tensor * placed) {
        backing & b = backed[placed];
        if (!b.t) {
            ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            b.ctx.reset(ggml_init(params));
            b.t = ggml_dup_tensor(b.ctx.get(), placed);
            ggml_set_name(b.t, ggml_get_name(placed));
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

    // the caller's half of a cross-stream sequence copy, llama copies the rows of the other layers
    void copy_stream(size_t src, size_t dst) {
        for (auto & [_, b] : backed) {
            const size_t n = b.t->nb[2];
            std::vector<uint8_t> rows(n);
            ggml_backend_tensor_get(b.t, rows.data(), src*n, n);
            ggml_backend_tensor_set(b.t, rows.data(), dst*n, n);
        }
    }

    bool rewrite(ggml_cgraph * gf) {
        std::map<ggml_tensor *, ggml_tensor *> to;
        for (int i = 0; i < gf->n_leafs; ++i) {
            ggml_tensor * leaf = gf->leafs[i];
            if (cache_layer(ggml_get_name(leaf)) < 0) {
                continue;
            }
            if (leaf->data != nullptr) {
                n_data++;
                continue;
            }
            if (!is_placeholder(buft_of(leaf))) {
                n_unplaced++;
                continue;
            }
            n_placed++;
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

static llama_model_ptr load_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, bool no_alloc) {
    std::vector<llama_model_kv_override> overrides(2);
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(overrides[0].key, sizeof(overrides[0].key), "qwen4exp.attention.indexer.top_k");
    overrides[0].val_i64 = TOP_K;

    llama_model_params mparams = llama_model_default_params();
    mparams.kv_overrides = overrides.data();
    if (no_alloc) {
        mparams.no_alloc  = true;
        mparams.load_mode = LLAMA_LOAD_MODE_NONE; // lazily mapped tensors cannot be loaded without allocation
    }
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
    bool      kv_unified;
    ggml_type type_kv;
};

// hook places the sparse attention layers' K/V (NULL: llama's own types), backer plays the caller (NULL: no rewrite)
// a context with the hook needs the backer: its graphs, the reserved ones included, read the placed tensors, which no
//     backend can run an op over (the CPU backend refuses the placeholder's buffers when it splits the graph)
static llama_context_params make_cparams(qsa_hook * hook, row_backer * backer, context_shape shape) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = shape.n_ubatch;
    cparams.n_seq_max       = shape.n_seq_max;
    cparams.kv_unified      = shape.kv_unified;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k          = shape.type_kv;
    cparams.type_v          = shape.type_kv;
    if (hook) {
        cparams.cb_memory_buft           = place_qsa;
        cparams.cb_memory_buft_user_data = hook;
    }
    if (backer) {
        cparams.cb_sched_new           = install_rewrite_caches;
        cparams.cb_sched_new_user_data = backer;
    }
    return cparams;
}

static llama_context_ptr new_context(llama_model * model, const llama_context_params & cparams) {
    g_log.clear();
    g_placed.clear();
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

static llama_memory_hybrid_idx * memory_of(llama_context * lctx) {
    auto * mem = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(lctx));
    if (mem == nullptr || mem->get_mem_attn() == nullptr) {
        throw std::runtime_error("the memory is not a hybrid memory with an indexer cache");
    }
    return mem;
}

// decodes one batch of n_tokens tokens, appending their logits to `out`; false if the decode fails
static bool decode(llama_context * lctx, common_batch & batch, size_t n_tokens, std::vector<float> & out) {
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        return false;
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));
    for (size_t i = 0; i < n_tokens; i++) {
        const float * l = llama_get_logits_ith(lctx, (int32_t) i);
        out.insert(out.end(), l, l + n_vocab);
    }
    return true;
}

// the logits of the sequences, decoded together: their first N_BATCH tokens in one batch, then one token of each per
//     batch, from an empty memory; with `copied` the second sequence is instead a copy of the first after its prompt,
//     which a shape that is not unified puts in a stream of its own; empty if a decode fails
static std::vector<float> get_logits(llama_context * lctx, row_backer * backer, const std::vector<std::vector<llama_token>> & seqs, bool copied) {
    llama_memory_t mem = llama_get_memory(lctx);
    llama_memory_clear(mem, true);
    if (backer) {
        backer->clear();
    }

    std::vector<float> ret;
    for (size_t beg = 0; beg < N_TOKENS; beg = beg == 0 ? N_BATCH : beg + 1) {
        const size_t end    = beg == 0 ? N_BATCH : beg + 1;
        const size_t n_seqs = copied && beg == 0 ? 1 : seqs.size();
        common_batch batch(lctx);
        for (size_t s = 0; s < n_seqs; ++s) {
            for (size_t i = beg; i < end; i++) {
                batch.add(seqs[s][i], (llama_pos) i, (llama_seq_id) s, true);
            }
        }
        if (!decode(lctx, batch, n_seqs*(end - beg), ret)) {
            return {};
        }
        if (copied && beg == 0) {
            llama_memory_seq_cp(mem, 0, 1, -1, -1);
            if (backer) {
                backer->copy_stream(0, 1);
            }
        }
    }
    return ret;
}

struct scenario {
    std::string                           what;
    context_shape                         shape;
    std::vector<std::vector<llama_token>> seqs;
    bool                                  copied = false;
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
        { "a prompt in one ubatch and decode steps",    { 64, 1, true,  GGML_TYPE_F16 },  { a } },
        { "a prompt in five ubatches and decode steps", { 5,  1, true,  GGML_TYPE_F16 },  { a } },
        { "two sequences in one KV stream",             { 64, 2, true,  GGML_TYPE_F16 },  { a, random_tokens() } },
        { "a Q8_0 K/V cache",                           { 5,  1, true,  GGML_TYPE_Q8_0 }, { a } },
        { "a sequence copied into a second stream",     { 64, 2, false, GGML_TYPE_F16 },  { a, random_tokens() }, true },
    };
}

// the hook was asked for every layer's K/V of both caches and for each rotation matrix, and the memory placed the
//     tensors on its answers; returns the number of layers placed on a placeholder
static int32_t check_asked(llama_context * lctx, const qsa_hook & hook, const std::string & what, checker & check) {
    llama_memory_hybrid_idx * mem = memory_of(lctx);

    int32_t n_placed = 0;
    bool    ok       = true;
    for (const auto & [cache, tag] : { std::make_pair(mem->get_mem_attn(), ""), std::make_pair(mem->get_mem_idx(), "idx_") }) {
        if (cache == nullptr) {
            continue;
        }
        const std::vector<ggml_tensor *> k_rot = cache->get_k_rot();
        const std::vector<ggml_tensor *> v_rot = cache->get_v_rot();
        for (const uint32_t il : cache->get_layer_ids()) {
            const asked_buft * a = hook.find(true, (int32_t) il, tag);
            if (a == nullptr) {
                ok = false;
                continue;
            }
            const ggml_backend_buffer_type_t want = a->answer ? a->answer : a->buft;
            n_placed += a->answer != nullptr;
            const ggml_tensor * k = cache->get_k_storage((int32_t) il);
            const ggml_tensor * v = cache->get_v_storage((int32_t) il);
            ok = ok && buft_of(k) == want && (v == nullptr || buft_of(v) == want);
            // the rotation is asked with the layer's own type, whatever its K/V were given, and kept there
            for (const ggml_tensor * rot : { k_rot[il], v_rot[il] }) {
                const asked_buft * r = rot ? hook.find(false, (int32_t) il, tag) : nullptr;
                ok = ok && (rot == nullptr || (r != nullptr && r->buft == a->buft && buft_of(rot) == a->buft));
            }
        }
    }
    check(ok && n_placed > 0, what + ": the hook is asked per kind, layer and tag, and its answers place the tensors (" +
        std::to_string(n_placed) + " layers placed)");
    return n_placed;
}

// the placed K/V tensors and their stream views have no data
static void check_placed_without_data(llama_context * lctx, int32_t n_placed, const std::string & what, checker & check) {
    const size_t n_stream = memory_of(lctx)->get_mem_attn()->get_n_stream();
    size_t n_views = 0;
    bool   ok      = true;
    for (const ggml_tensor * t : g_placed) {
        ok = ok && t->data == nullptr && is_placeholder(buft_of(t));
        n_views += t->view_src != nullptr;
    }
    // the K and V of each placed layer, each with one view per stream
    const size_t n_kv = 2*(size_t) n_placed;
    check(ok && g_placed.size() == n_kv*(1 + n_stream) && n_views == n_kv*n_stream,
        what + ": the placed K/V tensors and their stream views have no data (" + std::to_string(g_placed.size()) + ")");
}

// the plan records the placed tensors in placeholder buffers of size 0, and every other tensor as without the hook
static void check_plan(const llama_model * model, context_shape shape, const std::string & what, checker & check) {
    qsa_hook hook;
    hook.model = model;
    const llama_memory_plan stock  = llama_model_memory_plan(model, make_cparams(nullptr, nullptr, shape));
    const llama_memory_plan placed = llama_model_memory_plan(model, make_cparams(&hook, nullptr, shape));

    std::map<std::string, size_t> want_placed; // tensor name -> nbytes
    std::map<std::string, size_t> want_other;
    for (const auto & buf : stock.buffers) {
        const bool kv = buf.kind == LLAMA_MEMORY_PLAN_BUFFER_KV || buf.kind == LLAMA_MEMORY_PLAN_BUFFER_KV_SWA;
        for (const auto & t : buf.tensors) {
            const int32_t il = cache_layer(t.name.c_str());
            const bool qsa = kv && il >= 0 && llama_model_kv_rows_layer(*model, (uint32_t) il);
            (qsa ? want_placed : want_other)[t.name] = t.nbytes;
        }
    }

    std::map<std::string, size_t> got_placed;
    std::map<std::string, size_t> got_other;
    bool ok = true;
    for (const auto & buf : placed.buffers) {
        const bool ph = is_placeholder(buf.buft);
        const bool kv = buf.kind == LLAMA_MEMORY_PLAN_BUFFER_KV || buf.kind == LLAMA_MEMORY_PLAN_BUFFER_KV_SWA;
        ok = ok && (!ph || (kv && buf.size == 0));
        for (const auto & t : buf.tensors) {
            (ph ? got_placed : got_other)[t.name] = t.nbytes;
        }
    }
    check(ok && !want_placed.empty() && got_placed == want_placed && got_other == want_other,
        what + ": the plan records the placed tensors in size-0 placeholder buffers (" + std::to_string(got_placed.size()) + ")");
}

// cb_kv_rot_set still receives every rotation matrix of the attention cache, the placed layers' included
static void check_rot_set(llama_model * model, ggml_backend_buffer_type_t buft, context_shape shape, const std::string & what,
        checker & check) {
    qsa_hook hook;
    hook.model = model;
    std::vector<const ggml_tensor *> received;
    row_backer backer;
    backer.buft = buft;
    llama_context_params cparams = make_cparams(&hook, &backer, shape);
    cparams.cb_kv_rot_set = [](ggml_tensor * t, const void *, size_t size, void * ud) {
        if (size == ggml_nbytes(t)) {
            ((std::vector<const ggml_tensor *> *) ud)->push_back(t);
        }
    };
    cparams.cb_kv_rot_set_user_data = &received;
    llama_context_ptr lctx = new_context(model, cparams);

    const llama_kv_cache * attn = memory_of(lctx.get())->get_mem_attn();
    const std::vector<ggml_tensor *> k_rot = attn->get_k_rot();
    const std::vector<ggml_tensor *> v_rot = attn->get_v_rot();
    size_t n_rot        = 0;
    size_t n_rot_placed = 0;
    bool   ok           = true;
    for (const uint32_t il : attn->get_layer_ids()) {
        for (const ggml_tensor * rot : { k_rot[il], v_rot[il] }) {
            if (rot == nullptr) {
                continue;
            }
            n_rot++;
            n_rot_placed += is_placeholder(buft_of(attn->get_k_storage((int32_t) il)));
            ok = ok && std::find(received.begin(), received.end(), rot) != received.end();
        }
    }
    check(ok && n_rot_placed > 0, what + ": cb_kv_rot_set receives the rotation matrices of every layer (" +
        std::to_string(n_rot) + ", " + std::to_string(n_rot_placed) + " of placed layers)");
}

static void test_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string() + (devs.empty() ? " (CPU)" : " (GPU)");
    printf("%s\n", stem.c_str());
    llama_model_ptr model = load_model(path, devs, false);
    const ggml_backend_buffer_type_t buft = devs.empty() ? ggml_backend_cpu_buffer_type() : ggml_backend_dev_buffer_type(devs.front());

    const std::vector<scenario> scs = make_scenarios(model.get());
    for (const scenario & sc : scs) {
        const std::string what = stem + ": " + sc.what;

        llama_context_ptr lctx_stock = new_context(model.get(), make_cparams(nullptr, nullptr, sc.shape));
        const std::vector<float> ref = get_logits(lctx_stock.get(), nullptr, sc.seqs, sc.copied);

        qsa_hook hook;
        hook.model = model.get();
        row_backer backer;
        backer.buft = buft;
        llama_context_ptr lctx = new_context(model.get(), make_cparams(&hook, &backer, sc.shape));

        const int32_t n_placed = check_asked(lctx.get(), hook, what, check);
        check_placed_without_data(lctx.get(), n_placed, what, check);
        check_plan(model.get(), sc.shape, what, check);

        const std::vector<float> got = get_logits(lctx.get(), &backer, sc.seqs, sc.copied);

        check(backer.n_placed > 0 && backer.n_unplaced == 0 && backer.n_data == 0 && !backer.backed.empty(),
            what + ": the graphs read the placed K/V tensors (" + std::to_string(backer.backed.size()) + ")");
        check(!ref.empty() && got.size() == ref.size() && memcmp(got.data(), ref.data(), got.size()*sizeof(float)) == 0,
            what + ": logits backed by the caller's tensors are the stock context's, bitwise");

        if (ggml_is_quantized(sc.shape.type_kv)) {
            check_rot_set(model.get(), buft, sc.shape, what, check);
        }

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
        llama_memory_seq_div(mem, 0, 0, -1, 2);
        check(llama_memory_seq_pos_max(mem, 0) == pos_max && g_log.find("refusing the shift") != std::string::npos,
            stem + ": dividing positions is refused");

        g_log.clear();
        const size_t n_state     = llama_state_get_size(lctx.get());
        const bool   state_error = g_log.find("cannot hold them") != std::string::npos;
        g_log.clear();
        const size_t n_seq_state = llama_state_seq_get_size(lctx.get(), 0);
        check(n_state == 0 && state_error && n_seq_state == 0 && g_log.find("cannot hold them") != std::string::npos,
            stem + ": saving the state is refused");
    }

    // a memory built without allocation counts as having data: the hook still places the tensors, and nothing is refused
    llama_model_ptr model_na = load_model(path, devs, true);
    qsa_hook hook;
    hook.model = model_na.get();
    llama_context_ptr lctx_stock = new_context(model_na.get(), make_cparams(nullptr, nullptr, scs.front().shape));
    row_backer backer;
    backer.buft = buft;
    llama_context_ptr lctx       = new_context(model_na.get(), make_cparams(&hook, &backer, scs.front().shape));
    const llama_kv_cache * attn = memory_of(lctx.get())->get_mem_attn();
    bool placed = false;
    for (const uint32_t il : attn->get_layer_ids()) {
        placed = placed || is_placeholder(buft_of(attn->get_k_storage((int32_t) il)));
    }
    check(placed && !attn->has_layers_without_data() &&
        llama_memory_can_shift(llama_get_memory(lctx.get())) == llama_memory_can_shift(llama_get_memory(lctx_stock.get())),
        stem + ": a memory built without allocation refuses nothing");
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
