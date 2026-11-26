// checks the flash attention partial entry points for GGML_OP_CUSTOM nodes against the partial op, bitwise:
//   - CPU: a custom node whose function calls ggml_cpu_flash_attn_ext_partial_compute gives the bytes of
//     ggml_flash_attn_ext_partial, and ggml_cpu_flash_attn_ext_partial_work_size is the work size the CPU backend
//     plans for the partial
//   - Metal: a custom node whose device implementation calls ggml_metal_flash_attn_ext_partial_encode gives the bytes
//     of the partial on the same device, ggml_metal_flash_attn_ext_partial_supported takes the node, and
//     ggml_metal_flash_attn_ext_partial_extra_bytes is what the device's buffer type adds for the partial
//   - with an active count (ggml_flash_attn_ext_partial_set_active, mirrored by src[6] and the active index), an
//     active custom node gives the bytes of the partial without a count and an inactive one leaves its result
//     unwritten, for every count of the same graph
//
// the cases reach the vec path (small batches), the tiled path, a KV range off the cache tile (pad), a KV range
// viewed at an offset of the cache with its mask columns, masked and unmasked partials, ALiBi and softcap, and a
// quantized KV (dequantized to F16 first on Metal)
//
// the custom nodes are laid out as partial nodes: src[0..3] = q, k, v, mask, src[6] = the count; on Metal the node's
// buffer also holds the kernels' scratch past its data. A device that takes neither node is skipped and reported.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

struct fa_case {
    int64_t   dk;
    int64_t   dv;
    int64_t   nb;      // queries
    int64_t   kv;      // KV cells in the range
    int64_t   kv_off;  // first cell of the range in the cache
    ggml_type type_kv;
    bool      mask;
    float     max_bias;
    float     softcap;
};

static const fa_case CASES[] = {
    { 128, 128,  1, 256,  0, GGML_TYPE_F16,  true,  0.0f,  0.0f },
    { 128, 128,  1, 113,  0, GGML_TYPE_F16,  true,  0.0f,  0.0f },
    { 128, 128,  3, 256,  0, GGML_TYPE_F16,  false, 0.0f,  0.0f },
    { 128, 128,  1, 113, 64, GGML_TYPE_F16,  true,  0.0f,  0.0f },
    { 128, 128, 32, 256,  0, GGML_TYPE_F16,  true,  0.0f,  0.0f },
    { 128, 128, 32, 113,  0, GGML_TYPE_F16,  true,  0.0f,  0.0f },
    { 128, 128, 32, 113, 64, GGML_TYPE_F16,  true,  0.0f,  0.0f },
    { 128, 128, 32, 256,  0, GGML_TYPE_F16,  false, 0.0f, 30.0f },
    { 128, 128,  1, 256,  0, GGML_TYPE_Q8_0, true,  0.0f,  0.0f },
    { 128, 128, 32, 256,  0, GGML_TYPE_Q8_0, true,  0.0f,  0.0f },
    {  64,  64,  8, 256,  0, GGML_TYPE_F32,  true,  8.0f,  0.0f },
};

static const int64_t N_HEAD_KV = 2;
static const int64_t N_GQA     = 2;

static const int N_THREADS = 4;

// the index of the partial with a count
static const int32_t ACTIVE_INDEX = 1;

// a bit pattern no partial produces: a NaN with a payload
static const uint32_t SENTINEL = 0x7fc0beefu;

//
// the custom node
//

// the op parameters of one custom node, and the CPU work data its threads share
struct node_state {
    ggml_cpu_fa_partial_params cpu;
    std::vector<uint8_t>       wdata;
    int                        nth = 0; // the thread count of the last CPU compute
};

static void fa_partial_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    node_state * s = (node_state *) userdata;
    if (ith == 0) {
        s->nth = nth;
    }
    ggml_cpu_flash_attn_ext_partial_compute(dst, &s->cpu, ith, nth, s->wdata.data(), s->wdata.size());
}

#ifdef GGML_USE_METAL
// the state of each custom node the Metal implementation encodes
static std::unordered_map<const ggml_tensor *, node_state *> g_metal_nodes;

// the Metal implementation of fa_partial_cpu nodes on one device
struct metal_impl {
    ggml_backend_dev_t dev = nullptr;
};

static bool fa_partial_metal_supports(void * user_data, const ggml_tensor * node) {
    const metal_impl * impl = (const metal_impl *) user_data;
    return ggml_metal_flash_attn_ext_partial_supported(impl->dev, node);
}

static bool fa_partial_metal_encode(void * user_data, const ggml_tensor * node, void * native_ctx) {
    const metal_impl * impl = (const metal_impl *) user_data;

    const auto it = g_metal_nodes.find(node);
    if (it == g_metal_nodes.end()) {
        return false;
    }

    const ggml_cpu_fa_partial_params & p = it->second->cpu;

    const ggml_metal_fa_partial_params params = { p.scale, p.max_bias, p.logit_softcap, p.active_index };

    return ggml_metal_flash_attn_ext_partial_encode(impl->dev, native_ctx, node, &params);
}

// one per Metal device, registered for the device's lifetime
static std::vector<metal_impl> g_metal_impls;
#endif

//
// data
//

static std::vector<uint8_t> random_bytes(ggml_type type, int64_t n_per_row, int64_t n_rows, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(n_per_row * n_rows);
    for (float & v : values) {
        v = dist(rng);
    }
    std::vector<uint8_t> out(ggml_row_size(type, n_per_row) * n_rows);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), values.data(), out.size());
    } else {
        ggml_quantize_chunk(type, values.data(), out.data(), 0, n_rows, n_per_row, nullptr);
    }
    return out;
}

// zeros, small values and -inf, with every fourth query row fully masked
static std::vector<ggml_fp16_t> random_mask(int64_t kv, int64_t nb, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<ggml_fp16_t> out(kv * nb);
    for (int64_t r = 0; r < nb; ++r) {
        for (int64_t c = 0; c < kv; ++c) {
            const float u = dist(rng);
            const float v = (r % 4 == 3) ? -INFINITY : u < -0.5f ? -INFINITY : u < 0.0f ? 0.0f : u;
            out[r * kv + c] = ggml_fp32_to_fp16(v);
        }
    }
    return out;
}

static std::vector<uint8_t> get_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> out(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    return out;
}

static void set_sentinel(ggml_tensor * t) {
    std::vector<uint32_t> s(ggml_nbytes(t) / sizeof(uint32_t), SENTINEL);
    ggml_backend_tensor_set(t, s.data(), 0, ggml_nbytes(t));
}

static bool is_sentinel(const ggml_tensor * t) {
    std::vector<uint32_t> got(ggml_nbytes(t) / sizeof(uint32_t));
    ggml_backend_tensor_get(t, got.data(), 0, ggml_nbytes(t));
    for (uint32_t w : got) {
        if (w != SENTINEL) {
            return false;
        }
    }
    return true;
}

// the first element is the count, the others are not read
static void set_count(ggml_tensor * t, int32_t count) {
    const int32_t values[4] = { count, -1, -1, -1 };
    ggml_backend_tensor_set(t, values, 0, sizeof(values));
}

//
// the check
//

enum class outcome { ok, fail, skipped };

static bool is_metal(ggml_backend_dev_t dev) {
#ifdef GGML_USE_METAL
    return ggml_backend_dev_backend_reg(dev) == ggml_backend_metal_reg();
#else
    (void) dev;
    return false;
#endif
}

static outcome check(ggml_backend_dev_t dev, const fa_case & c, std::mt19937 & rng) {
    const bool on_cpu   = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const bool on_metal = is_metal(dev);
    if (!on_cpu && !on_metal) {
        return outcome::skipped;
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) {
        return outcome::skipped;
    }
    if (on_cpu) {
        ggml_backend_cpu_set_n_threads(backend.get(), N_THREADS);
    }

    const ggml_init_params params = {
        /*.mem_size   =*/ 32 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_context_ptr ctx_custom(ggml_init(params));

    const int64_t n_cells = c.kv_off + c.kv;

    ggml_tensor * q  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.dk, c.nb, N_HEAD_KV * N_GQA, 1);
    ggml_tensor * kc = ggml_new_tensor_4d(ctx.get(), c.type_kv, c.dk, n_cells, N_HEAD_KV, 1);
    ggml_tensor * vc = ggml_new_tensor_4d(ctx.get(), c.type_kv, c.dv, n_cells, N_HEAD_KV, 1);
    ggml_tensor * mc = c.mask ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_cells, c.nb, 1, 1) : nullptr;

    // the range [kv_off, kv_off + kv) of the cache, and its mask columns
    ggml_tensor * k = ggml_view_3d(ctx.get(), kc, c.dk, c.kv, N_HEAD_KV, kc->nb[1], kc->nb[2], c.kv_off * kc->nb[1]);
    ggml_tensor * v = ggml_view_3d(ctx.get(), vc, c.dv, c.kv, N_HEAD_KV, vc->nb[1], vc->nb[2], c.kv_off * vc->nb[1]);
    ggml_tensor * m = mc ? ggml_view_2d(ctx.get(), mc, c.kv, c.nb, mc->nb[1], c.kv_off * mc->nb[0]) : nullptr;

    ggml_tensor * n_active = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);

    const float scale = 1.0f / sqrtf((float) c.dk);

    // the partial op, without and with a count
    ggml_tensor * p_full = ggml_flash_attn_ext_partial(ctx.get(), q, k, v, m, scale, c.max_bias, c.softcap);
    ggml_tensor * p_act  = ggml_flash_attn_ext_partial(ctx.get(), q, k, v, m, scale, c.max_bias, c.softcap);
    for (ggml_tensor * p : { p_full, p_act }) {
        ggml_prec_set_acc(p, GGML_PREC_F32);
    }
    ggml_flash_attn_ext_partial_set_active(p_act, n_active, ACTIVE_INDEX);

    // the custom nodes laid out as the partials
    node_state s_full;
    node_state s_act;
    s_full.cpu = { scale, c.max_bias, c.softcap, GGML_PREC_F32, 0 };
    s_act.cpu  = { scale, c.max_bias, c.softcap, GGML_PREC_F32, ACTIVE_INDEX };

    ggml_tensor * args_full[7] = { q, k, v, m, nullptr, nullptr, nullptr  };
    ggml_tensor * args_act [7] = { q, k, v, m, nullptr, nullptr, n_active };

    ggml_tensor * c_full = ggml_custom_4d(ctx_custom.get(), GGML_TYPE_F32, p_full->ne[0], p_full->ne[1], p_full->ne[2], p_full->ne[3],
            args_full, 7, fa_partial_cpu, GGML_N_TASKS_MAX, &s_full);
    ggml_tensor * c_act  = ggml_custom_4d(ctx_custom.get(), GGML_TYPE_F32, p_act->ne[0], p_act->ne[1], p_act->ne[2], p_act->ne[3],
            args_act, 7, fa_partial_cpu, GGML_N_TASKS_MAX, &s_act);

    for (ggml_tensor * p : { p_full, p_act }) {
        if (!ggml_backend_supports_op(backend.get(), p)) {
            return outcome::skipped;
        }
    }

    bool ok = true;

    for (ggml_tensor * t : { c_full, c_act }) {
        if (!ggml_backend_supports_op(backend.get(), t)) {
            printf("    the device takes the partial but not the custom node\n");
            return outcome::fail;
        }
    }

    if (on_cpu) {
        // the work size the CPU backend plans for a graph of the partial alone
        ggml_cgraph * g1 = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(g1, p_full);
        const ggml_cplan plan = ggml_graph_plan(g1, N_THREADS, nullptr);
        const size_t wsize = ggml_cpu_flash_attn_ext_partial_work_size(c_full, N_THREADS);
        if (plan.work_size != wsize) {
            printf("    the work size %zu is not the planned %zu\n", wsize, plan.work_size);
            ok = false;
        }
        s_full.wdata.resize(wsize);
        s_act.wdata.resize(ggml_cpu_flash_attn_ext_partial_work_size(c_act, N_THREADS));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, p_full);
    ggml_build_forward_expand(gf, p_act);
    ggml_build_forward_expand(gf, c_full);
    ggml_build_forward_expand(gf, c_act);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buf) {
        return outcome::skipped;
    }

    // the custom nodes, each with the scratch its kernels use past its data
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());

    size_t extra = 0;
#ifdef GGML_USE_METAL
    if (on_metal) {
        extra = ggml_metal_flash_attn_ext_partial_extra_bytes(c_full);
        const size_t want = ggml_backend_buft_get_alloc_size(buft, p_full) - ggml_nbytes(p_full);
        if (extra != want || ggml_metal_flash_attn_ext_partial_extra_bytes(c_act) != want) {
            printf("    the extra bytes %zu are not the %zu the buffer type adds for the partial\n", extra, want);
            ok = false;
        }
        g_metal_nodes.clear();
        g_metal_nodes[c_full] = &s_full;
        g_metal_nodes[c_act]  = &s_act;
    }
#endif

    const size_t align   = ggml_backend_buft_get_alignment(buft);
    const size_t c_bytes = GGML_PAD(ggml_nbytes(c_full) + extra, align);

    ggml_backend_buffer_ptr buf_custom(ggml_backend_buft_alloc_buffer(buft, 2 * c_bytes));
    if (!buf_custom) {
        return outcome::fail;
    }
    char * base = (char *) ggml_backend_buffer_get_base(buf_custom.get());
    if (ggml_backend_tensor_alloc(buf_custom.get(), c_full, base) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(buf_custom.get(), c_act, base + c_bytes) != GGML_STATUS_SUCCESS) {
        return outcome::fail;
    }

    const std::vector<uint8_t> qb = random_bytes(GGML_TYPE_F32, c.dk, ggml_nrows(q), rng);
    const std::vector<uint8_t> kb = random_bytes(c.type_kv, c.dk, ggml_nrows(kc), rng);
    const std::vector<uint8_t> vb = random_bytes(c.type_kv, c.dv, ggml_nrows(vc), rng);
    ggml_backend_tensor_set(q,  qb.data(), 0, qb.size());
    ggml_backend_tensor_set(kc, kb.data(), 0, kb.size());
    ggml_backend_tensor_set(vc, vb.data(), 0, vb.size());
    if (mc) {
        const std::vector<ggml_fp16_t> mb = random_mask(n_cells, c.nb, rng);
        ggml_backend_tensor_set(mc, mb.data(), 0, mb.size() * sizeof(ggml_fp16_t));
    }

    // on the same graph, by count: 0 and 1 leave the partials with a count inactive, 2 runs them
    for (int32_t count : { 0, 1, 2 }) {
        set_count(n_active, count);

        set_sentinel(p_act);
        set_sentinel(c_full);
        set_sentinel(c_act);

        if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
            return outcome::fail;
        }

        if (on_cpu && (s_full.nth != N_THREADS || s_act.nth != N_THREADS)) {
            printf("    the custom nodes ran on %d and %d threads, not %d\n", s_full.nth, s_act.nth, N_THREADS);
            ok = false;
        }

        const std::vector<uint8_t> ref = get_bytes(p_full);

        if (get_bytes(c_full) != ref) {
            printf("    count %d: the custom node without a count differs from the partial\n", count);
            ok = false;
        }

        const bool active = count > ACTIVE_INDEX;
        if (active && get_bytes(p_act) != ref) {
            printf("    count %d: the active partial differs from the partial without a count\n", count);
            ok = false;
        }
        if (active && get_bytes(c_act) != ref) {
            printf("    count %d: the active custom node differs from the partial without a count\n", count);
            ok = false;
        }
        if (!active && !is_sentinel(c_act)) {
            printf("    count %d: the inactive custom node wrote its result\n", count);
            ok = false;
        }
    }

    return ok ? outcome::ok : outcome::fail;
}

int main() {
    ggml_backend_load_all();

#ifdef GGML_USE_METAL
    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    const size_t n_metal = reg != nullptr ? ggml_backend_reg_dev_count(reg) : 0;

    // the entries are not moved once registered
    g_metal_impls.resize(n_metal);
    for (size_t i = 0; i < n_metal; ++i) {
        g_metal_impls[i].dev = ggml_backend_reg_dev_get(reg, i);

        const ggml_backend_custom_op op = { fa_partial_cpu, fa_partial_metal_supports, fa_partial_metal_encode, &g_metal_impls[i], nullptr };
        if (!ggml_backend_dev_add_custom_op(g_metal_impls[i].dev, &op)) {
            printf("the Metal device %s refuses the custom op\n", ggml_backend_dev_name(g_metal_impls[i].dev));
            return 1;
        }
    }
#endif

    std::mt19937 rng(42);
    int n_checked = 0;
    int n_fail    = 0;
    for (const fa_case & c : CASES) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const outcome got = check(dev, c, rng);
            const char * verdict = got == outcome::ok ? "OK" : got == outcome::fail ? "FAIL" : "skipped";
            printf("  dk=%3lld dv=%3lld nb=%2lld kv=%3lld+%2lld %-5s mask=%d bias=%g scap=%g %-12s: %s\n",
                    (long long) c.dk, (long long) c.dv, (long long) c.nb, (long long) c.kv, (long long) c.kv_off,
                    ggml_type_name(c.type_kv), c.mask ? 1 : 0, c.max_bias, c.softcap, ggml_backend_dev_name(dev), verdict);
            n_checked += got == outcome::skipped ? 0 : 1;
            n_fail    += got == outcome::fail    ? 1 : 0;
        }
    }

    if (n_checked == 0) {
        printf("no device supports the flash attention partial\n");
        return 1;
    }
    printf("%d checked, %d failed\n", n_checked, n_fail);
    return n_fail == 0 ? 0 : 1;
}
