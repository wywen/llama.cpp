// checks the flash attention partial entry points for GGML_OP_CUSTOM nodes on the CPU and Metal backends:
//   - the partials of the chunks of a KV cache, each computed by a custom node over the chunk's views of K, V and
//     the mask, combined and normalized on the host, match the stock unchunked ggml_flash_attn_ext of the same
//     inputs on the same device
//   - the partials are deterministic: the same bytes on every run of the graph and, on the CPU, for every thread
//     count
//   - with an active count (src[6] and the active index), a custom node whose count is greater than its index gives
//     the bytes of the node without a count, and one whose count is not leaves its result unwritten, for every count
//     of the same graph
//   - CPU: ggml_cpu_flash_attn_ext_partial_work_size is the work size the CPU backend plans for the stock op
//   - Metal: ggml_metal_flash_attn_ext_partial_supported takes the nodes, ggml_metal_flash_attn_ext_partial_extra_bytes
//     is what the device's buffer type adds for a stock flash attention node of the same shapes, and the buffer type
//     sizes the custom nodes with it through the implementation's alloc_size
//
// the cases reach the vec path (small batches), the tiled path, chunk boundaries off the cache tile (pad), masked
// and unmasked partials, a chunk without valid cells, rows without valid cells, ALiBi and softcap, and a quantized
// KV (dequantized to F16 first on Metal)
//
// the custom nodes are laid out as the entry points take them: src[0..3] = q, k, v, mask, src[6] = the count. A
// device that does not take the stock op is skipped and reported.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <unordered_map>
#include <vector>

struct fa_case {
    int64_t              dk;
    int64_t              dv;
    int64_t              nb;          // queries
    std::vector<int64_t> bounds;      // chunk i holds the KV cells [bounds[i], bounds[i + 1])
    ggml_type            type_kv;
    bool                 mask;
    int64_t              empty_chunk; // a chunk whose mask columns are all -inf, or -1
    float                max_bias;
    float                softcap;
};

// on Metal a mask column view starts at a multiple of 4 bytes: the chunk boundaries are even
static const fa_case CASES[] = {
    { 128, 128,  1, { 0, 128, 256 },      GGML_TYPE_F16,  true,  -1, 0.0f,  0.0f },
    { 128, 128,  1, { 0, 114, 226, 256 }, GGML_TYPE_F16,  true,  -1, 0.0f,  0.0f },
    { 128, 128,  3, { 0, 128, 256 },      GGML_TYPE_F16,  false, -1, 0.0f,  0.0f },
    { 128, 128, 32, { 0, 128, 256 },      GGML_TYPE_F16,  true,  -1, 0.0f,  0.0f },
    { 128, 128, 32, { 0, 114, 226, 256 }, GGML_TYPE_F16,  true,  -1, 0.0f,  0.0f },
    { 128, 128,  8, { 0, 128, 256 },      GGML_TYPE_F16,  true,   1, 0.0f,  0.0f },
    { 128, 128, 32, { 0, 128, 256 },      GGML_TYPE_F16,  false, -1, 0.0f, 30.0f },
    { 128, 128,  1, { 0, 128, 256 },      GGML_TYPE_Q8_0, true,  -1, 0.0f,  0.0f },
    { 128, 128, 32, { 0, 128, 256 },      GGML_TYPE_Q8_0, true,  -1, 0.0f,  0.0f },
    {  64,  64,  8, { 0, 64, 192, 256 },  GGML_TYPE_F32,  true,  -1, 8.0f,  0.0f },
};

static const int64_t N_HEAD_KV = 2;
static const int64_t N_GQA     = 2;

// the CPU thread counts the graph runs with
static const int CPU_THREADS[] = { 4, 1, 3 };
static const int MAX_THREADS   = 4;

// the combined partials against the stock op
static const double MAX_NMSE = 5e-4;

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

// the node's data and the kernels' scratch past it
static size_t fa_partial_metal_alloc_size(void * user_data, const ggml_tensor * node) {
    (void) user_data;
    return ggml_nbytes(node) + ggml_metal_flash_attn_ext_partial_extra_bytes(node);
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

// zeros, small values and -inf, with every fourth query row fully masked and the columns [c0, c1) all -inf
static std::vector<ggml_fp16_t> random_mask(int64_t kv, int64_t nb, int64_t c0, int64_t c1, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<ggml_fp16_t> out(kv * nb);
    for (int64_t r = 0; r < nb; ++r) {
        for (int64_t c = 0; c < kv; ++c) {
            const float u      = dist(rng);
            const bool  masked = r % 4 == 3 || (c >= c0 && c < c1);
            const float v      = masked ? -INFINITY : u < -0.5f ? -INFINITY : u < 0.0f ? 0.0f : u;
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

static std::vector<float> get_floats(const ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
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
// the host oracle
//

// combine the partials of the chunks of one (head, query) row and normalize them as a caller does: the output of
// ggml_flash_attn_ext over the chunks' cells. part holds n_chunks partials of dv + 2 values, one after the other
static void fa_partials_combine(const float * part, int64_t n_chunks, int64_t dv, float * out) {
    float M = -INFINITY;
    for (int64_t i = 0; i < n_chunks; ++i) {
        M = std::max(M, part[i*(dv + 2) + dv + 1]);
    }

    float S = 0.0f;
    for (int64_t j = 0; j < dv; ++j) {
        out[j] = 0.0f;
    }

    // a partial without finite scores has M = -inf and contributes nothing
    for (int64_t i = 0; i < n_chunks; ++i) {
        const float * p  = part + i*(dv + 2);
        const float   Mi = p[dv + 1];
        const float   w  = Mi == -INFINITY ? 0.0f : expf(Mi - M);
        for (int64_t j = 0; j < dv; ++j) {
            out[j] += p[j]*w;
        }
        S += p[dv + 0]*w;
    }

    const float scale = S == 0.0f ? 0.0f : 1.0f/S;
    for (int64_t j = 0; j < dv; ++j) {
        out[j] *= scale;
    }
}

// the normalized mean squared error of a against b; infinite when either holds a value that is not finite
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double mse_a_b = 0.0;
    double mse_b_0 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return INFINITY;
        }
        mse_a_b += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        mse_b_0 += (double) b[i] * b[i];
    }
    return mse_b_0 == 0.0 ? mse_a_b : mse_a_b / mse_b_0;
}

// the error of the host combination of `parts` (the partials of the chunks, each [dv + 2, rows]) against `ref`
// ([dv, rows], the stock op)
static double combined_error(const std::vector<std::vector<float>> & parts, const std::vector<float> & ref, int64_t dv) {
    const int64_t n_chunks = (int64_t) parts.size();
    const int64_t n_rows   = (int64_t) ref.size() / dv;

    std::vector<float> part(n_chunks * (dv + 2));
    std::vector<float> got(ref.size());
    for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t i = 0; i < n_chunks; ++i) {
            memcpy(part.data() + i*(dv + 2), parts[i].data() + r*(dv + 2), (dv + 2)*sizeof(float));
        }
        fa_partials_combine(part.data(), n_chunks, dv, got.data() + r*dv);
    }

    return nmse(got, ref);
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

    const int64_t n_chunks = (int64_t) c.bounds.size() - 1;
    const int64_t n_cells  = c.bounds.back();

    const ggml_init_params params = {
        /*.mem_size   =*/ 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));        // the inputs and the stock op
    ggml_context_ptr ctx_custom(ggml_init(params)); // the custom nodes
    ggml_context_ptr ctx_shape(ggml_init(params));  // stock nodes of the chunks' shapes, never allocated

    ggml_tensor * q  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.dk, c.nb, N_HEAD_KV * N_GQA, 1);
    ggml_tensor * kc = ggml_new_tensor_4d(ctx.get(), c.type_kv, c.dk, n_cells, N_HEAD_KV, 1);
    ggml_tensor * vc = ggml_new_tensor_4d(ctx.get(), c.type_kv, c.dv, n_cells, N_HEAD_KV, 1);
    ggml_tensor * mc = c.mask ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_cells, c.nb, 1, 1) : nullptr;

    ggml_tensor * n_active = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);

    const float scale = 1.0f / sqrtf((float) c.dk);

    // the stock op over the whole cache
    ggml_tensor * ref = ggml_flash_attn_ext(ctx.get(), q, kc, vc, mc, scale, c.max_bias, c.softcap);
    ggml_prec_set_acc(ref, GGML_PREC_F32);

    if (!ggml_backend_supports_op(backend.get(), ref)) {
        return outcome::skipped;
    }

    // per chunk: a custom node without a count, and one with the count and the chunk's index
    std::vector<node_state>    s_full(n_chunks);
    std::vector<node_state>    s_act (n_chunks);
    std::vector<ggml_tensor *> c_full(n_chunks);
    std::vector<ggml_tensor *> c_act (n_chunks);
    std::vector<ggml_tensor *> shape (n_chunks);

    for (int64_t i = 0; i < n_chunks; ++i) {
        const int64_t c0 = c.bounds[i];
        const int64_t nc = c.bounds[i + 1] - c0;

        ggml_tensor * k = ggml_view_3d(ctx.get(), kc, c.dk, nc, N_HEAD_KV, kc->nb[1], kc->nb[2], c0 * kc->nb[1]);
        ggml_tensor * v = ggml_view_3d(ctx.get(), vc, c.dv, nc, N_HEAD_KV, vc->nb[1], vc->nb[2], c0 * vc->nb[1]);
        ggml_tensor * m = mc ? ggml_view_2d(ctx.get(), mc, nc, c.nb, mc->nb[1], c0 * mc->nb[0]) : nullptr;

        s_full[i].cpu = { scale, c.max_bias, c.softcap, GGML_PREC_F32, 0 };
        s_act [i].cpu = { scale, c.max_bias, c.softcap, GGML_PREC_F32, (int32_t) i };

        ggml_tensor * args_full[7] = { q, k, v, m, nullptr, nullptr, nullptr  };
        ggml_tensor * args_act [7] = { q, k, v, m, nullptr, nullptr, n_active };

        // the partial of each (head, query) row: the unnormalized output, S and M
        const int64_t ne0 = c.dv + 2;
        const int64_t ne1 = q->ne[2];
        const int64_t ne2 = q->ne[1];

        c_full[i] = ggml_custom_4d(ctx_custom.get(), GGML_TYPE_F32, ne0, ne1, ne2, 1, args_full, 7, fa_partial_cpu, GGML_N_TASKS_MAX, &s_full[i]);
        c_act [i] = ggml_custom_4d(ctx_custom.get(), GGML_TYPE_F32, ne0, ne1, ne2, 1, args_act,  7, fa_partial_cpu, GGML_N_TASKS_MAX, &s_act[i]);

        // a stock node of the chunk's shapes, with a contiguous mask
        ggml_tensor * m_shape = mc ? ggml_new_tensor_4d(ctx_shape.get(), GGML_TYPE_F16, nc, c.nb, 1, 1) : nullptr;
        shape[i] = ggml_flash_attn_ext(ctx_shape.get(), q, k, v, m_shape, scale, c.max_bias, c.softcap);
    }

    for (int64_t i = 0; i < n_chunks; ++i) {
        for (ggml_tensor * t : { c_full[i], c_act[i] }) {
            if (!ggml_backend_supports_op(backend.get(), t)) {
                printf("    the device takes the stock op but not the custom node of chunk %lld\n", (long long) i);
                return outcome::fail;
            }
        }
    }

    bool ok = true;

    if (on_cpu) {
        // the work size the CPU backend plans for a graph of the stock op alone, of the same q, K and V head sizes
        ggml_cgraph * g1 = ggml_new_graph(ctx_shape.get());
        ggml_build_forward_expand(g1, ref);
        for (int nth : CPU_THREADS) {
            const ggml_cplan plan = ggml_graph_plan(g1, nth, nullptr);
            const size_t wsize = ggml_cpu_flash_attn_ext_partial_work_size(c_full[0], nth);
            if (plan.work_size != wsize) {
                printf("    %d threads: the work size %zu is not the planned %zu\n", nth, wsize, plan.work_size);
                ok = false;
            }
        }
        for (int64_t i = 0; i < n_chunks; ++i) {
            s_full[i].wdata.resize(ggml_cpu_flash_attn_ext_partial_work_size(c_full[i], MAX_THREADS));
            s_act [i].wdata.resize(ggml_cpu_flash_attn_ext_partial_work_size(c_act [i], MAX_THREADS));
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, ref);
    for (int64_t i = 0; i < n_chunks; ++i) {
        ggml_build_forward_expand(gf, c_full[i]);
        ggml_build_forward_expand(gf, c_act[i]);
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buf) {
        return outcome::skipped;
    }

    // the custom nodes, sized by the device's buffer type: on Metal through the implementation's alloc_size
    ggml_backend_buffer_ptr buf_custom(ggml_backend_alloc_ctx_tensors(ctx_custom.get(), backend.get()));
    if (!buf_custom) {
        return outcome::fail;
    }

#ifdef GGML_USE_METAL
    if (on_metal) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());

        for (int64_t i = 0; i < n_chunks; ++i) {
            const size_t extra = ggml_metal_flash_attn_ext_partial_extra_bytes(c_full[i]);
            const size_t want  = ggml_backend_buft_get_alloc_size(buft, shape[i]) - ggml_nbytes(shape[i]);
            if (extra != want || ggml_metal_flash_attn_ext_partial_extra_bytes(c_act[i]) != want) {
                printf("    chunk %lld: the extra bytes %zu are not the %zu the buffer type adds for the stock op\n",
                        (long long) i, extra, want);
                ok = false;
            }
            if (ggml_backend_buft_get_alloc_size(buft, c_full[i]) != ggml_nbytes(c_full[i]) + extra) {
                printf("    chunk %lld: the buffer type does not size the custom node with its extra bytes\n", (long long) i);
                ok = false;
            }
        }

        g_metal_nodes.clear();
        for (int64_t i = 0; i < n_chunks; ++i) {
            g_metal_nodes[c_full[i]] = &s_full[i];
            g_metal_nodes[c_act [i]] = &s_act [i];
        }
    }
#else
    (void) shape;
#endif

    const std::vector<uint8_t> qb = random_bytes(GGML_TYPE_F32, c.dk, ggml_nrows(q), rng);
    const std::vector<uint8_t> kb = random_bytes(c.type_kv, c.dk, ggml_nrows(kc), rng);
    const std::vector<uint8_t> vb = random_bytes(c.type_kv, c.dv, ggml_nrows(vc), rng);
    ggml_backend_tensor_set(q,  qb.data(), 0, qb.size());
    ggml_backend_tensor_set(kc, kb.data(), 0, kb.size());
    ggml_backend_tensor_set(vc, vb.data(), 0, vb.size());
    if (mc) {
        const int64_t e0 = c.empty_chunk >= 0 ? c.bounds[c.empty_chunk]     : 0;
        const int64_t e1 = c.empty_chunk >= 0 ? c.bounds[c.empty_chunk + 1] : 0;
        const std::vector<ggml_fp16_t> mb = random_mask(n_cells, c.nb, e0, e1, rng);
        ggml_backend_tensor_set(mc, mb.data(), 0, mb.size() * sizeof(ggml_fp16_t));
    }

    // the partials of the first run, which every later run repeats
    std::vector<std::vector<uint8_t>> first;

    // a Metal device runs each count once more, on the same graph
    const std::vector<int> runs = on_cpu ? std::vector<int>(std::begin(CPU_THREADS), std::end(CPU_THREADS)) : std::vector<int>{ 0, 0 };

    for (int nth : runs) {
        if (on_cpu) {
            ggml_backend_cpu_set_n_threads(backend.get(), nth);
        }

        // on the same graph, by count: the chunks below the count are active
        for (int32_t count : { 0, 1, (int32_t) n_chunks }) {
            set_count(n_active, count);

            for (int64_t i = 0; i < n_chunks; ++i) {
                set_sentinel(c_full[i]);
                set_sentinel(c_act[i]);
            }

            if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
                return outcome::fail;
            }

            if (on_cpu) {
                for (int64_t i = 0; i < n_chunks; ++i) {
                    if (s_full[i].nth != nth || s_act[i].nth != nth) {
                        printf("    chunk %lld: the custom nodes ran on %d and %d threads, not %d\n",
                                (long long) i, s_full[i].nth, s_act[i].nth, nth);
                        ok = false;
                    }
                }
            }

            std::vector<std::vector<uint8_t>> got(n_chunks);
            for (int64_t i = 0; i < n_chunks; ++i) {
                got[i] = get_bytes(c_full[i]);
            }

            if (first.empty()) {
                first = got;

                std::vector<std::vector<float>> parts(n_chunks);
                for (int64_t i = 0; i < n_chunks; ++i) {
                    parts[i] = get_floats(c_full[i]);
                }

                const double err = combined_error(parts, get_floats(ref), c.dv);
                if (!(err <= MAX_NMSE)) {
                    printf("    the combined partials differ from the stock op: NMSE %g > %g\n", err, MAX_NMSE);
                    ok = false;
                }
            } else if (got != first) {
                printf("    %d threads, count %d: the partials differ from the first run\n", nth, count);
                ok = false;
            }

            for (int64_t i = 0; i < n_chunks; ++i) {
                const bool active = count > i;
                if (active && get_bytes(c_act[i]) != got[i]) {
                    printf("    %d threads, count %d: the active custom node of chunk %lld differs from the node without a count\n",
                            nth, count, (long long) i);
                    ok = false;
                }
                if (!active && !is_sentinel(c_act[i])) {
                    printf("    %d threads, count %d: the inactive custom node of chunk %lld wrote its result\n",
                            nth, count, (long long) i);
                    ok = false;
                }
            }
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

        const ggml_backend_custom_op op = {
            fa_partial_cpu, fa_partial_metal_supports, fa_partial_metal_encode, &g_metal_impls[i], fa_partial_metal_alloc_size,
        };
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
            printf("  dk=%3lld dv=%3lld nb=%2lld kv=%3lld chunks=%lld %-5s mask=%d empty=%2lld bias=%g scap=%g %-12s: %s\n",
                    (long long) c.dk, (long long) c.dv, (long long) c.nb, (long long) c.bounds.back(),
                    (long long) c.bounds.size() - 1, ggml_type_name(c.type_kv), c.mask ? 1 : 0, (long long) c.empty_chunk,
                    c.max_bias, c.softcap, ggml_backend_dev_name(dev), verdict);
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
