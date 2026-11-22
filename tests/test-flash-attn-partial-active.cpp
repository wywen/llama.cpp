// checks ggml_flash_attn_ext_partial_set_active on every backend device that supports the partial op:
//   - a partial whose active count is greater than its index gives the same bytes as the partial without
//     a count
//   - a partial whose active count is not greater than its index leaves its result unwritten (a sentinel
//     survives the compute)
//   - the count is read at compute time: changing it between two computes of the same graph flips both
//
// the cases reach the kernels and prepasses a partial runs through on the CPU and Metal backends:
// the vec path (small batches), the tiled path, a KV range off the cache tile (pad), a mask (blk) and a
// quantized KV (dequantized to F16 first)
//
// a device that does not support the op is skipped and reported

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct fa_case {
    int64_t   dk;
    int64_t   dv;
    int64_t   nb;      // queries
    int64_t   kv;      // KV cells in the range
    ggml_type type_kv;
    bool      mask;
};

static const fa_case CASES[] = {
    { 128, 128,  1, 256, GGML_TYPE_F16,  true  },
    { 128, 128,  1, 113, GGML_TYPE_F16,  true  },
    { 128, 128,  3, 256, GGML_TYPE_F16,  false },
    { 128, 128, 32, 256, GGML_TYPE_F16,  true  },
    { 128, 128, 32, 113, GGML_TYPE_F16,  true  },
    { 128, 128, 32, 256, GGML_TYPE_Q8_0, true  },
    {  64,  64,  8, 256, GGML_TYPE_F32,  true  },
};

static const int64_t N_HEAD_KV = 2;
static const int64_t N_GQA     = 2;

// a bit pattern no partial produces: a NaN with a payload
static const uint32_t SENTINEL = 0x7fc0beefu;

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

// zeros, small values and -inf, with every other query row of the first head fully masked
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

enum class outcome { ok, fail, skipped };

static outcome check(ggml_backend_dev_t dev, const fa_case & c, std::mt19937 & rng) {
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) {
        return outcome::skipped;
    }

    const ggml_init_params params = {
        /*.mem_size   =*/ 32 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));

    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.dk, c.nb, N_HEAD_KV * N_GQA, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), c.type_kv, c.dk, c.kv, N_HEAD_KV, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), c.type_kv, c.dv, c.kv, N_HEAD_KV, 1);
    ggml_tensor * m = c.mask ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, c.kv, c.nb, 1, 1) : nullptr;

    // one count for both partials, read through its first element: a is index 0, b is index 2
    ggml_tensor * n_active = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);

    const float scale = 1.0f / sqrtf((float) c.dk);

    ggml_tensor * p_ref = ggml_flash_attn_ext_partial(ctx.get(), q, k, v, m, scale, 0.0f, 0.0f);
    ggml_tensor * p_a   = ggml_flash_attn_ext_partial(ctx.get(), q, k, v, m, scale, 0.0f, 0.0f);
    ggml_tensor * p_b   = ggml_flash_attn_ext_partial(ctx.get(), q, k, v, m, scale, 0.0f, 0.0f);
    for (ggml_tensor * p : { p_ref, p_a, p_b }) {
        ggml_prec_set_acc(p, GGML_PREC_F32);
    }
    ggml_flash_attn_ext_partial_set_active(p_a, n_active, 0);
    ggml_flash_attn_ext_partial_set_active(p_b, n_active, 2);

    for (ggml_tensor * p : { p_ref, p_a, p_b }) {
        if (!ggml_backend_supports_op(backend.get(), p)) {
            return outcome::skipped;
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, p_ref);
    ggml_build_forward_expand(gf, p_a);
    ggml_build_forward_expand(gf, p_b);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buf) {
        return outcome::skipped;
    }

    const std::vector<uint8_t> qb = random_bytes(GGML_TYPE_F32, c.dk, ggml_nrows(q), rng);
    const std::vector<uint8_t> kb = random_bytes(c.type_kv, c.dk, ggml_nrows(k), rng);
    const std::vector<uint8_t> vb = random_bytes(c.type_kv, c.dv, ggml_nrows(v), rng);
    ggml_backend_tensor_set(q, qb.data(), 0, qb.size());
    ggml_backend_tensor_set(k, kb.data(), 0, kb.size());
    ggml_backend_tensor_set(v, vb.data(), 0, vb.size());
    if (m) {
        const std::vector<ggml_fp16_t> mb = random_mask(c.kv, c.nb, rng);
        ggml_backend_tensor_set(m, mb.data(), 0, mb.size() * sizeof(ggml_fp16_t));
    }

    bool ok = true;

    // on the same graph, by count: 0 runs neither, 1 and 2 run a only, 3 runs both
    for (int32_t count : { 0, 1, 2, 3 }) {
        set_count(n_active, count);

        set_sentinel(p_a);
        set_sentinel(p_b);

        if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
            return outcome::fail;
        }

        for (ggml_tensor * p : { p_a, p_b }) {
            const int32_t index = p == p_a ? 0 : 2;
            if (count > index && get_bytes(p) != get_bytes(p_ref)) {
                printf("    count %d: the active partial of index %d differs from the partial without a count\n", count, index);
                ok = false;
            }
            if (count <= index && !is_sentinel(p)) {
                printf("    count %d: the inactive partial of index %d wrote its result\n", count, index);
                ok = false;
            }
        }
    }

    return ok ? outcome::ok : outcome::fail;
}

int main() {
    ggml_backend_load_all();

    std::mt19937 rng(42);
    int n_checked = 0;
    int n_fail    = 0;
    for (const fa_case & c : CASES) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const outcome got = check(dev, c, rng);
            const char * verdict = got == outcome::ok ? "OK" : got == outcome::fail ? "FAIL" : "skipped";
            printf("  dk=%3lld dv=%3lld nb=%2lld kv=%3lld %-5s mask=%d %-12s: %s\n",
                    (long long) c.dk, (long long) c.dv, (long long) c.nb, (long long) c.kv,
                    ggml_type_name(c.type_kv), c.mask ? 1 : 0, ggml_backend_dev_name(dev), verdict);
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
