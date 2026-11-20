// checks that gathering row 0 of a table with GET_ROWS gives the same bits as casting a view of
//     that row to F32, on every backend device that supports both ops, for the types a per-layer
//     token table is stored in: a model that reads one row of a table may use either
//
// a type or device that does not support one of the two ops is skipped and reported

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const ggml_type TYPES[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL,
};

// the row width: a multiple of every type's block size
static const int64_t N_COLS = 1024;
static const int64_t N_ROWS = 4;

// a table of `type` filled from random floats, quantized through the type's own conversion
static std::vector<uint8_t> table_bytes(ggml_type type, std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> values(N_COLS * N_ROWS);
    for (float & v : values) {
        v = dist(rng);
    }
    std::vector<uint8_t> out(ggml_row_size(type, N_COLS) * N_ROWS);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), values.data(), out.size());
    } else {
        ggml_quantize_chunk(type, values.data(), out.data(), 0, N_ROWS, N_COLS, nullptr);
    }
    return out;
}

enum class outcome { same, differ, skipped };

static outcome check(ggml_backend_dev_t dev, ggml_type type, const std::vector<uint8_t> & bytes) {
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) {
        return outcome::skipped;
    }

    const ggml_init_params params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));

    ggml_tensor * table = ggml_new_tensor_2d(ctx.get(), type, N_COLS, N_ROWS);
    ggml_tensor * ids   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);

    ggml_tensor * gathered = ggml_get_rows(ctx.get(), table, ids);
    ggml_tensor * cast     = ggml_cast(ctx.get(), ggml_view_1d(ctx.get(), table, N_COLS, 0), GGML_TYPE_F32);

    if (!ggml_backend_supports_op(backend.get(), gathered) || !ggml_backend_supports_op(backend.get(), cast)) {
        return outcome::skipped;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, gathered);
    ggml_build_forward_expand(gf, cast);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buf) {
        return outcome::skipped;
    }

    ggml_backend_tensor_set(table, bytes.data(), 0, bytes.size());
    const int32_t row = 0;
    ggml_backend_tensor_set(ids, &row, 0, sizeof(row));

    if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
        return outcome::differ;
    }

    std::vector<float> a(N_COLS);
    std::vector<float> b(N_COLS);
    ggml_backend_tensor_get(gathered, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_get(cast,     b.data(), 0, b.size() * sizeof(float));
    return memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0 ? outcome::same : outcome::differ;
}

int main() {
    ggml_backend_load_all();

    std::mt19937 rng(42);
    int n_checked = 0;
    int n_fail    = 0;
    for (ggml_type type : TYPES) {
        const std::vector<uint8_t> bytes = table_bytes(type, rng);
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const outcome got = check(dev, type, bytes);
            const char * verdict = got == outcome::same ? "OK" : got == outcome::differ ? "FAIL" : "skipped";
            printf("  %-8s %-12s get_rows of row 0 == cast of its view: %s\n",
                    ggml_type_name(type), ggml_backend_dev_name(dev), verdict);
            n_checked += got == outcome::skipped ? 0 : 1;
            n_fail    += got == outcome::differ  ? 1 : 0;
        }
    }

    if (n_checked == 0) {
        printf("no device supports both ops for any type\n");
        return 1;
    }
    printf("%d checked, %d failed\n", n_checked, n_fail);
    return n_fail == 0 ? 0 : 1;
}
