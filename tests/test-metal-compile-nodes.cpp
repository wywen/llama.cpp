// Tests for compiling the pipelines of a range of a graph's nodes without encoding them (ggml_backend_metal_compile_nodes).
//
// The pipelines a compile looks up are the ones an encode of the same nodes looks up, read through the device's count
// of compiled pipelines (ggml_backend_metal_device_n_pipelines), which grows by one for each pipeline compiled:
//
// - a graph compiled first, over tensors that have no memory, compiles every pipeline a compute of the same graph sets:
//   the compute compiles nothing more; and a compile of it again compiles nothing;
// - a graph computed first compiles every pipeline a compile of the same graph looks up: the compile compiles nothing;
// - tensors in a buffer over caller ranges with no Metal buffer compile without the range's grow hook being called;
// - fusion statistics do not count a compile;
// - a range outside the graph and a backend that is not Metal fail.
//
// Each graph holds a fusable run (an RMS norm and the multiply after it) and matrix products of distinct types and widths,
// so the walk takes the same fusion and kernel choices as the encode. The second graph's product and tail are kernels
// the first does not set, so each order is read on pipelines of its own. Everything needs a Metal device, so the binary
// skips without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

static int g_n_fail = 0;

static void check(bool ok, const char * scenario, const char * what) {
    printf("%s %-20s %s\n", ok ? "PASS" : "FAIL", scenario, what);
    if (!ok) {
        g_n_fail++;
    }
}

// generic fusion debugging API, resolved through the backend registry's get_proc_address
typedef void * ggml_backend_fusion_t;

typedef ggml_backend_fusion_t (*fusion_get_t)        (ggml_backend_dev_t);
typedef void                  (*fusion_stats_init_t) (ggml_backend_fusion_t);
typedef void                  (*fusion_stats_reset_t)(ggml_backend_fusion_t);
typedef int                   (*fusion_stats_get_t)  (ggml_backend_fusion_t, const char **, uint64_t *, int);
typedef void                  (*fusion_set_enabled_t)(ggml_backend_fusion_t, bool);

constexpr int64_t kRows = 64;
constexpr int64_t kCols = 32;

// which of the two kernel sets a graph uses
enum class graph_kind {
    // f32 weights, a four-column product and a soft max
    f32_wide,
    // f16 weights, a one-column product and a scale
    f16_narrow,
};

// a graph of `kind` in a context of its own; with `allocate`, every tensor has memory of the backend, cleared
struct built_graph {
    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph         * gf  = nullptr;

    built_graph(graph_kind kind, ggml_backend_t backend, bool allocate) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*16 + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        const bool    f32   = kind == graph_kind::f32_wide;
        const int64_t width = f32 ? 4 : 1;

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kRows, width);
        ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kRows);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, kRows, kCols);

        ggml_tensor * n = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), g);
        ggml_tensor * p = ggml_mul_mat(ctx, w, n);
        ggml_tensor * out = f32 ? ggml_soft_max(ctx, p) : ggml_scale(ctx, p, 0.5f);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        if (allocate) {
            buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (buf != nullptr) {
                ggml_backend_buffer_clear(buf, 0);
            }
        }
    }

    ~built_graph() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
    }

    int n_nodes() const {
        return ggml_graph_n_nodes(gf);
    }
};

struct fixture {
    ggml_backend_reg_t reg     = nullptr;
    ggml_backend_dev_t dev     = nullptr;
    ggml_backend_t     backend = nullptr;

    ggml_backend_fusion_t finfo       = nullptr;
    fusion_stats_reset_t  stats_reset = nullptr;
    fusion_stats_get_t    stats_get   = nullptr;

    // false when there is no Metal device
    bool init() {
        reg = ggml_backend_metal_reg();
        if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
            return false;
        }
        dev = ggml_backend_reg_dev_get(reg, 0);

        auto get         = (fusion_get_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_get");
        auto stats_init  = (fusion_stats_init_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_init");
        auto set_enabled = (fusion_set_enabled_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_set_enabled");
        stats_reset = (fusion_stats_reset_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_reset");
        stats_get   = (fusion_stats_get_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_get");
        check(get && stats_init && set_enabled && stats_reset && stats_get, "setup", "the fusion debugging API is exported");
        if (!(get && stats_init && set_enabled && stats_reset && stats_get)) {
            return true;
        }

        // before the backend exists, so that it computes a graph in one command buffer, as one range is encoded
        finfo = get(dev);
        stats_init(finfo);
        set_enabled(finfo, true);

        backend = ggml_backend_metal_init();
        return backend != nullptr;
    }

    ~fixture() {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    size_t n_pipelines() const {
        return ggml_backend_metal_device_n_pipelines(dev);
    }

    uint64_t fusions() const {
        const int n = stats_get(finfo, nullptr, nullptr, 0);
        std::vector<const char *> labels(n);
        std::vector<uint64_t> counts(n);
        stats_get(finfo, labels.data(), counts.data(), n);
        uint64_t total = 0;
        for (uint64_t c : counts) {
            total += c;
        }
        return total;
    }
};

// compiled first, the graph's pipelines are every one its compute sets
static void test_compile_first(fixture & f) {
    const char * sc = "compile-first";

    const built_graph bare(graph_kind::f32_wide, f.backend, false);
    const size_t before = f.n_pipelines();
    f.stats_reset(f.finfo);
    check(ggml_backend_metal_compile_nodes(f.backend, bare.gf, 0, bare.n_nodes()) == GGML_STATUS_SUCCESS, sc,
            "a graph whose tensors have no memory compiles");
    const size_t compiled = f.n_pipelines();
    check(compiled > before, sc, "the compile compiles the graph's pipelines");
    check(f.fusions() == 0, sc, "fusion statistics do not count the compile");

    check(ggml_backend_metal_compile_nodes(f.backend, bare.gf, 0, bare.n_nodes()) == GGML_STATUS_SUCCESS &&
            f.n_pipelines() == compiled, sc, "compiling it again compiles nothing");

    const built_graph real(graph_kind::f32_wide, f.backend, true);
    check(real.buf != nullptr, sc, "the computed copy is allocated");
    check(ggml_backend_graph_compute(f.backend, real.gf) == GGML_STATUS_SUCCESS, sc, "the computed copy computes");
    check(f.fusions() > 0, sc, "the compute fuses the run the compile walked");
    check(f.n_pipelines() == compiled, sc, "the compute compiles no pipeline the compile did not");
    printf("     %-20s %zu pipelines compiled by the compile, none by the compute\n", sc, compiled - before);
}

// computed first, the graph's pipelines are every one its compile looks up
static void test_compute_first(fixture & f) {
    const char * sc = "compute-first";

    const built_graph real(graph_kind::f16_narrow, f.backend, true);
    check(real.buf != nullptr, sc, "the computed copy is allocated");
    const size_t before = f.n_pipelines();
    check(ggml_backend_graph_compute(f.backend, real.gf) == GGML_STATUS_SUCCESS, sc, "the graph computes");
    const size_t computed = f.n_pipelines();
    check(computed > before, sc, "the compute compiles the graph's pipelines");

    const built_graph bare(graph_kind::f16_narrow, f.backend, false);
    check(ggml_backend_metal_compile_nodes(f.backend, bare.gf, 0, bare.n_nodes()) == GGML_STATUS_SUCCESS, sc,
            "the graph without memory compiles");
    check(f.n_pipelines() == computed, sc, "the compile looks up no pipeline the compute did not set");
}

static int g_grow_calls = 0;

static void count_grow(void * /*user_data*/, int /*i*/, size_t /*bytes*/) {
    g_grow_calls++;
}

// tensors in a range with no Metal buffer compile without the range's grow hook
static void test_unbacked_range(fixture & f) {
    const char * sc = "unbacked-range";

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t size = 4*page;
    void * mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    check(mem != MAP_FAILED, sc, "host memory for one range is mapped");
    if (mem == MAP_FAILED) {
        return;
    }

    const std::array<ggml_backend_metal_range, 1> ranges = {{ { size, nullptr } }};
    ggml_backend_buffer_t rb = ggml_backend_metal_buffer_from_ranges(f.dev, mem, size, ranges.data(), 1);
    check(rb != nullptr, sc, "a buffer over a range with no Metal buffer is created");
    if (rb != nullptr) {
        check(ggml_backend_metal_buffer_set_range_grow(rb, count_grow, nullptr), sc, "the grow hook is installed");

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*4 + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        ggml_tensor * c = ggml_sub(ctx, a, b);
        bool ok = ggml_backend_tensor_alloc(rb, a, mem) == GGML_STATUS_SUCCESS;
        ok = ok && ggml_backend_tensor_alloc(rb, b, (uint8_t *) mem + page) == GGML_STATUS_SUCCESS;
        ok = ok && ggml_backend_tensor_alloc(rb, c, (uint8_t *) mem + 2*page) == GGML_STATUS_SUCCESS;
        check(ok, sc, "the tensors are placed in the range");

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);
        check(ggml_backend_metal_compile_nodes(f.backend, gf, 0, ggml_graph_n_nodes(gf)) == GGML_STATUS_SUCCESS, sc,
                "the graph compiles");
        check(g_grow_calls == 0, sc, "the range's grow hook is not called");

        ggml_free(ctx);
        ggml_backend_buffer_free(rb);
    }
    munmap(mem, size);
}

static void test_refused(fixture & f) {
    const char * sc = "refused";

    const built_graph bare(graph_kind::f32_wide, f.backend, false);
    const int n = bare.n_nodes();
    const std::array<std::array<int, 2>, 3> bad = {{ { -1, n }, { 0, n + 1 }, { 2, 1 } }};
    bool refused = true;
    for (const std::array<int, 2> & r : bad) {
        refused = refused && ggml_backend_metal_compile_nodes(f.backend, bare.gf, r[0], r[1]) == GGML_STATUS_FAILED;
    }
    check(refused, sc, "a range outside the graph fails");
    check(ggml_backend_metal_compile_nodes(f.backend, nullptr, 0, 0) == GGML_STATUS_FAILED, sc, "a null graph fails");

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu != nullptr) {
        check(ggml_backend_metal_compile_nodes(cpu, bare.gf, 0, n) == GGML_STATUS_FAILED, sc, "a backend that is not Metal fails");
        check(ggml_backend_metal_device_n_pipelines(ggml_backend_get_device(cpu)) == 0, sc, "a non-Metal device counts none");
        ggml_backend_free(cpu);
    }
}

int main(void) {
    fixture f;
    if (!f.init()) {
        printf("SKIP no Metal device\n");
        return 0;
    }
    if (f.backend == nullptr || f.finfo == nullptr) {
        printf("fixture setup failed\n");
        return 1;
    }

    test_compile_first(f);
    test_compute_first(f);
    test_unbacked_range(f);
    test_refused(f);

    if (g_n_fail > 0) {
        printf("%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
