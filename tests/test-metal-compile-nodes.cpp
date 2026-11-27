// Tests for compiling the pipelines of a range of a graph's nodes without encoding them (ggml_backend_metal_compile_nodes).
//
// The pipelines a compile looks up are the ones an encode of the same nodes looks up, read through a count of the
// pipelines compiled: the Metal library logs one debug line for each pipeline it compiles, which a log callback
// installed with ggml_log_set counts:
//
// - a graph compiled first, over tensors that have no memory, compiles every pipeline a compute of the same graph sets:
//   the compute compiles nothing more; and a compile of it again compiles nothing;
// - a compile cannot read whether the buffers allow a fusion, so it also compiles the kernels of the nodes unfused: a
//   graph computed first, both fused and with fusion off, compiles every pipeline a compile of it looks up, while a
//   graph computed fused only may leave the compile the unfused kernels to compile;
// - a chain of additions, which fuses only addends in one Metal buffer, compiles without memory, and a compute of it
//   with the addends in one buffer, in two or in three compiles nothing more;
// - tensors in a buffer over caller ranges with no Metal buffer compile;
// - fusion statistics do not count a compile;
// - a range outside the graph and a backend that is not Metal fail.
//
// Each graph holds a fusable run (an RMS norm and the multiply after it) and matrix products of distinct types and widths.
// The second graph's product and tail are kernels the first does not set, and the addition chain's kernels are its own,
// so each scenario is read on pipelines of its own. Everything needs a Metal device, so the binary
// skips without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// the pipelines compiled since the log callback was installed: the Metal library logs this line for each one
static std::atomic<size_t> g_n_compiled{0};

struct log_chain {
    ggml_log_callback prev      = nullptr;
    void *            prev_data = nullptr;
};

static void count_compiles(ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_DEBUG && strstr(text, "compiling pipeline:") != nullptr) {
        g_n_compiled++;
    }
    const log_chain & chain = *(const log_chain *) user_data;
    if (chain.prev != nullptr) {
        chain.prev(level, text, chain.prev_data);
    }
}

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
    fusion_set_enabled_t  set_enabled = nullptr;

    log_chain log;

    // false when there is no Metal device
    bool init() {
        ggml_log_get(&log.prev, &log.prev_data);
        ggml_log_set(count_compiles, &log);

        reg = ggml_backend_metal_reg();
        if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
            return false;
        }
        dev = ggml_backend_reg_dev_get(reg, 0);

        auto get         = (fusion_get_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_get");
        auto stats_init  = (fusion_stats_init_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_init");
        set_enabled = (fusion_set_enabled_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_set_enabled");
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
        ggml_log_set(log.prev, log.prev_data);
    }

    size_t n_pipelines() const {
        return g_n_compiled.load();
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

// computed first, fused and with fusion off, the graph's pipelines are every one its compile looks up; the compile also
// looks up the kernels of the fused nodes alone, which only the compute with fusion off sets
static void test_compute_first(fixture & f) {
    const char * sc = "compute-first";

    const built_graph real(graph_kind::f16_narrow, f.backend, true);
    check(real.buf != nullptr, sc, "the computed copy is allocated");
    const size_t before = f.n_pipelines();
    check(ggml_backend_graph_compute(f.backend, real.gf) == GGML_STATUS_SUCCESS, sc, "the graph computes");
    const size_t computed = f.n_pipelines();
    check(computed > before, sc, "the compute compiles the graph's pipelines");

    f.set_enabled(f.finfo, false);
    check(ggml_backend_graph_compute(f.backend, real.gf) == GGML_STATUS_SUCCESS, sc, "the graph computes unfused");
    f.set_enabled(f.finfo, true);
    const size_t both = f.n_pipelines();

    const built_graph bare(graph_kind::f16_narrow, f.backend, false);
    check(ggml_backend_metal_compile_nodes(f.backend, bare.gf, 0, bare.n_nodes()) == GGML_STATUS_SUCCESS, sc,
            "the graph without memory compiles");
    check(f.n_pipelines() == both, sc, "the compile looks up no pipeline the fused and unfused computes did not set");
    printf("     %-20s %zu pipelines compiled by the unfused compute, none by the compile\n", sc, both - computed);
}

// how the three addends of an addition chain are placed: in which buffer each is, or in none
enum class addend_layout {
    none,
    one_buffer, // a b c
    two_buffers, // a b | c
    three_buffers, // a | b | c
};

// x + a + b + c with the addends in contexts of their own, one per buffer of `layout`; x and the sums are in one more
struct add_chain_graph {
    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
    ggml_cgraph * gf        = nullptr;
    bool          allocated = true;

    add_chain_graph(addend_layout layout, ggml_backend_t backend) {
        const std::array<int, 3> group = layout == addend_layout::three_buffers ? std::array<int, 3>{ 0, 1, 2 } :
                                         layout == addend_layout::two_buffers   ? std::array<int, 3>{ 0, 0, 1 } :
                                                                                  std::array<int, 3>{ 0, 0, 0 };
        const int n_groups = group[2] + 1;

        for (int i = 0; i <= n_groups; i++) {
            ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead()*8 + (i == 0 ? ggml_graph_overhead() : 0),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctxs.push_back(ggml_init(params));
        }

        ggml_tensor * cur = ggml_new_tensor_2d(ctxs[0], GGML_TYPE_F32, kRows, 4);
        for (int i = 0; i < 3; i++) {
            ggml_tensor * addend = ggml_new_tensor_2d(ctxs[1 + group[i]], GGML_TYPE_F32, kRows, 4);
            cur = ggml_add(ctxs[0], cur, addend);
        }

        gf = ggml_new_graph(ctxs[0]);
        ggml_build_forward_expand(gf, cur);

        if (layout != addend_layout::none) {
            for (ggml_context * ctx : ctxs) {
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                if (buf == nullptr) {
                    allocated = false;
                    continue;
                }
                ggml_backend_buffer_clear(buf, 0);
                bufs.push_back(buf);
            }
        }
    }

    ~add_chain_graph() {
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
        for (ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
    }

    int n_nodes() const {
        return ggml_graph_n_nodes(gf);
    }
};

// the chain fuses only addends in one Metal buffer, which a compile cannot read, so it compiles the kernels of every
// layout: the chain of three, of two, and the addition alone
static void test_add_chain(fixture & f) {
    const char * sc = "add-chain";

    const add_chain_graph bare(addend_layout::none, f.backend);
    const size_t before = f.n_pipelines();
    check(ggml_backend_metal_compile_nodes(f.backend, bare.gf, 0, bare.n_nodes()) == GGML_STATUS_SUCCESS, sc,
            "a chain whose tensors have no buffer compiles");
    const size_t compiled = f.n_pipelines();
    check(compiled > before, sc, "the compile compiles the chain's pipelines");

    struct layout_case {
        addend_layout layout;
        bool          fuses;
        const char *  what;
    };
    const std::array<layout_case, 3> cases = {{
        { addend_layout::one_buffer,    true,  "addends in one buffer"    },
        { addend_layout::two_buffers,   true,  "addends in two buffers"   },
        { addend_layout::three_buffers, false, "addends in three buffers" },
    }};

    for (const layout_case & c : cases) {
        const add_chain_graph real(c.layout, f.backend);
        char what[128];
        snprintf(what, sizeof(what), "%s: allocated", c.what);
        check(real.allocated, sc, what);
        if (!real.allocated) {
            continue;
        }

        f.stats_reset(f.finfo);
        snprintf(what, sizeof(what), "%s: computes", c.what);
        check(ggml_backend_graph_compute(f.backend, real.gf) == GGML_STATUS_SUCCESS, sc, what);
        snprintf(what, sizeof(what), "%s: %s", c.what, c.fuses ? "fuses" : "does not fuse");
        check((f.fusions() > 0) == c.fuses, sc, what);
        snprintf(what, sizeof(what), "%s: compiles no pipeline the compile did not", c.what);
        check(f.n_pipelines() == compiled, sc, what);
    }

    check(ggml_backend_metal_compile_nodes(f.backend, bare.gf, 0, bare.n_nodes()) == GGML_STATUS_SUCCESS &&
            f.n_pipelines() == compiled, sc, "compiling it again compiles nothing");
    printf("     %-20s %zu pipelines compiled by the compile, none by the computes\n", sc, compiled - before);
}

// tensors in a range with no Metal buffer compile
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
    test_add_chain(f);
    test_unbacked_range(f);
    test_refused(f);

    if (g_n_fail > 0) {
        printf("%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
