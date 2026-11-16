// Tests for encoding ranges of a graph's nodes into command buffers the caller makes (ggml_backend_metal_encode_nodes),
// with the borrows that go with it: the device's queue, a tensor's Metal buffer, and the commit fence.
//
// The graph is a chain of layers. Layer k computes s_k = h_k + a_k, t_k = s_k + b_k and h_{k+1} = t_k * 0.5, so each
// layer holds one fusable run, the two adds, and the scale ends it. Every value is a small multiple of a power of two,
// so the arithmetic is exact with or without fusion. The reference is the stock graph compute in one command buffer
// (fusion statistics on, which makes the backend encode a graph in one command buffer).
//
// - ranges over the whole graph, or cut between the runs, in one or several caller buffers, compute every node
//   bitwise as the reference does and fuse the same runs;
// - a range boundary inside a run splits it, so one fusion fewer fires: the counters see the range boundaries; the
//   split run also writes its intermediate, which the fused kernel never stores, and every other node is unchanged;
// - the encode leaves the caller's buffer unenqueued and the backend does not wait for it;
// - a synchronize waits for a caller buffer committed before a commit fence, and only then.
//
// Everything needs a Metal device, so the binary skips without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include "metal-command-buffer.h"
#include "metal-nocopy-buffer.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <thread>
#include <utility>
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

constexpr int     kLayers       = 4;
constexpr int     kNodesPerLayer = 3;
constexpr int64_t kElems        = 1024;
constexpr int     kWaitMs       = 5000;

// a range of node indices [first, second), and the ranges encoded into one command buffer, in order
using node_range = std::pair<int, int>;
using buffer_ranges = std::vector<node_range>;

using bytes = std::vector<uint8_t>;

struct fixture {
    ggml_backend_reg_t    reg     = nullptr;
    ggml_backend_dev_t    dev     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_context        * ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph         * gf      = nullptr;
    void                * queue   = nullptr;

    ggml_backend_fusion_t finfo       = nullptr;
    fusion_stats_reset_t  stats_reset = nullptr;
    fusion_stats_get_t    stats_get   = nullptr;

    ggml_tensor * x = nullptr;
    std::vector<ggml_tensor *> a, b;

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

        // before the backend exists, so that it encodes a graph compute in one command buffer
        finfo = get(dev);
        stats_init(finfo);
        set_enabled(finfo, true);

        backend = ggml_backend_metal_init();
        if (backend == nullptr) {
            return false;
        }
        queue = ggml_backend_metal_device_get_mtl_queue(dev);

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(3*kLayers + 2*kLayers + 8) + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
        ggml_set_name(x, "x");
        ggml_tensor * h = x;
        for (int k = 0; k < kLayers; k++) {
            a.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems));
            b.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems));
            ggml_format_name(a[k], "a-%d", k);
            ggml_format_name(b[k], "b-%d", k);

            ggml_tensor * s = ggml_add(ctx, h, a[k]);
            ggml_tensor * t = ggml_add(ctx, s, b[k]);
            h = ggml_scale(ctx, t, 0.5f);
            ggml_format_name(s, "s-%d", k);
            ggml_format_name(t, "t-%d", k);
            ggml_format_name(h, "h-%d", k + 1);
        }
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, h);

        // every tensor has its own memory, so no fused output aliases another tensor
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buf != nullptr, "setup", "the tensors are allocated");
        check(ggml_graph_n_nodes(gf) == kNodesPerLayer*kLayers, "setup", "the graph has three nodes per layer");

        std::vector<float> v(kElems);
        auto set = [&](ggml_tensor * t, int seed) {
            for (int64_t i = 0; i < kElems; i++) {
                v[i] = (float) ((seed*7 + i) % 17) - 8.0f;
            }
            ggml_backend_tensor_set(t, v.data(), 0, kElems*sizeof(float));
        };
        set(x, 0);
        for (int k = 0; k < kLayers; k++) {
            set(a[k], 2*k + 1);
            set(b[k], 2*k + 2);
        }

        return true;
    }

    ~fixture() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    int n_nodes() const {
        return ggml_graph_n_nodes(gf);
    }

    void poison() const {
        for (int i = 0; i < n_nodes(); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            ggml_backend_tensor_memset(node, 0xff, 0, ggml_nbytes(node));
        }
    }

    std::vector<bytes> results() const {
        std::vector<bytes> res(n_nodes());
        for (int i = 0; i < n_nodes(); i++) {
            const ggml_tensor * node = ggml_graph_node(gf, i);
            res[i].resize(ggml_nbytes(node));
            ggml_backend_tensor_get(node, res[i].data(), 0, res[i].size());
        }
        return res;
    }

    // the bytes of s_0 = x + a_0, computed on the host; exact, since every value is a small integer
    bytes first_sum() const {
        std::vector<float> xs(kElems);
        std::vector<float> as(kElems);
        ggml_backend_tensor_get(x,    xs.data(), 0, kElems*sizeof(float));
        ggml_backend_tensor_get(a[0], as.data(), 0, kElems*sizeof(float));
        for (int64_t i = 0; i < kElems; i++) {
            xs[i] += as[i];
        }
        const uint8_t * p = (const uint8_t *) xs.data();
        return bytes(p, p + kElems*sizeof(float));
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

// the stock graph compute's results and fusion count
struct reference {
    std::vector<bytes> results;
    uint64_t fusions = 0;
};

static reference compute_reference(fixture & f) {
    f.poison();
    f.stats_reset(f.finfo);
    check(ggml_backend_graph_compute(f.backend, f.gf) == GGML_STATUS_SUCCESS, "reference", "the stock compute succeeds");
    return { f.results(), f.fusions() };
}

// encodes each buffer's ranges into a command buffer of its own, then commits the buffers in order and waits for them;
// true when every encode succeeded and every buffer completed
static bool run_buffers(fixture & f, const std::vector<buffer_ranges> & buffers) {
    bool ok = true;
    std::vector<void *> cmd_bufs;
    cmd_bufs.reserve(buffers.size());
    for (const buffer_ranges & ranges : buffers) {
        void * cb = test_metal_command_buffer_new(f.queue);
        cmd_bufs.push_back(cb);
        for (const node_range & r : ranges) {
            ok = ok && ggml_backend_metal_encode_nodes(f.backend, f.gf, r.first, r.second, cb) == GGML_STATUS_SUCCESS;
        }
    }
    for (void * cb : cmd_bufs) {
        test_metal_command_buffer_commit(cb);
    }
    for (void * cb : cmd_bufs) {
        test_metal_command_buffer_wait(cb);
        ok = ok && test_metal_command_buffer_completed(cb);
        test_metal_command_buffer_release(cb);
    }
    return ok;
}

static void test_ranges(fixture & f, const reference & ref) {
    struct row {
        const char *                scenario;
        std::vector<buffer_ranges>  buffers;
        uint64_t                    fusions;
        int                         unfused; // a node the stock compute fuses away and the ranges write, or -1
        const char *                what;
    };
    const int n = f.n_nodes();
    const int l = kNodesPerLayer;
    const std::vector<row> rows = {
        { "whole",       { { { 0, n } } },                                          ref.fusions,     -1,
            "one range over the graph" },
        { "between-runs", { { { 0, l }, { l, 2*l } }, { { 2*l, 3*l } }, { { 3*l, n } } }, ref.fusions, -1,
            "ranges cut between runs, two in one buffer" },
        { "inside-run",  { { { 0, 1 } }, { { 1, n } } },                            ref.fusions - 1, 0,
            "a range boundary inside the first run" },
    };

    for (const row & r : rows) {
        f.poison();
        f.stats_reset(f.finfo);
        check(run_buffers(f, r.buffers), r.scenario, "the caller's buffers encode and complete");
        std::vector<bytes> res = f.results();
        for (int i = 0; i < f.n_nodes(); i++) {
            if (res[i] != ref.results[i]) {
                printf("     %-20s differs at node %d (%s)\n", r.scenario, i, ggml_graph_node(f.gf, i)->name);
            }
        }
        if (r.unfused >= 0) {
            check(res[r.unfused] == f.first_sum(), r.scenario, "the split run writes the intermediate the stock compute fuses away");
            res[r.unfused] = ref.results[r.unfused];
        }
        check(res == ref.results, r.scenario, r.unfused >= 0 ? "every other node equals the stock compute bitwise"
                                                             : "every node equals the stock compute bitwise");
        check(f.fusions() == r.fusions, r.scenario, r.fusions == ref.fusions ? "the same runs fuse" : "the split run does not fuse");
        printf("     %-20s %s: %llu fusions (stock: %llu)\n", r.scenario, r.what,
                (unsigned long long) f.fusions(), (unsigned long long) ref.fusions);
    }
}

static void test_refused(fixture & f) {
    const char * sc = "refused";
    const int n = f.n_nodes();

    void * cb = test_metal_command_buffer_new(f.queue);
    const std::array<node_range, 3> bad = {{ { -1, n }, { 0, n + 1 }, { 2, 1 } }};
    bool refused = true;
    for (const node_range & r : bad) {
        refused = refused && ggml_backend_metal_encode_nodes(f.backend, f.gf, r.first, r.second, cb) == GGML_STATUS_FAILED;
    }
    check(refused, sc, "a range outside the graph fails");
    check(ggml_backend_metal_encode_nodes(f.backend, nullptr, 0, 0, cb) == GGML_STATUS_FAILED, sc, "a null graph fails");
    check(ggml_backend_metal_encode_nodes(f.backend, f.gf, 0, n, nullptr) == GGML_STATUS_FAILED, sc, "a null command buffer fails");

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu != nullptr) {
        check(ggml_backend_metal_encode_nodes(cpu, f.gf, 0, n, cb) == GGML_STATUS_FAILED, sc, "a backend that is not Metal fails");
        check(ggml_backend_metal_device_get_mtl_queue(ggml_backend_get_device(cpu)) == nullptr, sc, "a non-Metal device has no queue");
        ggml_backend_free(cpu);
    }
    check(f.queue != nullptr, sc, "the Metal device has its queue");
    check(!test_metal_command_buffer_enqueued(cb), sc, "the buffer stays unenqueued");
    test_metal_command_buffer_release(cb);
}

// runs ggml_backend_synchronize on a thread of its own
struct sync_probe {
    std::atomic<bool> done{false};
    std::thread       thread;

    explicit sync_probe(ggml_backend_t backend) : thread([this, backend] {
        ggml_backend_synchronize(backend);
        done = true;
    }) {}

    ~sync_probe() {
        thread.join();
    }

    // true when the synchronize returned within ms
    bool returns_within(int ms) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (!done && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return done;
    }
};

// the encode neither enqueues the caller's buffer nor makes the backend wait for it
static void test_unowned(fixture & f, const reference & ref) {
    const char * sc = "unowned";

    ggml_backend_synchronize(f.backend);
    f.poison();

    void * cb = test_metal_command_buffer_new(f.queue);
    check(ggml_backend_metal_encode_nodes(f.backend, f.gf, 0, f.n_nodes(), cb) == GGML_STATUS_SUCCESS, sc, "the graph encodes");
    check(!test_metal_command_buffer_enqueued(cb), sc, "the caller's buffer is not enqueued");
    {
        sync_probe probe(f.backend);
        const bool returned = probe.returns_within(kWaitMs);
        check(returned, sc, "a synchronize does not wait for the caller's buffer");
        if (!returned) {
            // the backend waits for the uncommitted buffer: commit it so that the wait ends
            test_metal_command_buffer_commit(cb);
        }
    }
    if (!test_metal_command_buffer_enqueued(cb)) {
        test_metal_command_buffer_commit(cb);
    }
    test_metal_command_buffer_wait(cb);
    check(test_metal_command_buffer_completed(cb) && f.results() == ref.results, sc, "committed by the caller, it computes the graph");
    test_metal_command_buffer_release(cb);
}

// a synchronize waits for a caller buffer committed before a commit fence; the buffer waits on an event the host
// signals, so the test sees the synchronize blocked behind it
static void test_fence(fixture & f, const reference & ref) {
    const char * sc = "fence";

    void * ev = test_metal_shared_event_new(f.dev);
    check(ev != nullptr, sc, "a shared event is made");
    if (ev == nullptr) {
        return;
    }

    ggml_backend_synchronize(f.backend);
    f.poison();

    void * cb = test_metal_command_buffer_new(f.queue);
    test_metal_command_buffer_encode_wait(cb, ev, 1);
    check(ggml_backend_metal_encode_nodes(f.backend, f.gf, 0, f.n_nodes(), cb) == GGML_STATUS_SUCCESS, sc, "the graph encodes behind the gate");
    test_metal_command_buffer_commit(cb);

    {
        sync_probe probe(f.backend);
        check(probe.returns_within(kWaitMs) && !test_metal_command_buffer_completed(cb), sc,
                "without a fence, a synchronize returns while the caller's buffer waits");
    }

    ggml_backend_metal_commit_fence(f.backend);
    {
        sync_probe probe(f.backend);
        check(!probe.returns_within(100), sc, "after a fence, a synchronize waits behind the caller's buffer");
        test_metal_shared_event_signal(ev, 1);
        check(probe.returns_within(kWaitMs), sc, "it returns once the buffer can run");
    }
    check(test_metal_command_buffer_completed(cb), sc, "the caller's buffer completed before the synchronize returned");
    check(f.results() == ref.results, sc, "and its results are in place");

    test_metal_command_buffer_wait(cb);
    test_metal_command_buffer_release(cb);
    test_metal_shared_event_release(ev);
}

// a tensor's Metal buffer and offset, for a view and for a buffer over caller ranges
static void test_tensor_buffer(fixture & f) {
    const char * sc = "tensor-buffer";

    {
        const size_t view_offs = 64*sizeof(float);
        ggml_tensor * v = ggml_view_1d(f.ctx, f.x, 16, view_offs);
        ggml_set_name(v, "x-view");
        check(ggml_backend_view_init(v) == GGML_STATUS_SUCCESS, sc, "a view is initialized");

        size_t offs_x = SIZE_MAX;
        size_t offs_v = SIZE_MAX;
        void * metal_x = ggml_backend_metal_tensor_mtl_buffer(f.x, &offs_x);
        void * metal_v = ggml_backend_metal_tensor_mtl_buffer(v,   &offs_v);
        check(metal_x != nullptr && metal_v == metal_x, sc, "a view resolves to its source's Metal buffer");
        check(offs_v == offs_x + view_offs, sc, "at the source's offset plus the view's");
    }

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t sub  = 4*page;
    void * mem = mmap(nullptr, 2*sub, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    check(mem != MAP_FAILED, sc, "host memory for two ranges is mapped");
    if (mem == MAP_FAILED) {
        return;
    }

    std::array<void *, 2> metal = {
        test_metal_nocopy_buffer_new(f.dev, mem, sub),
        test_metal_nocopy_buffer_new(f.dev, (uint8_t *) mem + sub, sub),
    };
    const std::array<ggml_backend_metal_range, 2> ranges = {{ { sub, metal[0] }, { sub, metal[1] } }};
    ggml_backend_buffer_t rb = ggml_backend_metal_buffer_from_ranges(f.dev, mem, 2*sub, ranges.data(), 2);
    check(rb != nullptr, sc, "a buffer over two ranges is created");
    if (rb != nullptr) {
        const std::array<size_t, 2> at = { 256, sub + 512 };
        bool ok = true;
        for (int i = 0; i < 2; i++) {
            ggml_tensor * t = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 16);
            ggml_format_name(t, "ranged-%d", i);
            ok = ok && ggml_backend_tensor_alloc(rb, t, (uint8_t *) mem + at[i]) == GGML_STATUS_SUCCESS;

            size_t offs = SIZE_MAX;
            void * m = ggml_backend_metal_tensor_mtl_buffer(t, &offs);
            ok = ok && m == metal[i] && offs == at[i] - i*sub;
        }
        check(ok, sc, "a tensor in a range resolves to the range's Metal buffer at its offset in the range");
        ggml_backend_buffer_free(rb);
    }

    for (void * m : metal) {
        if (m != nullptr) {
            test_metal_buffer_release(m);
        }
    }
    munmap(mem, 2*sub);
}

int main(void) {
    fixture f;
    if (!f.init()) {
        printf("SKIP no Metal device\n");
        return 0;
    }
    if (f.backend == nullptr || f.buf == nullptr) {
        printf("fixture setup failed\n");
        return 1;
    }

    const reference ref = compute_reference(f);
    check(ref.fusions == (uint64_t) kLayers, "reference", "the stock compute fuses the run of every layer");

    test_ranges(f, ref);
    test_refused(f);
    test_unowned(f, ref);
    test_fence(f, ref);
    test_tensor_buffer(f);

    if (g_n_fail > 0) {
        printf("%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
