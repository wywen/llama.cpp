// Tests for host cuts in the Metal backend's boundary-scheduled compute path.
//
// The correctness graph is a chain of layers. Layer k computes r_k = h_k + a_k, the host cut, and
// h_{k+1} = r_k * s, where s is one tensor shared by every layer. The callback at cut k reads r_k
// from host memory and writes s from it, alternating between {1, 2} and {3, 4} per layer, so a
// segment that ran before the callback, or a callback that read a stale r_k, changes the result.
// Every value is a small integer, so the arithmetic is exact on host and device alike. The
// reference is the same chain without cuts, each layer reading its own tensor preset to the slot
// values the callbacks wrote; the results must match bitwise.
//
// The timing case runs a chain of 40 layers of two matrix-vector products each, with a host cut
// between them, and prints the cost of a cut. It asserts nothing about the times.
//
// Everything needs a Metal device, so the binary skips without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_n_fail = 0;

static void check(bool ok, const char * scenario, const char * what) {
    printf("%s %-28s %s\n", ok ? "PASS" : "FAIL", scenario, what);
    if (!ok) {
        g_n_fail++;
    }
}

constexpr int     kLayers   = 6;
constexpr int64_t kElems    = 256;
constexpr size_t  kBytes    = kElems * sizeof(float);
constexpr float   kSentinel = -7.0f;
constexpr size_t  kPage     = 16384;

using host_t = std::vector<float>;

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The slot value layer k derives from a router output: {1, 2} on even layers, {3, 4} on odd ones.
static float slot_value(int k, float r) {
    return (float) ((k % 2 == 0 ? 1 : 3) + ((int64_t) r) % 2);
}

// The tensors of a context, allocated in caller-owned host memory wrapped as a Metal buffer, so
// the host reads and writes them directly while the graph runs.
struct host_arena {
    void                * mem = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    bool alloc(ggml_backend_t backend, ggml_context * ctx) {
        const size_t align = 256;
        size_t size = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == nullptr) {
                size += (ggml_nbytes(t) + align - 1) / align * align;
            }
        }
        size = (size + kPage - 1) / kPage * kPage;
        if (posix_memalign(&mem, kPage, size) != 0) {
            mem = nullptr;
            return false;
        }
        memset(mem, 0, size);
        buf = ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(backend), mem, size, size);
        if (buf == nullptr) {
            return false;
        }
        ggml_tallocr talloc = ggml_tallocr_new(buf);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data == nullptr && ggml_tallocr_alloc(&talloc, t) != GGML_STATUS_SUCCESS) {
                return false;
            }
        }
        return true;
    }

    void release() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        free(mem);
        mem = nullptr;
    }
};

static float * host_data(const ggml_tensor * t) {
    return (float *) t->data;
}

static void fill(ggml_tensor * t, float v) {
    std::fill(host_data(t), host_data(t) + ggml_nelements(t), v);
}

static host_t snapshot(const ggml_tensor * t) {
    return host_t(host_data(t), host_data(t) + ggml_nelements(t));
}

static bool same_bits(const host_t & a, const host_t & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static ggml_status compute(ggml_backend_t backend, ggml_cgraph * g) {
    const ggml_status st = ggml_backend_graph_compute(backend, g);
    ggml_backend_synchronize(backend);
    return st;
}

static void clear_schedules(ggml_backend_t backend) {
    ggml_backend_metal_set_host_cuts(backend, 0, nullptr, nullptr, 0, nullptr, nullptr);
    ggml_backend_metal_set_boundary_schedule(backend, 0, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr);
    ggml_backend_metal_clear_encode_window(backend);
}

// The layer chain, with and without cuts, on its own backend: a failing case leaves its backend
// in the error state for good.
struct chain {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx     = nullptr;
    host_arena     arena;

    ggml_tensor * x = nullptr;
    ggml_tensor * s = nullptr;
    std::vector<ggml_tensor *> a, s_ref, r, e, r_ref, e_ref;

    ggml_cgraph * gf     = nullptr; // reads s at every layer
    ggml_cgraph * gf_ref = nullptr; // reads s_ref[k] at layer k

    ggml_metal_event_t ev   = nullptr; // the host cuts'
    ggml_metal_event_t gate = nullptr; // a boundary schedule's waits
    ggml_metal_event_t aux  = nullptr; // a signal the callback commits on the queue
    uint64_t           aux_val = 0;

    bool init() {
        backend = ggml_backend_metal_init();
        if (backend == nullptr) {
            return false;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(8*kLayers + 16) + 2*ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
        s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
        ggml_tensor * h     = x;
        ggml_tensor * h_ref = x;
        for (int k = 0; k < kLayers; k++) {
            a.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems));
            s_ref.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems));

            r.push_back(ggml_add(ctx, h, a[k]));
            ggml_format_name(r[k], "router.%d", k);
            e.push_back(ggml_mul(ctx, r[k], s));
            ggml_format_name(e[k], "experts.%d", k);
            h = e[k];

            r_ref.push_back(ggml_add(ctx, h_ref, a[k]));
            e_ref.push_back(ggml_mul(ctx, r_ref[k], s_ref[k]));
            h_ref = e_ref[k];
        }
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, h);
        gf_ref = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_ref, h_ref);

        if (!arena.alloc(backend, ctx)) {
            return false;
        }

        for (int64_t j = 0; j < kElems; j++) {
            host_data(x)[j] = (float) (j % 7);
        }
        for (int k = 0; k < kLayers; k++) {
            for (int64_t j = 0; j < kElems; j++) {
                host_data(a[k])[j] = (float) ((j + k) % 3);
            }
        }

        ev   = ggml_backend_metal_event_init(backend);
        gate = ggml_backend_metal_event_init(backend);
        aux  = ggml_backend_metal_event_init(backend);
        return true;
    }

    ~chain() {
        if (backend != nullptr) {
            clear_schedules(backend);
            ggml_backend_synchronize(backend);
            ggml_backend_metal_event_free(backend, ev);
            ggml_backend_metal_event_free(backend, gate);
            ggml_backend_metal_event_free(backend, aux);
        }
        arena.release();
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    void seed() {
        fill(s, kSentinel);
        for (int k = 0; k < kLayers; k++) {
            fill(r[k], kSentinel);
            fill(e[k], kSentinel);
        }
    }
};

// What the host cut callback does, and what it saw.
struct cut_state {
    chain * c = nullptr;

    // r_k as the host replays it, and the slot values the callbacks write, per layer
    std::vector<host_t> r_expected;
    std::vector<host_t> slots;

    std::vector<int> fired;
    bool r_seen_ok = true;
    bool queue_ok  = true;

    bool open_gate   = false; // signal the gate to i + 1, which the next segment waits for
    bool queue_round = false; // commit a signal on the backend's queue and wait for it
    int  fail_at     = -1;    // return false at this cut
    int  saturate_at = -1;    // saturate the host cut event at this cut

    explicit cut_state(chain & ch) : c(&ch) {
        r_expected.assign(kLayers, host_t(kElems));
        slots.assign(kLayers, host_t(kElems));
        host_t h = snapshot(c->x);
        for (int k = 0; k < kLayers; k++) {
            for (int64_t j = 0; j < kElems; j++) {
                const float rv = h[j] + host_data(c->a[k])[j];
                r_expected[k][j] = rv;
                slots[k][j]      = slot_value(k, rv);
                h[j] = rv * slots[k][j];
            }
        }
    }

    void reset() {
        fired.clear();
        r_seen_ok = true;
        queue_ok  = true;
    }
};

static bool on_cut(void * user_data, int i, ggml_tensor * node) {
    cut_state & cs = *(cut_state *) user_data;
    chain & c = *cs.c;
    cs.fired.push_back(i);

    // the router output, read straight from host memory
    const float * rv = host_data(c.r[i]);
    if (node != c.r[i] || memcmp(rv, cs.r_expected[i].data(), kBytes) != 0) {
        cs.r_seen_ok = false;
    }
    float * sv = host_data(c.s);
    for (int64_t j = 0; j < kElems; j++) {
        sv[j] = slot_value(i, rv[j]);
    }

    if (cs.queue_round) {
        ggml_backend_metal_event_signal(c.backend, c.aux, ++c.aux_val);
        if (!ggml_backend_metal_event_host_wait(c.aux, c.aux_val, 5000)) {
            cs.queue_ok = false;
        }
    }
    if (cs.open_gate) {
        ggml_backend_metal_event_host_signal(c.gate, (uint64_t) i + 1);
    }
    if (i == cs.saturate_at) {
        ggml_backend_metal_event_host_signal(c.ev, UINT64_MAX);
    }
    return i != cs.fail_at;
}

static void set_host_cuts(chain & c, cut_state & cs, uint64_t base) {
    ggml_backend_metal_set_host_cuts(c.backend, kLayers, c.r.data(), c.ev, base, on_cut, &cs);
}

static std::vector<int> range(int lo, int hi) {
    std::vector<int> res;
    for (int k = lo; k < hi; k++) {
        res.push_back(k);
    }
    return res;
}

// The uncut chain, each layer reading its own preset slot.
struct reference {
    std::vector<host_t> r, e;

    bool run(chain & c, const cut_state & cs) {
        clear_schedules(c.backend);
        for (int k = 0; k < kLayers; k++) {
            memcpy(host_data(c.s_ref[k]), cs.slots[k].data(), kBytes);
            fill(c.r_ref[k], kSentinel);
            fill(c.e_ref[k], kSentinel);
        }
        if (compute(c.backend, c.gf_ref) != GGML_STATUS_SUCCESS) {
            return false;
        }
        r.clear();
        e.clear();
        for (int k = 0; k < kLayers; k++) {
            r.push_back(snapshot(c.r_ref[k]));
            e.push_back(snapshot(c.e_ref[k]));
        }
        return true;
    }

    // layers from `lo` on match bitwise
    bool matches(const chain & c, int lo) const {
        for (int k = lo; k < kLayers; k++) {
            if (!same_bits(snapshot(c.r[k]), r[k]) || !same_bits(snapshot(c.e[k]), e[k])) {
                return false;
            }
        }
        return true;
    }
};

// No gates: every segment after a cut is pre-encoded and committed after the callback, which
// itself commits work on the backend's queue and waits for it.
static void test_ordering(chain & c, cut_state & cs, const reference & ref) {
    const char * sc = "ordering, pre-encoded";
    cs.queue_round = true;
    const uint64_t base = 100;
    set_host_cuts(c, cs, base);

    for (int round = 0; round < 2; round++) {
        cs.reset();
        c.seed();
        check(compute(c.backend, c.gf) == GGML_STATUS_SUCCESS, sc, round == 0 ? "compute succeeds" : "a second compute succeeds");
        check(cs.fired == range(0, kLayers), sc, "every cut fires once, in order");
        check(cs.r_seen_ok, sc, "each callback reads its segment's output");
        check(cs.queue_ok, sc, "a callback's own queue work completes");
        check(ref.matches(c, 0), sc, "bitwise equal to the uncut chain");
        check(ggml_backend_metal_event_host_value(c.ev) == base + (uint64_t) (round + 1)*kLayers, sc,
                "the event advances one value per cut, across computes");
    }
    cs.queue_round = false;
    clear_schedules(c.backend);
}

// A gate wait on every expert segment, which only the callback opens: the callback must run
// before the gate wait.
static void test_gated(chain & c, cut_state & cs, const reference & ref) {
    const char * sc = "ordering, gated";
    std::vector<ggml_metal_event_t> no_ev(kLayers, nullptr);
    std::vector<uint64_t>           no_val(kLayers, 0);
    std::vector<ggml_metal_event_t> wait_ev(kLayers, c.gate);
    std::vector<uint64_t>           wait_val;
    for (int k = 0; k < kLayers; k++) {
        wait_val.push_back((uint64_t) k + 1);
    }
    ggml_backend_metal_set_boundary_schedule(c.backend, kLayers, c.r.data(), no_ev.data(), no_val.data(),
            kLayers, c.e.data(), wait_ev.data(), wait_val.data());
    set_host_cuts(c, cs, ggml_backend_metal_event_host_value(c.ev));

    cs.reset();
    cs.open_gate = true;
    c.seed();
    check(compute(c.backend, c.gf) == GGML_STATUS_SUCCESS, sc, "compute succeeds");
    check(cs.fired == range(0, kLayers), sc, "every cut fires once, in order");
    check(cs.r_seen_ok, sc, "each callback reads its segment's output");
    check(ref.matches(c, 0), sc, "bitwise equal to the uncut chain");
    cs.open_gate = false;
    clear_schedules(c.backend);
}

// An encode window (router.1, end]: the cuts in the segments it skips never fire. The window's
// first segment reads router.1 and the slot map, which the skipped callback would have written,
// so both are seeded from the reference.
static void test_window(chain & c, cut_state & cs, const reference & ref) {
    const char * sc = "encode window";
    set_host_cuts(c, cs, ggml_backend_metal_event_host_value(c.ev));
    ggml_backend_metal_set_encode_window(c.backend, c.r[1], nullptr, 0, nullptr, 0, nullptr, 0, nullptr);

    cs.reset();
    c.seed();
    memcpy(host_data(c.r[1]), ref.r[1].data(), kBytes);
    memcpy(host_data(c.s), cs.slots[1].data(), kBytes);
    check(compute(c.backend, c.gf) == GGML_STATUS_SUCCESS, sc, "compute succeeds");
    check(cs.fired == range(2, kLayers), sc, "only the cuts inside the window fire");
    check(snapshot(c.r[0]) == host_t(kElems, kSentinel) && snapshot(c.e[0]) == host_t(kElems, kSentinel), sc,
            "the skipped segments did not run");
    check(ref.matches(c, 1), sc, "bitwise equal to the uncut chain inside it");
    clear_schedules(c.backend);
}

static bool contains(const char * s, const char * part) {
    return s != nullptr && strstr(s, part) != nullptr;
}

// The callback never opens the gate the next segment waits for: the gate wait hits the abort
// limit and the graph fails without running that segment.
static void test_gate_abort() {
    const char * sc = "abort, gate never opened";
    chain c;
    if (!c.init()) {
        check(false, sc, "init");
        return;
    }
    cut_state cs(c);
    std::vector<ggml_metal_event_t> no_ev(kLayers, nullptr);
    std::vector<uint64_t>           no_val(kLayers, 0);
    ggml_tensor *      wait_node = c.e[1];
    ggml_metal_event_t wait_ev   = c.gate;
    uint64_t           wait_val  = 1;
    ggml_backend_metal_set_boundary_schedule(c.backend, kLayers, c.r.data(), no_ev.data(), no_val.data(),
            1, &wait_node, &wait_ev, &wait_val);
    ggml_backend_metal_set_paged_gate_abort_ms(c.backend, 300);
    set_host_cuts(c, cs, 0);

    c.seed();
    const double t0 = now_us();
    const ggml_status st = compute(c.backend, c.gf);
    const double dt = now_us() - t0;
    check(st == GGML_STATUS_FAILED, sc, "compute fails");
    check(dt >= 300000.0 && dt < 10000000.0, sc, "after the abort limit");
    check(contains(ggml_backend_metal_last_error(c.backend), "gate stayed closed"), sc, "the error names the gate");
    check(cs.fired == range(0, 2), sc, "the cuts ahead of the gate fired");
    check(snapshot(c.e[1]) == host_t(kElems, kSentinel), sc, "the gated segment did not run");
    check(compute(c.backend, c.gf) == GGML_STATUS_FAILED, sc, "the backend stays failed");
}

// The callback fails the graph: the segment after the cut is never committed.
static void test_callback_fails() {
    const char * sc = "abort, callback fails";
    chain c;
    if (!c.init()) {
        check(false, sc, "init");
        return;
    }
    cut_state cs(c);
    cs.fail_at = 1;
    set_host_cuts(c, cs, 0);

    c.seed();
    check(compute(c.backend, c.gf) == GGML_STATUS_FAILED, sc, "compute fails");
    check(contains(ggml_backend_metal_last_error(c.backend), "host cut 1 at node 'router.1'"), sc, "the error names the cut");
    check(cs.fired == range(0, 2), sc, "no cut fires after it");
    check(snapshot(c.e[0]) != host_t(kElems, kSentinel), sc, "the segments before it ran");
    check(snapshot(c.e[1]) == host_t(kElems, kSentinel), sc, "the segment after it did not run");
}

// The host cut event is saturated: the next cut's spin sees an abandoned value.
static void test_saturated() {
    const char * sc = "abort, event saturated";
    chain c;
    if (!c.init()) {
        check(false, sc, "init");
        return;
    }
    cut_state cs(c);
    cs.saturate_at = 0;
    set_host_cuts(c, cs, 0);

    c.seed();
    check(compute(c.backend, c.gf) == GGML_STATUS_FAILED, sc, "compute fails");
    check(contains(ggml_backend_metal_last_error(c.backend), "abandoned"), sc, "the error says abandoned");
    check(cs.fired == range(0, 1), sc, "no callback runs on a saturated value");
    check(snapshot(c.e[1]) == host_t(kElems, kSentinel), sc, "the segment after it did not run");
}

// 40 layers of a router product, a host cut and an experts product, at a size whose segments
// take a few hundred microseconds each. The callback reads 8 router outputs and writes an
// 8-entry slot map the experts read.
struct timing_state {
    std::vector<ggml_tensor *> r;
    ggml_tensor * s = nullptr;
    float sink = 0.0f;
};

static bool on_timing_cut(void * user_data, int i, ggml_tensor * node) {
    timing_state & ts = *(timing_state *) user_data;
    const float * rv = host_data(node);
    float * sv = host_data(ts.s);
    for (int j = 0; j < 8; j++) {
        ts.sink += rv[j];
        sv[j] = 1.0f;
    }
    (void) i;
    return true;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void test_timing() {
    const char * sc = "timing";
    constexpr int     n_layer = 40;
    constexpr int64_t D       = 4096;
    constexpr int     n_warm  = 5;
    constexpr int     n_iter  = 30;

    ggml_backend_t backend = ggml_backend_metal_init();
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*(4*n_layer + 16) + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_w = ggml_init(params);
    ggml_context * ctx_a = ggml_init(params);

    ggml_tensor * w_r = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F16, D, D);
    ggml_tensor * w_e = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F16, D, D);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);

    timing_state ts;
    ggml_tensor * x = ggml_new_tensor_1d(ctx_a, GGML_TYPE_F32, D);
    ts.s = ggml_new_tensor_1d(ctx_a, GGML_TYPE_F32, D);
    ggml_tensor * h = x;
    for (int k = 0; k < n_layer; k++) {
        ggml_tensor * r = ggml_mul_mat(ctx_a, w_r, h);
        ts.r.push_back(r);
        h = ggml_mul_mat(ctx_a, w_e, ggml_mul(ctx_a, r, ts.s));
    }
    ggml_cgraph * gf = ggml_new_graph(ctx_a);
    ggml_build_forward_expand(gf, h);

    host_arena arena;
    if (buf_w == nullptr || !arena.alloc(backend, ctx_a)) {
        check(false, sc, "init");
    } else {
        // weights 1/D and inputs 1 keep every activation near 1
        std::vector<ggml_fp16_t> w((size_t) (D*D), ggml_fp32_to_fp16(1.0f / (float) D));
        ggml_backend_tensor_set(w_r, w.data(), 0, ggml_nbytes(w_r));
        ggml_backend_tensor_set(w_e, w.data(), 0, ggml_nbytes(w_e));
        fill(x, 1.0f);
        fill(ts.s, 1.0f);

        ggml_metal_event_t ev = ggml_backend_metal_event_init(backend);
        std::vector<ggml_metal_event_t> no_ev(n_layer, nullptr);
        std::vector<uint64_t>           no_val(n_layer, 0);

        enum { UNCUT, SEGMENTED, HOST_CUT, N_VARIANT };
        const char * names[N_VARIANT] = { "uncut", "segmented", "host cut" };
        std::vector<double> times[N_VARIANT];
        bool ok = true;
        for (int it = 0; it < n_warm + n_iter; it++) {
            for (int v = 0; v < N_VARIANT; v++) {
                clear_schedules(backend);
                if (v == SEGMENTED) {
                    ggml_backend_metal_set_boundary_schedule(backend, n_layer, ts.r.data(), no_ev.data(), no_val.data(),
                            0, nullptr, nullptr, nullptr);
                } else if (v == HOST_CUT) {
                    ggml_backend_metal_set_host_cuts(backend, n_layer, ts.r.data(), ev,
                            ggml_backend_metal_event_host_value(ev), on_timing_cut, &ts);
                }
                const double t0 = now_us();
                ok = ok && compute(backend, gf) == GGML_STATUS_SUCCESS;
                const double dt = now_us() - t0;
                if (it >= n_warm) {
                    times[v].push_back(dt);
                }
            }
        }
        clear_schedules(backend);
        check(ok, sc, "every compute succeeds");

        double med[N_VARIANT];
        for (int v = 0; v < N_VARIANT; v++) {
            med[v] = median(times[v]);
            printf("timing: %-9s median %8.1f us per graph of %d layers\n", names[v], med[v], n_layer);
        }
        printf("timing: per cut, host cut over uncut      %6.1f us\n", (med[HOST_CUT]  - med[UNCUT])     / n_layer);
        printf("timing: per cut, segmented over uncut     %6.1f us\n", (med[SEGMENTED] - med[UNCUT])     / n_layer);
        printf("timing: per cut, host cut over segmented  %6.1f us\n", (med[HOST_CUT]  - med[SEGMENTED]) / n_layer);
        printf("timing: GPU work per layer, uncut         %6.1f us\n", med[UNCUT] / n_layer);

        ggml_backend_metal_event_free(backend, ev);
    }

    arena.release();
    if (buf_w != nullptr) {
        ggml_backend_buffer_free(buf_w);
    }
    ggml_free(ctx_a);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
}

int main() {
    {
        chain c;
        if (!c.init()) {
            printf("SKIP no Metal device\n");
            return 0;
        }

        cut_state cs(c);
        reference ref;
        check(ref.run(c, cs), "reference", "the uncut chain computes");

        test_ordering(c, cs, ref);
        test_gated(c, cs, ref);
        test_window(c, cs, ref);
    }
    test_gate_abort();
    test_callback_fails();
    test_saturated();
    test_timing();

    if (g_n_fail > 0) {
        printf("%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
