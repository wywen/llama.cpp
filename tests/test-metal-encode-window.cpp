// Tests for the Metal encode window's boundary copies.
//
// The window runs on a four-node add chain a0 -> a1 -> a2 -> a3 with a boundary cut after every
// node, so each node is its own segment. A window of (a0, a2] encodes a1 and a2 only: ingress
// copies land in a0 before a1 reads it, and egress copies read a2 after it is written. Every
// expectation is computed on the host by replaying the same copies and adds on plain arrays.
//
// The validation cases need no device work, but they do need a Metal backend to call into, so
// the whole binary skips when there is no Metal device.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <vector>

static int g_n_fail = 0;

static void check(bool ok, const char * scenario, const char * what) {
    printf("%s %-48s %s\n", ok ? "PASS" : "FAIL", scenario, what);
    if (!ok) {
        g_n_fail++;
    }
}

constexpr int64_t kElems = 64;
constexpr size_t  kBytes = kElems * sizeof(float);
constexpr float   kSentinel = -7.0f;

static const std::array<float, 4> kAddend = { 10.0f, 20.0f, 30.0f, 40.0f };

using host_t = std::vector<float>;

static host_t filled(float v) {
    return host_t(kElems, v);
}

// distinct per-element values, so a copy landing at the wrong offset is visible
static host_t ramp(float base) {
    host_t res(kElems);
    for (int64_t i = 0; i < kElems; i++) {
        res[i] = base + (float) i;
    }
    return res;
}

static void host_copy(const host_t & src, host_t & dst, const ggml_metal_tensor_range_copy & r) {
    const size_t n = r.size / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        dst[r.dst_offs / sizeof(float) + i] = src[r.src_offs / sizeof(float) + i];
    }
}

struct fixture {
    ggml_backend_t        backend = nullptr;
    ggml_context        * ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph         * gf      = nullptr;

    ggml_tensor * x = nullptr;
    std::array<ggml_tensor *, 4> addend{};
    std::array<ggml_tensor *, 4> chain{};
    ggml_tensor * stash_in  = nullptr;
    ggml_tensor * stash_out = nullptr;
    ggml_tensor * stash_out_head = nullptr; // view: bytes [0, 192) of stash_out
    ggml_tensor * stash_out_tail = nullptr; // view: bytes [64, 256) of stash_out, overlapping the head

    bool init() {
        backend = ggml_backend_metal_init();
        if (backend == nullptr) {
            return false;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*64 + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
        for (auto & t : addend) {
            t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
        }
        chain[0] = ggml_add(ctx, x, addend[0]);
        for (size_t i = 1; i < chain.size(); i++) {
            chain[i] = ggml_add(ctx, chain[i - 1], addend[i]);
        }
        stash_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
        stash_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
        stash_out_head = ggml_view_1d(ctx, stash_out, kElems*3/4, 0);
        stash_out_tail = ggml_view_1d(ctx, stash_out, kElems*3/4, kBytes/4);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, chain[3]);

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == nullptr) {
            return false;
        }

        set(x, filled(1.0f));
        for (size_t i = 0; i < addend.size(); i++) {
            set(addend[i], filled(kAddend[i]));
        }

        // a cut after every chain node: four one-node segments, and a non-empty schedule
        // routes compute through the path the encode window applies to
        std::array<ggml_tensor *, 4>      cuts = chain;
        std::array<ggml_metal_event_t, 4> sig_ev{};
        std::array<uint64_t, 4>           sig_val{};
        ggml_backend_metal_set_boundary_schedule(backend, (int) cuts.size(), cuts.data(), sig_ev.data(), sig_val.data(),
                0, nullptr, nullptr, nullptr);

        return true;
    }

    ~fixture() {
        if (backend != nullptr) {
            ggml_backend_metal_clear_encode_window(backend);
            ggml_backend_metal_set_boundary_schedule(backend, 0, nullptr, nullptr, nullptr, 0, nullptr, nullptr, nullptr);
        }
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

    static void set(ggml_tensor * t, const host_t & v) {
        ggml_backend_tensor_set(t, v.data(), 0, kBytes);
    }

    static host_t get(const ggml_tensor * t) {
        host_t res(kElems);
        ggml_backend_tensor_get(t, res.data(), 0, kBytes);
        return res;
    }

    // seeds the stashes and every chain node, so a node the window skips keeps its seed
    void seed(const host_t & in, const host_t & out) {
        set(stash_in, in);
        set(stash_out, out);
        for (auto * t : chain) {
            set(t, filled(kSentinel));
        }
    }

    bool compute() {
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        return st == GGML_STATUS_SUCCESS;
    }

    bool set_window(const std::vector<ggml_metal_tensor_range_copy> & ingress,
                    const std::vector<ggml_metal_tensor_range_copy> & egress) {
        return ggml_backend_metal_set_encode_window_ranged(backend, chain[0], chain[2],
                ingress.size(), ingress.data(), egress.size(), egress.data(), 0, nullptr);
    }
};

// What the window (a0, a2] computes: a0 = its seed with the ingress copies applied, a1 and a2
// from it, a3 untouched, stash_out = its seed with the egress copies applied in list order.
struct expected {
    host_t a0, a1, a2, a3, out;
};

static expected replay(const host_t & in, const host_t & out,
                       const std::vector<ggml_metal_tensor_range_copy> & ingress,
                       const std::vector<ggml_metal_tensor_range_copy> & egress,
                       const fixture & f) {
    expected e;
    e.a0 = filled(kSentinel);
    for (const auto & r : ingress) {
        host_copy(in, e.a0, r);
    }
    e.a1 = e.a0;
    e.a2 = e.a0;
    for (int64_t i = 0; i < kElems; i++) {
        e.a1[i] = e.a0[i] + kAddend[1];
        e.a2[i] = e.a1[i] + kAddend[2];
    }
    e.a3  = filled(kSentinel);
    e.out = out;
    for (const auto & r : egress) {
        // an egress copy reads either a2 or stash_out itself
        const host_t src = r.src == f.stash_out ? e.out : e.a2;
        host_copy(src, e.out, r);
    }
    return e;
}

static void run_window(fixture & f, const char * scenario,
                       const std::vector<ggml_metal_tensor_range_copy> & ingress,
                       const std::vector<ggml_metal_tensor_range_copy> & egress) {
    const host_t in  = ramp(1000.0f);
    const host_t out = ramp(-500.0f);

    check(f.set_window(ingress, egress), scenario, "window accepted");
    f.seed(in, out);
    check(f.compute(), scenario, "compute succeeded");

    const expected e = replay(in, out, ingress, egress, f);
    check(fixture::get(f.chain[0]) == e.a0,  scenario, "a0 holds the ingress ranges");
    check(fixture::get(f.chain[1]) == e.a1,  scenario, "a1 computed from a0");
    check(fixture::get(f.chain[2]) == e.a2,  scenario, "a2 computed from a1");
    check(fixture::get(f.chain[3]) == e.a3,  scenario, "a3 outside the window not computed");
    check(fixture::get(f.stash_out) == e.out, scenario, "stash_out holds the egress ranges");
}

static ggml_metal_tensor_range_copy range(ggml_tensor * src, size_t src_offs, ggml_tensor * dst, size_t dst_offs, size_t size) {
    return { src, dst, src_offs, dst_offs, size };
}

static void test_whole_tensor_default(fixture & f) {
    const char * scenario = "whole-tensor pairs";

    const std::array<ggml_metal_tensor_copy_pair, 1> ingress = {{ { f.stash_in, f.chain[0] } }};
    const std::array<ggml_metal_tensor_copy_pair, 1> egress  = {{ { f.chain[2], f.stash_out } }};
    ggml_backend_metal_set_encode_window(f.backend, f.chain[0], f.chain[2],
            ingress.size(), ingress.data(), egress.size(), egress.data(), 0, nullptr);

    const host_t in = ramp(1000.0f);
    f.seed(in, filled(0.0f));
    check(f.compute(), scenario, "compute succeeded");

    host_t a2 = in;
    for (auto & v : a2) {
        v += kAddend[1] + kAddend[2];
    }
    check(fixture::get(f.chain[0]) == in,              scenario, "a0 is stash_in in full");
    check(fixture::get(f.chain[2]) == a2,              scenario, "a2 computed from a0");
    check(fixture::get(f.chain[3]) == filled(kSentinel), scenario, "a3 outside the window not computed");
    check(fixture::get(f.stash_out) == a2,             scenario, "stash_out is a2 in full");

    // the ranged form of the same copies must give the same result
    run_window(f, "whole-tensor as ranges",
            { range(f.stash_in, 0, f.chain[0], 0, kBytes) },
            { range(f.chain[2], 0, f.stash_out, 0, kBytes) });
}

static void test_ranged_round_trips(fixture & f) {
    // one range each way, at different offsets on each side
    run_window(f, "one range",
            { range(f.stash_in, 16, f.chain[0], 64, 96) },
            { range(f.chain[2], 64, f.stash_out, 128, 96) });

    // two ranges each way; the second ingress and the first egress end at the tensor's end
    run_window(f, "two ranges, one ending at the end",
            { range(f.stash_in, 0,   f.chain[0], 32,  16),
              range(f.stash_in, 192, f.chain[0], 192, kBytes - 192) },
            { range(f.chain[2], 0,   f.stash_out, kBytes - 64, 64),
              range(f.chain[2], 128, f.stash_out, 0,           32) });

    // the second egress copy reads bytes the first one wrote, so it only sees a2's values if
    // the copies run in list order
    run_window(f, "copies run in list order",
            { range(f.stash_in, 0, f.chain[0], 0, kBytes) },
            { range(f.chain[2],  0, f.stash_out, 0,   64),
              range(f.stash_out, 0, f.stash_out, 128, 64) });

    // non-overlapping ranges of one tensor are a valid copy
    run_window(f, "disjoint ranges within one tensor",
            { range(f.stash_in, 0, f.chain[0], 0, kBytes) },
            { range(f.stash_out, 0, f.stash_out, kBytes/2, kBytes/2) });
}

static void test_validation(fixture & f) {
    // install a known-good window first: every rejected call below must leave it in place
    const std::vector<ggml_metal_tensor_range_copy> good_in  = { range(f.stash_in, 0, f.chain[0], 0, kBytes) };
    const std::vector<ggml_metal_tensor_range_copy> good_out = { range(f.chain[2], 0, f.stash_out, 0, kBytes) };
    check(f.set_window(good_in, good_out), "validation", "baseline window accepted");

    struct bad_case {
        const char * name;
        ggml_metal_tensor_range_copy r;
    };
    const std::vector<bad_case> cases = {
        { "NULL src",                      range(nullptr,    0,  f.chain[0], 0,  16) },
        { "NULL dst",                      range(f.stash_in, 0,  nullptr,    0,  16) },
        { "zero size",                     range(f.stash_in, 0,  f.chain[0], 0,  0) },
        { "source range past the end",     range(f.stash_in, kBytes - 16, f.chain[0], 0, 32) },
        { "destination range past the end", range(f.stash_in, 0, f.chain[0], kBytes - 16, 32) },
        { "source offset past the end",    range(f.stash_in, kBytes + 4, f.chain[0], 0, 4) },
        { "offset + size wraps",           range(f.stash_in, 4, f.chain[0], 0, SIZE_MAX - 3) },
        { "unaligned src_offs",            range(f.stash_in, 2,  f.chain[0], 0,  16) },
        { "unaligned dst_offs",            range(f.stash_in, 0,  f.chain[0], 6,  16) },
        { "unaligned size",                range(f.stash_in, 0,  f.chain[0], 0,  18) },
        { "overlap within one tensor",     range(f.stash_out, 0, f.stash_out, 32, 64) },
        { "overlap across views of one tensor",
                                           range(f.stash_out_head, 128, f.stash_out_tail, 64, 32) },
    };

    for (const auto & c : cases) {
        // in either list
        check(!f.set_window({ c.r }, good_out), c.name, "rejected as ingress");
        check(!f.set_window(good_in, { c.r }),  c.name, "rejected as egress");
    }

    // a bad range after a good one in the same list still rejects the whole call
    check(!f.set_window({ good_in[0], cases[2].r }, good_out), "bad range after a good one", "rejected");

    check(!ggml_backend_metal_set_encode_window_ranged(f.backend, f.chain[0], f.chain[2], 1, nullptr, 0, nullptr, 0, nullptr),
            "non-zero count with NULL list", "rejected");

    ggml_tensor * const null_node = nullptr;
    check(!ggml_backend_metal_set_encode_window_ranged(f.backend, f.chain[0], f.chain[2],
                good_in.size(), good_in.data(), good_out.size(), good_out.data(), 1, &null_node),
            "NULL out-of-band node", "rejected");

    // the baseline window survived every rejection
    const host_t in = ramp(1000.0f);
    f.seed(in, filled(0.0f));
    check(f.compute(), "validation", "compute after rejections succeeded");
    host_t a2 = in;
    for (auto & v : a2) {
        v += kAddend[1] + kAddend[2];
    }
    check(fixture::get(f.stash_out) == a2, "validation", "rejected calls left the installed window in place");
}

int main() {
    fixture f;
    if (!f.init()) {
        printf("SKIP no Metal device\n");
        return 0;
    }

    test_whole_tensor_default(f);
    test_ranged_round_trips(f);
    test_validation(f);

    if (g_n_fail > 0) {
        printf("%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
