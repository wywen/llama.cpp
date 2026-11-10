// Tests for host-pointer buffers split into disjoint Metal buffers at caller-given offsets.
//
// The host range has three sub-ranges; the last one does not end at a page boundary. A tensor over
// the whole range is in no Metal buffer, and views of it inside one sub-range resolve to that
// sub-range's Metal buffer. A graph scales a view in each sub-range into a view in the next one, so
// each kernel binds two different Metal buffers. Encoding a kernel on a view across a split point
// must abort; the binary checks this by running itself in a child process.
//
// Each sub-range is also unwrapped, replaced and rewrapped on its own: while one sub-range is unwrapped and its pages
// are mapped anew, kernels over the other sub-ranges compute, encoding a kernel over the unwrapped one aborts (again in
// a child process), and after rewrap the GPU reads and writes the new pages. Each sub-range has its own residency set,
// handed to the caller by the first call that names it. Whether a set's pages are actually wired is not observable
// here; the checks cover the call results and which Metal buffers each set holds. It skips when there is no Metal
// device.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

static int g_n_fail = 0;

static void check(bool ok, const char * scenario, const char * what) {
    printf("%s %-40s %s\n", ok ? "PASS" : "FAIL", scenario, what);
    if (!ok) {
        g_n_fail++;
    }
}

constexpr int     kSplits = 3;
constexpr int64_t kElems  = 256;
constexpr size_t  kBytes  = kElems * sizeof(float);
constexpr size_t  kInOffs = 256;  // a view's offset in its sub-range
constexpr size_t  kTail   = 512;  // the last sub-range ends this many bytes short of a page boundary

static ggml_metal_buffer_t metal_buf(ggml_backend_buffer_t buf) {
    return (ggml_metal_buffer_t) buf->context;
}

static bool find_id(const ggml_tensor * t, ggml_metal_buffer_id * res) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    return ggml_metal_buffer_find_id(metal_buf(buf), t, res);
}

struct fixture {
    ggml_backend_t        backend = nullptr;
    ggml_backend_dev_t    dev     = nullptr;
    ggml_context        * ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph         * gf      = nullptr;

    size_t page = 0;
    size_t sub  = 0; // size of each sub-range but the last
    size_t size = 0;
    void * mem  = nullptr;

    std::array<size_t, kSplits - 1> offsets{};

    ggml_tensor * all = nullptr;
    std::array<ggml_tensor *, kSplits> in{};
    std::array<ggml_tensor *, kSplits> out{};

    bool init() {
        backend = ggml_backend_metal_init();
        if (backend == nullptr) {
            return false;
        }
        dev = ggml_backend_get_device(backend);

        page = (size_t) sysconf(_SC_PAGESIZE);
        sub  = 4*page;
        size = kSplits*sub - kTail;
        for (int i = 0; i < kSplits - 1; i++) {
            offsets[i] = (i + 1)*sub;
        }
        // mapped rather than allocated, so that a sub-range's pages can be replaced with MAP_FIXED
        mem = mmap(nullptr, kSplits*sub, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (mem == MAP_FAILED) {
            mem = nullptr;
            return false;
        }

        buf = ggml_backend_metal_buffer_from_host_ptr_split(dev, mem, size, offsets.data(), kSplits - 1);
        check(buf != nullptr, "create", "a valid split buffer is created");
        if (buf == nullptr) {
            return false;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*32 + 8*ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
        all = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size/sizeof(float));
        ggml_set_name(all, "all");
        check(ggml_backend_tensor_alloc(buf, all, base) == GGML_STATUS_SUCCESS, "create", "a tensor over the whole range is allocated");

        for (int i = 0; i < kSplits; i++) {
            const size_t offs = i*sub + kInOffs;
            in[i] = ggml_view_1d(ctx, all, kElems, offs);
            ggml_format_name(in[i], "in-%d", i);
            check(ggml_backend_view_init(in[i]) == GGML_STATUS_SUCCESS, "create", "a view in a sub-range is initialized");
        }
        for (int i = 0; i < kSplits; i++) {
            out[i] = ggml_scale(ctx, in[i], 2.0f + (float) i);
            ggml_format_name(out[i], "out-%d", i);
        }
        gf = ggml_new_graph(ctx);
        for (int i = 0; i < kSplits; i++) {
            // the result lands at the end of the next sub-range, which for the last one is its unaligned end
            const int    j    = (i + 1) % kSplits;
            const size_t end  = j == kSplits - 1 ? size : (j + 1)*sub;
            check(ggml_backend_tensor_alloc(buf, out[i], base + end - kBytes) == GGML_STATUS_SUCCESS, "create", "a result tensor is allocated");
            ggml_build_forward_expand(gf, out[i]);
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
        if (mem != nullptr) {
            munmap(mem, kSplits*sub);
        }
    }

    // the sub-range a tensor lies in, from its address
    int sub_range_of(const ggml_tensor * t) const {
        return (int) (((const uint8_t *) t->data - (const uint8_t *) mem) / sub);
    }
};

static void test_resolve(fixture & f, const char * scenario) {
    std::set<void *> seen;
    for (int i = 0; i < kSplits; i++) {
        for (const ggml_tensor * t : { (const ggml_tensor *) f.in[i], (const ggml_tensor *) f.out[(i + kSplits - 1) % kSplits] }) {
            ggml_metal_buffer_id id{};
            const bool   found = find_id(t, &id);
            const size_t want  = (size_t) ((const uint8_t *) t->data - (const uint8_t *) f.mem) - i*f.sub;
            check(found && id.metal != nullptr, scenario, "a tensor inside a sub-range has a Metal buffer");
            check(id.offs == want, scenario, "its offset is relative to its sub-range");
            check(f.sub_range_of(t) == i, scenario, "the tensor lies in the expected sub-range");
            seen.insert(id.metal);
        }
    }
    check(seen.size() == (size_t) kSplits, scenario, "each sub-range has its own Metal buffer");

    ggml_metal_buffer_id id{};
    check(!find_id(f.all, &id) && id.metal == nullptr, scenario, "a tensor across the split points has none");

    ggml_tensor * cross = ggml_view_1d(f.ctx, f.all, kElems, f.sub - kBytes/2);
    ggml_backend_view_init(cross);
    check(!find_id(cross, &id) && id.metal == nullptr, scenario, "a view across one split point has none");
}

static void test_compute(fixture & f, const char * scenario, float base) {
    std::array<std::vector<float>, kSplits> src;
    for (int i = 0; i < kSplits; i++) {
        src[i].resize(kElems);
        for (int64_t k = 0; k < kElems; k++) {
            src[i][k] = base + 1000.0f*i + (float) k;
        }
        ggml_backend_tensor_set(f.in[i], src[i].data(), 0, kBytes);
        ggml_backend_tensor_memset(f.out[i], 0xff, 0, kBytes);
    }

    check(ggml_backend_graph_compute(f.backend, f.gf) == GGML_STATUS_SUCCESS, scenario, "the graph computes");

    for (int i = 0; i < kSplits; i++) {
        std::vector<float> res(kElems);
        ggml_backend_tensor_get(f.out[i], res.data(), 0, kBytes);
        bool ok = true;
        for (int64_t k = 0; k < kElems; k++) {
            ok = ok && res[k] == src[i][k]*(2.0f + (float) i);
        }
        check(ok, scenario, "a kernel reads one sub-range and writes the next");
    }
}

static void test_unwrap_rewrap(fixture & f) {
    check(ggml_backend_metal_buffer_end_residency(f.buf), "unwrap", "the caller takes the residency set");
    check(ggml_backend_metal_buffer_unwrap(f.buf), "unwrap", "the buffer is unwrapped");

    ggml_metal_buffer_id id{};
    bool all_nil = true;
    for (int i = 0; i < kSplits; i++) {
        all_nil = all_nil && find_id(f.in[i], &id) && id.metal == nullptr;
    }
    check(all_nil, "unwrap", "every sub-range keeps its range and has no Metal buffer");

    check(ggml_backend_metal_buffer_rewrap(f.buf), "rewrap", "the buffer is rewrapped");
    check(ggml_backend_metal_buffer_request_residency(f.buf), "rewrap", "the caller requests the residency set");

    bool own = true;
    for (int s = 0; s < kSplits; s++) {
        own = own && ggml_metal_buffer_rset_members(metal_buf(f.buf), s) == (uint64_t) 1 << s;
    }
    check(own, "rewrap", "each sub-range's set holds its own Metal buffer again");

    test_resolve(f, "rewrap");
    test_compute(f, "rewrap", -50000.0f);
}

static void test_validation(fixture & f) {
    const size_t page = f.page;
    const size_t size = 4*page;
    void * mem = f.mem;

    auto refused = [&](void * ptr, size_t sz, std::vector<size_t> offs, int n, const char * what) {
        ggml_backend_buffer_t b = ggml_backend_metal_buffer_from_host_ptr_split(f.dev, ptr, sz, offs.empty() ? nullptr : offs.data(), n);
        check(b == nullptr, "validation", what);
        if (b != nullptr) {
            ggml_backend_buffer_free(b);
        }
    };

    refused(mem, size, { 2*page, page }, 2, "decreasing offsets are refused");
    refused(mem, size, { page, page }, 2, "repeated offsets are refused");
    refused(mem, size, { 0 }, 1, "an offset at 0 is refused");
    refused(mem, size, { size }, 1, "an offset at the size is refused");
    refused(mem, size, { size + page }, 1, "an offset past the size is refused");
    refused(mem, size, { page + 64 }, 1, "an offset off a page boundary is refused");
    refused(mem, size, {}, 1, "missing offsets are refused");
    refused(mem, size, {}, -1, "a negative count is refused");
    refused((uint8_t *) mem + 64, size - 64, { page }, 1, "a pointer off a page boundary is refused");
    refused(mem, 0, {}, 0, "an empty range is refused");
    refused(nullptr, size, {}, 0, "a null pointer is refused");

    {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            ggml_backend_buffer_t b = ggml_backend_metal_buffer_from_host_ptr_split(cpu, mem, size, nullptr, 0);
            check(b == nullptr, "validation", "a non-Metal device is refused");
        }
    }

    // 64 Metal buffers is the limit: 63 split points of one page each are accepted, 64 are refused
    {
        std::vector<size_t> offs(64);
        for (size_t i = 0; i < offs.size(); i++) {
            offs[i] = (i + 1)*page;
        }
        void * big = mmap(nullptr, 65*page, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        ggml_backend_buffer_t b = ggml_backend_metal_buffer_from_host_ptr_split(f.dev, big, 64*page, offs.data(), 63);
        check(b != nullptr, "validation", "63 split points are accepted");
        if (b != nullptr) {
            ggml_backend_buffer_free(b);
        }
        refused(big, 65*page, offs, 64, "64 split points are refused");
        munmap(big, 65*page);
    }

    // the check runs before any page is wrapped, so the range is only reserved
    {
        const size_t max = ggml_backend_buft_get_max_size(ggml_backend_dev_buffer_type(f.dev));
        const size_t sz  = ((max + page - 1)/page + 2)*page;
        void * big = mmap(nullptr, sz, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (big != MAP_FAILED) {
            refused(big, sz, { page }, 1, "a sub-range over the max buffer size is refused");
            munmap(big, sz);
        }
    }

    {
        ggml_backend_buffer_t b = ggml_backend_metal_buffer_from_host_ptr_split(f.dev, mem, size, nullptr, 0);
        check(b != nullptr, "validation", "no split points give one Metal buffer");
        if (b != nullptr) {
            ggml_backend_buffer_free(b);
        }
    }
}

// a buffer from buffer_from_host_ptr keeps its overlapping views and still resolves a tensor that spans pages
static void test_unsplit(fixture & f) {
    ggml_backend_buffer_t b = ggml_backend_dev_buffer_from_host_ptr(f.dev, f.mem, f.size, f.size);
    check(b != nullptr, "unsplit", "a plain host-pointer buffer is created");
    if (b == nullptr) {
        return;
    }
    ggml_tensor * t = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, f.size/sizeof(float));
    ggml_backend_tensor_alloc(b, t, ggml_backend_buffer_get_base(b));
    ggml_metal_buffer_id id{};
    check(ggml_metal_buffer_find_id(metal_buf(b), t, &id) && id.metal != nullptr && id.offs == 0, "unsplit", "a tensor over the whole range has a Metal buffer");
    check(ggml_backend_metal_buffer_end_residency(b) && ggml_backend_metal_buffer_request_residency(b), "unsplit", "the whole-buffer residency calls succeed");
    check(ggml_metal_buffer_rset_members(metal_buf(b), 0) == 1, "unsplit", "its one set holds its Metal buffer");
    check(ggml_backend_metal_buffer_unwrap(b) && ggml_backend_metal_buffer_rewrap(b), "unsplit", "it is unwrapped and rewrapped as a whole");
    check(ggml_metal_buffer_find_id(metal_buf(b), t, &id) && id.metal != nullptr, "unsplit", "the tensor has a Metal buffer after rewrap");
    ggml_backend_buffer_free(b);
}


//
// sub-ranges released and recreated one at a time
//

using range_call = bool (*)(ggml_backend_buffer_t, int);

static const range_call kRangeCalls[] = {
    ggml_backend_metal_buffer_unwrap_range,
    ggml_backend_metal_buffer_rewrap_range,
    ggml_backend_metal_buffer_request_residency_range,
    ggml_backend_metal_buffer_end_residency_range,
};

// the Metal buffers residency set s holds, as a mask of their indices
static uint64_t members(ggml_backend_buffer_t buf, int s) {
    return ggml_metal_buffer_rset_members(metal_buf(buf), s);
}

static uint64_t bit(int i) {
    return (uint64_t) 1 << i;
}

static ggml_metal_buffer_id id_of(const ggml_tensor * t) {
    ggml_metal_buffer_id id{};
    find_id(t, &id);
    return id;
}

static std::vector<float> ramp(float base) {
    std::vector<float> v(kElems);
    for (int64_t k = 0; k < kElems; k++) {
        v[k] = base + (float) k;
    }
    return v;
}

// true when t holds src scaled by s
static bool holds(const ggml_tensor * t, const std::vector<float> & src, float s) {
    std::vector<float> res(kElems);
    ggml_backend_tensor_get(t, res.data(), 0, kBytes);
    bool ok = true;
    for (int64_t k = 0; k < kElems; k++) {
        ok = ok && res[k] == src[k]*s;
    }
    return ok;
}

// a graph that computes out[i] alone: it reads sub-range i and writes sub-range i + 1
static ggml_cgraph * graph_of(fixture & f, int i) {
    ggml_cgraph * gf = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(gf, f.out[i]);
    return gf;
}

// scales sub-range 2 into sub-range 0, which leaves sub-range 1 alone; true when the result is right
static bool compute_around_1(fixture & f, ggml_cgraph * gf, float base, bool async) {
    const std::vector<float> src = ramp(base);
    ggml_backend_tensor_set(f.in[2], src.data(), 0, kBytes);
    ggml_backend_tensor_memset(f.out[2], 0xff, 0, kBytes);
    const ggml_status st = async ? ggml_backend_graph_compute_async(f.backend, gf) : ggml_backend_graph_compute(f.backend, gf);
    if (async) {
        // sub-range 1 is unwrapped by the caller while this work may still be in flight
        return st == GGML_STATUS_SUCCESS;
    }
    return st == GGML_STATUS_SUCCESS && holds(f.out[2], src, 4.0f);
}

// the first call that names a sub-range hands that sub-range's set alone to the caller, and each set holds its own
// sub-range's Metal buffer only
static void test_range_handoff(fixture & f) {
    const char * sc = "range-handoff";

    check(ggml_backend_metal_buffer_end_residency_range(f.buf, 1), sc, "ending sub-range 1's residency succeeds");
    check(members(f.buf, 1) == bit(1), sc, "sub-range 1's set holds sub-range 1's Metal buffer alone");
    bool others = true;
    for (int s = 0; s < kSplits; s++) {
        others = others && (members(f.buf, s) & ~bit(s)) == 0;
    }
    check(others, sc, "no set holds another sub-range's Metal buffer");

    check(ggml_backend_metal_buffer_request_residency_range(f.buf, 0), sc, "requesting sub-range 0's residency succeeds");
    check(ggml_backend_metal_buffer_request_residency_range(f.buf, 0), sc, "requesting it again succeeds");
    check(ggml_backend_metal_buffer_request_residency_range(f.buf, 2), sc, "requesting sub-range 2's residency succeeds");
    check(members(f.buf, 0) == bit(0) && members(f.buf, 2) == bit(2), sc, "sub-ranges 0 and 2 have their own sets");
    check(ggml_backend_metal_buffer_end_residency_range(f.buf, 2), sc, "ending sub-range 2's residency succeeds");
    check(ggml_backend_metal_buffer_request_residency_range(f.buf, 2), sc, "requesting it after the end succeeds");
    check(members(f.buf, 0) == bit(0) && members(f.buf, 1) == bit(1) && members(f.buf, 2) == bit(2), sc,
            "ending one sub-range's residency leaves the other sets as they were");

    test_compute(f, sc, 7000.0f);
}

// sub-range 1 is unwrapped and its pages mapped anew while work over sub-ranges 0 and 2 is in flight and encoded; after
// rewrap the GPU reads and writes the new pages
static void test_range_replace(fixture & f) {
    const char * sc = "range-replace";

    // bytes the GPU must not see once sub-range 1 is replaced
    const std::vector<float> stale = ramp(-3000.0f);
    ggml_backend_tensor_set(f.in[1], stale.data(), 0, kBytes);

    const ggml_metal_buffer_id id0 = id_of(f.in[0]);
    const ggml_metal_buffer_id id2 = id_of(f.in[2]);

    ggml_cgraph * gf = graph_of(f, 2);

    check(ggml_backend_metal_buffer_end_residency_range(f.buf, 1), sc, "the caller ends sub-range 1's residency");
    check(compute_around_1(f, gf, 100.0f, true), sc, "work over sub-ranges 2 and 0 is submitted");
    check(ggml_backend_metal_buffer_unwrap_range(f.buf, 1), sc, "sub-range 1 is unwrapped meanwhile");
    ggml_backend_synchronize(f.backend);
    check(holds(f.out[2], ramp(100.0f), 4.0f), sc, "the work in flight over sub-ranges 2 and 0 computes correctly");

    check(id_of(f.in[1]).metal == nullptr, sc, "sub-range 1 has no Metal buffer");
    check(id_of(f.in[0]).metal == id0.metal && id_of(f.in[2]).metal == id2.metal, sc, "sub-ranges 0 and 2 keep theirs");
    check(members(f.buf, 1) == 0, sc, "sub-range 1's set holds nothing");
    check(members(f.buf, 0) == bit(0) && members(f.buf, 2) == bit(2), sc, "the other sets keep their Metal buffers");

    uint8_t * sub1 = (uint8_t *) f.mem + f.sub;
    void * p = mmap(sub1, f.sub, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0);
    check(p == sub1, sc, "sub-range 1's pages are replaced with MAP_FIXED");
    check(compute_around_1(f, gf, 200.0f, false), sc, "a kernel over sub-ranges 2 and 0 computes while 1 is unwrapped");

    // host access to the unwrapped sub-range works: new bytes go into its new pages
    const std::vector<float> fresh = ramp(5000.0f);
    ggml_backend_tensor_set(f.in[1], fresh.data(), 0, kBytes);
    ggml_backend_tensor_memset(f.out[0], 0xff, 0, kBytes);

    check(ggml_backend_metal_buffer_rewrap_range(f.buf, 1), sc, "sub-range 1 is rewrapped");
    check(ggml_backend_metal_buffer_request_residency_range(f.buf, 1), sc, "the caller requests sub-range 1's residency");
    check(id_of(f.in[1]).metal != nullptr, sc, "sub-range 1 has a Metal buffer again");
    check(id_of(f.in[0]).metal == id0.metal && id_of(f.in[2]).metal == id2.metal, sc, "sub-ranges 0 and 2 still keep theirs");
    check(members(f.buf, 1) == bit(1), sc, "sub-range 1's set holds its new Metal buffer");

    const std::vector<float> src0 = ramp(9000.0f);
    const std::vector<float> src2 = ramp(-9000.0f);
    ggml_backend_tensor_set(f.in[0], src0.data(), 0, kBytes);
    ggml_backend_tensor_set(f.in[2], src2.data(), 0, kBytes);
    ggml_backend_tensor_memset(f.out[1], 0xff, 0, kBytes);
    ggml_backend_tensor_memset(f.out[2], 0xff, 0, kBytes);
    check(ggml_backend_graph_compute(f.backend, f.gf) == GGML_STATUS_SUCCESS, sc, "the whole graph computes");
    check(holds(f.out[1], fresh, 3.0f), sc, "the GPU reads the new bytes of the replaced sub-range");
    check(holds(f.out[0], src0, 2.0f), sc, "the GPU writes into the replaced sub-range's new pages");
    check(holds(f.out[2], src2, 4.0f), sc, "the other sub-ranges still compute");
}

// per-range calls that name no sub-range of a split buffer are refused, and a whole-buffer call refuses, changing
// nothing, when one sub-range refuses
static void test_range_refusals(fixture & f) {
    const char * sc = "range-refusals";

    bool refused = true;
    for (range_call call : kRangeCalls) {
        refused = refused && !call(f.buf, -1) && !call(f.buf, kSplits) && !call(nullptr, 0);
    }
    check(refused, sc, "an index out of range or a null buffer is refused");

    ggml_backend_buffer_t plain = ggml_backend_dev_buffer_from_host_ptr(f.dev, f.mem, f.size, f.size);
    ggml_backend_buffer_t owned = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(f.dev), f.page);
    refused = plain != nullptr && owned != nullptr;
    for (range_call call : kRangeCalls) {
        refused = refused && !call(plain, 0) && !call(owned, 0);
    }
    check(refused, sc, "per-range calls on a plain host-pointer or device buffer are refused");
    if (plain != nullptr) {
        ggml_backend_buffer_free(plain);
    }
    if (owned != nullptr) {
        ggml_backend_buffer_free(owned);
    }

    check(ggml_backend_metal_buffer_end_residency_range(f.buf, 1), sc, "sub-range 1's residency ends");
    check(ggml_backend_metal_buffer_unwrap_range(f.buf, 1), sc, "sub-range 1 is unwrapped");
    check(!ggml_backend_metal_buffer_unwrap_range(f.buf, 1), sc, "unwrapping it again is refused");
    check(!ggml_backend_metal_buffer_request_residency_range(f.buf, 1), sc, "requesting its residency is refused");
    check(ggml_backend_metal_buffer_end_residency_range(f.buf, 1), sc, "ending its residency again succeeds");
    check(!ggml_backend_metal_buffer_rewrap_range(f.buf, 0), sc, "rewrapping a wrapped sub-range is refused");

    check(!ggml_backend_metal_buffer_unwrap(f.buf), sc, "a whole-buffer unwrap is refused");
    check(!ggml_backend_metal_buffer_rewrap(f.buf), sc, "a whole-buffer rewrap is refused");
    check(!ggml_backend_metal_buffer_request_residency(f.buf), sc, "a whole-buffer request is refused");
    check(id_of(f.in[0]).metal != nullptr && id_of(f.in[2]).metal != nullptr && id_of(f.in[1]).metal == nullptr, sc,
            "the refused whole-buffer calls change no sub-range");

    check(ggml_backend_metal_buffer_rewrap_range(f.buf, 1), sc, "sub-range 1 is rewrapped");
    check(ggml_backend_metal_buffer_request_residency(f.buf), sc, "a whole-buffer request then succeeds");
    test_compute(f, sc, 3000.0f);
}

//
// child processes that must abort
//

// runs this binary in `mode` in a child process; true when the child was waited for
static bool run_child(const char * mode, int * status, std::string * err) {
    char path[4096];
    uint32_t len = sizeof(path);
    if (_NSGetExecutablePath(path, &len) != 0) {
        return false;
    }

    std::string mode_s = mode;
    char * argv[] = { path, mode_s.data(), nullptr };

    std::vector<char *> envp;
    char no_bt[] = "GGML_NO_BACKTRACE=1";
    envp.push_back(no_bt);
    for (char ** e = environ; *e != nullptr; e++) {
        envp.push_back(*e);
    }
    envp.push_back(nullptr);

    // the child's stderr comes back through a pipe, so the check sees which abort fired
    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, fds[0]);

    pid_t pid = 0;
    const bool spawned = posix_spawn(&pid, path, &actions, nullptr, argv, envp.data()) == 0;
    posix_spawn_file_actions_destroy(&actions);
    close(fds[1]);

    char bytes[4096];
    ssize_t n = 0;
    while ((n = read(fds[0], bytes, sizeof(bytes))) > 0) {
        err->append(bytes, (size_t) n);
    }
    close(fds[0]);

    return spawned && waitpid(pid, status, 0) == pid;
}

static bool aborted(int status) {
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

// child mode: encode a kernel that reads a view across the first split point; it must not return
static int encode_across(fixture & f) {
    ggml_tensor * cross = ggml_view_1d(f.ctx, f.all, kElems, f.sub - kBytes/2);
    ggml_set_name(cross, "cross");
    ggml_backend_view_init(cross);
    ggml_tensor * res = ggml_scale(f.ctx, cross, 2.0f);
    ggml_backend_tensor_alloc(f.buf, res, (uint8_t *) f.mem + kInOffs);
    ggml_cgraph * gf = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(gf, res);
    ggml_backend_graph_compute(f.backend, gf);
    printf("encoding a kernel on a view across a split point returned\n");
    return 0;
}

static void test_encode_across(void) {
    int status = 0;
    std::string err;
    const bool waited = run_child("encode-across", &status, &err);
    check(waited && aborted(status), "encode", "a kernel on a view across a split point aborts");
    check(err.find("tensor 'cross' crosses a split point") != std::string::npos, "encode", "the abort names the tensor and the split");
}

// child mode: with sub-range 1 unwrapped, a kernel over sub-ranges 2 and 0 computes, then encoding one that reads
// sub-range 1 must not return
static int encode_unwrapped(fixture & f) {
    if (!ggml_backend_metal_buffer_end_residency_range(f.buf, 1) || !ggml_backend_metal_buffer_unwrap_range(f.buf, 1)) {
        fprintf(stderr, "sub-range 1 was not unwrapped\n");
        return 1;
    }
    if (compute_around_1(f, graph_of(f, 2), 100.0f, false)) {
        fprintf(stderr, "the wrapped sub-ranges compute\n");
    }
    ggml_backend_graph_compute(f.backend, graph_of(f, 1));
    printf("encoding a kernel on an unwrapped sub-range returned\n");
    return 0;
}

static void test_encode_unwrapped(void) {
    const char * sc = "encode-unwrapped";
    int status = 0;
    std::string err;
    const bool waited = run_child("encode-unwrapped", &status, &err);
    check(err.find("the wrapped sub-ranges compute") != std::string::npos, sc, "a kernel over the wrapped sub-ranges computes");
    check(waited && aborted(status), sc, "a kernel on a tensor in the unwrapped sub-range aborts");
    check(err.find("tensor 'in-1' is in sub-range 1") != std::string::npos, sc, "the abort names the tensor and the sub-range");
}

int main(int argc, char ** argv) {
    fixture f;
    if (!f.init()) {
        if (f.backend == nullptr) {
            printf("SKIP no Metal device\n");
            return 0;
        }
        printf("fixture setup failed\n");
        return 1;
    }

    if (argc > 1 && strcmp(argv[1], "encode-across") == 0) {
        return encode_across(f);
    }
    if (argc > 1 && strcmp(argv[1], "encode-unwrapped") == 0) {
        return encode_unwrapped(f);
    }

    test_resolve(f, "resolve");
    test_compute(f, "compute", 1.0f);
    test_range_handoff(f);
    test_range_replace(f);
    test_unwrap_rewrap(f);
    test_range_refusals(f);
    test_validation(f);
    test_unsplit(f);
    test_encode_across();
    test_encode_unwrapped();

    if (g_n_fail > 0) {
        printf("%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
