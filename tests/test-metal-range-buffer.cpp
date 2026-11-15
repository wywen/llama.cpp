// Tests for buffers over caller host memory whose Metal buffers the caller supplies, one per range
// (ggml_backend_metal_buffer_from_ranges).
//
// The host memory has three ranges of four pages each, the last one ending short of a page boundary; the caller wraps
// each in a no-copy Metal buffer of its own. Views of a tensor over the whole memory resolve, through the range table,
// to the Metal buffer of the range that holds them. A graph scales a view in each range into a view in the next one,
// so each kernel binds two different Metal buffers.
//
// Each range's Metal buffer is also replaced on its own: while range 1 has none and its pages are mapped anew, kernels
// over the other ranges compute; after the caller installs a new Metal buffer over the new pages the GPU reads and
// writes them. Encoding a kernel on a tensor in a range with no Metal buffer, or on a view across a range boundary,
// aborts, naming the tensor; the binary checks this by running itself in a child process. The backend takes no
// reference to the caller's Metal buffers. It skips when there is no Metal device.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-metal.h"
#include "ggml-metal-device.h"

#include "metal-nocopy-buffer.h"

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
    printf("%s %-24s %s\n", ok ? "PASS" : "FAIL", scenario, what);
    if (!ok) {
        g_n_fail++;
    }
}

constexpr int     kRanges = 3;
constexpr int64_t kElems  = 256;
constexpr size_t  kBytes  = kElems * sizeof(float);
constexpr size_t  kInOffs = 256;  // a view's offset in its range
constexpr size_t  kTail   = 512;  // the last range ends this many bytes short of a page boundary

static ggml_metal_buffer_id id_of(const ggml_tensor * t) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    return ggml_metal_buffer_get_id((ggml_metal_buffer_t) buf->context, t);
}

struct fixture {
    ggml_backend_t        backend = nullptr;
    ggml_backend_dev_t    dev     = nullptr;
    ggml_context        * ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph         * gf      = nullptr;

    size_t page = 0;
    size_t sub  = 0; // bytes of each range but the last, and the pages each range's Metal buffer wraps
    size_t size = 0;
    void * mem  = nullptr;

    std::array<void *, kRanges> metal{};

    ggml_tensor * all = nullptr;
    std::array<ggml_tensor *, kRanges> in{};
    std::array<ggml_tensor *, kRanges> out{};

    bool init() {
        backend = ggml_backend_metal_init();
        if (backend == nullptr) {
            return false;
        }
        dev = ggml_backend_get_device(backend);

        page = (size_t) sysconf(_SC_PAGESIZE);
        sub  = 4*page;
        size = kRanges*sub - kTail;
        // mapped rather than allocated, so that a range's pages can be replaced with MAP_FIXED
        mem = mmap(nullptr, kRanges*sub, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (mem == MAP_FAILED) {
            mem = nullptr;
            return false;
        }

        std::array<ggml_backend_metal_range, kRanges> ranges{};
        for (int i = 0; i < kRanges; i++) {
            metal[i] = test_metal_nocopy_buffer_new(dev, range_base(i), sub);
            ranges[i] = { i == kRanges - 1 ? sub - kTail : sub, metal[i] };
        }

        const size_t refs = test_metal_buffer_retain_count(metal[0]);
        buf = ggml_backend_metal_buffer_from_ranges(dev, mem, size, ranges.data(), kRanges);
        check(buf != nullptr, "create", "a buffer over three ranges is created");
        check(test_metal_buffer_retain_count(metal[0]) == refs, "create", "the backend takes no reference to a range's Metal buffer");
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
        check(ggml_backend_tensor_alloc(buf, all, base) == GGML_STATUS_SUCCESS, "create", "a tensor over the whole memory is allocated");

        for (int i = 0; i < kRanges; i++) {
            in[i] = ggml_view_1d(ctx, all, kElems, i*sub + kInOffs);
            ggml_format_name(in[i], "in-%d", i);
            check(ggml_backend_view_init(in[i]) == GGML_STATUS_SUCCESS, "create", "a view in a range is initialized");
        }
        for (int i = 0; i < kRanges; i++) {
            out[i] = ggml_scale(ctx, in[i], 2.0f + (float) i);
            ggml_format_name(out[i], "out-%d", i);
        }
        gf = ggml_new_graph(ctx);
        for (int i = 0; i < kRanges; i++) {
            // the result lands at the end of the next range, which for the last one is its unaligned end
            const int    j   = (i + 1) % kRanges;
            const size_t end = j == kRanges - 1 ? size : (j + 1)*sub;
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
        for (void * m : metal) {
            if (m != nullptr) {
                test_metal_buffer_release(m);
            }
        }
        if (mem != nullptr) {
            munmap(mem, kRanges*sub);
        }
    }

    uint8_t * range_base(int i) const {
        return (uint8_t *) mem + i*sub;
    }

    // replaces range i's Metal buffer with a new one over its pages
    bool rewrap(int i) {
        void * m = test_metal_nocopy_buffer_new(dev, range_base(i), sub);
        if (m == nullptr || !ggml_backend_metal_buffer_set_range(buf, i, m)) {
            if (m != nullptr) {
                test_metal_buffer_release(m);
            }
            return false;
        }
        if (metal[i] != nullptr) {
            test_metal_buffer_release(metal[i]);
        }
        metal[i] = m;
        return true;
    }

    // leaves range i with no Metal buffer; nothing in flight binds it
    bool unwrap(int i) {
        if (!ggml_backend_metal_buffer_set_range(buf, i, nullptr)) {
            return false;
        }
        test_metal_buffer_release(metal[i]);
        metal[i] = nullptr;
        return true;
    }
};

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

// a graph that computes out[i] alone: it reads range i and writes range i + 1
static ggml_cgraph * graph_of(fixture & f, int i) {
    ggml_cgraph * gf = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(gf, f.out[i]);
    return gf;
}

static void test_resolve(fixture & f, const char * scenario) {
    std::set<void *> seen;
    bool found = true;
    bool offs  = true;
    for (int i = 0; i < kRanges; i++) {
        for (const ggml_tensor * t : { (const ggml_tensor *) f.in[i], (const ggml_tensor *) f.out[(i + kRanges - 1) % kRanges] }) {
            const ggml_metal_buffer_id id = id_of(t);
            found = found && id.metal == f.metal[i];
            offs  = offs  && id.offs == (size_t) ((const uint8_t *) t->data - f.range_base(i));
            seen.insert(id.metal);
        }
    }
    check(found, scenario, "a tensor resolves to the Metal buffer of its range");
    check(offs, scenario, "at its offset in the range");
    check(seen.size() == (size_t) kRanges, scenario, "each range has its own Metal buffer");
}

static void test_compute(fixture & f, const char * scenario, float base) {
    std::array<std::vector<float>, kRanges> src;
    for (int i = 0; i < kRanges; i++) {
        src[i] = ramp(base + 1000.0f*i);
        ggml_backend_tensor_set(f.in[i], src[i].data(), 0, kBytes);
        ggml_backend_tensor_memset(f.out[i], 0xff, 0, kBytes);
    }

    check(ggml_backend_graph_compute(f.backend, f.gf) == GGML_STATUS_SUCCESS, scenario, "the graph computes");

    bool ok = true;
    for (int i = 0; i < kRanges; i++) {
        ok = ok && holds(f.out[i], src[i], 2.0f + (float) i);
    }
    check(ok, scenario, "each kernel reads one range and writes the next");
}

// range 1 loses its Metal buffer while work over ranges 2 and 0 is in flight, its pages are mapped anew, and the GPU
// reads and writes the new pages once the caller installs a Metal buffer over them
static void test_replace(fixture & f) {
    const char * sc = "replace";

    // bytes the GPU must not see once range 1 is replaced
    const std::vector<float> stale = ramp(-3000.0f);
    ggml_backend_tensor_set(f.in[1], stale.data(), 0, kBytes);

    void * const metal0 = f.metal[0];
    void * const metal2 = f.metal[2];

    ggml_cgraph * gf = graph_of(f, 2);

    const std::vector<float> early = ramp(100.0f);
    ggml_backend_tensor_set(f.in[2], early.data(), 0, kBytes);
    check(ggml_backend_graph_compute_async(f.backend, gf) == GGML_STATUS_SUCCESS, sc, "work over ranges 2 and 0 is submitted");
    check(f.unwrap(1), sc, "range 1's Metal buffer is removed meanwhile");
    ggml_backend_synchronize(f.backend);
    check(holds(f.out[2], early, 4.0f), sc, "the work in flight over ranges 2 and 0 computes correctly");

    check(id_of(f.in[0]).metal == metal0 && id_of(f.in[2]).metal == metal2, sc, "ranges 0 and 2 keep their Metal buffers");

    void * p = mmap(f.range_base(1), f.sub, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0);
    check(p == f.range_base(1), sc, "range 1's pages are replaced with MAP_FIXED");

    const std::vector<float> mid = ramp(200.0f);
    ggml_backend_tensor_set(f.in[2], mid.data(), 0, kBytes);
    check(ggml_backend_graph_compute(f.backend, gf) == GGML_STATUS_SUCCESS && holds(f.out[2], mid, 4.0f), sc,
            "a kernel over ranges 2 and 0 computes while range 1 has no Metal buffer");

    // host access to range 1 works without a Metal buffer: new bytes go into its new pages
    const std::vector<float> fresh = ramp(5000.0f);
    ggml_backend_tensor_set(f.in[1], fresh.data(), 0, kBytes);
    ggml_backend_tensor_memset(f.out[0], 0xff, 0, kBytes);

    check(f.rewrap(1), sc, "range 1 gets a new Metal buffer over its new pages");
    check(id_of(f.in[1]).metal == f.metal[1], sc, "range 1 resolves to the new Metal buffer");

    const std::vector<float> src0 = ramp(9000.0f);
    const std::vector<float> src2 = ramp(-9000.0f);
    ggml_backend_tensor_set(f.in[0], src0.data(), 0, kBytes);
    ggml_backend_tensor_set(f.in[2], src2.data(), 0, kBytes);
    ggml_backend_tensor_memset(f.out[1], 0xff, 0, kBytes);
    ggml_backend_tensor_memset(f.out[2], 0xff, 0, kBytes);
    check(ggml_backend_graph_compute(f.backend, f.gf) == GGML_STATUS_SUCCESS, sc, "the whole graph computes");
    check(holds(f.out[1], fresh, 3.0f), sc, "the GPU reads the new bytes of the replaced range");
    check(holds(f.out[0], src0, 2.0f), sc, "the GPU writes into the replaced range's new pages");
    check(holds(f.out[2], src2, 4.0f), sc, "the other ranges still compute");
}

// a Metal buffer may start before its range: here one Metal buffer over all pages backs every range
static void test_shared_metal(fixture & f) {
    const char * sc = "shared-metal";

    void * whole = test_metal_nocopy_buffer_new(f.dev, f.mem, kRanges*f.sub);
    const std::array<ggml_backend_metal_range, 2> ranges = {{ { f.page + kInOffs, whole }, { 2*f.page - kInOffs, whole } }};
    ggml_backend_buffer_t b = ggml_backend_metal_buffer_from_ranges(f.dev, f.mem, 3*f.page, ranges.data(), 2);
    check(b != nullptr, sc, "ranges off page boundaries inside one Metal buffer are accepted");
    if (b != nullptr) {
        ggml_tensor * t = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 16);
        ggml_backend_tensor_alloc(b, t, (uint8_t *) f.mem + 2*f.page);
        const ggml_metal_buffer_id id = id_of(t);
        check(id.metal == whole && id.offs == 2*f.page, sc, "a tensor in the second range resolves at its offset in the Metal buffer");
        ggml_backend_buffer_free(b);
    }
    test_metal_buffer_release(whole);
}

static void test_validation(fixture & f) {
    const char * sc = "validation";
    const size_t page = f.page;

    auto refused = [&](ggml_backend_dev_t dev, void * base, size_t size, std::vector<ggml_backend_metal_range> ranges, int n, const char * what) {
        ggml_backend_buffer_t b = ggml_backend_metal_buffer_from_ranges(dev, base, size, ranges.empty() ? nullptr : ranges.data(), n);
        check(b == nullptr, sc, what);
        if (b != nullptr) {
            ggml_backend_buffer_free(b);
        }
    };

    void * m0 = f.metal[0];
    refused(f.dev, f.mem, 2*page, { { page, nullptr } }, 1, "ranges short of the buffer are refused");
    refused(f.dev, f.mem, 2*page, { { page, nullptr }, { 2*page, nullptr } }, 2, "ranges past the buffer are refused");
    refused(f.dev, f.mem, 2*page, { { 0, nullptr }, { 2*page, nullptr } }, 2, "an empty range is refused");
    refused(f.dev, f.mem, 2*page, {}, 0, "no ranges are refused");
    refused(f.dev, f.mem, 2*page, { { 2*page, nullptr } }, -1, "a negative count is refused");
    refused(f.dev, nullptr, 2*page, { { 2*page, nullptr } }, 1, "a null base is refused");
    refused(f.dev, f.mem, 0, { { 0, nullptr } }, 1, "an empty buffer is refused");
    refused(f.dev, f.range_base(1), f.sub, { { f.sub, m0 } }, 1, "a Metal buffer over other memory is refused");
    refused(f.dev, f.mem, 8*page, { { 8*page, m0 } }, 1, "a Metal buffer shorter than its range is refused");

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        refused(cpu, f.mem, 2*page, { { 2*page, nullptr } }, 1, "a non-Metal device is refused");
        check(ggml_backend_metal_device_get_mtl_device(cpu) == nullptr, sc, "a non-Metal device has no MTLDevice");
    }
    check(ggml_backend_metal_device_get_mtl_device(f.dev) != nullptr, sc, "the Metal device has its MTLDevice");

    ggml_backend_buffer_t b = ggml_backend_metal_buffer_from_ranges(f.dev, f.mem, f.sub, std::vector<ggml_backend_metal_range>{ { f.sub, nullptr } }.data(), 1);
    check(b != nullptr, sc, "a buffer whose one range has no Metal buffer is created");
    if (b != nullptr) {
        check(!ggml_backend_metal_buffer_set_range(b, 1, m0) && !ggml_backend_metal_buffer_set_range(b, -1, m0), sc, "an index out of range is refused");
        check(!ggml_backend_metal_buffer_set_range(b, 0, f.metal[1]), sc, "a Metal buffer over another range is refused");
        check(ggml_backend_metal_buffer_set_range(b, 0, m0), sc, "a Metal buffer over the range is installed");
        ggml_backend_buffer_free(b);
    }

    ggml_backend_buffer_t plain = ggml_backend_dev_buffer_from_host_ptr(f.dev, f.mem, f.sub, f.sub);
    check(plain != nullptr && !ggml_backend_metal_buffer_set_range(plain, 0, m0), sc, "setting a range of a host-pointer buffer is refused");
    if (plain != nullptr) {
        ggml_backend_buffer_free(plain);
    }
    check(!ggml_backend_metal_buffer_set_range(nullptr, 0, m0), sc, "a null buffer is refused");
}

// a buffer type the caller makes for the Metal device is supported by it, one for another device is not
static void test_caller_buft(fixture & f) {
    const char * sc = "caller-buft";

    ggml_backend_buffer_type buft = {};
    buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "caller"; };
    buft.device = f.dev;
    check(ggml_backend_dev_supports_buft(f.dev, &buft), sc, "a caller's buffer type for the device is supported");

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        buft.device = cpu;
        check(!ggml_backend_dev_supports_buft(f.dev, &buft), sc, "one for another device is not");
    }
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

// child mode: encode a kernel that reads a view across the first range boundary; it must not return
static int encode_across(fixture & f) {
    ggml_tensor * cross = ggml_view_1d(f.ctx, f.all, kElems, f.sub - kBytes/2);
    ggml_set_name(cross, "cross");
    ggml_backend_view_init(cross);
    ggml_tensor * res = ggml_scale(f.ctx, cross, 2.0f);
    ggml_backend_tensor_alloc(f.buf, res, (uint8_t *) f.mem + kInOffs);
    ggml_cgraph * gf = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(gf, res);
    ggml_backend_graph_compute(f.backend, gf);
    printf("encoding a kernel on a view across a range boundary returned\n");
    return 0;
}

// child mode: with range 1 left without a Metal buffer, a kernel over ranges 2 and 0 computes, then encoding one that
// reads range 1 must not return
static int encode_missing(fixture & f) {
    if (!f.unwrap(1)) {
        fprintf(stderr, "range 1 kept its Metal buffer\n");
        return 1;
    }
    const std::vector<float> src = ramp(100.0f);
    ggml_backend_tensor_set(f.in[2], src.data(), 0, kBytes);
    if (ggml_backend_graph_compute(f.backend, graph_of(f, 2)) == GGML_STATUS_SUCCESS && holds(f.out[2], src, 4.0f)) {
        fprintf(stderr, "the other ranges compute\n");
    }
    ggml_backend_graph_compute(f.backend, graph_of(f, 1));
    printf("encoding a kernel on a range with no Metal buffer returned\n");
    return 0;
}

static void test_encode_across(void) {
    const char * sc = "encode-across";
    int status = 0;
    std::string err;
    const bool waited = run_child("encode-across", &status, &err);
    check(waited && aborted(status), sc, "a kernel on a view across a range boundary aborts");
    check(err.find("tensor 'cross' crosses a range boundary") != std::string::npos, sc, "the abort names the tensor");
}

static void test_encode_missing(void) {
    const char * sc = "encode-missing";
    int status = 0;
    std::string err;
    const bool waited = run_child("encode-missing", &status, &err);
    check(err.find("the other ranges compute") != std::string::npos, sc, "a kernel over the other ranges computes");
    check(waited && aborted(status), sc, "a kernel on a tensor in a range with no Metal buffer aborts");
    check(err.find("tensor 'in-1' is in range 1") != std::string::npos, sc, "the abort names the tensor and the range");
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
    if (argc > 1 && strcmp(argv[1], "encode-missing") == 0) {
        return encode_missing(f);
    }

    test_resolve(f, "resolve");
    test_compute(f, "compute", 1.0f);
    test_replace(f);
    test_resolve(f, "resolve-after-replace");
    test_shared_metal(f);
    test_validation(f);
    test_caller_buft(f);
    test_encode_across();
    test_encode_missing();

    const size_t refs = test_metal_buffer_retain_count(f.metal[0]);
    ggml_backend_buffer_free(f.buf);
    f.buf = nullptr;
    check(test_metal_buffer_retain_count(f.metal[0]) == refs, "free", "freeing the buffer releases nothing of the caller's");

    if (g_n_fail > 0) {
        printf("%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("all checks passed\n");
    return 0;
}
