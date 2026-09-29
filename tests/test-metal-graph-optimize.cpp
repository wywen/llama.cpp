// Tests for the Metal graph reorder's boundary barriers.
//
// ggml_graph_optimize hoists memory-concurrent nodes forward to widen the concurrent set.
// A caller that installs a boundary-event schedule and an encode window cuts the graph at
// the per-layer output markers, and the reorder runs before that schedule exists, so left
// alone it can move a node out of the segment that is supposed to produce it. These tests
// pin both halves of the fix: with the barriers off the hoist still happens (the reorder
// is untouched), and with them on nothing crosses a marker.
//
// Everything here uses hand-built graphs; scheduler cases allocate real CPU buffers.

#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal-common.h"
#include "ggml-metal.h"
#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

static int g_n_fail = 0;

static std::vector<std::string> node_names(ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(gf);

    std::vector<std::string> res;
    res.reserve(n);

    for (int i = 0; i < n; i++) {
        res.push_back(ggml_get_name(ggml_graph_node(gf, i)));
    }

    return res;
}

static std::string join(const std::vector<std::string> & names) {
    std::string res;

    for (size_t i = 0; i < names.size(); i++) {
        if (i > 0) {
            res += " ";
        }
        res += names[i];
    }

    return res;
}

static void check_order(const char * scenario, ggml_cgraph * gf, const std::vector<std::string> & expected) {
    const std::vector<std::string> actual = node_names(gf);

    if (actual == expected) {
        printf("PASS %-40s %s\n", scenario, join(actual).c_str());
        return;
    }

    printf("FAIL %-40s got [%s], expected [%s]\n", scenario, join(actual).c_str(), join(expected).c_str());

    g_n_fail++;
}

// a scratch context big enough for the tiny graphs below
struct test_ctx {
    ggml_context * ctx;

    test_ctx() {
        ggml_init_params params = {
            /*.mem_size   =*/ 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(params);
    }

    ~test_ctx() {
        ggml_free(ctx);
    }
};

static ggml_tensor * leaf(ggml_context * ctx, const char * name) {
    ggml_tensor * res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_set_name(res, name);

    return res;
}

static ggml_tensor * named(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);

    return t;
}

// An independent lookup into a persistent cache: nothing in the graph anchors it, which is
// exactly the node the reorder is free to hoist as far as it likes.
static ggml_tensor * cache_lookup(ggml_context * ctx, const char * name) {
    ggml_tensor * cache = leaf(ctx, "cache");
    ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_name(ids, "ids");

    return named(ggml_get_rows(ctx, cache, ids), name);
}

// n0 writes "a", n1 reads it -- n1 therefore conflicts with the concurrent set n0 opened,
// which is what makes the reorder look forward for something to hoist in front of n1.
//
//   a  -> mul(x, y)
//   b  -> mul(a, w)      conflicts with a
//   mk -> add(b, x)      the boundary marker
//   c  -> get_rows(...)  independent, and after the marker
static ggml_cgraph * build_hoist_across(ggml_context * ctx, const char * marker_name) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");
    ggml_tensor * w = leaf(ctx, "w");

    ggml_tensor * a  = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * b  = named(ggml_mul(ctx, a, w), "b");
    ggml_tensor * mk = named(ggml_add(ctx, b, x), marker_name);
    ggml_tensor * c  = cache_lookup(ctx, "c");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_build_forward_expand(gf, mk);
    ggml_build_forward_expand(gf, c);

    return gf;
}

// Same idea, but the marker is itself the node that opens the look-forward scan. Everything
// the scan picks up is emitted BEFORE the marker, so the scan has to be skipped entirely.
//
//   a  -> mul(x, y)
//   mk -> add(a, x)      the boundary marker, and the conflicting node
//   c  -> get_rows(...)  independent, and after the marker
static ggml_cgraph * build_marker_anchor(ggml_context * ctx) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");

    ggml_tensor * a  = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * mk = named(ggml_add(ctx, a, x), "l_out-0");
    ggml_tensor * c  = cache_lookup(ctx, "c");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, mk);
    ggml_build_forward_expand(gf, c);

    return gf;
}

// The marker packed into a fused group as a non-head member. The reorder moves whole groups,
// so the marker rides along with "s" unless the barrier test looks past the group's head.
//
//   a  -> mul(x, y)
//   b  -> mul(a, w)      conflicts with a
//   s  -> add(p, q)      independent, fusable head
//   mk -> add(s, v)      the boundary marker, fused onto s
static ggml_cgraph * build_fused_group(ggml_context * ctx) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");
    ggml_tensor * w = leaf(ctx, "w");
    ggml_tensor * p = leaf(ctx, "p");
    ggml_tensor * q = leaf(ctx, "q");
    ggml_tensor * v = leaf(ctx, "v");

    ggml_tensor * a  = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * b  = named(ggml_mul(ctx, a, w), "b");
    ggml_tensor * s  = named(ggml_add(ctx, p, q), "s");
    ggml_tensor * mk = named(ggml_add(ctx, s, v), "l_out-0");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_build_forward_expand(gf, s);
    ggml_build_forward_expand(gf, mk);

    return gf;
}

struct completion_probe {
    bool     woke_before_latch = false;
    uint64_t published[16]     = {};
    size_t   published_count   = 0;
    bool     overflow          = false;
    bool     failed            = false;
};

static void completion_publish(void * user, void *, uint64_t value) {
    auto * probe = static_cast<completion_probe *>(user);
    if (probe->published_count < std::size(probe->published)) {
        probe->published[probe->published_count++] = value;
    } else {
        probe->overflow = true;
    }
    if (value == UINT64_MAX && !probe->failed) {
        probe->woke_before_latch = true;
    }
}

static void completion_latch(void * user) {
    static_cast<completion_probe *>(user)->failed = true;
}

static void check_completion(bool condition, const char * scenario) {
    if (!condition) {
        printf("FAIL completion: %s\n", scenario);
        g_n_fail++;
    }
}

static int g_completion_event_refs = 0;

static void completion_retain_event(void *) {
    ++g_completion_event_refs;
}

static void completion_release_event(void *) {
    --g_completion_event_refs;
}

static void test_completion_growth() {
    auto * results = ggml_metal_result_storage_new();
    check_completion(results && ggml_metal_result_storage_reserve(results, 1), "result first slab");
    auto * original  = ggml_metal_result_storage_at(results, 0);
    original->status = 42;
    check_completion(ggml_metal_result_storage_reserve(results, 3), "result slab grows with pending pointer");
    check_completion(ggml_metal_result_storage_at(results, 0) == original && original->status == 42,
                     "callback-captured result pointer stays stable");
    check_completion(!ggml_metal_result_storage_reserve(results, SIZE_MAX), "result overflow refuses growth");
    check_completion(ggml_metal_result_storage_at(results, 0) == original && original->status == 42,
                     "failed result growth keeps earlier callback storage");
    check_completion(ggml_metal_result_storage_reserve(results, 3), "warm result reuse");
    ggml_metal_result_storage_free(results);
}

struct scheduler_probe {
    ggml_backend_t      backends[2]         = {};
    const ggml_tensor * nodes[3]            = {};
    int                 preflights          = 0;
    int                 finishes            = 0;
    int                 n_submissions       = -1;
    int                 first_submissions   = 0;
    int                 observed_splits     = 0;
    enum ggml_status    refusal             = GGML_STATUS_SUCCESS;
    enum ggml_status    result              = GGML_STATUS_FAILED;
    bool                valid               = true;
    bool                graph_seen          = false;
    int                 eval_after          = 0;
    int                 expected_eval_after = 0;
};

static void scheduler_graph_ready(ggml_backend_sched_t, void * user_data) {
    static_cast<scheduler_probe *>(user_data)->graph_seen = true;
}

static enum ggml_status scheduler_preflight(ggml_backend_sched_t sched, void * user_data) {
    scheduler_probe & p = *static_cast<scheduler_probe *>(user_data);
    ++p.preflights;
    p.observed_splits = ggml_backend_sched_preflight_n_splits(sched);
    p.valid &= p.graph_seen;
    p.valid &= p.observed_splits == 3;
    p.valid &= ggml_backend_sched_preflight_n_splits(nullptr) == -1;
    ggml_backend_sched_split_manifest absent_split = {};
    p.valid &= !ggml_backend_sched_preflight_split(nullptr, 0, &absent_split);
    p.valid &= ggml_backend_sched_preflight_node(nullptr, 0, 0) == nullptr;
    ggml_backend_sched_input_manifest absent_input = {};
    p.valid &= !ggml_backend_sched_preflight_input(nullptr, 0, 0, &absent_input);
    for (int i = 0; i < p.observed_splits; ++i) {
        ggml_backend_sched_split_manifest split = {};
        p.valid &= ggml_backend_sched_preflight_split(sched, i, &split);
        if (i >= 3) {
            continue;
        }
        p.valid &= split.backend == p.backends[i % 2] && split.n_nodes == 1 && split.n_inputs == 0;
        p.valid &= ggml_backend_sched_preflight_node(sched, i, 0) == p.nodes[i];
        p.valid &= ggml_backend_sched_preflight_node(sched, i, 1) == nullptr;
        ggml_backend_sched_input_manifest input = {};
        p.valid &= !ggml_backend_sched_preflight_input(sched, i, 0, &input);
        p.valid &= !ggml_backend_sched_preflight_split(sched, -1, &split);
        p.valid &= !ggml_backend_sched_preflight_split(sched, p.observed_splits, &split);
        p.valid &= !ggml_backend_sched_preflight_split(sched, i, nullptr);
        p.valid &= ggml_backend_sched_preflight_node(sched, i, -1) == nullptr;
        p.valid &= ggml_backend_sched_preflight_node(sched, -1, 0) == nullptr;
        p.valid &= !ggml_backend_sched_preflight_input(sched, i, -1, &input);
        p.valid &= !ggml_backend_sched_preflight_input(sched, -1, 0, &input);
        p.valid &= !ggml_backend_sched_preflight_input(sched, i, 0, nullptr);
    }
    return p.refusal;
}

static void scheduler_first_submit(ggml_backend_sched_t, enum ggml_status, void * user_data) {
    ++static_cast<scheduler_probe *>(user_data)->first_submissions;
}

static void scheduler_finished(ggml_backend_sched_t sched,
                               enum ggml_status     result,
                               int                  n_submissions,
                               void *               user_data) {
    scheduler_probe & p = *static_cast<scheduler_probe *>(user_data);
    ++p.finishes;
    p.result        = result;
    p.n_submissions = n_submissions;
    p.valid &= p.eval_after == p.expected_eval_after;
    p.valid &= ggml_backend_sched_preflight_n_splits(sched) == -1;
}

static bool scheduler_eval(ggml_tensor *, bool ask, void * user_data) {
    if (!ask) {
        ++static_cast<scheduler_probe *>(user_data)->eval_after;
    }
    return true;
}

static enum ggml_status (*scheduler_cpu_compute)(ggml_backend_t, ggml_cgraph *) = nullptr;
static int scheduler_cpu_calls                                                  = 0;
static int scheduler_cpu_fail_on_call                                           = 0;

static enum ggml_status scheduler_cpu_inject(ggml_backend_t backend, ggml_cgraph * graph) {
    ++scheduler_cpu_calls;
    if (scheduler_cpu_calls == scheduler_cpu_fail_on_call) {
        return GGML_STATUS_FAILED;
    }
    return scheduler_cpu_compute(backend, graph);
}

static void test_scheduler_manifest() {
    for (int scenario = 0; scenario < 6; ++scenario) {
        test_ctx      tc;
        ggml_tensor * x     = leaf(tc.ctx, "x");
        ggml_tensor * y     = leaf(tc.ctx, "y");
        ggml_tensor * a     = named(ggml_add(tc.ctx, x, y), "a");
        ggml_tensor * b     = named(ggml_mul(tc.ctx, a, y), "b");
        ggml_tensor * c     = named(ggml_add(tc.ctx, b, x), "c");
        ggml_cgraph * graph = ggml_new_graph(tc.ctx);
        ggml_build_forward_expand(graph, c);
        scheduler_probe p;
        p.nodes[0]                                   = a;
        p.nodes[1]                                   = b;
        p.nodes[2]                                   = c;
        p.backends[0]                                = ggml_backend_cpu_init();
        p.backends[1]                                = ggml_backend_cpu_init();
        ggml_backend_buffer_type   distinct_cpu_buft = *ggml_backend_cpu_buffer_type();
        ggml_backend_buffer_type_t bufts[]           = { ggml_backend_cpu_buffer_type(), &distinct_cpu_buft };
        ggml_backend_sched_t       sched             = ggml_backend_sched_new(p.backends, bufts, 2, 32, false, false);
        ggml_backend_sched_set_tensor_backend(sched, a, p.backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, b, p.backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, c, p.backends[0]);
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            check_completion(false, "CPU graph allocation with forced backend assignments");
            ggml_backend_sched_free(sched);
            ggml_backend_free(p.backends[0]);
            ggml_backend_free(p.backends[1]);
            continue;
        }
        const float values[16] = {};
        ggml_backend_tensor_set(x, values, 0, sizeof(values));
        ggml_backend_tensor_set(y, values, 0, sizeof(values));
        ggml_backend_sched_set_graph_compute_callback(sched, scheduler_graph_ready, &p);
        ggml_backend_sched_set_preflight_callback(sched, scheduler_preflight, &p);
        ggml_backend_sched_set_graph_submit_callback(sched, scheduler_first_submit, &p);
        ggml_backend_sched_set_submission_finished_callback(sched, scheduler_finished, &p);
        if (scenario >= 4) {
            ggml_backend_sched_set_eval_callback(sched, scheduler_eval, &p);
            p.expected_eval_after = scenario == 4 ? 3 : 1;
        }
        p.refusal                                    = scenario == 1 ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS;
        scheduler_cpu_calls                          = 0;
        scheduler_cpu_fail_on_call                   = scenario == 2 || scenario == 5 ? 2 : scenario == 3 ? 1 : 0;
        scheduler_cpu_compute                        = p.backends[1]->iface.graph_compute;
        p.backends[1]->iface.graph_compute           = scheduler_cpu_inject;
        p.backends[0]->iface.graph_compute           = scheduler_cpu_inject;
        enum ggml_status result                      = ggml_backend_sched_graph_compute_async(sched, graph);
        const int        expected_successful_calls[] = { 3, 0, 1, 0, 3, 1 };
        const int        expected_attempted_calls[]  = { 3, 0, 2, 1, 3, 2 };

        const enum ggml_status expected_status[] = { GGML_STATUS_SUCCESS, GGML_STATUS_ABORTED, GGML_STATUS_FAILED,
                                                     GGML_STATUS_FAILED,  GGML_STATUS_SUCCESS, GGML_STATUS_FAILED };

        if (!(p.valid && p.observed_splits == 3 && p.n_submissions == expected_successful_calls[scenario])) {
            printf(
                "scheduler scenario %d: splits=%d valid=%d status=%d preflights=%d finishes=%d accepted=%d calls=%d\n",
                scenario, p.observed_splits, p.valid, result, p.preflights, p.finishes, p.n_submissions,
                scheduler_cpu_calls);
        }
        check_completion(p.valid && p.preflights == 1 && p.finishes == 1 && p.result == result,
                         "CPU optimized split manifest and once finish");
        check_completion(result == expected_status[scenario], "CPU refusal and partial failure status");
        check_completion(p.n_submissions == expected_successful_calls[scenario],
                         "CPU accepted submissions before finish");
        check_completion(p.first_submissions == 1 && scheduler_cpu_calls == expected_attempted_calls[scenario],
                         "CPU preflight refusal prevents backend compute");
        p.backends[1]->iface.graph_compute = scheduler_cpu_compute;
        p.backends[0]->iface.graph_compute = scheduler_cpu_compute;
        ggml_backend_sched_synchronize(sched);
        check_completion(p.finishes == 1, "CPU finish precedes explicit backend sync");
        ggml_backend_sched_split_manifest inactive_split = {};
        ggml_backend_sched_input_manifest inactive_input = {};
        check_completion(ggml_backend_sched_preflight_n_splits(sched) == -1 &&
                             !ggml_backend_sched_preflight_split(sched, 0, &inactive_split) &&
                             ggml_backend_sched_preflight_node(sched, 0, 0) == nullptr &&
                             !ggml_backend_sched_preflight_input(sched, 0, 0, &inactive_input),
                         "CPU manifest unavailable outside preflight");
        ggml_backend_sched_free(sched);
        ggml_backend_free(p.backends[0]);
        ggml_backend_free(p.backends[1]);
    }
}

static bool scheduler_copy_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == static_cast<ggml_backend_buffer_type_t>(dev->context);
}

struct scheduler_copy_probe {
    ggml_tensor *       a;
    ggml_tensor *       b;
    ggml_tensor *       c;
    const ggml_tensor * copies[2]         = {};
    bool                valid             = true;
    int                 preflights        = 0;
    int                 finishes          = 0;
    int                 first_submissions = 0;
    int                 successful_calls  = -1;
    enum ggml_status    result            = GGML_STATUS_FAILED;
    int                 round             = 0;
    enum ggml_status    refusal           = GGML_STATUS_SUCCESS;
};

static enum ggml_status scheduler_copy_preflight(ggml_backend_sched_t sched, void * user_data) {
    auto & p = *static_cast<scheduler_copy_probe *>(user_data);
    ++p.preflights;
    p.valid &= ggml_backend_sched_preflight_n_splits(sched) == 3;
    bool found[2] = {};
    for (int s = 0; s < ggml_backend_sched_preflight_n_splits(sched); ++s) {
        ggml_backend_sched_split_manifest split = {};
        p.valid &= ggml_backend_sched_preflight_split(sched, s, &split);
        for (int i = 0; i < split.n_inputs; ++i) {
            ggml_backend_sched_input_manifest input = {};
            p.valid &= ggml_backend_sched_preflight_input(sched, s, i, &input);
            for (int j = 0; j < 2; ++j) {
                ggml_tensor * source   = j == 0 ? p.a : p.b;
                ggml_tensor * consumer = j == 0 ? p.b : p.c;
                if (input.source == source) {
                    found[j]    = true;
                    p.copies[j] = input.copy;
                    p.valid &= input.copy != source && input.copy == consumer->src[0];
                    p.valid &= input.kind == GGML_BACKEND_SCHED_TRANSFER_ASYNC_OR_SYNC;
                    p.valid &= split.n_nodes == 1 && ggml_backend_sched_preflight_node(sched, s, 0) == consumer;
                    if (input.copy) {
                        const std::string name   = ggml_get_name(input.copy);
                        const std::string suffix = "#" + std::to_string(p.round);
                        p.valid &= name.size() >= suffix.size() &&
                                   name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
                    } else {
                        p.valid = false;
                    }
                }
            }
            ggml_backend_sched_input_manifest invalid_input = {};
            p.valid &= !ggml_backend_sched_preflight_input(sched, s, -1, &invalid_input);
            p.valid &= !ggml_backend_sched_preflight_input(sched, s, split.n_inputs, &invalid_input);
            p.valid &= !ggml_backend_sched_preflight_input(sched, s, i, nullptr);
        }
    }
    p.valid &= found[0] && found[1];
    return p.refusal;
}

static void scheduler_copy_first_submit(ggml_backend_sched_t, enum ggml_status, void * user_data) {
    ++static_cast<scheduler_copy_probe *>(user_data)->first_submissions;
}

static void scheduler_copy_finished(ggml_backend_sched_t sched, enum ggml_status result, int calls, void * user_data) {
    auto & p = *static_cast<scheduler_copy_probe *>(user_data);
    ++p.finishes;
    p.result           = result;
    p.successful_calls = calls;
    p.valid &= ggml_backend_sched_preflight_n_splits(sched) == -1;
}

static void test_scheduler_copies() {
    ggml_backend_t             backends[]    = { ggml_backend_cpu_init(), ggml_backend_cpu_init() };
    ggml_backend_buffer_type   distinct_buft = *ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t bufts[]       = { ggml_backend_cpu_buffer_type(), &distinct_buft };
    ggml_backend_device        devices[]     = { *backends[0]->device, *backends[1]->device };
    for (int i = 0; i < 2; ++i) {
        devices[i].iface.supports_buft = scheduler_copy_supports_buft;
        devices[i].context             = bufts[i];
        backends[i]->device            = &devices[i];
    }
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 32, true, false);
    check_completion(ggml_backend_sched_get_n_copies(sched) == 4 && !backends[0]->iface.cpy_tensor_async &&
                         !backends[1]->iface.cpy_tensor_async,
                     "CPU parallel copies use synchronous fallback");
    scheduler_cpu_compute            = backends[0]->iface.graph_compute;
    scheduler_cpu_fail_on_call       = 0;
    backends[0]->iface.graph_compute = scheduler_cpu_inject;
    backends[1]->iface.graph_compute = scheduler_cpu_inject;
    for (int round = 0; round < 4; ++round) {
        if (round > 0) {
            ggml_backend_sched_reset(sched);
        }
        test_ctx      tc;
        ggml_tensor * x     = leaf(tc.ctx, "copy-x");
        ggml_tensor * y     = leaf(tc.ctx, "copy-y");
        ggml_tensor * a     = named(ggml_add(tc.ctx, x, y), "copy-a");
        ggml_tensor * b     = named(ggml_mul(tc.ctx, a, y), "copy-b");
        ggml_tensor * c     = named(ggml_add(tc.ctx, b, x), "copy-c");
        ggml_cgraph * graph = ggml_new_graph(tc.ctx);
        ggml_build_forward_expand(graph, c);
        ggml_backend_sched_set_tensor_backend(sched, a, backends[0]);
        ggml_backend_sched_set_tensor_backend(sched, b, backends[1]);
        ggml_backend_sched_set_tensor_backend(sched, c, backends[0]);
        scheduler_copy_probe p = { a, b, c };
        p.round                = round;
        ggml_backend_sched_set_preflight_callback(sched, scheduler_copy_preflight, &p);
        ggml_backend_sched_set_graph_submit_callback(sched, scheduler_copy_first_submit, &p);
        ggml_backend_sched_set_submission_finished_callback(sched, scheduler_copy_finished, &p);
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            check_completion(false, "CPU copy graph allocation");
            break;
        }
        float xs[16], ys[16];
        for (int i = 0; i < 16; ++i) {
            xs[i] = 2;
            ys[i] = 3;
        }
        ggml_backend_tensor_set(x, xs, 0, sizeof(xs));
        ggml_backend_tensor_set(y, ys, 0, sizeof(ys));
        scheduler_cpu_calls = 0;
        p.refusal           = GGML_STATUS_ABORTED;
        auto status         = ggml_backend_sched_graph_compute_async(sched, graph);
        check_completion(status == GGML_STATUS_ABORTED && p.valid && p.preflights == 1 && p.finishes == 1 &&
                             p.result == status && p.successful_calls == 0 && p.first_submissions == 1 &&
                             scheduler_cpu_calls == 0,
                         "CPU copy preflight refuses before backend compute");
        if (!p.valid || !p.copies[1]) {
            check_completion(false, "CPU copy manifest identifies destination");
            break;
        }
        const float sentinel[16] = { 42 };
        ggml_backend_tensor_set(c->src[0], sentinel, 0, sizeof(sentinel));
        status          = ggml_backend_sched_graph_compute_async(sched, graph);
        float still[16] = {};
        ggml_backend_tensor_get(c->src[0], still, 0, sizeof(still));
        check_completion(status == GGML_STATUS_ABORTED && p.valid && p.preflights == 2 && p.finishes == 2 &&
                             p.successful_calls == 0 && still[0] == 42 && p.first_submissions == 2 &&
                             scheduler_cpu_calls == 0,
                         "CPU copy refusal leaves destination untouched");
        p.refusal = GGML_STATUS_SUCCESS;
        status    = ggml_backend_sched_graph_compute_async(sched, graph);
        ggml_backend_sched_synchronize(sched);
        float output[16] = {};
        ggml_backend_tensor_get(c, output, 0, sizeof(output));
        bool correct = true;
        for (float value : output) {
            correct &= value == 17;
        }
        check_completion(status == GGML_STATUS_SUCCESS && p.valid && correct && p.preflights == 3 && p.finishes == 3 &&
                             p.result == status && p.successful_calls == 3 && p.first_submissions == 3 &&
                             scheduler_cpu_calls == 3,
                         "CPU copy fallback computes copied values");
    }
    ggml_backend_sched_free(sched);
    backends[0]->iface.graph_compute = scheduler_cpu_compute;
    backends[1]->iface.graph_compute = scheduler_cpu_compute;
    ggml_backend_free(backends[0]);
    ggml_backend_free(backends[1]);
}

struct scheduler_stop_probe {
    int           eval_after = 0;
    int           finishes   = 0;
    int           calls      = -1;
    bool          valid      = true;
    ggml_tensor * a          = nullptr;
    ggml_tensor * b          = nullptr;
};

static enum ggml_status scheduler_stop_preflight(ggml_backend_sched_t sched, void * user_data) {
    auto &                            p     = *static_cast<scheduler_stop_probe *>(user_data);
    ggml_backend_sched_split_manifest split = {};
    p.valid &= ggml_backend_sched_preflight_n_splits(sched) == 1 &&
               ggml_backend_sched_preflight_split(sched, 0, &split) && split.n_nodes == 2 &&
               ggml_backend_sched_preflight_node(sched, 0, 0) == p.a &&
               ggml_backend_sched_preflight_node(sched, 0, 1) == p.b;
    return GGML_STATUS_SUCCESS;
}

static bool scheduler_stop_eval(ggml_tensor *, bool ask, void * user_data) {
    if (!ask) {
        ++static_cast<scheduler_stop_probe *>(user_data)->eval_after;
        return false;
    }
    return true;
}

static void scheduler_stop_finished(ggml_backend_sched_t sched, enum ggml_status status, int calls, void * user_data) {
    auto & p = *static_cast<scheduler_stop_probe *>(user_data);
    ++p.finishes;
    p.calls = calls;
    p.valid &= status == GGML_STATUS_SUCCESS && ggml_backend_sched_preflight_n_splits(sched) == -1;
}

static void test_scheduler_eval_stop() {
    test_ctx      tc;
    ggml_tensor * x     = leaf(tc.ctx, "stop-x");
    ggml_tensor * y     = leaf(tc.ctx, "stop-y");
    ggml_tensor * a     = named(ggml_add(tc.ctx, x, y), "stop-a");
    ggml_tensor * b     = named(ggml_mul(tc.ctx, a, y), "stop-b");
    ggml_cgraph * graph = ggml_new_graph(tc.ctx);
    ggml_build_forward_expand(graph, b);
    ggml_backend_t       backend    = ggml_backend_cpu_init();
    ggml_backend_t       backends[] = { backend };
    ggml_backend_sched_t sched      = ggml_backend_sched_new(backends, nullptr, 1, 32, false, false);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        check_completion(false, "CPU eval-stop graph allocation");
    } else {
        const float xs[16] = { 2 };
        const float ys[16] = { 3 };
        ggml_backend_tensor_set(x, xs, 0, sizeof(xs));
        ggml_backend_tensor_set(y, ys, 0, sizeof(ys));
        scheduler_stop_probe p;
        p.a = a;
        p.b = b;
        ggml_backend_sched_set_preflight_callback(sched, scheduler_stop_preflight, &p);
        ggml_backend_sched_set_eval_callback(sched, scheduler_stop_eval, &p);
        ggml_backend_sched_set_submission_finished_callback(sched, scheduler_stop_finished, &p);
        const auto status = ggml_backend_sched_graph_compute_async(sched, graph);
        float      value  = 0;
        ggml_backend_tensor_get(a, &value, 0, sizeof(value));
        check_completion(status == GGML_STATUS_SUCCESS && p.valid && p.finishes == 1 && p.calls == 1 &&
                             p.eval_after == 1 && value == 5,
                         "CPU callback_eval stop limits subdivision to first node");
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
}

static void test_paged_projection() {
    test_ctx                     tc;
    ggml_cgraph *                gf      = build_hoist_across(tc.ctx, "l_out-0");
    ggml_tensor *                cuts[]  = { ggml_graph_node(gf, 2) };
    ggml_tensor *                waits[] = { ggml_graph_node(gf, 1) };
    ggml_metal_projection_config config  = {};
    config.n_cuts                        = 1;
    config.cut_nodes                     = cuts;
    config.n_waits                       = 1;
    config.wait_nodes                    = waits;
    ggml_metal_projection projection     = {};
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_SUCCESS,
                     "project uncut predecessor/wait/cut/epilogue");
    ggml_metal_segment * warm_segments = projection.segments;
    const int            warm_capacity = projection.capacity_segments;
    const ggml_tensor ** warm_scratch  = projection.node_scratch;
    check_completion(warm_segments != nullptr && warm_capacity >= gf->n_nodes, "projection owns reusable capacity");
    if (projection.n_segments == 3) {
        check_completion(projection.segments[0].start == 0 && projection.segments[0].end == 1 &&
                             projection.segments[1].start == 1 && projection.segments[1].end == 3 &&
                             projection.segments[1].wait_at_start && projection.segments[2].start == 3 &&
                             projection.segments[2].end == 4,
                         "wait splits before cut and epilogue survives");
    } else {
        check_completion(false, "segment count with wait and cut");
    }
    // Reuse without freeing the output: its storage must remain owned and stable.

    ggml_tensor * sparse_cuts[] = { ggml_graph_node(gf, 0), ggml_graph_node(gf, 2) };
    config.n_cuts               = 2;
    config.cut_nodes            = sparse_cuts;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_SUCCESS &&
                         projection.n_segments == 3 && projection.segments[0].end == 1 &&
                         projection.segments[1].end == 3 && projection.segments[2].end == 4,
                     "sparse cuts preserve split-final and graph-end spans");
    check_completion(projection.segments == warm_segments && projection.node_scratch == warm_scratch &&
                         projection.capacity_segments == warm_capacity && projection.capacity_nodes >= gf->n_nodes,
                     "warm projection keeps both allocations");
    // The same output is reused across successive graph configurations.
    config.n_cuts    = 1;
    config.cut_nodes = cuts;

    config.window_active        = true;
    config.first_node           = ggml_graph_node(gf, 0);
    config.last_node            = ggml_graph_node(gf, 2);
    ggml_tensor * out_of_band[] = { ggml_graph_node(gf, 3) };
    config.n_out_of_band        = 1;
    config.out_of_band          = out_of_band;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_SUCCESS,
                     "skipped epilogue out-of-band is safe");
    if (projection.n_segments == 3) {
        check_completion(projection.segments[0].skipped && !projection.segments[1].skipped &&
                             projection.segments[1].window_first && projection.segments[1].window_last &&
                             projection.segments[1].ingress_here && projection.segments[1].egress_here &&
                             projection.segments[2].skipped,
                         "window skips endpoints and epilogue");
    } else {
        check_completion(false, "window segment count");
    }

    out_of_band[0] = ggml_graph_node(gf, 1);
    check_completion(
        ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED && projection.n_segments == 0,
        "selected out-of-band refuses before encode");
    config.n_out_of_band = 0;
    config.last_node     = ggml_graph_node(gf, 0);
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED,
                     "reversed/equal window refuses");
    config.last_node             = ggml_graph_node(gf, 2);
    ggml_tensor * invalid_cuts[] = { ggml_graph_node(gf, 2), nullptr };
    config.cut_nodes             = invalid_cuts;
    config.n_cuts                = 2;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED, "null cut refuses");
    config.n_cuts    = 1;
    config.cut_nodes = nullptr;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED,
                     "missing cut array refuses");
    config.cut_nodes = cuts;
    config.n_waits   = -1;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED,
                     "negative wait count refuses");
    config.n_waits = 1;
    check_completion(ggml_metal_project_segments(nullptr, &config, &projection) == GGML_STATUS_FAILED,
                     "missing graph refuses");
    check_completion(projection.n_segments == 0 && projection.segments == warm_segments,
                     "failure clears logical result while retaining storage");
    check_completion(ggml_metal_project_segments(gf, &config, nullptr) == GGML_STATUS_FAILED, "missing output refuses");

    config.window_active   = false;
    config.n_cuts          = 1;
    ggml_tensor * absent[] = { leaf(tc.ctx, "absent") };
    config.cut_nodes       = absent;
    config.n_waits         = 0;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_SUCCESS &&
                         projection.n_segments == 1 && projection.segments[0].start == 0 &&
                         projection.segments[0].end == 4 && !projection.segments[0].skipped,
                     "cut absent from split leaves full uncut span");

    ggml_cgraph split_view = ggml_graph_view(gf, 1, 4);
    config.cut_nodes       = cuts;
    config.n_waits         = 1;
    check_completion(ggml_metal_project_segments(&split_view, &config, &projection) == GGML_STATUS_SUCCESS &&
                         projection.n_segments == 2 && projection.segments[0].start == 0 &&
                         projection.segments[0].end == 2 && projection.segments[1].end == 3,
                     "real scheduler graph view with size zero projects");

    const int saved_count = gf->n_nodes;
    gf->n_nodes           = gf->size + 1;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED,
                     "out-of-range graph refuses");
    gf->n_nodes = 0;
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED,
                     "empty split refuses");
    gf->n_nodes              = saved_count;
    ggml_tensor * saved_node = gf->nodes[1];
    gf->nodes[1]             = gf->nodes[0];
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_FAILED,
                     "duplicate node pointer refuses");
    gf->nodes[1] = saved_node;
    check_completion(
        ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_SUCCESS && projection.n_segments == 3,
        "reuse after duplicate-node failure");
    ggml_metal_projection_free(&projection);
    check_completion(!projection.segments && !projection.node_scratch && !projection.n_segments &&
                         !projection.capacity_segments && !projection.capacity_nodes,
                     "projection free resets workspace");
    check_completion(ggml_metal_project_segments(gf, &config, &projection) == GGML_STATUS_SUCCESS &&
                         projection.n_segments == 3 && projection.capacity_segments >= gf->n_nodes,
                     "freed workspace can be initialized again");
    ggml_metal_projection_free(&projection);
}

static void test_projection_boundaries() {
    test_ctx                     tc;
    ggml_cgraph *                gf      = build_hoist_across(tc.ctx, "l_out-0");
    ggml_tensor *                n[4]    = { ggml_graph_node(gf, 0), ggml_graph_node(gf, 1), ggml_graph_node(gf, 2),
                                             ggml_graph_node(gf, 3) };
    ggml_tensor *                cuts[]  = { n[0], n[1], n[2] };
    ggml_tensor *                waits[] = { n[0], n[0], n[2] };
    ggml_metal_projection_config cfg     = {};
    ggml_metal_projection        p       = {};
    auto s = [](int start, int end, bool skipped = false, bool wait = false, bool first = false, bool last = false,
                bool ingress = false, bool egress = false) {
        return ggml_metal_segment{ start, end, skipped, wait, first, last, ingress, egress };
    };
    auto verify = [&](const char * label, std::initializer_list<ggml_metal_segment> expected) {
        bool ok =
            ggml_metal_project_segments(gf, &cfg, &p) == GGML_STATUS_SUCCESS && p.n_segments == (int) expected.size();
        if (ok) {
            int i = 0;
            for (const auto & e : expected) {
                const auto & a = p.segments[i++];
                ok &= a.start == e.start && a.end == e.end && a.skipped == e.skipped &&
                      a.wait_at_start == e.wait_at_start && a.window_first == e.window_first &&
                      a.window_last == e.window_last && a.ingress_here == e.ingress_here &&
                      a.egress_here == e.egress_here;
            }
        }
        check_completion(ok, label);
    };
    cfg.n_waits    = 2;
    cfg.wait_nodes = waits;
    verify("wait at zero and multiple waits at one head", { s(0, 4, false, true) });
    cfg.n_waits   = 3;
    cfg.n_cuts    = 3;
    cfg.cut_nodes = cuts;
    verify("consecutive cuts and coincident post-cut wait",
           { s(0, 1, false, true), s(1, 2), s(2, 3, false, true), s(3, 4) });
    cfg.n_cuts        = 1;
    cfg.cut_nodes     = &cuts[1];
    cfg.n_waits       = 0;
    cfg.window_active = true;
    cfg.first_node    = n[1];
    cfg.last_node     = nullptr;
    verify("lower-only open upper edge", { s(0, 2, true), s(2, 4, false, false, true, true, true) });
    cfg.first_node = nullptr;
    cfg.last_node  = n[1];
    verify("upper-only open lower edge", { s(0, 2, false, false, true, true, false, true), s(2, 4, true) });
    cfg.first_node = n[3];
    cfg.last_node  = nullptr;
    verify("wholly skipped split", { s(0, 2, true), s(2, 4, true) });
    cfg.first_node = n[0];
    cfg.last_node  = n[1];
    verify("boundary at open segment edge", { s(0, 2, false, false, true, true, true, true), s(2, 4, true) });
    ggml_tensor * absent      = leaf(tc.ctx, "missing-endpoint");
    ggml_tensor * forbidden[] = { n[3] };
    cfg.first_node            = absent;
    cfg.last_node             = absent;
    cfg.n_out_of_band         = 1;
    cfg.out_of_band           = forbidden;
    check_completion(ggml_metal_project_segments(gf, &cfg, &p) == GGML_STATUS_FAILED && p.n_segments == 0,
                     "endpoint-free split still guards out-of-band node");
    cfg.n_out_of_band = 0;
    verify("endpoint-free split encodes full graph", { s(0, 2), s(2, 4) });
    ggml_tensor * extended[6] = { n[0], n[1], n[2], n[3], leaf(tc.ctx, "extra-0"), leaf(tc.ctx, "extra-1") };
    ggml_cgraph   larger      = *gf;
    larger.nodes              = extended;
    larger.n_nodes            = 6;
    larger.size               = 0;
    check_completion(ggml_metal_project_segments(&larger, &cfg, &p) == GGML_STATUS_SUCCESS && p.capacity_nodes >= 6 &&
                         p.capacity_segments >= 6 && p.n_segments == 2,
                     "workspace grows for larger valid view");
    ggml_metal_segment * grown_segments = p.segments;
    const ggml_tensor ** grown_scratch  = p.node_scratch;
    check_completion(ggml_metal_project_segments(gf, &cfg, &p) == GGML_STATUS_SUCCESS && p.segments == grown_segments &&
                         p.node_scratch == grown_scratch && p.capacity_segments >= 6 && p.capacity_nodes >= 6,
                     "smaller split after growth retains both buffers");
    ggml_metal_projection_free(&p);
}

struct receipt_record {
    uint64_t                   generation, physical_id;
    uint32_t                   split, first, end;
    ggml_metal_terminal_status status;
};

struct receipt_probe {
    receipt_record records[16]    = {};
    uint64_t       quiesced[8]    = {};
    size_t         record_count   = 0;
    size_t         quiesced_count = 0;
    bool           overflow       = false;
};

static void receipt_terminal(void *                     cookie,
                             uint64_t                   generation,
                             uint64_t                   id,
                             uint32_t                   split,
                             uint32_t                   first,
                             uint32_t                   end,
                             ggml_metal_terminal_status status) {
    auto * probe = static_cast<receipt_probe *>(cookie);
    if (probe->record_count < std::size(probe->records)) {
        probe->records[probe->record_count++] = { generation, id, split, first, end, status };
    } else {
        probe->overflow = true;
    }
}

static void receipt_quiesced(void * cookie, uint64_t generation) {
    auto * probe = static_cast<receipt_probe *>(cookie);
    if (probe->quiesced_count < std::size(probe->quiesced)) {
        probe->quiesced[probe->quiesced_count++] = generation;
    } else {
        probe->overflow = true;
    }
}

static void test_shared_producer_receipts() {
    for (bool epilogue : { false, true }) {
        test_ctx                     tc;
        ggml_cgraph *                full        = build_hoist_across(tc.ctx, "l_out-0");
        ggml_cgraph                  view        = ggml_graph_view(full, 0, epilogue ? 4 : 3);
        ggml_tensor *                final       = view.nodes[2];
        ggml_tensor *                cut_nodes[] = { final, view.nodes[view.n_nodes - 1] };
        ggml_metal_projection_config config      = {};
        config.n_cuts                            = 2;
        config.cut_nodes                         = cut_nodes;
        ggml_metal_projection projection         = {};
        check_completion(ggml_metal_project_segments(&view, &config, &projection) == GGML_STATUS_SUCCESS,
                         "shared producer projected from graph");
        ggml_metal_receipt_node expected[] = {
            { 0, 0, view.nodes[0],                      GGML_METAL_RECEIPT_SELECTED },
            { 0, 1, view.nodes[1],                      GGML_METAL_RECEIPT_SELECTED },
            { 0, 2, final,                              GGML_METAL_RECEIPT_SELECTED },
            { 0, 3, epilogue ? view.nodes[3] : nullptr, GGML_METAL_RECEIPT_SELECTED },
        };
        ggml_metal_receipt_binding_cut cuts[] = {
            { 0, 2,                  { final, &tc, 10 }                        },
            { 0, epilogue ? 3u : 2u, { view.nodes[view.n_nodes - 1], &tc, 20 } },
        };
        completion_probe failure;
        receipt_probe    receipts;
        auto *   tracker = ggml_metal_completion_new(completion_publish, completion_latch, nullptr, nullptr, &failure);
        uint64_t generation = 0, id = 0;
        check_completion(
            ggml_metal_receipts_begin(tracker, expected, (size_t) view.n_nodes, cuts, 2, receipt_terminal,
                                      receipt_quiesced, &receipts, &generation) == GGML_STATUS_SUCCESS,
            epilogue ? "epilogue cut binds separate final producer" : "no-epilogue cuts bind shared final producer");
        uint32_t split = UINT32_MAX;
        check_completion(ggml_metal_receipts_validate_view(tracker, &view, &projection, &split) && split == 0 &&
                             ggml_metal_completion_reserve_additional(tracker, 1) &&
                             ggml_metal_receipts_register(tracker, split, view.nodes, 0, view.n_nodes, &id),
                         "one physical span covers multiple ordinals and cuts");
        check_completion(ggml_metal_receipts_finish(tracker, generation, GGML_STATUS_SUCCESS) == GGML_STATUS_SUCCESS &&
                             ggml_metal_completion_complete(tracker, id, true) &&
                             !ggml_metal_completion_complete(tracker, id, true) && receipts.record_count == 1 &&
                             receipts.records[0].physical_id == id && receipts.records[0].first == 0 &&
                             receipts.records[0].end == (uint32_t) view.n_nodes && receipts.quiesced_count == 1 &&
                             failure.published_count == 0 && !failure.failed,
                         "one terminal receipt and no autonomous success for both graph shapes");
        ggml_metal_completion_free(tracker);
        ggml_metal_projection_free(&projection);
    }
}

static void test_receipt_core() {
    test_ctx                     tc;
    ggml_cgraph *                graph = build_hoist_across(tc.ctx, "l_out-0");
    ggml_tensor *                a = graph->nodes[0], *b = graph->nodes[1], *c = graph->nodes[2], *x = graph->nodes[3];
    ggml_tensor *                cut_nodes[] = { a, b, c };
    ggml_metal_projection_config config      = {};
    config.n_cuts                            = 3;
    config.cut_nodes                         = cut_nodes;
    ggml_metal_projection projection         = {};
    check_completion(ggml_metal_project_segments(graph, &config, &projection) == GGML_STATUS_SUCCESS,
                     "receipt test uses actual projection");
    ggml_metal_receipt_node expected[] = {
        { 0, 0, a, GGML_METAL_RECEIPT_SELECTED },
        { 0, 1, b, GGML_METAL_RECEIPT_SELECTED },
        { 0, 2, c, GGML_METAL_RECEIPT_SELECTED },
        { 0, 3, x, GGML_METAL_RECEIPT_SELECTED },
    };
    int                            event  = 0;
    ggml_metal_receipt_binding_cut cuts[] = {
        { 0, 0, { a, &event, 10 } },
        { 0, 2, { c, &event, 20 } }
    };
    completion_probe failure;
    receipt_probe    first, second;
    auto *           tracker = ggml_metal_completion_new(completion_publish, completion_latch, completion_retain_event,
                                                         completion_release_event, &failure);
    uint64_t         gen1 = 0, gen2 = 0, ida = 0, idb = 0, idc = 0, idx = 0;
    check_completion(ggml_metal_receipts_begin(tracker, expected, 4, cuts, 2, receipt_terminal, receipt_quiesced,
                                               &first, &gen1) == GGML_STATUS_SUCCESS,
                     "bind expected table and native cuts");
    uint32_t split = UINT32_MAX;
    check_completion(ggml_metal_receipts_validate_view(tracker, graph, &projection, &split) && split == 0,
                     "entire actual projection matches expected nodes");
    check_completion(ggml_metal_completion_reserve_additional(tracker, 4), "reserve physical slots");
    check_completion(ggml_metal_receipts_register(tracker, split, graph->nodes, 0, 1, &ida) &&
                         ggml_metal_receipts_register(tracker, split, graph->nodes, 1, 2, &idb) &&
                         ggml_metal_receipts_register(tracker, split, graph->nodes, 2, 3, &idc) &&
                         ggml_metal_receipts_register(tracker, split, graph->nodes, 3, 4, &idx),
                     "register every selected span");
    check_completion(!ggml_metal_completion_reserve_additional(tracker, SIZE_MAX) && !failure.failed,
                     "overflow reserve preserves registered physical work");
    check_completion(ggml_metal_receipts_finish(tracker, gen1, GGML_STATUS_SUCCESS) == GGML_STATUS_SUCCESS,
                     "finish closes exact coverage without GPU wait");
    std::promise<void> started;
    std::atomic<bool>  drained(false);
    std::thread        waiter([&] {
        started.set_value();
        ggml_metal_completion_wait(tracker);
        drained.store(true);
    });
    started.get_future().wait();
    ggml_metal_completion_complete(tracker, ida, true);
    ggml_metal_completion_complete(tracker, idc, true);
    check_completion(first.record_count == 2 && first.records[1].physical_id == idc && failure.published_count == 0,
                     "A/C receipts delivered without private prefix publication");
    check_completion(first.quiesced_count == 0, "generation remains alive while B and X are pending");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    check_completion(!drained.load(), "all-buffer drain waits for every physical receipt");
    ggml_metal_receipt_binding_cut next_cuts[] = {
        { 0, 0, { a, &event, 30 } },
        { 0, 2, { c, &event, 40 } }
    };
    check_completion(ggml_metal_receipts_begin(tracker, expected, 4, next_cuts, 2, receipt_terminal, receipt_quiesced,
                                               &second, &gen2) == GGML_STATUS_SUCCESS &&
                         gen2 > gen1,
                     "new generation begins before old callbacks return");
    ggml_metal_completion_complete(tracker, idx, true);
    ggml_metal_completion_complete(tracker, idb, true);
    waiter.join();
    check_completion(drained.load(), "drain returns only after terminal hooks quiesce");
    check_completion(first.record_count == 4 && first.quiesced_count == 1 && first.quiesced[0] == gen1 &&
                         second.record_count == 0 && !first.overflow && !second.overflow,
                     "old pending hooks keep old cookie until quiesced");
    check_completion(ggml_metal_receipts_finish(tracker, gen2, GGML_STATUS_ABORTED) == GGML_STATUS_ABORTED &&
                         failure.failed && !failure.woke_before_latch && !failure.overflow &&
                         failure.published_count == 1 && failure.published[0] == UINT64_MAX,
                     "failed closure latches before sentinel wake");
    check_completion(!ggml_metal_receipts_register(tracker, 0, graph->nodes, 0, 1, &ida),
                     "failure rejects new registrations");
    ggml_metal_completion_free(tracker);
    check_completion(g_completion_event_refs == 0, "old and new generation cut refs released");
    completion_probe refusal;
    receipt_probe    partial;
    tracker = ggml_metal_completion_new(completion_publish, completion_latch, completion_retain_event,
                                        completion_release_event, &refusal);
    ggml_metal_receipt_node ambiguous[] = {
        { 0, 0, a, GGML_METAL_RECEIPT_SELECTED },
        { 1, 0, a, GGML_METAL_RECEIPT_SELECTED },
    };
    check_completion(ggml_metal_receipts_begin(tracker, ambiguous, 2, cuts, 2, receipt_terminal, receipt_quiesced,
                                               &partial, &gen1) == GGML_STATUS_FAILED,
                     "ambiguous pointer across splits refused at begin");
    ggml_metal_receipt_binding_cut duplicate_logical[] = { cuts[0], cuts[0] };
    check_completion(ggml_metal_receipts_begin(tracker, expected, 4, duplicate_logical, 2, receipt_terminal,
                                               receipt_quiesced, &partial, &gen1) == GGML_STATUS_FAILED,
                     "duplicate logical event/value refused");
    ggml_metal_receipt_binding_cut wrong_coordinate[] = { cuts[0], cuts[1] };
    wrong_coordinate[1].native.node                   = a;
    check_completion(ggml_metal_receipts_begin(tracker, expected, 4, wrong_coordinate, 2, receipt_terminal,
                                               receipt_quiesced, &partial, &gen1) == GGML_STATUS_FAILED,
                     "mismatched cut coordinate refused");
    check_completion(ggml_metal_receipts_begin(tracker, expected, 4, cuts, 2, receipt_terminal, receipt_quiesced,
                                               &partial, &gen1) == GGML_STATUS_SUCCESS,
                     "invalid preflight leaves tracker reusable");
    ggml_cgraph           subview = ggml_graph_view(graph, 0, 2);
    ggml_metal_projection slice   = {};
    check_completion(ggml_metal_project_segments(&subview, &config, &slice) == GGML_STATUS_SUCCESS &&
                         ggml_metal_receipts_validate_view(tracker, &subview, &slice, &split),
                     "callback_eval contiguous subview refinement accepted");
    ggml_tensor * alien        = leaf(tc.ctx, "alien");
    ggml_tensor * corrupted[2] = { a, alien };
    ggml_cgraph   changed      = subview;
    changed.nodes              = corrupted;
    check_completion(!ggml_metal_receipts_validate_view(tracker, &changed, &slice, &split),
                     "alien actual node refused before enqueue");
    corrupted[1] = a;
    check_completion(!ggml_metal_receipts_validate_view(tracker, &changed, &slice, &split),
                     "duplicate actual pointer refused before enqueue");
    ggml_metal_projection wrong_window = {};
    config.window_active               = true;
    config.first_node                  = a;
    config.last_node                   = c;
    check_completion(ggml_metal_project_segments(&subview, &config, &wrong_window) == GGML_STATUS_SUCCESS &&
                         !ggml_metal_receipts_validate_view(tracker, &subview, &wrong_window, &split),
                     "changed selected/skipped window refused before enqueue");
    ggml_metal_projection_free(&wrong_window);
    config.window_active = false;
    check_completion(ggml_metal_completion_reserve_additional(tracker, 2) &&
                         ggml_metal_receipts_register(tracker, 0, graph->nodes, 0, 1, &ida) &&
                         ggml_metal_receipts_register(tracker, 0, graph->nodes, 1, 2, &idb),
                     "register early readers for incomplete submission");
    check_completion(!ggml_metal_receipts_register(tracker, 0, graph->nodes, 0, 1, &idx) &&
                         !ggml_metal_receipts_register(tracker, 0, corrupted, 0, 2, &idx),
                     "duplicate and alien spans refuse atomically");
    check_completion(ggml_metal_receipts_finish(tracker, gen1, GGML_STATUS_SUCCESS) == GGML_STATUS_FAILED &&
                         refusal.failed && !refusal.woke_before_latch && !refusal.overflow &&
                         refusal.published_count == 1 && refusal.published[0] == UINT64_MAX,
                     "early eval stop and missing future reader/cut fail at finish");
    check_completion(!ggml_metal_receipts_register(tracker, 0, graph->nodes, 2, 3, &idx),
                     "failure forbids new work but leaves previously registered callbacks alive");
    ggml_metal_completion_complete(tracker, idb, true);
    ggml_metal_completion_complete(tracker, ida, false);
    check_completion(partial.record_count == 2 && partial.records[0].status == GGML_METAL_TERMINAL_COMPLETED &&
                         partial.records[1].status == GGML_METAL_TERMINAL_FAILED && partial.quiesced_count == 1 &&
                         partial.quiesced[0] == gen1 && !partial.overflow,
                     "registered work drains and terminal hooks preserve failure status");
    ggml_metal_completion_free(tracker);
    ggml_metal_projection_free(&slice);
    check_completion(g_completion_event_refs == 0, "partial generation releases retained cut refs");
    completion_probe reuse;
    receipt_probe    warm;
    tracker              = ggml_metal_completion_new(completion_publish, completion_latch, completion_retain_event,
                                                     completion_release_event, &reuse);
    expected[3].flags    = GGML_METAL_RECEIPT_SKIPPED;
    config.window_active = true;
    config.first_node    = nullptr;
    config.last_node     = c;
    ggml_metal_projection windowed = {};
    check_completion(ggml_metal_project_segments(graph, &config, &windowed) == GGML_STATUS_SUCCESS,
                     "project selected nodes and skipped epilogue");
    for (uint64_t round = 0; round < 2; ++round) {
        ggml_metal_receipt_binding_cut warm_cuts[] = {
            { 0, 0, { a, &event, 50 + round * 20 } },
            { 0, 2, { c, &event, 60 + round * 20 } },
        };
        size_t       capacity_before = 0, capacity_after = 0;
        const void * storage_before =
            round == 1 ? ggml_metal_receipts_node_storage(tracker, &capacity_before) : nullptr;
        if (round == 1) {
            ggml_metal_receipt_node duplicate_expected[] = { expected[0], expected[1], expected[2], expected[3] };
            duplicate_expected[3].node                   = expected[0].node;
            uint64_t refused_generation                  = 0;
            check_completion(
                ggml_metal_receipts_begin(tracker, duplicate_expected, 4, warm_cuts, 2, receipt_terminal,
                                          receipt_quiesced, &warm, &refused_generation) == GGML_STATUS_FAILED &&
                    ggml_metal_receipts_node_storage(tracker, &capacity_after) == storage_before &&
                    capacity_after == capacity_before && capacity_before >= 4 && g_completion_event_refs == 2 &&
                    warm.quiesced_count == 1 && !reuse.failed,
                "duplicate expected pointer preserves warm slab, events and quiesced cookie");
        }
        uint64_t id = 0, generation = 0;
        check_completion(ggml_metal_receipts_begin(tracker, expected, 4, warm_cuts, 2, receipt_terminal,
                                                   receipt_quiesced, &warm, &generation) == GGML_STATUS_SUCCESS &&
                             ggml_metal_receipts_validate_view(tracker, graph, &windowed, &split),
                         "selected/skipped projection binds warm generation");
        if (round == 1) {
            check_completion(ggml_metal_receipts_node_storage(tracker, &capacity_after) == storage_before &&
                                 capacity_after == capacity_before,
                             "valid begin reuses warm slab after refused input");
        }
        check_completion(!ggml_metal_receipts_validate_view(tracker, graph, &windowed, &split),
                         "duplicate skipped node refuses overlapping actual view");
        check_completion(ggml_metal_completion_reserve_additional(tracker, 3) &&
                             ggml_metal_receipts_register(tracker, 0, graph->nodes, 0, 1, &id),
                         "warm receipt slot capacity succeeds without drain");
        ggml_metal_completion_complete(tracker, id, true);
        check_completion(ggml_metal_receipts_register(tracker, 0, graph->nodes, 1, 2, &id),
                         "second selected segment registered");
        ggml_metal_completion_complete(tracker, id, true);
        check_completion(
            ggml_metal_receipts_register(tracker, 0, graph->nodes, 2, 3, &id) &&
                ggml_metal_receipts_finish(tracker, generation, GGML_STATUS_SUCCESS) == GGML_STATUS_SUCCESS,
            "skipped node does not create a coverage hole at finish");
        ggml_metal_completion_complete(tracker, id, true);
        check_completion(warm.record_count == 3 * (round + 1) && warm.quiesced_count == round + 1 && !warm.overflow &&
                             reuse.published_count == 0 && g_completion_event_refs == 2,
                         "warm reuse keeps events until all terminal hooks and publishes no success");
    }
    ggml_metal_completion_free(tracker);
    ggml_metal_projection_free(&windowed);
    check_completion(g_completion_event_refs == 0, "warm reuse frees native event holds");
    ggml_metal_projection_free(&projection);
}

int main() {
    test_paged_projection();
    test_projection_boundaries();
    test_shared_producer_receipts();
    test_receipt_core();
    test_completion_growth();
    test_scheduler_copies();
    test_scheduler_manifest();
    test_scheduler_eval_stop();
    // The reorder only ever has something to do when the graph it is given is in the order
    // these builders lay it out, so check that first -- otherwise a change in how
    // ggml_build_forward_expand walks the graph would quietly make every case below vacuous.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "l_out-0");

        check_order("graph as built", gf, { "a", "b", "l_out-0", "c" });
    }

    // Positive control: with the barriers off, the reorder hoists the independent lookup
    // across the marker. This is the behaviour every non-boundary-scheduling caller gets,
    // and it must not change.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "l_out-0");

        ggml_graph_optimize(gf, false);

        check_order("barriers off: hoists across marker", gf, { "a", "c", "b", "l_out-0" });
    }

    // With the barriers on, the same node stays in its own segment.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "l_out-0");

        ggml_graph_optimize(gf, true);

        check_order("barriers on: stops at marker", gf, { "a", "b", "l_out-0", "c" });
    }

    // The barrier keys on the marker, not on the switch: a node that is not a marker is
    // still hoisted with the barriers on, so turning them on is not a blanket "no reorder".
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "ffn_out-0");

        ggml_graph_optimize(gf, true);

        check_order("barriers on: non-marker still hoists", gf, { "a", "c", "b", "ffn_out-0" });
    }

    // A marker that is itself the node the scan starts from.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_marker_anchor(tc.ctx);

        ggml_graph_optimize(gf, false);

        check_order("barriers off: hoists in front of marker", gf, { "a", "c", "l_out-0" });
    }

    {
        test_ctx     tc;
        ggml_cgraph * gf = build_marker_anchor(tc.ctx);

        ggml_graph_optimize(gf, true);

        check_order("barriers on: marker takes nothing along", gf, { "a", "l_out-0", "c" });
    }

    // Fusion packs "s" and the marker into one group. With the barriers off the group is
    // hoisted whole, which moves the marker itself -- the two staying adjacent is what says
    // they were really fused rather than reordered independently.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx);

        ggml_graph_optimize(gf, false);

        check_order("barriers off: fused group hoists", gf, { "a", "s", "l_out-0", "b" });
    }

    // ... and with them on the group holds its place, because a group is a barrier when any
    // of its members is a marker, not only when its head is.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx);

        ggml_graph_optimize(gf, true);

        check_order("barriers on: fused group holds", gf, { "a", "b", "s", "l_out-0" });
    }

    // The name test itself.
    {
        test_ctx tc;

        int n_bad = 0;

        const char * markers[] = { "l_out-0", "l_out-63" };
        for (size_t i = 0; i < sizeof(markers)/sizeof(markers[0]); i++) {
            if (!ggml_graph_node_is_boundary_marker(leaf(tc.ctx, markers[i]))) {
                printf("FAIL marker name: '%s' not recognised\n", markers[i]);
                n_bad++;
            }
        }

        const char * not_markers[] = { "l_out-", "l_out-0x", "l_out", "ffn_out-0", "" };
        for (size_t i = 0; i < sizeof(not_markers)/sizeof(not_markers[0]); i++) {
            if (ggml_graph_node_is_boundary_marker(leaf(tc.ctx, not_markers[i]))) {
                printf("FAIL marker name: '%s' recognised as a marker\n", not_markers[i]);
                n_bad++;
            }
        }

        if (n_bad == 0) {
            printf("PASS %-40s\n", "marker name matching");
        }

        g_n_fail += n_bad;
    }

    if (g_n_fail > 0) {
        printf("\n%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("\nall checks passed\n");

    return 0;
}
