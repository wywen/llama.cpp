// the scheduler's graph-rewrite callback: a caller substitutes an op in a graph with an equivalent chain of ops
// after the graph is built and before the scheduler splits and allocates it

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int   N_X     = 16;
static constexpr float SCALE   = 4.0f;
static constexpr char  PREFIX[] = "rewrite.";

// replaces each scale(x, a) with scale(scale(x, sqrt a), sqrt a) and rewires the readers of the replaced node
struct rewrite_state {
    ggml_context * ctx       = nullptr; // owns the tensors the rewrite adds
    bool           accept    = true;
    bool           dangling  = false;   // rewire a reader to a tensor that is not in the graph
    int            n_calls   = 0;
    int            n_rewrites = 0;
};

static bool rewrite_scale(ggml_backend_sched_t /*sched*/, ggml_cgraph * graph, void * user_data) {
    auto * state = static_cast<rewrite_state *>(user_data);
    state->n_calls++;
    if (!state->accept) {
        return false;
    }

    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            continue;
        }
        // a graph that is passed again holds the chain of an earlier rewrite
        if (node->op != GGML_OP_SCALE || strncmp(node->name, PREFIX, strlen(PREFIX)) == 0) {
            continue;
        }

        float s;
        memcpy(&s, node->op_params, sizeof(s));
        const float r = sqrtf(s);

        ggml_tensor * first  = ggml_scale(state->ctx, node->src[0], r);
        ggml_tensor * second = ggml_scale(state->ctx, first, r);
        ggml_format_name(first,  "%s%s.0", PREFIX, node->name);
        ggml_format_name(second, "%s%s.1", PREFIX, node->name);

        ggml_tensor * replacement = second;
        if (state->dangling) {
            replacement = ggml_scale(state->ctx, second, 1.0f);
        }
        ggml_build_forward_expand(graph, second);

        for (int j = 0; j < ggml_graph_n_nodes(graph); j++) {
            ggml_tensor * reader = ggml_graph_node(graph, j);
            if (reader == nullptr) {
                continue;
            }
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                if (reader->src[k] == node) {
                    reader->src[k] = replacement;
                }
            }
        }

        // drop the replaced node: nothing reads it any more
        ggml_graph_nodes(graph)[i] = nullptr;

        state->n_rewrites++;
    }
    return true;
}

// y = scale(x, a) + x, z = scale(x, a)^2
struct test_graph {
    ggml_context * ctx;
    ggml_cgraph  * graph;
    ggml_tensor  * y;
    ggml_tensor  * z;
};

static test_graph build_graph(ggml_tensor * x) {
    ggml_init_params params = {
        /*.mem_size   =*/ 9*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);

    ggml_tensor * s = ggml_scale(ctx, x, SCALE);
    ggml_set_name(s, "s");
    ggml_tensor * y = ggml_add(ctx, s, x);
    ggml_tensor * z = ggml_sqr(ctx, s);
    ggml_set_output(y);
    ggml_set_output(z);
    ggml_build_forward_expand(graph, y);
    ggml_build_forward_expand(graph, z);

    return { ctx, graph, y, z };
}

static int find_node(ggml_cgraph * graph, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        if (ggml_graph_node(graph, i) == t) {
            return i;
        }
    }
    return -1;
}

static int find_node_named(ggml_cgraph * graph, const char * name) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        if (strcmp(ggml_graph_node(graph, i)->name, name) == 0) {
            return i;
        }
    }
    return -1;
}

struct realloc_counts {
    int n_realloc = 0;
};

static void count_realloc(ggml_backend_sched_t, ggml_cgraph *, bool, bool, void * user_data) {
    static_cast<realloc_counts *>(user_data)->n_realloc++;
}

static int n_failed = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); n_failed++; } } while (0)

static void test_backend(ggml_backend_t backend) {
    printf("%s: %s\n", __func__, ggml_backend_name(backend));

    // the input sits in a weights buffer of the backend under test, so the ops that read it run there
    ggml_init_params wparams = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    ggml_tensor * x = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, N_X);
    ggml_set_name(x, "x");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> xs(N_X);
    for (int i = 0; i < N_X; i++) {
        xs[i] = 0.25f*(i - N_X/2);
    }
    ggml_backend_tensor_set(x, xs.data(), 0, sizeof(float)*N_X);

    std::vector<ggml_backend_t> backends = { backend };
    ggml_backend_t cpu = nullptr;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu = ggml_backend_cpu_init();
        backends.push_back(cpu);
    }
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 64, false, true);

    realloc_counts reallocs;
    ggml_backend_sched_set_realloc_callback(sched, count_realloc, &reallocs);

    auto compute = [&](test_graph & g, std::vector<float> & y, std::vector<float> & z) {
        CHECK(ggml_backend_sched_alloc_graph(sched, g.graph));
        CHECK(ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS);
        y.resize(N_X);
        z.resize(N_X);
        ggml_backend_tensor_get(g.y, y.data(), 0, sizeof(float)*N_X);
        ggml_backend_tensor_get(g.z, z.data(), 0, sizeof(float)*N_X);
    };

    // with no callback the graph is not changed
    std::vector<float> y_ref, z_ref;
    {
        test_graph g = build_graph(x);
        compute(g, y_ref, z_ref);
        CHECK(ggml_graph_n_nodes(g.graph) == 3);
        CHECK(find_node_named(g.graph, "s") == 0);
        for (int i = 0; i < N_X; i++) {
            const float s = xs[i]*SCALE;
            CHECK(y_ref[i] == s + xs[i]);
            CHECK(z_ref[i] == s*s);
        }
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }

    ggml_init_params rparams = { 64*ggml_tensor_overhead(), nullptr, true };
    rewrite_state state;
    state.ctx = ggml_init(rparams);
    ggml_backend_sched_set_graph_rewrite_callback(sched, rewrite_scale, &state);

    // the reserve sees the rewritten graph, so allocating a rewritten graph does not grow the compute buffer
    {
        test_graph g = build_graph(x);
        CHECK(ggml_backend_sched_reserve(sched, g.graph));
        CHECK(state.n_calls == 1);
        CHECK(state.n_rewrites == 1);
        ggml_free(g.ctx);
    }
    const size_t reserved = ggml_backend_sched_get_buffer_size(sched, backend);
    reallocs.n_realloc = 0;
    {
        test_graph g = build_graph(x);
        std::vector<float> y, z;
        compute(g, y, z);
        CHECK(state.n_calls == 2);
        CHECK(state.n_rewrites == 2);
        CHECK(reallocs.n_realloc == 0);
        CHECK(ggml_backend_sched_get_buffer_size(sched, backend) == reserved);

        // the chain replaced the scale node and sits before its readers
        CHECK(ggml_graph_n_nodes(g.graph) == 4);
        CHECK(find_node_named(g.graph, "s") == -1);
        const int i0 = find_node_named(g.graph, "rewrite.s.0");
        const int i1 = find_node_named(g.graph, "rewrite.s.1");
        CHECK(i0 == 0 && i1 == 1);
        CHECK(find_node(g.graph, g.y) > i1);
        CHECK(find_node(g.graph, g.z) > i1);
        CHECK(g.y->src[0] == ggml_graph_node(g.graph, i1));
        CHECK(g.z->src[0] == ggml_graph_node(g.graph, i1));
        CHECK(ggml_backend_sched_get_tensor_backend(sched, ggml_graph_node(g.graph, i1)) == backend);

        // the use counts that fusion reads count the rewired readers
        CHECK(ggml_node_get_use_count(g.graph, i0) == 1);
        CHECK(ggml_node_get_use_count(g.graph, i1) == 2);

        CHECK(y == y_ref);
        CHECK(z == z_ref);

        // allocating the same graph again passes the rewritten graph, which the callback leaves as it is
        ggml_backend_sched_reset(sched);
        compute(g, y, z);
        CHECK(state.n_calls == 3);
        CHECK(state.n_rewrites == 2);
        CHECK(ggml_graph_n_nodes(g.graph) == 4);
        CHECK(y == y_ref);
        CHECK(z == z_ref);

        // computing an allocated graph does not rewrite it again
        CHECK(ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS);
        CHECK(state.n_calls == 3);

        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }

    // the measuring entry points rewrite too
    {
        test_graph g = build_graph(x);
        std::vector<size_t> sizes(backends.size());
        CHECK(ggml_backend_sched_reserve_size(sched, g.graph, sizes.data()));
        CHECK(state.n_calls == 4);
        CHECK(ggml_graph_n_nodes(g.graph) == 4);
        ggml_free(g.ctx);

        g = build_graph(x);
        CHECK(ggml_backend_sched_split_graph(sched, g.graph));
        CHECK(state.n_calls == 5);
        CHECK(ggml_graph_n_nodes(g.graph) == 4);
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }

    // a refusal fails every entry point
    state.accept = false;
    {
        test_graph g = build_graph(x);
        std::vector<size_t> sizes(backends.size());
        CHECK(!ggml_backend_sched_alloc_graph(sched, g.graph));
        CHECK(ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_ALLOC_FAILED);
        CHECK(!ggml_backend_sched_reserve(sched, g.graph));
        CHECK(!ggml_backend_sched_reserve_size(sched, g.graph, sizes.data()));
        CHECK(!ggml_backend_sched_split_graph(sched, g.graph));
        CHECK(state.n_calls == 10);
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }
    state.accept = true;

    // a reader of a tensor that is not in the graph fails the allocation
    state.dangling = true;
    {
        test_graph g = build_graph(x);
        CHECK(!ggml_backend_sched_alloc_graph(sched, g.graph));
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }
    state.dangling = false;

    // a node that names itself as a source, as ggml_cast does, is not a cycle
    {
        test_graph g = build_graph(x);
        ggml_tensor * c = ggml_cast(g.ctx, g.y, GGML_TYPE_F32);
        CHECK(c->src[1] == c);
        ggml_set_output(c);
        ggml_build_forward_expand(g.graph, c);
        std::vector<float> y, z;
        compute(g, y, z);
        std::vector<float> cs(N_X);
        ggml_backend_tensor_get(c, cs.data(), 0, sizeof(float)*N_X);
        CHECK(y == y_ref);
        CHECK(cs == y_ref);
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }

    // a cleared callback leaves the graph as llama built it
    ggml_backend_sched_set_graph_rewrite_callback(sched, nullptr, nullptr);
    {
        const int n_calls = state.n_calls;
        test_graph g = build_graph(x);
        std::vector<float> y, z;
        compute(g, y, z);
        CHECK(state.n_calls == n_calls);
        CHECK(ggml_graph_n_nodes(g.graph) == 3);
        CHECK(y == y_ref);
        CHECK(z == z_ref);
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }

    ggml_free(state.ctx);
    ggml_backend_sched_free(sched);
    if (cpu) {
        ggml_backend_free(cpu);
    }
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    test_backend(cpu);
    ggml_backend_free(cpu);

    // Metal, when the machine has a device
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev) {
        ggml_backend_t metal = ggml_backend_dev_init(dev, nullptr);
        test_backend(metal);
        ggml_backend_free(metal);
    } else {
        printf("MTL0 not available, skipping\n");
    }

    printf("%s\n", n_failed == 0 ? "OK" : "FAILED");
    return n_failed == 0 ? 0 : 1;
}
