// the scheduler's split-compute callback: a caller computes each graph the scheduler would hand a backend, in place of
// ggml_backend_graph_compute_async, told the split and the position of the graph in it
//
// The graph is a chain of layers, h_{k+1} = h_k * 2 + x. With a GPU, layer k is pinned to the GPU or the CPU in turn,
// so the scheduler cuts the graph into several splits; without one it runs as one CPU split (the scheduler moves a node
// to the first backend with the same buffer type, so two CPU backends make one split too). Where the scheduler cuts is
// its own business: the test takes the number of splits and each node's backend from the scheduler, and the nodes of
// each split from the whole-split calls, which must cover the graph once, in order. Every value is a small integer, so
// the results compare exactly.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static constexpr int N_X      = 16;
static constexpr int N_LAYERS = 3;

static int n_failed = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); n_failed++; } } while (0)

// one call of the split-compute callback
struct split_call {
    int                      split_id;
    int                      first;
    ggml_backend_t           backend;
    std::vector<std::string> nodes;
    bool                     on_backend; // the scheduler placed every node of the graph on `backend`
};

struct split_recorder {
    std::vector<split_call> calls;
    int                     fail_at = -1; // the split whose first call fails without computing, -1 for none
};

static ggml_status record_split(ggml_backend_sched_t sched, int split_id, int first, ggml_backend_t backend, ggml_cgraph * graph, void * user_data) {
    auto * rec = static_cast<split_recorder *>(user_data);
    split_call call = { split_id, first, backend, {}, true };
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        call.nodes.push_back(node->name);
        call.on_backend = call.on_backend && ggml_backend_sched_get_tensor_backend(sched, node) == backend;
    }
    rec->calls.push_back(std::move(call));
    if (split_id == rec->fail_at) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_graph_compute_async(backend, graph);
}

struct submit_recorder {
    std::vector<ggml_status> statuses;
};

static void record_submit(ggml_backend_sched_t /*sched*/, ggml_status status, void * user_data) {
    static_cast<submit_recorder *>(user_data)->statuses.push_back(status);
}

static constexpr char ASKED[] = "scaled-1";

// asks for the scale node of the middle layer, so that the eval callback cuts the split holding it into views
static bool ask_for_middle_scale(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (!ask) {
        return true;
    }
    return strcmp(t->name, ASKED) == 0;
}

struct chain {
    ggml_context * ctx   = nullptr;
    ggml_cgraph  * graph = nullptr;
    ggml_tensor  * out   = nullptr;
    std::vector<ggml_tensor *> scaled; // the first node of each layer
    std::vector<ggml_tensor *> summed; // the second
};

static chain build_chain(ggml_tensor * x) {
    ggml_init_params params = {
        /*.mem_size   =*/ 2*N_LAYERS*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    chain c;
    c.ctx = ggml_init(params);
    c.graph = ggml_new_graph_custom(c.ctx, 16, false);

    ggml_tensor * h = x;
    for (int k = 0; k < N_LAYERS; k++) {
        ggml_tensor * s = ggml_scale(c.ctx, h, 2.0f);
        h = ggml_add(c.ctx, s, x);
        ggml_format_name(s, "scaled-%d", k);
        ggml_format_name(h, "summed-%d", k);
        c.scaled.push_back(s);
        c.summed.push_back(h);
    }
    ggml_set_output(h);
    ggml_build_forward_expand(c.graph, h);
    c.out = h;
    return c;
}

static std::vector<std::string> node_names(ggml_cgraph * graph) {
    std::vector<std::string> names;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        names.push_back(ggml_graph_node(graph, i)->name);
    }
    return names;
}

static void print_splits(const std::vector<split_call> & splits) {
    printf("scheduler: %zu split(s)\n", splits.size());
    for (const split_call & sp : splits) {
        printf("  split %d on %s:", sp.split_id, ggml_backend_name(sp.backend));
        for (const std::string & n : sp.nodes) {
            printf(" %s", n.c_str());
        }
        printf("\n");
    }
}

int main(void) {
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (gpu == nullptr) {
        gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    std::vector<ggml_backend_t> backends;
    if (gpu != nullptr) {
        backends.push_back(gpu);
    }
    backends.push_back(cpu);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 64, false, false);

    ggml_init_params wparams = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    ggml_tensor * x = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, N_X);
    ggml_set_name(x, "x");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, cpu);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> xs(N_X);
    for (int i = 0; i < N_X; i++) {
        xs[i] = (float) (i - N_X/2);
    }
    ggml_backend_tensor_set(x, xs.data(), 0, sizeof(float)*N_X);

    std::vector<float> expected(xs);
    for (int k = 0; k < N_LAYERS; k++) {
        for (int i = 0; i < N_X; i++) {
            expected[i] = expected[i]*2.0f + xs[i];
        }
    }

    // builds, places and computes the chain; returns the compute's status, the chain and the scheduler's split count
    struct run {
        ggml_status status;
        chain       c;
        int         n_splits;
    };
    auto compute = [&](std::vector<float> & out) {
        ggml_backend_sched_reset(sched);
        chain c = build_chain(x);
        for (int k = 0; k < N_LAYERS; k++) {
            ggml_backend_t b = backends[k % backends.size()];
            ggml_backend_sched_set_tensor_backend(sched, c.scaled[k], b);
            ggml_backend_sched_set_tensor_backend(sched, c.summed[k], b);
        }
        CHECK(ggml_backend_sched_alloc_graph(sched, c.graph));
        const int n = ggml_backend_sched_get_n_splits(sched);
        const ggml_status status = ggml_backend_sched_graph_compute(sched, c.graph);
        out.resize(N_X);
        ggml_backend_tensor_get(c.out, out.data(), 0, sizeof(float)*N_X);
        return run{ status, c, n };
    };

    // with no callback the backends compute the splits
    int n_splits = 0;
    {
        std::vector<float> out;
        run r = compute(out);
        n_splits = r.n_splits;
        CHECK(r.status == GGML_STATUS_SUCCESS);
        CHECK(out == expected);
        if (gpu != nullptr) {
            CHECK(n_splits > 1);
        }
        ggml_free(r.c.ctx);
    }

    split_recorder rec;
    ggml_backend_sched_set_split_compute_callback(sched, record_split, &rec);

    // the callback sees every split once, in order, as a whole, on its backend, and the splits cover the graph's
    // nodes once, in order; delegating computes as without it
    std::vector<split_call> splits;
    {
        std::vector<float> out;
        run r = compute(out);
        splits = rec.calls;
        print_splits(splits);
        CHECK(r.status == GGML_STATUS_SUCCESS);
        CHECK(out == expected);
        CHECK(r.n_splits == n_splits);
        CHECK((int) splits.size() == n_splits);
        std::vector<std::string> covered;
        for (size_t k = 0; k < splits.size(); k++) {
            const split_call & call = splits[k];
            CHECK(call.split_id == (int) k);
            CHECK(call.first == 0);
            CHECK(call.on_backend);
            covered.insert(covered.end(), call.nodes.begin(), call.nodes.end());
        }
        CHECK(covered == node_names(r.c.graph));
        ggml_free(r.c.ctx);
    }

    // with an eval callback the callback sees each view, with the index of its first node in the split: the split
    // holding the node asked for is viewed up to and including it, then from the next node on
    {
        rec.calls.clear();
        ggml_backend_sched_set_eval_callback(sched, ask_for_middle_scale, nullptr);
        std::vector<float> out;
        run r = compute(out);
        CHECK(r.status == GGML_STATUS_SUCCESS);
        CHECK(out == expected);

        std::vector<split_call> views;
        for (const split_call & sp : splits) {
            const std::vector<std::string> & nodes = sp.nodes;
            int cut = (int) nodes.size();
            for (int i = 0; i + 1 < (int) nodes.size(); i++) {
                if (nodes[i] == ASKED) {
                    cut = i + 1;
                }
            }
            views.push_back({ sp.split_id, 0, sp.backend, { nodes.begin(), nodes.begin() + cut }, true });
            if (cut < (int) nodes.size()) {
                views.push_back({ sp.split_id, cut, sp.backend, { nodes.begin() + cut, nodes.end() }, true });
            }
        }
        CHECK(rec.calls.size() == views.size());
        for (size_t i = 0; i < rec.calls.size() && i < views.size(); i++) {
            const split_call & call = rec.calls[i];
            CHECK(call.split_id == views[i].split_id);
            CHECK(call.first    == views[i].first);
            CHECK(call.backend  == views[i].backend);
            CHECK(call.nodes    == views[i].nodes);
        }
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        ggml_free(r.c.ctx);
    }

    // a failing status stops the compute: no later split is computed and, on the first split, the submit callback
    // sees the failure; after the first split it has already seen the submission succeed
    submit_recorder submits;
    ggml_backend_sched_set_graph_submit_callback(sched, record_submit, &submits);
    for (int fail_at = 0; fail_at < 2 && fail_at < n_splits; fail_at++) {
        rec.calls.clear();
        submits.statuses.clear();
        rec.fail_at = fail_at;
        std::vector<float> out;
        run r = compute(out);
        CHECK(r.status == GGML_STATUS_FAILED);
        CHECK((int) rec.calls.size() == fail_at + 1);
        CHECK(submits.statuses.size() == 1);
        if (!submits.statuses.empty()) {
            CHECK(submits.statuses[0] == (fail_at == 0 ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS));
        }
        ggml_free(r.c.ctx);
    }
    rec.fail_at = -1;
    ggml_backend_sched_set_graph_submit_callback(sched, nullptr, nullptr);

    // clearing the callback hands the splits back to the backends
    {
        rec.calls.clear();
        ggml_backend_sched_set_split_compute_callback(sched, nullptr, nullptr);
        std::vector<float> out;
        run r = compute(out);
        CHECK(r.status == GGML_STATUS_SUCCESS);
        CHECK(out == expected);
        CHECK(rec.calls.empty());
        ggml_free(r.c.ctx);
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    ggml_backend_free(cpu);
    if (gpu != nullptr) {
        ggml_backend_free(gpu);
    }

    if (n_failed > 0) {
        printf("%d check(s) failed\n", n_failed);
        return 1;
    }
    printf("all checks passed\n");
    return 0;
}
