// the backend scheduler with a caller allocator: the caller places the nodes, names the copies and gives each tensor
// its memory in its own compute buffers; the graph computes as the graph allocator computes it, bit for bit

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

static constexpr int N_K = 64;
static constexpr int N_M = 48;
static constexpr int N_T = 8;

static int n_failed = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); n_failed++; } } while (0)

struct test_graph {
    ggml_context * ctx;
    ggml_cgraph  * graph;
    ggml_tensor  * a;   // input
    ggml_tensor  * h;   // runs on the CPU
    ggml_tensor  * out;
};

// out = sqr(softmax(cont(transpose(scale_inplace(w x a))))) + softmax(...), with the sqr on the CPU
static test_graph build_graph(ggml_tensor * w) {
    ggml_init_params params = { 32*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_K, N_T);
    ggml_set_name(a, "a");
    ggml_set_input(a);
    ggml_tensor * c = ggml_mul_mat(ctx, w, a);
    ggml_tensor * d = ggml_scale_inplace(ctx, c, 0.5f);
    ggml_tensor * f = ggml_cont(ctx, ggml_transpose(ctx, d));
    ggml_tensor * g = ggml_soft_max(ctx, f);
    ggml_tensor * h = ggml_sqr(ctx, g);
    ggml_set_name(h, "h");
    ggml_tensor * out = ggml_add(ctx, h, g);
    ggml_set_name(out, "out");
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    return { ctx, graph, a, h, out };
}

static std::vector<float> input_values() {
    std::vector<float> v(N_K*N_T);
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = 0.01f*(float) ((i*37) % 101) - 0.5f;
    }
    return v;
}

// how the test caller allocates
enum alloc_mode {
    ALLOC_PACKED,     // each tensor at its own offset, in two chunks
    ALLOC_SHARED,     // an op that may write in place takes its last-read source's bytes
    ALLOC_OUT_OF_BOUNDS,
    ALLOC_MISALIGNED,
    ALLOC_NO_CHUNK,
    ALLOC_LEAVE_ONE,  // one tensor is left without memory
};

struct caller {
    ggml_backend_t backend;   // runs every node but h
    ggml_backend_t cpu;
    ggml_context * ctx_copies;
    std::map<ggml_backend_buffer_type_t, std::vector<ggml_backend_buffer_t>> buffers;
    alloc_mode mode     = ALLOC_PACKED;
    bool name_copies    = true;
    int  n_copies       = 0;
    int  n_refused      = 0;
};

static ggml_backend_buffer_type_t storage_buft(ggml_backend_sched_t sched, ggml_tensor * t) {
    ggml_tensor * root = t->view_src ? t->view_src : t;
    if (root->buffer) {
        return ggml_backend_buffer_get_type(root->buffer);
    }
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, root);
    return b ? ggml_backend_sched_get_buffer_type(sched, b) : nullptr;
}

static bool place(ggml_backend_sched_t sched, ggml_cgraph * graph, void * user_data) {
    auto * c = static_cast<caller *>(user_data);
    for (int i = 0; i < graph->n_leafs; i++) {
        ggml_tensor * leaf = graph->leafs[i];
        if (leaf->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_backend_sched_set_tensor_backend(sched, leaf, c->cpu);
        }
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->view_src) {
            if (ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, node->view_src)) {
                ggml_backend_sched_set_tensor_backend(sched, node, b);
            }
            continue;
        }
        ggml_backend_sched_set_tensor_backend(sched, node, strcmp(node->name, "h") == 0 ? c->cpu : c->backend);
    }
    if (!c->name_copies) {
        return true;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->view_src) {
            continue;
        }
        ggml_backend_t reader = ggml_backend_sched_get_tensor_backend(sched, node);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            ggml_backend_buffer_type_t buft = storage_buft(sched, src);
            if (buft == nullptr || ggml_backend_supports_buft(reader, buft)) {
                continue;
            }
            ggml_tensor * copy = ggml_new_tensor(c->ctx_copies, src->type, GGML_MAX_DIMS, src->ne);
            memcpy(copy->nb, src->nb, sizeof(src->nb));
            ggml_format_name(copy, "copy#%s", src->name);
            ggml_backend_sched_set_tensor_copy(sched, src, reader, copy);
            c->n_copies++;
        }
    }
    return true;
}

struct reserve_counts {
    int n_calls        = 0;
    int n_reservations = 0;
};

static bool count_reserve(ggml_backend_sched_t, const ggml_backend_sched_buffer_reservation *, size_t n, void * user_data) {
    auto * counts = static_cast<reserve_counts *>(user_data);
    counts->n_calls++;
    counts->n_reservations += (int) n;
    return true;
}

static size_t align_to(size_t n, size_t a) {
    return (n + a - 1)/a*a;
}

static bool alloc(ggml_backend_sched_t sched, ggml_cgraph * graph, bool /*reserve*/, void * user_data) {
    auto * c = static_cast<caller *>(user_data);

    // the tensors that need memory, in the order the graph computes them, and the last node reading each
    std::vector<ggml_tensor *> tensors;
    std::map<ggml_tensor *, int> last;
    auto add = [&](ggml_tensor * t) {
        if (t->data == nullptr && t->view_src == nullptr && last.count(t) == 0) {
            tensors.push_back(t);
            last[t] = -1;
        }
    };
    for (int i = 0; i < graph->n_leafs; i++) {
        add(graph->leafs[i]);
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        add(node);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = node->src[j];
            if (src != nullptr) {
                ggml_tensor * root = src->view_src ? src->view_src : src;
                if (last.count(root)) {
                    last[root] = i;
                }
            }
        }
    }

    std::map<ggml_backend_buffer_type_t, size_t> next;
    std::map<ggml_tensor *, std::pair<int, size_t>> places;
    bool left_one = false;
    for (ggml_tensor * t : tensors) {
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, t);
        ggml_backend_buffer_type_t buft = ggml_backend_sched_get_buffer_type(sched, b);
        const std::vector<ggml_backend_buffer_t> & bufs = c->buffers.at(buft);
        const size_t align = ggml_backend_buft_get_alignment(buft);
        const size_t size  = ggml_backend_buft_get_alloc_size(buft, t);

        if (c->mode == ALLOC_LEAVE_ONE && !left_one) {
            left_one = true;
            continue;
        }
        if (c->mode == ALLOC_OUT_OF_BOUNDS) {
            const size_t end = ggml_backend_buffer_get_size(bufs[0]);
            CHECK(!ggml_backend_sched_alloc_tensor(sched, t, 0, align_to(end - size/2, align)));
            c->n_refused++;
            return false;
        }
        if (c->mode == ALLOC_MISALIGNED) {
            CHECK(!ggml_backend_sched_alloc_tensor(sched, t, 0, 1));
            c->n_refused++;
            return false;
        }
        if (c->mode == ALLOC_NO_CHUNK) {
            CHECK(!ggml_backend_sched_alloc_tensor(sched, t, (int) bufs.size(), 0));
            c->n_refused++;
            return false;
        }

        std::pair<int, size_t> at;
        bool shared = false;
        if (c->mode == ALLOC_SHARED && t->op != GGML_OP_NONE && ggml_op_can_inplace(t->op)) {
            int node = -1;
            for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
                if (ggml_graph_node(graph, i) == t) {
                    node = i;
                }
            }
            for (int j = 0; j < GGML_MAX_SRC && !shared; j++) {
                ggml_tensor * src = t->src[j];
                if (src && places.count(src) && src->view_src == nullptr && last[src] == node && ggml_are_same_layout(src, t) &&
                        storage_buft(sched, src) == buft) {
                    at = places[src];
                    shared = true;
                }
            }
        }
        if (!shared) {
            // alternate the chunks, from the end of each
            const int chunk = (int) (places.size() % bufs.size());
            size_t & off = next[ggml_backend_buffer_get_type(bufs[chunk])];
            at = { chunk, align_to(off + 3*align, align) };
            off = at.second + size;
        }
        CHECK(ggml_backend_sched_alloc_tensor(sched, t, at.first, at.second));
        // a second allocation of the tensor is refused
        CHECK(!ggml_backend_sched_alloc_tensor(sched, t, at.first, at.second));
        places[t] = at;
    }
    return true;
}

static void test_backend(ggml_backend_t backend) {
    printf("%s: %s\n", __func__, ggml_backend_name(backend));

    ggml_init_params wparams = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, N_K, N_M);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> ws(N_K*N_M);
    for (size_t i = 0; i < ws.size(); i++) {
        ws[i] = 0.003f*(float) ((i*53) % 97) - 0.1f;
    }
    ggml_backend_tensor_set(w, ws.data(), 0, sizeof(float)*ws.size());
    const std::vector<float> as = input_values();

    const bool on_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_t cpu = on_cpu ? backend : ggml_backend_cpu_init();
    std::vector<ggml_backend_t> backends = { backend };
    if (!on_cpu) {
        backends.push_back(cpu);
    }

    auto compute = [&](ggml_backend_sched_t sched, test_graph & g, std::vector<float> & out) {
        if (!ggml_backend_sched_alloc_graph(sched, g.graph)) {
            return false;
        }
        ggml_backend_tensor_set(g.a, as.data(), 0, sizeof(float)*as.size());
        CHECK(ggml_backend_sched_graph_compute(sched, g.graph) == GGML_STATUS_SUCCESS);
        out.resize(N_M*N_T);
        ggml_backend_tensor_get(g.out, out.data(), 0, sizeof(float)*out.size());
        return true;
    };

    // the graph allocator's result
    std::vector<float> ref;
    {
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 64, false, true);
        test_graph g = build_graph(w);
        ggml_backend_sched_set_tensor_backend(sched, g.h, cpu);
        CHECK(compute(sched, g, ref));
        ggml_free(g.ctx);
        ggml_backend_sched_free(sched);
    }

    caller c;
    c.backend = backend;
    c.cpu     = cpu;
    ggml_init_params cparams = { 64*ggml_tensor_overhead(), nullptr, true };
    c.ctx_copies = ggml_init(cparams);

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 64, false, true);
    ggml_backend_sched_caller_alloc calls = { place, alloc, &c };
    CHECK(ggml_backend_sched_set_caller_alloc(sched, &calls));
    reserve_counts reserves;
    ggml_backend_sched_set_reserve_callback(sched, count_reserve, &reserves);
    for (ggml_backend_t b : backends) {
        ggml_backend_buffer_type_t buft = ggml_backend_sched_get_buffer_type(sched, b);
        if (c.buffers.count(buft)) {
            continue;
        }
        std::vector<ggml_backend_buffer_t> & bufs = c.buffers[buft];
        bufs = { ggml_backend_buft_alloc_buffer(buft, 1 << 20), ggml_backend_buft_alloc_buffer(buft, 1 << 20) };
        CHECK(ggml_backend_sched_set_compute_buffers(sched, buft, bufs.data(), (int) bufs.size()));
    }
    // a buffer of another type is refused
    if (!on_cpu) {
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_sched_get_buffer_type(sched, cpu);
        CHECK(!ggml_backend_sched_set_compute_buffers(sched, ggml_backend_sched_get_buffer_type(sched, backend), c.buffers.at(cpu_buft).data(), 1));
    }

    for (alloc_mode mode : { ALLOC_PACKED, ALLOC_SHARED }) {
        c.mode = mode;
        c.n_copies = 0;
        std::vector<float> out;

        // a reserve allocates the graph through the caller and leaves the scheduler reset
        {
            test_graph g = build_graph(w);
            const int n_calls = reserves.n_calls;
            CHECK(ggml_backend_sched_reserve(sched, g.graph));
            CHECK(g.out->data != nullptr);
            // the reserve callback runs, offering no chunk
            CHECK(reserves.n_calls == n_calls + 1);
            CHECK(reserves.n_reservations == 0);
            ggml_free(g.ctx);
            ggml_free(c.ctx_copies);
            c.ctx_copies = ggml_init(cparams);
        }

        test_graph g = build_graph(w);
        CHECK(compute(sched, g, out));
        CHECK(out == ref);
        CHECK(ggml_backend_sched_get_tensor_backend(sched, g.h) == cpu);
        printf("  mode %d: %d copies named, %d splits\n", (int) mode, c.n_copies, ggml_backend_sched_get_n_splits(sched));
        CHECK(on_cpu ? c.n_copies == 0 : c.n_copies > 0);

        // the extents are what the caller placed, in the caller's buffers
        size_t total = 0;
        for (int b = 0; b < (int) backends.size(); b++) {
            for (int k = 0; k < GGML_SCHED_MAX_CALLER_BUFFERS; k++) {
                ggml_backend_buffer_t chunk = nullptr;
                const size_t extent = ggml_backend_sched_get_compute_extent(sched, b, k, &chunk);
                CHECK(extent == 0 || chunk != nullptr);
                CHECK(chunk == nullptr || extent <= ggml_backend_buffer_get_size(chunk));
                total += extent;
            }
        }
        CHECK(total > 0);
        CHECK(ggml_backend_sched_get_buffer_size(sched, backend) == 2u << 20);

        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
        ggml_free(c.ctx_copies);
        c.ctx_copies = ggml_init(cparams);
    }

    // refusals fail the allocation
    for (alloc_mode mode : { ALLOC_OUT_OF_BOUNDS, ALLOC_MISALIGNED, ALLOC_NO_CHUNK, ALLOC_LEAVE_ONE }) {
        c.mode = mode;
        test_graph g = build_graph(w);
        std::vector<float> out;
        CHECK(!compute(sched, g, out));
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }
    CHECK(c.n_refused == 3);

    // a split that reads a tensor its backend cannot address needs a named copy
    if (!on_cpu) {
        c.mode = ALLOC_PACKED;
        c.name_copies = false;
        test_graph g = build_graph(w);
        std::vector<float> out;
        CHECK(!compute(sched, g, out));
        CHECK(!ggml_backend_sched_split_graph(sched, g.graph));
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
        c.name_copies = true;
    }

    // without the caller allocator the graph allocator runs again
    ggml_backend_sched_set_reserve_callback(sched, nullptr, nullptr);
    CHECK(ggml_backend_sched_set_caller_alloc(sched, nullptr));
    {
        test_graph g = build_graph(w);
        ggml_backend_sched_set_tensor_backend(sched, g.h, cpu);
        std::vector<float> out;
        CHECK(compute(sched, g, out));
        CHECK(out == ref);
        ggml_backend_sched_reset(sched);
        ggml_free(g.ctx);
    }

    ggml_backend_sched_free(sched);
    for (auto & [buft, bufs] : c.buffers) {
        for (ggml_backend_buffer_t b : bufs) {
            ggml_backend_buffer_free(b);
        }
    }
    ggml_free(c.ctx_copies);
    if (!on_cpu) {
        ggml_backend_free(cpu);
    }
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
}

int main() {
    ggml_backend_load_all();

    // a scheduler with copies of its inputs refuses a caller allocator
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, 64, true, true);
        ggml_backend_sched_caller_alloc calls = { place, alloc, nullptr };
        CHECK(ggml_backend_sched_get_n_copies(sched) == 1 || !ggml_backend_sched_set_caller_alloc(sched, &calls));
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    test_backend(cpu);
    ggml_backend_free(cpu);

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
