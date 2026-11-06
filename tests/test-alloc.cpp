#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <memory>
#include <vector>

//
// dummy backend with configurable max_buffer_size, tracks allocations

uint8_t * const alloc_base = (uint8_t *) 16;

struct dummy_backend_context {
    size_t max_buffer_size = 64;
    size_t alignment       = 8;

    ggml_backend_buffer_i              buffer_interface;
    ggml_backend_device                device;
    ggml_backend                       backend;
    std::vector<ggml_backend_buffer_t> buffers;

    bool custom_alloc_buffer_n_called   = false;
    bool custom_get_alloc_size_n_called = false;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t /*buft*/) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer_n_custom(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    ctx->custom_alloc_buffer_n_called = true;

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 64);
    if (buffer == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < n_tensors; i++) {
        tensors[i]->buffer = buffer;
        tensors[i]->data   = (char *) ggml_backend_buffer_get_base(buffer);
    }
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alloc_size_n_custom(
        ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    GGML_UNUSED(tensors);
    GGML_UNUSED(n_tensors);
    ctx->custom_get_alloc_size_n_called = true;
    return 64;
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t /*buft*/) {
    return true;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t /*buft*/) {
    return alloc_base;
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/) {
    return GGML_STATUS_SUCCESS;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/, uint8_t, size_t, size_t) {}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t /*buft*/, ggml_tensor * /*x*/, const void *, size_t, size_t) {}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t /*buft*/, const ggml_tensor * /*x*/, void *, size_t, size_t) {}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t /*buft*/, uint8_t /*val*/) {}

// ggml_backend_device interface

static enum ggml_backend_dev_type dummy_backend_device_get_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static bool dummy_backend_device_supports_op(ggml_backend_dev_t /*dev*/, const ggml_tensor * /*x*/) {
    return true;
}

static bool dummy_backend_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return device->context == buft->context;
}

// ggml_backend interface

static const char * dummy_backend_get_name(ggml_backend_t /*backend*/) {
    return "dummy_backend";
}

// dummy_backend

struct dummy_backend {
    std::unique_ptr<dummy_backend_context> context;
    ggml_backend_buffer_type               buffer_type;
};

static dummy_backend dummy_backend_init(size_t max_buffer_size, size_t alignment = 8) {
    dummy_backend b{};
    b.context                  = std::make_unique<dummy_backend_context>();
    b.context->alignment       = alignment;
    b.context->max_buffer_size = max_buffer_size;

    b.context->buffer_interface.free_buffer   = dummy_backend_buffer_free_buffer;
    b.context->buffer_interface.get_base      = dummy_backend_buffer_get_base;
    b.context->buffer_interface.init_tensor   = dummy_backend_buffer_init_tensor;
    b.context->buffer_interface.memset_tensor = dummy_backend_buffer_memset_tensor;
    b.context->buffer_interface.set_tensor    = dummy_backend_buffer_set_tensor;
    b.context->buffer_interface.get_tensor    = dummy_backend_buffer_get_tensor;
    b.context->buffer_interface.clear         = dummy_backend_buffer_clear;

    b.context->device.context             = b.context.get();
    b.context->device.iface.get_type      = dummy_backend_device_get_type;
    b.context->device.iface.supports_op   = dummy_backend_device_supports_op;
    b.context->device.iface.supports_buft = dummy_backend_device_supports_buft;

    b.context->backend.context        = b.context.get();
    b.context->backend.device         = &b.context->device;
    b.context->backend.iface.get_name = dummy_backend_get_name;

    b.buffer_type.device              = &b.context->device;
    b.buffer_type.context             = b.context.get();
    b.buffer_type.iface.get_name      = dummy_backend_buffer_type_get_name;
    b.buffer_type.iface.alloc_buffer  = dummy_backend_buffer_type_alloc_buffer;
    b.buffer_type.iface.get_alignment = dummy_backend_buffer_type_get_alignment;
    b.buffer_type.iface.get_max_size  = dummy_backend_buffer_type_get_max_size;
    b.buffer_type.iface.is_host       = dummy_backend_buffer_type_is_host;
    return b;
}

//
// test utilities

struct test_context_with_graph {
    ggml_context *   ctx;
    ggml_cgraph *    graph;
    ggml_context_ptr ctx_ptr;
};

static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}

static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}

static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}

static void assign_names(ggml_context * ctx, const char * prefix = "x") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, "%s%d", prefix, i++);
    }
}

static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "leaf not found: %s\n", tensor_name);
    return -1;
}

static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "node not found: %s", tensor_name);
    return -1;
}

static ggml_gallocr_ptr allocate_graph(ggml_cgraph * graph, ggml_tensor * out, ggml_backend_buffer_type_t buft) {
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_ptr galloc = ggml_gallocr_ptr(ggml_gallocr_new(buft));
    bool             result = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(result);
    return galloc;
}

//
// correctness checks for result allocations

static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}

static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto * buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}

static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still "alive", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}

static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}

static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}

//
// test cases

// Scenario where the first backend buffer is completely exhausted and there are further
// tensors which require a second buffer
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}

// Scenario where there is some space left in the first buffer, but not enough to accommodate
// a larger tensor, so a second buffer is required
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}

// Scenario where a single tensor exceeds the max buffer size - in this case the allocator
// should try to create a bigger buffer anyway, and wait for the backend to throw an error.
// Backends may report an artificially lower max size in some cases for compatibility reasons.
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}

// This test assumes a max of 16 buffer chunks, and tries to allocate tensors that would
// require more. Expectation is that the last buffer should grow to fit everything,
// leaving it to the backend to error out if it can't allocate that much.
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}

// Fill up leftover unallocated space of a chunk after allocating a large tensor that
// requires a new chunk.
static void test_fill_leftover_space() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = ggml_pad(ctx, x[0], 2, 0, 0, 0);
    x[3] = ggml_mean(ctx, x[1]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[3], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 12 + 16);
}

// Check that views don't require any extra memory
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}

static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}

static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}

// Check that previously allocated but freed memory is preferred over allocating
// additional memory, even if the remaining space in a chunk would match tensor size better
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}

// test for allocating on multiple devices with some tensors in the graph
// allocated externally (not by gallocr).
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, "a");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, "b");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, "x");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, "a0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "a1")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "b0")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "b1")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, "x1")] = 0;
    node_buffer_ids[get_node_id(graph, "x2")] = 0;
    node_buffer_ids[get_node_id(graph, "x3")] = 1;
    node_buffer_ids[get_node_id(graph, "x4")] = 1;
    node_buffer_ids[get_node_id(graph, "x5")] = 1;
    node_buffer_ids[get_node_id(graph, "x6")] = 1;
    node_buffer_ids[get_node_id(graph, "x7")] = 0;
    node_buffer_ids[get_node_id(graph, "x8")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}

static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

struct reserve_callback_state {
    bool accept = false;
    std::vector<ggml_backend_sched_buffer_reservation> reservations;
};

static bool reserve_callback(
        const ggml_backend_sched_buffer_reservation * reservations,
        size_t n_reservations,
        void * user_data) {
    auto * state = static_cast<reserve_callback_state *>(user_data);
    state->reservations.assign(reservations, reservations + n_reservations);
    return state->accept;
}

static void test_reserve_refusal_preserves_published_layout() {
    dummy_backend backend = dummy_backend_init(64, /*align*/ 4);
    auto [old_ctx, old_graph, old_ctx_ptr] = make_context();
    ggml_tensor * old_x0 = make_input_with_size(old_ctx, 16);
    ggml_tensor * old_x1 = make_input_with_size(old_ctx, 16);
    ggml_tensor * old_out = ggml_add(old_ctx, old_x0, old_x1);
    ggml_set_output(old_out);
    ggml_build_forward_expand(old_graph, old_out);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
    GGML_ASSERT(ggml_gallocr_reserve(galloc.get(), old_graph));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), old_graph));
    const size_t old_total = backend.context->allocated_total();

    reserve_callback_state callback;
    ggml_gallocr_set_reserve_callback(galloc.get(), reserve_callback, &callback);

    auto [new_ctx, new_graph, new_ctx_ptr] = make_context();
    ggml_tensor * new_x0 = make_input_with_size(new_ctx, 48);
    ggml_tensor * new_x1 = make_input_with_size(new_ctx, 48);
    ggml_tensor * new_out = ggml_add(new_ctx, new_x0, new_x1);
    ggml_set_output(new_out);
    ggml_build_forward_expand(new_graph, new_out);

    GGML_ASSERT(!ggml_gallocr_reserve(galloc.get(), new_graph));
    GGML_ASSERT(!callback.reservations.empty());
    GGML_ASSERT(backend.context->allocated_total() == old_total);
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), old_graph));
    check_all_allocated(old_graph);

    callback.accept = true;
    GGML_ASSERT(ggml_gallocr_reserve(galloc.get(), new_graph));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), new_graph));
    check_all_allocated(new_graph);
}

struct sched_callback_state {
    int  n_prepare     = 0;
    int  n_synchronize = 0;
    bool accept        = true;
    bool all_addressed = true;
};

static bool sched_prepare_callback(ggml_backend_sched_t /*sched*/, ggml_cgraph * graph, void * user_data) {
    auto * state = static_cast<sched_callback_state *>(user_data);
    state->n_prepare++;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        if (ggml_graph_node(graph, i)->data == nullptr) {
            state->all_addressed = false;
        }
    }
    return state->accept;
}

static void sched_synchronize_callback(ggml_backend_sched_t /*sched*/, void * user_data) {
    static_cast<sched_callback_state *>(user_data)->n_synchronize++;
}

static void test_sched_prepare_and_synchronize_callbacks() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);

    ggml_backend_t             backends[1] = { backend.get() };
    ggml_backend_buffer_type_t bufts[1]    = { ggml_backend_get_default_buffer_type(backend.get()) };
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(backends, bufts, 1, 16, false, false));

    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tensor * x   = make_input_1d(ctx, 8);
    ggml_tensor * out = ggml_scale(ctx, x, 2.0f);
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    // without callbacks both entry points succeed
    GGML_ASSERT(ggml_backend_sched_prepare_graph(sched.get(), graph));

    sched_callback_state state;
    ggml_backend_sched_set_graph_prepare_callback(sched.get(), sched_prepare_callback, &state);
    ggml_backend_sched_set_synchronize_callback(sched.get(), sched_synchronize_callback, &state);

    // fires once per allocation, after the tensors have addresses
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(state.n_prepare == 1);
    GGML_ASSERT(state.all_addressed);

    // fires on demand for a graph that is not allocated again
    GGML_ASSERT(ggml_backend_sched_prepare_graph(sched.get(), graph));
    GGML_ASSERT(state.n_prepare == 2);

    // computing an allocated graph does not fire it again
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(state.n_prepare == 2);

    // fires after the backends synchronized
    GGML_ASSERT(state.n_synchronize == 0);
    ggml_backend_sched_synchronize(sched.get());
    GGML_ASSERT(state.n_synchronize == 1);

    // a refusal fails the allocation
    state.accept = false;
    ggml_backend_sched_reset(sched.get());
    GGML_ASSERT(!ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(state.n_prepare == 3);
    GGML_ASSERT(!ggml_backend_sched_prepare_graph(sched.get(), graph));
    GGML_ASSERT(state.n_prepare == 4);

    // cleared callbacks are no longer invoked
    ggml_backend_sched_set_graph_prepare_callback(sched.get(), nullptr, nullptr);
    ggml_backend_sched_set_synchronize_callback(sched.get(), nullptr, nullptr);
    GGML_ASSERT(ggml_backend_sched_prepare_graph(sched.get(), graph));
    ggml_backend_sched_synchronize(sched.get());
    GGML_ASSERT(state.n_prepare == 4);
    GGML_ASSERT(state.n_synchronize == 1);
}

// CPU buffer type with a small maximum buffer size, so that a graph spans several chunks
static const char * small_chunk_buft_get_name(ggml_backend_buffer_type_t) {
    return "small_chunk_cpu";
}

static ggml_backend_buffer_t small_chunk_buft_alloc_buffer(ggml_backend_buffer_type_t, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
}

static size_t small_chunk_buft_get_alignment(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

static size_t small_chunk_buft_get_max_size(ggml_backend_buffer_type_t) {
    return 1024;
}

static bool small_chunk_buft_is_host(ggml_backend_buffer_type_t) {
    return true;
}

struct replan_callback_counts {
    int n_reserve     = 0;
    int n_synchronize = 0;
    std::vector<size_t> last_sizes;
};

static bool replan_reserve_callback(ggml_backend_sched_t, const ggml_backend_sched_buffer_reservation * r, size_t n, void * user_data) {
    auto * counts = static_cast<replan_callback_counts *>(user_data);
    counts->n_reserve++;
    counts->last_sizes.clear();
    for (size_t i = 0; i < n; i++) {
        counts->last_sizes.push_back(r[i].size);
    }
    return true;
}

static void replan_synchronize_callback(ggml_backend_sched_t, void * user_data) {
    static_cast<replan_callback_counts *>(user_data)->n_synchronize++;
}

// out = sum of n_branches scaled copies of one input of n_elements floats
static ggml_tensor * build_fan_in(ggml_context * ctx, ggml_cgraph * graph, int64_t n_elements, int n_branches) {
    ggml_tensor * x = make_input_1d(ctx, n_elements);
    std::vector<ggml_tensor *> branches;
    for (int i = 0; i < n_branches; i++) {
        branches.push_back(ggml_scale(ctx, x, 2.0f + i));
    }
    ggml_tensor * out = branches[0];
    for (int i = 1; i < n_branches; i++) {
        out = ggml_add(ctx, out, branches[i]);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);
    return out;
}

static std::vector<size_t> sched_extents(ggml_backend_sched_t sched, std::vector<ggml_backend_buffer_t> & buffers, int n_chunks) {
    std::vector<size_t> extents;
    buffers.assign(n_chunks, nullptr);
    for (int c = 0; c < n_chunks; c++) {
        extents.push_back(ggml_backend_sched_get_compute_extent(sched, 0, c, &buffers[c]));
    }
    return extents;
}

static void test_sched_compute_extent() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);

    ggml_backend_buffer_type buft{};
    buft.iface.get_name      = small_chunk_buft_get_name;
    buft.iface.alloc_buffer  = small_chunk_buft_alloc_buffer;
    buft.iface.get_alignment = small_chunk_buft_get_alignment;
    buft.iface.get_max_size  = small_chunk_buft_get_max_size;
    buft.iface.is_host       = small_chunk_buft_is_host;
    buft.device              = ggml_backend_get_device(backend.get());

    ggml_backend_t             backends[1] = { backend.get() };
    ggml_backend_buffer_type_t bufts[1]    = { &buft };
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(backends, bufts, 1, 64, false, false));

    std::vector<ggml_backend_buffer_t> buffers;

    // wide graph: more than one chunk
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        build_fan_in(ctx, graph, 128, 5);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

        std::vector<size_t> extents = sched_extents(sched.get(), buffers, 4);
        printf("wide extents: %zu %zu %zu %zu\n", extents[0], extents[1], extents[2], extents[3]);
        GGML_ASSERT(extents[0] > 0 && extents[1] > 0);
        for (int c = 0; c < 4; c++) {
            if (extents[c] == 0) {
                GGML_ASSERT(c >= 2 || buffers[c] == nullptr);
                continue;
            }
            GGML_ASSERT(buffers[c] != nullptr);
            GGML_ASSERT(extents[c] <= ggml_backend_buffer_get_size(buffers[c]));
            GGML_ASSERT(extents[c] <= 1024);
        }

        // the chunk buffers are the ones that back the graph's tensors
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_backend_buffer_t b = ggml_graph_node(graph, i)->buffer;
            GGML_ASSERT(b == buffers[0] || b == buffers[1] || b == buffers[2] || b == buffers[3]);
        }

        // a backend index past the chunk count reports nothing
        ggml_backend_buffer_t none = buffers[0];
        GGML_ASSERT(ggml_backend_sched_get_compute_extent(sched.get(), 0, 100, &none) == 0);
        GGML_ASSERT(none == nullptr);
    }

    const std::vector<size_t>                wide_extents = sched_extents(sched.get(), buffers, 4);
    const std::vector<ggml_backend_buffer_t> wide_buffers = buffers;

    // same topology with narrower tensors: the plan is reused, so the extent covers what the
    // narrower tensors actually address and never exceeds the wide extent
    {
        ggml_backend_sched_reset(sched.get());
        auto [ctx, graph, ctx_ptr] = make_context();
        build_fan_in(ctx, graph, 16, 5);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

        std::vector<size_t> extents = sched_extents(sched.get(), buffers, 4);
        printf("narrow same-topology extents: %zu %zu %zu %zu\n", extents[0], extents[1], extents[2], extents[3]);
        for (int c = 0; c < 4; c++) {
            GGML_ASSERT(extents[c] <= wide_extents[c]);
            GGML_ASSERT(buffers[c] == wide_buffers[c]);
        }
        GGML_ASSERT(extents[0] + extents[1] + extents[2] + extents[3] <
                    wide_extents[0] + wide_extents[1] + wide_extents[2] + wide_extents[3]);
    }

    // with replan_on_shrink the narrow graph is placed like a fresh plan and the buffers stay
    {
        std::vector<size_t> sizes;
        for (int c = 0; c < 4; c++) {
            sizes.push_back(wide_buffers[c] ? ggml_backend_buffer_get_size(wide_buffers[c]) : 0);
        }
        replan_callback_counts counts;
        ggml_backend_sched_set_reserve_callback(sched.get(), replan_reserve_callback, &counts);
        ggml_backend_sched_set_synchronize_callback(sched.get(), replan_synchronize_callback, &counts);
        ggml_backend_sched_set_replan_on_shrink(sched.get(), true);
        ggml_backend_sched_reset(sched.get());
        auto [ctx, graph, ctx_ptr] = make_context();
        build_fan_in(ctx, graph, 16, 5);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

        std::vector<size_t> extents = sched_extents(sched.get(), buffers, 4);
        printf("narrow replan extents: %zu %zu %zu %zu\n", extents[0], extents[1], extents[2], extents[3]);
        printf("replan reserve callbacks: %d (sizes:", counts.n_reserve);
        for (size_t v : counts.last_sizes) { printf(" %zu", v); }
        printf("), synchronize callbacks: %d\n", counts.n_synchronize);
        GGML_ASSERT(counts.n_reserve == 1);
        GGML_ASSERT(counts.last_sizes.size() == 2 && counts.last_sizes[0] == sizes[0] && counts.last_sizes[1] == sizes[1]);
        ggml_backend_sched_set_reserve_callback(sched.get(), nullptr, nullptr);
        ggml_backend_sched_set_synchronize_callback(sched.get(), nullptr, nullptr);
        GGML_ASSERT(extents[0] > 0 && extents[0] < wide_extents[0]);
        GGML_ASSERT(extents[1] == 0 && extents[2] == 0 && extents[3] == 0);
        for (int c = 0; c < 4; c++) {
            GGML_ASSERT(buffers[c] == wide_buffers[c]);
            GGML_ASSERT((buffers[c] ? ggml_backend_buffer_get_size(buffers[c]) : 0) == sizes[c]);
        }
        ggml_backend_sched_set_replan_on_shrink(sched.get(), false);
    }

    // a smaller graph is planned again: it fits in chunk 0 and leaves the other chunks untouched
    {
        ggml_backend_sched_reset(sched.get());
        auto [ctx, graph, ctx_ptr] = make_context();
        build_fan_in(ctx, graph, 16, 2);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

        std::vector<size_t> extents = sched_extents(sched.get(), buffers, 4);
        printf("small extents: %zu %zu %zu %zu\n", extents[0], extents[1], extents[2], extents[3]);
        GGML_ASSERT(extents[0] > 0);
        GGML_ASSERT(extents[0] < wide_extents[0]);
        GGML_ASSERT(extents[1] == 0 && extents[2] == 0 && extents[3] == 0);
        GGML_ASSERT(buffers[0] == wide_buffers[0]);
        GGML_ASSERT(buffers[1] == wide_buffers[1]);
    }

    // after a reserve the extent is the planned extent, and an alloc of the same graph reports at most that
    {
        ggml_backend_sched_ptr reserved(ggml_backend_sched_new(backends, bufts, 1, 64, false, false));
        auto [ctx, graph, ctx_ptr] = make_context();
        build_fan_in(ctx, graph, 128, 5);
        GGML_ASSERT(ggml_backend_sched_reserve(reserved.get(), graph));

        std::vector<ggml_backend_buffer_t> reserved_buffers;
        std::vector<size_t> planned = sched_extents(reserved.get(), reserved_buffers, 4);
        printf("reserved extents: %zu %zu %zu %zu\n", planned[0], planned[1], planned[2], planned[3]);
        GGML_ASSERT(planned[0] > 0 && planned[1] > 0);
        for (int c = 0; c < 4; c++) {
            GGML_ASSERT(planned[c] == 0 || reserved_buffers[c] != nullptr);
            GGML_ASSERT(planned[c] >= wide_extents[c]);
        }

        ggml_backend_sched_reset(reserved.get());
        GGML_ASSERT(ggml_backend_sched_alloc_graph(reserved.get(), graph));
        std::vector<size_t> actual = sched_extents(reserved.get(), reserved_buffers, 4);
        printf("alloc after reserve extents: %zu %zu %zu %zu\n", actual[0], actual[1], actual[2], actual[3]);
        for (int c = 0; c < 4; c++) {
            GGML_ASSERT(actual[c] <= planned[c]);
        }
    }
}

// one transformer block over n_tokens positions and a prefix of n_cells cells, then logits for
// n_outputs of the positions; the weights and cache live in weights_buf, outside the plan
struct block_graph {
    ggml_context_ptr      weights_ctx;
    ggml_backend_buffer_ptr weights_buf;
    ggml_context_ptr      ctx;
    ggml_cgraph *         graph;
};

// how a block's last step reads its rows: the outputs' rows gathered, or every row, as a graph that
// keeps every row's hidden state does
enum class block_rows { outputs, every };

static block_graph build_block(int64_t n_tokens, int64_t n_outputs, int64_t n_cells, block_rows rows_read = block_rows::outputs) {
    const int64_t n_embd = 8, n_ff = 16, n_vocab = 32, n_head = 4, n_cells_max = 64;
    block_graph g;
    ggml_init_params wparams{};
    wparams.mem_size = 8 * ggml_tensor_overhead();
    wparams.no_alloc = true;
    g.weights_ctx.reset(ggml_init(wparams));
    ggml_context * w = g.weights_ctx.get();
    ggml_tensor * wq   = ggml_new_tensor_2d(w, GGML_TYPE_F32, n_embd, n_head);
    ggml_tensor * keys = ggml_new_tensor_2d(w, GGML_TYPE_F32, n_head, n_cells_max);
    ggml_tensor * vals = ggml_new_tensor_2d(w, GGML_TYPE_F32, n_cells_max, n_embd);
    ggml_tensor * up   = ggml_new_tensor_2d(w, GGML_TYPE_F32, n_embd, n_ff);
    ggml_tensor * down = ggml_new_tensor_2d(w, GGML_TYPE_F32, n_ff, n_embd);
    ggml_tensor * head = ggml_new_tensor_2d(w, GGML_TYPE_F32, n_embd, n_vocab);
    g.weights_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(w, ggml_backend_cpu_buffer_type()));

    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;
    g.ctx.reset(ggml_init(params));
    ggml_context * ctx = g.ctx.get();
    ggml_tensor * x = make_input_1d(ctx, n_embd * n_tokens);
    x = ggml_reshape_2d(ctx, x, n_embd, n_tokens);
    ggml_tensor * q  = ggml_mul_mat(ctx, wq, ggml_rms_norm(ctx, x, 1e-6f));
    ggml_tensor * k  = ggml_view_2d(ctx, keys, n_head, n_cells, keys->nb[1], 0);
    ggml_tensor * kq = ggml_soft_max(ctx, ggml_mul_mat(ctx, k, q));
    ggml_tensor * v  = ggml_view_2d(ctx, vals, n_cells, n_embd, vals->nb[1], 0);
    x = ggml_add(ctx, x, ggml_mul_mat(ctx, v, kq));
    x = ggml_add(ctx, x, ggml_mul_mat(ctx, down, ggml_gelu(ctx, ggml_mul_mat(ctx, up, ggml_rms_norm(ctx, x, 1e-6f)))));
    ggml_tensor * read = x;
    if (rows_read == block_rows::outputs) {
        ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
        ggml_set_input(rows);
        read = ggml_get_rows(ctx, x, rows);
    }
    ggml_tensor * out = ggml_mul_mat(ctx, head, read);
    ggml_set_output(out);
    g.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g.graph, out);
    return g;
}

// the offset of each allocated node of graph inside its buffer, and the buffer
static std::vector<std::pair<ggml_backend_buffer_t, size_t>> node_placements(ggml_cgraph * graph) {
    std::vector<std::pair<ggml_backend_buffer_t, size_t>> placements;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        if (t->view_src != nullptr) {
            placements.emplace_back(nullptr, 0);
            continue;
        }
        placements.emplace_back(t->buffer, (size_t) ((char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer)));
    }
    return placements;
}

// the placement of each node of graph a plan places, in graph order, skipping views
static std::vector<std::pair<ggml_backend_buffer_t, size_t>> planned_placements(ggml_cgraph * graph) {
    std::vector<std::pair<ggml_backend_buffer_t, size_t>> placements;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        if (t->view_src == nullptr) {
            placements.emplace_back(t->buffer, (size_t) ((char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer)));
        }
    }
    return placements;
}

// A graph that only shrank is planned again under replan_on_shrink, and a fresh plan of it can need
// more of a chunk than the plan that sized the buffers left there. The allocator places the graph
// by that plan instead: where the wide graph was, with no reserve offered, the buffers as they were,
// and the extents what the graph addresses under that plan.
static void test_shrink_replan_keeps_the_fitting_plan() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);

    ggml_backend_buffer_type buft{};
    buft.iface.get_name      = small_chunk_buft_get_name;
    buft.iface.alloc_buffer  = small_chunk_buft_alloc_buffer;
    buft.iface.get_alignment = small_chunk_buft_get_alignment;
    buft.iface.get_max_size  = small_chunk_buft_get_max_size;
    buft.iface.is_host       = small_chunk_buft_is_host;
    buft.device              = ggml_backend_get_device(backend.get());

    ggml_backend_t             backends[1] = { backend.get() };
    ggml_backend_buffer_type_t bufts[1]    = { &buft };
    const int                  n_chunks    = 4;

    // a fresh plan of the narrow graph needs more of some chunk than the wide graph's plan
    std::vector<size_t> wide_sizes;
    std::vector<size_t> fresh_sizes;
    for (int narrow = 0; narrow < 2; narrow++) {
        ggml_backend_sched_ptr fresh(ggml_backend_sched_new(backends, bufts, 1, 64, false, false));
        block_graph g = narrow ? build_block(4, 1, 16) : build_block(4, 4, 64);
        replan_callback_counts counts;
        ggml_backend_sched_set_reserve_callback(fresh.get(), replan_reserve_callback, &counts);
        GGML_ASSERT(ggml_backend_sched_reserve(fresh.get(), g.graph));
        (narrow ? fresh_sizes : wide_sizes) = counts.last_sizes;
    }
    bool grows = fresh_sizes.size() > wide_sizes.size();
    for (size_t c = 0; c < std::min(fresh_sizes.size(), wide_sizes.size()); c++) {
        grows = grows || fresh_sizes[c] > wide_sizes[c];
    }
    GGML_ASSERT(grows && "the narrow graph's own plan fits the wide graph's chunks");

    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 1, 64, false, false));
    replan_callback_counts counts;
    ggml_backend_sched_set_reserve_callback(sched.get(), replan_reserve_callback, &counts);
    ggml_backend_sched_set_replan_on_shrink(sched.get(), true);

    std::vector<ggml_backend_buffer_t> buffers;
    block_graph wide = build_block(4, 4, 64);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), wide.graph));
    const std::vector<size_t>                wide_extents = sched_extents(sched.get(), buffers, n_chunks);
    const std::vector<ggml_backend_buffer_t> wide_buffers = buffers;
    std::vector<size_t>                      sizes;
    for (int c = 0; c < n_chunks; c++) {
        sizes.push_back(buffers[c] ? ggml_backend_buffer_get_size(buffers[c]) : 0);
    }
    const auto wide_placements = node_placements(wide.graph);
    const int  n_reserve       = counts.n_reserve;

    for (int pass = 0; pass < 2; pass++) {
        // first allocated directly, then reserved before it is allocated
        ggml_backend_sched_reset(sched.get());
        block_graph narrow = build_block(4, 1, 16);
        if (pass == 1) {
            GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), narrow.graph));
            const std::vector<size_t> reserved = sched_extents(sched.get(), buffers, n_chunks);
            for (int c = 0; c < n_chunks; c++) {
                GGML_ASSERT(reserved[c] <= wide_extents[c]);
            }
            ggml_backend_sched_reset(sched.get());
        }
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), narrow.graph));
        GGML_ASSERT(counts.n_reserve == n_reserve && "a kept plan offered a reserve");

        const std::vector<size_t> extents = sched_extents(sched.get(), buffers, n_chunks);
        printf("kept plan extents (pass %d): %zu %zu %zu %zu\n", pass, extents[0], extents[1], extents[2], extents[3]);
        for (int c = 0; c < n_chunks; c++) {
            GGML_ASSERT(buffers[c] == wide_buffers[c]);
            GGML_ASSERT((buffers[c] ? ggml_backend_buffer_get_size(buffers[c]) : 0) == sizes[c]);
            GGML_ASSERT(extents[c] <= wide_extents[c]);
        }
        GGML_ASSERT(node_placements(narrow.graph) == wide_placements);
        check_no_overlap(narrow.graph);
    }

    // a backend's graph optimizer may move a view of the cache, whose memory no plan places, as the
    // view narrows; the narrow graph so reordered is still held by the wide graph's plan
    {
        ggml_backend_sched_reset(sched.get());
        block_graph narrow = build_block(4, 1, 16);
        ggml_cgraph * graph = narrow.graph;
        int view = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            if (graph->nodes[i]->view_src != nullptr && graph->nodes[i]->data != nullptr) {
                view = i;
            }
        }
        GGML_ASSERT(view > 0 && "the block has no view of the cache to move");
        ggml_tensor * moved = graph->nodes[view];
        std::rotate(graph->nodes, graph->nodes + view, graph->nodes + view + 1);
        GGML_ASSERT(graph->nodes[0] == moved);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        GGML_ASSERT(counts.n_reserve == n_reserve && "a reordered narrow graph was planned afresh");
        sched_extents(sched.get(), buffers, n_chunks);
        for (int c = 0; c < n_chunks; c++) {
            GGML_ASSERT(buffers[c] == wide_buffers[c]);
        }
        GGML_ASSERT(planned_placements(graph) == planned_placements(wide.graph));
        check_no_overlap(graph);
    }

    // a graph whose own plan fits the chunks is placed by that plan and offered; a graph wider than
    // it but no wider than the wide graph is then placed by the wide graph's plan again
    {
        ggml_backend_sched_reset(sched.get());
        block_graph single = build_block(1, 1, 16);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), single.graph));
        GGML_ASSERT(counts.n_reserve == n_reserve + 1 && "a fitting plan was not adopted");
        ggml_backend_sched_reset(sched.get());
        block_graph narrow = build_block(4, 1, 16);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), narrow.graph));
        GGML_ASSERT(counts.n_reserve == n_reserve + 1 && "the wide graph's plan was not kept");
        GGML_ASSERT(node_placements(narrow.graph) == wide_placements);
        sched_extents(sched.get(), buffers, n_chunks);
        for (int c = 0; c < n_chunks; c++) {
            GGML_ASSERT(buffers[c] == wide_buffers[c]);
        }
    }

    // a graph wider than the wide graph is planned and offered as before
    ggml_backend_sched_reset(sched.get());
    block_graph wider = build_block(8, 8, 64);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), wider.graph));
    GGML_ASSERT(counts.n_reserve == n_reserve + 2);
    ggml_backend_sched_set_reserve_callback(sched.get(), nullptr, nullptr);
}

// The buffers are sized by a graph of one topology, and a reserve then names a graph of another
// that fits them, as a context whose graphs change shape after its first reserve does. That
// reserve's plan is the one a narrower graph of its topology falls back on: the narrow graph is
// placed where the reserved graph was, with no reserve offered, though a fresh plan of it would grow
// a chunk. A graph allocated without a reserve never takes that place.
static void test_covering_reserve_names_the_graphs_to_hold() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);

    ggml_backend_buffer_type buft{};
    buft.iface.get_name      = small_chunk_buft_get_name;
    buft.iface.alloc_buffer  = small_chunk_buft_alloc_buffer;
    buft.iface.get_alignment = small_chunk_buft_get_alignment;
    buft.iface.get_max_size  = small_chunk_buft_get_max_size;
    buft.iface.is_host       = small_chunk_buft_is_host;
    buft.device              = ggml_backend_get_device(backend.get());

    ggml_backend_t             backends[1] = { backend.get() };
    ggml_backend_buffer_type_t bufts[1]    = { &buft };
    const int                  n_chunks    = 4;

    for (int covering = 0; covering < 2; covering++) {
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 1, 64, false, false));
        replan_callback_counts counts;
        ggml_backend_sched_set_reserve_callback(sched.get(), replan_reserve_callback, &counts);
        ggml_backend_sched_set_replan_on_shrink(sched.get(), true);

        // the first topology sizes the buffers
        block_graph first = build_block(4, 4, 64, block_rows::outputs);
        GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), first.graph));
        std::vector<ggml_backend_buffer_t> buffers;
        sched_extents(sched.get(), buffers, n_chunks);
        std::vector<size_t> sizes;
        for (int c = 0; c < n_chunks; c++) {
            sizes.push_back(buffers[c] ? ggml_backend_buffer_get_size(buffers[c]) : 0);
        }

        // the second topology fits them
        ggml_backend_sched_reset(sched.get());
        block_graph second = build_block(4, 4, 64, block_rows::every);
        if (covering) {
            GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), second.graph));
            ggml_backend_sched_reset(sched.get());
            second = build_block(4, 4, 64, block_rows::every);
        }
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), second.graph));
        sched_extents(sched.get(), buffers, n_chunks);
        for (int c = 0; c < n_chunks; c++) {
            GGML_ASSERT((buffers[c] ? ggml_backend_buffer_get_size(buffers[c]) : 0) == sizes[c] &&
                        "the second topology grew the buffers");
        }
        const auto second_placements = node_placements(second.graph);
        const int  before            = counts.n_reserve;

        ggml_backend_sched_reset(sched.get());
        block_graph narrow = build_block(4, 4, 16, block_rows::every);
        const bool placed = ggml_backend_sched_alloc_graph(sched.get(), narrow.graph);
        printf("covering reserve %d: narrow graph placed %d, reserves offered %d\n", covering, placed,
               counts.n_reserve - before);
        if (covering) {
            GGML_ASSERT(placed);
            GGML_ASSERT(counts.n_reserve == before && "the reserved graph's plan was not kept");
            GGML_ASSERT(node_placements(narrow.graph) == second_placements);
            check_no_overlap(narrow.graph);
        } else {
            GGML_ASSERT(counts.n_reserve > before && "a graph allocated without a reserve became a covering plan");
        }
    }
}

// The buffers are sized by a wide graph of one topology, and a reserve then names a narrower graph
// of another. That reserve's plan covers its own topology only: a narrower graph of the first
// topology is still placed by the wide graph's plan, with no reserve offered, though a fresh plan of
// it would grow a chunk; a narrower graph of the second topology is placed by the reserve's plan.
static void test_narrow_reserve_of_another_topology_keeps_the_plan() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);

    ggml_backend_buffer_type buft{};
    buft.iface.get_name      = small_chunk_buft_get_name;
    buft.iface.alloc_buffer  = small_chunk_buft_alloc_buffer;
    buft.iface.get_alignment = small_chunk_buft_get_alignment;
    buft.iface.get_max_size  = small_chunk_buft_get_max_size;
    buft.iface.is_host       = small_chunk_buft_is_host;
    buft.device              = ggml_backend_get_device(backend.get());

    ggml_backend_t             backends[1] = { backend.get() };
    ggml_backend_buffer_type_t bufts[1]    = { &buft };

    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 1, 64, false, false));
    replan_callback_counts counts;
    ggml_backend_sched_set_reserve_callback(sched.get(), replan_reserve_callback, &counts);
    ggml_backend_sched_set_replan_on_shrink(sched.get(), true);

    // the wide graph of the first topology sizes the buffers
    block_graph wide = build_block(4, 4, 64, block_rows::outputs);
    GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), wide.graph));
    const std::vector<size_t> wide_sizes = counts.last_sizes;
    const size_t              wide_size  = ggml_backend_sched_get_buffer_size(sched.get(), backend.get());
    ggml_backend_sched_reset(sched.get());
    wide = build_block(4, 4, 64, block_rows::outputs);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), wide.graph));
    const auto wide_placements = node_placements(wide.graph);

    // a narrower reserve of the second topology fits them
    ggml_backend_sched_reset(sched.get());
    block_graph other = build_block(4, 4, 48, block_rows::every);
    GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), other.graph));
    // each reservation is max(current, planned) for one chunk
    GGML_ASSERT(counts.last_sizes == wide_sizes && "the second topology's reserve grew the buffers");
    ggml_backend_sched_reset(sched.get());
    other = build_block(4, 4, 48, block_rows::every);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), other.graph));
    const auto other_placements = node_placements(other.graph);
    const int n_reserve = counts.n_reserve;

    // a narrower graph of the first topology, whose fresh plan grows a chunk, keeps the wide plan
    ggml_backend_sched_reset(sched.get());
    block_graph narrow = build_block(4, 1, 16, block_rows::outputs);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), narrow.graph));
    printf("narrow graph of the first topology: reserves offered %d\n", counts.n_reserve - n_reserve);
    GGML_ASSERT(counts.n_reserve == n_reserve && "the other topology's reserve took the wide graph's plan");
    GGML_ASSERT(node_placements(narrow.graph) == wide_placements);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == wide_size);
    check_no_overlap(narrow.graph);

    // the second topology still falls back on its reserve's plan
    ggml_backend_sched_reset(sched.get());
    block_graph other_narrow = build_block(4, 1, 16, block_rows::every);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), other_narrow.graph));
    printf("narrow graph of the second topology: reserves offered %d\n", counts.n_reserve - n_reserve);
    GGML_ASSERT(counts.n_reserve == n_reserve && "the second topology's reserve plan was not kept");
    GGML_ASSERT(node_placements(other_narrow.graph) == other_placements);
    check_no_overlap(other_narrow.graph);
    ggml_backend_sched_set_reserve_callback(sched.get(), nullptr, nullptr);
}

// Test re-using gallocr for a different graph. The new graph has the same
// total size, but one of the chunks is larger, so reallocation is required.
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}

static void test_backend_graph_optimize(ggml_backend_t /*backend*/, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(graph->n_nodes == 3);
    params->add_alloc_dep(params->user_data, graph->nodes[0], graph->nodes[2]);
}

static bool graph_reuses_allocation(bool add_alloc_dep) {
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    x[2] = ggml_scale(ctx, x[1], 2.0f);
    x[3] = ggml_scale(ctx, x[2], 2.0f);

    ggml_set_output(x[3]);
    ggml_build_forward_expand(graph, x[3]);

    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    if (add_alloc_dep) {
        backend.context->backend.iface.graph_optimize = test_backend_graph_optimize;
    }

    ggml_backend_t             backend_ptr = &backend.context->backend;
    ggml_backend_buffer_type_t buft        = &backend.buffer_type;
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(&backend_ptr, &buft, 1, 8, false, true));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    return x[1]->data == x[2]->data;
}

static void test_graph_optimize_alloc_dep() {
    GGML_ASSERT(graph_reuses_allocation(false));
    GGML_ASSERT(!graph_reuses_allocation(true));
}

// keeps the first node of the split alive until its last one
static void keep_first_until_last(ggml_backend_t /*backend*/, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    if (graph->n_nodes >= 2) {
        params->add_alloc_dep(params->user_data, graph->nodes[0], graph->nodes[graph->n_nodes - 1]);
    }
}

static int graph_node_index(ggml_cgraph * graph, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        if (ggml_graph_node(graph, i) == t) {
            return i;
        }
    }
    return -1;
}

// The scheduler's graph copy (ggml_backend_sched_get_graph) gives every node of the original
// graph its original use count, so that a fusion check over it decides as one over a split does,
// and gives the nodes the scheduler inserts - split input copies, the views that keep their
// sources alive, alloc dependencies - a count of 0.
static void test_sched_graph_use_counts() {
    auto [ctx, graph, ctx_ptr] = make_context();

    // x -> a -> b, and c = a + b on backend A; d reads c on backend B, so x and c are split
    // inputs, and A keeps a alive until c
    ggml_tensor * x = make_input_with_size(ctx, 16);
    ggml_tensor * a = ggml_scale(ctx, x, 2.0f);
    ggml_tensor * b = ggml_scale(ctx, a, 2.0f);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_tensor * d = ggml_scale(ctx, c, 2.0f);
    assign_names(ctx);
    ggml_set_output(d);
    ggml_build_forward_expand(graph, d);

    std::vector<ggml_tensor *> nodes;
    std::vector<int32_t>       counts;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        nodes.push_back(ggml_graph_node(graph, i));
        counts.push_back(ggml_node_get_use_count(graph, i));
    }
    GGML_ASSERT(counts[graph_node_index(graph, a)] == 2);

    dummy_backend backend_a = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);
    backend_a.context->backend.iface.graph_optimize = keep_first_until_last;

    ggml_backend_t             backends[2] = { &backend_a.context->backend, &backend_b.context->backend };
    ggml_backend_buffer_type_t bufts[2]    = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(backends, bufts, 2, 16, false, false));
    for (ggml_tensor * t : { a, b, c }) {
        ggml_backend_sched_set_tensor_backend(sched.get(), t, backends[0]);
    }
    ggml_backend_sched_set_tensor_backend(sched.get(), d, backends[1]);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched.get()) == 2);

    ggml_cgraph * scheduled = ggml_backend_sched_get_graph(sched.get());

    int  n_original = 0;
    int  n_inserted = 0;
    bool has_dep    = false;
    for (int i = 0; i < ggml_graph_n_nodes(scheduled); i++) {
        ggml_tensor * node = ggml_graph_node(scheduled, i);
        const int32_t count = ggml_node_get_use_count(scheduled, i);

        const auto it = std::find(nodes.begin(), nodes.end(), node);
        if (it != nodes.end()) {
            GGML_ASSERT(count == counts[it - nodes.begin()]);
            n_original++;
        } else {
            GGML_ASSERT(count == 0);
            n_inserted++;
            has_dep = has_dep || node->src[0] == a;
        }
    }

    GGML_ASSERT(n_original == (int) nodes.size());
    // x and c each add a source view and a copy, and A adds the dependency on a
    GGML_ASSERT(n_inserted == 5);
    GGML_ASSERT(has_dep);
}

// Check that the size reported by ggml_backend_alloc_ctx_tensors_from_buft_size
// matches the actual size of the buffer allocated for the ctx tensors
static ggml_backend_buffer_ptr check_size_matches(ggml_backend_buffer_type_t buft, ggml_context * ctx) {
    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        tensors.push_back(t);
    }

    const size_t expected_size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(buft, tensors.data(), (int) tensors.size()) == expected_size);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
    GGML_ASSERT((buffer != nullptr) == (expected_size != 0));
    if (buffer) {
        GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == expected_size);
    }
    return buffer;
}

// Check that all tensors are placed in a single buffer when they fit within the
// backend's max size
static void test_buft_alloc_buffer_n_single_buffer() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { x[0], x[1], x[2] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(!ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
    for (ggml_tensor * t : tensors) {
        GGML_ASSERT(t->buffer == buffer.get());
        GGML_ASSERT(t->data != nullptr);
    }
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            GGML_ASSERT(!memory_overlap(tensors[i], tensors[j]));
        }
    }
}

// Check that tensors are split across multiple underlying buffers when they don't
// fit within the backend's max size
static void test_buft_alloc_buffer_n_multi_buffer() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[4] = { x[0], x[1], x[2], x[3] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 4));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
    for (ggml_tensor * t : tensors) {
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            GGML_ASSERT(!memory_overlap(tensors[i], tensors[j]));
        }
    }
}

// Check that allocating tensors with zero total size returns nullptr
static void test_buft_alloc_buffer_n_zero_size() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_1d(ctx, 0);
    x[1] = make_input_1d(ctx, 0);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2) == nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type) == nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 0);
}

// Check that tensors already allocated to a buffer are left in place, and only the
// remaining tensors are allocated to a new buffer
static void test_buft_alloc_buffer_n_already_allocated() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx, 8);
    a[1] = make_input_with_size(ctx, 8);
    assign_names(ctx, "a");

    ggml_backend_buffer_ptr buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type));
    GGML_ASSERT(buf_a != nullptr);

    ggml_tensor * b = make_input_with_size(ctx, 8);
    assign_names(ctx, "b");

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 8);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, a, 2) == nullptr);

    ggml_tensor * tensors[3] = { a[0], a[1], b };
    ggml_backend_buffer_ptr buf_b(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buf_b != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(a[0]->buffer == buf_a.get());
    GGML_ASSERT(a[1]->buffer == buf_a.get());
    GGML_ASSERT(b->buffer == buf_b.get());
    GGML_ASSERT(ggml_backend_buffer_get_size(buf_b.get()) == 8);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type) == nullptr);
}

// Check that views don't require any extra memory and share the base tensor's data
static void test_buft_alloc_buffer_n_views() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * base  = make_input_1d(ctx, 4);
    ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
    ggml_tensor * extra = make_input_1d(ctx, 2);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { base, view, extra };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
    GGML_ASSERT(base->buffer == buffer.get());
    GGML_ASSERT(view->buffer == buffer.get());
    GGML_ASSERT(extra->buffer == buffer.get());
    GGML_ASSERT(view->data == base->data);
}

// Check that the reported size matches the allocated size in various scenarios:
// single buffer, multi buffer, and views
static void test_alloc_ctx_tensors_from_buft_size_matches() {
    {
        dummy_backend backend      = dummy_backend_init(SIZE_MAX, 8);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 4);
        x[1] = make_input_with_size(ctx, 4);
        x[2] = make_input_with_size(ctx, 8);
        assign_names(ctx);

        GGML_UNUSED(x);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 24);
    }
    {
        dummy_backend backend      = dummy_backend_init(16);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 8);
        x[1] = make_input_with_size(ctx, 8);
        x[2] = make_input_with_size(ctx, 8);
        x[3] = make_input_with_size(ctx, 8);
        assign_names(ctx);

        GGML_UNUSED(x);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 32);
    }
    {
        dummy_backend backend      = dummy_backend_init(SIZE_MAX);
        auto [ctx, graph, ctx_ptr] = make_context();

        ggml_tensor * base  = make_input_1d(ctx, 4);
        ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
        ggml_tensor * extra = make_input_1d(ctx, 2);
        assign_names(ctx);

        GGML_UNUSED(view);
        GGML_UNUSED(extra);

        ggml_backend_buffer_ptr buffer = check_size_matches(&backend.buffer_type, ctx);
        GGML_ASSERT(buffer != nullptr);
        GGML_ASSERT(backend.context->allocated_total() == 24);
    }
}

// Check that a backend-provided alloc_buffer_n implementation takes precedence over
// the default one
static void test_buft_alloc_buffer_n_custom_override() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.alloc_buffer_n = dummy_backend_buffer_type_alloc_buffer_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->custom_alloc_buffer_n_called);
    GGML_ASSERT(backend.context->buffers.size() == 1);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 64);
    GGML_ASSERT(x[0]->buffer == buffer.get());
    GGML_ASSERT(x[1]->buffer == buffer.get());
}

// Check that get_alloc_size_n predicts a single buffer allocation
static void test_buft_get_alloc_size_n_single_buffer() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 16);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 16);
}

// Check that get_alloc_size_n accounts for splitting into multiple buffers
static void test_buft_get_alloc_size_n_multi_buffer() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[4] = { x[0], x[1], x[2], x[3] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 4) == 32);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 4));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer.get()));
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
}

// Check that a tensor larger than max_size is still accounted for
static void test_buft_get_alloc_size_n_single_tensor_exceeds_max() {
    dummy_backend backend      = dummy_backend_init(8);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x = make_input_with_size(ctx, 16);
    assign_names(ctx);

    ggml_tensor * tensors[1] = { x };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 1) == 16);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 1));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 16);
}

// Check that zero-size tensors report a total allocation size of 0
static void test_buft_get_alloc_size_n_zero_size() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_1d(ctx, 0);
    x[1] = make_input_1d(ctx, 0);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 0);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2) == nullptr);
}

// Check that already-allocated tensors are not counted again
static void test_buft_get_alloc_size_n_already_allocated() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx, 8);
    a[1] = make_input_with_size(ctx, 8);
    assign_names(ctx, "a");

    ggml_backend_buffer_ptr buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx, &backend.buffer_type));
    GGML_ASSERT(buf_a != nullptr);

    ggml_tensor * b = make_input_with_size(ctx, 8);
    assign_names(ctx, "b");

    ggml_tensor * tensors[3] = { a[0], a[1], b };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 3) == 8);

    ggml_backend_buffer_ptr buf_b(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buf_b != nullptr);
    GGML_ASSERT(backend.context->buffers.size() == 2);
    GGML_ASSERT(a[0]->buffer == buf_a.get());
    GGML_ASSERT(a[1]->buffer == buf_a.get());
    GGML_ASSERT(b->buffer == buf_b.get());
    GGML_ASSERT(ggml_backend_buffer_get_size(buf_b.get()) == 8);
}

// Check that views do not add to the projected allocation size
static void test_buft_get_alloc_size_n_views() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * base  = make_input_1d(ctx, 4);
    ggml_tensor * view  = ggml_view_1d(ctx, base, 2, 0);
    ggml_tensor * extra = make_input_1d(ctx, 2);
    assign_names(ctx);

    ggml_tensor * tensors[3] = { base, view, extra };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 3) == 24);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 3));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 24);
}

// Check that n_tensors == 0 is handled
static void test_buft_get_alloc_size_n_n_tensors_zero() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, nullptr, 0) == 0);
    GGML_ASSERT(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, nullptr, 0) == nullptr);
}

// Check that get_alloc_size_n respects the backend alignment
static void test_buft_get_alloc_size_n_alignment() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX, 16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 4);
    x[1] = make_input_with_size(ctx, 4);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 32);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 32);
}

// Check that a backend-provided get_alloc_size_n implementation takes precedence over
// the default one
static void test_buft_get_alloc_size_n_custom_override() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.alloc_buffer_n   = dummy_backend_buffer_type_alloc_buffer_n_custom;
    backend.buffer_type.iface.get_alloc_size_n = dummy_backend_buffer_type_get_alloc_size_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 64);
    GGML_ASSERT(backend.context->custom_get_alloc_size_n_called);

    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer_n(&backend.buffer_type, tensors, 2));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(backend.context->custom_alloc_buffer_n_called);
    GGML_ASSERT(ggml_backend_buffer_get_size(buffer.get()) == 64);
}

// Check that the custom get_alloc_size_n override is used by the context size helper
static void test_buft_get_alloc_size_n_custom_override_ctx_size() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.get_alloc_size_n = dummy_backend_buffer_type_get_alloc_size_n_custom;

    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    GGML_UNUSED(x);

    GGML_ASSERT(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, &backend.buffer_type) == 64);
}

// Check that querying the projected allocation size does not allocate anything
static void test_buft_get_alloc_size_n_does_not_allocate() {
    dummy_backend backend      = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    assign_names(ctx);

    ggml_tensor * tensors[2] = { x[0], x[1] };
    GGML_ASSERT(ggml_backend_buft_get_alloc_size_n(&backend.buffer_type, tensors, 2) == 16);
    GGML_ASSERT(backend.context->buffers.empty());
    GGML_ASSERT(x[0]->data == nullptr);
    GGML_ASSERT(x[1]->data == nullptr);
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main() {
    run("test_max_size_too_many_tensors", test_max_size_too_many_tensors);
    run("test_max_size_tensor_too_large", test_max_size_tensor_too_large);
    run("test_tensor_larger_than_max_size", test_tensor_larger_than_max_size);
    run("test_not_enough_chunks", test_not_enough_chunks);
    run("test_fill_leftover_space", test_fill_leftover_space);
    run("test_view_inplace", test_view_inplace);
    run("test_reuse_and_free", test_reuse_and_free);
    run("test_merge_free_block(32)", []() { test_merge_free_block(32); });
    run("test_merge_free_block(SIZE_MAX)", []() { test_merge_free_block(SIZE_MAX); });
    run("test_prefer_already_allocated_memory", test_prefer_already_allocated_memory);
    run("test_multiple_buffer_types", test_multiple_buffer_types);
    run("test_buffer_size_zero", test_buffer_size_zero);
    run("test_reserve_refusal_preserves_published_layout", test_reserve_refusal_preserves_published_layout);
    run("test_sched_prepare_and_synchronize_callbacks", test_sched_prepare_and_synchronize_callbacks);
    run("test_sched_compute_extent", test_sched_compute_extent);
    run("test_shrink_replan_keeps_the_fitting_plan", test_shrink_replan_keeps_the_fitting_plan);
    run("test_covering_reserve_names_the_graphs_to_hold", test_covering_reserve_names_the_graphs_to_hold);
    run("test_narrow_reserve_of_another_topology_keeps_the_plan", test_narrow_reserve_of_another_topology_keeps_the_plan);
    run("test_reallocation", test_reallocation);
    run("test_graph_optimize_alloc_dep", test_graph_optimize_alloc_dep);
    run("test_sched_graph_use_counts", test_sched_graph_use_counts);
    run("test_buft_alloc_buffer_n_single_buffer", test_buft_alloc_buffer_n_single_buffer);
    run("test_buft_alloc_buffer_n_multi_buffer", test_buft_alloc_buffer_n_multi_buffer);
    run("test_buft_alloc_buffer_n_zero_size", test_buft_alloc_buffer_n_zero_size);
    run("test_buft_alloc_buffer_n_already_allocated", test_buft_alloc_buffer_n_already_allocated);
    run("test_buft_alloc_buffer_n_views", test_buft_alloc_buffer_n_views);
    run("test_alloc_ctx_tensors_from_buft_size_matches", test_alloc_ctx_tensors_from_buft_size_matches);
    run("test_buft_alloc_buffer_n_custom_override", test_buft_alloc_buffer_n_custom_override);
    run("test_buft_get_alloc_size_n_single_buffer", test_buft_get_alloc_size_n_single_buffer);
    run("test_buft_get_alloc_size_n_multi_buffer", test_buft_get_alloc_size_n_multi_buffer);
    run("test_buft_get_alloc_size_n_single_tensor_exceeds_max", test_buft_get_alloc_size_n_single_tensor_exceeds_max);
    run("test_buft_get_alloc_size_n_zero_size", test_buft_get_alloc_size_n_zero_size);
    run("test_buft_get_alloc_size_n_already_allocated", test_buft_get_alloc_size_n_already_allocated);
    run("test_buft_get_alloc_size_n_views", test_buft_get_alloc_size_n_views);
    run("test_buft_get_alloc_size_n_n_tensors_zero", test_buft_get_alloc_size_n_n_tensors_zero);
    run("test_buft_get_alloc_size_n_alignment", test_buft_get_alloc_size_n_alignment);
    run("test_buft_get_alloc_size_n_custom_override", test_buft_get_alloc_size_n_custom_override);
    run("test_buft_get_alloc_size_n_custom_override_ctx_size", test_buft_get_alloc_size_n_custom_override_ctx_size);
    run("test_buft_get_alloc_size_n_does_not_allocate", test_buft_get_alloc_size_n_does_not_allocate);
    return 0;
}
