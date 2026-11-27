// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

// set the number of extra command buffers a graph compute is encoded across (default: 1)
// kernels are fused only within one command buffer, so with n_cb > 0 the split points can break a fusable group;
// with n_cb == 0 the whole graph is encoded in one command buffer and fuses the same way however it is sized
// obtained through ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_set_n_cb")
typedef void (*ggml_backend_metal_set_n_cb_t)(ggml_backend_t backend, int n_cb);

// the command buffer error that put the backend into its error state, or NULL if there is none
// lets a caller tell a GPU timeout from a fault or an abandoned queue; valid until the backend computes again or is freed
GGML_BACKEND_API const char * ggml_backend_metal_last_error(ggml_backend_t backend);

// Reorder barriers: the graph reorder runs while the scheduler splits the graph, before a caller can install a
// schedule, and may move a node across a cut it does not depend on. A barrier node keeps its place: no node moves
// across it in either direction, and a fusion group that contains it stays where it is.

// is `node` a reorder barrier? called once per graph node while a graph is reordered
typedef bool (*ggml_backend_metal_reorder_barrier_t)(const struct ggml_tensor * node, void * user_data);

// make every node for which `is_barrier` returns true a barrier (default: none); NULL removes the predicate
// applies to graphs reordered after the call, so install it before allocating the graph
// `user_data` is passed to every call and must outlive the predicate's installation
GGML_BACKEND_API void ggml_backend_metal_set_reorder_barrier(
        ggml_backend_t backend, ggml_backend_metal_reorder_barrier_t is_barrier, void * user_data);

// the installed predicate, NULL if there is none; `user_data` (may be NULL) receives its user data
GGML_BACKEND_API ggml_backend_metal_reorder_barrier_t ggml_backend_metal_get_reorder_barrier(
        ggml_backend_t backend, void ** user_data);

//
// buffers over caller host memory, resolved through a caller range table
//
// A buffer from ggml_backend_metal_buffer_from_ranges spans the caller's host memory [base, base + size), tiled by
// n_ranges consecutive ranges: range 0 starts at base, each next one where the previous ends, and the last ends at
// base + size. Each range has a Metal buffer that the caller supplies, or none. A kernel that uses a tensor binds the
// Metal buffer of the range that holds the tensor, at the tensor's offset in the range. Encoding a kernel that uses a
// tensor in a range with no Metal buffer, or a tensor that crosses a range boundary, aborts, naming the tensor. Host
// access to the tensors (set, get, memset, clear) reads and writes the host memory and needs no Metal buffer.
//
// Ownership: the backend makes no Metal buffer and no residency set for such a buffer, and never retains, releases or
// requests residency for the caller's objects. A range's Metal buffer is an id<MTLBuffer> of the device, passed as
// void *, whose contents start at or before the range and hold a non-empty prefix of it, or the whole range, e.g. a
// no-copy buffer over the range's first pages. The backend reads it when it encodes or blits work that uses a tensor in
// the range, from the call that installs it until the call that replaces it. The caller keeps it alive over that span,
// and after it until every command buffer that binds it has completed: the backend encodes with unretained references.
// Freeing the backend buffer releases nothing of the caller's.
//
// Prefixes: a kernel may touch every byte of each tensor it uses, [data, data + ggml_nbytes), so encoding one whose
// tensor ends past the prefix its range's Metal buffer holds aborts, naming the tensor; a view's bytes are those it
// spans, not its source's. A caller that wraps only the bytes in use grows the prefix by installing a longer Metal
// buffer over the same pages before encoding work that reaches further.
//
// A buffer type that a caller makes with a Metal device as its device is supported by that device, like the device's
// own types; its buffers must place every tensor in Metal buffers, e.g. in buffers from
// ggml_backend_metal_buffer_from_ranges.
//

// a range of a buffer from ggml_backend_metal_buffer_from_ranges
struct ggml_backend_metal_range {
    size_t size;  // bytes, > 0
    void * metal; // id<MTLBuffer> over the range, or NULL for none
};

// NULL, with a log line, when dev is not a Metal device, the ranges do not tile [base, base + size), or a Metal buffer
// is not of the device or holds no prefix of its range; the range table is copied
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_metal_buffer_from_ranges(
        ggml_backend_dev_t dev, void * base, size_t size, const struct ggml_backend_metal_range * ranges, int n_ranges);

// replace range i's Metal buffer with `metal` (NULL: none), checked as above; false, with a log line and nothing
// changed, when the buffer is not from ggml_backend_metal_buffer_from_ranges, i is not one of its ranges or `metal` is
// refused. Work that binds other ranges may be encoded or in flight meanwhile. Before replacing a range's Metal buffer
// the caller ensures that no encode that uses the range is running, unless both Metal buffers map the same pages from
// the same first byte (a longer or shorter prefix of them): an encode running meanwhile binds each tensor to one of the
// two, whole. Command buffers in flight that bind the previous one still use it, so the caller releases it only once
// they have completed.
GGML_BACKEND_API bool ggml_backend_metal_buffer_set_range(ggml_backend_buffer_t buffer, int i, void * metal);

// the id<MTLDevice> of a Metal device, as void *, not retained: valid while the device is; NULL, with a log line, for
// a device that is not a Metal device
GGML_BACKEND_API void * ggml_backend_metal_device_get_mtl_device(ggml_backend_dev_t dev);

//
// encoding into caller command buffers
//
// A caller that schedules a graph itself encodes ranges of its nodes into command buffers it makes from the device's
// queue, commits them in the order it chooses, and owns them throughout. The backend holds nothing of the caller's.
//

// Encode nodes [i0, i1) of `graph` into `cmd_buf`, an id<MTLCommandBuffer> of the backend's device passed as void *,
// as a graph compute encodes the same nodes into one of its command buffers: the same pipelines, the same fusion (a
// fused run never crosses i0 or i1), the same concurrency, with memory barriers between dependent nodes of the range.
// Every tensor the range uses must be in a Metal buffer of the device.
// The call makes and ends its own encoders on cmd_buf and neither enqueues, commits, retains nor waits for it; it
// holds nothing once it returns. The caller keeps cmd_buf and every buffer the range binds alive until cmd_buf has
// completed: the backend encodes with unretained references.
// GGML_STATUS_FAILED, logging why, for a range outside the graph, a NULL command buffer or a backend that is not Metal.
GGML_BACKEND_API enum ggml_status ggml_backend_metal_encode_nodes(
        ggml_backend_t backend, struct ggml_cgraph * graph, int i0, int i1, void * cmd_buf);

// Compile the pipeline of every kernel that ggml_backend_metal_encode_nodes would set encoding nodes [i0, i1) of
// `graph`, and encode nothing: the nodes are walked as that encode walks them, with the same fusion and the same
// kernel choices, so the pipelines are the ones it would look up, and each one not yet compiled is compiled. The
// tensors need no memory and no Metal buffer: none is looked up or bound. The device keeps a pipeline until it is
// freed, for every backend of it, so encoding the same nodes afterwards compiles nothing. GGML_OP_CUSTOM nodes are
// skipped, since their implementations set pipelines of their own; fusion statistics do not count the walk.
// Thread-safe with encodes and computes of any backend of the device.
// GGML_STATUS_FAILED, logging why, for a range outside the graph or a backend that is not Metal.
GGML_BACKEND_API enum ggml_status ggml_backend_metal_compile_nodes(
        ggml_backend_t backend, struct ggml_cgraph * graph, int i0, int i1);

// the id<MTLCommandQueue> the device's backends commit to, as void *, not retained: valid while the device is;
// NULL, with a log line, for a device that is not a Metal device
GGML_BACKEND_API void * ggml_backend_metal_device_get_mtl_queue(ggml_backend_dev_t dev);

// the id<MTLBuffer> that holds t's data, as void *, not retained, and in *offs the offset of the data in it; valid
// until the range holding t is replaced (ggml_backend_metal_buffer_set_range) or its buffer is freed; aborts, naming
// t, when t is not in a Metal buffer, its range has no Metal buffer or t ends past the prefix that buffer holds
GGML_BACKEND_API void * ggml_backend_metal_tensor_mtl_buffer(const struct ggml_tensor * t, size_t * offs);

// commit an empty command buffer to the backend's queue and make it the one ggml_backend_synchronize waits for, so a
// synchronize also waits for every command buffer the caller committed to the queue before the call
GGML_BACKEND_API void ggml_backend_metal_commit_fence(ggml_backend_t backend);

//
// Metal implementations of GGML_OP_CUSTOM nodes (ggml_backend_dev_add_custom_op in ggml-backend.h)
//
// On a Metal device, `native_ctx` is the backend's id<MTLComputeCommandEncoder>, as void *. The implementation sets
// its own pipelines, binds the node's tensors with ggml_backend_metal_tensor_mtl_buffer and dispatches. The encoder
// may be a concurrent one, so kernels of the implementation that depend on each other need a memory barrier
// (memoryBarrierWithScope:) between them. The backend sets the pipeline and bindings of each of its own kernels, so
// state the implementation leaves behind does not reach them. The backend encodes with unretained references: the
// caller keeps its pipelines, and any buffers it binds besides the node's tensors, alive until the command buffers
// that use them have completed. The graph reorder may move a CUSTOM node as it moves a stock op.
//
// The device's buffer types, and so the buffers from ggml_backend_dev_buffer_from_host_ptr and
// ggml_backend_metal_buffer_from_ranges, size a CUSTOM node by the implementation's alloc_size, and the encoder, the
// graph reorder and fusion take the node's memory to be all of those bytes: a node that lives in the scratch is
// ordered after the custom node, as it is after a node whose data it overlaps.
//

//
// flash attention partial, for the Metal implementation of a GGML_OP_CUSTOM node
//
// The partial of one KV range of a chunked flash attention, laid out and defined as in ggml-cpu.h
// (ggml_cpu_flash_attn_ext_partial_compute): a caller computes it as a GGML_OP_CUSTOM node of its own and encodes it
// with these from the implementation it registers on a Metal device. `dst` is laid out as there, whatever its op:
// src[0..3] = q, k, v and the mask (NULL: none), src[4] = src[5] = NULL, src[6] = the active count (I32, NULL: none),
// and dst [n_embd_v + 2, n_head, n_batch, ne3] F32. No op param of dst is read: the op parameters are passed in
// ggml_metal_fa_partial_params.
//
// The kernels use scratch past dst's data: dst's buffer holds ggml_nbytes(dst) +
// ggml_metal_flash_attn_ext_partial_extra_bytes(dst) bytes from dst's data, e.g. through a buffer type whose
// get_alloc_size reports them for the node. The scratch is the node's own while it runs; its contents are unspecified
// before and after.
//

struct ggml_metal_fa_partial_params {
    float   scale;         // as in ggml_flash_attn_ext
    float   max_bias;
    float   logit_softcap;
    int32_t active_index;  // read only when dst->src[6] is set
};

// does dev, a Metal device, take dst? what the device answers for a flash attention node of dst's sources, with a
// mask view at an offset that is a multiple of 4 bytes; false, with a log line, for a device that is not a Metal device
GGML_BACKEND_API bool ggml_metal_flash_attn_ext_partial_supported(ggml_backend_dev_t dev, const struct ggml_tensor * dst);

// the bytes past ggml_nbytes(dst) the kernels use: what a Metal buffer type's get_alloc_size adds for a flash
// attention node of dst's sources
GGML_BACKEND_API size_t ggml_metal_flash_attn_ext_partial_extra_bytes(const struct ggml_tensor * dst);

// encode the partial into native_ctx, the id<MTLComputeCommandEncoder> the custom op's encode receives on dev, the
// Metal device the implementation is registered on: the kernels, pipelines and prepasses the backend encodes for a
// flash attention node of dst's sources and these op parameters, with the kernels storing the partial of each row
// instead of the normalized output. The tensors are bound as ggml_backend_metal_tensor_mtl_buffer binds them, the
// scratch at dst's offset + ggml_nbytes(dst); the pipelines are dev's and live as long as it does. Dependent kernels
// of the partial are separated by memory barriers, and the encoder is left open. An inactive partial (the count is
// not greater than active_index) costs its launches only and leaves dst unwritten. False, encoding nothing, when dev
// is not a Metal device, does not take dst, or native_ctx is not an encoder of dev's Metal device.
GGML_BACKEND_API bool ggml_metal_flash_attn_ext_partial_encode(
        ggml_backend_dev_t dev, void * native_ctx, const struct ggml_tensor * dst,
        const struct ggml_metal_fa_partial_params * params);

#ifdef __cplusplus
}
#endif
