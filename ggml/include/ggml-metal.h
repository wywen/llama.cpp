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
// with n_cb > 0 the graph is split into a head encoded by the calling thread and n_cb slices
// encoded concurrently; kernels are fused only within one command buffer, so these split
// points can break a fusable group. with n_cb == 0 the whole graph is encoded in one command
// buffer, which fuses every graph the same way however it is sized
// obtained through ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_set_n_cb")
typedef void (*ggml_backend_metal_set_n_cb_t)(ggml_backend_t backend, int n_cb);

typedef struct ggml_metal_event * ggml_metal_event_t;

struct ggml_metal_tensor_copy_pair {
    struct ggml_tensor * src;
    struct ggml_tensor * dst;
};

// A device-side copy of `size` bytes from byte `src_offs` of src's data to byte `dst_offs` of
// dst's data. Offsets are relative to each tensor's own data, not to its backing buffer. Unlike
// a ggml view, a range needs no tensor of its own, so it also works on buffer types whose views
// cannot be resolved to a device buffer.
struct ggml_metal_tensor_range_copy {
    struct ggml_tensor * src;
    struct ggml_tensor * dst;
    size_t               src_offs;
    size_t               dst_offs;
    size_t               size;
};

GGML_BACKEND_API ggml_metal_event_t ggml_backend_metal_event_init(ggml_backend_t backend);
GGML_BACKEND_API void               ggml_backend_metal_event_free(ggml_backend_t backend, ggml_metal_event_t ev);

GGML_BACKEND_API void     ggml_backend_metal_event_host_signal(ggml_metal_event_t ev, uint64_t value);
GGML_BACKEND_API bool     ggml_backend_metal_event_host_wait  (ggml_metal_event_t ev, uint64_t value, uint64_t timeout_ms);
GGML_BACKEND_API uint64_t ggml_backend_metal_event_host_value (ggml_metal_event_t ev);

// Set `ev` to `value` once everything committed on this backend's queue before the call has completed, e.g. the async tensor copies of the last graph.
GGML_BACKEND_API void ggml_backend_metal_event_signal(ggml_backend_t backend, ggml_metal_event_t ev, uint64_t value);

// The command buffer error that put this backend into its error state, or NULL if it has not
// entered one. Valid until the backend computes again or is freed.
//
// A caller that only sees a failed compute cannot tell a GPU timeout from an allocation fault
// from an abandoned queue, and those want different responses. This is where the distinction is
// written down.
GGML_BACKEND_API const char * ggml_backend_metal_last_error(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_boundary_schedule(
        ggml_backend_t backend,
        int n_cuts,  struct ggml_tensor * const * cut_nodes,  ggml_metal_event_t * sig_ev,  const uint64_t * sig_val,
        int n_waits, struct ggml_tensor * const * wait_nodes, ggml_metal_event_t * wait_ev, const uint64_t * wait_val);

// Signal `ev` to `value` at the encode window's entry: in the split where the window's first_node
// matches (where its ingress copies fire), the paged path commits a command buffer holding no
// ops and no waits ahead of the window's first segment, and sets the event to `value` when that
// buffer completes. The value is set after everything enqueued before the window, every earlier
// split of the graph included, and never waits on the window's own schedule waits. Persistent
// like the boundary schedule and cleared with it by every set_boundary_schedule call; set it
// after the schedule it belongs to. A NULL `ev` clears it.
GGML_BACKEND_API void ggml_backend_metal_set_boundary_entry_signal(
        ggml_backend_t backend, ggml_metal_event_t ev, uint64_t value);

// Set how long, in milliseconds, the paged compute path waits for one segment's gate event before
// failing the graph with the backend error state set. While waiting it logs a warning every 10
// seconds naming the node, the elapsed time and the awaited event value. 0 waits without limit.
// Default 300000. Applies to subsequent computes on this backend.
//
// A wait event signaled to UINT64_MAX means the producer abandoned the schedule: Metal waits are
// satisfied by any signaled value >= the awaited one, so a saturated event would otherwise let the
// segment run on data that was never delivered. When a segment's wait event reads UINT64_MAX and the
// awaited value is lower, the paged path fails the graph instead of encoding the segment: the
// backend enters its error state and ggml_backend_metal_last_error names the node.
GGML_BACKEND_API void ggml_backend_metal_set_paged_gate_abort_ms(ggml_backend_t backend, uint32_t ms);

// Restrict the boundary-scheduled (paged) compute path to the node range
// (first_node, last_node] -- lower boundary EXCLUSIVE (pass the band-entry
// residual node itself), upper INCLUSIVE, either NULL for an open edge --
// with ordered device-side tensor copies at window entry (fired in the split
// where first_node matched) and exit (fired where last_node matched). The pair
// arrays are copied and owned by the Metal context until replaced or cleared.
// Each pair must contain non-NULL tensors with equal byte sizes.
// Persistent and pointer-keyed like the boundary schedule, projected per split:
// a split containing neither non-NULL boundary encodes in full. Only meaningful
// while a boundary schedule is set.
//
// `out_of_band` is a caller-owned array of full-graph nodes the caller asserts
// lie OUTSIDE the active band; it is copied and owned by the Metal context
// alongside the pair arrays. It enforces, and only enforces, the invariant:
//
//   no submitted split may ever encode a node that lies outside the active band
//
// which the per-split projection above cannot uphold on its own -- "contains
// neither boundary, so encode in full" is right for the input-processing splits
// but wrong for a trunk split landing after the boundary split, and the window
// cannot distinguish them. A split that would encode any named node is refused:
// graph_compute returns GGML_STATUS_FAILED with nothing enqueued and no event
// signalled, so the caller can fail the prefill and recover. It never changes
// which nodes are selected for encoding. Pass n_out_of_band == 0 to disable the
// check entirely -- behaviour is then exactly as before this parameter existed.
GGML_BACKEND_API void ggml_backend_metal_set_encode_window(
        ggml_backend_t backend,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     const struct ggml_metal_tensor_copy_pair * ingress,
        size_t n_egress,      const struct ggml_metal_tensor_copy_pair * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band);

// The same encode window, with byte-range copies in place of whole-tensor pairs. A caller that
// saves and restores a sub-range of a cache tensor around a windowed graph names that sub-range
// here; several ranges may name the same tensors.
//
// Every range must satisfy, or the call fails:
//   - src and dst are non-NULL and size > 0;
//   - src_offs + size <= ggml_nbytes(src) and dst_offs + size <= ggml_nbytes(dst);
//   - src_offs, dst_offs and size are multiples of 4 bytes (the macOS requirement of
//     MTLBlitCommandEncoder's buffer-to-buffer copy);
//   - when src and dst share storage (the same tensor, or views of one tensor), the source and
//     destination byte ranges do not overlap, since Metal leaves an overlapping copy within one
//     buffer undefined.
// Each list must be non-NULL when its count is non-zero.
//
// Copies in each list run in list order: each is its own blit pass, and a later pass that touches
// bytes an earlier pass wrote sees that write. A range may therefore read what an earlier range
// in the same list produced.
//
// Returns false, logs the first violation, and leaves any previously installed window in place
// when the arguments are invalid. Returns true once the window is installed. The whole-tensor
// call above is equivalent to this one with src_offs = dst_offs = 0 and size = ggml_nbytes(src),
// except that it applies no alignment rule.
GGML_BACKEND_API bool ggml_backend_metal_set_encode_window_ranged(
        ggml_backend_t backend,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     const struct ggml_metal_tensor_range_copy * ingress,
        size_t n_egress,      const struct ggml_metal_tensor_range_copy * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band);
GGML_BACKEND_API void ggml_backend_metal_clear_encode_window(ggml_backend_t backend);

// Make the Metal graph reorder stop at boundary marker nodes -- the per-layer output
// tensors a boundary-scheduling caller cuts the graph at. Off by default; a caller that
// installs a boundary schedule and an encode window MUST turn it on.
//
// ggml_backend_sched optimizes each split while it is splitting the graph, and only
// afterwards computes the splits -- which is where the schedule and the window are
// installed. So the reorder always runs before any schedule exists and cannot see the
// cuts, while the window is computed over the order the reorder has already produced.
// Left to itself the reorder hoists memory-concurrent nodes up to 64 positions forward
// across anything it considers safe, and a node anchored by nothing (a lookup into a
// persistent cache, say) can land in an earlier segment than the one that is supposed to
// produce it. The caller then sees a value produced outside its window and has to refuse
// the encode, since re-running the producer inside the window is not sound when that same
// segment also writes the cache.
//
// With this set, a marker node is a hard barrier: no node moves across one in either
// direction, so segment membership is the same before and after the reorder. Nodes still
// reorder freely within a segment. Off, the reorder is byte-for-byte what it always was.
GGML_BACKEND_API void ggml_backend_metal_set_reorder_barriers(ggml_backend_t backend, bool enable);

//
// buffers over host memory, released and recreated
//
// A buffer from the device's buffer_from_host_ptr wraps the caller's memory in no-copy Metal
// buffers. While one of them exists, the GPU reads and writes the physical pages it captured:
// pages the caller replaces under it (an eviction by mmap MAP_FIXED, say) are pages the GPU never
// sees. These calls let the caller release the Metal buffers so that it may replace the pages, and
// recreate them over the same memory afterwards. The backend buffer, its base and the tensors in it
// are unchanged throughout; kernels encoded after a rewrap bind the new Metal buffers.
//
// The first of these calls on a buffer hands its residency set to the caller: the set leaves the
// device's keep-alive collection, its residency ends, and from then on it is requested only by
// request_residency. A buffer whose device keeps no residency sets gets one of its own.
//
// unwrap releases every Metal buffer of the buffer. The caller ends its residency first and makes
// sure no command buffer that binds it is still encoding or in flight; until rewrap, encoding a
// kernel that uses a tensor in the buffer aborts. Host reads and writes of the buffer's tensors
// keep working, since they go to the memory directly. False, with nothing changed, when the buffer
// is not a Metal buffer over host memory or is already unwrapped.
//
// rewrap makes new Metal buffers over the same ranges as before and puts them in the buffer's
// residency set. The memory may be partly or wholly unpopulated: a page the host writes after the
// rewrap is what the GPU reads, and a page never written reads zeros. False, with the buffer still
// unwrapped, when the buffer is not unwrapped or the device refuses a Metal buffer.
//
// request_residency asks Metal to keep the buffer's memory resident, and may be repeated: a set
// requested again after its memory went idle is prepared ahead of the next command buffer instead
// of during it. end_residency ends that request. Both return false when the buffer is not a Metal
// buffer over host memory or the OS has no residency sets; request_residency also when the buffer
// is unwrapped. The set is attached to no queue.
//
// Calls on one buffer are serialized internally and may run on any thread.
GGML_BACKEND_API bool ggml_backend_metal_buffer_unwrap            (ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool ggml_backend_metal_buffer_rewrap            (ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool ggml_backend_metal_buffer_request_residency (ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool ggml_backend_metal_buffer_end_residency     (ggml_backend_buffer_t buffer);

// Buffer type for compute buffers that the caller places in memory it manages. It can be passed to
// ggml_backend_sched_new for the Metal backend of dev. alloc_buffer receives the requested size and
// must return a buffer whose tensors resolve to Metal buffers, for example one created with
// ggml_backend_dev_buffer_from_host_ptr on the Metal device. Alignment and per-tensor allocation sizes
// follow the Metal mapped buffer type; max_size bounds each buffer the scheduler requests. The buffer
// type must outlive every scheduler and buffer that uses it.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_metal_compute_buft_new(
        ggml_backend_dev_t dev,
        const char * name,
        ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size, void * user_data),
        size_t max_size,
        void * user_data);
GGML_BACKEND_API void   ggml_backend_metal_compute_buft_free(ggml_backend_buffer_type_t buft);
GGML_BACKEND_API void * ggml_backend_metal_compute_buft_user_data(ggml_backend_buffer_type_t buft);

#ifdef __cplusplus
}
#endif
