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

typedef struct ggml_metal_event * ggml_metal_event_t;

struct ggml_metal_tensor_copy_pair {
    struct ggml_tensor * src;
    struct ggml_tensor * dst;
};

// device-side copy of `size` bytes from byte `src_offs` of src's data to byte `dst_offs` of dst's data
// unlike a view, a range needs no tensor of its own, so it also works on buffer types whose views cannot be
// resolved to a device buffer
struct ggml_metal_tensor_range_copy {
    struct ggml_tensor * src;
    struct ggml_tensor * dst;
    size_t               src_offs;
    size_t               dst_offs;
    size_t               size;
};

//
// events
//

GGML_BACKEND_API ggml_metal_event_t ggml_backend_metal_event_init(ggml_backend_t backend);
GGML_BACKEND_API void               ggml_backend_metal_event_free(ggml_backend_t backend, ggml_metal_event_t ev);

GGML_BACKEND_API void     ggml_backend_metal_event_host_signal(ggml_metal_event_t ev, uint64_t value);
GGML_BACKEND_API bool     ggml_backend_metal_event_host_wait  (ggml_metal_event_t ev, uint64_t value, uint64_t timeout_ms);
GGML_BACKEND_API uint64_t ggml_backend_metal_event_host_value (ggml_metal_event_t ev);

// set `ev` to `value` once everything committed on this backend's queue before the call has completed
GGML_BACKEND_API void ggml_backend_metal_event_signal(ggml_backend_t backend, ggml_metal_event_t ev, uint64_t value);

// the command buffer error that put the backend into its error state, or NULL if there is none
// lets a caller tell a GPU timeout from a fault or an abandoned queue; valid until the backend computes again or is freed
GGML_BACKEND_API const char * ggml_backend_metal_last_error(ggml_backend_t backend);

//
// boundary-scheduled compute
//
// While a boundary schedule or host cuts are set, a graph compute is encoded sequentially on the calling thread,
// one command buffer per segment. Segments end after each cut node and start at each wait node. The settings
// persist across computes and are matched by node pointer against each computed graph (one split of a graph
// computes only the nodes it contains), until the caller replaces or clears them.
//

// install the boundary schedule: after cut_nodes[i] the segment's command buffer sets sig_ev[i] to sig_val[i] on
// completion (NULL event: no signal); before wait_nodes[i] the host waits for wait_ev[i] to reach wait_val[i], then
// the command buffer encodes the same wait. The arrays are copied. Each call clears the entry signal and leaves the
// host cuts and the encode window in place; n_cuts == 0 clears the schedule.
GGML_BACKEND_API void ggml_backend_metal_set_boundary_schedule(
        ggml_backend_t backend,
        int n_cuts,  struct ggml_tensor * const * cut_nodes,  ggml_metal_event_t * sig_ev,  const uint64_t * sig_val,
        int n_waits, struct ggml_tensor * const * wait_nodes, ggml_metal_event_t * wait_ev, const uint64_t * wait_val);

// set `ev` to `value` at the encode window's entry, from an empty command buffer committed ahead of the window's
// first segment: it completes after everything enqueued before the window and depends on none of the window's waits
// cleared by every ggml_backend_metal_set_boundary_schedule call, so set it after the schedule; NULL clears it
GGML_BACKEND_API void ggml_backend_metal_set_boundary_entry_signal(
        ggml_backend_t backend, ggml_metal_event_t ev, uint64_t value);

// how long a segment may wait for its gate before the graph fails with the backend in its error state
// (default: 300000 ms, 0: no limit); a warning is logged every 10 s of waiting
// a gate event that reads UINT64_MAX while a lower value is awaited means the producer abandoned the schedule:
// the graph fails the same way instead of running the segment on data that was never delivered
GGML_BACKEND_API void ggml_backend_metal_set_paged_gate_abort_ms(ggml_backend_t backend, uint32_t ms);

// Host cuts: run host code between two segments, after the first completed and before the second is committed,
// for a caller that must read a node's result to decide what the rest of the graph reads.
//
// Each node in `nodes` ends a segment. Its command buffer signals `ev` from the device; the host spins on the value
// and then calls fn(user_data, i, node), `i` being the node's index in `nodes`, on the thread computing the graph.
// fn may commit work to this backend's queue and wait for it, but must not compute on the backend or change its
// schedule, host cuts or encode window. A next segment without a gate wait or an entry signal is encoded before fn
// runs, so fn may change the contents of the buffers it binds but not which buffers they are.
//
// The first cut signals base_value + 1 and each later one the next value, across computes, until the list is
// replaced. The graph fails, with the backend in its error state, when a segment does not complete within the gate
// abort limit, when `ev` reads UINT64_MAX, or when fn returns false.
//
// A cut ends a fusion group like any segment boundary. A cut in a segment the encode window skips never fires.
// The node array is copied; each call replaces the list, n == 0 clears it.
typedef bool (*ggml_backend_metal_host_cut_fn)(void * user_data, int i, struct ggml_tensor * node);

GGML_BACKEND_API void ggml_backend_metal_set_host_cuts(
        ggml_backend_t backend,
        int n, struct ggml_tensor * const * nodes,
        ggml_metal_event_t ev, uint64_t base_value,
        ggml_backend_metal_host_cut_fn fn, void * user_data);

// Encode window: restrict the boundary-scheduled path to the nodes (first_node, last_node] (NULL for an open edge),
// with device-side copies on entry (in the split that contains first_node) and exit (in the split that contains
// last_node). A split that contains neither boundary is encoded in full. Each pair copies a whole tensor; both
// tensors must have the same size. The arrays are copied.
//
// `out_of_band` names nodes that lie outside the window. A split that would encode one of them fails with
// GGML_STATUS_FAILED before anything is enqueued: it catches a split that contains neither boundary but lies inside
// the excluded range, which the per-split rule above would encode in full. n_out_of_band == 0 disables the check.
GGML_BACKEND_API void ggml_backend_metal_set_encode_window(
        ggml_backend_t backend,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     const struct ggml_metal_tensor_copy_pair * ingress,
        size_t n_egress,      const struct ggml_metal_tensor_copy_pair * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band);

// the same encode window with byte-range copies, e.g. to save and restore a part of a cache tensor
// each range must lie within both tensors, have size > 0 and offsets and size that are multiples of 4 (a Metal blit
// requirement), and must not overlap its destination when src and dst share storage
// copies in a list run in order, so a range may read what an earlier one wrote
// returns false, logging the first violation and keeping the previous window, when a range is invalid
GGML_BACKEND_API bool ggml_backend_metal_set_encode_window_ranged(
        ggml_backend_t backend,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     const struct ggml_metal_tensor_range_copy * ingress,
        size_t n_egress,      const struct ggml_metal_tensor_range_copy * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band);
GGML_BACKEND_API void ggml_backend_metal_clear_encode_window(ggml_backend_t backend);

// Reorder barriers: the graph reorder runs while the scheduler splits the graph, before a caller can install a
// schedule, and may move a node across a cut it does not depend on. A barrier node keeps its place: no node moves
// across it in either direction, and a fusion group that contains it stays where it is.

// make the per-layer output tensors (named "l_out-<il>") barriers (default: off)
GGML_BACKEND_API void ggml_backend_metal_set_reorder_barriers(ggml_backend_t backend, bool enable);

// make every node whose name starts with one of `prefixes` a barrier; the strings are copied, n == 0 clears the list
// applies to graphs reordered after the call, so register every prefix the caller may cut at before allocating the graph
GGML_BACKEND_API void ggml_backend_metal_set_reorder_barrier_prefixes(
        ggml_backend_t backend, int n, const char * const * prefixes);

//
// buffers over host memory, released and recreated
//
// The Metal buffers of a buffer from buffer_from_host_ptr capture the physical pages of the caller's memory: pages
// the caller replaces under them (with mmap MAP_FIXED, say) are not seen by the GPU. These calls release the Metal
// buffers so that the pages can be replaced, and recreate them over the same memory. The backend buffer, its base
// and its tensors are unchanged.
//
// The first of these calls on a buffer hands its residency set to the caller: from then on the set is requested
// only by ggml_backend_metal_buffer_request_residency. Calls on one buffer are serialized and may run on any thread.
//

// release the Metal buffers; the caller ends the residency first and ensures no command buffer in flight binds the
// buffer. Encoding a kernel that uses the buffer aborts until rewrap; host access keeps working
GGML_BACKEND_API bool ggml_backend_metal_buffer_unwrap            (ggml_backend_buffer_t buffer);
// recreate the Metal buffers over the same ranges; pages never written read as zeros
GGML_BACKEND_API bool ggml_backend_metal_buffer_rewrap            (ggml_backend_buffer_t buffer);
// ask Metal to keep the buffer resident (may be repeated), or end that request
// false when the buffer is not over host memory or the OS has no residency sets; request_residency also when unwrapped
GGML_BACKEND_API bool ggml_backend_metal_buffer_request_residency (ggml_backend_buffer_t buffer);
GGML_BACKEND_API bool ggml_backend_metal_buffer_end_residency     (ggml_backend_buffer_t buffer);

// a buffer over host memory like one from ggml_backend_dev_buffer_from_host_ptr, but wrapped in disjoint Metal buffers
// split at split_offsets, for a caller that binds sub-ranges of one host allocation independently: a kernel that uses a
// tensor inside one sub-range wires only that sub-range's pages. A tensor or view that crosses a split point is in no
// Metal buffer, and encoding a kernel that uses it aborts. Unwrap, rewrap and the residency set calls work as above on
// every sub-range at once, and per sub-range as below. ptr must be page aligned; the offsets must be strictly
// increasing, page aligned and inside (0, size); there are at most 63 of them and no sub-range may be larger than the
// device's max buffer size. NULL on bad input
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_metal_buffer_from_host_ptr_split(
        ggml_backend_dev_t dev, void * ptr, size_t size, const size_t * split_offsets, int n_splits);

//
// sub-ranges of a split buffer, released and recreated one at a time
//
// For a caller that holds and replaces sub-ranges of one host allocation independently. A buffer from
// ggml_backend_metal_buffer_from_host_ptr_split has one residency set per sub-range, holding that sub-range's Metal
// buffer alone, and each sub-range is wrapped or unwrapped on its own. Sub-range i, 0 <= i <= n_splits, starts at
// split_offsets[i - 1] (0 for i == 0). The calls below act on sub-range i as the whole-buffer calls above act on a
// buffer. On a split buffer the whole-buffer calls act on every sub-range; they change nothing and return false when
// one sub-range refuses: unwrap when a sub-range is already unwrapped, rewrap when one is wrapped, request_residency
// when one is unwrapped.
//
// Handoff: the first call, per sub-range or whole-buffer, that names a sub-range hands that sub-range's residency set
// to the caller; a whole-buffer call names every sub-range. From then on the backend neither requests nor keeps alive
// that set, so the sub-range is resident only while the caller requests it. The sets of sub-ranges no call has named
// yet stay with the backend. A refused call hands nothing over.
//
// Calls on one buffer, per sub-range or whole-buffer, are serialized and may run on any thread; a call on one
// sub-range never changes another. Before unwrapping sub-range i the caller ends its residency and ensures no command
// buffer in flight binds sub-range i; work that binds other sub-ranges may be in flight, and may be encoded while
// sub-range i is unwrapped. Encoding a kernel that uses a tensor in an unwrapped sub-range aborts, naming the tensor.
//
// Each returns false, with a log line, when the buffer is not from ggml_backend_metal_buffer_from_host_ptr_split or i
// is not one of its sub-ranges, and otherwise when its whole-buffer counterpart would for that sub-range
//
GGML_BACKEND_API bool ggml_backend_metal_buffer_unwrap_range            (ggml_backend_buffer_t buffer, int i);
GGML_BACKEND_API bool ggml_backend_metal_buffer_rewrap_range            (ggml_backend_buffer_t buffer, int i);
GGML_BACKEND_API bool ggml_backend_metal_buffer_request_residency_range (ggml_backend_buffer_t buffer, int i);
GGML_BACKEND_API bool ggml_backend_metal_buffer_end_residency_range     (ggml_backend_buffer_t buffer, int i);

// buffer type for compute buffers that the caller places in memory it manages, for ggml_backend_sched_new
// alloc_buffer must return a buffer whose tensors resolve to Metal buffers, e.g. one from
// ggml_backend_dev_buffer_from_host_ptr on the Metal device; max_size bounds each buffer the scheduler requests
// alignment and allocation sizes follow the mapped Metal buffer type; the buffer type must outlive its buffers
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_metal_compute_buft_new(
        ggml_backend_dev_t dev,
        const char * name,
        ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size, void * user_data),
        size_t max_size,
        void * user_data);
GGML_BACKEND_API void   ggml_backend_metal_compute_buft_free(ggml_backend_buffer_type_t buft);

#ifdef __cplusplus
}
#endif
