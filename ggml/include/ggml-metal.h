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
// placement sparse memory
//
// Primitives for paging device memory under fixed tensor addresses. A placement heap is device
// memory in 16 KiB tiles; a sparse buffer is an address range whose tiles map onto heap tiles and
// unmap again, so tensors in it never move while the memory behind them does. Each call does one
// Metal operation and keeps no record of what is mapped where: the caller owns that bookkeeping.
//
// Metal does not check what these calls leave to the caller. A blit into an unmapped tile
// completes without an error: it is dropped, or, on a queue that wrote the tile before its unmap,
// it lands in the heap tile the tile last mapped onto. A read of an unmapped tile returns zeros.
//
// Placement sparse memory needs macOS 26.4 and a device that supports it; the constructors return
// NULL otherwise.

typedef struct ggml_metal_placement_heap * ggml_metal_placement_heap_t;

// Whether the backend's device and OS support placement sparse memory with 16 KiB tiles; false for
// a backend that is not Metal.
GGML_BACKEND_API bool ggml_backend_metal_placement_supported(ggml_backend_t backend);

// A heap of n_tiles tiles of private storage, resident for its life through its own residency set,
// which is attached to the backend's compute queue. NULL for a backend that is not Metal.
GGML_BACKEND_API ggml_metal_placement_heap_t ggml_backend_metal_placement_heap_new (ggml_backend_t backend, size_t n_tiles);
// The heap must have no tile mapped by any sparse buffer, no work in flight that touches it, and no
// blit queue still attached to it; nothing here tracks any of these, so the caller drains, unmaps
// and frees those queues first.
GGML_BACKEND_API void                        ggml_backend_metal_placement_heap_free(ggml_metal_placement_heap_t heap);

// The bytes of one tile of a heap or a sparse buffer.
GGML_BACKEND_API size_t ggml_backend_metal_placement_tile_bytes(void);

// A buffer of the backend's private kind over a placement sparse buffer of `size` bytes, rounded
// up to tiles, at the caller's address range from va_base: tensors in it get addresses in that
// range, which the caller reserves for at least the rounded size and keeps reserved until the
// buffer is freed. va_base is tile aligned. The buffer owns no memory and every tile starts
// unmapped; freeing it drops its mappings. Metal tracks no hazards on it: the caller orders every
// access with events. NULL for a backend that is not Metal.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_metal_sparse_buffer_new(ggml_backend_t backend, void * va_base, size_t size);

// n_tiles consecutive tiles of a sparse buffer from `tile`, and the consecutive heap tiles from
// `heap_tile` they map onto.
struct ggml_metal_tile_run {
    uint32_t tile;
    uint32_t heap_tile;
    uint32_t n_tiles;
};

// A Metal 4 command queue for mapping operations, which apply in submission order. Calls on one
// queue from several threads are each atomic; their order is the caller's.
typedef struct ggml_metal_mapping_queue * ggml_metal_mapping_queue_t;

GGML_BACKEND_API ggml_metal_mapping_queue_t ggml_backend_metal_mapping_queue_new (ggml_backend_t backend);
GGML_BACKEND_API void                       ggml_backend_metal_mapping_queue_free(ggml_metal_mapping_queue_t queue);

// Maps the runs of `buffer`'s tiles onto `heap`'s tiles, all in one mapping call, then signals
// `signal` to `value` on the queue once they applied (no signal when `signal` is NULL). False,
// with nothing queued, when a run is empty or outside the buffer or the heap, or the buffer is not
// a sparse buffer.
GGML_BACKEND_API bool ggml_backend_metal_map_tiles(ggml_metal_mapping_queue_t queue, ggml_metal_placement_heap_t heap, ggml_backend_buffer_t buffer,
        const struct ggml_metal_tile_run * runs, size_t n_runs, ggml_metal_event_t signal, uint64_t value);

// As map_tiles, unmapping the runs; heap_tile is ignored.
GGML_BACKEND_API bool ggml_backend_metal_unmap_tiles(ggml_metal_mapping_queue_t queue, ggml_backend_buffer_t buffer,
        const struct ggml_metal_tile_run * runs, size_t n_runs, ggml_metal_event_t signal, uint64_t value);

// A command queue for blits with one event as its timeline: the queue numbers its blits 1, 2, ...
// in submission order, and blit v signals the event to v once it completed. Each blit waits for
// the one before it, so blits complete as an in-order prefix. A blit whose command buffer fails
// does not signal and stops the timeline at the value before it: no later blit runs or signals,
// and the queue latches the failure's error. A caller checks the error rather than waiting on blit
// values without a bound once it is set; to resume, it signals the event past the failed value from
// the host, or it abandons the queue. Freeing the queue waits a bounded time for blits in flight;
// blits parked behind a failure stay parked. Submissions on one queue are serialized internally.
//
// `heap`, when not NULL, is the heap whose tiles the blits touch: its residency set is attached to
// the queue, and it must outlive the queue.
typedef struct ggml_metal_blit_queue * ggml_metal_blit_queue_t;

struct ggml_metal_wait {
    ggml_metal_event_t event;
    uint64_t           value;
};

GGML_BACKEND_API ggml_metal_blit_queue_t ggml_backend_metal_blit_queue_new  (ggml_backend_t backend, ggml_metal_placement_heap_t heap);
GGML_BACKEND_API void                    ggml_backend_metal_blit_queue_free (ggml_metal_blit_queue_t queue);
// The first failed command buffer's error, or NULL while none failed. Valid until the queue is freed.
GGML_BACKEND_API const char *            ggml_backend_metal_blit_queue_error(ggml_metal_blit_queue_t queue);
// The queue's timeline, owned by the queue; other queues may wait on it.
GGML_BACKEND_API ggml_metal_event_t      ggml_backend_metal_blit_queue_event(ggml_metal_blit_queue_t queue);

// Byte ranges are relative to each buffer's base and lie in one Metal buffer of it; any Metal
// backend buffer works (a shared staging buffer, a sparse buffer). Offsets and sizes are multiples
// of 4. Each returns the blit's value on the queue's timeline, or 0, with nothing committed, for an
// invalid range or a buffer of another backend.
GGML_BACKEND_API uint64_t ggml_backend_metal_blit_copy(ggml_metal_blit_queue_t queue,
        ggml_backend_buffer_t src, size_t src_offs, ggml_backend_buffer_t dst, size_t dst_offs, size_t size,
        const struct ggml_metal_wait * waits, size_t n_waits);
GGML_BACKEND_API uint64_t ggml_backend_metal_blit_fill(ggml_metal_blit_queue_t queue,
        ggml_backend_buffer_t dst, size_t offs, size_t size, uint8_t byte,
        const struct ggml_metal_wait * waits, size_t n_waits);

#ifdef __cplusplus
}
#endif
