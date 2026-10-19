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
// frame pool
//
// A fixed pool of device memory that region buffers borrow tiles from. The pool is one placement
// heap of n_slots tiles; a region buffer is a placement sparse buffer with a fixed address range
// whose tiles are mapped to pool slots and unmapped again, so tensors in a region never move while
// the memory behind them does. A tile is one 16 KiB sparse page.
//
// Work runs on two device timelines, each with values assigned by a host counter in submission
// order: the mapping queue applies map and unmap operations, and the blit queue copies bytes
// between a host-visible staging buffer and region tiles. A timeline call returns the value that
// signals once its work completed, or 0 when it was refused or the pool is in its error state.
//
// The pool checks the queued state of every tile and slot: a map needs an unmapped tile and a free
// slot, an unmap needs a mapped tile, and a blit needs every tile it touches mapped by a map whose
// value it waits for. A refused call logs the reason and queues nothing.
//
// A device error (a failed blit command buffer) puts the pool into a sticky error state: the
// error is logged, waits return false, the failed value never reports complete, and every later
// timeline call returns 0.
//
// Region buffers are GPU-private: host access goes through staging (fill and spill) or through
// ggml_backend_tensor_get/set, which blit through a temporary buffer.
//
// A region does not belong to a pool: it binds to the first pool that maps one of its tiles and stays
// bound until the region or the pool is freed. Freeing a pool unmaps the tiles of every region bound
// to it. A region and the pool it is bound to must not be freed concurrently.
//
// Submissions on each queue are serialized internally, so map and unmap, blits, and teardown may run
// on different threads; timeline values follow submission order.

typedef struct ggml_metal_frame_pool * ggml_metal_frame_pool_t;

// NULL when the device lacks placement sparse support or allocation fails.
GGML_BACKEND_API ggml_metal_frame_pool_t ggml_backend_metal_frame_pool_init(ggml_backend_t backend, size_t n_slots, size_t stage_bytes);
GGML_BACKEND_API void   ggml_backend_metal_frame_pool_free     (ggml_metal_frame_pool_t pool);
GGML_BACKEND_API size_t ggml_backend_metal_frame_pool_tile_bytes(ggml_metal_frame_pool_t pool);
GGML_BACKEND_API void * ggml_backend_metal_frame_pool_stage     (ggml_metal_frame_pool_t pool);

// A buffer of the Metal backend's private kind holding `size` bytes rounded up to tiles, at the
// caller's address range from va_base: tensors in it get addresses in that range, which the caller
// reserves (and keeps reserved) for at least the rounded size so no other allocation can alias it.
// va_base is aligned to a tile. The buffer owns no memory and every tile starts unmapped. NULL when
// the device lacks placement sparse support.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_metal_frame_region_new(ggml_backend_dev_t dev, void * va_base, size_t size);

// n_tiles consecutive tiles of a region from `tile`, bound to consecutive pool slots from `slot`.
struct ggml_metal_tile_run {
    uint32_t tile;
    uint32_t slot;
    uint32_t n_tiles;
};

// All runs go to the mapping queue in one batch and apply in submission order. Unmap ignores `slot`.
GGML_BACKEND_API uint64_t ggml_backend_metal_frame_pool_map  (ggml_metal_frame_pool_t pool, ggml_backend_buffer_t region, const struct ggml_metal_tile_run * runs, size_t n_runs);
GGML_BACKEND_API uint64_t ggml_backend_metal_frame_pool_unmap(ggml_metal_frame_pool_t pool, ggml_backend_buffer_t region, const struct ggml_metal_tile_run * runs, size_t n_runs);

// Blits on the blit queue, in submission order, each after mapping value `after_map` (0: none).
// offs, stage_offs and size are multiples of 4 bytes.
//   fill:  staging[stage_offs, +size) -> region[offs, +size)
//   clear: region[offs, +size) = value
//   spill: region[offs, +size) -> staging[stage_offs, +size), also after wait_ev reaches wait_value
//          when wait_ev is not NULL (a read-done event of the compute queue, say)
GGML_BACKEND_API uint64_t ggml_backend_metal_frame_pool_fill (ggml_metal_frame_pool_t pool, ggml_backend_buffer_t region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map);
GGML_BACKEND_API uint64_t ggml_backend_metal_frame_pool_clear(ggml_metal_frame_pool_t pool, ggml_backend_buffer_t region, size_t offs, size_t size, uint8_t value, uint64_t after_map);
GGML_BACKEND_API uint64_t ggml_backend_metal_frame_pool_spill(ggml_metal_frame_pool_t pool, ggml_backend_buffer_t region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map,
        ggml_metal_event_t wait_ev, uint64_t wait_value);

enum ggml_metal_frame_pool_queue {
    GGML_METAL_FRAME_POOL_MAP,
    GGML_METAL_FRAME_POOL_BLIT,
};

GGML_BACKEND_API uint64_t ggml_backend_metal_frame_pool_completed(ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q);
// False on timeout or once the pool is in its error state.
GGML_BACKEND_API bool     ggml_backend_metal_frame_pool_wait     (ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q, uint64_t value, uint64_t timeout_ms);

// The driver re-prepares an allocation no command buffer referenced for a while before the next
// command buffer that binds it can run. This commits one command buffer that references the heap
// and every region bound to the pool and runs an empty kernel (an encoder that dispatches nothing
// does not count as a reference); calling it more often than that idle threshold keeps them
// prepared.
GGML_BACKEND_API void ggml_backend_metal_frame_pool_keep_prepared(ggml_metal_frame_pool_t pool);

#ifdef __cplusplus
}
#endif
