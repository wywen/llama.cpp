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

// Install a nonblocking, thread-safe pager failure latch before beginning a batch.
// The latch must remain valid until all Metal buffers are drained and this hook is cleared.
GGML_BACKEND_API void ggml_backend_metal_set_boundary_failure_callback(ggml_backend_t backend,
                                                                       void (*latch_failure)(void *),
                                                                       void * user);

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
GGML_BACKEND_API void ggml_backend_metal_clear_encode_window(ggml_backend_t backend);

// Caller-owned reusable projection workspace. Zero-initialize before first use and
// free once at the end of its lifetime. Each call reuses/grows both buffers without
// requiring an intervening free; on failure n_segments is zero and storage is retained.
// Segment metadata is valid until the next projection/free, graph reset, or graph
// lifetime end. Storage may persist across graphs after logical reset. The caller
// must serialize projection, configuration changes and encoding.
struct ggml_metal_segment {
    int  start;
    int  end;
    bool skipped;
    bool wait_at_start;
    bool window_first;
    bool window_last;
    bool ingress_here;
    bool egress_here;
};

struct ggml_metal_projection {
    struct ggml_metal_segment * segments;
    int                         n_segments;
    int                         capacity_segments;
    const struct ggml_tensor ** node_scratch;
    int                         capacity_nodes;
};
GGML_BACKEND_API enum ggml_status ggml_backend_metal_project_graph(ggml_backend_t                 backend,
                                                                   const struct ggml_cgraph *     graph,
                                                                   struct ggml_metal_projection * out);
GGML_BACKEND_API void             ggml_metal_projection_free(struct ggml_metal_projection * projection);

// A receipt covers one actual selected Metal command buffer. Nodes are indexed by
// their optimized scheduler split; globally repeated tensor pointers are refused.
enum ggml_metal_receipt_node_flags {
    GGML_METAL_RECEIPT_SELECTED = 1,
    GGML_METAL_RECEIPT_SKIPPED  = 2,
};

struct ggml_metal_receipt_node {
    uint32_t                   split;
    uint32_t                   ordinal;
    const struct ggml_tensor * node;
    uint32_t                   flags;
};

struct ggml_metal_receipt_cut {
    uint32_t                   split;
    uint32_t                   ordinal;
    const struct ggml_tensor * node;
    ggml_metal_event_t         event;
    uint64_t                   value;
};

enum ggml_metal_terminal_status {
    GGML_METAL_TERMINAL_COMPLETED,
    GGML_METAL_TERMINAL_FAILED,
};

// Hooks run under the completion mutex: no allocation, blocking, throwing or reentry.
// Quiesced runs after finish and every terminal hook for that generation has returned.
typedef void (*ggml_metal_terminal_receipt_fn)(void *                          cookie,
                                               uint64_t                        generation,
                                               uint64_t                        physical_id,
                                               uint32_t                        split,
                                               uint32_t                        first_ordinal,
                                               uint32_t                        end_ordinal,
                                               enum ggml_metal_terminal_status status);
typedef void (*ggml_metal_generation_quiesced_fn)(void * cookie, uint64_t generation);
// Begin after the schedule and window are installed, before any backend enqueue.
// Expected nodes are split-ordered with contiguous ordinals starting at zero per split;
// cuts name its selected producers and must exactly match the installed event schedule.
// Arrays are copied. Tensor/graph metadata and wait-event wrappers remain caller-owned,
// immutable and live through submission and all-buffer drain. Different callback_eval
// partitions are allowed only when each actual projected view matches this table.
// Neither an unbegun paged compute nor an invalid span may enqueue GPU work.
// The engine owns dependency resolution and all success-event publication.
GGML_BACKEND_API enum ggml_status ggml_backend_metal_receipts_begin(ggml_backend_t                         backend,
                                                                    const struct ggml_metal_receipt_node * expected,
                                                                    size_t                                 n_expected,
                                                                    const struct ggml_metal_receipt_cut *  cuts,
                                                                    size_t                                 n_cuts,
                                                                    ggml_metal_terminal_receipt_fn         terminal,
                                                                    ggml_metal_generation_quiesced_fn      quiesced,
                                                                    void *                                 cookie,
                                                                    uint64_t *                             generation);
// Close once after all attempted scheduler calls (including failures), before sync.
// Exact selected-node coverage is required. This does not drain outstanding buffers;
// terminal receipts may arrive before or after finish and in any physical order.
// Retain cookie storage until quiesced, including when a newer generation begins.
// Graph-done additionally requires successful closure and all required terminal work.
GGML_BACKEND_API enum ggml_status ggml_backend_metal_receipts_finish(ggml_backend_t   backend,
                                                                     uint64_t         generation,
                                                                     enum ggml_status submission_status);

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

#ifdef __cplusplus
}
#endif
