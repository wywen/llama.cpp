// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include "ggml-metal.h"
#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

struct ggml_metal_projection;

struct ggml_metal_projection_config {
    int                          n_cuts;
    struct ggml_tensor * const * cut_nodes;
    int                          n_waits;
    struct ggml_tensor * const * wait_nodes;
    bool                         window_active;
    const struct ggml_tensor *   first_node;
    const struct ggml_tensor *   last_node;
    size_t                       n_out_of_band;
    struct ggml_tensor * const * out_of_band;
};
enum ggml_status ggml_metal_project_segments(const struct ggml_cgraph *                  graph,
                                             const struct ggml_metal_projection_config * config,
                                             struct ggml_metal_projection *              out);

enum ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(int debug);
void ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// is this node a segment boundary marker?
//
// a caller that installs a boundary-event schedule (see ggml_metal_set_boundary_schedule)
//   cuts the graph at the per-layer output tensors, and names those same tensors as the
//   endpoints of its encode window. they are therefore the nodes that nothing may be
//   moved across
//
// the reorder cannot simply be handed that cut list, because it runs before the list
//   exists: ggml_backend_sched optimizes every split while it is splitting the graph, and
//   only afterwards computes the splits - which is where such a caller gets to install
//   its schedule and window. the one property of a marker that is already present at
//   optimize time is its NAME, which is also how the caller resolves its endpoints in the
//   first place, so the name is what this matches. that works per split as well, since a
//   ggml_graph_view shares the tensors of the graph it views
//
// this is the single place the convention is written down - keep the caller in agreement
//   with it
bool ggml_graph_node_is_boundary_marker(const struct ggml_tensor * node);

// reorder the nodes in the graph to improve concurrency, while respecting fusion
//
// with stop_at_boundary_markers, a boundary marker node (see above) becomes a hard
//   barrier: no node is moved across one, in either direction. a caller that restricts
//   encoding to a node window computes that window over the order this function leaves
//   behind, so a node hoisted out of its segment is a value the caller can no longer
//   produce where its consumer expects it. off - the default - the reorder behaves
//   exactly as it always has
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf, bool stop_at_boundary_markers);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_mm   (const struct ggml_tensor * op, bool has_simdgroup_mm);
bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm);

// Terminal receipts and failure wakes are serialized by the tracker mutex.
// Hooks must not allocate, block, throw or reenter; latch precedes wake.
struct ggml_metal_completion;

struct ggml_metal_completion_cut {
    const struct ggml_tensor * node;
    void *                     event;
    uint64_t                   value;
};

typedef void (*ggml_metal_completion_publish)(void * user, void * event, uint64_t value);
typedef void (*ggml_metal_completion_latch)(void * user);
typedef void (*ggml_metal_completion_event_ref)(void * event);

struct ggml_metal_completion * ggml_metal_completion_new(ggml_metal_completion_publish   publish,
                                                         ggml_metal_completion_latch     latch,
                                                         ggml_metal_completion_event_ref retain_event,
                                                         ggml_metal_completion_event_ref release_event,
                                                         void *                          user);
void                           ggml_metal_completion_free(struct ggml_metal_completion * tracker);
// Reserve additional slots without recycling live identities or waiting for callbacks.
// A false return leaves existing in-flight slots usable; the caller latches failure before waking waiters.
bool ggml_metal_completion_reserve_additional(struct ggml_metal_completion * tracker, size_t n);
bool ggml_metal_completion_complete(struct ggml_metal_completion * tracker, uint64_t identity, bool success);
void ggml_metal_completion_fail(struct ggml_metal_completion * tracker);
void ggml_metal_completion_wait(struct ggml_metal_completion * tracker);
bool ggml_metal_completion_failed(struct ggml_metal_completion * tracker);

// Native cut events remain retained through pending callbacks; success is engine-owned.
struct ggml_metal_receipt_binding_cut {
    uint32_t                         split;
    uint32_t                         ordinal;
    struct ggml_metal_completion_cut native;
};
enum ggml_status ggml_metal_receipts_begin(struct ggml_metal_completion *                tracker,
                                           const struct ggml_metal_receipt_node *        expected,
                                           size_t                                        n_expected,
                                           const struct ggml_metal_receipt_binding_cut * cuts,
                                           size_t                                        n_cuts,
                                           ggml_metal_terminal_receipt_fn                terminal,
                                           ggml_metal_generation_quiesced_fn             quiesced,
                                           void *                                        cookie,
                                           uint64_t *                                    generation);
enum ggml_status ggml_metal_receipts_finish(struct ggml_metal_completion * tracker,
                                            uint64_t                       generation,
                                            enum ggml_status               submission_status);
// Validates every view node and its selected/skipped projection classification before enqueue.
// Bounded view data is borrowed only for this synchronous call.
bool             ggml_metal_receipts_validate_view(struct ggml_metal_completion *       tracker,
                                                   const struct ggml_cgraph *           graph,
                                                   const struct ggml_metal_projection * projection,
                                                   uint32_t *                           split);
bool             ggml_metal_receipts_register(struct ggml_metal_completion * tracker,
                                              uint32_t                       split,
                                              struct ggml_tensor * const *   nodes,
                                              int                            start,
                                              int                            end,
                                              uint64_t *                     physical_id);
bool             ggml_metal_receipts_active(struct ggml_metal_completion * tracker);

// Diagnostics for the reusable expected-node slab; borrowed until the next begin.
const void * ggml_metal_receipts_node_storage(struct ggml_metal_completion * tracker, size_t * capacity);

// Only the submitting thread reserves/reads the pointer table. Each callback writes its
// status before the terminal hook; command buffers stay retained through synchronize.
struct ggml_metal_seg_result {
    int    status;
    void * buffer;
};
struct ggml_metal_result_storage;
struct ggml_metal_result_storage * ggml_metal_result_storage_new(void);
void                               ggml_metal_result_storage_free(struct ggml_metal_result_storage * storage);
bool                           ggml_metal_result_storage_reserve(struct ggml_metal_result_storage * storage, size_t n);
struct ggml_metal_seg_result * ggml_metal_result_storage_at(struct ggml_metal_result_storage * storage, size_t i);

#ifdef __cplusplus
}
#endif
