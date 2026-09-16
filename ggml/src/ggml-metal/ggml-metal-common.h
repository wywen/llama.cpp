// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

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

#ifdef __cplusplus
}
#endif
