// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>

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

// reorder the nodes in the graph to improve concurrency, while respecting the fusions of a device with props
//
// with stop_at_boundary_markers, a boundary marker node (see above) becomes a hard
//   barrier: no node is moved across one, in either direction. a caller that restricts
//   encoding to a node window computes that window over the order this function leaves
//   behind, so a node hoisted out of its segment is a value the caller can no longer
//   produce where its consumer expects it. off - the default - the reorder behaves
//   exactly as it always has
//
// a node whose name starts with one of the n_barrier_prefixes barrier_prefixes is the same
//   kind of barrier, whether or not stop_at_boundary_markers is set (see
//   ggml_backend_metal_set_reorder_barrier_prefixes). with no prefixes and
//   stop_at_boundary_markers off, the reorder behaves exactly as it always has
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf, bool stop_at_boundary_markers, int n_barrier_prefixes, const char * const * barrier_prefixes);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_fwht (const struct ggml_tensor * op, size_t max_tg_mem);
bool ggml_metal_op_mul_mat_use_mm   (const struct ggml_tensor * op, bool has_simdgroup_mm);
bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm);

bool ggml_metal_op_mul_mat_use_mma    (const struct ggml_tensor * op);
bool ggml_metal_op_mul_mat_may_use_mma(const struct ggml_tensor * op); // graph structure only

// the few-row MMA kernel for a src0 type and rt src1 tiles: per 32-weight block (q4_0, q8_0 with one tile), q5_K, or the generic 64-weight chunk kernel
enum ggml_metal_mma_kind { GGML_METAL_MMA_KIND_BLK, GGML_METAL_MMA_KIND_Q5_K, GGML_METAL_MMA_KIND_GEN };
enum ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(enum ggml_type type, int rt);

// the src1 tiles of the few-row MMA kernels for mat-mul op: one 8-row tile, or two above 8 rows
int ggml_metal_mul_mv_mma_rt(const struct ggml_tensor * op);

// the weights of K per simdgroup step of the few-row MMA kernel for a src0 type and rt src1 tiles, 0 if none takes the type
int64_t ggml_metal_mul_mv_mma_k_step(enum ggml_type type, int rt);

#ifdef __cplusplus
}
#endif
