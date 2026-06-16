#pragma once

// Graph-rewrite: identity CPU barrier ops at layer boundaries.
//
// rrl_insert_layer_barriers inserts a GGML_OP_CUSTOM identity op after every
// `segment`-th l_out-<il> tensor in the just-built (not-yet-allocated) graph `gf`.
// This forces the backend scheduler to split Metal/CPU execution at those points,
// enabling per-segment residency management in later increments.
//
// The rewrite is numerically transparent: the custom callback is a plain memcpy that
// produces a byte-identical copy of its input.  Graph reuse (can_reuse) works normally
// because the barrier ops are inserted into gf before alloc, so the reserved graph and
// the executed graph have the same topology.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cgraph;
struct ggml_context;

/// Inserts identity CPU custom ops after every `segment`-th layer's `l_out-{il}` tensor,
/// forcing a Metal/CPU backend split there.  Returns the number of barriers inserted
/// (0 = no-op).  `gf` is the just-built, not-yet-allocated graph; `ctx` is the graph's
/// compute context (res->get_ctx()); `n_layer` is the model layer count.
///
/// Contract: call AFTER build_graph() and BEFORE ggml_backend_sched_alloc_graph() /
/// ggml_backend_sched_reserve().
int rrl_insert_layer_barriers(struct ggml_cgraph * gf,
                              struct ggml_context * ctx,
                              int n_layer, int segment);

/// Test introspection: total barriers inserted by the most recent rewrite.
/// Thread-safe (atomic); only meaningful in serialized test contexts.
int rrl_last_barrier_count(void);

#ifdef __cplusplus
}
#endif
