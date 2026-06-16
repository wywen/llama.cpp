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
//
// [Step 4 Increment 3a] The function now accepts an optional residency manager
// callback pair (void* mgr, void (*on_barrier)(void*, int)).  When both are non-null
// each barrier custom-op invokes on_barrier(mgr, il) after the identity memcpy.
// When either is null the behaviour is identical to Increment 2 (pure passthrough).
// This keeps the barriers TU model-agnostic: it never knows the residency manager.

#include "ggml-backend.h" // ggml_backend_dev_t, ggml_backend_buffer_t, ggml_tensor

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
/// `mgr` and `on_barrier` are the residency manager callback pair (Step 4 Inc 3a).
/// Pass both as NULL for the Increment-2 pure-passthrough behaviour.
///
/// Contract: call AFTER build_graph() and BEFORE ggml_backend_sched_alloc_graph() /
/// ggml_backend_sched_reserve().
int rrl_insert_layer_barriers(struct ggml_cgraph * gf,
                              struct ggml_context * ctx,
                              int n_layer, int segment,
                              void * mgr,
                              void (*on_barrier)(void *, int));

/// Test introspection: total barriers inserted by the most recent rewrite.
/// Thread-safe (atomic); only meaningful in serialized test contexts.
int rrl_last_barrier_count(void);

// [Step 4 Increment 3a] Residency-manager hook table.
//
// libllama (a two-level-namespace NOUNDEFS dylib on macOS) cannot reference
// symbols defined in the Rust crate.  Instead, call-through function pointers
// are stored here and installed at startup by the crate via
// rrl_install_residency_hooks.  Default is null (no-op) for every
// non-mmap-metal context.

typedef void *(*rrl_get_mgr_fn)(ggml_backend_dev_t dev);
typedef void  (*rrl_register_fn)(ggml_backend_dev_t dev, int il,
                                 ggml_backend_buffer_t * slot,
                                 struct ggml_tensor * k,
                                 struct ggml_tensor * v,
                                 int first_reader, int last_reader,
                                 const void * owner);
typedef void  (*rrl_on_barrier_fn)(void * mgr, int il);
typedef void  (*rrl_pre_decode_reset_fn)(void * mgr);
typedef int   (*rrl_plan_barriers_fn)(void * mgr, int * out, int max_out);
typedef int   (*rrl_armed_fn)(void * mgr);
typedef void  (*rrl_unregister_fn)(ggml_backend_dev_t dev, const void * owner);

/// Install real residency-manager function pointers.  Called once from the
/// Rust crate after mmap_metal_open succeeds.  Thread-hostile; call before
/// any llama_context using the mmap-metal device is constructed.
void rrl_install_residency_hooks(rrl_get_mgr_fn   get_mgr,
                                 rrl_register_fn  reg,
                                 rrl_on_barrier_fn on_barrier,
                                 rrl_pre_decode_reset_fn pre_decode_reset,
                                 rrl_plan_barriers_fn plan_barriers,
                                 rrl_armed_fn armed,
                                 rrl_unregister_fn unregister);

/// Return the residency manager pointer for dev (dispatches through hook).
void * rrl_residency_manager_ptr(ggml_backend_dev_t dev);

/// Whether eviction is armed for `mgr` (dispatches through hook). The
/// llama_context asks this — AFTER the kv-cache has registered its layers —
/// instead of reading the RRL_KV_* env itself, so the residency manager is the
/// single source of truth for "is eviction on" on the mmap-metal path. Returns
/// 0 when mgr is null (the bare no-manager path arms via RRL_KV_SEGMENT).
int rrl_residency_armed(void * mgr);

/// Drop the per-layer registrations made by `owner` (one kv-cache) for dev's
/// manager (dispatches through hook). Called from ~llama_kv_cache so a reused
/// mmap-metal device never carries slots into a destroyed cache's buffers (RAII
/// teardown). Owner-scoped so a live sibling sub-cache (iSWA) keeps its entries.
/// No-op if dev has no manager.
void rrl_residency_unregister(ggml_backend_dev_t dev, const void * owner);

/// Register a per-layer KV buffer (dispatches through hook).
/// `slot` points at llama's OWNING raw handle (inside the kv-cache ctxs_bufs
/// unique_ptr); the manager frees/recreates through it to stay single-owner.
/// `first_reader`/`last_reader` are the min/max model-layer indices that map to
/// this buffer once KV-cache reuse (`map_layer_ids`) is applied; for a non-shared
/// layer both equal `il`. The manager frees a buffer only after a barrier past
/// `last_reader` — the invariant that makes eviction safe under KV sharing.
/// `owner` identifies the registering kv-cache (its `this`), so unregistration
/// can be scoped to exactly the entries this cache made.
void rrl_residency_register(ggml_backend_dev_t dev, int il,
                            ggml_backend_buffer_t * slot,
                            struct ggml_tensor * k,
                            struct ggml_tensor * v,
                            int first_reader, int last_reader,
                            const void * owner);

/// Barrier callback trampoline (dispatches through hook).
void rrl_residency_on_barrier(void * mgr, int il);

/// Pre-decode reset: recreates all evicted (dead) layer buffers before the
/// next sched_alloc_graph call.  Must be called by process_ubatch's else-branch
/// (graph rebuild path) BEFORE ggml_backend_sched_alloc_graph.
/// No-op if mgr is null or eviction is disabled (segment=0).
void rrl_residency_pre_decode_reset(void * mgr);

#ifdef __cplusplus
}
#endif
