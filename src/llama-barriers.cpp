// Barrier hook trampolines.
//
// The barrier custom-op and the graph rewrite that inserts it (rrl_barrier_cb +
// the body of rrl_insert_layer_barriers) no longer live here — they were moved
// into the rust-recipes-llama crate, compiled into its own DSO alongside the
// ResidencyManager. libllama is a two-level-namespace NOUNDEFS dylib on macOS
// and cannot reference crate symbols, so everything routes through a function-
// pointer hook table the crate installs at device-open time.
//
// This file keeps only (a) that hook table and its installers, and (b) thin
// forwarders for the two symbols llama-context.cpp and the Rust tests still call
// by name (rrl_insert_layer_barriers, rrl_last_barrier_count). The benefit:
// iterating on the op or the rewrite is now a crate recompile (cxx-build,
// seconds) with NO llama.cpp re-patch / rebuild — only a change to a hook
// *signature* or a call-site placement still needs a patch edit.

#include "llama-barriers.h"

// ---------------------------------------------------------------------------
// [Step 4] Residency-manager hook table.
//
// Installed by the crate via rrl_install_residency_hooks. Zero-initialised
// (null = no-op) so every non-mmap-metal context is unaffected.
// ---------------------------------------------------------------------------

static rrl_get_mgr_fn          s_get_mgr_hook          = nullptr;
static rrl_register_fn         s_register_hook         = nullptr;
static rrl_on_barrier_fn       s_on_barrier_hook       = nullptr;
static rrl_pre_decode_reset_fn s_pre_decode_reset_hook = nullptr;
// Stored by the installer but unread since the op/rewrite moved to the crate (its
// rewrite calls the manager's copy_plan() directly). Retained only to keep
// rrl_install_residency_hooks's 7-pointer ABI stable; do NOT wire it back in here.
static rrl_plan_barriers_fn    s_plan_barriers_hook    = nullptr;
static rrl_armed_fn            s_armed_hook            = nullptr;
static rrl_unregister_fn       s_unregister_hook       = nullptr;

// ---------------------------------------------------------------------------
// [Step 4] Barrier-rewrite hook table.
//
// Installed by the crate via rrl_install_barrier_hooks. The rewrite + op live
// in the crate (it owns the ResidencyManager the op notifies), so these are the
// only link from the patched call-sites into that code.
// ---------------------------------------------------------------------------

static rrl_rewrite_fn          s_rewrite_hook          = nullptr;
static rrl_last_count_fn        s_last_count_hook       = nullptr;

extern "C" {

/// Called by the Rust crate (from mmap_metal_shim.cpp) to install the real
/// residency manager function pointers.  Must be called before the first
/// llama_context that uses the mmap-metal device is constructed.
void rrl_install_residency_hooks(rrl_get_mgr_fn get_mgr,
                                 rrl_register_fn reg,
                                 rrl_on_barrier_fn on_barrier,
                                 rrl_pre_decode_reset_fn pre_decode_reset,
                                 rrl_plan_barriers_fn plan_barriers,
                                 rrl_armed_fn armed,
                                 rrl_unregister_fn unregister) {
    s_get_mgr_hook          = get_mgr;
    s_register_hook         = reg;
    s_on_barrier_hook       = on_barrier;
    s_pre_decode_reset_hook = pre_decode_reset;
    s_plan_barriers_hook    = plan_barriers;
    s_armed_hook            = armed;
    s_unregister_hook       = unregister;
}

/// Install the barrier-rewrite + barrier-count function pointers (the op and the
/// rewrite that live in the crate DSO).  Called once from the crate alongside
/// rrl_install_residency_hooks.
void rrl_install_barrier_hooks(rrl_rewrite_fn rewrite,
                               rrl_last_count_fn last_count) {
    s_rewrite_hook    = rewrite;
    s_last_count_hook = last_count;
}

/// Return the residency manager pointer for dev (null if not mmap-metal or
/// hooks not installed).
void * rrl_residency_manager_ptr(ggml_backend_dev_t dev) {
    if (!s_get_mgr_hook) {
        return nullptr;
    }
    return s_get_mgr_hook(dev);
}

/// Register a per-layer KV buffer.  No-op if hooks not installed.
void rrl_residency_register(ggml_backend_dev_t dev, int il,
                            ggml_backend_buffer_t * slot,
                            struct ggml_tensor * k,
                            struct ggml_tensor * v,
                            int first_reader, int last_reader,
                            const void * owner) {
    if (!s_register_hook) {
        return;
    }
    s_register_hook(dev, il, slot, k, v, first_reader, last_reader, owner);
}

/// Barrier callback trampoline.  Retained so the rewrite hook signature (which
/// carries an on_barrier function pointer for the bare/no-manager path) stays
/// stable; the crate's op normally notifies its ResidencyManager directly.
void rrl_residency_on_barrier(void * mgr, int il) {
    if (!s_on_barrier_hook) {
        return;
    }
    s_on_barrier_hook(mgr, il);
}

/// Pre-decode reset: recreate all evicted layer buffers before sched_alloc_graph.
/// No-op if hooks not installed or mgr is null.
void rrl_residency_pre_decode_reset(void * mgr) {
    if (!s_pre_decode_reset_hook || !mgr) {
        return;
    }
    s_pre_decode_reset_hook(mgr);
}

/// Whether eviction is armed for mgr.  Returns 0 if hooks not installed or mgr
/// is null — the caller then arms the bare path via the fixed stride.
int rrl_residency_armed(void * mgr) {
    if (!s_armed_hook || !mgr) {
        return 0;
    }
    return s_armed_hook(mgr);
}

/// Drop the per-layer registrations made by `owner` for dev's manager (kv-cache
/// teardown). No-op if hooks not installed or dev has no manager.
void rrl_residency_unregister(ggml_backend_dev_t dev, const void * owner) {
    if (!s_unregister_hook) {
        return;
    }
    s_unregister_hook(dev, owner);
}

/// Graph rewrite: forwards to the crate's real implementation (s_rewrite_hook).
/// Same signature and call-sites as before the op/rewrite moved to the crate, so
/// llama-context.cpp is unchanged.  Returns 0 (no barriers) when the hook is not
/// installed (no mmap-metal device → non-mmap-metal context).
int rrl_insert_layer_barriers(struct ggml_cgraph * gf,
                              struct ggml_context * ctx,
                              int n_layer, int segment,
                              void * mgr,
                              void (*on_barrier)(void *, int)) {
    return s_rewrite_hook
               ? s_rewrite_hook(gf, ctx, n_layer, segment, mgr, on_barrier)
               : 0;
}

/// Test introspection: total barriers inserted by the most recent rewrite.
/// Forwards to the crate (the counter lives there now).  Returns 0 if the hook
/// is not installed.
int rrl_last_barrier_count(void) {
    return s_last_count_hook ? s_last_count_hook() : 0;
}

} // extern "C"
