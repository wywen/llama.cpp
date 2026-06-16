// Graph-rewrite: identity CPU barrier ops at layer boundaries.
//
// See llama-barriers.h for the public API.

#include "llama-barriers.h"

#include "ggml.h"
#include "ggml-impl.h"   // ggml_cgraph struct (nodes / n_nodes / size)

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Userdata for the barrier custom op.
//
// Each barrier node gets its own heap-allocated BarrierUserdata.  Owned in
// s_barrier_userdata (cleared at the start of each rewrite, which is also
// when the graph is rebuilt — so the old userdata is safe to destroy before
// the new graph is built).  Lifetime: must outlive every graph execution that
// reuses the current graph; cleared only when a new rewrite runs (once per
// graph rebuild, which happens when can_reuse is false — same semantics as
// Increment 2's node insertion).
// ---------------------------------------------------------------------------

struct BarrierUserdata {
    int il = -1;
    void * mgr = nullptr;
    void (*on_barrier)(void *, int) = nullptr;
};

// Owned storage for the current graph's userdata objects.  Cleared at the
// start of each rrl_insert_layer_barriers call (safe: old graph is
// being replaced).
static std::vector<std::unique_ptr<BarrierUserdata>> s_barrier_userdata;

// ---------------------------------------------------------------------------
// Identity callback — byte-identical passthrough + optional manager callback.
// NON-inplace: dst is a freshly allocated tensor shaped like a, so the
// scheduler sees a distinct node and is free to assign it to a different
// backend (CPU) than the upstream l_out (Metal).
// ---------------------------------------------------------------------------

static void rrl_barrier_cb(struct ggml_tensor * dst,
                            const struct ggml_tensor * a,
                            int ith, int nth, void * userdata) {
    (void) nth;
    if (ith == 0) {
        const size_t nbytes_dst = ggml_nbytes(dst);
        const size_t nbytes_a   = ggml_nbytes(a);
        const size_t nbytes_min = nbytes_dst < nbytes_a ? nbytes_dst : nbytes_a;
        if (nbytes_dst != nbytes_a) {
            fprintf(stderr, "[rrl_barrier] WARNING: size mismatch: "
                    "dst=%s nbytes=%zu, a=%s nbytes=%zu — copying min\n",
                    dst->name, nbytes_dst, a->name, nbytes_a);
        }
        // Identity passthrough: copy a → dst.
        memcpy(dst->data, a->data, nbytes_min);

        // NOTE (Step 4): this memcpy is structurally load-bearing, not a removable
        // identity. The barrier is a CPU custom-op (Metal rejects CUSTOM1), so the
        // scheduler also inserts a hidden CPU→Metal copy of dst into the next Metal
        // segment: TWO full-tensor copies per boundary even on UMA. A pointer alias
        // (dst->data = a->data) removes only this explicit copy, strands dst's
        // gallocr slot, and leaves the scheduler copy intact — a half-win. True
        // zero-copy needs dst pinned to Metal + a Metal-fence on_barrier trigger
        // (ggml_backend_sched_set_tensor_backend), tracked as Increment 4.

        // [Increment 3a] Fire the residency manager callback if wired.
        const BarrierUserdata * ud = static_cast<const BarrierUserdata *>(userdata);
        if (ud && ud->on_barrier && ud->mgr) {
            ud->on_barrier(ud->mgr, ud->il);
        }
    }
}

// ---------------------------------------------------------------------------
// State for rrl_last_barrier_count()
// ---------------------------------------------------------------------------

static std::atomic<int> s_last_barrier_count{0};

int rrl_last_barrier_count(void) {
    return s_last_barrier_count.load();
}

// ---------------------------------------------------------------------------
// Main rewrite
// ---------------------------------------------------------------------------

// Barrier-plan hook (installed via rrl_install_residency_hooks). Declared here
// because rrl_insert_layer_barriers below reads it to place barriers exactly
// where the residency manager's recreate/free logic expects them.
static rrl_plan_barriers_fn s_plan_barriers_hook = nullptr;

int rrl_insert_layer_barriers(struct ggml_cgraph * gf,
                               struct ggml_context * ctx,
                               int n_layer, int segment,
                               void * mgr,
                               void (*on_barrier_fn)(void *, int)) {
    // Clear owned userdata from any previous rewrite (safe: graph is being
    // rebuilt, so the previous graph's nodes are no longer live).
    s_barrier_userdata.clear();
    s_last_barrier_count.store(0);

    // Proceed if either a fixed stride is requested (segment >= 1) or a residency
    // manager is present to supply a barrier plan (SWA-weighted byte budget).
    if (!gf || !ctx || n_layer <= 0 || (segment < 1 && !mgr)) {
        return 0;
    }

    const int n_nodes = ggml_graph_n_nodes(gf);

    // ── Step 1: scan nodes for l_out-<il> tensors ─────────────────────────
    //
    // Build l_out[il] -> node pointer map.  Use a flat array indexed by il
    // (n_layer entries); absent entries stay null.

    std::vector<struct ggml_tensor *> l_out(static_cast<size_t>(n_layer), nullptr);
    int max_il = -1;
    int found  = 0;

    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * n = ggml_graph_node(gf, i);
        if (!n || !n->name[0]) {
            continue;
        }
        int il = -1;
        if (sscanf(n->name, "l_out-%d", &il) == 1 && il >= 0 && il < n_layer) {
            l_out[il] = n;
            if (il > max_il) {
                max_il = il;
            }
            found++;
        }
    }

    // Need at least 2 l_out tensors (so there's a non-final layer to barrier).
    if (found < 2 || max_il < 1) {
        return 0;
    }

    // ── Step 2: choose barrier layers ─────────────────────────────────────
    //
    // When a residency manager is present it is AUTHORITATIVE: ask it for the
    // barrier plan (it knows each KV buffer's size and reader range, placing
    // barriers by SWA-weighted byte budget or fixed stride, exactly where its
    // recreate/free logic expects). An empty plan therefore means NO barriers —
    // only WITHOUT a manager (bare Inc-2 path) do we fall back to the stride.
    // (Never barrier the final layer — nothing consumes a barrier after it.)

    std::vector<int> barrier_ils;
    bool planned = false;
    if (mgr && s_plan_barriers_hook) {
        std::vector<int> plan(static_cast<size_t>(n_layer), 0);
        const int n_plan = s_plan_barriers_hook(mgr, plan.data(), n_layer);
        if (n_plan > 0) {
            planned = true;
            for (int i = 0; i < n_plan; ++i) {
                const int il = plan[static_cast<size_t>(i)];
                if (il >= 0 && il < max_il && l_out[il] != nullptr) {
                    barrier_ils.push_back(il);
                }
            }
        }
    }
    if (!planned && !mgr && segment >= 1) {
        for (int il = segment - 1; il < max_il; il += segment) {
            if (l_out[il] != nullptr) {   // only if the layer's tensor exists
                barrier_ils.push_back(il);
            }
        }
    }

    if (barrier_ils.empty()) {
        return 0;
    }

    // ── Step 3: create barrier tensors ────────────────────────────────────
    //
    // One CUSTOM1 (non-inplace) tensor per barrier il.  These are allocated
    // inside `ctx` but NOT yet in gf->nodes.
    //
    // IMPORTANT: GGML_TENSOR_FLAG_COMPUTE must be set manually.  Normally
    // this flag is applied by ggml_build_forward_expand() during its parent
    // traversal, but barrier tensors are inserted AFTER that traversal
    // completes (inside model.build_graph()).  Without this flag the CPU
    // backend's compute thread silently skips the node (ggml-cpu.c:3049).

    std::vector<struct ggml_tensor *> barriers(static_cast<size_t>(n_layer), nullptr);

    for (int il : barrier_ils) {
        struct ggml_tensor * src_node = l_out[il];

        // Build userdata for this barrier.  If mgr/on_barrier_fn are null
        // (non-mmap-metal path) the BarrierUserdata still holds null pointers
        // and the callback silently skips the manager call — pure Increment-2
        // passthrough.
        auto ud = std::make_unique<BarrierUserdata>();
        ud->il = il;
        ud->mgr = mgr;
        ud->on_barrier = on_barrier_fn;
        BarrierUserdata * ud_raw = ud.get();
        s_barrier_userdata.push_back(std::move(ud));

        struct ggml_tensor * C =
            ggml_map_custom1(ctx, src_node, rrl_barrier_cb, /*n_tasks=*/1, ud_raw);
        ggml_format_name(C, "rrl_barrier-%d", il);
        C->flags |= GGML_TENSOR_FLAG_COMPUTE;
        barriers[il] = C;

        // Safety: the barrier must not already appear in the graph.
        for (int i = 0; i < n_nodes; ++i) {
            assert(ggml_graph_node(gf, i) != C && "barrier tensor unexpectedly already in graph");
        }
    }

    // ── Step 4: redirect consumers ────────────────────────────────────────
    //
    // Walk all existing graph nodes.  For each src slot pointing at a
    // l_out[il] that has a barrier, redirect to C[il].  Also fix view_src.
    //
    // Use POINTER comparison against the canonical l_out[il] pointers from
    // step 1 — not name matching — to avoid false matches on tensors with
    // similar names.
    //
    // The C[] tensors are not in gf yet, so there's no risk of self-redirect.

    // Build a lookup: l_out[il] pointer → barrier[il] pointer (only for barrier_ils).
    // Use two parallel vectors for O(n_barriers) linear scan — n_barriers << 32.
    std::vector<struct ggml_tensor *> redirect_from;
    std::vector<struct ggml_tensor *> redirect_to;
    redirect_from.reserve(barrier_ils.size());
    redirect_to.reserve(barrier_ils.size());
    for (int il : barrier_ils) {
        redirect_from.push_back(l_out[il]);
        redirect_to.push_back(barriers[il]);
    }

    auto find_redirect = [&](struct ggml_tensor * ptr) -> struct ggml_tensor * {
        for (size_t k = 0; k < redirect_from.size(); ++k) {
            if (redirect_from[k] == ptr) {
                return redirect_to[k];
            }
        }
        return nullptr;
    };

    int n_redirects = 0;
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor * n = ggml_graph_node(gf, i);
        if (!n) {
            continue;
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!n->src[j]) {
                break;
            }
            struct ggml_tensor * replacement = find_redirect(n->src[j]);
            if (replacement) {
                n->src[j] = replacement;
                n_redirects++;
            }
        }

        if (n->view_src) {
            struct ggml_tensor * replacement = find_redirect(n->view_src);
            if (replacement) {
                n->view_src = replacement;
                n_redirects++;
            }
        }
    }

    (void) n_redirects; // available for debug inspection

    // ── Step 5: insert C[il] into gf->nodes right after l_out[il] ─────────
    //
    // Rebuild the node array in-place.  We need direct gf->nodes access,
    // available via ggml-impl.h.

    const int n_barriers = static_cast<int>(barrier_ils.size());
    const int new_n_nodes = n_nodes + n_barriers;
    assert(new_n_nodes <= gf->size &&
           "not enough graph slots for barriers — max_nodes insufficient");

    // Build a set of barrier ils for O(1) lookup.
    // barrier_ils is small (<<32 elements); linear scan is fine.
    auto is_barrier_il = [&](int il) -> bool {
        for (int x : barrier_ils) {
            if (x == il) {
                return true;
            }
        }
        return false;
    };

    // Make a snapshot of the current node pointers.
    std::vector<struct ggml_tensor *> old_nodes(gf->nodes, gf->nodes + n_nodes);

    int out_idx = 0;
    for (int i = 0; i < n_nodes; ++i) {
        gf->nodes[out_idx++] = old_nodes[i];

        // If this node is a barrier l_out, append the barrier immediately after.
        int il_node = -1;
        if (old_nodes[i] && old_nodes[i]->name[0] &&
            sscanf(old_nodes[i]->name, "l_out-%d", &il_node) == 1 &&
            is_barrier_il(il_node)) {
            gf->nodes[out_idx++] = barriers[il_node];
        }
    }

    assert(out_idx == new_n_nodes);
    gf->n_nodes = new_n_nodes;

    s_last_barrier_count.store(n_barriers);
    return n_barriers;
}

// ---------------------------------------------------------------------------
// [Step 4 Increment 3a] Residency-manager hook table.
//
// The libllama dylib uses two-level namespace (NOUNDEFS), so it cannot have
// undefined external references to rrl_residency_* symbols that live in the
// Rust crate's mmap_metal_shim.cpp.  Instead we expose a hook-setter that the
// crate calls once (from rrl_residency_install_hooks) to install the real
// implementations, and self-contained no-op defaults cover every other case.
//
// All three hooks live inside the dylib; no undefined symbols required.
// The crate only needs to call rrl_install_residency_hooks (one function,
// exported from libllama, so it is resolved via the normal two-level path).
// ---------------------------------------------------------------------------

// The hook function-pointer typedefs (rrl_get_mgr_fn / rrl_register_fn /
// rrl_on_barrier_fn) and the extern "C" prototypes below are declared in
// llama-barriers.h (included at the top of this file) — not re-declared here.

// Installed hook table.  Zero-initialised (null = no-op).
static rrl_get_mgr_fn          s_get_mgr_hook          = nullptr;
static rrl_register_fn         s_register_hook         = nullptr;
// on_barrier is already handled via function pointer passed to
// rrl_insert_layer_barriers, so we store it here for use in context.cpp.
static rrl_on_barrier_fn       s_on_barrier_hook       = nullptr;
static rrl_pre_decode_reset_fn s_pre_decode_reset_hook = nullptr;
static rrl_armed_fn            s_armed_hook            = nullptr;
static rrl_unregister_fn       s_unregister_hook       = nullptr;
// s_plan_barriers_hook is declared near the top of the file (above
// rrl_insert_layer_barriers, which reads it). Defined/assigned in the installer.

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

/// Barrier callback trampoline.  No-op if hooks not installed.
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
/// is null — the caller then arms the bare path via RRL_KV_SEGMENT.
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

} // extern "C"
