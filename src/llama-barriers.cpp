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
#include <vector>

// ---------------------------------------------------------------------------
// Identity callback — byte-identical passthrough.
// NON-inplace: dst is a freshly allocated tensor shaped like a, so the
// scheduler sees a distinct node and is free to assign it to a different
// backend (CPU) than the upstream l_out (Metal).
// ---------------------------------------------------------------------------

static void rrl_barrier_cb(struct ggml_tensor * dst,
                            const struct ggml_tensor * a,
                            int ith, int nth, void * userdata) {
    (void) nth;
    (void) userdata;
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

int rrl_insert_layer_barriers(struct ggml_cgraph * gf,
                               struct ggml_context * ctx,
                               int n_layer, int segment) {
    s_last_barrier_count.store(0);

    if (!gf || !ctx || n_layer <= 0 || segment < 1) {
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
    // il = segment-1, 2*segment-1, ...  while il < max_il
    // (Never barrier the final layer — nothing consumes a barrier after it.)

    std::vector<int> barrier_ils;
    for (int il = segment - 1; il < max_il; il += segment) {
        if (l_out[il] != nullptr) {   // only if the layer's tensor exists
            barrier_ils.push_back(il);
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
        struct ggml_tensor * C =
            ggml_map_custom1(ctx, src_node, rrl_barrier_cb, /*n_tasks=*/1, /*userdata=*/nullptr);
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
