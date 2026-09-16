// Tests for the Metal graph reorder's boundary barriers.
//
// ggml_graph_optimize hoists memory-concurrent nodes forward to widen the concurrent set.
// A caller that installs a boundary-event schedule and an encode window cuts the graph at
// the per-layer output markers, and the reorder runs before that schedule exists, so left
// alone it can move a node out of the segment that is supposed to produce it. These tests
// pin both halves of the fix: with the barriers off the hoist still happens (the reorder
// is untouched), and with them on nothing crosses a marker.
//
// Everything here is a hand-built graph. No device, no model, no allocation.

#include "ggml.h"

#include "ggml-metal-common.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_n_fail = 0;

static std::vector<std::string> node_names(ggml_cgraph * gf) {
    const int n = ggml_graph_n_nodes(gf);

    std::vector<std::string> res;
    res.reserve(n);

    for (int i = 0; i < n; i++) {
        res.push_back(ggml_get_name(ggml_graph_node(gf, i)));
    }

    return res;
}

static std::string join(const std::vector<std::string> & names) {
    std::string res;

    for (size_t i = 0; i < names.size(); i++) {
        if (i > 0) {
            res += " ";
        }
        res += names[i];
    }

    return res;
}

static void check_order(const char * scenario, ggml_cgraph * gf, const std::vector<std::string> & expected) {
    const std::vector<std::string> actual = node_names(gf);

    if (actual == expected) {
        printf("PASS %-40s %s\n", scenario, join(actual).c_str());
        return;
    }

    printf("FAIL %-40s got [%s], expected [%s]\n", scenario, join(actual).c_str(), join(expected).c_str());

    g_n_fail++;
}

// a scratch context big enough for the tiny graphs below
struct test_ctx {
    ggml_context * ctx;

    test_ctx() {
        ggml_init_params params = {
            /*.mem_size   =*/ 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(params);
    }

    ~test_ctx() {
        ggml_free(ctx);
    }
};

static ggml_tensor * leaf(ggml_context * ctx, const char * name) {
    ggml_tensor * res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_set_name(res, name);

    return res;
}

static ggml_tensor * named(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);

    return t;
}

// An independent lookup into a persistent cache: nothing in the graph anchors it, which is
// exactly the node the reorder is free to hoist as far as it likes.
static ggml_tensor * cache_lookup(ggml_context * ctx, const char * name) {
    ggml_tensor * cache = leaf(ctx, "cache");
    ggml_tensor * ids   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_name(ids, "ids");

    return named(ggml_get_rows(ctx, cache, ids), name);
}

// n0 writes "a", n1 reads it -- n1 therefore conflicts with the concurrent set n0 opened,
// which is what makes the reorder look forward for something to hoist in front of n1.
//
//   a  -> mul(x, y)
//   b  -> mul(a, w)      conflicts with a
//   mk -> add(b, x)      the boundary marker
//   c  -> get_rows(...)  independent, and after the marker
static ggml_cgraph * build_hoist_across(ggml_context * ctx, const char * marker_name) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");
    ggml_tensor * w = leaf(ctx, "w");

    ggml_tensor * a  = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * b  = named(ggml_mul(ctx, a, w), "b");
    ggml_tensor * mk = named(ggml_add(ctx, b, x), marker_name);
    ggml_tensor * c  = cache_lookup(ctx, "c");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_build_forward_expand(gf, mk);
    ggml_build_forward_expand(gf, c);

    return gf;
}

// Same idea, but the marker is itself the node that opens the look-forward scan. Everything
// the scan picks up is emitted BEFORE the marker, so the scan has to be skipped entirely.
//
//   a  -> mul(x, y)
//   mk -> add(a, x)      the boundary marker, and the conflicting node
//   c  -> get_rows(...)  independent, and after the marker
static ggml_cgraph * build_marker_anchor(ggml_context * ctx) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");

    ggml_tensor * a  = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * mk = named(ggml_add(ctx, a, x), "l_out-0");
    ggml_tensor * c  = cache_lookup(ctx, "c");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, mk);
    ggml_build_forward_expand(gf, c);

    return gf;
}

// The marker packed into a fused group as a non-head member. The reorder moves whole groups,
// so the marker rides along with "s" unless the barrier test looks past the group's head.
//
//   a  -> mul(x, y)
//   b  -> mul(a, w)      conflicts with a
//   s  -> add(p, q)      independent, fusable head
//   mk -> add(s, v)      the boundary marker, fused onto s
static ggml_cgraph * build_fused_group(ggml_context * ctx) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");
    ggml_tensor * w = leaf(ctx, "w");
    ggml_tensor * p = leaf(ctx, "p");
    ggml_tensor * q = leaf(ctx, "q");
    ggml_tensor * v = leaf(ctx, "v");

    ggml_tensor * a  = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * b  = named(ggml_mul(ctx, a, w), "b");
    ggml_tensor * s  = named(ggml_add(ctx, p, q), "s");
    ggml_tensor * mk = named(ggml_add(ctx, s, v), "l_out-0");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_build_forward_expand(gf, s);
    ggml_build_forward_expand(gf, mk);

    return gf;
}

int main() {
    // The reorder only ever has something to do when the graph it is given is in the order
    // these builders lay it out, so check that first -- otherwise a change in how
    // ggml_build_forward_expand walks the graph would quietly make every case below vacuous.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "l_out-0");

        check_order("graph as built", gf, { "a", "b", "l_out-0", "c" });
    }

    // Positive control: with the barriers off, the reorder hoists the independent lookup
    // across the marker. This is the behaviour every non-boundary-scheduling caller gets,
    // and it must not change.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "l_out-0");

        ggml_graph_optimize(gf, false);

        check_order("barriers off: hoists across marker", gf, { "a", "c", "b", "l_out-0" });
    }

    // With the barriers on, the same node stays in its own segment.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "l_out-0");

        ggml_graph_optimize(gf, true);

        check_order("barriers on: stops at marker", gf, { "a", "b", "l_out-0", "c" });
    }

    // The barrier keys on the marker, not on the switch: a node that is not a marker is
    // still hoisted with the barriers on, so turning them on is not a blanket "no reorder".
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "ffn_out-0");

        ggml_graph_optimize(gf, true);

        check_order("barriers on: non-marker still hoists", gf, { "a", "c", "b", "ffn_out-0" });
    }

    // A marker that is itself the node the scan starts from.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_marker_anchor(tc.ctx);

        ggml_graph_optimize(gf, false);

        check_order("barriers off: hoists in front of marker", gf, { "a", "c", "l_out-0" });
    }

    {
        test_ctx     tc;
        ggml_cgraph * gf = build_marker_anchor(tc.ctx);

        ggml_graph_optimize(gf, true);

        check_order("barriers on: marker takes nothing along", gf, { "a", "l_out-0", "c" });
    }

    // Fusion packs "s" and the marker into one group. With the barriers off the group is
    // hoisted whole, which moves the marker itself -- the two staying adjacent is what says
    // they were really fused rather than reordered independently.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx);

        ggml_graph_optimize(gf, false);

        check_order("barriers off: fused group hoists", gf, { "a", "s", "l_out-0", "b" });
    }

    // ... and with them on the group holds its place, because a group is a barrier when any
    // of its members is a marker, not only when its head is.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx);

        ggml_graph_optimize(gf, true);

        check_order("barriers on: fused group holds", gf, { "a", "b", "s", "l_out-0" });
    }

    // The name test itself.
    {
        test_ctx tc;

        int n_bad = 0;

        const char * markers[] = { "l_out-0", "l_out-63" };
        for (size_t i = 0; i < sizeof(markers)/sizeof(markers[0]); i++) {
            if (!ggml_graph_node_is_boundary_marker(leaf(tc.ctx, markers[i]))) {
                printf("FAIL marker name: '%s' not recognised\n", markers[i]);
                n_bad++;
            }
        }

        const char * not_markers[] = { "l_out-", "l_out-0x", "l_out", "ffn_out-0", "" };
        for (size_t i = 0; i < sizeof(not_markers)/sizeof(not_markers[0]); i++) {
            if (ggml_graph_node_is_boundary_marker(leaf(tc.ctx, not_markers[i]))) {
                printf("FAIL marker name: '%s' recognised as a marker\n", not_markers[i]);
                n_bad++;
            }
        }

        if (n_bad == 0) {
            printf("PASS %-40s\n", "marker name matching");
        }

        g_n_fail += n_bad;
    }

    if (g_n_fail > 0) {
        printf("\n%d check(s) failed\n", g_n_fail);
        return 1;
    }

    printf("\nall checks passed\n");

    return 0;
}
