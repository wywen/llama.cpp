// checks the MUL_MAT+ADD packs of the Metal graph reorder (devices, pack readers, src1 rows), that the encoder fuses
// no MUL_MAT+ADD the reorder leaves unpacked, the reorder's boundary barriers and barrier name prefixes, and that the
// encoder's concurrency check covers the middle nodes of a fused group
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-fusion.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static constexpr int     n_tensors = 64;
static constexpr float   scale     = 2.0f;
static constexpr float   norm_eps  = 1e-6f;
// a mat-mul the few-row MMA kernels take: K a multiple of their 64-weight step, 2..16 src1 rows
static constexpr int64_t n_k       = 64;
static constexpr int64_t n_m       = 16;
static constexpr int64_t n_rows    = 8;
// src1 row counts below, inside and above the 2..16 rows of the few-row MMA kernels
static const std::vector<int64_t> batch_rows = { 1, 4, 9, 64, 512 };

struct device_case {
    const char * name;

    bool has_native_simdgroup_mm;
    bool packed;
};

// a device with probed simdgroup matrices that are native (MTLGPUFamilyApple7+) only if has_native_simdgroup_mm
static ggml_metal_device_props device_props(bool has_native_simdgroup_mm) {
    ggml_metal_device_props props = {};
    props.has_simdgroup_mm = has_native_simdgroup_mm;
    return props;
}

// the index of t in the nodes of graph, -1 if absent
static int node_index(ggml_cgraph * graph, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (ggml_graph_node(graph, i) == t) {
            return i;
        }
    }
    return -1;
}

// the reordered positions of the tracked nodes
static std::vector<int> node_positions(ggml_cgraph * graph, const std::vector<ggml_tensor *> & tracked) {
    std::vector<int> res;
    for (const ggml_tensor * t : tracked) {
        res.push_back(node_index(graph, t));
    }
    return res;
}

// allocates the tensors of ctx in a CPU buffer marked as weights, as the model loader does
static ggml_backend_buffer_t alloc_weights(ggml_context * ctx) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buffer;
}

// what the add sums with the mat-mul
enum add_operand_kind {
    ADD_RESIDUAL,  // a same-shape activation
    ADD_BIAS,      // a one-row bias weight
    ADD_BIAS_VIEW, // a view of a bias weight
};

static const char * add_operand_name(add_operand_kind kind) {
    switch (kind) {
        case ADD_RESIDUAL:  return "residual";
        case ADD_BIAS:      return "bias";
        case ADD_BIAS_VIEW: return "bias view";
    }
    return "?";
}

static ggml_tensor * new_add_operand(ggml_context * ctx, ggml_context * ctx_w, add_operand_kind kind, int64_t rows) {
    switch (kind) {
        case ADD_RESIDUAL:  return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_m, rows);
        case ADD_BIAS:      return ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_m, 1);
        case ADD_BIAS_VIEW: return ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, n_m), n_m, 1);
    }
    return nullptr;
}

// the reordered positions of a mat-mul with rows src1 rows, the add of an operand of kind to it and an independent
// mat-mul, which the reorder runs between the first mat-mul and the add unless the pair is packed
static std::vector<int> reorder_mul_mat_add(int64_t rows, add_operand_kind kind) {
    ggml_init_params params = { n_tensors*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx   = ggml_init(params);
    ggml_context * ctx_w = ggml_init(params);
    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, rows);
    ggml_tensor * mm    = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_k, n_m), x);
    ggml_tensor * add   = ggml_add(ctx, mm, new_add_operand(ctx, ctx_w, kind, rows));
    ggml_tensor * other = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_k, n_m), x);
    ggml_backend_buffer_t weights = alloc_weights(ctx_w);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, add);
    ggml_build_forward_expand(graph, other);
    ggml_graph_optimize(graph, false, 0, nullptr);
    const std::vector<int> res = node_positions(graph, { mm, add, other });

    ggml_backend_buffer_free(weights);
    ggml_free(ctx_w);
    ggml_free(ctx);
    return res;
}

// true if the independent mat-mul of reorder_mul_mat_add does not run between the mat-mul and the add
static bool add_packed(const std::vector<int> & pos) {
    return !(pos[0] < pos[2] && pos[2] < pos[1]);
}

static bool check_pack(const device_case & c) {
    const std::vector<int> pos = reorder_mul_mat_add(n_rows, ADD_RESIDUAL);
    const bool packed = add_packed(pos);

    const bool ok = packed == c.packed;
    std::printf("%s: mat-mul and add packed %d (expected %d): %s\n", c.name, packed, c.packed, ok ? "OK" : "FAIL");
    return ok;
}

static int run_pack_cases(const std::vector<device_case> & cases) {
    int failures = 0;
    for (const device_case & c : cases) {
        failures += check_pack(c) ? 0 : 1;
    }
    return failures;
}

// true if every row count in rows gives the reorder of the first one, on a device that fuses MUL_MAT+ADD
static bool same_reorder_for_rows(const std::vector<int64_t> & rows, add_operand_kind kind) {
    const std::vector<int> first = reorder_mul_mat_add(rows.front(), kind);
    for (auto r = rows.begin() + 1; r != rows.end(); ++r) {
        if (reorder_mul_mat_add(*r, kind) != first) {
            return false;
        }
    }
    return true;
}

// the pack must not depend on the batch size, or graphs with the same nodes get another allocation per ubatch size,
// and a bias add, which the encoder never fuses, must stay free to run next to independent nodes
static bool check_row_independent_pack(add_operand_kind kind) {
    const bool same = same_reorder_for_rows(batch_rows, kind);
    const bool packed = add_packed(reorder_mul_mat_add(batch_rows.front(), kind));
    const bool expected = kind == ADD_RESIDUAL;
    const bool ok = same && packed == expected;
    std::printf("MUL_MAT+ADD of a %s, reorder independent of src1 rows %d, packed %d (expected %d): %s\n",
        add_operand_name(kind), same, packed, expected, ok ? "OK" : "FAIL");
    return ok;
}

static int run_row_independent_pack_cases() {
    int failures = 0;
    for (add_operand_kind kind : { ADD_RESIDUAL, ADD_BIAS, ADD_BIAS_VIEW }) {
        failures += check_row_independent_pack(kind) ? 0 : 1;
    }
    return failures;
}

// true if the encoder's check fuses a few-row MUL_MAT with a same-shape residual, which is a weight if weight_res;
// the check compares Metal buffer ranges, so the tensors live in Metal buffers
static bool encoder_fuses_mul_mat_add(bool weight_res) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU));
    ggml_init_params params = { n_tensors*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx   = ggml_init(params);
    ggml_context * ctx_w = ggml_init(params);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, n_rows);
    ggml_tensor * mm  = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_k, n_m), x);
    ggml_tensor * add = ggml_add(ctx, mm, ggml_new_tensor_2d(weight_res ? ctx_w : ctx, GGML_TYPE_F32, n_m, n_rows));
    ggml_backend_buffer_t buffer  = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_init_params params_gf = { ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx_gf = ggml_init(params_gf);
    ggml_cgraph * gf = ggml_new_graph(ctx_gf);
    ggml_build_forward_expand(gf, add);

    const int idxs[] = { 0, 1 };
    int n_fused = 1;
    const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, 2, 0, GGML_METAL_FUSION_FULL, &n_fused);
    const bool fused = fusion != nullptr && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MUL_MAT_ADD;
    ggml_free(ctx_gf);

    ggml_backend_buffer_free(weights);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx_w);
    ggml_free(ctx);
    return fused;
}

// the encoder may fuse only what the reorder packs, so it must leave a same-shape residual alone when it is a weight
static bool check_encoder_skips_weight_residual() {
    const bool fuses_residual = encoder_fuses_mul_mat_add(false);
    const bool fuses_weight   = encoder_fuses_mul_mat_add(true);
    const bool ok = fuses_residual && !fuses_weight;
    std::printf("encoder fuses MUL_MAT+ADD of a residual %d (expected 1), of a weight %d (expected 0): %s\n",
        fuses_residual, fuses_weight, ok ? "OK" : "FAIL");
    return ok;
}

// true if the encoder runs the MoE routing (fused SOFT_MAX + ARGSORT + GET_ROWS + norm) concurrently with a pending
// node that reads r, when the argsort ids, which the fused kernel writes from a middle node, share r's memory if alias
static bool encoder_runs_topk_moe_concurrently(bool alias, bool * group_ends_free) {
    constexpr int64_t n_expert      = 8;
    constexpr int64_t n_expert_used = 2;
    constexpr int64_t n_tok         = 4;

    ggml_init_params params = { n_tensors*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);

    // the pending node reads r
    ggml_tensor * r       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tok);
    ggml_tensor * pending = ggml_scale(ctx, r, scale);

    // the routing as the MoE graph builds it
    ggml_tensor * logits  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tok);
    ggml_tensor * probs   = ggml_soft_max(ctx, logits);
    ggml_tensor * ids     = ggml_argsort_top_k(ctx, probs, n_expert_used);
    ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tok), ids);
    weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tok);
    ggml_tensor * sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_reshape_3d(ctx, ggml_div(ctx, weights, sum), 1, n_expert_used, n_tok);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, pending);
    ggml_build_forward_expand(gf, weights);

    // each tensor gets memory of its own, except that the argsort may take r's
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU));
    const size_t align = ggml_backend_buft_get_alignment(buft);
    std::vector<ggml_tensor *> owners;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src == nullptr) {
            owners.push_back(t);
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, owners.size()*GGML_PAD(ggml_nbytes(r), align) + align);
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buffer);
    size_t offs = 0;
    for (ggml_tensor * t : owners) {
        if (alias && t == ids->view_src) {
            continue;
        }
        GGML_ASSERT(ggml_nbytes(t) <= ggml_nbytes(r));
        ggml_backend_tensor_alloc(buffer, t, base + offs);
        offs += GGML_PAD(ggml_nbytes(r), align);
    }
    if (alias) {
        ggml_backend_tensor_alloc(buffer, ids->view_src, r->data);
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src != nullptr) {
            ggml_backend_view_init(t);
        }
    }

    // the nodes the encoder runs, the pending node first
    std::vector<int> idxs;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_op_is_empty(ggml_graph_node(gf, i)->op)) {
            idxs.push_back(i);
        }
    }
    GGML_ASSERT(ggml_graph_node(gf, idxs[0]) == pending);

    ggml_mem_ranges_t mrs = ggml_mem_ranges_init(0);
    ggml_mem_ranges_add(mrs, pending);

    int n_fused = 1;
    const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs.data(), (int) idxs.size(), 1, GGML_METAL_FUSION_FULL, &n_fused);
    GGML_ASSERT(fusion != nullptr && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE && n_fused == (int) idxs.size() - 1);

    *group_ends_free = ggml_mem_ranges_check(mrs, ggml_graph_node(gf, idxs[1])) &&
        ggml_mem_ranges_check(mrs, ggml_graph_node(gf, idxs.back()));
    const bool res = ggml_metal_fusion_concurrent(mrs, gf, idxs.data(), (int) idxs.size(), 1, true);

    ggml_mem_ranges_free(mrs);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return res;
}

// a fused kernel writes the dst of a middle node, so the group needs a barrier when that dst is still being read
static bool check_encoder_checks_fused_middle_nodes() {
    bool ends_free_alias = false;
    bool ends_free       = false;
    const bool concurrent_alias = encoder_runs_topk_moe_concurrently(true,  &ends_free_alias);
    const bool concurrent       = encoder_runs_topk_moe_concurrently(false, &ends_free);
    const bool ok = ends_free_alias && ends_free && !concurrent_alias && concurrent;
    std::printf("fused MoE routing next to a reader of the memory of its argsort ids: concurrent %d (expected 0), "
        "of other memory: concurrent %d (expected 1), first and last node free %d %d (expected 1 1): %s\n",
        concurrent_alias, concurrent, ends_free_alias, ends_free, ok ? "OK" : "FAIL");
    return ok;
}

static ggml_context * graph_ctx() {
    ggml_init_params params = { n_tensors*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    return ggml_init(params);
}

static void expand_all(ggml_cgraph * graph, const std::vector<ggml_tensor *> & outputs) {
    for (ggml_tensor * t : outputs) {
        ggml_build_forward_expand(graph, t);
    }
}

// the graph of outputs in build order, reordered for a device that fuses MUL_MAT+ADD
static ggml_cgraph * optimized_graph(ggml_context * ctx, const std::vector<ggml_tensor *> & outputs) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    expand_all(graph, outputs);
    ggml_graph_optimize(graph, false, 0, nullptr);
    return graph;
}

// the position of t among the nodes the encoder runs (views are skipped), -1 if absent
static int encoded_index(ggml_cgraph * graph, const ggml_tensor * t) {
    int res = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == t) {
            return res;
        }
        res += ggml_op_is_empty(node->op) ? 0 : 1;
    }
    return -1;
}

// true if the encoder runs the nodes back to back in this order, so they fuse
static bool encoded_in_a_row(ggml_cgraph * graph, const std::vector<ggml_tensor *> & nodes) {
    const int first = encoded_index(graph, nodes[0]);
    for (size_t j = 1; j < nodes.size(); ++j) {
        if (encoded_index(graph, nodes[j]) != first + (int) j) {
            return false;
        }
    }
    return true;
}

static bool report_reader(const char * name, bool packed, bool reader_after) {
    const bool ok = packed && reader_after;
    std::printf("%s: packed %d, reader after the write %d (expected 1, 1): %s\n", name, packed, reader_after, ok ? "OK" : "FAIL");
    return ok;
}

// x feeds a MUL_MAT+ADD pack chained with RMS_NORM+MUL, and c reads the residual sum h, an output inside the pack
static bool check_chained_pack_reader() {
    ggml_context * ctx = graph_ctx();
    ggml_tensor * x  = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, n_rows), scale);
    ggml_tensor * mm = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, n_m), x);
    ggml_tensor * h  = ggml_add(ctx, mm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_m, n_rows));
    ggml_tensor * n  = ggml_rms_norm(ctx, h, norm_eps);
    ggml_tensor * m  = ggml_mul(ctx, n, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_m));
    ggml_tensor * c  = ggml_scale(ctx, h, scale);

    ggml_cgraph * graph = optimized_graph(ctx, { m, c });
    const bool packed = encoded_in_a_row(graph, { mm, h, n, m });
    const bool after  = encoded_index(graph, c) > encoded_index(graph, h);
    ggml_free(ctx);

    return report_reader("reader of a chained MUL_MAT+ADD sum", packed, after);
}

// the boundary barriers of the graph reorder
//
// ggml_graph_optimize hoists memory-concurrent nodes forward to widen the concurrent set.
// A caller that installs a boundary-event schedule and an encode window cuts the graph at
// the per-layer output markers, and the reorder runs before that schedule exists, so left
// alone it can move a node out of the segment that is supposed to produce it. These tests
// pin both halves of the fix: with the barriers off the hoist still happens (the reorder
// is untouched), and with them on nothing crosses a marker.
//
// Every case is a hand-built graph: no device, no model, no allocation.

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
static ggml_cgraph * build_fused_group(ggml_context * ctx, const char * marker_name = "l_out-0") {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");
    ggml_tensor * w = leaf(ctx, "w");
    ggml_tensor * p = leaf(ctx, "p");
    ggml_tensor * q = leaf(ctx, "q");
    ggml_tensor * v = leaf(ctx, "v");

    ggml_tensor * a  = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * b  = named(ggml_mul(ctx, a, w), "b");
    ggml_tensor * s  = named(ggml_add(ctx, p, q), "s");
    ggml_tensor * mk = named(ggml_add(ctx, s, v), marker_name);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_build_forward_expand(gf, s);
    ggml_build_forward_expand(gf, mk);

    return gf;
}

// A router whose ops do not fuse, ending in the node a caller cuts at, followed by a lookup by
// expert ids that reads memory the host fills at that cut. The graph records no dependency
// between the lookup and the router, so the reorder is free to run the lookup first.
//
//   router_logits-0 -> mul_mat(w, x)
//   router_probs-0  -> scale(router_logits-0)   conflicts with router_logits-0
//   router_out-0    -> add(router_probs-0, x)   the cut node
//   expert_rows-0   -> get_rows(slots, ids)     independent of the router
//   expert_out-0    -> mul(expert_rows-0, router_out-0)
static ggml_cgraph * build_router_lookup(ggml_context * ctx) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * w = leaf(ctx, "w");

    ggml_tensor * logits = named(ggml_mul_mat(ctx, w, x), "router_logits-0");
    ggml_tensor * probs  = named(ggml_scale(ctx, logits, scale), "router_probs-0");
    ggml_tensor * out    = named(ggml_add(ctx, probs, x), "router_out-0");
    ggml_tensor * rows   = cache_lookup(ctx, "expert_rows-0");
    ggml_tensor * expert = named(ggml_mul(ctx, rows, out), "expert_out-0");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, probs);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, expert);

    return gf;
}

// The other direction: the cut node itself is independent of the node before it, so the reorder
// would run the cut first and leave a node that belongs before the cut behind it.
//
//   a     -> mul(x, y)
//   b     -> mul(a, w)        conflicts with a, belongs before the cut
//   cut-0 -> get_rows(...)    the cut node, independent
static ggml_cgraph * build_cut_hoisted(ggml_context * ctx) {
    ggml_tensor * x = leaf(ctx, "x");
    ggml_tensor * y = leaf(ctx, "y");
    ggml_tensor * w = leaf(ctx, "w");

    ggml_tensor * a   = named(ggml_mul(ctx, x, y), "a");
    ggml_tensor * b   = named(ggml_mul(ctx, a, w), "b");
    ggml_tensor * cut = cache_lookup(ctx, "cut-0");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);
    ggml_build_forward_expand(gf, cut);

    return gf;
}

// the cases of caller-registered barrier name prefixes; failures count into g_n_fail
static void run_barrier_prefix_cases() {
    const char * router_cut[] = { "router_out-" };
    const char * cut[]        = { "cut-" };
    const char * other[]      = { "nothing-", "router_out-1" };

    {
        test_ctx      tc;
        ggml_cgraph * gf = build_router_lookup(tc.ctx);

        check_order("router: graph as built", gf,
            { "router_logits-0", "router_probs-0", "router_out-0", "expert_rows-0", "expert_out-0" });
    }

    // positive control: without a barrier the lookup runs before the router's cut node
    {
        test_ctx      tc;
        ggml_cgraph * gf = build_router_lookup(tc.ctx);

        ggml_graph_optimize(gf, false, 0, nullptr);

        check_order("no prefix: lookup hoists above the cut", gf,
            { "router_logits-0", "expert_rows-0", "router_probs-0", "router_out-0", "expert_out-0" });
    }

    // the marker switch alone does not cover a cut at a node that is not a per-layer output
    {
        test_ctx      tc;
        ggml_cgraph * gf = build_router_lookup(tc.ctx);

        ggml_graph_optimize(gf, true, 0, nullptr);

        check_order("markers only: lookup hoists above the cut", gf,
            { "router_logits-0", "expert_rows-0", "router_probs-0", "router_out-0", "expert_out-0" });
    }

    // prefixes that match no node leave the reorder as it is
    {
        test_ctx      tc;
        ggml_cgraph * gf = build_router_lookup(tc.ctx);

        ggml_graph_optimize(gf, false, 2, other);

        check_order("unmatched prefixes: reorder unchanged", gf,
            { "router_logits-0", "expert_rows-0", "router_probs-0", "router_out-0", "expert_out-0" });
    }

    for (bool markers : { false, true }) {
        test_ctx      tc;
        ggml_cgraph * gf = build_router_lookup(tc.ctx);

        ggml_graph_optimize(gf, markers, 1, router_cut);

        check_order(markers ? "prefix + markers: lookup stays below" : "prefix: lookup stays below the cut", gf,
            { "router_logits-0", "router_probs-0", "router_out-0", "expert_rows-0", "expert_out-0" });
    }

    // the cut node itself is not hoisted above a node before it
    {
        test_ctx      tc;
        ggml_cgraph * gf = build_cut_hoisted(tc.ctx);

        ggml_graph_optimize(gf, false, 0, nullptr);

        check_order("no prefix: cut node hoists", gf, { "a", "cut-0", "b" });
    }

    {
        test_ctx      tc;
        ggml_cgraph * gf = build_cut_hoisted(tc.ctx);

        ggml_graph_optimize(gf, false, 1, cut);

        check_order("prefix: cut node holds", gf, { "a", "b", "cut-0" });
    }

    // the cut node opens the look-forward scan, which picks up nothing
    {
        test_ctx      tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "cut-0");

        ggml_graph_optimize(gf, false, 0, nullptr);

        check_order("no prefix: hoists across the cut", gf, { "a", "c", "b", "cut-0" });
    }

    {
        test_ctx      tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "cut-0");

        ggml_graph_optimize(gf, false, 1, cut);

        check_order("prefix: nothing crosses the cut", gf, { "a", "b", "cut-0", "c" });
    }

    // a cut node fused onto the head of a group: the group stays packed in both runs, and
    //   with the prefix it holds its place
    {
        test_ctx      tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx, "cut-0");

        ggml_graph_optimize(gf, false, 0, nullptr);

        check_order("no prefix: fused group hoists", gf, { "a", "s", "cut-0", "b" });
    }

    {
        test_ctx      tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx, "cut-0");

        ggml_graph_optimize(gf, false, 1, cut);

        check_order("prefix: fused group holds", gf, { "a", "b", "s", "cut-0" });
    }
}

static void ignore_alloc_dep(void * /*user_data*/, ggml_tensor * /*tensor*/, ggml_tensor * /*until*/) {}

// the prefixes a caller registers on the backend reach the reorder the scheduler runs, are
//   copied, and are gone once cleared
static void run_backend_prefix_cases() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend == nullptr || !ggml_backend_is_metal(backend)) {
        printf("FAIL %-40s no Metal backend\n", "backend prefixes");
        g_n_fail++;
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
        return;
    }

    ggml_backend_graph_optimize_params params = { ignore_alloc_dep, nullptr };

    const auto optimize = [&](const char * scenario, const std::vector<std::string> & expected) {
        test_ctx      tc;
        ggml_cgraph * gf = build_router_lookup(tc.ctx);

        backend->iface.graph_optimize(backend, gf, &params);

        check_order(scenario, gf, expected);
    };

    {
        // the strings die before the reorder runs
        std::string prefix = "router_out-";
        const char * prefixes[] = { prefix.c_str() };
        ggml_backend_metal_set_reorder_barrier_prefixes(backend, 1, prefixes);
        prefix.assign(prefix.size(), 'x');
    }

    optimize("backend prefix: lookup stays below", { "router_logits-0", "router_probs-0", "router_out-0", "expert_rows-0", "expert_out-0" });

    ggml_backend_metal_set_reorder_barrier_prefixes(backend, 0, nullptr);

    optimize("backend prefix cleared: lookup hoists", { "router_logits-0", "expert_rows-0", "router_probs-0", "router_out-0", "expert_out-0" });

    ggml_backend_free(backend);
}

// the boundary barrier cases, returns the number of failed checks
static int run_boundary_barrier_cases() {
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

        ggml_graph_optimize(gf, false, 0, nullptr);

        check_order("barriers off: hoists across marker", gf, { "a", "c", "b", "l_out-0" });
    }

    // With the barriers on, the same node stays in its own segment.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "l_out-0");

        ggml_graph_optimize(gf, true, 0, nullptr);

        check_order("barriers on: stops at marker", gf, { "a", "b", "l_out-0", "c" });
    }

    // The barrier keys on the marker, not on the switch: a node that is not a marker is
    // still hoisted with the barriers on, so turning them on is not a blanket "no reorder".
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_hoist_across(tc.ctx, "ffn_out-0");

        ggml_graph_optimize(gf, true, 0, nullptr);

        check_order("barriers on: non-marker still hoists", gf, { "a", "c", "b", "ffn_out-0" });
    }

    // A marker that is itself the node the scan starts from.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_marker_anchor(tc.ctx);

        ggml_graph_optimize(gf, false, 0, nullptr);

        check_order("barriers off: hoists in front of marker", gf, { "a", "c", "l_out-0" });
    }

    {
        test_ctx     tc;
        ggml_cgraph * gf = build_marker_anchor(tc.ctx);

        ggml_graph_optimize(gf, true, 0, nullptr);

        check_order("barriers on: marker takes nothing along", gf, { "a", "l_out-0", "c" });
    }

    // Fusion packs "s" and the marker into one group. With the barriers off the group is
    // hoisted whole, which moves the marker itself -- the two staying adjacent is what says
    // they were really fused rather than reordered independently.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx);

        ggml_graph_optimize(gf, false, 0, nullptr);

        check_order("barriers off: fused group hoists", gf, { "a", "s", "l_out-0", "b" });
    }

    // ... and with them on the group holds its place, because a group is a barrier when any
    // of its members is a marker, not only when its head is.
    {
        test_ctx     tc;
        ggml_cgraph * gf = build_fused_group(tc.ctx);

        ggml_graph_optimize(gf, true, 0, nullptr);

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

    run_barrier_prefix_cases();
    run_backend_prefix_cases();

    return g_n_fail;
}

int main() {
    const std::vector<device_case> devices = {
        { "native simdgroup matrices",       true,  true },
        { "probed or no simdgroup matrices", false, true },
    };
    const int failures = run_pack_cases(devices) + (check_chained_pack_reader() ? 0 : 1) + run_row_independent_pack_cases() +
        (check_encoder_skips_weight_residual() ? 0 : 1) + (check_encoder_checks_fused_middle_nodes() ? 0 : 1) +
        run_boundary_barrier_cases();

    return failures == 0 ? 0 : 1;
}
