#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_op * ggml_metal_op_t;

// [rrl] #135 Stage 2: returns non-zero iff tensor t is a per-expert mmap-metal
// weight tensor (the same gate rrl_is_expert_mmap_metal uses internally).
// Declared here so ggml-metal-context.m can drive the window planner using the
// same detection logic without duplicating it.
int rrl_is_expert_mmap_metal_c(const struct ggml_tensor *t);

// [rrl] PR2-A.2: async completion-handler evictor interface (C-callable from context.m).
//
// rrl_metal_cb_async_depth: returns the async in-flight depth D.  Default-ON:
//   D=2 when RRL_METAL_CB_ASYNC is UNSET; explicit =0 selects the synchronous
//   per-window waitUntilCompleted drain; a positive value selects that depth.
//
// rrl_async_window_take_records: move the accumulated async-records vector to a
//   heap allocation and return an opaque handle (NULL if nothing was accumulated).
//   Call once per window after the encode loop, before committing.
//
// rrl_async_window_reclaim_and_free: run rrl_evict_window_reclaim on each record
//   in the handle and free it.  Safe to call with NULL.  Call from the MTL
//   completion handler — the handler is the SOLE evictor in async mode.
int   rrl_metal_cb_async_depth(void);
// rrl_metal_set_cb_async_depth: crate-side override of D (#268). Sets a
//   process-global override consulted by rrl_metal_cb_async_depth before the
//   RRL_METAL_CB_ASYNC env value; pass < 0 to clear (fall back to env/default).
void  rrl_metal_set_cb_async_depth(int d);
void *rrl_async_window_take_records(void);
void  rrl_async_window_reclaim_and_free(void *handle);

// [rrl] PR2-B: W-expert sub-window entry point (C-callable from context.m).
//
// rrl_metal_cb_wexp: returns W from RRL_METAL_CB_WEXP env var, or 0 if unset.
//   Requires RRL_METAL_CB_ASYNC to also be set; context.m checks that condition.
//
// rrl_encode_expert_node_windows: encode one expert mul_mat_id node (node_idx in
//   gf) as ceil(N_routed/W) sub-CBs, each covering a W-expert group.  Called from
//   the async path in graph_compute when RRL_METAL_CB_WEXP is set and the current
//   singleton window is an expert node.  The entry owns all sem.wait() calls for
//   its sub-CBs; the caller must NOT acquire sem before calling.  Each sub-CB's
//   completion handler runs rrl_async_window_reclaim_and_free, releases the CB,
//   and signals sem.  Returns the number of sub-CBs committed (0 on fallback to the
//   single-CB path).
int rrl_metal_cb_wexp(void);
// rrl_metal_set_cb_wexp: crate-side override of W (#268). Process-global override
//   consulted by rrl_metal_cb_wexp before RRL_METAL_CB_WEXP; pass < 0 to clear.
void rrl_metal_set_cb_wexp(int w);
int rrl_encode_expert_node_windows(
        ggml_metal_device_t   dev,
        struct ggml_cgraph  * gf,
        int                   node_idx,
        void                * queue,          // id<MTLCommandQueue>, passed as void* (ObjC)
        void                * sem,            // dispatch_semaphore_t, passed as void*
        int                   D,
        int                   W,
        bool                * has_error_out); // set to true on CB error; may be NULL

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        struct ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion,
        int  rrl_cb_async_depth); // [rrl] #280: per-context async depth D snapshot

void ggml_metal_op_free(ggml_metal_op_t ctx);

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx);

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx);

//
// available ops:
//

// tokens per expert
size_t ggml_metal_op_mul_mat_id_extra_tpe(const struct ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t ggml_metal_op_mul_mat_id_extra_ids(const struct ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor * op);

size_t ggml_metal_op_flash_attn_ext_extra_pad(const struct ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_blk(const struct ggml_tensor * op);
size_t ggml_metal_op_flash_attn_ext_extra_tmp(const struct ggml_tensor * op);

int ggml_metal_op_concat            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_repeat            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_acc               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_unary             (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_glu               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_sum               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_sum_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_cumsum            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_get_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_set_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_diag              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_soft_max          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_ssm_conv          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_ssm_scan          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_rwkv              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_gated_delta_net   (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_solve_tri         (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_set               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_cpy               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pool_1d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pool_2d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat_id        (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_add_id            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_flash_attn_ext    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_bin               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_l2_norm           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_group_norm        (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_norm              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_rope              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_im2col            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_2d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_3d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_transpose_1d (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_transpose_2d (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_upscale           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pad               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pad_reflect_1d    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_roll              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_arange            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_argmax            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_argsort           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_top_k             (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_tri               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_opt_step_adamw    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_opt_step_sgd      (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_count_equal       (ggml_metal_op_t ctx, int idx);

#ifdef __cplusplus
}
#endif
