#pragma once

#include "ggml-metal.h"
#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

// The command buffer error that put this context into its error state, or NULL if it has not
// entered one. See ggml_backend_metal_last_error.
const char * ggml_metal_last_error(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

// See ggml_backend_metal_encode_nodes; the range is valid and cmd_buf is not NULL.
void ggml_metal_encode_nodes(ggml_metal_t ctx, struct ggml_cgraph * gf, int i0, int i1, ggml_metal_cmd_buf_t cmd_buf);

// See ggml_backend_metal_commit_fence.
void ggml_metal_commit_fence(ggml_metal_t ctx);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

// See ggml_backend_metal_set_reorder_barrier.
void ggml_metal_set_reorder_barrier(ggml_metal_t ctx, ggml_backend_metal_reorder_barrier_t is_barrier, void * user_data);

// See ggml_backend_metal_get_reorder_barrier.
ggml_backend_metal_reorder_barrier_t ggml_metal_get_reorder_barrier(ggml_metal_t ctx, void ** user_data);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
