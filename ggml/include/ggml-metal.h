// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

// Paged-decode boundary-event schedule (opaque event handle).
//
// A quantized KV cache pages by evicting/admitting per-layer regions mid-decode.
// Fencing those with GPU events (instead of a whole-graph host synchronize)
// lets the decode graph run as one fused stream while a host thread reclaims and
// prefetches KV in the background. `ggml_metal_event_t` is created on the
// backend's device and shared between the GPU-side encode (signal/wait injected
// by the schedule) and the host reclaim/prefetch threads.
typedef struct ggml_metal_event * ggml_metal_event_t;

GGML_BACKEND_API ggml_metal_event_t ggml_backend_metal_event_init(ggml_backend_t backend);
GGML_BACKEND_API void               ggml_backend_metal_event_free(ggml_backend_t backend, ggml_metal_event_t ev);

// Host-side event ops (no backend needed -- the event owns its device queue).
// `host_signal` publishes a value a GPU `wait` is blocked on (prefetch done);
// `host_wait` blocks until the GPU signals `value` (eviction's last read done),
// returning false on timeout; `host_value` reads the current signalled value.
GGML_BACKEND_API void     ggml_backend_metal_event_host_signal(ggml_metal_event_t ev, uint64_t value);
GGML_BACKEND_API bool     ggml_backend_metal_event_host_wait  (ggml_metal_event_t ev, uint64_t value, uint64_t timeout_ms);
GGML_BACKEND_API uint64_t ggml_backend_metal_event_host_value (ggml_metal_event_t ev);

// Install the persistent, pointer-keyed boundary schedule on `backend`'s Metal
// context. It applies to every subsequent graph compute until cleared (call
// with n_cuts == 0). The graph is cut into a command buffer after each
// `cut_nodes[i]` (emitting a completion signal on `sig_ev[i]` at `sig_val[i]`
// when non-NULL) and a command buffer boundary is opened before each
// `wait_nodes[j]` (emitting a GPU wait on `wait_ev[j]` for `wait_val[j]` when
// non-NULL). NULL event entries make a boundary a pure segmentation point (no
// fence) -- an unarmed decode installs the same cut set with all-NULL events so
// its segmentation matches the armed decode bit-for-bit.
GGML_BACKEND_API void ggml_backend_metal_set_boundary_schedule(
        ggml_backend_t backend,
        int n_cuts,  struct ggml_tensor * const * cut_nodes,  ggml_metal_event_t * sig_ev,  const uint64_t * sig_val,
        int n_waits, struct ggml_tensor * const * wait_nodes, ggml_metal_event_t * wait_ev, const uint64_t * wait_val);

#ifdef __cplusplus
}
#endif
