#pragma once

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

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

// The device backing this context -- the ONLY device whose events/queue this
// context's graph_compute uses. Callers that create ggml_metal_event_t for the
// #226 boundary schedule MUST init them on THIS device (ggml_metal_device_get()
// returns a fresh, unrelated device each call), or a host signal never reaches
// the waiting queue.
ggml_metal_device_t ggml_metal_get_device(ggml_metal_t ctx);

// #226 paged-decode boundary-event schedule. PERSISTENT: once set with
// n_cuts > 0, it applies to EVERY subsequent ggml_metal_graph_compute (NOT
// auto-cleared) until replaced or cleared with n_cuts == 0. Nodes are matched
// by POINTER, not index: a single llama_decode may compute the graph as several
// ggml_backend_sched SPLITS, each a graph_compute over a subset whose node
// INDICES differ from the full graph's; a persistent, pointer-keyed schedule
// lets each split cut/signal/wait at whichever of its own nodes appear in the
// schedule, and stays valid across the reused graph of later tokens. The CALLER
// clears it (n_cuts == 0) after the decode's whole split sequence -- e.g. right
// after llama_decode returns -- so unrelated computes (prefill, the cmd-buffer
// flush graph) take the stock path.
//
// When active, a graph_compute runs a SEQUENTIAL, per-boundary-committed encode
// instead of the parallel n_cb partition:
//   * the graph is cut into command buffers after each `cut_nodes[i]` that
//     appears in this graph, and at each `wait_nodes[j]` that appears (so a wait
//     aligns to a command-buffer start);
//   * after the command buffer ending at `cut_nodes[i]`, when `sig_ev[i]` != NULL
//     it emits encodeSignalEvent(sig_ev[i], sig_val[i]) -- a host reclaim thread
//     waits that value (ggml_metal_event_host_wait) then frees that boundary's
//     just-read KV buffers, with no decode-thread GPU drain;
//   * before the command buffer beginning at `wait_nodes[j]`, when `wait_ev[j]`
//     != NULL it emits encodeWaitForEvent(wait_ev[j], wait_val[j]) -- the GPU
//     stalls there until a prefetch thread has admitted (recreated + pread) that
//     region and host-signalled the value.
// A graph containing NONE of the schedule's nodes encodes as a single segment
// (functionally the stock result, just via the sequential path). All arrays are
// COPIED; the caller need not keep them alive past the call. cut_nodes /
// wait_nodes need not be sorted (matched by pointer). Duplicate node pointers in
// wait_nodes (e.g. a layer's K and V regions sharing one first-reader node) are
// all honoured.
void ggml_metal_set_boundary_schedule(
        ggml_metal_t ctx,
        int n_cuts,  struct ggml_tensor * const * cut_nodes,  ggml_metal_event_t * sig_ev, const uint64_t * sig_val,
        int n_waits, struct ggml_tensor * const * wait_nodes, ggml_metal_event_t * wait_ev, const uint64_t * wait_val);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
