#pragma once

// [MTP/EAGLE3 shared-KV] Helper for backing "dead" shared KV tensors so a
// shared-KV draft context can reserve its graph. A shared-KV cache (other !=
// null) reuses the target's KV tensors; on the bounded rolling-KV eviction path
// the residency manager keeps a dead layer's tensor->buffer null and re-points
// the real buffer just-in-time before each of the target's own decodes. But the
// draft's graph_reserve runs at build time, pre-decode, and a leaf with a null
// buffer makes the backend scheduler assign buffer_id = -1 and abort. A size-0
// dummy buffer (ggml returns a valid handle for size 0) makes reserve treat such
// a tensor as externally allocated; the scheduler caches that assignment and the
// target's next recreate_layer restores the real buffer before any GPU read.
//
// Kept in a header (free of llama internals) so the kv-cache constructor and the
// crate's pure-C++ unit test share one definition (shared_kv_backing_test.cpp).

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <vector>

// Back every null-buffer tensor in `tensors` with a single shared size-0 dummy
// buffer on `buft`, created lazily into `dummy` (which owns it). Tensors that
// already have a buffer are left untouched. Returns the number backed.
inline int rrl_back_shared_dead_kv(const std::vector<ggml_tensor *> & tensors,
                                   ggml_backend_buffer_type_t         buft,
                                   ggml_backend_buffer_ptr &          dummy) {
    int n_backed = 0;
    for (ggml_tensor * t : tensors) {
        if (t == nullptr || t->buffer != nullptr) {
            continue;
        }
        if (!dummy) {
            dummy.reset(ggml_backend_buft_alloc_buffer(buft, 0));
        }
        t->buffer = dummy.get();
        ++n_backed;
    }
    return n_backed;
}
