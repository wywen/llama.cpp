#include "llama-memory-alloc.h"

#include "llama-hparams.h"

#include "ggml-alloc.h"

static ggml_backend_buffer_t llama_memory_placeholder_buffer(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0);
    if (!buf) {
        return nullptr;
    }

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = buf;
    }

    return buf;
}

ggml_backend_buffer_t llama_memory_alloc_buffer(
        ggml_context               * ctx,
        ggml_backend_buffer_type_t   buft,
        const llama_hparams        & hparams) {
    if (hparams.no_alloc) {
        return llama_memory_placeholder_buffer(ctx, buft);
    }

    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
}

size_t llama_memory_buffer_size(ggml_context * ctx, ggml_backend_buffer_t buf) {
    // a real buffer holding no tensor bytes is also zero-sized, and the query agrees with it
    if (ggml_backend_buffer_get_size(buf) == 0) {
        return ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, ggml_backend_buffer_get_type(buf));
    }

    return ggml_backend_buffer_get_size(buf);
}
