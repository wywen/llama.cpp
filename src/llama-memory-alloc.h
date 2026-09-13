#pragma once

#include "ggml-backend.h"

#include <cstddef>

struct ggml_context;
struct llama_hparams;

// Backs the storage-owning tensors of a memory module's ggml context with a buffer of type `buft`.
// When the model is loaded without allocation, returns a zero-size placeholder instead and points every
// tensor at it, so the backend scheduler does not try to allocate them either.
// Returns nullptr when the allocation fails.
ggml_backend_buffer_t llama_memory_alloc_buffer(
        ggml_context               * ctx,
        ggml_backend_buffer_type_t   buft,
        const llama_hparams        & hparams);

// Bytes the tensors of `ctx` occupy in `buf`; for a zero-size placeholder, the bytes a real allocation takes.
size_t llama_memory_buffer_size(ggml_context * ctx, ggml_backend_buffer_t buf);
