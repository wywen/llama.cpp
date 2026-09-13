#include "llama-memory-alloc.h"

#include "llama-hparams.h"

#include "ggml-alloc.h"

#include <cstdlib>
#include <cstring>

static thread_local llama_memory_alloc_recorder * memory_alloc_recorder = nullptr;

llama_memory_alloc_recorder::llama_memory_alloc_recorder(llama_memory_alloc_mode mode) :
    mode(mode), outer(memory_alloc_recorder) {
    memory_alloc_recorder = this;
}

llama_memory_alloc_recorder::~llama_memory_alloc_recorder() {
    memory_alloc_recorder = outer;
}

static int32_t llama_memory_tensor_layer(const char * name) {
    const char * sep = strrchr(name, '_');
    if (sep == nullptr || sep[1] != 'l' || sep[2] == '\0') {
        return -1;
    }

    char * end = nullptr;
    const long il = strtol(sep + 2, &end, 10);
    if (*end != '\0' || il < 0) {
        return -1;
    }

    return (int32_t) il;
}

static std::vector<llama_memory_plan_tensor> llama_memory_storage_tensors(ggml_context * ctx) {
    std::vector<llama_memory_plan_tensor> res;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        // the same tensors ggml_backend_alloc_ctx_tensors_from_buft places in the buffer
        if (t->data != nullptr || t->view_src != nullptr) {
            continue;
        }

        llama_memory_plan_tensor & rec = res.emplace_back();
        rec.name   = ggml_get_name(t);
        rec.il     = llama_memory_tensor_layer(rec.name.c_str());
        rec.type   = t->type;
        rec.nbytes = ggml_nbytes(t);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            rec.ne[i] = t->ne[i];
            rec.nb[i] = t->nb[i];
        }
    }

    return res;
}

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
        ggml_context                  * ctx,
        ggml_backend_buffer_type_t      buft,
        const llama_hparams           & hparams,
        llama_memory_plan_buffer_kind   kind) {
    llama_memory_alloc_recorder * recorder = memory_alloc_recorder;
    if (recorder == nullptr) {
        return hparams.no_alloc ? llama_memory_placeholder_buffer(ctx, buft) : ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    }

    // the storage-owning tensors are only distinguishable before allocation assigns their data
    std::vector<llama_memory_plan_tensor> tensors = llama_memory_storage_tensors(ctx);

    const bool placeholder = hparams.no_alloc || recorder->mode == llama_memory_alloc_mode::plan;

    ggml_backend_buffer_t buf = placeholder ? llama_memory_placeholder_buffer(ctx, buft) : ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        return nullptr;
    }

    recorder->recorded.push_back({
        /*.kind    =*/ kind,
        /*.buft    =*/ buft,
        /*.size    =*/ llama_memory_buffer_size(ctx, buf),
        /*.tensors =*/ std::move(tensors),
    });

    return buf;
}

size_t llama_memory_buffer_size(ggml_context * ctx, ggml_backend_buffer_t buf) {
    // a real buffer holding no tensor bytes is also zero-sized, and the query agrees with it
    if (ggml_backend_buffer_get_size(buf) == 0) {
        return ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, ggml_backend_buffer_get_type(buf));
    }

    return ggml_backend_buffer_get_size(buf);
}
