#include "llama-ext.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-memory-alloc.h"
#include "llama-model.h"

#include <stdexcept>

llama_memory_plan llama_model_memory_plan(const llama_model * model, const llama_context_params & params) {
    if (!model) {
        throw std::runtime_error("model cannot be NULL");
    }

    const llama_context_params resolved = llama_context_params_resolve(*model, params);
    const llama_memory_cparams cparams = llama_cparams_memory_shape(*model, resolved);

    llama_memory_plan plan;
    plan.n_ctx       = cparams.n_ctx;
    plan.n_ctx_seq   = cparams.n_ctx_seq;
    plan.n_seq_max   = cparams.n_seq_max;
    plan.n_ubatch    = cparams.n_ubatch;
    plan.n_rs_seq    = cparams.n_rs_seq;
    plan.flash_attn  = cparams.flash_attn;
    plan.kv_unified  = cparams.kv_unified;
    plan.offload_kqv = cparams.offload_kqv;
    plan.has_memory  = false;

    if (model->hparams.vocab_only) {
        return plan;
    }

    LLAMA_LOG_INFO("%s: building the memory module with placeholder buffers, logged buffer sizes read zero\n", __func__);

    llama_memory_alloc_recorder recorder(llama_memory_alloc_mode::plan);
    {
        const llama_memory_ptr memory(llama_context_create_memory(*model, resolved, cparams));
        plan.has_memory = memory != nullptr;
    }
    plan.buffers = recorder.release();

    return plan;
}
