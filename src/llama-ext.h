#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Reserves the widest prompt graph, the one a scheduler reserve sizes the compute buffers for last.
// The result is valid as llama_graph_reserve's.
LLAMA_API struct ggml_cgraph * llama_graph_reserve_widest(struct llama_context * ctx);

// Whether the next decode reserves the scheduler again because a setting changed (a backend sampler,
// an adapter, the attention type, ...); that reserve may reallocate the compute buffers.
LLAMA_API bool llama_sched_reserve_pending(const struct llama_context * ctx);

// Runs a pending scheduler reserve now instead of at the next decode; does nothing when none is pending.
// Returns false when the reserve failed: discard the context.
LLAMA_API bool llama_sched_reserve_now(struct llama_context * ctx);

// Reserves the output buffer for n_outputs rows, after the embedding and backend sampler setup and before decoding.
// Rejects 0 or more than n_batch or n_outputs_max rows without changing the context. Returns false for a valid
// request when the allocation failed: discard the context, its previous output buffer may have been released.
LLAMA_API bool llama_output_reserve(struct llama_context * ctx, uint32_t n_outputs);

// The output buffer, valid until it grows or the context is freed.
LLAMA_API ggml_backend_buffer_t llama_get_output_buffer(const struct llama_context * ctx);

struct llama_fused_node_info {
    const struct ggml_tensor * tensor;
    int32_t layer;
};

// Copies the flash-attention nodes of the last decoded graph and their layers, at most `capacity` entries, and
// returns their total count. The tensors are valid until the next decode or graph reset.
LLAMA_API size_t llama_get_fused_node_info(
    const struct llama_context * ctx,
    struct llama_fused_node_info * out,
    size_t capacity);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

//
// memory planning
//

// the memory component a planned buffer backs
enum llama_memory_plan_buffer_kind {
    LLAMA_MEMORY_PLAN_BUFFER_KV,         // attention KV cache without a sliding window
    LLAMA_MEMORY_PLAN_BUFFER_KV_SWA,     // attention KV cache that masks attention to a sliding window; the tensor shapes give its cell count
    LLAMA_MEMORY_PLAN_BUFFER_RECURRENT,  // recurrent state
    LLAMA_MEMORY_PLAN_BUFFER_DSV4_STATE, // DeepSeek V4 compressor state
    LLAMA_MEMORY_PLAN_BUFFER_KV_ROT,     // attention rotation matrices of a KV cache
};

struct llama_memory_plan_tensor {
    std::string name;                // e.g. cache_k_l3, cache_idx_k_l3, cache_r_l0, dsv4_csa_state_kv_l2, cache_k_rot_l3
    int32_t     il;                  // layer from the trailing _l<N> of the name; -1 when it has none
    ggml_type   type;
    int64_t     ne[GGML_MAX_DIMS];
    size_t      nb[GGML_MAX_DIMS];
    size_t      nbytes;              // ggml_nbytes, before buffer alignment
};

struct llama_memory_plan_buffer {
    llama_memory_plan_buffer_kind kind;
    ggml_backend_buffer_type_t    buft;
    // the sum of the tensor sizes, each padded to the buffer type's alignment, as memory_breakdown reports;
    // a backend may round the allocation up further
    size_t                        size;
    std::vector<llama_memory_plan_tensor> tensors; // the tensors owning storage, in allocation order
};

// the memory a context would build for a set of context parameters
struct llama_memory_plan {
    // the resolved shape the memory is built with
    uint32_t n_ctx;
    uint32_t n_ctx_seq;
    uint32_t n_seq_max;
    uint32_t n_ubatch;
    uint32_t n_rs_seq;
    bool     flash_attn;  // as the memory is built; automatic flash attention counts as enabled
    bool     kv_unified;
    bool     offload_kqv;

    bool has_memory; // false when the architecture keeps no memory, or the model is vocab-only

    std::vector<llama_memory_plan_buffer> buffers; // in allocation order
};

// Builds the memory that a context created from `params` would own, recording its buffer allocations instead of
// making them, then destroys it. Allocates no tensor data (the host-side cell metadata is still built).
// Throws std::runtime_error for a null model or for parameters that context creation refuses; later failures of
// context creation (backends, samplers, output and compute buffers) are not checked.
LLAMA_API llama_memory_plan llama_model_memory_plan(const struct llama_model * model, const llama_context_params & params);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);
