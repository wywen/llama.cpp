#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

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

//
// [keep-separate Metal, step 1] per-expert page-alignment probe
//
// Diagnostic for the keep-separate per-expert loader: walks the loaded model's
// ffn_*_exps.weight tensors and, treating each expert as starting at
// tensor->data + e*nb[ndims-1] (the per-expert stride), reports whether every
// expert base is 16 KiB-aligned relative to `region_base`, the total raw expert
// bytes vs total strided (padded) bytes, and the per-expert buffer count.
//
// It also spot-checks bytes: for the FIRST matched weight tensor it copies the
// first `spot_n` bytes of expert 0 into spot0_out and of expert (n_exp-1) into
// spotN_out (caller-allocated, each >= spot_n). These let the test compare the
// region bytes against the GGUF source.
//
// Returns the number of per-expert buffers (slices) examined, or a negative
// value on error (e.g. null model). All out-params are optional (may be null).
struct rrl_per_expert_probe_out {
    int32_t n_fused_tensors;   // # of ffn_*_exps.weight tensors matched
    int32_t n_expert_buffers;  // total per-expert slices (sum of n_exp over matched)
    int32_t all_page_aligned;  // 1 if every expert base is 16 KiB-aligned
    int32_t n_misaligned;      // count of experts whose base is not 16 KiB-aligned
    uint64_t total_raw_bytes;  // sum of raw per-expert bytes (ggml_nbytes of a slice)
    uint64_t total_stride_bytes; // sum of strided per-expert bytes (n_exp*stride)
    uint64_t first_stride;     // per-expert stride of the first matched tensor
    uint64_t first_raw;        // raw per-expert bytes of the first matched tensor
    int64_t  first_n_exp;      // n_exp of the first matched tensor
    uint64_t first_base_off;   // (first tensor data - region_base) for expert 0
};

LLAMA_API int32_t rrl_per_expert_layout_probe(
        const struct llama_model * model,
        uint64_t region_base,
        struct rrl_per_expert_probe_out * out,
        uint8_t * spot0_out,
        uint8_t * spotN_out,
        size_t    spot_n);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

//
// pre-norm embeddings (hidden state before the final output norm)
//

// Set whether the context outputs pre-norm embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_pre_norm    (struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);
