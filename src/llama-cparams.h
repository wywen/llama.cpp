#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

// Parameters shared by context construction and memory planning.
struct llama_memory_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback

    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool kv_unified;

    enum llama_context_type ctx_type;

    // the context whose model weights or results this context reads, if any
    llama_context * ctx_other;
    // the memory is a view of ctx_other's memory (its cells, and its layers where shared), not one of its own
    bool mem_shares_other;
};

struct llama_cparams : llama_memory_cparams {
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    uint32_t graph_variant = 0; // the caller's value that the graph is built for, see llama_set_graph_variant

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0

    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool fused_top_k_mask;   // use fused top-k sparse attention mask
    bool auto_ftkm;
    bool fused_kv_rows;      // unified cache: attend to the cells of each sequence only (ggml_flash_attn_ext_rows)
    bool auto_fkvr;
    bool no_perf;
    bool op_offload;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()

    size_t moe_cache_size;
    float  moe_cache_cpu_share;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer
    bool embeddings_layer_inp_dev; // write enabled layer inputs to a device buffer instead of host
    bool embeddings_nextn_dev;     // write the unmasked h_nextn rows to a device buffer instead of host, see llama_set_embeddings_nextn_dev()

    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // context whose h_nextn device buffer holds the state rows of a batch, see llama_set_ctx_state_src()
    llama_context * ctx_state_src;

    bool (*cb_sched_new)(ggml_backend_sched_t sched, llama_context * ctx, void * user_data);
    void * cb_sched_new_user_data;
    ggml_backend_buffer_type_t (*cb_compute_buft)(ggml_backend_t backend, ggml_backend_buffer_type_t buft, void * user_data);
    void * cb_compute_buft_user_data;
    uint32_t n_graph_nodes_extra;
    void (*cb_outputs_enqueued)(ggml_backend_sched_t sched, void * user_data);
    void * cb_outputs_enqueued_user_data;
};
