#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }

        // Synthetic constructor for per-expert assembled tensors.
        // Skips GGUF validation; offs=0 and idx=0 are placeholders (never used
        // for data access — load_all_data detects per_expert_assembled entries
        // and assembles them from per-expert slices instead).
        llama_tensor_weight(uint16_t idx_, size_t offs_, ggml_tensor * tensor_)
            : idx(idx_), offs(offs_), tensor(tensor_) {}
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    bool check_tensors;
    bool no_alloc;

    bool per_expert_moe = false;  // true when moe.layout == "per_expert" in this GGUF
    // Set to use_mmap for a per_expert model: skip the assembly memcpy and point the
    // fused tensor's data directly into the GGUF mmap (requires the page-aligned
    // trailing-expert layout). False on the CPU managed-expert path (use_mmap=false),
    // which keeps the memcpy assembly via the `use_mmap && per_expert_zerocopy` gate.
    bool per_expert_zerocopy = false;
    // Names of fused expert tensors synthesised from per-expert slices (need assembly at load time).
    std::unordered_set<std::string> per_expert_assembled;

    // [keep-separate Metal, step 1] Monotonic counter that gives every fused
    // expert tensor routed to the mmap-metal buft its OWN ggml_context — hence
    // its own MTLBuffer whose base is 16 KiB-aligned. Without this, multiple
    // expert tensors share one per-layer buffer and only the first lands on a
    // page boundary (Metal's mapped-buffer alignment is 32, not 16 KiB), so
    // experts of the 2nd+ tensor would be misaligned. Encoded into buft_layer_key
    // so each is a distinct map entry.
    int metal_expert_ctx_seq = 0;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // Key for the buft -> ctx map.  For buffer types that want per-layer contexts
    // (currently "mmap-metal"), layer holds the transformer block index so each layer
    // gets its own ggml_context — hence its own MTLBuffer — instead of one monolithic
    // allocation.  For all other bufts layer == -1 (shared context, historical behaviour).
    // Mirrors the buft_layer_key / buft_layer_comparator in llama-kv-cache.cpp.
    struct buft_layer_key {
        ggml_backend_buffer_type_t buft;
        int                        layer; // -1 = shared context (non-per-layer buft)
    };
    struct buft_layer_comparator {
        bool operator()(const buft_layer_key & lhs, const buft_layer_key & rhs) const {
            const int c = strcmp(ggml_backend_buft_name(lhs.buft), ggml_backend_buft_name(rhs.buft));
            if (c != 0) {
                return c < 0;
            }
            return lhs.layer < rhs.layer;
        }
    };

    std::map<buft_layer_key, ggml_context_ptr, buft_layer_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        bool use_mmap,
        bool use_direct_io,
        bool check_tensors,
        bool no_alloc,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    struct ggml_tensor * create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base, const std::string & name, const std::initializer_list<int64_t> & ne, size_t offset, bool required = true);

    void done_getting_tensors(bool partial = false) const;

    // Returns true if `name` is a per-expert source tensor (e.g. blk.N.ffn_XXX_exps.E.suffix).
    // These are never loaded directly; they are assembled into fused tensors at load time.
    bool is_per_expert_source(const std::string & name) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // [rrl #125] Register an expert mmap range with the crate's zero-copy region
    // hook so the mmap-metal shim can distinguish EXPERT buffers (→ metadata-only)
    // from GENERAL weight buffers (→ real-mapped) during the buffer phase, which is
    // required when KV is routed onto the same mmap-metal device as the experts.
    // No-op when the hook is not installed (non-Apple / crate not linked).
    void rrl_register_expert_region(const void * base, size_t size) const;

    // [rrl #201] Register a general (non-expert) per-layer weight buffer with the
    // crate's weight-tensor hook so the mmap-metal shim's weight roller can page it
    // in/out at barriers. No-op when the hook is not installed (non-Apple / crate
    // not linked / weight rolling off).
    void rrl_register_weight_tensor(const char * name, const void * base, size_t size, size_t file_offset, int fd) const;

    // for backwards compatibility, does not support ggml-backend
    void load_data_for(struct ggml_tensor * cur) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
