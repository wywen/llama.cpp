#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"

// the classifier head of a user-defined embedding model is dropped by hand, and the sensitivity check of the
// generated models corrupts the memory module's caches and states directly
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-dsa.h"
#include "../src/llama-kv-cache-dsa-iswa.h"
#include "../src/llama-kv-cache-dsv4.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-kv-cache-msa.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
// values masked to the same infinity on both sides (e.g. logits padded with -inf) are skipped, any other non-finite value
// gives infinity, and all-zero references give NaN
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        if (!std::isfinite(a_i) || !std::isfinite(b_i)) {
            if (std::isinf(a_i) && a_i == b_i) {
                continue;
            }
            return INFINITY;
        }

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct tensor_data_params {
    size_t   seed;
    float    stdev;     // of the normal pool
    bool     sensitive; // see set_tensor_data
    llm_arch arch;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a pool of normal draws, generated once and reused for every model
static std::vector<float> g_normal_pool;
static const size_t NORMAL_POOL_SIZE = 16 * 1024 * 1024;

// fill the pool once; each tensor copies a slice of it
static void init_normal_pool(const size_t seed, const float stdev) {
    g_normal_pool.resize(NORMAL_POOL_SIZE);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);
    for (float & v : g_normal_pool) {
        v = dist(gen);
    }
}

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params &) {
    const size_t n = dst.size();
    const size_t pool_size = g_normal_pool.size();
    const size_t start = std::hash<std::string>{}(key) % pool_size;

    // copy a cyclic slice of the pool, so a tensor larger than the pool still fills
    size_t i = 0;
    size_t pos = start;
    while (i < n) {
        const size_t chunk = std::min(pool_size - pos, n - i);
        std::memcpy(dst.data() + i, g_normal_pool.data() + pos, chunk * sizeof(float));
        i += chunk;
        pos = 0;
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// RoPE factors divide the frequencies, a value close to 0 gives angles too large for F32 sin/cos to stay accurate
static void fill_rope_factors(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = 1.0f + fabsf(v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "ssm_a",        fill_ssm_a        },
    { "_resid_",      fill_resid_gate   },
    { "exp_probs_b",  fill_exp_probs_b  },
    { "rope_freqs",   fill_rope_factors },
    { "rope_factors", fill_rope_factors },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);

    // sensitive: scales and norm weights are near 1 and matrices keep the variance of their input, other tensors keep the pool's
    // with small values everywhere the product of scales, norms and matrices hides most inputs and cached state from the output
    // scales and norm weights are 1 + N(0, 0.01) rather than 1, so that dropping an optional one is not the identity
    if (fill == fill_normal) {
        const std::string name = tensor->name;
        // gemma4 multiplies the whole residual stream by each layer's output scale: at the pool's stdev the stream vanishes
        // with depth and the logits underflow to zero, with either set of weights
        const bool stream_scale = params.arch == LLM_ARCH_GEMMA4 && name.find(".layer_output_scale.") != std::string::npos;
        // activation scales (.scales) divide the activation: near 0 a few units dominate the output whatever the context
        const bool near_one = stream_scale ||
            (params.sensitive && (ends_with(name, ".scale") || ends_with(name, ".scales") ||
                (name.find("norm") != std::string::npos && ends_with(name, ".weight"))));
        if (near_one) {
            for (float & v : tmp) {
                v = 1.0f + v * (1.0e-2f / params.stdev);
            }
        } else if (params.sensitive && ends_with(name, ".weight") && ggml_n_dims(tensor) >= 2) {
            const float scale = 1.0f / (std::sqrt((float) tensor->ne[0]) * params.stdev);
            for (float & v : tmp) {
                v *= scale;
            }
        }
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --kv-shift               Check K-shift against a decode at the shifted positions (CPU only)\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

// number of experts of an arch's MoE model; gemma4 gets enough for a partial expert cache to matter
static uint32_t moe_expert_count(const llm_arch arch) {
    if (host_experts_test(arch)) {
        return 64;
    }
    return arch == LLM_ARCH_GEMMA4 ? 8 : arch == LLM_ARCH_CLOVER_MOE ? 4 : 2;
}

// writes arch-resolved GGUF keys into the fixture context, the test-side equivalent of llama_model_saver
struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("fatal error");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

// iswa_only: gemma4 with interleaved sliding window attention but without per-layer embeddings or shared KV layers
// n_layer_min: the fewest layers the model gets
// n_layer_nextn: the NextN (MTP) layers appended after the trunk layers
static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const enum llama_pooling_type pooling, const bool iswa_only = false,
        const uint32_t n_layer_min = 0, const uint32_t n_layer_nextn = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4) {
        // TODO: expand coverage to both GQA and MQA (see https://github.com/ggml-org/llama.cpp/pull/31401#pullrequestreview-5710165130)
        n_head = 4;
        if (iswa_only) {
            n_layer = 12; // two periods of 5 sliding window layers and 1 full attention layer, as in the Gemma 4 12B layer types
        } else {
            n_layer = 10; // last 5 layers reuse KV, both SWA and full attention
        }
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_LFM3) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_QWEN4EXP) {
        n_layer = 4; // two linear attention layers and two QSA layers, so the PLE layer can follow another
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, Engram, ratio-2 and ratio-1 sources, top-k reuse and a candidate-masked index layer
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_RICURSIVE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_LIMITE) {
        n_head  = 4;
        n_layer = 4; // dense connections on the last two layers
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_XING4_0) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_MISTRAL_LARGE4) {
            n_layer = 4; // cover NoPE, shared indexer and KV sharing layers
        }
    } else if (arch == LLM_ARCH_GLM5_NEXT) {
        // several heads so that -sm tensor can split them and the MLA flash attention has GQA > 1
        n_embd = 128;
        n_head = 16;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_METIS16) {
        n_layer = 4; // 2 passes over 2 physical layers, one attention and one Mamba-2
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE || arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_RICURSIVE) {
        n_head_kv = 2; // GQA coverage
    }
    if (arch == LLM_ARCH_BEAM) {
        n_head    = 4;
        n_head_kv = 2;
        n_layer   = 4;
    }
    n_layer = std::max(n_layer, n_layer_min);
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer + n_layer_nextn);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    if (n_layer_nextn > 0) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, n_layer_nextn);
    }

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_LFM3 ||
            arch == LLM_ARCH_KIMI_LINEAR || arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_KIMI_K31 ||
            arch == LLM_ARCH_GLM5_NEXT
            ) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
            // the MLA cache holds one latent head
            for (uint32_t & n : n_head_per_layer) {
                n = std::min(n, uint32_t(1));
            }
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_GEMMA4 && iswa_only) {
        // full attention layers have a single KV head, sliding window layers one per head: with the wider full attention
        // heads, the full attention KV rows are half as wide as the sliding window ones
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il % 6 < 5 ? n_head : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else if (arch == LLM_ARCH_METIS16) {
        // the layer types repeat in every pass
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il % 2 == 0 ? n_head_kv : 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_XING4_0) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    if (arch == LLM_ARCH_SENSENOVA_U1) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SPATIAL, 10000.0f);
    }

    if (arch == LLM_ARCH_XING4_0) {
        // Manifold-Constrained Hyper-Connections (MHC) multi-residual-stream blocks
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(20));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1e-6f);
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      iswa_only ? uint32_t(0) : n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      iswa_only ? uint32_t(0) : uint32_t(5));
        // full attention layers have wider heads than sliding window layers
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,            2*n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,          2*n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention as in Gemma 4 E2B, every 6th with iswa_only as in Gemma 4 12B
        const uint32_t period = iswa_only ? 6 : 5;
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back((il + 1) % period != 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_BEAM || arch == LLM_ARCH_RICURSIVE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN, pattern);
        }
    } else if (arch == LLM_ARCH_LFM3) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>({0, 0, 1}));
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, false);
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer + n_layer_nextn, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer; qwen4exp's
        // is the second one, after a layer of each kind, as a real model's PLE layer follows others
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ arch == LLM_ARCH_QWEN4EXP ? 2u : 0u }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_MISTRAL_LARGE4) {
        // layer 3 reads the KV cache of layer 2, both are NoPE
        ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN,     std::vector<uint32_t>({1, 1, 0, 0}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES,    std::vector<uint32_t>({1, 0, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_MAP,    std::vector<uint32_t>({0, 1, 2, 2}));
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,    30.0f);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 2, 2, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,            std::vector<uint32_t>({2, 4}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,       std::vector<uint32_t>({2, 4, 5}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({1, 3}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(3));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   uint32_t(64));
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({ 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                               67716810739261ull, 51510806800915ull, 30921347202721ull }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
        ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,             token_map);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    if (arch == LLM_ARCH_RICURSIVE) {
        std::vector<float> swiglu_clamp;
        swiglu_clamp.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            // the random weight scales keep the FFN activations small, a small limit still clamps so the roundtrip checks the per-layer values
            swiglu_clamp.push_back(il % 3 == 1 ? 0.0f : 1e-3f);
        }
        ms.add_kv(LLM_KV_EMBEDDING_SCALE, 16.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP,    swiglu_clamp);
    }

    if (arch == LLM_ARCH_METIS16) {
        ms.add_kv(LLM_KV_METIS_PASS_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,         uint32_t(4));
        ms.add_kv(LLM_KV_METIS_MEMORY_LENGTH,            uint32_t(64));
        ms.add_kv(LLM_KV_METIS_ROUTE_LENGTH,             uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_PASS_LORA_RANK,       uint32_t(8));
        ms.add_kv(LLM_KV_METIS_CONTINUATION_LOGIT_LIMIT, 4.0f);
        ms.add_kv(LLM_KV_SSM_NORM_EPS,                   1e-5f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,                n_embd/2);
    }

    if (arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_ATTENTION_SCALE,             0.1f);
        ms.add_kv(LLM_KV_ATTENTION_XSA_EPS,           1e-4f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SIGMOID_CAPPING, std::vector<float>({23.0f, 5.0f, 7.5f}));
        ms.add_kv(LLM_KV_MUDD_FEED_FORWARD_LENGTH,    uint32_t(8));
        ms.add_kv(LLM_KV_MUDD_TAP_COUNT,              uint32_t(2));
        ms.add_kv(LLM_KV_MUDD_TAP_INDICES,            std::vector<int32_t>({-1, -1, -1, -1, 0, 2, 1, 3}));
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }
    if (pooling != LLAMA_POOLING_TYPE_UNSPECIFIED) {
        ms.add_kv(LLM_KV_POOLING_TYPE,               uint32_t(pooling));
        ms.add_kv(LLM_KV_ATTENTION_CAUSAL,           false);
        ms.add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, uint32_t(2));
        if (pooling == LLAMA_POOLING_TYPE_RANK) {
            ms.add_kv(LLM_KV_CLASSIFIER_OUTPUT_LABELS, std::vector<std::string>({"LABEL_0"}));
        }
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               moe_expert_count(arch));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            case LLM_ARCH_MELLUM:
                gating_func = 1;
                break;
            case LLM_ARCH_DEEPSEEK4:
            case LLM_ARCH_DEEPSEEK41:
                gating_func = 4;
                break;
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        }
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }

    if (arch == LLM_ARCH_BEAM) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_CLOVER_MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        std::vector<float> clamp_exp;
        std::vector<float> clamp_shexp;
        for (uint32_t il = 0; il < n_layer; il++) {
            clamp_exp  .push_back(il % 2 ? 4.0f : 0.0f);
            clamp_shexp.push_back(il % 2 ? 5.0f : 0.0f);
        }
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   clamp_exp);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, clamp_shexp);
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-llama-archs-" + std::to_string(++id) + ".gguf")).string();
}

// the expert selections of a decode, observed through the eval callback: a near-tie in a router's scores can select
//     other experts when the batch is split differently, which changes everything after that token through the memory
struct router_observer {
    struct row {
        std::vector<int32_t> ids;    // the selected experts or expert groups, sorted
        float                margin; // between the last selected and the first other score, relative to the largest
    };

    bool active = false;
    // by node name (selection and layer), the rows of each graph that computed it, in decode order
    // a layer can select from two expert sets under one name: every graph computes both, one after the other
    std::map<std::string, std::vector<std::vector<row>>> routers;
};

// the rows of each selection named 'name' over the tokens of a decode, from the rows of the graphs in decode order,
//     or nothing when they do not cover the tokens
static std::map<std::string, std::vector<router_observer::row>> router_rows(const router_observer & obs, uint32_t n_tokens) {
    std::map<std::string, std::vector<router_observer::row>> res;
    for (const auto & [name, chunks] : obs.routers) {
        size_t n_rows = 0;
        for (const auto & chunk : chunks) {
            n_rows += chunk.size();
        }
        const size_t n_per_graph = n_rows/n_tokens;
        if (n_per_graph == 0 || n_rows % n_tokens != 0 || chunks.size() % n_per_graph != 0) {
            return {};
        }
        for (size_t i = 0; i < chunks.size(); i++) {
            auto & rows = res[name + (n_per_graph > 1 ? "#" + std::to_string(i % n_per_graph) : "")];
            rows.insert(rows.end(), chunks[i].begin(), chunks[i].end());
        }
    }
    for (const auto & [_, rows] : res) {
        if (rows.size() != n_tokens) {
            return {};
        }
    }
    return res;
}

static router_observer g_router_observer;

// the scores that the top-k selection of 'node', a view of the first k entries of a descending argsort, ranks
static const ggml_tensor * router_scores(const ggml_tensor * node) {
    const char * name = ggml_get_name(node);
    if (strncmp(name, "ffn_moe_topk-", 13) != 0 && strncmp(name, "ffn_moe_group_topk-", 19) != 0) {
        return nullptr;
    }
    if (node->op != GGML_OP_VIEW || node->src[0] == nullptr || node->src[0]->op != GGML_OP_ARGSORT) {
        return nullptr;
    }
    const ggml_tensor * scores = node->src[0]->src[0];
    if (scores == nullptr || scores->type != GGML_TYPE_F32 || !ggml_is_contiguous(scores) || !ggml_is_contiguous(node->src[0])) {
        return nullptr;
    }
    return scores;
}

static bool observe_router(ggml_tensor * t, bool ask, void * user_data) {
    auto * obs = (router_observer *) user_data;
    const ggml_tensor * scores = obs->active ? router_scores(t) : nullptr;
    if (ask || scores == nullptr) {
        // asks for the selections only, and continues the graph after any other tensor
        return ask ? scores != nullptr : true;
    }

    const int64_t n = scores->ne[0];
    const int64_t k = t->ne[0];
    const int64_t n_rows = ggml_nrows(scores);
    std::vector<float>   s(ggml_nelements(scores));
    std::vector<int32_t> order(ggml_nelements(t->src[0]));
    ggml_backend_tensor_get(scores, s.data(), 0, ggml_nbytes(scores));
    ggml_backend_tensor_get(t->src[0], order.data(), 0, ggml_nbytes(t->src[0]));

    auto & rows = obs->routers[ggml_get_name(t)].emplace_back();
    for (int64_t r = 0; r < n_rows; r++) {
        const float * sr = s.data() + r*n;
        router_observer::row row;
        row.ids.assign(order.begin() + r*n, order.begin() + r*n + k);
        std::sort(row.ids.begin(), row.ids.end());

        // masked scores are -inf, every expert used leaves nothing to tie with
        row.margin = INFINITY;
        if (k < n) {
            std::vector<float> sorted(sr, sr + n);
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            float scale = 0.0f;
            for (const float v : sorted) {
                if (std::isfinite(v)) {
                    scale = std::max(scale, std::fabs(v));
                }
            }
            if (std::isfinite(sorted[k])) {
                row.margin = scale > 0.0f ? (sorted[k - 1] - sorted[k])/scale : 0.0f;
            }
        }
        rows.push_back(std::move(row));
    }
    return true;
}

// the selections of a decode may differ from those of a reference decode only where the reference's margin is below
//     this, relative to the largest score of the row: 10x the largest margin at which they differed over 50 seeds
static const float max_router_flip_margin = 3.4e-4f;

// how much of a decode is compared with a reference decode of the same tokens, from the routers of both
struct routed_comparison {
    bool     routed     = false; // a router selects fewer experts than it ranks
    bool     consistent = true;  // both decodes ran the same routers over every token
    uint32_t n_compared = 0;     // the tokens before the first one whose selections differ
    // the first router of that token whose selections differ, in graph order: the later ones of the token take its output
    std::string flip_router;
    float       flip_margin = 0.0f; // the reference's margin there
};

// the position of a router in the graph of a token: by layer, the expert groups before the experts
static std::pair<int, int> router_order(const std::string & name) {
    const size_t dash = name.rfind('-');
    const int il = dash == std::string::npos ? -1 : std::atoi(name.c_str() + dash + 1);
    return { il, name.compare(0, 19, "ffn_moe_group_topk-") == 0 ? 0 : 1 };
}

static routed_comparison compare_routing(const router_observer & ref, const router_observer & cur, uint32_t n_tokens) {
    routed_comparison res;
    res.n_compared = n_tokens;
    const auto rows_ref = router_rows(ref, n_tokens);
    const auto rows_cur = router_rows(cur, n_tokens);
    // an observed selection whose rows do not cover the tokens leaves its decode without rows
    if (rows_ref.size() != rows_cur.size() || rows_ref.empty() != ref.routers.empty() || rows_cur.empty() != cur.routers.empty()) {
        res.consistent = false;
    }
    for (const auto & [name, rows] : rows_ref) {
        const auto it = rows_cur.find(name);
        if (it == rows_cur.end()) {
            res.consistent = false;
            continue;
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            res.routed = res.routed || std::isfinite(rows[t].margin);
            if (rows[t].ids == it->second[t].ids || t > res.n_compared) {
                continue;
            }
            if (t < res.n_compared || router_order(name) < router_order(res.flip_router)) {
                res.flip_router = name;
                res.flip_margin = rows[t].margin;
            }
            res.n_compared = t;
        }
    }
    return res;
}

// a decode whose selections first differ from the reference's at a clear margin, or at the first token, is not
//     explained by a near-tie
static bool routing_ok(const routed_comparison & r, uint32_t n_tokens) {
    if (!r.consistent) {
        return false;
    }
    if (r.n_compared == n_tokens) {
        return true;
    }
    return r.n_compared > 0 && r.flip_margin < max_router_flip_margin;
}

// observes the routers of the decodes made while it lives
struct router_observation {
    router_observation()  { g_router_observer = {}; g_router_observer.active = true; }
    ~router_observation() { g_router_observer = {}; }

    router_observer take() {
        router_observer res = std::move(g_router_observer);
        g_router_observer = {};
        g_router_observer.active = true;
        return res;
    }
};

static llama_model_ptr get_model(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const bool sensitive_weights = false,
        const bool load_mtp = false) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = load_mtp;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    tensor_data_params tensor_params = { seed, stdev, sensitive_weights, LLM_ARCH_UNKNOWN };
    if (gguf_ctx != nullptr) {
        tensor_params.arch = llama_arch_from_string(gguf_get_val_str(gguf_ctx, gguf_find_key(gguf_ctx, "general.architecture")));
    }
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    return model;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0,
        const enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_UNSPECIFIED,
        const enum llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO, const bool sensitive_weights = false,
        const bool load_mtp = false) {
    llama_model_ptr model = get_model(gguf_ctx, file, seed, stdev, devs, split_mode, tensor_buft_overrides, sensitive_weights, load_mtp);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.flash_attn_type = flash_attn_type;
    // inactive outside of the mixed batch check, where it asks for no tensor and the graphs compute as without it
    ctx_params.cb_eval           = observe_router;
    ctx_params.cb_eval_user_data = &g_router_observer;
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (pooling != LLAMA_POOLING_TYPE_UNSPECIFIED) {
        ctx_params.embeddings = true;
        ctx_params.n_seq_max  = 4;
    } else if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }
    if (gguf_ctx != nullptr && pooling != LLAMA_POOLING_TYPE_UNSPECIFIED && pooling != LLAMA_POOLING_TYPE_RANK) {
        // a user-defined model gets every optional tensor, but only a reranker has a classifier head
        model->cls       = nullptr;
        model->cls_b     = nullptr;
        model->cls_out   = nullptr;
        model->cls_out_b = nullptr;
    }
    if (gguf_ctx != nullptr && model->arch == LLM_ARCH_GEMMA4 && model->hparams.n_embd_per_layer == 0) {
        // without per-layer embeddings (Gemma 4 12B), full attention layers have no V projection and use K as V
        for (uint32_t il = 0; il < model->hparams.n_layer(); il++) {
            if (!model->hparams.is_swa(il)) {
                model->layers[il].wv      = nullptr;
                model->layers[il].wv_s    = nullptr;
                model->layers[il].wv_in_s = nullptr;
            }
        }
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

// a corrupted memory cell or a replaced input token must change the output by more than this relative amount
static const double min_sensitivity = 2.0e-3;

static double relative_change(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double diff = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == b[i]) {
            if (std::isfinite(a[i])) {
                norm += (double) a[i] * a[i];
            }
            continue; // also skips logits both masked to -inf
        }
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return INFINITY;
        }
        diff += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        norm += (double) a[i] * a[i];
    }
    return diff == 0.0 ? 0.0 : std::sqrt(diff / norm);
}

// output of the last token: logits, or the hidden state for models without an output head
static std::vector<float> decode_last(llama_context * lctx, const std::vector<llama_token> & tokens, const llama_pos pos0) {
    const llama_model * model = llama_get_model(lctx);
    common_batch batch(lctx);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.add(tokens[i], pos0 + i, 0, i + 1 == tokens.size());
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        throw std::runtime_error("failed to decode batch");
    }
    if (model->output == nullptr) {
        const float * embd = llama_get_embeddings_ith(lctx, -1);
        return std::vector<float>(embd, embd + llama_model_n_embd_out(model));
    }
    const float * logits = llama_get_logits_ith(lctx, -1);
    return std::vector<float>(logits, logits + llama_vocab_n_tokens(llama_model_get_vocab(model)));
}

// the attention caches and recurrent states inside a memory module, false if the module type is unknown
static bool collect_memory(llama_memory_i * mem, std::vector<llama_kv_cache *> & kvs, std::vector<llama_memory_recurrent *> & rss) {
    const auto add_kv = [&](llama_kv_cache * kv) {
        if (kv != nullptr) {
            kvs.push_back(kv);
        }
    };
    if (auto * m = dynamic_cast<llama_kv_cache *>(mem)) {
        add_kv(m);
    } else if (auto * m = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        add_kv(m->get_base());
        add_kv(m->get_swa());
    } else if (auto * m = dynamic_cast<llama_kv_cache_dsa *>(mem)) {
        add_kv(m->get_mla());
        add_kv(m->get_lid());
    } else if (auto * m = dynamic_cast<llama_kv_cache_dsa_iswa *>(mem)) {
        add_kv(m->get_dsa()->get_mla());
        add_kv(m->get_dsa()->get_lid());
        add_kv(m->get_swa());
    } else if (auto * m = dynamic_cast<llama_kv_cache_dsv4 *>(mem)) {
        add_kv(m->get_raw()->get_base());
        add_kv(m->get_raw()->get_swa());
        add_kv(m->get_csa());
        add_kv(m->get_hca());
        add_kv(m->get_lid());
    } else if (auto * m = dynamic_cast<llama_kv_cache_msa *>(mem)) {
        add_kv(m->get_base());
        add_kv(m->get_idx());
    } else if (auto * m = dynamic_cast<llama_memory_hybrid *>(mem)) {
        add_kv(m->get_mem_attn());
        if (auto * idx = dynamic_cast<llama_memory_hybrid_idx *>(mem)) {
            add_kv(idx->get_mem_idx());
        }
        rss.push_back(m->get_mem_recr());
    } else if (auto * m = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        add_kv(m->get_mem_attn()->get_base());
        add_kv(m->get_mem_attn()->get_swa());
        rss.push_back(m->get_mem_recr());
    } else if (auto * m = dynamic_cast<llama_memory_recurrent *>(mem)) {
        rss.push_back(m);
    } else {
        return false;
    }
    return true;
}

// overwrite every K and V tensor of dst with the one of src, as stale pages holding another sequence would; returns the tensors written
static size_t copy_kv_storage(const llama_kv_cache * src, const llama_kv_cache * dst) {
    size_t n_tensors = 0;
    for (const uint32_t il : dst->get_layer_ids()) {
        const ggml_tensor * ts[2] = {src->get_k_storage(il), src->get_v_storage(il)};
        ggml_tensor *       td[2] = {dst->get_k_storage(il), dst->get_v_storage(il)};
        for (int i = 0; i < 2; i++) {
            if (td[i] == nullptr) {
                continue;
            }
            GGML_ASSERT(ts[i] != nullptr && ggml_are_same_shape(ts[i], td[i]) && ts[i]->type == td[i]->type);
            std::vector<uint8_t> data(ggml_nbytes(ts[i]));
            ggml_backend_tensor_get(ts[i], data.data(), 0, data.size());
            ggml_backend_tensor_set(td[i], data.data(), 0, data.size());
            n_tensors++;
        }
    }
    return n_tensors;
}

// generated weights are random, so check that the paths a memory bug would corrupt actually reach the output
static bool check_sensitivity(llama_model * model, const size_t seed, std::string & result) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 0;
    ctx_params.n_batch         = 64;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_seq_max       = 1;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    if (model->output == nullptr) {
        ctx_params.embeddings   = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    const auto new_ctx = [&]() {
        llama_context_ptr lctx(llama_init_from_model(model, ctx_params));
        if (!lctx) {
            throw std::runtime_error("failed to create llama context");
        }
        return lctx;
    };

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> prompt = get_tokens(33, n_vocab, seed);
    const std::vector<llama_token> tokens(prompt.begin(), prompt.end() - 1);
    const llama_token next = prompt.back();

    bool ok = true;
    const auto check = [&](const char * name, const double rel) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s%s %.2e", result.empty() ? "" : ", ", name, rel);
        result += buf;
        if (!(rel > min_sensitivity)) {
            ok = false;
        }
    };

    std::vector<llama_kv_cache *> kvs;
    std::vector<llama_memory_recurrent *> rss;
    {
        const llama_context_ptr lctx = new_ctx();
        if (llama_get_memory(lctx.get()) == nullptr) {
            // without memory only the input can carry a difference
            const std::vector<float> ref = decode_last(lctx.get(), tokens, 0);
            std::vector<llama_token> replaced = tokens;
            replaced[0] = (replaced[0] + 1) % n_vocab;
            check("token", relative_change(ref, decode_last(new_ctx().get(), replaced, 0)));
            return ok;
        }
        if (!collect_memory(llama_get_memory(lctx.get()), kvs, rss)) {
            result = "unknown memory module";
            return false;
        }
    }

    const auto corrupted_output = [&](const std::function<void(llama_memory_i *)> & corrupt) {
        const llama_context_ptr lctx = new_ctx();
        decode_last(lctx.get(), tokens, 0);
        corrupt(llama_get_memory(lctx.get()));
        return decode_last(lctx.get(), {next}, tokens.size());
    };
    const std::vector<float> ref = corrupted_output([](llama_memory_i *) {});

    if (!kvs.empty()) {
        // every cell takes the rows of the same position in another sequence: content-addressed attention reads a
        // permutation of its cells unchanged, and sparse attention reads only the cells it selects
        std::vector<llama_token> other = tokens;
        for (llama_token & t : other) {
            t = (t + 1) % n_vocab;
        }
        const llama_context_ptr lctx_other = new_ctx();
        decode_last(lctx_other.get(), other, 0);
        std::vector<llama_kv_cache *> kvs_other;
        std::vector<llama_memory_recurrent *> rss_other;
        collect_memory(llama_get_memory(lctx_other.get()), kvs_other, rss_other);

        size_t n_tensors = 0;
        const std::vector<float> out = corrupted_output([&](llama_memory_i * mem) {
            std::vector<llama_kv_cache *> kvs_ctx;
            std::vector<llama_memory_recurrent *> rss_ctx;
            collect_memory(mem, kvs_ctx, rss_ctx);
            GGML_ASSERT(kvs_ctx.size() == kvs_other.size());
            for (size_t i = 0; i < kvs_ctx.size(); i++) {
                n_tensors += copy_kv_storage(kvs_other[i], kvs_ctx[i]);
            }
        });
        check("stale kv", n_tensors > 0 ? relative_change(ref, out) : 0.0);
    }
    if (!rss.empty()) {
        const std::vector<float> out = corrupted_output([&](llama_memory_i * mem) {
            std::vector<llama_kv_cache *> kvs_ctx;
            std::vector<llama_memory_recurrent *> rss_ctx;
            collect_memory(mem, kvs_ctx, rss_ctx);
            for (const llama_memory_recurrent * rs : rss_ctx) {
                for (const std::vector<ggml_tensor *> * states : {&rs->r_l, &rs->s_l}) {
                    for (ggml_tensor * t : *states) {
                        if (t != nullptr) {
                            const std::vector<uint8_t> zeros(ggml_nbytes(t), 0);
                            ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
                        }
                    }
                }
            }
        });
        check("zeroed recurrent state", relative_change(ref, out));
    }
    return ok;
}

// pooled output per sequence: n_embd values, or n_cls_out scores with rank pooling
static std::vector<float> get_embeddings(llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_seq     = llama_n_seq_max(lctx);
    const uint32_t n_seq_tok = tokens.size() / n_seq;
    const uint32_t n_out     = llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK ? llama_model_n_cls_out(model) : llama_model_n_embd_out(model);
    common_batch batch(lctx);
    for (uint32_t s = 0; s < n_seq; s++) {
        for (uint32_t pos = 0; pos < n_seq_tok; pos++) {
            batch.add(tokens[s*n_seq_tok + pos], pos, llama_seq_id(s), true);
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
        throw std::runtime_error("failed to encode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_seq*n_out);
    for (uint32_t s = 0; s < n_seq; s++) {
        const float * embd = llama_get_embeddings_seq(lctx, s);
        if (embd == nullptr) {
            throw std::runtime_error("no pooled embeddings for sequence " + std::to_string(s));
        }
        ret.insert(ret.end(), embd, embd + n_out);
    }
    return ret;
}

// pooled outputs are dominated by bias terms that do not depend on the input, this keeps only what differs between sequences
static std::vector<float> seq_deviation(const std::vector<float> & out, const size_t n_seq) {
    const size_t n_out = out.size() / n_seq;
    std::vector<float> ret(out.size());
    for (size_t i = 0; i < n_out; i++) {
        double mean = 0.0;
        for (size_t s = 0; s < n_seq; s++) {
            mean += out[s*n_out + i];
        }
        mean /= n_seq;
        for (size_t s = 0; s < n_seq; s++) {
            ret[s*n_out + i] = out[s*n_out + i] - mean;
        }
    }
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_FLINT:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_CLOVER_MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_XING4_0:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_KIMI_K31:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MISTRAL_LARGE4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_KOLIBRI1:
        case LLM_ARCH_METIS16:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

struct model_config {
    std::string             name;
    std::string             file_suffix;
    bool                    moe;
    enum llama_pooling_type pooling;           // unspecified for models that return logits
    bool                    iswa_only = false; // see get_gguf_ctx
    uint32_t                n_layer_nextn = 0; // see get_gguf_ctx
};

// encoder-only archs are generated as an embedding model and as a reranker
static std::vector<model_config> get_model_configs(const llm_arch arch) {
    if (arch == LLM_ARCH_BERT) {
        return {{"Dense", "dense", false, LLAMA_POOLING_TYPE_CLS}, {"Rank", "rank", false, LLAMA_POOLING_TYPE_RANK}};
    }
    std::vector<model_config> ret;
    if (!moe_mandatory(arch)) {
        ret.push_back({"Dense", "dense", false, LLAMA_POOLING_TYPE_UNSPECIFIED});
    }
    if (arch == LLM_ARCH_GEMMA4) {
        ret.push_back({"ISWA", "dense-iswa-only", false, LLAMA_POOLING_TYPE_UNSPECIFIED, true});
    }
    if (moe_implemented(arch)) {
        ret.push_back({"MoE", "moe", true, LLAMA_POOLING_TYPE_UNSPECIFIED});
    }
    return ret;
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

// archs whose generated model also gets a NextN layer, to check how its trunk and MTP graphs name their layer outputs
static bool nextn_test(const llm_arch arch) {
    return arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP;
}

// archs whose layers read their compress ratio from the file, so a NextN layer may be QSA or dense
static bool nextn_dense_test(const llm_arch arch) {
    return arch == LLM_ARCH_QWEN4EXP;
}

// archs whose saved test models include one with a NextN layer, <arch>-moe-nextn.gguf, beside those without, and
//     <arch>-moe-nextn-only.gguf, the NextN layer of that model alone
static bool nextn_save(const llm_arch arch) {
    return arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP;
}

// the layer of a node named exactly <prefix><il>, or -1; a view of the node extends the name and is not one
static int named_layer(const ggml_tensor * t, const char * prefix) {
    const char * name = ggml_get_name(t);
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || !isdigit((unsigned char) name[n])) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + n, &end, 10);
    return *end == '\0' ? (int) il : -1;
}

// each layer in [il_begin, il_end) has exactly one output named <prefix><il> for one of the prefixes, a node that computes it and not
//     a view, and no other layer has one; returns why not, empty if so
static std::string check_layer_outputs(ggml_cgraph * gf, const std::vector<const char *> & prefixes, const int il_begin, const int il_end) {
    if (gf == nullptr) {
        return "the graph does not build";
    }
    std::map<int, std::vector<const ggml_tensor *>> outputs;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        for (const char * prefix : prefixes) {
            const int il = named_layer(t, prefix);
            if (il >= 0) {
                outputs[il].push_back(t);
            }
        }
    }
    for (int il = il_begin; il < il_end; ++il) {
        const auto it = outputs.find(il);
        if (it == outputs.end() || it->second.size() != 1) {
            return format("layer %d has %zu outputs, expected 1", il, it == outputs.end() ? (size_t) 0 : it->second.size());
        }
        if (it->second[0]->view_src != nullptr) {
            return format("the output of layer %d (%s) is a view", il, ggml_get_name(it->second[0]));
        }
    }
    for (const auto & [il, nodes] : outputs) {
        if (il < il_begin || il >= il_end) {
            return format("%s is the output of a layer outside [%d, %d)", ggml_get_name(nodes[0]), il_begin, il_end);
        }
    }
    return "";
}

// a model with NextN layers splits into a trunk graph whose layers end at l_out-<il> or l_last-<il> and an MTP graph whose layers
//     end at mtp_post_ffn-<il>, il from n_layer on: each graph names each of its own layer outputs once and none of the other's
static bool check_nextn_outputs(const llm_arch arch, const size_t seed, const float stdev, std::string & result) {
    const uint32_t n_layer_nextn = 1;
    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe_mandatory(arch), LLAMA_POOLING_TYPE_UNSPECIFIED, /*iswa_only =*/ false,
        /*n_layer_min =*/ 0, n_layer_nextn);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t cpu_only[] = { nullptr };
    model_params.devices  = cpu_only;
    model_params.load_mtp = true;
    tensor_data_params tensor_params = { seed, stdev, /*sensitive =*/ false, arch };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
    if (!model) {
        result = "failed to create the model";
        return false;
    }
    const int n_layer = llama_model_n_layer(model.get());
    if (llama_model_n_layer_nextn(model.get()) != (int32_t) n_layer_nextn) {
        result = "the model has no NextN layer";
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 0;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    llama_context_ptr ctx_trunk(llama_init_from_model(model.get(), ctx_params));
    if (!ctx_trunk) {
        result = "failed to create the trunk context";
        return false;
    }
    ctx_params.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.ctx_other = ctx_trunk.get();
    llama_context_ptr ctx_mtp(llama_init_from_model(model.get(), ctx_params));
    if (!ctx_mtp) {
        result = "failed to create the MTP context";
        return false;
    }

    const std::vector<const char *> trunk_prefixes = { "l_out-", "l_last-" };
    const std::vector<const char *> mtp_prefixes   = { "mtp_post_ffn-" };
    const struct { uint32_t n_tokens; uint32_t n_outputs; } shapes[] = { { 1, 1 }, { 8, 1 }, { 8, 8 } };
    for (const auto & shape : shapes) {
        const std::string what = format("%u tokens, %u outputs", shape.n_tokens, shape.n_outputs);

        ggml_cgraph * gf_trunk = llama_graph_reserve(ctx_trunk.get(), shape.n_tokens, 1, shape.n_outputs);
        std::string err = check_layer_outputs(gf_trunk, trunk_prefixes, 0, n_layer);
        if (err.empty()) {
            err = check_layer_outputs(gf_trunk, mtp_prefixes, 0, 0);
        }
        if (!err.empty()) {
            result = "trunk graph, " + what + ": " + err;
            return false;
        }

        ggml_cgraph * gf_mtp = llama_graph_reserve(ctx_mtp.get(), shape.n_tokens, 1, shape.n_outputs);
        err = check_layer_outputs(gf_mtp, mtp_prefixes, n_layer, n_layer + n_layer_nextn);
        if (err.empty()) {
            err = check_layer_outputs(gf_mtp, trunk_prefixes, 0, 0);
        }
        if (!err.empty()) {
            result = "MTP graph, " + what + ": " + err;
            return false;
        }
    }
    result = format("%d trunk layers, %u NextN layer", n_layer, n_layer_nextn);
    return true;
}

// writes the tensors of the NextN layers of the model in src_path, and none of its other tensors, to dst_path: a file of
//     NextN layers alone, which has no token embedding, output norm or LM head of its own, nor the optional ones of a NextN layer
static void save_nextn_only(const std::string & src_path, const std::string & dst_path, const int n_layer) {
    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { /*no_alloc =*/ false, &ctx_data };
    gguf_context_ptr src(gguf_init_from_file(src_path.c_str(), params));
    ggml_context_ptr ctx_data_ptr(ctx_data);
    if (!src) {
        throw std::runtime_error("failed to read " + src_path);
    }
    gguf_context_ptr dst(gguf_init_empty());
    gguf_set_kv(dst.get(), src.get());
    const std::string prefix = format("blk.%d.", n_layer);
    const std::string own_head[] = { prefix + "nextn.embed_tokens.", prefix + "nextn.shared_head_head.", prefix + "nextn.shared_head_norm." };
    for (int64_t i = 0; i < gguf_get_n_tensors(src.get()); ++i) {
        const char * name = gguf_get_tensor_name(src.get(), i);
        const bool head = std::any_of(std::begin(own_head), std::end(own_head),
            [name](const std::string & p) { return strncmp(name, p.c_str(), p.size()) == 0; });
        if (strncmp(name, prefix.c_str(), prefix.size()) == 0 && !head) {
            gguf_add_tensor(dst.get(), ggml_get_tensor(ctx_data, name));
        }
    }
    if (!gguf_write_to_file(dst.get(), dst_path.c_str(), /*only_meta =*/ false)) {
        throw std::runtime_error("failed to write " + dst_path);
    }
}

// whether a node of the graph reads tensor t
static bool graph_reads(ggml_cgraph * gf, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (node->src[j] == t) {
                return true;
            }
        }
    }
    return false;
}

// a file of NextN layers alone has no token embedding or LM head and reads those of the model of the context it drafts for,
//     ctx_other, without sharing that context's memory: an MTP context over it needs ctx_other, its graph reads the very
//     tensors of the target model, and its memory is its own
static bool check_nextn_borrowed_head(const llm_arch arch, const size_t seed, const float stdev, std::string & result) {
    const uint32_t n_layer_nextn = 1;
    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe_mandatory(arch), LLAMA_POOLING_TYPE_UNSPECIFIED, /*iswa_only =*/ false,
        /*n_layer_min =*/ 0, n_layer_nextn);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t cpu_only[] = { nullptr };
    model_params.devices  = cpu_only;
    model_params.load_mtp = true;
    tensor_data_params tensor_params = { seed, stdev, /*sensitive =*/ false, arch };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
    if (!model) {
        result = "failed to create the model";
        return false;
    }
    const int n_layer = llama_model_n_layer(model.get());

    const std::string path_full  = get_tmp_gguf_path();
    const std::string path_nextn = get_tmp_gguf_path();
    llama_model_save_to_file(model.get(), path_full.c_str());
    save_nextn_only(path_full, path_nextn, n_layer);
    std::filesystem::remove(path_full);
    llama_model_ptr model_nextn(llama_model_load_from_file(path_nextn.c_str(), model_params));
    std::filesystem::remove(path_nextn);
    if (!model_nextn) {
        result = "failed to load the file of NextN layers";
        return false;
    }
    if (model_nextn->tok_embd != nullptr || model_nextn->output != nullptr || model_nextn->output_norm != nullptr) {
        result = "the file of NextN layers has a token embedding, an output norm or an LM head";
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 0;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    llama_context_ptr ctx_trunk(llama_init_from_model(model.get(), ctx_params));
    if (!ctx_trunk) {
        result = "failed to create the trunk context";
        return false;
    }

    ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    if (llama_context_ptr ctx_alone(llama_init_from_model(model_nextn.get(), ctx_params)); ctx_alone) {
        result = "an MTP context without ctx_other was created";
        return false;
    }
    ctx_params.ctx_other = ctx_trunk.get();
    llama_context_ptr ctx_mtp(llama_init_from_model(model_nextn.get(), ctx_params));
    if (!ctx_mtp) {
        result = "failed to create the MTP context";
        return false;
    }
    if (llama_get_ctx_other(ctx_mtp.get()) != ctx_trunk.get()) {
        result = "the MTP context does not read ctx_other";
        return false;
    }

    const struct { uint32_t n_tokens; uint32_t n_outputs; } shapes[] = { { 1, 1 }, { 8, 1 }, { 8, 8 } };
    for (const auto & shape : shapes) {
        ggml_cgraph * gf = llama_graph_reserve(ctx_mtp.get(), shape.n_tokens, 1, shape.n_outputs);
        if (gf == nullptr) {
            result = format("the MTP graph does not build for %u tokens, %u outputs", shape.n_tokens, shape.n_outputs);
            return false;
        }
        if (!graph_reads(gf, model->tok_embd) || !graph_reads(gf, model->output) ||
                (model->output_norm != nullptr && !graph_reads(gf, model->output_norm))) {
            result = format("the MTP graph for %u tokens, %u outputs does not read the token embedding, output norm and LM head of the target",
                shape.n_tokens, shape.n_outputs);
            return false;
        }
    }

    // memory that mirrored the target's would hold the cells the target decodes
    if (llama_get_memory_other(ctx_mtp.get()) != nullptr || llama_get_memory(ctx_mtp.get()) == llama_get_memory(ctx_trunk.get())) {
        result = "the MTP context shares the memory of ctx_other";
        return false;
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    decode_last(ctx_trunk.get(), get_tokens(8, n_vocab, seed), 0);
    if (llama_memory_seq_pos_max(llama_get_memory(ctx_mtp.get()), 0) != -1) {
        result = "the memory of the MTP context holds the cells the target decoded";
        return false;
    }

    result = format("%d trunk layers, %u NextN layer", n_layer, n_layer_nextn);
    return true;
}

// a file may give the NextN layer a compress ratio of 0 while the trunk layers keep theirs: the NextN layer then attends
//     densely, yet the trunk still sets the k-pool size, so the MTP graph must not leave inputs it fills unallocated
static bool check_nextn_dense_decode(const llm_arch arch, const size_t seed, const float stdev, std::string & result) {
    const uint32_t n_layer_nextn = 1;
    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe_mandatory(arch), LLAMA_POOLING_TYPE_UNSPECIFIED, /*iswa_only =*/ false,
        /*n_layer_min =*/ 0, n_layer_nextn);

    const std::string key_ratios = llama_arch_kv_name(arch, LLM_KV_ATTENTION_COMPRESS_RATIOS);
    const int64_t     id_ratios  = gguf_find_key(gguf_ctx.get(), key_ratios.c_str());
    if (id_ratios < 0) {
        result = "the model has no compress ratios";
        return false;
    }
    const size_t n_ratios = gguf_get_arr_n(gguf_ctx.get(), id_ratios);
    std::vector<uint32_t> ratios((const uint32_t *) gguf_get_arr_data(gguf_ctx.get(), id_ratios),
                                 (const uint32_t *) gguf_get_arr_data(gguf_ctx.get(), id_ratios) + n_ratios);
    ratios.back() = 0;
    gguf_set_arr_data(gguf_ctx.get(), key_ratios.c_str(), GGUF_TYPE_UINT32, ratios.data(), ratios.size());

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t cpu_only[] = { nullptr };
    model_params.devices  = cpu_only;
    model_params.load_mtp = true;
    tensor_data_params tensor_params = { seed, stdev, /*sensitive =*/ false, arch };
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_params, model_params));
    if (!model) {
        result = "failed to create the model";
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 0;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    llama_context_ptr ctx_trunk(llama_init_from_model(model.get(), ctx_params));
    if (!ctx_trunk) {
        result = "failed to create the trunk context";
        return false;
    }
    ctx_params.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.ctx_other = ctx_trunk.get();
    llama_context_ptr ctx_mtp(llama_init_from_model(model.get(), ctx_params));
    if (!ctx_mtp) {
        result = "failed to create the MTP context";
        return false;
    }

    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const uint32_t n_tokens = 8;
    const std::vector<llama_token> tokens = get_tokens(n_tokens, n_vocab, seed);
    const std::vector<float> h(llama_model_n_embd_out(model.get()), 0.0f);

    // a catch-up decode, then one draft step after it
    llama_pos pos = 0;
    for (const uint32_t n : { n_tokens - 1, 1u }) {
        common_batch batch(ctx_mtp.get());
        for (uint32_t i = 0; i < n; ++i, ++pos) {
            const int32_t idx = batch.add(tokens[pos], pos, 0, i + 1 == n);
            batch.set_embd_state(idx, { h.data(), 1, h.size() });
        }
        if (llama_process(ctx_mtp.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            result = format("the MTP context fails to decode %u tokens at position %d", n, pos - (llama_pos) n);
            return false;
        }
        const float * logits = llama_get_logits_ith(ctx_mtp.get(), -1);
        if (!std::all_of(logits, logits + n_vocab, [](float x) { return std::isfinite(x); })) {
            result = format("the MTP context gives non-finite logits for %u tokens", n);
            return false;
        }
    }

    result = format("NextN layer %d attends densely", llama_model_n_layer(model.get()));
    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    bool all_ok = true;
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        std::vector<model_config> configs = get_model_configs(arch);
        if (nextn_save(arch)) {
            configs.push_back({"MoE NextN", "moe-nextn", true, LLAMA_POOLING_TYPE_UNSPECIFIED, false, /*n_layer_nextn =*/ 1});
        }
        for (const model_config & cfg : configs) {
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), cfg.name.c_str());
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, cfg.moe, cfg.pooling, cfg.iswa_only, /*n_layer_min =*/ 0, cfg.n_layer_nextn);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, false, nullptr, 0, cfg.pooling,
                LLAMA_FLASH_ATTN_TYPE_AUTO, /*sensitive_weights =*/ true, /*load_mtp =*/ cfg.n_layer_nextn > 0);
            const std::string path = dir + "/" + llama_arch_name(arch) + "-" + cfg.file_suffix + ".gguf";
            LOG_INF("%s: '%s %s' -> %s ...", __func__, llama_arch_name(arch), cfg.name.c_str(), path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
            const std::uintmax_t size = std::filesystem::file_size(path);
            LOG_CNT(" (%.2f MB)\n", (double) size / (1024.0 * 1024.0));
            if (cfg.n_layer_nextn > 0) {
                const std::string path_only = dir + "/" + llama_arch_name(arch) + "-" + cfg.file_suffix + "-only.gguf";
                LOG_INF("%s: '%s %s' NextN layer alone -> %s\n", __func__, llama_arch_name(arch), cfg.name.c_str(), path_only.c_str());
                save_nextn_only(path, path_only, llama_model_n_layer(model_and_ctx.first.get()));
            }
            std::string sensitivity;
            if (check_sensitivity(model_and_ctx.first.get(), seed, sensitivity)) {
                LOG_INF("%s: %s model (%s) sensitivity: %s\n", __func__, llama_arch_name(arch), cfg.name.c_str(), sensitivity.c_str());
            } else {
                LOG_ERR("%s: %s model (%s) is not sensitive enough (minimum %.0e): %s\n",
                    __func__, llama_arch_name(arch), cfg.name.c_str(), min_sensitivity, sensitivity.c_str());
                all_ok = false;
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|%12s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|%12s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch", "Mixed tokens");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|------------|\n");
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (const model_config & cfg : get_model_configs(arch)) {
            const std::string & config_name = cfg.name;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, cfg.moe, cfg.pooling, cfg.iswa_only);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            // models without logits are compared on their pooled embeddings
            const auto get_output = [&](const std::pair<llama_model_ptr, llama_context_ptr> & model_and_ctx) {
                if (cfg.pooling != LLAMA_POOLING_TYPE_UNSPECIFIED) {
                    return get_embeddings(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);
                }
                return get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), tokens, encode);
            };
            // CPU references by flash attention
            std::map<bool, std::pair<llama_model_ptr, llama_context_ptr>> model_and_ctx_cpu;
            std::map<bool, std::vector<float>> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!cfg.moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llama_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};
                std::string mixed_tokens = "all"; // the tokens the mixed batch check compares
                std::string mixed_note;           // where the selections of a routed model first differ

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llama_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size, cfg.pooling);
                        logits_dev = get_output(model_and_ctx_dev);
                        // a device may turn flash attention off, compare it with the same attention implementation on the CPU
                        const bool flash_attn = model_and_ctx_dev.second->get_cparams().flash_attn;
                        if (logits_cpu[flash_attn].empty()) {
                            model_and_ctx_cpu[flash_attn] = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode, nullptr, 0, cfg.pooling,
                                flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED);
                            logits_cpu[flash_attn] = get_output(model_and_ctx_cpu[flash_attn]);
                        }
                        double nmse_val = nmse(logits_cpu[flash_attn], logits_dev);
                        if (cfg.pooling != LLAMA_POOLING_TYPE_UNSPECIFIED) {
                            const size_t n_seq = llama_n_seq_max(model_and_ctx_dev.second.get());
                            nmse_val = std::max(nmse_val, nmse(seq_deviation(logits_cpu[flash_attn], n_seq), seq_deviation(logits_dev, n_seq)));
                        }
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (!(nmse_val <= 1e-4)) { // also fails a NaN
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llama_arch_supports_mixed_batch(arch)) {
                                router_observer routers_chunks;
                                router_observer routers_mixed;
                                {
                                    router_observation observation;
                                    if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0) {
                                        throw std::runtime_error("failed to decode mixed batch");
                                    }
                                    routers_chunks = observation.take();
                                    if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                        throw std::runtime_error("failed to decode mixed batch");
                                    }
                                    routers_mixed = observation.take();
                                }

                                // a near-tie of a router can select other experts in the other decode, and the memory carries
                                //     that difference to every later token: compare the tokens before the first one
                                const routed_comparison routing = compare_routing(routers_chunks, routers_mixed, tokens.size());
                                const size_t n_vocab_out = logits_chunks.size()/tokens.size();
                                logits_chunks.resize(routing.n_compared*n_vocab_out);
                                logits_mixed.resize(routing.n_compared*n_vocab_out);
                                const double nmse_mixed = logits_chunks.empty() ? 0.0 : nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4 || !routing_ok(routing, tokens.size())) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                                if (routing.routed) {
                                    mixed_tokens = std::to_string(routing.n_compared) + "/" + std::to_string(tokens.size());
                                }
                                if (routing.n_compared < tokens.size()) {
                                    mixed_note = format("experts first differ at token %u in %s, relative margin %.3e",
                                        routing.n_compared, routing.flip_router.c_str(), routing.flip_margin);
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu[flash_attn].first.get(), model_and_ctx_cpu[flash_attn].second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && cfg.pooling == LLAMA_POOLING_TYPE_UNSPECIFIED && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    if (llama_arch_supports_save(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                        //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                        // the roundtrip compares on one device, so it uses the sensitive weights, under which a tensor the saver drops shows
                        const auto model_and_ctx_saved = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size, cfg.pooling,
                            LLAMA_FLASH_ATTN_TYPE_AUTO, /*sensitive_weights =*/ true);
                        const std::vector<float> logits_saved = get_output(model_and_ctx_saved);
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_saved.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size, cfg.pooling);
                            const std::vector<float> logits_roundtrip = get_output(model_and_ctx_roundtrip);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_saved.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_saved[i]) {
                                    test_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                            // the reloaded model must also keep the pooling type and the classifier head
                            const llama_model * model_dev = model_and_ctx_saved.first.get();
                            const llama_model * model_rt  = model_and_ctx_roundtrip.first.get();
                            bool head_ok = llama_pooling_type(model_and_ctx_roundtrip.second.get()) == llama_pooling_type(model_and_ctx_saved.second.get()) &&
                                (model_rt->cls_out != nullptr) == (model_dev->cls_out != nullptr) &&
                                llama_model_n_cls_out(model_rt) == llama_model_n_cls_out(model_dev);
                            for (uint32_t i = 0; head_ok && i < llama_model_n_cls_out(model_dev); i++) {
                                const char * label_dev = llama_model_cls_label(model_dev, i);
                                const char * label_rt  = llama_model_cls_label(model_rt, i);
                                head_ok = (label_dev == nullptr) == (label_rt == nullptr) && (label_dev == nullptr || strcmp(label_dev, label_rt) == 0);
                            }
                            if (!head_ok) {
                                test_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str, mixed_tokens.c_str());
                if (!mixed_note.empty()) {
                    LOG_INF("%s: %s %s %s: %s\n", __func__, llama_arch_name(arch), dc.label.c_str(), config_name.c_str(), mixed_note.c_str());
                }
            }
        }

        if (nextn_test(arch) && arch_supported(arch)) {
            std::string result;
            const bool ok = check_nextn_outputs(arch, seed, stdev, result);
            n_tests++;
            if (!ok) {
                n_failed++;
                all_ok = false;
            }
            LOG("%s: %s NextN layer outputs: %s (%s)\n", __func__, llama_arch_name(arch),
                ok ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m", result.c_str());
        }

        if (nextn_test(arch) && arch_supported(arch)) {
            std::string result;
            bool ok = false;
            try {
                ok = check_nextn_borrowed_head(arch, seed, stdev, result);
            } catch (const std::exception & e) {
                result = e.what();
            }
            n_tests++;
            if (!ok) {
                n_failed++;
                all_ok = false;
            }
            LOG("%s: %s NextN layers borrowing the target's head: %s (%s)\n", __func__, llama_arch_name(arch),
                ok ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m", result.c_str());
        }

        if (nextn_dense_test(arch) && arch_supported(arch)) {
            std::string result;
            bool ok = false;
            try {
                ok = check_nextn_dense_decode(arch, seed, stdev, result);
            } catch (const std::exception & e) {
                result = e.what();
            }
            n_tests++;
            if (!ok) {
                n_failed++;
                all_ok = false;
            }
            LOG("%s: %s NextN layer without QSA decodes: %s (%s)\n", __func__, llama_arch_name(arch),
                ok ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m", result.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

struct kv_shift_move {
    llama_pos p0;
    llama_pos p1;
    llama_pos delta;
};

static void decode_range(llama_context * lctx, const llama_token * tokens, uint32_t n_tokens, llama_pos pos0, llama_seq_id seq_id, bool output) {
    common_batch batch(lctx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], pos0 + i, seq_id, output || i == n_tokens - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }
}

// the raw attention scores of each layer are linear in K, unlike the logits they do not depend on how sharp the attention is
struct kq_capture {
    bool active = false;
    std::vector<std::pair<std::string, std::vector<float>>> data;
};

// the scores are compared cell by cell, so the cells of the probe must match: prefix and probe fit in the SWA window,
// otherwise the probe can reuse a masked SWA cell in one run and not in the other
static bool kq_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    kq_capture * cap = (kq_capture *) user_data;
    const char * name = ggml_get_name(t);
    const bool want = cap->active && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        (strncmp(name, "kq-", 3) == 0 || strncmp(name, "indexer_kq-", 11) == 0 || strncmp(name, "indexer_score-", 14) == 0);
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto it = std::find_if(cap->data.begin(), cap->data.end(), [&](const auto & e) { return e.first == name; });
        if (it == cap->data.end()) {
            cap->data.emplace_back(name, std::move(buf));
        } else {
            it->second = std::move(buf);
        }
    }
    return true;
}

struct kq_diff {
    double      worst = 0.0;
    std::string first_bad; // first tensor above the threshold, later layers can inherit its error through Q
    std::string bad;       // all tensors above the threshold, with their nmse
};

static kq_diff compare_kq(const std::vector<std::pair<std::string, std::vector<float>>> & ref,
                          const std::vector<std::pair<std::string, std::vector<float>>> & cur, double thold) {
    if (ref.size() != cur.size()) {
        throw std::runtime_error("kq tensor count mismatch");
    }
    kq_diff res;
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i].first != cur[i].first || ref[i].second.size() != cur[i].second.size()) {
            throw std::runtime_error("kq tensor mismatch: " + ref[i].first);
        }
        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref[i].second.size(); j++) {
            const double a = ref[i].second[j];
            const double b = cur[i].second[j];
            // masked rows may hold -inf in both
            if (!std::isfinite(a) && a == b) {
                continue;
            }
            mse_a_b += (a - b)*(a - b);
            mse_a_0 += a*a;
        }
        const double val = mse_a_b == 0.0 ? 0.0 : mse_a_0 == 0.0 ? INFINITY : mse_a_b/mse_a_0;
        res.worst = std::max(res.worst, std::isnan(val) ? INFINITY : val);
        if (!(val <= thold)) {
            if (res.first_bad.empty()) {
                res.first_bad = ref[i].first;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), " %s=%.1e", ref[i].first.c_str(), val);
            res.bad += buf;
        }
    }
    return res;
}

// decodes the prefix at pos_prefix, applies each step of moves with its own K-shift and returns the logits of the probe at pos_probe
static std::vector<float> get_logits_kv_shift(
        llama_model * model, llama_context * lctx, kq_capture & cap, const std::vector<llama_token> & prefix, const std::vector<llama_token> & probe,
        llama_pos pos_prefix, const std::vector<std::vector<kv_shift_move>> & steps, llama_pos pos_probe) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    cap.data.clear();
    llama_memory_clear(mem, true);
    decode_range(lctx, prefix.data(), prefix.size(), pos_prefix, 0, false);

    for (size_t i = 0; i < steps.size(); i++) {
        for (const kv_shift_move & mv : steps[i]) {
            llama_memory_seq_add(mem, 0, mv.p0, mv.p1, mv.delta);
        }
        // there is no public API to apply a pending shift, a decode of a throwaway token in seq 1 does it
        if (i + 1 < steps.size()) {
            decode_range(lctx, probe.data(), 1, 0, 1, false);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }

    cap.active = true;
    decode_range(lctx, probe.data(), probe.size(), pos_probe, 0, true);
    cap.active = false;

    std::vector<float> ret;
    ret.reserve(probe.size()*n_vocab);
    for (uint32_t i = 0; i < probe.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
    }
    return ret;
}

// a K-shift is correct if decoding at [d, d+n) and shifting by -d gives the same probe logits as decoding at [0, n)
static int test_kv_shift(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // n_prefix + n_probe stays below the fixture SWA window (n_ctx/8 = 32)
    const uint32_t  n_prefix = 20;
    const uint32_t  n_probe  = 8;
    const llama_pos d        = 64;
    const llama_pos gap      = 4;

    const std::vector<llama_token> tokens = get_tokens(n_prefix + n_probe, 128, seed);
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);
    const std::vector<llama_token> probe (tokens.begin() + n_prefix, tokens.end());

    const std::vector<std::vector<kv_shift_move>> steps_shift  = { { { d, d + (llama_pos) n_prefix, -d } } };
    const std::vector<std::vector<kv_shift_move>> steps_shift2 = { { { d, d + (llama_pos) n_prefix, -d/2 } }, { { d/2, d/2 + (llama_pos) n_prefix, -d/2 } } };
    // control: the second half of the prefix ends up gap positions too far, which tells if the probe is sensitive to K positions at all
    const std::vector<std::vector<kv_shift_move>> steps_ctrl   = { {
        { d,                           d + (llama_pos) n_prefix/2, -d       },
        { d + (llama_pos) n_prefix/2,  d + (llama_pos) n_prefix,   -d + gap },
    } };

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }
    const std::string template_row = "|%" + std::to_string(max_arch_name_length) + "s|%6s|%10s|%10s|%10s|%10s|%s\n";

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_row.c_str(), "Model arch.", "Config", "Shift kq", "Shift2 kq", "Logits", "Control kq", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !arch_supported(arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        if (encode) {
            continue;
        }
        // generated only as pooled embedding models, which keep no KV cache to shift
        if (arch == LLM_ARCH_BERT) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * config_name = moe ? "MoE" : "Dense";

            // the attention scale depends on the absolute position, so the hidden states of the prefix do too
            const double thold = arch == LLM_ARCH_MISTRAL_LARGE4 ? 1e-4 : 1e-6;

            // at least 4 layers, so that "every 4th layer" patterns (NoPE, full attention) are covered
            // exaone4 is SWA only with 64 layers, smallthinker always ropes with 4
            const uint32_t n_layer_min = arch == LLM_ARCH_EXAONE4 ? 64 : arch == LLM_ARCH_SMALLTHINKER ? 8 : 4;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, LLAMA_POOLING_TYPE_UNSPECIFIED, /*iswa_only =*/ false, n_layer_min);
            // different bases, so that a shift using the wrong one is caught
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE).c_str(),     20000.0f);
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE_SWA).c_str(), 10000.0f);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }

            llama_model_ptr model;
            try {
                model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            } catch (const std::exception & err) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;31mERROR\033[0m (model load)");
                n_tests++;
                n_failed++;
                continue;
            }
            // absolute position embeddings are not translation invariant, the comparison does not hold
            if (llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NONE) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;33mSKIP\033[0m (no rope)");
                continue;
            }

            LOG("|%*s|%6s|", (int) max_arch_name_length, llama_arch_name(arch), config_name);
            fflush(stdout);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = 0;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.n_seq_max       = 2;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = GGML_TYPE_F32;
            ctx_params.type_v          = GGML_TYPE_F32;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // differs from the SWA default, so that a shift using the wrong scale is caught
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            ctx_params.rope_freq_scale   = 0.5f;
            kq_capture cap;
            ctx_params.cb_eval           = kq_capture_cb;
            ctx_params.cb_eval_user_data = &cap;

            ud.verbosity = LOG_LEVEL_OUTPUT;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            ud.verbosity = verbosity;
            if (!lctx) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (context)\n", "", "", "", "");
                continue;
            }
            llama_memory_t mem = llama_get_memory(lctx.get());
            if (mem == nullptr || !llama_memory_can_shift(mem)) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (no shift)\n", "", "", "", "");
                continue;
            }

            std::string status;
            char shift_str[16]  = "";
            char shift2_str[16] = "";
            char logits_str[16] = "";
            char ctrl_str[16]   = "";
            n_tests++;
            try {
                const std::vector<float> logits_ref = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, 0, {}, n_prefix);
                const auto kq_ref = cap.data;
                if (verbosity >= LOG_LEVEL_INFO) {
                    for (const auto & e : kq_ref) {
                        LOG("[%s: %zu] ", e.first.c_str(), e.second.size());
                    }
                }
                const std::vector<float> logits_shift = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift, n_prefix);
                const kq_diff diff_shift = compare_kq(kq_ref, cap.data, thold);
                get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift2, n_prefix);
                const kq_diff diff_shift2 = compare_kq(kq_ref, cap.data, thold);
                const std::vector<float> logits_ctrl = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_ctrl, n_prefix + gap);
                const kq_diff diff_ctrl = compare_kq(kq_ref, cap.data, thold);

                const double nmse_logits = nmse(logits_ref, logits_shift);
                snprintf(shift_str,  sizeof(shift_str),  "%.2e", diff_shift.worst);
                snprintf(shift2_str, sizeof(shift2_str), "%.2e", diff_shift2.worst);
                snprintf(logits_str, sizeof(logits_str), "%.2e", nmse_logits);
                snprintf(ctrl_str,   sizeof(ctrl_str),   "%.2e", diff_ctrl.worst);

                if (kq_ref.empty()) {
                    status = "\033[1;33mSKIP\033[0m (no kq captured)";
                } else if (!diff_shift.first_bad.empty() || !diff_shift2.first_bad.empty()) {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m" + (diff_shift.bad.empty() ? " x2:" + diff_shift2.bad : diff_shift.bad);
                } else if (!(diff_ctrl.worst > 10*thold)) {
                    status = "\033[1;33mOK\033[0m (low sensitivity)";
                } else {
                    status = "\033[1;32mOK\033[0m";
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%10s|%10s|%10s|%10s|%s\n", shift_str, shift2_str, logits_str, ctrl_str, status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    const char * target_backend = nullptr;
    bool kv_shift = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llama_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--kv-shift") == 0) {
            kv_shift = true;
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    init_normal_pool(seed, stdev);

    try {
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        if (kv_shift) {
            return test_kv_shift(arch_filter, seed, stdev, verbosity);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
