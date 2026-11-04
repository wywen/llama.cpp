// checks that a MoE model whose expert weights are stored one tensor per expert, blk.N.ffn_<kind>_exps.<e>.weight,
//     loads like the same model with dense expert weights: the same weights and bitwise identical logits in every
//     load mode, and that a file with malformed per-expert tensors is refused
//
// the models are the MoE models written by test-llama-archs (-o <dir>), rewritten here with per-expert tensors

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

// the test checks the strides and source records of the loaded tensors
#include "../src/llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t PER_EXPERT_ALIGNMENT = 16384;

static std::string g_log;

// the position of a dense expert weight among the expert weights of its layer, input projections first, or -1
static int expert_weight_rank(const std::string & name) {
    static const std::regex re(R"(^blk\.\d+\.ffn_(gate_up|gate|up|down)_exps\.weight$)");
    static const char * kinds[] = { "gate_up", "gate", "up", "down" };
    std::smatch m;
    if (!std::regex_match(name, m, re)) {
        return -1;
    }
    for (int r = 0; r < 4; ++r) {
        if (m[1].str() == kinds[r]) {
            return r;
        }
    }
    return -1;
}

static int layer_of(const std::string & name) {
    int il = -1;
    sscanf(name.c_str(), "blk.%d.", &il);
    return il;
}

static std::string part_name(const std::string & name, int64_t expert) {
    const std::string stem = name.substr(0, name.size() - strlen(".weight"));
    return stem + "." + std::to_string(expert) + ".weight";
}

// a GGUF file read with its tensor data
struct gguf_file {
    gguf_context_ptr gguf;
    ggml_context_ptr ctx;

    explicit gguf_file(const std::string & path) {
        ggml_context * ctx_raw = nullptr;
        gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_raw };
        gguf.reset(gguf_init_from_file(path.c_str(), params));
        if (!gguf) {
            throw std::runtime_error("failed to read " + path);
        }
        ctx.reset(ctx_raw);
    }

    int64_t n_tensors() const {
        return gguf_get_n_tensors(gguf.get());
    }

    ggml_tensor * tensor(int64_t i) const {
        return ggml_get_tensor(ctx.get(), gguf_get_tensor_name(gguf.get(), i));
    }
};

// the KV pairs of `src` in a context whose tensors are placed at PER_EXPERT_ALIGNMENT
// the alignment of a context is read from a file, so the KV pairs go through one
static gguf_context_ptr aligned_kv(const gguf_context * src, const std::string & path_tmp) {
    gguf_context_ptr kv(gguf_init_empty());
    gguf_set_kv(kv.get(), src);
    gguf_set_val_u32(kv.get(), GGUF_KEY_GENERAL_ALIGNMENT, PER_EXPERT_ALIGNMENT);
    if (!gguf_write_to_file(kv.get(), path_tmp.c_str(), /*only_meta =*/ true)) {
        throw std::runtime_error("failed to write " + path_tmp);
    }
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr res(gguf_init_from_file(path_tmp.c_str(), params));
    std::filesystem::remove(path_tmp);
    if (!res || gguf_get_alignment(res.get()) != PER_EXPERT_ALIGNMENT) {
        throw std::runtime_error("failed to read back " + path_tmp);
    }
    return res;
}

// a defect of the per-expert tensors of the first expert weight in the file
enum class defect {
    NONE,
    MISSING_FIRST, // expert 0 has no tensor
    MISSING_LAST,  // the last expert has no tensor
    TYPE_MISMATCH, // expert 1 has another type
    BOTH_LAYOUTS,  // the dense tensor is stored as well
};

static const char * defect_name(defect d) {
    switch (d) {
        case defect::NONE:          return "none";
        case defect::MISSING_FIRST: return "missing first expert";
        case defect::MISSING_LAST:  return "missing last expert";
        case defect::TYPE_MISMATCH: return "mismatched expert type";
        case defect::BOTH_LAYOUTS:  return "dense and per-expert tensors";
    }
    return "?";
}

// write `src` with each dense expert weight stored as one tensor per expert, written expert by expert at the
//     position of the layer's first expert weight, the input projections of an expert before its down projection
// returns the names of the dense expert weights
static std::vector<std::string> write_per_expert(const gguf_file & src, const std::string & path, defect d) {
    std::map<int, std::vector<std::pair<int, ggml_tensor *>>> layers; // layer -> (rank, dense tensor)
    std::vector<std::string> dense;
    int64_t n_parts = 0;
    for (int64_t i = 0; i < src.n_tensors(); ++i) {
        ggml_tensor * t = src.tensor(i);
        const int rank = expert_weight_rank(t->name);
        if (rank >= 0) {
            GGML_ASSERT(t->ne[3] == 1);
            layers[layer_of(t->name)].emplace_back(rank, t);
            dense.push_back(t->name);
            n_parts += t->ne[2];
        }
    }
    if (dense.empty()) {
        throw std::runtime_error("no expert weights");
    }
    for (auto & [_, members] : layers) {
        std::sort(members.begin(), members.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    }

    ggml_init_params ip = { /*.mem_size =*/ (size_t) (n_parts + 1)*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context_ptr ctx_parts(ggml_init(ip));
    gguf_context_ptr out = aligned_kv(src.gguf.get(), path + ".kv");

    const ggml_tensor * defective = layers.begin()->second.front().second;
    for (int64_t i = 0; i < src.n_tensors(); ++i) {
        ggml_tensor * t = src.tensor(i);
        if (expert_weight_rank(t->name) < 0) {
            gguf_add_tensor(out.get(), t);
            continue;
        }
        const auto & members = layers.at(layer_of(t->name));
        const bool first_of_layer = std::none_of(members.begin(), members.end(), [&](const auto & m) {
            return gguf_find_tensor(src.gguf.get(), m.second->name) < i;
        });
        if (!first_of_layer) {
            continue;
        }
        const int64_t n_expert = t->ne[2];
        for (int64_t e = 0; e < n_expert; ++e) {
            for (const auto & [_, m] : members) {
                GGML_ASSERT(m->ne[2] == n_expert);
                if (m == defective && ((d == defect::MISSING_FIRST && e == 0) || (d == defect::MISSING_LAST && e == n_expert - 1))) {
                    continue;
                }
                ggml_type type = m->type;
                if (m == defective && d == defect::TYPE_MISMATCH && e == 1) {
                    // F16 rows are half the F32 rows, so the data stays within the source tensor
                    GGML_ASSERT(m->type == GGML_TYPE_F32);
                    type = GGML_TYPE_F16;
                }
                ggml_tensor * p = ggml_new_tensor_2d(ctx_parts.get(), type, m->ne[0], m->ne[1]);
                p->data = (uint8_t *) m->data + e*m->nb[2];
                ggml_set_name(p, part_name(m->name, e).c_str());
                gguf_add_tensor(out.get(), p);
            }
        }
        if (d == defect::BOTH_LAYOUTS && members.front().second == defective) {
            gguf_add_tensor(out.get(), defective);
        }
    }
    if (!gguf_write_to_file(out.get(), path.c_str(), false)) {
        throw std::runtime_error("failed to write " + path);
    }
    return dense;
}

// write `src` with its expert weights quantized: the input projections to Q4_K and down to Q8_0
static void write_quantized(const gguf_file & src, const std::string & path) {
    ggml_init_params ip = { /*.mem_size =*/ (size_t) src.n_tensors()*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context_ptr ctx_out(ggml_init(ip));
    gguf_context_ptr out(gguf_init_empty());
    gguf_set_kv(out.get(), src.gguf.get());

    std::vector<std::vector<uint8_t>> quantized;
    quantized.reserve(src.n_tensors());
    for (int64_t i = 0; i < src.n_tensors(); ++i) {
        const ggml_tensor * t = src.tensor(i);
        const int rank = expert_weight_rank(t->name);
        ggml_tensor * u = nullptr;
        if (rank < 0 || t->type != GGML_TYPE_F32) {
            u = ggml_dup_tensor(ctx_out.get(), t);
            u->data = t->data;
        } else {
            const ggml_type type = rank == 3 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_K;
            if (t->ne[0] % ggml_blck_size(type) != 0) {
                throw std::runtime_error(std::string(t->name) + " has rows that are not whole " + ggml_type_name(type) + " blocks");
            }
            u = ggml_new_tensor(ctx_out.get(), type, GGML_MAX_DIMS, t->ne);
            quantized.emplace_back(ggml_nbytes(u));
            ggml_quantize_chunk(type, (const float *) t->data, quantized.back().data(), 0, ggml_nrows(t), t->ne[0], nullptr);
            u->data = quantized.back().data();
        }
        ggml_set_name(u, t->name);
        gguf_add_tensor(out.get(), u);
    }
    if (!gguf_write_to_file(out.get(), path.c_str(), false)) {
        throw std::runtime_error("failed to write " + path);
    }
}

struct load_case {
    llama_load_mode load_mode;
    bool            check_tensors;
    bool            use_extra_bufts; // a repacked CPU weight is set whole and takes other kernels
    bool            experts_on_cpu;  // a tensor override keeps the expert weights in a CPU buffer
};

static std::string case_name(const load_case & c) {
    std::string res = llama_load_mode_name(c.load_mode);
    if (c.check_tensors) {
        res += ", checked";
    }
    if (!c.use_extra_bufts) {
        res += ", no repack";
    }
    if (c.experts_on_cpu) {
        res += ", experts on CPU";
    }
    return res;
}

struct progress_trace {
    float last = 0.0f;
    bool  monotonic = true;
};

static bool record_progress(float p, void * ud) {
    auto * trace = (progress_trace *) ud;
    trace->monotonic = trace->monotonic && p >= trace->last && p <= 1.0f;
    trace->last = p;
    return true;
}

struct loaded_model {
    llama_model_ptr    model;
    progress_trace     progress;
    std::vector<float> logits;
    std::string        log;
};

static std::vector<llama_token> get_tokens(uint32_t n_tokens, uint32_t n_vocab, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static llama_model_ptr load_model(const std::string & path, const load_case & c, std::vector<ggml_backend_dev_t> devs, progress_trace * progress) {
    static const llama_model_tensor_buft_override experts_on_cpu[] = {
        { "\\.ffn_(gate_up|gate|up|down)_exps\\.weight$", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode                    = c.load_mode;
    mparams.check_tensors                = c.check_tensors;
    mparams.use_extra_bufts              = c.use_extra_bufts;
    mparams.progress_callback            = record_progress;
    mparams.progress_callback_user_data  = progress;
    if (c.experts_on_cpu) {
        mparams.tensor_buft_overrides = experts_on_cpu;
    }
    if (devs.empty()) {
        mparams.n_gpu_layers = 0;
    } else {
        devs.push_back(nullptr);
        mparams.devices = devs.data();
    }
    g_log.clear();
    return llama_model_ptr(llama_model_load_from_file(path.c_str(), mparams));
}

// the logits of `tokens` in one batch, and then of more tokens one at a time, which take the vector kernels
static std::vector<float> get_logits(llama_model * model) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 0;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = 64;
    llama_context_ptr lctx(llama_init_from_model(model, cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create a context");
    }

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> tokens = get_tokens(16, n_vocab, 42);
    const size_t n_batch = 12;

    std::vector<float> ret;
    common_batch batch(lctx.get());
    for (size_t i = 0; i < n_batch; i++) {
        batch.add(tokens[i], i, 0, true);
    }
    if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode");
    }
    for (size_t i = 0; i < n_batch; i++) {
        const float * l = llama_get_logits_ith(lctx.get(), i);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    for (size_t i = n_batch; i < tokens.size(); i++) {
        common_batch one(lctx.get());
        one.add(tokens[i], i, 0, true);
        if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, one.get())) {
            throw std::runtime_error("failed to decode");
        }
        const float * l = llama_get_logits_ith(lctx.get(), 0);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    return ret;
}

static loaded_model load_and_run(const std::string & path, const load_case & c, const std::vector<ggml_backend_dev_t> & devs) {
    loaded_model res;
    res.model = load_model(path, c, devs, &res.progress);
    res.log = g_log;
    if (!res.model) {
        throw std::runtime_error("failed to load " + path + ":\n" + res.log);
    }
    res.logits = get_logits(res.model.get());
    return res;
}

static const ggml_tensor * find_tensor(const llama_model & model, const std::string & name) {
    for (const auto & [n, t] : model.tensors_by_name) {
        if (n == name) {
            return t;
        }
    }
    return nullptr;
}

// the bytes of a weight, or nothing for a repacked CPU weight, whose buffer cannot be read back
static bool read_weight(const ggml_tensor * t, std::vector<uint8_t> & out) {
    out.resize(ggml_nbytes(t));
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(out.data(), t->data, out.size());
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    return true;
}

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-80s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

// load the dense and the per-expert file and compare them
static void compare_loads(const std::string & path_dense, const std::string & path_parts, const std::vector<std::string> & dense_names,
        const load_case & c, const std::vector<ggml_backend_dev_t> & devs, const std::string & stem, checker & check) {
    const std::string label = stem + " (" + case_name(c) + ")";
    const loaded_model ref   = load_and_run(path_dense, c, devs);
    const loaded_model parts = load_and_run(path_parts, c, devs);

    check(ref.logits.size() == parts.logits.size() && memcmp(ref.logits.data(), parts.logits.data(), ref.logits.size()*sizeof(float)) == 0,
        label + ": logits are bitwise identical");

    // the dense tensors: contiguous, the same bytes, and without a source of their own
    bool dense_ok  = true;
    bool bytes_ok  = true;
    bool source_ok = true;
    size_t n_compared = 0;
    for (const std::string & name : dense_names) {
        const ggml_tensor * a = find_tensor(*ref.model, name);
        const ggml_tensor * b = find_tensor(*parts.model, name);
        if (a == nullptr || b == nullptr) {
            fprintf(stderr, "%s: %s is not loaded\n", label.c_str(), name.c_str());
            dense_ok = false;
            continue;
        }
        dense_ok = dense_ok && ggml_is_contiguous(b) && b->type == a->type && ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b);

        std::vector<uint8_t> bytes_a;
        std::vector<uint8_t> bytes_b;
        if (read_weight(a, bytes_a) && read_weight(b, bytes_b)) {
            bytes_ok = bytes_ok && bytes_a == bytes_b;
            n_compared++;
        }

        // the stock tensor has its own file range, the assembled one has none, and each expert keeps its own record
        const llama_tensor_source_info * src_a = ref.model->tensor_source_at(a->src_ordinal);
        source_ok = source_ok && src_a != nullptr && src_a->name == name && src_a->size == ggml_nbytes(a);
        source_ok = source_ok && b->src_ordinal == GGML_TENSOR_SRC_ORDINAL_NONE && parts.model->tensor_source_at(b->src_ordinal) == nullptr;
        for (int64_t e = 0; e < b->ne[2]; ++e) {
            const std::string pname = part_name(name, e);
            const bool recorded = std::any_of(parts.model->tensor_sources.begin(), parts.model->tensor_sources.end(), [&](const auto & s) {
                return s.name == pname && s.size == b->nb[2];
            });
            source_ok = source_ok && recorded;
        }
    }
    check(dense_ok,  label + ": expert weights are dense, with the stock shape and strides");
    if (n_compared > 0) {
        check(bytes_ok, label + ": expert weights are bitwise identical (" + std::to_string(n_compared) + " compared)");
    }
    check(source_ok, label + ": assembled weights report no source, their experts do");

    check(llama_model_size(parts.model.get()) == llama_model_size(ref.model.get()) &&
          llama_model_n_params(parts.model.get()) == llama_model_n_params(ref.model.get()),
        label + ": model size and parameter count match");
    check(parts.progress.monotonic && parts.progress.last == 1.0f, label + ": load progress rises to 1");
    check(parts.log.find("unused tensor") == std::string::npos, label + ": every per-expert tensor is used");
}

// a malformed per-expert file is refused with an error that names the problem
static void check_refusal(const gguf_file & src, const std::string & dir, const std::string & stem, defect d, const char * expected, checker & check) {
    const std::string path = dir + "/" + stem + "-per-expert-bad.gguf";
    write_per_expert(src, path, d);
    progress_trace progress;
    llama_model_ptr model = load_model(path, { LLAMA_LOAD_MODE_MMAP, false, true, false }, {}, &progress);
    const bool named = g_log.find(expected) != std::string::npos;
    if (model || !named) {
        fprintf(stderr, "%s", g_log.c_str());
    }
    check(!model && named, stem + ": " + defect_name(d) + " is refused");
    std::filesystem::remove(path);
}

static void test_model(const std::string & path_dense, const std::string & dir, const std::vector<load_case> & cases,
        const std::vector<ggml_backend_dev_t> & devs, bool with_refusals, checker & check) {
    const std::string stem       = std::filesystem::path(path_dense).stem().string();
    const std::string path_parts = dir + "/" + stem + "-per-expert.gguf";

    const gguf_file src(path_dense);
    const std::vector<std::string> dense_names = write_per_expert(src, path_parts, defect::NONE);
    printf("%s: %zu expert weights of %" PRId64 " experts\n", stem.c_str(), dense_names.size(),
        ggml_get_tensor(src.ctx.get(), dense_names.front().c_str())->ne[2]);

    for (const load_case & c : cases) {
        compare_loads(path_dense, path_parts, dense_names, c, devs, stem, check);
    }
    std::filesystem::remove(path_parts);

    if (with_refusals) {
        check_refusal(src, dir, stem, defect::MISSING_FIRST, "is missing",                 check);
        check_refusal(src, dir, stem, defect::MISSING_LAST,  "per-expert tensors",         check);
        check_refusal(src, dir, stem, defect::TYPE_MISMATCH, "but tensor",                 check);
        check_refusal(src, dir, stem, defect::BOTH_LAYOUTS,  "both whole and per expert",  check);
    }
}

int main(int argc, char ** argv) {
    std::string dir_models;
    bool all_devices = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            dir_models = argv[++i];
        } else if (strcmp(argv[i], "--all-devices") == 0) {
            all_devices = true;
        } else {
            dir_models.clear();
            break;
        }
    }
    if (dir_models.empty()) {
        fprintf(stderr, "usage: %s --models <dir written by test-llama-archs -o> [--all-devices]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    // the log of a load is printed when a check of it fails
    llama_log_set([](ggml_log_level, const char * text, void *) {
        g_log += text;
    }, nullptr);

    // the CPU (no devices), or with --all-devices every device that a model can be offloaded to
    std::vector<ggml_backend_dev_t> devs;
    if (all_devices) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                devs.push_back(dev);
            }
        }
        if (devs.empty()) {
            fprintf(stderr, "--all-devices: no GPU device\n");
            return 1;
        }
    }

    // separate gate and up experts, fused gate_up experts, and fused gate_up experts with a per-expert down scale
    const char * models[] = { "qwen3moe-moe.gguf", "qwen35moe-moe.gguf", "gemma4-moe.gguf" };

    const std::vector<load_case> cases = {
        { LLAMA_LOAD_MODE_MMAP,      false, true, false },
        { LLAMA_LOAD_MODE_NONE,      false, true, false },
        { LLAMA_LOAD_MODE_DIRECT_IO, false, true, false },
        { LLAMA_LOAD_MODE_MMAP,      true,  true, false },
        { LLAMA_LOAD_MODE_NONE,      true,  true, false },
        { LLAMA_LOAD_MODE_MMAP,      false, true, true  },
        { LLAMA_LOAD_MODE_NONE,      false, true, true  },
    };
    // Q4_K and Q8_0 weights are repacked on the CPU, and read back without repacking
    const std::vector<load_case> cases_quant = {
        { LLAMA_LOAD_MODE_MMAP,      false, true,  false },
        { LLAMA_LOAD_MODE_NONE,      true,  true,  false },
        { LLAMA_LOAD_MODE_DIRECT_IO, false, true,  false },
        { LLAMA_LOAD_MODE_MMAP,      false, false, false },
        { LLAMA_LOAD_MODE_NONE,      false, true,  true  },
    };

    checker check;
    for (const char * name : models) {
        const std::string path = dir_models + "/" + name;
        try {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("missing " + path);
            }
            test_model(path, dir_models, cases, devs, /*with_refusals =*/ true, check);

            const std::string stem       = std::filesystem::path(path).stem().string();
            const std::string path_quant = dir_models + "/" + stem + "-q4_k-q8_0.gguf";
            write_quantized(gguf_file(path), path_quant);
            test_model(path_quant, dir_models, cases_quant, devs, /*with_refusals =*/ false, check);
            std::filesystem::remove(path_quant);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", name, e.what());
            check(false, std::string(name) + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
