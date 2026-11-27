// checks the tensors that a caller claims (llama_model_params.claimed_tensors):
//   - a NO_DATA tensor is created as metadata at its natural shape and strides, with no buffer and no data, also where
//     it is declared (llama_model_params.declared_tensors); a NOT_CREATED tensor is never created
//   - the range of a claimed tensor is not kept mapped, and loading still reaches its end: the claimed bytes are not
//     among the bytes that loading reports progress over
//   - bad claims are refused, and the MoE cache is refused with tensors claimed with no data
//
// the models are the MoE models written by test-llama-archs (-o <dir>)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

// the test checks whether the model has tensors claimed with no data
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

static std::string g_log;

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-100s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

// a tensor of a model file
struct file_tensor {
    std::string name;
    ggml_type   type;
    int64_t     ne[GGML_MAX_DIMS];
    size_t      offs;   // from the start of the file
    size_t      nbytes;
};

static std::vector<file_tensor> read_file_tensors(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_context_ptr ctx(gguf_init_from_file(path.c_str(), { /*.no_alloc =*/ true, /*.ctx =*/ &meta }));
    if (!ctx) {
        throw std::runtime_error("failed to read " + path);
    }
    ggml_context_ptr meta_ptr(meta);

    std::vector<file_tensor> res;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
        const char * name = gguf_get_tensor_name(ctx.get(), i);
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        file_tensor ft;
        ft.name   = name;
        ft.type   = t->type;
        std::copy(t->ne, t->ne + GGML_MAX_DIMS, ft.ne);
        ft.offs   = gguf_get_data_offset(ctx.get()) + gguf_get_tensor_offset(ctx.get(), i);
        ft.nbytes = ggml_nbytes(t);
        res.push_back(ft);
    }
    return res;
}

static const file_tensor * find(const std::vector<file_tensor> & tensors, const std::string & name) {
    for (const auto & ft : tensors) {
        if (ft.name == name) {
            return &ft;
        }
    }
    return nullptr;
}

// writes the model at path without the tensor 'removed', and with a copy of its data named 'extra', which the model
// does not use
static void write_derived(const std::string & path, const std::string & removed, const char * extra, const std::string & out) {
    ggml_context * data = nullptr;
    gguf_context_ptr src(gguf_init_from_file(path.c_str(), { /*.no_alloc =*/ false, /*.ctx =*/ &data }));
    if (!src) {
        throw std::runtime_error("failed to read " + path);
    }
    ggml_context_ptr data_ptr(data);

    gguf_context_ptr dst(gguf_init_empty());
    gguf_set_kv(dst.get(), src.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(src.get()); ++i) {
        const char * name = gguf_get_tensor_name(src.get(), i);
        if (removed != name) {
            gguf_add_tensor(dst.get(), ggml_get_tensor(data, name));
        }
    }

    const ggml_tensor * rm = ggml_get_tensor(data, removed.c_str());
    ggml_context_ptr extra_ctx(ggml_init({ /*.mem_size =*/ ggml_tensor_overhead() + ggml_nbytes(rm) + 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false }));
    ggml_tensor * t_extra = ggml_dup_tensor(extra_ctx.get(), rm);
    memcpy(t_extra->data, rm->data, ggml_nbytes(rm));
    ggml_set_name(t_extra, extra);
    gguf_add_tensor(dst.get(), t_extra);

    if (!gguf_write_to_file(dst.get(), out.c_str(), false)) {
        throw std::runtime_error("failed to write " + out);
    }
}

struct load_result {
    llama_model_ptr model;
    float last_progress = -1.0f;
    float max_progress  = -1.0f;
};

static bool on_progress(float progress, void * ud) {
    auto * res = (load_result *) ud;
    res->last_progress = progress;
    res->max_progress  = std::max(res->max_progress, progress);
    return true;
}

// loads on the CPU, mapped and locked: the claimed ranges must stay out of the lock and be unmapped after the load
static void load_model(load_result & res, const std::string & path, const std::function<void(llama_model_params &)> & set_params) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers                = 0;
    mparams.load_mode                   = LLAMA_LOAD_MODE_MMAP_MLOCK;
    mparams.progress_callback           = on_progress;
    mparams.progress_callback_user_data = &res;
    if (set_params) {
        set_params(mparams);
    }
    g_log.clear();
    res.model.reset(llama_model_load_from_file(path.c_str(), mparams));
}

static bool refused(const std::string & path, const std::function<void(llama_model_params &)> & set_params, const char * message) {
    load_result res;
    load_model(res, path, set_params);
    return !res.model && g_log.find(message) != std::string::npos;
}

static bool has_natural_strides(const ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type) &&
        t->nb[1] == ggml_row_size(t->type, t->ne[0]) &&
        t->nb[2] == t->nb[1]*t->ne[1] &&
        t->nb[3] == t->nb[2]*t->ne[2];
}

// the tensor of the model named like ft, with ft's type and shape, natural strides, no buffer and no data
static bool is_no_data(const llama_model * model, const file_tensor & ft) {
    const ggml_tensor * t = llama_model_get_tensor(model, ft.name.c_str());
    return t != nullptr && t->type == ft.type && std::equal(ft.ne, ft.ne + GGML_MAX_DIMS, t->ne) &&
        has_natural_strides(t) && t->buffer == nullptr && t->data == nullptr && t->view_src == nullptr;
}

static bool loads_to_the_end(const load_result & res) {
    return res.last_progress == 1.0f && res.max_progress <= 1.0f;
}

#ifndef _WIN32
// whether the page that holds p is mapped in the process
static bool page_is_mapped(const uint8_t * p) {
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    return msync((void *) ((uintptr_t) p / page * page), (size_t) page, MS_ASYNC) == 0;
}

// checks that the file ranges of 'claimed' are unmapped after the load, against the mapping that holds the loaded
// tensor 'anchor' (a CPU tensor wrapped in place in the mapping)
static void check_unmapped(const llama_model * model, const file_tensor & anchor, const std::vector<const file_tensor *> & claimed,
        const std::string & label, checker & check) {
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);

    const ggml_tensor * t = llama_model_get_tensor(model, anchor.name.c_str());
    const uint8_t * base = t != nullptr && t->data != nullptr ? (const uint8_t *) t->data - anchor.offs : nullptr;
    const bool found = base != nullptr && (uintptr_t) base % page == 0 && page_is_mapped((const uint8_t *) t->data);
    check(found, label + ": the loaded tensor " + anchor.name + " lies in the mapping of the file");
    if (!found) {
        return;
    }

    size_t n_pages  = 0;
    bool   unmapped = true;
    for (const file_tensor * ft : claimed) {
        const size_t first = (ft->offs + page - 1)/page*page;
        const size_t last  = (ft->offs + ft->nbytes)/page*page;
        for (size_t offs = first; offs < last; offs += page) {
            unmapped = unmapped && !page_is_mapped(base + offs);
            n_pages++;
        }
    }
    if (n_pages == 0) {
        printf("  %-100s SKIP\n", (label + ": no claimed range holds a whole page").c_str());
        return;
    }
    check(unmapped, label + ": the " + std::to_string(n_pages) + " whole pages of the claimed ranges are unmapped");
}
#endif

static void test_model(const std::string & dir, const std::string & name, checker & check) {
    const std::string path = dir + "/" + name;
    const std::string stem = std::filesystem::path(path).stem().string();
    printf("%s\n", stem.c_str());

    const std::vector<file_tensor> tensors = read_file_tensors(path);
    const std::regex expert_weight(R"(^blk\.\d+\.ffn_(gate|up|gate_up|down)_exps\.weight$)");
    std::vector<const file_tensor *> experts;
    for (const auto & ft : tensors) {
        if (std::regex_match(ft.name, expert_weight)) {
            experts.push_back(&ft);
        }
    }
    const file_tensor * anchor = find(tensors, "output_norm.weight");
    if (experts.empty() || anchor == nullptr) {
        throw std::runtime_error("the model has no expert weights or no output_norm.weight");
    }
    const std::string e0 = experts.front()->name;

    // a stock load
    {
        load_result res;
        load_model(res, path, nullptr);
        check(res.model && !res.model->has_no_data_tensors() && loads_to_the_end(res),
            stem + ": a stock load reaches its end, without tensors claimed with no data");
    }

    // every expert weight claimed with no data
    {
        std::vector<llama_tensor_claim> claims;
        for (const file_tensor * ft : experts) {
            claims.push_back({ ft->name.c_str(), LLAMA_TENSOR_CLAIM_NO_DATA });
        }
        // a repeated claim with the same flags is accepted
        claims.push_back({ e0.c_str(), LLAMA_TENSOR_CLAIM_NO_DATA });

        load_result res;
        load_model(res, path, [&](llama_model_params & p) { p.claimed_tensors = claims.data(); p.n_claimed_tensors = claims.size(); });
        check((bool) res.model, stem + ": loads with " + std::to_string(experts.size()) + " expert weights claimed with no data");
        if (!res.model) {
            fprintf(stderr, "%s", g_log.c_str());
            return;
        }

        bool all_no_data = true;
        for (const file_tensor * ft : experts) {
            all_no_data = all_no_data && is_no_data(res.model.get(), *ft);
        }
        check(all_no_data, stem + ": each one is metadata at the file's shape and natural strides, with no buffer and no data");
        check(res.model->has_no_data_tensors(), stem + ": the model has tensors claimed with no data");
        check(loads_to_the_end(res), stem + ": the load reaches its end without the claimed bytes");
#ifndef _WIN32
        check_unmapped(res.model.get(), *anchor, experts, stem, check);
#endif

        // the MoE cache reads the expert weights' data, which the model does not hold
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx          = 0;
        cparams.n_ubatch       = 64;
        cparams.moe_cache_size = 1024*1024;
        g_log.clear();
        llama_context_ptr lctx(llama_init_from_model(res.model.get(), cparams));
        check(!lctx && g_log.find("MoE cache cannot be used") != std::string::npos,
            stem + ": the MoE cache is refused with tensors claimed with no data");
    }

    // a declared tensor claimed with no data, and a tensor of the files that the model does not use, claimed as never
    // created: the file holds the extra tensor in place of the declared one
    {
        const file_tensor & removed = *experts.front();
        const char * extra = "claims.extra.weight";
        const std::string derived = dir + "/" + stem + "-claims.gguf";
        write_derived(path, removed.name, extra, derived);
        const std::vector<file_tensor> derived_tensors = read_file_tensors(derived);
        const file_tensor * extra_ft  = find(derived_tensors, extra);
        const file_tensor * anchor_ft = find(derived_tensors, anchor->name);

        llama_tensor_decl decl = { removed.name.c_str(), removed.type, {} };
        std::copy(removed.ne, removed.ne + GGML_MAX_DIMS, decl.ne);
        const llama_tensor_claim both[] = {
            { removed.name.c_str(), LLAMA_TENSOR_CLAIM_NO_DATA },
            { extra,                LLAMA_TENSOR_CLAIM_NOT_CREATED },
        };

        {
            load_result res;
            load_model(res, derived, [&](llama_model_params & p) {
                p.declared_tensors = &decl; p.n_declared_tensors = 1;
                p.claimed_tensors = both; p.n_claimed_tensors = 2;
            });
            check((bool) res.model, stem + ": loads with a declared tensor claimed with no data and an unused tensor claimed");
            if (res.model) {
                check(is_no_data(res.model.get(), removed), stem + ": the declared tensor is metadata at its natural strides, with no buffer and no data");
                check(llama_model_get_tensor(res.model.get(), extra) == nullptr, stem + ": the tensor claimed as never created is not in the model");
                check(loads_to_the_end(res), stem + ": the load reaches its end without the declared and claimed bytes");
#ifndef _WIN32
                check_unmapped(res.model.get(), *anchor_ft, { extra_ft }, stem + " (derived)", check);
#endif
            } else {
                fprintf(stderr, "%s", g_log.c_str());
            }
        }

        const struct {
            const char * what;
            const char * message;
            std::function<void(llama_model_params &)> set;
        } cases[] = {
            { "an unused tensor of the files that is not claimed", "wrong number of tensors", [&](llama_model_params & p) {
                p.declared_tensors = &decl; p.n_declared_tensors = 1; p.claimed_tensors = both; p.n_claimed_tensors = 1; } },
            { "a declared tensor that is not claimed", "not claimed with no data", [&](llama_model_params & p) {
                p.declared_tensors = &decl; p.n_declared_tensors = 1; p.claimed_tensors = both + 1; p.n_claimed_tensors = 1; } },
        };
        for (const auto & c : cases) {
            check(refused(derived, c.set, c.message), stem + ": " + c.what + " is refused");
        }

        const llama_tensor_claim plain[] = { { removed.name.c_str(), LLAMA_TENSOR_CLAIM_NOT_CREATED }, both[1] };
        check(refused(derived, [&](llama_model_params & p) {
                p.declared_tensors = &decl; p.n_declared_tensors = 1; p.claimed_tensors = plain; p.n_claimed_tensors = 2; },
            "declared, but not claimed with no data"), stem + ": a declared tensor claimed as never created is refused");

        std::filesystem::remove(derived);
    }

    // bad claims and declarations of the stock file
    {
        const llama_tensor_decl in_files = { "token_embd.weight", GGML_TYPE_F32, { 1, 1, 1, 1 } };
        const llama_tensor_decl stray    = { "stray.weight",      GGML_TYPE_F32, { 32, 1, 1, 1 } };
        const llama_tensor_claim created[]   = { { e0.c_str(), LLAMA_TENSOR_CLAIM_NOT_CREATED } };
        const llama_tensor_claim missing[]   = { { "no.such.weight", LLAMA_TENSOR_CLAIM_NO_DATA } };
        const llama_tensor_claim unknown[]   = { { e0.c_str(), 1u << 1 } };
        const llama_tensor_claim conflict[]  = { { e0.c_str(), LLAMA_TENSOR_CLAIM_NO_DATA }, { e0.c_str(), LLAMA_TENSOR_CLAIM_NOT_CREATED } };
        const auto claim = [](const llama_tensor_claim * c, size_t n) {
            return [c, n](llama_model_params & p) { p.claimed_tensors = c; p.n_claimed_tensors = n; };
        };
        const struct {
            const char * what;
            const char * message;
            std::function<void(llama_model_params &)> set;
        } cases[] = {
            { "a tensor claimed as never created that the model creates", "but the model creates it", claim(created, 1) },
            { "a claimed tensor not in the files", "not in the model files", claim(missing, 1) },
            { "a claim with an unknown flag", "unknown flags", claim(unknown, 1) },
            { "a tensor claimed twice with different flags", "claimed twice with different flags", claim(conflict, 2) },
            { "a declared tensor of the files", "already in the model files", [&](llama_model_params & p) { p.declared_tensors = &in_files; p.n_declared_tensors = 1; } },
            { "a declared tensor the model does not use", "wrong number of tensors", [&](llama_model_params & p) { p.declared_tensors = &stray; p.n_declared_tensors = 1; } },
        };
        for (const auto & c : cases) {
            check(refused(path, c.set, c.message), stem + ": " + c.what + " is refused");
        }

        // without an output weight, the model creates the token embeddings again as the output
        if (find(tensors, "output.weight") == nullptr) {
            const llama_tensor_claim embd[] = { { "token_embd.weight", LLAMA_TENSOR_CLAIM_NO_DATA } };
            check(refused(path, claim(embd, 1), "creates it as a duplicate"), stem + ": a duplicated tensor claimed with no data is refused");
        }
    }
}

int main(int argc, char ** argv) {
    std::string dir_models;
    if (argc == 3 && strcmp(argv[1], "--models") == 0) {
        dir_models = argv[2];
    }
    if (dir_models.empty()) {
        fprintf(stderr, "usage: %s --models <dir written by test-llama-archs -o>\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    // the log of a load is printed when it fails
    llama_log_set([](ggml_log_level, const char * text, void *) {
        g_log += text;
    }, nullptr);

    // separate gate and up experts, fused gate_up experts, and fused gate_up experts with a per-expert down scale
    const char * models[] = { "qwen3moe-moe.gguf", "qwen35moe-moe.gguf", "gemma4-moe.gguf" };

    checker check;
    for (const char * name : models) {
        try {
            if (!std::filesystem::exists(dir_models + "/" + name)) {
                throw std::runtime_error("missing " + dir_models + "/" + name);
            }
            test_model(dir_models, name, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", name, e.what());
            check(false, std::string(name) + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
