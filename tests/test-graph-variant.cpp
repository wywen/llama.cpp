// Checks that the graph variant (llama_set_graph_variant) keys graph reuse.
//
// For a dense and a MoE generated model, a context decodes one token at a time, counting the calls of its graph
// rewrite for a computed graph (GGML_BACKEND_SCHED_REWRITE_ALLOC). A decode under the variant of the previous one
// reuses its graph and does not call the rewrite; a decode under another variant builds the graph again and calls it
// once.

#include "llama.h"
#include "llama-cpp.h"

#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

const char * const variant_models[] = { "llama-dense.gguf", "qwen3moe-moe.gguf" };

struct rewrite_count {
    size_t n_alloc = 0;
};

bool count_rewrite(ggml_backend_sched_t /*sched*/, ggml_cgraph * /*gf*/, ggml_backend_sched_rewrite_purpose purpose, void * user_data) {
    static_cast<rewrite_count *>(user_data)->n_alloc += purpose == GGML_BACKEND_SCHED_REWRITE_ALLOC;
    return true;
}

bool install_count_rewrite(ggml_backend_sched_t sched, llama_context * /*ctx*/, void * user_data) {
    ggml_backend_sched_set_graph_rewrite_callback(sched, count_rewrite, user_data);
    return true;
}

struct step {
    uint32_t variant;
    bool     rebuilds;
};

const step steps[] = {
    { 0, true  },
    { 0, false },
    { 0, false },
    { 1, true  },
    { 1, false },
    { 7, true  },
    { 0, true  },
    { 0, false },
};

int check_model(const std::string & path) {
    const std::string name = std::filesystem::path(path).filename().string();

    llama_model_params mparams = llama_model_default_params();
    static ggml_backend_dev_t no_devices[] = { nullptr };
    mparams.devices = no_devices;
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "%s: could not load\n", name.c_str());
        return 1;
    }

    rewrite_count count;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                      = 256;
    cparams.n_batch                    = 64;
    cparams.n_ubatch                   = 64;
    cparams.n_seq_max                  = 1;
    cparams.n_threads                  = 2;
    cparams.n_threads_batch            = 2;
    cparams.cb_sched_new               = install_count_rewrite;
    cparams.cb_sched_new_user_data     = &count;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fprintf(stderr, "%s: could not create a context\n", name.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    llama_token         token = 1 % llama_vocab_n_tokens(vocab);

    int n_fail = 0;
    for (size_t i = 0; i < sizeof(steps)/sizeof(steps[0]); ++i) {
        const step & s = steps[i];
        llama_set_graph_variant(ctx.get(), s.variant);
        const size_t  before = count.n_alloc;
        const int32_t reused = llama_perf_context(ctx.get()).n_reused;
        if (llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) != 0) {
            fprintf(stderr, "%s: decode %zu failed\n", name.c_str(), i);
            return n_fail + 1;
        }
        const size_t  calls    = count.n_alloc - before;
        const int32_t n_reused = llama_perf_context(ctx.get()).n_reused - reused;
        const bool    ok       = s.rebuilds ? calls == 1 && n_reused == 0 : calls == 0 && n_reused == 1;
        printf("  %-24s decode %zu, variant %u: %zu rewrite calls, %d reused, expected %s  %s\n", name.c_str(), i,
            s.variant, calls, n_reused, s.rebuilds ? "a rebuild" : "a reuse", ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
    return n_fail;
}

} // namespace

int main(int argc, char ** argv) {
    std::string models_dir;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            models_dir = argv[++i];
        } else {
            fprintf(stderr, "usage: %s --models DIR\n", argv[0]);
            return 1;
        }
    }
    if (models_dir.empty() || !std::filesystem::is_directory(models_dir)) {
        fprintf(stderr, "%s: --models must name a directory of generated models\n", argv[0]);
        return 1;
    }

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    int n_fail = 0;
    for (const char * file : variant_models) {
        const std::filesystem::path path = std::filesystem::path(models_dir) / file;
        if (!std::filesystem::exists(path)) {
            fprintf(stderr, "missing %s\n", path.string().c_str());
            ++n_fail;
            continue;
        }
        n_fail += check_model(path.string());
    }

    llama_backend_free();
    printf("%s\n", n_fail == 0 ? "OK" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
