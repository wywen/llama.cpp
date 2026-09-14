// Checks llama_model_memory_plan against the memory real context construction builds.
//
// For every generated model and a table of context parameters, the plan must describe exactly the
// buffers and tensors the context constructor allocates, and must refuse exactly the parameters
// context creation refuses.

#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-alloc.h"
#include "../src/llama-model.h"

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct context_case {
    const char *          name;
    uint32_t              n_ctx;
    uint32_t              n_seq_max;
    bool                  kv_unified;
    llama_flash_attn_type flash_attn_type;
    ggml_type             type_k;
    ggml_type             type_v;
    uint32_t              n_batch;
    uint32_t              n_ubatch;
    uint32_t              n_rs_seq;
    bool                  swa_full;
};

const context_case context_cases[] = {
    { "train-ctx",        0, 1, false, LLAMA_FLASH_ATTN_TYPE_DISABLED, GGML_TYPE_F16,  GGML_TYPE_F16,  2048, 512, 0, false },
    { "q8-k-streams",  1000, 3, false, LLAMA_FLASH_ATTN_TYPE_ENABLED,  GGML_TYPE_Q8_0, GGML_TYPE_F16,  2048,  64, 2, false },
    { "q8-v-unified",   700, 3, true,  LLAMA_FLASH_ATTN_TYPE_AUTO,     GGML_TYPE_F16,  GGML_TYPE_Q8_0,  100,   0, 1, true  },
    { "q4-kv-rounded",  300, 4, false, LLAMA_FLASH_ATTN_TYPE_AUTO,     GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,   64,  32, 0, false },
};

llama_context_params make_context_params(const context_case & c) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx           = c.n_ctx;
    params.n_seq_max       = c.n_seq_max;
    params.kv_unified      = c.kv_unified;
    params.flash_attn_type = c.flash_attn_type;
    params.type_k          = c.type_k;
    params.type_v          = c.type_v;
    params.n_batch         = c.n_batch;
    params.n_ubatch        = c.n_ubatch;
    params.n_rs_seq        = c.n_rs_seq;
    params.swa_full        = c.swa_full;
    params.n_threads       = 2;
    params.n_threads_batch = 2;
    return params;
}

enum class model_load {
    allocated,
    no_alloc,
    vocab_only,
};

llama_model_ptr load_cpu_model(const std::string & path, model_load load) {
    llama_model_params params = llama_model_default_params();
    static ggml_backend_dev_t no_devices[] = { nullptr };
    params.devices    = no_devices;
    params.no_alloc   = load == model_load::no_alloc;
    params.vocab_only = load == model_load::vocab_only;
    if (params.no_alloc) {
        // lazily mapped tensors cannot be loaded without allocation
        params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    return llama_model_ptr(llama_model_load_from_file(path.c_str(), params));
}

// collects the differences between a plan and what construction produced
class mismatch_log {
public:
    explicit mismatch_log(std::string subject) : subject(std::move(subject)) {}

    void add(const char * fmt, ...) {
        char buf[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        lines.emplace_back(buf);
    }

    bool empty() const { return lines.empty(); }

    void print() const {
        for (const auto & line : lines) {
            fprintf(stderr, "FAIL %s: %s\n", subject.c_str(), line.c_str());
        }
    }

private:
    std::string subject;
    std::vector<std::string> lines;
};

void compare_tensor(mismatch_log & log, size_t ib, const llama_memory_plan_tensor & planned, const llama_memory_plan_tensor & built) {
    const char * name = built.name.c_str();
    if (planned.name != built.name) {
        log.add("buffer %zu: tensor %s planned as %s", ib, name, planned.name.c_str());
    }
    if (planned.il != built.il) {
        log.add("buffer %zu: tensor %s layer %d planned as %d", ib, name, built.il, planned.il);
    }
    if (planned.type != built.type) {
        log.add("buffer %zu: tensor %s type %s planned as %s", ib, name, ggml_type_name(built.type), ggml_type_name(planned.type));
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (planned.ne[i] != built.ne[i] || planned.nb[i] != built.nb[i]) {
            log.add("buffer %zu: tensor %s dim %d ne/nb %" PRId64 "/%zu planned as %" PRId64 "/%zu",
                    ib, name, i, built.ne[i], built.nb[i], planned.ne[i], planned.nb[i]);
        }
    }
    if (planned.nbytes != built.nbytes) {
        log.add("buffer %zu: tensor %s nbytes %zu planned as %zu", ib, name, built.nbytes, planned.nbytes);
    }
}

void compare_buffers(
        mismatch_log & log,
        const std::vector<llama_memory_plan_buffer> & planned,
        const std::vector<llama_memory_plan_buffer> & built) {
    if (planned.size() != built.size()) {
        log.add("%zu buffers built, %zu planned", built.size(), planned.size());
        return;
    }
    for (size_t ib = 0; ib < built.size(); ++ib) {
        const auto & p = planned[ib];
        const auto & b = built[ib];
        if (p.kind != b.kind) {
            log.add("buffer %zu: kind %d planned as %d", ib, b.kind, p.kind);
        }
        if (p.buft != b.buft) {
            log.add("buffer %zu: type %s planned as %s", ib, ggml_backend_buft_name(b.buft), ggml_backend_buft_name(p.buft));
        }
        if (p.size != b.size) {
            log.add("buffer %zu: %zu bytes allocated, %zu planned", ib, b.size, p.size);
        }
        size_t tensor_bytes = 0;
        for (const auto & t : p.tensors) {
            tensor_bytes += t.nbytes;
        }
        if (tensor_bytes > p.size) {
            log.add("buffer %zu: planned tensors take %zu bytes, more than the planned %zu", ib, tensor_bytes, p.size);
        }
        if (p.tensors.size() != b.tensors.size()) {
            log.add("buffer %zu: %zu tensors built, %zu planned", ib, b.tensors.size(), p.tensors.size());
            continue;
        }
        for (size_t it = 0; it < b.tensors.size(); ++it) {
            compare_tensor(log, ib, p.tensors[it], b.tensors[it]);
        }
    }
}

// the context memory bytes llama reports per buffer type, independently of any recording
void compare_breakdown(mismatch_log & log, const llama_memory_plan & plan, const llama_context * ctx) {
    std::map<ggml_backend_buffer_type_t, size_t> planned;
    for (const auto & buf : plan.buffers) {
        planned[buf.buft] += buf.size;
    }
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        const auto it = planned.find(buft);
        const size_t planned_bytes = it == planned.end() ? 0 : it->second;
        if (mb.context != planned_bytes) {
            log.add("%s: context breakdown reports %zu bytes, %zu planned", ggml_backend_buft_name(buft), mb.context, planned_bytes);
        }
        planned.erase(buft);
    }
    for (const auto & [buft, bytes] : planned) {
        if (bytes > 0) {
            log.add("%s: %zu bytes planned in a buffer type the context does not report", ggml_backend_buft_name(buft), bytes);
        }
    }
}

void compare_shape(mismatch_log & log, const llama_memory_plan & plan, const llama_context * ctx) {
    const auto check = [&](const char * field, uint32_t built, uint32_t planned) {
        if (built != planned) {
            log.add("%s %u planned as %u", field, built, planned);
        }
    };
    check("n_ctx",     llama_n_ctx(ctx),     plan.n_ctx);
    check("n_ctx_seq", llama_n_ctx_seq(ctx), plan.n_ctx_seq);
    check("n_seq_max", llama_n_seq_max(ctx), plan.n_seq_max);
    check("n_ubatch",  llama_n_ubatch(ctx),  plan.n_ubatch);
    check("n_rs_seq",  llama_n_rs_seq(ctx),  plan.n_rs_seq);

    // on the CPU the flash attention probe keeps an automatic request enabled, so the built flag is also the resolved one
    const llama_cparams & cparams = ctx->get_cparams();
    check("flash_attn",  cparams.flash_attn,  plan.flash_attn);
    check("kv_unified",  cparams.kv_unified,  plan.kv_unified);
    check("offload_kqv", cparams.offload_kqv, plan.offload_kqv);

    const bool built_memory = llama_get_memory(ctx) != nullptr;
    if (built_memory != plan.has_memory) {
        log.add("memory built = %d, planned = %d", built_memory, plan.has_memory);
    }
}

struct outcome {
    size_t compared = 0; // cases where both a plan and a context were produced and compared
    size_t refused  = 0; // cases both the plan and context creation refused
    size_t failed   = 0;

    std::map<llama_memory_plan_buffer_kind, size_t> kinds_compared;
    size_t shared_kv_compared = 0; // compared cases whose model has layers reusing another layer's KV
};

std::optional<llama_memory_plan> try_plan(const llama_model * model, const llama_context_params & params, std::string & error) {
    try {
        return llama_model_memory_plan(model, params);
    } catch (const std::exception & e) {
        error = e.what();
        return std::nullopt;
    }
}

// plans a context, then builds it for real while recording the buffers the constructor allocates
void check_case(outcome & out, const std::string & subject, llama_model * model, const llama_context_params & params) {
    mismatch_log log(subject);

    const bool no_alloc_before = model->hparams.no_alloc;

    std::string plan_error;
    const std::optional<llama_memory_plan> plan = try_plan(model, params, plan_error);

    if (model->hparams.no_alloc != no_alloc_before) {
        log.add("planning changed hparams.no_alloc from %d", no_alloc_before);
    }

    std::vector<llama_memory_plan_buffer> built;
    llama_context_ptr ctx;
    {
        llama_memory_alloc_recorder recorder(llama_memory_alloc_mode::observe);
        ctx.reset(llama_init_from_model(model, params));
        built = recorder.release();
    }

    if (!plan && !ctx) {
        out.refused++;
    } else if (!plan) {
        log.add("context created but the plan refused: %s", plan_error.c_str());
    } else if (!ctx) {
        log.add("plan produced but context creation failed");
    } else {
        out.compared++;
        for (const auto & buf : plan->buffers) {
            out.kinds_compared[buf.kind]++;
        }
        const int32_t n_layer_kv = model->hparams.n_layer_kv_from_start;
        if (n_layer_kv >= 0 && (uint32_t) n_layer_kv < model->hparams.n_layer()) {
            out.shared_kv_compared++;
        }
        compare_shape(log, *plan, ctx.get());
        compare_buffers(log, plan->buffers, built);
        compare_breakdown(log, *plan, ctx.get());
    }

    if (!log.empty()) {
        log.print();
        out.failed++;
    }
}

#ifndef _WIN32
size_t peak_rss_bytes() {
    rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return (size_t) usage.ru_maxrss;
#else
    return (size_t) usage.ru_maxrss * 1024;
#endif
}

// a plan whose buffers exceed what the process could hold must still complete without growing the process
bool check_plan_does_not_allocate(const std::string & path) {
    const llama_model_ptr model = load_cpu_model(path, model_load::allocated);
    if (!model) {
        fprintf(stderr, "FAIL %s: could not load\n", path.c_str());
        return false;
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = 1u << 20;

    const size_t rss_before = peak_rss_bytes();
    const llama_memory_plan plan = llama_model_memory_plan(model.get(), params);
    const size_t rss_growth = peak_rss_bytes() - rss_before;

    size_t planned = 0;
    for (const auto & buf : plan.buffers) {
        planned += buf.size;
    }

    printf("large plan: %zu MiB planned, peak RSS grew %zu MiB\n", planned >> 20, rss_growth >> 20);

    if (planned < (1u << 30) || rss_growth > planned / 8) {
        fprintf(stderr, "FAIL large plan: %zu bytes planned should exceed 1 GiB while peak RSS grows by under an eighth of it (%zu)\n",
                planned, rss_growth);
        return false;
    }
    return true;
}
#else
bool check_plan_does_not_allocate(const std::string &) {
    printf("large plan: SKIPPED, peak RSS is not measured on Windows\n");
    return true;
}
#endif

bool check_vocab_only(const std::string & path) {
    const llama_model_ptr model = load_cpu_model(path, model_load::vocab_only);
    if (!model) {
        fprintf(stderr, "FAIL %s: could not load vocab-only\n", path.c_str());
        return false;
    }
    const llama_memory_plan plan = llama_model_memory_plan(model.get(), llama_context_default_params());
    if (plan.has_memory || !plan.buffers.empty()) {
        fprintf(stderr, "FAIL vocab-only %s: planned memory (has_memory = %d, %zu buffers)\n",
                path.c_str(), plan.has_memory, plan.buffers.size());
        return false;
    }
    return true;
}

bool check_null_model_throws() {
    try {
        llama_model_memory_plan(nullptr, llama_context_default_params());
    } catch (const std::runtime_error &) {
        return true;
    }
    fprintf(stderr, "FAIL null model: planning did not throw\n");
    return false;
}

// a planning recorder on one thread must neither capture nor replace the allocations of a context built on another
bool check_recorder_is_thread_local(const std::string & path) {
    const llama_model_ptr model = load_cpu_model(path, model_load::allocated);
    if (!model) {
        fprintf(stderr, "FAIL %s: could not load\n", path.c_str());
        return false;
    }

    llama_memory_alloc_recorder recorder(llama_memory_alloc_mode::plan);

    size_t allocated_layers = 0;
    std::thread builder([&]() {
        const llama_context_ptr ctx(llama_init_from_model(model.get(), llama_context_default_params()));
        const auto * cache = ctx ? dynamic_cast<const llama_kv_cache *>(llama_get_memory(ctx.get())) : nullptr;
        if (cache == nullptr) {
            return;
        }
        // the breakdown reports a placeholder at its would-be size, so look for tensor data instead
        for (const uint32_t il : cache->get_layer_ids()) {
            allocated_layers += cache->get_k_storage((int32_t) il)->data != nullptr;
        }
    });
    builder.join();

    if (allocated_layers == 0 || !recorder.buffers().empty()) {
        fprintf(stderr, "FAIL thread isolation: other thread allocated K storage for %zu layers, this thread recorded %zu buffers\n",
                allocated_layers, recorder.buffers().size());
        return false;
    }
    return true;
}

// a planned cache sharing cells with a live cache must record the same buffers a real one allocates,
// while leaving the live cache's cell metadata untouched
bool check_plan_leaves_shared_cells_alone(const std::string & path) {
    const llama_model_ptr model = load_cpu_model(path, model_load::allocated);
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 256;
    const llama_context_ptr ctx(model ? llama_init_from_model(model.get(), params) : nullptr);
    const auto * source = ctx ? dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx.get())) : nullptr;
    if (source == nullptr) {
        fprintf(stderr, "FAIL shared cells: %s did not build a plain KV cache\n", path.c_str());
        return false;
    }

    llama_token tokens[] = { 1, 2, 3, 4 };
    if (llama_decode(ctx.get(), llama_batch_get_one(tokens, (int32_t) std::size(tokens))) != 0) {
        fprintf(stderr, "FAIL shared cells: decode failed\n");
        return false;
    }
    const llama_pos pos_live = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);

    // layer 0 views the source's tensors, every other layer gets its own; the requested size yields to the source's
    const auto build_sharing_cache = [&](llama_memory_alloc_mode mode) {
        llama_memory_alloc_recorder recorder(mode);
        const llama_kv_cache cache(*model, model->hparams, GGML_TYPE_F16, GGML_TYPE_F16, true, false, false,
                2*source->get_size(), 1, 1, 0, LLAMA_SWA_TYPE_NONE, llama_get_memory(ctx.get()),
                nullptr, nullptr, [](int32_t il) { return il == 0 ? 0 : -1; });
        return recorder.release();
    };

    mismatch_log log("shared cells");

    const std::vector<llama_memory_plan_buffer> planned = build_sharing_cache(llama_memory_alloc_mode::plan);
    const llama_pos pos_after_plan = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);
    if (pos_live != 3 || pos_after_plan != pos_live) {
        log.add("live cache seq 0 ends at %d after decoding 4 tokens and at %d after planning", pos_live, pos_after_plan);
    }

    // a real sharing cache resets the source's cells by design, which also shows the probe above can see a reset
    const std::vector<llama_memory_plan_buffer> built = build_sharing_cache(llama_memory_alloc_mode::observe);
    const llama_pos pos_after_build = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);
    if (pos_after_build != -1) {
        log.add("control: building a real sharing cache left seq 0 ending at %d instead of resetting it", pos_after_build);
    }

    compare_buffers(log, planned, built);
    if (planned.empty()) {
        log.add("no buffer planned for the unshared layers");
    }

    log.print();
    return log.empty();
}

} // namespace

int main(int argc, char ** argv) {
    std::string models_dir;
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else {
            fprintf(stderr, "usage: %s --models DIR [-v]\n", argv[0]);
            return 1;
        }
    }
    if (models_dir.empty() || !std::filesystem::is_directory(models_dir)) {
        fprintf(stderr, "%s: --models must name a directory of generated models\n", argv[0]);
        return 1;
    }

    if (!verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
    llama_backend_init();

    std::vector<std::string> models;
    for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            models.push_back(entry.path().string());
        }
    }
    std::sort(models.begin(), models.end());
    if (models.empty()) {
        fprintf(stderr, "%s: no .gguf models in %s\n", argv[0], models_dir.c_str());
        return 1;
    }

    const std::string dense = (std::filesystem::path(models_dir) / "llama-dense.gguf").string();

    bool ok = check_plan_does_not_allocate(dense);
    ok = check_vocab_only(dense) && ok;
    ok = check_null_model_throws() && ok;
    ok = check_recorder_is_thread_local(dense) && ok;
    ok = check_plan_leaves_shared_cells_alone(dense) && ok;

    size_t models_loaded = 0;
    outcome out;

    for (const auto & path : models) {
        const std::string file = std::filesystem::path(path).filename().string();
        for (const model_load load : { model_load::allocated, model_load::no_alloc }) {
            llama_model_ptr model = load_cpu_model(path, load);
            if (!model) {
                fprintf(stderr, "FAIL %s: could not load\n", file.c_str());
                ok = false;
                continue;
            }
            models_loaded++;
            for (const auto & c : context_cases) {
                const llama_context_params params = make_context_params(c);
                const std::string subject = file + (load == model_load::no_alloc ? " no_alloc " : " ") + c.name;
                check_case(out, subject, model.get(), params);
            }
        }
    }

    printf("compared buffers by kind: kv %zu, kv_swa %zu, recurrent %zu, dsv4_state %zu\n",
            out.kinds_compared[LLAMA_MEMORY_PLAN_BUFFER_KV], out.kinds_compared[LLAMA_MEMORY_PLAN_BUFFER_KV_SWA],
            out.kinds_compared[LLAMA_MEMORY_PLAN_BUFFER_RECURRENT], out.kinds_compared[LLAMA_MEMORY_PLAN_BUFFER_DSV4_STATE]);
    printf("compared cases with shared KV layers: %zu\n", out.shared_kv_compared);
    printf("%zu model loads x %zu context cases: %zu compared, %zu refused by both, %zu failed\n",
            models_loaded, std::size(context_cases), out.compared, out.refused, out.failed);

    ok = ok && out.failed == 0;

    // the fixture models cover every buffer kind; a kind never compared means a model family went missing
    if (out.compared == 0) {
        fprintf(stderr, "FAIL no case was compared\n");
        ok = false;
    }
    const std::pair<llama_memory_plan_buffer_kind, const char *> fixture_kinds[] = {
        { LLAMA_MEMORY_PLAN_BUFFER_KV,         "kv"         },
        { LLAMA_MEMORY_PLAN_BUFFER_KV_SWA,     "kv_swa"     },
        { LLAMA_MEMORY_PLAN_BUFFER_RECURRENT,  "recurrent"  },
        { LLAMA_MEMORY_PLAN_BUFFER_DSV4_STATE, "dsv4_state" },
    };
    for (const auto & [kind, name] : fixture_kinds) {
        if (out.kinds_compared[kind] == 0) {
            fprintf(stderr, "FAIL no %s buffer was compared\n", name);
            ok = false;
        }
    }
    if (out.shared_kv_compared == 0) {
        fprintf(stderr, "FAIL no model with shared KV layers was compared\n");
        ok = false;
    }
    printf("test-memory-plan: %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
