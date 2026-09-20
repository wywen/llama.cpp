#include "../src/llama-ext.h"
#include "common.h"
#include "ggml-backend.h"
#include "speculative.h"

#include <cstdio>
#include <exception>

namespace {

struct backend_owner {
    ~backend_owner() { llama_backend_free(); }
};

struct batch_owner {
    llama_batch batch;

    explicit batch_owner(int32_t capacity) : batch(llama_batch_init(capacity, 0, 1)) {}

    ~batch_owner() { llama_batch_free(batch); }

    void set(int32_t index, llama_token token, llama_pos pos, bool logits) const {
        batch.token[index]     = token;
        batch.pos[index]       = pos;
        batch.n_seq_id[index]  = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index]    = logits;
    }
};

struct sampler_owner {
    llama_sampler * sampler = llama_sampler_init_greedy();

    ~sampler_owner() { llama_sampler_free(sampler); }
};

struct output_snapshot {
    ggml_backend_buffer_t buffer;
    size_t                bytes;
};

bool same_sizes(const common_speculative_mtp_buffer_sizes & lhs, const common_speculative_mtp_buffer_sizes & rhs) {
    return lhs.hidden_bytes == rhs.hidden_bytes && lhs.batch_bytes == rhs.batch_bytes &&
           lhs.sampling_bytes == rhs.sampling_bytes && lhs.sequence_bytes == rhs.sequence_bytes;
}

int fail(const char * message) {
    fprintf(stderr, "test-mtp-workspace: %s\n", message);
    return 1;
}

int run(const char * target_path, const char * draft_path) {
    common_params params;
    params.model.path        = target_path;
    params.n_ctx             = 128;
    params.n_batch           = 32;
    params.n_ubatch          = 32;
    params.n_parallel        = 1;
    params.n_sequences       = 1;
    params.warmup            = false;
    params.fit_params        = false;
    params.no_perf           = true;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    if (draft_path != nullptr) {
        params.speculative.draft.mparams.path = draft_path;
    }
    params.speculative.draft.n_max = 3;
    params.speculative.draft.p_min = 0.0f;
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams, &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams_batch, &params.cpuparams_batch);

    const auto output_limits     = common_speculative_get_output_limits(params.n_batch, 1, 3);
    params.n_outputs_max         = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    auto target = common_init_from_params(params);
    if (!target || !target->model() || !target->context()) {
        return fail("could not initialize target model and context");
    }
    llama_context * ctx_tgt = target->context();

    auto draft_params = common_base_params_to_speculative(params);
    auto draft        = common_speculative_init_from_params(draft_params, target->model(), ctx_tgt);
    if (!draft || !draft->context()) {
        return fail("could not initialize MTP context");
    }
    llama_context * ctx_dft = draft->context();

    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = ctx_dft;
    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
    if (!spec) {
        return fail("could not initialize MTP driver");
    }

    if (!llama_output_reserve(ctx_tgt, 4) || !llama_output_reserve(ctx_dft, 1)) {
        return fail("could not reserve configured target and draft output buffers");
    }

    ggml_backend_buffer_t target_buffer = llama_get_output_buffer(ctx_tgt);
    ggml_backend_buffer_t draft_buffer  = llama_get_output_buffer(ctx_dft);
    if (target_buffer == nullptr || draft_buffer == nullptr) {
        return fail("constructor did not reserve both output buffers");
    }
    const output_snapshot target_output{ target_buffer, ggml_backend_buffer_get_size(target_buffer) };
    const output_snapshot draft_output{ draft_buffer, ggml_backend_buffer_get_size(draft_buffer) };

    common_speculative_mtp_buffer_sizes before;
    if (!common_speculative_get_mtp_buffer_sizes(spec.get(), before)) {
        return fail("could not query MTP workspace sizes");
    }
    if (before.hidden_bytes == 0 || before.batch_bytes == 0 || before.sampling_bytes == 0 ||
        before.sequence_bytes == 0 || before.total() == 0) {
        return fail("MTP workspace query omitted an auxiliary buffer category");
    }

    llama_tokens prompt = common_tokenize(ctx_tgt, "The capital of France is", true, true);
    if (prompt.empty() || prompt.size() > (size_t) params.n_batch) {
        return fail("prompt does not fit the configured batch");
    }

    batch_owner   owner(params.n_batch);
    llama_batch & batch = owner.batch;
    batch.n_tokens      = (int32_t) prompt.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        owner.set(i, prompt[i], i, i + 1 == batch.n_tokens);
    }
    if (llama_decode(ctx_tgt, batch) != 0 || !common_speculative_process(spec.get(), batch)) {
        return fail("target prefill or MTP prefill processing failed");
    }

    common_speculative_begin(spec.get(), 0, prompt);
    sampler_owner     sampler;
    const llama_token leading = llama_sampler_sample(sampler.sampler, ctx_tgt, -1);

    llama_tokens drafted;
    drafted.reserve(3);
    const size_t draft_capacity = drafted.capacity();
    auto & request   = common_speculative_get_draft_params(spec.get(), 0);
    request.drafting = true;
    request.n_max    = 3;
    request.n_past   = (llama_pos) prompt.size();
    request.id_last  = leading;
    request.prompt   = &prompt;
    request.result   = &drafted;

    const bool      independent_draft = llama_get_ctx_other(ctx_dft) != ctx_tgt;
    const llama_pos draft_end_before  = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0);
    common_speculative_draft(spec.get());
    if (drafted.size() != 3) {
        return fail("MTP did not produce the configured three-token draft");
    }
    if (independent_draft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, draft_end_before + 1, -1)) {
        return fail("could not roll back independent draft state before verification");
    }

    batch.n_tokens = 4;
    owner.set(0, leading, (llama_pos) prompt.size(), true);
    for (int32_t i = 0; i < 3; ++i) {
        owner.set(i + 1, drafted[i], (llama_pos) prompt.size() + i + 1, true);
    }
    if (llama_decode(ctx_tgt, batch) != 0 || !common_speculative_process(spec.get(), batch)) {
        return fail("target verification or MTP verification processing failed");
    }

    int32_t accepted = 0;
    while (accepted < 3 && llama_sampler_sample(sampler.sampler, ctx_tgt, accepted) == drafted[accepted]) {
        ++accepted;
    }
    common_speculative_accept(spec.get(), 0, accepted);

    const llama_pos rollback_from = (llama_pos) prompt.size() + 1 + accepted;
    if (!llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, rollback_from, -1)) {
        return fail("could not roll back rejected target tokens");
    }
    if (independent_draft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, rollback_from, -1)) {
        return fail("could not roll back rejected independent draft tokens");
    }

    if (drafted.capacity() != draft_capacity) {
        return fail("caller-owned draft capacity changed during one round");
    }

    common_speculative_mtp_buffer_sizes after;
    if (!common_speculative_get_mtp_buffer_sizes(spec.get(), after)) {
        return fail("could not query MTP workspace sizes after one round");
    }
    if (!same_sizes(before, after)) {
        return fail("an auxiliary MTP buffer capacity changed during one round");
    }

    target_buffer = llama_get_output_buffer(ctx_tgt);
    draft_buffer  = llama_get_output_buffer(ctx_dft);
    if (target_buffer != target_output.buffer || draft_buffer != draft_output.buffer ||
        ggml_backend_buffer_get_size(target_buffer) != target_output.bytes ||
        ggml_backend_buffer_get_size(draft_buffer) != draft_output.bytes) {
        return fail("an output buffer changed after constructor reservation");
    }

    printf("test-mtp-workspace: OK (drafted=3 accepted=%d independent-draft=%d)\n", accepted, independent_draft);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s TARGET_GGUF [DRAFT_GGUF]\n", argv[0]);
        return 2;
    }

    common_init();
    llama_backend_init();
    backend_owner backend;

    try {
        return run(argv[1], argc == 3 ? argv[2] : nullptr);
    } catch (const std::exception & error) {
        fprintf(stderr, "test-mtp-workspace: %s\n", error.what());
        return 1;
    }
}
