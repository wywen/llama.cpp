#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

// the logits of every generation step, and the tokens sampled from them
struct generation {
    llama_tokens                    tokens;
    std::vector<std::vector<float>> logits;
};

// samples n_predict tokens, or with forced feeds those tokens instead so that the logits of every step stay comparable
static generation generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
        const llama_tokens * forced = nullptr) {
    generation result;
    llama_batch_ptr batch(1, 0, 1);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    for (int i = 0; i < n_predict; i++) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        result.logits.emplace_back(logits, logits + n_vocab);

        auto next_token = forced != nullptr ? (*forced)[i] : llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.tokens.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

// relative difference of two logit vectors; logits masked to the same infinity on both sides are skipped
static double logits_rel(const std::vector<float> & a, const std::vector<float> & b) {
    double diff = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == b[i] && std::isinf(a[i])) {
            continue;
        }
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return INFINITY;
        }
        diff += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        norm += (double) a[i] * a[i];
    }
    return diff == 0.0 ? 0.0 : std::sqrt(diff / norm);
}

// compares a generation fed the baseline tokens with the baseline, step by step: the logits must be bitwise identical
static bool matches_baseline(const char * test, const generation & result, const generation & expected) {
    if (result.logits.size() != expected.logits.size()) {
        LOG_ERR("\n%s: error: %zu steps generated, %zu expected\n", test, result.logits.size(), expected.logits.size());
        return false;
    }
    for (size_t i = 0; i < result.logits.size(); i++) {
        const auto & a = expected.logits[i];
        const auto & b = result.logits[i];
        if (a.size() != b.size() || memcmp(a.data(), b.data(), a.size()*sizeof(float)) != 0) {
            LOG_ERR("\n%s: error: logits of step %zu differ from the baseline, by %.3e relative\n", test, i, logits_rel(a, b));
            return false;
        }
    }
    LOG("\n%s: logits of all %zu steps are bitwise identical to the baseline\n", test, result.logits.size());
    return true;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static generation test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.tokens.empty()) {
        return {};
    }

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - check that the restored state serializes to the saved bytes
// - feed the baseline tokens and compare the logits of every step with the baseline, bitwise
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // the restored state must serialize to exactly the bytes it was restored from
    {
        std::ifstream file(params.out_file, std::ios::binary);
        const std::vector<uint8_t> file_bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        // the state data follows the magic, the version, the token count and the tokens
        const size_t offset = 3*sizeof(uint32_t) + n_token_count_out*sizeof(llama_token);

        std::vector<uint8_t> state(llama_state_get_size(ctx.get()));
        state.resize(llama_state_get_data(ctx.get(), state.data(), state.size()));

        if (offset > file_bytes.size() || state.size() != file_bytes.size() - offset ||
                memcmp(state.data(), file_bytes.data() + offset, state.size()) != 0) {
            LOG_ERR("\n%s: error: the restored state (%zu bytes) does not serialize to the saved state (%zu bytes)\n",
                    __func__, state.size(), file_bytes.size() - std::min(offset, file_bytes.size()));
            return false;
        }
    }

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate the baseline tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0, &expected_result.tokens);
    if (result.tokens.empty()) {
        return false;
    }

    if (!matches_baseline(__func__, result, expected_result)) {
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.tokens.empty()) {
        return false;
    }

    if (result.tokens != expected_result.tokens) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const generation & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.tokens.empty()) {
        return false;
    }

    if (result.tokens != expected_result.tokens) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 6/7: seq copy (scatter)
// - decode the same prefix on two sequences, interleaving seq 0 cells between the seq 1 cells
// - save the seq 1 state, free the interleaved seq 0 cells, and restore via the given io path
// - the restore destination is non-contiguous: scatter reads are batched per contiguous run
// - save again on the host and compare the two blobs byte for byte
static bool test_seq_cp_scatter(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, int test_num, bool on_device) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    LOG("\n=== Test %d: seq copy (%s, scatter) ===\n", test_num, on_device ? "device" : "host");

    const uint32_t flags = on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : LLAMA_STATE_SEQ_FLAGS_NONE;

    auto decode_one = [&](llama_token tok, int pos, llama_seq_id seq) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), tok, pos, { seq }, false);
        return llama_decode(ctx.get(), batch.get()) == 0;
    };

    // seq 0 cells 0,1,4 interleave the seq 1 cells 2,3,5
    if (!decode_one(tokens[0], 0, 0) ||
        !decode_one(tokens[1], 1, 0) ||
        !decode_one(tokens[0], 0, 1) ||
        !decode_one(tokens[1], 1, 1) ||
        !decode_one(tokens[2], 2, 0) ||
        !decode_one(tokens[2], 2, 1)) {
        LOG_ERR("%s: failed to build interleaved state\n", __func__);
        return false;
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, uint32_t fl, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size_ext(ctx.get(), seq_id, fl);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), state.data(), state.size(), seq_id, fl);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    // host blob: contains the KV data, used for the byte-for-byte comparison
    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_before)) {
        return false;
    }

    // save via the io path under test
    std::vector<uint8_t> state_save;
    if (!get_seq_state(1, flags, state_save)) {
        return false;
    }
    LOG_TRC("%s: seq 1 saved via %s, %zu bytes\n", __func__, on_device ? "device" : "host", state_save.size());

    // free seq 0's cells so the ring is fragmented: the restore destination (seq 1's interleaved cells) stays non-contiguous
    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    // restore via the io path under test
    const size_t nset = llama_state_seq_set_data_ext(ctx.get(), state_save.data(), state_save.size(), 1, flags);
    if (nset != state_save.size()) {
        LOG_ERR("%s: seq set data length %zu does not match expected length %zu\n", __func__, nset, state_save.size());
        return false;
    }
    LOG_TRC("%s: seq 1 restored via %s, %zu bytes\n", __func__, on_device ? "device" : "host", nset);

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, LLAMA_STATE_SEQ_FLAGS_NONE, state_after)) {
        return false;
    }

    // the blob is serialized in sequence cell order, so identical bytes iff the restore wrote the same KV
    if (state_before.size() != state_after.size() || memcmp(state_before.data(), state_after.data(), state_before.size()) != 0) {
        LOG_ERR("\n%s: error: restored KV state is not byte-identical to the saved state\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 8: state load (fragmented)
// - decode a prefix on two sequences of one cache, a token at a time, so that their cells interleave
// - remove sequence 0, which frees a cell between every two cells of sequence 1
// - decode a few more tokens on sequence 1, which go to the freed cells, and save the whole state
// - feed n_predict tokens on sequence 1 and record the logits of every step
// - restore the state into a new context, check that it serializes to the saved bytes, feed the same tokens and
//   compare the logits of every step, bitwise
static bool test_state_load_fragmented(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    LOG("\n=== Test 8: state load (fragmented) ===\n");

    const int32_t n_vocab   = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const size_t  n_prefix  = std::min<size_t>(tokens.size(), 32);
    const size_t  n_refill  = 4;
    const size_t  n_predict = params.n_predict;

    if (tokens.size() < n_refill + n_predict) {
        LOG_ERR("%s: the prompt is too short, %zu tokens\n", __func__, tokens.size());
        return false;
    }

    // decodes one token on sequence 1 and returns its logits
    const auto decode_one = [&](llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq, std::vector<float> * logits) {
        llama_batch_ptr batch(1, 0, 1);
        common_batch_add(batch.get(), tok, pos, { seq }, logits != nullptr);
        if (llama_decode(ctx, batch.get()) != 0) {
            LOG_ERR("%s: failed to decode token %zu of sequence %d\n", __func__, (size_t) pos, seq);
            return false;
        }
        if (logits) {
            const float * l = llama_get_logits_ith(ctx, -1);
            logits->assign(l, l + n_vocab);
        }
        return true;
    };

    // the tokens fed after the state is saved, and their positions follow the refill
    const auto feed = [&](llama_context * ctx, generation & out) {
        for (size_t i = 0; i < n_predict; ++i) {
            std::vector<float> logits;
            if (!decode_one(ctx, tokens[i], (llama_pos) (n_prefix + n_refill + i), 1, &logits)) {
                return false;
            }
            out.tokens.push_back(tokens[i]);
            out.logits.push_back(std::move(logits));
        }
        return true;
    };

    std::vector<uint8_t> state;
    generation expected;
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return false;
        }

        for (size_t i = 0; i < n_prefix; ++i) {
            if (!decode_one(ctx.get(), tokens[i], (llama_pos) i, 0, nullptr) ||
                !decode_one(ctx.get(), tokens[i], (llama_pos) i, 1, nullptr)) {
                return false;
            }
        }

        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
            LOG_ERR("%s: failed to remove sequence 0\n", __func__);
            return false;
        }

        for (size_t i = 0; i < n_refill; ++i) {
            if (!decode_one(ctx.get(), tokens[n_prefix + i], (llama_pos) (n_prefix + i), 1, nullptr)) {
                return false;
            }
        }

        state.resize(llama_state_get_size(ctx.get()));
        state.resize(llama_state_get_data(ctx.get(), state.data(), state.size()));

        if (!feed(ctx.get(), expected)) {
            return false;
        }
    }

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    if (llama_state_set_data(ctx.get(), state.data(), state.size()) != state.size()) {
        LOG_ERR("\n%s: failed to restore the state\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_restored(llama_state_get_size(ctx.get()));
    state_restored.resize(llama_state_get_data(ctx.get(), state_restored.data(), state_restored.size()));

    if (state_restored != state) {
        LOG_ERR("\n%s: error: the restored state (%zu bytes) does not serialize to the saved state (%zu bytes)\n",
                __func__, state_restored.size(), state.size());
        return false;
    }

    generation result;
    if (!feed(ctx.get(), result)) {
        return false;
    }

    if (!matches_baseline(__func__, result, expected)) {
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Run the full save/load test suite (tests 1-8) for a single model.
// Returns true if all tests pass, false otherwise.
static bool run_save_load_tests_for_model(const std::string & model_path, const struct common_params & base_params) {
    struct common_params params = base_params;
    params.model.path = model_path;

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());
        return false;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.tokens.empty()) {
        return false;
    }

    // Test 2: sequence removal isolation
    if (!test_seq_rm_isolated(model, params, tokens)) {
        return false;
    }

    // Test 3: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return false;
    }

    // Test 4: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return false;
    }

    // Test 5: seq copy (device)
    if (!test_seq_cp_device(model, params, tokens, result_baseline)) {
        return false;
    }

    // Test 6: seq copy (host, scatter)
    if (!test_seq_cp_scatter(model, params, tokens, 6, false)) {
        return false;
    }

    // Test 7: seq copy (device, scatter)
    if (!test_seq_cp_scatter(model, params, tokens, 7, true)) {
        return false;
    }

    // Test 8: state load (fragmented)
    if (!test_state_load_fragmented(model, params, tokens)) {
        return false;
    }

    LOG("\nAll tests passed.\n");

    return true;
}


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;
    // a restore must continue bitwise like the uninterrupted run. a restore that puts cells elsewhere changes the logits at
    // rounding level; CPU attention without flash attention over an F32 cache adds its terms in cell order and shows that,
    // while with flash attention over an F16 cache some misplaced cells still give the same bits
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cache_type_k    = GGML_TYPE_F32;
    params.cache_type_v    = GGML_TYPE_F32;

    common_init();

    // extract our own --models DIR option before handing the rest to the common arg parser
    std::string models_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--models") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --models requires a directory argument\n", __func__);
                return 1;
            }
            models_dir = argv[i + 1];
            i++;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int)filtered_argv.size() - 1;

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    if (!models_dir.empty()) {
        // run the suite over every dummy model in the directory
        if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
            LOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
            return 1;
        }

        std::vector<std::string> models;
        for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                models.push_back(entry.path().string());
            }
        }
        std::sort(models.begin(), models.end());

        if (models.empty()) {
            LOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
            return 1;
        }

        LOG_INF("%s: running save/load tests over %zu models in '%s'\n", __func__, models.size(), models_dir.c_str());

        size_t n_pass = 0;
        size_t n_fail = 0;
        for (const auto & model_path : models) {
            LOG("\n================================================================\n");
            LOG_INF("%s: model %s\n", __func__, model_path.c_str());

            if (run_save_load_tests_for_model(model_path, params)) {
                n_pass++;
            } else {
                n_fail++;
            }
        }

        LOG("\n================================================================\n");
        LOG_INF("%s: summary: %zu passed, %zu failed (of %zu)\n", __func__, n_pass, n_fail, models.size());

        return n_fail == 0 ? 0 : 1;
    }

    // single-model mode
    return run_save_load_tests_for_model(params.model.path, params) ? 0 : 1;
}
