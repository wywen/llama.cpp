#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

// decode tokens [p0, p1) of seq 0 in one batch
static bool decode_range(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos p0, llama_pos p1) {
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (llama_pos pos = p0; pos < p1; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == p1);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// decode tokens [p0, p1) of seq 0 one at a time and append the logits of each
static bool decode_each(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos p0, llama_pos p1, int n_vocab, std::vector<float> & logits) {
    logits.clear();
    for (llama_pos pos = p0; pos < p1; ++pos) {
        if (!decode_one(ctx, tokens[pos], pos)) {
            return false;
        }
        const float * l = llama_get_logits_ith(ctx, 0);
        if (l == nullptr) {
            return false;
        }
        logits.insert(logits.end(), l, l + n_vocab);
    }
    return true;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_shared   = 8;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 2;  // less than the n_prompt - n_shared tokens of the last decode
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return llama_init_from_model(model, cparams);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, n_shared) prefill; ctx_roll decodes
    // [n_shared, n_prompt) and rolls back its tail so the restore is pending at
    // replay, ctx_ref decodes only [n_shared, p0)
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) n_shared; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = n_shared; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_ref, batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = n_shared; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = std::fabs(l_roll[t] - l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, std::fabs(l_roll[t] - l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
}

// A rollback must restore the state of a context that never decoded the removed
// tokens, and must be refused when the snapshots cannot give that state. Every
// comparison is against ctx_ref, which decodes [0, rollback_pos) in one batch and
// then the removed tokens one at a time; the other contexts replay those tokens
// the same way.
static bool test_single_seq(const common_params & params, llama_model * model, const std::vector<llama_token> & tokens, uint32_t n_rollback, int n_vocab) {
    const char *    func         = __func__;
    const llama_pos n_tokens     = (llama_pos) tokens.size();
    const llama_pos rollback_pos = n_tokens - (llama_pos) n_rollback;

    // snapshots of DeepSeek V4 also hold the state before the last ubatch, so a rollback of the whole ubatch is valid there
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    const bool rollback_whole_ubatch = strcmp(arch, "deepseek4") == 0;

    // same ubatch shapes on both sides, so a valid rollback matches bitwise
    constexpr float eps = 0.0f;

    std::vector<llama_context *> ctxs;
    const auto new_ctx = [&]() {
        llama_context * ctx = make_ctx(params, model);
        if (ctx != nullptr) {
            ctxs.push_back(ctx);
        }
        return ctx;
    };
    const auto done = [&](bool ok) {
        for (llama_context * ctx : ctxs) {
            llama_free(ctx);
        }
        return ok;
    };

    std::vector<float> logits_ref;

    const auto rollback = [](llama_context * ctx, llama_pos p0) {
        return llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, -1);
    };

    // replay [p0, n_tokens) one token at a time and compare with the reference
    const auto expect_ref = [&](llama_context * ctx, llama_pos p0, const char * name) {
        std::vector<float> logits;
        if (!decode_each(ctx, tokens, p0, n_tokens, n_vocab, logits)) {
            fprintf(stderr, "%s : %s replay failed\n", func, name);
            return false;
        }
        const std::vector<float> ref(logits_ref.end() - logits.size(), logits_ref.end());
        float diff = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            diff = std::max(diff, std::fabs(logits[i] - ref[i]));
        }
        if (diff > eps) {
            fprintf(stderr, "%s : %s logits differ from the reference (max diff %g)\n", func, name, (double) diff);
            return false;
        }
        return true;
    };

    // the rollback must return false and leave the sequence unchanged
    const auto expect_refused = [&](llama_context * ctx, llama_pos p0, const char * name) {
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (rollback(ctx, p0)) {
            fprintf(stderr, "%s : %s rollback to %d was accepted\n", func, name, p0);
            return false;
        }
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != pos_max) {
            fprintf(stderr, "%s : %s refused rollback changed the sequence\n", func, name);
            return false;
        }
        return true;
    };

    llama_context * ctx_ref = new_ctx();
    llama_context * ctx_src = new_ctx();
    llama_context * ctx_dst = new_ctx();
    if (ctx_ref == nullptr || ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return done(false);
    }

    if (!decode_range(ctx_ref, tokens, 0, rollback_pos) || !decode_each(ctx_ref, tokens, rollback_pos, n_tokens, n_vocab, logits_ref)) {
        fprintf(stderr, "%s : reference decode failed\n", __func__);
        return done(false);
    }

    // Decode the whole prompt and roll back n_rollback tokens. Replaying them
    // crosses DSV4's ratio-4 compressor boundary.
    if (!decode_range(ctx_src, tokens, 0, n_tokens) || !rollback(ctx_src, rollback_pos)) {
        fprintf(stderr, "%s : decode and rollback failed\n", __func__);
        return done(false);
    }
    if (!expect_refused(ctx_src, rollback_pos - 1, "second pending")) {
        return done(false);
    }

    // save the rolled-back state while its restore is pending
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);

    // a loaded state holds no snapshots
    ckpt.load_tgt(ctx_dst, 0, 0);
    if (!expect_refused(ctx_dst, rollback_pos - 1, "after state load") ||
        !expect_ref(ctx_src, rollback_pos, "rollback") ||
        !expect_ref(ctx_dst, rollback_pos, "loaded")) {
        return done(false);
    }

    // ctx_src last decoded a single token
    if (!expect_refused(ctx_src, n_tokens - 2, "past a single-token decode")) {
        return done(false);
    }
    if (rollback_whole_ubatch) {
        if (!rollback(ctx_src, n_tokens - 1) || !expect_ref(ctx_src, n_tokens - 1, "single-token decode")) {
            fprintf(stderr, "%s : single-token rollback failed\n", __func__);
            return done(false);
        }
    } else if (!expect_refused(ctx_src, n_tokens - 1, "single-token decode")) {
        return done(false);
    }

    // roll back every token of the last decode
    llama_context * ctx_split = new_ctx();
    if (ctx_split == nullptr ||
        !decode_range(ctx_split, tokens, 0, rollback_pos) ||
        !decode_range(ctx_split, tokens, rollback_pos, n_tokens)) {
        fprintf(stderr, "%s : split decode failed\n", __func__);
        return done(false);
    }
    if (rollback_whole_ubatch) {
        if (!rollback(ctx_split, rollback_pos) || !expect_ref(ctx_split, rollback_pos, "whole ubatch")) {
            fprintf(stderr, "%s : whole ubatch rollback failed\n", __func__);
            return done(false);
        }
    } else if (!expect_refused(ctx_split, rollback_pos, "whole ubatch")) {
        return done(false);
    }

    // Load into contexts that have their own pending rollback and snapshots: a
    // different prompt's history, or the same prompt for a partial state load.
    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
    }

    llama_context * ctx_dirty = new_ctx();
    llama_context * ctx_part  = new_ctx();
    if (ctx_dirty == nullptr || ctx_part == nullptr ||
        !decode_range(ctx_dirty, noise,  0, n_tokens) || !rollback(ctx_dirty, rollback_pos) ||
        !decode_range(ctx_part,  tokens, 0, n_tokens) || !rollback(ctx_part,  rollback_pos)) {
        fprintf(stderr, "%s : dirty context setup failed\n", __func__);
        return done(false);
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);
    ckpt_partial.load_tgt(ctx_part, 0, partial_flags);

    if (!expect_refused(ctx_dirty, rollback_pos - 1, "after load over a pending rollback") ||
        !expect_refused(ctx_part,  rollback_pos - 1, "after partial state load") ||
        !expect_ref(ctx_dirty, rollback_pos, "dirty") ||
        !expect_ref(ctx_part,  rollback_pos, "partial")) {
        return done(false);
    }

    fprintf(stderr, "%s : rollbacks matched the reference and invalid rollbacks were refused\n", __func__);
    return done(true);
}

static llama_token tok_of(uint32_t stream, llama_pos pos, int n_vocab) {
    return (llama_token) ((7*(uint32_t) pos + 31*stream + 1) % (uint32_t) n_vocab);
}

// decode tokens [p0, p1) of `stream` into seq 0 in one batch; with `logits`, every token is an output and its logits are appended
static bool decode_stream(llama_context * ctx, uint32_t stream, llama_pos p0, llama_pos p1, int n_vocab, std::vector<float> * logits) {
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (llama_pos pos = p0; pos < p1; ++pos) {
        common_batch_add(batch, tok_of(stream, pos, n_vocab), pos, { 0 }, logits != nullptr || pos + 1 == p1);
    }
    bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);

    for (int32_t i = 0; ok && logits != nullptr && i < p1 - p0; ++i) {
        const float * l = llama_get_logits_ith(ctx, i);
        ok = l != nullptr;
        if (ok) {
            logits->insert(logits->end(), l, l + n_vocab);
        }
    }
    return ok;
}

// A restored cells snapshot must place the next ubatches exactly as the
// memory would have placed them at snapshot time. Cell data is not restored,
// so each comparison is set up to read only state the detour between snapshot
// and restore left untouched, against a context that never took the detour.
static bool test_cells_snapshot(const common_params & params, llama_model * model, uint32_t n_rs_seq_req, int n_vocab) {
    const char * func = __func__;

    constexpr uint32_t n_ubatch = 16;

    std::vector<llama_context *> ctxs;
    const auto new_ctx = [&](uint32_t n_seq_max) {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seq_max;
        cparams.n_rs_seq   = n_rs_seq_req;
        cparams.n_ctx      = 256*n_seq_max;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx != nullptr) {
            ctxs.push_back(ctx);
        }
        return ctx;
    };
    const auto done = [&](bool ok) {
        for (llama_context * ctx : ctxs) {
            llama_free(ctx);
        }
        return ok;
    };

    // same ubatch shapes of more than 8 tokens on both sides, so a correct restore matches bitwise
    const auto expect_same = [&](const std::vector<float> & a, const std::vector<float> & b, const char * name) {
        float diff = a.size() == b.size() ? 0.0f : INFINITY;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            diff = std::max(diff, std::fabs(a[i] - b[i]));
        }
        if (diff > 0.0f) {
            fprintf(stderr, "%s : n_rs_seq %u: %s logits differ from the reference (max diff %g)\n", func, n_rs_seq_req, name, (double) diff);
            return false;
        }
        return true;
    };
    const auto expect_pos_max = [&](llama_context * ctx, llama_pos expected, const char * name) {
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (pos_max != expected) {
            fprintf(stderr, "%s : n_rs_seq %u: %s: seq 0 ends at %d, expected %d\n", func, n_rs_seq_req, name, pos_max, expected);
            return false;
        }
        return true;
    };

    // A fresh sequence: after the restore its cell has no source again, so the
    // next decode starts from the zeroed state and position 0, as if the
    // multi-ubatch detour never happened.
    {
        llama_context * ctx     = new_ctx(1);
        llama_context * ctx_ref = new_ctx(1);
        if (ctx == nullptr || ctx_ref == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", func);
            return done(false);
        }

        llama_memory_cells_t snap = llama_memory_cells_snapshot(llama_get_memory(ctx), 0);
        if (snap == nullptr) {
            fprintf(stderr, "%s : no cells snapshot for a recurrent memory\n", func);
            return done(false);
        }

        bool ok = decode_stream(ctx, 1, 0, 3*n_ubatch, n_vocab, nullptr) && expect_pos_max(ctx, 3*n_ubatch - 1, "fresh detour");
        llama_memory_cells_restore(llama_get_memory(ctx), 0, snap);
        llama_memory_cells_free(snap);

        std::vector<float> logits;
        std::vector<float> logits_ref;
        ok = ok && expect_pos_max(ctx, -1, "fresh restore") &&
            decode_stream(ctx,     2, 0, 2*n_ubatch, n_vocab, &logits) &&
            decode_stream(ctx_ref, 2, 0, 2*n_ubatch, n_vocab, &logits_ref) &&
            expect_same(logits, logits_ref, "fresh sequence");
        if (!ok) {
            return done(false);
        }
    }

    // A pending partial removal: the restore must make it pending again even
    // though the detour consumed it. The detour decodes n_rollback tokens, so it
    // writes snapshot planes 0 .. n_rollback - 1 and leaves plane n_rollback,
    // the one the pending removal restores, as the prompt decode left it.
    {
        llama_context * ctx     = new_ctx(1);
        llama_context * ctx_ref = new_ctx(1);
        if (ctx == nullptr || ctx_ref == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", func);
            return done(false);
        }

        const uint32_t n_rollback = std::min<uint32_t>(llama_n_rs_seq(ctx), 3);
        if (n_rollback == 0) {
            fprintf(stderr, "%s : n_rs_seq %u: skipping the pending removal case because n_rs_seq is %u\n", func, n_rs_seq_req, llama_n_rs_seq(ctx));
        } else {
            const llama_pos n_prompt = 2*n_ubatch;
            const llama_pos p0       = n_prompt - (llama_pos) n_rollback;

            bool ok =
                decode_stream(ctx,     1, 0, n_prompt, n_vocab, nullptr) && llama_memory_seq_rm(llama_get_memory(ctx),     0, p0, -1) &&
                decode_stream(ctx_ref, 1, 0, n_prompt, n_vocab, nullptr) && llama_memory_seq_rm(llama_get_memory(ctx_ref), 0, p0, -1);
            if (!ok) {
                fprintf(stderr, "%s : n_rs_seq %u: prompt decode and removal failed\n", func, n_rs_seq_req);
                return done(false);
            }

            llama_memory_cells_t snap = llama_memory_cells_snapshot(llama_get_memory(ctx), 0);
            ok = snap != nullptr &&
                decode_stream(ctx, 3, p0, n_prompt, n_vocab, nullptr) && expect_pos_max(ctx, n_prompt - 1, "pending detour");
            llama_memory_cells_restore(llama_get_memory(ctx), 0, snap);
            llama_memory_cells_free(snap);

            std::vector<float> logits;
            std::vector<float> logits_ref;
            ok = ok && expect_pos_max(ctx, p0 - 1, "pending restore") &&
                decode_stream(ctx,     2, p0, p0 + n_ubatch, n_vocab, &logits) &&
                decode_stream(ctx_ref, 2, p0, p0 + n_ubatch, n_vocab, &logits_ref) &&
                expect_same(logits, logits_ref, "pending removal");
            if (!ok) {
                return done(false);
            }
        }
    }

    // A snapshot of a memory with other cell counts must leave the cells alone.
    {
        llama_context * ctx   = new_ctx(1);
        llama_context * ctx_2 = new_ctx(2);
        if (ctx == nullptr || ctx_2 == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", func);
            return done(false);
        }

        llama_memory_cells_t snap = llama_memory_cells_snapshot(llama_get_memory(ctx_2), 0);
        bool ok = snap != nullptr && decode_stream(ctx, 1, 0, n_ubatch, n_vocab, nullptr);
        if (ok) {
            llama_memory_cells_restore(llama_get_memory(ctx), 0, snap);
        }
        llama_memory_cells_free(snap);
        if (!ok || !expect_pos_max(ctx, n_ubatch - 1, "mismatched restore")) {
            return done(false);
        }
    }

    fprintf(stderr, "%s : n_rs_seq %u: restored cells placed the next ubatches like the reference\n", func, n_rs_seq_req);
    return done(true);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx = make_ctx(params, model);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        return 1;
    }

    // memories that keep no cells snapshot: DeepSeek V4, and the hybrid memories with sliding window attention or a sparse-attention indexer
    char arch[64] = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (strcmp(arch, "deepseek4") == 0 || strcmp(arch, "qwen4exp") == 0 || llama_model_n_swa(model) > 0) {
        llama_memory_cells_t snap = llama_memory_cells_snapshot(llama_get_memory(ctx), 0);
        if (snap != nullptr) {
            fprintf(stderr, "%s : unexpected cells snapshot for %s\n", __func__, arch);
            llama_memory_cells_free(snap);
            llama_free(ctx);
            return 1;
        }
    } else {
        for (uint32_t n_rs_seq : { 0u, 2u, 8u }) {
            if (!test_cells_snapshot(params, model, n_rs_seq, n_vocab)) {
                llama_free(ctx);
                return 1;
            }
        }
    }

    const uint32_t n_rs_seq = llama_n_rs_seq(ctx);

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx, "The quick brown fox jumps over the lazy dog", true);
    }
    llama_free(ctx);

    if (n_rs_seq == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        return 0;
    }
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    // every prefill batch has more than 8 tokens, so Metal uses the same matrix kernels on both sides of a comparison
    tokens.resize(std::max<size_t>(16, n_rs_seq + 1), tokens.back());

    if (!test_single_seq(params, model, tokens, n_rollback, n_vocab)) {
        return 1;
    }

    if (!test_multi_seq_split_replay(params, model, n_vocab)) {
        return 1;
    }

    return 0;
}
