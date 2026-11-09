// checks that a model whose per-layer token embedding table is served by the caller (llama_model_params.lookup_tables,
//     llama_context_params.lookup_rows) computes bitwise identical logits to the same model loaded with the table, when
//     the test serves the rows itself by reading the model file through a small cache of 16 KiB-aligned file pages:
//   - a prompt in one ubatch and in several ubatches, then single-token decode steps
//   - sequences with an EOS in the middle and as the first token, which cut the n-gram context of qwen4exp
//   - two sequences in one context
//   - a prompt where only the last token requests output, which prunes rows before the layers that share KV (gemma4)
//   - a batch of input embeddings
//   - a model loaded with a locked mapping, and a model split into three files with the table in the last
// and that only the pageable table is offered, that the served table has no buffer, that each ubatch asks for
//     rows_per_token rows per token, all inside the table, that llama_model_lookup_table_rows names the rows that were
//     asked for, that corrupted row bytes change the logits, that a failing caller fails the decode, and that a context
//     without a row reader is refused
//
// the models are written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

// the test checks that the stock table has no buffer, the layers that share KV, and the n-gram EOS
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static std::string g_log;

static const char * TABLE = "per_layer_token_embd.weight";

struct checker {
    int n_fail = 0;

    void operator()(bool ok, const std::string & what) {
        printf("  %-110s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
};

// a generated model with a pageable per-layer token table
struct model_case {
    const char * file;
    bool         shares_kv;       // some layers share KV, which a prompt with one output prunes
    bool         rows_span_pages; // some rows of the table straddle two 16 KiB file pages
    bool         split;           // also check a copy split into three files

    // the stock model computes the same logits for the same embedding batch; the generated qwen4exp model does not,
    //     loaded with the table or not, so only the rows that it asks for are checked
    bool embd_logits_stable;
};

static const model_case MODELS[] = {
    { "gemma4-dense.gguf", true, true, true, true },
    { "gemma4-moe.gguf", true, true, false, true },
    { "qwen4exp-moe.gguf", false, false, true, false },
};

// the tables that the loader offered; the per-layer token table is served
static bool choose_table(const char * name, int64_t /*n_rows*/, size_t /*row_size*/, void * ud) {
    ((std::vector<std::string> *) ud)->emplace_back(name);
    return strcmp(name, TABLE) == 0;
}

static llama_model_ptr load_model(const std::string & path, std::vector<ggml_backend_dev_t> devs, std::vector<std::string> * offered,
        llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO) {
    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = load_mode;
    if (offered) {
        mparams.lookup_tables           = choose_table;
        mparams.lookup_tables_user_data = offered;
    }
    if (devs.empty()) {
        mparams.n_gpu_layers = 0;
    } else {
        devs.push_back(nullptr);
        mparams.devices = devs.data();
    }
    g_log.clear();
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        throw std::runtime_error("failed to load " + path + ":\n" + g_log);
    }
    return model;
}

// copies the model into <dir>/<stem>-split-0000k-of-00003.gguf: the per-layer token table goes to the last file, after
//     the tensor that precedes it in the model, so that its offset differs from the unsplit one; returns the first path
static std::string write_split(const std::string & path, const std::string & dir) {
    constexpr int N_SPLIT = 3;

    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
    gguf_context_ptr src(gguf_init_from_file(path.c_str(), params));
    if (!src) {
        throw std::runtime_error("failed to read " + path);
    }
    ggml_context_ptr data(ctx_data);

    const int64_t n_tensors = gguf_get_n_tensors(src.get());
    const int64_t i_table   = gguf_find_tensor(src.get(), TABLE);
    if (i_table < 1) {
        throw std::runtime_error(std::string(TABLE) + " is missing or first in " + path);
    }

    std::vector<gguf_context_ptr> out;
    for (int k = 0; k < N_SPLIT; ++k) {
        out.emplace_back(gguf_init_empty());
        if (k == 0) {
            gguf_set_kv(out[k].get(), src.get());
        }
        gguf_set_val_u16(out[k].get(), "split.no", (uint16_t) k);
        gguf_set_val_u16(out[k].get(), "split.count", (uint16_t) N_SPLIT);
        gguf_set_val_i32(out[k].get(), "split.tensors.count", (int32_t) n_tensors);
    }
    // the other tensors are halved between the first two files
    int64_t n_other = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(src.get(), i);
        int k = N_SPLIT - 1;
        if (i != i_table && i != i_table - 1) {
            k = 2*n_other++ < n_tensors ? 0 : 1;
        }
        gguf_add_tensor(out[k].get(), ggml_get_tensor(ctx_data, name));
    }

    const std::string prefix = dir + "/" + std::filesystem::path(path).stem().string() + "-split";
    std::string first;
    for (int k = 0; k < N_SPLIT; ++k) {
        char split_path[1024];
        llama_split_path(split_path, sizeof(split_path), prefix.c_str(), k, N_SPLIT);
        if (!gguf_write_to_file(out[k].get(), split_path, false)) {
            throw std::runtime_error(std::string("failed to write ") + split_path);
        }
        if (k == 0) {
            first = split_path;
        }
    }
    return first;
}

// plays the caller: serves rows from the model file through a FIFO cache of a few aligned file pages
struct row_server {
    static constexpr size_t PAGE    = 16384;
    static constexpr int    N_SLOTS = 3;

    llama_lookup_table_info info = {};
    std::ifstream file;

    std::vector<uint8_t> slots = std::vector<uint8_t>(N_SLOTS*PAGE);
    int64_t slot_page[N_SLOTS] = { -1, -1, -1 };
    int     next_victim        = 0;

    bool    fail    = false;
    bool    corrupt = false;
    int64_t n_reads = 0; // of file pages
    int64_t n_evicts = 0;
    int64_t n_split  = 0; // rows that span two file pages
    std::vector<std::vector<int32_t>> calls;

    explicit row_server(const llama_model * model) {
        if (!llama_model_lookup_table_info(model, 0, &info) || info.path == nullptr) {
            throw std::runtime_error("no served table with a file");
        }
        file.open(info.path, std::ios::binary);
        if (!file) {
            throw std::runtime_error(std::string("failed to open ") + info.path);
        }
    }

    const uint8_t * page(int64_t p) {
        for (int s = 0; s < N_SLOTS; ++s) {
            if (slot_page[s] == p) {
                return slots.data() + s*PAGE;
            }
        }
        const int s = next_victim;
        next_victim = (next_victim + 1) % N_SLOTS;
        n_evicts += slot_page[s] >= 0;
        uint8_t * dst = slots.data() + s*PAGE;
        // a positioned read of one aligned page, short at the end of the file
        file.clear();
        file.seekg((std::streamoff) (p*PAGE));
        file.read((char *) dst, PAGE);
        if (file.gcount() <= 0) {
            throw std::runtime_error("failed to read the model file");
        }
        slot_page[s] = p;
        n_reads++;
        return dst;
    }

    bool read(const int32_t * rows, int32_t n_rows, void * dst, size_t row_size) {
        calls.emplace_back(rows, rows + n_rows);
        if (fail || row_size != info.row_size) {
            return false;
        }
        for (int32_t i = 0; i < n_rows; ++i) {
            uint8_t * out = (uint8_t *) dst + i*row_size;
            const uint64_t beg = info.offset + (uint64_t) rows[i]*row_size;
            const uint64_t end = beg + row_size;
            n_split += beg/PAGE != (end - 1)/PAGE;
            for (uint64_t pos = beg; pos < end; ) {
                const uint64_t p   = pos/PAGE;
                const uint64_t lim = std::min<uint64_t>(end, (p + 1)*PAGE);
                memcpy(out + (pos - beg), page((int64_t) p) + (pos - p*PAGE), lim - pos);
                pos = lim;
            }
            if (corrupt) {
                out[row_size/2] ^= 0x40;
            }
        }
        return true;
    }

    // the rows of every call since calls was cleared, in order
    std::vector<int32_t> all_rows() const {
        std::vector<int32_t> res;
        for (const auto & c : calls) {
            res.insert(res.end(), c.begin(), c.end());
        }
        return res;
    }

    bool rows_in_table() const {
        for (const auto & c : calls) {
            for (const int32_t r : c) {
                if (r < 0 || r >= info.n_rows) {
                    return false;
                }
            }
        }
        return true;
    }
};

static bool serve_rows(int32_t table, const int32_t * rows, int32_t n_rows, void * dst, size_t row_size, void * ud) {
    return table == 0 && ((row_server *) ud)->read(rows, n_rows, dst, row_size);
}

// the widest row count of a named tensor in the graphs that a context computes
struct row_watch {
    std::string name;
    int64_t     max_rows = 0;
};

static bool watch_rows(ggml_tensor * t, bool ask, void * ud) {
    auto * w = (row_watch *) ud;
    if (w->name != ggml_get_name(t)) {
        return !ask;
    }
    if (!ask) {
        w->max_rows = std::max(w->max_rows, t->ne[1]);
    }
    return true;
}

// the context shapes that the scenarios use
struct context_shape {
    uint32_t n_ubatch;
    uint32_t n_seq_max;
};

static constexpr context_shape ONE_UBATCH = { 64, 1 };
static constexpr context_shape UBATCHES   = { 5, 1 }; // the prompt takes three ubatches
static constexpr context_shape TWO_SEQS   = { 64, 2 };

static llama_context_params context_params(row_server * server, context_shape shape) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 0;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = shape.n_ubatch;
    cparams.n_seq_max       = shape.n_seq_max;
    if (server) {
        cparams.lookup_rows           = serve_rows;
        cparams.lookup_rows_user_data = server;
    }
    return cparams;
}

static llama_context_ptr new_context(llama_model * model, row_server * server, context_shape shape) {
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, context_params(server, shape)));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

static constexpr size_t N_TOKENS = 16; // a prompt of N_BATCH tokens, then single-token decode steps
static constexpr size_t N_BATCH  = 12;

// a sequence of N_TOKENS tokens that one context decodes
struct sequence {
    std::string              what;
    context_shape            shape;
    std::vector<llama_token> tokens;
    bool                     last_only = false; // only the last token of the prompt requests output
};

// the logits of the first N_BATCH tokens in one batch, then of the other tokens one at a time, from an empty memory;
// empty if a decode fails
static std::vector<float> get_logits(llama_context * lctx, const std::vector<llama_token> & tokens, bool last_only = false) {
    llama_memory_clear(llama_get_memory(lctx), true);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));

    std::vector<float> ret;
    common_batch batch(lctx);
    for (size_t i = 0; i < N_BATCH; i++) {
        batch.add(tokens[i], i, 0, !last_only || i == N_BATCH - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        return {};
    }
    for (size_t i = last_only ? N_BATCH - 1 : 0; i < N_BATCH; i++) {
        const float * l = llama_get_logits_ith(lctx, last_only ? -1 : (int32_t) i);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    for (size_t i = N_BATCH; i < tokens.size(); i++) {
        common_batch one(lctx);
        one.add(tokens[i], i, 0, true);
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, one.get())) {
            return {};
        }
        const float * l = llama_get_logits_ith(lctx, 0);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    return ret;
}

// the logits of two sequences decoded together: their first N_BATCH tokens in one batch, then one token of each per
//     batch; empty if a decode fails
static std::vector<float> get_logits_2(llama_context * lctx, const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    llama_memory_clear(llama_get_memory(lctx), true);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));

    std::vector<float> ret;
    for (size_t beg = 0; beg < a.size(); beg = beg == 0 ? N_BATCH : beg + 1) {
        const size_t end = beg == 0 ? N_BATCH : beg + 1;
        common_batch batch(lctx);
        for (size_t i = beg; i < end; i++) {
            batch.add(a[i], i, 0, true);
        }
        for (size_t i = beg; i < end; i++) {
            batch.add(b[i], i, 1, true);
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            return {};
        }
        for (size_t i = 0; i < 2*(end - beg); i++) {
            const float * l = llama_get_logits_ith(lctx, (int32_t) i);
            ret.insert(ret.end(), l, l + n_vocab);
        }
    }
    return ret;
}

// the logits of a batch of input embeddings, which read the rows that the architecture uses for them
static std::vector<float> get_embd_logits(llama_context * lctx, size_t n_tokens) {
    llama_memory_clear(llama_get_memory(lctx), true);
    const llama_model * model = llama_get_model(lctx);
    const int32_t  n_embd  = llama_model_n_embd_inp(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init((int32_t) n_tokens, n_embd, 1);
    std::mt19937 gen(7);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < n_tokens*n_embd; ++i) {
        batch.embd[i] = dis(gen);
    }
    for (size_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }
    batch.n_tokens = (int32_t) n_tokens;
    const int32_t rc = llama_decode(lctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        return {};
    }
    std::vector<float> ret;
    for (size_t i = 0; i < n_tokens; ++i) {
        const float * l = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    return ret;
}

static bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// llama_model_lookup_table_rows of tokens[n_context, n_tokens); empty if it fails
static std::vector<int32_t> query_rows(const llama_model * model, const llama_token * tokens, size_t n_tokens, size_t n_context) {
    llama_lookup_table_info info = {};
    if (!llama_model_lookup_table_info(model, 0, &info)) {
        return {};
    }
    std::vector<int32_t> rows((n_tokens - n_context)*info.rows_per_token, -1);
    if (!llama_model_lookup_table_rows(model, 0, tokens, (int32_t) n_tokens, (int32_t) n_context, rows.data())) {
        return {};
    }
    return rows;
}

// the token that cuts the n-gram context of qwen4exp; nothing special for gemma4
static llama_token eos_of(const llama_model * model) {
    return model->hparams.ple_n_heads > 0 ? (llama_token) model->hparams.ple_eos_token_id : 0;
}

static std::vector<sequence> make_sequences(const llama_model * model) {
    const uint32_t    n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const llama_token eos     = eos_of(model);
    std::mt19937 gen(42);
    auto random_tokens = [&]() {
        // the EOS only where a scenario plants it
        std::vector<llama_token> tokens;
        while (tokens.size() < N_TOKENS) {
            const llama_token t = (llama_token) (gen() % n_vocab);
            if (t != eos) {
                tokens.push_back(t);
            }
        }
        return tokens;
    };

    std::vector<sequence> res;
    res.push_back({ "a prompt in one ubatch and decode steps", ONE_UBATCH, random_tokens() });
    res.push_back({ "a prompt in three ubatches and decode steps", UBATCHES, res[0].tokens });
    res.push_back({ "an EOS in the prompt and in the decode steps", UBATCHES, res[0].tokens });
    res.back().tokens[6]  = eos;
    res.back().tokens[13] = eos;
    res.push_back({ "an EOS as the first token", UBATCHES, random_tokens() });
    res.back().tokens[0] = eos;
    res.push_back({ "only the prompt's last output (pruned rows)", ONE_UBATCH, res[0].tokens, true });
    return res;
}

// the stock logits of each scenario, from the model loaded with the table
struct reference {
    std::vector<sequence>           seqs;
    std::vector<std::vector<float>> logits;   // [seq]
    std::vector<llama_token>        second;   // decoded with seqs[0] in the two-sequence scenario
    std::vector<float>              logits_2; // of the two-sequence scenario
    std::vector<float>              embd;
};

// checks the rows that the decode of seq asked for: rows_per_token per token of each ubatch, as the shape cuts them,
//     all named by llama_model_lookup_table_rows: of the whole sequence, of its decode steps at once, and of each token
//     from a short window of predecessors
static void check_rows(const llama_model * model, const row_server & server, const sequence & seq, const std::string & label,
        checker & check) {
    const size_t rpt = server.info.rows_per_token;

    std::vector<size_t> want_sizes;
    for (size_t beg = 0; beg < N_BATCH; beg += seq.shape.n_ubatch) {
        want_sizes.push_back(rpt*std::min<size_t>(seq.shape.n_ubatch, N_BATCH - beg));
    }
    want_sizes.resize(want_sizes.size() + N_TOKENS - N_BATCH, rpt);
    std::vector<size_t> sizes;
    for (const auto & c : server.calls) {
        sizes.push_back(c.size());
    }
    check(sizes == want_sizes && server.rows_in_table(), label + ": " + std::to_string(rpt) +
        " rows per token in each ubatch, all inside the table");

    const std::vector<int32_t> asked = server.all_rows();
    if (asked.size() != N_TOKENS*rpt) {
        check(false, label + ": the row query names the rows that were asked for");
        return;
    }
    const llama_token * t = seq.tokens.data();
    const std::vector<int32_t> decode_steps(asked.begin() + N_BATCH*rpt, asked.end());
    check(query_rows(model, t, N_TOKENS, 0) == asked && query_rows(model, t, N_TOKENS, N_BATCH) == decode_steps,
        label + ": the row query names the rows of the sequence and of its decode steps");

    // four predecessors cover the n-gram context of each token
    constexpr size_t N_WINDOW = 4;
    bool window_ok = true;
    for (size_t k = N_WINDOW; window_ok && k < N_TOKENS; ++k) {
        const std::vector<int32_t> want(asked.begin() + k*rpt, asked.begin() + (k + 1)*rpt);
        window_ok = query_rows(model, t + k - N_WINDOW, N_WINDOW + 1, N_WINDOW) == want;
    }
    check(window_ok, label + ": the row query names the rows of each token from " + std::to_string(N_WINDOW) + " predecessors");
}

static void test_served(const std::string & path, const std::string & label, const model_case & mc, const reference & ref,
        const std::vector<ggml_backend_dev_t> & devs, llama_load_mode mode, uint32_t want_file, checker & check) {
    std::vector<std::string> offered;
    llama_model_ptr model = load_model(path, devs, &offered, mode);
    check(offered == std::vector<std::string>{ TABLE } && llama_model_n_lookup_tables(model.get()) == 1,
        label + ": the loader offers only the per-layer token table, and serves it");

    llama_lookup_table_info info = {};
    const bool has_info = llama_model_lookup_table_info(model.get(), 0, &info);
    const ggml_tensor * stock = model->get_tensor(TABLE);
    const int32_t want_rpt = model->hparams.ple_n_heads > 0 ? (int32_t) model->hparams.ple_n_heads : 1;
    check(has_info && strcmp(info.name, TABLE) == 0 && stock != nullptr && stock->buffer == nullptr &&
        info.type == stock->type && info.n_cols == stock->ne[0] && info.n_rows == stock->ne[1] &&
        info.row_size == ggml_row_size(stock->type, stock->ne[0]) && info.rows_per_token == want_rpt &&
        info.path != nullptr && info.file == want_file &&
        info.offset + info.n_rows*info.row_size <= std::filesystem::file_size(info.path),
        label + ": table info (" + std::to_string(info.rows_per_token) + " rows per token, file " +
        std::to_string(info.file) + " " + std::filesystem::path(info.path ? info.path : "").filename().string() +
        " at " + std::to_string(info.offset) + "), and the stock table has no buffer");

    row_server server(model.get());

    for (size_t s = 0; s < ref.seqs.size(); ++s) {
        const sequence & seq = ref.seqs[s];
        llama_context_ptr lctx = new_context(model.get(), &server, seq.shape);
        server.calls.clear();
        check(same_bits(get_logits(lctx.get(), seq.tokens, seq.last_only), ref.logits[s]),
            label + ": " + seq.what + ": logits match the stock model bitwise");
        if (!seq.last_only) {
            check_rows(model.get(), server, seq, label + ": " + seq.what, check);
        }
    }
    check(server.n_evicts > 0 && (!mc.rows_span_pages || server.n_split > 0), label + ": the cache evicts (" +
        std::to_string(server.n_evicts) + " of " + std::to_string(server.n_reads) + " page reads), rows span two pages (" +
        std::to_string(server.n_split) + ")");

    {
        llama_context_ptr lctx = new_context(model.get(), &server, TWO_SEQS);
        server.calls.clear();
        const std::vector<llama_token> & a = ref.seqs[0].tokens;
        const bool same = same_bits(get_logits_2(lctx.get(), a, ref.second), ref.logits_2);
        std::vector<int32_t> asked = server.all_rows();
        std::vector<int32_t> want  = query_rows(model.get(), a.data(), N_TOKENS, 0);
        const std::vector<int32_t> want_b = query_rows(model.get(), ref.second.data(), N_TOKENS, 0);
        want.insert(want.end(), want_b.begin(), want_b.end());
        std::sort(asked.begin(), asked.end());
        std::sort(want.begin(), want.end());
        check(same && server.rows_in_table() && asked == want,
            label + ": two sequences: logits match the stock model bitwise, and the row query names their rows");
    }

    llama_context_ptr lctx = new_context(model.get(), &server, ONE_UBATCH);

    server.calls.clear();
    const std::vector<float> embd = get_embd_logits(lctx.get(), 3);
    const size_t want_embd_rows = model->hparams.ple_n_heads > 0 ? 3*info.rows_per_token : 1;
    check(!embd.empty() && (!mc.embd_logits_stable || same_bits(embd, ref.embd)) && server.rows_in_table() &&
        server.all_rows().size() == want_embd_rows &&
        (model->hparams.ple_n_heads > 0 || server.all_rows() == std::vector<int32_t>{ 0 }),
        label + ": an embedding batch asks for " + std::to_string(want_embd_rows) + " rows" +
        (mc.embd_logits_stable ? " and matches the stock model bitwise" : ""));

    // the logits depend on the served bytes
    server.corrupt = true;
    const std::vector<float> corrupted = get_logits(lctx.get(), ref.seqs[0].tokens);
    check(!corrupted.empty() && !same_bits(corrupted, ref.logits[0]), label + ": corrupted row bytes change the logits");
    server.corrupt = false;

    // a failing caller fails the decode
    server.fail = true;
    llama_memory_clear(llama_get_memory(lctx.get()), true);
    common_batch batch(lctx.get());
    batch.add(ref.seqs[0].tokens[0], 0, 0, true);
    g_log.clear();
    const int32_t rc = llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
    check(rc != 0 && g_log.find("failed to read") != std::string::npos, label + ": a failing row reader fails the decode");
    server.fail = false;

    // a context without a row reader cannot compute
    g_log.clear();
    llama_context_ptr none(llama_init_from_model(model.get(), context_params(nullptr, ONE_UBATCH)));
    check(!none && g_log.find("no row reader") != std::string::npos, label + ": a context without a row reader fails to initialize");
}

static void test_model(const std::string & dir, const model_case & mc, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string path = dir + "/" + mc.file;
    const std::string stem = std::filesystem::path(path).stem().string() + (devs.empty() ? " (CPU)" : " (GPU)");
    printf("%s\n", stem.c_str());

    reference ref;
    {
        llama_model_ptr model = load_model(path, devs, nullptr);
        ref.seqs   = make_sequences(model.get());
        ref.second = ref.seqs[3].tokens;
        bool ok = true;
        for (const sequence & seq : ref.seqs) {
            llama_context_ptr lctx = new_context(model.get(), nullptr, seq.shape);
            ref.logits.push_back(get_logits(lctx.get(), seq.tokens, seq.last_only));
            ok = ok && !ref.logits.back().empty();
        }
        {
            llama_context_ptr lctx = new_context(model.get(), nullptr, TWO_SEQS);
            ref.logits_2 = get_logits_2(lctx.get(), ref.seqs[0].tokens, ref.second);
        }
        {
            llama_context_ptr lctx = new_context(model.get(), nullptr, ONE_UBATCH);
            ref.embd = get_embd_logits(lctx.get(), 3);
        }
        check(ok && !ref.logits_2.empty() && !ref.embd.empty(), stem + ": the stock model decodes");

        int32_t row = 0;
        check(llama_model_n_lookup_tables(model.get()) == 0 &&
            !llama_model_lookup_table_rows(model.get(), 0, ref.seqs[0].tokens.data(), 1, 0, &row),
            stem + ": without the callback, no table is served and the row query fails");

        // a prompt with one output prunes rows before the layers that share KV
        if (mc.shares_kv) {
            const llama_hparams & hp = model->hparams;
            row_watch watch;
            watch.name = "attn_norm-" + std::to_string(hp.n_layer_kv_from_start);
            {
                llama_context_params cparams = context_params(nullptr, ONE_UBATCH);
                cparams.cb_eval           = watch_rows;
                cparams.cb_eval_user_data = &watch;
                llama_context_ptr lctx_watch(llama_init_from_model(model.get(), cparams));
                get_logits(lctx_watch.get(), ref.seqs[0].tokens, true);
            }
            check(hp.n_layer_kv_from_start > 0 && hp.n_layer_kv_from_start < (int32_t) hp.n_layer() && watch.max_rows == 1,
                stem + ": " + std::to_string(hp.n_layer() - hp.n_layer_kv_from_start) + " of " + std::to_string(hp.n_layer()) +
                " layers share KV, and with one output the first of them computes one row");
        }
    }

    test_served(path, stem, mc, ref, devs, LLAMA_LOAD_MODE_AUTO, 0, check);
    test_served(path, stem + ", locked mapping", mc, ref, devs, LLAMA_LOAD_MODE_MMAP_MLOCK, 0, check);
    if (mc.split) {
        test_served(write_split(path, dir), stem + ", split in three files", mc, ref, devs, LLAMA_LOAD_MODE_AUTO, 2, check);
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
    // the log of a load is printed when it fails
    llama_log_set([](ggml_log_level, const char * text, void *) {
        g_log += text;
    }, nullptr);

    // the CPU (no devices), or with --all-devices every GPU device for the rest of the graph
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

    checker check;
    for (const model_case & mc : MODELS) {
        const std::string path = dir_models + "/" + mc.file;
        try {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("missing " + path);
            }
            test_model(dir_models, mc, devs, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", mc.file, e.what());
            check(false, std::string(mc.file) + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
