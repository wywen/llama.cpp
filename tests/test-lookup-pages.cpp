// checks that a model whose per-layer token embedding table is served by the caller (llama_model_params.lookup_tables,
//     llama_context_params.lookup_rows) computes bitwise identical logits to the same model loaded with the table, when
//     the test serves the rows itself by reading the model file through a small cache of 16 KiB-aligned file pages,
//     whose rows can span two pages:
//   - a multi-token prompt, then single-token decode steps
//   - a prompt where only the last token requests output, which prunes rows before the layers that share KV
//   - a batch of input embeddings, which reads the padding row
//   - a model loaded with a locked mapping
// and that only the pageable table is offered, that the served table has no buffer, that the caller is asked for the
//     rows of each ubatch in token order, that corrupted row bytes change the logits, that a failing caller fails the
//     decode, and that a context without a row reader is refused
//
// the models are written by test-llama-archs (-o <dir>)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"

// the test checks that the stock table has no buffer, and the layers that share KV
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
        printf("  %-100s %s\n", what.c_str(), ok ? "OK" : "FAIL");
        n_fail += ok ? 0 : 1;
    }
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

static llama_context_params context_params(row_server * server) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 0;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.n_ubatch        = 64;
    if (server) {
        cparams.lookup_rows           = serve_rows;
        cparams.lookup_rows_user_data = server;
    }
    return cparams;
}

static llama_context_ptr new_context(llama_model * model, row_server * server) {
    g_log.clear();
    llama_context_ptr lctx(llama_init_from_model(model, context_params(server)));
    if (!lctx) {
        throw std::runtime_error("failed to create a context:\n" + g_log);
    }
    return lctx;
}

static constexpr size_t N_TOKENS = 16; // a prompt of N_BATCH tokens, then single-token decode steps
static constexpr size_t N_BATCH  = 12;

// the logits of the first n_batch tokens in one batch, then of the other tokens one at a time, from an empty memory
// with last_only, only the last token of the prompt requests output; empty if a decode fails
static std::vector<float> get_logits(llama_context * lctx, const std::vector<llama_token> & tokens, size_t n_batch, bool last_only = false) {
    llama_memory_clear(llama_get_memory(lctx), true);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));

    std::vector<float> ret;
    common_batch batch(lctx);
    for (size_t i = 0; i < n_batch; i++) {
        batch.add(tokens[i], i, 0, !last_only || i == n_batch - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        return {};
    }
    for (size_t i = last_only ? n_batch - 1 : 0; i < n_batch; i++) {
        const float * l = llama_get_logits_ith(lctx, last_only ? -1 : (int32_t) i);
        ret.insert(ret.end(), l, l + n_vocab);
    }
    for (size_t i = n_batch; i < tokens.size(); i++) {
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

// the logits of a batch of input embeddings, which read the padding row of the per-layer table
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

static void test_model(const std::string & path, const std::vector<ggml_backend_dev_t> & devs, checker & check) {
    const std::string stem = std::filesystem::path(path).stem().string() + (devs.empty() ? " (CPU)" : " (GPU)");
    printf("%s\n", stem.c_str());

    std::vector<llama_token> tokens;
    std::vector<float> ref;
    std::vector<float> ref_last;
    std::vector<float> ref_embd;
    {
        llama_model_ptr model = load_model(path, devs, nullptr);
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        std::mt19937 gen(42);
        for (size_t i = 0; i < N_TOKENS; ++i) {
            tokens.push_back((llama_token) (gen() % n_vocab));
        }
        llama_context_ptr lctx = new_context(model.get(), nullptr);
        ref      = get_logits(lctx.get(), tokens, N_BATCH);
        ref_last = get_logits(lctx.get(), tokens, N_BATCH, true);
        ref_embd = get_embd_logits(lctx.get(), 3);
        check(!ref.empty() && !ref_last.empty() && !ref_embd.empty(), stem + ": the stock model decodes");
        check(llama_model_n_lookup_tables(model.get()) == 0, stem + ": without the callback, no table is served");

        // a prompt with one output prunes rows before the layers that share KV
        const llama_hparams & hp = model->hparams;
        row_watch watch;
        watch.name = "attn_norm-" + std::to_string(hp.n_layer_kv_from_start);
        {
            llama_context_params cparams = context_params(nullptr);
            cparams.cb_eval           = watch_rows;
            cparams.cb_eval_user_data = &watch;
            llama_context_ptr lctx_watch(llama_init_from_model(model.get(), cparams));
            get_logits(lctx_watch.get(), tokens, N_BATCH, true);
        }
        check(hp.n_layer_kv_from_start > 0 && hp.n_layer_kv_from_start < (int32_t) hp.n_layer() && watch.max_rows == 1,
            stem + ": " + std::to_string(hp.n_layer() - hp.n_layer_kv_from_start) + " of " + std::to_string(hp.n_layer()) +
            " layers share KV, and with one output the first of them computes one row");
    }

    for (const llama_load_mode mode : { LLAMA_LOAD_MODE_AUTO, LLAMA_LOAD_MODE_MMAP_MLOCK }) {
        const std::string label = stem + (mode == LLAMA_LOAD_MODE_AUTO ? "" : ", locked mapping");

        std::vector<std::string> offered;
        llama_model_ptr model = load_model(path, devs, &offered, mode);
        check(offered == std::vector<std::string>{ TABLE } && llama_model_n_lookup_tables(model.get()) == 1,
            label + ": the loader offers only the per-layer token table, and serves it");

        llama_lookup_table_info info = {};
        const bool has_info = llama_model_lookup_table_info(model.get(), 0, &info);
        const ggml_tensor * stock = model->get_tensor(TABLE);
        check(has_info && strcmp(info.name, TABLE) == 0 && stock != nullptr && stock->buffer == nullptr &&
            info.type == stock->type && info.n_cols == stock->ne[0] && info.n_rows == stock->ne[1] &&
            info.row_size == ggml_row_size(stock->type, stock->ne[0]) && info.path != nullptr && info.file == 0 &&
            info.offset + info.n_rows*info.row_size <= std::filesystem::file_size(info.path),
            label + ": table info, and the stock table has no buffer");

        row_server server(model.get());
        llama_context_ptr lctx = new_context(model.get(), &server);

        check(same_bits(get_logits(lctx.get(), tokens, N_BATCH), ref), label + ": logits of a prompt and decode steps match the stock model bitwise");
        bool order_ok = server.calls.size() == 1 + N_TOKENS - N_BATCH &&
            server.calls[0] == std::vector<int32_t>(tokens.begin(), tokens.begin() + N_BATCH);
        for (size_t i = N_BATCH; order_ok && i < N_TOKENS; ++i) {
            order_ok = server.calls[1 + i - N_BATCH] == std::vector<int32_t>{ tokens[i] };
        }
        check(order_ok, label + ": each ubatch asks for its rows in token order");
        check(server.n_split > 0 && server.n_evicts > 0, label + ": rows span two file pages (" + std::to_string(server.n_split) +
            "), and the cache evicts (" + std::to_string(server.n_evicts) + " of " + std::to_string(server.n_reads) + " page reads)");

        check(same_bits(get_logits(lctx.get(), tokens, N_BATCH, true), ref_last),
            label + ": with only the prompt's last output (pruned rows), logits match the stock model bitwise");

        server.calls.clear();
        check(same_bits(get_embd_logits(lctx.get(), 3), ref_embd) && server.calls.size() == 1 && server.calls[0] == std::vector<int32_t>{ 0 },
            label + ": an embedding batch asks for row 0 and matches the stock model bitwise");

        // the logits depend on the served bytes
        server.corrupt = true;
        const std::vector<float> corrupted = get_logits(lctx.get(), tokens, N_BATCH);
        check(!corrupted.empty() && !same_bits(corrupted, ref), label + ": corrupted row bytes change the logits");
        server.corrupt = false;

        // a failing caller fails the decode
        server.fail = true;
        llama_memory_clear(llama_get_memory(lctx.get()), true);
        common_batch batch(lctx.get());
        batch.add(tokens[0], 0, 0, true);
        g_log.clear();
        const int32_t rc = llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
        check(rc != 0 && g_log.find("failed to read") != std::string::npos, label + ": a failing row reader fails the decode");
        server.fail = false;

        // a context without a row reader cannot compute
        g_log.clear();
        llama_context_ptr none(llama_init_from_model(model.get(), context_params(nullptr)));
        check(!none && g_log.find("no row reader") != std::string::npos, label + ": a context without a row reader fails to initialize");
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

    // a dense and a MoE model with per-layer token embeddings
    const char * models[] = { "gemma4-dense.gguf", "gemma4-moe.gguf" };

    checker check;
    for (const char * name : models) {
        const std::string path = dir_models + "/" + name;
        try {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("missing " + path);
            }
            test_model(path, devs, check);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s: %s\n", name, e.what());
            check(false, std::string(name) + ": runs");
        }
    }

    llama_backend_free();
    printf("%s\n", check.n_fail == 0 ? "OK" : "FAIL");
    return check.n_fail == 0 ? 0 : 1;
}
