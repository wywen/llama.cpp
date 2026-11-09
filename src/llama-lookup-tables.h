#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// a lookup table whose rows the caller serves, see llama_model_params.lookup_tables
// the model holds its metadata and where the file stores it; each context asks the caller for the rows it reads
struct llama_lookup_table {
    int32_t     index = -1;
    std::string name;

    ggml_tensor * stock = nullptr; // metadata of the stock table, not allocated

    int64_t n_rows   = 0;
    size_t  row_size = 0; // of one row in the file

    int32_t rows_per_token = 0; // the rows that one token reads

    // the range of the file that holds the table, rows in order
    bool     has_file = false;
    uint32_t file     = 0;
    uint64_t offset   = 0;
};

// the lookup tables of a model whose rows the caller serves, see llama_model_params.lookup_tables
class llama_lookup_tables {
public:
    bool empty() const {
        return tables.empty();
    }

    int32_t size() const {
        return (int32_t) tables.size();
    }

    // table i, nullptr if out of range
    const llama_lookup_table * get(int32_t i) const;

    // the served table of the stock tensor t, nullptr if t is loaded
    const llama_lookup_table * of(const ggml_tensor * t) const;

    // at load: add the tables in index order, own the metadata of their stock tensors, then index them
    llama_lookup_table & add();
    void own_meta(ggml_context_ptr ctx);
    void index();

private:
    std::vector<std::unique_ptr<llama_lookup_table>> tables;
    std::unordered_map<const ggml_tensor *, const llama_lookup_table *> by_stock;

    std::vector<ggml_context_ptr> ctxs; // stock metadata
};

// the caller's row reader of a context, see llama_context_params.lookup_rows
struct llama_lookup_row_reader {
    llama_lookup_rows_callback read      = nullptr;
    void *                     user_data = nullptr;

    // the bytes of the rows `ids` of `table`, row i at dst + i*row_size; throws if a row is out of range or the caller
    // fails
    void read_rows(const llama_lookup_table & table, const int32_t * ids, int64_t n_ids, void * dst) const;
};

// check that no op of gf reads a served table: it has no data, only rows that the caller provides can be read
void llama_verify_lookup_tables_graph(ggml_cgraph * gf, const llama_lookup_tables & tables);
