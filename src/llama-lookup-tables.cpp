#include "llama-lookup-tables.h"

#include "llama-impl.h"

#include <cinttypes>
#include <stdexcept>

const llama_lookup_table * llama_lookup_tables::get(int32_t i) const {
    return i < 0 || i >= size() ? nullptr : tables[i].get();
}

const llama_lookup_table * llama_lookup_tables::of(const ggml_tensor * t) const {
    const auto it = by_stock.find(t);
    return it == by_stock.end() ? nullptr : it->second;
}

llama_lookup_table & llama_lookup_tables::add() {
    tables.push_back(std::make_unique<llama_lookup_table>());
    llama_lookup_table & table = *tables.back();
    table.index = (int32_t) tables.size() - 1;
    return table;
}

void llama_lookup_tables::own_meta(ggml_context_ptr ctx) {
    ctxs.push_back(std::move(ctx));
}

void llama_lookup_tables::index() {
    by_stock.clear();
    for (const auto & table : tables) {
        by_stock.emplace(table->stock, table.get());
    }
}

void llama_lookup_row_reader::read_rows(const llama_lookup_table & table, const int32_t * ids, int64_t n_ids, void * dst) const {
    for (int64_t i = 0; i < n_ids; ++i) {
        if (ids[i] < 0 || ids[i] >= table.n_rows) {
            throw std::runtime_error(format("lookup table %s: row %d is out of range [0, %" PRId64 ")",
                table.name.c_str(), ids[i], table.n_rows));
        }
    }
    if (!read(table.index, ids, (int32_t) n_ids, dst, table.row_size, user_data)) {
        throw std::runtime_error(format("lookup table %s: the caller failed to read %" PRId64 " rows", table.name.c_str(), n_ids));
    }
}

void llama_verify_lookup_tables_graph(ggml_cgraph * gf, const llama_lookup_tables & tables) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] != nullptr && tables.of(node->src[s]) != nullptr) {
                throw std::runtime_error(format("lookup table '%s' is served by the caller, but op '%s' (%s) reads the table itself",
                    node->src[s]->name, node->name, ggml_op_name(node->op)));
            }
        }
    }
}
