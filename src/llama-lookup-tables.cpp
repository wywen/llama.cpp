#include "llama-lookup-tables.h"

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
