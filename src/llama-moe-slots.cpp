#include "llama-moe-slots.h"

const llama_moe_slot_layer * llama_moe_slots::layer(int32_t il) const {
    const auto it = layers.find(il);
    return it == layers.end() ? nullptr : it->second.get();
}

const llama_moe_slot_layer * llama_moe_slots::of(const ggml_tensor * w) const {
    const auto it = by_stock.find(w);
    return it == by_stock.end() ? nullptr : it->second;
}

llama_moe_slot_layer & llama_moe_slots::add(int32_t il) {
    auto & layer = layers[il];
    GGML_ASSERT(!layer);
    layer = std::make_unique<llama_moe_slot_layer>();
    layer->il = il;
    return *layer;
}

void llama_moe_slots::own_meta(ggml_context_ptr ctx) {
    ctxs.push_back(std::move(ctx));
}

void llama_moe_slots::index() {
    by_stock.clear();
    for (const auto & [_, layer] : layers) {
        for (const auto & m : layer->members) {
            by_stock.emplace(m.stock, layer.get());
        }
    }
}
