#pragma once

#include "llama.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// a MoE layer whose expert weights are read from caller-filled slots, see llama_model_params.moe_slots
// the model holds the stride of the slots and the stock expert weights; the caller lays out, allocates and fills the slots
struct llama_moe_slot_layer {
    struct member {
        std::string   name;
        ggml_tensor * stock = nullptr; // metadata of the stock expert weight, not allocated
    };

    int32_t il       = -1;
    int32_t n_expert = 0;
    size_t  stride   = 0; // between consecutive slots, a multiple of the alignment of buft

    std::vector<member> members;

    ggml_backend_buffer_type_t buft = nullptr; // of the slots, as selected for the member tensors
};

// the slot layers of a model, see llama_model_params.moe_slots
class llama_moe_slots {
public:
    bool empty() const {
        return layers.empty();
    }

    // the slot layer il, nullptr if the layer loads its expert weights
    const llama_moe_slot_layer * layer(int32_t il) const;

    // the slot layer of the stock expert weight w, nullptr if w is loaded
    const llama_moe_slot_layer * of(const ggml_tensor * w) const;

    const std::map<int32_t, std::unique_ptr<llama_moe_slot_layer>> & all() const {
        return layers;
    }

    // at load: add the layers, own the metadata of their stock expert weights, then index them
    llama_moe_slot_layer & add(int32_t il);
    void own_meta(ggml_context_ptr ctx);
    void index();

private:
    std::map<int32_t, std::unique_ptr<llama_moe_slot_layer>> layers;
    std::unordered_map<const ggml_tensor *, const llama_moe_slot_layer *> by_stock;

    std::vector<ggml_context_ptr> ctxs; // stock metadata
};
