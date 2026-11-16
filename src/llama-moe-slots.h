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

// the padding unit of the members and slots when llama_model_params.moe_slot_alignment is 0
#define LLAMA_MOE_SLOT_ALIGNMENT_DEFAULT 16384

// a MoE layer whose expert weights are read from caller-filled slots, see llama_model_params.moe_slots
// the model holds the geometry of a slot and the file runs that fill it; the caller allocates the slots
struct llama_moe_slot_layer {
    struct member {
        std::string   name;
        ggml_tensor * stock  = nullptr; // metadata of the stock expert weight, not allocated
        size_t        offset = 0;       // in a slot
        size_t        size   = 0;       // of one expert
    };

    struct run {
        uint32_t file;
        uint64_t offset;
        uint64_t size;
        uint64_t slot_offset;
    };

    int32_t il        = -1;
    int32_t n_expert  = 0;
    size_t  alignment = 0;
    size_t  slot_size = 0; // a multiple of alignment

    std::vector<member> members; // in slot order

    ggml_backend_buffer_type_t buft = nullptr; // of the slots, as selected for the member tensors

    std::vector<std::vector<run>> runs; // [n_expert]

    // the slot map, zero padded to the alignment: the offset of the first slot
    size_t map_size() const {
        return GGML_PAD((size_t) n_expert*sizeof(int32_t), alignment);
    }
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
