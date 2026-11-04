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
// the model holds the geometry of a slot and the file runs that fill it; each context allocates its own slots
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

// the slots of one slot layer in a context: one buffer that holds the slot map, zero padded to the alignment, then
// n_slots slots of slot_size bytes
struct llama_moe_slot_region {
    const llama_moe_slot_layer * layer = nullptr;

    int32_t n_slots  = 0;
    size_t  map_size = 0; // round_up(n_expert*4, alignment), the offset of the first slot

    ggml_backend_buffer_t buf      = nullptr; // owned by llama_moe_slot_pool
    ggml_tensor *         region   = nullptr; // I8 [map_size + n_slots*slot_size], the whole buffer
    ggml_tensor *         slot_map = nullptr; // I32 [1, n_expert] at offset 0

    std::vector<ggml_tensor *> members; // as layer->members: [ne0, ne1, n_slots] at map_size + offset, nb[2] = slot_size

    // the member tensor that stands in for the stock expert weight w, nullptr if w is not a member
    ggml_tensor * get(const ggml_tensor * w) const;
};

// the slots of a context, for each slot layer of its model, see llama_context_params.moe_slots
class llama_moe_slot_pool {
public:
    // asks `n_slots` for the number of slots of each slot layer; NULL = no slots
    llama_moe_slot_pool(const llama_moe_slots & slots, llama_moe_slot_count_callback n_slots, void * user_data, bool no_alloc);

    // the slots of layer il, nullptr if the layer is not a slot layer or this context has no slots for it
    const llama_moe_slot_region * region(int32_t il) const;

    const llama_moe_slots & model_slots() const {
        return slots;
    }

    // every stock expert weight read from the slots of this context, with the tensor that stands in for it
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> substitutes() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    const llama_moe_slots & slots;

    std::map<int32_t, llama_moe_slot_region> regions;

    std::vector<ggml_context_ptr> ctxs; // region, slot map and member tensors
    struct buffer {
        ggml_backend_buffer_ptr    buf;
        ggml_backend_buffer_type_t buft;
        size_t                     size; // without allocation, the size it would have
    };
    std::vector<buffer> bufs;
};
