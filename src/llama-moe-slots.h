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

// the expert weights of MoE layers read from caller-filled slots, see llama_model_params.moe_slots
// llama allocates the regions and slot maps; the caller fills the slots and writes the maps
struct llama_moe_slot_region {
    struct member {
        std::string   name;
        ggml_tensor * stock  = nullptr; // metadata of the stock expert weight, not allocated
        ggml_tensor * tensor = nullptr; // over the region at `offset`, nb[2] = slot_size
        size_t        offset = 0;
        size_t        size   = 0;
    };

    struct run {
        uint32_t file;
        uint64_t offset;
        uint64_t size;
        uint64_t slot_offset;
    };

    int32_t il         = -1;
    int32_t n_expert   = 0;
    int32_t n_assigned = 0;
    int32_t n_spare    = 0;
    size_t  slot_size  = 0;

    std::vector<member> members; // in slot order

    ggml_backend_buffer_t buf      = nullptr; // the region, owned by llama_moe_slots
    ggml_tensor *         region   = nullptr;
    ggml_tensor *         slot_map = nullptr;

    std::vector<std::vector<run>> runs; // [n_expert]

    int64_t n_slots() const {
        return (int64_t) n_assigned + n_spare;
    }

    // the member tensor that stands in for the stock expert weight w, nullptr if w is not a member
    ggml_tensor * get(const ggml_tensor * w) const;
};

class llama_moe_slots {
public:
    bool empty() const {
        return regions.empty();
    }

    // the region of layer il, nullptr if the layer loads its expert weights
    const llama_moe_slot_region * layer(int32_t il) const;

    // the region that holds the stock expert weight w, nullptr if w is loaded
    const llama_moe_slot_region * of(const ggml_tensor * w) const;

    // every stock expert weight read from slots, with the tensor that stands in for it
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> substitutes() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // allocation, at load: add the regions, then allocate them
    llama_moe_slot_region & add(int32_t il);
    void own_meta(ggml_context_ptr ctx);
    void allocate(const std::map<int32_t, ggml_backend_buffer_type_t> & bufts, bool no_alloc);

private:
    std::map<int32_t, std::unique_ptr<llama_moe_slot_region>> regions;
    std::unordered_map<const ggml_tensor *, const llama_moe_slot_region *> by_stock;

    std::vector<ggml_context_ptr> ctxs; // stock metadata, member tensors, slot maps
    struct buffer {
        ggml_backend_buffer_ptr    buf;
        ggml_backend_buffer_type_t buft;
        size_t                     size; // without allocation, the size it would have
    };
    std::vector<buffer> bufs;
};
