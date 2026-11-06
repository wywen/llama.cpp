#include "llama-moe-slots.h"

#include "llama-impl.h"

#include <stdexcept>

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

ggml_tensor * llama_moe_slot_region::get(const ggml_tensor * w) const {
    for (size_t i = 0; i < members.size(); ++i) {
        if (layer->members[i].stock == w) {
            return members[i];
        }
    }
    return nullptr;
}

static void place(ggml_backend_buffer_t buf, ggml_tensor * t, size_t offset, bool no_alloc) {
    if (no_alloc) {
        t->buffer = buf;
        return;
    }
    if (ggml_backend_tensor_alloc(buf, t, (char *) ggml_backend_buffer_get_base(buf) + offset) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(format("failed to place tensor %s in its MoE slot buffer", t->name));
    }
}

llama_moe_slot_pool::llama_moe_slot_pool(const llama_moe_slots & slots, uint32_t n_sets, llama_moe_slot_pool_callback get_params, void * user_data, bool no_alloc) :
    slots(slots) {
    if (n_sets == 0) {
        throw std::runtime_error("the number of MoE slot sets must be at least 1");
    }
    regions.resize(n_sets);
    for (uint32_t set = 0; set < n_sets; ++set) {
        for (const auto & [_, layer] : slots.all()) {
            add_region((int32_t) set, layer.get(), get_params, user_data, no_alloc);
        }
    }
}

void llama_moe_slot_pool::add_region(int32_t set, const llama_moe_slot_layer * layer, llama_moe_slot_pool_callback get_params, void * user_data, bool no_alloc) {
    const int32_t il = layer->il;

    const llama_moe_slot_pool_params pp = get_params ? get_params(il, layer->n_expert, set, user_data) : llama_moe_slot_pool_params{ 0, nullptr };
    const int32_t n = pp.n_slots;
    if (n < 0) {
        throw std::runtime_error(format("layer %d, set %d: invalid number of MoE slots %d", il, set, n));
    }
    // the context never computes this layer while it reads this set
    if (n == 0) {
        return;
    }

    const size_t map_size = layer->map_size();
    const size_t size     = map_size + (size_t) n*layer->slot_size;

    ggml_backend_buffer_type_t buft = layer->buft;
    if (pp.buft != nullptr) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(layer->buft);
        if (ggml_backend_buft_get_device(pp.buft) != dev) {
            throw std::runtime_error(format("layer %d, set %d: the MoE slot buffer type %s is not on the device %s of the layer",
                il, set, ggml_backend_buft_name(pp.buft), dev ? ggml_backend_dev_name(dev) : "(none)"));
        }
        if (ggml_backend_buft_get_alignment(pp.buft) > layer->alignment) {
            throw std::runtime_error(format("layer %d, set %d: the alignment %zu of the MoE slot buffer type %s is greater than the MoE slot alignment %zu",
                il, set, ggml_backend_buft_get_alignment(pp.buft), ggml_backend_buft_name(pp.buft), layer->alignment));
        }
        if (ggml_backend_buft_get_max_size(pp.buft) < size) {
            throw std::runtime_error(format("layer %d, set %d: the max size %zu of the MoE slot buffer type %s is less than the %zu bytes of the slots",
                il, set, ggml_backend_buft_get_max_size(pp.buft), ggml_backend_buft_name(pp.buft), size));
        }
        buft = pp.buft;
    }

    llama_moe_slot_region & region = regions[set][il];
    region.layer    = layer;
    region.n_slots  = n;
    region.map_size = map_size;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*(layer->members.size() + 2),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to create ggml context");
    }

    // without allocation, a placeholder of the layer's buffer type, which the scheduler places as the real buffer
    ggml_backend_buffer_t buf = no_alloc ? ggml_backend_buft_alloc_buffer(layer->buft, 0) : ggml_backend_buft_alloc_buffer(buft, size);
    if (buf == nullptr) {
        throw std::runtime_error(format("unable to allocate %s buffer of %zu bytes for the MoE slots of layer %d, set %d",
            ggml_backend_buft_name(buft), size, il, set));
    }
    // the scheduler runs the ops that read weights on the backend of their buffer
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bufs.push_back({ ggml_backend_buffer_ptr(buf), ggml_backend_buffer_get_type(buf), size });
    region.buf = buf;

    // a switch of set must keep the graph's splits, so every set of a layer reads its slots on the same backend: the
    // choice depends on the buffer type's device and on whether it is host memory, not on the type object
    for (int32_t s = 0; s < set; ++s) {
        const auto it = regions[s].find(il);
        if (it == regions[s].end()) {
            continue;
        }
        ggml_backend_buffer_type_t other = ggml_backend_buffer_get_type(it->second.buf);
        ggml_backend_buffer_type_t mine  = ggml_backend_buffer_get_type(buf);
        if (ggml_backend_buft_get_device(other) != ggml_backend_buft_get_device(mine) ||
            ggml_backend_buft_is_host(other)    != ggml_backend_buft_is_host(mine)) {
            throw std::runtime_error(format("layer %d, set %d: the MoE slot buffer is of type %s, set %d's of type %s; every set of a layer must use buffer types of one device and of one host-ness",
                il, set, ggml_backend_buft_name(mine), s, ggml_backend_buft_name(other)));
        }
    }

    // the tensors of set 0 keep the names of a context with one set
    const std::string suffix = set == 0 ? "" : format(".set%d", set);

    region.region = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, size);
    ggml_format_name(region.region, "blk.%d.ffn_exps_slots%s", il, suffix.c_str());
    place(buf, region.region, 0, no_alloc);

    region.slot_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, layer->n_expert);
    ggml_format_name(region.slot_map, "blk.%d.ffn_exps_slot_map%s", il, suffix.c_str());
    place(buf, region.slot_map, 0, no_alloc);

    for (const auto & m : layer->members) {
        const ggml_tensor * s = m.stock;
        ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), s->type, s->ne[0], s->ne[1], n);
        t->nb[2] = layer->slot_size;
        t->nb[3] = layer->slot_size*n;
        std::string name = m.name;
        const size_t pos = name.rfind(".weight");
        GGML_ASSERT(pos != std::string::npos);
        name.replace(pos, std::string::npos, ".slots" + suffix);
        ggml_set_name(t, name.c_str());
        place(buf, t, region.map_size + m.offset, no_alloc);
        region.members.push_back(t);
    }

    // the slot map and its padding start zeroed, the slots uninitialized
    if (!no_alloc) {
        const std::vector<uint8_t> zeros(region.map_size, 0);
        ggml_backend_tensor_set(region.region, zeros.data(), 0, zeros.size());
    }

    ctxs.push_back(std::move(ctx));

    LLAMA_LOG_INFO("%s: layer %3d, set %d: %d slots of %zu bytes in %s\n", __func__, il, set, n, layer->slot_size, ggml_backend_buft_name(buft));
}

const llama_moe_slot_region * llama_moe_slot_pool::region(int32_t set, int32_t il) const {
    if (set < 0 || (size_t) set >= regions.size()) {
        return nullptr;
    }
    const auto it = regions[set].find(il);
    return it == regions[set].end() ? nullptr : &it->second;
}

const llama_moe_slot_region * llama_moe_slot_pool::widest(int32_t il) const {
    const llama_moe_slot_region * res = nullptr;
    for (const auto & set : regions) {
        const auto it = set.find(il);
        if (it != set.end() && (res == nullptr || it->second.n_slots > res->n_slots)) {
            res = &it->second;
        }
    }
    return res;
}

std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> llama_moe_slot_pool::substitutes() const {
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> res;
    for (const auto & set : regions) {
        for (const auto & [_, region] : set) {
            for (size_t i = 0; i < region.members.size(); ++i) {
                res.emplace_back(region.layer->members[i].stock, region.members[i]);
            }
        }
    }
    return res;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_slot_pool::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const buffer & b : bufs) {
        res[b.buft] += b.size;
    }
    return res;
}
