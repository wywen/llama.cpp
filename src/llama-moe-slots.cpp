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

llama_moe_slot_pool::llama_moe_slot_pool(const llama_moe_slots & slots, llama_moe_slot_pool_callback get_params, void * user_data, bool no_alloc) :
    slots(slots) {
    for (const auto & [il, layer] : slots.all()) {
        const llama_moe_slot_pool_params pp = get_params ? get_params(il, layer->n_expert, user_data) : llama_moe_slot_pool_params{ 0, nullptr };
        const int32_t n = pp.n_slots;
        if (n < 0) {
            throw std::runtime_error(format("layer %d: invalid number of MoE slots %d", il, n));
        }
        // the context never computes this layer
        if (n == 0) {
            continue;
        }

        const size_t map_size = layer->map_size();
        const size_t size     = map_size + (size_t) n*layer->slot_size;

        ggml_backend_buffer_type_t buft = layer->buft;
        if (pp.buft != nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(layer->buft);
            if (ggml_backend_buft_get_device(pp.buft) != dev) {
                throw std::runtime_error(format("layer %d: the MoE slot buffer type %s is not on the device %s of the layer",
                    il, ggml_backend_buft_name(pp.buft), dev ? ggml_backend_dev_name(dev) : "(none)"));
            }
            if (ggml_backend_buft_get_alignment(pp.buft) > layer->alignment) {
                throw std::runtime_error(format("layer %d: the alignment %zu of the MoE slot buffer type %s is greater than the MoE slot alignment %zu",
                    il, ggml_backend_buft_get_alignment(pp.buft), ggml_backend_buft_name(pp.buft), layer->alignment));
            }
            if (ggml_backend_buft_get_max_size(pp.buft) < size) {
                throw std::runtime_error(format("layer %d: the max size %zu of the MoE slot buffer type %s is less than the %zu bytes of the slots",
                    il, ggml_backend_buft_get_max_size(pp.buft), ggml_backend_buft_name(pp.buft), size));
            }
            buft = pp.buft;
        }

        llama_moe_slot_region & region = regions[il];
        region.layer    = layer.get();
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
            throw std::runtime_error(format("unable to allocate %s buffer of %zu bytes for the MoE slots of layer %d",
                ggml_backend_buft_name(buft), size, il));
        }
        // the scheduler runs the ops that read weights on the backend of their buffer
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.push_back({ ggml_backend_buffer_ptr(buf), ggml_backend_buffer_get_type(buf), size });
        region.buf = buf;

        region.region = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, size);
        ggml_format_name(region.region, "blk.%d.ffn_exps_slots", il);
        place(buf, region.region, 0, no_alloc);

        region.slot_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, layer->n_expert);
        ggml_format_name(region.slot_map, "blk.%d.ffn_exps_slot_map", il);
        place(buf, region.slot_map, 0, no_alloc);

        for (const auto & m : layer->members) {
            const ggml_tensor * s = m.stock;
            ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), s->type, s->ne[0], s->ne[1], n);
            t->nb[2] = layer->slot_size;
            t->nb[3] = layer->slot_size*n;
            std::string name = m.name;
            const size_t pos = name.rfind(".weight");
            GGML_ASSERT(pos != std::string::npos);
            name.replace(pos, std::string::npos, ".slots");
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

        LLAMA_LOG_INFO("%s: layer %3d: %d slots of %zu bytes in %s\n", __func__, il, n, layer->slot_size, ggml_backend_buft_name(buft));
    }
}

const llama_moe_slot_region * llama_moe_slot_pool::region(int32_t il) const {
    const auto it = regions.find(il);
    return it == regions.end() ? nullptr : &it->second;
}

std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> llama_moe_slot_pool::substitutes() const {
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> res;
    for (const auto & [_, region] : regions) {
        for (size_t i = 0; i < region.members.size(); ++i) {
            res.emplace_back(region.layer->members[i].stock, region.members[i]);
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
