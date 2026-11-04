#include "llama-moe-slots.h"

#include "llama-impl.h"

#include <stdexcept>

ggml_tensor * llama_moe_slot_region::get(const ggml_tensor * w) const {
    for (const member & m : members) {
        if (m.stock == w) {
            return m.tensor;
        }
    }
    return nullptr;
}

const llama_moe_slot_region * llama_moe_slots::layer(int32_t il) const {
    const auto it = regions.find(il);
    return it == regions.end() ? nullptr : it->second.get();
}

const llama_moe_slot_region * llama_moe_slots::of(const ggml_tensor * w) const {
    const auto it = by_stock.find(w);
    return it == by_stock.end() ? nullptr : it->second;
}

std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> llama_moe_slots::substitutes() const {
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> res;
    for (const auto & [_, region] : regions) {
        for (const auto & m : region->members) {
            res.emplace_back(m.stock, m.tensor);
        }
    }
    return res;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_slots::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const buffer & b : bufs) {
        res[b.buft] += b.size;
    }
    return res;
}

llama_moe_slot_region & llama_moe_slots::add(int32_t il) {
    auto & region = regions[il];
    GGML_ASSERT(!region);
    region = std::make_unique<llama_moe_slot_region>();
    region->il = il;
    return *region;
}

void llama_moe_slots::own_meta(ggml_context_ptr ctx) {
    ctxs.push_back(std::move(ctx));
}

// a buffer of `size` bytes, or without allocation a placeholder that holds nothing
static ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool no_alloc) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, no_alloc ? 0 : size);
    if (buf == nullptr) {
        throw std::runtime_error(format("unable to allocate %s buffer of %zu bytes for MoE slots", ggml_backend_buft_name(buft), size));
    }
    // the scheduler runs the ops that read weights on the backend of their buffer
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buf;
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

void llama_moe_slots::allocate(const std::map<int32_t, ggml_backend_buffer_type_t> & bufts, bool no_alloc) {
    // the slot maps of the layers in one buffer type share a buffer
    std::map<ggml_backend_buffer_type_t, std::vector<llama_moe_slot_region *>> by_buft;

    for (auto & [il, region] : regions) {
        ggml_backend_buffer_type_t buft = bufts.at(il);
        by_buft[buft].push_back(region.get());

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(region->members.size() + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context");
        }

        const size_t size = region->n_slots()*region->slot_size;
        ggml_backend_buffer_t buf = alloc_buffer(buft, size, no_alloc);
        bufs.push_back({ ggml_backend_buffer_ptr(buf), buft, size });
        region->buf = buf;

        region->region = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, size);
        ggml_format_name(region->region, "blk.%d.ffn_exps_slots", il);
        place(buf, region->region, 0, no_alloc);

        for (auto & m : region->members) {
            const ggml_tensor * s = m.stock;
            ggml_tensor * t = ggml_new_tensor_3d(ctx.get(), s->type, s->ne[0], s->ne[1], region->n_slots());
            t->nb[2] = region->slot_size;
            t->nb[3] = region->slot_size*region->n_slots();
            std::string name = m.name;
            const size_t pos = name.rfind(".weight");
            GGML_ASSERT(pos != std::string::npos);
            name.replace(pos, std::string::npos, ".slots");
            ggml_set_name(t, name.c_str());
            place(buf, t, m.offset, no_alloc);
            m.tensor = t;

            by_stock.emplace(s, region.get());
        }

        ctxs.push_back(std::move(ctx));
    }

    for (auto & [buft, layer_regions] : by_buft) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*layer_regions.size(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context");
        }

        for (llama_moe_slot_region * region : layer_regions) {
            region->slot_map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, region->n_expert);
            ggml_format_name(region->slot_map, "blk.%d.ffn_exps_slot_map", region->il);
        }

        if (no_alloc) {
            ggml_backend_buffer_t buf = alloc_buffer(buft, 0, true);
            bufs.push_back({ ggml_backend_buffer_ptr(buf), buft, ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft) });
            for (llama_moe_slot_region * region : layer_regions) {
                region->slot_map->buffer = buf;
            }
        } else {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer for MoE slot maps", ggml_backend_buft_name(buft)));
            }
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(buf, 0);
            bufs.push_back({ ggml_backend_buffer_ptr(buf), buft, ggml_backend_buffer_get_size(buf) });
        }

        ctxs.push_back(std::move(ctx));
    }

    for (const auto & [il, region] : regions) {
        LLAMA_LOG_INFO("%s: layer %3d: %d + %d slots of %zu bytes in %s\n", __func__, il,
            region->n_assigned, region->n_spare, region->slot_size, ggml_backend_buft_name(bufts.at(il)));
    }
}
