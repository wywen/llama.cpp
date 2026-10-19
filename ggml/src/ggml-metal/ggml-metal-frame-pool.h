#pragma once

#include "ggml-metal.h"
#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

// a region: one placement sparse buffer and the queued state of its tiles
struct ggml_metal_frame_region;

ggml_metal_frame_pool_t ggml_metal_frame_pool_init(ggml_metal_device_t dev, size_t n_slots, size_t stage_bytes);
void                    ggml_metal_frame_pool_free(ggml_metal_frame_pool_t pool);

size_t ggml_metal_frame_pool_get_tile_bytes(ggml_metal_frame_pool_t pool);
void * ggml_metal_frame_pool_get_stage     (ggml_metal_frame_pool_t pool);

// a device buffer at va_base over a new region of `size` bytes rounded up to tiles, or NULL
ggml_metal_buffer_t ggml_metal_frame_region_init(ggml_metal_device_t dev, void * va_base, size_t size);

// called by the device buffer that owns the region when it is freed
void ggml_metal_frame_region_free(struct ggml_metal_frame_region * region);

uint64_t ggml_metal_frame_pool_map  (ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, const struct ggml_metal_tile_run * runs, size_t n_runs);
uint64_t ggml_metal_frame_pool_unmap(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, const struct ggml_metal_tile_run * runs, size_t n_runs);

uint64_t ggml_metal_frame_pool_fill (ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map);
uint64_t ggml_metal_frame_pool_clear(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t size, uint8_t value, uint64_t after_map);
uint64_t ggml_metal_frame_pool_spill(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map,
        ggml_metal_event_t wait_ev, uint64_t wait_value);

uint64_t ggml_metal_frame_pool_completed(ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q);
bool     ggml_metal_frame_pool_wait     (ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q, uint64_t value, uint64_t timeout_ms);

void ggml_metal_frame_pool_keep_prepared(ggml_metal_frame_pool_t pool);

#ifdef __cplusplus
}
#endif
