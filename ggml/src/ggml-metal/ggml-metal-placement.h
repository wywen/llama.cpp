#pragma once

#include "ggml-metal.h"
#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_metal_placement_heap_t ggml_metal_placement_heap_init(ggml_metal_device_t dev, size_t n_tiles);
void                        ggml_metal_placement_heap_free(ggml_metal_placement_heap_t heap);
void *                      ggml_metal_placement_heap_contents(ggml_metal_placement_heap_t heap);

bool   ggml_metal_placement_is_supported(ggml_metal_device_t dev);
size_t ggml_metal_placement_tile_bytes(void);

// a device buffer at va_base over a new placement sparse buffer of `size` bytes rounded up to tiles, or NULL
ggml_metal_buffer_t ggml_metal_sparse_buffer_init(ggml_metal_device_t dev, void * va_base, size_t size);

ggml_metal_mapping_queue_t ggml_metal_mapping_queue_init(ggml_metal_device_t dev);
void                       ggml_metal_mapping_queue_free(ggml_metal_mapping_queue_t queue);

// heap is NULL for an unmap
bool ggml_metal_update_tiles(ggml_metal_mapping_queue_t queue, ggml_metal_placement_heap_t heap, ggml_metal_buffer_t buf,
        const struct ggml_metal_tile_run * runs, size_t n_runs, ggml_metal_event_t signal, uint64_t value);

ggml_metal_blit_queue_t ggml_metal_blit_queue_init (ggml_metal_device_t dev, ggml_metal_placement_heap_t heap);
void                    ggml_metal_blit_queue_free (ggml_metal_blit_queue_t queue);
const char *            ggml_metal_blit_queue_error(ggml_metal_blit_queue_t queue);
ggml_metal_event_t      ggml_metal_blit_queue_event(ggml_metal_blit_queue_t queue);

// the blit's value on the queue's timeline, or 0 when refused
uint64_t ggml_metal_blit_copy(ggml_metal_blit_queue_t queue, ggml_metal_buffer_t src, size_t src_offs, ggml_metal_buffer_t dst, size_t dst_offs, size_t size,
        const struct ggml_metal_wait * waits, size_t n_waits);
uint64_t ggml_metal_blit_fill(ggml_metal_blit_queue_t queue, ggml_metal_buffer_t dst, size_t offs, size_t size, uint8_t byte,
        const struct ggml_metal_wait * waits, size_t n_waits);

bool ggml_metal_keep_prepared(ggml_metal_placement_heap_t const * heaps, size_t n_heaps, ggml_metal_buffer_t const * bufs, size_t n_bufs);

#ifdef __cplusplus
}
#endif
