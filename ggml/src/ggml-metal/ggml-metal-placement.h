#pragma once

#include "ggml-metal.h"
#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_metal_placement_heap_t ggml_metal_placement_heap_init(ggml_metal_device_t dev, size_t n_tiles);
void                        ggml_metal_placement_heap_free(ggml_metal_placement_heap_t heap);

bool   ggml_metal_placement_is_supported(ggml_metal_device_t dev);
size_t ggml_metal_placement_tile_bytes(void);

// a device buffer at va_base over a new placement sparse buffer of `size` bytes rounded up to tiles, or NULL
ggml_metal_buffer_t ggml_metal_sparse_buffer_init(ggml_metal_device_t dev, void * va_base, size_t size);

#ifdef __cplusplus
}
#endif
