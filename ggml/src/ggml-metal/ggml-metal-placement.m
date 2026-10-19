#import "ggml-metal-placement.h"

#import "ggml-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// placement sparse resources and the Metal 4 mapping queue need macOS 26.4
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
#define GGML_METAL_HAS_PLACEMENT_SPARSE 1
#endif

#define GGML_METAL_PLACEMENT_TILE_BYTES 16384

#if defined(GGML_METAL_HAS_PLACEMENT_SPARSE)

// every heap and sparse buffer is created behind an availability check, so nothing
// below runs on an older OS
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

static const MTLSparsePageSize GGML_METAL_PLACEMENT_PAGE = MTLSparsePageSize16;

static bool ggml_metal_placement_supported(id<MTLDevice> device, const char * caller) {
    if (@available(macOS 26.4, *)) {
        if (device != nil && device.supportsPlacementSparse) {
            return true;
        }
        GGML_LOG_ERROR("%s: the device does not support placement sparse buffers\n", caller);
        return false;
    }
    GGML_LOG_ERROR("%s: placement sparse buffers need macOS 26.4\n", caller);
    return false;
}

bool ggml_metal_placement_is_supported(ggml_metal_device_t dev) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);
    if (@available(macOS 26.4, *)) {
        return device != nil && device.supportsPlacementSparse &&
            [device sparseTileSizeInBytesForSparsePageSize:GGML_METAL_PLACEMENT_PAGE] == GGML_METAL_PLACEMENT_TILE_BYTES;
    }
    return false;
}

size_t ggml_metal_placement_tile_bytes(void) {
    return GGML_METAL_PLACEMENT_TILE_BYTES;
}

//
// heaps
//

struct ggml_metal_placement_heap {
    id<MTLHeap>         heap;
    id<MTLResidencySet> rset;
    id<MTLCommandQueue> compute_queue; // ggml's queue, not owned

    size_t n_tiles;
};

ggml_metal_placement_heap_t ggml_metal_placement_heap_init(ggml_metal_device_t dev, size_t n_tiles) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);

    if (!ggml_metal_placement_supported(device, __func__)) {
        return NULL;
    }
    if (n_tiles == 0 || n_tiles > UINT32_MAX) {
        GGML_LOG_ERROR("%s: invalid heap of %zu tiles\n", __func__, n_tiles);
        return NULL;
    }
    if ([device sparseTileSizeInBytesForSparsePageSize:GGML_METAL_PLACEMENT_PAGE] != GGML_METAL_PLACEMENT_TILE_BYTES) {
        GGML_LOG_ERROR("%s: the device's sparse tile is not %d bytes\n", __func__, GGML_METAL_PLACEMENT_TILE_BYTES);
        return NULL;
    }

    ggml_metal_placement_heap_t res = calloc(1, sizeof(struct ggml_metal_placement_heap));

    res->n_tiles       = n_tiles;
    res->compute_queue = (id<MTLCommandQueue>) ggml_metal_device_get_queue(dev);

    @autoreleasepool {
        MTLHeapDescriptor * desc = [[MTLHeapDescriptor alloc] init];
        desc.type        = MTLHeapTypePlacement;
        desc.storageMode = MTLStorageModePrivate;
        desc.size        = n_tiles*GGML_METAL_PLACEMENT_TILE_BYTES;
        desc.maxCompatiblePlacementSparsePageSize = GGML_METAL_PLACEMENT_PAGE;

        res->heap = [device newHeapWithDescriptor:desc];
        [desc release];

        MTLResidencySetDescriptor * rdesc = [[MTLResidencySetDescriptor alloc] init];
        rdesc.label = @"ggml_metal_placement_heap";

        NSError * error = nil;
        res->rset = [device newResidencySetWithDescriptor:rdesc error:&error];
        [rdesc release];
        if (error) {
            GGML_LOG_ERROR("%s: residency set: %s\n", __func__, [[error description] UTF8String]);
        }

    }

    if (res->heap == nil || res->rset == nil || res->compute_queue == nil) {
        GGML_LOG_ERROR("%s: failed to create a heap of %zu tiles (%.2f MiB)\n", __func__,
                n_tiles, n_tiles*GGML_METAL_PLACEMENT_TILE_BYTES/1024.0/1024.0);
        [res->rset       release];
        [res->heap       release];
        free(res);
        return NULL;
    }

    [res->rset addAllocation:res->heap];
    [res->rset commit];
    [res->rset requestResidency];

    [res->compute_queue addResidencySet:res->rset];

    return res;
}

void ggml_metal_placement_heap_free(ggml_metal_placement_heap_t heap) {
    if (heap == NULL) {
        return;
    }

    [heap->compute_queue removeResidencySet:heap->rset];
    [heap->rset endResidency];
    [heap->rset removeAllAllocations];
    [heap->rset commit];

    [heap->rset       release];
    [heap->heap       release];

    free(heap);
}

//
// sparse buffers
//

ggml_metal_buffer_t ggml_metal_sparse_buffer_init(ggml_metal_device_t dev, void * va_base, size_t size) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);

    if (!ggml_metal_placement_supported(device, __func__)) {
        return NULL;
    }

    const size_t tile_bytes = GGML_METAL_PLACEMENT_TILE_BYTES;

    if ([device sparseTileSizeInBytesForSparsePageSize:GGML_METAL_PLACEMENT_PAGE] != tile_bytes) {
        GGML_LOG_ERROR("%s: the device's sparse tile is not %zu bytes\n", __func__, tile_bytes);
        return NULL;
    }

    if (size == 0 || va_base == NULL || (uintptr_t) va_base % tile_bytes != 0) {
        GGML_LOG_ERROR("%s: a sparse buffer needs a non-empty size and a base aligned to %zu bytes (size = %zu, base = %p)\n",
                __func__, tile_bytes, size, va_base);
        return NULL;
    }

    const size_t n_tiles = (size + tile_bytes - 1)/tile_bytes;
    const size_t bytes   = n_tiles*tile_bytes;

    if (n_tiles > UINT32_MAX) {
        GGML_LOG_ERROR("%s: a sparse buffer of %zu tiles is too large\n", __func__, n_tiles);
        return NULL;
    }

    id<MTLBuffer> metal = [device newBufferWithLength:bytes
                                              options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked
                              placementSparsePageSize:GGML_METAL_PLACEMENT_PAGE];
    if (metal == nil) {
        GGML_LOG_ERROR("%s: failed to create a placement sparse buffer of %zu tiles (%.2f MiB)\n", __func__,
                n_tiles, bytes/1024.0/1024.0);
        return NULL;
    }

    return ggml_metal_buffer_init_sparse(dev, metal, va_base, bytes);
}

#pragma clang diagnostic pop

#else

size_t ggml_metal_placement_tile_bytes(void) {
    return GGML_METAL_PLACEMENT_TILE_BYTES;
}

ggml_metal_placement_heap_t ggml_metal_placement_heap_init(ggml_metal_device_t dev, size_t n_tiles) {
    GGML_LOG_ERROR("%s: placement sparse buffers need the macOS 26.4 SDK\n", __func__);
    GGML_UNUSED(dev);
    GGML_UNUSED(n_tiles);
    return NULL;
}

ggml_metal_buffer_t ggml_metal_sparse_buffer_init(ggml_metal_device_t dev, void * va_base, size_t size) {
    GGML_LOG_ERROR("%s: placement sparse buffers need the macOS 26.4 SDK\n", __func__);
    GGML_UNUSED(dev);
    GGML_UNUSED(va_base);
    GGML_UNUSED(size);
    return NULL;
}

// without the SDK no heap exists, so nothing below is reachable

void ggml_metal_placement_heap_free(ggml_metal_placement_heap_t heap) { GGML_UNUSED(heap); }

#endif

