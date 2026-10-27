#import "ggml-metal-placement.h"

#import "ggml-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// placement sparse resources and the Metal 4 mapping queue need macOS 26.4
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
#define GGML_METAL_HAS_PLACEMENT_SPARSE 1
#endif

#define GGML_METAL_PLACEMENT_TILE_BYTES 16384

#if defined(GGML_METAL_HAS_PLACEMENT_SPARSE)

// every heap, sparse buffer and mapping queue is created behind an availability check, so nothing
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
    id<MTLHeap>   heap;
    // a shared buffer placed over the whole heap: the host's view of every tile
    id<MTLBuffer> alias;

    size_t n_tiles;

    // keep-prepared: an empty kernel on a queue of its own, so that it never waits behind a blit or a
    // compute command buffer parked on an event
    id<MTLCommandQueue>         keep_queue;
    id<MTLComputePipelineState> nop;
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

    res->n_tiles = n_tiles;

    @autoreleasepool {
        const size_t bytes = n_tiles*GGML_METAL_PLACEMENT_TILE_BYTES;

        MTLHeapDescriptor * desc = [[MTLHeapDescriptor alloc] init];
        desc.type        = MTLHeapTypePlacement;
        desc.storageMode = MTLStorageModeShared;
        desc.size        = bytes;
        desc.maxCompatiblePlacementSparsePageSize = GGML_METAL_PLACEMENT_PAGE;

        res->heap = [device newHeapWithDescriptor:desc];
        [desc release];

        if (res->heap != nil) {
            res->alias = [res->heap newBufferWithLength:bytes options:MTLResourceStorageModeShared offset:0];
        }

        res->keep_queue = [device newCommandQueue];

        NSError * error = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@"kernel void ggml_metal_placement_nop() {}" options:nil error:&error];
        id<MTLFunction> fn = [lib newFunctionWithName:@"ggml_metal_placement_nop"];
        if (error != nil && lib != nil) {
            GGML_LOG_WARN("%s: keep-prepared kernel: %s\n", __func__, [[error description] UTF8String]);
            error = nil;
        }
        if (fn != nil) {
            res->nop = [device newComputePipelineStateWithFunction:fn error:&error];
        }
        if (error != nil) {
            GGML_LOG_ERROR("%s: keep-prepared kernel: %s\n", __func__, [[error description] UTF8String]);
        }
        [fn release];
        [lib release];
    }

    if (res->heap == nil || res->alias == nil || res->alias.contents == NULL || res->keep_queue == nil || res->nop == nil) {
        GGML_LOG_ERROR("%s: failed to create a shared heap of %zu tiles (%.2f MiB)\n", __func__,
                n_tiles, n_tiles*GGML_METAL_PLACEMENT_TILE_BYTES/1024.0/1024.0);
        [res->nop        release];
        [res->keep_queue release];
        [res->alias      release];
        [res->heap       release];
        free(res);
        return NULL;
    }

    return res;
}

void ggml_metal_placement_heap_free(ggml_metal_placement_heap_t heap) {
    if (heap == NULL) {
        return;
    }

    [heap->nop        release];
    [heap->keep_queue release];
    [heap->alias      release];
    [heap->heap       release];

    free(heap);
}

void * ggml_metal_placement_heap_contents(ggml_metal_placement_heap_t heap) {
    return heap != NULL ? heap->alias.contents : NULL;
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
                                              options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked
                              placementSparsePageSize:GGML_METAL_PLACEMENT_PAGE];
    if (metal == nil) {
        GGML_LOG_ERROR("%s: failed to create a placement sparse buffer of %zu tiles (%.2f MiB)\n", __func__,
                n_tiles, bytes/1024.0/1024.0);
        return NULL;
    }

    return ggml_metal_buffer_init_sparse(dev, metal, va_base, bytes);
}

bool ggml_metal_sparse_buffer_renew(ggml_metal_device_t dev, ggml_metal_buffer_t buf) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);

    id<MTLBuffer> old = (id<MTLBuffer>) ggml_metal_buffer_get_sparse(buf);
    if (old == nil) {
        GGML_LOG_ERROR("%s: the buffer is not a sparse buffer\n", __func__);
        return false;
    }

    id<MTLBuffer> metal = [device newBufferWithLength:old.length
                                              options:old.resourceOptions
                              placementSparsePageSize:GGML_METAL_PLACEMENT_PAGE];
    if (metal == nil) {
        GGML_LOG_ERROR("%s: failed to create a placement sparse buffer of %.2f MiB\n", __func__,
                old.length/1024.0/1024.0);
        return false;
    }

    return ggml_metal_buffer_set_sparse(buf, metal);
}

//
// mapping
//

struct ggml_metal_mapping_queue {
    id<MTL4CommandQueue> queue;
};

ggml_metal_mapping_queue_t ggml_metal_mapping_queue_init(ggml_metal_device_t dev) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);

    if (!ggml_metal_placement_supported(device, __func__)) {
        return NULL;
    }

    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    if (queue == nil) {
        GGML_LOG_ERROR("%s: failed to create a Metal 4 command queue\n", __func__);
        return NULL;
    }

    ggml_metal_mapping_queue_t res = calloc(1, sizeof(struct ggml_metal_mapping_queue));
    res->queue = queue;

    return res;
}

void ggml_metal_mapping_queue_free(ggml_metal_mapping_queue_t queue) {
    if (queue == NULL) {
        return;
    }

    [queue->queue release];
    free(queue);
}

static bool ggml_metal_tile_runs_valid(const struct ggml_metal_tile_run * runs, size_t n_runs, size_t buf_tiles, size_t heap_tiles) {
    if (n_runs == 0 || runs == NULL) {
        GGML_LOG_ERROR("%s: no runs\n", __func__);
        return false;
    }
    for (size_t r = 0; r < n_runs; ++r) {
        const struct ggml_metal_tile_run * run = &runs[r];
        if (run->n_tiles == 0 || (size_t) run->tile + run->n_tiles > buf_tiles) {
            GGML_LOG_ERROR("%s: tiles [%u, +%u) outside a buffer of %zu tiles\n", __func__, run->tile, run->n_tiles, buf_tiles);
            return false;
        }
        if (heap_tiles > 0 && (size_t) run->heap_tile + run->n_tiles > heap_tiles) {
            GGML_LOG_ERROR("%s: heap tiles [%u, +%u) outside a heap of %zu tiles\n", __func__, run->heap_tile, run->n_tiles, heap_tiles);
            return false;
        }
    }
    return true;
}

bool ggml_metal_update_tiles(ggml_metal_mapping_queue_t queue, ggml_metal_placement_heap_t heap, ggml_metal_buffer_t buf,
        const struct ggml_metal_tile_run * runs, size_t n_runs, ggml_metal_event_t signal, uint64_t value) {
    id<MTLBuffer> sparse = (id<MTLBuffer>) ggml_metal_buffer_get_sparse(buf);
    if (sparse == nil) {
        GGML_LOG_ERROR("%s: the buffer is not a sparse buffer\n", __func__);
        return false;
    }

    const bool   map        = heap != NULL;
    const size_t buf_tiles  = ggml_metal_buffer_get_size(buf)/GGML_METAL_PLACEMENT_TILE_BYTES;
    const size_t heap_tiles = map ? heap->n_tiles : 0;

    if (!ggml_metal_tile_runs_valid(runs, n_runs, buf_tiles, heap_tiles)) {
        return false;
    }

    @autoreleasepool {
        MTL4UpdateSparseBufferMappingOperation * ops = malloc(n_runs*sizeof(MTL4UpdateSparseBufferMappingOperation));
        for (size_t r = 0; r < n_runs; ++r) {
            ops[r] = (MTL4UpdateSparseBufferMappingOperation) {
                .mode        = map ? MTLSparseTextureMappingModeMap : MTLSparseTextureMappingModeUnmap,
                .bufferRange = NSMakeRange(runs[r].tile, runs[r].n_tiles),
                .heapOffset  = map ? runs[r].heap_tile : 0,
            };
        }

        [queue->queue updateBufferMappings:sparse heap:(map ? heap->heap : nil) operations:ops count:n_runs];
        if (signal != NULL) {
            [queue->queue signalEvent:(id<MTLSharedEvent>) ggml_metal_event_get_obj(signal) value:value];
        }

        free(ops);
    }

    return true;
}

#pragma clang diagnostic pop

#else

bool ggml_metal_placement_is_supported(ggml_metal_device_t dev) {
    GGML_UNUSED(dev);
    return false;
}

size_t ggml_metal_placement_tile_bytes(void) {
    return GGML_METAL_PLACEMENT_TILE_BYTES;
}

ggml_metal_placement_heap_t ggml_metal_placement_heap_init(ggml_metal_device_t dev, size_t n_tiles) {
    GGML_LOG_ERROR("%s: placement sparse buffers need the macOS 26.4 SDK\n", __func__);
    GGML_UNUSED(dev);
    GGML_UNUSED(n_tiles);
    return NULL;
}

void * ggml_metal_placement_heap_contents(ggml_metal_placement_heap_t heap) {
    GGML_UNUSED(heap);
    return NULL;
}

ggml_metal_buffer_t ggml_metal_sparse_buffer_init(ggml_metal_device_t dev, void * va_base, size_t size) {
    GGML_LOG_ERROR("%s: placement sparse buffers need the macOS 26.4 SDK\n", __func__);
    GGML_UNUSED(dev);
    GGML_UNUSED(va_base);
    GGML_UNUSED(size);
    return NULL;
}

bool ggml_metal_sparse_buffer_renew(ggml_metal_device_t dev, ggml_metal_buffer_t buf) {
    GGML_UNUSED(dev);
    GGML_UNUSED(buf);
    return false;
}

ggml_metal_mapping_queue_t ggml_metal_mapping_queue_init(ggml_metal_device_t dev) {
    GGML_LOG_ERROR("%s: placement sparse buffers need the macOS 26.4 SDK\n", __func__);
    GGML_UNUSED(dev);
    return NULL;
}

// without the SDK no heap, sparse buffer or mapping queue exists, so nothing below is reachable

void ggml_metal_placement_heap_free(ggml_metal_placement_heap_t heap) { GGML_UNUSED(heap); }
void ggml_metal_mapping_queue_free (ggml_metal_mapping_queue_t queue) { GGML_UNUSED(queue); }

bool ggml_metal_update_tiles(ggml_metal_mapping_queue_t queue, ggml_metal_placement_heap_t heap, ggml_metal_buffer_t buf,
        const struct ggml_metal_tile_run * runs, size_t n_runs, ggml_metal_event_t signal, uint64_t value) {
    GGML_UNUSED(queue); GGML_UNUSED(heap); GGML_UNUSED(buf); GGML_UNUSED(runs); GGML_UNUSED(n_runs); GGML_UNUSED(signal); GGML_UNUSED(value);
    return false;
}

#endif

//
// blits
//

// what a queue's completion handlers share with it; whichever of them and the queue lets go last frees it
struct ggml_metal_blit_state {
    _Atomic int refs;

    // the first failed command buffer's error, owned
    _Atomic(char *) error;
};

static void ggml_metal_blit_state_release(struct ggml_metal_blit_state * state) {
    if (atomic_fetch_sub(&state->refs, 1) == 1) {
        free(atomic_load(&state->error));
        free(state);
    }
}

struct ggml_metal_blit_queue {
    ggml_metal_device_t dev;

    id<MTLCommandQueue> queue;

    // the queue's timeline: blit value v signals it to v
    ggml_metal_event_t ev;
    id<MTLSharedEvent> event;

    // the last value handed out; the lock is held from assigning a value until its blit is committed,
    // so commit order is value order
    uint64_t        issued;
    pthread_mutex_t lock;

    struct ggml_metal_blit_state * state;
};

ggml_metal_blit_queue_t ggml_metal_blit_queue_init(ggml_metal_device_t dev, ggml_metal_placement_heap_t heap) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    ggml_metal_event_t ev = ggml_metal_device_event_init(dev);
    if (queue == nil || ev == NULL) {
        GGML_LOG_ERROR("%s: failed to create a command queue and its event\n", __func__);
        [queue release];
        if (ev != NULL) {
            ggml_metal_device_event_free(dev, ev);
        }
        return NULL;
    }

    ggml_metal_blit_queue_t res = calloc(1, sizeof(struct ggml_metal_blit_queue));
    res->dev   = dev;
    res->queue = queue;
    res->ev    = ev;
    res->event = (id<MTLSharedEvent>) ggml_metal_event_get_obj(ev);
    pthread_mutex_init(&res->lock, NULL);

    res->state = calloc(1, sizeof(struct ggml_metal_blit_state));
    atomic_init(&res->state->refs, 1);
    atomic_init(&res->state->error, NULL);

    GGML_UNUSED(heap);

    return res;
}

void ggml_metal_blit_queue_free(ggml_metal_blit_queue_t queue) {
    if (queue == NULL) {
        return;
    }

    // a failed blit stops the timeline, so only a queue without an error drains; blits parked behind
    // a failure stay parked, and their handlers keep the shared state alive
    const uint64_t drain_ms = 10000;
    if (atomic_load(&queue->state->error) == NULL && queue->issued > 0 &&
        ![queue->event waitUntilSignaledValue:queue->issued timeoutMS:drain_ms]) {
        GGML_LOG_ERROR("%s: blits up to %llu did not complete in %llu ms\n", __func__,
                (unsigned long long) queue->issued, (unsigned long long) drain_ms);
    }

    ggml_metal_blit_state_release(queue->state);
    ggml_metal_device_event_free(queue->dev, queue->ev);
    pthread_mutex_destroy(&queue->lock);
    [queue->queue release];
    free(queue);
}

const char * ggml_metal_blit_queue_error(ggml_metal_blit_queue_t queue) {
    return atomic_load(&queue->state->error);
}

ggml_metal_event_t ggml_metal_blit_queue_event(ggml_metal_blit_queue_t queue) {
    return queue->ev;
}

static bool ggml_metal_blit_args_valid(size_t offs, size_t size, const char * caller) {
    if (size == 0 || offs % 4 != 0 || size % 4 != 0) {
        GGML_LOG_ERROR("%s: [%zu, +%zu): offset and size must be multiples of 4 and the size non-zero\n", caller, offs, size);
        return false;
    }
    return true;
}

static struct ggml_metal_buffer_id ggml_metal_blit_resolve(ggml_metal_buffer_t buf, size_t offs, size_t size, const char * caller) {
    const struct ggml_metal_buffer_id id = ggml_metal_buffer_get_range(buf, offs, size);
    if (id.metal == nil) {
        GGML_LOG_ERROR("%s: [%zu, +%zu) is not in one Metal buffer of a %zu byte buffer\n", caller, offs, size, ggml_metal_buffer_get_size(buf));
    }
    return id;
}

// A blit with the queue lock held: it waits for the previous blit's value, so that blits complete
// as an in-order prefix and a failed one stops every later one, then for the caller's waits.
// `encode` adds the blit encoder's work. The value signals from the completion handler, which sees
// the command buffer's status first, so a failed blit never signals.
static uint64_t ggml_metal_blit_submit(ggml_metal_blit_queue_t queue, const struct ggml_metal_wait * waits, size_t n_waits,
        void (^encode)(id<MTLBlitCommandEncoder> encoder)) {
    pthread_mutex_lock(&queue->lock);

    const uint64_t value = queue->issued + 1;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [queue->queue commandBuffer];

        [cmd_buf encodeWaitForEvent:queue->event value:value - 1];
        for (size_t i = 0; i < n_waits; ++i) {
            ggml_metal_event_encode_wait_value(waits[i].event, cmd_buf, waits[i].value);
        }

        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        encode(encoder);
        [encoder endEncoding];

        id<MTLSharedEvent> event = queue->event;
        struct ggml_metal_blit_state * state = queue->state;
        atomic_fetch_add(&state->refs, 1);

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            if (cb.status == MTLCommandBufferStatusCompleted) {
                event.signaledValue = value;
            } else {
                const char * desc = cb.error ? [[cb.error description] UTF8String] : "no error";
                GGML_LOG_ERROR("ggml_metal_blit_queue: blit %llu failed with status %d: %s\n",
                        (unsigned long long) value, (int) cb.status, desc);

                char * mine = strdup(desc);
                char * none = NULL;
                if (!atomic_compare_exchange_strong(&state->error, &none, mine)) {
                    free(mine);
                }
            }
            ggml_metal_blit_state_release(state);
        }];

        [cmd_buf commit];
    }

    queue->issued = value;

    pthread_mutex_unlock(&queue->lock);

    return value;
}

uint64_t ggml_metal_blit_copy(ggml_metal_blit_queue_t queue, ggml_metal_buffer_t src, size_t src_offs, ggml_metal_buffer_t dst, size_t dst_offs, size_t size,
        const struct ggml_metal_wait * waits, size_t n_waits) {
    if (!ggml_metal_blit_args_valid(src_offs, size, __func__) || !ggml_metal_blit_args_valid(dst_offs, size, __func__)) {
        return 0;
    }

    const struct ggml_metal_buffer_id from = ggml_metal_blit_resolve(src, src_offs, size, __func__);
    const struct ggml_metal_buffer_id to   = ggml_metal_blit_resolve(dst, dst_offs, size, __func__);
    if (from.metal == nil || to.metal == nil) {
        return 0;
    }

    return ggml_metal_blit_submit(queue, waits, n_waits, ^(id<MTLBlitCommandEncoder> encoder) {
        [encoder copyFromBuffer:(id<MTLBuffer>) from.metal sourceOffset:from.offs toBuffer:(id<MTLBuffer>) to.metal destinationOffset:to.offs size:size];
    });
}

uint64_t ggml_metal_blit_fill(ggml_metal_blit_queue_t queue, ggml_metal_buffer_t dst, size_t offs, size_t size, uint8_t byte,
        const struct ggml_metal_wait * waits, size_t n_waits) {
    if (!ggml_metal_blit_args_valid(offs, size, __func__)) {
        return 0;
    }

    const struct ggml_metal_buffer_id to = ggml_metal_blit_resolve(dst, offs, size, __func__);
    if (to.metal == nil) {
        return 0;
    }

    return ggml_metal_blit_submit(queue, waits, n_waits, ^(id<MTLBlitCommandEncoder> encoder) {
        [encoder fillBuffer:(id<MTLBuffer>) to.metal range:NSMakeRange(to.offs, size) value:byte];
    });
}

//
// keep-prepared
//

#if defined(GGML_METAL_HAS_PLACEMENT_SPARSE)

bool ggml_metal_keep_prepared(ggml_metal_placement_heap_t const * heaps, size_t n_heaps, ggml_metal_buffer_t const * bufs, size_t n_bufs) {
    if (n_heaps == 0) {
        return false;
    }
    size_t n_objs = 0;
    for (size_t i = 0; i < n_bufs; ++i) {
        n_objs += (size_t) ggml_metal_buffer_get_n_objs(bufs[i]);
    }

    id<MTLResource> * objs = malloc((n_objs > 0 ? n_objs : 1)*sizeof(id<MTLResource>));
    size_t k = 0;
    for (size_t i = 0; i < n_bufs; ++i) {
        for (int j = 0; j < ggml_metal_buffer_get_n_objs(bufs[i]); ++j) {
            objs[k++] = (id<MTLResource>) ggml_metal_buffer_get_obj(bufs[i], j);
        }
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [heaps[0]->keep_queue commandBuffer];

        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        for (size_t i = 0; i < n_heaps; ++i) {
            [encoder useHeap:heaps[i]->heap];
        }
        if (n_objs > 0) {
            [encoder useResources:objs count:n_objs usage:MTLResourceUsageRead];
        }
        [encoder setComputePipelineState:heaps[0]->nop];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];

        [cmd_buf commit];
    }

    free(objs);

    return true;
}

#else

bool ggml_metal_keep_prepared(ggml_metal_placement_heap_t const * heaps, size_t n_heaps, ggml_metal_buffer_t const * bufs, size_t n_bufs) {
    GGML_UNUSED(heaps); GGML_UNUSED(n_heaps); GGML_UNUSED(bufs); GGML_UNUSED(n_bufs);
    return false;
}

#endif
