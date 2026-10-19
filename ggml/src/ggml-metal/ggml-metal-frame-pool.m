#import "ggml-metal-frame-pool.h"

#import "ggml-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <pthread.h>
#include <stdatomic.h>

// placement sparse resources and the Metal 4 mapping queue need macOS 26.4
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
#define GGML_METAL_HAS_FRAME_POOL 1
#endif

#if defined(GGML_METAL_HAS_FRAME_POOL)

// every pool and region is created behind an availability check, so nothing below runs on an older OS
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

#define GGML_METAL_FRAME_POOL_NO_SLOT UINT32_MAX

static const MTLSparsePageSize GGML_METAL_FRAME_POOL_PAGE = MTLSparsePageSize16;

struct ggml_metal_frame_region {
    ggml_metal_device_t dev;

    // owned by the device buffer created over this region
    id<MTLBuffer> metal;

    size_t n_tiles;

    // the pool the region's tiles map into, from its first map until the region or the pool is freed;
    // the fields below are guarded by that pool's mapping lock
    ggml_metal_frame_pool_t pool;

    // index in the pool's region list
    size_t idx;

    // per tile, as queued: the bound slot (or NO_SLOT) and the mapping value of its map
    uint32_t * slot_of;
    uint64_t * mapped_at;
};

struct ggml_metal_frame_pool_timeline {
    id<MTLSharedEvent> event;

    // the last value handed out; guarded by the timeline's lock
    uint64_t issued;

    // held from assigning a value until its work is submitted, so that submission order is value order
    pthread_mutex_t lock;
};

struct ggml_metal_frame_pool {
    ggml_metal_device_t dev;

    id<MTLDevice>        device;
    id<MTLHeap>          heap;
    id<MTLBuffer>        stage;
    id<MTL4CommandQueue> map_queue;
    id<MTLCommandQueue>  blit_queue;
    id<MTLCommandQueue>  compute_queue; // ggml's queue, not owned
    id<MTLResidencySet>  rset;

    // an empty kernel: the driver skips an encoder that dispatches nothing, references included
    id<MTLComputePipelineState> nop;

    // the residency set is attached to the queues
    bool resident;

    size_t tile_bytes;
    size_t n_slots;
    size_t stage_bytes;

    // the mapping lock also guards the slot and tile bookkeeping and the region list; a blit takes it
    // inside the blit lock, never the other way around
    struct ggml_metal_frame_pool_timeline map;
    struct ggml_metal_frame_pool_timeline blit;

    // per slot, as queued: bound to a tile
    bool * slot_bound;

    // the first blit value whose command buffer failed, 0 while none has
    _Atomic uint64_t blit_failed_at;

    // the regions with tiles in this pool
    struct ggml_metal_frame_region ** regions;
    size_t n_regions;
    size_t cap_regions;
};

enum ggml_metal_frame_pool_blit_kind {
    GGML_METAL_FRAME_POOL_BLIT_FILL,
    GGML_METAL_FRAME_POOL_BLIT_CLEAR,
    GGML_METAL_FRAME_POOL_BLIT_SPILL,
    GGML_METAL_FRAME_POOL_BLIT_KEEP_PREPARED,
};

static const char * ggml_metal_frame_pool_blit_name(enum ggml_metal_frame_pool_blit_kind kind) {
    switch (kind) {
        case GGML_METAL_FRAME_POOL_BLIT_FILL:          return "fill";
        case GGML_METAL_FRAME_POOL_BLIT_CLEAR:         return "clear";
        case GGML_METAL_FRAME_POOL_BLIT_SPILL:         return "spill";
        case GGML_METAL_FRAME_POOL_BLIT_KEEP_PREPARED: return "keep_prepared";
    }
    return "?";
}

static bool ggml_metal_frame_pool_supported(id<MTLDevice> device, const char * caller) {
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

static size_t ggml_metal_frame_tile_bytes(id<MTLDevice> device) {
    return [device sparseTileSizeInBytesForSparsePageSize:GGML_METAL_FRAME_POOL_PAGE];
}

//
// pool
//

ggml_metal_frame_pool_t ggml_metal_frame_pool_init(ggml_metal_device_t dev, size_t n_slots, size_t stage_bytes) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);

    if (!ggml_metal_frame_pool_supported(device, __func__)) {
        return NULL;
    }

    if (n_slots == 0 || n_slots >= GGML_METAL_FRAME_POOL_NO_SLOT || stage_bytes == 0) {
        GGML_LOG_ERROR("%s: invalid pool: n_slots = %zu, stage_bytes = %zu\n", __func__, n_slots, stage_bytes);
        return NULL;
    }

    ggml_metal_frame_pool_t pool = calloc(1, sizeof(struct ggml_metal_frame_pool));

    pool->dev           = dev;
    pool->device        = device;
    pool->compute_queue = (id<MTLCommandQueue>) ggml_metal_device_get_queue(dev);
    pool->tile_bytes    = ggml_metal_frame_tile_bytes(device);
    pool->n_slots       = n_slots;
    pool->stage_bytes   = stage_bytes;

    pthread_mutex_init(&pool->map.lock,  NULL);
    pthread_mutex_init(&pool->blit.lock, NULL);

    pool->slot_bound = calloc(n_slots, sizeof(bool));

    @autoreleasepool {
        MTLHeapDescriptor * desc = [[MTLHeapDescriptor alloc] init];
        desc.type        = MTLHeapTypePlacement;
        desc.storageMode = MTLStorageModePrivate;
        desc.size        = n_slots*pool->tile_bytes;
        desc.maxCompatiblePlacementSparsePageSize = GGML_METAL_FRAME_POOL_PAGE;

        pool->heap = [device newHeapWithDescriptor:desc];
        [desc release];

        pool->stage      = [device newBufferWithLength:stage_bytes options:MTLResourceStorageModeShared];
        pool->map_queue  = [device newMTL4CommandQueue];
        pool->blit_queue = [device newCommandQueue];

        pool->map.event  = [device newSharedEvent];
        pool->blit.event = [device newSharedEvent];

        MTLResidencySetDescriptor * rdesc = [[MTLResidencySetDescriptor alloc] init];
        rdesc.label = @"ggml_metal_frame_pool";

        NSError * error = nil;
        pool->rset = [device newResidencySetWithDescriptor:rdesc error:&error];
        [rdesc release];

        if (error) {
            GGML_LOG_ERROR("%s: residency set: %s\n", __func__, [[error description] UTF8String]);
        }

        error = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:@"kernel void ggml_metal_frame_pool_nop() {}" options:nil error:&error];
        id<MTLFunction> fn = [lib newFunctionWithName:@"ggml_metal_frame_pool_nop"];
        if (fn != nil) {
            pool->nop = [device newComputePipelineStateWithFunction:fn error:&error];
        }
        if (error) {
            GGML_LOG_ERROR("%s: keep-prepared kernel: %s\n", __func__, [[error description] UTF8String]);
        }
        [fn release];
        [lib release];
    }

    if (pool->heap == nil || pool->stage == nil || pool->map_queue == nil || pool->blit_queue == nil ||
        pool->map.event == nil || pool->blit.event == nil || pool->rset == nil || pool->compute_queue == nil || pool->nop == nil) {
        GGML_LOG_ERROR("%s: failed to create a pool of %zu tiles (%.2f MiB) with %.2f MiB of staging\n", __func__,
                n_slots, n_slots*pool->tile_bytes/1024.0/1024.0, stage_bytes/1024.0/1024.0);
        ggml_metal_frame_pool_free(pool);
        return NULL;
    }

    [pool->rset addAllocation:pool->heap];
    [pool->rset commit];
    [pool->rset requestResidency];

    // the heap must be resident to every queue that touches a region: the mapping queue, the blit
    // queue and ggml's compute queue
    [pool->map_queue     addResidencySet:pool->rset];
    [pool->blit_queue    addResidencySet:pool->rset];
    [pool->compute_queue addResidencySet:pool->rset];

    pool->resident = true;

    return pool;
}

// queues an unmap of every mapped tile of the region and frees their slots; called with the mapping
// lock held, returns whether anything was queued
static bool ggml_metal_frame_pool_unmap_all(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region) {
    size_t n_ops = 0;
    MTL4UpdateSparseBufferMappingOperation * ops = malloc(region->n_tiles*sizeof(MTL4UpdateSparseBufferMappingOperation));

    for (size_t t = 0; t < region->n_tiles; ++t) {
        if (region->slot_of[t] == GGML_METAL_FRAME_POOL_NO_SLOT) {
            continue;
        }
        pool->slot_bound[region->slot_of[t]] = false;
        region->slot_of[t] = GGML_METAL_FRAME_POOL_NO_SLOT;

        if (n_ops > 0 && NSMaxRange(ops[n_ops - 1].bufferRange) == t) {
            ops[n_ops - 1].bufferRange.length++;
        } else {
            ops[n_ops++] = (MTL4UpdateSparseBufferMappingOperation) {
                .mode        = MTLSparseTextureMappingModeUnmap,
                .bufferRange = NSMakeRange(t, 1),
                .heapOffset  = 0,
            };
        }
    }

    if (n_ops > 0) {
        [pool->map_queue updateBufferMappings:region->metal heap:nil operations:ops count:n_ops];
    }
    free(ops);

    return n_ops > 0;
}

void ggml_metal_frame_pool_free(ggml_metal_frame_pool_t pool) {
    if (pool == NULL) {
        return;
    }

    // in-flight blits reference the pool from their completion handlers
    pthread_mutex_lock(&pool->blit.lock);
    if (pool->blit.issued > 0) {
        [pool->blit.event waitUntilSignaledValue:pool->blit.issued timeoutMS:UINT64_MAX];
    }
    pthread_mutex_unlock(&pool->blit.lock);

    // regions outlive the pool: unmap their tiles from the heap and detach them
    pthread_mutex_lock(&pool->map.lock);
    bool unmapped = false;
    for (size_t i = 0; i < pool->n_regions; ++i) {
        unmapped |= ggml_metal_frame_pool_unmap_all(pool, pool->regions[i]);
        pool->regions[i]->pool = NULL;
    }
    pool->n_regions = 0;
    if (unmapped) {
        [pool->map_queue signalEvent:pool->map.event value:++pool->map.issued];
    }
    if (pool->map.issued > 0) {
        [pool->map.event waitUntilSignaledValue:pool->map.issued timeoutMS:UINT64_MAX];
    }
    pthread_mutex_unlock(&pool->map.lock);

    if (pool->resident) {
        [pool->compute_queue removeResidencySet:pool->rset];
        [pool->blit_queue    removeResidencySet:pool->rset];
        [pool->map_queue     removeResidencySet:pool->rset];
        [pool->rset endResidency];
        [pool->rset removeAllAllocations];
        [pool->rset commit];
    }

    [pool->nop        release];
    [pool->rset       release];
    [pool->blit.event release];
    [pool->map.event  release];
    [pool->blit_queue release];
    [pool->map_queue  release];
    [pool->stage      release];
    [pool->heap       release];

    pthread_mutex_destroy(&pool->blit.lock);
    pthread_mutex_destroy(&pool->map.lock);

    free(pool->regions);
    free(pool->slot_bound);
    free(pool);
}

size_t ggml_metal_frame_pool_get_tile_bytes(ggml_metal_frame_pool_t pool) {
    return pool->tile_bytes;
}

void * ggml_metal_frame_pool_get_stage(ggml_metal_frame_pool_t pool) {
    return pool->stage.contents;
}

//
// regions
//

ggml_metal_buffer_t ggml_metal_frame_region_init(ggml_metal_device_t dev, void * va_base, size_t size) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);

    if (!ggml_metal_frame_pool_supported(device, __func__)) {
        return NULL;
    }

    const size_t tile_bytes = ggml_metal_frame_tile_bytes(device);

    if (size == 0 || va_base == NULL || (uintptr_t) va_base % tile_bytes != 0) {
        GGML_LOG_ERROR("%s: a region needs a non-empty size and a base aligned to %zu bytes (size = %zu, base = %p)\n",
                __func__, tile_bytes, size, va_base);
        return NULL;
    }

    const size_t n_tiles = (size + tile_bytes - 1)/tile_bytes;
    const size_t bytes   = n_tiles*tile_bytes;

    if (n_tiles >= GGML_METAL_FRAME_POOL_NO_SLOT) {
        GGML_LOG_ERROR("%s: a region of %zu tiles is too large\n", __func__, n_tiles);
        return NULL;
    }

    id<MTLBuffer> metal = [device newBufferWithLength:bytes
                                              options:MTLResourceStorageModePrivate
                              placementSparsePageSize:GGML_METAL_FRAME_POOL_PAGE];
    if (metal == nil) {
        GGML_LOG_ERROR("%s: failed to create a placement sparse buffer of %zu tiles (%.2f MiB)\n", __func__,
                n_tiles, bytes/1024.0/1024.0);
        return NULL;
    }

    struct ggml_metal_frame_region * region = calloc(1, sizeof(struct ggml_metal_frame_region));

    region->dev       = dev;
    region->metal     = metal;
    region->n_tiles   = n_tiles;
    region->slot_of   = malloc(n_tiles*sizeof(uint32_t));
    region->mapped_at = calloc(n_tiles, sizeof(uint64_t));

    for (size_t i = 0; i < n_tiles; ++i) {
        region->slot_of[i] = GGML_METAL_FRAME_POOL_NO_SLOT;
    }

    return ggml_metal_buffer_init_region(dev, metal, va_base, bytes, region);
}

void ggml_metal_frame_region_free(struct ggml_metal_frame_region * region) {
    ggml_metal_frame_pool_t pool = region->pool;

    if (pool != NULL) {
        pthread_mutex_lock(&pool->map.lock);

        // a later map of the same slots applies after this unmap on the in-order mapping queue
        if (ggml_metal_frame_pool_unmap_all(pool, region)) {
            [pool->map_queue signalEvent:pool->map.event value:++pool->map.issued];
        }

        struct ggml_metal_frame_region * last = pool->regions[--pool->n_regions];
        pool->regions[region->idx] = last;
        last->idx = region->idx;

        pthread_mutex_unlock(&pool->map.lock);
    }

    free(region->mapped_at);
    free(region->slot_of);
    free(region);
}

// adds the region to the pool's region list on its first map; called with the mapping lock held
static bool ggml_metal_frame_pool_attach(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region) {
    if (region->pool == pool) {
        return true;
    }
    if (region->pool != NULL) {
        GGML_LOG_ERROR("%s: the region has tiles in another frame pool\n", __func__);
        return false;
    }
    if (region->dev != pool->dev) {
        GGML_LOG_ERROR("%s: the region and the frame pool are on different devices\n", __func__);
        return false;
    }

    if (pool->n_regions == pool->cap_regions) {
        pool->cap_regions = pool->cap_regions ? 2*pool->cap_regions : 64;
        pool->regions     = realloc(pool->regions, pool->cap_regions*sizeof(struct ggml_metal_frame_region *));
    }
    region->pool = pool;
    region->idx  = pool->n_regions;
    pool->regions[pool->n_regions++] = region;

    return true;
}

//
// mapping
//

static bool ggml_metal_frame_pool_run_valid(ggml_metal_frame_pool_t pool, const struct ggml_metal_frame_region * region,
        const struct ggml_metal_tile_run * run, bool map) {
    if (run->n_tiles == 0 || (size_t) run->tile + run->n_tiles > region->n_tiles) {
        GGML_LOG_ERROR("%s: tiles [%u, +%u) outside a region of %zu tiles\n", __func__, run->tile, run->n_tiles, region->n_tiles);
        return false;
    }
    if (map && (size_t) run->slot + run->n_tiles > pool->n_slots) {
        GGML_LOG_ERROR("%s: slots [%u, +%u) outside a pool of %zu slots\n", __func__, run->slot, run->n_tiles, pool->n_slots);
        return false;
    }
    return true;
}

// one tile's bookkeeping change, kept so that a refused batch can be rolled back
struct ggml_metal_frame_pool_step {
    uint32_t tile;
    uint32_t prev_slot;
};

// binds the tiles of one run to their slots, or unbinds them, as long as each step is legal
static bool ggml_metal_frame_pool_apply_run(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region,
        const struct ggml_metal_tile_run * run, bool map, struct ggml_metal_frame_pool_step * steps, size_t * n_steps) {
    for (uint32_t i = 0; i < run->n_tiles; ++i) {
        const uint32_t tile = run->tile + i;
        const uint32_t cur  = region->slot_of[tile];

        if (map) {
            const uint32_t slot = run->slot + i;
            if (cur != GGML_METAL_FRAME_POOL_NO_SLOT) {
                GGML_LOG_ERROR("%s: map of tile %u, which is mapped to slot %u\n", __func__, tile, cur);
                return false;
            }
            if (pool->slot_bound[slot]) {
                GGML_LOG_ERROR("%s: map of tile %u to slot %u, which is bound to another tile\n", __func__, tile, slot);
                return false;
            }
            pool->slot_bound[slot] = true;
            region->slot_of[tile]  = slot;
        } else {
            if (cur == GGML_METAL_FRAME_POOL_NO_SLOT) {
                GGML_LOG_ERROR("%s: unmap of tile %u, which is not mapped\n", __func__, tile);
                return false;
            }
            pool->slot_bound[cur] = false;
            region->slot_of[tile] = GGML_METAL_FRAME_POOL_NO_SLOT;
        }

        steps[(*n_steps)++] = (struct ggml_metal_frame_pool_step) { tile, cur };
    }

    return true;
}

// undoes applied steps newest first, which restores the state before the batch exactly
static void ggml_metal_frame_pool_rollback(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region,
        const struct ggml_metal_frame_pool_step * steps, size_t n_steps) {
    for (size_t k = n_steps; k-- > 0; ) {
        const uint32_t tile = steps[k].tile;
        const uint32_t cur  = region->slot_of[tile];

        if (cur != GGML_METAL_FRAME_POOL_NO_SLOT) {
            pool->slot_bound[cur] = false;
        }
        if (steps[k].prev_slot != GGML_METAL_FRAME_POOL_NO_SLOT) {
            pool->slot_bound[steps[k].prev_slot] = true;
        }
        region->slot_of[tile] = steps[k].prev_slot;
    }
}

static uint64_t ggml_metal_frame_pool_update(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region,
        const struct ggml_metal_tile_run * runs, size_t n_runs, bool map) {
    if (n_runs == 0) {
        GGML_LOG_ERROR("%s: no runs\n", __func__);
        return 0;
    }

    size_t n_tiles = 0;
    for (size_t r = 0; r < n_runs; ++r) {
        if (!ggml_metal_frame_pool_run_valid(pool, region, &runs[r], map)) {
            return 0;
        }
        n_tiles += runs[r].n_tiles;
    }

    struct ggml_metal_frame_pool_step * steps = malloc(n_tiles*sizeof(struct ggml_metal_frame_pool_step));
    MTL4UpdateSparseBufferMappingOperation * ops = malloc(n_runs*sizeof(MTL4UpdateSparseBufferMappingOperation));

    uint64_t value = 0;

    pthread_mutex_lock(&pool->map.lock);

    if (atomic_load(&pool->blit_failed_at) != 0) {
        GGML_LOG_ERROR("%s: the pool is in its error state\n", __func__);
    } else if (region->pool != pool && !map) {
        GGML_LOG_ERROR("%s: unmap of a region with no tiles in this pool\n", __func__);
    } else if (ggml_metal_frame_pool_attach(pool, region)) {
        size_t n_steps = 0;
        bool ok = true;
        for (size_t r = 0; r < n_runs && ok; ++r) {
            ok = ggml_metal_frame_pool_apply_run(pool, region, &runs[r], map, steps, &n_steps);
        }

        if (!ok) {
            ggml_metal_frame_pool_rollback(pool, region, steps, n_steps);
        } else {
            value = ++pool->map.issued;

            for (size_t r = 0; r < n_runs; ++r) {
                ops[r] = (MTL4UpdateSparseBufferMappingOperation) {
                    .mode        = map ? MTLSparseTextureMappingModeMap : MTLSparseTextureMappingModeUnmap,
                    .bufferRange = NSMakeRange(runs[r].tile, runs[r].n_tiles),
                    .heapOffset  = map ? runs[r].slot : 0,
                };
                if (map) {
                    for (uint32_t i = 0; i < runs[r].n_tiles; ++i) {
                        region->mapped_at[runs[r].tile + i] = value;
                    }
                }
            }

            [pool->map_queue updateBufferMappings:region->metal heap:(map ? pool->heap : nil) operations:ops count:n_runs];
            [pool->map_queue signalEvent:pool->map.event value:value];
        }
    }

    pthread_mutex_unlock(&pool->map.lock);

    free(ops);
    free(steps);

    return value;
}

uint64_t ggml_metal_frame_pool_map(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, const struct ggml_metal_tile_run * runs, size_t n_runs) {
    return ggml_metal_frame_pool_update(pool, region, runs, n_runs, true);
}

uint64_t ggml_metal_frame_pool_unmap(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, const struct ggml_metal_tile_run * runs, size_t n_runs) {
    return ggml_metal_frame_pool_update(pool, region, runs, n_runs, false);
}

//
// blits
//

struct ggml_metal_frame_pool_blit_args {
    enum ggml_metal_frame_pool_blit_kind kind;

    size_t   offs;
    size_t   stage_offs;
    size_t   size;
    uint8_t  value;
    uint64_t after_map;

    ggml_metal_event_t wait_ev;
    uint64_t           wait_value;
};

static bool ggml_metal_frame_pool_blit_valid(ggml_metal_frame_pool_t pool, const struct ggml_metal_frame_region * region,
        const struct ggml_metal_frame_pool_blit_args * a) {
    const char * name = ggml_metal_frame_pool_blit_name(a->kind);

    if (a->size == 0 || a->offs % 4 != 0 || a->stage_offs % 4 != 0 || a->size % 4 != 0) {
        GGML_LOG_ERROR("%s: %s of [%zu, +%zu) from staging %zu: offsets and size must be non-zero multiples of 4\n",
                __func__, name, a->offs, a->size, a->stage_offs);
        return false;
    }
    if (a->offs + a->size > region->n_tiles*pool->tile_bytes) {
        GGML_LOG_ERROR("%s: %s of [%zu, +%zu) outside a region of %zu bytes\n", __func__, name, a->offs, a->size,
                region->n_tiles*pool->tile_bytes);
        return false;
    }
    if (a->kind != GGML_METAL_FRAME_POOL_BLIT_CLEAR && a->stage_offs + a->size > pool->stage_bytes) {
        GGML_LOG_ERROR("%s: %s of staging [%zu, +%zu) outside %zu bytes of staging\n", __func__, name, a->stage_offs, a->size,
                pool->stage_bytes);
        return false;
    }

    // the queued tile state is the mapping thread's, so read it under the mapping lock
    bool ok = true;

    pthread_mutex_lock(&pool->map.lock);

    if (a->after_map > pool->map.issued) {
        GGML_LOG_ERROR("%s: %s waits for mapping value %llu, which was never issued (last %llu)\n", __func__, name,
                (unsigned long long) a->after_map, (unsigned long long) pool->map.issued);
        ok = false;
    }

    const size_t t0 = a->offs/pool->tile_bytes;
    const size_t t1 = (a->offs + a->size - 1)/pool->tile_bytes;

    for (size_t t = t0; ok && t <= t1; ++t) {
        if (region->pool != pool || region->slot_of[t] == GGML_METAL_FRAME_POOL_NO_SLOT) {
            GGML_LOG_ERROR("%s: %s of [%zu, +%zu) touches tile %zu, which is not mapped in this pool\n", __func__, name,
                    a->offs, a->size, t);
            ok = false;
        } else if (region->mapped_at[t] > a->after_map) {
            GGML_LOG_ERROR("%s: %s of [%zu, +%zu) touches tile %zu, mapped at %llu, but waits only for mapping value %llu\n",
                    __func__, name, a->offs, a->size, t, (unsigned long long) region->mapped_at[t], (unsigned long long) a->after_map);
            ok = false;
        }
    }

    pthread_mutex_unlock(&pool->map.lock);

    return ok;
}

// commits a blit-queue command buffer as the next blit value, called with the blit lock held
static uint64_t ggml_metal_frame_pool_commit_blit(ggml_metal_frame_pool_t pool, id<MTLCommandBuffer> cmd_buf,
        enum ggml_metal_frame_pool_blit_kind kind) {
    const uint64_t value = ++pool->blit.issued;

    id<MTLSharedEvent> event = pool->blit.event;
    _Atomic uint64_t * failed_at = &pool->blit_failed_at;

    // the value signals from the completion handler, which sees the command buffer's status first, so
    // a failed blit never reports complete
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.status != MTLCommandBufferStatusCompleted) {
            GGML_LOG_ERROR("ggml_metal_frame_pool: %s, blit value %llu, failed with status %d: %s\n",
                    ggml_metal_frame_pool_blit_name(kind), (unsigned long long) value, (int) cb.status,
                    cb.error ? [[cb.error description] UTF8String] : "no error");

            uint64_t none = 0;
            atomic_compare_exchange_strong(failed_at, &none, value);

            // wake every waiter; the error state makes their waits fail
            event.signaledValue = UINT64_MAX;
            return;
        }
        if (atomic_load(failed_at) == 0) {
            event.signaledValue = value;
        }
    }];

    [cmd_buf commit];

    return value;
}

static uint64_t ggml_metal_frame_pool_blit(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region,
        const struct ggml_metal_frame_pool_blit_args * a) {
    uint64_t value = 0;

    pthread_mutex_lock(&pool->blit.lock);

    if (atomic_load(&pool->blit_failed_at) != 0) {
        GGML_LOG_ERROR("%s: the pool is in its error state\n", __func__);
    } else if (ggml_metal_frame_pool_blit_valid(pool, region, a)) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd_buf = [pool->blit_queue commandBuffer];

            if (a->after_map > 0) {
                [cmd_buf encodeWaitForEvent:pool->map.event value:a->after_map];
            }
            if (a->wait_ev != NULL) {
                ggml_metal_event_encode_wait_value(a->wait_ev, cmd_buf, a->wait_value);
            }

            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            switch (a->kind) {
                case GGML_METAL_FRAME_POOL_BLIT_FILL:
                    [encoder copyFromBuffer:pool->stage sourceOffset:a->stage_offs toBuffer:region->metal destinationOffset:a->offs size:a->size];
                    break;
                case GGML_METAL_FRAME_POOL_BLIT_CLEAR:
                    [encoder fillBuffer:region->metal range:NSMakeRange(a->offs, a->size) value:a->value];
                    break;
                case GGML_METAL_FRAME_POOL_BLIT_SPILL:
                    [encoder copyFromBuffer:region->metal sourceOffset:a->offs toBuffer:pool->stage destinationOffset:a->stage_offs size:a->size];
                    break;
                case GGML_METAL_FRAME_POOL_BLIT_KEEP_PREPARED:
                    GGML_ABORT("unreachable");
            }

            [encoder endEncoding];

            value = ggml_metal_frame_pool_commit_blit(pool, cmd_buf, a->kind);
        }
    }

    pthread_mutex_unlock(&pool->blit.lock);

    return value;
}

uint64_t ggml_metal_frame_pool_fill(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map) {
    const struct ggml_metal_frame_pool_blit_args a = {
        .kind = GGML_METAL_FRAME_POOL_BLIT_FILL, .offs = offs, .stage_offs = stage_offs, .size = size, .after_map = after_map,
    };
    return ggml_metal_frame_pool_blit(pool, region, &a);
}

uint64_t ggml_metal_frame_pool_clear(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t size, uint8_t value, uint64_t after_map) {
    const struct ggml_metal_frame_pool_blit_args a = {
        .kind = GGML_METAL_FRAME_POOL_BLIT_CLEAR, .offs = offs, .size = size, .value = value, .after_map = after_map,
    };
    return ggml_metal_frame_pool_blit(pool, region, &a);
}

uint64_t ggml_metal_frame_pool_spill(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map,
        ggml_metal_event_t wait_ev, uint64_t wait_value) {
    const struct ggml_metal_frame_pool_blit_args a = {
        .kind = GGML_METAL_FRAME_POOL_BLIT_SPILL, .offs = offs, .stage_offs = stage_offs, .size = size, .after_map = after_map,
        .wait_ev = wait_ev, .wait_value = wait_value,
    };
    return ggml_metal_frame_pool_blit(pool, region, &a);
}

//
// timelines
//

static struct ggml_metal_frame_pool_timeline * ggml_metal_frame_pool_timeline(ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q) {
    return q == GGML_METAL_FRAME_POOL_MAP ? &pool->map : &pool->blit;
}

uint64_t ggml_metal_frame_pool_completed(ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q) {
    const uint64_t signaled = ggml_metal_frame_pool_timeline(pool, q)->event.signaledValue;

    const uint64_t failed_at = atomic_load(&pool->blit_failed_at);
    if (q == GGML_METAL_FRAME_POOL_BLIT && failed_at != 0 && signaled >= failed_at) {
        return failed_at - 1;
    }

    return signaled;
}

bool ggml_metal_frame_pool_wait(ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q, uint64_t value, uint64_t timeout_ms) {
    struct ggml_metal_frame_pool_timeline * tl = ggml_metal_frame_pool_timeline(pool, q);

    pthread_mutex_lock(&tl->lock);
    const uint64_t issued = tl->issued;
    pthread_mutex_unlock(&tl->lock);

    if (value > issued) {
        GGML_LOG_ERROR("%s: wait for value %llu, which was never issued (last %llu)\n", __func__,
                (unsigned long long) value, (unsigned long long) issued);
        return false;
    }

    if (![tl->event waitUntilSignaledValue:value timeoutMS:timeout_ms]) {
        return false;
    }

    const uint64_t failed_at = atomic_load(&pool->blit_failed_at);

    return !(q == GGML_METAL_FRAME_POOL_BLIT && failed_at != 0 && value >= failed_at);
}

//
// keep-prepared
//

void ggml_metal_frame_pool_keep_prepared(ggml_metal_frame_pool_t pool) {
    pthread_mutex_lock(&pool->blit.lock);

    if (atomic_load(&pool->blit_failed_at) == 0) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd_buf = [pool->blit_queue commandBuffer];

            id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

            [encoder useHeap:pool->heap];

            pthread_mutex_lock(&pool->map.lock);

            const size_t n = pool->n_regions;
            id<MTLResource> * resources = malloc(n*sizeof(id<MTLResource>));
            for (size_t i = 0; i < n; ++i) {
                resources[i] = pool->regions[i]->metal;
            }

            pthread_mutex_unlock(&pool->map.lock);

            if (n > 0) {
                [encoder useResources:resources count:n usage:MTLResourceUsageRead];
            }
            free(resources);

            [encoder setComputePipelineState:pool->nop];
            [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

            [encoder endEncoding];

            ggml_metal_frame_pool_commit_blit(pool, cmd_buf, GGML_METAL_FRAME_POOL_BLIT_KEEP_PREPARED);
        }
    }

    pthread_mutex_unlock(&pool->blit.lock);
}

#else

ggml_metal_frame_pool_t ggml_metal_frame_pool_init(ggml_metal_device_t dev, size_t n_slots, size_t stage_bytes) {
    GGML_LOG_ERROR("%s: placement sparse buffers need the macOS 26.4 SDK\n", __func__);
    GGML_UNUSED(dev);
    GGML_UNUSED(n_slots);
    GGML_UNUSED(stage_bytes);
    return NULL;
}

ggml_metal_buffer_t ggml_metal_frame_region_init(ggml_metal_device_t dev, void * va_base, size_t size) {
    GGML_LOG_ERROR("%s: placement sparse buffers need the macOS 26.4 SDK\n", __func__);
    GGML_UNUSED(dev);
    GGML_UNUSED(va_base);
    GGML_UNUSED(size);
    return NULL;
}

// without the SDK no pool or region exists, so nothing below is reachable

void   ggml_metal_frame_pool_free          (ggml_metal_frame_pool_t pool) { GGML_UNUSED(pool); }
size_t ggml_metal_frame_pool_get_tile_bytes(ggml_metal_frame_pool_t pool) { GGML_UNUSED(pool); return 0; }
void * ggml_metal_frame_pool_get_stage     (ggml_metal_frame_pool_t pool) { GGML_UNUSED(pool); return NULL; }

void ggml_metal_frame_region_free(struct ggml_metal_frame_region * region) { GGML_UNUSED(region); }

uint64_t ggml_metal_frame_pool_map  (ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, const struct ggml_metal_tile_run * runs, size_t n_runs) { GGML_UNUSED(pool); GGML_UNUSED(region); GGML_UNUSED(runs); GGML_UNUSED(n_runs); return 0; }
uint64_t ggml_metal_frame_pool_unmap(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, const struct ggml_metal_tile_run * runs, size_t n_runs) { GGML_UNUSED(pool); GGML_UNUSED(region); GGML_UNUSED(runs); GGML_UNUSED(n_runs); return 0; }

uint64_t ggml_metal_frame_pool_fill (ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map) { GGML_UNUSED(pool); GGML_UNUSED(region); GGML_UNUSED(offs); GGML_UNUSED(stage_offs); GGML_UNUSED(size); GGML_UNUSED(after_map); return 0; }
uint64_t ggml_metal_frame_pool_clear(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t size, uint8_t value, uint64_t after_map) { GGML_UNUSED(pool); GGML_UNUSED(region); GGML_UNUSED(offs); GGML_UNUSED(size); GGML_UNUSED(value); GGML_UNUSED(after_map); return 0; }
uint64_t ggml_metal_frame_pool_spill(ggml_metal_frame_pool_t pool, struct ggml_metal_frame_region * region, size_t offs, size_t stage_offs, size_t size, uint64_t after_map,
        ggml_metal_event_t wait_ev, uint64_t wait_value) { GGML_UNUSED(pool); GGML_UNUSED(region); GGML_UNUSED(offs); GGML_UNUSED(stage_offs); GGML_UNUSED(size); GGML_UNUSED(after_map); GGML_UNUSED(wait_ev); GGML_UNUSED(wait_value); return 0; }

uint64_t ggml_metal_frame_pool_completed(ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q) { GGML_UNUSED(pool); GGML_UNUSED(q); return 0; }
bool     ggml_metal_frame_pool_wait     (ggml_metal_frame_pool_t pool, enum ggml_metal_frame_pool_queue q, uint64_t value, uint64_t timeout_ms) { GGML_UNUSED(pool); GGML_UNUSED(q); GGML_UNUSED(value); GGML_UNUSED(timeout_ms); return false; }

void ggml_metal_frame_pool_keep_prepared(ggml_metal_frame_pool_t pool) { GGML_UNUSED(pool); }

#endif
