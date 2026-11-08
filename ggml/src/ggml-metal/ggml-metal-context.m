#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

// terminal state of one segment's command buffer; desc is empty unless an error was described
struct ggml_metal_seg_result {
    int    status;
    char   desc[256];
};

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

// boundary-scheduled compute: gate wait warning slice and default abort limit
#define GGML_METAL_PAGED_GATE_SLICE_MS 10000
#define GGML_METAL_PAGED_GATE_ABORT_MS_DEFAULT 300000

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy;  // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;

    // stop the graph reorder at boundary marker nodes (see ggml_metal_set_reorder_barriers)
    bool use_reorder_barriers;

    // node-name prefixes the graph reorder treats as barriers (see ggml_metal_set_reorder_barrier_prefixes)
    // owned copies, replaced on each set
    int     reorder_n_prefixes;
    char ** reorder_prefixes;

    int debug_graph;

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers

    // first node of each extra command buffer, then the end of the graph
    int cb_start[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // terminal state of each segment, captured in its completion handler: a failure abandons the
    // rest of the queue and abandoned buffers carry no error, so reading back later would find
    // the first buffer that did not complete rather than the one that failed
    struct ggml_metal_seg_result * seg_results;
    int                            seg_results_cap;
    int                            seg_results_n;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
    // description of the error that set has_error (a copy, the command buffers are released)
    char last_error[256];

    // total time a segment may wait for its gate before the graph fails (0 = no limit)
    // see ggml_backend_metal_set_paged_gate_abort_ms
    uint32_t paged_gate_abort_ms;

    // boundary-event schedule (see ggml_metal_set_boundary_schedule), owned copies
    // nodes are matched by pointer against each computed graph and the schedule stays in force
    // across computes until cleared; with bsched_n_cuts == 0 the stock n_cb path runs
    int                    bsched_n_cuts;
    struct ggml_tensor  ** bsched_cut_nodes;  // node ptrs; segment ends (signal after)
    ggml_metal_event_t  *  bsched_sig_ev;     // per cut; NULL entry => no signal there
    uint64_t *             bsched_sig_val;
    int                    bsched_n_waits;
    struct ggml_tensor  ** bsched_wait_nodes; // node ptrs; segment starts (wait before)
    ggml_metal_event_t  *  bsched_wait_ev;
    uint64_t *             bsched_wait_val;
    // signaled on completion of an empty command buffer committed ahead of the first segment
    ggml_metal_event_t     bsched_entry_ev;
    uint64_t               bsched_entry_val;

    // encode window (see ggml_metal_set_encode_window): only segments intersecting
    // [first_node, last_node] are encoded, with ingress copies before and egress copies after
    // copy arrays are owned copies of byte ranges; the tensors remain caller-owned
    //
    // ewin_out_of_band is a guard, not a selector: a split that would encode one of these
    // full-graph nodes is rejected before any command buffer exists
    bool                                  ewin_active;
    struct ggml_tensor                 *  ewin_first_node;
    struct ggml_tensor                 *  ewin_last_node;
    size_t                                ewin_n_ingress;
    struct ggml_metal_tensor_range_copy * ewin_ingress;
    size_t                                ewin_n_egress;
    struct ggml_metal_tensor_range_copy * ewin_egress;
    size_t                                ewin_n_out_of_band;
    struct ggml_tensor                 ** ewin_out_of_band;

    // host cuts (see ggml_metal_set_host_cuts): owned copy of the node array, caller-owned event
    // the next host cut encoded signals hcut_val + 1
    int                    hcut_n;
    struct ggml_tensor  ** hcut_nodes;
    ggml_metal_event_t     hcut_ev;
    uint64_t               hcut_val;
    bool                (* hcut_fn)(void * user_data, int i, struct ggml_tensor * node);
    void                 * hcut_user_data;
};

static void             ggml_metal_bsched_clear        (ggml_metal_t ctx);
static void             ggml_metal_hcut_clear          (ggml_metal_t ctx);
static enum ggml_status ggml_metal_graph_compute_paged (ggml_metal_t ctx, struct ggml_cgraph * gf);

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
        if (queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

        {
            const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        res->use_graph_optimize = true;

        if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        res->use_reorder_barriers = false;
        res->reorder_n_prefixes   = 0;
        res->reorder_prefixes     = NULL;

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;
        res->last_error[0] = '\0';
        res->paged_gate_abort_ms = GGML_METAL_PAGED_GATE_ABORT_MS_DEFAULT;

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->buf_refs     = [[NSMutableArray alloc] init];

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    free(ctx->seg_results);
    ctx->seg_results = NULL;
    ctx->seg_results_cap = 0;
    ctx->seg_results_n = 0;

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    ggml_metal_bsched_clear(ctx);
    ggml_metal_clear_encode_window(ctx);
    ggml_metal_hcut_clear(ctx);
    ggml_metal_set_reorder_barrier_prefixes(ctx, 0, NULL);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

ggml_metal_device_t ggml_metal_get_device(ggml_metal_t ctx) {
    return ctx->dev;
}

// log the first command buffer, from index `from` on, that carries an error
// a fault abandons the buffers queued behind it, so the first one not completed is not
// necessarily the one that failed
static void ggml_metal_log_cmd_buf_error(id<MTLCommandBuffer> (^at)(size_t), size_t n, size_t from,
        char * out, size_t out_size) {
    for (size_t j = from; j < n; ++j) {
        id<MTLCommandBuffer> cmd_buf = at(j);
        if (!cmd_buf || [cmd_buf status] != MTLCommandBufferStatusError) {
            continue;
        }
        const char * desc = [[cmd_buf error].localizedDescription UTF8String];
        GGML_LOG_ERROR("%s: command buffer %d is the first carrying an error: %s\n", __func__,
                (int) j, desc);
        snprintf(out, out_size, "command buffer %d: %s", (int) j, desc);
        return;
    }
    GGML_LOG_ERROR("%s: no command buffer from %d on carries an error -- the queue was abandoned "
            "without one being recorded\n", __func__, (int) from);
    snprintf(out, out_size, "the queue was abandoned from command buffer %d on with no error "
            "recorded on any of them", (int) from);
}

const char * ggml_metal_last_error(ggml_metal_t ctx) {
    return ctx->last_error[0] != '\0' ? ctx->last_error : NULL;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ggml_metal_log_cmd_buf_error(^id<MTLCommandBuffer>(size_t j) {
                    return ctx->cmd_bufs[j].obj;
                }, (size_t) (n_cb + 1), (size_t) cb_idx, ctx->last_error, sizeof(ctx->last_error));
                ctx->has_error = true;
                return;
            }
        }
    }

    // report from what the segments recorded at completion (see seg_results)
    for (int i = 0; i < ctx->seg_results_n; ++i) {
        if (ctx->seg_results[i].status == (int) MTLCommandBufferStatusCompleted) {
            continue;
        }
        GGML_LOG_ERROR("%s: paged segment %d finished with status %d\n", __func__, i,
                ctx->seg_results[i].status);
        for (int j = i; j < ctx->seg_results_n; ++j) {
            if (ctx->seg_results[j].desc[0] == '\0') {
                continue;
            }
            GGML_LOG_ERROR("%s: paged segment %d is the first carrying an error: %s\n", __func__, j,
                    ctx->seg_results[j].desc);
            snprintf(ctx->last_error, sizeof(ctx->last_error), "paged segment %d: %s", j,
                    ctx->seg_results[j].desc);
            break;
        }
        break;
    }
    ctx->seg_results_n = 0;

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                ggml_metal_log_cmd_buf_error(^id<MTLCommandBuffer>(size_t j) {
                    return ctx->cmd_bufs_ext[j];
                }, (size_t) ctx->cmd_bufs_ext.count, i, ctx->last_error, sizeof(ctx->last_error));

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_dst];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ggml_metal_device_get_queue(ctx_dst->dev);
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

//
// boundary-event schedule
//

static void ggml_metal_bsched_clear(ggml_metal_t ctx) {
    free(ctx->bsched_cut_nodes);
    free(ctx->bsched_sig_ev);
    free(ctx->bsched_sig_val);
    free(ctx->bsched_wait_nodes);
    free(ctx->bsched_wait_ev);
    free(ctx->bsched_wait_val);
    ctx->bsched_cut_nodes  = NULL;
    ctx->bsched_sig_ev     = NULL;
    ctx->bsched_sig_val    = NULL;
    ctx->bsched_wait_nodes = NULL;
    ctx->bsched_wait_ev    = NULL;
    ctx->bsched_wait_val   = NULL;
    ctx->bsched_n_cuts     = 0;
    ctx->bsched_n_waits    = 0;
    ctx->bsched_entry_ev   = NULL;
    ctx->bsched_entry_val  = 0;
}

void ggml_metal_set_paged_gate_abort_ms(ggml_metal_t ctx, uint32_t ms) {
    ctx->paged_gate_abort_ms = ms;
}

void ggml_metal_set_boundary_entry_signal(ggml_metal_t ctx, ggml_metal_event_t ev, uint64_t value) {
    ctx->bsched_entry_ev  = ev;
    ctx->bsched_entry_val = ev != NULL ? value : 0;
}

void ggml_metal_set_boundary_schedule(
        ggml_metal_t ctx,
        int n_cuts,  struct ggml_tensor * const * cut_nodes,  ggml_metal_event_t * sig_ev, const uint64_t * sig_val,
        int n_waits, struct ggml_tensor * const * wait_nodes, ggml_metal_event_t * wait_ev, const uint64_t * wait_val) {
    ggml_metal_bsched_clear(ctx);
    if (n_cuts <= 0) {
        return; // cleared, computes take the stock n_cb path
    }

    ctx->bsched_n_cuts    = n_cuts;
    ctx->bsched_cut_nodes = malloc(sizeof(struct ggml_tensor *) * n_cuts);
    ctx->bsched_sig_ev    = malloc(sizeof(ggml_metal_event_t) * n_cuts);
    ctx->bsched_sig_val   = malloc(sizeof(uint64_t) * n_cuts);
    memcpy(ctx->bsched_cut_nodes, cut_nodes, sizeof(struct ggml_tensor *) * n_cuts);
    memcpy(ctx->bsched_sig_ev,    sig_ev,    sizeof(ggml_metal_event_t) * n_cuts);
    memcpy(ctx->bsched_sig_val,   sig_val,   sizeof(uint64_t) * n_cuts);

    if (n_waits > 0) {
        ctx->bsched_n_waits    = n_waits;
        ctx->bsched_wait_nodes = malloc(sizeof(struct ggml_tensor *) * n_waits);
        ctx->bsched_wait_ev    = malloc(sizeof(ggml_metal_event_t) * n_waits);
        ctx->bsched_wait_val   = malloc(sizeof(uint64_t) * n_waits);
        memcpy(ctx->bsched_wait_nodes, wait_nodes, sizeof(struct ggml_tensor *) * n_waits);
        memcpy(ctx->bsched_wait_ev,    wait_ev,    sizeof(ggml_metal_event_t) * n_waits);
        memcpy(ctx->bsched_wait_val,   wait_val,   sizeof(uint64_t) * n_waits);
    }
}

static void ggml_metal_hcut_clear(ggml_metal_t ctx) {
    free(ctx->hcut_nodes);
    ctx->hcut_n         = 0;
    ctx->hcut_nodes     = NULL;
    ctx->hcut_ev        = NULL;
    ctx->hcut_val       = 0;
    ctx->hcut_fn        = NULL;
    ctx->hcut_user_data = NULL;
}

void ggml_metal_set_host_cuts(
        ggml_metal_t ctx,
        int n, struct ggml_tensor * const * nodes,
        ggml_metal_event_t ev, uint64_t base_value,
        bool (*fn)(void * user_data, int i, struct ggml_tensor * node), void * user_data) {
    ggml_metal_hcut_clear(ctx);
    if (n <= 0) {
        return;
    }
    GGML_ASSERT(nodes != NULL && ev != NULL && fn != NULL);

    ctx->hcut_nodes = malloc(sizeof(struct ggml_tensor *) * n);
    GGML_ASSERT(ctx->hcut_nodes != NULL);
    memcpy(ctx->hcut_nodes, nodes, sizeof(struct ggml_tensor *) * n);
    ctx->hcut_n         = n;
    ctx->hcut_ev        = ev;
    ctx->hcut_val       = base_value;
    ctx->hcut_fn        = fn;
    ctx->hcut_user_data = user_data;
}

// owned whole-tensor ranges for a list of pairs: [0, ggml_nbytes(src)) on both sides
static struct ggml_metal_tensor_range_copy * ggml_metal_ewin_whole_ranges(size_t n, const struct ggml_metal_tensor_copy_pair * pairs) {
    if (n == 0) {
        return NULL;
    }
    struct ggml_metal_tensor_range_copy * res = malloc(sizeof(*res) * n);
    GGML_ASSERT(res != NULL);
    for (size_t i = 0; i < n; ++i) {
        res[i] = (struct ggml_metal_tensor_range_copy) {
            /*.src      =*/ pairs[i].src,
            /*.dst      =*/ pairs[i].dst,
            /*.src_offs =*/ 0,
            /*.dst_offs =*/ 0,
            /*.size     =*/ ggml_nbytes(pairs[i].src),
        };
    }
    return res;
}

// replaces the installed window; takes ownership of the validated ingress and egress, copies out_of_band
static void ggml_metal_ewin_install(
        ggml_metal_t ctx,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     struct ggml_metal_tensor_range_copy * ingress,
        size_t n_egress,      struct ggml_metal_tensor_range_copy * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band) {
    struct ggml_tensor ** out_of_band_copy = NULL;
    if (n_out_of_band > 0) {
        out_of_band_copy = malloc(sizeof(*out_of_band_copy) * n_out_of_band);
        GGML_ASSERT(out_of_band_copy != NULL);
        memcpy(out_of_band_copy, out_of_band, sizeof(*out_of_band_copy) * n_out_of_band);
    }

    ggml_metal_clear_encode_window(ctx);
    ctx->ewin_active        = true;
    ctx->ewin_first_node    = first_node;
    ctx->ewin_last_node     = last_node;
    ctx->ewin_n_ingress     = n_ingress;
    ctx->ewin_ingress       = ingress;
    ctx->ewin_n_egress      = n_egress;
    ctx->ewin_egress        = egress;
    ctx->ewin_n_out_of_band = n_out_of_band;
    ctx->ewin_out_of_band   = out_of_band_copy;
}

// whole-tensor form of the encode window (see ggml_backend_metal_set_encode_window)
// first_node is the exclusive lower endpoint and last_node the inclusive upper one, either may
// be NULL for an open edge. endpoints are matched by pointer and projected per split: a split
// holding neither endpoint is encoded in full
// every pair must have equal byte sizes, ggml_metal_set_encode_window_ranged is the byte-range form
// the context copies the arrays, the tensors remain caller-owned
void ggml_metal_set_encode_window(
        ggml_metal_t ctx,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     const struct ggml_metal_tensor_copy_pair * ingress,
        size_t n_egress,      const struct ggml_metal_tensor_copy_pair * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band) {
    GGML_ASSERT(n_ingress     == 0 || ingress     != NULL);
    GGML_ASSERT(n_egress      == 0 || egress      != NULL);
    GGML_ASSERT(n_out_of_band == 0 || out_of_band != NULL);
    GGML_ASSERT(n_ingress     <= SIZE_MAX / sizeof(struct ggml_metal_tensor_range_copy));
    GGML_ASSERT(n_egress      <= SIZE_MAX / sizeof(struct ggml_metal_tensor_range_copy));
    GGML_ASSERT(n_out_of_band <= SIZE_MAX / sizeof(*out_of_band));
    for (size_t i = 0; i < n_ingress; ++i) {
        GGML_ASSERT(ingress[i].src != NULL);
        GGML_ASSERT(ingress[i].dst != NULL);
        GGML_ASSERT(ggml_nbytes(ingress[i].src) == ggml_nbytes(ingress[i].dst));
    }
    for (size_t i = 0; i < n_egress; ++i) {
        GGML_ASSERT(egress[i].src != NULL);
        GGML_ASSERT(egress[i].dst != NULL);
        GGML_ASSERT(ggml_nbytes(egress[i].src) == ggml_nbytes(egress[i].dst));
    }
    for (size_t i = 0; i < n_out_of_band; ++i) {
        // a NULL entry would silently match nothing and weaken the guard
        GGML_ASSERT(out_of_band[i] != NULL);
    }

    struct ggml_metal_tensor_range_copy * ingress_copy = ggml_metal_ewin_whole_ranges(n_ingress, ingress);
    struct ggml_metal_tensor_range_copy * egress_copy  = ggml_metal_ewin_whole_ranges(n_egress,  egress);

    ggml_metal_ewin_install(ctx, first_node, last_node, n_ingress, ingress_copy, n_egress, egress_copy, n_out_of_band, out_of_band);
}

// the tensor whose allocation backs t, and the byte offset of t's data within it
static const struct ggml_tensor * ggml_metal_ewin_storage(const struct ggml_tensor * t, size_t * offs) {
    if (t->view_src != NULL) {
        *offs = t->view_offs;
        return t->view_src;
    }
    *offs = 0;
    return t;
}

// checks one byte-range copy against the rules of ggml_backend_metal_set_encode_window_ranged
static bool ggml_metal_ewin_range_is_valid(const char * list, size_t i, const struct ggml_metal_tensor_range_copy * r) {
    if (r->src == NULL || r->dst == NULL) {
        GGML_LOG_ERROR("%s: %s[%zu]: src and dst must be non-NULL\n", __func__, list, i);
        return false;
    }
    if (r->size == 0) {
        GGML_LOG_ERROR("%s: %s[%zu]: size must be non-zero\n", __func__, list, i);
        return false;
    }

    const size_t src_nbytes = ggml_nbytes(r->src);
    const size_t dst_nbytes = ggml_nbytes(r->dst);
    // two comparisons each so that offs + size cannot wrap
    if (r->src_offs > src_nbytes || r->size > src_nbytes - r->src_offs) {
        GGML_LOG_ERROR("%s: %s[%zu]: source range [%zu, +%zu) exceeds '%s' (%zu bytes)\n",
                __func__, list, i, r->src_offs, r->size, ggml_get_name(r->src), src_nbytes);
        return false;
    }
    if (r->dst_offs > dst_nbytes || r->size > dst_nbytes - r->dst_offs) {
        GGML_LOG_ERROR("%s: %s[%zu]: destination range [%zu, +%zu) exceeds '%s' (%zu bytes)\n",
                __func__, list, i, r->dst_offs, r->size, ggml_get_name(r->dst), dst_nbytes);
        return false;
    }

    // MTLBlitCommandEncoder copyFromBuffer requires the offsets and size to be multiples of 4 on macOS
    if (r->src_offs % 4 != 0 || r->dst_offs % 4 != 0 || r->size % 4 != 0) {
        GGML_LOG_ERROR("%s: %s[%zu]: src_offs (%zu), dst_offs (%zu) and size (%zu) must be multiples of 4\n",
                __func__, list, i, r->src_offs, r->dst_offs, r->size);
        return false;
    }

    // a copy between overlapping regions of one buffer is undefined in Metal
    size_t src_base = 0;
    size_t dst_base = 0;
    const struct ggml_tensor * src_root = ggml_metal_ewin_storage(r->src, &src_base);
    const struct ggml_tensor * dst_root = ggml_metal_ewin_storage(r->dst, &dst_base);
    if (src_root == dst_root) {
        const size_t s0 = src_base + r->src_offs;
        const size_t d0 = dst_base + r->dst_offs;
        if (s0 < d0 + r->size && d0 < s0 + r->size) {
            GGML_LOG_ERROR("%s: %s[%zu]: source and destination ranges overlap in the storage of '%s'\n",
                    __func__, list, i, ggml_get_name(src_root));
            return false;
        }
    }

    return true;
}

bool ggml_metal_set_encode_window_ranged(
        ggml_metal_t ctx,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     const struct ggml_metal_tensor_range_copy * ingress,
        size_t n_egress,      const struct ggml_metal_tensor_range_copy * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band) {
    if ((n_ingress > 0 && ingress == NULL) || (n_egress > 0 && egress == NULL) || (n_out_of_band > 0 && out_of_band == NULL)) {
        GGML_LOG_ERROR("%s: a list with a non-zero count must be non-NULL\n", __func__);
        return false;
    }
    if (n_ingress     > SIZE_MAX / sizeof(*ingress) ||
        n_egress      > SIZE_MAX / sizeof(*egress)  ||
        n_out_of_band > SIZE_MAX / sizeof(*out_of_band)) {
        GGML_LOG_ERROR("%s: list too long\n", __func__);
        return false;
    }
    for (size_t i = 0; i < n_ingress; ++i) {
        if (!ggml_metal_ewin_range_is_valid("ingress", i, &ingress[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < n_egress; ++i) {
        if (!ggml_metal_ewin_range_is_valid("egress", i, &egress[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < n_out_of_band; ++i) {
        if (out_of_band[i] == NULL) {
            GGML_LOG_ERROR("%s: out_of_band[%zu] is NULL\n", __func__, i);
            return false;
        }
    }

    struct ggml_metal_tensor_range_copy * ingress_copy = NULL;
    struct ggml_metal_tensor_range_copy * egress_copy  = NULL;
    if (n_ingress > 0) {
        ingress_copy = malloc(sizeof(*ingress_copy) * n_ingress);
        GGML_ASSERT(ingress_copy != NULL);
        memcpy(ingress_copy, ingress, sizeof(*ingress_copy) * n_ingress);
    }
    if (n_egress > 0) {
        egress_copy = malloc(sizeof(*egress_copy) * n_egress);
        GGML_ASSERT(egress_copy != NULL);
        memcpy(egress_copy, egress, sizeof(*egress_copy) * n_egress);
    }

    ggml_metal_ewin_install(ctx, first_node, last_node, n_ingress, ingress_copy, n_egress, egress_copy, n_out_of_band, out_of_band);

    return true;
}

void ggml_metal_clear_encode_window(ggml_metal_t ctx) {
    free(ctx->ewin_ingress);
    free(ctx->ewin_egress);
    free(ctx->ewin_out_of_band);
    ctx->ewin_active        = false;
    ctx->ewin_first_node    = NULL;
    ctx->ewin_last_node     = NULL;
    ctx->ewin_n_ingress     = 0;
    ctx->ewin_ingress       = NULL;
    ctx->ewin_n_egress      = 0;
    ctx->ewin_egress        = NULL;
    ctx->ewin_n_out_of_band = 0;
    ctx->ewin_out_of_band   = NULL;
}

// one byte-range device-side copy, encoded into cmd_buf
//
// each copy gets its own blit encoder: commands within one blit pass are unordered, while passes
// touching the same tracked resource are ordered, which makes a list of copies run in list order
static void ggml_metal_ewin_blit(id<MTLCommandBuffer> cmd_buf, const struct ggml_metal_tensor_range_copy * r) {
    struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(r->src);
    struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(r->dst);
    GGML_ASSERT(bid_src.metal != nil);
    GGML_ASSERT(bid_dst.metal != nil);

    const size_t src_offs = bid_src.offs + r->src_offs;
    const size_t dst_offs = bid_dst.offs + r->dst_offs;

    // validation rejects overlap between views of one tensor, this catches distinct tensors
    // that resolve to overlapping bytes of one device buffer
    GGML_ASSERT(bid_src.metal != bid_dst.metal || src_offs + r->size <= dst_offs || dst_offs + r->size <= src_offs);

    id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
    [encoder copyFromBuffer:bid_src.metal
               sourceOffset:src_offs
                   toBuffer:bid_dst.metal
          destinationOffset:dst_offs
                       size:r->size];
    [encoder endEncoding];
}

// stop the graph reorder at boundary marker nodes, see ggml_backend_metal_set_reorder_barriers
void ggml_metal_set_reorder_barriers(ggml_metal_t ctx, bool enable) {
    ctx->use_reorder_barriers = enable;
}

// replace the barrier name prefixes of the graph reorder, see ggml_backend_metal_set_reorder_barrier_prefixes
void ggml_metal_set_reorder_barrier_prefixes(ggml_metal_t ctx, int n, const char * const * prefixes) {
    for (int i = 0; i < ctx->reorder_n_prefixes; ++i) {
        free(ctx->reorder_prefixes[i]);
    }
    free(ctx->reorder_prefixes);
    ctx->reorder_n_prefixes = 0;
    ctx->reorder_prefixes   = NULL;

    if (n <= 0) {
        return;
    }
    GGML_ASSERT(prefixes != NULL);

    ctx->reorder_prefixes = malloc(sizeof(char *) * n);
    GGML_ASSERT(ctx->reorder_prefixes != NULL);
    for (int i = 0; i < n; ++i) {
        // an empty prefix would match every node and turn the reorder off
        GGML_ASSERT(prefixes[i] != NULL && prefixes[i][0] != '\0');
        ctx->reorder_prefixes[i] = strdup(prefixes[i]);
        GGML_ASSERT(ctx->reorder_prefixes[i] != NULL);
    }
    ctx->reorder_n_prefixes = n;
}

// does this split avoid encoding the nodes the caller named in out_of_band?
//
// a split holding no window endpoint is encoded in full, which cannot be told apart from a
// split that lies entirely outside the window, so the caller names the nodes that must not be
// encoded and this walks the split with the same skip rule as the encode loop
//
// returns true when clean, otherwise *out_bad_node_idx is the offending node's index in gf
static bool ggml_metal_ewin_split_is_in_band(
        ggml_metal_t         ctx,
        struct ggml_cgraph * gf,
        const int *          starts,
        int                  n_starts,
        int                  window_lo,
        int                  window_hi,
        bool                 window_applies,
        int *                out_bad_node_idx) {
    if (!ctx->ewin_active || ctx->ewin_n_out_of_band == 0) {
        return true;
    }

    const int n_nodes = gf->n_nodes;

    for (int seg = 0; seg < n_starts; ++seg) {
        const int seg_start = starts[seg];
        const int seg_end   = (seg + 1 < n_starts) ? starts[seg + 1] : n_nodes;
        if (seg_end <= seg_start) {
            continue;
        }
        // must mirror the encode loop's skip rule
        if (window_applies && (seg_end <= window_lo || seg_start > window_hi)) {
            continue;
        }
        for (int i = seg_start; i < seg_end; ++i) {
            struct ggml_tensor * node = gf->nodes[i];
            for (size_t k = 0; k < ctx->ewin_n_out_of_band; ++k) {
                if (ctx->ewin_out_of_band[k] == node) {
                    *out_bad_node_idx = i;
                    return false;
                }
            }
        }
    }

    return true;
}

enum ggml_metal_paged_wait_status {
    GGML_METAL_PAGED_WAIT_OK,
    GGML_METAL_PAGED_WAIT_TIMED_OUT,
    GGML_METAL_PAGED_WAIT_ABANDONED,
};

// host wait for ev to reach value, in slices of GGML_METAL_PAGED_GATE_SLICE_MS with a warning
// after each, until paged_gate_abort_ms (0: no limit) has passed
// a saturated event (UINT64_MAX while value is lower) means the producer abandoned the wait
static enum ggml_metal_paged_wait_status ggml_metal_paged_host_wait(
        ggml_metal_t ctx, ggml_metal_event_t ev, uint64_t value, bool spin,
        const char * node_name, const char * what) {
    uint64_t waited_ms = 0;
    while (true) {
        uint64_t slice_ms = GGML_METAL_PAGED_GATE_SLICE_MS;
        if (ctx->paged_gate_abort_ms != 0 && ctx->paged_gate_abort_ms - waited_ms < slice_ms) {
            slice_ms = ctx->paged_gate_abort_ms - waited_ms;
        }
        const bool reached = spin ? ggml_metal_event_host_spin(ev, value, slice_ms)
                                  : ggml_metal_event_host_wait(ev, value, slice_ms);
        if (reached) {
            const bool abandoned = value != UINT64_MAX && ggml_metal_event_host_value(ev) == UINT64_MAX;
            return abandoned ? GGML_METAL_PAGED_WAIT_ABANDONED : GGML_METAL_PAGED_WAIT_OK;
        }
        waited_ms += slice_ms;
        if (ctx->paged_gate_abort_ms != 0 && waited_ms >= ctx->paged_gate_abort_ms) {
            return GGML_METAL_PAGED_WAIT_TIMED_OUT;
        }
        GGML_LOG_WARN("%s: paged segment at node '%s' still waiting for %s after %llu s (event value %llu)\n",
                __func__, node_name, what, (unsigned long long) (waited_ms / 1000), (unsigned long long) value);
    }
}

// a host cut whose segment has been committed and whose callback has not yet run
struct ggml_metal_hcut_pending {
    bool                 active;
    int                  i;     // index in the host cut list
    struct ggml_tensor * node;
    uint64_t             value; // signaled by the segment's command buffer
};

// spins until the pending host cut's segment has completed, then runs the callback
// returns false, with the backend in its error state, on timeout, a saturated event or a failing callback
static bool ggml_metal_hcut_resolve(ggml_metal_t ctx, struct ggml_metal_hcut_pending * p) {
    if (!p->active) {
        return true;
    }
    p->active = false;

    const char * name = ggml_get_name(p->node);
    switch (ggml_metal_paged_host_wait(ctx, ctx->hcut_ev, p->value, true, name, "its host cut")) {
        case GGML_METAL_PAGED_WAIT_OK:
            if (ctx->hcut_fn(ctx->hcut_user_data, p->i, p->node)) {
                return true;
            }
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                    "host cut %d at node '%s': the callback failed the graph", p->i, name);
            break;
        case GGML_METAL_PAGED_WAIT_TIMED_OUT:
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                    "host cut %d at node '%s': its segment did not complete within %u ms", p->i, name,
                    ctx->paged_gate_abort_ms);
            break;
        case GGML_METAL_PAGED_WAIT_ABANDONED:
            snprintf(ctx->last_error, sizeof(ctx->last_error),
                    "host cut %d at node '%s': its event was abandoned (event value saturated)", p->i, name);
            break;
    }
    GGML_LOG_ERROR("%s: %s\n", __func__, ctx->last_error);
    ctx->has_error = true;
    return false;
}

// segments are encoded and committed one at a time on the calling thread (no dispatch_apply), so
// the submit order is deterministic and the event signals stay monotonic across command buffers
// failures surface at the next ggml_metal_synchronize, as in the stock path
static enum ggml_status ggml_metal_graph_compute_paged(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    const int n_nodes = gf->n_nodes;

    // the schedule is not cleared here, cuts and waits are matched by pointer against this graph
    // (which may be one split of several)
    const int                    n_cuts     = ctx->bsched_n_cuts;
    struct ggml_tensor ** const  cut_nodes  = ctx->bsched_cut_nodes;
    ggml_metal_event_t * const   sig_ev     = ctx->bsched_sig_ev;
    const uint64_t     * const   sig_val    = ctx->bsched_sig_val;
    const int                    n_waits    = ctx->bsched_n_waits;
    struct ggml_tensor ** const  wait_nodes = ctx->bsched_wait_nodes;
    ggml_metal_event_t * const   wait_ev    = ctx->bsched_wait_ev;
    const uint64_t     * const   wait_val   = ctx->bsched_wait_val;

    @autoreleasepool {
        ctx->gf = gf;

        // keep the memory wired (mirrors the stock graph_compute)
        ggml_metal_device_rsets_keep_alive(ctx->dev);

        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // per-node flags: cut ends and wait starts in this graph; hcut_idx is the index in the host cut list or -1
        bool * is_cut   = (bool *) calloc((size_t) n_nodes, sizeof(bool));
        bool * is_wait  = (bool *) calloc((size_t) n_nodes, sizeof(bool));
        int  * hcut_idx = (int  *) malloc(sizeof(int) * (size_t) n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            struct ggml_tensor * node = gf->nodes[i];
            for (int c = 0; c < n_cuts; ++c) {
                if (cut_nodes[c] == node) { is_cut[i] = true; break; }
            }
            for (int w = 0; w < n_waits; ++w) {
                if (wait_nodes[w] == node) { is_wait[i] = true; break; }
            }
            hcut_idx[i] = -1;
            for (int h = 0; h < ctx->hcut_n; ++h) {
                if (ctx->hcut_nodes[h] == node) { hcut_idx[i] = h; break; }
            }
        }

        // sorted unique segment start indices: 0, the node after every cut and host cut, every wait node
        const int cap = 3 * n_nodes + 2;
        int * starts = (int *) malloc(sizeof(int) * (size_t) cap);
        int n_starts = 0;
        starts[n_starts++] = 0;
        for (int i = 0; i < n_nodes; ++i) {
            if (is_cut[i] && i + 1 < n_nodes) { starts[n_starts++] = i + 1; }
            if (hcut_idx[i] >= 0 && i + 1 < n_nodes) { starts[n_starts++] = i + 1; }
            if (is_wait[i] && i > 0)          { starts[n_starts++] = i;     }
        }
        // insertion sort (n_starts is small) + dedup in place
        for (int a = 1; a < n_starts; ++a) {
            const int key = starts[a];
            int b = a - 1;
            while (b >= 0 && starts[b] > key) { starts[b + 1] = starts[b]; --b; }
            starts[b + 1] = key;
        }
        int m = 0;
        for (int a = 0; a < n_starts; ++a) {
            if (a == 0 || starts[a] != starts[a - 1]) { starts[m++] = starts[a]; }
        }
        n_starts = m;

        // resolve the encode window against this graph; window_lo is the first node in the window
        // and the ingress/egress copies fire only where their endpoint matched
        int  window_lo      = 0;
        int  window_hi      = n_nodes - 1;
        bool window_applies = false;
        bool blit_in_here   = false;
        bool blit_out_here  = false;
        if (ctx->ewin_active) {
            int idx_first = -1;
            int idx_last  = -1;
            for (int i = 0; i < n_nodes; ++i) {
                if (gf->nodes[i] == ctx->ewin_first_node) { idx_first = i; }
                if (gf->nodes[i] == ctx->ewin_last_node)  { idx_last  = i; }
            }
            const bool first_open  = ctx->ewin_first_node == NULL;
            const bool last_open   = ctx->ewin_last_node  == NULL;
            const bool first_found = idx_first >= 0;
            const bool last_found  = idx_last  >= 0;
            if (first_found || last_found || first_open || last_open) {
                if (first_found && last_found) {
                    GGML_ASSERT(idx_last > idx_first);
                }
                window_applies = !(first_open && last_open) &&
                                 (first_found || last_found);
                window_lo     = first_found ? idx_first + 1 : 0;
                window_hi     = last_found  ? idx_last      : n_nodes - 1;
                blit_in_here  = first_found;
                blit_out_here = last_found;
            }
        }
        // reject before anything is submitted so the failure is recoverable: no command buffer
        // was enqueued and no event signaled, hence a status and not a GGML_ASSERT
        {
            int bad_node_idx = -1;
            if (!ggml_metal_ewin_split_is_in_band(ctx, gf, starts, n_starts, window_lo, window_hi, window_applies, &bad_node_idx)) {
                struct ggml_tensor * bad_node = gf->nodes[bad_node_idx];
                GGML_LOG_ERROR("%s: encode window violation - split would encode out-of-band node '%s' "
                               "(node %d of %d in this split; window applies=%s lo=%d hi=%d)\n",
                        __func__, ggml_get_name(bad_node), bad_node_idx, n_nodes,
                        window_applies ? "true" : "false", window_lo, window_hi);

                free(starts);
                free(is_cut);
                free(is_wait);
                free(hcut_idx);

                return GGML_STATUS_FAILED;
            }
        }

        bool window_entered = false;

        // the host cut whose callback has yet to run; it runs before the next segment's entry
        // signal or gate wait (which it may open), or after that segment is encoded and before
        // it is enqueued
        struct ggml_metal_hcut_pending hcut_pending = { false, -1, NULL, 0 };

        // grown only while no handler is outstanding: handlers hold pointers into this array and
        // run on a Metal thread, so a later graph that would overflow captures nothing instead
        // ggml_metal_synchronize resets the count after waiting on the last buffer
        if (ctx->seg_results_n == 0 && ctx->seg_results_cap < n_starts) {
            free(ctx->seg_results);
            ctx->seg_results = (struct ggml_metal_seg_result *) malloc(
                    sizeof(struct ggml_metal_seg_result) * (size_t) n_starts);
            GGML_ASSERT(ctx->seg_results != NULL);
            ctx->seg_results_cap = n_starts;
        }

        for (int seg = 0; seg < n_starts; ++seg) {
            const int seg_start = starts[seg];
            const int seg_end   = (seg + 1 < n_starts) ? starts[seg + 1] : n_nodes;
            if (seg_end <= seg_start) {
                continue;
            }
            // outside the encode window: no command buffer is created, so neither its ops nor
            // its schedule signals fire
            if (window_applies && (seg_end <= window_lo || seg_start > window_hi)) {
                continue;
            }
            const bool window_first = window_applies && !window_entered;
            const bool window_last  = window_applies && seg_start <= window_hi && seg_end > window_hi;
            window_entered = true;

            const bool entry_signal = window_first && blit_in_here && ctx->bsched_entry_ev != NULL;

            // Pre-encode the segment behind a pending host cut when nothing has to wait on the
            // host before encoding it. It is enqueued only after the callback, so the callback
            // may commit work to this queue and wait for it.
            const bool pre_encode = hcut_pending.active && !entry_signal && !is_wait[seg_start];
            if (hcut_pending.active && !pre_encode && !ggml_metal_hcut_resolve(ctx, &hcut_pending)) {
                free(starts);
                free(is_cut);
                free(is_wait);
                free(hcut_idx);

                return GGML_STATUS_FAILED;
            }

            // window entry signal: its own buffer, committed before the first segment so that it
            // depends on none of the window's waits, nor on the host wait below
            if (entry_signal) {
                id<MTLCommandBuffer> entry_buf = [queue commandBufferWithUnretainedReferences];
                [entry_buf retain];
                [entry_buf enqueue];
                ggml_metal_event_signal_on_complete(ctx->bsched_entry_ev, (ggml_metal_cmd_buf_t) entry_buf, ctx->bsched_entry_val);
                [entry_buf commit];
                [ctx->cmd_bufs_ext addObject:entry_buf];
                ctx->cmd_buf_last = entry_buf;
            }

            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];
            if (!pre_encode) {
                [cmd_buf enqueue]; // reserve the queue slot so execution order matches creation order
            }

            // waits before this segment's ops; several waits can target the same node
            //
            // the host waits for the gate first and the GPU-side waits below remain the ordering
            // guarantee: a committed buffer parked on an event holds a queue slot and is timed
            // by the GPU, so it is committed only once its gate is satisfied
            //
            // on timeout, or when an event is saturated to UINT64_MAX (the producer abandoned the
            // schedule), the graph fails with this segment unencoded: its enqueued slot is
            // committed empty so the queue moves on, and the backend enters its error state
            bool gate_timed_out = false;
            bool gate_abandoned = false;
            if (is_wait[seg_start]) {
                struct ggml_tensor * wnode = gf->nodes[seg_start];
                for (int w = 0; w < n_waits && !gate_timed_out && !gate_abandoned; ++w) {
                    if (wait_nodes[w] == wnode && wait_ev[w] != NULL) {
                        const enum ggml_metal_paged_wait_status st =
                                ggml_metal_paged_host_wait(ctx, wait_ev[w], wait_val[w], false, ggml_get_name(wnode), "its gate");
                        gate_timed_out = st == GGML_METAL_PAGED_WAIT_TIMED_OUT;
                        gate_abandoned = st == GGML_METAL_PAGED_WAIT_ABANDONED;
                    }
                }
            }
            if (gate_timed_out || gate_abandoned) {
                if (gate_abandoned) {
                    snprintf(ctx->last_error, sizeof(ctx->last_error),
                            "paged segment at node '%s': its gate was abandoned (event value saturated)",
                            ggml_get_name(gf->nodes[seg_start]));
                } else {
                    snprintf(ctx->last_error, sizeof(ctx->last_error),
                            "paged segment at node '%s': its gate stayed closed for %u ms",
                            ggml_get_name(gf->nodes[seg_start]), ctx->paged_gate_abort_ms);
                }
                GGML_LOG_ERROR("%s: %s\n", __func__, ctx->last_error);
                ctx->has_error = true;

                [cmd_buf commit];
                [ctx->cmd_bufs_ext addObject:cmd_buf];
                ctx->cmd_buf_last = cmd_buf;

                free(starts);
                free(is_cut);
                free(is_wait);
                free(hcut_idx);

                return GGML_STATUS_FAILED;
            }

            if (is_wait[seg_start]) {
                struct ggml_tensor * node = gf->nodes[seg_start];
                for (int w = 0; w < n_waits; ++w) {
                    if (wait_nodes[w] == node && wait_ev[w] != NULL) {
                        ggml_metal_event_encode_wait_value(wait_ev[w], (ggml_metal_cmd_buf_t) cmd_buf, wait_val[w]);
                    }
                }
            }

            // window entry: copy in before the first encoded segment reads it, and after the
            // schedule waits so that the copy does not outrun its gate
            if (window_first && blit_in_here) {
                for (size_t i = 0; i < ctx->ewin_n_ingress; ++i) {
                    ggml_metal_ewin_blit(cmd_buf, &ctx->ewin_ingress[i]);
                }
            }

            // encode nodes [seg_start, seg_end) as the stock encode_async does over its slice
            ggml_metal_op_t op = ggml_metal_op_init(
                ctx->dev,
                (ggml_metal_cmd_buf_t) cmd_buf,
                gf,
                ctx->finfo,
                seg_start,
                seg_end,
                ctx->use_concurrency,
                false, // capture is not supported on the boundary-scheduled path
                ctx->debug_graph);

            for (int idx = 0; idx < ggml_metal_op_n_nodes(op); ++idx) {
                const int r = ggml_metal_op_encode(op, idx);
                if (r == 0) {
                    break;
                }
                idx += r - 1;
            }

            ggml_metal_op_free(op);

            // window exit: copy out after the last encoded segment produced it
            if (window_last && blit_out_here) {
                for (size_t i = 0; i < ctx->ewin_n_egress; ++i) {
                    ggml_metal_ewin_blit(cmd_buf, &ctx->ewin_egress[i]);
                }
            }

            // signals after this segment, which ends at a cut node
            const int seg_last = seg_end - 1;
            if (is_cut[seg_last]) {
                struct ggml_tensor * node = gf->nodes[seg_last];
                for (int c = 0; c < n_cuts; ++c) {
                    if (cut_nodes[c] == node && sig_ev[c] != NULL) {
                        // signal on completion so that a host waiter sees coherent shared-storage bytes
                        ggml_metal_event_signal_on_complete(sig_ev[c], (ggml_metal_cmd_buf_t) cmd_buf, sig_val[c]);
                    }
                }
            }

            // A host cut signals from the device, at the end of the command buffer, rather than
            // from a completion handler: the host spins on the value, and a completion handler
            // adds a Metal thread's wake-up to every cut. The signal follows every encoder of the
            // buffer, so the host reads what the segment wrote once it sees the value.
            struct ggml_metal_hcut_pending hcut_next = { false, -1, NULL, 0 };
            if (hcut_idx[seg_last] >= 0) {
                hcut_next = (struct ggml_metal_hcut_pending) { true, hcut_idx[seg_last], gf->nodes[seg_last], ++ctx->hcut_val };
                ggml_metal_event_encode_signal_value(ctx->hcut_ev, (ggml_metal_cmd_buf_t) cmd_buf, hcut_next.value);
            }

            // The pre-encoded segment's predecessor cut runs its callback now, before this
            // segment is enqueued. On failure the segment is dropped unenqueued.
            if (pre_encode && !ggml_metal_hcut_resolve(ctx, &hcut_pending)) {
                [cmd_buf release];

                free(starts);
                free(is_cut);
                free(is_wait);
                free(hcut_idx);

                return GGML_STATUS_FAILED;
            }
            hcut_pending = hcut_next;

            // capture this segment's outcome on completion (see seg_results); the slot is reserved
            // before commit so the handler only writes its own index
            if (ctx->seg_results_n < ctx->seg_results_cap) {
                struct ggml_metal_seg_result * slot = &ctx->seg_results[ctx->seg_results_n++];
                slot->status = -1;
                slot->desc[0] = '\0';
                [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    slot->status = (int) [cb status];
                    NSError * err = [cb error];
                    if (err != nil) {
                        const char * d = [[err localizedDescription] UTF8String];
                        if (d != NULL) {
                            snprintf(slot->desc, sizeof(slot->desc), "%s", d);
                        }
                    }
                }];
            }

            [cmd_buf commit];

            // ownership passes to cmd_bufs_ext, drained by ggml_metal_synchronize
            [ctx->cmd_bufs_ext addObject:cmd_buf];
            ctx->cmd_buf_last = cmd_buf;
        }

        // A host cut ending the last encoded segment runs its callback before the compute
        // returns, so nothing committed later on this backend can overtake it.
        const bool hcut_ok = ggml_metal_hcut_resolve(ctx, &hcut_pending);

        free(starts);
        free(is_cut);
        free(is_wait);
        free(hcut_idx);

        if (!hcut_ok) {
            return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    // while a boundary schedule or host cut list is set, take the boundary-scheduled path
    if (ctx->bsched_n_cuts > 0 || ctx->hcut_n > 0) {
        return ggml_metal_graph_compute_paged(ctx, gf);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_main nodes are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        // the main thread encodes [0, cb_start[0]) and the extra command buffer i encodes [cb_start[i], cb_start[i + 1])
        // a fusion cannot span two command buffers, so each boundary moves to the end of the fusable group it falls
        // into and the encoders fuse the same nodes as with a single command buffer
        {
            const int n_rest   = gf->n_nodes - MIN(n_main, gf->n_nodes);
            const int n_per_cb = n_cb > 0 ? (n_rest + n_cb - 1) / n_cb : 0;

            int i = 0;
            for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
                const int target = MIN(n_main + cb_idx*n_per_cb, gf->n_nodes);
                while (i < target) {
                    i += ggml_metal_fusion_max(gf, i);
                }
                ctx->cb_start[cb_idx] = MIN(i, gf->n_nodes);
            }
            ctx->cb_start[n_cb] = gf->n_nodes;
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                const bool ok = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error];

                [descriptor release];

                if (!ok) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                    [ctx->capture_scope release];
                    ctx->capture_scope = nil;
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            enum ggml_status res = GGML_STATUS_SUCCESS;

            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    res = GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb && res == GGML_STATUS_SUCCESS; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }
                    res = GGML_STATUS_FAILED;
                    break;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    res = GGML_STATUS_ABORTED;
                    break;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            [ctx->capture_scope release];
            ctx->capture_scope = nil;

            ctx->capture_started = false;

            if (res != GGML_STATUS_SUCCESS) {
                return res;
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf, ctx->use_reorder_barriers,
            ctx->reorder_n_prefixes, (const char * const *) ctx->reorder_prefixes);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_signal(ggml_metal_t ctx, ggml_metal_event_t ev, uint64_t value) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        // on completion, like the boundary schedule's cut signals on the same events
        ggml_metal_event_signal_on_complete(ev, (ggml_metal_cmd_buf_t) cmd_buf, value);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        int idx_start = 0;
        int idx_end   = ctx->cb_start[0];

        if (cb_idx < n_cb_l) {
            idx_start = ctx->cb_start[cb_idx];
            idx_end   = ctx->cb_start[cb_idx + 1];
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute == 0,
            ctx->debug_graph);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
