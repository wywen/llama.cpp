#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;

    // #226 paged-decode boundary-event schedule (PERSISTENT + pointer-keyed;
    // see ggml_metal_set_boundary_schedule). Owned copies, replaced on each set
    // and freed on clear/free. When bsched_n_cuts == 0 the stock n_cb encode
    // path runs unchanged. Nodes are matched by pointer against each computed
    // graph, so the schedule is NOT cleared after a graph_compute -- it stays in
    // force across the splits of one decode (and the reused graph of later
    // tokens) until the caller clears it.
    int                    bsched_n_cuts;
    struct ggml_tensor  ** bsched_cut_nodes;  // node ptrs; segment ends (signal after)
    ggml_metal_event_t  *  bsched_sig_ev;     // per cut; NULL entry => no signal there
    uint64_t *             bsched_sig_val;
    int                    bsched_n_waits;
    struct ggml_tensor  ** bsched_wait_nodes; // node ptrs; segment starts (wait before)
    ggml_metal_event_t  *  bsched_wait_ev;
    uint64_t *             bsched_wait_val;
};

// #226 boundary-event schedule helpers (definitions below).
static void             ggml_metal_bsched_clear        (ggml_metal_t ctx);
static enum ggml_status ggml_metal_graph_compute_paged (ggml_metal_t ctx, struct ggml_cgraph * gf);

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

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

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
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

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
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

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    ggml_metal_bsched_clear(ctx);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

ggml_metal_device_t ggml_metal_get_device(ggml_metal_t ctx) {
    return ctx->dev;
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
                ctx->has_error = true;
                return;
            }
        }
    }

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

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
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

// ---------------------------------------------------------------------------
// #226 paged-decode boundary-event schedule
// ---------------------------------------------------------------------------

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
}

void ggml_metal_set_boundary_schedule(
        ggml_metal_t ctx,
        int n_cuts,  struct ggml_tensor * const * cut_nodes,  ggml_metal_event_t * sig_ev, const uint64_t * sig_val,
        int n_waits, struct ggml_tensor * const * wait_nodes, ggml_metal_event_t * wait_ev, const uint64_t * wait_val) {
    ggml_metal_bsched_clear(ctx);
    if (n_cuts <= 0) {
        return; // cleared -> subsequent computes take the stock n_cb path
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

// Sequential per-boundary-committed encode (see ggml_metal_set_boundary_schedule).
// Runs on the calling thread with NO dispatch_apply: deterministic submit order
// is what makes the cross-command-buffer event signals monotonic and the waits
// land on the right segment. Async like the stock path -- failures surface at
// the next ggml_metal_synchronize (which drains cmd_bufs_ext), not here.
static enum ggml_status ggml_metal_graph_compute_paged(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    const int n_nodes = gf->n_nodes;

    // The schedule persists (pointer-keyed; NOT cleared here) -- see
    // ggml_metal_set_boundary_schedule. Cuts/waits are matched by node POINTER
    // against THIS graph, which may be one split of a multi-split decode.
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

        // Per-node flags: which of THIS graph's nodes are cut ends / wait starts
        // (a schedule node not present in this graph simply never matches). One
        // pass over the graph, O(n_nodes * (n_cuts + n_waits)) -- both schedule
        // sizes are ~n_layer, negligible next to the encode itself.
        bool * is_cut  = (bool *) calloc((size_t) n_nodes, sizeof(bool));
        bool * is_wait = (bool *) calloc((size_t) n_nodes, sizeof(bool));
        for (int i = 0; i < n_nodes; ++i) {
            struct ggml_tensor * node = gf->nodes[i];
            for (int c = 0; c < n_cuts; ++c) {
                if (cut_nodes[c] == node) { is_cut[i] = true; break; }
            }
            for (int w = 0; w < n_waits; ++w) {
                if (wait_nodes[w] == node) { is_wait[i] = true; break; }
            }
        }

        // Sorted-unique segment START indices: 0, every (cut index + 1), and
        // every wait index (each in (0, n_nodes)). Consecutive starts delimit
        // the segments; the last segment ends at n_nodes.
        const int cap = 2 * n_nodes + 2;
        int * starts = (int *) malloc(sizeof(int) * (size_t) cap);
        int n_starts = 0;
        starts[n_starts++] = 0;
        for (int i = 0; i < n_nodes; ++i) {
            if (is_cut[i] && i + 1 < n_nodes) { starts[n_starts++] = i + 1; }
            if (is_wait[i] && i > 0)          { starts[n_starts++] = i;     }
        }
        // insertion sort (n_starts is small, ~2*n_layer) + dedup in place
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

        // Encode each segment into its own command buffer, in order.
        for (int seg = 0; seg < n_starts; ++seg) {
            const int seg_start = starts[seg];
            const int seg_end   = (seg + 1 < n_starts) ? starts[seg + 1] : n_nodes;
            if (seg_end <= seg_start) {
                continue;
            }

            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];
            [cmd_buf enqueue]; // reserve the queue slot so execution order == creation order

            // Waits BEFORE this segment's ops. The segment begins exactly at a
            // wait node (every wait node was injected as a segment start); more
            // than one schedule wait can target the same node (a layer's K and V
            // regions sharing a first-reader), so emit ALL that match.
            if (is_wait[seg_start]) {
                struct ggml_tensor * node = gf->nodes[seg_start];
                for (int w = 0; w < n_waits; ++w) {
                    if (wait_nodes[w] == node && wait_ev[w] != NULL) {
                        ggml_metal_event_encode_wait_value(wait_ev[w], (ggml_metal_cmd_buf_t) cmd_buf, wait_val[w]);
                    }
                }
            }

            // Encode nodes [seg_start, seg_end) with fusion (one ggml_metal_op),
            // exactly as the stock encode_async does over its slice.
            ggml_metal_op_t op = ggml_metal_op_init(
                ctx->dev,
                (ggml_metal_cmd_buf_t) cmd_buf,
                gf,
                seg_start,
                seg_end,
                ctx->use_fusion,
                ctx->use_concurrency,
                false, // capture unsupported on the paged path
                ctx->debug_graph,
                ctx->debug_fusion);

            for (int idx = 0; idx < ggml_metal_op_n_nodes(op); ++idx) {
                const int r = ggml_metal_op_encode(op, idx);
                if (r == 0) {
                    break;
                }
                idx += r - 1;
            }

            ggml_metal_op_free(op);

            // Signals AFTER this segment (it ends at seg_end-1, a cut node).
            const int seg_last = seg_end - 1;
            if (is_cut[seg_last]) {
                struct ggml_tensor * node = gf->nodes[seg_last];
                for (int c = 0; c < n_cuts; ++c) {
                    if (cut_nodes[c] == node && sig_ev[c] != NULL) {
                        // Signal on COMPLETION (not mid-stream): a host reclaim
                        // thread waits this to spill the just-written KV to disk,
                        // and must see coherent shared-storage bytes (see
                        // ggml_metal_event_signal_on_complete's doc).
                        ggml_metal_event_signal_on_complete(sig_ev[c], (ggml_metal_cmd_buf_t) cmd_buf, sig_val[c]);
                    }
                }
            }

            [cmd_buf commit];

            // Hand ownership to cmd_bufs_ext (drained by ggml_metal_synchronize,
            // once per token) and track the last buffer for synchronize's wait.
            [ctx->cmd_bufs_ext addObject:cmd_buf];
            ctx->cmd_buf_last = cmd_buf;
        }

        free(starts);
        free(is_cut);
        free(is_wait);
    }

    // Persistent: do NOT clear the schedule here -- it stays in force for the
    // remaining splits of this decode and the reused graph of later tokens,
    // until the caller clears it (set_boundary_schedule with n_cuts == 0).
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // #226: while a (persistent) paged-decode schedule is set, divert to the
    // sequential per-boundary-committed encode with event signal/wait. It stays
    // in force until the caller clears it, so it covers every split of a decode.
    if (ctx->bsched_n_cuts > 0) {
        return ggml_metal_graph_compute_paged(ctx, gf);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

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
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
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

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
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
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
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

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
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

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

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
