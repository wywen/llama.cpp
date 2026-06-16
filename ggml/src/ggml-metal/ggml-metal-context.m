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
};

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

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
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

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
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

        // [rrl] #135 Stage 2: windowed-sequential compute mode.
        // When RRL_METAL_CB_WINDOW is set (and not in capture mode), walk gf->nodes
        // to build per-layer window ranges [start, end), commit one command buffer per
        // window, and drain each window (waitUntilCompleted) before starting the next.
        // This releases the GPU useResource residency for each layer's experts as soon
        // as that layer's CB completes, bounding peak to ~1-2 layers (~480 MiB-1 GiB).
        // The existing rolling msync evictor (installed via cb_eval) handles the CPU
        // page side; this change handles the GPU useResource side.
        // When unset (or in capture mode), falls through to the parallel dispatch_apply
        // path unchanged.
        const bool use_cb_window = (getenv("RRL_METAL_CB_WINDOW") != NULL) && !use_capture;
        if (use_cb_window) {
            // -- Window planner -------------------------------------------------
            // Walk gf->nodes to build a list of window boundaries.  A new window
            // starts at each node that is:
            //   1. a MUL_MAT_ID op, AND
            //   2. src[0] is a per-expert mmap-metal tensor
            //      (rrl_is_expert_mmap_metal — same gate the encoder uses), AND
            //   3. the layer id parsed from src[0]->name ("blk.<L>.ffn_...") is
            //      different from the layer id of the previous expert node seen.
            // This cuts one window per MoE layer.  Non-expert nodes before the
            // first expert node go into window 0 (the prefix window).  Non-expert
            // nodes after the last expert node (or between layers) go into the
            // current window's trailing range — they are emitted in the window that
            // opens at the first expert node of that layer.
            //
            // Implementation: collect the start indices of each new window; the end
            // of window[i] is start of window[i+1] (or gf->n_nodes for the last).

            struct rrl_window { int start; };
            NSMutableArray * windows_arr = [NSMutableArray array];

            {
                int prev_layer = -2; // -2 = no layer seen yet
                for (int i = 0; i < gf->n_nodes; ++i) {
                    struct ggml_tensor * node = gf->nodes[i];
                    if (node->op != GGML_OP_MUL_MAT_ID) { continue; }
                    struct ggml_tensor * s0 = node->src[0];
                    // Use the same gate as the encoder: only per-expert mmap-metal
                    // tensors trigger windowing (via the C-callable wrapper so we
                    // don't duplicate the detection logic in this .m file).
                    if (!rrl_is_expert_mmap_metal_c(s0)) { continue; }

                    // Parse layer id from src[0]->name ("blk.<L>.ffn_...").
                    int layer = -1;
                    const char * p = s0->name;
                    if (p[0]=='b' && p[1]=='l' && p[2]=='k' && p[3]=='.') {
                        p += 4;
                        int val = 0; bool ok = false;
                        while (*p >= '0' && *p <= '9') { val = val*10 + (*p-'0'); ++p; ok=true; }
                        if (ok && *p == '.') { layer = val; }
                    }

                    if (layer != prev_layer) {
                        // Cut a new window at this node.
                        struct rrl_window w = { .start = i };
                        [windows_arr addObject:[NSValue valueWithBytes:&w objCType:@encode(struct rrl_window)]];
                        prev_layer = layer;
                    }
                }
            }

            // Build the flat start[] array.  Window 0 always starts at 0 (prefix).
            // If the planner found no expert nodes, there is one implicit window
            // covering the whole graph.
            int n_win = (int) windows_arr.count + 1; // +1 for the prefix window
            int * win_starts = (int *) alloca((size_t)(n_win + 1) * sizeof(int));
            win_starts[0] = 0; // prefix window
            for (int wi = 0; wi < (int) windows_arr.count; ++wi) {
                struct rrl_window w;
                [[windows_arr objectAtIndex:wi] getValue:&w];
                win_starts[wi + 1] = w.start;
            }
            win_starts[n_win] = gf->n_nodes; // sentinel (end of last window)

            // -- Sequential commit loop -----------------------------------------
            // Two sub-paths:
            //
            //   Sync  (RRL_METAL_CB_ASYNC unset): per-window waitUntilCompleted
            //     before the next window starts; sync rolling-evict runs inside
            //     the encoder (ops.cpp).  Retain accounting: retain→count 1; last
            //     CB goes into cmd_bufs_ext→count 2; synchronize releases both.
            //
            //   Async (RRL_METAL_CB_ASYNC=D): counting semaphore of depth D limits
            //     in-flight CBs.  Each window's addCompletedHandler runs the reclaim
            //     (rrl_async_window_reclaim_and_free) and releases the CB.  After all
            //     windows are committed, drain the semaphore D times to ensure every
            //     handler has fired before graph_compute returns.  The handler is the
            //     SOLE evictor; the sync rolling-evict block in ops.cpp is bypassed.
            //     cmd_buf_last is set to nil (all work done; synchronize is a no-op).

            const int rrl_async_d = rrl_metal_cb_async_depth();

            if (rrl_async_d > 0) {
                // -- Async path (RRL_METAL_CB_ASYNC=D) ----------------------------
                // Semaphore starts at D: the main thread takes one slot before
                // creating each CB; the completion handler signals one slot back.
                dispatch_semaphore_t sem = dispatch_semaphore_create(rrl_async_d);

                for (int wi = 0; wi < n_win; ++wi) {
                    const int w_start = win_starts[wi];
                    const int w_end   = win_starts[wi + 1];

                    if (w_start >= w_end) { continue; } // skip empty windows

                    // Throttle: block until a slot is available (<D in-flight CBs).
                    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

                    id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
                    [cmd_buf retain]; // our explicit retain → count 1
                    [cmd_buf enqueue];

                    ggml_metal_op_t ctx_op = ggml_metal_op_init(
                        ctx->dev,
                        (ggml_metal_cmd_buf_t) cmd_buf,
                        gf,
                        w_start,
                        w_end,
                        ctx->use_fusion,
                        ctx->use_concurrency,
                        /*use_capture=*/false,
                        ctx->debug_graph,
                        ctx->debug_fusion);

                    for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                        const int res = ggml_metal_op_encode(ctx_op, idx);
                        if (res == 0) { break; }
                        idx += res - 1;
                    }

                    // ends encoding and frees the encoder
                    ggml_metal_op_free(ctx_op);

                    // Drain the async-records accumulated by this window's encoder
                    // calls into a heap allocation.  The completion handler owns it.
                    void * rrl_handle = rrl_async_window_take_records();

                    // Completion handler: sole evictor + release + signal.
                    // h is captured by value (opaque void* — no C++ copy needed).
                    // The handler executes on the Metal driver thread after the CB
                    // finishes; by that time all GPU reads of this window's experts
                    // are complete, so it is safe to msync-evict them.
                    void * h = rrl_handle; // explicit local for Block capture (C)
                    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                        if (cb.status != MTLCommandBufferStatusCompleted) {
                            ctx->has_error = true;
                        }
                        rrl_async_window_reclaim_and_free(h); // sole evictor; safe if NULL
                        [cb release];                          // balance our [retain] above
                        dispatch_semaphore_signal(sem);
                    }];

                    [cmd_buf commit]; // NO waitUntilCompleted
                    // DO NOT add to cmd_bufs_ext — the handler owns the release.
                }

                // Drain: acquire all D permits so every in-flight completion handler
                // has fired.  Each committed CB's handler signals exactly once, so the
                // initial D permits plus those signals make these D waits always
                // complete regardless of how many windows committed (no deadlock).
                for (int i = 0; i < rrl_async_d; ++i) {
                    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
                }
                // Restore the count to its creation value BEFORE releasing: libdispatch
                // traps (EXC_BREAKPOINT in _dispatch_semaphore_dispose) if a semaphore is
                // deallocated while its value is below the value it was created with.
                for (int i = 0; i < rrl_async_d; ++i) {
                    dispatch_semaphore_signal(sem);
                }
                dispatch_release(sem);

                // All window CBs are done and released.  Set cmd_buf_last to nil so
                // synchronize() skips its waitUntilCompleted (no live reference left).
                ctx->cmd_buf_last = nil;

                if (ctx->has_error) {
                    return GGML_STATUS_FAILED;
                }

            } else {
                // -- Sync path (RRL_METAL_CB_ASYNC unset) -------------------------
                // For each window [start, end):
                //   create CB → enqueue → build op → encode → free (ends encoding)
                //   → commit → waitUntilCompleted → check status
                // waitUntilCompleted after each CB releases that window's useResource
                // residency before the next window faults its experts.
                //
                // Memory management: commandBufferWithUnretainedReferences + explicit
                // [retain] gives us a retain count of 1.  We transfer that retain to
                // cmd_bufs_ext (for the last window) so synchronize() can hold a live
                // reference via cmd_buf_last.  Intermediate window CBs are released
                // immediately after drain (they have already completed).
                id<MTLCommandBuffer> last_win_cmd_buf = nil;

                for (int wi = 0; wi < n_win; ++wi) {
                    const int w_start = win_starts[wi];
                    const int w_end   = win_starts[wi + 1];

                    if (w_start >= w_end) { continue; } // skip empty windows

                    id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
                    [cmd_buf retain];
                    [cmd_buf enqueue];

                    ggml_metal_op_t ctx_op = ggml_metal_op_init(
                        ctx->dev,
                        (ggml_metal_cmd_buf_t) cmd_buf,
                        gf,
                        w_start,
                        w_end,
                        ctx->use_fusion,
                        ctx->use_concurrency,
                        /*use_capture=*/false,
                        ctx->debug_graph,
                        ctx->debug_fusion);

                    for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                        const int res = ggml_metal_op_encode(ctx_op, idx);
                        if (res == 0) { break; }
                        idx += res - 1;
                    }

                    // ends encoding and frees the encoder
                    ggml_metal_op_free(ctx_op);

                    [cmd_buf commit];

                    // Drain: waitUntilCompleted releases this window's useResource
                    // residency before the next window faults its experts.
                    [cmd_buf waitUntilCompleted];

                    MTLCommandBufferStatus status = [cmd_buf status];
                    if (status != MTLCommandBufferStatusCompleted) {
                        GGML_LOG_INFO("%s: [rrl] windowed CB %d failed with status %lu\n",
                                      __func__, wi, (unsigned long) status);
                        if (status == MTLCommandBufferStatusError) {
                            GGML_LOG_INFO("error: %s\n",
                                          [[cmd_buf error].localizedDescription UTF8String]);
                        }
                        ctx->has_error = true;
                        // Release the failed CB's retain (it is not stored in cmd_bufs_ext).
                        [cmd_buf release];
                        // Release any prior window CB we were holding in last_win_cmd_buf
                        // (it succeeded but we haven't added it to cmd_bufs_ext yet).
                        if (last_win_cmd_buf != nil) {
                            [last_win_cmd_buf release];
                            last_win_cmd_buf = nil;
                        }
                        return GGML_STATUS_FAILED;
                    }

                    // Release the previous last-window CB (it has completed and we are
                    // about to replace last_win_cmd_buf).
                    if (last_win_cmd_buf != nil) {
                        [last_win_cmd_buf release];
                    }
                    last_win_cmd_buf = cmd_buf; // transfer our retain to last_win_cmd_buf
                }

                // Track the last window's CB in cmd_bufs_ext so synchronize() can
                // wait on it via cmd_buf_last.  Retain count accounting:
                //   commandBufferWithUnretainedReferences starts at 0.
                //   Our [retain] (at top of loop)   → count 1
                //   addObject (array retains)        → count 2
                //   synchronize [release]            → count 1
                //   synchronize removeAllObjects     → count 0 → dealloc ✓
                // Do NOT call an extra [release] here — we need count=2 entering
                // synchronize so both its explicit [release] and removeAllObjects
                // balance correctly.
                if (last_win_cmd_buf != nil) {
                    [ctx->cmd_bufs_ext addObject:last_win_cmd_buf];
                    ctx->cmd_buf_last = last_win_cmd_buf;
                    // Our explicit [retain] is still live; addObject added the second
                    // retain.  Leave both — synchronize consumes them.
                }
            } // end sync/async branch
        } else {
        // -- Original parallel dispatch_apply path (unchanged) ------------------

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

        } // end else (original parallel path)

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
