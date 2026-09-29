#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#include <limits.h>
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

    // stop the graph reorder at boundary marker nodes (see ggml_metal_set_reorder_barriers)
    bool use_reorder_barriers;

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

    // Terminal state of each paged-encode segment, captured by that segment's
    // completion handler rather than read back off the buffer afterwards.
    //
    // A failure ABANDONS the rest of the queue, and an abandoned buffer carries
    // a bare status with no localizedDescription -- so reading state back after
    // the fact can only find the first buffer that did not complete, which is
    // usually not the one that failed. Capturing at completion records what
    // each buffer actually reported, at the instant it reported it.
    struct ggml_metal_result_storage * seg_results;
    int                            seg_results_n;
    struct ggml_metal_completion * completion;
    struct ggml_metal_projection       encode_projection;
    void (* completion_failure)(void *);
    void * completion_failure_user;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
    // localizedDescription of the command buffer that carried the error putting this context
    // into `has_error`, or an explicit note when the queue was abandoned without one. Owned
    // copy: the command buffers it came from are released immediately after.
    char last_error[256];

    // Paged-decode boundary-event schedule (PERSISTENT + pointer-keyed;
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
    struct ggml_metal_completion_cut * bsched_completion_cuts;
    size_t bsched_n_completion_cuts;
    struct ggml_metal_receipt_binding_cut * receipt_cuts;
    size_t receipt_cuts_capacity;
    uint64_t receipt_generation;
    bool receipt_open;
    int                    bsched_n_waits;
    struct ggml_tensor  ** bsched_wait_nodes; // node ptrs; segment starts (wait before)
    ggml_metal_event_t  *  bsched_wait_ev;
    uint64_t *             bsched_wait_val;

    // Encode window over the paged path (PERSISTENT + pointer-keyed, like the
    // boundary schedule above; see ggml_metal_set_encode_window). While set,
    // ggml_metal_graph_compute_paged encodes only the segments intersecting
    // [first_node, last_node] and moves boundary tensors through ordered ingress
    // copies before the first encoded segment and egress copies after the last.
    // Pair arrays are owned copies; the tensors remain caller-owned.
    //
    // ewin_out_of_band is the caller's assertion of which FULL-GRAPH nodes lie
    // outside the active band. It is not used to select what to encode -- it is
    // a guard: any split that would encode one of these nodes is rejected before
    // a single command buffer exists (see ggml_metal_ewin_split_is_in_band).
    // Empty (the default) makes the guard vacuous and the path byte-identical to
    // one compiled without it. Owned copy of the array; nodes stay caller-owned.
    bool                                  ewin_active;
    struct ggml_tensor                 *  ewin_first_node;
    struct ggml_tensor                 *  ewin_last_node;
    size_t                                ewin_n_ingress;
    struct ggml_metal_tensor_copy_pair *  ewin_ingress;
    size_t                                ewin_n_egress;
    struct ggml_metal_tensor_copy_pair *  ewin_egress;
    size_t                                ewin_n_out_of_band;
    struct ggml_tensor                 ** ewin_out_of_band;
};

// Boundary-event schedule helpers (definitions below).
static void             ggml_metal_bsched_clear        (ggml_metal_t ctx);
static enum ggml_status ggml_metal_graph_compute_paged (ggml_metal_t ctx, struct ggml_cgraph * gf);

static void ggml_metal_completion_publish_event(void * unused, void * event, uint64_t value) {
    (void) unused;
    ((id<MTLSharedEvent>) event).signaledValue = value;
}

static void ggml_metal_completion_latch_failure(void * raw) {
    ggml_metal_t ctx = raw;
    if (ctx->completion_failure) { ctx->completion_failure(ctx->completion_failure_user); }
}

static void ggml_metal_completion_retain_event(void * event) {
    [(id<MTLSharedEvent>) event retain];
}

static void ggml_metal_completion_release_event(void * event) {
    [(id<MTLSharedEvent>) event release];
}

static void ggml_metal_wait_paged_buffers(ggml_metal_t ctx) {
    for (id<MTLCommandBuffer> buffer in ctx->cmd_bufs_ext) {
        [buffer waitUntilCompleted];
    }
    if (ctx->completion) { ggml_metal_completion_wait(ctx->completion); }
}

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

        // off unless a caller asks for it: the stock path must reorder exactly as before
        res->use_reorder_barriers = false;

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
        res->last_error[0] = '\0';

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->seg_results = ggml_metal_result_storage_new();
        res->completion = ggml_metal_completion_new(ggml_metal_completion_publish_event,
                ggml_metal_completion_latch_failure, ggml_metal_completion_retain_event,
                ggml_metal_completion_release_event, res);
        if (!res->completion || !res->seg_results) {
            ggml_metal_free(res);
            return NULL;
        }

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);
    if (ctx->receipt_open) {
        (void) ggml_metal_receipts_context_finish(ctx, ctx->receipt_generation, GGML_STATUS_FAILED);
    }
    ggml_metal_wait_paged_buffers(ctx);
    ggml_metal_completion_free(ctx->completion);
    ggml_metal_projection_free(&ctx->encode_projection);
    ctx->completion = NULL;

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

    ggml_metal_result_storage_free(ctx->seg_results);
    ctx->seg_results = NULL;
    ctx->seg_results_n = 0;

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
    free(ctx->receipt_cuts);
    ggml_metal_clear_encode_window(ctx);

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

// Names the command buffer that actually FAILED, given the index of the first one found not
// completed.
//
// Those are usually not the same buffer. A GPU fault or timeout abandons every command buffer
// still queued behind the one that hit it, and an abandoned buffer sits at Scheduled with no
// error attached -- so the first not-completed index names the first ABANDONED buffer, which
// carries no information about what went wrong. Worse, the caller only prints an error
// description when the status IS Error, so a fault reported through an abandoned index prints no
// description at all and the run says only that some buffer "failed with status 3".
//
// Scanning on for a buffer carrying a real error is what turns that into a diagnosis: its
// localizedDescription is the only place the actual cause is written down.
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
    ggml_metal_wait_paged_buffers(ctx);
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

    // Callback status is captured before its terminal hook. Read error text only here,
    // while every command buffer remains retained and after all callbacks have drained.
    for (int i = 0; i < ctx->seg_results_n; ++i) {
        struct ggml_metal_seg_result * result = ggml_metal_result_storage_at(ctx->seg_results, (size_t) i);
        if (result->status == (int) MTLCommandBufferStatusCompleted) {
            continue;
        }
        GGML_LOG_ERROR("%s: paged segment %d finished with status %d\n", __func__, i, result->status);
        ctx->has_error = true;
        snprintf(ctx->last_error, sizeof(ctx->last_error), "paged segment %d finished with status %d", i, result->status);
        for (int j = i; j < ctx->seg_results_n; ++j) {
            struct ggml_metal_seg_result * later = ggml_metal_result_storage_at(ctx->seg_results, (size_t) j);
            id<MTLCommandBuffer> buffer = (id<MTLCommandBuffer>) later->buffer;
            NSError * error = [buffer error];
            if (error == nil) {
                continue;
            }
            const char * description = [[error localizedDescription] UTF8String];
            if (description != NULL) {
                GGML_LOG_ERROR("%s: paged segment %d is the first carrying an error: %s\n", __func__, j, description);
                snprintf(ctx->last_error, sizeof(ctx->last_error), "paged segment %d: %s", j, description);
            }
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
// paged-decode boundary-event schedule
// ---------------------------------------------------------------------------

static void ggml_metal_bsched_clear(ggml_metal_t ctx) {
    free(ctx->bsched_cut_nodes);
    free(ctx->bsched_sig_ev);
    free(ctx->bsched_sig_val);
    free(ctx->bsched_wait_nodes);
    free(ctx->bsched_wait_ev);
    free(ctx->bsched_wait_val);
    free(ctx->bsched_completion_cuts);
    ctx->bsched_completion_cuts = NULL;
    ctx->bsched_n_completion_cuts = 0;
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
    if (ctx->receipt_open) {
        (void) ggml_metal_receipts_context_finish(ctx, ctx->receipt_generation, GGML_STATUS_FAILED);
    }
    ggml_metal_bsched_clear(ctx);
    if (n_cuts == 0 && n_waits == 0) {
        return; // cleared -> subsequent computes take the stock n_cb path
    }
    if (n_cuts <= 0 || n_waits < 0 || !cut_nodes || !sig_ev || !sig_val ||
            (n_waits && (!wait_nodes || !wait_ev || !wait_val)) ||
            (size_t) n_cuts > SIZE_MAX / sizeof(struct ggml_metal_completion_cut) ||
            (size_t) n_cuts > SIZE_MAX / sizeof(struct ggml_tensor *) ||
            (size_t) n_cuts > SIZE_MAX / sizeof(ggml_metal_event_t) ||
            (size_t) n_cuts > SIZE_MAX / sizeof(uint64_t) ||
            (size_t) n_waits > SIZE_MAX / sizeof(struct ggml_tensor *) ||
            (size_t) n_waits > SIZE_MAX / sizeof(ggml_metal_event_t) ||
            (size_t) n_waits > SIZE_MAX / sizeof(uint64_t)) {
        goto allocation_failed;
    }

    ctx->bsched_cut_nodes = malloc(sizeof(struct ggml_tensor *) * (size_t) n_cuts);
    ctx->bsched_sig_ev    = malloc(sizeof(ggml_metal_event_t) * (size_t) n_cuts);
    ctx->bsched_sig_val   = malloc(sizeof(uint64_t) * (size_t) n_cuts);
    ctx->bsched_completion_cuts = calloc((size_t) n_cuts, sizeof(*ctx->bsched_completion_cuts));
    if (!ctx->bsched_cut_nodes || !ctx->bsched_sig_ev || !ctx->bsched_sig_val || !ctx->bsched_completion_cuts) {
        goto allocation_failed;
    }
    memcpy(ctx->bsched_cut_nodes, cut_nodes, sizeof(struct ggml_tensor *) * (size_t) n_cuts);
    memcpy(ctx->bsched_sig_ev, sig_ev, sizeof(ggml_metal_event_t) * (size_t) n_cuts);
    memcpy(ctx->bsched_sig_val, sig_val, sizeof(uint64_t) * (size_t) n_cuts);
    for (int i = 0; i < n_cuts; ++i) {
        if (!sig_ev[i]) { continue; }
        ctx->bsched_completion_cuts[ctx->bsched_n_completion_cuts++] =
            (struct ggml_metal_completion_cut) { cut_nodes[i], ggml_metal_event_get_obj(sig_ev[i]), sig_val[i] };
    }

    if (n_waits > 0) {
        ctx->bsched_wait_nodes = malloc(sizeof(struct ggml_tensor *) * (size_t) n_waits);
        ctx->bsched_wait_ev    = malloc(sizeof(ggml_metal_event_t) * (size_t) n_waits);
        ctx->bsched_wait_val   = malloc(sizeof(uint64_t) * (size_t) n_waits);
        if (!ctx->bsched_wait_nodes || !ctx->bsched_wait_ev || !ctx->bsched_wait_val) {
            goto allocation_failed;
        }
        memcpy(ctx->bsched_wait_nodes, wait_nodes, sizeof(struct ggml_tensor *) * (size_t) n_waits);
        memcpy(ctx->bsched_wait_ev, wait_ev, sizeof(ggml_metal_event_t) * (size_t) n_waits);
        memcpy(ctx->bsched_wait_val, wait_val, sizeof(uint64_t) * (size_t) n_waits);
    }
    ctx->bsched_n_cuts = n_cuts;
    ctx->bsched_n_waits = n_waits;
    return;

allocation_failed:
    ggml_metal_bsched_clear(ctx);
    ctx->has_error = true;
    snprintf(ctx->last_error, sizeof(ctx->last_error), "invalid or unallocatable Metal boundary schedule");
    ggml_metal_completion_fail(ctx->completion);
}

enum ggml_status ggml_metal_receipts_context_begin(
        ggml_metal_t ctx, const struct ggml_metal_receipt_node * expected, size_t n_expected,
        const struct ggml_metal_receipt_cut * cuts, size_t n_cuts,
        ggml_metal_terminal_receipt_fn terminal, ggml_metal_generation_quiesced_fn quiesced,
        void * cookie, uint64_t * generation) {
    if (!ctx || ctx->has_error || !ctx->completion_failure || ctx->receipt_open ||
        !cuts || !n_cuts || n_cuts != ctx->bsched_n_completion_cuts ||
        n_cuts > SIZE_MAX / sizeof(*ctx->receipt_cuts)) {
        return GGML_STATUS_FAILED;
    }
    if (n_cuts > ctx->receipt_cuts_capacity) {
        void * grown = realloc(ctx->receipt_cuts, n_cuts * sizeof(*ctx->receipt_cuts));
        if (!grown) {
            ggml_metal_completion_fail(ctx->completion);
            return GGML_STATUS_ALLOC_FAILED;
        }
        ctx->receipt_cuts = grown;
        ctx->receipt_cuts_capacity = n_cuts;
    }
    size_t j = 0;
    for (int i = 0; i < ctx->bsched_n_cuts; ++i) {
        if (!ctx->bsched_sig_ev[i]) {
            continue;
        }
        if (j >= n_cuts || cuts[j].node != ctx->bsched_cut_nodes[i] ||
            cuts[j].event != ctx->bsched_sig_ev[i] || cuts[j].value != ctx->bsched_sig_val[i]) {
            ggml_metal_completion_fail(ctx->completion);
            return GGML_STATUS_FAILED;
        }
        ctx->receipt_cuts[j] = (struct ggml_metal_receipt_binding_cut) {
            cuts[j].split, cuts[j].ordinal, ctx->bsched_completion_cuts[j],
        };
        ++j;
    }
    const enum ggml_status result = ggml_metal_receipts_begin(
        ctx->completion, expected, n_expected, ctx->receipt_cuts, n_cuts, terminal, quiesced, cookie, generation);
    if (result != GGML_STATUS_SUCCESS) {
        ggml_metal_completion_fail(ctx->completion);
        return result;
    }
    ctx->receipt_generation = *generation;
    ctx->receipt_open = true;
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_metal_receipts_context_finish(
        ggml_metal_t ctx, uint64_t generation, enum ggml_status submission_status) {
    if (!ctx) {
        return GGML_STATUS_FAILED;
    }
    if (!ctx->receipt_open || ctx->receipt_generation != generation) {
        ggml_metal_completion_fail(ctx->completion);
        ctx->has_error = true;
        snprintf(ctx->last_error, sizeof(ctx->last_error), "invalid Metal receipt generation finish");
        return GGML_STATUS_FAILED;
    }
    ctx->receipt_open = false;
    const enum ggml_status result = ggml_metal_receipts_finish(ctx->completion, generation, submission_status);
    if (result != GGML_STATUS_SUCCESS) {
        ctx->has_error = true;
        snprintf(ctx->last_error, sizeof(ctx->last_error), "Metal receipt submission failed or expected coverage missing");
    }
    return result;
}

void ggml_metal_set_boundary_failure_callback(ggml_metal_t ctx, void (*latch_failure)(void *), void * user) {
    if (ctx->completion_failure != latch_failure || ctx->completion_failure_user != user) {
        ggml_metal_wait_paged_buffers(ctx);
    }
    ctx->completion_failure = latch_failure;
    ctx->completion_failure_user = user;
}


// The banded-prefill encode window. `first_node` is the EXCLUSIVE lower
// boundary (the band-entry residual -- its own segment is outside the
// window) and `last_node` the INCLUSIVE upper boundary; either may be NULL
// for an open edge (from split start / to split end). Endpoints are matched
// by pointer against each computed graph and PROJECTED per split: a split
// containing only the lower boundary windows to its end, one containing
// only the upper windows from its start, and one containing neither
// non-NULL endpoint encodes in full -- which is exactly what the
// input-processing splits outside the trunk need (masks/positions must be
// recomputed every tile). Ordered ingress and egress copies fire only in the
// split where their boundary node matched. Every pair must contain two tensors
// of equal byte size because each blit copies the src's full extent onto the
// dst's allocation. The context owns copies of the pair arrays, not the tensors.
//
// `out_of_band` closes the hole that per-split projection leaves open. Because
// the window is resolved per split, a split holding NEITHER non-NULL endpoint
// gets window_applies == false and encodes in full -- intended for the
// input-processing splits, but indistinguishable from a TRUNK split that fell
// after the boundary split and whose nodes are all out of band. The caller
// therefore names the full-graph nodes it asserts lie OUTSIDE the active band
// (a banded caller passes each out-of-band layer's `l_out-<il>` marker) and
// ggml_metal_graph_compute_paged refuses -- with GGML_STATUS_FAILED, before any
// command buffer is created -- to submit a split that would encode one of them.
// The invariant enforced is: no submitted split may ever encode a node that
// lies outside the active band. This is a GUARD, not a selector: it never
// changes which segments are encoded, only whether the split is submitted at
// all. Matching is by pointer against each computed graph, so a named node
// absent from a split simply never fires. Pass n_out_of_band == 0 (the default
// for every non-banded caller) and the guard is vacuous. The context owns a
// copy of the array; the nodes remain caller-owned.
void ggml_metal_set_encode_window(
        ggml_metal_t ctx,
        struct ggml_tensor * first_node,  struct ggml_tensor * last_node,
        size_t n_ingress,     const struct ggml_metal_tensor_copy_pair * ingress,
        size_t n_egress,      const struct ggml_metal_tensor_copy_pair * egress,
        size_t n_out_of_band, struct ggml_tensor * const * out_of_band) {
    GGML_ASSERT(n_ingress     == 0 || ingress     != NULL);
    GGML_ASSERT(n_egress      == 0 || egress      != NULL);
    GGML_ASSERT(n_out_of_band == 0 || out_of_band != NULL);
    GGML_ASSERT(n_ingress     <= SIZE_MAX / sizeof(*ingress));
    GGML_ASSERT(n_egress      <= SIZE_MAX / sizeof(*egress));
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
        // A NULL entry would silently match nothing and quietly weaken the
        // guard, so reject it at the door rather than at encode time.
        GGML_ASSERT(out_of_band[i] != NULL);
    }

    struct ggml_metal_tensor_copy_pair * ingress_copy     = NULL;
    struct ggml_metal_tensor_copy_pair * egress_copy      = NULL;
    struct ggml_tensor                ** out_of_band_copy = NULL;
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
    ctx->ewin_ingress       = ingress_copy;
    ctx->ewin_n_egress      = n_egress;
    ctx->ewin_egress        = egress_copy;
    ctx->ewin_n_out_of_band = n_out_of_band;
    ctx->ewin_out_of_band   = out_of_band_copy;
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

// One full-tensor blit between two resident tensors, encoded into the given
// command buffer -- the transport for the encode window's boundary
// activation (device-side copy, no host round trip).
static void ggml_metal_ewin_blit(id<MTLCommandBuffer> cmd_buf, const struct ggml_tensor * src, const struct ggml_tensor * dst) {
    struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
    struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
    GGML_ASSERT(bid_src.metal != nil);
    GGML_ASSERT(bid_dst.metal != nil);

    id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
    [encoder copyFromBuffer:bid_src.metal
               sourceOffset:bid_src.offs
                   toBuffer:bid_dst.metal
          destinationOffset:bid_dst.offs
                       size:ggml_nbytes(src)];
    [encoder endEncoding];
}

// Stop the graph reorder at boundary marker nodes. See
// ggml_backend_metal_set_reorder_barriers for why a boundary-scheduling caller has to
// set this, and ggml_graph_node_is_boundary_marker for what counts as a marker.
void ggml_metal_set_reorder_barriers(ggml_metal_t ctx, bool enable) {
    ctx->use_reorder_barriers = enable;
}

enum ggml_status ggml_metal_project_graph(ggml_metal_t                   ctx,
                                          const struct ggml_cgraph *     graph,
                                          struct ggml_metal_projection * out) {
    if (!out) {
        return GGML_STATUS_FAILED;
    }
    out->n_segments = 0;
    if (!ctx || ctx->bsched_n_cuts <= 0 || ctx->has_error || ggml_metal_completion_failed(ctx->completion)) {
        return GGML_STATUS_FAILED;
    }
    const struct ggml_metal_projection_config config = {
        ctx->bsched_n_cuts,   ctx->bsched_cut_nodes, ctx->bsched_n_waits,     ctx->bsched_wait_nodes, ctx->ewin_active,
        ctx->ewin_first_node, ctx->ewin_last_node,   ctx->ewin_n_out_of_band, ctx->ewin_out_of_band,
    };
    return ggml_metal_project_segments(graph, &config, out);
}

// Sequential per-boundary-committed encode (see ggml_metal_set_boundary_schedule).
// Runs on the calling thread with NO dispatch_apply: deterministic submit order
// is what makes the cross-command-buffer event signals monotonic and the waits
// land on the right segment. Async like the stock path -- failures surface at
// the next ggml_metal_synchronize (which drains cmd_bufs_ext), not here.
// How long the encode will wait for a segment's gate before committing anyway.
// Generous: exceeding it is not an error, only a return to committing ahead of
// the data for that one segment.
#define GGML_METAL_PAGED_GATE_WAIT_MS 30000

static enum ggml_status ggml_metal_graph_compute_paged(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    // The schedule persists (pointer-keyed; NOT cleared here) -- see
    // ggml_metal_set_boundary_schedule. Cuts/waits are matched by node POINTER
    // against THIS graph, which may be one split of a multi-split decode.
    const int                    n_waits    = ctx->bsched_n_waits;
    struct ggml_tensor ** const  wait_nodes = ctx->bsched_wait_nodes;
    ggml_metal_event_t * const   wait_ev    = ctx->bsched_wait_ev;
    const uint64_t     * const   wait_val   = ctx->bsched_wait_val;

    @autoreleasepool {
        ctx->gf = gf;

        // keep the memory wired (mirrors the stock graph_compute)
        ggml_metal_device_rsets_keep_alive(ctx->dev);

        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        struct ggml_metal_projection * projection = &ctx->encode_projection;
        const enum ggml_status projected = ggml_metal_project_graph(ctx, gf, projection);
        if (projected != GGML_STATUS_SUCCESS) {
            ggml_metal_completion_fail(ctx->completion);
            return projected;
        }
        uint32_t receipt_split = UINT32_MAX;
        if (!ggml_metal_receipts_validate_view(ctx->completion, gf, projection, &receipt_split)) {
            ggml_metal_completion_fail(ctx->completion);
            return GGML_STATUS_FAILED;
        }

        int n_actual = 0;
        for (int seg = 0; seg < projection->n_segments; ++seg) {
            if (!projection->segments[seg].skipped) {
                ++n_actual;
            }
        }
        if (n_actual > INT_MAX - ctx->seg_results_n) {
            ggml_metal_completion_fail(ctx->completion);
            return GGML_STATUS_FAILED;
        }
        const int needed_results = n_actual + ctx->seg_results_n;
        if (!ggml_metal_completion_reserve_additional(ctx->completion, (size_t) n_actual) ||
            !ggml_metal_result_storage_reserve(ctx->seg_results, (size_t) needed_results)) {
            ggml_metal_completion_fail(ctx->completion);
            return GGML_STATUS_FAILED;
        }

        // Encode each segment into its own command buffer, in order.
        for (int seg = 0; seg < projection->n_segments; ++seg) {
            const struct ggml_metal_segment * projected_seg = &projection->segments[seg];
            if (projected_seg->skipped) {
                continue;
            }
            const int seg_start = projected_seg->start;
            const int seg_end   = projected_seg->end;

            uint64_t             identity = 0;
            id<MTLCommandBuffer> cmd_buf  = [queue commandBufferWithUnretainedReferences];
            if (!cmd_buf || !ggml_metal_receipts_register(ctx->completion, receipt_split, gf->nodes,
                                                          seg_start, seg_end, &identity)) {
                ggml_metal_completion_fail(ctx->completion);
                return GGML_STATUS_FAILED;
            }
            [cmd_buf retain];
            [cmd_buf enqueue]; // reserve the queue slot so execution order == creation order

            // Waits BEFORE this segment's ops. The segment begins exactly at a
            // wait node (every wait node was injected as a segment start); more
            // than one schedule wait can target the same node (a layer's K and V
            // regions sharing a first-reader), so emit ALL that match.
            //
            // The GPU-side waits below stay as the ordering guarantee, but the
            // host waits for the same values FIRST, so the buffer is normally
            // committed only once its data is already there.
            //
            // A committed buffer parked on an event is occupying a queue slot and
            // being timed by the GPU for the whole park -- committing the whole
            // chain up front is what let a late segment's timeout budget cover
            // every read ahead of it. Committing when the gate is already
            // satisfied means the buffer runs rather than waits, so a segment
            // holds device resources only while it is executing.
            //
            // On timeout this proceeds and commits anyway: the encoded wait below
            // is still correct, so the fallback is the old behaviour for that one
            // segment rather than a failure.
            if (projected_seg->wait_at_start) {
                struct ggml_tensor * wnode = gf->nodes[seg_start];
                for (int w = 0; w < n_waits; ++w) {
                    if (wait_nodes[w] == wnode && wait_ev[w] != NULL) {
                        (void) ggml_metal_event_host_wait(wait_ev[w], wait_val[w],
                                GGML_METAL_PAGED_GATE_WAIT_MS);
                    }
                }
            }

            if (projected_seg->wait_at_start) {
                struct ggml_tensor * node = gf->nodes[seg_start];
                for (int w = 0; w < n_waits; ++w) {
                    if (wait_nodes[w] == node && wait_ev[w] != NULL) {
                        ggml_metal_event_encode_wait_value(wait_ev[w], (ggml_metal_cmd_buf_t) cmd_buf, wait_val[w]);
                    }
                }
            }

            // Window entry: deliver the boundary activation into its
            // producer's allocation before the first encoded segment's ops
            // read it. After the schedule waits (the blit must not outrun an
            // admit gate) and before the compute encoder.
            if (projected_seg->ingress_here) {
                for (size_t i = 0; i < ctx->ewin_n_ingress; ++i) {
                    ggml_metal_ewin_blit(cmd_buf, ctx->ewin_ingress[i].src, ctx->ewin_ingress[i].dst);
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

            // Window exit: capture the boundary activation after the last
            // encoded segment's ops produced it. In-order queue => this blit
            // completes before any later compute that re-enters the window
            // blits it back in.
            if (projected_seg->egress_here) {
                for (size_t i = 0; i < ctx->ewin_n_egress; ++i) {
                    ggml_metal_ewin_blit(cmd_buf, ctx->ewin_egress[i].src, ctx->ewin_egress[i].dst);
                }
            }

            // The registered slot and retained result storage live until every callback has returned.
            struct ggml_metal_seg_result * slot = ggml_metal_result_storage_at(ctx->seg_results, (size_t) ctx->seg_results_n++);
            slot->status = -1;
            slot->buffer = (void *) cmd_buf; // retained in cmd_bufs_ext until synchronize
            struct ggml_metal_completion * completion = ctx->completion;
            [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                slot->status = (int) [cb status];
                ggml_metal_completion_complete(completion, identity, slot->status == (int) MTLCommandBufferStatusCompleted);
            }];

            [cmd_buf commit];

            // Hand ownership to cmd_bufs_ext (drained by ggml_metal_synchronize,
            // once per token) and track the last buffer for synchronize's wait.
            [ctx->cmd_bufs_ext addObject:cmd_buf];
            ctx->cmd_buf_last = cmd_buf;
        }

    }

    // Persistent: do NOT clear the schedule here -- it stays in force for the
    // remaining splits of this decode and the reused graph of later tokens,
    // until the caller clears it (set_boundary_schedule with n_cuts == 0).
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error || ggml_metal_completion_failed(ctx->completion)) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // A paged schedule requires a live receipt observer before GPU enqueue.
    if (ctx->bsched_n_cuts > 0 && !ggml_metal_receipts_active(ctx->completion)) {
        ggml_metal_completion_fail(ctx->completion);
        GGML_LOG_ERROR("%s: paged batch missing required receipt observer or completion failed\n", __func__);
        return GGML_STATUS_FAILED;
    }
    // A persistent paged schedule uses sequential per-boundary-committed encode
    // with event signals/waits for each split until the caller clears it.
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
        ggml_graph_optimize(gf, ctx->use_reorder_barriers);
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
