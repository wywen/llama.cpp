#include "metal-command-buffer.h"

#include "ggml-metal.h"

#import <Metal/Metal.h>

#if __has_feature(objc_arc)
#error "metal-command-buffer.mm counts references by hand: build it without ARC"
#endif

void * test_metal_command_buffer_new(void * queue) {
    if (queue == NULL) {
        return NULL;
    }
    @autoreleasepool {
        return [[(id<MTLCommandQueue>) queue commandBufferWithUnretainedReferences] retain];
    }
}

void test_metal_command_buffer_commit(void * cmd_buf) {
    [(id<MTLCommandBuffer>) cmd_buf commit];
}

void test_metal_command_buffer_wait(void * cmd_buf) {
    [(id<MTLCommandBuffer>) cmd_buf waitUntilCompleted];
}

bool test_metal_command_buffer_enqueued(void * cmd_buf) {
    return [(id<MTLCommandBuffer>) cmd_buf status] != MTLCommandBufferStatusNotEnqueued;
}

bool test_metal_command_buffer_completed(void * cmd_buf) {
    return [(id<MTLCommandBuffer>) cmd_buf status] == MTLCommandBufferStatusCompleted;
}

void test_metal_command_buffer_encode_wait(void * cmd_buf, void * event, uint64_t value) {
    [(id<MTLCommandBuffer>) cmd_buf encodeWaitForEvent:(id<MTLSharedEvent>) event value:value];
}

void test_metal_command_buffer_release(void * cmd_buf) {
    [(id<MTLCommandBuffer>) cmd_buf release];
}

void * test_metal_shared_event_new(ggml_backend_dev_t dev) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_backend_metal_device_get_mtl_device(dev);
    if (device == nil) {
        return NULL;
    }
    return [device newSharedEvent];
}

void test_metal_shared_event_signal(void * event, uint64_t value) {
    ((id<MTLSharedEvent>) event).signaledValue = value;
}

void test_metal_shared_event_release(void * event) {
    [(id<MTLSharedEvent>) event release];
}
