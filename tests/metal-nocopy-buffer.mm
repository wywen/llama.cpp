#include "metal-nocopy-buffer.h"

#include "ggml-metal.h"

#import <Metal/Metal.h>

#if __has_feature(objc_arc)
#error "metal-nocopy-buffer.mm counts references by hand: build it without ARC"
#endif

void * test_metal_nocopy_buffer_new(ggml_backend_dev_t dev, void * ptr, size_t size) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_backend_metal_device_get_mtl_device(dev);
    if (device == nil) {
        return NULL;
    }
    return [device newBufferWithBytesNoCopy:ptr length:size options:MTLResourceStorageModeShared deallocator:nil];
}

void test_metal_buffer_release(void * metal) {
    [(id<MTLBuffer>) metal release];
}

size_t test_metal_buffer_retain_count(void * metal) {
    return [(id<MTLBuffer>) metal retainCount];
}
