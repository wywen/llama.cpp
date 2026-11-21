#include "metal-custom-kernel.h"

#include "ggml-metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>

#if __has_feature(objc_arc)
#error "metal-custom-kernel.mm counts references by hand: build it without ARC"
#endif

static const char * k_add_one_source =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void kernel_add_one(\n"
    "        device const float * src [[buffer(0)]],\n"
    "        device       float * dst [[buffer(1)]],\n"
    "        constant     uint  & n   [[buffer(2)]],\n"
    "        uint i [[thread_position_in_grid]]) {\n"
    "    if (i < n) {\n"
    "        dst[i] = src[i] + 1.0f;\n"
    "    }\n"
    "}\n";

void * test_metal_add_one_pipeline_new(ggml_backend_dev_t dev) {
    id<MTLDevice> device = (id<MTLDevice>) ggml_backend_metal_device_get_mtl_device(dev);
    if (device == nil) {
        return NULL;
    }

    @autoreleasepool {
        NSError * error = nil;

        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:k_add_one_source]
                                                      options:nil
                                                        error:&error];
        if (library == nil) {
            fprintf(stderr, "%s: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"kernel_add_one"];
        [library release];
        if (function == nil) {
            fprintf(stderr, "%s: no kernel_add_one in the library\n", __func__);
            return NULL;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (pipeline == nil) {
            fprintf(stderr, "%s: %s\n", __func__, [[error description] UTF8String]);
            return NULL;
        }

        return pipeline;
    }
}

void test_metal_pipeline_release(void * pipeline) {
    [(id<MTLComputePipelineState>) pipeline release];
}

bool test_metal_add_one_encode(void * pipeline, const struct ggml_tensor * node, void * encoder) {
    const struct ggml_tensor * src = node->src[0];
    if (src == NULL || src->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(node) || ggml_nelements(src) != ggml_nelements(node)) {
        return false;
    }

    id<MTLComputePipelineState> pso = (id<MTLComputePipelineState>) pipeline;
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>) encoder;

    size_t offs_src = 0;
    size_t offs_dst = 0;
    id<MTLBuffer> buf_src = (id<MTLBuffer>) ggml_backend_metal_tensor_mtl_buffer(src,  &offs_src);
    id<MTLBuffer> buf_dst = (id<MTLBuffer>) ggml_backend_metal_tensor_mtl_buffer(node, &offs_dst);

    const uint32_t n  = (uint32_t) ggml_nelements(node);
    const NSUInteger max_nt = pso.maxTotalThreadsPerThreadgroup;
    const uint32_t   nt     = (uint32_t) (max_nt < 256 ? max_nt : 256);

    [enc setComputePipelineState:pso];
    [enc setBuffer:buf_src offset:offs_src atIndex:0];
    [enc setBuffer:buf_dst offset:offs_dst atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((n + nt - 1)/nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(nt, 1, 1)];

    return true;
}
