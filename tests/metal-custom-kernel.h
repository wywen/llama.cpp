#pragma once

// A Metal kernel a test compiles from source at run time, for the tests of GGML_OP_CUSTOM implementations registered
// with ggml_backend_dev_add_custom_op. Plain C, so that C++ tests can use it; metal-custom-kernel.mm
// implements it.

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// the id<MTLComputePipelineState> of a kernel that writes dst[i] = src[i] + 1 for contiguous F32 tensors, compiled
// from source for the Metal device `dev`, as void *, owned by the caller (one reference); NULL on failure
void * test_metal_add_one_pipeline_new(ggml_backend_dev_t dev);

void test_metal_pipeline_release(void * pipeline);

// encodes `node` = src[0] + 1 into `encoder`, an id<MTLComputeCommandEncoder> as void *, with `pipeline` from
// test_metal_add_one_pipeline_new; false when the node is not two contiguous F32 tensors of the same size
bool test_metal_add_one_encode(void * pipeline, const struct ggml_tensor * node, void * encoder);

#ifdef __cplusplus
}
#endif
