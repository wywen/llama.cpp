#pragma once

// Metal buffers over host memory for the tests of ggml_backend_metal_buffer_from_ranges, which take the caller's
// Metal buffers. Plain C, so that C++ tests can use them; metal-nocopy-buffer.mm implements them.

#include "ggml-backend.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// a shared-storage no-copy id<MTLBuffer> of the Metal device `dev` over [ptr, ptr + size), as void *, owned by the
// caller (one reference); ptr and size must be page aligned. NULL on failure
void * test_metal_nocopy_buffer_new(ggml_backend_dev_t dev, void * ptr, size_t size);

// drops the caller's reference
void test_metal_buffer_release(void * metal);

// the buffer's reference count, for checking that the backend takes none
size_t test_metal_buffer_retain_count(void * metal);

#ifdef __cplusplus
}
#endif
