#pragma once

// Command buffers and shared events a test makes itself, for the tests of encoding into caller command buffers
// (ggml_backend_metal_encode_nodes). Plain C, so that C++ tests can use them; metal-command-buffer.mm implements them.

#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// a command buffer with unretained references from `queue`, an id<MTLCommandQueue> as void *, owned by the caller
// (one reference); NULL on failure
void * test_metal_command_buffer_new(void * queue);

void test_metal_command_buffer_commit(void * cmd_buf);
void test_metal_command_buffer_wait (void * cmd_buf);

// true once the command buffer has been enqueued or committed
bool test_metal_command_buffer_enqueued (void * cmd_buf);
bool test_metal_command_buffer_completed(void * cmd_buf);

// makes the GPU wait, before anything encoded after the call, until `event` reaches `value`
void test_metal_command_buffer_encode_wait(void * cmd_buf, void * event, uint64_t value);

// drops the caller's reference
void test_metal_command_buffer_release(void * cmd_buf);

// an id<MTLSharedEvent> of the Metal device `dev`, as void *, owned by the caller (one reference); NULL on failure
void * test_metal_shared_event_new(ggml_backend_dev_t dev);

// sets the event's value from the host
void test_metal_shared_event_signal(void * event, uint64_t value);

void test_metal_shared_event_release(void * event);

#ifdef __cplusplus
}
#endif
