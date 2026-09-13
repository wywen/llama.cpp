#pragma once

#include "llama-ext.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <vector>

struct ggml_context;

// What llama_memory_alloc_buffer does with the buffers it records while a recorder is installed.
enum class llama_memory_alloc_mode {
    plan,    // back every buffer with a zero-size placeholder
    observe, // back every buffer as it would be without a recorder
};

// True while a planning recorder is installed on this thread: memory being built must not modify live state,
// and backs its buffers with zero-size placeholders as if the model were loaded without allocation.
bool llama_memory_alloc_planning();

// Records every buffer llama_memory_alloc_buffer backs on the installing thread while in scope.
// Other threads are unaffected. Recorders nest: the innermost one receives the buffers.
class llama_memory_alloc_recorder {
public:
    explicit llama_memory_alloc_recorder(llama_memory_alloc_mode mode);
    ~llama_memory_alloc_recorder();

    llama_memory_alloc_recorder(const llama_memory_alloc_recorder &) = delete;
    llama_memory_alloc_recorder & operator=(const llama_memory_alloc_recorder &) = delete;

    // the buffers recorded so far, in allocation order
    const std::vector<llama_memory_plan_buffer> & buffers() const { return recorded; }

    std::vector<llama_memory_plan_buffer> release() { return std::move(recorded); }

private:
    friend ggml_backend_buffer_ptr llama_memory_alloc_buffer(
            ggml_context *, ggml_backend_buffer_type_t, bool, llama_memory_plan_buffer_kind, const char *);
    friend bool llama_memory_alloc_planning();

    const llama_memory_alloc_mode mode;

    llama_memory_alloc_recorder * outer;

    std::vector<llama_memory_plan_buffer> recorded;
};

// Backs the storage-owning tensors of a memory module's ggml context with a buffer of type `buft`, as
// llama_memory_bufs::alloc_buf does, and records the buffer with the installed recorder, if any.
// `no_alloc` selects a zero-size placeholder; callers pass true while planning (see llama_memory_bufs).
// Throws std::runtime_error when the allocation fails.
ggml_backend_buffer_ptr llama_memory_alloc_buffer(
        ggml_context                  * ctx,
        ggml_backend_buffer_type_t      buft,
        bool                            no_alloc,
        llama_memory_plan_buffer_kind   kind,
        const char                    * name);
