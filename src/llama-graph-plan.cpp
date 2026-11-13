#include "llama-ext.h"

#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-memory-alloc.h"
#include "llama-model.h"

#include <memory>
#include <stdexcept>
#include <vector>

struct llama_graph_plan {
    // members in order: a member's ctx_other is an earlier one, and outlives it
    std::vector<std::unique_ptr<llama_context>> contexts;
    std::vector<std::vector<llm_graph_result_ptr>> graphs;

    ~llama_graph_plan() {
        // the graphs read the contexts' memory, and a context reads the memory of an earlier one
        graphs.clear();
        while (!contexts.empty()) {
            contexts.pop_back();
        }
    }
};

static llm_graph_result_ptr llama_graph_plan_shape_build(llama_context & ctx, const llama_graph_plan_shape & shape) {
    llama_memory_context_ptr mctx;
    if (llama_memory_i * memory = ctx.get_memory()) {
        mctx = shape.n_cells == 0 ? memory->init_full() : memory->init_full_extent(shape.n_cells);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            throw std::runtime_error(format("the memory cannot build a graph over %u cells", shape.n_cells));
        }
    }

    uint32_t n_tokens  = shape.n_tokens;
    uint32_t n_seqs    = shape.n_seqs;
    uint32_t n_outputs = shape.n_outputs;
    switch (shape.width) {
        case LLAMA_GRAPH_PLAN_WIDTH_UBATCH:
            break;
        case LLAMA_GRAPH_PLAN_WIDTH_WIDEST:
            {
                const auto widest = ctx.graph_shape_widest();
                n_tokens  = widest[0];
                n_seqs    = widest[1];
                n_outputs = widest[2];
            } break;
        default:
            throw std::runtime_error("unknown graph width");
    }
    if (n_tokens == 0 || n_seqs == 0 || n_outputs == 0) {
        throw std::runtime_error("a graph needs at least one token, sequence and output");
    }

    ctx.set_nextn_layer_offset(shape.nextn_layer_offset);
    llm_graph_result_ptr res = ctx.graph_plan(n_tokens, n_seqs, n_outputs, mctx.get(), shape.moe_slot_set);
    if (!res) {
        throw std::runtime_error(format("failed to build or rewrite the graph of %u tokens, %u sequences, %u outputs", n_tokens, n_seqs, n_outputs));
    }
    return res;
}

llama_graph_plan * llama_graph_plan_build(const std::vector<llama_graph_plan_member> & members) {
    // the planned contexts replay the construction logs; leave them to the real construction
    const llama_log_thread_quiet log_quiet;

    auto plan = std::make_unique<llama_graph_plan>();
    plan->contexts.reserve(members.size());
    plan->graphs.reserve(members.size());

    for (size_t i = 0; i < members.size(); ++i) {
        const llama_graph_plan_member & member = members[i];
        if (!member.model) {
            throw std::runtime_error("model cannot be NULL");
        }
        if (member.other >= (int32_t) i) {
            throw std::runtime_error(format("member %zu reads member %d, which is not an earlier one", i, member.other));
        }

        llama_context_params params = member.params;
        params.ctx_other = member.other >= 0 ? plan->contexts[member.other].get() : nullptr;

        params.cb_eval                      = nullptr;
        params.cb_eval_user_data            = nullptr;
        params.cb_reserve                   = nullptr;
        params.cb_reserve_user_data         = nullptr;
        params.cb_graph_prepare             = nullptr;
        params.cb_graph_prepare_user_data   = nullptr;
        params.cb_outputs_enqueued          = nullptr;
        params.sched_caller_alloc           = nullptr;
        params.cb_outputs_enqueued_user_data = nullptr;
        params.abort_callback               = nullptr;
        params.abort_callback_data          = nullptr;

        {
            // the memory, the MoE slots and the scheduler reserves of the context allocate nothing while planning
            llama_memory_alloc_recorder recorder(llama_memory_alloc_mode::plan);
            plan->contexts.emplace_back(new llama_context(*member.model, params));
        }

        llama_context & ctx = *plan->contexts.back();
        std::vector<llm_graph_result_ptr> & graphs = plan->graphs.emplace_back();
        graphs.reserve(member.shapes.size());
        for (const llama_graph_plan_shape & shape : member.shapes) {
            graphs.push_back(llama_graph_plan_shape_build(ctx, shape));
        }
        ctx.set_nextn_layer_offset(0);
    }

    return plan.release();
}

ggml_cgraph * llama_graph_plan_graph(const llama_graph_plan * plan, size_t member, size_t shape) {
    GGML_ASSERT(plan && member < plan->graphs.size() && shape < plan->graphs[member].size());
    return plan->graphs[member][shape]->get_gf();
}

const llama_context * llama_graph_plan_context(const llama_graph_plan * plan, size_t member) {
    GGML_ASSERT(plan && member < plan->contexts.size());
    return plan->contexts[member].get();
}

void llama_graph_plan_free(llama_graph_plan * plan) {
    delete plan;
}
