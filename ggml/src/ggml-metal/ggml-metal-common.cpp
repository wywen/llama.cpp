#include "ggml-metal-common.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-metal.h"
#include "ggml.h"

#include <algorithm>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

void ggml_metal_projection_free(struct ggml_metal_projection * projection) {
    if (!projection) {
        return;
    }
    free(projection->segments);
    free(projection->node_scratch);
    *projection = {};
}

enum ggml_status ggml_metal_project_segments(const struct ggml_cgraph *                  graph,
                                             const struct ggml_metal_projection_config * config,
                                             struct ggml_metal_projection *              out) {
    if (!out) {
        return GGML_STATUS_FAILED;
    }
    out->n_segments = 0;
    if (!graph || !config || config->n_cuts < 0 || config->n_waits < 0 || (config->n_cuts && !config->cut_nodes) ||
        (config->n_waits && !config->wait_nodes) || (config->n_out_of_band && !config->out_of_band)) {
        return GGML_STATUS_FAILED;
    }
    const int n_nodes = graph->n_nodes;
    if (n_nodes <= 0 || !graph->nodes || graph->size < 0 || (graph->size > 0 && n_nodes > graph->size) ||
        n_nodes > (INT_MAX - 2) / 2 || (size_t) n_nodes > SIZE_MAX / sizeof(*out->segments)) {
        return GGML_STATUS_FAILED;
    }
    for (int i = 0; i < config->n_cuts; ++i) {
        if (!config->cut_nodes[i]) {
            return GGML_STATUS_FAILED;
        }
    }
    for (int i = 0; i < config->n_waits; ++i) {
        if (!config->wait_nodes[i]) {
            return GGML_STATUS_FAILED;
        }
    }
    for (size_t i = 0; i < config->n_out_of_band; ++i) {
        if (!config->out_of_band[i]) {
            return GGML_STATUS_FAILED;
        }
    }

    int first = -1;
    int last  = -1;
    if (n_nodes > out->capacity_nodes) {
        if ((size_t) n_nodes > SIZE_MAX / sizeof(*out->node_scratch)) {
            return GGML_STATUS_FAILED;
        }
        void * scratch = realloc(out->node_scratch, (size_t) n_nodes * sizeof(*out->node_scratch));
        if (!scratch) {
            return GGML_STATUS_ALLOC_FAILED;
        }
        out->node_scratch   = (const struct ggml_tensor **) scratch;
        out->capacity_nodes = n_nodes;
    }
    for (int i = 0; i < n_nodes; ++i) {
        const struct ggml_tensor * node = graph->nodes[i];
        if (!node) {
            return GGML_STATUS_FAILED;
        }
        out->node_scratch[i] = node;
        if (node == config->first_node) {
            first = i;
        }
        if (node == config->last_node) {
            last = i;
        }
    }
    std::sort(out->node_scratch, out->node_scratch + n_nodes, std::less<const struct ggml_tensor *>());
    if (std::adjacent_find(out->node_scratch, out->node_scratch + n_nodes) != out->node_scratch + n_nodes) {
        return GGML_STATUS_FAILED;
    }
    if (first >= 0 && last >= 0 && last <= first) {
        return GGML_STATUS_FAILED;
    }

    if (n_nodes > out->capacity_segments) {
        void * segments = realloc(out->segments, (size_t) n_nodes * sizeof(*out->segments));
        if (!segments) {
            return GGML_STATUS_ALLOC_FAILED;
        }
        out->segments          = (struct ggml_metal_segment *) segments;
        out->capacity_segments = n_nodes;
    }
    struct ggml_metal_segment * segments       = out->segments;
    const bool                  window_applies = config->window_active && (first >= 0 || last >= 0) &&
                                                 (config->first_node != nullptr || config->last_node != nullptr);
    const int                   window_lo      = first >= 0 ? first + 1 : 0;
    const int                   window_hi      = last >= 0 ? last : n_nodes - 1;
    int                         start          = 0;
    int                         count          = 0;
    bool                        previous_cut   = false;
    for (int i = 0; i <= n_nodes; ++i) {
        bool wait = false;
        bool cut  = false;
        if (i < n_nodes) {
            const struct ggml_tensor * node = graph->nodes[i];
            for (int w = 0; w < config->n_waits; ++w) {
                if (config->wait_nodes[w] == node) {
                    wait = true;
                    break;
                }
            }
            for (int c = 0; c < config->n_cuts; ++c) {
                if (config->cut_nodes[c] == node) {
                    cut = true;
                    break;
                }
            }
        }
        if (i > start && (i == n_nodes || previous_cut || wait)) {
            segments[count]       = {};
            segments[count].start = start;
            segments[count].end   = i;
            ++count;
            start = i;
        }
        previous_cut = cut;
    }
    bool entered = false;
    for (int s = 0; s < count; ++s) {
        struct ggml_metal_segment * seg = &segments[s];
        seg->skipped                    = window_applies && (seg->end <= window_lo || seg->start > window_hi);
        if (seg->skipped) {
            continue;
        }
        seg->window_first               = window_applies && !entered;
        seg->window_last                = window_applies && seg->start <= window_hi && seg->end > window_hi;
        seg->ingress_here               = seg->window_first && first >= 0;
        seg->egress_here                = seg->window_last && last >= 0;
        entered                         = true;
        const struct ggml_tensor * head = graph->nodes[seg->start];
        for (int w = 0; w < config->n_waits; ++w) {
            if (config->wait_nodes[w] == head) {
                seg->wait_at_start = true;
                break;
            }
        }
        for (int i = seg->start; i < seg->end; ++i) {
            const struct ggml_tensor * node = graph->nodes[i];
            for (size_t k = 0; k < config->n_out_of_band; ++k) {
                if (config->out_of_band[k] == node) {
                    return GGML_STATUS_FAILED;
                }
            }
        }
    }
    out->n_segments = count;
    return GGML_STATUS_SUCCESS;
}

// the per-layer output tensors are named "l_out-<il>" by the graph builder, and that name
//   is what a boundary-scheduling caller resolves its cut nodes and window endpoints by
//
// matching the name is not elegant, but at optimize time it is the only handle there is:
//   the schedule that would otherwise name the cuts is installed later (see the header),
//   and a tensor carries no field a caller could mark ahead of time without also owning
//   the graph build. keeping the match here means there is exactly one string to change
static const char GGML_GRAPH_BOUNDARY_MARKER_PREFIX[] = "l_out-";

bool ggml_graph_node_is_boundary_marker(const struct ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }

    const size_t len = sizeof(GGML_GRAPH_BOUNDARY_MARKER_PREFIX) - 1;

    const char * name = ggml_get_name(node);

    if (strncmp(name, GGML_GRAPH_BOUNDARY_MARKER_PREFIX, len) != 0) {
        return false;
    }

    // the suffix is the layer index: require at least one digit and nothing else, so that
    //   only the generated per-layer names match
    const char * suffix = name + len;

    if (*suffix == '\0') {
        return false;
    }

    for (const char * p = suffix; *p != '\0'; p++) {
        if (*p < '0' || *p > '9') {
            return false;
        }
    }

    return true;
}

bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne11 = op->src[1]->ne[1];

    return !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne21 = op->src[2]->ne[1];

    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes, bool stop_at_boundary_markers) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_add_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        // keep track of the sources of the fused nodes as well
        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_add_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        return ggml_mem_ranges_add_dst(mrs, node.dst());
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_check_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_check_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        return ggml_mem_ranges_check_dst(mrs, node.dst());
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_CLAMP:
            case GGML_OP_TRI:
            case GGML_OP_DIAG:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_SCALE:
            case GGML_OP_UNARY:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_REPEAT:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    // helper to check if a node is a segment boundary that nothing may be moved across
    //
    // fusion never reorders anything by itself, but it does make a marker travel with the
    //   group it was packed into, so a group is a barrier when ANY of its nodes is one
    const auto & h_barrier = [stop_at_boundary_markers](const node_info & node) {
        if (!stop_at_boundary_markers) {
            return false;
        }

        if (ggml_graph_node_is_boundary_marker(node.node)) {
            return true;
        }

        for (const auto * fused : node.fused) {
            if (ggml_graph_node_is_boundary_marker(fused)) {
                return true;
            }
        }

        return false;
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            // every node this scan selects is emitted BEFORE node0, so when node0 is a
            //   boundary marker there is nothing it may legally pick up
            const int i1_end = h_barrier(node0) ? i0 + 1 : (i0 + N_FORWARD < n ? i0 + N_FORWARD : n);

            for (int i1 = i0 + 1; i1 < i1_end; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // never hoist a node from beyond a boundary marker, nor the marker itself
                if (h_barrier(node1)) {
                    break;
                }

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

void ggml_graph_optimize(ggml_cgraph * gf, bool stop_at_boundary_markers) {
    constexpr int MAX_FUSE = 16;

    const int n = gf->n_nodes;

    enum ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    //
    // the packing itself moves nothing: a group is contiguous, and the unfuse below writes
    //   its members back in their original relative order. so a group may straddle a
    //   boundary marker without any work crossing it - all the marker has to do is make
    //   the group it landed in immovable, which is what h_barrier above does
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes, stop_at_boundary_markers);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}

struct ggml_metal_completion {
    struct slot {
        uint64_t identity;
        bool     done;
        size_t   generation_index;
        uint32_t split;
        uint32_t first_ordinal;
        uint32_t end_ordinal;
    };

    struct receipt_node {
        ggml_metal_receipt_node entry;
        bool                    registered;
    };

    struct receipt_generation {
        uint64_t                          generation;
        std::vector<receipt_node>         nodes;
        ggml_metal_terminal_receipt_fn    terminal;
        ggml_metal_generation_quiesced_fn quiesced;
        void *                            cookie;
        size_t                            pending;
        bool                              closed;
        bool                              notified;
    };

    struct cut {
        ggml_metal_completion_cut entry;
    };

    std::mutex                      mutex;
    std::condition_variable         quiescent;
    std::vector<slot>               slots;
    std::vector<cut>                cuts;
    std::vector<receipt_generation> generations;
    uint64_t                        next_generation = 1;
    bool                            receipt_mode    = false;
    size_t                          pending         = 0;
    uint64_t                        next_identity   = 1;
    bool                            batch_open      = false;
    bool                            failed          = false;
    ggml_metal_completion_publish   publish;
    ggml_metal_completion_latch     latch;
    ggml_metal_completion_event_ref retain_event;
    ggml_metal_completion_event_ref release_event;
    void *                          user;
};

static ggml_metal_completion::receipt_node * ggml_metal_receipt_find(ggml_metal_completion::receipt_generation & gen,
                                                                     const ggml_tensor *                         node) {
    auto it = std::lower_bound(gen.nodes.begin(), gen.nodes.end(), node,
                               [](const ggml_metal_completion::receipt_node & entry, const ggml_tensor * key) {
                                   return std::less<const ggml_tensor *>{}(entry.entry.node, key);
                               });
    return it != gen.nodes.end() && it->entry.node == node ? &*it : nullptr;
}

static void ggml_metal_receipt_notify(ggml_metal_completion::receipt_generation & gen) {
    if (gen.closed && !gen.pending && !gen.notified) {
        gen.notified = true;
        gen.quiesced(gen.cookie, gen.generation);
    }
}

static void ggml_metal_completion_fail_locked(ggml_metal_completion * tracker) {
    if (tracker->failed) {
        return;
    }
    tracker->failed = true;
    tracker->latch(tracker->user);
    for (size_t i = 0; i < tracker->cuts.size(); ++i) {
        void * event = tracker->cuts[i].entry.event;
        if (!event) {
            continue;
        }
        bool already_woken = false;
        for (size_t j = 0; j < i; ++j) {
            already_woken |= tracker->cuts[j].entry.event == event;
        }
        if (!already_woken) {
            tracker->publish(tracker->user, event, UINT64_MAX);
        }
    }
}

ggml_metal_completion * ggml_metal_completion_new(ggml_metal_completion_publish   publish,
                                                  ggml_metal_completion_latch     latch,
                                                  ggml_metal_completion_event_ref retain_event,
                                                  ggml_metal_completion_event_ref release_event,
                                                  void *                          user) {
    if (!publish || !latch || (retain_event == nullptr) != (release_event == nullptr)) {
        return nullptr;
    }
    auto * tracker = new (std::nothrow) ggml_metal_completion;
    if (tracker) {
        tracker->publish       = publish;
        tracker->latch         = latch;
        tracker->retain_event  = retain_event;
        tracker->release_event = release_event;
        tracker->user          = user;
    }
    return tracker;
}

void ggml_metal_completion_free(ggml_metal_completion * tracker) {
    if (!tracker) {
        return;
    }
    ggml_metal_completion_wait(tracker);
    if (tracker->release_event) {
        for (const auto & cut : tracker->cuts) {
            tracker->release_event(cut.entry.event);
        }
    }
    delete tracker;
}

bool ggml_metal_completion_reserve_additional(ggml_metal_completion * tracker, size_t n) {
    std::lock_guard<std::mutex> lock(tracker->mutex);
    if (tracker->failed || !tracker->batch_open || n > SIZE_MAX - tracker->slots.size()) {
        return false;
    }
    const size_t required = tracker->slots.size() + n;
    if (required <= tracker->slots.capacity()) {
        return true;
    }
    const size_t capacity = tracker->slots.capacity();
    const size_t target   = std::max(required, capacity <= SIZE_MAX / 2 ? capacity * 2 : required);
    try {
        tracker->slots.reserve(target);
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

enum ggml_status ggml_metal_receipts_begin(ggml_metal_completion *                tracker,
                                           const ggml_metal_receipt_node *        expected,
                                           size_t                                 n_expected,
                                           const ggml_metal_receipt_binding_cut * cuts,
                                           size_t                                 n_cuts,
                                           ggml_metal_terminal_receipt_fn         terminal,
                                           ggml_metal_generation_quiesced_fn      quiesced,
                                           void *                                 cookie,
                                           uint64_t *                             generation) {
    if (!tracker || !expected || !n_expected || n_expected > INT_MAX || !cuts || !n_cuts || !terminal || !quiesced ||
        !generation) {
        return GGML_STATUS_FAILED;
    }
    std::lock_guard<std::mutex> lock(tracker->mutex);
    if (tracker->failed || tracker->batch_open || tracker->next_generation == UINT64_MAX) {
        return GGML_STATUS_FAILED;
    }
    ggml_metal_completion::receipt_generation gen = {
        tracker->next_generation, {}, terminal, quiesced, cookie, 0, false, false
    };
    const bool   recycle  = tracker->pending == 0;
    const size_t old_size = recycle ? 0 : tracker->cuts.size();
    if (n_cuts > SIZE_MAX - old_size) {
        return GGML_STATUS_FAILED;
    }
    try {
        for (size_t i = 0; i < n_expected; ++i) {
            const auto & node = expected[i];
            if (!node.node || (node.flags != GGML_METAL_RECEIPT_SELECTED && node.flags != GGML_METAL_RECEIPT_SKIPPED) ||
                (i && (node.split < expected[i - 1].split ||
                       (node.split == expected[i - 1].split && node.ordinal != expected[i - 1].ordinal + 1))) ||
                ((!i || node.split != expected[i - 1].split) && node.ordinal != 0)) {
                return GGML_STATUS_FAILED;
            }
        }
        for (size_t i = 0; i < n_cuts; ++i) {
            const auto &                    cut  = cuts[i];
            const ggml_metal_receipt_node * node = nullptr;
            for (size_t j = 0; j < n_expected; ++j) {
                if (expected[j].node == cut.native.node) {
                    node = &expected[j];
                    break;
                }
            }
            if (!node || node->split != cut.split || node->ordinal != cut.ordinal ||
                node->flags != GGML_METAL_RECEIPT_SELECTED || !cut.native.event || cut.native.value == UINT64_MAX) {
                return GGML_STATUS_FAILED;
            }
            for (size_t j = 0; j < i; ++j) {
                if (cuts[j].native.event == cut.native.event && cuts[j].native.value >= cut.native.value) {
                    return GGML_STATUS_FAILED;
                }
            }
            if (!recycle) {
                for (const auto & old : tracker->cuts) {
                    if (old.entry.event == cut.native.event && old.entry.value >= cut.native.value) {
                        return GGML_STATUS_FAILED;
                    }
                }
            }
        }
        tracker->generations.reserve((recycle ? 0 : tracker->generations.size()) + 1);
        tracker->cuts.reserve(old_size + n_cuts);
        const size_t slot_base = recycle ? 0 : tracker->slots.size();
        if (n_expected > SIZE_MAX - slot_base) {
            return GGML_STATUS_FAILED;
        }
        const size_t required_slots = slot_base + n_expected;
        if (required_slots > tracker->slots.capacity()) {
            const size_t capacity = tracker->slots.capacity();
            tracker->slots.reserve(std::max(required_slots, capacity <= SIZE_MAX / 2 ? capacity * 2 : required_slots));
        }
        if (recycle && !tracker->generations.empty()) {
            gen.nodes.swap(tracker->generations.back().nodes);
        }
        gen.nodes.clear();
        gen.nodes.reserve(n_expected);
        for (size_t i = 0; i < n_expected; ++i) {
            gen.nodes.push_back({ expected[i], false });
        }
        std::sort(gen.nodes.begin(), gen.nodes.end(),
                  [](const ggml_metal_completion::receipt_node & a, const ggml_metal_completion::receipt_node & b) {
                      return std::less<const ggml_tensor *>{}(a.entry.node, b.entry.node);
                  });
        for (size_t i = 1; i < gen.nodes.size(); ++i) {
            if (gen.nodes[i - 1].entry.node == gen.nodes[i].entry.node) {
                if (recycle && !tracker->generations.empty()) {
                    gen.nodes.swap(tracker->generations.back().nodes);
                }
                return GGML_STATUS_FAILED;
            }
        }
        if (recycle) {
            if (tracker->release_event) {
                for (const auto & old : tracker->cuts) {
                    tracker->release_event(old.entry.event);
                }
            }
            tracker->cuts.clear();
            tracker->generations.clear();
            tracker->slots.clear();
        }
        tracker->generations.push_back(std::move(gen));
        for (size_t i = 0; i < n_cuts; ++i) {
            tracker->cuts.push_back({ cuts[i].native });
            if (tracker->retain_event) {
                tracker->retain_event(cuts[i].native.event);
            }
        }
    } catch (const std::exception &) {
        if (recycle && !tracker->generations.empty() && gen.nodes.capacity() != 0) {
            gen.nodes.swap(tracker->generations.back().nodes);
        }
        return GGML_STATUS_ALLOC_FAILED;
    }
    tracker->batch_open   = true;
    tracker->receipt_mode = true;
    *generation           = tracker->next_generation++;
    return GGML_STATUS_SUCCESS;
}

const void * ggml_metal_receipts_node_storage(ggml_metal_completion * tracker, size_t * capacity) {
    if (!tracker || !capacity) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(tracker->mutex);
    if (tracker->generations.empty()) {
        *capacity = 0;
        return nullptr;
    }
    const auto & nodes = tracker->generations.back().nodes;
    *capacity          = nodes.capacity();
    return nodes.data();
}

bool ggml_metal_receipts_active(ggml_metal_completion * tracker) {
    std::lock_guard<std::mutex> lock(tracker->mutex);
    return tracker->receipt_mode && tracker->batch_open && !tracker->failed;
}

bool ggml_metal_receipts_validate_view(ggml_metal_completion *       tracker,
                                       const ggml_cgraph *           graph,
                                       const ggml_metal_projection * projection,
                                       uint32_t *                    split) {
    if (!tracker || !graph || graph->n_nodes <= 0 || !graph->nodes || !projection || !split) {
        return false;
    }
    std::lock_guard<std::mutex> lock(tracker->mutex);
    if (!tracker->receipt_mode || !tracker->batch_open || tracker->failed || !projection->n_segments) {
        return false;
    }
    auto &                                      gen      = tracker->generations.back();
    const ggml_metal_completion::receipt_node * previous = nullptr;
    int                                         cursor   = 0;
    for (int s = 0; s < projection->n_segments; ++s) {
        const auto & seg = projection->segments[s];
        if (seg.start != cursor || seg.end <= cursor || seg.end > graph->n_nodes) {
            return false;
        }
        for (int i = seg.start; i < seg.end; ++i) {
            const auto * node = ggml_metal_receipt_find(gen, graph->nodes[i]);
            if (!node || node->registered ||
                node->entry.flags != (seg.skipped ? GGML_METAL_RECEIPT_SKIPPED : GGML_METAL_RECEIPT_SELECTED) ||
                (previous &&
                 (node->entry.split != previous->entry.split || node->entry.ordinal != previous->entry.ordinal + 1))) {
                return false;
            }
            previous = node;
        }
        cursor = seg.end;
    }
    if (cursor != graph->n_nodes) {
        return false;
    }
    for (int s = 0; s < projection->n_segments; ++s) {
        const auto & seg = projection->segments[s];
        if (seg.skipped) {
            for (int i = seg.start; i < seg.end; ++i) {
                ggml_metal_receipt_find(gen, graph->nodes[i])->registered = true;
            }
        }
    }
    *split = previous->entry.split;
    return true;
}

bool ggml_metal_receipts_register(ggml_metal_completion * tracker,
                                  uint32_t                split,
                                  ggml_tensor * const *   nodes,
                                  int                     start,
                                  int                     end,
                                  uint64_t *              physical_id) {
    if (!tracker || !nodes || start < 0 || end <= start || !physical_id) {
        return false;
    }
    std::lock_guard<std::mutex> lock(tracker->mutex);
    if (!tracker->receipt_mode || !tracker->batch_open || tracker->failed ||
        tracker->slots.size() == tracker->slots.capacity() || tracker->next_identity == UINT64_MAX) {
        return false;
    }
    auto &   gen   = tracker->generations.back();
    uint32_t first = 0;
    for (int i = start; i < end; ++i) {
        const auto * node = ggml_metal_receipt_find(gen, nodes[i]);
        if (!node || node->registered || node->entry.flags != GGML_METAL_RECEIPT_SELECTED ||
            node->entry.split != split || (i != start && node->entry.ordinal != first + (uint32_t) (i - start))) {
            return false;
        }
        if (i == start) {
            first = node->entry.ordinal;
        }
    }
    if ((uint64_t) first + (uint64_t) (end - start) > UINT32_MAX) {
        return false;
    }
    *physical_id = tracker->next_identity++;
    tracker->slots.push_back(
        { *physical_id, false, tracker->generations.size() - 1, split, first, first + (uint32_t) (end - start) });
    for (int i = start; i < end; ++i) {
        ggml_metal_receipt_find(gen, nodes[i])->registered = true;
    }
    ++gen.pending;
    ++tracker->pending;
    return true;
}

enum ggml_status ggml_metal_receipts_finish(ggml_metal_completion * tracker,
                                            uint64_t                generation,
                                            enum ggml_status        submission_status) {
    if (!tracker) {
        return GGML_STATUS_FAILED;
    }
    std::lock_guard<std::mutex> lock(tracker->mutex);
    if (!tracker->receipt_mode || !tracker->batch_open || tracker->generations.back().generation != generation) {
        ggml_metal_completion_fail_locked(tracker);
        return GGML_STATUS_FAILED;
    }
    tracker->batch_open = false;
    auto & gen          = tracker->generations.back();
    gen.closed          = true;
    bool covered        = submission_status == GGML_STATUS_SUCCESS;
    for (const auto & node : gen.nodes) {
        if (node.entry.flags == GGML_METAL_RECEIPT_SELECTED && !node.registered) {
            covered = false;
        }
    }
    if (!covered) {
        ggml_metal_completion_fail_locked(tracker);
    }
    ggml_metal_receipt_notify(gen);
    return submission_status != GGML_STATUS_SUCCESS ? submission_status :
           tracker->failed                          ? GGML_STATUS_FAILED :
                                                      GGML_STATUS_SUCCESS;
}

bool ggml_metal_completion_complete(ggml_metal_completion * tracker, uint64_t identity, bool success) {
    std::lock_guard<std::mutex> lock(tracker->mutex);
    for (auto & slot : tracker->slots) {
        if (slot.identity != identity) {
            continue;
        }
        if (slot.done) {
            return false;
        }
        slot.done = true;
        if (!success) {
            ggml_metal_completion_fail_locked(tracker);
        }
        auto & gen = tracker->generations[slot.generation_index];
        gen.terminal(gen.cookie, gen.generation, slot.identity, slot.split, slot.first_ordinal, slot.end_ordinal,
                     success ? GGML_METAL_TERMINAL_COMPLETED : GGML_METAL_TERMINAL_FAILED);
        --gen.pending;
        --tracker->pending;
        ggml_metal_receipt_notify(gen);
        tracker->quiescent.notify_all();
        return true;
    }
    return false;
}

void ggml_metal_completion_fail(ggml_metal_completion * tracker) {
    std::lock_guard<std::mutex> lock(tracker->mutex);
    ggml_metal_completion_fail_locked(tracker);
}

void ggml_metal_completion_wait(ggml_metal_completion * tracker) {
    std::unique_lock<std::mutex> lock(tracker->mutex);
    tracker->quiescent.wait(lock, [tracker] { return tracker->pending == 0; });
}

bool ggml_metal_completion_failed(ggml_metal_completion * tracker) {
    std::lock_guard<std::mutex> lock(tracker->mutex);
    return tracker->failed;
}

struct ggml_metal_result_storage {
    std::vector<std::unique_ptr<ggml_metal_seg_result[]>> slabs;
    std::vector<ggml_metal_seg_result *>                  slots;
};

ggml_metal_result_storage * ggml_metal_result_storage_new(void) {
    return new (std::nothrow) ggml_metal_result_storage;
}

void ggml_metal_result_storage_free(ggml_metal_result_storage * storage) {
    delete storage;
}

bool ggml_metal_result_storage_reserve(ggml_metal_result_storage * storage, size_t n) {
    if (!storage || n > INT_MAX || n > SIZE_MAX / sizeof(ggml_metal_seg_result)) {
        return false;
    }
    const size_t count = storage->slots.size();
    if (n <= count) {
        return true;
    }
    const size_t target     = std::min((size_t) INT_MAX, std::max(n, count <= INT_MAX / 2 ? count * 2 : n));
    const size_t additional = target - count;
    std::unique_ptr<ggml_metal_seg_result[]> slab(new (std::nothrow) ggml_metal_seg_result[additional]);
    if (!slab) {
        return false;
    }
    try {
        storage->slots.reserve(target);
        storage->slabs.reserve(storage->slabs.size() + 1);
    } catch (const std::exception &) {
        return false;
    }
    ggml_metal_seg_result * ptr = slab.get();
    storage->slabs.push_back(std::move(slab));
    for (size_t i = 0; i < additional; ++i) {
        storage->slots.push_back(ptr + i);
    }
    return true;
}

ggml_metal_seg_result * ggml_metal_result_storage_at(ggml_metal_result_storage * storage, size_t i) {
    return storage && i < storage->slots.size() ? storage->slots[i] : nullptr;
}
