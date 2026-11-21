// Tests for Metal implementations of GGML_OP_CUSTOM nodes (ggml_backend_dev_add_custom_op). The node is a CPU
// custom op that computes dst = src + 1; its Metal implementation is a kernel the test compiles from source, and it
// accepts rows of a multiple of kRowMultiple elements only, so a node of another shape is refused.
//
// - registration returns false on the CPU device, whose backend takes no implementations, and refuses an entry without
//   a key or an encode function and a key that is registered already;
// - the device supports a CUSTOM node of the registered function that the implementation accepts, and no other one;
// - the scheduler keeps an accepted node in the Metal split, with no split of its own, and places a refused one on the
//   CPU; every graph computes, bitwise, the bytes a CPU-only run of it computes;
// - the encoder orders a CUSTOM node by the memory of its srcs and dst, as it does a stock op: it runs concurrently with
//   nodes it does not conflict with, takes a barrier after a node it reads, and a node that reads it takes one after it;
// - the graph reorder moves a CUSTOM node ahead of a node that waits for a barrier, as it moves a stock op.
//
// The encoder's concurrency decisions are read from its graph debug log (GGML_METAL_GRAPH_DEBUG). Everything but the
// registration on the CPU device needs a Metal device, so the binary skips the rest without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include "metal-custom-kernel.h"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

static int g_n_fail = 0;

static void check(bool ok, const char * scenario, const char * what) {
    printf("%s %-20s %s\n", ok ? "PASS" : "FAIL", scenario, what);
    if (!ok) {
        g_n_fail++;
    }
}

constexpr int64_t kElems       = 1000;
constexpr int64_t kRowMultiple = 4;

using bytes = std::vector<uint8_t>;

//
// the custom op: dst = src + 1 on the CPU, and the same through the Metal kernel
//

static std::atomic<int> g_n_cpu_calls{0};

static void add_one_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    if (ith == 0) {
        g_n_cpu_calls++;
    }

    const ggml_tensor * src = dst->src[0];

    const int64_t n   = ggml_nelements(dst);
    const int64_t per = (n + nth - 1)/nth;
    const int64_t i0  = ith*per;
    const int64_t i1  = i0 + per < n ? i0 + per : n;

    const float * x = (const float *) src->data;
    float       * y = (float       *) dst->data;
    for (int64_t i = i0; i < i1; i++) {
        y[i] = x[i] + 1.0f;
    }
}

// a function no implementation is registered for
static void add_one_cpu_unregistered(ggml_tensor * dst, int ith, int nth, void * userdata) {
    add_one_cpu(dst, ith, nth, userdata);
}

struct add_one_impl {
    void *           pipeline = nullptr;
    std::atomic<int> n_encodes{0};
};

static bool add_one_supports(void * user_data, const ggml_tensor * node) {
    (void) user_data;
    return node->ne[0] % kRowMultiple == 0;
}

static bool add_one_encode(void * user_data, const ggml_tensor * node, void * encoder) {
    auto * impl = (add_one_impl *) user_data;
    impl->n_encodes++;
    return test_metal_add_one_encode(impl->pipeline, node, encoder);
}

static bool add_one_encode_unused(void * user_data, const ggml_tensor * node, void * encoder) {
    (void) user_data;
    (void) node;
    (void) encoder;
    return false;
}

static ggml_tensor * add_one(ggml_context * ctx, ggml_tensor * a, ggml_custom_op_t fun = add_one_cpu) {
    ggml_tensor * args[] = { a };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3], args, 1, fun, GGML_N_TASKS_MAX, nullptr);
}

//
// the encoder's graph debug log: one line per encoded node, "<func>: node[<i>] - <op> <name> [(concurrent)]"
//

struct encoded_node {
    std::string op;
    std::string name;
    bool        concurrent;
};

static std::mutex                g_log_mutex;
static bool                      g_capturing = false;
static std::vector<encoded_node> g_encoded;

static void log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        const char * p = strstr(text, "node[");
        const char * q = p ? strstr(p, "] - ") : nullptr;
        if (g_capturing && q) {
            char op[64]   = {};
            char name[64] = {};
            char tag[64]  = {};
            const int n = sscanf(q + 4, "%63s %63s %63s", op, name, tag);
            if (n >= 2) {
                g_encoded.push_back({ op, name, n == 3 && strcmp(tag, "(concurrent)") == 0 });
            }
        }
    }
    if (level != GGML_LOG_LEVEL_DEBUG && level != GGML_LOG_LEVEL_INFO) {
        fputs(text, stderr);
    }
}

// the encode of the node named `name`, nullptr if it was not encoded
static const encoded_node * find_encoded(const std::vector<encoded_node> & encoded, const char * name) {
    for (const auto & e : encoded) {
        if (e.name == name) {
            return &e;
        }
    }
    return nullptr;
}

// the position of the node named `name` in the encode order, -1 if it was not encoded
static int encoded_position(const std::vector<encoded_node> & encoded, const char * name) {
    for (size_t i = 0; i < encoded.size(); i++) {
        if (encoded[i].name == name) {
            return (int) i;
        }
    }
    return -1;
}

//
// running a graph through the scheduler
//
// The inputs live in a weights buffer of the first backend, so the scheduler places the nodes that read them there;
// the nodes are allocated by the scheduler. Input k holds deterministic non-integral values.
//

struct graph_def {
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outputs;
    ggml_cgraph *              gf = nullptr;
};

typedef void (*graph_builder)(ggml_context * ctx_in, ggml_context * ctx, graph_def & def);

struct run_result {
    bool                        ok       = false;
    int                         n_splits = 0;
    std::vector<bytes>          outputs;
    std::vector<ggml_backend_t> placed; // the backend of each output
    std::vector<encoded_node>   encoded;
};

static run_result run(const std::vector<ggml_backend_t> & backends, graph_builder build) {
    run_result res;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*32 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_in = ggml_init(params);
    ggml_context * ctx    = ggml_init(params);

    graph_def def;
    build(ctx_in, ctx, def);

    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, backends[0]);
    ggml_backend_buffer_set_usage(buf_in, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (size_t k = 0; k < def.inputs.size(); k++) {
        ggml_tensor * t = def.inputs[k];
        std::vector<float> data(ggml_nelements(t));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (float) ((i*7 + k*13) % 97)*0.37f - 11.0f;
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    }

    std::vector<ggml_backend_t> bs = backends;
    ggml_backend_sched_t sched = ggml_backend_sched_new(bs.data(), nullptr, (int) bs.size(), GGML_DEFAULT_GRAPH_SIZE, false, false);

    res.ok = ggml_backend_sched_alloc_graph(sched, def.gf);
    if (res.ok) {
        res.n_splits = ggml_backend_sched_get_n_splits(sched);

        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_encoded.clear();
            g_capturing = true;
        }
        res.ok = ggml_backend_sched_graph_compute(sched, def.gf) == GGML_STATUS_SUCCESS;
        ggml_backend_sched_synchronize(sched);
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_capturing = false;
            res.encoded = g_encoded;
        }

        for (ggml_tensor * t : def.outputs) {
            bytes out(ggml_nbytes(t));
            ggml_backend_tensor_get(t, out.data(), 0, out.size());
            res.outputs.push_back(std::move(out));
            res.placed.push_back(ggml_backend_sched_get_tensor_backend(sched, t));
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx);
    ggml_free(ctx_in);

    return res;
}

static ggml_tensor * new_input(ggml_context * ctx_in, graph_def & def, const char * name, int64_t n = kElems) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, n);
    ggml_set_name(t, name);
    def.inputs.push_back(t);
    return t;
}

static void add_output(ggml_tensor * t, graph_def & def, const char * name) {
    ggml_set_name(t, name);
    ggml_set_output(t);
    def.outputs.push_back(t);
}

// a = x0 + x1, c = y + 1, b = z0 * z1: the custom node conflicts with neither stock op
static void build_independent(ggml_context * ctx_in, ggml_context * ctx, graph_def & def) {
    ggml_tensor * x0 = new_input(ctx_in, def, "x0");
    ggml_tensor * x1 = new_input(ctx_in, def, "x1");
    ggml_tensor * y  = new_input(ctx_in, def, "y");
    ggml_tensor * z0 = new_input(ctx_in, def, "z0");
    ggml_tensor * z1 = new_input(ctx_in, def, "z1");

    add_output(ggml_add(ctx, x0, x1), def, "a");
    add_output(add_one(ctx, y),       def, "c");
    add_output(ggml_mul(ctx, z0, z1), def, "b");

    def.gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : def.outputs) {
        ggml_build_forward_expand(def.gf, t);
    }
}

// t = x0 + x1, c = t + 1, u = 2*c: the custom node reads its predecessor and its successor reads it
static void build_dependent(ggml_context * ctx_in, ggml_context * ctx, graph_def & def) {
    ggml_tensor * x0 = new_input(ctx_in, def, "x0");
    ggml_tensor * x1 = new_input(ctx_in, def, "x1");

    ggml_tensor * t = ggml_add(ctx, x0, x1);
    add_output(t, def, "t");
    ggml_tensor * c = add_one(ctx, t);
    add_output(c, def, "c");
    add_output(ggml_scale(ctx, c, 2.0f), def, "u");

    def.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(def.gf, def.outputs.back());
}

// t = x0 + x1, u = 2*t, c = y + 1: u waits for a barrier after t, and the reorder can move c ahead of it
static void build_hoistable(ggml_context * ctx_in, ggml_context * ctx, graph_def & def) {
    ggml_tensor * x0 = new_input(ctx_in, def, "x0");
    ggml_tensor * x1 = new_input(ctx_in, def, "x1");
    ggml_tensor * y  = new_input(ctx_in, def, "y");

    ggml_tensor * t = ggml_add(ctx, x0, x1);
    add_output(t, def, "t");
    add_output(ggml_scale(ctx, t, 2.0f), def, "u");
    add_output(add_one(ctx, y), def, "c");

    def.gf = ggml_new_graph(ctx);
    for (ggml_tensor * o : def.outputs) {
        ggml_build_forward_expand(def.gf, o);
    }
}

// t = x0 + x1, c = t + 1 over rows the implementation refuses, u = 2*c
static void build_refused(ggml_context * ctx_in, ggml_context * ctx, graph_def & def) {
    ggml_tensor * x0 = new_input(ctx_in, def, "x0", kElems + 1);
    ggml_tensor * x1 = new_input(ctx_in, def, "x1", kElems + 1);

    ggml_tensor * t = ggml_add(ctx, x0, x1);
    add_output(t, def, "t");
    ggml_tensor * c = add_one(ctx, t);
    add_output(c, def, "c");
    add_output(ggml_scale(ctx, c, 2.0f), def, "u");

    def.gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(def.gf, def.outputs.back());
}

//
// tests
//

static void test_registration(ggml_backend_dev_t dev_metal, ggml_backend_dev_t dev_cpu, add_one_impl & impl) {
    const ggml_backend_custom_op op = { add_one_cpu, add_one_supports, add_one_encode, &impl };

    check(!ggml_backend_dev_add_custom_op(dev_cpu, &op), "registration", "the CPU device takes no implementation");

    ggml_backend_custom_op no_key = op;
    no_key.key = nullptr;
    check(!ggml_backend_dev_add_custom_op(dev_metal, &no_key), "registration", "an entry without a key is refused");

    ggml_backend_custom_op no_encode = op;
    no_encode.encode = nullptr;
    check(!ggml_backend_dev_add_custom_op(dev_metal, &no_encode), "registration", "an entry without an encode function is refused");

    check(ggml_backend_dev_add_custom_op(dev_metal, &op), "registration", "the implementation is registered");

    const ggml_backend_custom_op again = { add_one_cpu, nullptr, add_one_encode_unused, nullptr };
    check(!ggml_backend_dev_add_custom_op(dev_metal, &again), "registration", "a key registered already is refused");
}

static void test_supports(ggml_backend_dev_t dev_metal) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems);
    ggml_tensor * xr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kElems + 1);

    check(ggml_backend_dev_supports_op(dev_metal, add_one(ctx, x)), "supports", "an accepted node of the registered function");
    check(!ggml_backend_dev_supports_op(dev_metal, add_one(ctx, xr)), "supports", "not a node the implementation refuses");
    check(!ggml_backend_dev_supports_op(dev_metal, add_one(ctx, x, add_one_cpu_unregistered)), "supports", "not a node of another function");

    ggml_free(ctx);
}

static bool same_outputs(const run_result & a, const run_result & b) {
    return a.ok && b.ok && a.outputs == b.outputs;
}

static void test_graph(const char * scenario, graph_builder build, ggml_backend_t metal, ggml_backend_t cpu,
                       add_one_impl & impl, run_result & res) {
    g_n_cpu_calls = 0;
    const run_result ref = run({ cpu }, build);
    check(ref.ok && g_n_cpu_calls > 0, scenario, "the CPU-only run computes through the CPU function");

    g_n_cpu_calls   = 0;
    impl.n_encodes  = 0;
    res = run({ metal, cpu }, build);
    check(res.ok, scenario, "the Metal run computes");
    check(same_outputs(res, ref), scenario, "bitwise the bytes of the CPU-only run");
}

static void test_placement_and_order(ggml_backend_t metal, ggml_backend_t cpu, add_one_impl & impl) {
    run_result res;

    // outputs: a, c, b
    test_graph("independent", build_independent, metal, cpu, impl, res);
    {
        const encoded_node * c = find_encoded(res.encoded, "c");
        const encoded_node * b = find_encoded(res.encoded, "b");
        check(res.n_splits == 1 && res.placed.size() == 3 && res.placed[1] == metal, "independent", "the custom node stays in the one Metal split");
        check(impl.n_encodes > 0 && g_n_cpu_calls == 0, "independent", "the Metal implementation computes the node");
        check(c != nullptr && c->op == "CUSTOM" && c->concurrent, "independent", "no barrier before the custom node");
        check(b != nullptr && b->concurrent, "independent", "no barrier after the custom node");
    }

    // outputs: t, c, u
    test_graph("dependent", build_dependent, metal, cpu, impl, res);
    {
        const encoded_node * c = find_encoded(res.encoded, "c");
        const encoded_node * u = find_encoded(res.encoded, "u");
        check(res.n_splits == 1 && res.placed.size() == 3 && res.placed[1] == metal, "dependent", "the custom node stays in the one Metal split");
        check(c != nullptr && !c->concurrent, "dependent", "a barrier before the custom node, which reads its predecessor");
        check(u != nullptr && !u->concurrent, "dependent", "a barrier before the node that reads the custom node");
    }

    // outputs: t, u, c
    test_graph("hoistable", build_hoistable, metal, cpu, impl, res);
    {
        const encoded_node * c = find_encoded(res.encoded, "c");
        const int pos_c = encoded_position(res.encoded, "c");
        const int pos_u = encoded_position(res.encoded, "u");
        check(res.n_splits == 1, "hoistable", "one Metal split");
        check(pos_c >= 0 && pos_u >= 0 && pos_c < pos_u, "hoistable", "the reorder moves the custom node ahead of the node that waits");
        check(c != nullptr && c->concurrent, "hoistable", "the moved custom node runs with the nodes before the barrier");
    }

    // outputs: t, c, u
    test_graph("refused", build_refused, metal, cpu, impl, res);
    {
        check(res.placed.size() == 3 && res.placed[1] == cpu, "refused", "the scheduler places the refused node on the CPU");
        check(res.placed.size() == 3 && res.placed[0] == metal, "refused", "its predecessor stays on Metal");
        check(g_n_cpu_calls > 0, "refused", "the CPU function computes the refused node");
    }
}

int main(void) {
    // before any Metal backend exists, which reads it once
    setenv("GGML_METAL_GRAPH_DEBUG", "1", 1);
    ggml_log_set(log_callback, nullptr);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_dev_t dev_cpu = ggml_backend_get_device(cpu);

    ggml_backend_reg_t reg = ggml_backend_metal_reg();
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        const ggml_backend_custom_op op = { add_one_cpu, nullptr, add_one_encode_unused, nullptr };
        check(!ggml_backend_dev_add_custom_op(dev_cpu, &op), "registration", "the CPU device takes no implementation");
        ggml_backend_free(cpu);
        printf("SKIP no Metal device\n");
        return g_n_fail == 0 ? 0 : 1;
    }
    ggml_backend_dev_t dev_metal = ggml_backend_reg_dev_get(reg, 0);

    add_one_impl impl;
    impl.pipeline = test_metal_add_one_pipeline_new(dev_metal);
    check(impl.pipeline != nullptr, "setup", "the kernel compiles from source");
    if (impl.pipeline == nullptr) {
        ggml_backend_free(cpu);
        return 1;
    }

    ggml_backend_t metal = ggml_backend_dev_init(dev_metal, nullptr);
    check(metal != nullptr, "setup", "a Metal backend");
    if (metal == nullptr) {
        test_metal_pipeline_release(impl.pipeline);
        ggml_backend_free(cpu);
        return 1;
    }

    test_registration(dev_metal, dev_cpu, impl);
    test_supports(dev_metal);
    test_placement_and_order(metal, cpu, impl);

    // the registration stays for the device's lifetime, which outlives the binary's use of the pipeline
    ggml_backend_free(metal);
    ggml_backend_free(cpu);

    printf("%s: %d failure(s)\n", g_n_fail == 0 ? "OK" : "FAILED", g_n_fail);
    return g_n_fail == 0 ? 0 : 1;
}
