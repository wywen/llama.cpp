#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    GGML_BACKEND_API int ggml_cpu_has_sme2       (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

    //
    // flash attention partial, for the function of a GGML_OP_CUSTOM node
    //
    // Chunked flash attention splits a long KV into disjoint KV ranges. The partial of one range takes the inputs of
    // ggml_flash_attn_ext restricted to the range (views of K, V and of the matching mask columns) and stores the
    // unnormalized online-softmax state of each (head, query) row instead of the normalized output:
    //
    //   dst: [n_embd_v + 2, n_head, n_batch, ne3] F32, permuted like the output of ggml_flash_attn_ext
    //     dst[0 .. n_embd_v-1] = O = sum_i exp(s_i - M)*v_i
    //     dst[n_embd_v + 0]    = S = sum_i exp(s_i - M)
    //     dst[n_embd_v + 1]    = M = max_i s_i
    //
    // where s_i = scale*q*k_i (with logit_softcap applied), plus slope*mask_i. A row without a finite score (all of
    // its mask entries are -inf) has O = 0, S = 0 and M = -inf. The caller combines the partials of disjoint ranges
    // (M = max(Ma, Mb), O = Oa*exp(Ma - M) + Ob*exp(Mb - M), S = Sa*exp(Ma - M) + Sb*exp(Mb - M)) and normalizes the
    // result (O/S); a row with S = 0 adds nothing. The caller applies sinks when it normalizes.
    //
    // A caller computes the partial as a GGML_OP_CUSTOM node of its own and calls these from the node's function.
    // `dst` has the shape and type above, whatever its op: src[0..3] = q, k, v and the mask (F16, NULL: none),
    // src[4] = src[5] = NULL, src[6] = the active count (I32, NULL: none). The mask view may select columns of a
    // larger mask: its rows must be contiguous (nb[0] == sizeof(f16)) but its row stride may be that of the full mask.
    // With an active count, the partial is computed only when the count's first element (read at compute time) is
    // greater than active_index, e.g. the partial of chunk active_index of a number of chunks known at compute time
    // only. No op param of dst is read: the op parameters are passed in ggml_cpu_fa_partial_params.
    //

    struct ggml_cpu_fa_partial_params {
        float          scale;         // as in ggml_flash_attn_ext
        float          max_bias;
        float          logit_softcap;
        enum ggml_prec prec;          // as ggml_prec_set_acc sets on a flash attention node
        int32_t        active_index;  // read only when dst->src[6] is set
    };

    // the bytes of work data ggml_cpu_flash_attn_ext_partial_compute needs with nth threads: the work size the CPU
    // backend plans for a graph of one flash attention node of dst's sources, computed by nth threads
    GGML_BACKEND_API size_t ggml_cpu_flash_attn_ext_partial_work_size(const struct ggml_tensor * dst, int nth);

    // compute thread ith's share of the partial into dst's data. Each of the nth threads, ith in [0, nth), calls it
    // once with the same dst, params and wdata (one buffer of at least the work size for nth, shared by the threads);
    // the calls need no barrier between them, and dst is complete when all nth have returned. Each row is computed
    // whole by one thread, with the kernels of the CPU flash attention op, so the bytes do not depend on nth or on
    // the order of the calls. An inactive partial (the count is not greater than active_index) returns at once in
    // every thread and leaves dst unwritten.
    GGML_BACKEND_API void ggml_cpu_flash_attn_ext_partial_compute(
            struct ggml_tensor * dst, const struct ggml_cpu_fa_partial_params * params,
            int ith, int nth, void * wdata, size_t wsize);

#ifdef __cplusplus
}
#endif
