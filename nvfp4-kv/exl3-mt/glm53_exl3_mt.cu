// glm53_exl3_mt.cu -- M-tiled EXL3 fused MoE kernel for prefill on GLM-5.3-Flash EXL3 (DGX Spark, sm_121a).
//
// Derived from exllamav3's exl3_moe_kernel / exl3_gemm_kernel_inner (turboderp, MIT). The upstream inner
// kernel is hard-wired to TILESIZE_M == 16: every 16-row pass re-streams and re-decodes the whole trellis
// weight, so an expert with R rows decodes its weights ceil(R/16) times and the tensor cores idle behind
// the integer 3INST decode. This variant processes TILESIZE_M = 16*TBM rows per pass, so each dequantized
// B fragment feeds TBM MMAs. Everything else (cp.async pipeline, XOR-swizzled A tiles, split-K lock
// reduction, Hadamard pre/post passes, group barriers) follows upstream so results stay comparable.
//
// Only the 4-bit mcg codebook (cb = 1) used by this model is instantiated.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda/atomic>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <set>
#include <stdint.h>

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_dq.cuh"
#include "quant/hadamard_inner.cuh"
#include "quant/exl3_moe_common.cuh"

#define MT_BARRIER_OFFSET 65536          // ints; GEMM column locks live below, group barriers above
#define MT_LOCKS_INTS (MT_BARRIER_OFFSET + 4096)

// ----------------------------------------------------------------------------------------------------------------
// M-tiled GEMM inner: C[size_m, size_n] (fp16) = A[size_m, size_k] (fp16) x dequant(B), size_m <= 16*TBM.
// Runs on gridDim.x blocks that split the (tiles_k x tiles_n) space into contiguous slices; partial sums
// along k are accumulated through global C under per-column locks (upstream protocol).

template<int bits, int cb, int TBM, int TILESIZE_K, int TILESIZE_N, int SH_STAGES, int FRAG_STAGES>
__device__ __forceinline__ void mt_gemm_inner
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    half* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks
)
{
    constexpr int TILESIZE_M = 16 * TBM;
    constexpr int TILEBLOCKS_K = TILESIZE_K / 16;
    constexpr int TILEBLOCKS_N = TILESIZE_N / 16;
    constexpr int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);
    constexpr int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                          // halfs
    constexpr int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;    // uint16s
    constexpr int sh_c_size = (TILEBLOCKS_K > 1) ? 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP * TBM : 0;  // floats
    constexpr int A_COLS = TILESIZE_K / 8;
    constexpr int A_SWIZZLE_MASK = A_COLS - 1;
    constexpr int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    static_assert(EXL3_GEMM_BASE_THREADS == 256, "base threads");
    static_assert(TILESIZE_K % 16 == 0 && TILESIZE_N % 128 == 0, "tile shape");
    static_assert(FRAGS_N_PER_WARP >= 2 && FRAGS_N_PER_WARP % 2 == 0, "frags per warp");
    static_assert(SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
                  "insufficient shared memory for this shape");

    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    const int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    const int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    const int warp_id = t / 32;
    const int lane_id = t % 32;

    const int tiles_k = size_k / TILESIZE_K;
    const int tiles_n = size_n / TILESIZE_N;
    const int blocks_n = tiles_n * TILEBLOCKS_N;

    const int num_slices = gridDim.x;
    const int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    const int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    const int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_k = [&] (int s) { return s % tiles_k; };
    auto index_n = [&] (int s) { return s / tiles_k; };

    // Pipe 0: global -> shared
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    constexpr int gl_a_stride_k = TILESIZE_K;
    const half* gl_a_ptr = A + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    constexpr int load_a_iters = CEIL_DIVIDE(sh_a_stage_size / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    #pragma unroll
    for (int i = 0; i < load_a_iters; ++i)
    {
        int idx = i * EXL3_GEMM_BASE_THREADS + t;
        int k = idx % (gl_a_stride_k / 8);
        int m = idx / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = (m < size_m) && (m < TILESIZE_M);
    }

    const int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    constexpr int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    constexpr int sh0_b_stride_k = sh_b_stage_size;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    constexpr int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    #pragma unroll
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;
        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1: shared -> registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;
    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;
        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice1_k >= tiles_k) { slice1_k = 0; slice1_n++; }
    };

    // Pipe 2: MMA + reduction
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    constexpr int gl_c_stride_n = TILESIZE_N;
    half* gl_c_ptr = C + slice2_n * gl_c_stride_n;

    FragA frag_a[FRAG_STAGES][TBM];
    FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    FragC frag_c[TBM][FRAGS_N_PER_WARP];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;
        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            gl_c_ptr += gl_c_stride_n;
        }
    };

    auto async_load_gl = [&] ()
    {
        if (sub_k) { cp_async_fence(); return; }
        if (slice0_iters)
        {
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
            }
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
            }
            advance0();
        }
        cp_async_fence();
    };

    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TBM; ++m)
            {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a[buf][m], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);
            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }
        __syncthreads();
        advance1();
    };

    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < TBM; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[m][n] = {};
    };

    // Reduce the sub_k partial sums of this threadblock into sub_k == 0
    auto threadblock_reduce = [&] ()
    {
        if constexpr (TILEBLOCKS_K > 1)
        {
            auto store = [&] (int i)
            {
                if (sub_k == i)
                {
                    float* sh_red = sh_c + (TBM * FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int m = 0; m < TBM; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[m][n][j];
                }
                __syncthreads();
            };
            auto add = [&] (int i)
            {
                if (sub_k == i)
                {
                    float* sh_red = sh_c + (TBM * FRAGS_N_PER_WARP * 4) * t;
                    #pragma unroll
                    for (int m = 0; m < TBM; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) frag_c[m][n][j] += *sh_red++;
                }
            };
            if constexpr (TILEBLOCKS_K == 2) { store(1); add(0); }
            if constexpr (TILEBLOCKS_K == 3) { store(1); add(0); store(2); add(0); }
            if constexpr (TILEBLOCKS_K == 4) { store(3); add(2); store(1); add(0); store(2); add(0); }
        }
    };

    auto read_sum_gl = [&] ()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        int c = (lane_id % 4) * 2;
        #pragma unroll
        for (int m = 0; m < TBM; ++m)
        {
            int r0 = m * 16 + lane_id / 4;
            int r1 = r0 + 8;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                if (r0 < size_m)
                {
                    float2 v = __half22float2(*(half2*) (gl_c_ptr + r0 * size_n + (n0 + n) * 8 + c));
                    frag_c[m][n][0] += v.x; frag_c[m][n][1] += v.y;
                }
                if (r1 < size_m)
                {
                    float2 v = __half22float2(*(half2*) (gl_c_ptr + r1 * size_n + (n0 + n) * 8 + c));
                    frag_c[m][n][2] += v.x; frag_c[m][n][3] += v.y;
                }
            }
        }
    };

    auto write_sum_gl = [&] ()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        int c = (lane_id % 4) * 2;
        #pragma unroll
        for (int m = 0; m < TBM; ++m)
        {
            int r0 = m * 16 + lane_id / 4;
            int r1 = r0 + 8;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                if (r0 < size_m)
                    *(half2*) (gl_c_ptr + r0 * size_n + (n0 + n) * 8 + c) = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                if (r1 < size_m)
                    *(half2*) (gl_c_ptr + r1 * size_n + (n0 + n) * 8 + c) = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
            }
        }
    };

    auto reduce = [&] ()
    {
        threadblock_reduce();
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice2_n];
        barrier_acquire(lock, lock_i);
        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;
        if (!sub_k && !first) read_sum_gl();
        if (!sub_k) write_sum_gl();
        barrier_release(lock, lock_d, last);
        clear_frag_c();
    };

    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int m = 0; m < TBM; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c[m][n]);
    };

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i) async_load_gl();
    wait_stage();
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1) load_frags(0);

    #define MT_FSTAGE_OLD(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break;

    #define MT_FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load);

    if constexpr (FRAG_STAGES == 1) { while (true) { MT_FSTAGE_OLD(0, 0); } }
    if constexpr (FRAG_STAGES == 2) { while (true) { MT_FSTAGE(1, 0); MT_FSTAGE(0, 1); } }
    if constexpr (FRAG_STAGES == 3) { while (true) { MT_FSTAGE(1, 0); MT_FSTAGE(2, 1); MT_FSTAGE(0, 2); } }
    #undef MT_FSTAGE
    #undef MT_FSTAGE_OLD
}

// ----------------------------------------------------------------------------------------------------------------
// Fused MoE kernel, upstream structure with the M-tiled inner

template<int bits, int TBM, int TILESIZE_K, int TILESIZE_N, int SH_STAGES, int FRAG_STAGES>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_moe_mt_kernel(EXL3_MOE_KERNEL_ARGS)
{
    constexpr int TILESIZE_M = 16 * TBM;
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int block_threads = EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16;
    const int group_threads = MOE_SMS_PER_EXPERT * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    temp_state_g += (size_t) group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += (size_t) group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += (size_t) group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += (size_t) group_idx * max_tokens_per_expert * intermediate_dim;

    int* barrier_counters_sense = locks + MT_BARRIER_OFFSET;
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    int start = 0, end = 0, expert_idx_assign = 0;
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx)
    {
        start = end;
        end += expert_count[expert_idx];
        int token_count = end - start;
        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;
        if (expert_idx_assign++ % concurrency != group_idx) continue;

        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx];
        const half* exp_gate_suh = gate_suh[expert_idx];
        const half* exp_gate_svh = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis = up_trellis[expert_idx];
        const half* exp_up_suh = up_suh[expert_idx];
        const half* exp_up_svh = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx];
        const half* exp_down_suh = down_suh[expert_idx];
        const half* exp_down_svh = down_svh[expert_idx];

        // Gather + input Hadamard for g, u
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + (size_t) token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_inner<true, false>(in_ptr, temp_state_g + 128 * warp_idx, exp_gate_suh + 128 * token_off, 0.088388347648f);
                had_hf_r_128_inner<true, false>(in_ptr, temp_state_u + 128 * warp_idx, exp_up_suh + 128 * token_off, 0.088388347648f);
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }

        // g, u GEMMs, TILESIZE_M rows per pass
        auto gemm = [&] (const half* in_addr, half* out_addr, const uint16_t* trellis, int size_k, int size_n)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                mt_gemm_inner<bits, 1, TBM, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
                    (in_addr, trellis, out_addr, MIN(size_m, TILESIZE_M), size_k, size_n, locks);
                in_addr += TILESIZE_M * size_k;
                out_addr += TILESIZE_M * size_n;
                size_m -= TILESIZE_M;
            }
        };
        gemm(temp_state_g, temp_intermediate_g, exp_gate_trellis, hidden_dim, intermediate_dim);
        gemm(temp_state_u, temp_intermediate_u, exp_up_trellis, hidden_dim, intermediate_dim);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);

        // Output Hadamard for g, u + activation + input Hadamard for d
        {
            const int warps_per_token = intermediate_dim / 128;
            const int total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner
                (
                    temp_intermediate_g + 128 * warp_idx, temp_intermediate_u + 128 * warp_idx, temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off, exp_up_svh + 128 * token_off, exp_down_suh + 128 * token_off,
                    0.088388347648f, act_limit, act_function
                );
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }

        // d GEMM
        gemm(temp_intermediate_g, temp_state_g, exp_down_trellis, intermediate_dim, hidden_dim);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);

        // Output Hadamard for d + weighted scatter-add
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            const half* weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                half weight = weights[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                float* out_ptr = output_state + (size_t) token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_d_inner(temp_state_g + 128 * warp_idx, out_ptr, exp_down_svh + 128 * token_off, 0.088388347648f * __half2float(weight));
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }
    }
}


// ----------------------------------------------------------------------------------------------------------------
// Variant 2 ("smem-B"): each block decodes every 16x16 trellis tile of the current K32 x N128 slab ONCE into
// shared memory (fp16, [n][k] layout, padded row stride so ldmatrix is conflict-free) and 8 warps each own a
// 16*TBW-row M tile, reading B fragments with ldmatrix.x2. Decode cost per block per stage is fixed (two dq8
// per thread) while MMA work scales with TBW*8 m16 tiles, so decode is amortised 2x (TBW=1) or 4x (TBW=2)
// better than variant m64_k16_n256. 256 threads, no split-K inside the block (the two k16 halves run
// sequentially), so no shared-memory reduction is needed.

__device__ __forceinline__ void ldsm2(FragB& frag_b, const void* smem_ptr)
{
    uint32_t* b = reinterpret_cast<uint32_t*>(&frag_b);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b[0]), "=r"(b[1]) : "r"(smem));
}

template<int bits, int cb, int TBW, int SH_STAGES>
__device__ __forceinline__ void mt2_gemm_inner
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    half* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks
)
{
    constexpr int THREADS = 256, WARPS = 8;
    constexpr int TILESIZE_M = 16 * TBW * WARPS;
    constexpr int TILESIZE_K = 32, TILESIZE_N = 128;
    constexpr int TILEBLOCKS_K = TILESIZE_K / 16, TILEBLOCKS_N = TILESIZE_N / 16;
    constexpr int NFRAG = TILESIZE_N / 8;
    constexpr int BROW = TILESIZE_K + 8;                                     // padded k stride (halfs) of decoded B
    constexpr int TILE_U16 = 256 / 16 * bits;                                // packed uint16 per 16x16 tile
    constexpr int sh_a_stage = TILESIZE_M * TILESIZE_K;                      // halfs
    constexpr int sh_bp_stage = TILEBLOCKS_K * TILEBLOCKS_N * TILE_U16;      // uint16
    constexpr int sh_bd_stage = TILESIZE_N * BROW;                           // halfs
    constexpr int A_COLS = TILESIZE_K / 8, A_SWIZZLE_MASK = A_COLS - 1, A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;
    static_assert(SMEM_MAX >= SH_STAGES * (2 * sh_a_stage + 2 * sh_bp_stage + 2 * sh_bd_stage), "smem-B: insufficient shared memory");
    static_assert(TILEBLOCKS_K * TILEBLOCKS_N == 2 * WARPS, "decode phase assumes 16 tiles per stage");

    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_bp = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage);
    half* sh_bd = (half*) (sh_bp + SH_STAGES * sh_bp_stage);

    const int t = threadIdx.x, warp_id = t / 32, lane_id = t % 32;
    const int tiles_k = size_k / TILESIZE_K, tiles_n = size_n / TILESIZE_N, blocks_n = tiles_n * TILEBLOCKS_N;
    const int num_slices = gridDim.x;
    const int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    const int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    const int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;
    auto index_k = [&] (int s) { return s % tiles_k; };
    auto index_n = [&] (int s) { return s / tiles_k; };

    // Pipe 0: global -> shared (A rows of this pass, packed B tiles)
    int slice0_k = index_k(slice_beg), slice0_n = index_n(slice_beg), slice0_iters = slice_len;
    constexpr int gl_a_stride_k = TILESIZE_K;
    const half* gl_a_ptr = A + slice0_k * gl_a_stride_k;
    constexpr int load_a_iters = CEIL_DIVIDE(sh_a_stage / 8, THREADS);
    bool pred_a_gl[load_a_iters]; int load_a_gl[load_a_iters]; int load_a_sh[load_a_iters];
    #pragma unroll
    for (int i = 0; i < load_a_iters; ++i)
    {
        int idx = i * THREADS + t;
        int k = idx % (gl_a_stride_k / 8), m = idx / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = (m < size_m) && (m < TILESIZE_M);
    }
    const int gl_b_stride_k = blocks_n * TILEBLOCKS_K * TILE_U16;
    constexpr int gl_b_stride_n = TILEBLOCKS_N * TILE_U16;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    constexpr int load_b_iters = CEIL_DIVIDE(sh_bp_stage / 8, THREADS);
    bool pred_b_gl[load_b_iters]; int load_b_gl[load_b_iters];
    #pragma unroll
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * TILE_U16 / 8) + n;
        pred_b_gl[i] = i * THREADS + t < sh_bp_stage / 8;
    }
    auto advance0 = [&] ()
    {
        slice0_k++; slice0_iters--;
        if (slice0_k >= tiles_k) { slice0_k = 0; slice0_n++; gl_a_ptr = A + slice0_k * gl_a_stride_k; gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n; }
        else { gl_a_ptr += gl_a_stride_k; gl_b_ptr += gl_b_stride_k; }
    };
    auto async_load_gl = [&] ()
    {
        if (slice0_iters)
        {
            int stage = slice0_iters % SH_STAGES;
            const int4* gla = (const int4*) gl_a_ptr; int4* sha = (int4*) (sh_a + stage * sh_a_stage);
            #pragma unroll
            for (int i = 0; i < load_a_iters; ++i) if (pred_a_gl[i]) cp_async(sha + load_a_sh[i], gla + load_a_gl[i]);
            const int4* glb = (const int4*) gl_b_ptr; int4* shb = (int4*) (sh_bp + stage * sh_bp_stage);
            #pragma unroll
            for (int i = 0; i < load_b_iters; ++i) if (pred_b_gl[i]) cp_async(shb + THREADS * i + t, glb + load_b_gl[i]);
            advance0();
        }
        cp_async_fence();
    };

    // Pipe 2: compute + reduction
    int slice2_k = index_k(slice_beg), slice2_k0 = slice2_k, slice2_n = index_n(slice_beg), slice2_iters = slice_len;
    half* gl_c_ptr = C + slice2_n * TILESIZE_N;
    FragC frag_c[TBW][NFRAG];
    auto clear_frag_c = [&] () {
        #pragma unroll
        for (int m = 0; m < TBW; ++m)
            #pragma unroll
            for (int j = 0; j < NFRAG; ++j) frag_c[m][j] = {};
    };
    auto advance2 = [&] () { slice2_k++; slice2_iters--; if (slice2_k >= tiles_k) { slice2_k = 0; slice2_k0 = 0; slice2_n++; gl_c_ptr += TILESIZE_N; } };

    auto read_sum_gl = [&] () {
        int c = (lane_id % 4) * 2;
        #pragma unroll
        for (int m = 0; m < TBW; ++m) {
            int r0 = (warp_id * TBW + m) * 16 + lane_id / 4, r1 = r0 + 8;
            #pragma unroll
            for (int j = 0; j < NFRAG; ++j) {
                if (r0 < size_m) { float2 v = __half22float2(*(half2*) (gl_c_ptr + r0 * size_n + j * 8 + c)); frag_c[m][j][0] += v.x; frag_c[m][j][1] += v.y; }
                if (r1 < size_m) { float2 v = __half22float2(*(half2*) (gl_c_ptr + r1 * size_n + j * 8 + c)); frag_c[m][j][2] += v.x; frag_c[m][j][3] += v.y; }
            }
        }
    };
    auto write_sum_gl = [&] () {
        int c = (lane_id % 4) * 2;
        #pragma unroll
        for (int m = 0; m < TBW; ++m) {
            int r0 = (warp_id * TBW + m) * 16 + lane_id / 4, r1 = r0 + 8;
            #pragma unroll
            for (int j = 0; j < NFRAG; ++j) {
                if (r0 < size_m) *(half2*) (gl_c_ptr + r0 * size_n + j * 8 + c) = __floats2half2_rn(frag_c[m][j][0], frag_c[m][j][1]);
                if (r1 < size_m) *(half2*) (gl_c_ptr + r1 * size_n + j * 8 + c) = __floats2half2_rn(frag_c[m][j][2], frag_c[m][j][3]);
            }
        }
    };
    auto reduce = [&] () {
        int lock_i = tiles_k - slice2_k - 1, lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice2_n];
        barrier_acquire(lock, lock_i);
        bool first = lock_i == 0, last = lock_i + lock_d == tiles_k;
        if (!first) read_sum_gl();
        write_sum_gl();
        barrier_release(lock, lock_d, last);
        clear_frag_c();
    };

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i) async_load_gl();
    clear_frag_c();

    while (true)
    {
        async_load_gl();
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
        const int slot = slice2_iters % SH_STAGES;
        // decode this stage's 16 packed tiles into fp16 [n][k]
        {
            const uint32_t* bp = (const uint32_t*) (sh_bp + slot * sh_bp_stage);
            half* bd = sh_bd + slot * sh_bd_stage;
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                int tile = warp_id + WARPS * i;
                int kb = tile / TILEBLOCKS_N, nb = tile % TILEBLOCKS_N;
                FragB f0, f1;
                dq_dispatch<bits, cb>(bp + (kb * TILEBLOCKS_N + nb) * (TILE_U16 / 2), lane_id << 3, f0, f1);
                int n = nb * 16 + lane_id / 4, k = kb * 16 + (lane_id % 4) * 2;
                *(half2*) (bd + n * BROW + k) = f0[0];
                *(half2*) (bd + n * BROW + k + 8) = f0[1];
                *(half2*) (bd + (n + 8) * BROW + k) = f1[0];
                *(half2*) (bd + (n + 8) * BROW + k + 8) = f1[1];
            }
        }
        __syncthreads();
        // MMA: each warp its own 16*TBW rows, all 128 columns
        {
            const half* a = sh_a + slot * sh_a_stage;
            const half* bd = sh_bd + slot * sh_bd_stage;
            const int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            const int l16 = lane_id % 16;
            #pragma unroll
            for (int kk = 0; kk < TILEBLOCKS_K; ++kk)
            {
                FragA fa[TBW];
                #pragma unroll
                for (int m = 0; m < TBW; ++m)
                {
                    int R = (warp_id * TBW + m) * 16 + r;
                    int c_sw = (lane_id / 16 + kk * 2) ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                    ldsm4(fa[m], (const int4*) a + R * A_COLS + c_sw);
                }
                #pragma unroll
                for (int j = 0; j < NFRAG; ++j)
                {
                    FragB fb;
                    ldsm2(fb, bd + (j * 8 + (l16 % 8)) * BROW + kk * 16 + (l16 / 8) * 8);
                    #pragma unroll
                    for (int m = 0; m < TBW; ++m) ptx_mma_m16n8k16(fa[m], fb, frag_c[m][j]);
                }
            }
        }
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
        advance2();
        __syncthreads();            // everyone is done reading this slot before the next load reuses it
        if (!slice2_iters) break;
    }
}

template<int bits, int TBW, int SH_STAGES>
__global__ __launch_bounds__(256)
void exl3_moe_mt2_kernel(EXL3_MOE_KERNEL_ARGS)
{
    constexpr int TILESIZE_M = 16 * TBW * 8;
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int block_threads = 256;
    const int group_threads = MOE_SMS_PER_EXPERT * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    temp_state_g += (size_t) group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += (size_t) group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += (size_t) group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += (size_t) group_idx * max_tokens_per_expert * intermediate_dim;
    int* barrier_counters_sense = locks + MT_BARRIER_OFFSET;
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    int start = 0, end = 0, expert_idx_assign = 0;
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx)
    {
        start = end; end += expert_count[expert_idx];
        int token_count = end - start;
        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;
        if (expert_idx_assign++ % concurrency != group_idx) continue;
        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx]; const half* exp_gate_suh = gate_suh[expert_idx]; const half* exp_gate_svh = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis = up_trellis[expert_idx];     const half* exp_up_suh = up_suh[expert_idx];     const half* exp_up_svh = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx]; const half* exp_down_suh = down_suh[expert_idx]; const half* exp_down_svh = down_svh[expert_idx];
        {
            const int warps_per_token = hidden_dim / 128, total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token], token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + (size_t) token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_inner<true, false>(in_ptr, temp_state_g + 128 * warp_idx, exp_gate_suh + 128 * token_off, 0.088388347648f);
                had_hf_r_128_inner<true, false>(in_ptr, temp_state_u + 128 * warp_idx, exp_up_suh + 128 * token_off, 0.088388347648f);
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }
        auto gemm = [&] (const half* in_addr, half* out_addr, const uint16_t* trellis, int size_k, int size_n)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                mt2_gemm_inner<bits, 1, TBW, SH_STAGES>(in_addr, trellis, out_addr, MIN(size_m, TILESIZE_M), size_k, size_n, locks);
                in_addr += TILESIZE_M * size_k; out_addr += TILESIZE_M * size_n; size_m -= TILESIZE_M;
            }
        };
        gemm(temp_state_g, temp_intermediate_g, exp_gate_trellis, hidden_dim, intermediate_dim);
        gemm(temp_state_u, temp_intermediate_u, exp_up_trellis, hidden_dim, intermediate_dim);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        {
            const int warps_per_token = intermediate_dim / 128, total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner(temp_intermediate_g + 128 * warp_idx, temp_intermediate_u + 128 * warp_idx, temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off, exp_up_svh + 128 * token_off, exp_down_suh + 128 * token_off, 0.088388347648f, act_limit, act_function);
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }
        gemm(temp_intermediate_g, temp_state_g, exp_down_trellis, intermediate_dim, hidden_dim);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        {
            const int warps_per_token = hidden_dim / 128, total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start; const half* weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token]; half weight = weights[warp_idx / warps_per_token]; int token_off = warp_idx % warps_per_token;
                float* out_ptr = output_state + (size_t) token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_d_inner(temp_state_g + 128 * warp_idx, out_ptr, exp_down_svh + 128 * token_off, 0.088388347648f * __half2float(weight));
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }
    }
}


// ----------------------------------------------------------------------------------------------------------------
// Variant 3 ("auto"): one 256-thread kernel that picks the GEMM tile per EXPERT from its row count, so a layer
// whose experts range from 4 to 600 rows gets the 64-row M-tiled inner for the small ones (decode-bound, tiles
// mostly padded otherwise) and the shared-memory-B inner with 128- or 256-row tiles for the big ones (MMA at
// the dense-GEMM ceiling). Thresholds from the row sweep on GB10 (bench_rows.txt).

template<int bits, int SMALL_MAX, int MID_MAX>
__global__ __launch_bounds__(256)
void exl3_moe_mt3_kernel(EXL3_MOE_KERNEL_ARGS)
{
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int block_threads = 256;
    const int group_threads = MOE_SMS_PER_EXPERT * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    temp_state_g += (size_t) group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += (size_t) group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += (size_t) group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += (size_t) group_idx * max_tokens_per_expert * intermediate_dim;
    int* barrier_counters_sense = locks + MT_BARRIER_OFFSET;
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    int start = 0, end = 0, expert_idx_assign = 0;
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx)
    {
        start = end; end += expert_count[expert_idx];
        int token_count = end - start;
        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;
        if (expert_idx_assign++ % concurrency != group_idx) continue;
        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx]; const half* exp_gate_suh = gate_suh[expert_idx]; const half* exp_gate_svh = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis = up_trellis[expert_idx];     const half* exp_up_suh = up_suh[expert_idx];     const half* exp_up_svh = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx]; const half* exp_down_suh = down_suh[expert_idx]; const half* exp_down_svh = down_svh[expert_idx];
        {
            const int warps_per_token = hidden_dim / 128, total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token], token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + (size_t) token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_inner<true, false>(in_ptr, temp_state_g + 128 * warp_idx, exp_gate_suh + 128 * token_off, 0.088388347648f);
                had_hf_r_128_inner<true, false>(in_ptr, temp_state_u + 128 * warp_idx, exp_up_suh + 128 * token_off, 0.088388347648f);
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }
        auto gemm = [&] (const half* in_addr, half* out_addr, const uint16_t* trellis, int size_k, int size_n)
        {
            int size_m = token_count;
            if (token_count <= SMALL_MAX)
            {
                while (size_m > 0) { mt_gemm_inner<bits, 1, 4, 16, 256, 6, 2>(in_addr, trellis, out_addr, MIN(size_m, 64), size_k, size_n, locks); in_addr += 64 * size_k; out_addr += 64 * size_n; size_m -= 64; }
            }
            else if (token_count <= MID_MAX)
            {
                while (size_m > 0) { mt2_gemm_inner<bits, 1, 1, 3>(in_addr, trellis, out_addr, MIN(size_m, 128), size_k, size_n, locks); in_addr += 128 * size_k; out_addr += 128 * size_n; size_m -= 128; }
            }
            else
            {
                while (size_m > 0) { mt2_gemm_inner<bits, 1, 2, 3>(in_addr, trellis, out_addr, MIN(size_m, 256), size_k, size_n, locks); in_addr += 256 * size_k; out_addr += 256 * size_n; size_m -= 256; }
            }
        };
        gemm(temp_state_g, temp_intermediate_g, exp_gate_trellis, hidden_dim, intermediate_dim);
        gemm(temp_state_u, temp_intermediate_u, exp_up_trellis, hidden_dim, intermediate_dim);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        {
            const int warps_per_token = intermediate_dim / 128, total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner(temp_intermediate_g + 128 * warp_idx, temp_intermediate_u + 128 * warp_idx, temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off, exp_up_svh + 128 * token_off, exp_down_suh + 128 * token_off, 0.088388347648f, act_limit, act_function);
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }
        gemm(temp_intermediate_g, temp_state_g, exp_down_trellis, intermediate_dim, hidden_dim);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        {
            const int warps_per_token = hidden_dim / 128, total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start; const half* weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token]; half weight = weights[warp_idx / warps_per_token]; int token_off = warp_idx % warps_per_token;
                float* out_ptr = output_state + (size_t) token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_d_inner(temp_state_g + 128 * warp_idx, out_ptr, exp_down_svh + 128 * token_off, 0.088388347648f * __half2float(weight));
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Host side

typedef void (*fp_mt_kernel) (EXL3_MOE_KERNEL_ARGS);

struct MtVariant { fp_mt_kernel fn; int tbm; int tk; int tn; int sh; int fr; const char* name; int threads; };

// Variants (all 4-bit, mcg). m_tile = 16 * tbm rows per pass.
static MtVariant g_variants[] =
{
    { exl3_moe_mt_kernel<4, 1, 32, 256, 3, 3>, 1, 32, 256, 3, 3, "m16_k32_n256 (upstream shape)", 0 },
    { exl3_moe_mt_kernel<4, 2, 32, 256, 3, 3>, 2, 32, 256, 3, 3, "m32_k32_n256", 0 },
    { exl3_moe_mt_kernel<4, 4, 32, 128, 3, 2>, 4, 32, 128, 3, 2, "m64_k32_n128", 0 },
    { exl3_moe_mt_kernel<4, 4, 32, 256, 3, 2>, 4, 32, 256, 3, 2, "m64_k32_n256", 0 },
    { exl3_moe_mt_kernel<4, 8, 16, 128, 6, 2>, 8, 16, 128, 6, 2, "m128_k16_n128", 0 },
    { exl3_moe_mt_kernel<4, 4, 16, 256, 6, 2>, 4, 16, 256, 6, 2, "m64_k16_n256", 0 },
    { exl3_moe_mt2_kernel<4, 1, 3>, 8, 32, 128, 3, 1, "smemB_m128_k32_n128", 256 },
    { exl3_moe_mt2_kernel<4, 2, 3>, 16, 32, 128, 3, 1, "smemB_m256_k32_n128", 256 },
    { exl3_moe_mt3_kernel<4, 96, 160>, 16, 32, 128, 3, 1, "auto (<=96: m64 | <=160: smemB m128 | else smemB m256)", 256 },
};
static const int g_num_variants = sizeof(g_variants) / sizeof(g_variants[0]);
static std::set<void*> g_attr_set;

int mt_num_variants() { return g_num_variants; }
std::string mt_variant_name(int i) { TORCH_CHECK(i >= 0 && i < g_num_variants, "bad variant"); return g_variants[i].name; }
int mt_variant_m_tile(int i) { TORCH_CHECK(i >= 0 && i < g_num_variants, "bad variant"); return 16 * g_variants[i].tbm; }
int mt_locks_ints() { return MT_LOCKS_INTS; }

void exl3_moe_mt
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,
    const int act_function,
    const int K,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const float act_limit,
    const at::Tensor& locks,
    const int variant
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(K == 4, "exl3_moe_mt: only 4-bit experts are instantiated");
    TORCH_CHECK(variant >= 0 && variant < g_num_variants, "exl3_moe_mt: bad variant");
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DIM(hidden_state, 2);
    size_t bsz = hidden_state.size(0);
    size_t hidden_dim = hidden_state.size(1);
    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);
    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    size_t num_experts = expert_count.size(0) - 1;
    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DIM(token_sorted, 1);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    TORCH_CHECK_DTYPE(weight_sorted, kHalf);
    size_t num_experts_per_tok = token_sorted.size(0) / bsz;
    TORCH_CHECK_DTYPE(temp_state_g, kHalf);
    TORCH_CHECK_DTYPE(temp_state_u, kHalf);
    TORCH_CHECK_DIM(temp_state_g, 3);
    TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
    TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
    size_t max_tokens_per_expert = temp_state_g.size(1);
    size_t concurrency = temp_state_g.size(0);
    TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf);
    TORCH_CHECK_DTYPE(temp_intermediate_u, kHalf);
    TORCH_CHECK_DIM(temp_intermediate_g, 3);
    TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
    TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
    size_t intermediate_dim = temp_intermediate_g.size(2);
    TORCH_CHECK(hidden_dim % 256 == 0 && intermediate_dim % 256 == 0, "dims must be multiples of 256");
    TORCH_CHECK_DTYPE(locks, kInt);
    TORCH_CHECK(locks.numel() >= MT_LOCKS_INTS, "locks tensor too small");
    TORCH_CHECK_DIM(gate_ptrs_trellis, 1);
    TORCH_CHECK(gate_ptrs_trellis.size(0) == (int64_t) num_experts, "gate tensors vs num_experts");

    int device; cudaGetDevice(&device);
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= (size_t) num_sms, "concurrency too high for device");

    const MtVariant& v = g_variants[variant];
    TORCH_CHECK(hidden_dim % v.tn == 0 && intermediate_dim % v.tn == 0, "dims vs TILESIZE_N");
    TORCH_CHECK(hidden_dim % v.tk == 0 && intermediate_dim % v.tk == 0, "dims vs TILESIZE_K");
    int block_dim = v.threads ? v.threads : EXL3_GEMM_BASE_THREADS * v.tk / 16;
    dim3 grid_dim(MOE_SMS_PER_EXPERT, 1, concurrency);

    if (g_attr_set.find((void*) v.fn) == g_attr_set.end())
    {
        cuda_check(cudaFuncSetAttribute(v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        g_attr_set.insert((void*) v.fn);
    }

    void* _hidden_state = hidden_state.data_ptr();
    void* _temp_state_g = temp_state_g.data_ptr();
    void* _temp_state_u = temp_state_u.data_ptr();
    void* _temp_intermediate_g = temp_intermediate_g.data_ptr();
    void* _temp_intermediate_u = temp_intermediate_u.data_ptr();
    void* _output_state = output_state.data_ptr();
    void* _gate_ptrs_trellis = gate_ptrs_trellis.data_ptr();
    void* _gate_ptrs_suh = gate_ptrs_suh.data_ptr();
    void* _gate_ptrs_svh = gate_ptrs_svh.data_ptr();
    void* _up_ptrs_trellis = up_ptrs_trellis.data_ptr();
    void* _up_ptrs_suh = up_ptrs_suh.data_ptr();
    void* _up_ptrs_svh = up_ptrs_svh.data_ptr();
    void* _down_ptrs_trellis = down_ptrs_trellis.data_ptr();
    void* _down_ptrs_suh = down_ptrs_suh.data_ptr();
    void* _down_ptrs_svh = down_ptrs_svh.data_ptr();
    void* _expert_count = expert_count.data_ptr();
    void* _token_sorted = token_sorted.data_ptr();
    void* _weight_sorted = weight_sorted.data_ptr();
    int* _locks = (int*) locks.data_ptr();
    int i_hidden = (int) hidden_dim, i_inter = (int) intermediate_dim, i_ne = (int) num_experts,
        i_nept = (int) num_experts_per_tok, i_mtpe = (int) max_tokens_per_expert, i_conc = (int) concurrency;
    int Kg = K, Ku = K, Kd = K;
    float f_limit = act_limit;
    int i_act = act_function;

    void* kernelArgs[] =
    {
        &_hidden_state, &_temp_state_g, &_temp_state_u, &_temp_intermediate_g, &_temp_intermediate_u, &_output_state,
        &_gate_ptrs_trellis, &_gate_ptrs_suh, &_gate_ptrs_svh, &_up_ptrs_trellis, &_up_ptrs_suh, &_up_ptrs_svh,
        &_down_ptrs_trellis, &_down_ptrs_suh, &_down_ptrs_svh,
        &_expert_count, &_token_sorted, &_weight_sorted,
        (void*) &i_hidden, (void*) &i_inter, (void*) &i_ne, (void*) &i_nept, (void*) &i_mtpe, (void*) &i_conc,
        (void*) &f_limit, (void*) &i_act, (void*) &Kg, (void*) &Ku, (void*) &Kd, (void*) &_locks
    };
    cuda_check(cudaLaunchKernel((void*) v.fn, grid_dim, block_dim, kernelArgs, SMEM_MAX, stream));
    cuda_check(cudaPeekAtLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_moe_mt", &exl3_moe_mt, "M-tiled EXL3 fused MoE");
    m.def("num_variants", &mt_num_variants);
    m.def("variant_name", &mt_variant_name);
    m.def("variant_m_tile", &mt_variant_m_tile);
    m.def("locks_ints", &mt_locks_ints);
}
