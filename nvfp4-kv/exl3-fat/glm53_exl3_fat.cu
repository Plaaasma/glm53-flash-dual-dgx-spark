// glm53_exl3_fat.cu -- EXL3 routed-expert PREFILL kernels for vLLM's exl3.py (GLM-5.3-Flash, 2x DGX Spark, sm_121a).
//
// Provenance and licences
//   * The expert GEMM kernels ("fat" and "fast2"), the trellis decode helper, the fragment Hadamard and the one-matrix
//     input rotation are ported from jayleaton's glm53-tensorfold-spark kit (patches/0080 fast2 and patches/0170 fat,
//     file src/tensorfold/families/glm5_next/cuda/exl3_fast.cu of the patched TensorFold tree; Copyright 2026 Jay
//     Leaton, Apache License 2.0, https://x.com/jayleaton). Those patches modify TensorFold (Copyright (c) 2026
//     TensorFold contributors, MIT License).
//   * fat's data movement (cp.async-staged trellis words, gate/up sharing one rotated input, XOR-swizzled row stages,
//     the ticket scheduler) is, per that kit's NOTICE, adapted from the grouped fat-expert MoE of the GLM-5.3-Flash
//     EXL3 2x DGX Spark serving kits: Reederey87/glm53-flash-exl3-2x-dgx-spark (Apache-2.0), based on the kit by
//     Mia's AI Lab (MIT License for contributions before 2026-09-07, Copyright (c) 2026 Mia's AI Lab).
//   * The EXL3 trellis / MCG codebook format and the 128-point Hadamard conventions are those of ExLlamaV3 by
//     turboderp (https://github.com/turboderp-org/exllamav3, MIT License).
//
// What was changed for vLLM (Apache-2.0 section 4(b) statement of changes):
//   * Experts are addressed through vLLM's per-expert pointer tables (layer._exl3_ptrs: one int64 data_ptr per expert
//     for trellis / suh / svh), so gate and up can live interleaved in w13_trellis [E, 2, K/16, N/16, 64] and no copy
//     of the 1.8 GB/layer weights is made. Tiles inside an expert keep the checkpoint layout [K/16][N/16][32 words].
//   * Grouping is a single device plan kernel (counting sort of the routed pairs by expert, per-expert pass prefix
//     sums for both GEMMs, ticket counters): no host sync anywhere on the path.
//   * Rotated rows (Xg) and the down input (Xd) are stored in expert-sorted order, so a pass reads contiguous rows.
//   * The gate/up epilogue keeps fp32 throughout (TensorFold's bf16 emulation roundings removed) and applies
//     exllamav3 exl3_moe's activation order by default: silu(g), then min(., limit), times clamp(u, -limit, limit)
//     (act_mode 2); act_mode 1 = vLLM python-loop order silu(min(g, limit)) * clamp(u).
//   * The down epilogue writes each routed pair's row as fp16 into the rotated-input buffer (dead after gate/up, so no
//     extra scratch; TensorFold writes fp32 Y, 256 MiB at 2,048 tokens), and combine_kernel sums each token's slots in
//     slot order in fp32: deterministic, every output element written once. A second mode accumulates into the fp32
//     output with vector atomics (as exllamav3's exl3_moe does); measured ~1 ms slower a layer at 2,048 tokens.
//   * Down work items can cover two adjacent 128-column blocks (mode 2: one row stage feeds both, half the items).
//   * fast2 reads one shared rotated input for gate and up (TensorFold's 0270 does the same by passing Xg twice);
//     here the ring holds a single matrix of rows. Both kernels use fat's ticket scheduler; fast2 runs 2 CTAs an SM.
//
// Arithmetic per output element is TensorFold's: every element is one chain of m16n8k16 fp16 x fp16 -> fp32 mma
// over ascending k tiles of the whole K (no split-K, no fp16 partials), the column block is one 128-point Hadamard
// block transformed in fp32 on the accumulators, so a routed pair's result depends only on its own row.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace glm53fat {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
constexpr int NB = 8;                                // n tiles of an item: 128 columns, one Hadamard block
constexpr int MAXE = 1024;                           // experts a layer (plan kernel: one thread an expert)

// -- the EXL3 format (exl3_fast.cu, verbatim) ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves: b0 = rows
// (k) 2t, 2t+1 | 2t+8, 2t+9 of column (n) g, b1 the same of column g + 8 (g = lane / 4, t = lane % 4).
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

// 128-point transform of one row, 4 values a lane (column 4 * lane + j), butterflies bit 0 to bit 6 (exllamav3's
// had_*_r_128 order: 4-element butterfly in registers, then xor-shuffles with lane masks 1..16).
__device__ __forceinline__ void fwht_row(float (&v)[4], int lane) {
    float a = v[0], b = v[1];
    v[0] = a + b; v[1] = a - b;
    a = v[2]; b = v[3];
    v[2] = a + b; v[3] = a - b;
    a = v[0]; b = v[2];
    v[0] = a + b; v[2] = a - b;
    a = v[1]; b = v[3];
    v[1] = a + b; v[3] = a - b;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
}

// the row ring's 16-byte chunk c of row r (CPR chunks a row): XOR-swizzled so the 8 rows of an ldmatrix phase hit
// 8 distinct bank groups (fat)
template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    static_assert(CPR == 8 || CPR == 4, "rows of 64 or 32 halves");
    return CPR == 8 ? (c ^ (r & 7)) : (c ^ ((r >> 1) & 3));
}

__device__ __forceinline__ void red_add4(float* p, float4 v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    atomicAdd(reinterpret_cast<float4*>(p), v);
#else
    atomicAdd(p + 0, v.x); atomicAdd(p + 1, v.y); atomicAdd(p + 2, v.z); atomicAdd(p + 3, v.w);
#endif
}

// SwiGLU with the limit. mode 2 (default): exllamav3 exl3_moe's order (had_hf_r_128_guad_inner): silu(g), then
// min(., limit), u clamped to [-limit, limit]. mode 1: vLLM's LinearEXL3 loop order: silu(min(g, limit)) * clamp(u).
// limit <= 0: no limit (exl3_moe's act_limit == 0).
__device__ __forceinline__ float swiglu(float g, float u, float limit, int mode) {
    if (limit > 0.f) {
        u = fminf(fmaxf(u, -limit), limit);
        if (mode == 1) {
            g = fminf(g, limit);
            return g / (1.f + expf(-g)) * u;
        }
        return fminf(g / (1.f + expf(-g)), limit) * u;
    }
    return g / (1.f + expf(-g)) * u;
}

// -- arguments of the expert kernels ----------------------------------------------------------------------------------
struct ExpArgs {
    const half* X;            // [pairs, K] fp16 rows in expert-sorted order (Xg for gate/up, Xd for down)
    const int64_t* t0;        // per-expert trellis base pointers (uint32 words [K/16][N/16][32]): gate / down
    const int64_t* t1;        // up (gate/up only)
    const int64_t* s0;        // gate/up: svh_g; down: svh_d  (per-expert fp16 pointers, N values)
    const int64_t* s1;        // gate/up: svh_u
    const int64_t* sd;        // gate/up: suh_d (the down projection's input sign vector, N values)
    const int* offs;          // [E + 1] first sorted pair of each expert
    const int* pfx;           // [E + 1] first pass of each expert (this kernel's members a pass)
    int* ticket;              // dynamic item counter (zeroed by the plan kernel), nullptr = static stride
    half* xd;                 // gate/up output [pairs, N] fp16 (expert-sorted)
    float* out;               // down output [T, N] fp32, accumulated: out[row] += w_pair * y_pair (atomics mode)
    half* y;                  // down output [pairs, N] fp16, expert-sorted, unweighted (combine mode; nullptr = atomics)
    const int* spair;         // [pairs] sorted position -> pair index p = row * topk + slot
    const float* wts;         // [T * topk] routing weights
    int K, N, E, topk, act_mode;
    float limit;
    int probe;                // timing only (outputs wrong): 1 = down stores instead of atomics, 2 = no down output
};

// Kernel modes. 0: gate/up (MATS = 2 matrices, gate and up, of column block nb, one shared input). 1: down (MATS = 1).
// 2: down over two adjacent column blocks a work item (MATS = 2 "matrices" = column blocks 2 nb and 2 nb + 1 of the
// down matrix, one row stage serving both: half the items, half the row traffic and pipeline fills of mode 1).
template <int MODE>
struct Mode {
    static constexpr int MATS = MODE == 1 ? 1 : 2;
    static constexpr int CB = MODE == 2 ? 2 : 1;            // 128-column blocks an item
    static constexpr bool DOWN = MODE != 0;
    __device__ static __forceinline__ int col_block(int nb, int m) { return MODE == 2 ? 2 * nb + m : nb; }
};

// Epilogue of one item (fat's / fast2's code): ER rows a round, accumulators -> ep[mat][row][col] (fp32), then one
// warp a row (and matrix, for down) runs the 128-point transforms and the formulas.
template <int MODE, int MTL, int NG, int NGR, int W>
__device__ __forceinline__ void epilogue(const ExpArgs& a, float (&acc)[MTL][NG][4], float* ep, int cnt, int base,
                                         int nb, int e, int warp, int lane, int mat, int slice, const int* row_sh,
                                         const float* wt_sh) {
    using M = Mode<MODE>;
    constexpr int LDE = 132, ER = NGR * 8, ROUNDS = NG / NGR;
    const int g = lane >> 2, t = lane & 3;
    const int N = a.N;
    const int c0 = 4 * lane;
#pragma unroll
    for (int rd = 0; rd < ROUNDS; ++rd) {
        if (rd) __syncthreads();
        if (ER * rd < cnt) {
#pragma unroll
            for (int n = 0; n < NGR; ++n) {
                const int ng = rd * NGR + n;
                float* Ep = ep + ((size_t)mat * ER + 8 * n + 2 * t) * LDE + slice * 16 * MTL + g;
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    Ep[l * 16] = acc[l][ng][0];
                    Ep[LDE + l * 16] = acc[l][ng][1];
                    Ep[l * 16 + 8] = acc[l][ng][2];
                    Ep[LDE + l * 16 + 8] = acc[l][ng][3];
                }
            }
        }
        __syncthreads();
        if constexpr (MODE == 0) {
            for (int r = warp; r < ER && ER * rd + r < cnt; r += W) {
                const int j = base + ER * rd + r;               // sorted pair position
                float v[4], w[4];
                *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * LDE + c0);
                *reinterpret_cast<float4*>(w) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                fwht_row(v, lane);
                fwht_row(w, lane);
                const half* sg = reinterpret_cast<const half*>(a.s0[e]) + nb * 128 + c0;
                const half* su = reinterpret_cast<const half*>(a.s1[e]) + nb * 128 + c0;
                const half* sdd = reinterpret_cast<const half*>(a.sd[e]) + nb * 128 + c0;
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const float gg = v[jj] * HAD_SCALE * __half2float(sg[jj]);
                    const float uu = w[jj] * HAD_SCALE * __half2float(su[jj]);
                    v[jj] = swiglu(gg, uu, a.limit, a.act_mode) * __half2float(sdd[jj]);
                }
                fwht_row(v, lane);
                half2* o = reinterpret_cast<half2*>(a.xd + (size_t)j * N + nb * 128 + c0);
                o[0] = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
                o[1] = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
            }
        } else {
            const half* svh = reinterpret_cast<const half*>(a.s0[e]);
            for (int q = warp; q < ER * M::MATS; q += W) {
                const int r = q / M::MATS, m = q % M::MATS;
                const int i = ER * rd + r;                      // member index within the pass
                if (i >= cnt) break;                             // q ascends: warp-uniform
                const int col = M::col_block(nb, m) * 128 + c0;
                float v[4];
                *reinterpret_cast<float4*>(v) =
                    *reinterpret_cast<const float4*>(ep + ((size_t)m * ER + r) * LDE + c0);
                fwht_row(v, lane);
                const half* sv = svh + col;
                if (a.y) {                                      // combine mode: this pair's row, fp16, unweighted
                    half2* yo = reinterpret_cast<half2*>(a.y + (size_t)(base + i) * N + col);
                    yo[0] = __halves2half2(__float2half_rn(v[0] * HAD_SCALE * __half2float(sv[0])),
                                           __float2half_rn(v[1] * HAD_SCALE * __half2float(sv[1])));
                    yo[1] = __halves2half2(__float2half_rn(v[2] * HAD_SCALE * __half2float(sv[2])),
                                           __float2half_rn(v[3] * HAD_SCALE * __half2float(sv[3])));
                    continue;
                }
                const float wt = wt_sh[i] * HAD_SCALE;
                float4 o;
                o.x = v[0] * wt * __half2float(sv[0]);
                o.y = v[1] * wt * __half2float(sv[1]);
                o.z = v[2] * wt * __half2float(sv[2]);
                o.w = v[3] * wt * __half2float(sv[3]);
                if (a.probe == 0) red_add4(a.out + (size_t)row_sh[i] * N + col, o);
                else if (a.probe == 1) *reinterpret_cast<float4*>(a.out + (size_t)row_sh[i] * N + col) = o;
                else if (o.x == 123.f) a.out[0] = o.y;          // probe 2: keep the math, drop the output
            }
        }
    }
}

// Down items: the output row and routing weight of each member, loaded once at the item's start (they are only read by
// the epilogue, after the K loop's barriers).
template <int MODE, int BM, int THREADS>
__device__ __forceinline__ void load_row_meta(const ExpArgs& a, int base, int cnt, int* row_sh, float* wt_sh) {
    if constexpr (Mode<MODE>::DOWN) {
        for (int i = threadIdx.x; i < cnt; i += THREADS) {
            const int p = __ldg(a.spair + base + i);
            row_sh[i] = p / a.topk;
            wt_sh[i] = __ldg(a.wts + p);
        }
    }
}

// -- fat (patches/0170): weights AND member rows through one cp.async ring; decoded weights are the mma A operand --------
// Items are (expert e, pass of BM members, column block(s) nb), claimed by ticket (fat's scheduler), t-major with nb
// fastest, so the column blocks of a pass read its rows from L2 together and an expert's passes read its weights
// together. Thread 0 walks e forward (items ascend per CTA).
template <int MODE, int MTL, int NG, int KS, int NSA, int NGR>
struct FatCfg {
    static constexpr int MATS = Mode<MODE>::MATS;
    static constexpr int BM = NG * 8, WPM = NB / MTL, W = MATS * WPM, THREADS = W * 32;
    static constexpr int CPR = KS * 2;                              // 16-byte chunks of a row in a stage
    static constexpr int LDA = KS * 16;                             // halves a row (swizzled, no padding)
    static constexpr int LDE = 132, ER = NGR * 8;
    static constexpr size_t A_BYTES = (size_t)BM * LDA * sizeof(half);               // one (shared) matrix of rows
    static constexpr size_t W_BYTES = (size_t)MATS * KS * NB * 32 * sizeof(uint32_t);
    static constexpr size_t STAGE = A_BYTES + W_BYTES;
    static constexpr size_t RING = (size_t)NSA * STAGE;
    static constexpr size_t EPI = (size_t)MATS * ER * LDE * sizeof(float);
    static constexpr size_t SMEM = RING > EPI ? RING : EPI;
    static_assert(A_BYTES % 16 == 0, "stage alignment");
};

template <int MODE, int MTL, int NG, int KS, int NSA, int NGR, int MINB>
__global__ void __launch_bounds__(FatCfg<MODE, MTL, NG, KS, NSA, NGR>::THREADS, MINB) fat_kernel(const ExpArgs a) {
    using C = FatCfg<MODE, MTL, NG, KS, NSA, NGR>;
    using M = Mode<MODE>;
    constexpr int MATS = C::MATS, W = C::W, THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, CPR = C::CPR;
    static_assert(NG % 2 == 0 && NG % NGR == 0 && NGR % 2 == 0, "n groups in pairs");
    extern __shared__ __align__(16) unsigned char smem[];
    float* ep = reinterpret_cast<float*>(smem);              // [MATS][ER][LDE], after the K loop
    __shared__ int item_sh, e_sh;
    __shared__ int row_sh[M::DOWN ? BM : 1];
    __shared__ float wt_sh[M::DOWN ? BM : 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;   // columns slice * 16 MTL .. + 16 MTL of the item
    const int K = a.K, N = a.N, E = a.E;
    const int KT = K >> 4, NTILES = N >> 4, S = KT / KS, NBLK = N / 128 / M::CB;
    const int total = a.pfx[E] * NBLK;
    const int lr = (lane & 7) + ((lane >> 4) << 3);
    const int lc = (lane >> 3) & 1;

    int u = 0;                                               // thread 0: the expert of the last item
    int next = blockIdx.x;                                   // static stride (no ticket)
    for (;;) {
        __syncthreads();                                     // the previous item is done with smem
        if (threadIdx.x == 0) {
            const int it = a.ticket ? atomicAdd(a.ticket, 1) : next;
            item_sh = it;
            if (it < total) {
                const int tt = it / NBLK;
                while (u + 1 < E && __ldg(a.pfx + u + 1) <= tt) ++u;
                e_sh = u;
            }
        }
        next += gridDim.x;
        __syncthreads();
        const int item = item_sh;
        if (item >= total) break;
        const int tt = item / NBLK, nb = item % NBLK;
        const int e = e_sh;
        const int base = __ldg(a.offs + e) + (tt - __ldg(a.pfx + e)) * BM;
        const int cnt = min(BM, __ldg(a.offs + e + 1) - base);     // members of this pass (>= 1)
        const uint32_t* tw0 = reinterpret_cast<const uint32_t*>(__ldg(a.t0 + e));
        const uint32_t* tw1 = MODE == 0 ? reinterpret_cast<const uint32_t*>(__ldg(a.t1 + e)) : tw0;
        load_row_meta<MODE, BM, THREADS>(a, base, cnt, row_sh, wt_sh);

        // stage s: the member rows' k range [s KS 16, (s + 1) KS 16) (zero-filled past the last member), then the
        // item's trellis words of k tiles s KS .. s KS + KS - 1, all MATS matrices, the NB column tiles of each block
        auto load_stage = [&](int s) {
            unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            half* ra = reinterpret_cast<half*>(st);
            const int k0 = s * KS * 16;
#pragma unroll
            for (int i = threadIdx.x; i < BM * CPR; i += THREADS) {
                const int r = i / CPR, c = i % CPR;
                const bool ok = r < cnt;
                const half* src = a.X + (size_t)(base + (ok ? r : 0)) * K + k0 + c * 8;
                cp_async16(ra + (size_t)r * LDA + swz<CPR>(r, c) * 8, src, ok);
            }
            uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::A_BYTES);
            constexpr int WCH = MATS * KS * NB * 8;          // 16-byte chunks: 8 a 16 x 16 tile
#pragma unroll
            for (int i = threadIdx.x; i < WCH; i += THREADS) {
                const int q = i & 7, n = (i >> 3) % NB, kk = (i / (8 * NB)) % KS, m = i / (8 * NB * KS);
                const uint32_t* src = (m ? tw1 : tw0) +
                                      (((size_t)(s * KS + kk)) * NTILES + M::col_block(nb, m) * NB + n) * 32 + q * 4;
                cp_async16(rw + ((m * KS + kk) * NB + n) * 32 + q * 4, src, true);
            }
        };

        float acc[MTL][NG][4];
#pragma unroll
        for (int l = 0; l < MTL; ++l)
#pragma unroll
            for (int n = 0; n < NG; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

#pragma unroll
        for (int s = 0; s < NSA - 1; ++s) {
            if (s < S) load_stage(s);
            cp_commit();
        }
        cp_wait<NSA - 2>();
        __syncthreads();
        for (int s = 0; s < S; ++s) {
            if (s + NSA - 1 < S) load_stage(s + NSA - 1);
            cp_commit();
            const unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            const half* A = reinterpret_cast<const half*>(st);
            const uint32_t* Wd = reinterpret_cast<const uint32_t*>(st + C::A_BYTES) + (size_t)mat * KS * NB * 32 +
                                 slice * MTL * 32 + lane;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                uint32_t af[MTL][4];
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    uint32_t b0[2], b1[2];
                    decode_tile(Wd[(kk * NB + l) * 32], lane, b0, b1);
                    af[l][0] = b0[0]; af[l][1] = b1[0]; af[l][2] = b0[1]; af[l][3] = b1[1];
                }
                const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
                for (int np = 0; np < NG / 2; ++np) {
                    if (16 * np < cnt) {                         // warp-uniform
                        uint32_t b[4];
                        ldsm4(b, A + (16 * np + lr) * LDA + chunk * 8);
#pragma unroll
                        for (int l = 0; l < MTL; ++l) {
                            mma16816(acc[l][2 * np], af[l], b[0], b[1]);
                            mma16816(acc[l][2 * np + 1], af[l], b[2], b[3]);
                        }
                    }
                }
            }
            cp_wait<NSA - 2>();
            __syncthreads();
        }
        epilogue<MODE, MTL, NG, NGR, W>(a, acc, ep, cnt, base, nb, e, warp, lane, mat, slice, row_sh, wt_sh);
    }
}

// -- fast2 (patches/0080): weights decoded straight from registers loaded one stage ahead; only rows in the ring -------
template <int MODE, int MTL, int NG, int KS, int NSA, int NGR>
struct F2Cfg {
    static constexpr int MATS = Mode<MODE>::MATS;
    static constexpr int BM = NG * 8, WPM = NB / MTL, W = MATS * WPM, THREADS = W * 32;
    static constexpr int LDA = KS * 16 + 8, LDE = 132, ER = NGR * 8;
    static constexpr size_t RING = (size_t)NSA * BM * LDA * sizeof(half);           // one (shared) matrix of rows
    static constexpr size_t EPI = (size_t)MATS * ER * LDE * sizeof(float);
    static constexpr size_t SMEM = RING > EPI ? RING : EPI;
};

template <int MODE, int MTL, int NG, int KS, int NSA, int NGR, int MINB>
__global__ void __launch_bounds__(F2Cfg<MODE, MTL, NG, KS, NSA, NGR>::THREADS, MINB) fast2_kernel(const ExpArgs a) {
    using C = F2Cfg<MODE, MTL, NG, KS, NSA, NGR>;
    using M = Mode<MODE>;
    constexpr int W = C::W, THREADS = C::THREADS, BM = C::BM, LDA = C::LDA;
    static_assert(NG % 2 == 0 && NG % NGR == 0 && NGR % 2 == 0, "n groups in pairs");
    extern __shared__ __align__(16) unsigned char smem[];
    half* ring = reinterpret_cast<half*>(smem);              // [NSA][BM][LDA]
    float* ep = reinterpret_cast<float*>(smem);              // [MATS][ER][LDE], after the K loop
    __shared__ int item_sh, e_sh;
    __shared__ int row_sh[M::DOWN ? BM : 1];
    __shared__ float wt_sh[M::DOWN ? BM : 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;
    const int K = a.K, N = a.N, E = a.E;
    const int KT = K >> 4, NTILES = N >> 4, S = KT / KS, NBLK = N / 128 / M::CB;
    const int total = a.pfx[E] * NBLK;

    int u = 0;
    int next = blockIdx.x;
    for (;;) {
        __syncthreads();
        if (threadIdx.x == 0) {
            const int it = a.ticket ? atomicAdd(a.ticket, 1) : next;
            item_sh = it;
            if (it < total) {
                const int tt = it / NBLK;
                while (u + 1 < E && __ldg(a.pfx + u + 1) <= tt) ++u;
                e_sh = u;
            }
        }
        next += gridDim.x;
        __syncthreads();
        const int item = item_sh;
        if (item >= total) break;
        const int tt = item / NBLK, nb = item % NBLK;
        const int e = e_sh;
        const int base = __ldg(a.offs + e) + (tt - __ldg(a.pfx + e)) * BM;
        const int cnt = min(BM, __ldg(a.offs + e + 1) - base);
        load_row_meta<MODE, BM, THREADS>(a, base, cnt, row_sh, wt_sh);

        auto load_a = [&](int s) {
            constexpr int CPR = KS * 2;
            const int k0 = s * KS * 16;
            half* dst = ring + (size_t)(s % NSA) * BM * LDA;
#pragma unroll
            for (int i = threadIdx.x; i < BM * CPR; i += THREADS) {
                const int r = i / CPR, c = i % CPR;
                const bool ok = r < cnt;
                const half* src = a.X + (size_t)(base + (ok ? r : 0)) * K + k0 + c * 8;
                cp_async16(dst + (size_t)r * LDA + c * 8, src, ok);
            }
        };
        // this warp's trellis words: column tiles col_block * 8 + slice * MTL + l, k tile kt
        const uint32_t* tw = reinterpret_cast<const uint32_t*>(__ldg((MODE == 0 && mat ? a.t1 : a.t0) + e)) +
                             (size_t)(M::col_block(nb, mat) * NB + slice * MTL) * 32 + lane;
        uint32_t wr[KS][MTL];
        auto load_w = [&](int kt, uint32_t (&w)[MTL]) {
#pragma unroll
            for (int l = 0; l < MTL; ++l) w[l] = __ldg(tw + ((size_t)kt * NTILES + l) * 32);
        };

        float acc[MTL][NG][4];
#pragma unroll
        for (int l = 0; l < MTL; ++l)
#pragma unroll
            for (int n = 0; n < NG; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

#pragma unroll
        for (int s = 0; s < NSA - 1; ++s) {
            if (s < S) load_a(s);
            cp_commit();
        }
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) load_w(kk, wr[kk]);
        cp_wait<NSA - 2>();
        __syncthreads();
        for (int s = 0; s < S; ++s) {
            if (s + NSA - 1 < S) load_a(s + NSA - 1);
            cp_commit();
            const half* A = ring + (size_t)(s % NSA) * BM * LDA;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                uint32_t af[MTL][4];
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    uint32_t b0[2], b1[2];
                    decode_tile(wr[kk][l], lane, b0, b1);
                    af[l][0] = b0[0]; af[l][1] = b1[0]; af[l][2] = b0[1]; af[l][3] = b1[1];
                }
                if (s + 1 < S) load_w((s + 1) * KS + kk, wr[kk]);
#pragma unroll
                for (int np = 0; np < NG / 2; ++np) {
                    if (16 * np < cnt) {                         // warp-uniform
                        uint32_t b[4];
                        ldsm4(b, A + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * LDA + kk * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
                        for (int l = 0; l < MTL; ++l) {
                            mma16816(acc[l][2 * np], af[l], b[0], b[1]);
                            mma16816(acc[l][2 * np + 1], af[l], b[2], b[3]);
                        }
                    }
                }
            }
            cp_wait<NSA - 2>();
            __syncthreads();
        }
        epilogue<MODE, MTL, NG, NGR, W>(a, acc, ep, cnt, base, nb, e, warp, lane, mat, slice, row_sh, wt_sh);
    }
}

// -- plan: counting sort of the routed pairs by expert, pass prefix sums, ticket reset (one block) --------------------
// meta layout (int32): offs[E + 1] | pfx_gu[E + 1] | pfx_dn[E + 1] | tickets[8] | spair[P] | inv[P]
// (inv[p] = sorted position of pair p, -1 for a skipped pair)
__device__ __forceinline__ int block_incl_scan(int v, int* sh) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int d = 1; d < MAXE; d <<= 1) {
        const int add = tid >= d ? sh[tid - d] : 0;
        __syncthreads();
        sh[tid] += add;
        __syncthreads();
    }
    const int r = sh[tid];
    __syncthreads();
    return r;
}

__global__ void __launch_bounds__(MAXE) plan_kernel(const int64_t* __restrict__ local, int P, int E, int bm_gu,
                                                    int bm_dn, int* __restrict__ offs, int* __restrict__ pfx_gu,
                                                    int* __restrict__ pfx_dn, int* __restrict__ tickets,
                                                    int* __restrict__ spair, int* __restrict__ inv) {
    __shared__ int cnt[MAXE];
    __shared__ int scan[MAXE];
    const int tid = threadIdx.x;
    cnt[tid] = 0;
    if (tid < 8) tickets[tid] = 0;
    __syncthreads();
    for (int p = tid; p < P; p += MAXE) {
        const int64_t e = local[p];
        if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    const int c = tid < E ? cnt[tid] : 0;
    const int i0 = block_incl_scan(c, scan);
    const int i1 = block_incl_scan((c + bm_gu - 1) / bm_gu, scan);
    const int i2 = block_incl_scan((c + bm_dn - 1) / bm_dn, scan);
    if (tid < E) {
        offs[tid] = i0 - c;
        pfx_gu[tid] = i1 - (c + bm_gu - 1) / bm_gu;
        pfx_dn[tid] = i2 - (c + bm_dn - 1) / bm_dn;
        cnt[tid] = i0 - c;                                   // cursor
    }
    if (tid == E - 1) {
        offs[E] = i0;
        pfx_gu[E] = i1;
        pfx_dn[E] = i2;
    }
    __syncthreads();
    for (int p = tid; p < P; p += MAXE) {
        const int64_t e = local[p];
        int j = -1;
        if (e >= 0 && e < E) {
            j = atomicAdd(&cnt[e], 1);
            spair[j] = p;
        }
        inv[p] = j;
    }
}

// out[t] = sum over slots k (ascending) of w[t, k] * Y[inv[t * topk + k]], fp32, 8 columns a thread. Deterministic.
__global__ void __launch_bounds__(256) combine_kernel(const half* __restrict__ y, const int* __restrict__ inv,
                                                      const float* __restrict__ wts, float* __restrict__ out, int T,
                                                      int N, int topk) {
    const int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    const int cpr = N / 8;
    if (idx >= (int64_t)T * cpr) return;
    const int t = (int)(idx / cpr), c = (int)(idx % cpr) * 8;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < topk; ++k) {
        const int j = __ldg(inv + (size_t)t * topk + k);
        if (j < 0) continue;
        const float w = __ldg(wts + (size_t)t * topk + k);
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(y + (size_t)j * N + c));
        const half2* h = reinterpret_cast<const half2*>(&raw);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float2 f = __half22float2(h[q]);
            acc[2 * q] += w * f.x;
            acc[2 * q + 1] += w * f.y;
        }
    }
    float4* o = reinterpret_cast<float4*>(out + (size_t)t * N + c);
    o[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    o[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

__global__ void zero_tickets_kernel(int* tickets) {
    if (threadIdx.x < 8) tickets[threadIdx.x] = 0;
}

// -- input rotation: Xg[j] = fp16(H(x[row(j)] * suh_g[e(j)]) / sqrt(128)), j in expert-sorted order ------------------
// (exl3_fast.cu rot_in1_kernel's code, one (pair, 128-block of K) a warp, 8 warps a block)
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename TX>
__device__ __forceinline__ float to_f(TX v);
template <>
__device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <>
__device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

template <typename TX>
__global__ void __launch_bounds__(256) rot_kernel(const TX* __restrict__ x, int64_t x_stride,
                                                  const int64_t* __restrict__ local, const int* __restrict__ spair,
                                                  const int* __restrict__ offs, const int64_t* __restrict__ suh_ptrs,
                                                  half* __restrict__ xg, int K, int topk, int E, int64_t items) {
    const int64_t item = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= items) return;                     // warp-uniform
    const int KB = K / 128;
    const int j = (int)(item / KB), blk = (int)(item % KB);
    if (j >= __ldg(offs + E)) return;              // past the valid pairs
    const int lane = threadIdx.x & 31;
    const int p = __ldg(spair + j);
    const int row = p / topk;
    const int64_t e = __ldg(local + p);
    const half* suh = reinterpret_cast<const half*>(__ldg(suh_ptrs + e)) + blk * 128 + 4 * lane;
    const TX* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) v[jj] = to_f<TX>(xr[jj]) * __half2float(suh[jj]);
    fwht128(v, lane);
    half2* o = reinterpret_cast<half2*>(xg + (size_t)j * K + blk * 128 + 4 * lane);
    o[0] = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
    o[1] = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
}

// gate and up input sign vectors equal for every expert? (one rotated input serves both)
__global__ void suh_eq_kernel(const int64_t* __restrict__ g, const int64_t* __restrict__ u, int K,
                              int* __restrict__ flag) {
    const int e = blockIdx.x;
    const uint16_t* a = reinterpret_cast<const uint16_t*>(g[e]);
    const uint16_t* b = reinterpret_cast<const uint16_t*>(u[e]);
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        if (a[i] != b[i]) atomicExch(flag, 0);
}

// -- configurations ---------------------------------------------------------------------------------------------------
// (MODE, MTL, NG, KS, NSA, NGR): TensorFold's measured tilings (FAT_GU / FAT_DN, FAST2_GU / FAST2_DN): 2 column tiles a
// warp, 64 members an item, 4 k tiles a stage, 3 stages. fat runs 2 CTAs an SM when two fit 100 KB of shared memory.
template <int MODE, int MTL, int NG, int KS, int NSA, int NGR>
constexpr int fat_minb() {
    return 2 * (FatCfg<MODE, MTL, NG, KS, NSA, NGR>::SMEM + 1024 + 1024) <= 102400 ? 2 : 1;
}

struct KernInfo {
    const void* fn;
    int threads;
    size_t smem;
    int bm;
    int cb;                   // column blocks an item
    const char* name;
};

#define FAT_KI(MODE, MTL, NG, KS, NSA, NGR, NAME)                                                                        \
    KernInfo {                                                                                                           \
        (const void*)fat_kernel<MODE, MTL, NG, KS, NSA, NGR, fat_minb<MODE, MTL, NG, KS, NSA, NGR>()>,                   \
            FatCfg<MODE, MTL, NG, KS, NSA, NGR>::THREADS, FatCfg<MODE, MTL, NG, KS, NSA, NGR>::SMEM,                     \
            FatCfg<MODE, MTL, NG, KS, NSA, NGR>::BM, Mode<MODE>::CB, NAME                                                \
    }
#define F2_KI(MODE, MTL, NG, KS, NSA, NGR, NAME) F2_KIB(MODE, MTL, NG, KS, NSA, NGR, 2, NAME)
#define F2_KIB(MODE, MTL, NG, KS, NSA, NGR, MINB, NAME)                                                                  \
    KernInfo {                                                                                                           \
        (const void*)fast2_kernel<MODE, MTL, NG, KS, NSA, NGR, MINB>, F2Cfg<MODE, MTL, NG, KS, NSA, NGR>::THREADS,       \
            F2Cfg<MODE, MTL, NG, KS, NSA, NGR>::SMEM, F2Cfg<MODE, MTL, NG, KS, NSA, NGR>::BM, Mode<MODE>::CB, NAME       \
    }

// gate/up kernels (mode 0) and down kernels (modes 1, 2); index = the kernel id passed from Python. Names: the first
// word is what GLM53_EXL3_FAT_KERNEL accepts.
static const KernInfo g_gu[] = {
    FAT_KI(0, 2, 8, 4, 3, 4, "fat (gu m64 ks4 s3)"),
    F2_KI(0, 2, 8, 4, 3, 4, "fast2 (gu m64 ks4 s3)"),
    FAT_KI(0, 2, 8, 4, 4, 4, "fat-s4 (gu m64 ks4 s4)"),
    F2_KI(0, 2, 8, 4, 4, 4, "fast2-s4 (gu m64 ks4 s4)"),
    F2_KI(0, 2, 8, 4, 5, 4, "fast2-s5 (gu m64 ks4 s5)"),
    F2_KIB(0, 2, 8, 4, 4, 4, 1, "fast2-s4r (gu m64 ks4 s4, registers for 1 CTA an SM)"),
};
static const KernInfo g_dn[] = {
    FAT_KI(1, 2, 8, 4, 3, 8, "fat (dn m64 ks4 s3)"),
    F2_KI(1, 2, 8, 4, 3, 8, "fast2 (dn m64 ks4 s3)"),
    FAT_KI(2, 2, 8, 4, 3, 4, "fat-cb2 (dn m64 ks4 s3, 2 column blocks an item)"),
    F2_KI(2, 2, 8, 4, 3, 4, "fast2-cb2 (dn m64 ks4 s3, 2 column blocks an item)"),
    F2_KI(1, 2, 8, 4, 4, 8, "fast2-s4 (dn m64 ks4 s4)"),
    F2_KI(2, 2, 8, 4, 4, 4, "fast2-cb2s4 (dn m64 ks4 s4, 2 column blocks an item)"),
    F2_KI(1, 2, 8, 4, 3, 4, "fast2-r32 (dn m64 ks4 s3, 32-row epilogue rounds)"),
    F2_KI(1, 2, 8, 4, 4, 4, "fast2-r32s4 (dn m64 ks4 s4, 32-row epilogue rounds)"),
};
static constexpr int N_GU = sizeof(g_gu) / sizeof(g_gu[0]);
static constexpr int N_DN = sizeof(g_dn) / sizeof(g_dn[0]);

static std::mutex g_mu;
static std::map<std::pair<const void*, int>, int> g_grid;   // (kernel, device) -> CTAs (SMs x CTAs an SM)

static int grid_for(const KernInfo& k, int device) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto key = std::make_pair(k.fn, device);
    auto it = g_grid.find(key);
    if (it != g_grid.end()) return it->second;
    C10_CUDA_CHECK(cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)k.smem));
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k.fn, k.threads, k.smem));
    TORCH_CHECK(per_sm > 0, "glm53_exl3_fat: ", k.name, " does not fit an SM");
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const int g = sms * per_sm;
    g_grid[key] = g;
    return g;
}

}  // namespace glm53fat

using namespace glm53fat;

static void check_ptr_table(const at::Tensor& t, int64_t E, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kLong && t.is_contiguous() && t.dim() == 1 && t.numel() == E,
                "glm53_exl3_fat: ", name, " must be a contiguous CUDA int64 [E] pointer table");
}

// meta int32 elements needed for E experts and P pairs
int64_t meta_ints(int64_t E, int64_t P) { return 3 * (E + 1) + 8 + 2 * P; }

// The whole prefill MoE of one layer: plan, rotation, gate/up (+SwiGLU, down rotation), down, combine.
// combine = true: down writes each pair's fp16 row into xg (dead after gate/up) and combine_kernel writes every element
// of out (no zeroing needed; deterministic). combine = false: down accumulates into out with fp32 atomics (out must
// be zeroed by the caller). stage_mask: 1 plan, 2 rotation, 4 gate/up, 8 down, 16 combine (31 = all).
void moe_prefill(const at::Tensor& x, const at::Tensor& local, const at::Tensor& wts, at::Tensor& out, at::Tensor& xg,
                 at::Tensor& xd, at::Tensor& meta, const at::Tensor& g_tr, const at::Tensor& u_tr,
                 const at::Tensor& d_tr, const at::Tensor& g_suh, const at::Tensor& g_svh, const at::Tensor& u_svh,
                 const at::Tensor& d_suh, const at::Tensor& d_svh, int64_t topk, double limit, int64_t act_mode,
                 int64_t kern_gu, int64_t kern_dn, int64_t stage_mask, bool ticket, int64_t probe, bool combine) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.stride(1) == 1, "x: CUDA [T, K] with unit column stride");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "x: bf16 or fp16");
    const int64_t T = x.size(0), K = x.size(1);
    TORCH_CHECK(topk >= 1 && topk <= 32, "topk");
    const int64_t P = T * topk;
    TORCH_CHECK(local.is_cuda() && local.scalar_type() == at::kLong && local.is_contiguous() && local.numel() == P,
                "local: contiguous int64 [T * topk]");
    TORCH_CHECK(wts.is_cuda() && wts.scalar_type() == at::kFloat && wts.is_contiguous() && wts.numel() == P,
                "wts: contiguous fp32 [T * topk]");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.dim() == 2 &&
                    out.size(0) == T && out.size(1) == K,
                "out: contiguous fp32 [T, K]");
    TORCH_CHECK(xg.is_cuda() && xg.scalar_type() == at::kHalf && xg.is_contiguous() && xg.dim() == 2 &&
                    xg.size(0) >= P && xg.size(1) == K,
                "xg: contiguous fp16 [>= T * topk, K]");
    TORCH_CHECK(xd.is_cuda() && xd.scalar_type() == at::kHalf && xd.is_contiguous() && xd.dim() == 2 &&
                    xd.size(0) >= P,
                "xd: contiguous fp16 [>= T * topk, N]");
    const int64_t N = xd.size(1);
    const int64_t E = g_tr.numel();
    TORCH_CHECK(E >= 1 && E <= MAXE, "experts: 1..", MAXE);
    check_ptr_table(g_tr, E, "gate trellis");
    check_ptr_table(u_tr, E, "up trellis");
    check_ptr_table(d_tr, E, "down trellis");
    check_ptr_table(g_suh, E, "gate suh");
    check_ptr_table(g_svh, E, "gate svh");
    check_ptr_table(u_svh, E, "up svh");
    check_ptr_table(d_suh, E, "down suh");
    check_ptr_table(d_svh, E, "down svh");
    TORCH_CHECK(meta.is_cuda() && meta.scalar_type() == at::kInt && meta.is_contiguous() &&
                    meta.numel() >= meta_ints(E, P),
                "meta: contiguous int32 [>= meta_ints(E, P)]");
    TORCH_CHECK(kern_gu >= 0 && kern_gu < N_GU && kern_dn >= 0 && kern_dn < N_DN, "kernel ids");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0 && K % 64 == 0 && N % 64 == 0, "K, N: multiples of 128");
    TORCH_CHECK((K / 128) % g_dn[kern_dn].cb == 0, "down kernel: column blocks an item must divide K / 128");
    TORCH_CHECK(act_mode == 1 || act_mode == 2, "act_mode: 1 or 2");
    TORCH_CHECK(P < (int64_t)1 << 30 && P * K < (int64_t)1 << 40, "too many pairs");
    if (T == 0) return;

    int device = x.get_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    int* m = meta.data_ptr<int>();
    int* offs = m;
    int* pfx_gu = m + (E + 1);
    int* pfx_dn = m + 2 * (E + 1);
    int* tickets = m + 3 * (E + 1);
    int* spair = tickets + 8;
    int* inv = spair + P;
    const KernInfo& kg = g_gu[kern_gu];
    const KernInfo& kd = g_dn[kern_dn];

    if (stage_mask & 1) {
        plan_kernel<<<1, MAXE, 0, stream>>>(local.data_ptr<int64_t>(), (int)P, (int)E, kg.bm, kd.bm, offs, pfx_gu,
                                            pfx_dn, tickets, spair, inv);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (stage_mask & 12) {
        zero_tickets_kernel<<<1, 32, 0, stream>>>(tickets);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stage_mask & 2) {
        const int64_t items = P * (K / 128);
        const unsigned blocks = (unsigned)((items + 7) / 8);
        if (x.scalar_type() == at::kBFloat16)
            rot_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0), local.data_ptr<int64_t>(), spair,
                offs, g_suh.data_ptr<int64_t>(), reinterpret_cast<half*>(xg.data_ptr()), (int)K, (int)topk, (int)E,
                items);
        else
            rot_kernel<half><<<blocks, 256, 0, stream>>>(
                reinterpret_cast<const half*>(x.data_ptr()), x.stride(0), local.data_ptr<int64_t>(), spair, offs,
                g_suh.data_ptr<int64_t>(), reinterpret_cast<half*>(xg.data_ptr()), (int)K, (int)topk, (int)E, items);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    ExpArgs a{};
    a.offs = offs;
    a.spair = spair;
    a.wts = wts.data_ptr<float>();
    a.E = (int)E;
    a.topk = (int)topk;
    a.act_mode = (int)act_mode;
    a.limit = (float)limit;
    a.probe = (int)probe;
    if (stage_mask & 4) {
        ExpArgs g = a;
        g.X = reinterpret_cast<const half*>(xg.data_ptr());
        g.t0 = g_tr.data_ptr<int64_t>();
        g.t1 = u_tr.data_ptr<int64_t>();
        g.s0 = g_svh.data_ptr<int64_t>();
        g.s1 = u_svh.data_ptr<int64_t>();
        g.sd = d_suh.data_ptr<int64_t>();
        g.pfx = pfx_gu;
        g.ticket = ticket ? tickets + 0 : nullptr;
        g.xd = reinterpret_cast<half*>(xd.data_ptr());
        g.out = nullptr;
        g.K = (int)K;
        g.N = (int)N;
        void* args[] = {&g};
        C10_CUDA_CHECK(cudaLaunchKernel(kg.fn, dim3(grid_for(kg, device)), dim3(kg.threads), args, kg.smem, stream));
    }
    if (stage_mask & 8) {
        ExpArgs d = a;
        d.X = reinterpret_cast<const half*>(xd.data_ptr());
        d.t0 = d_tr.data_ptr<int64_t>();
        d.t1 = d.t0;
        d.s0 = d_svh.data_ptr<int64_t>();
        d.s1 = d.s0;
        d.sd = d.s0;
        d.pfx = pfx_dn;
        d.ticket = ticket ? tickets + 1 : nullptr;
        d.xd = nullptr;
        d.out = out.data_ptr<float>();
        d.y = combine ? reinterpret_cast<half*>(xg.data_ptr()) : nullptr;
        d.K = (int)N;
        d.N = (int)K;
        void* args[] = {&d};
        C10_CUDA_CHECK(cudaLaunchKernel(kd.fn, dim3(grid_for(kd, device)), dim3(kd.threads), args, kd.smem, stream));
    }
    if ((stage_mask & 16) && combine) {
        const int64_t threads = T * (K / 8);
        combine_kernel<<<(unsigned)((threads + 255) / 256), 256, 0, stream>>>(
            reinterpret_cast<const half*>(xg.data_ptr()), inv, wts.data_ptr<float>(), out.data_ptr<float>(), (int)T,
            (int)K, (int)topk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

bool shared_suh(const at::Tensor& g_suh, const at::Tensor& u_suh, int64_t K) {
    const at::cuda::OptionalCUDAGuard guard(g_suh.device());
    const int64_t E = g_suh.numel();
    check_ptr_table(g_suh, E, "gate suh");
    check_ptr_table(u_suh, E, "up suh");
    at::Tensor flag = at::ones({1}, g_suh.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g_suh.get_device()).stream();
    suh_eq_kernel<<<(unsigned)E, 256, 0, stream>>>(g_suh.data_ptr<int64_t>(), u_suh.data_ptr<int64_t>(), (int)K,
                                                    flag.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return flag.item<int>() == 1;
}

std::vector<std::string> kernel_names(bool down) {
    std::vector<std::string> r;
    const KernInfo* k = down ? g_dn : g_gu;
    const int n = down ? N_DN : N_GU;
    for (int i = 0; i < n; ++i) r.push_back(k[i].name);
    return r;
}

// registers, local bytes, dynamic smem, CTAs a launch (on the current device)
std::vector<int64_t> kernel_info(bool down, int64_t idx) {
    const KernInfo* k = down ? g_dn : g_gu;
    TORCH_CHECK(idx >= 0 && idx < (down ? N_DN : N_GU), "kernel id");
    cudaFuncAttributes at{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&at, k[idx].fn));
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    return {at.numRegs, (int64_t)at.localSizeBytes, (int64_t)k[idx].smem, grid_for(k[idx], device), k[idx].bm,
            k[idx].threads, k[idx].cb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_prefill", &moe_prefill, "EXL3 routed-expert prefill MoE (plan, rotation, gate/up, down)");
    m.def("shared_suh", &shared_suh, "gate/up input sign vectors equal for every expert (syncs)");
    m.def("meta_ints", &meta_ints, "int32 elements of the plan buffer for E experts and P pairs");
    m.def("kernel_names", &kernel_names);
    m.def("kernel_info", &kernel_info);
}
