// glm53_exl3_dec: grouped EXL3 routed-expert GEMV for vLLM decode windows (T <= 64 rows).
//
// Ported from TensorFold v0.6.0, src/tensorfold/cuda/exl3/experts_grouped.cuh
//   Copyright 2026 TensorFold contributors, Apache License 2.0 (LICENSES/Apache-2.0-TensorFold.txt).
// EXL3 format, trellis layout and the "mcg" codebook are after ExLlamaV3,
//   MIT License, Copyright (c) 2025 Turboderp (LICENSES/MIT-ExLlamaV3.txt).
// grouped_ld_kernel's load path is after glm53-tensorfold-spark patches/0580 (exl3_ld.cu),
//   Copyright 2026 Jay Leaton, Apache License 2.0 (LICENSES/NOTICE-glm53-tensorfold-spark.txt).
//
// Changes from TensorFold (2026-09-30, glm53 kit):
//   * only the 4-bit (K2 = 8) "mcg" codebook path is kept; the K2 switch and the 3inst/mul1 codebooks are gone;
//   * experts are addressed by stride from vLLM's stacked w13/w2 tensors (base + e * estride words) instead of
//     per-expert pointer tables, so nothing is copied or allocated per layer;
//   * warps are reduced through ONE shared [16][NT*16] buffer, added in warp order (the same order and so the
//     same bits as TensorFold's per-warp buffers), which cuts static shared memory 4-8x and raises occupancy;
//   * the A fragments (rotated activations) of the next k tile are prefetched with the trellis words;
//   * members are flat pair indices p = row * slots + slot (TensorFold encoded row * 32 + slot);
//   * grouped_ld_kernel: the same work item and bits with a 128-bit, PD-deep load ring (see below).
// Rows stay independent: every mma row is one (row, slot) pair, the K range of each warp and split is fixed by
// the shape, and warps / splits are summed in a fixed order, never with atomics.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace glm53_dec {

// Two mcg codebook values from two 16-bit states, as a half2 (first state in .x); ExLlamaV3's decode, bit for bit.
__device__ __forceinline__ uint32_t mcg_pair(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// 4 bits: a 16x16 tile is 32 words; lane L's eight values are windows over words L-1 and L, and they are exactly
// the B fragments of the tile's two n8 halves (columns 0-7 in b0, 8-15 in b1).
__device__ __forceinline__ void decode_tile4(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg_pair((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg_pair(s & 0xffffu, w >> 16);
    b1[0] = mcg_pair((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg_pair((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? __ldg(reinterpret_cast<const unsigned int*>(x)) : 0u;
}

__device__ __forceinline__ void load_a(uint32_t (&a)[4], const half* x0, const half* x1, bool ok0, bool ok1, int k) {
    a[0] = load_pair(x0 + k, ok0);
    a[1] = load_pair(x1 + k, ok1);
    a[2] = load_pair(x0 + k + 8, ok0);
    a[3] = load_pair(x1 + k + 8, ok1);
}

// One warp's k tiles [kt0, kt0 + nkt) of one expert matrix into acc; PF trellis tiles (rows of NT tiles) in flight.
template <int NT, int PF>
__device__ __forceinline__ void warp_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                           const half* x0, const half* x1, bool ok0, bool ok1, int lane,
                                           float (&acc)[NT][2][4]) {
    const size_t kstride = (size_t)NTILES * 32;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * 32 + lane;

    uint32_t pf[PF][NT];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) pf[d][i] = __ldg(tp + d * kstride + i * 32);
    uint32_t an[4];
    load_a(an, x0, x1, ok0, ok1, kt0 * 16);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT];
#pragma unroll
                for (int i = 0; i < NT; ++i) w[i] = pf[d][i];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i) pf[d][i] = __ldg(tp + (size_t)(it + PF) * kstride + i * 32);
                uint32_t a[4] = {an[0], an[1], an[2], an[3]};
                if (it + 1 < nkt) load_a(an, x0, x1, ok0, ok1, (kt0 + it + 1) * 16);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile4(w[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
}

// Program (distinct expert u, n block, member tile m + MT * (split + SK * mat)): up to 16 member pairs times the
// expert's W_q over the split's K range, warps added in order into Z[mat][split][pair][n] (fp32, no atomics).
template <int NT, int W, int PF>
__global__ void __launch_bounds__(W * 32) grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ Tbase,
    long long t_estride, long long t_matoff, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = Tbase + (size_t)e * t_estride + (mat ? t_matoff : 0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        rows_sh[threadIdx.x] = m < maxm ? members[u * maxm + m] : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    warp_tiles<NT, PF>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc);

    // warps' partial sums added in warp order through one buffer: ((w0 + w1) + w2) + ...
    constexpr int S = NT * 16 + 4;
    __shared__ float red[16 * S];
#pragma unroll 1
    for (int w = 0; w < W; ++w) {
        if (warp == w) {
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    float2* lo = reinterpret_cast<float2*>(&red[g * S + col]);
                    float2* hi = reinterpret_cast<float2*>(&red[(g + 8) * S + col]);
                    if (w == 0) {
                        *lo = make_float2(acc[i][h][0], acc[i][h][1]);
                        *hi = make_float2(acc[i][h][2], acc[i][h][3]);
                    } else {
                        float2 a = *lo, b = *hi;
                        a.x += acc[i][h][0]; a.y += acc[i][h][1];
                        b.x += acc[i][h][2]; b.y += acc[i][h][3];
                        *lo = a;
                        *hi = b;
                    }
                }
        }
        __syncthreads();
    }
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = red[row * S + col];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The same work item with the load path of jayleaton's GB10 kit (glm53-tensorfold-spark patches/0580, exl3_ld.cu,
// Copyright 2026 Jay Leaton, Apache-2.0; after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): the prologue in one round trip with the first PD
// k steps of trellis words issued before the member rows reach shared memory, then a PD-deep ring of 128-bit
// ld.global.nc.L1::no_allocate loads a warp (lane l loads words 4 (l % 8) .. + 3 of tile 4 v + l / 8 of each 512 B),
// staged through the warp's own 1 KB of shared memory into decode_tile4's layout, and the A fragments PD steps
// ahead. Same per-warp K ranges, mma chain and in-order warp sum as grouped_kernel, so the same bits.
__device__ __forceinline__ uint4 ldg_v4_na(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldg_pair_pinned(const half* x, bool ok) {
    uint32_t v = 0u;
    if (ok) asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(x));
    return v;
}

template <int NT, int W, int PD>
__global__ void __launch_bounds__(W * 32) grouped_ld_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ Tbase,
    long long t_estride, long long t_matoff, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm) {
    static_assert(NT % 4 == 0 && PD >= 1 && PD <= 4, "NT a multiple of 4, PD 1..4");
    constexpr int NV = NT / 4;
    const int u = blockIdx.x;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;

    // one round trip: ucount, uids[u], the tile's member codes (in bounds for every u < grid.x)
    const int cnt = __ldcg(ucount);
    const int e = __ldcg(uids + u);
    const int m = mtile * 16 + (int)threadIdx.x;
    const int code = (threadIdx.x < 16 && m < maxm) ? __ldcg(members + u * maxm + m) : -1;
    const int first = __ldcg(members + u * maxm + mtile * 16);
    if (u >= cnt || first < 0) return;                    // the same CTAs exit as in grouped_kernel

    __shared__ int rows_sh[16];
    __shared__ __align__(16) uint32_t stage_all[W][NT * 32];
    constexpr int S = NT * 16 + 4;
    __shared__ __align__(16) float red[16 * S];

    const int KT = K >> 4, NTILES = N >> 4;
    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const size_t kstep = (size_t)NTILES * 32;
    const uint32_t* tile = Tbase + (size_t)e * t_estride + (mat ? t_matoff : 0) +
                           ((size_t)kt0 * NTILES + (size_t)blockIdx.y * NT) * 32;
    uint32_t* stage = stage_all[warp];

    uint4 rv[PD][NV];
#pragma unroll
    for (int d = 0; d < PD; ++d)
        if (d < per_warp)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_v4_na(tile + d * kstep + 128 * v + 4 * lane);

    if (threadIdx.x < 16) rows_sh[threadIdx.x] = code;
    __syncthreads();
    const half* X = mat ? X1 : X0;
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const bool ok0 = r0 >= 0, ok1 = r1 >= 0;
    const half* x0 = X + (size_t)(ok0 ? r0 : 0) * K + 2 * t;
    const half* x1 = X + (size_t)(ok1 ? r1 : 0) * K + 2 * t;
    uint32_t ra[PD][4];
#pragma unroll
    for (int d = 0; d < PD; ++d)
        if (d < per_warp) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = ldg_pair_pinned(x0 + k, ok0);
            ra[d][1] = ldg_pair_pinned(x1 + k, ok1);
            ra[d][2] = ldg_pair_pinned(x0 + k + 8, ok0);
            ra[d][3] = ldg_pair_pinned(x1 + k + 8, ok1);
        }

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    for (int kb = 0; kb < per_warp; kb += PD) {
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int s = kb + d;
            if (s < per_warp) {
                const bool more = s + PD < per_warp;
                uint32_t a[4] = {ra[d][0], ra[d][1], ra[d][2], ra[d][3]};
                __syncwarp();                             // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more)
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_v4_na(tile + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);
                __syncwarp();
                uint32_t w[NT];
#pragma unroll
                for (int i = 0; i < NT; ++i) w[i] = stage[i * 32 + lane];
                if (more) {
                    const int k = (kt0 + s + PD) * 16;
                    ra[d][0] = ldg_pair_pinned(x0 + k, ok0);
                    ra[d][1] = ldg_pair_pinned(x1 + k, ok1);
                    ra[d][2] = ldg_pair_pinned(x0 + k + 8, ok0);
                    ra[d][3] = ldg_pair_pinned(x1 + k + 8, ok1);
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile4(w[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }

    // warps' partial sums added in warp order through one buffer (grouped_kernel's code)
#pragma unroll 1
    for (int w = 0; w < W; ++w) {
        if (warp == w) {
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    float2* lo = reinterpret_cast<float2*>(&red[g * S + col]);
                    float2* hi = reinterpret_cast<float2*>(&red[(g + 8) * S + col]);
                    if (w == 0) {
                        *lo = make_float2(acc[i][h][0], acc[i][h][1]);
                        *hi = make_float2(acc[i][h][2], acc[i][h][3]);
                    } else {
                        float2 a = *lo, b = *hi;
                        a.x += acc[i][h][0]; a.y += acc[i][h][1];
                        b.x += acc[i][h][2]; b.y += acc[i][h][3];
                        *lo = a;
                        *hi = b;
                    }
                }
        }
        __syncthreads();
    }
    const int nt0 = blockIdx.y * NT;
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = red[row * S + col];
    }
}

// W_q [K, N] fp16 of one 4-bit matrix through the same lane decode (tests only).
__global__ void dequant_kernel(const uint32_t* __restrict__ T, half* __restrict__ out, int K, int N) {
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x;
    const int NTILES = N >> 4;
    const uint32_t w = T[((size_t)kt * NTILES + nt) * 32 + lane];
    uint32_t b0[2], b1[2];
    decode_tile4(w, lane, b0, b1);
    const int g = lane >> 2, t = lane & 3;
    uint32_t v[4] = {b0[0], b0[1], b1[0], b1[1]};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        half2 h = *reinterpret_cast<half2*>(&v[q]);
        const int col = nt * 16 + g + 8 * (q >> 1);
        const int row = kt * 16 + 2 * t + 8 * (q & 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}

}  // namespace glm53_dec
