// glm53_exl3_dec: EXL3 routed experts for vLLM decode windows, one stream, no host sync, no atomics on outputs.
//
// Ported from TensorFold v0.6.0, src/tensorfold/cuda/exl3/experts.cu
//   Copyright 2026 TensorFold contributors, Apache License 2.0 (LICENSES/Apache-2.0-TensorFold.txt).
// EXL3 format (trellis, suh/svh Hadamard rotations) after ExLlamaV3,
//   MIT License, Copyright (c) 2025 Turboderp (LICENSES/MIT-ExLlamaV3.txt).
//
// Changes from TensorFold (2026-09-30, glm53 kit):
//   * group: reads vLLM's int64 (or int32) top-k ids directly, writes the int32 pick table, histogram + in-order
//     ranks instead of a per-expert scan, members are flat pair indices; ids outside [0, E) are skipped;
//   * suh/svh/trellis are strided views of vLLM's stacked w13_* / w2_* parameters (no copies);
//   * gate/up epilogue: act mode 2 = fp32 SwiGLU in ExLlamaV3 exl3_moe's order (silu first, then
//     min(silu(g), limit) and clamp(u, -limit, limit), limit <= 0 disables), mode 1 = TensorFold ACT_F32
//     (clamp g first); TensorFold's bf16-rounding mode 0 is dropped;
//   * down epilogue + top-k combine fused without the per-slot y buffer; skipped slots contribute exactly 0;
//   * warps-per-block for the small kernels, one C++ entry that launches the whole pipeline;
//   * an optional 128-bit load path for the grouped GEMV (grouped_ld_kernel, after jayleaton's patches/0580).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "dec_grouped.cuh"

namespace glm53_dec {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// ---------------------------------------------------------------------------------------------------------------
// Grouping: distinct experts (< E) in id order -> uids[0..ucount), members[u][m] = pair indices in pair order,
// -1 after the last; pick32[p] = the pair's expert or -1.
constexpr int GROUP_THREADS = 1024;

template <typename ID>
__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const ID* __restrict__ ids, int n, int E, int maxm,
                                                              int* __restrict__ uids, int* __restrict__ ucount,
                                                              int* __restrict__ members, int* __restrict__ pick32) {
    extern __shared__ int sh[];
    int* sp = sh;            // [n]
    int* cnt = sh + n;       // [E]
    int* pos = cnt + E;      // [E]
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int tid = threadIdx.x;
    for (int i = tid; i < n; i += GROUP_THREADS) {
        const long long e = (long long)ids[i];
        const int v = (e >= 0 && e < E) ? (int)e : -1;
        sp[i] = v;
        pick32[i] = v;
    }
    for (int e = tid; e < E; e += GROUP_THREADS) cnt[e] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += GROUP_THREADS)
        if (sp[i] >= 0) atomicAdd(&cnt[sp[i]], 1);       // integer counts: order-independent
    __syncthreads();

    // exclusive scan of "expert used" over contiguous chunks of experts
    const int per = (E + GROUP_THREADS - 1) / GROUP_THREADS;
    const int e0 = tid * per;
    int used = 0;
    for (int q = 0; q < per; ++q) {
        const int e = e0 + q;
        if (e < E && cnt[e] > 0) ++used;
    }
    const int lane = tid & 31, warp = tid >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        const int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
    for (int q = 0; q < per; ++q) {
        const int e = e0 + q;
        if (e < E && cnt[e] > 0) {
            pos[e] = place;
            uids[place] = e;
            ++place;
        }
    }
    __syncthreads();
    for (int i = tid; i < n; i += GROUP_THREADS) {
        const int e = sp[i];
        if (e < 0) continue;
        int rank = 0;
        for (int j = 0; j < i; ++j) rank += (sp[j] == e);
        if (rank < maxm) members[pos[e] * maxm + rank] = i;
    }
    for (int e = tid; e < E; e += GROUP_THREADS) {
        const int c = cnt[e];
        if (c > 0)
            for (int j = c; j < maxm; ++j) members[pos[e] * maxm + j] = -1;
    }
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (TensorFold's fwht128).
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

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (pair p, 4 x 128-blocks of K, matrix): Xh[p] = fp16(((x[row] * suh_e) @ H128) / sqrt(128)) for gate / up.
template <typename TIN>
__global__ void __launch_bounds__(128) rot_in_kernel(const TIN* __restrict__ x, int x_stride,
                                                     const int* __restrict__ pick, const half* __restrict__ suh,
                                                     long long suh_estride, long long suh_matoff,
                                                     half* __restrict__ out0, half* __restrict__ out1, int K,
                                                     int slots) {
    const int p = blockIdx.x, mat = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5);
    const int e = pick[p];
    if (e < 0 || blk * 128 >= K) return;
    const int row = p / slots;
    const half* s = suh + (size_t)e * suh_estride + (mat ? suh_matoff : 0) + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(s[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (pair p, 4 x 128-blocks of the width): splits summed in order, rotated, * svh, SwiGLU, * suh_d, rotated:
// Xd[p] = fp16(((act * suh_d) @ H128) / sqrt(128)).
__global__ void __launch_bounds__(128) gateup_epilogue_kernel(
    const float* __restrict__ Z, const int* __restrict__ pick, const half* __restrict__ svh, long long svh_estride,
    long long svh_matoff, const half* __restrict__ suh_d, long long suhd_estride, half* __restrict__ xd, int P, int N,
    int SK, float limit, int act_mode) {
    const int p = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.y * 4 + (threadIdx.x >> 5);
    const int e = pick[p];
    if (e < 0 || blk * 128 >= N) return;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    const half* sg_ = svh + (size_t)e * svh_estride + n;
    const half* su_ = svh + (size_t)e * svh_estride + svh_matoff + n;
    const half* sd_ = suh_d + (size_t)e * suhd_estride + n;
    const bool lim = limit > 0.f;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = gv[j] * HAD_SCALE * __half2float(sg_[j]);
        float uu = uv[j] * HAD_SCALE * __half2float(su_[j]);
        if (lim) uu = fminf(fmaxf(uu, -limit), limit);
        float act;
        if (act_mode == 1) {                 // TensorFold ACT_F32: limit the gate before SiLU
            if (lim) gg = fminf(gg, limit);
            act = gg / (1.f + expf(-gg));
        } else {                             // ExLlamaV3 exl3_moe order: SiLU, then limit
            act = gg / (1.f + expf(-gg));
            if (lim) act = fminf(act, limit);
        }
        v[j] = act * uu * __half2float(sd_[j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (row r, 128-block of the model width), one warp a slot: y = (splits summed in order) @ H128 / sqrt(128)
// * svh_d per slot, then out[r] = fma chain over slots in order of wts[r][k] * y_k (fp32, from 0).
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, long long svhd_estride,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    if (e >= 0) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
        const half* sv = svh_d + (size_t)e * svhd_estride + n;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(sv[j]);
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float4 u = part[q][lane];
        if (pick[r * slots + q] < 0) continue;        // a skipped slot adds nothing, whatever its weight
        const float w = wts[r * slots + q];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
    *reinterpret_cast<float4*>(out + (size_t)r * D + n) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}

// ---------------------------------------------------------------------------------------------------------------
struct GroupedArgs {
    const half* x0;
    const half* x1;
    const uint32_t* t;
    long long estride, matoff;   // words
    const int* uids;
    const int* ucount;
    const int* members;
    float* z;
    int K, N, P, SK, maxm, maxu, mats, nt, warps, pf, ld;
};

template <int NT, int W, int PF>
static void launch_grouped_t(const GroupedArgs& a, cudaStream_t stream) {
    const int MT = (a.maxm + 15) / 16;
    dim3 grid((unsigned)a.maxu, (unsigned)(a.N / (16 * NT)), (unsigned)(a.mats * a.SK * MT));
    grouped_kernel<NT, W, PF><<<grid, W * 32, 0, stream>>>(a.x0, a.x1, a.t, a.estride, a.matoff, a.uids, a.ucount,
                                                          a.members, a.z, a.K, a.N, a.P, a.SK, a.maxm);
}

template <int NT, int W, int PD>
static void launch_grouped_ld_t(const GroupedArgs& a, cudaStream_t stream) {
    const int MT = (a.maxm + 15) / 16;
    dim3 grid((unsigned)a.maxu, (unsigned)(a.N / (16 * NT)), (unsigned)(a.mats * a.SK * MT));
    grouped_ld_kernel<NT, W, PD><<<grid, W * 32, 0, stream>>>(a.x0, a.x1, a.t, a.estride, a.matoff, a.uids,
                                                             a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK, a.maxm);
}

// The compiled settings of the 128-bit load path (n tiles, warps, ring depth).
#define GLM53_DEC_LD_CONFIGS(X) \
    X(8, 4, 1) X(8, 4, 2) X(8, 4, 4) X(8, 8, 1) X(8, 8, 2) X(4, 4, 2) X(4, 4, 4) X(4, 8, 2) X(4, 8, 4) X(8, 2, 2)

// The compiled tile settings (n tiles a block, warps, trellis rows in flight).
#define GLM53_DEC_CONFIGS(X) \
    X(8, 4, 1) X(8, 4, 2) X(8, 4, 3) X(8, 4, 4) \
    X(4, 4, 1) X(4, 4, 2) X(4, 4, 4) \
    X(8, 2, 1) X(8, 2, 2) X(8, 2, 4) \
    X(4, 2, 2) X(4, 2, 4) \
    X(16, 2, 1) X(16, 2, 2) X(16, 4, 1) X(16, 4, 2) \
    X(8, 8, 1) X(8, 8, 2) X(4, 8, 2) X(4, 8, 4) \
    X(2, 4, 4) X(2, 8, 4) X(2, 2, 8) X(4, 1, 4) X(8, 1, 2)

static bool launch_grouped(const GroupedArgs& a, cudaStream_t stream) {
    if (a.ld == 1) {
#define GLM53_DEC_LD_CASE(NT_, W_, PD_) \
    if (a.nt == NT_ && a.warps == W_ && a.pf == PD_) { launch_grouped_ld_t<NT_, W_, PD_>(a, stream); return true; }
        GLM53_DEC_LD_CONFIGS(GLM53_DEC_LD_CASE)
#undef GLM53_DEC_LD_CASE
        return false;
    }
#define GLM53_DEC_CASE(NT_, W_, PF_) \
    if (a.nt == NT_ && a.warps == W_ && a.pf == PF_) { launch_grouped_t<NT_, W_, PF_>(a, stream); return true; }
    GLM53_DEC_CONFIGS(GLM53_DEC_CASE)
#undef GLM53_DEC_CASE
    return false;
}

static bool config_compiled(int nt, int warps, int pf, int ld) {
    if (ld == 1) {
#define GLM53_DEC_LD_HAS(NT_, W_, PD_) if (nt == NT_ && warps == W_ && pf == PD_) return true;
        GLM53_DEC_LD_CONFIGS(GLM53_DEC_LD_HAS)
#undef GLM53_DEC_LD_HAS
        return false;
    }
    if (ld != 0) return false;
#define GLM53_DEC_HAS(NT_, W_, PF_) if (nt == NT_ && warps == W_ && pf == PF_) return true;
    GLM53_DEC_CONFIGS(GLM53_DEC_HAS)
#undef GLM53_DEC_HAS
    return false;
}

}  // namespace glm53_dec

// ---------------------------------------------------------------------------------------------------------------
// Host side (called from dec.cpp after argument checks).

bool glm53_dec_config_ok(int64_t K, int64_t N, int64_t nt, int64_t warps, int64_t sk, int64_t pf, int64_t ld) {
    return glm53_dec::config_compiled((int)nt, (int)warps, (int)pf, (int)ld) && sk >= 1 &&
           K % (16 * sk * warps) == 0 && N % (16 * nt) == 0;
}

void glm53_dec_launch(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& wts, at::Tensor& out,
                      const at::Tensor& w13_trellis, const at::Tensor& w13_suh, const at::Tensor& w13_svh,
                      const at::Tensor& w2_trellis, const at::Tensor& w2_suh, const at::Tensor& w2_svh,
                      at::Tensor& xg, at::Tensor& xu, at::Tensor& xd, at::Tensor& z, at::Tensor& uids,
                      at::Tensor& ucount, at::Tensor& members, at::Tensor& pick32, double limit, int64_t act_mode,
                      const int64_t* cfg_gu, const int64_t* cfg_d, int64_t stop_after) {
    using namespace glm53_dec;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const int R = (int)x.size(0), D = (int)x.size(1), slots = (int)ids.size(1);
    const int E = (int)w13_trellis.size(0), I = (int)w13_svh.size(2);
    const int P = R * slots, maxm = R, maxu = std::min(P, E);

    // 1. group
    {
        const size_t smem = (size_t)(P + 2 * E) * sizeof(int);
        if (ids.scalar_type() == at::kLong)
            group_kernel<int64_t><<<1, GROUP_THREADS, smem, stream>>>(
                ids.data_ptr<int64_t>(), P, E, maxm, uids.data_ptr<int>(), ucount.data_ptr<int>(),
                members.data_ptr<int>(), pick32.data_ptr<int>());
        else
            group_kernel<int><<<1, GROUP_THREADS, smem, stream>>>(
                ids.data_ptr<int>(), P, E, maxm, uids.data_ptr<int>(), ucount.data_ptr<int>(),
                members.data_ptr<int>(), pick32.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stop_after == 1) return;
    // 2. rotate the inputs of gate and up, per routed pair
    {
        dim3 grid((unsigned)P, (unsigned)((D / 128 + 3) / 4), 2);
        auto s = reinterpret_cast<const half*>(w13_suh.data_ptr());
        auto o0 = reinterpret_cast<half*>(xg.data_ptr());
        auto o1 = reinterpret_cast<half*>(xu.data_ptr());
        if (x.scalar_type() == at::kBFloat16)
            rot_in_kernel<__nv_bfloat16><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.stride(0), pick32.data_ptr<int>(), s,
                w13_suh.stride(0), w13_suh.stride(1), o0, o1, D, slots);
        else
            rot_in_kernel<half><<<grid, 128, 0, stream>>>(reinterpret_cast<const half*>(x.data_ptr()),
                                                          (int)x.stride(0), pick32.data_ptr<int>(), s,
                                                          w13_suh.stride(0), w13_suh.stride(1), o0, o1, D, slots);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stop_after == 2) return;
    // 3. gate and up, grouped by distinct expert
    {
        GroupedArgs a;
        a.x0 = reinterpret_cast<const half*>(xg.data_ptr());
        a.x1 = reinterpret_cast<const half*>(xu.data_ptr());
        a.t = reinterpret_cast<const uint32_t*>(w13_trellis.data_ptr());
        a.estride = w13_trellis.stride(0) / 2;
        a.matoff = w13_trellis.stride(1) / 2;
        a.uids = uids.data_ptr<int>();
        a.ucount = ucount.data_ptr<int>();
        a.members = members.data_ptr<int>();
        a.z = z.data_ptr<float>();
        a.K = D; a.N = I; a.P = P; a.SK = (int)cfg_gu[2]; a.maxm = maxm; a.maxu = maxu; a.mats = 2;
        a.nt = (int)cfg_gu[0]; a.warps = (int)cfg_gu[1]; a.pf = (int)cfg_gu[3]; a.ld = (int)cfg_gu[4];
        TORCH_CHECK(launch_grouped(a, stream), "glm53_exl3_dec: gate/up tile setting not compiled");
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stop_after == 3) return;
    // 4. gate/up epilogue: SwiGLU and the down projection's input rotation
    {
        dim3 grid((unsigned)P, (unsigned)((I / 128 + 3) / 4));
        gateup_epilogue_kernel<<<grid, 128, 0, stream>>>(
            z.data_ptr<float>(), pick32.data_ptr<int>(), reinterpret_cast<const half*>(w13_svh.data_ptr()),
            w13_svh.stride(0), w13_svh.stride(1), reinterpret_cast<const half*>(w2_suh.data_ptr()),
            w2_suh.stride(0), reinterpret_cast<half*>(xd.data_ptr()), P, I, (int)cfg_gu[2], (float)limit,
            (int)act_mode);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stop_after == 4) return;
    // 5. down, grouped by distinct expert
    {
        GroupedArgs a;
        a.x0 = reinterpret_cast<const half*>(xd.data_ptr());
        a.x1 = a.x0;
        a.t = reinterpret_cast<const uint32_t*>(w2_trellis.data_ptr());
        a.estride = w2_trellis.stride(0) / 2;
        a.matoff = 0;
        a.uids = uids.data_ptr<int>();
        a.ucount = ucount.data_ptr<int>();
        a.members = members.data_ptr<int>();
        a.z = z.data_ptr<float>();
        a.K = I; a.N = D; a.P = P; a.SK = (int)cfg_d[2]; a.maxm = maxm; a.maxu = maxu; a.mats = 1;
        a.nt = (int)cfg_d[0]; a.warps = (int)cfg_d[1]; a.pf = (int)cfg_d[3]; a.ld = (int)cfg_d[4];
        TORCH_CHECK(launch_grouped(a, stream), "glm53_exl3_dec: down tile setting not compiled");
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (stop_after == 5) return;
    // 6. down epilogue + weighted top-k combine
    {
        dim3 grid((unsigned)R, (unsigned)(D / 128));
        down_combine_kernel<<<grid, (unsigned)(32 * slots), 0, stream>>>(
            z.data_ptr<float>(), pick32.data_ptr<int>(), reinterpret_cast<const half*>(w2_svh.data_ptr()),
            w2_svh.stride(0), wts.data_ptr<float>(), out.data_ptr<float>(), P, D, (int)cfg_d[2], slots);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void glm53_dec_dequant(const at::Tensor& T, at::Tensor& out) {
    const int K = (int)T.size(0) * 16, N = (int)T.size(1) * 16;
    dim3 grid((unsigned)(K / 16), (unsigned)(N / 16));
    glm53_dec::dequant_kernel<<<grid, 32, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const uint32_t*>(T.data_ptr()), reinterpret_cast<half*>(out.data_ptr()), K, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
