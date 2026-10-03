// glm53_exl3_dec bindings.
//
// Ported from TensorFold v0.6.0, src/tensorfold/cuda/exl3/experts.cpp
//   Copyright 2026 TensorFold contributors, Apache License 2.0 (LICENSES/Apache-2.0-TensorFold.txt).
// EXL3 format after ExLlamaV3, MIT License, Copyright (c) 2025 Turboderp (LICENSES/MIT-ExLlamaV3.txt).
//
// Changes from TensorFold (2026-09-30, glm53 kit): one entry (moe_decode) checks every argument, then launches the
// whole decode pipeline on the current stream (no host sync, CUDA-graph capturable); c10 stream API instead of
// ATen/cuda/CUDAContext.h (the image has no cusparse.h); dequant kept for tests.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

bool glm53_dec_config_ok(int64_t K, int64_t N, int64_t nt, int64_t warps, int64_t sk, int64_t pf, int64_t ld);
void glm53_dec_launch(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& wts, at::Tensor& out,
                      const at::Tensor& w13_trellis, const at::Tensor& w13_suh, const at::Tensor& w13_svh,
                      const at::Tensor& w2_trellis, const at::Tensor& w2_suh, const at::Tensor& w2_svh,
                      at::Tensor& xg, at::Tensor& xu, at::Tensor& xd, at::Tensor& z, at::Tensor& uids,
                      at::Tensor& ucount, at::Tensor& members, at::Tensor& pick32, double limit, int64_t act_mode,
                      const int64_t* cfg_gu, const int64_t* cfg_d, int64_t stop_after);
void glm53_dec_dequant(const at::Tensor& T, at::Tensor& out);

static void check_cuda(const at::Tensor& t, at::ScalarType dt, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt, "glm53_exl3_dec: ", name, " must be a CUDA ", dt,
                " tensor, got ", t.scalar_type(), t.is_cuda() ? " (cuda)" : " (cpu)");
}

static void check_contig(const at::Tensor& t, at::ScalarType dt, const char* name) {
    check_cuda(t, dt, name);
    TORCH_CHECK(t.is_contiguous(), "glm53_exl3_dec: ", name, " must be contiguous");
}

// Experts' tensors may be views; their inner dims must be dense and expert/matrix strides even (int16 -> words).
static void check_inner_dense(const at::Tensor& t, int inner_from, const char* name) {
    int64_t expect = 1;
    for (int d = (int)t.dim() - 1; d >= inner_from; --d) {
        TORCH_CHECK(t.stride(d) == expect || t.size(d) == 1, "glm53_exl3_dec: ", name, " inner dim ", d,
                    " is not dense");
        expect *= t.size(d);
    }
}

bool config_ok(int64_t K, int64_t N, int64_t nt, int64_t warps, int64_t sk, int64_t pf, int64_t ld) {
    return glm53_dec_config_ok(K, N, nt, warps, sk, pf, ld);
}

void moe_decode(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& wts, at::Tensor out,
                const at::Tensor& w13_trellis, const at::Tensor& w13_suh, const at::Tensor& w13_svh,
                const at::Tensor& w2_trellis, const at::Tensor& w2_suh, const at::Tensor& w2_svh, at::Tensor xg,
                at::Tensor xu, at::Tensor xd, at::Tensor z, at::Tensor uids, at::Tensor ucount, at::Tensor members,
                at::Tensor pick32, double limit, int64_t act_mode, std::vector<int64_t> cfg_gu,
                std::vector<int64_t> cfg_d, int64_t stop_after) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && x.dim() == 2 &&
                    x.stride(1) == 1,
                "glm53_exl3_dec: x must be a CUDA bf16/fp16 [T, D] tensor with unit inner stride");
    const int64_t R = x.size(0), D = x.size(1);
    TORCH_CHECK(R >= 1, "glm53_exl3_dec: no rows");
    TORCH_CHECK(ids.is_cuda() && (ids.scalar_type() == at::kLong || ids.scalar_type() == at::kInt) &&
                    ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == R,
                "glm53_exl3_dec: ids must be contiguous CUDA int64/int32 [T, k]");
    const int64_t slots = ids.size(1);
    TORCH_CHECK(slots >= 1 && slots <= 32, "glm53_exl3_dec: 1..32 experts a row");
    check_contig(wts, at::kFloat, "weights");
    TORCH_CHECK(wts.numel() == R * slots, "glm53_exl3_dec: weights must be [T, k]");
    check_contig(out, at::kFloat, "out");
    TORCH_CHECK(out.numel() == R * D, "glm53_exl3_dec: out must be [T, D]");

    check_cuda(w13_trellis, at::kShort, "w13_trellis");
    check_cuda(w2_trellis, at::kShort, "w2_trellis");
    check_cuda(w13_suh, at::kHalf, "w13_suh");
    check_cuda(w13_svh, at::kHalf, "w13_svh");
    check_cuda(w2_suh, at::kHalf, "w2_suh");
    check_cuda(w2_svh, at::kHalf, "w2_svh");
    TORCH_CHECK(w13_trellis.dim() == 5 && w2_trellis.dim() == 4 && w13_suh.dim() == 3 && w13_svh.dim() == 3 &&
                    w2_suh.dim() == 2 && w2_svh.dim() == 2,
                "glm53_exl3_dec: expected w13_trellis [E,2,D/16,I/16,64], w13_suh [E,2,D], w13_svh [E,2,I], "
                "w2_trellis [E,I/16,D/16,64], w2_suh [E,I], w2_svh [E,D]");
    const int64_t E = w13_trellis.size(0), I = w13_svh.size(2);
    TORCH_CHECK(w13_trellis.size(1) == 2 && w13_trellis.size(2) * 16 == D && w13_trellis.size(3) * 16 == I &&
                    w13_trellis.size(4) == 64,
                "glm53_exl3_dec: w13_trellis must be 4-bit [E, 2, D/16, I/16, 64]");
    TORCH_CHECK(w2_trellis.size(0) == E && w2_trellis.size(1) * 16 == I && w2_trellis.size(2) * 16 == D &&
                    w2_trellis.size(3) == 64,
                "glm53_exl3_dec: w2_trellis must be 4-bit [E, I/16, D/16, 64]");
    TORCH_CHECK(w13_suh.size(0) == E && w13_suh.size(1) == 2 && w13_suh.size(2) == D, "glm53_exl3_dec: w13_suh shape");
    TORCH_CHECK(w13_svh.size(0) == E && w13_svh.size(1) == 2, "glm53_exl3_dec: w13_svh shape");
    TORCH_CHECK(w2_suh.size(0) == E && w2_suh.size(1) == I, "glm53_exl3_dec: w2_suh shape");
    TORCH_CHECK(w2_svh.size(0) == E && w2_svh.size(1) == D, "glm53_exl3_dec: w2_svh shape");
    check_inner_dense(w13_trellis, 2, "w13_trellis");
    check_inner_dense(w2_trellis, 1, "w2_trellis");
    check_inner_dense(w13_suh, 2, "w13_suh");
    check_inner_dense(w13_svh, 2, "w13_svh");
    check_inner_dense(w2_suh, 1, "w2_suh");
    check_inner_dense(w2_svh, 1, "w2_svh");
    TORCH_CHECK(w13_trellis.stride(0) % 2 == 0 && w13_trellis.stride(1) % 2 == 0 && w2_trellis.stride(0) % 2 == 0,
                "glm53_exl3_dec: trellis expert strides must be whole 32-bit words");
    TORCH_CHECK(D % 128 == 0 && I % 128 == 0, "glm53_exl3_dec: D and I must be multiples of 128");
    TORCH_CHECK(R * slots + 2 * E <= 12288, "glm53_exl3_dec: too many pairs + experts for the grouping kernel");

    if (cfg_gu.size() == 4) cfg_gu.push_back(0);
    if (cfg_d.size() == 4) cfg_d.push_back(0);
    TORCH_CHECK(cfg_gu.size() == 5 && cfg_d.size() == 5, "glm53_exl3_dec: configs are (nt, warps, splits, pf[, ld])");
    TORCH_CHECK(glm53_dec_config_ok(D, I, cfg_gu[0], cfg_gu[1], cfg_gu[2], cfg_gu[3], cfg_gu[4]),
                "glm53_exl3_dec: gate/up config not compiled or does not divide the shape");
    TORCH_CHECK(glm53_dec_config_ok(I, D, cfg_d[0], cfg_d[1], cfg_d[2], cfg_d[3], cfg_d[4]),
                "glm53_exl3_dec: down config not compiled or does not divide the shape");
    TORCH_CHECK(act_mode == 1 || act_mode == 2, "glm53_exl3_dec: act_mode must be 1 (clamp, silu) or 2 (silu, clamp)");

    const int64_t P = R * slots, maxu = std::min(P, E);
    check_contig(xg, at::kHalf, "xg");
    check_contig(xu, at::kHalf, "xu");
    check_contig(xd, at::kHalf, "xd");
    check_contig(z, at::kFloat, "z");
    check_contig(uids, at::kInt, "uids");
    check_contig(ucount, at::kInt, "ucount");
    check_contig(members, at::kInt, "members");
    check_contig(pick32, at::kInt, "pick32");
    TORCH_CHECK(xg.numel() >= P * D && xu.numel() >= P * D && xd.numel() >= P * I, "glm53_exl3_dec: x scratch too small");
    TORCH_CHECK(z.numel() >= std::max(2 * cfg_gu[2] * P * I, cfg_d[2] * P * D), "glm53_exl3_dec: z scratch too small");
    TORCH_CHECK(uids.numel() >= maxu && ucount.numel() >= 1 && members.numel() >= maxu * R && pick32.numel() >= P,
                "glm53_exl3_dec: grouping scratch too small");
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&ids, &wts, &out, &w13_trellis, &w13_suh, &w13_svh, &w2_trellis, &w2_suh, &w2_svh,
                                &xg, &xu, &xd, &z, &uids, &ucount, &members, &pick32})
        TORCH_CHECK(t->device() == x.device(), "glm53_exl3_dec: all tensors must be on x's device");

    c10::cuda::CUDAGuard guard(x.device());
    glm53_dec_launch(x, ids, wts, out, w13_trellis, w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh, xg, xu, xd, z,
                     uids, ucount, members, pick32, limit, act_mode, cfg_gu.data(), cfg_d.data(), stop_after);
}

void dequant(const at::Tensor& T, at::Tensor out) {
    TORCH_CHECK(T.is_cuda() && T.scalar_type() == at::kShort && T.is_contiguous() && T.dim() == 3 && T.size(2) == 64,
                "T: contiguous int16 [K/16, N/16, 64] (4 bits)");
    check_contig(out, at::kHalf, "out");
    TORCH_CHECK(out.numel() == T.size(0) * 16 * T.size(1) * 16, "out must be [K, N]");
    c10::cuda::CUDAGuard guard(T.device());
    glm53_dec_dequant(T, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_decode", &moe_decode,
          "Routed EXL3 experts of a decode window: out[T, D] fp32 = sum_k w[t,k] * expert_ids[t,k](x[t]).");
    m.def("config_ok", &config_ok, "Whether a (K, N, nt, warps, splits, pf, ld) tile setting is compiled and divides.");
    m.def("dequant", &dequant, "W_q [K, N] fp16 of one 4-bit mcg trellis (tests).");
}
