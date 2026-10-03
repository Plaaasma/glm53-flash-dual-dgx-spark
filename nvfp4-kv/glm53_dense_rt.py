"""8-bit weights (W8A16) for GLM-5.3-Flash's non-expert projections and LM head.

The EXL3 checkpoint quantizes only the routed experts; every other matrix (KDA / DSA projections, shared experts,
dense MLP, LM head, MTP eh_proj) stays BF16: ~8.9 GB a rank read on every decode step, about half of a step's bytes.
This runtime converts those matrices to 8 bits after load and swaps in matmuls that keep activations in BF16. Two
formats, picked per matrix:

  fp8   FP8 e4m3, one fp32 scale per 128x128 block. For matrices that come from GLM's official FP8 release (<= 256
        distinct values in a 128x128 block: shared experts, dense MLP, DSA/MLA projections). Re-quantizing them on
        their own block grid is near-exact (0.15% relative error = the BF16 rounding of the release).
  int8  symmetric INT8, one fp32 scale per (row, 128 columns). For native-BF16 matrices (KDA q/k/v/o, LM head, MTP
        eh_proj): 0.66% relative error where FP8's 3-bit mantissa would cost 2.6% (measured on the checkpoint).

Kernels (one family; the format only changes the scale's row stride SROW = 128 or 1):
  M <= 64        GEMV-style Triton kernel streaming the 8-bit weight once (decode / MTP verify; CUDA-graph safe)
  64 < M <= 256  tiled Triton GEMM converting the weight to BF16 in the K loop
  M > 256        dequantize into one shared BF16 scratch, then cuBLAS (prefill chunks)

Env (read at conversion):
  GLM53_DENSE_W8=auto|origin|0   auto: every eligible matrix in its format; origin: only the fp8-origin matrices;
                                  0 (default): off, every matrix stays BF16
  GLM53_DENSE_MIN_NUMEL          skip matrices smaller than this (default 1048576)
  GLM53_DENSE_EXCLUDE            regex of module names to keep in BF16 (added to the built-in exclusions)
  GLM53_DENSE_SCRATCH_MIN_M      M above which the dequant + cuBLAS path is used (default 256)
"""
from __future__ import annotations

import os
import re
import time

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger("vllm.glm53_dense")

# Kept in BF16 whatever the mode: weights other code reads directly (MLA absorb of kv_b_proj, the KDA conv weights,
# the indexer's fused wk/weights projection), the MoE router, the vision tower, and the MTP draft head (replaced by
# the target's head after the draft loads).
_BUILTIN_EXCLUDE = (r"(kv_b_proj|conv1d|wk_weights_proj|\.gate$|visual|vision|merger|mm_projector"
                    r"|shared_head\.head)")


@triton.jit
def _w8_gemv_kernel(X, W, S, Y, P, M, N, K, sxm, swn, ssn, sym, KPER,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SROW: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    nmask = offs_n < N
    mmask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_lo = pid_k * KPER
    for kk in range(0, KPER, BK):
        k0 = k_lo + kk
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(X + offs_m[:, None] * sxm + offs_k[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(W + offs_n[:, None] * swn + offs_k[None, :], mask=nmask[:, None], other=0.0)
        s = tl.load(S + (offs_n // SROW) * ssn + k0 // 128, mask=nmask, other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    if SPLIT == 1:
        tl.store(Y + offs_m[:, None] * sym + offs_n[None, :], acc.to(Y.dtype.element_ty),
                 mask=mmask[:, None] & nmask[None, :])
    else:  # fp32 partials, summed in split order by _split_reduce_kernel (deterministic)
        tl.store(P + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _split_reduce_kernel(P, Y, M, N, sym, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(SPLIT):
        acc += tl.load(P + i * M * N + offs, mask=mask, other=0.0)
    tl.store(Y + (offs // N) * sym + offs % N, acc.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _w8_mm_kernel(X, W, S, Y, M, N, K, sxm, swn, ssn, sym,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr, SROW: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    width = GM * nn
    gid = pid // width
    first_m = gid * GM
    gsz = tl.minimum(nm - first_m, GM)
    pid_m = first_m + (pid % width) % gsz
    pid_n = (pid % width) // gsz
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mmask = offs_m < M
    nmask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(X + offs_m[:, None] * sxm + offs_k[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(W + offs_n[:, None] * swn + offs_k[None, :], mask=nmask[:, None], other=0.0)
        s = tl.load(S + (offs_n // SROW) * ssn + k0 // 128, mask=nmask, other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    tl.store(Y + offs_m[:, None] * sym + offs_n[None, :], acc.to(Y.dtype.element_ty),
             mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _dequant_kernel(W, S, O, N, K, swn, ssn, BN: tl.constexpr, BK: tl.constexpr, SROW: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_k = pk * BK + tl.arange(0, BK)
    m = offs_n < N
    w = tl.load(W + offs_n[:, None] * swn + offs_k[None, :], mask=m[:, None], other=0.0)
    s = tl.load(S + (offs_n // SROW) * ssn + (pk * BK) // 128, mask=m, other=0.0)
    tl.store(O + offs_n[:, None] * K + offs_k[None, :], (w.to(tl.float32) * s[:, None]).to(tl.bfloat16),
             mask=m[:, None])


def _gemv_cfg(M: int, N: int, K: int) -> tuple[int, int, int, int, int]:
    """(BM, BN, split, num_warps, num_stages) for M <= 64 (GB10 sweeps bench_w8a16.py / bench_split.py, 2026-10-01).

    Wide matrices (N >= 8192: KDA in-proj, LM head) have enough 64-column programs; narrow ones (N <= 4096) split K
    until there are >= 256 programs (~5 a SM) so enough loads are in flight: 110-175 -> 190-217 GB/s."""
    bm = 16 if M <= 16 else (32 if M <= 32 else 64)
    if N >= 8192:
        return bm, 64, 1, (8 if bm == 64 else 4), 4
    bn, split = 32, 1
    while triton.cdiv(N, bn) * split < 256 and K % (128 * split * 2) == 0 and K // (split * 2) >= 512 and split < 8:
        split *= 2
    return bm, bn, split, (8 if bm == 64 else 4), 4


class _State:
    scratch: dict[torch.device, torch.Tensor] = {}
    scratch_numel = 0
    min_m_scratch = int(os.environ.get("GLM53_DENSE_SCRATCH_MIN_M", "256"))


def _scratch(device: torch.device, numel: int) -> torch.Tensor:
    buf = _State.scratch.get(device)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(max(numel, _State.scratch_numel), dtype=torch.bfloat16, device=device)
        _State.scratch[device] = buf
    return buf


def w8_matmul(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor, srow: int, use_scratch: bool) -> torch.Tensor:
    """y [M, N] (x.dtype) = x [M, K] @ dequant(w8 [N, K], scale [ceil(N / srow), K / 128]).T"""
    M, K = x.shape
    N = w8.shape[0]
    if x.stride(1) != 1:
        x = x.contiguous()
    if M > _State.min_m_scratch and use_scratch:
        buf = _scratch(x.device, N * K)[: N * K].view(N, K)
        _dequant_kernel[(triton.cdiv(N, 64), K // 128)](w8, scale, buf, N, K, w8.stride(0), scale.stride(0),
                                                        BN=64, BK=128, SROW=srow, num_warps=4)
        return torch.nn.functional.linear(x, buf)
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M <= 64:
        bm, bn, split, nw, ns = _gemv_cfg(M, N, K)
        part = torch.empty((split, M, N), dtype=torch.float32, device=x.device) if split > 1 else y
        _w8_gemv_kernel[(triton.cdiv(N, bn), split)](x, w8, scale, y, part, M, N, K, x.stride(0), w8.stride(0),
                                                     scale.stride(0), y.stride(0), K // split, BM=bm, BN=bn, BK=128,
                                                     SROW=srow, SPLIT=split, num_warps=nw, num_stages=ns)
        if split > 1:
            _split_reduce_kernel[(triton.cdiv(M * N, 1024),)](part, y, M, N, y.stride(0), SPLIT=split, BLOCK=1024,
                                                              num_warps=4)
    else:
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
        _w8_mm_kernel[grid](x, w8, scale, y, M, N, K, x.stride(0), w8.stride(0), scale.stride(0), y.stride(0),
                            BM=128, BN=128, BK=128, GM=8, SROW=srow, num_warps=8, num_stages=3)
    return y


class W8DenseMethod:
    """Drop-in for UnquantizedLinearMethod / UnquantizedEmbeddingMethod `apply` once a matrix is converted."""

    def __init__(self, orig, use_scratch: bool):
        self.orig = orig
        self.use_scratch = use_scratch

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        y = w8_matmul(x.reshape(-1, shape[-1]), layer.weight, layer.glm53_w8_scale, layer.glm53_w8_srow,
                      self.use_scratch)
        if bias is not None:
            y = y + bias
        return y.reshape(*shape[:-1], y.shape[-1])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # already converted
        return None

    def __getattr__(self, name):  # anything else (embedding(), create_weights, ...) goes to the original method
        return getattr(self.orig, name)


@torch.no_grad()
def fp8_origin(w: torch.Tensor) -> bool:
    """True when sampled 128x128 blocks hold <= 256 distinct values (the matrix was FP8 in GLM's release)."""
    N, K = w.shape
    if N < 128 or K < 128:
        return False
    picks = [(0, 0), (N // 256, K // 256), (N // 128 - 1, K // 128 - 1), (N // 384, K // 128 - 1)]
    for bn, bk in picks:
        blk = w[bn * 128:(bn + 1) * 128, bk * 128:(bk + 1) * 128]
        if torch.unique(blk).numel() > 256:
            return False
    return True


@torch.no_grad()
def quantize_fp8_block(w: torch.Tensor, rows_per_chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (fp8 e4m3 [N, K], fp32 scales [ceil(N/128), K/128]); chunked to keep fp32 temporaries small."""
    N, K = w.shape
    q = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty(((N + 127) // 128, K // 128), dtype=torch.float32, device=w.device)
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        blk = w[r0:r1].float()
        nb = (r1 - r0 + 127) // 128
        pad = nb * 128 - (r1 - r0)
        if pad:
            blk = torch.cat([blk, blk.new_zeros(pad, K)], 0)
        b4 = blk.view(nb, 128, K // 128, 128)
        sc = (b4.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12)
        qq = (b4 / sc[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(nb * 128, K)
        q[r0:r1] = qq[: r1 - r0]
        s[r0 // 128: r0 // 128 + nb] = sc
        del blk, b4, qq
    return q, s


@torch.no_grad()
def quantize_int8_rowgroup(w: torch.Tensor, rows_per_chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (int8 [N, K], fp32 scales [N, K/128]): symmetric, one scale per row and 128 columns."""
    N, K = w.shape
    q = torch.empty((N, K), dtype=torch.int8, device=w.device)
    s = torch.empty((N, K // 128), dtype=torch.float32, device=w.device)
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        b = w[r0:r1].float().view(r1 - r0, K // 128, 128)
        sc = (b.abs().amax(dim=2) / 127.0).clamp(min=1e-12)
        q[r0:r1] = torch.round(b / sc[:, :, None]).clamp(-127, 127).to(torch.int8).view(r1 - r0, K)
        s[r0:r1] = sc
        del b
    return q, s


@torch.no_grad()
def convert_model(model: torch.nn.Module) -> None:
    mode = os.environ.get("GLM53_DENSE_W8", "0").strip().lower()
    if mode in ("", "0", "off", "false", "bf16"):
        return
    if getattr(model, "_glm53_dense_done", False):
        return
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead, UnquantizedEmbeddingMethod,
                                                                     VocabParallelEmbedding)

    min_numel = int(os.environ.get("GLM53_DENSE_MIN_NUMEL", str(1 << 20)))
    extra = os.environ.get("GLM53_DENSE_EXCLUDE", "").strip()
    excl = re.compile(_BUILTIN_EXCLUDE + (f"|({extra})" if extra else ""))
    mods = dict(model.named_modules())
    embed_ptrs = {m.weight.data_ptr() for m in mods.values()
                  if type(m) is VocabParallelEmbedding and getattr(m, "weight", None) is not None}
    t0 = time.time()
    counts = {"fp8": 0, "int8": 0}
    bytes_before = bytes_after = 0
    seen: dict[int, tuple] = {}
    kept: list[str] = []
    for name, m in mods.items():
        is_lin = isinstance(m, LinearBase) and isinstance(getattr(m, "quant_method", None), UnquantizedLinearMethod)
        is_head = isinstance(m, ParallelLMHead) and isinstance(getattr(m, "quant_method", None),
                                                               UnquantizedEmbeddingMethod)
        if not (is_lin or is_head):
            continue
        w = getattr(m, "weight", None)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_cuda or w.shape[1] % 128:
            continue
        if w.numel() < min_numel:
            continue
        if excl.search(name) or (is_head and w.data_ptr() in embed_ptrs):
            kept.append(name)
            continue
        key = w.data_ptr()
        try:
            if key in seen:  # the same Parameter under two modules: reuse its conversion
                q, s, srow, fmt = seen[key]
            else:
                if fp8_origin(w.data):
                    fmt, srow = "fp8", 128
                elif mode == "origin":
                    continue
                else:
                    fmt, srow = "int8", 1
                q, s = quantize_fp8_block(w.data) if fmt == "fp8" else quantize_int8_rowgroup(w.data)
                seen[key] = (q, s, srow, fmt)
                bytes_before += w.numel() * 2
                bytes_after += q.numel() + s.numel() * 4
            method = W8DenseMethod(m.quant_method, use_scratch=is_lin)
        except Exception as exc:  # noqa: BLE001 - this matrix stays BF16; the model stays consistent
            logger.warning("[glm53-dense] %s stays BF16: %r", name, exc)
            continue
        # one module at a time: weight, scale and method change together
        m.weight = torch.nn.Parameter(q, requires_grad=False)
        m.glm53_w8_scale = s
        m.glm53_w8_srow = srow
        m.quant_method = method
        if is_lin:
            _State.scratch_numel = max(_State.scratch_numel, q.numel())
        counts[fmt] += 1
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    model._glm53_dense_done = True
    n = counts["fp8"] + counts["int8"]
    logger.info("[glm53-dense] W8A16 (%s): %d fp8-block + %d int8-rowgroup matrices, %.2f GB -> %.2f GB this rank "
                "(%.1fs); kept BF16 by exclusion: %d", mode, counts["fp8"], counts["int8"], bytes_before / 1e9,
                bytes_after / 1e9, time.time() - t0, len(kept))
    if n:
        _warm(model)


@torch.no_grad()
def _warm(model: torch.nn.Module) -> None:
    """Compile every kernel variant now (decode sizes, one tile size, the scratch path), not on a request."""
    t0 = time.time()
    shapes = {}
    for _, m in model.named_modules():
        if isinstance(getattr(m, "quant_method", None), W8DenseMethod):
            shapes[(m.weight.shape[0], m.weight.shape[1], m.glm53_w8_srow, m.quant_method.use_scratch)] = m
    for (N, K, srow, scr), m in shapes.items():
        # every Triton specialization a serving step can hit (M == 1, M % 16 == 0 or not, each BM and path): a
        # first-time JIT in the middle of serving costs seconds (jit_monitor saw _w8_mm_kernel at M=65..255)
        for M in (1, 2, 3, 4, 5, 8, 16, 17, 24, 32, 33, 48, 64, 65, 100, 128, 200, 256, 300, 512):
            x = torch.zeros((M, K), dtype=torch.bfloat16, device=m.weight.device)
            w8_matmul(x, m.weight, m.glm53_w8_scale, m.glm53_w8_srow, scr)
    torch.cuda.synchronize()
    logger.info("[glm53-dense] warmed %d shapes in %.1fs", len(shapes), time.time() - t0)
