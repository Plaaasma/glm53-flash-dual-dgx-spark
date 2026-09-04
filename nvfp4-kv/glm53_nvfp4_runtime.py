# SPDX-License-Identifier: MIT
"""NVFP4 KV pool runtime for GLM-5.3 sparse MLA (vLLM EXL3 stack).

Pool rows: 288 B = 256 B packed e2m1 pairs + 32 B e4m3 per-16 block scales
(global scale folded to 1.0 -- kv_a_layernorm keeps the latent O(1); measured
identical reconstruction error to a calibrated scale on this model).

Read side emits fp8_ds_mla 656-byte rows for the prebuilt trtllm sparse-MLA
kernel: [512 B fp8 | 4 x fp32 per-128-group scales | 128 B rope zeros] --
layout confirmed EMPIRICALLY against concat_and_cache_mla output.

Validated standalone (test_kernel_equiv.py): real kernel on this scratch vs
real kernel on a real fp8 pool -> cos 0.99483 / rel 0.102 = NVFP4's intrinsic
noise, the same precision that scored GSM8K 98/100 on the SGLang stack.
Costs: decode gather 0.36 ms/layer (64K rows), prefill union 5.6 ms/layer
worst-case, write 0.036 ms/2048 tokens.

Activation: env GLM53_NVFP4_KV=1 (checked once at import).
"""
import os

import torch
import triton
import triton.language as tl

NV_BYTES = 288
DS_BYTES = 656

_ON = os.environ.get("GLM53_NVFP4_KV", "0") == "1"


def enabled() -> bool:
    return _ON


# --------------------------- write path ---------------------------
@triton.jit
def _quant_kernel(src_ptr, pool_ptr, slot_ptr, src_stride,
                  HALF: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slot_ptr + row).to(tl.int64)
    if slot < 0:
        return
    g = tl.arange(0, 32)
    goff = g[:, None] * 16 + tl.arange(0, 16)[None, :]
    gv = tl.load(src_ptr + row * src_stride + goff).to(tl.float32)
    bs = tl.maximum(tl.max(tl.abs(gv), axis=1) / 6.0, 1e-8).to(tl.float8e4nv)
    tl.store(pool_ptr + slot * 288 + 256 + g, bs.to(tl.uint8, bitcast=True))
    bs_f = bs.to(tl.float32)

    i = tl.arange(0, HALF)
    sb = i // 8
    s_i = tl.sum(tl.where(g[None, :] == sb[:, None], bs_f[None, :], 0.0), axis=1)
    lo_v = tl.load(src_ptr + row * src_stride + i * 2).to(tl.float32) / s_i
    hi_v = tl.load(src_ptr + row * src_stride + i * 2 + 1).to(tl.float32) / s_i

    a = tl.abs(lo_v)
    lo_c = tl.where(a < 0.25, 0, tl.where(a < 0.75, 1, tl.where(a < 1.25, 2,
           tl.where(a < 1.75, 3, tl.where(a < 2.5, 4, tl.where(a < 3.5, 5,
           tl.where(a < 5.0, 6, 7))))))) | tl.where(lo_v < 0, 8, 0)
    a = tl.abs(hi_v)
    hi_c = tl.where(a < 0.25, 0, tl.where(a < 0.75, 1, tl.where(a < 1.25, 2,
           tl.where(a < 1.75, 3, tl.where(a < 2.5, 4, tl.where(a < 3.5, 5,
           tl.where(a < 5.0, 6, 7))))))) | tl.where(hi_v < 0, 8, 0)
    tl.store(pool_ptr + slot * 288 + i, (lo_c | (hi_c << 4)).to(tl.uint8))


def write_nvfp4(kv_c: torch.Tensor, kv_cache: torch.Tensor,
                slot_mapping: torch.Tensor) -> None:
    """kv_c [T,512] -> NVFP4 rows at slot_mapping in kv_cache (viewed [-1,288])."""
    pool = kv_cache.view(-1, NV_BYTES)
    T = kv_c.shape[0]
    if T == 0:
        return
    _quant_kernel[(T,)](kv_c, pool, slot_mapping, kv_c.stride(0),
                        HALF=256, num_warps=4)


# --------------------------- read path ---------------------------
@triton.jit
def _e2m1(mag):
    e = mag >> 1
    m = mag & 1
    return tl.where(e == 0, 0.5 * m, tl.exp2((e - 1).to(tl.float32)) * (1 + 0.5 * m))


@triton.jit
def _gather_kernel(pool_ptr, idx_ptr, out_u8_ptr, out_f32_ptr, S,
                   HALF: tl.constexpr):
    row = tl.program_id(0)
    if row >= S:
        return
    src = tl.load(idx_ptr + row).to(tl.int64)
    src = tl.maximum(src, 0)  # -1-padded table entries: gather row 0, consumer masks them
    b = tl.arange(0, HALF)
    byts = tl.load(pool_ptr + src * 288 + b).to(tl.int32)
    lo_c = byts & 0xF
    hi_c = (byts >> 4) & 0xF
    lo = tl.where((lo_c & 8) != 0, -_e2m1(lo_c & 7), _e2m1(lo_c & 7))
    hi = tl.where((hi_c & 8) != 0, -_e2m1(hi_c & 7), _e2m1(hi_c & 7))
    sraw = tl.load(pool_ptr + src * 288 + 256 + (b * 2) // 16)
    s16 = sraw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    lo = lo * s16
    hi = hi * s16
    g128 = b // 64
    m = tl.maximum(tl.abs(lo), tl.abs(hi))
    amax = tl.max(tl.where(g128[:, None] == tl.arange(0, 4)[None, :],
                           m[:, None], 0.0), axis=0)
    gscale = tl.maximum(amax / 448.0, 1e-12)
    tl.store(out_f32_ptr + row * 164 + 128 + tl.arange(0, 4), gscale)
    inv = tl.sum(tl.where(g128[:, None] == tl.arange(0, 4)[None, :],
                          (1.0 / gscale)[None, :], 0.0), axis=1)
    tl.store(out_u8_ptr + row * 656 + b * 2,
             (lo * inv).to(tl.float8e4nv).to(tl.uint8, bitcast=True))
    tl.store(out_u8_ptr + row * 656 + b * 2 + 1,
             (hi * inv).to(tl.float8e4nv).to(tl.uint8, bitcast=True))
    r = tl.arange(0, 128)
    tl.store(out_u8_ptr + row * 656 + 528 + r, tl.zeros([128], tl.uint8))


_bufs: dict = {}


def _gather(pool: torch.Tensor, idx: torch.Tensor, out: torch.Tensor) -> None:
    S = idx.shape[0]
    f32 = out.reshape(-1).view(torch.float32).reshape(out.shape[0], DS_BYTES // 4)
    _gather_kernel[(S,)](pool, idx, out, f32, S, HALF=256, num_warps=4)


# Sync-free ceiling: the largest step that must stay host-sync-free. Mixed
# steps (the c16 + long-prefill case) are capped by design at 128 prefill
# tokens (GLM53_MIXED_PREFILL_CHUNK) + 16 seqs x 4 spec slots (MTP-3) = 192.
# 87 host round-trips per step (45 MLA unique/alloc syncs + 42 MoE .item()
# syncs) measured 830 ms/step under concurrent load, 2026-09-03; keeping
# every mixed step below this cap keeps the whole step launch-and-forget.
_SYNC_FREE_T = int(os.environ.get("GLM53_NVFP4_SYNCFREE_T", "192"))


def build_scratch(kv_cache: torch.Tensor, block_tables: torch.Tensor):
    """kv_cache: allocated pool (any shape, 288 B/token rows).
    block_tables: [T, 1, K] int32 physical token rows (-1 padded).
    Returns (scratch [S,656] uint8, tables [T,1,K] into scratch).

    T <= _SYNC_FREE_T (decode and mixed decode+prefill steps; <=64 also
    CUDA-graph captured): fixed shapes, no host syncs -- gathers all T*K
    rows (no dedup; duplicate rows cost bandwidth, ~37 ms/step worst case,
    which is noise next to the ~830 ms/step the host syncs cost) into one
    shared persistent buffer sized for the cap; tables = arange slice.
    T > _SYNC_FREE_T (pure-prefill chunks, eager): torch.unique dedup --
    there the union is ~context-sized, far smaller than T*K, and prefill
    is eager and latency-tolerant anyway.
    """
    pool = kv_cache.view(-1, NV_BYTES)
    T, one, K = block_tables.shape
    flat = block_tables.view(-1)
    dev = pool.device
    # The flashinfer wrapper routes decode-vs-paged partly on the cache's page
    # geometry: page_size must be 64 (like the real pool) or <=64-token steps
    # get sent to the paged kernel, which rejects them. Emit [P, 64, 656].
    if T <= _SYNC_FREE_T:
        S = T * K
        P = (S + 63) // 64
        key = ("shared", K, dev.index)
        ent = _bufs.get(key)
        if ent is None:
            pcap = (_SYNC_FREE_T * K + 63) // 64
            ent = (torch.zeros(pcap, 64, DS_BYTES, dtype=torch.uint8, device=dev),
                   torch.arange(pcap * 64, dtype=torch.int32, device=dev))
            _bufs[key] = ent
        buf, ar = ent
        _gather(pool, flat, buf.view(-1, DS_BYTES)[:S])
        return buf[:P], ar[:S].view(T, 1, K)
    uniq, inv = torch.unique(flat, return_inverse=True)
    S = uniq.shape[0]
    P = (S + 63) // 64
    scratch = torch.zeros(P, 64, DS_BYTES, dtype=torch.uint8, device=dev)
    _gather(pool, uniq.to(torch.int32), scratch.view(-1, DS_BYTES)[:S])
    return scratch, inv.to(torch.int32).view(T, 1, K)
