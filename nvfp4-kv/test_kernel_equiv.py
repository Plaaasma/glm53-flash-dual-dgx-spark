#!/usr/bin/env python3
"""THE decisive test: real sparse-MLA kernel on (A) real-op fp8 pool vs
(B) NVFP4 pool -> gather scratch. Plus (C) bf16 torch ground truth."""
import math, torch
from vllm import _custom_ops as ops
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as mla
import nvfp4_gather, nvfp4_dsmla

torch.manual_seed(7)
dev = "cuda"
N, TOPK, B, H = 8192, 2048, 96, 16          # pool rows, topk, batch, heads
LORA, ROPE = 512, 64
sm = 1.0 / math.sqrt(LORA + ROPE)

kv = torch.randn(N, LORA, dtype=torch.bfloat16, device=dev)
kpe = torch.zeros(N, ROPE, dtype=torch.bfloat16, device=dev)

# --- pool A: the real op ---
poolA = torch.zeros(N, 656, dtype=torch.uint8, device=dev)
ops.concat_and_cache_mla(kv, kpe, poolA.view(N, 1, 656),
                         torch.arange(N, device=dev),
                         "fp8_ds_mla", torch.tensor(1.0, device=dev))

# --- pool B: NVFP4 (288 B/row) ---
pk, sc = nvfp4_gather.quantize_to_nvfp4_triton(kv)
poolB = torch.cat([pk, sc], dim=1)

idx = torch.stack([torch.randperm(N, device=dev)[:TOPK] for _ in range(B)]).int()
q = torch.randn(B, 1, H, LORA + ROPE, dtype=torch.bfloat16, device=dev)
q[..., LORA:] = 0                                     # NoPE
ws = torch.zeros(160 * 1024 * 1024, dtype=torch.uint8, device=dev)
seq = torch.full((B,), TOPK, dtype=torch.int32, device=dev)

def run(pool, tables):
    return mla(query=q, kv_cache=pool.unsqueeze(1), workspace_buffer=ws,
               qk_nope_head_dim=128, kv_lora_rank=LORA, qk_rope_head_dim=ROPE,
               block_tables=tables.unsqueeze(1), seq_lens=seq, max_seq_len=TOPK,
               sparse_mla_top_k=TOPK, bmm1_scale=sm)

outA = run(poolA, idx)

# --- B: gather scratch, renumber tables to scratch rows ---
flat = idx.reshape(-1)
scratch = nvfp4_dsmla.gather_nvfp4_to_dsmla(poolB, flat)
tablesB = torch.arange(B * TOPK, device=dev, dtype=torch.int32).view(B, TOPK)
outB = run(scratch, tablesB)

# --- C: bf16 ground truth ---
sel = kv.float()[idx.long()]                          # [B, TOPK, 512]
qf = q.float().squeeze(1)[..., :LORA]                 # [B, H, 512]
att = torch.softmax(torch.einsum("bhd,btd->bht", qf, sel) * sm, dim=-1)
outC = torch.einsum("bht,btd->bhd", att, sel)

a, b, c = outA.float().view(B, H, -1), outB.float().view(B, H, -1), outC
def cmp(x, y):
    cos = torch.nn.functional.cosine_similarity(x.reshape(-1), y.reshape(-1), dim=0)
    rel = (x - y).norm() / y.norm()
    return f"cos {cos:.5f}  rel {rel:.4f}"
print("A(fp8 real)   vs C(bf16 truth):", cmp(a, c))
print("B(nvfp4 path) vs C(bf16 truth):", cmp(b, c))
print("B             vs A            :", cmp(b, a))
