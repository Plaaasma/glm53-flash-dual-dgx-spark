#!/usr/bin/env python3
"""Empirically determine the fp8_ds_mla 656-byte row layout via the real op."""
import torch
from vllm import _custom_ops as ops

torch.manual_seed(0)
dev = "cuda"
N = 4
kv_c = torch.randn(N, 512, dtype=torch.bfloat16, device=dev)
k_pe = torch.zeros(N, 64, dtype=torch.bfloat16, device=dev)   # NoPE -> zeros
pool = torch.zeros(8, 656, dtype=torch.uint8, device=dev)     # 8 slots, 1 tok/blk view
slot = torch.arange(N, dtype=torch.long, device=dev)
scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
ops.concat_and_cache_mla(kv_c, k_pe, pool.view(8, 1, 656), slot, "fp8_ds_mla", scale)

row = pool[0].cpu()
print("bytes 512:528 (scale region) as fp32:",
      row[512:528].view(torch.float32).tolist())
print("bytes 528:656 (rope region) nonzero count:",
      int((row[528:656] != 0).sum()))
# verify latent region decodes back
f8 = row[:512].view(torch.float8_e4m3fn).to(torch.float32)
ref = kv_c[0].float().cpu()
err = (f8 - ref).norm() / ref.norm()
print(f"latent rel err vs input (expect ~fp8 noise ~0.03): {err:.4f}")
# non-unit scale probe
scale2 = torch.tensor(0.25, dtype=torch.float32, device=dev)
ops.concat_and_cache_mla(kv_c, k_pe, pool.view(8, 1, 656), slot + 4, "fp8_ds_mla", scale2)
row2 = pool[4].cpu()
print("scale=0.25 -> scale region fp32:", row2[512:528].view(torch.float32).tolist())
f8b = row2[:512].view(torch.float8_e4m3fn).to(torch.float32) 
print(f"latent[0:4] w/ scale 0.25: {f8b[:4].tolist()} vs input {ref[:4].tolist()}")
