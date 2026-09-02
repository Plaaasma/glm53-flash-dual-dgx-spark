import time, torch
import nvfp4_gather, nvfp4_dsmla

dev = "cuda"
torch.manual_seed(3)
R = 200_000                                     # pool rows in play (~200K ctx)
kv = torch.randn(R, 512, dtype=torch.bfloat16, device=dev)
pk, sc = nvfp4_gather.quantize_to_nvfp4_triton(kv)
pool = torch.cat([pk, sc], dim=1)

def bench(fn, n=20):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

# decode step: 32 q-tokens x 2048 topk (c4 with DFlash2 k=7)
idx_d = torch.randint(0, R, (32 * 2048,), device=dev, dtype=torch.int32)
out_d = torch.empty(idx_d.shape[0], 656, dtype=torch.uint8, device=dev)
t = bench(lambda: nvfp4_dsmla.gather_nvfp4_to_dsmla(pool, idx_d, out_d))
print(f"decode-shape gather  (65,536 rows): {t:6.2f} ms/layer")

# prefill chunk: 2048 q x 2048 topk -> unique + gather + inverse remap
idx_p = torch.randint(0, R, (2048, 2048), device=dev, dtype=torch.int32)
def prefill_path():
    uniq, inv = torch.unique(idx_p, return_inverse=True)
    s = nvfp4_dsmla.gather_nvfp4_to_dsmla(pool, uniq.int())
    return inv.view(2048, 2048)
t2 = bench(prefill_path, n=8)
uniq = torch.unique(idx_p).shape[0]
print(f"prefill-chunk union ({uniq:,} unique of 4.2M refs): {t2:6.2f} ms/layer")

# write path at prefill rate
x = torch.randn(2048, 512, dtype=torch.bfloat16, device=dev)
t3 = bench(lambda: nvfp4_gather.quantize_to_nvfp4_triton(x))
print(f"write quantize (2048 tokens): {t3:6.3f} ms/layer")
