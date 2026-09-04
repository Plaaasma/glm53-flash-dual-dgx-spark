"""Sync-free gather path vs unique path: byte equivalence + no host syncs."""
import os, sys, torch
os.environ["GLM53_NVFP4_KV"] = "1"
os.environ["GLM53_NVFP4_SYNCFREE_T"] = "192"
sys.path.insert(0, "/nv")
import glm53_nvfp4_runtime as rt

torch.manual_seed(0)
dev = "cuda:0"
POOL_ROWS = 100_000
pool = torch.randint(0, 256, (POOL_ROWS, 288), dtype=torch.uint8, device=dev)

def ref_unique(tables):
    saved = rt._SYNC_FREE_T
    rt._SYNC_FREE_T = 0          # force the unique branch
    try:
        return rt.build_scratch(pool, tables)
    finally:
        rt._SYNC_FREE_T = saved

def check(T, K, pad_frac=0.1, dup=True):
    idx = torch.randint(0, POOL_ROWS, (T, 1, K), dtype=torch.int32, device=dev)
    if dup:
        idx[:, :, : K // 4] = idx[:1, :, : K // 4]      # cross-token duplicate rows
    pad = torch.rand(T, 1, K, device=dev) < pad_frac
    idx[pad] = -1
    sa, ta = rt.build_scratch(pool, idx)                # sync-free path
    sb, tb = ref_unique(idx)
    fa = sa.reshape(-1, 656)[ta.view(-1).long()]
    fb = sb.reshape(-1, 656)[tb.view(-1).long()]
    assert torch.equal(fa, fb), f"MISMATCH T={T} K={K}"
    print(f"  T={T:>3} K={K:>4}: byte-identical across {T*K} table entries")

print("equivalence (small K):")
for T in (1, 3, 17, 64, 65, 100, 192):
    check(T, 64)
print("equivalence (real K=2048):")
rt._bufs.clear()
for T in (20, 65, 148, 192):
    check(T, 2048)

print("sync-freedom on the fast path:")
torch.cuda.synchronize()
torch.cuda.set_sync_debug_mode("error")
try:
    idx = torch.randint(0, POOL_ROWS, (148, 1, 2048), dtype=torch.int32, device=dev)
    for _ in range(3):
        rt.build_scratch(pool, idx)                     # buffers exist: no allocs, no syncs
    print("  T=148 K=2048: zero synchronizing calls  ✓")
finally:
    torch.cuda.set_sync_debug_mode("default")
torch.cuda.synchronize()
print("ALL PASS")
