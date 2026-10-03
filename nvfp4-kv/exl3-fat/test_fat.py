"""glm53_exl3_fat: patch + correctness + benchmark on REAL GLM-5.3-Flash EXL3 weights, in a throwaway container.

Run (one GPU test at a time, memory-guarded):
  gpu_run.sh 2 -e PYTHONDONTWRITEBYTECODE=1 \
     -v "$PWD"/nvfp4-vllm/exl3-fat:/w:ro -v "$PWD"/nvfp4-vllm/exl3-mt:/mt:ro \
     -v "$PWD"/exl3-kit/overlay:/ov:ro -v <snapshot>:/hf:ro \
     glm53-flash-sm121:local-0904-it python3 /w/test_fat.py [patch|correct|bench|all]

1. patch: the image's own exl3.py (site-packages of this throwaway container) gets patch_exl3_mt.py, then
   patch_exl3_fat.py (and both again: idempotence), then is imported.
2. A fake vLLM layer is built from the real checkpoint (layer $LAYER, vLLM rank 0 of TP=2: gate/up columns, down
   rows), with $N_PHYS physical experts. Memory forbids all 288 (1.8 GB): the 288 VIRTUAL experts the router sees
   alias physical expert v % N_PHYS through vLLM's own pointer tables (build_exl3_fused_state), so routing,
   per-expert row counts and the DRAM traffic per expert are realistic (aliases are N_PHYS experts apart, far beyond
   what L2 holds), only the weight values repeat.
3. correct: production env (EXL3_TEMP_ROWS_FUSED=192, GLM53_EXL3_MT=1 VARIANT=8 TEMP_ROWS=1024 MIN_ROWS=32) with
   GLM53_EXL3_FAT=0 (current) vs =1 (new), both through the patched apply_exl3_fused_moe, vs an fp64 CPU reference
   (exllamav3 reconstruct -> fp64, exact Hadamards) on a subset of rows.
4. bench: per-layer time of the same calls at T = 128..2048, realistic skewed routing.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
SITE = "/usr/local/lib/python3.12/dist-packages"
EXL3_PY = SITE + "/vllm/model_executor/layers/quantization/exl3.py"

PROD_ENV = {
    "EXL3_FUSED_MOE": "1", "EXL3_MOE_ROW_TILE": "1", "EXL3_TEMP_ROWS_FUSED": "192",
    "GLM53_EXL3_MT": "1", "GLM53_EXL3_MT_VARIANT": "8", "GLM53_EXL3_MT_TEMP_ROWS": "1024", "GLM53_EXL3_MT_MIN_ROWS": "32",
    "GLM53_EXL3_FAT": "0", "GLM53_EXL3_FAT_MIN_TOKENS": "64", "GLM53_EXL3_FAT_MAX_TOKENS": "2048",
}
for k, v in PROD_ENV.items():
    os.environ.setdefault(k, v)


def memavail() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return -1.0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- patch
def run_patchers() -> None:
    env = dict(os.environ, GLM53_EXL3_MT_SO="/mt/glm53_exl3_mt.so", GLM53_EXL3_FAT_SO="/w/glm53_exl3_fat.so",
               GLM53_EXL3_FAT_RT="/w/glm53_exl3_fat_rt.py")
    for patcher in ("/ov/patch_exl3_mt.py", "/ov/patch_exl3_fat.py", "/ov/patch_exl3_mt.py", "/ov/patch_exl3_fat.py"):
        r = subprocess.run([sys.executable, patcher], env=env, capture_output=True, text=True)
        log(f"[patch] {os.path.basename(patcher)} rc={r.returncode}: {r.stdout.strip()} {r.stderr.strip()}")
        assert r.returncode == 0, "patcher failed"
    text = open(EXL3_PY).read()
    assert text.count("# [glm53-exl3-mt]") >= 1 and text.count("[glm53-exl3-fat] prefill-size calls") == 1
    log("[patch] exl3.py carries both patches once")


if MODE in ("patch", "all", "correct", "bench"):
    run_patchers()

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
mem0 = memavail()
log(f"[mem] MemAvailable at start {mem0:.2f} GiB")

from vllm.model_executor.layers.quantization import exl3 as X  # noqa: E402
import exllamav3_ext  # noqa: E402
import glm53_exl3_fat as FAT  # noqa: E402
import glm53_exl3_fat_rt as RT  # noqa: E402

torch.set_num_threads(4)
_CALLS = {"n": 0, "ok": 0}
_apply_local = RT.apply_local


def _counted_apply_local(*a, **k):
    _CALLS["n"] += 1
    r = _apply_local(*a, **k)
    _CALLS["ok"] += r is not None
    return r


RT.apply_local = _counted_apply_local
log(f"[import] vllm exl3 from {X.__file__}; fat kernels gu={FAT.kernel_names(False)} dn={FAT.kernel_names(True)}")
log(f"[mem] after imports {memavail():.2f} GiB available")

# ----------------------------------------------------------------------------------------------------------- weights
SNAP = "/hf"
LAYER = int(os.environ.get("LAYER", "20"))
N_PHYS = int(os.environ.get("N_PHYS", "48"))
E_V = 288
HID, FULL_I, TP = 4096, 2048, 2
I = FULL_I // TP
LIMIT = 10.0
TOPK = 8

from safetensors import safe_open  # noqa: E402

wmap = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
pre = f"model.language_model.layers.{LAYER}.mlp.experts."
handles: dict[str, object] = {}


def get(name: str, narrow: tuple[int, int, int] | None = None) -> torch.Tensor:
    f = wmap[name]
    if f not in handles:
        handles[f] = safe_open(f"{SNAP}/{f}", framework="pt", device="cpu")
    h = handles[f]
    if narrow is None:
        return h.get_tensor(name)
    dim, start, length = narrow
    sl = h.get_slice(name)
    idx = [slice(None)] * len(sl.get_shape())
    idx[dim] = slice(start, start + length)
    return sl[tuple(idx)]


t0 = time.time()


class FakeLayer(torch.nn.Module):
    pass


layer = FakeLayer()
layer.w13_trellis = torch.empty(N_PHYS, 2, HID // 16, I // 16, 64, dtype=torch.int16, device=dev)
layer.w13_suh = torch.empty(N_PHYS, 2, HID, dtype=torch.float16, device=dev)
layer.w13_svh = torch.empty(N_PHYS, 2, I, dtype=torch.float16, device=dev)
layer.w13_mcg = torch.empty(N_PHYS, 2, 1, dtype=torch.int32, device=dev)
layer.w2_trellis = torch.empty(N_PHYS, I // 16, HID // 16, 64, dtype=torch.int16, device=dev)
layer.w2_suh = torch.empty(N_PHYS, I, dtype=torch.float16, device=dev)
layer.w2_svh = torch.empty(N_PHYS, HID, dtype=torch.float16, device=dev)
layer.w2_mcg = torch.empty(N_PHYS, 1, dtype=torch.int32, device=dev)
for e in range(N_PHYS):
    for j, proj in enumerate(("gate_proj", "up_proj")):
        b = f"{pre}{e}.{proj}."
        layer.w13_trellis[e, j].copy_(get(b + "trellis", (1, 0, I // 16)))   # rank 0 columns (shard_exl3_col)
        layer.w13_suh[e, j].copy_(get(b + "suh"))
        layer.w13_svh[e, j].copy_(get(b + "svh", (0, 0, I)))
        layer.w13_mcg[e, j].copy_(get(b + "mcg").reshape(-1)[:1])
    b = f"{pre}{e}.down_proj."
    layer.w2_trellis[e].copy_(get(b + "trellis", (0, 0, I // 16)))            # rank 0 rows (shard_exl3_row)
    layer.w2_suh[e].copy_(get(b + "suh", (0, 0, I)))
    layer.w2_svh[e].copy_(get(b + "svh"))
    layer.w2_mcg[e].copy_(get(b + "mcg").reshape(-1)[:1])
handles.clear()
assert torch.all(layer.w13_mcg == X.MCG_MARKER_SIGNED_INT32) and torch.all(layer.w2_mcg == X.MCG_MARKER_SIGNED_INT32)
layer._exl3_hidden_size, layer._exl3_intermediate_local, layer._exl3_bits, layer._exl3_k_words = HID, I, 4, 64
phys = []
for e in range(N_PHYS):
    phys.append({
        "gate": X.make_linear_exl3(layer.w13_trellis[e, 0], layer.w13_suh[e, 0], layer.w13_svh[e, 0], layer.w13_mcg[e, 0]),
        "up": X.make_linear_exl3(layer.w13_trellis[e, 1], layer.w13_suh[e, 1], layer.w13_svh[e, 1], layer.w13_mcg[e, 1]),
        "down": X.make_linear_exl3(layer.w2_trellis[e], layer.w2_suh[e], layer.w2_svh[e], layer.w2_mcg[e]),
    })
inners = [phys[v % N_PHYS] for v in range(E_V)]
X.build_exl3_fused_state(layer, inners)                    # vLLM's own pointer tables (288 aliased entries) + temps
layer._exl3_inners = inners
wbytes = sum(t.numel() * t.element_size() for t in (layer.w13_trellis, layer.w13_suh, layer.w13_svh, layer.w2_trellis,
                                                     layer.w2_suh, layer.w2_svh))
torch.cuda.synchronize()
log(f"[load] layer {LAYER}: {N_PHYS} physical experts ({wbytes / 2**20:.0f} MiB) aliased to {E_V} virtual in "
    f"{time.time() - t0:.1f}s; concurrency {layer._exl3_fused_concurrency}; shared suh (gate==up) for all: "
    f"{bool(torch.equal(layer.w13_suh[:, 0], layer.w13_suh[:, 1]))}; MemAvailable {memavail():.2f} GiB")


# ----------------------------------------------------------------------------------------------------------- routing
def zipf_pop(s: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pop = 1.0 / torch.arange(1, E_V + 1, dtype=torch.float64) ** s
    return (pop / pop.sum())[torch.randperm(E_V, generator=g)]          # hot experts scattered over the ids


ZIPF_S = float(os.environ.get("ZIPF_S", "0.7"))


def routing(T: int, kind: str, seed: int):
    g = torch.Generator(device=dev).manual_seed(seed)
    if kind == "zipf":
        pop = zipf_pop(ZIPF_S, seed).to(dev, torch.float32)
        ids = torch.multinomial(pop.expand(T, -1), TOPK, replacement=False, generator=g)
    elif kind == "skew":       # a few experts in most tokens (rows 0.9 / 0.6 / 0.35 x T), the rest Zipf
        pop = zipf_pop(ZIPF_S, seed).to(dev, torch.float32)
        hot = torch.argsort(pop, descending=True)[:3]
        rest = pop.clone()
        rest[hot] = 0
        ids = torch.multinomial(rest.expand(T, -1), TOPK, replacement=False, generator=g)
        for i, p in enumerate((0.9, 0.6, 0.35)):
            sel = torch.rand(T, device=dev, generator=g) < p
            ids[sel, TOPK - 1 - i] = hot[i]
    elif kind == "uniform":
        ids = torch.rand(T, E_V, device=dev, generator=g).argsort(dim=1)[:, :TOPK]
    else:
        raise ValueError(kind)
    w = torch.rand(T, TOPK, device=dev, generator=g) + 0.05
    w = (w / w.sum(-1, keepdim=True) * 2.5).float()
    x = torch.randn(T, HID, device=dev, generator=g)
    x[:, :: 512] *= 8.0                                                   # a few outlier channels
    return x.to(torch.bfloat16), ids.long(), w


def row_stats(ids: torch.Tensor) -> str:
    c = torch.bincount(ids.reshape(-1), minlength=E_V)
    return (f"rows/expert min {int(c.min())} mean {c.float().mean():.1f} max {int(c.max())}, experts hit {int((c > 0).sum())}, "
            f">=300 rows: {int((c >= 300).sum())}, >1024: {int((c > 1024).sum())}")


# ----------------------------------------------------------------------------------------------------------- paths
def path(x, ids, w, fat: bool, kern: str | None = None):
    os.environ["GLM53_EXL3_FAT"] = "1" if fat else "0"
    if kern is not None:
        RT._ST["cfg"] = None
        os.environ["GLM53_EXL3_FAT_KERNEL"] = kern
    return X.apply_exl3_fused_moe(x, ids, w, layer, inners, None, LIMIT)


# ----------------------------------------------------------------------------------------------------------- fp64 ref
def hadamard128() -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(7):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(128.0)


H128 = hadamard128()


def had(v: torch.Tensor) -> torch.Tensor:
    return (v.reshape(-1, 128) @ H128).reshape(v.shape)


def deq(trellis: torch.Tensor, k_in: int, n_out: int) -> torch.Tensor:
    w = torch.empty(k_in, n_out, dtype=torch.float16, device=dev)
    exllamav3_ext.reconstruct(w, trellis, 4, True, False)
    return w.cpu().double()


def reference(x, ids, w, rows: torch.Tensor, act_mode: int = 2) -> torch.Tensor:
    """fp64 on the CPU: y = sum_k w_k * had(had(a_k * suh_d) @ Wd) * svh_d, a_k = SwiGLU_limit(gate, up) with
    gate = had(had(x * suh_g) @ Wg) * svh_g (exllamav3 LinearEXL3's definition), Wd/Wg exact (reconstruct)."""
    rows = rows.cpu()
    xs = x[rows.to(dev)].double().cpu()
    ids_s = ids[rows.to(dev)].cpu()
    ws = w[rows.to(dev)].double().cpu()
    ref = torch.zeros(len(rows), HID, dtype=torch.float64)
    phys_ids = ids_s % N_PHYS
    for e in sorted(set(phys_ids.reshape(-1).tolist())):
        r_idx, k_idx = (phys_ids == e).nonzero(as_tuple=True)
        Wg = deq(layer.w13_trellis[e, 0], HID, I)
        Wu = deq(layer.w13_trellis[e, 1], HID, I)
        Wd = deq(layer.w2_trellis[e], I, HID)
        sg, su = layer.w13_suh[e, 0].cpu().double(), layer.w13_suh[e, 1].cpu().double()
        vg, vu = layer.w13_svh[e, 0].cpu().double(), layer.w13_svh[e, 1].cpu().double()
        sd, vd = layer.w2_suh[e].cpu().double(), layer.w2_svh[e].cpu().double()
        h = xs[r_idx]
        g = had(had(h * sg) @ Wg) * vg
        u = had(had(h * su) @ Wu) * vu
        u = u.clamp(-LIMIT, LIMIT)
        if act_mode == 1:
            gg = g.clamp(max=LIMIT)
            a = gg * torch.sigmoid(gg) * u
        else:
            a = (g * torch.sigmoid(g)).clamp(max=LIMIT) * u
        y = had(had(a * sd) @ Wd) * vd
        ref.index_add_(0, r_idx, y * ws[r_idx, k_idx].unsqueeze(-1))
        del Wg, Wu, Wd
    return ref


def err(out: torch.Tensor, ref: torch.Tensor, rows: torch.Tensor) -> tuple[float, float]:
    o = out[rows.to(out.device)].double().cpu()
    return float((o - ref).norm() / ref.norm()), float((o - ref).abs().max() / ref.abs().max())


# ----------------------------------------------------------------------------------------------------------- correct
def correctness() -> None:
    log("\n== correctness (real weights; rel = ||out - ref||_F / ||ref||_F, max = max|out - ref| / max|ref|, "
        "on the reference rows)")
    # sanity of the reference itself vs exllamav3's LinearEXL3 (the python loop, act order 1) on one case
    cases = [(128, "zipf", 1), (512, "zipf", 2), (2048, "zipf", 3), (2048, "skew", 4), (512, "skew", 5)]
    log(f"{'case':>16} {'routing stats':>78} | {'cur rel':>9} {'cur max':>9} | {'new rel':>9} {'new max':>9} | "
        f"{'new vs cur (all rows) rel':>26} | rerun max|d| new / cur")
    for T, kind, seed in cases:
        x, ids, w = routing(T, kind, seed)
        rows = torch.linspace(0, T - 1, min(T, 48)).round().long()
        ref = reference(x, ids, w, rows)
        cur = path(x, ids, w, fat=False)
        cur2 = path(x, ids, w, fat=False)
        ok0 = _CALLS["ok"]
        new = path(x, ids, w, fat=True)
        new2 = path(x, ids, w, fat=True)
        torch.cuda.synchronize()
        assert _CALLS["ok"] == ok0 + 2 and not X._GLM53_FAT["off"], "fat path was not taken"
        crr = float((cur - cur2).abs().max())
        cr, cm = err(cur, ref, rows)
        nr, nm = err(new, ref, rows)
        nvc = float((new.double() - cur.double()).norm() / cur.double().norm())
        rr = float((new - new2).abs().max())
        log(f"{T:>6} {kind:>9} {row_stats(ids):>78} | {cr:9.2e} {cm:9.2e} | {nr:9.2e} {nm:9.2e} | {nvc:26.2e} | {rr:.1e} / {crr:.1e}")
        for kern in ("fat,fat", "fast2,fast2"):
            o = path(x, ids, w, fat=True, kern=kern)
            r1, m1 = err(o, ref, rows)
            d = float((o.double() - new.double()).norm() / new.double().norm())
            log(f"{'':>16}   kernels {kern:12s}: rel {r1:.2e} max {m1:.2e}; vs default rel {d:.1e}")
        os.environ.pop("GLM53_EXL3_FAT_KERNEL", None)
        RT._ST["cfg"] = None
        oa = RT.apply_local(x, X.map_topk_to_local(ids, E_V, None), w, layer, LIMIT, TOPK, combine=False)
        ra, ma = err(oa, ref, rows)
        log(f"{'':>16}   atomics mode (GLM53_EXL3_FAT_COMBINE=0): rel {ra:.2e} max {ma:.2e}")
        if T == 512 and kind == "zipf":
            loop = X.apply_exl3_python_loop(x, ids, w, inners, None, LIMIT)
            ref1 = reference(x, ids, w, rows, act_mode=1)
            lr, lm = err(loop, ref1, rows)
            d12 = float((ref1 - ref).norm() / ref.norm())
            log(f"{'':>16}   reference sanity: LinearEXL3 loop (fp16 GEMMs, act order 1) vs fp64 ref(order 1) rel "
                f"{lr:.2e} max {lm:.2e}; ref order 1 vs order 2 rel {d12:.1e}")
        del x, ids, w, cur, cur2, new, new2, ref
        torch.cuda.empty_cache()
    # fallbacks / edge cases
    log("\n== dispatch and edge cases")
    x, ids, w = routing(64, "zipf", 7)
    a = path(x, ids, w, fat=False)
    a2 = path(x, ids, w, fat=False)
    n0 = _CALLS["n"]
    b = path(x, ids, w, fat=True)
    log(f"tokens 64 (<= MIN_TOKENS): fat runtime not called: {_CALLS['n'] == n0}; on vs off max|d| "
        f"{float((a - b).abs().max()):.1e} (off vs off rerun {float((a - a2).abs().max()):.1e})")
    x, ids, w = routing(300, "zipf", 8)
    ids[::7, 3] = -1                                     # invalid / non-local ids -> sentinel, skipped by both paths
    ids[::11, 5] = 10_000
    a = path(x, ids, w, fat=False)
    ok0 = _CALLS["ok"]
    b = path(x, ids, w, fat=True)
    log(f"tokens 300 with invalid ids: fat path taken {_CALLS['ok'] == ok0 + 1}; new vs cur rel "
        f"{float((a - b).norm() / a.norm()):.2e}")
    os.environ["GLM53_EXL3_FAT_MAX_TOKENS"] = "256"
    RT._ST["cfg"] = None
    n0, ok0 = _CALLS["n"], _CALLS["ok"]
    c = path(x, ids, w, fat=True)
    log(f"tokens 300 > GLM53_EXL3_FAT_MAX_TOKENS=256: runtime called {_CALLS['n'] - n0}, returned None -> existing "
        f"path: {_CALLS['ok'] == ok0}; vs cur rel {float((a - c).norm() / a.norm()):.1e}")
    os.environ["GLM53_EXL3_FAT_MAX_TOKENS"] = "2048"
    RT._ST["cfg"] = None
    nc = torch.empty(300, HID * 2, dtype=torch.bfloat16, device=dev)[:, :HID]
    nc.copy_(x)
    d = path(nc, ids, w, fat=True)
    log(f"non-contiguous rows (row stride {nc.stride(0)}): vs contiguous rel {float((d - b).norm() / b.norm()):.1e}")
    del x, ids, w, a, a2, b, c, d, nc
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------- bench
def timeit(fn, iters: int, warm: int = 2) -> tuple[float, float]:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    t0 = time.perf_counter()
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / iters * 1e3
    return statistics.median(s.elapsed_time(e) for s, e in ev), wall


def mt_kernel_only(x, ids, w):
    """The current prefill path's kernel launch alone (exl3_moe_mt variant 8), args built the way vLLM builds them."""
    mod = X._glm53_mt_module()
    local = X.map_topk_to_local(ids, E_V, None)
    T = x.shape[0]
    flat_token = torch.arange(T, device=dev, dtype=torch.long).repeat_interleave(TOPK)
    order = local.argsort()
    ts, ws = flat_token[order], w.reshape(-1).half()[order]
    ec = torch.zeros(E_V + 1, dtype=torch.long, device=dev)
    ec.scatter_add_(0, local, torch.ones_like(local))
    xh = x.half()
    temps = X._glm53_mt_temps(dev, HID, I, int(layer._exl3_fused_concurrency))
    locks = X._glm53_mt_locks(dev, mod)
    p = layer._exl3_ptrs
    out = torch.zeros(T, HID, dtype=torch.float32, device=dev)

    def run():
        mod.exl3_moe_mt(xh, out, ec, ts, ws, *temps, X.MOE_ACT_SILU, 4, p["gate_trellis"], p["gate_suh"], p["gate_svh"],
                        p["up_trellis"], p["up_suh"], p["up_svh"], p["down_trellis"], p["down_suh"], p["down_svh"],
                        LIMIT, locks, 8)
    return run


SIG = os.environ.get("IDLE_SIG", "")          # host-written file: "idle" while the live server runs no request


def server_idle() -> bool | None:
    if not SIG:
        return None
    try:
        return open(SIG).read().strip() == "idle"
    except OSError:
        return None


def bench() -> None:
    log(f"\n== per-layer time, realistic routing (top-8 of 288, Zipf s={ZIPF_S}, hot ids scattered), {N_PHYS} "
        f"physical experts aliased to 288. Interleaved rounds (every variant once a round); per variant the median "
        f"and min of CUDA-event times per call. 'dram' = a 64 MiB device copy in the same rounds (GB/s read+write) "
        f"to show how much bandwidth the live server left us.")
    flops_pair = 2 * (2 * HID * I + I * HID)
    gnames, dnames = FAT.kernel_names(False), FAT.kernel_names(True)
    combos = [tuple(int(v) for v in c.split(":")) for c in
              os.environ.get("COMBOS", "3:5,1:3,0:0").split(",")]
    src = torch.empty(16 * 2**20, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    rounds = int(os.environ.get("ROUNDS", "16"))
    LONG = 24
    summary = []
    cases = [(int(c.split(":")[0]), c.split(":")[1]) for c in os.environ.get(
        "CASES", "80:zipf,128:zipf,192:zipf,256:zipf,512:zipf,1024:zipf,2048:zipf,2048:skew,2048:uniform").split(",")]
    for T, kind in cases:
        x, ids, w = routing(T, kind, 100 + T)
        local = X.map_topk_to_local(ids, E_V, None)
        out = torch.zeros(T, HID, dtype=torch.float32, device=dev)
        variants = {"current": lambda: path(x, ids, w, fat=False), "new": lambda: path(x, ids, w, fat=True)}
        if T > 192:
            variants["current kernel (MT v8)"] = mt_kernel_only(x, ids, w)
        for kg, kd in combos:
            variants[f"{gnames[kg].split()[0]}/{dnames[kd].split()[0]}"] = (
                lambda kg=kg, kd=kd: RT.apply_local(x, local, w, layer, LIMIT, TOPK, kern_gu=kg, kern_dn=kd))
        cfg = RT.config()
        for mask, nm in ((2, "  rot"), (4, "  gate/up"), (8, "  down"), (16, "  combine")):
            variants[nm] = (lambda mask=mask: RT.apply_local(x, local, w, layer, LIMIT, TOPK, out=out,
                                                             stage_mask=mask))
        if os.environ.get("PROBES"):
            for pr, nm in ((1, "  down, stores (probe)"), (2, "  down, no output (probe)")):
                variants[nm] = (lambda pr=pr: RT.apply_local(x, local, w, layer, LIMIT, TOPK, out=out, stage_mask=8,
                                                             probe=pr, combine=False))
            variants["  down, atomics"] = lambda: RT.apply_local(x, local, w, layer, LIMIT, TOPK, out=out,
                                                                 stage_mask=8, combine=False)
        for kg, kd in combos[:2]:
            variants[f"{gnames[kg].split()[0]}/{dnames[kd].split()[0]} atomics"] = (
                lambda kg=kg, kd=kd: RT.apply_local(x, local, w, layer, LIMIT, TOPK, kern_gu=kg, kern_dn=kd,
                                                    combine=False))
        variants["dram"] = lambda: dst.copy_(src)

        def dram_long():
            for _ in range(LONG):
                dst.copy_(src)
        variants["dram x24"] = dram_long
        times = {k: [] for k in variants}
        for f in variants.values():
            f()
        torch.cuda.synchronize()
        idle_rounds = 0
        t_case = time.time()
        r = 0
        while idle_rounds < rounds:
            if SIG:
                if time.time() - t_case > float(os.environ.get("GATE_BUDGET", "240")):
                    break
                if not server_idle():
                    time.sleep(0.05)
                    continue
            elif r >= rounds:
                break
            r += 1
            idle0 = server_idle()
            res = {}
            for k, f in variants.items():
                s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s0.record()
                f()
                e0.record()
                e0.synchronize()
                res[k] = s0.elapsed_time(e0)
            if SIG and not (idle0 and server_idle()):
                continue                                  # gated: keep only rounds that ran while the server was idle
            idle_rounds += 1
            for k in res:
                times[k].append(res[k])
        if not times["new"]:
            log(f"T={T} {kind}: no idle round")
            continue
        med = {k: statistics.median(v) for k, v in times.items()}
        mn = {k: min(v) for k, v in times.items()}
        # GPU share of each round: the long copy's solo time (LONG x the best short copy) over its time that round.
        # est = median over rounds of (variant time x share): the variant's time had the live server been idle,
        # assuming the GPU's time slices are shared evenly over a round.
        solo_long = LONG * mn["dram"]
        share = [min(1.0, solo_long / t) for t in times["dram x24"]]
        est = {k: statistics.median(t * sh for t, sh in zip(v, share)) for k, v in times.items()}
        tf = T * TOPK * flops_pair / 1e9
        gbs = 2 * src.numel() * 4 / 1e6
        log(f"T={T:5d} {kind:8s} {row_stats(ids)}; rounds kept {idle_rounds} of {r}; GPU share median "
            f"{statistics.median(share):.2f} (min {min(share):.2f} max {max(share):.2f})"
            f"{' (server idle)' if SIG else ''}; default kernels {gnames[cfg['kern_gu']].split()[0]}/"
            f"{dnames[cfg['kern_dn']].split()[0]}")
        for k in variants:
            extra = f"  {gbs / med[k]:6.0f} / {gbs / mn[k]:6.0f} GB/s" if k == "dram" else (
                f"  {tf / med[k]:5.1f} TFLOPS (median)" if not k.startswith("  ") else "")
            log(f"   {k:24s} median {med[k]:8.2f} ms   min {mn[k]:8.2f} ms   est. solo {est[k]:8.2f} ms{extra}")
        summary.append((T, kind, med["current"], med["new"], est["current"], est["new"],
                        est.get("current kernel (MT v8)"), statistics.median(share)))
        del x, ids, w, local, out, variants
        torch.cuda.empty_cache()
    log("\nsummary: T, routing, current / new median ms (speedup) | current / new est. solo ms (speedup) | "
        "MT-kernel-only est. solo | GPU share")
    for T, kind, mc, mnw, nc, nn, mk, gb in summary:
        mks = f"{mk:7.2f}" if mk else "      -"
        log(f"  {T:5d} {kind:8s} {mc:7.2f} / {mnw:7.2f} ({mc / mnw:4.2f}x) | {nc:7.2f} / {nn:7.2f} ({nc / nn:4.2f}x) | "
            f"{mks} | {gb:5.2f}")
    del src, dst
    torch.cuda.empty_cache()
    log(f"[mem] torch reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB, max allocated "
        f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB; MemAvailable {memavail():.2f} GiB (start {mem0:.2f})")


for kg in range(len(FAT.kernel_names(False))):
    log(f"[kernel] gu {FAT.kernel_names(False)[kg]:22s} regs/local/smem/grid/BM/threads {FAT.kernel_info(False, kg)}")
for kd in range(len(FAT.kernel_names(True))):
    log(f"[kernel] dn {FAT.kernel_names(True)[kd]:22s} regs/local/smem/grid/BM/threads {FAT.kernel_info(True, kd)}")
if MODE in ("correct", "all"):
    correctness()
if MODE in ("bench", "all"):
    bench()
log(f"[mem] end: torch max allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, MemAvailable "
    f"{memavail():.2f} GiB")
