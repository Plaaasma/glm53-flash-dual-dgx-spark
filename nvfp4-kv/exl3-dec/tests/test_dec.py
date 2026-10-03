"""Correctness of glm53_exl3_dec on real GLM-5.3-Flash experts (TP=2 rank 0) against vLLM's current exl3_moe path
and an fp64 CPU reference; CUDA-graph capture/replay with new routing; row independence; skipped ids.

Run (GPU, guarded):  gpu_run.sh 2 -v .../exl3-dec:/w -v ~/.cache/huggingface:/hf:ro glm53-flash-sm121:local-0904-it \
                         python3 /w/tests/test_dec.py
Env: LAYER (default 20), NEXP (experts loaded, default 24).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import common as C  # noqa: E402
import glm53_exl3_dec_rt as rt  # noqa: E402

LAYER = int(os.environ.get("LAYER", "20"))
NEXP = int(os.environ.get("NEXP", "24"))
TS = [1, 2, 3, 4, 5, 8, 16, 32, 64]
dev = torch.device("cuda")
torch.manual_seed(0)
t0 = time.time()

experts = list(range(0, 288, 288 // NEXP))[:NEXP]
layer = C.load_layer(LAYER, experts, dev)
torch.cuda.synchronize()
print(f"layer {LAYER}: loaded {NEXP} experts {experts[:4]}..{experts[-1]} (rank 0 of TP=2) in {time.time()-t0:.1f}s",
      flush=True)
C.mem_report("after load")

# ---- 1. the trellis decode, bit for bit: ours vs exllamav3 reconstruct vs the fp64 reference's unpack
import exllamav3_ext  # noqa: E402
import ref_exl3_fp64 as R  # noqa: E402

for name, tr in (("gate[0]", layer.w13_trellis[0, 0]), ("up[1]", layer.w13_trellis[1, 1]),
                 ("down[2]", layer.w2_trellis[2])):
    ours = rt.dequant(tr)
    ex = torch.empty_like(ours)
    exllamav3_ext.reconstruct(ex, tr.contiguous(), 4, True, False)
    ref = R.unpack(tr.cpu(), 4)
    same_ex = torch.equal(ours.view(torch.int16), ex.view(torch.int16))
    same_ref = torch.equal(ours.cpu().view(torch.int16), ref.view(torch.int16))
    print(f"decode {name:8s} {tuple(ours.shape)}: == exllamav3 reconstruct {same_ex}, == fp64-ref unpack {same_ref}")
    assert same_ex and same_ref, "trellis decode mismatch"
    del ours, ex, ref

# ---- 2. old vs new vs fp64 on the same windows
C.build_old_state(layer)
st = rt.prepare_layer(layer, max_rows=64, slots=8)
print(f"new path: cfg_gu {st.cfg_gu} cfg_d {st.cfg_d} act_mode {st.act_mode}, scratch "
      f"{st.scratch.nbytes()/2**20:.1f} MiB", flush=True)
gen = torch.Generator().manual_seed(1234)
cases, olds, news = [], [], []
pool = list(range(NEXP))
for T in TS:
    x = C.hidden(T, gen).to(dev)
    ids = C.routing_random(T, pool, gen).to(dev)
    w = C.routing_weights(T, gen).to(dev)
    o = C.old_apply(x, ids, w, layer)
    n = rt.decode_moe(x, ids, w, layer, C.LIMIT)
    torch.cuda.synchronize()
    cases.append((x, ids, w))
    olds.append(o.cpu())
    news.append(n.cpu())
    del o, n
C.mem_report("after windows")
t1 = time.time()
refs, stats = C.reference_multi([(x.cpu(), ids.cpu(), w.cpu()) for x, ids, w in cases], layer)
print(f"fp64 reference: {time.time()-t1:.0f}s; SwiGLU limit hits: silu(g) > {C.LIMIT}: {stats['g_clamped']} / "
      f"{stats['n']}, |u| > {C.LIMIT}: {stats['u_clamped']} / {stats['n']}", flush=True)

print("\n| T | distinct | current vs fp64 max / mean rel | new vs fp64 max / mean rel | new vs current max rel |")
print("|---|---|---|---|---|")
worse = []
rows_out = []
for T, (x, ids, w), o, n, r in zip(TS, cases, olds, news, refs):
    om, oa = C.rel_err(o, r)
    nm, na = C.rel_err(n, r)
    dm, _ = C.rel_err(n, o.double())
    nd = len(set(ids.flatten().tolist()))
    line = f"| {T} | {nd} | {om:.2e} / {oa:.2e} | {nm:.2e} / {na:.2e} | {dm:.2e} |"
    print(line)
    rows_out.append(line)
    if nm > om * 1.0001 or na > oa * 1.0001:
        worse.append(T)
    assert torch.isfinite(n).all()
print(f"new no worse than current vs fp64 (max and mean) at every T: {not worse}" + (f" (worse at {worse})" if worse else ""))

# act mode 1 (limit before silu, vLLM's python loop) vs 2 (exl3_moe order): differ only where the limit bites
x, ids, w = cases[TS.index(16)]
st1 = rt.prepare_layer(layer, max_rows=64, slots=8, act_mode=rt.ACT_CLAMP_SILU)
n1 = rt.decode_moe(x, ids, w, layer, C.LIMIT).cpu()
st2 = rt.prepare_layer(layer, max_rows=64, slots=8, act_mode=rt.ACT_SILU_CLAMP)
n2 = rt.decode_moe(x, ids, w, layer, C.LIMIT).cpu()
r1, _ = C.reference_multi([(x.cpu(), ids.cpu(), w.cpu())], layer, act="clamp_silu")
print(f"act mode 1 vs 2 at T=16: max rel diff {C.rel_err(n1, n2.double())[0]:.2e}; mode 1 vs its own fp64 "
      f"(clamp, silu) {C.rel_err(n1, r1[0])[0]:.2e}")

# ---- 3. determinism, row independence, skipped ids
x, ids, w = cases[TS.index(4)]
a = rt.decode_moe(x, ids, w, layer, C.LIMIT)
b = rt.decode_moe(x, ids, w, layer, C.LIMIT)
print(f"deterministic (same inputs twice, bitwise): {torch.equal(a, b)}")
rows_ok = all(torch.equal(rt.decode_moe(x[i:i + 1], ids[i:i + 1], w[i:i + 1], layer, C.LIMIT)[0], a[i])
              for i in range(4))
x64, ids64, w64 = cases[TS.index(64)]
a64 = rt.decode_moe(x64, ids64, w64, layer, C.LIMIT)
rows_ok64 = torch.equal(rt.decode_moe(x64[5:9], ids64[5:9], w64[5:9], layer, C.LIMIT), a64[5:9])
print(f"row independence (row i alone == row i of the 4-row window; rows 5..8 of a 64-row window): "
      f"{rows_ok} / {rows_ok64}")
ids_bad = ids.clone()
ids_bad[0, 3] = -1
ids_bad[2, 0] = NEXP            # vLLM's non-local sentinel (n_local)
w_zero = w.clone()
w_zero[0, 3] = 0
w_zero[2, 0] = 0
ids_fix = ids.clone()
sk = rt.decode_moe(x, ids_bad, w, layer, C.LIMIT)
zw = rt.decode_moe(x, ids_fix, w_zero, layer, C.LIMIT)
ob = C.old_apply(x, ids_bad, w, layer)
print(f"skipped ids (-1 and n_local): == zero-weight slots bitwise {torch.equal(sk, zw)}; vs current path with the "
      f"same ids max rel {C.rel_err(sk, ob.double().cpu())[0]:.2e}")

# ---- 4. CUDA graph capture, replay with new routing
print("\nCUDA graphs:")
graph_ok = True
for T in (1, 4, 16, 64):
    xs = C.hidden(T, gen).to(dev)
    ids_s = C.routing_random(T, pool, gen).to(dev)
    ws = C.routing_weights(T, gen).to(dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            rt.decode_moe(xs, ids_s, ws, layer, C.LIMIT)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s = rt.decode_moe(xs, ids_s, ws, layer, C.LIMIT)
    res = []
    for trial in range(3):
        xs.copy_(C.hidden(T, gen))
        ids_s.copy_(C.routing_correlated(T, pool, gen) if trial == 1 else C.routing_random(T, pool, gen))
        ws.copy_(C.routing_weights(T, gen))
        g.replay()
        eager = rt.decode_moe(xs, ids_s, ws, layer, C.LIMIT)
        torch.cuda.synchronize()
        res.append(torch.equal(out_s, eager))
    ok = all(res)
    graph_ok &= ok
    print(f"  T={T:2d}: captured; 3 replays with new x / routing == eager bitwise: {res}")
    del g
print(f"graph capture OK: {graph_ok}")
C.mem_report("end")
print(f"total {time.time()-t0:.0f}s")
