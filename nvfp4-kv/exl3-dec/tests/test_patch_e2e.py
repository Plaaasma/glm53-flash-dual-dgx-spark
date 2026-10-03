"""End-to-end: the image's vllm exl3.py patched by patch_exl3_mt.py + patch_exl3_decode.py (in this throwaway
container), driven through the real Exl3MoEMethod.process_weights_after_loading and apply_exl3_experts on a layer built
from real GLM-5.3-Flash experts (TP=2 rank 0), flag off vs on.

Run (container command):
  bash -c "python3 /opt/glm53/patch_exl3_mt.py && python3 /opt/glm53/patch_exl3_decode.py && python3 /w/tests/test_patch_e2e.py"
"""
import os
import sys
import types

os.environ.setdefault("GLM53_EXL3_DEC", "0")
import torch  # noqa: E402

# the copies the patcher installed into site-packages, imported before /w goes on sys.path
import glm53_exl3_dec_rt as RT  # noqa: E402
print(f"runtime: {RT.__file__}; extension: {RT.load_ext().__file__}")
assert "dist-packages" in RT.__file__ and "dist-packages" in RT.load_ext().__file__, "not the installed copies"

from vllm.model_executor.layers.quantization import exl3 as X  # noqa: E402

src = open(X.__file__).read()
assert "# [glm53-exl3-dec]" in src and "# [glm53-exl3-mt]" in src, "exl3.py is not patched"
print(f"patched module: {X.__file__} (dec + mt markers present)")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

dev = torch.device("cuda")
NEXP = 16
layer = C.load_layer(20, list(range(0, 288, 18))[:NEXP], dev, as_module=True)
for name in ("w13_mcg", "w2_mcg"):
    assert torch.all(getattr(layer, name) == X.MCG_MARKER_SIGNED_INT32), "mcg marker"

method = X.Exl3MoEMethod.__new__(X.Exl3MoEMethod)
method.bits, method._logged, method.quant_config = 4, False, None
method.moe = types.SimpleNamespace(swiglu_limit=10.0)
gen = torch.Generator().manual_seed(11)


def window(T, pool=range(NEXP)):
    x = C.hidden(T, gen).to(dev)
    ids = C.routing_random(T, list(pool), gen).to(dev).to(torch.int32)   # vLLM's topk ids are int32
    w = C.routing_weights(T, gen).to(dev)
    return x, ids, w


calls = {"n": 0}

# ---- flag off: the upstream path, nothing prepared
os.environ["GLM53_EXL3_DEC"] = "0"
method.process_weights_after_loading(layer)
assert layer._exl3_ptrs and getattr(layer, "_glm53_dec", "missing") is None, "flag off must leave no decode state"
cases = [window(T) for T in (1, 4, 16, 64)] + [window(80)]
off = [X.apply_exl3_experts(x, ids, w, layer, limit=10.0) for x, ids, w in cases]
assert layer._exl3_last_apply == "fused"
print("flag off: process_weights_after_loading + apply_exl3_experts OK (exl3_moe)")

# ---- flag on
os.environ["GLM53_EXL3_DEC"] = "1"
method.process_weights_after_loading(layer)
dec = layer._glm53_dec
assert dec is not None, "flag on must prepare the decode state"
print(f"flag on: prepared (max_rows {dec.max_rows}, slots {dec.slots}, cfg {dec.cfg_gu} / {dec.cfg_d}, scratch "
      f"{dec.scratch.nbytes() / 2**20:.1f} MiB); views alias the layer: "
      f"{dec.w13_trellis.data_ptr() == layer.w13_trellis.data_ptr() and dec.w2_svh.data_ptr() == layer.w2_svh.data_ptr()}")
real = X._GLM53_DEC["rt"].decode_moe


def counting(*a, **k):
    calls["n"] += 1
    return real(*a, **k)


X._GLM53_DEC["rt"].decode_moe = counting
ok = True
for (x, ids, w), o in zip(cases, off):
    n0 = calls["n"]
    on = X.apply_exl3_experts(x, ids, w, layer, limit=10.0)
    took = calls["n"] - n0
    T = x.shape[0]
    if T <= dec.max_rows:
        direct = real(x, ids.to(torch.int64), w, layer, 10.0).to(x.dtype)
        same = torch.equal(on, direct)
        rel = (on.float() - o.float()).abs().max().item() / o.float().abs().max().item()
        print(f"  T={T:2d}: new path taken {took == 1}; == glm53_exl3_dec_rt.decode_moe bitwise {same}; "
              f"vs flag off max rel {rel:.2e} (bf16 output)")
        ok &= took == 1 and same and rel < 2e-2 and not X._GLM53_DEC["failed"]
    else:
        rel = (on.float() - o.float()).abs().max().item() / o.float().abs().max().item()
        print(f"  T={T:2d} (> max_rows): fell back {took == 0}; vs flag off max rel {rel:.1e} (exl3_moe's atomics "
              f"may reorder; bf16 output)")
        ok &= took == 0 and rel < 1e-2

# expert_map (EP-style global -> local, some experts not local): the new path maps exactly like map_topk_to_local
emap = torch.full((32,), -1, dtype=torch.long)
emap[:NEXP] = torch.arange(NEXP)
layer.expert_map = emap
x, ids, w = window(8, pool=range(32))
os.environ["GLM53_EXL3_DEC"] = "0"
layer._glm53_dec = None
o_map = X.apply_exl3_experts(x, ids, w, layer, limit=10.0)
layer._glm53_dec = dec
n0 = calls["n"]
on_map = X.apply_exl3_experts(x, ids, w, layer, limit=10.0)
rel = (on_map.float() - o_map.float()).abs().max().item() / o_map.float().abs().max().item()
emap_d = layer.expert_map
local = X.map_topk_to_local(ids.to(torch.long), NEXP, emap_d).view(ids.shape)
direct = real(x, local, w, layer, 10.0).to(x.dtype)
print(f"  expert_map with non-local experts: new path completed {calls['n'] - n0 == 1 and not X._GLM53_DEC['failed']}; "
      f"== decode_moe on mapped ids bitwise {torch.equal(on_map, direct)}; vs flag off max rel {rel:.2e}; "
      f"non-local slots in the window {int((local == NEXP).sum())}")
ok &= calls["n"] - n0 == 1 and rel < 2e-2 and not X._GLM53_DEC["failed"] and torch.equal(on_map, direct)
layer.expert_map = None

# CUDA graph capture of the vLLM entry point, replayed with new routing
x, ids, w = window(4)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        X.apply_exl3_experts(x, ids, w, layer, limit=10.0)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
n0 = calls["n"]
with torch.cuda.graph(g):
    out_g = X.apply_exl3_experts(x, ids, w, layer, limit=10.0)
assert calls["n"] - n0 == 1 and not X._GLM53_DEC["failed"], "the captured graph must hold the new path"
res = []
for _ in range(3):
    x2, ids2, w2 = window(4)
    x.copy_(x2); ids.copy_(ids2); w.copy_(w2)
    g.replay()
    res.append(torch.equal(out_g, X.apply_exl3_experts(x, ids, w, layer, limit=10.0)))
torch.cuda.synchronize()
print(f"  CUDA graph of apply_exl3_experts (T=4, new path captured), 3 replays with new routing == eager bitwise: "
      f"{res}; still on the new path {not X._GLM53_DEC['failed']}")
ok &= all(res) and not X._GLM53_DEC["failed"]
del g

# a launch failure falls back for the rest of the process, logged once
def boom(*a, **k):
    raise RuntimeError("injected")


X._GLM53_DEC["rt"].decode_moe = boom
x, ids, w = cases[1]
fb = X.apply_exl3_experts(x, ids, w, layer, limit=10.0)
fb2 = X.apply_exl3_experts(x, ids, w, layer, limit=10.0)


def close(a, b):
    return (a.float() - b.float()).abs().max().item() / b.float().abs().max().item() < 1e-2


print(f"  injected launch failure: fell back to exl3_moe (~ flag off {close(fb, off[1])}), disabled afterwards "
      f"{X._GLM53_DEC['failed']}, second call still upstream {close(fb2, off[1])}")
ok &= close(fb, off[1]) and close(fb2, off[1]) and X._GLM53_DEC["failed"]
C.mem_report("end")
print("E2E OK" if ok else "E2E FAILED")
sys.exit(0 if ok else 1)
