"""Shared test helpers: real GLM-5.3-Flash EXL3 experts (TP=2 rank-0 shards, exactly as vLLM loads them), the
current vLLM decode path (exllamav3 exl3_moe with vLLM's own prologue), and an fp64 CPU reference.

vLLM's functions are taken from the image's exl3.py source by name (ast), so the tests run the very code vLLM
runs without importing vllm (keeps the test container's host RAM small).
"""
from __future__ import annotations

import ast
import json
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SNAP = os.environ.get(
    "GLM53_SNAPSHOT",
    "/hf/hub/models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3/snapshots/1fac3dbe6269a399ce5378261cdf250fde706180",
)
VLLM_EXL3 = os.environ.get(
    "VLLM_EXL3_PY", "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py"
)
TP_RANK, TP_SIZE = 0, 2
LIMIT = 10.0          # config.json swiglu_limit
TOPK = 8
BYTES_PER_EXPERT = 3 * 4096 * 1024 // 2   # gate + up + down trellis bytes on one rank (4 bits)


def vllm_funcs(names=("_narrow_tp", "shard_exl3_col", "shard_exl3_row", "_exl3_moe_accepts_num_active",
                      "map_topk_to_local", "_exl3_moe_launch", "apply_exl3_fused_moe")) -> types.SimpleNamespace:
    """The named top-level functions (and module constants) of vLLM's exl3.py, executed in a small namespace."""
    src = open(VLLM_EXL3).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(tgt, ast.Name) and tgt.id.isupper() and isinstance(node.value, (ast.Constant, ast.Tuple)):
                keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"torch": torch, "os": os, "Any": object, "F": torch.nn.functional, "__name__": "vllm_exl3_extract"}
    import __future__

    exec(compile(mod, VLLM_EXL3, "exec", flags=__future__.annotations.compiler_flag), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        raise RuntimeError(f"{VLLM_EXL3}: missing {missing}")
    return types.SimpleNamespace(**ns)


def _index():
    return json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]


def load_layer(layer_idx: int, experts: list[int], device="cuda", as_module=False):
    """Stacked vLLM-layout EXL3 tensors of ``experts`` of decoder layer ``layer_idx`` (rank 0 of TP=2)."""
    from safetensors import safe_open

    vf = vllm_funcs()
    wm = _index()
    E = len(experts)
    pre = f"model.language_model.layers.{layer_idx}.mlp.experts."
    t = {
        "w13_trellis": torch.empty((E, 2, 256, 64, 64), dtype=torch.int16, device=device),
        "w13_suh": torch.empty((E, 2, 4096), dtype=torch.float16, device=device),
        "w13_svh": torch.empty((E, 2, 1024), dtype=torch.float16, device=device),
        "w13_mcg": torch.empty((E, 2, 1), dtype=torch.int32, device=device),
        "w2_trellis": torch.empty((E, 64, 256, 64), dtype=torch.int16, device=device),
        "w2_suh": torch.empty((E, 1024), dtype=torch.float16, device=device),
        "w2_svh": torch.empty((E, 4096), dtype=torch.float16, device=device),
        "w2_mcg": torch.empty((E, 1), dtype=torch.int32, device=device),
    }
    handles = {}

    def get(name):
        f = wm[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(SNAP, f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    for i, e in enumerate(experts):
        for proj, j in (("gate_proj", 0), ("up_proj", 1)):
            for suf in ("trellis", "suh", "svh", "mcg"):
                v = vf.shard_exl3_col(get(f"{pre}{e}.{proj}.{suf}").contiguous(), suf, TP_RANK, TP_SIZE)
                t[f"w13_{suf}"][i, j].copy_(v.reshape(t[f"w13_{suf}"][i, j].shape))
        for suf in ("trellis", "suh", "svh", "mcg"):
            v = vf.shard_exl3_row(get(f"{pre}{e}.down_proj.{suf}").contiguous(), suf, TP_RANK, TP_SIZE)
            t[f"w2_{suf}"][i].copy_(v.reshape(t[f"w2_{suf}"][i].shape))
    handles.clear()
    if as_module:
        layer = torch.nn.Module()
        for k, v in t.items():
            layer.register_parameter(k, torch.nn.Parameter(v, requires_grad=False))
    else:
        layer = types.SimpleNamespace(**t)
    layer._exl3_hidden_size = 4096
    layer._exl3_intermediate_local = 1024
    layer._exl3_bits = 4
    layer._exl3_k = 4
    layer.experts_loaded = list(experts)
    return layer


def build_old_state(layer, rows: int = 128):
    """What vLLM's build_exl3_fused_state sets up for exl3_moe: pointer tables (the LinearEXL3 handles alias the
    stacked tensors' slices) and the shared fused temps."""
    import exllamav3_ext

    E = layer.w13_trellis.shape[0]
    dev = layer.w13_trellis.device

    def ptrs(t):
        return torch.tensor([int(t[e].data_ptr()) for e in range(E)], dtype=torch.int64, device=dev)

    layer._exl3_ptrs = {
        "gate_trellis": ptrs(layer.w13_trellis[:, 0]), "gate_suh": ptrs(layer.w13_suh[:, 0]),
        "gate_svh": ptrs(layer.w13_svh[:, 0]), "up_trellis": ptrs(layer.w13_trellis[:, 1]),
        "up_suh": ptrs(layer.w13_suh[:, 1]), "up_svh": ptrs(layer.w13_svh[:, 1]),
        "down_trellis": ptrs(layer.w2_trellis), "down_suh": ptrs(layer.w2_suh), "down_svh": ptrs(layer.w2_svh),
    }
    conc = max(1, int(exllamav3_ext.exl3_moe_max_concurrency(dev.index or 0)))
    layer._exl3_fused_temps = (
        torch.empty((conc, rows, 4096), dtype=torch.float16, device=dev),
        torch.empty((conc, rows, 4096), dtype=torch.float16, device=dev),
        torch.empty((conc, rows, 1024), dtype=torch.float16, device=dev),
        torch.empty((conc, rows, 1024), dtype=torch.float16, device=dev),
    )
    layer._exl3_fused_concurrency = conc
    return layer


_VF = None


def old_apply(x2d, ids, weights, layer, limit=LIMIT):
    """vLLM's current decode path: apply_exl3_fused_moe (prologue + one exl3_moe launch), fp32 [T, D]."""
    global _VF
    if _VF is None:
        _VF = vllm_funcs()
    E = layer.w13_trellis.shape[0]
    return _VF.apply_exl3_fused_moe(x2d, ids, weights, layer, [None] * E, None, limit)


# ---------------------------------------------------------------------------------------------------------- fp64
def expert_fp64(layer, i: int):
    """fp64 CPU weights (W [K, N] = diag(suh) H W_q H diag(svh)) of loaded expert slot i: gate, up, down."""
    import ref_exl3_fp64 as R

    def deq(tr, suh, svh):
        return R.dequantize(tr.cpu(), suh.cpu(), svh.cpu(), bits=4)

    g = deq(layer.w13_trellis[i, 0], layer.w13_suh[i, 0], layer.w13_svh[i, 0])
    u = deq(layer.w13_trellis[i, 1], layer.w13_suh[i, 1], layer.w13_svh[i, 1])
    d = deq(layer.w2_trellis[i], layer.w2_suh[i], layer.w2_svh[i])
    return g, u, d


def reference_multi(cases, layer, limit=LIMIT, act="silu_clamp"):
    """fp64 CPU references of several windows at once, streaming one expert's fp64 weights at a time (~100 MB):
    out[t] = sum_k w[t,k] * down(act(gate(x_t), up(x_t))), act in exl3_moe's order (silu, then limit) by default.
    ``cases`` = [(x2d, ids, weights)]; ids are loaded-slot (local) ids."""
    prepped = [(x.detach().double().cpu(), ids.cpu(), w.detach().double().cpu()) for x, ids, w in cases]
    outs = [torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float64) for x, _, _ in prepped]
    stats = {"g_clamped": 0, "u_clamped": 0, "n": 0}
    used = sorted(set(int(v) for _, ids, _ in prepped for v in ids.flatten().tolist() if int(v) >= 0))
    for e in used:
        g_w, u_w, d_w = expert_fp64(layer, e)
        for (x, ids, w), out in zip(prepped, outs):
            rows, ks = (ids == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            xe = x[rows]
            g = xe @ g_w
            u = xe @ u_w
            s = g / (1.0 + torch.exp(-g))
            if limit > 0:
                if act == "silu_clamp":
                    stats["g_clamped"] += int((s > limit).sum())
                    s = s.clamp(max=limit)
                else:
                    gc = g.clamp(max=limit)
                    stats["g_clamped"] += int((g > limit).sum())
                    s = gc / (1.0 + torch.exp(-gc))
                stats["u_clamped"] += int((u.abs() > limit).sum())
                u = u.clamp(-limit, limit)
            stats["n"] += g.numel()
            y = (s * u) @ d_w
            out.index_add_(0, rows, y * w[rows, ks].unsqueeze(-1))
        del g_w, u_w, d_w
    return outs, stats


def rel_err(a, ref):
    a = a.detach().double().cpu()
    d = (a - ref).abs()
    return float(d.max() / ref.abs().max()), float(d.mean() / ref.abs().mean())


# ------------------------------------------------------------------------------------------------------ routing
def routing_random(T, pool, gen, k=TOPK):
    """[T, k] int64 ids: k distinct experts a row drawn uniformly from ``pool`` (local ids)."""
    pool = torch.as_tensor(pool)
    rows = [pool[torch.randperm(len(pool), generator=gen)[:k]] for _ in range(T)]
    return torch.stack(rows).to(torch.int64)


def routing_correlated(T, pool, gen, k=TOPK, keep=None):
    """Each row keeps ``keep`` (default k/2) of the previous row's experts and draws the rest anew."""
    keep = k // 2 if keep is None else keep
    pool = torch.as_tensor(pool)
    rows = [pool[torch.randperm(len(pool), generator=gen)[:k]]]
    for _ in range(1, T):
        prev = rows[-1]
        kept = prev[torch.randperm(k, generator=gen)[:keep]]
        rest = pool[~torch.isin(pool, kept)]
        new = rest[torch.randperm(len(rest), generator=gen)[: k - keep]]
        rows.append(torch.cat([kept, new])[torch.randperm(k, generator=gen)])
    return torch.stack(rows).to(torch.int64)


def routing_weights(T, gen, k=TOPK):
    s = torch.rand((T, k), generator=gen) * 0.9 + 0.1
    return (s / s.sum(-1, keepdim=True)).float()


def hidden(T, gen, scales=(0.5, 1.0, 2.0, 4.0)):
    """bf16 rows ~ N(0, s^2) with s cycling through ``scales`` (the larger ones push SwiGLU into its limit)."""
    x = torch.randn((T, 4096), generator=gen)
    s = torch.tensor([scales[i % len(scales)] for i in range(T)]).unsqueeze(-1)
    return (x * s).to(torch.bfloat16)


def mem_report(tag=""):
    a = torch.cuda.memory_allocated() / 2**20
    r = torch.cuda.memory_reserved() / 2**20
    m = torch.cuda.max_memory_allocated() / 2**20
    av = 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                av = int(line.split()[1]) / 2**20
    except OSError:
        pass
    print(f"[mem{(' ' + tag) if tag else ''}] torch alloc {a:.0f} MiB reserved {r:.0f} MiB peak {m:.0f} MiB | "
          f"host MemAvailable {av:.1f} GiB", flush=True)
