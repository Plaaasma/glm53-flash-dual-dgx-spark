"""Runtime driver for glm53_exl3_fat: EXL3 routed-expert PREFILL for vLLM's exl3.py (GLM-5.3-Flash, DGX Spark).

Kernels: glm53_exl3_fat.cu (ported from jayleaton's glm53-tensorfold-spark patches 0080 / 0170, Apache-2.0; EXL3 format
of turboderp's ExLlamaV3, MIT). One call per MoE layer runs four launches and no host sync:

    plan     counting sort of the routed (token, slot) pairs by expert + per-expert pass prefix sums (device)
    rot      Xg[j] = fp16(H(x[row] * suh_gate[e]) / sqrt(128)) for every pair, expert-sorted
    gate/up  Xd[j] = fp16(H(act(H(Xg Wg) svh_g, H(Xg Wu) svh_u) * suh_d) / sqrt(128)), act = limited SwiGLU (fp32)
    down     Y[j] = fp16(H(Xd Wd) svh_d / sqrt(128)), written over Xg (dead by then)
    combine  out[t] = sum_k w[t, k] * Y[j(t, k)] in fp32, slots in order (deterministic; every element written)

Contract (same as apply_exl3_fused_moe's prefill branch): (x2d [T, hidden] bf16, ids [T, topk] int64, weights
[T, topk], layer, limit) -> fp32 [T, hidden]; or None when this layer / call cannot take the path (the caller then
runs the existing path; nothing has been launched).

Env knobs (read once, at first use):
  GLM53_EXL3_FAT_MAX_TOKENS  tokens the shared scratch is sized for (default 2048 = max_num_batched_tokens); larger
                             calls return None (existing path)
  GLM53_EXL3_FAT_KERNEL      gate/up,down kernel ids or names (default "fast2-s4,fast2-cb2s4"; see kernel_names())
  GLM53_EXL3_FAT_ACT         2 = exl3_moe order silu(g) then min(., limit) (default, matches the current kernels);
                             1 = vLLM python-loop order silu(min(g, limit))
  GLM53_EXL3_FAT_TICKET      1 = dynamic item claiming (default), 0 = static stride
  GLM53_EXL3_FAT_COMBINE     1 = fp16 per-pair rows + combine kernel (default), 0 = fp32 atomics into the output
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger("vllm.glm53_exl3_fat")

_ST: dict[str, Any] = {"ext": None, "tried": False, "scratch": {}, "cfg": None, "logged": set()}


def _log_once(key: str, level: int, msg: str, *args: Any) -> None:
    if key in _ST["logged"]:
        return
    _ST["logged"].add(key)
    logger.log(level, msg, *args)


def load_ext():
    """Import the extension once; None when unavailable."""
    if not _ST["tried"]:
        _ST["tried"] = True
        try:
            import glm53_exl3_fat as ext  # noqa: PLC0415

            _ST["ext"] = ext
        except Exception as exc:  # noqa: BLE001
            _log_once("ext", logging.WARNING, "[glm53-exl3-fat] extension unavailable (%s); prefill stays on the "
                      "existing path", exc)
    return _ST["ext"]


def _kernel_id(spec: str, names: list[str]) -> int:
    spec = spec.strip()
    if spec.isdigit():
        i = int(spec)
        if not 0 <= i < len(names):
            raise ValueError(f"kernel id {i} out of range")
        return i
    for i, n in enumerate(names):
        if n.split()[0] == spec:
            return i
    raise ValueError(f"unknown kernel {spec!r} (have {names})")


def config() -> dict[str, Any]:
    cfg = _ST["cfg"]
    if cfg is None:
        ext = load_ext()
        spec = os.environ.get("GLM53_EXL3_FAT_KERNEL", "fast2-s4,fast2-cb2s4").split(",")
        if len(spec) == 1:
            spec = spec * 2
        cfg = {
            "max_tokens": max(1, int(os.environ.get("GLM53_EXL3_FAT_MAX_TOKENS", "2048"))),
            "kern_gu": _kernel_id(spec[0], list(ext.kernel_names(False))) if ext else 0,
            "kern_dn": _kernel_id(spec[1], list(ext.kernel_names(True))) if ext else 0,
            "act": int(os.environ.get("GLM53_EXL3_FAT_ACT", "2")),
            "ticket": os.environ.get("GLM53_EXL3_FAT_TICKET", "1") != "0",
            "combine": os.environ.get("GLM53_EXL3_FAT_COMBINE", "1") != "0",
        }
        if cfg["act"] not in (1, 2):
            raise ValueError("GLM53_EXL3_FAT_ACT must be 1 or 2")
        _ST["cfg"] = cfg
    return cfg


def prepare(layer: torch.nn.Module) -> dict[str, Any] | None:
    """Per-layer state (cached on the layer): pointer tables and shapes. None if the layer cannot use the path."""
    st = layer.__dict__.get("_glm53_fat")
    if st is not None:
        return st or None
    ext = load_ext()
    reason = None
    ptrs = getattr(layer, "_exl3_ptrs", None)
    hidden = int(getattr(layer, "_exl3_hidden_size", 0) or 0)
    inter = int(getattr(layer, "_exl3_intermediate_local", 0) or 0)
    k = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 0)) or 0)
    if ext is None:
        reason = "extension unavailable"
    elif not ptrs:
        reason = "no EXL3 pointer tables"
    elif k != 4:
        reason = f"{k}-bit experts (only 4-bit mcg is built)"
    elif hidden % 128 or inter % 128 or hidden <= 0 or inter <= 0:
        reason = f"hidden {hidden} / intermediate {inter} not multiples of 128"
    else:
        n_exp = int(ptrs["gate_trellis"].numel())
        if n_exp > 1024:
            reason = f"{n_exp} experts > 1024"
        elif not ext.shared_suh(ptrs["gate_suh"], ptrs["up_suh"], hidden):
            reason = "gate and up input sign vectors differ (one rotated input cannot serve both)"
    if reason is not None:
        layer.__dict__["_glm53_fat"] = False
        _log_once("prep:" + reason, logging.WARNING, "[glm53-exl3-fat] layer not eligible (%s); existing path", reason)
        return None
    st = {
        "hidden": hidden,
        "inter": inter,
        "E": int(ptrs["gate_trellis"].numel()),
        "tables": tuple(ptrs[n] for n in ("gate_trellis", "up_trellis", "down_trellis", "gate_suh", "gate_svh",
                                          "up_svh", "down_suh", "down_svh")),
    }
    layer.__dict__["_glm53_fat"] = st
    return st


def scratch(device: torch.device, hidden: int, inter: int, n_exp: int, topk: int):
    """Shared buffers sized for GLM53_EXL3_FAT_MAX_TOKENS tokens (MoE layers run one at a time)."""
    cfg = config()
    pairs = cfg["max_tokens"] * topk
    key = (str(device), hidden, inter, n_exp, topk)
    s = _ST["scratch"].get(key)
    if s is None:
        ext = load_ext()
        s = (
            torch.empty((pairs, hidden), dtype=torch.float16, device=device),
            torch.empty((pairs, inter), dtype=torch.float16, device=device),
            torch.empty((int(ext.meta_ints(n_exp, pairs)),), dtype=torch.int32, device=device),
        )
        _ST["scratch"][key] = s
        logger.info(
            "[glm53-exl3-fat] scratch on %s for %d tokens x top%d: %.1f MiB (kernels %s / %s, act %d, ticket %s, "
            "combine %s)", device, cfg["max_tokens"], topk, sum(t.numel() * t.element_size() for t in s) / 2**20,
            ext.kernel_names(False)[cfg["kern_gu"]], ext.kernel_names(True)[cfg["kern_dn"]], cfg["act"], cfg["ticket"],
            cfg["combine"],
        )
    return s


def scratch_bytes(max_tokens: int = 2048, topk: int = 8, hidden: int = 4096, inter: int = 1024,
                  n_exp: int = 288) -> int:
    pairs = max_tokens * topk
    return pairs * hidden * 2 + pairs * inter * 2 + (3 * (n_exp + 1) + 8 + 2 * pairs) * 4


def apply_local(x2d: torch.Tensor, local: torch.Tensor, weights: torch.Tensor, layer: torch.nn.Module, limit: float,
                topk: int, *, kern_gu: int | None = None, kern_dn: int | None = None, out: torch.Tensor | None = None,
                stage_mask: int = 31, probe: int = 0, combine: bool | None = None) -> torch.Tensor | None:
    """`local`: [T * topk] (or [T, topk]) int64 LOCAL expert ids, anything outside [0, E) skipped (map_topk_to_local's
    sentinel). Returns fp32 [T, hidden] or None (nothing launched)."""
    st = prepare(layer)
    if st is None:
        return None
    tokens = int(x2d.shape[0])
    cfg = config()
    if tokens > cfg["max_tokens"] or x2d.dtype not in (torch.bfloat16, torch.float16) or x2d.stride(-1) != 1:
        return None
    ext = _ST["ext"]
    xg, xd, meta = scratch(x2d.device, st["hidden"], st["inter"], st["E"], topk)
    if local.dtype != torch.long:
        local = local.long()
    local = local.reshape(-1).contiguous()
    w = weights.reshape(-1)
    if w.dtype != torch.float32:
        w = w.float()
    w = w.contiguous()
    combine = cfg["combine"] if combine is None else bool(combine)
    if out is None:
        alloc = torch.empty if combine else torch.zeros     # combine writes every element
        out = alloc(tokens, st["hidden"], dtype=torch.float32, device=x2d.device)
    ext.moe_prefill(
        x2d, local, w, out, xg, xd, meta, *st["tables"], int(topk), float(limit), cfg["act"],
        cfg["kern_gu"] if kern_gu is None else int(kern_gu), cfg["kern_dn"] if kern_dn is None else int(kern_dn),
        int(stage_mask), bool(cfg["ticket"]), int(probe), combine,
    )
    return out


def to_local(ids: torch.Tensor, n_local: int, expert_map: torch.Tensor | None) -> torch.Tensor:
    """vLLM's map_topk_to_local: global ids -> local ids, invalid / non-local -> n_local (skipped)."""
    flat = ids.reshape(-1).long()
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    expert_map = expert_map.to(device=flat.device, dtype=torch.long)
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


def apply(x2d: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, layer: torch.nn.Module, limit: float,
          expert_map: torch.Tensor | None = None) -> torch.Tensor | None:
    """apply_exl3_fused_moe's prefill contract: (x2d [T, hidden], ids [T, topk] global, weights [T, topk], layer,
    limit) -> fp32 [T, hidden], or None."""
    st = prepare(layer)
    if st is None:
        return None
    topk = int(ids.shape[-1])
    return apply_local(x2d, to_local(ids, st["E"], expert_map), weights, layer, limit, topk)
