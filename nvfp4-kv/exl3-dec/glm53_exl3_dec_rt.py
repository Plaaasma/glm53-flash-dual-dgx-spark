"""glm53_exl3_dec runtime: grouped EXL3 routed experts for vLLM decode windows (GLM-5.3-Flash, 4-bit mcg).

Kernels ported from TensorFold v0.6.0 (Apache-2.0, Copyright 2026 TensorFold contributors), whose EXL3 kernels are
after ExLlamaV3 (MIT, Copyright (c) 2025 Turboderp); see csrc/ and LICENSES/.

Contract (the decode branch of vLLM's ``apply_exl3_fused_moe``):

    out = decode_moe(x2d, ids, weights, layer, limit)
        x2d     [T, D] bf16/fp16 (unit inner stride)
        ids     [T, k] int64/int32 local expert ids; ids outside [0, E) are skipped (add exactly 0)
        weights [T, k] routing weights (fp32 used as is, other dtypes upcast)
        layer   carries vLLM's w13_trellis [E,2,D/16,I/16,64] int16 (gate = 0, up = 1), w13_suh [E,2,D],
                w13_svh [E,2,I], w2_trellis [E,I/16,D/16,64], w2_suh [E,I], w2_svh [E,D] (fp16)
        limit   SwiGLU limit: act = min(silu(g), limit) * clamp(u, -limit, limit) (ExLlamaV3 exl3_moe order);
                limit <= 0 disables it
        -> out  [T, D] fp32 = sum_k weights[t, k] * expert_{ids[t, k]}(x[t])

Each distinct routed expert's trellis is read once per call (4 rows x top-8 -> ~21.5 experts instead of exl3_moe's
per-expert 16-row passes), there is no host sync and no allocation besides ``out``, and every launch's grid depends
only on T, so the call is CUDA-graph capturable. Per-layer state is views of the layer's own tensors (no copies);
one scratch per (device, shape) is shared by every layer (decode runs the layers one after another on one stream).
Call ``prepare_layer(layer)`` after the weights are loaded (before any graph capture: it allocates the scratch).

Env:
    GLM53_EXL3_DEC_MAX_ROWS   rows (tokens) the scratch holds; larger windows must take another path (default 64)
    GLM53_EXL3_DEC_SO         extension path when ``glm53_exl3_dec`` is not importable (default /opt/glm53/...so)
    GLM53_EXL3_DEC_CFG_GU     gate/up tile setting "nt,warps,splits,pf[,ld]" (default below)
    GLM53_EXL3_DEC_CFG_D      down tile setting
    GLM53_EXL3_DEC_ACT        2 = silu then limit (exl3_moe, default), 1 = limit then silu (vLLM's python loop)
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

import torch

DEFAULT_SO = "/opt/glm53/glm53_exl3_dec.so"
# (n tiles a block, warps a block, K splits, trellis rows in flight[, load path: 0 = 32-bit __ldg, 1 = 128-bit
# ld.global.nc ring]); fixed by the matrix shape, not by T, so a row's bits never depend on the other rows of its window.
DEFAULT_CFG_GU = (4, 4, 2, 2, 1)
DEFAULT_CFG_D = (4, 4, 1, 2, 1)
ACT_CLAMP_SILU = 1
ACT_SILU_CLAMP = 2

_EXT = None
_SCRATCH: dict[tuple, "Scratch"] = {}


def load_ext(path: str | None = None):
    """The compiled extension: ``import glm53_exl3_dec`` if installed, else the .so at ``path`` / env / default."""
    global _EXT
    if _EXT is not None:
        return _EXT
    try:
        import glm53_exl3_dec as mod  # noqa: F401
    except ImportError:
        so = path or os.environ.get("GLM53_EXL3_DEC_SO", DEFAULT_SO)
        spec = importlib.util.spec_from_file_location("glm53_exl3_dec", so)
        if spec is None or spec.loader is None:
            raise ImportError(f"glm53_exl3_dec: cannot load {so}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    _EXT = mod
    return mod


def env_max_rows() -> int:
    return max(1, int(os.environ.get("GLM53_EXL3_DEC_MAX_ROWS", "64")))


def _env_cfg(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    vals = tuple(int(v) for v in raw.replace(" ", "").split(","))
    if len(vals) not in (4, 5):
        raise ValueError(f"{name}={raw!r}: expected nt,warps,splits,pf[,ld]")
    return vals  # type: ignore[return-value]


class Scratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots; shared by every layer of one shape on one device."""

    def __init__(self, device, rows: int, slots: int, D: int, I: int, E: int, sk_gu: int, sk_d: int) -> None:
        P = rows * slots
        maxu = min(P, E)
        self.rows, self.slots, self.sk_gu, self.sk_d = rows, slots, sk_gu, sk_d
        self.xg = torch.empty((P, D), dtype=torch.float16, device=device)
        self.xu = torch.empty((P, D), dtype=torch.float16, device=device)
        self.xd = torch.empty((P, I), dtype=torch.float16, device=device)
        self.z = torch.empty((max(2 * sk_gu * I, sk_d * D) * P,), dtype=torch.float32, device=device)
        self.uids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        self.ucount = torch.zeros((1,), dtype=torch.int32, device=device)
        self.members = torch.full((maxu * rows,), -1, dtype=torch.int32, device=device)
        self.pick32 = torch.full((P,), -1, dtype=torch.int32, device=device)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.xg, self.xu, self.xd, self.z, self.uids, self.ucount,
                                                          self.members, self.pick32))


def get_scratch(device, rows: int, slots: int, D: int, I: int, E: int, sk_gu: int, sk_d: int) -> Scratch:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (str(dev), rows, slots, D, I, E, sk_gu, sk_d)
    sc = _SCRATCH.get(key)
    if sc is None:
        sc = Scratch(dev, rows, slots, D, I, E, sk_gu, sk_d)
        _SCRATCH[key] = sc
    return sc


def scratch_bytes_total() -> int:
    return sum(s.nbytes() for s in _SCRATCH.values())


@dataclass
class DecState:
    w13_trellis: torch.Tensor
    w13_suh: torch.Tensor
    w13_svh: torch.Tensor
    w2_trellis: torch.Tensor
    w2_suh: torch.Tensor
    w2_svh: torch.Tensor
    E: int
    D: int
    I: int
    max_rows: int
    slots: int
    cfg_gu: tuple[int, ...]
    cfg_d: tuple[int, ...]
    act_mode: int
    scratch: Scratch


def _plain(t: torch.Tensor) -> torch.Tensor:
    return t.data if isinstance(t, torch.nn.Parameter) else t


def prepare_layer(layer, max_rows: int | None = None, slots: int | None = None, cfg_gu=None, cfg_d=None,
                  act_mode: int | None = None) -> DecState:
    """Views of the layer's EXL3 tensors + the shared scratch; sets ``layer._glm53_dec``. Raises if unsupported."""
    ext = load_ext()
    w13t, w13s, w13v = _plain(layer.w13_trellis), _plain(layer.w13_suh), _plain(layer.w13_svh)
    w2t, w2s, w2v = _plain(layer.w2_trellis), _plain(layer.w2_suh), _plain(layer.w2_svh)
    if w13t.dim() != 5 or w13t.shape[1] != 2 or w13t.shape[-1] != 64 or w2t.dim() != 4 or w2t.shape[-1] != 64:
        raise ValueError(f"glm53_exl3_dec: needs 4-bit stacked trellises, got w13 {tuple(w13t.shape)} "
                         f"w2 {tuple(w2t.shape)}")
    if not w13t.is_cuda:
        raise ValueError("glm53_exl3_dec: weights are not on a CUDA device")
    E, D, I = int(w13t.shape[0]), int(w13t.shape[2]) * 16, int(w13t.shape[3]) * 16
    rows = int(max_rows or env_max_rows())
    if slots is None:
        slots = int(getattr(layer, "top_k", 0) or 0) or 8
    cfg_gu = tuple(cfg_gu or _env_cfg("GLM53_EXL3_DEC_CFG_GU", DEFAULT_CFG_GU))
    cfg_d = tuple(cfg_d or _env_cfg("GLM53_EXL3_DEC_CFG_D", DEFAULT_CFG_D))
    if act_mode is None:
        act_mode = int(os.environ.get("GLM53_EXL3_DEC_ACT", str(ACT_SILU_CLAMP)))
    cfg_gu = cfg_gu + (0,) * (5 - len(cfg_gu))
    cfg_d = cfg_d + (0,) * (5 - len(cfg_d))
    if not ext.config_ok(D, I, *cfg_gu):
        raise ValueError(f"glm53_exl3_dec: gate/up setting {cfg_gu} not compiled or does not divide K={D} N={I}")
    if not ext.config_ok(I, D, *cfg_d):
        raise ValueError(f"glm53_exl3_dec: down setting {cfg_d} not compiled or does not divide K={I} N={D}")
    sc = get_scratch(w13t.device, rows, slots, D, I, E, cfg_gu[2], cfg_d[2])
    st = DecState(w13t, w13s, w13v, w2t, w2s, w2v, E, D, I, rows, slots, cfg_gu, cfg_d, act_mode, sc)
    layer._glm53_dec = st
    return st


def decode_moe(x2d: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, layer, limit: float,
               *, stop_after: int = 0) -> torch.Tensor:
    """Routed experts of a decode window, fp32 [T, D]; see the module docstring. Raises before launching anything
    when the window does not fit the prepared scratch (the caller then takes its other path)."""
    st: DecState | None = getattr(layer, "_glm53_dec", None)
    if st is None:
        st = prepare_layer(layer)
    T = int(x2d.shape[0])
    if T == 0:
        return torch.zeros((0, st.D), dtype=torch.float32, device=x2d.device)
    if ids.numel() % T:
        raise ValueError(f"glm53_exl3_dec: {ids.numel()} ids for {T} rows")
    k = ids.numel() // T                     # ids may come flat ([T * k], map_topk_to_local) or [T, k]
    if T > st.max_rows or k > st.slots:
        raise ValueError(f"glm53_exl3_dec: window {T}x{k} exceeds the scratch {st.max_rows}x{st.slots}")
    x = x2d if x2d.stride(-1) == 1 else x2d.contiguous()
    if ids.dtype not in (torch.int64, torch.int32):
        ids = ids.to(torch.int64)
    ids = ids.reshape(T, k)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    w = weights.reshape(T, k)
    if w.dtype != torch.float32:
        w = w.float()
    if not w.is_contiguous():
        w = w.contiguous()
    out = torch.empty((T, st.D), dtype=torch.float32, device=x.device)
    sc = st.scratch
    load_ext().moe_decode(x, ids, w, out, st.w13_trellis, st.w13_suh, st.w13_svh, st.w2_trellis, st.w2_suh,
                          st.w2_svh, sc.xg, sc.xu, sc.xd, sc.z, sc.uids, sc.ucount, sc.members, sc.pick32,
                          float(limit), int(st.act_mode), list(st.cfg_gu), list(st.cfg_d), int(stop_after))
    return out


def dequant(trellis: torch.Tensor) -> torch.Tensor:
    """W_q [K, N] fp16 of one 4-bit trellis [K/16, N/16, 64] through the kernels' lane decode (tests)."""
    t = trellis.contiguous()
    out = torch.empty((t.shape[0] * 16, t.shape[1] * 16), dtype=torch.float16, device=t.device)
    load_ext().dequant(t, out)
    return out
