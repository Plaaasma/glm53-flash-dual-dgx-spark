#!/usr/bin/env python3
"""Grouped EXL3 routed-expert decode kernel (glm53_exl3_dec) for vLLM's exl3.py.

vLLM's decode MoE today is one exllamav3 exl3_moe launch per layer: 6 groups of 8 SMs walk the routed experts
and decode each expert's whole trellis for its <= 16 rows, ~1.2 ms per layer for a 4-row MTP verify window
(~110-160 GB/s). glm53_exl3_dec (TensorFold's grouped EXL3 kernel, ported; nvfp4-vllm/exl3-dec/) reads each
DISTINCT routed expert of the window once, spread over the whole GPU, with no host sync (CUDA-graph safe), fp32
SwiGLU in exl3_moe's order (silu, then the limit) and no atomics on the output.

This patcher:
  1. installs /opt/glm53/glm53_exl3_dec.so and /opt/glm53/glm53_exl3_dec_rt.py into site-packages (if present);
  2. vllm exl3.py, build_exl3_fused_state (end): prepares the layer's decode state (views of the layer's own
     w13_*/w2_* tensors, no copies) + one shared scratch, when GLM53_EXL3_DEC=1;
  3. vllm exl3.py, apply_exl3_fused_moe (top): windows of <= GLM53_EXL3_DEC_MAX_ROWS tokens take the new path;
     anything else (prefill, flag off, init or launch failure) stays on the existing code, logged once.

Runtime knobs (read once at weight load; the default is the upstream path, byte for byte):
  GLM53_EXL3_DEC=1               enable (default 0)
  GLM53_EXL3_DEC_MAX_ROWS=64     largest window (tokens) on the new path; sizes the shared scratch (~17 MiB at 64)
  GLM53_EXL3_DEC_CFG_GU / _CFG_D tile settings "nt,warps,splits,pf,ld" (defaults in glm53_exl3_dec_rt.py)
  GLM53_EXL3_DEC_ACT=2           2 = silu then limit (exl3_moe), 1 = limit then silu (vLLM's python loop)

Fails closed: if the vLLM seams drift the file is left untouched; that is an error (exit 1, boot stops) only when
GLM53_EXL3_DEC=1 asked for the kernel, otherwise a warning. Idempotent (marker "# [glm53-exl3-dec]").
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE_PACKAGES", "/usr/local/lib/python3.12/dist-packages"))
P = Path(os.environ.get("GLM53_EXL3_PY", str(SITE / "vllm/model_executor/layers/quantization/exl3.py")))
SO_SRC = Path(os.environ.get("GLM53_EXL3_DEC_SO", "/opt/glm53/glm53_exl3_dec.so"))
RT_SRC = Path(os.environ.get("GLM53_EXL3_DEC_RT", "/opt/glm53/glm53_exl3_dec_rt.py"))
SO_DST = SITE / "glm53_exl3_dec.so"
RT_DST = SITE / "glm53_exl3_dec_rt.py"
MARK = "# [glm53-exl3-dec]"
WANTED = os.environ.get("GLM53_EXL3_DEC", "0") == "1"

HELPER = '''
_GLM53_DEC: dict[str, Any] = {"rt": None, "tried": False, "failed": False, "logged": False}  # [glm53-exl3-dec]


def _glm53_dec_rt():
    st = _GLM53_DEC
    if not st["tried"]:
        st["tried"] = True
        try:
            import glm53_exl3_dec_rt as rt

            rt.load_ext()
            st["rt"] = rt
        except Exception as exc:  # noqa: BLE001
            logger.warning("[glm53-exl3-dec] runtime unavailable (%s); decode MoE stays on exl3_moe", exc)
    return st["rt"]


def _glm53_dec_prepare(layer: torch.nn.Module) -> None:
    """Per-layer decode state (views, no copies) + the shared scratch; None (= existing path) on any failure."""
    layer._glm53_dec = None
    if os.environ.get("GLM53_EXL3_DEC", "0") != "1" or _GLM53_DEC["failed"]:
        return
    rt = _glm53_dec_rt()
    if rt is None:
        return
    try:
        dec = rt.prepare_layer(layer)
    except Exception as exc:  # noqa: BLE001
        if not _GLM53_DEC["logged"]:
            _GLM53_DEC["logged"] = True
            logger.warning("[glm53-exl3-dec] layer not supported (%s); decode MoE stays on exl3_moe", exc)
        return
    layer._glm53_dec = dec
    if not _GLM53_DEC["logged"]:
        _GLM53_DEC["logged"] = True
        logger.info(
            "[glm53-exl3-dec] grouped decode MoE on: windows <= %d tokens x top-%d, experts %d, cfg gate/up %s "
            "down %s, act mode %d, scratch %.1f MiB per device (shared by all layers)",
            dec.max_rows, dec.slots, dec.E, dec.cfg_gu, dec.cfg_d, dec.act_mode, dec.scratch.nbytes() / 2**20,
        )


def _glm53_dec_apply(x2d, ids, weights, layer, n_exp, expert_map, limit):
    """The decode window on glm53_exl3_dec, or None for the existing path (nothing launched)."""
    dec = getattr(layer, "_glm53_dec", None)
    if dec is None or _GLM53_DEC["failed"]:
        return None
    tokens = int(x2d.shape[0])
    if tokens == 0 or tokens > dec.max_rows or int(ids.shape[-1]) > dec.slots:
        return None
    try:
        local = ids if expert_map is None else map_topk_to_local(ids, n_exp, expert_map).view(ids.shape)
        return _GLM53_DEC["rt"].decode_moe(x2d, local, weights, layer, limit)
    except Exception as exc:  # noqa: BLE001  (argument checks raise before any GPU work)
        _GLM53_DEC["failed"] = True
        logger.error("[glm53-exl3-dec] decode launch rejected (%s); exl3_moe for the rest of this process", exc)
        return None


'''

HELPER_ANCHOR = "def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:\n"

BUILD_OLD = """    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)
"""
BUILD_NEW = BUILD_OLD + "    _glm53_dec_prepare(layer)  # [glm53-exl3-dec]\n"

APPLY_OLD = """    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
"""
APPLY_NEW = """    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    _dec_out = _glm53_dec_apply(x2d, ids, weights, layer, n_exp, expert_map, limit)  # [glm53-exl3-dec]
    if _dec_out is not None:
        return _dec_out

    local = map_topk_to_local(ids, n_exp, expert_map)
"""

NEEDED = (
    "def apply_exl3_fused_moe(\n",
    "    tokens, hidden = x2d.shape\n    n_exp = len(inners)\n",
    "def map_topk_to_local(",
    "logger = init_logger(__name__)",
    "from typing import TYPE_CHECKING, Any",
    "\nimport os\n",
)


def install(src: Path, dst: Path) -> None:
    if not src.is_file():
        print(f"{src}: not present - not installed (GLM53_EXL3_DEC=1 will log a warning and keep exl3_moe)")
        return
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"{dst.name}: already installed - skipping")
        return
    shutil.copyfile(src, dst)
    print(f"installed {dst}")


def drift(msg: str) -> int:
    if WANTED:
        raise SystemExit(f"{P}: {msg} (vLLM exl3.py drift; GLM53_EXL3_DEC=1 cannot be honored)")
    print(f"WARNING {P}: {msg} (vLLM exl3.py drift) - not patched; GLM53_EXL3_DEC is off, continuing")
    return 0


def main() -> int:
    install(SO_SRC, SO_DST)
    install(RT_SRC, RT_DST)
    if not P.is_file():
        return drift("missing")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present - skipping")
        return 0
    for anchor, what in ((HELPER_ANCHOR, "helper anchor"), (BUILD_OLD, "build_exl3_fused_state seam"),
                         (APPLY_OLD, "apply_exl3_fused_moe seam")):
        if text.count(anchor) != 1:
            return drift(f"{what} not unique ({text.count(anchor)})")
    for needed in NEEDED:
        if needed not in text:
            return drift(f"expected {needed.strip()!r}")
    if text.index(BUILD_OLD) < text.index(HELPER_ANCHOR) or text.index(APPLY_OLD) < text.index(
            "def apply_exl3_fused_moe(\n"):
        return drift("seams are not inside their functions")
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    text = text.replace(BUILD_OLD, BUILD_NEW, 1)
    text = text.replace(APPLY_OLD, APPLY_NEW, 1)
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"patched {P.name}: grouped decode MoE dispatch (GLM53_EXL3_DEC={'1' if WANTED else '0'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
