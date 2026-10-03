#!/usr/bin/env python3
"""Prefill routed-expert kernels for vLLM's exl3.py: glm53_exl3_fat (nvfp4-vllm/exl3-fat/).

vLLM's prefill MoE today (patch_exl3_mt.py, GLM53_EXL3_MT=1 VARIANT=8) runs one exl3_moe_mt launch per layer:
6 groups of 8 SMs walk the experts one after another, ~28 ms per MoE layer per 2,048-token chunk (48% of
prefill). glm53_exl3_fat is TensorFold's "fat" / "fast2" grouped EXL3 expert GEMMs (jayleaton's
glm53-tensorfold-spark, patches 0080 / 0170, Apache-2.0), ported to vLLM's pointer tables: a device plan sorts the
routed pairs by expert, every (expert, 64-row pass, 128-column block) is a work item claimed by the whole GPU,
each trellis tile is decoded once per 64 rows into the mma A operand, and the Hadamard / SwiGLU epilogues run in
fp32 on the accumulators. No host sync.

This patcher:
  1. installs /opt/glm53/glm53_exl3_fat.so and /opt/glm53/glm53_exl3_fat_rt.py into site-packages (if present);
  2. vllm exl3.py, apply_exl3_fused_moe (after the local-id mapping): calls with more than
     GLM53_EXL3_FAT_MIN_TOKENS tokens take the new path when GLM53_EXL3_FAT=1; anything else (decode, CUDA-graph
     capture, flag off, extension / layer init failure, a launch rejected) stays on the existing code (exl3_moe /
     the M-tiled kernel / the LinearEXL3 loop), logged once.

Runtime knobs:
  GLM53_EXL3_FAT=1                 enable (default 0: the existing path, byte for byte; read per call)
  GLM53_EXL3_FAT_MIN_TOKENS=64     calls with tokens <= this stay on the existing path (read per call)
  GLM53_EXL3_FAT_MAX_TOKENS=2048   tokens the shared scratch is sized for (max_num_batched_tokens); larger -> existing
  GLM53_EXL3_FAT_KERNEL, GLM53_EXL3_FAT_ACT, GLM53_EXL3_FAT_TICKET: see glm53_exl3_fat_rt.py

Applies on the image's exl3.py with or without patch_exl3_mt.py / patch_exl3_decode.py, in any order (its anchors
are disjoint from theirs). Fails closed: if the vLLM seams drift the file is left untouched; that is an error
(exit 1, boot stops) only when GLM53_EXL3_FAT=1 asked for the kernel, otherwise a warning. Idempotent (marker
"# [glm53-exl3-fat]").
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE_PACKAGES", "/usr/local/lib/python3.12/dist-packages"))
P = Path(os.environ.get("GLM53_EXL3_PY", str(SITE / "vllm/model_executor/layers/quantization/exl3.py")))
SO_SRC = Path(os.environ.get("GLM53_EXL3_FAT_SO", "/opt/glm53/glm53_exl3_fat.so"))
RT_SRC = Path(os.environ.get("GLM53_EXL3_FAT_RT", "/opt/glm53/glm53_exl3_fat_rt.py"))
SO_DST = SITE / "glm53_exl3_fat.so"
RT_DST = SITE / "glm53_exl3_fat_rt.py"
MARK = "# [glm53-exl3-fat]"
WANTED = os.environ.get("GLM53_EXL3_FAT", "0") == "1"

HELPER = '''
_GLM53_FAT: dict[str, Any] = {"rt": None, "tried": False, "off": False}  # [glm53-exl3-fat]


def _glm53_fat_enabled() -> bool:
    return os.environ.get("GLM53_EXL3_FAT", "0") == "1" and not _GLM53_FAT["off"]


def _glm53_fat_min_tokens() -> int:
    return int(os.environ.get("GLM53_EXL3_FAT_MIN_TOKENS", "64"))


def _glm53_fat_prefill(x2d, local, weights, layer, limit, topk):
    """Prefill-size apply on glm53_exl3_fat. Returns fp32 [T, hidden], or None (nothing launched): existing path."""
    st = _GLM53_FAT
    if not st["tried"]:
        st["tried"] = True
        try:
            import glm53_exl3_fat_rt as rt

            if rt.load_ext() is None:
                raise RuntimeError("glm53_exl3_fat extension did not load")
            cfg = rt.config()
            st["rt"] = rt
            logger.info(
                "[glm53-exl3-fat] prefill experts on glm53_exl3_fat for calls > %d tokens (scratch for %d tokens)",
                _glm53_fat_min_tokens(), cfg["max_tokens"],
            )
        except Exception as exc:  # noqa: BLE001
            st["off"] = True
            logger.warning("[glm53-exl3-fat] unavailable (%s); prefill stays on the existing path", exc)
            return None
    rt = st["rt"]
    if rt is None or torch.cuda.is_current_stream_capturing():
        return None
    try:
        return rt.apply_local(x2d, local, weights, layer, limit, topk)
    except Exception as exc:  # noqa: BLE001  (argument checks raise before any GPU work)
        st["off"] = True
        logger.error("[glm53-exl3-fat] launch rejected (%s); existing path for the rest of this process", exc)
        return None


'''

ANCHOR = "def apply_exl3_fused_moe(\n"

SEAM_OLD = """    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
"""
SEAM_NEW = SEAM_OLD + """    if tokens > _glm53_fat_min_tokens() and _glm53_fat_enabled():  # [glm53-exl3-fat] prefill-size calls
        _fat_out = _glm53_fat_prefill(x2d, local, weights, layer, limit, topk)
        if _fat_out is not None:
            return _fat_out
"""

NEEDED = (
    "    tokens, hidden = x2d.shape\n    n_exp = len(inners)\n",
    "def map_topk_to_local(",
    "logger = init_logger(__name__)",
    "from typing import TYPE_CHECKING, Any",
    "\nimport os\n",
    "\nimport torch\n",
)


def install(src: Path, dst: Path) -> None:
    if not src.is_file():
        print(f"{src}: not present - not installed (GLM53_EXL3_FAT=1 will log a warning and keep the existing path)")
        return
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"{dst.name}: already installed - skipping")
        return
    shutil.copyfile(src, dst)
    print(f"installed {dst}")


def drift(msg: str) -> int:
    if WANTED:
        raise SystemExit(f"{P}: {msg} (vLLM exl3.py drift; GLM53_EXL3_FAT=1 cannot be honored)")
    print(f"WARNING {P}: {msg} (vLLM exl3.py drift) - not patched; GLM53_EXL3_FAT is off, continuing")
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
    if text.count(ANCHOR) != 1:
        return drift(f"helper anchor count {text.count(ANCHOR)} != 1")
    if text.count(SEAM_OLD) != 1:
        return drift(f"apply seam count {text.count(SEAM_OLD)} != 1")
    for needed in NEEDED:
        if needed not in text:
            return drift(f"expected {needed.strip()!r}")
    fn_at = text.index(ANCHOR)
    seam_at = text.index(SEAM_OLD)
    next_def = text.find("\ndef ", fn_at + len(ANCHOR))
    if not fn_at < seam_at < (next_def if next_def >= 0 else len(text)):
        return drift("apply seam is not inside apply_exl3_fused_moe")
    text = text.replace(ANCHOR, HELPER + ANCHOR, 1)
    text = text.replace(SEAM_OLD, SEAM_NEW, 1)
    P.write_text(text)
    print(f"patched {P.name}: glm53_exl3_fat prefill dispatch (GLM53_EXL3_FAT, > GLM53_EXL3_FAT_MIN_TOKENS tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
