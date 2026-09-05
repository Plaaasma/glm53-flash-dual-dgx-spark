#!/usr/bin/env python3
"""M-tiled EXL3 fused-MoE kernel for prefill (glm53_exl3_mt).

Upstream exllamav3's exl3_moe kernel processes 16 rows per pass and re-decodes the whole trellis weight for
every pass, so during prefill an expert with R rows decodes its weights ceil(R/16) times and experts with
more rows than the fused temps fall to a per-expert reconstruct + cuBLAS loop. glm53_exl3_mt is the same
kernel with 64-row M tiles (each dequantized fragment feeds 4 MMAs) plus large temps, measured 1.8-2.5x
faster per MoE layer on skewed prefill routing (nvfp4-kv/exl3-mt/README.md).

This patcher:
  1. installs the prebuilt extension (/opt/glm53/glm53_exl3_mt.so -> site-packages) if present;
  2. adds a prefill-only dispatch to vllm's exl3.py: when a step carries more tokens than the fused temps
     hold (never the CUDA-graph decode path), run exl3_moe_mt instead of upstream fused + fat-expert loop.

Runtime knobs (read per call, so the default is fail-closed):
  GLM53_EXL3_MT=1            enable (default 0: upstream behaviour, byte-for-byte)
  GLM53_EXL3_MT_VARIANT=5    kernel shape (5 = m64_k16_n256, the measured best)
  GLM53_EXL3_MT_TEMP_ROWS    rows per expert the MT temps hold (default 1024; hotter experts use the upstream loop)
  GLM53_EXL3_MT_MIN_ROWS     only use MT when the hottest expert has at least this many rows (default 32)

Fail closed if the vLLM seams drift.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE_PACKAGES", "/usr/local/lib/python3.12/dist-packages"))
P = Path(os.environ.get("GLM53_EXL3_PY", str(SITE / "vllm/model_executor/layers/quantization/exl3.py")))
SO_SRC = Path(os.environ.get("GLM53_EXL3_MT_SO", "/opt/glm53/glm53_exl3_mt.so"))
SO_DST = SITE / "glm53_exl3_mt.so"
MARK = "# [glm53-exl3-mt]"

HELPER = '''
_GLM53_MT: dict[str, Any] = {"mod": None, "tried": False, "temps": {}, "locks": {}}  # [glm53-exl3-mt]


def _glm53_mt_enabled() -> bool:
    return os.environ.get("GLM53_EXL3_MT", "0") == "1"


def _glm53_mt_variant() -> int:
    return int(os.environ.get("GLM53_EXL3_MT_VARIANT", "5"))


def _glm53_mt_rows() -> int:
    return max(64, int(os.environ.get("GLM53_EXL3_MT_TEMP_ROWS", "1024")))


def _glm53_mt_min_rows() -> int:
    return int(os.environ.get("GLM53_EXL3_MT_MIN_ROWS", "32"))


def _glm53_mt_module():
    st = _GLM53_MT
    if st["tried"]:
        return st["mod"]
    st["tried"] = True
    try:
        import glm53_exl3_mt as mod

        v = _glm53_mt_variant()
        logger.info(
            "[glm53-exl3-mt] loaded: %d variants, using v%d %s, temps %d rows, min rows %d",
            mod.num_variants(), v, mod.variant_name(v), _glm53_mt_rows(), _glm53_mt_min_rows(),
        )
        st["mod"] = mod
    except Exception as exc:  # noqa: BLE001
        logger.warning("[glm53-exl3-mt] extension unavailable (%s); prefill stays on the upstream path", exc)
    return st["mod"]


def _glm53_mt_temps(device: torch.device, hidden: int, intermediate: int, concurrency: int):
    rows = _glm53_mt_rows()
    key = (str(device), hidden, intermediate, concurrency, rows)
    temps = _GLM53_MT["temps"].get(key)
    if temps is None:
        temps = tuple(
            torch.empty((concurrency, rows, d), dtype=torch.float16, device=device)
            for d in (hidden, hidden, intermediate, intermediate)
        )
        _GLM53_MT["temps"][key] = temps
        logger.info(
            "[glm53-exl3-mt] temps %d x %d rows on %s (%.0f MiB)",
            concurrency, rows, device, sum(t.numel() * 2 for t in temps) / 2**20,
        )
    return temps


def _glm53_mt_locks(device: torch.device, mod) -> torch.Tensor:
    locks = _GLM53_MT["locks"].get(str(device))
    if locks is None:
        locks = torch.zeros(int(mod.locks_ints()), dtype=torch.int32, device=device)
        _GLM53_MT["locks"][str(device)] = locks
    return locks


def _glm53_mt_prefill(
    x2d, xh, out, ids, weights, inners, expert_map, limit, layer,
    expert_count, counts, token_sorted, weight_sorted, ptrs, k, max_rows,
):
    """M-tiled prefill apply. Returns `out`, or None to fall back to the upstream path (nothing launched)."""
    mod = _glm53_mt_module()
    if mod is None or int(k) != 4 or max_rows < _glm53_mt_min_rows():
        return None
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)
    concurrency = int(layer._exl3_fused_concurrency)
    temps = _glm53_mt_temps(x2d.device, hidden, intermediate, concurrency)
    locks = _glm53_mt_locks(x2d.device, mod)
    try:
        mod.exl3_moe_mt(
            xh, out, expert_count, token_sorted, weight_sorted,
            temps[0], temps[1], temps[2], temps[3],
            MOE_ACT_SILU, int(k),
            ptrs["gate_trellis"], ptrs["gate_suh"], ptrs["gate_svh"],
            ptrs["up_trellis"], ptrs["up_suh"], ptrs["up_svh"],
            ptrs["down_trellis"], ptrs["down_suh"], ptrs["down_svh"],
            float(limit), locks, _glm53_mt_variant(),
        )
    except Exception as exc:  # noqa: BLE001  (argument checks raise before any GPU work)
        logger.error("[glm53-exl3-mt] launch rejected (%s); upstream path for the rest of this process", exc)
        _GLM53_MT["mod"] = None
        return None
    rows_cap = int(temps[0].shape[1])
    if max_rows > rows_cap:
        fat = (counts > rows_cap).nonzero(as_tuple=False).view(-1)
        if fat.numel():
            apply_exl3_python_loop(
                x2d, ids, weights, inners, expert_map, limit,
                only_experts=set(int(i) for i in fat.tolist()), out=out,
            )
    return out


'''

SEAM_OLD = """    max_rows = int(counts.max().item())
    if fat_expert_log_enabled():
        record_exl3_fat_expert_stats(counts, max_rows=max_rows)

    if max_rows <= cap:
"""
SEAM_NEW = """    max_rows = int(counts.max().item())
    if fat_expert_log_enabled():
        record_exl3_fat_expert_stats(counts, max_rows=max_rows)

    if _glm53_mt_enabled():  # [glm53-exl3-mt] 64-row M tiles for prefill; decode never reaches here
        _mt_out = _glm53_mt_prefill(
            x2d, xh, out, ids, weights, inners, expert_map, limit, layer,
            expert_count, counts, token_sorted, weight_sorted, ptrs, k, max_rows,
        )
        if _mt_out is not None:
            return _mt_out

    if max_rows <= cap:
"""
ANCHOR = "def apply_exl3_fused_moe(\n"


def install_so() -> None:
    if not SO_SRC.is_file():
        print(f"{SO_SRC}: not present — extension not installed (GLM53_EXL3_MT=1 will log a warning and use upstream)")
        return
    if SO_DST.is_file() and SO_DST.stat().st_size == SO_SRC.stat().st_size and SO_DST.stat().st_mtime >= SO_SRC.stat().st_mtime:
        print(f"{SO_DST.name}: already installed — skipping")
        return
    shutil.copyfile(SO_SRC, SO_DST)
    print(f"installed {SO_DST}")


def main() -> int:
    install_so()
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"{P}: helper anchor not unique ({text.count(ANCHOR)})")
    if text.count(SEAM_OLD) != 1:
        raise SystemExit(f"{P}: prefill seam not unique ({text.count(SEAM_OLD)})")
    for needed in ("MOE_ACT_SILU = 0", "def apply_exl3_python_loop(", "logger = init_logger(__name__)", "from typing import TYPE_CHECKING, Any"):
        if needed not in text:
            raise SystemExit(f"{P}: expected '{needed}' (upstream drift)")
    text = text.replace(ANCHOR, HELPER + ANCHOR, 1)
    text = text.replace(SEAM_OLD, SEAM_NEW, 1)
    P.write_text(text)
    print(f"patched {P.name}: M-tiled prefill MoE dispatch (GLM53_EXL3_MT)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
