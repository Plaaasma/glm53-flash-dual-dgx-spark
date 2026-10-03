#!/usr/bin/env python3
"""Weight-only block-FP8 for the non-expert projections and LM head (glm53_dense_rt).

The EXL3 checkpoint quantizes only the routed experts; every other matrix stays BF16 and is ~half of the bytes a
decode step reads. glm53_dense_rt converts those matrices to FP8 e4m3 with 128x128 block scales after load and
swaps in Triton kernels that keep activations in BF16 (W8A16).

This patcher:
  1. installs the runtime (/opt/glm53/glm53_dense_rt.py -> vllm/glm53_dense_rt.py);
  2. calls glm53_dense_rt.convert_model(model) at the end of model_loader.utils.process_weights_after_loading,
     which every loader runs for the target and the MTP draft.

GLM53_DENSE_W8=0 (default) leaves every matrix in BF16 (the runtime returns immediately). Fail closed if the seam
drifts.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE_PACKAGES", "/usr/local/lib/python3.12/dist-packages"))
P = SITE / "vllm/model_executor/model_loader/utils.py"
RT_SRC = Path(os.environ.get("GLM53_DENSE_RT", "/opt/glm53/glm53_dense_rt.py"))
RT_DST = SITE / "vllm/glm53_dense_rt.py"
MARK = "# [glm53-dense]"

ANCHOR = """    # Needed for torchao model reloading via model.reload_weights
    # @kylesayrs @jerryzh168 this can be removed if callers move to `reload_weights`
    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)
"""
SEAM = ANCHOR + """
    # [glm53-dense] 8-bit W8A16 for the BF16 projections (GLM53_DENSE_W8, default off)
    from vllm import glm53_dense_rt as _glm53_dense

    _glm53_dense.convert_model(model)
"""


def main() -> int:
    if not RT_SRC.is_file():
        print(f"patch_dense_fp8: {RT_SRC} missing; skipping (BF16 projections)")
        return 0
    shutil.copyfile(RT_SRC, RT_DST)
    src = P.read_text()
    if MARK in src:
        print("patch_dense_fp8: seam already present; runtime refreshed")
        return 0
    if src.count(ANCHOR) != 1:
        print("patch_dense_fp8: FATAL anchor not found exactly once in", P, file=sys.stderr)
        return 1
    P.write_text(src.replace(ANCHOR, SEAM))
    print("patched model_loader/utils.py: glm53 dense FP8 conversion after load "
          f"(GLM53_DENSE_W8={os.environ.get('GLM53_DENSE_W8', '0')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
