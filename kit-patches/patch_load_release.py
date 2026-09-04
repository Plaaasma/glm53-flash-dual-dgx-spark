#!/usr/bin/env python3
"""Release caching-allocator slack after each weight-load pass.

InstantTensor yields device-resident tensors, so every weight passes through
the CUDA caching allocator as a transient (clone, then a TP narrow().contiguous()
compaction) before being copied into its parameter. Over 164 GB of churn the
allocator keeps ~2.4 GiB of freed-but-mapped segments (measured: allocated
82,387 MiB vs reserved 84,818 MiB right after load, 2026-09-04); on the 121 GiB
unified-memory Sparks that slack is exactly the margin graph capture needs,
and both InstantTensor boots tripped the memory watchdog. The mmap loader
copies CPU->param and never creates the transients. gc + empty_cache after
load_weights returns the slack to the OS; harmless for any loader.
Idempotent; fails closed on anchor drift.
"""
import os, sys
from pathlib import Path
MARK = "# [glm53-load-release]"
P = Path(os.environ.get("GLM53_DEFAULT_LOADER_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/default_loader.py"))
OLD = """        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
"""
NEW = OLD + """        import gc as _g53gc  """ + MARK + """
        _g53gc.collect()
        torch.cuda.empty_cache()
"""
DIAG_CALL = "        _glm53_load_diag(model, type(model).__name__)  # [glm53-load-diag]\n"
REL_LINES = NEW[len(OLD):]


def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if DIAG_CALL in t:
        if t.count(DIAG_CALL) != 1:
            raise SystemExit(f"{P}: diag call anchor not unique — refusing to patch")
        t = t.replace(DIAG_CALL, REL_LINES + DIAG_CALL, 1)   # release runs before the inventory
    else:
        if t.count(OLD) != 1:
            raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
        t = t.replace(OLD, NEW, 1)
    P.write_text(t); print(f"patched {P.name} (allocator slack released after load)"); return 0
if __name__ == "__main__":
    sys.exit(main())
