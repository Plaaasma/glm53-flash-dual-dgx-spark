#!/usr/bin/env python3
"""InstantTensor: rank-local Direct I/O instead of NCCL-sharded all-gather.

With TP>1, vLLM's instanttensor_weights_iterator passes the world process
group, so each rank reads a slice and the ranks all-gather over NCCL. On the
2x GB10 cluster that path left ~1.5-2 GiB of NCCL transport buffers resident
for the life of the communicator (boots peaked at 120.2-120.9 GiB used vs
119.3-119.5 with the mmap loader; both InstantTensor boots tripped the
memory watchdog, 2026-09-04). A standalone load with no process group
returned every byte after close. Env GLM53_IT_LOCAL_READS=1 (default) makes
each node read the full checkpoint from its own NVMe: no NCCL, ~35-60 s for
164 GB vs 259 s for the mmap loader. Idempotent; fails closed on drift.
"""
import os, sys
from pathlib import Path
MARK = "# [glm53-it-local]"
P = Path(os.environ.get("GLM53_WEIGHT_UTILS_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"))
OLD = """        device=device,
        process_group=process_group,
        copy=True,
    ) as f:
"""
NEW = """        device=device,
        """ + MARK + """ rank-local reads: the NCCL all-gather path keeps
        # ~1.5-2 GiB of transport buffers resident on this cluster.
        process_group=(None if __import__("os").environ.get("GLM53_IT_LOCAL_READS", "1") == "1"
                       else process_group),
        copy=True,
    ) as f:
"""
def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if t.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
    P.write_text(t.replace(OLD, NEW, 1)); print(f"patched {P.name} (InstantTensor rank-local reads)"); return 0
if __name__ == "__main__":
    sys.exit(main())
