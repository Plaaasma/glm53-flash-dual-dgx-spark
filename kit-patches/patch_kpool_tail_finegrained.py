#!/usr/bin/env python3
"""Let fine-grained prefix hits coexist with the K-pool tail scratch group.

``HybridKVCacheCoordinator`` disables ``enable_partial_hash_hits`` if any
manager has ``supports_fine_grained_hash_lookup = False`` and a block size
different from the hash unit. ``KpoolTailManager`` trips that: its block is
page-fitted (not 64) and the class flag is False. But the tail manager never
matches anything (``find_longest_cache_hit`` returns 0 / empty) and never
caches (``cache_blocks`` is a no-op), so the alignment of a hit is irrelevant
to it. Marking it fine-grained-capable only lets the MLA and mamba groups
(both already fine-grained-capable upstream) hit at ``--prefix-match-unit``
boundaries inside a 7936-token page, and lets the scheduler register mamba
partial-tail states at prompt ends. Measured need: agent-turn re-submits
replayed ~8K tokens per turn at page granularity (2026-09-04).

Fail closed on anchor drift; idempotent.
"""
import os
import sys
from pathlib import Path

MARK = "# [glm53-kpool-tail-finegrained]"
P = Path(os.environ.get(
    "GLM53_STKVM_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py"))

OLD = """class KpoolTailManager(FullAttentionManager):"""


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    if text.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one KpoolTailManager class anchor — refusing to patch")
    head, tail = text.split(OLD, 1)
    # first class-body line after the header is the docstring or the flag; we
    # append our override right after the existing flag line inside the class
    flag_old = "    supports_fine_grained_hash_lookup: ClassVar[bool] = False\n"
    body_idx = tail.find(flag_old)
    if body_idx < 0 or body_idx > 2000:
        raise SystemExit(f"{P}: KpoolTailManager fine-grained flag not found near class header — refusing to patch")
    tail = tail[:body_idx] + (
        "    # Never matches or caches, so hit alignment is irrelevant here;\n"
        "    # True lets the MLA/mamba groups take fine-grained hits.  " + MARK + "\n"
        "    supports_fine_grained_hash_lookup: ClassVar[bool] = True\n"
    ) + tail[body_idx + len(flag_old):]
    P.write_text(head + OLD + tail)
    print(f"patched {P.name} (KpoolTailManager: fine-grained hash lookup enabled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
