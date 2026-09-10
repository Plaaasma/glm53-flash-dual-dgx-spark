#!/usr/bin/env python3
"""Per-group prefix-cache hit probe for the hybrid coordinator (diagnostic, INFO, <= 1 line/s).

When a large prompt (> ~96K tokens by block-hash count) is looked up, log the final hybrid hit length, the
longest single-group hit and the per-group hit lengths. Tells apart "the prompt changed" (all groups end at
the same place) from "a KDA tail/checkpoint state was evicted" (MLA hit far ahead of the KDA hits).
Fail closed if the coordinator anchor drifts.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
P = Path(os.environ.get("GLM53_KV_COORDINATOR_PY", "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"))
MARK = "# [glm53-apc-probe]"
ANCHOR = "        num_uncached_common_prefix_tokens = longest_hit_length - hit_length\n"
SEAM = '''        if len(block_hashes) > 1500:  # [glm53-apc-probe] large prompts only (fine-grained 64-token hashes)
            import time as _t
            _now = _t.monotonic()
            if _now - getattr(self, "_g53_probe_ts", 0.0) > 1.0:
                self._g53_probe_ts = _now
                try:
                    logger.info("[glm53-dbg] apc hit: final=%d longest=%d by_group=%s hashes=%d",
                                hit_length, longest_hit_length, list(hit_length_by_group), len(block_hashes))
                except Exception:  # noqa: BLE001
                    pass
'''
def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"{P}: probe anchor not unique ({text.count(ANCHOR)})")
    if "logger = init_logger(__name__)" not in text and "logger = " not in text:
        raise SystemExit(f"{P}: no logger")
    P.write_text(text.replace(ANCHOR, SEAM + ANCHOR, 1))
    print(f"patched {P.name}: per-group APC hit probe")
    return 0
if __name__ == "__main__":
    sys.exit(main())
