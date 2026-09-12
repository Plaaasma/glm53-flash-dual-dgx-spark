#!/usr/bin/env python3
"""Keep a hit session's KDA checkpoints young in the LRU, and count prefix-cache evictions by group.

Why (2026-09-11): the block pool evicts by plain LRU on free time. A prefix-cache hit touches every attention
page of the prefix (they leave the free queue and are re-appended when the request ends) but only the single
KDA boundary state it resumes from; the session's earlier KDA checkpoints keep the age of the prefill that
created them. Under a sub-agent fan-out they are the oldest blocks in the pool and get evicted first, and once
they are gone the hybrid hit collapses to zero even though every attention page is still cached (probe:
`final=0 longest=461312`), i.e. a full re-prefill of a 460K session 19 minutes after its last turn.

What: after `find_longest_cache_hit`, for every Mamba/KDA group, look up the cached state block at each page
boundary inside the hit prefix (the page-level key is the fine-grained hash ending that page) and move it to
the tail of the free queue when it is idle there (no reference is taken, so nothing leaks). The session's
checkpoints then age with the session, and a partial eviction takes its attention tail pages first (a short
partial re-prefill) instead of its checkpoints. Also counts evictions by cache group once a minute
(`[glm53-apc] evicted cached blocks by group ...`) so the effect can be verified in the engine log.

Fail closed if the vLLM anchors drift.
"""
import os
from pathlib import Path

V = Path(os.environ.get("GLM53_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"))
MARK = "[glm53-apc-refresh]"

HELPER_MGR = '''
def _glm53_refresh_sparse_checkpoints(mgr, request, hit_length):  # [glm53-apc-refresh]
    """Move the hit prefix's cached KDA state blocks (one per page boundary) to the LRU tail; no refs taken."""
    try:
        if hit_length <= 0:
            return
        st = getattr(mgr, "_g53_refresh", None)
        if st is None:
            import time as _t
            groups = []
            for gi, group in enumerate(mgr.kv_cache_config.kv_cache_groups):
                if type(group.kv_cache_spec).__name__ == "MambaSpec":
                    groups.append((gi, int(group.kv_cache_spec.block_size)))
            st = mgr._g53_refresh = {"groups": groups, "hits": 0, "moved": 0, "t": _t.monotonic(), "clock": _t.monotonic}
        if not st["groups"]:
            return
        pool = mgr.block_pool
        hbs = int(pool.hash_block_size)
        queue = pool.free_block_queue
        hashes = request.block_hashes
        n_hashes = len(hashes)
        moved = 0
        for gi, bs in st["groups"]:
            if bs % hbs:
                continue
            step = bs // hbs
            # page i ends at (i + 1) * bs tokens and is keyed by the fine hash at index (i + 1) * step - 1
            for fidx in range(step - 1, min(hit_length // hbs, n_hashes), step):
                cached = pool.get_cached_block(hashes[fidx], [gi])
                if not cached:
                    continue
                blk = cached[0]
                if (blk.ref_cnt == 0 and not blk.is_null
                        and blk.prev_free_block is not None and blk.next_free_block is not None):
                    queue.remove(blk)
                    queue.append(blk)
                    moved += 1
        st["hits"] += 1
        st["moved"] += moved
        now = st["clock"]()
        if now - st["t"] >= 60.0:
            logger.info("[glm53-apc] refreshed %d KDA checkpoint blocks over %d cache hits in the last %.0f s",
                        st["moved"], st["hits"], now - st["t"])
            st["t"] = now
            st["hits"] = 0
            st["moved"] = 0
    except Exception as e:  # noqa: BLE001 -- bookkeeping must never break scheduling
        if not getattr(mgr, "_g53_refresh_err", False):
            mgr._g53_refresh_err = True
            logger.warning("[glm53-apc] checkpoint refresh disabled after error: %r", e)

'''
SEAM_MGR_OLD = ("        blocks = self.create_kv_cache_blocks(computed_blocks)\n"
                "        return blocks, num_new_computed_tokens, shared_prefix_boundary\n")
SEAM_MGR_NEW = ("        _glm53_refresh_sparse_checkpoints(self, request, num_new_computed_tokens)  # [glm53-apc-refresh]\n"
                + SEAM_MGR_OLD)

HELPER_POOL = '''
def _glm53_count_eviction(pool, evicted_hashes):  # [glm53-apc-refresh]
    """Per-group eviction counter, logged once a minute."""
    try:
        st = getattr(pool, "_g53_evict", None)
        if st is None:
            import time as _t
            st = pool._g53_evict = {"by_group": {}, "t": _t.monotonic(), "clock": _t.monotonic}
        for h in evicted_hashes:
            g = get_group_id(h)
            st["by_group"][g] = st["by_group"].get(g, 0) + 1
        now = st["clock"]()
        if now - st["t"] >= 60.0:
            logger.info("[glm53-apc] evicted cached blocks by group in the last %.0f s: %s (free queue %d)",
                        now - st["t"], dict(sorted(st["by_group"].items())), pool.free_block_queue.num_free_blocks)
            st["t"] = now
            st["by_group"] = {}
    except Exception:  # noqa: BLE001
        pass

'''
SEAM_POOL_OLD = ("        evicted_hashes = self._remove_cached_block_hashes(block)\n"
                 "        if not evicted_hashes:\n")
SEAM_POOL_NEW = ("        evicted_hashes = self._remove_cached_block_hashes(block)\n"
                 "        if evicted_hashes:  # [glm53-apc-refresh]\n"
                 "            _glm53_count_eviction(self, evicted_hashes)\n"
                 "        if not evicted_hashes:\n")


def patch(path: Path, helper: str, class_anchor: str, seam_old: str, seam_new: str, label: str) -> None:
    text = path.read_text()
    if MARK in text:
        print(f"{path.name}: {label} already present - skipping")
        return
    for needle in (class_anchor, seam_old):
        n = text.count(needle)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one anchor for {label} ({needle[:40]!r}), found {n} - refusing")
    text = text.replace(class_anchor, helper + class_anchor, 1).replace(seam_old, seam_new, 1)
    compile(text, str(path), "exec")
    path.write_text(text)
    print(f"patched {path.name}: {label}")


def main() -> int:
    patch(V / "v1/core/kv_cache_manager.py", HELPER_MGR, "\nclass KVCacheManager:\n", SEAM_MGR_OLD, SEAM_MGR_NEW, "checkpoint refresh on hit")
    patch(V / "v1/core/block_pool.py", HELPER_POOL, "\nclass BlockPool:\n", SEAM_POOL_OLD, SEAM_POOL_NEW, "eviction counter by group")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
