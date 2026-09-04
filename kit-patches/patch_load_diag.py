#!/usr/bin/env python3
"""Post-load memory inventory (diagnostic; env GLM53_LOAD_DIAG=1).

After each DefaultModelLoader.load_weights pass, logs /proc/meminfo and
/proc/self/status accounting, torch device + pinned-host allocator stats,
the byte total of tensors reachable from the model object graph, and every
CUDA tensor reachable via gc that the model does NOT own (top entries by
size). Used to locate the ~3 GiB residue InstantTensor boots carry into
graph capture (2026-09-04). Idempotent; fails closed on anchor drift.
"""
import os, sys
from pathlib import Path
MARK = "# [glm53-load-diag]"
P = Path(os.environ.get("GLM53_DEFAULT_LOADER_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/default_loader.py"))
FUNC = '''
def _glm53_load_diag(model, tag):  ''' + MARK + '''
    import os, gc
    if os.environ.get("GLM53_LOAD_DIAG", "0") != "1":
        return
    try:
        import torch
        torch.cuda.synchronize()
        mi = {}
        with open("/proc/meminfo") as f:
            for l in f:
                p = l.split(); mi[p[0].rstrip(":")] = int(p[1]) // 1024
        st = {}
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith(("VmRSS", "VmLck", "VmPin", "RssAnon", "RssFile", "RssShmem", "VmSwap")):
                    p = l.split(); st[p[0].rstrip(":")] = int(p[1]) // 1024
        owned = set(); owned_bytes = 0
        def walk(o, depth):
            nonlocal owned_bytes
            if depth > 5: return
            if torch.is_tensor(o):
                try:
                    if o.is_cuda:
                        k = o.untyped_storage().data_ptr()
                        if k not in owned:
                            owned.add(k); owned_bytes += o.untyped_storage().nbytes()
                except Exception: pass
                return
            if isinstance(o, (list, tuple)):
                for x in o: walk(x, depth + 1)
            elif isinstance(o, dict):
                for x in o.values(): walk(x, depth + 1)
            elif hasattr(o, "__dict__") and not isinstance(o, type):
                for x in vars(o).values(): walk(x, depth + 1)
        for m in model.modules():
            walk(m, 0)
        extra = []; seen = set()
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    k = o.untyped_storage().data_ptr()
                    if k in owned or k in seen: continue
                    seen.add(k); extra.append((o.untyped_storage().nbytes(), tuple(o.shape), str(o.dtype).replace("torch.", "")))
            except Exception: pass
        extra.sort(reverse=True); tot_extra = sum(e[0] for e in extra) / 2**20
        try: hs = torch.cuda.host_memory_stats(); host = f"pinned_alloc={hs.get('allocated_bytes.all.current',0)/2**20:.0f} pinned_res={hs.get('reserved_bytes.all.current',0)/2**20:.0f}"
        except Exception: host = "pinned=n/a"
        logger.info("[glm53-load-diag %s] meminfo MiB: MemAvailable=%d MemFree=%d Cached=%d Buffers=%d AnonPages=%d Mapped=%d Shmem=%d Mlocked=%d Unevictable=%d Slab=%d SUnreclaim=%d KernelStack=%d PageTables=%d Committed_AS=%d | proc MiB: %s | torch MiB: allocated=%.0f reserved=%.0f model-owned=%.0f %s | extra cuda tensors: n=%d total=%.0f MiB top=%s",
            tag, mi.get("MemAvailable",0), mi.get("MemFree",0), mi.get("Cached",0), mi.get("Buffers",0), mi.get("AnonPages",0), mi.get("Mapped",0), mi.get("Shmem",0), mi.get("Mlocked",0), mi.get("Unevictable",0), mi.get("Slab",0), mi.get("SUnreclaim",0), mi.get("KernelStack",0), mi.get("PageTables",0), mi.get("Committed_AS",0),
            st, torch.cuda.memory_allocated()/2**20, torch.cuda.memory_reserved()/2**20, owned_bytes/2**20, host,
            len(extra), tot_extra, [(round(e[0]/2**20), e[1], e[2]) for e in extra[:10]])
    except Exception as e:
        logger.warning("[glm53-load-diag] failed: %r", e)

'''
A_OLD = "logger = init_logger(__name__)\n"
B_OLD = """        logger.info_once(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights - self.counter_before_loading_weights,
        )
"""
B_NEW = B_OLD + "        _glm53_load_diag(model, type(model).__name__)  " + MARK + "\n"
def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping"); return 0
    if t.count(A_OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one function anchor, found {t.count(A_OLD)} — refusing to patch")
    t = t.replace(A_OLD, A_OLD + FUNC, 1)
    REL_END = "        torch.cuda.empty_cache()\n"
    if "[glm53-load-release]" in t and t.count(REL_END) == 1:
        t = t.replace(REL_END, REL_END + B_NEW[len(B_OLD):], 1)   # inventory measures AFTER the release
    else:
        if t.count(B_OLD) != 1:
            raise SystemExit(f"{P}: expected exactly one call anchor, found {t.count(B_OLD)} — refusing to patch")
        t = t.replace(B_OLD, B_NEW, 1)
    P.write_text(t); print(f"patched {P.name} (post-load memory inventory)"); return 0
if __name__ == "__main__":
    sys.exit(main())
