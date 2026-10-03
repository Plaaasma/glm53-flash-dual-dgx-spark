"""Per-layer decode MoE time, current vLLM path (prologue + exllamav3 exl3_moe) vs glm53_exl3_dec, inside CUDA graphs.

Each graph holds NCALL calls, each with its own x / routing, on NEXP real experts of one layer (TP=2 rank 0).
Consecutive calls draw from disjoint halves of the loaded experts when T <= 8 (so the previous call's experts are
never the ones in L2), the whole set for larger T. GB/s = distinct experts' trellis bytes / time.

Env: LAYER (20), NEXP (96), NCALL (16), REPS (5), MODES ("bench" | "tune" | "stages", comma list),
     TS (comma list of T), CFGS (tune: "nt,w,sk,pf/nt,w,sk,pf;..." gate-up/down pairs).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import common as C  # noqa: E402
import glm53_exl3_dec_rt as rt  # noqa: E402

LAYER = int(os.environ.get("LAYER", "20"))
NEXP = int(os.environ.get("NEXP", "96"))
NCALL = int(os.environ.get("NCALL", "16"))
REPS = int(os.environ.get("REPS", "5"))
MODES = os.environ.get("MODES", "bench").split(",")
dev = torch.device("cuda")
gen = torch.Generator().manual_seed(7)

experts = list(range(0, 288, 3))[:NEXP]
t0 = time.time()
layer = C.load_layer(LAYER, experts, dev)
C.build_old_state(layer)
rt.prepare_layer(layer, max_rows=64, slots=8)
torch.cuda.synchronize()
print(f"layer {LAYER}: {NEXP} experts loaded in {time.time()-t0:.0f}s ({NEXP * C.BYTES_PER_EXPERT / 2**20:.0f} MiB "
      f"trellis); GB/s below are 1e9 bytes/s over distinct experts' trellis bytes", flush=True)
C.mem_report("loaded")
half = NEXP // 2
POOLS = (list(range(0, half)), list(range(half, NEXP)))


def make_calls(T, kind):
    calls, distinct = [], []
    for i in range(NCALL):
        pool = POOLS[i % 2] if T <= 8 else list(range(NEXP))
        ids = C.routing_correlated(T, pool, gen) if kind == "correlated" else C.routing_random(T, pool, gen)
        calls.append((C.hidden(T, gen, scales=(1.0,)).to(dev), ids.to(dev), C.routing_weights(T, gen).to(dev)))
        distinct.append(len(set(ids.flatten().tolist())))
    return calls, sum(distinct) / len(distinct)


def time_graph(fn, calls):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls[:2]:
            fn(*c)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            fn(*c)
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(REPS):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / len(calls))
    del g
    return best


def old_fn(x, ids, w):
    return C.old_apply(x, ids, w, layer)


def new_fn(x, ids, w, stop_after=0):
    return rt.decode_moe(x, ids, w, layer, C.LIMIT, stop_after=stop_after)


def gbs(distinct, ms):
    return distinct * C.BYTES_PER_EXPERT / (ms * 1e-3) / 1e9


TS = [int(v) for v in os.environ.get("TS", "1,4,8,16,32,64").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "3"))

if "ceiling" in MODES:
    # what this GPU streams in this harness: a reduction over the loaded trellises (read once a call)
    bufs = [layer.w13_trellis.view(torch.float32).view(-1), layer.w2_trellis.view(torch.float32).view(-1)]
    nbytes = sum(b.numel() * 4 for b in bufs)
    calls = [(b,) for b in bufs] * 4
    t = min(time_graph(lambda b: b.sum(), calls) for _ in range(ROUNDS))
    print(f"\n## streaming ceiling: torch.sum over the {nbytes/2**20:.0f} MiB of trellis: "
          f"{nbytes / len(bufs) / (t * 1e-3) / 1e9:.0f} GB/s")

if "tune" in MODES:
    raw = os.environ.get("CFGS", "")
    cfgs = []
    for item in [s for s in raw.split(";") if s.strip()]:
        gu, d = item.split("/")
        cfgs.append((tuple(int(v) for v in gu.split(",")), tuple(int(v) for v in d.split(","))))
    if not cfgs:
        cfgs = [((8, 4, 4, 1), (8, 4, 1, 1))]
    print(f"\n## tune (random routing, best of {ROUNDS} interleaved rounds): ms per call (GB/s)")
    print("| cfg_gu / cfg_d | " + " | ".join(f"T={T}" for T in TS) + " |")
    print("|---|" + "---|" * len(TS))
    data = {T: make_calls(T, "random") for T in TS}
    best = {}
    for rnd in range(ROUNDS):
        for cg, cd in cfgs:
            rt._SCRATCH.clear()
            torch.cuda.empty_cache()
            try:
                rt.prepare_layer(layer, max_rows=64, slots=8, cfg_gu=cg, cfg_d=cd)
            except ValueError as exc:
                best[(cg, cd)] = None
                continue
            for T in TS:
                ms = time_graph(new_fn, data[T][0])
                key = (cg, cd, T)
                best[key] = min(best.get(key, float("inf")), ms)
        for T in TS:
            key = ("old", T)
            best[key] = min(best.get(key, float("inf")), time_graph(old_fn, data[T][0]))
    print("| current exl3_moe | " + " | ".join(f"{best[('old', T)]:.3f} ({gbs(data[T][1], best[('old', T)]):.0f})"
                                              for T in TS) + " |")
    for cg, cd in cfgs:
        if best.get((cg, cd), 1) is None:
            print(f"| {cg} / {cd} | not compiled / does not divide |")
            continue
        print(f"| {cg} / {cd} | " + " | ".join(f"{best[(cg, cd, T)]:.3f} ({gbs(data[T][1], best[(cg, cd, T)]):.0f})"
                                              for T in TS) + " |", flush=True)
    rt._SCRATCH.clear()
    torch.cuda.empty_cache()
    rt.prepare_layer(layer, max_rows=64, slots=8)

if "stages" in MODES:
    st = layer._glm53_dec
    print(f"\n## per-stage time (random routing), cfg_gu {st.cfg_gu} cfg_d {st.cfg_d}: ms per call")
    names = ["group", "rot_in", "gate+up", "gu epilogue", "down", "down+combine"]
    print("| T | distinct | " + " | ".join(names) + " | total |")
    print("|---|---|" + "---|" * (len(names) + 1))
    for T in TS:
        calls, dist = make_calls(T, "random")
        cum = [min(time_graph(lambda x, i, w, s=s: new_fn(x, i, w, stop_after=s), calls) for _ in range(ROUNDS))
               for s in range(1, 7)]
        parts = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, 6)]
        print(f"| {T} | {dist:.1f} | " + " | ".join(f"{p:.3f}" for p in parts) + f" | {cum[-1]:.3f} |", flush=True)

def quick_ceiling():
    """GB/s of a plain streaming read right now (a busy co-tenant on the GPU shows up here first)."""
    bufs = [layer.w13_trellis.view(torch.float32).view(-1), layer.w2_trellis.view(torch.float32).view(-1)]
    nbytes = sum(b.numel() * 4 for b in bufs) / len(bufs)
    t = time_graph(lambda b: b.sum(), [(b,) for b in bufs] * 2)
    return nbytes / (t * 1e-3) / 1e9


if "bench" in MODES:
    st = layer._glm53_dec
    KINDS = os.environ.get("KINDS", "random,correlated").split(",")
    MIN_CEIL = float(os.environ.get("MIN_CEIL", "215"))
    print(f"\n## old vs new (CUDA graphs, {NCALL} calls a graph, best of {REPS}); new cfg_gu {st.cfg_gu} "
          f"cfg_d {st.cfg_d}; ceiling = streaming read measured before / after the row")
    print("| routing | T | distinct experts | current ms | current GB/s | new ms | new GB/s | speedup | ceiling GB/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kind in KINDS:
        for T in TS:
            calls, dist = make_calls(T, kind)
            c0 = quick_ceiling()
            t_old = t_new = float("inf")
            for _ in range(ROUNDS):
                t_old = min(t_old, time_graph(old_fn, calls))
                t_new = min(t_new, time_graph(new_fn, calls))
            c1 = quick_ceiling()
            flag = "" if min(c0, c1) >= MIN_CEIL else " DISTURBED"
            print(f"| {kind} | {T} | {dist:.1f} | {t_old:.3f} | {gbs(dist, t_old):.0f} | {t_new:.3f} | "
                  f"{gbs(dist, t_new):.0f} | {t_old / t_new:.2f}x | {c0:.0f} / {c1:.0f}{flag} |", flush=True)
C.mem_report("end")
