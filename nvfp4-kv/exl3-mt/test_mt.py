"""Correctness + benchmark for the M-tiled EXL3 MoE kernel vs upstream exl3_moe.

Synthetic experts (random trellis bits decode to valid fp16 weights), hidden 4096, intermediate 2048 (TP=2 local),
4 bpw mcg codebook, SiLU. Reference for every row count is upstream exl3_moe with temps large enough to hold it.
"""
import os, sys, time, math, torch
sys.path.insert(0, "/w/build")
import exllamav3_ext as up
import glm53_exl3_mt as mt

torch.manual_seed(0)
dev = torch.device("cuda")
H, I, K, N_EXP = 4096, 2048, 4, int(os.environ.get("N_EXP", "32"))
CONC = up.exl3_moe_max_concurrency(0)
MAX_ROWS = int(os.environ.get("MAX_ROWS", "1024"))
LIMIT = 7.0

def rand_trellis(k_in, n_out):
    return torch.randint(-32768, 32767, (k_in // 16, n_out // 16, 16 * K), dtype=torch.int16, device=dev)
def rand_sign(n):
    return torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0).half()

experts = []
for e in range(N_EXP):
    experts.append(dict(
        gate=(rand_trellis(H, I), rand_sign(H), rand_sign(I)),
        up=(rand_trellis(H, I), rand_sign(H), rand_sign(I)),
        down=(rand_trellis(I, H), rand_sign(I), rand_sign(H)),
    ))
def ptrs(which, j):
    return torch.tensor([int(ex[which][j].data_ptr()) for ex in experts], dtype=torch.int64, device=dev)
P = {f"{w}_{n}": ptrs(w, j) for w in ("gate", "up", "down") for j, n in enumerate(("trellis", "suh", "svh"))}
temps = (torch.empty((CONC, MAX_ROWS, H), dtype=torch.half, device=dev), torch.empty((CONC, MAX_ROWS, H), dtype=torch.half, device=dev),
         torch.empty((CONC, MAX_ROWS, I), dtype=torch.half, device=dev), torch.empty((CONC, MAX_ROWS, I), dtype=torch.half, device=dev))
locks = torch.zeros(mt.locks_ints(), dtype=torch.int32, device=dev)
print(f"experts {N_EXP}, concurrency {CONC}, temps rows {MAX_ROWS}, variants: {[mt.variant_name(i) for i in range(mt.num_variants())]}")

def make_batch(rows_per_expert, topk=1):
    """Every expert gets exactly rows_per_expert rows (topk=1 routing)."""
    tokens = rows_per_expert * N_EXP
    x = (torch.randn(tokens, H, device=dev) * 0.5).half()
    expert_count = torch.zeros(N_EXP + 1, dtype=torch.long, device=dev); expert_count[:N_EXP] = rows_per_expert
    token_sorted = torch.arange(tokens, dtype=torch.long, device=dev)
    weight_sorted = torch.full((tokens,), 0.7, dtype=torch.half, device=dev)
    return x, expert_count, token_sorted, weight_sorted

def run_up(x, ec, ts, ws):
    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=dev)
    up.exl3_moe(x, out, ec, ts, ws, *temps, 0, K, K, K, P["gate_trellis"], P["gate_suh"], P["gate_svh"], P["up_trellis"], P["up_suh"], P["up_svh"],
                P["down_trellis"], P["down_suh"], P["down_svh"], True, False, True, False, True, False, LIMIT)
    return out
def run_mt(x, ec, ts, ws, variant):
    out = torch.zeros(x.shape[0], H, dtype=torch.float32, device=dev)
    mt.exl3_moe_mt(x, out, ec, ts, ws, *temps, 0, K, P["gate_trellis"], P["gate_suh"], P["gate_svh"], P["up_trellis"], P["up_suh"], P["up_svh"],
                   P["down_trellis"], P["down_suh"], P["down_svh"], LIMIT, locks, variant)
    return out

def bench(fn, iters):
    fn(); torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters

# ---- correctness
print("\n== correctness vs upstream exl3_moe (max |diff| / max |ref|)")
for rows in (1, 5, 16, 17, 64, 100, 192, 300, 600):
    if rows > MAX_ROWS: continue
    x, ec, ts, ws = make_batch(rows)
    ref = run_up(x, ec, ts, ws)
    line = f"rows/expert {rows:4d}: ref max {ref.abs().max().item():8.3f} |"
    ok_all = True
    for v in range(mt.num_variants()):
        got = run_mt(x, ec, ts, ws, v)
        d = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        finite = torch.isfinite(got).all().item()
        ok = finite and d < 2e-2
        ok_all &= ok
        line += f" v{v} {d:.1e}{'' if ok else ' FAIL'}"
    print(line + ("" if ok_all else "   <-- MISMATCH"))

# ---- benchmark
print("\n== time per layer-call (ms) for N_EXP experts, all with the same row count; last column = achieved TFLOPS of best")
flop_per_row = 2 * (2 * H * I + I * H)  # gate, up, down
hdr = f"{'rows':>5} {'tokens':>7} {'upstream':>9}" + "".join(f"{'v'+str(v):>9}" for v in range(mt.num_variants())) + f"{'best':>8} {'speedup':>8} {'TFLOPS':>7}"
print(hdr)
for rows in (4, 8, 16, 32, 57, 64, 96, 128, 192, 256, 512, 1024):
    if rows > MAX_ROWS: continue
    x, ec, ts, ws = make_batch(rows)
    iters = 3 if rows >= 256 else 6
    t_up = bench(lambda: run_up(x, ec, ts, ws), iters)
    t_v = [bench(lambda v=v: run_mt(x, ec, ts, ws, v), iters) for v in range(mt.num_variants())]
    best = min(range(len(t_v)), key=lambda i: t_v[i])
    tf = rows * N_EXP * flop_per_row / (t_v[best] * 1e-3) / 1e12
    print(f"{rows:5d} {rows*N_EXP:7d} {t_up:9.2f}" + "".join(f"{t:9.2f}" for t in t_v) + f"{'v'+str(best):>8} {t_up/t_v[best]:8.2f}x {tf:7.1f}")

# ---- reference points: what the same math costs as dense fp16 cuBLAS (weights pre-dequantized) = compute ceiling
print("\n== dense fp16 cuBLAS reference (per expert, weights already fp16): the ceiling a perfect fused kernel would approach")
wg = torch.randn(H, I, device=dev).half(); wd = torch.randn(I, H, device=dev).half()
for rows in (16, 64, 128, 256, 512, 1024):
    a = torch.randn(rows, H, device=dev).half()
    def dense():
        g = a @ wg; u = a @ wg; d = (g * u) @ wd
    t = bench(dense, 10)
    print(f"  rows {rows:5d}: {t*N_EXP:7.2f} ms for {N_EXP} experts  ({rows*N_EXP*flop_per_row/(t*N_EXP*1e-3)/1e12:5.1f} TFLOPS)")
print(f"\nfootprint: reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB")
