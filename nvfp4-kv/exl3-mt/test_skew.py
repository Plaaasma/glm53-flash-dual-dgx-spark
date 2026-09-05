"""Realistic prefill layer: 2044 tokens x top-8 over N_EXP experts with skewed routing (max ~570 rows like the live
fat-expert stats). Compares: upstream fused for everything (needs big temps), the kernel vLLM actually runs today
(fused <=192 rows + reconstruct+cuBLAS for fat experts, emulated), and the M-tiled kernel. Also measures the cost
of expert-order imbalance (id order vs largest-first)."""
import os, sys, time, torch
sys.path.insert(0, "/w/build")
import exllamav3_ext as up
import glm53_exl3_mt as mt
torch.manual_seed(1)
dev = torch.device("cuda")
H, I, K = 4096, 2048, 4
N_EXP = int(os.environ.get("N_EXP", "96")); TOKENS = int(os.environ.get("TOKENS", "2044")); TOPK = 8
CONC = up.exl3_moe_max_concurrency(0); ROWS = 1024; LIMIT = 7.0; V = int(os.environ.get("VARIANT", "5"))
def rand_trellis(k_in, n_out): return torch.randint(-32768, 32767, (k_in // 16, n_out // 16, 16 * K), dtype=torch.int16, device=dev)
def rand_sign(n): return torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0).half()
experts = [dict(gate=(rand_trellis(H, I), rand_sign(H), rand_sign(I)), up=(rand_trellis(H, I), rand_sign(H), rand_sign(I)), down=(rand_trellis(I, H), rand_sign(I), rand_sign(H))) for _ in range(N_EXP)]
P = {f"{w}_{n}": torch.tensor([int(ex[w][j].data_ptr()) for ex in experts], dtype=torch.int64, device=dev) for w in ("gate", "up", "down") for j, n in enumerate(("trellis", "suh", "svh"))}
temps = tuple(torch.empty((CONC, ROWS, d), dtype=torch.half, device=dev) for d in (H, H, I, I))
temps192 = tuple(torch.empty((CONC, 192, d), dtype=torch.half, device=dev) for d in (H, H, I, I))
locks = torch.zeros(mt.locks_ints(), dtype=torch.int32, device=dev)

# skewed routing: expert popularity ~ Zipf so the hottest expert gets ~570 of 16352 slots (matches live avg_max_rows)
pop = 1.0 / torch.arange(1, N_EXP + 1, device=dev).float() ** 0.55
pop = pop / pop.sum()
ids = torch.multinomial(pop.expand(TOKENS, -1), TOPK, replacement=False)      # [TOKENS, TOPK] expert ids, id 0 hottest
def batch(order):
    """order: permutation applied to expert ids (so kernel sees experts in that order)"""
    local = order[ids]
    flat_tok = torch.arange(TOKENS, device=dev).repeat_interleave(TOPK)
    flat_w = torch.full((TOKENS * TOPK,), 0.125, dtype=torch.half, device=dev)
    o = local.reshape(-1).argsort()
    ec = torch.zeros(N_EXP + 1, dtype=torch.long, device=dev); ec.scatter_add_(0, local.reshape(-1), torch.ones(TOKENS * TOPK, dtype=torch.long, device=dev))
    return flat_tok[o], flat_w[o], ec
x = (torch.randn(TOKENS, H, device=dev) * 0.5).half()
ident = torch.arange(N_EXP, device=dev)
perm = torch.randperm(N_EXP, device=dev)          # hot experts scattered (like real ids)
counts = batch(ident)[2][:N_EXP]
print(f"tokens {TOKENS} x top{TOPK} over {N_EXP} experts: rows/expert min {counts.min().item()} mean {counts.float().mean().item():.0f} max {counts.max().item()}; experts >192 rows: {(counts>192).sum().item()}")

def run_up(ts, ws, ec, tmp):
    out = torch.zeros(TOKENS, H, dtype=torch.float32, device=dev)
    up.exl3_moe(x, out, ec, ts, ws, *tmp, 0, K, K, K, P["gate_trellis"], P["gate_suh"], P["gate_svh"], P["up_trellis"], P["up_suh"], P["up_svh"], P["down_trellis"], P["down_suh"], P["down_svh"], True, False, True, False, True, False, LIMIT)
    return out
def run_mt(ts, ws, ec, v=V):
    out = torch.zeros(TOKENS, H, dtype=torch.float32, device=dev)
    mt.exl3_moe_mt(x, out, ec, ts, ws, *temps, 0, K, P["gate_trellis"], P["gate_suh"], P["gate_svh"], P["up_trellis"], P["up_suh"], P["up_svh"], P["down_trellis"], P["down_suh"], P["down_svh"], LIMIT, locks, v)
    return out
wbuf = torch.empty(H, I, dtype=torch.half, device=dev); wbuf_d = torch.empty(I, H, dtype=torch.half, device=dev)
def run_today(ts, ws, ec, order):
    """vLLM's shipped policy: fused kernel with 192-row temps (skips fat experts) + per-fat-expert reconstruct + cuBLAS."""
    out = run_up(ts, ws, ec, temps192)
    fat = (ec[:N_EXP] > 192).nonzero().view(-1).tolist()          # the .item()/.tolist() syncs vLLM pays too
    inv = torch.empty_like(order); inv[order] = ident
    for e in fat:
        ex = experts[inv[e].item()]
        n = int(ec[e].item()); h = x[:n]                              # rows of this expert (shape is what matters)
        up.reconstruct(wbuf, ex["gate"][0], K, True, False); g = (h @ wbuf).float()
        up.reconstruct(wbuf, ex["up"][0], K, True, False); u = (h @ wbuf).float()
        a = (torch.nn.functional.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)).half()
        up.reconstruct(wbuf_d, ex["down"][0], K, True, False); d = (a @ wbuf_d).float()
        out[:n] += d
    return out
def bench(fn, iters=4):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e3

ts, ws, ec = batch(perm)
ref = run_up(ts, ws, ec, temps); got = run_mt(ts, ws, ec)
print(f"correctness (skewed, id order): max rel diff {((got-ref).abs().max()/ref.abs().max()).item():.1e}")
flops = TOKENS * TOPK * 2 * (2 * H * I + I * H)
print(f"\n{'path':44s} {'ms/layer':>9} {'TFLOPS':>7}")
for name, fn in (("upstream fused, all experts (big temps)", lambda: run_up(ts, ws, ec, temps)),
                 ("vLLM today: fused<=192 + reconstruct/cuBLAS fat", lambda: run_today(ts, ws, ec, perm)),
                 (f"M-tiled v{V}, expert id order", lambda: run_mt(ts, ws, ec))):
    t = bench(fn); print(f"{name:44s} {t:9.2f} {flops/(t*1e-3)/1e12:7.1f}")
# largest-first order: relabel experts so ids are sorted by descending count (what an LPT assignment would give)
order_lpt = torch.empty_like(ident); order_lpt[counts.argsort(descending=True)] = ident
ts2, ws2, ec2 = batch(order_lpt)
for v in range(mt.num_variants()):
    t_id = bench(lambda: run_mt(ts, ws, ec, v)); t_lpt = bench(lambda: run_mt(ts2, ws2, ec2, v))
    print(f"M-tiled v{v} {mt.variant_name(v):32s} id-order {t_id:8.2f} ms   largest-first {t_lpt:8.2f} ms")
print(f"upstream fused                              id-order {bench(lambda: run_up(ts, ws, ec, temps)):8.2f} ms   largest-first {bench(lambda: run_up(ts2, ws2, ec2, temps)):8.2f} ms")
print(f"footprint: reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB")
