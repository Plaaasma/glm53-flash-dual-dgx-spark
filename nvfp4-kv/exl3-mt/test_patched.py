"""Apply patch_exl3_mt.py inside a throwaway container, then exercise vLLM's apply_exl3_fused_moe with the flag
off (upstream) and on (M-tiled) on fake layers/inners that carry the same pointer tables the real loader builds."""
import os, sys, time, types, subprocess, torch
r = subprocess.run([sys.executable, "/w/patch_exl3_mt.py"], capture_output=True, text=True); print(r.stdout.strip(), r.stderr.strip()[-300:])
assert r.returncode == 0
os.environ["EXL3_TEMP_ROWS_FUSED"] = "192"
import importlib
exl3 = importlib.import_module("vllm.model_executor.layers.quantization.exl3")
import exllamav3_ext as up
dev = torch.device("cuda"); torch.manual_seed(3)
H, I, K, N_EXP = 4096, 2048, 4, 96
class Pack:  # stands in for LinearEXL3 (only the attributes the pointer tables read)
    def __init__(self, k_in, n_out):
        self.trellis = torch.randint(-32768, 32767, (k_in // 16, n_out // 16, 16 * K), dtype=torch.int16, device=dev)
        self.suh = torch.where(torch.rand(k_in, device=dev) < 0.5, -1.0, 1.0).half()
        self.svh = torch.where(torch.rand(n_out, device=dev) < 0.5, -1.0, 1.0).half()
inners = [{"gate": Pack(H, I), "up": Pack(H, I), "down": Pack(I, H)} for _ in range(N_EXP)]
layer = types.SimpleNamespace(w13_trellis=inners[0]["gate"].trellis, _exl3_hidden_size=H, _exl3_intermediate_local=I, _exl3_bits=K)
exl3.build_exl3_fused_state(layer, inners)
print("fused state: cap", layer._exl3_fused_temps[0].shape, "concurrency", layer._exl3_fused_concurrency)

def routed(tokens, skew):
    pop = 1.0 / torch.arange(1, N_EXP + 1, device=dev).float() ** skew; pop /= pop.sum()
    ids = torch.multinomial(pop.expand(tokens, -1), 8, replacement=False)
    w = torch.softmax(torch.randn(tokens, 8, device=dev), -1)
    return ids, w
def run(flag, tokens, skew):
    os.environ["GLM53_EXL3_MT"] = flag
    torch.manual_seed(7); ids, w = routed(tokens, skew)
    x = (torch.randn(tokens, H, device=dev) * 0.5).half()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = exl3.apply_exl3_fused_moe(x, ids, w, layer, inners, None, 7.0)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1e3
    cnt = torch.bincount(ids.reshape(-1), minlength=N_EXP)
    return out, dt, int(cnt.max())

# 1) moderate skew so the hottest expert stays <= 192 rows: flag off == upstream fused kernel, flag on == MT kernel
out0, t0_, mx = run("0", 1200, 0.0); out1, t1_, _ = run("1", 1200, 0.0)
d = ((out1 - out0).abs().max() / out0.abs().max()).item()
print(f"tokens 1200, max rows {mx}: flag0 {t0_:.1f} ms  flag1 {t1_:.1f} ms  max rel diff {d:.1e}  {'OK' if d < 2e-2 and torch.isfinite(out1).all() else 'MISMATCH'}")
# 2) heavy skew: flag off uses the fat-expert python loop, which needs real LinearEXL3 -> only run flag on
out1, t1_, mx = run("1", 2044, 0.55)
print(f"tokens 2044, max rows {mx}: flag1 {t1_:.1f} ms  finite {torch.isfinite(out1).all().item()}  |out| max {out1.abs().max().item():.1f}")
# 3) tiny prefill tail (tokens > cap but few rows per expert): must take the upstream path (min rows guard)
out0, t0_, mx = run("0", 200, 0.0); out1, t1_, _ = run("1", 200, 0.0)
print(f"tokens 200, max rows {mx}: flag0 {t0_:.1f} ms flag1 {t1_:.1f} ms  identical {torch.equal(out0, out1)} (guard -> upstream)")
print("patched dispatch OK")
