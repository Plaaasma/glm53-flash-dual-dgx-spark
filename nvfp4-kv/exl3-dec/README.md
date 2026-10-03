# Grouped EXL3 routed-expert kernel for decode (`glm53_exl3_dec`)

## Why

A decode step (MTP k=3: a 4-row verify window plus three 1-row draft steps) spent ~55 ms of a ~150 ms step in
exllamav3's `exl3_moe`. That kernel walks the routed experts with 48 CTAs (6 groups of 8 SMs), and in the server it
reaches only ~110-160 GB/s. GB10 streams ~230 GB/s. TensorFold v0.6.0 has a grouped EXL3 GEMV that reads each
**distinct** routed expert of the window once, spread over the whole GPU. This directory is that kernel, ported to
vLLM's `exl3.py` and re-tuned for GB10, with the 128-bit load ring from jayleaton's glm53-tensorfold-spark kit
(patch 0580).

## Pipeline (one call, 6 launches on the current stream, no host sync, grids depend only on T)

1. `group`: distinct experts (in id order) and each one's member pairs from vLLM's top-k ids. Ids outside
   [0, E) add exactly 0, like `map_topk_to_local`'s non-local sentinel.
2. `rot_in`: per routed (row, slot), Xg / Xu = fp16((x · suh) H128 / √128).
3. `grouped(gate, up)`: one program per (distinct expert, n-block, K-split, 16-member tile). Each expert's 4-bit
   trellis is read once and the mcg decode goes straight into `mma.m16n8k16` B fragments. Warps are summed in a
   fixed order, with no atomics.
4. gate/up epilogue: splits in order, FWHT, × svh, then fp32 SwiGLU in `exl3_moe`'s order
   (`min(silu(g), L) · clamp(u, −L, L)`), × suh_d, FWHT, fp16.
5. `grouped(down)`: the same as step 3.
6. down epilogue + combine: FWHT, × svh_d, then `out[t] = Σ_k w[t,k] · y_k` in slot order.

Tile settings are fixed per matrix shape, not per T. So every output's reduction order is the same at any window
size, and a row's result never depends on the other rows in its window.

## Results (real weights, one MoE layer, TP=2 rank 0)

**Accuracy against an fp64 CPU reference.**
- It is ~2.5x closer to fp64 than the current path at every T: 3.1-4.0e-4 against 7.4e-4 to 1.1e-3 max relative
  error. `exl3_moe` rounds the GEMM outputs, the SwiGLU and the routing weights to fp16.
- Bitwise deterministic.
- CUDA-graph replays with new routing equal eager, bitwise.
- The trellis decode equals exllamav3's `reconstruct`, bit for bit.

**Speed** (CUDA graphs, 64 real experts of one layer, consecutive calls kept out of L2):

| T (rows) | current ms | new ms | new GB/s over distinct experts | speedup |
|---|---|---|---|---|
| 1 | 0.357 | 0.221 | 228 | 1.62x |
| 4 | 0.726 | 0.585 | 233 | 1.24x |
| 8 | 0.955 | 0.803 | 227 | 1.19x |
| 16 | 1.762 | 1.476 | 237 | 1.19x |
| 64 | 2.140 | 1.900 | 212 | 1.13x |

**In the server** (MTP k=3, one stream), the experts went from ~55 ms to ~30 ms a step. That is ~43 ms on a node
that also runs a desktop session, which takes GPU time slices. Decode went from 28.6 to 31.7 tok/s on top of the
8-bit dense weights. GSM8K-100: 99/100.

**Memory:** +17 MiB per rank, one scratch shared by all layers. The per-layer state is views of the layer's own
tensors.

## Files

| file | what |
|---|---|
| `csrc/dec_grouped.cuh` | the grouped GEMV (`grouped_kernel`, 32-bit loads; `grouped_ld_kernel`, 128-bit load ring), mcg decode |
| `csrc/dec.cu`, `csrc/dec.cpp` | grouping, rotations, epilogues, combine, launch sequence, bindings |
| `glm53_exl3_dec_rt.py` | runtime: `prepare_layer(layer)`, `decode_moe(x2d, ids, weights, layer, limit)` |
| `build_dec.py`, `build.sh` | build `glm53_exl3_dec.so` for sm_121a in the serving image (CPU only, ~15 s) |
| `tests/` | correctness vs the current path vs fp64, 128-bit == 32-bit bits, the boot patcher end to end, benchmarks |
| `LICENSES/`, `NOTICE` | Apache-2.0 (TensorFold, glm53-tensorfold-spark), MIT (ExLlamaV3) |

## Wiring

- `kit-patches/patch_exl3_decode.py` seams vLLM's `exl3.py`. It prepares the per-layer state at
  `build_exl3_fused_state` and sends windows of up to `GLM53_EXL3_DEC_MAX_ROWS` (64) tokens to the new path.
- Anything else stays on the existing code: prefill, the flag off, or an init or launch failure (logged once).
- `GLM53_EXL3_DEC=1` enables it. `GLM53_EXL3_DEC_ACT=1` switches to the Python loop's activation order.
- `start.sh` mounts the patcher, the `.so` and the runtime on both nodes.
