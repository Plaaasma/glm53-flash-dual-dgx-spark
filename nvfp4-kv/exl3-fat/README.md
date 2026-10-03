# Fused EXL3 routed-expert kernels for prefill (`glm53_exl3_fat`)

## Why

- **Prefill is dominated by the experts.** In a 2,048-token prefill chunk the routed experts were 48% of the
  time: 28 ms per MoE layer through the M-tiled kernel (`exl3-mt/`, ~15 TFLOPS).
- **A faster kernel exists for this exact setup.** jayleaton's glm53-tensorfold-spark kit, running these same
  EXL3 weights on the same GB10 pair, does it at ~13 ms with TensorFold's "fat" / "fast2" grouped expert GEMMs.

This directory ports those kernels to vLLM's `exl3.py`.

## What changed from the source

The full list is in the header of `glm53_exl3_fat.cu`:
- **SwiGLU epilogue:** kept in fp32 throughout. TensorFold emulates bf16 roundings there; this port uses
  exllamav3 `exl3_moe`'s activation order instead.
- **Down projection:** writes each routed pair's output as an fp16 row into the dead rotated-input buffer. A small
  combine kernel then sums each token's slots in a fixed order. That replaces fp32 atomics: deterministic output,
  ~0.6 ms faster per layer, no extra scratch.
- **Default kernel:** "fast2" (4-stage gate/up, down covering two column blocks per item), ~1.2x faster than
  "fat" here.

## Results (real weights, one MoE layer, TP=2 rank 0)

| tokens | current path | current ms | new ms | speedup |
|---|---|---|---|---|
| 128 | exl3_moe | 8.31 | 6.90 | 1.20x |
| 256 | M-tiled v8 | 12.42 | 7.63 | 1.63x |
| 512 | M-tiled v8 | 14.53 | 8.50 | 1.71x |
| 1024 | M-tiled v8 | 17.35 | 9.80 | 1.77x |
| 2048 (Zipf routing) | M-tiled v8 | 21.73 | 11.73 | 1.85x |
| 2048 (two experts > 1024 rows) | M-tiled v8 + per-expert loop | 24.43 | 12.12 | 2.02x |

- **Error vs an fp64 CPU reference:** 2.0-2.3x lower than the current path (4.1e-4 against 8.4-9.1e-4 relative),
  bitwise deterministic. The current path rounds GEMM outputs and split-K partials to fp16 and is not deterministic.
- **In the server:** cold prefill went from 797 / 823 / 661 tok/s to 1,026 / 1,105 / 1,082 tok/s at 9.4K / 32.7K /
  77.8K tokens.
- **Quality checks:** GSM8K-100 98/100. Teacher-forced NLL unchanged within noise (1.6327 against 1.6341 before
  the kernel).
- **Memory:** +160 MiB shared scratch per rank, allocated at the first prefill call. The M-tiled kernel's 120 MiB of
  temps are then never allocated, so the net is ~+40 MiB.

## Files

| file | what |
|---|---|
| `glm53_exl3_fat.cu` | plan, rotation, fat / fast2 gate/up and down kernels, combine, bindings |
| `glm53_exl3_fat_rt.py` | runtime: `apply(x2d, ids, weights, layer, limit)`, same contract as `apply_exl3_fused_moe`'s prefill branch |
| `build_fat.py`, `build.sh` | build `glm53_exl3_fat.so` for sm_121a in the serving image (CPU only, ~35 s) |
| `test_fat.py` | patch + correctness + benchmark harness on real weights in a throwaway container |
| `LICENSES/`, `NOTICE` | Apache-2.0 (glm53-tensorfold-spark), MIT (TensorFold 0.3.4, ExLlamaV3) |

## Wiring

- `kit-patches/patch_exl3_fat.py` runs after `patch_exl3_mt.py`. It sends calls above
  `GLM53_EXL3_FAT_MIN_TOKENS` (64, the largest CUDA-graph capture size) to the new kernels.
- Calls above `GLM53_EXL3_FAT_MAX_TOKENS` (2048, sizes the scratch), or any init failure, fall back to the M-tiled
  path.
- The new path never runs under graph capture.
- `GLM53_EXL3_FAT=1` enables it. Keep `GLM53_EXL3_MT=1` as the fallback.
