# M-tiled EXL3 MoE kernel for prefill (`glm53_exl3_mt`)

## Why

exllamav3's `exl3_moe_kernel` and `exl3_gemm_kernel_inner` are tensor-core kernels (mma.sync m16n8k16 with the
trellis dequantized in registers), but their M tile is hard-wired to 16 rows. Every 16-row pass re-streams and
re-decodes the whole trellis weight, so during prefill an expert with R rows decodes its weights ceil(R/16)
times and the tensor cores idle behind the integer 3INST decode (~6 TFLOPS achieved). Experts with more rows
than the fused temps (`EXL3_TEMP_ROWS_FUSED`, 192 here) fall out of the fused kernel into a per-expert Python
loop that reconstructs each expert to fp16 and runs cuBLAS (exllamav3 `AUTO_RECONSTRUCT_THRESHOLD = 144`),
which pays 9x the packed-weight traffic.

Decode is unaffected by any of this: at 1 to 24 rows per expert the fused kernel is bandwidth-bound reading
each expert once, which is the physical floor.

## What

`glm53_exl3_mt.cu` is the upstream kernel with the inner GEMM generalised to `TILESIZE_M = 16 * TBM` rows per
pass: A fragments and C accumulators become `[TBM]` arrays, each dequantized B fragment feeds TBM MMAs, and the
row-predicated split-K reduction handles the partial last tile. Everything else (cp.async pipeline, XOR-swizzled
A tiles, lock-ordered split-K, Hadamard pre/post passes, group barriers) is unchanged, so the K=32 variants are
bit-exact with upstream. Six variants are compiled; variant 5 (`m64_k16_n256`, 6 pipeline stages, 2 fragment
stages, 256 threads, 255 registers) wins everywhere it matters.

Measured on synthetic 4-bit experts, hidden 4096, intermediate 2048 (TP=2 local), GPU shared with the live
server so absolute times are inflated; ratios hold (`bench_rows.txt`, `bench_skew.txt`):

| rows per expert | upstream | variant 5 | speedup |
|---|---|---|---|
| 4 to 16 | 5.5 to 5.9 ms | 6.1 to 7.8 ms | 0.9x (not used here, see MIN_ROWS) |
| 57 (live prefill mean) | 18.8 ms | 8.7 ms | 2.2x |
| 128 | 41.1 ms | 18.0 ms | 2.3x |
| 512 | 182 ms | 72.9 ms | 2.5x |

Realistic skewed layer, 2044 tokens x top-8 over 96 experts (mean 170 rows, max 865): the path vLLM runs today
(fused for <=192 rows + reconstruct/cuBLAS for 20 fat experts) 163 ms; variant 5 88 to 94 ms; achieved ~9 to 11
TFLOPS against 20 to 35 TFLOPS for dense fp16 cuBLAS at the same shapes.

Through vLLM's `apply_exl3_fused_moe` (`test_patched.py`): 1200 tokens, 181 ms -> 65 ms, max relative
difference 3e-4.

## Wiring

- `kit-patches/patch_exl3_mt.py` installs `/opt/glm53/glm53_exl3_mt.so` into site-packages and adds a
  prefill-only dispatch after the existing `max_rows` sync in `apply_exl3_fused_moe`. The CUDA-graph decode path
  never reaches the seam.
- `start.sh` mounts the patcher and the `.so` on both nodes and forwards the knobs.
- Knobs (`.env`): `GLM53_EXL3_MT=1` enables (default 0 = upstream, byte-for-byte), `GLM53_EXL3_MT_VARIANT=5`,
  `GLM53_EXL3_MT_TEMP_ROWS=1024` (144 MiB of temps per rank; hotter experts use the upstream loop),
  `GLM53_EXL3_MT_MIN_ROWS=32` (below this the upstream kernel is as fast).
- Build: `nvfp4-kv/exl3-mt/build.sh [image]` (45 s, CPU only). The image ships nvcc for CUDA 13 / sm_121a but
  not `cusparse.h`, so the extension uses `c10/cuda/CUDAStream.h` rather than `ATen/cuda/CUDAContext.h`.

## Tests

- `test_mt.py`: correctness for 1 to 600 rows against upstream `exl3_moe`, then the row sweep benchmark.
- `test_skew.py`: the realistic skewed layer and the cost of expert ordering (largest-first buys only ~3%).
- `test_patched.py`: applies the patcher inside a throwaway container and runs vLLM's own apply flag off/on.

Run each with `docker run --rm --runtime=nvidia --memory=5g -v $PWD:/w -w /w --entrypoint python3 <image> test_x.py`
(the memory cap keeps a runaway away from the host watchdog line when a server is live on the same node).

## Shared-memory-B variants and the auto kernel (2026-09-10)

Variants 6 and 7 decode every 16x16 trellis tile once per block into shared memory (fp16, `[n][k]` with a padded
row stride) and let eight warps consume it through `ldmatrix.x2` against their own 16- or 32-row tiles. At 128+
rows per expert they reach 33 to 39 TFLOPS, the dense-cuBLAS ceiling on this GPU, but they waste MMA work on
padding for small experts. Variant 8 ("auto") picks the inner per expert from its row count: the 64-row M-tiled
inner up to 96 rows, shared-memory-B with 128-row tiles up to 160, 256-row tiles above. On the real routing
distribution (2044 tokens x top-8 over 288 experts, mean 57 rows, max 566, GPU idle):

| path | ms per layer |
|---|---|
| vLLM today (fused <=192 rows + reconstruct/cuBLAS) | 75.1 |
| variant 5 (m64) | 43.5 |
| variant 8 (auto) | 38.0 |

`bench_rows_idle.txt` and `bench_skew288.txt` are the idle-GPU runs. Shipped default: `GLM53_EXL3_MT_VARIANT=8`.

## Next step (not built)

The kernel is still decode-bound at ~10 TFLOPS. The remaining 2x is a Marlin-style restructuring: decode each
16x16 B tile once per block into shared memory as fp16, give every warp its own 16-row M tile, and feed all warps
from the same decoded tile with ldmatrix. That reuses each decoded weight 8 to 16 times instead of 4 without the
register pressure that stopped TBM=8 here.
