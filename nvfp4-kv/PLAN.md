# NVFP4 KV for the EXL3/vLLM stack — plan + recon (2026-09-01)

GOAL: NVFP4 (288 B/token) main sparse-MLA KV pool, like our SGLang build.
Payoff at current 13.5 GiB pool: 656 B/tok -> 288 B/tok = **~2.65M tokens
(from 1.16M), or same tokens + ~7.6 GiB freed.**

## Architecture: gather-dequant (the SGLang v1 design, better fit here)
The prebuilt FLASHINFER_MLA_SPARSE_SM120 kernel cannot be patched, but it
doesn't need to be: decode converts topk indices to PER-TOKEN physical rows
(`triton_convert_req_index_to_global_index`) and passes them as block_tables
(block=1). Seam: dequant the selected NVFP4 rows -> fp8 scratch (656-B layout),
renumber indices to scratch rows, launch unchanged kernel on the scratch.
Our bit-exact kernels port directly:
  /home/liam/glm53/patches/nvfp4_gather.py  (quantize_to_nvfp4_triton,
  gather_dequant_nvfp4_to_fp8, arithmetic e2m1 decode)

## Recon facts (image glm53-flash-sm121:local-0831)
- Pool layout: vllm/v1/kv_cache_interface.py:438-445 — "fp8_ds_mla" =
  656 B/token (512 fp8 latent + 4B scale + rope region). GLM is NoPE
  (rope dim 0) — verify rope region unused -> scratch can skip it.
- Decode read: v1/attention/backends/mla/flashinfer_mla_sparse.py
  forward_mqa: topk_indices -> triton_convert_req_index_to_global_index ->
  block_tables = per-token physical rows. INSERTION POINT is here.
- cache dtypes list: vllm/config/cache.py:27; branch points at
  kv_cache_interface.py:438 and :678 (model_version branches).
- Prefill backends: v1/attention/backends/mla/prefill/{registry,selector}.py
  — must identify the active one + its pool read. UNKNOWN #1.
- Write op: not yet located (concat_and_cache_mla equivalent). UNKNOWN #2.
- Indexer kpool: separate pool (kit patches KpoolTailManager); stays fp8.
- DFlash2 drafter KV: separate, bf16/auto; untouched.
- CUDA-graph interplay: scratch buffer must be capture-safe (fixed shapes:
  seqs x topk=2048 x 656). UNKNOWN #3 (graph mode FULL_AND_PIECEWISE).

## Phases (each service-touching step needs an explicit OK from Liam)
1. [read-only] finish recon: write op, active prefill backend, graph shapes.
2. [no service] port nvfp4_gather.py; standalone GPU tests on spare memory
   (bit-exactness vs pool rows; scratch layout match incl. 4B scale word).
3. [no service] integration patch set in /home/liam/glm53/nvfp4-vllm/overlay/:
   new cache_dtype "nvfp4_ds_mla" (288 B), write-path quantize hook, decode
   gather-dequant + index renumber, prefill equivalent. Mount via start.sh
   pattern (proven) or bake with BUILD=1 (preferred after the mount lesson:
   only pure-Python overlay on SAME image = safe; ext untouched).
4. [restart, ask] bring up with KV_CACHE_DTYPE=nvfp4_ds_mla on a validation
   window: greedy fat-path probe, needle 50/100/150K, GSM8K n=100, loop test,
   A/B decode+prefill speed. Rollback = KV_CACHE_DTYPE=fp8.

## Risks called out up front
- Per-step dequant cost: topk 2048 x layers; SGLang measured negligible, but
  here it's per DSA layer per decode step -> measure, don't assume.
- The 4-byte inline scale in the 656 layout: our format stores per-16 e4m3
  scales; scratch must present whatever the kernel reads. Verify exact layout.
- Chunked prefill reads MANY tokens (not topk-sparse) -> dequant volume larger;
  may need the row-tile-style chunking. Measure.

## RESULT (2026-09-02): SHIPPED AND VALIDATED
Pool 1,163,471 (fp8, util .84) -> **1,431,147 tokens (nvfp4, util .82)**: +23%
capacity while RAISING idle margin to 7.0/6.7 GiB (was 4-5). GSM8K **99/100,
0 errors, 43.7 tok/s agg** (best of any config). Degeneration 0/6. Greedy
fat-probe content IDENTICAL to fp8-KV baseline. Decode 24.5-29.3 code / 15.8
prose (stock band); prefill 674-679 tok/s (~3% union-gather cost, as measured
standalone). Tool calls OK via Tailscale.

Two integration surprises worth remembering:
1. The executing backend was flashinfer_mla_sparse_sm120.py, NOT the sibling
   flashinfer_mla_sparse.py the recon read first -- boot traceback settled it.
2. The flashinfer wrapper routes decode-vs-paged partly on cache PAGE GEOMETRY:
   a [S,1,656] scratch (page=1) sent <=64-token steps to the paged kernel,
   which rejects them. Scratch must be [P,64,656] like the real pool.
DFlash2 padded slot-share accepted 288-byte pages with no geometry assert.
Rollback: GLM53_NVFP4_KV=0 in exl3-kit/.env (one restart).
