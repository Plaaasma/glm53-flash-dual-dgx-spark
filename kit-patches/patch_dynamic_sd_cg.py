#!/usr/bin/env python3
"""Fix dynamic-SD CUDA-graph candidate query lengths for the MTP speculator.

vLLM's base ``CudaGraphManager._init_candidates`` derives per-range decode
query lengths under ``num_speculative_tokens_per_batch_size`` as
``K_range + (decode_query_len - num_speculative_tokens)``. That is right for
the target model (decode_query_len = K_max + 1) but the autoregressive/MTP
speculator reuses the same code for its draft-decode manager, whose
decode_query_len is 1 (one draft token per sequence per step regardless of
K). With K_max=3 the offset is -2, so a K=2 range yields query len 0 and
``round_up(num_tokens, 0)`` raises ZeroDivisionError (observed 2026-09-04;
upstream: vllm-project/vllm#48494).

Draft decode is 1 token/seq/step for every K, so clamping each derived
length to >= 1 and de-duplicating restores the intended candidate set for
all three managers: target {K_i+1}, draft-prefill {K_i+1}, draft-decode {1}.
Idempotent; fails closed if the anchor drifts.
"""
import os
import sys
from pathlib import Path

MARK = "# [glm53-dynamic-sd-cg]"
P = Path(os.environ.get(
    "GLM53_CUDAGRAPH_UTILS_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/cudagraph_utils.py"))

OLD = """            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
"""
NEW = """            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
            # Draft decode is one token per sequence per step for every K, so
            # the reconstruction above underflows to 0 for the speculator's
            # decode manager (decode_query_len=1). Clamp and de-duplicate.
            decode_query_lens = sorted({max(int(q), 1) for q in decode_query_lens})  """ + MARK + """
"""


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    n = text.count(OLD)
    if n != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {n} — refusing to patch")
    P.write_text(text.replace(OLD, NEW, 1))
    print(f"patched {P.name} (dynamic-SD cudagraph query lens clamped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
