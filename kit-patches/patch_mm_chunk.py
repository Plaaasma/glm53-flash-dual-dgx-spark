#!/usr/bin/env python3
"""Bound the host-memory peak of multimodal preprocessing by running the HF image processor in chunks.

`BaseMultiModalProcessor._cached_apply_hf_processor` hands every cache-missing image of a request to the HF
processor in ONE call. Preprocessing a 1080p screenshot takes ~200 MB of transient host RAM (measured), so a
cold request carrying N images peaks at N x that, which on this 121 GiB unified-memory box competes with the
serving processes' headroom (the head node's watchdog killed vLLM at 3 s into a 33-image cold prefill).

This patch processes the missing IMAGE items in chunks of GLM53_MM_CHUNK (default 4) and concatenates the
per-chunk `BatchFeature` outputs along dim 0 (the layout the single-call path produces: `pixel_values` is a
flat [sum patches, D] tensor, `image_grid_thw` is [n, 3]). Requests with videos, or with fewer missing images
than one chunk, use the original single call. Only active when the processor cache is enabled (that is the
path with a missing-items split); the uncached path is left alone.

Fail closed if the vLLM seam drifts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_MM_PROCESSOR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/processor.py",
    )
)
MARK = "# [glm53-mm-chunk]"

SEAM_OLD = """        with timing_ctx.record("apply_hf_processor"):
            (
                prompt_ids,
                mm_missing_processed_data,
                is_update_applied,
            ) = self._apply_hf_processor_main(
                prompt=inputs.prompt,
                mm_items=mm_missing_data_items,
                hf_processor_mm_kwargs=inputs.hf_processor_mm_kwargs,
                tokenization_kwargs=inputs.tokenization_kwargs,
                enable_hf_prompt_update=False,
            )
"""
SEAM_NEW = """        with timing_ctx.record("apply_hf_processor"):
            _chunked = self._glm53_apply_hf_processor_chunked(  # [glm53-mm-chunk]
                prompt=inputs.prompt,
                mm_items=mm_missing_data_items,
                hf_processor_mm_kwargs=inputs.hf_processor_mm_kwargs,
                tokenization_kwargs=inputs.tokenization_kwargs,
            )
            if _chunked is not None:
                prompt_ids, mm_missing_processed_data, is_update_applied = _chunked
            else:
                (
                    prompt_ids,
                    mm_missing_processed_data,
                    is_update_applied,
                ) = self._apply_hf_processor_main(
                    prompt=inputs.prompt,
                    mm_items=mm_missing_data_items,
                    hf_processor_mm_kwargs=inputs.hf_processor_mm_kwargs,
                    tokenization_kwargs=inputs.tokenization_kwargs,
                    enable_hf_prompt_update=False,
                )
"""

METHOD_ANCHOR = "    def _cached_apply_hf_processor(\n"
METHOD = '''    def _glm53_apply_hf_processor_chunked(  # [glm53-mm-chunk]
        self,
        prompt,
        mm_items,
        hf_processor_mm_kwargs,
        tokenization_kwargs,
    ):
        """Image-only, many-item requests: run the HF processor per chunk of GLM53_MM_CHUNK images and
        concatenate. Returns None to fall back to the single-call path."""
        import os as _os

        import torch as _torch

        try:
            chunk = int(_os.environ.get("GLM53_MM_CHUNK", "4"))
        except ValueError:
            chunk = 4
        if chunk <= 0:
            return None
        counts = mm_items.get_all_counts()
        n_img = counts.get("image", 0)
        if n_img <= chunk or any(c > 0 for k, c in counts.items() if k != "image"):
            return None
        image_items = mm_items["image"]
        if isinstance(prompt, str):
            prompt_ids = self._apply_hf_processor_text_only(prompt, tokenization_kwargs)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)
        parts = []
        for start in range(0, n_img, chunk):
            sub = self.info.parse_mm_data(
                {"image": [image_items.get(i) for i in range(start, min(start + chunk, n_img))]},
                validate=False,
            )
            parts.append(
                self._apply_hf_processor_mm_only(
                    mm_items=sub,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                )
            )
        # Only multimodal fields are per-item along dim 0 (batched: [n, ...]; flat: [sum patches, ...]).
        # Anything else in the BatchFeature comes from the dummy text (e.g. attention_mask [1, n]) and is
        # ignored downstream by from_hf_inputs, so keep the first chunk's value instead of concatenating.
        mm_fields = set(self._get_mm_fields_config(parts[0], hf_processor_mm_kwargs).keys())
        merged = type(parts[0])()
        for key in parts[0].keys():
            vals = [p[key] for p in parts]
            if key in mm_fields and all(isinstance(v, _torch.Tensor) for v in vals):
                merged[key] = _torch.cat(vals, dim=0)
            elif key in mm_fields and all(isinstance(v, list) for v in vals):
                merged[key] = [x for v in vals for x in v]
            else:
                merged[key] = vals[0]
        return prompt_ids, merged, False

'''


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    for label, needle in (("cached-apply seam", SEAM_OLD), ("method anchor", METHOD_ANCHOR)):
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: expected exactly one {label}, found {text.count(needle)} (upstream drift)")
    for needed in ("def _apply_hf_processor_text_only(", "def _apply_hf_processor_tokens_only(", "def _apply_hf_processor_mm_only("):
        if needed not in text:
            raise SystemExit(f"{P}: expected '{needed}' (upstream drift)")
    text = text.replace(SEAM_OLD, SEAM_NEW, 1)
    text = text.replace(METHOD_ANCHOR, METHOD + METHOD_ANCHOR, 1)
    P.write_text(text)
    print(f"patched {P.name}: HF image preprocessing runs in chunks of GLM53_MM_CHUNK (bounded host-memory peak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
