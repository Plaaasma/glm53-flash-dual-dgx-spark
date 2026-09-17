#!/usr/bin/env python3
"""Size the sparse-MLA indexer's persistent K-gather workspace for this model, not DeepSeek-V3.2's.

`get_max_prefill_buffer_size` returns max_model_len * 40 entries of 132 bytes: 4.75 GB at max_model_len 900K.
The 40 is documented as a fit to flashmla_sparse's workspace for DeepSeek-V3.2 at 163K context. On this stack
the indexer keys are kpool-compressed 4:1, so one 900K request needs 225K entries and sixteen of them (the
concurrency limit) need 3.6M = max_model_len * 4 (475 MB). The chunker already splits prefill batches whose
summed compressed length exceeds the workspace, so a smaller buffer only means more chunks in the rare
multi-request prefill step. GLM53_INDEXER_PREFILL_MULT (default 40 = upstream) sets the multiplier.
"""
import os
from pathlib import Path

P = Path(os.environ.get("GLM53_INDEXER_PY", "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"))
MARK = "[glm53-indexer-buffer]"
OLD = "    return max_model_len * 40\n"
NEW = ("    import os as _g53os  # [glm53-indexer-buffer]\n"
       "    try:\n"
       "        _mult = float(_g53os.environ.get(\"GLM53_INDEXER_PREFILL_MULT\", \"40\"))\n"
       "    except ValueError:\n"
       "        _mult = 40.0\n"
       "    return int(max_model_len * _mult)\n")


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present - skipping")
        return 0
    if text.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {text.count(OLD)} - refusing")
    text = text.replace(OLD, NEW, 1)
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"patched {P.name}: indexer prefill buffer sized by GLM53_INDEXER_PREFILL_MULT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
