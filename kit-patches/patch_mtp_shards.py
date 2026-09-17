#!/usr/bin/env python3
"""Load the MTP draft model from the shards that hold it, not the whole checkpoint.

The draft (Glm5NextMTP, one layer at index num_hidden_layers) lives in the same checkpoint as the target. vLLM's
loader enumerates every *.safetensors file for it, so InstantTensor streamed all 92 shards (164 GB, 45 s of NVMe
and a second post-load transient) to pick 3,481 tensors that live in 4 shards (8.2 GB). 2026-09-16: boot 32
showed the second "Loading weights took" at 45 s vs 35 s for the entire target model.

What: when `get_all_weights` is asked for a model whose class name contains "MTP", record the weight-name
prefixes it can use (`model.layers.<num_hidden_layers + i>.` for each MTP layer, with the multimodal
`model.language_model.` prefix normalised) and, in `_prepare_weights`, keep only the shards whose index entries
start with one of them. Any other model is untouched. Fails closed on anchor drift.
"""
import os
from pathlib import Path

P = Path(os.environ.get("GLM53_DEFAULT_LOADER_PY",
                        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/default_loader.py"))
MARK = "[glm53-mtp-shards]"

HELPER = '''
def _glm53_mtp_prefixes(model, model_config):  # [glm53-mtp-shards]
    """Weight-name prefixes an MTP draft model consumes, or None for any other model."""
    try:
        if "MTP" not in type(model).__name__:
            return None
        cfg = model_config.hf_config
        cfg = getattr(cfg, "text_config", None) or cfg
        n_hidden = int(getattr(cfg, "num_hidden_layers"))
        n_mtp = int(getattr(cfg, "num_nextn_predict_layers", 1) or 1)
        return tuple(f"model.layers.{n_hidden + i}." for i in range(n_mtp))
    except Exception:
        return None


def _glm53_filter_shards(hf_folder, files, prefixes):  # [glm53-mtp-shards]
    """Keep the safetensors shards whose index entries start with one of `prefixes`."""
    if not prefixes or not files:
        return files
    try:
        import json as _json
        with open(os.path.join(hf_folder, "model.safetensors.index.json")) as f:
            wm = _json.load(f)["weight_map"]
        keep = set()
        for name, fname in wm.items():
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]
            if name.startswith(prefixes):
                keep.add(fname)
        kept = [p for p in files if os.path.basename(p) in keep]
        if kept:
            logger.info("[glm53-mtp-shards] draft model: loading %d of %d shards (%s)", len(kept), len(files),
                        ", ".join(sorted(os.path.basename(p) for p in kept)))
            return kept
    except Exception as e:  # noqa: BLE001
        logger.warning("[glm53-mtp-shards] shard filter skipped: %r", e)
    return files

'''
SEAM_PREP_OLD = "        return hf_folder, hf_weights_files, use_safetensors\n"
SEAM_PREP_NEW = ("        hf_weights_files = _glm53_filter_shards(hf_folder, hf_weights_files,\n"
                 "                                                getattr(self, \"_glm53_keep_prefixes\", None))  # [glm53-mtp-shards]\n"
                 + SEAM_PREP_OLD)
SEAM_ALL_OLD = "        primary_weights = DefaultModelLoader.Source(\n            model_config.model,\n"
SEAM_ALL_NEW = ("        self._glm53_keep_prefixes = _glm53_mtp_prefixes(model, model_config)  # [glm53-mtp-shards]\n"
                + SEAM_ALL_OLD)
CLASS_ANCHOR = "\nclass DefaultModelLoader(BaseModelLoader):\n"


def main() -> int:
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present - skipping")
        return 0
    for needle in (SEAM_PREP_OLD, SEAM_ALL_OLD, CLASS_ANCHOR):
        n = text.count(needle)
        if n != 1:
            raise SystemExit(f"{P}: expected exactly one anchor ({needle[:50]!r}), found {n} - refusing")
    if "\nimport os\n" not in text and "\nimport os," not in text:
        text = text.replace("\nimport glob\n", "\nimport glob\nimport os\n", 1)
    text = text.replace(CLASS_ANCHOR, HELPER + CLASS_ANCHOR, 1)
    text = text.replace(SEAM_PREP_OLD, SEAM_PREP_NEW, 1).replace(SEAM_ALL_OLD, SEAM_ALL_NEW, 1)
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"patched {P.name}: MTP draft loads only its shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
