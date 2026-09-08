"""Tail-chunk cases: N not a multiple of the chunk. Fingerprint chunked vs single-call for N in (6, 130) tiny images."""
import os, sys, subprocess, hashlib
chunk = sys.argv[1]; os.environ["GLM53_MM_CHUNK"] = chunk
r = subprocess.run([sys.executable, "/w/patch_mm_chunk.py"], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
from PIL import Image
import torch
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
mc = ModelConfig(model="/model", tokenizer="/model", trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 256, "video": 1}, mm_processor_kwargs={"max_image_tokens": 1024}, mm_processor_cache_gb=1)
for N in (6, 130):
    proc = MULTIMODAL_REGISTRY.create_processor(mc, cache=MultiModalProcessorOnlyCache(mc))
    imgs = [Image.new("RGB", (64 + 8 * (i % 5), 48), (i % 256, (i * 7) % 256, 90)) for i in range(N)]
    out = proc.apply(ProcessorInputs(prompt="<|begin_of_image|><|image|><|end_of_image|>" * N + "x", mm_data_items=proc.info.parse_mm_data({"image": imgs})), TimingContext())
    h = hashlib.sha256(repr(out["prompt_token_ids"]).encode())
    for item in out["mm_kwargs"]["image"]:
        for k in sorted(item.keys()):
            d = item[k].data.contiguous().cpu(); d = d.view(torch.int16) if d.dtype == torch.bfloat16 else d; h.update(d.numpy().tobytes())
    print(f"chunk={chunk} N={N}: {len(out['prompt_token_ids'])} tokens, {len(out['mm_placeholders']['image'])} placeholders, fingerprint {h.hexdigest()[:16]}")
