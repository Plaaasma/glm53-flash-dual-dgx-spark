"""Apply patch_mm_chunk.py, then process N cold images through the CACHED processor path with GLM53_MM_CHUNK
set by argv; print a fingerprint of every output (token ids, placeholders, processed tensors) and the memory peak."""
import os, sys, subprocess, hashlib, resource, time
chunk, N = sys.argv[1], int(sys.argv[2]); os.environ["GLM53_MM_CHUNK"] = chunk
r = subprocess.run([sys.executable, "/w/patch_mm_chunk.py"], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-400:]
from PIL import Image, ImageDraw
import torch
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
def shot(i):
    im = Image.new("RGB", (1920, 1080), (30 + 3 * i, 40, 60)); d = ImageDraw.Draw(im)
    for y in range(0, 1080, 30): d.text((20, y), f"shot {i} line {y}", fill=(220, 220, 220))
    return im
imgs = [shot(i) for i in range(N)]
mc = ModelConfig(model="/model", tokenizer="/model", trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 128, "video": 1}, mm_processor_kwargs={"max_image_tokens": 1024}, mm_processor_cache_gb=2)
cache = MultiModalProcessorOnlyCache(mc)
proc = MULTIMODAL_REGISTRY.create_processor(mc, cache=cache)
items = proc.info.parse_mm_data({"image": imgs})
r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; t0 = time.time()
out = proc.apply(ProcessorInputs(prompt="<|begin_of_image|><|image|><|end_of_image|>" * N + "Describe.", mm_data_items=items), TimingContext())
dt = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
h = hashlib.sha256(); h.update(repr(out["prompt_token_ids"]).encode())
h.update(repr([(p.offset, p.length) if hasattr(p, "offset") else repr(p) for p in out["mm_placeholders"]["image"]]).encode())
mmk = out["mm_kwargs"]
n_t = 0
for item in mmk["image"]:
    for k in sorted(item.keys()):
        h.update(k.encode()); d = item[k].data.contiguous().cpu(); d = d.view(torch.int16) if d.dtype == torch.bfloat16 else d; h.update(d.numpy().tobytes()); n_t += 1
detail = f"{n_t} tensors hashed"
print(f"chunk={chunk} N={N}: tokens {len(out['prompt_token_ids'])}, {dt:.2f}s, maxrss delta +{r1-r0:.0f} MiB, fingerprint {h.hexdigest()[:16]} ({detail})")
# second, warm call (all cached): should be cheap and identical
r2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; t0 = time.time()
try:
    out2 = proc.apply(ProcessorInputs(prompt="<|begin_of_image|><|image|><|end_of_image|>" * N + "Describe.", mm_data_items=proc.info.parse_mm_data({"image": imgs})), TimingContext())
except Exception as e:
    import traceback; print("  warm repeat FAILED:", type(e).__name__, str(e)[:200]); raise SystemExit(0)
print(f"  warm repeat: {time.time()-t0:.2f}s, same token ids {out2['prompt_token_ids'] == out['prompt_token_ids']}, maxrss delta +{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024-r2:.0f} MiB")
