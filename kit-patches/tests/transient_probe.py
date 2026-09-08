"""Peak host memory of a COLD multimodal request as a function of image count, at max_image_tokens=1024.
One process per N so maxrss is a clean high-water mark."""
import sys, time, resource, os
from PIL import Image, ImageDraw
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
N = int(sys.argv[1])
def shot(i):
    im = Image.new("RGB", (1920, 1080), (30 + 3 * i, 40, 60)); d = ImageDraw.Draw(im)
    for y in range(0, 1080, 30): d.text((20, y), f"shot {i} line {y}", fill=(220, 220, 220))
    return im
imgs = [shot(i) for i in range(N)]
mc = ModelConfig(model="/model", tokenizer="/model", trust_remote_code=True, dtype="bfloat16", limit_mm_per_prompt={"image": 128, "video": 1}, mm_processor_kwargs={"max_image_tokens": 1024})
proc = MULTIMODAL_REGISTRY.create_processor(mc)
items = proc.info.parse_mm_data({"image": imgs})
r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; t0 = time.time()
out = proc.apply(ProcessorInputs(prompt="<|begin_of_image|><|image|><|end_of_image|>" * N + "Describe.", mm_data_items=items), TimingContext())
dt = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"N={N:3d}: {len(out['prompt_token_ids'])} prompt tokens, {dt:.2f}s, maxrss delta +{r1-r0:.0f} MiB ({(r1-r0)/N:.0f} MiB/image)")
