"""Build glm53_exl3_dec.so for GB10 (sm_121a) with the image's nvcc; run inside glm53-flash-sm121:local-0904-it:

    docker run --rm --network none --entrypoint "" -v "$PWD"/nvfp4-vllm/exl3-dec:/w \
        glm53-flash-sm121:local-0904-it python3 /w/build_dec.py

No GPU is needed. The result is /w/glm53_exl3_dec.so (copied out of /w/build).
"""
import os
import shutil
import sys
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
os.environ.setdefault("MAX_JOBS", "4")
from torch.utils.cpp_extension import load  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
t0 = time.time()
ext = load(
    name="glm53_exl3_dec",
    sources=[os.path.join(here, "csrc", "dec.cpp"), os.path.join(here, "csrc", "dec.cu")],
    extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--expt-relaxed-constexpr",
                       "-gencode=arch=compute_121a,code=sm_121a"]
                      + (["-Xptxas", "-v"] if os.environ.get("PTXAS_V") else []),
    extra_cflags=["-O3", "-std=c++17"],
    build_directory=build,
    verbose=bool(os.environ.get("VERBOSE")),
)
so = os.path.join(build, "glm53_exl3_dec.so")
shutil.copyfile(so, os.path.join(here, "glm53_exl3_dec.so"))
print(f"built {so} in {time.time() - t0:.0f}s -> {os.path.join(here, 'glm53_exl3_dec.so')}")
sys.exit(0)
