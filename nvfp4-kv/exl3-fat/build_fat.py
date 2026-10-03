"""Build glm53_exl3_fat.so for sm_121a inside glm53-flash-sm121:local-0904-it (no GPU needed).

  docker run --rm -v "$PWD"/nvfp4-vllm/exl3-fat:/w --entrypoint "" glm53-flash-sm121:local-0904-it \
      python3 /w/build_fat.py
"""
import os
import shutil
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
os.environ.setdefault("MAX_JOBS", "2")
from torch.utils.cpp_extension import load  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, "build")
os.makedirs(build, exist_ok=True)
t0 = time.time()
ext = load(
    name="glm53_exl3_fat",
    sources=[os.path.join(here, "glm53_exl3_fat.cu")],
    extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas", "-v", "-lineinfo", "--expt-relaxed-constexpr",
                       "-gencode=arch=compute_121a,code=sm_121a"],
    extra_cflags=["-O3", "-std=c++17"],
    build_directory=build,
    verbose=True,
)
shutil.copyfile(os.path.join(build, "glm53_exl3_fat.so"), os.path.join(here, "glm53_exl3_fat.so"))
print(f"built in {time.time() - t0:.0f}s: gate/up kernels {ext.kernel_names(False)}, down kernels {ext.kernel_names(True)}")
