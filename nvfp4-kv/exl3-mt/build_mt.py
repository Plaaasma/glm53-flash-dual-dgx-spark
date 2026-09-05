import os, sys, time, torch
from torch.utils.cpp_extension import load
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
here = os.path.dirname(os.path.abspath(__file__))
inc = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
t0 = time.time()
ext = load(name="glm53_exl3_mt", sources=[os.path.join(here, "glm53_exl3_mt.cu")],
           extra_include_paths=[inc],
           extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas", "-v", "-lineinfo", "--expt-relaxed-constexpr",
                              "-gencode=arch=compute_121a,code=sm_121a"],
           extra_cflags=["-O3", "-std=c++17"],
           build_directory=os.path.join(here, "build"), verbose=True)
print(f"built in {time.time()-t0:.0f}s; variants:", [ext.variant_name(i) for i in range(ext.num_variants())])
