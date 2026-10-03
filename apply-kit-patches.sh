#!/bin/bash
# Clone the MiaAI-Lab kit at the tested commit and lay this repo's files over it. Result: ./exl3-kit (the kit,
# patched, with this repo's start.sh, patchers and scripts) and ./nvfp4-vllm (the NVFP4 KV pool, the dense 8-bit
# runtime, the viz/maintenance runtime and the three EXL3 MoE kernel sources), which is where start.sh looks for
# them (../nvfp4-vllm next to the kit; NVFP4_DIR overrides).
#   ./apply-kit-patches.sh            lay out the files
#   ./apply-kit-patches.sh --build    ... and build the three kernels (docker run of the kit image, CPU only)
set -e
cd "$(dirname "$0")"
if [ ! -d exl3-kit ]; then
  git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git exl3-kit
fi
( cd exl3-kit && git checkout 493cb88 2>/dev/null || echo "note: pinned commit unavailable, using HEAD (re-verify anchors)" )
# 1) hunks against the kit's own overlay files (upstream at the pinned commit)
( cd exl3-kit && git apply --check ../kit-patches/exl3-kit.patch && git apply ../kit-patches/exl3-kit.patch ) \
  && echo "kit patches applied" || { echo "PATCH FAILED -- upstream drifted; see kit-patches/exl3-kit.patch hunks"; exit 1; }
# 2) this repo's launch script, patchers and recovery script
cp kit-patches/start.sh exl3-kit/start.sh
cp kit-patches/patch_*.py kit-patches/shm_cleanup.py exl3-kit/overlay/
mkdir -p exl3-kit/scripts && cp host-setup/glm53-autorecover.sh exl3-kit/scripts/autorecover.sh
# 3) runtimes and kernel sources where start.sh expects them
mkdir -p nvfp4-vllm/exl3-mt nvfp4-vllm/exl3-dec nvfp4-vllm/exl3-fat
cp nvfp4-kv/patch_nvfp4_kv.py nvfp4-kv/glm53_nvfp4_runtime.py nvfp4-kv/glm53_viz_runtime.py \
   nvfp4-kv/glm53_dense_rt.py nvfp4-vllm/
cp nvfp4-kv/exl3-mt/*.cu nvfp4-kv/exl3-mt/*.py nvfp4-kv/exl3-mt/*.sh nvfp4-vllm/exl3-mt/
cp -r nvfp4-kv/exl3-dec/csrc nvfp4-kv/exl3-dec/*.py nvfp4-kv/exl3-dec/build.sh nvfp4-vllm/exl3-dec/
cp nvfp4-kv/exl3-fat/*.cu nvfp4-kv/exl3-fat/*.py nvfp4-kv/exl3-fat/build.sh nvfp4-vllm/exl3-fat/
[ -f exl3-kit/.env ] || cp kit-patches/env.example exl3-kit/.env
if [ "${1:-}" = "--build" ]; then
  for k in mt dec fat; do nvfp4-vllm/exl3-$k/build.sh; done
fi
echo "now edit exl3-kit/.env (HF_TOKEN, HEAD_IP/WORKER_IP, NIC names, HEAD_GID/WORKER_GID);"
echo "build the kernels once (./apply-kit-patches.sh --build, or nvfp4-vllm/exl3-{mt,dec,fat}/build.sh): without a"
echo ".so its patcher skips itself and the stock kernel runs. Then: cd exl3-kit && ./start.sh"
