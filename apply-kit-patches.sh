#!/bin/bash
# Clone the MiaAI-Lab kit at the tested commit and lay this repo's files over it. Result: ./exl3-kit (the kit,
# patched, with this repo's start.sh and patchers) and ./nvfp4-vllm (the NVFP4 KV pool, viz runtime and M-tiled
# MoE kernel sources), which is where start.sh looks for them (../nvfp4-vllm next to the kit; NVFP4_DIR overrides).
set -e
cd "$(dirname "$0")"
if [ ! -d exl3-kit ]; then
  git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git exl3-kit
fi
( cd exl3-kit && git checkout 493cb88 2>/dev/null || echo "note: pinned commit unavailable, using HEAD (re-verify anchors)" )
# 1) hunks against the kit's own overlay files (upstream at the pinned commit)
( cd exl3-kit && git apply --check ../kit-patches/exl3-kit.patch && git apply ../kit-patches/exl3-kit.patch ) \
  && echo "kit patches applied" || { echo "PATCH FAILED -- upstream drifted; see kit-patches/exl3-kit.patch hunks"; exit 1; }
# 2) this repo's launch script and patchers (the full start.sh; new patchers; identical copies of the patched ones)
cp kit-patches/start.sh exl3-kit/start.sh
cp kit-patches/patch_*.py kit-patches/shm_cleanup.py exl3-kit/overlay/
# 3) the NVFP4 KV pool, viz runtime and MoE kernel sources where start.sh expects them
mkdir -p nvfp4-vllm/exl3-mt
cp nvfp4-kv/patch_nvfp4_kv.py nvfp4-kv/glm53_nvfp4_runtime.py nvfp4-kv/glm53_viz_runtime.py nvfp4-vllm/
cp nvfp4-kv/exl3-mt/*.cu nvfp4-kv/exl3-mt/*.py nvfp4-kv/exl3-mt/*.sh nvfp4-vllm/exl3-mt/
[ -f exl3-kit/.env ] || cp kit-patches/env.example exl3-kit/.env
echo "now edit exl3-kit/.env (HF_TOKEN, HEAD_IP/WORKER_IP, NIC names, NCCL_IB_GID_INDEX);"
echo "optional: build the M-tiled MoE kernel with nvfp4-vllm/exl3-mt/build.sh, then cd exl3-kit && ./start.sh"
