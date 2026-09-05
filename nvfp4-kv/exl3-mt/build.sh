#!/bin/bash
# Build glm53_exl3_mt.so for the serving image (needs the image's nvcc/torch; ~45 s, CPU only, no GPU needed).
# Usage: nvfp4-kv/exl3-mt/build.sh [IMAGE] -> writes ./glm53_exl3_mt.so next to this script.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE="${1:-glm53-flash-sm121:local-0904-it}"
mkdir -p "$HERE/build"
docker run --rm --memory=6g --cpus=8 -v "$HERE:/w" -w /w -e TORCH_CUDA_ARCH_LIST=12.1a --entrypoint python3 "$IMAGE" build_mt.py
cp "$HERE/build/glm53_exl3_mt.so" "$HERE/glm53_exl3_mt.so"
echo "built $HERE/glm53_exl3_mt.so (point EXL3MT_SO_HOST at it, or copy to /home/liam/glm53/nvfp4-vllm/exl3-mt/)"
