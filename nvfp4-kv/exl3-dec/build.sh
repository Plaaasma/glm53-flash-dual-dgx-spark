#!/bin/bash
# Build glm53_exl3_dec.so for the serving image (needs the image's nvcc/torch; CPU only, no GPU needed).
# Usage: nvfp4-kv/exl3-dec/build.sh [IMAGE] -> writes ./glm53_exl3_dec.so next to this script.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE="${1:-glm53-flash-sm121:local-0904-it}"
mkdir -p "$HERE/build"
docker run --rm --memory=6g --cpus=8 --network none -v "$HERE:/w" -w /w -e TORCH_CUDA_ARCH_LIST=12.1a --entrypoint python3 "$IMAGE" build_dec.py
echo "built $HERE/glm53_exl3_dec.so (start.sh looks for it in ../nvfp4-vllm/exl3-dec/ next to the kit)"
