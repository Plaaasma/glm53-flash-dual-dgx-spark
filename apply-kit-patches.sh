#!/bin/bash
# Clone the MiaAI-Lab kit at the tested commit and apply this repo's patches.
set -e
cd "$(dirname "$0")"
if [ ! -d exl3-kit ]; then
  git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git exl3-kit
fi
cd exl3-kit
git checkout 493cb88 2>/dev/null || echo "note: pinned commit unavailable, using HEAD (re-verify anchors)"
git apply --check ../kit-patches/exl3-kit.patch && git apply ../kit-patches/exl3-kit.patch \
  && echo "kit patches applied" || echo "PATCH FAILED -- upstream drifted; see kit-patches/exl3-kit.patch hunks"
cp ../kit-patches/env.example .env
echo "now edit exl3-kit/.env: HF_TOKEN, HEAD_IP/WORKER_IP, NIC names"
