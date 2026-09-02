#!/bin/bash
# One-shot host preparation for EACH DGX Spark node. Run as a sudoer.
# Everything here is required and none of it survives a reboot -- re-run then.
set -e

# 1. Fast compressed swap. The stock 16G /swap.img FILE swap is too slow to
#    absorb reclaim spikes on this UMA box -- reclaim stalls become full-node
#    livelocks (no OOM kill ever fires; only a power cycle recovers).
sudo modprobe zram num_devices=1 || true
if ! swapon --show | grep -q zram0; then
  echo 6G | sudo tee /sys/block/zram0/disksize > /dev/null
  sudo mkswap /dev/zram0 > /dev/null
  sudo swapon -p 100 /dev/zram0
fi

# 2. Proactive reclaim. swappiness biases toward the (fast, compressed) zram;
#    watermark_scale_factor=100 (1%) wakes kswapd early WITHOUT walling off
#    memory: 500 (5%) silently subtracts ~12 GiB from MemAvailable and makes
#    vLLM's startup free-memory check fail -- looks exactly like a leak.
sudo sysctl -w vm.swappiness=180 vm.watermark_scale_factor=100

# 3. Memory watchdog: SIGTERM (clean CUDA teardown), then SIGKILL, if
#    MemAvailable drops under 0.75 GiB. A raw SIGKILL mid-CUDA leaks ~20 GiB
#    of driver memory that only returns after minutes of quiesce (or reboot).
pkill -f "vllm-watch""dog" 2>/dev/null || true
setsid nohup "$(dirname "$0")/vllm-watchdog.sh" >/dev/null 2>&1 &
echo "node ready: zram + sysctls + watchdog armed"
