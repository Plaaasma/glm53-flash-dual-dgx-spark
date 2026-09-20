#!/bin/bash
# One-shot host preparation for EACH DGX Spark node. Run as a sudoer:
#   ./host-setup/setup-node.sh head      # on the node that runs start.sh (API server, rank 0)
#   ./host-setup/setup-node.sh worker    # on the other node
# Steps 1-3 do not survive a reboot: re-run then. Steps 4-5 persist.
set -e
role="${1:-head}"
case "$role" in head|worker) ;; *) echo "usage: $0 head|worker"; exit 1;; esac
container="glm53-exl3-$role"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1. Fast compressed swap. The stock 16G /swap.img FILE swap is too slow to absorb reclaim spikes on this
#    UMA box: reclaim stalls become full-node livelocks (no OOM kill ever fires; only a power cycle recovers).
#    zstd compresses the serving processes' cold pages 3.6-4.3:1 (lzo-rle: 2.9:1); 10 GiB keeps the periodic
#    reclaims from spilling into the file swap. start.sh re-applies the same settings at every boot.
sudo modprobe zram num_devices=1 || true
if ! swapon --show | grep -q zram0; then
  echo zstd | sudo tee /sys/block/zram0/comp_algorithm > /dev/null || true
  echo 10G | sudo tee /sys/block/zram0/disksize > /dev/null
  sudo mkswap /dev/zram0 > /dev/null
  sudo swapon -p 100 /dev/zram0
fi

# 2. Proactive reclaim. swappiness biases toward the (fast, compressed) zram; watermark_scale_factor=100 (1%)
#    wakes kswapd early WITHOUT walling off memory: 500 (5%) silently subtracts ~12 GiB from MemAvailable and
#    makes vLLM's startup free-memory check fail, which looks exactly like a leak.
sudo sysctl -w vm.swappiness=180 vm.watermark_scale_factor=100

# 3. Memory watchdog: SIGTERM (clean CUDA teardown), then SIGKILL, if MemAvailable drops under 0.75 GiB. A raw
#    SIGKILL mid-CUDA leaks ~20 GiB of driver memory that only returns after minutes of quiesce (or a reboot).
pkill -f "vllm-watch""dog" 2>/dev/null || true
setsid nohup "$HERE/vllm-watchdog.sh" >/dev/null 2>&1 &

# 4. Root helpers start.sh calls through sudo -n: cgroup reclaim (push cold pages to zram) and zram re-tune.
for h in glm53-reclaim glm53-zram; do
  sudo install -o root -g root -m 0755 "$HERE/$h" /usr/local/sbin/$h
  echo "$USER ALL=(root) NOPASSWD: /usr/local/sbin/$h" | sudo tee /etc/sudoers.d/$h > /dev/null
  sudo chmod 0440 /etc/sudoers.d/$h
done

# 5. Runtime reclaim from root's crontab: 2 GiB from the serving container every 20 min, and every minute while
#    MemAvailable is under 2.5 GiB. Both nodes creep up ~0.17 GiB/h under deep-context serving otherwise.
cron_20="*/20 * * * * /usr/local/sbin/glm53-reclaim $container 2 >> /var/tmp/glm53-reclaim.log 2>&1"
cron_1="* * * * * [ \$(awk \"/MemAvailable/{print int(\\\$2/1024)}\" /proc/meminfo) -lt 2500 ] && /usr/local/sbin/glm53-reclaim $container 2 >> /var/tmp/glm53-reclaim.log 2>&1"
( sudo crontab -l 2>/dev/null | grep -v "glm53-reclaim" ; echo "$cron_20"; echo "$cron_1" ) | sudo crontab -
echo "node ready ($role): zram zstd 10G + sysctls + watchdog armed + helpers installed + reclaim crons for $container"
