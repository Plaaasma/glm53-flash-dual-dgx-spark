#!/bin/bash
# One-shot host preparation for EACH DGX Spark node. Run as a sudoer:
#   ./host-setup/setup-node.sh head      # on the node that runs start.sh (API server, rank 0)
#   ./host-setup/setup-node.sh worker [--autorecover user@<head fabric IP>]   # on the other node
# Steps 1-3 do not survive a reboot: re-run then. Steps 4-6 persist.
set -e
role="${1:-head}"
case "$role" in head|worker) ;; *) echo "usage: $0 head|worker [--autorecover user@<head fabric IP>]"; exit 1;; esac
autorecover=""
[ "${2:-}" = "--autorecover" ] && autorecover="${3:?--autorecover needs user@<head fabric IP>}"
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

# 5. Runtime reclaim from root's crontab: 2 GiB from the serving container every minute while MemAvailable is under
#    2.5 GiB. (An unconditional reclaim every 20 min used to run here too: on a node with a desktop session it ran
#    17-40 min per pass at 100% CPU, recovered nothing and slowed that node's GPU, so it was removed.)
cron_1="* * * * * [ \$(awk \"/MemAvailable/{print int(\\\$2/1024)}\" /proc/meminfo) -lt 2500 ] && /usr/local/sbin/glm53-reclaim $container 2 >> /var/tmp/glm53-reclaim.log 2>&1"
( sudo crontab -l 2>/dev/null | grep -v "glm53-reclaim" ; echo "$cron_1" ) | sudo crontab -

# 6. Optional, on the worker: glm53-autorecover.service restarts the pair through ssh to the head when a rank has died,
#    the engine hangs or /health fails for 5 min (never a healthy server; at most 3 restarts an hour; each crash's
#    container logs are saved on the head under exl3-kit/logs/crash-<time>/). Needs the kit laid out next to this repo
#    (./apply-kit-patches.sh) at the same path on both nodes and an ssh key from this node to the head.
if [ -n "$autorecover" ]; then
  kit="$(cd "$HERE/../exl3-kit" && pwd)"
  sed -e "s|@USER@|$USER|g" -e "s|@HEAD_SSH@|$autorecover|" -e "s|@KIT@|$kit|" "$HERE/glm53-autorecover.service" \
    | sudo tee /etc/systemd/system/glm53-autorecover.service > /dev/null
  sudo systemctl daemon-reload && sudo systemctl enable --now glm53-autorecover
  echo "glm53-autorecover enabled (log: $kit/logs/autorecover.log). Disable it before stopping the server on purpose:"
  echo "  sudo systemctl disable --now glm53-autorecover"
fi
echo "node ready ($role): zram zstd 10G + sysctls + watchdog armed + helpers installed + reclaim cron for $container"
