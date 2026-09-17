#!/bin/bash
# Watch Liam's `engined` on BOTH Sparks (since 2026-09-09 it runs split across the two nodes; Liam's cap: never
# above 8 GB in total, so 4 GB per node). Cron (head, user liam): */5. Logs both, desktop-notifies (60-min
# cooldown) and writes an alert file the dashboard shows while either node or the total is over its cap.
WORKER="${WORKER:-169.254.152.37}"
CAP_NODE_MIB="${ENGINED_CAP_NODE_MIB:-6144}"
CAP_TOTAL_MIB="${ENGINED_CAP_TOTAL_MIB:-12288}"
DIR=/home/liam/cluster-dashboard
LOG=$DIR/engined_watch.log
ALERT=$DIR/engined_alert.json
STAMP=/tmp/engined_watch.notified
# Footprint = host RSS + GPU memory the driver attributes to the process (engined holds ~1.9 GB of GPU memory on
# each node, 2026-09-17, invisible to RSS). Both are unified memory here, so both count against vLLM's headroom.
eng_mib() { local rss gpu; rss=$(ps -o rss= -C engined | awk '{s+=$1} END{printf "%d", s/1024}'); gpu=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1 ~ /engined/ {s+=$2} END{printf "%d", s}'); echo $(( ${rss:-0} + ${gpu:-0} )); }
head_mib=$(eng_mib)
worker_mib=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$(declare -f eng_mib); eng_mib" 2>/dev/null)
now=$(date '+%Y-%m-%d %H:%M:%S')
[ -z "$worker_mib" ] && worker_mib=-1
echo "$now,head=$head_mib,worker=$worker_mib" >> "$LOG"
total=$(( head_mib + (worker_mib > 0 ? worker_mib : 0) ))
over=""
[ "$head_mib" -gt "$CAP_NODE_MIB" ] && over="head ${head_mib} MiB > ${CAP_NODE_MIB}"
[ "$worker_mib" -gt "$CAP_NODE_MIB" ] && over="${over:+$over; }worker ${worker_mib} MiB > ${CAP_NODE_MIB}"
[ "$total" -gt "$CAP_TOTAL_MIB" ] && over="${over:+$over; }total ${total} MiB > ${CAP_TOTAL_MIB}"
if [ -n "$over" ]; then
    printf '{"process":"engined","head_mib":%d,"worker_mib":%d,"total_mib":%d,"cap_node_mib":%d,"cap_total_mib":%d,"above":true,"detail":"%s","at":"%s"}\n' \
        "$head_mib" "$worker_mib" "$total" "$CAP_NODE_MIB" "$CAP_TOTAL_MIB" "$over" "$now" > "$ALERT"
    if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(stat -c %Y "$STAMP") )) -gt 3600 ]; then
        DISPLAY=:1 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus XDG_RUNTIME_DIR=/run/user/1000 \
            notify-send -u critical -t 0 "engined over its cap" "$over. vLLM's headroom on that node is what it competes with." 2>/dev/null || true
        touch "$STAMP"
    fi
else
    [ -f "$ALERT" ] && printf '{"process":"engined","head_mib":%d,"worker_mib":%d,"total_mib":%d,"cap_node_mib":%d,"cap_total_mib":%d,"above":false,"at":"%s"}\n' \
        "$head_mib" "$worker_mib" "$total" "$CAP_NODE_MIB" "$CAP_TOTAL_MIB" "$now" > "$ALERT"
fi
exit 0
