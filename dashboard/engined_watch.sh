#!/bin/bash
# Watch the worker's `engined` (claustro-engined-arena.service on SparkyPoo) RSS. Cap 4 GB, per Liam 2026-09-08.
# Cron (head, user liam): */5 * * * *. Logs every sample; desktop-notifies (60-min cooldown) and writes an alert
# file the dashboard shows while it is over the cap.
WORKER="${WORKER:-169.254.152.37}"
CAP_MIB="${ENGINED_CAP_MIB:-4096}"
DIR=/home/liam/cluster-dashboard
LOG=$DIR/engined_watch.log
ALERT=$DIR/engined_alert.json
STAMP=/tmp/engined_watch.notified
mib=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "ps -o rss= -C engined | awk '{s+=\$1} END{printf \"%d\", s/1024}'" 2>/dev/null)
now=$(date '+%Y-%m-%d %H:%M:%S')
if [ -z "$mib" ]; then echo "$now,unreachable" >> "$LOG"; exit 0; fi
echo "$now,$mib" >> "$LOG"
if [ "$mib" -gt "$CAP_MIB" ]; then
    printf '{"process":"engined","node":"SparkyPoo","rss_mib":%d,"cap_mib":%d,"above":true,"at":"%s"}\n' "$mib" "$CAP_MIB" "$now" > "$ALERT"
    if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(stat -c %Y "$STAMP") )) -gt 3600 ]; then
        DISPLAY=:1 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus XDG_RUNTIME_DIR=/run/user/1000 \
            notify-send -u critical -t 0 "engined over its 4 GB cap" "SparkyPoo engined RSS is ${mib} MiB (cap ${CAP_MIB}). Worker headroom is what keeps vLLM alive." 2>/dev/null || true
        touch "$STAMP"
    fi
else
    [ -f "$ALERT" ] && printf '{"process":"engined","node":"SparkyPoo","rss_mib":%d,"cap_mib":%d,"above":false,"at":"%s"}\n' "$mib" "$CAP_MIB" "$now" > "$ALERT"
fi

exit 0
