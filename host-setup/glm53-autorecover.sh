#!/usr/bin/env bash
# autorecover.sh: restart the GLM server when it is already dead, so a crashed rank costs minutes instead of hours.
# Runs on the worker node (the one that does not run start.sh) as glm53-autorecover.service, from exl3-kit/scripts/.
#
# It restarts only when:
#   * a worker traceback / CUDA fault appeared in either container since that container started, or
#   * either container is not running and no boot is in progress, or
#   * the engine is hung (a rank stopped answering: "No available shared memory broadcast block" 5x in 6 min), or
#   * the API failed /health for AUTORECOVER_DOWN_S seconds outside a boot.
# It never touches a healthy server. At most AUTORECOVER_MAX_PER_HOUR restarts an hour; every action is logged, and
# each crash's container logs are saved under the head's logs/crash-<time>/ before the restart removes them.
# A boot that has not reached "ready" within AUTORECOVER_BOOT_MAX_S counts as failed (start.sh can die in preflight and
# leave boot-state.json at "teardown").
set -u
KIT="$(cd "$(dirname "$0")/.." && pwd)"
HEAD_SSH="${AUTORECOVER_HEAD_SSH:?set AUTORECOVER_HEAD_SSH=user@<head fabric IP>}"
HEAD_KIT="${AUTORECOVER_HEAD_KIT:-$KIT}"   # the kit directory on the head (default: same path as here)
API="${AUTORECOVER_API:-http://${HEAD_SSH#*@}:${PORT:-8888}}"
HEAD_C="${CONTAINER_HEAD:-glm53-exl3-head}"; WORKER_C="${CONTAINER_WORKER:-glm53-exl3-worker}"
INTERVAL="${AUTORECOVER_INTERVAL_S:-15}"; DOWN_S="${AUTORECOVER_DOWN_S:-300}"; MAXH="${AUTORECOVER_MAX_PER_HOUR:-3}"
BOOT_MAX_S="${AUTORECOVER_BOOT_MAX_S:-900}"
LOG="${AUTORECOVER_LOG:-$KIT/logs/autorecover.log}"
DRY="${AUTORECOVER_DRY:-0}"; ONCE="${AUTORECOVER_ONCE:-0}"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
hs() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=auto -o ControlPath=/tmp/glm53-autorecover-%C -o ControlPersist=300 "$HEAD_SSH" "$@"; }

boot_state() {   # "<phase> <age_s>" of the head's boot-state.json
  hs "python3 -c 'import json,time;d=json.load(open(\"$HEAD_KIT/logs/boot-state.json\"));print(d.get(\"phase\",\"unknown\"),int(time.time()-float(d.get(\"t\",0))))'" 2>/dev/null || echo "unknown 0"
}
booting() {
  local ph age; read -r ph age <<<"$(boot_state)"
  case "$ph" in ready|healthy|failed|unknown) return 1;; esac
  [ "${age:-0}" -lt "$BOOT_MAX_S" ]
}
running() {      # $1 = local|head, $2 = container
  local run; [ "$1" = local ] && run="bash -c" || run="hs"
  [ "$($run "docker inspect -f '{{.State.Running}}' $2" 2>/dev/null)" = true ]
}
declare -A SINCE STARTED
worker_died() {  # result in $WD; the container log is scanned from its start once, then incrementally
  local run started since; WD=""; [ "$1" = local ] && run="bash -c" || run="hs"
  started=$($run "docker inspect -f '{{.State.StartedAt}}' $2" 2>/dev/null) || return 1
  if [ "${STARTED[$2]:-}" != "$started" ]; then STARTED[$2]=$started; SINCE[$2]=$started; fi
  since=${SINCE[$2]}; SINCE[$2]=$(date -u -d "-40 sec" +%Y-%m-%dT%H:%M:%SZ)
  WD=$($run "docker logs --since '$since' $2 2>&1 | grep -a -m1 -E 'WorkerProc hit an exception|EngineDeadError|Engine core proc .* died|illegal memory access|Worker proc .* died unexpectedly'" 2>/dev/null)
}
engine_hung() {
  local n; n=$(hs "docker logs --since 6m $HEAD_C 2>&1 | grep -a -c 'No available shared memory broadcast block' || true" 2>/dev/null)
  n=${n//[!0-9]/}; [ "${n:-0}" -ge 5 ]
}
save_logs() {    # keep both containers' logs + this node's Xid lines on the head before restart removes the containers
  local d="$HEAD_KIT/logs/crash-$(date +%m%d-%H%M)"
  hs "mkdir -p $d && docker logs $HEAD_C > $d/head.log 2>&1" || true
  docker logs "$WORKER_C" 2>&1 | hs "cat > $d/worker.log" || true
  dmesg -T 2>/dev/null | grep -i -E 'xid|nvrm' | tail -20 | hs "cat > $d/dmesg-worker.txt" || true
  log "saved crash logs to head:$d"
}

restarts=(); down_since=0
log "armed: api=$API head=$HEAD_SSH kit=$HEAD_KIT containers=$HEAD_C/$WORKER_C interval=${INTERVAL}s down=${DOWN_S}s max/h=$MAXH"
while :; do
  [ "$ONCE" = 1 ] || sleep "$INTERVAL"
  reason=""
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$API/health" || true)
  if [ "$code" = 200 ]; then down_since=0; else [ "$down_since" = 0 ] && down_since=$(date +%s); fi
  if booting; then [ "$ONCE" = 1 ] && { log "check: boot in progress"; break; }; continue; fi
  worker_died local "$WORKER_C"; wl=$WD; worker_died head "$HEAD_C"; wh=$WD
  if [ -n "$wl" ]; then reason="worker error: ${wl:0:180}"
  elif [ -n "$wh" ]; then reason="head error: ${wh:0:180}"
  elif ! running head "$HEAD_C"; then reason="head container not running"
  elif ! running local "$WORKER_C"; then reason="worker container not running"
  elif engine_hung; then reason="engine hung (a rank has not answered for 5+ minutes)"
  elif [ "$down_since" != 0 ] && [ $(( $(date +%s) - down_since )) -ge "$DOWN_S" ]; then reason="API /health failing for $(( $(date +%s) - down_since ))s"
  fi
  if [ -z "$reason" ]; then [ "$ONCE" = 1 ] && { log "check: healthy (api $code)"; break; }; continue; fi
  if [ "$DRY" = 1 ]; then log "DRY: would restart ($reason)"; [ "$ONCE" = 1 ] && break; continue; fi
  now=$(date +%s); keep=(); for t in "${restarts[@]}"; do [ $((now - t)) -lt 3600 ] && keep+=("$t"); done; restarts=("${keep[@]}")
  if [ "${#restarts[@]}" -ge "$MAXH" ]; then log "DEAD ($reason) but $MAXH restarts in the last hour: leaving it for a human"; sleep 600; continue; fi
  log "DEAD: $reason -> saving logs and restarting"
  save_logs
  restarts+=("$now")
  hs "cd $HEAD_KIT && (setsid nohup ./start.sh restart > logs/restart-auto-\$(date +%m%d-%H%M).log 2>&1 < /dev/null &)" || log "could not reach the head to restart"
  sleep 60
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$API/health")" = 200 ] && ! booting && { log "recovered after $(( $(date +%s) - now ))s"; break; }
    sleep 10
  done
  down_since=0
done
