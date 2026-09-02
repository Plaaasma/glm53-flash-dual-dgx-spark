#!/bin/bash
# vLLM memory watchdog: SIGKILL vLLM if MemAvailable < 768 MiB (0.2s cadence).
# Direct process kill, NOT docker kill -- dockerd can stall under the very
# memory pressure this guards against. Runs as a file so pkill -f "VLLM::"
# cannot self-match (the pattern is not in this process's cmdline).
LOG=/var/tmp/vllm-watchdog.log
echo "$(date '+%F %T') watchdog armed (threshold 786432 kB)" >> $LOG
while :; do
  a=$(awk '/MemAvailable/{print $2; exit}' /proc/meminfo)
  if [ "${a:-9999999}" -lt 786432 ]; then
    # TERM first: vLLM traps it and releases CUDA allocations cleanly --
    # a raw SIGKILL mid-CUDA leaked ~20 GiB of driver memory per node
    # (2026-09-02, recoverable only by reboot). KILL follows if TERM
    # does not relieve pressure within 3 s.
    pkill -TERM -f "VLLM::" 2>/dev/null
    pkill -TERM -x vllm 2>/dev/null
    sleep 3
    b=$(awk '/MemAvailable/{print $2; exit}' /proc/meminfo)
    if [ "${b:-9999999}" -lt 786432 ]; then
      pkill -9 -f "VLLM::" 2>/dev/null
      pkill -9 -x vllm 2>/dev/null
    fi
    ( docker kill glm53-exl3-head glm53-exl3-worker >/dev/null 2>&1 & )
    echo "$(date '+%F %T') TRIPPED MemAvailable=${a}kB -> vllm killed" >> $LOG
    sleep 5
  fi
  sleep 0.2
done
