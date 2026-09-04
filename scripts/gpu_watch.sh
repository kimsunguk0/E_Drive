#!/usr/bin/env bash
# Hourly nvidia-smi snapshot -> logs/gpu_watch.log
# Purpose: quantify contention with other users' jobs after the fact. Free VRAM
# moved between 0 and 80 GB during phase 1, so batch-size decisions need a record
# of what was actually free over time, not a one-off reading.
#
# Single-instance guard uses a pidfile, NOT `pgrep -f gpu_watch.sh` — that
# pattern also matches the shell issuing the check, so the guard would always
# think it is already running (and `pkill -f` on it would kill the caller).
PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
LOG=$PERSIST/logs/gpu_watch.log
PIDFILE=$PERSIST/logs/gpu_watch.pid

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running as pid $(cat "$PIDFILE")"; exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

while true; do
  {
    printf '\n===== %s =====\n' "$(date -Is)"
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
               --format=csv,noheader
    echo "-- compute apps --"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  } >> "$LOG" 2>&1
  sleep 3600
done
