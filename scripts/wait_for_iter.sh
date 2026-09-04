#!/usr/bin/env bash
# Block until the stage1 run clears a target iteration, dies, or goes NaN.
#
# WHY: the fp32-attention fix is unproven until the run passes the two iterations
# where the flash-attn runs blew up (~21,012 and ~31,416). Polling by hand wastes
# turns and risks missing the moment; this exits exactly once there is something
# to say, and prints the reason.
#
#   TARGET=21000 nohup setsid bash wait_for_iter.sh &
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
STAGE=${STAGE:-stage1}
TARGET=${TARGET:-21000}
EVERY=${EVERY:-120}
MAX_WAIT=${MAX_WAIT:-32400}      # 9 h ceiling so this can never hang forever

# Newest log for this stage. Do NOT pin the date here -- an earlier version did,
# and it silently matched nothing once the run rolled past midnight or the stage
# changed, so the waiter reported iter=0 forever.
logfile() { ls -t "$PERSIST"/logs/${STAGE}_2026*.log 2>/dev/null | head -1; }

cur_iter() {
  local l; l=$(logfile); [ -z "$l" ] && { echo 0; return; }
  tr '\r' '\n' < "$l" | grep -oE 'Iter \[[0-9]+/' | tail -1 | tr -dc '0-9'
}

trainers() {
  local n=0 p
  for p in $(pgrep -u "$USER" -f 'tools/train.py' 2>/dev/null); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && n=$((n+1))
  done
  echo "$n"
}

start=$(date +%s)
while :; do
  it=$(cur_iter); it=${it:-0}
  n=$(trainers)
  lg=$(logfile)

  if [ -n "$lg" ] && tr '\r' '\n' < "$lg" | tail -60 | grep -qE 'loss: nan|grad_norm: nan'; then
    echo "RESULT=NAN iter=$it"; exit 2
  fi
  if [ "$n" -eq 0 ]; then
    echo "RESULT=DEAD iter=$it (트레이너 없음 — 워치독이 정지시켰거나 크래시)"; exit 3
  fi
  if [ "$it" -ge "$TARGET" ]; then
    echo "RESULT=REACHED iter=$it target=$TARGET"; exit 0
  fi
  if [ $(( $(date +%s) - start )) -ge "$MAX_WAIT" ]; then
    echo "RESULT=TIMEOUT iter=$it target=$TARGET"; exit 4
  fi
  sleep "$EVERY"
done
