#!/usr/bin/env bash
# One-screen status for a running SparseDrive training job.
#   bash monitor_run.sh [stage1|stage2]
#
# Reports recent loss, throughput, ETA in OPTIMIZER STEPS (not raw dataloader
# iterations — with gradient accumulation those differ by ACCUM and quoting the
# raw iteration number overstates progress by 8x), GPU contention, and the latest
# checkpoint + its backup status.
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
SCRATCH=${SCRATCH:-/tmp/pm97}
STAGE=${1:-stage1}
ACCUM=${ACCUM:-8}
OFFICIAL_STEPS=43900

LOG=$(ls -t "$PERSIST"/logs/${STAGE}_*.log 2>/dev/null | head -1)
WD=$SCRATCH/work_dirs/$STAGE

hr() { printf '%s\n' "----------------------------------------------------------------"; }

echo "================================================================"
echo " $STAGE  |  $(date -Is)"
echo "================================================================"

if [ -z "$LOG" ]; then echo " no $STAGE log in $PERSIST/logs"; exit 0; fi
echo " log      : $LOG"
echo " work_dir : $WD"

# ---- alive? (match the python process, not this shell: a `pgrep -f` pattern
# ---- also matches the script issuing it, which is how you kill your own caller)
alive=0
for p in $(pgrep -u "$USER" -f 'tools/train.py' 2>/dev/null); do
  [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && alive=$((alive+1))
done
if [ "$alive" -gt 0 ]; then
  et=$(ps -o etime= -p "$(pgrep -u "$USER" -f 'tools/train.py' | head -1)" 2>/dev/null | tr -d ' ')
  echo " status   : RUNNING ($alive python proc, elapsed ${et:-?})"
else
  echo " status   : NOT RUNNING"
  tail -3 "$LOG" | tr '\r' '\n' | sed 's/^/            /'
fi
hr

# ---- progress / throughput ----
last=$(tr '\r' '\n' < "$LOG" | grep -oE 'Iter \[[0-9]+/[0-9]+\].*data_time: [0-9.]+' | tail -1)
if [ -n "$last" ]; then
  cur=$(sed -E 's/Iter \[([0-9]+)\/.*/\1/' <<<"$last")
  tot=$(sed -E 's/Iter \[[0-9]+\/([0-9]+)\].*/\1/' <<<"$last")
  t=$(grep -oE 'time: [0-9.]+' <<<"$last" | head -1 | cut -d' ' -f2)
  d=$(grep -oE 'data_time: [0-9.]+' <<<"$last" | cut -d' ' -f2)
  steps=$((cur / ACCUM))
  awk -v c="$cur" -v tt="$tot" -v t="$t" -v d="$d" -v a="$ACCUM" -v st="$steps" -v os="$OFFICIAL_STEPS" 'BEGIN{
    printf " dl iters : %d / %d  (%.2f%%)\n", c, tt, 100*c/tt;
    printf " opt steps: %d / %d  (%.2f%%)   [accum=%d]\n", st, os, 100*st/os, a;
    printf " iter time: %.3f s   data_time %.3f s (%.1f%% -> %s)\n", t, d, 100*d/t,
           (100*d/t > 15 ? "DATALOADER BOUND, raise WORKERS" : "not data bound");
    printf " samples/s: %.2f\n", 8/t;
    r=(tt-c)*t; printf " remaining: %.1f h  (%.2f days)\n", r/3600, r/86400;
  }'
fi
hr

# ---- loss / NaN ----
echo " recent loss (watch for nan):"
tr '\r' '\n' < "$LOG" | grep -oE 'Iter \[[0-9]+/[0-9]+\].*loss: [0-9.eE+-]+|nan' \
  | grep -oE 'Iter \[[0-9]+|loss: [0-9.eE+-]+|nan' | paste - - 2>/dev/null | tail -6 | sed 's/^/   /'
if tr '\r' '\n' < "$LOG" | grep -qi 'nan'; then
  echo "   *** 'nan' APPEARS IN THE LOG — check before trusting this run ***"
  echo "   *** if loss went NaN under fp16, try bf16 (stable on H200) and record it ***"
fi
hr

# ---- GPU ----
echo " GPU:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
           --format=csv,noheader | sed 's/^/   /'
echo "   co-resident processes (other tenants included):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/     /'
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && echo "     (none)"
hr

# ---- checkpoints ----
echo " checkpoints in $WD:"
ls -lt "$WD"/*.pth 2>/dev/null | head -4 | awk '{printf "   %s  %s %s %s\n", $NF, $5, $6, $7}' \
  || echo "   (none yet)"
echo " backups in $PERSIST/ckpt_backup/$STAGE:"
ls -lt "$PERSIST/ckpt_backup/$STAGE"/*.pth 2>/dev/null | head -3 | awk '{printf "   %s  %s %s %s\n", $NF, $5, $6, $7}' \
  || echo "   (none yet)"
bk=$PERSIST/logs/ckpt_backup.pid
if [ -f "$bk" ] && kill -0 "$(cat "$bk")" 2>/dev/null; then
  echo " backup loop: RUNNING (pid $(cat "$bk"))"
else
  echo " backup loop: NOT RUNNING  -> bash $PERSIST/scripts/ckpt_backup.sh &"
fi
echo "================================================================"
