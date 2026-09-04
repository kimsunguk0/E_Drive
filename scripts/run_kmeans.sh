#!/usr/bin/env bash
# Phase B — generate the four k-means anchor files SparseDrive's configs require.
#
# WHY THIS WRAPPER EXISTS (do not just run tools/kmeans/*.py directly)
#
# 1. sklearn's KMeans SEGFAULTS on this host (exit 139) with the message
#      "OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary
#       array for thread metadata"
#    This box has 128 cores, which exceeds the thread limit the installed
#    OpenBLAS was compiled for, and it corrupts memory instead of degrading.
#    Capping threads at 32 fixes it (64 still warned; 32 verified clean).
#    kmeans_det (K=900) and kmeans_motion died this way and silently produced
#    NO output file while the shell reported success.
#
# 2. The repo scripts pipe nothing and check nothing, so a crash looks like a
#    pass. Worse, `python ... | tail` makes $? the exit code of `tail`, so the
#    failure is invisible. This script checks the real exit code AND asserts the
#    expected file exists.
#
# Expected outputs (paths hardcoded in projects/configs/sparsedrive_small_*.py):
#   data/kmeans/kmeans_det_900.npy
#   data/kmeans/kmeans_map_100.npy
#   data/kmeans/kmeans_motion_6.npy    (fut_mode = 6)
#   data/kmeans/kmeans_plan_6.npy      (ego_fut_mode = 6)
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
source "$PERSIST/scripts/env_vars.sh" --activate

REPO=$PERSIST/src/SparseDrive
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# The actual fix. Must be set before numpy/OpenBLAS initialises.
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

LOG=$PERSIST/logs/kmeans_run.log
: > "$LOG"
mkdir -p data/kmeans vis/kmeans

declare -A EXPECT=(
  [det]=data/kmeans/kmeans_det_900.npy
  [map]=data/kmeans/kmeans_map_100.npy
  [motion]=data/kmeans/kmeans_motion_6.npy
  [plan]=data/kmeans/kmeans_plan_6.npy
)

fail=0
for s in det map motion plan; do
  want=${EXPECT[$s]}
  if [ -s "$want" ]; then
    echo "SKIP  kmeans_$s ($want already exists, $(stat -c %s "$want") bytes)" | tee -a "$LOG"
    continue
  fi
  echo "RUN   kmeans_$s -> $want" | tee -a "$LOG"
  # No pipe: we need python's own exit code, not a pipeline's.
  python "tools/kmeans/kmeans_$s.py" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL  kmeans_$s exited $rc$([ $rc -eq 139 ] && echo ' (SEGFAULT — raise the thread cap question again)')" | tee -a "$LOG"
    fail=1; continue
  fi
  if [ ! -s "$want" ]; then
    echo "FAIL  kmeans_$s exited 0 but $want was not written" | tee -a "$LOG"
    fail=1; continue
  fi
  echo "OK    kmeans_$s  ($(stat -c %s "$want") bytes)" | tee -a "$LOG"
done

echo "===== data/kmeans =====" | tee -a "$LOG"
ls -la data/kmeans/ | tee -a "$LOG"

if [ $fail -ne 0 ]; then
  echo "===== KMEANS INCOMPLETE — see $LOG =====" | tee -a "$LOG"
  exit 1
fi

# Preserve to NFS: regenerating costs ~10 min and training cannot start without these.
mkdir -p "$PERSIST/data/kmeans"
cp -a data/kmeans/. "$PERSIST/data/kmeans/"
echo "preserved to $PERSIST/data/kmeans" | tee -a "$LOG"
echo "===== KMEANS OK =====" | tee -a "$LOG"
