#!/usr/bin/env bash
# Evaluate a checkpoint on the nuScenes val split.
#
#   ./run_eval.sh stage1 /tmp/pm97/work_dirs/stage1/iter_70240.pth
#   ./run_eval.sh stage2 <ckpt> [--result_file out.pkl]
#
# Uses torchrun rather than the repo's tools/dist_test.sh for the same reason as
# launch_train.sh: torch >= 2.0's torch.distributed.launch appends
# `--local-rank=0` (hyphen) while the repo declares `--local_rank` (underscore),
# so dist_test.sh dies in argparse before doing anything.
#
# Requires patches 01 and 04 (mmcv 1.7.2 vs torch 2.1). Patch 04 in particular is
# what makes evaluation possible at all: mmcv's MMDistributedDataParallel eval
# path touches DDP internals torch 2.1 removed.
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
source "$PERSIST/scripts/env_vars.sh" --activate

STAGE=${1:?usage: run_eval.sh stage1|stage2 <checkpoint> [extra args]}
CKPT=${2:?usage: run_eval.sh stage1|stage2 <checkpoint> [extra args]}
shift 2

REPO=$PERSIST/src/SparseDrive
CFG=projects/configs/sparsedrive_small_${STAGE}.py
[ -f "$REPO/$CFG" ] || { echo "no such config: $CFG"; exit 1; }
[ -f "$CKPT" ]      || { echo "no such checkpoint: $CKPT"; exit 1; }

STAMP=$(date +%F_%H%M%S)
LOG=$PERSIST/logs/eval_${STAGE}_$(basename "$CKPT" .pth)_${STAMP}.log
PORT=${PORT:-28777}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32

{
  echo "===================================================================="
  echo " eval stage : $STAGE"
  echo " checkpoint : $CKPT ($(stat -c %s "$CKPT") bytes)"
  echo " sha256     : $(sha256sum "$CKPT" | cut -c1-32)…"
  echo " config     : $CFG"
  echo " free VRAM  : $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  echo " started    : $(date -Is)"
  echo "===================================================================="
} | tee -a "$LOG"

python -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" \
  tools/test.py "$CFG" "$CKPT" --launcher pytorch --eval bbox \
  "$@" >> "$LOG" 2>&1

rc=$?
echo "=== eval exited rc=$rc at $(date -Is) ===" | tee -a "$LOG"
echo "log: $LOG"
exit $rc
