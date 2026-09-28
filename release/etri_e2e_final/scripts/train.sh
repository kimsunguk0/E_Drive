#!/bin/bash
# usage: train.sh STAGE RAW_TRAIN_ROOT
#   STAGE 1 : public initializer  -> A2-H4-PROGRESS-FULL (24,931 updates, whole model)
#   STAGE 2 : shipped stage-1 ckpt -> L-FULL6 (+38,070 updates, whole model; smoke, verify, main)
#   STAGE 3 : stage-2 output       -> EXT-FULL-v7 (6,344 updates, trunk frozen, 15-candidate planner)
#   RAW_TRAIN_ROOT: official train data with the 376 <scenario>/ folders (annotation/, calibration/, camera_*/, meta/)
# Needs one GPU with >= 40 GB (trained on NVIDIA B200). Outputs land inside code/work_dirs and code/reports.
set -euo pipefail
source "$(dirname "$0")/common.sh"
STAGE="$1"; RAW="$(realpath "$2")"
PM=(-v "$RAW:/tmp/pm97/data/etri/train:ro"
    -v "$REL/data_pm97/data/etri/ego_cache.npz:/tmp/pm97/data/etri/ego_cache.npz:ro"
    -v "$REL/data_pm97/data/etri/ego_cache_5s.npz:/tmp/pm97/data/etri/ego_cache_5s.npz:ro")
run() { docker run --rm --gpus all --shm-size 32g -e CUDA_VISIBLE_DEVICES=0 "${CODE_MOUNT[@]}" "${PM[@]}" "$IMAGE" bash -c "$1"; }
case "$STAGE" in
  1) run "python experiments/a2_progress_full_20260921/train_full.py --gpu 0 --run-dir work_dirs/release_stage1/A2-H4-PROGRESS-FULL-s1" ;;
  2) run "W=experiments/a2_final_push_20260923/repro_lfull6.py; python \$W smoke && python \$W verify && python \$W main" ;;
  3) run "python experiments/a2_final_push_20260923/ext_v7rr/train_ext.py --arm full_repro --gpu 0 --tag v7rr" ;;
  3-shipped-parent) run "python experiments/a2_final_push_20260923/ext_v6/train_ext.py --arm full --gpu 0 --tag v7r" ;;
  *) echo "stage must be 1, 2, 3 or 3-shipped-parent"; exit 1 ;;
esac
