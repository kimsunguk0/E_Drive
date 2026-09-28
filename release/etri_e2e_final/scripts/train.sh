#!/bin/bash
# usage: train.sh STAGE PREP_DIR OUT_DIR   (PREP_DIR made by prepare_train.sh)
#   1: public initializer -> stage-1 model (24,931 updates)
#   2: shipped stage-1    -> stage-2 model (+38,070 updates; smoke, verify, main)
#   3: shipped stage-2    -> submitted model recipe (6,344 updates, trunk frozen)
# One GPU. Stages 1-2 check for ~174 GiB free memory at start (an idle B200); stage 3 uses ~15 GB.
# Outputs are written to OUT_DIR/stage*.
set -euo pipefail
source "$(dirname "$0")/common.sh"
STAGE="$1"; PREP="$(realpath "$2")"; OUT="$(realpath -m "$3")"; mkdir -p "$OUT"
M=(-e CUDA_VISIBLE_DEVICES=0 -v "$PREP/etri_768:/tmp/pm97/cache/etri_768:ro"
   -v "$PREP/meta_train:/tmp/pm97/data/etri/meta_train:ro" -v "$OUT:/out")
case "$STAGE" in
  1) dockrun "${M[@]}" "python experiments/a2_progress_full_20260921/train_full.py --gpu 0 --run-dir work_dirs/release_stage1/A2-H4-PROGRESS-FULL-s1" ;;
  2) dockrun "${M[@]}" "W=experiments/a2_final_push_20260923/repro_lfull6.py; python \$W smoke && python \$W verify && python \$W main" ;;
  3) dockrun "${M[@]}" "python experiments/a2_final_push_20260923/ext_v6/train_ext.py --arm full --gpu 0 --tag repro" ;;
  *) echo "stage must be 1, 2 or 3"; exit 1 ;;
esac
