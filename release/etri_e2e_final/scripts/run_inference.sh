#!/bin/bash
# usage: run_inference.sh TEST_ROOT OUT_DIR
#   TEST_ROOT: the official test folder with the 1,125 <clip_hash>/ directories
#   OUT_DIR  : receives work/ (per-clip predictions, FLOPs report) and package/submission.zip
set -euo pipefail
source "$(dirname "$0")/common.sh"
TEST="$(realpath "$1")"; OUT="$(realpath -m "$2")"; mkdir -p "$OUT"
FIRST="$(ls "$TEST" | head -1)"
docker run --rm --gpus all --shm-size 16g "${CODE_MOUNT[@]}" \
  -v "$TEST:/tmp/etri_test:ro" -v "$OUT:/out" "$IMAGE" bash -c "
set -e
E=experiments/a2_final_push_20260923/ext_v6
python \$E/build_ext_submission.py --ckpt $FINAL_CKPT --out-dir /out/work --shard 0/1
python \$E/build_ext_submission.py --ckpt $FINAL_CKPT --out-dir /out/work --merge
python \$E/build_ext_submission.py --ckpt $FINAL_CKPT --out-dir /out/work --flops /tmp/etri_test/$FIRST
python experiments/md_r0_reset_20260914/package_submission.py --submission /out/work/predictions.json \
  --flops-report /out/work/flops_report.json --clips-root /tmp/etri_test --out-dir /out/package --label EXT-FULL-v7"
echo "submission: $OUT/package/submission.zip"
