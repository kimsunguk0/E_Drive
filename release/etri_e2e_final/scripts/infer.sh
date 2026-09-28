#!/bin/bash
# usage: infer.sh TEST_ROOT OUT_DIR   ->  OUT_DIR/package/submission.zip
set -euo pipefail
source "$(dirname "$0")/common.sh"
TEST="$(realpath "$1")"; OUT="$(realpath -m "$2")"; mkdir -p "$OUT"
CLIPS=("$TEST"/*/); FIRST="$(basename "${CLIPS[0]}")"   # no pipe: `ls | head` + pipefail aborts on long listings
dockrun -v "$TEST:/tmp/etri_test:ro" -v "$OUT:/out" "set -e
E=experiments/a2_final_push_20260923/ext_v6
python \$E/build_ext_submission.py --ckpt $FINAL_CKPT --out-dir /out/work --shard 0/1
python \$E/build_ext_submission.py --ckpt $FINAL_CKPT --out-dir /out/work --merge
python \$E/build_ext_submission.py --ckpt $FINAL_CKPT --out-dir /out/work --flops /tmp/etri_test/$FIRST
python experiments/md_r0_reset_20260914/package_submission.py --submission /out/work/predictions.json \
  --flops-report /out/work/flops_report.json --clips-root /tmp/etri_test --out-dir /out/package --label EXT-FULL-v7"
