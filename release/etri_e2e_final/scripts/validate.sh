#!/bin/bash
# usage: [GPU=0] validate.sh [TEST_ROOT] [OUT_DIR]
#   correctness on the 8 bundled train clips (+ 20 official test clips if TEST_ROOT is given),
#   FLOPs and GPU forward latency  ->  OUT_DIR/validation.json (default validation/result)
set -euo pipefail
source "$(dirname "$0")/common.sh"
MNT=()
if [ -n "${1:-}" ]; then MNT=(-v "$(realpath "$1"):/tmp/etri_test:ro"); fi
OUT="$(realpath -m "${2:-$REL/validation/result}")"; mkdir -p "$OUT"
dockrun "${MNT[@]}" -v "$OUT:/vout" "python /release/tools/validate_release.py --ckpt $FINAL_CKPT --out /vout/validation.json"
