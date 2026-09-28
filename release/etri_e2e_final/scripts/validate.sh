#!/bin/bash
# correctness on bundled clips + FLOPs + GPU forward latency
set -euo pipefail
source "$(dirname "$0")/common.sh"
OUT="$(realpath -m "${1:-$REL/validation/result}")"; mkdir -p "$OUT"
dockrun -v "$OUT:/vout" "python /release/tools/validate_release.py --ckpt $FINAL_CKPT --out /vout/validation.json"
