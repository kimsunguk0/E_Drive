#!/bin/bash
# Correctness + RTX 4090 timing on the bundled clips (8 train fixtures, 20 official test clips).
set -euo pipefail
source "$(dirname "$0")/common.sh"
OUT="${1:-$REL/validation/result}"; mkdir -p "$OUT"
docker run --rm --gpus all --shm-size 8g "${CODE_MOUNT[@]}" \
  -v "$REL/validation:/validation:ro" -v "$REL/tools:/tools:ro" -v "$OUT:/out" "$IMAGE" \
  python /tools/validate_release.py --ckpt $FINAL_CKPT --out /out/validation.json
