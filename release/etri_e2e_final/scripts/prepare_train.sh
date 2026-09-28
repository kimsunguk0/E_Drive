#!/bin/bash
# usage: prepare_train.sh RAW_TRAIN_ROOT PREP_DIR   (RAW = 376 <scenario>.tar files or extracted <scenario>/ folders)
#   -> PREP_DIR/etri_768 (undistorted 768x432 images) and PREP_DIR/meta_train (parquet), read by train.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
RAW="$(realpath "$1")"; PREP="$(realpath -m "$2")"; mkdir -p "$PREP"
docker run --rm -v "$REL:/release:ro" -v "$RAW:/raw:ro" -v "$PREP:/prep" "$IMAGE" \
  python /release/tools/build_train_cache.py --raw /raw --out /prep --jobs "${JOBS:-8}"
