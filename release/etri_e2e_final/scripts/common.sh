#!/bin/bash
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-etri-e2e-ext:final}"
FINAL_CKPT=work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth
# run CMD inside the container after rebuilding the original code layout
dockrun() { local extra=("${@:1:$#-1}"); local cmd="${!#}"
  docker run --rm --gpus all --shm-size 32g -v "$REL:/release:ro" "${extra[@]}" "$IMAGE" \
    bash -c "python /release/tools/assemble.py >/dev/null && cd /NHNHOME/data/sukim/adcl && $cmd"; }
