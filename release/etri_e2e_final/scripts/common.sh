#!/bin/bash
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-etri-e2e-ext:final}"
FINAL_CKPT=work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth
# GPU=<index> exposes only that GPU (it becomes device 0 in the container); default: all GPUs, first one used
if [ "${GPU:-all}" = all ]; then GPU_ARG=(--gpus all); else GPU_ARG=(--gpus "device=$GPU"); fi
# run CMD inside the container after rebuilding the original code layout
dockrun() { local extra=("${@:1:$#-1}"); local cmd="${!#}"
  docker run --rm "${GPU_ARG[@]}" --shm-size 32g -v "$REL:/release:ro" "${extra[@]}" "$IMAGE" \
    bash -c "python /release/tools/assemble.py >/dev/null && cd /NHNHOME/data/sukim/adcl && $cmd"; }
