#!/bin/bash
# Shared settings. REL = this release directory.
REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-etri-e2e-ext:final}"
CODE_MOUNT=(-v "$REL/code:/NHNHOME/data/sukim/adcl")
FINAL_CKPT=work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth
