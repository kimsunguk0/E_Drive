#!/bin/bash
# Download the public backbone needed only by training stage 1 (not bundled, to keep the
# submission small): mmdetection3d nuImages Cascade Mask R-CNN R50 (COCO + nuImages).
set -euo pipefail
REL="$(cd "$(dirname "$0")/.." && pwd)"
DST="$REL/checkpoints/backbone_nuimages_cascade_r50.pth"
URL=https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
SHA=4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e
[ -f "$DST" ] || curl -fL -o "$DST" "$URL"
echo "$SHA  $DST" | sha256sum -c -
