#!/usr/bin/env bash
# Phase 5 — collect official checkpoints and pretrained backbones into ckpt/.
#
# Every URL here was read out of the corresponding repo's own README/docs at the
# cloned revision, not guessed:
#   SparseDrive  README.md:79-80          (github releases v1.0)
#   VAD          README.md:48-49          (Google Drive -> needs gdown)
#   StreamPETR   README.md:38-54, docs/data_preparation.md:23
#
# Records sha256 for every file in ckpt/CHECKSUMS.txt and prints a SKIPPED list
# at the end for anything that failed (dead link / auth wall) so it can be
# handled manually.
#
# Idempotent: a file that already exists with a recorded sha256 is skipped.
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/env_vars.sh"

CK=$PERSIST/ckpt
SUMS=$CK/CHECKSUMS.txt
LOG=$PERSIST/logs/download_ckpts.log
mkdir -p "$CK"/{sparsedrive,vad,streampetr,backbones} "$PERSIST/logs"
touch "$SUMS"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

declare -a SKIPPED=()
ok_n=0; skip_n=0

# get <subdir> <filename> <url>
get() {
  local sub="$1" name="$2" url="$3" dest="$CK/$1/$2"
  if [ -s "$dest" ] && grep -q "  $sub/$name\$" "$SUMS" 2>/dev/null; then
    log "SKIP  $sub/$name (already have it)"; skip_n=$((skip_n+1)); return 0
  fi
  log "GET   $sub/$name"
  # -L follows the github release redirect to S3; -f makes HTTP errors non-zero
  if ! curl -sSL -f --retry 4 --retry-delay 5 --retry-all-errors \
            -o "$dest.part" "$url"; then
    log "FAIL  $sub/$name  <- $url"
    SKIPPED+=("$sub/$name  <-  $url")
    rm -f "$dest.part"
    return 1
  fi
  mv "$dest.part" "$dest"
  record "$sub" "$name"
}

# gdrive <subdir> <filename> <fileid>
gdrive() {
  local sub="$1" name="$2" id="$3" dest="$CK/$1/$2"
  if [ -s "$dest" ] && grep -q "  $sub/$name\$" "$SUMS" 2>/dev/null; then
    log "SKIP  $sub/$name (already have it)"; skip_n=$((skip_n+1)); return 0
  fi
  if ! command -v gdown >/dev/null 2>&1; then
    log "FAIL  $sub/$name — gdown not installed"
    SKIPPED+=("$sub/$name  <-  https://drive.google.com/file/d/$id/view (gdown missing)")
    return 1
  fi
  log "GET   $sub/$name (google drive $id)"
  # Large Drive files serve an HTML confirm page unless gdown handles the token;
  # if Drive is rate-limiting, gdown exits non-zero and we report it as SKIPPED
  # rather than leaving an HTML file that looks like a checkpoint.
  if ! gdown --no-cookies -q -O "$dest.part" "$id" 2>>"$LOG"; then
    log "FAIL  $sub/$name — gdown failed (Drive quota or auth wall)"
    SKIPPED+=("$sub/$name  <-  https://drive.google.com/file/d/$id/view")
    rm -f "$dest.part"
    return 1
  fi
  # A Drive quota page is small HTML, never a real checkpoint.
  if [ "$(stat -c %s "$dest.part")" -lt 1000000 ] && head -c 200 "$dest.part" | grep -qi 'html'; then
    log "FAIL  $sub/$name — got an HTML page, not a checkpoint"
    SKIPPED+=("$sub/$name  <-  https://drive.google.com/file/d/$id/view (HTML quota page)")
    rm -f "$dest.part"
    return 1
  fi
  mv "$dest.part" "$dest"
  record "$sub" "$name"
}

record() {
  local sub="$1" name="$2" dest="$CK/$1/$2"
  local sha; sha=$(sha256sum "$dest" | awk '{print $1}')
  grep -v "  $sub/$name\$" "$SUMS" > "$SUMS.tmp" 2>/dev/null || true
  mv "$SUMS.tmp" "$SUMS"
  printf '%s  %s/%s\n' "$sha" "$sub" "$name" >> "$SUMS"
  log "OK    $sub/$name  ($(du -h "$dest" | cut -f1), sha256 ${sha:0:16}…)"
  ok_n=$((ok_n+1))
}

SD=https://github.com/swc-17/SparseDrive/releases/download/v1.0
SP=https://github.com/exiawsh/storage/releases/download/v1.0

# ---------------- SparseDrive (the primary model) ----------------
get sparsedrive sparsedrive_stage1.pth      "$SD/sparsedrive_stage1.pth"
get sparsedrive sparsedrive_stage2.pth      "$SD/sparsedrive_stage2.pth"
# reference training logs — tiny, and the only way to sanity-check our own
# training curve against the released numbers
get sparsedrive sparsedrive_stage1_log.txt  "$SD/sparsedrive_stage1_log.txt"
get sparsedrive sparsedrive_stage2_log.txt  "$SD/sparsedrive_stage2_log.txt"

# ---------------- backbones ----------------
# SparseDrive's quick_start.md actually asks for ResNet-50, not R101.
get backbones resnet50-19c8e357.pth  https://download.pytorch.org/models/resnet50-19c8e357.pth
# R101 requested by the work order; both torchvision variants exist, take V1.
get backbones resnet101-63fe2227.pth https://download.pytorch.org/models/resnet101-63fe2227.pth
# DD3D-pretrained VoVNet V2-99, distributed via StreamPETR releases.
get backbones fcos3d_vovnet_imgbackbone-remapped.pth "$SP/fcos3d_vovnet_imgbackbone-remapped.pth"
get backbones dd3d_det_final.pth                     "$SP/dd3d_det_final.pth"
# the upstream DDAD15M depth-pretrained V2-99 from TRI-ML (what DD3D starts from)
get backbones depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth \
    https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth
# nuImages-pretrained R50 used by StreamPETR's 428q config
get backbones cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth \
    https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth

# ---------------- StreamPETR ----------------
get streampetr stream_petr_vov_flash_800_bs2_seq_24e.pth        "$SP/stream_petr_vov_flash_800_bs2_seq_24e.pth"
get streampetr stream_petr_r50_flash_704_bs2_seq_90e.pth        "$SP/stream_petr_r50_flash_704_bs2_seq_90e.pth"
get streampetr stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth "$SP/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth"

# ---------------- VAD (Google Drive) ----------------
gdrive vad VAD_tiny.pth 1KgCC_wFqPH0CQqdr6Pp2smBX5ARPaqne
gdrive vad VAD_base.pth 1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS

# ---------------- timm ConvNeXt-S ----------------
# timm pulls from HF and caches; copy the resolved file so ckpt/ is
# self-contained and does not depend on the HF cache surviving a container reset.
log "GET   backbones/convnext_small (via timm/huggingface_hub)"
if python - <<'PY' >>"$LOG" 2>&1
import os, shutil, glob, sys
try:
    import timm, torch
    m = timm.create_model('convnext_small.fb_in22k_ft_in1k', pretrained=True)
    dest = os.path.join(os.environ['PERSIST'], 'ckpt', 'backbones',
                        'convnext_small_fb_in22k_ft_in1k.pth')
    torch.save(m.state_dict(), dest)
    print("saved", dest)
except Exception as e:
    print("timm convnext failed:", e); sys.exit(1)
PY
then
  record backbones convnext_small_fb_in22k_ft_in1k.pth
else
  log "FAIL  backbones/convnext_small (timm)"
  SKIPPED+=("backbones/convnext_small_fb_in22k_ft_in1k.pth  <-  timm 'convnext_small.fb_in22k_ft_in1k'")
fi

# ---------------- report ----------------
log "===== ckpt summary: $ok_n downloaded / $skip_n already present / ${#SKIPPED[@]} skipped ====="
if [ ${#SKIPPED[@]} -gt 0 ]; then
  log "----- SKIPPED (handle manually) -----"
  for s in "${SKIPPED[@]}"; do log "  $s"; done
fi
log "manifest: $SUMS"
sort -k2 "$SUMS" -o "$SUMS"
