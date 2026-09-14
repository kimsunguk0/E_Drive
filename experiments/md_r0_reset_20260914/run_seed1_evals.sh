#!/bin/bash
# Same T0 probe and batch-1 V0 for the seed 1 pair, at terminal and at best-on-V0.
set -u
cd /NHNHOME/data/sukim/adcl
NEW=data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json
SUP=data/etri/motiondrive_v2/r0reset_tplus_geometry_v2
T0=reports/md_r0_reset_20260914/t0_scenes.txt
ARM=$1; GPU=$2
for step in 20554 13704; do
  ck=work_dirs/md_r0_reset_20260914/$ARM/ckpt_step${step}.pth
  tag=${ARM}_probe_step${step}
  out=reports/md_r0_reset_20260914/eval/${tag}.json
  if [ ! -f "$out" ]; then
    CUDA_VISIBLE_DEVICES=$GPU env/venv/bin/python experiments/md_r0_reset_20260914/evaluate.py \
      --init "$ck" --tag "$tag" --gpu 0 --eval-split train --eval-stride 1 --eval-batch 4 \
      --probe-per-session 48 --scenes-file "$T0" --split-manifest "$NEW" --supervision-root "$SUP" \
      --run-dir work_dirs/md_r0_reset_20260914/evals/${tag} --out "$out" \
      >> reports/md_r0_reset_20260914/logs/${ARM}_evals.log 2>&1 || echo "FAILED $tag"
  fi
done
tag=${ARM}_tuneB1_step20554
out=reports/md_r0_reset_20260914/eval/${tag}.json
if [ ! -f "$out" ]; then
  CUDA_VISIBLE_DEVICES=$GPU env/venv/bin/python experiments/md_r0_reset_20260914/evaluate.py \
    --init work_dirs/md_r0_reset_20260914/$ARM/ckpt_step20554.pth --tag "$tag" --gpu 0 \
    --eval-split tune --eval-stride 5 --eval-batch 1 --split-manifest "$NEW" \
    --supervision-root "$SUP" --run-dir work_dirs/md_r0_reset_20260914/evals/${tag} --out "$out" \
    >> reports/md_r0_reset_20260914/logs/${ARM}_evals.log 2>&1 || echo "FAILED $tag"
fi
echo "DONE $ARM"
