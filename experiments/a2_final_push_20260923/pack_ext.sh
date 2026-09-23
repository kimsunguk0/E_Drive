#!/bin/bash
# usage: pack_ext.sh CKPT OUTDIR LABEL [EXT_DIR]
set -e
cd /NHNHOME/data/sukim/adcl
CK=$1; O=$2; LABEL=$3; D=${4:-experiments/a2_final_push_20260923}
B=$D/build_ext_submission.py
[ -f $B ] || cp experiments/a2_final_push_20260923/build_ext_submission.py $B
mkdir -p $O
i=0; pids=""
for g in 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g ~/cv2env/bin/python $B --ckpt $CK --out-dir $O --shard $i/6 > $O/log_shard$i.txt 2>&1 &
  pids="$pids $!"; i=$((i+1))
done
CUDA_VISIBLE_DEVICES=2 ~/cv2env/bin/python $B --ckpt $CK --out-dir $O --flops /tmp/etri_test/00017c73feca4fde > $O/log_flops.txt 2>&1 &
pids="$pids $!"
for p in $pids; do wait $p; done
~/cv2env/bin/python $B --ckpt $CK --out-dir $O --merge
~/cv2env/bin/python experiments/md_r0_reset_20260914/package_submission.py --submission $O/predictions.json \
  --flops-report $O/flops_report.json --clips-root /tmp/etri_test --out-dir $O/package --label $LABEL | tail -4
sha256sum $O/package/submission.zip
