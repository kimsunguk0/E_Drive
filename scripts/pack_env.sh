#!/usr/bin/env bash
# Phase 6 — freeze the working environment into $PERSIST so a container reset
# costs ~30 minutes instead of a full rebuild.
#
# Produces:
#   $PERSIST/env_drive_<date>.tar.gz   conda-pack of the whole env
#   $PERSIST/env_snapshot/pip_freeze.txt
#   $PERSIST/env_snapshot/micromamba_env_export.yml
#   $PERSIST/env_snapshot/nvidia-smi.txt
#   $PERSIST/env_snapshot/env_vars.txt          (the exact exports used to build)
#   $PERSIST/env_snapshot/cuobjdump_archs.txt   (proof the ext carries sm_86/89/90)
#
# Run this only when verify_env.py passes — packing a broken env is worse than
# not packing at all, so that is checked first.
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/env_vars.sh" --activate

SNAP=$PERSIST/env_snapshot
OUT=$PERSIST/env_drive_$(date +%F).tar.gz
mkdir -p "$SNAP"

say() { printf '\n== %s\n' "$*"; }

say "0/4 gate: verify_env.py must pass before packing"
if ! python "$PERSIST/scripts/verify_env.py" > "$SNAP/verify_at_pack_time.txt" 2>&1; then
  echo "REFUSING TO PACK: verify_env.py failed. See $SNAP/verify_at_pack_time.txt"
  grep -E '✗|FAIL' "$SNAP/verify_at_pack_time.txt" | head
  exit 1
fi
grep -E 'passed|READY' "$SNAP/verify_at_pack_time.txt" | tail -2

say "1/4 environment snapshot"
pip freeze > "$SNAP/pip_freeze.txt"
"$PERSIST/bin/micromamba" env export -n drive > "$SNAP/micromamba_env_export.yml" 2>/dev/null \
  || echo "(micromamba env export failed — pip_freeze.txt is the authoritative list)" \
     > "$SNAP/micromamba_env_export.yml"
nvidia-smi > "$SNAP/nvidia-smi.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
           --format=csv >> "$SNAP/nvidia-smi.txt" 2>&1

cat > "$SNAP/env_vars.txt" <<EOF
# Exact toolchain environment used to build every wheel in $PERSIST/wheels.
# Reproduce with: source $PERSIST/scripts/env_vars.sh --activate
PERSIST=$PERSIST
SCRATCH=$SCRATCH
MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX
CONDA_PREFIX=$CONDA_PREFIX
CC=$CC
CXX=$CXX
CUDAHOSTCXX=$CUDAHOSTCXX
NVCC_PREPEND_FLAGS=$NVCC_PREPEND_FLAGS
CUDA_HOME=$CUDA_HOME
TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
MAX_JOBS=$MAX_JOBS
PATH=$PATH
# versions
nvcc=$(nvcc --version | tail -2 | head -1 | tr -s ' ')
gcc=$($CC --version | head -1)
python=$(python -V 2>&1)
EOF

say "2/4 record embedded GPU archs (the thing that silently breaks on the 4090)"
{
  for so in "$CONDA_PREFIX"/lib/python3.9/site-packages/mmcv/_ext*.so \
            "$CONDA_PREFIX"/lib/python3.9/site-packages/deformable_aggregation_ext*.so; do
    [ -e "$so" ] || continue
    echo "### $so"
    cuobjdump --list-elf "$so" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort | uniq -c
  done
} > "$SNAP/cuobjdump_archs.txt"
cat "$SNAP/cuobjdump_archs.txt"

say "3/4 conda-pack"
python -m pip install --no-cache-dir -q conda-pack 2>&1 | tail -2
# --ignore-missing-files: pip-installed packages that were later overwritten by a
# different version leave stale records; conda-pack aborts on those by default.
conda-pack -p "$CONDA_PREFIX" -o "$OUT" --force --ignore-missing-files -j 8
ls -lh "$OUT"

say "4/4 done"
echo "  pack     : $OUT  ($(du -h "$OUT" | cut -f1))"
echo "  snapshot : $SNAP"
echo "  restore  : bash $PERSIST/scripts/setup_env.sh"
