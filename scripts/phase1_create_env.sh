#!/usr/bin/env bash
# Phase 1 — micromamba env "drive" + CUDA 12.1 toolchain
# Idempotent: safe to re-run.
set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/env_vars.sh"

mkdir -p "$SCRATCH"
eval "$("$PERSIST/bin/micromamba" shell hook -s bash)"

echo "=== [1/2] create env drive (python 3.9 / gxx 11 / ninja / cmake / git / aria2) ==="
# aria2 added here: Phase 7 requires aria2c and it is absent from the container.
if micromamba env list | grep -qE '^\s+drive\s'; then
  echo "env 'drive' already exists — skipping create"
else
  micromamba create -n drive -y -c conda-forge \
    python=3.9 gxx_linux-64=11 ninja cmake git aria2
fi

echo "=== [2/2] install CUDA toolkit 12.1 ==="
micromamba install -n drive -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit

echo "=== DONE Phase 1 env creation ==="
micromamba run -n drive python -V
micromamba run -n drive bash -lc 'echo "CONDA_PREFIX=$CONDA_PREFIX"; ls -l $CONDA_PREFIX/bin/nvcc; $CONDA_PREFIX/bin/nvcc --version | tail -2'
