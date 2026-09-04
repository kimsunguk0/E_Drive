#!/usr/bin/env bash
# Single source of truth for paths + toolchain env vars.
# Usage:  source /home/pm97/workspace/sukim/adcl/scripts/env_vars.sh [--activate]
# Safe to source repeatedly (idempotent).

export PERSIST=/home/pm97/workspace/sukim/adcl
export SCRATCH=/tmp/pm97
export MAMBA_ROOT_PREFIX=$SCRATCH/mamba

# --- Build toolchain pins (Phase 1 spec) ---
# H200(sm_90) build host, but 4090(sm_89) is the organizer's scoring GPU and
# 3090(sm_86) is a fallback. NEVER narrow this list.
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"

# nproc is 128, but PID 32531 (another training run, same UID) shares this CPU.
# Absolute rule 1 (do not disturb other processes) outranks the spec's
# MAX_JOBS=$(nproc), so we cap at 32. See BUILD_LOG.md.
export MAX_JOBS=${MAX_JOBS:-32}

if [ "${1:-}" = "--activate" ]; then
  # micromamba's activate.d hooks (binutils, gcc) read variables like ADDR2LINE
  # before setting them, so they blow up under `set -u`. Drop nounset for the
  # activation and restore the caller's setting afterwards.
  _had_u=0; case "$-" in *u*) _had_u=1;; esac
  set +u

  eval "$("$PERSIST/bin/micromamba" shell hook -s bash)"
  micromamba activate drive

  export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
  export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
  export CUDAHOSTCXX=$CXX
  export NVCC_PREPEND_FLAGS="-ccbin $CXX"
  export CUDA_HOME=$CONDA_PREFIX
  # Put conda bin FIRST so the system CUDA 13.2 under /usr/local can never win.
  export PATH=$CONDA_PREFIX/bin:$PATH
  hash -r

  [ "$_had_u" = "1" ] && set -u
  unset _had_u
fi
