#!/usr/bin/env bash
# Phase 6 — restore the training environment on a freshly reset container.
#
# Assumes ONLY that $PERSIST survived (it is on NFS) and that /tmp is empty.
# Everything else — the conda env, the compiled extensions — is rebuilt from the
# artifacts under $PERSIST.
#
#   (a) unpack env_drive_*.tar.gz  ->  $TARGET_PREFIX   + conda-unpack
#   (b) install $PERSIST/wheels/*.whl  (mmcv-full, deformable_aggregation_ext)
#   (c) export the toolchain env vars
#   (d) run verify_env.py and refuse to report success unless it passes
#
# Idempotent: if the target env already verifies, it exits 0 without touching
# anything. Re-running after a partial failure resumes cleanly.
#
# Usage:
#   bash setup_env.sh                       # restore to the normal location
#   TARGET_PREFIX=/tmp/pm97_restore_test/drive bash setup_env.sh   # rehearsal
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
source "$PERSIST/scripts/env_vars.sh"

TARGET_PREFIX=${TARGET_PREFIX:-$MAMBA_ROOT_PREFIX/envs/drive}
LOG=$PERSIST/logs/setup_env_$(date +%F_%H%M%S).log

say() { printf '\n\033[1m== %s\033[0m\n' "$*" | tee -a "$LOG"; }
info() { printf '   %s\n' "$*" | tee -a "$LOG"; }
die() { printf '\n\033[31mFATAL: %s\033[0m\n' "$*" | tee -a "$LOG"; exit 1; }

# Point the toolchain at TARGET_PREFIX (which may differ from the packed env's
# original prefix during a rehearsal) and run the gate.
activate_and_verify() {
  export CONDA_PREFIX="$TARGET_PREFIX"
  export CUDA_HOME="$TARGET_PREFIX"
  export CC="$TARGET_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$TARGET_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX"
  export NVCC_PREPEND_FLAGS="-ccbin $CXX"
  # conda bin FIRST so the system CUDA 13.2 under /usr/local can never win
  export PATH="$TARGET_PREFIX/bin:$PATH"
  export LD_LIBRARY_PATH="$TARGET_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  hash -r
  "$TARGET_PREFIX/bin/python" "$PERSIST/scripts/verify_env.py" 2>&1 \
    | grep -vE 'UserWarning|warnings\.warn|from pkg_resources|MMCV will release' \
    | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

say "setup_env.sh  ($(date -Is))"
info "PERSIST       = $PERSIST"
info "TARGET_PREFIX = $TARGET_PREFIX"
info "log           = $LOG"

# ---------------------------------------------------------------- (0) fast path
if [ -x "$TARGET_PREFIX/bin/python" ]; then
  say "(0) env already present — checking whether it still verifies"
  if activate_and_verify >/dev/null 2>&1; then
    say "env at $TARGET_PREFIX already verifies — nothing to do"
    exit 0
  fi
  info "present but does not verify; continuing with restore"
fi

# ------------------------------------------------------------- (a) unpack env
say "(a) unpack conda-pack tarball"
PACK=$(ls -1t "$PERSIST"/env_drive_*.tar.gz 2>/dev/null | head -1)
[ -n "$PACK" ] || die "no env_drive_*.tar.gz in $PERSIST — cannot restore.
       Rebuild from scratch with: bash $PERSIST/scripts/phase1_create_env.sh
       then follow BUILD_LOG.md phases 2-4."
info "using $PACK ($(du -h "$PACK" | cut -f1))"

if [ -x "$TARGET_PREFIX/bin/python" ]; then
  info "target already populated — skipping extract"
else
  mkdir -p "$TARGET_PREFIX"
  tar -xzf "$PACK" -C "$TARGET_PREFIX" || die "extract failed"
  info "extracted"
fi

# conda-unpack rewrites the absolute prefixes baked into scripts and .so RPATHs.
# It is a one-shot: it deletes itself afterwards, hence the -e guard.
if [ -x "$TARGET_PREFIX/bin/conda-unpack" ]; then
  info "running conda-unpack"
  # Invoke it through the unpacked env's own interpreter. conda-unpack's shebang
  # is `#!/usr/bin/env python`, and at this point the env is not on PATH yet —
  # and this container has no system `python` at all — so executing the script
  # directly dies with "/usr/bin/env: 'python': No such file or directory".
  "$TARGET_PREFIX/bin/python" "$TARGET_PREFIX/bin/conda-unpack" \
    || die "conda-unpack failed"
else
  info "conda-unpack already consumed (env was previously unpacked)"
fi

# ---------------------------------------------------------- (b) install wheels
say "(b) install locally-built wheels"
shopt -s nullglob
WHEELS=("$PERSIST"/wheels/*.whl)
shopt -u nullglob
if [ ${#WHEELS[@]} -eq 0 ]; then
  info "no wheels in $PERSIST/wheels (they may already be inside the packed env)"
else
  for w in "${WHEELS[@]}"; do info "  $(basename "$w")"; done
  # --no-deps: the packed env already has the exact pinned dependency set, and
  # letting pip resolve here is precisely how numpy got bumped to 2.x twice
  # during the original build.
  #
  # `python -m pip`, not `bin/pip`: the console scripts in a conda-packed env
  # carry a `#!/usr/bin/env python3.9` shebang, and the env is not on PATH yet
  # at this point (nor does this container have a system python3.9), so calling
  # bin/pip directly fails with "/usr/bin/env: 'python3.9': No such file".
  # Putting the env on PATH first also makes those scripts work.
  PATH="$TARGET_PREFIX/bin:$PATH" \
  "$TARGET_PREFIX/bin/python" -m pip install --no-deps --no-cache-dir \
      --force-reinstall "${WHEELS[@]}" >>"$LOG" 2>&1 \
      || die "wheel install failed (see $LOG)"
  info "installed"
fi

# ------------------------------------------------------- (c)+(d) export + verify
say "(c) export toolchain env vars / (d) verify"
activate_and_verify || die "verify_env.py FAILED — environment is NOT ready (see $LOG)"

say "RESTORE COMPLETE"
cat <<EOF | tee -a "$LOG"

   Activate it in a new shell with:

     source $PERSIST/scripts/env_vars.sh --activate

   If TARGET_PREFIX was overridden for a rehearsal, activate that prefix directly:

     export PATH=$TARGET_PREFIX/bin:\$PATH
     export CUDA_HOME=$TARGET_PREFIX
     export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

   Next steps for training: $PERSIST/README_NEXT.md
EOF
