#!/usr/bin/env bash
# Extract every downloaded archive into a single nuScenes root.
#
# Layout produced — ONE merged root, which is what upstream expects:
#   <root>/  maps/ maps/expansion/ samples/ sweeps/ can_bus/
#            v1.0-mini/ v1.0-trainval/ [v1.0-test/]
#
# Why merged and not a separate mini root: SparseDrive's scripts/create_data.sh
# runs the converter twice against a single --root-path ./data/nuscenes, once
# with --version v1.0-mini and once with --version v1.0. Splitting the roots
# breaks that script as shipped. Merging is safe because the 10 mini scenes are
# a subset of trainval, so the overlapping sample/sweep files are byte-identical
# and tar simply rewrites them with the same content. The version dirs
# (v1.0-mini/ vs v1.0-trainval/) keep the splits distinguishable.
#
# Usage:
#   ./extract_nuscenes.sh                                   # -> $PERSIST/data/nuscenes
#   NUSC_ROOT=$SCRATCH/data/nuscenes ./extract_nuscenes.sh   # -> local NVMe (training)
#   NUSC_ROOT=... PARALLEL=8 ./extract_nuscenes.sh           # 8 blobs at a time
#
# Idempotent PER DESTINATION: completion markers live under
# <root>/.extract_markers/, not next to the tarballs, so extracting the same
# archives to a second root does not get skipped because the first root already
# has them. (The original version kept markers in data/raw/ and had exactly that
# bug the moment a second destination was introduced.)
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/env_vars.sh"

RAW=$PERSIST/data/raw
ROOT=${NUSC_ROOT:-$PERSIST/data/nuscenes}
PARALLEL=${PARALLEL:-1}
MARK=$ROOT/.extract_markers
LOG=$PERSIST/logs/extract_$(basename "$(dirname "$ROOT")")_$(basename "$ROOT").log
mkdir -p "$ROOT" "$MARK" "$PERSIST/logs"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

PY=$(command -v python3 || echo "$MAMBA_ROOT_PREFIX/envs/drive/bin/python")

# One-time migration: adopt the old marker location so the already-populated
# original root is not needlessly re-extracted.
#
# ONLY for the default root. The legacy markers lived in data/raw/ next to the
# tarballs, i.e. they recorded "this archive was extracted *somewhere*", not
# "…into this root". Adopting them unconditionally marks every archive as done
# for a brand-new empty destination and the whole extraction silently no-ops.
if [ "$ROOT" = "$PERSIST/data/nuscenes" ]; then
  for f in "$RAW"/*.extracted; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .extracted)
    [ -f "$MARK/$n" ] || { : > "$MARK/$n"; log "adopted legacy marker for $n"; }
  done
fi

unzip_to() {  # unzip_to <zipfile> <destdir>   (unzip(1) is absent in this container)
  local z="$1" d="$2"
  mkdir -p "$d"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$z" -d "$d"
  else
    "$PY" - "$z" "$d" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as f:
    f.extractall(sys.argv[2])
PY
  fi
}

extract_one() {  # extract_one <archive-name>
  local name="$1" f="$RAW/$1"
  if [ ! -f "$f" ];             then log "MISS  $name (not downloaded) — skip"; return 0; fi
  if [ ! -f "$RAW/$name.ok" ];  then log "WARN  $name unverified download — skip"; return 0; fi
  if [ -f "$MARK/$name" ];      then log "SKIP  $name (already extracted here)"; return 0; fi

  log "TAR   $name -> $ROOT"
  local rc=0
  case "$name" in
    *.tgz|*.tar.gz) tar -xzf "$f" -C "$ROOT" || rc=$? ;;
    *.zip)          unzip_to "$f" "$ROOT"    || rc=$? ;;
  esac
  if [ $rc -ne 0 ]; then log "FAIL  $name — exit $rc"; return 1; fi
  : > "$MARK/$name"
  log "OK    $name"
}

# ---- metadata first (small, and defines the version dirs) -------------------
extract_one v1.0-mini.tgz
extract_one v1.0-trainval_meta.tgz
extract_one v1.0-test_meta.tgz

# ---- sensor blobs ----------------------------------------------------------
# Parallel is safe: the 10 trainval blobs hold disjoint scenes, so they write
# disjoint files. They do all create the same top-level dirs (samples/CAM_*,
# sweeps/...), and concurrent tar calls creating an existing directory is benign.
BLOBS=()
for i in 01 02 03 04 05 06 07 08 09 10; do BLOBS+=("v1.0-trainval${i}_blobs.tgz"); done
BLOBS+=(v1.0-test_blobs.tgz)

if [ "$PARALLEL" -le 1 ]; then
  for b in "${BLOBS[@]}"; do extract_one "$b"; done
else
  log "extracting blobs with PARALLEL=$PARALLEL"
  running=0
  for b in "${BLOBS[@]}"; do
    extract_one "$b" &
    running=$((running+1))
    if [ "$running" -ge "$PARALLEL" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
  done
  wait
fi

# ---- expansions ------------------------------------------------------------
# The map zip already contains expansion/ basemap/ prediction/, so it unpacks
# directly into maps/. can_bus.zip contains a can_bus/ dir, so it unpacks at the
# root — SparseDrive's create_data.sh passes --canbus ./data/nuscenes and looks
# for <canbus>/can_bus/ underneath.
n=nuScenes-map-expansion-v1.3.zip
if [ -f "$RAW/$n.ok" ] && [ ! -f "$MARK/$n" ]; then
  log "ZIP   $n -> $ROOT/maps/"; unzip_to "$RAW/$n" "$ROOT/maps" && : > "$MARK/$n"
else log "SKIP  $n"; fi

n=can_bus.zip
if [ -f "$RAW/$n.ok" ] && [ ! -f "$MARK/$n" ]; then
  log "ZIP   $n -> $ROOT/"; unzip_to "$RAW/$n" "$ROOT" && : > "$MARK/$n"
else log "SKIP  $n"; fi

# ---- structure assert ------------------------------------------------------
fail=0
assert_dir() {
  if [ -d "$1" ]; then log "  ✓ $1  ($(ls -1 "$1" 2>/dev/null | wc -l) entries)"
  else log "  ✗ MISSING $1"; fail=1; fi
}

log "===== asserting $ROOT ====="
for d in maps maps/expansion samples sweeps can_bus v1.0-trainval; do assert_dir "$ROOT/$d"; done
for d in v1.0-mini v1.0-test; do
  [ -d "$ROOT/$d" ] && assert_dir "$ROOT/$d" || log "  - $d not present"
done

# samples/ is shared by every split in this merged root, so these counts are a
# running total, not a per-split count.
log "===== keyframe counts in shared samples/ (running total) ====="
for cam in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT LIDAR_TOP; do
  log "  samples/$cam: $(ls -1 "$ROOT/samples/$cam" 2>/dev/null | wc -l)"
done
log "  (expect 404 = mini only | 34149 = full trainval | 40157 = trainval+test)"

if [ $fail -ne 0 ]; then log "===== STRUCTURE ASSERT FAILED ====="; exit 1; fi
log "===== structure OK ($(du -sh "$ROOT" 2>/dev/null | cut -f1)) ====="
