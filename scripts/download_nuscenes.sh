#!/usr/bin/env bash
# Phase 7 — nuScenes downloader.
#
# Reads urls.txt (one URL per line, '#' comments ignored) and fetches each file
# sequentially into $PERSIST/data/raw/ with aria2c -x16 -s16 -c (resumable).
#
# Idempotent: a file with a matching remote size and a passing integrity check
# gets a .ok marker and is skipped on re-run.
#
# INTEGRITY NOTE: nuScenes does not publish official md5 checksums for these
# archives, so there is no authoritative digest to compare against. Instead each
# download is verified by:
#   (1) exact byte-size match vs the server's Content-Length, and
#   (2) container-level CRC check — `gzip -t` for .tgz, zipfile.testzip() for
#       .zip — which validates every compressed member.
# The observed sha256 is then recorded in CHECKSUMS.sha256 so a later re-download
# on another machine can be diffed against this run.
#
# CONCURRENCY: measured on this host 2026-07-31 — a single file with -x16 gets
# ~11 MiB/s, and that is NOT a per-connection cap: raising -x further does
# nothing. Running a 2nd file concurrently yields ~5.5 + ~10 = ~15.5 MiB/s
# aggregate (+40%). A 3rd and 4th concurrent file get starved outright
# (CN:1, DL:0B), so the CDN caps concurrent transfers per client at 2.
# Therefore: run TWO workers, one forward and one reverse over urls.txt, and let
# them meet in the middle. They coordinate through <name>.lock claim dirs.
#
# Usage:
#   ./download_nuscenes.sh                        # single worker, whole list
#   WORKER=fwd ./download_nuscenes.sh &           # worker 1, top-down
#   WORKER=rev REVERSE=1 ./download_nuscenes.sh & # worker 2, bottom-up
#   ./download_nuscenes.sh mini                   # only filenames matching 'mini'
#   FILTER=trainval0 ./download_nuscenes.sh
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/env_vars.sh"

RAW=$PERSIST/data/raw
URLS=${URLS:-$PERSIST/scripts/urls.txt}
FILTER=${FILTER:-${1:-}}
WORKER=${WORKER:-main}
REVERSE=${REVERSE:-0}
SUMS=$RAW/CHECKSUMS.sha256
LOG=$PERSIST/logs/download_nuscenes.$WORKER.log

mkdir -p "$RAW" "$PERSIST/logs"
touch "$SUMS"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

# Record our own PID so this run can be stopped precisely.
# Do NOT stop it with `pkill -f download_nuscenes` — that pattern also matches
# the shell that issues the pkill, and it will kill the caller too.
# Correct: kill "$(cat $PERSIST/logs/download.pid)"
PIDFILE=$PERSIST/logs/download.$WORKER.pid
echo $$ > "$PIDFILE"
CLAIMED=""   # lock dir currently held, so a crash does not leave it stuck
LIST=""      # temp copy of the URL list (set later)
cleanup() {
  rm -f "$PIDFILE"
  [ -n "$LIST" ] && rm -f "$LIST"
  [ -n "$CLAIMED" ] && rmdir "$CLAIMED" 2>/dev/null
  return 0
}
trap cleanup EXIT
# Kill our aria2c child if we are terminated, so it does not keep running orphaned.
trap 'pkill -P $$ -x aria2c 2>/dev/null; cleanup; exit 143' TERM INT

# Claim a file so the other worker skips it. mkdir is atomic, so whichever
# worker wins the race owns the download; the loser moves on to the next URL.
# Stale locks (worker killed with -9) are reclaimed after LOCK_STALE_SEC.
LOCK_STALE_SEC=${LOCK_STALE_SEC:-900}
claim() {
  local d="$RAW/$1.lock"
  if mkdir "$d" 2>/dev/null; then CLAIMED="$d"; return 0; fi
  # someone holds it — steal only if clearly abandoned
  local age=$(( $(date +%s) - $(stat -c %Y "$d" 2>/dev/null || date +%s) ))
  if [ "$age" -gt "$LOCK_STALE_SEC" ] && ! pgrep -x aria2c >/dev/null; then
    log "      reclaiming stale lock on $1 (${age}s old, no aria2c running)"
    CLAIMED="$d"; touch "$d"; return 0
  fi
  return 1
}
release() { [ -n "$CLAIMED" ] && rmdir "$CLAIMED" 2>/dev/null; CLAIMED=""; }

# --- pick a downloader -------------------------------------------------------
DL=""
if command -v aria2c >/dev/null 2>&1; then
  DL=aria2c
elif [ -x "$MAMBA_ROOT_PREFIX/envs/drive/bin/aria2c" ]; then
  DL="$MAMBA_ROOT_PREFIX/envs/drive/bin/aria2c"
fi
if [ -z "$DL" ]; then
  log "WARN: aria2c not found; falling back to wget -c (slower, single stream)."
  log "      Install it with: micromamba install -n drive -c conda-forge aria2"
  DL=wget
fi
log "downloader: $DL"

# --- remote size -------------------------------------------------------------
remote_size() {
  # HEAD is unreliable on this CDN for some keys (returns 400), so use a 1-byte
  # ranged GET and read the total out of Content-Range.
  # NOTE: must not use awk's IGNORECASE — this container ships mawk, which
  # ignores it, so /^content-range:/ silently never matches "Content-Range:"
  # and every size check gets skipped. grep -i is portable.
  curl -sS -m 30 -r 0-0 -D - -o /dev/null "$1" 2>/dev/null \
    | tr -d '\r' | grep -i '^content-range:' | head -1 | sed 's#.*/##'
}

# --- integrity check ---------------------------------------------------------
verify_archive() {
  local f="$1"
  case "$f" in
    *.tgz|*.tar.gz)
      gzip -t "$f" 2>&1 && return 0 || return 1
      ;;
    *.zip)
      local py
      py=$(command -v python3 || echo "$MAMBA_ROOT_PREFIX/envs/drive/bin/python")
      "$py" - "$f" <<'PY'
import sys, zipfile
p = sys.argv[1]
try:
    with zipfile.ZipFile(p) as z:
        bad = z.testzip()
    sys.exit(1 if bad else 0)
except Exception as e:
    print(e, file=sys.stderr); sys.exit(1)
PY
      ;;
    *) return 0 ;;
  esac
}

total=0; done_n=0; skip_n=0; fail_n=0
declare -a FAILED=()

# Materialize the URL list first so REVERSE can flip it, and so the loop does
# not hold a read handle on urls.txt for hours.
LIST=$(mktemp "${TMPDIR:-/tmp}/nusc_urls.XXXXXX")   # cleaned up by cleanup()
if [ "$REVERSE" = "1" ]; then
  tac "$URLS" > "$LIST"; log "[$WORKER] iterating urls.txt in REVERSE"
else
  cat "$URLS" > "$LIST"; log "[$WORKER] iterating urls.txt forward"
fi

while IFS= read -r url; do
  url="${url%%#*}"; url="$(printf '%s' "$url" | tr -d '[:space:]')"
  [ -z "$url" ] && continue
  name="${url##*/}"
  [ -n "$FILTER" ] && case "$name" in *"$FILTER"*) ;; *) continue ;; esac
  total=$((total+1))

  if [ -f "$RAW/$name.ok" ]; then
    log "SKIP  $name (already verified)"
    skip_n=$((skip_n+1)); continue
  fi

  if ! claim "$name"; then
    log "LOCK  $name (held by the other worker) — skipping"
    skip_n=$((skip_n+1)); continue
  fi

  rsize=$(remote_size "$url")
  log "GET   $name  (remote ${rsize:-unknown} bytes)"

  if [ "$DL" = "wget" ]; then
    wget -c -q --progress=dot:giga -O "$RAW/$name" "$url" 2>>"$LOG"
  else
    "$DL" -x16 -s16 -c -k 1M --console-log-level=warn --summary-interval=60 \
          --auto-file-renaming=false --allow-overwrite=true \
          -d "$RAW" -o "$name" "$url" >>"$LOG" 2>&1
  fi
  rc=$?

  lsize=$(stat -c %s "$RAW/$name" 2>/dev/null || echo 0)
  if [ "$rc" -ne 0 ]; then
    log "FAIL  $name — downloader exit $rc (got $lsize bytes). Re-run to resume."
    FAILED+=("$name"); fail_n=$((fail_n+1)); release; continue
  fi
  if [ -n "$rsize" ] && [ "$lsize" != "$rsize" ]; then
    log "FAIL  $name — size mismatch: local $lsize != remote $rsize. Re-run to resume."
    FAILED+=("$name"); fail_n=$((fail_n+1)); release; continue
  fi

  log "      integrity check (CRC over all members)…"
  if ! verify_archive "$RAW/$name"; then
    log "FAIL  $name — archive CRC check FAILED. File is corrupt; delete it and re-run."
    FAILED+=("$name"); fail_n=$((fail_n+1)); release; continue
  fi

  # Per-file sidecar, not an append to one shared file: two workers appending to
  # CHECKSUMS.sha256 concurrently would interleave and lose lines. The combined
  # manifest is regenerated from the sidecars below.
  sha=$(sha256sum "$RAW/$name" | awk '{print $1}')
  printf '%s  %s\n' "$sha" "$name" > "$RAW/$name.sha256"
  : > "$RAW/$name.ok"
  log "OK    $name  ($lsize bytes, sha256 ${sha:0:16}…)"
  done_n=$((done_n+1))
  release
done < "$LIST"

# Rebuild the combined manifest from sidecars (idempotent, race-free).
# Glob the archive extensions only — a bare *.sha256 would also match $SUMS
# itself and duplicate every line back into it on each run.
cat "$RAW"/*.tgz.sha256 "$RAW"/*.zip.sha256 2>/dev/null | sort -u -k2 > "$SUMS.tmp" \
  && mv "$SUMS.tmp" "$SUMS"

log "===== [$WORKER] summary: $total considered / $done_n downloaded+verified / $skip_n skipped-or-locked / $fail_n failed ====="
if [ ${#FAILED[@]} -gt 0 ]; then
  log "[$WORKER] failed: ${FAILED[*]}"
  exit 1
fi
