#!/usr/bin/env bash
# Ranged-chunk downloader for objects that CloudFront refuses to serve whole.
#
# WHY THIS EXISTS
# v1.0-test_blobs.tgz is 57,596,693,546 bytes (53.6 GiB) and exceeds
# CloudFront's maximum object size, so a GET *without* a Range header returns
# HTTP 400 while any ranged GET returns 206 normally. aria2c's initial
# size-probe request carries no Range header, so it takes the 400 and dies with
# exit 22 before it ever starts. Verified 2026-07-31:
#
#   plain GET  v1.0-trainval10_blobs.tgz  (38.9 GiB) -> 200
#   plain GET  v1.0-test_blobs.tgz        (53.6 GiB) -> 400
#   ranged GET v1.0-test_blobs.tgz        any range  -> 206
#
# So we drive the ranges ourselves: append fixed-size chunks, resuming from
# whatever is already on disk. Single connection (~11 MiB/s here, matching what
# aria2c -x16 achieves anyway, since the cap is not per-connection).
#
# Usage: ./download_large_ranged.sh <url> [chunk_bytes]
#        ./download_large_ranged.sh https://.../v1.0-test_blobs.tgz
# Idempotent and resumable: re-run after an interruption and it continues.
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/env_vars.sh"

URL=${1:?usage: download_large_ranged.sh <url> [chunk_bytes]}
CHUNK=${2:-$((1024 * 1024 * 1024))}     # 1 GiB per request
RAW=$PERSIST/data/raw
NAME=${URL##*/}
DEST=$RAW/$NAME
LOG=$PERSIST/logs/download_ranged.$NAME.log
mkdir -p "$RAW" "$PERSIST/logs"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

TOTAL=$(curl -sS -m 30 -r 0-0 -D - -o /dev/null "$URL" 2>/dev/null \
        | tr -d '\r' | grep -i '^content-range:' | head -1 | sed 's#.*/##')
if [ -z "$TOTAL" ]; then
  log "FATAL: could not determine remote size for $NAME"
  exit 1
fi

if [ -f "$DEST.ok" ]; then
  log "SKIP $NAME (already verified)"
  exit 0
fi

OFF=$(stat -c %s "$DEST" 2>/dev/null || echo 0)
log "$NAME: total $TOTAL bytes, resuming at $OFF ($(awk -v o="$OFF" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*o/t}')%), chunk $CHUNK"

while [ "$OFF" -lt "$TOTAL" ]; do
  END=$((OFF + CHUNK - 1))
  [ "$END" -ge "$TOTAL" ] && END=$((TOTAL - 1))

  # >> appends, so a partial chunk write leaves the file short but never
  # corrupt in the middle; the next iteration recomputes OFF from the file size.
  if ! curl -sS -f --retry 5 --retry-delay 5 --retry-all-errors \
            -r "$OFF-$END" "$URL" >> "$DEST"; then
    log "chunk $OFF-$END failed; will retry from actual file size"
    sleep 5
  fi

  NEW=$(stat -c %s "$DEST" 2>/dev/null || echo 0)
  if [ "$NEW" -le "$OFF" ]; then
    log "no progress at offset $OFF — aborting to avoid a spin loop"
    exit 1
  fi
  OFF=$NEW
  log "  $(awk -v o="$OFF" -v t="$TOTAL" 'BEGIN{printf "%6.2f%%", 100*o/t}')  $OFF / $TOTAL"
done

LSIZE=$(stat -c %s "$DEST")
if [ "$LSIZE" != "$TOTAL" ]; then
  log "FAIL $NAME: size $LSIZE != expected $TOTAL"
  exit 1
fi

log "size OK; integrity check (gzip CRC over all members)…"
if ! gzip -t "$DEST"; then
  log "FAIL $NAME: gzip CRC check failed — file is corrupt"
  exit 1
fi

SHA=$(sha256sum "$DEST" | awk '{print $1}')
printf '%s  %s\n' "$SHA" "$NAME" > "$DEST.sha256"
: > "$DEST.ok"
log "OK   $NAME ($LSIZE bytes, sha256 ${SHA:0:16}…)"
