#!/usr/bin/env bash
# Block until the challenge download actually starts producing files.
#
# WHY: the Drive folder is currently returning
#   "Too many users have viewed or downloaded this file recently"
# for every file, so download_challenge.py is sitting in its retry loop. Google
# says the quota can take up to 24 h to clear. Polling that by hand wastes turns;
# this exits the moment the first file completes, or if the downloader dies.
#
#   nohup setsid bash wait_for_download.sh &
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
DEST=${DEST:-$PERSIST/data/challenge}
LOG=${LOG:-$PERSIST/logs/download_challenge.log}
EVERY=${EVERY:-120}
MAX_WAIT=${MAX_WAIT:-86400}      # 24 h -- Google's own stated worst case

# Count only the real python worker. `pgrep -f download_challenge` also matches
# the shell running this check, and killing/counting that has bitten this project
# repeatedly (exit 144, five times). Filter on /proc/PID/comm.
downloader_alive() {
  local p
  for p in $(pgrep -u "$USER" -f 'download_challenge.py' 2>/dev/null); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && return 0
  done
  return 1
}

# A finished file is one that is NOT still a .part temp.
completed_count() {
  find "$DEST" -type f ! -name '*.part' 2>/dev/null | wc -l
}

start=$(date +%s)
while :; do
  n=$(completed_count)
  if [ "$n" -gt 0 ]; then
    echo "RESULT=STARTED files=$n"
    find "$DEST" -type f ! -name '*.part' -printf '  %p  %s bytes\n' 2>/dev/null | head -5
    exit 0
  fi

  if ! downloader_alive; then
    # Ran out of passes, or crashed. Either way a human decision is needed.
    echo "RESULT=DOWNLOADER_GONE files=0"
    tail -3 "$LOG" 2>/dev/null | sed 's/^/  /'
    exit 3
  fi

  if [ $(( $(date +%s) - start )) -ge "$MAX_WAIT" ]; then
    echo "RESULT=TIMEOUT files=0 waited=${MAX_WAIT}s"
    exit 4
  fi
  sleep "$EVERY"
done
