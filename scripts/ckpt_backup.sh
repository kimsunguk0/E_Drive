#!/usr/bin/env bash
# Mirror training checkpoints from volatile $SCRATCH to persistent $PERSIST every
# 30 minutes.
#
# $SCRATCH is local NVMe and DISAPPEARS on container reset. A stage takes ~3.6
# days, so losing it means losing days of compute; the checkpoints are the single
# most expensive artifact produced here.
#
# Single-instance guard uses a pidfile, NOT `pgrep -f ckpt_backup.sh` — that
# pattern matches the shell doing the check, so the guard would self-trigger
# (and `pkill -f` on it kills the caller).
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
SCRATCH=${SCRATCH:-/tmp/pm97}
INTERVAL=${INTERVAL:-1800}

PIDFILE=$PERSIST/logs/ckpt_backup.pid
LOG=$PERSIST/logs/ckpt_backup.log

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running as pid $(cat "$PIDFILE")"; exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

mkdir -p "$PERSIST/ckpt_backup"

while true; do
  {
    printf '\n===== %s =====\n' "$(date -Is)"
    for wd in "$SCRATCH"/work_dirs/*/; do
      [ -d "$wd" ] || continue
      name=$(basename "$wd")
      case "$name" in smoke_*) continue ;; esac      # smoke runs are disposable
      dest=$PERSIST/ckpt_backup/$name
      mkdir -p "$dest"
      # --inplace avoids a second full-size temp copy on NFS for multi-GB files.
      # Keep logs and the config too: a checkpoint without the config that made
      # it is far less useful.
      rsync -a --inplace \
            --include='*/' \
            --include='*.pth' --include='*.py' --include='*.log' \
            --include='*.log.json' --include='*.txt' \
            --exclude='*' \
            "$wd" "$dest/" 2>&1 && echo "  synced $name -> $dest"
      latest=$(ls -t "$dest"/*.pth 2>/dev/null | head -1)
      [ -n "$latest" ] && echo "    latest: $(basename "$latest") $(stat -c %s "$latest") bytes"
    done
    df -h "$PERSIST" | tail -1 | sed 's/^/  NFS: /'
  } >> "$LOG" 2>&1
  sleep "$INTERVAL"
done
