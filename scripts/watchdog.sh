#!/usr/bin/env bash
# Keep a long training run alive.
#
# WHY: stage1 crashed at 2026-08-02 11:03 the moment its first in-training
# evaluation started (mmcv 1.7.2 touching a DDP internal torch 2.1 removed) and
# then sat idle for ~14.5 hours before anyone noticed. On a 3-day run that class
# of silent death is the main schedule risk, bigger than the crash itself.
#
# Every CHECK_EVERY seconds:
#   - if the trainer is alive          -> record a heartbeat, do nothing
#   - if it is gone and finished       -> stop watching
#   - if it is gone and unfinished     -> resume from the newest checkpoint
#
# Restarts are capped and backed off, because auto-restarting into a
# deterministic crash just burns GPU in a loop. After MAX_RESTARTS the watchdog
# gives up loudly and leaves the diagnosis to a human.
#
#   STAGE=stage1 nohup setsid bash watchdog.sh &
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
SCRATCH=${SCRATCH:-/tmp/pm97}
STAGE=${STAGE:-stage1}
CHECK_EVERY=${CHECK_EVERY:-300}
MAX_RESTARTS=${MAX_RESTARTS:-10}
TARGET_ITERS=${TARGET_ITERS:-351200}

PIDFILE=$PERSIST/logs/watchdog.$STAGE.pid
LOG=$PERSIST/logs/watchdog.$STAGE.log

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "watchdog already running as pid $(cat "$PIDFILE")"; exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

say() { printf '[%s] %s\n' "$(date -Is)" "$*" >> "$LOG"; }

# Count only real trainer processes. A `pgrep -f tools/train.py` pattern also
# matches the shell running this check, so filtering on /proc/PID/comm is what
# keeps the watchdog from seeing itself (and from killing its own caller).
trainer_count() {
  local n=0 p
  for p in $(pgrep -u "$USER" -f 'tools/train.py' 2>/dev/null); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && n=$((n+1))
  done
  echo "$n"
}

current_iter() {
  local l
  l=$(ls -t "$PERSIST"/logs/${STAGE}_2026*.log 2>/dev/null | head -1)
  [ -z "$l" ] && { echo 0; return; }
  tr '\r' '\n' < "$l" | grep -oE 'Iter \[[0-9]+/' | tail -1 | tr -dc '0-9'
}

# --- grad_norm surveillance ------------------------------------------------
# NaN is a lagging indicator: by the time it appears the model is already ruined.
# The 2026-08-02 divergence announced itself ~200 iters early as a monotone
# grad_norm climb (304 -> 810 -> 1549 -> nan) while the loss still looked sane.
# Watching the pre-clip grad_norm therefore buys hours of warning.
#
# Note the config clips to max_norm=25, so a logged grad_norm of 400 does NOT
# mean a 400-sized weight update — the update is bounded either way. What a large
# pre-clip norm signals is that raw gradients are approaching the fp16 range,
# which is the actual failure path (inf/nan cannot be repaired by clipping).
recent_gn() {   # print the last $1 logged grad_norm values, one per line
  local lg n=${1:-5}
  lg=$(ls -t "$PERSIST"/logs/${STAGE}_2026*.log 2>/dev/null | head -1)
  [ -z "$lg" ] && return
  tr '\r' '\n' < "$lg" | grep -oE 'grad_norm: [0-9.]+' \
    | awk -F': ' '{print $2}' | tail -"$n"
}

gn_all_above() {   # gn_all_above <count> <threshold>: true if ALL of the last
  local n=$1 thr=$2 vals c hits   # <count> samples exceed <threshold>
  vals=$(recent_gn "$n"); [ -z "$vals" ] && return 1
  c=$(printf '%s\n' "$vals" | grep -c .)
  [ "$c" -lt "$n" ] && return 1
  hits=$(printf '%s\n' "$vals" | awk -v t="$thr" '$1>t' | grep -c .)
  [ "$hits" -eq "$n" ]
}

GN_WARN=${GN_WARN:-300}     # sustained above this -> log a warning, keep going
# 800, not 1500. Both observed divergences (2026-08-04, iter ~21k and ~31.5k)
# crossed 800 and never came back down; the second one peaked at 1404.99 and so
# never tripped a 1500 threshold at all -- the NaN detector caught it instead,
# which is exactly the too-late outcome the gn check exists to avoid. Healthy
# stretches of this run sit at 20-40, so 800 is ~20x headroom over normal.
GN_STOP=${GN_STOP:-800}     # sustained above this -> diverging, stop the run

say "watchdog start: stage=$STAGE every=${CHECK_EVERY}s max_restarts=$MAX_RESTARTS target=$TARGET_ITERS"
say "  grad_norm surveillance: warn>$GN_WARN (5 pts), stop>$GN_STOP (3 pts); clip=25"
restarts=0

while true; do
  n=$(trainer_count)
  it=$(current_iter); it=${it:-0}
  # Resolve the log up front: the completion branch below reads it too, and under
  # `set -u` a variable only assigned inside the alive branch would be unset there.
  lg=$(ls -t "$PERSIST"/logs/${STAGE}_2026*.log 2>/dev/null | head -1)

  if [ "$n" -gt 0 ]; then
    # A diverged run is worse than a dead one: it burns GPU producing a ruined
    # model while looking perfectly healthy. stage1 went NaN at ~iter 100,164 on
    # 2026-08-04 and kept "training" for 5 more hours. Check the recent log tail
    # for NaN loss and stop the run so a human sees it, rather than restarting
    # into the same divergence.
    if [ -n "$lg" ] && tr '\r' '\n' < "$lg" | tail -40 | grep -qE 'loss: nan|grad_norm: nan'; then
      say "*** NaN DETECTED at iter=$it — stopping the run (diverged, not crashed) ***"
      say "    log: $lg"
      say "    recover: resume the newest CLEAN checkpoint with LOSS_SCALE=dynamic,"
      say "             and if it diverges again switch to bf16 (see BUILD_LOG.md)."
      for p in $(pgrep -u "$USER" -f 'tools/train.py' 2>/dev/null); do
        [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && kill "$p" 2>/dev/null
      done
      exit 2
    fi
    if gn_all_above 3 "$GN_STOP"; then
      say "*** grad_norm RUNAWAY at iter=$it — stopping before it goes NaN ***"
      say "    last 5 grad_norm: $(recent_gn 5 | tr '\n' ' ')"
      say "    log: $lg"
      say "    recover: resume the newest checkpoint; if it climbs again switch to"
      say "             bf16 (fp16 range is the suspected limit — see BUILD_LOG.md)."
      for p in $(pgrep -u "$USER" -f 'tools/train.py' 2>/dev/null); do
        [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && kill "$p" 2>/dev/null
      done
      exit 3
    fi

    if gn_all_above 5 "$GN_WARN"; then
      say "WARN grad_norm elevated (iter=$it): $(recent_gn 5 | tr '\n' ' ') — watching"
    else
      say "OK alive (procs=$n iter=$it/$TARGET_ITERS)"
    fi
  # Completion must NOT be judged from the last LOGGED iteration. log_config.interval
  # is 50 (mmdet logs every 51st iter), which does not divide the target, so the
  # final log line sits short of it -- stage2 finished at 5860 with its last log at
  # 5814. Treating that as a crash is what made the watchdog resume an already
  # complete run on 2026-08-06. The final checkpoint and the runner's own exit code
  # are the unambiguous signals.
  elif [ -f "$SCRATCH/work_dirs/$STAGE/iter_${TARGET_ITERS}.pth" ] \
    || [ -f "$PERSIST/ckpt_backup/$STAGE/iter_${TARGET_ITERS}.pth" ] \
    || { [ -n "$lg" ] && tr '\r' '\n' < "$lg" | tail -5 | grep -q '=== exited rc=0'; }; then
    say "DONE $STAGE finished (last logged iter=$it, target=$TARGET_ITERS) — watchdog exiting"
    exit 0
  elif [ "${it:-0}" -ge "$TARGET_ITERS" ]; then
    say "DONE reached $it/$TARGET_ITERS — watchdog exiting"
    exit 0
  else
    ckpt=$(ls -t "$SCRATCH"/work_dirs/$STAGE/iter_*.pth 2>/dev/null | head -1)
    [ -z "$ckpt" ] && ckpt=$(ls -t "$PERSIST"/ckpt_backup/$STAGE/iter_*.pth 2>/dev/null | head -1)

    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      say "GIVING UP: $restarts restarts already, trainer down at iter=$it. Human needed."
      say "  last log: $(ls -t "$PERSIST"/logs/${STAGE}_2026*.log 2>/dev/null | head -1)"
      exit 1
    fi

    restarts=$((restarts+1))
    if [ -n "$ckpt" ]; then
      say "DOWN at iter=$it -> restart #$restarts resuming from $(basename "$ckpt")"
      RESUME="$ckpt" nohup setsid bash "$PERSIST/scripts/launch_train.sh" "$STAGE" \
        >> "$PERSIST/logs/${STAGE}_runner.log" 2>&1 < /dev/null &
    else
      say "DOWN at iter=$it with NO checkpoint -> restart #$restarts from scratch"
      nohup setsid bash "$PERSIST/scripts/launch_train.sh" "$STAGE" \
        >> "$PERSIST/logs/${STAGE}_runner.log" 2>&1 < /dev/null &
    fi
    # back off so a deterministic crash does not spin: 5, 10, 20, ... minutes
    backoff=$(( CHECK_EVERY * (1 << (restarts > 4 ? 4 : restarts - 1)) ))
    say "  sleeping ${backoff}s before next check"
    sleep "$backoff"
    continue
  fi

  sleep "$CHECK_EVERY"
done
