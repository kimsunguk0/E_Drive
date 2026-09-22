#!/usr/bin/env bash
# Scrub, verify, then push the mirror. Refuses to push if anything still matches.
#
# The working repository's history carries B200 connection details and this
# mirror is public, so every push has to be scrubbed. Doing that by hand failed
# twice: experiment launchers keep hardcoding the interpreter path, and a guard
# that runs beside the push instead of before it does not stop anything.
set -euo pipefail

PATTERN='59\.150\.32\.1|korea_sdv01|DCTN-beyless_key'
BRANCH="${1:-motiondrive-v2-20260910}"
cd ~/edrive_mirror

mapfile -t leaky < <(git grep -lIE "$PATTERN" HEAD -- | sed 's/^HEAD://' | sort -u)
if [ "${#leaky[@]}" -gt 0 ]; then
    printf 'scrubbing %d file(s)\n' "${#leaky[@]}"
    for f in "${leaky[@]}"; do
        [ -f "$f" ] || continue
        sed -i "s#/home/korea_sdv01#/home/<B200-USER>#g; \
                s/korea_sdv01/<B200-USER>/g; \
                s/DCTN-beyless_key/<B200-KEY>/g; \
                s/59\.150\.32\.1/<B200-IP>/g" "$f"
    done
    git add -u
    git diff --cached --quiet || git commit -q -m "Scrub connection details before mirror push"
fi

tree_hits=$(git grep -cIE "$PATTERN" HEAD -- 2>/dev/null | wc -l)
msg_hits=$(git log --format=%B -50 | grep -cIE "$PATTERN" || true)
printf 'guard: tree %s, recent messages %s\n' "$tree_hits" "$msg_hits"
if [ "$tree_hits" -ne 0 ] || [ "$msg_hits" -ne 0 ]; then
    echo "REFUSING TO PUSH: connection details still present" >&2
    exit 1
fi

git push github "HEAD:$BRANCH"
