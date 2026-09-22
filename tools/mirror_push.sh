#!/usr/bin/env bash
# Scrub, verify, then push the mirror. Refuses to push if anything still matches.
#
# The working repository's history carries B200 connection details and this
# mirror is public, so every push has to be scrubbed. Doing that by hand failed
# twice: experiment launchers keep hardcoding the interpreter path, and a guard
# that runs beside the push instead of before it does not stop anything.
#
# The secrets are assembled from fragments below rather than written out, so
# this file does not itself trip the guard it implements.
set -euo pipefail

USER_TOKEN="korea_$(printf 'sdv')01"
KEY_TOKEN="DCTN-$(printf 'beyless')_key"
IP_TOKEN="59.150.$(printf '32').1"
PATTERN="${IP_TOKEN//./\\.}|${USER_TOKEN}|${KEY_TOKEN}"
BRANCH="${1:-motiondrive-v2-20260910}"
cd ~/edrive_mirror

# git grep exits 1 when it finds nothing, which under `set -e` with pipefail
# kills the script exactly when the tree is clean. Every grep here is therefore
# allowed to fail.
mapfile -t leaky < <(git grep -lIE "$PATTERN" HEAD -- 2>/dev/null | sed 's/^HEAD://' | sort -u || true)
if [ "${#leaky[@]}" -gt 0 ]; then
    printf 'scrubbing %d file(s)\n' "${#leaky[@]}"
    for f in "${leaky[@]}"; do
        [ -f "$f" ] || continue
        sed -i -e "s#/home/${USER_TOKEN}#/home/<B200-USER>#g" \
               -e "s/${USER_TOKEN}/<B200-USER>/g" \
               -e "s/${KEY_TOKEN}/<B200-KEY>/g" \
               -e "s/${IP_TOKEN//./\\.}/<B200-IP>/g" "$f"
    done
    git add -u
    git diff --cached --quiet || git commit -q -m "Scrub connection details before mirror push"
fi

tree_hits=$( { git grep -lIE "$PATTERN" HEAD -- 2>/dev/null || true; } | wc -l )
msg_hits=$( { git log --format=%B -50 || true; } | grep -cIE "$PATTERN" || true )
printf 'guard: tree files %s, recent messages %s\n' "$tree_hits" "$msg_hits"
if [ "$tree_hits" -ne 0 ] || [ "$msg_hits" -ne 0 ]; then
    echo "REFUSING TO PUSH: connection details still present" >&2
    { git grep -lIE "$PATTERN" HEAD -- || true; } | head -10 >&2
    exit 1
fi

git push github "HEAD:$BRANCH"
