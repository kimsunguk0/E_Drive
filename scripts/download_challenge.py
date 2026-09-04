#!/usr/bin/env python
"""Download the ETRI E2E Driving Challenge Drive folder, resumably.

WHY NOT `gdown --folder`
------------------------
The folder was published to all entrants at once, so Google throttles it:

    Too many users have viewed or downloaded this file recently.

`gdown --folder` aborts the whole run on the first such file. With 1,504 files that
means one throttled file costs the entire download. This script instead treats a
quota rejection as "try again later": it makes repeated passes, skipping what is
already on disk, until everything lands or the pass budget runs out.

ORDERING IS DELIBERATE
----------------------
baseline/ first (the guideline and the reference code -- small, and everything we
can plan without the bulk data depends on them), then the docker image, then
train/, then test/. If the quota bites we would rather be blocked on scene tars
than on the document that tells us the data format.

Resume rule: a file counts as done when it exists with a non-zero size AND, for
tar/tar.gz, passes an archive listing check. A half-written file from an
interrupted pass is deleted and retried rather than silently accepted -- the
nuScenes download taught us that a truncated archive looks fine to `ls`.

NOTE ON THE FILE LIST
---------------------
logs/challenge_filelist.json must come from the embeddedfolderview endpoint, NOT
from gdown. gdown's folder listing silently caps at 50 entries per folder, and
passing remaining_ok=True (needed to survive quota errors) suppresses the very
warning that would reveal the truncation. That combination made an earlier run
believe the challenge shipped 50 train / 50 test archives when it actually ships
376 / 1,125 -- matching the numbers printed in the guideline all along.

    python scripts/download_challenge.py                # all
    python scripts/download_challenge.py --only baseline
    python scripts/download_challenge.py --passes 20
"""
import argparse
import json
import os
import subprocess
import sys
import time

PERSIST = "/home/pm97/workspace/sukim/adcl"
FILELIST = os.path.join(PERSIST, "logs", "challenge_filelist.json")
DEST = os.path.join(PERSIST, "data", "challenge")
STATE = os.path.join(PERSIST, "logs", "challenge_download_state.json")

# Lower is earlier. Anything unlisted sorts last.
PRIORITY = {"baseline": 0, "baseline(docker)": 1, "train": 2, "test": 3}


def group_of(path):
    # gdown yields paths relative to the shared folder, i.e. "<group>/<file>"
    # with exactly one separator -- verified against logs/challenge_filelist.json.
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def archive_ok(path):
    """Cheap integrity check: can tar list it?  Catches truncation."""
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        cmd = ["tar", "-tzf", path]
    elif path.endswith(".tar"):
        cmd = ["tar", "-tf", path]
    else:
        return True          # unknown type: size check only
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=900)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def is_done(dest, state=None, key=None):
    """Has this file already landed intact?

    The archive check is the reliable part -- a truncated tar looks fine to
    os.stat -- but it reads the whole file, and with 1,504 archives (~250 GiB on
    NFS) re-verifying everything on every pass would cost more than the download.
    So a success is memoised in the state file together with the size it was
    verified at; a size change invalidates it and forces a re-check.
    """
    if not os.path.isfile(dest):
        return False
    size = os.path.getsize(dest)
    if size == 0:
        return False
    if state is not None and key is not None:
        rec = state.get(key)
        if isinstance(rec, dict) and rec.get('ok') and rec.get('size') == size:
            return True
    if not archive_ok(dest):
        return False
    if state is not None and key is not None:
        state[key] = {'ok': True, 'size': size}
    return True


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=12,
                    help="how many times to sweep the remaining files")
    ap.add_argument("--sleep", type=int, default=600,
                    help="seconds between passes (quota windows are ~hours)")
    ap.add_argument("--only", default=None,
                    help="restrict to one group: baseline, train, test, ...")
    ap.add_argument("--batch", type=int, default=40,
                    help="files attempted per pass (0 = the whole remaining list)")
    ap.add_argument("--gap", type=float, default=3.0,
                    help="seconds to wait between individual file attempts")
    ap.add_argument("--max-sleep", type=int, default=3600,
                    help="ceiling for the adaptive backoff between passes")
    args = ap.parse_args()

    import gdown

    rows = json.load(open(FILELIST))
    rows.sort(key=lambda r: (PRIORITY.get(group_of(r["path"]), 9), r["path"]))
    if args.only:
        rows = [r for r in rows if group_of(r["path"]) == args.only]
    print(f"대상 {len(rows)} 파일 -> {DEST}", flush=True)

    state = {}
    if os.path.isfile(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:      # noqa: BLE001
            state = {}

    cur_sleep = args.sleep
    for p in range(1, args.passes + 1):
        todo = []
        for r in rows:
            dest = os.path.join(DEST, r["path"])
            if is_done(dest, state, r["path"]):
                continue
            todo.append((r, dest))

        done_n = len(rows) - len(todo)
        remaining_total = len(todo)
        if not todo:
            print("전부 완료", flush=True)
            break

        # Attempt only a SLICE per pass, rotating through the list.
        #
        # WHY: sweeping all 1,464 remaining files every 10 minutes is ~8,800
        # requests/hour, and every one of them that hits the quota page is still a
        # request. The owner-side quota is not something we can influence, but an
        # IP-level throttle on top of it is -- and the timing fits: with a 103-file
        # list this script pulled 6.2 MiB/s, and it went to near-zero the moment the
        # list grew to 1,504. Slicing tests that hypothesis and costs nothing if the
        # hypothesis is wrong, since a blocked file would have failed either way.
        if args.batch and args.batch < len(todo):
            start = ((p - 1) * args.batch) % len(todo)
            todo = (todo + todo)[start:start + args.batch]
        print(f"\n=== pass {p}/{args.passes}  완료 {done_n}/{len(rows)}  "
              f"남음 {remaining_total}  이번 시도 {len(todo)} ===", flush=True)

        quota_hits = 0
        ok_hits = 0
        for idx, (r, dest) in enumerate(todo):
            if idx and args.gap:
                time.sleep(args.gap)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            try:
                # gdown does NOT reliably honour the ".part" name we pass -- observed
                # 2026-08-07 writing straight to the final filename instead. So take
                # the path gdown REPORTS rather than assuming it used `tmp`;
                # otherwise a perfectly good download is logged as an empty response.
                out = gdown.download(id=r["id"], output=tmp, quiet=True,
                                     resume=True)
                if out and os.path.isfile(out):
                    tmp = out
            except Exception as e:                       # noqa: BLE001
                msg = str(e)
                if "Too many users" in msg or "quota" in msg.lower():
                    quota_hits += 1
                    state[r["path"]] = "quota"
                    print(f"  quota  {r['path']}", flush=True)
                else:
                    state[r["path"]] = f"error: {msg[:120]}"
                    print(f"  ERROR  {r['path']}: {msg[:120]}", flush=True)
                continue

            if not out or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                quota_hits += 1
                state[r["path"]] = "empty"
                print(f"  빈응답 {r['path']}", flush=True)
                if os.path.isfile(tmp):
                    os.remove(tmp)
                continue

            if not archive_ok(tmp):
                # Truncated or an HTML error page renamed to .tar. Do not keep it.
                sz = os.path.getsize(tmp)
                os.remove(tmp)
                state[r["path"]] = "corrupt"
                print(f"  손상   {r['path']} ({human(sz)}) — 삭제 후 재시도 예정",
                      flush=True)
                continue

            if os.path.abspath(tmp) != os.path.abspath(dest):
                os.replace(tmp, dest)
            state[r["path"]] = {'ok': True, 'size': os.path.getsize(dest)}
            ok_hits += 1
            print(f"  OK     {r['path']}  {human(os.path.getsize(dest))}",
                  flush=True)
            json.dump(state, open(STATE, "w"), indent=1, ensure_ascii=False)

        json.dump(state, open(STATE, "w"), indent=1, ensure_ascii=False)
        remaining = sum(1 for r in rows
                        if not is_done(os.path.join(DEST, r["path"]), state, r["path"]))
        if remaining == 0:
            print("\n전부 완료", flush=True)
            break
        if p < args.passes:
            # Adaptive pacing: a pass that landed something means the quota window
            # is open, so press on quickly; a pass that landed nothing means backing
            # off is strictly better than retrying at the same rate.
            if ok_hits:
                sleep_s = args.sleep
                cur_sleep = args.sleep
            else:
                cur_sleep = min(int(cur_sleep * 2), args.max_sleep)
                sleep_s = cur_sleep
            print(f"  성공 {ok_hits} / quota·실패 {quota_hits}  — {sleep_s}s 후 재시도",
                  flush=True)
            time.sleep(sleep_s)

    # summary
    total = 0
    ok = 0
    for r in rows:
        dest = os.path.join(DEST, r["path"])
        if os.path.isfile(dest):
            total += os.path.getsize(dest)
            ok += 1
    print(f"\n=== 최종: {ok}/{len(rows)} 파일, {human(total)} ===", flush=True)
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
