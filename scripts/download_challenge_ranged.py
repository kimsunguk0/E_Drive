#!/usr/bin/env python
"""Download the ETRI challenge Drive folder using RANGED requests.

WHY THIS EXISTS
---------------
gdown (and any plain GET) hits Google's owner-side quota wall on this folder:

    <title>Google Drive - Quota exceeded</title>       # 2,009 bytes of HTML

Measured 2026-08-08 on the SAME file id with the SAME confirm token, seconds apart:

    plain GET                -> 2,009 B  "Quota exceeded" HTML
    GET with Range: bytes=0- -> real tar (ustar magic at offset 257)

So the quota is enforced on the non-ranged download path only. Requesting explicit
byte ranges gets the actual bytes. This mirrors what download_large_ranged.sh had
to do for nuScenes' 53.6 GiB test_blobs.tgz, where CloudFront rejected the
non-ranged GET outright -- same tool, different wall.

FLOW PER FILE
  1. GET  drive.google.com/uc?export=download&id=<ID>   -> virus-scan interstitial
     carrying a per-request uuid (large files never return bytes here)
  2. GET  drive.usercontent.google.com/download?...&confirm=t&uuid=<UUID>
     with `Range: bytes=<start>-<end>` for each chunk, appending to a .part file
  3. verify with `tar -tf`, then rename into place

Resumable at chunk granularity: an interrupted file restarts from its .part size,
so a killed run costs at most one chunk. Verified files are memoised in the state
file (size-keyed) so re-verification does not re-read 250 GiB from NFS every pass.

    python scripts/download_challenge_ranged.py
    python scripts/download_challenge_ranged.py --only train --workers 4
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PERSIST = "/home/pm97/workspace/sukim/adcl"
FILELIST = os.path.join(PERSIST, "logs", "challenge_filelist.json")
DEST = os.path.join(PERSIST, "data", "challenge")
STATE = os.path.join(PERSIST, "logs", "challenge_ranged_state.json")

PRIORITY = {"baseline": 0, "baseline(docker)": 1, "train": 2, "test": 3}
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_state_lock = threading.Lock()
_print_lock = threading.Lock()


def say(*a):
    with _print_lock:
        print(*a, flush=True)


def group_of(path):
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def human(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PiB"


def archive_ok(path):
    cmd = (["tar", "-tzf", path] if path.endswith((".tar.gz", ".tgz"))
           else ["tar", "-tf", path] if path.endswith(".tar") else None)
    if cmd is None:
        return True
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=1800).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def get_confirm_uuid(fid, timeout=60):
    """Fetch the interstitial and pull out the per-request uuid token."""
    url = f"https://drive.google.com/uc?export=download&id={fid}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", "ignore")
    m = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    if m:
        return m.group(1)
    # Small files can come straight back as bytes; signal that to the caller.
    if not html.lstrip().startswith("<!DOCTYPE"):
        return None
    if "Quota exceeded" in html:
        raise RuntimeError("quota-on-interstitial")
    raise RuntimeError("no-uuid")


def fetch_range(fid, uuid, start, end, timeout=180):
    url = ("https://drive.usercontent.google.com/download"
           f"?id={fid}&export=download&confirm=t&uuid={uuid}")
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Range")


def total_size_of(fid, uuid):
    """Content-Range on a 1-byte probe gives the full length."""
    _, cr = fetch_range(fid, uuid, 0, 0)
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    return None


def download_one(row, chunk, retries, state):
    path, fid = row["path"], row["id"]
    dest = os.path.join(DEST, path)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    uuid = get_confirm_uuid(fid)
    if uuid is None:
        raise RuntimeError("small-file-direct")     # not expected for these tars
    total = total_size_of(fid, uuid)
    if not total:
        raise RuntimeError("no-content-range")

    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have > total:
        os.remove(part)
        have = 0

    with open(part, "ab") as f:
        while have < total:
            end = min(have + chunk - 1, total - 1)
            for attempt in range(retries):
                try:
                    buf, _ = fetch_range(fid, uuid, have, end)
                    break
                except Exception:                    # noqa: BLE001
                    if attempt == retries - 1:
                        raise
                    time.sleep(2 * (attempt + 1))
                    # The uuid is per-request-ish; refresh it on retry.
                    try:
                        uuid = get_confirm_uuid(fid)
                    except Exception:                # noqa: BLE001
                        pass
            if not buf:
                raise RuntimeError(f"empty-chunk at {have}")
            # A quota HTML page is ~2 KB and would silently corrupt the archive.
            if have == 0 and buf[:9] == b"<!DOCTYPE":
                raise RuntimeError("quota-html")
            f.write(buf)
            have += len(buf)

    if not archive_ok(part):
        os.remove(part)
        raise RuntimeError("corrupt")
    os.replace(part, dest)
    with _state_lock:
        state[path] = {"ok": True, "size": total}
    return total


def is_done(dest, state, key):
    if not os.path.isfile(dest):
        return False
    size = os.path.getsize(dest)
    if size == 0:
        return False
    rec = state.get(key)
    if isinstance(rec, dict) and rec.get("ok") and rec.get("size") == size:
        return True
    if not archive_ok(dest):
        return False
    with _state_lock:
        state[key] = {"ok": True, "size": size}
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--chunk", type=int, default=32 * 1024 * 1024,
                    help="bytes per ranged request")
    ap.add_argument("--workers", type=int, default=3,
                    help="files in flight; keep small, this is one shared quota")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--passes", type=int, default=200)
    ap.add_argument("--sleep", type=int, default=120,
                    help="seconds between passes when nothing succeeded")
    args = ap.parse_args()

    rows = json.load(open(FILELIST))
    rows.sort(key=lambda r: (PRIORITY.get(group_of(r["path"]), 9), r["path"]))
    if args.only:
        rows = [r for r in rows if group_of(r["path"]) == args.only]

    state = {}
    if os.path.isfile(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:                            # noqa: BLE001
            state = {}

    say(f"대상 {len(rows)} 파일 -> {DEST}   chunk {human(args.chunk)}  "
        f"workers {args.workers}")

    for p in range(1, args.passes + 1):
        todo = [r for r in rows
                if not is_done(os.path.join(DEST, r["path"]), state, r["path"])]
        say(f"\n=== pass {p}/{args.passes}  완료 {len(rows)-len(todo)}/{len(rows)}  "
            f"남음 {len(todo)} ===")
        if not todo:
            say("전부 완료")
            break

        ok = [0]
        fail = [0]
        lock = threading.Lock()
        idx = [0]

        def worker():
            while True:
                with lock:
                    if idx[0] >= len(todo):
                        return
                    r = todo[idx[0]]
                    idx[0] += 1
                try:
                    n = download_one(r, args.chunk, args.retries, state)
                    with lock:
                        ok[0] += 1
                    say(f"  OK     {r['path']}  {human(n)}")
                    with _state_lock:
                        json.dump(state, open(STATE, "w"), indent=1,
                                  ensure_ascii=False)
                except Exception as e:               # noqa: BLE001
                    with lock:
                        fail[0] += 1
                    say(f"  실패   {r['path']}: {str(e)[:80]}")

        ts = [threading.Thread(target=worker, daemon=True)
              for _ in range(args.workers)]
        [t.start() for t in ts]
        [t.join() for t in ts]

        with _state_lock:
            json.dump(state, open(STATE, "w"), indent=1, ensure_ascii=False)
        say(f"  성공 {ok[0]} / 실패 {fail[0]}")
        if ok[0] == 0 and p < args.passes:
            say(f"  {args.sleep}s 후 재시도")
            time.sleep(args.sleep)

    done = [r for r in rows
            if os.path.isfile(os.path.join(DEST, r["path"]))]
    tot = sum(os.path.getsize(os.path.join(DEST, r["path"])) for r in done)
    say(f"\n=== 최종 {len(done)}/{len(rows)} 파일, {human(tot)} ===")
    return 0 if len(done) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
