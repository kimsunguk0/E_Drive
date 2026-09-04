#!/usr/bin/env python
"""Download the ETRI challenge Drive folder without gdown.

WHY NOT gdown
-------------
gdown 5.2.2 fails on files that a plain HTTP GET fetches fine. Measured
2026-08-08 on the same id, seconds apart:

    gdown  -> "Failed to retrieve file url: Cannot retrieve the public link of
               the file. You may need to change the permission to 'Anyone with
               the link'"
    urllib -> HTTP 200, 64,849,920 bytes, valid tar

The message is misleading: the files ARE public. gdown's link-resolution path is
simply breaking here, and it burned hours looking like a permissions or quota
problem. This script talks to the endpoint directly.

TWO SIZE CLASSES, TWO OUTCOMES
------------------------------
Google inserts a virus-scan interstitial above roughly 100 MB, and the owner-side
download quota is enforced on THAT path:

    test/*  ~59 MiB  -> bytes come straight back           (works)
    train/* ~631 MiB -> interstitial, then "Quota exceeded" (blocked)

test/ works anonymously; train/ does not, and no amount of retrying changes that.

WHAT ACTUALLY UNBLOCKS train/ (2026-08-10)
------------------------------------------
The quota is on the ANONYMOUS confirm path, not on the files. Evidence: every
train id returned "Quota exceeded" on confirm, including ones already sitting on
disk -- so it is not per-file capacity. The same files download from a logged-in
browser. Two things were therefore needed:

  1. cookies from a signed-in session (see COOKIES below), and
  2. replaying the interstitial's form VERBATIM. Signed in, that form carries
     `authuser` and a per-request `at` token; a hand-built confirm URL omits them
     and bounces back to the interstitial.

With both, the confirm returns application/octet-stream, 631,449,600 bytes.

    python scripts/download_challenge_direct.py --only test
    python scripts/download_challenge_direct.py --only train --workers 3
"""
import argparse
import http.cookiejar
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

PERSIST = "/home/pm97/workspace/sukim/adcl"
FILELIST = os.path.join(PERSIST, "logs", "challenge_filelist.json")
DEST = os.path.join(PERSIST, "data", "challenge")
STATE = os.path.join(PERSIST, "logs", "challenge_direct_state.json")

# A signed-in Google session, exported as Netscape cookies.txt. Deliberately
# OUTSIDE the repo: this is the user's live Google credential, and $PERSIST is a
# git repo. Never log it, never copy it in.
COOKIES = os.environ.get("GDRIVE_COOKIES",
                         os.path.expanduser("~/.config/gdrive_cookies.txt"))

PRIORITY = {"baseline": 0, "baseline(docker)": 1, "train": 2, "test": 3}
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_lock = threading.Lock()
_plock = threading.Lock()


def say(*a):
    with _plock:
        print(*a, flush=True)


def group_of(p):
    parts = p.split("/")
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


def is_done(dest, state, key):
    if not os.path.isfile(dest) or os.path.getsize(dest) == 0:
        return False
    size = os.path.getsize(dest)
    rec = state.get(key)
    if isinstance(rec, dict) and rec.get("ok") and rec.get("size") == size:
        return True
    if not archive_ok(dest):
        return False
    with _lock:
        state[key] = {"ok": True, "size": size}
    return True


class Quota(Exception):
    pass


_jar = None
_opener = None
_mtime = None
_jar_lock = threading.Lock()


def build_opener():
    """The ONE opener, carrying a signed-in session if cookies were supplied.

    WHY COOKIES (measured 2026-08-10)
    ---------------------------------
    Anonymous requests get the interstitial with a valid uuid, then the confirm
    round-trip returns 2,009 bytes of "Quota exceeded" -- for EVERY train file,
    including ones already on disk. The wall is not per-file capacity; it is the
    anonymous download path being closed. Signed in, the same files come back.

    WHY ONE SHARED JAR, SAVED BACK TO DISK
    --------------------------------------
    Google rotates __Secure-1PSIDTS / SIDCC on the fly and expects the NEW value
    on the next request. The first version of this loaded a private jar per
    worker thread and never wrote rotations back, so all three threads kept
    replaying the same stale token; ~7 minutes and 10 files in, Google dropped
    the session. The tell was in the interstitial form, not the error text --
    signed in it carries `authuser` and `at`, and those fields had vanished
    while the confirm still said "Quota exceeded". So: one jar, shared, and
    persisted after every request. Cookies are per-domain, not per-file, so
    sharing across threads is correct as well as necessary.

    WHY THE JAR IS RELOADED, NOT CACHED FOREVER
    -------------------------------------------
    Re-reads COOKIES whenever its mtime changes, so a refreshed export is picked
    up WITHOUT restarting: a long-lived run must survive the session expiring and
    the user pasting a new jar. (Caching the opener forever was the first bug
    here -- the process happily reported "세션 만료" for two passes while a valid
    cookie file sat on disk unread.) `save_cookies` stamps _mtime after writing
    so our own rotation-saves do not trigger a reload.
    """
    global _jar, _opener, _mtime
    with _jar_lock:
        mt = os.path.getmtime(COOKIES) if os.path.isfile(COOKIES) else None
        if _opener is None or mt != _mtime:
            handlers = []
            if mt is not None:
                _jar = http.cookiejar.MozillaCookieJar(COOKIES)
                _jar.load(ignore_discard=True, ignore_expires=True)
                handlers.append(urllib.request.HTTPCookieProcessor(_jar))
            else:
                _jar = None
            _opener = urllib.request.build_opener(*handlers)
            _mtime = mt
        return _opener


def save_cookies():
    """Persist rotated cookies. Write-then-rename: a crash mid-save would
    otherwise leave a truncated jar and log us out for good."""
    global _mtime
    if _jar is None:
        return
    with _jar_lock:
        try:
            tmp = COOKIES + ".tmp"
            _jar.save(tmp, ignore_discard=True, ignore_expires=True)
            os.chmod(tmp, 0o600)
            os.replace(tmp, COOKIES)
            _mtime = os.path.getmtime(COOKIES)
        except Exception:                             # noqa: BLE001
            pass


def signed_in(opener):
    """True if the session still authenticates.

    Checked at the top of every pass because a dead session and an exhausted
    quota produce the SAME "Quota exceeded" body -- without this the script
    would retry a logged-out session for hours and report it as a quota wall.
    """
    if _jar is None:
        return False
    try:
        req = urllib.request.Request("https://drive.google.com/drive/my-drive",
                                     headers={"User-Agent": UA})
        with opener.open(req, timeout=60) as r:
            return "accounts.google.com" not in r.geturl()
    except Exception:                                 # noqa: BLE001
        return False


def fetch(fid, dest, timeout=300, opener=None):
    """Stream the file to dest+'.part'. Raises Quota if Google refuses."""
    op = opener or urllib.request.build_opener()
    url = f"https://drive.google.com/uc?export=download&id={fid}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    part = dest + ".part"
    r = op.open(req, timeout=timeout)
    head = r.read(65536)
    if head[:9] == b"<!DOCTYPE":
        body = head + r.read()
        r.close()
        title = re.search(rb"<title>([^<]*)</title>", body)
        t = title.group(1).decode() if title else "html"
        # Virus-scan interstitial: replay its form exactly. >100 MB always lands
        # here, so this is the only path train/ has.
        #
        # Do NOT hand-build the query. Signed in, the form carries two fields a
        # hand-built URL misses -- `authuser` and a per-request `at` token -- and
        # without them the confirm bounces straight back to the same interstitial
        # (measured 2026-08-10: hand-built -> "Virus scan warning" again;
        # form-replayed -> 200, Content-Length 631,449,600, ustar at offset 257).
        s = body.decode("utf-8", "ignore")
        action = re.search(r'<form[^>]*action="([^"]+)"', s)
        fields = dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"',
                                 s))
        if not action or "uuid" not in fields:
            raise Quota(t)
        curl = (action.group(1).replace("&amp;", "&") + "?"
                + urllib.parse.urlencode(fields))
        r = op.open(urllib.request.Request(curl, headers={"User-Agent": UA}),
                    timeout=timeout)
        head = r.read(65536)
        if head[:9] == b"<!DOCTYPE":
            body = head + r.read()
            r.close()
            title = re.search(rb"<title>([^<]*)</title>", body)
            raise Quota(title.group(1).decode() if title else "html")
    n = len(head)
    try:
        with open(part, "wb") as f:
            f.write(head)
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                n += len(buf)
    finally:
        r.close()
    save_cookies()
    if not archive_ok(part):
        os.remove(part)
        raise RuntimeError("corrupt")
    os.replace(part, dest)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--passes", type=int, default=500)
    ap.add_argument("--sleep", type=int, default=120)
    ap.add_argument("--gap", type=float, default=0.0)
    args = ap.parse_args()

    rows = json.load(open(FILELIST))
    rows.sort(key=lambda r: (PRIORITY.get(group_of(r["path"]), 9), r["path"]))
    if args.only:
        rows = [r for r in rows if group_of(r["path"]) == args.only]

    state = {}
    if os.path.isfile(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:                             # noqa: BLE001
            state = {}

    say(f"대상 {len(rows)} 파일 -> {DEST}  workers {args.workers}")
    say("쿠키: " + ("로그인 세션 사용" if os.path.isfile(COOKIES)
                    else "없음 (익명 — train/은 차단됨)"))

    for p in range(1, args.passes + 1):
        todo = [r for r in rows
                if not is_done(os.path.join(DEST, r["path"]), state, r["path"])]
        say(f"\n=== pass {p}/{args.passes}  완료 {len(rows)-len(todo)}/{len(rows)}"
            f"  남음 {len(todo)} ===")
        if not todo:
            say("전부 완료")
            break

        if os.path.isfile(COOKIES) and not signed_in(build_opener()):
            say(f"  세션 만료 — 쿠키를 다시 넣어주세요: {COOKIES}")
            say(f"  {args.sleep}s 후 재확인")
            time.sleep(args.sleep)
            continue

        cnt = {"ok": 0, "quota": 0, "err": 0, "bytes": 0}
        idx = [0]

        def worker():
            # One opener (and one cookie jar) per thread: the confirm hop depends
            # on cookies the interstitial just set, and sharing a jar across
            # threads would let one file's hop clobber another's.
            op = build_opener()
            while True:
                with _lock:
                    if idx[0] >= len(todo):
                        return
                    r = todo[idx[0]]
                    idx[0] += 1
                if args.gap:
                    time.sleep(args.gap)
                dest = os.path.join(DEST, r["path"])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                try:
                    n = fetch(r["id"], dest, opener=op)
                    with _lock:
                        cnt["ok"] += 1
                        cnt["bytes"] += n
                        state[r["path"]] = {"ok": True, "size": n}
                        json.dump(state, open(STATE, "w"), indent=1,
                                  ensure_ascii=False)
                    say(f"  OK     {r['path']}  {human(n)}")
                except Quota as e:
                    with _lock:
                        cnt["quota"] += 1
                    if cnt["quota"] <= 3:
                        say(f"  차단   {r['path']}  ({e})")
                except Exception as e:                # noqa: BLE001
                    with _lock:
                        cnt["err"] += 1
                    say(f"  실패   {r['path']}: {str(e)[:70]}")

        ts = [threading.Thread(target=worker, daemon=True)
              for _ in range(args.workers)]
        [t.start() for t in ts]
        [t.join() for t in ts]

        with _lock:
            json.dump(state, open(STATE, "w"), indent=1, ensure_ascii=False)
        say(f"  성공 {cnt['ok']} ({human(cnt['bytes'])}) / 차단 {cnt['quota']}"
            f" / 오류 {cnt['err']}")
        if cnt["ok"] == 0 and p < args.passes:
            say(f"  {args.sleep}s 후 재시도")
            time.sleep(args.sleep)

    done = [r for r in rows if os.path.isfile(os.path.join(DEST, r["path"]))]
    tot = sum(os.path.getsize(os.path.join(DEST, r["path"])) for r in done)
    say(f"\n=== 최종 {len(done)}/{len(rows)} 파일, {human(tot)} ===")
    return 0 if len(done) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
