#!/usr/bin/env python
"""Docker 이미지(etri-vad:cu128)를 rootfs로 재구성. docker/podman 없이.

WHY THIS REPLACES THE SHELL VERSION
-----------------------------------
첫 구현(etri_docker_unpack.sh)은 opaque whiteout을 이렇게 처리했다:

    find "$d" -mindepth 1 -maxdepth 1 ! -newer "$d" -exec rm -rf {} +

mtime 비교는 레이어 순서와 아무 관계가 없다. 결과적으로 레이어 21~23이 설치한
`site-packages/mmdet3d`(1,116 엔트리)가 rootfs에서 사라졌고, "이미지에 mmdet3d가
없다"는 잘못된 결론까지 갈 뻔했다. 원본 레이어를 tar -tf로 직접 뒤져서야 드러났다.

올바른 규칙 (OCI image-spec):
  * `.wh.<name>`      -> 같은 디렉터리의 <name>을 삭제. **하위 레이어에만** 적용된다.
  * `.wh..wh..opq`    -> 그 디렉터리의 기존 내용을 전부 삭제. 역시 하위 레이어 대상.
  * 두 마커 모두 rootfs에 남기지 않는다.

따라서 레이어마다 (1) 목록을 먼저 읽어 마커를 해석하고 삭제를 적용한 뒤,
(2) 마커를 제외한 실제 파일을 푼다. 순서를 뒤집으면 방금 푼 파일을 자기 마커로
지우게 된다.

    python scripts/etri_docker_unpack.py [--src DIR] [--out DIR]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

WH = ".wh."
OPQ = ".wh..wh..opq"


def layer_members(tar):
    r = subprocess.run(["tar", "-tf", tar], capture_output=True, text=True,
                       check=True)
    return r.stdout.splitlines()


def apply_whiteouts(members, out):
    """마커를 해석해 하위 레이어의 경로를 지운다. 지운 개수를 돌려준다."""
    removed = 0
    for m in members:
        base = os.path.basename(m.rstrip("/"))
        if base == OPQ:
            d = os.path.join(out, os.path.dirname(m.rstrip("/")))
            if os.path.isdir(d):
                for e in os.listdir(d):
                    p = os.path.join(d, e)
                    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) \
                        and not os.path.islink(p) else os.remove(p)
                    removed += 1
        elif base.startswith(WH):
            t = os.path.join(out, os.path.dirname(m.rstrip("/")),
                             base[len(WH):])
            if os.path.islink(t) or os.path.isfile(t):
                os.remove(t)
                removed += 1
            elif os.path.isdir(t):
                shutil.rmtree(t, ignore_errors=True)
                removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/pm97/docker_etri")
    ap.add_argument("--out", default="/tmp/pm97/docker_etri/rootfs2")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.src, "manifest.json")))
    layers = man[0]["Layers"]
    os.makedirs(args.out, exist_ok=True)
    print(f"레이어 {len(layers)}개 -> {args.out}", flush=True)

    total_wh = 0
    for i, l in enumerate(layers, 1):
        tar = os.path.join(args.src, l)
        members = layer_members(tar)
        marks = [m for m in members
                 if os.path.basename(m.rstrip("/")).startswith(WH)]
        n = apply_whiteouts(marks, args.out)
        total_wh += n

        cmd = ["tar", "-xf", tar, "-C", args.out,
               "--no-same-owner", "--no-same-permissions",
               "--exclude", ".wh.*"]
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
        print(f"  [{i:2d}/{len(layers)}] {l.split('/')[-1][:12]}  "
              f"엔트리 {len(members):6d}  마커 {len(marks):4d}  삭제 {n:4d}",
              flush=True)

    leftover = subprocess.run(["find", args.out, "-name", ".wh.*"],
                              capture_output=True, text=True).stdout.split()
    print(f"\n마커 잔여: {len(leftover)}   총 whiteout 삭제: {total_wh}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
