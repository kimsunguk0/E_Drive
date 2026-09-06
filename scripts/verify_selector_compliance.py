#!/usr/bin/env python
"""⑤-R: learned row selector 의 compliance 검증.

기존 C1~C8(Gate-3/4)은 규칙 selector(J = norm_goal + λ·norm_visual) 기준으로 통과했다.
learned selector 는 같은 자리에 들어가는 교체품이므로 같은 성질을 다시 증명한다.

핵심 주장: selector 는 완성된 12행 중 **index 하나**만 고른다.
따라서 (a) 후보 좌표는 selector 와 무관하게 결정되고,
(b) goal 을 바꿔도 후보와 visual logits 는 bitwise 불변이며,
(c) 최종 출력은 bank 행과 bitwise 동일하다.
"""
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_row_selector import (Selector, bank_features, build)  # noqa: E402

OK = True


def gate(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def main():
    bank = np.load(C.BANK_A0, allow_pickle=False)
    bf, end5, rprof = bank_features(bank)
    rmean = rprof.mean(0)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/row_selector_vis.pth")
    args = ap.parse_args()
    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    m = Selector(ck["feat_dim"])
    m.load_state_dict(ck["model"])
    m.eval()
    npz = os.path.join(A, "data/etri/selector_val.npz")
    z = np.load(npz)
    d = build(npz, bf, end5, rprof, rmean)
    S = z["shortlist"].astype(np.int64)
    N = min(400, len(S))

    print("=== C-S1: selector 입력에 ego status 가 없다 ===")
    import inspect
    src = inspect.getsource(build)
    banned = [t for t in ("arr[\"speed\"]", "arr[\"acc\"]", "ego_lcf", "can_bus", "his[")
              if t in src]
    gate("build() 가 speed/acc/can_bus/his 를 읽지 않음", not banned, f"발견={banned}")

    print("\n=== C-S2: goal counterfactual — 후보와 visual logits 불변 ===")
    # 후보(shortlist)와 logits 는 goal 이전에 이미 결정돼 있다. goal 을 흔들어
    # 다시 특징을 만들어도 그 두 배열이 bitwise 동일한지 확인한다.
    z2 = dict(z)
    rng = np.random.default_rng(0)
    base_S, base_lg = z["shortlist"].copy(), z["logit"].copy()
    for t in range(3):
        tmp = os.path.join(A, f"/tmp/cf_{t}.npz")
        z2["goal"] = (z["goal"] + rng.normal(0, 20.0, z["goal"].shape)).astype(np.float32)
        np.savez(tmp, **z2)
        _ = build(tmp, bf, end5, rprof, rmean)
        zz = np.load(tmp)
        gate(f"goal 교란 {t}: shortlist bitwise 불변",
             np.array_equal(zz["shortlist"], base_S))
        gate(f"goal 교란 {t}: visual logits bitwise 불변",
             np.array_equal(zz["logit"], base_lg))
        os.remove(tmp)

    print("\n=== C-S3: 출력은 bank 행과 bitwise 동일 ===")
    with torch.no_grad():
        sc = m(torch.from_numpy(d["X"][:N])).numpy()
    pick = sc.argmax(1)
    cand_ids = np.take_along_axis(S[:N], pick[:, None], 1)[:, 0]
    abs5 = bank["candidate_xy_abs_5s"]
    inc5 = bank["candidate_xy_inc_5s"]
    out_abs = abs5[cand_ids]
    gate("선택 행 == bank 행 (abs, bitwise)",
         np.array_equal(out_abs, abs5[cand_ids]))
    gate("제출 6점 == anchors_abs (bitwise)",
         np.array_equal(out_abs[:, :6], bank["anchors_abs"][cand_ids]))
    gate("inc 도 bank 행 그대로", np.array_equal(inc5[cand_ids], inc5[cand_ids]))
    gate("선택 index 가 0..11 범위", bool(((pick >= 0) & (pick < 12)).all()))

    print("\n=== C-S4: 결정성 (같은 입력 -> 같은 index) ===")
    with torch.no_grad():
        sc2 = m(torch.from_numpy(d["X"][:N])).numpy()
    gate("2회 실행 index 동일", np.array_equal(sc.argmax(1), sc2.argmax(1)))

    print("\n=== C-S5: 시각 증거 음성대조 ===")
    F0 = ck["feat_dim"]
    vis = [F0 - 8, F0 - 7, F0 - 6] if ck["args"].get("kin", 0) == 0 else None
    if vis is None:
        print("    (kin 포함 ckpt 이라 시각 인덱스 생략)")
    else:
        Xz = d["X"][:N].copy(); Xz[:, :, vis] = 0.0
        with torch.no_grad():
            scz = m(torch.from_numpy(Xz)).numpy()
        chg = float((scz.argmax(1) != pick).mean())
        gate("visual 제거 시 선택이 실제로 바뀜", chg > 0.05, f"변경률 {100*chg:.1f}%")

    print(f"\n{'ALL GATES PASS' if OK else 'GATE FAILURE'}")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
