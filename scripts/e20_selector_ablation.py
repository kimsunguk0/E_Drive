#!/usr/bin/env python
"""⑤-S: learned row selector 가 실제로 쓰는 신호는 무엇인가 (특징군 절제).

각 군을 0 으로 만들고 realized 악화폭을 잰다. 다음 투자처를 정하는 근거.
"""
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_row_selector import Selector, bank_features, build, realized  # noqa: E402

# build() 의 concat 순서와 일치해야 한다
GROUPS = {
    "후보 기하(S3/S5/위치/heading/곡률)": list(range(0, 11)),
    "  └ timing profile r1..r5": list(range(11, 16)),
    "goal 거리(raw/정규화/순위)": [16, 17, 18],
    "goal 성분(dx,dy)": [19, 20],
    "visual logits": [21, 22, 23],
    "다양성(top-1 과의 거리)": [24, 25],
    "profile 편차(고정 prior)": [26, 27],
    "command": [28, 29, 30],
}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "work_dirs/row_selector_vis.pth"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "selector_r3"
    bank = np.load(C.BANK_A0, allow_pickle=False)
    bf, end5, rprof = bank_features(bank)
    ck = torch.load(os.path.join(A, ckpt), map_location="cpu")
    m = Selector(ck["feat_dim"]); m.load_state_dict(ck["model"]); m.eval()
    d = build(os.path.join(A, f"data/etri/{prefix}_val.npz"), bf, end5, rprof, rprof.mean(0))
    X, D3, w = d["X"], d["D3"], d["w"]
    assert X.shape[-1] == ck["feat_dim"], (X.shape[-1], ck["feat_dim"])

    with torch.no_grad():
        base = realized(m(torch.from_numpy(X)).numpy(), D3, w)
    orc = float(np.average(D3.min(1), weights=w))
    print(f"ckpt={ckpt}  prefix={prefix}  n={len(X)}  feature={X.shape[-1]}")
    print(f"  shortlist oracle {orc:.4f}   정상 {base:.4f}   regret {base-orc:.4f}\n")
    print(f"{'절제 대상':>34} {'realized':>9} {'악화':>9}")
    rows = []
    for name, idx in GROUPS.items():
        Xa = X.copy(); Xa[:, :, idx] = 0.0
        with torch.no_grad():
            r = realized(m(torch.from_numpy(Xa)).numpy(), D3, w)
        rows.append((name, r, r - base))
        print(f"{name:>34} {r:>9.4f} {r-base:>+9.4f}")
    print()
    rows.sort(key=lambda x: -x[2])
    print("중요도 순:")
    for i, (n, r, dlt) in enumerate(rows, 1):
        print(f"  {i}. {n.strip()}  {dlt:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
