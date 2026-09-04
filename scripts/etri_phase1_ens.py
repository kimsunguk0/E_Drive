#!/usr/bin/env python
"""부록6 항목2 — K-앙상블 재검. 시드 독립 학습 후 조립.

첫 K 곡선은 K개 백본을 **하나의 optimizer로 동시 학습**하고 공통 조기중단을 걸었다.
그러면 각 시드가 개별 최적점에 닿기 전에 같은 시점에 묶여 앙상블 이득이 사라진다.
이번엔 시드별로 완전히 독립 학습(개별 조기중단·개별 best)한 뒤 단일 Module로 조립한다.

EMA 1셀도 함께: 학습 중 가중치 지수이동평균을 유지해 best 대신 EMA를 평가.

    python scripts/etri_phase1_ens.py
"""
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from etri_priornet import (CKPT, PriorNet, cfg_of, run)      # noqa: E402
from etri_table import COLUMNS, ValSet, fmt                  # noqa: E402

OUT = "/home/pm97/workspace/sukim/adcl/logs/phase1_ens.json"
WIN = dict(arch="transformer", param="B", mirror=1, goal_yaw=1,
           drop_glitch=1, huber=0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=True)
    sp = np.load("/tmp/pm97/data/etri/val_clips.npz", allow_pickle=True)
    v = ValSet()
    res, preds, states = {}, [], []

    for s in range(4):
        c = cfg_of(**WIN, K=1, seed=s, epochs=60)
        net, p, pm, gtm, nrm = run(c, v, d, sp, dev)
        preds.append(p)
        states.append(copy.deepcopy(net.state_dict()))
        r = v.row(p)
        res[f"seed{s} 단독"] = r
        print(f"seed{s} 단독   " + " ".join(f"{c2}={fmt(r[c2])}" for c2 in COLUMNS),
              flush=True)

    # 출력 평균 = 조립 앙상블과 수학적으로 동일 (변형 B의 prior는 공통이므로
    # 잔차 평균 + prior = 평균 출력).
    for k in (2, 3, 4):
        avg = np.mean(preds[:k], axis=0)
        r = v.row(avg)
        res[f"독립 {k}시드 평균"] = r
        print(f"독립 {k}시드 평균  " + " ".join(f"{c2}={fmt(r[c2])}" for c2 in COLUMNS),
              flush=True)

    # 단일 Module로 조립해 저장 (규칙: 개별 ckpt 금지)
    s0, c0 = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=True), None
    from etri_priornet import build_feats
    sq, cx = build_feats(d, v.vi, 1, 0)
    big = PriorNet(sq.shape[-1], cx.shape[-1], "transformer", "B", K=4)
    for i, st in enumerate(states):
        big.nets[i].load_state_dict({k.split(".", 2)[2]: x
                                     for k, x in st.items()
                                     if k.startswith("nets.0.")})
        big.heads[i].load_state_dict({k.split(".", 2)[2]: x
                                      for k, x in st.items()
                                      if k.startswith("heads.0.")})
    os.makedirs(CKPT, exist_ok=True)
    torch.save({"state": big.state_dict(),
                "cfg": cfg_of(**WIN, K=4, seed=0, epochs=60),
                "norm": {k: np.asarray(x) for k, x in
                         zip(("mu_s", "sd_s", "mu_c", "sd_c"), nrm)}},
               f"{CKPT}/ens_independent_K4.pt")
    print(f"\n조립 ckpt {CKPT}/ens_independent_K4.pt "
          f"({os.path.getsize(f'{CKPT}/ens_independent_K4.pt')/2**20:.1f} MB, 단일 파일)")

    base = res["seed0 단독"]["가중"]
    best_ens = min(res[f"독립 {k}시드 평균"]["가중"] for k in (2, 3, 4))
    print(f"\n단일 시드 최선 {min(res[f'seed{s} 단독']['가중'] for s in range(4)):.4f}"
          f"  독립앙상블 최선 {best_ens:.4f}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"저장 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
