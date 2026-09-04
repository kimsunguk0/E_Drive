#!/usr/bin/env python
"""`PlanChallengeL2Loss`가 `src/challenge_metrics.py`와 같은 값을 내는지 검증한다.

이 테스트를 통과해야 새 loss를 source of truth로 삼을 수 있다. 목표는
    |loss - challenge_metric| < 1e-6
이고, mode 희석 제거와 실제 GT 배치까지 함께 확인한다.

    python scripts/test_plan_metric_loss.py
"""
import importlib
import os
import pickle
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
ADCL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANN = "/tmp/pm97/data/etri/pkl/overfit8.pkl"

OK = []


def check(name, got, want, tol=1e-6):
    d = abs(got - want)
    OK.append(d < tol)
    print(f"  {'PASS' if d < tol else 'FAIL'}  {name:46s} "
          f"{got:.9f} vs {want:.9f}  차 {d:.2e}")


def main():
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(ADCL, "src"))
    importlib.import_module("projects.mmdet3d_plugin")
    import torch
    from mmdet.models.builder import build_loss
    from challenge_metrics import l2_challenge, waypoint_weights

    loss = build_loss(dict(type='PlanChallengeL2Loss', loss_weight=1.0))
    B, M, T = 7, 3, 6
    rng = np.random.default_rng(0)

    # ---- 1. 무작위 텐서: 챌린지 metric과 동일해야 한다 ----
    print("1) 무작위 텐서 등가성")
    pd = rng.normal(0, 3, (B, M, T, 2))
    gd = rng.normal(0, 3, (B, T, 2))
    ci = rng.integers(0, M, B)
    cmd = np.zeros((B, M)); cmd[np.arange(B), ci] = 1
    got = float(loss(torch.tensor(pd), torch.tensor(gd),
                     torch.tensor(cmd, dtype=torch.float64)))
    sel = pd[np.arange(B), ci]
    want = float(l2_challenge(np.cumsum(sel, 1), np.cumsum(gd, 1))["L2_avg"])
    check("무작위 vs l2_challenge", got, want, tol=1e-5)   # eps=1e-6 때문에 여유

    # ---- 2. mode 희석 없음: 1-mode 와 3-mode 결과가 같아야 한다 ----
    print("\n2) mode 희석 제거")
    got1 = float(loss(torch.tensor(sel)[:, None], torch.tensor(gd),
                      torch.ones(B, 1, dtype=torch.float64)))
    check("1-mode == 3-mode(선택)", got1, got)

    # ---- 3. 완전 일치 시 0 이고 NaN 없음 ----
    print("\n3) 완전 일치 / NaN 안정성")
    z = float(loss(torch.tensor(gd)[:, None].repeat(1, M, 1, 1), torch.tensor(gd),
                   torch.tensor(cmd, dtype=torch.float64)))
    print(f"  {'PASS' if z < 1e-2 and np.isfinite(z) else 'FAIL'}  "
          f"{'완전 일치 시 loss':46s} {z:.9f}  (eps 때문에 정확히 0은 아니다)")
    OK.append(z < 1e-2 and np.isfinite(z))
    p = torch.tensor(gd)[:, None].repeat(1, M, 1, 1).requires_grad_(True)
    float(loss(p, torch.tensor(gd), torch.tensor(cmd, dtype=torch.float64))).__class__
    out = loss(p, torch.tensor(gd), torch.tensor(cmd, dtype=torch.float64))
    out.backward()
    fin = bool(torch.isfinite(p.grad).all())
    print(f"  {'PASS' if fin else 'FAIL'}  {'완전 일치에서 gradient 유한':46s} "
          f"{'finite' if fin else 'NaN/Inf'}")
    OK.append(fin)

    # ---- 4. 실제 GT 배치 (증분 규약 확인 포함) ----
    print("\n4) 실제 pkl 배치")
    infos = pickle.load(open(ANN, "rb"))["infos"][:32]
    gdt = np.stack([i["gt_ego_fut_trajs"].reshape(T, 2) for i in infos]).astype(np.float64)
    cmd2 = np.stack([i["gt_ego_fut_cmd"].reshape(M) for i in infos]).astype(np.float64)
    ci2 = cmd2.argmax(1)
    pdt = gdt[:, None].repeat(M, 1) if False else np.repeat(gdt[:, None], M, axis=1)
    pdt = pdt + rng.normal(0, 0.3, pdt.shape)
    got2 = float(loss(torch.tensor(pdt), torch.tensor(gdt), torch.tensor(cmd2)))
    sel2 = pdt[np.arange(len(infos)), ci2]
    want2 = float(l2_challenge(np.cumsum(sel2, 1), np.cumsum(gdt, 1))["L2_avg"])
    check("실제 GT vs l2_challenge", got2, want2, tol=1e-5)

    # ---- 5. 가중치가 정말 [11,11,5,5,2,2]/36 인가 ----
    print("\n5) waypoint 가중")
    w = loss.w.numpy()
    check("가중치 합", float(w.sum()), 1.0)
    check("가중치 == waypoint_weights()", float(np.abs(w - waypoint_weights()).max()),
          0.0)

    # ---- 6. 증분 규약: pkl은 증분, ego_cache는 누적 ----
    print("\n6) 증분/누적 규약")
    cache = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=True)
    sc = np.array([str(x) for x in cache["scenarios"]])[cache["scen_idx"]]
    fr = cache["frame"].astype(int)
    k = {(s, int(f)): j for j, (s, f) in enumerate(zip(sc, fr))}
    j = [k[(i["scene_token"], int(i["frame_idx"]))] for i in infos]
    check("cumsum(pkl) == ego_cache['fut']",
          float(np.abs(np.cumsum(gdt, 1) - cache["fut"][j]).max()), 0.0, tol=1e-4)

    print(f"\n{'='*72}\n{sum(OK)}/{len(OK)} 통과  "
          f"-> {'전체 PASS' if all(OK) else 'FAIL 있음'}")
    return 0 if all(OK) else 2


if __name__ == "__main__":
    sys.exit(main())
