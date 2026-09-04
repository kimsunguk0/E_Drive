#!/usr/bin/env python
"""§8 검증 — matcher 원본 + aux_loss_scale 방식이 이전 방식과 등가인가.

두 설정
    OLD  config에서 det/map loss weight와 matcher cost를 함께 x0.05
    NEW  matcher cost/loss weight는 원본, `aux_loss_scale=0.05`로 반환값만 스케일

확인할 것
    1. assignment index가 동일한가        (같은 batch, 같은 seed)
    2. loss dict의 aux 항이 수치적으로 같은가
    3. planning 항이 정확히 같은가        (건드리지 않았으므로 동일해야 한다)
    4. NEW의 matcher cost가 upstream VAD 원본과 같은가  (부채 청산의 핵심)

    python scripts/etri_auxscale_equiv.py
"""
import argparse
import importlib
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
CFGD = os.path.join(REPO, "projects", "configs", "VAD")


def run(cfg_path, batches, bs, seed=0):
    """(loss dict 값, assignment 기록)을 낸다."""
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet.datasets import build_dataloader
    from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d \
        import HungarianAssigner3D
    from projects.mmdet3d_plugin.core.bbox.assigners.map_hungarian_assigner_3d \
        import MapHungarianAssigner3D

    rec = []

    def wrap(cls, tag):
        orig = cls.assign

        def patched(self, *a, **k):
            r = orig(self, *a, **k)
            res = r[0] if isinstance(r, tuple) else r
            rec.append((tag, res.gt_inds.detach().cpu().clone()))
            return r
        cls.assign = patched
        return orig

    o1 = wrap(HungarianAssigner3D, "det")
    o2 = wrap(MapHungarianAssigner3D, "map")
    try:
        cfg = Config.fromfile(cfg_path)
        ds = build_dataset(cfg.data.train)
        dl = build_dataloader(ds, samples_per_gpu=bs, workers_per_gpu=2,
                              num_gpus=1, dist=False, seed=seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
        model.init_weights()
        mm = MMDataParallel(model.cuda(0), device_ids=[0])
        mm.train()
        # train_cfg는 model 안에 있다 (mmdet3d 규약).
        tc = cfg.model.train_cfg.pts if 'train_cfg' in cfg.model else cfg.train_cfg.pts
        costs = dict(
            det_cls=tc.assigner.cls_cost.weight,
            det_reg=tc.assigner.reg_cost.weight,
            map_cls=tc.map_assigner.cls_cost.weight,
            map_pts=tc.map_assigner.pts_cost.weight,
        )
        out = []
        it = iter(dl)
        for _ in range(batches):
            data = next(it)
            with torch.no_grad():
                losses = mm(**data)
            out.append({k: float(v.mean() if v.dim() else v)
                        for k, v in losses.items() if v is not None})
        del mm, model, dl, ds
        torch.cuda.empty_cache()
    finally:
        HungarianAssigner3D.assign = o1
        MapHungarianAssigner3D.assign = o2
    return out, rec, costs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="VAD_etri_c0t_overfit.py")
    ap.add_argument("--new", default="VAD_etri_c0t_overfit_v2.py")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--bs", type=int, default=2)
    args = ap.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    importlib.import_module("projects.mmdet3d_plugin")
    import torch

    print(f"OLD {args.old}  (loss weight & matcher cost 둘 다 x0.05)")
    print(f"NEW {args.new}  (원본 cost/weight + aux_loss_scale)")
    A, ra, ca = run(os.path.join(CFGD, args.old), args.batches, args.bs)
    B, rb, cb = run(os.path.join(CFGD, args.new), args.batches, args.bs)

    print(f"\nmatcher cost   OLD {ca}")
    print(f"               NEW {cb}")
    t4 = (cb["det_cls"] == 2.0 and cb["det_reg"] == 0.25
          and cb["map_cls"] == 2.0 and cb["map_pts"] == 1.0)
    print(f"  4) NEW의 matcher cost가 upstream 원본과 동일: "
          f"{'PASS' if t4 else 'FAIL'}")

    print(f"\nassign 호출 OLD {len(ra)}  NEW {len(rb)}")
    t1 = len(ra) == len(rb) and all(
        t1_ == t2_ and torch.equal(g1, g2) for (t1_, g1), (t2_, g2) in zip(ra, rb))
    print(f"  1) assignment index 동일: {'PASS' if t1 else 'FAIL'}")

    keys = sorted(set(A[0]) & set(B[0]))
    print(f"\n{'loss key':<26}{'OLD':>12}{'NEW':>12}{'상대차':>11}")
    worst_aux = worst_plan = 0.0
    for k in keys:
        va = np.mean([d[k] for d in A])
        vb = np.mean([d[k] for d in B])
        rel = abs(va - vb) / max(abs(va), 1e-9)
        if 'plan' in k:
            worst_plan = max(worst_plan, rel)
        else:
            worst_aux = max(worst_aux, rel)
        if abs(va) > 1e-9 or abs(vb) > 1e-9:
            print(f"{k:<26}{va:>12.6f}{vb:>12.6f}{rel:>11.2e}")
    t2 = worst_aux < 1e-4
    t3 = worst_plan < 1e-4
    print(f"\n  2) aux 항 수치 등가 (최대 상대차 {worst_aux:.2e}): "
          f"{'PASS' if t2 else 'FAIL'}")
    print(f"  3) planning 항 동일 (최대 상대차 {worst_plan:.2e}): "
          f"{'PASS' if t3 else 'FAIL'}")

    ok = t1 and t2 and t3 and t4
    print(f"\n=== 판정: {'PASS — 부채 청산. NEW를 쓰면 matching이 upstream과 완전 동일하다.' if ok else 'FAIL'} ===")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
