#!/usr/bin/env python
"""Hungarian assignment이 aux loss 축소 전후로 **동일**한지 실증한다.

왜 필요한가
----------
planning-isolated overfit에서 det/map loss를 1/20로 줄여야 하는데, mmdet은
`The classification weight for loss and matcher should be exactly the same`를
assert한다. 그래서 loss weight와 matching cost를 **함께** 줄였다.

코드상 근거는 있다.
    det: `cost = cls_cost + reg_cost`                       (iou_cost는 아예 안 쓴다)
    map: `cost = cls_cost + reg_cost + iou_cost + pts_cost`
둘 다 **이미 가중된** 비용의 단순 합이고, 내가 줄인 것은 비영 비용 전부
(det cls 2.0->0.1, reg 0.25->0.0125 / map cls 2.0->0.1, pts 1.0->0.05)로 배율이
모두 0.05다. `linear_sum_assignment`의 해는 비용행렬의 **양의 상수배**에 불변이므로
assignment는 같아야 한다.

하지만 "같아야 한다"와 "같다"는 다르다. 두 config로 같은 배치를 흘려
`assign()`이 낸 인덱스를 전수 비교한다.

    python scripts/etri_assign_invariance.py --batches 12
"""
import argparse
import importlib
import os
import sys

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
CFGD = os.path.join(REPO, "projects", "configs", "VAD")
ANN = "/tmp/pm97/data/etri/pkl/overfit8.pkl"


def run(cfg_path, batches, bs):
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
            # map assigner는 (AssignResult, order_index) 튜플을 낸다.
            res = r[0] if isinstance(r, tuple) else r
            extra = r[1] if isinstance(r, tuple) and len(r) > 1 else None
            rec.append((tag, res.gt_inds.detach().cpu().clone(),
                        None if getattr(res, "labels", None) is None
                        else res.labels.detach().cpu().clone(),
                        None if extra is None or not hasattr(extra, "detach")
                        else extra.detach().cpu().clone()))
            return r
        cls.assign = patched
        return orig

    o1 = wrap(HungarianAssigner3D, "det")
    o2 = wrap(MapHungarianAssigner3D, "map")
    try:
        cfg = Config.fromfile(cfg_path)
        cfg.data.train.ann_file = ANN
        ds = build_dataset(cfg.data.train)
        dl = build_dataloader(ds, samples_per_gpu=bs, workers_per_gpu=2,
                              num_gpus=1, dist=False, seed=0)
        torch.manual_seed(0)
        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
        model.init_weights()
        model = MMDataParallel(model.cuda(0), device_ids=[0])
        model.train()
        it = iter(dl)
        for _ in range(batches):
            data = next(it)
            with torch.no_grad():
                model(**data)
        del model, dl, ds
        torch.cuda.empty_cache()
    finally:
        HungarianAssigner3D.assign = o1
        MapHungarianAssigner3D.assign = o2
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="VAD_etri_overfit8.py")
    ap.add_argument("--scaled", default="VAD_etri_tvad_metric_overfit.py")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=2)
    args = ap.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    importlib.import_module("projects.mmdet3d_plugin")

    print(f"base   {args.base}   (원본 cost/loss weight)")
    print(f"scaled {args.scaled} (cost/loss 모두 x0.05)")
    print(f"배치 {args.batches} x bs {args.bs}\n")
    a = run(os.path.join(CFGD, args.base), args.batches, args.bs)
    b = run(os.path.join(CFGD, args.scaled), args.batches, args.bs)

    import torch
    if len(a) != len(b):
        print(f"FAIL  assign 호출 횟수 다름: {len(a)} vs {len(b)}")
        return 2
    bad, ndet, nmap = 0, 0, 0
    for (t1, g1, l1, e1), (t2, g2, l2, e2) in zip(a, b):
        same = (t1 == t2 and torch.equal(g1, g2)
                and ((l1 is None) == (l2 is None))
                and (l1 is None or torch.equal(l1, l2))
                and ((e1 is None) == (e2 is None))
                and (e1 is None or torch.equal(e1, e2)))
        if not same:
            bad += 1
            if bad <= 3:
                d = int((g1 != g2).sum()) if g1.shape == g2.shape else -1
                print(f"  차이 [{t1}] gt_inds 다른 원소 {d}")
        ndet += t1 == "det"
        nmap += t1 == "map"
    print(f"assign 호출 {len(a)}회 (det {ndet}, map {nmap})")
    print(f"불일치 {bad}회")
    print(f"\n판정: {'PASS — assignment 동일, 균등 배율 확인' if bad == 0 else 'FAIL — matching이 달라졌다. aux를 assignment 후에 스케일해야 한다'}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
