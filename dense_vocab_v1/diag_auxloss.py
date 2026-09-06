"""det/map 보조 loss 가 0.0000 으로 찍히는 원인 진단 — 한 배치 직접 실행."""
import importlib
import sys

import torch
from mmcv import Config
from mmcv.parallel import collate, scatter

CFG = "projects/configs/VAD/VAD_etri_s2_open.py"
cfg = Config.fromfile(CFG)
importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

ds = build_dataset(cfg.data.train)
model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                    test_cfg=cfg.get("test_cfg"))
head = model.pts_bbox_head
print("=== head 설정 ===")
for k in ("num_query", "num_classes", "map_num_vec", "map_num_classes",
          "num_reg_fcs", "with_box_refine", "as_two_stage", "fut_ts",
          "ego_fut_mode"):
    print("  %-18s %s" % (k, getattr(head, k, "N/A")))
print("  loss_cls weight     ", getattr(head.loss_cls, "loss_weight", None))
print("  loss_bbox weight    ", getattr(head.loss_bbox, "loss_weight", None))
print("  loss_map_cls weight ", getattr(head.loss_map_cls, "loss_weight", None))
print("  loss_map_pts weight ", getattr(head.loss_map_pts, "loss_weight", None))

# 여러 샘플을 모아 GT 가 있는 배치를 만든다
idxs = [0, 500, 1000, 2000]
samples = [ds[i] for i in idxs]
nb = [len(s["gt_labels_3d"].data) for s in samples]
nm = [len(s["map_gt_labels_3d"].data) for s in samples]
print("\n배치 GT: 객체 %s / map vec %s" % (nb, nm))

batch = collate(samples, samples_per_gpu=len(samples))
model = model.cuda().train()
batch = scatter(batch, [0])[0]

with torch.autograd.set_detect_anomaly(False):
    losses = model(return_loss=True, **batch)

print("\n=== loss_dict (고정밀) ===")
tot = 0.0
for k, v in losses.items():
    val = float(v.detach()) if torch.is_tensor(v) else float(v)
    flag = ""
    if k.startswith("loss") and abs(val) < 1e-8:
        flag = "  <== 정확히 0"
    print("  %-26s %.6e%s" % (k, val, flag))
    if k.startswith("loss"):
        tot += val
print("  %-26s %.6f" % ("(loss* 합계)", tot))
