"""det/map loss==0 원인: loss_single / map_loss_single 을 감싸 입출력을 찍는다."""
import importlib

import torch
from mmcv import Config
from mmcv.parallel import collate, scatter

cfg = Config.fromfile("projects/configs/VAD/VAD_etri_s2_open.py")
importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

ds = build_dataset(cfg.data.train)
m = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                test_cfg=cfg.get("test_cfg")).cuda().train()
head = m.pts_bbox_head

orig_ls = head.loss_single
orig_ms = head.map_loss_single
state = {"ls": 0, "ms": 0}


def wrap_ls(cls_scores, bbox_preds, traj_preds, traj_cls_scores,
            gt_bboxes_list, gt_labels_list, gt_attr_labels, gt_bboxes_ignore_list=None):
    if state["ls"] == 0:
        print("[loss_single] cls_scores %s absmax=%.3e | bbox_preds %s absmax=%.3e"
              % (tuple(cls_scores.shape), float(cls_scores.abs().max()),
                 tuple(bbox_preds.shape), float(bbox_preds.abs().max())))
        print("              gt_bboxes n=%s | gt_labels n=%s"
              % ([len(x) for x in gt_bboxes_list], [len(x) for x in gt_labels_list]))
    out = orig_ls(cls_scores, bbox_preds, traj_preds, traj_cls_scores,
                  gt_bboxes_list, gt_labels_list, gt_attr_labels, gt_bboxes_ignore_list)
    if state["ls"] == 0:
        print("              -> loss_cls=%.6e loss_bbox=%.6e loss_traj=%.6e loss_traj_cls=%.6e"
              % tuple(float(x) for x in out))
    state["ls"] += 1
    return out


def wrap_ms(*a, **k):
    out = orig_ms(*a, **k)
    if state["ms"] == 0:
        print("[map_loss_single] cls %s absmax=%.3e -> map_cls=%.6e map_pts=%.6e"
              % (tuple(a[0].shape), float(a[0].abs().max()),
                 float(out[0]), float(out[3])))
    state["ms"] += 1
    return out


head.loss_single = wrap_ls
head.map_loss_single = wrap_ms

b = scatter(collate([ds[i] for i in [0, 500]], samples_per_gpu=2), [0])[0]
print("입력 img:", tuple(b["img"].shape))
losses = m(return_loss=True, **b)
print("\nloss_cls=%.6e loss_map_cls=%.6e loss_plan_vocab=%.6e"
      % (float(losses["loss_cls"]), float(losses["loss_map_cls"]),
         float(losses["loss_plan_vocab"])))
