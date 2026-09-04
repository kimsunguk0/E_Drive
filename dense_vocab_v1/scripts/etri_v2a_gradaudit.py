#!/usr/bin/env python
"""v2a 발사 전 gradient 감사 — planning loss가 어느 파라미터 그룹에 도달하는가.

형님이 smoke 조건으로 요구한 항목:
    goal encoder gradient > 0
    content value projection gradient > 0
    raw spatial evidence gradient > 0          (backbone/neck. lr_mult=0이어도 grad는 흘러야 한다)
    BEV encoder / image backbone gradient > 0

주의: `lr_mult=0`은 **업데이트를 막는 것**이고 gradient를 막는 게 아니다. 여기서 0이
나오면 그건 freeze가 아니라 **경로가 끊긴 것**이므로 발사하면 안 된다.

planning loss만 따로 backward해서 v2a 경로를 분리한다 (aux는 0.05로 스케일되지만
det/map head를 통해 backbone까지 별도로 흐르므로 섞으면 판별이 안 된다).

    python scripts/etri_v2a_gradaudit.py --config <cfg> --ckpt <ckpt> --iters 3
"""
import argparse
import collections
import importlib
import os
import sys

GROUPS = [
    ("goal_decoder",        lambda k: "goal_decoder" in k),
    ("content_value_proj",  lambda k: "content_value_proj" in k),
    ("img_backbone",        lambda k: k.startswith("img_backbone")),
    ("img_neck",            lambda k: k.startswith("img_neck")),
    ("bev encoder",         lambda k: "transformer.encoder" in k),
    ("bev_embedding",       lambda k: "bev_embedding" in k),
    ("old ego_fut_decoder", lambda k: "ego_fut_decoder" in k),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(os.path.abspath(args.config))
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    ds = build_dataset(cfg.data.train)
    print(f"train 앵커 {len(ds):,}  data_infos {len(ds.data_infos):,}  "
          f"min_frame_idx {getattr(ds, 'min_frame_idx', 0)}")
    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"),
                        test_cfg=cfg.get("test_cfg"))
    if args.ckpt:
        sd = torch.load(os.path.abspath(args.ckpt), map_location="cpu")
        sd = sd.get("state_dict", sd)
        r = model.load_state_dict(sd, strict=False)
        print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)}"
              f"  unexpected {len(r.unexpected_keys)}")
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.train()

    acc = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    plan_seen = []
    for it in range(args.iters):
        data = collate([ds[it * 977 % len(ds)]], samples_per_gpu=1)
        losses = mm(return_loss=True, **data)
        plan = [v for k, v in losses.items()
                if "plan" in k and torch.is_tensor(v) and v.requires_grad]
        assert plan, f"planning loss가 없다: {list(losses.keys())}"
        loss = sum(v.sum() for v in plan)
        plan_seen = [k for k in losses if "plan" in k]
        mm.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in mm.module.named_parameters():
            if p.grad is None:
                continue
            g = float(p.grad.detach().abs().max())
            for gname, pred in GROUPS:
                if pred(name):
                    acc[gname] = max(acc[gname], g)
                    cnt[gname] += 1
        print(f"  iter {it}  planning loss {float(loss):.4f}  "
              f"({', '.join(plan_seen)})", flush=True)

    print(f"\n{'그룹':<22}{'파라미터수':>12}{'max |grad|':>16}{'판정':>10}")
    ok = True
    for gname, _ in GROUPS:
        n = cnt[gname] // max(args.iters, 1)
        g = acc[gname]
        if gname == "old ego_fut_decoder":
            verdict = "제거됨" if n == 0 else ("★살아있음★" if g > 0 else "grad 0")
        else:
            good = g > 0
            ok &= good
            verdict = "PASS" if good else "★FAIL★"
        print(f"{gname:<22}{n:>12}{g:>16.3e}{verdict:>10}")
    print(f"\n  planning loss만 backward했다. 위 그룹 전부에 grad가 도달해야")
    print(f"  '영상 content -> 궤적' 경로가 실제로 연결된 것이다.")
    print(f"  lr_mult=0(backbone/neck)은 업데이트만 막는다 -- grad는 0이 아니어야 한다.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
