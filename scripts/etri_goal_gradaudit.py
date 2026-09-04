#!/usr/bin/env python
"""C0-T gradient 감사 — planning loss가 goal encoder / BEV / backbone까지 가는가.

형님 §11-2 확인 항목 중 gradient 부분이다.
    goal encoder gradient 존재
    BEV / backbone gradient 존재
    can_bus 불변성 유지 (별도 스크립트)

`loss_plan_metric` 하나만 backward 해서 모듈별 grad norm을 본다.
과거 프레임 backbone은 `obtain_history_bev`의 `self.eval()` + `no_grad` 때문에
grad가 0인 것이 정상이다.

    python scripts/etri_goal_gradaudit.py
"""
import argparse
import importlib
import os
import sys

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))

GROUPS = [
    ("img_backbone (전체)", "img_backbone."),
    ("img_backbone.layer4", "img_backbone.layer4"),
    ("img_neck", "img_neck."),
    ("BEV encoder", "pts_bbox_head.transformer.encoder"),
    ("BEV embedding", "pts_bbox_head.bev_embedding"),
    ("agent decoder", "pts_bbox_head.transformer.decoder"),
    ("map decoder", "pts_bbox_head.transformer.map_decoder"),
    ("★ goal cond_mlp", "pts_bbox_head.goal_decoder.cond_mlp"),
    ("★ goal ts_embed", "pts_bbox_head.goal_decoder.ts_embed"),
    ("★ goal cross-attn", "pts_bbox_head.goal_decoder.blocks"),
    ("★ goal out Linear", "pts_bbox_head.goal_decoder.out"),
    ("(구) ego_fut_decoder", "pts_bbox_head.ego_fut_decoder"),
    ("(구) ego_agent_decoder", "pts_bbox_head.ego_agent_decoder"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_c0t_overfit.py"))
    ap.add_argument("--ckpt", default="/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth")
    ap.add_argument("--bs", type=int, default=2)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.ckpt = os.path.abspath(args.ckpt) if args.ckpt else ""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet.datasets import build_dataloader

    ds = build_dataset(cfg.data.train)
    dl = build_dataloader(ds, samples_per_gpu=args.bs, workers_per_gpu=4,
                          num_gpus=1, dist=False, seed=0)
    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
    model.init_weights()
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        own = model.state_dict()
        for k in [k for k, v in sd.items()
                  if k in own and tuple(own[k].shape) != tuple(v.shape)]:
            sd.pop(k)
        r = model.load_state_dict(sd, strict=False)
        gm = [k for k in r.missing_keys if "goal_decoder" in k]
        print(f"ckpt {os.path.basename(args.ckpt)}  missing {len(r.missing_keys)} "
              f"(그중 goal_decoder {len(gm)}개 -- 새 모듈이므로 정상)")
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.train()

    data = next(iter(dl))
    losses = mm(**data)
    plan = losses["loss_plan_metric"]
    plan = plan.mean() if plan.dim() else plan
    tot = sum(x.mean() if torch.is_tensor(x) and x.dim() else x
              for vs in losses.values() for x in ([vs] if torch.is_tensor(vs) else vs))
    print(f"\nloss_plan_metric {float(plan):.4f}   전체 loss {float(tot):.4f}   "
          f"planning 비중 {100*float(plan)/max(float(tot),1e-9):.1f}%")

    mm.zero_grad(set_to_none=True)
    plan.backward()

    def gn(pref):
        t, n = 0.0, 0
        for k, p in mm.module.named_parameters():
            if k.startswith(pref) and p.grad is not None:
                t += float(p.grad.detach().pow(2).sum())
                n += p.grad.numel()
        return (t ** 0.5, n)

    print(f"\nbackward = loss_plan_metric 만")
    print(f"{'모듈':<26}{'grad L2':>14}{'params':>12}{'상태':>8}")
    res = {}
    for name, pref in GROUPS:
        g, n = gn(pref)
        if n == 0:
            has = any(k.startswith(pref) for k, _ in mm.module.named_parameters())
            print(f"{name:<26}{'-':>14}{'-':>12}{'grad 0' if has else '모듈 없음':>8}")
            res[name] = 0.0 if has else None
            continue
        print(f"{name:<26}{g:>14.6e}{n:>12,d}{'OK' if g > 1e-12 else '0':>8}")
        res[name] = g

    print("\n=== 판정 ===")
    bad = []
    need = ["★ goal cond_mlp", "★ goal ts_embed", "★ goal cross-attn",
            "★ goal out Linear", "BEV encoder", "img_backbone (전체)"]
    for k in need:
        v = res.get(k)
        if not v:
            bad.append(k)
    if not bad:
        print("  PASS — goal encoder / cross-attn / 출력 / BEV / backbone 전부 gradient 도달")
        b = res["img_backbone (전체)"]
        o = res["★ goal out Linear"]
        print(f"    비율 backbone/goal_out = {b/max(o,1e-12):.3e}")
    else:
        for k in bad:
            print(f"  FAIL — {k} 에 gradient가 없다")
    old = res.get("(구) ego_fut_decoder")
    print(f"\n  참고: 구 ego_fut_decoder grad = {old}  "
          f"(goal_decoder를 쓰면 0이어야 정상 -- 출력 경로에서 빠졌다)")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
