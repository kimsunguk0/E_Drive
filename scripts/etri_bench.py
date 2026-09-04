#!/usr/bin/env python
"""단독 GPU 학습 처리량 실측 — B200 일정 추정의 미측정 계수를 메운다.

왜
--
B200 일정표에 이런 항이 들어 있었다.

    base/tiny 배수      5~7x      (FLOPs·config 근거 있음)
    B200/H200 실효      1.5~1.9x  (스펙 근거 있음)
    단독화 이득         1.3~1.6x  <- **근거 없는 짐작**

Phase G의 처리량 실측은 전부 **타 프로세스가 92,864 MiB·util 100%를 점유한 경합
상태**에서 뽑은 것이고, 단독 H200을 한 번도 측정하지 못했다. 그래서 "4x B200 2.5일"
추정에 +-40% 불확실성이 실려 있었다. GPU가 비었을 때 그 계수를 실측으로 바꾼다.

Phase G와 직접 비교되도록 조건을 맞춘다: 캐시 로더, workers 10, 워밍업 제외 중앙값.

    Phase G (경합)   bs 1: 0.743 s/iter   bs 4: 1.521   bs 8: 2.601

    python scripts/etri_bench.py --bs 1,4,8,16 --iters 40
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
PHASE_G = {1: 0.743, 4: 1.521, 8: 2.601}      # 경합 상태 실측 (logs/phaseG_report.md)


def bench(cfg, ann, bs, iters, workers, warmup):
    import torch
    from mmcv.parallel import MMDataParallel
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet.datasets import build_dataloader

    cfg.data.train.ann_file = ann
    ds = build_dataset(cfg.data.train)
    # mmdet 2.14의 build_dataloader는 shuffler_sampler를 모른다 (VAD 학습 스크립트는
    # custom build를 쓴다). 벤치에는 필요 없으므로 평범한 DataLoader로 만든다.
    dl = build_dataloader(ds, samples_per_gpu=bs, workers_per_gpu=workers,
                          num_gpus=1, dist=False, seed=0)
    torch.manual_seed(0)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"))
    model.init_weights()
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()

    dts, its = [], []
    it = iter(dl)
    t_prev = time.time()
    for k in range(iters + warmup):
        try:
            data = next(it)
        except StopIteration:
            it = iter(dl)
            data = next(it)
        t_data = time.time()
        losses = model(**data)
        loss = sum(v.mean() if v.dim() else v
                   for vs in losses.values()
                   for v in ([vs] if torch.is_tensor(vs) else vs))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
        opt.step()
        torch.cuda.synchronize()
        t_end = time.time()
        if k >= warmup:
            dts.append(t_data - t_prev)
            its.append(t_end - t_prev)
        t_prev = t_end
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    del model, opt, dl, ds
    torch.cuda.empty_cache()
    s_iter = float(np.median(its))
    return dict(bs=bs, s_iter=round(s_iter, 4),
                samples_s=round(bs / s_iter, 3),
                peak_MiB=round(peak),
                data_frac=round(float(np.median(dts)) / s_iter * 100, 2),
                n=len(its))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/VAD/VAD_etri_tvad_bootstrap.py"))
    ap.add_argument("--ann", default="/tmp/pm97/data/etri/pkl/overfit8.pkl")
    ap.add_argument("--bs", default="1,4,8")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)   # Phase G 조건
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    import torch
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU {torch.cuda.get_device_name(0)}  "
          f"여유 {free/2**20:,.0f} / {total/2**20:,.0f} MiB")
    print(f"config {os.path.basename(args.config)}  workers {args.workers}  "
          f"iters {args.iters} (워밍업 {args.warmup} 제외)\n")

    rows = []
    for bs in [int(x) for x in args.bs.split(",")]:
        try:
            r = bench(cfg, args.ann, bs, args.iters, args.workers, args.warmup)
        except RuntimeError as e:
            print(f"  bs {bs:2d}  실패: {str(e)[:80]}")
            continue
        g = PHASE_G.get(bs)
        r["phaseG_s_iter"] = g
        r["speedup_vs_contended"] = round(g / r["s_iter"], 3) if g else None
        rows.append(r)
        print(f"  bs {bs:2d}  {r['s_iter']:6.3f} s/iter  {r['samples_s']:6.3f} 샘플/s  "
              f"peak {r['peak_MiB']:7,d} MiB  data {r['data_frac']:5.2f}%"
              + (f"   vs 경합 {g:.3f} → **{r['speedup_vs_contended']:.2f}x**" if g else ""))

    sp = [r["speedup_vs_contended"] for r in rows if r["speedup_vs_contended"]]
    if sp:
        print(f"\n단독화 이득 실측: {min(sp):.2f}~{max(sp):.2f}x "
              f"(중앙 {np.median(sp):.2f}x)")
        print(f"  B200 일정표의 짐작값 1.3~1.6x 와 대조하라")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"저장 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
