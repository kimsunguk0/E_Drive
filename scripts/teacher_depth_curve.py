#!/usr/bin/env python
"""dense teacher의 controlled history-depth curve (val38, frame>=30 고정).

목적: current-only 가설의 정보 상한을 직접 잰다.
  depth 0 = 매 target frame 에서 reset 후 그 프레임 1장만 forward (순수 t0)
  depth N = reset 후 과거 N장(interval 5)을 먹인 뒤 target frame
  depth 6 = 기존 champion 조건

target frame, checkpoint, selector, 가중을 전부 고정하고 depth 만 바꾼다.

판정:
  depth0 slO@12 <= 0.20~0.25  -> t0 에 정보는 있다 = student representation 문제
  depth0 slO@12 >= 0.40, depth2~6 에서 크게 개선 -> t0 에 정보가 없다 = temporal 필요
  depth1~2 에서 대부분 회복 -> 2~3 frame sparse 모델로 충분

  python teacher_depth_curve.py --depths 0,1,2,3,6 --gpu 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
CFG = os.path.join(A, "work_dirs/pa2_softceexp/VAD_etri_pa2_softceexp.py")
CKPT = os.path.join(A, "work_dirs/pa2_softceexp/epoch_2.pth")
VAL_PKL = "/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl"
LAM = (0.0, 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="0,1,2,3,6")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(A, "logs/teacher_depth_curve.json"))
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]

    sys.path.insert(0, os.path.join(A, "scripts"))
    import sparse_common as C
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    rows_all = arr["val_idx"]
    keep = arr["frame"][rows_all] >= 30
    rows = rows_all[keep]
    weight = arr["val_weight"][keep]
    tgt = C.precompute_targets(arr, rows, bank, weight=weight)
    key_of = [(arr["scenarios"][arr["scen_idx"][r]], int(arr["frame"][r]))
              for r in rows]

    repo = os.path.join(A, "dense_vocab_v1")
    os.chdir(repo)
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(A, "scripts"))
    import importlib
    import torch
    import etri_vad_eval as E
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(CFG)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = VAL_PKL
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    model, _ = E.load_model(cfg, CKPT)
    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu])
    model.eval()
    gd = model.module.pts_bbox_head.goal_decoder
    holder = []
    gd.register_forward_hook(lambda m, i, o: holder.append(
        o[1].detach().float().cpu().numpy()[0]))
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j
              for j, i in enumerate(ds.data_infos)}

    if args.limit_scenes:
        keep_sc = sorted({s for s, _ in key_of})[:args.limit_scenes]
        sel = np.array([i for i, (s, _) in enumerate(key_of) if s in keep_sc])
        rows = rows[sel]
        key_of = [key_of[i] for i in sel]
        for k in ("D3gt", "goal_xy", "weight"):
            tgt[k] = tgt[k][sel]
        tgt["buckets"] = {k: np.asarray(v)[sel] for k, v in tgt["buckets"].items()}
    print(f"target frames = {len(rows)} (val38, frame>=30) depths={depths}",
          flush=True)

    results = {}
    for depth in depths:
        t0 = time.time()
        logits = np.empty((len(rows), int(gd.anchors_abs.shape[0])), np.float32)
        for n, (s, f) in enumerate(key_of):
            E.reset_stream(model.module)
            for k in range(depth, 0, -1):
                wk = (s, f - k * 5)
                if wk not in key2ds:
                    continue
                dw = collate([ds[key2ds[wk]]], samples_per_gpu=1)
                with torch.no_grad():
                    model(return_loss=False, rescale=True, **dw)
            data = collate([ds[key2ds[(s, f)]]], samples_per_gpu=1)
            holder.clear()
            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)
            logits[n] = holder[-1]
            if (n + 1) % 500 == 0:
                el = time.time() - t0
                print(f"    depth{depth} {n+1}/{len(rows)} "
                      f"{(n+1)/el:.1f} it/s", flush=True)
        rep = C.eval_logits(logits.astype(np.float64), tgt["D3gt"], tgt["goal_xy"],
                            tgt["cand_end5"], tgt["anchor_dist"], tgt["nms_tau"],
                            tgt["weight"], lambdas=LAM, buckets=tgt["buckets"])
        results[str(depth)] = rep
        bk = rep.get("buckets_lambda0.1", {})
        bs = " ".join(f"{k}={v['realized']:.3f}" for k, v in bk.items() if v)
        print(f"  depth {depth}: top1={rep['top1']:.4f} o@3={rep['oracle3']:.4f} "
              f"o@12={rep['oracle12']:.4f} slO@12={rep['shortlist_oracle12']:.4f} "
              f"real λ0={rep['realized']['0']:.4f} λ0.1={rep['realized']['0.1']:.4f}"
              f"  [{bs}]  ({time.time()-t0:.0f}s)", flush=True)
        np.save(os.path.join(A, f"logs/teacher_depth{depth}_logits.npy"), logits)

    json.dump(results, open(args.out, "w"), indent=1, default=float)
    print("\n=== depth curve (val38 frame>=30, 동일 프레임/체크포인트/selector) ===")
    print(f"{'depth':>6} {'top1':>8} {'o@12':>8} {'slO@12':>9} {'real λ0':>9} {'λ0.1':>8}")
    for d in depths:
        r = results[str(d)]
        print(f"{d:>6} {r['top1']:>8.4f} {r['oracle12']:>8.4f} "
              f"{r['shortlist_oracle12']:>9.4f} {r['realized']['0']:>9.4f} "
              f"{r['realized']['0.1']:>8.4f}")
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
