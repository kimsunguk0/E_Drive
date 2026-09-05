#!/usr/bin/env python
"""⑤-C2 distillation 용 teacher logits 덤프 (dense champion, 학습 시나리오).

teacher = pa2_softceexp/epoch_2 (Phase A 확정 champion). goal/cmd 미입력이므로
compliance 무관하고, 추론 시에는 사용하지 않는다(학습 라벨 용도).

비용 절감: 배포충실 depth-6 프로토콜(프레임당 7 forward, ~590ms) 대신 시나리오를
interval-5 로 순차 streaming 해 프레임당 1 forward(~84ms)로 만든다. dense VAD 가
학습된 2Hz cadence 와 동일하므로 teacher 품질에도 유리하다.

  python dump_teacher_logits.py --offsets 0,1 --out $A/data/etri/teacher_off01.npz --gpu 0
"""
import argparse
import os
import sys
import time

import numpy as np

A = "/NHNHOME/data/sukim/adcl"
CFG = os.path.join(A, "work_dirs/pa2_softceexp/VAD_etri_pa2_softceexp.py")
CKPT = os.path.join(A, "work_dirs/pa2_softceexp/epoch_2.pth")
TRAIN_PKL = "/tmp/pm97/data/etri/pkl/etri_train330_goal.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", default="0,1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--depth", type=int, default=0,
                    help="0=시나리오 연속 streaming, N=프레임마다 reset+선행 N프레임(배포충실)")
    args = ap.parse_args()
    offsets = [int(x) for x in args.offsets.split(",")]

    sys.path.insert(0, os.path.join(A, "scripts"))
    import sparse_common as C
    arr = C.load_arrays()
    split = C.make_split(arr, all_frames=True, holdout_offset=True)
    train_scen = set(np.unique(arr["scen_idx"][split["train_rows"]]).tolist())
    # (scenario, frame) -> ego_cache row  (학습 row 만)
    key2row = {}
    for r in split["train_rows"]:
        key2row[(arr["scenarios"][arr["scen_idx"][r]], int(arr["frame"][r]))] = int(r)

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
    cfg.data.test.ann_file = TRAIN_PKL
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    model, _ = E.load_model(cfg, CKPT)
    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu])
    model.eval()
    gd = model.module.pts_bbox_head.goal_decoder
    K = int(gd.anchors_abs.shape[0])
    holder = []
    gd.register_forward_hook(lambda m, i, o: holder.append(
        o[1].detach().float().cpu().numpy()[0]))

    # (scene_token, frame_idx) -> dataset index
    key2ds = {}
    for j, info in enumerate(ds.data_infos):
        key2ds[(info["scene_token"], int(info["frame_idx"]))] = j
    scenes = sorted({s for s, _ in key2ds} &
                    {arr["scenarios"][i] for i in train_scen})
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    print(f"scenes={len(scenes)} offsets={offsets} K={K}", flush=True)

    rows_out, logits_out = [], []
    t0 = time.time()
    done = 0
    for si, s in enumerate(scenes):
        for off in offsets:
            frames = sorted(f for (sc, f) in key2ds if sc == s and f % 5 == off)
            if not args.depth:
                E.reset_stream(model.module)
            for f in frames:
                row = key2row.get((s, f))
                if row is None:          # 학습 pool 밖(=probe/tuneval) 이면 건너뛴다
                    continue
                if args.depth:
                    # 배포충실: 매 프레임 reset 후 선행 depth 프레임(interval 5)을 먹인다
                    E.reset_stream(model.module)
                    for k in range(args.depth, 0, -1):
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
                rows_out.append(row)
                logits_out.append(holder[-1])
                done += 1
        if (si + 1) % 20 == 0 or si == len(scenes) - 1:
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            rem = (len(scenes) - si - 1) * (done / max(si + 1, 1))
            print(f"  [{si+1}/{len(scenes)}] {s} frames={done} "
                  f"{rate:.1f} it/s ETA {rem/max(rate,1e-9)/60:.1f}분", flush=True)

    rows_out = np.asarray(rows_out, np.int64)
    logits_out = np.asarray(logits_out, np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, rows=rows_out, logits=logits_out,
             teacher_ckpt=CKPT, offsets=np.asarray(offsets))
    print(f"saved {args.out}  rows={rows_out.shape} logits={logits_out.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
