#!/usr/bin/env python
"""Phase A tournament verdict metrics on our val (ETRI_SCOREDRIVE_DESIGN.md §11).

기존 etri_vad_eval.py 의 배포충실 streaming(시나리오별 reset + 선행 depth프레임)을
그대로 재사용하되, goal_decoder 의 **vocab logits 를 forward hook 으로 캡처**해
후보 순위 지표를 val 에서 계산한다:

    top-1 L2 (=oracle@1)  oracle@3/6/20   full-K cover   gap3(top1-oracle@3)
    top1-hit(최근접 anchor 적중률)   median rank(최근접 anchor 의 모델 순위)

D[c] 는 decoder 와 동일한 challenge 가중 [11,11,5,5,2,2]/36 을 anchors_abs(누적)와
GT(누적)에 적용한다. 학습 로그의 plan_vocab_* 와 정의가 같아 판끼리·baseline 과
사과-대-사과다.

    python eval_vocab_ranking.py <config> <ckpt> [--depth 6] [--limit N]
"""
import argparse
import os
import sys
import time

import numpy as np

SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SD)
import etri_vad_eval as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--ann-file",
                    default="/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl")
    ap.add_argument("--split", default="val_idx")
    ap.add_argument("--depth", type=int, default=6,
                    help="배포충실 누적 깊이(공식 clip=7프레임 -> 6). 0=연속 streaming")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump", default="",
                    help="주면 per-anchor logits/Dgt/meta 를 npz 로 저장(shortlist 실험용)")
    args = ap.parse_args()

    args.config = os.path.abspath(args.config)
    args.checkpoint = os.path.abspath(args.checkpoint)
    # ★ 모델 코드 트리 = config 가 속한 repo (dense_vocab_v1). E.REPO(옛 etri_vad
    # 트리)로 열면 AnchorGroundedVocabularyDecoder 가 없어 죽거나 다른 코드로 ckpt 를
    # 읽는 사고가 난다.
    repo = args.config.split("/projects/")[0]
    os.chdir(repo)
    sys.path.insert(0, repo)
    import importlib
    import torch
    # 추론 결정성: run-to-run top-1 jitter(근접 logit tie flip) 제거해 기준점을 고정.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    scen, frame, ds_idx, cache_idx = E.build_index(args.split, None, ds=ds)
    scens = sorted(set(scen))
    if args.limit:
        scens = scens[:args.limit]
    keep = np.isin(scen, scens)
    print(f"{args.split}: 앵커 {keep.sum():,} / 시나리오 {len(scens)} "
          f"depth={args.depth}", flush=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    model, load_info = E.load_model(cfg, args.checkpoint)
    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu])
    model.eval()

    gd = model.module.pts_bbox_head.goal_decoder
    anchors = gd.anchors_abs.detach().cpu().numpy().astype(np.float64)  # [K,6,2]
    w = gd.challenge_w.detach().cpu().numpy().astype(np.float64)        # [6]
    K = anchors.shape[0]
    holder = []

    def hook(m, i, o):
        # decoder.forward -> (wp, logits, index); logits [B=1,K]
        holder.append(o[1].detach().float().cpu().numpy()[0])
    gd.register_forward_hook(hook)

    # GT (누적) — 기존 top-1 평가와 동일한 ValSet 사용
    sys.path.insert(0, os.path.join(E.ADCL, "scripts"))
    sys.path.insert(0, os.path.join(E.ADCL, "src"))
    from etri_table import ValSet  # noqa: E402
    v = ValSet()
    assert np.array_equal(v.vi, cache_idx), "val 인덱스가 어긋났다"
    gt = np.asarray(v.gt, dtype=np.float64)          # [2280,6,2] 누적
    wrow = np.asarray(v.w, dtype=np.float64)         # [2280] proxy weight

    key2ds = {(i["scene_token"], int(i["frame_idx"])): j
              for j, i in enumerate(ds.data_infos)}

    N = len(scen)
    dumpL = np.full((N, K), np.nan, np.float32) if args.dump else None
    dumpD = np.full((N, K), np.nan, np.float32) if args.dump else None
    selD = np.full(N, np.nan)
    o3 = np.full(N, np.nan); o6 = np.full(N, np.nan); o20 = np.full(N, np.nan)
    coverD = np.full(N, np.nan)
    rank = np.full(N, -1, dtype=int)
    hit1 = np.zeros(N, bool)

    t0 = time.time(); n_done = 0
    for si, s in enumerate(scens):
        m = scen == s
        order = np.argsort(frame[m])
        rows = np.where(m)[0][order]
        E.reset_stream(model.module)
        for r in rows:
            if args.depth:
                E.reset_stream(model.module)
                for k in range(args.depth, 0, -1):
                    wk = (s, int(frame[r]) - k * 5)
                    if wk not in key2ds:
                        continue
                    dw = collate([ds[key2ds[wk]]], samples_per_gpu=1)
                    with torch.no_grad():
                        model(return_loss=False, rescale=True, **dw)
            data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
            holder.clear()
            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)
            logits = holder[-1]                       # [K]
            gtr = gt[r]                               # [6,2] 누적
            D = (w[None, :] * np.linalg.norm(
                anchors - gtr[None], axis=2)).sum(1)   # [K]
            od = np.argsort(-logits)                  # 모델 순위(내림차순)
            best = int(D.argmin())
            selD[r] = D[od[0]]
            o3[r] = D[od[:3]].min(); o6[r] = D[od[:6]].min()
            o20[r] = D[od[:20]].min(); coverD[r] = D.min()
            rank[r] = int(np.where(od == best)[0][0]) + 1
            hit1[r] = (od[0] == best)
            if args.dump:
                dumpL[r] = logits.astype(np.float32)
                dumpD[r] = D.astype(np.float32)
            n_done += 1
        el = time.time() - t0
        print(f"  [{si+1:2d}/{len(scens)}] {s} {len(rows)}앵커 "
              f"{n_done/max(el,1e-9):.2f} it/s "
              f"ETA {(keep.sum()-n_done)/max(n_done/el,1e-9)/60:.1f}분", flush=True)

    def report(mask, name):
        mm = mask & ~np.isnan(selD)
        if mm.sum() == 0:
            return None
        wm = wrow[mm]
        wmean = lambda a: float(np.average(a[mm], weights=wm))  # noqa: E731
        pmean = lambda a: float(a[mm].mean())                   # noqa: E731
        # OFFICIAL: challenge 가중 D3 [11,11,5,5,2,2]/36(per-timestep) + val proxy weight
        # (per-anchor). 모든 top-k 는 모델 logits 순위, oracle 는 그 순위 안 최소 D.
        # 대회 목표가 가중 L2 이므로 이 블록이 판정 기준이다.
        rec = dict(
            n=int(mm.sum()),
            # ---- official (weighted) ----
            top1_w=wmean(selD),
            oracle3_w=wmean(o3), oracle6_w=wmean(o6), oracle20_w=wmean(o20),
            cover_w=wmean(coverD),
            gap3_w=wmean(selD) - wmean(o3),
            gap6_w=wmean(selD) - wmean(o6),
            besthit_w=wmean(hit1.astype(float)),   # official-weight exact-best hit
            rank_med=float(np.median(rank[mm])),
            rank_p90=float(np.percentile(rank[mm], 90)),
            # ---- diagnostic (plain, 진단용) ----
            top1=pmean(selD), oracle3=pmean(o3), oracle6=pmean(o6),
            oracle20=pmean(o20), cover=pmean(coverD),
            gap3=pmean(selD) - pmean(o3), besthit=pmean(hit1.astype(float)))
        print(f"\n[{name}]  n={rec['n']}")
        print("  -- OFFICIAL (weighted D3 + proxy) --")
        print(f"  top-1 W {rec['top1_w']:.4f}   "
              f"oracle@3 {rec['oracle3_w']:.4f}  @6 {rec['oracle6_w']:.4f}  "
              f"@20 {rec['oracle20_w']:.4f}  coverK {rec['cover_w']:.4f}")
        print(f"  gap: top1-o3 {rec['gap3_w']:.4f}   top1-o6 {rec['gap6_w']:.4f}   "
              f"exact-best hit {rec['besthit_w']:.4f}   "
              f"rank med {rec['rank_med']:.0f} p90 {rec['rank_p90']:.0f}")
        print("  -- diagnostic (plain) --")
        print(f"  top-1 {rec['top1']:.4f}  o@3 {rec['oracle3']:.4f}  "
              f"o@6 {rec['oracle6']:.4f}  gap3 {rec['gap3']:.4f}  "
              f"best-hit {rec['besthit']:.4f}")
        return rec

    print("\n" + "=" * 60)
    tag = args.tag or os.path.basename(os.path.dirname(args.checkpoint))
    print(f"CKPT {args.checkpoint}  TAG {tag}")
    rows_out = {}
    rows_out["all"] = report(keep, f"{tag}  전체 val")
    rows_out["f>=30"] = report(keep & (frame >= 30), f"{tag}  frame>=30 (배포충실)")

    if args.dump:
        np.savez(args.dump, logits=dumpL, Dgt=dumpD, wrow=wrow,
                 frame=frame, scen=scen.astype(str), keep=keep,
                 anchors=anchors, cw=w)
        print(f"덤프 저장 {args.dump}  logits{dumpL.shape}")

    out = args.out or os.path.join(
        E.ADCL, "logs",
        f"rank_{tag}_{os.path.basename(args.checkpoint).replace('.pth','')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    json.dump(dict(config=args.config, checkpoint=args.checkpoint, tag=tag,
                   depth=args.depth, rows=rows_out), open(out, "w"), indent=1)
    print(f"\n저장 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
