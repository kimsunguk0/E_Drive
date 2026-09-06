#!/usr/bin/env python
"""⑤-C 체크포인트 최종 평가: val38(공식 가중) + tuneval, λ sweep, 버킷, seed 분산.

  python eval_sparse_scoredrive.py work_dirs/sc_pos_s0/best.pth work_dirs/sc_aux_s0/best.pth ...
     [--split val|tune|both] [--dump]

기준: dense champion selector realized(λ0.1) 0.2411, +velocity residual 0.2345.
게이트: realized<=0.25 KEEP / <=0.24 강함 / >0.27 부족, slO@12<=0.17 권장.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402

LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 3.0)


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu")
    fn = bool(ck.get("args", {}).get("feature_norm", 0))
    ln = bool(ck.get("args", {}).get("logit_norm", 0))
    a = ck.get("args", {})
    nh = int(a.get("n_hist", 0))
    m = TrainableSparseScoreDrive(
        C.BANK_A0, feature_norm=fn, logit_norm=ln, arch=a.get("arch", "resnet34"),
        offset_sample=bool(a.get("offset_sample", 0)),
        use_p1=bool(a.get("use_p1", 0)), n_hist=nh,
        corridor=tuple(float(x) for x in str(a.get("corridor", "0")).split(",") if x != ""),
        seq_head=a.get("seq_head", "none"),
        speed_head=bool(a.get("w_speed", 0) > 0), speed_gamma=0.0).to(device)
    m.fuse_mul = bool(a.get("fuse_mul", 1))
    m.merge_encode = bool(a.get("merge_encode", 0))
    m._eval_nhist = nh
    m._eval_hscale = float(a.get("hist_scale", 1.0))
    sd = ck["model"]
    m.load_state_dict(sd)
    m.eval()
    return m


def champion_reference(bank, arr):
    """dense champion(champ_det.npz)의 val38 realized 를 동일 harness 로 재계산."""
    ch = np.load(os.path.join(A, "logs/dump/champ_det.npz"), allow_pickle=True)
    logits = ch["logits"].astype(np.float64)          # [2280,1024] (val_idx 순)
    frame = ch["frame"]; keep = ch["keep"]
    v = keep & (frame >= 30)
    rows = arr["val_idx"]
    # champ_det rows 는 val_idx 순서 (val_weight==wrow, [:5] 동일로 확인됨)
    assert np.array_equal(frame.astype(np.int64), arr["frame"][rows]), \
        "champ_det 행 순서가 val_idx 와 다르다"
    tgt = C.precompute_targets(arr, rows, bank, weight=arr["val_weight"])
    sub = v
    rep = C.eval_logits(logits[sub], tgt["D3gt"][sub], tgt["goal_xy"][sub],
                        tgt["cand_end5"], tgt["anchor_dist"], tgt["nms_tau"],
                        tgt["weight"][sub], lambdas=LAMBDAS,
                        buckets={k: np.asarray(b)[sub] for k, b in tgt["buckets"].items()})
    return rep


def eval_on_split(model, arr, bank, lidar2img_t, device, split, batch, dump_dir, tag):
    """평가 대상 행을 먼저 거른 뒤 forward 한다.
    (temporal 은 과거 프레임이 없는 행을 forward 하면 파일이 없어 죽는다)"""
    nh = getattr(model, "_eval_nhist", 0)
    hs = getattr(model, "_eval_hscale", 1.0)
    if split == "val":
        rows = arr["val_idx"]
        weight = arr["val_weight"]
        sub = arr["frame"][rows] >= 30
    else:
        s = C.make_split(arr)
        rows = s["tune_rows"]
        weight = None
        sub = np.ones(len(rows), bool)
    if nh:
        sub = sub & C.history_available(arr, rows, nh)
    rows = rows[sub]
    weight = None if weight is None else weight[sub]
    logits = C.run_logits(model, arr, rows, lidar2img_t, device, batch=batch,
                          amp=True, n_hist=nh, hist_scale=hs)
    tgt = C.precompute_targets(arr, rows, bank, weight=weight)
    rep = C.eval_logits(logits, tgt["D3gt"], tgt["goal_xy"], tgt["cand_end5"],
                        tgt["anchor_dist"], tgt["nms_tau"], tgt["weight"],
                        lambdas=LAMBDAS, buckets=tgt["buckets"])
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        np.savez(os.path.join(dump_dir, f"logits_{tag}_{split}.npz"),
                 logits=logits.astype(np.float32), rows=rows,
                 D3gt=tgt["D3gt"].astype(np.float32),
                 goal_xy=tgt["goal_xy"], weight=tgt["weight"])
    return rep


def fmt(rep):
    r = rep["realized"]
    best = min(r, key=lambda k: r[k])
    lam = "  ".join(f"{k}={v:.4f}" for k, v in r.items())
    return (f"n={rep['n']:5d} top1={rep['top1']:.4f} o@3={rep['oracle3']:.4f} "
            f"o@6={rep['oracle6']:.4f} o@12={rep['oracle12']:.4f} "
            f"slO@12={rep['shortlist_oracle12']:.4f}\n"
            f"        realized λ: {lam}\n"
            f"        best λ={best} -> {r[best]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--split", default="both", choices=["val", "tune", "both"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--out", default=os.path.join(A, "logs/sparse_c_eval.json"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    lidar2img_t = torch.from_numpy(C.build_global_lidar2img())
    dump_dir = os.path.join(A, "logs/dump_sparse_c") if args.dump else ""

    splits = ["val", "tune"] if args.split == "both" else [args.split]
    results = {}

    print("=" * 100)
    cref = champion_reference(bank, arr)
    print(f"[dense champion  val38] {fmt(cref)}")
    print(f"    (published realized λ0.1 = 0.2411 ; +vel residual = 0.2345)")
    results["champion_val"] = cref
    print("=" * 100)

    for ck in args.ckpts:
        tag = os.path.basename(os.path.dirname(ck))
        model = load_model(ck, device)
        results[tag] = {}
        for sp in splits:
            rep = eval_on_split(model, arr, bank, lidar2img_t, device, sp,
                                args.batch, dump_dir, tag)
            results[tag][sp] = rep
            rr = rep["realized"]
            bl = min(rr, key=lambda k: rr[k])
            sl = rep["shortlist_oracle12"]
            gate = ("성공" if (sl <= 0.20 and rr[bl] <= 0.30) else
                    "KEEP" if (sl <= 0.25 and rr[bl] <= 0.35) else
                    "부분(slO만)" if sl <= 0.25 else "FAIL")
            strong = ""
            print(f"[{tag:16s} {sp:4s}] {fmt(rep)}  -> {gate}{strong}")
            if sp == "val" and "buckets_lambda0.1" in rep:
                bk = rep["buckets_lambda0.1"]
                bs = "  ".join(
                    f"{k}={v['realized']:.3f}(n{v['n']})" for k, v in bk.items()
                    if v is not None)
                print(f"    buckets(λ0.1 realized): {bs}")
        del model
        torch.cuda.empty_cache()

    json.dump(results, open(args.out, "w"), indent=1, default=float)
    print(f"\nsaved {args.out}")

    # seed 분산 요약 (이름에 _s0/_s1 규약)
    print("\n--- seed variance (val realized λ0.1) ---")
    arms = {}
    for tag, r in results.items():
        if tag == "champion_val" or "val" not in r:
            continue
        base = tag.rsplit("_s", 1)[0]
        rr = r["val"]["realized"]
        arms.setdefault(base, []).append(min(rr.values()))
    for base, vals in sorted(arms.items()):
        vals = np.array(vals)
        print(f"  {base:14s} n={len(vals)} mean={vals.mean():.4f} "
              f"std={vals.std():.4f} vals={np.round(vals,4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
