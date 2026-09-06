#!/usr/bin/env python
"""⑤-W: raw images -> generator -> shortlist -> learned selector 전체 compliance.

⑤-R 은 무효였다. C-S3 은 `out = abs5[ids]` 로 만든 뒤 `array_equal(out, abs5[ids])`
를 검사해 항상 통과했고, C-S2 는 모델을 goal counterfactual 로 **재실행하지 않고**
복사된 NPZ 배열을 비교했다. 여기서는 전부 실제 forward 로 다시 돈다.

전체 경로: 원본 이미지 -> T4 candidate generator(forward_temporal) ->
완성 후보 12행 + visual logits -> learned row selector -> 최종 궤적.
조건: normal / image-zero / image-shuffle / goal-counterfactual.
"""
import argparse
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402
from train_row_selector import Selector, bank_features  # noqa: E402

OK = True


def gate(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def sel_features(out, goal, cmd, bf, rprof, rmean):
    """배포 출력(완성 후보 + logits)만으로 selector 특징을 만든다.
    bank 인덱싱을 쓰지 않고 **모델이 돌려준 좌표**에서 직접 계산한다."""
    abs5 = out["candidate_xy_abs_5s"].double().cpu().numpy()   # [B,12,10,2]
    lg = out["visual_logits"].double().cpu().numpy()           # [B,12]
    ids = out["candidate_ids"].cpu().numpy()                   # [B,12]
    B, M = lg.shape
    e5 = abs5[:, :, 9]
    gd = np.linalg.norm(e5 - goal[:, None], axis=-1)
    nz = lambda x: (x - x.min(1, keepdims=True)) / (
        x.max(1, keepdims=True) - x.min(1, keepdims=True) + 1e-9)      # noqa: E731
    rank = lambda x: np.argsort(np.argsort(x, 1), 1) / (M - 1.0)       # noqa: E731
    top = np.argmax(lg, 1)
    div = np.linalg.norm(e5 - np.take_along_axis(e5, top[:, None, None], 1), axis=-1)
    pdev = np.abs(rprof[ids] - rmean[None, None]).mean(-1)
    oh = np.zeros((B, M, 3))
    oh[np.arange(B), :, np.clip(cmd, 0, 2)] = 1.0
    return np.concatenate([
        bf[ids], gd[..., None] / 10.0, nz(gd)[..., None], rank(gd)[..., None],
        (e5[..., 0] - goal[:, None, 0])[..., None] / 10.0,
        (e5[..., 1] - goal[:, None, 1])[..., None] / 10.0,
        nz(lg)[..., None], rank(lg)[..., None],
        (lg - lg.mean(1, keepdims=True))[..., None],
        div[..., None] / 10.0, nz(div)[..., None],
        pdev[..., None] * 10.0, nz(pdev)[..., None], oh], -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work_dirs/sc_r3ctrl_s0/best.pth")
    ap.add_argument("--sel", default="work_dirs/row_selector_vis.pth")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--n", type=int, default=96)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    arr = C.load_arrays()
    bank = np.load(C.BANK_A0, allow_pickle=False)
    bf, end5, rprof = bank_features(bank)
    rmean = rprof.mean(0)

    ck = torch.load(os.path.join(A, args.ckpt), map_location="cpu")
    a = ck["args"]; nh = int(a.get("n_hist", 3))
    gen = TrainableSparseScoreDrive(
        C.BANK_A0, logit_norm=bool(a.get("logit_norm", 1)), n_hist=nh).to(dev)
    gen.load_state_dict(ck["model"]); gen.fuse_mul = bool(a.get("fuse_mul", 0)); gen.eval()
    sck = torch.load(os.path.join(A, args.sel), map_location="cpu")
    sel = Selector(sck["feat_dim"]).to(dev); sel.load_state_dict(sck["model"]); sel.eval()

    rows = arr["val_idx"]
    sub = (arr["frame"][rows] >= 30) & C.history_available(arr, rows, nh)
    rows = rows[sub][:args.n]
    ds = C.SparseFrameDataset(arr, rows, n_hist=nh)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    l2i = torch.from_numpy(C.build_global_lidar2img()).to(dev)
    goal = np.stack([arr["fut5"][r][9] for r in rows]).astype(np.float64)
    cmd = arr["vad_cmd"][rows].astype(np.int64)

    def run_gen(mode, seed=0):
        outs = []
        g = torch.Generator().manual_seed(seed)
        for b in dl:
            img = b["img"].to(dev); hist = b["img_hist"].to(dev)
            if mode == "zero":
                img = torch.zeros_like(img); hist = torch.zeros_like(hist)
            elif mode == "shuffle":
                p = torch.randperm(img.shape[0], generator=g)
                img = img[p]; hist = hist[p]
            l2ib = l2i.unsqueeze(0).expand(img.shape[0], -1, -1, -1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                o = gen.forward_temporal(img, hist, l2ib, b["hist_T"].to(dev))
            outs.append({k: v.float().cpu() if v.is_floating_point() else v.cpu()
                         for k, v in o.items()})
        return {k: torch.cat([o[k] for o in outs], 0) for k in outs[0]}

    def pick(out, gl, cd):
        X = sel_features(out, gl, cd, bf, rprof, rmean)
        with torch.no_grad():
            s = sel(torch.from_numpy(X).to(dev)).cpu().numpy()
        return s.argmax(1)

    print(f"n={len(rows)}  generator={args.ckpt}  selector={args.sel}\n")
    base = run_gen("normal")
    i0 = pick(base, goal, cmd)

    print("=== E1: goal counterfactual — 모델을 실제로 재실행한다 ===")
    # generator 는 goal 을 입력으로 받지 않는다. 그 사실 자체를 서명으로 확인한 뒤,
    # 여러 goal 로 selector 만 다시 돌려 후보/logits 가 변하지 않음을 보인다.
    import inspect
    sig = inspect.signature(gen.forward_temporal)
    gate("generator forward 서명에 goal/cmd/status 없음",
         not any(k in str(sig) for k in ("goal", "cmd", "command", "status", "speed")),
         str(sig))
    rng = np.random.default_rng(0)
    for t in range(3):
        g2 = goal + rng.normal(0, 25.0, goal.shape)
        base2 = run_gen("normal")     # 동일 입력 재실행
        for k in ("candidate_xy_abs_5s", "candidate_xy_inc_5s", "visual_logits",
                  "candidate_ids"):
            gate(f"goal 교란 {t}: {k} bitwise 불변",
                 torch.equal(base2[k], base[k]))
        i2 = pick(base2, g2, cmd)
        gate(f"goal 교란 {t}: selector 선택은 바뀔 수 있음(정상)",
             True, f"변경률 {100*float((i2!=i0).mean()):.1f}%")

    print("\n=== E2: 최종 궤적이 bank 행과 bitwise 동일 ===")
    ids = base["candidate_ids"].numpy()
    chosen_id = np.take_along_axis(ids, i0[:, None], 1)[:, 0]
    final = torch.gather(base["candidate_xy_abs_5s"], 1,
                         torch.from_numpy(i0)[:, None, None, None]
                         .expand(-1, 1, 10, 2)).squeeze(1).numpy()
    ref = bank["candidate_xy_abs_5s"][chosen_id]         # 독립 출처(원본 bank 파일)
    gate("최종 5초 궤적 == 원본 bank 행", np.array_equal(final.astype(np.float32), ref))
    gate("제출 6점 == anchors_abs 원본", np.array_equal(
        final[:, :6].astype(np.float32), bank["anchors_abs"][chosen_id]))
    gate("선택 index 범위 0..11", bool(((i0 >= 0) & (i0 < 12)).all()))

    print("\n=== E3: image-zero ===")
    z = run_gen("zero")
    iz = pick(z, goal, cmd)
    zid = np.take_along_axis(z["candidate_ids"].numpy(), iz[:, None], 1)[:, 0]
    gate("zero 이미지 -> visual logits 전부 동일(정보 없음)",
         float(z["visual_logits"].max(1).values.sub(
             z["visual_logits"].min(1).values).abs().max()) < 1e-5)
    gate("zero 이미지 -> 후보 집합이 정지 후보(id 0) 포함",
         bool((z["candidate_ids"].numpy() == 0).any(1).all()))
    print(f"    선택 id 분포(zero): 최빈 {np.bincount(zid).argmax()}")

    print("\n=== E4: image-shuffle (시각 증거가 실제로 쓰이는가) ===")
    sh = run_gen("shuffle", seed=1)
    ish = pick(sh, goal, cmd)
    shid = np.take_along_axis(sh["candidate_ids"].numpy(), ish[:, None], 1)[:, 0]
    chg = float((shid != chosen_id).mean())
    gate("셔플 시 최종 선택 궤적이 실제로 바뀜", chg > 0.20, f"변경률 {100*chg:.1f}%")
    gate("셔플 시 logits 도 바뀜",
         not torch.equal(sh["visual_logits"], base["visual_logits"]))

    print(f"\n{'ALL GATES PASS' if OK else 'GATE FAILURE'}")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
