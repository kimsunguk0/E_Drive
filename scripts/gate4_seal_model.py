#!/usr/bin/env python
"""④-A compliance 봉합 (model 파트, 실제 champion forward).
  C3  full path: VAD forward -> decoder.build_candidates -> selector -> 6-point,
      6-point == candidate_inc row bitwise (전 sample). + predict_candidates smoke.
  C4  정식 표: normal/zero/constant/clip_shuffle/current_shuffle/history_shuffle/camera_perm
      × {realized L2, top1 ID overlap, top12 Jaccard, rank correlation}
  C6  runtime: transformer.use_can_bus==False, decoder forward 입력=(bev,image_evidence,B),
      img_metas whitelist(goal/cmd/status 없음)
  C7  clip 당 정확히 7 forward(depth6+main)

    python gate4_seal_model.py <config> <ckpt> [--limit-samples 30]
"""
import argparse, os, sys, json
import numpy as np
SD = "/NHNHOME/data/sukim/adcl/scripts"
sys.path.insert(0, SD)
import etri_vad_eval as E
import scoredrive_api as api
from scipy.stats import spearmanr

A = "/NHNHOME/data/sukim/adcl"
MAN = A + "/logs/compliance_manifest.json"
DEP = A + "/data/etri/bank_A0_deploy.npz"
E5 = "/tmp/pm97/data/etri/ego_cache_5s.npz"
CW = np.array([11, 11, 5, 5, 2, 2], np.float64) / 36.0


def find_img(data):
    import torch
    cur = data["img"]
    while not torch.is_tensor(cur):
        cur = cur.data if hasattr(cur, "data") else cur[0]
    return cur


def perturb(t, mode, rng):
    import torch
    if mode in ("normal", "clip_shuffle", "history_shuffle_frame", "current_normal"):
        return t
    if mode == "zero":
        return t * 0.0
    if mode == "constant":
        return t * 0.0 + float(t.mean())
    if mode == "pixel_shuffle":
        flat = t.reshape(-1); perm = rng.permutation(flat.shape[0])
        return flat[torch.from_numpy(perm).to(flat.device)].reshape(t.shape)
    if mode == "camera_perm":
        # [B,N,C,H,W] N=camera 축 순열
        cp = rng.permutation(t.shape[1])
        return t[:, torch.from_numpy(cp).to(t.device)]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--ann-file", default="/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl")
    ap.add_argument("--split", default="val_idx"); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--limit-samples", type=int, default=30); ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config); args.checkpoint = os.path.abspath(args.checkpoint)
    repo = args.config.split("/projects/")[0]; os.chdir(repo); sys.path.insert(0, repo)
    import importlib, torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    cfg.data.test.test_mode = True; cfg.data.test.ann_file = args.ann_file
    cfg.data.test.pop("samples_per_gpu", None); cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    scen, frame, ds_idx, cache_idx = E.build_index(args.split, None, ds=ds)
    torch.manual_seed(0); np.random.seed(0)
    model, _ = E.load_model(cfg, args.checkpoint)
    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu]); model.eval()
    gd = model.module.pts_bbox_head.goal_decoder
    gd.attach_deploy_bank(DEP)
    anchors = gd.anchors_abs.detach().cpu().numpy().astype(np.float64)
    dep = np.load(DEP, allow_pickle=True)
    abs5 = dep["candidate_xy_abs_5s"]; inc5 = dep["candidate_xy_inc_5s"]
    ids = dep["candidate_ids"]; adist = dep["anchor_dist"]; tau = float(dep["nms_tau"])
    fut5 = np.load(E5, allow_pickle=True)["fut5"].astype(np.float64)

    holder = []; arg_holder = []
    gd.register_forward_hook(lambda m, i, o: holder.append(o[1].detach().float().cpu().numpy()[0]))
    gd.register_forward_pre_hook(lambda m, i: arg_holder.append(len(i)))
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j for j, i in enumerate(ds.data_infos)}

    def stream(r, mode, rng):
        E.reset_stream(model.module)
        nfwd = 0
        warm = list(range(args.depth, 0, -1))
        if mode == "clip_shuffle":
            warm = list(rng.permutation(warm))
        for k in warm:
            wk = (scen[r], int(frame[r]) - k * 5)
            if wk not in key2ds: continue
            dw = collate([ds[key2ds[wk]]], samples_per_gpu=1)
            pm = {"zero": "zero", "constant": "constant",
                  "history_shuffle": "pixel_shuffle", "camera_perm": "camera_perm"}.get(mode, "normal")
            t = find_img(dw); t.copy_(perturb(t.clone(), pm, rng))
            with torch.no_grad(): model(return_loss=False, rescale=True, **dw)
            nfwd += 1
        data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
        pm = {"zero": "zero", "constant": "constant", "current_shuffle": "pixel_shuffle",
              "camera_perm": "camera_perm"}.get(mode, "normal")
        t = find_img(data); t.copy_(perturb(t.clone(), pm, rng))
        holder.clear()
        with torch.no_grad(): model(return_loss=False, rescale=True, **data)
        nfwd += 1
        return holder[-1], nfwd, data

    scens = sorted(set(scen)); rows = []
    for s in scens:
        m = (scen == s) & (frame >= 30)          # full-history 프레임만(공식 clip = 7 frame)
        w = np.where(m)[0]
        rows += list(w[np.argsort(frame[w])])
        if len(rows) >= args.limit_samples: break
    rows = rows[:args.limit_samples]

    def realized_and_ids(lg, r):
        gt5 = fut5[cache_idx[r]]; gt3 = gt5[:6]; goal = gt5[9]
        cand = api.build_candidates_np(lg[None], abs5, inc5, ids, adist, tau)
        cb = api.slice_batch(cand, 0)
        k = api.select_index_np(cb, goal, 0.1); cid = int(cb["candidate_ids"][k])
        rl = float((CW * np.linalg.norm(anchors[cid] - gt3, axis=-1)).sum())
        p6 = cb["candidate_xy_inc_5s"][k, :6]
        row_ok = np.array_equal(p6, inc5[cid, :6])
        return rl, cb["candidate_ids"].copy(), int(np.argmax(lg)), row_ok

    modes = ["normal", "zero", "constant", "clip_shuffle", "current_shuffle",
             "history_shuffle", "camera_perm"]
    R = {"C4_table": {}, "C7_forwards_per_clip": []}
    per = {mm: {"logits": {}, "realized": [], "cids": {}, "top1": {}, "rowok": True} for mm in modes}
    fwd_counts = []
    for i, r in enumerate(rows):
        for mm in modes:
            lg, nf, data = stream(r, mm, np.random.RandomState(2000 + i))
            if mm == "normal": fwd_counts.append(nf)
            rl, cids, t1, row_ok = realized_and_ids(lg, r)
            per[mm]["logits"][r] = lg; per[mm]["realized"].append(rl)
            per[mm]["cids"][r] = cids; per[mm]["top1"][r] = t1
            per[mm]["rowok"] = per[mm]["rowok"] and bool(row_ok)
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(rows)}", flush=True)

    def wmean(a): return float(np.mean(a))
    for mm in modes:
        t1_ov = np.mean([per[mm]["top1"][r] == per["normal"]["top1"][r] for r in rows])
        jac = np.mean([len(set(per[mm]["cids"][r]) & set(per["normal"]["cids"][r])) /
                       len(set(per[mm]["cids"][r]) | set(per["normal"]["cids"][r])) for r in rows])
        rc = np.mean([spearmanr(per[mm]["logits"][r], per["normal"]["logits"][r]).correlation
                      for r in rows])
        R["C4_table"][mm] = {"realized_L2": round(wmean(per[mm]["realized"]), 4),
                             "top1_ID_overlap": round(float(t1_ov), 3),
                             "top12_Jaccard": round(float(jac), 3),
                             "rank_corr": round(float(rc), 3)}
    R["C3_fullpath_row_identity"] = bool(per["normal"]["rowok"])
    R["C7_forwards_per_clip"] = sorted(set(fwd_counts))
    R["C7_exactly_7"] = bool(set(fwd_counts) == {args.depth + 1})

    # ---- predict_candidates smoke (goal-free API 실제 실행) ----
    try:
        r = rows[0]
        E.reset_stream(model.module)
        for k in range(args.depth, 0, -1):
            wk = (scen[r], int(frame[r]) - k * 5)
            if wk in key2ds:
                dw = collate([ds[key2ds[wk]]], samples_per_gpu=1)
                with torch.no_grad(): model(return_loss=False, rescale=True, **dw)
        data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
        img = data["img"]; imv = img
        while not torch.is_tensor(imv): imv = imv.data if hasattr(imv, "data") else imv[0]
        im = data["img_metas"]
        while not isinstance(im, list) or (im and not isinstance(im[0], dict)):
            im = im.data if hasattr(im, "data") else im[0]
        imv = imv.cuda(args.gpu)                  # 모델 device 로 이동
        with torch.no_grad():
            out = model.module.predict_candidates(imv, im, prev_bev=None)
        R["predict_candidates_smoke"] = {
            "ran": True, "keys": sorted(out.keys()),
            "shape_abs": list(out["candidate_xy_abs_5s"].shape),
            "n12": bool(out["candidate_ids"].shape[1] == 12)}
    except Exception as e:
        import traceback
        R["predict_candidates_smoke"] = {"ran": False, "err": str(e)[:200],
                                         "tb": traceback.format_exc()[-500:]}

    # ---- C6 runtime ----
    tr = model.module.pts_bbox_head.transformer
    R["C6_use_can_bus_False"] = bool(getattr(tr, "use_can_bus", True) is False)
    R["C6_decoder_forward_nargs"] = sorted(set(arg_holder))  # (bev_key,image_value,batch_size)=3
    im0 = ds.data_infos[0] if hasattr(ds, "data_infos") else {}
    meta_keys_ban = [k for k in ["ego_fut_goal", "vad_cmd", "ego_fut_cmd", "ego_his_trajs",
                                 "ego_lcf_feat", "command", "status"]
                     if k in (im0 if isinstance(im0, dict) else {})]
    R["C6_img_metas_banned_keys"] = meta_keys_ban

    man = json.load(open(MAN)) if os.path.exists(MAN) else {}
    man["seal_model"] = R
    json.dump(man, open(MAN, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(R, indent=2, ensure_ascii=False))
    gates = [R["C3_fullpath_row_identity"], R["C7_exactly_7"],
             R["predict_candidates_smoke"].get("ran", False),
             R["C6_use_can_bus_False"], R["C6_decoder_forward_nargs"] == [3],
             len(R["C6_img_metas_banned_keys"]) == 0]
    print("\nSEAL_MODEL_PASS" if all(gates) else "SEAL_MODEL_FAIL", gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
