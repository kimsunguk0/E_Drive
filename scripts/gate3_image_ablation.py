#!/usr/bin/env python
"""Gate-③ item 11 (C4 선행): raw 영상 ablation.

챔피언 실제 forward 를 normal / zero / constant / pixel-shuffle 4조건으로 돌려
main-frame vocab logits 를 캡처하고, in-model build_candidates + 외부 selector 를 통과시켜
'영상이 후보를 구동하고 goal 은 index 만 바꾼다'를 증명한다. 스트림 전체(warmup 포함)에
같은 perturbation 을 적용한다(temporal-BEV 오염 방지).

보고: 조건별 top-1 anchor, logit spread(max-min), 최종 선택 candidate id,
      goal 20종을 바꿔도 candidate/logit/id 불변 + 선택 path 는 항상 후보 row,
      zero/constant 에서 goal 을 바꿔도 최종 경로가 안 바뀌는지(=goal-only 아님).

    python gate3_image_ablation.py <config> <ckpt> [--limit-samples 40]
"""
import argparse, os, sys, numpy as np

SD = "/NHNHOME/data/sukim/adcl/scripts"
sys.path.insert(0, SD)
import etri_vad_eval as E
import scoredrive_api as api


def find_img(data):
    """collated dict 에서 이미지 텐서를 찾아 (컨테이너, setter) 반환."""
    import torch
    obj = data["img"]
    # DataContainer or list
    cur = obj
    while not torch.is_tensor(cur):
        if hasattr(cur, "data"):
            cur = cur.data
        elif isinstance(cur, (list, tuple)):
            cur = cur[0]
        else:
            raise RuntimeError(f"img tensor 못 찾음: {type(cur)}")
    return cur


def perturb(t, mode, rng):
    if mode == "normal":
        return t
    if mode == "zero":
        return t * 0.0
    if mode == "constant":
        return t * 0.0 + float(t.mean())
    if mode == "shuffle":
        flat = t.reshape(-1)
        perm = rng.permutation(flat.shape[0])
        import torch
        return flat[torch.from_numpy(perm).to(flat.device)].reshape(t.shape)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--ann-file", default="/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl")
    ap.add_argument("--split", default="val_idx")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--limit-samples", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
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
    K = gd.anchors_abs.shape[0]

    # deploy bank
    dep = np.load("/NHNHOME/data/sukim/adcl/data/etri/bank_A0_deploy.npz", allow_pickle=True)
    abs5 = dep["candidate_xy_abs_5s"]; inc5 = dep["candidate_xy_inc_5s"]
    ids = dep["candidate_ids"]; adist = dep["anchor_dist"]; tau = float(dep["nms_tau"])

    holder = []
    gd.register_forward_hook(lambda m, i, o: holder.append(o[1].detach().float().cpu().numpy()[0]))

    key2ds = {(i["scene_token"], int(i["frame_idx"])): j for j, i in enumerate(ds.data_infos)}
    scens = sorted(set(scen))
    # 표본 모으기
    rows_all = []
    for s in scens:
        m = scen == s; order = np.argsort(frame[m]); rows_all += list(np.where(m)[0][order])
        if len(rows_all) >= args.limit_samples:
            break
    rows_all = rows_all[:args.limit_samples]

    def run_cond(r, mode, rng):
        E.reset_stream(model.module)
        for k in range(args.depth, 0, -1):
            wk = (scen[r], int(frame[r]) - k * 5)
            if wk not in key2ds:
                continue
            dw = collate([ds[key2ds[wk]]], samples_per_gpu=1)
            t = find_img(dw); t.copy_(perturb(t.clone(), mode, rng))
            with torch.no_grad():
                model(return_loss=False, rescale=True, **dw)
        data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
        t = find_img(data); t.copy_(perturb(t.clone(), mode, rng))
        holder.clear()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        return holder[-1]

    modes = ["normal", "zero", "constant", "shuffle"]
    rng = np.random.RandomState(0)
    goals = rng.randn(20, 2) * 8
    stats = {mm: dict(top1=[], spread=[], selid=[], goalvary=[]) for mm in modes}
    normal_logits = {}
    for i, r in enumerate(rows_all):
        for mm in modes:
            lg = run_cond(r, mm, np.random.RandomState(1000 + i))  # shuffle seed per sample
            cand = api.build_candidates_np(lg[None], abs5, inc5, ids, adist, tau)
            cb = api.slice_batch(cand, 0)
            # goal 20종 -> 최종 선택 id 집합 (candidate/logit/id 불변 확인)
            base_ids = cb["candidate_ids"].copy(); base_vl = cb["visual_logits"].copy()
            sel_ids = set()
            inv = True
            for g in goals:
                _, j, cid = api.select_path_np(cb, g, 0.1)
                sel_ids.add(cid)
            if not (np.array_equal(base_ids, cb["candidate_ids"]) and
                    np.array_equal(base_vl, cb["visual_logits"])):
                inv = False
            stats[mm]["top1"].append(int(np.argmax(lg)))
            stats[mm]["spread"].append(float(lg.max() - lg.min()))
            stats[mm]["selid"].append(sorted(sel_ids))
            stats[mm]["goalvary"].append(len(sel_ids))
            stats[mm]["_inv"] = stats[mm].get("_inv", True) and inv
            if mm == "normal":
                normal_logits[r] = lg
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows_all)}", flush=True)

    print("\n===== Gate-③ raw-image ablation =====")
    print(f"samples={len(rows_all)}  goals per sample=20")
    for mm in modes:
        top1 = np.array(stats[mm]["top1"]); spread = np.array(stats[mm]["spread"])
        gv = np.array(stats[mm]["goalvary"])
        frac_stop = float((top1 == 0).mean())
        # zero/constant: goal 을 20종 바꿔도 최종 경로가 안 바뀌어야(=goal-only 아님)
        goal_invariant_final = float((gv == 1).mean())
        print(f"[{mm:8s}] top1=0(stop) {frac_stop*100:5.1f}%  "
              f"logit_spread p50={np.percentile(spread,50):.4f} "
              f"p99={np.percentile(spread,99):.4f}  "
              f"goal-changes-final-id: mean_distinct={gv.mean():.2f} "
              f"(final invariant to goal in {goal_invariant_final*100:.0f}% samples)  "
              f"candidates-invariant-to-goal={stats[mm]['_inv']}")
    # normal vs zero: 선택이 실제로 영상에 의존하는지
    same = np.mean([stats['normal']['top1'][i] == stats['zero']['top1'][i]
                    for i in range(len(rows_all))])
    print(f"\ntop1 identical(normal vs zero) = {same*100:.1f}%  "
          f"(낮을수록 영상 의존 증거)")
    print("\nINTERP: zero/constant 에서 top1 이 stop(0)로 몰리고 goal 을 바꿔도 최종이 "
          "안 바뀌면 => 파이프라인이 영상 구동이며 goal-only 로 퇴행하지 않음(규정 준수).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
