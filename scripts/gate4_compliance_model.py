#!/usr/bin/env python
"""④ Compliance suite — model 파트 (§12), 실제 champion forward.
  C2  goal/command counterfactual: 같은 image, goal/cmd 100회 변경 -> decoder logits bitwise 동일
  C7  reset/depth audit: clip 시작 state reset 이 이전 시나리오 오염을 격리(동일 warmup->동일 logits),
      history depth/간격 고정 확인

    python gate4_compliance_model.py <config> <ckpt> [--n-goals 100]
"""
import os, sys, json
import numpy as np

SD = "/NHNHOME/data/sukim/adcl/scripts"
sys.path.insert(0, SD)
import etri_vad_eval as E
MAN = "/NHNHOME/data/sukim/adcl/logs/compliance_manifest.json"


def set_goal(data, val):
    """collated data 의 ego_fut_goal 텐서를 val([2]) 로 덮어쓴다. 없으면 False."""
    import torch
    if "ego_fut_goal" not in data:
        return False
    obj = data["ego_fut_goal"]
    cur = obj
    while not torch.is_tensor(cur):
        if hasattr(cur, "data"):
            cur = cur.data
        elif isinstance(cur, (list, tuple)):
            cur = cur[0]
        else:
            return False
    cur.reshape(-1)[:2] = torch.tensor(val, dtype=cur.dtype, device=cur.device)
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--ann-file", default="/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl")
    ap.add_argument("--split", default="val_idx"); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n-goals", type=int, default=100); ap.add_argument("--gpu", type=int, default=0)
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
    holder = []
    gd.register_forward_hook(lambda m, i, o: holder.append(o[1].detach().float().cpu().numpy()[0]))
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j for j, i in enumerate(ds.data_infos)}

    def run_main(r, goal=None):
        E.reset_stream(model.module)
        for k in range(args.depth, 0, -1):
            wk = (scen[r], int(frame[r]) - k * 5)
            if wk not in key2ds: continue
            dw = collate([ds[key2ds[wk]]], samples_per_gpu=1)
            with torch.no_grad(): model(return_loss=False, rescale=True, **dw)
        data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
        goal_set = set_goal(data, goal) if goal is not None else None
        holder.clear()
        with torch.no_grad(): model(return_loss=False, rescale=True, **data)
        return holder[-1], goal_set

    scens = sorted(set(scen))
    samples = []
    for s in scens:
        m = scen == s; samples.append(int(np.where(m)[0][np.argsort(frame[m])][len(np.where(m)[0]) // 2]))
        if len(samples) >= 3: break

    R = {}
    rng = np.random.RandomState(0)

    # ---------------- C2 goal/command counterfactual ----------------
    c2_ok = True; goal_was_set = False; spreads = []
    for r in samples:
        base, gs = run_main(r, goal=None)
        for _ in range(args.n_goals):
            g = rng.randn(2) * 10
            lg, gs2 = run_main(r, goal=g)
            goal_was_set = goal_was_set or bool(gs2)
            if not np.array_equal(lg, base):
                c2_ok = False
        spreads.append(float(base.max() - base.min()))
    R["C2_goal_override_reached_pipeline"] = bool(goal_was_set)   # True=goal 을 실제 주입했음
    R["C2_logits_bitwise_invariant_to_goal"] = bool(c2_ok)
    R["C2_n_goals_per_sample"] = args.n_goals
    R["C2_samples"] = len(samples)

    # ---------------- C7 reset/depth audit ----------------
    r = samples[0]
    logA, _ = run_main(r)                       # reset+warmup depth6 -> main
    # 오염: 다른 시나리오 프레임을 reset 없이 흘림
    other = samples[-1]
    od = collate([ds[int(ds_idx[other])]], samples_per_gpu=1)
    with torch.no_grad(): model(return_loss=False, rescale=True, **od)
    logB, _ = run_main(r)                       # 다시 reset+warmup -> main
    R["C7_reset_isolates_prior_scenario"] = bool(np.array_equal(logA, logB))
    # depth 의존성: reset 후 warmup 0 vs depth6 는 달라야(=temporal state 실사용)
    E.reset_stream(model.module)
    d0 = collate([ds[int(ds_idx[r])]], samples_per_gpu=1); holder.clear()
    with torch.no_grad(): model(return_loss=False, rescale=True, **d0)
    log_nodepth = holder[-1]
    R["C7_depth_matters_state_used"] = bool(not np.array_equal(log_nodepth, logA))
    R["C7_depth"] = args.depth; R["C7_history_interval_frames"] = 5

    man = json.load(open(MAN)) if os.path.exists(MAN) else {}
    man["model"] = R
    json.dump(man, open(MAN, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(R, indent=2, ensure_ascii=False))
    gates = [R["C2_logits_bitwise_invariant_to_goal"], R["C7_reset_isolates_prior_scenario"],
             R["C7_depth_matters_state_used"]]
    print("\nMODEL_COMPLIANCE_PASS" if all(gates) else "MODEL_COMPLIANCE_FAIL", gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
