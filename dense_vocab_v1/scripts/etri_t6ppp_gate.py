#!/usr/bin/env python
"""T6''' — zero content 상태에서 **궤적 자체가 0인가** (배포 해상도 full forward).

기존 T6/T6'/T6''/shift는 모두 *변화량*을 재는 게이트다. "goal을 바꿔도 출력이 안
바뀐다"는 shortcut이 없다는 뜻이지만, **기존 VAD planner나 output bias가 zero
content에서 일정 궤적을 만들어 내는 경우**는 잡지 못한다. 그 상수 궤적은 goal에
불변이므로 T6'을 통과한다. 그래서 절대값 게이트가 따로 필요하다.

    T6'''  모든 현재·과거 영상 = 0, goal/command/shift는 임의
           -> max |pred_xy| < 1e-6

또한 최종 그래프가 `pred = ContentValueDecoder(...)` 인지 (`old_vad_pred + ...`가
아닌지) 코드로 확인한다. 가산 구조라면 zero content에서도 old head가 살아 있다.

측정은 **hand-assembled encode() 경로가 아니라 full forward_test**로 한다. 손으로
조립하면 남아 있는 가산 경로를 놓칠 수 있다. clip별 state reset + depth 6 replay로
배포와 동일하게 흘린다.

    python scripts/etri_t6ppp_gate.py --config <cfg> --ckpt <ckpt> --ann-file <val pkl>
"""
import argparse
import importlib
import os
import sys

import numpy as np

CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"
TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(os.path.abspath(args.config))
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = os.path.abspath(args.ann_file)
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    k2 = {(i["scene_token"], int(i["frame_idx"])): j
          for j, i in enumerate(ds.data_infos)}

    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    vi = np.load(SPLIT, allow_pickle=True)["val_idx"]
    vi = np.array([v for v in vi
                   if (scen_all[v], int(frame_all[v])) in k2
                   and int(frame_all[v]) >= args.depth * 5])
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(vi, min(args.n, len(vi)), replace=False))
    scen, frame = scen_all[sel], frame_all[sel]
    print(f"앵커 {len(sel)}  시나리오 {len(set(scen))}  깊이 {args.depth}")

    # ---- 구조 확인: 최종 그래프가 대체인가 가산인가 ----
    import inspect
    from projects.mmdet3d_plugin.VAD.VAD_head import VADHead
    src = inspect.getsource(VADHead.forward)
    goal_block = src.split("if self.goal_decoder is not None:")[-1].split("else:")[0]
    additive = any(t in goal_block for t in
                   ("ego_fut_decoder", "outputs_ego_trajs +", "+ wp", "wp +"))
    print(f"\n[구조] goal 분기에서 outputs_ego_trajs 생성식:")
    for line in goal_block.splitlines():
        if "outputs_ego_trajs" in line:
            print(f"        {line.strip()}")
    print(f"        가산(old planner 잔존) : "
          f"{'★있음 -- 즉시 수정 필요★' if additive else '없음 (순수 대체) PASS'}")

    torch.manual_seed(0)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(os.path.abspath(args.ckpt), map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    for k in [k for k, v in sd.items()
              if k in own and tuple(own[k].shape) != tuple(v.shape)]:
        sd.pop(k)
    r = model.load_state_dict(sd, strict=False)
    newk = [k for k in r.missing_keys
            if "goal_decoder" in k or "content_value" in k]
    print(f"\nckpt {os.path.basename(args.ckpt)}")
    print(f"  missing {len(r.missing_keys)}  그중 새 모듈(random init) {len(newk)}")
    for k in newk[:6]:
        print(f"    {k}")
    if len(newk) > 6:
        print(f"    ... +{len(newk)-6}")
    # STP3 collision metric은 CPU 텐서를 섞어 인덱싱해서 터진다. 여기서는 안 쓴다.
    model.compute_planner_metric_stp3 = lambda *a, **kw: {}
    mm = MMDataParallel(model.cuda(0), device_ids=[0])
    mm.eval()

    def run(zero_img, jitter_goal, jitter_shift):
        """배포 충실 replay 후 현재 프레임의 예측 궤적을 반환."""
        out_all = []
        for s, f in zip(scen, frame):
            mm.module.prev_frame_info = {"prev_bev": None, "scene_token": None,
                                        "prev_pos": 0, "prev_angle": 0}

            def feed(key):
                data = collate([ds[k2[key]]], samples_per_gpu=1)
                if zero_img:
                    t = data["img"][0].data[0]
                    data["img"][0]._data[0] = torch.zeros_like(t)
                if jitter_goal and "ego_fut_goal" in data:
                    g = data["ego_fut_goal"][0].data[0]
                    data["ego_fut_goal"][0]._data[0] = (
                        torch.from_numpy(rng.uniform(-60, 60, size=tuple(g.shape))
                                         ).to(g.dtype))
                    c = data["ego_fut_cmd"][0].data[0]
                    nc = torch.zeros_like(c)
                    flat = nc.reshape(-1, nc.shape[-1])
                    flat[:, int(rng.integers(0, nc.shape[-1]))] = 1.0
                    data["ego_fut_cmd"][0]._data[0] = flat.reshape(nc.shape)
                if jitter_shift:
                    for meta in data["img_metas"][0].data[0]:
                        cb = np.asarray(meta["can_bus"], dtype=np.float64)
                        cb[:3] += rng.uniform(-8, 8, size=3)
                        cb[-1] += rng.uniform(-0.6, 0.6)
                        meta["can_bus"] = cb
                with torch.no_grad():
                    return mm(return_loss=False, rescale=True, **data)

            for kq in range(args.depth, 0, -1):
                wk = (s, int(f) - kq * 5)
                if wk in k2:
                    feed(wk)
            out = feed((s, int(f)))
            out_all.append(out[0]["pts_bbox"]["ego_fut_preds"]
                           .detach().float().cpu().numpy())
        return np.stack(out_all)

    rows = [
        ("real  / goal 원본", False, False, False),
        ("zero  / goal 원본", True, False, False),
        ("zero  / goal·cmd 임의", True, True, False),
        ("zero  / goal·cmd·shift 임의", True, True, True),
    ]
    print(f"\n{'조건':<30}{'max|pred_xy|':>16}{'mean|pred_xy|':>16}"
          f"{'3초 |누적|':>13}")
    res = {}
    for name, zi, jg, js in rows:
        P = run(zi, jg, js)
        C = np.cumsum(P, axis=-2)
        res[name] = P
        print(f"{name:<30}{np.abs(P).max():>16.6e}{np.abs(P).mean():>16.6e}"
              f"{np.abs(C[..., -1, :]).max():>13.6e}", flush=True)

    print("\n=== 판정 ===")
    ok = True
    for name in [r[0] for r in rows[1:]]:
        m = float(np.abs(res[name]).max())
        p = m < TOL
        ok &= p
        print(f"  T6''' {name:<28} max|pred| {m:.3e}  "
              f"{'PASS' if p else '★FAIL★'}  (기준 < {TOL:g})")
    live = float(np.abs(res[rows[0][0]]).max())
    print(f"  positive control  real image  max|pred| {live:.3e}  "
          f"{'PASS (경로 살아 있음)' if live > TOL else '★FAIL (전부 죽었다)★'}")
    print(f"\n  구조 근거: content gate `img.abs().mean(1)/gate.mean().clamp_min(1e-6)`가")
    print(f"  img=0에서 0이 되고, content_value_norm(affine=False) / content_value_proj")
    print(f"  (bias=False) / value_norm(affine=False) / MultiheadAttention(bias=False) /")
    print(f"  `out(attended) - out(zeros_like)` 가 전 구간 bias-free라서 exact 0이다.")
    print(f"  주의: '영상 0'은 파이프라인 정규화 **이후** 텐서가 0이라는 뜻이고,")
    print(f"  이는 데이터셋 평균 이미지 = content 없음 기준이다 (검정색 이미지가 아니다).")
    return 0 if (ok and live > TOL) else 1


if __name__ == "__main__":
    sys.exit(main())
