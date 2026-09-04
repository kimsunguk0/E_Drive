#!/usr/bin/env python
"""VAD 체크포인트를 **우리 val**에서 챌린지 L2로 평가한다 + legal ablation.

왜 이 스크립트가 필요한가
------------------------
지금까지 우리가 가진 모든 숫자(PriorNet 0.0609, goal 등가속 0.1296, 등속 CV 0.3675)는
ego/goal 기반이다. **카메라 모델을 우리 val에서 측정한 적이 한 번도 없다.** 그리고
`etri_table.py`는 `pred[N,6,2]` 배열을 받는데 그걸 체크포인트에서 만들어주는 코드가
없었다. 이후의 모든 판정 기준 -- head tournament의 "weighted L2 >= 0.005 개선",
legal ablation의 "+0.03 absolute", 제출 #1의 `dLB = LB - Val` -- 이 전부 이 코드의
존재를 전제한다.

추론 방식: `tools/etri_test_submit.py`와 **동일**하게 시나리오별 streaming이다.
  * 시나리오 진입 시 `prev_frame_info` 리셋
  * 앵커를 frame 순서로 순차 forward (val 앵커는 frame 0,5,...,295 = 0.5초 간격)
  * `ego_fut_preds[cmd.argmax()]` 선택 후 **cumsum** (절대규칙 11: 출력=증분, 채점=누적)

test clip은 7프레임(-3.0s ~ 0)을 흘려보내고 마지막만 채점한다. val은 앵커 60개를
연속으로 흘리므로 frame >= 30 앵커가 그 조건(선행 6프레임)과 같다. 그래서 전체
2,280과 `frame>=30` 부분집합을 **둘 다** 보고한다.

    # zero-shot nuScenes 가중치를 ETRI val에
    python scripts/etri_vad_eval.py CFG ckpt/vad/VAD_tiny.pth
    # 스모크 (시나리오 2개)
    python scripts/etri_vad_eval.py CFG CKPT --limit 2
    # legal ablation
    python scripts/etri_vad_eval.py CFG CKPT --ablation image-zero
    python scripts/etri_vad_eval.py CFG CKPT --ablation can-bus-random
    python scripts/etri_vad_eval.py CFG CKPT --ablation clip-shuffle
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge")
REPO = os.path.normpath(REPO)
ADCL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"

ABLATIONS = ("none", "image-zero", "can-bus-random", "clip-shuffle",
             "current-only", "history-shuffle", "temporal-only", "goal-shuffle",
             "zero-current")
# `goal-shuffle`: 5초 goal만 다른 앵커의 것으로 바꾼다 (영상·can_bus·cmd는 원본).
# 형님 §11-3 판정 조건 "goal shuffle 시 trajectory 방향이 바뀜" 을 재는 것이다.
# 바뀌지 않으면 goal 경로가 죽은 것이고, 성능이 크게 나빠지면 goal에 과의존한다는
# 뜻이다 -- 둘 다 원하는 상태가 아니다.
# `temporal-only`: prev_bev만 버리고 **ego 변위(shift)는 살린다**.
#
# `current-only`로는 순수 temporal 효과를 못 잰다. 이유가 둘이다.
#   (1) forward_test가 prev_bev=None일 때 can_bus[:3]/[-1]을 0으로 만든다 (VAD.py:310).
#   (2) 더 근본적으로 modules/encoder.py:186-188 의 **상류 의도적 버그**:
#         shift_ref_2d = ref_2d      # .clone() 이어야 하는데 alias다
#         shift_ref_2d += shift[...] # in-place -> ref_2d 자체가 shift된다
#       그래서 prev_bev=None 분기의 `stack([ref_2d, ref_2d])`도 이미 shift된 값을 쓴다.
#       `use_can_bus=False`와 무관하게 ego 변위가 temporal self-attention의 reference
#       point로 항상 들어간다. 실측: prev_bev=None에서 can_bus 원본 vs 0 처리
#       출력 최대차 2.218 (민감도 대조군 깊이6은 6.714).
# 이 ablation은 get_bev_features 직전에 can_bus에 **정상 0.5초 delta**를 써넣어
# shift를 살린 채 prev_bev만 없앤다.

# `history-shuffle`은 streaming 평가에서 정의가 애매하다. prev_bev는 같은 배치의 큐가
# 아니라 **이전 앵커의 forward**에서 만들어지므로, "과거만 교체"를 하려면 이전 앵커를
# donor 영상으로 돌린 뒤 현재 앵커만 진짜 영상으로 돌려야 하는데 그 둘이 같은 forward
# 안에서 일어난다. 대신 `current-only`(앵커마다 prev_bev 리셋)로 temporal 기여를
# 직접 측정한다 -- 같은 질문에 더 깔끔하게 답한다.


def reset_stream(model):
    """etri_test_submit.py:17 과 동일."""
    model.prev_frame_info = {
        "prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}


def build_index(split_key, scenarios=None, ds=None):
    """앵커 -> (scenario, frame, dataset index, ego_cache 행). 매핑을 검증한다.

    `scenarios`를 주면 split을 무시하고 그 시나리오들의 **2Hz 앵커**(frame 0,5,...,295)를
    쓴다. val이 정확히 그 구조(시나리오당 60개)이므로 8-scene train 평가와 val 평가가
    같은 앵커 격자에서 비교된다.

    ★★ 반드시 `ds.data_infos`로 인덱스를 만들어야 한다. 원시 pkl 순서로 만들면 안 된다.
    mmdet3d `NuScenesDataset.load_annotations`가
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    로 **timestamp 정렬**을 한다. 우리 ETRI pkl은 시나리오별로 이어붙인 순서라
    timestamp 정렬이 아니다 -- 실측 **112,800개 중 28,500개(25.3%)가 다른 프레임**이다.
    (예: idx 14700 에서 pkl은 20260113-102709 f0, dataset은 20260113-92505 f0)

    이 버그로 평가가 **A 시나리오의 영상을 넣고 B 시나리오의 GT로 채점**해 왔다.
    산술도 맞는다: 0.25 x ~13 + 0.75 x 0.4 ~= 3.55 = 관측된 3.53 정체 구간.
    학습(overfit8.pkl, 8 시나리오)은 순서가 우연히 일치해 영향이 없었고, 그래서
    "학습 loss 0.32 vs 평가 3.53" 이라는 11배 괴리로 나타났다.
    공식 제출 tools/etri_test_submit.py 는 처음부터 끝까지 dataset 순서만 쓰므로 무사하다.
    """
    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])
    all_scen = scen_all[d["scen_idx"]]
    all_frame = d["frame"].astype(int)
    if scenarios:
        want = set(scenarios)
        idx = np.where(np.isin(all_scen, list(want)) & (all_frame % 5 == 0))[0]
        idx = idx[np.lexsort((all_frame[idx], all_scen[idx]))]
    else:
        sp = np.load(SPLIT, allow_pickle=True)
        idx = sp[split_key]
    scen = all_scen[idx]
    frame = all_frame[idx]
    assert ds is not None, "build_index는 dataset을 받아야 한다 (pkl 순서 사용 금지)"
    infos = ds.data_infos
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j
              for j, i in enumerate(infos)}
    ds_idx = np.array([key2ds[(s, int(f))] for s, f in zip(scen, frame)])
    # 역검증: 매핑이 틀리면 GT와 예측이 어긋나는데 손실은 정상으로 보인다.
    for j, s, f in zip(ds_idx, scen, frame):
        assert infos[j]["scene_token"] == s and int(infos[j]["frame_idx"]) == int(f)
    return scen, frame, ds_idx, idx


def load_model(cfg, ckpt_path):
    import torch
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    own = model.state_dict()
    # shape가 다른 키는 미리 뺀다. strict=False만으로는 shape 불일치에서 죽는다.
    drop = [k for k, v in sd.items()
            if k in own and tuple(own[k].shape) != tuple(v.shape)]
    for k in drop:
        sd.pop(k)
    res = model.load_state_dict(sd, strict=False)
    print(f"  ckpt {os.path.basename(ckpt_path)}")
    print(f"    shape 불일치로 제외 {len(drop)}: "
          + ", ".join(k.split('.')[-3:][0] + '…' for k in drop[:3])
          + (f" (+{len(drop)-3})" if len(drop) > 3 else ""))
    print(f"    missing {len(res.missing_keys)}  unexpected {len(res.unexpected_keys)}")
    if res.missing_keys[:4]:
        for k in res.missing_keys[:4]:
            print(f"      missing: {k}")
    # planner 관련 키가 missing이면 궤적이 랜덤이라는 뜻 -> 명시적으로 경고
    plan = [k for k in res.missing_keys if "ego_fut_decoder" in k]
    if plan:
        print(f"    !! planner 가중치 {len(plan)}개 missing -- 궤적이 랜덤이다")
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    return model, dict(dropped=drop, missing=res.missing_keys,
                       unexpected=res.unexpected_keys)


def apply_ablation(data, mode, donor, rng):
    """collate된 dict을 제자리에서 교란한다.

    * image-zero      : 영상 전부 0. goal/cmd/정렬 유지. moving에서 크게 붕괴해야 정상.
    * can-bus-random  : can_bus[7:16] (accel/yaw_rate/velocity) 교란.
                        use_can_bus=False이고 다른 경로에서 안 쓰면 출력이 **정확히** 같아야 한다.
    * clip-shuffle    : 다른 시나리오의 영상 + 그 시나리오의 lidar2img로 교체.
                        goal/cmd/can_bus는 원본 유지 = "같은 지시, 다른 장면".
                        캘리브레이션도 함께 바뀌므로 교란이 다소 강하다(주의: 통과가 쉬워진다).
    """
    if mode in ("image-zero", "zero-current"):
        # zero-current: 영상 0 + prev_bev 리셋. zero-image x temporal on/off 2x2의
        # 마지막 칸이다. 이 값으로
        #     zero-image temporal effect = L2(zero, current-only) - L2(zero, temporal)
        # 를 직접 분리해, "영상 content 없이도 temporal positional memory가 강한
        # trajectory source인가"를 확인한다.
        data["img"][0].data[0].zero_()
    elif mode == "can-bus-random":
        for im in data["img_metas"][0].data[0]:
            cb = np.asarray(im["can_bus"])
            cb[7:16] = rng.normal(0, 10, 9)
    elif mode == "goal-shuffle":
        if donor is None or "ego_fut_goal" not in data:
            return
        data["ego_fut_goal"][0].data[0].copy_(
            donor["ego_fut_goal"][0].data[0])
    elif mode == "clip-shuffle":
        if donor is None:
            return
        data["img"][0].data[0] = donor["img"][0].data[0].clone()
        for im, dm in zip(data["img_metas"][0].data[0],
                          donor["img_metas"][0].data[0]):
            im["lidar2img"] = dm["lidar2img"]
            im["filename"] = dm["filename"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--ann-file", default=ANN)
    ap.add_argument("--split", default="val_idx",
                    choices=("val_idx", "minival_idx"))
    ap.add_argument("--ablation", default="none", choices=ABLATIONS)
    ap.add_argument("--limit", type=int, default=0,
                    help="시나리오 N개만 (스모크)")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", default="",
                    help="쉼표 구분 또는 pkl 경로. 주면 --split 무시. "
                         "그 시나리오의 2Hz 앵커를 쓴다 (8-scene overfit 평가용)")
    ap.add_argument("--depth", type=int, default=0,
                    help="배포 충실 누적 깊이. 앵커마다 리셋 후 선행 K프레임(0.5초 간격)만 "
                         "흘린다. 공식 제출은 clip 7프레임이므로 **6이 배포값**이다. "
                         "0=시나리오 전체 연속 streaming(깊이 최대 59, 배포와 다름).")
    ap.add_argument("--repo", default=REPO,
                    help="플러그인을 임포트할 소스 트리. 기본은 메인 트리다. "
                         "동결본이나 worktree의 ckpt를 평가할 때는 **그 트리를 줘야 한다** "
                         "-- 안 그러면 다른 모델 코드로 ckpt를 읽는 사고가 난다. "
                         "실제로 v2a ckpt를 메인 트리로 읽어 "
                         "`VAD.__init__() got an unexpected keyword argument "
                         "'expose_image_evidence'`로 죽었다.")
    args = ap.parse_args()
    scen_list = None
    if args.scenarios:
        if args.scenarios.endswith(".pkl"):
            import pickle as _pk
            scen_list = sorted({i["scene_token"] for i in
                                _pk.load(open(args.scenarios, "rb"))["infos"]})
        else:
            scen_list = [x.strip() for x in args.scenarios.split(",") if x.strip()]

    # plugin import가 cwd=REPO를 요구하므로 chdir 전에 경로를 절대화한다.
    args.config = os.path.abspath(args.config)
    args.checkpoint = os.path.abspath(args.checkpoint)
    if args.out:
        args.out = os.path.abspath(args.out)
    repo = os.path.abspath(args.repo)
    os.chdir(repo)
    sys.path.insert(0, repo)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    import torch
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    print(f"dataset {type(ds).__name__} {len(ds):,}  "
          f"sample_interval {getattr(ds, 'sample_interval', '?')}")

    scen, frame, ds_idx, cache_idx = build_index(args.split, scen_list, ds=ds)
    # pkl 순서로 인덱싱하던 과거 버그를 다시 못 만들게 크기를 남긴다
    import pickle as _pk0
    _raw = _pk0.load(open(args.ann_file, "rb"))["infos"]
    _n_mis = sum(1 for a, b in zip(ds.data_infos, _raw)
                 if a["scene_token"] != b["scene_token"]
                 or int(a["frame_idx"]) != int(b["frame_idx"]))
    print(f"dataset 순서 vs pkl 순서 불일치 {_n_mis:,}/{len(_raw):,} "
          f"({100*_n_mis/max(len(_raw),1):.1f}%)  -> dataset 순서로 인덱싱함")
    scens = sorted(set(scen))
    if args.limit:
        scens = scens[:args.limit]
    keep = np.isin(scen, scens)
    print(f"{args.split}: 앵커 {keep.sum():,} / 시나리오 {len(scens)}  "
          f"frame {frame[keep].min()}~{frame[keep].max()}")
    print(f"ablation: {args.ablation}")

    # ★ 시드 고정이 필수다. ckpt에 없는 12개 레이어(cls_branches/map_cls_branches의
    # 최종 로짓)가 랜덤 초기화되고, 그게 어떤 agent/map query가 높은 점수를 받는지를
    # 바꿔 planner의 cross-attention까지 흔든다. 시드를 안 잡으면 **동일 설정 재실행에서
    # 가중 L2가 +-0.007 흔들려**(실측 11.0569 vs 11.0500) head tournament 채택 기준
    # 0.005보다 노이즈가 커진다. 한 프로세스 안에서는 완전히 결정적임을
    # scripts/etri_determinism.py 로 확인했다 (|A-B| = 0.0).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    model, load_info = load_model(cfg, args.checkpoint)
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()

    assert not (args.ablation == "history-shuffle" and not args.depth), \
        "history-shuffle은 --depth 가 있어야 정의된다 (배포값 --depth 6)"
    # temporal-only 용 can_bus 주입 hook. forward_test의 0 처리를 우회한다.
    cb_state = {"delta": None}
    if args.ablation == "temporal-only":
        _tc = type(model.module.pts_bbox_head.transformer)
        _orig_gbf = _tc.get_bev_features

        def _gbf(self, *a, **k):
            if cb_state["delta"] is not None:
                d3, da = cb_state["delta"]
                for m in k.get("img_metas", []):
                    cb = np.asarray(m["can_bus"])
                    cb[:3] = d3
                    cb[-1] = da
            return _orig_gbf(self, *a, **k)
        _tc.get_bev_features = _gbf

    rng = np.random.default_rng(args.seed)
    # clip-shuffle 기부자: 시나리오를 한 칸 회전시켜 짝을 만든다(자기 자신 금지).
    donor_of = {s: scens[(i + 1) % len(scens)] for i, s in enumerate(scens)}
    pos = {(s, int(f)): k for k, (s, f) in enumerate(zip(scen, frame))}

    # --depth 를 위해 임의 프레임을 dataset 인덱스로 찾는 표 (dataset 순서 기준)
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j
              for j, i in enumerate(ds.data_infos)}

    pred = np.full((len(scen), 6, 2), np.nan)
    t0 = time.time()
    n_done = 0
    for si, s in enumerate(scens):
        m = scen == s
        order = np.argsort(frame[m])
        rows = np.where(m)[0][order]
        reset_stream(model.module)
        for r in rows:
            if args.ablation in ("current-only", "temporal-only", "zero-current"):
                # 앵커마다 prev_bev를 버린다.
                reset_stream(model.module)
                if args.ablation == "temporal-only":
                    pf = (s, int(frame[r]) - 5)
                    cb_state["delta"] = None
                    if pf in key2ds:
                        a0 = np.asarray(ds.data_infos[key2ds[pf]]["can_bus"],
                                        dtype=float).copy()
                        a1 = np.asarray(ds.data_infos[ds_idx[r]]["can_bus"],
                                        dtype=float).copy()
                        # can_bus[-1]은 get_data_info가 채우는 patch_angle이라
                        # 원시 pkl에서는 0이다. 각도는 ego2global로 직접 만든다.
                        from pyquaternion import Quaternion as _Q
                        def _yaw(info):
                            m = _Q(info["ego2global_rotation"]).rotation_matrix
                            return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
                        y0 = _yaw(ds.data_infos[key2ds[pf]])
                        y1 = _yaw(ds.data_infos[ds_idx[r]])
                        cb_state["delta"] = (a1[:3] - a0[:3],
                                             (y1 - y0 + 180) % 360 - 180)
            elif args.depth:
                # ★ 배포 충실 모드. 공식 제출(tools/etri_test_submit.py)은 clip마다
                # prev_frame_info를 리셋하고 7프레임을 흘려 마지막만 채점한다. 즉 배포
                # 누적 깊이는 **항상 6**이다. 시나리오 60앵커를 연속으로 흘리면 깊이가
                # 59까지 가고, 예측 궤적 크기가 깊이에 따라 부풀어 오른다
                # (실측 3초 |p|: K=1 22.3 -> K=6 27.7, GT 21.4). 그래서 깊이를
                # 명시적으로 고정한다.
                reset_stream(model.module)
                for k in range(args.depth, 0, -1):
                    # history-shuffle: **과거만** 다른 시나리오로 교체하고 현재 프레임은
                    # 진짜를 쓴다. "현재 장면은 보이는데 직전 3초의 시각 이력이 남의
                    # 장면"인 조건 -- 종방향/감속에서 악화돼야 temporal visual cue가
                    # 실제로 유익하다는 근거가 된다. (`--depth` 없이는 정의가 안 된다.)
                    src = donor_of[s] if args.ablation == "history-shuffle" else s
                    wk = (src, int(frame[r]) - k * 5)
                    if wk not in key2ds:
                        continue
                    # ★ 캐시 금지: forward_test가 can_bus를 in-place로 깎는다.
                    dw = collate([ds[key2ds[wk]]], samples_per_gpu=1)
                    if args.ablation != "history-shuffle":
                        apply_ablation(dw, args.ablation, None, rng)
                    with torch.no_grad():
                        model(return_loss=False, rescale=True, **dw)
                if args.ablation == "history-shuffle":
                    # ★ VAD.py:292 -- scene_token이 바뀌면 forward_test가 prev_bev를
                    #   **버린다**. donor 프레임을 흘리면 prev_frame_info['scene_token']이
                    #   donor로 바뀌므로, 채점할 앵커에서 prev_bev가 None이 되어
                    #   current-only와 **자릿수까지 똑같은 값**이 나온다(실측 9.1387).
                    #   원래 시나리오로 되돌려 donor 이력을 살린다.
                    model.module.prev_frame_info["scene_token"] = s
            data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
            donor = None
            if args.ablation in ("clip-shuffle", "goal-shuffle"):
                dk = (donor_of[s], int(frame[r]))
                if dk in pos:
                    donor = collate([ds[int(ds_idx[pos[dk]])]],
                                    samples_per_gpu=1)
            apply_ablation(data, args.ablation, donor, rng)
            with torch.no_grad():
                out = model(return_loss=False, rescale=True, **data)
            fut = out[0]["pts_bbox"]["ego_fut_preds"]
            cmd = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(
                -1, fut.shape[0])[0]
            pred[r] = fut[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
            n_done += 1
        el = time.time() - t0
        print(f"  [{si+1:2d}/{len(scens)}] {s}  {len(rows)}앵커  "
              f"{n_done/el:.2f} it/s  ETA {(keep.sum()-n_done)/max(n_done/el,1e-9)/60:.1f}분",
              flush=True)

    assert not np.isnan(pred[keep]).any(), "NaN 예측이 남았다"

    # ---- 채점: 기존 표 코드를 그대로 탄다 (같은 metric, 같은 열 분해) ----
    sys.path.insert(0, os.path.join(ADCL, "scripts"))
    sys.path.insert(0, os.path.join(ADCL, "src"))
    from etri_table import (ValSet, render, COLUMNS, wl2, STOP,       # noqa: E402
                            SPEED_LO, SPEED_HI, SPEED_HI2)
    d = np.load(CACHE, allow_pickle=True)
    gt_all = d["fut"][cache_idx].astype(np.float64)
    speed = d["speed"][cache_idx].astype(np.float64)
    vcmd = d["vad_cmd"][cache_idx].astype(int)
    rows = []
    if scen_list is None:
        # val 경로: 프록시 행과 test 정합 가중치가 있다
        v = ValSet()
        assert np.array_equal(v.vi, cache_idx), "val 인덱스가 어긋났다"
        gt_all, W_all = v.gt, v.w
        proxy = v.label_proxy_row()
        rows.append((proxy["_name"], proxy))
        masks = v.masks()
    else:
        # 임의 시나리오 경로(8-scene overfit 등): 프록시·가중치 없음 -> 무가중 1.0
        W_all = np.ones(len(cache_idx))
        masks = {"가중": None, "무가중": np.ones(len(cache_idx), bool),
                 "right": vcmd == 0, "이동": speed >= STOP,
                 "3–8m/s": (speed >= SPEED_LO) & (speed < SPEED_HI),
                 "15+m/s": speed >= SPEED_HI2}

    def sub(mask, name):
        if mask.sum() == 0:
            return
        r = {}
        for c, cm in masks.items():
            mm = mask if cm is None else (mask & cm)
            r[c] = (wl2(pred[mm], gt_all[mm], W_all[mm]) if c == "가중"
                    else wl2(pred[mm], gt_all[mm], None))
            r["n_" + c] = int(mm.sum())
        rows.append((name, r))

    tag = f"{os.path.basename(args.checkpoint)} / {args.ablation}"
    sub(keep, f"{tag}  전체")
    sub(keep & (frame >= 30), f"{tag}  frame>=30 (배포충실)")

    note = "frame>=30 은 선행 앵커 6개 이상 = test clip(7프레임)과 동일 조건."
    if scen_list is not None:
        note += ("\n**임의 시나리오 모드**: 프록시 행과 test 정합 가중치가 없다. "
                 "'가중' 열은 무가중과 동일하다.")
    md = render(rows, title=f"VAD 평가 — {tag}", note=note,
                with_residual=scen_list is None)
    print("\n" + md)

    # 궤적 형태 진단 -- nuScenes prior(~20 m 고정)에서 벗어났는지 본다
    p3, g3 = pred[keep][:, -1], gt_all[keep][:, -1]
    print(f"3초 성분 p50   예측 x {np.percentile(p3[:,0],50):+7.2f} "
          f"y {np.percentile(p3[:,1],50):+7.2f}   |p| "
          f"{np.percentile(np.linalg.norm(p3,axis=1),50):6.2f}")
    print(f"               GT   x {np.percentile(g3[:,0],50):+7.2f} "
          f"y {np.percentile(g3[:,1],50):+7.2f}   |g| "
          f"{np.percentile(np.linalg.norm(g3,axis=1),50):6.2f}")
    sk = speed[keep]
    for lo, hi in ((8, 15), (15, 99)):
        m = (sk >= lo) & (sk < hi)
        if m.sum() > 20:
            print(f"  {lo}-{hi} m/s (n={m.sum():4d})  예측|p| p50 "
                  f"{np.percentile(np.linalg.norm(p3[m],axis=1),50):6.2f}  "
                  f"GT {np.percentile(np.linalg.norm(g3[m],axis=1),50):6.2f}")

    out = args.out or os.path.join(
        ADCL, "logs", f"vad_eval_{os.path.basename(args.checkpoint).replace('.pth','')}"
        f"_{args.ablation}.json")
    payload = dict(
        config=args.config, checkpoint=args.checkpoint, ablation=args.ablation,
        split=args.split, n_scenarios=len(scens), n_anchors=int(keep.sum()),
        load=dict(dropped=len(load_info["dropped"]),
                  missing=len(load_info["missing"]),
                  unexpected=len(load_info["unexpected"]),
                  dropped_keys=load_info["dropped"]),
        rows={n: {c: r[c] for c in COLUMNS} for n, r in rows},
        elapsed_s=round(time.time() - t0, 1))
    json.dump(payload, open(out, "w"), indent=1, ensure_ascii=False)
    np.save(out.replace(".json", "_pred.npy"), pred)
    print(f"저장 {out}\n     {out.replace('.json', '_pred.npy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
