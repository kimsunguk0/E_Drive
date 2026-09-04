#!/usr/bin/env python
"""학습된 영상 전용 스코어러로 val 전체의 사전 logits 를 덤프한다.

`etri_vad_eval.py` 와 같은 하네스를 쓴다 -- 같은 val 인덱스, 같은 클립별 stream
reset, 같은 depth 6 재생. 다른 점은 궤적 대신 **[N, K] logits** 를 저장하는 것이다.
선택은 `etri_vocab_select.py` 가 모델 밖에서 한다.

    python scripts/etri_vocab_dump.py <config> <ckpt> --out val_logits.npz
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch

ADCL = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE = "/tmp/pm97/data/etri/ego_cache.npz"
SPLIT = "/tmp/pm97/data/etri/val_clips.npz"


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--repo", required=True)
    # 기본값 없음. 해상도별 pkl 이 달라서 기본값을 두면 조용히 틀린 걸 먹인다
    # (실측 사고: 1536 모델에 768 pkl 을 먹여 val top-1 이 10.20 = 무작위 수준으로 붕괴,
    #  학습 지표는 sel 0.598 로 정상이었다).
    ap.add_argument("--ann-file", required=True,
                    help="config 의 해상도와 반드시 일치해야 한다")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    os.chdir(repo)
    sys.path.insert(0, repo)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    # 해상도 정합 검증: CachedImageGeometry.scale 과 pkl 이 가리키는 캐시가 맞아야 한다
    scales = [t.get('scale') for t in cfg.test_pipeline
              if t.get('type') == 'CachedImageGeometry']
    if scales:
        want = 'etri_1536' if abs(scales[0] - 0.8) < 1e-6 else 'etri_768'
        import pickle as _pk
        _p0 = _pk.load(open(args.ann_file, 'rb'))['infos'][0]
        _dp = next(iter(_p0['cams'].values()))['data_path']
        if want == 'etri_1536' and 'etri_1536' not in _dp:
            raise SystemExit(f"해상도 불일치: scale={scales[0]} 인데 pkl 이 {_dp}")
        if want == 'etri_768' and 'etri_1536' in _dp:
            raise SystemExit(f"해상도 불일치: scale={scales[0]} 인데 pkl 이 {_dp}")
        print(f"해상도 정합 OK: CachedImageGeometry scale={scales[0]}  ->  {_dp}")
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)

    ds = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    info = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0]).eval()

    dec = model.module.pts_bbox_head.goal_decoder
    assert dec is not None and not getattr(dec, 'uses_condition', True), \
        "이 스크립트는 영상 전용 스코어러 전용이다 (uses_condition=False)"
    K = int(dec.num_anchors)

    # head 출력에서 logits 를 가로챈다. forward_test 가 outs 를 반환하지 않으므로
    # 디코더 forward 를 감싸 마지막 호출의 logits 를 보관한다.
    grab = {}
    orig = type(dec).forward

    def wrapped(self, *a, **kw):
        out = orig(self, *a, **kw)
        grab['logits'] = out[1].detach().float().cpu().numpy()
        return out

    type(dec).forward = wrapped

    # val 앵커 인덱스 -- 반드시 ds.data_infos 로 매핑한다 (pkl 순서 금지)
    d = np.load(CACHE, allow_pickle=True)
    scen_all = np.array([str(x) for x in d["scenarios"]])[d["scen_idx"]]
    frame_all = d["frame"].astype(int)
    cache_idx = np.load(SPLIT, allow_pickle=True)["val_idx"]
    scen, frame = scen_all[cache_idx], frame_all[cache_idx]
    key2ds = {(i["scene_token"], int(i["frame_idx"])): j
              for j, i in enumerate(ds.data_infos)}
    ds_idx = np.array([key2ds[(s, int(f))] for s, f in zip(scen, frame)])
    for j, s, f in zip(ds_idx, scen, frame):
        assert ds.data_infos[j]["scene_token"] == s and \
            int(ds.data_infos[j]["frame_idx"]) == int(f)

    logits = np.zeros((len(cache_idx), K), dtype=np.float32)
    import mmcv
    prog = mmcv.ProgressBar(len(cache_idx))
    for r, (s, f, j) in enumerate(zip(scen, frame, ds_idx)):
        # 배포 충실: 클립별 reset 후 depth 개 선행 프레임을 0.5 s 간격으로 재생
        reset_stream(model.module)
        hist = [int(f) - 5 * k for k in range(args.depth, 0, -1)]
        for h in hist + [int(f)]:
            jj = key2ds.get((s, h))
            if jj is None:
                continue
            with torch.no_grad():
                model(return_loss=False, rescale=True,
                      **collate([ds[jj]], samples_per_gpu=1))
        logits[r] = grab['logits'][0]
        prog.update()

    np.savez(args.out, logits=logits, cache_idx=cache_idx,
             scen=scen, frame=frame, K=K)
    print(f"\n저장 {args.out}  logits {logits.shape}")
    print(f"  load: missing {len(info.get('missing_keys', []))} "
          f"unexpected {len(info.get('unexpected_keys', []))}"
          if isinstance(info, dict) else "")


if __name__ == "__main__":
    main()
