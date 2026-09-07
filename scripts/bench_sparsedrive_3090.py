#!/usr/bin/env python
"""SparseDrive-S 의 3090 실측 지연 — ETRI 누적 T_infer 규칙 기준.

질문4 답변 4: T_infer 는 초기화 이후 과거 시점 처리부터 마지막 궤적 출력까지
**모든 model forward 의 누적**이다. 따라서 streaming 모델은 프레임 수만큼 곱한다.

가중치는 무작위다(지연은 가중치와 무관). 데이터는 config 규격의 합성 입력이다.
전처리/후처리는 질문1 답변 4 에 따라 제외하고 model forward 만 잰다.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch


def build(cfg_path, dev):
    from mmcv import Config
    from mmdet.models import build_detector
    import projects.mmdet3d_plugin  # noqa: F401  (registry 등록)
    cfg = Config.fromfile(cfg_path)
    m = cfg.model
    m.pop("train_cfg", None)
    m.pop("test_cfg", None)
    # 사전학습 backbone 다운로드를 막는다(지연은 가중치와 무관)
    # 사전학습 가중치 로드를 전부 끊는다(지연은 가중치와 무관).
    m.pop("pretrained", None)
    m.pop("init_cfg", None)
    def _strip(d):
        if isinstance(d, dict):
            d.pop("init_cfg", None); d.pop("pretrained", None)
            for v in d.values():
                _strip(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                _strip(v)
    _strip(m)
    model = build_detector(m)
    return model.to(dev).eval(), cfg


def make_data(cfg, dev, n_cam=6, bs=1, t=0.0):
    ish = cfg.input_shape          # (W, H)
    W, H = int(ish[0]), int(ish[1])
    img = torch.randn(bs, n_cam, 3, H, W, device=dev)
    proj = torch.eye(4, device=dev).view(1, 1, 4, 4).repeat(bs, n_cam, 1, 1)
    proj[..., 0, 0] = 500.0; proj[..., 1, 1] = 500.0
    proj[..., 0, 2] = W / 2.0; proj[..., 1, 2] = H / 2.0
    proj[..., 2, 3] = 1.0
    image_wh = torch.tensor([[W, H]], device=dev, dtype=torch.float32).repeat(bs, n_cam, 1)
    Tg = np.eye(4); Tg[0, 3] = t * 10.0
    metas = [dict(T_global=Tg, T_global_inv=np.linalg.inv(Tg),
                  timestamp=t, box_type_3d=None, scene_token="s0") for _ in range(bs)]
    data = dict(
        projection_mat=proj, image_wh=image_wh,
        timestamp=torch.full((bs,), t, device=dev, dtype=torch.float64),
        img_metas=metas,
        gt_ego_fut_cmd=torch.zeros(bs, 3, device=dev),
        focal=torch.full((bs, n_cam), 500.0, device=dev),
    )
    data["gt_ego_fut_cmd"][:, 2] = 1.0
    return img, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="projects/configs/sparsedrive_small_stage2.py")
    ap.add_argument("--frames", type=int, default=4, help="ETRI 누적: 과거+현재 프레임 수")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", default="sparsedrive_latency_3090.json")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    print("torch %s | %s" % (torch.__version__, torch.cuda.get_device_name(0)), flush=True)
    model, cfg = build(args.cfg, dev)
    n_par = sum(p.numel() for p in model.parameters())
    print("SparseDrive params %.1fM | input %s | queue_length %s"
          % (n_par / 1e6, cfg.input_shape, getattr(cfg, "queue_length", "?")), flush=True)

    def reset():
        for mod in model.modules():
            if hasattr(mod, "reset") and callable(mod.reset) and mod is not model:
                try:
                    mod.reset()
                except Exception:
                    pass

    def one_clip(nf):
        """reset -> nf 프레임 순차 forward. 마지막이 제출 프레임."""
        reset()
        for i in range(nf):
            img, data = make_data(cfg, dev, t=0.5 * i)
            with torch.no_grad():
                out = model(img, **data)
        return out

    # 정상 동작 확인
    try:
        o = one_clip(2)
        print("forward OK, 출력 키:", list(o[0].keys()) if isinstance(o, list) else type(o),
              flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc(limit=8)
        print("FORWARD FAIL:", type(e).__name__, e)
        return 1

    rows = []
    for nf in (1, args.frames, 7):
        for _ in range(args.warmup):
            one_clip(nf)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.repeat):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            one_clip(nf)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.asarray(ts); med = float(np.median(ts))
        pen = 1.0 + max(0.0, med - 100.0) / 200.0
        rows.append(dict(frames=nf, median=med, p95=float(np.percentile(ts, 95)),
                         p99=float(np.percentile(ts, 99)), penalty=pen,
                         per_frame=med / nf))
        print("frames=%d  누적 median %8.2f ms  (프레임당 %6.2f)  p99 %8.2f  penalty x%.3f"
              % (nf, med, med / nf, np.percentile(ts, 99), pen), flush=True)
    json.dump(dict(params_M=n_par / 1e6, rows=rows), open(args.out, "w"), indent=1)
    print("\nsaved %s" % args.out)
    print("비교: 우리 T4 + learned selector = 46.997 ms (3090 실측), penalty x1.000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
