#!/usr/bin/env python
"""추론 비결정성 규명 + can_bus 불변성 in-process 테스트.

왜 필요한가
----------
`etri_vad_eval.py`를 동일 설정으로 두 번 돌렸더니 예측이 최대 **2.3e-01 m** 달랐고
가중 L2가 11.0569 vs 11.0500 (**±0.007**)로 흔들렸다. 이건 두 가지를 망가뜨린다.

1. **can_bus 불변성 테스트를 못 한다.** `< 1e-6` 기준은 노이즈에 묻힌다. 실제로
   ablation 차(2.08e-01)가 동일 설정 재현 차(2.32e-01)보다 작았다 -- 불변성 실패로
   오판할 뻔했다.
2. **판정 기준이 노이즈 아래다.** head tournament 채택 기준으로 세운 "가중 L2
   0.005 개선"이 측정 노이즈 ±0.007보다 작다. 이대로면 노이즈를 개선으로 읽는다.

그래서 이 스크립트는 **한 프로세스 안에서** 같은 앵커를 반복 forward 해
    A: 원본 / B: 원본 재실행 (노이즈 바닥) / C: can_bus[7:16] 교란
을 비교한다. 판정은 절대 임계가 아니라 **|A-C| 가 |A-B| 와 같은 규모인가**다.

`--deterministic`으로 TF32/cudnn autotune을 끄고 노이즈가 사라지는지도 확인한다.

    python scripts/etri_determinism.py CFG CKPT
    python scripts/etri_determinism.py CFG CKPT --deterministic
"""
import argparse
import importlib
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))
ANN = "/tmp/pm97/data/etri/pkl/vad_etri_infos_temporal_train.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=24, help="앵커 수")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    args.config = os.path.abspath(args.config)
    args.checkpoint = os.path.abspath(args.checkpoint)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

    import torch
    if args.deterministic:
        # 재현성을 깨는 흔한 3가지: cudnn autotune, TF32, 비결정 커널.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"deterministic={args.deterministic}  "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}  "
          f"matmul.tf32={torch.backends.cuda.matmul.allow_tf32}  "
          f"cudnn.tf32={torch.backends.cudnn.allow_tf32}")

    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    from etri_vad_eval import build_index, load_model, reset_stream

    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = ANN
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    scen, frame, ds_idx = build_index("val_idx")
    s0 = sorted(set(scen))[0]
    m = np.where(scen == s0)[0][np.argsort(frame[scen == s0])][:args.n]
    print(f"시나리오 {s0}  앵커 {len(m)}  frame {frame[m].min()}~{frame[m].max()}")

    model, _ = load_model(cfg, args.checkpoint)
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()
    use_cb = cfg.model["pts_bbox_head"]["transformer"].get("use_can_bus", True)
    print(f"use_can_bus={use_cb}\n")

    rng = np.random.default_rng(0)
    runs = {}
    for tag in ("A", "B", "C"):
        reset_stream(model.module)
        out = np.zeros((len(m), 6, 2))
        for k, r in enumerate(m):
            data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
            if tag == "C":
                for im in data["img_metas"][0].data[0]:
                    np.asarray(im["can_bus"])[7:16] = rng.normal(0, 10, 9)
            with torch.no_grad():
                o = model(return_loss=False, rescale=True, **data)
            fut = o[0]["pts_bbox"]["ego_fut_preds"]
            cmd = np.asarray(data["ego_fut_cmd"][0].data[0]).reshape(
                -1, fut.shape[0])[0]
            out[k] = fut[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
        runs[tag] = out
        print(f"  {tag} 완료  |3초변위| p50 "
              f"{np.percentile(np.linalg.norm(out[:, -1], axis=1), 50):.3f} m")

    ab = np.abs(runs["A"] - runs["B"]).max()
    ac = np.abs(runs["A"] - runs["C"]).max()
    print("\n=== 결과 ===")
    print(f"  |A-B|  동일 설정 재실행 (노이즈 바닥)   max {ab:.3e}")
    print(f"  |A-C|  can_bus[7:16] 교란              max {ac:.3e}")
    if ab < 1e-6:
        verdict = ("PASS  can_bus 불변" if ac < 1e-6
                   else "FAIL  can_bus가 출력에 영향")
    else:
        verdict = (f"결정론 확보 실패 -- 노이즈({ab:.1e})가 커서 판정 불가. "
                   f"|A-C|/|A-B| = {ac/ab:.2f}"
                   + ("  (노이즈 수준 = 불변으로 추정)" if ac <= 3 * ab
                      else "  (노이즈보다 큼 = 영향 의심)"))
    print(f"  판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
