#!/usr/bin/env python
"""canonical bootstrap 체크포인트를 만든다 — 모든 실험의 동일한 출발점.

왜
--
지금까지는 매 실험이 `VAD_tiny.pth`를 로드하면서 **12개 불일치 tensor**
(`cls_branches.{0,1,2}.6`, `map_cls_branches.{0,1,2}.6`)를 새로 초기화했다. 그 로짓이
어떤 agent/map query가 높은 점수를 받는지를 바꿔 planner의 cross-attention까지 흔들고,
seed를 안 잡으면 동일 설정 재실행에서 가중 L2가 ±0.004~0.007 흔들렸다
(실측 seed 0/1/2 = 10.9549 / 10.9588 / 10.9551).

seed를 매번 맞추는 것으로는 부족하다. **12개 tensor까지 파일에 박아** 이후 모든 실험이
`strict=True`, `missing 0 / unexpected 0`으로 로드하게 만든다. 그러면 A/B 차이가
초기화 노이즈가 아니라 설정 차이로 한정된다.

만드는 순서
----------
    1. seed 고정
    2. config로 모델 build (12개 tensor가 결정적으로 초기화됨)
    3. 축 리맵된 nuScenes 가중치를 strict=False로 덮어씀
       -> 12개는 2번의 값이 그대로 남는다 (분류 로짓은 기하 리맵 대상이 아니다)
    4. 전체 state_dict + 출처 메타데이터를 저장

BEV 격자 순열은 **적용하지 않는다** — zero-shot 결과가 열마다 갈리고(가중은 순열 없음이
0.026 우세, 이동·3–8m/s는 순열이 우세) 애초에 fine-tuning이 씻어낼 초기화 선택이라
zero-shot으로 판정할 문제가 아니다.

    python scripts/etri_bootstrap_ckpt.py \
      --config .../VAD_etri_tvad_bootstrap.py --nuscenes ckpt/vad/VAD_tiny.pth \
      --out /tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth
"""
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "etri_vad", "ETRI_E2E_Driving_Challenge"))


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nuscenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="ego,traj,ref,reg,map")
    args = ap.parse_args()
    for k in ("config", "nuscenes", "out"):
        setattr(args, k, os.path.abspath(getattr(args, k)))

    src_sha = sha256(args.nuscenes)
    cfg_sha = hashlib.sha256(open(args.config, "rb").read()).hexdigest()

    # 1) 리맵을 임시 파일로 (원본은 건드리지 않는다)
    tmp = args.out + ".remap.tmp"
    remap = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "etri_remap_ckpt.py")
    r = subprocess.run([sys.executable, remap, args.nuscenes, tmp,
                        "--parts", args.parts],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return r.returncode
    n_remap = sum(1 for ln in r.stdout.splitlines() if "pts_bbox_head." in ln)
    print(f"축 리맵 {n_remap}개 텐서 (--parts {args.parts})")

    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.models import build_model

    # 2) 결정적 초기화
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.init_weights()          # ImageNet R50 + 나머지 결정적 초기화
    own = model.state_dict()
    init12 = {k: v.clone() for k, v in own.items()
              if k.endswith(".6.weight") or k.endswith(".6.bias")}

    # 3) 리맵 가중치 덮어쓰기
    sd = torch.load(tmp, map_location="cpu")["state_dict"]
    drop = [k for k, v in sd.items()
            if k in own and tuple(own[k].shape) != tuple(v.shape)]
    for k in drop:
        sd.pop(k)
    res = model.load_state_dict(sd, strict=False)
    print(f"nuScenes 덮어쓰기: shape 불일치 제외 {len(drop)}  "
          f"missing {len(res.missing_keys)}  unexpected {len(res.unexpected_keys)}")
    # 제외된 12개가 2번의 결정적 초기화 값 그대로인지 확인
    now = model.state_dict()
    kept = [k for k in drop if k in init12
            and torch.equal(now[k], init12[k].to(now[k].device))]
    print(f"  결정적 초기화가 유지된 텐서 {len(kept)}/{len(drop)}")
    assert len(kept) == len(drop), "제외한 텐서가 덮어써졌다"

    # 4) 저장 -- 이후 strict=True 로드가 되어야 한다
    final = {k: v.cpu() for k, v in model.state_dict().items()}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=os.path.dirname(remap)).stdout.strip()
    except Exception:
        commit = ""
    meta = dict(
        provenance="canonical bootstrap for ETRI C0-T",
        nuscenes_src=os.path.basename(args.nuscenes), nuscenes_sha256=src_sha,
        config=os.path.basename(args.config), config_sha256=cfg_sha,
        remap_parts=args.parts, bev_permutation=False, init_seed=args.seed,
        adcl_commit=commit, n_tensors=len(final),
        note="12 class-logit tensors are baked in; load with strict=True")
    torch.save(dict(state_dict=final, meta=meta), args.out)
    os.remove(tmp)
    out_sha = sha256(args.out)
    meta["self_sha256"] = out_sha
    json.dump(meta, open(args.out + ".json", "w"), indent=1, ensure_ascii=False)

    print(f"\n저장 {args.out}  ({os.path.getsize(args.out)/1e6:.0f} MB, "
          f"{len(final)} 텐서)")
    print(f"  sha256 {out_sha}")
    print(f"  출처   {args.out}.json")

    # 검증: strict=True 로 다시 로드
    torch.manual_seed(args.seed + 999)      # 다른 seed로 build 해도 무관해야 한다
    m2 = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    r2 = m2.load_state_dict(torch.load(args.out, map_location="cpu")["state_dict"],
                            strict=True)
    print(f"\nstrict=True 재로드 검증: missing {len(r2.missing_keys)}  "
          f"unexpected {len(r2.unexpected_keys)}  -> "
          f"{'PASS' if not r2.missing_keys and not r2.unexpected_keys else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
