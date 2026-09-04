#!/usr/bin/env python
"""no-goal full330 epoch3 -> C0-T v2a 초기값(visual trunk)만 뽑는다.

형님 지시:
    포함  image backbone / image neck / BEV encoder / spatial cross-attention /
          temporal BEV / agent-map representation
    제외  기존 no-goal planning head, 기존 trajectory output layer

`pts_bbox_head.ego_fut_decoder.*` 6개가 바로 그 planning head다. 이걸 남기면
v2a에서 쓰지도 않는 가중치가 ckpt에 섞여 계보가 흐려진다. optimizer state도 버린다
(새 head는 random init이고 LR 그룹 구성이 달라서 이어받으면 안 된다).

goal decoder / content_value_proj / content_value_norm 은 ckpt에 없으므로
`load_from` 시 missing_keys로 나오고 deterministic random init 된다 -- 의도대로다.

    python scripts/etri_make_trunk_ckpt.py --src <epoch_3.pth> --out <trunk.pth>
"""
import argparse
import hashlib
import os
import sys

DROP_PREFIX = ('pts_bbox_head.ego_fut_decoder.',)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    ck = torch.load(os.path.abspath(args.src), map_location="cpu")
    sd = ck.get("state_dict", ck)
    dropped = [k for k in sd if k.startswith(DROP_PREFIX)]
    kept = {k: v for k, v in sd.items() if k not in set(dropped)}
    print(f"원본 {os.path.basename(args.src)}  epoch "
          f"{ck.get('meta', {}).get('epoch')}  키 {len(sd)}")
    for k in dropped:
        print(f"  제외  {k}  {tuple(sd[k].shape)}")
    print(f"유지 키 {len(kept)}  (optimizer state 버림)")
    out = {"state_dict": kept,
           "meta": {"lineage": "tvad_full330 no-goal epoch 3 visual trunk",
                    "source": os.path.abspath(args.src),
                    "source_epoch": ck.get("meta", {}).get("epoch"),
                    "dropped": dropped,
                    "note": "planning head 제거. C0-T v2a `load_from` 전용."}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, os.path.abspath(args.out))
    h = hashlib.sha256(open(os.path.abspath(args.out), "rb").read()).hexdigest()
    with open(os.path.abspath(args.out) + ".sha256", "w") as f:
        f.write(h + "\n")
    print(f"\n저장 {args.out}  "
          f"{os.path.getsize(os.path.abspath(args.out))/1048576:.0f} MB")
    print(f"sha256 {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
