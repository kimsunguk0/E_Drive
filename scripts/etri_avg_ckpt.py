#!/usr/bin/env python
"""체크포인트 가중치 평균 (SWA 방식).

왜: v2a 학습의 val L2 는 epoch 간 0.237~0.377 로 진동하고, 그 대부분이 종방향
스케일 s* 의 흔들림(0.975~1.002)이다. 진동하는 점들의 가중치를 평균하면 진동
중심에 놓인다.

규정: **val 에서 아무것도 적합하지 않는다.** "지정된 epoch 전부를 균등 평균"이라는
고정 규칙만 쓴다. 어느 체크포인트를 넣을지 val 점수로 고르면 그게 캘리브레이션
누수다 (절대규칙 3). 그래서 기본은 --epochs 로 명시받고, 사용자가 준 목록을
그대로 균등 평균한다.

주의: BN running stat 은 평균해도 되지만(여기 모델은 BN 이 frozen 이라 무의미),
num_batches_tracked 같은 정수 버퍼는 평균하면 안 되므로 첫 체크포인트 값을 쓴다.
"""
import argparse, os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", required=True, help="쉼표 구분. 예: 1,2,3,4")
    ap.add_argument("--root", default="/home/pm97/workspace/sukim/adcl/b200_eval")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eps = [int(x) for x in args.epochs.split(",")]
    paths = [os.path.join(args.root, args.run, f"epoch{e}", f"epoch_{e}.pth") for e in eps]
    for p in paths:
        assert os.path.isfile(p), f"없음: {p}"
    print(f"{args.run}  평균 대상 {len(paths)}개: epochs {eps}")

    acc, meta, n_float, n_copied = None, None, 0, 0
    for i, p in enumerate(paths):
        c = torch.load(p, map_location="cpu")
        sd = c.get("state_dict", c)
        if acc is None:
            meta = c.get("meta", {})
            acc = {}
            for k, v in sd.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    acc[k] = v.clone().double(); n_float += 1
                else:
                    acc[k] = v.clone(); n_copied += 1   # 정수 버퍼는 첫 판 값 유지
        else:
            assert set(sd) == set(acc), "state_dict 키가 다르다"
            for k, v in sd.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    assert v.shape == acc[k].shape, (k, v.shape, acc[k].shape)
                    acc[k] += v.double()
        print(f"  [{i+1}/{len(paths)}] {os.path.basename(os.path.dirname(p))}"
              f"  meta epoch {c.get('meta', {}).get('epoch')}"
              f"  iter {c.get('meta', {}).get('iter')}")

    out_sd = {}
    for k, v in acc.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out_sd[k] = (v / len(paths)).float()
        else:
            out_sd[k] = v
    bad = [k for k, v in out_sd.items()
           if torch.is_tensor(v) and v.is_floating_point() and not v.isfinite().all()]
    assert not bad, f"비유한 값: {bad[:5]}"
    meta = dict(meta); meta["avg_of"] = eps; meta["avg_run"] = args.run
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": out_sd, "meta": meta}, args.out)
    print(f"평균 {n_float}개 텐서, 복사 {n_copied}개 (비부동소수)")
    print(f"저장 {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
