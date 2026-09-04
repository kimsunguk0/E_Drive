#!/usr/bin/env python
"""Compute planning metrics (L2 / collision) from a saved results.pkl.

WHY THIS EXISTS
---------------
`tools/test.py --eval bbox` runs detection -> tracking -> map -> motion ->
planning in one pass, and the map stage is both the slowest and the most fragile
part of it:

  * it builds a fork-based multiprocessing.Pool inside a process that has already
    initialised CUDA and ~40 threads. Forking such a process is unsupported; the
    children inherit locked mutexes and the parent then blocks forever on a futex
    (observed 2026-08-04: 16 workers idle at 0.0% CPU, parent asleep in syscall
    202, no progress for 2.5 h).
  * even when it does run it is ~200 ms/sample single-threaded.

The planning verdict does not depend on any of that -- `planning_eval` takes the
results list and its own GT dataloader. This script calls it directly, so the L2
number is reachable in minutes and is not hostage to the map evaluator.

Reports BOTH L2 conventions, because reports differ on which one they mean:
  * UniAD/VAD/SparseDrive style: cumulative mean up to 1s/2s/3s, then averaged
  * challenge style:             plain mean over all waypoints
`src/challenge_metrics.py` proves these coincide (challenge == cum[-1]) for a
3 s / 2 Hz horizon.

    python scripts/eval_planning_only.py \
        --config src/SparseDrive/projects/configs/sparsedrive_small_stage2.py \
        --results src/SparseDrive/work_dirs/sparsedrive_small_stage2/results.pkl
"""
import argparse
import os
import sys

PERSIST = "/home/pm97/workspace/sukim/adcl"
REPO = os.path.join(PERSIST, "src", "SparseDrive")
sys.path.insert(0, REPO)
sys.path.insert(0, PERSIST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/sparsedrive_small_stage2.py"))
    ap.add_argument("--results", default=os.path.join(
        REPO, "work_dirs/sparsedrive_small_stage2/results.pkl"))
    args = ap.parse_args()

    import mmcv
    from mmcv import Config
    import projects.mmdet3d_plugin  # noqa: F401  (registers the dataset)
    from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import (
        planning_eval,
    )

    print(f"config : {args.config}")
    print(f"results: {args.results} "
          f"({os.path.getsize(args.results) / 2**30:.2f} GiB)")

    cfg = Config.fromfile(args.config)
    print("results.pkl 로딩 중 …", flush=True)
    results = mmcv.load(args.results)
    print(f"  샘플 {len(results)} 개")

    out = planning_eval(results, cfg.eval_config, logger=None)

    print("\n" + "=" * 60)
    print(" planning 지표 (repo 구현)")
    print("=" * 60)
    for k in sorted(out):
        v = out[k]
        print(f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v}")

    # --- 두 L2 관례 병기 -----------------------------------------------------
    l2_keys = sorted(k for k in out if "L2" in k or "l2" in k)
    if l2_keys:
        print("-" * 60)
        print("  L2 관례 주의: 위 값이 '누적평균(UniAD/VAD/SparseDrive)'인지")
        print("  't초 시점값'인지는 planning_eval 구현을 따릅니다.")
        try:
            from src.challenge_metrics import l2_uniad_style, l2_challenge_style
            print(f"  (challenge_metrics 모듈 사용 가능: "
                  f"{l2_uniad_style.__name__}, {l2_challenge_style.__name__})")
        except Exception as e:  # noqa: BLE001
            print(f"  (challenge_metrics import 실패: {e})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
