#!/usr/bin/env python
"""Compute the map (vectorized HD-map) mAP from a saved submission_vector.json.

WHY THIS EXISTS
---------------
Two separate defects kept the map metric from ever completing:

  1. patch 05 (mine) stored the threadpoolctl controller on `self`, so pickling a
     pool task -- `partial(self._evaluate_single, ...)` -- tried to pickle
     ctypes CDLL function pointers and died with
       AttributeError: Can't pickle local object 'CDLL.__init__.<locals>._FuncPtr'
  2. once that was fixed the real, upstream problem surfaced: the Pool is created
     with the default *fork* start method inside a process that has already
     initialised CUDA and ~40 threads. The children inherit mutexes that were
     locked at fork time; the parent then blocks on a futex forever. Observed
     2026-08-04: 16 workers idle at 0.0% CPU having used 5-12 s each, parent
     asleep in syscall 202, no progress for 2.5 h.

The fix is `multiprocessing.get_context('forkserver')` plus a worker initialiser
that caps BLAS threads (forkserver children do not inherit the parent's limit).

This script exercises that fix UNDER THE CONDITION THAT BROKE IT: it initialises
CUDA first, on purpose, before running the evaluator. Passing here means the map
metric will also survive inside tools/test.py, which always has CUDA up.

Reuses submission_vector.json from a previous run, so no 25-minute inference.

    python scripts/eval_map_only.py \
        --submission src/SparseDrive/work_dirs/sparsedrive_small_stage2/submission_vector.json

Deliberately allocates only a few MB of VRAM (shared-GPU constraint).
"""
import argparse
import os
import sys

PERSIST = "/home/pm97/workspace/sukim/adcl"
REPO = os.path.join(PERSIST, "src", "SparseDrive")
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        REPO, "projects/configs/sparsedrive_small_stage2.py"))
    ap.add_argument("--submission", default=os.path.join(
        REPO, "work_dirs/sparsedrive_small_stage2/submission_vector.json"))
    ap.add_argument("--no-cuda", action="store_true",
                    help="skip the CUDA init; the point of this script is to keep it")
    args = ap.parse_args()

    import torch
    from mmcv import Config
    import projects.mmdet3d_plugin  # noqa: F401
    from projects.mmdet3d_plugin.datasets.evaluation.map.vector_eval import (
        VectorEvaluate, MAP_EVAL_START_METHOD, N_WORKERS,
    )

    if not args.no_cuda:
        # Reproduce the condition that deadlocked fork: a live CUDA context plus
        # the thread pool torch spins up. A few MB is enough -- the hazard is the
        # context and the threads, not the allocation size.
        t = torch.zeros(256, 256, device="cuda")
        torch.cuda.synchronize()
        print(f"CUDA 초기화됨 (실패 조건 재현): {t.numel() * 4 / 2**20:.1f} MiB, "
              f"threads={len(os.listdir('/proc/self/task'))}")

    print(f"start method : {MAP_EVAL_START_METHOD}   workers: {N_WORKERS}")
    print(f"submission   : {args.submission} "
          f"({os.path.getsize(args.submission) / 2**20:.0f} MiB)")

    cfg = Config.fromfile(args.config)
    evaluator = VectorEvaluate(cfg.eval_config)
    out = evaluator.evaluate(args.submission, logger=None)

    print("\n" + "=" * 60)
    print(" map 지표")
    print("=" * 60)
    for k in sorted(out):
        v = out[k]
        print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
    print("=" * 60)
    print(" 논문 mapping mAP: 0.5689")
    return 0


if __name__ == "__main__":
    sys.exit(main())
