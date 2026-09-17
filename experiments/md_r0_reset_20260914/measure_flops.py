#!/usr/bin/env python3
"""Measure __flops__ the way the organisers' tool does, and merge it in.

The submission's top-level dict carries an integer "__flops__" beside the 1,125
clip entries; a submission without it cannot be judged against the 7,053 GFLOPs
first-round cutoff. The organisers' measure_flops.py uses PyTorch's own
torch.utils.flop_counter.FlopCounterMode over ONE forward with the real input
and sums the Global counts, so that is what is used here rather than the
torch.profiler figure recorded elsewhere -- the two counters do not agree and
the cutoff is judged on this one.

"Single frame" for the VAD baseline meant one of its seven sequential forwards.
This model consumes its four past frames inside ONE top-level forward, so its
single forward IS the clip, and that whole forward is counted. Nothing is
divided down to look smaller.
"""
from __future__ import annotations
import argparse, io, json, sys, tarfile
from pathlib import Path
import torch
import torch.utils.module_tracker as _module_tracker
from torch.utils.flop_counter import FlopCounterMode


class _NoHandle:
    def remove(self):
        pass


# Same workaround the organisers' measure_flops.py applies: the module tracker
# registers backward hooks that assert a grad_fn exists, which fails under
# torch.no_grad(). Disabling it changes only the tracker's module attribution,
# not the operator counts that are summed here.
_module_tracker.register_multi_grad_hook = lambda *a, **k: _NoHandle()

ROOT = Path("/NHNHOME/data/sukim/adcl")
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments/md_r0_reset_20260914")):
    sys.path.insert(0, _p)

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
import motiondrive_v2_training as mt
import matching_resolution as mr
import mr_deploy

CUTOFF_GFLOPS = 7053.0


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init", required=True)
    parser.add_argument("--clip", required=True, help="one extracted test clip directory")
    parser.add_argument("--submission", help="merge __flops__ into this submission json")
    parser.add_argument("--out", required=True, help="report json")
    parser.add_argument("--mr-detail", default="native", choices=["native", "lowdetail"])
    args = parser.parse_args()

    payload = torch.load(args.init, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
    mr.install(model, args.mr_detail)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()

    prepared = mr_deploy.prepare_mr_clip_inputs(Path(args.clip), detail=args.mr_detail)
    inputs = {k: v.to(device) for k, v in prepared.inputs.items()}

    with torch.no_grad():                      # warm up shapes and any lazy init
        model(**inputs)
    counter = FlopCounterMode(display=False, depth=3)
    with torch.no_grad(), counter:
        model(**inputs)
    counts = counter.get_flop_counts().get("Global", {})
    total = int(sum(counts.values()))

    report = {
        "schema_version": 1,
        "counter": "torch.utils.flop_counter.FlopCounterMode, Global sum, one forward",
        "why_this_counter": ("it is the counter the organisers' tools/measure_flops.py "
                             "uses; the torch.profiler number recorded in "
                             "rtx4090_mr_forward_cost.json is a different estimator and "
                             "is not what the cutoff is judged on"),
        "forwards_per_clip": 1,
        "scope": ("the whole top-level forward, which internally encodes the six current "
                  "cameras, the four past frames for the scene branch and the five-frame "
                  "motion canvas. Nothing is divided down."),
        "preprocessing_excluded": True,
        "flops": total,
        "gflops": total / 1e9,
        "cutoff_gflops": CUTOFF_GFLOPS,
        "headroom_x": CUTOFF_GFLOPS / (total / 1e9),
        "passes_cutoff": bool(total / 1e9 <= CUTOFF_GFLOPS),
        "by_operator_gflops": {str(k): v / 1e9 for k, v in
                               sorted(counts.items(), key=lambda x: -x[1])},
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
        "checkpoint": str(Path(args.init).resolve()),
        "model_state_sha256": mt.tensor_state_sha256(payload["model"]),
        "measured_on_clip": Path(args.clip).name,
    }

    if args.submission:
        path = Path(args.submission)
        submission = json.loads(path.read_text())
        clips_before = len([k for k in submission if k != "__flops__"])
        submission["__flops__"] = total
        path.write_text(json.dumps(submission))
        report["merged_into"] = str(path.resolve())
        report["clips_preserved"] = clips_before
        report["keys_after_merge"] = len(submission)

    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("flops", "gflops", "headroom_x", "passes_cutoff", "parameters")
                      if k in report}, indent=1))
    if args.submission:
        print(json.dumps({k: report[k] for k in
                          ("clips_preserved", "keys_after_merge")}, indent=1))


if __name__ == "__main__":
    main()
