#!/usr/bin/env python3
"""§8: a single fixed 50:50 weight average of the two MR-NATIVE seeds.

Only learned floating parameters are averaged. Buffers are handled separately:
BN running statistics, counters and the fixed scale/position constants are NOT
blindly averaged. Under the project's fixed BN policy they should be identical
across seeds, which is asserted rather than assumed; if any buffer differs the
script refuses to average it and says so instead of silently picking a rule.

This is one model forward, not an output average, not a GT-based selection.
Averaging across different graphs (LEN, LOWDETAIL, W64) is refused by design.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments/md_r0_reset_20260914")):
    sys.path.insert(0, _p)

import motiondrive_v2_training as mt

OUT = ROOT / "reports/md_exp_diagnosis_20260915/mr_soup.json"


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    args = parser.parse_args()

    a = torch.load(args.a, map_location="cpu", weights_only=False)
    b = torch.load(args.b, map_location="cpu", weights_only=False)

    ca, cb = a["manifest"]["model_config"], b["manifest"]["model_config"]
    if ca != cb:
        raise SystemExit("refusing to average checkpoints with different model configs")
    if a["manifest"].get("split_sha256") != b["manifest"].get("split_sha256"):
        raise SystemExit("refusing to average checkpoints from different split lineages")

    sa, sb = a["model"], b["model"]
    if sorted(sa) != sorted(sb):
        raise SystemExit("refusing to average checkpoints with different tensor sets")

    # Parameter names come from the module tree; everything else in the state
    # dict is a buffer and is treated as one.
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    import matching_resolution as mr
    model = MotionDriveV2(MotionDriveV2Config(**ca))
    mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
    mr.install(model, "native")
    parameter_names = {n for n, _ in model.named_parameters()}
    buffer_names = {n for n, _ in model.named_buffers()}
    unknown = set(sa) - parameter_names - buffer_names
    if unknown:
        raise SystemExit(f"unclassified state entries: {sorted(unknown)[:5]}")

    averaged, kept, refused = {}, [], []
    for name in sa:
        x, y = sa[name], sb[name]
        if name in parameter_names and x.is_floating_point():
            averaged[name] = (x.double() * 0.5 + y.double() * 0.5).to(x.dtype)
        elif torch.equal(x, y):
            averaged[name] = x.clone()
            kept.append(name)
        else:
            refused.append(name)
    if refused:
        raise SystemExit("these buffers differ between the seeds and are not auto-averaged; "
                         "decide the policy explicitly: " + ", ".join(sorted(refused)[:8]))

    integer_parameters = [n for n in parameter_names if not sa[n].is_floating_point()]
    payload = dict(a)
    payload["model"] = averaged
    Path(args.out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_checkpoint)

    report = {
        "schema_version": 1,
        "recipe": "fixed 50:50 average of learned floating parameters; one model forward",
        "sources": {
            "a": {"path": str(Path(args.a).resolve()),
                  "model_state_sha256": mt.tensor_state_sha256(sa)},
            "b": {"path": str(Path(args.b).resolve()),
                  "model_state_sha256": mt.tensor_state_sha256(sb)},
        },
        "graph": "MR-NATIVE (768x432 canvas, radius 4); both seeds share it exactly",
        "tensors": {
            "total": len(sa),
            "averaged_parameters": len(averaged) - len(kept),
            "buffers_identical_and_preserved": len(kept),
            "buffers_differing": len(refused),
            "integer_parameters": integer_parameters,
        },
        "buffer_policy": ("BN running statistics and the fixed scale/position constants were "
                          "bit-identical across seeds under the project's fixed-BN policy, so "
                          "they were preserved, not averaged. Nothing was averaged by default."),
        "preserved_buffers": sorted(kept),
        "output_checkpoint": str(Path(args.out_checkpoint).resolve()),
        "output_model_state_sha256": mt.tensor_state_sha256(averaged),
        "not_claimed": [
            "This is not the main path to a lower score and no combination search follows it.",
            "Averaging across different graphs (LEN, LOWDETAIL, W64) is refused by design.",
            "This is a weight average, not an output average or a GT-based candidate choice.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("tensors", "output_model_state_sha256")}, indent=1))


if __name__ == "__main__":
    main()
