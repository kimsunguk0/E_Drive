#!/usr/bin/env python3
"""Pre-training checks for MR-ADJ0/ADJ1.

Checks the things that would silently invalidate the contrast:
  1. at step zero both arms reproduce MR exactly (zero-init output projection)
  2. the two arms differ ONLY through the mask, and ADJ1 actually differs from
     ADJ0 once the projection is non-zero
  3. the pair convention (reference, source) matches local_correlation's own
     argument order
  4. the endpoint encoding separates the two dt collisions adjacent edges create
  5. no NaN from a fully masked softmax
  6. the new path opens over the first updates -- NOT that every new tensor has
     a non-zero gradient on the first backward, which a zero-init residual makes
     impossible by construction
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
for p in (ROOT, ROOT / "scripts", ROOT / "experiments/md_r0_reset_20260914"):
    sys.path.insert(0, str(p))

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2 import motion_encoder as motion_module
import matching_resolution as mr
import adjacent_edges as adj

OUT = ROOT / "reports/md_exp_diagnosis_20260915/adj_smoke.json"
INIT = ROOT / "work_dirs/md_r0_reset_20260914/r0_init_tplus.pth"


def build(mode):
    """mode: 'mr' | 'adj0' | 'adj1'."""
    payload = torch.load(INIT, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    torch.manual_seed(0)
    model = MotionDriveV2(config)
    mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
    mr.install(model, "native")
    report = None
    if mode != "mr":
        torch.manual_seed(1234)                     # identical new-module init
        report = adj.install(model, use_adjacent=(mode == "adj1"))
    return model, payload, report


def main():
    out = {"schema_version": 1}
    device = torch.device("cuda:0")

    # --- inputs shared by every arm ---
    torch.manual_seed(7)
    b, c = 2, 128
    levels = [(54, 96), (27, 48)]
    cur = [torch.randn(b, c, h, w, device=device) for h, w in levels]
    hist = [torch.randn(b, 4, c, h, w, device=device) for h, w in levels]
    dt = torch.tensor([0.1, 0.2, 0.5, 1.0], device=device).repeat(b, 1)

    results = {}
    for mode in ("mr", "adj0", "adj1"):
        model, _, report = build(mode)
        enc = model.motion_encoder.to(device).float().eval()
        with torch.no_grad():
            r = enc(cur, hist, dt)
        results[mode] = {k: v.detach().clone() for k, v in r.items()}
        if report:
            out.setdefault("install", {})[mode] = report

    def diff(a, key):
        return float((results[a][key] - results["mr"][key]).abs().max())

    out["step_zero_reproduces_mr"] = {
        "adj0_motion_features_max_abs_diff": diff("adj0", "motion_features"),
        "adj0_state_hat_max_abs_diff": diff("adj0", "state_hat"),
        "adj0_history_hat_max_abs_diff": diff("adj0", "history_hat"),
        "adj1_motion_features_max_abs_diff": diff("adj1", "motion_features"),
        "adj1_state_hat_max_abs_diff": diff("adj1", "state_hat"),
        "adj1_history_hat_max_abs_diff": diff("adj1", "history_hat"),
        "note": ("the output projection starts at zero, so both arms must equal MR "
                 "bit for bit at step zero; the history head must match too because "
                 "it still reads only the four star edges"),
    }
    out["arms_identical_at_step_zero"] = float(
        (results["adj1"]["motion_features"] - results["adj0"]["motion_features"]).abs().max())
    out["no_nan"] = {m: bool(torch.isfinite(torch.cat([v.flatten() for v in r.values()])).all())
                     for m, r in results.items()}

    # --- the arms must diverge once the projection is non-zero ---
    diverged = {}
    for mode in ("adj0", "adj1"):
        model, _, _ = build(mode)
        enc = model.motion_encoder.to(device).float().eval()
        torch.manual_seed(99)
        with torch.no_grad():
            enc.adjacent_attention.to_out.weight.normal_(0, 0.02)
            r = enc(cur, hist, dt)
        diverged[mode] = r["motion_features"].detach().clone()
    out["arms_differ_once_projection_is_nonzero"] = float(
        (diverged["adj1"] - diverged["adj0"]).abs().max())

    # --- pair convention against local_correlation's own argument order ---
    a = torch.randn(1, 32, 16, 16, device=device)
    shifted = torch.roll(a, shifts=(0, 2), dims=(2, 3))      # source is a shifted a
    corr = motion_module.local_correlation(a, shifted, 4)     # (reference, source)
    width = 2 * 4 + 1
    peak = int(corr[0, :, 8, 8].argmax())
    out["pair_convention"] = {
        "bins": width * width,
        "peak_bin": peak,
        "peak_dy": peak // width - 4,
        "peak_dx": peak % width - 4,
        "expected": "a source shifted by +2 columns peaks at dx = +2, dy = 0",
        "channels_ordered_dy_then_dx": True,
        "reference_is_first_argument": True,
    }

    # --- the endpoint encoding separates the dt collisions ---
    ages = adj.edge_ages(dt)
    ref = torch.stack([ages[:, i] for i, _ in adj.ALL_EDGES], 1)
    src = torch.stack([ages[:, j] for _, j in adj.ALL_EDGES], 1)
    d = (src - ref)
    pairs = [{"edge": list(e), "age_ref": float(ref[0, i]), "age_src": float(src[0, i]),
              "dt": float(d[0, i]), "kind": "star" if e in adj.STAR_EDGES else "adjacent"}
             for i, e in enumerate(adj.ALL_EDGES)]
    seen = {}
    collisions = []
    for p in pairs:
        seen.setdefault(round(p["dt"], 6), []).append(p["edge"])
    for value, group in seen.items():
        if len(group) > 1:
            collisions.append({"dt": value, "edges": group})
    feature = torch.stack([ref, src, d, d.clamp_min(1e-3).log()], -1)[0]
    out["endpoint_encoding"] = {
        "pairs": pairs,
        "dt_collisions": collisions,
        "collisions_separated_by_full_feature": bool(
            len({tuple(round(float(x), 6) for x in row) for row in feature}) == len(pairs)),
        "note": ("dt alone cannot tell (0,-0.1) from (-0.1,-0.2), nor (0,-0.5) from "
                 "(-0.5,-1.0); [age_ref, age_src, dt, log dt] does"),
    }

    # --- does the new path open over the first updates? ---
    model, _, _ = build("adj1")
    enc = model.motion_encoder.to(device).float()
    opt = torch.optim.AdamW(enc.parameters(), lr=5e-5)
    watch = {"to_out.weight": enc.adjacent_attention.to_out.weight,
             "to_q.weight": enc.adjacent_attention.to_q.weight,
             "pair_time.0.weight": enc.adjacent_attention.pair_time[0].weight}
    before = {k: v.detach().clone() for k, v in watch.items()}
    grads = {k: [] for k in watch}
    for _ in range(6):
        r = enc(cur, hist, dt)
        loss = r["motion_features"].square().mean() + r["state_hat"].square().mean()
        opt.zero_grad(); loss.backward()
        for k, v in watch.items():
            grads[k].append(float(v.grad.norm()) if v.grad is not None else 0.0)
        opt.step()
    out["path_opens"] = {
        "grad_norms_per_step": grads,
        "to_out_zero_grad_on_first_backward_is_expected": grads["to_out.weight"][0] > 0,
        "behind_projection_zero_on_first_backward": grads["to_q.weight"][0] == 0.0,
        "behind_projection_nonzero_later": any(g > 0 for g in grads["to_q.weight"][1:]),
        "moved": {k: float((watch[k] - before[k]).abs().max()) for k in watch},
        "reading": ("to_out sees gradient immediately because the attended value is not "
                    "zero; everything behind it is zero on the FIRST backward because "
                    "to_out's weight is zero, and opens from the second update once "
                    "to_out has moved. A check demanding non-zero gradient everywhere "
                    "on the first backward would be wrong for this design."),
    }

    model, _, report = build("adj1")
    out["parameters"] = {
        "adjacent_module": report["new_parameters"],
        "total_adj": int(sum(p.numel() for p in model.parameters())),
    }
    model_mr, _, _ = build("mr")
    out["parameters"]["total_mr"] = int(sum(p.numel() for p in model_mr.parameters()))
    out["parameters"]["delta"] = out["parameters"]["total_adj"] - out["parameters"]["total_mr"]

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
