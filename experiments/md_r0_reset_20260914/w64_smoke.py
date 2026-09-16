#!/usr/bin/env python3
"""Pre-training checks for MR-W64: graph, exact reinit set, gradient reachability,
and the initialisation-time correlation magnitude shift the wider descriptor causes."""
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

OUT = ROOT / "reports/md_exp_diagnosis_20260915/w64_smoke.json"
INIT = ROOT / "work_dirs/md_r0_reset_20260914/r0_init_tplus.pth"


def sha(state):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build(width):
    payload = torch.load(INIT, map_location="cpu", weights_only=False)
    config = MotionDriveV2Config(**payload["manifest"]["model_config"])
    model = MotionDriveV2(config)
    rebuild = mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
    widening = mr.rebuild_descriptor(model, width) if width else None
    mr.install(model, "native")
    return model, payload["model"], rebuild, widening


def main():
    out = {"schema_version": 1}
    device = torch.device("cuda:0")

    # --- 1. graph shapes at both widths ---
    shapes = {}
    for name, width in (("MR-NATIVE (control)", None), ("MR-W64", 64)):
        model, r0, rebuild, widening = build(width)
        enc = model.motion_encoder
        shapes[name] = {
            "correlation_channels": enc.config.correlation_channels,
            "correlation_radius": enc.config.correlation_radius,
            "bins_per_level": (2 * enc.config.correlation_radius + 1) ** 2,
            "projection_out_channels": [p.out_channels for p in enc.projections],
            "fuse_in_channels": enc.correlation_fuse[0].in_channels,
            "motion_grid": list(enc.config.motion_grid),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "widening": widening,
        }
    out["graph"] = shapes
    out["parameter_delta"] = (shapes["MR-W64"]["total_parameters"]
                              - shapes["MR-NATIVE (control)"]["total_parameters"])

    # --- 2. exactly the intended tensors are new; every other tensor is R0 ---
    model, r0, rebuild, widening = build(64)
    state = dict(r0)
    prefixes = ["motion_encoder.correlation_fuse.0."] + [p + "." for p in widening["projections"]]
    dropped = sorted(k for k in state if any(k.startswith(x) for x in prefixes))
    for k in dropped:
        state.pop(k)
    inc = model.load_state_dict(state, strict=False)
    loaded = {n: v for n, v in model.state_dict().items() if n not in set(dropped)}
    reference = {n: v for n, v in r0.items() if n not in set(dropped)}
    out["reinitialisation"] = {
        "dropped": dropped,
        "missing_keys_match": sorted(inc.missing_keys) == dropped,
        "no_unexpected_keys": not inc.unexpected_keys,
        "every_other_tensor_bitwise_r0": sha(loaded) == sha(reference),
        "r0_sha256": sha(r0),
        "measured_step_zero_sha256": sha(model.state_dict()),
        "step_zero_parity_with_mr_native": False,
    }

    # --- 3. gradient reachability of the NEW channels ---
    # Zero-padding both sides of a cosine correlation would give an identically
    # zero product whose gradient is also zero. Fresh init should not; verify it
    # on the real module over several updates, not one backward.
    enc = model.motion_encoder.to(device).float()
    b, t, c = 2, enc.config.n_history, enc.config.channels
    levels = [(54, 96), (27, 48)]
    opt = torch.optim.AdamW(enc.parameters(), lr=5e-5)
    new_params = {"projections.0.weight": enc.projections[0].weight,
                  "projections.1.weight": enc.projections[1].weight,
                  "correlation_fuse.0.weight": enc.correlation_fuse[0].weight}
    before = {k: v.detach().clone() for k, v in new_params.items()}
    grad_norms = {k: [] for k in new_params}
    torch.manual_seed(0)
    for step in range(5):
        cur = [torch.randn(b, c, h, w, device=device) for h, w in levels]
        hist = [torch.randn(b, t, c, h, w, device=device) for h, w in levels]
        dt = torch.tensor(enc.config.nominal_history_seconds, device=device).repeat(b, 1)
        res = enc(cur, hist, dt)
        loss = res["motion_features"].square().mean() + res["state_hat"].square().mean()
        opt.zero_grad(); loss.backward()
        for k, v in new_params.items():
            grad_norms[k].append(float(v.grad.norm()))
        opt.step()
    # Split the projection weight into the channels MR already had (0..31) and
    # the channels this experiment adds (32..63): both must actually move.
    p0 = enc.projections[0].weight
    moved_old = float((p0[:32] - before["projections.0.weight"][:32]).abs().max())
    moved_new = float((p0[32:] - before["projections.0.weight"][32:]).abs().max())
    out["gradient_reachability"] = {
        "grad_norms_per_step": grad_norms,
        "all_grad_norms_positive": all(g > 0 for v in grad_norms.values() for g in v),
        "projection0_max_abs_change_channels_0_31": moved_old,
        "projection0_max_abs_change_channels_32_63": moved_new,
        "new_channels_learn": moved_new > 0,
        "fuse_max_abs_change": float((enc.correlation_fuse[0].weight
                                      - before["correlation_fuse.0.weight"]).abs().max()),
    }

    # --- 4. the 1/sqrt(C) magnitude shift, measured on the real operator ---
    torch.manual_seed(1)
    mag = {}
    for C in (32, 64):
        a = torch.randn(4, C, 27, 48, device=device)
        h = torch.randn(4, C, 27, 48, device=device)
        corr = motion_module.local_correlation(a, h, 4)
        mag[f"C={C}"] = {"abs_mean": float(corr.abs().mean()),
                         "std": float(corr.std()),
                         "min": float(corr.min()), "max": float(corr.max())}
    mag["ratio_abs_mean_32_over_64"] = mag["C=32"]["abs_mean"] / mag["C=64"]["abs_mean"]
    mag["sqrt2_for_reference"] = 2 ** 0.5
    mag["bins_stay_bounded_in_[-1,1]"] = all(
        v["min"] >= -1.0000001 and v["max"] <= 1.0000001
        for k, v in mag.items() if isinstance(v, dict))
    out["correlation_magnitude"] = mag

    # --- 5. does GroupNorm downstream absorb it? compare fuse output stats ---
    stats = {}
    for name, width in (("MR-NATIVE (control)", None), ("MR-W64", 64)):
        m, _, _, _ = build(width)
        e = m.motion_encoder.to(device).float()
        torch.manual_seed(2)
        cur = [torch.randn(b, c, h, w, device=device) for h, w in levels]
        hist = [torch.randn(b, t, c, h, w, device=device) for h, w in levels]
        dt = torch.tensor(e.config.nominal_history_seconds, device=device).repeat(b, 1)
        with torch.no_grad():
            r = e(cur, hist, dt)
        stats[name] = {"motion_features_std": float(r["motion_features"].std()),
                       "motion_features_abs_mean": float(r["motion_features"].abs().mean()),
                       "state_hat_abs_mean": float(r["state_hat"].abs().mean())}
    out["post_groupnorm_statistics"] = stats

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
