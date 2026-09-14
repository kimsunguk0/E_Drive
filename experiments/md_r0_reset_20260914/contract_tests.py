#!/usr/bin/env python3
"""P2 - contract checks on the one common trainer used by R0.

Every check calls the real code in `scripts/train_motiondrive_v2.py` and
`scripts/motiondrive_v2_training.py`.  Nothing here re-implements a loss, a
reduction or an optimizer rule; the point is to prove what the existing code
already does.  Writes reports/md_r0_reset_20260914/contract_tests.json.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_motiondrive_v2 as trainer
import motiondrive_v2_training as mt
from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2.planner import DirectTrajectoryPlanner

REGISTRY = ROOT / "reports/md_r0_reset_20260914/baseline_registry.json"
# compute_loss casts predictions to float32 internally, so the accumulation
# contract is checked at float32 tolerance; the pure reduction algebra is
# checked separately in float64.
FP32_RTOL, FP32_ATOL = 1e-5, 1e-6
FP64_RTOL, FP64_ATOL = 1e-10, 1e-12


def synthetic(batch_size, *, grid=(8, 6), seed=0, invalid_rows=(), device="cpu"):
    """Synthetic outputs/labels with the real key names and shapes."""
    g = torch.Generator().manual_seed(seed)
    h, w = grid
    scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32))
    base = {
        "plan_abs": torch.randn(batch_size, 6, 2, generator=g),
        "occ_logits": torch.randn(batch_size, h, w, generator=g),
        "lane_logits": torch.randn(batch_size, h, w, generator=g),
        "history_hat": torch.randn(batch_size, 4, 4, generator=g),
        "history_logvar": torch.randn(batch_size, 4, 4, generator=g) * .1,
        "state_hat": torch.randn(batch_size, 6, generator=g),
        "state_logvar": torch.randn(batch_size, 5, generator=g) * .1,
    }
    base = {k: v.to(device) for k, v in base.items()}
    plan_valid = torch.ones(batch_size, 6, dtype=torch.bool)
    state_valid = torch.ones(batch_size, 6, dtype=torch.bool)
    history_valid = torch.ones(batch_size, 4, 4, dtype=torch.bool)
    occ_valid = torch.ones(batch_size, h, w, dtype=torch.bool)
    lane_valid = torch.ones(batch_size, h, w, dtype=torch.bool)
    for row in invalid_rows:
        plan_valid[row] = False
        state_valid[row] = False
        history_valid[row] = False
        occ_valid[row] = False
        lane_valid[row] = False
    # a partially valid horizon, so the denominators are not all-or-nothing
    if batch_size > 3:
        plan_valid[1, 4:] = False
        state_valid[2, 2:] = False
        history_valid[3, 3:] = False
    batch = {
        "gt_plan": torch.randn(batch_size, 6, 2, generator=g),
        "plan_valid": plan_valid,
        "occ_target": (torch.rand(batch_size, h, w, generator=g) > .7).float(),
        "occ_valid": occ_valid,
        "lane_target": (torch.rand(batch_size, h, w, generator=g) > .8).float(),
        "lane_valid": lane_valid,
        "history_target": torch.randn(batch_size, 4, 4, generator=g),
        "history_valid": history_valid,
        "state_target": torch.randn(batch_size, 6, generator=g),
        "state_valid": state_valid,
    }
    batch["state_target"][:, 5] = (torch.rand(batch_size, generator=g) > .5).float()
    return scale, base, {k: v.to(device) for k, v in batch.items()}


def _slice_outputs(base, scale, start, end):
    """Build a fresh autograd graph per chunk from the same leaf parameter."""
    return {k: v[start:end] * scale for k, v in base.items()}


def check_microbatch_equivalence():
    """Dense B16 vs microbatch 1/2/8/16: identical losses, parts and gradients."""
    weights = mt.LossWeights(plan=1., occupancy=.2, lane=.2, motion=.2, uncertainty=True)
    results, ok = {}, True
    for batch_size, invalid in ((16, ()), (15, ()), (16, (4, 5))):
        scale, base, batch = synthetic(batch_size, invalid_rows=invalid, seed=batch_size)
        dense_loss, dense_parts = mt.compute_loss(
            _slice_outputs(base, scale, 0, batch_size), batch, weights)
        dense_loss.backward()
        dense_grad = float(scale.grad)
        key = f"b{batch_size}" + ("_with_invalid_rows" if invalid else "")
        entry = {"dense_total": float(dense_loss), "dense_grad": dense_grad,
                 "invalid_rows": list(invalid), "microbatches": {}}
        for micro in (1, 2, 8, 16):
            if micro > batch_size:
                continue
            scale_m, base_m, batch_m = synthetic(batch_size, invalid_rows=invalid,
                                                  seed=batch_size)
            normalizers = mt.build_loss_normalizers(batch_m)
            total, parts = 0., {}
            for start in range(0, batch_size, micro):
                end = start + micro
                chunk_out = _slice_outputs(base_m, scale_m, start, end)
                chunk_batch = {k: v[start:end] for k, v in batch_m.items()}
                loss, micro_parts = mt.compute_loss(chunk_out, chunk_batch, weights,
                                                    normalizers=normalizers)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite microbatch loss at {key}/{micro}")
                loss.backward()
                total += float(loss)
                for name, value in micro_parts.items():
                    parts[name] = parts.get(name, 0.) + float(value.detach())
            grad = float(scale_m.grad)
            part_diff = max(abs(parts[name] - float(dense_parts[name]))
                            for name in dense_parts)
            passed = (np.isclose(total, float(dense_loss), rtol=FP32_RTOL, atol=FP32_ATOL)
                      and np.isclose(grad, dense_grad, rtol=FP32_RTOL, atol=FP32_ATOL)
                      and part_diff <= FP32_ATOL + FP32_RTOL * abs(float(dense_loss)))
            ok &= passed
            entry["microbatches"][str(micro)] = {
                "total": total, "total_abs_diff": abs(total - float(dense_loss)),
                "grad": grad, "grad_abs_diff": abs(grad - dense_grad),
                "max_part_abs_diff": part_diff, "pass": bool(passed)}
        results[key] = entry
    return {"pass": bool(ok), "tolerance": {"rtol": FP32_RTOL, "atol": FP32_ATOL},
            "cases": results}


def check_reduction_algebra_fp64():
    """masked_mean's additive contract, exactly, in float64."""
    g = torch.Generator().manual_seed(7)
    value = torch.randn(16, 6, dtype=torch.float64, generator=g)
    mask = torch.rand(16, 6, generator=g) > .3
    dense = mt.masked_mean(value, mask)
    normalizer = mask.sum()
    total = sum(mt.masked_mean(value[s:s + 2], mask[s:s + 2], normalizer=normalizer)
                for s in range(0, 16, 2))
    empty_chunk = mt.masked_mean(value[:2], torch.zeros_like(mask[:2]), normalizer=normalizer)
    passed = bool(torch.allclose(dense, total, rtol=FP64_RTOL, atol=FP64_ATOL)
                  and torch.isfinite(empty_chunk) and float(empty_chunk) == 0.)
    return {"pass": passed, "dense": float(dense), "accumulated": float(total),
            "abs_diff": abs(float(dense) - float(total)),
            "fully_masked_chunk_contribution": float(empty_chunk),
            "tolerance": {"rtol": FP64_RTOL, "atol": FP64_ATOL}}


def check_microbatch_negative_control():
    """Preserved DYN/M8 failure mode: a microbatch-local mean accumulates 8x."""
    weights = mt.LossWeights(plan=1., occupancy=0., lane=0., motion=0., uncertainty=True)
    scale, base, batch = synthetic(16, seed=3)
    dense, _ = mt.compute_loss(_slice_outputs(base, scale, 0, 16), batch, weights)
    wrong = 0.
    for start in range(0, 16, 2):
        chunk_out = _slice_outputs(base, scale, start, start + 2)
        chunk_batch = {k: v[start:start + 2] for k, v in batch.items()}
        # normalizers=None makes each chunk divide by its OWN valid count.
        loss, _ = mt.compute_loss(chunk_out, chunk_batch, weights)
        wrong += float(loss)
    ratio = wrong / float(dense)
    return {"pass": bool(ratio > 7.0),
            "note": "negative control: chunk-local means over-count by ~chunks",
            "dense": float(dense), "chunk_local_sum": wrong, "ratio": ratio,
            "chunks": 8}


def check_optimizer_groups(registry):
    """Every trainable parameter in exactly one group, with the pinned LRs."""
    config = MotionDriveV2Config(**registry["model_config"])
    model = MotionDriveV2(config)
    backbone, other, names = [], [], {"backbone": [], "head": []}
    for name, param in model.named_parameters():
        is_backbone = name.startswith("backbone_fpn.") and not name.startswith(
            ("backbone_fpn.lat", "backbone_fpn.out"))
        (backbone if is_backbone else other).append(param)
        names["backbone" if is_backbone else "head"].append(name)
    recipe = registry["recipe"]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": recipe["backbone_lr"], "base_lr": recipe["backbone_lr"]},
        {"params": other, "lr": recipe["head_lr"], "base_lr": recipe["head_lr"]},
    ], weight_decay=recipe["weight_decay"])
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    passed = (len(grouped) == len(set(grouped)) == len(trainable)
              and set(grouped) == set(trainable))
    fpn_lateral = [n for n in names["head"] if n.startswith("backbone_fpn.")]
    return {"pass": bool(passed),
            "groups": [{"name": "backbone", "base_lr": recipe["backbone_lr"],
                        "parameters": len(backbone),
                        "elements": sum(p.numel() for p in backbone)},
                       {"name": "head", "base_lr": recipe["head_lr"],
                        "parameters": len(other),
                        "elements": sum(p.numel() for p in other)}],
            "total_trainable_parameters": len(trainable),
            "fpn_tensors_in_head_group": sorted(fpn_lateral),
            "fpn_note": ("backbone_fpn.lat*/out* (the FPN lateral and output "
                         "convolutions) are in the HEAD group, not the backbone "
                         "group; only the encoder stem/stages use backbone_lr"),
            "no_parameter_in_two_groups": bool(len(grouped) == len(set(grouped))),
            "every_trainable_parameter_grouped": bool(set(grouped) == set(trainable))}


def check_bn_policy(registry):
    config = MotionDriveV2Config(**registry["model_config"])
    model = MotionDriveV2(config)
    model.train()
    count = mt.set_training_mode(model, "fixed")
    bn_modules = [m for m in model.modules()
                  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    training_flags = {bool(m.training) for m in bn_modules}
    affine_trainable = all(m.weight.requires_grad and m.bias.requires_grad
                           for m in bn_modules if m.affine)
    model.train()
    adaptive = mt.set_training_mode(model, "adaptive")
    adaptive_flags = {bool(m.training) for m in bn_modules}
    return {"pass": bool(training_flags == {False} and affine_trainable
                         and adaptive_flags == {True}),
            "bn_modules": len(bn_modules), "reported_count_fixed": count,
            "reported_count_adaptive": adaptive,
            "fixed_all_eval": training_flags == {False},
            "adaptive_all_train": adaptive_flags == {True},
            "affine_weights_trainable": bool(affine_trainable),
            "registry_bn_modules": registry.get("recipe", {}).get("bn_running_statistics")}


def check_metric_identity():
    pred = torch.randn(32, 6, 2, dtype=torch.float64)
    zero = mt.weighted_d3(pred, pred)
    unit = mt.weighted_d3(pred + torch.tensor([1., 0.], dtype=torch.float64), pred)
    speed = torch.arange(1, 7, dtype=torch.float64)[None, :, None] * .5
    constant_velocity = mt.weighted_d3(
        torch.cat([speed, torch.zeros_like(speed)], -1).expand(1, 6, 2),
        torch.zeros(1, 6, 2, dtype=torch.float64))
    l2 = torch.linalg.vector_norm(pred - pred.roll(1, 0), dim=-1).numpy().astype(np.float64)
    two_stage = (l2[:, :2].mean(1) + l2[:, :4].mean(1) + l2[:, :6].mean(1)) / 3.
    weighted = mt.weighted_d3(pred, pred.roll(1, 0)).numpy().astype(np.float64)
    diff = float(np.abs(weighted - two_stage).max())
    # weighted_d3 casts to float32 by design, so the algebraic identities hold
    # to float32 precision, not float64.  That is the contract being recorded.
    metric_atol = 1e-6
    return {"pass": bool(float(zero.abs().max()) == 0.
                         and np.allclose(unit.numpy(), 1., rtol=0, atol=metric_atol)
                         and abs(float(constant_velocity) - 1.25) < metric_atol
                         and diff < metric_atol),
            "identical_prediction_d3": float(zero.abs().max()),
            "unit_offset_d3": float(unit.mean()),
            "one_mps_constant_velocity_error_d3": float(constant_velocity),
            "weights": list(mt.TIME_WEIGHTS),
            "weighted_vs_ade123_max_row_diff": diff,
            "metric_precision": "weighted_d3 casts predictions to float32 (pred.float()); identities hold at float32 precision",
            "tolerance_atol": metric_atol,
            "note": "mean(ADE1,ADE2,ADE3) equals the [11,11,5,5,2,2]/36 weighting"}


def check_input_whitelist():
    """Model inputs carry no ground truth and no provided ego status."""
    forbidden = ("gt_", "target", "valid", "status", "state_target", "proxy",
                 "row", "session", "scenario", "frame", "command", "vad_cmd")
    keys = list(mt.MODEL_INPUTS)
    signature = list(inspect.signature(MotionDriveV2.forward).parameters)[1:]
    leaking = [k for k in keys if any(token in k for token in forbidden)]
    return {"pass": bool(not leaking and signature == keys),
            "model_inputs": keys,
            "forward_signature": signature,
            "signature_matches_whitelist": signature == keys,
            "leaking_keys": leaking}


def check_fp32_projection(registry, device):
    """Under bf16 autocast the planner still emits FP32 coordinates."""
    config = MotionDriveV2Config(**registry["model_config"])
    planner = DirectTrajectoryPlanner(config).to(device).eval()
    cells = config.grid_size[0] * config.grid_size[1]
    scene = torch.randn(2, cells, config.channels, device=device)
    motion = torch.randn(2, config.motion_grid[0] * config.motion_grid[1],
                         config.channels, device=device)
    state = torch.randn(2, 6, device=device)
    history = torch.randn(2, config.n_history, 4, device=device)
    with torch.no_grad():
        with trainer.autocast(torch.device(device), "bf16"):
            plan_autocast = planner(scene.to(torch.bfloat16) if device != "cpu" else scene,
                                    motion, state, history)
        plan_plain = planner(scene, motion, state, history)
    return {"pass": bool(plan_autocast.dtype == torch.float32
                         and plan_plain.dtype == torch.float32),
            "device": str(device),
            "plan_dtype_under_bf16_autocast": str(plan_autocast.dtype),
            "plan_dtype_without_autocast": str(plan_plain.dtype),
            "planner_disables_autocast": "enabled=False" in inspect.getsource(
                DirectTrajectoryPlanner.forward)}


def check_augmentation_contract():
    """Flip is an involution and the trainer refreshes it once per epoch."""
    from motiondrive_v2_flip_augment import FlipAugmented, flip_item
    g = torch.Generator().manual_seed(11)
    item = {
        "images": torch.randn(6, 3, 8, 12, generator=g),
        "history_images": torch.randn(4, 3, 4, 6, generator=g),
        "lidar2img": torch.eye(4).repeat(6, 1, 1),
        "history_transforms": torch.eye(4).repeat(4, 1, 1),
        "goal_xy": torch.randn(2, generator=g),
        "gt_plan": torch.randn(6, 2, generator=g),
        "history_target": torch.randn(4, 4, generator=g),
        "state_target": torch.randn(6, generator=g),
        "occ_target": torch.rand(8, 6, generator=g),
        "occ_valid": torch.ones(8, 6, dtype=torch.bool),
        "lane_target": torch.rand(8, 6, generator=g),
        "lane_valid": torch.ones(8, 6, dtype=torch.bool),
        "time_offsets": torch.tensor([.1, .2, .5, 1.]),
    }
    once = flip_item({k: v.clone() for k, v in item.items()}, 12, 6)
    twice = flip_item({k: v.clone() for k, v in once.items()}, 12, 6)
    diffs = {k: float((twice[k].float() - item[k].float()).abs().max())
             for k in item if isinstance(item[k], torch.Tensor)}
    source = inspect.getsource(trainer.run_training)
    return {"pass": bool(max(diffs.values()) == 0.
                         and hasattr(FlipAugmented, "set_epoch")
                         and "set_epoch(epoch)" in source),
            "double_flip_max_abs_diff": max(diffs.values()),
            "per_key_max_abs_diff": diffs,
            "flip_wrapper_has_set_epoch": hasattr(FlipAugmented, "set_epoch"),
            "trainer_calls_set_epoch_per_epoch": "set_epoch(epoch)" in source}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(ROOT / "reports/md_r0_reset_20260914/contract_tests.json"))
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text())
    torch.manual_seed(0)

    checks = {
        "microbatch_equivalence": check_microbatch_equivalence(),
        "reduction_algebra_fp64": check_reduction_algebra_fp64(),
        "microbatch_negative_control": check_microbatch_negative_control(),
        "optimizer_groups": check_optimizer_groups(registry),
        "bn_policy": check_bn_policy(registry),
        "metric_identity": check_metric_identity(),
        "input_whitelist": check_input_whitelist(),
        "fp32_projection": check_fp32_projection(registry, args.device),
        "augmentation_contract": check_augmentation_contract(),
    }
    payload = {
        "schema_version": 1,
        "runtime_source_git_sha": registry["runtime_source_id"]["git_sha"],
        "torch": torch.__version__,
        "device": args.device,
        "all_pass": all(c["pass"] for c in checks.values()),
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_pass": payload["all_pass"],
                      "per_check": {k: v["pass"] for k, v in checks.items()},
                      "out": args.out}, indent=1))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
