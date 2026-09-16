#!/usr/bin/env python3
"""E1 - existing train (T203) versus expanded train (Tplus), same R0, same budget.

The only difference between the two arms is which scenes the training dataset
draws from.  Model, loss, optimizer, schedule, augmentation, seed, evaluation
set and compute budget are identical, and both arms start from the SAME
rebound R0 initializer.

Training itself is `scripts/train_motiondrive_v2.py`; this file only pins the
protocol, installs the flip augmentation on the training dataset and keeps a
checkpoint at every planned evaluation point so §7.4's best-on-V0 selection is
possible after the fact.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/md_r0_reset_20260914"))

from build_grouped_split_v2 import sha256
from motiondrive_v2_training import tensor_state_sha256

NAME = "md_r0_reset_e1_data_scale"
BASE_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"
NEW_SPLIT = ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json"
SUPERVISION = ROOT / "data/etri/motiondrive_v2/r0reset_tplus_geometry_v2"
INIT = ROOT / "work_dirs/md_r0_reset_20260914/r0_init_tplus.pth"
REGISTRY = ROOT / "reports/md_r0_reset_20260914/baseline_registry.json"

ARMS = ("E1-T203", "E1-EXP", "E1-EXP-LONG", "E1-EXP-LEN",
        "MR-NATIVE", "MR-LOWDETAIL", "MR-W64", "MR-ADJ0", "MR-ADJ1")
EXPANDED_ARMS = ("E1-EXP", "E1-EXP-LONG", "E1-EXP-LEN",
                 "MR-NATIVE", "MR-LOWDETAIL", "MR-W64", "MR-ADJ0", "MR-ADJ1")
# The matching-resolution pair: the motion branch matches on a 768x432 canvas at
# radius 4, which keeps the original search reach in original-image pixels while
# quantising it four times more finely. The two arms differ ONLY in whether the
# canvas kept its native detail or went through a 384x216 bottleneck first.
MR_DETAIL = {"MR-NATIVE": "native", "MR-LOWDETAIL": "lowdetail",
             "MR-W64": "native", "MR-ADJ0": "native", "MR-ADJ1": "native"}
# MR-W64 keeps the MR-NATIVE graph exactly -- native 768x432 canvas, radius 4,
# 81 offsets at both levels, unchanged pooling, unchanged 128D planner
# interface -- and widens the matching descriptor alone.
MR_DESCRIPTOR = {"MR-W64": 64}
# MR-ADJ: three comparisons between neighbouring past frames, added to the four
# existing t0-anchored ones, over the SAME five images. The two arms carry the
# identical extra module and differ only in whether its mask lets the three new
# edges be read, so the contrast isolates the information and not the capacity.
ADJACENT_EDGES_ON = {"MR-ADJ0": False, "MR-ADJ1": True}
# Both arms build the new module from this one seed, so their new parameters
# start identical and the mask is the only difference at step zero.
ADJACENT_INIT_SEED = 20260917
# Interval-length auxiliary weight, added beside the unchanged D3 loss.
# 0.25 is a starting value on a mean L1 in metres, not a tuned or guaranteed one.
LENGTH_LAMBDA = {"E1-EXP-LEN": 0.25, "MR-NATIVE": 0.25, "MR-LOWDETAIL": 0.25,
                 "MR-W64": 0.25, "MR-ADJ0": 0.25, "MR-ADJ1": 0.25}
N0_ROWS = 54810                                   # existing train split
TPLUS_ROWS = 83700                                # expanded train split
BATCH = 16
MICROBATCH = 2
# E1 held compute fixed at <= 6 exposures of the EXISTING train split.
# The LONG arm instead gives the EXPANDED split the same 6 exposures, so its
# cosine horizon is longer from step one rather than being appended later.
ARM_UPDATES = {
    "E1-T203": math.ceil(6 * N0_ROWS / BATCH),    # 20554
    "E1-EXP": math.ceil(6 * N0_ROWS / BATCH),     # 20554
    "E1-EXP-LONG": math.ceil(6 * TPLUS_ROWS / BATCH),   # 31388
    "E1-EXP-LEN": math.ceil(6 * N0_ROWS / BATCH),       # 20554, matching E1-EXP exactly
    "MR-NATIVE": math.ceil(6 * N0_ROWS / BATCH),        # 20554
    "MR-LOWDETAIL": math.ceil(6 * N0_ROWS / BATCH),     # 20554
    "MR-W64": math.ceil(6 * N0_ROWS / BATCH),           # 20554, matching MR-NATIVE
    "MR-ADJ0": math.ceil(6 * N0_ROWS / BATCH),          # 20554, matching MR-NATIVE
    "MR-ADJ1": math.ceil(6 * N0_ROWS / BATCH),          # 20554, matching MR-NATIVE
}
# Both MR arms carry the length auxiliary, so the pair differs in detail alone.
LENGTH_LAMBDA_MR = 0.25
EVAL_EVERY = math.ceil(N0_ROWS / BATCH)           # 3426, one T203 exposure
WARMUP = 200
FLIP_P = 0.5
FLIP_WIDTHS = (768, 384)
BACKBONE_LR = 5e-6
HEAD_LR = 5e-5


def rows_sha(rows) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def build_datasets(arm, seed):
    from motiondrive_v2_data import MotionDriveDataset
    base = json.loads(BASE_SPLIT.read_text())
    common = dict(data_root="/tmp/pm97", split_manifest=str(NEW_SPLIT),
                  supervision_root=str(SUPERVISION), min_frame=30, max_samples=0,
                  seed=seed, history_contract="control")
    scenes = None if arm in EXPANDED_ARMS else sorted(base["splits"]["train"])
    training = MotionDriveDataset(split="train", frame_stride=1, augment=False,
                                  scenes=scenes, **common)
    evaluation = MotionDriveDataset(split="tune", frame_stride=5, augment=False, **common)
    return scenes, training, evaluation


def experiment_rows(dataset):
    return len(dataset)


def build_experiment(arm, seed, scenes, training, evaluation):
    updates = ARM_UPDATES[arm]
    registry = json.loads(REGISTRY.read_text())
    payload = torch.load(INIT, map_location="cpu", weights_only=False)
    initial_sha = tensor_state_sha256(payload["model"])
    if initial_sha != registry["model_state_sha256"]:
        raise SystemExit("initializer tensors differ from the pinned R0 registry")
    return {
        "schema_version": 1, "name": NAME, "arm": arm, "seed": seed,
        "question": "does the same R0 recipe improve V0 when more same-domain sessions are added?",
        "initializer": {"path": str(INIT), "checkpoint_sha256": sha256(INIT),
                        "model_state_sha256": initial_sha,
                        "source_run": registry["run_name"],
                        "source_checkpoint_sha256": registry["checkpoint_sha256"]},
        "split_manifest": {"path": str(NEW_SPLIT), "sha256": sha256(NEW_SPLIT)},
        "supervision": {"path": str(SUPERVISION),
                        "sha256": sha256(SUPERVISION / "supervision_manifest.json")},
        "train_scenes": scenes,
        "train_data": {"rows": len(training), "rows_sha256": rows_sha(training.rows),
                       "scenes": len(set(training.scene_names[training.rows]))},
        "tune_data": {"rows": len(evaluation), "rows_sha256": rows_sha(evaluation.rows),
                      "scenes": len(set(evaluation.scene_names[evaluation.rows]))},
        "expected_initial_model_state_sha256": initial_sha,
        "expected_optimizer_groups": [{"name": "backbone", "base_lr": BACKBONE_LR},
                                      {"name": "head", "base_lr": HEAD_LR}],
        "recipe": {"updates": updates, "eval_every": EVAL_EVERY, "warmup": WARMUP,
                   "batch": BATCH, "microbatch": MICROBATCH, "eval_batch": 4,
                   "optimizer": "fresh AdamW", "backbone_lr": BACKBONE_LR,
                   "head_lr": HEAD_LR, "weight_decay": .01, "grad_clip": 5.0,
                   "decay": "cosine", "cosine_horizon_updates": updates,
                   "precision": "bf16", "time_input": "nominal",
                   "bn_running_statistics": "fixed",
                   "auxiliary_weights": {"occupancy": .2, "lane": .2, "motion": .2},
                   "flip_probability": FLIP_P, "command_input": "off",
                   "provided_status_used": False,
                   "interval_length_auxiliary_lambda": LENGTH_LAMBDA.get(arm, 0.0)},
        "exposures_of_own_train": updates * BATCH / experiment_rows(training),
        "budget_note": (
            "E1's two arms share one update count, so the expanded arm sees each of its "
            "samples fewer times: a fixed-compute data-scale contrast, not an equal-epoch "
            "one. The LONG arm instead gives the expanded split the same six exposures, "
            "with the cosine horizon set to its own total from step one. Comparing LONG "
            "with the short expanded run is therefore a schedule-and-budget comparison, "
            "not the causal effect of exposure count alone."),
        "interval_length_auxiliary": ({
            "lambda": LENGTH_LAMBDA[arm],
            "loss": ("sum_n complete[n] * sum_k |ell_pred - ell_gt| / (6 * plan_complete), "
                     "with plan_complete the same full-effective-batch denominator the D3 "
                     "loss uses"),
            "added_to": "the existing total loss; the D3 and auxiliary weights are unchanged",
            "never_replaces_d3": ("an interval norm carries no direction, so a mirrored path "
                                  "scores identically under this term alone"),
            "contract": "reports/md_exp_diagnosis_20260915/length_auxiliary_contract.json",
        } if arm in LENGTH_LAMBDA else None),
        "matching_resolution": ({
            "detail": MR_DETAIL[arm],
            "motion_canvas_wh": [768, 432],
            "correlation_radius": 4,
            "bins_per_level": 81,
            "search_halfwidth_original_px": {"fine": 32, "coarse": 64},
            "unchanged_reach": ("radius 4 on the 768x432 canvas reproduces the original "
                                "fine +/-32 and coarse +/-64 original-image pixels, so only "
                                "the quantisation changes, not the reach"),
            "scene_branch": "untouched: its own backbone pass on the original 384x216 stack",
            "lowdetail_recipe": ("768 -> 384 -> 768 with PIL bilinear, inside the loader's own "
                                 "resize step, before jitter and normalisation"),
            "reinitialised_module": "motion_encoder.correlation_fuse.0 (178 -> 290 in-channels)",
            "step_zero_parity_with_len": False,
            "smoke": "reports/md_exp_diagnosis_20260915/mr_smoke.json",
            "correlation_channels": MR_DESCRIPTOR.get(arm, 32),
            "adjacent_edges": ({
                "control": "MR-ADJ0 (the same module, the three new edges masked out)",
                "changed": "the three adjacent-past comparisons become readable",
                "timestamps_s": [0, -0.1, -0.2, -0.5, -1.0],
                "star_edges": [[0, -0.1], [0, -0.2], [0, -0.5], [0, -1.0]],
                "adjacent_edges": [[-0.1, -0.2], [-0.2, -0.5], [-0.5, -1.0]],
                "pair_convention": "(reference, source); the source is the older frame, "
                                   "matching local_correlation's argument order",
                "new_observations": ("none: the same five images are read; only the number "
                                     "of relations computed between them changes"),
                "endpoint_encoding": ("[age_ref, age_src, dt, log dt] instead of [dt, log dt], "
                                      "applied identically in both arms, because dt alone "
                                      "cannot separate (0,-0.1) from (-0.1,-0.2) nor "
                                      "(0,-0.5) from (-0.5,-1.0)"),
                "star_path": "preserved bit for bit; the module contributes a residual",
                "output_projection_init": "zeros, so step zero reproduces MR exactly",
                "readouts": ("history still reads the four star edges and keeps its four "
                             "existing targets; state and the planner consume the residual-"
                             "updated motion feature"),
                "masked_softmax_safety": "star edges are never masked, so no query sees an "
                                         "all-masked memory",
                "use_adjacent": ADJACENT_EDGES_ON[arm],
                "new_module_seed": ADJACENT_INIT_SEED,
                "smoke": "reports/md_exp_diagnosis_20260915/adj_smoke.json",
            } if arm in ADJACENT_EDGES_ON else None),
            "descriptor_widening": ({
                "control": "MR-NATIVE (correlation_channels 32)",
                "changed": "correlation_channels 32 -> %d" % MR_DESCRIPTOR[arm],
                "unchanged": ("canvas 768x432, radius 4, 81 offsets at both levels, "
                              "post-correlation pooling to 12x16, temporal layers, the "
                              "128D planner interface, the scene branch and goal path, "
                              "the state/history heads and their GT definitions, Tplus, "
                              "the loss set with length lambda 0.25, augmentation and BN"),
                "reinitialised": ("motion_encoder.projections.0/1 and "
                                  "motion_encoder.correlation_fuse.0 (290 -> 418 in-channels)"),
                "initialisation": "fresh nn.Conv2d default; no reshape of the 32-channel "
                                  "tensors and no zero-padded channels",
                "correlation_definition": "cosine, unchanged; bins stay bounded in [-1, 1]",
                "init_magnitude_note": ("for uncorrelated features the typical bin magnitude "
                                        "falls as 1/sqrt(C); measured and recorded, not "
                                        "compensated, so the definition stays the one MR used"),
                "step_zero_parity_with_mr_native": False,
                "optimizer_group": "head (5e-5), the same group the 32-channel tensors were in",
            } if arm in MR_DESCRIPTOR else None),
        } if arm in MR_DETAIL else None),
        "checkpoint_selection": "lowest V0 official_d3 among planned eval steps; ties go to the earlier step",
        "provided_status_used": False,
    }


def trainer_argv(arm, seed, run_dir, scenes, gpu):
    command = [
        "--data-root", "/tmp/pm97",
        "--split-manifest", str(NEW_SPLIT),
        "--supervision-root", str(SUPERVISION),
        "--run-dir", str(Path(run_dir).resolve()),
        "--phase", "joint", "--goal-on", "1", "--state-on", "1",
        "--cross-cell-goal-mode", "zero", "--history-contract", "control",
        "--gpu", str(gpu), "--seed", str(seed), "--steps", str(ARM_UPDATES[arm]),
        "--batch", str(BATCH), "--microbatch", str(MICROBATCH), "--eval-batch", "4",
        "--workers", "4", "--eval-every", str(EVAL_EVERY),
        "--save-every", str(EVAL_EVERY), "--log-every", "50",
        "--lr", str(HEAD_LR), "--backbone-lr", str(BACKBONE_LR),
        "--weight-decay", "0.01", "--warmup", str(WARMUP), "--grad-clip", "5",
        "--alpha-occ", "0.2", "--alpha-lane", "0.2", "--alpha-motion", "0.2",
        "--uncertainty", "1", "--precision", "bf16", "--time-input", "nominal",
        "--bn-policy", "fixed", "--init", str(INIT),
        "--arch", "resnet50", "--motion-input-mode", "low_feature",
        "--train-stride", "1", "--eval-stride", "5",
        "--max-train-samples", "0", "--max-eval-samples", "0",
        "--eval-split", "tune",
        "--cuda-memory-limit-mib", "12000", "--cuda-min-free-mib", "8192",
    ]
    if scenes is not None:
        command.extend(("--train-scenes", *scenes))
    return command


@contextlib.contextmanager
def patched_runtime(arm, seed, scenes, experiment, run_dir):
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import motiondrive_v2_flip_augment as flip_module
    import evaluate_motiondrive_v2_planning as planning_eval
    from motiondrive_v2_flip_augment import FlipAugmented
    detail = MR_DETAIL.get(arm)
    if detail:
        import matching_resolution as mr
        from models.motiondrive_v2 import MotionDriveV2Config
        import models.motiondrive_v2.model as model_module

    run_dir = Path(run_dir).resolve()
    originals = {
        "compute_loss": trainer.compute_loss,
        "dataset": data_api.MotionDriveDataset,
        "flip_item": flip_module.flip_item,
        "train_inputs": trainer.model_inputs,
        "eval_inputs": planning_eval.planning_model_inputs,
        "model": model_api.MotionDriveV2,
        "forward_parts": None, "forward": None, "encoder_forward": None,
        "protocol": trainer._validate_experimental_protocol,
        "runtime": trainer._validate_experimental_runtime,
        "schedule": trainer._training_schedule_actions,
        "checkpoint": trainer.atomic_checkpoint,
        "json": trainer.atomic_json,
    }

    def dataset_factory(**kwargs):
        split, requested = kwargs.get("split"), kwargs.get("scenes")
        if split == "tune":
            if requested is not None or kwargs.get("augment"):
                raise ValueError("the tune evaluation set takes no scene filter and no augmentation")
            base = originals["dataset"](**kwargs)
            return mr.MotionCanvasDataset(base, detail) if detail else base
        if split != "train":
            raise ValueError("E1 uses only the train and tune splits")
        expected = None if arm in EXPANDED_ARMS else scenes
        if (list(requested) if requested else None) != (list(expected) if expected else None):
            raise ValueError("training dataset scenes differ from the pinned arm")
        if kwargs.get("augment") is not True:
            raise ValueError("the training dataset must keep augmentation on")
        base = originals["dataset"](**kwargs)
        if detail:
            base = mr.MotionCanvasDataset(base, detail)
        # the flip stream follows the training seed, as in the original screens
        return FlipAugmented(base, *FLIP_WIDTHS, p=FLIP_P, seed=seed)

    def validate_protocol(declared):
        if not isinstance(declared, dict) or declared.get("name") != NAME \
                or declared.get("arm") != arm or declared.get("seed") != seed \
                or declared.get("provided_status_used") is not False:
            raise ValueError("E1 experimental protocol mismatch")
        return None

    def validate_runtime(runtime_args, declared):
        expected = {"phase": "joint", "goal_on": 1, "state_on": 1,
                    "steps": ARM_UPDATES[arm],
                    "batch": BATCH, "microbatch": MICROBATCH, "eval_batch": 4,
                    "eval_every": EVAL_EVERY, "save_every": EVAL_EVERY,
                    "lr": HEAD_LR, "backbone_lr": BACKBONE_LR, "weight_decay": .01,
                    "warmup": WARMUP, "grad_clip": 5., "alpha_occ": .2,
                    "alpha_lane": .2, "alpha_motion": .2, "uncertainty": 1,
                    "precision": "bf16", "time_input": "nominal", "bn_policy": "fixed",
                    "arch": "resnet50", "motion_input_mode": "low_feature",
                    "cross_cell_goal_mode": "zero", "history_contract": "control",
                    "train_stride": 1, "eval_stride": 5, "max_train_samples": 0,
                    "max_eval_samples": 0, "eval_split": "tune"}
        bad = {k: (getattr(runtime_args, k), v) for k, v in expected.items()
               if getattr(runtime_args, k) != v}
        if bad:
            raise ValueError(f"E1 recipe mismatch: {bad}")
        if (runtime_args.seed != seed or not runtime_args.init or runtime_args.resume
                or runtime_args.pretrained or runtime_args.eval_only or runtime_args.cpu):
            raise ValueError("E1 runtime flags mismatch")
        expected_scenes = None if arm in EXPANDED_ARMS else list(scenes)
        if runtime_args.train_scenes != expected_scenes or runtime_args.eval_scenes is not None:
            raise ValueError("E1 scene arguments mismatch")

    def schedule(step, runtime_args, declared):
        if (runtime_args.steps != ARM_UPDATES[arm] or runtime_args.eval_every != EVAL_EVERY
                or runtime_args.save_every != EVAL_EVERY):
            raise ValueError("E1 schedule contract mismatch")
        due = step % EVAL_EVERY == 0 or step == runtime_args.steps
        return due, due

    def atomic_checkpoint(path, data):
        originals["checkpoint"](path, data)
        # Keep a weights-only snapshot at every planned point so the
        # best-on-V0 checkpoint can be evaluated after the run.
        if Path(path).name == "last.pth":
            light = {"model": data["model"], "manifest": data["manifest"],
                     "step": data["step"], "epoch": data["epoch"],
                     "best_metric": data["best_metric"]}
            originals["checkpoint"](run_dir / f"ckpt_step{int(data['step'])}.pth", light)

    def atomic_json(path, data):
        originals["json"](path, data)
        if Path(path).name == "latest_eval.json" and isinstance(data, dict) and "step" in data:
            originals["json"](run_dir / f"eval_step{int(data['step'])}.json", data)

    if detail:
        flip_module.flip_item = mr.wrap_flip_item(originals["flip_item"])

        def canvas_train_inputs(batch, **kwargs):
            return mr.model_inputs_with_canvas(originals["train_inputs"], batch, **kwargs)

        def canvas_eval_inputs(batch, time_input="raw"):
            return mr.model_inputs_with_canvas(
                originals["eval_inputs"], batch, time_input=time_input)

        def model_factory(config=None):
            built = originals["model"](config if config is not None else MotionDriveV2Config())
            built._mr_pending_rebuild = True
            return built

        def load_initial(model, common, declared=None):
            """R0 for everything except the resized fuse convolution."""
            report = mr.rebuild_correlation_fuse(model, mr.NEW_RADIUS)
            width = MR_DESCRIPTOR.get(arm)
            widening = mr.rebuild_descriptor(model, width) if width else None
            adjacency = None
            if arm in ADJACENT_EDGES_ON:
                import adjacent_edges as adj
                if originals["encoder_forward"] is None:
                    originals["encoder_forward"] = type(model.motion_encoder).forward
                # One fixed seed for both arms: the new module's weights are
                # identical and only the mask differs.
                torch.manual_seed(ADJACENT_INIT_SEED)
                adjacency = adj.install(model, ADJACENT_EDGES_ON[arm])
            if originals["forward_parts"] is None:
                originals["forward_parts"] = type(model).forward_parts
                originals["forward"] = type(model).forward
            mr.install(model, detail)
            state = dict(common["model"])
            rebuilt_prefixes = ["motion_encoder.correlation_fuse.0."]
            if widening is not None:
                rebuilt_prefixes.extend(p + "." for p in widening["projections"])
            if adjacency is not None:
                # The adjacent module has no counterpart in R0 at all, so its
                # tensors are new rather than rebuilt; they join the same set so
                # the strict check below still covers every OTHER tensor.
                rebuilt_prefixes.append("motion_encoder.adjacent_attention.")
            dropped = [k for k in state
                       if any(k.startswith(p) for p in rebuilt_prefixes)]
            # Tensors that exist in R0 but must not be loaded are popped. The
            # adjacent module's tensors have no R0 counterpart at all, so they
            # are never in `state`; they only have to appear in the expected
            # missing-key set below.
            new_keys = ([k for k in model.state_dict()
                         if k.startswith("motion_encoder.adjacent_attention.")]
                        if adjacency is not None else [])
            for key in dropped:
                state.pop(key)
            incompatible = model.load_state_dict(state, strict=False)
            expected = sorted(dropped + new_keys)
            if sorted(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
                raise ValueError(f"MR load must miss exactly {expected}, got "
                                 f"{incompatible.missing_keys} / {incompatible.unexpected_keys}")
            # Every tensor except the resized fuse convolution is R0 bit for bit;
            # that convolution is newly initialised, so the step-zero state SHA
            # cannot equal R0's. The pinned-SHA check is therefore replaced by
            # the measured value, and the real guarantee is the strict match on
            # everything else, asserted just above.
            loaded = {name: value for name, value in model.state_dict().items()
                      if name not in set(expected)}
            reference = {name: value for name, value in common["model"].items()
                         if name not in set(expected)}
            if tensor_state_sha256(loaded) != tensor_state_sha256(reference):
                raise ValueError("MR load changed a tensor it was supposed to keep")
            measured = tensor_state_sha256(model.state_dict())
            if declared is not None:
                declared["expected_initial_model_state_sha256"] = measured
                declared["mr_step_zero_state"] = {
                    "sha256_is_measured_not_pinned": True,
                    "reason": ("motion_encoder.correlation_fuse.0 is rebuilt for 81 bins and "
                               "newly initialised, so step-zero cannot match R0"),
                    "every_other_tensor_matches_r0": True,
                    "r0_model_state_sha256": tensor_state_sha256(common["model"]),
                    "measured_step_zero_sha256": measured,
                    "reinitialised": expected,
                }
            return {"mr_load": {"reinitialised": expected, "rebuild": report,
                                "widening": widening, "adjacency": adjacency,
                                "strict_for_every_other_tensor": True,
                                "measured_step_zero_sha256": measured}}

        trainer.model_inputs = canvas_train_inputs
        planning_eval.planning_model_inputs = canvas_eval_inputs
        model_api.MotionDriveV2 = model_factory
        model_module.MotionDriveV2 = model_factory
        trainer._load_initial_model_state = load_initial

    lam = LENGTH_LAMBDA.get(arm, 0.0)
    if lam:
        from length_auxiliary import wrap_compute_loss
        trainer.compute_loss = wrap_compute_loss(originals["compute_loss"], lam)
    data_api.MotionDriveDataset = dataset_factory
    trainer._validate_experimental_protocol = validate_protocol
    trainer._validate_experimental_runtime = validate_runtime
    trainer._training_schedule_actions = schedule
    trainer.atomic_checkpoint = atomic_checkpoint
    trainer.atomic_json = atomic_json
    try:
        yield
    finally:
        trainer.compute_loss = originals["compute_loss"]
        flip_module.flip_item = originals["flip_item"]
        trainer.model_inputs = originals["train_inputs"]
        planning_eval.planning_model_inputs = originals["eval_inputs"]
        model_api.MotionDriveV2 = originals["model"]
        if detail:
            model_module.MotionDriveV2 = originals["model"]
            if originals["encoder_forward"] is not None:
                from models.motiondrive_v2.motion_encoder import MotionEncoder
                MotionEncoder.forward = originals["encoder_forward"]
            if originals["forward_parts"] is not None:
                from models.motiondrive_v2.model import MotionDriveV2 as _Real
                _Real.forward_parts = originals["forward_parts"]
                _Real.forward = originals["forward"]
        data_api.MotionDriveDataset = originals["dataset"]
        trainer._validate_experimental_protocol = originals["protocol"]
        trainer._validate_experimental_runtime = originals["runtime"]
        trainer._training_schedule_actions = originals["schedule"]
        trainer.atomic_checkpoint = originals["checkpoint"]
        trainer.atomic_json = originals["json"]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenes, training, evaluation = build_datasets(args.arm, args.seed)
    experiment = build_experiment(args.arm, args.seed, scenes, training, evaluation)
    command = trainer_argv(args.arm, args.seed, args.run_dir, scenes, args.gpu)
    summary = {"arm": args.arm, "seed": args.seed, "gpu": args.gpu,
               "run_dir": str(Path(args.run_dir).resolve()),
               "train_rows": experiment["train_data"]["rows"],
               "train_scenes": experiment["train_data"]["scenes"],
               "train_rows_sha256": experiment["train_data"]["rows_sha256"],
               "tune_rows": experiment["tune_data"]["rows"],
               "tune_rows_sha256": experiment["tune_data"]["rows_sha256"],
               "updates": ARM_UPDATES[args.arm], "eval_every": EVAL_EVERY,
               "exposures_of_own_train": (ARM_UPDATES[args.arm] * BATCH
                                          / experiment["train_data"]["rows"])}
    print("PLAN " + json.dumps(summary), flush=True)
    if args.dry_run:
        return

    del training, evaluation
    import train_motiondrive_v2 as trainer
    run_dir = Path(args.run_dir).resolve()
    # The trainer refuses a non-empty run directory, so the protocol copy is
    # written next to the reports first and into the run directory afterwards.
    plan_path = (ROOT / "reports/md_r0_reset_20260914"
                 / f"experiment_{args.arm}-s{args.seed}.json")
    plan_path.write_text(json.dumps(experiment, indent=1, sort_keys=True) + "\n")
    with patched_runtime(args.arm, args.seed, scenes, experiment, run_dir):
        trainer.run_training(command, experiment=experiment)
    (run_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=1, sort_keys=True) + "\n")
    print("DONE " + json.dumps({"arm": args.arm, "run_dir": str(run_dir)}), flush=True)


if __name__ == "__main__":
    main()
