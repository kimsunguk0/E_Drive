"""Evaluate a trained V2 checkpoint using its verified source snapshot.

Tune evaluation opens only the original tune37 population. Confirmation12 requires
explicit train/held row files and an externally supplied frozen-candidate approval
receipt. --audit-only validates and strictly loads on CPU, without opening labels
or running a model. This program never creates an approval receipt.

"Approved" means the root agent's internal frozen experimental selection record
after tune selection. It is not an additional user permission requirement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

TASK = "experiments/sparsedrivev2_20260910"
PUBLIC_SHA = "330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735"
SPLIT_SHA = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
EGO_SHA = "d35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd"
CONFIRM_TRAIN_SHA = "701e7ea7b76acd6b0d99845a0f400c9b65a5c470131d2d3b6324192e09c28a35"
CONFIRM_HELD_SHA = "2809febcd692040870821e2575062722226b4f58152e53b7eed27d9cc5aaa3b7"
TUNE_SHA = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows_sha(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def read_rows(path):
    value = np.load(path, allow_pickle=False)
    require(value.ndim == 1 and value.dtype.kind in "iu", "Rows must be an integer NPY vector")
    require(len(value) > 0 and np.all(value[1:] > value[:-1]), "Rows must be nonempty, sorted and unique")
    return value.astype(np.int64)


def check_approval(receipt, expected):
    """Pure guard, also suitable for caller preflight; never grants approval."""
    require(isinstance(receipt, dict), "Confirmation requires a candidate approval JSON object")
    require(receipt.get("schema") == "sparsedrivev2_confirmation_candidate_v1", "Unknown confirmation approval schema")
    require(receipt.get("approved") is True and receipt.get("frozen") is True,
            "Confirmation candidate must already be approved and frozen")
    require(receipt.get("population") == "confirmation12", "Approval must name confirmation12")
    for key, value in expected.items():
        require(receipt.get(key) == value, f"Confirmation approval mismatch: {key}")


def inspect_checkpoint(checkpoint, *, population="tune", bank_path=None, train_rows_path=None,
                       eval_rows_path=None, approval_path=None, source_root=None,
                       public_checkpoint=None, base=None):
    """Metadata/identity preflight. No PlanDataset, future labels or inference."""
    require(population in ("tune", "confirmation12"), "Only tune and confirmation12 are enabled")
    checkpoint = Path(checkpoint).resolve()
    checkpoint_sha = file_sha(checkpoint)
    approval = None
    if population == "confirmation12":
        require(train_rows_path is not None and eval_rows_path is not None and approval_path is not None,
                "Confirmation requires explicit --train-rows, --eval-rows and --approved-candidate")
        approval = json.loads(Path(approval_path).read_text())
        check_approval(approval, {"checkpoint_sha256": checkpoint_sha})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(file_sha(checkpoint) == checkpoint_sha, "Checkpoint changed while loading")
    require(isinstance(payload, dict) and all(k in payload for k in ("model", "manifest", "step", "result")),
            "Expected a trained V2 checkpoint payload")
    manifest = payload["manifest"]
    arguments = manifest["arguments"]
    if "averaging" in payload:
        from average_checkpoints import validate_averaging_metadata
        validate_averaging_metadata(payload)
        derived_receipt = json.loads(Path(str(checkpoint) + ".json").read_text())
        require(derived_receipt["sha256"] == checkpoint_sha, "Averaged checkpoint sidecar hash mismatch")
        require(derived_receipt["averaging"] == payload["averaging"], "Averaging metadata/sidecar mismatch")
    else:
        require(isinstance(payload["result"], dict) and payload["result"]["step"] == payload["step"],
                "Checkpoint/result step mismatch")
    require(manifest["torch"] == str(torch.__version__), "Use the checkpoint's exact PyTorch runtime")
    require(manifest["score_mode"] == "imitation", "Unsupported trained score mode")
    require(manifest["public_checkpoint_sha256"] == PUBLIC_SHA, "Checkpoint lacks pinned public ancestry")
    public = Path(public_checkpoint or arguments["checkpoint"]).resolve()
    require(file_sha(public) == PUBLIC_SHA, "Public initialization artifact changed")
    goal_mode = arguments.get("goal_mode", "none")
    require(goal_mode in ("none", "selection"), "Unsupported goal mode")
    require(arguments["status_mode"] in ("zero", "causal_selection"), "Unsupported status mode")

    source_root = Path(source_root or Path(arguments["run_dir"]) / "source").resolve()
    sources = manifest["source_sha256"]
    required = {f"{TASK}/{name}" for name in ("data.py", "train.py", "losses.py", "public_model.py")}
    if goal_mode == "selection":
        required.add(f"{TASK}/goal_selector.py")
    require(required <= set(sources), "Checkpoint source closure is incomplete")
    for relative, expected_sha in sources.items():
        path = (source_root / relative).resolve()
        require(path.is_relative_to(source_root), "Source entry escapes the recorded snapshot")
        require(file_sha(path) == expected_sha, f"Snapshot source changed: {relative}")

    split_path = Path(manifest["train"]["split_manifest"]).resolve()
    require(file_sha(split_path) == SPLIT_SHA == manifest["train"]["split_sha256"], "Primary split mismatch")
    split = json.loads(split_path.read_text())
    ego = Path(manifest["train"]["ego_cache"]).resolve()
    require(file_sha(ego) == EGO_SHA == manifest["train"]["ego_cache_sha256"], "Ego source mismatch")
    with np.load(ego, allow_pickle=False) as z:
        names = z["scenarios"].astype(str)[z["scen_idx"]]
        frames = z["frame"]
    eligible_train = np.flatnonzero(np.isin(names, split["splits"]["train"]) & (frames >= 30))
    train_rows_path = train_rows_path or arguments.get("train_rows")
    train_rows = read_rows(train_rows_path) if train_rows_path else eligible_train
    require(np.isin(train_rows, eligible_train).all(), "Training rows leave primary train203")
    recorded_train_sha = manifest["train"].get("allowed_rows_sha256")
    require(recorded_train_sha is not None and rows_sha(train_rows) == recorded_train_sha,
            "Supplied training rows differ from checkpoint's allowed fit population")
    require(len(train_rows) == manifest["train"]["allowed_rows"], "Allowed training row count mismatch")
    if population == "tune":
        expected_eval = np.flatnonzero(np.isin(names, split["splits"]["tune"]) & (frames >= 30) & (frames % 5 == 0))
        require(rows_sha(expected_eval) == TUNE_SHA, "Original tune37 identity changed")
        eval_rows = read_rows(eval_rows_path) if eval_rows_path else expected_eval
        require(np.array_equal(eval_rows, expected_eval), "Tune evaluation requires all original 1,998 rows")
    else:
        eval_rows = read_rows(eval_rows_path)
        require(rows_sha(train_rows) == CONFIRM_TRAIN_SHA and len(train_rows) == 46170,
                "Confirmation requires the frozen train171 population")
        require(rows_sha(eval_rows) == CONFIRM_HELD_SHA and len(eval_rows) == 1728,
                "Confirmation requires the frozen held32 population")
        require(np.isin(eval_rows, eligible_train).all() and (frames[eval_rows] % 5 == 0).all(), "Invalid confirmation row identities")
    fit_scenes, eval_scenes = set(names[train_rows]), set(names[eval_rows])
    fit_sessions = {split["scene_to_session"][s] for s in fit_scenes}
    eval_sessions = {split["scene_to_session"][s] for s in eval_scenes}
    require(not fit_scenes & eval_scenes and not fit_sessions & eval_sessions,
            "Evaluation overlaps the bank/model fit population")

    bank = Path(bank_path or arguments["bank"]).resolve()
    bank_sha = file_sha(bank)
    sidecar_path = Path(str(bank) + ".json")
    sidecar = json.loads(sidecar_path.read_text())
    require(bank_sha == manifest["bank_sha256"] == sidecar.get("bank_sha256"), "Frozen bank hash mismatch")
    init = manifest["initialization_audit"]
    require(file_sha(sidecar_path) == init["bank_sidecar_sha256"], "Bank sidecar differs from initialization receipt")
    with np.load(bank, allow_pickle=False) as z:
        bank_rows = z["train_rows"]
        metadata = json.loads(str(z["metadata_json"]))
        require(np.array_equal(bank_rows, train_rows), "Bank must fit exactly the supplied allowed training rows")
        require(str(z["train_rows_sha256"]) == rows_sha(bank_rows), "Bank internal row hash mismatch")
    for key, expected in (("partition", "train"), ("split_sha256", SPLIT_SHA),
                          ("train_rows_sha256", rows_sha(train_rows))):
        require(metadata.get(key) == expected == sidecar.get(key), f"Bank metadata mismatch: {key}")
        require(init["bank_training_metadata"].get(key) == expected, f"Checkpoint ancestry mismatch: {key}")
    require(set(metadata["train_scenes"]) == fit_scenes == set(sidecar["train_scenes"]), "Bank scenes disagree with its row identities")
    require(init["bank_train_rows_sha256"] == rows_sha(train_rows), "Initialization fit rows mismatch")
    if approval is not None:
        check_approval(approval, {"checkpoint_sha256": checkpoint_sha, "bank_sha256": bank_sha,
                                 "train_rows_sha256": rows_sha(train_rows), "eval_rows_sha256": rows_sha(eval_rows),
                                 "split_sha256": SPLIT_SHA})
    receipt = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha, "step": payload["step"],
               "population": population, "rows": len(eval_rows), "rows_sha256": rows_sha(eval_rows),
               "fit_rows": len(train_rows), "fit_rows_sha256": rows_sha(train_rows),
               "fit_scenes": len(fit_scenes), "fit_sessions": len(fit_sessions),
               "evaluation_scenes": len(eval_scenes), "evaluation_sessions": len(eval_sessions),
               "bank": str(bank), "bank_sha256": bank_sha, "bank_sidecar_sha256": file_sha(sidecar_path),
               "public_checkpoint_sha256": PUBLIC_SHA, "source_root": str(source_root), "source_sha256": sources,
               "split_sha256": SPLIT_SHA, "ego_sha256": EGO_SHA,
               "goal_mode": goal_mode, "status_mode": arguments["status_mode"],
               "approval_sha256": file_sha(approval_path) if approval_path else None,
               "preflight_future_labels_opened": False}
    if "averaging" in payload:
        receipt["averaging"] = payload["averaging"]
    # Optimizer tensors are unnecessary for evaluation and can be released now.
    payload.pop("optimizer", None)
    return SimpleNamespace(payload=payload, manifest=manifest, receipt=receipt, bank=bank, public=public,
                           source_root=source_root, split_path=split_path, ego=ego, goal_mode=goal_mode,
                           base=Path(base or arguments["base"]).resolve(), eval_rows=eval_rows,
                           eval_rows_path=eval_rows_path, population=population,
                           train_rows=train_rows, train_rows_path=train_rows_path)


@contextmanager
def recorded_runtime(plan):
    """Import the captured runtime, not whichever training files are live now."""
    names = ("data", "losses", "public_model", "goal_selector", "train")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    old_path = sys.path[:]
    sys.path.insert(0, str(plan.source_root / TASK))
    try:
        modules = {name: importlib.import_module(name) for name in ("data", "public_model", "train")}
        if plan.goal_mode == "selection":
            modules["goal_selector"] = importlib.import_module("goal_selector")
        yield SimpleNamespace(**modules)
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def load_verified_model(plan, runtime):
    model, coverage = runtime.public_model.PublicSparseDriveV2.from_public_checkpoint(
        plan.public, bank_path=plan.bank, backend="native", score_mode="imitation")
    bank_keys = ("path_vocab", "vel_vocab", "traj_vocab", "traj_mask")
    bank_expected = {k: getattr(model._trajectory_head, k).clone() for k in bank_keys}
    if plan.goal_mode == "selection":
        adapter = plan.manifest["public_load_coverage"]["goal_adapter"]
        require(adapter["state_dict_base_prefix"] == "base.", "Unexpected goal wrapper state prefix")
        model = runtime.goal_selector.GoalConditionedSelector(model, goal_scale=adapter["normalization_metres"])
    model.load_state_dict(plan.payload["model"], strict=True)
    for key, value in bank_expected.items():
        require(torch.equal(getattr(model._trajectory_head, key), value), f"Checkpoint mutated immutable bank buffer: {key}")
    for key, value in model.state_dict().items():
        require(not value.is_floating_point() or torch.isfinite(value).all().item(), f"Nonfinite checkpoint tensor: {key}")
    if "averaging" in plan.payload:
        from average_checkpoints import verify_averaged_state
        plan.receipt["average_verification"] = verify_averaged_state(plan.payload, model)
    model.eval()
    return model, coverage


def build_dataset(plan, runtime):
    arguments = plan.manifest["arguments"]
    kwargs = dict(base=plan.base, split_manifest=plan.split_path, ego_cache=plan.ego,
                  split="tune" if plan.population == "tune" else "train", stride=5,
                  rows_file=plan.eval_rows_path, augment=False,
                  image_size=tuple(plan.manifest["train"]["image_wh"]), status_mode=arguments["status_mode"])
    if "goal_mode" in inspect.signature(runtime.data.PlanDataset).parameters:
        kwargs["goal_mode"] = plan.goal_mode
    else:
        require(plan.goal_mode == "none", "Recorded loader lacks goal support")
    dataset = runtime.data.PlanDataset(**kwargs)
    require(np.array_equal(dataset.rows, plan.eval_rows), "Runtime dataset changed the approved evaluation rows")
    actual = dataset.provenance()
    original = plan.manifest["train"]
    for key in ("calibration_sha256", "ego_cache_sha256", "split_sha256", "camera_order", "image_wh", "normalization"):
        require(actual[key] == original[key], f"Evaluation input contract changed: {key}")
    expected_status = plan.manifest["validation"] if plan.population == "tune" else original
    if actual["status_source"] is not None:
        require(actual["status_source"]["sha256"] == expected_status["status_source"]["sha256"], "Status overlay changed")
    return dataset


def verify_bank_output(model, output, runtime):
    runtime.train.verify_output(output)
    bank = model._trajectory_head.traj_vocab.flatten(0, 1)
    ids = output["candidate_ids"]
    require(ids.dtype in (torch.int32, torch.int64) and ids.shape == output["scores"].shape,
            "Malformed complete candidate IDs")
    require(((ids >= 0) & (ids < len(bank))).all().item(), "Candidate ID outside immutable bank")
    require(torch.equal(output["candidate_xy"], bank[ids, :6, :2]), "Candidate coordinates are not fixed bank rows")
    require(torch.equal(output["trajectory"], bank[output["selected_candidate_id"], :6, :2]), "Prediction is not the selected fixed bank row")
    valid = model._trajectory_head.traj_mask.flatten(0, 1)[ids, :6].bool().all(-1)
    require(torch.equal(valid, output["candidate_valid"]), "Candidate validity differs from fixed bank")


@torch.inference_mode()
def evaluate_model(model, dataset, runtime, *, device, precision="bf16", batch_size=8,
                   workers=4, goal_mode="none", dump_coarse=False):
    """Same forward/autocast/metric order as train.evaluate; return arrays/results."""
    device = torch.device(device)
    require(precision in ("bf16", "fp32"), "Unsupported precision")
    require(precision == "fp32" or device.type == "cuda", "BF16 parity evaluation requires CUDA")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    chunks, sessions, scenarios = {}, [], []

    def append(key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float() if value.dtype == torch.bfloat16 else value.detach()
            value = value.cpu().numpy()
        chunks.setdefault(key, []).append(value)

    for batch in loader:
        x = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
        inputs = dict(goal_selection=goal_mode == "selection") if "goal_selection" in inspect.signature(runtime.data.model_inputs).parameters else {}
        with autocast:
            output = model(**runtime.data.model_inputs(x, **inputs))
        verify_bank_output(model, output, runtime)
        pred = output["trajectory"].float()
        costs = runtime.data.d3(output["candidate_xy"], x["gt_plan"][:, None].expand_as(output["candidate_xy"]))
        valid = output["candidate_valid"].bool()
        require(torch.isfinite(costs[valid]).all().item(), "Nonfinite valid candidate D3")
        delta = pred - x["gt_plan"].float()
        append("rows", batch["row"])
        append("pred", pred)
        append("d3", runtime.data.d3(pred, x["gt_plan"]))
        append("shortlist_oracle", costs.masked_fill(~valid, float("inf")).amin(-1))
        append("point_l2", torch.linalg.vector_norm(delta, dim=-1))
        append("error_xy", delta)
        append("candidate_id", output["selected_candidate_id"])
        sessions.extend(batch["session"])
        scenarios.extend(batch["scenario"])
        if dump_coarse:
            for key in ("candidate_ids", "scores", "candidate_valid", "path_ids", "velocity_ids"):
                append(key, output[key])
            for index, stage in enumerate(output["coarse"]):
                for key in ("path_ids", "velocity_ids", "path_scores", "velocity_scores"):
                    append(f"coarse{index}_{key}", stage[key])
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    require(np.array_equal(arrays["rows"], dataset.rows), "Evaluation row order mismatch")
    arrays["scenario"], arrays["session"] = np.asarray(scenarios), np.asarray(sessions)
    error, oracle, point = arrays["d3"], arrays["shortlist_oracle"], arrays["point_l2"]
    by_session = {}
    for session, value in zip(sessions, error):
        by_session.setdefault(session, []).append(float(value))
    xy = arrays["error_xy"].astype(np.float64)
    abs_xy = (np.abs(xy) * WEIGHTS[None, :, None]).sum(1).mean(0)
    energy = (xy ** 2 * WEIGHTS[None, :, None]).sum(1).mean(0)
    result = {"n": len(error), "official_d3": float(error.mean()), "shortlist_oracle_d3": float(oracle.mean()),
              "selection_regret": float((error - oracle).mean()), "point_l2_metres_05_to_30": point.mean(0).astype(float).tolist(),
              "prefix_ade_1_2_3s": [float(point[:, :n].mean()) for n in (2, 4, 6)],
              "three_second_endpoint_l2": float(point[:, -1].mean()),
              "weighted_absolute_xy_metres": abs_xy.tolist(), "weighted_squared_xy_metres2": energy.tolist(),
              "weighted_squared_longitudinal_share": float(energy[0] / energy.sum()) if energy.sum() else None,
              "session_d3": {k: float(np.mean(v)) for k, v in by_session.items()},
              "axis_note": "|x| and |y| do not add to D3; squared-energy share is a separate statistic"}
    return arrays, result


def check_reference(arrays, path):
    with np.load(path, allow_pickle=False) as reference:
        checked = []
        for key in ("rows", "pred", "d3", "shortlist_oracle", "candidate_id", "point_l2", "error_xy"):
            if key in reference:
                require(np.array_equal(arrays[key], reference[key]), f"Bitwise training-eval parity failed: {key}")
                checked.append(key)
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path), "bitwise_equal_keys": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--population", choices=("tune", "confirmation12"), default="tune")
    parser.add_argument("--bank")
    parser.add_argument("--train-rows")
    parser.add_argument("--eval-rows")
    parser.add_argument("--approved-candidate", help="Internal frozen selection JSON from root; not additional user permission")
    parser.add_argument("--source-root")
    parser.add_argument("--public-checkpoint")
    parser.add_argument("--base")
    parser.add_argument("--output", required=True, help="New output directory; existing directories are rejected")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dump-coarse", action="store_true")
    parser.add_argument("--reference-predictions")
    parser.add_argument("--audit-only", action="store_true", help="Strict CPU load, no label dataset or inference")
    args = parser.parse_args()
    if not args.audit_only:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1", "4"), "Expose exactly one user-assigned GPU: 0, 1, or 4")
    require(args.workers >= 0 and (args.batch is None or args.batch > 0), "Invalid batch/workers")
    directory = Path(args.output).resolve()
    require(not directory.exists(), "Evaluation output already exists")
    torch.set_num_threads(4)
    plan = inspect_checkpoint(args.checkpoint, population=args.population, bank_path=args.bank,
                             train_rows_path=args.train_rows, eval_rows_path=args.eval_rows,
                             approval_path=args.approved_candidate, source_root=args.source_root,
                             public_checkpoint=args.public_checkpoint, base=args.base)
    with recorded_runtime(plan) as runtime:
        model, _ = load_verified_model(plan, runtime)
        receipt = {**plan.receipt, "evaluator_sha256": file_sha(__file__), "torch": str(torch.__version__),
                   "audit_only": args.audit_only, "strict_state_load": True, "checkpoint_bank_buffers_exact": True}
        arrays = None
        if args.audit_only:
            result = {"status": "PASS_CPU_PROVENANCE_AND_STRICT_LOAD", "labels_opened": False, "model_inference": False}
        else:
            dataset = build_dataset(plan, runtime)
            batch = args.batch or plan.manifest["arguments"]["eval_batch"]
            precision = plan.manifest["arguments"]["precision"]
            model.cuda()
            arrays, result = evaluate_model(model, dataset, runtime, device="cuda:0", precision=precision,
                                             batch_size=batch, workers=args.workers, goal_mode=plan.goal_mode,
                                             dump_coarse=args.dump_coarse)
            receipt.update(batch=batch, precision=precision, physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                           evaluation_input_contract=dataset.provenance(), coarse_dump=args.dump_coarse)
            if args.reference_predictions:
                receipt["reference_parity"] = check_reference(arrays, args.reference_predictions)
            result.update(status="completed", step=plan.payload["step"])
    directory.mkdir(parents=True, exist_ok=False)
    if arrays is not None:
        np.savez_compressed(directory / "predictions.npz", **arrays)
        receipt["predictions_sha256"] = file_sha(directory / "predictions.npz")
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (directory / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
