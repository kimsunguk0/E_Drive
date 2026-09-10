#!/usr/bin/env python3
"""CPU interface audit for the frozen 12-session confirmation experiment.

Unlike the original metadata-only audit, this explicitly instantiates PlanDataset:
its global future array is loaded and selected-label finiteness is checked. No
model, prediction, metric, label distribution, or reserve dataset is evaluated.
Only files underneath this split_audit report directory may be created.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f"Refusing to replace frozen file: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)
    return {"path": str(path.resolve()), "file_sha256": sha(path)}


def publish_json(path, value):
    return publish(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def rejected(call, expected):
    try:
        call()
    except ValueError as error:
        require(expected in str(error), f"Unexpected rejection: {error}")
        return str(error)
    raise ValueError("Required provenance rejection did not happen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("/NHNHOME/data/sukim/adcl"))
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    wt = args.worktree.resolve()
    output = wt / "reports/sparsedrivev2_20260910/split_audit"
    code = wt / "experiments/sparsedrivev2_20260910"
    sys.path.insert(0, str(code))
    from data import PlanDataset, EXPECTED_SPLIT_SHA, rows_sha
    from train import verify_initialization, PUBLIC_SHA

    sources = {str(code / name): sha(code / name) for name in ("data.py", "train.py", "losses.py", "public_model.py")}
    canonical = output / "primary_manifest.json"
    confirmation_path = output / "confirmation12_manifest.json"
    audit_path = output / "audit.json"
    audit = json.loads(audit_path.read_text())
    require(sha(canonical) == EXPECTED_SPLIT_SHA, "Primary split changed")
    require(sha(confirmation_path) == audit["manifests"]["confirmation12"]["sha256"], "Confirmation membership changed")
    confirmation = json.loads(confirmation_path.read_text())
    primary = json.loads(canonical.read_text())
    contracts = audit["row_contracts"]["confirmation12"]
    train_rows_path = output / "rows/confirmation12_train.npy"
    held_original = output / "rows/confirmation12_confirmation12.npy"
    tune_rows_path = output / "rows/confirmation12_tune.npy"
    for label, path in (("train", train_rows_path), ("confirmation12", held_original), ("tune", tune_rows_path)):
        array = np.load(path, allow_pickle=False)
        require(sha(path) == contracts[label]["sha256"], f"NPY file changed: {label}")
        require(rows_sha(array) == contracts[label]["row_array_sha256"], f"Row indices changed: {label}")
    held_rows_path = output / "rows/confirmation12_held_rows.npy"
    held_alias = publish(held_rows_path, held_original.read_bytes())
    train_scenes = set(confirmation["splits"]["train"])
    held_scenes = set(confirmation["splits"]["confirmation12"])
    require(len(train_scenes) == 171 and len(held_scenes) == 32, "Unexpected confirmation population")
    require(not train_scenes & held_scenes, "Confirmation scene overlap")
    require(train_scenes | held_scenes == set(primary["splits"]["train"]), "Original train203 was not preserved")
    scene_files = {
        "train": publish_json(output / "confirmation12_train_scenes.json", {"scenes": sorted(train_scenes)}),
        "held": publish_json(output / "confirmation12_held_scenes.json", {"scenes": sorted(held_scenes)}),
    }

    dataset_contracts = {}
    for mode in ("zero", "causal_selection"):
        common = dict(base=args.base, split_manifest=canonical, status_mode=mode)
        fit = PlanDataset(**common, split="train", rows_file=train_rows_path)
        held = PlanDataset(**common, split="train", rows_file=held_rows_path, stride=5)
        tune = PlanDataset(**common, split="tune", rows_file=tune_rows_path, stride=5)
        fit_meta, held_meta, tune_meta = fit.provenance(), held.provenance(), tune.provenance()
        require((len(fit), len(held), len(tune)) == (46170, 1728, 1998), "Runtime row counts differ")
        require(np.array_equal(fit.allowed_rows, np.load(train_rows_path)), "Allowed fit row set differs")
        require(np.array_equal(held.allowed_rows, np.load(held_rows_path)), "Held row set differs")
        require(set(fit_meta["scenes"]) == train_scenes and set(held_meta["scenes"]) == held_scenes, "Runtime scenes differ")
        require(len(fit_meta["sessions"]) == 60 and len(held_meta["sessions"]) == 12, "Runtime sessions differ")
        require(not set(fit_meta["sessions"]) & set(held_meta["sessions"]), "Fit/held sessions overlap")
        require(not set(fit_meta["sessions"]) & set(tune_meta["sessions"]), "Fit/tune sessions overlap")
        require(not set(held_meta["sessions"]) & set(tune_meta["sessions"]), "Held/tune sessions overlap")
        dataset_contracts[mode] = {"train171": fit_meta, "confirmation12": held_meta, "tune37": tune_meta}
        if mode == "zero":
            fit_zero, held_zero, tune_zero = fit, held, tune

    checkpoint = args.checkpoint or wt / "checkpoints/sparsedrivev2_20260910/public/sparsedrive_navsimv1_92p2.ckpt"
    bank = args.bank or wt / "cache/sparsedrivev2_20260910/bank/p1024_v256_native100m_v8.npz"
    full_train = PlanDataset(args.base, canonical, "train", status_mode="zero")
    positive = verify_initialization(checkpoint, bank, full_train, tune_zero)
    negative = {
        "full_train203_bank_into_confirmation171_tune37": rejected(
            lambda: verify_initialization(checkpoint, bank, fit_zero, tune_zero), "exactly this run's allowed training rows"),
        "full_train203_bank_into_confirmation171_held32": rejected(
            lambda: verify_initialization(checkpoint, bank, fit_zero, held_zero), "exactly this run's allowed training rows"),
        "nonpublic_initialization": rejected(
            lambda: verify_initialization(Path(__file__), bank, fit_zero, held_zero), "pinned official NAVSIMv1 public checkpoint"),
    }
    with tempfile.TemporaryDirectory(prefix=".negative_bank_receipt_", dir=output) as temporary:
        fake = Path(temporary) / "bank.npz"
        fake.symlink_to(bank.resolve())
        fake.with_suffix(".npz.json").write_text(json.dumps({"bank_sha256": "0" * 64}))
        negative["corrupt_bank_sidecar"] = rejected(
            lambda: verify_initialization(checkpoint, fake, full_train, tune_zero), "Bank sidecar hash mismatch")

    for source, digest in sources.items():
        require(sha(source) == digest, f"Runtime source changed during audit: {source}")
    report = {
        "status": "PASS_CONFIRMATION12_RUNTIME_AND_ANCESTRY_REJECTION",
        "scope": {"gpu_used": False, "models_loaded_or_run": False, "prediction_metrics_computed": False,
                  "reserve_dataset_instantiated": False, "global_future_cache_loaded_by_PlanDataset": True,
                  "selected_train_held_tune_target_finiteness_checked_by_loader": True,
                  "label_distributions_examined": False, "runtime_sources_changed": False},
        "inputs": {"original_audit_sha256": sha(audit_path), "primary_manifest_sha256": sha(canonical),
                   "confirmation_membership_sha256": sha(confirmation_path), "runtime_source_sha256": sources,
                   "runtime_check_script_sha256": sha(__file__), "public_checkpoint_sha256": sha(checkpoint),
                   "expected_public_checkpoint_sha256": PUBLIC_SHA, "primary_bank_sha256": sha(bank)},
        "row_files": {"train": str(train_rows_path), "held_alias": held_alias, "held_original": str(held_original),
                      "tune": str(tune_rows_path)},
        "bank_extraction_scene_files": scene_files,
        "dataset_contracts": dataset_contracts,
        "primary_bank_primary_train_positive_control": positive,
        "required_rejections": negative,
        "execution_contract": {
            "fit": "primary_manifest.json + split=train + confirmation12_train.npy; exact bank fit rows must equal allowed_rows",
            "selection": "unchanged primary tune37; choose architecture, budget and checkpoint here",
            "confirmation": "after decisions freeze, primary_manifest.json + split=train + confirmation12_held_rows.npy + stride=5; one final confirmation",
            "public_initialization": "start again from pinned public NAVSIMv1 checkpoint; do not resume any train203/fulltrain learned weight or reuse its fitted bank",
            "bank_fit": "new train-only extraction with --split primary_manifest.json --partition train --scenes-json confirmation12_train_scenes.json --stride 1 --min-frame 30",
            "future_bank_verification": "a newly fitted train171 bank must pass verify_initialization with the train171 and held32 datasets before CUDA model creation",
            "reserve136": "excluded from fit, selection and this interface audit",
            "confirmation_is_not_a_new_pristine_dataset": "membership and historical results existed; this contract removes learned ancestry exposure in the new run",
        },
    }
    receipt = publish_json(output / "confirmation12_runtime.json", report)
    print(json.dumps({"status": report["status"], "receipt": receipt,
                      "rows": {k: {"n": v["rows"], "sha": v["rows_sha256"]} for k, v in dataset_contracts["zero"].items()},
                      "required_rejections": negative}, indent=2))


if __name__ == "__main__":
    main()
