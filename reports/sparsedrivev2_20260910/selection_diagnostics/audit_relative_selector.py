"""Audit the head on saved tune outputs; no image/model forward or GPU access."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path.cwd()
SOURCE = ROOT / "experiments/sparsedrivev2_20260910"
sys.path.insert(0, str(SOURCE))
from relative_selector import (CandidateRelativeSelector, FEATURE_NAMES, FEATURE_VERSION,
                               make_candidate_features)
from data import PlanDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


torch.set_num_threads(2)
torch.manual_seed(1049)
dataset = PlanDataset("/NHNHOME/data/sukim/adcl",
                      ROOT / "reports/sparsedrivev2_20260910/split_audit/primary_manifest.json",
                      "tune", status_mode="causal_selection", goal_mode="selection", limit=8)
status = torch.from_numpy(dataset.status)
goal = torch.from_numpy(dataset.goal[dataset.rows])
result = {"created_unix": time.time(), "feature_version": FEATURE_VERSION,
          "feature_names": FEATURE_NAMES, "device": "cpu", "precision": "float32",
          "new_training_or_model_forward": False, "held_evaluation": False,
          "source_sha256": sha(SOURCE / "relative_selector.py"),
          "test_sha256": sha(SOURCE / "test_relative_selector.py"),
          "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "status_source": dataset.status_source, "rows": dataset.rows.tolist(),
          "rows_sha256": hashlib.sha256(dataset.rows.astype("<i8").tobytes()).hexdigest(),
          "goal_source": dataset.provenance()["goal_source"], "cases": []}
for case in ["old500_v256_cpu8_v1", "dense_none_cpu8_v1", "dense_selection_cpu8_v1"]:
    path = ROOT / "reports/sparsedrivev2_20260910/selection_diagnostics" / case / "candidate_dump.npz"
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z["rows"], dataset.rows)
        output = {"candidate_xy": torch.from_numpy(z["candidate_xy"]),
                  "candidate_ids": torch.from_numpy(z["candidate_ids"]),
                  "scores": torch.from_numpy(z["logits"])}
        costs = z["costs"]
    b, k = output["scores"].shape
    output["candidate_valid"] = torch.ones(b, k, dtype=torch.bool)
    selected = output["scores"].argmax(-1)
    output["selected_candidate_id"] = output["candidate_ids"][torch.arange(b), selected]
    output["trajectory"] = output["candidate_xy"][torch.arange(b), selected]
    wrapper = CandidateRelativeSelector(DummyBase())
    features = make_candidate_features(output, status, goal)
    assert torch.isfinite(features).all()
    rescored = wrapper.rescore(output, status, goal)
    for key in output:
        assert torch.equal(output[key], rescored[key]), key
    assert output["candidate_xy"] is rescored["candidate_xy"]
    assert output["candidate_ids"] is rescored["candidate_ids"]
    changed = wrapper.rescore(output, status + 1., goal * -1.)
    assert changed["candidate_xy"] is output["candidate_xy"]
    assert changed["candidate_ids"] is output["candidate_ids"]
    result["cases"].append({"case": case, "dump_sha256": sha(path), "rows": b,
                            "candidates": k, "feature_shape": list(features.shape),
                            "feature_abs_max": features.abs().amax((0, 1)).tolist(),
                            "all_features_finite": True, "all_original_fields_bitwise_equal": True,
                            "candidate_tensor_identity_preserved": True,
                            "selected_d3_unchanged": float(costs[np.arange(b), selected.numpy()].mean())})
result["passed"] = True
target = ROOT / "reports/sparsedrivev2_20260910/selection_diagnostics/relative_selector_audit.json"
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"path": str(target), "passed": True, "source_sha256": result["source_sha256"],
                  "cases": [{k: v for k, v in c.items() if k != "feature_abs_max"} for c in result["cases"]]}, indent=2))
