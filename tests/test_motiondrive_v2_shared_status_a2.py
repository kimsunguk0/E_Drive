"""Focused CPU contracts for the opt-in A2 runner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import run_motiondrive_v2_shared_status_a2 as run


def test_parent_identities_are_seed_matched_and_distinct():
    assert set(run.PARENTS) == {0, 1}
    assert all(set(value) == {"checkpoint_sha256", "model_state_sha256", "sidecar_sha256"}
               and all(len(digest) == 64 for digest in value.values())
               for value in run.PARENTS.values())
    assert run.PARENTS[0] != run.PARENTS[1]


def test_experiment_declares_query_conditioning_not_film_or_planner_status():
    args = SimpleNamespace(arm="provided_causal_5d", seed=1,
        expected_status_overlay_sha256="s", smoke_only=False)
    prepared = {"initial_model_state_sha256": "i", "query_state_sha256": "q",
                "missing_keys": ["shared_status_query_fusion.status_scale"]}
    result = run.build_experiment(args, {},
        {"train_rows": 54810, "train_rows_sha256": "tr",
         "tune_rows": 1998, "tune_rows_sha256": "tu"}, {}, prepared)
    route = result["conditioning"]
    assert result["name"] == "shared_status_a2_query" and result["seed"] == 1
    assert route["location"] == "scene query_context before image attention"
    assert route["changes_attention_values"] is False
    assert route["direct_planner_status_argument"] is False
    assert route["mlp"] == [5, 32, 32]


def test_runtime_mapping_and_seed_are_strict_in_cpu_preflight():
    common = dict(gpu=0, workers=4, cuda_memory_limit_mib=12000,
                  cuda_min_free_mib=8192, preflight_only=True)
    for seed in (0, 1):
        for arm, uuid in run.GPU_ASSIGNMENTS.items():
            result = run.validate_runtime(SimpleNamespace(
                seed=seed, arm=arm, expected_physical_gpu_uuid=uuid, **common))
            assert result["gpu_used"] is False
    with pytest.raises(ValueError, match="assignment"):
        run.validate_runtime(SimpleNamespace(seed=0, arm="zero",
            expected_physical_gpu_uuid=run.GPU_ASSIGNMENTS["provided_causal_5d"], **common))


def test_trainer_argv_preserves_recipe_and_selected_seed(tmp_path):
    args = SimpleNamespace(data_root="/tmp/pm97", split_manifest="/p/split.json",
        supervision_root="/p/supervision", run_dir=str(tmp_path / "new"),
        init="/p/last.pth", smoke_only=False, seed=1)
    argv = run.trainer_argv(args)
    pairs = dict(zip(argv[::2], argv[1::2]))
    assert pairs["--seed"] == "1" and pairs["--steps"] == "2000"
    assert pairs["--batch"] == "16" and pairs["--microbatch"] == "2"
    assert pairs["--lr"] == "0.00005" and pairs["--backbone-lr"] == "0.000005"
    assert pairs["--warmup"] == "100" and pairs["--save-every"] == "500"
    assert pairs["--eval-every"] == "2000" and pairs["--time-input"] == "nominal"


def test_runtime_patch_restores_globals_and_schedule_on_exception():
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval
    before = (model_api.MotionDriveV2, data_api.MotionDriveDataset, trainer.model_inputs,
              planning_eval.planning_model_inputs, trainer._load_initial_model_state)
    with pytest.raises(RuntimeError, match="sentinel"):
        with run.patched_training_runtime({}, "zero", 0, "x", "y", []):
            experiment = {"smoke_only_never_training_initializer": False}
            assert trainer._training_schedule_actions(500, None, experiment) == (False, True)
            assert trainer._training_schedule_actions(1500, None, experiment) == (False, False)
            assert trainer._training_schedule_actions(2000, None, experiment) == (True, True)
            raise RuntimeError("sentinel")
    after = (model_api.MotionDriveV2, data_api.MotionDriveDataset, trainer.model_inputs,
             planning_eval.planning_model_inputs, trainer._load_initial_model_state)
    assert after == before


def test_source_manifest_binds_exact_inherited_closure(tmp_path):
    files = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest()
             for name in run.SOURCE_FILES}
    manifest = {"schema_version": 1, "git_sha": "a" * 40, "file_sha256": files}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert run.validate_source_manifest(path, digest)["file_sha256"] == files
    manifest["file_sha256"]["models/motiondrive_v2/shared_status_query.py"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs"):
        run.validate_source_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())
