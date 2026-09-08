import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.motiondrive_v2.shared_status import SharedCausalStatusFiLM
from scripts.motiondrive_v2_shared_status_data import (
    SharedStatusDataset, causal_status5_from_pose_records, shared_status_model_inputs,
)
from scripts import run_motiondrive_v2_shared_status_a1 as run


def _batch(status=None):
    n = 2
    return {
        "images": torch.zeros(n, 6, 3, 2, 2),
        "history_images": torch.zeros(n, 4, 3, 2, 2),
        "lidar2img": torch.zeros(n, 6, 4, 4),
        "history_transforms": torch.zeros(n, 4, 4, 4),
        "time_offsets": torch.ones(n, 4),
        "goal_xy": torch.zeros(n, 2),
        "provided_status5": torch.arange(10, dtype=torch.float32).reshape(n, 5)
        if status is None else status,
        "gt_plan": torch.full((n, 6, 2), 99.),
        "state_target": torch.full((n, 6), -99.),
    }


def test_train_and_terminal_input_adapters_preserve_only_supplied_status():
    from scripts.motiondrive_v2_training import model_inputs as base
    from scripts.evaluate_motiondrive_v2_planning import planning_model_inputs as terminal
    batch = _batch()
    status = batch["provided_status5"].clone()
    train = shared_status_model_inputs(base, batch, time_input="nominal")
    external = run._add_status(terminal(batch, "nominal"), batch)
    assert set(train) == set(external) == {
        "images", "history_images", "lidar2img", "history_transforms",
        "time_offsets", "goal_xy", "provided_status5"}
    assert torch.equal(train["provided_status5"], status)
    assert torch.equal(external["provided_status5"], status)
    batch["gt_plan"].add_(1000)
    batch["state_target"].mul_(-5)
    batch["goal_xy"].add_(123)  # goal remains its legitimate, separate old path
    assert torch.equal(shared_status_model_inputs(base, batch,
                                                  time_input="nominal")["provided_status5"], status)
    assert torch.equal(run._add_status(terminal(batch, "nominal"), batch)["provided_status5"], status)


def test_dataset_adapter_is_transparent_and_arm_specific():
    class Base:
        rows = np.array([2, 4], np.int64)
        arr = {"frame": np.array([0, 0, 30, 0, 35], np.int64)}
        scene_names = np.array(["x"] * 5)
        split_sha = "base-lineage"
        def __len__(self): return 2
        def __getitem__(self, index): return {"row": int(self.rows[index])}
        def set_epoch(self, epoch): self.epoch = epoch
    overlay = {"train": {"row": np.array([2, 4], np.int64),
                           "frame": np.array([30, 35], np.int64),
                           "status5": np.arange(10, dtype=np.float32).reshape(2, 5)}}
    provided = SharedStatusDataset(Base(), "train", overlay, "provided_causal_5d")
    zero = SharedStatusDataset(Base(), "train", overlay, "zero")
    assert provided.split_sha == "base-lineage"
    assert torch.equal(provided[1]["provided_status5"], torch.arange(5, 10).float())
    assert not bool(zero[1]["provided_status5"].count_nonzero())
    provided.set_epoch(7)
    assert provided.base.epoch == 7


def test_nominal_pose_record_adapter_ignores_future_and_unrelated_values():
    records = []
    for frame in range(-10, 1):
        t = frame / 10
        records.append({"frame": frame, "x": 2 * t + .5 * t * t, "y": -.5 * t,
                        "z": 0., "roll": 0., "pitch": 0., "yaw": .1 * t})
    records.extend([
        {"frame": 1, "x": 1e6, "y": -1e6, "z": 10., "roll": 2., "pitch": 3., "yaw": 4.},
        {"frame": 50, "x": -1e6, "y": 1e6, "z": -10., "roll": -2., "pitch": -3., "yaw": -4.},
    ])
    expected = causal_status5_from_pose_records(records)
    changed = [dict(row) for row in records]
    for row in changed:
        if row["frame"] > 0:
            for key in ("x", "y", "z", "roll", "pitch", "yaw"):
                row[key] += 99999
    assert np.array_equal(causal_status5_from_pose_records(changed), expected)


def test_schedule_is_terminal_eval_and_rolling_500_checkpoint():
    dummy = SimpleNamespace()
    expected = {499: (False, False), 500: (False, True), 1000: (False, True),
                1500: (False, False), 1999: (False, False), 2000: (True, True)}
    with run.patched_training_runtime({}, "zero", "x", "y", []):
        import train_motiondrive_v2 as trainer
        experiment = {"smoke_only_never_training_initializer": False}
        assert {step: trainer._training_schedule_actions(step, dummy, experiment)
                for step in expected} == expected


def test_runtime_mapping_is_exact_and_preflight_does_not_initialize_cuda():
    base = dict(seed=0, gpu=0, workers=4, cuda_memory_limit_mib=12000,
                cuda_min_free_mib=8192, preflight_only=True)
    for arm, uuid in run.GPU_ASSIGNMENTS.items():
        report = run.validate_runtime(SimpleNamespace(arm=arm,
            expected_physical_gpu_uuid=uuid, **base))
        assert report["gpu_used"] is False
    with pytest.raises(ValueError, match="assignment"):
        run.validate_runtime(SimpleNamespace(arm="zero",
            expected_physical_gpu_uuid=run.GPU_ASSIGNMENTS["provided_causal_5d"], **base))


def test_source_manifest_binds_exact_current_closure(tmp_path):
    files = {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest()
             for name in run.SOURCE_FILES}
    manifest = {"schema_version": 1, "git_sha": "a" * 40, "file_sha256": files}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert run.validate_source_manifest(path, digest)["file_sha256"] == files
    manifest["file_sha256"]["models/motiondrive_v2/shared_status.py"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs"):
        run.validate_source_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_runtime_patch_restores_all_globals_on_exception():
    import models.motiondrive_v2 as model_api
    import motiondrive_v2_data as data_api
    import train_motiondrive_v2 as trainer
    import evaluate_motiondrive_v2_planning as planning_eval
    before = (model_api.MotionDriveV2, data_api.MotionDriveDataset, trainer.model_inputs,
              planning_eval.planning_model_inputs, trainer._load_initial_model_state)
    with pytest.raises(RuntimeError, match="sentinel"):
        with run.patched_training_runtime({}, "zero", "x", "y", []):
            raise RuntimeError("sentinel")
    after = (model_api.MotionDriveV2, data_api.MotionDriveDataset, trainer.model_inputs,
             planning_eval.planning_model_inputs, trainer._load_initial_model_state)
    assert after == before


def test_film_outer_autocast_stays_fp32_inside_and_returns_parent_dtype():
    module = SharedCausalStatusFiLM()
    observed = {}
    handles = []
    for name, child in (("first", module.status_mlp[0]), ("final", module.status_mlp[-1])):
        handles.append(child.register_forward_hook(
            lambda _m, inp, out, key=name: observed.update({key: (inp[0].dtype, out.dtype)})))
    scene = torch.randn(2, 128, 2, 3, dtype=torch.bfloat16)
    status = torch.randn(2, 5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(scene, status)
    for handle in handles: handle.remove()
    assert observed == {"first": (torch.float32, torch.float32),
                        "final": (torch.float32, torch.float32)}
    assert out.dtype == torch.bfloat16


def test_zero_arm_initial_bias_grad_and_provided_weight_path_opens():
    module = SharedCausalStatusFiLM()
    scene = torch.ones(2, 128, 2, 2)
    zero = torch.zeros(2, 5)
    module(scene, zero).sum().backward()
    assert module.status_mlp[-1].bias.grad.abs().sum() > 0
    assert module.status_mlp[0].weight.grad.abs().sum() == 0
    with torch.no_grad():
        module.status_mlp[-1].weight.fill_(.01)
    module.zero_grad(set_to_none=True)
    status = torch.tensor([[1., 2., 3., 4., 5.], [-1., 1., -2., 2., -3.]])
    module(scene, status).sum().backward()
    assert module.status_mlp[0].weight.grad.abs().sum() > 0


def test_trainer_argv_is_exact_fixed_recipe(tmp_path):
    args = SimpleNamespace(data_root="/tmp/pm97", split_manifest="/p/split.json",
        supervision_root="/p/supervision", run_dir=str(tmp_path / "new"), init="/p/last.pth",
        smoke_only=False)
    argv = run.trainer_argv(args)
    pairs = dict(zip(argv[::2], argv[1::2]))
    assert pairs["--steps"] == "2000" and pairs["--save-every"] == "500"
    assert pairs["--eval-every"] == "2000" and pairs["--warmup"] == "100"
    assert pairs["--lr"] == "0.00005" and pairs["--backbone-lr"] == "0.000005"
    assert pairs["--batch"] == "16" and pairs["--microbatch"] == "2"
    assert pairs["--time-input"] == "nominal" and pairs["--cross-cell-goal-mode"] == "zero"


def test_smoke_is_exact_two_updates_and_no_eval_or_periodic_save(tmp_path):
    args = SimpleNamespace(data_root="/tmp/pm97", split_manifest="/p/split.json",
        supervision_root="/p/supervision", run_dir=str(tmp_path / "a1_smoke_only"),
        init="/p/last.pth", smoke_only=True)
    pairs = dict(zip(run.trainer_argv(args)[::2], run.trainer_argv(args)[1::2]))
    assert pairs["--steps"] == "2" and pairs["--log-every"] == "1"
    with run.patched_training_runtime({}, "zero", "x", "y", []):
        import train_motiondrive_v2 as trainer
        assert trainer._training_schedule_actions(1, None,
            {"smoke_only_never_training_initializer": True}) == (False, False)
        assert trainer._training_schedule_actions(2, None,
            {"smoke_only_never_training_initializer": True}) == (False, False)
