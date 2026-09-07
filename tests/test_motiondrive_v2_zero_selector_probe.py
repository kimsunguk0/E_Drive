"""CPU-only P5-Z readout tests; no base model or GPU result is produced."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import train_motiondrive_v2_zero_selector_probe as probe


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_artifact(tmp_path, split, *, base_seed=0, n=8, pilot=False, row_offset=0):
    gen = torch.Generator().manual_seed(10 + row_offset)
    features = torch.randn(n, probe.FEATURE_DIM, generator=gen)
    gt = torch.randn(n, 6, 2, generator=gen)
    move = gt + .2 * torch.randn(n, 6, 2, generator=gen)
    candidate = torch.stack((torch.zeros_like(move), move), 1)
    zero_cost = probe.weighted_d3(candidate[:, probe.ZERO], gt)
    move_cost = probe.weighted_d3(candidate[:, probe.MOVE], gt)
    costs = torch.stack((zero_cost, move_cost), 1)
    labels = torch.where(zero_cost < move_cost, torch.zeros(n, dtype=torch.long),
                         torch.ones(n, dtype=torch.long))
    rows = torch.arange(row_offset, row_offset + n, dtype=torch.int64)
    metadata = {
        "schema_version": 1, "status": "completed", "purpose": "P5-Z offline diagnostic cache",
        "split": split, "base_seed": base_seed, "checkpoint_step": 6000,
        "checkpoint_sha256": probe.P4_CHECKPOINT_SHA256[base_seed],
        "completed_run_manifest_sha256": "1" * 64, "training_git_sha": probe.P4_GIT_SHA,
        "source_manifest_sha256": "2" * 64, "source_files": {"fixed.py": "3" * 64},
        "execution_scope": "ordered_first8_pilot" if pilot else "full_fixed_split",
        "pilot_samples": 8 if pilot else 0, "full_cache_completed": not pilot,
        "data": {"n": n, "cached_n": n, "rows_sha256": probe.rows_sha256(rows.numpy()),
                 "scenes": 2, "sessions": 3,
                 "split_sha256": probe.SPLIT_SHA256,
                 "supervision_manifest_sha256": probe.SUPERVISION_SHA256,
                 "canonical_calibration_sha256": probe.CALIBRATION_SHA256,
                 "ego_cache_sha256": "4" * 64, "augment": False,
                 "frame_stride": 1 if split == "train" else 5},
        "feature_contract": {"dtype": "float32", "dimension": probe.FEATURE_DIM,
                             "components": list(probe.FEATURE_COMPONENTS),
                             "allowed": [item["name"] for item in probe.FEATURE_COMPONENTS]},
        "candidate_order": list(probe.CANDIDATE_ORDER),
        "class_order": {"zero": probe.ZERO, "move": probe.MOVE},
        "immutable_tune_rowwise_plan_gt_d3_bitwise_verified": split == "tune",
        "final_val_accessed": False, "selection_performed": False,
    }
    payload = {"features": features, "candidate_plans": candidate.float(),
               "candidate_costs": costs.float(), "labels": labels, "gt_plan": gt.float(),
               "rows": rows, "metadata": metadata}
    cache_path = tmp_path / f"{split}.pth"
    torch.save(payload, cache_path)
    ids = [{"row": int(row), "scenario": f"scene{row // 4}",
            "session": (probe.SESSION_046 if row == row_offset else f"session{row // 4}"),
            "frame": 30 + int(row)} for row in rows]
    manifest = copy.deepcopy(metadata)
    manifest.update(cache_sha256=sha(cache_path), ids=ids,
                    diagnostic_buckets=[("steady", "depart", "nonstop")[i % 3] for i in range(n)])
    manifest_path = tmp_path / f"{split}.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return cache_path, manifest_path, payload, manifest


def test_head_architecture_and_initialization_are_exact():
    torch.manual_seed(2)
    head = probe.ZeroSelectorHead()
    assert [type(module).__name__ for module in head.network] == [
        "LayerNorm", "Linear", "SiLU", "Linear"]
    assert head.network[1].in_features == 790 and head.network[1].out_features == 64
    assert torch.count_nonzero(head.network[-1].weight) == 0
    assert torch.equal(head.network[-1].bias, torch.tensor([-2., 2.]))
    logits = head(torch.randn(5, 790))
    assert torch.equal(logits, torch.tensor([[-2., 2.]]).repeat(5, 1))
    assert probe.select_classes(logits).tolist() == [probe.MOVE] * 5


def test_inference_tie_strictly_keeps_move():
    logits = torch.tensor([[1., 1.], [2., 1.], [1., 2.]])
    assert probe.select_classes(logits).tolist() == [probe.MOVE, probe.ZERO, probe.MOVE]


@pytest.mark.parametrize("update,expected", [
    (1, 1e-5), (100, 1e-3), (550, 5e-4), (1000, 0.)])
def test_fixed_warmup_cosine_schedule(update, expected):
    assert probe.learning_rate(update) == pytest.approx(expected, abs=1e-12)


def test_loss_is_exact_ce_plus_quarter_expected_d3_over_train_scale():
    logits = torch.tensor([[0., 0.], [2., -1.]], requires_grad=True)
    labels = torch.tensor([1, 0])
    costs = torch.tensor([[2., 1.], [3., 4.]])
    value, parts = probe.selector_loss(logits, labels, costs, 2.)
    expected = torch.nn.functional.cross_entropy(logits, labels) + \
        .25 * (logits.softmax(1) * costs).sum(1).mean() / 2.
    assert torch.equal(value, expected)
    value.backward()
    assert torch.isfinite(logits.grad).all()
    assert set(parts) == {"ce", "expected_d3"}


def test_deterministic_batch_stream_is_exact_batch128_and_seeded():
    one = probe._batch_indices(200, torch.Generator().manual_seed(7))
    two = probe._batch_indices(200, torch.Generator().manual_seed(7))
    a, b = next(one), next(two)
    assert len(a) == probe.BATCH and torch.equal(a, b) and len(torch.unique(a)) == probe.BATCH


def test_full_cache_load_recomputes_costs_labels_and_pins_hash(tmp_path, monkeypatch):
    monkeypatch.setitem(probe.EXPECTED, "train", 8)
    monkeypatch.setitem(probe.EXPECTED_INVENTORY, "train", {"scenes": 2, "sessions": 3})
    cp, mp, payload, manifest = cache_artifact(tmp_path, "train")
    loaded, sidecar = probe.load_cache(cp, sha(cp), mp, sha(mp), split="train", base_seed=0)
    assert torch.equal(loaded["features"], payload["features"])
    assert sidecar["ids"] == manifest["ids"]


@pytest.mark.parametrize("mutation", ["pilot", "extra_key", "feature", "zero", "cost", "label", "rows"])
def test_cache_load_fails_closed(mutation, tmp_path, monkeypatch):
    monkeypatch.setitem(probe.EXPECTED, "train", 8)
    monkeypatch.setitem(probe.EXPECTED_INVENTORY, "train", {"scenes": 2, "sessions": 3})
    cp, mp, payload, manifest = cache_artifact(tmp_path, "train", pilot=mutation == "pilot")
    if mutation == "extra_key": payload["raw_goal"] = torch.zeros(8, 2)
    if mutation == "feature": payload["features"] = payload["features"].double()
    if mutation == "zero": payload["candidate_plans"][0, 0, 0, 0] = 1.
    if mutation == "cost": payload["candidate_costs"][0, 0] += 1.
    if mutation == "label": payload["labels"][0] = 1 - payload["labels"][0]
    if mutation == "rows": payload["rows"][0] += 100
    if mutation != "pilot":
        torch.save(payload, cp)
        manifest["cache_sha256"] = sha(cp)
        mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        probe.load_cache(cp, sha(cp), mp, sha(mp), split="train", base_seed=0)


def test_train_tune_pair_requires_same_base_and_disjoint_rows(tmp_path, monkeypatch):
    monkeypatch.setitem(probe.EXPECTED, "train", 8)
    monkeypatch.setitem(probe.EXPECTED, "tune", 8)
    monkeypatch.setitem(probe.EXPECTED_INVENTORY, "train", {"scenes": 2, "sessions": 3})
    monkeypatch.setitem(probe.EXPECTED_INVENTORY, "tune", {"scenes": 2, "sessions": 3})
    tr_cp, tr_mp, _, _ = cache_artifact(tmp_path, "train", row_offset=0)
    tu_cp, tu_mp, _, _ = cache_artifact(tmp_path, "tune", row_offset=100)
    train, tm = probe.load_cache(tr_cp, sha(tr_cp), tr_mp, sha(tr_mp), split="train", base_seed=0)
    tune, um = probe.load_cache(tu_cp, sha(tu_cp), tu_mp, sha(tu_mp), split="tune", base_seed=0)
    assert probe.validate_cache_pair(train, tm, tune, um, 0)
    tune["rows"][0] = train["rows"][0]
    with pytest.raises(ValueError, match="overlap"):
        probe.validate_cache_pair(train, tm, tune, um, 0)


def test_summary_reports_regret_buckets_sessions_and_046_exclusion():
    costs = torch.tensor([[0., 2.], [3., 1.], [0., 4.], [2., 1.]])
    labels = torch.tensor([0, 1, 0, 1])
    selections = torch.tensor([1, 1, 0, 0])
    ids = [{"session": probe.SESSION_046}, {"session": "s1"},
           {"session": "s1"}, {"session": "s2"}]
    result = probe.summarize(costs, labels, selections, ids,
                             ["steady", "depart", "nonstop", "nonstop"])
    assert result["all"]["selected_d3"] == 1.25
    assert result["all"]["regret_to_oracle"] == 0.75
    assert result["buckets"]["steady"]["n"] == 1
    assert result["excluding_session_046"]["n"] == 3
    assert set(result["sessions"]) == {probe.SESSION_046, "s1", "s2"}


def test_tune_bootstrap_is_whole_session_frame_weighted_and_fixed():
    costs = torch.tensor([[2., 1.], [0., 3.], [4., 2.], [1., 1.]])
    labels = torch.tensor([1, 0, 1, 1])
    selected = torch.tensor([1, 0, 0, 1])
    ids = [{"session": "a"}, {"session": "a"}, {"session": "a"}, {"session": "b"}]
    value = probe.summarize(costs, labels, selected, ids,
                            ["steady", "depart", "nonstop", "nonstop"], session_bootstrap=True)
    boot = value["selector_minus_original_p4_session_cluster"]
    assert boot["delta"] == pytest.approx(((1-1) + (0-3) + (4-2) + (1-1)) / 4)
    assert boot["repeats"] == probe.BOOTSTRAP_REPEATS and boot["seed"] == probe.BOOTSTRAP_SEED
    assert boot["clusters"] == 2 and not boot["frame_iid_bootstrap"]


def test_tiny_training_uses_train_scale_and_accesses_tune_only_after_last(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "UPDATES", 4)
    monkeypatch.setattr(probe, "WARMUP", 1)
    monkeypatch.setattr(probe, "BATCH", 4)
    monkeypatch.setattr(probe, "BOOTSTRAP_REPEATS", 100)
    monkeypatch.setitem(probe.EXPECTED, "train", 8)
    monkeypatch.setitem(probe.EXPECTED, "tune", 8)
    monkeypatch.setitem(probe.EXPECTED_INVENTORY, "train", {"scenes": 2, "sessions": 3})
    monkeypatch.setitem(probe.EXPECTED_INVENTORY, "tune", {"scenes": 2, "sessions": 3})
    _, _, train, tm = cache_artifact(tmp_path, "train", row_offset=0)
    _, _, tune, um = cache_artifact(tmp_path, "tune", row_offset=100)
    head, optimizer, evidence = probe.train_head(
        train, tune, tm, um, base_seed=0, head_seed=1, device=torch.device("cpu"))
    assert evidence["tune_first_selector_forward_after_update"] == 4
    assert not evidence["tune_used_for_hyperparameters_or_selection"]
    assert evidence["initial_all_move_proof"]["all_train_and_tune_rows_select_move_without_head_forward"]
    assert not evidence["initial_all_move_proof"]["tune_selector_forward_performed_before_update1000"]
    assert evidence["c_scale_train_mean_d_zero"] == pytest.approx(
        max(float(train["candidate_costs"][:, 0].double().mean()), 1e-3))
    assert evidence["initial_head_state_sha256"] != evidence["final_head_state_sha256"]
    assert len(optimizer.state) > 0 and set(evidence["metrics"]) == {"train", "tune"}


def test_cli_allows_exact_six_run_indices_only():
    common = ["--train-cache", "tc", "--expected-train-cache-sha256", "a" * 64,
              "--train-manifest", "tm", "--expected-train-manifest-sha256", "b" * 64,
              "--tune-cache", "uc", "--expected-tune-cache-sha256", "c" * 64,
              "--tune-manifest", "um", "--expected-tune-manifest-sha256", "d" * 64,
              "--run-dir", "out"]
    for base in (0, 1):
        for head in (0, 1, 2):
            parsed = probe.arguments(common + ["--base-seed", str(base), "--head-seed", str(head)])
            assert (parsed.base_seed, parsed.head_seed) == (base, head)
    with pytest.raises(SystemExit):
        probe.arguments(common + ["--base-seed", "0", "--head-seed", "3"])


def test_recipe_has_no_tunable_threshold_lambda_or_checkpoint_selection():
    assert probe.RECIPE["updates"] == 1000 and probe.RECIPE["batch"] == 128
    assert probe.RECIPE["head_seeds"] == [0, 1, 2]
    assert probe.LAMBDA_EXPECTED_D3 == .25
    source = Path(probe.__file__).read_text()
    assert "best.pth" not in source and "final_val" in source


def test_cpu_contract_suite_does_not_initialize_cuda():
    assert not torch.cuda.is_initialized()
