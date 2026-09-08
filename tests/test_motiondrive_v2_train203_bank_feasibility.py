import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import analyze_motiondrive_v2_train203_bank_feasibility as bank
from scripts.motiondrive_v2_training import weighted_d3


def trajectories(n=5):
    value = np.zeros((n, 10, 2), np.float32)
    for i in range(n):
        value[i, :, 0] = (i + 1) * np.arange(1, 11, dtype=np.float32)
        value[i, :, 1] = i * .25
    return value


def selected_digest(ego, rows, fut5):
    digest = hashlib.sha256()
    names = ego["scenarios"].astype(str)[ego["scen_idx"].astype(np.int64)]
    for local, row in enumerate(np.asarray(rows, np.int64)):
        digest.update(str(names[row]).encode()); digest.update(b"\0")
        digest.update(np.asarray(row, dtype="<i8").tobytes())
        digest.update(np.asarray(ego["frame"][row], dtype="<i8").tobytes())
        digest.update(np.asarray(fut5[local], dtype="<f4").tobytes())
        digest.update(np.asarray(ego["goal"][row], dtype="<f4").tobytes())
    return digest.hexdigest()


def test_prefix_embedding_is_fixed_physical_sqrt_weighted_prefix_only():
    value = trajectories(2)
    changed_tail = value.copy(); changed_tail[:, 6:] += 999
    actual = bank.prefix_embedding(value)
    expected = (value[:, :6] * np.sqrt(bank.OFFICIAL_W)[None, :, None]).reshape(2, 12)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, bank.prefix_embedding(changed_tail))


def test_official_d3_matches_training_implementation_and_broadcasts():
    rng = np.random.default_rng(3)
    pred = rng.normal(size=(4, 6, 2)).astype(np.float32)
    gt = rng.normal(size=(4, 6, 2)).astype(np.float32)
    expected = weighted_d3(torch.from_numpy(pred), torch.from_numpy(gt)).numpy()
    assert np.array_equal(bank.official_d3(pred, gt), expected)
    matrix = bank.official_d3(pred[:, None], gt[None])
    assert matrix.shape == (4, 4)
    assert np.array_equal(np.diag(matrix), expected)


def test_unique_pool_excludes_exact_zero_and_deduplicates_full10_by_lowest_row():
    value = trajectories(4)
    value[1] = value[0]
    value[3] = 0
    rows, unique = bank.unique_nonzero_pool(value, np.asarray([9, 4, 7, 2]))
    assert rows.tolist() == [4, 7]
    assert len({row.tobytes() for row in unique}) == 2
    assert not np.any(np.all(unique == 0, axis=(1, 2)))


def test_unique_medoid_assignment_ties_by_source_row_and_never_reuses():
    centers = np.asarray([[0.], [0.], [10.]], np.float32)
    features = np.asarray([[1.], [-1.], [9.]], np.float32)
    chosen = bank.assign_unique_medoids(centers, features, np.asarray([8, 3, 5]))
    assert chosen.tolist() == [1, 0, 2]


def test_validate_bank_rejects_duplicate_full_trajectory(monkeypatch):
    monkeypatch.setattr(bank, "K_BANK", 4); monkeypatch.setattr(bank, "K_CLUSTER", 3)
    value = trajectories(3)
    artifact = np.concatenate([np.zeros((1, 10, 2), np.float32), value])
    bank.validate_bank(artifact, np.asarray([-1, 10, 11, 12], np.int64))
    artifact[3] = artifact[2]
    with pytest.raises(ValueError, match="not unique"):
        bank.validate_bank(artifact, np.asarray([-1, 10, 11, 12], np.int64))


def test_validate_bank_rejects_nonzero_row0_or_duplicate_source(monkeypatch):
    monkeypatch.setattr(bank, "K_BANK", 4); monkeypatch.setattr(bank, "K_CLUSTER", 3)
    artifact = np.concatenate([np.zeros((1, 10, 2), np.float32), trajectories(3)])
    bad = artifact.copy(); bad[0, 0, 0] = 1
    with pytest.raises(ValueError, match="row0"):
        bank.validate_bank(bad, np.asarray([-1, 10, 11, 12], np.int64))
    with pytest.raises(ValueError, match="source rows"):
        bank.validate_bank(artifact, np.asarray([-1, 10, 10, 12], np.int64))


def test_bank_sources_must_be_current_train_and_intact_full10(monkeypatch):
    monkeypatch.setattr(bank, "K_BANK", 4); monkeypatch.setattr(bank, "K_CLUSTER", 3)
    ego5 = trajectories(8)
    sources = np.asarray([-1, 2, 4, 6], np.int64)
    artifact = np.concatenate([np.zeros((1, 10, 2), np.float32), ego5[sources[1:]]])
    bank.validate_bank_sources(artifact, sources, np.asarray([2, 4, 6, 7]), ego5)
    with pytest.raises(ValueError, match="outside"):
        bank.validate_bank_sources(artifact, sources, np.asarray([2, 4, 7]), ego5)
    changed = artifact.copy(); changed[2, 9, 0] += .01
    with pytest.raises(ValueError, match="intact"):
        bank.validate_bank_sources(changed, sources, np.asarray([2, 4, 6]), ego5)


def test_build_bank_uses_intact_real_rows_and_stable_published_order(monkeypatch):
    monkeypatch.setattr(bank, "K_BANK", 4); monkeypatch.setattr(bank, "K_CLUSTER", 3)
    settings = dict(bank.KMEANS_SETTINGS); settings["n_clusters"] = 3
    monkeypatch.setattr(bank, "KMEANS_SETTINGS", settings)

    class FakeKMeans:
        def __init__(self, **kwargs):
            assert kwargs["n_clusters"] == 3
            self.cluster_centers_ = np.asarray([[1.] * 12, [3.] * 12, [2.] * 12])
            self.n_iter_, self.inertia_ = 2, 1.5

        def fit_predict(self, values):
            return np.asarray([0, 2, 2, 1], np.int64)

    monkeypatch.setattr(bank, "MiniBatchKMeans", FakeKMeans)
    value = trajectories(4)
    rows = np.asarray([20, 21, 22, 23], np.int64)
    artifact, sources, clusters, support, _ = bank.build_bank(value, rows)
    assert np.all(artifact[0] == 0)
    assert set(sources[1:].tolist()) <= set(rows.tolist())
    for output, source in zip(artifact[1:], sources[1:]):
        assert np.array_equal(output, value[np.flatnonzero(rows == source)[0]])
    assert support[1:].tolist() == sorted(support[1:].tolist(), reverse=True)


def test_stable_shortlist_tie_breaks_by_candidate_id():
    pred = np.zeros((1, 6, 2), np.float32)
    artifact = np.zeros((13, 10, 2), np.float32)
    artifact[:, 6:, 0] = np.arange(13)[:, None]
    ids, distance = bank.stable_shortlist(pred, artifact)
    assert np.all(distance == 0)
    assert ids.tolist() == [list(range(12))]


def test_proxy_shortlist_uses_prediction_not_gt_and_selectors_return_bank_rows():
    artifact = np.zeros((20, 10, 2), np.float32)
    artifact[:, :, 0] = np.arange(20)[:, None]
    pred = np.zeros((2, 6, 2), np.float32); pred[0, :, 0] = 7; pred[1, :, 0] = 9
    gt_a = np.zeros_like(pred); gt_b = np.full_like(pred, 100)
    goal = np.asarray([[19., 0.], [1., 0.]], np.float32)
    first = bank.proxy_selectors(pred, gt_a, goal, artifact)
    second = bank.proxy_selectors(pred, gt_b, goal, artifact)
    assert np.array_equal(first["shortlist_ids"], second["shortlist_ids"])
    assert first["prefix_choice"].tolist() == [7, 9]
    assert first["endpoint_choice"].tolist() == [12, 3]
    for key in ("prefix_choice", "endpoint_choice"):
        chosen = first[key]
        assert np.array_equal(artifact[chosen], artifact[first["shortlist_ids"]][
            np.arange(2), np.argmax(first["shortlist_ids"] == chosen[:, None], axis=1)])


def test_full_bank_and_m12_oracles_are_distinct_coverage_bounds():
    artifact = np.zeros((13, 10, 2), np.float32)
    artifact[:, :, 0] = np.arange(13)[:, None]
    gt = artifact[12:13]
    vectors = bank.per_metric_oracles(gt, artifact, chunk=3)
    assert vectors["d3"][0] == 0 and vectors["endpoint5"][0] == 0
    pred = artifact[0:1, :6]
    proxy = bank.proxy_selectors(pred, gt[:, :6], gt[:, 9], artifact)
    assert proxy["shortlist_oracle"][0] > 0


def test_full_bank_d3_oracle_uses_frozen_torch_fp32_metric_order():
    rng = np.random.default_rng(19)
    gt = rng.normal(size=(3, 10, 2)).astype(np.float32)
    artifact = rng.normal(size=(7, 10, 2)).astype(np.float32)
    observed = bank.per_metric_oracles(gt, artifact, chunk=2)["d3"]
    expected = bank.official_d3(gt[:, None, :6], artifact[None, :, :6]).min(1)
    assert np.array_equal(observed, expected)


def test_selected_trajectories_requires_prefix_endpoint_mask_and_pinned_digest(monkeypatch):
    fut5 = trajectories(2)
    ego = {"scenarios": np.asarray(["s"]), "scen_idx": np.asarray([0, 0]),
           "frame": np.asarray([30, 31]), "fut": fut5[:, :6].copy(),
           "goal": fut5[:, 9].copy()}
    ego5 = {"fut5": fut5.copy(), "mask5": np.ones((2, 10), np.float32)}
    rows = np.asarray([0, 1], np.int64)
    monkeypatch.setitem(bank.EXPECTED_SELECTED_IDENTITY_SHA256, "train",
                        selected_digest(ego, rows, fut5))
    value, mask = bank.selected_trajectories(ego, ego5, rows, split="train")
    assert np.array_equal(value, fut5) and mask.all()
    bad = dict(ego5); bad["mask5"] = ego5["mask5"].copy(); bad["mask5"][0, 9] = 0
    with pytest.raises(ValueError, match="all ten"):
        bank.selected_trajectories(ego, bad, rows, split="train")
    bad = dict(ego5); bad["fut5"] = ego5["fut5"].copy(); bad["fut5"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="prefix"):
        bank.selected_trajectories(ego, bad, rows, split="train")
    bad = dict(ego5); bad["fut5"] = ego5["fut5"].copy(); bad["fut5"][0, 9, 0] += 1
    with pytest.raises(ValueError, match="endpoint"):
        bank.selected_trajectories(ego, bad, rows, split="train")


def test_control_report_join_is_exact_for_rows_gt_and_predictions():
    identity = [(4, "scene", "session", 30)]
    gt = np.zeros((1, 6, 2), np.float32)
    rows = [{"identity": identity[0], "gt": gt[0], "pred": gt[0] + 1}]
    pred, observed_gt = bank.validated_control_arrays(rows, identity, gt, name="C0")
    assert np.array_equal(observed_gt, gt) and np.array_equal(pred, gt + 1)
    with pytest.raises(ValueError, match="row order"):
        bank.validated_control_arrays(rows, [(5, "scene", "session", 30)], gt, name="C0")
    bad = [{**rows[0], "gt": gt[0] + 1}]
    with pytest.raises(ValueError, match="GT"):
        bank.validated_control_arrays(bad, identity, gt, name="C0")


def test_fit_stage_never_selects_tune_or_reports(monkeypatch, tmp_path):
    inputs = []
    for name in ("split", "ego", "ego5"):
        path = tmp_path / name; path.write_bytes(name.encode()); inputs.append(path)
    receipts = {name: {"path": str(path), "sha256": bank.file_sha(path), "bytes": path.stat().st_size}
                for name, path in zip(("split", "ego", "ego5"), inputs)}
    monkeypatch.setattr(bank, "load_split", lambda path: ({}, receipts["split"]))
    monkeypatch.setattr(bank, "load_cache_metadata",
                        lambda a, b: ({"frame": np.arange(2)}, {}, np.asarray(["s", "s"]),
                                      {"ego": receipts["ego"], "ego5": receipts["ego5"]}))
    seen = []
    monkeypatch.setattr(bank, "selected_rows", lambda *args: seen.append(args[-1]) or np.arange(2))
    monkeypatch.setattr(bank, "selected_trajectories",
                        lambda *args, **kwargs: (seen.append(kwargs["split"]) or trajectories(2), None))
    artifact = np.concatenate([np.zeros((1, 10, 2), np.float32), trajectories(1023)])
    sources = np.concatenate([[-1], np.arange(1023)]).astype(np.int64)
    clusters = sources.astype(np.int32); supports = np.ones(1024, np.int64)

    class Model:
        n_iter_, inertia_ = 1, 0.
    monkeypatch.setattr(bank, "build_bank", lambda *args: (artifact, sources, clusters, supports, Model()))
    monkeypatch.setattr(bank, "require_runtime", lambda: None)
    output = tmp_path / "new-output"
    args = argparse.Namespace(output_dir=output, split_manifest=inputs[0],
                              ego_cache=inputs[1], ego5_cache=inputs[2])
    assert bank.fit(args) == 0
    assert seen == ["train", "train"]
    manifest = json.loads((output / "bank_manifest.json").read_text())
    assert manifest["scope"]["tune_labels_accessed"] is False
    assert manifest["scope"]["shared_ego5_bytes_include_unselected_rows"] is True
    assert manifest["scope"]["unselected_rows_used_for_fit_statistics_metrics_or_selection"] is False


def test_fit_cli_has_no_tune_or_prediction_inputs():
    args = bank.arguments(["fit", "--split-manifest", "s", "--ego-cache", "e",
                           "--ego5-cache", "e5", "--output-dir", "o"])
    assert args.phase == "fit"
    assert not hasattr(args, "c0_eval") and not hasattr(args, "output")


def test_evaluate_requires_external_frozen_manifest_and_exact_c_inputs():
    sha = "a" * 64
    args = bank.arguments(["evaluate", "--split-manifest", "s", "--ego-cache", "e",
        "--ego5-cache", "e5", "--bank-manifest", "bm", sha,
        "--p7-source-manifest", "sm", sha, "--c0-eval", "c0", sha,
        "--c0-manifest", "m0", sha, "--c1-eval", "c1", sha,
        "--c1-manifest", "m1", sha, "--output", "out"])
    assert args.phase == "evaluate" and args.bank_manifest == (Path("bm"), sha)


def test_runtime_requires_cpu_single_thread(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    seen = []
    monkeypatch.setattr(bank, "validate_runtime_versions", lambda *values: seen.append(values))
    bank.require_runtime()
    assert len(seen) == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        bank.require_runtime()


def test_fit_runtime_versions_are_exactly_pinned():
    expected = bank.EXPECTED_FIT_RUNTIME
    bank.validate_runtime_versions(expected["python"], expected["numpy"],
                                   expected["scipy"], expected["sklearn"])
    with pytest.raises(ValueError, match="runtime mismatch"):
        bank.validate_runtime_versions(expected["python"], expected["numpy"],
                                       expected["scipy"], "different")


def test_json_serializer_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "report.json"
    bank.write_new_json(path, {"x": np.float32(1.25), "a": np.asarray([1, 2])})
    assert json.loads(path.read_text()) == {"a": [1, 2], "x": 1.25}
    with pytest.raises(ValueError, match="overwrite"):
        bank.write_new_json(path, {})


def test_boundaries_and_fixed_recipe_are_source_explicit():
    source = Path(bank.__file__).read_text()
    assert "full_bank_oracle_is_representation_coverage_only" in source
    assert "negative_m12_result_does_not_close_multimodal_generation" in source
    assert "c_predictions_are_already_indirectly_goal_conditioned" in source
    assert "bank_frozen_before_any_tune_label_or_prediction_access" in source
    assert "P_VALUES" not in source and "V_VALUES" not in source
    assert bank.K_BANK == 1024 and bank.SHORTLIST_M == 12 and bank.FIT_SEED == 20260908
    assert bank.KMEANS_SETTINGS["n_clusters"] == 1023


def frozen_manifest():
    recipe = {
        "bank_rows": bank.K_BANK, "reserved_exact_zero_rows": 1,
        "cluster_rows": bank.K_CLUSTER, "seed": bank.FIT_SEED,
        "embedding": "flatten(sqrt([11,11,5,5,2,2]/36)*metric_xy_first6)",
        "embedding_dtype": "float32", "trajectory_dtype": "float32",
        "objective_boundary": "weighted squared K-means surrogate, not exact official D3",
        "tail_used_in_fit": False, "scaler": None,
        "kmeans": dict(bank.KMEANS_SETTINGS), "threadpool_limit": 1,
        "medoid": "unique full10 real train tensor; nearest squared embedding; ties source row",
        "published_order": "support descending, source row ascending, original cluster id ascending"}
    return {
        "schema_version": 1,
        "status": "bank_frozen_before_any_tune_label_or_prediction_access",
        "scope": {"train_rows_only": True, "tune_labels_accessed": False,
                  "final_rows_accessed": False, "gpu_used": False,
                  "shared_ego5_bytes_include_unselected_rows": True,
                  "unselected_rows_used_for_fit_statistics_metrics_or_selection": False},
        "inputs": {
            "split": {"sha256": bank.EXPECTED_SPLIT_SHA256},
            "ego": {"sha256": bank.EXPECTED_EGO_SHA256},
            "ego5": {"sha256": bank.EXPECTED_EGO5_SHA256,
                     "bytes": bank.EXPECTED_EGO5_BYTES}},
        "train": {"scenes": 203, "rows": bank.EXPECTED_TRAIN_ROWS,
                  "rows_sha256": bank.EXPECTED_TRAIN_ROWS_SHA256,
                  "selected_identity_sha256": bank.EXPECTED_SELECTED_IDENTITY_SHA256["train"],
                  "fut5_sha256": "f" * 64},
        "recipe": recipe, "fit": {"n_iter": 3, "inertia": 1.},
        "artifacts": {name: {"path": name, "sha256": "a" * 64}
                      for name in ("bank.npy", "source_rows.npy", "cluster_ids.npy",
                                   "cluster_support.npy")},
        "environment": {"python": bank.EXPECTED_FIT_RUNTIME["python"] + " details",
                        "numpy": bank.EXPECTED_FIT_RUNTIME["numpy"],
                        "scipy": bank.EXPECTED_FIT_RUNTIME["scipy"],
                        "sklearn": bank.EXPECTED_FIT_RUNTIME["sklearn"],
                        "expected_fit_runtime": dict(bank.EXPECTED_FIT_RUNTIME),
                        "threads": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                    "MKL_NUM_THREADS": "1"}},
    }


@pytest.mark.parametrize("mutation", ["recipe", "input", "rows", "runtime"])
def test_frozen_manifest_fails_closed_on_recipe_and_provenance_mutation(mutation):
    value = frozen_manifest()
    bank.validate_frozen_manifest_contract(value)
    bad = copy.deepcopy(value)
    if mutation == "recipe":
        bad["recipe"]["seed"] += 1
    elif mutation == "input":
        bad["inputs"]["ego5"]["sha256"] = "b" * 64
    elif mutation == "rows":
        bad["train"]["rows"] -= 1
    else:
        bad["environment"]["sklearn"] = "other"
    with pytest.raises(ValueError):
        bank.validate_frozen_manifest_contract(bad)
