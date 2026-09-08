from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import run_motiondrive_v2_gt7_vs_gt24 as probe


def test_mask_preserves_exact_state5_goal2_and_removes_17_fields():
    source = torch.randn(9, 24)
    masked = probe.mask_to_gt7(source)
    assert torch.equal(masked[:, :5], source[:, :5])
    assert torch.equal(masked[:, 22:], source[:, 22:])
    assert not bool(masked[:, 5:22].count_nonzero())
    assert probe.MASKED_INDICES == tuple(range(5, 22))


def test_mask_rejects_wrong_shape_dtype_and_nonfinite():
    for bad in (torch.randn(2, 23), torch.randn(2, 24).double(),
                torch.full((2, 24), float("nan"))):
        with pytest.raises(ValueError, match="masking requires"):
            probe.mask_to_gt7(bad)


def test_exact_recipe_reuses_fixed_24d_family():
    assert probe.base.N_EPOCHS == 60
    assert probe.base.BATCH_SIZE == 1024
    assert probe.base.N_STEPS == 3240
    model = probe.base.tiny_model()
    assert model[0].in_features == 24 and model[-1].out_features == 12


def test_a1_nominal_replaces_only_physical_state5_before_normalization():
    raw = torch.randn(4, 24, dtype=torch.float32)
    status = torch.randn(4, 5).numpy().astype("float32")
    changed = probe.replace_state5_with_nominal(raw, status)
    assert torch.equal(changed[:, :5], torch.from_numpy(status))
    assert torch.equal(changed[:, 5:], raw[:, 5:])


def test_actual_canonical_raw24_nominal_replacement_normalization_and_mask_composes():
    torch.manual_seed(7)
    labels = {
        "state_target": torch.randn(8, 6),
        "history_target": torch.randn(8, 4, 4),
        "goal_xy": torch.randn(8, 2),
    }
    labels["state_target"][:, 5] = torch.arange(8) % 2
    raw = probe.base.canonical_raw24(labels)
    assert raw.dtype == torch.float32
    normalizer = probe.base.fit_shared_normalizer(raw)
    status = torch.randn(8, 5).numpy().astype("float32")
    composed = probe.mask_to_gt7(probe.base.apply_normalizer(
        probe.replace_state5_with_nominal(raw, status), normalizer).float())
    assert composed.shape == (8, 24)
    assert not bool(composed[:, 5:22].count_nonzero())


def test_source_closure_is_exact_current_candidate():
    actual = probe.sha256(probe.__file__)
    sources = probe.validate_sources(actual)
    assert sources["scripts/run_motiondrive_v2_p7_state_sufficiency.py"] == probe.BASE_SCRIPT_SHA256
    with pytest.raises(ValueError, match="self SHA"):
        probe.validate_sources("0" * 64)


def test_terminal_evaluation_saves_same_pass_predictions_and_float64_metric():
    torch.manual_seed(4)
    model = probe.base.tiny_model().eval()
    x = torch.randn(7, 24)
    y = torch.randn(7, 6, 2)
    prediction, row_d3, metrics = probe.evaluate_once(model, x, y)
    assert prediction.shape == (7, 6, 2) and row_d3.shape == (7,)
    assert metrics["official_d3"] == pytest.approx(
        row_d3.numpy().astype("float64").mean(), rel=0, abs=0)
    assert set(metrics) == {"official_d3", "ade1", "ade2", "ade3"}
