"""Isolated P3 image-state query experiment; the legacy V2 sources stay intact.

Both arms retain the original scene, raw motion and image-predicted state token.
Only the new query adapter sees either the SAME predicted 22-vector or zeros.
This is a neural query residual, never a trajectory/kinematic postprocessor.

Migration reads a SHA-pinned, trusted local trainer checkpoint (which may contain
Python/NumPy RNG pickles). It verifies the complete legacy state with strict=True,
then adds exactly four explicitly initialized tensors and strictly loads the new
architecture. It neither writes a checkpoint nor certifies run/OS-exit lineage.
Training freeze policy, optimizer, dataset and deployment packaging are external.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn

from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
from models.motiondrive_v2.planner import DirectTrajectoryPlanner
from scripts.export_motiondrive_v2_inference import validate_complete_config
from scripts.motiondrive_v2_training import tensor_state_sha256


ARCHITECTURE = "motiondrive_v2_image_state_query_v1"
ADAPTER_STATE_KEYS = frozenset(
    f"planner.query_adapter.{layer}.{kind}"
    for layer in (0, 2) for kind in ("weight", "bias"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _boolean(value):
    _require(type(value) is bool, "query_adapter_on must be an explicit boolean")
    return value


def _config(value):
    saved = value.to_dict() if isinstance(value, MotionDriveV2Config) else value
    result = validate_complete_config(saved)
    _require(result.n_history == 4, "P3 requires exactly four image-inferred histories (22-vector)")
    _require(result.state_on is True, "P3 retains the legacy image-state token ON in BOTH arms")
    return result


def _device(value):
    result = torch.device(value)
    _require(result.type in ("cpu", "cuda") and
             (result.type != "cuda" or result.index is not None),
             "Use cpu or an explicit logical cuda:N device")
    return result


def _strict_state(model, state):
    _require(isinstance(state, Mapping), "model state must be a tensor mapping")
    target = model.state_dict()
    _require(set(state) == set(target), "Strict model state key mismatch")
    for name, expected in target.items():
        tensor = state[name]
        _require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided,
                 f"Invalid tensor state: {name}")
        _require(tensor.shape == expected.shape and tensor.dtype == expected.dtype,
                 f"Strict model state shape/dtype mismatch: {name}")
        _require(bool(torch.isfinite(tensor).all()), f"Nonfinite model state: {name}")
    model.load_state_dict(state, strict=True)


class ImageStateQueryPlanner(DirectTrajectoryPlanner):
    """Adds query conditioning without removing any legacy memory input."""

    def __init__(self, config, query_adapter_on=False):
        config = _config(config)
        super().__init__(config)
        self.query_adapter_on = _boolean(query_adapter_on)
        c = config.channels
        self.query_adapter = nn.Sequential(nn.Linear(c + 22, c), nn.GELU(), nn.Linear(c, c))
        # Only the final projection is zero: W1 remains able to provide a
        # nonconstant feature and W2 receives a gradient on the first update.
        nn.init.zeros_(self.query_adapter[2].weight)
        nn.init.zeros_(self.query_adapter[2].bias)

    def adapt_queries(self, queries, normalized_status):
        """The scalar policy changes information, not the executed MLP shape."""
        _require(normalized_status.shape == (queries.shape[0], 22), "Expected normalized [B,22] image state")
        with torch.autocast(device_type=queries.device.type, enabled=False):
            queries = queries.float()
            used = normalized_status.float() * float(_boolean(self.query_adapter_on))
            evidence = used[:, None].expand(-1, 6, -1)
            return queries + self.query_adapter(torch.cat([queries, evidence], dim=-1))

    def forward(self, scene_features, motion_features, predicted_state, predicted_history):
        # Mirrors the small legacy planner forward; only query construction is
        # extended. Existing parameter/buffer names and memory order stay intact.
        _require(self.config.state_on is True, "The original state token must remain ON")
        b = scene_features.shape[0]
        _require(predicted_state.shape == (b, 6) and predicted_history.shape == (b, 4, 4),
                 "Expected image-predicted state [B,6] and history [B,4,4]")
        with torch.autocast(device_type=scene_features.device.type, enabled=False):
            scene = scene_features.float() + self.scene_position(self.scene_xy.float())[None]
            motion = motion_features.float() + self.motion_type.float()
            state = predicted_state.float() / self.state_scale
            history = (predicted_history.float() / self.history_scale).flatten(1)
            status = torch.cat([state, history], -1)
            state_token = self.state_projection(status)[:, None]
            memory = torch.cat([scene, motion, state_token], 1)
            queries = self.waypoint_queries.float()[None].expand(b, -1, -1)
            queries = self.adapt_queries(queries, status)
            decoded = self.decoder(queries, memory)
            normalized_xy = self.xy_head(decoded.float())
            return normalized_xy * normalized_xy.new_tensor(self.config.plan_output_scale)


class MotionDriveV2QueryAdapter(MotionDriveV2):
    """Same six-input public forward and feature APIs, explicitly new architecture."""

    architecture = ARCHITECTURE

    def __init__(self, config, query_adapter_on=False):
        config = _config(config)
        super().__init__(config)
        self.planner = ImageStateQueryPlanner(config, query_adapter_on)

    @property
    def query_adapter_on(self):
        return self.planner.query_adapter_on

    @query_adapter_on.setter
    def query_adapter_on(self, value):
        self.planner.query_adapter_on = _boolean(value)


def model_from_query_payload(payload, device="cpu"):
    """Strict P3 restore with explicit metadata; never infer missing config/flags.

    Required top-level keys: architecture, query_adapter_on, model_config, model.
    If a training manifest is present, its architecture/flag/config must also be
    explicit and agree. Other trainer fields (step/optimizer/RNG) are not loaded.
    This function validates model identity, not provenance of a dict's file.
    """
    device = _device(device)
    _require(isinstance(payload, Mapping), "Expected a P3 payload mapping")
    _require(payload.get("architecture") == ARCHITECTURE, "Unsupported or missing P3 architecture")
    flag = _boolean(payload.get("query_adapter_on"))
    config = _config(payload.get("model_config"))
    if "manifest" in payload:
        manifest = payload["manifest"]
        _require(isinstance(manifest, Mapping), "P3 manifest must be a mapping")
        _require(manifest.get("architecture") == ARCHITECTURE, "Manifest architecture mismatch")
        _require(_boolean(manifest.get("query_adapter_on")) == flag, "Manifest query_adapter_on mismatch")
        manifest_config = _config(manifest.get("model_config"))
        _require(config.to_dict() == manifest_config.to_dict(), "Manifest model_config mismatch")
    # Construction/strict restoration must not advance the training sampler RNG.
    with torch.random.fork_rng(devices=[]):
        model = MotionDriveV2QueryAdapter(config, flag)
        _strict_state(model, payload.get("model"))
    return model.to(device).eval()


def _stream_sha256(stream):
    digest = hashlib.sha256()
    stream.seek(0)
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def migrate_legacy_checkpoint(path, expected_sha256, adapter_seed=0, adapter_on=False, device="cpu"):
    """Return a migrated model and JSON-safe evidence without altering any file.

    ``path`` must be a trusted local training checkpoint: weights_only=False is
    necessary for the legacy trainer's NumPy/Python RNG payload. The caller pins
    its full SHA and independently verifies completed C1/T1/OS-exit lineage.
    """
    device = _device(device)
    flag = _boolean(adapter_on)
    _require(type(adapter_seed) is int and 0 <= adapter_seed < 2 ** 63,
             "adapter_seed must be an integer in [0, 2**63)")
    _require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
             "Explicit lowercase full checkpoint SHA256 is required")
    path = Path(path).resolve(strict=True)
    _require(path.is_file(), "Legacy checkpoint must be a file")
    with path.open("rb") as stream:
        observed_sha = _stream_sha256(stream)
        _require(observed_sha == expected_sha256, "Legacy checkpoint SHA256 mismatch")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=False)
        _require(_stream_sha256(stream) == observed_sha, "Legacy checkpoint changed during loading")
    _require(isinstance(payload, dict) and isinstance(payload.get("manifest"), dict),
             "Legacy checkpoint requires an explicit manifest")
    _require("architecture" not in payload and "query_adapter_on" not in payload,
             "Migration accepts legacy checkpoints, not an already augmented payload")
    config = _config(payload["manifest"].get("model_config"))
    _require(type(payload.get("step")) is int and payload["step"] > 0,
             "Legacy checkpoint requires a positive integer step")
    original = payload.get("model")
    with torch.random.fork_rng(devices=[]):
        legacy = MotionDriveV2(config)
        _strict_state(legacy, original)
        old_keys = set(legacy.state_dict())
        original_sha = tensor_state_sha256(legacy.state_dict())
        torch.random.default_generator.manual_seed(adapter_seed)
        model = MotionDriveV2QueryAdapter(config, flag)
        initial = model.state_dict()
        _require(set(initial) - old_keys == ADAPTER_STATE_KEYS and old_keys <= set(initial),
                 "Migration must add exactly the four declared adapter tensors")
        # Initialize W1 from a seed scoped to this adapter, independent of any
        # constructor random draws in the frozen backbone/legacy planner.
        torch.random.default_generator.manual_seed(adapter_seed)
        model.planner.query_adapter[0].reset_parameters()
        initial.update({name: tensor.detach().clone() for name, tensor in original.items()})
        _strict_state(model, initial)
        migrated = model.state_dict()
        _require(all(torch.equal(migrated[name], original[name]) for name in old_keys),
                 "Migration altered a legacy tensor")
        _require(tensor_state_sha256({name: migrated[name] for name in old_keys}) == original_sha,
                 "Migration changed legacy tensor bytes")
        adapter_state = {name: migrated[name] for name in ADAPTER_STATE_KEYS}
        report = {
            "schema_version": 1, "architecture": ARCHITECTURE, "query_adapter_on": flag,
            "source_checkpoint_path": str(path), "source_checkpoint_sha256": observed_sha,
            "source_checkpoint_step": payload["step"], "source_model_state_sha256": original_sha,
            "initial_model_state_sha256": tensor_state_sha256(migrated),
            "adapter_state_sha256": tensor_state_sha256(adapter_state),
            "added_state_keys": sorted(ADAPTER_STATE_KEYS),
            "adapter_parameter_count": sum(t.numel() for t in adapter_state.values()),
            "adapter_seed": adapter_seed, "legacy_tensors_bitwise_preserved": True,
            "final_adapter_projection_exact_zero": bool(
                torch.count_nonzero(model.planner.query_adapter[2].weight) == 0 and
                torch.count_nonzero(model.planner.query_adapter[2].bias) == 0),
            "source_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "optimizer_state_loaded": False, "run_or_os_exit_lineage_verified": False,
            "initial_prediction_parity_verified": False,
            "scope": "strict tensor migration only; caller must verify physical predictions and run lineage",
        }
    # A second path hash also catches source replacement after the opened stream.
    with path.open("rb") as stream:
        _require(_stream_sha256(stream) == observed_sha, "Legacy checkpoint changed during migration")
    json.dumps(report, allow_nan=False)
    return model.to(device).eval(), report


__all__ = ["ARCHITECTURE", "ADAPTER_STATE_KEYS", "ImageStateQueryPlanner",
           "MotionDriveV2QueryAdapter", "migrate_legacy_checkpoint", "model_from_query_payload"]
