#!/usr/bin/env python3
"""Collect what a reviewer needs to re-run the candidate: source, weights, env."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_grouped_split_v2 import sha256

REPORTS = ROOT / "reports/md_r0_reset_20260914"
OUT = REPORTS / "reproduction_manifest.json"
PIP_FREEZE = REPORTS / "environment_pip_freeze.txt"
PYTHON = ROOT / "env/venv/bin/python"

INFERENCE_SOURCES = [
    "models/motiondrive_v2/__init__.py",
    "models/motiondrive_v2/config.py",
    "models/motiondrive_v2/model.py",
    "models/motiondrive_v2/scene_encoder.py",
    "models/motiondrive_v2/motion_encoder.py",
    "models/motiondrive_v2/planner.py",
    "models/motiondrive_v2/cross_cell_goal_residual.py",
    "models/motiondrive_v2_inputs.py",
    "models/motiondrive_v2_input_contract.py",
    "models/motiondrive_v2_temporal_contract.py",
    "experiments/md_r0_reset_20260914/submit_official_test.py",
    "experiments/md_r0_reset_20260914/build_submission.py",
]
TRAINING_SOURCES = [
    "scripts/train_motiondrive_v2.py",
    "scripts/motiondrive_v2_training.py",
    "scripts/motiondrive_v2_data.py",
    "scripts/motiondrive_v2_flip_augment.py",
    "scripts/build_grouped_split_v2.py",
    "scripts/build_scene_supervision_v2.py",
    "experiments/md_r0_reset_20260914/train.py",
    "experiments/md_r0_reset_20260914/evaluate.py",
    "experiments/md_r0_reset_20260914/contract_tests.py",
    "experiments/md_r0_reset_20260914/p3_split.py",
    "experiments/md_r0_reset_20260914/p3_supervision.py",
    "experiments/md_r0_reset_20260914/p3_supervision_finish.py",
    "experiments/md_r0_reset_20260914/p3_rebind_reports.py",
    "experiments/md_r0_reset_20260914/p3_rebind_init.py",
]
PRETRAINED = "ckpt/backbones/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth"

DOCKERFILE = """# Draft image for reproducing the MotionDrive V2 candidate.
# NOT built or tested here; it records the environment the run actually used.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update \\
 && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3-pip \\
      libgl1 libglib2.0-0 \\
 && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch 2.7.1+cu128 is the build the reported numbers were produced with.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \\
      torch==2.7.1 torchvision
COPY environment_pip_freeze.txt /tmp/environment_requirements.txt
RUN pip install --no-cache-dir -r /tmp/environment_requirements.txt

WORKDIR /workspace
# Mount the repository, the checkpoint and the official test tars, then:
#   python experiments/md_r0_reset_20260914/submit_official_test.py \\
#     --init <checkpoint> --test-root <test tars> --out submission.json \\
#     --label review --gpu 0
"""


def main() -> None:
    candidate = json.loads((REPORTS / "candidate_registry.json").read_text())
    # The venv is uv-created and has no pip; read the installed distributions
    # straight from the environment's own metadata instead.
    freeze = subprocess.run(
        [str(PYTHON), "-c",
         "import importlib.metadata as m;"
         "print(chr(10).join(sorted(f'{d.metadata[\"Name\"]}=={d.version}'"
         " for d in m.distributions() if d.metadata['Name'])))"],
        capture_output=True, text=True, check=True).stdout
    PIP_FREEZE.write_text(freeze)
    (ROOT / "experiments/md_r0_reset_20260914/Dockerfile").write_text(DOCKERFILE)

    def sha_map(paths):
        out = {}
        for rel in paths:
            path = ROOT / rel
            out[rel] = sha256(path) if path.exists() else None
        return out

    torch_version = subprocess.run(
        [str(PYTHON), "-c", "import torch,numpy;print(torch.__version__);print(numpy.__version__);"
         "print(torch.version.cuda)"],
        capture_output=True, text=True, check=True).stdout.split()

    manifest = {
        "schema_version": 1,
        "candidate": candidate["candidate"],
        "checkpoint": {"path": candidate["checkpoint_path"],
                       "sha256": candidate["checkpoint_sha256"],
                       "model_state_sha256": candidate["model_state_sha256"],
                       "contains": "model tensors and the run manifest; optimizer state kept separately",
                       "optimizer_checkpoint": candidate["training"]["optimizer_checkpoint_preserved"]},
        "model_config": candidate["model_config"],
        "training_recipe": candidate["training"]["recipe"],
        "data_provenance": {
            "split_manifest": candidate["training"]["split_manifest"],
            "supervision": candidate["training"]["supervision"],
            "train_data": candidate["training"]["train_data"],
            "tune_data": candidate["training"]["tune_data"],
            "initializer": candidate["training"]["initializer"],
        },
        "pretrained_origin": {
            "file": PRETRAINED,
            "sha256": sha256(ROOT / PRETRAINED) if (ROOT / PRETRAINED).exists() else None,
            "description": ("public mmdetection3d nuImages weight "
                            "cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim; the only external "
                            "initialization in the lineage, applied before any ETRI training"),
            "never_trained_on_tune_or_val": True,
        },
        "sources": {"inference": sha_map(INFERENCE_SOURCES),
                    "training": sha_map(TRAINING_SOURCES)},
        "runtime_source_git_sha": candidate["training"]["runtime_source_git_sha"],
        "environment": {
            "python": platform.python_version(),
            "venv_python": subprocess.run([str(PYTHON), "-V"], capture_output=True,
                                          text=True, check=True).stdout.strip(),
            "torch": torch_version[0], "numpy": torch_version[1], "cuda": torch_version[2],
            "installed_distributions": str(PIP_FREEZE),
            "training_device": "NVIDIA B200",
        },
        "docker": {
            "dockerfile": "experiments/md_r0_reset_20260914/Dockerfile",
            "status": "DRAFT_NOT_BUILT",
            "note": "records the environment actually used; it has not been built or tested here",
        },
        "how_to_reproduce_inference": [
            "mount the repository, the checkpoint and the official test tars",
            ("python experiments/md_r0_reset_20260914/submit_official_test.py --init <ckpt> "
             "--test-root <tars> --out submission.json --label review --gpu 0"),
            "the writer reads only calibration.parquet, ego_pose.parquet and ten JPEGs per clip",
        ],
        "pending": candidate["pending"],
    }
    OUT.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    missing = [k for group in manifest["sources"].values() for k, v in group.items() if v is None]
    print(json.dumps({"written": str(OUT), "installed_distributions": str(PIP_FREEZE),
                      "inference_sources": len(INFERENCE_SOURCES),
                      "training_sources": len(TRAINING_SOURCES),
                      "missing_sources": missing,
                      "pretrained_sha256": manifest["pretrained_origin"]["sha256"],
                      "torch": manifest["environment"]["torch"]}, indent=1))


if __name__ == "__main__":
    main()
