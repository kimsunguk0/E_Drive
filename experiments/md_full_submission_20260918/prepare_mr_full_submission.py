#!/usr/bin/env python3
"""Prepare the completed status-free MR FULL candidate without training or uploading.

Uses the same raw-input builder and packaging path as the scored MR-NATIVE-s1.
Every output is written to a new candidate directory; previous submissions stay
intact. Run from the B200 repository using the existing cv2-enabled environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/md_r0_reset_20260914"
FULL = ROOT / "work_dirs/md_progress_residual_20260917/MR-NATIVE-FULL-s1"
LABEL = "MR-NATIVE-FULL-s1"
EXPECTED_STEP = 24931
EXPECTED_CLIPS = 1125


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gpu", type=int, choices=range(4), default=0)
    parser.add_argument("--clips-root", type=Path, default=Path("/tmp/etri_test"))
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "reports/md_full_submission_20260918")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite candidate records: {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    import torch
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(EXP)]
    import motiondrive_v2_training as mt

    source_checkpoint = FULL / f"ckpt_step{EXPECTED_STEP}.pth"
    run_manifest = json.loads((FULL / "manifest.json").read_text())
    experiment = json.loads((FULL / "experiment.json").read_text())
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_manifest = payload["manifest"]
    assert run_manifest["status"] == "completed"
    assert run_manifest["step"] == EXPECTED_STEP
    assert payload["step"] == EXPECTED_STEP
    assert experiment["provided_status_used"] is False
    assert experiment["full_fit"]["rows"] == 101520
    assert experiment["full_fit"]["unique_scenes"] == 376
    assert experiment["full_fit"]["checkpoint_selection_from_evaluation"] is False
    config = json.loads(json.dumps(checkpoint_manifest["model_config"]))
    assert config == run_manifest["model_config"]
    assert config["correlation_radius"] == 4 and config["correlation_channels"] == 32
    assert config["cross_cell_goal_mode"] == "zero"
    assert config["history_frame_offsets"] == [1, 2, 5, 10]
    forbidden_parameters = [name for name in payload["model"]
                            if name.startswith(("shared_status", "progress_head"))]
    assert not forbidden_parameters, forbidden_parameters

    directories = sorted(p.name for p in args.clips_root.iterdir() if p.is_dir())
    tar_tokens = sorted(p.stem for p in (ROOT / "test").glob("*.tar"))
    assert len(directories) == EXPECTED_CLIPS and directories == tar_tokens
    source_sha = sha256(source_checkpoint)
    model_sha = mt.tensor_state_sha256(payload["model"])
    preserved = ROOT / "work_dirs/md_full_submission_20260918/preserved/ckpt_step24931.pth"
    preserved.parent.mkdir(parents=True, exist_ok=True)
    if preserved.exists():
        assert sha256(preserved) == source_sha
    else:
        shutil.copy2(source_checkpoint, preserved)
        preserved.chmod(0o444)
    assert sha256(preserved) == source_sha

    preflight = {
        "candidate": LABEL, "checkpoint": str(preserved),
        "source_checkpoint": str(source_checkpoint), "checkpoint_sha256": source_sha,
        "model_state_sha256": model_sha, "step": EXPECTED_STEP,
        "checkpoint_selection": "predeclared terminal, no in-fit metric selection",
        "train_scenes": 376, "train_rows": 101520,
        "provided_status_used": False, "a2_a3_or_progress_parameters": forbidden_parameters,
        "model_config": config, "expected_clips": EXPECTED_CLIPS,
        "test_tar_tokens_match_extracted_directories": True,
        "state_on_meaning": "use image-predicted state/history, not provided numeric state",
        "pose_use": "scene alignment at frames -1,-2,-5,-10 relative to current",
        "goal_use": "+50 target position conditions shared scene features",
        "motion_inputs": "unwarped current/past image features and fixed nominal intervals",
        "heldout_accuracy": None,
        "in_fit_diagnostic_not_generalization": run_manifest["best_metric"],
        "no_new_training": True, "no_upload": True,
    }
    write_json(out / "preflight.json", preflight)
    del payload
    shutil.copy2(FULL / "experiment.json", out / "training_experiment.json")
    shutil.copy2(FULL / "manifest.json", out / "training_manifest.json")

    tracked = subprocess.check_output(["git", "ls-files", "models", "scripts", str(EXP.relative_to(ROOT)),
                                       "experiments/md_full_submission_20260918"],
                                      cwd=ROOT, text=True).splitlines()
    # Fingerprint executable source broadly; only a subset is used for inference.
    sources = {name: sha256(ROOT / name) for name in tracked
               if name.endswith(".py") and (ROOT / name).is_file()}
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(Path(__file__))
    write_json(out / "source_manifest.json", {
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Python source fingerprints; this is not a claim all files are executed",
        "files": sources,
    })
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "scipy", "Pillow", "pyarrow")}
    import cv2
    versions.update(opencv=cv2.__version__, python=sys.version, cuda=torch.version.cuda)
    write_json(out / "environment.json", versions)

    stages = []
    def run(stage, argv):
        started = time.monotonic()
        record = {"stage": stage,
                  "argv": [arg.replace(str(Path.home()), "~") for arg in argv],
                  "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        print(json.dumps({"stage": stage, "event": "start"}), flush=True)
        with (out / f"{stage}.log").open("w") as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=os.environ.copy(), text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        record.update(returncode=code, elapsed_seconds=time.monotonic() - started)
        stages.append(record)
        write_json(out / "execution.json", stages)
        if code:
            raise SystemExit(f"{stage} failed: {code}; see {out / (stage + '.log')}")

    # Redirect the existing checker's output instead of replacing prior MR records.
    parity_code = (f"import sys;sys.path.insert(0,{str(EXP)!r});"
                   "from pathlib import Path;import mr_raw_b1 as m;"
                   f"m.OUT=Path({str(out / 'raw_b1_fixture8.json')!r});m.main()")
    run("raw_b1_parity", [sys.executable, "-c", parity_code, "--init", str(preserved),
                          "--gpu", "0", "--precision", "bf16", "--detail", "native",
                          "--label", LABEL])
    parity = json.loads((out / "raw_b1_fixture8.json").read_text())
    assert parity["inputs_all_bitwise_equal"] and parity["max_plan_abs_xy_diff_m"] == 0

    run("flops", [sys.executable, str(EXP / "measure_flops.py"), "--init", str(preserved),
                  "--clip", str(args.clips_root / directories[0]), "--mr-detail", "native",
                  "--out", str(out / "flops.json")])
    flops = json.loads((out / "flops.json").read_text())
    assert flops["passes_cutoff"] and flops["model_state_sha256"] == model_sha

    # Confirm the retained production path still reproduces a scored prediction.
    old = ROOT / "reports/md_r0_reset_20260914/submission/official_test_MR-NATIVE-s1.validation.json"
    old_meta = json.loads(old.read_text())
    run("previous_submission_replay", [sys.executable, str(EXP / "build_submission.py"),
        "--init", old_meta["checkpoint"], "--clips-root", str(args.clips_root),
        "--out", str(out / "previous_submission_replay.json"), "--label", "MR-NATIVE-s1-replay",
        "--gpu", "0", "--precision", "bf16", "--mr-detail", "native", "--limit", "1"])
    old_saved = json.loads(Path(old_meta["submission_file"]).read_text())
    replay = json.loads((out / "previous_submission_replay.json").read_text())
    assert replay[directories[0]] == old_saved[directories[0]], "Scored MR path no longer matches"

    # Add progress reporting and reject unexpected input keys on every real clip.
    build_code = f"""
import sys,json,time
sys.path.insert(0,{str(EXP)!r})
import build_submission as b
import mr_deploy
original_prepare=mr_deploy.prepare_mr_clip_inputs
def checked_prepare(*args,**kwargs):
    result=original_prepare(*args,**kwargs)
    assert set(result.inputs)=={{'images','history_images','lidar2img','history_transforms','time_offsets','goal_xy','motion_current','motion_history'}}
    assert result.metadata['input_contract']['provided_status_or_timestamp_input'] is False
    return result
mr_deploy.prepare_mr_clip_inputs=checked_prepare
original_predict=b.predict
count=0
started=time.monotonic()
def progress(*args,**kwargs):
    global count
    result=original_predict(*args,**kwargs)
    count+=1
    if count==1 or count%100==0 or count>=1125:
        print(json.dumps({{'clips_processed':count,'elapsed_seconds':round(time.monotonic()-started,1)}}),flush=True)
    return result
b.predict=progress
b.main()
"""
    prediction = out / "official_test_MR-NATIVE-FULL-s1.json"
    run("official_test_inference", [sys.executable, "-c", build_code, "--init", str(preserved),
        "--clips-root", str(args.clips_root), "--out", str(prediction), "--label", LABEL,
        "--gpu", "0", "--precision", "bf16", "--mr-detail", "native"])
    validation = json.loads(prediction.with_suffix(".validation.json").read_text())
    assert validation["clips"] == EXPECTED_CLIPS
    assert validation["checks"]["clip_state_isolation_pass"]
    assert validation["checkpoint_sha256"] == source_sha
    assert validation["model_state_sha256"] == model_sha

    package = out / "submission"
    run("package", [sys.executable, str(EXP / "package_submission.py"), "--submission", str(prediction),
        "--flops-report", str(out / "flops.json"), "--clips-root", str(args.clips_root),
        "--out-dir", str(package), "--label", LABEL])
    with zipfile.ZipFile(package / "submission.zip") as archive:
        assert archive.namelist() == ["submission.json"] and archive.testzip() is None
        assert archive.read("submission.json") == (package / "submission.json").read_bytes()
    saved = json.loads((package / "submission.json").read_text())
    assert sorted(k for k in saved if k != "__flops__") == directories
    assert type(saved["__flops__"]) is int and saved["__flops__"] == flops["flops"]
    assert sha256(source_checkpoint) == source_sha == sha256(preserved)
    write_json(out / "completion.json", {
        "candidate": LABEL, "status": "ready_not_uploaded", "new_training_updates": 0,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint_sha256": source_sha, "model_state_sha256": model_sha,
        "preserved_checkpoint": str(preserved), "clips": EXPECTED_CLIPS,
        "provided_status_used": False, "raw_b1_fixtures_bitwise_equal": 8,
        "previous_scored_prediction_exact_replay": True,
        "clip_state_isolation_max_abs_diff_m": validation["checks"]["clip_state_isolation_max_abs_diff_m"],
        "flops": flops["flops"], "submission_zip": str(package / "submission.zip"),
        "submission_zip_sha256": sha256(package / "submission.zip"),
        "submission_json_sha256": sha256(package / "submission.json"),
        "uploaded": False, "submission_quota_consumed": 0,
        "heldout_or_server_score": None,
    })
    print((out / "completion.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
