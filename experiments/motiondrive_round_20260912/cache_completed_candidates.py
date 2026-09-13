"""Freeze M8_GEN and extract its candidates once, for both selector arms.

This is an in-sample train prediction cache: the generator was trained on the
same train rows, so it is not out-of-fold. Tune is never used for gradient in
either the generator or the selectors.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "tune"), required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists():
        raise SystemExit(f"output is immutable: {out}")
    D = ROOT / "experiments/motiondrive_round_20260912"
    sys.path.insert(0, str(D)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    import run_round_m8 as rs
    from motiondrive_v2_data import MotionDriveDataset
    from motiondrive_v2_training import model_inputs

    run = Path(a.run_dir).resolve()
    payload = torch.load(run / "last.pth", map_location="cpu", weights_only=False)
    manifest = json.loads((run / "manifest.json").read_text())
    rs._ARM["name"] = "m8_gen"
    from models.motiondrive_v2 import MotionDriveV2Config
    config = MotionDriveV2Config(**manifest["model_config"])
    model = rs.install_multimode(rs.make_model(config), config, 8)
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("frozen generator did not load strictly")
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    dataset = MotionDriveDataset(
        data_root="/tmp/pm97",
        split_manifest=str(ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
        split=a.split, supervision_root=str(ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"),
        min_frame=30, frame_stride=1 if a.split == "train" else 5,
        max_samples=a.limit, augment=False, seed=0, history_contract="control")
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch, shuffle=False,
                                         num_workers=a.workers, pin_memory=True)
    store, started = {}, time.time()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            inputs = model_inputs(batch, time_input="nominal",
                                  nominal_history_seconds=config.nominal_history_seconds)
            inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(**inputs)
            last = model.planner.last
            piece = {
                "row": batch["row"].numpy().astype(np.int64),
                "candidate_xy": last["candidates"].float().cpu().numpy().astype(np.float32),
                "candidate_hidden": last["hidden"].float().cpu().numpy().astype(np.float32),
                "base_scores": last["scores"].float().cpu().numpy().astype(np.float32),
                "goal_xy": inputs["goal_xy"].float().cpu().numpy().astype(np.float32),
                "gt_plan": batch["gt_plan"].numpy().astype(np.float32),
                "session": np.asarray(batch["session_id"]),
                "scenario": np.asarray(batch["scenario"]),
            }
            for k, v in piece.items():
                store.setdefault(k, []).append(v)
            if (index + 1) % 200 == 0:
                print(json.dumps({"batches": index + 1,
                                  "rows": int(sum(len(x) for x in store["row"]))}), flush=True)
    after = model.state_dict()
    if any(not torch.equal(before[k], after[k]) for k in before):
        raise RuntimeError("the frozen generator changed during extraction")
    arrays = {k: np.concatenate(v, 0) for k, v in store.items()}
    arrays["candidate_valid"] = np.isfinite(arrays["candidate_xy"]).all((-1, -2))

    out.mkdir(parents=True)
    files = {}
    for k, v in arrays.items():
        path = out / f"{k}.npy"
        np.save(path, v, allow_pickle=False)
        files[k] = {"path": path.name, "shape": list(v.shape), "dtype": str(v.dtype),
                    "sha256": sha(path)}
    (out / "manifest.json").write_text(json.dumps({
        "schema": "m8_candidate_cache_v1", "status": "completed", "split": a.split,
        "rows": int(len(arrays["row"])), "modes": int(arrays["candidate_xy"].shape[1]),
        "generator_run": str(run), "generator_sha256": sha(run / "last.pth"),
        "in_sample_note": ("train rows were seen by the generator; this is an in-sample "
                           "prediction cache, not out-of-fold"),
        "goal_note": "one provided goal; scene_goal already consumed upstream, selection_goal stored here",
        "gt_note": "stored separately, for loss and evaluation only",
        "batch_size": a.batch, "precision": "bf16 autocast", "files": files,
        "elapsed_seconds": time.time() - started}, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "output": str(out), "rows": len(arrays["row"]),
                      "manifest_sha256": sha(out / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
