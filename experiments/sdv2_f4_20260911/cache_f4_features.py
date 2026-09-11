"""Export frozen-B8k intermediates for the F4 velocity-coverage scorer.

Only image-derived tensors and fixed bank geometry are written. The provided
goal is carried alongside for the terminal score only; GT waypoints go to a
separate array the scorer never reads. Dataset construction is reused from the
existing candidate cache so the population and contract checks are identical.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch

BINS = (8, 16)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--worktree", required=True)
    p.add_argument("--cache-module", required=True)
    p.add_argument("--evaluator-source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("train", "tune"), required=True)
    p.add_argument("--base", default="/NHNHOME/data/sukim/adcl")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    out = Path(a.output).resolve()
    if out.exists():
        raise SystemExit(f"output is immutable: {out}")
    sys.path.insert(0, str(Path(a.worktree).resolve()))
    cc = module_from(a.cache_module, "_f4_cache_helpers")
    ev = module_from(a.evaluator_source, "_f4_eval")

    ev.check_gpu(a.gpu)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    plan = ev.inspect_checkpoint(a.checkpoint, worktree=a.worktree, base=a.base)
    arguments = plan.manifest["arguments"]
    if arguments.get("common_status"):
        raise SystemExit("F4 fixes on the status-free B arm")

    started = time.time()
    with ev.isolated_runtime(plan.source) as runtime:
        model = ev.strict_load(plan, runtime).cuda().eval()
        # Capture intermediates with runtime hooks: editing the pinned sources
        # would break the checkpoint's recorded source_sha256.
        fusion_class = runtime.temporal_model.ImageTemporalFusion
        fusions = [m for m in model.modules() if isinstance(m, fusion_class)]
        if len(fusions) != 1:
            raise RuntimeError(f"expected one fusion module, saw {len(fusions)}")
        fusion = fusions[0]
        grabbed = {}

        def take_current(_m, args):
            grabbed["current"] = args[0]
            grabbed["attention_calls"] = 0

        def take_unconditioned(_m, _args, output):
            # The module runs attention twice: unconditioned first, then conditioned.
            grabbed["attention_calls"] = grabbed.get("attention_calls", 0) + 1
            if grabbed["attention_calls"] == 1:
                grabbed["unconditioned"] = output[0]

        # The surviving path queries are only reachable after the loop's local
        # top-k, so take the outer-sum input to the trajectory deformable step:
        # traj[i, j] = path[i] + velocity[j], hence traj[:, :, 0] is the path
        # token up to one per-row constant, which no relative score can see.
        def take_traj(_m, args):
            grabbed["traj"] = args[0]

        handles = [fusion.register_forward_pre_hook(take_current),
                   fusion.attention.register_forward_hook(take_unconditioned)]
        layers = model.base._trajectory_head.decoder.layers
        handles.append(layers[-1].t_deform_model.register_forward_pre_hook(take_traj))
        attends = []

        def bins_of():
            current, unconditioned = grabbed["current"], grabbed["unconditioned"]
            b, c, h, w = current.shape
            now = current.flatten(2).transpose(1, 2)
            readout = (now.float() + unconditioned.float()).transpose(1, 2).reshape(b, c, h, w)
            return torch.nn.functional.adaptive_avg_pool2d(readout, BINS)

        dataset, provenance, identities = cc.build_population(plan, runtime, ev, a.split, a.limit)
        loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch, shuffle=False,
                                             num_workers=a.workers, pin_memory=True)
        store = {}
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                inputs = ev.input_tensors(batch, runtime, False, "cuda:0")
                if "gt_plan" in inputs or "status" in inputs or "perception_status" in inputs:
                    raise RuntimeError("inference inputs must carry no label or provided state")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**inputs)
                if grabbed.get("attention_calls") != 2 or "traj" not in grabbed:
                    raise RuntimeError("expected one unconditioned pass and the outer sum")
                exported = bins_of()
                b = exported.shape[0]
                paths, velocities = result["path_ids"].shape[1], result["velocity_ids"].shape[1]
                traj = grabbed["traj"]
                if traj.shape[:2] != (b, paths * velocities):
                    raise RuntimeError("captured outer sum does not match the retained grid")
                path_tokens = traj.reshape(b, paths, velocities, -1)[:, :, 0]
                piece = {
                    "rows": batch["row"].numpy().astype(np.int64),
                    "path_ids": result["path_ids"].cpu().numpy().astype(np.int32),
                    "path_tokens": path_tokens.float().cpu().numpy().astype(np.float16),
                    "temporal_bins": exported.reshape(b, exported.shape[1], -1)
                                     .permute(0, 2, 1).cpu().numpy().astype(np.float16),
                    "stage1_velocity_logits": result["coarse"][0]["velocity_scores"]
                                     .float().cpu().numpy().astype(np.float32),
                    "goal_xy": inputs["goal_xy"].float().cpu().numpy().astype(np.float32),
                    "gt": batch["gt_plan"].numpy().astype(np.float32),
                }
                for k, v in piece.items():
                    store.setdefault(k, []).append(v)
                if (i + 1) % 200 == 0:
                    print(json.dumps({"batches": i + 1,
                                      "rows": int(sum(len(x) for x in store["rows"]))}), flush=True)
        arrays = {k: np.concatenate(v, 0) for k, v in store.items()}

    for handle in handles:
        handle.remove()
    for layer, original_attend in attends:
        layer.attend = original_attend
    if not np.array_equal(arrays["rows"], identities["rows"]):
        raise RuntimeError("row order drifted during export")
    arrays["session"] = identities["session"].astype("U")
    arrays["scene"] = identities["scene"].astype("U")

    out.mkdir(parents=True)
    files = {}
    for k, v in arrays.items():
        path = out / f"{k}.npy"
        np.save(path, v, allow_pickle=False)
        files[k] = {"path": path.name, "shape": list(v.shape), "dtype": str(v.dtype),
                    "sha256": sha(path)}
    manifest = {
        "schema": "sdv2_f4_feature_cache_v1", "status": "completed", "split": a.split,
        "rows": int(len(arrays["rows"])),
        "rows_sha256": hashlib.sha256(arrays["rows"].astype("<i8").tobytes()).hexdigest(),
        "checkpoint": str(Path(a.checkpoint).resolve()),
        "checkpoint_sha256": plan.receipt["checkpoint_sha256"],
        "bank_sha256": plan.receipt.get("bank_sha256"),
        "spatial_bins": list(BINS), "batch_size": a.batch, "precision": "bf16 autocast",
        "provided_status_in_graph": False,
        "goal_route": "carried for terminal scoring only; not an encoder input",
        "gt_route": "written separately; never a scorer feature",
        "dataset_provenance": provenance, "files": files,
        "elapsed_seconds": time.time() - started, "torch": str(torch.__version__),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True,
                                                  default=str) + "\n")
    print(json.dumps({"status": "completed", "output": str(out),
                      "manifest_sha256": sha(out / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
