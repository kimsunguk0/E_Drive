"""Train only a candidate-relative scoring head on a frozen image-model cache.

The cache contains separate input features and D3 labels. All output coordinates
remain existing candidate rows. Two predeclared objectives can be compared on
exactly the same image-model shortlist: soft CE, and centered cost regression.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
from relative_selector import RelativeScoreHead

TUNE_SHA = "1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88"
TIME_WEIGHTS = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scoring_loss(scores, costs, valid, objective, temperature):
    """Supervision-only entry. Its labels are never passed to a scoring head."""
    valid = valid.bool()
    count = valid.sum(-1, keepdim=True)
    if not count.min() or not torch.isfinite(scores).all() or not torch.isfinite(costs).all():
        raise ValueError("Finite scores/costs and at least one valid candidate required")
    if objective == "soft_ce":
        target = (-costs / temperature).masked_fill(~valid, -torch.inf).softmax(-1)
        log_prob = scores.masked_fill(~valid, -torch.inf).log_softmax(-1).masked_fill(~valid, 0)
        return -(target * log_prob).sum(-1).mean()
    if objective != "centered_d3":
        raise ValueError(objective)
    # Subtracting a common per-row cost cannot change which candidate is best.
    # Squared regression estimates conditional mean cost, up to that common term.
    pred = scores - (scores * valid).sum(-1, keepdim=True) / count
    target = -(costs - (costs * valid).sum(-1, keepdim=True) / count) / temperature
    return (((pred - target).square() * valid).sum(-1) / count.squeeze(-1)).mean()


def read_cache(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed" or manifest["tune_rows_sha256"] != TUNE_SHA:
        raise ValueError("A completed cache of the exact original tune population is required")
    feature_path = Path(__file__).with_name("relative_selector.py")
    if file_sha(feature_path) != manifest["feature_source_sha256"]:
        raise ValueError("Candidate feature implementation differs from the cache")
    keys = ("features", "base_logits", "costs", "valid", "candidate_ids", "candidate_xy", "rows", "gt_xy")
    required = {f"{split}/{key}.npy" for split in ("train", "tune") for key in keys}
    if not required <= set(manifest["artifacts"]):
        raise ValueError("Cache manifest does not cover every consumed artifact")
    for relative, record in manifest["artifacts"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or file_sha(path) != record["sha256"]:
            raise ValueError(f"Cached artifact changed: {relative}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Cached shape/dtype changed: {relative}")
    for kind in ("checkpoint", "bank"):
        if file_sha(manifest[kind]["path"]) != manifest[kind]["sha256"]:
            raise ValueError(f"Frozen {kind} changed after feature extraction")
    with np.load(manifest["bank"]["path"], allow_pickle=False) as z:
        bank_xy = z["traj_xyz8"][..., :6, :2].reshape(-1, 6, 2)
        bank_valid = z["mask8"][..., :6].astype(bool).all(-1).reshape(-1)
    result = {}
    for split in ("train", "tune"):
        cpu = {key: np.load(directory / split / f"{key}.npy", mmap_mode="r", allow_pickle=False)
               for key in keys}
        rows = cpu["rows"]
        digest = hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()
        expected = manifest["sampled_train_rows_sha256"] if split == "train" else TUNE_SHA
        if digest != expected or len(np.unique(rows)) != len(rows):
            raise ValueError("Cached row identity mismatch")
        if cpu["features"].shape != (*cpu["costs"].shape, 32):
            raise ValueError("Expected exactly 32 candidate-relative input features")
        shape = cpu["costs"].shape
        if (len(shape) != 2 or shape[0] != len(rows) or cpu["base_logits"].shape != shape
                or cpu["valid"].shape != shape or cpu["valid"].dtype != np.bool_
                or cpu["candidate_ids"].shape != shape or cpu["candidate_ids"].dtype != np.int64
                or cpu["candidate_xy"].shape != (*shape, 6, 2) or cpu["gt_xy"].shape != (len(rows), 6, 2)
                or rows.dtype != np.int64 or not cpu["valid"].any(-1).all()):
            raise ValueError("Cached candidate/label shapes or types are malformed")
        for key in ("features", "base_logits", "costs", "candidate_xy", "gt_xy"):
            if cpu[key].dtype != np.float32 or not np.isfinite(cpu[key]).all():
                raise ValueError(f"Expected finite float32 cached {key}")
        if (cpu["costs"] < 0).any():
            raise ValueError("D3 labels cannot be negative")
        for first in range(0, len(rows), 256):
            sl = slice(first, first+256)
            ids = cpu["candidate_ids"][sl]
            if (ids < 0).any() or (ids >= len(bank_xy)).any():
                raise ValueError("Cached candidate ID is outside the immutable bank")
            if (not np.array_equal(cpu["candidate_xy"][sl], bank_xy[ids])
                    or not np.array_equal(cpu["valid"][sl], bank_valid[ids])):
                raise ValueError("Cached coordinates/validity differ from the immutable bank")
            point = np.linalg.norm(cpu["candidate_xy"][sl].astype(np.float64)
                                   - cpu["gt_xy"][sl, None].astype(np.float64), axis=-1)
            if np.max(np.abs((point*TIME_WEIGHTS).sum(-1) - cpu["costs"][sl])) > 1e-5:
                raise ValueError("Cached training/evaluation costs are not the stated D3 labels")
        tensors = {key: torch.from_numpy(np.array(cpu[key], copy=True)).cuda()
                   for key in ("features", "base_logits", "costs", "valid")}
        result[split] = {"cpu": cpu, "gpu": tensors}
    if np.intersect1d(result["train"]["cpu"]["rows"], result["tune"]["cpu"]["rows"]).size:
        raise ValueError("Training and tune rows overlap")
    return manifest, result


@torch.inference_mode()
def evaluate(head, dataset, directory, step):
    head.eval()
    g, c = dataset["gpu"], dataset["cpu"]
    selections, costs = [], []
    for start in range(0, len(c["rows"]), 128):
        sl = slice(start, start + 128)
        scores = g["base_logits"][sl] + head(g["features"][sl])
        scores = scores.masked_fill(~g["valid"][sl].bool(), -torch.inf)
        index = scores.argmax(-1)
        if not g["valid"][sl].gather(1, index[:, None]).all():
            raise ValueError("Head selected an invalid candidate")
        selections.append(index.cpu().numpy())
        costs.append(g["costs"][sl].gather(1, index[:, None]).squeeze(1).cpu().numpy())
    index, selected_cost = np.concatenate(selections), np.concatenate(costs)
    row = np.arange(len(index))
    prediction = np.asarray(c["candidate_xy"])[row, index]
    delta = prediction.astype(np.float64) - c["gt_xy"].astype(np.float64)
    point = np.linalg.norm(delta, axis=-1)
    independent = (point * TIME_WEIGHTS).sum(-1)
    if np.max(np.abs(independent - selected_cost)) > 1e-5:
        raise ValueError("Independent D3 differs from cached candidate labels")
    valid_cost = np.where(c["valid"], c["costs"], np.inf)
    oracle = valid_cost.min(-1)
    result = {"step": step, "n": len(index), "official_d3": float(independent.mean()),
              "shortlist_oracle_d3": float(oracle.mean()),
              "selection_regret": float((independent - oracle).mean()),
              "point_l2_metres_05_to_30": point.mean(0).tolist(),
              "three_second_endpoint_l2": float(point[:, -1].mean()),
              "metric_recompute_max_abs_error": float(np.abs(independent-selected_cost).max())}
    np.savez_compressed(directory / f"eval_{step:06d}.npz", rows=c["rows"], pred=prediction,
                        d3=independent, point_l2=point, error_xy=delta,
                        candidate_id=np.asarray(c["candidate_ids"])[row, index], shortlist_oracle=oracle)
    write_json(directory / f"eval_{step:06d}.json", result)
    print(json.dumps({"evaluation": result}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--cache", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--objective", choices=("soft_ce", "centered_d3"), required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=.1)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "4"):
        raise ValueError("Expose exactly one allocated physical GPU")
    if min(a.steps, a.batch, a.eval_every, a.temperature) <= 0:
        raise ValueError("Positive training parameters required")
    directory = Path(a.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed); torch.set_num_threads(4)
        cache = Path(a.cache).resolve()
        cache_manifest, datasets = read_cache(cache)
        head = RelativeScoreHead().cuda().float()
        optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sources = [Path(__file__), Path(__file__).with_name("relative_selector.py")]
        (directory / "source").mkdir()
        for path in sources:
            shutil.copy2(path, directory / "source" / path.name)
        manifest = {"arguments": vars(a), "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]),
                    "cache_path": str(cache), "cache_manifest_sha256": file_sha(cache / "manifest.json"),
                    "cache_manifest": cache_manifest, "torch": str(torch.__version__),
                    "source_sha256": {path.name: file_sha(path) for path in sources},
                    "model": "frozen image model shortlist plus candidate-relative score residual",
                    "precision": "float32 head, frozen bf16 base cache", "coordinates": "unchanged cache bank rows",
                    "primary_comparison": "fixed terminal step; tune best reported separately"}
        write_json(directory / "manifest.json", manifest)
        initial = evaluate(head, datasets["tune"], directory, 0)
        train = datasets["train"]["gpu"]
        n = len(train["features"])
        order, offset, epoch = torch.randperm(n, device="cuda"), 0, 0
        history, best = [], float("inf")
        for step in range(1, a.steps + 1):
            if offset >= n:
                order, offset, epoch = torch.randperm(n, device="cuda"), 0, epoch + 1
            ids = order[offset:offset + a.batch]; offset += a.batch
            factor = min(step / max(a.warmup, 1), 1.)
            if step > a.warmup:
                factor *= .5 * (1 + math.cos(math.pi * (step-a.warmup) / max(a.steps-a.warmup, 1)))
            optimizer.param_groups[0]["lr"] = a.lr * factor
            head.train(); optimizer.zero_grad(set_to_none=True)
            score = train["base_logits"][ids] + head(train["features"][ids])
            loss = scoring_loss(score, train["costs"][ids], train["valid"][ids], a.objective, a.temperature)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 50 == 0:
                record = {"step": step, "epoch": epoch, "loss": float(loss.detach()),
                          "grad_norm": float(norm), "seconds": time.time()-started}
                with (directory / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                result = evaluate(head, datasets["tune"], directory, step)
                history.append(result)
                payload = {"relative_head": head.state_dict(), "optimizer": optimizer.state_dict(),
                           "manifest": manifest, "step": step, "epoch": epoch, "result": result}
                for name in ["last.pth"] + (["best.pth"] if result["official_d3"] < best else []):
                    temporary = directory / (name + ".tmp")
                    torch.save(payload, temporary); temporary.replace(directory / name)
                best = min(best, result["official_d3"])
                write_json(directory / "progress.json", {"status": "running", "step": step,
                    "terminal_so_far": result, "best_d3": best, "elapsed_seconds": time.time()-started})
        write_json(directory / "result.json", {"status": "completed", "initial": initial,
            "terminal": history[-1], "best_d3": best, "history": history,
            "elapsed_seconds": time.time()-started, "peak_cuda_bytes": torch.cuda.max_memory_allocated()})
    except BaseException:
        write_json(directory / "failure.json", {"status": "failed", "traceback": traceback.format_exc(),
                                               "elapsed_seconds": time.time()-started})
        raise


if __name__ == "__main__":
    main()
