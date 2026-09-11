"""Velocity-coverage scorer over completed bank rows, and its trainer.

The frozen B arm supplies image-conditioned path tokens, a spatially binned
image-only temporal readout, and stage-1 velocity logits defined for all 1,024
velocities. This head scores P x V completed bank rows and returns one of them
verbatim. No provided ego status enters anywhere; the provided goal enters only
the terminal per-candidate score.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

WEIGHTS = (11 / 36, 11 / 36, 5 / 36, 5 / 36, 2 / 36, 2 / 36)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def tensor_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        h.update(name.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


class CoverageScorer(nn.Module):
    """Score completed candidates from image features plus bank geometry.

    ``visual`` False zeroes every image-derived input while keeping the module,
    its parameter count and its initialisation identical, which is the matched
    control for what the extra visual features are worth.
    """

    def __init__(self, *, width=64, token_dim=256, visual=True):
        super().__init__()
        self.width, self.visual = width, bool(visual)
        self.path_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, width), nn.GELU())
        self.bin_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, width))
        # One velocity profile -> one query -> one read of the shared image bins.
        self.velocity_query = nn.Sequential(nn.Linear(12, width), nn.GELU(), nn.Linear(width, width))
        self.readout = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.velocity_embed = nn.Sequential(nn.Linear(12, width), nn.GELU(), nn.Linear(width, width))
        # 12 interval velocities, 10 interval accelerations, 1 stage-1 logit.
        self.geometry = nn.Sequential(nn.Linear(23, width), nn.GELU(), nn.Linear(width, width))
        self.score = nn.Sequential(nn.Linear(4 * width + 5, 128), nn.ReLU(),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, path_tokens, bins, velocity_profiles, candidate_xy, valid,
                stage1_logits, goal_xy):
        b, p, _ = path_tokens.shape
        v = velocity_profiles.shape[1]
        if candidate_xy.shape[:3] != (b, p, v):
            raise ValueError("candidate_xy must be [B,P,V,6,2] matching the grid")
        with torch.autocast(device_type=path_tokens.device.type, enabled=False):
            xy = torch.where(valid[..., None, None], candidate_xy.float(), 0.)
            step = torch.diff(xy, dim=3, prepend=torch.zeros_like(xy[:, :, :, :1])) * 2.
            accel = torch.diff(step, dim=3) * (2. / 3.)
            profiles = velocity_profiles.float()

            path = self.path_proj(path_tokens.float())
            query = self.velocity_query(profiles)
            keys = self.bin_proj(bins.float())
            if self.visual:
                read = self.readout(query, keys, keys, need_weights=False)[0]
            else:
                path = torch.zeros_like(path)
                read = torch.zeros(b, v, self.width, device=path.device)
            logits = stage1_logits.float() if self.visual else torch.zeros_like(stage1_logits)

            geometry_in = torch.cat([step.flatten(3), accel.flatten(3),
                                     logits[:, None, :, None].expand(b, p, v, 1)], -1)
            features = torch.cat([
                path[:, :, None].expand(b, p, v, self.width),
                read[:, None].expand(b, p, v, self.width),
                self.velocity_embed(profiles)[:, None].expand(b, p, v, self.width),
                self.geometry(geometry_in),
                goal_xy[:, None, None, :].expand(b, p, v, 2) / 50.,
                (xy[:, :, :, -1] - goal_xy[:, None, None, :]) / 50.,
                torch.ones(b, p, v, 1, device=path.device),
            ], -1)
            scores = self.score(features).squeeze(-1)
            return torch.where(valid, scores, torch.full_like(scores, -1e4))


class Cache:
    KEYS = ("rows", "path_ids", "path_tokens", "temporal_bins",
            "stage1_velocity_logits", "goal_xy", "gt", "session")

    def __init__(self, directory, device):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("status") != "completed":
            raise ValueError("only a completed cache may be consumed")
        self.arrays = {}
        for key in self.KEYS:
            path = self.directory / f"{key}.npy"
            recorded = self.manifest["files"][key]
            if recorded["path"] != path.name or sha(path) != recorded["sha256"]:
                raise ValueError("cache checksum mismatch: " + key)
            self.arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        self.n = len(self.arrays["rows"])
        self.device = device

    def field(self, key, index):
        return torch.from_numpy(np.ascontiguousarray(self.arrays[key][index])).to(self.device)


def d3_of(pred, gt, weights):
    return (torch.linalg.vector_norm(pred - gt[..., None, None, :, :], dim=-1) * weights).sum(-1)


def velocity_descriptor(bank):
    """Path-independent speed descriptor: cumulative arclength and interval speed.

    The bank factorises shape by speed, so the arclength profile of any path
    under velocity j describes j. Path 0 is the reference.
    """
    reference = bank[0]                                   # [V,6,2]
    step = torch.diff(reference, dim=1, prepend=torch.zeros_like(reference[:, :1]))
    distance = torch.linalg.vector_norm(step, dim=-1)     # [V,6]
    return torch.cat([distance.cumsum(-1) / 50., distance * 2. / 20.], -1)


def build_batch(cache, index, bank, mask, velocity_ids, descriptor, weights, stage1=None):
    path_ids = cache.field("path_ids", index).long()
    b, p = path_ids.shape
    logits = cache.field("stage1_velocity_logits", index)
    ids = velocity_ids
    if ids is None:                                       # per-row stage-1 top-k
        ids = torch.topk(logits, stage1, -1).indices      # [B,V]
    if ids.ndim == 1:
        ids = ids[None].expand(b, -1)
    v = ids.shape[1]
    xy = bank[path_ids[:, :, None], ids[:, None, :]]
    valid = mask[path_ids[:, :, None], ids[:, None, :]]
    gt = cache.field("gt", index).float()
    inputs = dict(
        path_tokens=cache.field("path_tokens", index).float(),
        bins=cache.field("temporal_bins", index).float(),
        velocity_profiles=descriptor[ids],
        candidate_xy=xy, valid=valid,
        stage1_logits=torch.gather(logits, 1, ids),
        goal_xy=cache.field("goal_xy", index).float())
    return inputs, d3_of(xy, gt, weights), valid


@torch.inference_mode()
def evaluate(head, cache, bank, mask, velocity_ids, descriptor, weights, directory,
             step, batch=8, prefix="eval", stage1=None):
    head.eval()
    chunks = {k: [] for k in ("d3", "oracle", "candidate_id", "point_l2")}
    for start in range(0, cache.n, batch):
        index = np.arange(start, min(start + batch, cache.n))
        inputs, cost, valid = build_batch(cache, index, bank, mask, velocity_ids,
                                          descriptor, weights, stage1)
        scores = head(**inputs)
        flat_cost = cost.flatten(1)
        pick = scores.flatten(1).argmax(1)
        rows = torch.arange(len(index), device=cache.device)
        chunks["d3"].append(flat_cost[rows, pick].cpu().numpy())
        chunks["oracle"].append(torch.where(valid.flatten(1), flat_cost,
                                            torch.full_like(flat_cost, 1e9)).min(1).values.cpu().numpy())
        chunks["candidate_id"].append(pick.cpu().numpy())
        xy = inputs["candidate_xy"].flatten(1, 2)[rows, pick]
        gt = cache.field("gt", index).float()
        chunks["point_l2"].append(torch.linalg.vector_norm(xy - gt, dim=-1).cpu().numpy())
    arrays = {k: np.concatenate(v, 0) for k, v in chunks.items()}
    report = {"step": step, "n": int(cache.n),
              "official_d3": float(arrays["d3"].mean()),
              "shortlist_oracle_d3": float(arrays["oracle"].mean()),
              "selection_regret": float((arrays["d3"] - arrays["oracle"]).mean()),
              "point_l2": arrays["point_l2"].mean(0).tolist()}
    atomic_json(Path(directory) / f"{prefix}_{step:06d}.json", report)
    np.savez_compressed(Path(directory) / f"{prefix}_{step:06d}.npz",
                        rows=np.asarray(cache.arrays["rows"]),
                        session=np.asarray(cache.arrays["session"]), **arrays)
    head.train()
    return report


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--tune-cache", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", required=True,
                   choices=("n64", "n64_novisual", "w1024", "w1024_novisual"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.1)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in tuple(str(i) for i in range(8)):
        raise RuntimeError("assign exactly one GPU")
    directory = Path(a.run_dir).resolve()
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    started = time.time()
    try:
        random.seed(a.seed); np.random.seed(a.seed)
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda")
        with np.load(a.bank, allow_pickle=False) as f:
            key = next(k for k in ("traj_xyz8", "traj_xy8", "traj_vocab") if k in f)
            raw = np.array(f[key])
            bank = torch.from_numpy(raw[..., :6, :2]).float().to(device)
            mask_key = "traj_mask" if "traj_mask" in f else None
            mask = (torch.from_numpy(np.array(f[mask_key])[..., :6]).bool().all(-1).to(device)
                    if mask_key else torch.ones(bank.shape[:2], dtype=torch.bool, device=device))
        weights = torch.tensor(WEIGHTS, device=device)
        train = Cache(a.train_cache, device)
        tune = Cache(a.tune_cache, device)
        if set(np.asarray(train.arrays["session"]).tolist()) & set(np.asarray(tune.arrays["session"]).tolist()):
            raise RuntimeError("train and tune sessions overlap")
        if train.manifest["checkpoint_sha256"] != tune.manifest["checkpoint_sha256"]:
            raise RuntimeError("caches come from different frozen bases")

        descriptor = velocity_descriptor(bank)
        # n64 reproduces the base's own stage-1 shortlist, which is per row.
        stage1 = 64 if a.arm.startswith("n64") else None
        velocity_ids = None if stage1 else torch.arange(bank.shape[1], device=device)

        head = CoverageScorer(visual=not a.arm.endswith("novisual")).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        source = Path(__file__).resolve()
        (directory / "source").mkdir(exist_ok=True)
        shutil.copy2(source, directory / "source" / source.name)
        manifest = dict(schema="sdv2_f4_coverage_head_v1", arguments=vars(a),
                        source_sha256={source.name: sha(source)},
                        bank_sha256=sha(a.bank),
                        frozen_base_sha256=train.manifest["checkpoint_sha256"],
                        candidates=(stage1 or int(bank.shape[1])) * int(train.arrays["path_ids"].shape[1]),
                        velocity_count=stage1 or int(bank.shape[1]),
                        velocity_source=("per-row stage-1 top-64" if stage1 else "all bank velocities"),
                        parameters=sum(x.numel() for x in head.parameters()),
                        initial_head_sha256=tensor_hash(head.state_dict()),
                        provided_status_used=False,
                        goal_route="terminal per-candidate score only",
                        physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]), pid=os.getpid())
        atomic_json(directory / "manifest.json", manifest)

        initial = evaluate(head, tune, bank, mask, velocity_ids, descriptor, weights,
                           directory, 0, a.eval_batch, stage1=stage1)
        print(json.dumps({"initial": initial}), flush=True)
        generator = torch.Generator().manual_seed(a.seed + 1)
        permutation, cursor, seen = torch.randperm(train.n, generator=generator), 0, 0
        accum = max(a.batch // a.microbatch, 1)
        log = open(directory / "train.jsonl", "w")
        for step in range(1, a.steps + 1):
            factor = min(step / max(a.warmup, 1), 1.)
            if step > a.warmup:
                factor *= .5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1)))
            for group in optimizer.param_groups:
                group["lr"] = a.lr * factor
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(accum):
                if cursor + a.microbatch > train.n:
                    permutation, cursor = torch.randperm(train.n, generator=generator), 0
                index = permutation[cursor:cursor + a.microbatch].numpy()
                cursor += a.microbatch; seen += len(index)
                inputs, cost, valid = build_batch(train, np.sort(index), bank, mask,
                                                  velocity_ids, descriptor, weights, stage1)
                scores = head(**inputs).flatten(1)
                # One softmax over every candidate in the row, never per chunk.
                target = torch.softmax(-cost.flatten(1) / a.temperature, -1)
                loss = -(target * torch.log_softmax(scores, -1)).sum(-1).mean() / accum
                loss.backward()
                total += float(loss)
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if step % 10 == 0:
                log.write(json.dumps({"step": step, "loss": total, "grad_norm": float(norm),
                                      "lr": a.lr * factor, "seen": seen}) + "\n")
                log.flush()
            if step % a.eval_every == 0 or step == a.steps:
                report = evaluate(head, tune, bank, mask, velocity_ids, descriptor,
                                  weights, directory, step, a.eval_batch, stage1=stage1)
                print(json.dumps(report), flush=True)
                torch.save({"head": head.state_dict(), "step": step, "manifest": manifest},
                           directory / f"step_{step:06d}.pth")
        terminal = json.loads((directory / f"eval_{a.steps:06d}.json").read_text())
        atomic_json(directory / "result.json",
                    dict(status="completed", initial=initial, terminal=terminal,
                         elapsed_seconds=time.time() - started, seen=seen))
        print(json.dumps({"status": "completed", "terminal": terminal}), flush=True)
    except BaseException:
        atomic_json(directory / "failure.json",
                    dict(status="failed", traceback=traceback.format_exc(),
                         elapsed_seconds=time.time() - started))
        raise


if __name__ == "__main__":
    main()
