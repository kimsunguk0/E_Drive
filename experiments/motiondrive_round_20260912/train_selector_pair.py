"""Train one selector over frozen M8 candidates, with or without the goal.

Both arms share the frozen generator, the cached candidates, the architecture,
the initialisation, the batch order and the budget. The only difference is
whether the five goal slots carry the provided goal or zeros, which is what
isolates the question the work order asks.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TIME_WEIGHTS = (11 / 36, 11 / 36, 5 / 36, 5 / 36, 2 / 36, 2 / 36)


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


def masked_soft_cost_ce(logits, costs, valid, temperature=0.1):
    if logits.shape != costs.shape or logits.shape != valid.shape:
        raise ValueError("Expected matching [B,M] tensors")
    if valid.dtype != torch.bool or not bool(valid.any(-1).all()):
        raise ValueError("Each row needs a valid candidate")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not bool(torch.isfinite(logits[valid]).all()):
        raise ValueError("Non-finite valid logits")
    if not bool(torch.isfinite(costs[valid]).all()):
        raise ValueError("Non-finite valid costs")
    target_logits = (-costs.detach().float() / temperature).masked_fill(~valid, -torch.inf)
    target = F.softmax(target_logits, dim=-1)
    masked_logits = logits.float().masked_fill(~valid, -torch.inf)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    safe_log_prob = log_prob.masked_fill(~valid, 0.0)
    return -(target * safe_log_prob).sum(-1).mean()


class CompletedCandidateSelector(nn.Module):
    """Residual on top of the frozen base score; goal enters only here."""

    def __init__(self, channels=128, steps=6):
        super().__init__()
        self.hidden = nn.Sequential(nn.LayerNorm(channels * steps),
                                    nn.Linear(channels * steps, 128), nn.GELU())
        self.score = nn.Sequential(nn.Linear(128 + 34 + 1 + 5, 128), nn.GELU(),
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    @staticmethod
    def geometry(xy):
        position = xy / xy.new_tensor([80.0, 64.0])
        velocity = torch.diff(xy, dim=-2, prepend=torch.zeros_like(xy[..., :1, :])) * 2.0
        acceleration = torch.diff(velocity, dim=-2) * 2.0 / 3.0
        return torch.cat([position.flatten(-2),
                          (velocity / xy.new_tensor([20.0, 5.0])).flatten(-2),
                          (acceleration / 3.0).flatten(-2)], -1)

    def forward(self, hidden, xy, base_scores, valid, goal_xy, use_goal):
        b, m = base_scores.shape
        centered = base_scores - (base_scores * valid).sum(-1, keepdim=True) / valid.sum(
            -1, keepdim=True).clamp_min(1)
        if use_goal:
            goal = torch.cat([goal_xy[:, None].expand(b, m, 2) / 50.0,
                              (xy[:, :, -1] - goal_xy[:, None]) / 50.0,
                              torch.ones(b, m, 1, device=xy.device)], -1)
        else:
            goal = torch.zeros(b, m, 5, device=xy.device)
        features = torch.cat([self.hidden(hidden.reshape(b, m, -1)), self.geometry(xy),
                              centered[..., None], goal], -1)
        residual = self.score(features).squeeze(-1)
        return base_scores.detach() + residual


class Cache:
    KEYS = ("row", "candidate_xy", "candidate_hidden", "base_scores",
            "candidate_valid", "goal_xy", "gt_plan", "session")

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
        self.n = len(self.arrays["row"])
        self.device = device

    def batch(self, index):
        take = lambda k: torch.from_numpy(np.ascontiguousarray(self.arrays[k][index])).to(self.device)
        xy = take("candidate_xy").float()
        gt = take("gt_plan").float()
        distance = torch.linalg.vector_norm(xy - gt[:, None], dim=-1)
        cost = (distance * distance.new_tensor(TIME_WEIGHTS)).sum(-1)
        return (take("candidate_hidden").float(), xy, take("base_scores").float(),
                take("candidate_valid").bool(), take("goal_xy").float(), cost)


@torch.inference_mode()
def evaluate(head, cache, use_goal, directory, step, batch=128, prefix="eval"):
    head.eval()
    chunks = {k: [] for k in ("d3", "oracle", "base_d3", "mode", "point_l2")}
    for start in range(0, cache.n, batch):
        index = np.arange(start, min(start + batch, cache.n))
        hidden, xy, base, valid, goal, cost = cache.batch(index)
        scores = head(hidden, xy, base, valid, goal, use_goal)
        rows = torch.arange(len(index), device=cache.device)
        pick = scores.masked_fill(~valid, -torch.inf).argmax(-1)
        base_pick = base.masked_fill(~valid, -torch.inf).argmax(-1)
        chunks["d3"].append(cost[rows, pick].cpu().numpy())
        chunks["base_d3"].append(cost[rows, base_pick].cpu().numpy())
        chunks["oracle"].append(cost.masked_fill(~valid, float("inf")).min(-1).values.cpu().numpy())
        chunks["mode"].append(pick.cpu().numpy())
        gt = torch.from_numpy(np.ascontiguousarray(cache.arrays["gt_plan"][index])).to(cache.device)
        chunks["point_l2"].append(torch.linalg.vector_norm(
            xy[rows, pick] - gt, dim=-1).cpu().numpy())
    arrays = {k: np.concatenate(v, 0) for k, v in chunks.items()}
    report = {"step": step, "n": int(cache.n), "official_d3": float(arrays["d3"].mean()),
              "oracle_d3": float(arrays["oracle"].mean()),
              "selection_regret": float((arrays["d3"] - arrays["oracle"]).mean()),
              "base_scorer_d3": float(arrays["base_d3"].mean()),
              "point_l2": arrays["point_l2"].mean(0).tolist(),
              "mode_histogram": np.bincount(arrays["mode"], minlength=8).tolist()}
    atomic_json(Path(directory) / f"{prefix}_{step:06d}.json", report)
    np.savez_compressed(Path(directory) / f"{prefix}_{step:06d}.npz",
                        row=np.asarray(cache.arrays["row"]),
                        session=np.asarray(cache.arrays["session"]), **arrays)
    head.train()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--tune-cache", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--goal", choices=("on", "off"), required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.1)
    a = p.parse_args()
    if a.batch % a.microbatch:
        raise ValueError("effective batch must be a whole number of microbatches")
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
        train, tune = Cache(a.train_cache, device), Cache(a.tune_cache, device)
        if set(np.asarray(train.arrays["session"]).tolist()) & set(
                np.asarray(tune.arrays["session"]).tolist()):
            raise RuntimeError("train and tune sessions overlap")
        if train.manifest["generator_sha256"] != tune.manifest["generator_sha256"]:
            raise RuntimeError("the two caches came from different generators")
        use_goal = a.goal == "on"
        head = CompletedCandidateSelector().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        source = Path(__file__).resolve()
        (directory / "source").mkdir(exist_ok=True)
        shutil.copy2(source, directory / "source" / source.name)
        manifest = dict(schema="m8_selector_v1", arguments=vars(a),
                        source_sha256={source.name: sha(source)},
                        generator_sha256=train.manifest["generator_sha256"],
                        selection_goal=("provided" if use_goal else "zeroed"),
                        scene_goal="already consumed upstream by the frozen generator",
                        trainable="selector only; the generator is frozen and cached",
                        parameters=sum(x.numel() for x in head.parameters()),
                        initial_head_sha256=hashlib.sha256(b"".join(
                            v.detach().cpu().numpy().tobytes()
                            for _, v in sorted(head.state_dict().items()))).hexdigest(),
                        physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"], pid=os.getpid())
        atomic_json(directory / "manifest.json", manifest)
        initial = evaluate(head, tune, use_goal, directory, 0)
        print(json.dumps({"initial": initial}), flush=True)
        generator = torch.Generator().manual_seed(a.seed + 1)
        permutation, cursor = torch.randperm(train.n, generator=generator), 0
        accumulation = a.batch // a.microbatch
        log = open(directory / "train.jsonl", "w")
        for step in range(1, a.steps + 1):
            factor = min(step / max(a.warmup, 1), 1.0)
            if step > a.warmup:
                factor *= .5 * (1 + math.cos(math.pi * (step - a.warmup) / max(a.steps - a.warmup, 1)))
            for group in optimizer.param_groups:
                group["lr"] = a.lr * factor
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(accumulation):
                if cursor + a.microbatch > train.n:
                    permutation, cursor = torch.randperm(train.n, generator=generator), 0
                index = np.sort(permutation[cursor:cursor + a.microbatch].numpy())
                cursor += a.microbatch
                hidden, xy, base, valid, goal, cost = train.batch(index)
                scores = head(hidden, xy, base, valid, goal, use_goal)
                loss = masked_soft_cost_ce(scores, cost, valid, a.temperature) / accumulation
                loss.backward()
                total += float(loss)
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 10 == 0:
                log.write(json.dumps({"step": step, "loss": total, "grad_norm": float(norm),
                                      "lr": a.lr * factor}) + "\n")
                log.flush()
            if step % a.eval_every == 0 or step == a.steps:
                report = evaluate(head, tune, use_goal, directory, step)
                print(json.dumps(report), flush=True)
                torch.save({"head": head.state_dict(), "step": step, "manifest": manifest},
                           directory / f"step_{step:06d}.pth")
        terminal = json.loads((directory / f"eval_{a.steps:06d}.json").read_text())
        train_probe = evaluate(head, train, use_goal, directory, a.steps, prefix="train_eval")
        atomic_json(directory / "result.json",
                    dict(status="completed", initial=initial, terminal=terminal,
                         train_probe=train_probe, elapsed_seconds=time.time() - started))
        print(json.dumps({"status": "completed", "terminal": terminal,
                          "train_probe": train_probe}), flush=True)
    except BaseException:
        atomic_json(directory / "failure.json",
                    dict(status="failed", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
