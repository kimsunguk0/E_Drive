"""Matched, bounded SparseDriveV2 ETRI training with separate D3 labels."""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from data import PlanDataset, d3, file_sha, model_inputs, rows_sha
from losses import selection_loss
from public_model import PublicSparseDriveV2
from goal_selector import GoalConditionedSelector

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SHA = "330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735"


def verify_initialization(checkpoint, bank_path, train, val):
    if file_sha(checkpoint) != PUBLIC_SHA:
        raise ValueError("Initialization must be the pinned official NAVSIMv1 public checkpoint")
    bank_path = Path(bank_path)
    sidecar = json.loads(bank_path.with_suffix(bank_path.suffix + ".json").read_text())
    if sidecar.get("bank_sha256", sidecar.get("artifact_sha256")) != file_sha(bank_path):
        raise ValueError("Bank sidecar hash mismatch")
    with np.load(bank_path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata_json"]))
        bank_rows = z["train_rows"]
        if bank_rows.dtype.kind not in "iu" or bank_rows.ndim != 1:
            raise ValueError("Bank training row indices are malformed")
        if (not np.array_equal(bank_rows, train.allowed_rows)
                or metadata["train_rows_sha256"] != rows_sha(bank_rows)
                or str(z["train_rows_sha256"]) != rows_sha(bank_rows)):
            raise ValueError("Bank must be fitted using exactly this run's allowed training rows")
    if metadata["partition"] != "train" or metadata["split_sha256"] != file_sha(train.split_manifest):
        raise ValueError("Bank training partition/split provenance mismatch")
    bank_scenes = set(metadata["train_scenes"])
    val_scenes = set(val.provenance()["scenes"])
    if bank_scenes & val_scenes:
        raise ValueError("Bank construction includes validation scenes")
    return {"bank_sidecar_sha256":file_sha(bank_path.with_suffix(bank_path.suffix + ".json")),
            "bank_train_rows_sha256":rows_sha(bank_rows), "bank_training_metadata":metadata}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fixed_bn_train(model):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def transfer(batch):
    return {k:v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}


def autocast(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()


def verify_output(output):
    for key in ("trajectory", "candidate_xy", "scores"):
        if not torch.isfinite(output[key]).all():
            raise RuntimeError(f"Nonfinite model output: {key}")
    valid = output.get("candidate_valid", torch.ones_like(output["scores"],dtype=torch.bool))
    if valid.ndim == 3:
        valid = valid.all(-1)
    index = output["scores"].argmax(-1)
    if not torch.gather(valid.bool(),1,index[:,None]).all():
        raise RuntimeError("The selected candidate is invalid")
    selected = torch.gather(output["candidate_ids"],1,index[:,None]).squeeze(1)
    if not torch.equal(selected,output["selected_candidate_id"]):
        raise RuntimeError("Selection IDs differ from the scored complete candidates")


@torch.inference_mode()
def evaluate(model, dataset, args, directory, step):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.eval_batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    errors, oracle_errors, predictions, rows, sessions = [], [], [], [], []
    point_errors, xy_errors = [], []
    ids = []
    for batch in loader:
        x = transfer(batch)
        with autocast(args.precision):
            output = model(**model_inputs(x, goal_selection=args.goal_mode == "selection"))
        verify_output(output)
        pred = output["trajectory"].float()
        if not torch.isfinite(pred).all():
            raise RuntimeError("Nonfinite validation prediction")
        candidate = output["candidate_xy"]
        costs = d3(candidate, x["gt_plan"][:, None].expand_as(candidate))
        valid = output.get("candidate_valid", torch.isfinite(output["scores"]))
        if valid.ndim == 3:
            valid = valid.all(-1)
        if not valid.any(-1).all():
            raise RuntimeError("Validation sample has no complete candidate")
        oracle = costs.masked_fill(~valid.bool(), float("inf")).amin(-1)
        errors.append(d3(pred, x["gt_plan"]).cpu().numpy())
        delta = pred - x["gt_plan"].float()
        point_errors.append(torch.linalg.vector_norm(delta,dim=-1).cpu().numpy())
        xy_errors.append(delta.cpu().numpy())
        oracle_errors.append(oracle.cpu().numpy())
        predictions.append(pred.cpu().numpy())
        rows.append(batch["row"].numpy())
        ids.append(output["selected_candidate_id"].cpu().numpy())
        sessions.extend(batch["session"])
    error, oracle = np.concatenate(errors), np.concatenate(oracle_errors)
    point_error, xy_error = np.concatenate(point_errors), np.concatenate(xy_errors)
    by_session = {}
    for session, value in zip(sessions, error):
        by_session.setdefault(session, []).append(float(value))
    result = {"step": step, "n": len(error), "official_d3": float(error.mean()),
              "shortlist_oracle_d3": float(oracle.mean()),
              "selection_regret": float((error-oracle).mean()),
              "point_l2_metres_05_to_30": point_error.mean(0).astype(float).tolist(),
              "prefix_ade_1_2_3s": [float(point_error[:,:n].mean()) for n in (2,4,6)],
              "three_second_endpoint_l2": float(point_error[:,-1].mean()),
              "session_d3": {k:float(np.mean(v)) for k,v in by_session.items()}}
    np.savez_compressed(directory / f"eval_{step:06d}.npz", rows=np.concatenate(rows),
                        pred=np.concatenate(predictions), d3=error, shortlist_oracle=oracle,
                        point_l2=point_error, error_xy=xy_error,
                        candidate_id=np.concatenate(ids))
    atomic_json(directory / f"eval_{step:06d}.json", result)
    print(json.dumps({"evaluation": result}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--base", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--status-mode", choices=("zero", "causal_selection"), required=True)
    p.add_argument("--goal-mode", choices=("none", "selection"), default="none")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--temperature", type=float, default=.1)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--train-rows")
    p.add_argument("--eval-rows")
    p.add_argument("--eval-split", choices=("train", "tune"), default="tune")
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=0)
    args = p.parse_args()
    device = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device not in ("0", "1", "4"):
        raise RuntimeError("Expose exactly one user-assigned GPU: 0, 1, or 4")
    if min(args.steps, args.batch, args.eval_batch, args.eval_every) < 1:
        raise ValueError("Positive step and batch counts required")
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        seed_all(args.seed)
        torch.set_num_threads(4)
        common = dict(base=args.base, split_manifest=args.split_manifest, status_mode=args.status_mode,
                      goal_mode=args.goal_mode)
        train = PlanDataset(**common, split="train", rows_file=args.train_rows,
                            augment=True, seed=args.seed, limit=args.train_limit)
        val = PlanDataset(**common, split=args.eval_split, rows_file=args.eval_rows, stride=5,
                          augment=False, limit=args.eval_limit)
        train_meta, val_meta = train.provenance(), val.provenance()
        if set(train_meta["sessions"]) & set(val_meta["sessions"]):
            raise ValueError("Training and validation sessions overlap")
        initialization = verify_initialization(args.checkpoint,args.bank,train,val)
        model, coverage = PublicSparseDriveV2.from_public_checkpoint(
            args.checkpoint, bank_path=args.bank, backend="native", score_mode="imitation")
        if args.goal_mode == "selection":
            model = GoalConditionedSelector(model)
            coverage["goal_adapter"] = {"new_parameter_elements": 512, "initialization": "zeros, no bias",
                "normalization_metres": 50., "use": "fixed complete bank selection",
                "state_dict_base_prefix": "base."}
        model.cuda()
        parameters = [(n,v) for n,v in model.named_parameters() if v.requires_grad]
        backbone = [v for n,v in parameters if n.removeprefix("base.").startswith("_backbone.")]
        head = [v for n,v in parameters if not n.removeprefix("base.").startswith("_backbone.")]
        if not backbone or not head:
            raise ValueError("Public backbone/head parameter groups were not found")
        optimizer = torch.optim.AdamW([{"params":backbone,"lr":args.backbone_lr},
                                      {"params":head,"lr":args.lr}], weight_decay=args.weight_decay)
        base_lrs = [args.backbone_lr, args.lr]
        runtime = [Path(__file__).with_name(name) for name in ("train.py","data.py","losses.py","public_model.py","goal_selector.py")]
        runtime.extend((ROOT/"third_party/SparseDriveV2/navsim/agents/sparsedrive/ops/src").glob("*"))
        runtime = [path for path in runtime if path.is_file()]
        sources = {str(path.relative_to(ROOT)):file_sha(path) for path in runtime}
        for path in runtime:
            destination = directory / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        manifest = {"arguments": vars(args), "physical_gpu":int(device), "pid":os.getpid(),
                    "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                    "source_sha256":sources, "torch":str(torch.__version__),
                    "public_checkpoint_sha256":file_sha(args.checkpoint), "bank_sha256":file_sha(args.bank),
                    "initialization_audit":initialization,
                    "train":train_meta, "validation":val_meta, "public_load_coverage":coverage,
                    "score_mode":"imitation", "bn":"fixed running stats; affine trainable",
                    "training_rng_seed": args.seed + 1,
                    "loss":"fine D3 soft CE + 0.5*(mean path-at-GT-progress D3 CE + mean time-weighted progress-error CE); velocity auxiliary is not Euclidean D3"}
        atomic_json(directory / "manifest.json", manifest)
        # Wrapper construction consumes RNG even with zero-initialized weights.
        # Reset after all construction so both arms see the same shuffle/dropout RNG.
        seed_all(args.seed + 1)
        loader = DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=args.workers,
                            pin_memory=True,drop_last=False)
        iterator = iter(loader)
        epoch = 0
        best = float("inf")
        history = []
        initial = evaluate(model,val,args,directory,0)
        for step in range(1,args.steps+1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                train.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            factor = min(step / max(args.warmup,1), 1.)
            if step > args.warmup:
                factor *= .5 * (1 + math.cos(math.pi * (step-args.warmup) / max(args.steps-args.warmup,1)))
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * factor
            fixed_bn_train(model)
            x = transfer(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args.precision):
                output = model(**model_inputs(x, goal_selection=args.goal_mode == "selection"))
            verify_output(output)
            with torch.autocast("cuda", enabled=False):
                losses = selection_loss(output,x["gt_plan"],model._trajectory_head.path_vocab[..., :2],
                                        model._trajectory_head.vel_vocab,temperature=args.temperature)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Nonfinite training loss")
            losses["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % 10 == 0:
                record = {"step":step,"epoch":epoch,"seconds":time.time()-started,
                          "grad_norm":float(norm), "lr":optimizer.param_groups[1]["lr"],
                          **{k:float(v.detach()) for k,v in losses.items()}}
                with (directory / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                print(json.dumps(record),flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                result = evaluate(model,val,args,directory,step)
                history.append(result)
                checkpoint = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                              "step":step,"epoch":epoch,"manifest":manifest,"result":result}
                save_checkpoint(directory / "last.pth",checkpoint)
                if result["official_d3"] < best:
                    best = result["official_d3"]
                    save_checkpoint(directory / "best.pth",checkpoint)
                atomic_json(directory / "progress.json",{"status":"running","step":step,
                    "terminal_so_far":result,"best_d3":best,"elapsed_seconds":time.time()-started})
        atomic_json(directory / "result.json",{"status":"completed","initial":initial,
            "terminal":history[-1],"best_d3":best,"steps":args.steps,"history":history,
            "elapsed_seconds":time.time()-started,"peak_cuda_bytes":torch.cuda.max_memory_allocated()})
    except BaseException:
        atomic_json(directory / "failure.json",{"status":"failed","traceback":traceback.format_exc(),
                                                "elapsed_seconds":time.time()-started})
        raise


if __name__ == "__main__":
    main()
