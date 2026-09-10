"""Reproduce public checkpoint provenance, tensor coverage, and model smoke checks.

Run from worktree root with CUDA_VISIBLE_DEVICES set to the allocated GPU.
Use --download only if the checkpoint is missing; no environment installation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch

from experiments.sparsedrivev2_20260910.public_model import (
    DeformableFeatureAggregation, PUBLIC_SHA256, PublicSparseDriveV2, load_native_extension,
)

PIN = "696ef77924eb9e0a4b4047d013a50e9854bfa026"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection(b, h, w, device):
    # Ground X forward, Y left, Z up. Synthetic pinhole for numerical checks.
    p = torch.zeros(b, 3, 4, 4, device=device)
    p[:, :, 0, 0], p[:, :, 0, 1] = w/2, -w*.75
    p[:, :, 1, 0], p[:, :, 1, 2], p[:, :, 1, 3] = h/2, -w*.75, w*.75*1.5
    p[:, :, 2, 0], p[:, :, 3, 3] = 1, 1
    return p


def compare_dfa(device):
    torch.manual_seed(7)
    native = DeformableFeatureAggregation(8, "native").to(device).eval()
    reference = copy.deepcopy(native)
    reference.backend = "grid"
    base_feature = torch.randn(1, 4, 256, device=device)
    base_maps = [torch.randn(1, 3, 256, h, w, device=device) for h, w in ((16,32),(8,16),(4,8),(2,4))]
    anchors = torch.randn(1, 4, 8, 2, device=device)
    anchors[..., 0] = torch.linspace(4, 24, 8, device=device)
    p = projection(1, 64, 128, device)
    wh = torch.tensor([128, 64], device=device).expand(1, 3, 2)
    outputs, gradients = [], []
    # Non-default stream tests the deliberate current-stream launch adaptation.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for module in (native, reference):
            feat = base_feature.detach().clone().requires_grad_()
            maps = [x.detach().clone().requires_grad_() for x in base_maps]
            result = module(feat, anchors, maps, p, wh)
            grad = torch.autograd.grad(result.square().mean(), (feat, *maps))
            outputs.append(result.detach())
            gradients.append([x.detach() for x in grad])
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    error = float((outputs[0] - outputs[1]).abs().max())
    gradient_errors = [float((a-b).abs().max()) for a,b in zip(*gradients)]
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-4)
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-3)
    return {"max_output_error": error, "max_gradient_errors": gradient_errors,
            "current_stream_test": True, "passed": True}


def cpu_smoke(model):
    # Complete layers and pretrained weights, smaller bank only for CPU cost.
    head = model._trajectory_head
    bank = {"path_vocab": head.path_vocab[:4], "vel_vocab": head.vel_vocab[:4],
            "traj_vocab": head.traj_vocab[:4, :4], "traj_mask": head.traj_mask[:4, :4]}
    small = PublicSparseDriveV2(bank, backend="grid", path_filter=(4,2), velocity_filter=(4,2))
    state = {k:v for k,v in model.state_dict().items() if k not in {"_trajectory_head."+x for x in bank}}
    small.load_state_dict(state, strict=False)
    small.eval()
    t = time.perf_counter()
    with torch.no_grad():
        result = small(torch.zeros(1,3,3,64,128), projection(1,64,128,"cpu"))
    assert torch.isfinite(result["scores"]).all()
    return {"seconds": time.perf_counter()-t, "trajectory_shape": list(result["trajectory"].shape),
            "candidate_count": result["scores"].shape[1], "passed": True}


def gpu_smoke(model, device, backward=False):
    model = model.to(device).eval()
    images = torch.randn(1,3,3,256,512,device=device)
    p = projection(1,256,512,device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(images, p)
        torch.cuda.synchronize(device)
        t = time.perf_counter()
        result = model(images, p)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter()-t
    assert torch.isfinite(result["scores"]).all()
    head = model._trajectory_head
    expected = head.traj_vocab.flatten(0,1)[result["candidate_ids"], :6, :2]
    assert torch.equal(expected, result["candidate_xy"])
    chosen = head.traj_vocab.flatten(0,1)[result["selected_candidate_id"], :6, :2]
    assert torch.equal(chosen, result["trajectory"])
    with torch.no_grad():
        varied = model(images, p, status=torch.randn(1,8,device=device))
    assert torch.equal(head.traj_vocab.flatten(0,1)[varied["candidate_ids"], :6, :2], varied["candidate_xy"])
    report = {"seconds": seconds, "trajectory_shape": list(result["trajectory"].shape),
              "candidate_count": result["scores"].shape[1], "fixed_bank_row_identity": True,
              "varied_status_preserves_bank_rows": True, "passed": True}
    if backward:
        model.train()
        t = time.perf_counter()
        result = model(images, p)
        # Independent nontrivial score targets exercise all reused proposal paths.
        target = torch.randn_like(result["scores"])
        loss = torch.nn.functional.cross_entropy(result["scores"], target.softmax(-1))
        for coarse in result["coarse"]:
            loss = loss + torch.nn.functional.cross_entropy(coarse["path_scores"], torch.zeros(1,dtype=torch.long,device=device))
            loss = loss + torch.nn.functional.cross_entropy(coarse["velocity_scores"], torch.zeros(1,dtype=torch.long,device=device))
        loss.backward()
        torch.cuda.synchronize(device)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        report.update(backward_seconds=time.perf_counter()-t, loss=float(loss.detach()),
                      finite_gradient_tensors=len(gradients), backward_passed=True)
    report["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device)/1e9
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sparsedrivev2_20260910/public/sparsedrive_navsimv1_92p2.ckpt")
    parser.add_argument("--bank")
    parser.add_argument("--output", default="reports/sparsedrivev2_20260910/public_init/bootstrap.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists() and args.download:
        executable = shutil.which("hf")
        if executable is None:
            executable = str(Path(__import__("sys").executable).with_name("hf"))
        subprocess.run([executable,"download","wenchaosun/SparseDriveV2",checkpoint.name,
                        "--local-dir",str(checkpoint.parent)],check=True)
    revision = subprocess.check_output(["git","-C","third_party/SparseDriveV2","rev-parse","HEAD"],text=True).strip()
    if revision != PIN:
        raise RuntimeError(f"Wrong source revision: {revision}")
    sha = file_hash(checkpoint)
    if sha != PUBLIC_SHA256:
        raise RuntimeError(f"Wrong public checkpoint hash: {sha}")
    model, coverage = PublicSparseDriveV2.from_public_checkpoint(checkpoint,bank_path=args.bank)
    report = {"source_revision":revision,"checkpoint_sha256":sha,"checkpoint_bytes":checkpoint.stat().st_size,
              "torch_version":torch.__version__,"checkpoint_weights_only":True,"coverage":coverage,
              "adaptation": {"public_input":"current RGB front-left/front/front-right at 512x256",
              "image_mean":[.485,.456,.406],"image_std":[.229,.224,.225],"image_hw_order":"height,width",
              "public_timesteps":8,"etri_output_timesteps":6,"timesteps_seconds":[.5,1.,1.5,2.,2.5,3.],
              "score_default":"imitation logits for ETRI D3 training; original NAVSIM metric product also exposed",
              "status_default":"eight zeros; optional values affect fixed-bank selection only",
              "label_in_forward":False,"gridmask_training":"disabled; dataset augmentation controlled outside model"}}
    if not args.skip_cpu:
        report["cpu_smoke"] = cpu_smoke(model)
        print("CPU",report["cpu_smoke"],flush=True)
    if args.device.startswith("cuda"):
        load_native_extension()
        report["dfa_parity"] = compare_dfa(args.device)
        print("DFA",report["dfa_parity"],flush=True)
        report["gpu_smoke"] = gpu_smoke(model,args.device,args.backward)
        print("GPU",report["gpu_smoke"],flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2))
    print(str(target),flush=True)


if __name__ == "__main__":
    main()
