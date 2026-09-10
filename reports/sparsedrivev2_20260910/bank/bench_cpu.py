"""Read-only public-model CPU timing; no GPU access or source edits."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import sys, time, json, pathlib, statistics, hashlib, gc, resource
import torch
sys.path.insert(0, "experiments/sparsedrivev2_20260910")
from public_model import PublicSparseDriveV2, load_etri_bank
from data import PlanDataset, model_inputs

out = pathlib.Path("reports/sparsedrivev2_20260910/bank/cpu_model_timing_memory.json")
assert not out.exists()
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
ds = PlanDataset("/NHNHOME/data/sukim/adcl", "reports/sparsedrivev2_20260910/split_audit/primary_manifest.json",
                 "tune", status_mode="causal_selection", limit=1)
x = ds[0]
inputs = {k: v.unsqueeze(0) for k,v in model_inputs(x).items()}
checkpoint = "checkpoints/sparsedrivev2_20260910/public/sparsedrive_navsimv1_92p2.ckpt"
banks = ["p1024_v256_native100m_v8", "p1024_v512_native100m_progress6", "p1024_v1024_native100m_progress6"]
model, coverage = PublicSparseDriveV2.from_public_checkpoint(checkpoint,
    bank_path="cache/sparsedrivev2_20260910/bank/" + banks[0] + ".npz", backend="grid")
model.eval()
h = model._trajectory_head
param_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
results = {}
with torch.inference_mode():
    for name in banks:
        for key,value in load_etri_bank("cache/sparsedrivev2_20260910/bank/"+name+".npz").items():
            h._buffers[key] = value.float().contiguous()
        model._validate_bank()
        gc.collect()
        model(**inputs)
        times = []
        for rep in range(3):
            before = time.perf_counter()
            output = model(**inputs)
            times.append(time.perf_counter()-before)
        features = model._backbone(inputs["images"])
        iv = features[-1].permute(0,1,3,4,2).flatten(1,3)
        layer = h.decoder.layers[0]
        st = model._status_encoding(inputs["status"])
        branch = []
        for rep in range(7):
            before = time.perf_counter()
            vel = h.vel_pos_embed(h.vel_vocab).unsqueeze(0)+st[:,None]
            vel = vel+layer.v_img_attention(vel,iv,iv,need_weights=False)[0]
            vel = layer.attend(vel,"v")
            layer.vel_mlp(vel).topk(64,1)
            branch.append(time.perf_counter()-before)
        sizes = {key:int(value.numel()*value.element_size()) for key,value in h._buffers.items()}
        results[name] = {"bank_shapes":{k:list(v.shape) for k,v in h._buffers.items()},
            "bank_buffer_bytes":sizes, "bank_buffer_total_bytes":sum(sizes.values()),
            "parameter_bytes":param_bytes, "whole_forward_seconds":times,
            "whole_forward_median_seconds":statistics.median(times), "velocity_stage0_seconds":branch,
            "velocity_stage0_median_seconds":statistics.median(branch),
            "coarse_shapes":[{"paths":list(c["path_scores"].shape), "velocities":list(c["velocity_scores"].shape)} for c in output["coarse"]],
            "final_candidate_shape":list(output["candidate_xy"].shape),
            "process_peak_rss_kib_so_far":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        print(json.dumps({"bank":name, "full_cpu_seconds":statistics.median(times),
            "velocity_cpu_seconds":statistics.median(branch), "bank_MiB":sum(sizes.values())/(1024**2)}),flush=True)
report = {"device":"CPU only; CUDA_VISIBLE_DEVICES empty", "torch":torch.__version__, "threads":4,
    "dtype":"float32", "batch":1, "image_shape":list(inputs["images"].shape), "warmup":1,
    "whole_repeats":3, "velocity_repeats":7,
    "public_model_sha256":hashlib.sha256(pathlib.Path("experiments/sparsedrivev2_20260910/public_model.py").read_bytes()).hexdigest(),
    "source_row":x["row"], "public_parameter_reuse":coverage["reused_learned_parameter_numel"],
    "note":"CPU reference grid_sample, not B200 native latency or training memory; identical pretrained weights/current frame; bank buffers are nongradient fp32 and have no Adam states",
    "results":results}
out.write_text(json.dumps(report,indent=2)+"\n")
