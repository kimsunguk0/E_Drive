"""V2.0 full-forward latency on RTX 3090, swept over past-frame resolution.

Latency does not depend on weight values, so the model is randomly initialised;
only the architecture and the input shapes matter. Protocol follows the 09-08
P7 canary: batch 1, CUDA events with synchronize, warmup then timed repeats,
model forward only. Reference point: P7-C trained LAST6000 measured 38.69 ms
median on this machine.
"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/work")
from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2.model import MotionDriveV2
from models.motiondrive_v2.shared_status_query import install_shared_status_query

CFG = json.loads(sys.argv[1])
WARMUP, REPEATS = 20, 50
dev = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
cfg = MotionDriveV2Config(**CFG)
model = install_shared_status_query(MotionDriveV2(cfg)).to(dev).eval()
print("params %.2fM  gpu %s" % (sum(p.numel() for p in model.parameters())/1e6,
                                torch.cuda.get_device_name(0)), flush=True)
CUR_H, CUR_W = 432, 768
NH = cfg.n_history
def make(hist_h, hist_w, cur_h=CUR_H, cur_w=CUR_W):
    g = torch.Generator(device="cpu").manual_seed(0)
    return dict(
        images=torch.randn(1, 6, 3, cur_h, cur_w, generator=g).to(dev),
        history_images=torch.randn(1, NH, 3, hist_h, hist_w, generator=g).to(dev),
        lidar2img=torch.eye(4).repeat(1, 6, 1, 1).to(dev),
        history_transforms=torch.eye(4).repeat(1, NH, 1, 1).to(dev),
        time_offsets=torch.tensor([[0.1, 0.2, 0.5, 1.0]]).to(dev),
        goal_xy=torch.tensor([[50.0, 0.0]]).to(dev),
        provided_status5=torch.zeros(1, 5, device=dev))
def bench(tag, hist_hw, cur_hw=(CUR_H, CUR_W)):
    inp = make(*hist_hw, *cur_hw)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        for _ in range(WARMUP): model(**inp)
        torch.cuda.synchronize()
        ms = []
        for _ in range(REPEATS):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); model(**inp); e.record(); torch.cuda.synchronize()
            ms.append(s.elapsed_time(e))
    a = np.array(ms)
    torch.cuda.reset_peak_memory_stats()
    print("  %-34s median %7.2f  p95 %7.2f  p99 %7.2f  mean %7.2f  min %7.2f" % (
        tag, np.median(a), np.percentile(a, 95), np.percentile(a, 99), a.mean(), a.min()), flush=True)
    return {"tag": tag, "hist_hw": list(hist_hw), "cur_hw": list(cur_hw),
            "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}
out = []
print("\n=== past-frame resolution sweep (current views fixed at 432x768) ===", flush=True)
out.append(bench("history 216x384  (CURRENT)", (216, 384)))
out.append(bench("history 288x512  (1.33x)", (288, 512)))
out.append(bench("history 324x576  (1.5x)", (324, 576)))
out.append(bench("history 432x768  (2x = full)", (432, 768)))
print("\n=== also raising the current views ===", flush=True)
out.append(bench("history 432x768 + current 540x960", (432, 768), (540, 960)))
json.dump({"schema_version": 1, "gpu": torch.cuda.get_device_name(0),
           "protocol": {"batch": 1, "warmup": WARMUP, "repeats": REPEATS,
                        "cuda_events": True, "model_forward_only": True,
                        "random_init_weights": True},
           "reference_p7_trained_median_ms": 38.69,
           "budget": {"dev_3090_median_ms": 80.0, "dev_3090_p95_ms": 90.0,
                      "official_4090_ms": 100.0},
           "results": out}, open("/work/v2_latency_resolution.json", "w"), indent=1)
print("\nwrote /work/v2_latency_resolution.json", flush=True)
