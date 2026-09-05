#!/usr/bin/env python
"""⑤-D temporal SparseScoreDrive 의 3090 실측 지연시간.

배포 경로(forward_temporal: 완성 후보 12개 반환, full-K 미노출)를 통째로 잰다.
ETRI 규정: T_infer = stream reset 이후 최종 출력까지 모든 model forward 누적.
temporal 모델은 현재+과거를 한 번의 model forward 안에서 처리하므로 그 1회가 전부다.

  python bench_temporal_3090.py --bank bank_A0_deploy.npz --repeat 50
"""
import argparse
import json
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_A0_deploy.npz")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--out", default="temporal_latency_3090.json")
    args = ap.parse_args()

    from sparse_scoredrive import SparseScoreDrive
    dev = torch.device("cuda:0")
    print("torch %s | %s" % (torch.__version__, torch.cuda.get_device_name(0)),
          flush=True)

    configs = [
        ("T1 current-only", 0, 1.0),
        ("T2 (t-0.5)", 1, 1.0),
        ("T3 (t-1.0)", 2, 1.0),
        ("T4 (t-1.5)", 3, 1.0),
        ("T4 hist 1/2", 3, 0.5),
        ("T7 비교", 6, 1.0),
    ]
    rows = []
    for tag, nh, scale in configs:
        m = SparseScoreDrive(args.bank, logit_norm=True, n_hist=nh).to(dev).eval()
        img = torch.randn(1, 6, 3, 432, 768, device=dev)
        h, w = int(432 * scale), int(768 * scale)
        ih = torch.randn(1, max(nh, 1), 6, 3, h, w, device=dev)
        hT = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(
            1, max(nh, 1), 4, 4).contiguous()
        l2i = torch.randn(1, 6, 4, 4, device=dev)
        l2i[:, :, 3, 3] = 1.0

        def run():
            with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
                if nh == 0:
                    return m(img, l2i)
                return m.forward_temporal(img, ih, l2i, hT)

        out = run()
        assert set(out) == {"candidate_xy_abs_5s", "candidate_xy_inc_5s",
                            "visual_logits", "candidate_ids"}, sorted(out)
        assert out["candidate_xy_abs_5s"].shape == (1, 12, 10, 2)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.asarray(ts)
        peak = torch.cuda.max_memory_allocated() / 1e6
        pen = 1.0 + max(0.0, float(np.median(ts)) - 100.0) / 200.0
        rows.append(dict(tag=tag, n_hist=nh, hist_scale=scale,
                         median=float(np.median(ts)), p90=float(np.percentile(ts, 90)),
                         p99=float(np.percentile(ts, 99)), std=float(ts.std()),
                         peak_mb=float(peak), penalty=pen))
        print("%-18s median %7.2f ms  p99 %7.2f  peak %6.0f MB  penalty x%.3f"
              % (tag, np.median(ts), np.percentile(ts, 99), peak, pen), flush=True)
        del m
        torch.cuda.empty_cache()

    json.dump(rows, open(args.out, "w"), indent=1)
    print("\nsaved %s" % args.out)
    t4 = [r for r in rows if r["n_hist"] == 3 and r["hist_scale"] == 1.0][0]
    print("GATE: T4 full-res median %.2f ms %s 100ms"
          % (t4["median"], "<=" if t4["median"] <= 100 else ">"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
