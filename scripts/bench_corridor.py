#!/usr/bin/env python
"""⑤-F: path-aware scorer(corridor ribbon + sequence head) 의 3090 지연시간.

학습 전에 잰다. 100ms 를 넘는 구성은 학습하지 않는다.
"""
import argparse, json, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bank_A0_deploy.npz")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--out", default="corridor_latency_3090.json")
    args = ap.parse_args()
    from sparse_scoredrive import SparseScoreDrive
    dev = torch.device("cuda:0")
    print("torch %s | %s" % (torch.__version__, torch.cuda.get_device_name(0)), flush=True)

    C0 = (0.0,)
    C3 = (0.0, 0.75, -0.75)
    C5 = (0.0, 0.75, -0.75, 1.5, -1.5)
    configs = [
        ("T7 base",              6, C0, "none", False),
        ("T7 +seq",              6, C0, "tcn",  False),
        ("T7 +cor3",             6, C3, "none", False),
        ("T7 +cor5",             6, C5, "none", False),
        ("T7 +cor5+seq",         6, C5, "tcn",  False),
        ("T7 +cor5+seq+p1",      6, C5, "tcn",  True),
        ("T4 +cor5+seq",         3, C5, "tcn",  False),
        ("T4 +cor5+seq+p1",      3, C5, "tcn",  True),
    ]
    rows = []
    for tag, nh, cor, sh, p1 in configs:
        try:
            m = SparseScoreDrive(args.bank, logit_norm=True, n_hist=nh,
                                 corridor=cor, seq_head=sh, use_p1=p1).to(dev).eval()
        except Exception as e:
            print("%-20s SKIP (%s)" % (tag, e), flush=True); continue
        img = torch.randn(1, 6, 3, 432, 768, device=dev)
        ih = torch.randn(1, nh, 6, 3, 432, 768, device=dev)
        hT = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(1, nh, 4, 4).contiguous()
        l2i = torch.randn(1, 6, 4, 4, device=dev); l2i[:, :, 3, 3] = 1.0

        def run():
            with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
                return m.forward_temporal(img, ih, l2i, hT)

        out = run()
        assert out["candidate_xy_abs_5s"].shape == (1, 12, 10, 2)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.repeat):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.asarray(ts)
        med = float(np.median(ts)); peak = torch.cuda.max_memory_allocated() / 1e6
        pen = 1.0 + max(0.0, med - 100.0) / 200.0
        rows.append(dict(tag=tag, n_hist=nh, n_lat=len(cor), seq=sh, p1=p1,
                         median=med, p95=float(np.percentile(ts, 95)),
                         p99=float(np.percentile(ts, 99)), peak_mb=float(peak),
                         penalty=pen))
        print("%-20s median %7.2f ms  p95 %7.2f  p99 %7.2f  peak %6.0f MB  x%.3f %s"
              % (tag, med, np.percentile(ts, 95), np.percentile(ts, 99), peak, pen,
                 "OK" if med <= 100 else "OVER"), flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(rows, open(args.out, "w"), indent=1)
    print("\nsaved %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
