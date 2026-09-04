#!/usr/bin/env python
"""Phase 2 gate — GPU availability + an fp16 matmul baseline.

The point of the baseline is NOT to measure the H200's peak. This GPU is shared
with other users' training jobs, so the number here is "what we actually got
while contending", and it is recorded so a later re-run can be compared against
it to judge how much contention changed.

Hard constraint: stay under 2 GB of VRAM and finish in seconds. Another user's
job is resident on this device.
"""
import json
import os
import sys
import time

import torch

BUDGET_BYTES = 2 * 1024 ** 3
N = 8192           # 8192^2 fp16 = 128 MiB per matrix; 3 matrices = 384 MiB
WARMUP, ITERS = 5, 50


def main():
    out = {}
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version()
    out["arch_list"] = torch.cuda.get_arch_list()

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1
    out["available"] = True
    out["device_name"] = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()
    out["capability"] = list(cap)
    free, total = torch.cuda.mem_get_info()
    out["vram_total_gib"] = round(total / 1024 ** 3, 2)
    out["vram_free_gib_at_start"] = round(free / 1024 ** 3, 2)

    print(f"device      : {out['device_name']}")
    print(f"capability  : {cap}")
    print(f"torch       : {torch.__version__} (cuda {torch.version.cuda}, cudnn {out['cudnn']})")
    print(f"arch_list   : {' '.join(out['arch_list'])}")
    print(f"VRAM        : {out['vram_free_gib_at_start']} GiB free / {out['vram_total_gib']} GiB total")

    if cap != (9, 0):
        print(f"FAIL: expected compute capability (9, 0), got {cap}")
        return 1

    # --- who else is on the device (context for the number below) ---
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        others = [l.strip() for l in q.stdout.strip().splitlines() if l.strip()]
        out["coresident_procs"] = others
        print(f"co-resident : {others if others else 'none'}")
    except Exception as e:                                   # noqa: BLE001
        out["coresident_procs"] = f"query failed: {e}"

    # --- fp16 matmul ---
    torch.cuda.reset_peak_memory_stats()
    a = torch.randn(N, N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, N, device="cuda", dtype=torch.float16)

    for _ in range(WARMUP):
        c = a @ b
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    flops = 2.0 * N ** 3 * ITERS
    tflops = flops / dt / 1e12
    peak = torch.cuda.max_memory_allocated()

    out["matmul"] = {
        "shape": [N, N], "dtype": "float16", "iters": ITERS,
        "seconds": round(dt, 4),
        "tflops": round(tflops, 1),
        "peak_vram_mib": round(peak / 1024 ** 2, 1),
    }
    print(f"fp16 matmul : {N}x{N} x{ITERS} -> {tflops:.1f} TFLOP/s "
          f"({dt:.3f}s, peak {peak / 1024 ** 2:.0f} MiB)")

    assert c.isfinite().all(), "matmul produced non-finite values"

    if peak > BUDGET_BYTES:
        print(f"FAIL: peak VRAM {peak / 1024 ** 3:.2f} GiB exceeds the 2 GiB budget")
        return 1

    del a, b, c
    torch.cuda.empty_cache()

    dest = os.environ.get("BENCH_OUT")
    if dest:
        with open(dest, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {dest}")
    print("\nGATE 2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
