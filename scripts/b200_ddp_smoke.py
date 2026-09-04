import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # NCCL should see the intended rank device before communicator creation.
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()

    if rank == 0:
        print("phase=initialized", flush=True)
    # Start small so a platform collective problem fails at a precise phase.
    x = torch.ones(512 * 1024, dtype=torch.bfloat16, device=device)
    warmups = 3
    for _ in range(warmups):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("phase=warmup_complete", flush=True)
    start = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Report the slowest rank, which determines synchronous DDP throughput.
    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    worst_elapsed = elapsed_tensor.item()
    if rank == 0:
        print("phase=all_reduce_complete", flush=True)

    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        _ = a @ a
    torch.cuda.synchronize()
    start = time.perf_counter()
    repetitions = 3
    for _ in range(repetitions):
        y = a @ a
    torch.cuda.synchronize()
    gemm_ms = (time.perf_counter() - start) * 1000.0 / repetitions
    gemm_tensor = torch.tensor(gemm_ms, dtype=torch.float64, device=device)
    dist.all_reduce(gemm_tensor, op=dist.ReduceOp.MAX)

    expected = float(dist.get_world_size() ** (iterations + warmups))
    local_ok = torch.tensor(
        int(torch.isfinite(x).all().item() and x[0].item() == expected and torch.isfinite(y).all().item()),
        device=device,
    )
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        payload_bytes = x.numel() * x.element_size()
        algorithmic_gbps = payload_bytes * iterations / worst_elapsed / 1e9
        bus_gbps = algorithmic_gbps * 2.0 * (dist.get_world_size() - 1) / dist.get_world_size()
        print(
            {
                "world_size": dist.get_world_size(),
                "devices": [torch.cuda.get_device_name(i) for i in range(dist.get_world_size())],
                "all_reduce_payload_mib": payload_bytes / (1024**2),
                "all_reduce_ms": worst_elapsed * 1000.0 / iterations,
                "algorithmic_GBps": algorithmic_gbps,
                "estimated_bus_GBps": bus_gbps,
                "bf16_gemm_8192_ms_slowest_rank": gemm_tensor.item(),
                "finite_and_collective_correct": bool(local_ok.item()),
            }
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
