"""CPU-only tests of our own-process shared GPU safety controls."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_motiondrive_v2 as trainer


def test_default_memory_policy_does_not_query_or_touch_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: pytest.fail("unexpected CUDA query"))
    assert trainer.configure_cuda_memory(torch.device("cpu")) == {"enabled": False}
    assert trainer.configure_cuda_memory(torch.device("cuda:0")) == {"enabled": False}
    trainer.check_cuda_headroom(torch.device("cpu"))


@pytest.mark.parametrize("limit,reserve", [(-1, 1), (1, -1), (True, 1), (1., 1), (1, "1"), (0, 1), (1, 0)])
def test_malformed_memory_policy_rejected(limit, reserve):
    with pytest.raises(ValueError):
        trainer.configure_cuda_memory(torch.device("cuda:0"), limit, reserve)


def test_cap_is_applied_on_selected_device_only_after_headroom_check(monkeypatch):
    device = torch.device("cuda:2")
    calls = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (23000 * trainer.MIB, 180000 * trainer.MIB))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(total_memory=180000 * trainer.MIB))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: calls.append((f, d)))
    result = trainer.configure_cuda_memory(device, 12000, 8192)
    assert calls == [(12000 / 180000, device)]
    assert result["allocator_limit_mib"] == 12000 and result["min_free_mib"] == 8192
    assert "only" in result["scope"]


def test_insufficient_space_refuses_before_cap_or_model_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (20000 * trainer.MIB, 180000 * trainer.MIB))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: pytest.fail("must refuse first"))
    with pytest.raises(RuntimeError, match="Insufficient CUDA headroom"):
        trainer.configure_cuda_memory(torch.device("cuda:0"), 12000, 8192)


def test_memory_limit_refuses_cpu():
    with pytest.raises(ValueError, match="CUDA device"):
        trainer.configure_cuda_memory(torch.device("cpu"), 12000, 8192)


def test_runtime_pressure_fail_closed_without_signaling_any_process(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (8000 * trainer.MIB, 180000 * trainer.MIB))
    with pytest.raises(RuntimeError, match="stopping only this trainer"):
        trainer.check_cuda_headroom(torch.device("cuda:3"), 8192)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (8192 * trainer.MIB, 180000 * trainer.MIB))
    trainer.check_cuda_headroom(torch.device("cuda:3"), 8192)


def test_microbatch_slice_is_cpu_view_and_order_preserving():
    batch = {"images": torch.arange(12).reshape(6, 2), "row": torch.arange(6),
             "scenario": [f"s{i}" for i in range(6)]}
    part = trainer.slice_batch(batch, 2, 4)
    assert part["images"].data_ptr() == batch["images"][2:].data_ptr()
    assert part["row"].tolist() == [2, 3] and part["scenario"] == ["s2", "s3"]
    with pytest.raises(ValueError, match="Non-sample"):
        trainer.slice_batch({**batch, "bad": torch.ones(2)}, 0, 1)
    with pytest.raises(TypeError, match="Unsupported"):
        trainer.slice_batch({**batch, "bad": "scalar"}, 0, 1)


def test_resume_restores_sharing_and_microbatch_before_gpu_allocation():
    saved_args = {"phase": "joint", "microbatch": 2, "cuda_memory_limit_mib": 12000,
                  "cuda_min_free_mib": 8192, "bn_policy": "fixed"}
    saved_args.update({key: 1 for key in ("alpha_occ", "alpha_lane", "alpha_motion", "uncertainty", "lr",
                                        "backbone_lr", "weight_decay", "warmup", "precision", "seed")})
    saved = {"model_config": {"goal_on": True, "state_on": True, "backbone_arch": "resnet50"},
             "arguments": saved_args}
    args = SimpleNamespace(resume="checkpoint", goal_on=1, state_on=1, arch="resnet50", phase="joint",
                           microbatch=0, cuda_memory_limit_mib=0, cuda_min_free_mib=0, bn_policy="adaptive")
    trainer.restore_run_configuration(args, saved, set())
    assert (args.microbatch, args.cuda_memory_limit_mib, args.cuda_min_free_mib, args.bn_policy) == (2, 12000, 8192, "fixed")
    args.cuda_memory_limit_mib = 0
    with pytest.raises(ValueError, match="incompatible explicit --cuda-memory-limit-mib"):
        trainer.restore_run_configuration(args, saved, {"--cuda-memory-limit-mib"})
    source = Path(trainer.__file__).read_text()
    main = source[source.index("def main():"):]
    assert main.index("restore_run_configuration(") < main.index("configure_cuda_memory(") < main.index("model.to(device)")
