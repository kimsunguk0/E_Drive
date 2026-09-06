#!/usr/bin/env python
"""⑤-F 게이트: corridor ribbon + sequence head 의 기하/불변식 검증."""
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from sparse_scoredrive import SparseScoreDrive  # noqa: E402

OK = True


def gate(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    bank = C.BANK_A0
    base = SparseScoreDrive(bank, logit_norm=True, n_hist=3).to(dev).eval()
    cor = SparseScoreDrive(bank, logit_norm=True, n_hist=3,
                           corridor=(0.0, 0.75, -0.75, 1.5, -1.5),
                           seq_head="tcn").to(dev).eval()
    id1 = SparseScoreDrive(bank, logit_norm=True, n_hist=3,
                           corridor=(0.0,), seq_head="none").to(dev).eval()
    id1.load_state_dict(base.state_dict())

    print("=== G1: corridor=(0.0,) 은 기존과 bitwise 동일해야 한다 ===")
    B = 2
    img = torch.randn(B, 6, 3, 432, 768, device=dev)
    hist = torch.randn(B, 3, 6, 3, 432, 768, device=dev)
    l2i = torch.from_numpy(C.build_global_lidar2img()).to(dev)
    l2ib = l2i.unsqueeze(0).expand(B, -1, -1, -1)
    hT = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(B, 3, 4, 4).contiguous()
    with torch.no_grad():
        a = base.temporal_logits(img, hist, l2ib, hT)
        b = id1.temporal_logits(img, hist, l2ib, hT)
    gate("bitwise 동일", torch.equal(a, b), f"maxdiff={float((a-b).abs().max()):.3e}")

    print("\n=== G2: ribbon 기하 (법선 방향, 거리) ===")
    rb = cor.candidate_ribbon.reshape(1024, 10, 5, 2).cpu().numpy()
    ctr = cor.candidate_abs.cpu().numpy()
    gate("중심(offset 0)이 원 waypoint 와 bitwise 동일",
         np.array_equal(rb[:, :, 0, :], ctr))
    d = np.linalg.norm(rb[:, :, 1, :] - rb[:, :, 0, :], axis=-1)
    gate("+0.75m 오프셋 거리", np.allclose(d, 0.75, atol=1e-5),
         f"min={d.min():.4f} max={d.max():.4f}")
    d2 = np.linalg.norm(rb[:, :, 3, :] - rb[:, :, 0, :], axis=-1)
    gate("+1.5m 오프셋 거리", np.allclose(d2, 1.5, atol=1e-5),
         f"min={d2.min():.4f} max={d2.max():.4f}")
    # 진행방향과 직교
    prev = np.concatenate([np.zeros_like(ctr[:, :1]), ctr[:, :-1]], 1)
    dv = ctr - prev
    nv = np.linalg.norm(dv, axis=-1)
    mov = nv > 0.05
    lat = rb[:, :, 1, :] - rb[:, :, 0, :]
    dot = np.abs((dv * lat).sum(-1))[mov] / (nv[mov] * 0.75)
    gate("진행방향과 직교", dot.max() < 1e-5, f"max|cos|={dot.max():.3e}")
    # 허용오차는 좌표 크기에 맞춘 fp32 ulp 기준이어야 한다(좌표가 최대 ~113m).
    asym = np.abs((rb[:, :, 1, :] - ctr) + (rb[:, :, 2, :] - ctr))
    ulp = float(np.spacing(np.float32(np.abs(ctr).max())))
    gate("좌우 대칭 (fp32 ulp 이내)", asym.max() <= ulp,
         f"max={asym.max():.3e} = {asym.max()/ulp:.2f} ulp, median={np.median(asym):.1e}")

    print("\n=== G3: C4 불변식 (증거 0 -> 점수 0) ===")
    z = torch.zeros(B, 6, 3, 432, 768, device=dev)
    zh = torch.zeros(B, 3, 6, 3, 432, 768, device=dev)
    with torch.no_grad():
        lz = cor.temporal_logits(z, zh, l2ib, hT)
        out = cor.forward_temporal(z, zh, l2ib, hT) if hasattr(cor, "forward_temporal") else None
    gate("zero image -> logits 전부 동일(정보 없음)",
         float(lz.max() - lz.min()) < 1e-5, f"spread={float(lz.max()-lz.min()):.3e}")
    if out is not None:
        idx = out[2] if isinstance(out, (tuple, list)) and len(out) > 2 else None
        if idx is not None:
            gate("zero image -> 후보0(정지) 선택", int(idx.flatten()[0]) == 0,
                 f"idx={idx.flatten().tolist()[:2]}")

    print("\n=== G4: seq_head 0 -> 0 ===")
    x = torch.zeros(4, 128, 10, device=dev)
    with torch.no_grad():
        y = cor.seq_head(x)
    gate("bias 없음 확인", float(y.abs().max()) == 0.0, f"max={float(y.abs().max()):.3e}")

    print("\n=== G5: corridor waypoint 가시성 ===")
    from sparse_scoredrive import project_candidate_points
    for nm, pts, L in (("center", base.candidate_abs, 1),
                       ("ribbon5", cor.candidate_ribbon, 5)):
        _, vis = project_candidate_points(pts, l2ib[:1], (432, 768), base.heights)
        v = vis.any(dim=1).any(dim=-1).float().mean()   # cams, heights 중 하나라도
        print(f"    {nm:8s} 가시율 {100*float(v):.2f}%")
    print(f"\n{'ALL GATES PASS' if OK else 'GATE FAILURE'}")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
