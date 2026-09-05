#!/usr/bin/env python
"""⑤-D temporal 기하 필수 검증 (user spec)."""
import os
import sys

import numpy as np
import torch

A = "/NHNHOME/data/sukim/adcl"
sys.path.insert(0, os.path.join(A, "scripts"))
import sparse_common as C  # noqa: E402
from train_sparse_scoredrive import TrainableSparseScoreDrive  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail))


def main():
    arr = C.load_arrays()
    sp = C.make_split(arr, all_frames=True, holdout_offset=True)
    dev = torch.device("cuda:0")
    l2i = torch.from_numpy(C.build_global_lidar2img())

    # 1) hist_T[k=0] 가 항등이 되는가? -> 과거 0 스텝이면 current-only 와 정확히 일치
    #    (stride 0 은 정의상 현재 시점: his index 30)
    r = int(sp["train_rows"][12345])
    T0 = C.history_transforms(arr, r, n_hist=1, stride=0)[0]
    check("stride0 변환 = 항등 (current-only 정확 일치)",
          np.allclose(T0, np.eye(4), atol=1e-6),
          "max|T-I|=%.2e" % np.abs(T0 - np.eye(4)).max())

    # 2) 실제 hist_T 가 자차 이동량과 일치하는가
    Ts = C.history_transforms(arr, r, n_hist=3)
    origin_in_past = [Ts[k][:3, 3] for k in range(3)]   # 현재 원점의 과거 좌표
    d = [float(np.linalg.norm(o[:2])) for o in origin_in_past]
    sp_ = float(arr["speed"][r])
    check("과거 시점에서 본 현재 원점 거리 ≈ speed×0.5k",
          all(abs(d[k] - sp_ * 0.5 * (k + 1)) < max(1.5, 0.35 * sp_ * 0.5 * (k + 1))
              for k in range(3)),
          "d=%s vs 예상 %s (speed %.1f)" %
          (np.round(d, 2).tolist(),
           np.round([sp_ * 0.5 * (k + 1) for k in range(3)], 2).tolist(), sp_))

    # 3) 시나리오 경계: history_available 이 frame>=15 만 허용
    rows = sp["train_rows"]
    av = C.history_available(arr, rows, n_hist=3)
    check("history 가용 = frame>=15 만",
          int(arr["frame"][rows][~av].max(initial=-1)) < 15 and
          int(arr["frame"][rows][av].min()) >= 15,
          "제외 %d / %d" % ((~av).sum(), len(rows)))

    # 4) frame 간격이 정확히 0.5초 (ego2global 교차검증은 probe 에서 확인)
    check("HIS_STRIDE=5 이고 his 간격 0.1s -> 0.5s", C.HIS_STRIDE == 5)

    # 5) temporal forward 가 n_hist=0 경로와 동일 표본을 쓰는가 (항등 변환 주입)
    m = TrainableSparseScoreDrive(C.BANK_A0, logit_norm=True, n_hist=1).to(dev).eval()
    ds = C.SparseFrameDataset(arr, rows[av][:2], n_hist=1)
    b0, b1 = ds[0], ds[1]
    img = torch.stack([b0["img"], b1["img"]]).to(dev)
    l2ib = l2i.to(dev).unsqueeze(0).expand(2, -1, -1, -1)
    ident = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(2, 1, 4, 4)
    with torch.no_grad():
        f_now, v_now = m.sample_candidate_features(
            *(lambda lv, p4: (lv,))(*m.encode_images(img)), l2ib, (432, 768))
        l2i_h = torch.einsum("bcij,bkjm->bkcim", l2ib, ident)
        f_id, v_id = m.sample_candidate_features(
            *(lambda lv, p4: (lv,))(*m.encode_images(img)), l2i_h[:, 0], (432, 768))
    check("항등 hist_T 주입 시 표본이 현재와 bitwise 동일",
          torch.equal(f_now, f_id) and torch.equal(v_now, v_id),
          "max diff %.2e" % float((f_now - f_id).abs().max()))

    # 6) goal/cmd/status 미입력 (시그니처 감사)
    import inspect
    sig = inspect.signature(m.forward_logits_temporal).parameters
    bad = [k for k in sig if any(x in k.lower()
                                 for x in ("goal", "cmd", "command", "status",
                                           "speed", "vel", "pose", "can_bus"))]
    check("temporal forward 시그니처에 goal/cmd/status/pose 없음", not bad,
          str(list(sig)))

    # 7) 실제 temporal forward 동작 + 과거가 실제로 다른 픽셀을 본다
    ds3 = C.SparseFrameDataset(arr, rows[av][:2], n_hist=3)
    a0, a1 = ds3[0], ds3[1]
    ih = torch.stack([a0["img_hist"], a1["img_hist"]]).to(dev)
    hT = torch.stack([a0["hist_T"], a1["hist_T"]]).to(dev)
    m3 = TrainableSparseScoreDrive(C.BANK_A0, logit_norm=True, n_hist=3).to(dev).eval()
    with torch.no_grad():
        lg = m3.forward_logits_temporal(img, ih, l2ib, hT)
    check("temporal forward 동작", lg.shape == (2, 1024) and
          bool(torch.isfinite(lg).all()), str(tuple(lg.shape)))
    # 과거 정렬이 실제로 좌표를 바꾸는지
    diff = float((hT[:, 0] - torch.eye(4, device=dev)).abs().max())
    check("hist_T 가 항등이 아님 (정렬이 실제로 일어남)", diff > 0.05,
          "max|T-I|=%.3f" % diff)

    print("\n%s" % ("ALL TEMPORAL GATES PASS" if ok_all else "SOME GATES FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
