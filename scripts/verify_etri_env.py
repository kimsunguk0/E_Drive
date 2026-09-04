#!/usr/bin/env python
"""주최측 Docker 환경(회수본)이 우리 H200에서 실제로 도는지 검증.

`import torch`가 되는 것만으로는 부족하다. 이 대회에서 실제로 깨질 수 있는 지점 세
군데를 직접 때린다.

1. **버전 assert** — torch 2.7.1 / mmcv 1.4.0 / mmdet 2.14.0 / mmdet3d 0.17.1 /
   numpy 1.26.4. 하나라도 다르면 우리가 손댄 것이므로 즉시 실패시킨다.
2. **MultiScaleDeformableAttnFunction의 CUDA forward + backward를 1회** — mmcv의
   컴파일된 `_ext.so`를 실제로 호출하는 유일한 확실한 방법이다. sm_90 cubin이 들어
   있다는 정적 확인(cuobjdump)과, 그게 이 드라이버에서 실행되는지는 별개다.
   backward까지 도는지 봐야 한다 -- VAD 학습에서 쓰이는 경로가 그쪽이다.
3. **인터프리터 자기완결성** — venv는 원래 `/opt/uv-python/...`을 심볼릭 링크로
   가리켰다. 이미지 밖에서는 그 경로가 없으므로 시스템 python으로 조용히 흘러가
   "되는 것처럼 보이다가" 다른 버전으로 돌 수 있다. sys.prefix/sys.executable이
   회수 경로 안을 가리키는지, site-packages가 이미지 것인지 확인한다.

    env_etri_extracted/venv_dev/bin/python scripts/verify_etri_env.py
"""
import os
import sys

EXPECT = {"torch": "2.7.1", "torchvision": "0.22.1", "mmcv": "1.4.0",
          "mmdet": "2.14.0", "mmseg": "0.14.1", "mmdet3d": "0.17.1",
          "numpy": "1.26.4"}
ROOT = "/home/pm97/workspace/sukim/adcl/env_etri_extracted"

fails = []


def check(cond, label, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(label)
    return cond


def main():
    print("=== 1. 인터프리터 자기완결성 ===")
    print(f"  sys.executable = {sys.executable}")
    print(f"  sys.prefix     = {sys.prefix}")
    print(f"  version        = {sys.version.split()[0]}")
    check(sys.version.split()[0] == "3.10.20", "python 3.10.20")
    # 경로를 하드코딩하지 않는다. 이 스크립트는 임의의 복구 경로에서 돌아야 하고,
    # 첫 판은 ROOT를 박아둬서 리허설이 멀쩡한데도 실패로 찍혔다.
    # 실제로 확인해야 하는 것은 두 가지뿐이다:
    #   (1) 인터프리터가 시스템 python이 아니다  (2) site-packages가 이 venv 것이다
    real = os.path.realpath(sys.executable)
    check(not real.startswith(("/usr/bin", "/usr/local/bin", "/bin")),
          "시스템 python이 아님", real)
    import numpy
    check(os.path.realpath(numpy.__file__).startswith(
        os.path.realpath(sys.prefix)), "site-packages가 이 venv 것",
        os.path.dirname(numpy.__file__))

    print("\n=== 2. 버전 assert ===")
    import importlib
    for mod, want in EXPECT.items():
        try:
            m = importlib.import_module(mod)
            got = getattr(m, "__version__", "?").split("+")[0]
            check(got == want, f"{mod} == {want}", f"실제 {got}")
        except Exception as e:                        # noqa: BLE001
            check(False, f"{mod} import", str(e)[:90])

    print("\n=== 3. CUDA / 디바이스 ===")
    import torch
    check(torch.cuda.is_available(), "torch.cuda.is_available()")
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        print(f"  device   = {torch.cuda.get_device_name(0)}")
        print(f"  capability = sm_{cc[0]}{cc[1]}   torch CUDA {torch.version.cuda}")
        check(cc[0] >= 9, "sm_90 이상 (H200)", f"sm_{cc[0]}{cc[1]}")

    print("\n=== 4. MultiScaleDeformableAttn CUDA forward + backward ===")
    try:
        from mmcv.ops.multi_scale_deform_attn import \
            MultiScaleDeformableAttnFunction
        dev = torch.device("cuda")
        # VAD가 쓰는 것과 같은 형상 축소판. 작게 잡아 VRAM 2GB 이하로 유지한다.
        bs, nq, nh, ch, npt = 2, 64, 8, 32, 4
        shapes = torch.tensor([[16, 16], [8, 8]], dtype=torch.long, device=dev)
        start = torch.tensor([0, 256], dtype=torch.long, device=dev)
        nkey = int((shapes[:, 0] * shapes[:, 1]).sum())
        value = torch.rand(bs, nkey, nh, ch, device=dev,
                           dtype=torch.float32, requires_grad=True)
        loc = torch.rand(bs, nq, nh, len(shapes), npt, 2, device=dev,
                         dtype=torch.float32, requires_grad=True)
        wts = torch.rand(bs, nq, nh, len(shapes), npt, device=dev,
                         dtype=torch.float32, requires_grad=True)
        wts = (wts / wts.sum(-1, keepdim=True).sum(-2, keepdim=True)).detach()
        wts.requires_grad_(True)
        out = MultiScaleDeformableAttnFunction.apply(value, shapes, start, loc,
                                                     wts, 64)
        check(out.shape == (bs, nq, nh * ch), "forward 형상",
              f"{tuple(out.shape)}")
        check(torch.isfinite(out).all().item(), "forward 유한값")
        out.sum().backward()
        for nm, t in (("value", value), ("loc", loc), ("weights", wts)):
            g = t.grad
            check(g is not None and torch.isfinite(g).all().item()
                  and g.abs().sum().item() > 0, f"backward grad {nm}",
                  "" if g is None else f"|g|={g.abs().sum().item():.4g}")
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        print(f"  peak VRAM = {peak:.1f} MiB")
        check(peak < 2048, "VRAM 2GB 이하", f"{peak:.1f} MiB")
    except Exception as e:                            # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(False, "MultiScaleDeformableAttn", str(e)[:120])

    print("\n=== 결과 ===")
    if fails:
        print(f"실패 {len(fails)}건: " + ", ".join(fails))
        return 1
    print("전부 통과 — 주최측 환경이 이 호스트에서 그대로 동작한다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
