#!/usr/bin/env python
"""재현성 감사: 학습 시작 상태(=ep4 로드 직후)의 지문을 찍는다.

각 run 은 load_from=ep4 로 시작하므로, 어느 config 로 찍든 **동일**해야 한다.
서로 다르면 "ep4 가 안 실렸다 / 다른 코드로 읽었다"는 사고다.

출력:
  - ep4 load missing/unexpected (goal_decoder 포함 여부)
  - goal_decoder state_dict sha256 (첫 optimizer step 전 파라미터 지문)
  - 고정 32 앵커(val frame>=30, 결정적)의 logits sha256 + 0-step plain top-1 L2

    python probe_start.py <config> <ckpt> [--gpu N]
"""
import argparse, hashlib, os, sys
import numpy as np
SD = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SD)
import etri_vad_eval as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--ann-file", default="/tmp/pm97/data/etri/pkl/etri_val38_goal.pkl")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.checkpoint = os.path.abspath(args.checkpoint)
    repo = args.config.split("/projects/")[0]
    os.chdir(repo); sys.path.insert(0, repo)
    import importlib, torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate
    torch.manual_seed(0); np.random.seed(0)
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))
    from mmdet3d.datasets import build_dataset
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.pop("samples_per_gpu", None); cfg.data.test.pop("map_ann_file", None)
    ds = build_dataset(cfg.data.test)
    scen, frame, ds_idx, cache_idx = E.build_index("val_idx", None, ds=ds)
    mask = frame >= 30
    order = np.lexsort((frame[mask], scen[mask]))
    sel_rows = np.where(mask)[0][order][:32]     # 고정 32 앵커

    model, info = E.load_model(cfg, args.checkpoint)
    gd_missing = [k for k in info["missing"] if "goal_decoder" in k]
    gd_unexp = [k for k in info["unexpected"] if "goal_decoder" in k]
    # decoder state_dict 지문 (첫 step 전)
    sd = {k: v for k, v in model.state_dict().items()
          if k.startswith("pts_bbox_head.goal_decoder.")}
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().float().numpy().tobytes())
    dec_hash = h.hexdigest()[:16]

    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu]); model.eval()
    gd = model.module.pts_bbox_head.goal_decoder
    anchors = gd.anchors_abs.detach().cpu().numpy().astype(np.float64)
    w = gd.challenge_w.detach().cpu().numpy().astype(np.float64)
    holder = []
    gd.register_forward_hook(lambda m, i, o: holder.append(
        o[1].detach().float().cpu().numpy()[0]))
    sys.path.insert(0, os.path.join(E.ADCL, "scripts"))
    sys.path.insert(0, os.path.join(E.ADCL, "src"))
    from etri_table import ValSet
    v = ValSet(); gt = np.asarray(v.gt, np.float64)
    L, top1 = [], []
    for r in sel_rows:
        E.reset_stream(model.module)
        data = collate([ds[int(ds_idx[r])]], samples_per_gpu=1)
        holder.clear()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        lg = holder[-1]; L.append(lg)
        D = (w * np.linalg.norm(anchors - gt[r][None], axis=2)).sum(1)
        top1.append(D[int(lg.argmax())])
    Larr = np.stack(L)
    logits_hash = hashlib.sha256(np.round(Larr, 4).tobytes()).hexdigest()[:16]
    print("=" * 56)
    print(f"PROBE  cfg={os.path.basename(args.config)}  ckpt={os.path.basename(args.checkpoint)}")
    print(f"  load: missing={len(info['missing'])} unexpected={len(info['unexpected'])} "
          f"dropped={len(info['dropped'])}  | goal_decoder missing={len(gd_missing)} unexpected={len(gd_unexp)}")
    print(f"  decoder_sha={dec_hash}")
    print(f"  logits32_sha={logits_hash}")
    print(f"  0step plain top-1 (32 fixed) = {np.mean(top1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
