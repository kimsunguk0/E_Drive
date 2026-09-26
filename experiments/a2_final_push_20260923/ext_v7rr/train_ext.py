#!/usr/bin/env python3
"""Second-stage fine-tune of ExtModel (K image-generated 5 s candidates, goal
endpoint selection). See ext_model.py for the compliance shape.

Trunk frozen, planner fine-tuned (organisers: freezing part of one network is
allowed 2-stage training). Training assigns each sample to the candidate whose
5.0 s endpoint is nearest the GT 5.0 s point -- which is exactly the goal
point -- so training and inference use the same selection rule.

--arm twin: parent L-TRAIN3106 (train310), V0 held out -> the measurement.
--arm full: parent L-FULL6 (376 scenes), V0 in-fit -> the submission candidate,
            trained with the identical recipe.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / 'scripts', ROOT / 'experiments/a2_progress_h4_20260920',
          ROOT / 'experiments/md_r0_reset_20260914', ROOT / 'experiments/a2_long_motion_20260922', HERE):
    sys.path.insert(0, str(p))

import train_long as base
from ext_model import ExtModel, NEW_KEYS, K, EXT
from models.motiondrive_v2 import MotionDriveV2Config

PARENTS = {'twin': 'work_dirs/a2_long_motion_20260922/L-TRAIN3106-s1/ckpt_step34326.pth',
           'full': 'work_dirs/a2_long_motion_20260922/L-FULL6-s1/ckpt_step38070.pth',
           'twin_lenw': 'work_dirs/a2_lenw_20260922/L-TRAIN3106-LENW-s1/ckpt_step34326.pth',
           'full_repro': 'work_dirs/a2_repro_20260923/L-FULL6-s1/ckpt_step38070.pth',
           'full_lenw': 'work_dirs/a2_lenw_full_20260923/L-FULL6-LENW-s1/ckpt_step38070.pth'}
PARENT_UPRIGHT_VIEW = ROOT / 'reports/a2_final_push_20260923/views/L34.npz'
FUT5 = np.load('/tmp/pm97/data/etri/ego_cache_5s.npz')['fut5']
W6N = np.array([11, 11, 5, 5, 2, 2]) / 36.0
RUNS = ROOT / 'work_dirs/a2_ext_select_20260923'
REPORT = ROOT / 'reports/a2_ext_select_20260923'


def build(parent, device):
    cp = torch.load(ROOT / parent, map_location='cpu', weights_only=False)
    model = ExtModel(MotionDriveV2Config(**cp['manifest']['model_config']))
    base.mr.rebuild_correlation_fuse(model, 4)
    missing, unexpected = model.load_state_dict(cp['model'], strict=False)
    assert not unexpected and set(missing) == set(NEW_KEYS), (missing, unexpected)
    model.planner.init_ext_from_parent()
    return model.to(device), cp


def inputs(batch, device):
    b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    x = base.mr.model_inputs_with_canvas(base.trainer.model_inputs, b, time_input='nominal')
    x['provided_status5'] = b['provided_status5']
    return x


def planner_modes(model, x):
    """Frozen trunk (no grad), trainable planner: same graph as ExtModel.forward."""
    model._provided_status_context = x['provided_status5']
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            parts = model.forward_parts(x['images'], x['history_images'], x['lidar2img'],
                                        x['history_transforms'], x['time_offsets'], x['goal_xy'],
                                        x.get('motion_current'), x.get('motion_history'))
    finally:
        model._provided_status_context = None
        model._decoded_capture.clear()
    return model.planner(parts['scene_features'], parts['motion_features'], parts['state_hat'],
                         parts['history_hat'], parts['motion_pair_features'])


EXT_W = float(os.environ.get('EXT_EXT_W', '0.3'))
ALL_W = float(os.environ.get('EXT_ALL_W', '0.0'))


def loss_fn(modes, gt5, w6):
    d5 = torch.linalg.norm(modes[:, :, -1] - gt5[:, None, -1], dim=-1)
    a = d5.detach().argmin(1)
    ch = modes[torch.arange(len(a), device=modes.device), a]
    main = (torch.linalg.norm(ch[:, :6] - gt5[:, :6], dim=-1) @ w6).mean()
    ext = torch.linalg.norm(ch[:, 6:] - gt5[:, 6:], dim=-1).mean()
    loss = main + EXT_W * ext
    if ALL_W:
        # every candidate stays a plausible plan, so a wrong pick costs less
        loss = loss + ALL_W * (torch.linalg.norm(modes[:, :, :6] - gt5[:, None, :6], dim=-1) @ w6).mean()
    return loss, a, main


def evaluate(model, loader, device, sessions_by_row, parent_rows=None, parent_d3=None):
    sel, modes_all, gts, rows, picks = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = inputs(batch, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(**x)
            modes_all.append(out['plan_modes'].float().cpu()); sel.append(out['plan_abs'].float().cpu())
            picks.append(out['selected_mode'].cpu()); gts.append(batch['gt_plan'].float())
            rows += [int(r) for r in batch['row']]
    sel, modes_all, gts, picks = (torch.cat(v).numpy() for v in (sel, modes_all, gts, picks))
    rows = np.asarray(rows)
    d3 = np.linalg.norm(sel - gts, axis=-1) @ W6N
    per_mode = np.linalg.norm(modes_all[:, :, :6] - gts[:, None], axis=-1) @ W6N
    res = dict(n=len(rows), PREFIX=float(d3.mean()), oracle=float(per_mode.min(1).mean()),
               per_mode=[float(v) for v in per_mode.mean(0)],
               mode_hist=np.bincount(picks, minlength=K).tolist(),
               ext_5s_err=float(np.linalg.norm(modes_all[np.arange(len(rows)), picks, -1] - FUT5[rows, -1], axis=-1).mean()))
    if parent_rows is not None and len(rows) == len(parent_rows):
        assert np.array_equal(rows, parent_rows)
        diff = d3 - parent_d3
        sess = np.asarray([sessions_by_row[int(r)] for r in rows]); uniq = sorted(set(sess))
        idx = {s: np.flatnonzero(sess == s) for s in uniq}
        rng = np.random.default_rng(0)
        boots = [np.mean(diff[np.concatenate([idx[uniq[j]] for j in r])]) for r in rng.integers(0, len(uniq), (5000, len(uniq)))]
        res.update(parent_upright=float(parent_d3.mean()), delta=float(diff.mean()),
                   ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   sessions_improved=int(sum(diff[idx[s]].mean() < 0 for s in uniq)), sessions=len(uniq))
    return res, dict(rows=rows, sel=sel, modes=modes_all, gt=gts, picks=picks)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--arm', choices=sorted(PARENTS), required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--lr-planner', type=float, default=2e-5)
    ap.add_argument('--lr-new', type=float, default=1e-3)
    ap.add_argument('--evals', type=int, default=4)
    ap.add_argument('--tag', default='s1')
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    full = args.arm.startswith('full')
    name = f'EXT-{args.arm.upper()}-{args.tag}' + ('-smoke' if args.smoke else '')
    run = RUNS / name; run.mkdir(parents=True, exist_ok=True)
    assert not any(run.glob('ckpt_*.pth')), 'Refuse overwrite: ' + str(run)
    REPORT.mkdir(parents=True, exist_ok=True)
    log = open(REPORT / f'{name}.jsonl', 'a')
    def emit(kind, **kw):
        rec = dict(kind=kind, time=time.strftime('%H:%M:%S'), **kw)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()

    model, cp = build(PARENTS[args.arm], device)
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith('planner.'))
    new = [p for n, p in model.named_parameters() if n in NEW_KEYS]
    old = [p for n, p in model.named_parameters() if n.startswith('planner.') and n not in NEW_KEYS]
    opt = torch.optim.AdamW([dict(params=old, lr=args.lr_planner, weight_decay=.01),
                             dict(params=new, lr=args.lr_new, weight_decay=0.)])
    base_lrs = [args.lr_planner, args.lr_new]

    raw_train, raw_tune = base.nominal.raw_datasets(full, 1)
    train_ds = base.wrapped_eval(raw_train, full)
    val_ds = base.wrapped_eval(raw_tune, full)
    stride = 997 if args.smoke else args.stride
    sub = Subset(train_ds, np.arange(0, len(train_ds), stride).tolist())
    loader = DataLoader(sub, batch_size=16, shuffle=True, num_workers=10, pin_memory=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed), persistent_workers=not args.smoke)
    val_sub = Subset(val_ds, list(range(64))) if args.smoke else val_ds
    val_loader = DataLoader(val_sub, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    records = json.loads((ROOT / 'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json').read_text())['records']
    sessions_by_row = {int(r['row']): r['session'] for r in records}
    pv = np.load(PARENT_UPRIGHT_VIEW, allow_pickle=True)
    parent_rows, parent_d3 = (pv['rows'], np.linalg.norm(pv['up'] - pv['gt'], axis=-1) @ W6N) if not full else (None, None)

    per_epoch = len(loader)
    total = 20 if args.smoke else int(round(args.epochs * per_epoch))
    eval_at = sorted(set([0, total] + [int(round(total * i / args.evals)) for i in range(1, args.evals)]))
    emit('plan', env={k: v for k, v in os.environ.items() if k.startswith('EXT_')}, arm=args.arm, parent=PARENTS[args.arm], K=K, ext=EXT, train_rows=len(sub), steps=total,
         eval_at=eval_at, lr_planner=args.lr_planner, lr_new=args.lr_new, V0_in_fit=full,
         selection='argmin_k |candidate_k(5.0 s) - goal|; first six points returned unchanged')
    w6 = torch.tensor(W6N, dtype=torch.float32, device=device)

    def do_eval(step):
        res, arr = evaluate(model, val_loader, device, sessions_by_row,
                            None if args.smoke else parent_rows, parent_d3)
        emit('eval', step=step, V0_in_fit=full, **res)
        return arr

    step = 0; t0 = time.time(); hist = np.zeros(K, int); running = []
    if 0 in eval_at:
        do_eval(0)
    while step < total:
        for batch in loader:
            if step >= total:
                break
            rows = batch['row'].numpy()
            gt5 = torch.from_numpy(FUT5[rows]).to(device)
            assert torch.allclose(gt5[:, :6].cpu(), batch['gt_plan'].float(), atol=1e-4)
            lr_scale = min(1., (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1., step / max(1, total))))
            for g, lr in zip(opt.param_groups, base_lrs):
                g['lr'] = lr * lr_scale
            modes = planner_modes(model, inputs(batch, device))
            loss, a, main = loss_fn(modes, gt5, w6)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
            opt.step(); step += 1
            hist += np.bincount(a.cpu().numpy(), minlength=K); running.append(float(main))
            if step % 50 == 0 or args.smoke:
                emit('train', step=step, of=total, train_d3=float(np.mean(running[-50:])), grad_norm=float(gn),
                     assign_hist=hist.tolist(), sec_per_step=(time.time() - t0) / step)
                hist[:] = 0
            if step in eval_at and step:
                arr = do_eval(step)
                torch.save(dict(model=model.state_dict(), step=step, manifest=dict(cp['manifest'], ext_select=dict(
                    arm=args.arm, parent=PARENTS[args.arm], K=K, ext=EXT, steps=total, V0_in_fit=full,
                    selection='nearest 5.0 s endpoint to goal'))), run / f'ckpt_step{step}.pth')
                np.savez_compressed(run / f'v0_step{step}.npz', **arr)
    emit('done', step=step, seconds=time.time() - t0)


if __name__ == '__main__':
    main()
