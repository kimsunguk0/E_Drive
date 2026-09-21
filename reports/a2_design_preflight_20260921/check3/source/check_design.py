"""Bounded train-fixture checks. Never calls optimizer.step or DEV evaluator."""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
for path in (ROOT, ROOT/'scripts', ROOT/'experiments/a2_progress_fourarm_20260921', HERE):
    sys.path.insert(0, str(path))
import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate
from design import (SplitReadPlanner, LaneGeometryHead, GEOMETRY_POLICY,
                    geometry_targets, geometry_loss, geometry_normalizers)
from arm_model import load_export, FourArmModel
from verify import datasets, inputs, differences, KEYS
from motiondrive_v2_training import (to_device, tensor_state_sha256, compute_loss,
    LossWeights, build_loss_normalizers, set_training_mode)
from train_fourarm import length_wrapper
from models.motiondrive_v2.config import MotionDriveV2Config
from motiondrive_v2_data import grid_centers, MEAN, STD
from build_scene_supervision_v2 import read_scene_metadata, rasterize_lanes, camera_visible, transform_points

PARENT = ROOT/'work_dirs/a2_progress_fourarm_20260921/P-CTRL-s1/ckpt_step3426.pth'
PARENT_SHA = '5021300b0d876a6b59b2137d78fd96ab21b80bfed7141ebc1248e5c8bc05a37a'
INDICES = [0, 30000, 60000]  # fixed before reading errors; train only
SPEC = {'schema': 1, 'kind': 'untrained_splitread_design_check', 'parent_sha256': PARENT_SHA,
        'decoder_shared': True, 'length_read_and_head_independent': True,
        'heading_read_and_head_independent': True, 'decoded_skip_retained': True}
SPEC_BYTES = list(hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).digest())


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def build_split(base):
    rng = torch.get_rng_state().clone()
    # Whole-model deepcopy retains Python hook closures bound to the old A2
    # instance. Reconstruct the common graph so its hooks bind to the new owner.
    with torch.random.fork_rng(devices=[]):
        model = FourArmModel(copy.deepcopy(base.config), execution_config=copy.deepcopy(base.execution_config))
        model.load_state_dict(base.state_dict(), strict=True)
        model.planner = SplitReadPlanner(model.planner)
    assert torch.equal(rng, torch.get_rng_state())
    model.register_buffer('splitread_design_signature', torch.tensor(SPEC_BYTES, dtype=torch.uint8))
    return model


def reconstruct(payload):
    if payload['design'] != SPEC:
        raise ValueError('Wrong prototype graph declaration')
    state = payload['model']
    if not torch.equal(state['splitread_design_signature'].cpu(), torch.tensor(SPEC_BYTES, dtype=torch.uint8)):
        raise ValueError('Wrong prototype graph signature')
    manifest = payload['base_manifest']
    base = FourArmModel(MotionDriveV2Config(**manifest['model_config']),
                        execution_config=manifest['execution_config'])
    model = build_split(base)
    model.load_state_dict(state, strict=True)
    return model


def geometric_synthetic():
    p = GEOMETRY_POLICY
    def target(lines, xy):
        x = np.asarray(xy, float).reshape(-1, 1, 2)
        return geometry_targets(lines, x, np.ones(x.shape[:-1], bool))
    cases = {}
    a = target([[[0, 0], [10, 0]]], [[3.4, 1.2]])
    assert np.allclose(a['offset'][0, 0], [0, -1.2]) and a['offset_valid'].all() and a['axis_valid'].all()
    assert np.allclose(a['axis'][0, 0], [1, 0])
    cases['continuous_segment_not_vertex'] = True
    a = target([[[-10, -1], [10, -1]], [[-10, 1], [10, 1]]], [[0, 0]])
    assert not a['offset_valid'].any() and a['axis_valid'].all()
    cases['parallel_equal_distance_offset_off_axis_on'] = True
    a = target([[[-10, 0], [10, 0]], [[0, -10], [0, 10]]], [[0, 0]])
    assert a['offset_valid'].all() and not a['axis_valid'].any()
    cases['crossing_same_point_offset_on_axis_off'] = True
    a = target([[[0, 0], [0, 0]], [[0, 0], [10, 0]]], [[3, 5]])
    assert not a['offset_valid'].any() and not a['axis_valid'].any()
    cases['degenerate_and_outside_support_safe'] = True
    a = target([], [[0, 0]])
    assert not a['offset_valid'].any() and not a['axis_valid'].any()
    cases['empty_annotation_unknown'] = True
    # A weak bend with a common vertex: orientation must not depend on line order.
    line = np.array([[-2, 0], [0, 0], [2, .2]])
    a, b = target([line], [[0, 0]]), target([line[::-1]], [[0, 0]])
    assert np.array_equal(a['offset_valid'], b['offset_valid']) and np.allclose(a['axis'], b['axis'])
    cases['exact_tie_polyline_reversal_invariant'] = True
    crossing = []
    for angle in np.deg2rad([0., 10., 20.]):
        v = np.array([np.cos(angle), np.sin(angle)])
        crossing.append(np.stack([-v, v]))
    for order in itertools.permutations(range(3)):
        a = target([crossing[i] for i in order], [[0, 0]])
        assert a['offset_valid'].all() and not a['axis_valid'].any(), order
    cases['three_way_crossing_all_six_orders_invariant'] = True
    return cases


def prepare_geometry(tr, examples, report_dir):
    grids = grid_centers()
    out, records, overlays = [], [], []
    for i, item in zip(INDICES, examples):
        scene, frame = item['scenario'], int(item['frame'])
        meta = json.loads((tr.supervision_root/(scene+'.json')).read_text())
        paths = [Path(p) for p in meta['sources'] if p.endswith('/annotation/map.parquet')]
        assert len(paths) == 1
        source = paths[0].parent.parent
        started = time.monotonic()
        frames, _, poses, _, lines, schema = read_scene_metadata(source)
        pos = {int(f): j for j, f in enumerate(frames)}[frame]
        w2e = np.linalg.inv(poses[pos])
        local = [transform_points(line, w2e) for line in lines]
        visible = camera_visible(item['lidar2img'].numpy())
        old_target, old_valid = rasterize_lanes(lines, w2e, visible)
        assert np.array_equal(old_target, item['lane_target'].numpy()), scene
        assert np.array_equal(old_valid, item['lane_valid'].numpy()), scene
        # No ego future is used by target generation.
        result = geometry_targets(local, grids, old_valid[0])
        inverse = geometry_targets([x[::-1] for x in reversed(local)], grids, old_valid[0])
        mirror_lines = [x*np.asarray([1, -1, 1]) for x in local]
        flipped = geometry_targets(mirror_lines, grids, old_valid[0, :, ::-1])
        flip_errors, order_errors = {}, {}
        for key in ('offset', 'axis'):
            mask = key+'_valid'
            assert np.array_equal(result[mask], inverse[mask]), (scene, mask, 'reversal')
            assert np.array_equal(result[mask][:, ::-1], flipped[mask]), (scene, mask, 'flip')
            desired = result[key][:, ::-1].copy(); desired[..., 1] *= -1
            flip_errors[key] = float(np.max(np.abs(desired-flipped[key])))
            order_errors[key] = float(np.max(np.abs(result[key]-inverse[key])))
            assert flip_errors[key] < 2e-5 and order_errors[key] < 2e-5, (scene, key, flip_errors, order_errors)
        data = {key: torch.from_numpy(result[key].transpose(2, 0, 1).copy()) for key in ('offset', 'axis')}
        data.update({key+'_valid': torch.from_numpy(result[key+'_valid'].copy()) for key in ('offset', 'axis')})
        out.append(data)
        support = old_valid[0] & (result['distance'] <= GEOMETRY_POLICY['radius_m'])
        records.append({'dataset_index': i, 'row': int(item['row']), 'scene': scene, 'frame': frame,
            'source_map': str(paths[0]), 'source_map_sha256': sha_file(paths[0]), 'map_schema': schema,
            'segments_in_ROI': result['segments'], 'original_valid_cells': int(old_valid.sum()),
            'within_4m_supported_cells': int(support.sum()),
            'offset_valid_cells': int(result['offset_valid'].sum()), 'axis_valid_cells': int(result['axis_valid'].sum()),
            'ambiguous_offset_cells': int((support & ~result['offset_valid']).sum()),
            'ambiguous_axis_cells': int((support & ~result['axis_valid']).sum()),
            'old_raster_and_valid_exactly_reproduced': True, 'flip_maxdiff': flip_errors,
            'polyline_order_reversal_maxdiff': order_errors,
            'metadata_and_target_wall_seconds': time.monotonic()-started})
        overlays.append((item, local, result))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(overlays), 2, figsize=(13, 4*len(overlays)))
    for axes_row, (item, local, result) in zip(axes, overlays):
        ax = axes_row[0]
        for line in local:
            ax.plot(line[:, 1], line[:, 0], color='gray', lw=.7)
        mask = result['offset_valid'].copy()
        mask[1::2] = False; mask[:, 1::2] = False
        xy, delta = grids[mask], result['offset'][mask]
        ax.quiver(xy[:, 1], xy[:, 0], delta[:, 1], delta[:, 0], angles='xy', scale_units='xy', scale=1, width=.002)
        ax.set(xlim=(20, -20), ylim=(-10, 60), title=f"{item['scenario']} frame {item['frame']}: nearest segment offsets", xlabel='ego y (m)', ylabel='ego x (m)')
        ax.set_aspect('equal')
        ax = axes_row[1]
        rgb = item['images'][0].numpy().transpose(1, 2, 0)*STD+MEAN
        ax.imshow(np.clip(rgb, 0, 1))
        matrix = item['lidar2img'][0].numpy()
        for line in local:
            dense = []
            for a, b in zip(line[:-1], line[1:]):
                count = min(1000, max(2, int(np.linalg.norm(b-a)/.3)+1))
                dense.append(a[None]+np.linspace(0, 1, count)[:, None]*(b-a)[None])
            if not dense:
                continue
            q = np.concatenate(dense)
            cam = np.c_[q, np.ones(len(q))] @ matrix.T
            valid = cam[:, 2] > .05
            uv = cam[:, :2]/np.maximum(cam[:, 2:3], .05)
            valid &= (uv[:, 0] >= 0)&(uv[:, 0] < 768)&(uv[:, 1] >= 0)&(uv[:, 1] < 432)
            ax.scatter(uv[valid, 0], uv[valid, 1], s=.5, c='cyan', alpha=.6)
        ax.set(xlim=(0, 768), ylim=(432, 0), title='Existing current map projected onto cached front RGB')
    fig.tight_layout(); fig.savefig(report_dir/'train_geometry_overlay.png', dpi=120); plt.close(fig)
    return default_collate(out), records


def loss_checks(geo):
    # Deliberately include NaN invalid targets and uneven mask counts.
    geo = {k: v.clone() for k, v in geo.items()}
    for key in ('offset', 'axis'):
        geo[key][~geo[key+'_valid'][:, None].expand_as(geo[key])] = float('nan')
    counts = geometry_normalizers(geo)
    generator = torch.Generator().manual_seed(202609211)
    p = torch.randn((len(geo['offset']), 4, 64, 48), generator=generator, requires_grad=True)
    full = geometry_loss(p, geo, counts)
    gf, = torch.autograd.grad(full, p)
    q = p.detach().clone().requires_grad_(True)
    micro = sum(geometry_loss(q[i:i+1], {k:v[i:i+1] for k,v in geo.items()}, counts) for i in range(len(q)))
    gm, = torch.autograd.grad(micro, q)
    assert torch.isfinite(full) and abs(float(full-micro)) < 1e-6 and torch.allclose(gf, gm, atol=1e-8, rtol=1e-6)
    empty = {k: v.clone() for k,v in geo.items()}
    for key in ('offset', 'axis'):
        empty[key].fill_(float('nan')); empty[key+'_valid'].zero_()
    z = p.detach().clone().requires_grad_(True)
    zero = geometry_loss(z, empty, geometry_normalizers(empty))
    gz, = torch.autograd.grad(zero, z)
    assert float(zero) == 0 and torch.isfinite(gz).all() and not gz.count_nonzero()
    return {'full_effective_batch_counts': {k:int(v) for k,v in counts.items()},
            'full_minus_micro_loss': float(full-micro), 'max_gradient_difference': float((gf-gm).abs().max()),
            'partial_invalid_NaN_safe': True, 'all_invalid_graph_zero': True}


def norm(grads):
    return sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**.5


def main(args):
    report = ROOT/'reports/a2_design_preflight_20260921'/args.attempt
    runtime = ROOT/'work_dirs/a2_design_preflight_20260921'/args.attempt
    if report.exists():
        raise ValueError('Refuse to overwrite prior attempt')
    report.mkdir(parents=True); runtime.mkdir(parents=True)
    result = {'status': 'running', 'optimizer_updates': 0, 'DEV_evaluator_called': False,
              'scope': 'geometry_only_CPU' if args.geometry_only else 'full_design_preflight',
              'train_indices': INDICES, 'geometry_policy': GEOMETRY_POLICY,
              'parent_checkpoint': str(PARENT), 'parent_sha256': sha_file(PARENT),
              'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def mark(label, data):
        result[label] = data; atomic(report/'checks.json', result)
        print(label, json.dumps(data, ensure_ascii=False), flush=True)
    try:
        assert result['parent_sha256'] == PARENT_SHA
        mark('synthetic_geometry', geometric_synthetic())
        tr, _ = datasets()
        examples = [tr[i] for i in INDICES]
        raw = default_collate(examples)
        geo_cpu, geo_records = prepare_geometry(tr, examples, report)
        mark('actual_train_geometry', geo_records)
        mark('geometry_loss', loss_checks(geo_cpu))
        if args.geometry_only:
            result['status'] = 'passed'; atomic(report/'checks.json', result)
            print('GEOMETRY_CHECKS_COMPLETE', str(report), flush=True)
            return
        device = torch.device('cuda')
        batch = to_device(raw, device); x = inputs(batch); geo = to_device(geo_cpu, device)
        base, parent = load_export(PARENT, 'cpu')
        before = tensor_state_sha256(base.state_dict())
        split = build_split(base)
        delta = sum(p.numel() for p in split.parameters())-sum(p.numel() for p in base.parameters())
        assert delta == 83328
        lp = list(split.planner.length_read.parameters())+list(split.planner.length_head.parameters())
        hp = list(split.planner.heading_read.parameters())+list(split.planner.heading_head.parameters())
        assert not ({p.data_ptr() for p in lp} & {p.data_ptr() for p in hp})
        split_before = tensor_state_sha256(split.state_dict())
        mark('construction', {'extra_parameters': delta, 'branch_storage_independent': True,
                              'CPU_RNG_preserved': True, 'source_model_state_sha256': before})
        base.cuda().eval(); split.cuda().eval()
        parity = {}
        frozen = None
        for precision in ('fp32', 'bf16'):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
                a = base(**x); b = split(**x)
            diff = differences(a, b)
            assert diff['plan_abs'] < 1e-4, diff
            assert max(diff[k] for k in KEYS if k != 'plan_abs') == 0, diff
            parity[precision] = diff
            if precision == 'bf16':
                frozen = {k:v.detach() for k,v in a.items() if isinstance(v,torch.Tensor)}
                expected = {k:b[k].cpu() for k in KEYS}
        mark('actual_checkpoint_initial_parity', {'train_rows': len(examples), 'tolerance_m': 1e-4, 'values': parity})
        del a, b
        # Existing source boundary: no new provided-status/goal -> motion/state path.
        boundary = {}
        for key, amount in [('provided_status5', 5.), ('goal_xy', 10.)]:
            changed = dict(x); changed[key] = x[key]+amount
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = split(**changed)
            d = differences(frozen, out, ('motion_features','motion_pair_features','state_hat','history_hat'))
            assert max(d.values()) == 0, d
            boundary[key] = d
            del out
        mark('input_boundary', boundary)
        planner_args = [frozen[k].float() for k in ('scene_features','motion_features','state_hat','history_hat','motion_pair_features')]
        pred = split.planner(*planner_args)
        dp = torch.diff(torch.cat([torch.zeros_like(pred[:, :1]), pred], 1), dim=1)
        gt = batch['gt_plan'].float(); dg = torch.diff(torch.cat([torch.zeros_like(gt[:, :1]), gt], 1), dim=1)
        length = (dp.norm(dim=-1)-dg.norm(dim=-1)).abs().mean()
        g = torch.autograd.grad(length, lp+hp, retain_graph=True, allow_unused=True)
        n_l, n_h = norm(g[:len(lp)]), norm(g[len(lp):])
        assert n_l > 0 and n_h/max(n_l,1e-12) < 1e-4, (n_l,n_h)
        w = pred.new_tensor([11,11,5,5,2,2])/36
        prefix = ((pred-gt).norm(dim=-1)*w).sum(-1).mean()
        g = torch.autograd.grad(prefix, lp+hp, allow_unused=True)
        n_pl, n_ph = norm(g[:len(lp)]), norm(g[len(lp):])
        assert n_pl > 0 and n_ph > 0
        mark('split_gradient_routes', {'length_only_length_branch_norm': n_l, 'length_only_heading_branch_norm': n_h,
            'length_only_cross_branch_norm_ratio': n_h/n_l, 'PREFIX_length_branch_norm': n_pl,
            'PREFIX_heading_branch_norm': n_ph})
        del pred, dp, length, prefix, g
        # Copied branch VJPs sum to the original shared module at initialization.
        raw_parent = []
        hook = base.planner.xy_head.register_forward_hook(lambda m,a,o:raw_parent.append(o))
        base.planner(*planner_args); hook.remove()
        rp = raw_parent[0]; rs = split.planner.raw_logits(*planner_args)
        coefficient = torch.linspace(-.7, 1.1, rp.numel(), device=device).reshape_as(rp)
        pa = [base.planner.temporal_read.attention.in_proj_weight, base.planner.xy_head[1].weight]
        pb = [split.planner.length_read.attention.in_proj_weight, split.planner.heading_read.attention.in_proj_weight,
              split.planner.length_head[1].weight, split.planner.heading_head[1].weight]
        ga = torch.autograd.grad((rp*coefficient).sum(), pa)
        gb = torch.autograd.grad((rs*coefficient).sum(), pb)
        rel = [float((ga[i]-gb[2*i]-gb[2*i+1]).norm()/ga[i].norm().clamp_min(1e-12)) for i in range(2)]
        assert max(rel) < 1e-4, rel
        mark('initial_shared_gradient_sum', {'read_and_hidden_relative_error': rel})
        del rp, rs, raw_parent, ga, gb, frozen, planner_args
        gc.collect(); torch.cuda.empty_cache()
        # Geometry supervision uses exactly the raster read by existing consumers.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202609212)
            head = LaneGeometryHead().to(device)
        seen = {}
        hooks = [base.scene_encoder.occ_head.register_forward_pre_hook(lambda m,a:seen.update(occ=a[0])),
                 base.scene_encoder.lane_head.register_forward_pre_hook(lambda m,a:seen.update(lane=a[0])),
                 base.planner.register_forward_pre_hook(lambda m,a:seen.update(planner=a[0]))]
        set_training_mode(base, 'fixed')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = base(**x)
        raster = out['scene_features'].transpose(1,2).reshape(len(examples),128,64,48)
        assert seen['occ'] is seen['lane']
        assert torch.equal(raster, seen['occ']) and torch.equal(out['scene_features'], seen['planner'])
        geo_loss = geometry_loss(head(raster), geo, geometry_normalizers(geo))
        common_norm = to_device(build_loss_normalizers(raw), device)
        weights = LossWeights(occupancy=.2,lane=.2,motion=.2,uncertainty=True)
        total, parts = length_wrapper(compute_loss,.25)(out,batch,weights,normalizers=common_norm)
        main_loss = parts['plan_d3']+.25*parts['plan_interval_length']
        names = ['backbone_fpn.layer1.0.conv1.weight','scene_encoder.key_proj.0.weight']
        params = [dict(base.named_parameters())[n] for n in names]
        gg = torch.autograd.grad(.05*geo_loss, params+list(head.parameters()), retain_graph=True, allow_unused=True)
        gp = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() and float(g.norm())>0 for g in gg)
        feature_grads = [{'parameter':n,'lambda_geo_gradient_norm':float(a.norm()),'PREFIX_LEN_gradient_norm':float(b.norm()),
                          'ratio':float(a.norm()/b.norm().clamp_min(1e-12))} for n,a,b in zip(names,gg,gp)]
        mark('lane_shared_training_path', {'same_shared_raster': True, 'lambda_geo_for_probe':.05,
            'geometry_loss':float(geo_loss),'common_loss':float(total),'gradient_samples':feature_grads,
            'all_new_head_parameters_have_nonzero_finite_gradients':True})
        for h in hooks: h.remove()
        del out, raster, seen, geo_loss, main_loss, total, parts, gg, gp, head
        base.eval(); gc.collect(); torch.cuda.empty_cache()
        # Train-only auxiliary has not changed the deployment model or its state.
        assert tensor_state_sha256(base.state_dict()) == before
        assert tensor_state_sha256(split.state_dict()) == split_before
        mark('no_updates', {'base_parameters_and_buffers_unchanged':True,'split_parameters_and_buffers_unchanged':True,
                           'optimizer_created':False,'optimizer_updates':0})
        payload = {'design':SPEC,'model':{k:v.cpu() for k,v in split.state_dict().items()},'base_manifest':parent['manifest']}
        torch.save(payload,runtime/'split_prototype.pth')
        torch.save({'inputs':{k:v.cpu() for k,v in x.items()},'expected':expected},runtime/'fixtures.pth')
        for corruption in ('graph','signature'):
            bad = copy.deepcopy(payload)
            if corruption == 'graph': bad['design']['decoder_shared'] = False
            else: bad['model']['splitread_design_signature'][0] ^= 1
            try: reconstruct(bad)
            except ValueError: pass
            else: raise AssertionError('Wrong graph accepted: '+corruption)
        # Reload in a separate process after freeing current CUDA models.
        del base, split, x, batch, geo, parent, payload, bad, expected
        gc.collect(); torch.cuda.empty_cache()
        command = [sys.executable,str(Path(__file__)), '--reload', str(runtime)]
        subprocess.run(command, check=True)
        mark('fresh_process_reconstruction', json.loads((runtime/'reload.json').read_text()))
        result['status']='passed'; atomic(report/'checks.json', result)
        print('DESIGN_CHECKS_COMPLETE',str(report),flush=True)
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        atomic(report/'checks.json', result)
        raise


def reload_check(path):
    payload = torch.load(path/'split_prototype.pth',map_location='cpu',weights_only=False)
    model = reconstruct(payload).cuda().eval()
    fixture = torch.load(path/'fixtures.pth',map_location='cuda',weights_only=True)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        out = model(**fixture['inputs'])
    diff = differences(out,fixture['expected'])
    assert max(diff.values()) < 1e-4, diff
    atomic(path/'reload.json',{'strict_reconstruction_passed':True,'wrong_graph_and_signature_rejected':True,
                             'max_differences':diff,'trained':False})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='check1');p.add_argument('--reload',type=Path)
    p.add_argument('--geometry-only', action='store_true')
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.reload:reload_check(args.reload)
    else:main(args)
