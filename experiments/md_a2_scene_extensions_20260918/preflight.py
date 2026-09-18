"""Actual-image contracts and gradients for the two independent scene arms."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_scene_extension import nominal, legacy, REPORT
sys.path.insert(0, str(HERE))
from scene_extensions import SceneExtensionModel, SIDE, QREFINE, SIDE_CAMERA_IDS, SIDE_POSE_INDICES, EXTRA_PREFIX
from side_data import SideSceneDataset, SIDE_KEY, SIDE_FLIP, wrap_side_flip
from a2_model import A2NominalModel
from nominal_data import NominalStatusDataset
from models.motiondrive_v2 import MotionDriveV2Config
from models.motiondrive_v2.scene_encoder import project_scene_points
from motiondrive_v2_training import tensor_state_sha256, model_inputs, to_device
from motiondrive_v2_flip_augment import flip_item
import train_motiondrive_v2 as trainer

ROOT = nominal.ROOT
mr = legacy.mr
CHECK_KEYS = ('scene_features', 'occ_logits', 'lane_logits', 'plan_abs',
              'motion_features', 'state_hat', 'history_hat')


def make_model(arm, common):
    trainer.seed_all(1)
    cfg = MotionDriveV2Config(**trainer.initialization_configuration(common['manifest'],
        goal_on=1, state_on=1, explicit_arch='resnet50', cross_cell_goal_mode='zero', history_contract='control'))
    cfg.motion_input_mode = 'low_feature'
    model = (A2NominalModel(cfg, arm=arm) if arm == 'A2-BASE-NOM'
             else SceneExtensionModel(cfg, arm=arm))
    mr.rebuild_correlation_fuse(model, 4)
    state = {k: v for k, v in common['model'].items() if not k.startswith('motion_encoder.correlation_fuse.0.')}
    missing = model.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys
    shared = {k: v for k, v in model.state_dict().items() if not k.startswith(EXTRA_PREFIX)}
    assert tensor_state_sha256(shared) == nominal.BASE_INITIAL_SHA
    return model.eval().cuda()


def inputs(batch, side=False):
    result = mr.model_inputs_with_canvas(model_inputs, batch, time_input='nominal')
    result['provided_status5'] = batch['provided_status5']
    if side:
        result[SIDE_KEY] = batch[SIDE_KEY]
    return result


def differences(a, b, keys=CHECK_KEYS):
    return {k: float((a[k].float() - b[k].float()).abs().max()) for k in keys}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise ValueError('Preflight uses only physical GPU3')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    REPORT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    train, tune = nominal.raw_datasets(False, 1)
    # All required historical files, without decoding every training image.
    jobs = []
    for dataset in (train, tune):
        for scene in sorted(set(dataset.scene_names[dataset.rows])):
            frame = dataset.arr['frame'][dataset.rows[dataset.scene_names[dataset.rows] == scene]]
            wanted = {int(f) - off for f in frame for off in (1, 5)}
            for cam in ('camera_front_left', 'camera_front_right'):
                jobs.append((dataset.image_root / scene / cam, wanted))
    def verify_files(job):
        folder, wanted = job
        existing = {int(p.stem) for p in folder.glob('*.jpg')}
        missing = sorted(wanted - existing)
        if missing:
            raise ValueError(f'Missing SIDE images in {folder}: {missing[:5]}')
        return len(wanted)
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = sum(pool.map(verify_files, jobs))
    data = NominalStatusDataset(SideSceneDataset(mr.MotionCanvasDataset(tune, 'native')))
    item = data[0]
    original = NominalStatusDataset(mr.MotionCanvasDataset(tune, 'native'))[0]
    assert item[SIDE_KEY].shape == (4, 3, 216, 384)
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(item[key], value), key
    flip = wrap_side_flip(mr.wrap_flip_item(flip_item))
    flipped = flip(item, 768, 384)
    twice = flip(flipped, 768, 384)
    assert torch.equal(flipped[SIDE_KEY], item[SIDE_KEY][list(SIDE_FLIP)].flip(-1))
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            assert torch.allclose(twice[key], value, atol=1e-4, rtol=0) if value.is_floating_point() else torch.equal(twice[key], value), key
    # Check actual source requests and same-camera photometric parameters.
    tune.augment = True
    tune.set_epoch(3)
    calls = []
    old_image = tune._image
    def trace(scene, camera, frame, size, jitter):
        calls.append((camera, frame, size, None if jitter is None else np.asarray(jitter).tolist()))
        return old_image(scene, camera, frame, size, jitter)
    tune._image = trace
    data[0]
    del tune._image
    for camera, offset in [('camera_front_left', 1), ('camera_front_left', 5),
                           ('camera_front_right', 1), ('camera_front_right', 5)]:
        past = next(x for x in calls if x[0] == camera and x[1] == int(item['frame']) - offset)
        current = next(x for x in calls if x[0] == camera and x[1] == int(item['frame']))
        assert past[3] == current[3] and past[2] == (384, 216)
    tune.augment = False
    tune.set_epoch(0)
    batch = to_device(default_collate([data[i] for i in range(8)]), torch.device('cuda:0'))
    x = inputs(batch)
    side_x = inputs(batch, side=True)
    common = torch.load(legacy.INITIALIZER, map_location='cpu', weights_only=False)
    base = make_model('A2-BASE-NOM', common)
    side = make_model(SIDE, common)
    refine = make_model(QREFINE, common)
    del common
    result = {'status': 'running', 'gpu': 3, 'checked_historical_image_files': checked,
        'data_rows': {'train': len(train), 'V0': len(tune)},
        'side_camera_ids': list(SIDE_CAMERA_IDS), 'side_pose_indices': list(SIDE_POSE_INDICES),
        'shared_initial_sha256': nominal.BASE_INITIAL_SHA,
        'parameter_counts': {name: sum(p.numel() for p in model.parameters())
                             for name, model in [('BASE', base), ('SIDE', side), ('QREFINE', refine)]},
        'original_inputs_unchanged': True, 'side_photometric_camera_consistency': True,
        'flip_roundtrip': True, 'parity': {}}
    for mode, amp in [('fp32', False), ('bf16', True)]:
        side.side_scene_enabled = False
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            reference = base(**x)
            side_off = side(**side_x)
            qr = refine(**x)
        result['parity'][mode] = {'SIDE_disabled': differences(reference, side_off),
                                  'QREFINE_zero_init': differences(reference, qr)}
        assert max(result['parity'][mode]['SIDE_disabled'].values()) <= 5e-4
        assert max(result['parity'][mode]['QREFINE_zero_init'].values()) <= 5e-4
        del reference, side_off, qr
    side.side_scene_enabled = True
    # Camera projection and time identity, including mirror permutation.
    cams = (0, 0, 0, 0) + SIDE_CAMERA_IDS
    poses = (0, 1, 2, 3) + SIDE_POSE_INDICES
    matrices, ids, times = side.scene_encoder._view_geometry(x['lidar2img'], x['history_transforms'], x['time_offsets'], cams, poses)
    assert torch.equal(times[:, -4:], -x['time_offsets'][:, list(SIDE_POSE_INDICES)])
    for j, (c, t) in enumerate(zip(SIDE_CAMERA_IDS, SIDE_POSE_INDICES)):
        assert torch.equal(matrices[:, 10+j], x['lidar2img'][:, c] @ x['history_transforms'][:, t])
    fbatch = to_device(default_collate([flipped]), torch.device('cuda:0'))
    fx = inputs(fbatch)
    fm, _, _ = side.scene_encoder._view_geometry(fx['lidar2img'], fx['history_transforms'], fx['time_offsets'], cams, poses)
    points = side.scene_encoder.points
    reflected_points = points.clone(); reflected_points[..., 1] *= -1
    order = [0, 2, 1, 3, 5, 4, 6, 7, 8, 9, 12, 13, 10, 11]
    before, bv = project_scene_points(points, matrices[:1, order], (432, 768))
    after, av = project_scene_points(reflected_points, fm, (432, 768))
    mask = bv & av
    flip_error = max(float((after[..., 0][mask] + before[..., 0][mask]).abs().max()),
                     float((after[..., 1][mask] - before[..., 1][mask]).abs().max()))
    # The existing BASE mask uses [0,W), while pixel-center reflection is
    # u -> W-1-u. Its outer one-pixel strip is therefore not mirror symmetric.
    # Preserve the matched BASE function and verify that ONLY that strip differs.
    projected = torch.einsum('bvij,qhj->bvqhi', matrices[:1, order], points)
    uv = projected[..., :2] / projected[..., 2:3].clamp_min(1e-5)
    u = uv[..., 0]
    rim = ((u > -1.001) & (u < 0.001)) | ((u > 766.999) & (u < 768.001))
    mask_difference = bv ^ av
    assert not bool((mask_difference & ~rim).any()) and flip_error < 3e-5
    result['legacy_one_pixel_frustum_rim_mask_differences'] = int(mask_difference.sum())
    result['frustum_mask_policy'] = 'BASE [0,W) retained; mirror-mask differences restricted to its outer 1px strip'
    result['side_projection_mirror_max_abs'] = flip_error
    _, visible = project_scene_points(points, matrices[:1], (432, 768))
    result['side_visible_point_counts'] = visible[:, -4:].sum((0, 2, 3)).tolist()
    assert min(result['side_visible_point_counts']) > 0
    print('PARITY_AND_GEOMETRY ' + json.dumps(result), flush=True)
    # Planning loss reaches every added observation.
    tiny = {k: v[:2] for k, v in side_x.items()}
    tiny[SIDE_KEY] = tiny[SIDE_KEY].detach().clone().requires_grad_(True)
    weight = torch.tensor([11, 11, 5, 5, 2, 2], device='cuda') / 36
    def plan_loss(output):
        return (torch.linalg.vector_norm(output['plan_abs'] - batch['gt_plan'][:2], dim=-1) * weight).sum(-1).mean()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = plan_loss(side(**tiny))
    loss.backward()
    grad = tiny[SIDE_KEY].grad
    assert grad is not None and torch.isfinite(grad).all()
    norms = grad.transpose(0, 1).flatten(1).norm(dim=1)
    assert bool((norms > 0).all())
    result['side_planning_image_gradient_norms'] = norms.tolist()
    side.zero_grad(set_to_none=True)
    # A zero output projection opens the query-update gradient after its first step.
    tinyq = {k: v[:2] for k, v in x.items()}
    module = refine.scene_encoder.evidence_attention
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = plan_loss(refine(**tinyq))
    loss.backward()
    assert torch.isfinite(module.output.weight.grad).all() and module.output.weight.grad.norm() > 0
    result['QREFINE_initial_output_projection_gradient_norm'] = float(module.output.weight.grad.norm())
    torch.optim.SGD([module.output.weight], lr=1e-3).step()
    refine.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = plan_loss(refine(**tinyq))
    loss.backward()
    result['QREFINE_query_gradients_after_projection_step'] = {}
    for name, param in module.query_update.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.norm() > 0
        result['QREFINE_query_gradients_after_projection_step'][name] = float(param.grad.norm())
    assert module.query_update[0].weight.grad[:, 32:].norm() > 0
    refine.zero_grad(set_to_none=True)
    with torch.no_grad(): module.output.weight.zero_()
    # Trained BASE replay and status/side-image isolation of the motion branch.
    terminal_path = ROOT / 'work_dirs/md_a2_nominal_mh4_20260918/A2-BASE-NOM-s1/ckpt_step20554.pth'
    terminal = torch.load(terminal_path, map_location='cpu', weights_only=False)
    for model in (base, side, refine):
        missing = model.load_state_dict(terminal['model'], strict=False)
        assert not missing.unexpected_keys and all(k.startswith(EXTRA_PREFIX) for k in missing.missing_keys)
    del terminal
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        replay = base(**x)
        qr = refine(**x)
        normal = side(**side_x)
        swap_image = side(**dict(side_x, side_scene_images=side_x[SIDE_KEY].flip(0)))
        swap_status = side(**dict(side_x, provided_status5=side_x['provided_status5'].flip(0)))
        qr_status = refine(**dict(x, provided_status5=x['provided_status5'].flip(0)))
    stored = json.loads(terminal_path.with_name('final_eval.json').read_text())['records']
    expected = torch.tensor([a['pred_abs_xy'] for a in stored[:8]], device='cuda')
    result['BASE_terminal_replay_max_abs'] = float((replay['plan_abs'] - expected).abs().max())
    assert result['BASE_terminal_replay_max_abs'] <= 1e-4
    result['trained_BASE_QREFINE_zero_init_parity'] = differences(replay, qr)
    assert max(result['trained_BASE_QREFINE_zero_init_parity'].values()) <= 5e-4
    motion_keys = ('motion_features', 'state_hat', 'history_hat')
    result['motion_invariance'] = {'SIDE_image_swap': differences(normal, swap_image, motion_keys),
        'SIDE_status_swap': differences(normal, swap_status, motion_keys),
        'QREFINE_status_swap': differences(qr, qr_status, motion_keys)}
    assert all(max(v.values()) == 0 for v in result['motion_invariance'].values())
    result['side_image_swap_scene_max_abs'] = differences(normal, swap_image, ('scene_features',))['scene_features']
    assert result['side_image_swap_scene_max_abs'] > 0
    result.update(status='passed', elapsed_s=time.monotonic()-start,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        training_scope='one temporary QREFINE projection-only SGD step for gradient connectivity; no candidate retained')
    (REPORT / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
