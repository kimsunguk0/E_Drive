#!/usr/bin/env python3
"""Test A: exact no-goal trunk equivalence against the frozen source snapshot.

Capture is only permitted while the three trunk files are byte-identical to the
frozen tree.  Later checks load the same deterministic checkpoint/input and
compare ``ego_fut_preds`` bit-for-bit with that captured reference.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects
import projects.mmdet3d_plugin  # noqa: F401 - registry side effects


ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path('/home/pm97/workspace/sukim/adcl/freeze/tvad_full330_nogoal_v1/src_snapshot')
ARTIFACT = Path('/tmp/pm97/c0t_v2a_trunk_reference.pt')
CONFIG = ROOT / 'projects/configs/VAD/VAD_etri_c0t_nogoal_control_v3.py'
CHECKPOINT = Path('/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth')
TRUNK_FILES = (
    'projects/mmdet3d_plugin/VAD/VAD_head.py',
    'projects/mmdet3d_plugin/VAD/VAD_transformer.py',
    'projects/mmdet3d_plugin/VAD/modules/spatial_cross_attention.py',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes(require_equal=False):
    out = {}
    for rel in TRUNK_FILES:
        worktree_hash = sha256(ROOT / rel)
        frozen_hash = sha256(FROZEN / rel)
        if require_equal and worktree_hash != frozen_hash:
            raise AssertionError(f'capture refused: {rel} differs from frozen source')
        out[rel] = frozen_hash
    return out


def deterministic_forward(expose, mlvl_override=None):
    torch.manual_seed(20260824)
    cfg = Config.fromfile(str(CONFIG))
    if expose:
        # Enable the full side-channel while keeping the no-goal decoder.
        cfg.model.expose_image_evidence = True
        cfg.model.pts_bbox_head.transformer.expose_image_evidence = True
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, str(CHECKPOINT), map_location='cpu')
    head = model.pts_bbox_head

    # Keep this a real VAD head forward, but make the CPU test small enough to
    # rerun frequently.  The same frozen checkpoint weights and deterministic
    # 10x10 BEV/input are used for capture and every comparison.
    old = head.bev_embedding.weight.detach()[:100].clone()
    head.bev_h = 10
    head.bev_w = 10
    head.bev_embedding = nn.Embedding.from_pretrained(old, freeze=False)
    seen = 0
    for module in head.modules():
        if hasattr(module, 'expose_image_evidence'):
            if bool(module.expose_image_evidence) == expose:
                seen += 1
    if expose and seen == 0:
        raise AssertionError('no expose_image_evidence switch found')

    dataset = build_dataset(cfg.data.train)
    sample = dataset[0]
    queue_metas = sample['img_metas'].data
    current_meta = queue_metas[max(queue_metas)]
    mlvl_feats = ([torch.randn(1, 6, head.embed_dims, 4, 6)]
                  if mlvl_override is None else mlvl_override)

    def value(name):
        return sample[name].data

    model.eval()
    with torch.no_grad():
        head_kwargs = dict(
            ego_his_trajs=value('ego_his_trajs'),
            ego_lcf_feat=value('ego_lcf_feat'),
            ego_fut_goal=value('ego_fut_goal'),
            ego_fut_cmd=value('ego_fut_cmd'))
        if expose:
            head_kwargs['image_content_feats'] = mlvl_feats
        out = head(
            mlvl_feats, [current_meta], prev_bev=None, **head_kwargs)
    return out['ego_fut_preds'].cpu(), [tensor.cpu() for tensor in mlvl_feats]


def capture_from_frozen_runtime():
    """Run a fresh interpreter whose ``projects`` package is the snapshot."""
    env = os.environ.copy()
    old_pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = '{}:{}:{}'.format(FROZEN, ROOT, old_pythonpath)
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()),
         '--emit-frozen-reference'],
        cwd='/tmp', env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-reference', action='store_true')
    parser.add_argument('--emit-frozen-reference', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--expose-image-evidence', action='store_true')
    args = parser.parse_args()

    if args.emit_frozen_reference:
        imported_root = Path(projects.__file__).resolve().parents[1]
        if imported_root != FROZEN:
            raise AssertionError(
                f'expected frozen runtime {FROZEN}, imported {imported_root}')
        output, _ = deterministic_forward(expose=False)
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'ego_fut_preds': output,
            'frozen_hashes': frozen_hashes(require_equal=False),
        }, ARTIFACT)
        print(f'frozen_runtime_captured={ARTIFACT}')
        print('reference_checksum={:.12f}'.format(output.double().sum().item()))
        return

    if args.capture_reference:
        capture_from_frozen_runtime()
        return

    if not ARTIFACT.exists():
        capture_from_frozen_runtime()
    reference = torch.load(ARTIFACT, map_location='cpu', weights_only=True)
    if args.expose_image_evidence:
        # Generate the input through the unchanged construction path first;
        # opt-in parameters must not perturb the deterministic input itself.
        control, mlvl_feats = deterministic_forward(expose=False)
        control_diff = (control - reference['ego_fut_preds']).abs()
        if torch.count_nonzero(control_diff):
            raise AssertionError('control no longer matches frozen reference')
        output, _ = deterministic_forward(
            expose=True, mlvl_override=mlvl_feats)
    else:
        output, _ = deterministic_forward(expose=False)
    diff = (output - reference['ego_fut_preds']).abs()
    print(f'expose_image_evidence={int(args.expose_image_evidence)}')
    print('max_abs_diff={:.12e}'.format(diff.max().item()))
    print('num_nonzero={}'.format(torch.count_nonzero(diff).item()))
    if torch.count_nonzero(diff):
        raise AssertionError('no-goal ego_fut_preds changed')


if __name__ == '__main__':
    main()
