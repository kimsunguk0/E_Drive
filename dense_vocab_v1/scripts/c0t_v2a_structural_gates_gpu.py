#!/usr/bin/env python3
"""Tests B-D for C0-T v2a, including the trained v1 negative control."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401


V2_CONFIG = 'projects/configs/VAD/VAD_etri_c0t_v2a_overfit.py'
V2_CKPT = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
V1_CONFIG = 'projects/configs/VAD/VAD_etri_c0t_overfit_v3.py'
V1_CKPT = '/tmp/pm97/ckpt/negctrl/c0t_positional_shortcut_v1_ep12.pth'


def build(config_path, checkpoint):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.eval()
    # GPU 원본 해상도. Codex 컨테이너에 CUDA가 없어 bev_h=10 / 64x96 으로 돌았는데,
    # 구조 게이트는 해상도 무관이지만 **실제 배포 해상도에서도 exact zero인지**는
    # 따로 확인해야 한다 (deformable sampling 경로가 달라진다).
    model = model.cuda()
    return cfg, model


def sample_inputs(cfg):
    dataset = build_dataset(cfg.data.train)
    first = dataset[0]
    # Index 300 is in a different scenario in this 8-scenario, 2400-row set.
    second = dataset[300]

    def current(sample):
        image = sample['img'].data[-1].unsqueeze(0).cuda()   # 원본 448x768 유지
        metas = sample['img_metas'].data
        return image, metas, sample

    image1, metas1, first = current(first)
    image2, _, _ = current(second)
    return image1, image2, metas1, first


def sample_tensor(sample, name):
    t = sample[name].data
    return t.cuda() if hasattr(t, 'cuda') else t


def encode(model, trunk_feats, content_feats, meta, prev_bev=None):
    head = model.pts_bbox_head
    dtype = trunk_feats[0].dtype
    bev_queries = head.bev_embedding.weight.to(dtype)
    bev_mask = torch.zeros(
        (1, head.bev_h, head.bev_w), dtype=dtype,
        device=bev_queries.device)
    bev_pos = head.positional_encoding(bev_mask).to(dtype)
    return head.transformer.get_bev_features(
        trunk_feats, bev_queries, head.bev_h, head.bev_w,
        image_content_feats=content_feats,
        grid_length=(head.real_h / head.bev_h, head.real_w / head.bev_w),
        bev_pos=bev_pos, img_metas=[meta], prev_bev=prev_bev)


def vary_shift(meta):
    shifted = copy.deepcopy(meta)
    shifted['can_bus'] = shifted['can_bus'].copy()
    shifted['can_bus'][0] += 4.25
    shifted['can_bus'][1] -= 1.75
    shifted['can_bus'][-2] += 0.35
    shifted['can_bus'][-1] += 17.0
    return shifted


def maxdiff(a, b):
    return (a - b).abs().max().item()


def main():
    torch.manual_seed(20260824)
    cfg, model = build(V2_CONFIG, V2_CKPT)
    image1, image2, metas, sample = sample_inputs(cfg)
    meta = metas[max(metas)]
    goal1 = sample_tensor(sample, 'ego_fut_goal').reshape(1, 2)
    goal2 = goal1 + goal1.new_tensor([[17.0, -9.0]])
    cmd = sample_tensor(sample, 'ego_fut_cmd').reshape(1, 3)
    decoder = model.pts_bbox_head.goal_decoder

    with torch.no_grad():
        trunk1, content1 = model.extract_feat(image1, img_metas=[meta])
        _, content2 = model.extract_feat(image2, img_metas=[meta])
        wired_out = model.pts_bbox_head(
            trunk1, [meta], prev_bev=None,
            image_content_feats=content1,
            ego_his_trajs=sample_tensor(sample, 'ego_his_trajs'),
            ego_lcf_feat=sample_tensor(sample, 'ego_lcf_feat'),
            ego_fut_goal=goal1,
            ego_fut_cmd=cmd)
        wired_key = wired_out['bev_embed']
        wired_value = model.pts_bbox_head.transformer.get_image_evidence()
        wired_direct = decoder(wired_key, wired_value, goal1, cmd)
        wiring_diff = maxdiff(wired_out['ego_fut_preds'][:, 0], wired_direct)
        bev_key = encode(model, trunk1, content1, meta)
        evidence1 = model.pts_bbox_head.transformer.get_image_evidence().clone()
        # Only V changes: trunk, geometry, goal, command, and K stay fixed.
        encode(model, trunk1, content2, meta)
        evidence2 = model.pts_bbox_head.transformer.get_image_evidence().clone()

        constant = evidence1.mean(dim=1, keepdim=True).expand_as(evidence1)
        t6 = maxdiff(
            decoder(bev_key, constant, goal1, cmd),
            decoder(bev_key, constant, goal2, cmd))

        zero_value = torch.zeros_like(evidence1)
        t6_double = maxdiff(
            decoder(bev_key, zero_value, goal1, cmd),
            decoder(bev_key, zero_value, goal2, cmd))

        shifted_key = encode(
            model, trunk1, [torch.zeros_like(x) for x in content1],
            vary_shift(meta))
        shift_gate = maxdiff(
            decoder(bev_key, zero_value, goal1, cmd),
            decoder(shifted_key, zero_value, goal2, cmd))

        t8 = maxdiff(
            decoder(bev_key, evidence1, goal1, cmd),
            decoder(bev_key, evidence2, goal1, cmd))
        evidence_replace_diff = maxdiff(evidence1, evidence2)

        # T6': every current/past camera tensor is exactly zero.  This runs
        # the actual biased backbone/FPN; only the separate content stream is
        # required to be zero.
        zeros = torch.zeros(1, 7, 6, 3, image1.shape[-2], image1.shape[-1],
                            device=image1.device)
        prev = None
        zero_content_max = 0.0
        for index in range(7):
            trunk_zero, content_zero = model.extract_feat(
                zeros[:, index], img_metas=[metas[index]])
            zero_content_max = max(
                zero_content_max,
                max(x.abs().max().item() for x in content_zero))
            prev = encode(
                model, trunk_zero, content_zero, metas[index], prev_bev=prev)
        zero_evidence = model.pts_bbox_head.transformer.get_image_evidence()
        zero_evidence_max = zero_evidence.abs().max().item()
        t6_prime = maxdiff(
            decoder(prev, zero_evidence, goal1, cmd),
            decoder(prev, zero_evidence, goal2, cmd))

    print('v2a.T6_constant_value_goal_diff={:.12e}'.format(t6))
    print('v2a.head_wiring_diff={:.12e}'.format(wiring_diff))
    print('v2a.zero_image_content_max={:.12e}'.format(zero_content_max))
    print('v2a.zero_image_evidence_max={:.12e}'.format(zero_evidence_max))
    print('v2a.T6_prime_all_zero_images_goal_diff={:.12e}'.format(t6_prime))
    print('v2a.T6_double_prime_zero_content_goal_diff={:.12e}'.format(t6_double))
    print('v2a.shift_zero_content_goal_and_shift_diff={:.12e}'.format(shift_gate))
    print('v2a.T8_content_replace_output_diff={:.12e}'.format(t8))
    print('v2a.T8_evidence_replace_diff={:.12e}'.format(evidence_replace_diff))

    assert t6 <= 1e-6
    assert wiring_diff == 0.0
    assert zero_content_max == 0.0
    assert zero_evidence_max == 0.0
    assert t6_prime == 0.0
    assert t6_double == 0.0
    assert shift_gate == 0.0
    assert t8 > 1e-6 and evidence_replace_diff > 1e-6

    # Preserved trained v1 is the adversarial negative control for the gates.
    torch.manual_seed(20260824)
    v1_cfg, v1 = build(V1_CONFIG, V1_CKPT)
    _, _, v1_metas, v1_sample = sample_inputs(v1_cfg)
    v1_meta = v1_metas[max(v1_metas)]
    v1_goal1 = sample_tensor(v1_sample, 'ego_fut_goal').reshape(1, 2)
    v1_goal2 = v1_goal1 + v1_goal1.new_tensor([[17.0, -9.0]])
    v1_cmd = sample_tensor(v1_sample, 'ego_fut_cmd').reshape(1, 3)
    v1_decoder = v1.pts_bbox_head.goal_decoder

    with torch.no_grad():
        constant_bev = torch.randn(1, 100, v1.pts_bbox_head.embed_dims,
                                   device=v1_goal1.device)
        constant_bev[:] = constant_bev[:, :1]
        v1_t6 = maxdiff(
            v1_decoder(constant_bev, v1_goal1, v1_cmd),
            v1_decoder(constant_bev, v1_goal2, v1_cmd))

        zeros = torch.zeros(1, 7, 6, 3, image1.shape[-2], image1.shape[-1],
                            device=image1.device)
        prev = None
        for index in range(7):
            trunk_zero = v1.extract_feat(
                zeros[:, index], img_metas=[v1_metas[index]])
            prev = encode(
                v1, trunk_zero, None, v1_metas[index], prev_bev=prev)
        v1_t6_prime = maxdiff(
            v1_decoder(prev, v1_goal1, v1_cmd),
            v1_decoder(prev, v1_goal2, v1_cmd))
        # Independent current-only gate: remove backbone image content while
        # preserving learned BEV queries/position/geometry, with no temporal
        # BEV.  v1 has no separate V tensor to zero directly.
        current_only_zero_bev = encode(
            v1, trunk_zero, None, v1_metas[6], prev_bev=None)
        v1_t6_double = maxdiff(
            v1_decoder(current_only_zero_bev, v1_goal1, v1_cmd),
            v1_decoder(current_only_zero_bev, v1_goal2, v1_cmd))

    print('v1.T6_constant_bev_goal_diff={:.12e}'.format(v1_t6))
    print('v1.T6_prime_all_zero_images_goal_diff={:.12e}'.format(v1_t6_prime))
    print('v1.T6_double_prime_zero_image_content_goal_diff={:.12e}'.format(
        v1_t6_double))
    assert v1_t6 <= 1e-6
    assert v1_t6_prime > 1e-6
    assert v1_t6_double > 1e-6


if __name__ == '__main__':
    main()
