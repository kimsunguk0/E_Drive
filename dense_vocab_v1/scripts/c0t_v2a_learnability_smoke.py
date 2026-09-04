#!/usr/bin/env python3
"""Test E: bounded 150-step C0-T v2a learnability and gradient smoke.

The 150 optimizer steps use cached current-frame encoder outputs from one row
per overfit scenario.  A separate real-image end-to-end backward verifies the
same loss reaches the BEV encoder and image backbone.  This is intentionally
not an MMEngine training run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401


CONFIG = 'projects/configs/VAD/VAD_etri_c0t_v2a_overfit.py'
CHECKPOINT = '/tmp/pm97/ckpt/vad_tiny_etri_bootstrap_v1_seed0.pth'
INDICES = tuple(range(0, 2400, 300))


def grad_l1(parameters):
    return sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in parameters if parameter.grad is not None)


def encode(model, trunk_feats, content_feats, metas):
    head = model.pts_bbox_head
    batch = trunk_feats[0].shape[0]
    dtype = trunk_feats[0].dtype
    bev_queries = head.bev_embedding.weight.to(dtype)
    bev_mask = torch.zeros(
        (batch, head.bev_h, head.bev_w), dtype=dtype,
        device=bev_queries.device)
    bev_pos = head.positional_encoding(bev_mask).to(dtype)
    key = head.transformer.get_bev_features(
        trunk_feats, bev_queries, head.bev_h, head.bev_w,
        image_content_feats=content_feats,
        grid_length=(head.real_h / head.bev_h, head.real_w / head.bev_w),
        bev_pos=bev_pos, img_metas=metas, prev_bev=None)
    value = head.transformer.get_image_evidence()
    return key, value


def metric_loss(head, decoder, key, value, goals, commands, gt, masks):
    increments = decoder(key, value, goals, commands)
    modes = increments.unsqueeze(1).repeat(1, head.ego_fut_mode, 1, 1)
    return head.loss_plan_metric(modes, gt, commands, masks)


def main():
    torch.manual_seed(20260824)
    cfg = Config.fromfile(CONFIG)
    dataset = build_dataset(cfg.data.train)
    rows = [dataset[index] for index in INDICES]
    scene_tokens = [row['img_metas'].data[6]['scene_token'] for row in rows]
    if len(set(scene_tokens)) != 8:
        raise AssertionError(f'expected 8 scenarios, got {scene_tokens}')

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model.eval()
    head = model.pts_bbox_head
    old = head.bev_embedding.weight.detach()[:100].clone()
    head.bev_h = 10
    head.bev_w = 10
    head.bev_embedding = nn.Embedding.from_pretrained(old, freeze=False)
    head.transformer.rotate_center = [5, 5]
    decoder = head.goal_decoder

    images = torch.stack([row['img'].data[-1] for row in rows])
    batch, cams = images.shape[:2]
    images = F.interpolate(
        images.flatten(0, 1), size=(64, 96), mode='bilinear',
        align_corners=False).view(batch, cams, 3, 64, 96)
    metas = [row['img_metas'].data[6] for row in rows]
    goals = torch.cat([row['ego_fut_goal'].data.reshape(1, 2) for row in rows])
    commands = torch.cat([row['ego_fut_cmd'].data.reshape(1, 3) for row in rows])
    gt = torch.cat([row['ego_fut_trajs'].data.reshape(1, 6, 2) for row in rows])
    masks = torch.cat([row['ego_fut_masks'].data.reshape(1, 6) for row in rows])

    with torch.no_grad():
        trunk, content = model.extract_feat(images.clone(), img_metas=metas)
        cached_key, cached_value = encode(model, trunk, content, metas)
        cached_key = cached_key.detach()
        cached_value = cached_value.detach()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=2e-3)
    losses = []
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss = metric_loss(
            head, decoder, cached_key, cached_value,
            goals, commands, gt, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

    # Decoder submodule audit on the cached optimization graph.
    optimizer.zero_grad(set_to_none=True)
    audit_loss = metric_loss(
        head, decoder, cached_key, cached_value,
        goals, commands, gt, masks)
    audit_loss.backward()
    cached_grads = {
        'goal_mlp': grad_l1(decoder.cond_mlp.parameters()),
        'query_embedding': grad_l1(decoder.ts_embed.parameters()),
        'cross_attention': grad_l1(decoder.cross_attn.parameters()),
        'output_layer': grad_l1(decoder.out.parameters()),
    }

    # One end-to-end current-frame probe.  No optimizer step is taken here;
    # it exists only to prove loss connectivity through encoder and backbone.
    model.zero_grad(set_to_none=True)
    one_image = images[:1].clone()
    one_meta = metas[:1]
    trunk, content = model.extract_feat(one_image, img_metas=one_meta)
    key, value = encode(model, trunk, content, one_meta)
    end_to_end_loss = metric_loss(
        head, decoder, key, value, goals[:1], commands[:1],
        gt[:1], masks[:1])
    end_to_end_loss.backward()
    end_to_end_grads = {
        'bev_encoder': grad_l1(head.transformer.encoder.parameters()),
        'image_backbone': grad_l1(model.img_backbone.parameters()),
    }

    print('scenario_count={}'.format(len(set(scene_tokens))))
    print('loss_plan_metric.iter001={:.9f}'.format(losses[0]))
    print('loss_plan_metric.iter050={:.9f}'.format(losses[49]))
    print('loss_plan_metric.iter100={:.9f}'.format(losses[99]))
    print('loss_plan_metric.iter150={:.9f}'.format(losses[149]))
    print('loss_plan_metric.min={:.9f}'.format(min(losses)))
    for name, value in cached_grads.items():
        print('grad_l1.{}={:.12e}'.format(name, value))
    for name, value in end_to_end_grads.items():
        print('grad_l1.{}={:.12e}'.format(name, value))

    assert losses[-1] < losses[0]
    for name, value in {**cached_grads, **end_to_end_grads}.items():
        if not value > 0.0:
            raise AssertionError(f'no gradient reached {name}: {value}')


if __name__ == '__main__':
    main()
