import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.VAD.utils.traj_lr_warmup import get_traj_warmup_loss_weight
from projects.mmdet3d_plugin.VAD.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


@HEADS.register_module()
class VADHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 tot_epoch=None,
                 use_traj_lr_warmup=False,
                 motion_decoder=None,
                 motion_map_decoder=None,
                 use_pe=False,
                 motion_det_score=None,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 ego_his_encoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanAgentDisLoss', loss_weight=0.1),
                 loss_plan_dir=dict(type='PlanMapThetaLoss', loss_weight=0.1),
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 loss_plan_reg_cum=None,   # 누적 planning loss (채점 단위 정렬)
                 loss_plan_metric=None,    # 채점 함수와 동일 (PlanChallengeL2Loss)
                 goal_decoder=None,        # C0-T: goal-conditioned waypoint decoder
                 aux_loss_scale=None,      # det/map/traj loss를 assignment 후에 스케일
                 valid_fut_ts=6,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.motion_map_decoder = motion_map_decoder
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_his_encoder = ego_his_encoder
        self.ego_fut_mode = ego_fut_mode
        self.ego_agent_decoder = ego_agent_decoder
        self.ego_map_decoder = ego_map_decoder
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.goal_decoder_cfg = goal_decoder
        self.aux_loss_scale = aux_loss_scale
        self.valid_fut_ts = valid_fut_ts

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is VADHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight
        
        self.traj_bg_cls_weight = 0

        super(VADHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        
        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
                'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)
        
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)
        self.loss_plan_reg_cum = (build_loss(loss_plan_reg_cum)
                                  if loss_plan_reg_cum is not None else None)
        self.loss_plan_metric = (build_loss(loss_plan_metric)
                                 if loss_plan_metric is not None else None)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        if self.motion_decoder is not None:
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')

        if self.motion_map_decoder is not None:
            self.lane_encoder = LaneNet(256, 128, 3)
            self.motion_map_decoder = build_transformer_layer_sequence(self.motion_map_decoder)
            if self.use_pe:
                self.pos_mlp = nn.Linear(2, self.embed_dims)
        
        if self.ego_his_encoder is not None:
            self.ego_his_encoder = LaneNet(2, self.embed_dims//2, 3)
        else:
            self.ego_query = nn.Embedding(1, self.embed_dims)	

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = build_transformer_layer_sequence(self.ego_agent_decoder)
            if self.use_pe:
                self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)
            if self.use_pe:
                self.ego_map_pos_mlp = nn.Linear(2, self.embed_dims)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims*2 + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims*2
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode*self.fut_ts*2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        # C0-T: goal-conditioned waypoint decoder.
        # `goal_decoder` config를 주면 ego 궤적을 이쪽에서 낸다. 기존 ego_fut_decoder
        # 경로는 그대로 남겨 no-goal 대조군을 같은 코드로 돌릴 수 있게 한다.
        if self.goal_decoder_cfg is not None:
            from .goal_decoder import (
                DenseVocabularyDecoder, GoalWaypointDecoder,
                ImageEvidenceGoalWaypointDecoder)
            from .vocab_decoder import ImageOnlyVocabularyDecoder
            from .anchor_vocab_decoder import AnchorGroundedVocabularyDecoder
            cfg = dict(self.goal_decoder_cfg)
            decoder_type = cfg.pop('type', 'GoalWaypointDecoder')
            cfg.setdefault('embed_dims', self.embed_dims)
            cfg.setdefault('fut_ts', self.fut_ts)
            # ImageOnlyVocabularyDecoder 는 cmd 를 아예 받지 않는다.
            if decoder_type not in ('ImageOnlyVocabularyDecoder',
                                    'AnchorGroundedVocabularyDecoder'):
                cfg.setdefault('cmd_dim', self.ego_fut_mode)
            decoder_classes = {
                'GoalWaypointDecoder': GoalWaypointDecoder,
                'ImageEvidenceGoalWaypointDecoder': ImageEvidenceGoalWaypointDecoder,
                'DenseVocabularyDecoder': DenseVocabularyDecoder,
                'ImageOnlyVocabularyDecoder': ImageOnlyVocabularyDecoder,
                'AnchorGroundedVocabularyDecoder': AnchorGroundedVocabularyDecoder,
            }
            if decoder_type not in decoder_classes:
                raise ValueError(f'unknown goal decoder: {decoder_type}')
            self.goal_decoder = decoder_classes[decoder_type](**cfg)
        else:
            self.goal_decoder = None

        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(self.fut_mode*2*self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.motion_decoder is not None:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        if self.motion_map_decoder is not None:
            for p in self.motion_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.lane_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            if self.use_pe:
                xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        if self.ego_his_encoder is not None:
            for p in self.ego_his_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_agent_decoder is not None:
            for p in self.ego_agent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_map_decoder is not None:
            for p in self.ego_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                image_content_feats=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                ego_fut_goal=None,
                ego_fut_cmd=None,
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                image_content_feats=image_content_feats,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                image_content_feats=image_content_feats,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid() # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord[..., :2].clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)
            
        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # motion_query
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            # [M, B, D], M=A*fut_mode
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                key_padding_mask=invalid_motion_idx)

            if self.motion_map_decoder is not None:
                # map preprocess
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                map_score = map_outputs_classes[-1]
                map_pos = map_outputs_coords_bev[-1]
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    motion_coords, map_query, map_score, map_pos,
                    map_thresh=self.map_thresh, dis_thresh=self.dis_thresh,
                    pe_normalization=self.pe_normalization, use_fix_pad=True)
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

                # position encoding
                if self.use_pe:
                    (num_query, batch) = ca_motion_query.shape[:2] 
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                    motion_pos = self.pos_mlp(motion_pos)
                    map_pos = map_pos.permute(1, 0, 2)
                    map_pos = self.pos_mlp(map_pos)
                else:
                    motion_pos, map_pos = None, None
                
                ca_motion_query = self.motion_map_decoder(
                    query=ca_motion_query,
                    key=map_query,
                    value=map_query,
                    query_pos=motion_pos,
                    key_pos=map_pos,
                    key_padding_mask=key_padding_mask)
            else:
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            batch_size = outputs_coords_bev[-1].shape[0]
            motion_hs = motion_hs.permute(1, 0, 2).unflatten(
                dim=1, sizes=(num_agent, self.fut_mode)
            )
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            )
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        else:
            raise NotImplementedError('Not implement yet')

        outputs_traj = self.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        outputs_traj_class = self.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        (batch, num_agent) = motion_hs.shape[:2]
             
        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        # planning
        (batch, num_agent) = motion_hs.shape[:2]
        if self.ego_his_encoder is not None:
            ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
        else:
            ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        # Interaction
        ego_query = ego_his_feats
        ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)
        agent_conf = outputs_classes[-1]
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        agent_query = self.agent_fus_mlp(agent_query) # [B, A, fut_mode, 2*D] -> [B, A, D]
        agent_pos = outputs_coords_bev[-1]
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)
        # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=agent_pos_emb.permute(1, 0, 2),
            key_padding_mask=agent_mask)

        # ego <-> map interaction
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        map_pos_emb = self.ego_map_pos_mlp(map_pos)
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=map_pos_emb.permute(1, 0, 2),
            key_padding_mask=map_mask)

        if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:                
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:                
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]  

        # Ego prediction
        ego_vocab_logits = None
        ego_vocab_index = None
        if self.goal_decoder is not None:
            # C0-T. goal은 어느 시각 증거를 읽을지만 결정한다. v1은 보존된
            # temporal-BEV negative control이고, v2a는 별도 image evidence V를 쓴다.
            # ego_fut_mode(3) 축은 유지한다 -- loss/평가 코드가 cmd로 slice한다.
            # goal 조건이 이미 command를 포함하므로 세 mode는 같은 값을 쓴다.
            if not getattr(self.goal_decoder, 'uses_condition', True):
                # 규정 준수 계보: goal/cmd 를 planner 에 **넘기지 않는다.**
                # 선택은 모델 밖의 고정 규칙이 logits 의 top-M 을 받아 수행한다.
                # 이 분기에서 g/c 를 만들지조차 않는 것이 의도다 (코드 심사 대비).
                image_evidence = self.transformer.get_image_evidence()
                decoded = self.goal_decoder(
                    bev_embed, image_evidence, batch_size)
            else:
                assert ego_fut_goal is not None, \
                    'goal_decoder를 켰는데 ego_fut_goal이 안 들어왔다 (배관 누락)'
                g = ego_fut_goal.reshape(batch_size, 2)
                if ego_fut_cmd is None:
                    c = g.new_zeros(batch_size, self.ego_fut_mode)
                    c[:, -1] = 1.0
                else:
                    c = ego_fut_cmd.reshape(
                        batch_size, self.ego_fut_mode).to(g.dtype)
                if getattr(self.goal_decoder, 'requires_image_evidence', False):
                    image_evidence = self.transformer.get_image_evidence()
                    decoded = self.goal_decoder(
                        bev_embed, image_evidence, g, c)      # [B, fut_ts, 2]
                else:
                    decoded = self.goal_decoder(bev_embed, g, c)
            if isinstance(decoded, tuple):
                wp, ego_vocab_logits, ego_vocab_index = decoded
            else:
                wp = decoded
            outputs_ego_trajs = wp.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        else:
            outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
            outputs_ego_trajs = outputs_ego_trajs.reshape(
                outputs_ego_trajs.shape[0], self.ego_fut_mode, self.fut_ts, 2)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
            'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
            'map_all_cls_scores': map_outputs_classes,
            'map_all_bbox_preds': map_outputs_coords,
            'map_all_pts_preds': map_outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ego_fut_preds': outputs_ego_trajs,
            'ego_vocab_logits': ego_vocab_logits,
            'ego_vocab_index': ego_vocab_index,
        }

        return outs

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec, self.map_code_size)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        # positive가 없으면 대입할 것이 없다. 빈 텐서끼리도 dtype이 다르면
        # 'Index put requires the source and destination dtypes match'로 죽는다.
        # 330 시나리오에는 gt_names가 완전히 빈 프레임이 623/99,000 있다.
        if pos_inds.numel() > 0:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        # num_gt_bbox == 0 이면 원소가 0개라 view(0, -1)의 -1이 모호해져 RuntimeError가
        # 난다. `filter_empty_gt=False`로 range 안에 박스가 없는 프레임을 학습에 넣으면
        # 바로 터진다(실측). shape을 명시해 0-GT 프레임을 정상 통과시킨다.
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).reshape(
            num_gt_bbox, gt_traj_c)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        # GT가 0개일 때 sampling_result.pos_gt_bboxes는 [0, 4]로 나오는데
        # bbox_targets는 [0, gt_bbox_c(=9)]다. 빈 텐서끼리도 shape가 안 맞으면
        # broadcast 오류가 난다. 대입할 것이 없으면 건너뛴다.
        if pos_inds.numel() > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        # positive가 없으면 대입할 것이 없다. 빈 텐서끼리도 dtype이 다르면
        # 'Index put requires the source and destination dtypes match'로 죽는다.
        # 330 시나리오에는 gt_names가 완전히 빈 프레임이 623/99,000 있다.
        if pos_inds.numel() > 0:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        # in-patch lane 0개 프레임(약 181/99,000)에서는 positive가 없다.
        # 빈 텐서 대입은 dtype/shape가 어긋나 죽고, assigned_shift도 float 빈 텐서가
        # 되어 인덱싱이 IndexError를 낸다. 대입할 것이 없으면 건너뛴다.
        if pos_inds.numel() > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            pts_targets[pos_inds] = gt_shifts_pts[
                sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds,
                      lane_score_preds,
                      agent_preds,
                      agent_fut_preds,
                      agent_score_preds,
                      agent_fut_cls_preds):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        # ★★ loss_weight == 0 인 항은 **그래프를 만들지 않는다**.
        #
        # 이건 최적화가 아니라 정정이다. weight 0 항은 loss에 0을, gradient에도 0을
        # 기여해야 하는데 현재 구현은 `self.loss_weight * f(pred, target)` 처럼
        # **f를 먼저 계산한 뒤 0을 곱한다**. f의 backward가 NaN이면 IEEE754에서
        # `0 * NaN = NaN` 이므로 weight 0 항이 전체 gradient를 죽인다.
        #
        # 실측 (C0-T v2a 330 pilot, iter 622, autograd anomaly detection):
        #     Error detected in Atan2Backward0
        #       VAD_head.py:1237  loss_plan_dir
        #       plan_loss.py:384  loss_bbox = self.loss_weight * plan_map_dir_loss(...)
        #       plan_loss.py:436  traj_yaw = atan2(diff(pred_y), diff(pred_x))
        #   loss는 finite(atan2(0,0)=0)인데 grad만 NaN이었고, `torch.nan_to_num`은
        #   **값**만 고치므로 아무 도움이 되지 않았다. 결과: 139/431 파라미터의 grad가
        #   NaN -> clip_grad_norm_의 total_norm이 NaN -> 전 grad에 NaN 곱 ->
        #   AdamW가 lr=0인 동결 파라미터까지 `0 * NaN = NaN`으로 죽였다.
        #
        # 같은 부류의 함정이 이 파일들에 여럿 있다 (전부 0에서 backward가 0/0):
        #     plan_loss.py:405  linalg.norm(pred[:,-1] - pred[:,0])   시작=끝
        #     plan_loss.py:410,418,431 / 110,130,289  norm(pred - target)  정확히 일치
        # v2a는 goal decoder의 `out.weight`가 std=1e-3 초기화라 초기 예측 증분이
        # 거의 0이어서 이 함정들을 실제로 밟는다. no-goal 계보의 사전학습된
        # `ego_fut_decoder`는 증분이 0이 아니라 99,000 iteration 동안 밟지 않았다.
        #
        # 현재 config는 planning aux 4개가 모두 weight 0이고 `PlanChallengeL2Loss`
        # (weight 1.0) 하나만 활성이다. 그래서 이 가드가 곧 전부를 차단한다.
        # 나중에 weight를 되살릴 때는 `plan_loss.py`의 atan2/norm에 0-근방 가드를
        # 먼저 넣어야 한다.
        zero = ego_fut_preds.new_zeros(())

        def _w(loss_mod):
            return float(getattr(loss_mod, 'loss_weight', 1.0))

        if _w(self.loss_plan_reg) != 0.0:
            loss_plan_l1 = self.loss_plan_reg(
                ego_fut_preds,
                ego_fut_gt,
                loss_plan_l1_weight
            )
        else:
            loss_plan_l1 = zero

        if _w(self.loss_plan_bound) != 0.0:
            loss_plan_bound = self.loss_plan_bound(
                ego_fut_preds[ego_fut_cmd==1],
                lane_preds,
                lane_score_preds,
                weight=ego_fut_masks
            )
        else:
            loss_plan_bound = zero

        if _w(self.loss_plan_col) != 0.0:
            loss_plan_col = self.loss_plan_col(
                ego_fut_preds[ego_fut_cmd==1],
                agent_preds,
                agent_fut_preds,
                agent_score_preds,
                agent_fut_cls_preds,
                weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
            )
        else:
            loss_plan_col = zero

        if _w(self.loss_plan_dir) != 0.0:
            loss_plan_dir = self.loss_plan_dir(
                ego_fut_preds[ego_fut_cmd==1],
                lane_preds,
                lane_score_preds,
                weight=ego_fut_masks
            )
        else:
            loss_plan_dir = zero

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)
        
        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        # ---- 누적 planning loss (기본 꺼짐, config의 loss_plan_reg_cum로 켠다) ----
        #
        # 왜: 베이스라인은 **증분**에 L1을 걸지만 채점은 **누적**이다. 8-scene overfit
        # 실측에서 종방향 증분 오차의 부호가 6스텝 모두 같은 앵커가 75.6%였고,
        # `3초 누적종오차 / 증분종오차합`의 p50이 정확히 1.000이었다 -- 즉 증분 오차가
        # 완전히 같은 방향으로 누적된다. 그래서 증분 L1이 0.62일 때 누적 챌린지 L2가
        # 2.77까지 벌어진다(증분에 같은 식을 적용하면 1.13). 증분 loss는 이 상관을
        # 보지 않으므로 모델이 "매 스텝 조금씩 같은 쪽으로 틀리는" 속도 편향을 고칠
        # 유인이 없다.
        #
        # 누적에 걸면 6스텝 일관 오차가 최대 6배로 벌점을 받는다. 채점 단위와 정렬된다.
        # 채점 함수와 동일한 loss. 증분/3-mode 희석/성분별 L1 세 문제를 한 번에 없앤다.
        # weight 텐서로 감싸지 않고 GT command mode를 **먼저 slice**하는 것이 핵심이다
        # (mmdet L1Loss의 reduction='mean'이 분모로 B*M*T*2 전체를 써서 실측 정확히
        #  3.0000배 희석됐다).
        if getattr(self, 'loss_plan_metric', None) is not None:
            loss_plan_dict['loss_plan_metric'] = self.loss_plan_metric(
                ego_fut_preds, ego_fut_gt, ego_fut_cmd, ego_fut_masks)

        if getattr(self, 'loss_plan_reg_cum', None) is not None:
            loss_plan_dict['loss_plan_reg_cum'] = self.loss_plan_reg_cum(
                ego_fut_preds.cumsum(dim=-2),
                ego_fut_gt.cumsum(dim=-2),
                loss_plan_l1_weight)

        return loss_plan_dict
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        # `filter_empty_gt=False` 로 range 안 박스가 0개인 프레임이 들어오면
        # bbox_targets가 전부 0이고 normalize_bbox의 log(0) = -inf 때문에
        # isnotnan이 **전부 False**가 된다. 그러면 mmdet l1_loss의
        # `assert target.numel() > 0` 이 터진다.
        # bs>=2 이고 배치에 GT 있는 샘플이 섞이면 살아남으므로(실제 학습이 그랬다)
        # 잠재 크래시로 남아 있었다. 배치 전체가 0-GT면 바로 죽는다.
        # graph를 끊지 않으려고 예측의 0배를 쓴다.
        if isnotnan.sum() == 0:
            loss_bbox = bbox_preds.sum() * 0.0
        else:
            loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10],
                normalized_bbox_targets[isnotnan, :10],
                bbox_weights[isnotnan, :10],
                avg_factor=num_total_pos)

        # traj regression loss
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        neg_inds = (bbox_weights[:, 0] == 0)
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        if isnotnan.sum() == 0:
            loss_traj = best_traj_preds.sum() * 0.0
        else:
            loss_traj = self.loss_traj(
                best_traj_preds[isnotnan],
                traj_targets[isnotnan],
                traj_weights[isnotnan],
                avg_factor=num_total_pos)

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    def get_traj_cls_target(self,
             traj_preds,
             traj_targets,
             gt_fut_masks,
             neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        # loss_plan_* 과 같은 이유로 weight 0이면 그래프를 만들지 않는다.
        # `PtsDirCosLoss`는 코사인 유사도라 방향 벡터가 영벡터면(차선의 중복 점)
        # 분모가 0 -> backward가 0/0. weight 0 항이 전체 gradient를 죽인다.
        # 현재 config는 loss_map_dir=0.0 이다 (우리 lane 표현과 기하가 안 맞아서).
        if float(getattr(self.loss_map_dir, 'loss_weight', 1.0)) != 0.0:
            loss_dir = self.loss_map_dir(
                denormed_pts_preds_dir[isnotnan,:,:],
                pts_targets_dir[isnotnan,:,:],
                dir_weights[isnotnan,:],
                avg_factor=num_total_pos)
        else:
            loss_dir = pts_preds.new_zeros(())

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        ego_fut_preds = preds_dicts['ego_fut_preds']
        ego_vocab_logits = preds_dicts.get('ego_vocab_logits')

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1][..., 0:2], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']
        if 'loss_plan_reg_cum' in loss_planning_dict:
            loss_dict['loss_plan_reg_cum'] = \
                loss_planning_dict['loss_plan_reg_cum']
        if 'loss_plan_metric' in loss_planning_dict:
            loss_dict['loss_plan_metric'] = \
                loss_planning_dict['loss_plan_metric']
        if ego_vocab_logits is not None:
            loss_dict.update(self.goal_decoder.vocabulary_loss(
                ego_vocab_logits, ego_fut_gt, ego_fut_masks))

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        # ---- aux loss 스케일링 (형님 §8) ----
        # 이전에는 planning을 지배항으로 만들려고 config에서 det/map loss weight와
        # **matcher cost를 함께** 줄였다. mmdet이 둘을 같게 assert하기 때문이다.
        # 균등 배율이면 Hungarian 해는 불변이고 실제로 그것을 실측했지만
        # (실제 비용행렬 72개에서 assignment 불일치 0/72, 비균등 교란 시 7/72),
        # matcher를 건드린 상태로 남겨두는 건 방법론적 부채다.
        #
        # 그래서 matcher cost와 원래 loss weight는 **원본 그대로** 두고,
        # assignment와 각 aux loss 계산이 끝난 **뒤에** 반환값만 스케일한다.
        # 이러면 matching은 upstream VAD와 완전히 동일하고 gradient 비중만 바뀐다.
        # planning 항(`plan`)은 건드리지 않는다.
        if self.aux_loss_scale is not None and self.aux_loss_scale != 1.0:
            for k in list(loss_dict.keys()):
                if 'plan' in k:
                    continue
                loss_dict[k] = loss_dict[k] * self.aux_loss_scale

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """
        
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask


    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask
