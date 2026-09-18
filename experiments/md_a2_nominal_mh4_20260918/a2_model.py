"""A2 query-only model with optional four independent scene evidence reads."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path("/NHNHOME/data/sukim/adcl")
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"experiments/md_shared_dynamics_20260917"))
from factorized_model import SharedDynamicsMotionDriveV2
from models.motiondrive_v2.scene_encoder import masked_softmax
ARMS=("A2-FULL-NOM","A2-BASE-NOM","A2-MH4-NOM")
MH4_PREFIX="scene_encoder.evidence_attention."

class MultiHeadEvidence(nn.Module):
    """Q/K=32 per head; 4 image-value slices; no query residual into values."""
    def __init__(self):
        super().__init__()
        # Explicit identities consume no random draws, preserving the MR rebuild.
        self.query_matrix=nn.Parameter(torch.eye(32).repeat(4,1,1))
        self.key_matrix=nn.Parameter(torch.eye(32).repeat(4,1,1))
        self.output_matrix=nn.Parameter(torch.eye(128))
    def forward(self,query,key,value,valid):
        if query.shape[-1]!=32 or key.shape[-1]!=32 or value.shape[-1]!=128:
            raise ValueError("MH4 requires Q/K32 and V128")
        with torch.autocast(device_type=query.device.type,enabled=False):
            q,k,v=query.float(),key.float(),value.float()
            reads=[]
            for head in range(4):
                qh=F.linear(q,self.query_matrix[head].float())
                kh=F.linear(k,self.key_matrix[head].float())
                scores=(qh[:,:,None]*kh).sum(-1)/(32.**.5)
                attention=masked_softmax(scores,valid)
                reads.append((attention[...,None]*v[...,head*32:(head+1)*32]).sum(2))
            return F.linear(torch.cat(reads,-1),self.output_matrix.float())
    def assert_identity(self):
        if not torch.equal(self.query_matrix.detach(),torch.eye(32,device=self.query_matrix.device).repeat(4,1,1)):
            raise ValueError("MH4 query initialization differs")
        if not torch.equal(self.key_matrix.detach(),torch.eye(32,device=self.key_matrix.device).repeat(4,1,1)):
            raise ValueError("MH4 key initialization differs")
        if not torch.equal(self.output_matrix.detach(),torch.eye(128,device=self.output_matrix.device)):
            raise ValueError("MH4 output initialization differs")

class A2NominalModel(SharedDynamicsMotionDriveV2):
    VALID_ARMS={**SharedDynamicsMotionDriveV2.VALID_ARMS,**{name:0 for name in ARMS}}
    def __init__(self,config,*,arm):
        if arm not in ARMS:raise ValueError("Unknown nominal A2 arm")
        super().__init__(config,arm=arm)
        if arm=="A2-MH4-NOM":
            if config.channels!=128 or config.scene_attention_channels!=32:
                raise ValueError("MH4 geometry mismatch")
            self.scene_encoder.add_module("evidence_attention",MultiHeadEvidence())
        if self.shared_status_feature_conditioner is not None or self.progress_head is not None:
            raise ValueError("Nominal A2 arms must remain query-only direct planning")
