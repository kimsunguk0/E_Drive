"""Image-only offset prediction for the existing shared-scene sample sources.

Status/goal still condition only the existing scene query. The offset module
never receives that query, conditioned scene values, or provided status.
"""
import torch
from torch import nn
from torch.nn import functional as F
from common import A2NominalModel
from models.motiondrive_v2.scene_encoder import SharedSceneEncoder, project_scene_points

OFFSET_PREFIX='scene_encoder.offset_predictor.'

class OffsetPredictor(nn.Module):
    # camera one-hot(6), FPN one-hot(2), time(1), height(1), normalized xy(2)
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(128+12,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.radius=2.
    def forward(self,visual,metadata):
        with torch.autocast(device_type=visual.device.type,enabled=False):
            return self.radius*torch.tanh(self.network(torch.cat([visual.float(),metadata.float()],-1)))
    def assert_zero(self):
        assert not bool(self.network[-1].weight.count_nonzero())
        assert not bool(self.network[-1].bias.count_nonzero())

def shifted_grid(grid,offset_feature_cells,feature_hw,original_valid,image_hw):
    """Offset is [B,Q,V,H,2]; grid is [B,V,Q,H,2], align_corners=False."""
    h,w=feature_hw
    delta=offset_feature_cells.permute(0,2,1,3,4).float()*grid.new_tensor([2./w,2./h])
    moved=grid.float()+delta
    ih,iw=image_hw
    uv=(moved+1)*moved.new_tensor([iw/2.,ih/2.])-.5
    inside=(uv[...,0]>=0)&(uv[...,0]<iw)&(uv[...,1]>=0)&(uv[...,1]<ih)&torch.isfinite(uv).all(-1)
    # Exact zero offset inherits the existing projector's validity, avoiding a
    # round-trip pixel conversion changing a boundary point at initialization.
    zero=(offset_feature_cells==0).all(-1).permute(0,2,1,3)
    valid=original_valid & (inside|zero)
    safe=torch.where(valid[...,None],moved,torch.full_like(moved,2.))
    return safe,valid

class LearnedSampleSceneEncoder(SharedSceneEncoder):
    def _offset_read(self,value,key,grid,valid,camera_ids,times,level,image_hw):
        initial=self._sample(value,grid)
        b,q,v,nh,_=initial.shape
        cameras=F.one_hot(camera_ids,6).to(grid).view(1,1,v,1,6).expand(b,q,v,nh,6)
        levels=F.one_hot(torch.tensor(level,device=grid.device),2).to(grid).view(1,1,1,1,2).expand(b,q,v,nh,2)
        tt=times[:,None,:,None,None].expand(b,q,v,nh,1)
        hh=grid.new_tensor(self.config.heights).view(1,1,1,nh,1).expand(b,q,v,nh,1)/2.
        xy=grid.permute(0,2,1,3,4)
        metadata=torch.cat([cameras,levels,tt,hh,xy],-1)
        offset=self.offset_predictor(initial,metadata)
        moved,mask=shifted_grid(grid,offset,value.shape[-2:],valid,image_hw)
        new_value=self._sample(value,moved)
        new_key=self._sample(key,moved)
        if self.collect_sampling_stats:
            active=valid.permute(0,2,1,3)
            z=offset.detach()[active]
            self.sampling_stats.append({'level':level,'views':v,'feature_hw':list(value.shape[-2:]),
              'initial_valid':int(active.sum()),'lost_valid':int((valid & ~mask).sum()),
              'mean_abs_cells':float(z.abs().mean()) if z.numel() else 0.,
              'max_abs_cells':float(z.abs().max()) if z.numel() else 0.,
              'saturated_components':int((z.abs()>1.9).sum()),'components':z.numel()})
        return new_value,new_key,mask

    def forward(self,current_levels,history_levels,current_global,lidar2img,
                history_transforms,time_offsets,goal_xy,image_hw,*,
                history_camera_ids=None,history_pose_indices=None):
        if not self.sampling_enabled:
            return super().forward(current_levels,history_levels,current_global,lidar2img,
                history_transforms,time_offsets,goal_xy,image_hw,
                history_camera_ids=history_camera_ids,history_pose_indices=history_pose_indices)
        b=lidar2img.shape[0];c=self.config.channels
        matrices,camera_ids,view_times=self._view_geometry(lidar2img,history_transforms,time_offsets,
            history_camera_ids,history_pose_indices)
        if any(x.shape[1]!=matrices.shape[1]-6 for x in history_levels):raise ValueError('source mismatch')
        metadata=(self.camera_keys[camera_ids][None,:,None,:]+self.height_keys[None,None]
                  +self.time_key(view_times[...,None])[:,:,None])
        used_goal=goal_xy if self.config.goal_on else torch.zeros_like(goal_xy)
        scale=self.cell_xy.new_tensor([80.,64.])
        context=torch.cat([self.cell_xy[None].expand(b,-1,-1)/scale,
          used_goal[:,None].expand(-1,len(self.cell_xy),-1)/scale,
          (self.cell_xy[None]-used_goal[:,None])/scale],-1)
        q_context=self.query_context(context)
        projected_levels=[]
        for i,(cur,past) in enumerate(zip(current_levels,history_levels)):
            def projected(feat,layer):
                out=layer(feat.flatten(0,1))
                return out.reshape(feat.shape[0],feat.shape[1],*out.shape[1:])
            projected_levels.append((projected(cur,self.value_proj[i]),projected(past,self.value_proj[i]),
                projected(cur,self.key_proj[i]),projected(past,self.key_proj[i])))
        chunks=[];visible_chunks=[]
        for start in range(0,len(self.points),self.config.scene_chunk_size):
            stop=start+self.config.scene_chunk_size
            grid,valid=project_scene_points(self.points[start:stop],matrices,image_hw)
            values=[];keys=[];masks=[]
            for level,(cv,pv,ck,pk) in enumerate(projected_levels):
                a,ak,am=self._offset_read(cv,ck,grid[:,:6],valid[:,:6],camera_ids[:6],view_times[:,:6],level,image_hw)
                z,zk,zm=self._offset_read(pv,pk,grid[:,6:],valid[:,6:],camera_ids[6:],view_times[:,6:],level,image_hw)
                level_values=torch.cat([a,z],2)
                level_keys=torch.cat([ak,zk],2)+metadata[:,None]+self.scale_keys[level]
                level_valid=torch.cat([am,zm],1)
                values.append(level_values.flatten(2,3));keys.append(level_keys.flatten(2,3))
                masks.append(level_valid.permute(0,2,1,3).flatten(2,3))
            value,key,valid=torch.cat(values,2),torch.cat(keys,2),torch.cat(masks,2)
            base=(value*valid[...,None]).sum(2)/valid.sum(2).clamp_min(1)[...,None]
            query=self.query_image(base)+q_context[:,start:stop]
            evidence=self._aggregate_evidence(query,key,value,valid)
            chunks.append(base+evidence.to(base.dtype));visible_chunks.append(valid.any(-1))
        scene=torch.cat(chunks,1);visible=torch.cat(visible_chunks,1)
        scene=scene+self.global_proj(current_global.mean(dim=(1,3,4)))[:,None]
        scene=self._apply_cross_cell_goal_residual(scene,visible,goal_xy)
        raster=self.refine(scene.transpose(1,2).reshape(b,c,*self.config.grid_size))
        return {'scene_features':raster.flatten(2).transpose(1,2),'occ_logits':self.occ_head(raster).float(),
          'lane_logits':self.lane_head(raster).float(),'scene_visible':visible.reshape(b,1,*self.config.grid_size)}

class LearnedSampleModel(A2NominalModel):
    def __init__(self,config):
        super().__init__(config,arm='A2-BASE-NOM')
        # Preserve the existing query hook and every old parameter. Fork RNG so
        # the subsequently rebuilt MR fuse has the identical initialization.
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(2026091901)
            self.scene_encoder.add_module('offset_predictor',OffsetPredictor())
        self.scene_encoder.__class__=LearnedSampleSceneEncoder
        self.scene_encoder.sampling_enabled=True
        self.scene_encoder.collect_sampling_stats=False
        self.scene_encoder.sampling_stats=[]
