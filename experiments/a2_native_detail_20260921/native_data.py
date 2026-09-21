"""Same RGB source/crop/jitter for matched native and down-up detail arms."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image,ImageEnhance

ROOT=Path('/NHNHOME/data/sukim/adcl')
CACHE=ROOT/'data/etri/motiondrive_v2/native_detail_1152_20260921'
KEY='high_detail_images'

class NativeDataset(torch.utils.data.Dataset):
    def __init__(self,base,arm,require_complete=True):
        self.base=base;self.arm=arm
        if arm not in ('M-LOW','M-NATIVE','S-LOW','S-NATIVE'):raise ValueError(arm)
        p=CACHE/('manifest.json' if require_complete else 'plan.json')
        if not p.exists():raise ValueError('Native cache not ready: '+str(p))
        self.plan=json.loads((CACHE/'plan.json').read_text())
        if base.split_sha!=self.plan['split_manifest_sha256']:raise ValueError('Wrong DEV split')
    def __getattr__(self,name):
        if name=='base':raise AttributeError(name)
        return getattr(self.base,name)
    def __len__(self):return len(self.base)
    def set_epoch(self,e):self.base.set_epoch(e)
    def __getitem__(self,index):
        from motiondrive_v2_data import CAMERA_ORDER,MEAN,STD
        item=self.base[index];row=int(self.base.rows[index]);scene=str(self.base.scene_names[row]);frame=int(self.base.arr['frame'][row])
        jitter=None
        if self.base.augment:
            jitter=np.random.default_rng(self.base.seed+self.base.epoch*1000003+row).uniform(.9,1.1,(6,3))
        if self.arm.startswith('M-'):
            sources=[(0,frame)]+[(0,frame-int(o)) for o in self.base.history_offsets]
        else:sources=[(i,frame) for i in range(6)]
        images=[]
        for cam,fi in sources:
            with Image.open(CACHE/scene/CAMERA_ORDER[cam]/f'{fi:08d}.jpg') as handle:im=handle.convert('RGB')
            if im.size!=(1152,648):raise ValueError('Wrong native cache dimensions')
            if self.arm.endswith('LOW'):
                im=im.resize((768,432),Image.Resampling.BILINEAR).resize((1152,648),Image.Resampling.BILINEAR)
            if jitter is not None:
                for cls,factor in zip((ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color),jitter[cam]):
                    im=cls(im).enhance(float(factor))
            a=np.asarray(im,np.float32)/255.
            images.append(torch.from_numpy(((a-MEAN)/STD).transpose(2,0,1).copy()))
        item[KEY]=torch.stack(images)
        return item

def wrap_flip(original,arm):
    def flip(item,width_full,width_hist):
        out=original(item,width_full,width_hist)
        if KEY in item:
            v=torch.flip(item[KEY],dims=[-1])
            if arm.startswith('S-'):v=v[[0,2,1,3,5,4]]
            out[KEY]=v
        return out
    return flip
