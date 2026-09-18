"""Nominal status input wrappers; never overwrite state/history targets."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path("/NHNHOME/data/sukim/adcl")
for p in (ROOT,ROOT/"scripts",ROOT/"experiments/md_progress_residual_20260917"):
    sys.path.insert(0,str(p))
from full_train import UniqueSplitUnion,make_union
CACHE=ROOT/"data/etri/motiondrive_v2/a2_nominal_status_20260918"
PRODUCER=ROOT/"experiments/md_a2_deploy_status_20260918/nominal_status.py"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def load_cache(allowed_splits):
    manifest=json.loads((CACHE/"manifest.json").read_text())
    if manifest["producer_sha256"]!=sha(PRODUCER) or manifest["supervision_unchanged"] is not True:
        raise ValueError("Nominal producer contract changed")
    rows=[];values=[]
    for split in allowed_splits:
        artifact=manifest["splits"][split];path=Path(artifact["path"])
        if sha(path)!=artifact["sha256"]:raise ValueError("Input cache SHA mismatch")
        with np.load(path,allow_pickle=False) as z:
            rr=z["row"].copy();vv=z["status5"].copy()
        if len(rr)!=artifact["rows"] or hashlib.sha256(np.asarray(rr,dtype="<i8").tobytes()).hexdigest()!=artifact["rows_sha256"]:
            raise ValueError("Input row contract mismatch")
        rows.append(rr);values.append(vv)
    rows=np.concatenate(rows);values=np.concatenate(values)
    order=np.argsort(rows);rows=rows[order];values=values[order]
    if len(np.unique(rows))!=len(rows) or not np.isfinite(values).all():raise ValueError("Invalid cache table")
    return rows,values,manifest

class NominalStatusDataset(torch.utils.data.Dataset):
    def __init__(self,base,*,allowed_splits=("train","tune")):
        self.base=base
        rows,values,self.nominal_manifest=load_cache(allowed_splits)
        positions=np.searchsorted(rows,np.asarray(base.rows,dtype=np.int64))
        if (positions>=len(rows)).any() or not np.array_equal(rows[positions],base.rows):
            raise ValueError("Dataset rows are outside the allowed nominal input splits")
        self.nominal_status=np.ascontiguousarray(values[positions]);self.nominal_status.setflags(write=False)
        self.allowed_nominal_splits=tuple(allowed_splits)
    def __len__(self):return len(self.base)
    def __getattr__(self,name):
        if name=="base":raise AttributeError(name)
        return getattr(self.base,name)
    def set_epoch(self,epoch):self.base.set_epoch(epoch)
    def __getitem__(self,index):
        item=self.base[index]
        state,valid=item.get("state_target"),item.get("state_valid")
        if not isinstance(state,torch.Tensor) or state.shape!=(6,) or not bool(valid[:5].all()):
            raise ValueError("Original state supervision contract changed")
        # A separate tensor with its own mirror transform in the existing flip wrapper.
        item["provided_status5"]=torch.from_numpy(self.nominal_status[index].copy())
        return item
