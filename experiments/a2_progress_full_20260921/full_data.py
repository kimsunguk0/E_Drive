"""FULL-only H4 status cache; never used by the DEV dataset wrapper."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
ROOT=Path('/NHNHOME/data/sukim/adcl')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/a2_progress_h4_20260920'))
import h4_status
CACHE=ROOT/'data/etri/motiondrive_v2/a2_h4_status_full_20260921'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
class FullH4StatusDataset(torch.utils.data.Dataset):
    def __init__(self,base):
        self.base=base;manifest=json.loads((CACHE/'manifest.json').read_text())
        assert manifest['producer_sha256']==sha(h4_status.__file__)
        assert manifest['split_manifest_sha256']==base.split_sha and manifest['supervision_unchanged']
        rows=[];values=[]
        for split in ('train','tune','val'):
            entry=manifest['splits'][split];assert sha(entry['path'])==entry['sha256']
            with np.load(entry['path'],allow_pickle=False) as z:rows.append(z['row']);values.append(z['status5'])
        rows=np.concatenate(rows);values=np.concatenate(values);order=np.argsort(rows);rows=rows[order];values=values[order]
        assert len(rows)==101520 and len(np.unique(rows))==len(rows) and np.isfinite(values).all()
        loc=np.searchsorted(rows,base.rows)
        assert not np.any(loc>=len(rows)) and np.array_equal(rows[loc],base.rows)
        self.values=np.ascontiguousarray(values[loc]);self.values.setflags(write=False)
    def __getattr__(self,name):
        if name=='base':raise AttributeError(name)
        return getattr(self.base,name)
    def __len__(self):return len(self.base)
    def set_epoch(self,epoch):return self.base.set_epoch(epoch)
    def __getitem__(self,index):
        item=self.base[index];item['provided_status5']=torch.from_numpy(self.values[index].copy());return item
