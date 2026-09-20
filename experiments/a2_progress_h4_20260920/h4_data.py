"""Independent cached provided_status5. State/history supervision is unchanged."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch

ROOT=Path('/NHNHOME/data/sukim/adcl')
CACHE=ROOT/'data/etri/motiondrive_v2/a2_h4_status_20260920'
PRODUCER=Path(__file__).with_name('h4_status.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class H4StatusDataset(torch.utils.data.Dataset):
    def __init__(self,base):
        self.base=base;manifest=json.loads((CACHE/'manifest.json').read_text())
        if manifest['producer_sha256']!=sha(PRODUCER):raise ValueError('H4 producer changed')
        if manifest['split_manifest_sha256']!=base.split_sha or manifest['supervision_unchanged'] is not True:raise ValueError('Data contract mismatch')
        rows=[];status=[]
        for name in ('train','tune'):
            a=manifest['splits'][name];path=Path(a['path'])
            if sha(path)!=a['sha256']:raise ValueError('Status cache hash changed')
            with np.load(path,allow_pickle=False) as z:rows.append(z['row']);status.append(z['status5'])
        rows=np.concatenate(rows);values=np.concatenate(status);order=np.argsort(rows);rows=rows[order];values=values[order]
        loc=np.searchsorted(rows,base.rows)
        if np.any(loc>=len(rows)) or not np.array_equal(rows[loc],base.rows):raise ValueError('Rows outside DEV cache')
        self.values=np.ascontiguousarray(values[loc]);self.values.setflags(write=False)
    def __getattr__(self,name):
        if name=='base':raise AttributeError(name)
        return getattr(self.base,name)
    def __len__(self):return len(self.base)
    def set_epoch(self,epoch):return self.base.set_epoch(epoch)
    def __getitem__(self,index):
        item=self.base[index];item['provided_status5']=torch.from_numpy(self.values[index].copy());return item
