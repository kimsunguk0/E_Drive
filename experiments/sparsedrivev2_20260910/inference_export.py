"""Verify optimizer-stripped artifacts against their source-state export receipt."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch


def state_fingerprint(state):
    """Content digest over sorted tensor names/dtypes/shapes and exact raw bytes."""
    digest=hashlib.sha256();entries={}
    for name,value in sorted(state.items()):
        if not isinstance(value,torch.Tensor):raise ValueError('State dictionary must contain tensors only')
        tensor=value.detach().cpu().contiguous()
        raw=tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        meta={'dtype':str(tensor.dtype),'shape':list(tensor.shape),'bytes':len(raw),
              'sha256':hashlib.sha256(raw).hexdigest()}
        digest.update(name.encode()+b'\0'+json.dumps(meta,sort_keys=True,separators=(',',':')).encode()+b'\n')
        entries[name]=meta
    return {'sha256':digest.hexdigest(),'tensor_count':len(entries),
            'bytes':sum(x['bytes'] for x in entries.values()),'byte_order':sys.byteorder,'tensors':entries}


def validate_export_receipt(payload,artifact_sha256,receipt_path,state_key):
    """No original checkpoint/training data is opened at runtime."""
    path=Path(receipt_path);raw=path.read_bytes();receipt=json.loads(raw)
    if receipt.get('schema')!='sparsedrivev2_optimizer_stripped_export_v1':raise ValueError('Unknown export schema')
    if receipt.get('artifact_sha256')!=artifact_sha256:raise ValueError('Export receipt artifact SHA mismatch')
    if receipt.get('state_key')!=state_key or receipt.get('removed_keys')!=['optimizer']:
        raise ValueError('Export must remove optimizer only')
    if 'optimizer' in payload or sorted(payload)!=receipt.get('exported_payload_keys'):
        raise ValueError('Export payload fields disagree with receipt')
    if sorted(set(payload)|{'optimizer'})!=receipt.get('source_payload_keys'):
        raise ValueError('Export changed non-optimizer payload fields')
    source=receipt.get('source_checkpoint_sha256')
    if not isinstance(source,str) or len(source)!=64:raise ValueError('Missing source checkpoint identity')
    fingerprint=state_fingerprint(payload[state_key])
    if fingerprint!=receipt.get('source_state_fingerprint') or fingerprint!=receipt.get('exported_state_fingerprint'):
        raise ValueError('Export changed trained state tensors')
    return {'receipt_sha256':hashlib.sha256(raw).hexdigest(),'artifact_sha256':artifact_sha256,
        'source_checkpoint_sha256':source,'state_sha256':fingerprint['sha256'],
        'state_tensor_count':fingerprint['tensor_count'],'optimizer_removed':True,
        'state_identity_exact':True,'source_checkpoint_reopened_at_runtime':False}
