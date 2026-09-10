"""CPU synthetic same-forward base selection and export tensor-identity checks."""
import copy,hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
try:
    from .evaluate_relative_selector import paired_base_prediction
    from .inference_export import state_fingerprint,validate_export_receipt
except ImportError:
    from evaluate_relative_selector import paired_base_prediction
    from inference_export import state_fingerprint,validate_export_receipt


def main():
    torch.set_num_threads(2)
    bank=torch.arange(2*3*8*3,dtype=torch.float32).reshape(2,3,8,3)
    model=SimpleNamespace(_trajectory_head=SimpleNamespace(traj_vocab=bank))
    ids=torch.arange(6).reshape(2,3)
    output={'candidate_ids':ids,'candidate_xy':bank.flatten(0,1)[ids,:6,:2],
        'candidate_valid':torch.tensor([[True,False,True],[True,True,False]]),
        'base_scores':torch.tensor([[1.,999.,2.],[1.,2.,999.]]),
        'scores':torch.tensor([[100.,0.,0.],[100.,0.,0.]])}
    before={k:v.clone() for k,v in output.items()}
    pred,selected=paired_base_prediction(model,output)
    assert torch.equal(selected,torch.tensor([2,4]))
    assert torch.equal(pred,bank.flatten(0,1)[selected,:6,:2])
    assert all(torch.equal(output[k],v) for k,v in before.items())
    state={'weight':torch.randn(2,3),'count':torch.tensor(7),'bf16':torch.ones(2,dtype=torch.bfloat16)}
    fingerprint=state_fingerprint(state)
    payload={'model':state,'manifest':{'example':'synthetic'},'step':1}
    receipt={'schema':'sparsedrivev2_optimizer_stripped_export_v1','artifact_sha256':'1'*64,
        'source_checkpoint_sha256':'2'*64,'state_key':'model','removed_keys':['optimizer'],
        'source_payload_keys':['manifest','model','optimizer','step'],'exported_payload_keys':sorted(payload),
        'source_state_fingerprint':fingerprint,'exported_state_fingerprint':fingerprint}
    with tempfile.TemporaryDirectory() as name:
        path=Path(name)/'receipt.json';path.write_text(json.dumps(receipt))
        validate_export_receipt(payload,'1'*64,path,'model')
        changed=copy.deepcopy(payload);changed['model']['weight'][0,0]+=1
        try:validate_export_receipt(changed,'1'*64,path,'model')
        except ValueError:pass
        else:raise AssertionError('Changed state accepted')
        changed=copy.deepcopy(payload);changed['optimizer']={}
        try:validate_export_receipt(changed,'1'*64,path,'model')
        except ValueError:pass
        else:raise AssertionError('Optimizer retained in export')
    print(json.dumps({'passed':True,'device':'cpu','synthetic_only':True,'held_files_opened':False,
        'checks':['same-forward base scores choose an exact existing candidate, independent of head winner',
                  'invalid high-scoring base candidate is excluded','output candidate tensors and scores unchanged',
                  'export fingerprint supports scalar integer and BF16 state tensors',
                  'changed export state or retained optimizer rejected']},indent=2))


if __name__=='__main__':main()
