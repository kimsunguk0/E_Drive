"""Synthetic CPU provenance/rejection checks; never treats fixture weights as trained results."""
import copy,hashlib,json,tempfile
from pathlib import Path
import torch
try:
    from .relative_artifact import load_relative_artifact,RELATIVE_SOURCE_SHA256,TRAINER_SOURCE_SHA256,TUNE_SHA256,require_confirmation_binding,CONFIRM_TRAIN_SHA256,CONFIRM_HELD_SHA256
    from .relative_selector import RelativeScoreHead,FEATURE_NAMES,FEATURE_VERSION
except ImportError:
    from relative_artifact import load_relative_artifact,RELATIVE_SOURCE_SHA256,TRAINER_SOURCE_SHA256,TUNE_SHA256,require_confirmation_binding,CONFIRM_TRAIN_SHA256,CONFIRM_HELD_SHA256
    from relative_selector import RelativeScoreHead,FEATURE_NAMES,FEATURE_VERSION


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='relative_artifact_cpu_') as name:
        directory=Path(name);cache_file=directory/'manifest.json';head_file=directory/'head.pth'
        cache={'schema':'sparsedrivev2_selection_cache_v1','status':'completed',
            'feature_source_sha256':RELATIVE_SOURCE_SHA256,'feature_version':FEATURE_VERSION,
            'feature_names':list(FEATURE_NAMES),'feature_dim':32,'candidate_count':200,
            'base_goal_mode':'none','goal_feature_mode':'selection','precision':'bf16',
            'held_population_evaluated':False,'tune_rows_sha256':TUNE_SHA256,
            'checkpoint':{'path':'THIS_BASE_FILE_IS_NOT_OPENED','sha256':'1'*64},
            'bank':{'path':'THIS_BANK_FILE_IS_NOT_OPENED','sha256':'2'*64},
            'provenance':{'checkpoint_sha256':'1'*64,'bank_sha256':'2'*64},
            'artifacts':{'train/gt_xy.npy':{'path':'LABEL_FILE_MUST_NOT_BE_OPENED'}}}
        cache_file.write_text(json.dumps(cache)+'\n')
        payload={'relative_head':RelativeScoreHead().state_dict(),'step':1,'result':{'step':1},
            'manifest':{'torch':str(torch.__version__),'source_sha256':{
                'relative_selector.py':RELATIVE_SOURCE_SHA256,'train_cached_selector.py':TRAINER_SOURCE_SHA256},
                'arguments':{'objective':'soft_ce'},'cache_path':str(directory),
                'cache_manifest_sha256':sha(cache_file),'cache_manifest':cache}}
        def save(value):torch.save(value,head_file);return sha(head_file)
        digest=save(payload)
        head,metadata,receipt=load_relative_artifact(head_file,expected_sha256=digest,
            base_checkpoint_sha256='1'*64,bank_sha256='2'*64,base_goal_mode='none')
        assert all(torch.equal(head.state_dict()[key],value) for key,value in payload['relative_head'].items())
        assert receipt['runtime_feature_or_label_files_opened'] is False
        checks.append('valid metadata/FP32 head exact load; no referenced feature, label, bank or base file opened by head-only verifier')
        confirmation_cache=copy.deepcopy(cache)
        confirmation_cache.update(allowed_train_rows_sha256=CONFIRM_TRAIN_SHA256,confirmation12_excluded_from_fit_population=True)
        confirmation_cache['provenance'].update(fit_rows_sha256=CONFIRM_TRAIN_SHA256,fit_rows=46170,split_sha256='4'*64)
        approval={'schema':'sparsedrivev2_confirmation_candidate_v1','population':'confirmation12','approved':True,'frozen':True,
            'checkpoint_sha256':'1'*64,'bank_sha256':'2'*64,'train_rows_sha256':CONFIRM_TRAIN_SHA256,
            'eval_rows_sha256':CONFIRM_HELD_SHA256,'split_sha256':'4'*64,'relative_head_sha256':digest,
            'relative_source_sha256':RELATIVE_SOURCE_SHA256,'relative_trainer_sha256':TRAINER_SOURCE_SHA256,
            'relative_cache_manifest_sha256':receipt['cache_manifest_sha256'],'relative_objective':'soft_ce',
            'feature_goal_mode':'selection','base_goal_mode':'none','evaluation_batch_size':1,
            'evaluation_precision':'bf16_base_fp32_head',
            'evaluation_metric':'D3_prefix_ADE_1_2_3_seconds_weights_11_11_5_5_2_2_over36'}
        require_confirmation_binding(approval,receipt,confirmation_cache)
        checks.append('synthetic frozen decision accepts exact base/head/cache/rows/source/objective binding')
        for key in ('relative_head_sha256','relative_cache_manifest_sha256','relative_source_sha256',
                    'relative_trainer_sha256','checkpoint_sha256','bank_sha256','train_rows_sha256',
                    'eval_rows_sha256','relative_objective','feature_goal_mode','base_goal_mode','frozen',
                    'evaluation_batch_size','evaluation_precision','evaluation_metric'):
            altered=copy.deepcopy(approval);altered[key]='WRONG'
            try:require_confirmation_binding(altered,receipt,confirmation_cache)
            except ValueError:pass
            else:raise AssertionError('Confirmation accepted mismatch: '+key)
        checks.append('all fifteen frozen-decision binding mutations rejected before any held data access')
        for unapproved in (None,{},dict(approval,approved=False)):
            try:require_confirmation_binding(unapproved,receipt,confirmation_cache)
            except ValueError:pass
            else:raise AssertionError('Unapproved confirmation accepted')
        wrong_fit=copy.deepcopy(confirmation_cache);wrong_fit['confirmation12_excluded_from_fit_population']=False
        try:require_confirmation_binding(approval,receipt,wrong_fit)
        except ValueError:checks.append('fit population including held rows rejected')
        else:raise AssertionError('Wrong fit population accepted')
        def rejected(label,value,**kwargs):
            digest=save(value)
            try:load_relative_artifact(head_file,expected_sha256=digest,**kwargs)
            except (ValueError,RuntimeError,KeyError):checks.append(label);return
            raise AssertionError(label+' was not rejected')
        rejected('wrong frozen base rejected',payload,base_checkpoint_sha256='3'*64)
        rejected('wrong frozen bank rejected',payload,bank_sha256='3'*64)
        rejected('wrong base goal mode rejected',payload,base_goal_mode='selection')
        bad=copy.deepcopy(payload);bad['manifest']['source_sha256']['train_cached_selector.py']='3'*64
        rejected('unrecognized trainer rejected',bad)
        bad=copy.deepcopy(payload);bad['manifest']['cache_manifest']['goal_feature_mode']='none'
        rejected('embedded/external provenance disagreement rejected',bad)
        bad=copy.deepcopy(payload);bad['relative_head'].pop(next(iter(bad['relative_head'])))
        rejected('missing learned tensor rejected by strict load',bad)
        bad=copy.deepcopy(payload);bad['relative_head']['mlp.4.weight'].fill_(torch.nan)
        rejected('nonfinite learned tensor rejected',bad)
        bad=copy.deepcopy(payload);bad['step']=0;bad['result']['step']=0
        rejected('zero-step head rejected',bad)
        save(payload)
        try:load_relative_artifact(head_file,expected_sha256='0'*64)
        except ValueError:checks.append('selected head SHA mismatch rejected')
        else:raise AssertionError('SHA mismatch accepted')
    report={'passed':True,'device':'cpu','synthetic_fixtures_only':True,'checks':checks}
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
