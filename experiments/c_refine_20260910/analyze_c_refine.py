"""Read-only paired analysis of fixed-terminal real/zero C scene-head runs.

Reads saved evaluation NPZ/JSON, training logs and provenance manifests only.
No model forward, bank refit, candidate cache feature/GT read or checkpoint load.
TUNE is repeated exploratory data. Captured tokens mix image evidence, candidate
embeddings and common-perception status; a gain is not pure image causality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

SCENE_SHA='be8d9eef8e1e9f087852c1a81d0f5f4437e618a146d02d5037de60181b360ae5'
C_SHA='b9dcc56af7c2c4c3cfc80d844fe862a2d8f730d7b8d7a813ee997d33abfec2ff'
TRAIN_SHA='75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854'
TUNE_SHA='1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88'
WEIGHTS=np.asarray([11,11,5,5,2,2],np.float64)/36


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def row_sha(value):return hashlib.sha256(np.asarray(value,dtype=np.int64).tobytes()).hexdigest()


def exact(left,right):
    return left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def paired_bootstrap(delta,sessions,repetitions=20000,seed=0):
    """Sample sessions with replacement, pool rows, retain paired row differences."""
    delta=np.asarray(delta,np.float64);sessions=np.asarray(sessions)
    require(delta.ndim==1 and sessions.shape==delta.shape and np.isfinite(delta).all(),'Invalid paired inputs')
    unique,inverse=np.unique(sessions,return_inverse=True)
    require(len(unique)>0 and repetitions>0,'Empty bootstrap')
    count=np.bincount(inverse).astype(np.float64)
    sums=np.bincount(inverse,weights=delta).astype(np.float64)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(unique),size=(repetitions,len(unique)))
    samples=sums[draws].sum(-1)/count[draws].sum(-1)
    return {'mean_delta':float(delta.mean()),'ci95':np.percentile(samples,[2.5,97.5]).tolist(),
        'bootstrap_std':float(samples.std()),'fraction_bootstrap_delta_below_zero':float(np.mean(samples<0)),
        'n_rows':len(delta),'n_sessions':len(unique),'repetitions':repetitions,'seed':seed,
        'method':'paired session-cluster percentile bootstrap; pooled row mean in each resample',
        'session':{str(s):{'n':int(n),'mean_delta':float(total/n)} for s,n,total in zip(unique,count,sums)}}


def stats(arrays):
    error=arrays['d3'].astype(np.float64);oracle=arrays['shortlist_oracle'].astype(np.float64)
    point=arrays['point_l2'].astype(np.float64)
    require(point.shape==(len(error),6),'Expected point L2 [N,6]')
    require(np.isfinite(error).all() and np.isfinite(point).all() and np.isfinite(oracle).all(),'Nonfinite metrics')
    require(np.all(error+2e-6>=oracle),'Selected error below candidate oracle')
    require(np.allclose(point@WEIGHTS,error,atol=2e-6,rtol=2e-6),'Saved selected point L2 and D3 disagree')
    return {'n':len(error),'d3':float(error.mean()),'oracle':float(oracle.mean()),
        'regret':float((error-oracle).mean()),'d3_p99':float(np.percentile(error,99)),
        'point_l2_mean':point.mean(0).tolist(),'point_l2_p99':np.percentile(point,99,axis=0).tolist(),
        'd3_below_point20':float(error.mean())<.20,'d3_below_point15':float(error.mean())<.15,
        'session_d3':{str(s):float(error[arrays['session_index']==s].mean()) for s in np.unique(arrays['session_index'])}}


def read_eval(directory,step,prefix='eval'):
    npz=directory/f'{prefix}_{step:06d}.npz';js=directory/f'{prefix}_{step:06d}.json'
    with np.load(npz,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
    required={'rows','session_index','d3','shortlist_oracle','base_d3','candidate_id','pred','point_l2'}
    require(required<=set(arrays),'Saved evaluation fields missing')
    n=1998 if prefix=='eval' else 54810
    require(len(arrays['rows'])==n,'Evaluation population changed')
    require(row_sha(arrays['rows'])==(TUNE_SHA if prefix=='eval' else TRAIN_SHA),'Evaluation row identity/order mismatch')
    require(arrays['pred'].shape==(n,6,2) and arrays['candidate_id'].shape==(n,),'Malformed prediction/ID')
    require(np.all((arrays['candidate_id']>=0)&(arrays['candidate_id']<1024*1024)),'Candidate ID outside pinned bank')
    result=json.loads(js.read_text());summary=stats(arrays)
    require(result['step']==step and result['n']==n,'Saved evaluation JSON population/step mismatch')
    for key,value in (('official_d3',summary['d3']),('shortlist_oracle_d3',summary['oracle']),('selection_regret',summary['regret'])):
        require(np.isclose(result[key],value,atol=2e-7,rtol=2e-6),'Saved JSON/NPZ metric mismatch: '+key)
    return arrays,summary,{'npz':str(npz),'npz_sha256':sha(npz),'json':str(js),'json_sha256':sha(js)}


def read_run(directory,expected_mode):
    directory=Path(directory).resolve();manifest=json.loads((directory/'manifest.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    require(result['status']=='completed','Run is not completed')
    require(manifest['schema']=='c_scene_selector_frozen_v1' and manifest['arguments']['mode']==expected_mode,'Wrong head run/mode')
    args=manifest['arguments'];steps=int(args['steps']);interval=int(args['eval_every'])
    require(steps==4000 and interval==500 and args['seed']==0 and args['batch']==128,'Predeclared matched head schedule changed')
    require(result['steps']==steps and manifest['rng_seed']==args['seed']+1,'Terminal step/RNG changed')
    require(manifest['source_sha256']['c_scene_selector.py']==SCENE_SHA,'Head implementation changed')
    require(str(torch.__version__)==manifest['torch'],'Batch-order reconstruction requires the training CPU torch version')
    source_root=(directory/'source').resolve()
    for name,digest in manifest['source_sha256'].items():
        path=(source_root/name).resolve()
        require(path.is_relative_to(source_root) and sha(path)==digest,'Measured source changed: '+name)
    for split,expected_sha in (('train',TRAIN_SHA),('tune',TUNE_SHA)):
        cache=manifest[f'{split}_cache_manifest']
        path=Path(args[f'{split}_cache'])/'manifest.json'
        require(sha(path)==manifest[f'{split}_cache_manifest_sha256'],'Cache manifest identity changed')
        require(canonical(json.loads(path.read_text()))==canonical(cache),'Embedded/disk cache manifest mismatch')
        require(cache['status']=='completed' and cache['rows_sha256']==expected_sha,'Cache population changed')
        require(cache['checkpoint_sha256']==C_SHA and cache['bank_sha256']==manifest['bank_sha256'],'Cache base/bank changed')
        require(cache['counts']['path_filter']==[128,20] and cache['counts']['velocity_filter']==[64,64]
                and cache['counts']['candidate_count']==1280,'Cache candidate counts changed')
        require(cache['source_sha256']['c_scene_selector.py']==SCENE_SHA,'Cache token source changed')
    curves={};saved={};artifacts={}
    for step in [0]+list(range(interval,steps+1,interval)):
        arrays,summary,artifact=read_eval(directory,step)
        if saved:
            for key in ('rows','session_index','shortlist_oracle','base_d3'):
                require(exact(arrays[key],saved[0][key]),'Frozen candidate population changed across head evaluations: '+key)
        saved[step]=arrays;curves[step]=summary;artifacts[str(step)]=artifact
    require(len(np.unique(saved[0]['session_index']))==11,'Expected original 11 tune sessions')
    train,train_summary,train_artifact=read_eval(directory,steps,prefix='train_eval')
    for label,arrays in (('train',train),('tune',saved[0])):
        names=manifest[label+'_cache_manifest']['sessions']
        ids=arrays['session_index']
        require(ids.dtype.kind in 'iu' and np.all((ids>=0)&(ids<len(names))),'Saved session index outside manifest names')
    for name,summary in (('terminal',curves[steps]),('train',train_summary)):
        require(result[name]['n']==summary['n'] and np.isclose(result[name]['official_d3'],summary['d3'],atol=2e-7,rtol=2e-6),
                'Run completion receipt differs from saved full evaluation')
    log=[json.loads(s) for s in (directory/'train.jsonl').read_text().splitlines()]
    log_steps=[int(x['step']) for x in log]
    expected_logs=[1]+list(range(50,steps+1,50))
    require(log_steps==expected_logs,'Missing, duplicated or extra training log steps')
    return {'directory':directory,'manifest':manifest,'result':result,'saved':saved,'curves':curves,
            'train':train,'train_summary':train_summary,'log':log,
            'artifacts':artifacts,'train_artifact':train_artifact,
            'manifest_sha256':sha(directory/'manifest.json'),'source_sha256':manifest['source_sha256']}


def matched_contract(real,zero):
    left,right=real['manifest'],zero['manifest']
    for key in ('source_sha256','initial_head_sha256','bank_sha256','train_cache_manifest_sha256',
                'tune_cache_manifest_sha256','rng_seed','parameters','torch','numpy'):
        require(left[key]==right[key],'Matched head contract differs: '+key)
    excluded={'mode','run_dir'}
    require({k:v for k,v in left['arguments'].items() if k not in excluded}==
            {k:v for k,v in right['arguments'].items() if k not in excluded},'Matched arguments differ beyond mode/run-dir')
    require(len(real['log'])==len(zero['log']),'Training logs differ in length')
    for a,b in zip(real['log'],zero['log']):
        for key in ('step','epoch','seen','exposure','lr','batch_rows_sha256'):
            require(a[key]==b[key],'Matched training data/schedule diverged: '+key)
    for step in real['saved']:
        a,b=real['saved'][step],zero['saved'][step]
        for key in ('rows','session_index','shortlist_oracle','base_d3'):
            require(exact(a[key],b[key]),'Matched evaluation population/candidate oracle differs: '+key)
    for key in ('pred','candidate_id','d3'):
        require(exact(real['saved'][0][key],zero['saved'][0][key]),'Matched initial decisions differ: '+key)
    require(exact(real['train']['rows'],zero['train']['rows']) and
            exact(real['train']['session_index'],zero['train']['session_index']) and
            exact(real['train']['shortlist_oracle'],zero['train']['shortlist_oracle']),
            'Full train evaluation population/candidate oracle differs')
    require(set(left['train_cache_manifest']['sessions']).isdisjoint(left['tune_cache_manifest']['sessions']),
            'Train/tune sessions overlap')
    return {'same_initial_head_sha256':left['initial_head_sha256'],'same_source_sha256':left['source_sha256'],
        'same_rows_and_oracles_all_checkpoints':True,'same_logged_training_batches':True,
        'initial_pred_and_ids_byte_equal':True,'train_tune_sessions_disjoint':True}


def reconstruct_batches(run):
    args=run['manifest']['arguments'];rows=run['train']['rows'];n=len(rows)
    rng=torch.Generator(device='cpu').manual_seed(run['manifest']['rng_seed'])
    permutation=torch.randperm(n,generator=rng);cursor=epoch=seen=0
    logs={x['step']:x for x in run['log']};digest=hashlib.sha256()
    for step in range(1,args['steps']+1):
        if cursor>=n:
            permutation=torch.randperm(n,generator=rng);cursor=0;epoch+=1
        indices=permutation[cursor:cursor+args['batch']].numpy();cursor+=len(indices);seen+=len(indices)
        batch=rows[indices];digest.update(np.asarray(batch,np.int64).tobytes())
        if step in logs:
            record=logs[step]
            require(record['batch_rows_sha256']==row_sha(batch) and record['epoch']==epoch and record['seen']==seen,
                    'Recorded batch SHA/epoch/exposure differs from deterministic full reconstruction')
    require(run['result']['seen']==seen,'Final training exposure differs')
    return {'all_steps_reconstructed':args['steps'],'all_sequence_rows_sha256':digest.hexdigest(),
            'seen':seen,'exposure':seen/n,'logged_steps_checked':len(logs),'seed':run['manifest']['rng_seed']}


def read_baseline(path,rows):
    with np.load(path,allow_pickle=False) as z:
        values={key:z[key] for key in ('rows','pred','candidate_id','d3','final_oracle','session')}
    require(exact(values['rows'],rows),'Original C diagnostic rows differ')
    return values,{'path':str(Path(path).resolve()),'sha256':sha(path),
        'd3':float(values['d3'].mean()),'oracle':float(values['final_oracle'].mean()),
        'regret':float((values['d3']-values['final_oracle']).mean())}


def make_plot(real,zero,baselines,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for run,label,color in ((real,'Real candidate tokens','#0868ac'),(zero,'Zero token control','#e6550d')):
        steps=sorted(run['curves'])
        for ax,key in zip(axes,('d3','regret')):
            ax.plot(steps,[run['curves'][s][key] for s in steps],marker='o',ms=3,label=label,color=color)
            ax.scatter([4000],[run['train_summary'][key]],marker='x',s=50,color=color,label=label+' full TRAIN')
    axes[0].axhline(baselines['C_V10']['d3'],color='.45',ls=':',label='Original C V10')
    axes[0].axhline(baselines['C_V64']['oracle'],color='.25',ls='--',label='V64 candidate oracle (GT)')
    axes[0].axhline(.15,color='.55',lw=.8,ls='-.',label='Requested 0.15 reference')
    for ax,title in zip(axes,('Saved TUNE D3','Saved TUNE selection regret')):
        ax.set(xlabel='Head optimizer updates (C frozen)',ylabel='Metres',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
    fig.suptitle('Repeated TUNE1998; fixed 4000-step terminal, one training seed',fontsize=11)
    fig.savefig(path,dpi=180);plt.close(fig)


def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def test_constant_paired_shift(self):
            values=np.full(15,-.125);sessions=np.repeat([0,1,2],[2,5,8])
            r=paired_bootstrap(values,sessions,20000,0)
            self.assertEqual(r['ci95'],[-.125,-.125]);self.assertEqual(r['mean_delta'],-.125)
        def test_cluster_pooled_weighting_and_determinism(self):
            values=np.asarray([0.,1.,1.,1.]);sessions=np.asarray([0,1,1,1])
            a=paired_bootstrap(values,sessions,20000,0);b=paired_bootstrap(values,sessions,20000,0)
            self.assertEqual(a,b);self.assertEqual(a['mean_delta'],.75);self.assertEqual(a['ci95'],[0.,1.])
        def test_signed_zero_not_byte_equal(self):
            self.assertFalse(exact(np.asarray([0.],np.float32),np.asarray([-0.],np.float32)))
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(r.wasSuccessful(),'Analysis CPU tests failed')


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--real-dir');p.add_argument('--zero-dir');p.add_argument('--output')
    p.add_argument('--v10-diagnostic',default='reports/c_refine_20260910/velocity_stage64_v1/diagnostics.npz')
    p.add_argument('--v64-diagnostic',default='reports/c_refine_20260910/velocity_64_64_v1/diagnostics.npz')
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return
    require(a.real_dir and a.zero_dir and a.output,'Both completed run directories and new output required')
    output=Path(a.output).resolve();require(not output.exists(),'Analysis output is immutable')
    real,zero=read_run(a.real_dir,'real'),read_run(a.zero_dir,'zero')
    contract=matched_contract(real,zero);order=reconstruct_batches(real)
    rows=real['saved'][0]['rows'];v10,b10=read_baseline(a.v10_diagnostic,rows);v64,b64=read_baseline(a.v64_diagnostic,rows)
    initial=real['saved'][0]
    names=np.asarray(real['manifest']['tune_cache_manifest']['sessions'])
    require(np.array_equal(names[initial['session_index']],v64['session']) and np.array_equal(v10['session'],v64['session']),
            'Saved row-to-session mapping differs from original C diagnostic')
    require(exact(initial['pred'],v64['pred']) and exact(initial['candidate_id'],v64['candidate_id']),
            'Initial cached head and independent original-C V64 B8 decisions differ')
    require(np.allclose(initial['d3'],v64['d3'],atol=2e-6,rtol=2e-6), 'Original-C V64 metric differs beyond arithmetic tolerance')
    require(np.allclose(initial['shortlist_oracle'],v64['final_oracle'],atol=2e-6,rtol=2e-6),'V64 cached oracle differs')
    sessions=real['saved'][4000]['session_index']
    pairs={}
    for left,right,delta in (('real','zero',real['saved'][4000]['d3'].astype(np.float64)-zero['saved'][4000]['d3']),
            ('real','C_V10',real['saved'][4000]['d3'].astype(np.float64)-v10['d3']),
            ('real','C_V64',real['saved'][4000]['d3'].astype(np.float64)-v64['d3']),
            ('zero','C_V10',zero['saved'][4000]['d3'].astype(np.float64)-v10['d3']),
            ('zero','C_V64',zero['saved'][4000]['d3'].astype(np.float64)-v64['d3'])):
        pairs[left+'_minus_'+right]=paired_bootstrap(delta,sessions)
    summary={'status':'completed','fixed_terminal_step':4000,'contract':contract,'batch_order':order,
        'baseline':{'C_V10':b10,'C_V64':b64},'paired_terminal_comparisons':pairs,
        'real':{'curves':real['curves'],'full_train':real['train_summary']},
        'zero':{'curves':zero['curves'],'full_train':zero['train_summary']},
        'interpretation':'Candidate tokens mix image evidence, candidate embeddings and common-perception status; real/zero difference is not pure image causality.',
        'validation':'Repeated exploratory TUNE, single training seed; no untouched holdout and no posthoc best-checkpoint selection.',
        'GT_inputs':'Saved evaluation outputs only; no candidate cache tensors, GT files, model forward or bank read',
        'fixed_bank_identity_scope':'Checked by producer/evaluator; this analyzer verifies saved IDs and provenance, not bank coordinates independently.',
        'v64_initial_pred_id_bytes_equal':True,
        'cached_vs_fp64_v64_d3_maxdiff':float(np.max(np.abs(initial['d3'].astype(np.float64)-v64['d3'])))}
    output.mkdir(parents=True);shutil.copy2(__file__,output/Path(__file__).name)
    (output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    make_plot(real,zero,summary['baseline'],output/'learning_curves.png')
    receipt={'script_sha256':sha(__file__),'torch':str(torch.__version__),'numpy':str(np.__version__),
        'arguments':vars(a),'run_manifests':{'real':real['manifest_sha256'],'zero':zero['manifest_sha256']},
        'eval_artifacts':{'real':real['artifacts'],'zero':zero['artifacts']},
        'train_artifacts':{'real':real['train_artifact'],'zero':zero['train_artifact']},
        'analysis_sha256':sha(output/'analysis.json'),'plot_sha256':sha(output/'learning_curves.png')}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    lines=['# C 최종 선택기 고정 terminal 분석','', '| 구성 | TUNE D3 | oracle | regret | TRAIN D3 |','|---|---:|---:|---:|---:|']
    for name,run in (('real token',real),('zero token',zero)):
        r=run['curves'][4000];lines.append(f"| {name} | {r['d3']:.9f} | {r['oracle']:.9f} | {r['regret']:.9f} | {run['train_summary']['d3']:.9f} |")
    pair=pairs['real_minus_zero']
    lines+=['',f"원 C V10 D3 {b10['d3']:.9f}, V64 D3 {b64['d3']:.9f}. Real−zero paired D3 {pair['mean_delta']:+.9f}, 11-session bootstrap 95% CI [{pair['ci95'][0]:+.9f}, {pair['ci95'][1]:+.9f}]. 음수는 real이 낮다는 뜻이다.",'',
        '중간 평가는 500step마다 모두 기록했으며 고정 terminal은 4000step이다. 전체 TRAIN 평가와 반복 사용된 TUNE을 구분한다. 후보 token은 시각 정보뿐 아니라 후보 embedding과 공통 인지 상태의 효과도 포함하므로 순수 영상 인과 기여로 해석하지 않는다. 단일 seed이며 독립 미사용 holdout 결과가 아니다.','',
        '초기 head 가중치 SHA·source·cache·원 C·bank·행/session/oracle·기록된 batch SHA가 일치했다. 동일 RNG로 전체 batch 순서를 재구성했다. 초기 선택 XY/ID는 독립 V64 B8 진단과 바이트 단위로 같았다. 각 waypoint p99, 전체 curve 및 paired 비교는 analysis.json에 있다.','']
    (output/'ANALYSIS_KO.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'completed','output':str(output),'terminal_real':real['curves'][4000],
                      'terminal_zero':zero['curves'][4000],'real_minus_zero':pair},allow_nan=False))


if __name__=='__main__':main()
