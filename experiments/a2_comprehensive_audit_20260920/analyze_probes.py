"""Compare identical train probes and frozen interventions without fitting."""
from pathlib import Path
import hashlib,importlib.util,json,numpy as np
ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_comprehensive_audit_20260920'
spec=importlib.util.spec_from_file_location('old_diagnosis',ROOT/'experiments/a2_error_diagnosis_20260920/analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
W=np.array([11,11,5,5,2,2])/36

def main():
    out={};arrays={};sources={}
    for arm in ['QREFINE','FRESH','CONT_SELECTED','CONT_TERMINAL','TEMPORAL']:
        folder=ROOT/'reports/a2_error_diagnosis_20260920' if arm in ['QREFINE','FRESH'] else REPORT
        out[arm]={}
        for scope in ['train_probe','V0']:
            path=folder/f'{arm}_{scope}.npz';sources[str(path)]=old.digest(path)
            with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            d=old.diagnose(a,a['ON']);speed=np.linalg.norm(old.segments(a['gt'])[:,:4],axis=-1).mean(1)*2;score=np.linalg.norm(a['ON']-a['gt'],axis=-1)@W
            d['nonstop_speed_bins']={label:old.subset_stats(score,(a['bucket']=='nonstop')&mask,len(score)) for label,mask in [('lt5',speed<5),('5to10',(speed>=5)&(speed<10)),('ge10',speed>=10)]}
            if scope=='train_probe':
                arrays[arm]=a
                if arm!='QREFINE':assert np.array_equal(a['row'],arrays['QREFINE']['row']) and np.array_equal(a['gt'],arrays['QREFINE']['gt'])
            out[arm][scope]=d
    for name,arms in [('QREFINE_CONT',['QREFINE','CONT_SELECTED']),('THREE_EQUAL',['QREFINE','CONT_SELECTED','TEMPORAL'])]:
        a=dict(arrays[arms[0]]);p=np.mean([arrays[n]['ON'] for n in arms],axis=0);d=old.diagnose(a,p);del d['current_stop_readout'];out[name]={'train_probe':d}
    sensitivity={}
    for arm in ['CONT_SELECTED','TEMPORAL']:
        with np.load(REPORT/f'{arm}_V0.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        plus,minus=a['STATUS_VX_PLUS_0P1'],a['STATUS_VX_MINUS_0P1'];dp=old.segments(a['ON']);tangent=dp/np.maximum(np.linalg.norm(dp,axis=-1)[...,None],1e-8)
        response=((plus-minus)*tangent).sum(-1)/.2
        sensitivity[arm]={'longitudinal_plan_response_m_per_mps_by_point_mean':response.mean(0).tolist(),'response_abs_mean':abs(response).mean(0).tolist(),'fraction_positive_by_point':(response>0).mean(0).tolist(),'unit':'m of output displacement / m/s of scene-query provided vx','interpretation':'Local sensitivity only; not direct integration, a performance cap, or legal approval.'}
    result={'models':out,'scene_status_sensitivity':sensitivity,'same_1024_train_rows_and_GT':True,'sources_sha256':sources,'script_sha256':old.digest(Path(__file__)),'limitations':['Train/V0 have different scenes, not a causal generalization-gap decomposition.','Coarse speed bins do not fully equalize scene difficulty.','No fit or model change; all interventions are OOD.']}
    (REPORT/'train_and_intervention_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'train_PREFIX':{a:d['train_probe']['PREFIX'] for a,d in out.items()},'same_train_rows':True}))

if __name__=='__main__':main()
