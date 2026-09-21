"""Independent replay of provided DEV statistics plus FULL in-fit intervals."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path('/NHNHOME/data/sukim/adcl')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--reference-analysis',required=True);a=ap.parse_args()
    original=json.loads(Path(a.reference_analysis).read_text());result={};identity=None
    for name,rel in [('DIRECT','work_dirs/a2_progress_h4_20260920/A2-H4-DIRECT-s1/final_eval.json'),('PROGRESS','work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json'),('FULL_infit','work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/final_eval.json')]:
        path=ROOT/rel;rows=json.loads(path.read_text())['records'];rows.sort(key=lambda r:(r['session'],r['scenario'],r['frame']))
        keys=[(r['row'],r['session'],r['scenario'],r['frame'],r['gt_abs_xy']) for r in rows]
        if identity is None:identity=keys
        else:assert keys==identity
        pred=np.asarray([r['pred_abs_xy'] for r in rows],np.float64);gt=np.asarray([r['gt_abs_xy'] for r in rows],np.float64)
        delta=lambda v:np.diff(np.concatenate((np.zeros_like(v[:,:1]),v),1),axis=1)
        dp,dg=delta(pred),delta(gt);lp,lg=np.linalg.norm(dp,axis=-1),np.linalg.norm(dg,axis=-1)
        error=np.linalg.norm(dp-dg,axis=-1);length=abs(lp-lg)
        th=np.arctan2(dp[...,1],dp[...,0])-np.arctan2(dg[...,1],dg[...,0]);th=np.arctan2(np.sin(th),np.cos(th))
        mask=(lp>.01)&(lg>.01);angular=2*lp*lg*(1-np.cos(th));assert np.allclose(error**2,length**2+angular)
        groups=np.array([r['bucket'] for r in rows]);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if name in original['DEV']:
            ref=original['DEV'][name];assert digest==ref['sha256']
            assert np.allclose(error.mean(0),ref['displacement_vector_mae_m_by_interval'],atol=1e-14,rtol=0)
        result[name]={'source':rel,'source_sha256':digest,'rows':len(rows),
            'vector_MAE_per_interval':error.mean(0).tolist(),'vector_MAE_by_second':error.mean(0).reshape(3,2).mean(1).tolist(),
            'length_MAE_per_interval':length.mean(0).tolist(),'length_signed_mean_per_interval':(lp-lg).mean(0).tolist(),
            'heading_MAE_deg_provided_mask':[(abs(th[:,j][mask[:,j]]).mean()*180/np.pi) for j in range(6)],
            'mean_LEN':float(length.mean()),'mean_VEC':float(error.mean()),'VEC_to_LEN_loss_ratio':float(error.mean()/length.mean()),
            'mean_square_radial':(length**2).mean(0).tolist(),'mean_square_angular':angular.mean(0).tolist(),
            'groups':{k:{'rows':int((groups==k).sum()),'vector_MAE_by_second':error[groups==k].mean(0).reshape(3,2).mean(1).tolist(),
                'mean_LEN':float(length[groups==k].mean()),'mean_VEC':float(error[groups==k].mean())} for k in sorted(set(groups))}}
    result['scope']={'FULL':'in-fit V0; not server per-row statistics','DEV':'same repeated-development V0, no new model inference',
        'angle_mask':'Both GT and predicted segment length >0.01m, reproduces supplied proposal; differs from earlier GT-only >0.05m positional projection mask.',
        'squared_decomposition':'Exact per-interval vector-error square identity, not additive official PREFIX.',
        'provided_DEV_hashes_and_values_reproduced':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('Provided DEV hashes and values reproduced; FULL in-fit diagnostic added.')
if __name__=='__main__':main()
