from pathlib import Path
import json,datetime,numpy as np
r=Path('/NHNHOME/data/sukim/adcl');out=r/'reports/md_a2_turn_analysis_20260918'
s=np.load(out/'prediction_snapshot.npz');meta=np.load(out/'analysis_metadata.npz')
g=s['BASE_NOM_gt'];p=s['MH4_NOM_pred'];w=np.array([11,11,5,5,2,2],float)/36
g0=np.concatenate([np.zeros((len(g),1,2)),g],1)
p0=np.concatenate([np.zeros((len(p),1,2)),p],1)
sg=np.concatenate([np.zeros((len(g),1)),np.cumsum(np.linalg.norm(np.diff(g0,axis=1),axis=2),axis=1)],1)
sp=np.concatenate([np.zeros((len(p),1)),np.cumsum(np.linalg.norm(np.diff(p0,axis=1),axis=2),axis=1)],1)
valid=(sp[:,1:]<=sg[:,-1,None]+1e-9)&(sg[:,-1,None]>1e-5)
g_at_pred_progress=np.zeros_like(g)
for i in range(len(g)):
 for axis in range(2):
  keep=np.r_[True,np.diff(sg[i])>1e-9]
  g_at_pred_progress[i,:,axis]=np.interp(sp[i,1:],sg[i,keep],g0[i,keep,axis])
shape=p-g_at_pred_progress
progress=g_at_pred_progress-g
assert np.max(np.abs(shape+progress-(p-g)))<1e-10
result={'scope':'Matched predicted arclength on GT piecewise-linear path; descriptive output decomposition, not causal intervention or deployable correction',
'model':'MH4_NOM step17130','mask':'predicted cumulative arclength <= GT total arclength; GT total >1e-5m',
'nonadditivity':'vector components add exactly; their Euclidean norms do not add to PREFIX',
'groups':{},'semantic_examples':[]}
for name,mask in [('left_3s',meta['vad_cmd']==1),('right_3s',meta['vad_cmd']==0),('other_3s',meta['vad_cmd']==2)]:
 weighted_mask=valid[mask]*w;den=weighted_mask.sum()
 a=float((np.linalg.norm(shape[mask],axis=-1)*weighted_mask).sum()/den)
 b=float((np.linalg.norm(progress[mask],axis=-1)*weighted_mask).sum()/den)
 c=float((np.linalg.norm((p-g)[mask],axis=-1)*weighted_mask).sum()/den)
 result['groups'][name]={'n_rows':int(mask.sum()),'valid_waypoint_fraction':float(valid[mask].mean()),
 'valid_PREFIX_weight_fraction':float(den/mask.sum()),'original_error_on_same_valid_support_m':c,
 'same_progress_shape_difference_m':a,'GT_path_progress_difference_m':b}
examples=json.loads((out/'semantic_turn_examples.json').read_text())
for example in examples:
 i=int(np.where(s['BASE_NOM_rows']==example['row'])[0][0])
 common=min(sg[i,-1],sp[i,-1])
 def interp(points,arc):
  keep=np.r_[True,np.diff(arc)>1e-9]
  return np.array([np.interp(common,arc[keep],points[keep,axis]) for axis in range(2)])
 ga=interp(g0[i],sg[i]);pa=interp(p0[i],sp[i])
 result['semantic_examples'].append({'row':example['row'],'direction':example['direction'],'selection':example['selection'],
 'gt_total_length_m':float(sg[i,-1]),'pred_total_length_m':float(sp[i,-1]),
 'common_distance_m':float(common),'gt_xy_at_common_distance':ga.tolist(),'pred_xy_at_common_distance':pa.tolist(),
 'shape_separation_at_common_distance_m':float(np.linalg.norm(pa-ga)),'lateral_difference_at_common_distance_m':float(pa[1]-ga[1])})
(out/'progress_shape_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('NOW',datetime.datetime.now().isoformat())
for arm in ['A2-BASE-NOM','A2-MH4-NOM','A2-FULL-NOM']:
 d=r/'work_dirs/md_a2_nominal_mh4_20260918'/f'{arm}-s1';m=json.loads((d/'manifest.json').read_text());e=json.loads((d/'final_eval.json').read_text())
 print(arm,m['status'],m.get('step'),e['report']['step'],e['report']['official_d3'])
