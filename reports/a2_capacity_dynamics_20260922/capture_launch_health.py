from pathlib import Path
import datetime,json,hashlib,os,subprocess,math
root=Path('/NHNHOME/data/sukim/adcl'); report=root/'reports/a2_capacity_dynamics_20260922'; runs=root/'work_dirs/a2_capacity_dynamics_20260922'
now=datetime.datetime.now(datetime.timezone.utc); kst=datetime.timezone(datetime.timedelta(hours=9))
state=json.loads((report/'runtime/main_orchestrator.json').read_text()); assert state['status']=='training'
models={};streams={};experiments={}
for job in state['jobs']:
    arm=job['arm']; os.kill(job['pid'],0)
    run=runs/f'{arm}-s1'; rows=[]
    for line in (run/'metrics.jsonl').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r.get('kind')=='train':rows.append(r)
    last=rows[-1]; assert last['step']>=100
    ref=rows[-51]; seconds=(last['elapsed_seconds']-ref['elapsed_seconds'])/(last['step']-ref['step'])
    def eta(step,extra):return (now+datetime.timedelta(seconds=max(0,step-last['step'])*seconds+extra)).astimezone(kst).isoformat()
    manifest=json.loads((run/'manifest.json').read_text());assert manifest.get('nonfinite_count',0)==0
    assert all(math.isfinite(r[k]) for r in rows for k in ('total','plan_d3','grad_norm'))
    protocol=json.loads((report/f'protocol_{arm}.json').read_text());experiments[arm]=protocol
    assert not protocol['full_fit']['enabled'] and protocol['source_commit']=='b2683cbb03f9963279f9302d37df5d33f7048d12'
    models[arm]={'gpu':job['gpu'],'pid':job['pid'],'step':last['step'],'updates':10277,
        'elapsed_seconds':last['elapsed_seconds'],'recent_seconds_per_update':seconds,'nonfinite_observed':0,
        'nonfinite_evidence':'All logged loss and grad_norm finite; live trainer aborts on a nonfinite microbatch; final manifest counter is not yet populated.',
        'first_validation_eta_kst':eta(3426,90),'terminal_eta_kst':eta(10277,300),
        'eta_assumption':'last 50 training intervals persist; 90s first / 300s combined validation and saving allowance',
        'last_train_PREFIX':last['plan_d3'],'train_score_is_not_V0':True,
        'peak_allocated_bytes':last['cuda_memory']['peak_allocated_bytes'],
        'optimizer_groups':json.loads((run/'optimizer_groups.json').read_text()),
        'gradient_audit':json.loads((run/'gradient_audit.json').read_text())[-1]}
    if arm=='C-AGENT':models[arm].update(agent_future=last['agent_future'],agent_future_weighted=last['agent_future_weighted'],actual_total_includes_agent=True)
    streams[arm]={r['step']:r['sample_order_sha256'] for r in rows}
common=set.intersection(*(set(v) for v in streams.values()));assert all(len({s[i] for s in streams.values()})==1 for i in common)
source=next(iter(experiments.values()))['source']
assert all(e['source']==source for e in experiments.values())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in source.items())
health={'status':'running_verified','checked_utc':now.isoformat(),'checked_kst':now.astimezone(kst).isoformat(),
    'source_commit':'b2683cbb03f9963279f9302d37df5d33f7048d12','parent_DEV_PREFIX':.14459766188205064,
    'initial_all_arms_PREFIX':.14459766436030752,'arms':models,
    'same_sample_stream_steps_checked':len(common),'same_sample_stream_last_step':max(common),
    'all_declared_source_hashes_match':True,'completed_smoke_not_used_as_initializer':True,
    'new_V0_results_available':False,'FULL_started':False,'official_upload':False}
p=report/'launch_health.json';p.write_text(json.dumps(health,indent=2)+'\n')
print(json.dumps(health,indent=2))
