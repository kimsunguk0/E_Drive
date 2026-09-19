"""A2 RGB distillation, matched to the completed G0 continuation recipe."""
import argparse, contextlib, json, os, shutil
from visual_teacher import *
from common import trainer, legacy, nominal, PARENT, tensor_state_sha256
import train_next as previous
import motiondrive_v2_training as mt

ARM='A2-VIS-TEACHER'
CONTROL=ROOT/'work_dirs/a2_next_20260919/A2-G0-s1'

class Factory:
    VALID_ARMS={ARM:0,'A2-G0':0}
    def __new__(cls,config,*,arm):
        assert arm==ARM
        return VisualStudent(config)


@contextlib.contextmanager
def runtime(d,run,smoke,protocol,teacher,coefficient):
    with legacy.patched_runtime(ARM,1,d,run,smoke):
        old_loader=trainer._load_initial_model_state;old_validator=trainer._validate_experimental_runtime
        old_loss=trainer.compute_loss;old_json=trainer.atomic_json;old_norm=mt.build_loss_normalizers
        def initialize(model,common,experiment=None):
            expected=d['initializer']['model_state_sha256']
            assert tensor_state_sha256(common['model'])==expected
            load_student_parent(model,common['model'])
            actual=tensor_state_sha256(model.state_dict())
            d['expected_initial_model_state_sha256']=actual
            d['initial_load']={'strict_parent_plus_projector':True,
                'shared_parent_state_sha256':expected,'measured_model_state_sha256':actual}
            protocol.write_text(json.dumps(d,indent=2)+'\n')
            return {'initial_load':d['initial_load']}
        def validate(args,value):
            assert args.warmup==100 and args.init==str(PARENT)
            args.warmup=200
            try:old_validator(args,value)
            finally:args.warmup=100
        def normalizers(raw):
            values=old_norm(raw)
            if coefficient:
                target,valid=teacher_targets(teacher,raw['images'])
                raw['_visual_target']=target;raw['_visual_mask']=valid
                values['visual_count']=valid.sum()
            return values
        def loss(output,batch,weights,**kwargs):
            visual_count=None
            if coefficient:
                kwargs=dict(kwargs)
                visual_count=kwargs['normalizers']['visual_count']
                kwargs['normalizers']={k:v for k,v in kwargs['normalizers'].items() if k!='visual_count'}
            total,parts=old_loss(output,batch,weights,**kwargs)
            auxiliary=weights.occupancy*parts['occ_bce']+weights.lane*parts['lane_bce']+weights.motion*parts['motion']
            # Eval metrics do not call this loss; training captures exactly one scene pass.
            vis=torch.zeros((),device=output['plan_abs'].device)
            if coefficient:
                vis=visual_loss(output['_visual_feature'],active_model[0].visual_projector,
                    batch['_visual_target'],batch['_visual_mask'],visual_count)
                total=total+coefficient*vis
            return total,dict(parts,total=total,weighted_auxiliary=auxiliary,
                visual_cosine=vis,weighted_visual=coefficient*vis)
        active_model=[]
        original_init=initialize
        def initialize_with_reference(model,common,experiment=None):
            result=original_init(model,common,experiment);active_model.append(model);return result
        def write_json(path,payload):
            old_json(path,payload)
            if Path(path).name=='final_eval.json' and payload.get('records'):
                step=payload['report']['step']
                old_json(run/f'diagnostics_step{step}.json',nominal.diagnostics(payload['records']))
                old_json(run/f'predictions_step{step}.json',payload)
        trainer._load_initial_model_state=initialize_with_reference
        trainer._validate_experimental_runtime=validate
        trainer.compute_loss=loss;trainer.atomic_json=write_json;mt.build_loss_normalizers=normalizers
        try:yield
        finally:
            trainer._load_initial_model_state=old_loader;trainer._validate_experimental_runtime=old_validator
            trainer.compute_loss=old_loss;trainer.atomic_json=old_json;mt.build_loss_normalizers=old_norm


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=range(4),required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--zero-smoke',action='store_true')
    args=p.parse_args();assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    torch.set_num_threads(4);run=Path(args.run_dir).resolve()
    assert not run.exists() or not any(run.iterdir()),'Never overwrite any run'
    preflight=json.loads((REPORT/'visual_preflight.json').read_text())
    coefficient=0. if args.zero_smoke else preflight['lambda_vis']
    teacher=None if args.zero_smoke else load_teacher()
    teacher_hash=None if teacher is None else tensor_state_sha256(teacher.state_dict())
    with previous.configure('A2-G0'):
        legacy.SharedDynamicsMotionDriveV2=Factory;legacy.REPORT_DIR=REPORT
        d=previous.declaration('A2-G0',args.gpu,run,args.zero_smoke)
        d.update(name='a2_visual_teacher_20260919',arm=ARM,
            question='training-only RGB spatial supervision, identical deployed student',
            visual_teacher=teacher_manifest(),lambda_vis=coefficient,
            coefficient_selection={'rule':preflight['rule'],'report_sha256':sha(REPORT/'visual_preflight.json')},
            control={'run':str(CONTROL),'reuse':'same parent, complete G0 recipe and row stream; checked in receipt',
                'PREFIX':.16508406852237295,'parent_PREFIX':.16425176970362365},
            train_only_state_prefix=PROJECTOR_PREFIX)
        for path in (Path(__file__),Path(__file__).with_name('visual_teacher.py')):
            d['source'][str(path.relative_to(ROOT))]=sha(path)
        protocol=REPORT/f'protocol_{run.name}.json'
        assert not protocol.exists()
        protocol.write_text(json.dumps(d,indent=2)+'\n')
        print('PLAN '+json.dumps({'run':str(run),'coefficient':coefficient,'recipe':d['recipe']}),flush=True)
        with runtime(d,run,args.zero_smoke,protocol,teacher,coefficient):
            trainer.run_training(previous.argv_for('A2-G0',run,args.zero_smoke),experiment=d)
        (run/'experiment.json').write_text(json.dumps(d,indent=2)+'\n')
        if teacher is not None:
            assert teacher_hash==tensor_state_sha256(teacher.state_dict())
            assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
            (run/'teacher_unchanged.json').write_text(json.dumps({'sha256':teacher_hash,'unchanged':True})+'\n')
        shutil.copy2(ROOT/'reports/a2_next_20260919/parent_initial_eval.json',run/'initial_eval.json')

if __name__=='__main__':main()
