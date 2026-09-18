"""Actual-image parity, command mapping/mirror, forward routing, gradients and cost."""
import json
import os
import time
import numpy as np
from torch.utils.data import default_collate
from command_common import *
from command_data import (CommandDataset, commands_for_frames, encode_label,
                          LABELS, MIRROR, KEY, wrap_command_flip)
from command_model import CommandA2Model, COMMAND_PREFIX

CHECK = ('scene_features','occ_logits','lane_logits','plan_abs','motion_features','state_hat','history_hat')

def diffs(a,b,keys=CHECK):
    return {k:float((a[k].float()-b[k].float()).abs().max()) for k in keys}

def make_model(command, common):
    trainer.seed_all(1)
    cfg = MotionDriveV2Config(**trainer.initialization_configuration(common['manifest'],
        goal_on=1, state_on=1, explicit_arch='resnet50', cross_cell_goal_mode='zero', history_contract='control'))
    cfg.motion_input_mode = 'low_feature'
    model = CommandA2Model(cfg) if command else A2NominalModel(cfg, arm='A2-BASE-NOM')
    mr.rebuild_correlation_fuse(model, 4)
    state = {k:v for k,v in common['model'].items() if not k.startswith('motion_encoder.correlation_fuse.0.')}
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys
    shared = {k:v for k,v in model.state_dict().items() if not k.startswith(COMMAND_PREFIX)}
    assert tensor_state_sha256(shared) == nominal.BASE_INITIAL_SHA
    return model.cuda().eval()

def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected ValueError')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.monotonic()
    result = dict(status='running', physical_gpu=3, preflight_weights_used_for_training=False)
    ts = {'frame_id':[7,5,6,8], 'timestamp':[700.,500.,600.,800.]}
    cm = {'timestamp':[600.,500.,800.,700.], 'command':['TURN_RIGHT','LANE_KEEP','U_TURN','TURN_LEFT']}
    assert commands_for_frames(ts,cm,[5,6,7]).tolist() == [0,2,1]
    assert commands_for_frames(ts,dict(cm,command=['TURN_RIGHT','LANE_KEEP','LANE_CHANGE_L','TURN_LEFT']),[5,6,7]).tolist() == [0,2,1]
    expect_value_error(lambda: commands_for_frames(ts,cm,[4]))
    expect_value_error(lambda: commands_for_frames(dict(ts,frame_id=[5,5,6,8]),cm,[5]))
    expect_value_error(lambda: commands_for_frames(ts,dict(cm,timestamp=[600.,500.,700.,700.]),[7]))
    expect_value_error(lambda: encode_label('UNKNOWN'))
    result['raw_parser'] = dict(exact_shuffled_timestamp_join=True, future_command_invariant=True,
        rejects_missing_or_duplicate=True, unknown_label_rejected=True, no_pose_or_gt_arguments=True)

    train, _ = nominal.raw_datasets(False, 1)
    data = CommandDataset(NominalStatusDataset(mr.MotionCanvasDataset(train, 'native')))
    indices = [0, int(np.flatnonzero(data.command_ids == LABELS.index('TURN_LEFT'))[0])]
    raw = default_collate([data[i] for i in indices])
    batch = to_device(raw, torch.device('cuda:0'))
    x = inputs(batch); xb = inputs(batch, command=False)
    common = torch.load(legacy.INITIALIZER, map_location='cpu', weights_only=False)
    base, model = make_model(False, common), make_model(True, common)
    result['shared_initial_sha256'] = nominal.BASE_INITIAL_SHA
    result['preflight_train_rows'] = raw['row'].tolist()
    result['zero_init_parity'] = {}
    for mode,amp in [('FP32',False),('BF16',True)]:
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            a,b = base(**xb),model(**x)
        delta = diffs(a,b)
        assert max(delta.values()) == 0., (mode,delta)
        result['zero_init_parity'][mode] = delta
        del a,b
    from motiondrive_v2_flip_augment import flip_item
    flip = wrap_command_flip(mr.wrap_flip_item(flip_item))
    item = data[indices[1]]
    flipped = flip(item,768,384)
    twice = flip(flipped,768,384)
    for key,value in item.items():
        if isinstance(value,torch.Tensor):
            if value.is_floating_point():
                assert torch.allclose(value,twice[key],atol=1e-4,rtol=0),key
            else:
                assert torch.equal(value,twice[key]),key
    for i in range(6):
        modified = dict(item, provided_command=torch.eye(6)[i])
        assert flip(modified,768,384)[KEY].argmax().item() == MIRROR[i]
    result['flip'] = dict(all_six_labels=True, raw_item_unchanged=True,
        tensor_double_flip_parity=True, mapping=dict(zip(LABELS,[LABELS[i] for i in MIRROR])))
    weight = batch['gt_plan'].new_tensor([11,11,5,5,2,2])/36
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output = model(**x)
        loss = (torch.linalg.vector_norm(output['plan_abs']-batch['gt_plan'],dim=-1)*weight).sum(-1).mean()
    loss.backward()
    grad = model.shared_command_query.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm()>0
    result['command_PREFIX_gradient_norm'] = float(grad.norm())
    result['optimizer_group'] = dict(name='head', lr=5e-5,
        parameter_names=[n for n,p in model.named_parameters() if n.startswith(COMMAND_PREFIX)],
        requires_grad=all(p.requires_grad for p in model.shared_command_query.parameters()))
    del output,loss
    model.zero_grad(set_to_none=True)
    # Disposable nonzero conditioning demonstrates a live route without changing any candidate.
    with torch.no_grad():
        model.shared_command_query.weight.copy_(torch.linspace(-.2,.2,192,device='cuda').reshape(32,6).roll(5,0))
    seen = {}
    enc = model.scene_encoder
    hooks = [enc.occ_head.register_forward_pre_hook(lambda m,a:seen.update(occ=a[0])),
        enc.lane_head.register_forward_pre_hook(lambda m,a:seen.update(lane=a[0])),
        model.planner.register_forward_pre_hook(lambda m,a:seen.update(planner=a[0]))]
    with torch.inference_mode():
        original = model(**x)
    assert seen['occ'] is seen['lane']
    assert torch.equal(seen['occ'].flatten(2).transpose(1,2).float(),seen['planner'].float())
    assert torch.equal(original['scene_features'].float(),seen['planner'].float())
    with torch.inference_mode():
        changed = model(**dict(x,provided_command=x[KEY].roll(1,1)))
    isolation = diffs(original,changed,('motion_features','state_hat','history_hat'))
    assert max(isolation.values()) == 0
    impact = diffs(original,changed,('scene_features','plan_abs'))
    assert min(impact.values()) > 0
    result.update(command_motion_isolation=isolation, nonzero_command_route=impact,
        shared_consumers_same_tensor=True, command_value_concatenation=False)
    for h in hooks:
        h.remove()
    del original,changed,seen
    invalid = dict(x,provided_command=torch.zeros_like(x[KEY]))
    expect_value_error(lambda:model(**invalid))
    assert model._command_context is None and model._provided_status_context is None
    with torch.inference_mode():
        assert torch.isfinite(model(**x)['plan_abs']).all()
    result['exception_cleanup_and_clip_isolation'] = True
    from torch.utils.flop_counter import FlopCounterMode
    import torch.utils.module_tracker as mt
    class NoHandle:
        def remove(self):
            pass
    old = mt.register_multi_grad_hook
    mt.register_multi_grad_hook = lambda *a,**kw:NoHandle()
    result['cost'] = {}
    try:
        for name,m,inps in [('BASE',base,xb),('COMMAND',model,x)]:
            tiny = {k:v[:1] for k,v in inps.items()}
            with torch.no_grad(), FlopCounterMode(display=False) as counter:
                m(**tiny)
            times = []
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(14):
                    torch.cuda.synchronize();t=time.monotonic();m(**tiny);torch.cuda.synchronize()
                    if i>=4:
                        times.append(1000*(time.monotonic()-t))
            result['cost'][name] = dict(official_counter_flops=int(sum(counter.get_flop_counts().get('Global',{}).values())),
                B200_B1_forward_ms_median=float(np.median(times)), parameters=sum(p.numel() for p in m.parameters()),
                RTX4090_forward_ms=None)
    finally:
        mt.register_multi_grad_hook = old
    assert result['cost']['COMMAND']['official_counter_flops'] < 7053e9
    result.update(status='passed', elapsed_seconds=time.monotonic()-start,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cost_scope='complete top-level B1; B200 timing is not RTX4090 latency')
    (REPORT/'tests_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
