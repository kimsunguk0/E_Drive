"""P3 CPU 계약: planner-only backward, immutable encoder, 고정 두 팔 데이터 순서."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts import train_motiondrive_v2_query_adapter as training
from scripts.motiondrive_v2_training import LossWeights, compute_loss, tensor_state_sha256


class TinyPlanner(nn.Module):
    def __init__(self, on=True):
        super().__init__()
        self.xy = nn.Linear(2, 2)
        self.query_adapter = nn.Sequential(nn.Linear(22, 4), nn.GELU(), nn.Linear(4, 2))
        self.on = on

    def forward(self, scene, motion, state, history):
        result = self.xy(scene)
        if self.on:
            result = result + self.query_adapter(torch.cat([state, history.flatten(1)], -1))[:, None]
        return result.float()


class TinyModel(nn.Module):
    def __init__(self, on=True):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.planner = TinyPlanner(on)
        self.parts_calls, self.full_calls, self.modes = 0, 0, []

    def forward_parts(self, images, history_images, lidar2img, history_transforms, time_offsets, goal_xy):
        self.parts_calls += 1
        self.modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        value = self.encoder(images[:, 0, :2, 0, 0])
        n = len(value)
        state = torch.cat([value] * 3, 1)
        history = torch.cat([value] * 2, 1)[:, None].expand(n, 4, 4)
        return {"scene_features": value[:, None].expand(n, 6, 2),
                "motion_features": value[:, None], "state_hat": state, "history_hat": history,
                "state_logvar": torch.zeros(n, 5), "history_logvar": torch.zeros(n, 4, 4),
                "occ_logits": value[:, :1, None, None].expand(n, 1, 3, 4),
                "lane_logits": value[:, 1:, None, None].expand(n, 1, 3, 4)}

    def plan_from_features(self, *args):
        return self.planner(*args)

    def forward(self, **inputs):
        self.full_calls += 1
        parts = self.forward_parts(**inputs)
        return {**parts, "plan_abs": self.plan_from_features(parts["scene_features"],
                parts["motion_features"], parts["state_hat"], parts["history_hat"])}


def batch(n=5, empty=False):
    generator = torch.Generator().manual_seed(19)
    rand = lambda *shape: torch.randn(*shape, generator=generator)
    result = {"images": rand(n,6,3,1,1), "history_images": rand(n,4,3,1,1),
              "lidar2img": torch.eye(4).expand(n,6,4,4), "history_transforms": torch.eye(4).expand(n,4,4,4),
              "time_offsets": torch.ones(n,4), "goal_xy": rand(n,2), "gt_plan": rand(n,6,2),
              "plan_valid": torch.ones(n,6,dtype=torch.bool),
              "state_target": rand(n,6), "state_valid": torch.ones(n,6,dtype=torch.bool),
              "history_target": rand(n,4,4), "history_valid": torch.ones(n,4,4,dtype=torch.bool),
              "occ_target": torch.zeros(n,1,3,4), "occ_valid": torch.ones(n,1,3,4,dtype=torch.bool),
              "lane_target": torch.ones(n,1,3,4), "lane_valid": torch.ones(n,1,3,4,dtype=torch.bool),
              "row": torch.arange(n), "scenario": ["train"] * n, "session_id": ["session"] * n,
              "frame": torch.arange(n) + 30}
    result["state_target"][:,5] = torch.arange(n) % 2
    result["occ_target"][:2] = 1
    result["occ_valid"][-1] = False
    result["state_valid"][::2,:3] = False
    result["history_valid"][::2,0] = False
    result["plan_valid"][1,0] = False
    if empty:
        for key in ("plan_valid", "state_valid", "history_valid", "occ_valid", "lane_valid"):
            result[key].zero_()
    return result


@pytest.mark.parametrize("on", [False, True])
def test_only_planner_is_trainable_and_optimizer_owned(on):
    model = TinyModel(on)
    named = training.configure_planner_training(model)
    optimizer = torch.optim.AdamW([p for _,p in named])
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.planner.parameters()}
    assert all(n.startswith("planner.") for n,_ in named)
    assert not model.training and not model.encoder.training and not model.encoder[1].training
    assert model.planner.training


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("chunk", [1,2,3,5])
@pytest.mark.parametrize("uncertainty", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_microbatch_loss_gradients_match_full_logical_batch(on, chunk, uncertainty, empty):
    torch.manual_seed(43)
    model = TinyModel(on)
    training.configure_planner_training(model)
    reference = copy.deepcopy(model)
    data, weights = batch(empty=empty), LossWeights(uncertainty=uncertainty)
    frozen = training.frozen_state_sha(model)
    out = training.training_forward(reference, data, torch.device("cpu"))
    loss, parts = compute_loss(out, data, weights)
    loss.backward()
    actual = training.backward_logical_batch(model, data, torch.device("cpu"), weights,
                                             microbatch=chunk, reserve_mib=0)
    for key in parts:
        torch.testing.assert_close(actual[key],parts[key],atol=2e-6,rtol=2e-6)
    for (name,p), (other,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-6)
    assert model.parts_calls == (len(data["images"])+chunk-1)//chunk and model.full_calls == 0
    assert model.modes == [(False,False)] * model.parts_calls
    training.assert_frozen(model, frozen)


def test_step_preserves_frozen_weights_and_bn_buffers():
    model = TinyModel()
    named = training.configure_planner_training(model)
    frozen, before = training.frozen_state_sha(model), tensor_state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW([p for _,p in named],lr=5e-5,weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    training.backward_logical_batch(model,batch(),torch.device("cpu"),LossWeights(),reserve_mib=0)
    optimizer.step()
    training.assert_frozen(model,frozen)
    assert tensor_state_sha256(model.state_dict()) != before


@pytest.mark.parametrize("mode", ["parameter", "buffer", "gradient", "requires_grad"])
def test_frozen_guard_rejects_mutations(mode):
    model=TinyModel(); training.configure_planner_training(model)
    frozen=training.frozen_state_sha(model)
    parameter=next(model.encoder.parameters())
    if mode=="parameter":
        with torch.no_grad(): parameter.add_(1)
    elif mode=="buffer": model.encoder[1].running_mean.add_(1)
    elif mode=="gradient": parameter.grad=torch.ones_like(parameter)
    else: parameter.requires_grad_(True)
    with pytest.raises(RuntimeError): training.assert_frozen(model,frozen)


def initial_payload():
    return {"step":3000,"manifest":{"split_sha256":training.SPLIT_SHA,
        "supervision_manifest_sha256":training.SUPERVISION_SHA,
        "time_input":"nominal","time_input_policy":training.time_input_policy("nominal"),
        "arguments":{"time_input":"nominal"},"model_config":{"goal_on":True,"state_on":True,
        "motion_input_mode":"low_feature","plan_output_scale":[10.,5.]}}}


@pytest.mark.parametrize("field", ["step","split","supervision","time","arguments","policy","goal","state","mode","scale"])
def test_strict_initial_lineage_rejects_wrong_checkpoint_contract(field):
    payload=initial_payload(); training.validate_initial_payload(payload)
    manifest=payload["manifest"]
    if field=="step": payload["step"]=1000
    elif field=="split": manifest["split_sha256"]="bad"
    elif field=="supervision": manifest["supervision_manifest_sha256"]="bad"
    elif field=="time": manifest["time_input"]="raw"
    elif field=="arguments": manifest["arguments"]["time_input"]="raw"
    elif field=="policy": manifest["time_input_policy"]={}
    elif field in ("goal","state"): manifest["model_config"][field+"_on"]=False
    elif field=="mode": manifest["model_config"]["motion_input_mode"]="legacy"
    else: manifest["model_config"]["plan_output_scale"]=[1.,1.]
    with pytest.raises(ValueError): training.validate_initial_payload(payload)


def test_independent_loader_generators_keep_arm_order_despite_global_rng():
    dataset=[{"row":i} for i in range(43)]
    first,_=training.training_loader(dataset,batch_size=4,workers=0)
    a=[v["row"].tolist() for v in first]
    torch.randn(137)
    second,_=training.training_loader(dataset,batch_size=4,workers=0)
    assert a == [v["row"].tolist() for v in second]


@pytest.mark.parametrize("raw_uuid", ["GPU-example","example"])
def test_expected_single_uuid_mapping_normalizes_only_prefix(monkeypatch,raw_uuid):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-example")
    monkeypatch.setenv("MOTIONDRIVE_EXPECTED_GPU_UUID","GPU-example")
    monkeypatch.setenv("MOTIONDRIVE_EXPECTED_PHYSICAL_GPU","4")
    monkeypatch.setattr(torch.cuda,"device_count",lambda:1)
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:None)
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _:SimpleNamespace(uuid=raw_uuid))
    mapping=training.validate_cuda_namespace(torch.device("cuda:0"))
    assert mapping["physical_gpu"]==4 and mapping["observed_uuid"]=="GPU-example"


@pytest.mark.parametrize("field,value", [("CUDA_VISIBLE_DEVICES","GPU-a,GPU-b"),
    ("MOTIONDRIVE_EXPECTED_GPU_UUID",""),("MOTIONDRIVE_EXPECTED_PHYSICAL_GPU","6")])
def test_namespace_failure_before_cuda_activation(monkeypatch,field,value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-a")
    monkeypatch.setenv("MOTIONDRIVE_EXPECTED_GPU_UUID","GPU-a")
    monkeypatch.setenv("MOTIONDRIVE_EXPECTED_PHYSICAL_GPU","4")
    monkeypatch.setenv(field,value)
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:pytest.fail("금지 장치 활성화"))
    with pytest.raises(ValueError):training.validate_cuda_namespace(torch.device("cuda:0"))


def test_full_forward_evaluation_does_not_use_training_parts_wrapper():
    model=TinyModel(); data=batch(); data["plan_valid"].fill_(True)
    result,_=training.evaluate(model,[data],torch.device("cpu"),"bf16",time_input="nominal")
    assert model.full_calls==1 and result["n"]==5


def test_schedule_and_cli_do_not_expose_run_dependent_training_knobs():
    argv=["--init","init.pth","--expected-init-sha256",training.INIT_SHA,
          "--adapter-on","1","--run-dir","new"]
    parsed=training.arguments(argv)
    assert parsed.device=="cuda:0" and parsed.adapter_on==1
    assert training.learning_rate(0)==pytest.approx(1e-6)
    assert training.learning_rate(50)==pytest.approx(5e-5)
    assert training.learning_rate(999)<1e-9
    for extra in (["--steps","1"],["--batch","2"],["--lr","1"],["--device","cuda:1"]):
        with pytest.raises(SystemExit):training.arguments(argv+extra)


def test_real_migration_both_arms_share_initial_state_and_strict_restore(tmp_path,monkeypatch):
    import models.motiondrive_v2.model as legacy_module
    from models.motiondrive_v2 import MotionDriveV2, MotionDriveV2Config
    from models.motiondrive_v2_query_adapter import migrate_legacy_checkpoint, model_from_query_payload
    class Backbone(nn.Module):
        def __init__(self,channels,arch):
            super().__init__(); self.layer=nn.Conv2d(3,channels,1)
    monkeypatch.setattr(legacy_module,"ResNet34FPN128",Backbone)
    cfg=MotionDriveV2Config(channels=16,backbone_arch="resnet34",grid_size=(2,2),
        motion_grid=(2,2),correlation_channels=4,scene_attention_channels=4,
        planner_layers=1,motion_input_mode="low_feature",plan_output_scale=(10.,5.))
    base=MotionDriveV2(cfg)
    payload=initial_payload(); payload["model"]=base.state_dict()
    payload["manifest"]["model_config"]=cfg.to_dict()
    path=tmp_path/"original.pth"; torch.save(payload,path)
    sha=training.sha256(path)
    models=[]
    for flag in (False,True):
        model,report=migrate_legacy_checkpoint(path,sha,adapter_seed=0,adapter_on=flag)
        models.append(model)
        assert report["source_checkpoint_sha256"]==sha
        assert report["legacy_tensors_bitwise_preserved"] is True
        training.configure_planner_training(model)
        metadata={"architecture":training.ARCHITECTURE,"query_adapter_on":flag,"model_config":cfg.to_dict()}
        checkpoint={**metadata,"model":model.state_dict(),"manifest":dict(metadata),"step":1000,"rng":{}}
        restored=model_from_query_payload(checkpoint)
        assert tensor_state_sha256(restored.state_dict())==tensor_state_sha256(model.state_dict())
        checkpoint["manifest"]["query_adapter_on"]=not flag
        with pytest.raises(ValueError):model_from_query_payload(checkpoint)
    assert tensor_state_sha256(models[0].state_dict())==tensor_state_sha256(models[1].state_dict())
    assert training.frozen_state_sha(models[0])==training.frozen_state_sha(models[1])
    assert training.sha256(path)==sha


def test_source_snapshot_keeps_new_files_inside_hash_mapping():
    snapshot=training.source_snapshot()
    assert len(snapshot["file_sha256"])==17
    for name in ("scripts/train_motiondrive_v2_query_adapter.py","models/motiondrive_v2_query_adapter.py",
                 "scripts/run_motiondrive_v2_query_trial.py"):
        assert snapshot["file_sha256"][name]==training.sha256(training.ROOT/name)


def test_existing_run_refused_before_any_cuda_or_checkpoint_migration(tmp_path,monkeypatch):
    monkeypatch.setattr(training,"validate_data_paths",lambda _: {})
    monkeypatch.setattr(training,"validate_initial_reference",lambda: {})
    monkeypatch.setattr(training,"source_snapshot",lambda: {})
    monkeypatch.setattr(training,"validate_cuda_namespace",lambda _:pytest.fail("기존 run 실행 금지"))
    run=tmp_path/"existing"; run.mkdir(); marker=run/"keep.txt"; marker.write_text("preserved")
    with pytest.raises(FileExistsError):
        training.main(["--init","original.pth","--expected-init-sha256",training.INIT_SHA,
                       "--adapter-on","0","--run-dir",str(run)])
    assert marker.read_text()=="preserved" and list(run.iterdir())==[marker]


def test_numerics_policy_matches_original_p2_without_forward(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn,"benchmark",True)
    monkeypatch.setattr(torch.backends.cudnn,"deterministic",False)
    monkeypatch.setattr(torch.backends.cuda.matmul,"allow_tf32",True)
    policy=training.configure_numerics()
    assert policy=={"cudnn_benchmark":False,"cudnn_deterministic":True,
                   "matmul_allow_tf32":False,"precision":"bf16","planner_precision":"float32"}


@pytest.mark.parametrize("field,value", [("n",1997),("n_sessions",10),("time_input","raw"),
    ("official_d3",np.nextafter(training.INITIAL_D3,np.inf)),("official_d3",float("nan"))])
def test_initial_metric_gate_has_no_tolerance_or_score_repair(field,value):
    report={"n":1998,"n_sessions":11,"time_input":"nominal","official_d3":training.INITIAL_D3}
    assert training.initial_metric_gate(report)["passed"]
    report[field]=value
    with pytest.raises(ValueError):training.initial_metric_gate(report)


def test_unfrozen_old_baseline_is_not_substituted_for_frozen_context():
    with pytest.raises(ValueError):
        training.initial_metric_gate({"n":1998,"n_sessions":11,"time_input":"nominal",
                                      "official_d3":training.UNFROZEN_INITIAL_D3})


def test_frozen_reference_is_sha_pinned_before_read(tmp_path,monkeypatch):
    monkeypatch.setattr(training,"ROOT",tmp_path)
    path=tmp_path/"reports/p3_query_frozen_full_tune_gpu5.json"
    path.parent.mkdir()
    path.write_text("{}")
    with pytest.raises(ValueError,match="SHA"):
        training.validate_initial_reference()


def test_dataset_inventory_requires_original_rows_and_session_count():
    import hashlib
    rows=np.array([0,1,2,3],dtype=np.int64)
    dataset=SimpleNamespace(rows=rows,scene_names=np.array(["a","a","b","b"]),
                            manifest={"scene_to_session":{"a":"s0","b":"s1"}})
    expected=hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
    assert training.dataset_inventory(dataset,expected,(4,2,2))["rows_sha256"]==expected
    with pytest.raises(ValueError):training.dataset_inventory(dataset,expected,(4,2,1))
    dataset.rows=rows[::-1]
    with pytest.raises(ValueError):training.dataset_inventory(dataset,expected,(4,2,2))


def full_records():
    records=[{"scenario":f"scene{s}","session":f"session{s%11}","frame":30+5*f,
              "d3":float(s+f)/100,"proxy":1.} for s in range(37) for f in range(54)]
    ids=[(row["scenario"],row["session"],row["frame"]) for row in records]
    report={"n":1998,"official_d3":float(np.mean([row["d3"] for row in records]))}
    return report,records,ids


@pytest.mark.parametrize("change", ["duplicate","order","missing","metric","nan","session","extra"])
def test_full_tune_records_identity_and_aggregate_are_strict(change):
    report,records,ids=full_records()
    training.validate_eval_records(report,records,ids)
    if change=="duplicate":records[-1]=records[0]
    elif change=="order":records.reverse()
    elif change=="missing":records.pop()
    elif change=="metric":report["official_d3"]+=1e-9
    elif change=="nan":records[0]["d3"]=float("nan")
    elif change=="session":records[0]["session"]="foreign"
    else:records[0]["new_field"]=1
    with pytest.raises(ValueError):training.validate_eval_records(report,records,ids)


def test_new_evaluation_file_refuses_overwrite(tmp_path):
    path=tmp_path/"eval_step0000.json"
    training.write_new_evaluation(path,{"step":0})
    before=path.read_bytes()
    with pytest.raises(FileExistsError):training.write_new_evaluation(path,{"step":250})
    assert path.read_bytes()==before
