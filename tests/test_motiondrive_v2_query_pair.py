"""P3 고정 protocol 합성 CPU 검증; 실제 결과/GT를 읽지 않는다."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_motiondrive_v2_query_pair as analysis


def report_rows(delta=0.):
    rows=[{"scenario":f"scene{i:02d}","session":f"session{i%11:02d}","frame":30+5*f,
           "d3":.4+i*.002+f*.0001+delta,"proxy":1.+i} for i in range(37) for f in range(54)]
    sessions={s:float(np.mean([r["d3"] for r in rows if r["session"]==s]))
              for s in dict.fromkeys(r["session"] for r in rows)}
    return {"n":1998,"n_sessions":11,"time_input":"nominal",
            "official_d3":float(np.mean([r["d3"] for r in rows])),"session_d3":sessions,
            "session_mean_d3":float(np.mean(list(sessions.values())))},rows


def bundle(arm,delta=0.):
    flag=arm=="state"; gpu=5 if flag else 4
    run=Path(f"/synthetic/work_dirs/motiondrive_v2/p3_query_{arm}_s0")
    source={"git_sha":"a"*40,"tracked_changes":[],"file_sha256":{f"file{i}.py":"f"*64 for i in range(17)}}
    m={"status":"completed","step":1000,"nonfinite_count":0,"received_signal":None,"source_unchanged":True,
       "architecture":analysis.ARCH,"query_adapter_on":flag,"init_checkpoint_sha256":analysis.INIT_SHA,
       "split_sha256":analysis.SPLIT_SHA,"supervision_manifest_sha256":analysis.SUP_SHA,
       "canonical_calibration_sha256":analysis.CAL_SHA,"time_input":"nominal","time_input_policy":{"mode":"nominal"},
       "schedule":copy.deepcopy(analysis.SCHEDULE),"loss_weights":copy.deepcopy(analysis.LOSS),
       "numerics":copy.deepcopy(analysis.NUMERICS),"model_config":{"goal_on":True,"state_on":True,
       "motion_input_mode":"low_feature","plan_output_scale":[10.,5.]},
       "initial_model_sha":"1"*64,"frozen_initial_sha256":"2"*64,"frozen_final_sha256":"2"*64,
       "frozen_state_sha_final":"2"*64,"train_rows_sha256":"3"*64,"eval_rows_sha256":"4"*64,
       "sample_order_sha256":"5"*64,"data_counts":{"train":54810,"eval":1998},
       "dataset_inventory":{"train":{"n":54810,"scenes":203,"sessions":72,"rows_sha256":"3"*64},
                            "eval":{"n":1998,"scenes":37,"sessions":11,"rows_sha256":"4"*64}},
       "initial_metric_gate":{"passed":True,"expected_official_d3":analysis.INITIAL_D3,
                              "observed_official_d3":analysis.INITIAL_D3,"absolute_tolerance":0.,
                              "reference_sha256":analysis.INITIAL_REFERENCE_SHA},
       "initial_reference":{"path":"/synthetic/proof.json","sha256":analysis.INITIAL_REFERENCE_SHA},
       "migration":{"initial_model_state_sha256":"1"*64,"source_checkpoint_sha256":analysis.INIT_SHA,
                    "source_checkpoint_step":3000,"adapter_seed":0,"query_adapter_on":flag,
                    "legacy_tensors_bitwise_preserved":True},
       "trainable_names":["planner.a","planner.query_adapter.0.weight"],"trainable_parameters":123,
       "git_sha":"a"*40,"source":source,"pid":2000000001+gpu,"run_dir":str(run),
       "device_mapping":{"physical_gpu":gpu,"logical_device":"cuda:0","observed_uuid_raw":f"gpu{gpu}",
                         "observed_uuid":f"GPU-gpu{gpu}","cuda_visible_devices":f"GPU-gpu{gpu}"},
       "data":{},"forward_contract":"동결 encoder를 매번 계산","auxiliary_loss_contract":"상수",
       "sample_order_policy":"같은 seed0","selection":"LAST1000","arguments":{"adapter_on":int(flag),
       "run_dir":str(run),"init":"common"},"best_step":750,"best_metric":.3}
    supervisor_path=run.parents[2]/"logs/motiondrive_v2"/(run.name+".supervisor.json")
    s={"status":"process_exited","outcome":"completed_cleanly","actual_returncode":0,
       "supervisor_exit_code":0,"termination_signal":None,"pressure_event":None,"received_signals":[],
       "parent_pid":2000000011+gpu,"child_pid":m["pid"],
       "request":{"arm":arm,"adapter_on":flag,"run_dir":str(run),"manifest_path":str(run/"manifest.json"),
                  "record":str(supervisor_path),"physical_gpu":gpu,"initializer_sha256":analysis.INIT_SHA,
                  "expected_commit":"a"*40,"gpu_uuid":f"GPU-gpu{gpu}"},
       "completion_evidence":{"frozen_state_sha256":"2"*64,"manifest_sha256":"6"*64,"last_sha256":"7"*64}}
    report,rows=report_rows(delta)
    e={"step":1000,"time_input":"nominal","architecture":analysis.ARCH,"query_adapter_on":flag,
       "source":copy.deepcopy(source),"report":report,"records":rows}
    return {"manifest":m,"supervisor":s,"evaluation":e}


def test_constant_improvement_screening_and_ratio_do_not_use_proxy():
    a,b=bundle("control"),bundle("state",-.02)
    result=analysis.analyze(a,b)
    assert result["screening"]["passed"] is True
    assert result["primary"]["state_minus_control"]["delta"]==pytest.approx(-.02)
    np.testing.assert_allclose(result["primary"]["state_minus_control"]["ci95"],[-.02,-.02],atol=1e-15)
    assert result["protocol"]["proxy_used"] is False
    assert sum(row["n"] for row in result["per_session"].values())==1998
    assert result["secondary_best_logged_only"]["control"]["step"]==750


@pytest.mark.parametrize("delta,passed", [(0.,False),(-.009,False),(.02,False),(-.03,True)])
def test_screening_delta_threshold_is_fixed(delta,passed):
    result=analysis.analyze(bundle("control"),bundle("state",delta))
    assert result["screening"]["passed"] is passed


def test_bootstrap_resamples_whole_unequal_sessions_not_session_means():
    delta=np.array([1.,1.,1.,-1.]); sessions=np.array(["large"]*3+["small"])
    result=analysis.paired_cluster_bootstrap(delta,sessions,repeats=10000,seed=20260907)
    assert result["delta"]==.5  # session-mean would incorrectly return 0
    assert result["ci95"]==[-1.,1.]
    assert result==analysis.paired_cluster_bootstrap(delta,sessions,repeats=10000,seed=20260907)


@pytest.mark.parametrize("weights", [[1.,-1.],[1.,float("nan")],[0.,0.],[1.,0.]])
def test_negative_nonfinite_or_empty_session_weights_rejected(weights):
    with pytest.raises(ValueError):analysis.paired_cluster_bootstrap([0.,1.],["a","b"],weights=weights)


@pytest.mark.parametrize("field,value", [("actual_returncode",-11),("supervisor_exit_code",139),
    ("status","running"),("outcome","failed"),("pressure_event",{"error":"pressure"}),
    ("received_signals",[15]),("termination_signal",11),("actual_returncode",True)])
def test_actual_os_exit_not_completed_manifest_controls_gate(field,value):
    a,b=bundle("control"),bundle("state")
    b["supervisor"][field]=value
    with pytest.raises(ValueError):analysis.analyze(a,b)


def test_alive_parent_or_child_rejected():
    b=bundle("control"); run=Path(b["manifest"]["run_dir"])
    with pytest.raises(ValueError):
        analysis.validate_exit(b["supervisor"],b["manifest"],"control",run,
            Path(b["supervisor"]["request"]["record"]),lambda _:True)


@pytest.mark.parametrize("field,value", [("step",0),("nonfinite_count",1),("status","failed"),
    ("init_checkpoint_sha256","b"*64),("frozen_final_sha256","b"*64),("sample_order_sha256","b"*64),
    ("train_rows_sha256","b"*64),("eval_rows_sha256","b"*64),("initial_model_sha","b"*64),
    ("time_input","raw"),("query_adapter_on",False)])
def test_manifest_and_pair_fairness_guards(field,value):
    a,b=bundle("control"),bundle("state")
    b["manifest"][field]=value
    with pytest.raises(ValueError):analysis.analyze(a,b)


@pytest.mark.parametrize("field,value", [("lr",1e-4),("batch",8),("microbatch",4),("seed",1),
                                       ("steps",6000),("warmup",100)])
def test_schedule_fixed_even_if_both_arms_are_changed(field,value):
    a,b=bundle("control"),bundle("state")
    for x in (a,b):x["manifest"]["schedule"][field]=value
    with pytest.raises(ValueError):analysis.analyze(a,b)


@pytest.mark.parametrize("change", ["duplicate","reverse","missing","session","frame","nan","negative_proxy","metric","source","step"])
def test_record_identity_order_finiteness_and_last_only(change):
    a,b=bundle("control"),bundle("state"); e=b["evaluation"]
    if change=="duplicate":e["records"][-1]=e["records"][0]
    elif change=="reverse":e["records"].reverse()
    elif change=="missing":e["records"].pop()
    elif change=="session":e["records"][0]["session"]="foreign"
    elif change=="frame":e["records"][0]["frame"]=0
    elif change=="nan":e["records"][0]["d3"]=float("nan")
    elif change=="negative_proxy":e["records"][0]["proxy"]=-1
    elif change=="metric":e["report"]["official_d3"]+=1e-8
    elif change=="source":e["source"]["file_sha256"]["file0.py"]="e"*64
    else:e["step"]=750
    with pytest.raises(ValueError):analysis.analyze(a,b)


def test_moved_gpu_or_pid_cannot_be_silently_relabelled():
    a,b=bundle("control"),bundle("state")
    b["supervisor"]["request"]["physical_gpu"]=4
    with pytest.raises(ValueError):analysis.analyze(a,b)
    b=bundle("state"); b["supervisor"]["child_pid"]+=10
    with pytest.raises(ValueError):analysis.analyze(a,b)


def test_read_only_inputs_detect_change_and_preserve_before_bytes(tmp_path):
    path=tmp_path/"source.json"; path.write_text('{"a":1}')
    inputs=analysis.Inputs(); assert inputs.json(path)=={"a":1}; inputs.verify()
    path.write_text('{"a":2}')
    with pytest.raises(ValueError):inputs.verify()


@pytest.mark.parametrize("text", ['{"a":NaN}','{"a":1,"a":2}'])
def test_nonfinite_and_duplicate_json_keys_rejected(tmp_path,text):
    path=tmp_path/"source.json"; path.write_text(text)
    with pytest.raises(ValueError):analysis.Inputs().json(path)


def test_main_refuses_existing_output_before_loading_inputs(tmp_path,monkeypatch):
    path=tmp_path/"report.json"; path.write_text("keep")
    monkeypatch.setattr(analysis,"load_arm",lambda *_:pytest.fail("원본 조회 금지"))
    with pytest.raises(ValueError):analysis.main(["--control-run","a","--state-run","b","--out",str(path)])
    assert path.read_text()=="keep"


def test_explicit_r2_run_suffix_reaches_original_contract_validation(tmp_path):
    run=tmp_path/"work_dirs/motiondrive_v2/p3_query_state_s0_r2"; run.mkdir(parents=True)
    with pytest.raises(ValueError,match="일반 원본 파일"):
        analysis.load_arm(run,"state",analysis.Inputs())
    invalid=run.with_name("p1_g1s1_s0")
    with pytest.raises(ValueError,match="P3 고정 run"):
        analysis.load_arm(invalid,"state",analysis.Inputs())
