#!/usr/bin/env python3
"""P3 LAST1000 두 팔의 독립 CPU 감사와 11세션 paired screening.

과거 P2의 종료 게이트를 재사용하거나 완화하지 않는다. 미래 좌표/영상/추가
GT를 읽지 않고 기존 scalar D3만 비교한다. BEST는 별도 참고이고 채택 기준이 아니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCH = "motiondrive_v2_image_state_query_v1"
INIT_SHA = "5cd98d28157abf169f0f2378c3d0ab29d43f4d6382e9c03ee611e7fc5d864c20"
SPLIT_SHA = "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
SUP_SHA = "ba1ba04ebd2ac40dea5a27fa89f46c6a17de8aa48ab29da20a62fb9e17718d93"
CAL_SHA = "8bd130de0ab9081bcea486011abd071e8502c0ab0033c965fe448f70e612f961"
INITIAL_D3 = .4454619773599255
INITIAL_REFERENCE_SHA = "f32680b3f13f5f85045e27120c63db7afa1e24d3c566dc756ce8ac4c7e3158e1"
SCHEDULE = {"steps":1000,"batch":16,"microbatch":2,"eval_batch":4,"workers":4,
    "seed":0,"adapter_seed":0,"lr":5e-5,"warmup":50,"decay":"cosine","weight_decay":.01,
    "grad_clip":5.,"precision":"bf16","eval_steps":[0,250,500,750,1000]}
LOSS = {"plan":1.,"occupancy":.2,"lane":.2,"motion":.2,"uncertainty":True}
NUMERICS = {"cudnn_benchmark":False,"cudnn_deterministic":True,"matmul_allow_tf32":False,
            "precision":"bf16","planner_precision":"float32"}
COMMON_FIELDS = ("git_sha","source","model_config","schedule","loss_weights","numerics",
    "data","data_counts","dataset_inventory","initial_model_sha","frozen_initial_sha256",
    "frozen_final_sha256","train_rows_sha256","eval_rows_sha256","sample_order_sha256",
    "trainable_names","trainable_parameters","split_sha256","supervision_manifest_sha256",
    "canonical_calibration_sha256","time_input","time_input_policy","forward_contract",
    "auxiliary_loss_contract","sample_order_policy","selection","initial_reference")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False) == json.dumps(b,sort_keys=True,allow_nan=False)


def valid_sha(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.sha={}

    def pin(self,path,expected=None):
        path=Path(path)
        require(path.is_file() and not path.is_symlink(),f"일반 원본 파일이 필요합니다: {path}")
        path=path.resolve()
        value=sha256(path)
        require(expected is None or value==expected,f"파일 SHA 불일치: {path}")
        require(str(path) not in self.sha or self.sha[str(path)]==value,f"입력 파일이 바뀌었습니다: {path}")
        self.sha[str(path)]=value
        return path

    def json(self,path,expected=None):
        path=self.pin(path,expected)
        def nonfinite(value):
            raise ValueError(f"JSON 비유한 숫자: {value}")
        def distinct(pairs):
            result={}
            for key,value in pairs:
                require(key not in result,f"중복 JSON 키: {key}")
                result[key]=value
            return result
        return json.loads(path.read_text(),parse_constant=nonfinite,object_pairs_hook=distinct)

    def verify(self):
        for path,expected in self.sha.items():
            require(sha256(path)==expected,f"읽는 동안 원본 변경: {path}")


def validate_manifest(m,arm):
    flag=arm=="state"
    require(arm in ("control","state"),"P3 두 팔만 허용합니다")
    require(m.get("status")=="completed" and type(m.get("step")) is int and m["step"]==1000
        and type(m.get("nonfinite_count")) is int and m["nonfinite_count"]==0
        and m.get("received_signal") is None and m.get("source_unchanged") is True,
        "LAST1000 완료/비유한 값/중단 상태 검사 실패")
    require(m.get("architecture")==ARCH and m.get("query_adapter_on") is flag,"P3 architecture/팔 불일치")
    require(m.get("init_checkpoint_sha256")==INIT_SHA and m.get("split_sha256")==SPLIT_SHA
        and m.get("supervision_manifest_sha256")==SUP_SHA and m.get("canonical_calibration_sha256")==CAL_SHA
        and m.get("time_input")=="nominal","C1/T1 원본 계보 불일치")
    require(same(m.get("schedule"),SCHEDULE) and same(m.get("loss_weights"),LOSS)
        and same(m.get("numerics"),NUMERICS),"사전 고정 일정/손실/수치 정책 불일치")
    cfg=m.get("model_config",{})
    require(cfg.get("goal_on") is True and cfg.get("state_on") is True
        and cfg.get("motion_input_mode")=="low_feature" and list(cfg.get("plan_output_scale",[]))==[10.,5.],
        "공통 base config G1/S1/low_feature/scale 불일치")
    for field in ("initial_model_sha","frozen_initial_sha256","frozen_final_sha256",
                  "train_rows_sha256","eval_rows_sha256","sample_order_sha256"):
        require(valid_sha(m.get(field)),f"SHA 누락 또는 형식 오류: {field}")
    require(m["frozen_initial_sha256"]==m["frozen_final_sha256"]==m.get("frozen_state_sha_final"),
            "동결 encoder 시작/종료 SHA 불일치")
    migration=m.get("migration",{})
    require(migration.get("initial_model_state_sha256")==m["initial_model_sha"]
        and migration.get("source_checkpoint_sha256")==INIT_SHA
        and migration.get("source_checkpoint_step")==3000 and migration.get("adapter_seed")==0
        and migration.get("query_adapter_on") is flag and migration.get("legacy_tensors_bitwise_preserved") is True,
        "strict 초기화 계보 불일치")
    require(m.get("data_counts")=={"train":54810,"eval":1998},"표본 수 불일치")
    for split,n,scenes,sessions in (("train",54810,203,72),("eval",1998,37,11)):
        info=m.get("dataset_inventory",{}).get(split,{})
        require((info.get("n"),info.get("scenes"),info.get("sessions"))==(n,scenes,sessions)
            and info.get("rows_sha256")==m[split+"_rows_sha256"],"데이터 행/세션 계보 불일치")
    initial=m.get("initial_metric_gate",{})
    require(initial.get("passed") is True and initial.get("expected_official_d3")==INITIAL_D3
        and initial.get("observed_official_d3")==INITIAL_D3 and initial.get("absolute_tolerance")==0.,
        "초기 full-tune exact 재현 게이트 미통과")
    require(initial.get("reference_sha256")==INITIAL_REFERENCE_SHA
        and m.get("initial_reference",{}).get("sha256")==INITIAL_REFERENCE_SHA,
        "동결 초기 함수의 전 프레임 byte-parity 증거가 아닙니다")
    names=m.get("trainable_names",[])
    require(bool(names) and len(names)==len(set(names)) and all(n.startswith("planner.") for n in names)
        and any(n.startswith("planner.query_adapter.") for n in names),"planner-only 학습 계약 오류")
    require(type(m.get("trainable_parameters")) is int and m["trainable_parameters"]>0,"학습 파라미터 수 오류")
    require(m.get("best_step") in (250,500,750,1000) and finite(m.get("best_metric")) and m["best_metric"]>=0,
            "BEST 참고값의 step/유한성 오류")
    source=m.get("source",{})
    require(source.get("git_sha")==m.get("git_sha") and not source.get("tracked_changes")
        and len(source.get("file_sha256",{}))==17 and all(valid_sha(s) for s in source["file_sha256"].values()),
        "학습 source17 계보 오류")


def validate_exit(s,m,arm,run,supervisor_path,alive):
    require(s.get("status")=="process_exited" and s.get("outcome")=="completed_cleanly"
        and type(s.get("actual_returncode")) is int and s["actual_returncode"]==0
        and type(s.get("supervisor_exit_code")) is int and s["supervisor_exit_code"]==0
        and s.get("termination_signal") is None and s.get("pressure_event") is None
        and s.get("received_signals")==[] and not s.get("validation_error"),"실제 clean OS 종료가 아닙니다")
    for key in ("parent_pid","child_pid"):
        require(type(s.get(key)) is int and s[key]>0 and not alive(s[key]),"프로세스 생존 또는 PID 형식 오류")
    require(m.get("pid")==s["child_pid"],"trainer와 supervisor child PID 불일치")
    request=s.get("request",{})
    expected_gpu=4 if arm=="control" else 5
    require(request.get("arm")==arm and request.get("adapter_on") is (arm=="state")
        and request.get("run_dir")==str(run) and request.get("manifest_path")==str(run/"manifest.json")
        and request.get("record")==str(supervisor_path) and request.get("physical_gpu")==expected_gpu
        and request.get("initializer_sha256")==INIT_SHA and request.get("expected_commit")==m["git_sha"],
        "supervisor 요청/팔/원본 정체성 불일치")
    device=m.get("device_mapping",{})
    uuid=request.get("gpu_uuid")
    raw=device.get("observed_uuid_raw","")
    require(isinstance(uuid,str) and uuid.startswith("GPU-") and device.get("observed_uuid")==uuid
        and (raw if raw.startswith("GPU-") else "GPU-"+raw)==uuid
        and device.get("cuda_visible_devices")==uuid and device.get("logical_device")=="cuda:0"
        and device.get("physical_gpu")==expected_gpu,"실제 CUDA UUID/물리 GPU 매핑 불일치")
    evidence=s.get("completion_evidence",{})
    require(evidence.get("frozen_state_sha256")==m["frozen_final_sha256"]
        and valid_sha(evidence.get("manifest_sha256")) and valid_sha(evidence.get("last_sha256")),
        "감독 완료 증거 누락")


def validate_evaluation(e,m,expected_step=1000):
    require(type(e.get("step")) is int and e["step"]==expected_step and e.get("time_input")=="nominal"
        and e.get("architecture")==ARCH and e.get("query_adapter_on") is m["query_adapter_on"]
        and same(e.get("source"),m["source"]),"평가 step/모델/source 불일치; BEST 혼합 금지")
    report,rows=e.get("report",{}),e.get("records",[])
    require(report.get("n")==1998 and report.get("n_sessions")==11 and report.get("time_input")=="nominal"
        and len(rows)==1998,"동일 tune1998/11세션 평가 필요")
    ids=[]
    for r in rows:
        require(set(r)=={"scenario","session","frame","d3","proxy"},"기존 scalar record schema만 허용")
        require(isinstance(r["scenario"],str) and isinstance(r["session"],str)
            and type(r["frame"]) is int and finite(r["d3"]) and r["d3"]>=0
            and finite(r["proxy"]) and r["proxy"]>=0,"비유한 점수/음수 가중치/정체성 오류")
        ids.append((r["scenario"],r["session"],r["frame"]))
    require(len(set(ids))==1998 and len({a for a,_,_ in ids})==37 and len({b for _,b,_ in ids})==11,
            "중복/누락 scene-session-frame")
    for scene in {a for a,_,_ in ids}:
        selected=[(session,frame) for name,session,frame in ids if name==scene]
        require(len({s for s,_ in selected})==1 and sorted(f for _,f in selected)==list(range(30,300,5)),
                "scene별 고정 54 frame 또는 session 매핑 불일치")
    values=np.asarray([r["d3"] for r in rows],np.float64)
    require(finite(report.get("official_d3")) and float(values.mean())==report["official_d3"],
            "공식 D3 저장 집계와 scalar records 불일치")
    sessions={s:float(np.mean([r["d3"] for r in rows if r["session"]==s]))
              for s in dict.fromkeys(b for _,b,_ in ids)}
    require(same(report.get("session_d3"),sessions)
        and report.get("session_mean_d3")==float(np.mean(list(sessions.values()))),"session 집계 불일치")
    return ids,values


def checkpoint_audit(path,manifest,step,inputs):
    import torch
    from scripts.motiondrive_v2_training import tensor_state_sha256
    path=inputs.pin(path)
    payload=torch.load(path,map_location="cpu",weights_only=False)  # 신뢰된 팀 checkpoint, 원본 SHA 먼저 고정
    require(type(payload.get("step")) is int and payload["step"]==step
        and payload.get("architecture")==ARCH and payload.get("query_adapter_on") is manifest["query_adapter_on"]
        and same(payload.get("model_config"),manifest["model_config"]),"체크포인트 step/config/팔 불일치")
    stored=payload.get("manifest",{})
    for field in COMMON_FIELDS:
        if field in ("sample_order_sha256","frozen_final_sha256") and step==0:
            continue
        require(same(stored.get(field),manifest.get(field)),f"체크포인트/최종 manifest 계보 불일치: {field}")
    require(stored.get("architecture")==ARCH and stored.get("query_adapter_on") is manifest["query_adapter_on"],
            "체크포인트 중복 architecture 선언 불일치")
    state=payload.get("model",{})
    require(bool(state) and all(isinstance(x,torch.Tensor) and x.device.type=="cpu"
        and bool(torch.isfinite(x).all()) for x in state.values()),"모델 tensor 비유한 값/타입 오류")
    frozen={name:value for name,value in state.items() if not name.startswith("planner.")}
    require(bool(frozen) and tensor_state_sha256(frozen)==manifest["frozen_initial_sha256"],"실제 frozen tensor SHA 불일치")
    if step==0:
        require(tensor_state_sha256(state)==manifest["initial_model_sha"] and payload.get("optimizer",{}).get("state")=={},
                "공통 initial 전체 tensor 또는 새 optimizer 불일치")
    else:
        require(payload.get("sample_order_sha256")==manifest["sample_order_sha256"],"LAST sampler SHA 불일치")
        optimizer=payload.get("optimizer",{}).get("state",{})
        require(bool(optimizer),"LAST optimizer state 누락")
        for value in optimizer.values():
            require(float(value.get("step",-1))==1000,"optimizer step 불일치")
            require(all(not isinstance(x,torch.Tensor) or bool(torch.isfinite(x).all()) for x in value.values()),
                    "optimizer 비유한 tensor")
    result={"path":str(path),"sha256":inputs.sha[str(path)],"step":step,
            "model_state_sha256":tensor_state_sha256(state),"frozen_state_sha256":tensor_state_sha256(frozen),
            "model_tensors_finite":True,"model_forward_performed":False}
    del payload
    return result


def load_arm(run_dir,arm,inputs,alive=None):
    run=Path(run_dir).resolve()
    require(arm in ("control","state") and re.fullmatch(f"p3_query_{arm}_s0(?:_r[2-9][0-9]*)?",run.name) is not None
        and run.parent.name=="motiondrive_v2"
        and run.parent.parent.name=="work_dirs","P3 고정 run 경로가 아닙니다")
    root=run.parents[2]
    supervisor=root/"logs/motiondrive_v2"/(run.name+".supervisor.json")
    m=inputs.json(run/"manifest.json"); validate_manifest(m,arm)
    s=inputs.json(supervisor)
    validate_exit(s,m,arm,run,supervisor,alive or (lambda pid:Path(f"/proc/{pid}").exists()))
    evidence=s["completion_evidence"]
    inputs.pin(run/"manifest.json",evidence["manifest_sha256"])
    inputs.pin(run/"last.pth",evidence["last_sha256"])
    require(s["request"].get("root")==str(root) and m.get("run_dir")==str(run),"원 repo/run 경로 불일치")
    source=s["request"]["sources"]
    require(source.get("git_sha")==m["git_sha"] and all(source.get("file_sha256",{}).get(k)==v
            for k,v in m["source"]["file_sha256"].items()),"훈련/감독 source 계보 불일치")
    for name,digest in source["file_sha256"].items():
        path=root/name
        require(path.resolve().is_relative_to(root) and valid_sha(digest),"허용되지 않은 소스 경로/SHA")
        inputs.pin(path,digest)
    for key,expected in (("init",INIT_SHA),("split",SPLIT_SHA),("supervision",SUP_SHA),("calibration",CAL_SHA)):
        item=m["data"].get(key,{})
        require(item.get("sha256")==expected,"감독 데이터 SHA 선언 불일치")
        inputs.pin(item["path"],expected)
    inputs.pin(m["initial_reference"]["path"],INITIAL_REFERENCE_SHA)
    e=inputs.json(run/"eval_step1000.json")
    ids,values=validate_evaluation(e,m)
    split=inputs.json(m["data"]["split"]["path"],SPLIT_SHA)
    require({name for name,_,_ in ids}==set(split["splits"]["tune"])
        and all(split["scene_to_session"].get(name)==session for name,session,_ in ids),"실제 rawtime tune session 매핑 불일치")
    zero=inputs.json(run/"eval_step0000.json"); zero_ids,_=validate_evaluation(zero,m,0)
    require(ids==zero_ids and zero["report"]["official_d3"]==INITIAL_D3,"initial과 LAST 표본 또는 초기 exact D3 불일치")
    cp={name:checkpoint_audit(run/f"{name}.pth",m,step,inputs) for name,step in (("initial",0),("last",1000))}
    return {"manifest":m,"supervisor":s,"evaluation":e,"ids":ids,"values":values,"checkpoints":cp}


def paired_cluster_bootstrap(delta,sessions,*,weights=None,repeats=10000,seed=20260907):
    delta=np.asarray(delta,np.float64); sessions=np.asarray(sessions)
    weights=np.ones(delta.shape,np.float64) if weights is None else np.asarray(weights,np.float64)
    require(delta.ndim==1 and delta.size>0 and sessions.shape==delta.shape and weights.shape==delta.shape,
            "cluster 통계 배열 규격 오류")
    require(np.isfinite(delta).all() and np.isfinite(weights).all() and (weights>=0).all() and weights.sum()>0,
            "비유한 값 또는 음수/전체0 가중치")
    require(type(repeats) is int and repeats>0 and type(seed) is int and seed>=0,"bootstrap 설정 오류")
    unique=sorted(set(sessions.tolist()))
    totals=np.asarray([np.sum(delta[sessions==s]*weights[sessions==s]) for s in unique])
    counts=np.asarray([weights[sessions==s].sum() for s in unique])
    require((counts>0).all(),"가중치가 0인 세션은 허용하지 않습니다")
    picked=np.random.default_rng(seed).integers(0,len(unique),size=(repeats,len(unique)))
    draws=totals[picked].sum(1)/counts[picked].sum(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return {"delta":float(totals.sum()/counts.sum()),"ci95":[float(lo),float(hi)],
            "repeats":repeats,"seed":seed,"clusters":len(unique),
            "method":"세션 uniform 복원추출; 선택된 모든 프레임의 오차 합/프레임 수. frame IID 아님"}


def analyze(control,state):
    a,b=control["manifest"],state["manifest"]
    validate_manifest(a,"control"); validate_manifest(b,"state")
    for arm,bundle in (("control",control),("state",state)):
        run=Path(bundle["manifest"]["run_dir"])
        validate_exit(bundle["supervisor"],bundle["manifest"],arm,run,
            run.parents[2]/"logs/motiondrive_v2"/(run.name+".supervisor.json"),
            lambda pid:Path(f"/proc/{pid}").exists())
    for field in COMMON_FIELDS:
        require(field in a and field in b and same(a[field],b[field]),f"두 팔 공통 조건 불일치: {field}")
    aa={k:v for k,v in a["arguments"].items() if k not in ("run_dir","adapter_on")}
    bb={k:v for k,v in b["arguments"].items() if k not in ("run_dir","adapter_on")}
    require(same(aa,bb),"run/adapter flag 외 CLI 차이가 있습니다")
    ai,av=validate_evaluation(control["evaluation"],a)
    bi,bv=validate_evaluation(state["evaluation"],b)
    require(ai==bi,"팔별 (scenario,session,frame) 순서 불일치")
    require([r["proxy"] for r in control["evaluation"]["records"]]==[r["proxy"] for r in state["evaluation"]["records"]],
            "원본 proxy 값이 다릅니다(통계에는 사용하지 않음)")
    stats=paired_cluster_bootstrap(bv-av,[session for _,session,_ in ai])
    passed=stats["delta"]<=-.01 and stats["ci95"][1]<0
    per_session={}
    for session in sorted({s for _,s,_ in ai}):
        mask=np.asarray([s==session for _,s,_ in ai])
        per_session[session]={"n":int(mask.sum()),"control":float(av[mask].mean()),
            "state":float(bv[mask].mean()),"delta":float((bv-av)[mask].mean())}
    return {"status":"P3 LAST1000 paired screening 완료","primary":{
        "control_official_d3":float(av.mean()),"state_official_d3":float(bv.mean()),
        "state_over_control":float(bv.mean()/av.mean()) if av.mean()>0 else None,
        "state_minus_control":stats},"screening":{"criterion":"delta<=-0.01 and ci95.upper<0",
        "passed":bool(passed),"meaning":"seed 복제 검토 기준일 뿐 최종 모델 채택 또는 수상 성능 증명 아님"},
        "secondary_best_logged_only":{arm:{"step":m.get("best_step"),"official_d3":m.get("best_metric")}
            for arm,m in (("control",a),("state",b))},"per_session":per_session,
        "protocol":{"step":1000,"n":1998,"scenes":37,"sessions":11,"split_sha256":SPLIT_SHA,
            "time_point_weights":[11/36,11/36,5/36,5/36,2/36,2/36],"frame_weight":"모든 frame 동일 가중 1",
            "scalar_d3_source":"기존 evaluator의 공식 시간가중 D3. XY를 읽지 않으므로 시점별 재계산은 하지 않음",
            "row_identity":"scalar에는 row 정수가 없음. 동일 eval_rows_sha256와 scene/session/frame 순서 공동 검사",
            "proxy_used":False,"finalval_test_used":False,"model_forward_performed":False,
            "bootstrap":"11 rawtime session paired cluster, 10000 draws, seed20260907"},
        "limits":["단일 seed 및 반복 사용한 tune37의 탐색 결과", "BEST는 주판정에 혼합하지 않음",
            "원 P2 4팔 clean-exit 실패 판정은 바꾸지 않음", "잘못된 과거 기하에서 warm-start한 한계가 남음",
            "GT 상태의 규칙 외삽/좌표 보정/성능 상한 추정/자동 제출 없음"]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--control-run",required=True); parser.add_argument("--state-run",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    output=Path(args.out)
    require(not output.exists() and not output.is_symlink(),"기존 분석 결과를 덮어쓰지 않습니다")
    inputs=Inputs()
    control=load_arm(args.control_run,"control",inputs)
    state=load_arm(args.state_run,"state",inputs)
    result=analyze(control,state)
    result["audit"]={arm:{"checkpoints":bundle["checkpoints"],
        "actual_returncode":bundle["supervisor"]["actual_returncode"],
        "supervisor_exit_code":bundle["supervisor"]["supervisor_exit_code"]} for arm,bundle in (("control",control),("state",state))}
    inputs.verify()
    result["input_sha256"]=dict(inputs.sha)
    result["analysis_source_sha256"]=sha256(__file__)
    result["all_inputs_unchanged"]=True
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    with output.open("x") as stream:
        stream.write(text)
    print(json.dumps({"out":str(output),"sha256":sha256(output),"screening":result["screening"]},ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
