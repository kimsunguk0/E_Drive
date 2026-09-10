"""Fixed-checkpoint tune diagnostics for entropy, ranking, and geometry.

Only completed runs or explicitly pinned immutable snapshots are accepted.
No coordinate refinement, fitting, held evaluation, or runtime-file edits.
CPU float32 is the default and is not an exact substitute for CUDA BF16.
"""
from __future__ import annotations
import argparse, hashlib, importlib, inspect, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np

W = np.array([11,11,5,5,2,2], np.float64) / 36


def sha(path):
    h = hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def entropy(p):
    p=np.asarray(p,np.float64)
    return float(-(p[p>0]*np.log(p[p>0])).sum())


def probabilities(logits):
    z=np.asarray(logits,np.float64)
    shifted=z-z.max()
    logp=shifted-np.log(np.exp(shifted).sum())
    return np.exp(logp),logp


def distribution(logits,cost,temp):
    from scipy.stats import spearmanr
    p,logp=probabilities(logits)
    q,logq=probabilities(-np.asarray(cost,np.float64)/temp)
    ce=float(-(q*logp).sum());hq=entropy(q);hp=entropy(p)
    selected=int(np.argmax(logits));best=int(np.argmin(cost))
    correlation=None
    if np.ptp(logits)>1e-12 and np.ptp(cost)>1e-12:
        correlation=float(spearmanr(logits,-np.asarray(cost)).statistic)
    result={"target_entropy":hq,"model_entropy":hp,"fine_ce":ce,"fine_kl_q_p":ce-hq,
            "uniform_ce":float(np.log(len(cost))),"ce_minus_uniform":ce-float(np.log(len(cost))),
            "target_effective_count":float(np.exp(hq)),"model_effective_count":float(np.exp(hp)),
            "score_cost_spearman":correlation,"selected_cost":float(cost[selected]),
            "oracle_cost":float(cost[best]),"top1_regret":float(cost[selected]-cost[best]),
            "expected_cost_model":float(p@cost),"expected_cost_target":float(q@cost),
            "expected_cost_uniform":float(np.mean(cost)),"selected_probability":float(p[selected]),
            "target_probability_at_selected":float(q[selected]),"cost_spread":float(np.ptp(cost)),
            "logit_spread":float(np.ptp(logits)),"logit_std":float(np.std(logits))}
    for eps in (.01,.02,.05,.1):
        near=np.asarray(cost)<=float(cost[best])+eps
        result[f"within_{eps:g}m"]={"candidate_count":int(near.sum()),"model_mass":float(p[near].sum()),
                                  "target_mass":float(q[near].sum()),"selected_hit":bool(near[selected])}
    return result,p,q


def marginals(p,q,path_ids,velocity_ids):
    result={}
    for name,ids in (("path",path_ids),("velocity",velocity_ids)):
        values,index=np.unique(ids,return_inverse=True)
        pm=np.bincount(index,weights=p);qm=np.bincount(index,weights=q)
        result[name]={"ids":values.tolist(),"model_mass":pm.tolist(),"target_mass":qm.tolist(),
                      "model_entropy":entropy(pm),"target_entropy":entropy(qm),
                      "kl_target_model":float((qm[qm>0]*np.log(qm[qm>0]/np.maximum(pm[qm>0],1e-300))).sum())}
    result["target_path_velocity_mutual_information"]=result["path"]["target_entropy"]+result["velocity"]["target_entropy"]-entropy(q)
    result["model_path_velocity_mutual_information"]=result["path"]["model_entropy"]+result["velocity"]["model_entropy"]-entropy(p)
    return result


def geometry(candidate,oracle_index):
    pair=(np.linalg.norm(candidate[:,None].astype(np.float64)-candidate[None,:].astype(np.float64),axis=-1)*W).sum(-1)
    result={"exact_unique_trajectories":int(len(np.unique(candidate.reshape(len(candidate),-1),axis=0)))}
    for eps in (.001,.01,.025,.05,.1):
        counts=(pair<=eps).sum(1)
        result[f"within_{eps:g}m"]={"mean_neighbors_including_self":float(counts.mean()),
                                  "oracle_neighbors_including_self":int(counts[oracle_index]),
                                  "fraction_pairs_excluding_diagonal":float(((pair<=eps).sum()-len(pair))/max(len(pair)*(len(pair)-1),1))}
    return result


def self_test():
    result,p,q=distribution(np.zeros(4),np.ones(4),.1)
    assert abs(result["fine_ce"]-np.log(4))<1e-12 and abs(result["fine_kl_q_p"])<1e-12
    result,p,q=distribution(np.zeros(4),np.array([0.,.1,.2,.3]),.1)
    assert result["target_entropy"]<np.log(4) and abs(result["fine_kl_q_p"]-(np.log(4)-entropy(q)))<1e-12
    result,p,q=distribution(-np.array([0.,.1,.2,.3])/.1,np.array([0.,.1,.2,.3]),.1)
    assert abs(result["fine_kl_q_p"])<1e-12 and result["score_cost_spearman"]>.999
    q=np.array([.4,.1,.4,.1]);m=marginals(q,q,np.array([0,0,1,1]),np.array([0,1,0,1]))
    assert abs(m["path"]["target_entropy"]-np.log(2))<1e-12
    assert abs(m["target_path_velocity_mutual_information"])<1e-12
    print("entropy/KL/marginal contracts PASS")


def main(args):
    import torch
    checkpoint=Path(args.checkpoint).resolve()
    if args.checkpoint_sha256:
        expected=args.checkpoint_sha256
    else:
        if not args.launch_receipt:
            raise ValueError("Provide completed launch receipt or pre-pinned checkpoint SHA")
        receipt=json.loads(Path(args.launch_receipt).read_text())
        if receipt.get("status")!="completed" or receipt.get("returncode")!=0:
            raise ValueError("Refusing a checkpoint from an unfinished run; make an immutable snapshot first")
        expected=sha(checkpoint)
    before=checkpoint.stat()
    if sha(checkpoint)!=expected:
        raise ValueError("Checkpoint SHA mismatch")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    checkpoint_step=int(payload["step"])
    after=checkpoint.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) or sha(checkpoint)!=expected:
        raise ValueError("Checkpoint changed while loading")
    manifest=payload["manifest"]
    source=Path(args.source_dir) if args.source_dir else checkpoint.parent/"source/experiments/sparsedrivev2_20260910"
    if not source.is_dir():
        raise ValueError("Saved checkpoint runtime source directory required")
    sys.path.insert(0,str(source))
    data=importlib.import_module("data");pm=importlib.import_module("public_model");losses=importlib.import_module("losses")
    source_hashes={}
    for name in ("data.py","public_model.py","losses.py"):
        rel="experiments/sparsedrivev2_20260910/"+name
        source_hashes[name]=sha(source/name)
        if source_hashes[name]!=manifest["source_sha256"][rel]:
            raise ValueError("Saved source hash mismatch: "+name)
    bank_path=Path(manifest["arguments"]["bank"])
    if sha(bank_path)!=manifest["bank_sha256"]:
        raise ValueError("Frozen bank SHA mismatch")
    state=payload["model"]
    goal_mode=manifest["arguments"].get("goal_mode","none")
    prefix="base." if goal_mode=="selection" else ""
    keys=("path_vocab","vel_vocab","traj_vocab","traj_mask")
    bank={k:state[prefix+"_trajectory_head."+k] for k in keys}
    model=pm.PublicSparseDriveV2(bank,backend="grid" if args.device=="cpu" else "native",mask_invalid_candidates=True)
    if goal_mode=="selection":
        goal=importlib.import_module("goal_selector")
        source_hashes["goal_selector.py"]=sha(source/"goal_selector.py")
        if source_hashes["goal_selector.py"]!=manifest["source_sha256"]["experiments/sparsedrivev2_20260910/goal_selector.py"]:
            raise ValueError("Goal selector source mismatch")
        model=goal.GoalConditionedSelector(model)
    model.load_state_dict(state,strict=True)
    del payload,state
    torch.set_num_threads(args.threads)
    if args.device!="cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0","1","4"):
        raise ValueError("Explicitly expose one authorized GPU for later GPU diagnostics")
    device=torch.device(args.device);model.to(device).eval()
    kwargs=dict(base=manifest["arguments"]["base"],split_manifest=manifest["arguments"]["split_manifest"],
                split="tune",status_mode=manifest["arguments"]["status_mode"],augment=False,limit=args.limit)
    if "goal_mode" in inspect.signature(data.PlanDataset).parameters:
        kwargs["goal_mode"]=goal_mode
    dataset=data.PlanDataset(**kwargs)
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=False)
    arrays={};records=[];cal_p=[];cal_q=[];started=time.time();head=model._trajectory_head
    def keep(key,value):
        arrays.setdefault(key,[]).append(np.asarray(value))
    with torch.inference_mode():
        for i in range(len(dataset)):
            sample=dataset[i]
            params=inspect.signature(data.model_inputs).parameters
            raw=data.model_inputs(sample,goal_selection=goal_mode=="selection") if "goal_selection" in params else data.model_inputs(sample)
            x={k:v.unsqueeze(0).to(device) for k,v in raw.items()}
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.precision=="bf16"):
                output=model(**x)
            gt=sample["gt_plan"].unsqueeze(0).to(device)
            costs=data.d3(output["candidate_xy"],gt[:,None].expand_as(output["candidate_xy"]))
            valid=output["candidate_valid"][0].bool()
            candidate=output["candidate_xy"][0].float().cpu().numpy()
            logits=output["scores"][0].float().cpu().numpy();ids=output["candidate_ids"][0].cpu().numpy()
            cost=costs[0].cpu().numpy();ok=valid.cpu().numpy()
            if not ok.all():
                raise ValueError("This dense-bank diagnostic expects all final rows valid; add explicit padding support for mixed banks")
            stats,p,q=distribution(logits,cost,args.temperature)
            ref=float(losses.soft_cost_ce(output["scores"],costs,args.temperature,valid[None]))
            if abs(ref-stats["fine_ce"])>2e-5:
                raise ValueError("Fine CE parity failure")
            vcount=len(head.vel_vocab);pid=ids//vcount;vid=ids%vcount
            selected=int(np.argmax(logits));oracle=int(np.argmin(cost))
            if not np.array_equal(output["trajectory"][0].float().cpu().numpy(),candidate[selected]):
                raise ValueError("Selected trajectory is not a completed candidate row")
            pc,vc=losses.coarse_costs(head.path_vocab[...,:2],head.vel_vocab,gt)
            pc,vc=pc[0].cpu().numpy(),vc[0].cpu().numpy()
            selected_errors=candidate[selected]-gt[0].cpu().numpy()
            oracle_errors=candidate[oracle]-gt[0].cpu().numpy()
            stats.update({"row":int(sample["row"]),"scenario":str(sample["scenario"]),"session":str(sample["session"]),
                "selected_candidate_id":int(ids[selected]),"oracle_candidate_id":int(ids[oracle]),
                "selected_path":int(pid[selected]),"selected_velocity":int(vid[selected]),
                "oracle_given_selected_path":float(cost[pid==pid[selected]].min()),
                "oracle_given_selected_velocity":float(cost[vid==vid[selected]].min()),
                "global_path_at_gt_progress_min":float(pc.min()),"global_velocity_progress_min":float(vc.min()),
                "selected_path_at_gt_progress_cost":float(pc[pid[selected]]),"selected_velocity_progress_cost":float(vc[vid[selected]]),
                "selected_weighted_abs_xy":(np.abs(selected_errors)*W[:,None]).sum(0).tolist(),
                "oracle_weighted_abs_xy":(np.abs(oracle_errors)*W[:,None]).sum(0).tolist(),
                "marginals":marginals(p,q,pid,vid),"geometry":geometry(candidate,oracle),"coarse":[]})
            for stage,c in enumerate(output["coarse"]):
                stage_summary={}
                for name,fullcost in (("path",pc),("velocity",vc)):
                    cid=c[name+"_ids"][0].cpu().numpy();clog=c[name+"_scores"][0].float().cpu().numpy();ccost=fullcost[cid]
                    cm,_,_=distribution(clog,ccost,args.temperature)
                    stage_summary[name]=cm
                    keep(f"coarse{stage}_{name}_ids",cid);keep(f"coarse{stage}_{name}_logits",clog);keep(f"coarse{stage}_{name}_costs",ccost)
                stats["coarse"].append(stage_summary)
            keep("rows",sample["row"]);keep("candidate_ids",ids);keep("candidate_xy",candidate);keep("costs",cost)
            keep("logits",logits);keep("model_probability",p);keep("target_probability",q);keep("gt_xy",gt[0].cpu().numpy())
            keep("path_costs_all",pc);keep("velocity_costs_all",vc)
            cal_p.extend(p);cal_q.extend(q);records.append(stats)
            print(json.dumps({"row":stats["row"],"CE":stats["fine_ce"],"Hq":stats["target_entropy"],"KL":stats["fine_kl_q_p"],
                              "D3":stats["selected_cost"],"regret":stats["top1_regret"],"seconds":time.time()-started}),flush=True)
    np.savez_compressed(outdir/"candidate_dump.npz",**{k:np.stack(v) for k,v in arrays.items()})
    numeric=("fine_ce","target_entropy","model_entropy","fine_kl_q_p","ce_minus_uniform","target_effective_count",
             "model_effective_count","selected_cost","oracle_cost","top1_regret","expected_cost_model","expected_cost_target",
             "expected_cost_uniform","logit_spread","logit_std","selected_probability","target_probability_at_selected")
    summary={name:float(np.mean([r[name] for r in records])) for name in numeric}
    validcorr=[r["score_cost_spearman"] for r in records if r["score_cost_spearman"] is not None]
    summary["score_cost_spearman_mean"]=float(np.mean(validcorr)) if validcorr else None
    calibration=[];cal_p=np.asarray(cal_p);cal_q=np.asarray(cal_q)
    edges=[0,.0025,.005,.0075,.01,.025,.05,.1,.2,.5,1.00001]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(cal_p>=lo)&(cal_p<hi)
        if mask.any():calibration.append({"p_lo":lo,"p_hi":hi,"count":int(mask.sum()),"mean_model_p":float(cal_p[mask].mean()),"mean_target_q":float(cal_q[mask].mean())})
    report={"checkpoint":str(checkpoint),"checkpoint_sha256":expected,"checkpoint_step":checkpoint_step,
            "bank_sha256":manifest["bank_sha256"],"source_dir":str(source),"source_sha256":source_hashes,
            "device":str(device),"precision":args.precision,"threads":args.threads,"temperature":args.temperature,
            "n":len(records),"row_selection":"deterministic uniform indices across original tune rows before seeing predictions",
            "rows_sha256":hashlib.sha256(np.asarray(dataset.rows,dtype="<i8").tobytes()).hexdigest(),
            "status_mode":kwargs["status_mode"],"goal_mode":goal_mode,"summary":summary,
            "soft_target_probability_calibration":calibration,"records":records,"seconds":time.time()-started,
            "warning":"small tune subset; entropy/KL describe temperature-defined labels, not calibrated driving probabilities; CPU fp32 rows/candidates may differ from native CUDA bf16; no causal claim or held evaluation",
            "diagnostics_code_sha256":sha(__file__)}
    (outdir/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint");parser.add_argument("--checkpoint-sha256");parser.add_argument("--launch-receipt")
    parser.add_argument("--source-dir");parser.add_argument("--output-dir");parser.add_argument("--limit",type=int,default=8)
    parser.add_argument("--temperature",type=float,default=.1);parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",default="cpu");parser.add_argument("--precision",choices=("fp32","bf16"),default="fp32")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if not args.checkpoint or not args.output_dir:parser.error("--checkpoint and --output-dir required")
        main(args)
