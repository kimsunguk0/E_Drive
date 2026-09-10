"""Start a durable single-GPU child and preserve its real exit status."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2)+"\n")
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--gpu",type=int,choices=(0,1,4),required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--directory",required=True)
    p.add_argument("--supervise",action="store_true")
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command or Path(a.name).name!=a.name:
        raise ValueError("A plain run name and exact child command are required")
    directory=Path(a.directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/(a.name+".json")
    log=directory/(a.name+".log")
    if not a.supervise:
        if receipt.exists() or log.exists():
            raise FileExistsError("Launch names are immutable; use a new attempt name")
        used=int(subprocess.check_output(["nvidia-smi","-i",str(a.gpu),"--query-gpu=memory.used",
                                         "--format=csv,noheader,nounits"],text=True).strip())
        if used>100:
            raise RuntimeError(f"Assigned GPU {a.gpu} is occupied ({used} MiB)")
        argv=[sys.executable,__file__,"--gpu",str(a.gpu),"--name",a.name,"--directory",str(directory),
              "--supervise","--",*command]
        with log.open("x") as stream:
            proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                  start_new_session=True,cwd=Path(__file__).resolve().parents[2])
        # Supervisor owns the receipt, avoiding parent/child overwrite races.
        print(json.dumps({"supervisor_pid":proc.pid,"gpu":a.gpu,"log":str(log),"receipt":str(receipt)}),flush=True)
        return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
             PYTHONUNBUFFERED="1")
    start=time.time()
    proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL)
    data={"name":a.name,"status":"running","gpu":a.gpu,"supervisor_pid":os.getpid(),
          "child_pid":proc.pid,"argv":command,"started_unix":start,"log":str(log)}
    write(receipt,data)
    rc=proc.wait()
    data.update(status="completed" if rc==0 else "failed",returncode=rc,finished_unix=time.time(),
                elapsed_seconds=time.time()-start)
    write(receipt,data)
    sys.exit(rc if rc>=0 else 128-rc)


if __name__=="__main__":
    main()
