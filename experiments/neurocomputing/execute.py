"""Durable sequential driver with status, logs and a final evidence audit."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,json,sys,time,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');os.replace(tmp,p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/neurocomputing_v1');p.add_argument('--timeout-hours',type=float,default=24)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir(exist_ok=True)
    # Preserve the exact submission candidate separately from training sources.
    archive=out/'paper_v6_before'
    if not archive.exists():shutil.copytree(ROOT/'paper-v6',archive)
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in archive.iterdir() if f.is_file()}
    write(out/'paper_v6_before_manifest.json',manifest)
    stages=['prepare','flow','ablation','tabsyn','tabddpm','references','audit'];start=time.time()
    lock=out/'driver.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    except FileExistsError:raise SystemExit('Driver lock exists. Inspect its PID before restarting.')
    try:
        for i,stage in enumerate(stages):
            command=[sys.executable,'-u','-m','experiments.neurocomputing.'+('audit' if stage=='audit' else 'run'),'--out',str(out)]
            command+=['--require-complete'] if stage=='audit' else ['--stage',stage]
            logpath=logs/f'{i:02d}_{stage}.log'
            with logpath.open('a',encoding='utf8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                while True:
                    state={'driver_pid':os.getpid(),'worker_pid':child.pid,'stage':stage,'stage_index':i+1,'stages':len(stages),
                           'elapsed_seconds':round(time.time()-start,1),'log':str(logpath),'status':'RUNNING'}
                    write(out/'RUN_STATUS.json',state)
                    try:code=child.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        if time.time()-start>a.timeout_hours*3600:
                            child.terminate();child.wait();state.update(status='TIMED_OUT');write(out/'RUN_STATUS.json',state);return 2
                if code:
                    state.update(status='FAILED',exit_code=code);write(out/'RUN_STATUS.json',state);return code
        state.update(status='COMPLETE',elapsed_seconds=round(time.time()-start,1));write(out/'RUN_STATUS.json',state);return 0
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':raise SystemExit(main())
