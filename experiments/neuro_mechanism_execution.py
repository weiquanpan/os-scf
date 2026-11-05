"""Three disjoint dataset processes for fits whose times are not benchmark claims."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def launch(module,args,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);children=[]
    try:
        for ds in ['lendingclub','bikesharing','taxi']:
            log=open(folder/f'{ds}.log','w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-m',module,*args,'--dataset',ds],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((ds,proc,log))
        while any(proc.poll() is None for _,proc,_ in children):
            failed=[ds for ds,proc,_ in children if proc.poll() not in (None,0)]
            if failed:raise RuntimeError(f'Dataset worker failed: {failed}; see {folder}')
            time.sleep(10)
        assert all(proc.returncode==0 for _,proc,_ in children)
    finally:
        for ds,proc,log in children:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()
