"""Partition the frozen P2 seed loop, preserving its exact per-cell implementation."""
from experiments.neuro_mechanism_utils import *
from experiments import neuro_mechanism_p2 as p2
from experiments import neuro_mechanism_fast_metrics as fast
from experiments.neuro_mechanism_p2_resume import summarize
import argparse,inspect,subprocess,sys
PARTITIONS=[[201,204,207,210],[202,205,208],[203,206,209]]

def worker(seeds):
    # Four literal, audited execution-only substitutions; model and metric code untouched.
    source=inspect.getsource(p2.run)
    replacements={"def run(out,expand=False):":"def run_subset(out,expand=False):",
                  "for seed in CONFIG['seeds']:":"for seed in WORKER_SEEDS:",
                  "    nc.dump(folder/'checks.json',checks())":"    # Parent owns the shared checks file.",
                  "    summarize(folder)":"    # Parent summarizes only after every worker has exited successfully."}
    for before,after in replacements.items():
        assert source.count(before)==1,(before,source.count(before))
        source=source.replace(before,after)
    ns=dict(vars(p2),WORKER_SEEDS=seeds,scores=fast.scores)
    generated=OUT/'p2/parallel_execution'/('worker_'+str(seeds[0])+'.py')
    generated.write_text(source,encoding='utf-8')
    exec(compile(source,str(generated),'exec'),ns)
    ns['run_subset'](OUT,True)

def run():
    folder=OUT/'p2/parallel_execution'
    assert json.loads((OUT/'p2/screen_decision.json').read_text())['expand']
    freeze(folder,dict(partitions=PARTITIONS,reason='Independent seed processes; no training/math/config/gate change.',
                      timing='Training seconds under concurrency are descriptive only; P4 runs separately.'),
           [__file__,ROOT/'experiments/neuro_mechanism_fast_metrics.py',ROOT/'experiments/neuro_mechanism_p2_resume.py'],
           [OUT/'p2/independent_process_replay.json'])
    if not (folder/'screen_decision_initial.json').exists():shutil.copy2(OUT/'p2/screen_decision.json',folder/'screen_decision_initial.json')
    children=[]
    try:
        for i,seeds in enumerate(PARTITIONS):
            log=open(folder/f'worker{i}.log','w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-m','experiments.neuro_mechanism_p2_parallel','--seeds',*[str(s) for s in seeds]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((proc,log))
        while any(proc.poll() is None for proc,_ in children):
            failed=[proc.returncode for proc,_ in children if proc.poll() not in (None,0)]
            if failed:raise RuntimeError(f'P2 worker failed: {failed}; see worker logs')
            count=len(list((OUT/'p2/runs').glob('seed*/*/metrics.json')))
            print(f'P2 full-grid progress: {count}/2420 paired cells',flush=True)
            time.sleep(15)
        assert all(proc.returncode==0 for proc,_ in children)
    finally:
        for proc,log in children:
            if proc.poll() is None:proc.terminate();proc.wait()
            log.close()
    summarize(OUT/'p2')
    assert json.loads((OUT/'p2/STATUS.json').read_text())['rows']==4840
    nc.dump(folder/'completion.json',dict(status='COMPLETE',workers=3,rows=4840))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',nargs='+',type=int);args=parser.parse_args()
    worker(args.seeds) if args.seeds else run()
