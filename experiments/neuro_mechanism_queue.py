"""Sequential GPU queue. P2 must finish before P1 -> P3 -> P0."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/neuro_mechanisms_20261103_v1'
def run(module,*args):
    if args and args[0] in ['p1','p3','p0'] and (OUT/args[0]/'STATUS.json').exists():
        assert json.loads((OUT/args[0]/'STATUS.json').read_text())['status']=='COMPLETE'
        print('ALREADY COMPLETE',args[0],flush=True)
        return
    print('START',module,*args,flush=True)
    subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,check=True)
    print('DONE',module,*args,flush=True)
if __name__=='__main__':
    deadline=time.monotonic()+4*3600
    while not (OUT/'p2/STATUS.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('P2 completion marker not received; no later stage started')
        time.sleep(10)
    decision=json.loads((OUT/'p2/screen_decision.json').read_text())
    if decision['expand']:
        if not (OUT/'p2/parallel_execution/completion.json').exists():run('experiments.neuro_mechanism_p2_parallel')
    else:print('P2 screen gate failed; no Cartesian expansion. Proceeding to prespecified real-data diagnostics.',flush=True)
    run('experiments.neuro_mechanism_stage','p1')
    run('experiments.neuro_mechanism_stage','p3')
    run('experiments.neuro_mechanism_stage','p0')
    print('ALL CORE STAGES COMPLETE',flush=True)
