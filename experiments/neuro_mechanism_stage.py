"""Canonical JSON configuration comparison; tuple/list representation only."""
from experiments.neuro_mechanism_utils import *
from experiments.neuro_mechanism_execution import launch
import argparse

def canonical(config):return json.loads(json.dumps(config))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['p1','p3','p0','p4_curve']);parser.add_argument('--dataset');args=parser.parse_args()
    if not args.dataset:
        freeze(OUT/'execution_repairs'/args.stage,dict(change='Canonical JSON tuple/list representation before provenance equality check.',
               experimental_values_changed=False,model_code_changed=False),[__file__])
    if args.stage in ['p1','p3']:
        from experiments import neuro_mechanism_real as stage
        stage.P1=canonical(stage.P1);stage.P3=canonical(stage.P3)
        if args.stage=='p3':stage.p3()
        elif args.dataset:stage.p1([args.dataset],False)
        else:
            # Freeze agrees with the already saved JSON; preserve the initial failure logs.
            old=OUT/'p1/worker_logs';saved=OUT/'p1/startup_failure_logs'
            if old.exists() and not saved.exists():
                saved.mkdir(parents=True)
                for p in old.glob('*.log'):shutil.copy2(p,saved/p.name)
            freeze(OUT/'p1',stage.P1,[stage.__file__,ROOT/'experiments/neuro_mechanism_utils.py',ROOT/'experiments/neuro_mechanism_fast_metrics.py',ROOT/'experiments/neuro_mechanism_execution.py'],stage.inputs())
            launch('experiments.neuro_mechanism_stage',['p1'],OUT/'p1/worker_logs')
            stage.p1()
    elif args.stage=='p0':
        from experiments import neuro_mechanism_p0 as stage
        stage.CONFIG=canonical(stage.CONFIG);stage.run()
    else:
        from experiments import neuro_mechanism_cost_curve as stage
        stage.CONFIG=canonical(stage.CONFIG);stage.run()
