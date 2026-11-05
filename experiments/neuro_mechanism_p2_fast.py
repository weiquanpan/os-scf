"""Resume the unchanged P2 protocol with validated equivalent GPU SW evaluation."""
from experiments import neuro_mechanism_p2 as p2
from experiments import neuro_mechanism_fast_metrics as fast
from experiments.neuro_mechanism_utils import *
import argparse
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expand',action='store_true');args=parser.parse_args()
    folder=OUT/'p2/metric_backend'
    freeze(folder,dict(engine='CUDA float64',projection_definition='unchanged NumPy draws',tolerance=1e-12),
           [__file__,ROOT/'experiments/neuro_mechanism_fast_metrics.py'])
    if not (folder/'transition.json').exists():
        nc.dump(folder/'transition.json',dict(verification=fast.verify(),
                previously_completed_metrics={str(p.relative_to(OUT)):nc.sha(p) for p in (OUT/'p2/runs').glob('seed*/*/metrics.json')},
                reason='Runtime improvement only; no outcome inspection or configuration/gate change.',
                interrupted_process='Stopped after completed metrics were checkpointed; incomplete cell resumed from saved model.'))
    p2.scores=fast.scores
    p2.run(OUT,args.expand)
