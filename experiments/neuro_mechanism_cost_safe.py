"""Run P4 with the adapter's event log redirected to the new study directory."""
from experiments.neuro_mechanism_utils import *
from experiments import neuro_mechanism_cost_validity as cost
if __name__=='__main__':
    freeze(OUT/'p4/log_routing',dict(change='Set in-memory bundle.out to the follow-up output before TabSyn.fit; frozen main logs untouched.'),[__file__])
    if (OUT/'p4/STATUS.json').exists():
        assert json.loads((OUT/'p4/STATUS.json').read_text())['status']=='COMPLETE'
        print('P4 already complete; preserving timing records.',flush=True)
        raise SystemExit(0)
    original_bundle=nc.bundle
    def bundle(*args,**kwargs):
        result=original_bundle(*args,**kwargs);result.out=OUT/'p4';return result
    nc.bundle=bundle
    cost.p4()
