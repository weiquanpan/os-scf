"""Sequential validation quality/time curves at preselected alphas, fresh seed 110."""
from experiments.neuro_mechanism_utils import *
from experiments.neuro_mechanism_fast_metrics import sw
from experiments.neuro_mechanism_p0 import fit_prefixes
from experiments.neuro_mechanism_real import DATASETS,BASE
from experiments.neurocomputing.models import TabSynAdapter

CONFIG=dict(stage='P4_curve',datasets=DATASETS,split=3101,seed=110,alphas='P0 method-specific selection, fixed before this run',
            budgets=[.25,.5,1.,2.,4.],projections=1024,projection_seeds=list(PROJ_SEEDS),
            notes=['Sequential GPU fits after the main P4 benchmark has finished.',
                   'Same dataset and fresh seed as P4; two methods x three datasets = six optimizer trajectories.',
                   'Time includes preprocessing, optimizer, cumulative validation checkpoints, and checkpoint writes.',
                   'P0 external alpha/step search overhead remains a separately reported expense.',
                   'TabSyn has a final P4 quality/time point only; no invented intermediate frontier.',
                   'Separately remeasure TabSyn+Phi end-to-end sampling, including categorical fixing and reconstruction.'])
def repaired_timing():
    rows=[];cfg=json.loads((MAIN/'protocol.json').read_text())['tabsyn']
    for ds in DATASETS:
        b=nc.bundle(ds,3101,MAIN,'val');nc.seed(110);model=TabSynAdapter(b,False,cfg)
        model.load_bundle(OUT/'p4/runs'/ds/'tabsyn/checkpoint.pt')
        def sample():
            x,k=model.sample(5000,50110);k=nc.fixk(b,k);return nc.decode(b,x[:,b.free],k)
        sample();times=[]
        for _ in range(5):
            nc.sync();start=time.perf_counter();sample();nc.sync();times.append(time.perf_counter()-start)
        rows.append(dict(dataset=ds,method='tabsyn_repair',sampling_5000_median_seconds=float(np.median(times)),
                         sampling_5000_min_seconds=min(times),includes_reconstruction=True))
        del model
    table=pd.DataFrame(rows);table.to_csv(OUT/'p4/repaired_sampling_timing.csv',index=False)
    d=pd.read_csv(OUT/'p4/metrics.csv')
    for row in rows:
        mask=(d.dataset==row['dataset'])&(d.method=='tabsyn_repair')
        for k in ['sampling_5000_median_seconds','sampling_5000_min_seconds']:d.loc[mask,k]=row[k]
    d.to_csv(OUT/'p4/metrics_with_repair_timing.csv',index=False)
def run():
    folder=OUT/'p4_curve'
    freeze(folder,CONFIG,[__file__,ROOT/'experiments/neuro_mechanism_p0.py',ROOT/'experiments/neuro_mechanism_fast_metrics.py'],
           [OUT/'p0/selection'/ds/'split3101/selected.json' for ds in DATASETS])
    assert (OUT/'p4/STATUS.json').exists()
    if (folder/'STATUS.json').exists():
        assert json.loads((folder/'STATUS.json').read_text())['status']=='COMPLETE'
        print('P4 curves already complete; preserving timing records.',flush=True)
        return
    repaired_timing()
    for ds in DATASETS:
        b=nc.bundle(ds,3101,MAIN,'val');sd=array(b.train_x).std(0).clip(1e-6)
        selected=json.loads((OUT/'p0/selection'/ds/'split3101/selected.json').read_text())
        for method in ['osscf','direct_repair']:
            dest=folder/'runs'/ds/method;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'metrics.json').exists():continue
            records=[];kind='hard' if method=='osscf' else 'direct';alpha=selected[method]['alpha']
            k=nc.empirical_k(b,len(b.ref_x),610);budgets=[int(BASE[ds]*v) for v in CONFIG['budgets']]
            nc.sync();start=time.perf_counter()
            def callback(model,norm,payload):
                x=nc.sample_flow(b,model,norm,k,710,kind)
                if kind=='direct':x=nc.decode(b,x[:,b.free],k)
                value,projections=sw(array(x)/sd,array(b.ref_x)/sd)
                torch.save(payload,dest/f'steps{payload["steps"]}.pt');nc.sync()
                records.append(dict(dataset=ds,method=method,alpha=alpha,steps=payload['steps'],full_sw=value,
                                    pipeline_cumulative_seconds=time.perf_counter()-start,
                                    optimizer_seconds=payload['optimizer_seconds'],parameters=payload['parameters']))
            fit_prefixes(b,kind,alpha,110,budgets,callback);nc.dump(dest/'metrics.json',records)
            print(f'P4 curve {ds} {method} complete',flush=True)
    rows=[r for p in (folder/'runs').glob('*/*/metrics.json') for r in json.loads(p.read_text())]
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',rows=len(rows),optimizer_trajectories=6,exclusive_GPU=False))
if __name__=='__main__':run()
