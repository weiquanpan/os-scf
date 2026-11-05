"""P4 sequential same-machine cost audit and P5 implied-inequality checks."""
from experiments.neuro_mechanism_utils import *
from experiments.neuro_mechanism_fast_metrics import scores
from experiments.neuro_mechanism_real import DATASETS,SPLITS
from experiments.neurocomputing.models import TabSynAdapter
import argparse,gc,subprocess

CONFIG4=dict(stage='P4',datasets=DATASETS,split=3101,seed=110,reference='val',sampling_rows=5000,timing_repetitions=5,
             models=['osscf','direct_repair','tabsyn'],flow_settings='method-specific P0 selection',
             tabsyn_config=json.loads((MAIN/'protocol.json').read_text())['tabsyn'],
             notes=['Fresh sequential training on the same machine; one seed/split per dataset for cost only.',
                    'Desktop GPU cannot be certified exclusive; no exclusive-device or universal speed-ratio claim.',
                    'Training pipeline timing includes preprocessing, fitting, internal selection, and checkpoint save.',
                    'Flow external search cost reported separately from P0; TabSyn internal validation included in its fit.',
                    'Do not equate different achieved validation SW; compare quality and time jointly.',
                    'P0 candidate trajectories supply flow SW/time frontiers; TabSyn is a final point, not a traced frontier.',
                    'TabSyn and TabSyn+Phi share a trained model; report both outputs without double-counting training.'])
CONFIG5=dict(stage='P5',datasets=DATASETS,splits=SPLITS,seeds=list(range(101,106)),
             methods=['osscf','direct_repair','tabsyn','tabsyn_repair','free_tabsyn','tabddpm'],
             inequalities={'lendingclub':['installment*term_months >= loan_amnt'],
                           'bikesharing':['cnt >= registered','cnt >= casual'],
                           'taxi':['total_amount >= fare_amount']},
             tolerance='max(0.01, 1e-6*abs(rhs)) in original units',
             notes=['Reuses existing test samples; new diagnostic, no training and no selection.',
                    'Real reference violation rates reported; especially important for noisy Taxi rows.',
                    'Structural validity is shared by exact-reconstruction methods, not an OS-only benefit.',
                    'Conditional SW and existing TSTR feature diagnostics reuse completed C1 rather than duplicate evidence.',
                    'No leakage-prone Taxi task using constituents of the total as predictors.'])

def gpu():
    try:return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.SubprocessError):return 'unavailable'

def timing(fn):
    fn();times=[]
    for _ in range(5):
        nc.sync();start=time.perf_counter();fn();nc.sync();times.append(time.perf_counter()-start)
    return dict(sampling_5000_median_seconds=float(np.median(times)),sampling_5000_min_seconds=min(times),sampling_5000_all_seconds=times)

def p4():
    folder=OUT/'p4';inputs=[OUT/'p0/selection'/ds/'split3101/selected.json' for ds in DATASETS]
    freeze(folder,CONFIG4,[__file__,ROOT/'experiments/neuro_mechanism_utils.py',ROOT/'experiments/neuro_mechanism_fast_metrics.py'],inputs)
    for ds in DATASETS:
        b=nc.bundle(ds,3101,MAIN,'val');choice=json.loads((OUT/'p0/selection'/ds/'split3101/selected.json').read_text())
        for method in CONFIG4['models']:
            dest=folder/'runs'/ds/method;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'metrics.json').exists():continue
            gc.collect();torch.cuda.empty_cache();nc.seed(110);nc.sync()
            snapshot=gpu();initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            if method in ['osscf','direct_repair']:
                kind='hard' if method=='osscf' else 'direct';cfg=choice[method]
                model,norm,seconds,hist=nc.train_flow(b,kind,cfg['alpha'],cfg['steps'],110)
                torch.save(dict(model=model.state_dict(),normalizer=norm,**cfg,seed=110),dest/'checkpoint.pt')
                nc.sync();wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
                def sample(n):
                    k=nc.empirical_k(b,n,610);x=nc.sample_flow(b,model,norm,k,710,kind)
                    if kind=='direct':x=nc.decode(b,x[:,b.free],k)
                    return x,k
                train_parameters=inference_parameters=sum(p.numel() for p in model.parameters())
                info=dict(alpha=cfg['alpha'],steps=cfg['steps'],optimizer_seconds=seconds)
            else:
                model=TabSynAdapter(b,False,CONFIG4['tabsyn_config']);info=model.fit(dest/'fit',110)
                model.save_bundle(dest/'checkpoint.pt');nc.sync();wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
                def sample(n):return model.sample(n,50110)
                train_parameters=sum(p.numel() for m in [model.vae,model.d] for p in m.parameters())
                inference_parameters=sum(p.numel() for m in [model.decoder,model.d] for p in m.parameters())
            measured=timing(lambda:sample(5000));x,k=sample(len(b.ref_x))
            nc.frame(b,x,k).to_csv(dest/'samples.csv',index=False)
            row=dict(dataset=ds,method=method,seed=110,split=3101,training_pipeline_seconds=wall,
                     train_parameters=train_parameters,inference_parameters=inference_parameters,
                     training_peak_allocated_mib=peak/2**20,training_peak_increment_mib=(peak-initial)/2**20,
                     gpu_before=snapshot,**measured,**scores(x,b.ref_x,b.train_x,len(b.free)),fit_info=info)
            nc.dump(dest/'metrics.json',row)
            if method=='tabsyn':
                kk=nc.fixk(b,k);xx=nc.decode(b,x[:,b.free],kk);nc.frame(b,xx,kk).to_csv(dest/'repaired_samples.csv',index=False)
                nc.dump(dest/'repaired_metrics.json',dict(row,method='tabsyn_repair',**scores(xx,b.ref_x,b.train_x,len(b.free))))
            del model,x,k
            print(f'P4 {ds} {method} complete',flush=True)
    rows=[]
    for p in (folder/'runs').glob('*/*/*metrics.json'):
        row=json.loads(p.read_text());row.pop('fit_info');row.pop('sampling_5000_all_seconds');rows.append(row)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',fits=9,metric_rows=len(rows),exclusive_GPU=False))

def violation(frame,ds):
    if ds=='lendingclub':
        term=pd.to_numeric(frame['term_months'].astype(str).str.extract(r'(\d+)')[0]).to_numpy()
        tests=[('payment_total_ge_principal',frame.installment.to_numpy()*term,frame.loan_amnt.to_numpy())]
    elif ds=='bikesharing':tests=[('cnt_ge_registered',frame.cnt.to_numpy(),frame.registered.to_numpy()),('cnt_ge_casual',frame.cnt.to_numpy(),frame.casual.to_numpy())]
    else:tests=[('total_ge_fare',frame.total_amount.to_numpy(),frame.fare_amount.to_numpy())]
    result=[]
    for name,lhs,rhs in tests:
        defined=np.isfinite(lhs)&np.isfinite(rhs);error=np.maximum(rhs[defined]-lhs[defined],0)
        bad=error>np.maximum(.01,1e-6*np.abs(rhs[defined]))
        result.append(dict(constraint=name,rows=len(frame),defined_rows=int(defined.sum()),undefined_rate=float((~defined).mean()),
                           violations=int(bad.sum()),violation_rate=float(bad.mean()) if len(bad) else None,
                           invalid_or_violation_rate=float((bad.sum()+(~defined).sum())/len(frame)),
                           mean_positive_deficit=float(error.mean()) if len(error) else None,max_positive_deficit=float(error.max()) if len(error) else None))
    return result

def p5():
    folder=OUT/'p5';inputs=[]
    for ds,sp,s,m in itertools.product(DATASETS,SPLITS,CONFIG5['seeds'],CONFIG5['methods']):
        inputs.append(MAIN/f'split{sp}'/'final'/ds/f'seed{s}'/f'{m}.csv')
    inputs += [MAIN/f'split{sp}'/'data'/ds/'test.csv' for ds,sp in itertools.product(DATASETS,SPLITS)]
    freeze(folder,CONFIG5,[__file__,ROOT/'experiments/neuro_mechanism_utils.py'],inputs)
    rows=[]
    for ds,sp in itertools.product(DATASETS,SPLITS):
        ref=pd.read_csv(MAIN/f'split{sp}'/'data'/ds/'test.csv')
        rows.extend(dict(dataset=ds,split=sp,seed=0,method='real_reference',**r) for r in violation(ref,ds))
        for s,m in itertools.product(CONFIG5['seeds'],CONFIG5['methods']):
            frame=pd.read_csv(MAIN/f'split{sp}'/'final'/ds/f'seed{s}'/f'{m}.csv')
            rows.extend(dict(dataset=ds,split=sp,seed=s,method=m,**r) for r in violation(frame,ds))
    d=pd.DataFrame(rows);d.to_csv(folder/'metrics.csv',index=False)
    d.groupby(['dataset','method','constraint'])[['violation_rate','mean_positive_deficit']].mean().to_csv(folder/'summary.csv')
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',rows=len(d),trained_models=0,
                                    conditional_diagnostics_reused='outputs/neurocomputing_advantage_c1_v1'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['p4','p5']);args=parser.parse_args()
    p4() if args.stage=='p4' else p5()
