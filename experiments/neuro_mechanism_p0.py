"""Expanded, equal-search-budget method-specific vs shared validation selection."""
from experiments.neuro_mechanism_utils import *
from experiments.neuro_mechanism_fast_metrics import sw,scores
from experiments.neurocomputing.evaluate import utility
from experiments.neuro_mechanism_real import DATASETS,SPLITS,BASE
import argparse

CONFIG=dict(stage='P0',datasets=DATASETS,splits=SPLITS,selection_seeds=[91,92],final_seeds=[101,102,103,104,105],
            alphas=[0.,.25,.5,.75,1.],step_multipliers=[.25,.5,1.,2.,4.],base_steps=BASE,
            width=128,lr=.001,sample_steps=80,projections=1024,projection_seeds=PROJ_SEEDS,
            selection_metric='mean validation Full SW over 3 projection sets and 2 selection seeds',
            selectors=['method_specific','shared'],test_access='after all nine selection tables are complete',
            notes=['Same 25 configurations and two seeds searched for each method; no test-based choice.',
                   'Training reproduces the original global RNG recipe. Save budget prefixes from one max-budget trajectory.',
                   'Shared and method-specific selectors computed from exactly the same validation candidates.',
                   'Reusing a checkpoint is allowed only if dataset/split/seed/alpha/steps match.',
                   'Main test sets are previously examined; this is a protocol sensitivity analysis, not fresh external validation.',
                   'Taxi uniform-quantile SW uses train-fitted midpoint empirical CDF; secondary geometry, not selected on.',
                   'Split-block sign flips have only 8 configurations; overlapping random splits limit inference.',
                   'Full candidate search time and selected training time are reported separately; desktop GPU not exclusive.',
                   'Independent dataset workers share GPU; P0 fit times are descriptive, and P4 provides separate sequential timing.'])

def candidate_tag(ds,sp,s,method,alpha):return f'{ds}_split{sp}_seed{s}_{method}_a{alpha}'

def fit_prefixes(b,kind,alpha,seed,budgets,callback):
    """Identical optimizer path to common.train_flow; callback excludes RNG/time changes."""
    nc.seed(seed);target=b.train_x[:,b.free] if kind=='hard' else b.train_x
    norm=make_free_flow_normalizer(target,whiten_shrinkage=alpha);z=norm.transform(target);kh=b.spec.one_hot(b.train_k)
    model=ConditionalVelocity(z.shape[1],kh.shape[1],128).to(nc.DEVICE);opt=torch.optim.Adam(model.parameters(),lr=.001)
    nc.sync();start=time.perf_counter();paused=0.;history=[]
    for j in range(max(budgets)):
        ix=torch.randint(len(z),(b.cfg.batch_size,),device=nc.DEVICE);z1=z[ix];z0=torch.randn_like(z1);t=torch.rand(len(ix),1,device=nc.DEVICE)
        loss=(model((1-t)*z0+t*z1,kh[ix],t)-(z1-z0)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite flow loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if j%100==0:history.append(dict(step=j+1,loss=float(loss.detach())))
        if j+1 in budgets:
            nc.sync();at=time.perf_counter()
            payload=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},normalizer=norm,
                         alpha=alpha,steps=j+1,seed=seed,optimizer_seconds=at-start-paused,history=history.copy(),
                         parameters=sum(p.numel() for p in model.parameters()))
            with nc.isolated_rng(seed+98765):callback(model,norm,payload)
            nc.sync();paused+=time.perf_counter()-at
    return model,norm

def prefix_check():
    b=nc.bundle('bikesharing',3101,MAIN,'val');saved={}
    def callback(m,n,p):saved[p['steps']]=p
    m,n=fit_prefixes(b,'hard',.25,9191,[2,4],callback)
    errors=[]
    for steps in [2,4]:
        ref,_,_,_=nc.train_flow(b,'hard',.25,steps,9191)
        errors.append(max(float(torch.max(torch.abs(v-saved[steps]['model'][k]))) for k,v in ref.state_dict().items()))
    assert max(errors)==0,errors
    return dict(status='PASS',prefix_2_and_4_step_max_state_error=max(errors))

def select(folder,datasets=None):
    for ds,sp in itertools.product(datasets or DATASETS,SPLITS):
        b=nc.bundle(ds,sp,MAIN,'val');sd=array(b.train_x).std(0).clip(1e-6)
        budgets=[int(BASE[ds]*m) for m in CONFIG['step_multipliers']]
        dest=folder/'selection'/ds/f'split{sp}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'selected.json').exists():continue
        for alpha,s,method in itertools.product(CONFIG['alphas'],CONFIG['selection_seeds'],['osscf','direct_repair']):
            tag=candidate_tag(ds,sp,s,method,alpha);base=dest/'candidates'/tag;base.mkdir(parents=True,exist_ok=True)
            if all((base/f'steps{v}.json').exists() for v in budgets):continue
            kind='hard' if method=='osscf' else 'direct';k=nc.empirical_k(b,len(b.ref_x),s+500)
            def save_candidate(model,norm,payload):
                steps=payload['steps'];x=nc.sample_flow(b,model,norm,k,s+600,kind)
                if kind=='direct':x=nc.decode(b,x[:,b.free],k)
                mean,vals=sw(array(x)/sd,array(b.ref_x)/sd)
                row=dict(dataset=ds,split=sp,seed=s,method=method,alpha=alpha,steps=steps,full_sw=mean,
                         optimizer_seconds=payload['optimizer_seconds'],parameters=payload['parameters'])
                row.update({f'full_sw_proj{i}':v for i,v in enumerate(vals)})
                torch.save(payload,base/f'steps{steps}.pt');nc.dump(base/f'steps{steps}.json',row)
            fit_prefixes(b,kind,alpha,s,budgets,save_candidate)
            print(f'P0 select {tag} complete',flush=True)
        rows=[json.loads(p.read_text()) for p in (dest/'candidates').glob('*/steps*.json')]
        d=pd.DataFrame(rows);d.to_csv(dest/'candidates.csv',index=False);selected={}
        for method in ['osscf','direct_repair']:
            q=d[d.method==method].groupby(['alpha','steps']).full_sw.mean().sort_index()
            a,t=q.idxmin();selected[method]=dict(alpha=float(a),steps=int(t),validation_sw=float(q.min()))
        q=d.groupby(['alpha','steps']).full_sw.mean().sort_index();a,t=q.idxmin()
        selected['shared']=dict(alpha=float(a),steps=int(t),validation_sw=float(q.min()))
        nc.dump(dest/'selected.json',selected)

def quantile_uniform(x,train):
    x,train=array(x),array(train);result=np.empty_like(x)
    for j in range(x.shape[1]):
        levels,counts=np.unique(train[:,j],return_counts=True);mid=(np.cumsum(counts)-counts/2)/len(train)
        result[:,j]=np.interp(x[:,j],levels,mid,left=0.,right=1.)
    return result

def checkpoint_for(b,seed,method,alpha,steps):
    name='osscf' if method=='osscf' else 'direct'
    paths=[MAIN/f'split{b.split}'/'final'/b.name/f'seed{seed}'/f'{name}.pt',
           MAIN/f'split{b.split}'/'ablation'/b.name/f'seed{seed}'/f'{name}_alpha{alpha}.pt']
    for path in paths:
        if not path.exists():continue
        payload=torch.load(path,weights_only=False,map_location=nc.DEVICE)
        if payload['alpha']==alpha and payload['steps']==steps and payload['seed']==seed:
            meta=json.loads(path.with_suffix('.metrics.json').read_text())
            payload['optimizer_seconds']=meta.get('training_seconds')
            return payload,str(path.relative_to(ROOT))
    return None,None

def final(folder,datasets=None):
    assert all((folder/'selection'/ds/f'split{sp}'/'selected.json').exists() for ds,sp in itertools.product(DATASETS,SPLITS))
    for ds,sp in itertools.product(datasets or DATASETS,SPLITS):
        b=nc.bundle(ds,sp,MAIN,'test');selected=json.loads((folder/'selection'/ds/f'split{sp}'/'selected.json').read_text())
        for seed,method in itertools.product(CONFIG['final_seeds'],['osscf','direct_repair']):
            k=nc.empirical_k(b,len(b.ref_x),seed+500);kind='hard' if method=='osscf' else 'direct'
            configs={(float(selected[key]['alpha']),int(selected[key]['steps'])) for key in [method,'shared']}
            for alpha,steps in sorted(configs):
                dest=folder/'final'/ds/f'split{sp}'/f'seed{seed}'/method/f'a{alpha}_steps{steps}'
                dest.mkdir(parents=True,exist_ok=True)
                if (dest/'metrics.json').exists():continue
                payload,reused=checkpoint_for(b,seed,method,alpha,steps)
                if payload is None:
                    model,norm,seconds,hist=nc.train_flow(b,kind,alpha,steps,seed)
                    payload=dict(model=model.state_dict(),normalizer=norm,alpha=alpha,steps=steps,seed=seed,optimizer_seconds=seconds,history=hist)
                else:model,norm=restore(payload,b.spec.one_hot(b.train_k[:1]).shape[1])
                torch.save(payload,dest/'checkpoint.pt')
                x=nc.sample_flow(b,model,norm,k,seed+600,kind)
                if kind=='direct':x=nc.decode(b,x[:,b.free],k)
                frame=nc.frame(b,x,k);frame.to_csv(dest/'samples.csv',index=False)
                row=dict(dataset=ds,split=sp,seed=seed,method=method,alpha=alpha,steps=steps,
                         reused_checkpoint=reused,optimizer_seconds=payload['optimizer_seconds'],
                         selectors=[sel for sel,key in [('method_specific',method),('shared','shared')] if selected[key]['alpha']==alpha and selected[key]['steps']==steps],
                         **scores(x,b.ref_x,b.train_x,len(b.free)),
                         **{('legacy_'+key if key in ['free_sw','dep_sw'] else key):value for key,value in nc.metrics(b,x,k).items()})
                if ds=='taxi':row['quantile_full_sw']=sw(quantile_uniform(x,b.train_x),quantile_uniform(b.ref_x,b.train_x))[0]
                nc.dump(dest/'utility.json',utility(b,frame,method,seed))
                nc.dump(dest/'metrics.json',row)
            print(f'P0 final {ds} split={sp} seed={seed} {method} complete',flush=True)

def summarize(folder):
    rows=[]
    for p in (folder/'final').glob('*/*/*/*/*/metrics.json'):
        row=json.loads(p.read_text());sels=row.pop('selectors')
        for sel in sels:rows.append(dict(row,selector=sel))
    d=pd.DataFrame(rows);d.to_csv(folder/'metrics.csv',index=False)
    d.groupby(['dataset','selector','method'])[['full_sw','free_sw','dep_sw','sw']].mean().to_csv(folder/'summary.csv')
    tests=[]
    for (ds,sel),g in d.groupby(['dataset','selector']):
        for metric in ['full_sw','free_sw','dep_sw']:
            paired=g.pivot(index=['split','seed'],columns='method',values=metric)
            delta=paired.osscf-paired.direct_repair;blocks=delta.groupby('split').mean()
            tests.append(dict(dataset=ds,selector=sel,metric=metric,**paired_summary(blocks),
                              paired_training_rows=len(delta),unit='three overlapping split means; descriptive inference',
                              seed_wins=int((delta<0).sum())))
    pd.DataFrame(tests).to_csv(folder/'split_block_statistics.csv',index=False)
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',selector_rows=len(d),unique_final_configs=len(list((folder/'final').glob('*/*/*/*/*/metrics.json')))))

def run():
    folder=OUT/'p0';ps=[MAIN/f'split{sp}'/'data'/ds/f'{part}.csv' for ds,sp,part in itertools.product(DATASETS,SPLITS,['train','val','test'])]
    freeze(folder,CONFIG,[__file__,ROOT/'experiments/neuro_mechanism_utils.py',ROOT/'experiments/neuro_mechanism_real.py',ROOT/'experiments/neuro_mechanism_fast_metrics.py',ROOT/'experiments/neuro_mechanism_execution.py'],ps)
    if (folder/'STATUS.json').exists():
        assert json.loads((folder/'STATUS.json').read_text())['status']=='COMPLETE'
        print('P0 already complete; preserving original wall-clock measurements.',flush=True)
        return
    nc.dump(folder/'prefix_check.json',prefix_check())
    from experiments.neuro_mechanism_execution import launch
    start=time.perf_counter()
    launch('experiments.neuro_mechanism_p0',['--worker-phase','select'],folder/'selection_worker_logs')
    selection_seconds=time.perf_counter()-start;start=time.perf_counter()
    launch('experiments.neuro_mechanism_p0',['--worker-phase','final'],folder/'final_worker_logs')
    nc.dump(folder/'wall_times.json',dict(selection_stage_wall_seconds=selection_seconds,final_stage_wall_seconds=time.perf_counter()-start,
                                       dataset_workers=3,includes_validation_and_IO=True,exclusive_GPU=False))
    summarize(folder)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--worker-phase',choices=['select','final']);parser.add_argument('--dataset',choices=DATASETS);args=parser.parse_args()
    if args.check_only:print(prefix_check())
    elif args.worker_phase:
        assert args.dataset
        start=time.perf_counter()
        select(OUT/'p0',[args.dataset]) if args.worker_phase=='select' else final(OUT/'p0',[args.dataset])
        nc.dump(OUT/'p0/worker_timing'/f'{args.worker_phase}_{args.dataset}.json',dict(stage=args.worker_phase,dataset=args.dataset,wall_seconds=time.perf_counter()-start))
    else:run()
