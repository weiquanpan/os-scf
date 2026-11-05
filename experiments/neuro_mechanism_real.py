"""P1 ridge sensitivity and P3 frozen-checkpoint ODE diagnostics."""
from experiments.neuro_mechanism_utils import *
from experiments.neuro_mechanism_fast_metrics import sw,scores
import argparse
DATASETS=['lendingclub','bikesharing','taxi']
SPLITS=[3101,3102,3103]
BASE={'lendingclub':1600,'bikesharing':1200,'taxi':1600}
P1=dict(stage='P1',datasets=DATASETS,splits=SPLITS,seeds=[107,108,109],alpha=0.,
        epsilons=[.1,.01,.001,.0001,.00001],steps=BASE,sample_steps=80,reference='val',
        projections=1024,projection_seeds=PROJ_SEEDS,
        notes=['Fixed budgets and no selection; matched index/time/free-noise streams.',
               'All epsilon settings retained; monotone worsening is a hypothesis, not a required outcome.',
               'Covariance spectra evaluated in float64; training follows original float32 normalizer.',
               'Linear dependence implies ambient singularity; finite free condition number is measured, not assumed.',
               'Three overlapping splits do not constitute three independent populations.',
               'Independent dataset workers may share GPU; optimizer seconds are not a fair cost benchmark.'])
P3=dict(stage='P3',datasets=DATASETS,splits=SPLITS,seeds=[101,102,103,104,105],
        sample_steps=[4,8,16,32,80],reference='test',projections=1024,projection_seeds=PROJ_SEEDS,
        jacobian_states=64,jacobian_times=[0.,.5,.9,.975,.9875],
        notes=['Reuse all 90 frozen main flow checkpoints; no training or test-based model selection.',
               'S80 samples and original 128-direction metrics must reproduce archived results.',
               'Jacobian norms at visited states are empirical local diagnostics, not global Lipschitz constants.',
               'Norms reported in network z coordinates and original standardized numeric coordinates.',
               'Timing synchronized with warmup and five repetitions; desktop GPU not exclusive.',
               'Repeated S and projection seeds are not independent training replicates.'])

def inputs(include_checkpoints=False):
    ps=[]
    for ds,sp in itertools.product(DATASETS,SPLITS):
        base=MAIN/f'split{sp}'/'data'/ds
        ps += [base/'train.csv',base/'val.csv']
        if include_checkpoints:
            ps.append(base/'test.csv')
            for s,m in itertools.product(P3['seeds'],['osscf','direct']):
                ps.append(MAIN/f'split{sp}'/'final'/ds/f'seed{s}'/f'{m}.pt')
    return ps

def spectrum(b):
    rows=[]
    for space,x in [('ambient',b.train_x),('free',b.train_x[:,b.free])]:
        x=array(x);z=(x-x.mean(0))/x.std(0,ddof=1).clip(1e-5)
        C=np.cov(z.T);ev=np.linalg.eigvalsh(C);tol=ev[-1]*len(ev)*np.finfo(float).eps
        rows.append(dict(dataset=b.name,split=b.split,space=space,dimension=len(ev),
                         eigenvalues=ev.tolist(),min_eigenvalue=float(ev[0]),max_eigenvalue=float(ev[-1]),
                         numerical_rank=int((ev>tol).sum()),rank_tolerance=float(tol),
                         condition_number=float(ev[-1]/ev[0]) if ev[0]>tol else None,
                         condition_number_status='finite' if ev[0]>tol else 'numerically_singular'))
    return rows

def p1(datasets=None,finalize=True):
    folder=OUT/'p1';freeze(folder,P1,[__file__,ROOT/'experiments/neuro_mechanism_utils.py',ROOT/'experiments/neuro_mechanism_fast_metrics.py',ROOT/'experiments/neuro_mechanism_execution.py'],inputs())
    spectra=[]
    for ds,sp in itertools.product(datasets or DATASETS,SPLITS):
        b=nc.bundle(ds,sp,MAIN,'val');spectra+=spectrum(b)
        kh=b.spec.one_hot(b.train_k)
        for s in P1['seeds']:
            k=nc.empirical_k(b,len(b.ref_x),s+500);ks=b.spec.one_hot(k)
            for eps in P1['epsilons']:
                probes=[]
                for method in ['osscf','direct_repair']:
                    dest=folder/'runs'/ds/f'split{sp}'/f'seed{s}'/f'eps{eps}'/method
                    dest.mkdir(parents=True,exist_ok=True)
                    if (dest/'metrics.json').exists():continue
                    target=b.train_x[:,b.free] if method=='osscf' else b.train_x
                    model,norm,payloads=train_paired(target,kh,len(b.free),s,0.,BASE[ds],eps=eps,batch=b.cfg.batch_size)
                    payload=payloads[BASE[ds]];probes.append(payload['rng_probes'])
                    torch.save(payload,dest/'checkpoint.pt')
                    free=sample_paired(model,norm,ks,len(b.free),s)[:,:len(b.free)]
                    x=nc.decode(b,free,k)
                    nc.frame(b,x,k).to_csv(dest/'samples.csv',index=False)
                    row=dict(dataset=ds,split=sp,seed=s,eps=eps,method=method,**scores(x,b.ref_x,b.train_x,len(b.free)),
                             W_operator_norm=float(torch.linalg.matrix_norm(norm.W,ord=2)),
                             optimizer_seconds=payload['optimizer_seconds'],parameters=payload['parameters'])
                    nc.dump(dest/'metrics.json',row)
                if len(probes)==2:assert probes[0]==probes[1]
                print(f'P1 {ds} split={sp} seed={s} eps={eps} complete',flush=True)
    if not finalize:return
    nc.dump(folder/'spectra.json',spectra)
    pd.DataFrame([{k:v for k,v in r.items() if k!='eigenvalues'} for r in spectra]).to_csv(folder/'spectra.csv',index=False)
    rows=[json.loads(p.read_text()) for p in (folder/'runs').glob('*/*/*/*/*/metrics.json')]
    d=pd.DataFrame(rows);d.to_csv(folder/'metrics.csv',index=False)
    d.groupby(['dataset','eps','method'])[['full_sw','free_sw','dep_sw','W_operator_norm']].mean().to_csv(folder/'summary.csv')
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',rows=len(d),spectra=len(spectra)))

def load_main(ds,sp,s,method,b):
    name='osscf' if method=='osscf' else 'direct'
    path=MAIN/f'split{sp}'/'final'/ds/f'seed{s}'/f'{name}.pt'
    payload=torch.load(path,weights_only=False,map_location=nc.DEVICE)
    model,norm=restore(payload,b.spec.one_hot(b.train_k[:1]).shape[1])
    return model,norm,payload

def jacobians(b,model,norm,k,s):
    # Start with exactly the full generation shape, then keep the first 64 states.
    with nc.isolated_rng(s+600):z=torch.randn(len(k),model.dim,device=nc.DEVICE)[:64]
    kh=b.spec.one_hot(k[:64]);rows=[];clipped=0;total=0
    for j in range(80):
        t=torch.full((len(z),1),j/80,device=nc.DEVICE)
        if j in [0,40,72,78,79]:
            zz=z.detach().requires_grad_(True);v=model(zz,kh,t)
            J=torch.stack([torch.autograd.grad(v[:,i].sum(),zz,retain_graph=True)[0] for i in range(model.dim)],1)
            A=norm.W_inv if norm.W_inv is not None else torch.eye(model.dim,device=nc.DEVICE)
            B=norm.W if norm.W is not None else torch.eye(model.dim,device=nc.DEVICE)
            JS=A[None]@J@B[None]
            for coordinate,matrix in [('z',J),('standardized_numeric',JS)]:
                values=torch.linalg.matrix_norm(matrix,ord=2).detach().cpu().numpy()
                rows.append(dict(time=j/80,coordinate=coordinate,n=len(z),mean=float(values.mean()),
                                 p95=float(np.quantile(values,.95)),max=float(values.max())))
        with torch.no_grad():
            proposal=z+model(z,kh,t)/80;clipped+=int((proposal.abs()>6).sum());total+=proposal.numel();z=proposal.clamp(-6,6)
    for row in rows:row['trajectory_clip_fraction']=clipped/total
    return rows

def p3():
    folder=OUT/'p3';freeze(folder,P3,[__file__,ROOT/'experiments/neuro_mechanism_utils.py',ROOT/'experiments/neuro_mechanism_fast_metrics.py'],inputs(True))
    for ds,sp in itertools.product(DATASETS,SPLITS):
        b=nc.bundle(ds,sp,MAIN,'test');sd=array(b.train_x).std(0).clip(1e-6)
        for s in P3['seeds']:
            k=nc.empirical_k(b,len(b.ref_x),s+500)
            for method in ['osscf','direct_repair']:
                dest=folder/'runs'/ds/f'split{sp}'/f'seed{s}'/method;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'metrics.json').exists():continue
                model,norm,payload=load_main(ds,sp,s,method,b);kind='hard' if method=='osscf' else 'direct'
                records=[];samples={}
                def generate(steps):
                    x=nc.sample_flow(b,model,norm,k,s+600,kind,steps)
                    return nc.decode(b,x[:,b.free],k) if kind=='direct' else x
                for steps in P3['sample_steps']:
                    x=generate(steps);seconds=[]
                    for _ in range(5):
                        nc.sync();start=time.perf_counter();generate(steps);nc.sync();seconds.append(time.perf_counter()-start)
                    mean,vals=sw(array(x)/sd,array(b.ref_x)/sd)
                    row=dict(dataset=ds,split=sp,seed=s,method=method,sample_steps=steps,full_sw=mean,
                             projection_sd=float(np.std(vals,ddof=1)),sampling_seconds_median=float(np.median(seconds)),
                             sampling_seconds_min=min(seconds),sample_rows=len(k),parameters=sum(p.numel() for p in model.parameters()),
                             alpha=payload['alpha'],training_steps=payload['steps'])
                    row.update({f'full_sw_proj{i}':v for i,v in enumerate(vals)})
                    if steps==80:
                        archived=MAIN/f'split{sp}'/'final'/ds/f'seed{s}'/f'{method}.csv'
                        saved=pd.read_csv(archived)[b.cols].to_numpy(np.float32)
                        row['replay_max_abs_error']=float(np.max(np.abs(x.cpu().numpy()-saved)))
                        assert row['replay_max_abs_error']==0,row
                        old=json.loads(archived.with_suffix('.metrics.json').read_text())
                        row['original_sw_replay_abs_error']=abs(nc.metrics(b,x,k)['sw']-old['sw'])
                        assert row['original_sw_replay_abs_error']<1e-10
                    records.append(row);samples[f'steps{steps}']=x.cpu().numpy()
                jr=[dict(dataset=ds,split=sp,seed=s,method=method,**r) for r in jacobians(b,model,norm,k,s)]
                nc.dump(dest/'jacobians.json',jr);np.savez_compressed(dest/'samples.npz',**samples)
                nc.dump(dest/'metrics.json',records)
            print(f'P3 {ds} split={sp} seed={s} complete',flush=True)
    rows=[r for p in (folder/'runs').glob('*/*/*/*/metrics.json') for r in json.loads(p.read_text())]
    d=pd.DataFrame(rows);d.to_csv(folder/'metrics.csv',index=False)
    d.groupby(['dataset','method','sample_steps'])[['full_sw','sampling_seconds_median']].mean().to_csv(folder/'summary.csv')
    jr=[r for p in (folder/'runs').glob('*/*/*/*/jacobians.json') for r in json.loads(p.read_text())]
    pd.DataFrame(jr).to_csv(folder/'jacobians.csv',index=False)
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',rows=len(d),jacobian_rows=len(jr),checkpoints=90,
                                    max_sample_replay_error=float(d.replay_max_abs_error.max()),max_metric_replay_error=float(d.original_sw_replay_abs_error.max())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['p1','p3']);p.add_argument('--dataset',choices=DATASETS);args=p.parse_args()
    if args.stage=='p3':p3()
    elif args.dataset:p1([args.dataset],False)
    else:
        # Parent initializes shared provenance before disjoint workers start.
        freeze(OUT/'p1',P1,[__file__,ROOT/'experiments/neuro_mechanism_utils.py',ROOT/'experiments/neuro_mechanism_fast_metrics.py',ROOT/'experiments/neuro_mechanism_execution.py'],inputs())
        from experiments.neuro_mechanism_execution import launch
        launch('experiments.neuro_mechanism_real',['p1'],OUT/'p1/worker_logs')
        p1()
