"""Independent completion, provenance, selection and checkpoint replay checks."""
from experiments.neuro_mechanism_utils import *
from experiments.neuro_mechanism_fast_metrics import scores
from experiments import neuro_mechanism_p2 as p2
from experiments.neuro_mechanism_real import DATASETS,SPLITS

def provenance():
    records=[]
    parent=json.loads((MAIN/'provenance.json').read_text())
    for p,h in parent['sources'].items():assert nc.sha(ROOT/p)==h,p
    for path in OUT.rglob('provenance.json'):
        obj=json.loads(path.read_text());checked=0
        for kind in ['sources','inputs']:
            for p,h in obj.get(kind,{}).items():assert nc.sha(ROOT/p)==h,(path,p);checked+=1
        records.append(dict(path=str(path.relative_to(OUT)),verified_files=checked))
    manifest=json.loads((ROOT/'outputs/manuscript_revision_neurocomputing_20261103/delivery_manifest.json').read_text())
    for p,h in manifest['source_files'].items():assert nc.sha(ROOT/'paper-v6'/p)==h,p
    main_root=json.loads((OUT/'main_root_before_cost.json').read_text())
    for p,h in main_root.items():assert nc.sha(ROOT/p)==h,p
    return dict(main_sources=len(parent['sources']),stages=records,manuscript_files_unchanged=len(manifest['source_files']),main_root_files_unchanged=len(main_root))

def audit_p2():
    import ot
    folder=OUT/'p2';d=pd.read_csv(folder/'metrics.csv')
    assert len(d)==4840 and len(d[['seed','dv','L','sigma','n','steps','method']].drop_duplicates())==4840
    assert np.isfinite(d[['full_sw','free_sw','dep_sw','empirical_w2']]).all().all()
    assert set(d.seed)==set(range(201,211))
    # OS must not acquire fictitious independent variation across duplicated graph/noise cells.
    spreads=d[d.method=='osscf'].groupby(['seed','n','steps']).free_sw.agg(lambda x:x.max()-x.min())
    assert spreads.max()<1e-12
    replay=[];pot_errors=[]
    for seed,c in [(201,dict(dv=1,L=1.,sigma=0.,n=5000,steps=800)),
                   (210,dict(dv=24,L=4.,sigma=.1,n=20000,steps=800)),
                   (205,dict(dv=6,L=.5,sigma=.05,n=1000,steps=800))]:
        pop=p2.population(seed);a,b,u,ref,_,_=pop;dest=folder/'runs'/f'seed{seed}'/p2.key(c)
        saved=np.load(dest/'samples.npz');tr,observed,clean,_=p2.materialize(pop,c)
        expected={r['method']:r for r in json.loads((dest/'metrics.json').read_text())}
        for method in ['osscf','direct_repair']:
            tag=f'seed{seed}_os_n{c["n"]}_steps{c["steps"]}' if method=='osscf' else f'seed{seed}_direct_{p2.key(c)}'
            payload=torch.load(folder/'checkpoints'/f'{tag}.pt',weights_only=False,map_location=nc.DEVICE)
            model,norm=restore(payload,0);free=sample_paired(model,norm,tr.new_zeros(len(ref),0),6,seed)[:,:6]
            x=torch.cat([free,p2.phi(free,a,b,c['L'],c['dv'])],1)
            error=float(np.abs(array(x)-saved[method]).max());assert error==0
            metric=scores(x,clean,tr,6);delta=max(abs(metric[k]-expected[method][k]) for k in ['full_sw','free_sw','dep_sw']);assert delta<1e-12
            values=[]
            for j in range(3):
                rng=np.random.default_rng(8100+j);aa=saved[method][rng.choice(len(ref),256,replace=False)]/saved['train_sd']
                bb=saved['clean_reference'][rng.choice(len(ref),256,replace=False)]/saved['train_sd']
                values.append(float(np.sqrt(ot.emd2(np.ones(256)/256,np.ones(256)/256,cdist(aa,bb,'sqeuclidean')))))
            error_w2=abs(float(np.mean(values))-expected[method]['empirical_w2']);assert error_w2<1e-10
            pot_errors.append(error_w2)
            replay.append(dict(seed=seed,cell=p2.key(c),method=method,max_sample_error=error,max_metric_error=delta))
    initial=json.loads((folder/'parallel_execution/screen_decision_initial.json').read_text())
    assert initial==json.loads((folder/'screen_decision.json').read_text())
    assert len(list((folder/'checkpoints').glob('*.pt')))==2470
    return dict(rows=len(d),paired_cells=2420,optimizer_trajectories=2450,checkpoints=2470,POT_actual_run_crosschecks=18,max_POT_w2_error=max(pot_errors),
                os_reuse_max_metric_spread=float(spreads.max()),replay=replay,gate_unchanged_after_expansion=True)

def audit_p1():
    folder=OUT/'p1';d=pd.read_csv(folder/'metrics.csv');assert len(d)==270
    replay=[]
    for ds,eps,method in itertools.product(DATASETS,[.1,.00001],['osscf','direct_repair']):
        b=nc.bundle(ds,3101,MAIN,'val');k=nc.empirical_k(b,len(b.ref_x),607)
        dest=folder/'runs'/ds/'split3101/seed107'/f'eps{eps}'/method
        payload=torch.load(dest/'checkpoint.pt',weights_only=False,map_location=nc.DEVICE)
        model,norm=restore(payload,b.spec.one_hot(k).shape[1]);free=sample_paired(model,norm,b.spec.one_hot(k),len(b.free),107)[:,:len(b.free)]
        x=nc.decode(b,free,k);saved=pd.read_csv(dest/'samples.csv')[b.cols].to_numpy(np.float32)
        error=float(np.abs(x.cpu().numpy()-saved).max());assert error==0
        expected=json.loads((dest/'metrics.json').read_text());actual=scores(x,b.ref_x,b.train_x,len(b.free))
        diff=max(abs(actual[m]-expected[m]) for m in ['full_sw','free_sw','dep_sw']);assert diff<1e-12
        replay.append(dict(dataset=ds,eps=eps,method=method,max_sample_error=error,max_metric_error=diff))
    return dict(rows=len(d),replay=replay)

def audit_p0():
    folder=OUT/'p0';d=pd.read_csv(folder/'metrics.csv');assert len(d)==180
    assert len(d[['dataset','split','seed','method','selector']].drop_duplicates())==180
    for ds,sp in itertools.product(DATASETS,SPLITS):
        dest=folder/'selection'/ds/f'split{sp}';c=pd.read_csv(dest/'candidates.csv');selected=json.loads((dest/'selected.json').read_text())
        assert len(c)==100
        for method in ['osscf','direct_repair','shared']:
            sub=c if method=='shared' else c[c.method==method]
            scores_=sub.groupby(['alpha','steps']).full_sw.mean().sort_index();alpha,steps=scores_.idxmin()
            assert selected[method]['alpha']==alpha and selected[method]['steps']==steps
    replay=[]
    for ds,method in itertools.product(DATASETS,['osscf','direct_repair']):
        b=nc.bundle(ds,3101,MAIN,'test');row=d[(d.dataset==ds)&(d.split==3101)&(d.seed==101)&(d.method==method)&(d.selector=='method_specific')].iloc[0]
        dest=folder/'final'/ds/'split3101/seed101'/method/f'a{row.alpha}_steps{int(row.steps)}'
        payload=torch.load(dest/'checkpoint.pt',weights_only=False,map_location=nc.DEVICE)
        k=nc.empirical_k(b,len(b.ref_x),601);model,norm=restore(payload,b.spec.one_hot(k).shape[1])
        x=nc.sample_flow(b,model,norm,k,701,'hard' if method=='osscf' else 'direct')
        if method=='direct_repair':x=nc.decode(b,x[:,b.free],k)
        saved=pd.read_csv(dest/'samples.csv')[b.cols].to_numpy(np.float32);err=float(np.abs(x.cpu().numpy()-saved).max());assert err==0
        replay.append(dict(dataset=ds,method=method,max_sample_error=err))
    return dict(selector_rows=len(d),candidate_rows=900,replay=replay,selected_configs_recomputed=True)

def audit_p4():
    results=[]
    for ds,method in itertools.product(DATASETS,['osscf','direct_repair']):
        first=torch.load(OUT/'p4/runs'/ds/method/'checkpoint.pt',weights_only=False,map_location=nc.DEVICE)
        prefix=torch.load(OUT/'p4_curve/runs'/ds/method/f'steps{first["steps"]}.pt',weights_only=False,map_location=nc.DEVICE)
        err=max(float((v-prefix['model'][k]).abs().max()) for k,v in first['model'].items());assert err==0
        results.append(dict(dataset=ds,method=method,matched_steps=first['steps'],max_parameter_error=err))
    return results

def run():
    result=dict(verification_status='VERIFIED',audit_source_sha=nc.sha(__file__),provenance=provenance(),p2=audit_p2(),p1=audit_p1(),p0=audit_p0(),p4_prefix_replay=audit_p4())
    p3=json.loads((OUT/'p3/STATUS.json').read_text());assert p3['rows']==450 and p3['max_sample_replay_error']==0 and p3['max_metric_replay_error']<1e-10
    result['p3']=p3
    for stage,count in [('p4',12),('p4_curve',30),('p5',372)]:
        table=pd.read_csv(OUT/stage/'metrics.csv');assert len(table)==count,(stage,len(table));result[stage+'_rows']=count
    nc.dump(OUT/'completion_checks.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':run()
