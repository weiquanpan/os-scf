"""P2: predeclared synthetic mechanism screen, independently seeded instances."""
from experiments.neuro_mechanism_utils import *
import argparse

def cells():
    anchor=dict(dv=6,L=1.,sigma=0.,n=5000,steps=800)
    result=[]
    for axis,levels in [('dv',[1,3,6,12,24]),('L',[.5,1.,2.,4.]),('sigma',[0.,.01,.05,.1]),('n',[1000,5000,20000]),('steps',[400,800,1600])]:
        for level in levels:
            c=dict(anchor); c[axis]=level
            if c not in result: result.append(c)
    for sigma in [0.,.1]:
        c=dict(anchor,dv=24,L=4.,sigma=sigma)
        if c not in result: result.append(c)
    for sigma in [.01,.05,.1]: result.append(dict(anchor,L=.5,sigma=sigma))
    return result

CONFIG=dict(stage='P2',du=6,cells=cells(),seeds=list(range(201,211)),alpha=1.,batch=512,width=128,lr=.001,
            sample_steps=80,reference_rows=2048,projections=1024,projection_seeds=list(PROJ_SEEDS),
            exact_empirical_W2_subsample=256,exact_empirical_W2_repeats=3,
            phi='phi_j=.5*(a_j dot u)+.5*sin((2L-1)*(b_j dot u)); a,b unit rows; global row Lipschitz <= L',
            free_distribution='two-component correlated full-density Gaussian mixture; equal mixing; du=6',
            seed_unit='independent synthetic population/map, train/reference draws and training seed; paired methods and scan levels',
            normalizations='training column SD per cell, identical between methods; also fixed free SW; report standardized Jacobian',
            gate=dict(primary_cell='dv24 L1 sigma0 n5000 steps800',min_relative_free_gain=.05,min_wins=8,
                      free_delta_CI_upper_below_zero=True,relative_gain_increase_dv24_minus_dv1=.05,
                      action_pass='expand complete 240-cell dv x L x sigma x n grid at 800 steps with the same 10 seeds',
                      action_fail='retain all screen outcomes; no full Cartesian expansion; proceed to P1/P3/P0'),
            notes=['Training target includes observed dependent noise only for Direct; both evaluated after the same exact reconstruction.',
                   'Report clean and observed reference metrics separately; clean is primary mechanism outcome.',
                   'OS checkpoints reused across dv/L/noise; these cells are not independent OS runs.',
                   'L controls a rigorous per-output derivative upper bound and nonlinear frequency, not a measured global Lipschitz equality.',
                   'Random sine features replace the suggested polynomial/MLP to give a global derivative bound without truncating Gaussian support.',
                   'Same width/steps, not same parameters or wall time. No population W2 claims from empirical subsamples.',
                   'Nonlinear observed-clean empirical W2 is an offset diagnostic, not a certified support lower bound.',
                   'Affine L=.5 graph-distance floor is exact in declared evaluation coordinates.',
                   'Exploratory screen; bootstrap intervals across 10 synthetic instances; no claims about all distributions.'])

def key(c): return f"dv{c['dv']}_L{c['L']}_noise{c['sigma']}_n{c['n']}_steps{c['steps']}"

def population(seed):
    rng=np.random.default_rng(seed+10000)
    aa=rng.normal(size=(24,6)); aa/=np.linalg.norm(aa,axis=1,keepdims=True)
    bb=rng.normal(size=(24,6)); bb/=np.linalg.norm(bb,axis=1,keepdims=True)
    def draw(n):
        z=rng.normal(size=(n,6)); shared=rng.normal(size=(n,1)); component=rng.choice([-1.,1.],size=(n,1))
        return (.8*z+.3*shared+.65*component*np.array([1.,-.8,.6,-.4,.2,1.])).astype(np.float32)
    tr=draw(20000); ref=draw(2048)
    et=rng.normal(size=(20000,24)).astype(np.float32); er=rng.normal(size=(2048,24)).astype(np.float32)
    return [torch.tensor(v,device=nc.DEVICE) for v in (aa.astype(np.float32),bb.astype(np.float32),tr,ref,et,er)]

def phi(u,a,b,L,dv): return .5*(u@a[:dv].T)+.5*torch.sin((2*L-1)*(u@b[:dv].T))

def materialize(pop,c):
    a,b,u,r,et,er=pop; u=u[:c['n']];dv=c['dv'];L=c['L']
    clean=torch.cat([u,phi(u,a,b,L,dv)],1); cref=torch.cat([r,phi(r,a,b,L,dv)],1)
    tr=clean.clone(); tr[:,6:]+=c['sigma']*et[:c['n'],:dv]
    ref=cref.clone(); ref[:,6:]+=c['sigma']*er[:,:dv]
    return tr,ref,cref,clean

def jacobian_summary(pop,c,train):
    a,b,u,*_=pop;u=u[:min(2048,c['n'])];omega=2*c['L']-1;dv=c['dv']
    jac=.5*a[:dv][None]+.5*omega*torch.cos(omega*(u@b[:dv].T))[:,:,None]*b[:dv][None]
    sd=train.std(0,correction=0).clamp_min(1e-6)
    js=jac*sd[:6][None,None,:]/sd[6:][None,:,None]
    norms=torch.linalg.matrix_norm(js,ord=2)
    return dict(standardized_jacobian_mean=float(norms.mean()),standardized_jacobian_p95=float(torch.quantile(norms,.95)))

def run(out,expand=False):
    folder=out/'p2'; freeze(folder,CONFIG,[__file__,ROOT/'experiments/neuro_mechanism_utils.py'])
    nc.dump(folder/'checks.json',checks())
    design=CONFIG['cells']
    if expand:
        assert json.loads((folder/'screen_decision.json').read_text())['expand']
        design=[dict(dv=d,L=l,sigma=s,n=n,steps=800) for d,l,s,n in itertools.product([1,3,6,12,24],[.5,1.,2.,4.],[0.,.01,.05,.1],[1000,5000,20000])]
    for seed in CONFIG['seeds']:
        pop=population(seed); a,b,u,r,_,_=pop
        cached={}
        for n in sorted({c['n'] for c in design}):
            budgets=sorted({c['steps'] for c in design if c['n']==n})
            paths={v:folder/'checkpoints'/f'seed{seed}_os_n{n}_steps{v}.pt' for v in budgets}
            if not all(p.exists() for p in paths.values()):
                _,_,payloads=train_paired(u[:n],u.new_zeros(n,0),6,seed,1.,max(budgets),save_at=budgets)
                for steps,payload in payloads.items():
                    p=paths[steps];p.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,p)
            for steps,path in paths.items():
                payload=torch.load(path,weights_only=False,map_location=nc.DEVICE)
                model,norm=restore(payload,0)
                gen=sample_paired(model,norm,u.new_zeros(len(r),0),6,seed)
                cached[(n,steps)]=(gen,payload)
        for c in design:
            dest=folder/'runs'/f'seed{seed}'/key(c);dest.mkdir(parents=True,exist_ok=True)
            if (dest/'metrics.json').exists():continue
            tr,ref,cref,clean=materialize(pop,c)
            cp=folder/'checkpoints'/f'seed{seed}_direct_{key(c)}.pt'
            if cp.exists():payload=torch.load(cp,weights_only=False,map_location=nc.DEVICE);model,norm=restore(payload,0)
            else:
                model,norm,payloads=train_paired(tr,tr.new_zeros(len(tr),0),6,seed,1.,c['steps'])
                payload=payloads[c['steps']];torch.save(payload,cp)
            gen=sample_paired(model,norm,tr.new_zeros(len(ref),0),6,seed)[:,:6]
            osgen,ospayload=cached[(c['n'],c['steps'])]
            assert payload['rng_probes']['1']==ospayload['rng_probes']['1']
            sd=array(tr).std(0).clip(1e-6)
            offsets=[]
            for j in range(3):
                ix=np.random.default_rng(8100+j).choice(len(ref),256,replace=False)
                cost=cdist(array(ref)[ix]/sd,array(cref)[ix]/sd,'sqeuclidean')
                ii,jj=linear_sum_assignment(cost);offsets.append(float(np.sqrt(cost[ii,jj].mean())))
            noise_info=dict(reference_clean_empirical_offset=float(np.mean(offsets)))
            if c['L']==.5:
                B=.5*array(a[:c['dv']])*sd[:6][None,:]/sd[6:][:,None]
                residual=array(ref)[:,6:]/sd[6:]-array(ref)[:,:6]/sd[:6]@B.T
                floor2=np.einsum('ni,ij,nj->n',residual,np.linalg.inv(np.eye(c['dv'])+B@B.T),residual)
                noise_info['affine_reference_support_floor']=float(np.sqrt(np.maximum(floor2,0).mean()))
                assert noise_info['affine_reference_support_floor']<=np.sqrt(np.square((array(ref)-array(cref))/sd).sum(1).mean())+1e-6
            rows=[];samples={}
            for method,free,meta in [('osscf',osgen,ospayload),('direct_repair',gen,payload)]:
                x=torch.cat([free,phi(free,a,b,c['L'],c['dv'])],1)
                row=dict(seed=seed,method=method,**c,**scores(x,cref,tr,6),**jacobian_summary(pop,c,tr),
                         parameters=meta['parameters'],optimizer_seconds=meta['optimizer_seconds'],fit_seconds=meta['fit_seconds'])
                row.update({'observed_'+k:v for k,v in scores(x,ref,tr,6).items()})
                row['empirical_w2']=empirical_w2(array(x)/sd,array(cref)/sd)
                row['observed_empirical_w2']=empirical_w2(array(x)/sd,array(ref)/sd)
                row.update(noise_info)
                rows.append(row);samples[method]=array(x)
            np.savez_compressed(dest/'samples.npz',**samples,reference=array(ref),clean_reference=array(cref),train_sd=sd)
            nc.dump(dest/'metrics.json',rows)
            print(f'P2 seed={seed} {key(c)} complete',flush=True)
    summarize(folder)

def summarize(folder):
    rows=[r for p in (folder/'runs').glob('seed*/*/metrics.json') for r in json.loads(p.read_text())]
    d=pd.DataFrame(rows);d.to_csv(folder/'metrics.csv',index=False)
    groups=['dv','L','sigma','n','steps'];records=[]
    for vals,g in d.groupby(groups):
        for metric in ['free_sw','full_sw','dep_sw','empirical_w2','observed_full_sw']:
            p=g.pivot(index='seed',columns='method',values=metric)
            diff=p.osscf-p.direct_repair
            records.append(dict(zip(groups,vals),metric=metric,**paired_summary(diff),os_mean=float(p.osscf.mean()),direct_mean=float(p.direct_repair.mean()),relative_gain=float(1-p.osscf.mean()/p.direct_repair.mean())))
    summary=pd.DataFrame(records);summary.to_csv(folder/'summary.csv',index=False)
    q=summary.query('L==1 and sigma==0 and n==5000 and steps==800 and metric=="free_sw"').set_index('dv')
    hi=q.loc[24];lo=q.loc[1]
    decision=dict(expand=bool(hi.relative_gain>=.05 and hi.wins>=8 and hi.ci_high<0 and hi.relative_gain-lo.relative_gain>=.05),
                  high_dv=hi.to_dict(),low_dv=lo.to_dict(),gain_increase=float(hi.relative_gain-lo.relative_gain),
                  interpretation='Predeclared resource gate, not a claim of general superiority.')
    nc.dump(folder/'screen_decision.json',decision)
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',rows=len(d),cells=len(d[groups].drop_duplicates()),expand=decision['expand']))
    print(json.dumps(decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expand',action='store_true');p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if args.check_only: print(checks());print(json.dumps(CONFIG,indent=2))
    else: run(OUT,args.expand)
