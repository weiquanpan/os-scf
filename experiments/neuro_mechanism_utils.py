"""Auditable utilities for the separate 2026-11 mechanism follow-up."""
from __future__ import annotations
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib, itertools, json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from experiments.neurocomputing import common as nc
from constrained_tabular_fm.continuous_original_space import make_free_flow_normalizer
from constrained_tabular_fm.mechanism_ablations import ConditionalVelocity

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT/'outputs/neurocomputing_v2'
OUT = ROOT/'outputs/neuro_mechanisms_20261103_v1'
PROJ_SEEDS = (17001, 17002, 17003)

def freeze(folder, config, files, inputs=()):
    parent = json.loads((MAIN/'provenance.json').read_text())
    for p,h in parent['sources'].items():
        assert nc.sha(ROOT/p) == h, f'Frozen main source changed: {p}'
    sources = {str(Path(p).resolve().relative_to(ROOT)): nc.sha(p) for p in files}
    identity = dict(config=config, sources=sources, inputs={str(Path(p).relative_to(ROOT)):nc.sha(p) for p in inputs}, parent_identity=parent['identity'])
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'provenance.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity, 'Frozen follow-up configuration/source/input changed'
    else:
        nc.dump(path, identity)
        nc.dump(folder/'protocol.json', config)
        for p in files:
            dest=folder/'source_snapshot'/Path(p).resolve().relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    return identity

def array(x):
    return x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x,dtype=np.float64)

def sw(a,b, projections=1024, seeds=PROJ_SEEDS):
    """Equal-size empirical sliced W2. Three MC estimates, not three replicates."""
    a,b=array(a),array(b)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    vals=[]
    for seed in seeds:
        v=np.random.default_rng(seed).normal(size=(a.shape[1],projections))
        v/=np.linalg.norm(v,axis=0,keepdims=True)
        vals.append(float(np.sqrt(np.square(np.sort(a@v,axis=0)-np.sort(b@v,axis=0)).mean())))
    return float(np.mean(vals)), vals

def scores(x,ref,train,du,projections=1024,seeds=PROJ_SEEDS):
    x,ref,train=map(array,(x,ref,train))
    sd=train.std(0).clip(1e-6)
    result={}
    for name,sl in [('full',slice(None)),('free',slice(0,du)),('dep',slice(du,None))]:
        mean,vals=sw(x[:,sl]/sd[sl],ref[:,sl]/sd[sl],projections,seeds)
        result[name+'_sw']=mean
        result[name+'_projection_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
        for i,v in enumerate(vals): result[f'{name}_sw_proj{i}']=v
    return result

def empirical_w2(a,b,n=256,repeats=3):
    """Exact equal-mass empirical OT on declared subsamples; sqrt of squared cost."""
    a,b=array(a),array(b); vals=[]
    for j in range(repeats):
        rng=np.random.default_rng(8100+j)
        aa=a[rng.choice(len(a),n,replace=False)]; bb=b[rng.choice(len(b),n,replace=False)]
        cost=cdist(aa,bb,'sqeuclidean'); i,k=linear_sum_assignment(cost)
        vals.append(float(np.sqrt(cost[i,k].mean())))
    return float(np.mean(vals))

def norm_for(x,alpha,eps=1e-4):
    norm=make_free_flow_normalizer(x,whiten_shrinkage=alpha)
    if eps != 1e-4:
        z=(x-norm.mean)/norm.std
        C=torch.cov(z.T); eye=torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
        lam,V=torch.linalg.eigh((1-alpha)*C+alpha*eye+eps*eye)
        assert float(lam.min())>0
        norm.W=V@torch.diag(lam.rsqrt())@V.T
        norm.W_inv=V@torch.diag(lam.sqrt())@V.T
    return norm

def train_paired(x,k,du,seed,alpha,steps,eps=1e-4,batch=512,save_at=(),callback=None):
    """Same index/time/free Gaussian streams across unequal model dimensions."""
    nc.seed(seed); nc.sync(); start=time.perf_counter()
    norm=norm_for(x,alpha,eps); z=norm.transform(x)
    model=ConditionalVelocity(x.shape[1],k.shape[1],128).to(nc.DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    streams=[torch.Generator(device=nc.DEVICE).manual_seed(seed+v) for v in (1000,2000,3000,4000)]
    history=[]; checkpoints={}; probes={}
    nc.sync(); train_start=time.perf_counter()
    for step in range(1,steps+1):
        ix=torch.randint(len(z),(batch,),generator=streams[0],device=nc.DEVICE)
        t=torch.rand((batch,1),generator=streams[1],device=nc.DEVICE)
        free=torch.randn((batch,du),generator=streams[2],device=nc.DEVICE)
        dep=torch.randn((batch,x.shape[1]-du),generator=streams[3],device=nc.DEVICE)
        noise=torch.cat([free,dep],1); target=z[ix]
        loss=(model((1-t)*noise+t*target,k[ix],t)-(target-noise)).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step in (1,steps):
            probes[str(step)]=hashlib.sha256(ix.cpu().numpy().tobytes()+t.cpu().numpy().tobytes()+free.cpu().numpy().tobytes()).hexdigest()
        if step%100==0 or step==1: history.append(dict(step=step,loss=float(loss.detach())))
        if step in set(save_at)|{steps}:
            nc.sync(); elapsed=time.perf_counter()-train_start
            payload=dict(model={a:b.detach().clone() for a,b in model.state_dict().items()},normalizer=norm,
                         seed=seed,alpha=alpha,eps=eps,steps=step,optimizer_seconds=elapsed,
                         fit_seconds=elapsed+train_start-start,parameters=sum(p.numel() for p in model.parameters()),
                         history=history.copy(),rng_probes=probes.copy())
            checkpoints[step]=payload
            if callback: callback(step,model,norm,payload)
    return model,norm,checkpoints

def restore(payload,kdim):
    model=ConditionalVelocity(payload['model']['net.4.bias'].numel(),kdim,128).to(nc.DEVICE)
    model.load_state_dict(payload['model']); model.eval()
    return model,payload['normalizer']

@torch.no_grad()
def sample_paired(model,norm,k,du,seed,steps=80):
    n=len(k)
    free=torch.randn((n,du),device=nc.DEVICE,generator=torch.Generator(device=nc.DEVICE).manual_seed(seed+600))
    dep=torch.randn((n,model.dim-du),device=nc.DEVICE,generator=torch.Generator(device=nc.DEVICE).manual_seed(seed+601))
    z=torch.cat([free,dep],1)
    for j in range(steps):
        t=torch.full((n,1),j/steps,device=nc.DEVICE)
        z=(z+model(z,k,t)/steps).clamp(-6,6)
    return norm.inverse(z)

def paired_summary(values):
    v=np.asarray(values,dtype=float)
    rng=np.random.default_rng(911)
    means=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    signs=np.array(list(itertools.product([-1,1],repeat=len(v)))) if len(v)<=15 else rng.choice([-1,1],size=(32768,len(v)))
    return dict(mean=float(v.mean()),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),
                wins=int((v<0).sum()),n=len(v),signflip_p=float((np.abs((signs*v).mean(1))>=abs(v.mean())-1e-15).mean()))

def checks():
    a=np.array([[0.,0.],[1.,2.],[3.,4.]])
    assert sw(a,a)[0]==0
    assert abs(empirical_w2(a,a+1,n=3,repeats=1)-np.sqrt(2))<1e-12
    x=torch.randn(256,4,device=nc.DEVICE)
    for eps in (.1,.00001):
        norm=norm_for(x,0,eps)
        assert torch.max(torch.abs(norm.inverse(norm.transform(x))-x))<2e-5
    # POT independent solver verifies the equal-weight assignment implementation.
    import ot
    b=a+np.array([.25,.75]); cost=cdist(a,b,'sqeuclidean')
    assert abs(empirical_w2(a,b,3,1)**2-ot.emd2(np.ones(3)/3,np.ones(3)/3,cost))<1e-10
    return dict(status='PASS',identity_sw=0,exact_w2_verified_against_POT=True,normalizer_roundtrip=True)
