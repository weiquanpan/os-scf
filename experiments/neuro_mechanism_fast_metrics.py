"""Numerically equivalent float64 CUDA SW; fixed NumPy projection matrices."""
from experiments.neuro_mechanism_utils import *
from experiments import neuro_mechanism_utils as original

def sw(a,b,projections=1024,seeds=PROJ_SEEDS):
    aa,bb=array(a),array(b)
    assert aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
    ta=torch.as_tensor(aa,device=nc.DEVICE,dtype=torch.float64)
    tb=torch.as_tensor(bb,device=nc.DEVICE,dtype=torch.float64)
    vals=[]
    for seed in seeds:
        v=np.random.default_rng(seed).normal(size=(aa.shape[1],projections))
        v/=np.linalg.norm(v,axis=0,keepdims=True)
        tv=torch.as_tensor(v,device=nc.DEVICE,dtype=torch.float64)
        ap=torch.sort(ta@tv,dim=0).values;bp=torch.sort(tb@tv,dim=0).values
        vals.append(float((ap-bp).square().mean().sqrt()))
    return float(np.mean(vals)),vals

def scores(x,ref,train,du,projections=1024,seeds=PROJ_SEEDS):
    x,ref,train=map(array,(x,ref,train));sd=train.std(0).clip(1e-6);result={}
    for name,sl in [('full',slice(None)),('free',slice(0,du)),('dep',slice(du,None))]:
        mean,vals=sw(x[:,sl]/sd[sl],ref[:,sl]/sd[sl],projections,seeds)
        result[name+'_sw']=mean
        result[name+'_projection_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.
        for i,v in enumerate(vals):result[f'{name}_sw_proj{i}']=v
    return result

def verify():
    rng=np.random.default_rng(92001);errors=[];timings=[]
    for n,d in [(2048,1),(2048,6),(2048,30),(5000,11)]:
        a=rng.standard_t(4,size=(n,d));b=rng.normal(size=(n,d))
        start=time.perf_counter();ref=original.sw(a,b);cpu=time.perf_counter()-start
        nc.sync();start=time.perf_counter();got=sw(a,b);nc.sync();gpu=time.perf_counter()-start
        errors.append(float(np.max(np.abs(np.array(ref[1])-np.array(got[1])))))
        timings.append(dict(n=n,d=d,cpu_seconds=cpu,gpu_seconds=gpu))
    assert max(errors)<1e-12,errors
    return dict(status='VERIFIED',max_absolute_cpu_gpu_error=max(errors),checks=4,timings=timings,
                identical_projection_matrices=True,identical_float64_metric_definition=True)

if __name__=='__main__':print(json.dumps(verify(),indent=2))
