from __future__ import annotations

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import sys, json, hashlib, random, time, platform, shutil
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from constrained_tabular_fm import lendingclub_methods as lc, bikesharing_methods as bike
from constrained_tabular_fm.continuous_original_space import make_free_flow_normalizer
from constrained_tabular_fm.mechanism_ablations import ConditionalVelocity
from constrained_tabular_fm.metrics import sliced_wasserstein
torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()

def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding='utf8')
    os.replace(tmp,path)

def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)

def sync():
    if DEVICE.type=='cuda': torch.cuda.synchronize()

@contextmanager
def isolated_rng(value):
    py=random.getstate(); npstate=np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if DEVICE.type=='cuda' else []):
        seed(value)
        try: yield
        finally: random.setstate(py); np.random.set_state(npstate)

def event(out, **row):
    row={'time':time.strftime('%Y-%m-%dT%H:%M:%S'),**row}
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with (out/'events.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(row,default=str)+'\n')
    print(json.dumps(row,default=str),flush=True)

def normalize(s):
    if pd.api.types.is_numeric_dtype(s): return s.map(lambda x:str(int(x)) if pd.notna(x) and float(x).is_integer() else str(x))
    return s.fillna('__MISSING__').astype(str).str.strip()

class Codec:
    """Training-only vocabulary, with explicit public-schema domains and OOV token."""
    def __init__(self, frame, cats, domains=None):
        self.cats=list(cats); self.names=[]; domains=domains or {}
        for c in cats:
            observed=set(normalize(frame[c]))
            domain=list(map(str,domains.get(c,[])))
            names=domain+sorted(observed-set(domain))
            if c not in domains: names+=['__UNKNOWN__'] if '__UNKNOWN__' not in names else []
            self.names.append(tuple(names))
        self.names=tuple(self.names); self.sizes=tuple(map(len,self.names))
        self.maps=[{v:i for i,v in enumerate(ns)} for ns in self.names]
    def encode(self, frame):
        result=[]
        for c,m in zip(self.cats,self.maps):
            a=normalize(frame[c]).map(m)
            if a.isna().any() and '__UNKNOWN__' not in m: raise ValueError(f'Out-of-schema category: {c}')
            result.append(a.fillna(m.get('__UNKNOWN__',-1)).to_numpy(np.int64))
        return np.stack(result,1)
    def decode(self, k):
        a=k.detach().cpu().numpy() if torch.is_tensor(k) else k
        return {c:[ns[int(i)] for i in a[:,j]] for j,(c,ns) in enumerate(zip(self.cats,self.names))}

TAXI_FREE=['trip_distance','duration_minutes','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge','congestion_surcharge','airport_fee']
TAXI_COLS=TAXI_FREE+['total_amount']
TAXI_CATS=['VendorID','RatecodeID','payment_type','passenger_count','pickup_hour','pickup_weekday','PULocationID','DOLocationID']

class TaxiSpec:
    def __init__(self,x,codec):
        self.category_names=codec.names; self.category_sizes=codec.sizes; self.free_dim=len(TAXI_FREE)
        self.lower=torch.quantile(x,.001,dim=0).clamp_min(0); self.upper=torch.quantile(x,.999,dim=0)
        self.thresholds=(.02,)
    def one_hot(self,k): return torch.cat([F.one_hot(k[:,j],n).float() for j,n in enumerate(self.category_sizes)],1)
    def clamp_continuous(self,x): return torch.minimum(torch.maximum(x,self.lower),self.upper)

def prepare_pool(ds,out):
    folder=Path(out)/'pools'/ds; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        m=json.loads((folder/'manifest.json').read_text()); assert sha(folder/'pool.csv')==m['pool_sha256']; return
    if ds in ['lendingclub','bikesharing']:
        source=ROOT/('data/lendingclub_preprocessed/train.csv' if ds=='lendingclub' else 'data/bike_sharing_preprocessed/train.csv')
        f=pd.read_csv(source); f.insert(0,'source_row',np.arange(len(f)))
        meta={'source':str(source),'source_sha256':sha(source),'historically_used_pool':True,'raw_rows':len(f)}
    else:
        source=ROOT/'data/nyc_taxi_neuro/yellow_tripdata_2024-01.parquet'
        raw=pd.read_parquet(source); raw['source_row']=np.arange(len(raw)); raw=raw.rename(columns={'Airport_fee':'airport_fee'})
        duration=(raw.tpep_dropoff_datetime-raw.tpep_pickup_datetime).dt.total_seconds()/60
        raw['duration_minutes']=duration; raw['pickup_hour']=raw.tpep_pickup_datetime.dt.hour; raw['pickup_weekday']=raw.tpep_pickup_datetime.dt.dayofweek
        required=TAXI_COLS+TAXI_CATS
        ok=raw[required].notna().all(axis=1)
        ok &= (raw.tpep_pickup_datetime.dt.year==2024)&(raw.tpep_pickup_datetime.dt.month==1)
        ok &= duration.between(1,180)&raw.trip_distance.between(.1,100)&raw.payment_type.isin([1,2])
        ok &= raw.passenger_count.between(1,6)&(raw[TAXI_COLS].ge(0).all(axis=1))&raw.total_amount.between(.01,1000)
        # A priori operational domain; do NOT filter by equation residual.
        f=raw.loc[ok,['source_row']+required].copy()
        # Vendor 1's extra aggregates congestion/airport fees in this source.
        # This empirically audited accounting convention is NOT claimed as a
        # universally valid taxi law. Unmatched rows remain in the benchmark.
        correction=np.where(f.VendorID==1,f.congestion_surcharge+f.airport_fee,0.)
        residual=f.total_amount-(f[TAXI_FREE[2:]].sum(axis=1)-correction)
        meta={'source':str(source),'source_sha256':sha(source),'raw_rows':len(raw),'quality_rows':len(f),
              'formula':'sum(fee fields) - 1[VendorID=1]*(congestion_surcharge+airport_fee); empirical January2024 accounting convention; residuals retained',
              'formula_filter_applied':False,'tolerance':.02,'reference_violation_rate':float((residual.abs()>.02).mean()),
              'residual_abs_quantiles':{str(q):float(residual.abs().quantile(q)) for q in [.5,.9,.99,1.]},
              'official_source':'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'}
        # Development sample isolated before any formal splits; schema audit only.
        ix=np.random.default_rng(7100).permutation(len(f)); dev=f.iloc[ix[:5000]]
        dev.to_csv(folder/'development.csv',index=False); f=f.iloc[ix[5000:105000]].copy()
        meta['development_source_rows']=dev.source_row.astype(int).tolist()
    columns=[c for c in f if c!='source_row']
    f['_row_group']=pd.util.hash_pandas_object(f[columns],index=False).map(lambda v:format(v,'016x'))
    meta.update(exact_duplicate_rows=int(f.duplicated(columns).sum()),duplicate_policy='retain rows; identical records stay in one partition',pool_rows=len(f))
    f.to_csv(folder/'pool.csv',index=False); meta['pool_sha256']=sha(folder/'pool.csv'); dump(folder/'manifest.json',meta)

def prepare_split(ds,split,out,config):
    prepare_pool(ds,out); folder=Path(out)/f'split{split}'/'data'/ds
    if (folder/'manifest.json').exists():
        m=json.loads((folder/'manifest.json').read_text()); assert all(sha(folder/name)==h for name,h in m['files'].items()); return
    f=pd.read_csv(Path(out)/'pools'/ds/'pool.csv'); rng=np.random.default_rng(split)
    sizes=config['sizes'][ds]; assert len(f)>=sum(sizes),(ds,len(f),sizes)
    groups=list(f.groupby('_row_group',sort=False).indices.values())
    rng.shuffle(groups);groups.sort(key=len,reverse=True)
    remaining=np.array(sizes+[len(f)-sum(sizes)]);allocated=[[],[],[],[]]
    for indices in groups:
        candidates=np.flatnonzero(remaining>=len(indices))
        if not len(candidates):raise ValueError('Cannot assign duplicate groups to exact partition sizes')
        weights=remaining[candidates]/remaining[candidates].sum();part=int(rng.choice(candidates,p=weights))
        allocated[part].extend(indices);remaining[part]-=len(indices)
    assert not remaining.any()
    folder.mkdir(parents=True,exist_ok=True)
    for label,indices in zip(['train','val','test'],allocated):
        rng.shuffle(indices);f.iloc[indices].to_csv(folder/f'{label}.csv',index=False)
    dump(folder/'manifest.json',{'dataset':ds,'split':split,'sizes':sizes,'pool_sha256':sha(Path(out)/'pools'/ds/'pool.csv'),
         'files':{p.name:sha(p) for p in folder.glob('*.csv')}})

def bundle(ds,split,out,reference='val'):
    folder=Path(out)/f'split{split}'/'data'/ds
    tr=pd.read_csv(folder/'train.csv'); ref=pd.read_csv(folder/f'{reference}.csv')
    mod=lc if ds=='lendingclub' else bike if ds=='bikesharing' else None
    cols=list(mod.CONTINUOUS_COLUMNS) if mod else TAXI_COLS; cats=list(mod.CATEGORICAL_COLUMNS) if mod else TAXI_CATS
    # Fixed schema domains are public metadata, not inferred from held-out rows.
    domains=bike.PREFERRED_CATEGORY_ORDER if ds=='bikesharing' else {'term_months':['36','60']} if ds=='lendingclub' else {'payment_type':['1','2'],'pickup_hour':list(map(str,range(24))),'pickup_weekday':list(map(str,range(7)))}
    codec=Codec(tr,cats,domains)
    for c in cats: tr[c]=normalize(tr[c]); ref[c]=normalize(ref[c])
    x=torch.tensor(tr[cols].to_numpy(np.float32),device=DEVICE); rx=torch.tensor(ref[cols].to_numpy(np.float32),device=DEVICE)
    k=torch.tensor(codec.encode(tr),device=DEVICE); rk=torch.tensor(codec.encode(ref),device=DEVICE)
    cfg=(lc.LendingClubAllMethodsConfig() if ds=='lendingclub' else bike.BikeSharingAllMethodsConfig()) if mod else NS()
    cfg.batch_size=512 if ds!='bikesharing' else 256
    if mod: spec=mod.build_spec(x,codec,cfg)
    else: spec=TaxiSpec(x,codec)
    b=NS(name=ds,split=split,out=Path(out),reference=reference,train_df=tr,ref_df=ref,cols=cols,cats=cats,codec=codec,
         train_x=x,train_k=k,ref_x=rx,ref_k=rk,spec=spec,cfg=cfg,mod=mod,device=DEVICE,free=list(range(spec.free_dim)),dep=list(range(spec.free_dim,len(cols))))
    b.generated_cats=[j for j,c in enumerate(cats) if not(ds=='bikesharing' and c=='workingday')]
    return b

def fixk(b,k):
    if b.name=='bikesharing':
        k=k.clone(); target=b.spec.workingday_target(k); names=b.codec.names[6]
        k[:,6]=torch.where(target==1,names.index('1'),names.index('0'))
    return k

def decode(b,u,k):
    u=torch.minimum(torch.maximum(u,b.spec.lower[b.free].to(u)),b.spec.upper[b.free].to(u))
    if b.name=='lendingclub': return b.spec.decode_free(u,k)
    if b.name=='bikesharing': return b.spec.decode_free(u)
    vendor1=b.codec.names[0].index('1') if '1' in b.codec.names[0] else -1
    correction=(k[:,0]==vendor1).float()*(u[:,-2]+u[:,-1])
    return torch.cat([u,(u[:,2:].sum(1)-correction)[:,None]],1)

def empirical_k(b,n,s):
    with isolated_rng(s): return fixk(b,b.train_k[torch.randint(len(b.train_k),(n,),device=DEVICE)].clone())

def frame(b,x,k):
    f=pd.DataFrame(x.detach().cpu().numpy(),columns=b.cols)
    for c,a in b.codec.decode(k).items():f[c]=a
    return f

def metrics(b,x,k):
    a=x.detach().cpu().numpy().astype(float); r=b.ref_x.cpu().numpy().astype(float); tr=b.train_x.cpu().numpy().astype(float)
    assert a.shape==r.shape and np.isfinite(a).all()
    mu=tr.mean(0); sd=tr.std(0).clip(1e-6); a0=(a-mu)/sd; r0=(r-mu)/sd
    result={'sw':sliced_wasserstein(a0,r0,seed=17),'free_sw':sliced_wasserstein(a0[:,b.free],r0[:,b.free],seed=19),'dep_sw':sliced_wasserstein(a0[:,b.dep],r0[:,b.dep],seed=23)}
    if b.name=='lendingclub': res=b.spec.residuals(x,k).cpu().numpy(); tau=np.array(b.spec.thresholds)
    elif b.name=='bikesharing':res=b.spec.residuals(x).cpu().numpy();tau=np.array([.5])
    else:
        vendor1=b.codec.names[0].index('1') if '1' in b.codec.names[0] else -1
        total=x[:,2:-1].sum(1)-(k[:,0]==vendor1).float()*(x[:,-3]+x[:,-2])
        res=(x[:,-1]-total).cpu().numpy()[:,None];tau=np.array([.02])
    result.update(violation_rate=float((np.abs(res)>tau).any(1).mean()),constraint_mae=float(np.abs(res).mean()))
    if b.name=='bikesharing': result['logic_violation']=float(b.spec.workingday_logic_violation(k).float().mean())
    ac=pd.DataFrame(a).corr().fillna(0).to_numpy(); rc=pd.DataFrame(r).corr().fillna(0).to_numpy(); mask=np.triu_indices(len(b.cols),1)
    result['pearson_mae']=float(np.abs(ac-rc)[mask].mean())
    ac=pd.DataFrame(a).corr(method='spearman').fillna(0).to_numpy();rc=pd.DataFrame(r).corr(method='spearman').fillna(0).to_numpy()
    result['spearman_mae']=float(np.abs(ac-rc)[mask].mean())
    tv=[]; ka=k.cpu().numpy();kr=b.ref_k.cpu().numpy()
    for j,n in enumerate(b.codec.sizes):tv.append(.5*np.abs(np.bincount(ka[:,j],minlength=n)/len(ka)-np.bincount(kr[:,j],minlength=n)/len(kr)).sum())
    result['category_tv']=float(np.mean(tv))
    unknown_generated=np.zeros(len(ka),dtype=bool);unknown_reference=np.zeros(len(kr),dtype=bool)
    for j,names in enumerate(b.codec.names):
        if '__UNKNOWN__' in names:
            index=names.index('__UNKNOWN__');unknown_generated|=(ka[:,j]==index);unknown_reference|=(kr[:,j]==index)
    result['generated_unknown_rate']=float(unknown_generated.mean());result['reference_unknown_rate']=float(unknown_reference.mean())
    ints=['casual','registered','cnt'] if b.name=='bikesharing' else ['open_acc','total_acc'] if b.name=='lendingclub' else []
    if ints:
        vals=a[:,[b.cols.index(c) for c in ints]];result['integer_violation']=float((np.abs(vals-np.round(vals))>1e-5).any(1).mean())
    if b.name=='lendingclub':
        v=a[:,b.cols.index('fico_range_low')]; result['fico_grid_violation']=float((np.abs(v/5-np.round(v/5))>1e-5).mean())
    result['negative_violation']=float((a<0).any(1).mean())
    return result

def train_flow(b,kind,alpha,steps,s,width=128,lr=.001):
    seed(s); target=b.train_x[:,b.free] if kind=='hard' else b.train_x
    norm=make_free_flow_normalizer(target,whiten_shrinkage=alpha); z=norm.transform(target); kh=b.spec.one_hot(b.train_k)
    model=ConditionalVelocity(z.shape[1],kh.shape[1],width).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=lr)
    sync(); t0=time.perf_counter(); history=[]
    for j in range(steps):
        idx=torch.randint(len(z),(b.cfg.batch_size,),device=DEVICE); z1=z[idx];z0=torch.randn_like(z1);t=torch.rand(len(idx),1,device=DEVICE)
        loss=(model((1-t)*z0+t*z1,kh[idx],t)-(z1-z0)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite flow loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if j%100==0 or j==steps-1:history.append({'step':j+1,'loss':float(loss.detach())})
    sync(); return model,norm,time.perf_counter()-t0,history

@torch.no_grad()
def sample_flow(b,model,norm,k,s,kind,steps=80):
    model.eval()
    with isolated_rng(s):
        z=torch.randn(len(k),model.dim,device=DEVICE); kh=b.spec.one_hot(k)
        for i in range(steps):
            t=torch.full((len(k),1),i/steps,device=DEVICE); z=(z+model(z,kh,t)/steps).clamp(-6,6)
        x=norm.inverse(z)
        return decode(b,x,k) if kind=='hard' else b.spec.clamp_continuous(x)

def provenance(out,config):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'experiments/neurocomputing').glob('*.py'))+list((ROOT/'constrained_tabular_fm').glob('*.py'))
    sources+=[ROOT/'experiments/evaluate_downstream_privacy_minimal.py']
    sources+=list((ROOT/'external/tabsyn/tabsyn').rglob('*.py'))+list((ROOT/'external/tab-ddpm').rglob('*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    identity=hashlib.sha256(json.dumps({'config':config,'sources':hashes},sort_keys=True).encode()).hexdigest()
    path=out/'provenance.json'
    if path.exists():
        old=json.loads(path.read_text());
        if old['identity']!=identity:raise RuntimeError('Source/protocol changed: choose a new output root; cached runs must not be silently reused.')
        return identity
    for p in sources:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    dump(out/'protocol.json',config)
    dump(path,{'identity':identity,'sources':hashes,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,
               'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,'os':platform.platform(),'timing':'synchronized; desktop GPU activity not controlled'})
    return identity
