"""Independent raw-label diagnostics: never collapse held-out categories to OOV.

This post-hoc evaluator does not train/select models or alter archived metrics.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse,json,hashlib
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

CATS={
 'lendingclub':['term_months','grade','home_ownership','verification_status','purpose','application_type','loan_status'],
 'bikesharing':['season','yr','mnth','hr','holiday','weekday','workingday','weathersit'],
 'taxi':['VendorID','RatecodeID','payment_type','passenger_count','pickup_hour','pickup_weekday','PULocationID','DOLocationID']}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def normalized(f,cols):
    out=pd.DataFrame(index=f.index)
    for c in cols:
        s=f[c]
        if pd.api.types.is_numeric_dtype(s):out[c]=s.map(lambda v:str(int(v)) if pd.notna(v) and float(v).is_integer() else str(v))
        else:out[c]=s.fillna('__MISSING__').astype(str).str.strip()
    return out

def tv(p,q):
    ix=p.index.union(q.index);return float(.5*np.abs(p.reindex(ix,fill_value=0)-q.reindex(ix,fill_value=0)).sum())

def main(out):
    out=Path(out);folder=out/'raw_category_audit';folder.mkdir(exist_ok=True)
    refs={};cache_path=folder/'cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {};rows=[];inputs={}
    for p in sorted(out.glob('split*/final/*/seed*/*.metrics.json')):
        row=json.loads(p.read_text());ds=row['dataset'];sp=row['split'];key=(ds,sp)
        sample=p.with_name(p.name.replace('.metrics.json','.csv'));source_hash=digest(sample)
        assert source_hash==row['sample_sha256'],str(sample)
        refpath=out/f'split{sp}'/'data'/ds/'test.csv';refhash=digest(refpath)
        tag=f'{source_hash}:{refhash}:{digest(__file__)}'
        if tag not in cache:
            cats=CATS[ds]
            if key not in refs:
                ref=normalized(pd.read_csv(refpath),cats)
                refs[key]=(ref,{c:ref[c].value_counts(normalize=True) for c in cats},
                           {cs:ref[list(cs)].value_counts(normalize=True) for cs in combinations(cats,2)},ref.value_counts(normalize=True))
            ref,marginals,pairs,joint=refs[key];f=normalized(pd.read_csv(sample),cats)
            cache[tag]={'raw_marginal_category_tv':float(np.mean([tv(f[c].value_counts(normalize=True),marginals[c]) for c in cats])),
                        'raw_pairwise_category_tv':float(np.mean([tv(f[list(cs)].value_counts(normalize=True),v) for cs,v in pairs.items()])),
                        'raw_joint_category_tv':tv(f.value_counts(normalize=True),joint)}
        rows.append({**{k:row[k] for k in ['dataset','split','seed','method']},**cache[tag]})
        inputs[str(sample.relative_to(out))]=source_hash;inputs[str(refpath.relative_to(out))]=refhash
    d=pd.DataFrame(rows);d.to_csv(folder/'per_run.csv',index=False)
    if len(d):
        cols=[c for c in d if c.startswith('raw_')]
        d.groupby(['dataset','split','method'])[cols].agg(['mean','std','count']).to_csv(folder/'within_split.csv')
        d.groupby(['dataset','split','method'])[cols].mean().groupby(['dataset','method']).agg(['mean','std','count']).to_csv(folder/'between_split.csv')
    cache_path.write_text(json.dumps(cache,indent=2),encoding='utf8')
    (folder/'manifest.json').write_text(json.dumps({'source_sha256':digest(__file__),'inputs':inputs,'rows':len(rows),
       'definition':'Raw nominal labels retained; no test-vocabulary expansion in models and no OOV collapsing in these diagnostics. Empirical joint TV is strongly sample-size dependent.',
       'complete_main_protocol':(out/'COMPLETE.json').exists()},indent=2),encoding='utf8')
    print(json.dumps({'category_rows':len(rows),'complete_main_protocol':(out/'COMPLETE.json').exists()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();main(a.out)
