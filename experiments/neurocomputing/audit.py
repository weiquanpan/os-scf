"""Replay saved evidence; mark COMPLETE only when every declared task exists."""
from .common import *
from .evaluate import summarize
import argparse

def audit(out,require_complete=False):
    out=Path(out);config=json.loads((out/'protocol.json').read_text());issues=[];checks=[]
    provenance_data=json.loads((out/'provenance.json').read_text())
    for rel,digest in provenance_data['sources'].items():
        if sha(out/'source_snapshot'/rel)!=digest:issues.append('snapshot hash mismatch: '+rel)
    methods=['osscf','direct','direct_repair','tabsyn','tabsyn_repair','free_tabsyn','tabddpm','train_bootstrap']
    expected=0;found=0;max_error=0.;caps=[]
    for ds in config['datasets']:
        for split in config['split_seeds']:
            data=out/f'split{split}'/'data'/ds
            if not (data/'manifest.json').exists():issues.append(f'missing split {ds}/{split}');continue
            mf=json.loads((data/'manifest.json').read_text())
            for name,digest in mf['files'].items():
                if sha(data/name)!=digest:issues.append(f'data changed: {data/name}')
            frames=[pd.read_csv(data/f'{s}.csv') for s in ['train','val','test']]
            ids=[set(f.source_row) for f in frames];groups=[set(f._row_group) for f in frames]
            for a,c in [(0,1),(0,2),(1,2)]:
                if ids[a]&ids[c] or groups[a]&groups[c]:issues.append(f'partition overlap {ds}/{split}')
            b=bundle(ds,split,out,'test')
            mlist=methods+(['osscf_rounded','direct_repair_rounded','tabsyn_repair_rounded','free_tabsyn_rounded'] if ds!='taxi' else [])
            for seed_value in config['training_seeds']:
                folder=out/f'split{split}'/'final'/ds/f'seed{seed_value}'
                for method in mlist:
                    expected+=1;p=folder/f'{method}.metrics.json'
                    if not p.exists():issues.append(f'missing metric {ds}/{split}/{seed_value}/{method}');continue
                    found+=1;old=json.loads(p.read_text());path=folder/f'{method}.csv'
                    if sha(path)!=old['sample_sha256']:issues.append(f'sample changed: {path}')
                    f=pd.read_csv(path);x=torch.tensor(f[b.cols].to_numpy(np.float32),device=DEVICE);k=torch.tensor(b.codec.encode(f),device=DEVICE)
                    actual=metrics(b,x,k)
                    err=max(abs(old[key]-val) for key,val in actual.items());max_error=max(max_error,err)
                    if err>1e-5:issues.append(f'metric replay mismatch {path}: {err}')
                    if method in ['osscf','direct_repair','free_tabsyn','tabsyn_repair'] and actual['violation_rate']!=0:issues.append(f'hard equation failure {path}')
                    if method.endswith('_rounded') and actual.get('integer_violation',0)!=0:issues.append(f'integer failure {path}')
                    checks.append({'dataset':ds,'split':split,'seed':seed_value,'method':method,'max_error':err})
                for method in [m for m in mlist if m!='train_bootstrap']+['real_matched']:
                    p=folder/f'{method}.utility.json'
                    if not p.exists():issues.append(f'missing utility {ds}/{split}/{seed_value}/{method}');continue
                    rows=json.loads(p.read_text())
                    if {r['utility_model'] for r in rows}!={'rf','linear'} or any(r['status']!='ok' for r in rows):issues.append(f'invalid utility {p}')
                for m in ['tabsyn','free_tabsyn']:
                    p=folder/m/'trained.json'
                    if p.exists():caps.append({'dataset':ds,'split':split,'method':m,**json.loads(p.read_text())})
                if ds!='taxi':
                    for alpha in config['flow']['alphas']:
                        for m in ['osscf','direct','direct_repair']:
                            p=out/f'split{split}'/'ablation'/ds/f'seed{seed_value}'/f'{m}_alpha{alpha}.metrics.json'
                            if not p.exists():issues.append(f'missing ablation {p}')
    folder=out/'audit';folder.mkdir(exist_ok=True)
    pd.DataFrame(checks).to_csv(folder/'metric_replay.csv',index=False);pd.DataFrame(caps).to_csv(folder/'tabsyn_budget_flags.csv',index=False)
    result={'status':'PASS' if not issues else 'INCOMPLETE_OR_FAILED','expected_metric_rows':expected,'found_metric_rows':found,'max_replay_error':max_error,'issues':issues,
            'tabsyn_cap_limited_runs':sum(r['vae_hit_cap'] or r['diffusion_hit_cap'] for r in caps),'interpretation':'budget caps are not evidence of convergence; no acceptance or superiority claim'}
    dump(folder/'report.json',result)
    if not issues:dump(out/'COMPLETE.json',result)
    summarize(out)
    if require_complete and issues:raise RuntimeError(f'{len(issues)} missing or failed checks; see {folder}/report.json')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.out,a.require_complete),indent=2))
