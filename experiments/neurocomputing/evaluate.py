from .common import *
from experiments.evaluate_downstream_privacy_minimal import DATASETS, DatasetSpec, evaluate_utility

def spec_for(b):
    if b.name in DATASETS:return DATASETS[b.name]
    # Predict total from trip attributes only; exclude all accounting operands.
    return DatasetSpec('taxi','regression',Path(),Path(),'total_amount',None,tuple(b.cols),tuple(b.cats),
                       tuple(['total_amount']+TAXI_FREE[2:]),())

def utility(b,f,method,s):
    rows=[]
    for model in ['rf','linear']:
        args=NS(utility_model=model,n_estimators=100,max_depth=12,min_samples_leaf=5,n_jobs=4,linear_max_iter=2000)
        row=evaluate_utility(f,b.ref_df,spec_for(b),'real' if method=='real_matched' else 'synthetic',method,s,args)
        row.update(split=b.split);rows.append(row)
    return rows

def persist_samples(b,x,k,folder,method,s,metadata=None,with_utility=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{method}.csv'; f=frame(b,x,k);f.to_csv(path,index=False)
    record={'dataset':b.name,'split':b.split,'seed':s,'method':method,'reference':b.reference,'sample_sha256':sha(path),**(metadata or {}),**metrics(b,x,k)}
    dump(folder/f'{method}.metrics.json',record)
    if with_utility:dump(folder/f'{method}.utility.json',utility(b,f,method,s))
    return record

def rounded_samples(b,x,k):
    u=x[:,b.free].clone()
    if b.name=='bikesharing':
        for c in ['casual','registered']:u[:,b.cols.index(c)]=u[:,b.cols.index(c)].round().clamp_min(0)
    elif b.name=='lendingclub':
        for c in ['open_acc','total_acc']:u[:,b.cols.index(c)]=u[:,b.cols.index(c)].round().clamp_min(0)
        j=b.cols.index('fico_range_low');u[:,j]=(u[:,j]/5).round()*5
    else:return x,k
    # Reconstruct AFTER lattice projection; avoid fractional quantile-clamping
    # after rounding by adjusting to the feasible integer lattice bounds.
    lo=b.spec.lower.clone();hi=b.spec.upper.clone()
    try:
        for c in (['casual','registered'] if b.name=='bikesharing' else ['open_acc','total_acc','fico_range_low']):
            j=b.cols.index(c);grid=5 if c=='fico_range_low' else 1
            b.spec.lower[j]=torch.ceil(lo[j]/grid)*grid;b.spec.upper[j]=torch.floor(hi[j]/grid)*grid
        return decode(b,u,fixk(b,k)),fixk(b,k)
    finally:b.spec.lower=lo;b.spec.upper=hi

def summarize(out):
    out=Path(out);records=[];utilities=[]
    for p in out.glob('split*/final/*/seed*/*.metrics.json'):records.append(json.loads(p.read_text()))
    for p in out.glob('split*/final/*/seed*/*.utility.json'):utilities+=json.loads(p.read_text())
    folder=out/'summary';folder.mkdir(exist_ok=True)
    if records:
        d=pd.DataFrame(records);d.to_csv(folder/'metrics_per_run.csv',index=False)
        metrics_cols=[c for c in ['sw','free_sw','dep_sw','violation_rate','pearson_mae','spearman_mae','category_tv','integer_violation','fico_grid_violation','negative_violation'] if c in d]
        means=d.groupby(['dataset','split','method'])[metrics_cols].mean()
        d.groupby(['dataset','split','method'])[metrics_cols].agg(['mean','std','count']).to_csv(folder/'within_split.csv')
        means.groupby(['dataset','method']).agg(['mean','std','count']).to_csv(folder/'between_split.csv')
        rows=[]
        for ds,group in d.groupby('dataset'):
            a=group[group.method=='osscf'].set_index(['split','seed'])
            for method in ['direct_repair','free_tabsyn','tabsyn_repair','tabddpm']:
                b=group[group.method==method].set_index(['split','seed']);ix=a.index.intersection(b.index)
                for split,values in (a.loc[ix,'sw']-b.loc[ix,'sw']).groupby(level='split'):
                    rows.append({'dataset':ds,'split':int(split),'comparison':'osscf - '+method,'mean_paired_sw_difference':float(values.mean()),'seed_sd':float(values.std()) if len(values)>1 else None,'n_seeds':len(values)})
        pd.DataFrame(rows).to_csv(folder/'paired_sw_by_split.csv',index=False)
        # Split means get equal weight. Do not pool seeds as independent data.
        main=d[d.method.isin(['osscf','direct','direct_repair','tabsyn','tabsyn_repair','free_tabsyn','tabddpm','train_bootstrap'])]
        bysplit=main.groupby(['dataset','split','method'])[['sw','violation_rate']].mean()
        table=bysplit.groupby(['dataset','method']).agg(['mean','std','count'])
        table.to_latex(folder/'fidelity_table.tex',float_format=lambda v:f'{v:.4f}',na_rep='--')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=list(main.dataset.unique());fig,axes=plt.subplots(1,len(names),figsize=(5*len(names),4),squeeze=False)
        for ax,ds in zip(axes[0],names):
            stats=bysplit.loc[ds]['sw'].groupby('method').agg(['mean','std']).sort_values('mean')
            ax.barh(stats.index,stats['mean'],xerr=stats['std'].fillna(0),color=['#5754c6' if m=='osscf' else '#89a6c9' for m in stats.index])
            ax.set_title(ds);ax.set_xlabel('Full SW (mean of split means; bars = split SD)');ax.invert_yaxis()
        fig.tight_layout();fig.savefig(folder/'fidelity.pdf');fig.savefig(folder/'fidelity.png',dpi=160);plt.close(fig)
    if utilities:
        u=pd.DataFrame(utilities);u.to_csv(folder/'utility_per_run.csv',index=False)
        vals=[c for c in ['roc_auc','average_precision','r2','rmse','mae'] if c in u]
        u.groupby(['dataset','split','method','utility_model'])[vals].agg(['mean','std','count']).to_csv(folder/'utility_within_split.csv')
    abl=[]
    for p in out.glob('split*/ablation/*/seed*/*.metrics.json'):abl.append(json.loads(p.read_text()))
    if abl:pd.DataFrame(abl).to_csv(folder/'ablation_per_run.csv',index=False)
    dump(folder/'status.json',{'formal_metric_rows':len(records),'utility_rows':len(utilities),'ablation_rows':len(abl),'complete':(out/'COMPLETE.json').exists()})
    return len(records),len(utilities),len(abl)
