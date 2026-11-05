from __future__ import annotations
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','4')
from .common import *
from .models import TabSynAdapter,checkpoint,load,ddpm_setup,ddpm_fit,ddpm_sample
from .evaluate import persist_samples,rounded_samples,summarize,utility
import argparse,traceback,gc

def select_flow(b,config):
    dest=b.out/f'split{b.split}'/'selection'/b.name/'flow';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text())
    rows=[]; cfg=config['flow'];base=cfg['base_steps'][b.name]
    for alpha in cfg['alphas']:
        for mult in cfg['step_multipliers']:
            steps=max(1,int(base*mult))
            for s in config['selection_seeds']:
                k=empirical_k(b,len(b.ref_x),s+500)
                for kind in ['hard','direct']:
                    p=dest/f'a{alpha}_steps{steps}_seed{s}_{kind}.json'
                    if p.exists():row=json.loads(p.read_text())
                    else:
                        model,norm,seconds,hist=train_flow(b,kind,alpha,steps,s)
                        x=sample_flow(b,model,norm,k,s+600,kind,cfg['sample_steps'])
                        if kind=='direct':x=decode(b,x[:,b.free],k)
                        row={'alpha':alpha,'steps':steps,'seed':s,'kind':kind,'seconds':seconds,'sw':metrics(b,x,k)['sw']};dump(p,row)
                    rows.append(row)
            event(b.out,stage='flow_selection',dataset=b.name,split=b.split,alpha=alpha,steps=steps)
    d=pd.DataFrame(rows);d.to_csv(dest/'candidates.csv',index=False);scores=d.groupby(['alpha','steps']).sw.mean();alpha,steps=scores.idxmin()
    result={'alpha':float(alpha),'steps':int(steps),'validation_sw':float(scores.min()),'criterion':'mean hard and direct+Phi validation SW across selection seeds'}
    dump(dest/'selected.json',result);return result

def select_ddpm(b,config):
    dest=b.out/f'split{b.split}'/'selection'/b.name/'tabddpm';dest.mkdir(parents=True,exist_ok=True)
    data=dest/'data';ddpm_setup(b,data)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text()),data
    rows=[]
    for steps in config['tabddpm']['candidate_steps']:
        for s in config['selection_seeds']:
            folder=dest/f'steps{steps}_seed{s}'
            if (folder/'validation.json').exists():row=json.loads((folder/'validation.json').read_text())
            else:
                args=ddpm_fit(b,folder,data,s,steps,config['tabddpm']);x,k=ddpm_sample(b,folder,args,len(b.ref_x))
                row={'steps':steps,'seed':s,**metrics(b,x,k)};dump(folder/'validation.json',row)
            rows.append(row)
    d=pd.DataFrame(rows);d.to_csv(dest/'candidates.csv',index=False);scores=d.groupby('steps').sw.mean()
    selected={'steps':int(scores.idxmin()),'validation_sw':float(scores.min())};dump(dest/'selected.json',selected);return selected,data

def run_flows(b,config,ablation=False):
    selected=select_flow(b,config) if not ablation else None
    test=bundle(b.name,b.split,b.out,'test');n=len(test.ref_x)
    for s in config['training_seeds']:
        k=empirical_k(b,n,s+500)
        configs=[(selected['alpha'],selected['steps'])] if not ablation else [(a,config['flow']['base_steps'][b.name]) for a in config['flow']['alphas']]
        for alpha,steps in configs:
            folder=b.out/f'split{b.split}'/('ablation' if ablation else 'final')/b.name/f'seed{s}';folder.mkdir(parents=True,exist_ok=True)
            suffix=f'_alpha{alpha}' if ablation else ''
            for kind,name in [('hard','osscf'),('direct','direct')]:
                marker=folder/f'{name}{suffix}.done.json'
                if marker.exists():continue
                model,norm,seconds,hist=train_flow(b,kind,alpha,steps,s)
                checkpoint(folder/f'{name}{suffix}.pt',{'model':model.state_dict(),'normalizer':norm,'alpha':alpha,'steps':steps,'seed':s})
                pd.DataFrame(hist).to_csv(folder/f'{name}{suffix}.loss.csv',index=False)
                x=sample_flow(b,model,norm,k,s+600,kind,config['flow']['sample_steps']);sync();t=time.perf_counter()
                sample_flow(b,model,norm,k,s+600,kind,config['flow']['sample_steps']);sync();sample_seconds=time.perf_counter()-t
                meta={'alpha':alpha,'steps':steps,'training_seconds':seconds,'sampling_seconds':sample_seconds}
                persist_samples(test,x,k,folder,name+suffix,s,meta,with_utility=not ablation)
                if kind=='direct':
                    x=decode(b,x[:,b.free],k);name='direct_repair'
                    persist_samples(test,x,k,folder,name+suffix,s,meta,with_utility=not ablation)
                if not ablation and b.name!='taxi':
                    xx,kk=rounded_samples(b,x,k);persist_samples(test,xx,kk,folder,name+'_rounded',s,meta)
                dump(marker,{'completed':True});event(b.out,stage='ablation' if ablation else 'formal_flow',dataset=b.name,split=b.split,seed=s,kind=kind,alpha=alpha)
            gc.collect();torch.cuda.empty_cache()

def run_tabsyn(b,config,pilot=False):
    test=None if pilot else bundle(b.name,b.split,b.out,'test')
    for s in config['selection_seeds'] if pilot else config['training_seeds']:
        for free in [False,True]:
            name='free_tabsyn' if free else 'tabsyn';folder=b.out/f'split{b.split}'/('pilot' if pilot else 'final')/b.name/f'seed{s}'
            folder.mkdir(parents=True,exist_ok=True);marker=folder/f'{name}.done.json'
            if marker.exists():continue
            seed(s);model=TabSynAdapter(b,free,config['tabsyn']);info=model.fit(folder/name,s)
            target=b if pilot else test
            sync();t=time.perf_counter();x,k=model.sample(len(target.ref_x),s+50000);sync();info['sampling_seconds']=time.perf_counter()-t
            persist_samples(target,x,k,folder,name,s,info,with_utility=not pilot)
            if not free:
                kk=fixk(b,k);xx=decode(b,x[:,b.free],kk);persist_samples(target,xx,kk,folder,'tabsyn_repair',s,info,with_utility=not pilot)
            if not pilot and b.name!='taxi':
                xx,kk=rounded_samples(b,x,k);persist_samples(target,xx,kk,folder,('free_tabsyn' if free else 'tabsyn_repair')+'_rounded',s,info)
            dump(marker,{'completed':True});event(b.out,stage='tabsyn_complete',dataset=b.name,split=b.split,pilot=pilot,**info)
            del model;gc.collect();torch.cuda.empty_cache()

def run_ddpm(b,config):
    choice,data=select_ddpm(b,config);test=bundle(b.name,b.split,b.out,'test')
    for s in config['training_seeds']:
        folder=b.out/f'split{b.split}'/'final'/b.name/f'seed{s}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'tabddpm.done.json').exists():continue
        args=ddpm_fit(b,folder/'tabddpm',data,s,choice['steps'],config['tabddpm'])
        sync();t=time.perf_counter();x,k=ddpm_sample(b,folder/'tabddpm',args,len(test.ref_x));sync()
        info=json.loads((folder/'tabddpm/trained.json').read_text());info['sampling_seconds']=time.perf_counter()-t
        persist_samples(test,x,k,folder,'tabddpm',s,info);dump(folder/'tabddpm.done.json',{'completed':True})
        event(b.out,stage='tabddpm_complete',dataset=b.name,split=b.split,seed=s)

def references(b,config):
    test=bundle(b.name,b.split,b.out,'test')
    for s in config['training_seeds']:
        folder=b.out/f'split{b.split}'/'final'/b.name/f'seed{s}'
        if (folder/'train_bootstrap.metrics.json').exists() and (folder/'real_matched.utility.json').exists():continue
        ix=np.random.default_rng(s+2000).integers(len(b.train_x),size=len(test.ref_x))
        persist_samples(test,b.train_x[ix],b.train_k[ix],folder,'train_bootstrap',s,with_utility=False)
        f=b.train_df.sample(n=len(test.ref_df),random_state=s,replace=False)
        dump(folder/'real_matched.utility.json',utility(test,f,'real_matched',s))
    p=b.out/f'split{b.split}'/'data'/b.name/'reference_audit.json'
    oov={c:float((~normalize(test.ref_df[c]).isin(b.codec.names[j])).mean()) for j,c in enumerate(b.cats)}
    dump(p,{'test_reference_structure':metrics(test,test.ref_x,test.ref_k),'test_oov_by_column':oov,'train_rows':len(b.train_df),'test_rows':len(test.ref_df)})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='outputs/neurocomputing_v1');parser.add_argument('--config',default=str(Path(__file__).with_name('protocol.json')))
    parser.add_argument('--stage',choices=['prepare','pilot','flow','ablation','tabsyn','tabddpm','references','all','summarize'],default='all')
    parser.add_argument('--datasets',nargs='+');parser.add_argument('--splits',nargs='+',type=int);parser.add_argument('--seeds',nargs='+',type=int);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.out).resolve()
    if args.smoke:
        config['sizes']={d:[256,64,64] for d in config['datasets']};config['split_seeds']=[3101];config['training_seeds']=[101];config['selection_seeds']=[91]
        config['flow']['base_steps']={d:4 for d in config['datasets']};config['flow']['alphas']=[.5];config['flow']['step_multipliers']=[1];config['flow']['sample_steps']=4
        config['tabsyn'].update(vae_max_epochs=2,diffusion_max_epochs=2,check_every=1,diffusion_width=128,sample_steps=4)
        config['tabddpm']['candidate_steps']=[4];config['tabddpm']['timesteps']=10
        config['smoke_only']=True
    if args.stage=='summarize':print(summarize(out));return
    provenance(out,config)
    datasets=args.datasets or config['datasets'];splits=args.splits or config['split_seeds']
    if args.seeds:config['training_seeds']=args.seeds
    try:
        for ds in datasets:
            for sp in splits:
                prepare_split(ds,sp,out,config)
                if args.stage=='prepare':continue
                b=bundle(ds,sp,out,'val')
                if args.stage in ['pilot']:run_tabsyn(b,config,pilot=True)
                if args.stage in ['all','flow']:run_flows(b,config)
                if args.stage in ['all','ablation'] and ds!='taxi':run_flows(b,config,ablation=True)
                if args.stage in ['all','tabsyn']:run_tabsyn(b,config)
                if args.stage in ['all','tabddpm']:run_ddpm(b,config)
                if args.stage in ['all','references']:references(b,config)
                summarize(out)
        event(out,stage='invocation_complete',requested_stage=args.stage,datasets=datasets,splits=splits)
    except BaseException as e:
        dump(out/f'failure_{int(time.time())}.json',{'stage':args.stage,'error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
