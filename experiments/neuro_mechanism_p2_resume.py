"""P2 summary repair: distinguish training n from the number of seed replicates."""
from experiments.neuro_mechanism_utils import *
from experiments import neuro_mechanism_p2 as p2
from experiments import neuro_mechanism_fast_metrics as fast
import argparse

def summarize(folder):
    rows=[r for p in (folder/'runs').glob('seed*/*/metrics.json') for r in json.loads(p.read_text())]
    d=pd.DataFrame(rows);d.to_csv(folder/'metrics.csv',index=False)
    groups=['dv','L','sigma','n','steps'];records=[]
    for vals,g in d.groupby(groups):
        for metric in ['free_sw','full_sw','dep_sw','empirical_w2','observed_full_sw']:
            pivot=g.pivot(index='seed',columns='method',values=metric)
            stat=paired_summary(pivot.osscf-pivot.direct_repair);stat['replicates']=stat.pop('n')
            records.append(dict(zip(groups,vals),metric=metric,**stat,os_mean=float(pivot.osscf.mean()),
                                direct_mean=float(pivot.direct_repair.mean()),relative_gain=float(1-pivot.osscf.mean()/pivot.direct_repair.mean())))
    summary=pd.DataFrame(records);summary.to_csv(folder/'summary_corrected.csv',index=False)
    q=summary.query('L==1 and sigma==0 and n==5000 and steps==800 and metric=="free_sw"').set_index('dv')
    assert set(q.index)=={1,3,6,12,24}
    hi=q.loc[24];lo=q.loc[1]
    decision=dict(expand=bool(hi.relative_gain>=.05 and hi.wins>=8 and hi.ci_high<0 and hi.relative_gain-lo.relative_gain>=.05),
                  high_dv=hi.to_dict(),low_dv=lo.to_dict(),gain_increase=float(hi.relative_gain-lo.relative_gain),
                  interpretation='Original predeclared resource gate; only summary field collision repaired.')
    nc.dump(folder/'screen_decision.json',decision)
    nc.dump(folder/'STATUS.json',dict(status='COMPLETE',rows=len(d),cells=len(d[groups].drop_duplicates()),expand=decision['expand'],
                                    authoritative_summary='summary_corrected.csv',superseded_summary='summary.csv: n column wrongly held replicate count'))
    print(json.dumps(decision),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expand',action='store_true');parser.add_argument('--summary-only',action='store_true');args=parser.parse_args()
    freeze(OUT/'p2/summary_repair',dict(change='Rename statistical replicate count to replicates, preserve training n.',
            training_changed=False,metrics_changed=False,gate_changed=False),[__file__])
    p2.scores=fast.scores;p2.summarize=summarize
    summarize(OUT/'p2') if args.summary_only else p2.run(OUT,args.expand)
