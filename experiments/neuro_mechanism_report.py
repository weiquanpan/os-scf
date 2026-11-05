"""Evidence-led tables, publication figures and Chinese interpretation of P0-P5."""
from experiments.neuro_mechanism_utils import *
from experiments.neuro_mechanism_real import DATASETS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
REPORT=OUT/'report'
NAMES={'lendingclub':'LendingClub','bikesharing':'BikeSharing','taxi':'Taxi'}
COLORS={'osscf':'#1768ac','direct_repair':'#e57928','tabsyn':'#7c4da5','tabsyn_repair':'#4c9575'}
DISPLAY={'osscf':'OS-SCF','direct_repair':r'Direct+$\Phi$','tabsyn':'TabSyn','tabsyn_repair':r'TabSyn+$\Phi$'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})

def save(fig,name):
    fig.savefig(REPORT/(name+'.png'),bbox_inches='tight');fig.savefig(REPORT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def holm(values):
    p=np.asarray(values);order=np.argsort(p);adj=np.empty(len(p));last=0.
    for j,i in enumerate(order):last=max(last,float(p[i])*(len(p)-j));adj[i]=min(1.,last)
    return adj

def format_table(df):return df.to_string(index=False,float_format=lambda x:f'{x:.6f}')

def p2_report():
    d=pd.read_csv(OUT/'p2/metrics.csv');s=pd.read_csv(OUT/'p2/summary_corrected.csv')
    s['holm_p_all_cell_metrics']=holm(s.signflip_p);s.to_csv(REPORT/'p2_all_cell_statistics.csv',index=False)
    anchor=dict(dv=6,L=1.,sigma=0.,n=5000,steps=800)
    def delta(config):
        q=d.copy()
        for k,v in config.items():q=q[q[k]==v]
        pivot=q.pivot(index='seed',columns='method',values='free_sw')
        return pivot.osscf-pivot.direct_repair
    primary=[]
    for name,axis,low,high,pred in [('dependency_dimension','dv',1,24,'negative'),('nonlinearity_bound','L',.5,4.,'negative'),
                                    ('graph_noise','sigma',0.,.1,'positive'),('training_samples','n',1000,20000,'positive'),('training_steps','steps',400,1600,'positive')]:
        lo=dict(anchor);lo[axis]=low;hi=dict(anchor);hi[axis]=high
        diff=delta(hi)-delta(lo);stat=paired_summary(diff)
        primary.append(dict(contrast=name,predicted_direction=pred,**stat))
    p=pd.DataFrame(primary);p['holm_p_5_primary']=holm(p.signflip_p)
    p['interval_supports_prediction']=[(row.ci_high<0 if row.predicted_direction=='negative' else row.ci_low>0) for row in p.itertuples()]
    p.to_csv(REPORT/'p2_primary_contrasts.csv',index=False)
    grid=s[(s.metric=='free_sw')&(s.steps==800)]
    monotonic=[]
    for axis,others,increasing in [('dv',['L','sigma','n'],True),('L',['dv','sigma','n'],True),('sigma',['dv','L','n'],False),('n',['dv','L','sigma'],False)]:
        flags=[]
        for _,g in grid.groupby(others):
            gains=g.sort_values(axis).relative_gain.to_numpy();flags.append(bool((np.diff(gains)>=-1e-12).all() if increasing else (np.diff(gains)<=1e-12).all()))
        monotonic.append(dict(axis=axis,slices=len(flags),strictly_monotone_in_predicted_direction=sum(flags),fraction=float(np.mean(flags))))
    monotonic=pd.DataFrame(monotonic);monotonic.to_csv(REPORT/'p2_monotonicity.csv',index=False)
    fig,axes=plt.subplots(4,3,figsize=(12,12),layout='constrained')
    limit=max(5.,float(np.ceil(abs(grid.relative_gain).max()*100/5)*5))
    for i,noise in enumerate([0.,.01,.05,.1]):
        for j,n in enumerate([1000,5000,20000]):
            ax=axes[i,j];q=grid[(grid.sigma==noise)&(grid.n==n)].pivot(index='L',columns='dv',values='relative_gain')*100
            im=ax.imshow(-q.values,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
            ax.set_xticks(range(5),[1,3,6,12,24]);ax.set_yticks(range(4),[.5,1,2,4]);ax.set_title(f'noise={noise:g}; n={n:,}')
            ax.set_xlabel('dependent dimensions');ax.set_ylabel('L')
            for y in range(4):
                for x in range(5):ax.text(x,y,f'{-q.values[y,x]:.0f}',ha='center',va='center',fontsize=8,color='white' if abs(q.values[y,x])>.6*limit else '#111111')
    fig.colorbar(im,ax=axes,shrink=.55,label='100 x (OS - Direct+Phi) / Direct+Phi, Free SW; negative favors OS')
    fig.suptitle('All 240 synthetic cells: 10 paired instances per cell; 800 training steps')
    save(fig,'p2_phase_map')
    for metric,filename in [('full_sw','p2_full_sw_phase_map'),('empirical_w2','p2_empirical_w2_phase_map')]:
        metric_grid=s[(s.metric==metric)&(s.steps==800)]
        fig,axes=plt.subplots(4,3,figsize=(12,12),layout='constrained')
        bound=max(5.,float(np.ceil(abs(metric_grid.relative_gain).max()*100/5)*5))
        for i,noise in enumerate([0.,.01,.05,.1]):
            for j,n in enumerate([1000,5000,20000]):
                ax=axes[i,j];q=metric_grid[(metric_grid.sigma==noise)&(metric_grid.n==n)].pivot(index='L',columns='dv',values='relative_gain')*100
                im=ax.imshow(-q.values,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
                ax.set_xticks(range(5),[1,3,6,12,24]);ax.set_yticks(range(4),[.5,1,2,4]);ax.set_title(f'noise={noise:g}; n={n:,}')
                ax.set_xlabel('dependent dimensions');ax.set_ylabel('L')
                for y in range(4):
                    for x in range(5):
                        value=-q.values[y,x];label='0' if abs(value)<.5 else f'{value:.0f}'
                        ax.text(x,y,label,ha='center',va='center',fontsize=8,color='white' if abs(value)>.6*bound else '#111111')
        fig.colorbar(im,ax=axes,shrink=.55,label='Relative difference (%), OS - Direct+Phi; negative favors OS')
        title='Full SW' if metric=='full_sw' else 'Exact empirical W2: 256-row subsets, 3 draws'
        fig.suptitle(title+'; all 240 cells, 10 paired instances per cell')
        save(fig,filename)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,axis,levels in zip(axes,['dv','L','steps'],[[1,3,6,12,24],[.5,1,2,4],[400,800,1600]]):
        q=s[s.metric=='free_sw'].copy()
        for k,v in anchor.items():
            if k!=axis:q=q[q[k]==v]
        q=q.sort_values(axis);ax.errorbar(q[axis],q['mean'],yerr=np.stack([q['mean']-q.ci_low,q.ci_high-q['mean']]),marker='o',capsize=3,color='#1768ac')
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(levels);ax.set_xlabel(axis);ax.set_ylabel('OS - Direct+Phi, Free SW')
    fig.suptitle('Prespecified anchor contrasts; percentile intervals over 10 paired synthetic instances')
    save(fig,'p2_anchor_contrasts')
    # Noise floor is reported only where the graph is affine.
    floor=d[(d.L==.5)&(d.method=='osscf')&(d.steps==800)].groupby(['dv','sigma','n'])[['affine_reference_support_floor','reference_clean_empirical_offset']].mean()
    floor.to_csv(REPORT/'p2_affine_noise_floor.csv')
    gate=json.loads((OUT/'p2/screen_decision.json').read_text())
    return ['P2：完整240格，加400/1600步两格，共242格×10配对seed=4840行指标。独立优化轨迹2450条；OS跨图映射配置合理复用，不能把复用行当独立重复。',
            f'冻结主门槛：dv24相对FreeSW改善{100*gate["high_dv"]["relative_gain"]:.2f}%，胜{gate["high_dv"]["wins"]}/10；dv1改善{100*gate["low_dv"]["relative_gain"]:.2f}%。因此按原规则展开完整网格。',
            '五个预定轴的端点差分（比较的是OS−Direct的FreeSW差值；负值有利OS；方向随假设而异）：',format_table(p),
            '整个网格中符合严格单调预测的切片比例（这是描述，不是显著性检验）：',format_table(monotonic),
            '每格区间是10个合成人口/映射/训练实例之间的探索性区间。精确符号翻转均值检验需要对称性/可交换性条件；五个主要对比做Holm校正，全部单格/指标的校正另表保留。多个投影seed不是额外训练重复。',
            '相图支持的是这个随机特征图、这组归一化和有限预算下的现象；不能据此把更多依赖维数的效应唯一归因于流形奇异性，或外推成任意非线性结构的定理。']

def p1_report():
    d=pd.read_csv(OUT/'p1/metrics.csv');spectra=pd.read_csv(OUT/'p1/spectra.csv');records=[]
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for col,ds in enumerate(DATASETS):
        for method in ['osscf','direct_repair']:
            for row,metric in enumerate(['full_sw','dep_sw']):
                q=d[(d.dataset==ds)&(d.method==method)].groupby(['split','eps'])[metric].mean().unstack('split').sort_index()
                mean=q.mean(1);sd=q.std(1);ax=axes[row,col]
                ax.plot(q.index,mean,'o-',label=DISPLAY[method],color=COLORS[method]);ax.fill_between(q.index,mean-sd,mean+sd,alpha=.12,color=COLORS[method])
                ax.set_xscale('log');ax.set_xlabel('ridge epsilon');ax.set_ylabel(metric);ax.set_title(NAMES[ds]);ax.grid(alpha=.15)
            q=d[(d.dataset==ds)&(d.method==method)].groupby('eps').full_sw.mean().sort_index(ascending=False)
            records.append(dict(dataset=ds,method=method,full_sw_eps_point1=float(q.iloc[0]),full_sw_eps_point00001=float(q.iloc[-1]),
                                small_minus_large=float(q.iloc[-1]-q.iloc[0]),monotone_worse_as_epsilon_decreases=bool((np.diff(q)>=0).all()),
                                relative_range=float((q.max()-q.min())/q.mean())))
    axes[0,0].legend(fontsize=8);fig.suptitle('P1: fixed training budget; shading = SD of three overlapping split means')
    save(fig,'p1_epsilon_sensitivity');r=pd.DataFrame(records);r.to_csv(REPORT/'p1_epsilon_endpoints.csv',index=False)
    return ['P1：270次固定预算拟合，3split×3新seed×5epsilon×2方法×3数据集；使用验证集，没有按epsilon筛选赢家。',
            '训练协方差实测秩/条件数（数值奇异的condition_number留空，不写成有限值）：',
            format_table(spectra[['dataset','split','space','numerical_rank','dimension','min_eigenvalue','condition_number']]),
            'epsilon从.1降到.00001的FullSW变化：',format_table(r),
            '非单调曲线同样保留。奇异协方差和W放大界本身不构成Direct生成误差必然单调恶化的证明。']

def p3_report():
    d=pd.read_csv(OUT/'p3/metrics.csv');jac=pd.read_csv(OUT/'p3/jacobians.csv');records=[]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        for method in ['osscf','direct_repair']:
            q=d[(d.dataset==ds)&(d.method==method)].groupby('sample_steps').full_sw.mean()
            ax.plot(q.index,q,'o-',label=DISPLAY[method],color=COLORS[method])
        ax.set_xscale('log',base=2);ax.set_xticks([4,8,16,32,80],[4,8,16,32,80]);ax.set_xlabel('Euler steps');ax.set_ylabel('Full SW');ax.set_title(NAMES[ds])
        os=d[(d.dataset==ds)&(d.method=='osscf')&(d.sample_steps==16)].set_index(['split','seed'])
        direct=d[(d.dataset==ds)&(d.method=='direct_repair')&(d.sample_steps==80)].set_index(['split','seed'])
        block=(os.full_sw-direct.full_sw).groupby('split').mean();stat=paired_summary(block)
        records.append(dict(dataset=ds,os_16=float(os.full_sw.mean()),direct_80=float(direct.full_sw.mean()),
                            relative_gap=float(os.full_sw.mean()/direct.full_sw.mean()-1),**stat,
                            observed_sampling_speed_ratio=float(direct.sampling_seconds_median.mean()/os.sampling_seconds_median.mean())))
    axes[0].legend(fontsize=8);save(fig,'p3_ode_sensitivity')
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for col,ds in enumerate(DATASETS):
        for row,coord in enumerate(['z','standardized_numeric']):
            for method in ['osscf','direct_repair']:
                q=jac[(jac.dataset==ds)&(jac.coordinate==coord)&(jac.method==method)].groupby('time')['mean'].mean()
                axes[row,col].plot(q.index,q,'o-',label=DISPLAY[method],color=COLORS[method])
            axes[row,col].set_title(NAMES[ds]+' / '+coord);axes[row,col].set_xlabel('time');axes[row,col].set_ylabel('mean local spectral norm')
    axes[0,0].legend(fontsize=8);save(fig,'p3_local_jacobians')
    r=pd.DataFrame(records);r.to_csv(REPORT/'p3_os16_vs_direct80.csv',index=False)
    endpoint=jac[jac.time==.9875].groupby(['dataset','coordinate','method'])[['mean','p95','max','trajectory_clip_fraction']].mean().reset_index()
    endpoint.to_csv(REPORT/'p3_endpoint_jacobians.csv',index=False)
    return ['P3：全部90个冻结checkpoint、5个Euler步数，共450组采样。80步的样本和原128投影指标均按逐次记录进行重放核验。',
            '预定OS 16步与Direct+Phi 80步对照：',format_table(r),
            '局部Jacobian端点诊断（t=.9875，64条轨迹；下表max是每模型最大值再取平均）：',format_table(endpoint),
            '速度比只描述当前硬件的顺序采样计时；质量不等价时不能单独拿速度比宣称同质加速。有限轨迹局部Jacobian不证明全局Lipschitz常数或端点发散。',
            '现稿附录S2的ell_k是学得场b_theta的Lipschitz常数，不是理想b*的常数。因此，不能直接从图支撑理想场的端点奇异性推出该命题中的g(ell_k)爆炸。当前有限权重Linear/SiLU网络的全局Lipschitz上界有限。']

def p0_report():
    d=pd.read_csv(OUT/'p0/metrics.csv');old=pd.read_csv(OUT/'p3/metrics.csv');records=[];settings=[];urows=[];mc=[]
    for ds in DATASETS:
        for selector in ['method_specific','shared']:
            sub=d[(d.dataset==ds)&(d.selector==selector)]
            for method in ['osscf','direct_repair']:
                q=sub[sub.method==method]
                records.append(dict(dataset=ds,selector=selector,method=method,full_sw=float(q.full_sw.mean()),
                                    free_sw=float(q.free_sw.mean()),dep_sw=float(q.dep_sw.mean()),
                                    quantile_full_sw=float(q.quantile_full_sw.mean()) if ds=='taxi' else np.nan))
            pivot=sub.pivot(index=['split','seed'],columns='method',values='full_sw');delta=pivot.osscf-pivot.direct_repair
            projected=[]
            for i in range(3):
                p=sub.pivot(index=['split','seed'],columns='method',values=f'full_sw_proj{i}')
                projected.append(p.osscf-p.direct_repair)
            mc.append(dict(dataset=ds,selector=selector,mean_paired_difference=float(delta.mean()),
                           mean_projection_MC_SD=float(pd.concat(projected,axis=1).std(axis=1).mean()),
                           projection_MC_SD_of_aggregated_difference=float(np.std([x.mean() for x in projected],ddof=1))))
        for sp in [3101,3102,3103]:
            chosen=json.loads((OUT/'p0/selection'/ds/f'split{sp}'/'selected.json').read_text())
            for method,config in chosen.items():settings.append(dict(dataset=ds,split=sp,selector_key=method,**config))
    for path in (OUT/'p0/final').glob('*/*/*/*/*/utility.json'):
        meta=json.loads(path.with_name('metrics.json').read_text())
        for r in json.loads(path.read_text()):
            for selector in meta['selectors']:urows.append(dict(r,selector=selector))
    u=pd.DataFrame(urows);u.to_csv(REPORT/'p0_utility_per_selector.csv',index=False)
    utilities=[]
    for ds,g in u[u.utility_model=='rf'].groupby('dataset'):
        metric='roc_auc' if ds=='lendingclub' else 'r2'
        for (sel,method),v in g.groupby(['selector','method']):utilities.append(dict(dataset=ds,selector=sel,method=method,rf_metric=metric,value=float(v[metric].mean())))
    r=pd.DataFrame(records);r.to_csv(REPORT/'p0_fidelity.csv',index=False)
    search_times=[]
    for path in (OUT/'p0/worker_timing').glob('*.json'):search_times.append(json.loads(path.read_text()))
    pd.DataFrame(search_times).to_csv(REPORT/'p0_search_and_final_wall_times.csv',index=False)
    pd.DataFrame(settings).to_csv(REPORT/'p0_selected_settings.csv',index=False);pd.DataFrame(mc).to_csv(REPORT/'p0_projection_noise.csv',index=False)
    pd.DataFrame(utilities).to_csv(REPORT/'p0_rf_utility.csv',index=False)
    stats=pd.read_csv(OUT/'p0/split_block_statistics.csv');stats['holm_p_all_p0']=holm(stats.signflip_p)
    stats.to_csv(REPORT/'p0_block_statistics.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        values=[]
        for method in ['osscf','direct_repair']:
            values.append([old[(old.dataset==ds)&(old.method==method)&(old.sample_steps==80)].full_sw.mean(),
                           d[(d.dataset==ds)&(d.method==method)&(d.selector=='shared')].full_sw.mean(),
                           d[(d.dataset==ds)&(d.method==method)&(d.selector=='method_specific')].full_sw.mean()])
        for i,method in enumerate(['osscf','direct_repair']):ax.plot(range(3),values[i],'o-',label=DISPLAY[method],color=COLORS[method])
        ax.set_xticks(range(3),['Original\nshared','Expanded\nshared','Expanded\nper method']);ax.set_ylabel('test Full SW, 1024 x 3 projections');ax.set_title(NAMES[ds])
    axes[0].legend(fontsize=8);fig.suptitle('Point means only; split variation is shown in the paired-difference figure')
    save(fig,'p0_selection_point_means')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        blocks=[]
        selections=[old[(old.dataset==ds)&(old.sample_steps==80)],
                    d[(d.dataset==ds)&(d.selector=='shared')],d[(d.dataset==ds)&(d.selector=='method_specific')]]
        for selection in selections:
            p=selection.pivot(index=['split','seed'],columns='method',values='full_sw')
            blocks.append((p.osscf-p.direct_repair).groupby('split').mean())
        curves=pd.concat(blocks,axis=1)
        for sp,row in curves.iterrows():ax.plot(range(3),row,'o-',alpha=.65,lw=1,label=f'split {sp}')
        ax.plot(range(3),curves.mean(0),'D-',color='black',lw=2,label='mean of splits')
        ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(3),['Original\nshared','Expanded\nshared','Expanded\nper method'])
        ax.set_title(NAMES[ds]);ax.set_ylabel(r'Full SW: OS-SCF - Direct+$\Phi$')
    axes[0].legend(fontsize=7);fig.suptitle('Three overlapping split means; each averages five paired training seeds')
    save(fig,'p0_selection_comparison')
    final_records=[json.loads(p.read_text()) for p in (OUT/'p0/final').glob('*/*/*/*/*/metrics.json')]
    reused=sum(r['reused_checkpoint'] is not None for r in final_records)
    return ['P0：两个方法都搜索25格×2选择seed；共900个验证候选。按方法选择和共享选择从同一候选表计算。test按两个选择器所需配置并集训练/复用，并非把同一个模型重复算作新拟合。',
            f'最终不同模型配置记录{len(final_records)}份，其中复用旧checkpoint {reused}份、新训练{len(final_records)-reused}份；用于两种选择器的报告行共180行。',
            '测试保真度：',format_table(r),'RF下游效用：',format_table(pd.DataFrame(utilities)),
            'FullSW的按split分块统计：',format_table(stats[stats.metric=='full_sw']),
            '投影蒙特卡洛噪声（3组方向只是同一组生成样本的测量重复）：',format_table(pd.DataFrame(mc)),
            '原共享与扩展共享之间同时改变了搜索网格和选择度量，不能把这段差异全部归因于共享规则。扩展共享与扩展按方法选择才是同一搜索空间内对选择规则的对照。',
            '本次三个数据集的按方法选择FullSW块区间均跨0；15对训练seed中OS分别赢LC 8对、Bike 8对、Taxi 10对。平均值更低没有建立稳定/统计确定的普遍优势。',
            '只有3个重叠split，不能把15个训练结果当15个独立数据集；8种块翻转的双侧p值通常最小.25。区间也是描述性的。此前已使用过的test集合不会因换协议重新变成未触碰的外部验证。',
            'Taxi分位数SW压缩尾部并改变几何，独立作为稳健补充；它没有参与超参数选择，不能替换原指标来隐藏不利结果。',
            '搜索/最终评估进程的完整墙钟（3数据集并行，不能把三个进程时间求和当实际总墙钟）：',format_table(pd.DataFrame(search_times)),
            json.dumps(json.loads((OUT/'p0/wall_times.json').read_text()),ensure_ascii=False)]

def p4_report():
    d=pd.read_csv(OUT/'p4/metrics_with_repair_timing.csv');c=pd.read_csv(OUT/'p4_curve/metrics.csv');ratios=[];thresholds=[]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        for method in ['osscf','direct_repair']:
            q=c[(c.dataset==ds)&(c.method==method)].sort_values('steps')
            ax.plot(q.pipeline_cumulative_seconds,q.full_sw,'o-',label=DISPLAY[method],color=COLORS[method])
        for method,marker in [('tabsyn','s'),('tabsyn_repair','D')]:
            q=d[(d.dataset==ds)&(d.method==method)].iloc[0]
            ax.scatter([q.training_pipeline_seconds],[q.full_sw],marker=marker,label=DISPLAY[method],color=COLORS[method],s=45)
        os=d[(d.dataset==ds)&(d.method=='osscf')].iloc[0];tab=d[(d.dataset==ds)&(d.method=='tabsyn_repair')].iloc[0]
        for method,multiplier in itertools.product(['osscf','direct_repair'],[1.,1.05]):
            curve=c[(c.dataset==ds)&(c.method==method)].sort_values('steps');hit=curve[curve.full_sw<=multiplier*tab.full_sw]
            thresholds.append(dict(dataset=ds,method=method,threshold_multiplier=multiplier,threshold=multiplier*float(tab.full_sw),
                                   reached=not hit.empty,first_observed_steps=int(hit.iloc[0].steps) if not hit.empty else None,
                                   first_observed_seconds=float(hit.iloc[0].pipeline_cumulative_seconds) if not hit.empty else None,
                                   best_observed_full_sw=float(curve.full_sw.min())))
        ratios.append(dict(dataset=ds,tabsyn_over_os_fit_time=float(tab.training_pipeline_seconds/os.training_pipeline_seconds),
                           tabsyn_over_os_sample_time=float(tab.sampling_5000_median_seconds/os.sampling_5000_median_seconds),
                           os_full_sw=float(os.full_sw),tabsyn_repair_full_sw=float(tab.full_sw)))
        ax.set_xscale('log');ax.set_xlabel('Fit + internal validation (seconds)\nExternal search excluded');ax.set_ylabel('validation Full SW');ax.set_title(NAMES[ds])
    axes[0].legend(fontsize=8);save(fig,'p4_quality_time')
    r=pd.DataFrame(ratios);r.to_csv(REPORT/'p4_descriptive_ratios.csv',index=False)
    pd.DataFrame(thresholds).to_csv(REPORT/'p4_quality_thresholds.csv',index=False)
    return ['P4：同机顺序重训9个模型，再顺序跑6条预算曲线；各数据集一个新seed110。表中TabSyn+Phi与TabSyn共享模型，不能重复计算训练成本。',
            'TabSyn沿用现稿的有上限早停adapter与已审计上游组件，并非完整官方recipe的等价重现；具体配置及是否触达上限保留在逐次fit_info中。',
            format_table(d[['dataset','method','train_parameters','inference_parameters','training_pipeline_seconds','sampling_5000_median_seconds','full_sw']]),
            '当前环境的描述性比值：',format_table(r),'预先规定的TabSyn+Phi最终质量阈值及放宽5%阈值：',format_table(pd.DataFrame(thresholds)),
            '未达到阈值时不填运行时间；首次观察到达仅限五个预算checkpoint，不是连续时间命中测量。TabSyn无中间曲线，不能把其最终训练时间当成最早达到阈值的时间。',
            '桌面GPU未保证严格独占，因此这仍不是严格独占设备的基准。比值没有跨seed时间置信区间，也不代表同质量加速。图中TabSyn只是最终点，不是虚构的训练时间前沿。',
            'P0外部超参数搜索与TabSyn内部早停的开销口径不同：曲线中单次拟合/内部验证时间单列，P0全部搜索预算另表保留；不能据此写“千分之一总计算成本”。']

def p5_report():
    d=pd.read_csv(OUT/'p5/metrics.csv');s=d.groupby(['dataset','method','constraint'])[['violation_rate','undefined_rate','invalid_or_violation_rate']].mean().reset_index()
    s.to_csv(REPORT/'p5_inequalities.csv',index=False)
    return ['P5：复用270份生成表及9份真实参考，检查隐含不等式；不新增带目标泄漏的下游任务。',format_table(s),
            '条件SW/稀有组覆盖与关系诊断复用 outputs/neurocomputing_advantage_c1_v1。此前C1并未发现跨三个真实数据集的一般条件保真优势：LC及Taxi部分指标仍不利。',
            '依赖列重构带来的有效性是OS、Direct+Phi、TabSyn+Phi和Free-TabSyn的共享收益；原始全表生成器的不一致不能被当作OS独有优势。']

def tex_assets():
    def table(name,caption,label,columns,header,rows):
        lines=[r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
               r'\begin{tabular}{'+columns+'}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
        lines += [' & '.join(row)+r' \\' for row in rows]
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (REPORT/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    d=pd.read_csv(OUT/'p0/metrics.csv');q=d[d.selector=='method_specific'].groupby(['dataset','method','split'])[['full_sw','free_sw','dep_sw']].mean()
    summary=q.groupby(['dataset','method']).agg(['mean','std']);rows=[]
    for ds,method in itertools.product(DATASETS,['osscf','direct_repair']):
        r=summary.loc[(ds,method)]
        rows.append([NAMES[ds],DISPLAY[method]]+[f'${r[(metric,"mean")]:.4f}\\pm {r[(metric,"std")]:.4f}$' for metric in ['full_sw','free_sw','dep_sw']])
    table('p0_method_specific.tex',r'Method-specific validation selection on the expanded grid. Means and descriptive SDs are across three overlapping split means, each averaging five seeds. SW uses three fixed sets of 1,024 directions; lower is better. These reused test partitions do not provide fresh external confirmation.',
          'tab:followup-p0','llrrr',['Dataset','Method','Full SW','Free SW','Dep. SW'],rows)
    s=pd.read_csv(OUT/'p2/summary_corrected.csv');q=s[(s.L==1)&(s.sigma==0)&(s.n==5000)&(s.steps==800)&(s.metric=='free_sw')].sort_values('dv')
    rows=[[str(int(r.dv)),f'{r.os_mean:.4f}',f'{r.direct_mean:.4f}',f'{r["mean"]:.4f}',f'$[{r.ci_low:.4f}, {r.ci_high:.4f}]$'] for _,r in q.iterrows()]
    table('p2_dependency_anchor.tex',r'Prespecified dependency-dimension slice: $d_u=6$, $L=1$, no graph noise, 5,000 training rows and 800 steps. Differences are OS-SCF minus Direct+$\Phi$. Percentile intervals use ten paired synthetic instances and are exploratory; OS checkpoints are reused across $d_v$.',
          'tab:followup-p2','rrrrl',[r'$d_v$','OS Free SW',r'Direct+$\Phi$ Free SW',r'$\Delta$ Free SW',r'95\% interval'],rows)
    d=pd.read_csv(OUT/'p4/metrics_with_repair_timing.csv');rows=[]
    for ds,method in itertools.product(DATASETS,['osscf','direct_repair','tabsyn_repair']):
        r=d[(d.dataset==ds)&(d.method==method)].iloc[0]
        rows.append([NAMES[ds],DISPLAY[method],f'{r.train_parameters/1e6:.3f}',f'{r.training_pipeline_seconds:.2f}',f'{1000*r.sampling_5000_median_seconds:.2f}',f'{r.full_sw:.4f}'])
    table('p4_cost_quality.tex',r'Fresh sequential same-machine cost audit, split 3101 and seed 110. Training includes preprocessing, fitting, internal validation and saving; external P0 search is separate. Sampling is the median of five warmed runs for 5,000 rows, including reconstruction. The desktop GPU is not certified exclusive. Quality is validation Full SW; different achieved quality precludes an automatic equal-quality speedup claim.',
          'tab:followup-p4','llrrrr',['Dataset','Method','Train params (M)','Fit (s)','Sample (ms)','Full SW'],rows)

def run():
    checks=json.loads((OUT/'completion_checks.json').read_text());assert checks['verification_status']=='VERIFIED'
    REPORT.mkdir(parents=True,exist_ok=True)
    text=['本次P2→P1→P3→P0实验及P4/P5补充结果',
          '数值状态：VERIFIED（见completion_checks.json）；科学解释状态：ANALYZED，不能把数值可复现等同于机制已经被证明。',
          '所有新结果保存在独立目录，原主实验及paper-v6文件哈希保持不变。本文不把可选P6重新启动；此前Jacobian小测试的未通过结论保留。']
    for fn in [p2_report,p1_report,p3_report,p0_report,p4_report,p5_report]:text+=['',*fn()]
    tex_assets()
    fidelity=pd.read_csv(REPORT/'p0_fidelity.csv');primary=fidelity[fidelity.selector=='method_specific']
    simple=[]
    for ds in DATASETS:
        q=primary[primary.dataset==ds].set_index('method');a=float(q.loc['osscf','full_sw']);b=float(q.loc['direct_repair','full_sw'])
        simple.append(f'{NAMES[ds]}：OS {a:.6f}，Direct+Phi {b:.6f}，平均相对降低{100*(1-a/b):.2f}%。')
    brief=['结论：更可靠的优势证据集中在高冗余、有限预算的受控合成情形；真实数据平均值小幅改善，但没有建立跨split稳定的普遍保真优势。',
           'P2：完整240格+两个预算点，10个配对实例。预定高依赖主配置Free SW降低43.48%、Full SW降低36.79%、256行精确经验W2降低7.80%，均10/10配对胜。五个主要对比中，依赖维数和预算端点对比得到支持；非线性、噪声、样本量的简单单调预测未得到支持。',
           'P0：各方法独立验证选择后的测试Full SW：',*simple,
           '这三个平均差的split分块区间都跨0；LC/Bike/Taxi分别8/15、8/15、10/15训练配对胜。Taxi分位数SW仍以Direct更好，不能宣称所有度量全面获胜。',
           'P1：LC和Bike环境协方差秩亏得到实测确认；“epsilon越小Direct必然越差、OS更平坦”的预测未普遍成立。',
           'P3：Direct局部Jacobian较大，但OS 16步未普遍达到Direct 80步的质量。现稿S2的ell属于学得场，不能直接用理想场的端点奇异性推出该界爆炸。',
           'P4：指定配置的同机单seed顺序计时中，TabSyn/OS拟合耗时比约11—20，5k行采样耗时比约44—53；不是千倍，也不是等质量或包含外部调参的总成本比。LC上流模型没有达到当前TabSyn+Phi的最终验证SW阈值。GPU未保证严格独占。',
           'P5：隐含约束检查已完成，条件诊断复用C1；有效性收益由所有相应精确重构方法共享。P6未重启此前未通过门槛的Jacobian改动。',
           '数值复现核验通过；原107个主实验源码和30个论文交付文件未变。完整逐次结果、10组PNG/PDF图、3份LaTeX表片段和复现命令均已保存。']
    (REPORT/'EXECUTIVE_zh.txt').write_text('\n\n'.join(brief)+'\n',encoding='utf-8')
    text=brief+['','以下为完整结果与方法审查。']+text
    risks=[
        ('Simpson paradox','按dataset、split、dv/L/noise/n分别报告；不以跨数据集合并均值抹掉异质性。'),
        ('Ecological fallacy','合成实例与真实split是明确分析单位；不推断单个用户、借款人或任意真实总体的效果。'),
        ('Berkson selection','真实数据来自既有筛选池，split重复且部分重叠；外推受筛选和历史检视限制。'),
        ('Collider bias','不按test效果筛掉配置或分组；P2网格扩展由冻结筛查门槛决定并公开。'),
        ('Base-rate neglect','不等式报告真实参考率和未定义率；C1稀有类别覆盖保留，不能只报看似漂亮的零违约。'),
        ('Regression to mean','筛查后完整网格使用同10seed，是展开探索，不冒充独立确认；鲜明的高dv筛查效果可能高估泛化收益。'),
        ('Survivorship','全部242格和所有方法保留；软件汇总故障、计算后端变更、并行执行有修复记录，不删除不利格子。'),
        ('Look-elsewhere','五个主要轴对比做Holm校正；全部单格指标另做整体校正，未校正区间只作探索性图示。'),
        ('Forking paths','配置、随机数、门槛冻结；修复统计字段n冲突及tuple/list的JSON表示差异，SW后端改变经双精度核对；没有结果驱动的改seed或调权重。'),
        ('Correlation/causation','维数扫描同时改变容量/输出任务/损失分配；谱、局部Jacobian与性能关系不单独证明因果机制。'),
        ('Reverse causality','合成轴由实验指定，性能不会反向改变轴；真实数据关联诊断不据此推出因果方向。')]
    nc.dump(REPORT/'interpretation.json',dict(status='ANALYZED',fallacy_scan_coverage='11/11',items=[dict(name=k,assessment=v) for k,v in risks],
            numeric_reproduction='VERIFIED separately in ../completion_checks.json',statistical_limitations=['three overlapping real splits','ten synthetic instances','finite-sample exact OT','local empirical Jacobians','one cost seed']))
    text+=['','统计与复现审查（11/11）：']+[f'{i+1}. {k}: {v}' for i,(k,v) in enumerate(risks)]
    text+=['','技术修复记录：P2最初汇总将训练样本数n覆盖为重复次数；原错误summary.csv保留，正式表为summary_corrected.csv。未改变训练或逐次指标。',
           'P1首次启动的配置校验因JSON把tuple读成list而误报变化；用规范JSON表示后与原冻结值一致，原失败日志保留，未改变实验配置。TabSyn进度日志另行重定向到本次目录，避免写入旧实验。',
           'SW从CPU双精度切换到等价GPU双精度，固定投影矩阵不变，核验最大差异约机器精度。独立进程800步重训参数逐数复现后，按seed/数据集并行非计时实验；P3、P4和P4曲线顺序计时。',
           '详细设定：../PLAN_zh.txt。理论成立条件与不能外推的部分：../THEORY_NOTES_zh.txt。所有原始样本、checkpoint和逐次指标均可追溯。']
    (REPORT/'RESULTS_zh.txt').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    manifest=dict(report_source_sha=nc.sha(__file__),artifacts={str(p.relative_to(OUT)):nc.sha(p) for p in REPORT.iterdir() if p.is_file() and p.name!='manifest.json'},
                  numeric_checks_sha=nc.sha(OUT/'completion_checks.json'))
    nc.dump(REPORT/'manifest.json',manifest)
    print('Report complete:',REPORT,flush=True)

if __name__=='__main__':run()
