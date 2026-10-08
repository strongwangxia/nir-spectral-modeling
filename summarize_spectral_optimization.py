from pathlib import Path
import os,csv,json,statistics,math
import numpy as np
import joblib
from spectral_optimization_core import predict_model,preprocess,metrics
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'建模结果'/'模型优化_前400训练_后133测试'
summary=json.loads((OUT/'优化结果.json').read_text(encoding='utf-8'))
rows=summary['finalists'];winner=summary['winner_test'];cvwinner=summary['winner_cv']
sepwinner=min(rows,key=lambda r:r['SEP'])
joblib.dump(joblib.load(OUT/(sepwinner['id']+'_model.joblib')),OUT/'最低SEP模型.joblib')
arrays=np.load(OUT/'对照绘图数据.npz')
ids=arrays['ids'];y=arrays['y'];preds=arrays['predictions'];candidate_ids=arrays['candidate_ids']
assert len(ids)==133 and not set(ids)&{460,461,462,463,494,495,496,520,521}
for r in rows:
    p=preds[np.flatnonzero(candidate_ids==r['id'])[0]]
    e=[float(a-b) for a,b in zip(p,y)]
    assert math.isclose(statistics.mean(abs(v) for v in e),r['MAE'],rel_tol=1e-12)
    assert math.isclose(statistics.stdev(e),r['SEP'],rel_tol=1e-12)
    assert math.isclose(r['RMSEP']**2,132/133*r['SEP']**2+r['Bias']**2,rel_tol=1e-12)
oldrows=list(csv.DictReader((ROOT/'建模结果'/'前400建模_后142预测'/'后142条逐样本预测.csv').open(encoding='utf-8-sig')))
oldmap={int(r['光谱编号']):r for r in oldrows}
for label,column in [('保留异常候选','保留模型预测'),('排除异常候选','排除模型预测')]:
    r=next(r for r in rows if r['strategy']==label and r['stage']=='原始基线')
    p=preds[np.flatnonzero(candidate_ids==r['id'])[0]]
    assert np.allclose(p,[float(oldmap[int(i)][column]) for i in ids],rtol=1e-9,atol=1e-9)

# Save feature/target correlation for interpretation, not for test-driven selection.
x=arrays['x'];TX=arrays['train_X'];ty=arrays['train_y']
selected=set(summary['fold_fit_protocol']['training_selected_ids'])
normal=np.array([i not in selected for i in range(1,401)])
spec={'name':'SG一阶_w17_p3','base':'raw','kind':'sg','window':17,'poly':3,'deriv':1}
DX,_=preprocess(TX,x,spec)
def correlations(A,b):
    E=A-A.mean(0);f=b-b.mean()
    return (E.T@f)/np.sqrt((E*E).sum(0)*(f*f).sum())
rawcorr=correlations(TX[normal],ty[normal]);dcorr=correlations(DX[normal],ty[normal])
with (OUT/'训练集波长相关性参考.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['波长','原始光谱与化学值Pearson相关','SG一阶导数与化学值Pearson相关'])
    w.writerows(zip(x,rawcorr,dcorr))

os.environ['MPLCONFIGDIR']=str(ROOT/'光谱图'/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
order=sorted(rows,key=lambda r:r['MAE'])
fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
names=[r['display'].replace('异常候选','').replace(' / ','·') for r in order]
colors=['#267b9b' if r['strategy']=='保留异常候选' else '#d18632' for r in order]
for ax,metric in zip(axes,['MAE','SEP']):
    bars=ax.barh(np.arange(len(order)),[r[metric] for r in order],color=colors,height=.7)
    ax.set_yticks(np.arange(len(order)),names,fontproperties=font)
    ax.invert_yaxis();ax.bar_label(bars,fmt='%.4f',padding=5,fontsize=10)
    ax.set_xlim(0,max(r[metric] for r in order)*1.19)
    ax.set_xlabel(metric);ax.set_title(f'相同 133 条测试样本的 {metric}',fontproperties=font,fontsize=15)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('模型优化对比：蓝色保留训练异常候选，橙色排除',fontproperties=font,fontsize=18)
fig.savefig(OUT/'优化模型指标对比.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
for ax,r,tag in zip(axes,[winner,sepwinner],['MAE 最低方案','SEP 最低方案']):
    p=preds[np.flatnonzero(candidate_ids==r['id'])[0]]
    ax.scatter(y,p,color='#267b9b',s=30,alpha=.8,edgecolors='white',linewidths=.3)
    ax.plot([1.3,4.7],[1.3,4.7],color='#777777',ls='--',lw=1)
    ax.set_xlim(1.3,4.7);ax.set_ylim(1.3,4.7);ax.set_aspect('equal')
    ax.set_xlabel('真实化学值',fontproperties=font);ax.set_ylabel('预测化学值',fontproperties=font)
    ax.set_title(f'{tag}：{r["strategy"]}核岭回归\nMAE={r["MAE"]:.4f}，SEP={r["SEP"]:.4f}',fontproperties=font,fontsize=14)
    ax.grid(alpha=.18)
fig.suptitle('前 400 条训练，固定后 133 条测试',fontproperties=font,fontsize=18)
fig.savefig(OUT/'优选模型预测对比.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
ax.plot(x,rawcorr,label='原始光谱',color='#267b9b',lw=1.4)
ax.plot(x,dcorr,label='SG 一阶导数',color='#d18632',lw=1.4)
ax.axvspan(1050,1450,color='#65a9a5',alpha=.12,label='最低 SEP 模型的选定波段')
ax.axhline(0,color='#888888',lw=.8);ax.set_ylim(-1.05,1.05)
ax.set_xlabel('波长（原始单位）',fontproperties=font);ax.set_ylabel('Pearson 相关系数',fontproperties=font)
ax.set_title('仅用训练集名单外 390 条：波长与化学值的相关性',fontproperties=font,fontsize=16)
ax.legend(prop=font);ax.grid(alpha=.18)
fig.savefig(OUT/'训练集波段相关性.png',dpi=180);plt.close(fig)

def desc(r):
    if r['family']=='Blend':return '融合：'+str(r['member_ids'])+'，权重 '+str(r['weights'])
    cfg=r['preprocess'];band=' + '.join(f'{a:g}–{b:g}' for a,b in r['bands'])
    params=', '.join(f'{k}={r[k]}' for k in ['components','alpha','gamma_multiplier','C','epsilon'] if k in r)
    return f'{r["strategy"]}；{cfg["name"]}；波段 {band}；{r["family"]}；{params}'

baseline_keep=next(r for r in rows if r['stage']=='原始基线' and r['training_n']==400)
baseline_exclude=next(r for r in rows if r['stage']=='原始基线' and r['training_n']==390)
comp=lambda base,opt,metric:100*(1-opt[metric]/base[metric])
lines=['# 光谱模型优化报告','',
'## 结果','',
f'共比较 {summary["candidate_count"]} 组训练交叉验证配置，冻结 12 个模型后，对同一组 133 条样本进行预测对比。按 MAE 优先排序，最佳观察结果为 {winner["display"]}（{winner["id"]}）；按 SEP 排序则是 {sepwinner["display"]}（{sepwinner["id"]}）。没有一个模型同时在两项指标上最低。','',
'### MAE 最低方案','',desc(winner),
f'- 训练：400 条，保留名单内 10 条训练样本。SG 一阶导数：17 点窗口、3 阶多项式、波长步长 2，先全谱处理后选择变量。使用 950–1650 全部 351 个波长点。',
f'- 列标准化的均值/标准差及目标均值/标准差均由训练数据拟合。RBF 核岭 alpha={winner["alpha"]}；gamma={winner["gamma_multiplier"]}/351={winner["gamma_multiplier"]/351:.12g}。',
f'- MAE={winner["MAE"]:.6f}，SEP={winner["SEP"]:.6f}，RMSEP={winner["RMSEP"]:.6f}，Bias={winner["Bias"]:.6f}，R²={winner["R2"]:.6f}。',
f'- 相比原保留模型，MAE 降低 {comp(baseline_keep,winner,"MAE"):.2f}%，SEP 降低 {comp(baseline_keep,winner,"SEP"):.2f}%。相比此前较好的排除基线，MAE 降低 {comp(baseline_exclude,winner,"MAE"):.2f}%。','',
'### SEP 最低方案','',desc(sepwinner),
f'- 训练：390 条，排除前 400 条中的 10 条指定样本。SG 一阶导数同为17点、3阶，保留1050–1450共201个波长点；SG在全谱上计算再裁切，区间边缘导数会用到相邻窗口点。',
f'- RBF 核岭 alpha={sepwinner["alpha"]}，gamma={sepwinner["gamma_multiplier"]}/201={sepwinner["gamma_multiplier"]/201:.12g}；列标准化及目标标准化均来自训练数据。',
f'- MAE={sepwinner["MAE"]:.6f}，SEP={sepwinner["SEP"]:.6f}，RMSEP={sepwinner["RMSEP"]:.6f}，Bias={sepwinner["Bias"]:.6f}，R²={sepwinner["R2"]:.6f}。',
f'- 相比原排除模型，MAE 降低 {comp(baseline_exclude,sepwinner,"MAE"):.2f}%，SEP 降低 {comp(baseline_exclude,sepwinner,"SEP"):.2f}%。','',
'两个核岭方案的MAE只相差 '+f'{sepwinner["MAE"]-winner["MAE"]:.6f}'+ '，不据此声称有统计显著差异。如果以MAE作为主指标，使用MAE最低方案；如果优先降低SEP/RMSEP，则使用区间核岭方案。这里的最佳仅指已冻结候选在这133条上的观察结果，不是全局最优保证。','',
'## 全部冻结候选结果','',
'| 模型 | 训练数 | CV MAE | 测试 MAE | SEP | Bias | RMSEP |','|---|---:|---:|---:|---:|---:|---:|',
*[f'| {r["display"]} | {r["training_n"]} | {r["CV_MAE"]:.6f} | {r["MAE"]:.6f} | {r["SEP"]:.6f} | {r["Bias"]:.6f} | {r["RMSEP"]:.6f} |' for r in order],'',
'## 划分与评价口径','',
'- 训练候选：光谱编号1–400，按编号关联建模集化学成分.xlsx。保留方案使用400条；排除方案使用390条。',
'- 训练排除编号：'+', '.join(map(str,summary['fold_fit_protocol']['training_selected_ids']))+'。',
'- 测试：401–542中排除460、461、462、463、494、495、496、520、521后剩余133条。不是文件夹“验证集”中的136条，也不是从总样本中另取最后133条。',
'- 目标化学值沿用源表原始数值；非线性模型内部仅作训练均值/标准差线性标准化，预测后转换回原尺度。没有进行对数变换或使用测试Bias校正预测。',
'- e=预测−真实；MAE=mean(abs(e))；SEP=std(e,ddof=1)，即扣除平均误差后以n−1为分母；Bias=mean(e)；RMSEP=sqrt(mean(e²))。','',
'## 搜索方法','',
'1. 两种训练策略使用相同3次重复5折划分，随机种子20260918、20260919、20260920。为与目标133条的评价范围一致，所有候选的主CV评分只计算名单外相同390条训练样本。保留策略仍可在各训练折使用名单内样本，但验证折样本始终不进入拟合。这个CV口径与此前对每种训练总体分别评分有所不同。',
'2. 比较28种全波段预处理：原始、数值一阶导数、SNV、MSC、线性去趋势、SNV加去趋势、SG平滑、不同窗口/阶数SG一阶/二阶导数、SNV或MSC加SG一阶导数。SG一阶窗口7/11/17/25/35、阶数2/3；二阶窗口11/17/25、阶数3。PLS比较1–25成分。MSC参考谱每折仅使用训练样本计算；其他处理为逐样本操作。',
'3. 每种训练策略保留CV较优的3种预处理，比较以950/1050/1150/1250/1350/1450/1550/1650为边界的连续区间；较优2种预处理另比较两个不相邻100单位区间的组合。所有区间通过训练CV选择，并非先看测试效果选区间。',
'4. 对训练CV选出的2个较优特征方案以及最佳全波段方案比较RBF-SVR与RBF核岭。SVR的C=1/10/100，epsilon=0.03/0.1（目标标准化后的尺度）；核岭alpha=0.001/0.01/0.1/1；两者gamma均为0.01/0.1/1/10除以所选变量数，输入每列在训练折内标准化。',
'5. 对每个策略最佳PLS、SVR、核岭，比较两两25%/50%/75%加权与三模型等权融合。权重仅按训练OOF预测选择。',
'6. 每个策略冻结既有基线、全谱预处理PLS、波段PLS、SVR、核岭、融合各1个，共12个。冻结配置写入文件后才计算测试指标，之后没有依据测试结果继续扩展参数搜索。','',
'## 解释与边界','',
'训练CV最低的方案为 '+cvwinner['display']+'（'+cvwinner['id']+'），CV MAE='+f'{cvwinner["CV_MAE"]:.6f}'+ '；它在133条测试集上并非最优。这提示训练内部的小幅排名不能保证在新样本上复现。另存了训练CV优选模型，便于区分“训练选择”与“测试观察优胜”。','',
'本次并非所有优化都有效：较优波段PLS的测试MAE没有超过原排除基线，核岭才带来更清晰的改善。最低MAE模型仍使用全波段；1050–1450区间方案在SEP/RMSEP上更好。因此不能笼统得出“波段越窄越好”或“异常候选必须剔除”。','',
'这133条在此前分析中已多次被查看，本次还用于比较冻结模型并确认观察优胜。因此它们是开发对照集；训练CV搜索误差也存在多重筛选的乐观偏差，不能当作嵌套CV或完全独立外部验证结果。未重新进行独立外层嵌套CV。真正的最终精度仍需未参与模型选择、且有真实化学值的数据确认。当前筛选还限定了样本适用范围，结果不代表被排除类型的预测精度。','',
'## 已完成核验','',
'- 光谱和化学值编号1–542一一对应，无缺失、非有限值或波长网格错位。',
'- PLS实现已与scikit-learn的scale=False结果对照；SG对二次函数导数及MSC对仿射变换的校正检查通过。',
'- 两个12成分既有基线在133条上的预测与上一轮保存结果一致（1e−9容差）。',
'- 12个最终模型保存后重新加载预测一致，指标从保存预测独立复算通过；源数据未改动。','',
'## 输出','',
'- 最佳测试表现模型.joblib：最低MAE方案。',
'- 最低SEP模型.joblib：最低SEP/RMSEP方案。',
'- 训练CV优选模型.joblib：仅按训练交叉验证挑选的融合方案。',
'- 133条预测结果.csv：真实化学值、主模型预测及误差、12个候选的预测。',
'- 最终模型对比.csv：完整指标、波段、预处理和参数。',
'- 训练CV全部候选.jsonl：7212组搜索配置及CV结果。',
'- 优化结果.json：完整配置、指标、输入哈希及依赖版本。',
'- 工作区predict_optimized_spectra.py提供保存模型的批量预测入口；不会自动对其他文件夹执行预测。','',
'实现参考：[SciPy SG滤波](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)；[scikit-learn SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)。实际运行版本记录于优化结果.json。']
(OUT/'模型优化报告.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(('scikit-learn' if k=='sklearn' else k)+'=='+v for k,v in summary['versions'].items())+'\n',encoding='utf-8')
print(json.dumps({'best_MAE':winner['MAE'],'best_MAE_SEP':winner['SEP'],'best_SEP':sepwinner['SEP'],'best_SEP_MAE':sepwinner['MAE'],'MAE_gain_vs_previous_best_percent':comp(baseline_exclude,winner,'MAE'),'best_SEP_gain_vs_previous_best_percent':comp(baseline_exclude,sepwinner,'SEP'),'independent_recalculation':'passed','baseline_reproduction':'passed'},ensure_ascii=False,indent=2))
