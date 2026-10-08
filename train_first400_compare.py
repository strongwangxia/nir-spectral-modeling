"""PLS calibration using IDs 1..400, with untouched test IDs 401..542."""
from pathlib import Path
import os,json,csv,zipfile,hashlib
import xml.etree.ElementTree as ET
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'建模结果'/'前400建模_后142预测'
OUT.mkdir(parents=True,exist_ok=True)
SELECTED={33,166,226,294,350,351,352,374,375,376,460,461,462,463,494,495,496,520,521}
MAXCOMP=15
SEEDS=[20260918,20260919,20260920]

def load_data():
    paths=sorted((ROOT/'建模集').glob('*.csv'),key=lambda p:int(p.stem))
    ids=np.array([int(p.stem) for p in paths])
    assert np.array_equal(ids,np.arange(1,543)), 'Expected IDs 1..542'
    spectra=[np.loadtxt(p,delimiter=',',encoding='utf-8-sig',ndmin=2) for p in paths]
    x=spectra[0][:,0]
    assert all(v.shape==(len(x),2) and np.array_equal(v[:,0],x) for v in spectra)
    assert np.all(np.diff(x)>0)
    X=np.stack([v[:,1] for v in spectra]);assert np.isfinite(X).all()
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(ROOT/'建模集化学成分.xlsx') as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in e.findall('.//s:t',ns)) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        tree=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        labels={}
        for row in tree.findall('.//s:row',ns):
            vals={}
            for c in row:
                v=c.find('s:v',ns)
                if v is None:continue
                value=strings[int(v.text)] if c.get('t')=='s' else v.text
                vals[''.join(ch for ch in c.get('r') if ch.isalpha())]=value
            if 'A' not in vals or 'B' not in vals:continue
            try:key=int(float(vals['A']));value=float(vals['B'])
            except ValueError:
                assert row.get('r')=='1', f'Invalid chemical row {row.get("r")}'
                continue
            assert key not in labels,f'Duplicate chemical sample ID {key}'
            labels[key]=value
    assert set(labels)==set(ids.tolist()),'Chemical IDs do not match spectral IDs'
    y=np.array([labels[int(i)] for i in ids]);assert np.isfinite(y).all()
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes['建模集化学成分.xlsx']=hashlib.sha256((ROOT/'建模集化学成分.xlsx').read_bytes()).hexdigest()
    return ids,x,X,y,hashes

def pls_fit_path(A,b,maxcomp=MAXCOMP):
    xm=A.mean(0);ym=float(b.mean());E=A-xm;f=b-ym
    W=[];P=[];Q=[];betas=[]
    for k in range(maxcomp):
        w=E.T@f;norm=np.linalg.norm(w)
        if norm<1e-13:raise ValueError(f'PLS covariance degenerate at {k+1}')
        w=w/norm;t=E@w;den=float(t@t)
        if den<1e-18:raise ValueError(f'PLS score degenerate at {k+1}')
        p=E.T@t/den;q=float(f@t/den)
        W.append(w);P.append(p);Q.append(q)
        wa=np.column_stack(W);pa=np.column_stack(P)
        betas.append(wa@np.linalg.solve(pa.T@wa,np.array(Q)))
        E-=np.outer(t,p);f-=t*q
    return xm,ym,np.column_stack(betas)

def scores(true,pred):
    errors=pred-true;n=len(errors);bias=float(errors.mean())
    sep=float(np.sqrt(np.sum((errors-bias)**2)/(n-1)))
    rmse=float(np.sqrt(np.mean(errors**2)))
    assert np.isclose(rmse**2,((n-1)/n)*sep**2+bias**2,rtol=1e-12)
    return {'n':n,'SEP':sep,'MAE':float(np.mean(abs(errors))),'Bias':bias,'RMSEP':rmse}

def main():
    ids,x,X,y,hashes=load_data()
    trainbase=np.flatnonzero(ids<=400);test=np.flatnonzero(ids>400)
    flagged=np.isin(ids,list(SELECTED))
    assert len(trainbase)==400 and len(test)==142
    assert flagged[trainbase].sum()==10 and flagged[test].sum()==9
    features={'原始光谱':X,'一阶导数':np.gradient(X,x,axis=1,edge_order=2),'SNV':(X-X.mean(1,keepdims=True))/X.std(1,keepdims=True)}
    assert all(np.isfinite(A).all() for A in features.values())
    # Verify the PLS implementation against a known full-rank linear model.
    rng=np.random.default_rng(5);tx=rng.normal(size=(45,5))+3;ty=tx@np.array([1.,-2.,.5,0.,3.])+7
    mx,my,beta=pls_fit_path(tx,ty,5)
    assert np.max(abs((tx-mx)@beta[:,-1]+my-ty))<1e-8
    # The same fold assignments are reused; exclusion occurs only within the training population.
    partitions=[np.array_split(np.random.default_rng(seed).permutation(trainbase),5) for seed in SEEDS]
    configs=[('保留指定光谱',trainbase,'model_keep.npz'),('排除指定光谱',trainbase[~flagged[trainbase]],'model_exclude.npz')]
    result=[];cvrows=[];predictions={};models={}
    for strategy,train,filename in configs:
        best=None
        for preprocessing,A in features.items():
            squared=np.zeros(MAXCOMP);nvalid=0
            for folds in partitions:
                for basevalid in folds:
                    valid=np.intersect1d(train,basevalid)
                    fit=np.setdiff1d(train,valid)
                    assert not np.intersect1d(fit,test).size
                    xm,ym,betas=pls_fit_path(A[fit],y[fit])
                    pred=(A[valid]-xm)@betas+ym
                    squared+=np.sum((pred-y[valid,None])**2,axis=0)
                    nvalid+=len(valid)
            assert nvalid==len(train)*len(SEEDS)
            rmsecv=np.sqrt(squared/nvalid)
            for j,v in enumerate(rmsecv):cvrows.append([strategy,preprocessing,j+1,float(v)])
            k=int(np.argmin(rmsecv))
            if best is None or rmsecv[k]<best[0]:best=(float(rmsecv[k]),preprocessing,k+1)
        cv_error,preprocessing,ncomp=best
        xm,ym,betas=pls_fit_path(features[preprocessing][train],y[train],ncomp)
        coef=betas[:,-1]
        pred=(features[preprocessing][test]-xm)@coef+ym
        assert np.isfinite(pred).all()
        np.savez(OUT/filename,wavelength=x,x_mean=xm,y_mean=ym,coef=coef,preprocessing=np.array(preprocessing),n_components=ncomp,training_ids=ids[train])
        saved=np.load(OUT/filename)
        assert np.allclose((features[str(saved['preprocessing'])][test]-saved['x_mean'])@saved['coef']+saved['y_mean'],pred,rtol=1e-12,atol=1e-12)
        predictions[strategy]=pred
        metrics=scores(y[test],pred)
        subgroups={label:scores(y[test][m],pred[m]) for label,m in [('指定名单9条',flagged[test]),('其余133条',~flagged[test])]}
        models[strategy]={'training_count':len(train),'training_selected_count':int(flagged[train].sum()),'preprocessing':preprocessing,'components':ncomp,'RMSECV':cv_error,'test_metrics':metrics,'subgroups':subgroups,'file':filename}
        result.append([strategy,len(train),preprocessing,ncomp,cv_error,len(test),metrics['SEP'],metrics['MAE'],metrics['Bias'],metrics['RMSEP']])
        print(strategy,json.dumps(models[strategy],ensure_ascii=False),flush=True)
    with (OUT/'模型评价指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['模型','建模样本数','预处理','PLS成分数','训练集RMSECV','测试样本数','SEP','MAE','Bias','RMSEP']);w.writerows(result)
    with (OUT/'后142条逐样本预测.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['光谱编号','是否属于指定19条','真实化学值','保留模型预测','排除模型预测','保留模型预测减真实','排除模型预测减真实','保留模型绝对误差','排除模型绝对误差'])
        for j,i in enumerate(test):
            p=predictions['保留指定光谱'][j];q=predictions['排除指定光谱'][j]
            w.writerow([int(ids[i]),bool(flagged[i]),y[i],p,q,p-y[i],q-y[i],abs(p-y[i]),abs(q-y[i])])
    with (OUT/'训练集调参记录.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['模型','预处理','PLS成分数','三次五折合并RMSECV']);w.writerows(cvrows)
    metadata={'selected_19':sorted(SELECTED),'selected_in_train':ids[trainbase][flagged[trainbase]].tolist(),'selected_in_test':ids[test][flagged[test]].tolist(),'train_base_ids':[1,400],'test_ids':[401,542],'cv_seeds':SEEDS,'cv_folds':5,'preprocessing_candidates':list(features),'component_candidates':[1,MAXCOMP],'SEP_definition':'sqrt(sum((prediction-reference-Bias)^2)/(n-1)); Bias=mean(prediction-reference)','MAE_definition':'mean(abs(prediction-reference))','models':models,'input_sha256':hashes}
    (OUT/'实验记录.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    make_plot_and_report(ids,test,y,predictions,models,metadata)

def make_plot_and_report(ids,test,y,predictions,models,metadata):
    os.environ['MPLCONFIGDIR']=str(ROOT/'光谱图'/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    special=np.isin(ids[test],list(SELECTED))
    low=min(y[test].min(),*(p.min() for p in predictions.values()))-.15
    high=max(y[test].max(),*(p.max() for p in predictions.values()))+.15
    for ax,(strategy,pred) in zip(axes,predictions.items()):
        ax.scatter(y[test][~special],pred[~special],s=25,color='#276c9b',alpha=.75,label='其余 133 条')
        ax.scatter(y[test][special],pred[special],s=45,marker='^',color='#d04a35',label='名单内 9 条')
        ax.plot([low,high],[low,high],ls='--',lw=1,color='#777777')
        ax.set_xlim(low,high);ax.set_ylim(low,high);ax.set_aspect('equal')
        m=models[strategy];scores_=m['test_metrics']
        ax.set_title(f'{strategy}（建模 {m["training_count"]} 条）\nSEP={scores_["SEP"]:.4f}   MAE={scores_["MAE"]:.4f}',fontproperties=font,fontsize=14)
        ax.set_xlabel('真实化学值',fontproperties=font);ax.set_ylabel('预测化学值',fontproperties=font)
        ax.legend(prop=font);ax.grid(alpha=.2)
    fig.suptitle('1–400 号建模，401–542 号共同测试',fontproperties=font,fontsize=18)
    fig.savefig(OUT/'后142条预测对比.png',dpi=190);plt.close(fig)
    lines=['# 前400条建模、后142条预测：保留与排除指定光谱','',
    '按光谱文件名的数值编号排序，以 1–400 号为训练候选集，401–542 号为共同测试集。通过编号匹配建模集化学成分.xlsx 的真实化学值；542 个编号均唯一且与光谱一一对应。未使用验证集文件夹，也未修改源数据。','',
    '## 划分与排除','',
    '指定19条中，前400条包含10条：'+', '.join(map(str,metadata['selected_in_train']))+'。',
    '后142条包含9条：'+', '.join(map(str,metadata['selected_in_test']))+'。','',
    '保留模型使用400条训练，排除模型仅从训练端移除上述10条，使用390条训练。后142条（含名单内9条）全部保留用于两种模型共同测试。因此“包含/排除19条”在本次固定划分中实际是训练端保留/排除10条，不是400对381。','',
    '## 模型与参数选择','',
    '两套模型均采用PLS1回归。仅在各自训练集内，比较原始光谱、一阶导数、SNV三种预处理与1–15个PLS成分数，以3次重复5折交叉验证合并RMSE最低的组合作为最终配置。两模型复用前400条的折分配，排除方案在相应折内移除名单样本。所有中心化与PLS系数均在每折训练数据中拟合。数值一阶导数以实际波长为横轴，内部采用中心差分、端点二阶单边差分；没有平滑。SNV按每条光谱均值和总体标准差计算。','',
    '预处理及参数锁定后，在各自完整训练集拟合模型，最后对相同142条测试光谱预测。没有按测试误差重选预处理、成分数或名单。源化学值按原数值建模，未作对数等变换。','',
    '| 模型 | 训练数 | 预处理 | PLS成分 | RMSECV | SEP | MAE | Bias | RMSEP |',
    '|---|---:|---|---:|---:|---:|---:|---:|---:|',
    *[f'| {name} | {m["training_count"]} | {m["preprocessing"]} | {m["components"]} | {m["RMSECV"]:.6f} | {m["test_metrics"]["SEP"]:.6f} | {m["test_metrics"]["MAE"]:.6f} | {m["test_metrics"]["Bias"]:.6f} | {m["test_metrics"]["RMSEP"]:.6f} |' for name,m in models.items()],
    '', '## 评价定义','',
    '误差 e_i = 预测值 − 真实值；Bias = mean(e_i)。',
    'MAE = sum(|e_i|)/n。',
    'SEP = sqrt(sum((e_i−Bias)^2)/(n−1))，即去偏差的预测误差标准差；测试集n=142。',
    'RMSEP = sqrt(sum(e_i^2)/n)，额外提供以区分去偏SEP与未去偏均方根误差。所有指标与化学值同单位，原文件未给出明确单位。SEP计算时扣除Bias仅用于指标定义，没有校正或修改任何预测值。','',
    '## 测试集分组诊断','',
    '| 模型 | 测试子集 | n | SEP | MAE | Bias | RMSEP |','|---|---|---:|---:|---:|---:|---:|',
    *[f'| {name} | {group} | {m["n"]} | {m["SEP"]:.6f} | {m["MAE"]:.6f} | {m["Bias"]:.6f} | {m["RMSEP"]:.6f} |' for name,model in models.items() for group,m in model['subgroups'].items()],
    '', '主要结论应以全部142条共同测试结果为准，不能只评价排除名单之外的133条。该名单来自前面查看全部光谱的目选筛查，因此这是指定名单与指定划分下的对照实验，不是从未查看过数据的最终外部验证。','',
    '## 可复现文件','',
    '后142条逐样本预测.csv包含真实值、两套预测及误差；模型评价指标.csv包含主要指标；训练集调参记录.csv包含全部90组配置的训练交叉验证误差。model_keep.npz与model_exclude.npz保存波长、训练编号、预处理名、均值、成分数及回归系数。对新光谱按同一预处理计算z，预测为(z−x_mean)@coef+y_mean。实验记录.json记录输入文件哈希、划分、参数和完整指标。模型保存后已重新读取并复算142条预测，结果一致。']
    (OUT/'模型对比报告.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
