"""Training-only staged search, followed by one fixed 133-sample comparison."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,csv,itertools,time,hashlib,functools
import numpy as np
import joblib,scipy,sklearn
from scipy.linalg import cho_factor,cho_solve
from sklearn.svm import SVR
from spectral_optimization_core import preprocess,band_mask,pls_path,squared_dist,nonlinear_prepare,fit_model,predict_model,metrics,verify_core
from train_first400_compare import load_data,SELECTED

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'建模结果'/'模型优化_前400训练_后133测试'
OUT.mkdir(parents=True,exist_ok=True)
MAXCOMP=25
SEEDS=[20260918,20260919,20260920]
START=time.time()
records=[];counter=0
def log(message):
    line=f'[{time.time()-START:.1f}s] {message}'
    print(line,flush=True)
    with (OUT/'运行日志.txt').open('a',encoding='utf-8') as f:f.write(line+'\n')

def preprocessing_specs():
    specs=[]
    def add(name,base='raw',kind='none',**kwargs):specs.append({'name':name,'base':base,'kind':kind,**kwargs})
    add('原始');add('数值一阶导数',kind='gradient');add('SNV',base='snv');add('MSC',base='msc');add('线性去趋势',base='detrend');add('SNV加去趋势',base='snv_detrend')
    for w in [9,17,25]:add(f'SG平滑_w{w}_p2',kind='sg',window=w,poly=2,deriv=0)
    for w,p in itertools.product([7,11,17,25,35],[2,3]):add(f'SG一阶_w{w}_p{p}',kind='sg',window=w,poly=p,deriv=1)
    for w in [11,17,25]:add(f'SG二阶_w{w}_p3',kind='sg',window=w,poly=3,deriv=2)
    for base in ['snv','msc']:
        for w in [11,17,25]:add(f'{base.upper()}加SG一阶_w{w}_p2',base=base,kind='sg',window=w,poly=2,deriv=1)
    return specs

def main():
    global counter
    verify_core();log('Verified PLS against sklearn, SG polynomial derivative, and MSC affine correction.')
    ids,x,ALLX,ALLY,hashes=load_data()
    trainbase=ids<=400;flag=np.isin(ids,list(SELECTED));test=(ids>400)&~flag
    assert trainbase.sum()==400 and test.sum()==133 and flag[trainbase].sum()==10
    X=ALLX[trainbase];y=ALLY[trainbase];normal=~flag[trainbase]
    common=np.flatnonzero(normal)
    strategies={'保留异常候选':np.arange(400),'排除异常候选':common}
    partitions=[np.array_split(np.random.default_rng(seed).permutation(400),5) for seed in SEEDS]
    splits={name:[(r,f,np.setdiff1d(train,valid),np.intersect1d(train,valid)) for r,folds in enumerate(partitions) for f,valid in enumerate(folds)] for name,train in strategies.items()}
    for name,sp in splits.items():
        assert all(set(fit).isdisjoint(valid) and np.max(fit)<400 for _,_,fit,valid in sp)
    plan={'training_ids':[1,400],'training_selected_ids':ids[trainbase&flag].tolist(),'excluded_test_ids':ids[(ids>400)&flag].tolist(),'test_ids':ids[test].tolist(),'primary_selection_metric':'MAE on same 390 non-listed training samples, pooled across 3x5-fold OOF','seeds':SEEDS,'max_pls_components':MAXCOMP,'preprocessing_candidates':preprocessing_specs(),'band_boundaries':[950,1050,1150,1250,1350,1450,1550,1650],'protocol':'Select preprocessing, bands and nonlinear parameters on training CV only; freeze family finalists before reading test y for evaluation. Test winner is best observed among frozen finalists, not unbiased external validation.'}
    (OUT/'预先固定搜索方案.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    nonmsc={}
    @functools.lru_cache(maxsize=64)
    def transformed(key,fit_tuple):
        spec=json.loads(key)
        if spec['base']!='msc':
            if key not in nonmsc:nonmsc[key]=preprocess(X,x,spec)[0]
            return nonmsc[key]
        reference=X[list(fit_tuple)].mean(0)
        return preprocess(X,x,spec,reference)[0]

    def record(config,pred):
        global counter
        e=pred[:,normal]-y[None,normal]
        assert np.isfinite(e).all()
        counter+=1
        r={'id':f'C{counter:05d}',**config,'CV_MAE':float(np.mean(abs(e))),'CV_RMSE':float(np.sqrt(np.mean(e*e))),'CV_MAE_by_repeat':np.mean(abs(e),axis=1).tolist()}
        records.append(r)
        return r

    def cv_pls(strategy,spec,bands,stage):
        bm=band_mask(x,bands);kmax=min(MAXCOMP,int(bm.sum())-2)
        pred=np.full((3,400,kmax),np.nan)
        key=json.dumps(spec,sort_keys=True)
        for rep,fold,fit,valid in splits[strategy]:
            A=transformed(key,tuple(fit))[:,bm]
            xm,ym,betas=pls_path(A[fit],y[fit],kmax)
            pred[rep,valid]=(A[valid]-xm)@betas+ym
        candidates=[]
        for k in range(kmax):
            cfg={'strategy':strategy,'stage':stage,'family':'PLS','preprocess':spec,'bands':bands,'components':k+1}
            candidates.append(record(cfg,pred[:,:,k]))
        best=min(candidates,key=lambda r:(r['CV_MAE'],r['components']))
        return {**best,'_pred':pred[:,:,best['components']-1].copy()}

    full={};bandresults={};baseline={};nonlinear={};finalists=[]
    fullband=[[float(x[0]),float(x[-1])]]
    specs=preprocessing_specs()
    for strategy in strategies:
        full[strategy]=[]
        for j,spec in enumerate(specs):
            r=cv_pls(strategy,spec,fullband,'全波段预处理PLS')
            full[strategy].append(r)
            if spec['name']=='数值一阶导数':
                cfg={'strategy':strategy,'stage':'原始基线','family':'PLS','preprocess':spec,'bands':fullband,'components':12}
                preds=np.full((3,400),np.nan)
                key=json.dumps(spec,sort_keys=True)
                for rep,fold,fit,valid in splits[strategy]:
                    A=transformed(key,tuple(fit));xm,ym,B=pls_path(A[fit],y[fit],12)
                    preds[rep,valid]=(A[valid]-xm)@B[:,-1]+ym
                baseline[strategy]={**record(cfg,preds),'_pred':preds}
            if (j+1)%7==0:log(f'{strategy}: preprocessing {j+1}/{len(specs)}, best CV MAE={min(v["CV_MAE"] for v in full[strategy]):.6f}')
        best=min(full[strategy],key=lambda r:r['CV_MAE'])
        log(f'{strategy}: best full-spectrum {best["preprocess"]["name"]}, PLS {best["components"]}, CV MAE={best["CV_MAE"]:.6f}')
    write_records()

    boundaries=[950,1050,1150,1250,1350,1450,1550,1650]
    windows=[[[float(a),float(b)]] for a,b in itertools.combinations(boundaries,2) if not(a==950 and b==1650)]
    for strategy in strategies:
        tops=sorted(full[strategy],key=lambda r:r['CV_MAE'])[:3]
        bandresults[strategy]=[]
        for rank,pre in enumerate(tops):
            for bands in windows:bandresults[strategy].append(cv_pls(strategy,pre['preprocess'],bands,'连续区间PLS'))
            log(f'{strategy}: contiguous windows {rank+1}/3; best CV MAE={min(r["CV_MAE"] for r in bandresults[strategy]):.6f}')
        blocks=[[float(a),float(b)] for a,b in zip(boundaries[:-1],boundaries[1:])]
        for rank,pre in enumerate(tops[:2]):
            for ia,ib in itertools.combinations(range(7),2):
                if ib==ia+1:continue # Already evaluated as one contiguous interval.
                bandresults[strategy].append(cv_pls(strategy,pre['preprocess'],[blocks[ia],blocks[ib]],'双区间PLS'))
            log(f'{strategy}: two-band combinations {rank+1}/2 completed')
    write_records()

    for strategy in strategies:
        pool=full[strategy]+bandresults[strategy]
        # Two best feature representations plus best full-band representation, all chosen via CV.
        sortedpool=sorted(pool,key=lambda r:r['CV_MAE'])
        feature_tops=[];seen=set()
        for r in sortedpool+[min(full[strategy],key=lambda v:v['CV_MAE'])]:
            key=json.dumps([r['preprocess'],r['bands']],sort_keys=True)
            if key not in seen:feature_tops.append(r);seen.add(key)
            if len(feature_tops)==2:break
        fb=min(full[strategy],key=lambda v:v['CV_MAE'])
        key=json.dumps([fb['preprocess'],fb['bands']],sort_keys=True)
        if key not in seen:feature_tops.append(fb)
        nonlinear[strategy]=[]
        for fi,feature in enumerate(feature_tops):
            spec=feature['preprocess'];bands=feature['bands'];bm=band_mask(x,bands);key=json.dumps(spec,sort_keys=True)
            grids=[]
            for gamma,C,epsilon in itertools.product([.01,.1,1,10],[1.,10.,100.],[.03,.1]):
                grids.append({'family':'SVR','gamma_multiplier':gamma,'C':C,'epsilon':epsilon})
            for gamma,alpha in itertools.product([.01,.1,1,10],[.001,.01,.1,1.]):
                grids.append({'family':'KRR','gamma_multiplier':gamma,'alpha':alpha})
            preds=[np.full((3,400),np.nan) for _ in grids]
            for rep,fold,fit,valid in splits[strategy]:
                A=transformed(key,tuple(fit))[:,bm]
                Z,Zv,zy,scaler=nonlinear_prepare(A[fit],y[fit],A[valid])
                dist=squared_dist(Z,Z);vdist=squared_dist(Zv,Z)
                kernels={g:(np.exp(-g/A.shape[1]*dist),np.exp(-g/A.shape[1]*vdist)) for g in [.01,.1,1,10]}
                for gi,g in enumerate(grids):
                    K,Kv=kernels[g['gamma_multiplier']]
                    if g['family']=='SVR':
                        estimator=SVR(kernel='precomputed',C=g['C'],epsilon=g['epsilon'],tol=1e-5,cache_size=128)
                        yp=estimator.fit(K,zy).predict(Kv)
                    else:
                        dual=cho_solve(cho_factor(K+g['alpha']*np.eye(len(K)),lower=True,check_finite=False),zy,check_finite=False)
                        yp=Kv@dual
                    preds[gi][rep,valid]=yp*scaler['ys']+scaler['ym']
            for g,p in zip(grids,preds):
                cfg={'strategy':strategy,'stage':'非线性模型','preprocess':spec,'bands':bands,**g}
                nonlinear[strategy].append({**record(cfg,p),'_pred':p})
            log(f'{strategy}: nonlinear feature {fi+1}/{len(feature_tops)} completed; best CV MAE={min(r["CV_MAE"] for r in nonlinear[strategy]):.6f}')
    write_records()

    for strategy in strategies:
        bestfull=min(full[strategy],key=lambda r:r['CV_MAE'])
        bestband=min(bandresults[strategy],key=lambda r:r['CV_MAE'])
        bestpls=min([bestfull,bestband],key=lambda r:r['CV_MAE'])
        bestsvr=min([r for r in nonlinear[strategy] if r['family']=='SVR'],key=lambda r:r['CV_MAE'])
        bestkrr=min([r for r in nonlinear[strategy] if r['family']=='KRR'],key=lambda r:r['CV_MAE'])
        components=[bestpls,bestsvr,bestkrr];blends=[]
        for ra,rb in itertools.combinations(components,2):
            for weight in [.25,.5,.75]:
                pred=weight*ra['_pred']+(1-weight)*rb['_pred']
                cfg={'strategy':strategy,'stage':'融合','family':'Blend','member_ids':[ra['id'],rb['id']],'weights':[weight,1-weight]}
                blends.append({**record(cfg,pred),'_pred':pred,'_members':[ra,rb]})
        pred=sum(r['_pred'] for r in components)/3
        cfg={'strategy':strategy,'stage':'融合','family':'Blend','member_ids':[r['id'] for r in components],'weights':[1/3]*3}
        blends.append({**record(cfg,pred),'_pred':pred,'_members':components})
        bestblend=min(blends,key=lambda r:r['CV_MAE'])
        for label,r in [('既有基线',baseline[strategy]),('预处理PLS',bestfull),('波段PLS',bestband),('SVR',bestsvr),('核岭回归',bestkrr),('融合',bestblend)]:
            finalists.append({**r,'display':strategy+' / '+label})
    write_records()
    configs=[strip(r) for r in finalists]
    # Freeze all configurations before evaluating held-out predictions; no subsequent search.
    (OUT/'冻结的最终对照模型.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2),encoding='utf-8')
    training_winner=min(finalists,key=lambda r:r['CV_MAE'])
    log(f'FROZEN {len(finalists)} finalists from {len(records)} configurations. Training-CV winner: {training_winner["display"]}, {training_winner["id"]}')
    fitted={}
    def fit_result(r):
        if r['id'] in fitted:return fitted[r['id']]
        if r['family']=='Blend':
            model={'config':strip(r),'members':[fit_result(m) for m in r['_members']],'weights':r['weights']}
        else:
            train=strategies[r['strategy']]
            model=fit_model(X[train],y[train],x,strip(r))
            model['training_ids']=ids[train].copy()
        fitted[r['id']]=model
        return model
    evaluation=[];preds={}
    for r in finalists:
        model=fit_result(r);pred=predict_model(model,ALLX[test])
        score=metrics(ALLY[test],pred)
        assert np.isfinite(pred).all()
        row={**strip(r),'training_n':len(strategies[r['strategy']]),**score}
        evaluation.append(row);preds[r['id']]=pred
        joblib.dump(model,OUT/(r['id']+'_model.joblib'))
        reloaded=joblib.load(OUT/(r['id']+'_model.joblib'))
        assert np.allclose(predict_model(reloaded,ALLX[test]),pred,rtol=1e-12,atol=1e-12)
        log(f'TEST {r["display"]}: MAE={score["MAE"]:.6f}, SEP={score["SEP"]:.6f}')
    winner=min(evaluation,key=lambda r:(r['MAE'],r['SEP']))
    joblib.dump(fitted[winner['id']],OUT/'最佳测试表现模型.joblib')
    joblib.dump(fitted[training_winner['id']],OUT/'训练CV优选模型.joblib')
    with (OUT/'133条预测结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['光谱编号','真实化学值','最佳模型预测','最佳模型误差','最佳模型绝对误差']+[r['display']+'预测' for r in finalists]
        w=csv.writer(f);w.writerow(fields)
        for j,i in enumerate(np.flatnonzero(test)):
            p=preds[winner['id']][j]
            w.writerow([int(ids[i]),float(ALLY[i]),float(p),float(p-ALLY[i]),float(abs(p-ALLY[i])),*[float(preds[r['id']][j]) for r in finalists]])
    with (OUT/'最终模型对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['模型','候选ID','训练样本数','算法','预处理','波段','参数','CV_MAE','CV_RMSE','测试样本数','MAE','SEP','Bias','RMSEP','R2']
        w=csv.writer(f);w.writerow(fields)
        for r in sorted(evaluation,key=lambda r:r['MAE']):
            params={k:r[k] for k in ['components','C','epsilon','gamma_multiplier','alpha','member_ids','weights'] if k in r}
            w.writerow([r['display'],r['id'],r['training_n'],r['family'],r.get('preprocess',{}).get('name','见成员模型'),json.dumps(r.get('bands',[])),json.dumps(params),r['CV_MAE'],r['CV_RMSE'],r['n'],r['MAE'],r['SEP'],r['Bias'],r['RMSEP'],r['R2']])
    summary={'winner_test':winner,'winner_cv':strip(training_winner),'finalists':evaluation,'candidate_count':len(records),'fold_fit_protocol':plan,'input_sha256':hashes,'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},'elapsed_seconds':time.time()-START}
    (OUT/'优化结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez(OUT/'对照绘图数据.npz',ids=ids[test],y=ALLY[test],predictions=np.stack([preds[r['id']] for r in finalists]),labels=np.array([r['display'] for r in finalists]),candidate_ids=np.array([r['id'] for r in finalists]),x=x,train_y=y,train_X=X)
    log('DONE '+json.dumps({'winner':winner['display'],'MAE':winner['MAE'],'SEP':winner['SEP'],'id':winner['id'],'n_candidates':len(records)},ensure_ascii=False))

def strip(r):return {k:v for k,v in r.items() if not k.startswith('_')}
def write_records():
    with (OUT/'训练CV全部候选.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(strip(r),ensure_ascii=False)+'\n')

if __name__=='__main__':main()
