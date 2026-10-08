"""Refit the fixed best exclusion strategy on all 523 eligible spectra."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import joblib
from train_first400_compare import load_data,SELECTED
from spectral_optimization_core import fit_model,predict_model,metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'参赛提交'
OUT.mkdir(exist_ok=True)
ids,x,X,y,hashes=load_data()
eligible=~np.isin(ids,list(SELECTED))
assert eligible.sum()==523
config={'family':'KRR','preprocess':{'name':'SG一阶_w17_p3','base':'raw','kind':'sg','window':17,'poly':3,'deriv':1},'bands':[[1050.,1450.]],'gamma_multiplier':0.1,'alpha':0.001}
# Configuration is locked from the previous exclusion-model comparison.
previous=json.loads((ROOT/'建模结果'/'模型优化_前400训练_后133测试'/'优化结果.json').read_text(encoding='utf-8'))
chosen=next(r for r in previous['finalists'] if r['strategy']=='排除异常候选' and r['family']=='KRR')
for key in config:assert config[key]==chosen[key]
model=fit_model(X[eligible],y[eligible],x,config)
model['training_ids']=ids[eligible]
model['excluded_ids']=np.array(sorted(SELECTED))
model['purpose']='Final submission model fitted on all 523 non-listed calibration samples.'
joblib.dump(model,OUT/'最终模型_523条_区间核岭.joblib')
files=sorted((ROOT/'验证集').glob('*.csv'),key=lambda p:int(p.stem))
validation_ids=np.array([int(p.stem) for p in files])
assert np.array_equal(validation_ids,np.arange(1,137))
spectra=[np.loadtxt(p,delimiter=',',encoding='utf-8-sig',ndmin=2) for p in files]
assert all(a.shape==(len(x),2) and np.array_equal(a[:,0],x) for a in spectra)
V=np.stack([a[:,1] for a in spectra]);assert np.isfinite(V).all()
pred=predict_model(model,V)
assert np.isfinite(pred).all()
assert np.allclose(predict_model(joblib.load(OUT/'最终模型_523条_区间核岭.joblib'),V),pred,rtol=1e-12,atol=1e-12)
with (OUT/'验证集136条预测结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['预测集样本序号','预测值'])
    w.writerows((int(i),format(float(p),'.10f')) for i,p in zip(validation_ids,pred))
fitted=predict_model(model,X[eligible])
meta={'training_n':523,'excluded_ids':sorted(SELECTED),'training_ids':ids[eligible].tolist(),'config':config,'gamma_actual':float(model['gamma']),'wavelength_count':int(model['band_mask'].sum()),'validation_n':136,'validation_prediction_range':[float(pred.min()),float(pred.max())],'training_fit_metrics_not_validation':metrics(y[eligible],fitted),'previous_development_test_133':{k:chosen[k] for k in ['MAE','SEP','RMSEP','Bias','training_n']},'validation_metrics':None,'note':'验证集未提供真实化学值，不能计算其MAE/SEP。既往133条开发对照现已纳入523条最终训练，不再是最终模型独立测试集。','training_input_hashes':hashes,'validation_input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(OUT/'最终模型与预测记录.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:meta[k] for k in ['training_n','excluded_ids','gamma_actual','wavelength_count','validation_n','validation_prediction_range']},ensure_ascii=False,indent=2))
