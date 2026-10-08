"""Apply an optimized saved model without retraining.

Example: python predict_optimized_spectra.py --folder 验证集 --output predictions.csv
Use the same Python dependency versions recorded in 优化结果.json.
"""
from pathlib import Path
import argparse,csv
import numpy as np
import joblib
from spectral_optimization_core import predict_model

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=root/'建模结果'/'模型优化_前400训练_后133测试'/'最佳测试表现模型.joblib')
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model=joblib.load(args.model)
    reference=model
    while 'members' in reference:reference=reference['members'][0]
    x=reference['wavelength']
    paths=sorted(args.folder.glob('*.csv'),key=lambda p:int(p.stem))
    if not paths:raise ValueError('No input spectra found')
    rows=[]
    for path in paths:
        a=np.loadtxt(path,delimiter=',',encoding='utf-8-sig',ndmin=2)
        if a.shape!=(len(x),2) or not np.array_equal(a[:,0],x) or not np.isfinite(a).all():
            raise ValueError(f'Invalid spectrum or wavelength mismatch: {path}')
        rows.append(a[:,1])
    pred=predict_model(model,np.stack(rows))
    assert np.isfinite(pred).all()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['光谱编号','预测化学值'])
        w.writerows((p.stem,float(v)) for p,v in zip(paths,pred))
    print(f'Saved {len(paths)} predictions to {args.output}')

if __name__=='__main__':main()
