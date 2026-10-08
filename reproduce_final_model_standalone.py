"""Standalone reproduction: read original data, refit fixed KRR, predict 136 spectra.
Usage: python reproduce_final_model_standalone.py --data-dir . --output-dir reproduction
No project-module imports or prior optimization outputs are required.
"""
from pathlib import Path
import argparse, csv, json, zipfile
import xml.etree.ElementTree as ET
import numpy as np
from scipy.signal import savgol_filter
from sklearn.kernel_ridge import KernelRidge
import joblib

EXCLUDED = [33,166,226,294,350,351,352,374,375,376,460,461,462,463,494,495,496,520,521]

def read_spectra(folder, count):
    paths = sorted(folder.glob('*.csv'), key=lambda p: int(p.stem))
    ids = np.array([int(p.stem) for p in paths])
    if not np.array_equal(ids, np.arange(1, count+1)):
        raise ValueError(f'Unexpected sample IDs: {folder}')
    arrays = [np.loadtxt(p, delimiter=',', encoding='utf-8-sig', ndmin=2) for p in paths]
    x = arrays[0][:, 0]
    if not np.array_equal(x, np.arange(950, 1652, 2)):
        raise ValueError('Unexpected spectral grid')
    if not all(a.shape == (351,2) and np.array_equal(a[:,0],x) for a in arrays):
        raise ValueError('Spectral grids differ')
    X = np.stack([a[:,1] for a in arrays])
    if not np.isfinite(X).all(): raise ValueError('Nonfinite absorbance')
    return ids, x, X

def read_chemistry(path, ids):
    ns = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    labels = {}
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in e.findall('.//s:t',ns))
                       for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row',ns):
            cells = {}
            for c in row:
                v = c.find('s:v',ns)
                if v is None: continue
                cells[''.join(ch for ch in c.get('r') if ch.isalpha())] = strings[int(v.text)] if c.get('t')=='s' else v.text
            if 'A' not in cells or 'B' not in cells: continue
            try: key, value = int(float(cells['A'])), float(cells['B'])
            except ValueError:
                if row.get('r') != '1': raise
                continue
            if key in labels: raise ValueError('Duplicate chemical ID')
            labels[key] = value
    if set(labels) != set(ids.tolist()): raise ValueError('Chemical IDs mismatch')
    y = np.array([labels[int(i)] for i in ids])
    if not np.isfinite(y).all(): raise ValueError('Nonfinite chemistry')
    return y

def features(X, x):
    # Smooth and differentiate the FULL spectrum before selecting the interval.
    derivative = savgol_filter(X, 17, 3, deriv=1, delta=2.0, axis=1, mode='interp')
    return derivative[:, (x >= 1050) & (x <= 1450)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('reproduction'))
    parser.add_argument('--compare-csv', type=Path, help='Optional prior prediction CSV for numerical verification')
    args = parser.parse_args()
    ids, x, X = read_spectra(args.data_dir/'建模集',542)
    y = read_chemistry(args.data_dir/'建模集化学成分.xlsx',ids)
    vid, vx, V = read_spectra(args.data_dir/'验证集',136)
    if not np.array_equal(x,vx): raise ValueError('Validation grid mismatch')
    keep = ~np.isin(ids,EXCLUDED)
    A, B = features(X[keep],x), features(V,x)
    xm, xs = A.mean(0), A.std(0,ddof=0)
    xs = np.where(xs < 1e-12,1.0,xs)
    ym, ys = float(y[keep].mean()), float(y[keep].std(ddof=0))
    estimator = KernelRidge(alpha=0.001,kernel='rbf',gamma=0.1/A.shape[1])
    estimator.fit((A-xm)/xs,(y[keep]-ym)/ys)
    pred = estimator.predict((B-xm)/xs)*ys+ym
    assert A.shape==(523,201) and pred.shape==(136,) and np.isfinite(pred).all()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'验证集136条预测结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['预测集样本序号','预测值'])
        writer.writerows((int(i),f'{p:.10f}') for i,p in zip(vid,pred))
    joblib.dump({'estimator':estimator,'xm':xm,'xs':xs,'ym':ym,'ys':ys,'wavelength':x,
                 'training_ids':ids[keep],'excluded_ids':EXCLUDED,
                 'preprocess':{'window':17,'poly':3,'deriv':1,'delta':2,'mode':'interp','band':[1050,1450]}},
                args.output_dir/'复现模型.joblib')
    record={'training_n':int(keep.sum()),'feature_n':A.shape[1],'prediction_n':len(pred),
            'alpha':0.001,'gamma':0.1/A.shape[1],'prediction_min':float(pred.min()),'prediction_max':float(pred.max())}
    if args.compare_csv:
        previous=np.loadtxt(args.compare_csv,delimiter=',',encoding='utf-8-sig',skiprows=1)
        assert np.array_equal(previous[:,0],vid)
        record['max_abs_difference']=float(np.max(np.abs(previous[:,1]-pred)))
        assert record['max_abs_difference']<1e-8,record
    (args.output_dir/'复现核验.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
