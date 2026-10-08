"""Preprocessing and serializable PLS/SVR/KRR prediction helpers."""
import numpy as np
from scipy.signal import savgol_filter, detrend
from scipy.linalg import cho_factor, cho_solve
from sklearn.svm import SVR

def snv(X):
    scale=X.std(axis=1,keepdims=True)
    if np.any(scale<1e-14):raise ValueError('Constant spectrum')
    return (X-X.mean(axis=1,keepdims=True))/scale

def preprocess(X,x,spec,reference=None):
    A=np.asarray(X,dtype=float).copy()
    base=spec.get('base','raw')
    if base.startswith('snv'):A=snv(A)
    if base=='msc':
        if reference is None:reference=A.mean(axis=0)
        r=reference-reference.mean();den=r@r
        slopes=(A-A.mean(axis=1,keepdims=True))@r/den
        if np.any(abs(slopes)<1e-12):raise ValueError('Degenerate MSC slope')
        intercepts=A.mean(axis=1)-slopes*reference.mean()
        A=(A-intercepts[:,None])/slopes[:,None]
    if 'detrend' in base:A=detrend(A,axis=1,type='linear')
    kind=spec.get('kind','none')
    if kind=='gradient':A=np.gradient(A,x,axis=1,edge_order=2)
    if kind=='sg':
        step=float(np.diff(x)[0]);assert np.allclose(np.diff(x),step)
        A=savgol_filter(A,spec['window'],spec['poly'],deriv=spec['deriv'],delta=step,axis=1,mode='interp')
    if not np.isfinite(A).all():raise ValueError('Nonfinite preprocessing')
    return A,reference

def band_mask(x,bands):
    mask=np.zeros(len(x),dtype=bool)
    for low,high in bands:mask|=(x>=low)&(x<=high)
    if mask.sum()<5:raise ValueError('Too few spectral variables')
    return mask

def pls_path(A,y,k):
    xm=A.mean(0);ym=float(y.mean())
    E=A-xm;scale=float(np.sqrt(np.mean(E*E)))
    if scale<1e-16:raise ValueError('No spectral variance')
    E=E/scale;f=y-ym
    W=[];P=[];Q=[];coefs=[]
    for _ in range(k):
        w=E.T@f;norm=np.linalg.norm(w)
        if norm<1e-13:raise ValueError('PLS exhausted covariance')
        w/=norm;t=E@w;den=float(t@t)
        if den<1e-18:raise ValueError('PLS degenerate score')
        p=E.T@t/den;q=float(f@t/den)
        W.append(w);P.append(p);Q.append(q)
        wa=np.column_stack(W);pa=np.column_stack(P)
        coefs.append((wa@np.linalg.solve(pa.T@wa,np.array(Q)))/scale)
        E-=np.outer(t,p);f-=t*q
    return xm,ym,np.column_stack(coefs)

def squared_dist(A,B):
    return np.maximum((A*A).sum(1)[:,None]+(B*B).sum(1)[None,:]-2*A@B.T,0)

def nonlinear_prepare(A,y,B):
    xm=A.mean(0);xs=A.std(0);xs=np.where(xs<1e-12,1,xs)
    ym=float(y.mean());ys=float(y.std())
    return (A-xm)/xs,(B-xm)/xs,(y-ym)/ys,{'xm':xm,'xs':xs,'ym':ym,'ys':ys}

def fit_model(X,y,x,config):
    spec=config['preprocess'];bands=config['bands']
    A,reference=preprocess(X,x,spec)
    bm=band_mask(x,bands);A=A[:,bm]
    model={'config':config,'wavelength':x.copy(),'reference':reference,'band_mask':bm}
    family=config['family']
    if family=='PLS':
        xm,ym,B=pls_path(A,y,config['components'])
        model.update({'xm':xm,'ym':ym,'coef':B[:,-1]})
    else:
        Z,_,z_y,scaler=nonlinear_prepare(A,y,A)
        gamma=config['gamma_multiplier']/A.shape[1]
        K=np.exp(-gamma*squared_dist(Z,Z))
        if family=='SVR':
            estimator=SVR(kernel='precomputed',C=config['C'],epsilon=config['epsilon'],tol=1e-5,cache_size=256)
            estimator.fit(K,z_y);model['estimator']=estimator
        elif family=='KRR':
            model['dual']=cho_solve(cho_factor(K+config['alpha']*np.eye(len(K)),lower=True,check_finite=False),z_y,check_finite=False)
        else:raise ValueError(family)
        model.update({'scaler':scaler,'Z':Z,'gamma':gamma})
    return model

def predict_model(model,X):
    if 'members' in model:
        return sum(w*predict_model(m,X) for w,m in zip(model['weights'],model['members']))
    cfg=model['config'];A,_=preprocess(X,model['wavelength'],cfg['preprocess'],model['reference'])
    A=A[:,model['band_mask']]
    if cfg['family']=='PLS':return (A-model['xm'])@model['coef']+model['ym']
    s=model['scaler'];Z=(A-s['xm'])/s['xs']
    K=np.exp(-model['gamma']*squared_dist(Z,model['Z']))
    pred=model['estimator'].predict(K) if cfg['family']=='SVR' else K@model['dual']
    return pred*s['ys']+s['ym']

def metrics(true,pred):
    e=np.asarray(pred)-np.asarray(true);bias=float(e.mean())
    return {'n':len(e),'MAE':float(np.mean(abs(e))),'SEP':float(np.std(e,ddof=1)),'Bias':bias,'RMSEP':float(np.sqrt(np.mean(e*e))),'R2':float(1-np.sum(e*e)/np.sum((true-np.mean(true))**2))}

def verify_core():
    from sklearn.cross_decomposition import PLSRegression
    rng=np.random.default_rng(610);A=rng.normal(size=(80,20))+2;y=rng.normal(size=80);B=rng.normal(size=(12,20))
    xm,ym,betas=pls_path(A,y,8)
    for k in [1,4,8]:
        ref=PLSRegression(n_components=k,scale=False).fit(A,y).predict(B).ravel()
        assert np.allclose((B-xm)@betas[:,k-1]+ym,ref,rtol=1e-8,atol=1e-8)
    x=np.arange(950,1652,2,dtype=float);polynomial=(x/1000)**2
    d,_=preprocess(polynomial[None,:],x,{'base':'raw','kind':'sg','window':11,'poly':2,'deriv':1})
    assert np.allclose(d[0],2*x/1e6,atol=1e-12)
    target=np.sin(x/100)+2;corrupted=np.stack([target*.7+2,target*1.4-.3])
    corrected,_=preprocess(corrupted,x,{'base':'msc'},target)
    assert np.allclose(corrected,np.stack([target,target]),atol=1e-12)
