#!/usr/bin/env python
"""Fit reference fold classifiers once; apply them unchanged to perturbed features."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_xy(path, participants, cond, low_max, high_min):
    with np.load(path,allow_pickle=False) as d:
        X_all, subject, conditions, blocks = d['X'],d['subject'],d['cond'],d['block']
        meta=json.loads(str(d['metadata']))
    if len(X_all)!=len(subject) or not np.isfinite(X_all).all():
        raise ValueError('Invalid feature rows')
    keep=(conditions==cond) if cond!='both' else np.ones(len(subject),bool)
    X_all,subject,blocks,conditions=X_all[keep],subject[keep],blocks[keep],conditions[keep]
    subs=np.array(sorted(set(subject)))
    if not len(subs):
        raise ValueError('No selected subjects')
    X=np.vstack([X_all[subject==s].mean(axis=0) for s in subs])
    rows=[(str(s),int(b),str(c)) for s,b,c in zip(subject,blocks,conditions)]
    p=pd.read_csv(participants,sep='\t').set_index('participant_id')
    if not p.index.is_unique:
        raise ValueError('Duplicate participant IDs')
    bdi=pd.to_numeric(p.loc[subs,'BDI'],errors='coerce').to_numpy()
    y=np.where(bdi<=low_max,0,np.where(bdi>=high_min,1,-1))
    keep=(y>=0)&np.isfinite(bdi)
    return X[keep],y[keep],subs[keep],meta,rows


def signature(meta):
    return {k:meta[k] for k in ('weights_hash','artifact_hashes','source_hashes',
            'image_sha256','variant','dtype','channels','condition','sfreq','n_times')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['fit','apply'])
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--participants',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--cond',choices=['EO','EC','both'],default='EC')
    p.add_argument('--low-max',type=float,default=7)
    p.add_argument('--high-min',type=float,default=13)
    p.add_argument('--C',type=float,default=1.)
    p.add_argument('--cv-repeats',type=int,default=10)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    versions={'sklearn':sklearn.__version__,'numpy':np.__version__,
              'pandas':pd.__version__,'joblib':joblib.__version__}
    if a.action=='fit':
        if a.low_max>=a.high_min or a.C<=0 or a.cv_repeats<1 or a.folds<2:
            raise ValueError('Invalid labels, C or cross-validation settings')
        X,y,subs,meta,rows=load_xy(a.features,a.participants,a.cond,a.low_max,a.high_min)
        if meta['mode']!='rn':
            raise ValueError('Fit the fixed classifier bundle from the RN reference only')
        counts=np.bincount(y,minlength=2)
        if min(counts)<a.folds:
            raise ValueError(f'Need at least {a.folds} subjects per class for CV; found {counts.tolist()}. Feature variability can still be studied without a probe.')
        if a.bundle.exists():
            raise FileExistsError(a.bundle)
        models=[]
        for repeat in range(a.cv_repeats):
            fold_models=[]
            cv=StratifiedKFold(n_splits=a.folds,shuffle=True,random_state=repeat)
            for train,test in cv.split(X,y):
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=a.C,
                    class_weight='balanced',solver='lbfgs',random_state=0,max_iter=5000))
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    clf.fit(X[train],y[train])
                fold_models.append((test,clf))
            models.append(fold_models)
        bundle={'models':models,'subject':subs,'y':y,'rows':rows,
                'signature':signature(meta),'versions':versions,
                'participants_sha256':file_hash(a.participants),
                'cond':a.cond,'low_max':a.low_max,'high_min':a.high_min,
                'reference_sha256':file_hash(a.features)}
        a.bundle.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(bundle,a.bundle)
    else:
        bundle=joblib.load(a.bundle)
        if versions!=bundle['versions'] or file_hash(a.participants)!=bundle['participants_sha256']:
            raise ValueError('Probe environment or labels changed')
        X,y,subs,meta,rows=load_xy(a.features,a.participants,bundle['cond'],bundle['low_max'],bundle['high_min'])
        if signature(meta)!=bundle['signature'] or rows!=bundle['rows']:
            raise ValueError('Feature controls or subject/block membership changed')
        if not np.array_equal(subs,bundle['subject']) or not np.array_equal(y,bundle['y']):
            raise ValueError('Subject identities or labels changed')
    proba=np.zeros((len(bundle['models']),len(y)))
    for r,folds in enumerate(bundle['models']):
        for test,clf in folds:
            proba[r,test]=clf.predict_proba(X[test])[:,1]
    bacc=np.array([balanced_accuracy_score(y,p>=.5) for p in proba])
    auc=np.array([roc_auc_score(y,p) for p in proba])
    summary={'action':a.action,'n_subjects':len(y),'feature_dim':X.shape[1],
             'balanced_accuracy_mean':float(bacc.mean()),'auc_mean':float(auc.mean()),
             'classifier_bundle_sha256':file_hash(a.bundle),
             'feature_mode':meta['mode'],'perturbation_seed':meta['perturbation_seed']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,subject=subs,y=y,proba=proba,bacc=bacc,auc=auc,
                        summary=json.dumps(summary))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
