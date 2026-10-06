#!/usr/bin/env python
"""Compare paired out-of-fold probabilities from a frozen classifier bundle."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits


def read(path):
    with np.load(path,allow_pickle=False) as d:
        return {k:d[k].copy() for k in d.files}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if len(a.runs)<2:
        p.error('Need at least two perturbed probe outputs')
    if a.out.exists():
        raise FileExistsError(a.out)
    ref=read(a.reference); meta=json.loads(str(ref['summary']))
    if meta['feature_mode']!='rn':
        raise ValueError('Reference probe must use RN features')
    values,seeds=[],[]
    for path in a.runs:
        d=read(path); m=json.loads(str(d['summary']))
        if (m['classifier_bundle_sha256']!=meta['classifier_bundle_sha256'] or
            m['feature_mode']!='sr' or not np.array_equal(d['subject'],ref['subject']) or
            not np.array_equal(d['y'],ref['y']) or d['proba'].shape!=ref['proba'].shape):
            raise ValueError('Classifier, subjects, labels, or folds differ')
        values.append(d['proba']); seeds.append(m['perturbation_seed'])
    if len(set(seeds))!=len(values):
        raise ValueError('Duplicate perturbation seeds')
    arr=np.stack(values); reference=ref['proba']
    mean_probs=arr.mean(axis=1); mean_ref=reference.mean(axis=0)
    flips=(arr>=.5)!=(reference>=.5)
    a.out.mkdir(parents=True)
    with (a.out/'subject_probability_variability.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['subject','label','reference_mean_probability',
            'mean_probability_std_across_runs','max_paired_probability_difference',
            'paired_decision_flips'])
        for i,s in enumerate(ref['subject']):
            w.writerow([s,int(ref['y'][i]),float(mean_ref[i]),
                float(mean_probs[:,i].std(ddof=1)),float(np.abs(arr[:,:,i]-reference[:,i]).max()),
                int(flips[:,:,i].sum())])
    summary={'n_runs':len(values),'n_subjects':len(ref['subject']),
             'maximum_paired_probability_difference':float(np.abs(arr-reference).max()),
             'paired_decision_flips':int(flips.sum()),
             'classifier_bundle_sha256':meta['classifier_bundle_sha256']}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
