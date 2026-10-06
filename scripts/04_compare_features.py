#!/usr/bin/env python
"""Check all controls, reference repeatability and report feature variation."""
import argparse
import csv
import ctypes
import json
from pathlib import Path
import numpy as np


def load(path):
    with np.load(path,allow_pickle=False) as d:
        return d['X'].copy(),[(str(s),int(b),str(c)) for s,b,c in zip(d['subject'],d['block'],d['cond'])],json.loads(str(d['metadata']))


def compare(base,repetitions):
    ref,rows,meta=load(base/'reference.npz')
    repeat,other,second=load(base/'reference_repeat.npz')
    if meta['mode']!='rn' or second['mode']!='rn':
        raise RuntimeError('Both reference executions must use RN')
    if rows!=other or not np.array_equal(ref,repeat):
        raise RuntimeError('Ordinary-rounding references differ; resolve other variability first')
    data,seeds=[],[]
    controls=('weights_hash','artifact_hashes','source_hashes','image_sha256','variant',
              'dtype','channels','sfreq','n_times','condition','batch_size','host','cpu','torch_build')
    for name in ['reference_repeat']+[f'sr_rep-{i}' for i in range(1,repetitions+1)]:
        x,r,m=load(base/(name+'.npz'))
        if rows!=r or any(m[k]!=meta[k] for k in controls):
            raise RuntimeError(f'Controls differ: {name}')
        if name!='reference_repeat':
            if m['mode']!='sr':
                raise RuntimeError('Perturbed run is not SR')
            data.append(x); seeds.append(m['perturbation_seed'])
    if len(set(seeds))!=repetitions:
        raise RuntimeError('Perturbation seeds must differ')
    arr=np.stack(data)
    ranges=np.ptp(arr,axis=0)
    diff=np.max(np.abs(arr-ref),axis=0)
    sd=arr.std(axis=0,ddof=1)
    with (base/'block_variability.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['subject','block','condition','max_feature_range','max_abs_difference_from_reference','feature_rms_difference','mean_feature_std'])
        for i,row in enumerate(rows):
            w.writerow([*row,float(ranges[i].max()),float(diff[i].max()),
                float(np.sqrt(np.mean((arr[:,i]-ref[i])**2))),float(sd[i].mean())])
    with (base/'subject_variability.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['subject','condition','n_blocks','max_feature_range','feature_rms_difference'])
        for subject in sorted(set(r[0] for r in rows)):
            for condition in sorted(set(r[2] for r in rows if r[0]==subject)):
                idx=[i for i,r in enumerate(rows) if r[0]==subject and r[2]==condition]
                values=arr[:,idx].mean(axis=1); baseline=ref[idx].mean(axis=0)
                w.writerow([subject,condition,len(idx),float(np.ptp(values,axis=0).max()),
                            float(np.sqrt(np.mean((values-baseline)**2)))])
    summary={'n_runs':repetitions,'n_subjects':len(set(r[0] for r in rows)),
             'n_blocks':len(rows),'n_features':ref.shape[1],
             'reference_repeatable':True,'all_controls_identical':True,
             'perturbation_seeds':seeds,'maximum_feature_range':float(ranges.max()),
             'maximum_absolute_difference_from_reference':float(diff.max()),
             'feature_rms_difference':float(np.sqrt(np.mean((arr-ref)**2))),
             'scope':meta['scope']}
    (base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=3); a=p.parse_args()
    if a.repetitions<2:
        p.error('At least two repetitions are needed')
    try:
        ctypes.CDLL('/usr/local/lib/libprism-static.so').interflop_prism_set_rounding_mode(1)
    except OSError:
        pass  # May also run in the ordinary environment.
    compare(a.experiment,a.repetitions)

if __name__=='__main__':
    main()
