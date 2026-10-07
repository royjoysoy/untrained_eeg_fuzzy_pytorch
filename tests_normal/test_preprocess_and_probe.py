import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def preprocess_module():
    spec=importlib.util.spec_from_file_location('preprocess',ROOT/'scripts/01_preprocess.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_channels_and_event_consistency(tmp_path):
    m=preprocess_module()
    assert len(m.CHANNELS_64)==64 and len(set(m.CHANNELS_64))==64
    assert len(m.CHANNELS_60)==60
    rows=[]
    for block in range(1,7):
        cond='Eyes Closed' if block%2 else 'Eyes Open'
        rows.extend([{'onset':block*70,'trial_type':cond,'value':block},
                     {'onset':block*70+2,'trial_type':cond,'value':block+10}])
    path=tmp_path/'events.tsv'; pd.DataFrame(rows).to_csv(path,sep='\t',index=False)
    assert m.block_onsets(path)=={i:i*70 for i in range(1,7)}
    rows[0]['trial_type']='Eyes Open'
    pd.DataFrame(rows).to_csv(path,sep='\t',index=False)
    with pytest.raises(ValueError,match='disagree'):
        m.block_onsets(path)


def test_fixed_probe_apply_does_not_refit(tmp_path):
    subs=np.array([f'sub-{i:03d}' for i in range(1,13)])
    rng=np.random.default_rng(0)
    X=rng.normal(size=(12,4)); X[6:,0]+=2
    labels=tmp_path/'participants.tsv'
    pd.DataFrame({'participant_id':subs,'BDI':[0]*6+[20]*6}).to_csv(labels,sep='\t',index=False)
    metadata={'weights_hash':'fixed','artifact_hashes':{'input':'fixed'},'source_hashes':{},
              'image_sha256':'fixed','variant':'eeg','dtype':'float32','channels':['A'],
              'condition':'EC','sfreq':250,'n_times':256,'mode':'rn','perturbation_seed':1001}
    def save(path,x,mode):
        np.savez(path,X=x,subject=subs,block=np.ones(12,int),cond=np.array(['EC']*12),
                 metadata=json.dumps({**metadata,'mode':mode}))
    reference=tmp_path/'reference.npz'; save(reference,X,'rn')
    bundle=tmp_path/'probe.joblib'; output=tmp_path/'reference_probe.npz'
    common=['--participants',str(labels),'--bundle',str(bundle)]
    command=[sys.executable,str(ROOT/'scripts/03_probe_bdi.py')]
    subprocess.run(command+['fit','--features',str(reference),'--out',str(output),
                   '--cv-repeats','1','--folds','3']+common,check=True,capture_output=True)
    original=hashlib.sha256(bundle.read_bytes()).hexdigest()
    perturbed=tmp_path/'sr.npz'; save(perturbed,X+0.05,'sr')
    # Fail immediately if applying attempts to call fit.
    runner="import runpy,sys; from sklearn.linear_model import LogisticRegression; LogisticRegression.fit=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('refit')); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    out=tmp_path/'sr_probe.npz'
    subprocess.run([sys.executable,'-c',runner,str(ROOT/'scripts/03_probe_bdi.py'),
                    'apply','--features',str(perturbed),'--out',str(out)]+common,
                   check=True,capture_output=True)
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==original
    with np.load(out) as data:
        assert data['proba'].shape==(1,12) and np.isfinite(data['proba']).all()
    save(perturbed,X,'sr')
    # Swapping subjects must be rejected even when feature dimensions match.
    with np.load(perturbed) as d:
        wrong={k:d[k].copy() for k in d.files}
    wrong['subject']=subs[::-1]; np.savez(perturbed,**wrong)
    bad=subprocess.run(command+['apply','--features',str(perturbed),
                       '--out',str(tmp_path/'bad.npz')]+common,capture_output=True,text=True)
    assert bad.returncode!=0 and 'membership changed' in bad.stderr


def test_complete_ec_blocks_survive_truncated_eo(tmp_path, monkeypatch):
    m=preprocess_module()
    class Raw:
        ch_names=list(m.CHANNELS_64)
        info={'bads':[]}
        def rename_channels(self, mapping):
            self.ch_names=[mapping[c] for c in self.ch_names]
        def pick(self, names):
            self.ch_names=list(names)
        def reorder_channels(self, names):
            self.ch_names=list(names)
        def set_channel_types(self, *a, **k): pass
        def filter(self, *a, **k): pass
        def set_eeg_reference(self, *a, **k): pass
        def resample(self, *a, **k): pass
        def get_data(self): return np.ones((64,450))
    monkeypatch.setattr(m.mne.io,'read_raw_eeglab',lambda *a,**k:Raw())
    path=tmp_path/'events.tsv'
    pd.DataFrame([{'onset':i*70,'value':i,
                   'trial_type':'Eyes Closed' if i%2 else 'Eyes Open'}
                  for i in range(1,7)]).to_csv(path,sep='\t',index=False)
    with pytest.raises(ValueError,match='block 6 runs past'):
        m.preprocess_subject(Path('test.set'),path,1,40,1,64,'both')
    d=m.preprocess_subject(Path('test.set'),path,1,40,1,64,'EC')
    assert d['X'].shape==(3,64,60)
    assert d['block'].tolist()==[1,3,5] and d['cond'].tolist()==['EC']*3
