"""Exercise reference/SR CLI, comparison, and altered-input rejection on real-data crops."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uncnn1d.controls import Rounding, sha256

r=Rounding()
root=Path('/work')
source=root/'experiments/real_eeg64'
subprocess.run([sys.executable,str(root/'scripts/validate_real.py')],check=True)
with tempfile.TemporaryDirectory(prefix='pipeline_validation_',dir=root/'experiments') as directory:
    dest=Path(directory)
    manifest=json.loads((source/'manifest.json').read_text())
    shutil.copy2(source/'weights.pt',dest/'weights.pt')
    np.save(dest/'inputs.npy',np.array(np.load(source/'inputs.npy',mmap_mode='r')[:2,:,:256]))
    with np.load(source/'rows.npz',allow_pickle=False) as d:
        np.savez(dest/'rows.npz',**{k:d[k][:2] for k in d.files})
    manifest['n_times']=256
    manifest['validation_crop']=True
    manifest['artifact_hashes']={name:sha256(dest/name) for name in ('inputs.npy','weights.pt','rows.npz')}
    (dest/'manifest.json').write_text(json.dumps(manifest))
    cmd=[sys.executable,str(root/'scripts/02_extract_features.py'),'run','--experiment',str(dest)]
    for name in ('reference','reference_repeat'):
        subprocess.run(cmd+['--mode','rn','--reference-name',name],check=True)
    for repetition in (1,2,3):
        subprocess.run(cmd+['--mode','sr','--repetition',str(repetition),
                           '--perturbation-seed',str(1000+repetition)],check=True)
    subprocess.run([sys.executable,str(root/'scripts/04_compare_features.py'),
                    '--experiment',str(dest)],check=True)
    result=json.loads((dest/'summary.json').read_text())
    assert result['maximum_feature_range']>0
    with (dest/'inputs.npy').open('ab') as f:
        f.write(b'changed')
    bad=subprocess.run(cmd+['--mode','sr','--repetition','4'],capture_output=True,text=True)
    assert bad.returncode!=0 and 'Fixed artifact changed' in bad.stderr
    result['altered_input_rejected']=True
    result['validation_time_points']=256
    (source/'pipeline_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('End-to-end real-data crop validation passed.')
