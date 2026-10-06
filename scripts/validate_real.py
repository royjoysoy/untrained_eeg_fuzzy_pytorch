"""Small actual-data check, using the real experiment's fixed checkpoint."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from uncnn1d import load_model
from uncnn1d.controls import Rounding, weights_hash
base=Path('/work/experiments/real_eeg64')
m=json.loads((base/'manifest.json').read_text())
r=Rounding()
torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
model=load_model(m['in_channels'],variant=m['variant'],seed=m['weight_seed'])
model.load_state_dict(torch.load(base/'weights.pt',weights_only=True))
x=torch.from_numpy(np.array(np.load(base/'inputs.npy',mmap_mode='r')[0,:,:256],copy=True)).unsqueeze(0)
before=weights_hash(model)
a=model(x).numpy().copy(); b=model(x).numpy().copy()
assert np.array_equal(a,b)
values=[]
for seed in (1001,1002):
    r.seed(seed); r.mode('sr')
    try:
        value=model(x)
    finally:
        r.mode('rn')
    values.append(value.numpy().copy())
assert all(np.isfinite(v).all() for v in values)
assert not np.array_equal(*values)
assert before==weights_hash(model)==m['weights_hash']
info={'reference_repeatable':True,'perturbations_observed':True,'weights_unchanged':True,
      'input_channels':m['in_channels'],'feature_dim':a.shape[1],
      'validation_time_points':256,'maximum_feature_difference':float(np.abs(values[0]-values[1]).max())}
(base/'validation.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
