#!/usr/bin/env python
"""Prepare fixed inputs/weights once, or extract features under controlled RN/SR."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import scipy
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncnn1d import VARIANTS, build_input, load_model
from uncnn1d.controls import Rounding, sha256, source_hashes, weights_hash, cpu_signature


def prepare(args, rounding):
    files = sorted(args.deriv.glob('sub-*.npz'))
    if args.subjects:
        files = [args.deriv / f'{s}.npz' for s in sorted(args.subjects)]
    if not files or len(files) != len(set(files)):
        raise ValueError('No derivatives or duplicate subjects')
    channels, sfreq, n_times = None, None, None
    subjects, blocks, conditions, inputs = [], [], [], []
    input_hashes = {}
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            X, names = d['X'], d['ch_names'].tolist()
            block, cond = d['block'], d['cond']
            rate = float(d['sfreq'])
            if X.ndim != 3 or X.shape[1] != len(names) or X.shape[2] < 32:
                raise ValueError(f'{path}: invalid EEG shape')
            if len(names) != len(set(names)) or len(names) not in (60, 64):
                raise ValueError(f'{path}: expected 60 or 64 unique electrodes')
            if not np.isfinite(X).all() or not np.isfinite(rate) or rate <= 0:
                raise ValueError(f'{path}: nonfinite data or sampling rate')
            if len(block) != len(X) or len(cond) != len(X) or len(set(block.tolist())) != len(block):
                raise ValueError(f'{path}: duplicate blocks or mismatched metadata')
            for b,c in zip(block,cond):
                if int(b) not in range(1,7) or c != ('EC' if int(b)%2 else 'EO'):
                    raise ValueError(f'{path}: block/condition disagreement')
            if channels is None:
                channels, sfreq, n_times = names, rate, X.shape[2]
            if names != channels or rate != sfreq or X.shape[2] != n_times:
                raise ValueError(f'{path}: channel order, sampling rate or length differs')
            input_hashes[path.name] = sha256(path)
            for i in np.argsort(block):
                if args.cond != 'both' and cond[i] != args.cond:
                    continue
                inputs.append(build_input(X[i], rank_size=VARIANTS[args.variant].rank_size))
                subjects.append(path.stem); blocks.append(int(block[i])); conditions.append(str(cond[i]))
    if not inputs:
        raise ValueError('No blocks selected')
    if args.experiment.exists():
        raise FileExistsError('Use a new experiment folder; fixed artifacts cannot be overwritten')
    args.experiment.mkdir(parents=True)
    cached = args.experiment / 'inputs.npy'
    arr = np.lib.format.open_memmap(cached, mode='w+', dtype=args.dtype,
                                   shape=(len(inputs), *inputs[0].shape))
    for i,x in enumerate(inputs):
        arr[i] = x
    arr.flush(); del arr
    np.savez(args.experiment / 'rows.npz', subject=np.array(subjects),
             block=np.array(blocks), cond=np.array(conditions))
    model = load_model(3*len(channels), variant=args.variant, seed=args.seed,
                       dtype=getattr(torch,args.dtype))
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    checkpoint = args.experiment / 'weights.pt'
    torch.save(model.state_dict(), checkpoint)
    manifest = {'variant': args.variant, 'weight_seed': args.seed, 'dtype': args.dtype,
                'in_channels': 3*len(channels), 'channels': channels, 'sfreq': sfreq,
                'n_times': n_times, 'condition': args.cond, 'batch_size': 1,
                'feature_dim': model.feature_dim(), 'feature_names': model.feature_names(),
                'weights_hash': weights_hash(model), 'source_hashes': source_hashes(),
                'derivative_hashes': input_hashes, 'torch': torch.__version__,
                'numpy': np.__version__, 'trained_checkpoint': bool(args.checkpoint),
                'scipy': scipy.__version__,
                'image_sha256': os.environ['FUZZY_IMAGE_SHA256'],
                'artifact_hashes': {p.name:sha256(p) for p in
                    (cached, checkpoint, args.experiment/'rows.npz')}}
    (args.experiment / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared {len(inputs)} blocks from {len(set(subjects))} subjects: {args.experiment}')


def extract(args, rounding):
    m = json.loads((args.experiment/'manifest.json').read_text())
    if source_hashes() != m['source_hashes']:
        raise RuntimeError('Experiment code changed after preparation')
    if os.environ['FUZZY_IMAGE_SHA256'] != m['image_sha256']:
        raise RuntimeError('Container image changed')
    if torch.__version__ != m['torch'] or np.__version__ != m['numpy'] or scipy.__version__ != m['scipy']:
        raise RuntimeError('Library versions changed')
    for name, value in m['artifact_hashes'].items():
        if sha256(args.experiment/name) != value:
            raise RuntimeError(f'Fixed artifact changed: {name}')
    out = args.experiment / (f'sr_rep-{args.repetition}.npz' if args.mode=='sr' else args.reference_name+'.npz')
    if out.exists():
        raise FileExistsError(out)
    model = load_model(m['in_channels'], variant=m['variant'], seed=m['weight_seed'],
                       dtype=getattr(torch,m['dtype']))
    model.load_state_dict(torch.load(args.experiment/'weights.pt',weights_only=True),strict=True)
    if weights_hash(model) != m['weights_hash']:
        raise RuntimeError('Weights differ')
    X = np.load(args.experiment/'inputs.npy', mmap_mode='r', allow_pickle=False)
    rounding.seed(args.perturbation_seed)
    rows = []
    for i in range(len(X)):
        x = torch.from_numpy(np.array(X[i],copy=True)).unsqueeze(0)
        rounding.mode(args.mode)
        try:
            result = model(x)
        finally:
            rounding.mode('rn')
        rows.append(result.squeeze(0).double().numpy())
    features = np.vstack(rows)
    if features.shape != (len(X),m['feature_dim']) or not np.isfinite(features).all():
        raise RuntimeError('Incorrect or nonfinite features')
    if weights_hash(model) != m['weights_hash']:
        raise RuntimeError('Inference changed weights')
    metadata = {**{k:v for k,v in m.items() if k != 'feature_names'},
                'mode':args.mode, 'repetition': args.repetition,
                'perturbation_seed':args.perturbation_seed,
                'prism_startup_seed':int(os.environ['PRISM_SEED']),
                'prism_seed_reported':int(rounding.lib.interflop_prism_get_seed()),
                'precision_binary32':24, 'precision_binary64':53,
                'scope':'CNN forward including GroupNorm and PyTorch feature readout',
                'host':os.uname().nodename, 'cpu':cpu_signature(),
                'torch_build':torch.__config__.show()}
    with np.load(args.experiment/'rows.npz',allow_pickle=False) as index:
        np.savez_compressed(out,X=features,subject=index['subject'],block=index['block'],
                            cond=index['cond'], metadata=json.dumps(metadata))
    print(f'Saved {out}: {features.shape}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--deriv',type=Path,default=Path('data/derivatives'))
    p.add_argument('--subjects',nargs='+')
    p.add_argument('--variant',choices=list(VARIANTS),default='eeg')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--checkpoint',type=Path,help='Optional trained state_dict for this exact architecture')
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--cond',choices=['EC','EO','both'],default='both')
    p.add_argument('--mode',choices=['rn','sr'],default='rn')
    p.add_argument('--repetition',type=int,default=1)
    p.add_argument('--perturbation-seed',type=int,default=1001)
    p.add_argument('--reference-name',choices=['reference','reference_repeat'],default='reference')
    a=p.parse_args()
    if a.repetition < 1:
        p.error('Repetition must be positive')
    if a.action == 'run' and os.environ.get('PRISM_SEED') != str(a.perturbation_seed):
        p.error('Launch via container.sh so PRISM_SEED matches --perturbation-seed before PyTorch loads')
    rounding=Rounding()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    prepare(a,rounding) if a.action=='prepare' else extract(a,rounding)

if __name__=='__main__':
    main()
