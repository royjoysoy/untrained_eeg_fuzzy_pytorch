#!/usr/bin/env python
"""Keep independently validated EC/EO blocks from ds003478 run 01."""
import argparse
import importlib.util
import json
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def select_blocks(events, duration, boundaries):
    values = pd.to_numeric(events['value'], errors='coerce')
    markers = events.loc[values.isin(list(range(1, 7)) + list(range(11, 17)))].copy()
    markers['block'] = values.loc[markers.index].astype(int) % 10
    markers['onset'] = pd.to_numeric(markers['onset'], errors='coerce')
    starts = markers.groupby('block')['onset'].min().to_dict()
    valid, rejected = {}, {}
    for block in range(1, 7):
        rows = markers[markers.block == block]
        label = 'Eyes Closed' if block % 2 else 'Eyes Open'
        times = np.sort(rows.onset.to_numpy())
        texts = rows.trial_type.astype(str)
        reason = None
        if not len(times):
            reason = 'missing block markers'
        elif not np.isfinite(times).all() or times[0] < 0:
            reason = 'invalid marker times'
        elif any(t.startswith('Eyes') and not t.startswith(label) for t in texts):
            reason = 'condition label disagrees with numeric code'
        else:
            start, end = float(times[0]), float(times[0] + 60)
            if end > duration + 1e-6:
                reason = '60-second window exceeds recording'
            elif times[-1] < end:
                reason = 'markers do not span a full 60-second block'
            elif any(start < onset < end for other, onset in starts.items() if other != block):
                reason = 'another condition/block starts inside window'
            elif any(start < b < end for b in boundaries):
                reason = 'recording discontinuity inside window'
            else:
                valid[block] = start
        if reason:
            rejected[str(block)] = reason
    return valid, rejected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bids-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError('Use a new empty output folder')
    spec = importlib.util.spec_from_file_location('settings', Path(__file__).with_name('01_preprocess.py'))
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    for condition in ('EC', 'EO'):
        (a.out / condition).mkdir(parents=True, exist_ok=True)
    report = dict(source=str(a.bids_root), run='01', channels=64,
                  channel_order=settings.CHANNELS_64, low_hz=1, high_hz=40,
                  sfreq=250, seconds=60, reference='average', ica=False,
                  policy='retain each valid block independently; no automatic artifact rejection',
                  mne=mne.__version__, numpy=np.__version__, pandas=pd.__version__,
                  subjects={}, status='running')

    def save():
        temporary = a.out / 'preprocessing.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(a.out / 'preprocessing.json')

    save()
    participants = a.bids_root / 'participants.tsv'
    if participants.exists():
        for condition in ('EC', 'EO'):
            (a.out / condition / 'participants.tsv').write_bytes(participants.read_bytes())
    for directory in sorted(a.bids_root.glob('sub-*')):
        if not directory.is_dir():
            continue
        subject = directory.name
        entry = report['subjects'][subject] = {}
        if subject == 'sub-038':
            entry['excluded'] = 'dataset flags INVALID PARTICIPANT'
            save()
            continue
        try:
            stem = directory / 'eeg' / f'{subject}_task-Rest_run-01'
            events = pd.read_csv(f'{stem}_events.tsv', sep='\t')
            raw = mne.io.read_raw_eeglab(f'{stem}_eeg.set', preload=True, verbose='error')
            names = [name.upper() for name in raw.ch_names]
            if len(names) != len(set(names)):
                raise ValueError('duplicate channel names')
            raw.rename_channels(dict(zip(raw.ch_names, names)))
            raw.pick(settings.CHANNELS_64)
            raw.reorder_channels(settings.CHANNELS_64)
            if raw.info['bads']:
                raise ValueError(f'flagged bad electrodes: {raw.info["bads"]}')
            boundary_events = events[events['value'].astype(str).str.lower().eq('boundary')]
            boundaries = list(pd.to_numeric(boundary_events.onset, errors='raise'))
            boundaries += [float(onset - raw.first_time) for onset, description in
                           zip(raw.annotations.onset, raw.annotations.description)
                           if 'boundary' in description.lower() or 'bad' in description.lower()]
            valid, rejected = select_blocks(events, raw.n_times / raw.info['sfreq'], boundaries)
            entry['rejected_blocks'] = rejected
            raw.set_channel_types({name: 'eeg' for name in raw.ch_names})
            raw.filter(1, 40, fir_design='firwin',
                       skip_by_annotation=('edge', 'bad_acq_skip', 'boundary'), verbose='error')
            raw.set_eeg_reference('average', projection=False, verbose='error')
            raw.resample(250, verbose='error')
            data = raw.get_data()
            for condition, parity in [('EC', 1), ('EO', 0)]:
                blocks, segments = [], []
                for block, onset in sorted(valid.items()):
                    if block % 2 != parity:
                        continue
                    start = int(round(onset * 250))
                    segment = data[:, start:start + 15000]
                    if segment.shape != (64, 15000) or not np.isfinite(segment).all():
                        rejected[str(block)] = 'incomplete or nonfinite processed segment'
                        continue
                    blocks.append(block)
                    segments.append(segment)
                entry[condition] = blocks
                if segments:
                    np.savez_compressed(a.out / condition / f'{subject}.npz',
                                        X=np.stack(segments).astype(np.float32),
                                        block=np.array(blocks), cond=np.array([condition] * len(blocks)),
                                        ch_names=np.array(raw.ch_names), sfreq=250.)
            logging.info('%s EC=%s EO=%s rejected=%s', subject, entry['EC'], entry['EO'], rejected)
        except Exception as error:
            entry['error'] = f'{type(error).__name__}: {error}'
            logging.exception('%s could not be processed', subject)
        save()
    report['status'] = 'completed'
    report['counts'] = {condition: {
        'subjects': sum(bool(entry.get(condition)) for entry in report['subjects'].values()),
        'blocks': sum(len(entry.get(condition, [])) for entry in report['subjects'].values()),
        'subjects_with_three_blocks': sum(len(entry.get(condition, [])) == 3 for entry in report['subjects'].values()),
    } for condition in ('EC', 'EO')}
    report['subject_errors'] = sum('error' in entry for entry in report['subjects'].values())
    save()
    logging.info('Finished %s; subject errors=%s', report['counts'], report['subject_errors'])


if __name__ == '__main__':
    main()
