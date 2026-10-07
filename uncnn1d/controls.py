"""PRISM runtime controls verified against this image's prism_api.h."""
import ctypes
import hashlib
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def weights_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Rounding:
    def __init__(self):
        if os.environ.get('VFC_BACKENDS') != 'libinterflop_ieee.so':
            raise RuntimeError('Use scripts/container.sh; other VFC backends must remain IEEE')
        self.lib = ctypes.CDLL('/usr/local/lib/libprism-static.so')
        for name in ('set_rounding_mode', 'set_default_virtual_precision_binary32',
                     'set_default_virtual_precision_binary64'):
            getattr(self.lib, 'interflop_prism_' + name).argtypes = [ctypes.c_int32]
        self.lib.interflop_prism_set_seed.argtypes = [ctypes.c_uint64]
        self.lib.interflop_prism_get_rounding_mode.restype = ctypes.c_int32
        self.lib.interflop_prism_get_seed.restype = ctypes.c_uint64
        self.lib.interflop_prism_set_default_virtual_precision_binary32(24)
        self.lib.interflop_prism_set_default_virtual_precision_binary64(53)
        self.mode('rn')

    def mode(self, mode):
        number = {'sr': 0, 'rn': 1}[mode]
        self.lib.interflop_prism_set_rounding_mode(number)
        if self.lib.interflop_prism_get_rounding_mode() != number:
            raise RuntimeError('PRISM mode did not take effect')

    def seed(self, seed):
        if not 0 <= seed < 2**64:
            raise ValueError('Perturbation seed must be uint64')
        self.lib.interflop_prism_set_seed(seed)


def source_hashes():
    base = Path(__file__).resolve().parents[1]
    paths = sorted((base / 'uncnn1d').glob('*.py')) + [
        base / 'scripts/02_extract_features.py', base / 'scripts/container.sh']
    return {str(p.relative_to(base)): sha256(p) for p in paths}


def cpu_signature():
    # Frequency and other /proc/cpuinfo fields can change during a job.
    wanted = {'vendor_id', 'model name', 'flags'}
    return sorted(set(line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.split(':', 1)[0].strip() in wanted))
