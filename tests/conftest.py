"""Model determinism tests use ordinary rounding, even inside the Fuzzy image."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncnn1d.controls import Rounding


def pytest_sessionstart(session):
    Rounding().mode('rn')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
