"""Record what every run used and how long it took (for the manuscript's methods and costs).

    with run_info(out_dir, "extract_cnn", dataset="ds003478", extra={...}):
        ...work...

Writes <out>/run_info/<name>[_<dataset>]_<slurm job id or pid>_<start time>.json with:
git commit (and whether the tree had local changes), command line, Python and package
versions, CPU model, usable CPUs, PyTorch threads, host, SLURM job/array ids and
resources, start/end time and wall seconds. Never overwrites an existing file.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "torch", "pyriemann", "fooof", "aeon",
            "numba", "joblib", "threadpoolctl", "pyyaml", "matplotlib")
SLURM_VARS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME",
              "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST", "SLURM_CLUSTER_NAME")


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def snapshot() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    info = dict(git_commit=_git("rev-parse", "HEAD"), git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
                git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
                command=" ".join(sys.argv), python=platform.python_version(), packages=versions,
                host=socket.gethostname(), cpu_model=_cpu_model(),
                usable_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
                omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                slurm={k: os.environ[k] for k in SLURM_VARS if k in os.environ})
    try:
        import torch
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


@contextlib.contextmanager
def run_info(out: Path, name: str, dataset: str | None = None, extra: dict | None = None):
    start = datetime.now()
    t0 = time.time()
    rec = dict(name=name, dataset=dataset, start=start.isoformat(timespec="seconds"), **snapshot())
    status = "ok"
    try:
        yield rec  # callers may add counts to rec while running
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        rec.update(end=datetime.now().isoformat(timespec="seconds"), wall_seconds=round(time.time() - t0, 2),
                   status=status, **(extra or {}))
        job = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
        dest = Path(out) / "run_info" / f"{name}{'_' + dataset if dataset else ''}_{job}_{start:%Y%m%dT%H%M%S}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_text(json.dumps(rec, indent=2, default=str) + "\n")
