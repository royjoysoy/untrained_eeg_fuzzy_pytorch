#!/usr/bin/env python
"""Compute budget for the manuscript: SLURM accounting + the run_info records.

    python optimization/compute_cost.py --out results/optimization/11-5-26

Reads <out>/job_ids.txt (written by slurm/run_optimization.sh, one "<stage> <dataset> <job id>"
per line) and asks sacct for every job and array task: state, elapsed (wall), total CPU
time, allocated CPUs, CPU-time (elapsed x CPUs), MaxRSS, node. Also collects
<out>/run_info/*.json (git commit, versions, CPU model, wall seconds of each script).
Writes <out>/summary/compute_cost.csv (one row per job step), compute_by_stage.csv
(CPU-hours, wall hours, peak memory per stage x dataset) and run_info.csv.
Run it soon after the jobs end: sacct keeps records for a limited time.
"""

import argparse
import io
import json
import subprocess
from pathlib import Path

import pandas as pd

FIELDS = "JobID,JobName,State,Elapsed,ElapsedRaw,TotalCPU,CPUTimeRAW,AllocCPUS,MaxRSS,NodeList,Start,End"


def mem_gb(v):
    if not isinstance(v, str) or not v:
        return float("nan")
    unit = v[-1].upper()
    try:
        x = float(v[:-1]) if unit in "KMGT" else float(v)
    except ValueError:
        return float("nan")
    return x * {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1, "T": 2**10}.get(unit, 1 / 2**30)


def cpu_seconds(v):
    """sacct TotalCPU like [D-]HH:MM:SS(.sss) or MM:SS.sss -> seconds."""
    if not isinstance(v, str) or not v:
        return float("nan")
    days = 0
    if "-" in v:
        d, v = v.split("-", 1)
        days = int(d)
    parts = [float(x) for x in v.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dest = a.out / "summary"
    dest.mkdir(parents=True, exist_ok=True)
    jobs = pd.read_csv(a.out / "job_ids.txt", sep=" ", names=["stage", "dataset", "job"], dtype=str)
    txt = subprocess.run(["sacct", "-P", "-n", "--units=M", "-o", FIELDS, "-j", ",".join(jobs.job)],
                         capture_output=True, text=True, check=True).stdout
    acct = pd.read_csv(io.StringIO(txt), sep="|", names=FIELDS.split(","), dtype=str)
    acct["job"] = acct.JobID.str.extract(r"^(\d+)")[0]
    acct = acct.merge(jobs, on="job", how="left")
    acct["max_rss_gb"] = acct.MaxRSS.map(mem_gb)
    acct["total_cpu_s"] = acct.TotalCPU.map(cpu_seconds)
    acct["elapsed_s"] = pd.to_numeric(acct.ElapsedRaw, errors="coerce")
    acct["alloc_cpu_s"] = pd.to_numeric(acct.CPUTimeRAW, errors="coerce")
    acct.to_csv(dest / "compute_cost.csv", index=False)
    # one row per job or array task (the ".batch" step carries the memory, the parent the times)
    main_rows = acct[~acct.JobID.str.contains(r"\.")].copy()
    mem = acct.groupby(acct.JobID.str.replace(r"\..*$", "", regex=True)).max_rss_gb.max()
    main_rows["max_rss_gb"] = main_rows.JobID.map(mem)
    by = main_rows.groupby(["stage", "dataset"]).agg(
        n_jobs=("JobID", "size"), failed=("State", lambda s: int((~s.str.startswith("COMPLETED")).sum())),
        used_cpu_hours=("total_cpu_s", lambda s: s.sum() / 3600),
        allocated_cpu_hours=("alloc_cpu_s", lambda s: s.sum() / 3600),
        wall_hours_sum=("elapsed_s", lambda s: s.sum() / 3600),
        wall_hours_longest=("elapsed_s", lambda s: s.max() / 3600),
        peak_mem_gb=("max_rss_gb", "max")).reset_index()
    by.loc[len(by)] = ["TOTAL", "", by.n_jobs.sum(), by.failed.sum(), by.used_cpu_hours.sum(),
                       by.allocated_cpu_hours.sum(), by.wall_hours_sum.sum(), by.wall_hours_longest.max(),
                       by.peak_mem_gb.max()]
    by.to_csv(dest / "compute_by_stage.csv", index=False, float_format="%.3g")
    runs = [json.loads(f.read_text()) for f in sorted((a.out / "run_info").glob("*.json"))]
    if runs:
        r = pd.json_normalize(runs)
        r.to_csv(dest / "run_info.csv", index=False)
        print("environments seen:", r.groupby(["git_commit", "cpu_model"]).size().to_dict())
    print(by.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
