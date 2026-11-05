"""Download OpenNeuro datasets when they are not on disk (no account, no extra package).

OpenNeuro mirrors every dataset in a public S3 bucket. We list it over plain HTTPS and fetch
only the files we need (e.g. run-01), skipping files already downloaded with the right size.
S3 serves the latest snapshot, so we check that its version is the one the config expects.

    python -c "from untrained_eeg.download import download_openneuro; \\
               download_openneuro('ds003478', 'data/raw/ds003478', ['participants.tsv', '*_run-01_*'])"
"""

import fnmatch
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_files(dataset, bucket=BUCKET):
    """{path inside the dataset: size in bytes} for every file under the prefix `dataset`."""
    files, token = {}, None
    while True:
        query = {"list-type": "2", "prefix": f"{dataset}/"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{bucket}?{urllib.parse.urlencode(query)}") as r:
            root = ET.fromstring(r.read())
        for item in root.findall("s3:Contents", _NS):
            key = item.find("s3:Key", _NS).text
            files[key.removeprefix(f"{dataset}/")] = int(item.find("s3:Size", _NS).text)
        token = root.findtext("s3:NextContinuationToken", namespaces=_NS)
        if not token:
            return files


def _fetch(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")  # a half-written file never looks complete
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(out)


def download_openneuro(dataset, dest, include=("*",), version=None, workers=8, bucket=BUCKET):
    """Download the files of `dataset` whose name matches one of `include` into `dest`.

    `workers` files at a time: S3 limits the speed of each connection, not the total.
    `bucket`: another public S3 bucket (then `dataset` is a prefix in it, e.g. MPI-LEMON).
    """
    dest = Path(dest)
    files = {p: s for p, s in list_files(dataset, bucket).items()
             if any(fnmatch.fnmatch(Path(p).name, pat) or fnmatch.fnmatch(p, pat) for pat in include)}
    todo = {p: s for p, s in files.items() if not ((dest / p).exists() and (dest / p).stat().st_size == s)}
    print(f"{dataset}: {len(files)} files ({sum(files.values()) / 1e9:.1f} GB), {len(todo)} to download")
    with ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(_fetch, f"{bucket}/{dataset}/{urllib.parse.quote(p)}", dest / p) for p in sorted(todo)]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()  # re-raises a failed download
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
    if version:  # OpenNeuro only: its CHANGES file starts with the version it serves
        latest = urllib.request.urlopen(f"{BUCKET}/{dataset}/CHANGES").read().decode().split()[0]
        if latest != version:
            print(f"WARNING: OpenNeuro now serves {dataset} v{latest}, the project uses v{version}")
    return dest


def download_urls(urls, dest, workers=8):
    """Download {file name: url} into `dest`, `workers` at a time; existing files are skipped."""
    dest = Path(dest)
    todo = {n: u for n, u in urls.items() if not (dest / n).exists()}
    print(f"{len(urls)} files, {len(todo)} to download into {dest}")
    with ThreadPoolExecutor(workers) as pool:
        for job in as_completed([pool.submit(_fetch, u, dest / n) for n, u in todo.items()]):
            job.result()
    return dest
