#!/usr/bin/env python3
"""Fetch a seeded patient subset of LIDC-IDRI's CT from TCIA's public NBIA REST API.

Usage: .venv-lidc/bin/python scripts/datasets/lidc_fetch.py <out_dir> [--patients 300] [--seed 0]
                                                              [--workers 6] [--list-only]

Writes under <out_dir> (data/lidc, gitignored):
  manifest_series.json   every series TCIA lists for the collection (15,116 rows on 2026-10-07:
                         1,018 CT, 237 DX, 53 CR, 6,949 SEG, 6,859 SR), fetched once and kept
  patients.json          the chosen patients, one CT series each, sorted by PatientID
  dicom/<PatientID>/     that series' .dcm files, flat, plus a .done record with the count

Selection. The collection has 1,010 patients with CT (eight have two CT series; the one with
more slices is taken). The subset is stratified by scanner manufacturer in the collection's
own proportions (GE 670 / Siemens 205 / Philips 74 / Toshiba 69 of the 1,018 series), largest
remainder, seeded random within each stratum, so a 300-patient draw sees the same mix of
reconstruction kernels and slice thicknesses as the full set. 300 patients at seed 0 are
about 38 GB and 72k slices; the full set is 128 GB.

Download. getImage returns one ZIP per series. TCIA serves a few connections at once without
complaint, so `--workers` series download in parallel, each streaming to a temp file and
unzipping into the patient's directory. A patient whose .done record exists is skipped, so
the script is idempotent and resumable. A series whose unzipped .dcm count differs from the
manifest's ImageCount is kept and the count recorded: LIDC has a handful of such series and
the preprocessor decides per volume.

No authentication: LIDC-IDRI is a public collection (CC BY 3.0) and the v1 API needs none.
"""
import argparse
import collections
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "LIDC-IDRI"
KEEP = ("PatientID", "SeriesInstanceUID", "StudyInstanceUID", "ImageCount", "FileSize",
        "Manufacturer", "ManufacturerModelName")


def get(url, timeout=900):
    return urlopen(Request(url, headers={"User-Agent": "lean4-mlir lidc_fetch"}), timeout=timeout)


def fetch_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    with get(f"{API}/getSeries?Collection={COLLECTION}", timeout=300) as r:
        data = json.load(r)
    with open(path + ".tmp", "w") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)
    return data


def select(manifest, n, seed):
    """One CT series per patient, then a manufacturer-stratified seeded sample of n patients."""
    best = {}
    for s in manifest:
        if s.get("Modality") != "CT":
            continue
        p = s["PatientID"]
        if p not in best or int(s["ImageCount"]) > int(best[p]["ImageCount"]):
            best[p] = s
    rows = sorted(best.values(), key=lambda s: s["PatientID"])
    if n >= len(rows):
        return rows
    strata = collections.defaultdict(list)
    for s in rows:
        strata[s.get("Manufacturer") or "?"].append(s)
    quota = {m: n * len(v) / len(rows) for m, v in strata.items()}
    alloc = {m: int(q) for m, q in quota.items()}
    for m in sorted(quota, key=lambda m: quota[m] - alloc[m], reverse=True)[: n - sum(alloc.values())]:
        alloc[m] += 1
    rng = random.Random(seed)
    chosen = []
    for m in sorted(strata):
        chosen += rng.sample(strata[m], alloc[m])
    return sorted(chosen, key=lambda s: s["PatientID"])


def fetch_series(row, dicom_dir, retries=5):
    pid, uid, n = row["PatientID"], row["SeriesInstanceUID"], int(row["ImageCount"])
    pdir = os.path.join(dicom_dir, pid)
    done = os.path.join(pdir, ".done")
    if os.path.exists(done):
        with open(done) as f:
            return pid, "skip", json.load(f)["files"]
    os.makedirs(pdir, exist_ok=True)
    url = f"{API}/getImage?SeriesInstanceUID={uid}"
    for attempt in range(retries):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".zip", dir=dicom_dir)
            with os.fdopen(fd, "wb") as f, get(url) as r:
                shutil.copyfileobj(r, f, 1 << 20)
            with zipfile.ZipFile(tmp) as z:
                for m in z.namelist():
                    if not m.lower().endswith(".dcm"):
                        continue
                    with z.open(m) as src, open(os.path.join(pdir, os.path.basename(m)), "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
            os.remove(tmp)
            dcm = [os.path.join(pdir, f) for f in os.listdir(pdir) if f.endswith(".dcm")]
            rec = {"PatientID": pid, "SeriesInstanceUID": uid, "ImageCount": n,
                   "files": len(dcm), "bytes": sum(os.path.getsize(f) for f in dcm)}
            with open(done, "w") as f:
                json.dump(rec, f)
            return pid, "ok" if len(dcm) == n else "count-mismatch", len(dcm)
        except (URLError, HTTPError, zipfile.BadZipFile, TimeoutError, OSError) as e:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            wait = 15 * 2 ** attempt
            print(f"  {pid}: attempt {attempt + 1} failed ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    return pid, "failed", 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--patients", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--list-only", action="store_true", help="write patients.json and stop")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = fetch_manifest(os.path.join(args.out_dir, "manifest_series.json"))
    chosen = select(manifest, args.patients, args.seed)
    gb = sum(int(s["FileSize"]) for s in chosen) / 1e9
    slices = sum(int(s["ImageCount"]) for s in chosen)
    makers = dict(collections.Counter(s["Manufacturer"] for s in chosen))
    print(f"{len(chosen)} patients, {slices} slices, {gb:.1f} GB, seed {args.seed}; {makers}", flush=True)
    with open(os.path.join(args.out_dir, "patients.json"), "w") as f:
        json.dump([{k: s.get(k) for k in KEEP} for s in chosen], f, indent=1)
    if args.list_only:
        return

    dicom_dir = os.path.join(args.out_dir, "dicom")
    os.makedirs(dicom_dir, exist_ok=True)
    t0, got, tally = time.time(), 0, collections.Counter()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_series, s, dicom_dir): s for s in chosen}
        for i, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            pid, status, files = fut.result()
            tally[status] += 1
            if status not in ("skip", "failed"):
                got += int(s["FileSize"])
            print(f"[{i}/{len(chosen)}] {pid} {status} {files}/{s['ImageCount']} slices"
                  f"  {got / 1e9:.1f} GB fetched in {(time.time() - t0) / 60:.0f} min", flush=True)
    print("done:", dict(tally), flush=True)
    sys.exit(1 if tally["failed"] else 0)


if __name__ == "__main__":
    main()
