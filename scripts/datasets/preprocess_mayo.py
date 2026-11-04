#!/usr/bin/env python3
"""The held-out set of the CT denoising demo: Mayo's LDCT-and-Projection-data chest pairs as
pair-file records (planning/lidc_ct_denoising_demo.md §10.4).

  .venv-lidc/bin/python scripts/datasets/preprocess_mayo.py [data/mayo] [--stride 4] [--workers 3]
      [--patients N] [--only C002] [--delete-dicom] [--keep C002 ...]

Why. Every net in the demo is trained on LIDC-IDRI with OUR projection-domain noise simulation
(preprocess_lidc.py). Mayo's collection (TCIA, CC BY 4.0 since v7, July 2026) is the field's
reference set of paired images: 50 chest patients with both "Full Dose Images" and "Low Dose
Images" series (all Siemens, 17,313 slice pairs, 18.2 GB of DICOM), the low-dose arm made by the
vendor's own noise insertion in the projection domain at 10% of the routine dose. Other patients,
another institution, other scanners and kernels (B50f, sharp), another noise insertion: a net
scored here was never tuned on anything in it, so it is the generalisation check. No Mayo image is
used for training or for choosing a knob.

What it writes — the §5 pair-file format at N = 1 (the 25% LIDC build has N = 2), so
`F32.loadCtPair`, every scorer and `lidc_records.py` read it unchanged:

    val.bin   uint32 count, then per record: full S² uint16 (HU + 1024, 0..4095), low S² uint16,
              mask S² uint8 — all zero (Mayo has no reader outlines; nodule contrast kept stays an
              LIDC result)
    val.idx   uint32 patient count, then one record count per patient (patients in id order)
    val.json  one row per record: patient, slice (index in z order of the matched slices), z_mm,
              nodule_px (0), spacing, thickness, kernel, kvp, ma (the full-dose arm's tube
              current), ma_low, maker, model, and the noise estimates of §3's estimator (the
              median 7×7 local std over flat soft tissue): sigma_full_hu (the scan's own),
              sigma_add_hu (low − full: exactly the inserted noise, both arms being
              reconstructions of one scan), sigma_low_hu (total), calibrated (>= 500 flat pixels)
    manifest_mayo.json  args, per-patient counts, unmatched slices, the dose-ratio medians

Per patient: both series are fetched from the public NBIA API (no account) into
<out>/dicom/<pid>/{full,low}/ (skipped when complete), the two arms are matched slice to slice by
z position (2 dp), every `--stride`-th matched slice in z order is kept (every 4th: the LIDC val
split's stride), HU = raw·slope + intercept with values below −1024 set to air (−1000, as
preprocess_lidc.read_volume) and clipped to [−1024, 3071]. A worker fetches and converts one
patient and writes <out>/tmp_preprocess/<pid>.bin + .json; a patient whose .json exists is
skipped, so a rerun resumes. `--delete-dicom` removes a patient's DICOM once its records are
written, except the `--keep` patients (default C002, which
scripts/probes/lidc_mayo_noise_check.py reads). The temp files are concatenated in patient order
at the end. Expected: ≈ 4,330 records × 5·512² B ≈ 5.7 GB (§8), ≈ 35–60 min, most of it the
download (≈ 18 GB over 3 connections).

The check this data carries by itself: at 10% dose the inserted noise should be √9 = 3× the
scan's own and the total √10 ≈ 3.16× (§3; C002 measured 2.99 / 3.15). The manifest reports both
medians over every calibrated slice and their spread over patients.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import collections
import io
import json
import multiprocessing
import shutil
import struct
import sys
import time
import traceback
import zipfile
from urllib.request import Request, urlopen

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_lidc import CAL_MIN_PIXELS, HU_MAX, HU_MIN, encode, flat_soft_tissue, noise_level  # noqa: E402

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "LDCT-and-Projection-data"
META_KEYS = ("spacing", "thickness", "kernel", "kvp", "ma", "ma_low", "maker", "model")


# ---- the public NBIA API ---------------------------------------------------------------------

def get(url, timeout=900):
    return urlopen(Request(url, headers={"User-Agent": "lean4-mlir preprocess_mayo"}), timeout=timeout).read()


def chest_pairs(manifest):
    """patient id -> {"full": series, "low": series} for the chest patients with both image arms."""
    by = collections.defaultdict(dict)
    for s in manifest:
        desc = s.get("SeriesDescription") or ""
        if s.get("BodyPartExamined") == "CHEST" and "Images" in desc:
            by[s["PatientID"]]["low" if "Low" in desc else "full"] = s
    return {p: v for p, v in by.items() if "full" in v and "low" in v}


def fetch(series, out, attempts=3):
    """One series' DICOM into `out/` (skipped when its ImageCount .dcm files are there)."""
    os.makedirs(out, exist_ok=True)
    if len([f for f in os.listdir(out) if f.endswith(".dcm")]) >= int(series["ImageCount"]):
        return False
    for k in range(attempts):
        try:
            blob = get(f"{API}/getImage?SeriesInstanceUID={series['SeriesInstanceUID']}")
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for m in z.namelist():
                    if m.lower().endswith(".dcm"):
                        with open(os.path.join(out, os.path.basename(m)), "wb") as f:
                            f.write(z.read(m))
            return True
        except Exception as e:  # a dropped connection is retried, not fatal
            if k + 1 == attempts:
                raise
            print(f"  retry {k + 1} for {out}: {e!r}", file=sys.stderr, flush=True)
            time.sleep(10 * (k + 1))


def load_manifest(out_dir):
    path = os.path.join(out_dir, "manifest_series.json")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(get(f"{API}/getSeries?Collection={COLLECTION}", timeout=300))
    with open(path) as f:
        return chest_pairs(json.load(f))


# ---- one patient -------------------------------------------------------------------------------

def headers(d):
    """z (2 dp) -> (path, header) for the CT slices of one series directory, duplicates by z dropped."""
    out = {}
    rows = []
    for f in os.listdir(d):
        if f.endswith(".dcm"):
            path = os.path.join(d, f)
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if str(getattr(ds, "Modality", "CT")) != "CT":
                continue
            rows.append((round(float(ds.ImagePositionPatient[2]), 2), int(getattr(ds, "InstanceNumber", 0) or 0), path, ds))
    rows.sort(key=lambda r: (r[0], r[1]))
    for z, _, path, ds in rows:
        out.setdefault(z, (path, ds))
    return out, len(rows) - len(out)


def hu_of(path, size):
    ds = pydicom.dcmread(path)
    if int(ds.Rows) != size or int(ds.Columns) != size:
        raise ValueError(f"{path}: {int(ds.Rows)}x{int(ds.Columns)}, not {size}")
    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) \
        + float(getattr(ds, "RescaleIntercept", 0) or 0)
    hu[hu < HU_MIN] = -1000.0
    return np.clip(hu, HU_MIN, HU_MAX)


def process_patient(job):
    pid, pair, a = job
    t0 = time.time()
    tmp_bin = os.path.join(a["tmp_dir"], pid + ".bin")
    tmp_json = os.path.join(a["tmp_dir"], pid + ".json")
    if os.path.exists(tmp_json):
        with open(tmp_json) as f:
            r = json.load(f)
        r["resumed"] = True
        return r
    try:
        pdir = os.path.join(a["out_dir"], "dicom", pid)
        fetched = [fetch(pair[arm], os.path.join(pdir, arm)) for arm in ("full", "low")]
        t_fetch = time.time() - t0
        (fz, dup_f), (lz, dup_l) = headers(os.path.join(pdir, "full")), headers(os.path.join(pdir, "low"))
        common = sorted(set(fz) & set(lz))
        if not common:
            return {"patient": pid, "skip": "no z position shared by the two arms"}
        sel = common[:: max(1, a["stride"])]
        size = a["size"]
        ds0, dl0 = fz[common[0]][1], lz[common[0]][1]
        meta = {"spacing": float(ds0.PixelSpacing[0]),
                "thickness": float(getattr(ds0, "SliceThickness", 0) or 0),
                "kernel": str(getattr(ds0, "ConvolutionKernel", "")),
                "kvp": float(getattr(ds0, "KVP", 0) or 0),
                "ma": float(getattr(ds0, "XRayTubeCurrent", 0) or 0),
                "ma_low": float(getattr(dl0, "XRayTubeCurrent", 0) or 0),
                "maker": str(getattr(ds0, "Manufacturer", "")),
                "model": str(getattr(ds0, "ManufacturerModelName", ""))}
        c = size // 2
        yy, xx = np.ogrid[:size, :size]
        inside = ((yy - c) ** 2 + (xx - c) ** 2) <= c * c
        zero_mask = np.zeros(size * size, np.uint8).tobytes()
        rows = []
        with open(tmp_bin + ".part", "wb") as f:
            for z in sel:
                full = hu_of(fz[z][0], size)
                low = hu_of(lz[z][0], size)
                f.write(encode(full).tobytes())
                f.write(encode(low).tobytes())
                f.write(zero_mask)
                flat = flat_soft_tissue(full.astype(np.float64), inside)
                cal = int(flat.sum()) >= CAL_MIN_PIXELS
                rows.append({"patient": pid, "slice": int(common.index(z)), "z_mm": float(z), "nodule_px": 0,
                             **meta,
                             "sigma_full_hu": noise_level(full, flat) if cal else float("nan"),
                             "sigma_add_hu": noise_level(low - full, flat) if cal else float("nan"),
                             "sigma_low_hu": noise_level(low, flat) if cal else float("nan"),
                             "calibrated": cal})
        os.replace(tmp_bin + ".part", tmp_bin)
        cal_rows = [r for r in rows if r["calibrated"]]
        med = lambda key: float(np.median([r[key] for r in cal_rows])) if cal_rows else None  # noqa: E731
        r = {"patient": pid, "records": len(rows), "common": len(common),
             "full_slices": len(fz), "low_slices": len(lz),
             "unmatched": len(fz) + len(lz) - 2 * len(common), "dropped_duplicates": dup_f + dup_l,
             "series": {arm: pair[arm]["SeriesInstanceUID"] for arm in ("full", "low")},
             "image_count": {arm: int(pair[arm]["ImageCount"]) for arm in ("full", "low")},
             **meta, "calibrated_slices": len(cal_rows),
             "sigma_full_hu_median": med("sigma_full_hu"), "sigma_add_hu_median": med("sigma_add_hu"),
             "sigma_low_hu_median": med("sigma_low_hu"),
             "inserted_over_own_median": float(np.median([r["sigma_add_hu"] / r["sigma_full_hu"] for r in cal_rows])) if cal_rows else None,
             "total_over_own_median": float(np.median([r["sigma_low_hu"] / r["sigma_full_hu"] for r in cal_rows])) if cal_rows else None,
             "fetched": any(fetched), "fetch_seconds": t_fetch, "seconds": time.time() - t0, "rows": rows}
        if a["delete_dicom"] and pid not in a["keep"]:
            shutil.rmtree(pdir, ignore_errors=True)
            r["dicom_deleted"] = True
        with open(tmp_json + ".part", "w") as f:
            json.dump(r, f)
        os.replace(tmp_json + ".part", tmp_json)
        return r
    except Exception as e:  # one bad patient must not take the pool down
        return {"patient": pid, "skip": f"error: {e!r}", "trace": traceback.format_exc()}


# ---- driver ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", nargs="?", default="data/mayo")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--patients", type=int, default=0, help="only the first N patient ids")
    ap.add_argument("--only", default=None, help="one patient id")
    ap.add_argument("--delete-dicom", action="store_true")
    ap.add_argument("--keep", nargs="*", default=["C002"])
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    pairs = load_manifest(args.out_dir)
    pids = sorted(pairs)
    if args.only:
        pids = [args.only]
    if args.patients:
        pids = pids[: args.patients]
    tmp_dir = os.path.join(args.out_dir, "tmp_preprocess")
    os.makedirs(tmp_dir, exist_ok=True)
    a = {"out_dir": args.out_dir, "tmp_dir": tmp_dir, "stride": args.stride, "size": args.size,
         "delete_dicom": args.delete_dicom, "keep": set(args.keep)}
    total_slices = sum(int(pairs[p]["full"]["ImageCount"]) for p in pids)
    print(f"{len(pids)} chest patients with both arms, {total_slices} slices per arm, stride {args.stride}, "
          f"{args.workers} workers{', DICOM deleted after conversion' if args.delete_dicom else ''}", flush=True)
    jobs = [(p, pairs[p], a) for p in pids]
    results = {}
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(process_patient, jobs), 1):
            results[r["patient"]] = r
            el = time.time() - t0
            if "skip" in r:
                print(f"[{i}/{len(jobs)}] {r['patient']} SKIP {r['skip']}", flush=True)
            else:
                print(f"[{i}/{len(jobs)}] {r['patient']} {r['records']} records of {r['common']} matched slices "
                      f"({r['unmatched']} unmatched), {r['kernel']} {r['ma']:.0f}/{r['ma_low']:.0f} mA, "
                      f"own {r['sigma_full_hu_median'] or float('nan'):.1f} HU, inserted/own "
                      f"{r['inserted_over_own_median'] or float('nan'):.2f}, total/own "
                      f"{r['total_over_own_median'] or float('nan'):.2f}, {r['seconds']:.0f}s"
                      f"{' (resumed)' if r.get('resumed') else ''}  ({el / 60:.0f} min, "
                      f"ETA {el / i * (len(jobs) - i) / 60:.0f} min)", flush=True)

    rec_bytes = 5 * args.size * args.size
    keep = [p for p in pids if "skip" not in results[p]]
    n_rec = sum(results[p]["records"] for p in keep)
    out_bin = os.path.join(args.out_dir, "val.bin")
    with open(out_bin + ".tmp", "wb") as out:
        out.write(struct.pack("<I", n_rec))
        for p in keep:
            with open(os.path.join(tmp_dir, p + ".bin"), "rb") as src:
                shutil.copyfileobj(src, out, 1 << 24)
    assert os.path.getsize(out_bin + ".tmp") == 4 + n_rec * rec_bytes
    os.replace(out_bin + ".tmp", out_bin)
    with open(os.path.join(args.out_dir, "val.idx"), "wb") as f:
        f.write(struct.pack("<I", len(keep)))
        for p in keep:
            f.write(struct.pack("<I", results[p]["records"]))
    rows = [row for p in keep for row in results[p]["rows"]]
    with open(os.path.join(args.out_dir, "val.json"), "w") as f:
        json.dump(rows, f)
    shutil.rmtree(tmp_dir)

    cal = [r for r in rows if r["calibrated"]]
    ratios_add = [r["sigma_add_hu"] / r["sigma_full_hu"] for r in cal]
    ratios_tot = [r["sigma_low_hu"] / r["sigma_full_hu"] for r in cal]
    per_pat_add = [results[p]["inserted_over_own_median"] for p in keep if results[p]["inserted_over_own_median"]]
    per_pat_tot = [results[p]["total_over_own_median"] for p in keep if results[p]["total_over_own_median"]]

    def spread(xs):
        xs = sorted(xs)
        return {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]} if xs else None

    manifest = {"args": vars(args), "collection": COLLECTION, "record_bytes": rec_bytes, "realisations": 1,
                "counts": {"patients": len(keep), "records": n_rec, "bytes": 4 + n_rec * rec_bytes,
                           "matched_slices": sum(results[p]["common"] for p in keep),
                           "unmatched_slices": sum(results[p]["unmatched"] for p in keep),
                           "calibrated": len(cal)},
                "dose_check": {"inserted_over_own_median": float(np.median(ratios_add)) if cal else None,
                               "total_over_own_median": float(np.median(ratios_tot)) if cal else None,
                               "predicted": {"inserted_over_own": 3.0, "total_over_own": 10 ** 0.5},
                               "per_patient_inserted_over_own": spread(per_pat_add),
                               "per_patient_total_over_own": spread(per_pat_tot)},
                "sigma_full_hu": spread([r["sigma_full_hu"] for r in cal]),
                "sigma_add_hu": spread([r["sigma_add_hu"] for r in cal]),
                "kernels": dict(collections.Counter(results[p]["kernel"] for p in keep)),
                "makers": dict(collections.Counter(results[p]["maker"] for p in keep)),
                "patients": {p: {k: v for k, v in results[p].items() if k != "rows"} for p in keep},
                "skips": {p: results[p]["skip"] for p in pids if "skip" in results[p]},
                "seconds": time.time() - t0}
    with open(os.path.join(args.out_dir, "manifest_mayo.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for p in pids:
        if "trace" in results[p]:
            print(f"--- {p}\n{results[p]['trace']}", file=sys.stderr)
    print(json.dumps({k: manifest[k] for k in ("counts", "dose_check", "sigma_full_hu", "sigma_add_hu", "kernels", "skips")}), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min → {out_bin}", flush=True)


if __name__ == "__main__":
    main()
