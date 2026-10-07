#!/usr/bin/env python3
"""Pre-process LIDC-IDRI CT to full-dose / low-dose slice pairs for the CT denoising demo
(planning/demo_slate.md §3).

Usage: .venv-lidc/bin/python scripts/datasets/preprocess_lidc.py <dicom_dir> <xml_dir> <out_dir> [options]

  <dicom_dir>  data/lidc/dicom — one directory per patient, that patient's CT series as flat
               .dcm files with the .done record scripts/datasets/lidc_fetch.py writes
  <xml_dir>    data/lidc/xml — TCIA's LIDC-XML-only.zip unpacked: one XML per series, the four
               readers' unblinded nodule outlines and characteristics
  <out_dir>    data/lidc

Options:
  --val-frac 0.2        patient-wise split, seeded
  --val-stride 4        val keeps every nodule-bearing slice of its volumes plus every 4th slice,
                        in z order: the nodules are the instrument, the stride is for PSNR/SSIM
  --train-slices 40     slices per training patient: every nodule-bearing slice first (evenly
                        thinned if there are more than this), then evenly spaced slices to fill
  --realisations 2      independent low-dose realisations per slice; Noise2Noise trains on two
  --angles 720          parallel-beam views over 180° for the projector
  --dose 0.25           the simulated low-dose fraction (quarter dose) of EACH slice's own dose
  --i0 0                full-dose photons per detector element, unattenuated; 0 (the default)
                        calibrates it per slice so the added noise is what the dose fraction
                        implies for that slice's own noise (below); --i0-ref is the reference
  --i0-ref 1e5          the calibration's reference I0
  --sigma-e 0           electronic noise: std, in counts, of a zero-mean Gaussian on the detector;
                        off by default, since I0 is a calibrated effective quantity here
  --filter shepp-logan  the FBP apodisation (scikit-image's names: ramp, shepp-logan, cosine,
                        hamming, hann); the clinical kernels are apodised ramps
  --seed 0
  --workers 30          patients processed in parallel, one core each (numpy is single-threaded here)
  --patients N          only the first N landed patients, sorted by ID: smoke tests
  --size 512            every LIDC CT is 512×512; a volume that is not is skipped and reported

Writes train.bin, val.bin, each a uint32 count then per record
    full:  S·S uint16      HU + 1024, clipped to 0..4095, row-major — the full-dose image as scanned
    low:   N × S·S uint16  the same encoding after the simulated dose reduction, N independent
                           realisations (--realisations) in order
    mask:  S·S uint8       how many of the four readers outlined a nodule at the pixel (0..4)
  = (3 + 2N)·S² bytes per record: 1,835,008 at S = 512, N = 2; the Lean loader checks the file
  length against count · (3 + 2N)·S². Beside them: train.idx / val.idx (uint32 volume count,
  then one uint32 slice count per volume, in file order — val is written volume by volume in z
  order so per-volume scoring can walk it), train.json / val.json (one row per record: patient, slice
  index, z, spacing, thickness, kernel, kVp, mA, maker, the nodule pixel count, the added
  noise's std in HU over the reconstruction circle, the scan's calibrated I0 and the two
  noise levels it came from), nodules.json (one row per reader per
  nodule: slices, centroid, area and equivalent diameter at the largest outline, the nine
  characteristics when the reader gave them, the <3 mm flag), split.json and
  manifest_lidc.json (every parameter above, the counts, the skips).

The low-dose simulation, and why it is done this way.

  Low-dose CT noise is not image-domain Gaussian: photon starvation along long paths makes
  correlated streaks after filtered back-projection. The field's reference pairs, Mayo's
  LDCT-and-Projection-data on TCIA, are themselves made by inserting noise in the projection
  domain at 25% (abdomen) or 10% (chest) of the routine dose. This is the same construction
  on LIDC, whose instrument (four readers' nodule outlines) Mayo lacks:

    μ    = μ_water · (1 + HU / 1000), clipped at 0, zero outside the reconstruction circle
    p    = R μ · Δ                       line integrals (mm⁻¹ · mm); R the parallel-beam Radon
                                         transform at --angles views, Δ the pixel spacing
    I    = I0_add · exp(−p)              expected counts for the noise to be ADDED
    N    = Poisson(I) + Normal(0, σ_e)   detected counts
    n    = −ln(max(N, 1) / I0_add) − p   the extra projection noise
    ΔHU  = FBP(n / Δ) / μ_water · 1000   filtered back-projection (--filter)
    low  = HU + ΔHU

  The scanned image already carries full-dose noise, variance ∝ 1/I0. Adding independent noise
  at I0_add brings the total to the low-dose level when 1/I0_add = 1/(dose·I0) − 1/I0, that is
  I0_add = I0 · dose / (1 − dose) = I0/3 at quarter dose. FBP is linear, so the noise image is
  the FBP of the noise sinogram alone and the full-dose target is the scanned image itself: its
  resolution is untouched, and the projector's own discretisation blur (0.4% at 720 views)
  never enters the pair.

  I0 is not a constant of the field: it is the dose, and LIDC spans 40 to 500 mAs on four
  makers' scanners, each modulating its tube current slice by slice to hold the noise level.
  At the textbook 10⁵ photons per ray a 512-pixel parallel-beam reconstruction is far noisier
  than any clinical quarter-dose chest (100-300 HU added in soft tissue, against 40-60), and
  one I0 per scan leaves the shoulder-level slices, whose paths are longest, at five to eight
  times their own noise while the mid-chest sits at target. So I0 is calibrated per slice, as
  the scanner's modulation would set it. The slice's own noise σ_full is the median 7×7 local
  standard deviation over homogeneous soft tissue (windows wholly within −200..300 HU with
  mean −50..150); the added noise must measure √(1/dose − 1)·σ_full by the same estimator
  (√3·σ_full at quarter dose: total variance four times the slice's own); FBP is linear and
  the Poisson variance is 1/I, so one realisation at --i0-ref measures σ_ref and
  I0_add = I0_ref·(σ_ref/target)² exactly, up to the photon-starvation floor. A slice with
  too little flat soft tissue to measure (under 500 windows: the apices, a few bases) takes its
  volume's median I0_add. σ_full, the target, the achieved σ_add and I0_add are in every row;
  the per-patient medians are in manifest_lidc.json. The same estimator on a Mayo chest pair
  (LDCT-and-Projection-data C002, Siemens B50f, 10% dose) gives inserted/own 2.99 against the
  3.00 its dose predicts, and total/own 3.15 against √10 = 3.16. --i0 > 0 skips the
  calibration and uses that value for every slice. Simplifications, named: parallel-beam
  rather than fan or helical geometry; no bowtie filter, so the periphery is noisier than a
  real scanner's; one μ_water (0.019 mm⁻¹, about 70 keV) and no beam hardening; one
  apodisation for every scan where the scanners each had their own kernel.

  HU below −1024 (the padding outside the scanned circle; GE writes −2000 raw) become air.
  Slices are deduplicated by z position; a series whose .dcm count disagrees with TCIA's
  ImageCount is used as it landed. The XML is matched by SeriesInstanceUid, then by
  StudyInstanceUID for the headers that lack it; when two files match, the one with more
  reading sessions is taken.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing
import random
import shutil
import struct
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pydicom
from skimage.draw import disk as draw_disk
from skimage.draw import polygon as draw_polygon
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from skimage.transform import iradon, radon

MU_WATER = 0.019  # mm^-1, about 70 keV
HU_MIN, HU_MAX = -1024, 3071
CHARS = ("subtlety", "internalStructure", "calcification", "sphericity", "margin",
         "lobulation", "spiculation", "texture", "malignancy")
META_KEYS = ("spacing", "thickness", "kernel", "kvp", "ma", "exposure", "maker", "model")

XML = None  # (by_series, by_study), set before the Pool forks so the workers inherit it


# ---- XML: the four readers' outlines -------------------------------------------------------

def strip(tag):
    return tag.rsplit("}", 1)[-1]


def text(el, name):
    if el is None:
        return None
    for c in el:
        if strip(c.tag) == name:
            return (c.text or "").strip()
    return None


def parse_xml_dir(xml_dir):
    """series_uid -> [(path, sessions)] and study_uid -> the same; sessions = one list of
    nodules per reading session, each nodule its characteristics and its per-slice ROIs."""
    by_series, by_study, n = {}, {}, 0
    for dp, _, fs in os.walk(xml_dir):
        for f in sorted(fs):
            if not f.lower().endswith(".xml"):
                continue
            path = os.path.join(dp, f)
            try:
                root = ET.parse(path).getroot()
            except ET.ParseError as e:
                print(f"  xml parse error {path}: {e}", file=sys.stderr)
                continue
            if strip(root.tag) != "LidcReadMessage":
                continue
            hdr = next((c for c in root if strip(c.tag) == "ResponseHeader"), None)
            series, study = text(hdr, "SeriesInstanceUid"), text(hdr, "StudyInstanceUID")
            sessions = []
            for rs in root:
                if strip(rs.tag) != "readingSession":  # CXRreadingSession is the radiographs
                    continue
                nods = []
                for nod in rs:
                    if strip(nod.tag) != "unblindedReadNodule":
                        continue
                    ch = next((c for c in nod if strip(c.tag) == "characteristics"), None)
                    chars = {k: int(float(v)) for k in CHARS if (v := text(ch, k))} if ch is not None else {}
                    rois = []
                    for roi in nod:
                        if strip(roi.tag) != "roi":
                            continue
                        pts = [(int(text(e, "xCoord")), int(text(e, "yCoord")))
                               for e in roi if strip(e.tag) == "edgeMap"]
                        rois.append({"sop": text(roi, "imageSOP_UID"),
                                     "z": float(text(roi, "imageZposition") or "nan"),
                                     "inclusion": (text(roi, "inclusion") or "TRUE").upper() == "TRUE",
                                     "pts": pts})
                    nods.append({"id": text(nod, "noduleID"), "chars": chars, "rois": rois})
                sessions.append({"radiologist": text(rs, "servicingRadiologistID"), "nodules": nods,
                                 "non_nodules": sum(1 for c in rs if strip(c.tag) == "nonNodule")})
            rec = (path, sessions)
            if series:
                by_series.setdefault(series, []).append(rec)
            if study:
                by_study.setdefault(study, []).append(rec)
            n += 1
    return by_series, by_study, n


def nodule_masks(sessions, size, zs, sops, spacing, dz):
    """The per-slice reader-count mask, uint8 [Z, S, S], and one table row per reader per nodule."""
    Z, S = len(zs), size
    count = np.zeros((Z, S, S), np.uint8)
    sop_index = {s: i for i, s in enumerate(sops)}
    r_small = max(1, int(round(1.5 / spacing)))  # a <3 mm nodule is one point; a 3 mm disk stands in
    rows = []
    for r, sess in enumerate(sessions):
        reader = np.zeros((Z, S, S), bool)
        for nod in sess["nodules"]:
            m = np.zeros((Z, S, S), bool)
            touched = set()
            # inclusions first, then the exclusions cut out of them
            for roi in sorted(nod["rois"], key=lambda ro: not ro["inclusion"]):
                zi = sop_index.get(roi["sop"])
                if zi is None:
                    if not np.isfinite(roi["z"]) or Z == 0:
                        continue
                    zi = int(np.argmin(np.abs(zs - roi["z"])))
                    if abs(zs[zi] - roi["z"]) > max(0.5 * dz, 0.01):
                        continue
                pts = roi["pts"]
                if not pts:
                    continue
                if len(pts) < 3:
                    for x, y in pts:
                        rr, cc = draw_disk((y, x), r_small, shape=(S, S))
                        m[zi, rr, cc] = roi["inclusion"]
                else:
                    xs = np.array([p[0] for p in pts])
                    ys = np.array([p[1] for p in pts])
                    rr, cc = draw_polygon(ys, xs, shape=(S, S))
                    m[zi, rr, cc] = roi["inclusion"]
                    if roi["inclusion"]:  # the outline pixels belong to the nodule too
                        m[zi, np.clip(ys, 0, S - 1), np.clip(xs, 0, S - 1)] = True
                touched.add(zi)
            if not touched or not m.any():
                continue
            areas = m.reshape(Z, -1).sum(1)
            zbest = int(areas.argmax())
            area_mm2 = float(areas[zbest]) * spacing * spacing
            zz, yy, xx = np.nonzero(m)
            rows.append({"reader": r, "radiologist": sess["radiologist"], "nodule": nod["id"],
                         "small": all(len(ro["pts"]) < 3 for ro in nod["rois"]),
                         "slices": sorted(int(t) for t in touched),
                         "z_mm": [float(zs[min(touched)]), float(zs[max(touched)])],
                         "centroid": [float(xx.mean()), float(yy.mean()), float(zs[zz].mean())],
                         "area_px": int(areas[zbest]), "area_mm2": area_mm2,
                         "diam_mm": 2.0 * (area_mm2 / np.pi) ** 0.5, "voxels": int(len(zz)),
                         **nod["chars"]})
            reader |= m
        count += reader.astype(np.uint8)
    return count, rows


# ---- DICOM -----------------------------------------------------------------------------------

def read_volume(pdir, size):
    """(vol_hu float32 [Z,S,S], zs, sops, meta) sorted by z, duplicates dropped; or (None, why)."""
    hdrs = []
    for f in os.listdir(pdir):
        if not f.endswith(".dcm"):
            continue
        path = os.path.join(pdir, f)
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if str(getattr(ds, "Modality", "CT")) != "CT":
            continue
        hdrs.append((float(ds.ImagePositionPatient[2]), int(getattr(ds, "InstanceNumber", 0) or 0), path))
    if not hdrs:
        return None, "no CT slices"
    hdrs.sort()
    uniq = [h for i, h in enumerate(hdrs) if i == 0 or abs(h[0] - hdrs[i - 1][0]) > 1e-3]
    ds0 = pydicom.dcmread(uniq[0][2], stop_before_pixels=True)
    if int(ds0.Rows) != size or int(ds0.Columns) != size:
        return None, f"{int(ds0.Rows)}x{int(ds0.Columns)}, not {size}"
    vol = np.empty((len(uniq), size, size), np.float32)
    sops = []
    for i, (_, _, path) in enumerate(uniq):
        ds = pydicom.dcmread(path)
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) \
            + float(getattr(ds, "RescaleIntercept", 0) or 0)
        hu[hu < HU_MIN] = -1000.0  # padding outside the scanned circle is air
        vol[i] = np.clip(hu, HU_MIN, HU_MAX)
        sops.append(str(ds.SOPInstanceUID))
    zs = np.array([h[0] for h in uniq])
    meta = {"spacing": float(ds0.PixelSpacing[0]),
            "thickness": float(getattr(ds0, "SliceThickness", 0) or 0),
            "kernel": str(getattr(ds0, "ConvolutionKernel", "")),
            "kvp": float(getattr(ds0, "KVP", 0) or 0),
            "ma": float(getattr(ds0, "XRayTubeCurrent", 0) or 0),
            "exposure": float(getattr(ds0, "Exposure", 0) or 0),
            "maker": str(getattr(ds0, "Manufacturer", "")),
            "model": str(getattr(ds0, "ManufacturerModelName", "")),
            "series": str(ds0.SeriesInstanceUID), "study": str(ds0.StudyInstanceUID),
            "dz": float(np.median(np.diff(zs))) if len(zs) > 1 else 0.0,
            "dropped_duplicates": len(hdrs) - len(uniq)}
    return (vol, zs, sops, meta), None


# ---- the low-dose simulation ----------------------------------------------------------------

class Projector:
    def __init__(self, size, angles, filter_name):
        self.theta = np.linspace(0.0, 180.0, angles, endpoint=False)
        self.filter_name = filter_name
        c = size // 2
        yy, xx = np.ogrid[:size, :size]
        self.inside = ((yy - c) ** 2 + (xx - c) ** 2) <= c * c  # scikit-image's reconstruction circle

    def sinogram(self, hu, spacing):
        """p = R μ · Δ, float32 [S, views]: the one expensive transform per slice."""
        mu = MU_WATER * (1.0 + hu.astype(np.float64) / 1000.0)
        mu = np.where(self.inside, np.maximum(mu, 0.0), 0.0)
        return (radon(mu, theta=self.theta, circle=True) * spacing).astype(np.float32)

    def noise_hu(self, p, spacing, i0_add, sigma_e, rng):
        """One realisation of the added noise, in HU, from the sinogram p."""
        p = p.astype(np.float64)
        counts = rng.poisson(i0_add * np.exp(-p)).astype(np.float64)
        if sigma_e > 0:
            counts += rng.normal(0.0, sigma_e, p.shape)
        n = -np.log(np.maximum(counts, 1.0) / i0_add) - p
        d_mu = iradon(n / spacing, theta=self.theta, circle=True, filter_name=self.filter_name,
                      output_size=self.inside.shape[0])
        return (d_mu / MU_WATER * 1000.0).astype(np.float32)


# ---- calibrating I0 to the slice's own noise ---------------------------------------------------

CAL_WIN = 7
CAL_MIN_PIXELS = 500


def local_std(img):
    m = uniform_filter(img, CAL_WIN)
    m2 = uniform_filter(img * img, CAL_WIN)
    return np.sqrt(np.maximum(m2 - m * m, 0.0) * (CAL_WIN ** 2 / (CAL_WIN ** 2 - 1.0)))


def flat_soft_tissue(hu, inside):
    """Pixels whose 7×7 window is homogeneous soft tissue: every value in −200..300 HU, mean in −50..150."""
    mean = uniform_filter(hu, CAL_WIN)
    return (inside & (maximum_filter(hu, CAL_WIN) <= 300) & (minimum_filter(hu, CAL_WIN) >= -200)
            & (mean >= -50) & (mean <= 150))


def noise_level(img, flat):
    """The median 7×7 local std over the flat soft-tissue pixels: the one estimator used on both
    the scan's own noise and the added noise."""
    return float(np.median(local_std(img.astype(np.float64))[flat]))


def encode(hu):
    return np.clip(np.rint(hu) + 1024.0, 0, 4095).astype("<u2")


def select_slices(mask, split, train_slices, val_stride):
    Z = mask.shape[0]
    nod = [i for i in range(Z) if mask[i].any()]
    if split == "val":
        return sorted(set(nod) | set(range(0, Z, max(1, val_stride))))
    k = min(len(nod), train_slices)
    sel = set(np.array(nod)[np.linspace(0, len(nod) - 1, k).round().astype(int)].tolist()) if k else set()
    want = train_slices - len(sel)
    others = [i for i in range(Z) if i not in sel]
    if want > 0 and others:
        pick = np.linspace(0, len(others) - 1, min(want, len(others))).round().astype(int)
        sel |= {others[i] for i in pick}
    return sorted(sel)


def process_patient(job):
    pid, pdir, split, a, tmp_dir = job
    t0 = time.time()
    try:
        res, err = read_volume(pdir, a["size"])
        if err:
            return {"patient": pid, "split": split, "skip": err}
        vol, zs, sops, meta = res
        by_series, by_study = XML
        recs = by_series.get(meta["series"]) or by_study.get(meta["study"]) or []
        if len(recs) > 1:
            recs = [max(recs, key=lambda r: len(r[1]))]
        sessions = recs[0][1] if recs else []
        mask, nodules = nodule_masks(sessions, a["size"], zs, sops, meta["spacing"], meta["dz"])
        for row in nodules:
            row["patient"] = pid
        sel = select_slices(mask, split, a["train_slices"], a["val_stride"])
        proj = Projector(a["size"], a["angles"], a["filter"])
        pnum = int(pid.rsplit("-", 1)[-1])
        spacing = meta["spacing"]
        gain = (1.0 / a["dose"] - 1.0) ** 0.5          # √3 at quarter dose
        i0_ref_add = a["i0_ref"] * a["dose"] / (1.0 - a["dose"])
        # pass 1: the sinogram of every selected slice, and I0 where the slice's own noise is measurable
        sinos, flats, own, i0 = {}, {}, {}, {}
        for zi in sel:
            hu = vol[zi]
            sinos[zi] = proj.sinogram(hu, spacing)
            if a["i0"] > 0:
                i0[zi] = a["i0"] * a["dose"] / (1.0 - a["dose"])
                continue
            flat = flat_soft_tissue(hu.astype(np.float64), proj.inside)
            if flat.sum() < CAL_MIN_PIXELS:
                continue
            flats[zi] = flat
            own[zi] = noise_level(hu, flat)
            rng = np.random.default_rng([a["seed"], pnum, 1_000_000 + zi])
            s_ref = noise_level(proj.noise_hu(sinos[zi], spacing, i0_ref_add, a["sigma_e"], rng), flat)
            i0[zi] = float(np.clip(i0_ref_add * (s_ref / (gain * own[zi])) ** 2, 1e2, 1e12))
        fallback = float(np.median(list(i0.values()))) if i0 else i0_ref_add
        # pass 2: the pairs
        rows = []
        with open(os.path.join(tmp_dir, pid + ".bin"), "wb") as f:
            for zi in sel:
                hu = vol[zi]
                i0_add = i0.get(zi, fallback)
                f.write(encode(hu).tobytes())
                for k in range(a["realisations"]):
                    rng = np.random.default_rng([a["seed"], pnum, zi, k])
                    d_k = proj.noise_hu(sinos[zi], spacing, i0_add, a["sigma_e"], rng)
                    f.write(encode(hu + d_k).tobytes())
                    if k == 0:
                        d_hu = d_k
                f.write(mask[zi].tobytes())
                s_full = own.get(zi, float("nan"))
                rows.append({"patient": pid, "slice": int(zi), "z_mm": float(zs[zi]),
                             "nodule_px": int(np.count_nonzero(mask[zi])),
                             "noise_hu_std": float(d_hu[proj.inside].std()),
                             "i0_add": i0_add, "calibrated": zi in own, "sigma_full_hu": s_full,
                             "sigma_add_target_hu": s_full * gain,
                             "sigma_add_hu": noise_level(d_hu, flats[zi]) if zi in own else float("nan"),
                             **{k: meta[k] for k in META_KEYS}})
        del sinos
        ratios = [r["sigma_add_hu"] / r["sigma_add_target_hu"] for r in rows if r["calibrated"]]
        calib = {"i0_add_median": float(np.median([r["i0_add"] for r in rows])) if rows else None,
                 "sigma_full_hu_median": float(np.median(list(own.values()))) if own else None,
                 "calibrated_slices": len(own), "achieved_over_target_median": float(np.median(ratios)) if ratios else None}
        return {"patient": pid, "split": split, "slices": len(sel), "volume_slices": int(len(zs)),
                "nodule_slices": int(sum(1 for i in range(len(zs)) if mask[i].any())),
                "dropped_duplicates": meta["dropped_duplicates"], "readers": len(sessions),
                **calib,
                "xml": os.path.relpath(recs[0][0], a["xml_dir"]) if recs else None,
                "nodules": nodules, "rows": rows, "seconds": time.time() - t0}
    except Exception as e:  # one bad volume must not take the pool down
        return {"patient": pid, "split": split, "skip": f"error: {e!r}", "trace": traceback.format_exc()}


# ---- driver ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dicom_dir")
    ap.add_argument("xml_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-stride", type=int, default=4)
    ap.add_argument("--train-slices", type=int, default=40)
    ap.add_argument("--realisations", type=int, default=2)
    ap.add_argument("--angles", type=int, default=720)
    ap.add_argument("--dose", type=float, default=0.25)
    ap.add_argument("--i0", type=float, default=0.0)
    ap.add_argument("--i0-ref", type=float, default=1e5)
    ap.add_argument("--sigma-e", type=float, default=0.0)
    ap.add_argument("--filter", default="shepp-logan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--patients", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    t0 = time.time()

    landed = sorted(p for p in os.listdir(args.dicom_dir)
                    if os.path.exists(os.path.join(args.dicom_dir, p, ".done")))
    if args.patients:
        landed = landed[: args.patients]
    if not landed:
        sys.exit("no landed patients (no .done records) under " + args.dicom_dir)
    order = landed[:]
    random.Random(args.seed).shuffle(order)
    n_val = int(round(len(landed) * args.val_frac)) if len(landed) > 1 else 0
    split = {"train": sorted(order[n_val:]), "val": sorted(order[:n_val])}
    val_set = set(split["val"])

    global XML
    by_series, by_study, n_xml = parse_xml_dir(args.xml_dir)
    XML = (by_series, by_study)
    i0_note = (f"I0_add fixed at {args.i0 * args.dose / (1 - args.dose):.3g}" if args.i0 > 0
               else f"I0 calibrated per slice against {args.i0_ref:.3g}")
    print(f"{len(landed)} patients ({len(split['train'])} train / {n_val} val), {n_xml} XML files "
          f"({len(by_series)} series, {len(by_study)} studies); dose {args.dose}, {i0_note}, "
          f"{args.angles} views, {args.filter}, {args.realisations} realisations", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "tmp_preprocess")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    a = dict(vars(args))
    jobs = [(pid, os.path.join(args.dicom_dir, pid), "val" if pid in val_set else "train", a, tmp_dir)
            for pid in landed]
    results = {}
    ctx = multiprocessing.get_context("fork")  # the workers inherit XML
    with ctx.Pool(min(args.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(process_patient, jobs), 1):
            results[r["patient"]] = r
            el = time.time() - t0
            if "skip" in r:
                print(f"[{i}/{len(jobs)}] {r['patient']} SKIP {r['skip']}", flush=True)
            else:
                sf, ratio = r["sigma_full_hu_median"], r["achieved_over_target_median"]
                cal = (f"σ_full {sf:.1f} HU, achieved/target {ratio:.3f} on {r['calibrated_slices']} slices, "
                       f"I0_add {r['i0_add_median']:.3g}" if sf is not None else f"I0_add {r['i0_add_median']:.3g}")
                print(f"[{i}/{len(jobs)}] {r['patient']} {r['split']} {r['slices']}/{r['volume_slices']} slices, "
                      f"{r['nodule_slices']} with nodules, {r['readers']} readers, {cal}, {r['seconds']:.0f}s"
                      f"  ({el / 60:.0f} min, ETA {el / i * (len(jobs) - i) / 60:.0f} min)", flush=True)

    rec_bytes = (3 + 2 * args.realisations) * args.size * args.size
    counts = {}
    for name, pids in split.items():
        keep = [p for p in pids if "skip" not in results[p]]
        n_rec = sum(results[p]["slices"] for p in keep)
        out_bin = os.path.join(args.out_dir, f"{name}.bin")
        with open(out_bin + ".tmp", "wb") as out:
            out.write(struct.pack("<I", n_rec))
            for p in keep:  # each temp file goes as soon as it is copied: the disk holds one output, not two
                with open(os.path.join(tmp_dir, p + ".bin"), "rb") as src:
                    shutil.copyfileobj(src, out, 1 << 24)
                os.remove(os.path.join(tmp_dir, p + ".bin"))
        assert os.path.getsize(out_bin + ".tmp") == 4 + n_rec * rec_bytes
        os.replace(out_bin + ".tmp", out_bin)
        with open(os.path.join(args.out_dir, f"{name}.idx"), "wb") as f:
            f.write(struct.pack("<I", len(keep)))
            for p in keep:
                f.write(struct.pack("<I", results[p]["slices"]))
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump([row for p in keep for row in results[p]["rows"]], f)
        counts[name] = {"patients": len(keep), "records": n_rec, "bytes": 4 + n_rec * rec_bytes}
    shutil.rmtree(tmp_dir)

    nodules = [row for r in results.values() for row in r.get("nodules", [])]
    with open(os.path.join(args.out_dir, "nodules.json"), "w") as f:
        json.dump(nodules, f)
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=1)
    skips = {p: r["skip"] for p, r in results.items() if "skip" in r}
    for p, r in results.items():
        if "trace" in r:
            print(f"--- {p}\n{r['trace']}", file=sys.stderr)
    readers = {}
    for r in results.values():
        if "skip" not in r:
            readers[r["readers"]] = readers.get(r["readers"], 0) + 1
    ok = [r for r in results.values() if "skip" not in r and r["rows"]]
    allrows = [row for r in ok for row in r["rows"]]
    cal_rows = [row for row in allrows if row["calibrated"]]
    def spread(xs):
        xs = sorted(xs)
        return {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1]} if xs else None
    manifest = {"args": a, "mu_water_per_mm": MU_WATER, "record_bytes": rec_bytes, "counts": counts,
                "slices": {"total": len(allrows), "calibrated": len(cal_rows)},
                "i0_add": spread(row["i0_add"] for row in allrows),
                "sigma_full_hu": spread(row["sigma_full_hu"] for row in cal_rows),
                "achieved_over_target": spread(row["sigma_add_hu"] / row["sigma_add_target_hu"] for row in cal_rows),
                "uncalibrated_patients": [r["patient"] for r in ok if args.i0 <= 0 and r["calibrated_slices"] == 0],
                "calibration": {r["patient"]: {k: r[k] for k in ("i0_add_median", "sigma_full_hu_median",
                                                                 "calibrated_slices", "achieved_over_target_median")}
                                for r in ok},
                "nodules": {"rows": len(nodules), "small": sum(1 for n in nodules if n["small"]),
                            "patients_with_nodules": len({n["patient"] for n in nodules})},
                "readers_per_patient": readers, "xml_files": n_xml, "skips": skips,
                "unmatched_xml": [p for p, r in results.items() if "skip" not in r and r["xml"] is None],
                "seconds": time.time() - t0}
    with open(os.path.join(args.out_dir, "manifest_lidc.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("counts", "slices", "nodules", "readers_per_patient", "i0_add",
                                               "sigma_full_hu", "achieved_over_target", "uncalibrated_patients",
                                               "skips", "unmatched_xml")}), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
