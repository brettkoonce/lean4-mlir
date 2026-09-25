"""Shared pieces of the gravitational-wave demo — planning/gw_detection_demo.md §2, §3.

Everything that touches GWOSC or a strain file lives here so the preprocess, the
scorer and the gate agree on one PSD convention (§3's warning: the injection SNR
and the whitening must use the SAME PSD, or every row of Table 1 shifts together).

Runs in .venv-gw (gwosc, gwpy, pycbc, lalsuite), never the pinned .venv.

  fetch(url, raw_dir)                  download once into data/gw/raw/, return the path
  event_urls(name)                     the 4096-s GWOSC files holding a catalogued event
                                       (32-s snippets stopped with GWTC-4; cut() a window)
  bulk_urls(ifo, gps_start, gps_end)   the 4096-s O3a strain files covering a range
  joint_clean_segments(start, end)     GPS stretches where H1 AND L1 are science-mode,
                                       CBC_CAT3 and free of CBC hardware injections
  pick_o3a_pairs(n)                    n (H1, L1) file-pair URLs fully inside such stretches,
                                       spread over O3a a day apart
  local_pairs()                        the pairs already under data/gw/raw/, no network
  read_gwosc_hdf5(path)                strain, gps_start, dt, DQ bits, injection bits
  cut(strain, meta, t_start, t_end)    a NaN-checked pycbc TimeSeries for a GPS range
  clean_seconds(meta)                  which seconds are science-mode and injection-free
  psd_of(strain_ts, seg=4)             median-Welch PSD (4-s segments), pycbc FrequencySeries
  condition(strain_ts)                 15-Hz highpass, 2-s crop each end
  whiten_bandpass(strain_ts, psd, ...) whitened, 20–500 Hz strain for pictures and spectrograms
"""
import json
import os
import urllib.request

import h5py
import numpy as np
from pycbc.types import TimeSeries
from pycbc.filter import highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation

FS = 4096
DEFAULT_RAW = os.path.join("data", "gw", "raw")


def fetch(url, raw_dir=DEFAULT_RAW):
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, os.path.basename(url))
    if not os.path.exists(path):
        tmp = path + ".part"
        print(f"fetch {url}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def event_urls(name, duration=4096, sample_rate=FS, ifos=("H1", "L1")):
    """GWOSC stopped producing 32-s event snippets with GWTC-4, so ask for the 4096-s
    file (a bulk-release file that happens to contain the event) and cut() from it.
    Bulk files can hold NaN gaps away from the event; cut() checks the window."""
    from gwosc.locate import get_event_urls
    urls = get_event_urls(name, duration=duration, sample_rate=sample_rate, format="hdf5")
    out = {}
    for u in urls:
        for ifo in ifos:
            if os.path.basename(u).startswith(f"{ifo[0]}-{ifo}_"):
                out[ifo] = u
    missing = [i for i in ifos if i not in out]
    if missing:
        raise RuntimeError(f"{name}: no {duration}-s snippet for {missing} in {urls}")
    return out


RUN = "O3a"
FILE_LEN = 4096
IFOS = ("H1", "L1")


def bulk_urls(ifo, gps_start, gps_end, sample_rate=FS, run=RUN):
    from gwosc.locate import get_urls
    return list(get_urls(ifo, gps_start, gps_end, sample_rate=sample_rate,
                         format="hdf5", dataset=run))


def _intersect(a, b):
    out = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return out


def _retry(fn, tries=4, wait=15.0):
    """gwosc.org drops connections under a burst of timeline queries; back off and retry."""
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  gwosc: {type(e).__name__}, retry in {wait:.0f} s", flush=True)
            time.sleep(wait)
            wait *= 2


def joint_clean_segments(gps_start, gps_end, ifos=IFOS,
                         flags=("DATA", "CBC_CAT3", "NO_CBC_HW_INJ")):
    """The GWOSC timeline's per-flag segments, intersected over both detectors."""
    from gwosc.timeline import get_segments
    segs = None
    for ifo in ifos:
        for fl in flags:
            these = _retry(lambda: [tuple(x) for x in get_segments(f"{ifo}_{fl}", gps_start, gps_end)])
            segs = these if segs is None else _intersect(segs, these)
    return sorted(segs)


def local_pairs(raw_dir=DEFAULT_RAW, run=RUN, ifos=IFOS):
    """The (slot, {ifo: path}) pairs already on disk, by slot: what pick_o3a_pairs
    chose on an earlier run, without asking the timeline again."""
    import re
    found = {}
    for name in sorted(os.listdir(raw_dir)) if os.path.isdir(raw_dir) else []:
        m = re.match(rf"([HL])-([HL]1)_GWOSC_{run}_4KHZ_R1-(\d+)-{FILE_LEN}\.hdf5$", name)
        if m and m.group(2) in ifos:
            found.setdefault(int(m.group(3)), {})[m.group(2)] = os.path.join(raw_dir, name)
    return [(slot, paths) for slot, paths in sorted(found.items()) if len(paths) == len(ifos)]


def pick_o3a_pairs(n, spread=86400, start_offset=7 * 86400, run=RUN, ifos=IFOS):
    """Walk the run a `spread`-second step at a time; in each step take the first
    4096-s file slot (on GWOSC's 4096-aligned grid) that lies wholly inside a joint
    clean segment. Returns [(slot_gps, {ifo: url})], n of them, in GPS order."""
    from gwosc.datasets import run_segment
    r0, r1 = run_segment(run)
    pairs = []
    t = r0 + start_offset
    while len(pairs) < n and t < r1:
        for a, b in joint_clean_segments(t, t + spread, ifos=ifos):
            slot = -(-a // FILE_LEN) * FILE_LEN          # first grid point >= a
            if slot + FILE_LEN <= b:
                urls = {}
                for ifo in ifos:
                    u = [x for x in bulk_urls(ifo, slot, slot + FILE_LEN, run=run)
                         if f"-{slot}-{FILE_LEN}." in x]
                    if u:
                        urls[ifo] = u[0]
                if len(urls) == len(ifos):
                    pairs.append((slot, urls))
                    break
        t += spread
    if len(pairs) < n:
        raise RuntimeError(f"only {len(pairs)} clean file pairs found in {run}")
    return pairs


def read_gwosc_hdf5(path):
    """The GWOSC HDF5 layout: strain/Strain with Xstart/Xspacing, and two per-second
    bitmasks under quality/ whose bit names are stored beside them."""
    with h5py.File(path, "r") as f:
        d = f["strain/Strain"]
        strain = d[:].astype(np.float64)
        gps_start = float(d.attrs["Xstart"])
        dt = float(d.attrs["Xspacing"])
        dq = f["quality/simple/DQmask"][:]
        dq_names = [n.decode() for n in f["quality/simple/DQShortnames"][:]]
        inj = f["quality/injections/Injmask"][:]
        inj_names = [n.decode() for n in f["quality/injections/InjShortnames"][:]]
    meta = dict(path=path, gps_start=gps_start, dt=dt, n=len(strain),
                dq=dq, dq_names=dq_names, inj=inj, inj_names=inj_names)
    return strain, meta


INJ_EXCLUDE = ("NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_DETCHAR_HW_INJ", "NO_STOCH_HW_INJ")


def clean_seconds(meta, dq_bit="DATA", cbc_cat="CBC_CAT3", inj_flags=INJ_EXCLUDE):
    """Per-second boolean: science-mode data at the given category AND no transient
    hardware injection. The injection names are NO_*_HW_INJ, so a SET bit means clean.
    NO_CW_HW_INJ is deliberately not required: the continuous-wave injections ran for
    the whole of O3 (the bit is clear in every second of every file), and they are
    narrow spectral lines the whitening removes with the instrument's own."""
    dq, names = meta["dq"], meta["dq_names"]
    ok = np.ones(len(dq), dtype=bool)
    for bit_name in (dq_bit, cbc_cat):
        if bit_name not in names:
            raise KeyError(f"{bit_name} not in DQ names {names}")
        ok &= (dq >> names.index(bit_name)) & 1 == 1
    inj, inames = meta["inj"], meta["inj_names"]
    for n in inj_flags:
        if n not in inames:
            raise KeyError(f"{n} not in injection names {inames}")
        ok &= (inj >> inames.index(n)) & 1 == 1
    return ok


def to_ts(strain, meta):
    return TimeSeries(strain, delta_t=meta["dt"], epoch=meta["gps_start"])


def cut(strain, meta, t_start, t_end):
    """The samples in [t_start, t_end) as a TimeSeries; refuses a window with a gap."""
    i0 = int(round((t_start - meta["gps_start"]) / meta["dt"]))
    i1 = int(round((t_end - meta["gps_start"]) / meta["dt"]))
    if i0 < 0 or i1 > len(strain):
        raise ValueError(f"[{t_start}, {t_end}) outside {os.path.basename(meta['path'])}")
    seg = strain[i0:i1]
    n_nan = int(np.isnan(seg).sum())
    if n_nan:
        raise ValueError(f"{n_nan} NaN samples in [{t_start}, {t_end}) of "
                         f"{os.path.basename(meta['path'])}")
    return TimeSeries(seg, delta_t=meta["dt"], epoch=t_start)


def condition(ts, f_high=15.0, crop=2.0):
    ts = highpass(ts, f_high)
    return ts.crop(crop, crop)


def psd_of(ts, seg=4.0, low_frequency_cutoff=15.0, for_delta_f=None):
    """Median-averaged Welch on `seg`-second segments — median so a glitch does not
    bias it — interpolated to the data's frequency grid and truncated to a `seg`-s
    filter so whitening by it does not ring. Returns a pycbc FrequencySeries."""
    psd = ts.psd(seg, avg_method="median")
    psd = interpolate(psd, for_delta_f if for_delta_f is not None else ts.delta_f)
    psd = inverse_spectrum_truncation(psd, int(seg * ts.sample_rate),
                                      low_frequency_cutoff=low_frequency_cutoff)
    return psd


def whiten_bandpass(ts, psd, f_lo=20.0, f_hi=500.0):
    """Divide the strain by sqrt(S_n) in the frequency domain, then band-pass.
    Unit variance per sample only inside the band (§11's second note)."""
    from pycbc.filter import lowpass_fir, highpass_fir
    white = (ts.to_frequencyseries() / psd ** 0.5).to_timeseries()
    white = highpass_fir(white, f_lo, 512)
    white = lowpass_fir(white, f_hi, 512)
    return white


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
