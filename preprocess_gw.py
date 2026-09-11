#!/usr/bin/env python3
"""Build the gravitational-wave detection sets — planning/gw_detection_demo.md §3.

Real O3a strain from GWOSC, cut into 2-s H1+L1 windows, half of them carrying a
binary-black-hole chirp injected at a chosen network SNR. Two noise sets share every
window position and every injection: `gauss` is Gaussian noise coloured by the file's
own PSD (the theorem's regime), `real` is the strain itself. Written per split:

  data/gw/{gauss,real}_{train,val}.bin   f32 [N, 2, 64, 128]  log(power / band median)
                                         of a 64-band constant-Q (Morlet, Q = 8)
                                         filterbank, 20-500 Hz log-spaced, hop 64
  data/gw/labels_{train,val}.bin         int32, 1 = injected
  data/gw/meta_{train,val}.npz           per window: file slot, GPS start, the injection's
                                         masses / sky / t_c / per-detector and network SNR /
                                         distance, the matched-filter statistics (below)
  data/gw/examples.npz                   a few injected val windows in the time domain
  data/gw/manifest.json                  geometry, priors, file list, per-file diagnostics

One PSD per detector per file (median Welch, 4 s) does everything: colours the Gaussian
twin, whitens both noise sets, whitens every injection and its template. Whitening is
linear, so window = whitened noise + whitened signal, and no injection leaks into a
neighbouring window. SNR is the optimal SNR in the WHITENED domain, pycbc.filter.sigma
against the measured PSD of the whitened Gaussian twin (flat in band), over 20-500 Hz;
sigma of the raw signal against the raw PSD is stored beside it as a diagnostic, and
the two agree to a constant if the conventions do (§3's warning made a number).

The matched-filter rows of Table 1 are computed here, since the templates are in hand:
for every window, pycbc.filter.matched_filter with the whitened template of the
window's own chirp (a random draw from the prior for a noise-only window), max |rho|
over the window per detector (`mf_<set>_<ifo>_max`) and within 2 ms of the detector-
frame coalescence time (`mf_<set>_<ifo>_tc`; a noise-only window uses the time and sky
position it drew, so the known-time rows have a noise-only distribution too); the
coherent network search statistic (`mf_<set>_net_max`, max over time and the +-10 ms
delay) and the known-time, known-delay quadrature sum (`mf_<set>_netk_tc`). The `_z0`
twins are the SINGLE sample at the detector-frame time, no maximisation at all: the
closed form's exact regime, and what Gate 1 checks (the +-2 ms max hands a signal
~0.2 of SNR that the noise-set threshold does not see).

  .venv-gw/bin/python preprocess_gw.py [--pairs=26] [--val-pairs=6] [--workers=8]
                                       [--out=data/gw] [--smoke]
  .venv-gw/bin/python preprocess_gw.py --examples=14,18     # only rewrite examples.npz, from
                                                           # the first val pair, at that SNR
"""
import gc
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# one thread per worker: the work is 8k-point FFTs, and 13 workers each opening a
# 32-thread BLAS pool put this box at a load of 330 and got nothing done
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import gw_common as gw  # noqa: E402

FS = gw.FS
IFOS = gw.IFOS
WIN = 2.0
N_WIN = int(WIN * FS)                    # 8192
PAD = 16.0                               # template whitening pad
PAD_OFF = 7.0
N_PAD = int(PAD * FS)
EDGE = 8.0                               # seconds dropped at each end of the whitened file
BAND = (20.0, 500.0)
ROLL = 5.0                               # raised-cosine edge width of the band mask
PSD_SEG = 4.0
HIGHPASS = 15.0
MASS = (10.0, 50.0)
SNR = (4.0, 20.0)
TC = (0.7, 1.8)
F_LOWER = 20.0
APPROX = "IMRPhenomD"
N_BANDS, Q, HOP = 64, 8.0, 64
N_FRAMES = N_WIN // HOP                  # 128
TAPER = int(0.1 * FS)
TC_TOL = int(0.002 * FS)                 # around the DETECTOR-frame coalescence time
EXAMPLE_SNR = [8.0, 12.0]                # network SNR range of the figure's example windows
DELAY_TOL = int(0.010 * FS)              # H1-L1 light travel: the coherent statistic's delay range
LOG_FLOOR = -5.0
SEED = 20260911


# ── whitening ──────────────────────────────────────────────────────────────────
def whitening_filter(psd_raw, n, fs=FS, seg=PSD_SEG, low=HIGHPASS, band=BAND, roll=ROLL):
    """rfft-bin array W(f) = B(f) / sqrt(S_trunc(f)) for a series of n samples: the raw
    PSD interpolated to this grid, inverse-spectrum-truncated to a `seg`-second filter
    (so whitening does not ring at the lines), times a raised-cosine band mask. The same
    W, on the same construction, whitens the file and (on a 16-s grid) every template."""
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    nb = n // 2 + 1
    df = fs / n
    psd = interpolate(psd_raw, df, length=nb)
    psd = inverse_spectrum_truncation(psd, int(seg * fs), low_frequency_cutoff=low)
    p = psd.numpy()
    f = np.arange(nb) * df
    lo, hi = band
    mask = np.zeros(nb)
    mask[(f >= lo) & (f <= hi)] = 1.0
    r = (f > lo - roll) & (f < lo)
    mask[r] = 0.5 * (1 - np.cos(np.pi * (f[r] - (lo - roll)) / roll))
    r = (f > hi) & (f < hi + roll)
    mask[r] = 0.5 * (1 + np.cos(np.pi * (f[r] - hi) / roll))
    W = np.zeros(nb)
    good = p > 0
    W[good] = mask[good] / np.sqrt(p[good])
    return W


def colour_white_noise(rng, psd_raw, n, fs=FS):
    """Gaussian noise with one-sided PSD psd_raw in pycbc's units: rfft coefficients of
    variance n S(f) fs / 2 (so a Welch estimate of the result returns S)."""
    from pycbc.psd import interpolate
    nb = n // 2 + 1
    p = interpolate(psd_raw, fs / n, length=nb).numpy()
    amp = np.sqrt(np.maximum(p, 0) * n * fs / 2)
    z = (rng.standard_normal(nb) + 1j * rng.standard_normal(nb)) / np.sqrt(2)
    return np.fft.irfft(z * amp, n)


def whiten(x, W):
    return np.fft.irfft(np.fft.rfft(x) * W, len(x))


def whiten_short(x, W16):
    """A 2-s signal whitened on the 16-s grid, so the 4-s filter response never wraps."""
    pad = np.zeros(N_PAD)
    i0 = int(PAD_OFF * FS)
    pad[i0:i0 + N_WIN] = x
    return whiten(pad, W16)[i0:i0 + N_WIN]


# ── spectrogram ────────────────────────────────────────────────────────────────
def make_bank(n_fft=2 * N_WIN, fs=FS):
    """Gaussian (Morlet) bands on the rfft grid of a zero-padded window: centres
    log-spaced over BAND, sigma_f = f / Q. Returns [N_BANDS, n_fft//2+1] and the centres."""
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    centres = np.geomspace(BAND[0], BAND[1], N_BANDS)
    G = np.exp(-0.5 * ((f[None, :] - centres[:, None]) / (centres[:, None] / Q)) ** 2)
    return G, centres


def spectrogram(x, G):
    """|analytic band signal|^2 sampled every HOP samples, as log(p / median over time).
    The band signal is decimated exactly by folding its one-sided spectrum modulo the
    decimated length (sampling in time is aliasing in frequency), so the cost is 64
    small inverse FFTs instead of 64 full ones. The window is zero-padded to 2x so the
    circular convolution never wraps the window's end into its start."""
    n_fft = 2 * N_WIN
    X = np.fft.rfft(x, n_fft)                        # [n_fft/2 + 1]
    Y = G * X[None, :]                               # one-sided, analytic
    Y[:, 0] *= 0.5
    Y[:, -1] *= 0.5
    n_dec = n_fft // HOP                             # 256 decimated samples of the 2x pad
    full = np.zeros((G.shape[0], n_fft), dtype=complex)
    full[:, :Y.shape[1]] = Y
    folded = full.reshape(G.shape[0], HOP, n_dec).sum(axis=1)     # k = c*n_dec + m -> m
    y = np.fft.ifft(folded, axis=1) * (n_dec / n_fft) * 2          # x[HOP j], analytic
    p = np.abs(y[:, :N_FRAMES]) ** 2
    med = np.median(p, axis=1, keepdims=True)
    return np.maximum(np.log(p / np.maximum(med, 1e-30) + 1e-30), LOG_FLOOR).astype(np.float32)


def spectrogram_slow(x, G):
    """The same by a full inverse FFT per band; the self-test for the fold."""
    n_fft = 2 * N_WIN
    X = np.fft.rfft(x, n_fft)
    Y = G * X[None, :]
    Y[:, 0] *= 0.5
    Y[:, -1] *= 0.5
    full = np.zeros((G.shape[0], n_fft), dtype=complex)
    full[:, :Y.shape[1]] = Y
    y = np.fft.ifft(full, axis=1) * 2
    p = np.abs(y[:, :N_WIN:HOP]) ** 2
    med = np.median(p, axis=1, keepdims=True)
    return np.maximum(np.log(p / np.maximum(med, 1e-30) + 1e-30), LOG_FLOOR).astype(np.float32)


# ── one file pair ──────────────────────────────────────────────────────────────
def place(ts, w0_gps):
    """The samples of a pycbc TimeSeries falling in [w0, w0 + 2 s), with a half-Hann
    ramp over the first 0.1 s when the waveform began before the window."""
    out = np.zeros(N_WIN)
    i0 = int(round((float(ts.start_time) - w0_gps) * FS))
    src = ts.numpy()
    a, b = max(i0, 0), min(i0 + len(src), N_WIN)
    if b > a:
        out[a:b] = src[a - i0:b - i0]
    if i0 < 0:
        out[:TAPER] *= 0.5 * (1 - np.cos(np.pi * np.arange(TAPER) / TAPER))
    return out


def process_pair(job):
    slot, paths, split, p_inj, out_dir, limit, n_examples = job
    from pycbc.detector import Detector
    from pycbc.filter import matched_filter, sigma
    from pycbc.psd import interpolate
    from pycbc.types import TimeSeries
    from pycbc.waveform import get_td_waveform

    t_start = time.time()
    rng = np.random.default_rng([SEED, slot])
    dets = {ifo: Detector(ifo) for ifo in IFOS}
    G, _ = make_bank()

    raw, meta, ok = {}, {}, None
    for ifo in IFOS:
        s, m = gw.read_gwosc_hdf5(paths[ifo])
        raw[ifo], meta[ifo] = s, m
        o = gw.clean_seconds(m)
        ok = o if ok is None else ok & o
    gps0 = meta["H1"]["gps_start"]
    assert meta["L1"]["gps_start"] == gps0, (meta["H1"]["gps_start"], meta["L1"]["gps_start"])

    diag = {}
    W, W16, white, Sw2, praw2 = {}, {}, {}, {}, {}
    for ifo in IFOS:
        ts = gw.condition(gw.to_ts(raw[ifo], meta[ifo]), f_high=HIGHPASS, crop=2.0)
        x = ts.numpy()
        if np.isnan(x).any():
            raise ValueError(f"{os.path.basename(paths[ifo])}: NaN inside the file")
        n = len(x)
        psd_raw = ts.psd(PSD_SEG, avg_method="median")
        W[ifo] = whitening_filter(psd_raw, n)
        W16[ifo] = whitening_filter(psd_raw, N_PAD)
        gauss = colour_white_noise(rng, psd_raw, n)
        white[ifo] = dict(real=whiten(x, W[ifo]), gauss=whiten(gauss, W[ifo]))
        del gauss, x, ts                      # ~0.5 GB of 16.8M-sample temporaries per detector
        raw[ifo] = None
        sw = {k: TimeSeries(v, delta_t=1 / FS).psd(PSD_SEG, avg_method="median")
              for k, v in white[ifo].items()}
        Sw2[ifo] = interpolate(sw["gauss"], 1 / WIN, length=N_WIN // 2 + 1)
        praw2[ifo] = interpolate(psd_raw, 1 / WIN, length=N_WIN // 2 + 1)
        f = sw["gauss"].sample_frequencies.numpy()
        inband = (f >= BAND[0] + ROLL) & (f <= BAND[1] - ROLL)
        g, r = sw["gauss"].numpy()[inband], sw["real"].numpy()[inband]
        diag[ifo] = dict(sw_gauss_level=float(np.median(g)),
                         sw_gauss_flatness=float(np.percentile(g, 99) / np.percentile(g, 1)),
                         sw_real_over_gauss=float(np.median(r) / np.median(g)),
                         sw_real_flatness=float(np.percentile(r, 99) / np.percentile(r, 1)))
    del raw
    gc.collect()
    start_c = gps0 + 2.0
    n_c = len(white["H1"]["real"])

    # windows on the integer-second grid, wholly inside the clean seconds of BOTH detectors
    w0s = []
    t = start_c + EDGE
    while t + WIN <= start_c + n_c / FS - EDGE:
        s = int(t - gps0)
        if ok[s] and ok[s + 1]:
            w0s.append(t)
        t += WIN
    if limit:
        w0s = w0s[:limit]
    n_w = len(w0s)

    spec = {k: np.zeros((n_w, 2, N_BANDS, N_FRAMES), dtype=np.float32) for k in ("gauss", "real")}
    labels = np.zeros(n_w, dtype=np.int32)
    cols = ("m1", "m2", "ra", "dec", "pol", "tc", "snr_target", "rho_h1", "rho_l1", "rho_net",
            "sigraw_h1", "sigraw_l1", "dist_mpc",
            "mf_gauss_h1_max", "mf_gauss_l1_max", "mf_gauss_h1_tc", "mf_gauss_l1_tc",
            "mf_real_h1_max", "mf_real_l1_max", "mf_real_h1_tc", "mf_real_l1_tc",
            "mf_gauss_h1_tmax", "mf_gauss_l1_tmax", "mf_real_h1_tmax", "mf_real_l1_tmax",
            "mf_gauss_net_max", "mf_gauss_netk_tc", "mf_real_net_max", "mf_real_netk_tc",
            "mf_gauss_h1_z0", "mf_gauss_l1_z0", "mf_real_h1_z0", "mf_real_l1_z0",
            "mf_gauss_netk_z0", "mf_real_netk_z0",
            "overlap_h1", "overlap_l1", "delay_h1", "delay_l1", "tc_h1", "tc_l1")
    M = {c: np.full(n_w, np.nan) for c in cols}
    examples = []

    for k, w0 in enumerate(w0s):
        i0 = int(round((w0 - start_c) * FS))
        inject = rng.random() < p_inj
        m1, m2 = np.sort(rng.uniform(*MASS, size=2))[::-1]
        ra, dec, pol = rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)
        tc = rng.uniform(*TC)
        M["m1"][k], M["m2"][k], M["tc"][k] = m1, m2, tc
        M["ra"][k], M["dec"][k], M["pol"][k] = ra, dec, pol
        for ifo in IFOS:
            delay = dets[ifo].time_delay_from_earth_center(ra, dec, w0 + tc)
            M[f"delay_{ifo.lower()}"][k] = delay
            M[f"tc_{ifo.lower()}"][k] = tc + delay
        hp, hc = get_td_waveform(approximant=APPROX, mass1=m1, mass2=m2, delta_t=1 / FS,
                                 f_lower=F_LOWER)
        hp.start_time += w0 + tc
        hc.start_time += w0 + tc
        tmpl_raw = place(hp, w0)
        shift = -int(round(tc * FS))
        tmpl = {ifo: TimeSeries(np.roll(whiten_short(tmpl_raw, W16[ifo]), shift), delta_t=1 / FS)
                for ifo in IFOS}
        sig = {ifo: np.zeros(N_WIN) for ifo in IFOS}
        if inject:
            labels[k] = 1
            rho, sraw = {}, {}
            for ifo in IFOS:
                h = dets[ifo].project_wave(hp, hc, ra, dec, pol)
                h_raw = place(h, w0)
                sig[ifo] = whiten_short(h_raw, W16[ifo])
                rho[ifo] = sigma(TimeSeries(sig[ifo], delta_t=1 / FS), psd=Sw2[ifo],
                                 low_frequency_cutoff=BAND[0], high_frequency_cutoff=BAND[1])
                sraw[ifo] = sigma(TimeSeries(h_raw, delta_t=1 / FS), psd=praw2[ifo],
                                  low_frequency_cutoff=BAND[0], high_frequency_cutoff=BAND[1])
            net = np.sqrt(sum(v ** 2 for v in rho.values()))
            target = rng.uniform(*SNR)
            a = target / net
            for ifo in IFOS:
                sig[ifo] *= a
                M[f"rho_{ifo.lower()}"][k] = a * rho[ifo]
                M[f"sigraw_{ifo.lower()}"][k] = a * sraw[ifo]
            M["snr_target"][k] = target
            for ifo in IFOS:     # template-signal match in the whitened domain, max over lag and phase
                q = tmpl[ifo].numpy()
                qa = np.fft.ifft(np.fft.fft(q) * 2 * (np.fft.fftfreq(N_WIN) > 0))   # analytic template
                corr = np.fft.ifft(np.fft.fft(sig[ifo]) * np.conj(np.fft.fft(qa)))
                M[f"overlap_{ifo.lower()}"][k] = np.abs(corr).max() / (
                    np.linalg.norm(q) * np.linalg.norm(sig[ifo]))
            M["rho_net"][k] = target
            M["dist_mpc"][k] = 1.0 / a
        for set_name in ("gauss", "real"):
            zz = {}
            for c, ifo in enumerate(IFOS):
                d = white[ifo][set_name][i0:i0 + N_WIN] + sig[ifo]
                spec[set_name][k, c] = spectrogram(d, G)
                z = np.abs(matched_filter(tmpl[ifo], TimeSeries(d, delta_t=1 / FS), psd=Sw2[ifo],
                                          low_frequency_cutoff=BAND[0],
                                          high_frequency_cutoff=BAND[1]).numpy())
                zz[ifo] = z
                M[f"mf_{set_name}_{ifo.lower()}_max"][k] = z.max()
                M[f"mf_{set_name}_{ifo.lower()}_tmax"][k] = int(np.argmax(z)) / FS
                j = int(round(M[f"tc_{ifo.lower()}"][k] * FS))          # known detector-frame time
                M[f"mf_{set_name}_{ifo.lower()}_tc"][k] = z[max(j - TC_TOL, 0):j + TC_TOL + 1].max()
                M[f"mf_{set_name}_{ifo.lower()}_z0"][k] = z[min(j, N_WIN - 1)]   # the one sample
            # the coherent two-detector SEARCH statistic: |z_H1(t)|^2 + max over the light-travel
            # delay of |z_L1(t + d)|^2, maximised over t (a sliding max does the delay search in
            # O(N)); and the known-time, known-delay one, whose closed form is Marcum Q_2
            net2 = zz["H1"] ** 2 + maximum_filter1d(zz["L1"] ** 2, 2 * DELAY_TOL + 1, mode="wrap")
            M[f"mf_{set_name}_net_max"][k] = np.sqrt(net2.max())
            M[f"mf_{set_name}_netk_tc"][k] = np.sqrt(M[f"mf_{set_name}_h1_tc"][k] ** 2 +
                                                     M[f"mf_{set_name}_l1_tc"][k] ** 2)
            M[f"mf_{set_name}_netk_z0"][k] = np.sqrt(M[f"mf_{set_name}_h1_z0"][k] ** 2 +
                                                     M[f"mf_{set_name}_l1_z0"][k] ** 2)
            if inject and set_name == "gauss" and len(examples) < n_examples \
                    and EXAMPLE_SNR[0] <= M["rho_net"][k] <= EXAMPLE_SNR[1]:
                examples.append(dict(
                    slot=slot, k=k, w0=w0, m1=m1, m2=m2, tc=tc, rho_net=M["rho_net"][k],
                    gauss=np.stack([white[i]["gauss"][i0:i0 + N_WIN] + sig[i] for i in IFOS]),
                    real=np.stack([white[i]["real"][i0:i0 + N_WIN] + sig[i] for i in IFOS]),
                    signal=np.stack([sig[i] for i in IFOS]),
                    spec_gauss=spec["gauss"][k].copy(), spec_real=spec["real"][k].copy()))
    shard = os.path.join(out_dir, "shards", f"{slot}.npz")
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    np.savez(shard, gauss=spec["gauss"], real=spec["real"], labels=labels,
             slot=np.full(n_w, slot, dtype=np.int64), w0=np.array(w0s), **M)
    if examples:
        np.savez(os.path.join(out_dir, "shards", f"{slot}_examples.npz"),
                 **{f"{i}_{key}": v for i, e in enumerate(examples) for key, v in e.items()})
    dt = time.time() - t_start
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  {slot} {split}: {n_w} windows, {int(labels.sum())} injected, "
          f"{dt:.0f} s, peak {peak_gb:.1f} GB  sw_real/gauss H1 {diag['H1']['sw_real_over_gauss']:.3f} "
          f"L1 {diag['L1']['sw_real_over_gauss']:.3f}", flush=True)
    return dict(slot=slot, split=split, n=n_w, n_inj=int(labels.sum()), seconds=dt,
                clean_seconds=int(ok.sum()), diag=diag, shard=shard)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    n_pairs, n_val, workers, out_dir, smoke = 26, 6, 8, "data/gw", False
    for a in sys.argv[1:]:
        if a.startswith("--pairs="):
            n_pairs = int(a.split("=")[1])
        elif a.startswith("--val-pairs="):
            n_val = int(a.split("=")[1])
        elif a.startswith("--workers="):
            workers = int(a.split("=")[1])
        elif a.startswith("--out="):
            out_dir = a.split("=")[1]
        elif a == "--smoke":
            smoke = True
    limit = 48 if smoke else 0
    if smoke:
        n_pairs, n_val = 2, 1
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    ex_arg = [a for a in sys.argv[1:] if a.startswith("--examples=")]
    if ex_arg:                                     # the figure's windows only, no dataset
        EXAMPLE_SNR[:] = [float(v) for v in ex_arg[0].split("=")[1].split(",")]
        slot, paths = gw.local_pairs()[0]
        tmp = os.path.join(out_dir, "examples_tmp")
        process_pair((slot, paths, "val", 1 / 3, tmp, 600, 4))
        e = np.load(os.path.join(tmp, "shards", f"{slot}_examples.npz"))
        np.savez(os.path.join(out_dir, "examples.npz"), **{f"{slot}_{k}": e[k] for k in e.files})
        import shutil
        shutil.rmtree(tmp)
        print(f"wrote {out_dir}/examples.npz at network SNR {EXAMPLE_SNR}  ({time.time() - t0:.0f} s)")
        return

    have = gw.local_pairs()
    if len(have) >= n_pairs:                       # an earlier pick, already on disk
        pairs = have[:n_pairs]
        paths = [p for _, p in pairs]
        print(f"{len(pairs)} file pairs on disk, slots {pairs[0][0]} .. {pairs[-1][0]}", flush=True)
    else:
        pairs = gw.pick_o3a_pairs(n_pairs)
        print(f"{len(pairs)} file pairs, slots {pairs[0][0]} .. {pairs[-1][0]}", flush=True)
        with ThreadPoolExecutor(4) as ex:
            paths = list(ex.map(lambda p: {ifo: gw.fetch(u) for ifo, u in p[1].items()}, pairs))
        print(f"fetched in {time.time() - t0:.0f} s", flush=True)

    val_idx = set(range(len(pairs))[::max(1, len(pairs) // n_val)][:n_val]) if n_val else set()
    jobs = []
    for i, ((slot, _), pth) in enumerate(zip(pairs, paths)):
        split = "val" if i in val_idx else "train"
        jobs.append((slot, pth, split, 1 / 3 if split == "val" else 0.5, out_dir, limit,
                     4 if split == "val" else 0))
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        reports = list(ex.map(process_pair, jobs))
    print(f"processed in {time.time() - t0:.0f} s", flush=True)

    rng = np.random.default_rng(SEED)
    sizes = {}
    for split in ("train", "val"):
        shards = [np.load(r["shard"]) for r in reports if r["split"] == split]
        if not shards:
            continue
        keys = [k for k in shards[0].files]
        cat = {k: np.concatenate([s[k] for s in shards]) for k in keys}
        n = len(cat["labels"])
        order = rng.permutation(n) if split == "train" else np.arange(n)
        for set_name in ("gauss", "real"):
            cat[set_name][order].tofile(os.path.join(out_dir, f"{set_name}_{split}.bin"))
        cat["labels"][order].tofile(os.path.join(out_dir, f"labels_{split}.bin"))
        np.savez(os.path.join(out_dir, f"meta_{split}.npz"),
                 **{k: v[order] for k, v in cat.items() if k not in ("gauss", "real")})
        sizes[split] = dict(n=int(n), n_inj=int(cat["labels"].sum()))
        print(f"wrote {split}: {n} windows ({int(cat['labels'].sum())} injected) -> "
              f"{out_dir}/{{gauss,real}}_{split}.bin", flush=True)
    ex_files = [os.path.join(out_dir, "shards", f"{r['slot']}_examples.npz") for r in reports]
    ex_all = {}
    for f in ex_files:
        if os.path.exists(f):
            e = np.load(f)
            ex_all.update({f"{os.path.basename(f).split('_')[0]}_{k}": e[k] for k in e.files})
    if ex_all:
        np.savez(os.path.join(out_dir, "examples.npz"), **ex_all)

    manifest = dict(
        run=gw.RUN, ifos=list(IFOS), fs=FS, window_s=WIN, band_hz=list(BAND), band_roll_hz=ROLL,
        highpass_hz=HIGHPASS, psd=dict(method="welch median", segment_s=PSD_SEG,
                                       truncation_s=PSD_SEG),
        injections=dict(approximant=APPROX, f_lower=F_LOWER, mass_prior=list(MASS),
                        snr_prior=list(SNR), tc_prior=list(TC), spins=0,
                        snr_definition="optimal network SNR of the in-window signal, whitened "
                                       "domain, over the band; light systems are truncated at "
                                       "the window start with a 0.1-s half-Hann ramp",
                        p_inject=dict(train=0.5, val=1 / 3), distance_ref_mpc=1.0),
        spectrogram=dict(bands=N_BANDS, q=Q, hop=HOP, frames=N_FRAMES, kernel="morlet",
                         centres_hz=[float(c) for c in make_bank()[1]],
                         value="log(power / median over time per band + 1e-6)"),
        matched_filter=dict(statistic="max |rho| over the window, phase-maximised, pycbc; "
                                      "network: max over t and the +-10 ms delay of "
                                      "|z_H1(t)|^2 + |z_L1(t+d)|^2",
                            tc_window_ms=2.0, delay_ms=10.0,
                            template="the window's own chirp, hp only "
                                     "(noise-only windows: a prior draw)"),
        layout="[N, 2 (H1, L1), 64 bands, 128 frames] f32; labels int32",
        seed=SEED, edge_s=EDGE, files=[dict(slot=r["slot"], split=r["split"], n=r["n"],
                                            n_inj=r["n_inj"], clean_seconds=r["clean_seconds"],
                                            seconds=round(r["seconds"], 1), diag=r["diag"])
                                       for r in reports],
        sizes=sizes, smoke=smoke, total_seconds=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out_dir}/manifest.json  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
