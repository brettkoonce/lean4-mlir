#!/usr/bin/env python3
"""Gate 0 of the gravitational-wave demo — planning/gw_detection_demo.md §8, §9.

Fetch the GWOSC file holding GW150914 for H1 and L1, cut a window of PSD_WINDOW
seconds around the event, estimate each detector's PSD from that window, run PyCBC's matched filter with an IMRPhenomD template
at the event's detector-frame masses, and report the peak SNR per detector and the
network SNR. Published: H1 ~ 20, L1 ~ 13, network ~ 24. The gate is within 1 of
each; a wrong PSD convention, sample rate or GPS offset misses by a factor.

Also fetches one science-mode O3a 4096-s file pair and reports its clean-second
count, so Phase 1 starts from a file that is known to read.

  .venv-gw/bin/python scripts/gw_gate0.py [--out=runs/<dir>] [--no-bulk]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gw_common as gw  # noqa: E402

EVENT = "GW150914"
PUBLISHED = dict(H1=20.0, L1=13.0, network=24.0)
# source-frame 36 + 29 Msun at z = 0.09 (Abbott et al. 2016); the template must match
# what the detector saw, which is the redshifted mass
M1, M2 = 36.0 * 1.09, 29.0 * 1.09
F_LOWER = 20.0
GATE_TOL = 1.0
PSD_WINDOW = 128.0   # seconds around the event: 32 s is the tutorial's, 128 s steadies the PSD



def event_snr(ifo, url, out_dir):
    from pycbc.filter import matched_filter
    from pycbc.waveform import get_td_waveform
    from gwosc.datasets import event_gps

    t0 = event_gps(EVENT)
    path = gw.fetch(url)
    strain, meta = gw.read_gwosc_hdf5(path)
    assert abs(meta["dt"] - 1 / gw.FS) < 1e-12, meta["dt"]
    ts = gw.cut(strain, meta, t0 - PSD_WINDOW / 2, t0 + PSD_WINDOW / 2)
    ts = gw.condition(ts)
    psd = gw.psd_of(ts)

    hp, _ = get_td_waveform(approximant="IMRPhenomD", mass1=M1, mass2=M2,
                            delta_t=ts.delta_t, f_lower=F_LOWER)
    hp.resize(len(ts))
    template = hp.cyclic_time_shift(hp.start_time)
    snr = matched_filter(template, ts, psd=psd, low_frequency_cutoff=F_LOWER)
    snr = snr.crop(4 + 4, 4)
    t = snr.sample_times.numpy()
    a = np.abs(snr.numpy())
    near = np.abs(t - t0) < 0.1
    i = np.argmax(np.where(near, a, 0.0))
    peak, t_peak = float(a[i]), float(t[i])

    # a picture for the record: whitened, band-passed strain around the peak
    white = gw.whiten_bandpass(ts, psd)
    w_t = white.sample_times.numpy()
    sel = (w_t > t_peak - 0.25) & (w_t < t_peak + 0.05)
    np.save(os.path.join(out_dir, f"{ifo}_whitened.npy"),
            np.stack([w_t[sel] - t_peak, white.numpy()[sel]]))
    return dict(ifo=ifo, snr=peak, t_peak=t_peak, t_peak_minus_t0=t_peak - t0,
                psd_seg=4.0, psd_window=PSD_WINDOW, template=dict(approximant="IMRPhenomD", m1=M1, m2=M2,
                                           f_lower=F_LOWER), file=os.path.basename(path))


def bulk_probe():
    """One O3a file pair chosen from the joint clean timeline (the first slot a week
    into the run); Phase 1's preprocess picks the rest the same way."""
    slot, urls = gw.pick_o3a_pairs(1)[0]
    rows = dict(slot=slot)
    for ifo in ("H1", "L1"):
        path = gw.fetch(urls[ifo])
        strain, meta = gw.read_gwosc_hdf5(path)
        ok = gw.clean_seconds(meta)
        rows[ifo] = dict(file=os.path.basename(path), gps_start=meta["gps_start"],
                         seconds=int(len(ok)), clean_seconds=int(ok.sum()),
                         nan_samples=int(np.isnan(strain).sum()),
                         dq_names=meta["dq_names"], inj_names=meta["inj_names"])
    return rows


def plot_whitened(out_dir, res):
    """The whitened, band-passed strain of both detectors around the peak, L1 shifted
    by the measured delay and sign-flipped (the detectors' arms are nearly anti-aligned);
    the picture the section's figure panel (a) grows from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for ifo, color, flip in (("H1", "tab:orange", 1.0), ("L1", "tab:blue", -1.0)):
        t, w = np.load(os.path.join(out_dir, f"{ifo}_whitened.npy"))
        ax.plot(t * 1e3, flip * w, lw=0.8, color=color,
                label=f"{ifo}{'  (x -1, shifted to H1)' if flip < 0 else ''}   SNR {res[ifo]['snr']:.1f}")
    ax.set_xlabel("ms from H1 peak")
    ax.set_ylabel("whitened strain (sigma)")
    ax.set_title(f"{EVENT}, whitened 20-500 Hz, PSD from {PSD_WINDOW:.0f} s around the event",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate0_whitened.png"), dpi=130)


def main():
    out_dir = "runs/2026-09-11-gw-gate0"
    do_bulk = True
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out_dir = a.split("=", 1)[1]
        elif a == "--no-bulk":
            do_bulk = False
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    urls = gw.event_urls(EVENT)
    res = {ifo: event_snr(ifo, urls[ifo], out_dir) for ifo in ("H1", "L1")}
    net = float(np.sqrt(sum(r["snr"] ** 2 for r in res.values())))
    delay_ms = 1e3 * (res["L1"]["t_peak"] - res["H1"]["t_peak"])

    print(f"\n{EVENT}  (template IMRPhenomD {M1:.1f} + {M2:.1f} Msun detector frame, "
          f"f_lower {F_LOWER:.0f} Hz, median-Welch PSD 4 s)")
    print(f"{'':10s}{'SNR':>8s}{'published':>11s}{'diff':>7s}   peak - t0 [ms]")
    ok = True
    for ifo in ("H1", "L1"):
        r = res[ifo]
        d = r["snr"] - PUBLISHED[ifo]
        ok &= abs(d) <= GATE_TOL
        print(f"{ifo:10s}{r['snr']:8.2f}{PUBLISHED[ifo]:11.1f}{d:+7.2f}   "
              f"{1e3 * r['t_peak_minus_t0']:+.1f}")
    d = net - PUBLISHED["network"]
    ok &= abs(d) <= GATE_TOL
    print(f"{'network':10s}{net:8.2f}{PUBLISHED['network']:11.1f}{d:+7.2f}   "
          f"L1 - H1 delay {delay_ms:+.1f} ms (light travel <= 10 ms)")
    ok &= abs(delay_ms) <= 10.0
    print(f"\nGATE 0: {'PASS' if ok else 'FAIL'}  (each within {GATE_TOL} of published)")
    plot_whitened(out_dir, res)

    summary = dict(event=EVENT, per_ifo=res, network_snr=net, delay_ms=delay_ms,
                   published=PUBLISHED, gate_pass=bool(ok))
    if do_bulk:
        summary["o3a_probe"] = bulk_probe()
        for ifo, r in summary["o3a_probe"].items():
            print(f"O3a {ifo}: {r}")
    summary["seconds"] = time.time() - t_start
    gw.save_json(summary, os.path.join(out_dir, "gate0.json"))
    print(f"wrote {out_dir}/gate0.json  ({summary['seconds']:.0f} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
