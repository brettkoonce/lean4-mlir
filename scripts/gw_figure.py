#!/usr/bin/env python3
"""The gravitational-wave demo's figure — planning/gw_detection_demo.md §7, grown from
scripts/mock_gw_figure.py, whose numbers were the closed form and whose CNN was a
placeholder; every curve here is measured.

  (a) a whitened 2-s H1 window from the val set with its injected chirp overlaid
  (b) the spectrograms the CNN sees: a noise-only window and the injected one
  (c) detection probability against injected network SNR in Gaussian noise: the
      closed form Q_2(rho, rho*) at the search's own threshold rho* (the doc's
      formulation, Gabbard et al.'s comparison), PyCBC's coherent matched-filter
      search on the same windows, and the CNN. The known-time rows that calibrate
      the closed form (Gate 1) stay in the table.
  (d) the same in real O3a noise

  .venv-gw/bin/python scripts/gw_figure.py <table_val.json> [out.png] [--data=data/gw]
                                           [--cnn-real=<table_val.json>] [--net=B0|CNN]
                                           [--events=catalogue.json]

`table_val.json` is `gw_metrics.py table --out=<dir>`'s output; pass the one written
with `--logits=` so the CNN rows are in it. `--cnn-real` takes panel (d)'s CNN row
from a second table (the model trained on real noise), so each panel shows the arm
trained on the noise it is tested on. Blue is the model, orange the physics.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import ncx2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1e1b", "#6b6963"
FS = 4096
BAND = (20.0, 500.0)


def main():
    table = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "gw_detect.png"
    data, events, cnn_real, net = "data/gw", None, None, "B0"
    for a in sys.argv[2:]:
        if a.startswith("--data="):
            data = a.split("=", 1)[1]
        elif a.startswith("--net="):
            net = a.split("=", 1)[1]
        elif a.startswith("--events="):
            events = json.load(open(a.split("=", 1)[1]))
        elif a.startswith("--cnn-real="):
            cnn_real = json.load(open(a.split("=", 1)[1]))
    T = json.load(open(table))
    if cnn_real is not None and "CNN" in cnn_real["results"]["real"]:
        T["results"]["real"]["CNN"] = cnn_real["results"]["real"]["CNN"]
    pfa = T["pfa"]
    ex = np.load(os.path.join(data, "examples.npz"))
    man = json.load(open(os.path.join(data, "manifest.json")))
    centres = np.array(man["spectrogram"]["centres_hz"])

    # the example window: the first stored one with a network SNR nearest 12
    keys = sorted({k[:-len("_rho_net")] for k in ex.files if k.endswith("_rho_net")},
                  key=lambda k: abs(float(ex[k + "_rho_net"]) - 12.0))
    k = keys[0]
    gauss, sig = ex[k + "_gauss"], ex[k + "_signal"]
    rho, m1, m2, tc = float(ex[k + "_rho_net"]), float(ex[k + "_m1"]), float(ex[k + "_m2"]), float(ex[k + "_tc"])
    spec_inj = ex[k + "_spec_gauss"]
    lbl = np.fromfile(os.path.join(data, "labels_val.bin"), dtype=np.int32)
    k0 = int(np.where(lbl == 0)[0][0])
    with open(os.path.join(data, "gauss_val.bin"), "rb") as f:
        f.seek(k0 * 2 * 64 * 128 * 4)
        spec_noise = np.fromfile(f, dtype=np.float32, count=2 * 64 * 128).reshape(2, 64, 128)
    noise_only = gauss - sig
    sigma = noise_only[0].std()
    t = np.arange(gauss.shape[1]) / FS
    frames = (np.arange(128) * 64) / FS

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig = plt.figure(figsize=(13.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])

    # (a) the whitened window with its chirp
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(t, gauss[0] / sigma, color=MUTED, lw=0.5, alpha=0.9, label="whitened H1 window, noise + chirp")
    ax.plot(t, sig[0] / sigma, color=ORANGE, lw=0.9, label=f"the injected chirp, network SNR {rho:.1f}")
    ax.set_xlim(0, 2); ax.set_ylim(-6, 6)
    ax.set_xlabel("time (s)"); ax.set_ylabel("whitened strain (σ units)")
    ax.set_title(f"(a)  {m1:.0f} + {m2:.0f} M☉ chirp, network SNR {rho:.0f}, whitened H1", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.text(0.02, 0.03, f"IMRPhenomD, coalescence at {tc:.2f} s; whitened by the file's own PSD, 20–500 Hz",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) the two spectrograms
    vmin, vmax = np.percentile(spec_noise[0], [2, 99.8])
    for row, (S, title) in enumerate([(spec_noise[0], "noise only"),
                                      (spec_inj[0], f"with the chirp, SNR {rho:.0f}")]):
        ax = fig.add_subplot(gs[row, 1])
        ax.pcolormesh(frames, centres, S, vmin=vmin, vmax=vmax, cmap="viridis", shading="auto")
        ax.set_yscale("log"); ax.set_ylim(BAND[0], BAND[1])
        ax.set_yticks([20, 50, 100, 200, 500]); ax.set_yticklabels(["20", "50", "100", "200", "500"])
        ax.set_ylabel("Hz")
        if row == 0:
            ax.set_title(f"(b)  what the CNN sees: {man['spectrogram']['bands']} bands × "
                         f"{man['spectrogram']['frames']} frames", loc="left")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)")
        ax.text(0.02, 0.86, title, transform=ax.transAxes, fontsize=8, color="white")

    # (c), (d) detection probability against injected SNR
    rhos = np.linspace(0, 20, 400)
    for row, (set_name, label) in enumerate([("gauss", "Gaussian noise (O3a PSD)"),
                                             ("real", "real O3a strain")]):
        R = T["results"][set_name]
        ax = fig.add_subplot(gs[row, 2])
        srch = R["matched filter network"]
        ceiling = ncx2.sf(srch["thr"] ** 2, 4, rhos ** 2)
        ax.plot(rhos, ceiling, color=INK, lw=1.6,
                label=f"closed form Q₂(ρ, ρ*), ρ* = {srch['thr']:.2f}")
        xs = np.array([r["lo"] + 0.5 for r in srch["rows"]])
        pd = np.array([r["pd"] for r in srch["rows"]], dtype=float)
        sg = np.array([r["sig"] for r in srch["rows"]], dtype=float)
        ax.errorbar(xs, pd, yerr=2 * sg, color=INK, lw=1.0, ls=":", marker="o", ms=3,
                    capsize=0, label="PyCBC coherent search")
        if "CNN" in R:
            cnn = R["CNN"]
            pc = np.array([r["pd"] for r in cnn["rows"]], dtype=float)
            sc = np.array([r["sig"] for r in cnn["rows"]], dtype=float)
            ax.errorbar(xs, pc, yerr=2 * sc, color=BLUE, lw=1.4, ls="--", marker="s", ms=3,
                        capsize=0, label=f"{net} on spectrograms, trained on "
                                         f"{'Gaussian' if row == 0 or cnn_real is None else 'real'}")
            ax.text(0.98, 0.06, f"SNR at P_d = ½:  search {srch['rho50']:.2f}   CNN {cnn['rho50']:.2f}",
                    transform=ax.transAxes, fontsize=7.2, color=MUTED, ha="right")
        if events and row == 0:
            for name, snr in events:
                if snr <= 20:
                    ax.plot([snr, snr], [1.02, 1.05], color=ORANGE, lw=1.0)
            ax.text(0.3, 1.07, "catalogue events at their network SNR", fontsize=7, color=ORANGE)
        ax.set_xlim(2, 20); ax.set_ylim(0, 1.15 if events and row == 0 else 1.05)
        ax.set_ylabel(f"P_d at P_fa = {pfa:g} per window")
        if row == 1:
            ax.set_xlabel("injected optimal network SNR ρ")
        ax.set_title(("(c)" if row == 0 else "(d)") + f"  {label}", loc="left")
        ax.legend(loc="lower right", frameon=False, fontsize=6.8, bbox_to_anchor=(1.0, 0.13))
        ax.grid(True, color="0.9", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.savefig(out, dpi=190, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
