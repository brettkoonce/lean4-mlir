"""Mock figure for a gravitational-wave detection demo (the Physics section's next entry).

Computed from the physics alone, no GWOSC download: an analytic Advanced LIGO
noise curve, a Newtonian inspiral chirp whitened by it, and the matched filter's
closed-form detection probability. It is the template the real figure script grows
from, not a result: no network trained through the stack appears in it, and the
"CNN" curve is a placeholder drawn one unit of SNR below the theorem.

  python3 scripts/mock_gw_figure.py out.png      # system python3: numpy, scipy, matplotlib
"""
import sys
import numpy as np
from scipy import signal
from scipy.stats import ncx2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "gw_mock.png"
rng = np.random.default_rng(3)

# ── the detector: an analytic fit to the aLIGO design noise curve ───────────
# S_n(f) = S0 [ x^-4.14 - 5 x^-2 + 111 (1 - x^2 + x^4/2) / (1 + x^2/2) ], x = f / 215 Hz
# (Ajith & Bose 2009), with two spectral lines added for realism: mains at 60 Hz
# and a violin mode near 500 Hz. Whitening divides all of it out.
def psd(f):
    x = np.maximum(f, 10.0) / 215.0
    s = 1e-49 * (x ** -4.14 - 5 * x ** -2 + 111 * (1 - x ** 2 + x ** 4 / 2) / (1 + x ** 2 / 2))
    for f0, w, h in ((60.0, 0.3, 300.0), (500.0, 1.0, 60.0)):
        s = s * (1 + h * (w ** 2) / ((f - f0) ** 2 + w ** 2))
    return s


fs, T = 4096, 2.0
N = int(fs * T)
t = np.arange(N) / fs
freqs = np.fft.rfftfreq(N, 1 / fs)
Sn = psd(freqs)
band = signal.butter(4, [20, 500], btype="band", fs=fs, output="sos")

# ── the source: a 15 + 15 solar-mass binary, Newtonian chirp to 220 Hz ──────
MSUN_S = 4.925e-6                       # G M_sun / c^3 in seconds
m1 = m2 = 15.0
Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
tc = 1.62                               # coalescence time inside the window
f_max = 220.0
tau = np.maximum(tc - t, 1e-4)
f_gw = (1 / np.pi) * (Mc * MSUN_S) ** (-5 / 8) * (5 / (256 * tau)) ** (3 / 8)
live = f_gw < f_max
f_gw = np.where(live, f_gw, f_max)
amp = f_gw ** (2 / 3)
phase = 2 * np.pi * np.cumsum(f_gw) / fs
h = amp * np.cos(phase)
# ringdown: the last cycles die off over ~5 ms after the chirp reaches f_max
t_end = t[live][-1]
h = np.where(t > t_end, h * np.exp(-(t - t_end) / 0.005), h)
h[t < 0.05] = 0.0
h *= signal.windows.tukey(N, 0.05)


def whiten(x):
    X = np.fft.rfft(x) / np.sqrt(Sn)
    return signal.sosfiltfilt(band, np.fft.irfft(X, N))


h_w = whiten(h)
noise_w = signal.sosfiltfilt(band, rng.standard_normal(N))
sigma = noise_w.std()
h_w *= 1.0 / np.linalg.norm(h_w / sigma)   # unit optimal SNR in whitened units
SNR = 24.0                         # GW150914's network SNR; at 12 the chirp is invisible by eye
data_w = noise_w + SNR * h_w


# ── the network's input: 64 log-spaced bins over 20–500 Hz x 128 frames ────
def spectrogram(x):
    """62 ms windows, 16 ms hop: 125 frames; power per bin normalised by the
    noise's median in that bin (a second whitening, on the display) and mapped
    onto 64 log-spaced bins."""
    f, tt, Z = signal.stft(x, fs=fs, nperseg=256, noverlap=256 - 64, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    P = P / np.median(P, axis=1, keepdims=True)
    fl = np.geomspace(20, 500, 64)
    S = np.stack([np.interp(fl, f, P[:, k]) for k in range(P.shape[1])], 1)
    return fl, tt, np.log10(S + 1e-3)


fl, tt, S_sig = spectrogram(data_w)
_, _, S_noise = spectrogram(noise_w)
vmin, vmax = np.percentile(S_noise, [2, 99.8])

# ── the theorem: matched filter in Gaussian noise, unknown phase ────────────
# rho = |<d, h>| is Rayleigh under noise, so P_fa(rho*) = exp(-rho*^2 / 2) per
# trial, and under a signal of SNR rho the statistic^2 is non-central chi^2 with
# 2 dof: P_d = Q_1(rho, rho*), the Marcum Q-function.
rho_star = 8.0
rhos = np.linspace(0, 30, 400)
p_theorem = ncx2.sf(rho_star ** 2, 2, rhos ** 2)
p_bank = ncx2.sf(rho_star ** 2, 2, (0.97 * rhos) ** 2)      # 3 % bank mismatch
p_cnn = ncx2.sf(rho_star ** 2, 2, np.maximum(rhos - 1.0, 0) ** 2)  # PLACEHOLDER
events = [("GW150914", 24.4), ("GW151226", 13.1), ("GW170104", 13.0), ("GW170608", 14.9),
          ("GW170814", 17.7), ("GW170817", 33.0), ("GW190521", 14.7), ("GW190814", 25.0),
          ("GW151012", 10.0), ("GW170729", 10.8), ("GW170809", 12.4), ("GW170818", 11.3),
          ("GW170823", 11.5)]

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1e1b", "#6b6963"
fig = plt.figure(figsize=(13.2, 4.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])

# (a) whitened strain with the injected chirp
ax = fig.add_subplot(gs[:, 0])
ax.plot(t, data_w / sigma, color=MUTED, lw=0.5, alpha=0.9, label="whitened H1 strain, noise + chirp")
ax.plot(t, SNR * h_w / sigma, color=ORANGE, lw=0.9, label=f"the injected chirp, optimal SNR {SNR:.0f}")
ax.set_xlim(0, T); ax.set_ylim(-6, 6)
ax.set_xlabel("time (s)"); ax.set_ylabel("whitened strain (σ units)")
ax.set_title(f"(a)  15 + 15 M☉ chirp at SNR {SNR:.0f} in aLIGO noise, whitened", loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=7.5)
ax.text(0.02, 0.03, "Newtonian inspiral to 220 Hz + ringdown; noise from the analytic aLIGO curve",
        transform=ax.transAxes, fontsize=7, color=MUTED)
for s in ("top", "right"): ax.spines[s].set_visible(False)

# (b) the spectrograms the network sees, noise only and with the signal
for row, (S, title) in enumerate([(S_noise, "noise only"), (S_sig, f"with the chirp, SNR {SNR:.0f}")]):
    ax = fig.add_subplot(gs[row, 1])
    ax.pcolormesh(tt, fl, S, vmin=vmin, vmax=vmax, cmap="viridis", shading="auto")
    ax.set_yscale("log"); ax.set_ylim(20, 500)
    ax.set_yticks([20, 50, 100, 200, 500]); ax.set_yticklabels(["20", "50", "100", "200", "500"])
    ax.set_ylabel("Hz")
    if row == 0:
        ax.set_title("(b)  what the CNN sees: 64 log bins × 128 frames", loc="left")
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("time (s)")
    ax.text(0.02, 0.86, title, transform=ax.transAxes, fontsize=8, color="white")

# (c) the theorem and the placeholder
ax = fig.add_subplot(gs[:, 2])
ax.plot(rhos, p_theorem, color=INK, lw=1.6, label="matched filter, true template (closed form)")
ax.plot(rhos, p_bank, color=INK, lw=1.0, ls=":", label="matched filter, template bank (3 % mismatch)")
ax.plot(rhos, p_cnn, color=BLUE, lw=1.4, ls="--", label="CNN on spectrograms — placeholder, the demo's number")
ax.axvline(rho_star, color=MUTED, lw=0.8, ls="-.")
ax.text(rho_star + 0.3, 0.06, f"threshold ρ* = {rho_star:.0f}", fontsize=7.5, color=MUTED)
for name, snr in events:
    ax.plot([snr, snr], [1.02, 1.05], color=ORANGE, lw=1.0)
for name, snr, y in (("GW150914", 24.4, 1.065), ("GW151226", 13.1, 1.065), ("GW170817 →", 28.6, 1.065)):
    ax.text(snr, y, name, fontsize=6.5, color=ORANGE, ha="center", va="bottom")
ax.text(0.3, 1.09, "catalogue events at their network SNR", fontsize=7, color=ORANGE, ha="left", va="bottom")
ax.set_xlim(0, 30); ax.set_ylim(0, 1.15)
ax.set_xlabel("injected optimal SNR ρ"); ax.set_ylabel("detection probability at ρ* = 8")
ax.set_title("(c)  the ceiling is a theorem: P_d = Q₁(ρ, ρ*)", loc="left")
ax.legend(loc="lower right", frameon=False, fontsize=7.2, bbox_to_anchor=(1.0, 0.08))
ax.grid(True, color="0.9", lw=0.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)

fig.savefig(out, dpi=190, facecolor="white")
print(f"wrote {out}")
