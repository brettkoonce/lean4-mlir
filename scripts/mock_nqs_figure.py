"""Mock figure for the neural-quantum-state section mock (planning/transformer_wavefunction_demo.md).

Computed from the physics alone on 2026-09-11 with the system python3 (numpy +
matplotlib; the pinned .venv has no matplotlib). Writes its PNG and JSON into the
current directory. It is the template the real figure script grows from, not a
result: no network trained through the stack appears in it.
"""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
N, J = 12, 1.0
K = 1 << N
ks = np.arange(K)
bits = (ks[:, None] >> np.arange(N)[None]) & 1
S = 1.0 - 2.0 * bits                      # [K, N] spins ±1
FL = ks[:, None] ^ (1 << np.arange(N))[None]   # flipped-config index [K, N]
zz = (S * np.roll(S, -1, axis=1)).sum(1)  # Σ_i s_i s_{i+1}, periodic
half = N // 2

def exact_ground(h):
    H = np.zeros((K, K))
    H[ks, ks] = -J * zz
    for i in range(N):
        H[ks, FL[:, i]] = -h
    w, v = np.linalg.eigh(H)
    g = v[:, 0]
    C = ((g * g) * S[:, 0] * S[:, half]).sum()
    Mx = 0.0
    for i in range(N):
        Mx += (g * g[FL[:, i]]).sum()
    return w[0], C, Mx / N

def mean_field(h):
    phi = np.linspace(0, np.pi/2, 20001)
    return N * (-J * np.cos(phi)**2 - h * np.sin(phi)).min()

def logcosh(x):
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2*ax)) - np.log(2.0)

def train_rbm(h, alpha, steps=2500, lr=0.03, seed=0):
    r = np.random.default_rng(seed)
    M = alpha * N
    W = 0.01 * r.standard_normal((M, N)); b = np.zeros(M); a = np.zeros(N)
    params = [W, b, a]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for it in range(1, steps + 1):
        Th = S @ W.T + b
        lp = S @ a + logcosh(Th).sum(1)
        p = np.exp(2 * (lp - lp.max())); p /= p.sum()
        ratio = np.exp(lp[FL] - lp[:, None])          # ψ(flip_i s)/ψ(s)  [K, N]
        Eloc = -J * zz - h * ratio.sum(1)
        E = (p * Eloc).sum()
        var = (p * (Eloc - E) ** 2).sum()
        hist.append((E, var))
        wgt = 2 * p * (Eloc - E)
        T = np.tanh(Th)
        gW = np.einsum("s,sm,sn->mn", wgt, T, S)
        gb = wgt @ T
        ga = wgt @ S
        lr_t = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / steps)))
        for i, (prm, g) in enumerate(zip(params, [gW, gb, ga])):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** it); vh = v[i] / (1 - b2 ** it)
            prm -= lr_t * mh / (np.sqrt(vh) + eps)
    Th = S @ W.T + b; lp = S @ a + logcosh(Th).sum(1)
    p = np.exp(2 * (lp - lp.max())); p /= p.sum()
    C = (p * S[:, 0] * S[:, half]).sum()
    E, var = hist[-1]
    return E, var, C, hist

hs = np.round(np.arange(0.2, 2.01, 0.2), 2)
if os.path.exists("nqs.json"):
    res = json.load(open("nqs.json"))
else:
    res = {"N": N, "h": hs.tolist(), "exact": [], "mf": [], "rbm1": [], "rbm4": []}
    for h in hs:
        E0, C0, Mx0 = exact_ground(h)
        Emf = mean_field(h)
        res["exact"].append(dict(E=float(E0), C=float(C0), Mx=float(Mx0)))
        res["mf"].append(dict(E=float(Emf), relerr=float((Emf - E0) / abs(E0))))
        for alpha, key in [(1, "rbm1"), (4, "rbm4")]:
            E, var, C, hist = train_rbm(h, alpha)
            res[key].append(dict(E=float(E), var=float(var), C=float(C), relerr=float((E - E0) / abs(E0))))
        print(f"h={h:.1f} E0={E0:.4f} mf={res['mf'][-1]['relerr']:.3e} a1={res['rbm1'][-1]['relerr']:.3e} a4={res['rbm4'][-1]['relerr']:.3e} var4={res['rbm4'][-1]['var']:.2e} C0={C0:.3f} C4={res['rbm4'][-1]['C']:.3f}", flush=True)
    # training curve at the critical point for panel context
    _, _, _, hist_c = train_rbm(1.0, 4)
    E0c, _, _ = exact_ground(1.0)
    res["curve_h1"] = dict(E0=float(E0c), E=[float(e) for e, _ in hist_c[::10]], var=[float(v) for _, v in hist_c[::10]])
    json.dump(res, open("nqs.json", "w"), indent=1)

# ── figure ────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
BLUE, ORANGE, INK, MUTED, LINE = "#2a78d6", "#eb6834", "#1f1e1b", "#6b6963", "#bdbab1"
fig = plt.figure(figsize=(13.2, 4.3), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
# (a) schematic
ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
cfg = S[rng.integers(K)]
for i, s in enumerate(cfg):
    x = 0.6 + i * 0.78
    ax.annotate("", xy=(x, 8.2 + 0.55 * s), xytext=(x, 8.2 - 0.55 * s),
                arrowprops=dict(arrowstyle="-|>", color=BLUE if s > 0 else ORANGE, lw=1.6))
    ax.text(x, 7.0, f"{int(s):+d}", ha="center", fontsize=7, color=MUTED, family="DejaVu Sans Mono")
ax.plot([0.35, 0.35 + 11 * 0.78 + 0.5], [8.2, 8.2], color=LINE, lw=0.8, zorder=0)
ax.text(5.0, 9.6, "σ ∈ {±1}ⁿ, one configuration of the periodic chain, N = 12", ha="center", fontsize=8, color=INK)
ax.text(5.0, 6.1, "H = −J Σ σᶻᵢ σᶻᵢ₊₁ − h Σ σˣᵢ", ha="center", fontsize=9.5, color=INK)
ax.annotate("", xy=(5.0, 4.6), xytext=(5.0, 5.6), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
ax.add_patch(plt.Rectangle((2.2, 2.9), 5.6, 1.7, fc="#eef4fc", ec=BLUE, lw=1.0))
ax.text(5.0, 4.05, "network  log ψθ(σ)", ha="center", fontsize=9, color=INK)
ax.text(5.0, 3.35, ".dense N→αN (logcosh) · sum   |   or the Ch 2–3 MLP", ha="center", fontsize=7.2, color=MUTED, family="DejaVu Sans Mono")
ax.annotate("", xy=(5.0, 1.75), xytext=(5.0, 2.85), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
ax.text(5.0, 1.2, "E[θ] = Σₛ pₛ E_loc(s),   pₛ ∝ ψ(s)²,   all 4096 s enumerated", ha="center", fontsize=8, color=INK)
ax.text(5.0, 0.45, "∂E/∂θ = 2 Σₛ pₛ (E_loc − E) ∂ log ψ / ∂θ  →  one host weight per row", ha="center", fontsize=7.4, color=MUTED)
ax.set_title("(a)  the ansatz: amplitudes are positive, so log ψ is one real number", loc="left")
# (b) relative energy error
ax = fig.add_subplot(gs[1])
mf = [d["relerr"] for d in res["mf"]]
r1 = [max(d["relerr"], 1e-7) for d in res["rbm1"]]
r4 = [max(d["relerr"], 1e-7) for d in res["rbm4"]]
ax.plot(hs, mf, color=MUTED, ls="--", lw=1.2, marker="o", ms=3.5, label="product state (mean field)")
ax.plot(hs, r1, color=ORANGE, lw=1.6, marker="o", ms=4, label="RBM α = 1")
ax.plot(hs, r4, color=BLUE, lw=1.6, marker="o", ms=4, label="RBM α = 4")
ax.set_yscale("log"); ax.set_xlabel("transverse field h / J"); ax.set_ylabel("(E − E₀) / |E₀|")
ax.axvline(1.0, color=LINE, lw=0.8); ax.text(1.02, ax.get_ylim()[1] if False else 0.5, "h = J", fontsize=7.5, color=MUTED, transform=ax.get_xaxis_transform())
ax.grid(axis="y", color="#ecebe6", lw=0.6); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_color(LINE)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.set_title("(b)  energy error against exact diagonalisation", loc="left")
# (c) correlation
ax = fig.add_subplot(gs[2])
C0 = [d["C"] for d in res["exact"]]; C4 = [d["C"] for d in res["rbm4"]]; C1 = [d["C"] for d in res["rbm1"]]
hf = np.linspace(0.2, 2.0, 200)
ax.plot(hs, C0, color=INK, lw=1.4, label="exact")
ax.plot(hs, C1, color=ORANGE, ls="none", marker="o", ms=4.5, mfc="white", mew=1.2, label="RBM α = 1")
ax.plot(hs, C4, color=BLUE, ls="none", marker="o", ms=4.5, label="RBM α = 4")
ax.axvline(1.0, color=LINE, lw=0.8); ax.text(1.02, 0.5, "h = J", fontsize=7.5, color=MUTED, transform=ax.get_xaxis_transform())
ax.set_xlabel("transverse field h / J"); ax.set_ylabel("⟨σᶻ₁ σᶻ₇⟩, the order across half the ring")
ax.set_ylim(-0.05, 1.05); ax.grid(axis="y", color="#ecebe6", lw=0.6); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_color(LINE)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_title("(c)  the transition, read off the trained ψ", loc="left")
fig.savefig("nqs_ising.png", dpi=190, facecolor="white")
print("saved")
