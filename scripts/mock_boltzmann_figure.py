"""Mock figure for the Boltzmann-generator section mock (planning/boltzmann_generator_demo.md).

Computed from the physics alone on 2026-09-11 with the system python3 (numpy +
matplotlib; the pinned .venv has no matplotlib). Writes its PNG and JSON into the
current directory. It is the template the real figure script grows from, not a
result: no network trained through the stack appears in it.
"""
import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(0)

# ── Müller-Brown ──────────────────────────────────────────────────────
A  = np.array([-200., -100., -170., 15.])
a  = np.array([-1., -1., -6.5, 0.7])
b  = np.array([0., 0., 11., 0.6])
c  = np.array([-10., -10., -6.5, 0.7])
x0 = np.array([1., 0., -0.5, -1.])
y0 = np.array([0., 0.5, 1.5, 1.])

def U(P):
    x, y = P[..., 0:1], P[..., 1:2]
    dx, dy = x - x0, y - y0
    return (A * np.exp(a*dx*dx + b*dx*dy + c*dy*dy)).sum(-1)

def gradU(P):
    x, y = P[..., 0:1], P[..., 1:2]
    dx, dy = x - x0, y - y0
    e = A * np.exp(a*dx*dx + b*dx*dy + c*dy*dy)
    gx = (e * (2*a*dx + b*dy)).sum(-1)
    gy = (e * (b*dx + 2*c*dy)).sum(-1)
    return np.stack([gx, gy], -1)

def descend(P, steps=600, eta=1e-4, cap=0.01):
    P = P.copy()
    for _ in range(steps):
        g = gradU(P)
        n = np.linalg.norm(g, axis=-1, keepdims=True)
        step = eta * g
        scale = np.minimum(1.0, cap / (eta * n + 1e-12))
        P -= step * scale
    return P

minima0 = np.array([[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]])
minima = descend(minima0, steps=3000, eta=5e-5)
Umin = U(minima)
print("minima", minima, Umin)
saddles = np.array([[-0.822, 0.624], [0.212, 0.293]])
print("saddles U", U(saddles))

def basin(P):
    Q = descend(P)
    d = ((Q[:, None, :] - minima[None]) ** 2).sum(-1)
    return d.argmin(1)

# ── quadrature ────────────────────────────────────────────────────────
xs = np.linspace(-1.7, 1.3, 360); ys = np.linspace(-0.7, 2.2, 360)
X, Y = np.meshgrid(xs, ys); G = np.stack([X, Y], -1).reshape(-1, 2)
UG = U(G)
dA = (xs[1]-xs[0]) * (ys[1]-ys[0])
BG = basin(G)  # basin of every grid cell

def exact(kT):
    w = np.exp(-(UG - UG.min()) / kT); Z = w.sum()
    p = w / Z
    pops = np.array([p[BG == k].sum() for k in range(3)])
    meanU = (p * UG).sum()
    return pops, meanU, p

def dF(pops, kT):  # ΔF_AB = -kT ln(pA/pB)
    return -kT * np.log(pops[0] / pops[1])

def exact_samples(kT, n):
    _, _, p = exact(kT)
    idx = rng.choice(len(p), size=n, p=p)
    jit = (rng.random((n, 2)) - 0.5) * np.array([xs[1]-xs[0], ys[1]-ys[0]])
    return G[idx] + jit

# ── Langevin ──────────────────────────────────────────────────────────
def langevin(P0, kT, steps, dt=1e-4, thin=20, burn=0):
    P = P0.copy(); out = []
    s = np.sqrt(2 * kT * dt)
    for i in range(steps):
        P = P - gradU(P) * dt + s * rng.standard_normal(P.shape)
        if i >= burn and i % thin == 0: out.append(P.copy())
    return np.array(out)  # [T, chains, 2]

# ── exact flow-matching field (particle posterior), standardized coords ──
CEN = np.array([-0.2, 0.75]); SC = 0.8
to_z = lambda P: (P - CEN) / SC
to_x = lambda Z: Z * SC + CEN

def flow(particles_z, n, nfe, t_end=0.02, keep=0):
    z = rng.standard_normal((n, 2)); traj = [z.copy()]
    ts = np.linspace(1.0, t_end, nfe + 1)
    for k in range(nfe):
        t, tn = ts[k], ts[k+1]
        d2 = ((z[:, None, :] - (1-t) * particles_z[None]) ** 2).sum(-1)
        logw = -d2 / (2 * t * t); logw -= logw.max(1, keepdims=True)
        w = np.exp(logw); w /= w.sum(1, keepdims=True)
        Ez0 = w @ particles_z
        v = (z - Ez0) / t
        z = z + v * (tn - t)
        traj.append(z.copy())
    return z, np.array(traj)

def energy_distance(Xs, Ys):
    def md(P, Q):
        return np.sqrt(((P[:, None, :] - Q[None]) ** 2).sum(-1)).mean()
    return 2 * md(Xs, Ys) - md(Xs, Xs) - md(Ys, Ys)

def stats(P, kT, name):
    bs = basin(P); pops = np.array([(bs == k).mean() for k in range(3)])
    return dict(name=name, pops=pops.tolist(), meanU=float(U(P).mean()), dF=float(dF(pops + 1e-12, kT)))

results = {}
kT_train = 20.0
ex_pops, ex_U, _ = exact(kT_train)
ref = to_z(exact_samples(kT_train, 2000))
results["exact20"] = dict(pops=ex_pops.tolist(), meanU=float(ex_U), dF=float(dF(ex_pops, kT_train)))
print("exact20", results["exact20"])

# training data: 8 chains, random starts, kT=20
P0 = np.stack([rng.uniform(-1.5, 1.1, 8), rng.uniform(-0.4, 2.0, 8)], -1)
L20 = langevin(P0, kT_train, steps=100_000, thin=20, burn=10_000)  # [4500, 8, 2]
L20f = L20.reshape(-1, 2)
results["langevin20"] = stats(L20f, kT_train, "langevin 8x1e5")
results["langevin20"]["ED"] = float(energy_distance(to_z(L20f[rng.choice(len(L20f), 2000, replace=False)]), ref))
print("langevin20", results["langevin20"])

# exact-field flow at kT=20
part20 = to_z(exact_samples(kT_train, 6000))
z50, traj = flow(part20, 1500, 50)
F50 = to_x(z50)
results["flow20_nfe50"] = stats(F50, kT_train, "exact-field flow NFE 50")
results["flow20_nfe50"]["ED"] = float(energy_distance(z50, ref))
z10, _ = flow(part20, 1500, 10)
F10 = to_x(z10)
results["flow20_nfe10"] = stats(F10, kT_train, "exact-field flow NFE 10")
results["flow20_nfe10"]["ED"] = float(energy_distance(z10, ref))
z2, _ = flow(part20, 1500, 2)
results["flow20_nfe2"] = stats(to_x(z2), kT_train, "exact-field flow NFE 2")
results["flow20_nfe2"]["ED"] = float(energy_distance(z2, ref))
prior = rng.standard_normal((1500, 2))
results["prior"] = stats(to_x(prior), kT_train, "N(0,I) prior")
results["prior"]["ED"] = float(energy_distance(prior, ref))
edfloor = energy_distance(to_z(exact_samples(kT_train, 2000)), ref)
results["ED_floor"] = float(edfloor)
print({k: results[k] for k in ["flow20_nfe50", "flow20_nfe10", "flow20_nfe2", "prior", "ED_floor"]})

# temperature transfer
UF = U(F50); BF = basin(F50)
for kTp in [12.0, 8.0]:
    exp_, exU, _ = exact(kTp)
    w = np.exp(-UF * (1/kTp - 1/kT_train)); w /= w.sum()
    ess = 1.0 / (w**2).sum()
    pops = np.array([w[BF == k].sum() for k in range(3)])
    # Langevin at kT' from a start in B, 2e5 steps
    Lc = langevin(minima[1:2].copy(), kTp, steps=200_000, thin=50)  # [4000,1,2]
    Lcf = Lc[:, 0, :]
    lst = stats(Lcf, kTp, "langevin from B")
    # exact-field flow trained at kT'
    partp = to_z(exact_samples(kTp, 6000))
    zp, _ = flow(partp, 1500, 50)
    fst = stats(to_x(zp), kTp, "exact-field flow at kT'")
    results[f"transfer{int(kTp)}"] = dict(exact=dict(pops=exp_.tolist(), meanU=float(exU), dF=float(dF(exp_, kTp))),
        reweighted=dict(pops=pops.tolist(), meanU=float((w*UF).sum()), dF=float(dF(pops+1e-12, kTp)), ess=float(ess)),
        langevin=lst, flow=fst)
    print(kTp, results[f"transfer{int(kTp)}"])
    if kTp == 8.0:
        L8 = Lcf; F8 = to_x(zp)

json.dump(results, open("results.json", "w"), indent=1)

# ── figure ────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1e1b", "#6b6963"
greys = LinearSegmentedColormap.from_list("warmgrey", ["#ffffff", "#ecebe6", "#d8d6cf", "#bdbab1", "#9e9b91"])
levels = np.arange(-150, 60, 10)
UGc = np.clip(UG.reshape(X.shape), -160, 60)

def surface(ax, lw=0.35):
    ax.contourf(X, Y, UGc, levels=levels, cmap=greys, extend="max")
    ax.contour(X, Y, UGc, levels=levels, colors="#8d8a80", linewidths=lw)
    ax.set_xlim(-1.6, 1.25); ax.set_ylim(-0.6, 2.1)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.tick_params(length=2, color=MUTED); 
    for s in ax.spines.values(): s.set_color("#bdbab1"); s.set_linewidth(0.6)

def label_wells(ax):
    for (mx, my), u, name, off in zip(minima, Umin, "ABC", [(-0.05, 0.13), (0.06, -0.15), (0.16, 0.02)]):
        ax.text(mx+off[0], my+off[1], f"{name}  U={u:.0f}", ha="center", va="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.scatter(saddles[:, 0], saddles[:, 1], marker="x", s=18, c=INK, linewidths=0.8, zorder=5)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
# (a) surface + Langevin training data
ax = axes[0]; surface(ax)
sub = L20f[rng.choice(len(L20f), 3000, replace=False)]
ax.scatter(sub[:, 0], sub[:, 1], s=3, c=ORANGE, alpha=0.45, linewidths=0, zorder=3)
label_wells(ax)
ax.set_title("(a)  kT = 20: the surface, and the Langevin training set", loc="left")
ax.text(0.02, 0.02, "8 chains × 10⁵ steps, thinned; orange", transform=ax.transAxes, fontsize=7.5, color=MUTED)
# (b) exact flow trajectories
ax = axes[1]; surface(ax, lw=0.25)
Tx = to_x(traj)  # [51, 1500, 2]
sel = rng.choice(1500, 70, replace=False)
for i in sel:
    ax.plot(Tx[:, i, 0], Tx[:, i, 1], color=BLUE, lw=0.7, alpha=0.55, zorder=3)
ax.scatter(Tx[0, sel, 0], Tx[0, sel, 1], s=9, facecolors="white", edgecolors=MUTED, linewidths=0.6, zorder=4)
ax.scatter(F50[:, 0], F50[:, 1], s=3, c=BLUE, alpha=0.5, linewidths=0, zorder=4)
ax.set_title("(b)  the flow, N(0, I) → exp(−U/kT), 70 of 1500 paths", loc="left")
ax.text(0.02, 0.02, "hollow: noise at t = 1;  blue: samples at t = 0, NFE 50", transform=ax.transAxes, fontsize=7.5, color=MUTED)
# (c) kT = 8, Langevin stuck vs flow
ax = axes[2]; surface(ax, lw=0.25)
ax.plot(L8[:, 0], L8[:, 1], color=ORANGE, lw=0.4, alpha=0.5, zorder=3)
ax.scatter(L8[::4, 0], L8[::4, 1], s=3, c=ORANGE, alpha=0.6, linewidths=0, zorder=4)
ax.scatter(F8[:, 0], F8[:, 1], s=3, c=BLUE, alpha=0.5, linewidths=0, zorder=4)
label_wells(ax)
t8 = results["transfer8"]
txt = ("well populations A / B / C\n"
       f"exact      {t8['exact']['pops'][0]:.3f} / {t8['exact']['pops'][1]:.3f} / {t8['exact']['pops'][2]:.3f}\n"
       f"Langevin   {t8['langevin']['pops'][0]:.3f} / {t8['langevin']['pops'][1]:.3f} / {t8['langevin']['pops'][2]:.3f}\n"
       f"flow       {t8['flow']['pops'][0]:.3f} / {t8['flow']['pops'][1]:.3f} / {t8['flow']['pops'][2]:.3f}")
ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=7.2, family="DejaVu Sans Mono", va="bottom", color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d8d6cf", lw=0.6))
ax.set_title("(c)  kT = 8: a chain started in B, 2×10⁵ steps, and the flow", loc="left")
fig.savefig("boltzmann_mb.png", dpi=190, facecolor="white")
print("saved")
