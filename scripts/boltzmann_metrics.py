#!/usr/bin/env python3
"""Score the Boltzmann-generator demo — planning/boltzmann_generator_demo.md §5, §6.1, §6.2.

The target is a density with a closed form, so every column is a number by
quadrature on the grid preprocess_boltzmann.py wrote, and the first row of
every table is exact. A sample set is scored as a free-energy estimate:

  p_A / p_B / p_C   well populations, basin by gradient DESCENT on the samples
  <U>               mean of the closed-form U over the samples
  dF_AB             -kT ln(p_A / p_B), the free-energy difference from the populations
  energy distance   2E|X-Y| - E|X-X'| - E|Y-Y'| against the independent exact draw,
                    reported as a multiple of the exact-vs-exact floor
  KL(model||exact)  from the model's own log-density (the continuity-equation
                    integration the Lean exe writes beside its samples, `logp`)
  KL(exact||model)  from the model's log-density on the exact draw (`nll` mode)

Modes:
  score     <label>=<samples.bin> ...  [--gate] [--out=DIR]
            Table 1: quadrature, the Langevin training set, every sample file given,
            the N(0, I) prior. A `<samples>.logp.bin` beside a file fills the KL
            column; a `<samples>.reflogp.bin` fills the reverse one. `--gate`
            applies the regression gate to the FIRST file: energy distance <= 10x
            the floor and every basin within 0.03 of quadrature.
  transfer  <samples.bin> [--out=DIR]
            Table 2: reweight the kT = 20 samples to kT' = 12 and 8 by
            exp(-U (1/kT' - 1/kT)), with the effective sample size, beside a fresh
            Langevin chain of 2e5 steps started in well B. With a `.logp.bin` beside
            the samples the model-corrected weights exp(-U/kT') / p_theta are a
            second row. Writes `transfer.json` and the resampled clouds into --out.
  field     <field.bin>
            §6.2: the L2 error of the dumped velocity field against the exact
            marginal velocity (x - E[x0 | x_t]) / t, per t, weighted by p_t.

Sample files are flat f32 [N, 2] in STANDARDISED coordinates (z = (x - c)/s),
the way the Lean exe writes them; U is evaluated in the original coordinates.
numpy only.
"""
import json, os, sys
import numpy as np

np.seterr(over="ignore", invalid="ignore")

DATA = "data/boltzmann"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]


def flag(name, default=None):
    return next((f.split("=", 1)[1] for f in flags if f.startswith(f"--{name}=")), default)


with open(f"{DATA}/manifest.json") as f:
    MAN = json.load(f)
GRID = np.load(f"{DATA}/mb_grid.npz")
K = MAN["constants"]
A, a, b, c = (np.array(K[k]) for k in ("A", "a", "b", "c"))
X0, Y0 = np.array(K["x0"]), np.array(K["y0"])
CEN, SC = np.array(MAN["centre"]), MAN["scale"]
KT = MAN["kT"]
MINIMA = np.array(MAN["minima"])
xs, ys = GRID["xs"], GRID["ys"]
UG = GRID["U"].reshape(-1)
BG = GRID["basin"].reshape(-1).astype(int)
G = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
dA = float(GRID["dA"])
EXACT = {float(kT): dict(pops=GRID["pops"][i], meanU=float(GRID["meanU"][i]),
                         dF=float(GRID["dF"][i]), logZ=float(GRID["logZ"][i]))
         for i, kT in enumerate(GRID["kT"])}


def U(P):
    x, y = P[..., 0:1], P[..., 1:2]
    dx, dy = x - X0, y - Y0
    return (A * np.exp(a * dx * dx + b * dx * dy + c * dy * dy)).sum(-1)


def gradU(P):
    x, y = P[..., 0:1], P[..., 1:2]
    dx, dy = x - X0, y - Y0
    e = A * np.exp(a * dx * dx + b * dx * dy + c * dy * dy)
    return np.stack([(e * (2 * a * dx + b * dy)).sum(-1),
                     (e * (b * dx + 2 * c * dy)).sum(-1)], -1)


def descend(P, steps=1500, eta=1e-4, cap=0.01):
    P = P.copy()
    for _ in range(steps):
        g = gradU(P)
        n = np.linalg.norm(g, axis=-1, keepdims=True)
        P -= eta * g * np.minimum(1.0, cap / (eta * n + 1e-12))
    return P


def basin(P):
    d = ((descend(P)[:, None, :] - MINIMA[None]) ** 2).sum(-1)
    return d.argmin(1)


to_x = lambda Z: Z.astype(np.float64) * SC + CEN
to_z = lambda P: (P - CEN) / SC


def load(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 2)


def energy_distance(x, y, cap=2048, seed=0):
    """O(n^2); subsample so it stays in ms. Same statistic as toy2d_metrics.py."""
    rng = np.random.default_rng(seed)
    if len(x) > cap: x = x[rng.choice(len(x), cap, replace=False)]
    if len(y) > cap: y = y[rng.choice(len(y), cap, replace=False)]
    d = lambda p, q: np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1))
    return float(2 * d(x, y).mean() - d(x, x).mean() - d(y, y).mean())


REF = load(f"{DATA}/mb_kT20_ref.bin")
half = len(REF) // 2
ED_FLOOR = energy_distance(REF[:half], REF[half:])


def log_p_exact(Z, kT):
    """log density of exp(-U/kT)/Z in STANDARDISED coordinates (Jacobian s^2)."""
    return -U(to_x(Z)) / kT - EXACT[kT]["logZ"] + 2.0 * np.log(SC)


U_CAP = 1000.0   # a sample in the wall (U > 1000, the highest saddle is -41) counts as 1000, and is counted


def dF_of(pops, kT):
    """-kT ln(p_A / p_B); +-inf when a well is empty rather than a clamp artefact."""
    if pops[0] <= 0 or pops[1] <= 0:
        return float("inf") if pops[0] <= 0 else float("-inf")
    return float(-kT * np.log(pops[0] / pops[1]))


def row_stats(Z, kT=KT, w=None):
    """Populations, mean energy and dF of a (possibly weighted) cloud."""
    Ux = U(to_x(Z))
    bs = basin(to_x(Z))
    if w is None:
        w = np.full(len(Z), 1.0 / len(Z))
    pops = np.array([w[bs == k].sum() for k in range(3)])
    return dict(pops=pops, meanU=float((w * np.minimum(Ux, U_CAP)).sum()), dF=dF_of(pops, kT),
                basins=bs, nWall=int((Ux > U_CAP).sum()))


def fmt_row(label, st, ed=None, kl=None, klr=None, ess=None, w=28):
    p = st["pops"]
    if st.get("nWall"):
        label = f"{label} [{st['nWall']} in the wall]"
    dF = f"{st['dF']:6.1f}" if np.isfinite(st["dF"]) else f"{'--':>6}"
    s = f"  {label:<{w}} {p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f}   {st['meanU']:7.1f}   {dF}"
    if ed is not None:
        s += f"   {ed / ED_FLOOR:6.1f}x ({ed:.4f})" if ed > 0 else f"   {'floor':>6} ({ed:.4f})"
    else:
        s += " " * 20
    if kl is not None or klr is not None:
        s += f"   {kl if kl is not None else float('nan'):6.3f} / {klr if klr is not None else float('nan'):6.3f}"
    if ess is not None:
        s += f"   {ess:.0f}"
    return s


def kl_from_logp(Z, logp, kT=KT):
    """KL(model || exact) = E_model[log p_theta - log p]. Finite only where the
    model puts mass inside the box; outside it the exact density is what U says."""
    return float(np.mean(logp - log_p_exact(Z, kT)))


def kl_reverse_from_reflogp(reflogp, kT=KT):
    """KL(exact || model) = E_exact[log p - log p_theta] on the exact draw."""
    return float(np.mean(log_p_exact(REF[:len(reflogp)], kT) - reflogp))


# ── score: table 1 ──────────────────────────────────────────────────────────
def score(items):
    ex = EXACT[KT]
    print(f"kT = {KT:.0f}, reference {len(REF)} exact points, energy-distance floor "
          f"(exact vs exact, n = {half}) = {ED_FLOOR:.4f}")
    print()
    hdr = f"  {'source':<28} {'p_A / p_B / p_C':<23} {'<U>':>7}   {'dF_AB':>6}   {'energy (x floor)':<18}   KL(m||e) / KL(e||m)"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(fmt_row("quadrature (exact)", dict(pops=ex["pops"], meanU=ex["meanU"], dF=ex["dF"]), ED_FLOOR))
    train = load(f"{DATA}/mb_kT20.bin")
    st = row_stats(train)
    print(fmt_row(f"Langevin, {MAN['langevin']['chains']} chains x 1e5 steps", st, energy_distance(train, REF)))
    results = []
    for label, path in items:
        Z = load(path)
        st = row_stats(Z)
        ed = energy_distance(Z, REF)
        kl = klr = None
        lp = path[:-4] + ".logp.bin"
        if os.path.exists(lp):
            logp = np.fromfile(lp, dtype=np.float32)[:len(Z)].astype(np.float64)
            kl = kl_from_logp(Z, logp)
        rp = path[:-4] + ".reflogp.bin"
        if os.path.exists(rp):
            klr = kl_reverse_from_reflogp(np.fromfile(rp, dtype=np.float32).astype(np.float64))
        print(fmt_row(f"{label} (n={len(Z)})", st, ed, kl, klr))
        results.append(dict(label=label, n=int(len(Z)), pops=st["pops"].tolist(), meanU=st["meanU"],
                            dF=st["dF"], ed=ed, ed_x_floor=ed / ED_FLOOR, kl=kl, kl_rev=klr))
        # Model-corrected populations at the training temperature: importance
        # weights p_20 / p_theta on the model's own samples. The exact row
        # recovered from an imperfect model, if the density is right.
        if kl is not None:
            lw = log_p_exact(Z, KT) - logp
            w = np.exp(lw - lw.max()); w /= w.sum()
            stc = row_stats(Z, w=w)
            print(fmt_row(f"  ^ reweighted by p_20/p_theta", stc, None, None, None, 1.0 / (w ** 2).sum()))
            results[-1]["corrected"] = dict(pops=stc["pops"].tolist(), meanU=stc["meanU"], dF=stc["dF"],
                                            ess=float(1.0 / (w ** 2).sum()))
    prior = np.random.default_rng(0).standard_normal((2048, 2))
    print(fmt_row("N(0, I) prior, no flow", row_stats(prior), energy_distance(prior, REF)))
    print()
    print(f"  at n = 2048 a population carries about +-0.01 of sampling error; the KL columns "
          f"are nats, KL(m||e) from the model's samples, KL(e||m) on the exact draw")
    out = flag("out")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(f"{out}/table1.json", "w") as f:
            json.dump(dict(exact=dict(pops=ex["pops"].tolist(), meanU=ex["meanU"], dF=ex["dF"]),
                           ed_floor=ED_FLOOR, rows=results), f, indent=1)
    if "--gate" in flags and results:
        r = results[0]
        dp = np.abs(np.array(r["pops"]) - ex["pops"]).max()
        ok = r["ed_x_floor"] <= 10.0 and dp <= 0.03
        print(f"  => gate on '{r['label']}': energy {r['ed_x_floor']:.1f}x floor (need <= 10x) and "
              f"max basin error {dp:.3f} (need <= 0.03)  [{'PASS' if ok else 'FAIL'}]")
        sys.exit(0 if ok else 1)


# ── transfer: table 2 ───────────────────────────────────────────────────────
def transfer(path):
    Z = load(path)
    Ux = U(to_x(Z))
    bs = basin(to_x(Z))
    lp = path[:-4] + ".logp.bin"
    logp = np.fromfile(lp, dtype=np.float32)[:len(Z)].astype(np.float64) if os.path.exists(lp) else None
    out = flag("out")
    res = {}
    print(f"samples drawn once at kT = {KT:.0f} (n = {len(Z)}), reweighted to kT'")
    print()
    hdr = f"  {'kT':>3}  {'source':<36} {'p_A / p_B / p_C':<23} {'<U>':>7}   {'dF_AB':>6}   ESS"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for kTp in MAN["kTs"][1:]:
        ex = EXACT[float(kTp)]
        rows = {}
        print(f"  {kTp:>3.0f}  " + fmt_row("quadrature (exact)", dict(pops=ex["pops"], meanU=ex["meanU"], dF=ex["dF"]), w=36)[2:])
        rows["exact"] = dict(pops=ex["pops"].tolist(), meanU=ex["meanU"], dF=ex["dF"])
        Lc = load(f"{DATA}/mb_kT{int(kTp)}_langevinB.bin")
        st = row_stats(Lc, kT=kTp)
        print("       " + fmt_row("Langevin from B, 2e5 steps", st, w=36)[2:])
        rows["langevin"] = dict(pops=st["pops"].tolist(), meanU=st["meanU"], dF=st["dF"])
        # (a) assume the model's density is exactly p_20
        lw = -Ux * (1.0 / kTp - 1.0 / KT)
        w = np.exp(lw - lw.max()); w /= w.sum()
        ess = 1.0 / (w ** 2).sum()
        pops = np.array([w[bs == k].sum() for k in range(3)])
        st = dict(pops=pops, meanU=float((w * np.minimum(Ux, U_CAP)).sum()), dF=dF_of(pops, kTp))
        print("       " + fmt_row(f"flow at kT = {KT:.0f}, reweighted", st, ess=ess, w=36)[2:])
        rows["reweighted"] = dict(pops=pops.tolist(), meanU=st["meanU"], dF=st["dF"], ess=float(ess))
        if out:
            os.makedirs(out, exist_ok=True)
            rs = np.random.default_rng(int(kTp)).choice(len(Z), size=len(Z), p=w)
            Z[rs].astype(np.float32).tofile(f"{out}/reweighted_kT{int(kTp)}.bin")
        # (b) the model-error correction: w = exp(-U/kT') / p_theta(z)
        if logp is not None:
            lw2 = -Ux / kTp - logp
            w2 = np.exp(lw2 - lw2.max()); w2 /= w2.sum()
            ess2 = 1.0 / (w2 ** 2).sum()
            pops2 = np.array([w2[bs == k].sum() for k in range(3)])
            st2 = dict(pops=pops2, meanU=float((w2 * np.minimum(Ux, U_CAP)).sum()), dF=dF_of(pops2, kTp))
            print("       " + fmt_row("  ^ corrected by 1/p_theta", st2, ess=ess2, w=36)[2:])
            rows["corrected"] = dict(pops=pops2.tolist(), meanU=st2["meanU"], dF=st2["dF"], ess=float(ess2))
            if out:
                rs = np.random.default_rng(int(kTp) + 100).choice(len(Z), size=len(Z), p=w2)
                Z[rs].astype(np.float32).tofile(f"{out}/corrected_kT{int(kTp)}.bin")
        res[str(int(kTp))] = rows
    print()
    print("  dF within 2 units of quadrature at kT' = 12 is gate B; the Langevin row is the MCMC control")
    if out:
        with open(f"{out}/transfer.json", "w") as f:
            json.dump(res, f, indent=1)
        print(f"  wrote {out}/transfer.json and the resampled clouds")
    # Gate B tests the section's claim: a sample set drawn once at kT = 20 gives
    # the free energy at kT' = 12 within 2 units. With the model's density in
    # hand that is the corrected row (Noe et al.'s weights, exp(-U/kT')/p_theta);
    # without it the naive row carries the model's own bias in full.
    key = "corrected" if "corrected" in res["12"] else "reweighted"
    d12 = abs(res["12"][key]["dF"] - res["12"]["exact"]["dF"])
    dn = abs(res["12"]["reweighted"]["dF"] - res["12"]["exact"]["dF"])
    print(f"  => gate B ({key} row): |dF - dF_exact| at kT' = 12 is {d12:.1f} (naive reweighting: {dn:.1f})  "
          f"[{'PASS' if d12 <= 2.0 else 'FAIL'}]")
    sys.exit(0 if d12 <= 2.0 else 1)


# ── field: §6.2, the velocity field against its exact marginal ──────────────
def field(path):
    """The Lean `field` mode writes a header line `nT side lo hi t0 t1 ...` in
    `<path>.txt` and v_theta as f32 [nT, side, side, 2] on a lattice over
    [lo, hi]^2 in standardised coordinates."""
    with open(path[:-4] + ".txt") as f:
        parts = f.read().split()
    nT, side = int(parts[0]), int(parts[1])
    lo, hi = float(parts[2]), float(parts[3])
    ts = [float(v) for v in parts[4:4 + nT]]
    V = np.fromfile(path, dtype=np.float32).reshape(nT, side * side, 2).astype(np.float64)
    g = np.linspace(lo, hi, side)
    L = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)          # lattice, standardised
    # The particle set: the grid density, coarsened 3x so the pair count stays
    # in the tens of millions per t.
    sub = np.arange(0, len(xs), 3)
    P = to_z(np.stack(np.meshgrid(xs[sub], ys[sub]), -1).reshape(-1, 2))
    Up = GRID["U"][np.ix_(sub, sub)].reshape(-1)
    logw0 = -(Up - Up.min()) / KT
    w0 = np.exp(logw0); w0 /= w0.sum()
    print(f"field on a {side}x{side} lattice over [{lo}, {hi}]^2, {nT} times, {len(P)} particles")
    print()
    print(f"  {'t':>5}   {'E_pt |v_theta - v*|^2':>22}   {'E_pt |v*|^2':>12}   rel")
    tot_err = tot_kin = 0.0
    for i, t in enumerate(ts):
        # posterior over particles: w0 * N(x_t; (1-t) x0, t^2 I)
        errs = np.zeros(len(L)); vs2 = np.zeros(len(L)); pt = np.zeros(len(L))
        for lo_i in range(0, len(L), 512):
            Lb = L[lo_i:lo_i + 512]
            d2 = ((Lb[:, None, :] - (1 - t) * P[None]) ** 2).sum(-1)
            logw = np.log(w0)[None] - d2 / (2 * t * t)
            m = logw.max(1, keepdims=True)
            w = np.exp(logw - m)
            s = w.sum(1, keepdims=True)
            Ez0 = (w @ P) / s
            vstar = (Lb - Ez0) / t
            pt[lo_i:lo_i + 512] = (np.exp(m[:, 0]) * s[:, 0]) / (2 * np.pi * t * t)
            errs[lo_i:lo_i + 512] = ((V[i, lo_i:lo_i + 512] - vstar) ** 2).sum(-1)
            vs2[lo_i:lo_i + 512] = (vstar ** 2).sum(-1)
        pw = pt / pt.sum()
        e, k = float((pw * errs).sum()), float((pw * vs2).sum())
        tot_err += e; tot_kin += k
        print(f"  {t:5.2f}   {e:22.4f}   {k:12.4f}   {e / max(k, 1e-12):5.3f}")
    print()
    print(f"  mean over t: field error {tot_err / nT:.4f}, exact kinetic energy {tot_kin / nT:.4f}")


if not args:
    sys.exit(__doc__)
mode = args[0]
if mode == "score":
    items = []
    for it in args[1:]:
        if "=" in it:
            lab, p = it.split("=", 1)
        else:
            lab, p = os.path.basename(it), it
        items.append((lab, p))
    score(items)
elif mode == "transfer":
    transfer(args[1])
elif mode == "field":
    field(args[1])
else:
    sys.exit(f"unknown mode {mode!r}: score | transfer | field")
