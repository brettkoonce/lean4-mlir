#!/usr/bin/env python3
"""Instrument and bracket for the neural-quantum-state demo —
planning/transformer_wavefunction_demo.md §2, §5, §8.

Transverse-field Ising chain, periodic, H = -J sum_i s^z_i s^z_{i+1} - h sum_i s^x_i.
Every number a trained wavefunction reports is bracketed by two closed forms:

  enumeration  N <= 14: the 2^N x 2^N Hamiltonian, `eigh` for E0 and the ground
               state, from which <s^x> and <s^z_1 s^z_{1+r}> are exact sums.
  Jordan-Wigner any even N: the free-fermion sum with the even-parity sector's
               antiperiodic momenta k = (2m+1) pi / N (Lieb, Schultz & Mattis
               1961; Pfeuty 1970). E0 = -sum_k sqrt(J^2 + h^2 - 2 J h cos k),
               <s^x> = -(1/N) dE0/dh by Hellmann-Feynman, and the correlation
               <s^z_0 s^z_r> is the r x r Toeplitz determinant det[G(a - b - 1)]
               of the fermion contraction G(n) = <B_0 A_n> (A = c† + c, B = c† - c),
               in the rotated frame where the coupling is s^x s^x and the field s^z:
               G(n) = -(1/N) sum_k [cos(kn) (h - J cos k) + sin(kn) J sin k] / eps_k
               with eps_k = sqrt(J^2 + h^2 - 2 J h cos k). The index convention
               is pinned by `gate`, which demands the two instruments agree to
               1e-10 at N = 12 before any N = 64 number is quoted (§8); they
               agree to 7e-15.

Modes:
  gate                          N = 12, h in 0.2..2.0: enumeration vs Jordan-Wigner
  exact N h [--J=1]             print E0, <s^x>, C(r) for r = 1..N/2 (both instruments)
  mf N h                        the mean-field product-state row (the floor, §3 R0)
  score <label>=<metrics.json> ...  [--out=DIR] [--gate]
                                one row per file the Lean exe wrote, columns
                                (E - E0)/|E0|, Var(E_loc), <s^x>, C(N/2), each
                                beside its exact value; `--gate` applies §5's
                                regression gate (rel err <= 1e-4, var <= 1e-2)
                                to the FIRST file. Writes table.md into --out.
  ladder <arch>=<json> ...      same as score, one line per file, machine-readable
  j1j2 N J2 [--J1=1]            the J1-J2 chain: E0 by Lanczos in the S_z = 0 sector, the
                                Marshall-sign check at J2 = 0, the Majumdar-Ghosh −3/8 at J2 = J1/2,
                                and the floor (uniform amplitudes × Marshall signs)
  table3 <dir> [--out=FILE]     Table 3: J1-J2 rows (arch × J2 × sign prior) with rel err,
                                Var(E_loc), fidelity to the ED vector (from `<prefix>_psi.bin`),
                                C(N/2)
  sweep <dir> [--out=FILE]      the field sweep: one row per h found in <dir>/*_metrics.json
                                (mean field, then every arch's (E - E0)/|E0| and Var(E_loc),
                                then C(N/2) exact and per arch); `_noref`/`_symref` files
                                become their own columns
  ablation <dir> [--out=FILE]   Table 2: uniform start vs reference per (arch, h) in <dir>

The Lean exe (`lake exe nqs-ising`) writes `<prefix>_metrics.json` with the
sample-mean energy, the variance of the local energy, <s^x> and the full
correlation function C(r), r = 0..N/2, measured on its own samples (exact
weights at N <= 14, Monte Carlo above). numpy only.
"""
import functools, json, math, os, sys
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]


def flag(name, default=None):
    return next((f.split("=", 1)[1] for f in flags if f.startswith(f"--{name}=")), default)


# ─── enumeration, N <= 14 ───────────────────────────────────────────────────

def spins(N):
    """[2^N, N] array of ±1: bit i of the index is spin i, bit set = +1."""
    c = np.arange(1 << N, dtype=np.int64)
    return np.where(((c[:, None] >> np.arange(N)[None, :]) & 1) == 1, 1.0, -1.0)


def hamiltonian(N, h, J=1.0):
    S = spins(N)
    diag = -J * np.sum(S * np.roll(S, -1, axis=1), axis=1)
    H = np.diag(diag)
    c = np.arange(1 << N)
    for i in range(N):
        H[c, c ^ (1 << i)] -= h
    return H, S


@functools.lru_cache(maxsize=None)
def enumerate_exact(N, h, J=1.0):
    """Cached: a 4096 x 4096 `eigh` is seconds, and the tables ask per h many times."""
    H, S = hamiltonian(N, h, J)
    w, v = np.linalg.eigh(H)
    psi = v[:, 0]
    p = psi * psi
    E0 = w[0]
    # <s^x>: sum over configs of psi(c) psi(c ^ bit i)
    c = np.arange(1 << N)
    sx = np.mean([np.sum(psi[c] * psi[c ^ (1 << i)]) for i in range(N)])
    corr = np.array([np.sum(p * S[:, 0] * S[:, r % N]) for r in range(N // 2 + 1)])
    return dict(E0=float(E0), sx=float(sx), corr=corr, gap=float(w[1] - w[0]))


# ─── Jordan-Wigner, any even N ──────────────────────────────────────────────

def jw_momenta(N):
    return (2 * np.arange(N) + 1) * np.pi / N


@functools.lru_cache(maxsize=None)
def jw_exact(N, h, J=1.0, rmax=None):
    if N % 2:
        raise SystemExit("Jordan-Wigner here is the even-N, even-parity sector")
    k = jw_momenta(N)
    eps = np.sqrt(J * J + h * h - 2 * J * h * np.cos(k))
    E0 = -np.sum(eps)
    sx = np.mean((h - J * np.cos(k)) / eps)
    # fermion contraction in the rotated frame (coupling s^x s^x, field s^z):
    #   G(n) = <B_0 A_n>, A = c^dag + c, B = c^dag - c
    def G(n):
        return -np.mean(np.cos(k * n) * (h - J * np.cos(k)) / eps
                        + np.sin(k * n) * (J * np.sin(k)) / eps)
    rmax = N // 2 if rmax is None else rmax
    corr = [1.0]
    for r in range(1, rmax + 1):
        M = np.array([[G(a - b - 1) for b in range(r)] for a in range(r)])
        corr.append(float(np.linalg.det(M)))
    return dict(E0=float(E0), sx=float(sx), corr=np.array(corr))


def exact(N, h, J=1.0):
    h, J = round(float(h), 8), round(float(J), 8)
    return enumerate_exact(N, h, J) if N <= 14 else jw_exact(N, h, J)


# ─── the mean-field product state (§1: the reference, §3 R0: the floor) ─────

def mean_field(N, h, J=1.0):
    """|phi*>^N with sin(phi) = h/2J (h < 2J) else phi = pi/2. Per-site
    energy -J cos^2(phi) - h sin(phi); <s^x> = sin(phi); C(r) = cos^2(phi)."""
    phi = math.asin(h / (2 * J)) if h < 2 * J else math.pi / 2
    c, s = math.cos(phi), math.sin(phi)
    corr = np.array([1.0] + [c * c] * (N // 2))
    return dict(E=N * (-J * c * c - h * s), sx=s, corr=corr, phi=phi)


# ─── the J1-J2 Heisenberg chain, S_z = 0 sector (§2, rung R4) ───────────────
#   H = J1 Σ S_i·S_{i+1} + J2 Σ S_i·S_{i+2}, S = σ/2, periodic. In the σ^z basis the
#   diagonal is Σ_b J_b σ_i σ_j / 4 and an antiparallel bond exchanges with J_b / 2.

def sector_basis(N):
    c = np.arange(1 << N, dtype=np.int64)
    pc = np.zeros_like(c)
    for i in range(N):
        pc += (c >> i) & 1
    return c[pc == N // 2]


def marshall_phase(basis, N):
    """(−1)^(up spins on the even sublattice), as a phase 0 / π."""
    nA = np.zeros_like(basis)
    for i in range(0, N, 2):
        nA += (basis >> i) & 1
    return np.where(nA % 2 == 1, np.pi, 0.0)


def j1j2_bonds(N, J1, J2):
    return [(i, (i + 1) % N, J1) for i in range(N)] + [(i, (i + 2) % N, J2) for i in range(N)]


@functools.lru_cache(maxsize=None)
def j1j2_exact(N, J2, J1=1.0):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    basis = sector_basis(N)
    D = basis.size
    S = np.where(((basis[:, None] >> np.arange(N)[None, :]) & 1) == 1, 1.0, -1.0)
    rows, cols, vals = [np.arange(D)], [np.arange(D)], [np.zeros(D)]
    for i, j, Jb in j1j2_bonds(N, J1, J2):
        vals[0] += Jb * S[:, i] * S[:, j] / 4.0
        anti = S[:, i] != S[:, j]
        src = basis[anti]
        dst = src ^ ((1 << i) | (1 << j))
        rows.append(np.searchsorted(basis, dst))
        cols.append(np.arange(D)[anti])
        vals.append(np.full(src.size, Jb / 2.0))
    H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(D, D)).tocsr()
    if D <= 3000:
        w, v = np.linalg.eigh(H.toarray())
        w, v = w[:6], v[:, :6]
    else:
        w, v = spl.eigsh(H, k=6, which="SA", tol=1e-12)
        o = np.argsort(w)
        w, v = w[o], v[:, o]
    E0 = w[0]
    # the ground SPACE: at the Majumdar-Ghosh point the two dimerisations are
    # degenerate on a ring, and a fidelity against one vector would be arbitrary
    ground = v[:, w - E0 < 1e-8]
    E1 = w[ground.shape[1]] if ground.shape[1] < w.size else float("nan")
    psi = v[:, 0]
    psi = psi / np.linalg.norm(psi)
    # fix the global sign so the Marshall convention reads as +
    ph = marshall_phase(basis, N)
    ms = np.where(ph > 1, -1.0, 1.0)
    if np.sum(psi * ms) < 0:
        psi = -psi
    p = psi * psi
    corr = np.array([np.sum(p * S[:, 0] * S[:, r % N]) for r in range(N // 2 + 1)])
    # how much of the weight carries the Marshall sign
    marshall = float(np.sum(p[np.sign(psi) == ms]))
    # the floor: uniform amplitudes times the Marshall signs
    u = ms / np.sqrt(D)
    E_floor = float(u @ (H @ u))
    return dict(E0=float(E0), gap=float(E1 - E0), psi=psi, ground=ground, basis=basis, corr=corr,
                marshall=marshall, E_floor=E_floor, D=D, degeneracy=int(ground.shape[1]))


def read_psi(path, N):
    """<prefix>_psi.bin from the exe: [M, 2] f32 of (log|ψ|, φ) in the sector's
    ascending configuration order, the order `sector_basis` produces."""
    a = np.fromfile(path, dtype=np.float32).reshape(-1, 2)
    cfg = sector_basis(N)
    if cfg.size != a.shape[0]:
        raise SystemExit(f"{path}: {a.shape[0]} rows, sector has {cfg.size}")
    return cfg, a[:, 0].astype(np.float64), a[:, 1].astype(np.float64)


def fidelity(ex, psi_path, N):
    cfg, la, ph = read_psi(psi_path, N)
    idx = np.searchsorted(ex["basis"], cfg)
    amp = np.exp(la - la.max()) * np.exp(1j * ph)
    amp = amp / np.linalg.norm(amp)
    # projection onto the ground space (one vector away from a degeneracy)
    G = ex["ground"][idx].astype(complex)
    return float(np.sum(np.abs(G.conj().T @ amp) ** 2))


# ─── modes ──────────────────────────────────────────────────────────────────

def sci(x):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.1f}e{e:+03d}"


def mode_gate():
    N = 12
    worst = 0.0
    print(f"gate: enumeration vs Jordan-Wigner at N = {N}")
    print(f"{'h':>5} {'E0 enum':>14} {'E0 JW':>14} {'|dE0|':>9} {'|d<sx>|':>9} {'max|dC(r)|':>11}")
    for h in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        a = enumerate_exact(N, h)
        b = jw_exact(N, h)
        dE = abs(a["E0"] - b["E0"])
        dx = abs(a["sx"] - b["sx"])
        dC = np.max(np.abs(a["corr"] - b["corr"]))
        worst = max(worst, dE, dx, dC)
        print(f"{h:5.1f} {a['E0']:14.8f} {b['E0']:14.8f} {dE:9.1e} {dx:9.1e} {dC:11.1e}")
    ok = worst < 1e-10
    print(f"worst disagreement {worst:.1e}  ->  {'PASS' if ok else 'FAIL'} (gate 1e-10)")
    sys.exit(0 if ok else 1)


def mode_exact(N, h, J):
    ex = exact(N, h, J)
    mf = mean_field(N, h, J)
    print(f"N = {N}, h = {h}, J = {J}")
    print(f"  E0          {ex['E0']:.10f}   (per site {ex['E0']/N:.10f})")
    print(f"  <s^x>       {ex['sx']:.10f}")
    print("  C(r)        " + " ".join(f"{c:.6f}" for c in ex["corr"]))
    print(f"  mean field  E = {mf['E']:.10f}  rel err {(mf['E']-ex['E0'])/abs(ex['E0']):.3e}  "
          f"<s^x> = {mf['sx']:.6f}  C = {mf['corr'][1]:.6f}")
    if N <= 14:
        print(f"  gap         {ex['gap']:.6e}")
    if N % 2 == 0 and N <= 14:
        jw = jw_exact(N, h, J)
        print(f"  JW check    |dE0| = {abs(jw['E0']-ex['E0']):.1e}, "
              f"max|dC| = {np.max(np.abs(jw['corr']-ex['corr'])):.1e}")


def mode_j1j2(N, J2, J1):
    ex = j1j2_exact(N, J2, J1)
    print(f"J1-J2 chain, N = {N}, J1 = {J1}, J2 = {J2}: sector S_z = 0 has {ex['D']} states")
    print(f"  E0          {ex['E0']:.10f}   (per site {ex['E0']/N:.10f})   gap {ex['gap']:.6f}"
          f"   ground-space dimension {ex['degeneracy']}")
    print(f"  floor       uniform × Marshall: E = {ex['E_floor']:.10f}  rel {(ex['E_floor']-ex['E0'])/abs(ex['E0']):.3e}")
    print(f"  Marshall    weight on the Marshall sign: {ex['marshall']:.10f}")
    print("  C(r)        " + " ".join(f"{c:.6f}" for c in ex["corr"]))
    if abs(J2 - 0.5 * J1) < 1e-12:
        print(f"  Majumdar-Ghosh: E0/N = {ex['E0']/N:.12f} vs −3/8 J1 = {-0.375*J1:.12f}  "
              f"(|diff| {abs(ex['E0']/N + 0.375*J1):.1e})")


def exact_for(m):
    if m.get("model", "ising") == "j1j2":
        return j1j2_exact(int(m["N"]), round(float(m["J2"]), 8), round(float(m.get("J", 1.0)), 8))
    return exact(int(m["N"]), float(m["h"]), float(m.get("J", 1.0)))


def mode_table3(d, out_path):
    ms = _load_dir(d)
    ms = [m for m in ms if m.get("model") == "j1j2"]
    if not ms:
        raise SystemExit(f"no j1j2 *_metrics.json in {d}")
    N = ms[0]["N"]
    r2 = N // 2
    name = {"mlp": "MLP", "vit": "ViT", "gpt": "GPT"}
    lines = [f"| N = {N} | J2/J1 | sign prior | (E − E0)/\\|E0\\| | Var(E_loc) | fidelity | C({r2}) (exact) |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for J2 in sorted({round(m["J2"], 6) for m in ms}):
        ex = j1j2_exact(N, J2, 1.0)
        lines.append(f"| uniform × Marshall (floor) | {J2:g} | yes | {sci((ex['E_floor']-ex['E0'])/abs(ex['E0']))} | · | · | · |")
        for a in ("mlp", "vit"):
            for v in ("ref", "noref"):
                m = next((m for m in ms if m["arch"] == a and _variant(m) == v and abs(m["J2"] - J2) < 1e-6), None)
                if m is None:
                    continue
                rel = (m["E"] - ex["E0"]) / abs(ex["E0"])
                psi_path = m["_file"].replace("_metrics.json", "_psi.bin")
                F = fidelity(ex, psi_path, N) if os.path.exists(psi_path) else float("nan")
                lines.append(f"| {name[a]} | {J2:g} | {'yes' if v == 'ref' else 'no'} | {sci(rel)} | {sci(m['var'])} | "
                             f"{F:.4f} | {m['corr'][r2]:.4f} ({ex['corr'][r2]:.4f}) |")
        mg = "  (Majumdar-Ghosh: exactly −3/8 per site)" if abs(J2 - 0.5) < 1e-9 else ""
        lines.append(f"| exact (Lanczos, {ex['D']} states) | {J2:g} | — | 0 | 0 | 1 | {ex['corr'][r2]:.4f} |{mg}")
    t = "\n".join(lines)
    print(t)
    if out_path:
        with open(out_path, "w") as f:
            f.write(t + "\n")


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def score_rows(items):
    rows = []
    for label, path in items:
        m = load_metrics(path)
        N, h, J = int(m["N"]), float(m["h"]), float(m.get("J", 1.0))
        ex = exact_for(m)
        rel = (m["E"] - ex["E0"]) / abs(ex["E0"])
        r2 = N // 2
        rows.append(dict(label=label, N=N, h=h, E=m["E"], E0=ex["E0"], rel=rel,
                         var=m["var"], sx=m["sx"], sx0=ex["sx"],
                         corr=m["corr"][r2], corr0=ex["corr"][r2],
                         corr_all=m["corr"], corr0_all=list(ex["corr"]),
                         params=m.get("params", 0), steps=m.get("steps", 0),
                         seconds=m.get("seconds", 0), arch=m.get("arch", "?"),
                         useRef=m.get("useRef", True), samples=m.get("samples", 0)))
    return rows


def table(rows):
    N, h = rows[0]["N"], rows[0]["h"]
    r2 = N // 2
    mf = mean_field(N, h)
    ex = exact(N, h)
    hdr = f"| N = {N}, h/J = {h:g} | params | (E - E0)/\\|E0\\| | Var(E_loc) | <s^x> | C({r2}) |"
    out = [hdr, "|---|---:|---:|---:|---:|---:|"]
    out.append(f"| mean field (floor) | 0 | {sci((mf['E']-ex['E0'])/abs(ex['E0']))} | 0 | "
               f"{mf['sx']:.4f} | {mf['corr'][r2]:.4f} |")
    for r in rows:
        out.append(f"| {r['label']} | {r['params']:,} | {sci(r['rel'])} | {sci(r['var'])} | "
                   f"{r['sx']:.4f} | {r['corr']:.4f} |")
    src = "enumeration" if N <= 14 else "Jordan-Wigner"
    out.append(f"| exact ({src}) | — | 0 | 0 | {ex['sx']:.4f} | {ex['corr'][r2]:.4f} |")
    return "\n".join(out)


def mode_score(items, out_dir, gate):
    rows = score_rows(items)
    t = table(rows)
    print(t)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "table.md"), "w") as f:
            f.write(t + "\n")
        with open(os.path.join(out_dir, "score.json"), "w") as f:
            json.dump(rows, f, indent=1)
    if gate:
        r = rows[0]
        ok = abs(r["rel"]) <= 1e-4 and r["var"] <= 1e-2
        print(f"gate on '{r['label']}': rel err {r['rel']:.2e} (<= 1e-4), "
              f"var {r['var']:.2e} (<= 1e-2) -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


def _variant(m):
    if not m.get("useRef", True):
        return "noref"
    if m.get("symRef", False):
        return "symref"
    return "ref"


def _load_dir(d):
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*_metrics.json"))):
        m = load_metrics(f)
        m["_file"] = f
        out.append(m)
    return out


def mode_sweep(d, out_path):
    ms = _load_dir(d)
    if not ms:
        raise SystemExit(f"no *_metrics.json in {d}")
    N = ms[0]["N"]
    hs = sorted({round(m["h"], 6) for m in ms})
    cols = []
    for a in ("mlp", "vit", "gpt"):
        for v in ("ref", "symref", "noref"):
            if any(m["arch"] == a and _variant(m) == v for m in ms):
                cols.append((a, v))
    def name(a, v):
        n = {"mlp": "MLP", "vit": "ViT", "gpt": "GPT"}[a]
        return n + {"ref": "", "symref": " (sym. ref)", "noref": " (uniform start)"}[v]
    r2 = N // 2
    src = "enumeration" if N <= 14 else "Jordan-Wigner"
    hdr = f"| h/J | E0 ({src}) | mean field |" + "".join(f" {name(a, v)} |" for a, v in cols) \
          + "".join(f" Var, {name(a, v)} |" for a, v in cols) + f" C({r2}) exact |" \
          + "".join(f" C({r2}), {name(a, v)} |" for a, v in cols)
    lines = [hdr, "|---:|---:|---:|" + "---:|" * (3 * len(cols) + 1)]
    for h in hs:
        ex = exact(N, h)
        mf = mean_field(N, h)
        row = f"| {h:g} | {ex['E0']:.4f} | {sci((mf['E'] - ex['E0']) / abs(ex['E0']))} |"
        rels, vars_, cs = [], [], []
        for a, v in cols:
            m = next((m for m in ms if m["arch"] == a and _variant(m) == v and abs(m["h"] - h) < 1e-6), None)
            if m is None:
                rels.append("·"); vars_.append("·"); cs.append("·")
            else:
                rel = (m["E"] - ex["E0"]) / abs(ex["E0"])
                se = math.sqrt(max(m["var"], 0) / max(m.get("samples", 1), 1)) / abs(ex["E0"])
                rels.append(sci(rel) + (f" ± {sci(se)}" if N > 14 else ""))
                vars_.append(sci(m["var"]))
                cs.append(f"{m['corr'][r2]:.4f}")
        row += "".join(f" {x} |" for x in rels) + "".join(f" {x} |" for x in vars_)
        row += f" {ex['corr'][r2]:.4f} |" + "".join(f" {x} |" for x in cs)
        lines.append(row)
    t = "\n".join(lines)
    print(t)
    if out_path:
        with open(out_path, "w") as f:
            f.write(t + "\n")


def mode_ablation(d, out_path):
    ms = _load_dir(d)
    N = ms[0]["N"]
    keys = sorted({(m["arch"], round(m["h"], 6)) for m in ms if _variant(m) == "noref"},
                  key=lambda k: (k[1], k[0]))
    lines = ["| N = 12 | h/J | uniform start | mean-field reference | ratio |", "|---|---:|---:|---:|---:|"]
    for a, h in keys:
        ex = exact(N, h)
        def rel_of(v):
            m = next((m for m in ms if m["arch"] == a and _variant(m) == v and abs(m["h"] - h) < 1e-6), None)
            return None if m is None else (m["E"] - ex["E0"]) / abs(ex["E0"])
        u, r = rel_of("noref"), rel_of("ref")
        ratio = "·" if (u is None or r is None or r == 0) else f"{u / r:,.0f}×"
        name = {"mlp": "MLP residual", "vit": "ViT residual", "gpt": "GPT"}[a]
        lines.append(f"| {name} | {h:g} | {sci(u) if u is not None else '·'} | {sci(r) if r is not None else '·'} | {ratio} |")
    t = "\n".join(lines)
    print(t)
    if out_path:
        with open(out_path, "w") as f:
            f.write(t + "\n")


def mode_ladder(items):
    for r in score_rows(items):
        print(json.dumps({k: v for k, v in r.items() if k not in ("corr_all", "corr0_all")}))


def main():
    if not args:
        print(__doc__)
        sys.exit(1)
    mode = args[0]
    J = float(flag("J", "1.0"))
    if mode == "gate":
        mode_gate()
    elif mode == "exact":
        mode_exact(int(args[1]), float(args[2]), J)
    elif mode == "mf":
        N, h = int(args[1]), float(args[2])
        mf, ex = mean_field(N, h, J), exact(N, h, J)
        print(f"mean field N = {N} h = {h}: E = {mf['E']:.8f} (E0 {ex['E0']:.8f}, "
              f"rel {(mf['E']-ex['E0'])/abs(ex['E0']):.3e}), <s^x> = {mf['sx']:.6f}, "
              f"C = {mf['corr'][1]:.6f}, phi = {mf['phi']:.6f}")
    elif mode == "j1j2":
        mode_j1j2(int(args[1]), float(args[2]), float(flag("J1", "1.0")))
    elif mode == "table3":
        mode_table3(args[1], flag("out"))
    elif mode == "sweep":
        mode_sweep(args[1], flag("out"))
    elif mode == "ablation":
        mode_ablation(args[1], flag("out"))
    elif mode in ("score", "ladder"):
        items = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args[1:]]
        if mode == "score":
            mode_score(items, flag("out"), "--gate" in flags)
        else:
            mode_ladder(items)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
