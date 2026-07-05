"""Per-pair LipSDP scorecard generator (the tighter-Lipschitz-constant pass).

Produces LeanMlir/Proofs/LipschitzCertScorecardSDP.lean (capped net) and
LipschitzCertScorecardSDPUncon.lean (unconstrained net): for each ordered
class pair (i,j) needed by a certified image, a LipSDP-Neuron certificate
(Fazlyab et al. 2019, one hidden layer)

    (g x - g x')^2 <= rho * ||x - x'||^2,   g = <W2_i - W2_j, relu(W1 .)>

witnessed in Lean by an exact rational LDL^T factorization of
S = 2 diag(T) - vv^T - (1/rho) T G1 T  (PSD <=> the bound, via
LeanMlir/Proofs/LipschitzCertPairSDP.lean). The SDP over diagonal T is
solved numerically here; only the rationalized (rho, T, L, d) enters Lean.

Certification criterion per image (exact rationals, what Lean checks):
for every j != y:  Lp_{y,j} * eps <= logit_y - logit_j, with rho <= Lp^2.
No sqrt2, no global product constant. Result (eps = 1/10, first 100 test
images): capped 34 -> 69/100, unconstrained 1 -> 63/100 (PGD bracket:
72 / 69). Run from repo root: python3 scripts/lipschitz_cert_pair_sdp.py
"""
import numpy as np, struct
from fractions import Fraction
from math import ceil, sqrt
from scipy.linalg import sqrtm
from scipy.optimize import minimize

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data") + os.sep
OUT_C = os.path.join(ROOT, "LeanMlir/Proofs/LipschitzCertScorecardSDP.lean")
OUT_U = os.path.join(ROOT, "LeanMlir/Proofs/LipschitzCertScorecardSDPUncon.lean")
N_IMG = 100
EPS = Fraction(1, 10)
CAP, EPOCHS, LR, BS = 4.0, 36, 0.15, 64
DEN_U, DEN_C = 128, 256
H, K, DIM = 8, 10, 49

# images already defined in LipschitzCertScorecard.lean (img<i>)
EXISTING_IMGS = {0, 3, 5, 10, 13, 14, 17, 25, 28, 29, 34, 36, 37, 39, 41, 52,
                 56, 60, 68, 69, 70, 71, 74, 79, 81, 82, 85, 86, 89, 90, 93,
                 94, 95, 98}
# hpre defs already there: hpreC<i> for the 34 capped, hpreU82 for uncon #82
EXISTING_HPRE_C = set(EXISTING_IMGS)  # hpreC<i> exists for all 34 capped ones
EXISTING_HPRE_C.discard(82)           # 82 has BOTH hpreC82 and hpreU82
EXISTING_HPRE_C.add(82)
EXISTING_HPRE_U = {82}

def load_images(fn):
    with open(fn, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

def load_labels(fn):
    with open(fn, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

Xtr_raw = load_images(D + "train-images-idx3-ubyte"); ytr = load_labels(D + "train-labels-idx1-ubyte")
Xte_raw = load_images(D + "t10k-images-idx3-ubyte"); yte = load_labels(D + "t10k-labels-idx1-ubyte")

def pool_sums(X):
    return X.reshape(-1, 7, 4, 7, 4).astype(np.int64).sum(axis=(2, 4)).reshape(-1, 49)

Str = pool_sums(Xtr_raw); Ste = pool_sums(Xte_raw)
Xtr = Str / 4080.0

def train(cap=None, epochs=12, lr=LR, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0 / DIM), (H, DIM))
    W2 = rng.normal(0, np.sqrt(2.0 / H), (K, H))
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BS):
            xb = Xtr[idx[b:b+BS]]; yb = ytr[idx[b:b+BS]]
            h = xb @ W1.T; hr = np.maximum(h, 0); z = hr @ W2.T
            z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            g = p.copy(); g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            W1 -= lr * ((g @ W2) * (h > 0)).T @ xb; W2 -= lr * g.T @ hr
            if cap is not None:
                s1 = np.linalg.svd(W1, compute_uv=False)[0]
                s2 = np.linalg.svd(W2, compute_uv=False)[0]
                if s1 > cap: W1 *= cap / s1
                if s2 > cap: W2 *= cap / s2
    return W1, W2

W1u, W2u = train(cap=None, epochs=12)
W1q = np.round(W1u * DEN_U).astype(np.int64); W2q = np.round(W2u * DEN_U).astype(np.int64)
assert list(W1q[0, :5]) == [3, -18, -108, -264, -161], "uncon drift"
assert list(W2q[0, :5]) == [-93, -292, 295, -35, 189], "uncon drift"
W1c, W2c = train(cap=CAP, epochs=EPOCHS)
W1cq = np.round(W1c * DEN_C).astype(np.int64); W2cq = np.round(W2c * DEN_C).astype(np.int64)

# ── exact per-image facts ──
def image_facts(W1z, W2z, den):
    out = []
    for i in range(N_IMG):
        s = Ste[i].astype(object)
        pre = [int(v) for v in W1z.astype(object) @ s]           # /(den*4080)
        hid = np.array([max(v, 0) for v in pre], dtype=object)
        logit = W2z.astype(object) @ hid                          # /(den^2*4080)
        y = int(yte[i])
        ms = {j: Fraction(int(logit[y] - logit[j]), den * den * 4080)
              for j in range(K) if j != y}
        out.append((y, ms, pre))
    return out

# ── LipSDP per pair: numeric solve + exact rational certificate ──
def rho_of_theta(theta, Q, G1f):
    T = np.diag(np.exp(theta))
    A = 2 * T - Q
    ew = np.linalg.eigvalsh(A)
    if ew.min() <= 1e-9:
        return 1e12 * (1 - ew.min())
    Ah = np.real(sqrtm(np.linalg.inv(A)))
    Mm = Ah @ T @ G1f @ T @ Ah
    return np.linalg.eigvalsh(Mm).max()

def solve_pair(vf, G1f):
    Q = np.outer(vf, vf)
    best = None
    for s in (max(np.abs(vf).max()**2, 1e-2), 2 * float(vf @ vf) + 1e-2, 10.0):
        r = minimize(rho_of_theta, np.log(np.full(H, s)), args=(Q, G1f),
                     method="Nelder-Mead",
                     options={"maxiter": 6000, "xatol": 1e-11, "fatol": 1e-13})
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    return best

def ldl_exact(S):
    Lm = [[Fraction(0)] * H for _ in range(H)]
    dm = [Fraction(0)] * H
    for i in range(H):
        s = S[i][i] - sum(Lm[i][k] ** 2 * dm[k] for k in range(i))
        if s < 0:
            return None, None
        dm[i] = s
        Lm[i][i] = Fraction(1)
        for jj in range(i + 1, H):
            num = S[jj][i] - sum(Lm[jj][k] * Lm[i][k] * dm[k] for k in range(i))
            if s == 0:
                if num != 0:
                    return None, None
                Lm[jj][i] = Fraction(0)
            else:
                Lm[jj][i] = num / s
    return Lm, dm

def rational_cert(a, b, W2z, G1q, den):
    """Certificate for ordered gap (a - b); S depends only on {a,b}."""
    W1f = None  # unused
    W2f = W2z / den
    vf = W2f[a] - W2f[b]
    G1f = np.array([[float(Fraction(int(G1q[x][y]), den ** 2)) for y in range(H)]
                    for x in range(H)])
    rho_f, theta = solve_pair(vf, G1f)
    TQ = [max(Fraction(round(np.exp(t) * 1024), 1024), Fraction(0)) for t in theta]
    vQ = [Fraction(int(W2z[a][k] - W2z[b][k]), den) for k in range(H)]
    GQ = [[Fraction(int(G1q[x][y]), den ** 2) for y in range(H)] for x in range(H)]
    for bump in (1.01, 1.02, 1.05, 1.1, 1.25):
        rhoQ = Fraction(ceil(rho_f * bump * 10 ** 6), 10 ** 6)
        S = [[2 * (TQ[x] if x == y else Fraction(0)) - vQ[x] * vQ[y]
              - Fraction(1) / rhoQ * (TQ[x] * GQ[x][y] * TQ[y])
              for y in range(H)] for x in range(H)]
        Lm, dm = ldl_exact(S)
        if Lm is not None:
            break
    assert Lm is not None, f"LDL failed for pair ({a},{b})"
    LpQ = Fraction(ceil(sqrt(float(rhoQ)) * 10 ** 4), 10 ** 4)
    while LpQ ** 2 < rhoQ:
        LpQ += Fraction(1, 10 ** 4)
    # exact re-verification
    for x in range(H):
        for y in range(H):
            assert S[x][y] == sum(Lm[x][i] * dm[i] * Lm[y][i] for i in range(H))
    assert all(d >= 0 for d in dm) and all(t >= 0 for t in TQ)
    return dict(rho=rhoQ, T=TQ, v=vQ, L=Lm, d=dm, Lp=LpQ)

# ── Lean emission helpers ──
def frac(q):
    q = Fraction(q)
    return f"(({q.numerator} : ℝ)/{q.denominator})" if q.denominator != 1 else f"({q.numerator} : ℝ)"

def vec(vals):
    return "![" + ", ".join(frac(v) for v in vals) + "]"

def matr(M):
    return "![" + ",\n    ".join(vec(r) for r in M) + "]"

def row_int(vals, den):
    return "![" + ", ".join(f"(({int(v)} : ℝ)/{den})" for v in vals) + "]"

def emit_net(tag, W1z, W2z, G1q, den, facts, w1name, w2name, g1name, g1eq,
             mlpname, existing_hpre, hpre_prefix, out_path, imgs_already,
             other_import=None):
    """tag: 'C' (capped) or 'U' (uncon)."""
    sq_pfx = f"pairSq{tag}"
    # which images certify under per-pair LipSDP, exact
    pair_cache = {}
    def cert_of(a, b):
        key = (min(a, b), max(a, b))
        if key not in pair_cache:
            pair_cache[key] = rational_cert(key[0], key[1], W2z, G1q, den)
        return pair_cache[key]
    certified = []
    for i, (y, ms, _pre) in enumerate(facts):
        if min(ms.values()) <= 0:
            continue
        if all(cert_of(y, j)["Lp"] * EPS <= ms[j] for j in ms):
            certified.append(i)
    classes = sorted({facts[i][0] for i in certified})
    ordered_pairs = sorted({(facts[i][0], j) for i in certified
                            for j in range(K) if j != facts[i][0]})
    unordered = sorted({(min(a, b), max(a, b)) for a, b in ordered_pairs})
    print(f"[{tag}] certified {len(certified)}/{N_IMG}: {certified}")
    print(f"[{tag}] classes {classes}, {len(unordered)} unordered pairs, "
          f"{len(ordered_pairs)} ordered")

    L = []
    A = L.append
    A("import LeanMlir.Proofs.LipschitzCertPairSDP")
    A("import LeanMlir.Proofs.LipschitzCertScorecard")
    if other_import:
        A(f"import LeanMlir.Proofs.{other_import}")
    A("")
    netdesc = ("spectrally-capped /256 net (`mlpS`)" if tag == "C"
               else "unconstrained /128 net (`mlpT`)")
    base = 34 if tag == "C" else 1
    A(f"/-! # Per-pair LipSDP scorecard — {netdesc}")
    A("")
    A(f"The tighter-Lipschitz-constant pass over the SAME first-{N_IMG} MNIST test")
    A(f"subset and SAME ε = {EPS} as `LipschitzCertScorecard.lean`: replacing the")
    A("global `√2·∏‖Wᵢ‖` criterion by per-pair LipSDP certificates lifts the")
    A(f"count from **{base}/{N_IMG} to {len(certified)}/{N_IMG}** — no retraining, no new data, just a")
    A("less lossy constant on each pairwise logit gap. Each class pair carries")
    A("an exact rational LDLᵀ PSD witness (`s*_def`/`s*_ldl`), turned into the")
    A("squared gap bound by `pair_sq_bound`; each image then needs only its")
    A(f"9 margins to clear `Lp·ε` (`certifiedS{tag}<i>`). Empirical PGD bracket:")
    A(f"{'72' if tag == 'C' else '69'}/{N_IMG} — the cert ≤ TRUE ≤ PGD sandwich is nearly closed.")
    A("")
    A("Generated by `scripts/lipschitz_cert_pair_sdp.py`; weights/images/")
    A("certificates are DATA (SDP solved off-line, verified exactly here). -/")
    A("")
    A("namespace Proofs")
    A("namespace LipschitzCertDemo")
    A("")
    A("open scoped BigOperators")
    A("")
    A("-- ════════════════════════════════════════════════════════════")
    A(f"-- § Pair certificates ({len(unordered)} unordered class pairs)")
    A("-- ════════════════════════════════════════════════════════════")
    A("")
    for (a, b) in unordered:
        c = cert_of(a, b)
        nm = f"{a}{b}{tag}"
        A(f"/-- Pair ({a},{b}): ρ = {float(c['rho']):.3f}, Lp = {float(c['Lp']):.4f} "
          f"(√2·L-product criterion would need "
          f"{'27.95' if tag == 'C' else '90.22'}). -/")
        A(f"noncomputable def vP{nm} : Fin 8 → ℝ := {vec(c['v'])}")
        A("")
        A(f"noncomputable def tP{nm} : Fin 8 → ℝ := {vec(c['T'])}")
        A("")
        A(f"noncomputable def dP{nm} : Fin 8 → ℝ := {vec(c['d'])}")
        A("")
        A(f"noncomputable def lP{nm} : Fin 8 → Fin 8 → ℝ :=")
        A("  " + matr(c["L"]))
        A("")
        A(f"noncomputable def sP{nm} : Fin 8 → Fin 8 → ℝ :=")
        A("  " + matr([[2 * (c["T"][x] if x == y else Fraction(0))
                        - c["v"][x] * c["v"][y]
                        - Fraction(1) / c["rho"] * (c["T"][x] * Fraction(int(G1q[x][y]), den ** 2) * c["T"][y])
                        for y in range(H)] for x in range(H)]))
        A("")
        A(f"theorem vP{nm}_eq : ∀ t, vP{nm} t = {w2name} {a} t - {w2name} {b} t := by")
        A("  intro t")
        A("  fin_cases t <;>")
        A(f"    · simp [vP{nm}, {w2name}]")
        A("      try norm_num")
        A("")
        A(f"theorem tP{nm}_nonneg : ∀ k, 0 ≤ tP{nm} k := by")
        A("  intro k")
        A("  fin_cases k <;>")
        A(f"    · simp [tP{nm}]")
        A("      try norm_num")
        A("")
        A(f"theorem dP{nm}_nonneg : ∀ i, 0 ≤ dP{nm} i := by")
        A("  intro i")
        A("  fin_cases i <;>")
        A(f"    · simp [dP{nm}]")
        A("      try norm_num")
        A("")
        A("set_option maxHeartbeats 3200000 in")
        A(f"theorem sP{nm}_def : ∀ a b, sP{nm} a b")
        A(f"    = 2 * (if a = b then tP{nm} a else 0) - vP{nm} a * vP{nm} b")
        A(f"      - (1/{frac(c['rho'])}) * (tP{nm} a * ({g1name} a b * tP{nm} b)) := by")
        A("  intro a b")
        A("  fin_cases a <;> fin_cases b <;>")
        A(f"    · simp [sP{nm}, tP{nm}, vP{nm}, {g1name}]")
        A("      try norm_num")
        A("")
        A("set_option maxHeartbeats 3200000 in")
        A(f"theorem sP{nm}_ldl : ∀ a b, sP{nm} a b = ∑ i, lP{nm} a i * (dP{nm} i * lP{nm} b i) := by")
        A("  intro a b")
        A("  fin_cases a <;> fin_cases b <;>")
        A(f"    · simp [sP{nm}, lP{nm}, dP{nm}, Fin.sum_univ_succ]")
        A("      try norm_num")
        A("")
        A(f"theorem {sq_pfx}_{a}_{b} : ∀ u u' : EuclideanSpace ℝ (Fin 49),")
        A(f"    (({mlpname} u {a} - {mlpname} u {b}) - ({mlpname} u' {a} - {mlpname} u' {b})) ^ 2")
        A(f"      ≤ {frac(c['rho'])} * ‖u - u'‖ ^ 2 := by")
        A("  intro u u'")
        A(f"  have e : ∀ w : EuclideanSpace ℝ (Fin 49), {mlpname} w {a} - {mlpname} w {b}")
        A(f"      = ∑ t, vP{nm} t * max (denseE {w1name} w t) 0 := by")
        A("    intro w")
        A(f"    have h : {mlpname} w {a} - {mlpname} w {b}")
        A(f"        = ∑ t, ({w2name} {a} t - {w2name} {b} t) * max (denseE {w1name} w t) 0 :=")
        A(f"      mlp_gap_eq {w1name} {w2name} {a} {b} w")
        A("    rw [h]")
        A(f"    exact Finset.sum_congr rfl fun t _ => by rw [vP{nm}_eq t]")
        A("  rw [e u, e u']")
        A(f"  exact pair_sq_bound {w1name} {g1name} {g1eq} vP{nm} tP{nm} tP{nm}_nonneg (by norm_num)")
        A(f"    (lipsdp_slack_of_cert {g1name} sP{nm} lP{nm} vP{nm} tP{nm} dP{nm} dP{nm}_nonneg")
        A(f"      sP{nm}_def sP{nm}_ldl) u u'")
        A("")
    # reverse-order wrappers
    for (a, b) in ordered_pairs:
        if a < b:
            continue
        lo, hi = b, a
        c = cert_of(lo, hi)
        A(f"theorem {sq_pfx}_{a}_{b} : ∀ u u' : EuclideanSpace ℝ (Fin 49),")
        A(f"    (({mlpname} u {a} - {mlpname} u {b}) - ({mlpname} u' {a} - {mlpname} u' {b})) ^ 2")
        A(f"      ≤ {frac(c['rho'])} * ‖u - u'‖ ^ 2 := by")
        A("  intro u u'")
        A(f"  have e : ({mlpname} u {a} - {mlpname} u {b}) - ({mlpname} u' {a} - {mlpname} u' {b})")
        A(f"      = -(({mlpname} u {lo} - {mlpname} u {hi}) - ({mlpname} u' {lo} - {mlpname} u' {hi})) := by")
        A("    ring")
        A("  rw [e, neg_sq]")
        A(f"  exact {sq_pfx}_{lo}_{hi} u u'")
        A("")
    A("-- ════════════════════════════════════════════════════════════")
    A(f"-- § Per-image certificates ({len(certified)}/{N_IMG} at ε = {EPS})")
    A("-- ════════════════════════════════════════════════════════════")
    A("")
    den_h = den * 4080
    for i in certified:
        y, ms, pre = facts[i]
        # image def (skip if an imported file already has it)
        if i not in imgs_already:
            A(f"/-- MNIST test image #{i} (digit {y}), exact pixel sums /4080. -/")
            A(f"noncomputable def img{i} : EuclideanSpace ℝ (Fin 49) :=")
            A("  WithLp.toLp 2 " + row_int(Ste[i], 4080))
            A("")
        # hpre def+eval (reuse the base scorecard's when present)
        if i in existing_hpre:
            hpre = f"hpre{'C' if tag == 'C' else 'U'}{i}"
        else:
            hpre = f"hpre{hpre_prefix}{i}"
            A(f"noncomputable def {hpre} : Fin 8 → ℝ :=")
            A("  " + row_int(pre, den_h))
            A("")
            A(f"theorem {hpre}_eval : ∀ k : Fin 8, denseE {w1name} img{i} k = {hpre} k := by")
            A("  intro k")
            A("  fin_cases k <;>")
            A(f"    · simp [denseE_apply, {w1name}, img{i}, {hpre}, Fin.sum_univ_succ]")
            A("      norm_num")
            A("")
        A("set_option maxHeartbeats 3200000 in")
        A(f"/-- Test #{i} (digit {y}): LipSDP-per-pair certified at ε = {EPS} — each")
        A(f"    of the 9 margins clears its own `Lp·ε`. -/")
        A(f"theorem certifiedS{tag}{i} (δ : EuclideanSpace ℝ (Fin 49)) (hδ : ‖δ‖ < {frac(EPS)}) :")
        A(f"    ∀ j, j ≠ {y} → {mlpname} (img{i} + δ) j < {mlpname} (img{i} + δ) {y} := by")
        A(f"  have hout : ∀ jj : Fin 10, {mlpname} img{i} jj = ∑ k, {w2name} jj k * max ({hpre} k) 0 := by")
        A("    intro jj")
        A(f"    show denseE {w2name} (reluE (denseE {w1name} img{i})) jj = _")
        A("    rw [denseE_apply]")
        A("    refine Finset.sum_congr rfl fun k _ => ?_")
        A(f"    rw [reluE_apply, {hpre}_eval k]")
        A("  intro j hj")
        A("  fin_cases j")
        for j in range(K):
            if j == y:
                A("  · exact absurd rfl hj")
            else:
                c = cert_of(y, j)
                A(f"  · refine certified_at_eps_pair (Lp := {frac(c['Lp'])}) {sq_pfx}_{y}_{j}")
                A("      (by norm_num) (by norm_num) ?_ δ hδ")
                A("    rw [hout, hout]")
                A(f"    simp [{w2name}, {hpre}, Fin.sum_univ_succ, max_def]")
                A("    try norm_num")
        A("")
    A("-- ════════════════════════════════════════════════════════════")
    A("-- § Aggregate — mechanized (peer of `scorecard`)")
    A("-- ════════════════════════════════════════════════════════════")
    A("")
    lname = "sdpCappedCerts" if tag == "C" else "sdpUnconCerts"
    A(f"/-- LipSDP-certified witnesses on the {'capped' if tag == 'C' else 'unconstrained'} net: "
      f"`(subset index, image, class)`. -/")
    A(f"noncomputable def {lname} : List (ℕ × EuclideanSpace ℝ (Fin 49) × Fin 10) :=")
    trips = ", ".join(f"({i}, img{i}, {facts[i][0]})" for i in certified)
    A(f"  [{trips}]")
    A("")
    A(f"theorem {lname}_certified :")
    A(f"    ∀ p ∈ {lname}, CertifiedAt {mlpname} ((1 : ℝ)/10) p.2.1 p.2.2 :=")
    A("  List.forall_iff_forall_mem.mp")
    A("    ⟨" + ", ".join(f"certifiedS{tag}{i}" for i in certified) + "⟩")
    A("")
    A(f"/-- **The LipSDP scorecard**: {len(certified)}/{N_IMG} — vs {base}/{N_IMG} under the")
    A("    global √2·∏-norm criterion, same net, same ε, same images; the")
    A("    constant was the bottleneck, not the network. -/")
    A(f"theorem scorecard_sdp{'' if tag == 'C' else '_uncon'} :")
    A(f"    {lname}.length = {len(certified)} ∧")
    A(f"      ∀ p ∈ {lname}, CertifiedAt {mlpname} ((1 : ℝ)/10) p.2.1 p.2.2 :=")
    A(f"  ⟨rfl, {lname}_certified⟩")
    A("")
    A("end LipschitzCertDemo")
    A("end Proofs")
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[{tag}] wrote {out_path}: {len(L)} lines")
    return certified

G1cq = W1cq.astype(object) @ W1cq.astype(object).T
G1uq = W1q.astype(object) @ W1q.astype(object).T
facts_c = image_facts(W1cq, W2cq, DEN_C)
facts_u = image_facts(W1q, W2q, DEN_U)

cert_c = emit_net("C", W1cq, W2cq, G1cq, DEN_C, facts_c,
                  "W1s", "W2s", "G1s", "G1s_eq", "mlpS",
                  EXISTING_HPRE_C, "SC", OUT_C, EXISTING_IMGS)
cert_u = emit_net("U", W1q, W2q, G1uq, DEN_U, facts_u,
                  "W1t", "W2t", "G1t", "G1t_eq", "mlpT",
                  EXISTING_HPRE_U, "SU", OUT_U,
                  EXISTING_IMGS | set(cert_c),
                  other_import="LipschitzCertScorecardSDP")
print(f"\nDONE: capped {len(cert_c)}/100, uncon {len(cert_u)}/100")
