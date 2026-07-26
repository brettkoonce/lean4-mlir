"""CROWN-IBP L-infinity scorecard for the full-input nets (planning/crown_ibp.md phase 2).

Produces LeanMlir/Proofs/Certificates/LipschitzCertScorecardCrown.lean (capped
sigma<=2 net) and LipschitzCertScorecardCrownUncon.lean: per-image pixel-L-infinity
certificates on the SAME first-100 MNIST test subset, same nets and same eps grid
{1,2,4,8}/255 as the IBP tier -- so the result is a NEW COLUMN in the existing
table, directly comparable, not a new experiment.

What changes vs the IBP tier: IBP concretizes to an interval after every layer,
so the box grows multiplicatively with depth. CROWN relaxes each unstable ReLU by
a linear envelope, back-substitutes the margin row v = W2[y] - W2[j] through W1
to ONE composite row A, and takes eps*||A||_1 ONCE. Engine: CrownBound.lean.

Reuse (nothing re-derived):
  <W1_t, x0>   committed hpre* dotZ facts (ImgsA/ImgsB)
  ||W1_t||_1   committed absr*/absrow* facts (the IBP scorecard)
  the [l,u] box literally denseLo/denseHi W1 -- gotcha 4 is enforced by the
               capstone's STATEMENT, so a float recomputation cannot sneak in
  ||A||_1      ONE new absSumZ(combZ ...) kernel fact per (image, class): the
               kernel forms A from the committed w1z rows, so the 784 entries of
               A are never emitted (gotcha 2).

Rational sizes: the upper-envelope slope is rounded UP to a /2^8 grid, which
crown_ibp_probe.py measures as costing zero images. Coefficients therefore live
at /2^16 and A at /2^24 -- nowhere near the LipSDP tier's ~230-digit regime.
"""
import os
import re
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "LeanMlir" / "Proofs" / "Certificates"
NETS = OUTDIR / "LipschitzCertScorecardFullNets.lean"

H, K, DIM, DEN, PIX, N_IMG = 16, 10, 784, 256, 255, 100
EPS_GRID = [(1, "e1"), (2, "e2"), (4, "e4"), (8, "e8")]
KBITS = 8                      # slope grid: s rounded up to a multiple of 2^-KBITS
DS = 1 << KBITS                # slope denominator
DC = DEN * DS                  # coefficient denominator   (65536)
DW = DEN                       # weight denominator        (256)
DP = DEN * PIX                 # pre-activation denominator(65280)
DCC = DC * DP                  # constant denominator      (4278190080)
DA = DC * DW                   # crown-row denominator     (16777216)
N_EMIT = int(os.environ.get("SCORECARD_N_EMIT", 8))


# ── data: the COMMITTED weights, not a retrain ───────────────────────────────
def _parse_int_list(text):
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        out.append(-(int(tok.split()[1]) + 1) if tok.startswith("Int.negSucc") else int(tok))
    return out


def load_nets():
    src = NETS.read_text()
    nets = {}
    for tag in ("SF", "TF"):
        W1 = np.array([_parse_int_list(
            src.split(f"def w1z{tag}{k} : List ℤ := [")[1].split("]")[0]) for k in range(H)],
            dtype=object)
        blk = src.split(f"noncomputable def W2{tag} : Fin {K} → Fin {H} → ℝ")[1]
        rows = []
        for ln in blk.splitlines():
            nums = re.findall(rf"\((-?\d+) : ℝ\)/{DEN}", ln)
            if len(nums) == H:
                rows.append([int(v) for v in nums])
            if len(rows) == K:
                break
        nets[tag] = (W1, np.array(rows, dtype=object))
    return nets


def available_images(tag):
    """Images whose per-image `hpre` dot data this tier can REUSE.

    CROWN certifies images IBP never did, so the committed corpus does not carry
    pre-activation data for all of them (`hpreTF1` exists only in the DISABLED
    LipSDP file). Rather than re-emit that data, the images that carry THEOREMS
    are drawn from what is already committed and in the build. This is a choice
    about which witnesses are exhibited; the MEASURED counts run over all 100
    images regardless and are unaffected."""
    avail = set()
    for f in ("LipschitzCertScorecardFullImgsA", "LipschitzCertScorecardFullImgsB"):
        src = (OUTDIR / f"{f}.lean").read_text()
        avail |= {int(m) for m in re.findall(rf"theorem hpre{tag}(\d+)_eval", src)}
    return avail


def load_mnist():
    d = ROOT / "data"
    with open(d / "t10k-images-idx3-ubyte", "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)
    with open(d / "t10k-labels-idx1-ubyte", "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return X[:N_IMG].astype(object), y[:N_IMG].astype(int)


# ── the certificate, in exact integers ───────────────────────────────────────
def crown_image(W1q, W2q, A1, pre, y, num):
    """Exact CROWN-IBP bound for one image at eps = num/255.

    Returns (certifies, per-neuron data, per-class data). All numerators are
    integers; the denominators are the module constants."""
    lo = [pre[t] - num * A1[t] for t in range(H)]      # /DP
    hi = [pre[t] + num * A1[t] for t in range(H)]      # /DP
    alpha, slope, status = [], [], []
    for t in range(H):
        if hi[t] <= 0:
            status.append("dead"); alpha.append(0); slope.append(0)
        elif lo[t] >= 0:
            status.append("active"); alpha.append(0); slope.append(0)
        else:
            status.append("unstable")
            alpha.append(1 if hi[t] > -lo[t] else 0)
            # s = ceil( u/(u-l) * DS ) / DS  -- rounded UP, so u <= s*(u-l) exactly
            width = hi[t] - lo[t]
            slope.append(-((-hi[t] * DS) // width))    # ceil division, width > 0
    for t in range(H):
        if status[t] == "unstable":
            assert hi[t] * DS <= slope[t] * (hi[t] - lo[t]), (t, "chord")

    acz, ccz, nrm, lbs = {}, {}, {}, {}
    ok = True
    for j in range(K):
        v = [int(W2q[y][t] - W2q[j][t]) for t in range(H)]     # /DEN
        a, c = [], []
        for t in range(H):
            if status[t] == "dead":
                a.append(0); c.append(0)
            elif status[t] == "active":
                a.append(v[t] * DS); c.append(0)               # v/DEN  ->  /DC
            elif v[t] >= 0:
                a.append(v[t] * alpha[t] * DS); c.append(0)    # v*alpha/DEN -> /DC
            else:
                a.append(v[t] * slope[t])                      # v*s/(DEN*DS) -> /DC
                c.append(-(v[t] * slope[t] * lo[t]))           # -(v*s*l) -> /DCC
        acz[j], ccz[j] = a, c
        if j == y:
            nrm[j] = 0
            continue
        Arow = [sum(a[t] * int(W1q[t][i]) for t in range(H)) for i in range(DIM)]
        nrm[j] = sum(abs(z) for z in Arow)                     # /DA
        lb = (sum(Fraction(a[t], DC) * Fraction(pre[t], DP) for t in range(H))
              - Fraction(num, PIX) * Fraction(nrm[j], DA)
              + sum(Fraction(cc, DCC) for cc in c))
        lbs[j] = lb
        if lb <= 0:
            ok = False
    return ok, (lo, hi, alpha, slope), (acz, ccz, nrm, lbs)


def frac_lit(numer, denom):
    return f"(({numer} : ℝ)/{denom})"


def zlist(xs):
    return "[" + ", ".join(str(v) if v >= 0 else f"Int.negSucc {-v - 1}" for v in xs) + "]"


def emit(tag, W1q, W2q, Xraw, yte, out_path, counts_ibp, pgd, netdesc, ibp_import):
    A1 = [int(sum(abs(int(w)) for w in W1q[t])) for t in range(H)]
    pres = {i: [int(sum(int(W1q[t][q]) * int(Xraw[i][q]) for q in range(DIM)))
                for t in range(H)] for i in range(N_IMG)}

    # exact MEASUREMENT over all N_IMG, every radius
    certs = {en: [] for _, en in EPS_GRID}
    data = {}
    for num, en in EPS_GRID:
        for i in range(N_IMG):
            ok, neu, cls = crown_image(W1q, W2q, A1, pres[i], int(yte[i]), num)
            if ok:
                certs[en].append(i)
                data[(en, i)] = (neu, cls)
    avail = available_images(tag)
    emit_set = {en: [i for i in certs[en] if i in avail][:N_EMIT] for _, en in EPS_GRID}
    # each emitted image is PROVED once, at the largest radius it is emitted at,
    # and carried down the grid by CertifiedAtLinf.mono
    best = {}
    for num, en in EPS_GRID:
        for i in emit_set[en]:
            if i not in best or num > best[i][0]:
                best[i] = (num, en)

    L = []
    A = L.append
    A("import LeanMlir.Proofs.Foundation.CrownBound")
    A(f"import LeanMlir.Proofs.Certificates.{ibp_import}")
    A("")
    A(f"/-! # CROWN-IBP L∞ scorecard, full 784-dim input — {netdesc}")
    A("")
    A("Pixel-L∞ certificates by CROWN linear relaxation over the SAME first-100")
    A("MNIST test subset, nets and ε grid as the IBP tier — a new COLUMN in that")
    A("table, not a new experiment:")
    A("")
    A("| ε | 1/255 | 2/255 | 4/255 | 8/255 |")
    A("|---|---|---|---|---|")
    A("| IBP (box) | " + " | ".join(str(counts_ibp[en]) for _, en in EPS_GRID) + " |")
    A("| **CROWN** | " + " | ".join(f"**{len(certs[en])}**" for _, en in EPS_GRID) + " |")
    A("| PGD-L∞ (upper bracket) | " + " | ".join(str(pgd[en]) for _, en in EPS_GRID) + " |")
    A("")
    A("IBP concretizes to an interval after every layer, so the box grows")
    A("multiplicatively with depth. CROWN never concretizes in the middle: each")
    A("unstable ReLU gets a linear envelope, the margin row `v = W2 y · − W2 j ·`")
    A("is back-substituted through `W1` to ONE composite row `A`, and `ε‖A‖₁` is")
    A("taken ONCE — so the cancellation between rows of `W1` survives in `A`")
    A("instead of dying to IBP's per-row `‖·‖₁`.")
    A("")
    A("**Theorem vs. measurement.** Soundness is in the ENGINE")
    A("(`Foundation/CrownBound.lean`), proved once. The counts above are")
    A(f"exact-rational MEASUREMENTS over the first {N_IMG} images; the first")
    A(f"{N_EMIT} certifying images at each radius carry a `CertifiedAtLinf`")
    A("THEOREM, and the aggregate below states only those. Each such image is")
    A("proved at the LARGEST radius it is emitted at and carried down the grid by")
    A("`CertifiedAtLinf.mono`. Counts are lower bounds: a loose relaxation cannot")
    A("prove an image UNcertifiable.")
    A("")
    A("The images carrying theorems are drawn from those the corpus already")
    A("commits `hpre` dot data for. CROWN certifies images IBP never did, and the")
    A("corpus only ever emitted per-image data for images some earlier tier")
    A("needed — so exhibiting one of the others would mean emitting NEW 784-term")
    A("dot data rather than reusing committed data. Which witnesses are exhibited")
    A("is a presentation choice; the measured counts run over all")
    A(f"{N_IMG} images either way.")
    A("")
    A("Reuse: `⟨W1ₜ, x₀⟩` is the committed `hpre*` dotZ data and `‖W1ₜ‖₁` the")
    A("committed `absr*` data. The `[l,u]` relaxed against are literally")
    A("`denseLo/denseHi W1` — forced by the capstone's statement, so no float")
    A("recomputation can enter. The one new kernel fact per `(image, class)` is")
    A("`absSumZ (combZ …)`: the kernel FORMS `A` from the committed weight rows,")
    A(f"so `A`'s {DIM} entries are never emitted.")
    A("")
    A(f"The upper-envelope slope is rounded up to a `/2^{KBITS}` grid (sound: `relu` is")
    A("convex, so any `s` at or above the chord dominates it), which is what keeps")
    A(f"coefficients at `/{DC}` and `A` at `/{DA}`. Generated by")
    A("`scripts/crown_ibp_scorecard.py`; weights/images are DATA. -/")
    A("")
    A("set_option maxRecDepth 100000")
    A("set_option maxHeartbeats 3200000")
    A("")
    A("namespace Proofs")
    A("namespace LipschitzCertDemo")
    A("")
    A("open scoped BigOperators")
    A("")
    A("-- ════════ the weight rows, as the list the kernel combines ════════")
    A("")
    A(f"def rows{tag} : List (List ℤ) := [" + ", ".join(f"w1z{tag}{t}" for t in range(H)) + "]")
    A("")
    A(f"theorem rows{tag}_len : (rows{tag}).length = {H} := by decide +kernel")
    A("")
    A(f"theorem rows{tag}_rn : ∀ r ∈ rows{tag}, r.length = {DIM} := by decide +kernel")
    A("")
    A(f"/-- `W1{tag}` read through the row list — the bridge `crownRow_comb` needs. -/")
    A(f"theorem hW{tag} : ∀ (t : Fin {H}) (i : Fin {DIM}),")
    A(f"    W1{tag} t i = ((((rows{tag}).getD (t : ℕ) []).getD (i : ℕ) 0 : ℤ) : ℝ)/{DW} := by")
    A("  intro t i")
    A("  fin_cases t <;> rfl")
    A("")

    proved = []
    for i in sorted(best):
        num, en = best[i]
        (lo, hi, alpha, slope), (acz, ccz, nrm, lbs) = data[(en, i)]
        y = int(yte[i])
        st = f"{tag}{en}_{i}"
        eps = frac_lit(num, PIX)
        A(f"-- ════════ image {i} (digit {y}), ε = {num}/255 ════════")
        A("")
        A(f"noncomputable def lo{st} : Fin {H} → ℝ :=")
        A("  ![" + ", ".join(frac_lit(lo[t], DP) for t in range(H)) + "]")
        A(f"noncomputable def hi{st} : Fin {H} → ℝ :=")
        A("  ![" + ", ".join(frac_lit(hi[t], DP) for t in range(H)) + "]")
        A(f"noncomputable def al{st} : Fin {H} → ℝ :=")
        A("  ![" + ", ".join(f"({alpha[t]} : ℝ)" for t in range(H)) + "]")
        A(f"noncomputable def sl{st} : Fin {H} → ℝ :=")
        A("  ![" + ", ".join(frac_lit(slope[t], DS) for t in range(H)) + "]")
        A("")
        box = (f"(fun q => imgF{i} q - {eps}) (fun q => imgF{i} q + {eps})")
        A("/-- The committed `hpre` dot, in the `∑`-form `denseLo_uniform` leaves")
        A("    behind. (The IBP tier's `_sum` peers only exist for the images IT")
        A("    emitted, which is a different set — CROWN certifies more.) -/")
        A(f"theorem hpsum{tag}{i} : ∀ t, (∑ q, W1{tag} t q * imgF{i} q) = hpre{tag}{i} t := by")
        A("  intro t")
        A(f"  rw [← denseE_apply]; exact hpre{tag}{i}_eval t")
        A("")
        A(f"theorem hlo{st} : ∀ t, denseLo W1{tag} {box} t = lo{st} t := by")
        A("  intro t")
        A(f"  rw [denseLo_uniform, hpsum{tag}{i}, absrow{tag}]")
        A(f"  fin_cases t <;> norm_num [lo{st}, hpre{tag}{i}, absr{tag}]")
        A("")
        A(f"theorem hhi{st} : ∀ t, denseHi W1{tag} {box} t = hi{st} t := by")
        A("  intro t")
        A(f"  rw [denseHi_uniform, hpsum{tag}{i}, absrow{tag}]")
        A(f"  fin_cases t <;> norm_num [hi{st}, hpre{tag}{i}, absr{tag}]")
        A("")
        A("/-- The relaxation, proved once per NEURON — nothing here depends on")
        A("    which wrong class is being separated. -/")
        A(f"theorem hrx{st} : ∀ (t : Fin {H}) (v : ℝ),")
        A(f"    ReluLB v (lo{st} t) (hi{st} t)")
        A(f"      (relaxA (lo{st} t) (hi{st} t) (al{st} t) (sl{st} t) v)")
        A(f"      (relaxC (lo{st} t) (hi{st} t) (al{st} t) (sl{st} t) v) := by")
        A("  intro t v")
        A("  refine reluLB_relax ?_ ?_ ?_ ?_ <;>")
        A(f"    fin_cases t <;> norm_num [al{st}, sl{st}, lo{st}, hi{st}]")
        A("")
        # Per-class data lives in NAMED defs, never behind a `![…]` index: a
        # numeral matrix index (`![…] 7`) does not reduce under `norm_num`, and
        # `rw` cannot match a numeral against `fin_cases`'s `⟨j, _⟩`. Assembling
        # the matrix form only at the end lets `exact` close that gap by defeq.
        for j in range(K):
            A(f"def acz{st}_{j} : List ℤ := {zlist(acz[j])}")
        A("")
        for j in range(K):
            A(f"def ccz{st}_{j} : List ℤ := {zlist(ccz[j])}")
        A("")
        for j in range(K):
            A(f"noncomputable def acr{st}_{j} : Fin {H} → ℝ :=")
            A(f"  fun t => ((acz{st}_{j}.getD (t : ℕ) 0 : ℤ) : ℝ)/{DC}")
        A("")
        for j in range(K):
            A(f"noncomputable def ccr{st}_{j} : Fin {H} → ℝ :=")
            A(f"  fun t => ((ccz{st}_{j}.getD (t : ℕ) 0 : ℤ) : ℝ)/{DCC}")
        A("")
        A(f"noncomputable def acr{st} : Fin {K} → Fin {H} → ℝ :=")
        A("  ![" + ", ".join(f"acr{st}_{j}" for j in range(K)) + "]")
        A(f"noncomputable def ccr{st} : Fin {K} → Fin {H} → ℝ :=")
        A("  ![" + ", ".join(f"ccr{st}_{j}" for j in range(K)) + "]")
        A("")
        for j in range(K):
            if j == y:
                continue
            A(f"theorem hrel{st}_{j} : ∀ t : Fin {H},")
            A(f"    ReluLB (W2{tag} {y} t - W2{tag} {j} t) (lo{st} t) (hi{st} t)")
            A(f"      (acr{st}_{j} t) (ccr{st}_{j} t) := by")
            A("  intro t")
            A(f"  refine (hrx{st} t (W2{tag} {y} t - W2{tag} {j} t)).congr ?_ ?_ <;>")
            A("    fin_cases t <;>")
            A(f"      simp [acr{st}_{j}, ccr{st}_{j}, acz{st}_{j}, ccz{st}_{j}, relaxA, relaxC,")
            A(f"        lo{st}, hi{st}, al{st}, sl{st}, W2{tag}] <;>")
            A("      norm_num")
        A("")
        A(f"theorem hrel{st} : ∀ j : Fin {K}, j ≠ {y} → ∀ t : Fin {H},")
        A(f"    ReluLB (W2{tag} {y} t - W2{tag} j t)")
        A(f"      (denseLo W1{tag} {box} t) (denseHi W1{tag} {box} t)")
        A(f"      (acr{st} j t) (ccr{st} j t) := by")
        A("  intro j hj t")
        A(f"  rw [hlo{st}, hhi{st}]")
        A("  fin_cases j")
        for j in range(K):
            A("  · exact absurd rfl hj" if j == y else f"  · exact hrel{st}_{j} t")
        A("")
        A("-- ‖A‖₁: ONE kernel fact per class — the kernel forms `A` from the")
        A(f"-- committed rows, so its {DIM} entries never appear here.")
        for j in range(K):
            if j == y:
                continue
            A(f"theorem nrm{st}_{j} : absSumZ (combZ {DIM} acz{st}_{j} rows{tag}) = {nrm[j]} := by")
            A("  decide +kernel")
        A("")
        for j in range(K):
            if j == y:
                continue
            A(f"theorem hl1{st}_{j} : (∑ q, |crownRow (acr{st}_{j}) W1{tag} q|) = "
              f"(({nrm[j]} : ℝ)/({DC} * {DW})) :=")
            A(f"  crownRow_l1 acz{st}_{j} rows{tag} W1{tag} {DC} {DW} hW{tag}")
            A(f"    (by decide +kernel) rows{tag}_len rows{tag}_rn (by norm_num) nrm{st}_{j}")
        A("")
        for j in range(K):
            if j == y:
                continue
            A(f"theorem hcert{st}_{j} :")
            A(f"    0 < (∑ q, crownRow (acr{st}_{j}) W1{tag} q * imgF{i} q)")
            A(f"          - {eps} * (∑ q, |crownRow (acr{st}_{j}) W1{tag} q|)")
            A(f"          + ∑ t, ccr{st}_{j} t := by")
            A(f"  rw [← crownRow_dot, hl1{st}_{j}]")
            A(f"  simp only [hpre{tag}{i}_eval]")
            A(f"  norm_num [acr{st}_{j}, ccr{st}_{j}, acz{st}_{j}, ccz{st}_{j}, hpre{tag}{i},")
            A("    Fin.sum_univ_succ]")
        A("")
        A(f"theorem hcert{st} : ∀ j : Fin {K}, j ≠ {y} →")
        A(f"    0 < (∑ q, crownRow (acr{st} j) W1{tag} q * imgF{i} q)")
        A(f"          - {eps} * (∑ q, |crownRow (acr{st} j) W1{tag} q|) + ∑ t, ccr{st} j t := by")
        A("  intro j hj")
        A("  fin_cases j")
        for j in range(K):
            A("  · exact absurd rfl hj" if j == y else f"  · exact hcert{st}_{j}")
        A("")
        A(f"/-- Test #{i} (digit {y}): CROWN-certified at pixel-L∞ ε = {num}/255. -/")
        A(f"theorem certCR{st} : CertifiedAtLinf mlp{tag} {eps} imgF{i} {y} :=")
        A(f"  crown2_certified_at_eps W1{tag} W2{tag} (acr{st}) (ccr{st}) hrel{st} hcert{st}")
        A("")
        proved.append((i, num, en))

    # per-radius corollaries by monotonicity
    parts = []
    for num, en in EPS_GRID:
        if not emit_set[en]:
            continue
        eps = frac_lit(num, PIX)
        for i in emit_set[en]:
            bnum, ben = best[i]
            if ben == en:
                continue
            A(f"/-- Test #{i} at ε = {num}/255, from the {bnum}/255 certificate. -/")
            A(f"theorem certCR{tag}{en}_{i} : CertifiedAtLinf mlp{tag} {eps} imgF{i} "
              f"{int(yte[i])} :=")
            A(f"  (certCR{tag}{ben}_{i}).mono (by norm_num)")
            A("")
        lname = f"crown{'Capped' if tag == 'SF' else 'Uncon'}Certs{en}"
        A(f"noncomputable def {lname} : List (ℕ × EuclideanSpace ℝ (Fin {DIM}) × Fin {K}) :=")
        A("  [" + ",\n   ".join(f"({i}, imgF{i}, {int(yte[i])})" for i in emit_set[en]) + "]")
        A("")
        A(f"theorem {lname}_certified :")
        A(f"    ∀ p ∈ {lname}, CertifiedAtLinf mlp{tag} {eps} p.2.1 p.2.2 :=")
        A("  List.forall_iff_forall_mem.mp")
        wits = [f"certCR{tag}{en}_{i}" for i in emit_set[en]]
        # `List.Forall p [a]` is `p a`, not a 1-tuple — ⟨…⟩ would be ill-formed
        A("    " + (wits[0] if len(wits) == 1 else "⟨" + ", ".join(wits) + "⟩"))
        A("")
        parts.append((lname, eps, len(emit_set[en])))

    A(f"/-- **The CROWN-IBP L∞ scorecard, {netdesc}** — MEASURED "
      + ", ".join(f"{len(certs[en])}/{N_IMG} @ {num}/255" for num, en in EPS_GRID)
      + " (IBP box: " + "/".join(str(counts_ibp[en]) for _, en in EPS_GRID)
      + "; PGD-L∞ bracket " + "/".join(str(pgd[en]) for _, en in EPS_GRID) + "). -/")
    A(f"theorem scorecard_crown{'' if tag == 'SF' else '_uncon'} :")
    A("    " + " ∧\n    ".join(
        f"({ln}.length = {cnt} ∧ ∀ p ∈ {ln}, CertifiedAtLinf mlp{tag} {ep} p.2.1 p.2.2)"
        for ln, ep, cnt in parts) + " :=")
    A("  ⟨" + ", ".join(f"⟨rfl, {ln}_certified⟩" for ln, ep, cnt in parts) + "⟩")
    A("")
    A("end LipschitzCertDemo")
    A("end Proofs")
    out_path.write_text("\n".join(L) + "\n")
    print(f"[{tag}] {out_path.name}: {len(L)} lines, "
          f"certified " + "/".join(str(len(certs[en])) for _, en in EPS_GRID)
          + f" (IBP " + "/".join(str(counts_ibp[en]) for _, en in EPS_GRID) + ")"
          + f", {len(proved)} images proved, "
          + f"{sum(K - 1 for _ in proved)} kernel facts")
    return certs


if __name__ == "__main__":
    nets = load_nets()
    Xraw, yte = load_mnist()
    IBP = {"SF": {"e1": 92, "e2": 88, "e4": 69, "e8": 24},
           "TF": {"e1": 87, "e2": 42, "e4": 2, "e8": 0}}
    PGD = {"SF": {"e1": 93, "e2": 93, "e4": 92, "e8": 88},
           "TF": {"e1": 95, "e2": 92, "e4": 85, "e8": 36}}
    emit("SF", *nets["SF"], Xraw, yte, OUTDIR / "LipschitzCertScorecardCrown.lean",
         IBP["SF"], PGD["SF"], "spectrally-capped σ≤2 net (`mlpSF`)",
         "LipschitzCertScorecardIBP")
    emit("TF", *nets["TF"], Xraw, yte, OUTDIR / "LipschitzCertScorecardCrownUncon.lean",
         IBP["TF"], PGD["TF"], "unconstrained net (`mlpTF`)",
         "LipschitzCertScorecardIBPUncon")
