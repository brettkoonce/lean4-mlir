"""Certified-accuracy scorecard generator (planning/archive/post_audit_roadmap.md §1).

Produces LeanMlir/Proofs/Certificates/LipschitzCert/Scorecard.lean: over the first 100 MNIST
test images (4x4-pooled, exact pixel-sum rationals), the images whose prediction
is certified robust at eps = 1/10 (pooled-feature L2) by the Lipschitz-margin
certificate, on two nets:

* the UNCONSTRAINED trained /128 net committed in LipschitzCert/Instance.lean
  (L = Schatten-8 product 63.79 -- reproduced here from the same seed/recipe);
* a SPECTRALLY-CAPPED sibling: same recipe + projected SGD onto sigma_max <= 4
  after every step (host-side rescaling, as mnist-mlp-spectral), 36 epochs,
  rationalized to /256.

Certification criterion (exactly what the Lean side proves): the true label
leads every other logit by margin m with (14143/10000)*L*eps <= m, where
14143/10000 >= sqrt 2. Only certified images get per-image theorems; the
aggregate is the honest direction ("at least K of 100"), since an upper-bound
L can never prove an image UNcertifiable.

Also runs an L2-PGD attack (empirical, not proof) on both quantized nets for
the cert <= TRUE <= PGD sandwich table.

Not in CI: it needs MNIST in data/ (directly or through the generator it imports), which CI
does not have. Regenerate by hand and confirm the committed Lean comes back byte-identical.
"""
import numpy as np, os, sys
from fractions import Fraction
from math import ceil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "LeanMlir/Proofs/Certificates/LipschitzCert/Scorecard.lean")
N_IMG = 100
# How many of the certified images carry per-image THEOREMS (hpre/margin/
# certified blocks). The counts stay MEASURED over all N_IMG in exact rationals
# and are emitted below as the `certMargins` data table, which is what
# downstream measurement passes (scripts/certs/lipschitz_cert_float.py) read — so
# capping here cannot silently shrink a reported number. All `img<i>` defs are
# kept regardless: they are cheap and `lipschitz_cert_pair_sdp.py` hardcodes
# that set as EXISTING_IMGS. See planning/archive/scorecard_trim.md.
N_EMIT = int(os.environ.get("SCORECARD_N_EMIT", 8))
EPS = Fraction(1, 10)
SQRT2_UB = Fraction(14143, 10000)   # >= sqrt 2; the factor the Lean proof uses
CAP, EPOCHS, LR, BS = 4.0, 36, 0.15, 64
DEN_U, DEN_C = 128, 256

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))  # the shared helpers
from _mnist_io import mnist, pool_sums, train_mlp  # noqa: E402

Xtr_raw, ytr = mnist("train"); Xte_raw, yte = mnist("test")

Str = pool_sums(Xtr_raw); Ste = pool_sums(Xte_raw)
Xtr = Str / 4080.0; Xte = Ste / 4080.0
H, K, DIM = 8, 10, 49

def train(cap=None, epochs=12, lr=LR, seed=0):
    return train_mlp(Xtr, ytr, H, K, epochs=epochs, lr=lr, bs=BS, cap=cap, seed=seed)

def acc_of(W1f, W2f):
    return (np.argmax(np.maximum(Xte @ W1f.T, 0) @ W2f.T, 1) == yte).mean()

# ── nets ──
W1, W2 = train(cap=None, epochs=12)
W1q = np.round(W1 * DEN_U).astype(np.int64); W2q = np.round(W2 * DEN_U).astype(np.int64)
# must reproduce the committed LipschitzCertInstance weights exactly
assert list(W1q[0, :5]) == [3, -18, -108, -264, -161], "unconstrained net drifted from committed W1t"
assert list(W2q[0, :5]) == [-93, -292, 295, -35, 189], "unconstrained net drifted from committed W2t"

W1c, W2c = train(cap=CAP, epochs=EPOCHS)
W1cq = np.round(W1c * DEN_C).astype(np.int64); W2cq = np.round(W2c * DEN_C).astype(np.int64)
acc_u = acc_of(W1q / DEN_U, W2q / DEN_U); acc_c = acc_of(W1cq / DEN_C, W2cq / DEN_C)
print(f"q-acc: uncon {acc_u:.4f}, capped {acc_c:.4f}")

# ── Schatten-8 product L, exact ──
def gram(Wq):
    return Wq.astype(object) @ Wq.astype(object).T

def s8_B(G, den):
    Hm = G @ G
    S = Fraction(int((Hm ** 2).sum()), den ** 8)
    B = Fraction(ceil((float(S) ** 0.125) * 1000), 1000)
    while B ** 8 < S:
        B += Fraction(1, 1000)
    return B

G1c, G2c = gram(W1cq), gram(W2cq)
H1c, H2c = G1c @ G1c, G2c @ G2c
B1c, B2c = s8_B(G1c, DEN_C), s8_B(G2c, DEN_C)
Lc = B1c * B2c
Lu = Fraction(63791259, 1000000)    # mlpT_lip_gram2 (committed)
print(f"capped: B1'={B1c} B2'={B2c} L={float(Lc):.4f}   uncon L={float(Lu)}")

# ── per-image exact margins + certification at EPS ──
def image_facts(W1z, W2z, den):
    """(pred==label, margin Fraction, preact numerators) per image, exact."""
    facts = []
    for i in range(N_IMG):
        s = Ste[i].astype(object)
        pre = [int(v) for v in W1z.astype(object) @ s]     # /(den*4080)
        hid = np.array([max(v, 0) for v in pre], dtype=object)
        logit = [int(v) for v in W2z.astype(object) @ hid]  # /(den^2*4080)
        y = int(yte[i])
        m = Fraction(logit[y] - max(logit[c] for c in range(K) if c != y),
                     den * den * 4080)
        facts.append((m > 0, m, pre))
    return facts

facts_u = image_facts(W1q, W2q, DEN_U)
facts_c = image_facts(W1cq, W2cq, DEN_C)
cert_u = [i for i, (ok, m, _) in enumerate(facts_u) if ok and SQRT2_UB * Lu * EPS <= m]
cert_c = [i for i, (ok, m, _) in enumerate(facts_c) if ok and SQRT2_UB * Lc * EPS <= m]
# The MEASURED counts — these are what the header reports and what the data
# table below carries. `cert_c`/`cert_u` are narrowed to the theorem-carrying
# subset later (see N_EMIT), after `need_imgs` has taken the full union.
measured_c, measured_u = len(cert_c), len(cert_u)
cert_c_all, cert_u_all = list(cert_c), list(cert_u)
print(f"certified at eps={EPS}: uncon {measured_u}/{N_IMG} {cert_u}")
print(f"                        capped {measured_c}/{N_IMG} {cert_c}")

# ── PGD (empirical upper bracket), L2 ball radius EPS, on the float nets ──
def pgd_robust(W1f, W2f, eps, steps=100, restarts=4):
    robust = 0
    rng = np.random.default_rng(1)
    for i in range(N_IMG):
        x0 = Xte[i]; y = int(yte[i])
        if np.argmax(W2f @ np.maximum(W1f @ x0, 0)) != y:
            continue
        broken = False
        for r in range(restarts):
            d = rng.normal(size=DIM) if r else np.zeros(DIM)
            if r: d *= eps / np.linalg.norm(d)
            for _ in range(steps):
                pre = W1f @ (x0 + d); hr = np.maximum(pre, 0)
                z = W2f @ hr; z = z - z.max()
                p = np.exp(z); p /= p.sum()
                gz = p.copy(); gz[y] -= 1
                g = W1f.T @ ((W2f.T @ gz) * (pre > 0))
                gn = np.linalg.norm(g)
                if gn > 0: d += (2.5 * eps / steps) * g / gn
                dn = np.linalg.norm(d)
                if dn > eps: d *= eps / dn
            if np.argmax(W2f @ np.maximum(W1f @ (x0 + d), 0)) != y:
                broken = True; break
        robust += not broken
    return robust

pgd_u = pgd_robust(W1q / DEN_U, W2q / DEN_U, float(EPS))
pgd_c = pgd_robust(W1cq / DEN_C, W2cq / DEN_C, float(EPS))
print(f"PGD-robust at eps={float(EPS)}: uncon {pgd_u}/{N_IMG}, capped {pgd_c}/{N_IMG}")

# ═══ emit Lean ═══
from _leanlit import frac, rrow as row  # noqa: E402

def mat(M, den):
    return "![" + ",\n    ".join(row(r, den) for r in M) + "]"

def rowq(vals, den):
    return "![" + ", ".join(f"(({int(v)} : ℚ)/{den})" for v in vals) + "]"

def matq(M, den):
    return "![" + ",\n    ".join(rowq(r, den) for r in M) + "]"

need_imgs = sorted(set(cert_u) | set(cert_c))    # ALL measured images keep a def
# cap: from here on `cert_c`/`cert_u` are the theorem-carrying subsets. Applied
# after `need_imgs` so every measured image still gets its `img<i>` definition
# (cheap, and `lipschitz_cert_pair_sdp.py` hardcodes that set as EXISTING_IMGS).
cert_c = cert_c[:N_EMIT]
cert_u = cert_u[:N_EMIT]
print(f"emitting theorems for {len(cert_c)}/{measured_c} capped, "
      f"{len(cert_u)}/{measured_u} uncon images; {len(need_imgs)} img defs kept",
      flush=True)
DEN_HC = DEN_C * 4080
L = []
A = L.append
A("import LeanMlir.Proofs.Certificates.LipschitzCert.Instance")
A("")
A("/-! # Certified-accuracy scorecard")
A("")
# Scope disclaimer — hand-added to the committed file and formerly NOT emitted
# here, so any regeneration silently deleted it (planning/archive/scorecard_trim.md §2.7).
A("**REDUCED CERTIFICATE MODEL** — this file's concrete net is the 4×4-pooled 49-dim")
A("MNIST family (width-8 hidden, /128–/256 rational weights), NOT the canonical")
A("784→512→512→10 `mlpVerified`; chosen so every margin/norm/SOS check is exact rational")
A("arithmetic in-kernel. Canonical surface: [`Proofs/MlpCanonical.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/Small/MlpCanonical.lean).")
A("")
A("The one-input certificate of `LipschitzCert.Instance`, scaled to a")
A(f"dataset-level claim over a FIXED subset — the first {N_IMG} MNIST test images")
A(f"(4×4-pooled, exact pixel-sum rationals) — at a FIXED radius ε = {EPS}")
A("(pooled-feature L2; a pooled coordinate is a 16-pixel block average, so ε")
A("is a 4×4-block-averaged pixel budget of 0.1·255 ≈ 25.5 gray levels")
A("concentrated on one block, or spread L2-wise across blocks). Two nets:")
A("")
A(f"* **unconstrained** — the committed /128 net (`W1t`/`W2t`, q-acc {acc_u:.3f}),")
A(f"  Schatten-8 product L = {float(Lu):.2f} (`mlpT_lip_gram2`):")
A(f"  **{measured_u}/{N_IMG} certified** at ε (measured, see below);")
A(f"* **spectrally capped** — same recipe + projected SGD onto ‖Wᵢ‖₂ ≤ {CAP:g}")
A(f"  (host-side rescaling after every step, as `mnist-mlp-spectral`), {EPOCHS} epochs,")
A(f"  /{DEN_C}-rationalized (`W1s`/`W2s`, q-acc {acc_c:.3f}), Schatten-8 product")
A(f"  L = {float(Lc):.2f}: **{measured_c}/{N_IMG} certified** at the same ε (measured).")
A("")
A("Same theorem, same ε — the training method decides whether the certificate")
A(f"bites (tighter caps, 1.5–2, cost too much clean accuracy at this scale:")
A(f"σ ≤ 2 → 66% test acc; σ ≤ 4 keeps {acc_c:.1%} vs {acc_u:.1%} unconstrained).")
A("")
A("**Theorem vs. measurement — read this before quoting a number.** Soundness")
A("lives in the ENGINE (`certified_at_eps` + `LipschitzCert.Basic`), proved once —")
A("kernel-checking the 57th image buys nothing the 56th didn't. The counts above")
A(f"are exact-rational MEASUREMENTS over the first {N_IMG} images, carried in full by")
A("the `certMargin*` data table at the aggregate below (which is what downstream")
A(f"measurement passes read); the first {N_EMIT} certified images per net additionally")
A("carry `hpre*`/`margin*`/`certified*` THEOREMS, and `scorecard` states only")
A(f"those. Every `img<i>` of the measured set is kept regardless — the {len(need_imgs)} image")
A("definitions are cheap and other tiers reference them.")
A("")
A(f"Each of the {N_EMIT} EMITTED images gets a margin lemma (exact rational, in-kernel)")
A("and a `∀ δ, ‖δ‖ < ε → argmax fixed` theorem via `certified_at_eps`. Every OTHER")
A("certified image appears only as a `-- certMargin*` comment line: a measurement,")
A("carrying no lemma of any kind. The aggregate")
A("count is the honest direction only (\"at least K of 100\") — an upper-bound L")
A("cannot prove an image UNcertifiable. Empirical bracket (not proof): L2-PGD")
A(f"(100 steps, 4 restarts) leaves uncon {pgd_u}/{N_IMG}, capped {pgd_c}/{N_IMG}")
A("robust at the same ε — cert ≤ TRUE ≤ PGD.")
A("")
A("Generated by `scripts/certs/lipschitz_cert_scorecard.py`; weights/images are DATA. -/")
A("")
A("namespace Proofs")
A("namespace LipschitzCertDemo")
A("")
A("open scoped BigOperators")
A("")
A("-- ════════════════════════════════════════════════════════════")
A(f"-- § The spectrally-capped net (σ ≤ {CAP:g} projected SGD, /{DEN_C} rationals)")
A("-- ════════════════════════════════════════════════════════════")
A("")
A(f"/-- Capped-net hidden weights (8×49), entries `k/{DEN_C}`. -/")
A("def W1sQ : Fin 8 → Fin 49 → ℚ :=")
A("  " + matq(W1cq, DEN_C))
A("")
A("noncomputable def W1s : Fin 8 → Fin 49 → ℝ := castM W1sQ")
A("")
A(f"/-- Capped-net output weights (10×8), entries `k/{DEN_C}`. -/")
A("def W2sQ : Fin 10 → Fin 8 → ℚ :=")
A("  " + matq(W2cq, DEN_C))
A("")
A("noncomputable def W2s : Fin 10 → Fin 8 → ℝ := castM W2sQ")
A("")
A("/-- The capped trained MLP: dense → ReLU → dense. -/")
A("noncomputable def mlpS : EuclideanSpace ℝ (Fin 49) → EuclideanSpace ℝ (Fin 10) :=")
A("  denseE W2s ∘ reluE ∘ denseE W1s")
A("")
A(f"/-- `G1s = W1s·W1sᵀ` (8×8, denominators {DEN_C}² = {DEN_C**2}). -/")
A("def G1sQ : Fin 8 → Fin 8 → ℚ :=")
A("  " + matq(G1c, DEN_C**2))
A("")
A("noncomputable def G1s : Fin 8 → Fin 8 → ℝ := castM G1sQ")
A("")
A("def G2sQ : Fin 10 → Fin 10 → ℚ :=")
A("  " + matq(G2c, DEN_C**2))
A("")
A("noncomputable def G2s : Fin 10 → Fin 10 → ℝ := castM G2sQ")
A("")
A(f"/-- `H1s = G1s²` (denominators {DEN_C}⁴ = {DEN_C**4}). -/")
A("def H1sQ : Fin 8 → Fin 8 → ℚ :=")
A("  " + matq(H1c, DEN_C**4))
A("")
A("noncomputable def H1s : Fin 8 → Fin 8 → ℝ := castM H1sQ")
A("")
A("def H2sQ : Fin 10 → Fin 10 → ℚ :=")
A("  " + matq(H2c, DEN_C**4))
A("")
A("noncomputable def H2s : Fin 10 → Fin 10 → ℝ := castM H2sQ")
A("")
A("theorem G1s_eq : ∀ a b, G1s a b = ∑ j, W1s a j * W1s b j :=")
A("  gram_eq_of_check G1sQ W1sQ (by decide +kernel)")
A("")
A("theorem G2s_eq : ∀ a b, G2s a b = ∑ j, W2s a j * W2s b j :=")
A("  gram_eq_of_check G2sQ W2sQ (by decide +kernel)")
A("")
A("theorem H1s_eq : ∀ a b, H1s a b = ∑ c, G1s c a * G1s c b :=")
A("  gram_eq_of_check H1sQ (fun a c => G1sQ c a) (by decide +kernel)")
A("")
A("theorem H2s_eq : ∀ a b, H2s a b = ∑ c, G2s c a * G2s c b :=")
A("  gram_eq_of_check H2sQ (fun a c => G2sQ c a) (by decide +kernel)")
A("")
A(f"/-- Schatten-8 bound for the capped hidden layer: B₁ = {B1c} (cap {CAP:g}). -/")
A(f"theorem W1s_lip_gram2 : LipschitzL2 {frac(B1c)} (denseE W1s) := by")
A("  refine denseE_lipschitzL2_gram2 W1s G1s H1s (by norm_num) G1s_eq H1s_eq ?_")
A("  simp [H1s, H1sQ, castM, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"theorem W2s_lip_gram2 : LipschitzL2 {frac(B2c)} (denseE W2s) := by")
A("  refine denseE_lipschitzL2_gram2 W2s G2s H2s (by norm_num) G2s_eq H2s_eq ?_")
A("  simp [H2s, H2sQ, castM, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"/-- Capped-net Schatten-8 product: L = {float(Lc):.3f} vs {float(Lu):.2f} unconstrained —")
A("    the projection is what makes the fixed-ε certificate bite. -/")
A(f"theorem mlpS_lip_gram2 : LipschitzL2 {frac(Lc)} mlpS := by")
A("  have h := W2s_lip_gram2.comp (reluE_lipschitzL2.comp W1s_lip_gram2 (by norm_num)) (by norm_num)")
A(f"  have e : {frac(B2c)} * (1 * {frac(B1c)}) = {frac(Lc)} := by norm_num")
A("  rw [e] at h; exact h")
A("")
A("-- ════════════════════════════════════════════════════════════")
A(f"-- § The image data: first-{N_IMG} test images needed by a certificate")
A("-- ════════════════════════════════════════════════════════════")
A("")
for i in need_imgs:
    A(f"/-- MNIST test image #{i} (digit {int(yte[i])}), exact pixel sums /4080. -/")
    A(f"noncomputable def img{i} : EuclideanSpace ℝ (Fin 49) :=")
    A("  WithLp.toLp 2 " + row(Ste[i], 4080))
    A("")
A("-- ════════════════════════════════════════════════════════════")
A(f"-- § Per-image certificates, capped net ({len(cert_c)}/{N_IMG} at ε = {EPS})")
A("-- ════════════════════════════════════════════════════════════")
A("")


def emit_image(i, net, W, Wname, mlpname, lipname, Lnet, m, pre, den_h):
    y = int(yte[i])
    A(f"noncomputable def hpre{net}{i} : Fin 8 → ℝ :=")
    A("  " + row(pre, den_h))
    A("")
    A(f"theorem hpre{net}{i}_eval : ∀ k : Fin 8, denseE {Wname} img{i} k = hpre{net}{i} k := by")
    A("  intro k")
    A("  fin_cases k <;>")
    A(f"    · simp [denseE_apply, {Wname}, {Wname}Q, castM, img{i}, hpre{net}{i}, Fin.sum_univ_succ]")
    A("      norm_num")
    A("")
    A(f"theorem margin{net}{i} : ∀ j : Fin 10, j ≠ {y} →")
    A(f"    {frac(m)} ≤ {mlpname} img{i} {y} - {mlpname} img{i} j := by")
    A(f"  have hout : ∀ jj : Fin 10, {mlpname} img{i} jj =")
    A(f"      ∑ k : Fin 8, {net_W2[net]} jj k * max (hpre{net}{i} k) 0 :=")
    A(f"    mlp_out_eq {Wname} {net_W2[net]} hpre{net}{i}_eval")
    A("  intro j hj")
    A("  fin_cases j <;>")
    A("    first")
    A("    | exact absurd rfl hj")
    A("    | · rw [hout, hout]")
    A(f"        simp [{net_W2[net]}, {net_W2[net]}Q, castM, hpre{net}{i}, Fin.sum_univ_succ, max_def]")
    A("        norm_num")
    A("")
    A(f"/-- Test #{i} (digit {y}): certified at ε = {EPS} — margin {float(m):.3f} ≥ √2·L·ε. -/")
    A(f"theorem certified{net}{i} (δ : EuclideanSpace ℝ (Fin 49)) (hδ : ‖δ‖ < {frac(EPS)}) :")
    A(f"    ∀ j, j ≠ {y} → {mlpname} (img{i} + δ) j < {mlpname} (img{i} + δ) {y} :=")
    A(f"  certified_at_eps {lipname} (by norm_num) margin{net}{i} (by norm_num)")
    A("    (by norm_num) δ hδ")
    A("")


net_W2 = {"C": "W2s", "U": "W2t"}
# cap: the first N_EMIT certified images per net carry theorems. `measured_*`
# keep the full exact-rational counts for the header and the data table.
for i in cert_c:
    emit_image(i, "C", W1cq, "W1s", "mlpS", "mlpS_lip_gram2", Lc,
               facts_c[i][1], facts_c[i][2], DEN_HC)
A("-- ════════════════════════════════════════════════════════════")
A(f"-- § Per-image certificates, unconstrained net ({len(cert_u)}/{N_IMG} at the same ε)")
A("-- ════════════════════════════════════════════════════════════")
A("")
for i in cert_u:
    emit_image(i, "U", W1q, "W1t", "mlpT", "mlpT_lip_gram2", Lu,
               facts_u[i][1], facts_u[i][2], DEN_U * 4080)
A("-- ════════════════════════════════════════════════════════════")
A("-- § Aggregate — the mechanized scorecard")
A("-- ════════════════════════════════════════════════════════════")
A("")
# The MEASURED population, as data. This is deliberately a comment and not a
# Lean declaration: a count over a fixed subset is exact rational arithmetic,
# not a theorem, and emitting it costs zero elaboration. It is also the contract
# with downstream measurement passes — scripts/certs/lipschitz_cert_float.py parses
# these lines, so capping N_EMIT above can never shrink the population a
# downstream tier measures over. Format: `-- certMargin<net> <i> <class> <n>/<d>`.
A(f"-- ════════ MEASURED margins (data, not theorems) ════════")
A(f"-- Every image certified at ε = {EPS}: {measured_c} capped, {measured_u} unconstrained, over the")
A(f"-- first {N_IMG} test images. Exact rationals, computed off-line — the population")
A(f"-- the header's counts are measured over, and what downstream measurement")
A(f"-- passes read (scripts/certs/lipschitz_cert_float.py). The first {N_EMIT} per net also")
A("-- carry theorems above; these lines are arithmetic, and prove nothing.")
for i in cert_c_all:
    m = facts_c[i][1]
    A(f"-- certMarginC {i} {int(yte[i])} {m.numerator}/{m.denominator}")
for i in cert_u_all:
    m = facts_u[i][1]
    A(f"-- certMarginU {i} {int(yte[i])} {m.numerator}/{m.denominator}")
A("")


def wrap(items, per_line, first, rest):
    """Emit `items` `per_line` to a line, `first` opening the first line."""
    out = []
    for s in range(0, len(items), per_line):
        chunk = ", ".join(items[s:s + per_line])
        out.append((first if s == 0 else rest) + chunk +
                   ("," if s + per_line < len(items) else ""))
    return out


# The mechanized aggregate, stated in `CertifiedAt` (defined in `DenseEuclid`, the engine every
# certificate tier's scorecard is stated in).
A("/-- The capped-net certificate witnesses: `(subset index, image, class)`,")
A("    one triple per `certifiedC<i>` theorem, in index order. -/")
A("noncomputable def cappedCerts : List (ℕ × EuclideanSpace ℝ (Fin 49) × Fin 10) :=")
for ln in wrap([f"({i}, img{i}, {int(yte[i])})" for i in cert_c], 5, "  [", "   "):
    A(ln)
A(L.pop() + "]")
A("")
A("/-- The unconstrained-net certificate witnesses (`certifiedU<i>`). -/")
A("noncomputable def unconCerts : List (ℕ × EuclideanSpace ℝ (Fin 49) × Fin 10) :=")
for ln in wrap([f"({i}, img{i}, {int(yte[i])})" for i in cert_u], 5, "  [", "   "):
    A(ln)
A(L.pop() + "]")
A("")
A("/-- **Every capped-net witness is certified** — the aggregate is no longer")
A("    bookkeeping over a bare index list: the proof term is literally the")
A(f"    tuple of the {len(cert_c)} per-image theorems. -/")
A("theorem cappedCerts_certified :")
A(f"    ∀ p ∈ cappedCerts, CertifiedAt mlpS {frac(EPS)} p.2.1 p.2.2 :=")
A("  List.forall_iff_forall_mem.mp")
if len(cert_c) == 1:
    A(L.pop() + f" certifiedC{cert_c[0]}")
else:
    for ln in wrap([f"certifiedC{i}" for i in cert_c], 5, "    ⟨", "     "):
        A(ln)
    A(L.pop() + "⟩")
A("")
A("theorem unconCerts_certified :")
A(f"    ∀ p ∈ unconCerts, CertifiedAt mlpT {frac(EPS)} p.2.1 p.2.2 :=")
A("  List.forall_iff_forall_mem.mp")
if len(cert_u) == 1:
    A(L.pop() + f" certifiedU{cert_u[0]}")
else:
    for ln in wrap([f"certifiedU{i}" for i in cert_u], 5, "    ⟨", "     "):
        A(ln)
    A(L.pop() + "⟩")
A("")
A(f"/-- **The proved core of the scorecard**: the {len(cert_c)} capped-net and {len(cert_u)}")
A("    unconstrained-net witnesses in `cappedCerts`/`unconCerts` each carry a")
A(f"    `CertifiedAt … ({EPS})` proof (pooled L2; `cappedCerts_certified`/")
A(f"    `unconCerts_certified`). The dataset counts ({measured_c}/{N_IMG} capped,")
A(f"    {measured_u}/{N_IMG} unconstrained) are exact-rational measurements recorded in the")
A("    `certMargin*` lines above, not theorems. Lower bounds only: an upper-bound L")
A("    cannot prove an image uncertifiable. -/")
A("theorem scorecard :")
A(f"    (cappedCerts.length = {len(cert_c)} ∧")
A(f"      ∀ p ∈ cappedCerts, CertifiedAt mlpS {frac(EPS)} p.2.1 p.2.2) ∧")
A(f"    (unconCerts.length = {len(cert_u)} ∧")
A(f"      ∀ p ∈ unconCerts, CertifiedAt mlpT {frac(EPS)} p.2.1 p.2.2) :=")
A("  ⟨⟨rfl, cappedCerts_certified⟩, ⟨rfl, unconCerts_certified⟩⟩")
A("")
A("end LipschitzCertDemo")
A("end Proofs")

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(L)} lines, {len(need_imgs)} images, "
      f"{len(cert_c)}+{len(cert_u)} certificates")
