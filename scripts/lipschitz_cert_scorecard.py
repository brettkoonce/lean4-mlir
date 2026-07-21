"""Certified-accuracy scorecard generator (planning/post_audit_roadmap.md §1).

Produces LeanMlir/Proofs/Certificates/LipschitzCertScorecard.lean: over the first 100 MNIST
test images (4x4-pooled, exact pixel-sum rationals), the images whose prediction
is certified robust at eps = 1/10 (pooled-feature L2) by the Lipschitz-margin
certificate, on two nets:

* the UNCONSTRAINED trained /128 net committed in LipschitzCertInstance.lean
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
"""
import numpy as np, struct
from fractions import Fraction
from math import ceil

D = "/home/skoonce/lean/klawd_max_power/lean4-jax/data/"
OUT = "/home/skoonce/lean/klawd_max_power/lean4-jax/LeanMlir/Proofs/Certificates/LipschitzCertScorecard.lean"
N_IMG = 100
EPS = Fraction(1, 10)
SQRT2_UB = Fraction(14143, 10000)   # >= sqrt 2; the factor the Lean proof uses
CAP, EPOCHS, LR, BS = 4.0, 36, 0.15, 64
DEN_U, DEN_C = 128, 256

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
Xtr = Str / 4080.0; Xte = Ste / 4080.0
H, K, DIM = 8, 10, 49

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
print(f"certified at eps={EPS}: uncon {len(cert_u)}/{N_IMG} {cert_u}")
print(f"                        capped {len(cert_c)}/{N_IMG} {cert_c}")

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
def frac(q):
    q = Fraction(q)
    return f"(({q.numerator} : ℝ)/{q.denominator})" if q.denominator != 1 else f"({q.numerator} : ℝ)"

def row(vals, den):
    return "![" + ", ".join(f"(({int(v)} : ℝ)/{den})" for v in vals) + "]"

def mat(M, den):
    return "![" + ",\n    ".join(row(r, den) for r in M) + "]"

need_imgs = sorted(set(cert_u) | set(cert_c))
DEN_HC = DEN_C * 4080
L = []
A = L.append
A("import LeanMlir.Proofs.Certificates.LipschitzCertInstance")
A("")
A("/-! # Certified-accuracy scorecard (post_audit_roadmap §1)")
A("")
A("The one-input certificate of `LipschitzCertInstance.lean`, scaled to a")
A(f"dataset-level claim over a FIXED subset — the first {N_IMG} MNIST test images")
A(f"(4×4-pooled, exact pixel-sum rationals) — at a FIXED radius ε = {EPS}")
A("(pooled-feature L2; a pooled coordinate is a 16-pixel block average, so ε")
A("is a 4×4-block-averaged pixel budget of 0.1·255 ≈ 25.5 gray levels")
A("concentrated on one block, or spread L2-wise across blocks). Two nets:")
A("")
A(f"* **unconstrained** — the committed /128 net (`W1t`/`W2t`, q-acc {acc_u:.3f}),")
A(f"  Schatten-8 product L = {float(Lu):.2f} (`mlpT_lip_gram2`):")
A(f"  **{len(cert_u)}/{N_IMG} certified** at ε;")
A(f"* **spectrally capped** — same recipe + projected SGD onto ‖Wᵢ‖₂ ≤ {CAP:g}")
A(f"  (host-side rescaling after every step, as `mnist-mlp-spectral`), {EPOCHS} epochs,")
A(f"  /{DEN_C}-rationalized (`W1s`/`W2s`, q-acc {acc_c:.3f}), Schatten-8 product")
A(f"  L = {float(Lc):.2f}: **{len(cert_c)}/{N_IMG} certified** at the same ε.")
A("")
A("Same theorem, same ε — the training method decides whether the certificate")
A(f"bites (the roadmap's caps 1.5–2 cost too much clean accuracy at this scale:")
A(f"σ ≤ 2 → 66% test acc; σ ≤ 4 keeps {acc_c:.1%} vs {acc_u:.1%} unconstrained).")
A("")
A("Each certified image gets a margin lemma (exact rational, in-kernel) and a")
A("`∀ δ, ‖δ‖ < ε → argmax fixed` theorem via `certified_at_eps`; the aggregate")
A("count is the honest direction only (\"at least K of 100\") — an upper-bound L")
A("cannot prove an image UNcertifiable. Empirical bracket (not proof): L2-PGD")
A(f"(100 steps, 4 restarts) leaves uncon {pgd_u}/{N_IMG}, capped {pgd_c}/{N_IMG}")
A("robust at the same ε — cert ≤ TRUE ≤ PGD.")
A("")
A("Generated by `scripts/lipschitz_cert_scorecard.py`; weights/images are DATA. -/")
A("")
A("namespace Proofs")
A("namespace LipschitzCertDemo")
A("")
A("open scoped BigOperators")
A("")
A("/-- `√2 ≤ 14143/10000` — the rational majorant the per-image radius checks use. -/")
A("theorem sqrt_two_le_rat : Real.sqrt 2 ≤ ((14143 : ℝ)/10000) := by")
A("  rw [show ((14143 : ℝ)/10000) = Real.sqrt (((14143 : ℝ)/10000) ^ 2) from")
A("    (Real.sqrt_sq (by norm_num)).symm]")
A("  exact Real.sqrt_le_sqrt (by norm_num)")
A("")
A("/-- Specialize the Tsuzuku certificate to a FIXED radius ε: if the margin")
A("    clears the rational check `(14143/10000)·L·ε ≤ m` (kernel-checkable —")
A("    no `√2`), every `‖δ‖ < ε` leaves class `i` the strict argmax. -/")
A("theorem certified_at_eps {n k : ℕ} {L m ε : ℝ}")
A("    {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}")
A("    (hf : LipschitzL2 L f) (hL : 0 < L) {x : EuclideanSpace ℝ (Fin n)}")
A("    {i : Fin k} (hmargin : ∀ j, j ≠ i → m ≤ f x i - f x j)")
A("    (hε : ((14143 : ℝ)/10000) * L * ε ≤ m) (hε0 : 0 ≤ ε)")
A("    (δ : EuclideanSpace ℝ (Fin n)) (hδ : ‖δ‖ < ε) :")
A("    ∀ j, j ≠ i → f (x + δ) j < f (x + δ) i := by")
A("  refine lipschitz_margin_certified_radius hf hL hmargin (lt_of_lt_of_le hδ ?_)")
A("  rw [le_div_iff₀ (mul_pos (Real.sqrt_pos.mpr (by norm_num)) hL)]")
A("  calc ε * (Real.sqrt 2 * L) ≤ ε * (((14143 : ℝ)/10000) * L) := by")
A("        have h2 : (0:ℝ) ≤ L := le_of_lt hL")
A("        have := mul_le_mul_of_nonneg_right sqrt_two_le_rat h2")
A("        exact mul_le_mul_of_nonneg_left this hε0")
A("    _ = ((14143 : ℝ)/10000) * L * ε := by ring")
A("    _ ≤ m := hε")
A("")
A("-- ════════════════════════════════════════════════════════════")
A(f"-- § The spectrally-capped net (σ ≤ {CAP:g} projected SGD, /{DEN_C} rationals)")
A("-- ════════════════════════════════════════════════════════════")
A("")
A(f"/-- Capped-net hidden weights (8×49), entries `k/{DEN_C}`. -/")
A("noncomputable def W1s : Fin 8 → Fin 49 → ℝ :=")
A("  " + mat(W1cq, DEN_C))
A("")
A(f"/-- Capped-net output weights (10×8), entries `k/{DEN_C}`. -/")
A("noncomputable def W2s : Fin 10 → Fin 8 → ℝ :=")
A("  " + mat(W2cq, DEN_C))
A("")
A("/-- The capped trained MLP: dense → ReLU → dense. -/")
A("noncomputable def mlpS : EuclideanSpace ℝ (Fin 49) → EuclideanSpace ℝ (Fin 10) :=")
A("  denseE W2s ∘ reluE ∘ denseE W1s")
A("")
A(f"/-- `G1s = W1s·W1sᵀ` (8×8, denominators {DEN_C}² = {DEN_C**2}). -/")
A("noncomputable def G1s : Fin 8 → Fin 8 → ℝ :=")
A("  " + mat(G1c, DEN_C**2))
A("")
A("noncomputable def G2s : Fin 10 → Fin 10 → ℝ :=")
A("  " + mat(G2c, DEN_C**2))
A("")
A(f"/-- `H1s = G1s²` (denominators {DEN_C}⁴ = {DEN_C**4}). -/")
A("noncomputable def H1s : Fin 8 → Fin 8 → ℝ :=")
A("  " + mat(H1c, DEN_C**4))
A("")
A("noncomputable def H2s : Fin 10 → Fin 10 → ℝ :=")
A("  " + mat(H2c, DEN_C**4))
A("")
A("set_option maxHeartbeats 3200000 in")
A("theorem G1s_eq : ∀ a b, G1s a b = ∑ j, W1s a j * W1s b j := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [G1s, W1s, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A("set_option maxHeartbeats 3200000 in")
A("theorem G2s_eq : ∀ a b, G2s a b = ∑ j, W2s a j * W2s b j := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [G2s, W2s, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A("set_option maxHeartbeats 1600000 in")
A("theorem H1s_eq : ∀ a b, H1s a b = ∑ c, G1s c a * G1s c b := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [H1s, G1s, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A("set_option maxHeartbeats 1600000 in")
A("theorem H2s_eq : ∀ a b, H2s a b = ∑ c, G2s c a * G2s c b := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [H2s, G2s, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A(f"/-- Schatten-8 bound for the capped hidden layer: B₁ = {B1c} (cap {CAP:g}). -/")
A(f"theorem W1s_lip_gram2 : LipschitzL2 {frac(B1c)} (denseE W1s) := by")
A("  refine denseE_lipschitzL2_gram2 W1s G1s H1s (by norm_num) G1s_eq H1s_eq ?_")
A("  simp [H1s, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"theorem W2s_lip_gram2 : LipschitzL2 {frac(B2c)} (denseE W2s) := by")
A("  refine denseE_lipschitzL2_gram2 W2s G2s H2s (by norm_num) G2s_eq H2s_eq ?_")
A("  simp [H2s, Fin.sum_univ_succ]")
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
    A(f"    · simp [denseE_apply, {Wname}, img{i}, hpre{net}{i}, Fin.sum_univ_succ]")
    A("      norm_num")
    A("")
    A(f"theorem margin{net}{i} : ∀ j : Fin 10, j ≠ {y} →")
    A(f"    {frac(m)} ≤ {mlpname} img{i} {y} - {mlpname} img{i} j := by")
    A(f"  have hout : ∀ jj : Fin 10, {mlpname} img{i} jj =")
    A(f"      ∑ k : Fin 8, {net_W2[net]} jj k * max (hpre{net}{i} k) 0 := by")
    A("    intro jj")
    A(f"    show denseE {net_W2[net]} (reluE (denseE {Wname} img{i})) jj = _")
    A("    rw [denseE_apply]")
    A("    refine Finset.sum_congr rfl fun k _ => ?_")
    A(f"    rw [reluE_apply, hpre{net}{i}_eval k]")
    A("  intro j hj")
    A("  fin_cases j <;>")
    A("    first")
    A("    | exact absurd rfl hj")
    A("    | · rw [hout, hout]")
    A(f"        simp [{net_W2[net]}, hpre{net}{i}, Fin.sum_univ_succ, max_def]")
    A("        norm_num")
    A("")
    A(f"/-- Test #{i} (digit {y}): certified at ε = {EPS} — margin {float(m):.3f} ≥ √2·L·ε. -/")
    A(f"theorem certified{net}{i} (δ : EuclideanSpace ℝ (Fin 49)) (hδ : ‖δ‖ < {frac(EPS)}) :")
    A(f"    ∀ j, j ≠ {y} → {mlpname} (img{i} + δ) j < {mlpname} (img{i} + δ) {y} :=")
    A(f"  certified_at_eps {lipname} (by norm_num) margin{net}{i} (by norm_num)")
    A("    (by norm_num) δ hδ")
    A("")


net_W2 = {"C": "W2s", "U": "W2t"}
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
A("-- § Aggregate")
A("-- ════════════════════════════════════════════════════════════")
A("")
A(f"/-- Indices (into the fixed first-{N_IMG} MNIST test subset) certified at")
A(f"    ε = {EPS} on the CAPPED net — one `certifiedC<i>` theorem each. -/")
A("def certifiedCappedIdx : List ℕ :=")
A("  [" + ", ".join(str(i) for i in cert_c) + "]")
A("")
A(f"/-- Indices certified on the UNCONSTRAINED net (`certifiedU<i>`). -/")
A("def certifiedUnconIdx : List ℕ :=")
A("  [" + ", ".join(str(i) for i in cert_u) + "]")
A("")
A(f"/-- **The scorecard**: at ε = {EPS} (pooled L2), the capped net certifies")
A(f"    {len(cert_c)}/{N_IMG} of the fixed test subset, the unconstrained net {len(cert_u)}/{N_IMG} —")
A("    same theorem, same ε; training (σ-projection) decides whether the")
A("    certificate bites. Lower bounds only: an upper-bound L cannot prove")
A("    an image uncertifiable. -/")
A(f"theorem scorecard_counts :")
A(f"    certifiedCappedIdx.length = {len(cert_c)} ∧ certifiedUnconIdx.length = {len(cert_u)} :=")
A("  ⟨rfl, rfl⟩")
A("")
A("end LipschitzCertDemo")
A("end Proofs")

with open(OUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(L)} lines, {len(need_imgs)} images, "
      f"{len(cert_c)}+{len(cert_u)} certificates")
