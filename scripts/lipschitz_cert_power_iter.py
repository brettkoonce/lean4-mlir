import numpy as np, struct
from fractions import Fraction
from math import ceil

rng = np.random.default_rng(0)
D = "/home/skoonce/lean/klawd_max_power/lean4-jax/data/"

def load_images(fn):
    with open(fn, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

def load_labels(fn):
    with open(fn, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

Xtr_raw = load_images(D + "train-images-idx3-ubyte"); ytr = load_labels(D + "train-labels-idx1-ubyte")
Str = Xtr_raw.reshape(-1, 7, 4, 7, 4).astype(np.int64).sum(axis=(2, 4)).reshape(-1, 49)
Xtr = Str / 4080.0
H, K, DIM = 8, 10, 49
W1 = rng.normal(0, np.sqrt(2.0 / DIM), (H, DIM)); W2 = rng.normal(0, np.sqrt(2.0 / H), (K, H))
lr, bs = 0.15, 64
for ep in range(12):
    idx = rng.permutation(len(Xtr))
    for b in range(0, len(Xtr), bs):
        xb = Xtr[idx[b:b+bs]]; yb = ytr[idx[b:b+bs]]
        h = xb @ W1.T; hr = np.maximum(h, 0); z = hr @ W2.T
        z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = p.copy(); g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
        W1 -= lr * ((g @ W2) * (h > 0)).T @ xb; W2 -= lr * g.T @ hr
W1q = np.round(W1 * 128).astype(np.int64); W2q = np.round(W2 * 128).astype(np.int64)

# sanity: same weights as committed run
import hashlib
print("W1q hash", hashlib.md5(W1q.tobytes()).hexdigest()[:8], "W2q hash", hashlib.md5(W2q.tobytes()).hexdigest()[:8])

DENW = 128
def gram(Wq):  # G = W Wᵀ exact, integer numerators over 128²=16384
    k = Wq.shape[0]
    return [[int(sum(int(Wq[a, j]) * int(Wq[b, j]) for j in range(Wq.shape[1]))) for b in range(k)] for a in range(k)]

G1 = gram(W1q); G2 = gram(W2q)
DEN_G = DENW * DENW  # 16384

def schatten4_B(G):
    S = Fraction(sum(g * g for row in G for g in row), DEN_G ** 2)  # ‖G‖_F² = Σσ⁴
    B = Fraction(ceil((float(S) ** 0.25) * 1000), 1000)
    while B ** 4 < S:
        B += Fraction(1, 1000)
    return S, B

S1, B1 = schatten4_B(G1); S2, B2 = schatten4_B(G2)
LL = B2 * B1
m = Fraction(6953, 500)
print(f"B1={B1}={float(B1)}, B2={B2}={float(B2)}, L=B2*B1={LL}={float(LL):.4f}")
print(f"tight radius = {float(m)/(2**0.5*float(LL)):.5f}  (frobenius was 0.04635)")

# power-iteration vectors: top right-singular vectors, scaled to ints /1000
def pi_lower(Wq, scale=1000):
    W = Wq / DENW
    _, _, Vt = np.linalg.svd(W)
    v = np.round(Vt[0] * scale).astype(np.int64)
    num = [int(sum(int(Wq[i, j]) * int(v[j]) for j in range(Wq.shape[1]))) for i in range(Wq.shape[0])]
    # (Wv)_i = num_i/(128), with v integer; Σ(Wv)² = Σnum²/128², Σv² integer
    sq_Wv = Fraction(sum(x * x for x in num), DENW ** 2)
    sq_v = Fraction(sum(int(x) * int(x) for x in v))
    ray = (sq_Wv / sq_v) ** Fraction(1)  # = ℓ² candidate
    ell = Fraction(int((float(sq_Wv / sq_v) ** 0.5) * 1000), 1000)
    while ell ** 2 * sq_v > sq_Wv:
        ell -= Fraction(1, 1000)
    return v, ell, sq_v, sq_Wv

v1, ell1, sv1, sWv1 = pi_lower(W1q)
v2, ell2, sv2, sWv2 = pi_lower(W2q)
print(f"ell1={ell1}={float(ell1)} (sigma1 7.4525), ell2={ell2}={float(ell2)} (sigma1 7.7008)")

def introw(vals):
    return "![" + ", ".join(f"({int(x)} : ℝ)" for x in vals) + "]"

def gramrow(vals):
    return "![" + ", ".join(f"(({int(x)} : ℝ)/16384)" for x in vals) + "]"

def frac(q):
    q = Fraction(q)
    return f"(({q.numerator} : ℝ)/{q.denominator})" if q.denominator != 1 else f"({q.numerator} : ℝ)"

L = []
A = L.append
A("")
A("-- ════════════════════════════════════════════════════════════")
A("-- § Power-iteration certificate: certified two-sided spectral sandwich")
A("--")
A("-- Upper: the Gram (Schatten-4) bound ‖W‖₂ ≤ ‖WWᵀ‖_F^(1/2) = (Σσᵢ⁴)^(1/4) —")
A(f"--   B₁={float(B1)} / B₂={float(B2)} vs Frobenius 14.555/14.576 ⇒ L drops")
A(f"--   {212.15:.0f}→{float(LL):.1f} and the certified radius grows 0.0463→{float(m)/(2**0.5*float(LL)):.4f} (2.4×).")
A("-- Lower: the power-iteration singular vector, rationalized, certifies that")
A(f"--   ANY valid Lipschitz constant is ≥ ℓ₁={float(ell1)} / ℓ₂={float(ell2)} — so the Gram")
A("--   bound provably sits within 24%/26% of the per-layer optimum.")
A("-- ════════════════════════════════════════════════════════════")
A("")
A("/-- Exact Gram matrix `G1t = W1t·W1tᵀ` (8×8, denominators 128² = 16384). -/")
A("noncomputable def G1t : Fin 8 → Fin 8 → ℝ :=")
A("  ![" + ",\n    ".join(gramrow(G1[a]) for a in range(H)) + "]")
A("")
A("/-- Exact Gram matrix `G2t = W2t·W2tᵀ` (10×10). -/")
A("noncomputable def G2t : Fin 10 → Fin 10 → ℝ :=")
A("  ![" + ",\n    ".join(gramrow(G2[a]) for a in range(K)) + "]")
A("")
A("theorem G1t_eq : ∀ a b, G1t a b = ∑ j, W1t a j * W1t b j := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [G1t, W1t, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A("theorem G2t_eq : ∀ a b, G2t a b = ∑ j, W2t a j * W2t b j := by")
A("  intro a b")
A("  fin_cases a <;> fin_cases b <;>")
A("    · simp [G2t, W2t, Fin.sum_univ_succ]")
A("      norm_num")
A("")
A(f"/-- Schatten-4 Lipschitz bound for the hidden layer: B₁ = {B1} ≈ (Σσ⁴)^(1/4). -/")
A(f"theorem W1t_lip_gram : LipschitzL2 {frac(B1)} (denseE W1t) := by")
A(f"  refine denseE_lipschitzL2_gram W1t G1t (by norm_num) G1t_eq ?_")
A("  simp [G1t, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"theorem W2t_lip_gram : LipschitzL2 {frac(B2)} (denseE W2t) := by")
A(f"  refine denseE_lipschitzL2_gram W2t G2t (by norm_num) G2t_eq ?_")
A("  simp [G2t, Fin.sum_univ_succ]")
A("  norm_num")
A("")
A(f"/-- The tightened product certificate: L = B₂·(1·B₁) = {LL}. -/")
A(f"theorem mlpT_lip_gram : LipschitzL2 {frac(LL)} mlpT := by")
A("  have h := W2t_lip_gram.comp (reluE_lipschitzL2.comp W1t_lip_gram (by norm_num)) (by norm_num)")
A(f"  have e : {frac(B2)} * (1 * {frac(B1)}) = {frac(LL)} := by norm_num")
A("  rw [e] at h; exact h")
A("")
A(f"theorem trained_radius_gram_pos : 0 < {frac(m)} / (Real.sqrt 2 * {frac(LL)}) :=")
A("  div_pos (by norm_num)")
A("    (mul_pos (Real.sqrt_pos.mpr (by norm_num)) (by norm_num))")
A("")
A("/-- **The tightened trained certificate.** Same trained net, same margin, the")
A(f"    Gram bound in place of Frobenius: every `‖δ‖ < {float(m):.3f}/(√2·{float(LL):.2f}) ≈ {float(m)/(2**0.5*float(LL)):.4f}`")
A("    (2.4× the Frobenius radius) leaves the prediction fixed. -/")
A("theorem trained_demo_certified_gram (δ : EuclideanSpace ℝ (Fin 49))")
A(f"    (hδ : ‖δ‖ < {frac(m)} / (Real.sqrt 2 * {frac(LL)})) :")
A("    ∀ j, j ≠ 2 → mlpT (xt + δ) j < mlpT (xt + δ) 2 :=")
A("  lipschitz_margin_certified_radius mlpT_lip_gram (by norm_num) xt_margin hδ")
A("")
A("/-- Rationalized power-iteration vector for `W1t` (top right-singular direction ×1000). -/")
A("noncomputable def v1t : EuclideanSpace ℝ (Fin 49) :=")
A("  WithLp.toLp 2 " + introw(v1))
A("")
A("/-- Rationalized power-iteration vector for `W2t`. -/")
A("noncomputable def v2t : EuclideanSpace ℝ (Fin 8) :=")
A("  WithLp.toLp 2 " + introw(v2))
A("")
A(f"/-- **Certified lower bound**: ANY `L` with `LipschitzL2 L (denseE W1t)` is ≥ {ell1}.")
A(f"    With `W1t_lip_gram : LipschitzL2 {float(B1)} …`, the true `‖W1t‖₂` is sandwiched in")
A(f"    `[{float(ell1)}, {float(B1)}]` — the Gram bound is provably ≤ {float(B1)/float(ell1):.3f}× optimal. -/")
A(f"theorem W1t_lip_lower : ∀ L : ℝ, LipschitzL2 L (denseE W1t) → {frac(ell1)} ≤ L := by")
A("  intro L hL")
A("  refine lipschitzL2_lower_euclid hL (by norm_num) v1t 0 ?_ ?_")
A("  · simp [v1t, Fin.sum_univ_succ]")
A("    norm_num")
A("  · have hc : ∀ i : Fin 8, (denseE W1t v1t - denseE W1t 0) i = ∑ j, W1t i j * v1t j := by")
A("      intro i")
A("      show (∑ j, W1t i j * v1t j) - (∑ j, W1t i j * (0 : EuclideanSpace ℝ (Fin 49)) j) = _")
A("      simp")
A("    simp only [sub_zero, hc]")
A("    simp [W1t, v1t, Fin.sum_univ_succ]")
A("    norm_num")
A("")
A(f"theorem W2t_lip_lower : ∀ L : ℝ, LipschitzL2 L (denseE W2t) → {frac(ell2)} ≤ L := by")
A("  intro L hL")
A("  refine lipschitzL2_lower_euclid hL (by norm_num) v2t 0 ?_ ?_")
A("  · simp [v2t, Fin.sum_univ_succ]")
A("    norm_num")
A("  · have hc : ∀ i : Fin 10, (denseE W2t v2t - denseE W2t 0) i = ∑ j, W2t i j * v2t j := by")
A("      intro i")
A("      show (∑ j, W2t i j * v2t j) - (∑ j, W2t i j * (0 : EuclideanSpace ℝ (Fin 8)) j) = _")
A("      simp")
A("    simp only [sub_zero, hc]")
A("    simp [W2t, v2t, Fin.sum_univ_succ]")
A("    norm_num")

out = "/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax/8f48005c-0a69-42db-9283-1fd4bbae3fe3/scratchpad/pi_snippet.lean"
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out)
