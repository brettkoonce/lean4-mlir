"""Robustness certificate x float bridge (the 2026-07-02 audit's gap #1).

Emits LeanMlir/Proofs/LipschitzCertFloat.lean: the scorecard's Lipschitz-margin
certificates composed with the FloatBridge forward budgets, so the certified
images are provably robust for the FLOAT-EVALUATED net, not just its exact-R
idealization:

  for every ||delta||_2 < eps and every rounded input y within u32*(11/10) of
  img+delta (coordinatewise; covers input quantization), the binary32-rounded
  forward M.mlp2F keeps the true class the strict argmax — for ANY rounding
  model M with M.u <= u32.

The composition is pure margin arithmetic: the Tsuzuku argument gives the real
logit gap at img+delta >= m - sqrt(2)*L*eps; the 2-layer FloatBridge budget B
(layerBudget chain, gamma-form rational bound) perturbs each float logit by at
most B; so any margin clearing (14143/10000)*L*eps + 2*B keeps the float argmax
(strict). Images and margins are parsed from the committed
LipschitzCertScorecard.lean (the source of truth) — no retraining.
"""
import re
from fractions import Fraction

SRC = "/home/skoonce/lean/klawd_max_power/lean4-jax/LeanMlir/Proofs/LipschitzCertScorecard.lean"
OUT = "/home/skoonce/lean/klawd_max_power/lean4-jax/LeanMlir/Proofs/LipschitzCertFloat.lean"
src = open(SRC).read()

FRAC_RE = re.compile(r"\(\((-?\d+) : ℝ\)/(\d+)\)|\((-?\d+) : ℝ\)")

def parse_fracs(block):
    out = []
    for m in FRAC_RE.finditer(block):
        if m.group(1) is not None:
            out.append(Fraction(int(m.group(1)), int(m.group(2))))
        else:
            out.append(Fraction(int(m.group(3))))
    return out

def grab_block(anchor):
    i = src.index(anchor)
    return src[i:src.index("\n\n", i)]

W1s = parse_fracs(grab_block("def W1s :"))
W2s = parse_fracs(grab_block("def W2s :"))
assert len(W1s) == 8 * 49 and len(W2s) == 10 * 8

idx_block = grab_block("def certifiedCappedIdx")
certified = [int(t) for t in re.search(r"\[([\d,\s]+)\]", idx_block).group(1).split(",")]
assert len(certified) == 34

margins, labels, imgs = {}, {}, {}
for k in certified:
    mm = re.search(
        rf"theorem marginC{k} : ∀ j : Fin 10, j ≠ (\d+) →\n\s*\(\((\d+) : ℝ\)/(\d+)\) ≤",
        src)
    assert mm, f"marginC{k} not found"
    labels[k] = int(mm.group(1))
    margins[k] = Fraction(int(mm.group(2)), int(mm.group(3)))
    imgs[k] = parse_fracs(grab_block(f"def img{k} :"))
    assert len(imgs[k]) == 49 and all(0 <= v <= 1 for v in imgs[k])

# ---------------------------------------------------------------- budget chain
L = Fraction(19760433, 1000000)          # mlpS_lip_gram2 (Schatten-8, committed)
EPS = Fraction(1, 10)
U = Fraction(1, 2 ** 24)                 # u32
A = Fraction(11, 10)                     # |img_c + delta_c| <= 1 + eps
EIN = U * A                              # input-quantization budget

w0 = max(abs(v) for v in W1s)            # exact max |W1s|
w1 = max(abs(v) for v in W2s)

def ceil_to(x, den):
    return Fraction(-((-x.numerator * den) // x.denominator), den)

# gamma-form majorants: (1+u)^(m+2) - 1 <= q  via  q >= (m+2)u/(1-(m+2)u)
q0 = ceil_to(51 * U / (1 - 51 * U), 10 ** 12)
q1 = ceil_to(10 * U / (1 - 10 * U), 10 ** 12)

A1 = 49 * w0 * A                                          # layerAct 49 w0 0 (11/10)
E0 = ceil_to(q0 * (49 * w0 * (A + EIN)) + 49 * w0 * EIN, 10 ** 9)
B = ceil_to(q1 * (8 * w1 * (A1 + E0)) + 8 * w1 * E0, 10 ** 7)

thr = Fraction(14143, 10000) * L * EPS + 2 * B
passing = [k for k in certified if thr < margins[k]]
print(f"w0={w0} w1={w1}  A1={float(A1):.3f}  E0={float(E0):.3e}  B={float(B):.3e}")
print(f"float threshold {float(thr):.6f} vs pure-R threshold {float(Fraction(14143,10000)*L*EPS):.6f}")
print(f"float-certified: {len(passing)}/{len(certified)} capped images", flush=True)
dropped = [k for k in certified if k not in passing]
if dropped:
    print("dropped:", dropped, "margins:", {k: float(margins[k]) for k in dropped})

def lit(fr):
    fr = Fraction(fr)
    if fr.denominator == 1:
        return f"({fr.numerator} : ℝ)"
    return f"(({fr.numerator} : ℝ)/{fr.denominator})"

# ---------------------------------------------------------------- per-image blocks
img_blocks = []
for k in passing:
    lab = labels[k]
    img_blocks.append(f"""set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem img{k}_abs_le : ∀ c : Fin 49, |img{k} c| ≤ 1 := by
  intro c
  fin_cases c <;> (simp [img{k}]; try norm_num)

/-- **Float-certified at ε = 1/10** (image #{k}): every binary32-accuracy
    rounded forward of every quantized input within the L2 ball keeps class
    {lab} the strict argmax. -/
theorem certifiedC{k}_float (M : FloatModel) (hMu : M.u ≤ u32)
    (δ : EuclideanSpace ℝ (Fin 49)) (hδ : ‖δ‖ < ((1 : ℝ)/10))
    (y : Vec 49) (hy : ∀ c, |y c - (img{k} + δ) c| ≤ u32 * ((11 : ℝ)/10)) :
    ∀ j, j ≠ {lab} →
      M.mlp2F W1sV zb8 W2sV zb10 y j < M.mlp2F W1sV zb8 W2sV zb10 y {lab} :=
  certifiedFloat_of_margin M hMu img{k} img{k}_abs_le marginC{k}
    (by norm_num) δ hδ y hy
""")

agg_list = ", ".join(str(k) for k in passing)

body = f'''import LeanMlir.Proofs.LipschitzCertScorecard
import LeanMlir.Proofs.Float.FloatBridge

/-! # The robustness certificate composed with the float bridge

The 2026-07-02 audit's gap #1, closed: the scorecard's per-image Lipschitz-
margin certificates (`LipschitzCertScorecard.lean`, exact-ℝ net) composed with
the FloatBridge forward budgets, certifying the FLOAT-EVALUATED capped net.

For each image below: for **every** L2 perturbation `‖δ‖ < ε = 1/10` and
**every** rounded input `y` within `u32·(11/10)` of `img + δ` coordinatewise
(covering input quantization), the rounded two-layer forward `M.mlp2F` — for
**any** rounding model `M` at binary32 accuracy or better (`M.u ≤ u32`) —
keeps the true class the strict argmax.

The composition is pure margin arithmetic (`certified_at_eps_close`): the
Tsuzuku gap at `img + δ` is at least `m − √2·L·ε`; the 2-layer float budget
`B ≤ {lit(B)}` (γ-form `layerBudget` chain at the capped net's exact
magnitude bounds `|W1s| ≤ {lit(w0)}`, `|W2s| ≤ {lit(w1)}`, inputs `≤ 11/10`)
perturbs each logit by at most `B`, so any margin clearing
`(14143/10000)·L·ε + 2·B ≈ {float(thr):.4f}` (vs the ℝ-only threshold
`≈ {float(Fraction(14143,10000)*L*EPS):.4f}`) keeps the float argmax:
**{len(passing)}/34** of the ℝ-certified images survive the float widening
{"(all of them)" if not dropped else f"(dropped: {dropped})"}.

What this does NOT close (unchanged trust boundary, `Binary32Instance.lean`):
the kernel↔model gap — FMA contraction, reduction reassociation, "the GPU
rounds like `rndP`" — and `rndP`'s overflow/subnormal idealization.

Generated by `scripts/lipschitz_cert_float.py` from the committed scorecard
data; weights/images/margins are DATA here. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The rounded two-layer MLP forward and its uniform budget
-- ════════════════════════════════════════════════════════════════

namespace FloatModel

variable (M : FloatModel)

/-- The rounded 2-layer MLP forward: rounded dense, bare (exact) relu,
    rounded dense — the 2-layer face of `mlpF`. -/
noncomputable def mlp2F {{d₀ d₁ d₂ : Nat}}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (x : Vec d₀) : Vec d₂ :=
  M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))

/-- **2-layer MLP forward error, uniform budgets, quantized input.** If the
    device input `y` is within `ein` of the real input `x` coordinatewise
    (input quantization), every rounded logit is within the closed-form
    2-layer `layerBudget` chain of the exact-ℝ logit. The 2-layer face of
    `mlp_float_close_uniform`, with the fresh-input `e = 0` generalized to
    `e = ein`. -/
theorem mlp2_float_close_uniform {{d₀ d₁ d₂ : Nat}}
    {{W₀ : Mat d₀ d₁}} {{b₀ : Vec d₁}} {{W₁ : Mat d₁ d₂}} {{b₁ : Vec d₂}}
    {{x y : Vec d₀}} {{w₀ β₀ w₁ β₁ a ein : ℝ}}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁) (ha : 0 ≤ a)
    (hein : 0 ≤ ein)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hx : ∀ i, |x i| ≤ a) (hy : ∀ i, |y i - x i| ≤ ein) (k : Fin d₂) :
    |M.mlp2F W₀ b₀ W₁ b₁ y k -
        Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) k| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a ein) := by
  have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a ein :=
    layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha hein
  -- layer 0 at the quantized input, uniformized
  have l0 : ∀ j, |M.dense W₀ b₀ y j - Proofs.dense W₀ b₀ x j| ≤
      layerBudget M.u d₀ w₀ β₀ a ein :=
    fun j => (M.dense_close W₀ b₀ y x ein hein hy j).trans
      (M.denseErr_le_uniform hw₀ hein hW₀ hb₀ hx j)
  -- relu: exact pass-through
  have r0 : ∀ j, |relu d₁ (M.dense W₀ b₀ y) j -
      relu d₁ (Proofs.dense W₀ b₀ x) j| ≤ layerBudget M.u d₀ w₀ β₀ a ein :=
    fun j => relu_close _ _ _ l0 j
  -- layer 1 with the inherited error; real hidden magnitude ≤ layerAct
  have ha₁ : ∀ i, |relu d₁ (Proofs.dense W₀ b₀ x) i| ≤ layerAct d₀ w₀ β₀ a :=
    fun i => (relu_abs_le _ i).trans (dense_abs_le ha hW₀ hb₀ hx i)
  exact (M.dense_close W₁ b₁ _ _ _ hE₀0 r0 k).trans
    (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ k)

end FloatModel

namespace LipschitzCertDemo

open scoped BigOperators
open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Generic pieces: coordinate bound, budget monotonicity, composition
-- ════════════════════════════════════════════════════════════════

/-- A single coordinate is bounded by the L2 norm. -/
theorem coord_abs_le_norm {{n : ℕ}} (v : EuclideanSpace ℝ (Fin n)) (c : Fin n) :
    |v c| ≤ ‖v‖ := by
  have h : (v c) ^ 2 ≤ ‖v‖ ^ 2 := by
    rw [euclid_norm_sq]
    exact Finset.single_le_sum (f := fun l => (v l) ^ 2)
      (fun l _ => sq_nonneg _) (Finset.mem_univ c)
  calc |v c| = Real.sqrt ((v c) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (‖v‖ ^ 2) := Real.sqrt_le_sqrt h
    _ = ‖v‖ := Real.sqrt_sq (norm_nonneg v)

/-- Replacing the γ-factor and the inherited error by upper bounds bounds
    `layerBudget` (public copy of the bridge-internal monotonicity step). -/
theorem layerBudget_le_of' {{u : ℝ}} {{m : ℕ}} {{w β A E g Ē : ℝ}}
    (hu : 0 ≤ u) (hw : 0 ≤ w) (hβ : 0 ≤ β) (hA : 0 ≤ A)
    (hG : (1 + u) ^ (m + 2) - 1 ≤ g) (hE0 : 0 ≤ E) (hE : E ≤ Ē) :
    layerBudget u m w β A E ≤
      g * ((m : ℝ) * w * (A + Ē) + β) + (m : ℝ) * w * Ē := by
  have hG0 : (0 : ℝ) ≤ (1 + u) ^ (m + 2) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hmw : (0 : ℝ) ≤ (m : ℝ) * w := mul_nonneg (Nat.cast_nonneg m) hw
  have hb0 : (0 : ℝ) ≤ (m : ℝ) * w * (A + E) + β := by
    have : (0:ℝ) ≤ (m : ℝ) * w * (A + E) :=
      mul_nonneg hmw (add_nonneg hA hE0)
    linarith
  have h1 : ((1 + u) ^ (m + 2) - 1) * ((m : ℝ) * w * (A + E) + β)
      ≤ g * ((m : ℝ) * w * (A + Ē) + β) := by
    refine mul_le_mul hG ?_ hb0 (le_trans hG0 hG)
    have : (m : ℝ) * w * (A + E) ≤ (m : ℝ) * w * (A + Ē) :=
      mul_le_mul_of_nonneg_left (by linarith) hmw
    linarith
  have h2 : (m : ℝ) * w * E ≤ (m : ℝ) * w * Ē :=
    mul_le_mul_of_nonneg_left hE hmw
  show ((1 + u) ^ (m + 2) - 1) * ((m : ℝ) * w * (A + E) + β)
      + (m : ℝ) * w * E ≤ _
  linarith

/-- **The certificate composed with a per-logit evaluation budget.** If the
    ℝ logit map is `L`-Lipschitz with margin `m` at `x`, and the margin
    clears the float-widened threshold `(14143/10000)·L·ε + 2·B`, then ANY
    evaluation `z'` within `B` of the ℝ logits at `x + δ` keeps class `i`
    the strict argmax — for every `‖δ‖ < ε`. Pure margin arithmetic; `z'`
    is the deployed (rounded) forward. -/
theorem certified_at_eps_close {{n k : ℕ}} {{L m ε B : ℝ}}
    {{f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}}
    (hf : LipschitzL2 L f) (hL : 0 < L)
    {{x : EuclideanSpace ℝ (Fin n)}} {{i : Fin k}}
    (hmargin : ∀ j, j ≠ i → m ≤ f x i - f x j)
    (hclear : ((14143 : ℝ)/10000) * L * ε + 2 * B < m)
    (hε0 : 0 ≤ ε) (_hB0 : 0 ≤ B)
    (δ : EuclideanSpace ℝ (Fin n)) (hδ : ‖δ‖ < ε) (z' : Fin k → ℝ)
    (hz' : ∀ j, |z' j - f (x + δ) j| ≤ B) :
    ∀ j, j ≠ i → z' j < z' i := by
  intro j hj
  have hgap := logit_gap_stable hf x δ (Ne.symm hj)
  have hmj := hmargin j hj
  have hs2 : Real.sqrt 2 * L * ‖δ‖ ≤ ((14143 : ℝ)/10000) * L * ε := by
    have h2 : (0:ℝ) ≤ Real.sqrt 2 * L :=
      mul_nonneg (Real.sqrt_nonneg 2) hL.le
    calc Real.sqrt 2 * L * ‖δ‖ ≤ Real.sqrt 2 * L * ε :=
          mul_le_mul_of_nonneg_left hδ.le h2
      _ ≤ ((14143 : ℝ)/10000) * L * ε := by
          have h3 : Real.sqrt 2 * L ≤ ((14143 : ℝ)/10000) * L :=
            mul_le_mul_of_nonneg_right sqrt_two_le_rat hL.le
          exact mul_le_mul_of_nonneg_right h3 hε0
  have hi := abs_le.mp (hz' i)
  have hjj := abs_le.mp (hz' j)
  linarith [hgap, hmj, hs2, hi.1, hjj.2]

-- ════════════════════════════════════════════════════════════════
-- § The capped net in `Vec` space + the ℝ-side tie
-- ════════════════════════════════════════════════════════════════

/-- Capped hidden weights in the `Mat` (input×output) convention. -/
noncomputable def W1sV : Mat 49 8 := fun i k => W1s k i
noncomputable def W2sV : Mat 8 10 := fun k c => W2s c k
noncomputable def zb8 : Vec 8 := fun _ => 0
noncomputable def zb10 : Vec 10 := fun _ => 0

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
theorem W1sV_abs_le : ∀ (i : Fin 49) (j : Fin 8), |W1sV i j| ≤ {lit(w0)} := by
  intro i j
  fin_cases i <;> fin_cases j <;> (simp [W1sV, W1s]; try norm_num)

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
theorem W2sV_abs_le : ∀ (i : Fin 8) (j : Fin 10), |W2sV i j| ≤ {lit(w1)} := by
  intro i j
  fin_cases i <;> fin_cases j <;> (simp [W2sV, W2s]; try norm_num)

theorem zb8_abs_le : ∀ j : Fin 8, |zb8 j| ≤ 0 := fun j => by norm_num [zb8]
theorem zb10_abs_le : ∀ j : Fin 10, |zb10 j| ≤ 0 := fun j => by norm_num [zb10]

/-- The `Vec`-space real forward IS `mlpS`, coordinatewise (transposed
    convention, zero biases, `relu = reluE`). -/
theorem real_tie (y : EuclideanSpace ℝ (Fin 49)) (k : Fin 10) :
    Proofs.dense W2sV zb10 (relu 8 (Proofs.dense W1sV zb8 (fun c => y c))) k
      = mlpS y k := by
  have hinner : ∀ m : Fin 8,
      Proofs.dense W1sV zb8 (fun c => y c) m = denseE W1s y m := by
    intro m
    show (∑ i, y i * W1sV i m) + zb8 m = ∑ i, W1s m i * y i
    rw [show zb8 m = (0:ℝ) from rfl, add_zero]
    exact Finset.sum_congr rfl fun i _ => mul_comm _ _
  show (∑ m, relu 8 (Proofs.dense W1sV zb8 (fun c => y c)) m * W2sV m k)
      + zb10 k = ∑ m, W2s k m * reluE (denseE W1s y) m
  rw [show zb10 k = (0:ℝ) from rfl, add_zero]
  refine Finset.sum_congr rfl fun m _ => ?_
  have hr : relu 8 (Proofs.dense W1sV zb8 (fun c => y c)) m
      = reluE (denseE W1s y) m := by
    show (if Proofs.dense W1sV zb8 (fun c => y c) m > 0
          then Proofs.dense W1sV zb8 (fun c => y c) m else 0)
        = max (denseE W1s y m) 0
    rw [hinner m]
    by_cases h : denseE W1s y m > 0
    · rw [if_pos h, max_eq_left h.le]
    · rw [if_neg h, max_eq_right (not_lt.mp h)]
  rw [hr]
  exact mul_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § The rational budget chain (γ-form, parametric in `M.u ≤ u32`)
-- ════════════════════════════════════════════════════════════════

theorem capped_E0_nonneg (M : FloatModel) :
    (0:ℝ) ≤ layerBudget M.u 49 {lit(w0)} 0 ((11:ℝ)/10) (u32 * ((11:ℝ)/10)) :=
  layerBudget_nonneg M.u_nonneg (by norm_num) le_rfl (by norm_num)
    (by norm_num [u32])

/-- Layer-0 budget at the capped-net magnitudes, with the input-quantization
    error `u32·(11/10)` inherited. -/
theorem capped_E0_le (M : FloatModel) (hMu : M.u ≤ u32) :
    layerBudget M.u 49 {lit(w0)} 0 ((11:ℝ)/10) (u32 * ((11:ℝ)/10))
      ≤ {lit(E0)} := by
  refine (layerBudget_le_of' M.u_nonneg (by norm_num) le_rfl (by norm_num)
    (M.gamma_num (q := {lit(q0)}) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num [u32]) le_rfl).trans ?_
  norm_num [u32]

/-- The whole 2-layer budget: every rounded logit within `{lit(B)}`. -/
theorem capped_B_le (M : FloatModel) (hMu : M.u ≤ u32) :
    layerBudget M.u 8 {lit(w1)} 0 (layerAct 49 {lit(w0)} 0 ((11:ℝ)/10))
      (layerBudget M.u 49 {lit(w0)} 0 ((11:ℝ)/10) (u32 * ((11:ℝ)/10)))
      ≤ {lit(B)} := by
  have hA : layerAct 49 {lit(w0)} 0 ((11:ℝ)/10) = {lit(A1)} := by
    norm_num [layerAct]
  rw [hA]
  refine (layerBudget_le_of' M.u_nonneg (by norm_num) le_rfl (by norm_num)
    (M.gamma_num (q := {lit(q1)}) hMu (by norm_num [u32]) (by norm_num [u32]))
    (capped_E0_nonneg M) (capped_E0_le M hMu)).trans ?_
  norm_num

/-- **The composed per-image engine**: margin clears the float-widened
    threshold ⇒ the rounded forward of any quantized perturbed input keeps
    the class. -/
theorem certifiedFloat_of_margin (M : FloatModel) (hMu : M.u ≤ u32)
    (x : EuclideanSpace ℝ (Fin 49)) (hx1 : ∀ c, |x c| ≤ 1)
    {{i : Fin 10}} {{m : ℝ}}
    (hmargin : ∀ j, j ≠ i → m ≤ mlpS x i - mlpS x j)
    (hclear : ((14143 : ℝ)/10000) * ((19760433 : ℝ)/1000000) * ((1:ℝ)/10)
      + 2 * {lit(B)} < m)
    (δ : EuclideanSpace ℝ (Fin 49)) (hδ : ‖δ‖ < ((1 : ℝ)/10))
    (y : Vec 49) (hy : ∀ c, |y c - (x + δ) c| ≤ u32 * ((11 : ℝ)/10)) :
    ∀ j, j ≠ i →
      M.mlp2F W1sV zb8 W2sV zb10 y j < M.mlp2F W1sV zb8 W2sV zb10 y i := by
  have hmag : ∀ c, |(x + δ) c| ≤ (11:ℝ)/10 := by
    intro c
    have h2 : |δ c| ≤ ‖δ‖ := coord_abs_le_norm δ c
    calc |(x + δ) c| = |x c + δ c| := rfl
      _ ≤ |x c| + |δ c| := abs_add_le _ _
      _ ≤ 1 + 1/10 := add_le_add (hx1 c) (by linarith)
      _ = (11:ℝ)/10 := by norm_num
  have hclose : ∀ k, |M.mlp2F W1sV zb8 W2sV zb10 y k - mlpS (x + δ) k|
      ≤ {lit(B)} := by
    intro k
    have h := M.mlp2_float_close_uniform (x := fun c => (x + δ) c) (y := y)
      (by norm_num : (0:ℝ) ≤ {lit(w0)}) le_rfl
      (by norm_num : (0:ℝ) ≤ {lit(w1)}) (by norm_num : (0:ℝ) ≤ (11:ℝ)/10)
      (by norm_num [u32] : (0:ℝ) ≤ u32 * ((11:ℝ)/10))
      W1sV_abs_le zb8_abs_le W2sV_abs_le zb10_abs_le hmag hy k
    rw [real_tie] at h
    exact h.trans (capped_B_le M hMu)
  exact certified_at_eps_close mlpS_lip_gram2 (by norm_num) hmargin hclear
    (by norm_num) (by norm_num) δ hδ _ hclose

-- ════════════════════════════════════════════════════════════════
-- § The float-certified images
-- ════════════════════════════════════════════════════════════════

{"".join(img_blocks)}
/-- Indices float-certified at ε = 1/10 on the capped net — one
    `certifiedC<i>_float` theorem each. -/
def certifiedFloatIdx : List ℕ := [{agg_list}]

/-- **The float scorecard**: {len(passing)} of the 34 ℝ-certified images
    survive the `2·B` float widening (binary32 forward + input
    quantization) — lower bound only, as before. -/
theorem float_scorecard_count : certifiedFloatIdx.length = {len(passing)} := rfl

end LipschitzCertDemo
end Proofs
'''

with open(OUT, "w") as f:
    f.write(body)
print(f"wrote {OUT}", flush=True)
