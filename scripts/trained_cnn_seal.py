"""Level-3 seal for the trained-CNN witness (the MLP rung's pdiv_fwd, CNN edition).

Emits LeanMlir/Proofs/Training/TrainedCnnSeal.lean: one whole-net Jacobian entry
pdiv fwd X ic jc computed in CLOSED FORM at the trained weights, by peeling
the network with pdiv_comp from the output side and carrying exact
backward-cotangent tables:

  t4 (8)  = pdiv(dense5 ∘ relu∘dense4)         at r3V, class jc
  t3 (18) = ... ∘ relu∘dense3                  at p2f
  t2 (72) = ... ∘ maxPoolFlat                  at flatten r2V (argmax routing)
  m2 (72) = relu mask2 ⊙ t2                    (relu fold at flatten c2V)
  t1 (72) = conv2d_input_grad_formula W2 m2    (conv2 input-VJP)
  m1 (72) = relu mask1 ⊙ t1
  pd      = conv2d_input_grad_formula W1 m1 [ic]

Each pool/conv stage converts its pdiv_comp sum into the layer witness's
explicit backward via HasVJPAt.correct, then evaluates it in-kernel.
Seal: HasVJPAt.backward_ne_zero_of_pdiv_ne + fderiv/not-constant forms —
the exact theorem set TrainedMlpWitness carries, now for the conv net.

Runs scripts/trained_cnn_witness.py first (deterministic) to reproduce the
trained weights and forward tables; weights/input are DATA here.
"""
import numpy as np
from fractions import Fraction

# ---- reproduce the witness (trains the net, computes exact tables, and
# ---- rewrites TrainedCnnWitness.lean with identical content)
exec(open("/home/skoonce/lean/klawd_max_power/lean4-jax/scripts/trained_cnn_witness.py").read())

SEAL_OUT = "/home/skoonce/lean/klawd_max_power/lean4-jax/LeanMlir/Proofs/Training/TrainedCnnSeal.lean"
JC = 7  # sealed output class = the witness's (correct) prediction

# ---------------------------------------------------------------- seal tables (exact)
mask4 = [1 if d4F[m] > 0 else 0 for m in range(8)]
mask3 = [1 if d3F[m] > 0 else 0 for m in range(8)]
mask2 = np.vectorize(lambda v: 1 if v > 0 else 0)(c2F)
mask1 = np.vectorize(lambda v: 1 if v > 0 else 0)(c1F)

t4 = [sum(W4r[k, m] * mask4[m] * W5r[m, JC] for m in range(8)) for k in range(8)]
t3 = [sum(W3r[k3, m] * mask3[m] * t4[m] for m in range(8)) for k3 in range(18)]

# pool routing: t2[ci,hi,wi] = t3[window] iff (ci,hi,wi) is its window's argmax
argmax2 = np.empty((C, 6, 6), dtype=int)
for ci in range(C):
    for hi in range(6):
        for wi in range(6):
            argmax2[ci, hi, wi] = 1 if r2F[ci, hi, wi] == poolF[ci, hi // 2, wi // 2] else 0
assert argmax2.reshape(C, 3, 2, 3, 2).transpose(0, 1, 3, 2, 4).reshape(C, 9, 4).sum(-1).max() == 1

t2 = np.empty((C, 6, 6), dtype=object)
for ci in range(C):
    for hi in range(6):
        for wi in range(6):
            t2[ci, hi, wi] = t3[ci * 9 + (hi // 2) * 3 + (wi // 2)] if argmax2[ci, hi, wi] else Fraction(0)

M2 = np.vectorize(lambda m, v: v if m else Fraction(0))(mask2, t2)

def input_grad(Wr, cin, dyT):
    """mirror of conv2d_input_grad_formula (pH=pW=1, 3x3):
    out[ci,hi,wi] = sum_{co,ho,wo} [ho<=hi+1 & hi+1-ho<3 & wo<=wi+1 & wi+1-wo<3]
                    Wr[co, ci*9 + (hi+1-ho)*3 + (wi+1-wo)] * dyT[co,ho,wo]"""
    oc = dyT.shape[0]
    out = np.empty((cin, 6, 6), dtype=object)
    for ci in range(cin):
        for hi in range(6):
            for wi in range(6):
                s = Fraction(0)
                for co in range(oc):
                    for ho in range(6):
                        for wo in range(6):
                            kh, kw = hi + 1 - ho, wi + 1 - wo
                            if 0 <= kh < 3 and 0 <= kw < 3:
                                s += Wr[co, ci * 9 + kh * 3 + kw] * dyT[co, ho, wo]
                out[ci, hi, wi] = s
    return out

t1 = input_grad(W2r, C, M2)
M1 = np.vectorize(lambda m, v: v if m else Fraction(0))(mask1, t1)
pdT = input_grad(W1r, 1, M1)

# pick the input pixel with the largest |entry|
best = max(((abs(pdT[0, hi, wi]), hi, wi) for hi in range(6) for wi in range(6)))
_, HI, WI = best
PD = pdT[0, HI, WI]
IC = HI * 6 + WI  # flat index in Vec (1*6*6), ci = 0
assert PD != 0, "chosen Jacobian entry is zero — pick another (i, j)"
print(f"seal entry: d logit_{JC} / d pixel ({HI},{WI})  [flat {IC}]  = {PD} ≈ {float(PD):.4f}")

# float finite-difference sanity check
eps = 1e-5
xb = Xte4[wi_idx:wi_idx + 1].copy()
xp = xb.copy(); xp[0, 0, HI, WI] += eps
xm = xb.copy(); xm[0, 0, HI, WI] -= eps
fd = (forward_q(xp)[0, JC] - forward_q(xm)[0, JC]) / (2 * eps)
assert abs(fd - float(PD)) < 1e-4, f"finite-difference mismatch: {fd} vs {float(PD)}"
print(f"finite-difference check: {fd:.6f} vs exact {float(PD):.6f} ✓")

# ---------------------------------------------------------------- Lean emission
def fm2(k, n):
    return f"(⟨{k}, by norm_num⟩ : Fin ({n}))"

JCL = "⟨7, by norm_num⟩"  # Fin 10 inferred from context

# suffix composite spellings (left-nested; rightmost factor = input side)
R72 = "relu (2 * (2*3) * (2*3))"
G4 = "dense W5 b5 ∘ (relu 8 ∘ dense W4 b4)"
G3 = f"({G4}) ∘ (relu 8 ∘ dense W3 b3)"
G2 = f"({G3}) ∘ maxPoolFlat 2 3 3"
G2r = f"({G2}) ∘ {R72}"
G1 = f"({G2r}) ∘ flatConv (h := 2*3) (w := 2*3) W2 b2"
G1r = f"({G1}) ∘ {R72}"

def t3lit(t):
    return t3_lit(t)

# per-case S2 lemmas (pool stage)
s2_cases = []
for k2 in range(72):
    ci, hi, wi = k2 // 36, (k2 % 36) // 6, k2 % 6
    ho, wo = hi // 2, wi // 2
    win_flat = ci * 9 + ho * 3 + wo
    if argmax2[ci, hi, wi]:
        branch = f"""  rw [if_pos (by
    intro a b
    fin_cases a <;> fin_cases b <;>
      (simp [winRow, winCol, winRowInv, winColInv, r2V]; try norm_num))]
  show t3V {fm2(win_flat, '2*3*3')} = t2V {fm2(k2, '2*(2*3)*(2*3)')}
  norm_num [t3V, t2V]"""
    else:
        # violating cell: the window's argmax
        for a in range(2):
            for b in range(2):
                if argmax2[ci, 2 * ho + a, 2 * wo + b]:
                    av, bv = a, b
        branch = f"""  rw [if_neg (by
    intro hmax
    have hv := hmax (⟨{av}, by norm_num⟩ : Fin 2) (⟨{bv}, by norm_num⟩ : Fin 2)
    simp [winRow, winCol, winRowInv, winColInv, r2V] at hv
    norm_num at hv)]
  show (0 : ℝ) = t2V {fm2(k2, '2*(2*3)*(2*3)')}
  norm_num [t2V]"""
    s2_cases.append(f"""set_option maxRecDepth 16384 in
set_option maxHeartbeats 4000000 in
theorem S2_c{k2} :
    pdiv ({G2}) (Tensor3.flatten r2V) {fm2(k2, '2*(2*3)*(2*3)')} {JCL} = t2V {fm2(k2, '2*(2*3)*(2*3)')} := by
  have hP : DifferentiableAt ℝ (maxPoolFlat 2 3 3) (Tensor3.flatten r2V) :=
    maxPoolFlat_differentiableAt (c := 2) (h := 3) (w := 3) r2V r2_smooth
      (by norm_num) (by norm_num) (by norm_num)
  have hG : DifferentiableAt ℝ ({G3}) (maxPoolFlat 2 3 3 (Tensor3.flatten r2V)) := by
    rw [pooled_eq]; exact diffG3_p2f
  rw [pdiv_comp _ _ _ hP hG]
  simp only [pooled_eq]
  simp only [S3]
  rw [← (maxPoolFlat_has_vjp_at r2V r2_smooth).correct t3V {fm2(k2, '2*(2*3)*(2*3)')}]
  show (if MaxPool2IsArgmax (c := 2) (h := 3) (w := 3) r2V
        {fm2(ci, 2)} {fm2(hi, 6)} {fm2(wi, 6)}
      then Tensor3.unflatten t3V {fm2(ci, 2)}
        (winRow {fm2(hi, 6)}) (winCol {fm2(wi, 6)}) else 0)
    = t2V {fm2(k2, '2*(2*3)*(2*3)')}
{branch}
""")

s2_agg_bullets = "\n".join(f"  · exact S2_c{k}" for k in range(72))

# per-case S1 lemmas (conv2 stage)
s1_cases = []
for k1 in range(72):
    ci, hi, wi = k1 // 36, (k1 % 36) // 6, k1 % 6
    s1_cases.append(f"""set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem S1_c{k1} :
    pdiv ({G1}) (Tensor3.flatten z1V) {fm2(k1, '2*(2*3)*(2*3)')} {JCL} = t1V {fm2(k1, '2*(2*3)*(2*3)')} := by
  have hC : DifferentiableAt ℝ (flatConv (h := 2*3) (w := 2*3) W2 b2) (Tensor3.flatten z1V) :=
    (flatConv_differentiable W2 b2) _
  have hG : DifferentiableAt ℝ ({G2r})
      (flatConv (h := 2*3) (w := 2*3) W2 b2 (Tensor3.flatten z1V)) := by
    rw [c2flat_eq]; exact diffG2r_c2
  rw [pdiv_comp _ _ _ hC hG]
  simp only [c2flat_eq]
  simp only [S2r]
  rw [← conv2wit.correct m2V {fm2(k1, '2*(2*3)*(2*3)')}]
  show conv2d_input_grad_formula W2 (Tensor3.unflatten m2V)
      {fm2(ci, 2)} {fm2(hi, 6)} {fm2(wi, 6)} = t1V {fm2(k1, '2*(2*3)*(2*3)')}
  rw [show (Tensor3.unflatten m2V : Tensor3 2 6 6) = M2T from Tensor3.unflatten_flatten M2T]
  simp [conv2d_input_grad_formula, W2, M2T, t1V, Fin.sum_univ_succ]
  try norm_num
""")

s1_agg_bullets = "\n".join(f"  · exact S1_c{k}" for k in range(72))

# S2r / S0r final-eval bullets (mask folds)
def mask_fold_bullets(preV, maskT, inV, n72='2*(2*3)*(2*3)'):
    out = []
    for m in range(72):
        ci, hi, wi = m // 36, (m % 36) // 6, m % 6
        out.append(f"""  · show (if {preV} {fm2(ci,2)} {fm2(hi,6)} {fm2(wi,6)} > 0 then (1:ℝ) else 0) * {inV} {fm2(m, n72)}
        = {maskT} {fm2(ci,2)} {fm2(hi,6)} {fm2(wi,6)}
    norm_num [{preV}, {inV}, {maskT}]""")
    return "\n".join(out)

s2r_bullets = mask_fold_bullets("c2V", "M2T", "t2V")
s0r_bullets = mask_fold_bullets("c1V", "M1T", "t1V")

body = f'''import LeanMlir.Proofs.Training.TrainedCnnWitness
import LeanMlir.Proofs.Training.JacobianSeal

/-! # Level-3 seal for the trained-CNN witness

The MLP rung's `pdiv_fwd`/`trainedMlp_backward_nontrivial` program at the
CONVOLUTIONAL witness: one whole-net Jacobian entry of the trained CNN,
computed in closed form by peeling `mnistCnnNoBnForward` with `pdiv_comp`
from the output side. Exact backward-cotangent tables (all in-kernel
rationals): dense head slices (`t4V`/`t3V`), the max-pool argmax routing
(`t2V`, via `MaxPool2IsArgmax` at each of the 72 positions), the ReLU mask
folds (`m2V`/`m1V`), and the conv input-VJPs (`t1V` and the final entry,
via `conv2d_input_grad_formula` through `HasVJPAt.correct`).

The sealed entry: `∂ logit_{JC} / ∂ pixel ({HI},{WI})` at the witness =
`{PD.numerator}/{PD.denominator}` ≈ {float(PD):.4f} ≠ 0, hence
`trainedCnn_backward_nontrivial` (the proven backward is not the zero map),
`trainedCnn_jacobian_nonzero` (`fderiv ≠ 0`), and `trainedCnn_not_constant`.
Generated by `scripts/trained_cnn_seal.py`; tables are DATA here. -/

namespace Proofs
namespace TrainedCnn

open Classical

-- ════════════════════════════════════════════════════════════════
-- § Backward-cotangent tables (exact, class {JC})
-- ════════════════════════════════════════════════════════════════

/-- dense-head slice `pdiv (dense5 ∘ relu∘dense4) r3V · {JC}`. -/
noncomputable def t4V : Vec 8 := {vec_lit(t4)}

/-- + relu∘dense3: `pdiv (head) p2f · {JC}`. -/
noncomputable def t3V : Vec (2 * 3 * 3) := {vec_lit(t3)}

/-- + max-pool: argmax routing of `t3V` (zero off-argmax). -/
noncomputable def t2V : Vec (2 * (2*3) * (2*3)) := {vec_lit(t2.reshape(-1))}

/-- relu mask₂ ⊙ t2V, tensor form (the conv2 backward cotangent). -/
noncomputable def M2T : Tensor3 2 6 6 :=
  {t3lit(M2)}

noncomputable def m2V : Vec (2 * (2*3) * (2*3)) := Tensor3.flatten M2T

/-- conv2 input-VJP of `m2V`. -/
noncomputable def t1V : Vec (2 * (2*3) * (2*3)) := {vec_lit(t1.reshape(-1))}

/-- relu mask₁ ⊙ t1V, tensor form (the conv1 backward cotangent). -/
noncomputable def M1T : Tensor3 2 6 6 :=
  {t3lit(M1)}

noncomputable def m1V : Vec (2 * (2*3) * (2*3)) := Tensor3.flatten M1T

-- ════════════════════════════════════════════════════════════════
-- § Shared differentiability facts
-- ════════════════════════════════════════════════════════════════

theorem diffL4_r3V : DifferentiableAt ℝ (relu 8 ∘ dense W4 b4) r3V :=
  (relu_differentiableAt_of_smooth 8 _ d4_ne).comp r3V ((dense_differentiable W4 b4) r3V)

theorem diffG4_r3V : DifferentiableAt ℝ ({G4}) r3V :=
  ((dense_differentiable W5 b5) _).comp r3V diffL4_r3V

theorem diffL3_p2f : DifferentiableAt ℝ (relu 8 ∘ dense W3 b3) p2f :=
  (relu_differentiableAt_of_smooth 8 _ d3_ne).comp p2f ((dense_differentiable W3 b3) p2f)

theorem r3_pt : (relu 8 ∘ dense W3 b3) p2f = r3V := r3_eq

theorem diffG3_p2f : DifferentiableAt ℝ ({G3}) p2f := by
  refine DifferentiableAt.comp p2f ?_ diffL3_p2f
  rw [r3_pt]; exact diffG4_r3V

/-- relu(conv2 table) = the max-pool input table, pointwise fold. -/
theorem r2flat_eq :
    {R72} (Tensor3.flatten c2V) = Tensor3.flatten r2V := by
  rw [relu_flatten]
  congr 1
  funext o hi wi
  exact congrFun (congrFun (congrFun r2_eq o) hi) wi

theorem diffP_r2 : DifferentiableAt ℝ (maxPoolFlat 2 3 3) (Tensor3.flatten r2V) :=
  maxPoolFlat_differentiableAt (c := 2) (h := 3) (w := 3) r2V r2_smooth
    (by norm_num) (by norm_num) (by norm_num)

theorem diffG2_r2 : DifferentiableAt ℝ ({G2}) (Tensor3.flatten r2V) := by
  refine DifferentiableAt.comp _ ?_ diffP_r2
  rw [pooled_eq]; exact diffG3_p2f

theorem diffG2r_c2 : DifferentiableAt ℝ ({G2r}) (Tensor3.flatten c2V) := by
  refine DifferentiableAt.comp _ ?_
    (relu_differentiableAt_of_smooth _ _ (fun k => flatten_ne_zero c2_ne k))
  rw [r2flat_eq]; exact diffG2_r2

/-- conv2 at the relu(conv1) table = the conv2 table, flat form. -/
theorem c2flat_eq :
    flatConv (h := 2*3) (w := 2*3) W2 b2 (Tensor3.flatten z1V) = Tensor3.flatten c2V := by
  show Tensor3.flatten (conv2d W2 b2 (Tensor3.unflatten (Tensor3.flatten z1V))) = _
  rw [Tensor3.unflatten_flatten]
  congr 1
  funext o hi wi
  exact conv2_eq o hi wi

theorem diffG1_z1 : DifferentiableAt ℝ ({G1}) (Tensor3.flatten z1V) := by
  refine DifferentiableAt.comp _ ?_ ((flatConv_differentiable W2 b2) _)
  rw [c2flat_eq]; exact diffG2r_c2

/-- relu(conv1 table) = the conv2-input table, pointwise fold. -/
theorem z1flat_eq :
    {R72} (Tensor3.flatten c1V) = Tensor3.flatten z1V := by
  rw [relu_flatten]
  congr 1
  funext o hi wi
  exact congrFun (congrFun (congrFun z1_eq o) hi) wi

theorem diffG1r_c1 : DifferentiableAt ℝ ({G1r}) (Tensor3.flatten c1V) := by
  refine DifferentiableAt.comp _ ?_
    (relu_differentiableAt_of_smooth _ _ (fun k => flatten_ne_zero c1_ne k))
  rw [z1flat_eq]; exact diffG1_z1

/-- conv1 at the witness input = the conv1 table, flat form. -/
theorem c1flat_eq :
    flatConv (h := 2*3) (w := 2*3) W1 b1 X = Tensor3.flatten c1V := by
  show Tensor3.flatten (conv2d W1 b1 (Tensor3.unflatten (Tensor3.flatten T0))) = _
  rw [Tensor3.unflatten_flatten]
  congr 1
  funext o hi wi
  exact conv1_eq o hi wi

/-- The conv2 layer witness, type-ascribed at `flatConv` (defeq). -/
noncomputable def conv2wit :
    HasVJPAt (flatConv (h := 2*3) (w := 2*3) W2 b2) (Tensor3.flatten z1V) :=
  (hasVJP3_to_hasVJP (conv2d_has_vjp3 (h := 2*3) (w := 2*3) W2 b2)).toHasVJPAt
    (Tensor3.flatten z1V)

/-- The conv1 layer witness, type-ascribed at `flatConv` (defeq). -/
noncomputable def conv1wit :
    HasVJPAt (flatConv (h := 2*3) (w := 2*3) W1 b1) X :=
  (hasVJP3_to_hasVJP (conv2d_has_vjp3 (h := 2*3) (w := 2*3) W1 b1)).toHasVJPAt X

-- ════════════════════════════════════════════════════════════════
-- § S4/S3: the dense head slices
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem d4inner : ∀ k m : Fin 8,
    pdiv (relu 8 ∘ dense W4 b4) r3V k m
      = W4 k m * (if d4V m > 0 then (1:ℝ) else 0) := by
  intro k m
  have hd : DifferentiableAt ℝ (dense W4 b4) r3V := (dense_differentiable W4 b4) r3V
  have hr : DifferentiableAt ℝ (relu 8) (dense W4 b4 r3V) :=
    relu_differentiableAt_of_smooth 8 _ d4_ne
  rw [pdiv_comp _ _ _ hd hr]
  have hterm : ∀ l : Fin 8,
      pdiv (dense W4 b4) r3V k l * pdiv (relu 8) (dense W4 b4 r3V) l m
        = if l = m then W4 k l * (if d4V l > 0 then (1:ℝ) else 0) else 0 := by
    intro l
    rw [pdiv_dense, pdiv_relu 8 _ d4_ne, d4_eq]
    by_cases hlm : l = m
    · rw [if_pos hlm, if_pos hlm]
    · rw [if_neg hlm, if_neg hlm, mul_zero]
  rw [Finset.sum_congr rfl (fun l _ => hterm l),
      Finset.sum_ite_eq' Finset.univ m
        (fun l => W4 k l * (if d4V l > 0 then (1:ℝ) else 0))]
  simp

set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem S4 : ∀ k : Fin 8,
    pdiv ({G4}) r3V k {JCL} = t4V k := by
  intro k
  rw [pdiv_comp _ _ _ diffL4_r3V ((dense_differentiable W5 b5) _)]
  simp only [d4inner, pdiv_dense]
  fin_cases k <;> (simp [W4, W5, d4V, t4V, Fin.sum_univ_succ]; try norm_num)

set_option maxRecDepth 16384 in
set_option maxHeartbeats 8000000 in
theorem d3inner : ∀ (k : Fin (2*3*3)) (m : Fin 8),
    pdiv (relu 8 ∘ dense W3 b3) p2f k m
      = W3 k m * (if d3V m > 0 then (1:ℝ) else 0) := by
  intro k m
  have hd : DifferentiableAt ℝ (dense W3 b3) p2f := (dense_differentiable W3 b3) p2f
  have hr : DifferentiableAt ℝ (relu 8) (dense W3 b3 p2f) :=
    relu_differentiableAt_of_smooth 8 _ d3_ne
  rw [pdiv_comp _ _ _ hd hr]
  have hterm : ∀ l : Fin 8,
      pdiv (dense W3 b3) p2f k l * pdiv (relu 8) (dense W3 b3 p2f) l m
        = if l = m then W3 k l * (if d3V l > 0 then (1:ℝ) else 0) else 0 := by
    intro l
    rw [pdiv_dense, pdiv_relu 8 _ d3_ne, d3_eq]
    by_cases hlm : l = m
    · rw [if_pos hlm, if_pos hlm]
    · rw [if_neg hlm, if_neg hlm, mul_zero]
  rw [Finset.sum_congr rfl (fun l _ => hterm l),
      Finset.sum_ite_eq' Finset.univ m
        (fun l => W3 k l * (if d3V l > 0 then (1:ℝ) else 0))]
  simp

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
theorem S3 : ∀ k : Fin (2*3*3),
    pdiv ({G3}) p2f k {JCL} = t3V k := by
  intro k
  have hout : DifferentiableAt ℝ ({G4}) ((relu 8 ∘ dense W3 b3) p2f) := by
    rw [r3_pt]; exact diffG4_r3V
  rw [pdiv_comp _ _ _ diffL3_p2f hout]
  simp only [r3_pt]
  simp only [d3inner, S4]
  fin_cases k <;> (simp [W3, d3V, t3V, t4V, Fin.sum_univ_succ]; try norm_num)

-- ════════════════════════════════════════════════════════════════
-- § S2: the max-pool stage (argmax routing), per position
-- ════════════════════════════════════════════════════════════════

{"".join(s2_cases)}
theorem S2 : ∀ k : Fin (2 * (2*3) * (2*3)),
    pdiv ({G2}) (Tensor3.flatten r2V) k {JCL} = t2V k := by
  intro k
  fin_cases k
{s2_agg_bullets}

-- ════════════════════════════════════════════════════════════════
-- § S2r: the conv2-side ReLU mask fold
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
theorem S2r : ∀ m : Fin (2 * (2*3) * (2*3)),
    pdiv ({G2r}) (Tensor3.flatten c2V) m {JCL} = m2V m := by
  intro m
  have hr : DifferentiableAt ℝ ({R72}) (Tensor3.flatten c2V) :=
    relu_differentiableAt_of_smooth _ _ (fun k => flatten_ne_zero c2_ne k)
  have hG : DifferentiableAt ℝ ({G2}) ({R72} (Tensor3.flatten c2V)) := by
    rw [r2flat_eq]; exact diffG2_r2
  rw [pdiv_comp _ _ _ hr hG]
  simp only [r2flat_eq]
  simp only [S2]
  rw [Finset.sum_congr rfl (fun k2 _ => by
    rw [pdiv_relu _ _ (fun k => flatten_ne_zero c2_ne k), ite_mul, zero_mul]),
    Finset.sum_ite_eq Finset.univ m
      (fun k2 => (if Tensor3.flatten c2V m > 0 then (1:ℝ) else 0) * t2V k2)]
  simp only [Finset.mem_univ, if_true]
  fin_cases m
{s2r_bullets}

-- ════════════════════════════════════════════════════════════════
-- § S1: the conv2 input-VJP stage, per position
-- ════════════════════════════════════════════════════════════════

{"".join(s1_cases)}
theorem S1 : ∀ k : Fin (2 * (2*3) * (2*3)),
    pdiv ({G1}) (Tensor3.flatten z1V) k {JCL} = t1V k := by
  intro k
  fin_cases k
{s1_agg_bullets}

-- ════════════════════════════════════════════════════════════════
-- § S0r: the conv1-side ReLU mask fold
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
theorem S0r : ∀ m : Fin (2 * (2*3) * (2*3)),
    pdiv ({G1r}) (Tensor3.flatten c1V) m {JCL} = m1V m := by
  intro m
  have hr : DifferentiableAt ℝ ({R72}) (Tensor3.flatten c1V) :=
    relu_differentiableAt_of_smooth _ _ (fun k => flatten_ne_zero c1_ne k)
  have hG : DifferentiableAt ℝ ({G1}) ({R72} (Tensor3.flatten c1V)) := by
    rw [z1flat_eq]; exact diffG1_z1
  rw [pdiv_comp _ _ _ hr hG]
  simp only [z1flat_eq]
  simp only [S1]
  rw [Finset.sum_congr rfl (fun k1 _ => by
    rw [pdiv_relu _ _ (fun k => flatten_ne_zero c1_ne k), ite_mul, zero_mul]),
    Finset.sum_ite_eq Finset.univ m
      (fun k1 => (if Tensor3.flatten c1V m > 0 then (1:ℝ) else 0) * t1V k1)]
  simp only [Finset.mem_univ, if_true]
  fin_cases m
{s0r_bullets}

-- ════════════════════════════════════════════════════════════════
-- § The sealed Jacobian entry
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 16384 in
set_option maxHeartbeats 16000000 in
/-- **The whole-net Jacobian entry, exactly**: `∂ logit_{JC} / ∂ pixel ({HI},{WI})`
    at the trained weights and the real witness input. -/
theorem pdiv_fwd_entry :
    pdiv (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X
      {fm2(IC, '1 * (2*3) * (2*3)')} {JCL} = {lit(PD)} := by
  have hassoc : pdiv (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X
        {fm2(IC, '1 * (2*3) * (2*3)')} {JCL}
      = pdiv (({G1r}) ∘ flatConv (h := 2*3) (w := 2*3) W1 b1) X
        {fm2(IC, '1 * (2*3) * (2*3)')} {JCL} := rfl
  rw [hassoc]
  have hC : DifferentiableAt ℝ (flatConv (h := 2*3) (w := 2*3) W1 b1) X :=
    (flatConv_differentiable W1 b1) X
  have hG : DifferentiableAt ℝ ({G1r})
      (flatConv (h := 2*3) (w := 2*3) W1 b1 X) := by
    rw [c1flat_eq]; exact diffG1r_c1
  rw [pdiv_comp _ _ _ hC hG]
  simp only [c1flat_eq]
  simp only [S0r]
  rw [← conv1wit.correct m1V {fm2(IC, '1 * (2*3) * (2*3)')}]
  show conv2d_input_grad_formula W1 (Tensor3.unflatten m1V)
      (⟨0, by norm_num⟩ : Fin 1) {fm2(HI, 6)} {fm2(WI, 6)} = {lit(PD)}
  rw [show (Tensor3.unflatten m1V : Tensor3 2 6 6) = M1T from Tensor3.unflatten_flatten M1T]
  simp [conv2d_input_grad_formula, W1, M1T, Fin.sum_univ_succ]
  try norm_num

theorem pdiv_fwd_entry_ne :
    pdiv (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X
      {fm2(IC, '1 * (2*3) * (2*3)')} {JCL} ≠ 0 := by
  rw [pdiv_fwd_entry]; norm_num

/-- **Level 3: the trained-weight CNN backward is not the zero map** —
    the seal the MLP rung carries, now at the convolutional witness. -/
theorem trainedCnn_backward_nontrivial :
    trainedCnn_has_vjp_at.backward (basisVec {JCL}) {fm2(IC, '1 * (2*3) * (2*3)')} ≠ 0 :=
  trainedCnn_has_vjp_at.backward_ne_zero_of_pdiv_ne pdiv_fwd_entry_ne

/-- The `fderiv` form: the whole-net Jacobian at the trained CNN witness is nonzero. -/
theorem trainedCnn_jacobian_nonzero :
    fderiv ℝ (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X ≠ 0 := by
  intro h
  apply pdiv_fwd_entry_ne
  unfold pdiv
  rw [h]
  rfl

/-- The trained CNN is not a constant function. -/
theorem trainedCnn_not_constant :
    ¬ (∀ u v : Vec (1 * (2*3) * (2*3)),
        mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5 u
          = mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5 v) := by
  intro h
  apply trainedCnn_jacobian_nonzero
  have hc : mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5
      = fun _ => mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5 X :=
    funext fun u => h u X
  rw [hc]
  exact (hasFDerivAt_const _ X).fderiv

end TrainedCnn
end Proofs
'''

with open(SEAL_OUT, "w") as f:
    f.write(body)
print(f"wrote {SEAL_OUT}", flush=True)
