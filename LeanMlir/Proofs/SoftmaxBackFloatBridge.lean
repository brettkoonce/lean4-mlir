import LeanMlir.Proofs.BnBackFloatBridge
import LeanMlir.Proofs.FloatComposeBridge

/-! # ℝ→Float32 bridge for the SOFTMAX-JACOBIAN backward (A3 §1f — the vit crux)

A3 (planning/a3_backward_deepnet_assembly.md §1f): the attention backward's genuinely-new op — the
**row-coupled softmax Jacobian** `J = diag(p) − p·pᵀ`. The certified per-row softmax VJP
(`softmax_has_vjp`, `Attention.lean`) is

  `softmaxBack p dy i = p_i · (dy_i − ⟨p, dy⟩)`,   `⟨p, dy⟩ = Σ_j p_j · dy_j`

— `diag(p)·dy` minus the rank-1 `p·(pᵀdy)`. Unlike the diagonal activation backs (`diagBack`), this
couples a whole row (the `⟨p, dy⟩` reduction), so its float closeness threads `mul_close`/`reduction_close`/
`sub_close'` exactly like the BatchNorm input-gradient (`bnGradInput_close`): the float softmax weights
`fp` are supplied within `ep` of the real `p` (= `smErr`, the softmax transcendental budget, since the
weights carry `exp`), and the map is **linear in the cotangent `dy`**, so — like BN-back — its `FloatClose`
modulus IS the magnitude bound at `Cdy := e`.

`floatClose_softmaxBack`/`floatBridges_softmaxBack` deliver the row VJP as a composable bridge. This is
the heart of the scaled-dot-product-attention backward (`sdpa = softmax(QKᵀ/√d)·V`): the Mat-space
assembly chains this row map (per query row) with the V-matmul and score-matmul backwards (each a rounded
dot, `dot_close`), the forward `sdpa_close`'s peer. Kept general in `P` (`|p| ≤ P`) so it instantiates
for softmax (`P = 1`) and any other row-stochastic weighting.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The softmax-Jacobian VJP (real + float), per row
-- ════════════════════════════════════════════════════════════════

/-- **The row softmax-Jacobian backward** — the certified `softmax_has_vjp` row VJP at saved weights
    `p = softmax(z)`: `dy ↦ pᵢ·(dyᵢ − ⟨p, dy⟩)`. `diag(p)·dy − p·(pᵀdy)` (the rank-1 coupling). -/
noncomputable def softmaxBack {c : Nat} (p : Vec c) (dy : Vec c) : Vec c :=
  fun i => p i * (dy i - ∑ j : Fin c, p j * dy j)

/-- **The float row softmax-Jacobian backward** at a supplied float weight `fp` (within `ep` of the
    real `p` — the `smErr` transcendental budget): `dy ↦ fl(fpᵢ ⊗ (dyᵢ ⊖ fl(Σ fpⱼ⊗dyⱼ)))`. Mirrors
    `softmaxBack` op-for-op in `M`-rounded arithmetic. -/
noncomputable def FloatModel.softmaxBackF {c : Nat} (M : FloatModel) (fp : Vec c) (dy : Vec c) :
    Vec c :=
  fun i => M.mul (fp i) (M.sub (dy i) (M.sum (fun j => M.mul (fp j) (dy j))))

/-- The assembled per-entry rounding budget (the nested `mulErr`/reduction bound the proof yields). -/
noncomputable def FloatModel.softmaxBackBudget (M : FloatModel) (c : Nat) (P A ep : ℝ) : ℝ :=
  let u := M.u
  let γc := (1 + u) ^ (c + 1) - 1
  let eq := FloatModel.mulErr u P A ep 0
  let eS := γc * ((c : ℝ) * (P * A + eq)) + (c : ℝ) * eq
  let MSf := (c : ℝ) * (P * A) + eS
  let eSub := u * (A + MSf) + eS
  FloatModel.mulErr u P (A + (c : ℝ) * (P * A)) ep eSub

-- ════════════════════════════════════════════════════════════════
-- § Real magnitude + Lipschitz (the map is linear in `dy`)
-- ════════════════════════════════════════════════════════════════

/-- Real inner product magnitude: `|⟨p, dy⟩| ≤ c·P·A`. -/
private theorem softmaxBack_inner_abs_le {c : Nat} (p dy : Vec c) {P A : ℝ} (hP0 : 0 ≤ P)
    (hp : ∀ j, |p j| ≤ P) (hdy : ∀ j, |dy j| ≤ A) :
    |∑ j : Fin c, p j * dy j| ≤ (c : ℝ) * (P * A) := by
  calc |∑ j : Fin c, p j * dy j| ≤ ∑ j, |p j * dy j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _j : Fin c, (P * A) := Finset.sum_le_sum fun j _ => by
        rw [abs_mul]; exact mul_le_mul (hp j) (hdy j) (abs_nonneg _) hP0
    _ = (c : ℝ) * (P * A) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **Real softmax-back magnitude:** `|softmaxBack p dy i| ≤ P·(A + c·P·A)`. -/
theorem softmaxBack_abs_le {c : Nat} (p dy : Vec c) {P A : ℝ} (hP0 : 0 ≤ P)
    (hp : ∀ j, |p j| ≤ P) (hdy : ∀ j, |dy j| ≤ A) (i : Fin c) :
    |softmaxBack p dy i| ≤ P * (A + (c : ℝ) * (P * A)) := by
  show |p i * (dy i - ∑ j, p j * dy j)| ≤ _
  rw [abs_mul]
  have hsub : |dy i - ∑ j, p j * dy j| ≤ A + (c : ℝ) * (P * A) :=
    (abs_sub _ _).trans (add_le_add (hdy i) (softmaxBack_inner_abs_le p dy hP0 hp hdy))
  exact mul_le_mul (hp i) hsub (abs_nonneg _) hP0

/-- **Real softmax-back is linear in `dy`** ⇒ 1-row-Lipschitz: the difference at two cotangents is
    `softmaxBack` of their difference, so it inherits the magnitude bound at `Cdy := e`. -/
theorem softmaxBack_sub_abs_le {c : Nat} (p dyt dya : Vec c) {P e : ℝ} (hP0 : 0 ≤ P)
    (hp : ∀ j, |p j| ≤ P) (hd : ∀ j, |dyt j - dya j| ≤ e) (i : Fin c) :
    |softmaxBack p dyt i - softmaxBack p dya i| ≤ P * (e + (c : ℝ) * (P * e)) := by
  have hSt : (∑ j, p j * dyt j) - (∑ j, p j * dya j) = ∑ j, p j * (dyt j - dya j) := by
    rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl (fun j _ => by ring)
  have hrw : softmaxBack p dyt i - softmaxBack p dya i
      = p i * ((dyt i - dya i) - ∑ j, p j * (dyt j - dya j)) := by
    show p i * (dyt i - ∑ j, p j * dyt j) - p i * (dya i - ∑ j, p j * dya j) = _
    rw [← hSt]; ring
  rw [hrw, abs_mul]
  have hinner : |∑ j, p j * (dyt j - dya j)| ≤ (c : ℝ) * (P * e) :=
    softmaxBack_inner_abs_le p (fun j => dyt j - dya j) hP0 hp hd
  have hsub : |(dyt i - dya i) - ∑ j, p j * (dyt j - dya j)| ≤ e + (c : ℝ) * (P * e) :=
    (abs_sub _ _).trans (add_le_add (hd i) hinner)
  exact mul_le_mul (hp i) hsub (abs_nonneg _) hP0

-- ════════════════════════════════════════════════════════════════
-- § The float rounding budget (threads mul/sum/sub)
-- ════════════════════════════════════════════════════════════════

/-- **Softmax-back float closeness (per entry).** The deployed float row VJP is within
    `softmaxBackBudget` of the certified real one, with the float weights `fp` supplied within `ep`
    of `p` (discharged at instantiation by the forward softmax `smErr`). Threads the per-term
    `mul_close`, the `⟨p, dy⟩` `reduction_close`, the `sub_close'`, and the final `mul_close` — the
    softmax-row peer of `bnGradInput_close`. -/
theorem softmaxBack_close {c : Nat} (M : FloatModel) (p fp dy : Vec c) {P A ep : ℝ}
    (hP0 : 0 ≤ P)
    (hp : ∀ j, |p j| ≤ P) (hfp : ∀ j, |fp j - p j| ≤ ep) (hdy : ∀ j, |dy j| ≤ A) (i : Fin c) :
    |M.softmaxBackF fp dy i - softmaxBack p dy i| ≤ M.softmaxBackBudget c P A ep := by
  have hu := M.u_nonneg
  -- step 1: per-term products
  set eq := FloatModel.mulErr M.u P A ep 0 with heq
  have hterm : ∀ j, |M.mul (fp j) (dy j) - p j * dy j| ≤ eq := fun j =>
    M.mul_close (hfp j) (by simp) (hp j) (hdy j)
  have htermr : ∀ j, |p j * dy j| ≤ P * A := fun j => by
    rw [abs_mul]; exact mul_le_mul (hp j) (hdy j) (abs_nonneg _) hP0
  -- step 2: the reduction ⟨p, dy⟩
  set eS := ((1 + M.u) ^ (c + 1) - 1) * ((c : ℝ) * (P * A + eq)) + (c : ℝ) * eq with heS
  have hsum : |M.sum (fun j => M.mul (fp j) (dy j)) - ∑ j, p j * dy j| ≤ eS :=
    reduction_close M _ _ hterm htermr
  have hsumr_mag : |∑ j, p j * dy j| ≤ (c : ℝ) * (P * A) :=
    softmaxBack_inner_abs_le p dy hP0 hp hdy
  have hsumf_mag : |M.sum (fun j => M.mul (fp j) (dy j))| ≤ (c : ℝ) * (P * A) + eS := by
    have htri := abs_sub_le (M.sum (fun j => M.mul (fp j) (dy j))) (∑ j, p j * dy j) 0
    simp only [sub_zero] at htri
    calc |M.sum (fun j => M.mul (fp j) (dy j))|
        ≤ |M.sum (fun j => M.mul (fp j) (dy j)) - ∑ j, p j * dy j| + |∑ j, p j * dy j| := htri
      _ ≤ eS + (c : ℝ) * (P * A) := add_le_add hsum hsumr_mag
      _ = (c : ℝ) * (P * A) + eS := by ring
  -- step 3: the subtraction dyᵢ ⊖ ⟨p, dy⟩
  set MSf := (c : ℝ) * (P * A) + eS with hMSf
  have hsub0 : |M.sub (dy i) (M.sum (fun j => M.mul (fp j) (dy j))) - (dy i - ∑ j, p j * dy j)|
      ≤ M.u * (A + MSf) + (0 + eS) := sub_close' M (by simp) hsum (hdy i) hsumf_mag
  rw [zero_add] at hsub0
  have hsub_mag : |M.sub (dy i) (M.sum (fun j => M.mul (fp j) (dy j)))| ≤ (1 + M.u) * (A + MSf) :=
    sub_mag M (hdy i) hsumf_mag
  have hsubr_mag : |dy i - ∑ j, p j * dy j| ≤ A + (c : ℝ) * (P * A) :=
    (abs_sub _ _).trans (add_le_add (hdy i) hsumr_mag)
  -- step 4: the final product fpᵢ ⊗ (·)
  set eSub := M.u * (A + MSf) + eS with heSub
  have hfinal : |M.mul (fp i) (M.sub (dy i) (M.sum (fun j => M.mul (fp j) (dy j))))
      - p i * (dy i - ∑ j, p j * dy j)| ≤ FloatModel.mulErr M.u P (A + (c : ℝ) * (P * A)) ep eSub :=
    M.mul_close (hfp i) hsub0 (hp i) hsubr_mag
  show |M.softmaxBackF fp dy i - softmaxBack p dy i| ≤ M.softmaxBackBudget c P A ep
  simp only [FloatModel.softmaxBackF, softmaxBack, FloatModel.softmaxBackBudget]
  exact hfinal

-- ════════════════════════════════════════════════════════════════
-- § The FloatClose / FloatBridges wrap
-- ════════════════════════════════════════════════════════════════

/-- **The row softmax-Jacobian backward is `FloatClose`** — output magnitude `ReMag(A) + budget`
    (`ReMag(A) = P·(A + c·P·A)`), modulus `e ↦ budget(A) + ReMag(e)` (the map is linear in `dy`, so —
    like `floatClose_bnBack` — the Lipschitz part is the real magnitude at `Cdy := e`). The float
    weights `fp` are within `ep` of `p` (the supplied softmax `smErr`). -/
theorem floatClose_softmaxBack {c : Nat} (M : FloatModel) (p fp : Vec c) {P ep : ℝ}
    (hP0 : 0 ≤ P) (hp : ∀ j, |p j| ≤ P) (hfp : ∀ j, |fp j - p j| ≤ ep) (A : ℝ) :
    FloatClose A (P * (A + (c : ℝ) * (P * A)) + M.softmaxBackBudget c P A ep)
      (softmaxBack p) (M.softmaxBackF fp)
      (fun e => M.softmaxBackBudget c P A ep + P * (e + (c : ℝ) * (P * e))) := by
  refine ⟨fun dy hdy i => ?_, fun dyt dya e hdya hdyt hd i => ?_⟩
  · have hreal := softmaxBack_abs_le p dy hP0 hp hdy i
    have hbudget := softmaxBack_close M p fp dy hP0 hp hfp hdy i
    have hB0 : 0 ≤ M.softmaxBackBudget c P A ep := (abs_nonneg _).trans hbudget
    refine ⟨hreal.trans (le_add_of_nonneg_right hB0), ?_⟩
    calc |M.softmaxBackF fp dy i|
        ≤ |M.softmaxBackF fp dy i - softmaxBack p dy i| + |softmaxBack p dy i| := by
          simpa using abs_sub_le (M.softmaxBackF fp dy i) (softmaxBack p dy i) 0
      _ ≤ M.softmaxBackBudget c P A ep + P * (A + (c : ℝ) * (P * A)) := add_le_add hbudget hreal
      _ = P * (A + (c : ℝ) * (P * A)) + M.softmaxBackBudget c P A ep := by ring
  · have hbudget := softmaxBack_close M p fp dyt hP0 hp hfp hdyt i
    have hlip := softmaxBack_sub_abs_le p dyt dya hP0 hp hd i
    calc |M.softmaxBackF fp dyt i - softmaxBack p dya i|
        ≤ |M.softmaxBackF fp dyt i - softmaxBack p dyt i|
          + |softmaxBack p dyt i - softmaxBack p dya i| := abs_sub_le _ _ _
      _ ≤ M.softmaxBackBudget c P A ep + P * (e + (c : ℝ) * (P * e)) := add_le_add hbudget hlip

/-- **The row softmax-Jacobian backward float-bridges.** The transformer attention backward's
    genuinely-new op (`diag(p) − p·pᵀ` applied to the cotangent); instantiate `p := softmax(scores)`,
    `P := 1`, `ep := smErr` (the forward softmax budget). The heart of `sdpaBack`. -/
theorem floatBridges_softmaxBack {c : Nat} (M : FloatModel) (p fp : Vec c) {P ep : ℝ}
    (hc : 0 < c) (hP0 : 0 ≤ P) (hp : ∀ j, |p j| ≤ P) (hfp : ∀ j, |fp j - p j| ≤ ep) :
    FloatBridges (softmaxBack p) := fun A hA =>
  ⟨_, _, _, (floatClose_softmaxBack M p fp hP0 hp hfp A).cod_nonneg hA hc,
    floatClose_softmaxBack M p fp hP0 hp hfp A⟩

end Proofs
