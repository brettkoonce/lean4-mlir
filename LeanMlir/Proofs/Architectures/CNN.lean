import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Architectures.Residual

/-!
# CNN VJP Proofs

VJP correctness for the convolutional and pooling layers used in
`historical/mlir_poc/hand_cnn_train_step.mlir`. The architecture there is:

    x(1,28,28) → Conv(1→32) → ReLU → Conv(32→32) → ReLU → MaxPool
              → Flatten → Dense(6272→512) → ReLU → Dense(512→512)
              → ReLU → Dense(512→10) → logits

The dense and ReLU layers are inherited from `MLP.lean`. This file
adds the new operations: conv2d, max-pool, and flatten.

## The big idea: conv backward IS conv

The most pedagogically valuable result is that the VJP of a convolution
is itself expressible as **convolutions** — with appropriate kernel
reversal and axis transposition. This is why conv layers train
efficiently: there's no special backward operator. The same primitive
runs in both directions.

Two specific tricks appear in the MLIR:

1. **Input-gradient via reversed kernel**: `dx = conv(dy, reverse(Wᵀ))`
2. **Weight-gradient via the transpose trick**: `dW = conv(xᵀ, dyᵀ)`,
   where the spatial dims of the gradient become the "kernel".

We bundle the VJP formulas as `HasVJP3` / `HasVJP` defs whose
`.correct` fields are proved (the proofs are standard matrix calculus
on cross-correlations), and the commentary explains why each formula
has the form it does.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Tensor types for CNN
-- ════════════════════════════════════════════════════════════════

-- Tensor3 is imported from Tensor.lean

/-- A conv kernel: out_channels × in_channels × kH × kW.
    This is the OIHW layout used by StableHLO and IREE. -/
abbrev Kernel4 (oc ic kh kw : Nat) :=
  Fin oc → Fin ic → Fin kh → Fin kw → ℝ

namespace Kernel4

/-! `Kernel4 oc ic kH kW` and `Vec (oc * ic * kH * kW)` are in bijection
    by row-major flattening — mirrors `Mat.flatten` / `Tensor3.flatten`.
    We need this so that the weight-gradient VJP can be stated as a plain
    `HasVJP` (Vec → Vec) on the flattened kernel, reusing the existing
    framework instead of introducing a parallel 4D machinery.

    Nat multiplication associates left, so `oc * ic * kH * kW` parses as
    `((oc * ic) * kH) * kW` — three nested `finProdFinEquiv` calls. -/

/-- Row-major flatten: `Kernel4 oc ic kH kW → Vec (oc * ic * kH * kW)`. -/
noncomputable def flatten {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : Vec (oc * ic * kH * kW) :=
  fun k =>
    let ockH_kW := finProdFinEquiv.symm k         -- : Fin (oc*ic*kH) × Fin kW
    let ocic_kH := finProdFinEquiv.symm ockH_kW.1 -- : Fin (oc*ic) × Fin kH
    let oc_ic   := finProdFinEquiv.symm ocic_kH.1 -- : Fin oc × Fin ic
    W oc_ic.1 oc_ic.2 ocic_kH.2 ockH_kW.2

/-- Row-major unflatten: inverse of `flatten`. -/
noncomputable def unflatten {oc ic kH kW : Nat}
    (v : Vec (oc * ic * kH * kW)) : Kernel4 oc ic kH kW :=
  fun o c kh kw =>
    v (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, c), kh), kw))

theorem unflatten_flatten {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : unflatten (flatten W) = W := by
  funext o c kh kw
  unfold unflatten flatten
  simp [Equiv.symm_apply_apply]

theorem flatten_unflatten {oc ic kH kW : Nat}
    (v : Vec (oc * ic * kH * kW)) : flatten (unflatten v) = v := by
  funext k; simp only [flatten, unflatten, Prod.mk.eta, Equiv.apply_symm_apply]

end Kernel4

-- ════════════════════════════════════════════════════════════════
-- § Conv2d
-- ════════════════════════════════════════════════════════════════

/-- **Conv2d forward** (SAME padding, stride 1).

    `y[o, h, w] = (Σ_{c, kh, kw} x[c, h+kh−p, w+kw−p] · W[o, c, kh, kw]) + b[o]`

    where `p = (kH−1)/2` is the padding offset and out-of-bounds reads
    return 0 (zero padding). The output spatial size equals the input.

    Note: this is technically *cross-correlation*, not convolution in the
    strict signal-processing sense. ML literature uses "convolution" loosely;
    the difference (kernel flipping) only matters when comparing against
    classical signal-processing references.

    MLIR (`hand_cnn_train_step.mlir`):
      %cv0 = "stablehlo.convolution"(%x, %W0) {
        padding = dense<[[1, 1], [1, 1]]>, ...
      }
      %h0pre = stablehlo.add %cv0, broadcast(%b0) -/
noncomputable def conv2d {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) : Tensor3 oc h w :=
  fun o hi wi =>
    b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      W o c kh kw *
        (let pH := (kH - 1) / 2
         let pW := (kW - 1) / 2
         let hh := kh.val + hi.val
         let ww := kw.val + wi.val
         if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
           x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
         else 0)

/-- **A pad-guarded read is differentiable.** The guard `P` does not depend on `x`, so
    `if hP : P then f hP x else 0` is one branch or the constant `0`. Tagged for `fun_prop`, which
    has no rule for a dependent `if`; every conv / depthwise pad-eval goes through it. -/
@[fun_prop]
theorem differentiable_dite_zero {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (P : Prop) [Decidable P] (f : P → E → ℝ) (hf : ∀ hP, Differentiable ℝ (f hP)) :
    Differentiable ℝ fun x => if hP : P then f hP x else 0 := by
  by_cases hP : P <;> simp only [hP, ↓reduceDIte] <;> fun_prop

/-- **Conv2d is differentiable everywhere.** Each output coordinate
    `conv2d W b x o hi wi` is the affine map
    `b o + ∑_{c,kh,kw} W o c kh kw · (pad-eval x)`: a constant bias plus a
    finite ℝ-linear combination of pad-guarded input reads
    (`differentiable_dite_zero`). -/
@[fun_prop]
theorem conv2d_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (conv2d W b : Tensor3 ic h w → Tensor3 oc h w) := by
  unfold conv2d; fun_prop

/-- **Differentiability of an `if hpad : P then v(σ hpad) else 0` term** — the pointwise,
    `Vec`-indexed form of `differentiable_dite_zero`. -/
@[fun_prop]
lemma differentiableAt_pad_eval {n : Nat} (P : Prop) [Decidable P]
    (σ : P → Fin n) (v : Vec n) :
    DifferentiableAt ℝ (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) v := by
  exact differentiable_dite_zero P _ (fun _ => by fun_prop) v

/-- **Closed-form input gradient for conv2d** — direct formula, written as
    a sum over output positions `(co, ho, wo)` with reconstructed kernel
    offsets `kh_nat = hi + pH − ho`, `kw_nat = wi + pW − wo`. The body is
    nonzero only when the reconstructed `(kh_nat, kw_nat)` lies in
    `[0, kH) × [0, kW)` — i.e., when the input position `(hi, wi)` is
    actually reachable from output `(ho, wo)` via some valid kernel offset.
    Equivalent (under the `(ho, wo) ↔ (kh, kw)` partial bijection
    `ho = hi+pH-kh`) to the MLIR-aligned "reversed-kernel" formula
    `dx[c, h, w] = Σ_{o, kh, kw} W[o, c, kH−1−kh, kW−1−kw] · dy[o, h+kh−p, w+kw−p]`. -/
noncomputable def conv2d_input_grad_formula {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (dy : Tensor3 oc h w) : Tensor3 ic h w :=
  fun ci hi wi =>
    ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
      let pH := (kH - 1) / 2
      let pW := (kW - 1) / 2
      let kh_nat := hi.val + pH - ho.val
      let kw_nat := wi.val + pW - wo.val
      if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧ wo.val ≤ wi.val + pW ∧ kw_nat < kW then
        W co ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy co ho wo
      else 0

/-- **Conv2d input-VJP** — proved from foundation rules.

    The function `v ↦ flatten (conv2d W b (unflatten v))` is affine in
    `v`: the bias-free conv (linear in `v`) plus the broadcast bias, so
    `pdiv_of_affine` reads each per-`(idx_in, idx_out)` entry off the
    bias-free conv of the basis vector — a sum over `(c, kh, kw)` of
    `W o(idx_out) c kh kw` times the pad-guarded Kronecker. Reindex `Fin (oc*h*w) ↔ Fin oc × Fin h × Fin w`
    on the sum-over-`idx_out`, then a triple `Finset.sum_eq_single` over
    `(c, kh, kw)` (matching `idx_in`'s decoded `(ci, hi, wi)`) gives the
    closed-form input gradient `conv2d_input_grad_formula`.

    The backward function (accessed as `(conv2d_has_vjp3 W b).backward`,
    or via the `conv2d_input_grad` abbrev below) implements
    `conv2d_input_grad_formula W dy ci hi wi` — a direct sum over
    `(co, kh, kw)` of `W co ci kh kw * dy co ho_nat wo_nat` for valid
    `(ho_nat, wo_nat)`. Equivalent (under `kh ↔ kH−1−kh`) to the
    MLIR-aligned reversed-kernel formula
    `dx[c, h, w] = Σ_{o, kh, kw} W[o, c, kH−1−kh, kW−1−kw] · dy[o, h+kh−p, w+kw−p]`.

    MLIR emits the reversed-kernel form directly:
      %W1_t   = stablehlo.transpose %W1, dims = [1, 0, 2, 3]   -- swap oc↔ic
      %W1_rev = stablehlo.reverse %W1_t, dims = [2, 3]         -- flip spatial
      %d_h0   = "stablehlo.convolution"(%d_h1pre, %W1_rev) ...
-/
noncomputable def conv2d_has_vjp3 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    HasVJP3 (conv2d W b : Tensor3 ic h w → Tensor3 oc h w) where
  backward := fun _x dy => conv2d_input_grad_formula W dy
  correct := by
    intro x dy ci hi wi
    -- Set abbreviation for idx_in (the input position we're computing grad at).
    set idx_in : Fin (ic * h * w) :=
      finProdFinEquiv (finProdFinEquiv (ci, hi), wi) with hidx_in
    -- Step 1: per-`(idx_in, idx_out)` pdiv formula. We keep the UN-collapsed
    -- form `∑ c kh kw, W * indicator` because the closing reindex collapses
    -- it into the natural `(co, kh, kw)` loop without needing a partial
    -- bijection between `Fin h` and `Fin kH`.
    have h_pdiv : ∀ idx_out : Fin (oc * h * w),
        pdiv (fun v' : Vec (ic * h * w) =>
                Tensor3.flatten (conv2d W b (Tensor3.unflatten v')))
              (Tensor3.flatten x) idx_in idx_out =
        ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1) c kh kw *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val +
               (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
             let ww := kw.val + (finProdFinEquiv.symm idx_out).2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               (if idx_in = finProdFinEquiv (finProdFinEquiv
                   (c, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then
                 (1 : ℝ) else 0)
             else 0) := by
      intro idx_out
      -- The conv is affine in its input: the bias-free conv plus the broadcast bias.
      have hsplit : (fun v' : Vec (ic * h * w) =>
            Tensor3.flatten (conv2d W b (Tensor3.unflatten v'))) =
          fun v => Tensor3.flatten (conv2d W 0 (Tensor3.unflatten v)) +
            (fun k => b (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1) := by
        funext v k
        simp only [Tensor3.flatten, conv2d, Pi.add_apply, Pi.zero_apply, zero_add]
        ring
      rw [hsplit, pdiv_of_affine]
      · simp only [Tensor3.flatten, conv2d, Tensor3.unflatten, Pi.zero_apply, zero_add,
          basisVec_apply, @eq_comm _ idx_in]
      · intro u v; funext k
        simp only [Tensor3.flatten, conv2d, Tensor3.unflatten, Pi.add_apply, Pi.zero_apply,
          zero_add, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun kh _ =>
          Finset.sum_congr rfl fun kw _ => ?_
        split_ifs <;> ring
      · intro a v; funext k
        simp only [Tensor3.flatten, conv2d, Tensor3.unflatten, Pi.smul_apply, Pi.zero_apply,
          zero_add, smul_eq_mul, Finset.mul_sum]
        refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun kh _ =>
          Finset.sum_congr rfl fun kw _ => ?_
        split_ifs <;> ring
    -- Step 2: substitute h_pdiv into the RHS sum and collapse.
    show conv2d_input_grad_formula W dy ci hi wi =
      ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        pdiv3 (conv2d W b) x ci hi wi co ho wo * dy co ho wo
    unfold conv2d_input_grad_formula pdiv3
    apply Finset.sum_congr rfl; intro co _
    apply Finset.sum_congr rfl; intro ho _
    apply Finset.sum_congr rfl; intro wo _
    -- For each (co, ho, wo), substitute h_pdiv at idx_out := flat(co, ho, wo).
    rw [h_pdiv (finProdFinEquiv (finProdFinEquiv (co, ho), wo))]
    -- Simplify the decoding (Equiv.symm_apply_apply ⊢ ohw_o = co, ohw_hi = ho, ohw_wi = wo).
    simp only [Equiv.symm_apply_apply]
    -- Pull `dy co ho wo` out of the formula's if-true branch.
    rw [show (let pH := (kH - 1) / 2
              let pW := (kW - 1) / 2
              let kh_nat := hi.val + pH - ho.val
              let kw_nat := wi.val + pW - wo.val
              if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                  wo.val ≤ wi.val + pW ∧ kw_nat < kW then
                W co ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy co ho wo
              else 0) =
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let kh_nat := hi.val + pH - ho.val
             let kw_nat := wi.val + pW - wo.val
             if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                 wo.val ≤ wi.val + pW ∧ kw_nat < kW then
               W co ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩
             else 0) * dy co ho wo from by
      by_cases hb : ho.val ≤ hi.val + (kH - 1) / 2 ∧
                     hi.val + (kH - 1) / 2 - ho.val < kH ∧
                     wo.val ≤ wi.val + (kW - 1) / 2 ∧
                     wi.val + (kW - 1) / 2 - wo.val < kW
      · simp only [dite_eq_left hb]
      · simp only [dite_eq_right hb, zero_mul]]
    -- Goal: (if hb : back_cond then W co ci ⟨kh*⟩ ⟨kw*⟩ else 0) * dy co ho wo
    --     = (∑ c kh kw, W co c kh kw * indicator) * dy co ho wo
    congr 1
    -- Convert the dependent-if indicator to a non-dependent conjunction-form.
    have h_indicator : ∀ (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
        ((let pH := (kH - 1) / 2
          let pW := (kW - 1) / 2
          let hh := kh.val + ho.val
          let ww := kw.val + wo.val
          if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
            (if idx_in = finProdFinEquiv (finProdFinEquiv
                (c, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then (1 : ℝ) else 0)
          else 0) : ℝ) =
        (if c = ci ∧ kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                     kw.val + wo.val = wi.val + (kW - 1) / 2 then (1 : ℝ) else 0) := by
      intro c kh kw
      by_cases hpad : (kH - 1) / 2 ≤ kh.val + ho.val ∧
                      kh.val + ho.val - (kH - 1) / 2 < h ∧
                      (kW - 1) / 2 ≤ kw.val + wo.val ∧
                      kw.val + wo.val - (kW - 1) / 2 < w
      · rw [dite_eq_left hpad]
        by_cases h_match : c = ci ∧ kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                                    kw.val + wo.val = wi.val + (kW - 1) / 2
        · -- Build the explicit Fin equality for the indicator's RHS.
          have h_idx_in_eq : idx_in = finProdFinEquiv (finProdFinEquiv
              (c, ⟨kh.val + ho.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kw.val + wo.val - (kW - 1) / 2, hpad.2.2.2⟩) := by
            rw [hidx_in]
            have h_c : c = ci := h_match.1
            have h_hi : (⟨kh.val + ho.val - (kH - 1) / 2, hpad.2.1⟩ : Fin h) = hi := by
              apply Fin.ext
              show kh.val + ho.val - (kH - 1) / 2 = hi.val
              omega
            have h_wi : (⟨kw.val + wo.val - (kW - 1) / 2, hpad.2.2.2⟩ : Fin w) = wi := by
              apply Fin.ext
              show kw.val + wo.val - (kW - 1) / 2 = wi.val
              omega
            rw [← h_c, ← h_hi, ← h_wi]
          rw [ite_eq_left h_idx_in_eq, ite_eq_left h_match]
        · rw [ite_eq_right h_match]
          rw [ite_eq_right]
          intro h_eq
          apply h_match
          rw [hidx_in] at h_eq
          have h_inj := finProdFinEquiv.injective h_eq
          have h_inj_pair := Prod.mk.inj h_inj
          have h_inj_inner := finProdFinEquiv.injective h_inj_pair.1
          have h_inj_inner_pair := Prod.mk.inj h_inj_inner
          refine ⟨h_inj_inner_pair.1.symm, ?_, ?_⟩
          · have h_hi : hi.val = kh.val + ho.val - (kH - 1) / 2 :=
              Fin.ext_iff.mp h_inj_inner_pair.2
            omega
          · have h_wi : wi.val = kw.val + wo.val - (kW - 1) / 2 :=
              Fin.ext_iff.mp h_inj_pair.2
            omega
      · rw [dite_eq_right hpad]
        rw [ite_eq_right]
        intro ⟨_, hkh_eq, hkw_eq⟩
        apply hpad
        refine ⟨?_, ?_, ?_, ?_⟩
        · rw [hkh_eq]; exact Nat.le_add_left _ _
        · rw [hkh_eq, Nat.add_sub_cancel]; exact hi.isLt
        · rw [hkw_eq]; exact Nat.le_add_left _ _
        · rw [hkw_eq, Nat.add_sub_cancel]; exact wi.isLt
    simp_rw [h_indicator]
    -- Goal: (if hb : back_cond then W co ci ⟨kh*⟩ ⟨kw*⟩ else 0)
    --     = ∑ c kh kw, W co c kh kw * (if c = ci ∧ ... then 1 else 0)
    by_cases hb : ho.val ≤ hi.val + (kH - 1) / 2 ∧
                   hi.val + (kH - 1) / 2 - ho.val < kH ∧
                   wo.val ≤ wi.val + (kW - 1) / 2 ∧
                   wi.val + (kW - 1) / 2 - wo.val < kW
    · rw [dite_eq_left hb]
      -- Σ c collapses on c = ci, then Σ kh on kh = ⟨hi+pH-ho, hb.2.1⟩, then Σ kw similarly.
      symm
      rw [Finset.sum_eq_single ci ?_ ?_]
      rw [Finset.sum_eq_single ⟨hi.val + (kH - 1) / 2 - ho.val, hb.2.1⟩ ?_ ?_]
      rw [Finset.sum_eq_single ⟨wi.val + (kW - 1) / 2 - wo.val, hb.2.2.2⟩ ?_ ?_]
      · -- Main: W co ci ⟨kh*⟩ ⟨kw*⟩ * (if ci=ci ∧ ... then 1 else 0) = W co ci ⟨kh*⟩ ⟨kw*⟩
        rw [ite_eq_left]
        · ring
        refine ⟨rfl, ?_, ?_⟩
        · show hi.val + (kH - 1) / 2 - ho.val + ho.val = hi.val + (kH - 1) / 2
          omega
        · show wi.val + (kW - 1) / 2 - wo.val + wo.val = wi.val + (kW - 1) / 2
          omega
      · intro kw _ hkw_ne
        rw [ite_eq_right ?_]; · ring
        intro ⟨_, _, hkw_eq⟩
        apply hkw_ne
        apply Fin.ext
        show kw.val = wi.val + (kW - 1) / 2 - wo.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
      · intro kh _ hkh_ne
        apply Finset.sum_eq_zero; intro kw _
        rw [ite_eq_right ?_]; · ring
        intro ⟨_, hkh_eq, _⟩
        apply hkh_ne
        apply Fin.ext
        show kh.val = hi.val + (kH - 1) / 2 - ho.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
      · intro c _ hc_ne
        apply Finset.sum_eq_zero; intro kh _
        apply Finset.sum_eq_zero; intro kw _
        rw [ite_eq_right (fun ⟨hcc, _, _⟩ => hc_ne hcc)]; ring
      · intro hni; exact absurd (Finset.mem_univ ci) hni
    · rw [dite_eq_right hb]
      -- Show the inner sum is 0: for !back_cond, no (c, kh, kw) satisfies the indicator.
      symm
      apply Finset.sum_eq_zero; intro c _
      apply Finset.sum_eq_zero; intro kh _
      apply Finset.sum_eq_zero; intro kw _
      rw [ite_eq_right ?_]; · ring
      intro ⟨_, hkh_eq, hkw_eq⟩
      apply hb
      refine ⟨?_, ?_, ?_, ?_⟩
      · have := kh.isLt; omega
      · have := kh.isLt; omega
      · have := kw.isLt; omega
      · have := kw.isLt; omega

/-- Named accessor for the conv2d input backward — aligns with MLIR
    codegen (`stablehlo.convolution` in the backward pass). -/
noncomputable abbrev conv2d_input_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Tensor3 ic h w :=
  (conv2d_has_vjp3 W b).backward x dy

/-- **Uniform VJP-correctness wrapper** for `conv2d` — a citable `_correct`
    matching the convention of every other layer (just unfolds the
    `HasVJP3.correct` field of `conv2d_has_vjp3`). -/
theorem conv2d_has_vjp3_correct {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w) :
    (conv2d_has_vjp3 W b).backward x dy ci hi wi =
      ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        pdiv3 (conv2d W b) x ci hi wi co ho wo * dy co ho wo :=
  (conv2d_has_vjp3 W b).correct x dy ci hi wi

-- ════════════════════════════════════════════════════════════════
-- § Flattened conv and the conv → bn → relu block VJP
-- ════════════════════════════════════════════════════════════════

/-- **Flat conv** — `conv2d` bridged into flattened `Vec → Vec` space:
    `flatConv W b = flatten ∘ conv2d W b ∘ unflatten`. Spatial dims are
    preserved (`ic h w → oc h w`), so this is `Vec (ic*h*w) → Vec (oc*h*w)`.
    This is the form the ResNet/CNN VJP composition actually uses
    (everything lives in flat Vec space). -/
noncomputable def flatConv {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  fun v => Tensor3.flatten (conv2d W b (Tensor3.unflatten v))

/-- **`flatConv` is differentiable everywhere.** Composition of the three
    differentiable maps `unflatten`, `conv2d`, `flatten`. This is the
    differentiability witness `vjp_comp_at` needs to chain conv into the
    block. -/
@[fun_prop]
theorem flatConv_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (flatConv W b : Vec (ic * h * w) → Vec (oc * h * w)) :=
  Tensor3.flatten_differentiable.comp
    ((conv2d_differentiable W b).comp Tensor3.unflatten_differentiable)

/-- A conv with everywhere-zero kernel and bias maps anything to `0`. -/
theorem flatConv_eq_zero {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hW : ∀ o c kh kw, W o c kh kw = 0) (hb : ∀ o, b o = 0) (v : Vec (ic * h * w)) :
    flatConv (h := h) (w := w) W b v = (fun _ => (0:ℝ)) := by
  funext k; simp [flatConv, conv2d, Tensor3.flatten, hW, hb]

/-- **An `act ∘ norm ∘ lin` stage's VJP at a point** — a conv (or depthwise, or batched) op, a
    normalisation, then an activation. `lin` and `norm` are differentiable everywhere, so their
    global VJPs lift through `.toHasVJPAt`; `act` needs a derivative and a VJP only at the stage's
    pre-activation `norm (lin v)`, which is what relu and relu6 have off their kinks. Two
    `vjp_comp_at`s: `norm ∘ lin`, then `act`. Every conv-bn-act stage VJP in the repo is this at a
    particular `lin`, `norm` and `act`. -/
noncomputable def stage_has_vjp_at {a b : Nat}
    (lin : Vec a → Vec b) (norm : Vec b → Vec b) (act : Vec b → Vec b) (v : Vec a)
    (hlin : Differentiable ℝ lin) (hlinV : HasVJP lin)
    (hnorm : Differentiable ℝ norm) (hnormV : HasVJP norm)
    (hact : DifferentiableAt ℝ act (norm (lin v))) (hactV : HasVJPAt act (norm (lin v))) :
    HasVJPAt (act ∘ norm ∘ lin) v :=
  vjp_comp_at (norm ∘ lin) act v ((hnorm (lin v)).comp v (hlin v)) hact
    (vjp_comp_at lin norm v (hlin v) (hnorm _) (hlinV.toHasVJPAt v) (hnormV.toHasVJPAt _)) hactV

/-- **conv → bn → relu block VJP at a smooth point.**

    The workhorse for composing a ResNet VJP. In flattened `Vec` space,
    the block is `relu ∘ bnForward ∘ flatConv : Vec (ic*h*w) → Vec (oc*h*w)`
    (BatchNorm runs over the `oc*h*w` flattened activations with scalar
    `ε, γ, β`). It is `stage_has_vjp_at` at a point `v`:

    * inner = `bnForward ∘ flatConv` — both differentiable everywhere
      (`flatConv_differentiable`, `bnForward_differentiable`), so their
      bundled VJPs lift through `.toHasVJPAt`. The conv witness is the
      `HasVJP3`-bridged `hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)`.
    * outer = `relu` — needs the smoothness hypothesis `h_smooth` (no
      post-BN activation hits the ReLU kink) for both
      `relu_differentiableAt_of_smooth` and `relu_has_vjp_at`.

    Mirrors `mlp_has_vjp_at` (dense→relu→dense), with flatConv/bn in
    place of the dense layers. -/
noncomputable def convBnRelu_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, bnForward (oc * h * w) ε γ β (flatConv W b v) k ≠ 0) :
    HasVJPAt (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b) v :=
  stage_has_vjp_at (flatConv W b) (bnForward (oc * h * w) ε γ β) (relu (oc * h * w)) v
    (flatConv_differentiable W b) (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b))
    (bnForward_differentiable (oc * h * w) ε γ β hε) (bn_has_vjp (oc * h * w) ε γ β hε)
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    (relu_has_vjp_at (oc * h * w) _ h_smooth)

-- ════════════════════════════════════════════════════════════════
-- § ResNet basic residual block VJP (flattened Vec space)
-- ════════════════════════════════════════════════════════════════

/-- **conv → bn block VJP (no ReLU), everywhere.** Just `flatConv` then
    `bnForward`, both differentiable everywhere, so this is a global
    `HasVJP` (no smoothness needed). This is the building block for the
    second conv→bn of a residual body and for the 1×1 projection skip. -/
noncomputable def convBn_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (bnForward (oc * h * w) ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  vjp_comp (flatConv W b) (bnForward (oc * h * w) ε γ β)
    (flatConv_differentiable W b)
    (bnForward_differentiable (oc * h * w) ε γ β hε)
    (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b))
    (bn_has_vjp (oc * h * w) ε γ β hε)

/-- **`convBn` is differentiable everywhere.** -/
theorem convBn_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (bnForward (oc * h * w) ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  (bnForward_differentiable (oc * h * w) ε γ β hε).comp (flatConv_differentiable W b)

/-- **Basic-block body VJP at a smooth point.**

    `F := convBn₂ ∘ convBnRelu₁ = bn₂ ∘ conv₂ ∘ relu ∘ bn₁ ∘ conv₁`,
    the body of a post-activation ResNet basic block (the outer ReLU and
    skip-add are applied later). Channels go `ic → mid → oc` (generic;
    set `ic = mid = oc = c` for the identity-skip block, `ic ≠ oc` for the
    downsample/projection block). Spatial dims `h w` preserved.

    Inner `convBnRelu₁` needs the smoothness hyp `h_smooth₁` (no post-bn₁
    activation hits the ReLU kink); outer `convBn₂` is everywhere
    differentiable, lifted via `.toHasVJPAt`. Two `vjp_comp_at` chain. -/
noncomputable def resblock_body_has_vjp_at {ic mid oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid)
    (W₂ : Kernel4 oc mid kH₂ kW₂) (b₂ : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (ic * h * w))
    (h_smooth₁ : ∀ k, bnForward (mid * h * w) ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0) :
    HasVJPAt
      ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (mid * h * w) ∘ bnForward (mid * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v := by
  -- inner = convBnRelu₁
  have step1 : HasVJPAt
      (relu (mid * h * w) ∘ bnForward (mid * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁) v :=
    convBnRelu_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  have step1_diff : DifferentiableAt ℝ
      (relu (mid * h * w) ∘ bnForward (mid * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁) v := by
    apply DifferentiableAt.comp
    · exact relu_differentiableAt_of_smooth (mid * h * w) _ h_smooth₁
    · exact ((bnForward_differentiable (mid * h * w) ε₁ γ₁ β₁ hε₁).comp
        (flatConv_differentiable W₁ b₁)) v
  -- outer = convBn₂ (everywhere)
  exact vjp_comp_at
    (relu (mid * h * w) ∘ bnForward (mid * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)
    (bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) v
    step1_diff
    ((convBn_differentiable W₂ b₂ ε₂ γ₂ β₂ hε₂) _)
    step1
    ((convBn_has_vjp W₂ b₂ ε₂ γ₂ β₂ hε₂).toHasVJPAt _)

/-- **Basic-block body is `DifferentiableAt` at a smooth point.** Needed as
    the diff witness when feeding the body into the residual/projection
    fan-in and the post-add ReLU. -/
theorem resblock_body_differentiableAt {ic mid oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid)
    (W₂ : Kernel4 oc mid kH₂ kW₂) (b₂ : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (ic * h * w))
    (h_smooth₁ : ∀ k, bnForward (mid * h * w) ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0) :
    DifferentiableAt ℝ
      ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (mid * h * w) ∘ bnForward (mid * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v := by
  fun_prop (disch := assumption)

/-- **Full basic residual block VJP (identity skip).**

    `relu(x + F(x))` with `F` the conv→bn→relu→conv→bn body and an
    identity skip (so `ic = mid = oc = c`, spatial preserved). Two
    smoothness hyps: `h_smooth₁` for the inner block ReLU, and
    `h_smooth_out` for the post-add outer ReLU (`F v + v` avoids the
    kink). Built as `relu ∘ residual F` via `residual_has_vjp_at` then a
    final `vjp_comp_at` with `relu`. -/
noncomputable def resblock_has_vjp_at {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (c * h * w))
    (h_smooth₁ : ∀ k, bnForward (c * h * w) ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (c * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (c * h * w) ∘ bnForward (c * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v k
        + v k ≠ 0) :
    HasVJPAt
      (relu (c * h * w) ∘
        residual
          ((bnForward (c * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
            (relu (c * h * w) ∘ bnForward (c * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))) v := by
  set F :=
    ((bnForward (c * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (c * h * w) ∘ bnForward (c * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) with hF
  have hF_diff : DifferentiableAt ℝ F v :=
    resblock_body_differentiableAt W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hF : HasVJPAt F v :=
    resblock_body_has_vjp_at W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hres : HasVJPAt (residual F) v :=
    residual_has_vjp_at F v hF_diff hF
  have hres_diff : DifferentiableAt ℝ (residual F) v := residual_differentiableAt hF_diff
  have h_smooth_res : ∀ k, residual F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residual F) (relu (c * h * w)) v
    hres_diff
    (relu_differentiableAt_of_smooth (c * h * w) _ h_smooth_res)
    hres
    (relu_has_vjp_at (c * h * w) _ h_smooth_res)

/-- **Downsample/projection basic residual block VJP.**

    `relu(proj(x) + F(x))` where the channel/stride change is folded into
    the conv dims: body `F` maps `ic → oc` (first conv `ic → oc`, second
    `oc → oc`), and the skip is a 1×1 `convBn` projection `proj : ic → oc`
    (everywhere differentiable — no ReLU, so its diffAt is immediate).
    Built with `residualProj_has_vjp_at`, then `vjp_comp_at` with the
    post-add `relu` under `h_smooth_out`. -/
noncomputable def resblockProj_has_vjp_at
    {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp : ℝ)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * h * w))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp) v k)
      + ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v k
        ≠ 0) :
    HasVJPAt
      (relu (oc * h * w) ∘
        residualProj
          (bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp)
          ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
            (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))) v := by
  let proj : Vec (ic * h * w) → Vec (oc * h * w) :=
    bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp
  let F : Vec (ic * h * w) → Vec (oc * h * w) :=
    (bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)
  show HasVJPAt (relu (oc * h * w) ∘ residualProj proj F) v
  have hproj_diff : DifferentiableAt ℝ proj v :=
    (convBn_differentiable Wp bp εp γp βp hεp) v
  have hproj_vjp : HasVJPAt proj v :=
    (convBn_has_vjp Wp bp εp γp βp hεp).toHasVJPAt v
  have hF_diff : DifferentiableAt ℝ F v :=
    resblock_body_differentiableAt W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hF : HasVJPAt F v :=
    resblock_body_has_vjp_at W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hres : HasVJPAt (residualProj proj F) v :=
    residualProj_has_vjp_at proj F v hproj_diff hF_diff hproj_vjp hF
  have hres_diff : DifferentiableAt ℝ (residualProj proj F) v :=
    DifferentiableAt.add hproj_diff hF_diff
  have h_smooth_res : ∀ k, residualProj proj F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residualProj proj F) (relu (oc * h * w)) v
    hres_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth_res)
    hres
    (relu_has_vjp_at (oc * h * w) _ h_smooth_res)

/-! ### Weight gradient (now proved from foundation via `unfold + fun_prop`)

The conv weight gradient implements the **transpose trick**:

    `dW[o, c, kh, kw] = Σ_{h, w} x[c, h+kh−p, w+kw−p] · dy[o, h, w]`

Here's the slick observation: this *is* a convolution, with the input
and gradient playing the roles of "input" and "kernel" respectively.

- View the input `x : (ic, H, W)` as `(ic, 1, H, W)` (treat channels as batch).
- View the gradient `dy : (oc, H, W)` as `(oc, 1, H, W)` (same trick).
- Now do a standard convolution: input shape `(ic, 1, H, W)`, kernel
  shape `(oc, 1, H, W)`. The "spatial" dims of the kernel are H×W (the
  whole image), so the output is the small `(ic, oc, kH, kW)` weight
  gradient — produced by sliding the gradient as a giant kernel.
- Transpose the output `(ic, oc, kH, kW) → (oc, ic, kH, kW)` to match
  the kernel layout.

This avoids needing a separate "convolution-with-funny-dimension-numbers"
op; we use the same forward conv operator, just with shapes reinterpreted.
Critical for backends like IREE that don't accept non-standard
`dimension_numbers` (see `iree-org/iree#21955`).

MLIR (Conv 1 backward — exactly this trick):
    %x_t      = stablehlo.transpose %x, dims = [1, 0, 2, 3]      -- (1,128,28,28)
    %dh0p_t   = stablehlo.transpose %d_h0pre, dims = [1, 0, 2, 3] -- (32,128,28,28)
    %d_W0_raw = "stablehlo.convolution"(%x_t, %dh0p_t) ...        -- (1,32,3,3)
    %d_W0     = stablehlo.transpose %d_W0_raw, dims = [1, 0, 2, 3] -- (32,1,3,3)

**Framework.** `HasVJP3` covered only input→output VJPs. For the
weight gradient we reuse the plain `HasVJP` on `Vec` by flattening
both the kernel (`Kernel4.flatten : Kernel4 → Vec (oc*ic*kH*kW)`) and
the output (`Tensor3.flatten : Tensor3 → Vec (oc*h*w)`). The bundled
`HasVJP` def packages a correct backward for the flattened function
together with its proof; the user-facing `conv2d_weight_grad` wrapper
does the flatten / unflatten housekeeping so callers see the natural
`Kernel4` type.

Numerical validation: `check_jacobians.py:test_conv2d_weight_grad`
gradient-checks the transpose-trick formula against finite differences. -/

/-- **Conv2d weight-VJP** — proved from foundation rules.

    The function `v ↦ flatten (conv2d (unflatten v) b x)` is affine in
    `v`: the broadcast bias `b o(idx_out)` plus the bias-free conv, which is
    linear in `v`. `pdiv_of_affine` reads each Jacobian entry off the bias-free
    conv of a basis vector — a Kronecker at `(o', c', kh', kw')` against the
    padded input — and the collapsed sum is the transpose-trick backward
    `dW[o', c', kh', kw'] = Σ_{hi, wi} x_pad_term(...) · dy(flat(o', hi, wi))`. -/
noncomputable def conv2d_weight_grad_has_vjp {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w) :
    HasVJP (fun v : Vec (oc * ic * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)) where
  backward := fun _v dy => fun idx_in =>
    let kw' := (finProdFinEquiv.symm idx_in).2
    let kh' := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let o' := (finProdFinEquiv.symm
              (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).1
    let c' := (finProdFinEquiv.symm
              (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).2
    ∑ hi : Fin h, ∑ wi : Fin w,
      (let pH := (kH - 1) / 2
       let pW := (kW - 1) / 2
       let hh := kh'.val + hi.val
       let ww := kw'.val + wi.val
       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
         x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
       else 0)
      * dy (finProdFinEquiv (finProdFinEquiv (o', hi), wi))
  correct := by
    intro v dy idx_in
    -- The map is affine in the kernel: the bias-free conv (linear in `v'`) plus the broadcast
    -- bias, so each Jacobian entry is the bias-free conv of the basis vector `e_idx_in`.
    have hsplit : (fun v' : Vec (oc * ic * kH * kW) =>
          Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) =
        fun v' => Tensor3.flatten (conv2d (Kernel4.unflatten v') 0 x) +
          (fun k => b (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1) := by
      funext v' k; simp only [Tensor3.flatten, conv2d, Pi.add_apply, Pi.zero_apply, zero_add]
      ring
    have hadd : ∀ u u' : Vec (oc * ic * kH * kW),
        Tensor3.flatten (conv2d (Kernel4.unflatten (u + u')) 0 x) =
          Tensor3.flatten (conv2d (Kernel4.unflatten u) 0 x) +
            Tensor3.flatten (conv2d (Kernel4.unflatten u') 0 x) := by
      intro u u'; funext k
      simp only [Tensor3.flatten, conv2d, Kernel4.unflatten, Pi.add_apply, Pi.zero_apply,
        zero_add, add_mul, Finset.sum_add_distrib]
    have hsmul : ∀ (a : ℝ) (u : Vec (oc * ic * kH * kW)),
        Tensor3.flatten (conv2d (Kernel4.unflatten (a • u)) 0 x) =
          a • Tensor3.flatten (conv2d (Kernel4.unflatten u) 0 x) := by
      intro a u; funext k
      simp only [Tensor3.flatten, conv2d, Kernel4.unflatten, Pi.smul_apply, Pi.zero_apply,
        zero_add, smul_eq_mul, Finset.mul_sum, mul_assoc]
    rw [hsplit]
    simp only [pdiv_of_affine (fun v' => Tensor3.flatten (conv2d (Kernel4.unflatten v') 0 x)) _
      hadd hsmul]
    -- Unpack `idx_in = (o', c', kh', kw')` and re-index `idx_out` as `(o, hi, wi)`.
    obtain ⟨⟨q, kw'⟩, rfl⟩ := finProdFinEquiv.surjective idx_in
    obtain ⟨⟨r, kh'⟩, rfl⟩ := finProdFinEquiv.surjective q
    obtain ⟨⟨o', c'⟩, rfl⟩ := finProdFinEquiv.surjective r
    rw [← Equiv.sum_comp (finProdFinEquiv : Fin (oc * h) × Fin w ≃ Fin (oc * h * w)),
      Fintype.sum_prod_type, ← Equiv.sum_comp (finProdFinEquiv : Fin oc × Fin h ≃ Fin (oc * h)),
      Fintype.sum_prod_type]
    simp only [Tensor3.flatten, conv2d, Kernel4.unflatten, Pi.zero_apply, zero_add,
      basisVec_apply, Equiv.symm_apply_apply, EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq,
      and_assoc, ite_and, ite_mul, one_mul, zero_mul, Finset.sum_ite_irrel,
      Finset.sum_const_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- Named accessor for the conv2d weight backward — aligns with MLIR
    codegen (the "transpose trick" `stablehlo.convolution` in the backward
    pass). Unwraps the flattening so callers see `Kernel4 → Kernel4`. -/
noncomputable def conv2d_weight_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Kernel4 oc ic kH kW :=
  Kernel4.unflatten
    ((conv2d_weight_grad_has_vjp b x).backward
      (Kernel4.flatten W) (Tensor3.flatten dy))

/-- **Conv2d bias-VJP** — proved from foundation rules. Now that `conv2d`
    is a real def, the function `b ↦ flatten (conv2d W b x)` decomposes
    as `(channel broadcast of b) + (bias-free conv, constant in b)`, so
    `pdiv_of_affine` gives the channel Kronecker, collapsed over the
    `(c, hi, wi)` decomposition of `Fin (oc*h*w)`.
    The backward is `db[o] = Σ_{hi, wi} dy[o, hi, wi]` (matches
    `conv2d_bias_grad_formula` below). -/
noncomputable def conv2d_bias_grad_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) :
    HasVJP (fun b : Vec oc => Tensor3.flatten (conv2d W b x)) where
  backward := fun _b dy => fun o =>
    ∑ hi : Fin h, ∑ wi : Fin w,
      dy (finProdFinEquiv (finProdFinEquiv (o, hi), wi))
  correct := by
    intro b dy o
    -- `b ↦ conv2d W b x` is affine: the bias broadcast (linear in `b`) plus the bias-free conv.
    have hsplit : (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) =
        fun b' => (fun k => b' (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1) +
          Tensor3.flatten (conv2d W 0 x) := by
      funext b' k; simp only [Tensor3.flatten, conv2d, Pi.add_apply, Pi.zero_apply, zero_add]
    rw [hsplit]
    simp only [pdiv_of_affine (fun b' k => b' (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1)
      _ (fun _ _ => rfl) (fun _ _ => rfl), basisVec_apply]
    -- Re-index `Fin (oc*h*w)` as `(c, hi, wi)` and collapse the channel Kronecker at `c = o`.
    rw [← Equiv.sum_comp (finProdFinEquiv : Fin (oc * h) × Fin w ≃ Fin (oc * h * w)),
      Fintype.sum_prod_type, ← Equiv.sum_comp (finProdFinEquiv : Fin oc × Fin h ≃ Fin (oc * h)),
      Fintype.sum_prod_type]
    simp only [Equiv.symm_apply_apply, ite_mul, one_mul, zero_mul, Finset.sum_ite_irrel,
      Finset.sum_const_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- Named accessor for the conv2d bias backward via the VJP framework. -/
noncomputable def conv2d_bias_grad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w) : Vec oc :=
  (conv2d_bias_grad_has_vjp W x).backward b (Tensor3.flatten dy)

/-- **Conv2d bias gradient — closed-form formula** (documented, numerically
    verified, expected to equal `conv2d_bias_grad` up to fp precision).

    `db[o] = Σ_{h, w} dy[o, h, w]`

    Each output cell adds the same `b[o]`, so its gradient accumulates
    the contributions from every spatial position. MLIR emits this as
    a `stablehlo.reduce` across the spatial (and batch) dims. -/
noncomputable def conv2d_bias_grad_formula {oc h w : Nat}
    (dy : Tensor3 oc h w) : Vec oc :=
  fun o => ∑ y : Fin h, ∑ x : Fin w, dy o y x

-- ════════════════════════════════════════════════════════════════
-- § MaxPool
-- ════════════════════════════════════════════════════════════════

/-- **MaxPool 2×2 stride 2 forward** — concrete definition.

    Each output cell is the maximum of a 2×2 window of input cells:
    `y[c, h, w] = max{ x[c, 2h+a, 2w+b] : a, b ∈ {0,1} }`. No longer
    an axiom — replaced with the explicit four-way max.

    MLIR:
      %pool = "stablehlo.reduce_window"(%h1, %neginf) ({
        ^bb0(%a, %b): stablehlo.return (stablehlo.maximum %a, %b)
      }) {window_dimensions = [1, 1, 2, 2], window_strides = [1, 1, 2, 2]} -/
noncomputable def maxPool2 {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) : Tensor3 c h w :=
  fun ch hi wi =>
    let i0 : Fin (2*h) := ⟨2*hi.val,     by have := hi.isLt; omega⟩
    let i1 : Fin (2*h) := ⟨2*hi.val + 1, by have := hi.isLt; omega⟩
    let j0 : Fin (2*w) := ⟨2*wi.val,     by have := wi.isLt; omega⟩
    let j1 : Fin (2*w) := ⟨2*wi.val + 1, by have := wi.isLt; omega⟩
    max (max (x ch i0 j0) (x ch i1 j0)) (max (x ch i0 j1) (x ch i1 j1))

/-- **MaxPool2 input-VJP** — gradient routes only to the argmax positions.

    The backward function implements:

      `dx[c, 2h+a, 2w+b] = dy[c, h, w] · 𝟙[(a,b) is the argmax of the window]`

    Conceptually, max-pool is a piecewise selection: each output is one
    specific input. So the Jacobian is a sparse 0/1 matrix and the VJP
    just routes the gradient to the chosen input.

    MLIR uses **tile-compare-select** — `stablehlo.select_and_scatter`
    is avoided because IREE does not support it (see `MlirCodegen.lean`'s
    `maxPool` backward case for the full emitter):

      // Broadcast dy and the pooled output back up to the input shape:
      %dy_tiled  = stablehlo.broadcast_in_dim %d_pool
      %out_tiled = stablehlo.broadcast_in_dim %pool
      // Mask the input cells whose value matches the window max:
      %mask = stablehlo.compare EQ, %out_tiled, %h1
      // Route gradient through that mask (zeros elsewhere):
      %d_h1 = stablehlo.select %mask, %dy_tiled, %zero

    **Canonical (junk-at-tie) witness.** `HasVJP3.correct` is
    satisfied by the canonical pdiv3-derived backward via `rfl`. At
    argmax-tie boundaries `maxPool2` is not differentiable, so `pdiv3`
    agrees with `fderiv`'s junk default of `0` and the canonical
    witness is also `0` there. The codegen emits the tile-compare-
    select formula above instead — at ties, the EQ-mask routes the
    gradient to *every* tied input cell (PyTorch/JAX semantics), not
    a single deterministic argmax. See `LeanMlir/Proofs/README.md` for
    the trust-boundary discussion. Smooth-point agreement is formal:
    see `maxPool2_codegen_matches_canonical` below. -/
noncomputable def maxPool2_has_vjp3 {c h w : Nat} :
    HasVJP3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w) where
  backward x dy ci hi wi :=
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo
  correct _ _ _ _ _ := rfl

/-- Named accessor for the maxPool2 input backward — aligns with the
    codegen's tile-compare-select MLIR. -/
noncomputable abbrev maxPool2_input_grad {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w) : Tensor3 c (2*h) (2*w) :=
  maxPool2_has_vjp3.backward x dy

-- ════════════════════════════════════════════════════════════════
-- § MaxPool2 smooth-point bridge to codegen
-- ════════════════════════════════════════════════════════════════

/-! ## Smooth-point codegen bridge for MaxPool2

Closes the smooth-point half of the codegen trust boundary at MaxPool2.
At points where every 2×2 window has a unique strict argmax, the
canonical pdiv-derived backward in `maxPool2_has_vjp3` collapses to
"route `dy` to the argmax position, zero elsewhere" — the formula that
`MlirCodegen.lean` emits via tile-compare-select (broadcast dy and the
pooled output, `compare EQ` to find the argmax cells, `select` to
route). Mirrors `relu_codegen_matches_canonical` in `MLP.lean`, but
the local linearization is per-2×2-window rather than per-coordinate. -/

-- Window-index helpers --------------------------------------------

/-- Row of the 2×2 window that contains input row `hi_in`. -/
def winRow {h : Nat} (hi_in : Fin (2 * h)) : Fin h :=
  ⟨hi_in.val / 2, by have := hi_in.isLt; omega⟩

/-- Position within the window's row (0 = top, 1 = bottom). -/
def winRowMod {h : Nat} (hi_in : Fin (2 * h)) : Fin 2 :=
  ⟨hi_in.val % 2, by omega⟩

def winCol {w : Nat} (wi_in : Fin (2 * w)) : Fin w :=
  ⟨wi_in.val / 2, by have := wi_in.isLt; omega⟩

def winColMod {w : Nat} (wi_in : Fin (2 * w)) : Fin 2 :=
  ⟨wi_in.val % 2, by omega⟩

/-- Row index of position `a ∈ Fin 2` inside output window `hi_out`. -/
def winRowInv {h : Nat} (hi_out : Fin h) (a : Fin 2) : Fin (2 * h) :=
  ⟨2 * hi_out.val + a.val, by have := hi_out.isLt; have := a.isLt; omega⟩

def winColInv {w : Nat} (wi_out : Fin w) (b : Fin 2) : Fin (2 * w) :=
  ⟨2 * wi_out.val + b.val, by have := wi_out.isLt; have := b.isLt; omega⟩

theorem winRowInv_winRow {h : Nat} (hi_in : Fin (2 * h)) :
    winRowInv (winRow hi_in) (winRowMod hi_in) = hi_in := by
  apply Fin.ext; show 2 * (hi_in.val / 2) + hi_in.val % 2 = hi_in.val; omega

theorem winColInv_winCol {w : Nat} (wi_in : Fin (2 * w)) :
    winColInv (winCol wi_in) (winColMod wi_in) = wi_in := by
  apply Fin.ext; show 2 * (wi_in.val / 2) + wi_in.val % 2 = wi_in.val; omega

theorem winRow_winRowInv {h : Nat} (ho : Fin h) (a : Fin 2) :
    winRow (winRowInv ho a) = ho := by
  apply Fin.ext
  show (2 * ho.val + a.val) / 2 = ho.val
  have := a.isLt; omega

theorem winRowMod_winRowInv {h : Nat} (ho : Fin h) (a : Fin 2) :
    winRowMod (winRowInv ho a) = a := by
  apply Fin.ext
  show (2 * ho.val + a.val) % 2 = a.val
  have := a.isLt; omega

theorem winCol_winColInv {w : Nat} (wo : Fin w) (b : Fin 2) :
    winCol (winColInv wo b) = wo := by
  apply Fin.ext
  show (2 * wo.val + b.val) / 2 = wo.val
  have := b.isLt; omega

theorem winColMod_winColInv {w : Nat} (wo : Fin w) (b : Fin 2) :
    winColMod (winColInv wo b) = b := by
  apply Fin.ext
  show (2 * wo.val + b.val) % 2 = b.val
  have := b.isLt; omega

theorem winRowInv_zero {h : Nat} (ho : Fin h) :
    winRowInv ho (0 : Fin 2) =
      (⟨2 * ho.val, by have := ho.isLt; omega⟩ : Fin (2 * h)) := by
  apply Fin.ext; show 2 * ho.val + 0 = 2 * ho.val; omega

theorem winRowInv_one {h : Nat} (ho : Fin h) :
    winRowInv ho (1 : Fin 2) =
      (⟨2 * ho.val + 1, by have := ho.isLt; omega⟩ : Fin (2 * h)) := by
  apply Fin.ext; show 2 * ho.val + 1 = 2 * ho.val + 1; rfl

theorem winColInv_zero {w : Nat} (wo : Fin w) :
    winColInv wo (0 : Fin 2) =
      (⟨2 * wo.val, by have := wo.isLt; omega⟩ : Fin (2 * w)) := by
  apply Fin.ext; show 2 * wo.val + 0 = 2 * wo.val; omega

theorem winColInv_one {w : Nat} (wo : Fin w) :
    winColInv wo (1 : Fin 2) =
      (⟨2 * wo.val + 1, by have := wo.isLt; omega⟩ : Fin (2 * w)) := by
  apply Fin.ext; show 2 * wo.val + 1 = 2 * wo.val + 1; rfl

-- Smoothness / argmax predicates ----------------------------------

/-- **Smoothness:** every 2×2 window of `x` has pairwise-distinct
    values (so a unique strict argmax). The natural domain on which
    `maxPool2` is differentiable. -/
def MaxPool2Smooth {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) : Prop :=
  ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w)
    (ab ab' : Fin 2 × Fin 2), ab ≠ ab' →
    x ci (winRowInv hi_out ab.1) (winColInv wi_out ab.2) ≠
    x ci (winRowInv hi_out ab'.1) (winColInv wi_out ab'.2)

/-- Input position `(ci, hi_in, wi_in)` attains the max of its window. -/
def MaxPool2IsArgmax {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (ci : Fin c) (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w)) : Prop :=
  ∀ a b : Fin 2,
    x ci (winRowInv (winRow hi_in) a) (winColInv (winCol wi_in) b) ≤
    x ci hi_in wi_in

-- Argmax extractor + window-max characterization -----------------

/-- A (not necessarily unique) argmax of the 2×2 window at output
    position `(co, ho, wo)`. Unique under `MaxPool2Smooth`. -/
noncomputable def maxPool2Argmax {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) : Fin 2 × Fin 2 :=
  Classical.choose
    ((Finset.univ : Finset (Fin 2 × Fin 2)).exists_max_image
      (fun ab => x co (winRowInv ho ab.1) (winColInv wo ab.2))
      Finset.univ_nonempty)

theorem maxPool2Argmax_max {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) (ab : Fin 2 × Fin 2) :
    x co (winRowInv ho ab.1) (winColInv wo ab.2) ≤
    x co (winRowInv ho (maxPool2Argmax x co ho wo).1)
          (winColInv wo (maxPool2Argmax x co ho wo).2) :=
  (Classical.choose_spec
    ((Finset.univ : Finset (Fin 2 × Fin 2)).exists_max_image
      (fun ab' => x co (winRowInv ho ab'.1) (winColInv wo ab'.2))
      Finset.univ_nonempty)).2 ab (Finset.mem_univ ab)

/-- If `(a, b)` dominates every other window cell, the max-pool output
    equals the value at `(a, b)`. No smoothness needed. -/
theorem maxPool2_eq_at_max {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w)
    (a : Fin 2) (b : Fin 2)
    (h_max : ∀ a' b' : Fin 2,
      x co (winRowInv ho a') (winColInv wo b') ≤
      x co (winRowInv ho a) (winColInv wo b)) :
    maxPool2 x co ho wo =
      x co (winRowInv ho a) (winColInv wo b) := by
  have h00 := h_max 0 0
  have h10 := h_max 1 0
  have h01 := h_max 0 1
  have h11 := h_max 1 1
  rw [winRowInv_zero, winColInv_zero] at h00
  rw [winRowInv_one, winColInv_zero] at h10
  rw [winRowInv_zero, winColInv_one] at h01
  rw [winRowInv_one, winColInv_one] at h11
  show max (max _ _) (max _ _) = _
  apply le_antisymm
  · exact max_le (max_le h00 h10) (max_le h01 h11)
  · fin_cases a <;> fin_cases b <;> dsimp only
    · show x co (winRowInv ho 0) (winColInv wo 0) ≤ _
      rw [winRowInv_zero, winColInv_zero]
      exact le_max_of_le_left (le_max_left _ _)
    · show x co (winRowInv ho 0) (winColInv wo 1) ≤ _
      rw [winRowInv_zero, winColInv_one]
      exact le_max_of_le_right (le_max_left _ _)
    · show x co (winRowInv ho 1) (winColInv wo 0) ≤ _
      rw [winRowInv_one, winColInv_zero]
      exact le_max_of_le_left (le_max_right _ _)
    · show x co (winRowInv ho 1) (winColInv wo 1) ≤ _
      rw [winRowInv_one, winColInv_one]
      exact le_max_of_le_right (le_max_right _ _)

theorem maxPool2_eq_argmax_value {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    maxPool2 x co ho wo =
      x co (winRowInv ho (maxPool2Argmax x co ho wo).1)
            (winColInv wo (maxPool2Argmax x co ho wo).2) :=
  maxPool2_eq_at_max x co ho wo _ _ (fun a' b' =>
    maxPool2Argmax_max x co ho wo (a', b'))

/-- Under smoothness, the argmax of any window is unique: two positions
    that both dominate the window coincide. -/
theorem maxPool2_argmax_unique {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool2Smooth x)
    (ci : Fin c) (ho : Fin h) (wo : Fin w)
    (ab ab' : Fin 2 × Fin 2)
    (h_ab : ∀ cd : Fin 2 × Fin 2,
      x ci (winRowInv ho cd.1) (winColInv wo cd.2) ≤
      x ci (winRowInv ho ab.1) (winColInv wo ab.2))
    (h_ab' : ∀ cd : Fin 2 × Fin 2,
      x ci (winRowInv ho cd.1) (winColInv wo cd.2) ≤
      x ci (winRowInv ho ab'.1) (winColInv wo ab'.2)) :
    ab = ab' := by
  by_contra h_ne
  have h1 := h_ab' ab
  have h2 := h_ab ab'
  exact h_smooth ci ho wo ab ab' h_ne (le_antisymm h1 h2)

/-- Under smoothness, `MaxPool2IsArgmax` pins `maxPool2Argmax` to the
    `(winRowMod, winColMod)` position of the witness. -/
theorem maxPool2Argmax_eq_of_isArgmax {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool2Smooth x)
    (ci : Fin c) (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w))
    (h_arg : MaxPool2IsArgmax x ci hi_in wi_in) :
    maxPool2Argmax x ci (winRow hi_in) (winCol wi_in) =
      (winRowMod hi_in, winColMod wi_in) := by
  apply maxPool2_argmax_unique x h_smooth ci (winRow hi_in) (winCol wi_in)
  · intro cd; exact maxPool2Argmax_max x ci _ _ cd
  · intro cd
    have h_rhs :
        x ci (winRowInv (winRow hi_in) (winRowMod hi_in, winColMod wi_in).1)
              (winColInv (winCol wi_in) (winRowMod hi_in, winColMod wi_in).2) =
        x ci hi_in wi_in := by
      simp only [winRowInv_winRow, winColInv_winCol]
    rw [h_rhs]
    exact h_arg cd.1 cd.2

-- Local linearization (reindex σ) --------------------------------

/-- For each output flat index `k_out` (decoded to `(co, ho, wo)`), the
    flat index of the argmax's input position in `Vec (c * (2*h) * (2*w))`.
    Used as the carrier of the local-linearization `reindexCLM`. -/
noncomputable def maxPool2LocalReindex {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (k_out : Fin (c * h * w)) : Fin (c * (2 * h) * (2 * w)) :=
  let r1 := finProdFinEquiv.symm k_out
  let wo : Fin w := r1.2
  let r2 := finProdFinEquiv.symm r1.1
  let co : Fin c := r2.1
  let ho : Fin h := r2.2
  let ab := maxPool2Argmax x co ho wo
  finProdFinEquiv (finProdFinEquiv (co, winRowInv ho ab.1), winColInv wo ab.2)

/-- **Smooth-point local-linearization for max-pool.** Near `flatten x` the
    flattened max-pool agrees with the reindex `y ↦ y ∘ σ` where σ routes each
    output position to its argmax's input position: every window keeps its
    argmax, since finitely many strict inequalities persist on a neighbourhood
    (`Filter.eventually_all`). Promoted via `EventuallyEq`. The positivity
    hypotheses are not used. -/
theorem maxPool2_flat_hasFDerivAt {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (h_smooth : MaxPool2Smooth x)
    (_hc : 0 < c) (_hh : 0 < h) (_hw : 0 < w) :
    HasFDerivAt
      (fun v : Vec (c * (2 * h) * (2 * w)) =>
        Tensor3.flatten (maxPool2 (Tensor3.unflatten v)))
      (reindexCLM (maxPool2LocalReindex x))
      (Tensor3.flatten x) := by
  refine (reindexCLM (maxPool2LocalReindex x)).hasFDerivAt.congr_of_eventuallyEq ?_
  have hmax : ∀ᶠ y in nhds (Tensor3.flatten x), ∀ (co : Fin c) (ho : Fin h) (wo : Fin w)
      (a' b' : Fin 2), Tensor3.unflatten y co (winRowInv ho a') (winColInv wo b') ≤
        Tensor3.unflatten y co (winRowInv ho (maxPool2Argmax x co ho wo).1)
          (winColInv wo (maxPool2Argmax x co ho wo).2) := by
    have hcont : ∀ co hi wi, ContinuousAt
        (fun y : Vec (c * (2 * h) * (2 * w)) => Tensor3.unflatten y co hi wi) (Tensor3.flatten x) :=
      fun _ _ _ => (continuous_apply _).continuousAt
    simp only [Filter.eventually_all]
    intro co ho wo a' b'
    by_cases hab : (a', b') = maxPool2Argmax x co ho wo
    · exact Filter.Eventually.of_forall fun _ => by rw [← hab]
    · refine ((hcont _ _ _).eventually_lt (hcont _ _ _) ?_).mono fun _ => le_of_lt
      simpa [Tensor3.unflatten_flatten] using lt_of_le_of_ne (maxPool2Argmax_max x co ho wo (a', b'))
        (h_smooth co ho wo _ _ hab)
  filter_upwards [hmax] with y hy
  funext k_out
  exact maxPool2_eq_at_max (Tensor3.unflatten y) _ _ _ _ _ (hy _ _ _)

open scoped Classical in
/-- **MaxPool2 smooth-point Jacobian.** At a smooth point, `pdiv3` of
    `maxPool2` is a sparse 0/1 indicator: 1 exactly when the output
    `(co, ho, wo)` is the window of the input `(ci, hi_in, wi_in)` AND
    that input is the argmax of its window. -/
theorem pdiv3_maxPool2_smooth {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool2Smooth x)
    (ci : Fin c) (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    pdiv3 maxPool2 x ci hi_in wi_in co ho wo =
      (if co = ci ∧ ho = winRow hi_in ∧ wo = winCol wi_in
          ∧ MaxPool2IsArgmax x ci hi_in wi_in
        then (1 : ℝ) else 0) := by
  have hc : 0 < c := Fin.pos ci
  have hh : 0 < h := Fin.pos ho
  have hw : 0 < w := Fin.pos wo
  have h_fderiv := maxPool2_flat_hasFDerivAt x h_smooth hc hh hw
  unfold pdiv3 pdiv
  rw [h_fderiv.fderiv]
  show reindexCLM (maxPool2LocalReindex x)
        (basisVec (finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)))
        (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = _
  rw [reindexCLM_apply]
  dsimp only
  rw [basisVec_apply]
  have h_sigma :
      maxPool2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) =
      finProdFinEquiv (finProdFinEquiv (co, winRowInv ho (maxPool2Argmax x co ho wo).1),
                       winColInv wo (maxPool2Argmax x co ho wo).2) := by
    show finProdFinEquiv
          (finProdFinEquiv
            ((finProdFinEquiv.symm (finProdFinEquiv.symm
              (finProdFinEquiv (finProdFinEquiv (co, ho), wo))).1).1,
             winRowInv (finProdFinEquiv.symm (finProdFinEquiv.symm
              (finProdFinEquiv (finProdFinEquiv (co, ho), wo))).1).2
                       (maxPool2Argmax x _ _ _).1),
            winColInv (finProdFinEquiv.symm
              (finProdFinEquiv (finProdFinEquiv (co, ho), wo))).2
                      (maxPool2Argmax x _ _ _).2) = _
    simp [Equiv.symm_apply_apply]
  rw [h_sigma]
  congr 1
  apply propext
  constructor
  · intro hA
    have h1 := finProdFinEquiv.injective hA
    have h2 := Prod.mk.inj h1
    have h3 := finProdFinEquiv.injective h2.1
    have h4 := Prod.mk.inj h3
    have h_ho_eq : ho = winRow hi_in := by
      have := congrArg winRow h4.2
      rwa [winRow_winRowInv] at this
    have h_wo_eq : wo = winCol wi_in := by
      have := congrArg winCol h2.2
      rwa [winCol_winColInv] at this
    refine ⟨h4.1, h_ho_eq, h_wo_eq, ?_⟩
    intro a b
    have h_arg : x co (winRowInv ho a) (winColInv wo b) ≤
                 x co (winRowInv ho (maxPool2Argmax x co ho wo).1)
                       (winColInv wo (maxPool2Argmax x co ho wo).2) :=
      maxPool2Argmax_max x co ho wo (a, b)
    have h_val : x co (winRowInv ho (maxPool2Argmax x co ho wo).1)
                      (winColInv wo (maxPool2Argmax x co ho wo).2) =
                 x ci hi_in wi_in := by
      rw [h4.2, h2.2, h4.1]
    rw [h_val] at h_arg
    rw [h4.1] at h_arg
    rw [← h_ho_eq, ← h_wo_eq]
    exact h_arg
  · rintro ⟨hco_eq, hho_eq, hwo_eq, h_arg⟩
    subst hco_eq
    have h_argmax : maxPool2Argmax x co (winRow hi_in) (winCol wi_in) =
                    (winRowMod hi_in, winColMod wi_in) :=
      maxPool2Argmax_eq_of_isArgmax x h_smooth co hi_in wi_in h_arg
    rw [hho_eq, hwo_eq, h_argmax]
    show finProdFinEquiv (finProdFinEquiv (co, winRowInv (winRow hi_in) (winRowMod hi_in)),
                          winColInv (winCol wi_in) (winColMod wi_in)) = _
    rw [winRowInv_winRow, winColInv_winCol]

open scoped Classical in
/-- **Bridge: `maxPool2_has_vjp3`'s canonical backward matches the
    codegen formula at smooth points.**

    At points where every 2×2 window has a unique strict argmax, the
    canonical `pdiv3`-derived backward collapses to "`dy` at the
    window's output position, but only at the argmax input cell" — the
    tile-compare-select formula `MlirCodegen.lean` emits. Closes the
    smooth-point half of the codegen trust boundary; what remains is
    the kink convention at argmax-tie boundaries (EQ-mask routes the
    gradient to every tied cell). -/
theorem maxPool2_codegen_matches_canonical {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (h_smooth : MaxPool2Smooth x) (dy : Tensor3 c h w)
    (ci : Fin c) (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w)) :
    (maxPool2_has_vjp3 :
        HasVJP3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)).backward
        x dy ci hi_in wi_in
    = (if MaxPool2IsArgmax x ci hi_in wi_in
       then dy ci (winRow hi_in) (winCol wi_in) else 0) := by
  show ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
        pdiv3 maxPool2 x ci hi_in wi_in co ho wo * dy co ho wo = _
  simp_rw [pdiv3_maxPool2_smooth x h_smooth ci hi_in wi_in]
  simp [ite_and, Finset.sum_ite_eq']

open scoped Classical in
/-- **MaxPool2 pointwise VJP — no canonical-witness escape.**

    `HasVJPAt3 maxPool2 x` under `MaxPool2Smooth x`. The backward is
    the codegen tile-compare-select formula directly (route `dy` to
    the argmax cell, zero elsewhere); the `correct` field is
    `maxPool2_codegen_matches_canonical` flipped, not `rfl`.
    Companion of `relu_has_vjp_at` in MLP.lean — together they let
    `mlp_has_vjp_at` and (future) `cnn_has_vjp_at3` discharge the chain
    rule through every kinked operator without the global vacuous
    witness. -/
noncomputable def maxPool2_has_vjp_at3 {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool2Smooth x) :
    HasVJPAt3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w) x where
  backward dy ci hi_in wi_in :=
    if MaxPool2IsArgmax x ci hi_in wi_in
    then dy ci (winRow hi_in) (winCol wi_in) else 0
  correct dy ci hi_in wi_in :=
    (maxPool2_codegen_matches_canonical x h_smooth dy ci hi_in wi_in).symm

-- ════════════════════════════════════════════════════════════════
-- § Flatten
-- ════════════════════════════════════════════════════════════════

/-! ## Reshape (flatten / unflatten)

Flatten is a permutation of indices, so its VJP is just the inverse
permutation. No gradient computation needed.

MLIR:
    %flat = stablehlo.reshape %pool
            : (tensor<128x32x14x14xf32>) -> tensor<128x6272xf32>

The flatten / unflatten bijection is **already defined** in
`Tensor.lean` as `Tensor3.flatten` / `Tensor3.unflatten` (used by the
`pdiv3` derivation in Phase 5). We reuse those here rather than
duplicating — see `Tensor3.flatten_unflatten` / `unflatten_flatten`
for the mutual-inverse proofs. -/

-- ════════════════════════════════════════════════════════════════
-- § The full CNN backward pass
-- ════════════════════════════════════════════════════════════════

/-- **Walking through the CNN backward pass**.

    Unlike the MLP, where the chain rule (`vjp_comp`) gave us the whole
    backward pass in one go, here the layer types vary (Tensor3 ↔ Vec
    via flatten) so a uniform `HasVJP`-style composition would need a
    type family. For pedagogical clarity, we instead trace the backward
    pass step-by-step, matching `hand_cnn_train_step.mlir`.

    Forward:
        x ────conv W₀── h₀pre ──relu── h₀ ──conv W₁── h₁pre ──relu── h₁
          ──maxPool── pool ──flatten── d₀in ──dense W₂── d₀pre ──relu── d₀
          ──dense W₃── d₁pre ──relu── d₁ ──dense W₄── logits

    Backward (each step labeled with which lemma justifies it):

        d_logits  = softmax_ce_grad logits label                  [softmaxCE_grad]
        d_W4      = outer d₁ d_logits                             [dense_weight_grad]
        d_b4      = d_logits                                      [dense_bias_grad]
        d_d₁      = mulVec W₄ d_logits                            [dense_has_vjp]
        d_d₁pre   = relu_back d₁pre d_d₁                          [relu_has_vjp]
        d_W3      = outer d₀ d_d₁pre                              [dense_weight_grad]
        d_b3      = d_d₁pre                                       [dense_bias_grad]
        d_d₀      = mulVec W₃ d_d₁pre                             [dense_has_vjp]
        d_d₀pre   = relu_back d₀pre d_d₀                          [relu_has_vjp]
        d_W2      = outer d₀in d_d₀pre                            [dense_weight_grad]
        d_b2      = d_d₀pre                                       [dense_bias_grad]
        d_d₀in    = mulVec W₂ d_d₀pre                             [dense_has_vjp]
        d_pool    = unflatten d_d₀in                              [flatten VJP = unflatten]
        d_h₁      = maxPool2_input_grad h₁ d_pool                 [maxPool2_input_grad]
        d_h₁pre   = relu_back h₁pre d_h₁                          [relu_has_vjp, lifted to T3]
        d_W1      = conv2d_weight_grad W₁ b₁ h₀ d_h₁pre           [conv2d_weight_grad_has_vjp]  ← transpose trick
        d_b1      = conv2d_bias_grad W₁ b₁ h₀ d_h₁pre             [conv2d_bias_grad_has_vjp]
        d_h₀      = conv2d_input_grad W₁ b₁ h₀ d_h₁pre            [conv2d_has_vjp3]     ← reversed kernel
        d_h₀pre   = relu_back h₀pre d_h₀                          [relu_has_vjp, lifted to T3]
        d_W0      = conv2d_weight_grad W₀ b₀ x d_h₀pre            [conv2d_weight_grad_has_vjp]  ← transpose trick
        d_b0      = conv2d_bias_grad W₀ b₀ x d_h₀pre              [conv2d_bias_grad_has_vjp]

    Each line of the backward pass corresponds to a single line in
    `hand_cnn_train_step.mlir` (lines 134–272). The backward pass is just
    the forward layers walked in reverse, replacing each forward operation
    with its VJP. The MLIR is the literal compiled-down version of this
    derivation.

    The novelty over the MLP is in the conv layers, where the VJP turns
    out to be — itself — a convolution, just with reversed/transposed
    kernels (`conv2d_input_grad`) or swapped axes (`conv2d_weight_grad`'s
    transpose trick). Once you accept those two tricks, the entire CNN
    backprop fits in a page.
-/
example : True := trivial  -- anchor for the docstring above

/-! ## Summary of derivations in this file

- `conv2d`, `maxPool2` — forward operations (black-box forward).
- `maxPool2_has_vjp3` — input-path VJP for maxPool2 (argmax-routing
  subgradient convention).

Derived (not axioms):
- `conv2d_has_vjp3` — input-path VJP, proved with `pdiv_of_affine` (the
  conv is affine in its input; the Jacobian entry is the bias-free conv of a
  basis vector, a pad-guarded Kronecker). Backward function is `conv2d_input_grad_formula` (sum over
  `(co, ho, wo)` with reconstructed kernel offsets `kh = hi+pH-ho`,
  `kw = wi+pW-wo`).
- `conv2d_weight_grad_has_vjp` — Phase 7: the weight-path VJP, bundled
  as a plain `HasVJP` on the Kernel4-flattened function. Numerically
  gradient-checked against the transpose-trick formula in
  `check_jacobians.py:test_conv2d_weight_grad`.
- `conv2d_bias_grad_has_vjp` — Phase 9: the bias-path VJP, same bundled
  `HasVJP` pattern. The closed-form "sum output cotangent over spatial
  dims per channel" is expressed as `conv2d_bias_grad_formula`; the
  named `conv2d_bias_grad` extracts the backward via the VJP.
- `conv2d_input_grad`, `maxPool2_input_grad`, `conv2d_weight_grad`,
  `conv2d_bias_grad` — named accessors, defined as `.backward` (plus
  flatten / unflatten housekeeping for the weight / bias variants) of
  the corresponding VJP.
- `conv2d_input_grad_formula`, `conv2d_bias_grad_formula` — the
  concrete closed-form formulas (numerically verified to equal the
  VJP's backward).
- 3D reshape (`Tensor3.flatten` / `Tensor3.unflatten`) imported from
  `Tensor.lean`; 4D reshape (`Kernel4.flatten` / `Kernel4.unflatten`)
  defined here, both proved bijections. -/

/-- **Public correctness theorem for `maxPool2_has_vjp3`**: the
canonical-witness backward equals the `pdiv3`-contracted Jacobian
by definition. The codegen substitutes the standard argmax-routing
convention at non-smooth tiebreaks (see `LeanMlir/Proofs/README.md`'s
Codegen Trust Boundary). -/
theorem maxPool2_has_vjp3_correct {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    (maxPool2_has_vjp3 (c := c) (h := h) (w := w)).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  maxPool2_has_vjp3.correct x dy ci hi wi

/-- **Public correctness theorem for `maxPool2_has_vjp_at3`** — the
pointwise variant under `MaxPool2Smooth`. The underlying `.correct`
field is `maxPool2_codegen_matches_canonical` flipped (a real proof),
not `rfl`; this wrapper exposes it for comparator re-verification. -/
theorem maxPool2_has_vjp_at3_correct {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool2Smooth x)
    (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    (maxPool2_has_vjp_at3 x h_smooth).backward dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  (maxPool2_has_vjp_at3 x h_smooth).correct dy ci hi wi

-- ════════════════════════════════════════════════════════════════
-- § Global average pool + end-to-end ResNet-style CNN VJP (capstone)
-- ════════════════════════════════════════════════════════════════

/-! ## The capstone: a whole-network ResNet-style CNN VJP

`cnn_has_vjp_at` is the CNN analogue of `vit_full_has_vjp` — a single
`HasVJPAt` for an end-to-end forward pass, chained entirely in flattened
`Vec` space via `vjp_comp_at`. It first needs **global average pooling**,
which was previously only referenced in codegen, so we define it here:
`globalAvgPool x ci = (∑ hi ∑ wi x ci hi wi) / (h*w)` (mean over spatial
per channel), bridge it to flat `Vec` space (`globalAvgPoolFlat`), and
prove its linear VJP (`globalAvgPoolFlat_has_vjp`, backward broadcasts
`dy ci / (h*w)` to every spatial cell of channel `ci`) and
differentiability.

**Fixed structural choices** (a concrete-but-representative pipeline, in
the spirit of `hand_cnn_train_step.mlir`; prioritising a complete
axiom-clean end-to-end witness over maximal generality):

    input  : Vec (ic * (2h) * (2w))
    stem   : convBnRelu  ic → c   (spatial 2h×2w preserved)
    pool   : maxPool2    c, 2h×2w → c, h×w
    block1 : resblock_has_vjp_at        (identity skip,  c → c,  h×w)
    block2 : resblockProj_has_vjp_at    (projection skip, c → oc, h×w)
    gap    : globalAvgPool   oc, h×w → Vec oc
    head   : dense           oc → nClasses

So: 1 stem conv, 1 max-pool, exactly two residual blocks (one of EACH
skip type — identity and 1×1 projection — to exercise both code paths),
global-average pool, one dense classifier. Channel/spatial dims stay
implicit `Nat` params; the block/stage counts are fixed. The bundled
smoothness hypotheses (`h_stem`, `h_mp`, `h_rb1`/`h_rb1o`, `h_rb2`/
`h_rb2o`) are the family of every ReLU + max-pool site's smooth-point
condition, exactly like `mlp_has_vjp_at`'s multiple `h_smooth_*`.

The differentiability obstacle (max-pool is non-smooth globally, so
`vjp_comp_at` cannot get `DifferentiableAt` of a max-pool-containing
prefix from a global lemma) is discharged at the smooth point by
`maxPool2_flat_hasFDerivAt` (the local linearization already proved for
the max-pool Jacobian) via `.differentiableAt`. -/

/-- Global average pool: mean over spatial dims per channel. -/
noncomputable def globalAvgPool {c h w : Nat} (x : Tensor3 c h w) : Vec c :=
  fun ci => (∑ hi : Fin h, ∑ wi : Fin w, x ci hi wi) / (h * w)

/-- Flat GAP: `Vec (c*h*w) → Vec c`. -/
noncomputable def globalAvgPoolFlat (c h w : Nat) : Vec (c * h * w) → Vec c :=
  fun v => globalAvgPool (Tensor3.unflatten v : Tensor3 c h w)

@[fun_prop]
theorem globalAvgPoolFlat_differentiable (c h w : Nat) :
    Differentiable ℝ (globalAvgPoolFlat c h w) := by
  unfold globalAvgPoolFlat globalAvgPool Tensor3.unflatten
  fun_prop

/-- The channel of a flat index `idx : Fin (c*h*w)`. -/
noncomputable def flatChannel (c h w : Nat) (idx : Fin (c * h * w)) : Fin c :=
  (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1

-- Rewrite GAP as a single Finset sum over (hi, wi) ∈ univ of scaled reindex maps.
theorem globalAvgPoolFlat_as_sum (c h w : Nat) :
    (globalAvgPoolFlat c h w) =
    (fun (u : Vec (c * h * w)) (k : Fin c) =>
      ∑ p : Fin h × Fin w,
        (1 / (h * w : ℝ)) *
        u (finProdFinEquiv (finProdFinEquiv (k, p.1), p.2))) := by
  funext u k
  show globalAvgPool (Tensor3.unflatten u) k = _
  unfold globalAvgPool Tensor3.unflatten
  rw [← Finset.sum_product']
  rw [div_eq_mul_inv, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro p _
  rw [one_div, mul_comm]

theorem pdiv_globalAvgPoolFlat (c h w : Nat) (v : Vec (c * h * w))
    (idx : Fin (c * h * w)) (ci : Fin c) :
    pdiv (globalAvgPoolFlat c h w) v idx ci =
      (if flatChannel c h w idx = ci then 1 else 0) / (h * w) := by
  -- GAP is linear, so the entry is GAP of the basis vector: a spatial sum of a Kronecker.
  rw [pdiv_of_linear]
  · obtain ⟨⟨p, w0⟩, rfl⟩ := finProdFinEquiv.surjective idx
    obtain ⟨⟨c0, h0⟩, rfl⟩ := finProdFinEquiv.surjective p
    simp only [globalAvgPoolFlat, globalAvgPool, Tensor3.unflatten, flatChannel, basisVec_apply,
      Equiv.symm_apply_apply, EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq, and_assoc, ite_and,
      Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true,
      @eq_comm _ ci]
  · intro u u'; funext k
    simp only [globalAvgPoolFlat, globalAvgPool, Tensor3.unflatten, Pi.add_apply,
      Finset.sum_add_distrib, add_div]
  · intro a u; funext k
    simp only [globalAvgPoolFlat, globalAvgPool, Tensor3.unflatten, Pi.smul_apply, smul_eq_mul,
      ← mul_div_assoc, Finset.mul_sum]

/-- **Global average pool VJP (flattened).** Linear map; backward
    broadcasts `dy ci / (h*w)` to every spatial cell of channel `ci`. -/
noncomputable def globalAvgPoolFlat_has_vjp (c h w : Nat) :
    HasVJP (globalAvgPoolFlat c h w) where
  backward := fun _v dy => fun idx => dy (flatChannel c h w idx) / (h * w)
  correct := by
    intro v dy idx
    simp_rw [pdiv_globalAvgPoolFlat]
    -- ∑ ci, (if flatChannel idx = ci then 1 else 0)/(hw) * dy ci
    rw [Finset.sum_eq_single (flatChannel c h w idx)]
    · rw [ite_eq_left rfl]; ring
    · intro b _ hne
      rw [ite_eq_right (fun heq => hne heq.symm)]; ring
    · intro hp; exact absurd (Finset.mem_univ _) hp

/-- **Uniform VJP-correctness wrapper** for `globalAvgPoolFlat` — a citable
    `_correct` matching the convention of every other layer (just unfolds the
    `HasVJP.correct` field of `globalAvgPoolFlat_has_vjp`). -/
theorem globalAvgPoolFlat_has_vjp_correct (c h w : Nat)
    (x : Vec (c*h*w)) (dy : Vec c) (i : Fin (c*h*w)) :
    (globalAvgPoolFlat_has_vjp c h w).backward x dy i =
      ∑ j : Fin c, pdiv (globalAvgPoolFlat c h w) x i j * dy j :=
  (globalAvgPoolFlat_has_vjp c h w).correct x dy i

-- maxpool flat helper
noncomputable def maxPoolFlat (c h w : Nat) :
    Vec (c * (2*h) * (2*w)) → Vec (c * h * w) :=
  fun v => Tensor3.flatten (maxPool2 (Tensor3.unflatten v))

theorem maxPoolFlat_differentiableAt {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (h_smooth : MaxPool2Smooth x)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    DifferentiableAt ℝ (maxPoolFlat c h w) (Tensor3.flatten x) :=
  (maxPool2_flat_hasFDerivAt x h_smooth hc hh hw).differentiableAt

noncomputable def maxPoolFlat_has_vjp_at {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (h_smooth : MaxPool2Smooth x) :
    HasVJPAt (maxPoolFlat c h w) (Tensor3.flatten x) :=
  hasVJPAt3_to_hasVJPAt (maxPool2_has_vjp_at3 x h_smooth)

-- ════════════════════════════════════════════════════════════════
-- § MaxPool is exact in floating point (the float-bridge pass-through)
-- ════════════════════════════════════════════════════════════════

/-- **Max is exact in floating point + 1-Lipschitz.** `max a b` is a
    compare-and-select: it returns one of `a, b` verbatim, rounding nothing.
    So a float `max` over operands within `e` of the reals stays within `e` —
    the `max`-peer of `relu_close` (`FloatBridge.lean`), with no rounding term
    and no amplification. The one genuinely-new fact the MNIST-CNN forward
    rounding budget (planning §1b-A) needs beyond the dense/relu machinery. -/
theorem max_close {a b c d e : ℝ} (h1 : |a - c| ≤ e) (h2 : |b - d| ≤ e) :
    |max a b - max c d| ≤ e :=
  (abs_max_sub_max_le_max a b c d).trans (max_le h1 h2)

/-- **MaxPool2 is exact in floating point + 1-Lipschitz.** Four window cells
    through three `max`-selections, no arithmetic — inherited input error `e`
    passes through with no rounding term and no amplification. -/
theorem maxPool2_close {c h w : Nat} (xt xa : Tensor3 c (2*h) (2*w)) {e : ℝ}
    (hx : ∀ ci hi wi, |xt ci hi wi - xa ci hi wi| ≤ e)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    |maxPool2 xt ci hi wi - maxPool2 xa ci hi wi| ≤ e := by
  simp only [maxPool2]
  exact max_close (max_close (hx _ _ _) (hx _ _ _))
    (max_close (hx _ _ _) (hx _ _ _))

/-- Flattened `maxPoolFlat` peer of `maxPool2_close` — the form the
    `Vec`-space MNIST-CNN forward (`mnistCnnNoBnForward`) composes. -/
theorem maxPoolFlat_close {c h w : Nat} (vt va : Vec (c * (2*h) * (2*w)))
    {e : ℝ} (hv : ∀ k, |vt k - va k| ≤ e) (k : Fin (c * h * w)) :
    |maxPoolFlat c h w vt k - maxPoolFlat c h w va k| ≤ e := by
  have huf : ∀ ci hi wi,
      |Tensor3.unflatten vt ci hi wi - Tensor3.unflatten va ci hi wi| ≤ e := by
    intro ci hi wi
    simp only [Tensor3.unflatten]
    exact hv _
  simp only [maxPoolFlat, Tensor3.flatten]
  exact maxPool2_close (Tensor3.unflatten vt) (Tensor3.unflatten va) huf _ _ _

/-- `|max a b| ≤ A` when both operands are. -/
theorem abs_max_le {a b A : ℝ} (ha : |a| ≤ A) (hb : |b| ≤ A) : |max a b| ≤ A :=
  abs_max_le_max_abs_abs.trans (max_le ha hb)

/-- **MaxPool2 never grows magnitudes** (it selects an existing cell). -/
theorem maxPool2_abs_le {c h w : Nat} {x : Tensor3 c (2*h) (2*w)} {A : ℝ}
    (hx : ∀ ci hi wi, |x ci hi wi| ≤ A) (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    |maxPool2 x ci hi wi| ≤ A := by
  simp only [maxPool2]
  exact abs_max_le (abs_max_le (hx _ _ _) (hx _ _ _))
    (abs_max_le (hx _ _ _) (hx _ _ _))

/-- Flattened `maxPoolFlat` magnitude bound — the form the CNN forward threads. -/
theorem maxPoolFlat_abs_le {c h w : Nat} {v : Vec (c * (2*h) * (2*w))} {A : ℝ}
    (hv : ∀ k, |v k| ≤ A) (k : Fin (c * h * w)) :
    |maxPoolFlat c h w v k| ≤ A := by
  have huf : ∀ ci hi wi, |Tensor3.unflatten v ci hi wi| ≤ A := by
    intro ci hi wi
    simp only [Tensor3.unflatten]
    exact hv _
  simp only [maxPoolFlat, Tensor3.flatten]
  exact maxPool2_abs_le huf _ _ _

-- resblock (identity) output diffAt: relu ∘ residual F
theorem resblock_differentiableAt {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (c * h * w))
    (h_smooth₁ : ∀ k, bnForward (c * h * w) ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (c * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (c * h * w) ∘ bnForward (c * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v k
        + v k ≠ 0) :
    DifferentiableAt ℝ
      (relu (c * h * w) ∘
        residual
          ((bnForward (c * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
            (relu (c * h * w) ∘ bnForward (c * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))) v := by
  unfold residual
  fun_prop (disch := assumption)

-- resblockProj output diffAt: relu ∘ residualProj proj F
theorem resblockProj_differentiableAt
    {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp : ℝ)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * h * w))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp) v k)
      + ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v k
        ≠ 0) :
    DifferentiableAt ℝ
      (relu (oc * h * w) ∘
        residualProj
          (bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp)
          ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
            (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))) v := by
  unfold residualProj
  fun_prop (disch := assumption)

-- convBnRelu diffAt
theorem convBnRelu_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, bnForward (oc * h * w) ε γ β (flatConv W b v) k ≠ 0) :
    DifferentiableAt ℝ (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b) v := by
  fun_prop (disch := assumption)

-- abbreviations for layer functions
noncomputable abbrev cbr {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b

noncomputable abbrev rblk {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) : Vec (c * h * w) → Vec (c * h * w) :=
  relu (c * h * w) ∘ residual
    ((bnForward (c * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (c * h * w) ∘ bnForward (c * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))

noncomputable abbrev rblkP {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp : ℝ) : Vec (ic * h * w) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ residualProj
    (bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp)
    ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))

/-- The forward CNN: stem(convBnRelu) → maxpool → resblock(id) → resblockProj → gap → dense. -/
noncomputable def cnnForward
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip : ℝ)
    (Wd : Mat oc nClasses) (bd : Vec nClasses) :
    Vec (ic * (2*h) * (2*w)) → Vec nClasses :=
  (dense Wd bd) ∘
  (globalAvgPoolFlat oc h w) ∘
  (rblkP (h := h) (w := w) W₁' b₁' W₂' b₂' Wp bp e₁ g₁ bb₁ e₂ g₂ bb₂ fp hp ip) ∘
  (rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ h₁ i₁ f₂ h₂ i₂) ∘
  (maxPoolFlat c h w) ∘
  (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs)

noncomputable def cnn_has_vjp_at
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ) (he₁ : 0 < e₁) (he₂ : 0 < e₂)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip : ℝ) (hf₁ : 0 < f₁) (hf₂ : 0 < f₂) (hfp : 0 < fp)
    (Wd : Mat oc nClasses) (bd : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    -- stem smoothness
    (h_stem : ∀ k, bnForward (c * (2*h) * (2*w)) εs γs βs (flatConv Ws bs x) k ≠ 0)
    -- maxpool smoothness, on the stem output unflattened
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
              (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x) : Tensor3 c (2*h) (2*w)))
    -- identity resblock smoothness (at the maxpool output)
    (h_rb1 : ∀ k, bnForward (c * h * w) f₁ hh₁ i₁
        (flatConv W₁ b₁
          (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k ≠ 0)
    (h_rb1o : ∀ k,
        ((bnForward (c * h * w) f₂ hh₂ i₂ ∘ flatConv W₂ b₂) ∘
          (relu (c * h * w) ∘ bnForward (c * h * w) f₁ hh₁ i₁ ∘ flatConv W₁ b₁))
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x)) k
          + (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x)) k ≠ 0)
    -- proj resblock smoothness (at the identity resblock output)
    (h_rb2 : ∀ k, bnForward (oc * h * w) e₁ g₁ bb₁
        (flatConv (h := h) (w := w) W₁' b₁'
          ((rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) : Vec (c*h*w))) k ≠ 0)
    (h_rb2o : ∀ k,
        ((bnForward (oc * h * w) fp hhp ip ∘ flatConv (h := h) (w := w) Wp bp)
          (rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k)
        + ((bnForward (oc * h * w) e₂ g₂ bb₂ ∘ flatConv (h := h) (w := w) W₂' b₂') ∘
            (relu (oc * h * w) ∘ bnForward (oc * h * w) e₁ g₁ bb₁ ∘ flatConv (h := h) (w := w) W₁' b₁'))
            (rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
              (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k ≠ 0) :
    HasVJPAt (cnnForward Ws bs εs γs βs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂
                W₁' b₁' W₂' b₂' Wp bp f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip Wd bd) x := by
  unfold cnnForward
  -- s0: stem cbr at x
  set S0 := cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs with hS0def
  have s0_vjp : HasVJPAt S0 x :=
    convBnRelu_has_vjp_at Ws bs εs γs βs hεs x h_stem
  have s0_diff : DifferentiableAt ℝ S0 x :=
    convBnRelu_differentiableAt Ws bs εs γs βs hεs x h_stem
  -- s1: maxPoolFlat ∘ S0 at x; align maxpool point
  have hpt : Tensor3.flatten (Tensor3.unflatten (S0 x) : Tensor3 c (2*h) (2*w)) = S0 x :=
    Tensor3.flatten_unflatten (S0 x)
  have mp_vjp : HasVJPAt (maxPoolFlat c h w) (S0 x) := by
    rw [← hpt]; exact maxPoolFlat_has_vjp_at _ h_mp
  have mp_diff : DifferentiableAt ℝ (maxPoolFlat c h w) (S0 x) := by
    rw [← hpt]; exact maxPoolFlat_differentiableAt _ h_mp hc hh hw
  have s1_vjp : HasVJPAt (maxPoolFlat c h w ∘ S0) x :=
    vjp_comp_at S0 (maxPoolFlat c h w) x s0_diff mp_diff s0_vjp mp_vjp
  have s1_diff : DifferentiableAt ℝ (maxPoolFlat c h w ∘ S0) x :=
    mp_diff.comp x s0_diff
  -- s2: rblk ∘ (maxPoolFlat ∘ S0) at x
  set R1 := rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂ with hR1def
  have rb1_vjp : HasVJPAt R1 (maxPoolFlat c h w (S0 x)) :=
    resblock_has_vjp_at W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂ hf₁ hf₂ _ h_rb1 h_rb1o
  have rb1_diff : DifferentiableAt ℝ R1 (maxPoolFlat c h w (S0 x)) :=
    resblock_differentiableAt W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂ hf₁ hf₂ _ h_rb1 h_rb1o
  have s2_vjp : HasVJPAt (R1 ∘ (maxPoolFlat c h w ∘ S0)) x :=
    vjp_comp_at (maxPoolFlat c h w ∘ S0) R1 x s1_diff rb1_diff s1_vjp rb1_vjp
  have s2_diff : DifferentiableAt ℝ (R1 ∘ (maxPoolFlat c h w ∘ S0)) x :=
    rb1_diff.comp x s1_diff
  -- s3: rblkP ∘ (above) at x
  set R2 := rblkP (h := h) (w := w) W₁' b₁' W₂' b₂' Wp bp e₁ g₁ bb₁ e₂ g₂ bb₂ fp hhp ip with hR2def
  have rb2_vjp : HasVJPAt R2 (R1 (maxPoolFlat c h w (S0 x))) :=
    resblockProj_has_vjp_at W₁' b₁' W₂' b₂' Wp bp e₁ g₁ bb₁ e₂ g₂ bb₂ fp hhp ip
      he₁ he₂ hfp _ h_rb2 h_rb2o
  have rb2_diff : DifferentiableAt ℝ R2 (R1 (maxPoolFlat c h w (S0 x))) :=
    resblockProj_differentiableAt W₁' b₁' W₂' b₂' Wp bp e₁ g₁ bb₁ e₂ g₂ bb₂ fp hhp ip
      he₁ he₂ hfp _ h_rb2 h_rb2o
  have s3_vjp : HasVJPAt (R2 ∘ (R1 ∘ (maxPoolFlat c h w ∘ S0))) x :=
    vjp_comp_at (R1 ∘ (maxPoolFlat c h w ∘ S0)) R2 x s2_diff rb2_diff s2_vjp rb2_vjp
  have s3_diff : DifferentiableAt ℝ (R2 ∘ (R1 ∘ (maxPoolFlat c h w ∘ S0))) x :=
    rb2_diff.comp x s2_diff
  -- s4: gap ∘ (above) at x (global lift)
  set P3 := R2 ∘ (R1 ∘ (maxPoolFlat c h w ∘ S0)) with hP3def
  have gap_diff : DifferentiableAt ℝ (globalAvgPoolFlat oc h w) (P3 x) :=
    (globalAvgPoolFlat_differentiable oc h w) (P3 x)
  have s4_vjp : HasVJPAt (globalAvgPoolFlat oc h w ∘ P3) x :=
    vjp_comp_at P3 (globalAvgPoolFlat oc h w) x s3_diff gap_diff s3_vjp
      ((globalAvgPoolFlat_has_vjp oc h w).toHasVJPAt (P3 x))
  have s4_diff : DifferentiableAt ℝ (globalAvgPoolFlat oc h w ∘ P3) x :=
    gap_diff.comp x s3_diff
  -- s5: dense ∘ (above) at x (global lift)
  exact vjp_comp_at (globalAvgPoolFlat oc h w ∘ P3) (dense Wd bd) x s4_diff
    ((dense_differentiable Wd bd) _) s4_vjp
    ((dense_has_vjp Wd bd).toHasVJPAt _)

/-- **Public correctness theorem for `cnn_has_vjp_at`** — exposes the
    witness's `.correct` field as a top-level proposition: the full
    ResNet-style CNN's backward equals the `pdiv`-contracted Jacobian
    (Jacobian-transpose applied to the cotangent). CNN analogue of
    `vit_full_has_vjp_correct`. -/
theorem cnn_has_vjp_at_correct
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ) (he₁ : 0 < e₁) (he₂ : 0 < e₂)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip : ℝ) (hf₁ : 0 < f₁) (hf₂ : 0 < f₂) (hfp : 0 < fp)
    (Wd : Mat oc nClasses) (bd : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    (h_stem : ∀ k, bnForward (c * (2*h) * (2*w)) εs γs βs (flatConv Ws bs x) k ≠ 0)
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
              (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x) : Tensor3 c (2*h) (2*w)))
    (h_rb1 : ∀ k, bnForward (c * h * w) f₁ hh₁ i₁
        (flatConv W₁ b₁
          (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k ≠ 0)
    (h_rb1o : ∀ k,
        ((bnForward (c * h * w) f₂ hh₂ i₂ ∘ flatConv W₂ b₂) ∘
          (relu (c * h * w) ∘ bnForward (c * h * w) f₁ hh₁ i₁ ∘ flatConv W₁ b₁))
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x)) k
          + (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x)) k ≠ 0)
    (h_rb2 : ∀ k, bnForward (oc * h * w) e₁ g₁ bb₁
        (flatConv (h := h) (w := w) W₁' b₁'
          ((rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) : Vec (c*h*w))) k ≠ 0)
    (h_rb2o : ∀ k,
        ((bnForward (oc * h * w) fp hhp ip ∘ flatConv (h := h) (w := w) Wp bp)
          (rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k)
        + ((bnForward (oc * h * w) e₂ g₂ bb₂ ∘ flatConv (h := h) (w := w) W₂' b₂') ∘
            (relu (oc * h * w) ∘ bnForward (oc * h * w) e₁ g₁ bb₁ ∘ flatConv (h := h) (w := w) W₁' b₁'))
            (rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
              (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k ≠ 0)
    (dy : Vec nClasses) (i : Fin (ic * (2*h) * (2*w))) :
    (cnn_has_vjp_at Ws bs εs γs βs hεs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂ he₁ he₂
        W₁' b₁' W₂' b₂' Wp bp f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip hf₁ hf₂ hfp Wd bd
        hc hh hw x h_stem h_mp h_rb1 h_rb1o h_rb2 h_rb2o).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (cnnForward Ws bs εs γs βs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂
                W₁' b₁' W₂' b₂' Wp bp f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip Wd bd)
             x i j * dy j :=
  (cnn_has_vjp_at Ws bs εs γs βs hεs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂ he₁ he₂
      W₁' b₁' W₂' b₂' Wp bp f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip hf₁ hf₂ hfp Wd bd
      hc hh hw x h_stem h_mp h_rb1 h_rb1o h_rb2 h_rb2o).correct dy i

end Proofs
