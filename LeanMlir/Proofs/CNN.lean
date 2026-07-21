import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.Residual

/-!
# CNN VJP Proofs

VJP correctness for the convolutional and pooling layers used in
`mlir_poc/hand_cnn_train_step.mlir`. The architecture there is:

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

open Finset BigOperators Classical

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
  funext k
  -- After `change`, Lean's Prod struct-eta already collapses the innermost
  -- pair `(c.1, c.2)` to `c = fPF.symm (..).1`, so we start by collapsing
  -- that innermost `fPF (fPF.symm ...)` directly. Two more round-trips follow,
  -- each needing an explicit Prod-eta `show` + another `Equiv.apply_symm_apply`.
  change v (finProdFinEquiv
    (finProdFinEquiv
      (finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1),
       (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2),
     (finProdFinEquiv.symm k).2)) = v k
  rw [Equiv.apply_symm_apply]
  rw [show ((finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1,
            (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2) =
           finProdFinEquiv.symm (finProdFinEquiv.symm k).1 from rfl,
      Equiv.apply_symm_apply]
  rw [show ((finProdFinEquiv.symm k).1, (finProdFinEquiv.symm k).2) =
           finProdFinEquiv.symm k from rfl,
      Equiv.apply_symm_apply]

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

/-- **Conv2d is differentiable everywhere.** Each output coordinate
    `conv2d W b x o hi wi` is the affine map
    `b o + ∑_{c,kh,kw} W o c kh kw · (pad-eval x)`: a constant bias plus a
    finite ℝ-linear combination of input coordinates (the dependent
    `if`-pad-eval being either a projection or the constant `0`).
    `differentiable_pi` reduces to per-coordinate differentiability;
    `DifferentiableAt.fun_sum` lifts the triple sum, and each pad-eval
    summand is a projection (pad true) or constant (pad false). -/
theorem conv2d_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (conv2d W b : Tensor3 ic h w → Tensor3 oc h w) := by
  apply differentiable_pi.mpr; intro o
  apply differentiable_pi.mpr; intro hi
  apply differentiable_pi.mpr; intro wi
  show Differentiable ℝ (fun x : Tensor3 ic h w =>
    b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      W o c kh kw *
        (let pH := (kH - 1) / 2
         let pW := (kW - 1) / 2
         let hh := kh.val + hi.val
         let ww := kw.val + wi.val
         if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
           x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
         else 0))
  apply Differentiable.const_add
  intro x
  apply DifferentiableAt.fun_sum; intro c _
  apply DifferentiableAt.fun_sum; intro kh _
  apply DifferentiableAt.fun_sum; intro kw _
  apply DifferentiableAt.const_mul
  set pH := (kH - 1) / 2
  set pW := (kW - 1) / 2
  set hh := kh.val + hi.val
  set ww := kw.val + wi.val
  by_cases hP : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w
  · rw [show (fun x : Tensor3 ic h w =>
          if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
            x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩ else 0) =
        (fun x : Tensor3 ic h w => x c ⟨hh - pH, hP.2.1⟩ ⟨ww - pW, hP.2.2.2⟩) from by
      funext x; rw [dif_pos hP]]
    fun_prop
  · rw [show (fun x : Tensor3 ic h w =>
          if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
            x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩ else 0) =
        (fun _ : Tensor3 ic h w => (0 : ℝ)) from by funext x; rw [dif_neg hP]]
    exact differentiableAt_const _

/-- **Differentiability of an `if hpad : P then v(σ hpad) else 0` term.**
    The dependent-`if` would otherwise stymie `fun_prop`. By proof
    irrelevance, the chosen branch is a CLM (eval-at-σ if `P`, the
    constant `0` otherwise) — both differentiable in `v`. -/
lemma differentiableAt_pad_eval {n : Nat} (P : Prop) [Decidable P]
    (σ : P → Fin n) (v : Vec n) :
    DifferentiableAt ℝ (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) v := by
  by_cases hP : P
  · rw [show (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) =
            (fun y => y (σ hP)) from by funext y; rw [dif_pos hP]]
    fun_prop
  · rw [show (fun y : Vec n => if h : P then y (σ h) else (0 : ℝ)) =
            (fun _ => (0 : ℝ)) from by funext y; rw [dif_neg hP]]
    exact differentiableAt_const _

/-- **Pdiv of a per-output dependent if-eval-or-zero family.**
    Given a per-output dependent if `fun v k' ↦ if h : P k' then v (σ k' h) else 0`,
    its pdiv at `(idx_in, idx_out)` is the indicator
    `if P holds at idx_out ∧ σ matches idx_in then 1 else 0`. The proof uses
    `fderiv_apply` to extract the `idx_out`-th component, then `by_cases` on
    `P idx_out` to discharge the dependent-if. -/
lemma pdiv_pi_pad_eval {n m : Nat}
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            if h : P k' then v' (σ k' h) else (0 : ℝ))
          v idx_in idx_out =
    if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0 := by
  unfold pdiv
  have h_diff_pi : DifferentiableAt ℝ (fun (v' : Vec n) (k' : Fin m) =>
      if h : P k' then v' (σ k' h) else (0 : ℝ)) v := by
    rw [differentiableAt_pi]
    intro k'
    exact differentiableAt_pad_eval (P k') (σ k') v
  rw [show fderiv ℝ (fun (v' : Vec n) (k' : Fin m) =>
              if h : P k' then v' (σ k' h) else (0 : ℝ)) v (basisVec idx_in) idx_out
        = fderiv ℝ (fun v' : Vec n =>
            (fun v'' k' => if h : P k' then v'' (σ k' h) else (0 : ℝ)) v' idx_out)
            v (basisVec idx_in) from by
    rw [fderiv_apply h_diff_pi idx_out]; rfl]
  by_cases hpad : P idx_out
  · rw [show (fun v' : Vec n =>
          (fun v'' k' => if h : P k' then v'' (σ k' h) else (0 : ℝ)) v' idx_out) =
        (fun v' : Vec n => v' (σ idx_out hpad)) from by
      funext v'
      show (if h : P idx_out then v' (σ idx_out h) else (0 : ℝ)) = v' (σ idx_out hpad)
      rw [dif_pos hpad]]
    rw [show (fun v' : Vec n => v' (σ idx_out hpad)) =
          ((ContinuousLinearMap.proj (σ idx_out hpad) : Vec n →L[ℝ] ℝ) : Vec n → ℝ)
        from rfl]
    rw [ContinuousLinearMap.fderiv]
    show (ContinuousLinearMap.proj (σ idx_out hpad) : Vec n →L[ℝ] ℝ) (basisVec idx_in) = _
    rw [ContinuousLinearMap.proj_apply, basisVec_apply, dif_pos hpad]
  · rw [show (fun v' : Vec n =>
          (fun v'' k' => if h : P k' then v'' (σ k' h) else (0 : ℝ)) v' idx_out) =
        (fun _ => (0 : ℝ)) from by
      funext v'
      show (if h : P idx_out then v' (σ idx_out h) else (0 : ℝ)) = 0
      rw [dif_neg hpad]]
    rw [(hasFDerivAt_const (0 : ℝ) v).fderiv, dif_neg hpad]
    rfl

/-- **Pdiv of `c_const * pad-eval` family.** Combines `pdiv_mul`,
    `pdiv_const`, and `pdiv_pi_pad_eval` for the conv2d per-summand
    pattern: a `k'`-varying constant times the dependent if-eval-or-zero. -/
lemma pdiv_const_mul_pi_pad_eval {n m : Nat}
    (c_const : Fin m → ℝ)
    (P : Fin m → Prop) [∀ k, Decidable (P k)]
    (σ : (k : Fin m) → P k → Fin n)
    (v : Vec n) (idx_in : Fin n) (idx_out : Fin m) :
    pdiv (fun (v' : Vec n) (k' : Fin m) =>
            c_const k' *
            (if h : P k' then v' (σ k' h) else (0 : ℝ)))
          v idx_in idx_out =
    c_const idx_out *
    (if h : P idx_out then (if σ idx_out h = idx_in then (1 : ℝ) else 0) else 0) := by
  have h_const_diff : DifferentiableAt ℝ
      (fun (_ : Vec n) (k' : Fin m) => c_const k') v := differentiableAt_const _
  have h_pad_diff : DifferentiableAt ℝ
      (fun (v' : Vec n) (k' : Fin m) =>
        if h : P k' then v' (σ k' h) else (0 : ℝ)) v := by
    rw [differentiableAt_pi]
    intro k'
    exact differentiableAt_pad_eval (P k') (σ k') v
  rw [pdiv_mul _ _ _ h_const_diff h_pad_diff]
  rw [show pdiv (fun (_ : Vec n) (k' : Fin m) => c_const k') v idx_in idx_out = 0
      from pdiv_const _ _ _ _]
  rw [pdiv_pi_pad_eval]
  ring

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
    `v`: a constant `b o(idx_out)` plus a triple sum over `(c, kh, kw)`
    of `W o(idx_out) c kh kw * (if pad-cond then v(reindex) else 0)`.
    Each summand factors as `(constant W) * (if-pad-conditional in v)`,
    so `pdiv_add` + `pdiv_const` + `pdiv_finset_sum` (×3) + `pdiv_mul` +
    a `by_cases` on the pad condition (CLM-projection on the pad-true
    branch, constant zero on the pad-false branch) collapse the
    per-`(idx_in, idx_out)` pdiv. Reindex `Fin (oc*h*w) ↔ Fin oc × Fin h × Fin w`
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
      -- Set abbreviations for idx_out components (idx_out unpacks to (ohw_o, ohw_hi, ohw_wi)).
      set ohw_wi : Fin w := (finProdFinEquiv.symm idx_out).2 with hohw_wi
      set ohw_hi : Fin h :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2 with hohw_hi
      set ohw_o : Fin oc :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 with hohw_o
      -- Decompose `f = (constant b) + (sum over c kh kw of W * if-pad-cond)`.
      rw [show (fun v' : Vec (ic * h * w) =>
                Tensor3.flatten (conv2d W b (Tensor3.unflatten v'))) =
            (fun v' k =>
              (fun (_ : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v' k +
              (fun (v'' : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'') c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v' k) from by
        funext v' k
        unfold Tensor3.flatten conv2d
        rfl]
      have h_b_diff : DifferentiableAt ℝ
          (fun (_ : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
            b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) (Tensor3.flatten x) :=
        differentiableAt_const _
      have h_lin_diff : DifferentiableAt ℝ
          (fun (v'' : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
            ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
              W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
               let ww := kw.val + (finProdFinEquiv.symm k').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 (Tensor3.unflatten v'') c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) (Tensor3.flatten x) := by
        rw [differentiableAt_pi]
        intro k'
        apply DifferentiableAt.fun_sum; intro c _
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            (c, ⟨kh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kw.val + (finProdFinEquiv.symm k').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv (fun (_ : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                  b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
                (Tensor3.flatten x) idx_in idx_out = 0
          from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute pdiv over the c-sum.
      rw [show (fun (v'' : Vec (ic * h * w)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'') c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
            (fun v'' k' => ∑ c : Fin ic,
              (fun (cc : Fin ic) (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) c v'' k') from rfl]
      have h_c_diff : ∀ cc ∈ (Finset.univ : Finset (Fin ic)),
          DifferentiableAt ℝ
            (fun (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
              ∑ kh : Fin kH, ∑ kw : Fin kW,
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) (Tensor3.flatten x) := by
        intro cc _
        rw [differentiableAt_pi]
        intro k''
        apply DifferentiableAt.fun_sum; intro kh _
        apply DifferentiableAt.fun_sum; intro kw _
        apply DifferentiableAt.mul (differentiableAt_const _)
        unfold Tensor3.unflatten
        exact differentiableAt_pad_eval _
          (fun hpad => finProdFinEquiv (finProdFinEquiv
            (cc, ⟨kh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kw.val + (finProdFinEquiv.symm k'').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
      rw [pdiv_finset_sum _ _ _ h_c_diff]
      -- Now distribute over kh and kw inside each c-summand.
      have h_inner_c : ∀ cc : Fin ic,
          pdiv (fun (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
                  ∑ kh : Fin kH, ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) (Tensor3.flatten x) idx_in idx_out =
          ∑ kh : Fin kH, ∑ kw : Fin kW,
            W ohw_o cc kh kw *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val + ohw_hi.val
             let ww := kw.val + ohw_wi.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               (if idx_in = finProdFinEquiv (finProdFinEquiv
                   (cc, ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then (1 : ℝ) else 0)
             else 0) := by
        intro cc
        -- Distribute over kh.
        rw [show (fun (v''' : Vec (ic * h * w)) (k'' : Fin (oc * h * w)) =>
                  ∑ kh : Fin kH, ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v''' k'' => ∑ kh : Fin kH,
                (fun (khh : Fin kH) (v'''' : Vec (ic * h * w)) (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v'''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) kh v''' k'') from rfl]
        have h_kh_diff : ∀ khh ∈ (Finset.univ : Finset (Fin kH)),
            DifferentiableAt ℝ
              (fun (v'''' : Vec (ic * h * w)) (k''' : Fin (oc * h * w)) =>
                ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) cc khh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v'''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) (Tensor3.flatten x) := by
          intro khh _
          rw [differentiableAt_pi]
          intro k'''
          apply DifferentiableAt.fun_sum; intro kw _
          apply DifferentiableAt.mul (differentiableAt_const _)
          unfold Tensor3.unflatten
          exact differentiableAt_pad_eval _
            (fun hpad => finProdFinEquiv (finProdFinEquiv
              (cc, ⟨khh.val + (finProdFinEquiv.symm
                (finProdFinEquiv.symm k''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kw.val + (finProdFinEquiv.symm k''').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
        rw [pdiv_finset_sum _ _ _ h_kh_diff]
        congr 1; ext khh
        -- Distribute over kw.
        rw [show (fun (v'''' : Vec (ic * h * w)) (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1) cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         (Tensor3.unflatten v'''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v'''' k''' => ∑ kw : Fin kW,
                (fun (kww : Fin kW) (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1) cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) kw v'''' k''') from rfl]
        have h_kw_diff : ∀ kww ∈ (Finset.univ : Finset (Fin kW)),
            DifferentiableAt ℝ
              (fun (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1) cc khh kww *
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     (Tensor3.unflatten v''''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) (Tensor3.flatten x) := by
          intro kww _
          rw [differentiableAt_pi]
          intro k''''
          apply DifferentiableAt.mul (differentiableAt_const _)
          unfold Tensor3.unflatten
          exact differentiableAt_pad_eval _
            (fun hpad => finProdFinEquiv (finProdFinEquiv
              (cc, ⟨khh.val + (finProdFinEquiv.symm
                (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2, hpad.2.2.2⟩)) _
        rw [pdiv_finset_sum _ _ _ h_kw_diff]
        congr 1; ext kww
        -- Per-(cc, khh, kww) summand: factor as (W constant) * (dite in v).
        -- After unfolding `Tensor3.unflatten`, the inner becomes
        -- `if hpad : ... then v(σ) else 0`, fitting `pdiv_const_mul_pi_pad_eval`.
        rw [show (fun (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1) cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       (Tensor3.unflatten v''''') cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
              (fun (v''''' : Vec (ic * h * w)) (k'''' : Fin (oc * h * w)) =>
                (fun k''''' : Fin (oc * h * w) =>
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1) cc khh kww) k'''' *
                (if hpad : (kH - 1) / 2 ≤ khh.val + (finProdFinEquiv.symm
                              (finProdFinEquiv.symm k'''').1).2.val ∧
                            khh.val + (finProdFinEquiv.symm
                              (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2 < h ∧
                            (kW - 1) / 2 ≤ kww.val + (finProdFinEquiv.symm k'''').2.val ∧
                            kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2 < w then
                  v''''' (finProdFinEquiv (finProdFinEquiv
                    (cc, ⟨khh.val + (finProdFinEquiv.symm
                            (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
                    ⟨kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2,
                      hpad.2.2.2⟩))
                else 0)) from by
          funext v''''' k''''
          unfold Tensor3.unflatten
          rfl]
        rw [pdiv_const_mul_pi_pad_eval
          (fun k''''' : Fin (oc * h * w) =>
            W ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1) cc khh kww)
          (fun k'''' => (kH - 1) / 2 ≤ khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'''').1).2.val ∧
            khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2 < h ∧
            (kW - 1) / 2 ≤ kww.val + (finProdFinEquiv.symm k'''').2.val ∧
            kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2 < w)
          (fun k'''' hpad => finProdFinEquiv (finProdFinEquiv
            (cc, ⟨khh.val + (finProdFinEquiv.symm
              (finProdFinEquiv.symm k'''').1).2.val - (kH - 1) / 2, hpad.2.1⟩),
            ⟨kww.val + (finProdFinEquiv.symm k'''').2.val - (kW - 1) / 2, hpad.2.2.2⟩))]
        -- Show the result matches the desired form (with ohw_o, ohw_hi, ohw_wi abbreviations).
        show W ohw_o cc khh kww * _ = W ohw_o cc khh kww * _
        congr 1
        -- Goal: dite-form on the LHS = dite-form on the RHS (after symmetrizing the
        -- equality direction: σ idx_out h = idx_in vs idx_in = ...).
        by_cases hpad : (kH - 1) / 2 ≤ khh.val + ohw_hi.val ∧
                       khh.val + ohw_hi.val - (kH - 1) / 2 < h ∧
                       (kW - 1) / 2 ≤ kww.val + ohw_wi.val ∧
                       kww.val + ohw_wi.val - (kW - 1) / 2 < w
        · rw [dif_pos hpad, dif_pos hpad]
          by_cases heq : finProdFinEquiv (finProdFinEquiv
              (cc, ⟨khh.val + ohw_hi.val - (kH - 1) / 2, hpad.2.1⟩),
              ⟨kww.val + ohw_wi.val - (kW - 1) / 2, hpad.2.2.2⟩) = idx_in
          · rw [if_pos heq, if_pos heq.symm]
          · rw [if_neg heq, if_neg (fun h => heq h.symm)]
        · rw [dif_neg hpad, dif_neg hpad]
      simp_rw [h_inner_c]
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
      · simp only [dif_pos hb]
      · simp only [dif_neg hb, zero_mul]]
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
      · rw [dif_pos hpad]
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
          rw [if_pos h_idx_in_eq, if_pos h_match]
        · rw [if_neg h_match]
          rw [if_neg]
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
      · rw [dif_neg hpad]
        rw [if_neg]
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
    · rw [dif_pos hb]
      -- Σ c collapses on c = ci, then Σ kh on kh = ⟨hi+pH-ho, hb.2.1⟩, then Σ kw similarly.
      symm
      rw [Finset.sum_eq_single ci ?_ ?_]
      rw [Finset.sum_eq_single ⟨hi.val + (kH - 1) / 2 - ho.val, hb.2.1⟩ ?_ ?_]
      rw [Finset.sum_eq_single ⟨wi.val + (kW - 1) / 2 - wo.val, hb.2.2.2⟩ ?_ ?_]
      · -- Main: W co ci ⟨kh*⟩ ⟨kw*⟩ * (if ci=ci ∧ ... then 1 else 0) = W co ci ⟨kh*⟩ ⟨kw*⟩
        rw [if_pos]
        · ring
        refine ⟨rfl, ?_, ?_⟩
        · show hi.val + (kH - 1) / 2 - ho.val + ho.val = hi.val + (kH - 1) / 2
          omega
        · show wi.val + (kW - 1) / 2 - wo.val + wo.val = wi.val + (kW - 1) / 2
          omega
      · intro kw _ hkw_ne
        rw [if_neg ?_]; · ring
        intro ⟨_, _, hkw_eq⟩
        apply hkw_ne
        apply Fin.ext
        show kw.val = wi.val + (kW - 1) / 2 - wo.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
      · intro kh _ hkh_ne
        apply Finset.sum_eq_zero; intro kw _
        rw [if_neg ?_]; · ring
        intro ⟨_, hkh_eq, _⟩
        apply hkh_ne
        apply Fin.ext
        show kh.val = hi.val + (kH - 1) / 2 - ho.val
        omega
      · intro hni; exact absurd (Finset.mem_univ _) hni
      · intro c _ hc_ne
        apply Finset.sum_eq_zero; intro kh _
        apply Finset.sum_eq_zero; intro kw _
        rw [if_neg (fun ⟨hcc, _, _⟩ => hc_ne hcc)]; ring
      · intro hni; exact absurd (Finset.mem_univ ci) hni
    · rw [dif_neg hb]
      -- Show the inner sum is 0: for !back_cond, no (c, kh, kw) satisfies the indicator.
      symm
      apply Finset.sum_eq_zero; intro c _
      apply Finset.sum_eq_zero; intro kh _
      apply Finset.sum_eq_zero; intro kw _
      rw [if_neg ?_]; · ring
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
theorem flatConv_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (flatConv W b : Vec (ic * h * w) → Vec (oc * h * w)) :=
  Tensor3.flatten_differentiable.comp
    ((conv2d_differentiable W b).comp Tensor3.unflatten_differentiable)

/-- **conv → bn → relu block VJP at a smooth point.**

    The workhorse for composing a ResNet VJP. In flattened `Vec` space,
    the block is `relu ∘ bnForward ∘ flatConv : Vec (ic*h*w) → Vec (oc*h*w)`
    (BatchNorm runs over the `oc*h*w` flattened activations with scalar
    `ε, γ, β`). We build `HasVJPAt` at a point `v` via two `vjp_comp_at`:

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
    HasVJPAt (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b) v := by
  -- inner step: bnForward ∘ flatConv (both differentiable everywhere)
  have hconv_diff : Differentiable ℝ (flatConv W b : Vec (ic * h * w) → Vec (oc * h * w)) :=
    flatConv_differentiable W b
  have hbn_diff : Differentiable ℝ (bnForward (oc * h * w) ε γ β) :=
    bnForward_differentiable (oc * h * w) ε γ β hε
  have step1 : HasVJPAt (bnForward (oc * h * w) ε γ β ∘ flatConv W b) v :=
    vjp_comp_at (flatConv W b) (bnForward (oc * h * w) ε γ β) v
      (hconv_diff v)
      (hbn_diff _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).toHasVJPAt v)
      ((bn_has_vjp (oc * h * w) ε γ β hε).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnForward (oc * h * w) ε γ β ∘ flatConv W b) v :=
    DifferentiableAt.comp v (hbn_diff (flatConv W b v)) (hconv_diff v)
  -- outer step: relu (needs smoothness)
  exact vjp_comp_at (bnForward (oc * h * w) ε γ β ∘ flatConv W b)
    (relu (oc * h * w)) v
    step1_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    step1
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
  apply DifferentiableAt.comp
  · exact (convBn_differentiable W₂ b₂ ε₂ γ₂ β₂ hε₂) _
  · apply DifferentiableAt.comp
    · exact relu_differentiableAt_of_smooth (mid * h * w) _ h_smooth₁
    · exact ((bnForward_differentiable (mid * h * w) ε₁ γ₁ β₁ hε₁).comp
        (flatConv_differentiable W₁ b₁)) v

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
  have hres_diff : DifferentiableAt ℝ (residual F) v := by
    show DifferentiableAt ℝ (biPath F (fun x => x)) v
    exact DifferentiableAt.add hF_diff differentiable_id.differentiableAt
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
    `v`: a constant `b o(idx_out)` plus a triple sum over `(c, kh, kw)`
    of `(unflatten v) o(idx_out) c kh kw * x_pad_term`. Each summand
    factors as `(reindex of v) * (x-only constant)`, so `pdiv_add` +
    `pdiv_const` + `pdiv_finset_sum` (×3) + `pdiv_mul` + `pdiv_reindex`
    collapse the per-(idx_in, idx_out) pdiv. Triple-sum collapse via
    `Finset.sum_eq_single` gives the transpose-trick backward
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
    -- Set abbreviations for the unpacked idx_in components (kernel position).
    set kw' : Fin kW := (finProdFinEquiv.symm idx_in).2 with hkw'
    set kh' : Fin kH :=
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2 with hkh'
    set o' : Fin oc := (finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).1 with ho'
    set c' : Fin ic := (finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).2 with hc'
    -- Identity: idx_in unpacks to (o', c', kh', kw') and re-packs back.
    have h_idx_in_eq :
        finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o', c'), kh'), kw') = idx_in := by
      rw [ho', hc', hkh', hkw']
      rw [show ((finProdFinEquiv.symm
                  (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).1,
                (finProdFinEquiv.symm
                  (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1).2) =
               finProdFinEquiv.symm
                  (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1 from rfl,
          Equiv.apply_symm_apply]
      rw [show ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1,
                (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2) =
               finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1 from rfl,
          Equiv.apply_symm_apply]
      rw [show ((finProdFinEquiv.symm idx_in).1, (finProdFinEquiv.symm idx_in).2) =
               finProdFinEquiv.symm idx_in from rfl,
          Equiv.apply_symm_apply]
    -- Step 1: per-idx_out pdiv formula.
    have h_pdiv : ∀ idx_out : Fin (oc * h * w),
        pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v idx_in idx_out =
        (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 = o' then
          (let pH := (kH - 1) / 2
           let pW := (kW - 1) / 2
           let hh := kh'.val +
             (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
           let ww := kw'.val + (finProdFinEquiv.symm idx_out).2.val
           if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
             x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
           else 0)
         else 0) := by
      intro idx_out
      -- Set abbreviations for idx_out components, prevents auto-resugaring.
      set ohw_wi : Fin w := (finProdFinEquiv.symm idx_out).2 with hohw_wi
      set ohw_hi : Fin h :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2 with hohw_hi
      set ohw_o : Fin oc :=
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 with hohw_o
      -- Decompose f = (constant b) + (sum over c kh kw of (reindex_v * x_pad_const)).
      rw [show (fun v' : Vec (oc * ic * kH * kW) =>
                Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) =
            (fun v' k =>
              (fun (_ : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v' k +
              (fun (v'' : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v'')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v' k) from by
        funext v' k
        unfold Tensor3.flatten conv2d
        rfl]
      have h_b_diff : DifferentiableAt ℝ
          (fun (_ : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
            b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) v :=
        differentiableAt_const _
      have h_lin_diff : DifferentiableAt ℝ
          (fun (v'' : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
            ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
              (Kernel4.unflatten v'')
                ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
               let ww := kw.val + (finProdFinEquiv.symm k').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) v := by
        unfold Kernel4.unflatten; fun_prop
      rw [pdiv_add _ _ _ h_b_diff h_lin_diff]
      rw [show pdiv (fun (_ : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                  b ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1))
                v idx_in idx_out = 0
          from pdiv_const _ _ _ _]
      rw [zero_add]
      -- Distribute over the triple sum.
      rw [show (fun (v'' : Vec (oc * ic * kH * kW)) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v'')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
            (fun v'' k' => ∑ c : Fin ic,
              (fun (cc : Fin ic) (v''' : Vec (oc * ic * kH * kW))
                  (k'' : Fin (oc * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) c v'' k') from rfl]
      have h_c_summand_diff : ∀ cc ∈ (Finset.univ : Finset (Fin ic)),
          DifferentiableAt ℝ
            (fun (v''' : Vec (oc * ic * kH * kW)) (k'' : Fin (oc * h * w)) =>
              ∑ kh : Fin kH, ∑ kw : Fin kW,
                (Kernel4.unflatten v''')
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) v := by
        intro cc _; unfold Kernel4.unflatten; fun_prop
      rw [pdiv_finset_sum _ _ _ h_c_summand_diff]
      have h_inner_c : ∀ cc : Fin ic,
          pdiv (fun (v''' : Vec (oc * ic * kH * kW))
                    (k'' : Fin (oc * h * w)) =>
                ∑ kh : Fin kH, ∑ kw : Fin kW,
                  (Kernel4.unflatten v''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) cc kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) v idx_in idx_out =
          ∑ kh : Fin kH, ∑ kw : Fin kW,
            (if idx_in = finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              (ohw_o, cc), kh), kw)
              then (1 : ℝ) else 0) *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val + ohw_hi.val
             let ww := kw.val + ohw_wi.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0) := by
        intro cc
        rw [show (fun (v''' : Vec (oc * ic * kH * kW))
                      (k'' : Fin (oc * h * w)) =>
                  ∑ kh : Fin kH, ∑ kw : Fin kW,
                    (Kernel4.unflatten v''')
                      ((finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)
                      cc kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k'').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v''' k'' => ∑ kh : Fin kH,
                (fun (khh : Fin kH) (v'''' : Vec (oc * ic * kH * kW))
                    (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    (Kernel4.unflatten v'''')
                      ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                      cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) kh v''' k'') from rfl]
        have h_kh_summand_diff : ∀ khh ∈ (Finset.univ : Finset (Fin kH)),
            DifferentiableAt ℝ
              (fun (v'''' : Vec (oc * ic * kH * kW)) (k''' : Fin (oc * h * w)) =>
                ∑ kw : Fin kW,
                  (Kernel4.unflatten v'''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                    cc khh kw *
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                   let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) v := by
          intro khh _; unfold Kernel4.unflatten; fun_prop
        rw [pdiv_finset_sum _ _ _ h_kh_summand_diff]
        congr 1; ext khh
        rw [show (fun (v'''' : Vec (oc * ic * kH * kW))
                      (k''' : Fin (oc * h * w)) =>
                  ∑ kw : Fin kW,
                    (Kernel4.unflatten v'''')
                      ((finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).1)
                      cc khh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := khh.val +
                         (finProdFinEquiv.symm (finProdFinEquiv.symm k''').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k''').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0)) =
              (fun v'''' k''' => ∑ kw : Fin kW,
                (fun (kww : Fin kW) (v''''' : Vec (oc * ic * kH * kW))
                    (k'''' : Fin (oc * h * w)) =>
                  (Kernel4.unflatten v''''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                    cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) kw v'''' k''') from rfl]
        have h_kw_summand_diff : ∀ kww ∈ (Finset.univ : Finset (Fin kW)),
            DifferentiableAt ℝ
              (fun (v''''' : Vec (oc * ic * kH * kW)) (k'''' : Fin (oc * h * w)) =>
                (Kernel4.unflatten v''''')
                  ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                  cc khh kww *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := khh.val +
                   (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                 let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) v := by
          intro kww _; unfold Kernel4.unflatten; fun_prop
        rw [pdiv_finset_sum _ _ _ h_kw_summand_diff]
        congr 1; ext kww
        -- Per-summand: factor as (reindex v) * (constant in v).
        rw [show (fun (v''''' : Vec (oc * ic * kH * kW))
                      (k'''' : Fin (oc * h * w)) =>
                  (Kernel4.unflatten v''''')
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).1)
                    cc khh kww *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := khh.val +
                       (finProdFinEquiv.symm (finProdFinEquiv.symm k'''').1).2.val
                     let ww := kww.val + (finProdFinEquiv.symm k'''').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) =
              (fun v''''' k'''' =>
                (fun (v'''''' : Vec (oc * ic * kH * kW))
                    (k''''' : Fin (oc * h * w)) =>
                  v'''''' (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                      khh), kww))) v''''' k'''' *
                (fun (_ : Vec (oc * ic * kH * kW))
                    (k''''' : Fin (oc * h * w)) =>
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k''''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0)) v''''' k'''') from by
          funext v''''' k''''
          unfold Kernel4.unflatten
          rfl]
        have h_reindex_diff : DifferentiableAt ℝ
            (fun (v'''''' : Vec (oc * ic * kH * kW))
                 (k''''' : Fin (oc * h * w)) =>
              v'''''' (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                  khh), kww))) v :=
          (reindexCLM (fun k''''' : Fin (oc * h * w) =>
            finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                khh), kww))).differentiableAt
        have h_xpad_const_diff : DifferentiableAt ℝ
            (fun (_ : Vec (oc * ic * kH * kW))
                 (k''''' : Fin (oc * h * w)) =>
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := khh.val +
                 (finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).2.val
               let ww := kww.val + (finProdFinEquiv.symm k''''').2.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)) v :=
          differentiableAt_const _
        rw [pdiv_mul _ _ _ h_reindex_diff h_xpad_const_diff]
        rw [show (fun (v'''''' : Vec (oc * ic * kH * kW))
                      (k''''' : Fin (oc * h * w)) =>
                  v'''''' (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).1, cc),
                      khh), kww))) =
              (fun y k''''' =>
                y ((fun k'''''' : Fin (oc * h * w) =>
                  finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                    ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''''').1).1, cc),
                      khh), kww)) k''''')) from rfl]
        rw [pdiv_reindex (fun k'''''' : Fin (oc * h * w) =>
            finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              ((finProdFinEquiv.symm (finProdFinEquiv.symm k'''''').1).1, cc),
                khh), kww))]
        rw [show pdiv (fun (_ : Vec (oc * ic * kH * kW))
                          (k''''' : Fin (oc * h * w)) =>
                  (let pH := (kH - 1) / 2
                   let pW := (kW - 1) / 2
                   let hh := khh.val +
                     (finProdFinEquiv.symm (finProdFinEquiv.symm k''''').1).2.val
                   let ww := kww.val + (finProdFinEquiv.symm k''''').2.val
                   if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                     x cc ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                   else 0))
                v idx_in idx_out = 0
            from pdiv_const _ _ _ _]
        ring
      simp_rw [h_inner_c]
      -- Now: ∑ c kh kw, (if idx_in = ker_idx(ohw_o, c, kh, kw) then 1 else 0) * xpad
      --      = if ohw_o = o' then xpad(c', kh', kw') else 0.
      -- Rewrite indicator: by injectivity, idx_in = ker_idx(ohw_o, c, kh, kw)
      -- iff (ohw_o, c, kh, kw) = (o', c', kh', kw').
      have h_indicator : ∀ c : Fin ic, ∀ kh : Fin kH, ∀ kw : Fin kW,
          (idx_in = finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
              (ohw_o, c), kh), kw)) ↔
          (ohw_o = o' ∧ c = c' ∧ kh = kh' ∧ kw = kw') := by
        intro c kh kw
        constructor
        · intro h
          rw [← h_idx_in_eq] at h
          have hpair := finProdFinEquiv.injective h
          have hpair2 := finProdFinEquiv.injective (Prod.mk.inj hpair).1
          have hpair3 := finProdFinEquiv.injective (Prod.mk.inj hpair2).1
          refine ⟨?_, ?_, ?_, ?_⟩
          · exact (Prod.mk.inj hpair3).1.symm
          · exact (Prod.mk.inj hpair3).2.symm
          · exact (Prod.mk.inj hpair2).2.symm
          · exact (Prod.mk.inj hpair).2.symm
        · rintro ⟨ho_eq, hc_eq, hkh_eq, hkw_eq⟩
          rw [← h_idx_in_eq, ← ho_eq, ← hc_eq, ← hkh_eq, ← hkw_eq]
      simp_rw [h_indicator]
      -- Triple sum collapse via Finset.sum_eq_single (3 levels deep).
      -- For c ≠ c': inner sum is 0 since the conjunction (... ∧ c = c' ∧ ...) is false.
      rw [Finset.sum_eq_single c'
            (fun c _ hc_ne =>
              Finset.sum_eq_zero (fun kh _ =>
                Finset.sum_eq_zero (fun kw _ => by
                  rw [if_neg (fun ⟨_, hc, _, _⟩ => hc_ne hc), zero_mul])))
            (fun hni => absurd (Finset.mem_univ c') hni)]
      rw [Finset.sum_eq_single kh'
            (fun kh _ hkh_ne =>
              Finset.sum_eq_zero (fun kw _ => by
                rw [if_neg (fun ⟨_, _, hkh, _⟩ => hkh_ne hkh), zero_mul]))
            (fun hni => absurd (Finset.mem_univ kh') hni)]
      rw [Finset.sum_eq_single kw'
            (fun kw _ hkw_ne => by
              rw [if_neg (fun ⟨_, _, _, hkw⟩ => hkw_ne hkw), zero_mul])
            (fun hni => absurd (Finset.mem_univ kw') hni)]
      -- Final: (if (ohw_o = o' ∧ c' = c' ∧ kh' = kh' ∧ kw' = kw') then 1 else 0) * xpad(c', kh', kw')
      --        = if ohw_o = o' then xpad(c', kh', kw') else 0
      by_cases h_o : ohw_o = o'
      · rw [if_pos ⟨h_o, rfl, rfl, rfl⟩, one_mul, if_pos h_o]
      · rw [if_neg (fun ⟨h, _⟩ => h_o h), zero_mul, if_neg h_o]
    -- Step 2: substitute h_pdiv into the backward sum and collapse.
    show (∑ hi : Fin h, ∑ wi : Fin w,
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh'.val + hi.val
             let ww := kw'.val + wi.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
            * dy (finProdFinEquiv (finProdFinEquiv (o', hi), wi))) =
          ∑ idx_out : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
              v idx_in idx_out * dy idx_out
    simp_rw [h_pdiv]
    -- Now goal: ∑ hi wi, xpad * dy(flat(o', hi, wi)) = ∑ idx_out, (if o(idx_out) = o' then xpad' else 0) * dy(idx_out)
    -- Convert idx_out → ((o, hi), wi) → (o, hi, wi) via two Fintype.sum_equiv's.
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun idx_out : Fin (oc * h * w) =>
          (if (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1 = o' then
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh'.val +
               (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
             let ww := kw'.val + (finProdFinEquiv.symm idx_out).2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
           else 0) * dy idx_out)
        (fun pair : Fin (oc * h) × Fin w =>
          (if (finProdFinEquiv.symm pair.1).1 = o' then
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh'.val + (finProdFinEquiv.symm pair.1).2.val
             let ww := kw'.val + pair.2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
           else 0) * dy (finProdFinEquiv pair))
        (fun idx_out => by
          show _ * _ = _ * _
          rw [Equiv.apply_symm_apply])]
    rw [Fintype.sum_prod_type]
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun pair_h : Fin (oc * h) =>
          ∑ wi : Fin w,
            (if (finProdFinEquiv.symm pair_h).1 = o' then
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh'.val + (finProdFinEquiv.symm pair_h).2.val
               let ww := kw'.val + wi.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)
             else 0) * dy (finProdFinEquiv (pair_h, wi)))
        (fun ohi : Fin oc × Fin h =>
          ∑ wi : Fin w,
            (if ohi.1 = o' then
              (let pH := (kH - 1) / 2
               let pW := (kW - 1) / 2
               let hh := kh'.val + ohi.2.val
               let ww := kw'.val + wi.val
               if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                 x c' ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
               else 0)
             else 0) * dy (finProdFinEquiv (finProdFinEquiv ohi, wi)))
        (fun pair_h => by
          have h_inv : finProdFinEquiv (finProdFinEquiv.symm pair_h
              : Fin oc × Fin h) = pair_h :=
            Equiv.apply_symm_apply _ _
          simp_rw [h_inv])]
    rw [Fintype.sum_prod_type]
    -- Goal: ∑ hi, ∑ wi, xpad * dy(flat(o', hi, wi)) = ∑ o, ∑ hi, ∑ wi, (if o = o' then xpad else 0) * dy(...)
    -- Pull (if o = o' then ... else 0) and collapse o-sum via Finset.sum_eq_single.
    rw [Finset.sum_eq_single o'
          (fun o _ ho_ne =>
            Finset.sum_eq_zero (fun hi _ =>
              Finset.sum_eq_zero (fun wi _ => by
                rw [if_neg ho_ne, zero_mul])))
          (fun hni => absurd (Finset.mem_univ o') hni)]
    -- Now the outer (if o' = o' then ... else 0) collapses to the body.
    apply Finset.sum_congr rfl
    intro hi _
    apply Finset.sum_congr rfl
    intro wi _
    rw [if_pos rfl]

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
    as `(channel-reindex from b) + (W,x-only term constant in b)`. Apply
    `pdiv_add` + `pdiv_reindex` + `pdiv_const`, then collapse the
    Kronecker over the `(c, hi, wi)` decomposition of `Fin (oc*h*w)`.
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
    -- Step 1: pdiv decomposition. f decomposes as `b ↦ b(chan idx)` + `(W,x term)`.
    have h_pdiv : ∀ idx : Fin (oc * h * w),
        pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o idx =
        (if o = (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1
          then (1:ℝ) else 0) := by
      intro idx
      rw [show (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) =
            (fun b' k =>
              (fun y : Vec oc => fun k' : Fin (oc * h * w) =>
                y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) b' k +
              (fun (_ : Vec oc) (k' : Fin (oc * h * w)) =>
                ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                  W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                    (let pH := (kH - 1) / 2
                     let pW := (kW - 1) / 2
                     let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                     let ww := kw.val + (finProdFinEquiv.symm k').2.val
                     if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                       x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                     else 0)) b' k) from by
        funext b' k
        unfold Tensor3.flatten conv2d
        rfl]
      have h_reindex_diff : DifferentiableAt ℝ
          (fun y : Vec oc => fun k' : Fin (oc * h * w) =>
            y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) b :=
        (reindexCLM (fun k' : Fin (oc * h * w) =>
          (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)).differentiableAt
      have h_const_diff : DifferentiableAt ℝ
          (fun (_ : Vec oc) (k' : Fin (oc * h * w)) =>
            ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
              W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                (let pH := (kH - 1) / 2
                 let pW := (kW - 1) / 2
                 let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                 let ww := kw.val + (finProdFinEquiv.symm k').2.val
                 if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                   x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                 else 0)) b :=
        differentiableAt_const _
      rw [pdiv_add _ _ _ h_reindex_diff h_const_diff]
      rw [show (fun y : Vec oc => fun k' : Fin (oc * h * w) =>
                  y ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1)) =
            (fun y => fun k' => y ((fun k'' : Fin (oc * h * w) =>
                (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1) k')) from rfl]
      rw [pdiv_reindex (fun k'' : Fin (oc * h * w) =>
            (finProdFinEquiv.symm (finProdFinEquiv.symm k'').1).1)]
      rw [show pdiv (fun (_ : Vec oc) (k' : Fin (oc * h * w)) =>
                  ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
                    W ((finProdFinEquiv.symm (finProdFinEquiv.symm k').1).1) c kh kw *
                      (let pH := (kH - 1) / 2
                       let pW := (kW - 1) / 2
                       let hh := kh.val + (finProdFinEquiv.symm (finProdFinEquiv.symm k').1).2.val
                       let ww := kw.val + (finProdFinEquiv.symm k').2.val
                       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
                         x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
                       else 0))
                  b o idx = 0
          from pdiv_const _ _ _ _]
      ring
    -- Step 2: substitute and collapse the Kronecker via two stages of finProdFinEquiv.
    simp_rw [h_pdiv]
    -- Σ idx, [if o = chan(idx) then 1 else 0] * dy idx
    -- Convert idx ∈ Fin (oc*h*w) to ((c, hi), wi) via finProdFinEquiv.symm twice.
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun idx : Fin (oc * h * w) =>
          (if o = (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1 then (1:ℝ) else 0)
          * dy idx)
        (fun pair : Fin (oc * h) × Fin w =>
          (if o = (finProdFinEquiv.symm pair.1).1 then (1:ℝ) else 0)
          * dy (finProdFinEquiv pair))
        (fun idx => by
          show _ * _ = _ * _
          rw [Equiv.apply_symm_apply])]
    rw [Fintype.sum_prod_type]
    rw [Fintype.sum_equiv finProdFinEquiv.symm
        (fun pair_h : Fin (oc * h) =>
          ∑ wi : Fin w,
            (if o = (finProdFinEquiv.symm pair_h).1 then (1:ℝ) else 0)
            * dy (finProdFinEquiv (pair_h, wi)))
        (fun ch_pair : Fin oc × Fin h =>
          ∑ wi : Fin w,
            (if o = ch_pair.1 then (1:ℝ) else 0)
            * dy (finProdFinEquiv (finProdFinEquiv ch_pair, wi)))
        (fun pair_h => by
          have h_inv : finProdFinEquiv (finProdFinEquiv.symm pair_h
                : Fin oc × Fin h) = pair_h :=
            Equiv.apply_symm_apply _ _
          simp_rw [h_inv])]
    rw [Fintype.sum_prod_type]
    -- Goal is now ∑ c, ∑ hi, ∑ wi, (if o = c then 1 else 0) * dy (flat (c, hi, wi))
    have h_pull : ∀ c : Fin oc,
        (∑ hi : Fin h, ∑ wi : Fin w,
          (if o = c then (1:ℝ) else 0)
          * dy (finProdFinEquiv (finProdFinEquiv (c, hi), wi))) =
        (if o = c then (1:ℝ) else 0) *
          ∑ hi : Fin h, ∑ wi : Fin w,
            dy (finProdFinEquiv (finProdFinEquiv (c, hi), wi)) := by
      intro c
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro hi _
      rw [Finset.mul_sum]
    simp_rw [h_pull, ite_mul, one_mul, zero_mul]
    rw [Finset.sum_ite_eq Finset.univ o (fun c =>
        ∑ hi : Fin h, ∑ wi : Fin w,
          dy (finProdFinEquiv (finProdFinEquiv (c, hi), wi)))]
    simp

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

/-- **Smooth-point local-linearization for max-pool.** On a metric ball
    around `flatten x`, the flattened max-pool agrees with the reindex
    `y ↦ y ∘ σ` where σ routes each output position to its argmax's
    input position. Promoted via `EventuallyEq`. -/
theorem maxPool2_flat_hasFDerivAt {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (h_smooth : MaxPool2Smooth x)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    HasFDerivAt
      (fun v : Vec (c * (2 * h) * (2 * w)) =>
        Tensor3.flatten (maxPool2 (Tensor3.unflatten v)))
      (reindexCLM (maxPool2LocalReindex x))
      (Tensor3.flatten x) := by
  haveI : Nonempty (Fin c) := ⟨⟨0, hc⟩⟩
  haveI : Nonempty (Fin h) := ⟨⟨0, hh⟩⟩
  haveI : Nonempty (Fin w) := ⟨⟨0, hw⟩⟩
  -- Per-window gap function: positive everywhere under smoothness.
  let gap : Fin c × Fin h × Fin w × (Fin 2 × Fin 2) × (Fin 2 × Fin 2) → ℝ :=
    fun p => if p.2.2.2.1 = p.2.2.2.2 then 1
             else |x p.1 (winRowInv p.2.1 p.2.2.2.1.1) (winColInv p.2.2.1 p.2.2.2.1.2)
                  - x p.1 (winRowInv p.2.1 p.2.2.2.2.1) (winColInv p.2.2.1 p.2.2.2.2.2)|
  have hgap_pos : ∀ p, 0 < gap p := by
    intro ⟨co, ho, wo, ab, ab'⟩
    show 0 < (if ab = ab' then (1 : ℝ) else _)
    by_cases hab : ab = ab'
    · rw [if_pos hab]; norm_num
    · rw [if_neg hab]
      exact abs_pos.mpr (sub_ne_zero.mpr (h_smooth co ho wo ab ab' hab))
  -- Radius = (inf gap) / 4 — leaves slack 2r < r_raw for the diff argument.
  let univ_S : Finset (Fin c × Fin h × Fin w × (Fin 2 × Fin 2) × (Fin 2 × Fin 2)) :=
    Finset.univ
  set r_raw := univ_S.inf' Finset.univ_nonempty gap with hr_raw_def
  have hr_raw_pos : 0 < r_raw := by
    refine (Finset.lt_inf'_iff _).mpr ?_
    intro p _; exact hgap_pos p
  set r := r_raw / 4 with hr_def
  have hr_pos : 0 < r := by show 0 < r_raw / 4; linarith
  have hgap_le : ∀ co ho wo ab ab',
      gap (co, ho, wo, ab, ab') ≥ r_raw := fun co ho wo ab ab' =>
    Finset.inf'_le _ (Finset.mem_univ _)
  have h_local : Set.EqOn
      (fun v : Vec (c * (2 * h) * (2 * w)) =>
        Tensor3.flatten (maxPool2 (Tensor3.unflatten v)))
      (reindexCLM (maxPool2LocalReindex x) : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w))
      (Metric.ball (Tensor3.flatten x) r) := by
    intro y hy
    have hy_norm : ‖y - Tensor3.flatten x‖ < r := by
      rwa [Metric.mem_ball, dist_eq_norm] at hy
    have hy_coord : ∀ k, |y k - Tensor3.flatten x k| < r := by
      intro k
      have h1 : ‖(y - Tensor3.flatten x) k‖ ≤ ‖y - Tensor3.flatten x‖ :=
        norm_le_pi_norm (y - Tensor3.flatten x) k
      rw [Real.norm_eq_abs] at h1
      have : |y k - Tensor3.flatten x k| ≤ ‖y - Tensor3.flatten x‖ := by
        show |(y - Tensor3.flatten x) k| ≤ _
        exact h1
      linarith
    funext k_out
    set r1 := finProdFinEquiv.symm k_out with hr1
    set wo : Fin w := r1.2 with hwo
    set r2 := finProdFinEquiv.symm r1.1 with hr2
    set co : Fin c := r2.1 with hco
    set ho : Fin h := r2.2 with hho
    set ab := maxPool2Argmax x co ho wo with hab
    have h_max_y : ∀ a' b' : Fin 2,
        Tensor3.unflatten y co (winRowInv ho a') (winColInv wo b') ≤
        Tensor3.unflatten y co (winRowInv ho ab.1) (winColInv wo ab.2) := by
      intro a' b'
      by_cases h_eq : (a', b') = ab
      · have ha : a' = ab.1 := congrArg Prod.fst h_eq
        have hb : b' = ab.2 := congrArg Prod.snd h_eq
        rw [ha, hb]
      · have h_le_x : x co (winRowInv ho a') (winColInv wo b') ≤
                      x co (winRowInv ho ab.1) (winColInv wo ab.2) :=
          maxPool2Argmax_max x co ho wo (a', b')
        have h_ne_x : x co (winRowInv ho a') (winColInv wo b') ≠
                      x co (winRowInv ho ab.1) (winColInv wo ab.2) :=
          h_smooth co ho wo (a', b') ab h_eq
        have h_strict_x : x co (winRowInv ho a') (winColInv wo b') <
                          x co (winRowInv ho ab.1) (winColInv wo ab.2) :=
          lt_of_le_of_ne h_le_x h_ne_x
        have h_diff_x : x co (winRowInv ho ab.1) (winColInv wo ab.2) -
                        x co (winRowInv ho a') (winColInv wo b') ≥ r_raw := by
          have h_gap := hgap_le co ho wo (a', b') ab
          have h_gap_expanded : gap (co, ho, wo, (a', b'), ab) =
              |x co (winRowInv ho a') (winColInv wo b') -
               x co (winRowInv ho ab.1) (winColInv wo ab.2)| := by
            show (if (a', b') = ab then (1 : ℝ) else _) = _
            rw [if_neg h_eq]
          rw [h_gap_expanded] at h_gap
          rw [abs_sub_comm] at h_gap
          have h_pos : 0 ≤ x co (winRowInv ho ab.1) (winColInv wo ab.2) -
                       x co (winRowInv ho a') (winColInv wo b') := by linarith
          rwa [abs_of_nonneg h_pos] at h_gap
        set k_ab : Fin (c * (2 * h) * (2 * w)) :=
          finProdFinEquiv (finProdFinEquiv (co, winRowInv ho ab.1), winColInv wo ab.2)
        set k_ab' : Fin (c * (2 * h) * (2 * w)) :=
          finProdFinEquiv (finProdFinEquiv (co, winRowInv ho a'), winColInv wo b')
        have h_unflat_ab : Tensor3.unflatten y co (winRowInv ho ab.1) (winColInv wo ab.2)
                          = y k_ab := rfl
        have h_unflat_ab' : Tensor3.unflatten y co (winRowInv ho a') (winColInv wo b')
                           = y k_ab' := rfl
        have h_flat_x_ab : Tensor3.flatten x k_ab =
            x co (winRowInv ho ab.1) (winColInv wo ab.2) := by
          show x (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab).1).1
                  (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab).1).2
                  (finProdFinEquiv.symm k_ab).2 = _
          simp [k_ab, Equiv.symm_apply_apply]
        have h_flat_x_ab' : Tensor3.flatten x k_ab' =
            x co (winRowInv ho a') (winColInv wo b') := by
          show x (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab').1).1
                  (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab').1).2
                  (finProdFinEquiv.symm k_ab').2 = _
          simp [k_ab', Equiv.symm_apply_apply]
        rw [h_unflat_ab, h_unflat_ab']
        have hy_ab := hy_coord k_ab
        have hy_ab' := hy_coord k_ab'
        rw [h_flat_x_ab] at hy_ab
        rw [h_flat_x_ab'] at hy_ab'
        have h_lhs_ge : y k_ab - y k_ab' ≥
            (x co (winRowInv ho ab.1) (winColInv wo ab.2) -
             x co (winRowInv ho a') (winColInv wo b')) - 2 * r := by
          have h1 := abs_sub_lt_iff.mp hy_ab
          have h2 := abs_sub_lt_iff.mp hy_ab'
          linarith
        have h_2r_lt : 2 * r < r_raw := by show 2 * (r_raw / 4) < r_raw; linarith
        linarith
    show maxPool2 (Tensor3.unflatten y) co ho wo = y (maxPool2LocalReindex x k_out)
    rw [maxPool2_eq_at_max (Tensor3.unflatten y) co ho wo ab.1 ab.2 h_max_y]
    show Tensor3.unflatten y co (winRowInv ho ab.1) (winColInv wo ab.2) =
         y (maxPool2LocalReindex x k_out)
    rfl
  exact (reindexCLM (maxPool2LocalReindex x)).hasFDerivAt.congr_of_eventuallyEq
    (h_local.eventuallyEq_of_mem (Metric.ball_mem_nhds _ hr_pos))

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
  rw [Finset.sum_eq_single ci
      (fun co _ hne_co => by
        rw [Finset.sum_eq_zero]
        intro ho _
        rw [Finset.sum_eq_zero]
        intro wo _
        rw [if_neg (fun ⟨h1, _, _, _⟩ => hne_co h1)]
        ring)
      (fun h => absurd (Finset.mem_univ ci) h)]
  rw [Finset.sum_eq_single (winRow hi_in)
      (fun ho _ hne_ho => by
        rw [Finset.sum_eq_zero]
        intro wo _
        rw [if_neg (fun ⟨_, h2, _, _⟩ => hne_ho h2)]
        ring)
      (fun h => absurd (Finset.mem_univ _) h)]
  rw [Finset.sum_eq_single (winCol wi_in)
      (fun wo _ hne_wo => by
        rw [if_neg (fun ⟨_, _, h3, _⟩ => hne_wo h3)]
        ring)
      (fun h => absurd (Finset.mem_univ _) h)]
  by_cases h_arg : MaxPool2IsArgmax x ci hi_in wi_in
  · rw [if_pos ⟨rfl, rfl, rfl, h_arg⟩, if_pos h_arg]
    ring
  · rw [if_neg (fun ⟨_, _, _, h⟩ => h_arg h), if_neg h_arg]
    ring

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
- `conv2d_has_vjp3` — Phase 4: input-path VJP, proved from foundation
  rules using the per-coord pdiv chain plus a custom `pdiv_pi_pad_eval`
  helper for the dependent-`if hpad : pad then v(σ hpad) else 0`
  pattern. Backward function is `conv2d_input_grad_formula` (sum over
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
  rw [globalAvgPoolFlat_as_sum]
  -- summands differentiable
  have h_summand_diff : ∀ p ∈ (Finset.univ : Finset (Fin h × Fin w)),
      DifferentiableAt ℝ
        (fun (u : Vec (c * h * w)) (k : Fin c) =>
          (1 / (h * w : ℝ)) *
          u (finProdFinEquiv (finProdFinEquiv (k, p.1), p.2))) v := by
    intro p _
    have h_const : DifferentiableAt ℝ
        (fun (_ : Vec (c * h * w)) (_ : Fin c) => (1 / (h * w : ℝ))) v :=
      differentiableAt_const _
    have h_reindex : DifferentiableAt ℝ
        (fun (u : Vec (c * h * w)) (k : Fin c) =>
          u (finProdFinEquiv (finProdFinEquiv (k, p.1), p.2))) v :=
      (reindexCLM (fun k : Fin c =>
        finProdFinEquiv (finProdFinEquiv (k, p.1), p.2))).differentiableAt
    exact h_const.mul h_reindex
  rw [pdiv_finset_sum _ _ _ h_summand_diff]
  -- each summand pdiv
  have hterm : ∀ p : Fin h × Fin w,
      pdiv (fun (u : Vec (c * h * w)) (k : Fin c) =>
              (1 / (h * w : ℝ)) *
              u (finProdFinEquiv (finProdFinEquiv (k, p.1), p.2))) v idx ci =
      (1 / (h * w : ℝ)) *
        (if idx = finProdFinEquiv (finProdFinEquiv (ci, p.1), p.2) then 1 else 0) := by
    intro p
    have h_prod :
        (fun (u : Vec (c * h * w)) (k : Fin c) =>
          (1 / (h * w : ℝ)) *
          u (finProdFinEquiv (finProdFinEquiv (k, p.1), p.2))) =
        (fun u k =>
          (fun (_ : Vec (c * h * w)) (_ : Fin c) => (1 / (h * w : ℝ))) u k *
          (fun (u' : Vec (c * h * w)) (k' : Fin c) =>
            u' (finProdFinEquiv (finProdFinEquiv (k', p.1), p.2))) u k) := rfl
    have h_const_diff : DifferentiableAt ℝ
        (fun (_ : Vec (c * h * w)) (_ : Fin c) => (1 / (h * w : ℝ))) v :=
      differentiableAt_const _
    have h_reindex_diff : DifferentiableAt ℝ
        (fun (u' : Vec (c * h * w)) (k' : Fin c) =>
          u' (finProdFinEquiv (finProdFinEquiv (k', p.1), p.2))) v :=
      (reindexCLM (fun k' : Fin c =>
        finProdFinEquiv (finProdFinEquiv (k', p.1), p.2))).differentiableAt
    rw [h_prod, pdiv_mul _ _ _ h_const_diff h_reindex_diff]
    rw [pdiv_const, pdiv_reindex (fun k' : Fin c =>
          finProdFinEquiv (finProdFinEquiv (k', p.1), p.2))]
    ring
  simp_rw [hterm]
  rw [← Finset.mul_sum]
  -- ∑ p, (if idx = enc(ci,p.1,p.2) then 1 else 0) = if channel idx = ci then 1 else 0
  by_cases hch : flatChannel c h w idx = ci
  · rw [if_pos hch]
    -- the unique p matching is the spatial coords of idx
    set p0 : Fin h × Fin w :=
      ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2,
       (finProdFinEquiv.symm idx).2) with hp0
    rw [Finset.sum_eq_single p0]
    · rw [if_pos]
      · ring
      · -- idx = enc(ci, p0.1, p0.2)
        rw [hp0]
        show idx = finProdFinEquiv
          (finProdFinEquiv (ci, (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2),
            (finProdFinEquiv.symm idx).2)
        rw [← hch]
        show idx = finProdFinEquiv
          (finProdFinEquiv ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1,
            (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2),
            (finProdFinEquiv.symm idx).2)
        rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
    · intro p _ hne
      rw [if_neg]
      intro heq
      apply hne
      -- idx = enc(ci,p.1,p.2) and idx = enc(ci,p0.1,p0.2) ⟹ p = p0
      have hidx0 : idx = finProdFinEquiv
          (finProdFinEquiv (ci, p0.1), p0.2) := by
        rw [hp0]
        show idx = finProdFinEquiv
          (finProdFinEquiv (ci, (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2),
            (finProdFinEquiv.symm idx).2)
        rw [← hch]
        show idx = finProdFinEquiv
          (finProdFinEquiv ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1,
            (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2),
            (finProdFinEquiv.symm idx).2)
        rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
      rw [hidx0] at heq
      have h1 := finProdFinEquiv.injective heq
      have h2 : finProdFinEquiv (ci, p.1) = finProdFinEquiv (ci, p0.1) := (Prod.mk.inj h1).1.symm
      have hwe : p.2 = p0.2 := (Prod.mk.inj h1).2.symm
      have hhe : p.1 = p0.1 := (Prod.mk.inj (finProdFinEquiv.injective h2)).2
      exact Prod.ext hhe hwe
    · intro hp; exact absurd (Finset.mem_univ _) hp
  · rw [if_neg hch, zero_div]
    rw [Finset.sum_eq_zero, mul_zero]
    intro p _
    rw [if_neg]
    intro heq
    apply hch
    -- idx = enc(ci,p.1,p.2) ⟹ channel idx = ci
    unfold flatChannel
    rw [heq, Equiv.symm_apply_apply, Equiv.symm_apply_apply]

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
    · rw [if_pos rfl]; ring
    · intro b _ hne
      rw [if_neg (fun heq => hne heq.symm)]; ring
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
    |max a b - max c d| ≤ e := by
  rw [abs_le] at h1 h2 ⊢
  refine ⟨?_, ?_⟩
  · have h : max c d - e ≤ max a b := by
      rcases le_total c d with hcd | hcd
      · rw [max_eq_right hcd]
        exact le_trans (by linarith [h2.1]) (le_max_right a b)
      · rw [max_eq_left hcd]
        exact le_trans (by linarith [h1.1]) (le_max_left a b)
    linarith
  · have h : max a b - e ≤ max c d := by
      rcases le_total a b with hab | hab
      · rw [max_eq_right hab]
        exact le_trans (by linarith [h2.2]) (le_max_right c d)
      · rw [max_eq_left hab]
        exact le_trans (by linarith [h1.2]) (le_max_left c d)
    linarith

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
theorem abs_max_le {a b A : ℝ} (ha : |a| ≤ A) (hb : |b| ≤ A) : |max a b| ≤ A := by
  rcases le_total a b with h | h
  · rwa [max_eq_right h]
  · rwa [max_eq_left h]

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
  set F :=
    ((bnForward (c * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (c * h * w) ∘ bnForward (c * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) with hF
  have hF_diff : DifferentiableAt ℝ F v :=
    resblock_body_differentiableAt W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hres_diff : DifferentiableAt ℝ (residual F) v := by
    show DifferentiableAt ℝ (biPath F (fun x => x)) v
    exact DifferentiableAt.add hF_diff differentiable_id.differentiableAt
  have h_smooth_res : ∀ k, residual F v k ≠ 0 := h_smooth_out
  exact (relu_differentiableAt_of_smooth (c * h * w) _ h_smooth_res).comp v hres_diff

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
  set proj := (bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp) with hproj
  set F :=
    ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) with hF
  have hproj_diff : DifferentiableAt ℝ proj v :=
    (convBn_differentiable Wp bp εp γp βp hεp) v
  have hF_diff : DifferentiableAt ℝ F v :=
    resblock_body_differentiableAt W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hres_diff : DifferentiableAt ℝ (residualProj proj F) v := by
    show DifferentiableAt ℝ (biPath proj F) v
    exact DifferentiableAt.add hproj_diff hF_diff
  have h_smooth_res : ∀ k, residualProj proj F v k ≠ 0 := h_smooth_out
  exact (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth_res).comp v hres_diff

-- convBnRelu diffAt
theorem convBnRelu_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, bnForward (oc * h * w) ε γ β (flatConv W b v) k ≠ 0) :
    DifferentiableAt ℝ (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b) v := by
  have hinner : DifferentiableAt ℝ (bnForward (oc * h * w) ε γ β ∘ flatConv W b) v :=
    ((bnForward_differentiable (oc * h * w) ε γ β hε).comp (flatConv_differentiable W b)) v
  exact (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v hinner

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
