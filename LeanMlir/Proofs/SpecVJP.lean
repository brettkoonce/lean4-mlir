import LeanMlir.VerifiedNets
import LeanMlir.Proofs.MLP

/-! # Spec → math (the verification tie), Rung 1: the linear classifier

The shape `#guard` in `MainResnet34Verified` only checks the *parameter interface*
(typechecking). This file is the first rung of connecting a readable `VerifiedNetSpec`
to the actual **math** — the proven VJP — on the simplest net, the Chapter-2 linear
classifier (`dense 784→10`).

The pattern (extends to MLP → conv nets, each rigid/per-net):
  1. `denote` maps the spec's layers to the Mathlib math function the proofs are about;
  2. a `rfl` lemma ties the spec's denotation to that named function (`mnistLinear`);
  3. the whole-model VJP theorem is stated about *the spec's denotation* and discharged
     by the audited op-level VJP (`dense_has_vjp`).

If the spec's `layers` drifts from `[.dense 784 10]`, step 2/3 stop reducing and the
proofs fail to typecheck — so the readable architecture is provably the verified one,
at the math level, not just the shape level.
-/

open Proofs

/- `linearVerified` (the single dense 784→10 spec) is imported from `LeanMlir.VerifiedNets`
   — the *same* object `MainMnistLinearVerified` trains, so the VJP below is about the
   trainer's exact spec, not a copy. The shape tie (`toSpecs == …`) lives there too. -/

/-- Math denotation of the linear spec. The Chapter-2 model is a single dense layer, so
    `[.dense 784 10]` denotes to the Mathlib `dense W b`. Any other layer list is not the
    linear model (`0`), which makes the tie below drift-sensitive. -/
noncomputable def denoteLinear (layers : List VLayer) (W : Mat 784 10) (b : Vec 10) :
    Vec 784 → Vec 10 :=
  match layers with
  | [.dense 784 10] => dense W b
  | _               => fun _ => 0

/-- **Spec ≡ the proven model.** `linearVerified`'s denotation is exactly `mnistLinear`
    (the function the Chapter-2 VJP capstone is about) — by `rfl`, so it's checked by the
    kernel and breaks if `linearVerified.layers` changes. -/
theorem linearVerified_denote_eq (W : Mat 784 10) (b : Vec 10) :
    denoteLinear linearVerified.layers W b = mnistLinear W b := rfl

/-- **The spec carries the math.** The linear spec's denotation has the proven VJP —
    discharged by the audited `dense_has_vjp`. This is the whole-model verification
    stated about the *readable layer list*, not a hand-written function. -/
noncomputable def linearVerified_has_vjp (W : Mat 784 10) (b : Vec 10) :
    HasVJP (denoteLinear linearVerified.layers W b) :=
  dense_has_vjp W b

/-- …and its correctness headline carries over verbatim (the backward is the
    `pdiv`-contracted Jacobian of the spec's denotation). -/
theorem linearVerified_has_vjp_correct (W : Mat 784 10) (b : Vec 10)
    (x : Vec 784) (dy : Vec 10) (i : Fin 784) :
    (linearVerified_has_vjp W b).backward x dy i
      = ∑ j : Fin 10, pdiv (denoteLinear linearVerified.layers W b) x i j * dy j :=
  (linearVerified_has_vjp W b).correct x dy i

/-! ## Rung 2: the MLP — the first genuine `vjp_comp` fold

The linear model was the degenerate case (one layer, no fold). The MLP's denotation is a
*chain* — `dense ∘ relu ∘ dense ∘ relu ∘ dense` (`mlpForward`) — and its VJP is built by
folding `vjp_comp_at` down that chain (`mlp_has_vjp_at`). So this is where the spec→math
tie first exercises the chain rule, not just a single op. -/

/-- Math denotation of the MLP spec: the 5-layer list denotes to `mlpForward`. -/
noncomputable def denoteMLP (layers : List VLayer)
    (W₀ : Mat 784 512) (b₀ : Vec 512) (W₁ : Mat 512 512) (b₁ : Vec 512)
    (W₂ : Mat 512 10) (b₂ : Vec 10) : Vec 784 → Vec 10 :=
  match layers with
  | [.dense 784 512, .relu, .dense 512 512, .relu, .dense 512 10] =>
      mlpForward W₀ b₀ W₁ b₁ W₂ b₂
  | _ => fun _ => 0

/-- **Spec ≡ the proven model.** `mlpVerified`'s denotation is exactly `mlpForward`
    (`dense ∘ relu ∘ dense ∘ relu ∘ dense`) — by `rfl`, drift-sensitive. -/
theorem mlpVerified_denote_eq (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) :
    denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂ = mlpForward W₀ b₀ W₁ b₁ W₂ b₂ := rfl

/-- **The spec carries the math (canonical witness).** The MLP spec's denotation has a
    VJP — the global `pdiv`-derived witness (`mlp_has_vjp`; relu uses the framework
    subgradient convention at the kinks, per `Proofs/README.md`). -/
noncomputable def mlpVerified_has_vjp (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) :
    HasVJP (denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂) :=
  mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂

/-- **The spec carries the math (the real fold).** At a smooth input — the two ReLU
    pre-activations avoid zero — the MLP spec's denotation has a VJP built by *folding*
    `vjp_comp_at` through `dense → relu → dense → relu → dense` (no `rfl` escape at the
    kinks). This is the chain rule applied to the spec, the step linear couldn't show. -/
noncomputable def mlpVerified_has_vjp_at (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) (x : Vec 784)
    (h0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h1 : ∀ k, dense W₁ b₁ (relu 512 (dense W₀ b₀ x)) k ≠ 0) :
    HasVJPAt (denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂) x :=
  mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1

/-- …correctness headline for the canonical witness carries over to the spec. -/
theorem mlpVerified_has_vjp_correct (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10)
    (x : Vec 784) (dy : Vec 10) (i : Fin 784) :
    (mlpVerified_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).backward x dy i
      = ∑ j : Fin 10, pdiv (denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  (mlpVerified_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).correct x dy i
