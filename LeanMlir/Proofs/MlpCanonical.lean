import LeanMlir.Proofs.SgdDescentMlp
import LeanMlir.Proofs.LinBackFloatBridge
import LeanMlir.Proofs.MlpFaithfulPoC

/-! # The CANONICAL MNIST MLP — 784→512→512→10 (ReLU, biased)

`mlpVerified` (`LeanMlir/VerifiedNets.lean`, Chapter 2) is the repo's canonical MNIST
reference architecture: `[.dense 784 512, .relu, .dense 512 512, .relu, .dense 512 10]`.
Every runnable MNIST MLP path uses it (verified/e4m3/pgd/spectral/smooth trainers, the
committed `verified_mlir/mlp_train_step.mlir` render, the baselines, `margin_probe.py`).

This file makes the canonical claim a CHECKABLE LEAN SURFACE: the generic MLP proof
chain (whole-net VJP, float-gradient closeness, float-SGD descent, the input-VJP
FloatBridges, the emitted-train-step tie) instantiated at the literal canonical dims.
Each declaration below IS the corresponding generic theorem at `(784, 512, 512, 10)` —
`#check` shows the specialized statement; the 3-axiom audit covers them all. The
spec-level partner is `SpecVJP.lean`'s `mlpVerified_denote_eq` / `mlpVerified_has_vjp*`
(stated over `mlpVerified.layers` itself; that file lives outside the Mathlib-only seam).

The OTHER MNIST proof population — the trained-weight certificate instances
(`LipschitzCert{Instance,Scorecard*,Float}`, `TrainedMlpWitness`, `TrainedLinearDescent`)
— deliberately lives on a REDUCED model (4×4-pooled 49-dim inputs, width-8 hidden,
/128–/256 rational weights): every margin, Schatten/Gram sum, and LDLᵀ SOS witness is
exact rational arithmetic checked in-kernel, which is infeasible today at 512-wide
fan-ins (and the unconstrained canonical net's spectral-product cert is MEASURED
vacuous — L ≈ 39 ⇒ 0% certified — which is why randomized smoothing, which DOES run on
the canonical net, exists). Those files carry a reduced-model banner pointing here. -/

namespace Proofs
namespace MlpCanonical

-- These canonical-dims instantiations are Props stated as `def`s on purpose: their
-- types are the generic theorems' statements specialized at (784, 512, 512, 10),
-- inferred rather than restated (`theorem` would force spelling each one out).
set_option linter.defProp false

/-- Canonical whole-net pointwise VJP: `mlp_has_vjp_at` at (784, 512, 512, 10) —
    the honest conditional witness (both hidden layers off-kink at `x`). -/
noncomputable def has_vjp_at :=
  mlp_has_vjp_at (d₀ := 784) (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical backward-correctness: the canonical witness's backward IS the
    Jacobian-transpose contraction. -/
noncomputable def has_vjp_correct :=
  mlp_has_vjp_correct (d₀ := 784) (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical output-layer float-SGD descent (`mlp_output_float_sgd_descends`
    at the canonical dims): one binary32-model SGD step on W₂ decreases the
    real CE loss, margins carried. -/
noncomputable def output_float_sgd_descends :=
  mlp_output_float_sgd_descends (d₀ := 784) (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical hidden-layer float-SGD descent. -/
noncomputable def hidden_float_sgd_descends :=
  mlp_hidden_float_sgd_descends (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical input-layer float-SGD descent — the whole canonical MLP is
    float-fused descent, layer by layer. -/
noncomputable def input_float_sgd_descends :=
  mlp_input_float_sgd_descends (d₀ := 784) (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical W₁ float-gradient closeness. -/
noncomputable def w1_grad_close :=
  mlp_w1_grad_close (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical W₀ float-gradient closeness. -/
noncomputable def w0_grad_close :=
  mlp_w0_grad_close (d₀ := 784) (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical whole-MLP backward float bridge (`mlpInputGrad_floatBridges` at
    the canonical dims): the float-evaluated input-VJP chain is FloatBridges. -/
noncomputable def inputGrad_floatBridges :=
  mlpInputGrad_floatBridges (d₀ := 784) (d₁ := 512) (d₂ := 512) (d₃ := 10)

/-- Canonical emitted-train-step tie (`MlpPoC.mlp_train_step_tied_certified` at
    the canonical dims): every SGD op of the emitted graph denotes the certified
    loss-descent step of the REAL canonical forward. -/
noncomputable def train_step_tied_certified :=
  MlpPoC.mlp_train_step_tied_certified (d₀ := 784) (d₁ := 512) (d₂ := 512) (d₃ := 10)

end MlpCanonical
end Proofs
