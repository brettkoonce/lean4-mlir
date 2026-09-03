import LeanMlir.Proofs.Float.ViTFloatBridge
import LeanMlir.Proofs.Codegen.AdjointChainBridge

/-! # The GELU saturation constant, in the two shapes its consumers want

The analysis lives in `Architectures/GeluSaturation.lean` (`geluScalarDeriv_abs_le`,
`geluScalar_lipschitz`): the tanh-form GELU is globally `3/2`-Lipschitz, because past the
small-`|x|` region `gelu′`'s `sech²` factor decays like `e^{−2√(2/π)|x|}` and beats the cubic
polynomial growth. This file packages it for the two tiers that use it:

* `lipOnWindow_gelu` — the adjoint-chain gain instance. In fact the gain is *global*, so the
  window argument is ignored; that is the point of the saturation bound.
* `floatClose_gelu_sat` — the flat-modulus `FloatClose`, `egelu + 3/2·e`.

⚠ **`floatClose_gelu` now states the `min` of its polynomial modulus and this one**
(`ViTFloatBridge.lean`), so the composable instance every net folds already carries the
saturation constant and `floatClose_gelu_sat` is the pinned flat-only form, kept because it
is what the adjoint-chain write-up quotes. Before 2026-09-03 the polynomial was the only
modulus in the fold — ~400 at ConvNeXt's operating magnitudes (A ≈ 20) against a true
constant of ≈ 1.13 — and that alone put a whole-net ConvNeXt-T budget past what `norm_num`
will evaluate (`planning/float_budget_numbers.md` §3.3.0).
-/

namespace Proofs

/-- **GELU has windowed gain `3/2`** — the adjoint-chain instance (in fact the
    gain is global: no window needed). -/
theorem lipOnWindow_gelu {n : Nat} (A : ℝ) :
    LipOnWindow A (3 / 2) (gelu n) := by
  intro u v e he _hu _hv hd i
  calc |gelu n u i - gelu n v i|
      = |geluScalar (u i) - geluScalar (v i)| := rfl
    _ ≤ 3 / 2 * |u i - v i| := geluScalar_lipschitz _ _
    _ ≤ 3 / 2 * e := by
        have := hd i
        linarith

/-- **`floatClose_gelu` with the flat modulus alone**: same rounding budget `egelu`, and the
    input-shift modulus `3/2·e` — the saturation constant without the `min` against the
    magnitude polynomial that `floatClose_gelu` now carries. At ConvNeXt's operating
    magnitudes (A ≈ 20) the polynomial branch is ~250× worse, so this IS the composable
    instance's modulus there; at `A ≲ 1/2` the polynomial branch is the tighter one, which
    is why the shipped leaf takes the `min` rather than replacing one with the other. -/
theorem floatClose_gelu_sat {n : Nat} (fgelu : ℝ → ℝ) {egelu A : ℝ}
    (hegelu : 0 ≤ egelu) (hA : 0 ≤ A)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu) :
    FloatClose A (A + egelu) (gelu n) (fun v i => fgelu (v i))
      (fun e => egelu + 3 / 2 * e) := by
  refine ⟨(floatClose_gelu fgelu hegelu hA hg).1, fun vt va e _hva _hvt hd i => ?_⟩
  calc |fgelu (vt i) - gelu n va i|
      = |fgelu (vt i) - geluScalar (va i)| := rfl
    _ ≤ |fgelu (vt i) - geluScalar (vt i)|
        + |geluScalar (vt i) - geluScalar (va i)| :=
        abs_sub_le (fgelu (vt i)) (geluScalar (vt i)) (geluScalar (va i))
    _ ≤ egelu + 3 / 2 * e := by
        refine add_le_add (hg _) ((geluScalar_lipschitz _ _).trans ?_)
        have := hd i
        linarith

end Proofs
