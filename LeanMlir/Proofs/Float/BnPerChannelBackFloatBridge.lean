import LeanMlir.Proofs.Codegen.BnBackComposeBridge
import LeanMlir.Proofs.Float.BnPerChannelFloatBridge

/-! # Per-channel BatchNorm BACKWARD float-bridge (the cifarBn/r34/LN backward discharge)

`floatClose_bnBack` (`BnBackComposeBridge`) bridges **one** flat BatchNorm backward
(`bn_grad_input`, scalar `γ`, single normalization group). The deep nets carry per-channel BN over
the Tensor3 activation layout; its certified backward is `bnPerChannelTensor3_grad_input`
(`PerChannelBN.lean`) — proven faithful (`bnPerChannelTensor3_grad_input_correct`) to the
block-diagonal Jacobian.

That backward is, *as a function of the cotangent*, the **same conjugation** the forward uses
(`floatBridges_bnPerChannelTensor3`): relabel to Mat-split (`reassocFwd = gather E`), run the
per-channel block-diagonal `bn_grad_input` (`bnPerChannel_grad_input = perRowIdxFlat …`
definitionally), relabel back (`reassocBack = gather E.symm`). So the float bridge is the same
two-rung lift with `floatClose_bnBack` in the per-row slot:

* `floatBridges_bnPerChannelFlatBack` — `FloatClose.perRowIdx` of the per-channel
  `floatClose_bnBack` (uniform `G`/`S`/`Xh`/`es`/`exh` across channels ⇒ a single budget);
* `floatBridges_bnPerChannelBack` — conjugate by the two `floatBridges_gather` layout permutations.

This discharges the abstract `FloatBridges bnB…` hypotheses of `cifarBn_grad_floatBridges`, exactly
as `floatBridges_bnPerChannelTensor3` discharges the forward `cifarBn_floatBridges`'s BN hyps. The
shared per-channel BN/LN backward op r34/convnext/vit also fold.
-/

namespace Proofs

/-- **Per-channel BN backward (Mat-split layout) float-bridges.** The block-diagonal lift of
    `floatClose_bnBack`: each channel `c` runs its own three-term `bn_grad_input` over its `m`-wide
    spatial slab (saved input `Mat.unflatten X c`, supplied float stats `fs c`/`fxh c`), all sharing
    a uniform budget (bounds `G`/`S`/`Xh`, moduli `es`/`exh`). Because
    `bnPerChannel_grad_input = perRowIdxFlat oc m (per-channel bn_grad_input)` definitionally, this
    is `FloatClose.perRowIdx` of the per-channel `floatClose_bnBack`. -/
theorem floatBridges_bnPerChannelFlatBack {oc m : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (X : Vec (oc * m)) (fs : Fin oc → ℝ) (fxh : Fin oc → Vec m)
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hm : 0 < m)
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c - bnIstd m (Mat.unflatten X c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd m (Mat.unflatten X c) ε| ≤ S)
    (hxh : ∀ c i, |bnXhat m ε (Mat.unflatten X c) i| ≤ Xh)
    (hfxh : ∀ c i, |fxh c i - bnXhat m ε (Mat.unflatten X c) i| ≤ exh) :
    FloatBridges (fun dy => bnPerChannel_grad_input oc m ε γ X dy) := by
  have hbn : (fun dy => bnPerChannel_grad_input oc m ε γ X dy)
      = perRowIdxFlat oc m
          (fun c => fun dyc => bn_grad_input m ε (γ c) (Mat.unflatten X c) dyc) := by
    funext dy idx; rfl
  rw [hbn]
  intro A hA
  have hg := fun c => floatClose_bnBack M (Mat.unflatten X c) (fxh c) (fs c) hm
    (hγ c) (hs c) (hSabs c) (hxh c) (hfxh c) (A := A)
  have hpr := FloatClose.perRowIdx (d := m) oc hg
  exact ⟨_, _, _, hpr.cod_nonneg hA (Nat.mul_pos hoc hm), hpr⟩

/-- **Per-channel BN backward (network Tensor3 layout) float-bridges.** Conjugate the Mat-split
    `floatBridges_bnPerChannelFlatBack` by the layout permutations `reassocFwd = gather E` and
    `reassocBack = gather E.symm` (each `floatBridges_gather`, modulus `id`) via `FloatBridges.comp`
    — exactly the forward `floatBridges_bnPerChannelTensor3` shape with the backward in the middle.
    Bridges the certified `bnPerChannelTensor3_grad_input` (the BatchNorm backward the CIFAR-BN /
    ResNet-34 gradients contain). Discharges `cifarBn_grad_floatBridges`'s `FloatBridges bnB…`. -/
theorem floatBridges_bnPerChannelBack {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (x : Vec (oc * h * w)) (fs : Fin oc → ℝ) (fxh : Fin oc → Vec (h * w))
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hhw : 0 < h * w)
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c - bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ S)
    (hxh : ∀ c i, |bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ Xh)
    (hfxh : ∀ c i, |fxh c i - bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ exh) :
    FloatBridges (fun dy => bnPerChannelTensor3_grad_input oc h w ε γ x dy) := by
  have hflat := floatBridges_bnPerChannelFlatBack M γ (reassocFwd oc h w x) fs fxh hoc hhw
    hγ hs hSabs hxh hfxh
  have heq : (fun dy => bnPerChannelTensor3_grad_input oc h w ε γ x dy)
      = gather (reassocEquiv oc h w).symm
          ∘ (fun dy' => bnPerChannel_grad_input oc (h * w) ε γ (reassocFwd oc h w x) dy')
          ∘ gather (reassocEquiv oc h w) := by
    funext dy; rfl
  rw [heq]
  exact ((floatBridges_gather (reassocEquiv oc h w)).comp hflat).comp
    (floatBridges_gather (reassocEquiv oc h w).symm)

end Proofs
