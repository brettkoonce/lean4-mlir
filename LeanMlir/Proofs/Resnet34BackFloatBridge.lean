import LeanMlir.Proofs.CnnBackFloatBridge

/-! # ℝ→Float32 bridge for the ResNet-34 BACKWARD: the residual basic-block input-VJP

A3 (planning/a3_backward_deepnet_assembly.md): r34, "the first Imagenette backward." The r34
identity basic block forward is `rblkPC = relu ∘ residual F` with `F = bn₂∘conv₂ ∘ relu∘bn₁∘conv₁`
(per-channel BN, non-strided 3×3 convs) — `relu(F(x) + x)`. Its input-gradient VJP at a smooth
point is the key new structural element the doc flags for r34: the **residual-skip backward**.

The KEY observation (planning doc §2, "Residual fan-in (backward)"): the skip routes the cotangent
to BOTH branches and ADDS, so the block backward is itself a *forward* composition
`residual bF ∘ reluMaskBack` — and `residual` is exactly the `Proofs.residual` combinator the
forward uses, whose float bridge `FloatBridges.residual` is already proven. So the residual-skip
backward needs **no new combinator**: it reuses `FloatBridges.residual` (the rounded skip-add is the
backward's too), with `bF` the reverse of `F` assembled from the per-op backward bridges
(`convFlatBack` / `reluMaskBack` / the supplied BN-backs). The dominant r34 block (13 of the 16 are
identity blocks).

`bF = convFlatBack W₁ ∘ bnB₁ ∘ reluMaskBack ∘ convFlatBack W₂ ∘ bnB₂` (the reverse of `F`); the two
per-channel BN-backs `bnB₁`/`bnB₂` are supplied as `FloatBridges` facts (discharge with
`floatBridges_bnPerChannelBack`), exactly as `cifarBn_grad_floatBridges` and the forward
`cifarBn_floatBridges` supply their BNs.

Remaining for the r34 whole net: the **down**-block (needs a strided-conv backward + the two-branch
`residualProj` backward) and the stem (strided conv) + GAP backward — a forward/backward-shared gap
(the forward r34 whole-net float bridge is itself only block-level so far).
-/

namespace Proofs

/-- The r34 identity basic-block input-gradient VJP at a smooth point — the **reverse of `rblkPC`**.
    `relu(F(x)+x)` backward = the ReLU mask, then the residual split (cotangent to both the body and
    the skip, added): `residual bF ∘ reluMaskBack`, with `bF` the reverse of `F = bn₂∘conv₂ ∘
    relu∘bn₁∘conv₁`. The ReLU kinks read the fixed sign masks `m_out`/`m_mid`; the BN-backs `bnB₁`/
    `bnB₂` are the per-channel BatchNorm backwards. -/
noncomputable def r34IdBlockBack {c h w : Nat}
    (W₁ W₂ : Kernel4 c c 3 3)
    (bnB1 bnB2 : Vec (c * h * w) → Vec (c * h * w))
    (m_out m_mid : Fin (c * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid] :
    Vec (c * h * w) → Vec (c * h * w) :=
  Proofs.residual
      (convFlatBack (h := h) (w := w) W₁ ∘ bnB1 ∘ reluMaskBack m_mid
        ∘ convFlatBack (h := h) (w := w) W₂ ∘ bnB2)
    ∘ reluMaskBack m_out

/-- **The r34 identity-block input-gradient VJP float-bridges.** Assembled in one `.comp` chain:
    the inner body backward `bF` (a `convFlatBack`/`reluMaskBack`/BN-back chain) is wrapped by
    `FloatBridges.residual` (the residual-skip backward — same combinator as the forward, the
    rounded skip-add), then composed after the outer ReLU mask. The two BN-backs are supplied as
    `FloatBridges` facts (discharge with `floatBridges_bnPerChannelBack`). The backward peer of the
    r34 forward identity block; closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem floatBridges_r34IdBlockBack {c h w : Nat} (M : FloatModel)
    (W₁ W₂ : Kernel4 c c 3 3)
    (bnB1 bnB2 : Vec (c * h * w) → Vec (c * h * w))
    (m_out m_mid : Fin (c * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid]
    {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < c * h * w)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w') (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w')
    (hbnB1 : FloatBridges bnB1) (hbnB2 : FloatBridges bnB2) :
    FloatBridges (r34IdBlockBack W₁ W₂ bnB1 bnB2 m_out m_mid) := by
  unfold r34IdBlockBack
  exact (floatBridges_reluMaskBack m_out).comp
    (FloatBridges.residual M
      ((((hbnB2.comp (floatBridges_convBack M W₂ hw' hn hW₂)).comp
          (floatBridges_reluMaskBack m_mid)).comp hbnB1).comp
          (floatBridges_convBack M W₁ hw' hn hW₁)))

end Proofs
