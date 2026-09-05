import LeanMlir.Proofs.Architectures.EfficientNetChainClose
import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie
import LeanMlir.Proofs.Float.EfficientNetWholeBackFloatBridge

/-! # ⭐⭐ `efficientnetInputGradB` IS the certified whole-net EfficientNet-B0 gradient

The fourth whole-net backward tie, after ResNet-34 (`Resnet34BackCertifiedTie.lean`),
MobileNetV2 (`MobileNetV2WholeBackCertifiedTie.lean`) and ConvNeXt-T. With it, the reading of
`b0_grad_float_le` (`EfficientNetBackFloatBudget.lean`) stops being *"every piece of this chain
is the certified gradient"* and becomes **"the chain IS the certified whole-net gradient"**.

⚠ **It had to wait for the padding re-spelling.** Until 2026-09-05 the chain reversed a
`flatConvStride2` stem while every shipped B0 artifact emits the XLA-`SAME` phase, so this tie
would have certified a program no artifact runs
(`planning/xla_same_respell_and_blueprint_audit.md`, step 7 waits on step 3a).

## The four pieces

1. `efficientnetB_has_vjp` — the generic five-stage apex `head ∘ b3 ∘ b2 ∘ b1 ∘ stem`, four
   `vjp_comp`s and nothing else. Dimension-generic and parametric in every slot, the batched
   `HasVJP` peer of `mobilenetv2PC_has_vjp_at`'s ten-stage `HasVJPAt` chain. B0's chain needs no
   smooth point: swish and the SE sigmoid are differentiable everywhere, where MobileNetV2's
   relu6 and ResNet's relu are not.
2. `stemBBack_eq_vjp_backward` / `headFwdBBack_eq_vjp_backward` — the two concrete endpoint
   ties. Each is one `rw` of a per-example leaf tie and then `rfl`: the batched stage's VJP is
   `vjp_comp`-built, so its backward already reduces to the composition of the stage backwards,
   and `batchMap`'s VJP reduces to the leaf backward applied row-wise.
3. `efficientnetInputGradB_eq_efficientnetForwardB_vjp` — the tie, with the stem and head
   concrete and the three MBConv blocks opaque, exactly the `mnv2InputGrad_eq_mobilenetv2_vjp`
   shape.
4. `efficientnetForwardB_has_vjp_committed` — the apex carried across
   `efficientnetForwardB_eq_chain` to the committed forward, so the shape check is load-bearing
   rather than cited.

⭐ **`batchMap_has_vjp`'s transport does not block the reduction, and the planning note that
said it would is withdrawn.** It is built as `(batchMap_eq_rowwiseFlat f).symm ▸
hasVJPMat_to_hasVJP (rowwise_has_vjp_mat …)`, and §5's standing trap is that an `Eq.mpr` blocks
`.backward` from reducing. It does not here: the transported equation holds by `funext … ; rfl`,
and proof irrelevance is definitional in Lean, so `.backward` reduces straight through the `▸`
to the leaf backward applied row-wise — checked as a bare `rfl`, and every tie below relies on it.

⛔ **But size, not the transport, is what a `▸` costs you here.** The same argument says the
`▸` in piece 4 is harmless, and at the whole-net type it is not *affordable*: a `rfl` between
`efficientnetForwardB_has_vjp_committed`'s `.backward` and the untransported witness's exhausts
the kernel's deterministic budget, as does instantiating the tie's three block slots at the
concrete MBConv blocks. So the tie is stated on the chain, with the blocks opaque — the
MobileNetV2 discipline — and `efficientnetForwardB_eq_chain` is what says the chain is the
committed net. Without that `rfl` the tie would be a statement about five variables, which is how
ResNet-34's tie reversed the wrong pool for a month (§3.10).

⭐ **General `N`.** `planning/float_budget_numbers.md` section 5 item 3 scoped this tie at
`N = 1`, because the FLOAT chain's BatchNorm slot needs a batched float leaf that does not exist.
The certified chain's BatchNorm slot is `bnBatchLA_has_vjp`, which exists at every `N`, so the
tie is stated for every batch size. The float number is a separate statement and is unmoved.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The generic five-stage apex
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network EfficientNet-B0 VJP.** The VJP of a five-stage chain

      `head ∘ b3 ∘ b2 ∘ b1 ∘ stem`

    — the batched representative B0: strided stem, MBConv1, MBConv6-strided, MBConv6-residual,
    head. Four `vjp_comp`s, dimension-generic and parametric in every component. `HasVJP`
    everywhere, with no smooth point: unlike MobileNetV2's relu6 chain, every activation on
    this path (swish, the SE sigmoid) is differentiable on all of `ℝ`. -/
noncomputable def efficientnetB_has_vjp {s0 s1 s2 s3 s4 s5 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3)
    (b3 : Vec s3 → Vec s4) (head : Vec s4 → Vec s5)
    (dstem : Differentiable ℝ stem) (db1 : Differentiable ℝ b1) (db2 : Differentiable ℝ b2)
    (db3 : Differentiable ℝ b3) (dhead : Differentiable ℝ head)
    (hstem : HasVJP stem) (hb1 : HasVJP b1) (hb2 : HasVJP b2) (hb3 : HasVJP b3)
    (hhead : HasVJP head) :
    HasVJP (head ∘ b3 ∘ b2 ∘ b1 ∘ stem) :=
  let v1 := vjp_comp _ _ dstem db1 hstem hb1
  let v2 := vjp_comp _ _ (db1.comp dstem) db2 v1 hb2
  let v3 := vjp_comp _ _ (db2.comp (db1.comp dstem)) db3 v2 hb3
  vjp_comp _ _ (db3.comp (db2.comp (db1.comp dstem))) dhead v3 hhead

-- ════════════════════════════════════════════════════════════════
-- § The two concrete endpoint ties
-- ════════════════════════════════════════════════════════════════

/-- **The STEM tie.** The hand-written `batchMap (flatConvStride2XlaBack) ∘ bnBack ∘ swishBack`
    IS `stemB`'s certified backward. One `rw` of the odd-phase leaf tie, then `rfl` — the
    stage's VJP is `vjp_comp`-built so its backward is already the composition, and the leaf's
    backward is input-independent (a convolution is linear), so the row-wise `batchMap` lift
    matches at every saved input. -/
theorem stemBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    (StableHLO.batchMap N (flatConvStride2XlaBack (h := h) (w := w) W)
      ∘ (bnBatchLA_has_vjp N oc h w ε hε γ β).backward
          (StableHLO.batchMap N (flatConvStride2Xla W b) x)
      ∘ (swish_has_vjp (N * (oc * h * w))).backward
          (StableHLO.bnBatchLA N oc h w ε γ β (StableHLO.batchMap N (flatConvStride2Xla W b) x)))
      = (stemB_has_vjp N (h := h) (w := w) W b ε hε γ β).backward x := by
  rw [flatConvStride2XlaBack_eq_vjp_backward hkH hkW W b (fun _ => 0)]
  rfl

/-- **The HEAD tie.** The four-stage hand chain `batchMap (convFlatBack) ∘ bnBack ∘ swishBack ∘
    batchMap gapBack ∘ batchMap (dense Wᵀ 0)` IS `headFwdB`'s certified backward. `gapBack` needs
    no rewrite: it is definitionally the global-average-pool VJP's backward. -/
theorem headFwdBBack_eq_vjp_backward {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (x : Vec (N * (c * h * w))) :
    ((StableHLO.batchMap N (convFlatBack (h := h) (w := w) Wh)
        ∘ (bnBatchLA_has_vjp N oc h w εh hεh γh βh).backward
            (StableHLO.batchMap N (flatConv Wh bh) x)
        ∘ (swish_has_vjp (N * (oc * h * w))).backward
            (StableHLO.bnBatchLA N oc h w εh γh βh (StableHLO.batchMap N (flatConv Wh bh) x)))
      ∘ StableHLO.batchMap N (gapBack oc h w)
      ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wfc) (0 : Vec oc)))
      = (headFwdB_has_vjp N (h := h) (w := w) Wh bh εh hεh γh βh Wfc bfc).backward x := by
  rw [convFlatBack_eq_vjp_backward (by simp) (by simp) Wh bh (fun _ => 0),
      dense_transpose_eq_vjp_backward Wfc bfc (fun _ => 0)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The tie
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 400000 in
set_option maxHeartbeats 2000000 in
/-- ⭐⭐ **THE TIE — `efficientnetInputGradB` IS the certified whole-net B0 gradient.** The
    committed backward chain, with its stem/head BatchNorm and swish slots filled by the
    certified per-op backwards and its three MBConv blocks left opaque, equals the backward of
    `efficientnetB_has_vjp` instantiated at the five stages `efficientnetForwardB_eq_chain`
    proves the committed forward composes.

    Everything the reader has to trust is in the statement: the LHS is the deployed chain
    (`EfficientNetWholeBackFloatBridge.lean`, the same term `efficientnet_grad_floatBridges` and
    the budget are about), the RHS is `HasVJP.backward` of a composition of certified VJPs, and
    `HasVJP.correct` pins that to the true Jacobian-transpose. -/
theorem efficientnetInputGradB_eq_efficientnetForwardB_vjp
    (N : Nat)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (b1 : Vec (N * (32 * 112 * 112)) → Vec (N * (16 * 112 * 112)))
    (b2 : Vec (N * (16 * 112 * 112)) → Vec (N * (24 * 56 * 56)))
    (b3 : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (db1 : Differentiable ℝ b1) (db2 : Differentiable ℝ b2) (db3 : Differentiable ℝ b3)
    (hb1 : HasVJP b1) (hb2 : HasVJP b2) (hb3 : HasVJP b3)
    (x : Vec (N * (3 * 224 * 224))) :
    efficientnetInputGradB N Ws Wh Wfc
      ((bnBatchLA_has_vjp N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((swish_has_vjp (N * (32 * 112 * 112))).backward
        (StableHLO.bnBatchLA N 32 112 112 εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x)))
      ((bnBatchLA_has_vjp N 1280 56 56 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh)
          (b3 (b2 (b1 (stemB N (h := 112) (w := 112) Ws bs εs γs βs x))))))
      ((swish_has_vjp (N * (1280 * 56 * 56))).backward
        (StableHLO.bnBatchLA N 1280 56 56 εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh)
            (b3 (b2 (b1 (stemB N (h := 112) (w := 112) Ws bs εs γs βs x)))))))
      (hb1.backward (stemB N (h := 112) (w := 112) Ws bs εs γs βs x))
      (hb2.backward (b1 (stemB N (h := 112) (w := 112) Ws bs εs γs βs x)))
      (hb3.backward (b2 (b1 (stemB N (h := 112) (w := 112) Ws bs εs γs βs x))))
      = (efficientnetB_has_vjp
          (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3
          (headFwdB N (h := 56) (w := 56) Wh bh εh γh βh Wfc bfc)
          (stemB_differentiable N (h := 112) (w := 112) Ws bs εs hεs γs βs) db1 db2 db3
          (headFwdB_differentiable N (h := 56) (w := 56) Wh bh εh hεh γh βh Wfc bfc)
          (stemB_has_vjp N (h := 112) (w := 112) Ws bs εs hεs γs βs) hb1 hb2 hb3
          (headFwdB_has_vjp N (h := 56) (w := 56) Wh bh εh hεh γh βh Wfc bfc)).backward x := by
  unfold efficientnetInputGradB
  rw [stemBBack_eq_vjp_backward (N := N) (h := 112) (w := 112) (by decide) (by decide)
        Ws bs εs hεs γs βs x,
      headFwdBBack_eq_vjp_backward (N := N) (h := 56) (w := 56) Wh bh εh hεh γh βh Wfc bfc
        (b3 (b2 (b1 (stemB N (h := 112) (w := 112) Ws bs εs γs βs x))))]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The tie at the CONCRETE blocks, against the COMMITTED forward
-- ════════════════════════════════════════════════════════════════

/-- **The apex, restated on the committed `efficientnetForwardB`.** This is where
    `efficientnetForwardB_eq_chain` does load-bearing work rather than being cited in prose: the
    apex is proven about the `∘`-chain, the committed forward is written in nested-application
    form, and this definition does not typecheck without the shape check saying they are the
    same function.

    ⚠ Do not try to compute with it. The `▸` is harmless in principle —
    `efficientnetForwardB_eq_chain` holds pointwise by `rfl` — but at this type the kernel cannot
    afford to reduce through it: `(this).backward x = (efficientnetForwardB_has_vjp …).backward x`
    is a deterministic timeout as a `rfl`. This definition is here to say the committed forward
    HAS the certified VJP; the tie is stated on the chain. -/
noncomputable def efficientnetForwardB_has_vjp_committed
    (N : Nat)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (εd1 : ℝ) (hεd1 : 0 < εd1) (γd1 βd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (εp1 : ℝ) (hεp1 : 0 < εp1) (γp1 βp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (εe2 : ℝ) (hεe2 : 0 < εe2) (γe2 βe2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 : ℝ) (hεd2 : 0 < εd2) (γd2 βd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 : ℝ) (hεp2 : 0 < εp2) (γp2 βp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (εe3 : ℝ) (hεe3 : 0 < εe3) (γe3 βe3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (εd3 : ℝ) (hεd3 : 0 < εd3) (γd3 βd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (εp3 : ℝ) (hεp3 : 0 < εp3) (γp3 βp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10) :
    HasVJP (efficientnetForwardB
      N Ws bs εs γs βs Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2
      γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3
      bd3 εd3 γd3 βd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 γp3 βp3 Wh bh εh γh βh Wfc bfc) :=
  (funext (efficientnetForwardB_eq_chain
      N Ws bs εs γs βs Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2
      γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3
      bd3 εd3 γd3 βd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 γp3 βp3 Wh bh εh γh βh Wfc bfc)).symm ▸
    efficientnetForwardB_has_vjp
      N Ws bs εs hεs γs βs Wd1 bd1 εd1 hεd1 γd1 βd1 Wz1a bz1a Wz1b bz1b Wp1 bp1 εp1 hεp1 γp1 βp1
      We2 be2 εe2 hεe2 γe2 βe2 Wd2 bd2 εd2 hεd2 γd2 βd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 hεp2
      γp2 βp2 We3 be3 εe3 hεe3 γe3 βe3 Wd3 bd3 εd3 hεd3 γd3 βd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3
      hεp3 γp3 βp3 Wh bh εh hεh γh βh Wfc bfc

end Proofs
