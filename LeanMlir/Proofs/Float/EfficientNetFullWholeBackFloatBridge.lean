import LeanMlir.Proofs.Float.EfficientNetWholeBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE-NET input-gradient backward of the PAPER EfficientNet-B0

`EfficientNetWholeBackFloatBridge.lean` defines `efficientnetInputGradB`, the input-gradient
chain of the three-block representative, and threads a `FloatBridgesTo` through it. This file
is the same for the net `efficientnetForwardB_full` IS — **all sixteen MBConv blocks** of the
`[t,c,n,s,k]` table, head 320→1280 at 7×7 — and it exists for one reason: so that the sixteen-
block certified tie (`Foundation/EfficientNetFullWholeBackCertifiedTie.lean`) is a statement
about a named chain of the same shape as the representative's, with the stem, head, GAP and
classifier endpoints concrete and `batchMap`-lifted and the sixteen block backwards supplied.

⛔ **No number is stated about this chain**, and that is a decision rather than a limit
(`planning/proofs_tier_to_paper_nets.md` 3.3(b)): at the shipped leaves the sixteen-block
backward's certified WINDOW is `9.112·10²⁶⁴⁸` (`b0_full_back_chain`), which — now that
`norm_num`'s "ceiling" is known to be the `exponentiation.threshold` option and not a wall —
is a numeral Lean could carry and nobody should write down. The chain is here for the tie.
-/

namespace Proofs

open FloatModel

variable {nCls : Nat}

/-- **The batched whole-net input-gradient backward of the sixteen-block EfficientNet-B0** —
    the reverse of `efficientnetForwardB_full = head ∘ b16 ∘ … ∘ b1 ∘ stem`: classifier-back →
    GAP-back → head-conv-bn-swish-back → the sixteen MBConv block backs → stem-conv-bn-swish-back.
    The block backs and the stem/head BN+swish backs are supplied; the conv/GAP/dense leaves are
    concrete, `batchMap`-lifted over the `N` examples, the stem at the XLA-`SAME` phase. -/
noncomputable def efficientnetInputGradB_full (N : Nat)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 nCls)
    (bnBs swBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh swBh : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (b1B : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4B : Vec (N * (40 * 28 * 28)) → Vec (N * (24 * 56 * 56)))
    (b5B : Vec (N * (40 * 28 * 28)) → Vec (N * (40 * 28 * 28)))
    (b6B : Vec (N * (80 * 14 * 14)) → Vec (N * (40 * 28 * 28)))
    (b7B : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b8B : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b9B : Vec (N * (112 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b10B : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b11B : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b12B : Vec (N * (192 * 7 * 7)) → Vec (N * (112 * 14 * 14)))
    (b13B : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b14B : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b15B : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b16B : Vec (N * (320 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    : Vec (N * nCls) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N (flatConvStride2XlaBack (h := 112) (w := 112) Ws) ∘ bnBs ∘ swBs)
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B
  ∘ (StableHLO.batchMap N (convFlatBack (h := 7) (w := 7) Wh) ∘ bnBh ∘ swBh)
  ∘ StableHLO.batchMap N (gapBack 1280 7 7)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wfc) (0 : Vec 1280))

/-- **The float sixteen-block input-gradient skeleton** — `efficientnetInputGradB_full` with
    each concrete `batchMap`-lifted slot replaced by the model's rounded peer and each supplied
    backward by its float map. -/
noncomputable def efficientnetInputGradBF_full (N : Nat) (M : FloatModel)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 nCls)
    (bnBsF swBsF : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBhF swBhF : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (b1BF : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2BF : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3BF : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4BF : Vec (N * (40 * 28 * 28)) → Vec (N * (24 * 56 * 56)))
    (b5BF : Vec (N * (40 * 28 * 28)) → Vec (N * (40 * 28 * 28)))
    (b6BF : Vec (N * (80 * 14 * 14)) → Vec (N * (40 * 28 * 28)))
    (b7BF : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b8BF : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b9BF : Vec (N * (112 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b10BF : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b11BF : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b12BF : Vec (N * (192 * 7 * 7)) → Vec (N * (112 * 14 * 14)))
    (b13BF : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b14BF : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b15BF : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b16BF : Vec (N * (320 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    : Vec (N * nCls) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N
      (M.flatConvF (h := 2 * 112) (w := 2 * 112) (IR.reverseSwap Ws) (fun _ => 0)
        ∘ decimateOddBack 32 112 112) ∘ bnBsF ∘ swBsF)
  ∘ b1BF ∘ b2BF ∘ b3BF ∘ b4BF ∘ b5BF ∘ b6BF ∘ b7BF ∘ b8BF ∘ b9BF ∘ b10BF ∘ b11BF ∘ b12BF ∘ b13BF ∘ b14BF ∘ b15BF ∘ b16BF
  ∘ (StableHLO.batchMap N
      (M.flatConvF (h := 7) (w := 7) (IR.reverseSwap Wh) (fun _ => 0)) ∘ bnBhF ∘ swBhF)
  ∘ StableHLO.batchMap N (gapBackF M 1280 7 7)
  ∘ StableHLO.batchMap N (M.dense (Mat.transpose Wfc) (0 : Vec 1280))

set_option maxRecDepth 100000 in
/-- **The sixteen-block backward float-bridges TO its float skeleton** — one `.comp` thread over
    the concrete endpoints, the supplied stem/head BN+swish backwards and the sixteen supplied
    block backwards; `efficientnet_grad_floatBridgesTo` at the paper depth. -/
noncomputable def efficientnet_full_grad_floatBridgesTo (N : Nat) (M : FloatModel)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 nCls)
    (bnBs swBs bnBsF swBsF : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh swBh bnBhF swBhF : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (b1B b1BF : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B b2BF : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B b3BF : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4B b4BF : Vec (N * (40 * 28 * 28)) → Vec (N * (24 * 56 * 56)))
    (b5B b5BF : Vec (N * (40 * 28 * 28)) → Vec (N * (40 * 28 * 28)))
    (b6B b6BF : Vec (N * (80 * 14 * 14)) → Vec (N * (40 * 28 * 28)))
    (b7B b7BF : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b8B b8BF : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b9B b9BF : Vec (N * (112 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b10B b10BF : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b11B b11BF : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b12B b12BF : Vec (N * (192 * 7 * 7)) → Vec (N * (112 * 14 * 14)))
    (b13B b13BF : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b14B b14BF : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b15B b15BF : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b16B b16BF : Vec (N * (320 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    {ws wh wfc : ℝ} (hws : 0 ≤ ws) (hwh : 0 ≤ wh) (hwfc : 0 ≤ wfc) (hnc : 0 < nCls)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ wh)
    (hWfc : ∀ i j, |Wfc i j| ≤ wfc)
    (hbnBs : FloatBridgesTo bnBs bnBsF) (hswBs : FloatBridgesTo swBs swBsF)
    (hbnBh : FloatBridgesTo bnBh bnBhF) (hswBh : FloatBridgesTo swBh swBhF)
    (hb1B : FloatBridgesTo b1B b1BF)
    (hb2B : FloatBridgesTo b2B b2BF)
    (hb3B : FloatBridgesTo b3B b3BF)
    (hb4B : FloatBridgesTo b4B b4BF)
    (hb5B : FloatBridgesTo b5B b5BF)
    (hb6B : FloatBridgesTo b6B b6BF)
    (hb7B : FloatBridgesTo b7B b7BF)
    (hb8B : FloatBridgesTo b8B b8BF)
    (hb9B : FloatBridgesTo b9B b9BF)
    (hb10B : FloatBridgesTo b10B b10BF)
    (hb11B : FloatBridgesTo b11B b11BF)
    (hb12B : FloatBridgesTo b12B b12BF)
    (hb13B : FloatBridgesTo b13B b13BF)
    (hb14B : FloatBridgesTo b14B b14BF)
    (hb15B : FloatBridgesTo b15B b15BF)
    (hb16B : FloatBridgesTo b16B b16BF)
    :
    FloatBridgesTo
      (efficientnetInputGradB_full N Ws Wh Wfc bnBs swBs bnBh swBh b1B b2B b3B b4B b5B b6B b7B b8B b9B b10B b11B b12B b13B b14B b15B b16B)
      (efficientnetInputGradBF_full N M Ws Wh Wfc bnBsF swBsF bnBhF swBhF b1BF b2BF b3BF b4BF b5BF b6BF b7BF b8BF b9BF b10BF b11BF b12BF b13BF b14BF b15BF b16BF) := by
  unfold efficientnetInputGradB_full efficientnetInputGradBF_full
  have hstem := (hswBs.comp hbnBs).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_flatConvStride2XlaBack (h := 112) (w := 112) M Ws hws (by positivity) hWs))
  have hhead := (hswBh.comp hbnBh).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_convBack (h := 7) (w := 7) M Wh hwh (by positivity) hWh))
  have h0 := (FloatBridgesTo.batchMap N
      (floatBridgesTo_linBack M Wfc hwfc hnc hWfc)).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_gapBack M 1280 7 7 (by norm_num) (by norm_num) (by norm_num)))
  have hH := h0.comp hhead
  have hB16 := hH.comp hb16B
  have hB15 := hB16.comp hb15B
  have hB14 := hB15.comp hb14B
  have hB13 := hB14.comp hb13B
  have hB12 := hB13.comp hb12B
  have hB11 := hB12.comp hb11B
  have hB10 := hB11.comp hb10B
  have hB9 := hB10.comp hb9B
  have hB8 := hB9.comp hb8B
  have hB7 := hB8.comp hb7B
  have hB6 := hB7.comp hb6B
  have hB5 := hB6.comp hb5B
  have hB4 := hB5.comp hb4B
  have hB3 := hB4.comp hb3B
  have hB2 := hB3.comp hb2B
  have hB1 := hB2.comp hb1B
  exact hB1.comp hstem

end Proofs
