import LeanMlir.Proofs.Float.MobileNetV2BackFloatBridge
import LeanMlir.Proofs.Float.BnBatchFloatBridge
import LeanMlir.Proofs.Float.EfficientNetFullWholeBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE MobileNetV2 backward at TRUE BATCH-NORM

`Resnet34WholeBackFloatBridge.lean`'s MobileNetV2 peer is the PER-EXAMPLE chain
`mnv2PaperInputGrad` (`Foundation/MobileNetV2PaperWholeBackCertifiedTie.lean`), the reverse of
`mobilenetv2ForwardPaper` — the net the retired `MobileNetV2Render` emitted. This file is the
same for the net the shipped trainers run: `mobilenetv2ForwardB_full`, all seventeen bottlenecks
at **`bnBatchLA`**, at a variable batch `N`, and it exists so that the batch-BN certified tie
(`Foundation/MobileNetV2WholeBackCertifiedTieB.lean`, tier T6 of
`planning/proofs_tier_to_paper_nets.md` §4.2) is a statement about a NAMED chain.

⭐ **Cheaper than ResNet-34's batched chain, and for the same structural reason 4.2b found:
MobileNetV2 has no stem pool.** Every concrete endpoint here is `batchMap N` of a per-example
leaf — a convolution, a GAP and a dense are batch-separable and their float peers lift by
`FloatBridgesTo.batchMap` — so this file needs none of `Resnet34WholeBackFloatBridgeB.lean`'s
row-indexed `batchMapAux` machinery.

⚠ Padding is XLA-`SAME` at the stem (`flatConvStride2XlaBack`, whose float peer scatters with
`decimateOddBack`), where ResNet-34's is symmetric. Identical types, different certificates —
the one convention this chain and r34's do not share.

⛔ **No number is stated about this chain.** §4.2's T5 is a float budget and
`planning/float_budget_numbers.md` closed that thread; the seventeen-block per-example backward
had no statable number either (§3.2(b): window `1.246e323`, and no cap rescues a backward). The
chain is here for the tie.
-/

namespace Proofs

open FloatModel

/-- **The batched whole-net input-gradient backward of the seventeen-bottleneck MobileNetV2** —
    the exact reverse of `mobilenetv2ForwardB_full = head ∘ b17 ∘ … ∘ b1 ∘ stem`: dense-back →
    GAP-back → the head's relu6 mask, BatchNorm back and 1×1 conv back → the seventeen bottleneck
    backwards → the stem's relu6 mask, BatchNorm back and XLA-`SAME` 3×3/s2 conv back. The block
    backwards and the two BatchNorm backs are supplied; the conv, GAP and dense leaves are
    concrete and lifted over the `N` examples. -/
noncomputable def mnv2InputGradB (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 nCls)
    (bnBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (b1B : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4B : Vec (N * (32 * 28 * 28)) → Vec (N * (24 * 56 * 56)))
    (b5B : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b6B : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b7B : Vec (N * (64 * 14 * 14)) → Vec (N * (32 * 28 * 28)))
    (b8B : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b9B : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b10B : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b11B : Vec (N * (96 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b12B : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b13B : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b14B : Vec (N * (160 * 7 * 7)) → Vec (N * (96 * 14 * 14)))
    (b15B : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b16B : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b17B : Vec (N * (320 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (m_stem : Fin (N * (32 * 112 * 112)) → Prop) [DecidablePred m_stem]
    (m_head : Fin (N * (1280 * 7 * 7)) → Prop) [DecidablePred m_head] :
    Vec (N * nCls) → Vec (N * (3 * (2 * 112) * (2 * 112))) :=
  (StableHLO.batchMap N (flatConvStride2XlaBack (h := 112) (w := 112) Ws)
      ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B ∘ b17B
  ∘ (StableHLO.batchMap N (convFlatBack (h := 7) (w := 7) Wh)
      ∘ bnBh ∘ reluMaskBack m_head)
  ∘ StableHLO.batchMap N (gapBack 1280 7 7)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wfc) (0 : Vec 1280))

/-- **The float seventeen-bottleneck batched skeleton** — `mnv2InputGradB` with each concrete slot
    replaced by the model's rounded peer and each supplied backward by its float map.
    ⛔ The stem's scatter is `decimateOddBack`, the XLA-`SAME` phase's, not `decimateBack`. -/
noncomputable def mnv2InputGradBF (N : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 nCls)
    (bnBsF : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBhF : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (b1BF : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2BF : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3BF : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4BF : Vec (N * (32 * 28 * 28)) → Vec (N * (24 * 56 * 56)))
    (b5BF : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b6BF : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b7BF : Vec (N * (64 * 14 * 14)) → Vec (N * (32 * 28 * 28)))
    (b8BF : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b9BF : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b10BF : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b11BF : Vec (N * (96 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b12BF : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b13BF : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b14BF : Vec (N * (160 * 7 * 7)) → Vec (N * (96 * 14 * 14)))
    (b15BF : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b16BF : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b17BF : Vec (N * (320 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (m_stem : Fin (N * (32 * 112 * 112)) → Prop) [DecidablePred m_stem]
    (m_head : Fin (N * (1280 * 7 * 7)) → Prop) [DecidablePred m_head] :
    Vec (N * nCls) → Vec (N * (3 * (2 * 112) * (2 * 112))) :=
  (StableHLO.batchMap N
      (M.flatConvF (h := 2 * 112) (w := 2 * 112) (IR.reverseSwap Ws) (fun _ => 0)
        ∘ decimateOddBack 32 112 112)
      ∘ bnBsF ∘ reluMaskBack m_stem)
  ∘ b1BF ∘ b2BF ∘ b3BF ∘ b4BF ∘ b5BF ∘ b6BF ∘ b7BF ∘ b8BF ∘ b9BF ∘ b10BF ∘ b11BF ∘ b12BF ∘ b13BF ∘ b14BF ∘ b15BF ∘ b16BF ∘ b17BF
  ∘ (StableHLO.batchMap N
      (M.flatConvF (h := 7) (w := 7) (IR.reverseSwap Wh) (fun _ => 0))
      ∘ bnBhF ∘ reluMaskBack m_head)
  ∘ StableHLO.batchMap N (gapBackF M 1280 7 7)
  ∘ StableHLO.batchMap N (M.dense (Mat.transpose Wfc) (0 : Vec 1280))

set_option maxRecDepth 100000 in
/-- **The batched seventeen-bottleneck backward float-bridges TO its float skeleton** — one
    `.comp` thread over the concrete endpoints, the two supplied BatchNorm backwards and the
    seventeen supplied block backwards. `mnv2_grad_floatBridgesTo`'s batch-BN peer. -/
noncomputable def mnv2_grad_floatBridgesToB (N : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 nCls)
    (bnBs bnBsF : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh bnBhF : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (b1B b1BF : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B b2BF : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B b3BF : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4B b4BF : Vec (N * (32 * 28 * 28)) → Vec (N * (24 * 56 * 56)))
    (b5B b5BF : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b6B b6BF : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b7B b7BF : Vec (N * (64 * 14 * 14)) → Vec (N * (32 * 28 * 28)))
    (b8B b8BF : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b9B b9BF : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b10B b10BF : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b11B b11BF : Vec (N * (96 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b12B b12BF : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b13B b13BF : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b14B b14BF : Vec (N * (160 * 7 * 7)) → Vec (N * (96 * 14 * 14)))
    (b15B b15BF : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b16B b16BF : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b17B b17BF : Vec (N * (320 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (m_stem : Fin (N * (32 * 112 * 112)) → Prop) [DecidablePred m_stem]
    (m_head : Fin (N * (1280 * 7 * 7)) → Prop) [DecidablePred m_head]
    {ws wh wfc : ℝ} (hws : 0 ≤ ws) (hwh : 0 ≤ wh) (hwfc : 0 ≤ wfc) (hnc : 0 < nCls)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ wh)
    (hWfc : ∀ i j, |Wfc i j| ≤ wfc)
    (hbnBs : FloatBridgesTo bnBs bnBsF) (hbnBh : FloatBridgesTo bnBh bnBhF)
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
    (hb17B : FloatBridgesTo b17B b17BF) :
    FloatBridgesTo
      (mnv2InputGradB N Ws Wh Wfc bnBs bnBh b1B b2B b3B b4B b5B b6B b7B b8B b9B b10B b11B b12B b13B b14B b15B b16B b17B m_stem m_head)
      (mnv2InputGradBF N M Ws Wh Wfc bnBsF bnBhF b1BF b2BF b3BF b4BF b5BF b6BF b7BF b8BF b9BF b10BF b11BF b12BF b13BF b14BF b15BF b16BF b17BF m_stem m_head) := by
  unfold mnv2InputGradB mnv2InputGradBF
  have hstem := ((floatBridgesTo_reluMaskBack m_stem).comp hbnBs).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_flatConvStride2XlaBack (h := 112) (w := 112) M Ws hws (by norm_num) hWs))
  have hhead := ((floatBridgesTo_reluMaskBack m_head).comp hbnBh).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_convBack (h := 7) (w := 7) M Wh hwh (by norm_num) hWh))
  have h0 := (FloatBridgesTo.batchMap N (floatBridgesTo_linBack M Wfc hwfc hnc hWfc)).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_gapBack M 1280 7 7 (by norm_num) (by norm_num) (by norm_num)))
  have hB18 := h0.comp hhead
  have hB17 := hB18.comp hb17B
  have hB16 := hB17.comp hb16B
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
