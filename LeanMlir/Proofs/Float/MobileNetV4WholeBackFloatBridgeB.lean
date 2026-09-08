import LeanMlir.Proofs.Float.EfficientNetFullWholeBackFloatBridge
import LeanMlir.Proofs.Float.BnBatchFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE MobileNetV4-Conv-M backward at TRUE BATCH-NORM

`MobileNetV4FullBVJP.lean` gives MobileNetV4-Conv-M's batch-BN forward a certified whole-net
`HasVJPAt` (T1). This file names the backward that VJP is about: `mnv4InputGradB`, the exact
reverse of `mobilenetv4ForwardB_full = head ∘ b21 ∘ … ∘ b1 ∘ fused ∘ stem`, at a variable batch
`N` and a variable class count. It exists so the certified tie
(`Foundation/MobileNetV4WholeBackCertifiedTieB.lean`, tier **T6** of
`planning/mnv4_proofs_tier.md` §Session 3) is a statement about a NAMED chain.

⚠⚠ **No accuracy is quoted for this net.** Conv-M has no Imagenette run and no verified ImageNet
run; what pins the artifact to the reference's function is the pair of ties re-run 2026-09-07
(forward `max |Δ| = 3.770e-06`, gradient inside the reference's own fp32 floor).

## ⭐⭐ The chain is stated TO THE IMAGE, and its last node is one step past the artifact

`mnv4StemTiedB` (T3) records that MobileNetV4's committed backward **stops at the stem conv's
WEIGHT gradient**: no render emits a gradient into `%x`, so there is no
`convStridedXlaBackBatched` token and the artifact never computes the input gradient. This chain
runs one `flatConvStride2XlaBack` past that point, exactly as EfficientNet-B0's does at the
identical XLA-`SAME` stem (`efficientnetInputGradB_full`) — the mathematical input gradient is the
statable object, and the stem-BN cotangent it factors through is the one the artifact has and
`mnv4StemCotN` already ties. Either choice is honest; this header is which.

## ⭐⭐ Not one new float leaf

MobileNetV4's stem is EfficientNet-B0's (3×3/s2 at the XLA-`SAME` phase, so
`floatBridgesTo_flatConvStride2XlaBack` and its `decimateOddBack` scatter), its two head convs are
1×1 plain convolutions (`floatBridgesTo_convBack`), and its GAP-and-dense tail is ResNet-34's
(`floatBridgesTo_gapBack`, `floatBridgesTo_linBack`). Every concrete endpoint here is
`batchMap N` of a per-example leaf, so the whole file is `FloatBridgesTo.batchMap` and `.comp` —
none of ResNet-34's row-indexed `batchMapAux` machinery (its float side was deleted 2026-09-08), for the reason
4.2b found on MobileNetV2: **no stem pool.**

⚠ **Two padding phases in one chain.** The stem scatters with `decimateOddBack` (XLA-`SAME`); the
fused stage and the three strided depthwises are SYMMETRIC and sit inside the supplied block
backwards, where their phase is the block's business. Do not "tidy" one to match the other.

⚠ **The fused stage is a supplied slot, not a concrete endpoint.** It is MobileNetV4's stage 0 —
a conv-bn-swish and a 1×1 project — and it is opaque here for the same reason the 21 UIB blocks
are: the tie composes certified backwards, it does not re-derive them.

⛔ **No number is stated about this chain.** `planning/float_budget_numbers.md` closed the
float-budget thread by user decision on 2026-09-05, and a batched backward number moves with `N`
besides. The chain is here for the tie.
-/

namespace Proofs

open FloatModel


/-- **The batched whole-net input-gradient backward of MobileNetV4-Conv-M** — the exact reverse
    of `mobilenetv4ForwardB_full`: dense-back → GAP-back → the second head conv's relu mask,
    BatchNorm back and 1×1 conv back → the first head conv's three → the twenty-one UIB block
    backwards → the fused stage's → the stem's relu mask, BatchNorm back and XLA-`SAME` 3×3/s2
    conv back.

    The fused stage's and the twenty-one blocks' backwards are supplied, as are the three
    BatchNorm backs; the conv, GAP and dense leaves are concrete and lifted over the `N`
    examples. ⚠ **Two head convs**: Conv-M's head is `%h1W` (256 → 960) then `%hW` (960 → 1280)
    before the pool, so this chain has two conv-BN-relu endpoints at 7×7 where MobileNetV2's has
    one and ResNet-34's has none. -/
noncomputable def mnv4InputGradB (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (Wh1 : Kernel4 960 256 1 1) (Wh : Kernel4 1280 960 1 1)
    (Wd : Mat 1280 nCls)
    (bnBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh1 : Vec (N * (960 * 7 * 7)) → Vec (N * (960 * 7 * 7)))
    (bnBh : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (fB : Vec (N * (48 * 56 * 56)) → Vec (N * (32 * 112 * 112)))
    (b1B : Vec (N * (80 * 28 * 28)) → Vec (N * (48 * 56 * 56)))
    (b2B : Vec (N * (80 * 28 * 28)) → Vec (N * (80 * 28 * 28)))
    (b3B : Vec (N * (160 * 14 * 14)) → Vec (N * (80 * 28 * 28)))
    (b4B : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b5B : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b6B : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b7B : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b8B : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b9B : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b10B : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b11B : Vec (N * (256 * 7 * 7)) → Vec (N * (160 * 14 * 14)))
    (b12B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b13B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b14B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b15B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b16B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b17B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b18B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b19B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b20B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b21B : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (m_stem : Fin (N * (32 * 112 * 112)) → Prop) [DecidablePred m_stem]
    (m_h1 : Fin (N * (960 * 7 * 7)) → Prop) [DecidablePred m_h1]
    (m_h : Fin (N * (1280 * 7 * 7)) → Prop) [DecidablePred m_h] :
    Vec (N * nCls) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N (flatConvStride2XlaBack (h := 112) (w := 112) Ws)
      ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ fB
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B ∘ b17B ∘ b18B ∘ b19B ∘ b20B ∘ b21B
  ∘ (StableHLO.batchMap N (convFlatBack (h := 7) (w := 7) Wh1)
      ∘ bnBh1 ∘ reluMaskBack m_h1)
  ∘ (StableHLO.batchMap N (convFlatBack (h := 7) (w := 7) Wh)
      ∘ bnBh ∘ reluMaskBack m_h)
  ∘ StableHLO.batchMap N (gapBack 1280 7 7)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec 1280))


/-- **The float MobileNetV4-Conv-M input-gradient skeleton** — `mnv4InputGradB` with each concrete
    slot replaced by the model's rounded peer and each supplied backward by its float map.
    ⛔ The stem's scatter is `decimateOddBack`, the XLA-`SAME` phase's, not `decimateBack`. -/
noncomputable def mnv4InputGradBF (N : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 32 3 3 3) (Wh1 : Kernel4 960 256 1 1) (Wh : Kernel4 1280 960 1 1)
    (Wd : Mat 1280 nCls)
    (bnBsF : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh1F : Vec (N * (960 * 7 * 7)) → Vec (N * (960 * 7 * 7)))
    (bnBhF : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (fBF : Vec (N * (48 * 56 * 56)) → Vec (N * (32 * 112 * 112)))
    (b1BF : Vec (N * (80 * 28 * 28)) → Vec (N * (48 * 56 * 56)))
    (b2BF : Vec (N * (80 * 28 * 28)) → Vec (N * (80 * 28 * 28)))
    (b3BF : Vec (N * (160 * 14 * 14)) → Vec (N * (80 * 28 * 28)))
    (b4BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b5BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b6BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b7BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b8BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b9BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b10BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b11BF : Vec (N * (256 * 7 * 7)) → Vec (N * (160 * 14 * 14)))
    (b12BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b13BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b14BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b15BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b16BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b17BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b18BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b19BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b20BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b21BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (m_stem : Fin (N * (32 * 112 * 112)) → Prop) [DecidablePred m_stem]
    (m_h1 : Fin (N * (960 * 7 * 7)) → Prop) [DecidablePred m_h1]
    (m_h : Fin (N * (1280 * 7 * 7)) → Prop) [DecidablePred m_h] :
    Vec (N * nCls) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N
      (M.flatConvF (h := 2 * 112) (w := 2 * 112) (IR.reverseSwap Ws) (fun _ => 0)
        ∘ decimateOddBack 32 112 112)
      ∘ bnBsF ∘ reluMaskBack m_stem)
  ∘ fBF
  ∘ b1BF ∘ b2BF ∘ b3BF ∘ b4BF ∘ b5BF ∘ b6BF ∘ b7BF ∘ b8BF ∘ b9BF ∘ b10BF ∘ b11BF ∘ b12BF ∘ b13BF ∘ b14BF ∘ b15BF ∘ b16BF ∘ b17BF ∘ b18BF ∘ b19BF ∘ b20BF ∘ b21BF
  ∘ (StableHLO.batchMap N
      (M.flatConvF (h := 7) (w := 7) (IR.reverseSwap Wh1) (fun _ => 0))
      ∘ bnBh1F ∘ reluMaskBack m_h1)
  ∘ (StableHLO.batchMap N
      (M.flatConvF (h := 7) (w := 7) (IR.reverseSwap Wh) (fun _ => 0))
      ∘ bnBhF ∘ reluMaskBack m_h)
  ∘ StableHLO.batchMap N (gapBackF M 1280 7 7)
  ∘ StableHLO.batchMap N (M.dense (Mat.transpose Wd) (0 : Vec 1280))


set_option maxRecDepth 100000 in
/-- **The batched MobileNetV4-Conv-M backward float-bridges TO its float skeleton** — one `.comp`
    thread over the concrete endpoints, the three supplied BatchNorm backwards, the supplied fused
    stage and the twenty-one supplied block backwards. `mnv2_grad_floatBridgesToB` at Conv-M's
    ladder, with the second head conv the only structural difference. -/
noncomputable def mnv4_grad_floatBridgesToB (N : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 32 3 3 3) (Wh1 : Kernel4 960 256 1 1) (Wh : Kernel4 1280 960 1 1)
    (Wd : Mat 1280 nCls)
    (bnBs bnBsF : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh1 bnBh1F : Vec (N * (960 * 7 * 7)) → Vec (N * (960 * 7 * 7)))
    (bnBh bnBhF : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (fB fBF : Vec (N * (48 * 56 * 56)) → Vec (N * (32 * 112 * 112)))
    (b1B b1BF : Vec (N * (80 * 28 * 28)) → Vec (N * (48 * 56 * 56)))
    (b2B b2BF : Vec (N * (80 * 28 * 28)) → Vec (N * (80 * 28 * 28)))
    (b3B b3BF : Vec (N * (160 * 14 * 14)) → Vec (N * (80 * 28 * 28)))
    (b4B b4BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b5B b5BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b6B b6BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b7B b7BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b8B b8BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b9B b9BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b10B b10BF : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b11B b11BF : Vec (N * (256 * 7 * 7)) → Vec (N * (160 * 14 * 14)))
    (b12B b12BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b13B b13BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b14B b14BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b15B b15BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b16B b16BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b17B b17BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b18B b18BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b19B b19BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b20B b20BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b21B b21BF : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (m_stem : Fin (N * (32 * 112 * 112)) → Prop) [DecidablePred m_stem]
    (m_h1 : Fin (N * (960 * 7 * 7)) → Prop) [DecidablePred m_h1]
    (m_h : Fin (N * (1280 * 7 * 7)) → Prop) [DecidablePred m_h]
    {ws wh1 wh wd : ℝ} (hws : 0 ≤ ws) (hwh1 : 0 ≤ wh1) (hwh : 0 ≤ wh) (hwd : 0 ≤ wd)
    (hnc : 0 < nCls)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWh1 : ∀ o c kh kw, |Wh1 o c kh kw| ≤ wh1)
    (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ wh) (hWd : ∀ i j, |Wd i j| ≤ wd)
    (hbnBs : FloatBridgesTo bnBs bnBsF) (hbnBh1 : FloatBridgesTo bnBh1 bnBh1F)
    (hbnBh : FloatBridgesTo bnBh bnBhF)
    (hfB : FloatBridgesTo fB fBF)
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
    (hb17B : FloatBridgesTo b17B b17BF)
    (hb18B : FloatBridgesTo b18B b18BF)
    (hb19B : FloatBridgesTo b19B b19BF)
    (hb20B : FloatBridgesTo b20B b20BF)
    (hb21B : FloatBridgesTo b21B b21BF)
    :
    FloatBridgesTo
      (mnv4InputGradB N Ws Wh1 Wh Wd bnBs bnBh1 bnBh fB b1B b2B b3B b4B b5B b6B b7B b8B b9B b10B b11B b12B b13B b14B b15B b16B b17B b18B b19B b20B b21B m_stem m_h1 m_h)
      (mnv4InputGradBF N M Ws Wh1 Wh Wd bnBsF bnBh1F bnBhF fBF b1BF b2BF b3BF b4BF b5BF b6BF b7BF b8BF b9BF b10BF b11BF b12BF b13BF b14BF b15BF b16BF b17BF b18BF b19BF b20BF b21BF m_stem m_h1 m_h) := by
  unfold mnv4InputGradB mnv4InputGradBF
  have hstem := ((floatBridgesTo_reluMaskBack m_stem).comp hbnBs).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_flatConvStride2XlaBack (h := 112) (w := 112) M Ws hws (by norm_num) hWs))
  have hhead1 := ((floatBridgesTo_reluMaskBack m_h1).comp hbnBh1).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_convBack (h := 7) (w := 7) M Wh1 hwh1 (by norm_num) hWh1))
  have hhead2 := ((floatBridgesTo_reluMaskBack m_h).comp hbnBh).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_convBack (h := 7) (w := 7) M Wh hwh (by norm_num) hWh))
  have h0 := (FloatBridgesTo.batchMap N (floatBridgesTo_linBack M Wd hwd hnc hWd)).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_gapBack M 1280 7 7 (by norm_num) (by norm_num) (by norm_num)))
  have hH2 := h0.comp hhead2
  have hH1 := hH2.comp hhead1
  have hB21 := hH1.comp hb21B
  have hB20 := hB21.comp hb20B
  have hB19 := hB20.comp hb19B
  have hB18 := hB19.comp hb18B
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
  have hF := hB1.comp hfB
  exact hF.comp hstem

end Proofs