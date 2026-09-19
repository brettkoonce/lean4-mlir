import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The MobileNetV2 / MobileNetV4 backward chains — the ℝ maps the MobileNet ties are about

The hand-composed reverse of the committed MobileNet forwards, as plain `def`s on the cotangent:
the batched chains `mnv2InputGradB` (seventeen bottlenecks, the reverse of `mobilenetv2ForwardB_full`) and
`mnv4InputGradB` (MobileNetV4-Conv-M: the fused stage, twenty-one UIB blocks and two head convs,
the reverse of `mobilenetv4ForwardB_full`), both at a variable batch `N` and class count. Each
chain keeps its block backwards and its BatchNorm backwards as *supplied* maps and spells only the
endpoints, so that the certified tie (`MobileNetV2WholeBackCertifiedTieB`,
`MobileNetV4WholeBackCertifiedTieB`) is a statement about a NAMED chain of the forward's shape.

⚠ Padding is XLA-`SAME` at every stem (`flatConvStride2XlaBack`, the odd-phase scatter
`decimateOddBack`) and at MobileNetV2's strided depthwises (`depthwiseStride2FlatXlaBack`), the
TF-origin convention — not the symmetric `flatConvStride2Back` ResNet takes. MobileNetV4 has two
padding phases in one chain: its stem is XLA-`SAME`, while the fused stage and its three strided
depthwises are symmetric and sit inside the supplied block backwards. Do not "tidy" one to match
the other.

⭐ Neither net has a stem pool, so every batched endpoint is `StableHLO.batchMap N` of a
per-example leaf; none of ResNet's row-indexed `batchMapAux` lift is needed here.

⚠⚠ `mnv4InputGradB` is stated TO THE IMAGE, one step past the artifact: MobileNetV4's committed
backward stops at the stem conv's WEIGHT gradient (no render emits a gradient into `%x`), so this
chain runs one `flatConvStride2XlaBack` past that point, exactly as EfficientNet-B0's does at the
identical stem. ⚠ Conv-M has no quoted accuracy; what pins its artifact to the reference is the
pair of ties re-run 2026-09-07.

No number is stated about either chain. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The batched whole-net chains (true batch-norm, variable N and class count)
-- ════════════════════════════════════════════════════════════════

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

/-- **The batched whole-net input-gradient backward of MobileNetV4-Conv-M** — the exact reverse
    of `mobilenetv4ForwardB_full`: dense-back → GAP-back → the second head conv's relu mask,
    BatchNorm back and 1×1 conv back → the first head conv's three → the twenty-one UIB block
    backwards → the fused stage's → the stem's relu mask, BatchNorm back and XLA-`SAME` 3×3/s2
    conv back.

    The fused stage's and the twenty-one blocks' backwards are supplied, as are the three
    BatchNorm backs; the conv, GAP and dense leaves are concrete and lifted over the `N`
    examples. ⚠ **Two head convs**: Conv-M's head is `%h1W` (256 → 960) then `%hW` (960 → 1280)
    before the pool, so this chain has two conv-BN-relu endpoints at 7×7 where MobileNetV2's has
    one and ResNet-34's has none. ⚠ The fused stage is a supplied slot, not a concrete endpoint:
    it is stage 0 — a conv-bn-swish and a 1×1 project — and opaque for the same reason the blocks
    are; the tie composes certified backwards, it does not re-derive them. -/
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

end Proofs
