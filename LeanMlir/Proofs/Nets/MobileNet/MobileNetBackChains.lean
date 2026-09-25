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

⚠ Padding: MobileNetV2 is XLA-`SAME` at its stem (`flatConvStride2XlaBack`, the odd-phase
scatter `decimateOddBack`) and its strided depthwises (`depthwiseStride2FlatXlaBack`), the
TF-origin convention. MobileNetV4 follows timm's `mobilenetv4_conv_medium` and is SYMMETRIC at
every strided site — its stem is ResNet's `flatConvStride2Back`.

⭐ Neither net has a stem pool, so every batched endpoint is `StableHLO.batchMap N` of a
per-example leaf; none of ResNet's row-indexed `batchMapAux` lift is needed here.

⚠⚠ `mnv4InputGradB` is stated TO THE IMAGE, one step past the artifact: MobileNetV4's committed
backward stops at the stem conv's WEIGHT gradient (no render emits a gradient into `%x`), so this
chain runs one `flatConvStride2Back` past that point, as EfficientNet-B0's does at its stem.

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
    of `mobilenetv4ForwardB_full` (timm's `mobilenetv4_conv_medium`): dense-back, relabelled to
    `[N, 1280, 1, 1]` → `conv_head`'s relu mask, BatchNorm back and 1×1 conv back at `1×1` →
    relabelled to `[N, 960]`, GAP-back → the first head conv's relu mask, BatchNorm back and 1×1
    conv back at 7×7 → the twenty-one UIB block backwards → the fused stage's → the stem's relu
    mask, BatchNorm back and symmetric 3×3/s2 conv back.

    The fused stage's and the twenty-one blocks' backwards are supplied, as are the three
    BatchNorm backs; the conv, GAP, relabel and dense leaves are concrete and lifted over the `N`
    examples. The two relabellings are the head's `[N, c] ↔ [N, c, 1, 1]` (`castIdx` in the
    graph, no text in the render). -/
noncomputable def mnv4InputGradB (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (Wh1 : Kernel4 960 256 1 1) (Wh : Kernel4 1280 960 1 1)
    (Wd : Mat 1280 nCls)
    (bnBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh1 : Vec (N * (960 * 7 * 7)) → Vec (N * (960 * 7 * 7)))
    (bnBh : Vec (N * (1280 * 1 * 1)) → Vec (N * (1280 * 1 * 1)))
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
    (m_h : Fin (N * (1280 * 1 * 1)) → Prop) [DecidablePred m_h] :
    Vec (N * nCls) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N (flatConvStride2Back (h := 112) (w := 112) Ws)
      ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ fB
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B ∘ b17B ∘ b18B ∘ b19B ∘ b20B ∘ b21B
  ∘ (StableHLO.batchMap N (convFlatBack (h := 7) (w := 7) Wh1)
      ∘ bnBh1 ∘ reluMaskBack m_h1)
  ∘ ((StableHLO.batchMap N (gapBack 960 7 7)
        ∘ fun v i => v (Fin.cast (by rw [Nat.mul_one, Nat.mul_one]) i))
      ∘ (StableHLO.batchMap N (convFlatBack (h := 1) (w := 1) Wh)
        ∘ bnBh ∘ reluMaskBack m_h))
  ∘ ((fun v i => v (Fin.cast (by rw [Nat.mul_one, Nat.mul_one]) i))
      ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec 1280)))

end Proofs
