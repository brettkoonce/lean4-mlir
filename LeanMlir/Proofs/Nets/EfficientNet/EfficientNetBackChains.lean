import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The EfficientNet-B0 backward chain — the ℝ map the B0 whole-net tie is about

The hand-composed reverse of the committed EfficientNet-B0 forward, as a plain `def` on the
cotangent: the batched sixteen-block paper net `efficientnetInputGradBFull` (the reverse of
`efficientnetForwardBFull`, at a variable batch `N` and class count). The chain keeps its block
backwards, its BatchNorm backwards and its swish backwards as *supplied* maps and spells only the
endpoints, so that the certified tie (`EfficientNetFullWholeBackCertifiedTie`) is a statement
about a NAMED chain of the forward's shape.

⚠ The stem is XLA-`SAME` (`flatConvStride2XlaBack`, the odd-phase scatter), the TF-origin
convention; the strided depthwises inside the blocks are symmetric and sit in the supplied block
backwards. B0 has no stem pool, so every batched endpoint is `StableHLO.batchMap N` of a
per-example leaf.

No number is stated about this chain. -/

namespace Proofs

/-- **The batched whole-net input-gradient backward of the sixteen-block EfficientNet-B0** —
    the reverse of `efficientnetForwardBFull = head ∘ b16 ∘ … ∘ b1 ∘ stem`: classifier-back →
    GAP-back → head-conv-bn-swish-back → the sixteen MBConv block backs → stem-conv-bn-swish-back.
    The block backs and the stem/head BN+swish backs are supplied; the conv/GAP/dense leaves are
    concrete, `batchMap`-lifted over the `N` examples, the stem at the XLA-`SAME` phase. -/
noncomputable def efficientnetInputGradBFull {nCls : Nat} (N : Nat)
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

end Proofs
