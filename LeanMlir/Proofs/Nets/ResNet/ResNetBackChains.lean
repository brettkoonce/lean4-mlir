import LeanMlir.Proofs.Foundation.BackwardMaps

/-! # The ResNet-34 / ResNet-50 backward chains — the ℝ maps the ResNet ties are about

The hand-composed reverse of the committed ResNet forwards, as plain `def`s on the cotangent:
the batched chains `r34InputGradB` / `r50InputGradB` (the reverses of `resnet34ForwardBFull` /
`resnet50ForwardBFull`, at a variable batch `N`, R50 also at a variable resolution `q`). Each
chain keeps its block backwards and its BatchNorm backwards as *supplied* maps and spells only
the endpoints — the stem's strided conv-back, the 3×3/s2 pool-back, the GAP-back and the
dense-back — so that the certified tie (`ResNet34BackCertifiedTieB`,
`ResNet50WholeBackCertifiedTieB`, on the leaf ties of `ConvBackCertifiedTie`) is a statement
about a NAMED chain of the forward's shape.

Note: `maxPool3s2FlatBackB`, the batched pool backward, is `StableHLO.batchMapAux` and not
`batchMap`: the pool's backward is indexed by the saved forward activation and every example
has its own, so a `batchMap` would hand example 0's argmax pattern to all of them. It is the one
endpoint the batch axis changed, and the reason this leaf imports `StableHLO.lean`.

Padding is SYMMETRIC at every stride-2 site of both nets (`flatConvStride2Back`), the
PyTorch-origin convention — not the XLA-`SAME` `flatConvStride2XlaBack` the TF-origin stems take.
At a variable `q` every dimension is written as an explicit `2 * (…)` nest, never `8 * q`: those
are equal Nats and NOT definitionally equal terms, and each stage demands its operand at exactly
the spelling it names.

No number is stated about any of these chains. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The batched whole-net chains (true batch-norm, variable N; R50 at variable q)
-- ════════════════════════════════════════════════════════════════

/-- **The batched whole-net input-gradient backward of ResNet-34** — the exact reverse of
    `resnet34ForwardBFull = head ∘ [3,4,6,3] ∘ stem`: dense-back → GAP-back → the sixteen basic
    blocks' backwards → the 3×3/s2 pool back → the stem's relu mask, BatchNorm back and 7×7/s2
    conv back. The block backwards and the stem's BatchNorm back are supplied; the conv, pool, GAP
    and dense leaves are concrete and lifted over the `N` examples. `N` is a variable: the batch
    size is never pinned (the 224-px resolution and the 64…512 widths are literals). -/
noncomputable def r34InputGradB (N : Nat) {nCls : Nat}
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 512 nCls)
    (bnBs : Vec (N * (64 * (2 * 56) * (2 * 56))) → Vec (N * (64 * (2 * 56) * (2 * 56))))
    (xpool : Vec (N * (64 * (2 * 56) * (2 * 56))))
    (e1B e0B : Vec (N * (512 * 7 * 7)) → Vec (N * (512 * 7 * 7)))
    (d4B : Vec (N * (512 * 7 * 7)) → Vec (N * (256 * 14 * 14)))
    (c4B c3B c2B c1B c0B : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (d3B : Vec (N * (256 * 14 * 14)) → Vec (N * (128 * 28 * 28)))
    (b2B b1B b0B : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (d2B : Vec (N * (128 * 28 * 28)) → Vec (N * (64 * 56 * 56)))
    (a2B a1B a0B : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (m_stem : Fin (N * (64 * (2 * 56) * (2 * 56))) → Prop) [DecidablePred m_stem] :
    Vec (N * nCls) → Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) :=
  (StableHLO.batchMap N (flatConvStride2Back (h := 2 * 56) (w := 2 * 56) Ws)
      ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ maxPool3s2FlatBackB N 64 56 56 xpool
  ∘ a0B ∘ a1B ∘ a2B
  ∘ d2B
  ∘ b0B ∘ b1B ∘ b2B
  ∘ d3B
  ∘ c0B ∘ c1B ∘ c2B ∘ c3B ∘ c4B
  ∘ d4B
  ∘ e0B ∘ e1B
  ∘ StableHLO.batchMap N (gapBack 512 7 7)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec 512))

/-- **The batched whole-net input-gradient backward of ResNet-50** — the exact reverse of
    `resnet50ForwardBFull = head ∘ [3,4,6,3] bottlenecks ∘ stem`: dense-back → GAP-back → the
    sixteen bottleneck backwards → the 3×3/s2 pool back → the stem's relu mask, BatchNorm back
    and 7×7/s2 conv back. The bottleneck backwards and the stem's BatchNorm back are supplied;
    the conv, pool, GAP and dense leaves are concrete and lifted over the `N` examples. `q` is a
    binder, so one chain covers `resnet50in_fwd` (`q = 7`, 224 px) and `resnet50in160_fwd`
    (`q = 5`, 160 px). Stem and head ARE ResNet-34's
    (`r34StemB` / `r34HeadB`) at R50's widths. -/
noncomputable def r50InputGradB (N q : Nat) {nCls : Nat}
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 2048 nCls)
    (bnBs : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (xpool : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (b1B : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b2B : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b3B : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b4B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b5B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b6B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b7B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b8B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b9B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b10B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b11B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b12B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b13B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b14B : Vec (N * (2048 * q * q)) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b15B : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (b16B : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (m_stem : Fin (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Prop) [DecidablePred m_stem] :
    Vec (N * nCls) → Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) :=
  (StableHLO.batchMap N (flatConvStride2Back (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q))))) Ws)
      ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ maxPool3s2FlatBackB N 64 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xpool
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B
  ∘ StableHLO.batchMap N (gapBack 2048 q q)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec 2048))

end Proofs
