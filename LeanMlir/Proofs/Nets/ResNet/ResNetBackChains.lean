import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The ResNet-34 / ResNet-50 backward chains — the ℝ maps the ResNet ties are about

The hand-composed reverse of the committed ResNet forwards, as plain `def`s on the cotangent:
the two basic-block backwards (`r34IdBlockBack`, `r34DownBlockBack`), the per-example whole-net
chain `r34InputGrad` (the reverse of `resnet34Forward_full_pc`), and the batched chains
`r34InputGradB` / `r50InputGradB` (the reverses of `resnet34ForwardB_full` /
`resnet50ForwardB_full`, at a variable batch `N`, R50 also at a variable resolution `q`). Each
chain keeps its sixteen block backwards and its BatchNorm backwards as *supplied* maps and spells
only the endpoints — the stem's strided conv-back, the 3×3/s2 pool-back, the GAP-back and the
dense-back — so that the certified tie (`ResNet34BackCertifiedTie`, `ResNet34BackCertifiedTieB`,
`ResNet50WholeBackCertifiedTieB`) is a statement about a NAMED chain of the forward's shape.

⛔ `maxPool3s2FlatBackB`, the batched pool backward, is `StableHLO.batchMapAux` and not
`batchMap`: the pool's backward is indexed by the saved forward activation and every example
has its own, so a `batchMap` would hand example 0's argmax pattern to all of them. It is the one
endpoint the batch axis changed, and the reason this leaf imports `StableHLO.lean`.

⚠ Padding is SYMMETRIC at every stride-2 site of both nets (`flatConvStride2Back`), the
PyTorch-origin convention — not the XLA-`SAME` `flatConvStride2XlaBack` the TF-origin stems take.
⚠ At a variable `q` every dimension is written as an explicit `2 * (…)` nest, never `8 * q`: those
are equal Nats and NOT definitionally equal terms, and each stage demands its operand at exactly
the spelling it names.

Moved here from the float bridges that defined them beside their float twins on 2026-09-08
(`planning/archive/float_second_pass.md`); no number is stated about any of these chains. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The two basic-block backwards (per-channel BN, per-example)
-- ════════════════════════════════════════════════════════════════

/-- The r34 identity basic-block input-gradient VJP at a smooth point — the **reverse of `rblkPC`**.
    `relu(F(x)+x)` backward = the ReLU mask, then the residual split (cotangent to both the body and
    the skip, added): `residual bF ∘ reluMaskBack`, with `bF` the reverse of `F = bn₂∘conv₂ ∘
    relu∘bn₁∘conv₁`. The ReLU kinks read the fixed sign masks `m_out`/`m_mid`; the BN-backs `bnB₁`/
    `bnB₂` are the per-channel BatchNorm backwards, supplied. The residual-skip backward is the
    forward's own `Proofs.residual`: the skip routes the cotangent to both branches and adds. -/
noncomputable def r34IdBlockBack {c h w : Nat}
    (W₁ W₂ : Kernel4 c c 3 3)
    (bnB1 bnB2 : Vec (c * h * w) → Vec (c * h * w))
    (m_out m_mid : Fin (c * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid] :
    Vec (c * h * w) → Vec (c * h * w) :=
  Proofs.residual
      (convFlatBack (h := h) (w := w) W₁ ∘ bnB1 ∘ reluMaskBack m_mid
        ∘ convFlatBack (h := h) (w := w) W₂ ∘ bnB2)
    ∘ reluMaskBack m_out

/-- The r34 downsample basic-block input-gradient VJP at a smooth point — the **reverse of
    `rblkPStridedPC`**. `relu(proj(x) + body(x))` backward = the ReLU mask, then the two-branch
    fan-in `bProj(dy') + bBody(dy')` (both branches non-trivial, summed). The strided convs reverse
    via `flatConvStride2Back`; the BN-backs `bnB₁`/`bnB₂`/`bnBp` are the per-channel BatchNorm
    backwards (supplied). -/
noncomputable def r34DownBlockBack {ic oc h w kHp kWp : Nat}
    (W₁ : Kernel4 oc ic 3 3) (W₂ : Kernel4 oc oc 3 3) (Wp : Kernel4 oc ic kHp kWp)
    (bnB1 bnB2 bnBp : Vec (oc * h * w) → Vec (oc * h * w))
    (m_out m_mid : Fin (oc * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid] :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  (fun dy j =>
      (flatConvStride2Back (h := h) (w := w) Wp ∘ bnBp) dy j
      + (flatConvStride2Back (h := h) (w := w) W₁ ∘ bnB1 ∘ reluMaskBack m_mid
          ∘ convFlatBack (h := h) (w := w) W₂ ∘ bnB2) dy j)
    ∘ reluMaskBack m_out

-- ════════════════════════════════════════════════════════════════
-- § The per-example whole-net chain (the [3,4,6,3] fold at 224 px, 10 classes)
-- ════════════════════════════════════════════════════════════════

/-- The whole ResNet-34 input-gradient VJP at a smooth point — the **exact reverse of
    `resnet34Forward_full_pc`**: `dense ∘ GAP ∘ [3,4,6,3] blocks ∘ maxpool ∘ stem` reversed. The
    stem/GAP/maxpool/dense endpoints are concrete (`flatConvStride2Back`/`gapBack`/`maxPool3s2FlatBack`/
    `dense (transposeᵀ) 0`); the 16 block backwards `a0B..e1B` are supplied (each an
    `r34IdBlockBack` or `r34DownBlockBack` at the tie). The `[3,4,6,3]` stage structure is in the
    block maps' dims (down-blocks change channels×spatial; identity blocks preserve). The pool is
    the 3×3/s2 stem pool's backward — the 2×2 `maxPoolFlatBack` this chain once used is a different
    function of the same type, which is what the tie found. -/
noncomputable def r34InputGrad (Ws : Kernel4 64 3 7 7) (Wd : Mat 512 10)
    (bnBs : Vec (64 * 112 * 112) → Vec (64 * 112 * 112))
    (e1B e0B : Vec (512 * 7 * 7) → Vec (512 * 7 * 7))
    (d4B : Vec (512 * 7 * 7) → Vec (256 * 14 * 14))
    (c4B c3B c2B c1B c0B : Vec (256 * 14 * 14) → Vec (256 * 14 * 14))
    (d3B : Vec (256 * 14 * 14) → Vec (128 * 28 * 28))
    (b2B b1B b0B : Vec (128 * 28 * 28) → Vec (128 * 28 * 28))
    (d2B : Vec (128 * 28 * 28) → Vec (64 * 56 * 56))
    (a2B a1B a0B : Vec (64 * 56 * 56) → Vec (64 * 56 * 56))
    (xmp : Tensor3 64 112 112)
    (m_stem : Fin (64 * 112 * 112) → Prop) [DecidablePred m_stem] :
    Vec 10 → Vec (3 * 224 * 224) :=
  (flatConvStride2Back (h := 112) (w := 112) Ws ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ maxPool3s2FlatBack xmp
  ∘ a0B ∘ a1B ∘ a2B
  ∘ d2B
  ∘ b0B ∘ b1B ∘ b2B
  ∘ d3B
  ∘ c0B ∘ c1B ∘ c2B ∘ c3B ∘ c4B
  ∘ d4B
  ∘ e0B ∘ e1B
  ∘ gapBack 512 7 7
  ∘ dense (Mat.transpose Wd) (0 : Vec 512)

-- ════════════════════════════════════════════════════════════════
-- § The batched 3×3/s2 pool backward
-- ════════════════════════════════════════════════════════════════

/-- **The batched 3×3/s2 max-pool backward** — `maxPool3s2FlatBack` per example, on that example's
    OWN saved stem activation. ⛔ It is `batchMapAux` and not `batchMap`: a `batchMap` would hand
    example 0's argmax pattern to every example (`StableHLO.batchMapAux`'s own header records the
    same trap on the emitter side). This is `den (.maxPool3s2BackB …)` up to the two spellings of
    the scatter (`ResNet34StepTieB.mpInB` is the `maxPool3s2BackFlat` one). -/
noncomputable def maxPool3s2FlatBackB (N c h w : Nat) (v : Vec (N * (c * (2*h) * (2*w)))) :
    Vec (N * (c * h * w)) → Vec (N * (c * (2*h) * (2*w))) :=
  StableHLO.batchMapAux N
    (fun xv : Vec (c * (2*h) * (2*w)) =>
      maxPool3s2FlatBack (c := c) (h := h) (w := w) (Tensor3.unflatten xv)) v

-- ════════════════════════════════════════════════════════════════
-- § The batched whole-net chains (true batch-norm, variable N; R50 at variable q)
-- ════════════════════════════════════════════════════════════════

/-- **The batched whole-net input-gradient backward of ResNet-34** — the exact reverse of
    `resnet34ForwardB_full = head ∘ [3,4,6,3] ∘ stem`: dense-back → GAP-back → the sixteen basic
    blocks' backwards → the 3×3/s2 pool back → the stem's relu mask, BatchNorm back and 7×7/s2
    conv back. The block backwards and the stem's BatchNorm back are supplied; the conv, pool, GAP
    and dense leaves are concrete and lifted over the `N` examples. ⭐ `N` is a variable: this chain
    carries no numerals. -/
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
    `resnet50ForwardB_full = head ∘ [3,4,6,3] bottlenecks ∘ stem`: dense-back → GAP-back → the
    sixteen bottleneck backwards → the 3×3/s2 pool back → the stem's relu mask, BatchNorm back
    and 7×7/s2 conv back. The bottleneck backwards and the stem's BatchNorm back are supplied;
    the conv, pool, GAP and dense leaves are concrete and lifted over the `N` examples. ⭐ `q` is a
    binder, so one chain covers `resnet50in_fwd` (`q = 7`, 224 px) and `resnet50in160_fwd`
    (`q = 5`, 160 px — the net the quoted 76.66% trains). Stem and head ARE ResNet-34's
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
