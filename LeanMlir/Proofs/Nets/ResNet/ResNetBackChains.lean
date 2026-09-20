import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The ResNet-34 / ResNet-50 backward chains — the ℝ maps the ResNet ties are about

The hand-composed reverse of the committed ResNet forwards, as plain `def`s on the cotangent:
the batched chains `r34InputGradB` / `r50InputGradB` (the reverses of `resnet34ForwardB_full` /
`resnet50ForwardB_full`, at a variable batch `N`, R50 also at a variable resolution `q`). Each
chain keeps its block backwards and its BatchNorm backwards as *supplied* maps and spells only
the endpoints — the stem's strided conv-back, the 3×3/s2 pool-back, the GAP-back and the
dense-back — so that the certified tie (`ResNet34BackCertifiedTieB`,
`ResNet50WholeBackCertifiedTieB`, on the leaf ties of `ResNet34BackCertifiedTie`) is a statement
about a NAMED chain of the forward's shape. (The per-example r34 chain and its two block
backwards were retired with their renderer on 2026-09-19.)

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
-- § The batched 3×3/s2 pool backward
-- ════════════════════════════════════════════════════════════════

/-- **The batched 3×3/s2 max-pool backward** — `maxPool3s2FlatBack` per example, on that example's
    OWN saved stem activation. ⛔ It is `batchMapAux` and not `batchMap`: a `batchMap` would hand
    example 0's argmax pattern to every example (`StableHLO.batchMapAux`'s own header records the
    same trap on the emitter side). This IS `den (.maxPool3s2BackB …)`:
    `den_maxPool3s2BackB_eq_flatBackB` below equates the two spellings of the scatter
    (`ResNet34StepTieB.mpInB` is the `maxPool3s2BackFlat` one). -/
noncomputable def maxPool3s2FlatBackB (N c h w : Nat) (v : Vec (N * (c * (2*h) * (2*w)))) :
    Vec (N * (c * h * w)) → Vec (N * (c * (2*h) * (2*w))) :=
  StableHLO.batchMapAux N
    (fun xv : Vec (c * (2*h) * (2*w)) =>
      maxPool3s2FlatBack (c := c) (h := h) (w := w) (Tensor3.unflatten xv)) v

/-- **The two spellings of the 3×3/s2 scatter are one map, at every input.** The render's
    `den (.maxPool3s2Back …)` is `StableHLO.maxPool3s2BackFlat`, a triple sum against a `0/1`
    indicator; the chain's `maxPool3s2FlatBack` is the same sum over the flat index with the
    indicator folded into the `if`. `sum_flat3` re-indexes one into the other. No smoothness
    hypothesis: this is about the scatter itself, not about the VJP it equals at a smooth point
    (`maxPool3s2FlatBack_eq_vjp_backward`). -/
theorem maxPool3s2BackFlat_eq_flatBack (c h w : Nat) (xv : Vec (c * (2*h) * (2*w)))
    (dyv : Vec (c * h * w)) :
    StableHLO.maxPool3s2BackFlat c h w xv dyv
      = maxPool3s2FlatBack (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w)) dyv := by
  funext idx
  have hidx : finProdFinEquiv
      (finProdFinEquiv ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1,
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2),
        (finProdFinEquiv.symm idx).2) = idx := by
    rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
  simp only [StableHLO.maxPool3s2BackFlat, maxPool3s2FlatBack]
  rw [sum_flat3 (fun k => if maxPool3s2LocalReindex
        (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w)) k = idx then dyv k else 0)]
  refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl fun ho _ =>
    Finset.sum_congr rfl fun wo _ => ?_
  rw [hidx]
  simp only [Tensor3.unflatten]
  split <;> simp

/-- **The batched stem-pool node the ResNet renders emit denotes the chain's batched scatter**
    — the `.maxPool3s2BackB` bridge at the map `r34InputGradB` / `r50InputGradB` use, with no
    hypothesis. Until this lemma the only bridge was `maxPool3s2Back_faithful`, at the per-example
    constructor no shipped render emits. -/
theorem den_maxPool3s2BackB_eq_flatBackB {N c h w : Nat} (xN : String)
    (x : Vec (N * (c * (2*h) * (2*w)))) (e : StableHLO.SHlo (N * (c * h * w))) :
    StableHLO.den (.maxPool3s2BackB xN x e) = maxPool3s2FlatBackB N c h w x (StableHLO.den e) := by
  have hf : StableHLO.maxPool3s2BackFlat c h w
      = fun xv : Vec (c * (2*h) * (2*w)) =>
          maxPool3s2FlatBack (c := c) (h := h) (w := w) (Tensor3.unflatten xv) :=
    funext fun xv => funext fun dyv => maxPool3s2BackFlat_eq_flatBack c h w xv dyv
  rw [StableHLO.den_maxPool3s2BackB, maxPool3s2FlatBackB, hf]

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
