import LeanMlir.Proofs.Float.Resnet34WholeBackFloatBridge
import LeanMlir.Proofs.Float.BnBatchFloatBridge
import LeanMlir.Proofs.Float.EfficientNetBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE ResNet-34 backward at TRUE BATCH-NORM

`Resnet34WholeBackFloatBridge.lean` names `r34InputGrad`, the input-gradient chain of the
PER-EXAMPLE ResNet-34 — the reverse of `resnet34Forward_full_pc`, the net the retired
`ResNet34Render` emitted. This file is the same for the net `resnet34ForwardB_full` IS: the
[3,4,6,3] ladder at **`bnBatchLA`**, the training-mode BatchNorm every committed ResNet-34 train
step actually emits (`planning/proofs_tier_to_paper_nets.md` §4.0), at a variable batch `N`.

It exists for one reason: so that the batch-BN certified tie
(`Foundation/Resnet34BackCertifiedTieB.lean`, tier T6 of §4.2) is a statement about a **named
chain** of the same shape as the per-example one, with the stem, pool, GAP and dense endpoints
concrete and `batchMap`-lifted and the sixteen block backwards supplied.
`EfficientNetFullWholeBackFloatBridge.lean` plays exactly this role for B0.

## What the batch axis changed, and it is one op

⭐ Every endpoint but one is `batchMap N` of the per-example leaf the `_pc` chain already used —
a convolution, a GAP and a dense are batch-separable, and their float peers lift by
`FloatBridgesTo.batchMap`. ⛔ **The 3×3/s2 stem pool is not**, and not because the op couples
examples: it does not. Its BACKWARD is indexed by the saved forward activation, and each example
has its own. So the batched pool backward is `batchMapAux` — the lift that hands example `n` its
OWN slice of the saved activation — and `FloatClose.batchMapAux` / `FloatBridgesTo.batchMapAux`
below are the two lemmas that costs. They are `FloatClose.batchMap`'s proof with the function
allowed to depend on the row, and they belong beside it in `EfficientNetBackFloatBridge.lean`;
they are stated here because that file has 300-odd downstream modules and this one has none
(`planning/proofs_tier_to_paper_nets.md` §5, the root-file-lemma rule).

⛔ **No number is stated about this chain**, and that is a decision rather than a limit. §4.2's
T5 is a float BUDGET, and `planning/float_budget_numbers.md` closed that thread: the per-example
r34 backward is `8.857·10²⁴⁵` and the batched one is larger (`bnGradInputReMag`'s gain carries
`Xh² = N·h·w`, so the numeral moves with the batch). The chain is here for the tie.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The row-indexed batched lift — `batchMapAux`
--   ⚠ These belong next to `FloatClose.batchMap` / `FloatBridgesTo.batchMap`
--   (`Float/EfficientNetBackFloatBridge.lean`); they are here for the rebuild cost.
-- ════════════════════════════════════════════════════════════════

/-- `batchMapAux` at one output index, by definition: example `p.1`'s own slice of the saved
    auxiliary and of the input. The `batchMap_apply` peer. -/
theorem batchMapAux_apply {s a b : Nat} (N : Nat) (f : Vec s → Vec a → Vec b)
    (aux : Vec (N * s)) (v : Vec (N * a)) (idx : Fin (N * b)) :
    StableHLO.batchMapAux N f aux v idx
      = f (fun i : Fin s => aux (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i)))
          (fun i : Fin a => v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i)))
          (finProdFinEquiv.symm idx).2 := rfl

/-- **The row-indexed batched seam — `FloatClose.batchMapAux`.** If every example's own map
    `f (aux_n)` is `FloatClose A B _ _ L` with the SAME magnitude and modulus, so is the batch.
    `FloatClose.batchMap` with the function allowed to depend on the row: the batch is still
    `N` independent copies, they are just not copies of one function. -/
theorem FloatClose.batchMapAux {s a b : Nat} (N : Nat) {A B : ℝ}
    {f fF : Vec s → Vec a → Vec b} {L : ℝ → ℝ} (aux : Vec (N * s))
    (hf : ∀ r : Fin N, FloatClose A B
      (f (StableHLO.batchSlice N s aux r)) (fF (StableHLO.batchSlice N s aux r)) L) :
    FloatClose A B (StableHLO.batchMapAux N f aux) (StableHLO.batchMapAux N fF aux) L := by
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hslice : ∀ i, |v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ A :=
      fun i => hv (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    rw [batchMapAux_apply, batchMapAux_apply]
    exact (hf (finProdFinEquiv.symm idx).1).1
      (fun i => v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))) hslice
      (finProdFinEquiv.symm idx).2
  · have hva' : ∀ i, |va (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ A :=
      fun i => hva (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    have hvt' : ∀ i, |vt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ A :=
      fun i => hvt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    have hd' : ∀ i, |vt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
                  - va (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ e :=
      fun i => hd (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    rw [batchMapAux_apply, batchMapAux_apply]
    exact (hf (finProdFinEquiv.symm idx).1).2
      (fun i => vt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i)))
      (fun i => va (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))) e hva' hvt' hd'
      (finProdFinEquiv.symm idx).2

/-- **`FloatBridgesTo.batchMapAux`** — the row-indexed batched lift with the float map named.
    ⚠ The per-row bridges must share `mag` and `mod`; the caller supplies one budget for the
    family, which is what makes the batch's envelope batch-size-free. ⛔ `mag`'s nonnegativity
    is a separate hypothesis rather than read off a row: at `N = 0` there is no row to read it
    from, and the statement is still true (and vacuous) there. -/
noncomputable def FloatBridgesTo.batchMapAux {s a b : Nat} (N : Nat)
    {f fF : Vec s → Vec a → Vec b} (aux : Vec (N * s))
    (mag : ℝ → ℝ) (mod : ℝ → ℝ → ℝ) (hmag : ∀ A, 0 ≤ A → 0 ≤ mag A)
    (hf : ∀ r : Fin N, ∀ A, 0 ≤ A → FloatClose A (mag A)
      (f (StableHLO.batchSlice N s aux r)) (fF (StableHLO.batchSlice N s aux r)) (mod A)) :
    FloatBridgesTo (StableHLO.batchMapAux N f aux) (StableHLO.batchMapAux N fF aux) :=
  ⟨mag, mod, fun A hA =>
    ⟨hmag A hA, FloatClose.batchMapAux N aux (fun r => hf r A hA)⟩⟩

-- ════════════════════════════════════════════════════════════════
-- § The batched 3×3/s2 pool backward
-- ════════════════════════════════════════════════════════════════

/-- **The batched 3×3/s2 max-pool backward** — `maxPool3s2FlatBack` per example, on that example's
    OWN saved stem activation. ⛔ It is `batchMapAux` and not `batchMap`: a `batchMap` would hand
    example 0's argmax pattern to every example (`StableHLO.batchMapAux`'s own header records the
    same trap on the emitter side). This is `den (.maxPool3s2BackB …)` up to the two spellings of
    the scatter (`ResNet34TiePoCB.mpInB` is the `maxPool3s2BackFlat` one). -/
noncomputable def maxPool3s2FlatBackB (N c h w : Nat) (v : Vec (N * (c * (2*h) * (2*w)))) :
    Vec (N * (c * h * w)) → Vec (N * (c * (2*h) * (2*w))) :=
  StableHLO.batchMapAux N
    (fun xv : Vec (c * (2*h) * (2*w)) =>
      maxPool3s2FlatBack (c := c) (h := h) (w := w) (Tensor3.unflatten xv)) v

/-- The float batched pool backward — the model's rounded masked reduction, per example. -/
noncomputable def FloatModel.maxPool3s2FlatBackBF (M : FloatModel) (N c h w : Nat)
    (v : Vec (N * (c * (2*h) * (2*w)))) :
    Vec (N * (c * h * w)) → Vec (N * (c * (2*h) * (2*w))) :=
  StableHLO.batchMapAux N
    (fun xv : Vec (c * (2*h) * (2*w)) =>
      M.maxPool3s2FlatBackF (c := c) (h := h) (w := w) (Tensor3.unflatten xv)) v

/-- **The batched 3×3/s2 pool backward float-bridges TO its float peer.** The per-example
    bridge's `mag`/`mod` do not depend on the saved activation — the `4` is
    `maxPool3s2Back_mask_sum_abs_le`'s fibre bound and the rounding is at width `c·h·w`, both
    facts about the WINDOW GEOMETRY and not about which cell won — so one budget serves every
    example and `FloatBridgesTo.batchMapAux` applies with no per-row case split. -/
noncomputable def floatBridgesTo_maxPool3s2BackB (M : FloatModel) (N c h w : Nat)
    (v : Vec (N * (c * (2*h) * (2*w)))) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    FloatBridgesTo (maxPool3s2FlatBackB N c h w v) (M.maxPool3s2FlatBackBF N c h w v) :=
  FloatBridgesTo.batchMapAux N v
    (fun A => 4 * A + ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A))
    (fun A e => ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) + 4 * e)
    (fun A hA => by
      have hu := M.u_nonneg
      have h1 : (1 : ℝ) ≤ (1 + M.u) ^ (c*h*w + 1) := one_le_pow₀ (by linarith)
      have h2 := mul_nonneg (by linarith : (0:ℝ) ≤ (1 + M.u) ^ (c*h*w + 1) - 1)
        (by linarith : (0:ℝ) ≤ 4 * A)
      linarith)
    (fun r A hA => ((floatBridgesTo_maxPool3s2Back M
      (Tensor3.unflatten (StableHLO.batchSlice N (c * (2*h) * (2*w)) v r)) hc hh hw).close A hA).2)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched input-gradient VJP (the [3,4,6,3] fold at batch BN)
-- ════════════════════════════════════════════════════════════════

/-- **The batched whole-net input-gradient backward of ResNet-34** — the exact reverse of
    `resnet34ForwardB_full = head ∘ [3,4,6,3] ∘ stem`: dense-back → GAP-back → the sixteen basic
    blocks' backwards → the 3×3/s2 pool back → the stem's relu mask, BatchNorm back and 7×7/s2
    conv back. The block backwards and the stem's BatchNorm back are supplied; the conv, pool, GAP
    and dense leaves are concrete and lifted over the `N` examples.

    ⚠ Padding is SYMMETRIC at the stem (`flatConvStride2Back`, not the XLA-`SAME`
    `flatConvStride2XlaBack` B0's stem takes) — identical types, different certificates.
    ⭐ `N` is a variable: this chain carries no numerals. -/
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

/-- **The float batched ResNet-34 input-gradient skeleton** — `r34InputGradB` with each concrete
    slot replaced by the model's rounded peer and each supplied backward by its float map.
    `reluMaskBack` and `decimateBack` are structural selects/scatters and exact in float, so they
    are unchanged; ⛔ the 3×3/s2 pool's backward is NOT — its windows overlap, so it accumulates
    and rounds, per example. -/
noncomputable def r34InputGradBF (N : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 512 nCls)
    (bnBsF : Vec (N * (64 * (2 * 56) * (2 * 56))) → Vec (N * (64 * (2 * 56) * (2 * 56))))
    (xpool : Vec (N * (64 * (2 * 56) * (2 * 56))))
    (e1BF e0BF : Vec (N * (512 * 7 * 7)) → Vec (N * (512 * 7 * 7)))
    (d4BF : Vec (N * (512 * 7 * 7)) → Vec (N * (256 * 14 * 14)))
    (c4BF c3BF c2BF c1BF c0BF : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (d3BF : Vec (N * (256 * 14 * 14)) → Vec (N * (128 * 28 * 28)))
    (b2BF b1BF b0BF : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (d2BF : Vec (N * (128 * 28 * 28)) → Vec (N * (64 * 56 * 56)))
    (a2BF a1BF a0BF : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (m_stem : Fin (N * (64 * (2 * 56) * (2 * 56))) → Prop) [DecidablePred m_stem] :
    Vec (N * nCls) → Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) :=
  (StableHLO.batchMap N
      (M.flatConvF (h := 2 * (2 * 56)) (w := 2 * (2 * 56)) (IR.reverseSwap Ws) (fun _ => 0)
        ∘ decimateBack 64 (2 * 56) (2 * 56))
      ∘ bnBsF ∘ reluMaskBack m_stem)
  ∘ M.maxPool3s2FlatBackBF N 64 56 56 xpool
  ∘ a0BF ∘ a1BF ∘ a2BF
  ∘ d2BF
  ∘ b0BF ∘ b1BF ∘ b2BF
  ∘ d3BF
  ∘ c0BF ∘ c1BF ∘ c2BF ∘ c3BF ∘ c4BF
  ∘ d4BF
  ∘ e0BF ∘ e1BF
  ∘ StableHLO.batchMap N (gapBackF M 512 7 7)
  ∘ StableHLO.batchMap N (M.dense (Mat.transpose Wd) (0 : Vec 512))

set_option maxRecDepth 100000 in
/-- **The batched whole-net ResNet-34 backward float-bridges TO its float skeleton** — one
    `.comp` thread over the concrete endpoints, the supplied stem BatchNorm backward and the
    sixteen supplied block backwards. `r34_grad_floatBridgesTo` at true batch-norm and a variable
    batch size; the backward peer of `resnet34ForwardB_full`. -/
noncomputable def r34_grad_floatBridgesToB (N : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 512 nCls)
    (bnBs bnBsF : Vec (N * (64 * (2 * 56) * (2 * 56))) → Vec (N * (64 * (2 * 56) * (2 * 56))))
    (xpool : Vec (N * (64 * (2 * 56) * (2 * 56))))
    (e1B e0B e1BF e0BF : Vec (N * (512 * 7 * 7)) → Vec (N * (512 * 7 * 7)))
    (d4B d4BF : Vec (N * (512 * 7 * 7)) → Vec (N * (256 * 14 * 14)))
    (c4B c3B c2B c1B c0B c4BF c3BF c2BF c1BF c0BF :
      Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (d3B d3BF : Vec (N * (256 * 14 * 14)) → Vec (N * (128 * 28 * 28)))
    (b2B b1B b0B b2BF b1BF b0BF : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (d2B d2BF : Vec (N * (128 * 28 * 28)) → Vec (N * (64 * 56 * 56)))
    (a2B a1B a0B a2BF a1BF a0BF : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (m_stem : Fin (N * (64 * (2 * 56) * (2 * 56))) → Prop) [DecidablePred m_stem]
    {ws wd : ℝ} (hws : 0 ≤ ws) (hwd : 0 ≤ wd) (hnc : 0 < nCls)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWd : ∀ i j, |Wd i j| ≤ wd)
    (hbnBs : FloatBridgesTo bnBs bnBsF)
    (he1B : FloatBridgesTo e1B e1BF) (he0B : FloatBridgesTo e0B e0BF)
    (hd4B : FloatBridgesTo d4B d4BF)
    (hc4B : FloatBridgesTo c4B c4BF) (hc3B : FloatBridgesTo c3B c3BF)
    (hc2B : FloatBridgesTo c2B c2BF) (hc1B : FloatBridgesTo c1B c1BF)
    (hc0B : FloatBridgesTo c0B c0BF) (hd3B : FloatBridgesTo d3B d3BF)
    (hb2B : FloatBridgesTo b2B b2BF) (hb1B : FloatBridgesTo b1B b1BF)
    (hb0B : FloatBridgesTo b0B b0BF) (hd2B : FloatBridgesTo d2B d2BF)
    (ha2B : FloatBridgesTo a2B a2BF) (ha1B : FloatBridgesTo a1B a1BF)
    (ha0B : FloatBridgesTo a0B a0BF) :
    FloatBridgesTo
      (r34InputGradB N Ws Wd bnBs xpool e1B e0B d4B c4B c3B c2B c1B c0B d3B
        b2B b1B b0B d2B a2B a1B a0B m_stem)
      (r34InputGradBF N M Ws Wd bnBsF xpool e1BF e0BF d4BF c4BF c3BF c2BF c1BF c0BF d3BF
        b2BF b1BF b0BF d2BF a2BF a1BF a0BF m_stem) := by
  unfold r34InputGradB r34InputGradBF
  have hstem : FloatBridgesTo
      (StableHLO.batchMap N (flatConvStride2Back (h := 2 * 56) (w := 2 * 56) Ws)
        ∘ bnBs ∘ reluMaskBack m_stem)
      (StableHLO.batchMap N
          (M.flatConvF (h := 2 * (2 * 56)) (w := 2 * (2 * 56)) (IR.reverseSwap Ws) (fun _ => 0)
            ∘ decimateBack 64 (2 * 56) (2 * 56))
        ∘ bnBsF ∘ reluMaskBack m_stem) :=
    ((floatBridgesTo_reluMaskBack m_stem).comp hbnBs).comp
      (FloatBridgesTo.batchMap N
        (floatBridgesTo_flatConvStride2Back (h := 2 * 56) (w := 2 * 56) M Ws hws
          (by norm_num) hWs))
  have h0 := (FloatBridgesTo.batchMap N (floatBridgesTo_linBack M Wd hwd hnc hWd)).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_gapBack M 512 7 7 (by norm_num) (by norm_num) (by norm_num)))
  have hE := (h0.comp he1B).comp he0B
  have hD4 := hE.comp hd4B
  have hC := ((((hD4.comp hc4B).comp hc3B).comp hc2B).comp hc1B).comp hc0B
  have hD3 := hC.comp hd3B
  have hB := ((hD3.comp hb2B).comp hb1B).comp hb0B
  have hD2 := hB.comp hd2B
  have hA := ((hD2.comp ha2B).comp ha1B).comp ha0B
  have hMP := hA.comp
    (floatBridgesTo_maxPool3s2BackB M N 64 56 56 xpool (by norm_num) (by norm_num) (by norm_num))
  exact hMP.comp hstem

end Proofs
