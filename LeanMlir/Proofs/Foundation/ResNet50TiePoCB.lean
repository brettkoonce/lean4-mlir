import LeanMlir.Proofs.Foundation.ResNet50FaithfulPoCB
import LeanMlir.Proofs.Foundation.ResNet34TiePoCB
import LeanMlir.Proofs.Architectures.ResNet50FullBVJP
import LeanMlir.Proofs.Foundation.BceLossCot

/-! # ResNet-50's T3 §1a TIE — the un-fused, batched whole-net thread

`ResNet50FaithfulPoCB.lean` makes every parameter GRADIENT node of ResNet-50's batched train step
`den`-faithful for an ARBITRARY cotangent. This removes the "arbitrary": each is pinned to the one
the emitted backward chain delivers, so the whole train step is `den`-composed forward → loss →
backward with no free activation and no symbolic cotangent. With T1 and T2 that is ResNet-50's T3,
and it makes this the third net whose train-step tie is about the artifact its quoted accuracy
comes from.

⭐⭐ **The block cotangents are NOT derived here.** `ResNet50FullBVJP.lean`'s
`r50{Id,Proj,Down}B_has_vjp_at` ARE the certified block backwards, and `ResNet50BackB0.lean`'s
`r50{Bottleneck,ProjBlock,DownBlock}BackBatchedGraph_faithful` family already proves the emitted
backward subgraph denotes exactly them. The three `*CotIn_eq_vjp` lemmas below are that statement
in this file's vocabulary, so the cross-block chain is a composition of certified VJPs rather than
a re-derivation — the economy ResNet-34's 4.2a and MobileNetV2's 4.2c both took.

⭐⭐ **THE LOSS COTANGENT IS A BINDER, and for this net it had to be.** ResNet-34's and
MobileNetV2's capstones compute `g` internally from `smoothedLossCotGraph`. ResNet-50 ships BOTH
losses: `bce := false` artifacts carry the six-op label-smoothed softmax chain and `bce := true`
ones — including `resnet50in160_lambaccdp8x64bce`, where the 76.66% comes from — carry
BCE-with-logits' three-op chain. So `r50_net_tiedB` takes `g` as a hypothesis and the two loss
corollaries instantiate it: `r50_lossCot_is_smoothedCE_grad` and `r50_lossCot_is_bce_grad`. That is
4b's "the head takes `g` as a BINDER" made necessary rather than merely tidier.

⭐ **The stem tie is three nodes, not four, and the head tie is ResNet-34's.** `ResNet50RenderB` has
no `convBias` flag at all — its `zb` bakes `false` — so no conv-bias gradient op is ever emitted and
there is nothing to keep "to cover the flag", unlike r34's and MobileNetV2's ties.
`ResNet34TieB.r34HeadTiedB` is generic in `{c nCls}` and `r34HeadCotBlk` in the same, so the head
is reused verbatim at 2048 channels — as `r34HeadB` itself was in T1.

## The emitted chain, node for node

`ResNet50RenderB.bnkIdBackGradB`, from the block-output cotangent `dyOut`:

```
%da  = selectPosB(a)     %dn3 = bnBatchBack(g3, c3)   %dc3 = convBackBatched(W3)
%dr2 = selectPosB(n2)    %dn2 = bnBatchBack(g2, c2)   %dc2 = convBackBatched(W2)
%dr1 = selectPosB(n1)    %dn1 = bnBatchBack(g1, c1)   %dc1 = convBackBatched(W1)
%dx  = addVB(%dc1, %da)
```

and the nine parameter nodes read `W1 ← %dn1`, `g1/bt1 ← %dr1`, `W2 ← %dn2`, `g2/bt2 ← %dr2`,
`W3 ← %dn3`, `g3/bt3 ← %da`. ⚠ Off by one on any of those and the gradient is silently wrong; the
render's own comment records the same trap on the stochastic-depth cotangent.

The projection blocks add `%dnp = bnBatchBack(gp, cp, %da)` and `%dcp = convBack(Wp)`, and their
fan-in is `addVB(%dc1, %dcp)` — both branches nontrivial.

## Honest residual

⚠ **One `add_comm` per projection form.** The render emits `addVB(body, projection)` where
`residualProj proj body` adds `proj + body`, so `r50{Proj,Down}CotIn_eq_vjp` carry a commutation.
The identity block needs none. Same seam T2's graph faithfulness has, for the same reason.

⛔ **ONE REPLICA.** In `resnet50in160_lambaccdp8x64bce` every gradient node is followed by
`all_reduce(add)/4` as emitted text outside the AST, so every statement here is at the per-replica
gradient node (`Foundation/DataParallel.lean`, §4d). The 8× accumulation sits between the gradient
and the optimizer as `momVNextF` at `(μ := akeep)`, and the LAMB tail is `lamb_triple_faithful` —
both certified, neither part of this file.

⚠ **No smoothness hypothesis in the capstone**, exactly as r34's and mnv2's: the folds are `∀ cot`
statements instantiated at explicitly constructed cotangents. The relu-kink and positivity
conditions enter ONLY in the three `*CotIn_eq_vjp` lemmas, which say those cotangents ARE the
certified whole-net backward. `N` and `q` are both binders.
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.EnetTiePoC Proofs.ResNet34TieB

namespace Proofs.ResNet50TieB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The identity bottleneck — the render's cotangent chain
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the block's pre-relu sum `a` — the outer relu's mask applied to `dyOut`. Feeds
    bn₃'s γ/β directly AND, through the identity skip, the block-input fan-in. -/
noncomputable def r50IdCotA (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (residual (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) xin) dyOut

/-- Cotangent at conv₃'s output — `r50IdCotA` through bn₃'s backward. Feeds `W₃`. -/
noncomputable def r50IdCotC3 (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.ε₃ p.γ₃
    (batchMap N (flatConv p.W₃ p.b₃)
      (cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin)))
    (r50IdCotA N h w p xin dyOut)

/-- Cotangent at bn₂'s output — conv₃'s input-VJP masked by the second relu. Feeds `γ₂`/`β₂`. -/
noncomputable def r50IdCotN2 (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
      (batchMap N (flatConv p.W₂ p.b₂)
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin)))
    (cInB N p.W₃ p.b₃ (r50IdCotC3 N h w p xin dyOut))

/-- Cotangent at conv₂'s output — through bn₂'s backward. Feeds `W₂`. -/
noncomputable def r50IdCotC2 (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.ε₂ p.γ₂
    (batchMap N (flatConv p.W₂ p.b₂)
      (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin))
    (r50IdCotN2 N h w p xin dyOut)

/-- Cotangent at bn₁'s output — conv₂'s input-VJP masked by the first relu. Feeds `γ₁`/`β₁`. -/
noncomputable def r50IdCotN1 (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) xin))
    (cInB N p.W₂ p.b₂ (r50IdCotC2 N h w p xin dyOut))

/-- Cotangent at conv₁'s output — through bn₁'s backward. Feeds `W₁`. -/
noncomputable def r50IdCotC1 (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.ε₁ p.γ₁ (batchMap N (flatConv p.W₁ p.b₁) xin)
    (r50IdCotN1 N h w p xin dyOut)

/-- **The block-INPUT cotangent**: the residual fan-in `addVB(convBack(dn1), da)` the render emits. -/
noncomputable def r50IdCotIn (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  fun i => cInB N p.W₁ p.b₁ (r50IdCotC1 N h w p xin dyOut) i
    + r50IdCotA N h w p xin dyOut i

/-- ⭐⭐ **The emitted fan-in IS the certified bottleneck VJP's backward.** `rfl` after the graph
    lemma: the render's ten-node backward subgraph denotes `(r50IdB_has_vjp_at …).backward dyOut`. -/
theorem r50IdCotIn_eq_vjp (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) (hq : R50IdPos p)
    (xin dyOut : Vec (N * (oc * h * w))) (hs : R50IdSmoothAt N h w p xin) (cotN : String) :
    r50IdCotIn N h w p xin dyOut
      = (r50IdB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := r50BottleneckBackBatchedGraph_faithful (N := N) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃ xin (.operand cotN dyOut)
    hs.hm1 hs.hm2 hs.hout
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [r50IdB_has_vjp_at, ← h]
  rfl


-- ════════════════════════════════════════════════════════════════
-- § The stride-1 projection bottleneck — stage 1 block 0
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the pre-relu sum. Feeds bn₃'s AND the projection's γ/β, and the projection's
    whole backward branch. -/
noncomputable def r50ProjCotA (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (residualProj (projB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
      (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
        cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
        cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) xin) dyOut

/-- Cotangent at conv₃'s output. Feeds `W₃`. -/
noncomputable def r50ProjCotC3 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.ε₃ p.γ₃
    (batchMap N (flatConv p.W₃ p.b₃)
      (cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin)))
    (r50ProjCotA N h w p xin dyOut)

/-- Cotangent at bn₂'s output. Feeds `γ₂`/`β₂`. -/
noncomputable def r50ProjCotN2 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
      (batchMap N (flatConv p.W₂ p.b₂)
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin)))
    (cInB N p.W₃ p.b₃ (r50ProjCotC3 N h w p xin dyOut))

/-- Cotangent at conv₂'s output. Feeds `W₂`. -/
noncomputable def r50ProjCotC2 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.ε₂ p.γ₂
    (batchMap N (flatConv p.W₂ p.b₂)
      (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin))
    (r50ProjCotN2 N h w p xin dyOut)

/-- Cotangent at bn₁'s output. Feeds `γ₁`/`β₁`. -/
noncomputable def r50ProjCotN1 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) xin))
    (cInB N p.W₂ p.b₂ (r50ProjCotC2 N h w p xin dyOut))

/-- Cotangent at conv₁'s output. Feeds `W₁`. -/
noncomputable def r50ProjCotC1 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.ε₁ p.γ₁ (batchMap N (flatConv p.W₁ p.b₁) xin)
    (r50ProjCotN1 N h w p xin dyOut)

/-- Cotangent at the PROJECTION conv's output — `r50ProjCotA` through the skip BN's backward.
    Feeds `Wp`. ⚠ It reads the UNMASKED-by-drop `%da`, which is the render's rule: the projection
    branch is never dropped. -/
noncomputable def r50ProjCotCp (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.εp p.γp (batchMap N (flatConv p.Wp p.bp) xin)
    (r50ProjCotA N h w p xin dyOut)

/-- **The block-INPUT cotangent**: `addVB(convBack(dn1), convBack(dnp))` — both branches
    nontrivial, unlike the identity block's. -/
noncomputable def r50ProjCotIn (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Vec (N * (ic * h * w)) :=
  fun i => cInB N p.W₁ p.b₁ (r50ProjCotC1 N h w p xin dyOut) i
    + cInB N p.Wp p.bp (r50ProjCotCp N h w p xin dyOut) i

/-- ⭐ **The projected fan-in IS the certified stride-1 projection block's backward.** ⚠ One
    `add_comm`: the render emits `addVB(body, projection)` and the graph builds
    `addV(projection, body)`. -/
theorem r50ProjCotIn_eq_vjp (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w)))
    (hs : R50ProjSmoothAt N h w p xin) (cotN : String) :
    r50ProjCotIn N h w p xin dyOut
      = (r50ProjB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := r50ProjBlockBackBatchedGraph_faithful (N := N) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃
    p.Wp p.bp p.εp hq.hp p.γp p.βp xin (.operand cotN dyOut) hs.hm1 hs.hm2 hs.hout
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [r50ProjB_has_vjp_at, ← h]
  funext i
  exact add_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § The strided projection bottleneck — stages 2/3/4 block 0
--   ⚠⚠ v1.5: conv₁/bn₁/relu₁ run at `2h × 2w`; only conv₂ and the skip are strided.
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the pre-relu sum. -/
noncomputable def r50DownCotA (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (residualProj (projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
      (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
        cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
        cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) xin) dyOut

/-- Cotangent at conv₃'s output. Feeds `W₃` (a stride-1 1×1 at `h × w`). -/
noncomputable def r50DownCotC3 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.ε₃ p.γ₃
    (batchMap N (flatConv p.W₃ p.b₃)
      (cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin)))
    (r50DownCotA N h w p xin dyOut)

/-- Cotangent at bn₂'s output, at `h × w`. Feeds `γ₂`/`β₂`. -/
noncomputable def r50DownCotN2 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
      (batchMap N (flatConvStride2 p.W₂ p.b₂)
        (cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin)))
    (cInB N p.W₃ p.b₃ (r50DownCotC3 N h w p xin dyOut))

/-- Cotangent at the STRIDED conv₂'s output. Feeds `W₂`. -/
noncomputable def r50DownCotC2 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.ε₂ p.γ₂
    (batchMap N (flatConvStride2 p.W₂ p.b₂)
      (cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin))
    (r50DownCotN2 N h w p xin dyOut)

/-- Cotangent at bn₁'s output — at the INPUT grid `2h × 2w`, because conv₂ upsamples. Feeds
    `γ₁`/`β₁`. ⚠⚠ Writing this at `h × w` typechecks nowhere, and it is the one place a reader can
    get v1.5's shape wrong. -/
noncomputable def r50DownCotN1 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  reluMaskB (N * (mid * (2 * h) * (2 * w)))
    (bnBatchLA N mid (2 * h) (2 * w) p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) xin))
    (cStridedInB N p.W₂ p.b₂ (r50DownCotC2 N h w p xin dyOut))

/-- Cotangent at conv₁'s output, at `2h × 2w`. Feeds `W₁`, an ORDINARY conv weight gradient. -/
noncomputable def r50DownCotC1 (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  bnInB N mid (2 * h) (2 * w) p.ε₁ p.γ₁ (batchMap N (flatConv p.W₁ p.b₁) xin)
    (r50DownCotN1 N h w p xin dyOut)

/-- Cotangent at the STRIDED projection conv's output. Feeds `Wp`. -/
noncomputable def r50DownCotCp (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.εp p.γp (batchMap N (flatConvStride2 p.Wp p.bp) xin)
    (r50DownCotA N h w p xin dyOut)

/-- **The block-INPUT cotangent**: `addVB(convBack(dn1), convStridedBack(dnp))`. -/
noncomputable def r50DownCotIn (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (ic * (2 * h) * (2 * w))) :=
  fun i => cInB N p.W₁ p.b₁ (r50DownCotC1 N h w p xin dyOut) i
    + cStridedInB N p.Wp p.bp (r50DownCotCp N h w p xin dyOut) i

/-- ⭐ **The projected fan-in IS the certified strided block's backward.** One `add_comm`, as the
    stride-1 projection's is. -/
theorem r50DownCotIn_eq_vjp (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyOut : Vec (N * (oc * h * w))) (hs : R50DownSmoothAt N h w p xin) (cotN : String) :
    r50DownCotIn N h w p xin dyOut
      = (r50DownB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := r50DownBlockBackBatchedGraph_faithful (N := N) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃
    p.Wp p.bp p.εp hq.hp p.γp p.βp xin (.operand cotN dyOut) hs.hm1 hs.hm2 hs.hout
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [r50DownB_has_vjp_at, ← h]
  funext i
  exact add_comm _ _


-- ════════════════════════════════════════════════════════════════
-- § The per-block-type tie bundles — every parameter node at its chain cotangent
-- ════════════════════════════════════════════════════════════════

/-! Each conjunct is `ResNet50FaithfulPoCB`'s `∀ cot` fold instantiated at the cotangent the
render's chain delivers, so nothing here is a new proof: the bundles are the §1 fold with the
freedom removed. `reassocB` bridges the conv/relu index `N·(c·h·w)` to the BatchNorm parameter
ops' `N·(c·(h·w))`. ⛔ There are NO conv-bias conjuncts: `ResNet50RenderB` has no `convBias` flag,
so those ops are never emitted and every slot here is exercised by the artifact. -/
/-- **Identity bottleneck, tied.** All NINE parameter nodes — three conv weights and three
    BatchNorm γ/β pairs — denote the certified batched `Σ_n` gradient at the real forward
    activations and the real backward-chain cotangent driven by `dyOut`. ⚠ Each BatchNorm's γ/β
    reads the cotangent at THAT BatchNorm's output (`cotN1`, `cotN2`, `cotA`) while its conv reads
    the one at the conv's output (`cotC1`, `cotC2`, `cotC3`); off by one and the gradient is
    silently wrong. -/
def r50IdTiedB (N h w : Nat) {mid oc : Nat} (xN cotN vN epsStr : String) (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin
  let r2 := cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap N (flatConv p.W₁ p.b₁) xin
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let c3 := batchMap N (flatConv p.W₃ p.b₃) r2
  let cotA := r50IdCotA N h w p xin dyOut
  let cotC3 := r50IdCotC3 N h w p xin dyOut
  let cotN2 := r50IdCotN2 N h w p xin dyOut
  let cotC2 := r50IdCotC2 N h w p xin dyOut
  let cotN1 := r50IdCotN1 N h w p xin dyOut
  let cotC1 := r50IdCotC1 N h w p xin dyOut
  (∀ idx : Fin (mid * oc * 1 * 1),
      den (SHlo.convWeightGradB xN p.b₁ xin p.W₁ (.operand cotN cotC1)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * oc * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₁
                      (Tensor3.unflatten (batchSlice N (oc * h * w) xin n))))
                 (Kernel4.flatten p.W₁) idx j * batchSlice N (mid * h * w) cotC1 n j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N (mid) (h) (w) c1)
            (.operand cotN (reassocB N (mid) (h) (w) cotN1))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₁ γ' p.β₁
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c1)))
                 p.γ₁ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN1) j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N (mid) (h) (w) cotN1))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₁ p.γ₁ β'
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c1)))
                 p.β₁ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN1) j)
  ∧
  (∀ idx : Fin (mid * mid * 3 * 3),
      den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN cotC2)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * mid * 3 * 3) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₂
                      (Tensor3.unflatten (batchSlice N (mid * h * w) r1 n))))
                 (Kernel4.flatten p.W₂) idx j * batchSlice N (mid * h * w) cotC2 n j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N (mid) (h) (w) c2)
            (.operand cotN (reassocB N (mid) (h) (w) cotN2))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₂ γ' p.β₂
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c2)))
                 p.γ₂ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN2) j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N (mid) (h) (w) cotN2))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₂ p.γ₂ β'
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c2)))
                 p.β₂ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN2) j)
  ∧
  (∀ idx : Fin (oc * mid * 1 * 1),
      den (SHlo.convWeightGradB xN p.b₃ r2 p.W₃ (.operand cotN cotC3)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₃
                      (Tensor3.unflatten (batchSlice N (mid * h * w) r2 n))))
                 (Kernel4.flatten p.W₃) idx j * batchSlice N (oc * h * w) cotC3 n j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnGammaGradB vN epsStr p.ε₃ (reassocB N (oc) (h) (w) c3)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.ε₃ γ' p.β₃
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) c3)))
                 p.γ₃ k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.ε₃ p.γ₃ β'
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) c3)))
                 p.β₃ k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)

theorem r50_idblock_tiedB (N h w : Nat) {mid oc : Nat} (xN cotN vN epsStr : String) (p : R50IdW mid oc)
    (xin dyOut : Vec (N * (oc * h * w))) :
    r50IdTiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold r50IdTiedB
  intro r1 r2 c1 c2 c3 cotA cotC3 cotN2 cotC2 cotN1 cotC1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₁ xin p.W₁ cotC1 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₁ p.γ₁ p.β₁
                 (reassocB N mid h w c1) (reassocB N mid h w cotN1) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₁ p.γ₁ p.β₁
                 (bnchwFwd N mid h w (reassocB N mid h w c1)) (reassocB N mid h w cotN1) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₂ r1 p.W₂ cotC2 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₂ p.γ₂ p.β₂
                 (reassocB N mid h w c2) (reassocB N mid h w cotN2) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₂ p.γ₂ p.β₂
                 (bnchwFwd N mid h w (reassocB N mid h w c2)) (reassocB N mid h w cotN2) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₃ r2 p.W₃ cotC3 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₃ p.γ₃ p.β₃
                 (reassocB N oc h w c3) (reassocB N oc h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₃ p.γ₃ p.β₃
                 (bnchwFwd N oc h w (reassocB N oc h w c3)) (reassocB N oc h w cotA) k

/-- ⭐ **Stride-1 projection bottleneck, tied.** Twelve nodes: the identity block's nine plus the
    1×1 skip's weight and its BatchNorm γ/β. ⚠ The skip's conv is an ORDINARY `convWeightGradB` —
    stage 1 block 0 changes channels but not resolution, which is the whole reason this block form
    exists. ⚠ The skip's three nodes read `cotA`, the UNMASKED post-relu cotangent: the projection
    branch is never stochastic-depth dropped, which is the render's own rule. -/
def r50ProjTiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin
  let r2 := cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap N (flatConv p.W₁ p.b₁) xin
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let c3 := batchMap N (flatConv p.W₃ p.b₃) r2
  let cp := batchMap N (flatConv p.Wp p.bp) xin
  let cotA := r50ProjCotA N h w p xin dyOut
  let cotC3 := r50ProjCotC3 N h w p xin dyOut
  let cotN2 := r50ProjCotN2 N h w p xin dyOut
  let cotC2 := r50ProjCotC2 N h w p xin dyOut
  let cotN1 := r50ProjCotN1 N h w p xin dyOut
  let cotC1 := r50ProjCotC1 N h w p xin dyOut
  let cotCp := r50ProjCotCp N h w p xin dyOut
  (∀ idx : Fin (mid * ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.b₁ xin p.W₁ (.operand cotN cotC1)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₁
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 (Kernel4.flatten p.W₁) idx j * batchSlice N (mid * h * w) cotC1 n j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N (mid) (h) (w) c1)
            (.operand cotN (reassocB N (mid) (h) (w) cotN1))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₁ γ' p.β₁
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c1)))
                 p.γ₁ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN1) j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N (mid) (h) (w) cotN1))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₁ p.γ₁ β'
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c1)))
                 p.β₁ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN1) j)
  ∧
  (∀ idx : Fin (mid * mid * 3 * 3),
      den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN cotC2)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * mid * 3 * 3) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₂
                      (Tensor3.unflatten (batchSlice N (mid * h * w) r1 n))))
                 (Kernel4.flatten p.W₂) idx j * batchSlice N (mid * h * w) cotC2 n j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N (mid) (h) (w) c2)
            (.operand cotN (reassocB N (mid) (h) (w) cotN2))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₂ γ' p.β₂
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c2)))
                 p.γ₂ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN2) j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N (mid) (h) (w) cotN2))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₂ p.γ₂ β'
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c2)))
                 p.β₂ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN2) j)
  ∧
  (∀ idx : Fin (oc * mid * 1 * 1),
      den (SHlo.convWeightGradB xN p.b₃ r2 p.W₃ (.operand cotN cotC3)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₃
                      (Tensor3.unflatten (batchSlice N (mid * h * w) r2 n))))
                 (Kernel4.flatten p.W₃) idx j * batchSlice N (oc * h * w) cotC3 n j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnGammaGradB vN epsStr p.ε₃ (reassocB N (oc) (h) (w) c3)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.ε₃ γ' p.β₃
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) c3)))
                 p.γ₃ k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.ε₃ p.γ₃ β'
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) c3)))
                 p.β₃ k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)
  ∧
  (∀ idx : Fin (oc * ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.bp xin p.Wp (.operand cotN cotCp)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.bp
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 (Kernel4.flatten p.Wp) idx j * batchSlice N (oc * h * w) cotCp n j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnGammaGradB vN epsStr p.εp (reassocB N (oc) (h) (w) cp)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.εp γ' p.βp
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cp)))
                 p.γp k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.εp p.γp β'
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cp)))
                 p.βp k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)

theorem r50_projblock_tiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    r50ProjTiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold r50ProjTiedB
  intro r1 r2 c1 c2 c3 cp cotA cotC3 cotN2 cotC2 cotN1 cotC1 cotCp
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₁ xin p.W₁ cotC1 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₁ p.γ₁ p.β₁
                 (reassocB N mid h w c1) (reassocB N mid h w cotN1) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₁ p.γ₁ p.β₁
                 (bnchwFwd N mid h w (reassocB N mid h w c1)) (reassocB N mid h w cotN1) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₂ r1 p.W₂ cotC2 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₂ p.γ₂ p.β₂
                 (reassocB N mid h w c2) (reassocB N mid h w cotN2) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₂ p.γ₂ p.β₂
                 (bnchwFwd N mid h w (reassocB N mid h w c2)) (reassocB N mid h w cotN2) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₃ r2 p.W₃ cotC3 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₃ p.γ₃ p.β₃
                 (reassocB N oc h w c3) (reassocB N oc h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₃ p.γ₃ p.β₃
                 (bnchwFwd N oc h w (reassocB N oc h w c3)) (reassocB N oc h w cotA) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.bp xin p.Wp cotCp idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.εp p.γp p.βp
                 (reassocB N oc h w cp) (reassocB N oc h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.εp p.γp p.βp
                 (bnchwFwd N oc h w (reassocB N oc h w cp)) (reassocB N oc h w cotA) k

/-- **Strided projection bottleneck, tied.** Twelve nodes, and TWO of the four conv weights are
    the strided op. ⚠⚠ v1.5: `W₁` is an ordinary `convWeightGradB` at the INPUT grid `2h × 2w` and
    its BatchNorm reduces there too; only `W₂` (the 3×3) and `Wp` (the skip) are strided. Both
    strided nodes are SYMMETRIC padding — `flatConvStride2`, not the XLA-`SAME` twin B0 and
    MobileNetV2 use. -/
def r50DownTiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin
  let r2 := cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap N (flatConv p.W₁ p.b₁) xin
  let c2 := batchMap N (flatConvStride2 p.W₂ p.b₂) r1
  let c3 := batchMap N (flatConv p.W₃ p.b₃) r2
  let cp := batchMap N (flatConvStride2 p.Wp p.bp) xin
  let cotA := r50DownCotA N h w p xin dyOut
  let cotC3 := r50DownCotC3 N h w p xin dyOut
  let cotN2 := r50DownCotN2 N h w p xin dyOut
  let cotC2 := r50DownCotC2 N h w p xin dyOut
  let cotN1 := r50DownCotN1 N h w p xin dyOut
  let cotC1 := r50DownCotC1 N h w p xin dyOut
  let cotCp := r50DownCotCp N h w p xin dyOut
  (∀ idx : Fin (mid * ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.b₁ xin p.W₁ (.operand cotN cotC1)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * (2 * h) * (2 * w)),
            pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₁
                      (Tensor3.unflatten (batchSlice N (ic * (2 * h) * (2 * w)) xin n))))
                 (Kernel4.flatten p.W₁) idx j * batchSlice N (mid * (2 * h) * (2 * w)) cotC1 n j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N (mid) (2 * h) (2 * w) c1)
            (.operand cotN (reassocB N (mid) (2 * h) (2 * w) cotN1))) k
        = ∑ j : Fin ((mid) * (N * ((2 * h) * (2 * w)))),
            pdiv (fun γ' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((2 * h) * (2 * w))) p.ε₁ γ' p.β₁
                      (bnchwFwd N (mid) (2 * h) (2 * w) (reassocB N (mid) (2 * h) (2 * w) c1)))
                 p.γ₁ k j * bnchwFwd N (mid) (2 * h) (2 * w) (reassocB N (mid) (2 * h) (2 * w) cotN1) j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := 2 * h) (w := 2 * w)
            (.operand cotN (reassocB N (mid) (2 * h) (2 * w) cotN1))) k
        = ∑ j : Fin ((mid) * (N * ((2 * h) * (2 * w)))),
            pdiv (fun β' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((2 * h) * (2 * w))) p.ε₁ p.γ₁ β'
                      (bnchwFwd N (mid) (2 * h) (2 * w) (reassocB N (mid) (2 * h) (2 * w) c1)))
                 p.β₁ k j * bnchwFwd N (mid) (2 * h) (2 * w) (reassocB N (mid) (2 * h) (2 * w) cotN1) j)
  ∧
  (∀ idx : Fin (mid * mid * 3 * 3),
      den (SHlo.convStridedWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN cotC2)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * mid * 3 * 3) =>
                    flatConvStride2 (Kernel4.unflatten v') p.b₂
                      (batchSlice N (mid * (2 * h) * (2 * w)) r1 n))
                 (Kernel4.flatten p.W₂) idx j * batchSlice N (mid * h * w) cotC2 n j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N (mid) (h) (w) c2)
            (.operand cotN (reassocB N (mid) (h) (w) cotN2))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₂ γ' p.β₂
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c2)))
                 p.γ₂ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN2) j)
  ∧
  (∀ k : Fin (mid),
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N (mid) (h) (w) cotN2))) k
        = ∑ j : Fin ((mid) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (mid) =>
                    bnPerChannelFlat (mid) (N * ((h) * (w))) p.ε₂ p.γ₂ β'
                      (bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) c2)))
                 p.β₂ k j * bnchwFwd N (mid) (h) (w) (reassocB N (mid) (h) (w) cotN2) j)
  ∧
  (∀ idx : Fin (oc * mid * 1 * 1),
      den (SHlo.convWeightGradB xN p.b₃ r2 p.W₃ (.operand cotN cotC3)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₃
                      (Tensor3.unflatten (batchSlice N (mid * h * w) r2 n))))
                 (Kernel4.flatten p.W₃) idx j * batchSlice N (oc * h * w) cotC3 n j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnGammaGradB vN epsStr p.ε₃ (reassocB N (oc) (h) (w) c3)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.ε₃ γ' p.β₃
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) c3)))
                 p.γ₃ k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.ε₃ p.γ₃ β'
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) c3)))
                 p.β₃ k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)
  ∧
  (∀ idx : Fin (oc * ic * 1 * 1),
      den (SHlo.convStridedWeightGradB xN p.bp xin p.Wp (.operand cotN cotCp)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                    flatConvStride2 (Kernel4.unflatten v') p.bp
                      (batchSlice N (ic * (2 * h) * (2 * w)) xin n))
                 (Kernel4.flatten p.Wp) idx j * batchSlice N (oc * h * w) cotCp n j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnGammaGradB vN epsStr p.εp (reassocB N (oc) (h) (w) cp)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun γ' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.εp γ' p.βp
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cp)))
                 p.γp k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)
  ∧
  (∀ k : Fin (oc),
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N (oc) (h) (w) cotA))) k
        = ∑ j : Fin ((oc) * (N * ((h) * (w)))),
            pdiv (fun β' : Vec (oc) =>
                    bnPerChannelFlat (oc) (N * ((h) * (w))) p.εp p.γp β'
                      (bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cp)))
                 p.βp k j * bnchwFwd N (oc) (h) (w) (reassocB N (oc) (h) (w) cotA) j)

theorem r50_downblock_tiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    r50DownTiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold r50DownTiedB
  intro r1 r2 c1 c2 c3 cp cotA cotC3 cotN2 cotC2 cotN1 cotC1 cotCp
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₁ xin p.W₁ cotC1 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₁ p.γ₁ p.β₁
                 (reassocB N mid (2 * h) (2 * w) c1) (reassocB N mid (2 * h) (2 * w) cotN1) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₁ p.γ₁ p.β₁
                 (bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) c1))
                 (reassocB N mid (2 * h) (2 * w) cotN1) k
  · intro idx; exact ResNet34PoCB.convStridedWGradB_den xN cotN p.b₂ r1 p.W₂ cotC2 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₂ p.γ₂ p.β₂
                 (reassocB N mid h w c2) (reassocB N mid h w cotN2) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₂ p.γ₂ p.β₂
                 (bnchwFwd N mid h w (reassocB N mid h w c2)) (reassocB N mid h w cotN2) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₃ r2 p.W₃ cotC3 idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₃ p.γ₃ p.β₃
                 (reassocB N oc h w c3) (reassocB N oc h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₃ p.γ₃ p.β₃
                 (bnchwFwd N oc h w (reassocB N oc h w c3)) (reassocB N oc h w cotA) k
  · intro idx; exact ResNet34PoCB.convStridedWGradB_den xN cotN p.bp xin p.Wp cotCp idx
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.εp p.γp p.βp
                 (reassocB N oc h w cp) (reassocB N oc h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.εp p.γp p.βp
                 (bnchwFwd N oc h w (reassocB N oc h w cp)) (reassocB N oc h w cotA) k


-- ════════════════════════════════════════════════════════════════
-- § The whole-net capstone
--   ⭐ The stem and head bundles are ResNet-34's, reused verbatim: `r34StemTiedB` is generic in
--   `{ic oc}` and `r34HeadTiedB` in `{c nCls}`, and ResNet-50's stem and head ARE those functions
--   at different widths (`ResNet50FullB.lean` builds the net from `r34StemB` and `r34HeadB`).
--   ⚠ `r34StemTiedB` carries a conv-BIAS conjunct that ResNet-50 never emits; it costs one
--   delegation and is true at `bias = 0`, so the stem contributes 3 exercised slots of 4.
-- ════════════════════════════════════════════════════════════════

set_option maxHeartbeats 1600000 in
/-- ⭐⭐ **The whole batch-BN ResNet-50 train step, tied.** Threading `resnet50ForwardB_full`'s own
    prefixes as the block inputs and an arbitrary loss cotangent `g` down through the certified head
    backward and the sixteen certified bottleneck backwards, every parameter GRADIENT node of the
    net — stem 3, twelve identity bottlenecks × 9, four projection bottlenecks × 12, dense 2 —
    denotes the certified batched `Σ_n` gradient. That is **161**, the render's own census and the
    signature of `resnet50_fwd.mlir` minus `%x`. No free activation and no symbolic cotangent
    below the loss.

    ⭐⭐ **`g` IS A BINDER, and for this net it had to be.** ResNet-50 ships both losses — the
    label-smoothed softmax chain on the `bce := false` artifacts and BCE-with-logits' three-op
    chain on the `bce := true` ones, including `resnet50in160_lambaccdp8x64bce` where the quoted
    76.66% comes from. `r50_lossCot_is_smoothedCE_grad` and `r50_lossCot_is_bce_grad` instantiate
    it; neither is privileged.

    ⭐ **No smoothness hypothesis, no `0 < ε`, and `N` and `q` are both binders.** The folds are
    `∀ cot` statements at explicitly constructed cotangents. The kink and positivity conditions
    enter only in `r50{Id,Proj,Down}CotIn_eq_vjp`, which say those cotangents ARE the certified
    whole-net backward — the two halves of the tie, kept apart because they have different
    hypotheses.

    ⛔ **One replica.** In `resnet50in160_lambaccdp8x64bce` every gradient node is followed by
    `all_reduce(add)/4` as emitted text outside the AST (`DataParallel.lean`, §4d), and the 8×
    accumulation and the LAMB tail sit downstream of every node named here. -/
theorem r50_net_tiedB (N q : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) (g : Vec (N * nCls)) :
    let dy16 := r34HeadCotBlk N q q w.Wd w.bd (r50Pre16 N q w x) g
    let dy15 := r50IdCotIn N q q w.s4b2 (r50Pre15 N q w x) dy16
    let dy14 := r50IdCotIn N q q w.s4b1 (r50Pre14 N q w x) dy15
    let dy13 := r50DownCotIn N q q w.s4b0 (r50Pre13 N q w x) dy14
    let dy12 := r50IdCotIn N (2 * q) (2 * q) w.s3b5 (r50Pre12 N q w x) dy13
    let dy11 := r50IdCotIn N (2 * q) (2 * q) w.s3b4 (r50Pre11 N q w x) dy12
    let dy10 := r50IdCotIn N (2 * q) (2 * q) w.s3b3 (r50Pre10 N q w x) dy11
    let dy9 := r50IdCotIn N (2 * q) (2 * q) w.s3b2 (r50Pre9 N q w x) dy10
    let dy8 := r50IdCotIn N (2 * q) (2 * q) w.s3b1 (r50Pre8 N q w x) dy9
    let dy7 := r50DownCotIn N (2 * q) (2 * q) w.s3b0 (r50Pre7 N q w x) dy8
    let dy6 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 N q w x) dy7
    let dy5 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 N q w x) dy6
    let dy4 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 N q w x) dy5
    let dy3 := r50DownCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 N q w x) dy4
    let dy2 := r50IdCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Pre2 N q w x) dy3
    let dy1 := r50IdCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Pre1 N q w x) dy2
    let cotPool := r50ProjCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Pre0 N q w x) dy1
    r34StemTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotPool
  ∧ r50ProjTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b0 (r50Pre0 N q w x) dy1
  ∧ r50IdTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b1 (r50Pre1 N q w x) dy2
  ∧ r50IdTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b2 (r50Pre2 N q w x) dy3
  ∧ r50DownTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b0 (r50Pre3 N q w x) dy4
  ∧ r50IdTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b1 (r50Pre4 N q w x) dy5
  ∧ r50IdTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b2 (r50Pre5 N q w x) dy6
  ∧ r50IdTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b3 (r50Pre6 N q w x) dy7
  ∧ r50DownTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b0 (r50Pre7 N q w x) dy8
  ∧ r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b1 (r50Pre8 N q w x) dy9
  ∧ r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b2 (r50Pre9 N q w x) dy10
  ∧ r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b3 (r50Pre10 N q w x) dy11
  ∧ r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b4 (r50Pre11 N q w x) dy12
  ∧ r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b5 (r50Pre12 N q w x) dy13
  ∧ r50DownTiedB N q q xN cotN vN epsStr w.s4b0 (r50Pre13 N q w x) dy14
  ∧ r50IdTiedB N q q xN cotN vN epsStr w.s4b1 (r50Pre14 N q w x) dy15
  ∧ r50IdTiedB N q q xN cotN vN epsStr w.s4b2 (r50Pre15 N q w x) dy16
  ∧ r34HeadTiedB N q q xN cotN w.Wd w.bd (r50Pre16 N q w x) g := by
  intro dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotPool
  exact ⟨r34_stem_tiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotPool,
    r50_projblock_tiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b0 (r50Pre0 N q w x) dy1,
    r50_idblock_tiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b1 (r50Pre1 N q w x) dy2,
    r50_idblock_tiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b2 (r50Pre2 N q w x) dy3,
    r50_downblock_tiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b0 (r50Pre3 N q w x) dy4,
    r50_idblock_tiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b1 (r50Pre4 N q w x) dy5,
    r50_idblock_tiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b2 (r50Pre5 N q w x) dy6,
    r50_idblock_tiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b3 (r50Pre6 N q w x) dy7,
    r50_downblock_tiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b0 (r50Pre7 N q w x) dy8,
    r50_idblock_tiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b1 (r50Pre8 N q w x) dy9,
    r50_idblock_tiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b2 (r50Pre9 N q w x) dy10,
    r50_idblock_tiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b3 (r50Pre10 N q w x) dy11,
    r50_idblock_tiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b4 (r50Pre11 N q w x) dy12,
    r50_idblock_tiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b5 (r50Pre12 N q w x) dy13,
    r50_downblock_tiedB N q q xN cotN vN epsStr w.s4b0 (r50Pre13 N q w x) dy14,
    r50_idblock_tiedB N q q xN cotN vN epsStr w.s4b1 (r50Pre14 N q w x) dy15,
    r50_idblock_tiedB N q q xN cotN vN epsStr w.s4b2 (r50Pre15 N q w x) dy16,
    r34_head_tiedB N q q xN cotN w.Wd w.bd (r50Pre16 N q w x) g⟩


-- ════════════════════════════════════════════════════════════════
-- § The two losses the binder `g` is instantiated at
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The label-smoothed cotangent, for every `bce := false` artifact.** Row by row, the six-op
    chain `ResNet50RenderB` emits is `(1/B)·∂/∂logits` of soft-target cross-entropy against the
    SMOOTHED target, at that example's real logits. The only hypothesis is that the example's
    target sums to 1 — a one-hot, or mixup's convex combination. -/
theorem r50_lossCot_is_smoothedCE_grad (N q : Nat) {nCls : Nat} (hK : 0 < nCls)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) (t : Vec (N * (1 * nCls)))
    (n : Fin N) (j : Fin nCls)
    (ht : ∑ k : Fin nCls, Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1) k = 1) :
    den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
          (rowB N nCls (resnet50ForwardB_full N q w x)) t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec nCls => fun _ : Fin 1 =>
            softCE nCls (smoothTarget nCls α
              (Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1))) z')
          (Mat.unflatten (batchSlice N (1 * nCls)
            (rowB N nCls (resnet50ForwardB_full N q w x)) n) (0 : Fin 1)) j 0) / B :=
  smoothedLossCotGraph_row N nCls hK α B aStr negAK bStr logN ohN _ t n j ht

/-- ⭐⭐ **The BCE-with-logits cotangent, for every `bce := true` artifact — including the one the
    76.66% comes from.** Row by row, the three-op chain `sigmoidB → subB → divConstB` is
    `∂/∂logits` of `Σ_k (softplus(z_k) − t_k·z_k)` at that example's real logits, over the baked
    `N·K`. ⚠⚠ The divisor is `N·K`, not `N`: timm's `BinaryCrossEntropy` is `reduction='mean'` over
    `B×C`, and at `K = 1000` the two differ by 1000× on the effective step. ⭐ NO hypothesis on the
    target at all, where the smoothed-CE row needs its mass to be 1 — BCE is per-class and
    separable, which is the point under mixup. -/
theorem r50_lossCot_is_bce_grad (N q : Nat) {nCls : Nat} (bStr logN ohN : String)
    (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) (t : Vec (N * (1 * nCls)))
    (n : Fin N) (j : Fin nCls) :
    den (bceLossCotGraph N nCls ((N : ℝ) * (nCls : ℝ)) bStr logN ohN
          (rowB N nCls (resnet50ForwardB_full N q w x)) t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec nCls => fun _ : Fin 1 =>
            bceLogits nCls (Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1)) z')
          (Mat.unflatten (batchSlice N (1 * nCls)
            (rowB N nCls (resnet50ForwardB_full N q w x)) n) (0 : Fin 1)) j 0)
        / ((N : ℝ) * (nCls : ℝ)) :=
  bceLossCotGraph_row_committed N nCls bStr logN ohN _ t n j

end Proofs.ResNet50TieB
