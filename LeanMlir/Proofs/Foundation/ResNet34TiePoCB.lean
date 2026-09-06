import LeanMlir.Proofs.Foundation.ResNet34FaithfulPoCB
import LeanMlir.Proofs.Architectures.EfficientNetTiePoC
import LeanMlir.Proofs.Architectures.ResNet34FullBVJP
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Architectures.EfficientNetFaithfulPoCG

/-! # ResNet-34's T3 §1a TIE at TRUE BATCH-NORM — the un-fused, batched whole-net thread

`ResNet34FaithfulPoCB.lean` (4.1e) makes every parameter GRADIENT node of the batched ResNet-34
train step `den`-faithful for an arbitrary cotangent. This file removes the "arbitrary": each
cotangent is pinned to the one the emitted backward chain delivers, so the whole train step is
`den`-composed forward → loss → backward with no free activation and no symbolic cotangent.

It is the batched peer of `ResNet34TiePoC.lean`, and four things about it are different in kind.

⭐⭐ **The block cotangents are NOT derived here.** `ResNet34ChainClose.lean` spells out per-block
cotangent vectors by hand, because no whole-block VJP existed when it was written. 4.1d's
`r34IdB_has_vjp_at` / `r34DownB_has_vjp_at` ARE the certified block backwards, so a block's input
cotangent is a `.backward` application — and `r34{BasicBlock,DownBlock}BackBatchedGraph_faithful`
already proves the emitted backward subgraph denotes exactly it. `r34IdCotIn_eq_vjp` and
`r34DownCotIn_eq_vjp` below are that statement in the vocabulary this file threads, and they are
what make the cross-block chain a composition of certified VJPs rather than a re-derivation.

⭐ **The loss cotangent is the LABEL-SMOOTHED one, at a general target.** `ResNet34RenderB` composes
the head cotangent from six kit ops — `softmaxRow → subB → scaleB → addVB → shiftB → divConstB`, α
baked at 0.1 and the `ls0` variants at 0 — and the target arrives as the graph input `%onehot`,
which under mixup or cutmix is a soft vector drawn on the host. `Foundation/SmoothedLossCot.lean`
is that cotangent's lemma; the head fold below is stated at it, not at `softmax − oneHot`.

⭐ **`N` is a binder.** The capstone takes `(N : Nat)`, exactly as `efficientnet_net_tied` does;
the artifacts at 32 (`resnet34_sgd/adam_train_step`) or 64 (`resnet34in_momdp64`) are instances.
T3 carries no numerals, so nothing here pins the batch.

⛔ **What the tie cannot reach: the all-reduce.** In `resnet34in_momdp64` each `*GradB` node is
followed by `all_reduce(add)/4` as emitted TEXT (`emitGradAllReduce`, a declared carve-out outside
the `SHlo` AST). Every statement below is at the PER-REPLICA gradient node; the mean across
replicas is section 4d's business.

## ⛔ The parameter census is 110, not the 146 `ResNet34TiePoC.lean` names

`resnet34TrainStepFaithfulV` and `ResNet34RenderB` both default to `convBias := false`: the conv
biases are gone from the signature (BatchNorm subsumes them, and He et al.'s `.convBn` has none),
bound instead to the zero constants `zeroBiasPrelude` emits. So `resnet34_sgd_train_step.mlir`
carries **110** SGD-updated tensors — stem 3 + 13 identity blocks × 6 + 3 downsample blocks × 9 +
dense 2 — and 146 is the census at `convBias := true`. (The per-example `resnet34_train_step.mlir`
carried the same 110 and was retired the same day by 4c leg 1.) The bias
conjuncts below are kept (they are one delegation each and they cover the flag), and they are about
ops the committed artifacts do not emit. Nothing about this weakens a theorem: every fold is
`∀`-quantified over op instances, and `bias = 0` is one of them.

## Conventions, stated because nothing here checks them

| | |
|---|---|
| BatchNorm | **batch** (`bnBatchLA`, reduce `[0,2,3]`, width `N·h·w`) |
| padding | **symmetric** at all seven stride-2 sites (`convStrided*`, not the XLA-`SAME` twins) |
| stem pool | 3×3/s2 (`maxPool3s2Flat`), not 2×2 |
| activation | relu, TWO kinks per block (the body's mid-relu and the post-residual outer one) |
| optimizer form | the RAW gradient (`*GradB`); the fused `θ − lr·g` appears in no batched r34 step |
| loss | label-smoothed softmax-CE at a general target, batch-meaned |
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet34TieB

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB bnBackB cInB gapInB)

-- ════════════════════════════════════════════════════════════════
-- § Chain-cotangent helpers new to ResNet-34
--   `reassocB` / `bnBackB` / `cInB` / `gapInB` are EfficientNet's and are reused verbatim;
--   what r34 adds is the relu mask, the strided conv input-VJP and the 3×3/s2 pool backward.
-- ════════════════════════════════════════════════════════════════

/-- **The relu backward mask** — `den (.selectPosB _ pre e) = fun i => if pre i > 0 then e i else 0`.
    r34 applies it twice per block (the body's mid-relu and the post-residual outer one) and once at
    the stem. -/
noncomputable def reluMaskB (n : Nat) (pre dy : Vec n) : Vec n :=
  fun i => if pre i > 0 then dy i else 0

/-- **Batched STRIDED conv input-VJP** (= `den convStridedBackBatched`; upsamples `h → 2h`). The
    strided peer of EfficientNet's `cInB`. ⚠ SYMMETRIC padding — `flatConvStride2`, not the
    XLA-`SAME` twin. -/
noncomputable def cStridedInB (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (dy : Vec (N * (oc * h * w))) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  batchMap N (fun d => (flatConvStride2_has_vjp W b).backward (fun _ => 0) d) dy

/-- **Batched true-BN input-cotangent, as the EMITTED backward computes it.** Written as the `den`
    of the backward op rather than as the certified VJP's `.backward`, because that is the form the
    render's chain is in and `den` ignores the name strings — so every cotangent below is literally
    what the artifact's bytes compute. ⭐ It takes no `β`: the BatchNorm input-gradient does not
    depend on the shift, which `bnInB_eq_bnBackB` records by holding for every `β`. -/
noncomputable def bnInB (N oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  den (SHlo.bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) "" "" "" ε γ x (.operand "" dy))

/-- **…and it IS the certified `bnBatchLA` VJP**, for every `β` and every `0 < ε`. This is
    `bnBatchLABack_faithful`, and it is the only step in this file's cotangent chain that is not
    `rfl` — everything else (the relu masks, the conv and strided-conv input-VJPs, the pool
    backward) denotes its certified backward definitionally. -/
theorem bnInB_eq_bnBackB (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * h * w))) :
    bnInB N oc h w ε γ x dy = bnBackB N oc h w ε hε γ β x dy :=
  bnBatchLABack_faithful "" "" "" ε γ β hε x (.operand "" dy)

/-- **Batched 3×3/s2 max-pool backward** (= `den maxPool3s2BackB`): the `select_and_scatter`
    denotation, per example on that example's own saved activation — which is why it is
    `batchMapAux` and not `batchMap`. -/
noncomputable def mpInB (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (dy : Vec (N * (c * h * w))) : Vec (N * (c * (2 * h) * (2 * w))) :=
  batchMapAux N (maxPool3s2BackFlat c h w) x dy

-- ════════════════════════════════════════════════════════════════
-- § The identity basic block — the render's cotangent chain, then the 8 parameter folds
-- ════════════════════════════════════════════════════════════════

/-! `ResNet34RenderB.idBackGradB`, node for node, from the block-output cotangent `dyOut`:

```
%da  = selectPosB(a)   %dn2 = bnBatchBack(g2, c2)   %dc2 = convBackBatched(W2)
%dr1 = selectPosB(n1)  %dn1 = bnBatchBack(g1, c1)   %dc1 = convBackBatched(W1)
%dx  = addVB(%dc1, %da)
```

and the eight parameter nodes read `W1,b1 ← %dn1`, `g1,bt1 ← %dr1`, `W2,b2 ← %dn2`,
`g2,bt2 ← %da`. The four defs below are those four cotangents. -/

/-- Cotangent at the block's pre-relu sum `a` — the outer relu's mask applied to `dyOut`. It feeds
    bn₂'s γ/β directly AND (through the identity skip) the block-input fan-in. -/
noncomputable def r34IdCotA (N h w : Nat) {c : Nat} (p : R34IdW c)
    (xin dyOut : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  reluMaskB (N * (c * h * w))
    (residual (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) xin) dyOut

/-- Cotangent at conv₂'s output — `r34IdCotA` through bn₂'s backward. Feeds `W₂`/`b₂`. -/
noncomputable def r34IdCotC2 (N h w : Nat) {c : Nat} (p : R34IdW c)
    (xin dyOut : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  bnInB N c h w p.ε₂ p.γ₂
    (batchMap N (flatConv p.W₂ p.b₂) (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin))
    (r34IdCotA N h w p xin dyOut)

/-- Cotangent at bn₁'s output — conv₂'s input-VJP masked by the body's mid-relu. Feeds `γ₁`/`β₁`. -/
noncomputable def r34IdCotN1 (N h w : Nat) {c : Nat} (p : R34IdW c)
    (xin dyOut : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  reluMaskB (N * (c * h * w))
    (bnBatchLA N c h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) xin))
    (cInB N p.W₂ p.b₂ (r34IdCotC2 N h w p xin dyOut))

/-- Cotangent at conv₁'s output — `r34IdCotN1` through bn₁'s backward. Feeds `W₁`/`b₁`. -/
noncomputable def r34IdCotC1 (N h w : Nat) {c : Nat} (p : R34IdW c)
    (xin dyOut : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  bnInB N c h w p.ε₁ p.γ₁ (batchMap N (flatConv p.W₁ p.b₁) xin)
    (r34IdCotN1 N h w p xin dyOut)

/-- **The block-INPUT cotangent**: the residual fan-in `addVB(convBack(dn1), da)` the render emits —
    the body branch plus the identity skip. -/
noncomputable def r34IdCotIn (N h w : Nat) {c : Nat} (p : R34IdW c)
    (xin dyOut : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  fun i => cInB N p.W₁ p.b₁ (r34IdCotC1 N h w p xin dyOut) i
    + r34IdCotA N h w p xin dyOut i

/-- ⭐⭐ **The emitted fan-in IS the certified block VJP's backward.** Not a re-derivation: the
    render's seven-node backward subgraph denotes `(r34IdB_has_vjp_at …).backward dyOut`, which is
    `r34BasicBlockBackBatchedGraph_faithful` read in this file's vocabulary. This is what makes the
    cross-block thread a composition of certified VJPs. -/
theorem r34IdCotIn_eq_vjp (N h w : Nat) {c : Nat} (p : R34IdW c) (hq : R34IdPos p)
    (xin dyOut : Vec (N * (c * h * w))) (hs : R34IdSmoothAt N h w p xin) (cotN : String) :
    r34IdCotIn N h w p xin dyOut
      = (r34IdB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := r34BasicBlockBackBatchedGraph_faithful (N := N) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ xin (.operand cotN dyOut) hs.hmid hs.hout
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [r34IdB_has_vjp_at, ← h]
  rfl


-- ════════════════════════════════════════════════════════════════
-- § The downsample basic block — the render's chain, then 12 parameter folds
-- ════════════════════════════════════════════════════════════════

/-! `ResNet34RenderB.downBackGradB` adds one branch to the identity block's chain:

```
%da  = selectPosB(a)   %dn2 = bnBatchBack(g2,c2)  %dc2 = convBackBatched(W2)
%dr1 = selectPosB(n1)  %dn1 = bnBatchBack(g1,c1)  %dc1 = convStridedBackBatched(W1)
%dnp = bnBatchBack(gp,cp)                          %dcp = convStridedBackBatched(Wp)
%dx  = addVB(%dc1, %dcp)
```

so `bnₚ`'s γ/β read the SAME `%da` that bn₂'s do — both feed the one `addVB` — and the projection's
`Wp`/`bp` read `%dnp`. -/

/-- The downsample block's pre-relu sum, as `r34DownB` composes it. -/
@[reducible] noncomputable def r34DownPre (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) :=
  residualProj (projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) xin

/-- Cotangent at the pre-relu sum. Feeds bn₂'s AND bnₚ's γ/β — they share the one `addVB`. -/
noncomputable def r34DownCotA (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w)) (r34DownPre N h w p xin) dyOut

/-- Cotangent at conv₂'s output. Feeds `W₂`/`b₂`. -/
noncomputable def r34DownCotC2 (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.ε₂ p.γ₂
    (batchMap N (flatConv p.W₂ p.b₂)
      (cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin))
    (r34DownCotA N h w p xin dyOut)

/-- Cotangent at bn₁'s output. Feeds `γ₁`/`β₁`. -/
noncomputable def r34DownCotN1 (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (bnBatchLA N oc h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConvStride2 p.W₁ p.b₁) xin))
    (cInB N p.W₂ p.b₂ (r34DownCotC2 N h w p xin dyOut))

/-- Cotangent at the STRIDED conv₁'s output. Feeds `W₁`/`b₁`. -/
noncomputable def r34DownCotC1 (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.ε₁ p.γ₁ (batchMap N (flatConvStride2 p.W₁ p.b₁) xin)
    (r34DownCotN1 N h w p xin dyOut)

/-- Cotangent at the PROJECTION conv's output. Feeds `Wp`/`bp`; its BN's γ/β read `r34DownCotA`. -/
noncomputable def r34DownCotCp (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.εp p.γp (batchMap N (flatConvStride2 p.Wp p.bp) xin)
    (r34DownCotA N h w p xin dyOut)

/-- **The block-INPUT cotangent**: the projected-residual fan-in `addVB(%dc1, %dcp)`. ⚠ Both
    operands are real backward subgraphs here — unlike the identity block, where the skip passes
    `%da` through verbatim. -/
noncomputable def r34DownCotIn (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (ic * (2 * h) * (2 * w))) :=
  fun i => cStridedInB N p.W₁ p.b₁ (r34DownCotC1 N h w p xin dyOut) i
    + cStridedInB N p.Wp p.bp (r34DownCotCp N h w p xin dyOut) i

/-- ⭐⭐ **The projected fan-in IS the certified downsample-block VJP's backward.** The identity
    block's `r34IdCotIn_eq_vjp` at the strided shape. ⚠ One `add_comm`: the render emits
    `addVB(body, projection)` and `r34DownBlockBackBatchedGraph` builds `addV(projection, body)`.
    Same vector, and the emitted order is the one this file threads. -/
theorem r34DownCotIn_eq_vjp (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc) (hq : R34DownPos p)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w)))
    (hs : R34DownSmoothAt N h w p xin) (cotN : String) :
    r34DownCotIn N h w p xin dyOut
      = (r34DownB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := r34DownBlockBackBatchedGraph_faithful (N := N) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.Wp p.bp p.εp hq.hp p.γp p.βp xin (.operand cotN dyOut)
    hs.hmid hs.hout
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [r34DownB_has_vjp_at, ← h]
  funext i
  exact add_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § The stem — the render's chain from the first block's input cotangent
-- ════════════════════════════════════════════════════════════════

/-! `ResNet34RenderB`, after the sixteen block backwards:

```
%dmp = maxPool3s2BackB(str)   %dsr = selectPosB(stn)   %dsn = bnBatchBack(sg, stc)
```

with `sW,sb ← %dsn` and `sg,sbt ← %dsr`. -/

/-- Cotangent at the stem's post-relu, pre-pool activation — the 3×3/s2 pool's backward applied to
    the cotangent block 1 delivers at its input. -/
noncomputable def r34StemCotP (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (cotPool : Vec (N * (oc * h * w))) : Vec (N * (oc * (2 * h) * (2 * w))) :=
  mpInB N oc h w (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x) cotPool

/-- Cotangent at the stem BN's output — the stem relu's mask. Feeds `γs`/`βs`. -/
noncomputable def r34StemCotN (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (cotPool : Vec (N * (oc * h * w))) : Vec (N * (oc * (2 * h) * (2 * w))) :=
  reluMaskB (N * (oc * (2 * h) * (2 * w)))
    (bnBatchLA N oc (2 * h) (2 * w) εs γs βs (batchMap N (flatConvStride2 Ws bs) x))
    (r34StemCotP N h w Ws bs εs γs βs x cotPool)

/-- Cotangent at the stem conv's output. Feeds `sW`/`sb`. -/
noncomputable def r34StemCotC (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (cotPool : Vec (N * (oc * h * w))) : Vec (N * (oc * (2 * h) * (2 * w))) :=
  bnInB N oc (2 * h) (2 * w) εs γs (batchMap N (flatConvStride2 Ws bs) x)
    (r34StemCotN N h w Ws bs εs γs βs x cotPool)


-- ════════════════════════════════════════════════════════════════
-- § The per-block-type tie bundles — every parameter node at its chain cotangent
-- ════════════════════════════════════════════════════════════════

/-! Each conjunct is `ResNet34FaithfulPoCB`'s `∀ cot` fold instantiated at the cotangent the
render's chain delivers, so nothing here is a new proof: the bundles are the §1 fold with the
freedom removed. `reassocB` bridges the conv/relu index `N·(c·h·w)` to the BatchNorm parameter
ops' `N·(c·(h·w))`.

⚠ The two conv-BIAS conjuncts in each block are about `conv{,Strided}BiasGradB`, which the
committed artifacts do NOT emit — both renders run `convBias := false` and bind the bias operand to
`zeroBiasPrelude`'s zero constant. They are kept because they cost one delegation each and they
cover the flag. -/

/-- **Identity basic block, tied.** All eight parameter nodes — conv₁/conv₂ weight and bias, bn₁/bn₂
    γ and β — denote the certified batched `Σ_n` gradient at the real forward activations and the
    real backward-chain cotangent driven by `dyOut`. -/
def r34IdTiedB (N h w : Nat) {c : Nat} (xN cotN vN epsStr : String) (p : R34IdW c)
    (xin dyOut : Vec (N * (c * h * w))) : Prop :=
  let r1 := cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin
  let c1 := batchMap N (flatConv p.W₁ p.b₁) xin
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let cotA := r34IdCotA N h w p xin dyOut
  let cotC2 := r34IdCotC2 N h w p xin dyOut
  let cotN1 := r34IdCotN1 N h w p xin dyOut
  let cotC1 := r34IdCotC1 N h w p xin dyOut
  -- conv₁ (stride-1, c → c), cot = cotC1
  (∀ idx : Fin (c * c * 3 * 3),
      den (SHlo.convWeightGradB xN p.b₁ xin p.W₁ (.operand cotN cotC1)) idx
        = ∑ n : Fin N, ∑ j : Fin (c * h * w),
            pdiv (fun v' : Vec (c * c * 3 * 3) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₁
                      (Tensor3.unflatten (batchSlice N (c * h * w) xin n))))
                 (Kernel4.flatten p.W₁) idx j * batchSlice N (c * h * w) cotC1 n j)
  ∧ (∀ o : Fin c,
      den (SHlo.convBiasGradB (h := h) (w := w) p.W₁ xin p.b₁ (.operand cotN cotC1)) o
        = ∑ n : Fin N, ∑ j : Fin (c * h * w),
            pdiv (fun b' : Vec c =>
                    Tensor3.flatten (conv2d p.W₁ b'
                      (Tensor3.unflatten (batchSlice N (c * h * w) xin n))))
                 p.b₁ o j * batchSlice N (c * h * w) cotC1 n j)
  -- bn₁ γ/β, cot = cotN1
  ∧ (∀ k : Fin c,
      den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N c h w c1)
            (.operand cotN (reassocB N c h w cotN1))) k
        = ∑ j : Fin (c * (N * (h * w))),
            pdiv (fun γ' : Vec c =>
                    bnPerChannelFlat c (N * (h * w)) p.ε₁ γ' p.β₁
                      (bnchwFwd N c h w (reassocB N c h w c1)))
                 p.γ₁ k j * bnchwFwd N c h w (reassocB N c h w cotN1) j)
  ∧ (∀ k : Fin c,
      den (SHlo.bnBetaGradB (N := N) (oc := c) (h := h) (w := w)
            (.operand cotN (reassocB N c h w cotN1))) k
        = ∑ j : Fin (c * (N * (h * w))),
            pdiv (fun β' : Vec c =>
                    bnPerChannelFlat c (N * (h * w)) p.ε₁ p.γ₁ β'
                      (bnchwFwd N c h w (reassocB N c h w c1)))
                 p.β₁ k j * bnchwFwd N c h w (reassocB N c h w cotN1) j)
  -- conv₂ (stride-1, c → c), cot = cotC2
  ∧ (∀ idx : Fin (c * c * 3 * 3),
      den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN cotC2)) idx
        = ∑ n : Fin N, ∑ j : Fin (c * h * w),
            pdiv (fun v' : Vec (c * c * 3 * 3) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₂
                      (Tensor3.unflatten (batchSlice N (c * h * w) r1 n))))
                 (Kernel4.flatten p.W₂) idx j * batchSlice N (c * h * w) cotC2 n j)
  ∧ (∀ o : Fin c,
      den (SHlo.convBiasGradB (h := h) (w := w) p.W₂ r1 p.b₂ (.operand cotN cotC2)) o
        = ∑ n : Fin N, ∑ j : Fin (c * h * w),
            pdiv (fun b' : Vec c =>
                    Tensor3.flatten (conv2d p.W₂ b'
                      (Tensor3.unflatten (batchSlice N (c * h * w) r1 n))))
                 p.b₂ o j * batchSlice N (c * h * w) cotC2 n j)
  -- bn₂ γ/β, cot = cotA (the outer relu's mask — the same node the identity skip carries)
  ∧ (∀ k : Fin c,
      den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N c h w c2)
            (.operand cotN (reassocB N c h w cotA))) k
        = ∑ j : Fin (c * (N * (h * w))),
            pdiv (fun γ' : Vec c =>
                    bnPerChannelFlat c (N * (h * w)) p.ε₂ γ' p.β₂
                      (bnchwFwd N c h w (reassocB N c h w c2)))
                 p.γ₂ k j * bnchwFwd N c h w (reassocB N c h w cotA) j)
  ∧ (∀ k : Fin c,
      den (SHlo.bnBetaGradB (N := N) (oc := c) (h := h) (w := w)
            (.operand cotN (reassocB N c h w cotA))) k
        = ∑ j : Fin (c * (N * (h * w))),
            pdiv (fun β' : Vec c =>
                    bnPerChannelFlat c (N * (h * w)) p.ε₂ p.γ₂ β'
                      (bnchwFwd N c h w (reassocB N c h w c2)))
                 p.β₂ k j * bnchwFwd N c h w (reassocB N c h w cotA) j)

theorem r34_idblock_tiedB (N h w : Nat) {c : Nat} (xN cotN vN epsStr : String) (p : R34IdW c)
    (xin dyOut : Vec (N * (c * h * w))) :
    r34IdTiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold r34IdTiedB
  intro r1 c1 c2 cotA cotC2 cotN1 cotC1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₁ xin p.W₁ cotC1 idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.W₁ xin p.b₁ cotC1 o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₁ p.γ₁ p.β₁
                 (reassocB N c h w c1) (reassocB N c h w cotN1) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₁ p.γ₁ p.β₁
                 (bnchwFwd N c h w (reassocB N c h w c1)) (reassocB N c h w cotN1) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₂ r1 p.W₂ cotC2 idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.W₂ r1 p.b₂ cotC2 o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₂ p.γ₂ p.β₂
                 (reassocB N c h w c2) (reassocB N c h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₂ p.γ₂ p.β₂
                 (bnchwFwd N c h w (reassocB N c h w c2)) (reassocB N c h w cotA) k


/-- **Downsample basic block, tied.** All twelve parameter nodes — the STRIDED conv₁, the stride-1
    conv₂ and the 1×1/s2 option-B projection, each with bias and BatchNorm γ/β. ⚠ Both stride-2
    sites are SYMMETRIC padding (`convStrided*GradB`, whose `den` is `flatConvStride2_*`), which is
    ResNet's convention and NOT B0's or MobileNetV2's XLA-`SAME`. -/
def r34DownTiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String) (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ xin
  let c1 := batchMap N (flatConvStride2 p.W₁ p.b₁) xin
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let cp := batchMap N (flatConvStride2 p.Wp p.bp) xin
  let cotA := r34DownCotA N h w p xin dyOut
  let cotC2 := r34DownCotC2 N h w p xin dyOut
  let cotN1 := r34DownCotN1 N h w p xin dyOut
  let cotC1 := r34DownCotC1 N h w p xin dyOut
  let cotCp := r34DownCotCp N h w p xin dyOut
  -- STRIDED conv₁ (ic → oc), cot = cotC1
  (∀ idx : Fin (oc * ic * 3 * 3),
      den (SHlo.convStridedWeightGradB xN p.b₁ xin p.W₁ (.operand cotN cotC1)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * 3 * 3) =>
                    flatConvStride2 (Kernel4.unflatten v') p.b₁
                      (batchSlice N (ic * (2 * h) * (2 * w)) xin n))
                 (Kernel4.flatten p.W₁) idx j * batchSlice N (oc * h * w) cotC1 n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convStridedBiasGradB (h := h) (w := w) p.W₁ xin p.b₁ (.operand cotN cotC1)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    flatConvStride2 p.W₁ b' (batchSlice N (ic * (2 * h) * (2 * w)) xin n))
                 p.b₁ o j * batchSlice N (oc * h * w) cotC1 n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N oc h w c1)
            (.operand cotN (reassocB N oc h w cotN1))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.ε₁ γ' p.β₁
                      (bnchwFwd N oc h w (reassocB N oc h w c1)))
                 p.γ₁ k j * bnchwFwd N oc h w (reassocB N oc h w cotN1) j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w cotN1))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.ε₁ p.γ₁ β'
                      (bnchwFwd N oc h w (reassocB N oc h w c1)))
                 p.β₁ k j * bnchwFwd N oc h w (reassocB N oc h w cotN1) j)
  -- conv₂ (stride-1, oc → oc), cot = cotC2
  ∧ (∀ idx : Fin (oc * oc * 3 * 3),
      den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN cotC2)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * oc * 3 * 3) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.b₂
                      (Tensor3.unflatten (batchSlice N (oc * h * w) r1 n))))
                 (Kernel4.flatten p.W₂) idx j * batchSlice N (oc * h * w) cotC2 n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convBiasGradB (h := h) (w := w) p.W₂ r1 p.b₂ (.operand cotN cotC2)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    Tensor3.flatten (conv2d p.W₂ b'
                      (Tensor3.unflatten (batchSlice N (oc * h * w) r1 n))))
                 p.b₂ o j * batchSlice N (oc * h * w) cotC2 n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N oc h w c2)
            (.operand cotN (reassocB N oc h w cotA))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.ε₂ γ' p.β₂
                      (bnchwFwd N oc h w (reassocB N oc h w c2)))
                 p.γ₂ k j * bnchwFwd N oc h w (reassocB N oc h w cotA) j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w cotA))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.ε₂ p.γ₂ β'
                      (bnchwFwd N oc h w (reassocB N oc h w c2)))
                 p.β₂ k j * bnchwFwd N oc h w (reassocB N oc h w cotA) j)
  -- 1×1/s2 projection (ic → oc), cot = cotCp; its BN reads the SHARED cotA
  ∧ (∀ idx : Fin (oc * ic * 1 * 1),
      den (SHlo.convStridedWeightGradB xN p.bp xin p.Wp (.operand cotN cotCp)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                    flatConvStride2 (Kernel4.unflatten v') p.bp
                      (batchSlice N (ic * (2 * h) * (2 * w)) xin n))
                 (Kernel4.flatten p.Wp) idx j * batchSlice N (oc * h * w) cotCp n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convStridedBiasGradB (h := h) (w := w) p.Wp xin p.bp (.operand cotN cotCp)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    flatConvStride2 p.Wp b' (batchSlice N (ic * (2 * h) * (2 * w)) xin n))
                 p.bp o j * batchSlice N (oc * h * w) cotCp n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr p.εp (reassocB N oc h w cp)
            (.operand cotN (reassocB N oc h w cotA))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.εp γ' p.βp
                      (bnchwFwd N oc h w (reassocB N oc h w cp)))
                 p.γp k j * bnchwFwd N oc h w (reassocB N oc h w cotA) j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w cotA))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.εp p.γp β'
                      (bnchwFwd N oc h w (reassocB N oc h w cp)))
                 p.βp k j * bnchwFwd N oc h w (reassocB N oc h w cotA) j)

theorem r34_downblock_tiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String)
    (p : R34DownW ic oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyOut : Vec (N * (oc * h * w))) :
    r34DownTiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold r34DownTiedB
  intro r1 c1 c2 cp cotA cotC2 cotN1 cotC1 cotCp
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convStridedWGradB_den xN cotN p.b₁ xin p.W₁ cotC1 idx
  · intro o;   exact ResNet34PoCB.convStridedBGradB_den cotN p.W₁ xin p.b₁ cotC1 o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₁ p.γ₁ p.β₁
                 (reassocB N oc h w c1) (reassocB N oc h w cotN1) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₁ p.γ₁ p.β₁
                 (bnchwFwd N oc h w (reassocB N oc h w c1)) (reassocB N oc h w cotN1) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₂ r1 p.W₂ cotC2 idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.W₂ r1 p.b₂ cotC2 o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ε₂ p.γ₂ p.β₂
                 (reassocB N oc h w c2) (reassocB N oc h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.ε₂ p.γ₂ p.β₂
                 (bnchwFwd N oc h w (reassocB N oc h w c2)) (reassocB N oc h w cotA) k
  · intro idx; exact ResNet34PoCB.convStridedWGradB_den xN cotN p.bp xin p.Wp cotCp idx
  · intro o;   exact ResNet34PoCB.convStridedBGradB_den cotN p.Wp xin p.bp cotCp o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.εp p.γp p.βp
                 (reassocB N oc h w cp) (reassocB N oc h w cotA) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.εp p.γp p.βp
                 (bnchwFwd N oc h w (reassocB N oc h w cp)) (reassocB N oc h w cotA) k

/-- **Stem, tied.** The 7×7/s2 conv's weight and bias and its BatchNorm's γ/β, at the cotangent
    that reaches the stem through block 1's input fan-in and the 3×3/s2 pool's backward. -/
def r34StemTiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (cotPool : Vec (N * (oc * h * w))) : Prop :=
  let sc := batchMap N (flatConvStride2 Ws bs) x
  let cotN' := r34StemCotN N h w Ws bs εs γs βs x cotPool
  let cotC := r34StemCotC N h w Ws bs εs γs βs x cotPool
  (∀ idx : Fin (oc * ic * 7 * 7),
      den (SHlo.convStridedWeightGradB xN bs x Ws (.operand cotN cotC)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * (2 * h) * (2 * w)),
            pdiv (fun v' : Vec (oc * ic * 7 * 7) =>
                    flatConvStride2 (Kernel4.unflatten v') bs
                      (batchSlice N (ic * (2 * (2 * h)) * (2 * (2 * w))) x n))
                 (Kernel4.flatten Ws) idx j * batchSlice N (oc * (2 * h) * (2 * w)) cotC n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convStridedBiasGradB (h := 2 * h) (w := 2 * w) Ws x bs (.operand cotN cotC)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * (2 * h) * (2 * w)),
            pdiv (fun b' : Vec oc =>
                    flatConvStride2 Ws b'
                      (batchSlice N (ic * (2 * (2 * h)) * (2 * (2 * w))) x n))
                 bs o j * batchSlice N (oc * (2 * h) * (2 * w)) cotC n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc (2 * h) (2 * w) sc)
            (.operand cotN (reassocB N oc (2 * h) (2 * w) cotN'))) k
        = ∑ j : Fin (oc * (N * ((2 * h) * (2 * w)))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * ((2 * h) * (2 * w))) εs γ' βs
                      (bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w) sc)))
                 γs k j * bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w) cotN') j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := 2 * h) (w := 2 * w)
            (.operand cotN (reassocB N oc (2 * h) (2 * w) cotN'))) k
        = ∑ j : Fin (oc * (N * ((2 * h) * (2 * w)))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * ((2 * h) * (2 * w))) εs γs β'
                      (bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w) sc)))
                 βs k j * bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w) cotN') j)

theorem r34_stem_tiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (cotPool : Vec (N * (oc * h * w))) :
    r34StemTiedB N h w xN cotN vN epsStr Ws bs εs γs βs x cotPool := by
  unfold r34StemTiedB
  intro sc cotN' cotC
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convStridedWGradB_den xN cotN bs x Ws cotC idx
  · intro o;   exact ResNet34PoCB.convStridedBGradB_den cotN Ws x bs cotC o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN εs γs βs
                 (reassocB N oc (2 * h) (2 * w) sc) (reassocB N oc (2 * h) (2 * w) cotN') k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN εs γs βs
                 (bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w) sc))
                 (reassocB N oc (2 * h) (2 * w) cotN') k


-- ════════════════════════════════════════════════════════════════
-- § The head — the smoothed loss cotangent, the certified head backward, the two dense folds
-- ════════════════════════════════════════════════════════════════

/-- `Vec (N·(1·K)) → Vec (N·K)`: the loss chain runs at one ROW per example (`softmaxRow` needs a
    row index) and the dense parameter ops at the plain per-example width. The render writes one
    SSA name for both, because `1 * K = K` as an emitted shape; in Lean the two indices are
    propositionally but not definitionally equal, so the cast is explicit. -/
noncomputable def unrowB (N K : Nat) (v : Vec (N * (1 * K))) : Vec (N * K) :=
  fun i => v (Fin.cast (congrArg (N * ·) (Nat.one_mul K)).symm i)

/-- **The head's block-side cotangent**, as the CERTIFIED head backward delivers it. The head is
    `batchMap(dense) ∘ batchMap(GAP)` — both smooth, both `batchMap` of a per-example op — so
    `r34HeadB_has_vjp` is global and this needs no smoothness hypothesis, which is the one place in
    the whole net where that is true. -/
noncomputable def r34HeadCotBlk (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (dy : Vec (N * nCls)) : Vec (N * (c * h * w)) :=
  (r34HeadB_has_vjp N h w Wd bd).backward xin dy

/-- **Head, tied.** The classifier's weight and bias nodes denote the certified batched `Σ_n`
    gradient at the real GAP output and the loss cotangent. ⚠ The bias conjunct's Jacobian witness
    carries a zero activation rather than the real one: `dense`'s derivative in `b` is the identity
    whatever `x` is, so the statement is `x`-free and there is no per-example choice to make (the
    same shape `EfficientNetTiePoC`'s bias conjuncts take). -/
def r34HeadTiedB (N h w : Nat) {c nCls : Nat} (xN cotN : String) (Wd : Mat c nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (dy : Vec (N * nCls)) : Prop :=
  let a := batchMap N (globalAvgPoolFlat c h w) xin
  (∀ (i : Fin c) (j : Fin nCls),
      den (SHlo.denseWeightGradB (c := nCls) xN a (.operand cotN dy)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ k : Fin nCls,
            pdiv (fun v : Vec (c * nCls) => dense (Mat.unflatten v) bd (batchSlice N c a n))
                 (Mat.flatten Wd) (finProdFinEquiv (i, j)) k * batchSlice N nCls dy n k)
  ∧ (∀ j : Fin nCls,
      den (SHlo.denseBiasGradB (N := N) (.operand cotN dy)) j
        = ∑ n : Fin N, ∑ k : Fin nCls,
            pdiv (fun b' : Vec nCls => dense Wd b' (fun _ => 0)) bd j k
              * batchSlice N nCls dy n k)

theorem r34_head_tiedB (N h w : Nat) {c nCls : Nat} (xN cotN : String) (Wd : Mat c nCls)
    (bd : Vec nCls) (xin : Vec (N * (c * h * w))) (dy : Vec (N * nCls)) :
    r34HeadTiedB N h w xN cotN Wd bd xin dy := by
  unfold r34HeadTiedB
  intro a
  refine ⟨?_, ?_⟩
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN a Wd bd dy i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN Wd (fun _ => 0) bd dy j


/-- The inverse cast of `unrowB`: the head's logits, at the one-row-per-example index the loss
    chain's `softmaxRow` consumes. -/
noncomputable def rowB (N K : Nat) (v : Vec (N * K)) : Vec (N * (1 * K)) :=
  fun i => v (Fin.cast (congrArg (N * ·) (Nat.one_mul K)) i)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net capstone
-- ════════════════════════════════════════════════════════════════

set_option maxHeartbeats 1600000 in
/-- ⭐⭐ **The whole batch-BN ResNet-34 train step, tied.** Threading `resnet34ForwardB_full`'s own
    prefixes as the block inputs and the label-smoothed loss cotangent down through the certified
    head backward and the sixteen certified block backwards, every parameter GRADIENT node of the
    net — stem 4, thirteen identity blocks × 8, three downsample blocks × 12, dense 2 — denotes the
    certified batched `Σ_n` gradient. No free activation and no symbolic cotangent.

    ⭐ **`N` is a binder and there is no smoothness hypothesis.** The folds are `∀ cot` statements
    instantiated at explicitly-constructed cotangents, so the capstone needs neither `0 < ε` nor a
    relu-kink condition. Those enter only in `r34IdCotIn_eq_vjp` / `r34DownCotIn_eq_vjp`, which say
    the constructed chain IS the certified whole-net backward — the two halves of the tie, kept
    apart because they have different hypotheses.

    ⚠ Of the 146 conjunct slots, the committed artifacts exercise **110**: both r34 renders run
    `convBias := false`, so the 36 conv-bias nodes are not emitted (the biases are
    `zeroBiasPrelude`'s zero constants).

    ⛔ One replica. In `resnet34in_momdp64` every gradient node is followed by
    `all_reduce(add)/4` as emitted text outside the AST. -/
theorem r34_net_tiedB (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (t : Vec (N * (1 * nCls))) :
    -- the label-smoothed loss cotangent at the real logits and a general target
    let g : Vec (N * nCls) :=
      unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (resnet34ForwardB_full N w x)) t))
    -- the backward chain: the certified head backward, then the certified block backwards
    let dyE1 := r34HeadCotBlk N 7 7 w.Wd w.bd (r34Pre16 N w x) g
    let dyE0 := r34IdCotIn N 7 7 w.e1 (r34Pre15 N w x) dyE1
    let dyD4 := r34IdCotIn N 7 7 w.e0 (r34Pre14 N w x) dyE0
    let dyC4 := r34DownCotIn N 7 7 w.d4 (r34Pre13 N w x) dyD4
    let dyC3 := r34IdCotIn N 14 14 w.c4 (r34Pre12 N w x) dyC4
    let dyC2 := r34IdCotIn N 14 14 w.c3 (r34Pre11 N w x) dyC3
    let dyC1 := r34IdCotIn N 14 14 w.c2 (r34Pre10 N w x) dyC2
    let dyC0 := r34IdCotIn N 14 14 w.c1 (r34Pre9 N w x) dyC1
    let dyD3 := r34IdCotIn N 14 14 w.c0 (r34Pre8 N w x) dyC0
    let dyB2 := r34DownCotIn N 14 14 w.d3 (r34Pre7 N w x) dyD3
    let dyB1 := r34IdCotIn N 28 28 w.b2 (r34Pre6 N w x) dyB2
    let dyB0 := r34IdCotIn N 28 28 w.b1 (r34Pre5 N w x) dyB1
    let dyD2 := r34IdCotIn N 28 28 w.b0 (r34Pre4 N w x) dyB0
    let dyA2 := r34DownCotIn N 28 28 w.d2 (r34Pre3 N w x) dyD2
    let dyA1 := r34IdCotIn N 56 56 w.a2 (r34Pre2 N w x) dyA2
    let dyA0 := r34IdCotIn N 56 56 w.a1 (r34Pre1 N w x) dyA1
    let cotPool := r34IdCotIn N 56 56 w.a0 (r34Pre0 N w x) dyA0
    r34StemTiedB N 56 56 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotPool
  ∧ r34IdTiedB N 56 56 xN cotN vN epsStr w.a0 (r34Pre0 N w x) dyA0
  ∧ r34IdTiedB N 56 56 xN cotN vN epsStr w.a1 (r34Pre1 N w x) dyA1
  ∧ r34IdTiedB N 56 56 xN cotN vN epsStr w.a2 (r34Pre2 N w x) dyA2
  ∧ r34DownTiedB N 28 28 xN cotN vN epsStr w.d2 (r34Pre3 N w x) dyD2
  ∧ r34IdTiedB N 28 28 xN cotN vN epsStr w.b0 (r34Pre4 N w x) dyB0
  ∧ r34IdTiedB N 28 28 xN cotN vN epsStr w.b1 (r34Pre5 N w x) dyB1
  ∧ r34IdTiedB N 28 28 xN cotN vN epsStr w.b2 (r34Pre6 N w x) dyB2
  ∧ r34DownTiedB N 14 14 xN cotN vN epsStr w.d3 (r34Pre7 N w x) dyD3
  ∧ r34IdTiedB N 14 14 xN cotN vN epsStr w.c0 (r34Pre8 N w x) dyC0
  ∧ r34IdTiedB N 14 14 xN cotN vN epsStr w.c1 (r34Pre9 N w x) dyC1
  ∧ r34IdTiedB N 14 14 xN cotN vN epsStr w.c2 (r34Pre10 N w x) dyC2
  ∧ r34IdTiedB N 14 14 xN cotN vN epsStr w.c3 (r34Pre11 N w x) dyC3
  ∧ r34IdTiedB N 14 14 xN cotN vN epsStr w.c4 (r34Pre12 N w x) dyC4
  ∧ r34DownTiedB N 7 7 xN cotN vN epsStr w.d4 (r34Pre13 N w x) dyD4
  ∧ r34IdTiedB N 7 7 xN cotN vN epsStr w.e0 (r34Pre14 N w x) dyE0
  ∧ r34IdTiedB N 7 7 xN cotN vN epsStr w.e1 (r34Pre15 N w x) dyE1
  ∧ r34HeadTiedB N 7 7 xN cotN w.Wd w.bd (r34Pre16 N w x) g := by
  intro g dyE1 dyE0 dyD4 dyC4 dyC3 dyC2 dyC1 dyC0 dyD3 dyB2 dyB1 dyB0 dyD2 dyA2 dyA1 dyA0 cotPool
  exact ⟨r34_stem_tiedB N 56 56 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotPool,
    r34_idblock_tiedB N 56 56 xN cotN vN epsStr w.a0 (r34Pre0 N w x) dyA0,
    r34_idblock_tiedB N 56 56 xN cotN vN epsStr w.a1 (r34Pre1 N w x) dyA1,
    r34_idblock_tiedB N 56 56 xN cotN vN epsStr w.a2 (r34Pre2 N w x) dyA2,
    r34_downblock_tiedB N 28 28 xN cotN vN epsStr w.d2 (r34Pre3 N w x) dyD2,
    r34_idblock_tiedB N 28 28 xN cotN vN epsStr w.b0 (r34Pre4 N w x) dyB0,
    r34_idblock_tiedB N 28 28 xN cotN vN epsStr w.b1 (r34Pre5 N w x) dyB1,
    r34_idblock_tiedB N 28 28 xN cotN vN epsStr w.b2 (r34Pre6 N w x) dyB2,
    r34_downblock_tiedB N 14 14 xN cotN vN epsStr w.d3 (r34Pre7 N w x) dyD3,
    r34_idblock_tiedB N 14 14 xN cotN vN epsStr w.c0 (r34Pre8 N w x) dyC0,
    r34_idblock_tiedB N 14 14 xN cotN vN epsStr w.c1 (r34Pre9 N w x) dyC1,
    r34_idblock_tiedB N 14 14 xN cotN vN epsStr w.c2 (r34Pre10 N w x) dyC2,
    r34_idblock_tiedB N 14 14 xN cotN vN epsStr w.c3 (r34Pre11 N w x) dyC3,
    r34_idblock_tiedB N 14 14 xN cotN vN epsStr w.c4 (r34Pre12 N w x) dyC4,
    r34_downblock_tiedB N 7 7 xN cotN vN epsStr w.d4 (r34Pre13 N w x) dyD4,
    r34_idblock_tiedB N 7 7 xN cotN vN epsStr w.e0 (r34Pre14 N w x) dyE0,
    r34_idblock_tiedB N 7 7 xN cotN vN epsStr w.e1 (r34Pre15 N w x) dyE1,
    r34_head_tiedB N 7 7 xN cotN w.Wd w.bd (r34Pre16 N w x) g⟩


/-- ⭐ **And the cotangent the capstone threads is the smoothed loss's gradient.** Row by row: the
    `g` above is, at example `n` and class `j`, `(1/B)·∂/∂logits` of soft-target cross-entropy
    against the SMOOTHED target `(1−α)·t + α/K`, at that example's real logits. The only hypothesis
    is that the example's target sums to 1 — a one-hot, or mixup's convex combination of two.
    Together with the capstone this closes the top of the chain: every parameter node denotes the
    certified gradient at the cotangent of the loss the trainer actually minimises. -/
theorem r34_lossCot_is_smoothedCE_grad (N : Nat) {nCls : Nat} (hK : 0 < nCls)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (t : Vec (N * (1 * nCls)))
    (n : Fin N) (j : Fin nCls)
    (ht : ∑ k : Fin nCls, Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1) k = 1) :
    den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
          (rowB N nCls (resnet34ForwardB_full N w x)) t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec nCls => fun _ : Fin 1 =>
            softCE nCls (smoothTarget nCls α
              (Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1))) z')
          (Mat.unflatten (batchSlice N (1 * nCls)
            (rowB N nCls (resnet34ForwardB_full N w x)) n) (0 : Fin 1)) j 0) / B :=
  smoothedLossCotGraph_row N nCls hK α B aStr negAK bStr logN ohN _ t n j ht

end Proofs.ResNet34TieB
