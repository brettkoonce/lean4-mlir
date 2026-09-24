import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTieG
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackNet
import LeanMlir.Proofs.Foundation.DataParallelSyncKit
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncB

/-! # EfficientNet-B0's data-parallel step at SYNCHRONISED BatchNorm IS the single-device step at `R·N`

`EfficientNetStepTieG.lean` (T3) threads the label-smoothed loss cotangent down the batch-BN
backward chain on ONE device and ties every parameter gradient node to the certified gradient.
This is its data-parallel twin, for the render `EfficientNetRender` emits at `replicas > 1`: `R`
replicas at batch `N`, every one of the 49 BatchNorms synchronised (`bnFwdSite` / `bnBackSite` /
`bnGammaSite`), every parameter gradient all-reduced by its mean. The capstone
`efficientnet_net_syncTiedG` says that, for every parameter,

    mean over the R replicas of replica r's gradient node, loss divided by B
      = the single-device gradient node at the global batch R·N, loss divided by R·B

— the gradient node `EnetTiePoCG.efficientnet_net_tiedG` at `N := R·N` ties to the certified
gradient. The spec it is stated against has not moved: the right-hand side is T3's chain at
`N := R·N` — its forward prefixes, loss cotangent and block `.backward`s verbatim, and its
in-block cotangents as named definitions (`xCotEc`, `tCotDn`, …) that unfold to `enetExpTiedG`'s
`let`s, so each right-hand node is T3's node by `rfl`.

## Five steps

ResNet-34's four (`ResNet34SyncStepTieB.lean`), and one B0 needs because of how its T3 is written.

0. **The block VJP is the chain** (§ 0). T3 threads each block's input cotangent as a certified
   VJP's `.backward`; a replica computes its own by the explicit chain, and only the explicit
   chain can be sharded. `xCotIn_eq_vjp` and its four peers say the two agree — the
   `BatchedBackLinks` stage graphs, read at an `.operand` leaf, with `bnBatchLABack_faithful`
   turning the one non-`rfl` link into `bnBackB`.
1. **Sharding** (§§ 2, 5). Every non-BN link is per-example — conv, depthwise and strided
   depthwise input-VJPs, swish and sigmoid masks, and the squeeze-excite backward: the gate
   cotangent `gateCotB` (`seReduceB`'s `den`) and the fused input-VJP `seInB` (`seBackBatched`'s)
   both read one example at a time, so they shard like ResNet-34's pool backward. The BN link is
   `bnSyncInB`, P2 on the graph, read through `bnInB_eq_bnBackB` onto T3's `bnBackB`.
2. **The collectives.** ResNet-34's conv, dense and BatchNorm ones, plus the depthwise,
   strided-depthwise and XLA-`SAME` stem conv weights from `DataParallelSyncKit`, which also
   holds the depthwise, GAP and dense links of steps 1 and 3 that MobileNetV2 shares (§ 3 moved
   there).
3. **Homogeneity** (§§ 1, 4). Every link is linear in its cotangent; most are a certified VJP's
   `.backward`, so `HasVJP.backward_smul` is the whole proof.
4. **The divisor.** Replica `r`'s loss cotangent is `R ×` its shard of the global one
   (`replicaLossCot_eq`), and steps 1–3 carry that `R` down to cancel each collective's `1/R`.

## What is covered, and what is NOT claimed

The DP render runs `convBias := false`, so it emits **213** parameter collectives — stem 3, b1 10,
fifteen MBConv6 blocks × 13, head 5 — and all 213 are tied. T3 carries 49 more conjuncts, one per
conv bias (a `bnBetaGradB` at the conv-output cotangent), for the `convBias := true` census; the
DP artifacts do not emit them and they are not tied here.

⚠ The replicas' saved forward activations enter as the shards of the single-device forward's;
that the sync forward graph computes exactly those is `EfficientNetSyncB`'s
`efficientnetFwdGraphSync_full_shard`, the forward half. ⚠ That the replicas' inputs are the
shards of one batch is the driver's. ⚠ T3 states the chain without stochastic depth or classifier
dropout, so this does too: the `drop` / `dropdo` DP variants add a `dropPathB` on each residual
branch and a `dropoutB` before the classifier — per-example diagonal scalings, which shard and are
linear, but whose chain neither tier states. The `bf16` DP variants emit different gradient nodes
and are not covered. ⚠ The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.EnetSyncTieG

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB bnBackB swBackB sigBackB cInB dInB dStridedInB gapInB seInB
  gateCotB)
open Proofs.ResNet34TieB (bnInB bnInB_eq_bnBackB rowB unrowB)
open Proofs.ResNet34SyncTieB
open Proofs.MBConvSyncTieB

-- ════════════════════════════════════════════════════════════════
-- § 0. The single-device chain, named — and each block's input cotangent IS its VJP
-- ════════════════════════════════════════════════════════════════

/-- **The tail every MBConv block shares** — depthwise BatchNorm → swish → squeeze-excite →
    1×1 project → project BatchNorm. `MBW` and `MBWNoExp` both carry it; naming it once lets the
    three block kinds share one tail chain. -/
structure EnTail (mid oc rd : Nat) where
  dε : ℝ
  dγ : Vec mid
  dβ : Vec mid
  z1 : Mat mid rd
  zb1 : Vec rd
  z2 : Mat rd mid
  zb2 : Vec mid
  pW : Kernel4 oc mid 1 1
  pb : Vec oc
  pε : ℝ
  pγ : Vec oc
  pβ : Vec oc

/-- The tail of an MBConv6 block's weights. Reducible, so `(tailOf p).dε` IS `p.dε` to every
    tactic — the block's positivity hypotheses are the tail's. -/
@[reducible] def tailOf {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw) : EnTail mid oc rd :=
  ⟨p.dε, p.dγ, p.dβ, p.z1, p.zb1, p.z2, p.zb2, p.pW, p.pb, p.pε, p.pγ, p.pβ⟩

/-- The tail of the MBConv1 block's weights. -/
@[reducible] def tailOfNoExp {ic oc rd kh kw : Nat} (p : MBWNoExp ic oc rd kh kw) : EnTail ic oc rd :=
  ⟨p.dε, p.dγ, p.dβ, p.z1, p.zb1, p.z2, p.zb2, p.pW, p.pb, p.pε, p.pγ, p.pβ⟩

/-! The tail's forward activations, from the depthwise conv output `dc` — `enetExpTiedG`'s `dn`,
`dr`, `s`, `e1`, `z`, `e2`, `se`, `pc`, definitionally. -/

section
variable (N h w : Nat) {mid oc rd : Nat} (t : EnTail mid oc rd)

noncomputable def tDn (dc : Vec (N * (mid * h * w))) : Vec (N * (mid * h * w)) :=
  bnBatchLA N mid h w t.dε t.dγ t.dβ dc

noncomputable def tDr (dc : Vec (N * (mid * h * w))) : Vec (N * (mid * h * w)) :=
  swish (N * (mid * h * w)) (tDn N h w t dc)

noncomputable def tS (dc : Vec (N * (mid * h * w))) : Vec (N * mid) :=
  batchMap N (globalAvgPoolFlat mid h w) (tDr N h w t dc)

noncomputable def tE1 (dc : Vec (N * (mid * h * w))) : Vec (N * rd) :=
  batchMap N (dense t.z1 t.zb1) (tS N h w t dc)

noncomputable def tZ (dc : Vec (N * (mid * h * w))) : Vec (N * rd) :=
  swish (N * rd) (tE1 N h w t dc)

noncomputable def tE2 (dc : Vec (N * (mid * h * w))) : Vec (N * mid) :=
  batchMap N (dense t.z2 t.zb2) (tZ N h w t dc)

noncomputable def tSe (dc : Vec (N * (mid * h * w))) : Vec (N * (mid * h * w)) :=
  seB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2 (tDr N h w t dc)

noncomputable def tPc (dc : Vec (N * (mid * h * w))) : Vec (N * (oc * h * w)) :=
  batchMap N (flatConv t.pW t.pb) (tSe N h w t dc)

/-! The tail's backward chain from the block-output cotangent `dy` — `enetExpTiedG`'s `cotPbn` …
`cotDc`, definitionally. -/

noncomputable def tCotPbn (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnBackB N oc h w t.pε hp t.pγ t.pβ (tPc N h w t dc) dy

noncomputable def tCotSeOut (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  cInB N (h := h) (w := w) t.pW t.pb (tCotPbn N h w t hp dc dy)

noncomputable def tDgate (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) : Vec (N * mid) :=
  gateCotB N mid h w (tDr N h w t dc) (tCotSeOut N h w t hp dc dy)

noncomputable def tCotE2 (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) : Vec (N * mid) :=
  sigBackB (N * mid) (tE2 N h w t dc) (tDgate N h w t hp dc dy)

noncomputable def tCotZ (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w))) (dy : Vec (N * (oc * h * w))) :
    Vec (N * rd) :=
  rowDenseBackFlat N rd mid t.z2 (tCotE2 N h w t hp dc dy)

noncomputable def tCotE1 (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) : Vec (N * rd) :=
  swBackB (N * rd) (tE1 N h w t dc) (tCotZ N h w t hp dc dy)

noncomputable def tCotDxSe (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  seInB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2 (tDr N h w t dc) (tCotSeOut N h w t hp dc dy)

noncomputable def tCotDn (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  swBackB (N * (mid * h * w)) (tDn N h w t dc) (tCotDxSe N h w t hp dc dy)

noncomputable def tCotDc (hd : 0 < t.dε) (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))
    (dy : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  bnBackB N mid h w t.dε hd t.dγ t.dβ dc (tCotDn N h w t hp dc dy)

end

/-! The stride-1 expand front (b3, b5, …, and the widenings b9 / b16). -/

section
variable (N h w : Nat) {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)

noncomputable def xEc (xin : Vec (N * (ic * h * w))) : Vec (N * (mid * h * w)) :=
  batchMap N (flatConv p.eW p.eb) xin

noncomputable def xEn (xin : Vec (N * (ic * h * w))) : Vec (N * (mid * h * w)) :=
  bnBatchLA N mid h w p.eε p.eγ p.eβ (xEc N h w p xin)

noncomputable def xEr (xin : Vec (N * (ic * h * w))) : Vec (N * (mid * h * w)) :=
  swish (N * (mid * h * w)) (xEn N h w p xin)

noncomputable def xDc (xin : Vec (N * (ic * h * w))) : Vec (N * (mid * h * w)) :=
  batchMap N (depthwiseFlat p.dW p.db) (xEr N h w p xin)

noncomputable def xCotEr (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w)))
    (dy : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  dInB N p.dW p.db (tCotDc N h w (tailOf p) hd hp (xDc N h w p xin) dy)

noncomputable def xCotEn (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w)))
    (dy : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  swBackB (N * (mid * h * w)) (xEn N h w p xin) (xCotEr N h w p hd hp xin dy)

noncomputable def xCotEc (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  bnBackB N mid h w p.eε he p.eγ p.eβ (xEc N h w p xin) (xCotEn N h w p hd hp xin dy)

/-- The widening block's input cotangent — the expand conv's input-VJP. -/
noncomputable def xCotIn (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * h * w))) (dy : Vec (N * (oc * h * w))) : Vec (N * (ic * h * w)) :=
  cInB N (h := h) (w := w) p.eW p.eb (xCotEc N h w p he hd hp xin dy)

end

/-- The residual block's input cotangent — the body's, plus the identity skip. -/
noncomputable def rCotIn (N h w : Nat) {c mid rd kh kw : Nat} (p : MBW c mid c rd kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) (xin dy : Vec (N * (c * h * w))) :
    Vec (N * (c * h * w)) :=
  fun i => xCotIn N h w p he hd hp xin dy i + dy i

/-! The strided front (b2, b4, b6, b12): expand at the input grid `2h×2w`. -/

section
variable (N h w : Nat) {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)

noncomputable def sEc (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  batchMap N (flatConv p.eW p.eb) xin

noncomputable def sEn (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  bnBatchLA N mid (2 * h) (2 * w) p.eε p.eγ p.eβ (sEc N h w p xin)

noncomputable def sEr (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  swish (N * (mid * (2 * h) * (2 * w))) (sEn N h w p xin)

noncomputable def sDc (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (mid * h * w)) :=
  batchMap N (depthwiseStride2Flat p.dW p.db) (sEr N h w p xin)

noncomputable def sCotEr (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  dStridedInB N p.dW p.db (tCotDc N h w (tailOf p) hd hp (sDc N h w p xin) dy)

noncomputable def sCotEn (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  swBackB (N * (mid * (2 * h) * (2 * w))) (sEn N h w p xin) (sCotEr N h w p hd hp xin dy)

noncomputable def sCotEc (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  bnBackB N mid (2 * h) (2 * w) p.eε he p.eγ p.eβ (sEc N h w p xin) (sCotEn N h w p hd hp xin dy)

/-- The strided block's input cotangent. -/
noncomputable def sCotIn (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) :
    Vec (N * (ic * (2 * h) * (2 * w))) :=
  cInB N (h := 2 * h) (w := 2 * w) p.eW p.eb (sCotEc N h w p he hd hp xin dy)

end

/-! The MBConv1 front (b1): the depthwise runs on the block input. -/

noncomputable def nDc (N h w : Nat) {ic oc rd kh kw : Nat} (p : MBWNoExp ic oc rd kh kw)
    (xin : Vec (N * (ic * h * w))) : Vec (N * (ic * h * w)) :=
  batchMap N (depthwiseFlat p.dW p.db) xin

/-- The MBConv1 block's input cotangent — the depthwise's input-VJP. -/
noncomputable def nCotIn (N h w : Nat) {ic oc rd kh kw : Nat} (p : MBWNoExp ic oc rd kh kw)
    (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w))) (dy : Vec (N * (oc * h * w))) :
    Vec (N * (ic * h * w)) :=
  dInB N p.dW p.db (tCotDc N h w (tailOfNoExp p) hd hp (nDc N h w p xin) dy)

/-! The stem (`enetStemTiedG`'s chain) and the head (`enetHeadTiedG`'s). -/

section
variable (N h w : Nat) {ic oc kHs kWs : Nat} (Ws : Kernel4 oc ic kHs kWs) (bs : Vec oc)

noncomputable def stStc (x : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) :=
  batchMap N (flatConvStride2Xla Ws bs) x

noncomputable def stStn (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    Vec (N * (oc * h * w)) :=
  bnBatchLA N oc h w εs γs βs (stStc N h w Ws bs x)

noncomputable def stCotBnS (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (dy : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  swBackB (N * (oc * h * w)) (stStn N h w Ws bs εs γs βs x) dy

noncomputable def stCotStc (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnBackB N oc h w εs hεs γs βs (stStc N h w Ws bs x) (stCotBnS N h w Ws bs εs γs βs x dy)

end

section
variable (N h w : Nat) {c oc : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc)

noncomputable def hdHc (xin : Vec (N * (c * h * w))) : Vec (N * (oc * h * w)) :=
  batchMap N (flatConv Wh bh) xin

noncomputable def hdHn (εh : ℝ) (γh βh : Vec oc) (xin : Vec (N * (c * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnBatchLA N oc h w εh γh βh (hdHc N h w Wh bh xin)

noncomputable def hdGap (εh : ℝ) (γh βh : Vec oc) (xin : Vec (N * (c * h * w))) : Vec (N * oc) :=
  batchMap N (globalAvgPoolFlat oc h w) (swish (N * (oc * h * w)) (hdHn N h w Wh bh εh γh βh xin))

end

noncomputable def hdCotHr (N h w : Nat) {oc nC : Nat} (Wfc : Mat oc nC) (g : Vec (N * nC)) :
    Vec (N * (oc * h * w)) :=
  gapInB N oc h w (rowDenseBackFlat N oc nC Wfc g)

section
variable (N h w : Nat) {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ)

noncomputable def hdCotHsw (γh βh : Vec oc) (Wfc : Mat oc nC) (xin : Vec (N * (c * h * w)))
    (g : Vec (N * nC)) :
    Vec (N * (oc * h * w)) :=
  swBackB (N * (oc * h * w)) (hdHn N h w Wh bh εh γh βh xin) (hdCotHr N h w Wfc g)

noncomputable def hdCotHbn (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nC)) : Vec (N * (oc * h * w)) :=
  bnBackB N oc h w εh hεh γh βh (hdHc N h w Wh bh xin) (hdCotHsw N h w Wh bh εh γh βh Wfc xin g)

/-- The head's input cotangent — the 1×1 conv's input-VJP. -/
noncomputable def hdCotIn (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nC)) : Vec (N * (c * h * w)) :=
  cInB N (h := h) (w := w) Wh bh (hdCotHbn N h w Wh bh εh hεh γh βh Wfc xin g)

end

/-! ### The stage backwards, written out

Each is the `BatchedBackLinks` stage graph's faithfulness read at an `.operand` leaf: the graph's
`den` IS the chain above node for node, except the BatchNorm link, which `bnBatchLABack_faithful`
turns into `bnBackB`. -/

theorem den_bnBatchLABack_eq_bnBackB {N oc h w : Nat} (gN xN es : String) (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec oc) (x : Vec (N * (oc * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.bnBatchLABack gN xN es ε γ x e) = bnBackB N oc h w ε hε γ β x (den e) :=
  bnBatchLABack_faithful gN xN es ε γ β hε x e

theorem cbsB_back_eq (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * (ic * h * w))) (dy : Vec (N * (oc * h * w))) :
    (cbsB_has_vjp N (h := h) (w := w) W b ε hε γ β).backward x dy
      = cInB N (h := h) (w := w) W b (bnBackB N oc h w ε hε γ β (batchMap N (flatConv W b) x)
          (swBackB (N * (oc * h * w)) (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x)) dy)) := by
  have hg := cbsBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy)
  rw [den_operand] at hg
  rw [← hg]
  show cInB N W b (den (SHlo.bnBatchLABack _ _ _ ε γ _ _)) = _
  rw [den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]
  rfl

theorem dwbsB_back_eq (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) (x dy : Vec (N * (c * h * w))) :
    (dwbsB_has_vjp N (h := h) (w := w) W b ε hε γ β).backward x dy
      = dInB N W b (bnBackB N c h w ε hε γ β (batchMap N (depthwiseFlat W b) x)
          (swBackB (N * (c * h * w)) (bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x)) dy)) := by
  have hg := dwbsBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy)
  rw [den_operand] at hg
  rw [← hg]
  show dInB N W b (den (SHlo.bnBatchLABack _ _ _ ε γ _ _)) = _
  rw [den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]
  rfl

theorem dwbsSB_back_eq (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (dy : Vec (N * (c * h * w))) :
    (dwbsSB_has_vjp N (h := h) (w := w) W b ε hε γ β).backward x dy
      = dStridedInB N W b (bnBackB N c h w ε hε γ β (batchMap N (depthwiseStride2Flat W b) x)
          (swBackB (N * (c * h * w))
            (bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x)) dy)) := by
  have hg := dwbsSBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy)
  rw [den_operand] at hg
  rw [← hg]
  show dStridedInB N W b (den (SHlo.bnBatchLABack _ _ _ ε γ _ _)) = _
  rw [den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]
  rfl

theorem projB_back_eq (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * (ic * h * w))) (dy : Vec (N * (oc * h * w))) :
    (projB_has_vjp N (h := h) (w := w) W b ε hε γ β).backward x dy
      = cInB N (h := h) (w := w) W b (bnBackB N oc h w ε hε γ β (batchMap N (flatConv W b) x) dy) := by
  have hg := projBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy)
  rw [den_operand] at hg
  rw [← hg]
  show cInB N W b (den (SHlo.bnBatchLABack _ _ _ ε γ _ _)) = _
  rw [den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]
  rfl

/-! ### Each block's input cotangent IS its certified VJP

T3 threads the block-output cotangents by the block VJPs' `.backward`; the replicas compute theirs
by the explicit chain. These say the two agree, so the sharding argument (which needs the explicit
chain) lands on T3's own `.backward` terms. `HasVJP.backward_unique` swaps the bundle-level witness
for the unfolded one, which is then the stage composition by `rfl`. -/

theorem xCotIn_eq_vjp (N h w : Nat) {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w)))
    (dy : Vec (N * (oc * h * w))) :
    (mbExpW_has_vjp N h w p he hd hp).backward xin dy = xCotIn N h w p he hd hp xin dy := by
  rw [HasVJP.backward_unique (mbExpW_has_vjp N h w p he hd hp)
    (mbExpFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ p.dW p.db p.dε hd p.dγ p.dβ
      p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ)]
  have hc : (mbExpFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ p.dW p.db p.dε hd
      p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ).backward xin dy
      = (cbsB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ).backward xin
          ((dwbsB_has_vjp N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ).backward
            (cbsB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ xin)
            ((seB_has_vjp N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2).backward
              (dwbsB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ
                (cbsB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ xin))
              ((projB_has_vjp N (h := h) (w := w) p.pW p.pb p.pε hp p.pγ p.pβ).backward
                (seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2
                  (dwbsB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ
                    (cbsB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ xin))) dy))) := rfl
  rw [hc, projB_back_eq, dwbsB_back_eq, cbsB_back_eq]
  rfl

theorem rCotIn_eq_vjp (N h w : Nat) {c mid rd kh kw : Nat} (p : MBW c mid c rd kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) (xin dy : Vec (N * (c * h * w))) :
    (mbResidW_has_vjp N h w p he hd hp).backward xin dy = rCotIn N h w p he hd hp xin dy := by
  rw [HasVJP.backward_unique (mbResidW_has_vjp N h w p he hd hp)
    (mbResidFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ p.dW p.db p.dε hd p.dγ p.dβ
      p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ)]
  have hc : (mbResidFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ p.dW p.db p.dε hd
      p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ).backward xin dy
      = fun i => (cbsB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ).backward xin
          ((dwbsB_has_vjp N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ).backward
            (cbsB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ xin)
            ((seB_has_vjp N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2).backward
              (dwbsB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ
                (cbsB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ xin))
              ((projB_has_vjp N (h := h) (w := w) p.pW p.pb p.pε hp p.pγ p.pβ).backward
                (seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2
                  (dwbsB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ
                    (cbsB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ xin))) dy))) i + dy i := rfl
  rw [hc, projB_back_eq, dwbsB_back_eq, cbsB_back_eq]
  rfl

theorem sCotIn_eq_vjp (N h w : Nat) {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dy : Vec (N * (oc * h * w))) :
    (mbStridedW_has_vjp N h w p he hd hp).backward xin dy = sCotIn N h w p he hd hp xin dy := by
  rw [HasVJP.backward_unique (mbStridedW_has_vjp N h w p he hd hp)
    (mbStridedFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ p.dW p.db p.dε hd p.dγ
      p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ)]
  have hc : (mbStridedFwdB_has_vjp N (h := h) (w := w) p.eW p.eb p.eε he p.eγ p.eβ p.dW p.db p.dε hd
      p.dγ p.dβ p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ).backward xin dy
      = (cbsB_has_vjp N (h := 2 * h) (w := 2 * w) p.eW p.eb p.eε he p.eγ p.eβ).backward xin
          ((dwbsSB_has_vjp N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ).backward
            (cbsB N (h := 2 * h) (w := 2 * w) p.eW p.eb p.eε p.eγ p.eβ xin)
            ((seB_has_vjp N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2).backward
              (dwbsSB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ
                (cbsB N (h := 2 * h) (w := 2 * w) p.eW p.eb p.eε p.eγ p.eβ xin))
              ((projB_has_vjp N (h := h) (w := w) p.pW p.pb p.pε hp p.pγ p.pβ).backward
                (seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2
                  (dwbsSB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ
                    (cbsB N (h := 2 * h) (w := 2 * w) p.eW p.eb p.eε p.eγ p.eβ xin))) dy))) := rfl
  rw [hc, projB_back_eq, dwbsSB_back_eq, cbsB_back_eq]
  rfl

theorem nCotIn_eq_vjp (N h w : Nat) {ic oc rd kh kw : Nat} (p : MBWNoExp ic oc rd kh kw)
    (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w))) (dy : Vec (N * (oc * h * w))) :
    (mbNoExpW_has_vjp N h w p hd hp).backward xin dy = nCotIn N h w p hd hp xin dy := by
  rw [HasVJP.backward_unique (mbNoExpW_has_vjp N h w p hd hp)
    (mbNoExpFwdB_has_vjp N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ
      p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ)]
  have hc : (mbNoExpFwdB_has_vjp N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ
      p.z1 p.zb1 p.z2 p.zb2 p.pW p.pb p.pε hp p.pγ p.pβ).backward xin dy
      = (dwbsB_has_vjp N (h := h) (w := w) p.dW p.db p.dε hd p.dγ p.dβ).backward xin
          ((seB_has_vjp N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2).backward
            (dwbsB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ xin)
            ((projB_has_vjp N (h := h) (w := w) p.pW p.pb p.pε hp p.pγ p.pβ).backward
              (seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2
                (dwbsB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ xin)) dy)) := rfl
  rw [hc, projB_back_eq, dwbsB_back_eq]
  rfl

theorem hdCotIn_eq_vjp (N h w : Nat) {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc)
    (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nC)) :
    (headFwdB_has_vjp N (h := h) (w := w) Wh bh εh hεh γh βh Wfc bfc).backward xin g
      = hdCotIn N h w Wh bh εh hεh γh βh Wfc xin g := by
  have hg := headBackBatchedGraph_faithful Wh bh εh hεh γh βh Wfc bfc xin (.operand "" g)
  rw [den_operand] at hg
  rw [← hg]
  show cInB N Wh bh (den (SHlo.bnBatchLABack _ _ _ εh γh _ _)) = _
  rw [den_bnBatchLABack_eq_bnBackB _ _ _ εh hεh γh βh]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — every link B0 adds to ResNet-34's is linear in its cotangent
-- ════════════════════════════════════════════════════════════════

theorem bnBackB_smul (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (oc * h * w))) : IsHomog (bnBackB N oc h w ε hε γ β x) :=
  HasVJP.backward_smul _ _

theorem swBackB_smul (n : Nat) (x : Vec n) : IsHomog (swBackB n x) :=
  HasVJP.backward_smul _ _

theorem sigBackB_smul (n : Nat) (x : Vec n) : IsHomog (sigBackB n x) :=
  HasVJP.backward_smul _ _

theorem seInB_smul (N : Nat) {c h w rd : Nat} (W₁ : Mat c rd) (b₁ : Vec rd) (W₂ : Mat rd c)
    (b₂ : Vec c) (x : Vec (N * (c * h * w))) : IsHomog (seInB N (h := h) (w := w) W₁ b₁ W₂ b₂ x) :=
  HasVJP.backward_smul _ _

/-- The SE gate cotangent `Σ_{h,w} x ⊙ dy` is linear in `dy`. -/
theorem gateCotB_smul (N c h w : Nat) (x : Vec (N * (c * h * w))) :
    IsHomog (gateCotB N c h w x) := by
  intro s dy
  funext idx
  simp only [gateCotB, batchSlice, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun _ _ => ?_)
  split_ifs <;> ring

-- ════════════════════════════════════════════════════════════════
-- § 2. Sharding — every link, on a replica, is the shard of the global link
-- ════════════════════════════════════════════════════════════════

theorem swBackB_shard {R N n : Nat} (X DY : Vec ((R * N) * n)) (r : Fin R) :
    swBackB (N * n) (batchShard R N n X r) (batchShard R N n DY r)
      = batchShard R N n (swBackB ((R * N) * n) X DY) r := rfl

theorem sigBackB_shard {R N n : Nat} (X DY : Vec ((R * N) * n)) (r : Fin R) :
    sigBackB (N * n) (batchShard R N n X r) (batchShard R N n DY r)
      = batchShard R N n (sigBackB ((R * N) * n) X DY) r := rfl

/-- The SE gate cotangent, per example: channel `k`'s spatial sum of `x ⊙ dy`. -/
noncomputable def gateEx (c h w : Nat) (xs ds : Vec (c * h * w)) : Vec c :=
  fun k => ∑ q : Fin (c * h * w), if flatChannel c h w q = k then xs q * ds q else 0

/-- ⭐ **The SE gate cotangent reads one example at a time** — `seReduceB` is `batchMapAux` of
    `gateEx` — so it shards like ResNet-34's pool backward. -/
theorem gateCotB_shard {R N : Nat} (c h w : Nat) (X DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    gateCotB N c h w (batchShard R N (c * h * w) X r) (batchShard R N (c * h * w) DY r)
      = batchShard R N c (gateCotB (R * N) c h w X DY) r :=
  (batchShard_batchMapAux (gateEx c h w) X DY r).symm

/-- The fused SE input-VJP is the per-example `seBlockFull` VJP, lifted — `seBackBatched`'s `den`. -/
theorem seInB_eq_batchMapAux (N : Nat) {c h w rd : Nat} (W₁ : Mat c rd) (b₁ : Vec rd)
    (W₂ : Mat rd c) (b₂ : Vec c) (x dy : Vec (N * (c * h * w))) :
    seInB N (h := h) (w := w) W₁ b₁ W₂ b₂ x dy
      = batchMapAux N (seBlockFull_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward x dy := by
  have hg := seBackBatched_faithful "" "" "" "" "" W₁ b₁ W₂ b₂ x (.operand "" dy)
  rw [den_operand] at hg
  exact hg.symm

theorem seInB_shard {R N : Nat} {c h w rd : Nat} (W₁ : Mat c rd) (b₁ : Vec rd) (W₂ : Mat rd c)
    (b₂ : Vec c) (X DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    seInB N (h := h) (w := w) W₁ b₁ W₂ b₂ (batchShard R N (c * h * w) X r)
        (batchShard R N (c * h * w) DY r)
      = batchShard R N (c * h * w) (seInB (R * N) (h := h) (w := w) W₁ b₁ W₂ b₂ X DY) r := by
  rw [seInB_eq_batchMapAux, seInB_eq_batchMapAux]
  exact (batchShard_batchMapAux _ X DY r).symm

/-- ⭐⭐ **The sync-BN backward on replica `r` is shard `r` of the certified global BN backward** —
    `bnSyncInB_shard` (P2 at the network index) read through `bnInB_eq_bnBackB`, so the right-hand
    side is `bnBackB`, T3's own BN link. -/
theorem bnSyncInB_shard_bnBackB (R : Nat) (hR : 0 < R) (N oc h w : Nat) (hm : N * (h * w) ≠ 0)
    (hM : (R * N) * (h * w) ≠ 0) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (xs dys : Fin R → Vec (N * (oc * h * w))) (X DY : Vec ((R * N) * (oc * h * w)))
    (hxs : ∀ r, xs r = batchShard R N (oc * h * w) X r)
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    bnSyncInB R hR N oc h w ε γ xs dys r
      = batchShard R N (oc * h * w) (bnBackB (R * N) oc h w ε hε γ β X DY) r := by
  rw [bnSyncInB_shard R hR N oc h w hm hM ε γ xs dys X DY hxs hdys r,
    bnInB_eq_bnBackB (R * N) oc h w ε hε γ β]

-- ════════════════════════════════════════════════════════════════
-- § 4. The single-device chain is linear in the block-output cotangent
-- ════════════════════════════════════════════════════════════════

section
variable (N h w : Nat) {mid oc rd : Nat} (t : EnTail mid oc rd)

section
variable (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w)))

theorem tCotPbn_smul :
    IsHomog (tCotPbn N h w t hp dc) :=
  bnBackB_smul _ _ _ _ _ _ _ _ _

theorem tCotSeOut_smul :
    IsHomog (tCotSeOut N h w t hp dc) := by
  intro s dy
  unfold tCotSeOut; rw [tCotPbn_smul, cInB_smul]

theorem tDgate_smul :
    IsHomog (tDgate N h w t hp dc) := by
  intro s dy
  unfold tDgate; rw [tCotSeOut_smul, gateCotB_smul]

theorem tCotE2_smul :
    IsHomog (tCotE2 N h w t hp dc) := by
  intro s dy
  unfold tCotE2; rw [tDgate_smul, sigBackB_smul]

theorem tCotZ_smul :
    IsHomog (tCotZ N h w t hp dc) := by
  intro s dy
  unfold tCotZ; rw [tCotE2_smul, rowDenseBackFlat_smul]

theorem tCotE1_smul :
    IsHomog (tCotE1 N h w t hp dc) := by
  intro s dy
  unfold tCotE1; rw [tCotZ_smul, swBackB_smul]

theorem tCotDxSe_smul :
    IsHomog (tCotDxSe N h w t hp dc) := by
  intro s dy
  unfold tCotDxSe; rw [tCotSeOut_smul, seInB_smul]

theorem tCotDn_smul :
    IsHomog (tCotDn N h w t hp dc) := by
  intro s dy
  unfold tCotDn; rw [tCotDxSe_smul, swBackB_smul]

end

theorem tCotDc_smul (hd : 0 < t.dε) (hp : 0 < t.pε) (dc : Vec (N * (mid * h * w))) :
    IsHomog (tCotDc N h w t hd hp dc) := by
  intro s dy
  unfold tCotDc; rw [tCotDn_smul, bnBackB_smul]

end

section
variable (N h w : Nat) {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)

theorem xCotEr_smul (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w))) :
    IsHomog (xCotEr N h w p hd hp xin) := by
  intro s dy
  unfold xCotEr; rw [tCotDc_smul, dInB_smul]

theorem xCotEn_smul (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w))) :
    IsHomog (xCotEn N h w p hd hp xin) := by
  intro s dy
  unfold xCotEn; rw [xCotEr_smul, swBackB_smul]

theorem xCotEc_smul (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * h * w))) :
    IsHomog (xCotEc N h w p he hd hp xin) := by
  intro s dy
  unfold xCotEc; rw [xCotEn_smul, bnBackB_smul]

theorem sCotEr_smul (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (sCotEr N h w p hd hp xin) := by
  intro s dy
  unfold sCotEr; rw [tCotDc_smul, dStridedInB_smul]

theorem sCotEn_smul (hd : 0 < p.dε) (hp : 0 < p.pε) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (sCotEn N h w p hd hp xin) := by
  intro s dy
  unfold sCotEn; rw [sCotEr_smul, swBackB_smul]

theorem sCotEc_smul (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (sCotEc N h w p he hd hp xin) := by
  intro s dy
  unfold sCotEc; rw [sCotEn_smul, bnBackB_smul]

end

theorem stCotBnS_smul (N h w : Nat) {ic oc kHs kWs : Nat} (Ws : Kernel4 oc ic kHs kWs) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (stCotBnS N h w Ws bs εs γs βs x) :=
  swBackB_smul _ _

theorem stCotStc_smul (N h w : Nat) {ic oc kHs kWs : Nat} (Ws : Kernel4 oc ic kHs kWs) (bs : Vec oc)
    (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (stCotStc N h w Ws bs εs hεs γs βs x) := by
  intro s dy
  unfold stCotStc; rw [stCotBnS_smul, bnBackB_smul]

theorem hdCotHr_smul (N h w : Nat) {oc nC : Nat} (Wfc : Mat oc nC) :
    IsHomog (hdCotHr N h w Wfc) := by
  intro s g
  unfold hdCotHr; rw [rowDenseBackFlat_smul, gapInB_smul]

theorem hdCotHsw_smul (N h w : Nat) {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ)
    (γh βh : Vec oc) (Wfc : Mat oc nC) (xin : Vec (N * (c * h * w))) :
    IsHomog (hdCotHsw N h w Wh bh εh γh βh Wfc xin) := by
  intro s g
  unfold hdCotHsw; rw [hdCotHr_smul, swBackB_smul]

theorem hdCotHbn_smul (N h w : Nat) {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ)
    (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC) (xin : Vec (N * (c * h * w))) :
    IsHomog (hdCotHbn N h w Wh bh εh hεh γh βh Wfc xin) := by
  intro s g
  unfold hdCotHbn; rw [hdCotHsw_smul, bnBackB_smul]

-- ════════════════════════════════════════════════════════════════
-- § 5. The replica chain, and its shard lemmas
--   Saved activations are the shards of the single-device forward's (the forward half,
--   `EfficientNetSyncB`, is what says a replica computes exactly those); every cotangent is the
--   replica's own, from the family `dys` of block-output cotangents. Each BatchNorm link is
--   `bnSyncInB` — `bnBackSite`'s `replicas > 1` branch — and needs no `0 < ε`.
-- ════════════════════════════════════════════════════════════════

/-! ### The tail, on the replicas -/

section
variable (R : Nat) (hR : 0 < R) (N h w : Nat)

noncomputable def tsCotPbn {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w t.pε t.pγ (fun r => batchShard R N (oc * h * w) (tPc (R * N) h w t DC) r)
    dys r

noncomputable def tsCotSeOut {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (mid * h * w)) :=
  cInB N (h := h) (w := w) t.pW t.pb (tsCotPbn R hR N h w t DC dys r)

noncomputable def tsDgate {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * mid) :=
  gateCotB N mid h w (batchShard R N (mid * h * w) (tDr (R * N) h w t DC) r)
    (tsCotSeOut R hR N h w t DC dys r)

noncomputable def tsCotE2 {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * mid) :=
  sigBackB (N * mid) (batchShard R N mid (tE2 (R * N) h w t DC) r) (tsDgate R hR N h w t DC dys r)

noncomputable def tsCotZ {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * rd) :=
  rowDenseBackFlat N rd mid t.z2 (tsCotE2 R hR N h w t DC dys r)

noncomputable def tsCotE1 {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * rd) :=
  swBackB (N * rd) (batchShard R N rd (tE1 (R * N) h w t DC) r) (tsCotZ R hR N h w t DC dys r)

noncomputable def tsCotDxSe {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (mid * h * w)) :=
  seInB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2
    (batchShard R N (mid * h * w) (tDr (R * N) h w t DC) r) (tsCotSeOut R hR N h w t DC dys r)

noncomputable def tsCotDn {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (mid * h * w)) :=
  swBackB (N * (mid * h * w)) (batchShard R N (mid * h * w) (tDn (R * N) h w t DC) r)
    (tsCotDxSe R hR N h w t DC dys r)

noncomputable def tsCotDc {mid oc rd : Nat} (t : EnTail mid oc rd)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w t.dε t.dγ (fun r => batchShard R N (mid * h * w) DC r)
    (tsCotDn R hR N h w t DC dys) r

section
variable {mid oc rd : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (t : EnTail mid oc rd)
include hN hh hw

theorem tsCotPbn_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotPbn R hR N h w t DC dys r = batchShard R N (oc * h * w) (tCotPbn (R * N) h w t hp DC DY) r :=
  bnSyncInB_shard_bnBackB R hR N oc h w (nhw_ne_zero hN hh hw)
    (nhw_ne_zero (Nat.mul_pos hR hN) hh hw) t.pε hp t.pγ t.pβ _ dys _ DY (fun _ => rfl) hdys r

theorem tsCotSeOut_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotSeOut R hR N h w t DC dys r
      = batchShard R N (mid * h * w) (tCotSeOut (R * N) h w t hp DC DY) r := by
  unfold tsCotSeOut
  rw [tsCotPbn_shard R hR N h w hN hh hw t hp DC dys DY hdys r, cInB_shard]
  rfl

theorem tsDgate_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsDgate R hR N h w t DC dys r = batchShard R N mid (tDgate (R * N) h w t hp DC DY) r := by
  unfold tsDgate
  rw [tsCotSeOut_shard R hR N h w hN hh hw t hp DC dys DY hdys r, gateCotB_shard]
  rfl

theorem tsCotE2_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotE2 R hR N h w t DC dys r = batchShard R N mid (tCotE2 (R * N) h w t hp DC DY) r := by
  unfold tsCotE2
  rw [tsDgate_shard R hR N h w hN hh hw t hp DC dys DY hdys r, sigBackB_shard]
  rfl

theorem tsCotZ_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotZ R hR N h w t DC dys r = batchShard R N rd (tCotZ (R * N) h w t hp DC DY) r := by
  unfold tsCotZ
  rw [tsCotE2_shard R hR N h w hN hh hw t hp DC dys DY hdys r, rowDenseBackFlat_shard]
  rfl

theorem tsCotE1_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotE1 R hR N h w t DC dys r = batchShard R N rd (tCotE1 (R * N) h w t hp DC DY) r := by
  unfold tsCotE1
  rw [tsCotZ_shard R hR N h w hN hh hw t hp DC dys DY hdys r, swBackB_shard]
  rfl

theorem tsCotDxSe_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotDxSe R hR N h w t DC dys r
      = batchShard R N (mid * h * w) (tCotDxSe (R * N) h w t hp DC DY) r := by
  unfold tsCotDxSe
  rw [tsCotSeOut_shard R hR N h w hN hh hw t hp DC dys DY hdys r, seInB_shard]
  rfl

theorem tsCotDn_shard (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotDn R hR N h w t DC dys r = batchShard R N (mid * h * w) (tCotDn (R * N) h w t hp DC DY) r := by
  unfold tsCotDn
  rw [tsCotDxSe_shard R hR N h w hN hh hw t hp DC dys DY hdys r, swBackB_shard]
  rfl

theorem tsCotDc_shard (hd : 0 < t.dε) (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotDc R hR N h w t DC dys r
      = batchShard R N (mid * h * w) (tCotDc (R * N) h w t hd hp DC DY) r :=
  bnSyncInB_shard_bnBackB R hR N mid h w (nhw_ne_zero hN hh hw)
    (nhw_ne_zero (Nat.mul_pos hR hN) hh hw) t.dε hd t.dγ t.dβ _ _ DC _ (fun _ => rfl)
    (tsCotDn_shard R hR N h w hN hh hw t hp DC dys DY hdys) r

end

/-! ### The fronts, on the replicas -/

noncomputable def xsCotEr {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (mid * h * w)) :=
  dInB N p.dW p.db (tsCotDc R hR N h w (tailOf p) (xDc (R * N) h w p XIN) dys r)

noncomputable def xsCotEn {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (mid * h * w)) :=
  swBackB (N * (mid * h * w)) (batchShard R N (mid * h * w) (xEn (R * N) h w p XIN) r)
    (xsCotEr R hR N h w p XIN dys r)

noncomputable def xsCotEc {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.eε p.eγ (fun r => batchShard R N (mid * h * w) (xEc (R * N) h w p XIN) r)
    (xsCotEn R hR N h w p XIN dys) r

/-- A replica's input cotangent at a widening block (b9, b16). -/
noncomputable def xsCotIn {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (ic * h * w)) :=
  cInB N (h := h) (w := w) p.eW p.eb (xsCotEc R hR N h w p XIN dys r)

/-- A replica's input cotangent at a residual block — the body's plus the skip's. -/
noncomputable def rsCotIn {c mid rd kh kw : Nat} (p : MBW c mid c rd kh kw)
    (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w))) (r : Fin R) :
    Vec (N * (c * h * w)) :=
  fun i => xsCotIn R hR N h w p XIN dys r i + dys r i

noncomputable def ssCotEr {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (r : Fin R) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  dStridedInB N p.dW p.db (tsCotDc R hR N h w (tailOf p) (sDc (R * N) h w p XIN) dys r)

noncomputable def ssCotEn {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (r : Fin R) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  swBackB (N * (mid * (2 * h) * (2 * w)))
    (batchShard R N (mid * (2 * h) * (2 * w)) (sEn (R * N) h w p XIN) r)
    (ssCotEr R hR N h w p XIN dys r)

noncomputable def ssCotEc {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (r : Fin R) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  bnSyncInB R hR N mid (2 * h) (2 * w) p.eε p.eγ
    (fun r => batchShard R N (mid * (2 * h) * (2 * w)) (sEc (R * N) h w p XIN) r)
    (ssCotEn R hR N h w p XIN dys) r

/-- A replica's input cotangent at a strided block (b2, b4, b6, b12). -/
noncomputable def ssCotIn {ic mid oc rd kh kw : Nat} (p : MBW ic mid oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (r : Fin R) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  cInB N (h := 2 * h) (w := 2 * w) p.eW p.eb (ssCotEc R hR N h w p XIN dys r)

/-- A replica's input cotangent at the MBConv1 block (b1). -/
noncomputable def nsCotIn {ic oc rd kh kw : Nat} (p : MBWNoExp ic oc rd kh kw)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) :
    Vec (N * (ic * h * w)) :=
  dInB N p.dW p.db (tsCotDc R hR N h w (tailOfNoExp p) (nDc (R * N) h w p XIN) dys r)

end

noncomputable def stsCotBnS (R N h w : Nat) {ic oc kHs kWs : Nat} (Ws : Kernel4 oc ic kHs kWs)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) : Vec (N * (oc * h * w)) :=
  swBackB (N * (oc * h * w)) (batchShard R N (oc * h * w) (stStn (R * N) h w Ws bs εs γs βs X) r)
    (dys r)

noncomputable def stsCotStc (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kHs kWs : Nat}
    (Ws : Kernel4 oc ic kHs kWs) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w εs γs (fun r => batchShard R N (oc * h * w) (stStc (R * N) h w Ws bs X) r)
    (stsCotBnS R N h w Ws bs εs γs βs X dys) r

noncomputable def hdsCotHsw (R N h w : Nat) {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc)
    (εh : ℝ) (γh βh : Vec oc) (Wfc : Mat oc nC) (XIN : Vec ((R * N) * (c * h * w)))
    (gs : Fin R → Vec (N * nC)) (r : Fin R) : Vec (N * (oc * h * w)) :=
  swBackB (N * (oc * h * w)) (batchShard R N (oc * h * w) (hdHn (R * N) h w Wh bh εh γh βh XIN) r)
    (hdCotHr N h w Wfc (gs r))

section
variable (R : Nat) (hR : 0 < R) (N h w : Nat)

noncomputable def hdsCotHbn {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ)
    (γh βh : Vec oc) (Wfc : Mat oc nC) (XIN : Vec ((R * N) * (c * h * w)))
    (gs : Fin R → Vec (N * nC)) (r : Fin R) :
    Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w εh γh (fun r => batchShard R N (oc * h * w) (hdHc (R * N) h w Wh bh XIN) r)
    (hdsCotHsw R N h w Wh bh εh γh βh Wfc XIN gs) r

/-- A replica's input cotangent at the head. -/
noncomputable def hdsCotIn {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ)
    (γh βh : Vec oc) (Wfc : Mat oc nC) (XIN : Vec ((R * N) * (c * h * w)))
    (gs : Fin R → Vec (N * nC)) (r : Fin R) :
    Vec (N * (c * h * w)) :=
  cInB N (h := h) (w := w) Wh bh (hdsCotHbn R hR N h w Wh bh εh γh βh Wfc XIN gs r)

section
variable {ic mid oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBW ic mid oc rd kh kw)
include hN hh hw

theorem xsCotEr_shard (hd : 0 < p.dε) (hp : 0 < p.pε) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    xsCotEr R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (xCotEr (R * N) h w p hd hp XIN DY) r := by
  unfold xsCotEr
  rw [tsCotDc_shard R hR N h w hN hh hw (tailOf p) hd hp _ dys DY hdys r, dInB_shard]
  rfl

theorem xsCotEn_shard (hd : 0 < p.dε) (hp : 0 < p.pε) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    xsCotEn R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (xCotEn (R * N) h w p hd hp XIN DY) r := by
  unfold xsCotEn
  rw [xsCotEr_shard R hR N h w hN hh hw p hd hp XIN dys DY hdys r, swBackB_shard]
  rfl

theorem xsCotEc_shard (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
    (r : Fin R) :
    xsCotEc R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (xCotEc (R * N) h w p he hd hp XIN DY) r :=
  bnSyncInB_shard_bnBackB R hR N mid h w (nhw_ne_zero hN hh hw)
    (nhw_ne_zero (Nat.mul_pos hR hN) hh hw) p.eε he p.eγ p.eβ _ _ _ _ (fun _ => rfl)
    (xsCotEn_shard R hR N h w hN hh hw p hd hp XIN dys DY hdys) r

theorem xsCotIn_shard (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
    (r : Fin R) :
    xsCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (xCotIn (R * N) h w p he hd hp XIN DY) r := by
  unfold xsCotIn
  rw [xsCotEc_shard R hR N h w hN hh hw p he hd hp XIN dys DY hdys r, cInB_shard]
  rfl

end

theorem rsCotIn_shard {c mid rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBW c mid c rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))
    (DY : Vec ((R * N) * (c * h * w))) (hdys : ∀ r, dys r = batchShard R N (c * h * w) DY r)
    (r : Fin R) :
    rsCotIn R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (rCotIn (R * N) h w p he hd hp XIN DY) r := by
  unfold rsCotIn
  rw [xsCotIn_shard R hR N h w hN hh hw p he hd hp XIN dys DY hdys r, hdys r]
  rfl

section
variable {ic mid oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBW ic mid oc rd kh kw)
include hN hh hw

theorem ssCotEr_shard (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
    (r : Fin R) :
    ssCotEr R hR N h w p XIN dys r
      = batchShard R N (mid * (2 * h) * (2 * w)) (sCotEr (R * N) h w p hd hp XIN DY) r := by
  unfold ssCotEr
  rw [tsCotDc_shard R hR N h w hN hh hw (tailOf p) hd hp _ dys DY hdys r, dStridedInB_shard]
  rfl

theorem ssCotEn_shard (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
    (r : Fin R) :
    ssCotEn R hR N h w p XIN dys r
      = batchShard R N (mid * (2 * h) * (2 * w)) (sCotEn (R * N) h w p hd hp XIN DY) r := by
  unfold ssCotEn
  rw [ssCotEr_shard R hR N h w hN hh hw p hd hp XIN dys DY hdys r, swBackB_shard]
  rfl

theorem ssCotEc_shard (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
    (r : Fin R) :
    ssCotEc R hR N h w p XIN dys r
      = batchShard R N (mid * (2 * h) * (2 * w)) (sCotEc (R * N) h w p he hd hp XIN DY) r :=
  bnSyncInB_shard_bnBackB R hR N mid (2 * h) (2 * w)
    (nhw_ne_zero hN (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    (nhw_ne_zero (Nat.mul_pos hR hN) (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    p.eε he p.eγ p.eβ _ _ _ _ (fun _ => rfl)
    (ssCotEn_shard R hR N h w hN hh hw p hd hp XIN dys DY hdys) r

theorem ssCotIn_shard (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
    (r : Fin R) :
    ssCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w)) (sCotIn (R * N) h w p he hd hp XIN DY) r := by
  unfold ssCotIn
  rw [ssCotEc_shard R hR N h w hN hh hw p he hd hp XIN dys DY hdys r, cInB_shard]
  rfl

end

theorem nsCotIn_shard {ic oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBWNoExp ic oc rd kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
    (r : Fin R) :
    nsCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (nCotIn (R * N) h w p hd hp XIN DY) r := by
  unfold nsCotIn
  rw [tsCotDc_shard R hR N h w hN hh hw (tailOfNoExp p) hd hp _ dys DY hdys r, dInB_shard]
  rfl

end

theorem stsCotBnS_shard (R N h w : Nat) {ic oc kHs kWs : Nat} (Ws : Kernel4 oc ic kHs kWs)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    stsCotBnS R N h w Ws bs εs γs βs X dys r
      = batchShard R N (oc * h * w) (stCotBnS (R * N) h w Ws bs εs γs βs X DY) r := by
  unfold stsCotBnS
  rw [hdys r, swBackB_shard]
  rfl

theorem stsCotStc_shard (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kHs kWs : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (Ws : Kernel4 oc ic kHs kWs) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs)
    (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    stsCotStc R hR N h w Ws bs εs γs βs X dys r
      = batchShard R N (oc * h * w) (stCotStc (R * N) h w Ws bs εs hεs γs βs X DY) r :=
  bnSyncInB_shard_bnBackB R hR N oc h w (nhw_ne_zero hN hh hw)
    (nhw_ne_zero (Nat.mul_pos hR hN) hh hw) εs hεs γs βs _ _ _ _ (fun _ => rfl)
    (stsCotBnS_shard R N h w Ws bs εs γs βs X dys DY hdys) r

theorem hdCotHr_shard {R N : Nat} (h w : Nat) {oc nC : Nat} (Wfc : Mat oc nC)
    (G : Vec ((R * N) * nC)) (r : Fin R) :
    hdCotHr N h w Wfc (batchShard R N nC G r)
      = batchShard R N (oc * h * w) (hdCotHr (R * N) h w Wfc G) r := by
  unfold hdCotHr
  rw [rowDenseBackFlat_shard, gapInB_shard]

theorem hdsCotHsw_shard (R N h w : Nat) {c oc nC : Nat} (Wh : Kernel4 oc c 1 1) (bh : Vec oc)
    (εh : ℝ) (γh βh : Vec oc) (Wfc : Mat oc nC) (XIN : Vec ((R * N) * (c * h * w)))
    (gs : Fin R → Vec (N * nC)) (G : Vec ((R * N) * nC))
    (hgs : ∀ r, gs r = batchShard R N nC G r) (r : Fin R) :
    hdsCotHsw R N h w Wh bh εh γh βh Wfc XIN gs r
      = batchShard R N (oc * h * w) (hdCotHsw (R * N) h w Wh bh εh γh βh Wfc XIN G) r := by
  unfold hdsCotHsw
  rw [hgs r, hdCotHr_shard, swBackB_shard]
  rfl

section
variable (R : Nat) (hR : 0 < R) (N h w : Nat)

theorem hdsCotHbn_shard {c oc nC : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC)
    (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nC)) (G : Vec ((R * N) * nC))
    (hgs : ∀ r, gs r = batchShard R N nC G r) (r : Fin R) :
    hdsCotHbn R hR N h w Wh bh εh γh βh Wfc XIN gs r
      = batchShard R N (oc * h * w) (hdCotHbn (R * N) h w Wh bh εh hεh γh βh Wfc XIN G) r :=
  bnSyncInB_shard_bnBackB R hR N oc h w (nhw_ne_zero hN hh hw)
    (nhw_ne_zero (Nat.mul_pos hR hN) hh hw) εh hεh γh βh _ _ _ _ (fun _ => rfl)
    (hdsCotHsw_shard R N h w Wh bh εh γh βh Wfc XIN gs G hgs) r

theorem hdsCotIn_shard {c oc nC : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC)
    (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nC)) (G : Vec ((R * N) * nC))
    (hgs : ∀ r, gs r = batchShard R N nC G r) (r : Fin R) :
    hdsCotIn R hR N h w Wh bh εh γh βh Wfc XIN gs r
      = batchShard R N (c * h * w) (hdCotIn (R * N) h w Wh bh εh hεh γh βh Wfc XIN G) r := by
  unfold hdsCotIn
  rw [hdsCotHbn_shard R hR N h w hN hh hw Wh bh εh hεh γh βh Wfc XIN gs G hgs r, cInB_shard]
  rfl

/-! ### The scaled-shard invariant across each block

Replicas at `R ×` the shards of the single-device block-output cotangent `DY` hand the next block
up `R ×` the shards of T3's own `.backward` — sharding, then the explicit chain IS the VJP (§ 0),
then `HasVJP.backward_smul`. -/

theorem xsCotIn_scaled {ic mid oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBW ic mid oc rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    xsCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w)
          (fun i => (R : ℝ) * (mbExpW_has_vjp (R * N) h w p he hd hp).backward XIN DY i) r := by
  rw [xsCotIn_shard R hR N h w hN hh hw p he hd hp XIN dys _ hdys r, ← xCotIn_eq_vjp,
    HasVJP.backward_smul]

theorem rsCotIn_scaled {c mid rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBW c mid c rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))
    (DY : Vec ((R * N) * (c * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (c * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    rsCotIn R hR N h w p XIN dys r
      = batchShard R N (c * h * w)
          (fun i => (R : ℝ) * (mbResidW_has_vjp (R * N) h w p he hd hp).backward XIN DY i) r := by
  rw [rsCotIn_shard R hR N h w hN hh hw p he hd hp XIN dys _ hdys r, ← rCotIn_eq_vjp,
    HasVJP.backward_smul]

theorem ssCotIn_scaled {ic mid oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBW ic mid oc rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    ssCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w))
          (fun i => (R : ℝ) * (mbStridedW_has_vjp (R * N) h w p he hd hp).backward XIN DY i) r := by
  rw [ssCotIn_shard R hR N h w hN hh hw p he hd hp XIN dys _ hdys r, ← sCotIn_eq_vjp,
    HasVJP.backward_smul]

theorem nsCotIn_scaled {ic oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (p : MBWNoExp ic oc rd kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    nsCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w)
          (fun i => (R : ℝ) * (mbNoExpW_has_vjp (R * N) h w p hd hp).backward XIN DY i) r := by
  rw [nsCotIn_shard R hR N h w hN hh hw p hd hp XIN dys _ hdys r, ← nCotIn_eq_vjp,
    HasVJP.backward_smul]

theorem hdsCotIn_scaled {c oc nC : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC)
    (bfc : Vec nC) (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nC))
    (G : Vec ((R * N) * nC)) (hgs : ∀ r, gs r = batchShard R N nC (fun i => (R : ℝ) * G i) r)
    (r : Fin R) :
    hdsCotIn R hR N h w Wh bh εh γh βh Wfc XIN gs r
      = batchShard R N (c * h * w) (fun i => (R : ℝ) *
          (headFwdB_has_vjp (R * N) (h := h) (w := w) Wh bh εh hεh γh βh Wfc bfc).backward XIN G i)
          r := by
  rw [hdsCotIn_shard R hR N h w hN hh hw Wh bh εh hεh γh βh Wfc XIN gs _ hgs r,
    ← hdCotIn_eq_vjp (bfc := bfc), HasVJP.backward_smul]

end

-- ════════════════════════════════════════════════════════════════
-- § 6. The per-block DP ties — every parameter collective the render emits
--   Tags are the render's: each collective is named for its parameter (`{p}eW`, `{p}eg`, …),
--   and each `BnSync` reads the forward statistics `bnFwdSite` tags `{p}egmu` / `{p}egvar`.
-- ════════════════════════════════════════════════════════════════

/-- **The tail, DP-tied** — nine collectives: the depthwise BatchNorm's γ and β, the SE reduce and
    excite dense layers' weight and bias, the project conv weight, the project BatchNorm's γ and β.
    Each equals the single-device gradient node at the global batch, at T3's chain cotangents. -/
def tailSyncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc rd : Nat}
    (pfx xN cotN vN epsStr : String) (t : EnTail mid oc rd) (hp : 0 < t.pε)
    (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  BnSync R hR N mid h w s!"{pfx}dg" s!"{pfx}dbt" vN epsStr cotN t.dε DC
      (tsCotDn R hR N h w t DC dys) (tCotDn (R * N) h w t hp DC DY)
  ∧ DenseSync R hR N s!"{pfx}zW1" s!"{pfx}zb1" xN cotN (tS (R * N) h w t DC)
      (tsCotE1 R hR N h w t DC dys) (tCotE1 (R * N) h w t hp DC DY)
  ∧ DenseSync R hR N s!"{pfx}zW2" s!"{pfx}zb2" xN cotN (tZ (R * N) h w t DC)
      (tsCotE2 R hR N h w t DC dys) (tCotE2 (R * N) h w t hp DC DY)
  ∧ ConvWSync R hR N h w s!"{pfx}pW" xN cotN t.pb (tSe (R * N) h w t DC) t.pW
      (tsCotPbn R hR N h w t DC dys) (tCotPbn (R * N) h w t hp DC DY)
  ∧ BnSync R hR N oc h w s!"{pfx}pg" s!"{pfx}pbt" vN epsStr cotN t.pε (tPc (R * N) h w t DC) dys DY

theorem tail_syncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc rd : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (t : EnTail mid oc rd)
    (hp : 0 < t.pε) (DC : Vec ((R * N) * (mid * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    tailSyncTiedG R hR N h w pfx xN cotN vN epsStr t hp DC dys DY := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact bnSync_of_scaled R hR N mid h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [tsCotDn_shard R hR N h w hN hh hw t hp DC dys _ hdys r, tCotDn_smul])
  · exact denseSync_of_scaled R hR N _ _ _ _ _ _ _ (fun r => by
      rw [tsCotE1_shard R hR N h w hN hh hw t hp DC dys _ hdys r, tCotE1_smul])
  · exact denseSync_of_scaled R hR N _ _ _ _ _ _ _ (fun r => by
      rw [tsCotE2_shard R hR N h w hN hh hw t hp DC dys _ hdys r, tCotE2_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [tsCotPbn_shard R hR N h w hN hh hw t hp DC dys _ hdys r, tCotPbn_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ hdys

/-- **A stride-1 MBConv6 block, DP-tied** (the nine residual blocks and the two widenings) —
    thirteen collectives: the expand conv weight, the expand BatchNorm's γ and β, the depthwise
    weight, then the tail's nine. The body is the same with or without the identity skip; the skip
    lives in the cotangent thread (`rsCotIn`). -/
def expSyncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc rd kh kw : Nat}
    (pfx xN cotN vN epsStr : String) (p : MBW ic mid oc rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε)
    (hp : 0 < p.pε) (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  ConvWSync R hR N h w s!"{pfx}eW" xN cotN p.eb XIN p.eW
      (xsCotEc R hR N h w p XIN dys) (xCotEc (R * N) h w p he hd hp XIN DY)
  ∧ BnSync R hR N mid h w s!"{pfx}eg" s!"{pfx}ebt" vN epsStr cotN p.eε (xEc (R * N) h w p XIN)
      (xsCotEn R hR N h w p XIN dys) (xCotEn (R * N) h w p hd hp XIN DY)
  ∧ DepthwiseWSync R hR N h w s!"{pfx}dW" xN cotN p.db (xEr (R * N) h w p XIN) p.dW
      (tsCotDc R hR N h w (tailOf p) (xDc (R * N) h w p XIN) dys)
      (tCotDc (R * N) h w (tailOf p) hd hp (xDc (R * N) h w p XIN) DY)
  ∧ tailSyncTiedG R hR N h w pfx xN cotN vN epsStr (tailOf p) hp (xDc (R * N) h w p XIN) dys DY

theorem exp_syncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc rd kh kw : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String)
    (p : MBW ic mid oc rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    expSyncTiedG R hR N h w pfx xN cotN vN epsStr p he hd hp XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, tail_syncTiedG R hR N h w hN hh hw pfx xN cotN vN epsStr (tailOf p) hp _
    dys DY hdys⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [xsCotEc_shard R hR N h w hN hh hw p he hd hp XIN dys _ hdys r, xCotEc_smul])
  · exact bnSync_of_scaled R hR N mid h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [xsCotEn_shard R hR N h w hN hh hw p hd hp XIN dys _ hdys r, xCotEn_smul])
  · exact depthwiseWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [tsCotDc_shard R hR N h w hN hh hw (tailOf p) hd hp _ dys _ hdys r, tCotDc_smul])

/-- **A strided MBConv6 block, DP-tied** (b2, b4, b6, b12) — thirteen collectives, the expand pair
    at the input grid `2h×2w` and the depthwise strided. -/
def stridedSyncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc rd kh kw : Nat}
    (pfx xN cotN vN epsStr : String) (p : MBW ic mid oc rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε)
    (hp : 0 < p.pε) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  ConvWSync R hR N (2 * h) (2 * w) s!"{pfx}eW" xN cotN p.eb XIN p.eW
      (ssCotEc R hR N h w p XIN dys) (sCotEc (R * N) h w p he hd hp XIN DY)
  ∧ BnSync R hR N mid (2 * h) (2 * w) s!"{pfx}eg" s!"{pfx}ebt" vN epsStr cotN p.eε
      (sEc (R * N) h w p XIN) (ssCotEn R hR N h w p XIN dys) (sCotEn (R * N) h w p hd hp XIN DY)
  ∧ DepthwiseStridedWSync R hR N h w s!"{pfx}dW" xN cotN p.db (sEr (R * N) h w p XIN) p.dW
      (tsCotDc R hR N h w (tailOf p) (sDc (R * N) h w p XIN) dys)
      (tCotDc (R * N) h w (tailOf p) hd hp (sDc (R * N) h w p XIN) DY)
  ∧ tailSyncTiedG R hR N h w pfx xN cotN vN epsStr (tailOf p) hp (sDc (R * N) h w p XIN) dys DY

theorem strided_syncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc rd kh kw : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String)
    (p : MBW ic mid oc rd kh kw) (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    stridedSyncTiedG R hR N h w pfx xN cotN vN epsStr p he hd hp XIN dys DY := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  refine ⟨?_, ?_, ?_, tail_syncTiedG R hR N h w hN hh hw pfx xN cotN vN epsStr (tailOf p) hp _
    dys DY hdys⟩
  · exact convWSync_of_scaled R hR N (2 * h) (2 * w) _ _ _ _ _ _ _ _ (fun r => by
      rw [ssCotEc_shard R hR N h w hN hh hw p he hd hp XIN dys _ hdys r, sCotEc_smul])
  · exact bnSync_of_scaled R hR N mid (2 * h) (2 * w) (nhw_ne_zero hN h2h h2w) _ _ _ _ _ _ _ _ _ (fun r => by
      rw [ssCotEn_shard R hR N h w hN hh hw p hd hp XIN dys _ hdys r, sCotEn_smul])
  · exact depthwiseStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [tsCotDc_shard R hR N h w hN hh hw (tailOf p) hd hp _ dys _ hdys r, tCotDc_smul])

/-- **The MBConv1 block, DP-tied** (b1) — ten collectives: the depthwise weight, then the tail's
    nine. -/
def noExpSyncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc rd kh kw : Nat}
    (pfx xN cotN vN epsStr : String) (p : MBWNoExp ic oc rd kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  DepthwiseWSync R hR N h w s!"{pfx}dW" xN cotN p.db XIN p.dW
      (tsCotDc R hR N h w (tailOfNoExp p) (nDc (R * N) h w p XIN) dys)
      (tCotDc (R * N) h w (tailOfNoExp p) hd hp (nDc (R * N) h w p XIN) DY)
  ∧ tailSyncTiedG R hR N h w pfx xN cotN vN epsStr (tailOfNoExp p) hp (nDc (R * N) h w p XIN)
      dys DY

theorem noExp_syncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc rd kh kw : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String)
    (p : MBWNoExp ic oc rd kh kw) (hd : 0 < p.dε) (hp : 0 < p.pε)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    noExpSyncTiedG R hR N h w pfx xN cotN vN epsStr p hd hp XIN dys DY :=
  ⟨depthwiseWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [tsCotDc_shard R hR N h w hN hh hw (tailOfNoExp p) hd hp _ dys _ hdys r, tCotDc_smul]),
    tail_syncTiedG R hR N h w hN hh hw pfx xN cotN vN epsStr (tailOfNoExp p) hp _ dys DY hdys⟩

/-- **The stem, DP-tied** — three collectives: the 3×3/s2 XLA-`SAME` conv weight and its
    BatchNorm's γ and β. -/
def stemSyncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kHs kWs : Nat}
    (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic kHs kWs) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs)
    (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  ConvStridedXlaWSync R hR N h w "sW" xN cotN bs X Ws
      (stsCotStc R hR N h w Ws bs εs γs βs X dys) (stCotStc (R * N) h w Ws bs εs hεs γs βs X DY)
  ∧ BnSync R hR N oc h w "sg" "sbt" vN epsStr cotN εs (stStc (R * N) h w Ws bs X)
      (stsCotBnS R N h w Ws bs εs γs βs X dys) (stCotBnS (R * N) h w Ws bs εs γs βs X DY)

theorem stem_syncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kHs kWs : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic kHs kWs)
    (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    stemSyncTiedG R hR N h w xN cotN vN epsStr Ws bs εs hεs γs βs X dys DY :=
  ⟨convStridedXlaWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [stsCotStc_shard R hR N h w hN hh hw Ws bs εs hεs γs βs X dys _ hdys r, stCotStc_smul]),
    bnSync_of_scaled R hR N oc h w (nhw_ne_zero hN hh hw)
      _ _ _ _ _ _ _ _ _ (fun r => by
      rw [stsCotBnS_shard R N h w Ws bs εs γs βs X dys _ hdys r, stCotBnS_smul])⟩

/-- **The head, DP-tied** — five collectives: the 1×1 conv weight, its BatchNorm's γ and β, and the
    classifier's weight and bias, the last two at the loss cotangent itself. -/
def headSyncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {c oc nC : Nat}
    (xN cotN vN epsStr dN : String) (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh)
    (γh βh : Vec oc) (Wfc : Mat oc nC) (XIN : Vec ((R * N) * (c * h * w)))
    (gs : Fin R → Vec (N * nC)) (G : Vec ((R * N) * nC)) : Prop :=
  ConvWSync R hR N h w "hW" xN cotN bh XIN Wh
      (hdsCotHbn R hR N h w Wh bh εh γh βh Wfc XIN gs)
      (hdCotHbn (R * N) h w Wh bh εh hεh γh βh Wfc XIN G)
  ∧ BnSync R hR N oc h w "hg" "hbt" vN epsStr cotN εh (hdHc (R * N) h w Wh bh XIN)
      (hdsCotHsw R N h w Wh bh εh γh βh Wfc XIN gs) (hdCotHsw (R * N) h w Wh bh εh γh βh Wfc XIN G)
  ∧ DenseSync R hR N "Wd" "bd" dN cotN (hdGap (R * N) h w Wh bh εh γh βh XIN) gs G

theorem head_syncTiedG (R : Nat) (hR : 0 < R) (N h w : Nat) {c oc nC : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (xN cotN vN epsStr dN : String) (Wh : Kernel4 oc c 1 1)
    (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc) (Wfc : Mat oc nC)
    (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nC)) (G : Vec ((R * N) * nC))
    (hgs : ∀ r, gs r = batchShard R N nC (fun i => (R : ℝ) * G i) r) :
    headSyncTiedG R hR N h w xN cotN vN epsStr dN Wh bh εh hεh γh βh Wfc XIN gs G :=
  ⟨convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [hdsCotHbn_shard R hR N h w hN hh hw Wh bh εh hεh γh βh Wfc XIN gs _ hgs r,
        hdCotHbn_smul]),
    bnSync_of_scaled R hR N oc h w (nhw_ne_zero hN hh hw)
      _ _ _ _ _ _ _ _ _ (fun r => by
      rw [hdsCotHsw_shard R N h w Wh bh εh γh βh Wfc XIN gs _ hgs r, hdCotHsw_smul]),
    denseSync_of_scaled R hR N _ _ _ _ _ gs G hgs⟩

-- ════════════════════════════════════════════════════════════════
-- § 7. The whole-net capstone
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐⭐ **The synchronised-BN data-parallel EfficientNet-B0 step IS the single-device step at the
    global batch.** `R` replicas at batch `N`, each dividing its loss by `B`, each running the
    render's sync-BN backward chain from its own label-smoothed cotangent; every parameter's
    all-reduced mean gradient — stem 3, b1 10, fifteen MBConv6 blocks × 13, head 5: the 213 the
    render emits at `convBias := false` — equals the single-device batch-BN gradient node at batch
    `R·N`, loss divided by `R·B`, at the cotangent T3's chain delivers there.

    The right-hand chain is `EnetTiePoCG.efficientnet_net_tiedG`'s at `N := R·N`, `B := R·B`:
    verbatim for the forward prefixes `a0 … a16`, the loss cotangent `g` and the block-output
    cotangents `dy16 … dy0` threaded by the certified block VJPs' `.backward`; by `rfl` for the
    in-block cotangents, which are this file's named chain. That capstone ties those nodes to the
    certified gradient, so the two together say the DP step's update is the certified gradient of
    the mean loss over all `R·N` examples. It carries the same fifty `0 < ε` hypotheses, because
    T3's single-device chain does.

    ⭐ The left-hand chain is the replicas' own: sync-BN backward at every one of the 49
    BatchNorms (`bnSyncInB`, a collective each), per-example conv / depthwise / squeeze-excite /
    swish / head links, each replica's own loss cotangent. -/
theorem efficientnet_net_syncTiedG (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (xN vN epsStr cotN dN : String) (w : B0Weights)
    (hεw : w.EpsPos)
    (aStr negAK bStr logN ohN : String) (α B : ℝ)
    (x : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * (1 * 10))) :
    -- ── the single-device step at the global batch `R·N`, loss divided by `R·B`: T3's chain ──
    let a0  : Vec ((R * N) * (32 * 112 * 112)) :=
      stemB (R * N) (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ x
    let a1  : Vec ((R * N) * (16 * 112 * 112)) := mbNoExpW (R * N) 112 112 w.b1 a0
    let a2  : Vec ((R * N) * (24 * 56 * 56))   := mbStridedW (R * N) 56 56 w.b2 a1
    let a3  : Vec ((R * N) * (24 * 56 * 56))   := mbResidW (R * N) 56 56 w.b3 a2
    let a4  : Vec ((R * N) * (40 * 28 * 28))   := mbStridedW (R * N) 28 28 w.b4 a3
    let a5  : Vec ((R * N) * (40 * 28 * 28))   := mbResidW (R * N) 28 28 w.b5 a4
    let a6  : Vec ((R * N) * (80 * 14 * 14))   := mbStridedW (R * N) 14 14 w.b6 a5
    let a7  : Vec ((R * N) * (80 * 14 * 14))   := mbResidW (R * N) 14 14 w.b7 a6
    let a8  : Vec ((R * N) * (80 * 14 * 14))   := mbResidW (R * N) 14 14 w.b8 a7
    let a9  : Vec ((R * N) * (112 * 14 * 14))  := mbExpW (R * N) 14 14 w.b9 a8
    let a10 : Vec ((R * N) * (112 * 14 * 14))  := mbResidW (R * N) 14 14 w.b10 a9
    let a11 : Vec ((R * N) * (112 * 14 * 14))  := mbResidW (R * N) 14 14 w.b11 a10
    let a12 : Vec ((R * N) * (192 * 7 * 7))    := mbStridedW (R * N) 7 7 w.b12 a11
    let a13 : Vec ((R * N) * (192 * 7 * 7))    := mbResidW (R * N) 7 7 w.b13 a12
    let a14 : Vec ((R * N) * (192 * 7 * 7))    := mbResidW (R * N) 7 7 w.b14 a13
    let a15 : Vec ((R * N) * (192 * 7 * 7))    := mbResidW (R * N) 7 7 w.b15 a14
    let a16 : Vec ((R * N) * (320 * 7 * 7))    := mbExpW (R * N) 7 7 w.b16 a15
    let g    : Vec ((R * N) * 10) :=
      unrowB (R * N) 10 (den (smoothedLossCotGraph (R * N) 10 α ((R : ℝ) * B) aStr negAK bStr logN
        ohN (rowB (R * N) 10
          (headFwdB (R * N) (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb a16)) t))
    let dy16 : Vec ((R * N) * (320 * 7 * 7))   := (headFwdB_has_vjp (R * N) (h := 7) (w := 7)
      w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb).backward a16 g
    let dy15 : Vec ((R * N) * (192 * 7 * 7))   :=
      (mbExpW_has_vjp (R * N) 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p).backward a15 dy16
    let dy14 : Vec ((R * N) * (192 * 7 * 7))   :=
      (mbResidW_has_vjp (R * N) 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p).backward a14 dy15
    let dy13 : Vec ((R * N) * (192 * 7 * 7))   :=
      (mbResidW_has_vjp (R * N) 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p).backward a13 dy14
    let dy12 : Vec ((R * N) * (192 * 7 * 7))   :=
      (mbResidW_has_vjp (R * N) 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p).backward a12 dy13
    let dy11 : Vec ((R * N) * (112 * 14 * 14)) :=
      (mbStridedW_has_vjp (R * N) 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p).backward a11 dy12
    let dy10 : Vec ((R * N) * (112 * 14 * 14)) :=
      (mbResidW_has_vjp (R * N) 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p).backward a10 dy11
    let dy9  : Vec ((R * N) * (112 * 14 * 14)) :=
      (mbResidW_has_vjp (R * N) 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p).backward a9 dy10
    let dy8  : Vec ((R * N) * (80 * 14 * 14))  :=
      (mbExpW_has_vjp (R * N) 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p).backward a8 dy9
    let dy7  : Vec ((R * N) * (80 * 14 * 14))  :=
      (mbResidW_has_vjp (R * N) 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p).backward a7 dy8
    let dy6  : Vec ((R * N) * (80 * 14 * 14))  :=
      (mbResidW_has_vjp (R * N) 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p).backward a6 dy7
    let dy5  : Vec ((R * N) * (40 * 28 * 28))  :=
      (mbStridedW_has_vjp (R * N) 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p).backward a5 dy6
    let dy4  : Vec ((R * N) * (40 * 28 * 28))  :=
      (mbResidW_has_vjp (R * N) 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p).backward a4 dy5
    let dy3  : Vec ((R * N) * (24 * 56 * 56))  :=
      (mbStridedW_has_vjp (R * N) 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p).backward a3 dy4
    let dy2  : Vec ((R * N) * (24 * 56 * 56))  :=
      (mbResidW_has_vjp (R * N) 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p).backward a2 dy3
    let dy1  : Vec ((R * N) * (16 * 112 * 112)) :=
      (mbStridedW_has_vjp (R * N) 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p).backward a1 dy2
    let dy0  : Vec ((R * N) * (32 * 112 * 112)) :=
      (mbNoExpW_has_vjp (R * N) 112 112 w.b1 hεw.b1.d hεw.b1.p).backward a0 dy1
    -- ── replica `r`, loss divided by `B`, its own sync-BN chain ──
    let gs : Fin R → Vec (N * 10) := fun r =>
      unrowB N 10 (den (smoothedLossCotGraph N 10 α B aStr negAK bStr logN ohN
        (rowB N 10 (batchShard R N 10
          (headFwdB (R * N) (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb a16) r))
        (batchShard R N (1 * 10) t r)))
    let e16 := hdsCotIn R hR N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW a16 gs
    let e15 := xsCotIn R hR N 7 7 w.b16 a15 e16
    let e14 := rsCotIn R hR N 7 7 w.b15 a14 e15
    let e13 := rsCotIn R hR N 7 7 w.b14 a13 e14
    let e12 := rsCotIn R hR N 7 7 w.b13 a12 e13
    let e11 := ssCotIn R hR N 7 7 w.b12 a11 e12
    let e10 := rsCotIn R hR N 14 14 w.b11 a10 e11
    let e9  := rsCotIn R hR N 14 14 w.b10 a9 e10
    let e8  := xsCotIn R hR N 14 14 w.b9 a8 e9
    let e7  := rsCotIn R hR N 14 14 w.b8 a7 e8
    let e6  := rsCotIn R hR N 14 14 w.b7 a6 e7
    let e5  := ssCotIn R hR N 14 14 w.b6 a5 e6
    let e4  := rsCotIn R hR N 28 28 w.b5 a4 e5
    let e3  := ssCotIn R hR N 28 28 w.b4 a3 e4
    let e2  := rsCotIn R hR N 56 56 w.b3 a2 e3
    let e1  := ssCotIn R hR N 56 56 w.b2 a1 e2
    let e0  := nsCotIn R hR N 112 112 w.b1 a0 e1
    -- ── every collective the render emits IS the single-device node ──
    stemSyncTiedG R hR N 112 112 xN cotN vN epsStr w.sW w.sb w.sε hεw.s w.sγ w.sβ x e0 dy0
    ∧ noExpSyncTiedG R hR N 112 112 "b1" xN cotN vN epsStr w.b1 hεw.b1.d hεw.b1.p a0 e1 dy1
    ∧ stridedSyncTiedG R hR N 56 56 "b2" xN cotN vN epsStr w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 e2 dy2
    ∧ expSyncTiedG R hR N 56 56 "b3" xN cotN vN epsStr w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 e3 dy3
    ∧ stridedSyncTiedG R hR N 28 28 "b4" xN cotN vN epsStr w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 e4 dy4
    ∧ expSyncTiedG R hR N 28 28 "b5" xN cotN vN epsStr w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 e5 dy5
    ∧ stridedSyncTiedG R hR N 14 14 "b6" xN cotN vN epsStr w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 e6 dy6
    ∧ expSyncTiedG R hR N 14 14 "b7" xN cotN vN epsStr w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 e7 dy7
    ∧ expSyncTiedG R hR N 14 14 "b8" xN cotN vN epsStr w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 e8 dy8
    ∧ expSyncTiedG R hR N 14 14 "b9" xN cotN vN epsStr w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 e9 dy9
    ∧ expSyncTiedG R hR N 14 14 "b10" xN cotN vN epsStr w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 e10 dy10
    ∧ expSyncTiedG R hR N 14 14 "b11" xN cotN vN epsStr w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 e11 dy11
    ∧ stridedSyncTiedG R hR N 7 7 "b12" xN cotN vN epsStr w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 e12 dy12
    ∧ expSyncTiedG R hR N 7 7 "b13" xN cotN vN epsStr w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 e13 dy13
    ∧ expSyncTiedG R hR N 7 7 "b14" xN cotN vN epsStr w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 e14 dy14
    ∧ expSyncTiedG R hR N 7 7 "b15" xN cotN vN epsStr w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 e15 dy15
    ∧ expSyncTiedG R hR N 7 7 "b16" xN cotN vN epsStr w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 e16 dy16
    ∧ headSyncTiedG R hR N 7 7 xN cotN vN epsStr dN w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW a16 gs g := by
  intro a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16
    g dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dy0
    gs e16 e15 e14 e13 e12 e11 e10 e9 e8 e7 e6 e5 e4 e3 e2 e1 e0
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  -- the divisor: each replica's loss cotangent is `R ×` its shard of `g`
  have sG : ∀ r, gs r = batchShard R N 10 (fun i => (R : ℝ) * g i) r :=
    fun r => replicaLossCot_eq R N 10 hR α B aStr negAK bStr logN ohN _ t r
  -- the scaled-shard invariant, block by block down the chain
  have s16 := hdsCotIn_scaled R hR N 7 7 hN h7 h7 w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb a16 gs g sG
  have s15 := xsCotIn_scaled R hR N 7 7 hN h7 h7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 e16 dy16 s16
  have s14 := rsCotIn_scaled R hR N 7 7 hN h7 h7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 e15 dy15 s15
  have s13 := rsCotIn_scaled R hR N 7 7 hN h7 h7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 e14 dy14 s14
  have s12 := rsCotIn_scaled R hR N 7 7 hN h7 h7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 e13 dy13 s13
  have s11 := ssCotIn_scaled R hR N 7 7 hN h7 h7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 e12 dy12 s12
  have s10 := rsCotIn_scaled R hR N 14 14 hN h14 h14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 e11 dy11 s11
  have s9 := rsCotIn_scaled R hR N 14 14 hN h14 h14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 e10 dy10 s10
  have s8 := xsCotIn_scaled R hR N 14 14 hN h14 h14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 e9 dy9 s9
  have s7 := rsCotIn_scaled R hR N 14 14 hN h14 h14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 e8 dy8 s8
  have s6 := rsCotIn_scaled R hR N 14 14 hN h14 h14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 e7 dy7 s7
  have s5 := ssCotIn_scaled R hR N 14 14 hN h14 h14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 e6 dy6 s6
  have s4 := rsCotIn_scaled R hR N 28 28 hN h28 h28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 e5 dy5 s5
  have s3 := ssCotIn_scaled R hR N 28 28 hN h28 h28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 e4 dy4 s4
  have s2 := rsCotIn_scaled R hR N 56 56 hN h56 h56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 e3 dy3 s3
  have s1 := ssCotIn_scaled R hR N 56 56 hN h56 h56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 e2 dy2 s2
  have s0 := nsCotIn_scaled R hR N 112 112 hN h112 h112 w.b1 hεw.b1.d hεw.b1.p a0 e1 dy1 s1
  exact ⟨stem_syncTiedG R hR N 112 112 hN h112 h112 xN cotN vN epsStr w.sW w.sb w.sε hεw.s w.sγ w.sβ
      x e0 dy0 s0,
    noExp_syncTiedG R hR N 112 112 hN h112 h112 "b1" xN cotN vN epsStr w.b1 hεw.b1.d hεw.b1.p a0 e1 dy1 s1,
    strided_syncTiedG R hR N 56 56 hN h56 h56 "b2" xN cotN vN epsStr w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 e2 dy2 s2,
    exp_syncTiedG R hR N 56 56 hN h56 h56 "b3" xN cotN vN epsStr w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 e3 dy3 s3,
    strided_syncTiedG R hR N 28 28 hN h28 h28 "b4" xN cotN vN epsStr w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 e4 dy4 s4,
    exp_syncTiedG R hR N 28 28 hN h28 h28 "b5" xN cotN vN epsStr w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 e5 dy5 s5,
    strided_syncTiedG R hR N 14 14 hN h14 h14 "b6" xN cotN vN epsStr w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 e6 dy6 s6,
    exp_syncTiedG R hR N 14 14 hN h14 h14 "b7" xN cotN vN epsStr w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 e7 dy7 s7,
    exp_syncTiedG R hR N 14 14 hN h14 h14 "b8" xN cotN vN epsStr w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 e8 dy8 s8,
    exp_syncTiedG R hR N 14 14 hN h14 h14 "b9" xN cotN vN epsStr w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 e9 dy9 s9,
    exp_syncTiedG R hR N 14 14 hN h14 h14 "b10" xN cotN vN epsStr w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 e10
      dy10 s10,
    exp_syncTiedG R hR N 14 14 hN h14 h14 "b11" xN cotN vN epsStr w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 e11
      dy11 s11,
    strided_syncTiedG R hR N 7 7 hN h7 h7 "b12" xN cotN vN epsStr w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 e12
      dy12 s12,
    exp_syncTiedG R hR N 7 7 hN h7 h7 "b13" xN cotN vN epsStr w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 e13 dy13 s13,
    exp_syncTiedG R hR N 7 7 hN h7 h7 "b14" xN cotN vN epsStr w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 e14 dy14 s14,
    exp_syncTiedG R hR N 7 7 hN h7 h7 "b15" xN cotN vN epsStr w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 e15 dy15 s15,
    exp_syncTiedG R hR N 7 7 hN h7 h7 "b16" xN cotN vN epsStr w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 e16 dy16 s16,
    head_syncTiedG R hR N 7 7 hN h7 h7 xN cotN vN epsStr dN w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW a16
      gs g sG⟩

end Proofs.EnetSyncTieG
