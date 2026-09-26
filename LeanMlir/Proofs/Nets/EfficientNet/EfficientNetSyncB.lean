import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0
import LeanMlir.Proofs.Foundation.DataParallelSyncKit

/-! # EfficientNet-B0's data-parallel forward at SYNCHRONISED BatchNorm — replica `r` IS shard `r`

`EfficientNetFullB0.lean` says the typed batch-BN graph denotes `efficientnetForwardBFull N w`
on one device. At `replicas > 1` the data-parallel render normalises every one of B0's 49
BatchNorms with the GLOBAL batch's statistics — the sync-BN composition `bnFwdSite` emits: this
replica's mean all-reduced, then Chan's `σ²_r + (μ_r − μ)²` all-reduced, packed, then `bnSyncF`.
This file is its data-parallel twin: that forward graph, stated as a family over the `R`
replicas, denotes on replica `r` exactly `batchShard r` of the single-device forward at the global
batch `R·N`.

    den (efficientnetFwdGraphSyncFull R hR N epsStr w e r)
      = batchShard R N 10 (efficientnetForwardBFull (R * N) w X) r

given that each replica's input is its shard of one global batch `X`. **The spec does not move**:
the right-hand side is the committed `efficientnetForwardBFull`, at `N := R·N`.

## How it is proved

ResNet-34's recipe (`ResNet34SyncB.lean`), block by block, the shard hypothesis
`∀ r, den (e r) = batchShard R N _ X r` carried from each block into the next:

* every conv, depthwise, strided depthwise, squeeze-excite, GAP and dense node is a per-example
  lift and commutes with sharding (`den_batchOp_shard`) — the SE block included: its squeeze
  (GAP), both dense layers and the gate multiply all act on one example at a time, so it is one
  `batchMap` and needs nothing new;
* swish is pointwise (`den_swishF_shard`), and so is the residual `addV` (`den_addV_shard`);
* every BatchNorm is `bnSyncSiteLA`, whose shard lemma `den_bnSyncSiteLA` is P1 on the graph.

The collectives are tagged as `bnFwdSite` tags them: a BN site whose γ is `%{p}eg` gathers
`{p}egmu` and `{p}egvar`, each over a `[c]` statistic. `den` ignores every string, so the tags fix
which graph is stated, not what it denotes.

## What is NOT claimed here

The backward and the parameter collectives are the step half (`EfficientNetSyncStepTieG.lean`).
That the `R` replicas' inputs are the shards of one batch is the driver's. The emitted
artifacts' stochastic-depth and classifier-dropout variants add per-example scalings this file
does not state. The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The pointwise nodes B0 adds to ResNet-34's kit
-- ════════════════════════════════════════════════════════════════

/-- Swish on every replica denotes the shard of the global swish — pointwise, like relu. -/
theorem den_swishF_shard {R N n : Nat} (e : Fin R → SHlo (N * n)) (X : Vec ((R * N) * n))
    (he : ∀ r, den (e r) = batchShard R N n X r) (r : Fin R) :
    den (.swishF (e r)) = batchShard R N n (swish ((R * N) * n) X) r := by
  rw [swishF_faithful, he]
  rfl

/-- The MBConv identity skip (`addV`) on every replica is the shard of the global one. -/
theorem den_addV_shard {R N n : Nat} (a b : Fin R → SHlo (N * n)) (A B : Vec ((R * N) * n))
    (ha : ∀ r, den (a r) = batchShard R N n A r) (hb : ∀ r, den (b r) = batchShard R N n B r)
    (r : Fin R) :
    den (.addV (a r) (b r)) = batchShard R N n (fun j => A j + B j) r := by
  rw [den_addV, ha, hb]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Per-block replica families + their shard lemmas
--   Names are `EfficientNetFullB0`'s graph names; the collective tags are the render's
--   (`{p}eg` → `{p}egmu` / `{p}egvar`).
-- ════════════════════════════════════════════════════════════════

/-- MBConv1 (b1, no expand) at sync-BN, over the replica family: depthwise → sync-BN → swish → SE →
    1×1 project → sync-BN. -/
def mbNoExpGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc rd kh kw : Nat} (q : MBWNoExp ic oc rd kh kw) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%{p}pg" s!"%{p}pbt" epsStr s!"{p}pgmu" s!"{p}pgvar" [oc] [oc] R hR
    q.pε q.pγ q.pβ
    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" q.pW q.pb)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          q.z1 q.zb1 q.z2 q.zb2)
        (.swishF (bnSyncSiteLA s!"%{p}dg" s!"%{p}dbt" epsStr s!"{p}dgmu" s!"{p}dgvar" [ic] [ic] R hR
          q.dε q.dγ q.dβ
          (fun r => .batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" q.dW q.db)
            (e r)) r)))) r

theorem mbNoExpGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (q : MBWNoExp ic oc rd kh kw)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (mbNoExpGraphSync p epsStr R hR N h w q e r)
      = batchShard R N (oc * h * w) (mbNoExpW (R * N) h w q X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hdc := den_batchOp_shard (N := N)
    (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" q.dW q.db) e X he
  have hdn := den_bnSyncSiteLA s!"%{p}dg" s!"%{p}dbt" epsStr s!"{p}dgmu" s!"{p}dgvar" [ic] [ic]
    R hR hm q.dε q.dγ q.dβ _ _ hdc
  have hdr := den_swishF_shard _ _ hdn
  have hse := den_batchOp_shard (N := N)
    (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb" q.z1 q.zb1 q.z2 q.zb2)
    _ _ hdr
  have hpc := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" q.pW q.pb) _ _ hse
  exact den_bnSyncSiteLA s!"%{p}pg" s!"%{p}pbt" epsStr s!"{p}pgmu" s!"{p}pgvar" [oc] [oc]
    R hR hm q.pε q.pγ q.pβ _ _ hpc r

/-- The MBConv6 body at sync-BN — expand 1×1 → sync-BN → swish → depthwise → sync-BN → swish →
    SE → project 1×1 → sync-BN — shared by the residual (b3, b5, …) and the no-skip widening
    (b9, b16) blocks. -/
def mbBodyGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc rd kh kw : Nat} (q : MBW ic mid oc rd kh kw) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%{p}pg" s!"%{p}pbt" epsStr s!"{p}pgmu" s!"{p}pgvar" [oc] [oc] R hR
    q.pε q.pγ q.pβ
    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" q.pW q.pb)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          q.z1 q.zb1 q.z2 q.zb2)
        (.swishF (bnSyncSiteLA s!"%{p}dg" s!"%{p}dbt" epsStr s!"{p}dgmu" s!"{p}dgvar" [mid] [mid]
          R hR q.dε q.dγ q.dβ
          (fun r => .batchOp (N := N) (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" q.dW q.db)
            (.swishF (bnSyncSiteLA s!"%{p}eg" s!"%{p}ebt" epsStr s!"{p}egmu" s!"{p}egvar" [mid] [mid]
              R hR q.eε q.eγ q.eβ
              (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}eW" s!"%{p}eb" q.eW q.eb)
                (e r)) r))) r)))) r

theorem mbBodyGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (q : MBW ic mid oc rd kh kw)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (mbBodyGraphSync p epsStr R hR N h w q e r)
      = batchShard R N (oc * h * w) (mbExpW (R * N) h w q X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hec := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}eW" s!"%{p}eb" q.eW q.eb) e X he
  have hen := den_bnSyncSiteLA s!"%{p}eg" s!"%{p}ebt" epsStr s!"{p}egmu" s!"{p}egvar" [mid] [mid]
    R hR hm q.eε q.eγ q.eβ _ _ hec
  have her := den_swishF_shard _ _ hen
  have hdc := den_batchOp_shard (N := N)
    (.depthwise (h := h) (w := w) s!"%{p}dW" s!"%{p}db" q.dW q.db) _ _ her
  have hdn := den_bnSyncSiteLA s!"%{p}dg" s!"%{p}dbt" epsStr s!"{p}dgmu" s!"{p}dgvar" [mid] [mid]
    R hR hm q.dε q.dγ q.dβ _ _ hdc
  have hdr := den_swishF_shard _ _ hdn
  have hse := den_batchOp_shard (N := N)
    (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb" q.z1 q.zb1 q.z2 q.zb2)
    _ _ hdr
  have hpc := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" q.pW q.pb) _ _ hse
  exact den_bnSyncSiteLA s!"%{p}pg" s!"%{p}pbt" epsStr s!"{p}pgmu" s!"{p}pgvar" [oc] [oc]
    R hR hm q.pε q.pγ q.pβ _ _ hpc r

/-- The no-skip widening block (b9, b16) is the body alone. -/
def mbExpGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc rd kh kw : Nat} (q : MBW ic mid oc rd kh kw) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  mbBodyGraphSync p epsStr R hR N h w q e

theorem mbExpGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (q : MBW ic mid oc rd kh kw)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (mbExpGraphSync p epsStr R hR N h w q e r)
      = batchShard R N (oc * h * w) (mbExpW (R * N) h w q X) r :=
  mbBodyGraphSync_shard p epsStr R hR N h w hN hh hw q e X he r

/-- The residual MBConv6 block (b3, b5, b7, b8, b10, b11, b13–b15): the body plus the identity skip,
    `addV body e` — body first, the order `mbResidGraphB` uses. -/
def mbResidGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {c mid rd kh kw : Nat} (q : MBW c mid c rd kh kw) (e : Fin R → SHlo (N * (c * h * w))) :
    Fin R → SHlo (N * (c * h * w)) :=
  fun r => .addV (mbBodyGraphSync p epsStr R hR N h w q e r) (e r)

theorem mbResidGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {c mid rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (q : MBW c mid c rd kh kw)
    (e : Fin R → SHlo (N * (c * h * w))) (X : Vec ((R * N) * (c * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r) (r : Fin R) :
    den (mbResidGraphSync p epsStr R hR N h w q e r)
      = batchShard R N (c * h * w) (mbResidW (R * N) h w q X) r :=
  den_addV_shard _ e _ X (mbBodyGraphSync_shard p epsStr R hR N h w hN hh hw q e X he) he r

/-- The strided MBConv6 block (b2, b4, b6, b12): expand at the input grid `2h×2w`, the strided
    depthwise down to `h×w`, SE and project there. Three sync sites, the expand one at `2h×2w`. -/
def mbStridedGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc rd kh kw : Nat} (q : MBW ic mid oc rd kh kw)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%{p}pg" s!"%{p}pbt" epsStr s!"{p}pgmu" s!"{p}pgvar" [oc] [oc] R hR
    q.pε q.pγ q.pβ
    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" q.pW q.pb)
      (.batchOp (N := N) (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb"
          q.z1 q.zb1 q.z2 q.zb2)
        (.swishF (bnSyncSiteLA s!"%{p}dg" s!"%{p}dbt" epsStr s!"{p}dgmu" s!"{p}dgvar" [mid] [mid]
          R hR q.dε q.dγ q.dβ
          (fun r => .batchOp (N := N)
            (.depthwiseStrided (h := h) (w := w) s!"%{p}dW" s!"%{p}db" q.dW q.db)
            (.swishF (bnSyncSiteLA s!"%{p}eg" s!"%{p}ebt" epsStr s!"{p}egmu" s!"{p}egvar" [mid] [mid]
              R hR q.eε q.eγ q.eβ
              (fun r => .batchOp (N := N)
                (.conv (h := 2 * h) (w := 2 * w) s!"%{p}eW" s!"%{p}eb" q.eW q.eb) (e r)) r))) r)))) r

theorem mbStridedGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc rd kh kw : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (q : MBW ic mid oc rd kh kw)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (mbStridedGraphSync p epsStr R hR N h w q e r)
      = batchShard R N (oc * h * w) (mbStridedW (R * N) h w q X) r := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm2 := nhw_ne_zero hN h2h h2w
  have hm := nhw_ne_zero hN hh hw
  have hec := den_batchOp_shard (N := N)
    (.conv (h := 2 * h) (w := 2 * w) s!"%{p}eW" s!"%{p}eb" q.eW q.eb) e X he
  have hen := den_bnSyncSiteLA s!"%{p}eg" s!"%{p}ebt" epsStr s!"{p}egmu" s!"{p}egvar" [mid] [mid]
    R hR hm2 q.eε q.eγ q.eβ _ _ hec
  have her := den_swishF_shard _ _ hen
  have hdc := den_batchOp_shard (N := N)
    (.depthwiseStrided (h := h) (w := w) s!"%{p}dW" s!"%{p}db" q.dW q.db) _ _ her
  have hdn := den_bnSyncSiteLA s!"%{p}dg" s!"%{p}dbt" epsStr s!"{p}dgmu" s!"{p}dgvar" [mid] [mid]
    R hR hm q.dε q.dγ q.dβ _ _ hdc
  have hdr := den_swishF_shard _ _ hdn
  have hse := den_batchOp_shard (N := N)
    (.seBlock (h := h) (w := w) s!"%{p}zWa" s!"%{p}zba" s!"%{p}zWb" s!"%{p}zbb" q.z1 q.zb1 q.z2 q.zb2)
    _ _ hdr
  have hpc := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}pW" s!"%{p}pb" q.pW q.pb) _ _ hse
  exact den_bnSyncSiteLA s!"%{p}pg" s!"%{p}pbt" epsStr s!"{p}pgmu" s!"{p}pgvar" [oc] [oc]
    R hR hm q.pε q.pγ q.pβ _ _ hpc r

/-- Stem at sync-BN, over the replica family: 3×3/s2 conv (XLA-`SAME`) → sync-BN → swish. -/
def stemGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .swishF (bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc] R hR εs γs βs
    (fun r => .batchOp (N := N) (.convStridedXla (h := h) (w := w) "%sW" "%sb" Ws bs) (e r)) r)

theorem stemGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (stemGraphSync epsStr R hR N h w Ws bs εs γs βs e r)
      = batchShard R N (oc * h * w) (stemB (R * N) (h := h) (w := w) Ws bs εs γs βs X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N)
    (.convStridedXla (h := h) (w := w) "%sW" "%sb" Ws bs) e X he
  have hn := den_bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc] R hR hm εs γs βs _ _ hc
  exact den_swishF_shard _ _ hn r

/-- Head at sync-BN, over the replica family: 1×1 conv → sync-BN → swish → GAP → dense. -/
def headGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {c oc nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : Fin R → SHlo (N * (c * h * w))) : Fin R → SHlo (N * nC) :=
  fun r => .batchOp (N := N) (.dense "%Wfc" "%bfc" Wfc bfc)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.swishF (bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR εh γh βh
        (fun r => .batchOp (N := N) (.conv (h := h) (w := w) "%hW" "%hb" Wh bh) (e r)) r)))

theorem headGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {c oc nC : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (e : Fin R → SHlo (N * (c * h * w))) (X : Vec ((R * N) * (c * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r) (r : Fin R) :
    den (headGraphSync epsStr R hR N h w Wh bh εh γh βh Wfc bfc e r)
      = batchShard R N nC (headFwdB (R * N) (h := h) (w := w) Wh bh εh γh βh Wfc bfc X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N) (.conv (h := h) (w := w) "%hW" "%hb" Wh bh) e X he
  have hn := den_bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR hm εh γh βh _ _ hc
  have hr := den_swishF_shard _ _ hn
  have hg := den_batchOp_shard (N := N) (.gap (c := oc) (h := h) (w := w)) _ _ hr
  exact den_batchOp_shard (N := N) (.dense "%Wfc" "%bfc" Wfc bfc) _ _ hg r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel EfficientNet-B0 forward graph, over the replica family.** The
    single-device `efficientnetFwdGraphBFull` with every one of the 49 BatchNorms a `bnSyncSiteLA`
    over all `R` replicas, fed each replica's own input subgraph `e r`; block prefixes and
    collective tags are `EfficientNetRender`'s. -/
def efficientnetFwdGraphSyncFull (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String)
    (w : B0Weights) (e : Fin R → SHlo (N * (3 * 224 * 224))) : Fin R → SHlo (N * 10) :=
  headGraphSync epsStr R hR N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
    (mbExpGraphSync "b16" epsStr R hR N 7 7 w.b16
      (mbResidGraphSync "b15" epsStr R hR N 7 7 w.b15
        (mbResidGraphSync "b14" epsStr R hR N 7 7 w.b14
          (mbResidGraphSync "b13" epsStr R hR N 7 7 w.b13
            (mbStridedGraphSync "b12" epsStr R hR N 7 7 w.b12
              (mbResidGraphSync "b11" epsStr R hR N 14 14 w.b11
                (mbResidGraphSync "b10" epsStr R hR N 14 14 w.b10
                  (mbExpGraphSync "b9" epsStr R hR N 14 14 w.b9
                    (mbResidGraphSync "b8" epsStr R hR N 14 14 w.b8
                      (mbResidGraphSync "b7" epsStr R hR N 14 14 w.b7
                        (mbStridedGraphSync "b6" epsStr R hR N 14 14 w.b6
                          (mbResidGraphSync "b5" epsStr R hR N 28 28 w.b5
                            (mbStridedGraphSync "b4" epsStr R hR N 28 28 w.b4
                              (mbResidGraphSync "b3" epsStr R hR N 56 56 w.b3
                                (mbStridedGraphSync "b2" epsStr R hR N 56 56 w.b2
                                  (mbNoExpGraphSync "b1" epsStr R hR N 112 112 w.b1
                                    (stemGraphSync epsStr R hR N 112 112 w.sW w.sb w.sε w.sγ w.sβ
                                      e)))))))))))))))))

/-- **The forward at synchronised BatchNorm: replica `r`'s forward IS shard `r` of the global-batch
    forward.** Given that the replicas' inputs are the shards of one batch `X` of `R·N` images, the
    sync-BN graph on replica `r` denotes `batchShard r` of `efficientnetForwardBFull (R * N) w X`
    — the committed batch-BN forward, at the global batch. One block lemma per stage, the shard
    hypothesis threaded from each into the next. -/
theorem efficientnetFwdGraphSyncFull_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) (w : B0Weights) (e : Fin R → SHlo (N * (3 * 224 * 224)))
    (X : Vec ((R * N) * (3 * 224 * 224)))
    (he : ∀ r, den (e r) = batchShard R N (3 * 224 * 224) X r) (r : Fin R) :
    den (efficientnetFwdGraphSyncFull R hR N epsStr w e r)
      = batchShard R N 10 (efficientnetForwardBFull (R * N) w X) r := by
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  have s0 := stemGraphSync_shard epsStr R hR N 112 112 hN h112 h112 w.sW w.sb w.sε w.sγ w.sβ e X he
  have s1 := mbNoExpGraphSync_shard "b1" epsStr R hR N 112 112 hN h112 h112 w.b1 _ _ s0
  have s2 := mbStridedGraphSync_shard "b2" epsStr R hR N 56 56 hN h56 h56 w.b2 _ _ s1
  have s3 := mbResidGraphSync_shard "b3" epsStr R hR N 56 56 hN h56 h56 w.b3 _ _ s2
  have s4 := mbStridedGraphSync_shard "b4" epsStr R hR N 28 28 hN h28 h28 w.b4 _ _ s3
  have s5 := mbResidGraphSync_shard "b5" epsStr R hR N 28 28 hN h28 h28 w.b5 _ _ s4
  have s6 := mbStridedGraphSync_shard "b6" epsStr R hR N 14 14 hN h14 h14 w.b6 _ _ s5
  have s7 := mbResidGraphSync_shard "b7" epsStr R hR N 14 14 hN h14 h14 w.b7 _ _ s6
  have s8 := mbResidGraphSync_shard "b8" epsStr R hR N 14 14 hN h14 h14 w.b8 _ _ s7
  have s9 := mbExpGraphSync_shard "b9" epsStr R hR N 14 14 hN h14 h14 w.b9 _ _ s8
  have s10 := mbResidGraphSync_shard "b10" epsStr R hR N 14 14 hN h14 h14 w.b10 _ _ s9
  have s11 := mbResidGraphSync_shard "b11" epsStr R hR N 14 14 hN h14 h14 w.b11 _ _ s10
  have s12 := mbStridedGraphSync_shard "b12" epsStr R hR N 7 7 hN h7 h7 w.b12 _ _ s11
  have s13 := mbResidGraphSync_shard "b13" epsStr R hR N 7 7 hN h7 h7 w.b13 _ _ s12
  have s14 := mbResidGraphSync_shard "b14" epsStr R hR N 7 7 hN h7 h7 w.b14 _ _ s13
  have s15 := mbResidGraphSync_shard "b15" epsStr R hR N 7 7 hN h7 h7 w.b15 _ _ s14
  have s16 := mbExpGraphSync_shard "b16" epsStr R hR N 7 7 hN h7 h7 w.b16 _ _ s15
  exact headGraphSync_shard epsStr R hR N 7 7 hN h7 h7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb _ _ s16 r

end StableHLO

end Proofs
