import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullB
import LeanMlir.Proofs.Foundation.DataParallel.SyncKit

/-! # MobileNetV2's data-parallel forward at SYNCHRONISED BatchNorm — replica `r` IS shard `r`

`MobileNetV2FullB.lean` says the typed batch-BN graph denotes `mobilenetv2ForwardBFull N w`
on one device. `MobileNetV2RenderB`'s data-parallel step normalises with the GLOBAL batch's
statistics at `replicas > 1`: every one of the 52 BatchNorm sites is the sync-BN composition —
this replica's mean all-reduced, then Chan's `σ²_r + (μ_r − μ)²` all-reduced, packed, then
`bnSyncF`. This file is the data-parallel twin of `mobilenetv2FwdGraphBFull_faithful`: that forward graph, stated as a family over the
`R` replicas, denotes on replica `r` exactly `batchShard r` of the single-device forward at the
global batch `R·N`.

    den (mobilenetv2FwdGraphSyncFull R hR N epsStr w e r)
      = batchShard R N nCls (mobilenetv2ForwardBFull (R * N) w X) r

given that each replica's input is its shard of one global batch `X`. **The spec does not
move**: the right-hand side is the committed `mobilenetv2ForwardBFull`, at `N := R·N`.

## How it is proved

As `ResNet34SyncB.lean` proves ResNet-34's: by induction on the chain, one block at a time, with
the shard hypothesis `∀ r, den (e r) = batchShard R N _ X r` carried from block to block.

* every conv, depthwise, GAP and dense node is a per-example lift (`den_batchOp_shard`), and so
  is the XLA-`SAME` strided conv and depthwise — the padding lives inside the per-example map;
* relu6 is pointwise, so it commutes with sharding (`den_relu6_shard`, the peer of
  `den_relu_shard`); the identity skip is `den_addVB_shard`;
* every BatchNorm site is `bnSyncSiteLA`, whose shard lemma `den_bnSyncSiteLA` is the sync-BN
  shard identity on the graph, read at the network index.

The graph reuses `ResNet34SyncB`'s site verbatim — it is net-agnostic — so the only new lemma is
the relu6 one; the rest of this file is MobileNetV2's six block shapes and their chain.

## Names

Parameter names are `MobileNetV2FullB`'s (`%b{k}{e,d,p}{W,g,bt}`, `%sW`, `%hW`, …, the
`convBias := false` zero-bias operands `%zb{c}`). A BatchNorm site with γ `%b{k}dg` gathers its
statistics as `%arsum` / `%armean` of `b{k}dgmu` and `b{k}dgvar`, each over a `[c]` vector — the
γ name without `%`, then `mu` / `var`.

## What is NOT claimed here

The backward and the parameter collectives are `MobileNetV2SyncStepTieB.lean`'s. That the `R`
replicas' inputs are the shards of one batch is the driver's, as in `DataParallel.Sync`. The
lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § relu6 — pointwise, so sharding commutes with it
-- ════════════════════════════════════════════════════════════════

/-- **relu6 on every replica is the shard of the global relu6** — the clamp reads one cell, so
    cutting the batch before or after it is the same. The MobileNetV2 peer of `den_relu_shard`,
    stated at the whole-batch `relu6` the committed forward is written in. -/
theorem den_relu6_shard {R N n : Nat} (e : Fin R → SHlo (N * n)) (X : Vec ((R * N) * n))
    (he : ∀ r, den (e r) = batchShard R N n X r) (r : Fin R) :
    den (.batchOp (N := N) (.relu6 (n := n)) (e r))
      = batchShard R N n (relu6 ((R * N) * n) X) r := by
  rw [den_batchOp_relu6_eq_relu6F, relu6F_faithful, he]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Per-block replica families + their shard lemmas
-- ════════════════════════════════════════════════════════════════

/-- Stem at sync-BN, over the replica family: 3x3/s2 XLA-`SAME` conv → sync-BN → relu6. -/
def mnv2StemGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu6 (n := oc * h * w))
    (bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc] R hR εs γs βs
      (fun r => .batchOp (N := N) (.convStridedXla (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) (e r))
      r)

theorem mnv2StemGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc kH kW : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (mnv2StemGraphSync epsStr R hR N h w Ws bs εs γs βs e r)
      = batchShard R N (oc * h * w) (mnv2StemB (R * N) h w Ws bs εs γs βs X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N)
    (.convStridedXla (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) e X he
  have hn := den_bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc]
    R hR hm εs γs βs _ _ hc
  exact den_relu6_shard _ _ hn r

/-- `t = 1` bottleneck (b1) at sync-BN, over the replica family: depthwise → sync-BN → relu6 →
    project 1x1 → sync-BN. Two sync sites. -/
def mnv2NoExpGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (p : IVWNoExp ic oc) (e : Fin R → SHlo (N * (ic * h * w))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR p.pε p.pγ p.pβ
    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := ic * h * w))
        (bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
          [ic] [ic] R hR p.dε p.dγ p.dβ
          (fun r => .batchOp (N := N)
            (.depthwise (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{ic}" p.dW p.db) (e r)) r)))
    r

theorem mnv2NoExpGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVWNoExp ic oc)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (mnv2NoExpGraphSync pfx epsStr R hR N h w p e r)
      = batchShard R N (oc * h * w) (mnv2NoExpB (R * N) h w p X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hdc := den_batchOp_shard (N := N)
    (.depthwise (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{ic}" p.dW p.db) e X he
  have hdn := den_bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
    [ic] [ic] R hR hm p.dε p.dγ p.dβ _ _ hdc
  have hdr := den_relu6_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb) _ _ hdr
  exact den_bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR hm p.pε p.pγ p.pβ _ _ hpc r

/-- Stride-1 no-skip bottleneck (b11, b17) at sync-BN, over the replica family: expand →
    depthwise → project, a sync site after each, relu6 after the first two. -/
def mnv2ExpOnlyGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (p : IVW ic mid oc) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR p.pε p.pγ p.pβ
    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := mid * h * w))
        (bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
          [mid] [mid] R hR p.dε p.dγ p.dβ
          (fun r => .batchOp (N := N)
            (.depthwise (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db)
            (.batchOp (N := N) (.relu6 (n := mid * h * w))
              (bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
                [mid] [mid] R hR p.eε p.eγ p.eβ
                (fun r => .batchOp (N := N)
                  (.conv (h := h) (w := w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) (e r)) r)))
          r)))
    r

theorem mnv2ExpOnlyGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW ic mid oc)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (mnv2ExpOnlyGraphSync pfx epsStr R hR N h w p e r)
      = batchShard R N (oc * h * w) (mnv2ExpOnlyB (R * N) h w p X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hec := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) e X he
  have hen := den_bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
    [mid] [mid] R hR hm p.eε p.eγ p.eβ _ _ hec
  have her := den_relu6_shard _ _ hen
  have hdc := den_batchOp_shard (N := N)
    (.depthwise (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db) _ _ her
  have hdn := den_bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
    [mid] [mid] R hR hm p.dε p.dγ p.dβ _ _ hdc
  have hdr := den_relu6_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb) _ _ hdr
  exact den_bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR hm p.pε p.pγ p.pβ _ _ hpc r

/-- Stride-1 skip bottleneck at sync-BN: the body plus the `addVB` identity skip, the block input
    shared between both arms on every replica. -/
def mnv2ResidGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid : Nat}
    (p : IVW c mid c) (e : Fin R → SHlo (N * (c * h * w))) : Fin R → SHlo (N * (c * h * w)) :=
  fun r => .addVB (mnv2ExpOnlyGraphSync pfx epsStr R hR N h w p e r) (e r)

theorem mnv2ResidGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {c mid : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW c mid c)
    (e : Fin R → SHlo (N * (c * h * w))) (X : Vec ((R * N) * (c * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r) (r : Fin R) :
    den (mnv2ResidGraphSync pfx epsStr R hR N h w p e r)
      = batchShard R N (c * h * w) (mnv2ResidB (R * N) h w p X) r :=
  den_addVB_shard _ e _ X (mnv2ExpOnlyGraphSync_shard pfx epsStr R hR N h w hN hh hw p e X he) he r

/-- Stride-2 downsampling bottleneck at sync-BN, over the replica family: expand at `2h x 2w`, the
    XLA-`SAME` strided depthwise, project at `h x w`; three sync sites. -/
def mnv2StridedGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (p : IVW ic mid oc) (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR p.pε p.pγ p.pβ
    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := mid * h * w))
        (bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
          [mid] [mid] R hR p.dε p.dγ p.dβ
          (fun r => .batchOp (N := N)
            (.depthwiseStridedXla (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db)
            (.batchOp (N := N) (.relu6 (n := mid * (2 * h) * (2 * w)))
              (bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
                [mid] [mid] R hR p.eε p.eγ p.eβ
                (fun r => .batchOp (N := N)
                  (.conv (h := 2 * h) (w := 2 * w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) (e r))
                r)))
          r)))
    r

theorem mnv2StridedGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW ic mid oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w))))
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (mnv2StridedGraphSync pfx epsStr R hR N h w p e r)
      = batchShard R N (oc * h * w) (mnv2StridedB (R * N) h w p X) r := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm := nhw_ne_zero hN hh hw
  have hm2 := nhw_ne_zero hN h2h h2w
  have hec := den_batchOp_shard (N := N)
    (.conv (h := 2 * h) (w := 2 * w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) e X he
  have hen := den_bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
    [mid] [mid] R hR hm2 p.eε p.eγ p.eβ _ _ hec
  have her := den_relu6_shard _ _ hen
  have hdc := den_batchOp_shard (N := N)
    (.depthwiseStridedXla (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db) _ _ her
  have hdn := den_bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
    [mid] [mid] R hR hm p.dε p.dγ p.dβ _ _ hdc
  have hdr := den_relu6_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb) _ _ hdr
  exact den_bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR hm p.pε p.pγ p.pβ _ _ hpc r

/-- Head at sync-BN, over the replica family: 1x1 conv → sync-BN → relu6 → GAP → dense. -/
def mnv2HeadGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * nCls) :=
  fun r => .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.batchOp (N := N) (.relu6 (n := oc * h * w))
        (bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR εh γh βh
          (fun r => .batchOp (N := N) (.conv (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh) (e r)) r)))

theorem mnv2HeadGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc nCls : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : Fin R → SHlo (N * (ic * h * w)))
    (X : Vec ((R * N) * (ic * h * w))) (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r)
    (r : Fin R) :
    den (mnv2HeadGraphSync epsStr R hR N h w Wh bh εh γh βh Wd bd e r)
      = batchShard R N nCls (mnv2HeadB (R * N) h w Wh bh εh γh βh Wd bd X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N) (.conv (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh) e X he
  have hn := den_bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc]
    R hR hm εh γh βh _ _ hc
  have hr := den_relu6_shard _ _ hn
  have hg := den_batchOp_shard (N := N) (.gap (c := oc) (h := h) (w := w)) _ _ hr
  exact den_batchOp_shard (N := N) (.dense "%Wd" "%bd" Wd bd) _ _ hg r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel MobileNetV2 forward graph, over the replica family.**
    `mobilenetv2FwdGraphBFull` with every BatchNorm a `bnSyncSiteLA` over all `R` replicas; block
    prefixes, parameter names and collective tags are the render's. -/
def mobilenetv2FwdGraphSyncFull (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : MNV2BWeights nCls) (e : Fin R → SHlo (N * (3 * (2 * 112) * (2 * 112)))) :
    Fin R → SHlo (N * nCls) :=
  mnv2HeadGraphSync epsStr R hR N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
    (mnv2ExpOnlyGraphSync "17" epsStr R hR N 7 7 w.b17
      (mnv2ResidGraphSync "16" epsStr R hR N 7 7 w.b16
        (mnv2ResidGraphSync "15" epsStr R hR N 7 7 w.b15
          (mnv2StridedGraphSync "14" epsStr R hR N 7 7 w.b14
            (mnv2ResidGraphSync "13" epsStr R hR N 14 14 w.b13
              (mnv2ResidGraphSync "12" epsStr R hR N 14 14 w.b12
                (mnv2ExpOnlyGraphSync "11" epsStr R hR N 14 14 w.b11
                  (mnv2ResidGraphSync "10" epsStr R hR N 14 14 w.b10
                    (mnv2ResidGraphSync "9" epsStr R hR N 14 14 w.b9
                      (mnv2ResidGraphSync "8" epsStr R hR N 14 14 w.b8
                        (mnv2StridedGraphSync "7" epsStr R hR N 14 14 w.b7
                          (mnv2ResidGraphSync "6" epsStr R hR N 28 28 w.b6
                            (mnv2ResidGraphSync "5" epsStr R hR N 28 28 w.b5
                              (mnv2StridedGraphSync "4" epsStr R hR N 28 28 w.b4
                                (mnv2ResidGraphSync "3" epsStr R hR N 56 56 w.b3
                                  (mnv2StridedGraphSync "2" epsStr R hR N 56 56 w.b2
                                    (mnv2NoExpGraphSync "1" epsStr R hR N 112 112 w.b1
                                      (mnv2StemGraphSync epsStr R hR N 112 112
                                        w.sW w.sb w.sε w.sγ w.sβ e))))))))))))))))))

/-- **At synchronised BatchNorm, replica `r`'s forward is shard `r` of the global-batch
    forward.** Given that the replicas' inputs are the shards of one batch `X` of `R·N` examples,
    the sync-BN graph on replica `r` denotes `batchShard r` of `mobilenetv2ForwardBFull (R * N)
    w X` — the committed batch-BN forward, at the global batch. One block lemma per stage, the
    shard hypothesis threaded from each into the next. -/
theorem mobilenetv2FwdGraphSyncFull_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : MNV2BWeights nCls)
    (e : Fin R → SHlo (N * (3 * (2 * 112) * (2 * 112))))
    (X : Vec ((R * N) * (3 * (2 * 112) * (2 * 112))))
    (he : ∀ r, den (e r) = batchShard R N (3 * (2 * 112) * (2 * 112)) X r) (r : Fin R) :
    den (mobilenetv2FwdGraphSyncFull R hR N epsStr w e r)
      = batchShard R N nCls (mobilenetv2ForwardBFull (R * N) w X) r := by
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  have s0 := mnv2StemGraphSync_shard epsStr R hR N 112 112 hN h112 h112
    w.sW w.sb w.sε w.sγ w.sβ e X he
  have s1 := mnv2NoExpGraphSync_shard "1" epsStr R hR N 112 112 hN h112 h112 w.b1 _ _ s0
  have s2 := mnv2StridedGraphSync_shard "2" epsStr R hR N 56 56 hN h56 h56 w.b2 _ _ s1
  have s3 := mnv2ResidGraphSync_shard "3" epsStr R hR N 56 56 hN h56 h56 w.b3 _ _ s2
  have s4 := mnv2StridedGraphSync_shard "4" epsStr R hR N 28 28 hN h28 h28 w.b4 _ _ s3
  have s5 := mnv2ResidGraphSync_shard "5" epsStr R hR N 28 28 hN h28 h28 w.b5 _ _ s4
  have s6 := mnv2ResidGraphSync_shard "6" epsStr R hR N 28 28 hN h28 h28 w.b6 _ _ s5
  have s7 := mnv2StridedGraphSync_shard "7" epsStr R hR N 14 14 hN h14 h14 w.b7 _ _ s6
  have s8 := mnv2ResidGraphSync_shard "8" epsStr R hR N 14 14 hN h14 h14 w.b8 _ _ s7
  have s9 := mnv2ResidGraphSync_shard "9" epsStr R hR N 14 14 hN h14 h14 w.b9 _ _ s8
  have s10 := mnv2ResidGraphSync_shard "10" epsStr R hR N 14 14 hN h14 h14 w.b10 _ _ s9
  have s11 := mnv2ExpOnlyGraphSync_shard "11" epsStr R hR N 14 14 hN h14 h14 w.b11 _ _ s10
  have s12 := mnv2ResidGraphSync_shard "12" epsStr R hR N 14 14 hN h14 h14 w.b12 _ _ s11
  have s13 := mnv2ResidGraphSync_shard "13" epsStr R hR N 14 14 hN h14 h14 w.b13 _ _ s12
  have s14 := mnv2StridedGraphSync_shard "14" epsStr R hR N 7 7 hN h7 h7 w.b14 _ _ s13
  have s15 := mnv2ResidGraphSync_shard "15" epsStr R hR N 7 7 hN h7 h7 w.b15 _ _ s14
  have s16 := mnv2ResidGraphSync_shard "16" epsStr R hR N 7 7 hN h7 h7 w.b16 _ _ s15
  have s17 := mnv2ExpOnlyGraphSync_shard "17" epsStr R hR N 7 7 hN h7 h7 w.b17 _ _ s16
  exact mnv2HeadGraphSync_shard epsStr R hR N 7 7 hN h7 h7
    w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb _ _ s17 r

end StableHLO

end Proofs
