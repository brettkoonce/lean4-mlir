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

    den (mobilenetv2FwdGraphSyncFull R hR N epsStr w bf16 e r)
      = batchShard R N nCls (mobilenetv2ForwardBFull (R * N) w X) r

given that each replica's input is its shard of one global batch `X`. **The spec does not
move**: the right-hand side is the committed `mobilenetv2ForwardBFull`, at `N := R·N`, and it does
not move with the precision either: `bf16` selects the conv and depthwise kinds at every site
(`.convAt` / `.convStridedXlaAt` / `.depthwiseAt` / `.depthwiseStridedXlaAt`, as
`MobileNetV2FullB`), `bf16 := true` being the kinds the `*bf16` DP renders emit —
`mobilenetv2in_rmsdp64wxdols0eps0001bf16`, the run the book reports, among them; each conv's
shard step erases the switch through `Bf16Erasure` (`Bf16Fold.denOp_convAt_id`, …) and the rest
of the proof is the f32 one. The `*do*` DP artifacts' forward — that run's — is
`mobilenetv2FwdGraphSyncFullDo`: the same graph with replica `r`'s `dropoutB` at its own mask
`%do` between the GAP and the dense, and `mobilenetv2FwdGraphSyncFullDo_shard` says it denotes
shard `r` of `mobilenetv2ForwardBFullDo` at the global batch and the global mask the replicas'
masks are the shards of.

## How it is proved

As `ResNet34SyncB.lean` proves ResNet-34's: by induction on the chain, one block at a time, with
the shard hypothesis `∀ r, den (e r) = batchShard R N _ X r` carried from block to block.

* every conv, depthwise, GAP and dense node is a per-example lift (`den_batchOp_shard`), and so
  is the XLA-`SAME` strided conv and depthwise — the padding lives inside the per-example map;
* relu6 is pointwise, so it commutes with sharding (`den_relu6_shard`, the peer of
  `den_relu_shard`); the identity skip is `den_addVB_shard`;
* every BatchNorm site is `bnSyncSiteLA`, whose shard lemma `den_bnSyncSiteLA` is the sync-BN
  shard identity on the graph, read at the network index.

The graph reuses `ResNet34SyncB`'s site verbatim — it is net-agnostic — and relu6's shard lemma
is the kit's, so this file is MobileNetV2's six block shapes and their chain.

## Names

Parameter names are `MobileNetV2FullB`'s (`%b{k}{e,d,p}{W,g,bt}`, `%sW`, `%hW`, …, the
`convBias := false` zero-bias operands `%zb{c}`). A BatchNorm site with γ `%b{k}dg` gathers its
statistics as `%arsum` / `%armean` of `b{k}dgmu` and `b{k}dgvar`, each over a `[c]` vector — the
γ name without `%`, then `mu` / `var`.

## What is NOT claimed here

The backward and the parameter collectives are `MobileNetV2SyncStepTieB.lean`'s. That the `R`
replicas' inputs are the shards of one batch is the driver's, as in `DataParallel.Sync`. The rounding the
bf16 kinds perform is not modelled, as f32 rounding is not: `den` is over ℝ at either value of
the flag. The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block replica families + their shard lemmas
-- ════════════════════════════════════════════════════════════════

/-- Stem at sync-BN, over the replica family: 3x3/s2 XLA-`SAME` conv → sync-BN → relu6. -/
def mnv2StemGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu6 (n := oc * h * w))
    (bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc] R hR εs γs βs
      (fun r => .batchOp (N := N) (.convStridedXlaAt bf16 id (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) (e r))
      r)

theorem mnv2StemGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc kH kW : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (mnv2StemGraphSync epsStr R hR N h w Ws bs εs γs βs bf16 e r)
      = batchShard R N (oc * h * w) (mnv2StemB (R * N) h w Ws bs εs γs βs X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N)
    (.convStridedXlaAt bf16 id (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) e X he
  simp only [Bf16Fold.denOp_convStridedXlaAt_id] at hc
  have hn := den_bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc]
    R hR hm εs γs βs _ _ hc
  exact den_relu6_shard _ _ hn r

/-- `t = 1` bottleneck (b1) at sync-BN, over the replica family: depthwise → sync-BN → relu6 →
    project 1x1 → sync-BN. Two sync sites. -/
def mnv2NoExpGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (p : IVWNoExp ic oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * h * w))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR p.pε p.pγ p.pβ
    (fun r => .batchOp (N := N) (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := ic * h * w))
        (bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
          [ic] [ic] R hR p.dε p.dγ p.dβ
          (fun r => .batchOp (N := N)
            (.depthwiseAt bf16 id (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{ic}" p.dW p.db) (e r)) r)))
    r

theorem mnv2NoExpGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVWNoExp ic oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (mnv2NoExpGraphSync pfx epsStr R hR N h w p bf16 e r)
      = batchShard R N (oc * h * w) (mnv2NoExpB (R * N) h w p X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hdc := den_batchOp_shard (N := N)
    (.depthwiseAt bf16 id (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{ic}" p.dW p.db) e X he
  simp only [Bf16Fold.denOp_depthwiseAt_id] at hdc
  have hdn := den_bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
    [ic] [ic] R hR hm p.dε p.dγ p.dβ _ _ hdc
  have hdr := den_relu6_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb) _ _ hdr
  simp only [Bf16Fold.denOp_convAt_id] at hpc
  exact den_bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR hm p.pε p.pγ p.pβ _ _ hpc r

/-- Stride-1 no-skip bottleneck (b11, b17) at sync-BN, over the replica family: expand →
    depthwise → project, a sync site after each, relu6 after the first two. -/
def mnv2ExpOnlyGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (p : IVW ic mid oc) (bf16 : Bool) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR p.pε p.pγ p.pβ
    (fun r => .batchOp (N := N) (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := mid * h * w))
        (bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
          [mid] [mid] R hR p.dε p.dγ p.dβ
          (fun r => .batchOp (N := N)
            (.depthwiseAt bf16 id (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db)
            (.batchOp (N := N) (.relu6 (n := mid * h * w))
              (bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
                [mid] [mid] R hR p.eε p.eγ p.eβ
                (fun r => .batchOp (N := N)
                  (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) (e r)) r)))
          r)))
    r

theorem mnv2ExpOnlyGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW ic mid oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (mnv2ExpOnlyGraphSync pfx epsStr R hR N h w p bf16 e r)
      = batchShard R N (oc * h * w) (mnv2ExpOnlyB (R * N) h w p X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hec := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hec
  have hen := den_bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
    [mid] [mid] R hR hm p.eε p.eγ p.eβ _ _ hec
  have her := den_relu6_shard _ _ hen
  have hdc := den_batchOp_shard (N := N)
    (.depthwiseAt bf16 id (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db) _ _ her
  simp only [Bf16Fold.denOp_depthwiseAt_id] at hdc
  have hdn := den_bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
    [mid] [mid] R hR hm p.dε p.dγ p.dβ _ _ hdc
  have hdr := den_relu6_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb) _ _ hdr
  simp only [Bf16Fold.denOp_convAt_id] at hpc
  exact den_bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR hm p.pε p.pγ p.pβ _ _ hpc r

/-- Stride-1 skip bottleneck at sync-BN: the body plus the `addVB` identity skip, the block input
    shared between both arms on every replica. -/
def mnv2ResidGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid : Nat}
    (p : IVW c mid c) (bf16 : Bool)
    (e : Fin R → SHlo (N * (c * h * w))) : Fin R → SHlo (N * (c * h * w)) :=
  fun r => .addVB (mnv2ExpOnlyGraphSync pfx epsStr R hR N h w p bf16 e r) (e r)

theorem mnv2ResidGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {c mid : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW c mid c) (bf16 : Bool)
    (e : Fin R → SHlo (N * (c * h * w))) (X : Vec ((R * N) * (c * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r) (r : Fin R) :
    den (mnv2ResidGraphSync pfx epsStr R hR N h w p bf16 e r)
      = batchShard R N (c * h * w) (mnv2ResidB (R * N) h w p X) r :=
  den_addVB_shard _ e _ X (mnv2ExpOnlyGraphSync_shard pfx epsStr R hR N h w hN hh hw p bf16 e X he) he r

/-- Stride-2 downsampling bottleneck at sync-BN, over the replica family: expand at `2h x 2w`, the
    XLA-`SAME` strided depthwise, project at `h x w`; three sync sites. -/
def mnv2StridedGraphSync (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (p : IVW ic mid oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR p.pε p.pγ p.pβ
    (fun r => .batchOp (N := N) (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb)
      (.batchOp (N := N) (.relu6 (n := mid * h * w))
        (bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
          [mid] [mid] R hR p.dε p.dγ p.dβ
          (fun r => .batchOp (N := N)
            (.depthwiseStridedXlaAt bf16 id (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db)
            (.batchOp (N := N) (.relu6 (n := mid * (2 * h) * (2 * w)))
              (bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
                [mid] [mid] R hR p.eε p.eγ p.eβ
                (fun r => .batchOp (N := N)
                  (.convAt bf16 id (h := 2 * h) (w := 2 * w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) (e r))
                r)))
          r)))
    r

theorem mnv2StridedGraphSync_shard (pfx epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW ic mid oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w))))
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (mnv2StridedGraphSync pfx epsStr R hR N h w p bf16 e r)
      = batchShard R N (oc * h * w) (mnv2StridedB (R * N) h w p X) r := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm := nhw_ne_zero hN hh hw
  have hm2 := nhw_ne_zero hN h2h h2w
  have hec := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := 2 * h) (w := 2 * w) s!"%b{pfx}eW" s!"%zb{mid}" p.eW p.eb) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hec
  have hen := den_bnSyncSiteLA s!"%b{pfx}eg" s!"%b{pfx}ebt" epsStr s!"b{pfx}egmu" s!"b{pfx}egvar"
    [mid] [mid] R hR hm2 p.eε p.eγ p.eβ _ _ hec
  have her := den_relu6_shard _ _ hen
  have hdc := den_batchOp_shard (N := N)
    (.depthwiseStridedXlaAt bf16 id (h := h) (w := w) s!"%b{pfx}dW" s!"%zb{mid}" p.dW p.db) _ _ her
  simp only [Bf16Fold.denOp_depthwiseStridedXlaAt_id] at hdc
  have hdn := den_bnSyncSiteLA s!"%b{pfx}dg" s!"%b{pfx}dbt" epsStr s!"b{pfx}dgmu" s!"b{pfx}dgvar"
    [mid] [mid] R hR hm p.dε p.dγ p.dβ _ _ hdc
  have hdr := den_relu6_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%b{pfx}pW" s!"%zb{oc}" p.pW p.pb) _ _ hdr
  simp only [Bf16Fold.denOp_convAt_id] at hpc
  exact den_bnSyncSiteLA s!"%b{pfx}pg" s!"%b{pfx}pbt" epsStr s!"b{pfx}pgmu" s!"b{pfx}pgvar"
    [oc] [oc] R hR hm p.pε p.pγ p.pβ _ _ hpc r

/-- Head at sync-BN, over the replica family: 1x1 conv → sync-BN → relu6 → GAP → dense. -/
def mnv2HeadGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * nCls) :=
  fun r => .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.batchOp (N := N) (.relu6 (n := oc * h * w))
        (bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR εh γh βh
          (fun r => .batchOp (N := N) (.convAt bf16 id (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh) (e r)) r)))

theorem mnv2HeadGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc nCls : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool) (e : Fin R → SHlo (N * (ic * h * w)))
    (X : Vec ((R * N) * (ic * h * w))) (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r)
    (r : Fin R) :
    den (mnv2HeadGraphSync epsStr R hR N h w Wh bh εh γh βh Wd bd bf16 e r)
      = batchShard R N nCls (mnv2HeadB (R * N) h w Wh bh εh γh βh Wd bd X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N) (.convAt bf16 id (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hc
  have hn := den_bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc]
    R hR hm εh γh βh _ _ hc
  have hr := den_relu6_shard _ _ hn
  have hg := den_batchOp_shard (N := N) (.gap (c := oc) (h := h) (w := w)) _ _ hr
  exact den_batchOp_shard (N := N) (.dense "%Wd" "%bd" Wd bd) _ _ hg r

/-- `mnv2HeadGraphSync` with the classifier-dropout site: replica `r`'s `dropoutB` at its own mask
    `ms r` (the render's per-replica `%do` input) between the GAP and the dense —
    `mnv2HeadGraphBDo` over the replica family. -/
def mnv2HeadGraphSyncDo (epsStr mName : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool) (ms : Fin R → Vec (N * oc))
    (e : Fin R → SHlo (N * (ic * h * w))) : Fin R → SHlo (N * nCls) :=
  fun r => .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (.dropoutB mName (ms r)
      (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
        (.batchOp (N := N) (.relu6 (n := oc * h * w))
          (bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR εh γh βh
            (fun r => .batchOp (N := N) (.convAt bf16 id (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh)
              (e r)) r))))

/-- With the replicas' masks the shards of one global mask `M`, replica `r`'s dropout head is
    shard `r` of the global `mnv2HeadBDo` — the site's step is `batchShard_zipWith` at the
    multiply, the rest `mnv2HeadGraphSync_shard`'s. -/
theorem mnv2HeadGraphSyncDo_shard (epsStr mName : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc nCls : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool) (ms : Fin R → Vec (N * oc))
    (M : Vec ((R * N) * oc)) (hm : ∀ r, ms r = batchShard R N oc M r)
    (e : Fin R → SHlo (N * (ic * h * w)))
    (X : Vec ((R * N) * (ic * h * w))) (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r)
    (r : Fin R) :
    den (mnv2HeadGraphSyncDo epsStr mName R hR N h w Wh bh εh γh βh Wd bd bf16 ms e r)
      = batchShard R N nCls (mnv2HeadBDo (R * N) h w Wh bh εh γh βh Wd bd M X) r := by
  have hnz := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N) (.convAt bf16 id (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hc
  have hn := den_bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc]
    R hR hnz εh γh βh _ _ hc
  have hr := den_relu6_shard _ _ hn
  have hg := den_batchOp_shard (N := N) (.gap (c := oc) (h := h) (w := w)) _ _ hr
  have hd : ∀ r, den (SHlo.dropoutB (N := N) (n := oc) mName (ms r)
        (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
          (.batchOp (N := N) (.relu6 (n := oc * h * w))
            (bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR εh γh βh
              (fun r => .batchOp (N := N)
                (.convAt bf16 id (h := h) (w := w) "%hW" s!"%zb{oc}" Wh bh) (e r)) r))))
      = batchShard R N oc (dropout M (batchMap (R * N) (globalAvgPoolFlat oc h w)
          (cbrB (R * N) (h := h) (w := w) Wh bh εh γh βh X))) r := by
    intro r
    rw [den_dropoutB, hg r, hm r]
    rfl
  unfold mnv2HeadBDo
  exact den_batchOp_shard (N := N) (.dense "%Wd" "%bd" Wd bd) _ _ hd r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel MobileNetV2 forward graph, over the replica family.**
    `mobilenetv2FwdGraphBFull` with every BatchNorm a `bnSyncSiteLA` over all `R` replicas; block
    prefixes, parameter names and collective tags are the render's. -/
def mobilenetv2FwdGraphSyncFull (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : MNV2BWeights nCls) (bf16 : Bool) (e : Fin R → SHlo (N * (3 * (2 * 112) * (2 * 112)))) :
    Fin R → SHlo (N * nCls) :=
  mnv2HeadGraphSync epsStr R hR N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb bf16
    (mnv2ExpOnlyGraphSync "17" epsStr R hR N 7 7 w.b17 bf16
      (mnv2ResidGraphSync "16" epsStr R hR N 7 7 w.b16 bf16
        (mnv2ResidGraphSync "15" epsStr R hR N 7 7 w.b15 bf16
          (mnv2StridedGraphSync "14" epsStr R hR N 7 7 w.b14 bf16
            (mnv2ResidGraphSync "13" epsStr R hR N 14 14 w.b13 bf16
              (mnv2ResidGraphSync "12" epsStr R hR N 14 14 w.b12 bf16
                (mnv2ExpOnlyGraphSync "11" epsStr R hR N 14 14 w.b11 bf16
                  (mnv2ResidGraphSync "10" epsStr R hR N 14 14 w.b10 bf16
                    (mnv2ResidGraphSync "9" epsStr R hR N 14 14 w.b9 bf16
                      (mnv2ResidGraphSync "8" epsStr R hR N 14 14 w.b8 bf16
                        (mnv2StridedGraphSync "7" epsStr R hR N 14 14 w.b7 bf16
                          (mnv2ResidGraphSync "6" epsStr R hR N 28 28 w.b6 bf16
                            (mnv2ResidGraphSync "5" epsStr R hR N 28 28 w.b5 bf16
                              (mnv2StridedGraphSync "4" epsStr R hR N 28 28 w.b4 bf16
                                (mnv2ResidGraphSync "3" epsStr R hR N 56 56 w.b3 bf16
                                  (mnv2StridedGraphSync "2" epsStr R hR N 56 56 w.b2 bf16
                                    (mnv2NoExpGraphSync "1" epsStr R hR N 112 112 w.b1 bf16
                                      (mnv2StemGraphSync epsStr R hR N 112 112
                                        w.sW w.sb w.sε w.sγ w.sβ bf16 e))))))))))))))))))

/-- **At synchronised BatchNorm, replica `r`'s forward is shard `r` of the global-batch
    forward.** Given that the replicas' inputs are the shards of one batch `X` of `R·N` examples,
    the sync-BN graph on replica `r` denotes `batchShard r` of `mobilenetv2ForwardBFull (R * N)
    w X` — the committed batch-BN forward, at the global batch. One block lemma per stage, the
    shard hypothesis threaded from each into the next. -/
theorem mobilenetv2FwdGraphSyncFull_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : MNV2BWeights nCls) (bf16 : Bool)
    (e : Fin R → SHlo (N * (3 * (2 * 112) * (2 * 112))))
    (X : Vec ((R * N) * (3 * (2 * 112) * (2 * 112))))
    (he : ∀ r, den (e r) = batchShard R N (3 * (2 * 112) * (2 * 112)) X r) (r : Fin R) :
    den (mobilenetv2FwdGraphSyncFull R hR N epsStr w bf16 e r)
      = batchShard R N nCls (mobilenetv2ForwardBFull (R * N) w X) r := by
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  have s0 := mnv2StemGraphSync_shard epsStr R hR N 112 112 hN h112 h112
    w.sW w.sb w.sε w.sγ w.sβ bf16 e X he
  have s1 := mnv2NoExpGraphSync_shard "1" epsStr R hR N 112 112 hN h112 h112 w.b1 bf16 _ _ s0
  have s2 := mnv2StridedGraphSync_shard "2" epsStr R hR N 56 56 hN h56 h56 w.b2 bf16 _ _ s1
  have s3 := mnv2ResidGraphSync_shard "3" epsStr R hR N 56 56 hN h56 h56 w.b3 bf16 _ _ s2
  have s4 := mnv2StridedGraphSync_shard "4" epsStr R hR N 28 28 hN h28 h28 w.b4 bf16 _ _ s3
  have s5 := mnv2ResidGraphSync_shard "5" epsStr R hR N 28 28 hN h28 h28 w.b5 bf16 _ _ s4
  have s6 := mnv2ResidGraphSync_shard "6" epsStr R hR N 28 28 hN h28 h28 w.b6 bf16 _ _ s5
  have s7 := mnv2StridedGraphSync_shard "7" epsStr R hR N 14 14 hN h14 h14 w.b7 bf16 _ _ s6
  have s8 := mnv2ResidGraphSync_shard "8" epsStr R hR N 14 14 hN h14 h14 w.b8 bf16 _ _ s7
  have s9 := mnv2ResidGraphSync_shard "9" epsStr R hR N 14 14 hN h14 h14 w.b9 bf16 _ _ s8
  have s10 := mnv2ResidGraphSync_shard "10" epsStr R hR N 14 14 hN h14 h14 w.b10 bf16 _ _ s9
  have s11 := mnv2ExpOnlyGraphSync_shard "11" epsStr R hR N 14 14 hN h14 h14 w.b11 bf16 _ _ s10
  have s12 := mnv2ResidGraphSync_shard "12" epsStr R hR N 14 14 hN h14 h14 w.b12 bf16 _ _ s11
  have s13 := mnv2ResidGraphSync_shard "13" epsStr R hR N 14 14 hN h14 h14 w.b13 bf16 _ _ s12
  have s14 := mnv2StridedGraphSync_shard "14" epsStr R hR N 7 7 hN h7 h7 w.b14 bf16 _ _ s13
  have s15 := mnv2ResidGraphSync_shard "15" epsStr R hR N 7 7 hN h7 h7 w.b15 bf16 _ _ s14
  have s16 := mnv2ResidGraphSync_shard "16" epsStr R hR N 7 7 hN h7 h7 w.b16 bf16 _ _ s15
  have s17 := mnv2ExpOnlyGraphSync_shard "17" epsStr R hR N 7 7 hN h7 h7 w.b17 bf16 _ _ s16
  exact mnv2HeadGraphSync_shard epsStr R hR N 7 7 hN h7 h7
    w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb bf16 _ _ s17 r

/-- **The sync-BN data-parallel MobileNetV2 forward graph with classifier dropout**, over the
    replica family — `mobilenetv2FwdGraphSyncFull` with `mnv2HeadGraphSyncDo` as its head, replica
    `r` at its mask `ms r`; the forward of the `*do*` DP train steps. -/
def mobilenetv2FwdGraphSyncFullDo (R : Nat) (hR : 0 < R) (N : Nat) (epsStr mName : String)
    {nCls : Nat} (w : MNV2BWeights nCls) (bf16 : Bool) (ms : Fin R → Vec (N * 1280))
    (e : Fin R → SHlo (N * (3 * (2 * 112) * (2 * 112)))) : Fin R → SHlo (N * nCls) :=
  mnv2HeadGraphSyncDo epsStr mName R hR N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb bf16 ms
    (mnv2ExpOnlyGraphSync "17" epsStr R hR N 7 7 w.b17 bf16
      (mnv2ResidGraphSync "16" epsStr R hR N 7 7 w.b16 bf16
        (mnv2ResidGraphSync "15" epsStr R hR N 7 7 w.b15 bf16
          (mnv2StridedGraphSync "14" epsStr R hR N 7 7 w.b14 bf16
            (mnv2ResidGraphSync "13" epsStr R hR N 14 14 w.b13 bf16
              (mnv2ResidGraphSync "12" epsStr R hR N 14 14 w.b12 bf16
                (mnv2ExpOnlyGraphSync "11" epsStr R hR N 14 14 w.b11 bf16
                  (mnv2ResidGraphSync "10" epsStr R hR N 14 14 w.b10 bf16
                    (mnv2ResidGraphSync "9" epsStr R hR N 14 14 w.b9 bf16
                      (mnv2ResidGraphSync "8" epsStr R hR N 14 14 w.b8 bf16
                        (mnv2StridedGraphSync "7" epsStr R hR N 14 14 w.b7 bf16
                          (mnv2ResidGraphSync "6" epsStr R hR N 28 28 w.b6 bf16
                            (mnv2ResidGraphSync "5" epsStr R hR N 28 28 w.b5 bf16
                              (mnv2StridedGraphSync "4" epsStr R hR N 28 28 w.b4 bf16
                                (mnv2ResidGraphSync "3" epsStr R hR N 56 56 w.b3 bf16
                                  (mnv2StridedGraphSync "2" epsStr R hR N 56 56 w.b2 bf16
                                    (mnv2NoExpGraphSync "1" epsStr R hR N 112 112 w.b1 bf16
                                      (mnv2StemGraphSync epsStr R hR N 112 112
                                        w.sW w.sb w.sε w.sγ w.sβ bf16 e))))))))))))))))))

/-- `mobilenetv2FwdGraphSyncFull_shard` at the site: with the replicas' inputs the shards of one
    batch `X` and their masks the shards of one mask `M`, replica `r`'s dropout forward denotes
    shard `r` of `mobilenetv2ForwardBFullDo (R * N) w M X`. -/
theorem mobilenetv2FwdGraphSyncFullDo_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr mName : String) {nCls : Nat} (w : MNV2BWeights nCls) (bf16 : Bool)
    (ms : Fin R → Vec (N * 1280)) (M : Vec ((R * N) * 1280))
    (hm : ∀ r, ms r = batchShard R N 1280 M r)
    (e : Fin R → SHlo (N * (3 * (2 * 112) * (2 * 112))))
    (X : Vec ((R * N) * (3 * (2 * 112) * (2 * 112))))
    (he : ∀ r, den (e r) = batchShard R N (3 * (2 * 112) * (2 * 112)) X r) (r : Fin R) :
    den (mobilenetv2FwdGraphSyncFullDo R hR N epsStr mName w bf16 ms e r)
      = batchShard R N nCls (mobilenetv2ForwardBFullDo (R * N) w M X) r := by
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  have s0 := mnv2StemGraphSync_shard epsStr R hR N 112 112 hN h112 h112
    w.sW w.sb w.sε w.sγ w.sβ bf16 e X he
  have s1 := mnv2NoExpGraphSync_shard "1" epsStr R hR N 112 112 hN h112 h112 w.b1 bf16 _ _ s0
  have s2 := mnv2StridedGraphSync_shard "2" epsStr R hR N 56 56 hN h56 h56 w.b2 bf16 _ _ s1
  have s3 := mnv2ResidGraphSync_shard "3" epsStr R hR N 56 56 hN h56 h56 w.b3 bf16 _ _ s2
  have s4 := mnv2StridedGraphSync_shard "4" epsStr R hR N 28 28 hN h28 h28 w.b4 bf16 _ _ s3
  have s5 := mnv2ResidGraphSync_shard "5" epsStr R hR N 28 28 hN h28 h28 w.b5 bf16 _ _ s4
  have s6 := mnv2ResidGraphSync_shard "6" epsStr R hR N 28 28 hN h28 h28 w.b6 bf16 _ _ s5
  have s7 := mnv2StridedGraphSync_shard "7" epsStr R hR N 14 14 hN h14 h14 w.b7 bf16 _ _ s6
  have s8 := mnv2ResidGraphSync_shard "8" epsStr R hR N 14 14 hN h14 h14 w.b8 bf16 _ _ s7
  have s9 := mnv2ResidGraphSync_shard "9" epsStr R hR N 14 14 hN h14 h14 w.b9 bf16 _ _ s8
  have s10 := mnv2ResidGraphSync_shard "10" epsStr R hR N 14 14 hN h14 h14 w.b10 bf16 _ _ s9
  have s11 := mnv2ExpOnlyGraphSync_shard "11" epsStr R hR N 14 14 hN h14 h14 w.b11 bf16 _ _ s10
  have s12 := mnv2ResidGraphSync_shard "12" epsStr R hR N 14 14 hN h14 h14 w.b12 bf16 _ _ s11
  have s13 := mnv2ResidGraphSync_shard "13" epsStr R hR N 14 14 hN h14 h14 w.b13 bf16 _ _ s12
  have s14 := mnv2StridedGraphSync_shard "14" epsStr R hR N 7 7 hN h7 h7 w.b14 bf16 _ _ s13
  have s15 := mnv2ResidGraphSync_shard "15" epsStr R hR N 7 7 hN h7 h7 w.b15 bf16 _ _ s14
  have s16 := mnv2ResidGraphSync_shard "16" epsStr R hR N 7 7 hN h7 h7 w.b16 bf16 _ _ s15
  have s17 := mnv2ExpOnlyGraphSync_shard "17" epsStr R hR N 7 7 hN h7 h7 w.b17 bf16 _ _ s16
  exact mnv2HeadGraphSyncDo_shard epsStr mName R hR N 7 7 hN h7 h7
    w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb bf16 ms M hm _ _ s17 r

end StableHLO

end Proofs
