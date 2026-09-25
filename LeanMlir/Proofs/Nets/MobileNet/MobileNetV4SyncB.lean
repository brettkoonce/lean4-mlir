import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB
import LeanMlir.Proofs.Foundation.DataParallelSyncKit

/-! # MobileNetV4-Conv-M's data-parallel forward at SYNCHRONISED BatchNorm — replica `r` IS shard `r`

`MobileNetV4FullB.lean` (T2) says the typed batch-BN graph denotes `mobilenetv4ForwardBFull N w`
on one device. `MobileNetV4RenderB`'s data-parallel step normalises with the GLOBAL batch's
statistics at `replicas > 1`: every one of the 77 BatchNorm sites is the sync-BN composition —
this replica's mean all-reduced, then Chan's `σ²_r + (μ_r − μ)²` all-reduced, packed, then
`bnSyncF`. This file is T2's data-parallel twin: that forward graph, stated as a family over the
`R` replicas, denotes on replica `r` exactly `batchShard r` of the single-device forward at the
global batch `R·N`.

    den (mnv4FwdGraphSyncFull R hR N epsStr w e r)
      = batchShard R N nCls (mobilenetv4ForwardBFull (R * N) w X) r

given that each replica's input is its shard of one global batch `X`. ⭐ **The spec does not
move**: the right-hand side is the committed `mobilenetv4ForwardBFull`, at `N := R·N`.

## How it is proved

As `ResNet34SyncB.lean` and `MobileNetV2SyncB.lean` prove theirs: one block at a time, with the
shard hypothesis `∀ r, den (e r) = batchShard R N _ X r` carried from block to block.

* every conv, depthwise (stride 1 and SYMMETRIC stride 2), strided conv, GAP and dense node is a
  per-example lift (`den_batchOp_shard`) — the padding lives inside the per-example map;
* relu is pointwise (`den_relu_shard`); the identity skip is `den_addVB_shard`; the head's two
  `1×1` relabellings are `den_castIdx_shard` (sharding commutes with a per-example relabel);
* every BatchNorm site is `bnSyncSiteLA`, whose shard lemma `den_bnSyncSiteLA` is P1 on the graph
  read at the network index.

⭐ **Same shape as T2, and for T2's reasons.** The block lemmas are GENERIC IN THE ROW (`s :
UibSpec`), so every width is a projection of a variable and nothing evaluates; each one closes on
the row-typed `mnv4BodyOfRow (R * N) s p` / `mnv4StridedBodyOfRow` by the dispatch hypotheses
T2 uses (`s.preDWk ≠ 0`, `s.postDWk = 0`, …), discharged by `decide` at the concrete rows. The
skip is one generic combinator (`mnv4SkipGraphSync`, the peer of `mnv4SkipGraphB`), which keeps
the whole-net term linear in the depth. The five resolution groups are proved against
`mnv4Res*Layer (R * N) w` through T2's `_fwd_apply` peels, and the whole net is a `have`-chain of
eight stage lemmas over `mobilenetv4ForwardBFull`'s own prefixes — no `CertLayer.comp` is ever
peeled at a literal width.

## The index seam

The conv/relu chain runs at the left-assoc index `N·(c·h·w)`; `bnSyncF` and its statistics nodes
at `N·(c·(h·w))`. As in ResNet-34's and MobileNetV2's twins the site is `ResNet34SyncB`'s
`bnSyncSiteLA`, which carries the `mul_assoc` relabelling as `castIdx` on the AST value (the
emitted text does not change: `skel` never sees an index), and `den_bnSyncSiteLA` reads P1 back at
the network index through `batchShard_castIdx`. Nothing about the seam is MobileNetV4's own.

## Names

Parameter names are `MobileNetV4FullB`'s, read off the row (`%u{p}{q,e,d,p}{W,g,bt}`, `%f0cW`,
`%sW`, `%h1W`, `%hW`, …), and every bias operand is the shared zero `%zb{c}`. A BatchNorm site with
γ `%u{p}qg` gathers its statistics as `%arsum` / `%armean` of `u{p}qgmu` and `u{p}qgvar`, each over
a `[c]` vector — the γ name without `%`, then `mu` / `var`, the tag `bnFwdSite` is handed.

## What is NOT claimed here

⚠ The backward and the parameter collectives are the T3 half (`MobileNetV4SyncStepTieB.lean`).
⚠ Every conv here is bias-free by construction: `MobileNetV4RenderB` has no `convBias` flag and
binds each bias slot to the zero `%zb{c}`, so no bias is trained and no bias gradient is emitted
(the statement is `∀ w`, and zero biases are one instance). ⚠ The
statement is at the f32 nodes; the `*bf16` artifact's bf16 conv twins are outside it. ⚠ That the
`R` replicas' inputs ARE the shards of one batch is the driver's, as in `DataParallelSync.lean`.
⚠ The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The head's relabelling — per-example, so sharding commutes with it
-- ════════════════════════════════════════════════════════════════

/-- **A per-example relabel on every replica is the shard of the relabelled global batch** — the
    head's `[N, c] ↔ [N, c, 1, 1]` casts, read at the network index. -/
theorem den_castIdx_shard {R N a b : Nat} (hab : a = b) (h : N * a = N * b)
    (e : Fin R → SHlo (N * a)) (X : Vec ((R * N) * a))
    (he : ∀ r, den (e r) = batchShard R N a X r) (r : Fin R) :
    den (castIdx h (e r))
      = batchShard R N b (fun i => X (Fin.cast (congrArg ((R * N) * ·) hab).symm i)) r := by
  rw [den_castIdx, he, batchShard_castIdx hab]

-- ════════════════════════════════════════════════════════════════
-- § The stem and the fused stage — generic in their widths
-- ════════════════════════════════════════════════════════════════

/-- Stem at sync-BN, over the replica family: 3×3/s2 symmetric conv → sync-BN → relu. -/
def mnv4StemGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc] R hR εs γs βs
      (fun r => .batchOp (N := N) (.convStrided (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) (e r))
      r)

theorem mnv4StemGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc kH kW : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (mnv4StemGraphSync epsStr R hR N h w Ws bs εs γs βs e r)
      = batchShard R N (oc * h * w) (mnv4StemB (R * N) h w Ws bs εs γs βs X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc := den_batchOp_shard (N := N)
    (.convStrided (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) e X he
  have hn := den_bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc]
    R hR hm εs γs βs _ _ hc
  exact den_relu_shard _ _ hn r

/-- Fused stage at sync-BN, over the replica family: 3×3/s2 symmetric conv → sync-BN → relu →
    1×1 project → sync-BN. Two sync sites, no skip. -/
def mnv4FusedGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => bnSyncSiteLA "%f0pg" "%f0pbt" epsStr "f0pgmu" "f0pgvar" [oc] [oc] R hR εp γp βp
    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) "%f0pW" s!"%zb{oc}" Wp bp)
      (.batchOp (N := N) (.relu (n := mid * h * w))
        (bnSyncSiteLA "%f0cg" "%f0cbt" epsStr "f0cgmu" "f0cgvar" [mid] [mid] R hR εc γc βc
          (fun r => .batchOp (N := N)
            (.convStrided (h := h) (w := w) "%f0cW" s!"%zb{mid}" Wc bc) (e r)) r)))
    r

theorem mnv4FusedGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc kH kW : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (hεc : 0 < εc) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (mnv4FusedGraphSync epsStr R hR N h w Wc bc εc γc βc Wp bp εp γp βp e r)
      = batchShard R N (oc * h * w)
          ((mnv4FusedStage (R * N) (cbReluStridedLayer (h := h) (w := w) (R * N) Wc bc εc hεc γc βc)
            (projLayer (h := h) (w := w) (R * N) Wp bp εp hεp γp βp)).fwd X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hcc := den_batchOp_shard (N := N)
    (.convStrided (h := h) (w := w) "%f0cW" s!"%zb{mid}" Wc bc) e X he
  have hcn := den_bnSyncSiteLA "%f0cg" "%f0cbt" epsStr "f0cgmu" "f0cgvar" [mid] [mid]
    R hR hm εc γc βc _ _ hcc
  have hcs := den_relu_shard _ _ hcn
  have hpc := den_batchOp_shard (N := N) (.conv (h := h) (w := w) "%f0pW" s!"%zb{oc}" Wp bp) _ _ hcs
  exact den_bnSyncSiteLA "%f0pg" "%f0pbt" epsStr "f0pgmu" "f0pgvar" [oc] [oc]
    R hR hm εp γp βp _ _ hpc r

-- ════════════════════════════════════════════════════════════════
-- § The UIB bodies — GENERIC IN THE ROW, one per family, as T2's
-- ════════════════════════════════════════════════════════════════

/-- **ExtraDW body at sync-BN**, over the replica family — both depthwises present: pre-DW →
    sync-BN → expand → sync-BN → relu → post-DW → sync-BN → relu → project → sync-BN. Four sync
    sites. The body only; the identity skip is `mnv4SkipGraphSync`. -/
def mnv4ExtraDWBodyGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (p : UibParams s) (e : Fin R → SHlo (N * (s.ic * s.h * s.h))) :
    Fin R → SHlo (N * (s.oc * s.h * s.h)) :=
  fun r => bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR p.ez p.gz p.bz2
    (fun r => .batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (bnSyncSiteLA s!"%u{s.p}dg" s!"%u{s.p}dbt" epsStr s!"u{s.p}dgmu" s!"u{s.p}dgvar"
          [s.ic * s.expand] [s.ic * s.expand] R hR p.ed p.gd p.bd2
          (fun r => .batchOp (N := N) (.depthwise (c := s.ic * s.expand) (h := s.h) (w := s.h)
              s!"%u{s.p}dW" s!"%zb{s.ic * s.expand}" p.Wd p.bd)
            (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
              (bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
                [s.ic * s.expand] [s.ic * s.expand] R hR p.ee p.ge p.be2
                (fun r => .batchOp (N := N)
                  (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
                    s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
                  (bnSyncSiteLA s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr s!"u{s.p}qgmu"
                    s!"u{s.p}qgvar" [s.ic] [s.ic] R hR p.eq_ p.gq p.bq2
                    (fun r => .batchOp (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
                        s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq) (e r))
                    r))
                r)))
          r)))
    r

/-- ⭐ The ExtraDW body at sync-BN is shard `r` of the row-typed body at `R·N` — generic in the
    row, the dispatch hypotheses T2's `mnv4ExtraDWBodyGraphB_faithful` takes. -/
theorem mnv4ExtraDWBodyGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat)
    (hN : 0 < N) (s : UibSpec) (hh : 0 < s.h) (p : UibParams s) (hq : s.preDWk ≠ 0)
    (hd : s.postDWk ≠ 0) (e : Fin R → SHlo (N * (s.ic * s.h * s.h)))
    (X : Vec ((R * N) * (s.ic * s.h * s.h)))
    (he : ∀ r, den (e r) = batchShard R N (s.ic * s.h * s.h) X r) (r : Fin R) :
    den (mnv4ExtraDWBodyGraphSync epsStr R hR N s p e r)
      = batchShard R N (s.oc * s.h * s.h) ((mnv4BodyOfRow (R * N) s p).fwd X) r := by
  have hm := nhw_ne_zero hN hh hh
  have hqc := den_batchOp_shard (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
    s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq) e X he
  have hqn := den_bnSyncSiteLA s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr s!"u{s.p}qgmu" s!"u{s.p}qgvar"
    [s.ic] [s.ic] R hR hm p.eq_ p.gq p.bq2 _ _ hqc
  have hec := den_batchOp_shard (N := N)
    (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
      s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be) _ _ hqn
  have hen := den_bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
    [s.ic * s.expand] [s.ic * s.expand] R hR hm p.ee p.ge p.be2 _ _ hec
  have her := den_relu_shard _ _ hen
  have hdc := den_batchOp_shard (N := N) (.depthwise (c := s.ic * s.expand) (h := s.h) (w := s.h)
    s!"%u{s.p}dW" s!"%zb{s.ic * s.expand}" p.Wd p.bd) _ _ her
  have hdn := den_bnSyncSiteLA s!"%u{s.p}dg" s!"%u{s.p}dbt" epsStr s!"u{s.p}dgmu" s!"u{s.p}dgvar"
    [s.ic * s.expand] [s.ic * s.expand] R hR hm p.ed p.gd p.bd2 _ _ hdc
  have hdr := den_relu_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
      s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz) _ _ hdr
  refine (den_bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR hm p.ez p.gz p.bz2 _ _ hpc r).trans
    (congrArg (fun z => batchShard R N (s.oc * s.h * s.h) z r) ?_)
  simp only [mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot, ite_eq_right hq,
    ite_eq_right hd, mnv4DWBnLayer, mnv4DWReluLayer, cbReluLayer, projLayer, CertLayer.comp_fwd,
    projB, cbReluB, dwbB, dwbReluB, denOp, Function.comp_apply]

/-- **ConvNeXt-like body at sync-BN** — pre-DW only (`postDWk = 0`): the absent depthwise emits no
    tokens, exactly as in T2. Three sync sites. -/
def mnv4ConvNeXtBodyGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (p : UibParams s) (e : Fin R → SHlo (N * (s.ic * s.h * s.h))) :
    Fin R → SHlo (N * (s.oc * s.h * s.h)) :=
  fun r => bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR p.ez p.gz p.bz2
    (fun r => .batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
          [s.ic * s.expand] [s.ic * s.expand] R hR p.ee p.ge p.be2
          (fun r => .batchOp (N := N) (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h)
              (w := s.h) s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
            (bnSyncSiteLA s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr s!"u{s.p}qgmu" s!"u{s.p}qgvar"
              [s.ic] [s.ic] R hR p.eq_ p.gq p.bq2
              (fun r => .batchOp (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
                  s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq) (e r))
              r))
          r)))
    r

theorem mnv4ConvNeXtBodyGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat)
    (hN : 0 < N) (s : UibSpec) (hh : 0 < s.h) (p : UibParams s) (hq : s.preDWk ≠ 0)
    (hd : s.postDWk = 0) (e : Fin R → SHlo (N * (s.ic * s.h * s.h)))
    (X : Vec ((R * N) * (s.ic * s.h * s.h)))
    (he : ∀ r, den (e r) = batchShard R N (s.ic * s.h * s.h) X r) (r : Fin R) :
    den (mnv4ConvNeXtBodyGraphSync epsStr R hR N s p e r)
      = batchShard R N (s.oc * s.h * s.h) ((mnv4BodyOfRow (R * N) s p).fwd X) r := by
  have hm := nhw_ne_zero hN hh hh
  have hqc := den_batchOp_shard (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
    s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq) e X he
  have hqn := den_bnSyncSiteLA s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr s!"u{s.p}qgmu" s!"u{s.p}qgvar"
    [s.ic] [s.ic] R hR hm p.eq_ p.gq p.bq2 _ _ hqc
  have hec := den_batchOp_shard (N := N)
    (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
      s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be) _ _ hqn
  have hen := den_bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
    [s.ic * s.expand] [s.ic * s.expand] R hR hm p.ee p.ge p.be2 _ _ hec
  have her := den_relu_shard _ _ hen
  have hpc := den_batchOp_shard (N := N)
    (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
      s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz) _ _ her
  refine (den_bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR hm p.ez p.gz p.bz2 _ _ hpc r).trans
    (congrArg (fun z => batchShard R N (s.oc * s.h * s.h) z r) ?_)
  simp only [mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot, ite_eq_right hq,
    ite_eq_left hd, mnv4DWBnLayer, cbReluLayer, projLayer, CertLayer.id'_fwd,
    CertLayer.comp_fwd, projB, cbReluB, dwbB, denOp, Function.comp_apply]

/-- **FFN body at sync-BN** — neither depthwise: expand → sync-BN → relu → project → sync-BN. -/
def mnv4FfnBodyGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (p : UibParams s) (e : Fin R → SHlo (N * (s.ic * s.h * s.h))) :
    Fin R → SHlo (N * (s.oc * s.h * s.h)) :=
  fun r => bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR p.ez p.gz p.bz2
    (fun r => .batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
          [s.ic * s.expand] [s.ic * s.expand] R hR p.ee p.ge p.be2
          (fun r => .batchOp (N := N) (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h)
              (w := s.h) s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be) (e r))
          r)))
    r

theorem mnv4FfnBodyGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat)
    (hN : 0 < N) (s : UibSpec) (hh : 0 < s.h) (p : UibParams s) (hq : s.preDWk = 0)
    (hd : s.postDWk = 0) (e : Fin R → SHlo (N * (s.ic * s.h * s.h)))
    (X : Vec ((R * N) * (s.ic * s.h * s.h)))
    (he : ∀ r, den (e r) = batchShard R N (s.ic * s.h * s.h) X r) (r : Fin R) :
    den (mnv4FfnBodyGraphSync epsStr R hR N s p e r)
      = batchShard R N (s.oc * s.h * s.h) ((mnv4BodyOfRow (R * N) s p).fwd X) r := by
  have hm := nhw_ne_zero hN hh hh
  have hec := den_batchOp_shard (N := N)
    (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
      s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be) e X he
  have hen := den_bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
    [s.ic * s.expand] [s.ic * s.expand] R hR hm p.ee p.ge p.be2 _ _ hec
  have her := den_relu_shard _ _ hen
  have hpc := den_batchOp_shard (N := N)
    (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
      s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz) _ _ her
  refine (den_bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR hm p.ez p.gz p.bz2 _ _ hpc r).trans
    (congrArg (fun z => batchShard R N (s.oc * s.h * s.h) z r) ?_)
  simp only [mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot, ite_eq_left hq,
    ite_eq_left hd, cbReluLayer, projLayer, CertLayer.id'_fwd, CertLayer.comp_fwd, projB, cbReluB,
    denOp, Function.comp_apply]

/-- **Strided block at sync-BN** — rows 1, 3 and 11: the BN-only pre-DW and the expand at `2h`, the
    post-DW (`.depthwiseStrided`, symmetric) carrying the stride, the project at `h`. Four sync
    sites; no skip (`ic ≠ oc`). -/
def mnv4StridedGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (p : UibParams s) (e : Fin R → SHlo (N * (s.ic * (2 * s.h) * (2 * s.h)))) :
    Fin R → SHlo (N * (s.oc * s.h * s.h)) :=
  fun r => bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR p.ez p.gz p.bz2
    (fun r => .batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (bnSyncSiteLA s!"%u{s.p}dg" s!"%u{s.p}dbt" epsStr s!"u{s.p}dgmu" s!"u{s.p}dgvar"
          [s.ic * s.expand] [s.ic * s.expand] R hR p.ed p.gd p.bd2
          (fun r => .batchOp (N := N)
            (.depthwiseStrided (c := s.ic * s.expand) (h := s.h) (w := s.h)
              s!"%u{s.p}dW" s!"%zb{s.ic * s.expand}" p.Wd p.bd)
            (.batchOp (N := N) (.relu (n := s.ic * s.expand * (2 * s.h) * (2 * s.h)))
              (bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
                [s.ic * s.expand] [s.ic * s.expand] R hR p.ee p.ge p.be2
                (fun r => .batchOp (N := N)
                  (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := 2 * s.h) (w := 2 * s.h)
                    s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
                  (bnSyncSiteLA s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr s!"u{s.p}qgmu"
                    s!"u{s.p}qgvar" [s.ic] [s.ic] R hR p.eq_ p.gq p.bq2
                    (fun r => .batchOp (N := N)
                      (.depthwise (c := s.ic) (h := 2 * s.h) (w := 2 * s.h)
                        s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq) (e r))
                    r))
                r)))
          r)))
    r

theorem mnv4StridedGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N : Nat)
    (hN : 0 < N) (s : UibSpec) (hh : 0 < s.h) (p : UibParams s) (hq : s.preDWk ≠ 0)
    (e : Fin R → SHlo (N * (s.ic * (2 * s.h) * (2 * s.h))))
    (X : Vec ((R * N) * (s.ic * (2 * s.h) * (2 * s.h))))
    (he : ∀ r, den (e r) = batchShard R N (s.ic * (2 * s.h) * (2 * s.h)) X r) (r : Fin R) :
    den (mnv4StridedGraphSync epsStr R hR N s p e r)
      = batchShard R N (s.oc * s.h * s.h) ((mnv4StridedBodyOfRow (R * N) s p).fwd X) r := by
  have hm := nhw_ne_zero hN hh hh
  have h2 : 0 < 2 * s.h := by omega
  have hm2 := nhw_ne_zero hN h2 h2
  have hqc := den_batchOp_shard (N := N) (.depthwise (c := s.ic) (h := 2 * s.h) (w := 2 * s.h)
    s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq) e X he
  have hqn := den_bnSyncSiteLA s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr s!"u{s.p}qgmu" s!"u{s.p}qgvar"
    [s.ic] [s.ic] R hR hm2 p.eq_ p.gq p.bq2 _ _ hqc
  have hec := den_batchOp_shard (N := N)
    (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := 2 * s.h) (w := 2 * s.h)
      s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be) _ _ hqn
  have hen := den_bnSyncSiteLA s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr s!"u{s.p}egmu" s!"u{s.p}egvar"
    [s.ic * s.expand] [s.ic * s.expand] R hR hm2 p.ee p.ge p.be2 _ _ hec
  have her := den_relu_shard _ _ hen
  have hdc := den_batchOp_shard (N := N)
    (.depthwiseStrided (c := s.ic * s.expand) (h := s.h) (w := s.h)
      s!"%u{s.p}dW" s!"%zb{s.ic * s.expand}" p.Wd p.bd) _ _ her
  have hdn := den_bnSyncSiteLA s!"%u{s.p}dg" s!"%u{s.p}dbt" epsStr s!"u{s.p}dgmu" s!"u{s.p}dgvar"
    [s.ic * s.expand] [s.ic * s.expand] R hR hm p.ed p.gd p.bd2 _ _ hdc
  have hdr := den_relu_shard _ _ hdn
  have hpc := den_batchOp_shard (N := N)
    (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
      s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz) _ _ hdr
  refine (den_bnSyncSiteLA s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr s!"u{s.p}pgmu" s!"u{s.p}pgvar"
    [s.oc] [s.oc] R hR hm p.ez p.gz p.bz2 _ _ hpc r).trans
    (congrArg (fun z => batchShard R N (s.oc * s.h * s.h) z r) ?_)
  simp only [mnv4StridedBodyOfRow, mnv4UibStridedBody, mnv4PreDWSlot, ite_eq_right hq,
    mnv4DWBnLayer, mnv4DWReluStridedLayer, cbReluLayer, projLayer, CertLayer.comp_fwd, projB,
    cbReluB, dwbB, dwbReluBstrided, denOp, Function.comp_apply]

/-- ⭐ **One skip row at sync-BN: its body's family, plus the identity skip, replica by replica** —
    the peer of `mnv4SkipGraphB`, and a named combinator for the same reason: the add needs the
    block's input family twice, and kept folded the whole-net term stays linear in the depth. -/
def mnv4SkipGraphSync {R N n : Nat} (body : (Fin R → SHlo (N * n)) → Fin R → SHlo (N * n))
    (e : Fin R → SHlo (N * n)) : Fin R → SHlo (N * n) :=
  fun r => .addVB (body e r) (e r)

/-- A skip row at sync-BN is shard `r` of `residual` of whatever its body shards to — generic in
    the body, so one lemma serves all eighteen and the body's shard lemma is the only input. -/
theorem mnv4SkipGraphSync_shard {R N n : Nat}
    (body : (Fin R → SHlo (N * n)) → Fin R → SHlo (N * n)) (f : Vec ((R * N) * n) → Vec ((R * N) * n))
    (hb : ∀ (e' : Fin R → SHlo (N * n)) (X' : Vec ((R * N) * n)),
      (∀ r, den (e' r) = batchShard R N n X' r) → ∀ r, den (body e' r) = batchShard R N n (f X') r)
    (e : Fin R → SHlo (N * n)) (X : Vec ((R * N) * n))
    (he : ∀ r, den (e r) = batchShard R N n X r) (r : Fin R) :
    den (mnv4SkipGraphSync body e r) = batchShard R N n (Proofs.residual f X) r :=
  den_addVB_shard (body e) e (f X) X (hb e X he) he r

/-- Head at sync-BN, over the replica family, timm's order: 1×1 conv → sync-BN → relu, GAP,
    `conv_head` 1×1 conv → sync-BN → relu on the pooled features (that BN's statistics over the
    GLOBAL batch alone), dense — with the two `castIdx` relabellings T2's head carries. -/
def mnv4HeadGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : Fin R → SHlo (N * (c * h * w))) :
    Fin R → SHlo (N * nCls) :=
  fun r => .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (castIdx (mnv4_pool11 N oc).symm
      (.batchOp (N := N) (.relu (n := oc * 1 * 1))
        (bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR ε2 γ2 β2
          (fun r => .batchOp (N := N) (.conv (h := 1) (w := 1) "%hW" s!"%zb{oc}" W2 b2)
            (castIdx (mnv4_pool11 N mid)
              (.batchOp (N := N) (.gap (c := mid) (h := h) (w := w))
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (bnSyncSiteLA "%h1g" "%h1bt" epsStr "h1gmu" "h1gvar" [mid] [mid] R hR ε1 γ1 β1
                    (fun r => .batchOp (N := N) (.conv (h := h) (w := w) "%h1W" s!"%zb{mid}" W1 b1)
                      (e r))
                    r)))))
          r)))

theorem mnv4HeadGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {c mid oc nCls : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (hε1 : 0 < ε1) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (hε2 : 0 < ε2) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : Fin R → SHlo (N * (c * h * w)))
    (X : Vec ((R * N) * (c * h * w))) (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r)
    (r : Fin R) :
    den (mnv4HeadGraphSync epsStr R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd e r)
      = batchShard R N nCls
          ((mnv4Head (R * N) (cbReluLayer (h := h) (w := w) (R * N) W1 b1 ε1 hε1 γ1 β1)
            (gapLayer (R * N) (c := mid) (h := h) (w := w))
            (cbReluLayer (h := 1) (w := 1) (R * N) W2 b2 ε2 hε2 γ2 β2)
            (denseLayer (R * N) Wd bd)).fwd X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hm1 := nhw_ne_zero hN Nat.one_pos Nat.one_pos
  have hc1 := den_batchOp_shard (N := N) (.conv (h := h) (w := w) "%h1W" s!"%zb{mid}" W1 b1) e X he
  have hn1 := den_bnSyncSiteLA "%h1g" "%h1bt" epsStr "h1gmu" "h1gvar" [mid] [mid]
    R hR hm ε1 γ1 β1 _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hg := den_batchOp_shard (N := N) (.gap (c := mid) (h := h) (w := w)) _ _ hr1
  have hp := den_castIdx_shard (by rw [Nat.mul_one, Nat.mul_one] : mid = mid * 1 * 1)
    (mnv4_pool11 N mid) _ _ hg
  have hc2 := den_batchOp_shard (N := N) (.conv (h := 1) (w := 1) "%hW" s!"%zb{oc}" W2 b2) _ _ hp
  have hn2 := den_bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc]
    R hR hm1 ε2 γ2 β2 _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hf := den_castIdx_shard (by rw [Nat.mul_one, Nat.mul_one] : oc * 1 * 1 = oc)
    (mnv4_pool11 N oc).symm _ _ hr2
  exact den_batchOp_shard (N := N) (.dense "%Wd" "%bd" Wd bd) _ _ hf r

-- ════════════════════════════════════════════════════════════════
-- § The five resolution groups, at their table rows
--   Each closes on `mnv4Res*Layer (R * N) w` through T2's `_fwd_apply` peel — proved between
--   variables there — so no `CertLayer.comp` is unfolded at a literal width here.
-- ════════════════════════════════════════════════════════════════

/-- Trunk group **Res28** at sync-BN — rows 1–2. -/
def mnv4Res28GraphSync (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (48 * 56 * 56))) :
    Fin R → SHlo (N * (80 * 28 * 28)) :=
  mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row2 w.b2)
    (mnv4StridedGraphSync epsStr R hR N mnv4Row1 w.b1 e)

theorem mnv4Res28GraphSync_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (epsStr : String)
    {nCls : Nat} (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (48 * 56 * 56)))
    (X : Vec ((R * N) * (48 * 56 * 56))) (he : ∀ r, den (e r) = batchShard R N (48 * 56 * 56) X r)
    (r : Fin R) :
    den (mnv4Res28GraphSync R hR N epsStr w e r)
      = batchShard R N (80 * 28 * 28) ((mnv4Res28Layer (R * N) w).fwd X) r := by
  rw [mnv4Res28Layer_fwd_apply, CertLayer.residual_fwd]
  have b1 := mnv4StridedGraphSync_shard epsStr R hR N hN mnv4Row1 (by decide) w.b1 (by decide)
    e X he
  exact mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row2 w.b2)
    (mnv4BodyOfRow (R * N) mnv4Row2 w.b2).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row2 (by decide) w.b2 (by decide)
      (by decide))
    _ _ b1 r

/-- Trunk group **Res14a** at sync-BN — rows 3–6. -/
def mnv4Res14aGraphSync (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (80 * 28 * 28))) :
    Fin R → SHlo (N * (160 * 14 * 14)) :=
  mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row6 w.b6)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row5 w.b5)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row4 w.b4)
    (mnv4StridedGraphSync epsStr R hR N mnv4Row3 w.b3 e)))

theorem mnv4Res14aGraphSync_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (80 * 28 * 28)))
    (X : Vec ((R * N) * (80 * 28 * 28))) (he : ∀ r, den (e r) = batchShard R N (80 * 28 * 28) X r)
    (r : Fin R) :
    den (mnv4Res14aGraphSync R hR N epsStr w e r)
      = batchShard R N (160 * 14 * 14) ((mnv4Res14aLayer (R * N) w).fwd X) r := by
  rw [mnv4Res14aLayer_fwd_apply]
  simp only [CertLayer.residual_fwd]
  have b3 := mnv4StridedGraphSync_shard epsStr R hR N hN mnv4Row3 (by decide) w.b3 (by decide)
    e X he
  have b4 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row4 w.b4)
    (mnv4BodyOfRow (R * N) mnv4Row4 w.b4).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row4 (by decide) w.b4 (by decide)
      (by decide)) _ _ b3
  have b5 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row5 w.b5)
    (mnv4BodyOfRow (R * N) mnv4Row5 w.b5).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row5 (by decide) w.b5 (by decide)
      (by decide)) _ _ b4
  exact mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row6 w.b6)
    (mnv4BodyOfRow (R * N) mnv4Row6 w.b6).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row6 (by decide) w.b6 (by decide)
      (by decide)) _ _ b5 r

/-- Trunk group **Res14b** at sync-BN — rows 7–10: ExtraDW, ConvNeXt, FFN, ConvNeXt. -/
def mnv4Res14bGraphSync (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (160 * 14 * 14))) :
    Fin R → SHlo (N * (160 * 14 * 14)) :=
  mnv4SkipGraphSync (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row10 w.b10)
    (mnv4SkipGraphSync (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row9 w.b9)
    (mnv4SkipGraphSync (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row8 w.b8)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row7 w.b7) e)))

theorem mnv4Res14bGraphSync_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (160 * 14 * 14)))
    (X : Vec ((R * N) * (160 * 14 * 14)))
    (he : ∀ r, den (e r) = batchShard R N (160 * 14 * 14) X r) (r : Fin R) :
    den (mnv4Res14bGraphSync R hR N epsStr w e r)
      = batchShard R N (160 * 14 * 14) ((mnv4Res14bLayer (R * N) w).fwd X) r := by
  rw [mnv4Res14bLayer_fwd_apply]
  simp only [CertLayer.residual_fwd]
  have b7 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row7 w.b7)
    (mnv4BodyOfRow (R * N) mnv4Row7 w.b7).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row7 (by decide) w.b7 (by decide)
      (by decide)) e X he
  have b8 := mnv4SkipGraphSync_shard (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row8 w.b8)
    (mnv4BodyOfRow (R * N) mnv4Row8 w.b8).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN mnv4Row8 (by decide) w.b8 (by decide)
      (by decide)) _ _ b7
  have b9 := mnv4SkipGraphSync_shard (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row9 w.b9)
    (mnv4BodyOfRow (R * N) mnv4Row9 w.b9).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN mnv4Row9 (by decide) w.b9 (by decide)
      (by decide)) _ _ b8
  exact mnv4SkipGraphSync_shard (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row10 w.b10)
    (mnv4BodyOfRow (R * N) mnv4Row10 w.b10).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN mnv4Row10 (by decide) w.b10 (by decide)
      (by decide)) _ _ b9 r

/-- Trunk group **Res7a** at sync-BN — rows 11–15. -/
def mnv4Res7aGraphSync (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (160 * 14 * 14))) :
    Fin R → SHlo (N * (256 * 7 * 7)) :=
  mnv4SkipGraphSync (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row15 w.b15)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row14 w.b14)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row13 w.b13)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row12 w.b12)
    (mnv4StridedGraphSync epsStr R hR N mnv4Row11 w.b11 e))))

theorem mnv4Res7aGraphSync_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (160 * 14 * 14)))
    (X : Vec ((R * N) * (160 * 14 * 14)))
    (he : ∀ r, den (e r) = batchShard R N (160 * 14 * 14) X r) (r : Fin R) :
    den (mnv4Res7aGraphSync R hR N epsStr w e r)
      = batchShard R N (256 * 7 * 7) ((mnv4Res7aLayer (R * N) w).fwd X) r := by
  rw [mnv4Res7aLayer_fwd_apply]
  simp only [CertLayer.residual_fwd]
  have b11 := mnv4StridedGraphSync_shard epsStr R hR N hN mnv4Row11 (by decide) w.b11
    (by decide) e X he
  have b12 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row12 w.b12)
    (mnv4BodyOfRow (R * N) mnv4Row12 w.b12).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row12 (by decide) w.b12 (by decide)
      (by decide)) _ _ b11
  have b13 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row13 w.b13)
    (mnv4BodyOfRow (R * N) mnv4Row13 w.b13).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row13 (by decide) w.b13 (by decide)
      (by decide)) _ _ b12
  have b14 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row14 w.b14)
    (mnv4BodyOfRow (R * N) mnv4Row14 w.b14).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row14 (by decide) w.b14 (by decide)
      (by decide)) _ _ b13
  exact mnv4SkipGraphSync_shard (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row15 w.b15)
    (mnv4BodyOfRow (R * N) mnv4Row15 w.b15).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN mnv4Row15 (by decide) w.b15 (by decide)
      (by decide)) _ _ b14 r

/-- Trunk group **Res7b** at sync-BN — rows 16–21. -/
def mnv4Res7bGraphSync (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (256 * 7 * 7))) :
    Fin R → SHlo (N * (256 * 7 * 7)) :=
  mnv4SkipGraphSync (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row21 w.b21)
    (mnv4SkipGraphSync (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row20 w.b20)
    (mnv4SkipGraphSync (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row19 w.b19)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row18 w.b18)
    (mnv4SkipGraphSync (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row17 w.b17)
    (mnv4SkipGraphSync (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row16 w.b16) e)))))

theorem mnv4Res7bGraphSync_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (256 * 7 * 7)))
    (X : Vec ((R * N) * (256 * 7 * 7))) (he : ∀ r, den (e r) = batchShard R N (256 * 7 * 7) X r)
    (r : Fin R) :
    den (mnv4Res7bGraphSync R hR N epsStr w e r)
      = batchShard R N (256 * 7 * 7) ((mnv4Res7bLayer (R * N) w).fwd X) r := by
  rw [mnv4Res7bLayer_fwd_apply]
  simp only [CertLayer.residual_fwd]
  have b16 := mnv4SkipGraphSync_shard (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row16 w.b16)
    (mnv4BodyOfRow (R * N) mnv4Row16 w.b16).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN mnv4Row16 (by decide) w.b16 (by decide)
      (by decide)) e X he
  have b17 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row17 w.b17)
    (mnv4BodyOfRow (R * N) mnv4Row17 w.b17).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row17 (by decide) w.b17 (by decide)
      (by decide)) _ _ b16
  have b18 := mnv4SkipGraphSync_shard (mnv4ExtraDWBodyGraphSync epsStr R hR N mnv4Row18 w.b18)
    (mnv4BodyOfRow (R * N) mnv4Row18 w.b18).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN mnv4Row18 (by decide) w.b18 (by decide)
      (by decide)) _ _ b17
  have b19 := mnv4SkipGraphSync_shard (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row19 w.b19)
    (mnv4BodyOfRow (R * N) mnv4Row19 w.b19).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN mnv4Row19 (by decide) w.b19 (by decide)
      (by decide)) _ _ b18
  have b20 := mnv4SkipGraphSync_shard (mnv4FfnBodyGraphSync epsStr R hR N mnv4Row20 w.b20)
    (mnv4BodyOfRow (R * N) mnv4Row20 w.b20).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN mnv4Row20 (by decide) w.b20 (by decide)
      (by decide)) _ _ b19
  exact mnv4SkipGraphSync_shard (mnv4ConvNeXtBodyGraphSync epsStr R hR N mnv4Row21 w.b21)
    (mnv4BodyOfRow (R * N) mnv4Row21 w.b21).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN mnv4Row21 (by decide) w.b21 (by decide)
      (by decide)) _ _ b20 r

/-- The fused stage's shard lemma, restated at `mnv4FusedStack (R * N) w` — so the whole-net proof
    never unfolds the stack, for the reason `mnv4FusedStack_graph_faithful` records. -/
theorem mnv4FusedStack_graphSync_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : Fin R → SHlo (N * (32 * 112 * 112))) (X : Vec ((R * N) * (32 * 112 * 112)))
    (he : ∀ r, den (e r) = batchShard R N (32 * 112 * 112) X r) (r : Fin R) :
    den (mnv4FusedGraphSync epsStr R hR N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
          w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt e r)
      = batchShard R N (48 * 56 * 56) ((mnv4FusedStack (R * N) w).fwd X) r :=
  mnv4FusedGraphSync_shard epsStr R hR N 56 56 hN (by norm_num) (by norm_num)
    w.f0cW w.f0cb w.f0cE w.hf0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE w.hf0pE w.f0pg w.f0pbt e X he r

/-- The head's, restated at `mnv4HeadStack (R * N) w`. Same reason. -/
theorem mnv4HeadStack_graphSync_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : Fin R → SHlo (N * (256 * 7 * 7))) (X : Vec ((R * N) * (256 * 7 * 7)))
    (he : ∀ r, den (e r) = batchShard R N (256 * 7 * 7) X r) (r : Fin R) :
    den (mnv4HeadGraphSync epsStr R hR N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt
          w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd e r)
      = batchShard R N nCls ((mnv4HeadStack (R * N) w).fwd X) r :=
  mnv4HeadGraphSync_shard epsStr R hR N 7 7 hN (by norm_num) (by norm_num)
    w.h1W w.h1b w.h1E w.hh1E w.h1g w.h1bt w.hW w.hb w.hE w.hhE w.hg w.hbt w.Wd w.bd e X he r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel MobileNetV4-Conv-M forward graph, over the replica family.** T2's
    `mnv4FwdGraphBFull` with every one of its 77 BatchNorms a `bnSyncSiteLA` over all `R` replicas;
    parameter names are read off the rows and collective tags are the render's. -/
def mnv4FwdGraphSyncFull (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : Fin R → SHlo (N * (3 * 224 * 224))) : Fin R → SHlo (N * nCls) :=
  mnv4HeadGraphSync epsStr R hR N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt
    w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd
    (mnv4Res7bGraphSync R hR N epsStr w
      (mnv4Res7aGraphSync R hR N epsStr w
        (mnv4Res14bGraphSync R hR N epsStr w
          (mnv4Res14aGraphSync R hR N epsStr w
            (mnv4Res28GraphSync R hR N epsStr w
              (mnv4FusedGraphSync epsStr R hR N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt
                (mnv4StemGraphSync epsStr R hR N 112 112 w.sW w.sb w.sE w.sg w.sbt e)))))))

/-- ⭐⭐ **T2 at synchronised BatchNorm: replica `r`'s forward IS shard `r` of the global-batch
    forward.** Given that the replicas' inputs are the shards of one batch `X` of `R·N` images,
    the sync-BN graph on replica `r` denotes `batchShard r` of `mobilenetv4ForwardBFull (R * N)
    w X` — the committed batch-BN forward, at the global batch. Eight stage lemmas — the stem, the
    fused stage, the five resolution groups, the head — the shard hypothesis threaded from each
    into the next, over the forward's own prefixes. -/
theorem mnv4FwdGraphSyncFull_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : Fin R → SHlo (N * (3 * 224 * 224))) (X : Vec ((R * N) * (3 * 224 * 224)))
    (he : ∀ r, den (e r) = batchShard R N (3 * 224 * 224) X r) (r : Fin R) :
    den (mnv4FwdGraphSyncFull R hR N epsStr w e r)
      = batchShard R N nCls (mobilenetv4ForwardBFull (R * N) w X) r := by
  have h112 : 0 < 112 := by norm_num
  have s0 := mnv4StemGraphSync_shard epsStr R hR N 112 112 hN h112 h112
    w.sW w.sb w.sE w.sg w.sbt e X he
  have s1 := mnv4FusedStack_graphSync_shard R hR N hN epsStr w _ _ s0
  have s2 := mnv4Res28GraphSync_shard R hR N hN epsStr w _ _ s1
  have s3 := mnv4Res14aGraphSync_shard R hR N hN epsStr w _ _ s2
  have s4 := mnv4Res14bGraphSync_shard R hR N hN epsStr w _ _ s3
  have s5 := mnv4Res7aGraphSync_shard R hR N hN epsStr w _ _ s4
  have s6 := mnv4Res7bGraphSync_shard R hR N hN epsStr w _ _ s5
  unfold mobilenetv4ForwardBFull mnv4Pre6 mnv4Pre5 mnv4Pre4 mnv4Pre3 mnv4Pre2 mnv4Pre1 mnv4Pre0
  exact mnv4HeadStack_graphSync_shard R hR N hN epsStr w _ _ s6 r

end StableHLO

end Proofs
