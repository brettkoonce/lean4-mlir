import LeanMlir.Proofs.Nets.ResNet.ResNet34FullB
import LeanMlir.Proofs.Foundation.DataParallelSync

/-! # ResNet-34's data-parallel forward at SYNCHRONISED BatchNorm — replica `r` IS shard `r`

`ResNet34FullB.lean` (T2) says the typed batch-BN graph denotes `resnet34ForwardB_full N w` on
one device. `ResNet34RenderB` renders the data-parallel step differently since 2026-09-21: at
`replicas > 1` every BatchNorm site is `bnFwdSite`'s sync-BN composition — this replica's mean
all-reduced, then Chan's `σ²_r + (μ_r − μ)²` all-reduced, packed, then `bnSyncF` — so the
statistics each replica normalises by are the GLOBAL batch's. This file is T2's data-parallel
twin: that forward graph, stated as a family over the `R` replicas, denotes on replica `r`
exactly `batchShard r` of the single-device forward at the global batch `R·N`.

    den (resnet34FwdGraphSync_full R hR N epsStr w e r)
      = batchShard R N nCls (resnet34ForwardB_full (R * N) w X) r

given that each replica's input is its shard of one global batch `X`. ⭐ **The spec does not
move**: the right-hand side is the committed `resnet34ForwardB_full`, at `N := R·N`.

## How it is proved

By induction on the chain, one block at a time, exactly as T2 is — with the shard hypothesis
`∀ r, den (e r) = batchShard R N _ X r` as the invariant carried from block to block:

* every conv, relu, pool, GAP and dense node is a per-example lift, and sharding commutes with it
  (`batchShard_batchMap`, and `relu` pointwise) — `den_batchOp_shard`, `den_relu_shard`;
* every BatchNorm site is `bnSyncSiteLA`, whose shard lemma `den_bnSyncSiteLA` is
  `DataParallelSync.den_bnSyncF_allReduce` (P1 on the graph) read at the network's index.

## The index seam

The conv/relu chain runs at the left-assoc index `N·(c·h·w)`; `bnSyncF` and its statistics nodes
at `N·(c·(h·w))`, the index `bnBatchTensor4` is stated at. `bnBatchF` hides that seam inside its
`den` (`bnBatchLA` is `bnBatchTensor4` conjugated by the `mul_assoc` relabelling). The sync site
cannot, because its statistics subgraph is shared across replicas, so it carries the relabelling
as `castIdx` on the AST value — `h ▸ e`, the move `ConvNeXtRenderB`'s `reassocB` already makes.
It changes no emitted text: `skel` never sees an index, and the render writes the same SSA name
at both types (`bnFwdSite`'s `zbn` operand beside its `zin` one).

## What is NOT claimed here

⚠ The backward and the parameter collectives are the T3 half (`ResNet34SyncStepTieB.lean`). ⚠ That
the `R` replicas' inputs ARE the shards of one batch is the driver's, as in
`DataParallelSync.lean`. ⚠ The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The index cast, and sharding through it
-- ════════════════════════════════════════════════════════════════

/-- **Relabel an AST value's index along a proved equality.** `h ▸ e`: the same graph, typed at
    `m` instead of `n`. The emitted text does not change, because `skel` erases indices. -/
def castIdx {n m : Nat} (h : n = m) (e : SHlo n) : SHlo m := h ▸ e

theorem den_castIdx {n m : Nat} (h : n = m) (e : SHlo n) :
    den (castIdx h e) = fun i => den e (Fin.cast h.symm i) := by
  subst h; rfl

/-- The `mul_assoc` relabelling under `N * ·` — the seam between the network's left-assoc
    `N·(c·h·w)` and the BatchNorm ops' `N·(c·(h·w))`. -/
theorem laAssoc (N oc h w : Nat) : N * (oc * h * w) = N * (oc * (h * w)) :=
  congrArg (N * ·) (Nat.mul_assoc oc h w)

/-- **Sharding commutes with relabelling the per-example index.** The batch axis is outside the
    per-example one, so relabelling within an example and cutting the batch do not interact. -/
theorem batchShard_castIdx {R N a b : Nat} (hab : a = b) (X : Vec ((R * N) * a)) (r : Fin R) :
    batchShard R N b (fun i => X (Fin.cast (congrArg ((R * N) * ·) hab).symm i)) r
      = fun i => batchShard R N a X r (Fin.cast (congrArg (N * ·) hab).symm i) := by
  subst hab; rfl

-- ════════════════════════════════════════════════════════════════
-- § The non-BN nodes: sharding commutes with every per-example lift
-- ════════════════════════════════════════════════════════════════

/-- A per-example node on every replica denotes the shard of the same node on the global batch. -/
theorem den_batchOp_shard {R N a b : Nat} (op : BatchableOp a b) (e : Fin R → SHlo (N * a))
    (X : Vec ((R * N) * a)) (he : ∀ r, den (e r) = batchShard R N a X r) (r : Fin R) :
    den (.batchOp (N := N) op (e r)) = batchShard R N b (batchMap (R * N) (denOp op) X) r := by
  rw [den_batchOp, he, batchShard_batchMap]

/-- …stated for relu at the whole-batch `relu`, the form the committed forward is written in. -/
theorem den_relu_shard {R N n : Nat} (e : Fin R → SHlo (N * n)) (X : Vec ((R * N) * n))
    (he : ∀ r, den (e r) = batchShard R N n X r) (r : Fin R) :
    den (.batchOp (N := N) (.relu (n := n)) (e r)) = batchShard R N n (relu ((R * N) * n) X) r := by
  rw [den_batchOp_relu_eq_reluF, reluF_faithful, he]
  rfl

/-- The residual fan-in on every replica is the shard of the global one. -/
theorem den_addVB_shard {R N n : Nat} (a b : Fin R → SHlo (N * n)) (A B : Vec ((R * N) * n))
    (ha : ∀ r, den (a r) = batchShard R N n A r) (hb : ∀ r, den (b r) = batchShard R N n B r)
    (r : Fin R) :
    den (.addVB (a r) (b r)) = batchShard R N n (fun j => A j + B j) r := by
  rw [den_addVB, ha, hb]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The BatchNorm site
-- ════════════════════════════════════════════════════════════════

/-- **One sync-BN forward site, at the network index, on replica `r`** — `bnFwdSite`'s
    `replicas > 1` branch: `bnSyncF` of this replica's operand, reading `syncStats` over all `R`
    replicas' operands (the mean collective `t`, then Chan's variance collective `t'`), with the
    `mul_assoc` relabelling on the way in and out. -/
def bnSyncSiteLA (gN bN es t t' : String) (ds ds' : List Nat) (R : Nat) (hR : 0 < R)
    {N oc h w : Nat} (ε : ℝ) (γ β : Vec oc) (x : Fin R → SHlo (N * (oc * h * w))) (r : Fin R) :
    SHlo (N * (oc * h * w)) :=
  castIdx (laAssoc N oc h w).symm
    (.bnSyncF gN bN es ε γ β (castIdx (laAssoc N oc h w) (x r))
      (syncStats R hR t t' ds ds' (fun r' => castIdx (laAssoc N oc h w) (x r'))))

/-- ⭐⭐ **The sync-BN site on replica `r` is shard `r` of the global-batch BatchNorm.**
    `den_bnSyncF_allReduce` (P1 on the graph), carried across the `mul_assoc` seam: the right-hand
    side is `bnBatchLA` — what `bnBatchF` denotes — at `N := R·N`. -/
theorem den_bnSyncSiteLA (gN bN es t t' : String) (ds ds' : List Nat) (R : Nat) (hR : 0 < R)
    {N oc h w : Nat} (hm : N * (h * w) ≠ 0) (hM : (R * N) * (h * w) ≠ 0) (ε : ℝ) (γ β : Vec oc)
    (x : Fin R → SHlo (N * (oc * h * w))) (X : Vec ((R * N) * (oc * h * w)))
    (hx : ∀ r, den (x r) = batchShard R N (oc * h * w) X r) (r : Fin R) :
    den (bnSyncSiteLA gN bN es t t' ds ds' R hR ε γ β x r)
      = batchShard R N (oc * h * w) (bnBatchLA (R * N) oc h w ε γ β X) r := by
  have hx' : ∀ r, den (castIdx (laAssoc N oc h w) (x r))
      = batchShard R N (oc * (h * w))
          (fun i => X (Fin.cast (congrArg ((R * N) * ·) (Nat.mul_assoc oc h w)).symm i)) r := by
    intro r
    rw [den_castIdx, hx]
    exact (batchShard_castIdx (Nat.mul_assoc oc h w) X r).symm
  unfold bnSyncSiteLA
  rw [den_castIdx, den_bnSyncF_allReduce R hR hm hM gN bN es t t' ds ds' ε γ β _ _ hx' r]
  exact (batchShard_castIdx (Nat.mul_assoc oc h w).symm _ r).symm

-- ════════════════════════════════════════════════════════════════
-- § Per-block replica families + their shard lemmas
--   Names and tags are `ResNet34RenderB`'s: BN site `{p}g1` gathers `%arsum{p}g1mu` /
--   `%armean{p}g1var`, each over a `[c]` statistic.
-- ════════════════════════════════════════════════════════════════

/-- The reduction width a BatchNorm site needs nonzero, from the three positive dimensions. -/
theorem nhw_ne_zero {N h w : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) : N * (h * w) ≠ 0 :=
  Nat.pos_iff_ne_zero.mp (Nat.mul_pos hN (Nat.mul_pos hh hw))

/-- Identity basic block at sync-BN, over the replica family: `relu(addV(bn₂(conv₂(relu(bn₁(conv₁
    e)))), e))` with both BatchNorms `bnSyncSiteLA`. -/
def r34IdGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat}
    (pw : R34IdW c) (e : Fin R → SHlo (N * (c * h * w))) : Fin R → SHlo (N * (c * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := c * h * w))
    (.addVB
      (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [c] [c] R hR
        pw.ε₂ pw.γ₂ pw.β₂
        (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" c) pw.W₂ pw.b₂)
          (.batchOp (N := N) (.relu (n := c * h * w))
            (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [c] [c] R hR
              pw.ε₁ pw.γ₁ pw.β₁
              (fun r => .batchOp (N := N)
                (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" c) pw.W₁ pw.b₁) (e r)) r)))
        r)
      (e r))

theorem r34IdGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R34IdW c)
    (e : Fin R → SHlo (N * (c * h * w))) (X : Vec ((R * N) * (c * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r) (r : Fin R) :
    den (r34IdGraphSync p epsStr R hR N h w pw e r)
      = batchShard R N (c * h * w) (r34IdB (R * N) h w pw X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  have hc1 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" c) pw.W₁ pw.b₁) e X he
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [c] [c]
    R hR hm hM pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" c) pw.W₂ pw.b₂) _ _ hr1
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [c] [c]
    R hR hm hM pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have ha := den_addVB_shard _ e _ X hn2 he
  exact den_relu_shard _ _ ha r

/-- Downsample basic block at sync-BN, over the replica family: `relu(addV(bnₚ(projection),
    bn₂(conv₂(relu(bn₁(convStrided₁ e))))))` — projection first, as `residualProj` and the render
    order it. Three sync sites. -/
def r34DownGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (pw : R34DownW ic oc) (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc] R hR
        pw.εp pw.γp pw.βp
        (fun r => .batchOp (N := N)
          (.convStrided (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) (e r)) r)
      (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [oc] [oc] R hR
        pw.ε₂ pw.γ₂ pw.β₂
        (fun r => .batchOp (N := N) (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" oc) pw.W₂ pw.b₂)
          (.batchOp (N := N) (.relu (n := oc * h * w))
            (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [oc] [oc] R hR
              pw.ε₁ pw.γ₁ pw.β₁
              (fun r => .batchOp (N := N)
                (.convStrided (h := h) (w := w) s!"%{p}W1" (biasName false "" oc) pw.W₁ pw.b₁) (e r))
              r)))
        r))

theorem r34DownGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R34DownW ic oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (r34DownGraphSync p epsStr R hR N h w pw e r)
      = batchShard R N (oc * h * w) (r34DownB (R * N) h w pw X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  have hcp := den_batchOp_shard (N := N)
    (.convStrided (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e X he
  have hnp := den_bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc]
    R hR hm hM pw.εp pw.γp pw.βp _ _ hcp
  have hc1 := den_batchOp_shard (N := N)
    (.convStrided (h := h) (w := w) s!"%{p}W1" (biasName false "" oc) pw.W₁ pw.b₁) e X he
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [oc] [oc]
    R hR hm hM pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" oc) pw.W₂ pw.b₂) _ _ hr1
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [oc] [oc]
    R hR hm hM pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have ha := den_addVB_shard _ _ _ _ hnp hn2
  exact den_relu_shard _ _ ha r

/-- Stem at sync-BN, over the replica family: 7×7/s2 conv → sync-BN → relu → 3×3/s2 max-pool. -/
def r34StemGraphSync (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.maxPool3s2 (c := oc) (h := h) (w := w))
    (.batchOp (N := N) (.relu (n := oc * (2 * h) * (2 * w)))
      (bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc] R hR εs γs βs
        (fun r => .batchOp (N := N)
          (.convStrided (h := 2 * h) (w := 2 * w) "%sW" (biasName false "" oc) Ws bs) (e r)) r))

theorem r34StemGraphSync_shard (epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : Fin R → SHlo (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (X : Vec ((R * N) * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * (2 * h)) * (2 * (2 * w))) X r) (r : Fin R) :
    den (r34StemGraphSync epsStr R hR N h w Ws bs εs γs βs e r)
      = batchShard R N (oc * h * w) (r34StemB (R * N) h w Ws bs εs γs βs X) r := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm := nhw_ne_zero hN h2h h2w
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) h2h h2w
  have hc := den_batchOp_shard (N := N)
    (.convStrided (h := 2 * h) (w := 2 * w) "%sW" (biasName false "" oc) Ws bs) e X he
  have hn := den_bnSyncSiteLA "%sg" "%sbt" epsStr "sgmu" "sgvar" [oc] [oc]
    R hR hm hM εs γs βs _ _ hc
  have hr := den_relu_shard _ _ hn
  exact den_batchOp_shard (N := N) (.maxPool3s2 (c := oc) (h := h) (w := w)) _ _ hr r

/-- Head over the replica family: GAP then dense — no BatchNorm, so T2's head graph per replica. -/
def r34HeadGraphSync {R : Nat} (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls)
    (e : Fin R → SHlo (N * (c * h * w))) : Fin R → SHlo (N * nCls) :=
  fun r => r34HeadGraphB N h w Wd bd (e r)

theorem r34HeadGraphSync_shard {R : Nat} (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls)
    (bd : Vec nCls) (e : Fin R → SHlo (N * (c * h * w))) (X : Vec ((R * N) * (c * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r) (r : Fin R) :
    den (r34HeadGraphSync N h w Wd bd e r) = batchShard R N nCls (r34HeadB (R * N) h w Wd bd X) r := by
  have hg := den_batchOp_shard (N := N) (.gap (c := c) (h := h) (w := w)) e X he
  exact den_batchOp_shard (N := N) (.dense "%Wd" "%bd" Wd bd) _ _ hg r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel ResNet-34 forward graph, over the replica family.** T2's
    `resnet34FwdGraphB_full` with every BatchNorm a `bnSyncSiteLA` over all `R` replicas; block
    prefixes and collective tags are `ResNet34RenderB`'s. -/
def resnet34FwdGraphSync_full (R : Nat) (hR : 0 < R) (N : Nat) (epsStr : String) {nCls : Nat}
    (w : R34BWeights nCls) (e : Fin R → SHlo (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    Fin R → SHlo (N * nCls) :=
  r34HeadGraphSync N 7 7 w.Wd w.bd
    (r34IdGraphSync "s4b1" epsStr R hR N 7 7 w.e1
      (r34IdGraphSync "s4b0" epsStr R hR N 7 7 w.e0
        (r34DownGraphSync "d4" epsStr R hR N 7 7 w.d4
          (r34IdGraphSync "s3b4" epsStr R hR N 14 14 w.c4
            (r34IdGraphSync "s3b3" epsStr R hR N 14 14 w.c3
              (r34IdGraphSync "s3b2" epsStr R hR N 14 14 w.c2
                (r34IdGraphSync "s3b1" epsStr R hR N 14 14 w.c1
                  (r34IdGraphSync "s3b0" epsStr R hR N 14 14 w.c0
                    (r34DownGraphSync "d3" epsStr R hR N 14 14 w.d3
                      (r34IdGraphSync "s2b2" epsStr R hR N 28 28 w.b2
                        (r34IdGraphSync "s2b1" epsStr R hR N 28 28 w.b1
                          (r34IdGraphSync "s2b0" epsStr R hR N 28 28 w.b0
                            (r34DownGraphSync "d2" epsStr R hR N 28 28 w.d2
                              (r34IdGraphSync "s1b2" epsStr R hR N 56 56 w.a2
                                (r34IdGraphSync "s1b1" epsStr R hR N 56 56 w.a1
                                  (r34IdGraphSync "s1b0" epsStr R hR N 56 56 w.a0
                                    (r34StemGraphSync epsStr R hR N 56 56 w.sW w.sb w.sε w.sγ w.sβ
                                      e)))))))))))))))))

/-- ⭐⭐ **T2 at synchronised BatchNorm: replica `r`'s forward IS shard `r` of the global-batch
    forward.** Given that the replicas' inputs are the shards of one batch `X` of `R·N` examples,
    the sync-BN graph on replica `r` denotes `batchShard r` of `resnet34ForwardB_full (R * N) w X`
    — the committed batch-BN forward, at the global batch. One block lemma per stage, the shard
    hypothesis threaded from each into the next. -/
theorem resnet34FwdGraphSync_full_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N)
    (epsStr : String) {nCls : Nat} (w : R34BWeights nCls)
    (e : Fin R → SHlo (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (X : Vec ((R * N) * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (he : ∀ r, den (e r) = batchShard R N (3 * (2 * (2 * 56)) * (2 * (2 * 56))) X r) (r : Fin R) :
    den (resnet34FwdGraphSync_full R hR N epsStr w e r)
      = batchShard R N nCls (resnet34ForwardB_full (R * N) w X) r := by
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  have s0 := r34StemGraphSync_shard epsStr R hR N 56 56 hN h56 h56 w.sW w.sb w.sε w.sγ w.sβ e X he
  have s1 := r34IdGraphSync_shard "s1b0" epsStr R hR N 56 56 hN h56 h56 w.a0 _ _ s0
  have s2 := r34IdGraphSync_shard "s1b1" epsStr R hR N 56 56 hN h56 h56 w.a1 _ _ s1
  have s3 := r34IdGraphSync_shard "s1b2" epsStr R hR N 56 56 hN h56 h56 w.a2 _ _ s2
  have s4 := r34DownGraphSync_shard "d2" epsStr R hR N 28 28 hN h28 h28 w.d2 _ _ s3
  have s5 := r34IdGraphSync_shard "s2b0" epsStr R hR N 28 28 hN h28 h28 w.b0 _ _ s4
  have s6 := r34IdGraphSync_shard "s2b1" epsStr R hR N 28 28 hN h28 h28 w.b1 _ _ s5
  have s7 := r34IdGraphSync_shard "s2b2" epsStr R hR N 28 28 hN h28 h28 w.b2 _ _ s6
  have s8 := r34DownGraphSync_shard "d3" epsStr R hR N 14 14 hN h14 h14 w.d3 _ _ s7
  have s9 := r34IdGraphSync_shard "s3b0" epsStr R hR N 14 14 hN h14 h14 w.c0 _ _ s8
  have s10 := r34IdGraphSync_shard "s3b1" epsStr R hR N 14 14 hN h14 h14 w.c1 _ _ s9
  have s11 := r34IdGraphSync_shard "s3b2" epsStr R hR N 14 14 hN h14 h14 w.c2 _ _ s10
  have s12 := r34IdGraphSync_shard "s3b3" epsStr R hR N 14 14 hN h14 h14 w.c3 _ _ s11
  have s13 := r34IdGraphSync_shard "s3b4" epsStr R hR N 14 14 hN h14 h14 w.c4 _ _ s12
  have s14 := r34DownGraphSync_shard "d4" epsStr R hR N 7 7 hN h7 h7 w.d4 _ _ s13
  have s15 := r34IdGraphSync_shard "s4b0" epsStr R hR N 7 7 hN h7 h7 w.e0 _ _ s14
  have s16 := r34IdGraphSync_shard "s4b1" epsStr R hR N 7 7 hN h7 h7 w.e1 _ _ s15
  exact r34HeadGraphSync_shard N 7 7 w.Wd w.bd _ _ s16 r

end StableHLO

end Proofs
