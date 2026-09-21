import LeanMlir.Proofs.Nets.ResNet.ResNet50FullB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncB

/-! # ResNet-50's data-parallel forward at SYNCHRONISED BatchNorm — replica `r` IS shard `r`

`ResNet50FullB` (T2) says the typed batch-BN graph denotes `resnet50ForwardB_full N q w` on one
device. At `replicas > 1` the data-parallel render makes every BatchNorm site `bnFwdSite`'s sync-BN
composition — this replica's mean all-reduced, then Chan's `σ²_r + (μ_r − μ)²` all-reduced, packed,
then `bnSyncF` — so the statistics each replica normalises by are the GLOBAL batch's. This file is
T2's data-parallel twin: that forward graph, stated as a family over the `R` replicas, denotes on
replica `r` exactly `batchShard r` of the single-device forward at the global batch `R·N`.

    den (resnet50FwdGraphSync_full R hR N q epsStr w e r)
      = batchShard R N nCls (resnet50ForwardB_full (R * N) q w X) r

given that each replica's input is its shard of one global batch `X`. ⭐ **The spec does not
move**: the right-hand side is the committed `resnet50ForwardB_full`, at `N := R·N`. ⭐ The
resolution stays a binder: one statement covers `q = 7` (224 px) and `q = 5` (160 px), the ladder
written as T2 writes it, `2 * (…)` nests throughout.

## How it is proved

By induction on the chain, one block at a time, exactly as T2 is — with the shard hypothesis
`∀ r, den (e r) = batchShard R N _ X r` as the invariant carried from block to block. Every piece
that is not about the bottleneck is `ResNet34SyncB`'s, imported:

* every conv, relu, pool, GAP and dense node is a per-example lift, and sharding commutes with it
  (`den_batchOp_shard`, `den_relu_shard`);
* every BatchNorm site is `bnSyncSiteLA`, whose shard lemma `den_bnSyncSiteLA` is
  `DataParallelSync.den_bnSyncF_allReduce` (P1 on the graph) read at the network's index;
* the stem and the head are ResNet-34's sync graphs verbatim (`r34StemGraphSync`,
  `r34HeadGraphSync`) — R50's stem and head ARE those functions at other widths, and they emit the
  same names (`%sW`, `%sg`, `%sbt`, `%Wd`, `%bd`).

What is new is the three bottleneck forms at variable shapes (`N ic mid oc h w` all variables),
each with three sync sites (four with the projection). ⚠ The two projection forms add in the
RENDER's order, `addVB(body, projection)`, where `residualProj proj body` adds `proj + body`; that
costs one commutation, `den_addVB_shard_comm`, exactly the `add_comm` T2 carries.

## The index seam

The conv/relu chain runs at the left-assoc index `N·(c·h·w)`; `bnSyncF` and its statistics nodes at
`N·(c·(h·w))`. The sync site carries the relabelling as `castIdx` on the AST value (ResNet-34's
`bnSyncSiteLA`), which changes no emitted text.

## What is NOT claimed here

⚠ No stochastic depth: the graph is the drop-path-free forward, as T2's is. ⚠ The f32 nodes — the
bf16 conv twins are not this statement. ⚠ The DP artifacts run with no conv biases; the bias operand
is `biasName false "" c`, the render's own function, and the bias fields stay `∀`-quantified as in
T2. ⚠ The backward and the parameter collectives are the T3 half
(`ResNet50SyncTieB.r50_net_syncTiedB`). ⚠ That the `R` replicas' inputs ARE the shards of one batch
is the driver's, as in `DataParallelSync`. ⚠ The lowerer's `all_reduce` is trusted as every other
op's lowering is.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The residual fan-in in the render's order
-- ════════════════════════════════════════════════════════════════

/-- The residual fan-in on every replica, with the SECOND operand first on the right: the render
    emits `addVB(body, projection)` and `residualProj proj body` adds `proj + body`. -/
theorem den_addVB_shard_comm {R N n : Nat} (a b : Fin R → SHlo (N * n)) (A B : Vec ((R * N) * n))
    (ha : ∀ r, den (a r) = batchShard R N n A r) (hb : ∀ r, den (b r) = batchShard R N n B r)
    (r : Fin R) :
    den (.addVB (a r) (b r)) = batchShard R N n (fun j => B j + A j) r := by
  rw [den_addVB, ha, hb]
  funext i
  exact add_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § Per-block replica families + their shard lemmas
--   Names and tags are the render's: BN site `{p}g1` gathers `%arsum{p}g1mu` /
--   `%armean{p}g1var`, each over a `[c]` statistic; the projection's is `{p}gp`.
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck at sync-BN, over the replica family:
    `relu(addVB(bn₃(conv₃(relu(bn₂(conv₂(relu(bn₁(conv₁ e))))))), e))`, every BatchNorm a
    `bnSyncSiteLA`. T2's `r50IdGraphB` with the three sites swapped. -/
def r50IdGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc : Nat}
    (pw : R50IdW mid oc) (e : Fin R → SHlo (N * (oc * h * w))) : Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc] R hR
        pw.ε₃ pw.γ₃ pw.β₃
        (fun r => .batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid] R hR
              pw.ε₂ pw.γ₂ pw.β₂
              (fun r => .batchOp (N := N)
                (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
                    R hR pw.ε₁ pw.γ₁ pw.β₁
                    (fun r => .batchOp (N := N)
                      (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁)
                      (e r))
                    r)))
              r)))
        r)
      (e r))

theorem r50IdGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R50IdW mid oc)
    (e : Fin R → SHlo (N * (oc * h * w))) (X : Vec ((R * N) * (oc * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (oc * h * w) X r) (r : Fin R) :
    den (r50IdGraphSync p epsStr R hR N h w pw e r)
      = batchShard R N (oc * h * w) (r50IdB (R * N) h w pw X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  have hc1 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁) e X he
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
    R hR hm hM pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂) _ _ hr1
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid]
    R hR hm hM pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hc3 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃) _ _ hr2
  have hn3 := den_bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc]
    R hR hm hM pw.ε₃ pw.γ₃ pw.β₃ _ _ hc3
  have ha := den_addVB_shard _ e _ X hn3 he
  exact den_relu_shard _ _ ha r

/-- ⭐ Stride-1 projection bottleneck at sync-BN — stage 1 block 0: the identity bottleneck's body
    plus a 1×1 conv → sync-BN skip at unchanged resolution, added in the render's order
    `addVB(body, projection)`. Four sync sites. -/
def r50ProjGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc] R hR
        pw.ε₃ pw.γ₃ pw.β₃
        (fun r => .batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid] R hR
              pw.ε₂ pw.γ₂ pw.β₂
              (fun r => .batchOp (N := N)
                (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
                    R hR pw.ε₁ pw.γ₁ pw.β₁
                    (fun r => .batchOp (N := N)
                      (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁)
                      (e r))
                    r)))
              r)))
        r)
      (bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc] R hR
        pw.εp pw.γp pw.βp
        (fun r => .batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) (e r))
        r))

theorem r50ProjGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R50ProjW ic mid oc)
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (r50ProjGraphSync p epsStr R hR N h w pw e r)
      = batchShard R N (oc * h * w) (r50ProjB (R * N) h w pw X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  have hc1 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁) e X he
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
    R hR hm hM pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂) _ _ hr1
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid]
    R hR hm hM pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hc3 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃) _ _ hr2
  have hn3 := den_bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc]
    R hR hm hM pw.ε₃ pw.γ₃ pw.β₃ _ _ hc3
  have hcp := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e X he
  have hnp := den_bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc]
    R hR hm hM pw.εp pw.γp pw.βp _ _ hcp
  have ha := den_addVB_shard_comm _ _ _ _ hn3 hnp
  exact den_relu_shard _ _ ha r

/-- Strided projection bottleneck at sync-BN — stages 2/3/4 block 0. ⚠⚠ v1.5: the stride is on the
    **3×3** and the 1×1 skip, so conv₁ and its sync-BN (bn₁) run at the INPUT resolution
    `2h × 2w` — that site's statistics reduce over `N·(2h)·(2w)` per replica. Four sync sites,
    added in the render's order `addVB(body, projection)`. -/
def r50DownGraphSync (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc] R hR
        pw.ε₃ pw.γ₃ pw.β₃
        (fun r => .batchOp (N := N)
          (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid] R hR
              pw.ε₂ pw.γ₂ pw.β₂
              (fun r => .batchOp (N := N)
                (.convStrided (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * (2 * h) * (2 * w)))
                  (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
                    R hR pw.ε₁ pw.γ₁ pw.β₁
                    (fun r => .batchOp (N := N)
                      (.conv (h := 2 * h) (w := 2 * w) s!"%{p}W1" (biasName false "" mid)
                        pw.W₁ pw.b₁)
                      (e r))
                    r)))
              r)))
        r)
      (bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc] R hR
        pw.εp pw.γp pw.βp
        (fun r => .batchOp (N := N)
          (.convStrided (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) (e r))
        r))

theorem r50DownGraphSync_shard (p epsStr : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R50ProjW ic mid oc)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w))))
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (r50DownGraphSync p epsStr R hR N h w pw e r)
      = batchShard R N (oc * h * w) (r50DownB (R * N) h w pw X) r := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  have hm2 := nhw_ne_zero hN h2h h2w
  have hM2 := nhw_ne_zero (Nat.mul_pos hR hN) h2h h2w
  have hc1 := den_batchOp_shard (N := N)
    (.conv (h := 2 * h) (w := 2 * w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁) e X he
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
    R hR hm2 hM2 pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.convStrided (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂) _ _ hr1
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid]
    R hR hm hM pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hc3 := den_batchOp_shard (N := N)
    (.conv (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃) _ _ hr2
  have hn3 := den_bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc]
    R hR hm hM pw.ε₃ pw.γ₃ pw.β₃ _ _ hc3
  have hcp := den_batchOp_shard (N := N)
    (.convStrided (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e X he
  have hnp := den_bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc]
    R hR hm hM pw.εp pw.γp pw.βp _ _ hcp
  have ha := den_addVB_shard_comm _ _ _ _ hn3 hnp
  exact den_relu_shard _ _ ha r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel ResNet-50 forward graph, over the replica family.** T2's
    `resnet50FwdGraphB_full` with every BatchNorm a `bnSyncSiteLA` over all `R` replicas; block
    prefixes (`s1b0` … `s4b2`) and collective tags are the render's. The stem and head are
    ResNet-34's sync graphs, whose names R50 shares. -/
def resnet50FwdGraphSync_full (R : Nat) (hR : 0 < R) (N q : Nat) (epsStr : String) {nCls : Nat}
    (w : R50BWeights nCls)
    (e : Fin R → SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    Fin R → SHlo (N * nCls) :=
  r34HeadGraphSync N q q w.Wd w.bd
    (r50IdGraphSync "s4b2" epsStr R hR N q q w.s4b2
      (r50IdGraphSync "s4b1" epsStr R hR N q q w.s4b1
        (r50DownGraphSync "s4b0" epsStr R hR N q q w.s4b0
          (r50IdGraphSync "s3b5" epsStr R hR N (2 * q) (2 * q) w.s3b5
            (r50IdGraphSync "s3b4" epsStr R hR N (2 * q) (2 * q) w.s3b4
              (r50IdGraphSync "s3b3" epsStr R hR N (2 * q) (2 * q) w.s3b3
                (r50IdGraphSync "s3b2" epsStr R hR N (2 * q) (2 * q) w.s3b2
                  (r50IdGraphSync "s3b1" epsStr R hR N (2 * q) (2 * q) w.s3b1
                    (r50DownGraphSync "s3b0" epsStr R hR N (2 * q) (2 * q) w.s3b0
                      (r50IdGraphSync "s2b3" epsStr R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b3
                        (r50IdGraphSync "s2b2" epsStr R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b2
                          (r50IdGraphSync "s2b1" epsStr R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b1
                            (r50DownGraphSync "s2b0" epsStr R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b0
                              (r50IdGraphSync "s1b2" epsStr R hR N (2 * (2 * (2 * q)))
                                  (2 * (2 * (2 * q))) w.s1b2
                                (r50IdGraphSync "s1b1" epsStr R hR N (2 * (2 * (2 * q)))
                                    (2 * (2 * (2 * q))) w.s1b1
                                  (r50ProjGraphSync "s1b0" epsStr R hR N (2 * (2 * (2 * q)))
                                      (2 * (2 * (2 * q))) w.s1b0
                                    (r34StemGraphSync epsStr R hR N (2 * (2 * (2 * q)))
                                        (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ
                                      e)))))))))))))))))

/-- ⭐⭐ **T2 at synchronised BatchNorm: replica `r`'s forward IS shard `r` of the global-batch
    forward.** Given that the replicas' inputs are the shards of one batch `X` of `R·N` examples,
    the sync-BN graph on replica `r` denotes `batchShard r` of `resnet50ForwardB_full (R * N) q w X`
    — the committed batch-BN forward, at the global batch, at the same resolution binder `q`
    (`q = 7` the 224-px net, `q = 5` the 160-px one). One block lemma per block, the shard
    hypothesis threaded from each into the next. `0 < q` is what makes every BatchNorm's reduction
    width nonzero. -/
theorem resnet50FwdGraphSync_full_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (q : Nat)
    (hq : 0 < q) (epsStr : String) {nCls : Nat} (w : R50BWeights nCls)
    (e : Fin R → SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (X : Vec ((R * N) * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (he : ∀ r, den (e r)
      = batchShard R N (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))) X r)
    (r : Fin R) :
    den (resnet50FwdGraphSync_full R hR N q epsStr w e r)
      = batchShard R N nCls (resnet50ForwardB_full (R * N) q w X) r := by
  have h1 : 0 < q := hq
  have h2 : 0 < 2 * q := by omega
  have h4 : 0 < 2 * (2 * q) := by omega
  have h8 : 0 < 2 * (2 * (2 * q)) := by omega
  have s0 := r34StemGraphSync_shard epsStr R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8
    w.sW w.sb w.sε w.sγ w.sβ e X he
  have s1 := r50ProjGraphSync_shard "s1b0" epsStr R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
    hN h8 h8 w.s1b0 _ _ s0
  have s2 := r50IdGraphSync_shard "s1b1" epsStr R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
    hN h8 h8 w.s1b1 _ _ s1
  have s3 := r50IdGraphSync_shard "s1b2" epsStr R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
    hN h8 h8 w.s1b2 _ _ s2
  have s4 := r50DownGraphSync_shard "s2b0" epsStr R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b0 _ _ s3
  have s5 := r50IdGraphSync_shard "s2b1" epsStr R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b1 _ _ s4
  have s6 := r50IdGraphSync_shard "s2b2" epsStr R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b2 _ _ s5
  have s7 := r50IdGraphSync_shard "s2b3" epsStr R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b3 _ _ s6
  have s8 := r50DownGraphSync_shard "s3b0" epsStr R hR N (2 * q) (2 * q) hN h2 h2 w.s3b0 _ _ s7
  have s9 := r50IdGraphSync_shard "s3b1" epsStr R hR N (2 * q) (2 * q) hN h2 h2 w.s3b1 _ _ s8
  have s10 := r50IdGraphSync_shard "s3b2" epsStr R hR N (2 * q) (2 * q) hN h2 h2 w.s3b2 _ _ s9
  have s11 := r50IdGraphSync_shard "s3b3" epsStr R hR N (2 * q) (2 * q) hN h2 h2 w.s3b3 _ _ s10
  have s12 := r50IdGraphSync_shard "s3b4" epsStr R hR N (2 * q) (2 * q) hN h2 h2 w.s3b4 _ _ s11
  have s13 := r50IdGraphSync_shard "s3b5" epsStr R hR N (2 * q) (2 * q) hN h2 h2 w.s3b5 _ _ s12
  have s14 := r50DownGraphSync_shard "s4b0" epsStr R hR N q q hN h1 h1 w.s4b0 _ _ s13
  have s15 := r50IdGraphSync_shard "s4b1" epsStr R hR N q q hN h1 h1 w.s4b1 _ _ s14
  have s16 := r50IdGraphSync_shard "s4b2" epsStr R hR N q q hN h1 h1 w.s4b2 _ _ s15
  exact r34HeadGraphSync_shard N q q w.Wd w.bd _ _ s16 r

-- ⭐ `q = 7` IS the 224-px net and `q = 5` the 160-px one: the capstone at each, the input bound at
-- the literal shape. Instantiation only — the statement is proved once, at the variable `q`.
example (R N : Nat) (hR : 0 < R) (hN : 0 < N) (epsStr : String) {nCls : Nat}
    (w : R50BWeights nCls) (e : Fin R → SHlo (N * (3 * 224 * 224)))
    (X : Vec ((R * N) * (3 * 224 * 224)))
    (he : ∀ r, den (e r) = batchShard R N (3 * 224 * 224) X r) (r : Fin R) :
    den (resnet50FwdGraphSync_full R hR N 7 epsStr w e r)
      = batchShard R N nCls (resnet50ForwardB_full (R * N) 7 w X) r :=
  resnet50FwdGraphSync_full_shard R hR N hN 7 (by norm_num) epsStr w e X he r

example (R N : Nat) (hR : 0 < R) (hN : 0 < N) (epsStr : String) {nCls : Nat}
    (w : R50BWeights nCls) (e : Fin R → SHlo (N * (3 * 160 * 160)))
    (X : Vec ((R * N) * (3 * 160 * 160)))
    (he : ∀ r, den (e r) = batchShard R N (3 * 160 * 160) X r) (r : Fin R) :
    den (resnet50FwdGraphSync_full R hR N 5 epsStr w e r)
      = batchShard R N nCls (resnet50ForwardB_full (R * N) 5 w X) r :=
  resnet50FwdGraphSync_full_shard R hR N hN 5 (by norm_num) epsStr w e X he r

end StableHLO

end Proofs
