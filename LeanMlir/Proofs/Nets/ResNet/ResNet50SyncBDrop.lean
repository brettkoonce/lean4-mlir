module

public import LeanMlir.Proofs.Nets.ResNet.ResNet50SyncB
public import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBDrop
import LeanMlir.Proofs.Foundation.Bf16Erasure
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # ResNet-50's data-parallel forward with stochastic depth — replica `r` IS shard `r`

`ResNet50SyncB` says the sync-BN forward graph on replica `r` denotes shard `r` of the
single-device forward at the global batch, drop-free. This file is that statement with the
sixteen stochastic-depth sites of `ResNet50FullBDrop`:

    den (resnet50FwdGraphSyncFullDrop R hR N q epsStr w bf16 sd e r)
      = batchShard R N nCls (resnet50ForwardBFullDrop (R * N) q w sd X) r

`sd : Option (Fin 16 → Vec (R * N))` is the GLOBAL mask, one scalar per example of the global
batch per site. Replica `r`'s graph drops block `k`'s branch at ITS shard of site `k`
(`exampleShard`), the DP render's per-replica `%dp<k>` input. `none` emits no node and the
statement is `resnet50FwdGraphSyncFull_shard`'s; `some` is the `drop` DP artifacts' forward
(`resnet50in_emalambaccdp4x128wxclipdropbce` and its twins).

It is the forward half `ResNet50SyncTieB.r50_net_syncTiedB` reads its saved activations from:
that tie takes the replicas' activations as shards of the single-device forward at the global
mask, and this file says the replicas' forward graphs compute those.

## How it is proved

`ResNet50SyncB`'s induction with one more step per block. The site sits between bn₃'s sync site
and the skip add, and a drop site on a replica family is the shard of the global site
(`den_dropPathOptG_shard`, which is `dropPathOpt_shard` on the graph). Everything else is the
drop-free proof.

## What is NOT claimed here

That the `R` replicas' inputs and masks ARE the shards of one batch and one mask is the driver's.
How the masks are drawn is not modelled. The stem and head carry no site and are ResNet-34's
sync graphs, as in `ResNet50SyncB`.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-block replica families with a site + their shard lemmas
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck at sync-BN with a stochastic-depth site, over the replica family:
    `r50IdGraphSync` with replica `r`'s branch through `dropPathOptG` at its shard of the mask.
    Drop-free shape:
    `relu(addVB(bn₃(conv₃(relu(bn₂(conv₂(relu(bn₁(conv₁ e))))))), e))`, every BatchNorm a
    `bnSyncSiteLA`. The single-device `r50IdGraphB` with the three sites swapped. -/
noncomputable def r50IdGraphSyncD (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc : Nat}
    (pw : R50IdW mid oc) (bf16 : Bool) (s : Option (Vec (R * N))) (e : Fin R → SHlo (N * (oc * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (dropPathOptG dpN (s.map fun M => exampleShard R N M r)
      (bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc] R hR
        pw.ε₃ pw.γ₃ pw.β₃
        (fun r => .batchOp (N := N)
          (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid] R hR
              pw.ε₂ pw.γ₂ pw.β₂
              (fun r => .batchOp (N := N)
                (.convAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
                    R hR pw.ε₁ pw.γ₁ pw.β₁
                    (fun r => .batchOp (N := N)
                      (.convAt bf16 id (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁)
                      (e r))
                    r)))
              r)))
        r))
      (e r))

theorem r50IdGraphSyncD_shard (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R50IdW mid oc) (bf16 : Bool) (s : Option (Vec (R * N)))
    (e : Fin R → SHlo (N * (oc * h * w))) (X : Vec ((R * N) * (oc * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (oc * h * w) X r) (r : Fin R) :
    den (r50IdGraphSyncD p epsStr dpN R hR N h w pw bf16 s e r)
      = batchShard R N (oc * h * w) (r50IdBD (R * N) h w pw s X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc1 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hc1
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
    R hR hm pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂) _ _ hr1
  simp only [Bf16Fold.denOp_convAt_id] at hc2
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid]
    R hR hm pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hc3 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃) _ _ hr2
  simp only [Bf16Fold.denOp_convAt_id] at hc3
  have hn3 := den_bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc]
    R hR hm pw.ε₃ pw.γ₃ pw.β₃ _ _ hc3
  have hd := den_dropPathOptG_shard dpN s _ _ hn3
  have ha := den_addVB_shard _ e _ X hd he
  exact den_relu_shard _ _ ha r

/-- Stride-1 projection bottleneck at sync-BN with a stochastic-depth site on the branch
    (the projection is not dropped) — stage 1 block 0: the identity bottleneck's body
    plus a 1×1 conv → sync-BN skip at unchanged resolution, added in the render's order
    `addVB(body, projection)`. Four sync sites. -/
noncomputable def r50ProjGraphSyncD (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (bf16 : Bool) (s : Option (Vec (R * N))) (e : Fin R → SHlo (N * (ic * h * w))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (dropPathOptG dpN (s.map fun M => exampleShard R N M r)
      (bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc] R hR
        pw.ε₃ pw.γ₃ pw.β₃
        (fun r => .batchOp (N := N)
          (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid] R hR
              pw.ε₂ pw.γ₂ pw.β₂
              (fun r => .batchOp (N := N)
                (.convAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
                    R hR pw.ε₁ pw.γ₁ pw.β₁
                    (fun r => .batchOp (N := N)
                      (.convAt bf16 id (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁)
                      (e r))
                    r)))
              r)))
        r))
      (bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc] R hR
        pw.εp pw.γp pw.βp
        (fun r => .batchOp (N := N)
          (.convAt bf16 id (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) (e r))
        r))

theorem r50ProjGraphSyncD_shard (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R50ProjW ic mid oc) (bf16 : Bool) (s : Option (Vec (R * N)))
    (e : Fin R → SHlo (N * (ic * h * w))) (X : Vec ((R * N) * (ic * h * w)))
    (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r) (r : Fin R) :
    den (r50ProjGraphSyncD p epsStr dpN R hR N h w pw bf16 s e r)
      = batchShard R N (oc * h * w) (r50ProjBD (R * N) h w pw s X) r := by
  have hm := nhw_ne_zero hN hh hw
  have hc1 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hc1
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
    R hR hm pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂) _ _ hr1
  simp only [Bf16Fold.denOp_convAt_id] at hc2
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid]
    R hR hm pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hc3 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃) _ _ hr2
  simp only [Bf16Fold.denOp_convAt_id] at hc3
  have hn3 := den_bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc]
    R hR hm pw.ε₃ pw.γ₃ pw.β₃ _ _ hc3
  have hcp := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hcp
  have hnp := den_bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc]
    R hR hm pw.εp pw.γp pw.βp _ _ hcp
  have hd := den_dropPathOptG_shard dpN s _ _ hn3
  have ha := den_addVB_shard_comm _ _ _ _ hd hnp
  exact den_relu_shard _ _ ha r

/-- Strided projection bottleneck at sync-BN with a stochastic-depth site on the branch (the
    projection is not dropped) — stages 2/3/4 block 0. v1.5: the stride is on the
    **3×3** and the 1×1 skip, so conv₁ and its sync-BN (bn₁) run at the INPUT resolution
    `2h × 2w` — that site's statistics reduce over `N·(2h)·(2w)` per replica. Four sync sites,
    added in the render's order `addVB(body, projection)`. -/
noncomputable def r50DownGraphSyncD (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pw : R50ProjW ic mid oc) (bf16 : Bool) (s : Option (Vec (R * N))) (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) :
    Fin R → SHlo (N * (oc * h * w)) :=
  fun r => .batchOp (N := N) (.relu (n := oc * h * w))
    (.addVB
      (dropPathOptG dpN (s.map fun M => exampleShard R N M r)
      (bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc] R hR
        pw.ε₃ pw.γ₃ pw.β₃
        (fun r => .batchOp (N := N)
          (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃)
          (.batchOp (N := N) (.relu (n := mid * h * w))
            (bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid] R hR
              pw.ε₂ pw.γ₂ pw.β₂
              (fun r => .batchOp (N := N)
                (.convStridedAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂)
                (.batchOp (N := N) (.relu (n := mid * (2 * h) * (2 * w)))
                  (bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
                    R hR pw.ε₁ pw.γ₁ pw.β₁
                    (fun r => .batchOp (N := N)
                      (.convAt bf16 id (h := 2 * h) (w := 2 * w) s!"%{p}W1" (biasName false "" mid)
                        pw.W₁ pw.b₁)
                      (e r))
                    r)))
              r)))
        r))
      (bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc] R hR
        pw.εp pw.γp pw.βp
        (fun r => .batchOp (N := N)
          (.convStridedAt bf16 id (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) (e r))
        r))

theorem r50DownGraphSyncD_shard (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pw : R50ProjW ic mid oc) (bf16 : Bool) (s : Option (Vec (R * N)))
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w))))
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (r50DownGraphSyncD p epsStr dpN R hR N h w pw bf16 s e r)
      = batchShard R N (oc * h * w) (r50DownBD (R * N) h w pw s X) r := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm := nhw_ne_zero hN hh hw
  have hm2 := nhw_ne_zero hN h2h h2w
  have hc1 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := 2 * h) (w := 2 * w) s!"%{p}W1" (biasName false "" mid) pw.W₁ pw.b₁) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hc1
  have hn1 := den_bnSyncSiteLA s!"%{p}g1" s!"%{p}bt1" epsStr s!"{p}g1mu" s!"{p}g1var" [mid] [mid]
    R hR hm2 pw.ε₁ pw.γ₁ pw.β₁ _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hc2 := den_batchOp_shard (N := N)
    (.convStridedAt bf16 id (h := h) (w := w) s!"%{p}W2" (biasName false "" mid) pw.W₂ pw.b₂) _ _ hr1
  simp only [Bf16Fold.denOp_convStridedAt_id] at hc2
  have hn2 := den_bnSyncSiteLA s!"%{p}g2" s!"%{p}bt2" epsStr s!"{p}g2mu" s!"{p}g2var" [mid] [mid]
    R hR hm pw.ε₂ pw.γ₂ pw.β₂ _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hc3 := den_batchOp_shard (N := N)
    (.convAt bf16 id (h := h) (w := w) s!"%{p}W3" (biasName false "" oc) pw.W₃ pw.b₃) _ _ hr2
  simp only [Bf16Fold.denOp_convAt_id] at hc3
  have hn3 := den_bnSyncSiteLA s!"%{p}g3" s!"%{p}bt3" epsStr s!"{p}g3mu" s!"{p}g3var" [oc] [oc]
    R hR hm pw.ε₃ pw.γ₃ pw.β₃ _ _ hc3
  have hcp := den_batchOp_shard (N := N)
    (.convStridedAt bf16 id (h := h) (w := w) s!"%{p}Wp" (biasName false "" oc) pw.Wp pw.bp) e X he
  simp only [Bf16Fold.denOp_convStridedAt_id] at hcp
  have hnp := den_bnSyncSiteLA s!"%{p}gp" s!"%{p}btp" epsStr s!"{p}gpmu" s!"{p}gpvar" [oc] [oc]
    R hR hm pw.εp pw.γp pw.βp _ _ hcp
  have hd := den_dropPathOptG_shard dpN s _ _ hn3
  have ha := den_addVB_shard_comm _ _ _ _ hd hnp
  exact den_relu_shard _ _ ha r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel ResNet-50 forward graph with stochastic depth, over the replica
    family.** `resnet50FwdGraphSyncFull` with block `k`'s branch through `dropPathOptG (dpName k)`
    at replica `r`'s shard of the global site `r50Sd sd k`. -/
noncomputable def resnet50FwdGraphSyncFullDrop (R : Nat) (hR : 0 < R) (N q : Nat) (epsStr : String) {nCls : Nat}
    (w : R50BWeights nCls) (bf16 : Bool) (sd : Option (Fin 16 → Vec (R * N)))
    (e : Fin R → SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    Fin R → SHlo (N * nCls) :=
  r34HeadGraphSync N q q w.Wd w.bd
    (r50IdGraphSyncD "s4b2" epsStr (dpName 15) R hR N q q w.s4b2 bf16
        (r50Sd sd 15)
      (r50IdGraphSyncD "s4b1" epsStr (dpName 14) R hR N q q w.s4b1 bf16
          (r50Sd sd 14)
        (r50DownGraphSyncD "s4b0" epsStr (dpName 13) R hR N q q w.s4b0 bf16
            (r50Sd sd 13)
          (r50IdGraphSyncD "s3b5" epsStr (dpName 12) R hR N (2 * q) (2 * q) w.s3b5 bf16
              (r50Sd sd 12)
            (r50IdGraphSyncD "s3b4" epsStr (dpName 11) R hR N (2 * q) (2 * q) w.s3b4 bf16
                (r50Sd sd 11)
              (r50IdGraphSyncD "s3b3" epsStr (dpName 10) R hR N (2 * q) (2 * q) w.s3b3 bf16
                  (r50Sd sd 10)
                (r50IdGraphSyncD "s3b2" epsStr (dpName 9) R hR N (2 * q) (2 * q) w.s3b2 bf16
                    (r50Sd sd 9)
                  (r50IdGraphSyncD "s3b1" epsStr (dpName 8) R hR N (2 * q) (2 * q) w.s3b1 bf16
                      (r50Sd sd 8)
                    (r50DownGraphSyncD "s3b0" epsStr (dpName 7) R hR N (2 * q) (2 * q) w.s3b0 bf16
                        (r50Sd sd 7)
                      (r50IdGraphSyncD "s2b3" epsStr (dpName 6) R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 bf16
                          (r50Sd sd 6)
                        (r50IdGraphSyncD "s2b2" epsStr (dpName 5) R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 bf16
                            (r50Sd sd 5)
                          (r50IdGraphSyncD "s2b1" epsStr (dpName 4) R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 bf16
                              (r50Sd sd 4)
                            (r50DownGraphSyncD "s2b0" epsStr (dpName 3) R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 bf16
                                (r50Sd sd 3)
                              (r50IdGraphSyncD "s1b2" epsStr (dpName 2) R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 bf16
                                  (r50Sd sd 2)
                                (r50IdGraphSyncD "s1b1" epsStr (dpName 1) R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 bf16
                                    (r50Sd sd 1)
                                  (r50ProjGraphSyncD "s1b0" epsStr (dpName 0) R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 bf16
                                      (r50Sd sd 0)
                                    (r34StemGraphSync epsStr R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ bf16
                                      e)))))))))))))))))

/-- **Forward-graph faithfulness at synchronised BatchNorm with stochastic depth: replica `r`'s
    forward IS shard `r` of the global-batch forward at the global mask.** Given that the
    replicas' inputs are the shards of one batch `X` of `R·N` examples, the graph on replica `r`
    denotes `batchShard r` of `resnet50ForwardBFullDrop (R * N) q w sd X`. `0 < q` makes every
    BatchNorm's reduction width nonzero. -/
theorem resnet50FwdGraphSyncFullDrop_shard (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (q : Nat)
    (hq : 0 < q) (epsStr : String) {nCls : Nat} (w : R50BWeights nCls) (bf16 : Bool)
    (sd : Option (Fin 16 → Vec (R * N)))
    (e : Fin R → SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (X : Vec ((R * N) * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (he : ∀ r, den (e r)
      = batchShard R N (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))) X r)
    (r : Fin R) :
    den (resnet50FwdGraphSyncFullDrop R hR N q epsStr w bf16 sd e r)
      = batchShard R N nCls (resnet50ForwardBFullDrop (R * N) q w sd X) r := by
  have h1 : 0 < q := hq
  have h2 : 0 < 2 * q := by omega
  have h4 : 0 < 2 * (2 * q) := by omega
  have h8 : 0 < 2 * (2 * (2 * q)) := by omega
  have s0 := r34StemGraphSync_shard epsStr R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8
    w.sW w.sb w.sε w.sγ w.sβ bf16 e X he
  have s1 := r50ProjGraphSyncD_shard "s1b0" epsStr (dpName 0) R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
    hN h8 h8 w.s1b0 bf16 (r50Sd sd 0) _ _ s0
  have s2 := r50IdGraphSyncD_shard "s1b1" epsStr (dpName 1) R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
    hN h8 h8 w.s1b1 bf16 (r50Sd sd 1) _ _ s1
  have s3 := r50IdGraphSyncD_shard "s1b2" epsStr (dpName 2) R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
    hN h8 h8 w.s1b2 bf16 (r50Sd sd 2) _ _ s2
  have s4 := r50DownGraphSyncD_shard "s2b0" epsStr (dpName 3) R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b0 bf16 (r50Sd sd 3) _ _ s3
  have s5 := r50IdGraphSyncD_shard "s2b1" epsStr (dpName 4) R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b1 bf16 (r50Sd sd 4) _ _ s4
  have s6 := r50IdGraphSyncD_shard "s2b2" epsStr (dpName 5) R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b2 bf16 (r50Sd sd 5) _ _ s5
  have s7 := r50IdGraphSyncD_shard "s2b3" epsStr (dpName 6) R hR N (2 * (2 * q)) (2 * (2 * q))
    hN h4 h4 w.s2b3 bf16 (r50Sd sd 6) _ _ s6
  have s8 := r50DownGraphSyncD_shard "s3b0" epsStr (dpName 7) R hR N (2 * q) (2 * q)
    hN h2 h2 w.s3b0 bf16 (r50Sd sd 7) _ _ s7
  have s9 := r50IdGraphSyncD_shard "s3b1" epsStr (dpName 8) R hR N (2 * q) (2 * q)
    hN h2 h2 w.s3b1 bf16 (r50Sd sd 8) _ _ s8
  have s10 := r50IdGraphSyncD_shard "s3b2" epsStr (dpName 9) R hR N (2 * q) (2 * q)
    hN h2 h2 w.s3b2 bf16 (r50Sd sd 9) _ _ s9
  have s11 := r50IdGraphSyncD_shard "s3b3" epsStr (dpName 10) R hR N (2 * q) (2 * q)
    hN h2 h2 w.s3b3 bf16 (r50Sd sd 10) _ _ s10
  have s12 := r50IdGraphSyncD_shard "s3b4" epsStr (dpName 11) R hR N (2 * q) (2 * q)
    hN h2 h2 w.s3b4 bf16 (r50Sd sd 11) _ _ s11
  have s13 := r50IdGraphSyncD_shard "s3b5" epsStr (dpName 12) R hR N (2 * q) (2 * q)
    hN h2 h2 w.s3b5 bf16 (r50Sd sd 12) _ _ s12
  have s14 := r50DownGraphSyncD_shard "s4b0" epsStr (dpName 13) R hR N q q
    hN h1 h1 w.s4b0 bf16 (r50Sd sd 13) _ _ s13
  have s15 := r50IdGraphSyncD_shard "s4b1" epsStr (dpName 14) R hR N q q
    hN h1 h1 w.s4b1 bf16 (r50Sd sd 14) _ _ s14
  have s16 := r50IdGraphSyncD_shard "s4b2" epsStr (dpName 15) R hR N q q
    hN h1 h1 w.s4b2 bf16 (r50Sd sd 15) _ _ s15
  rw [resnet50ForwardBFullDrop, r50PreD16_apply, r50PreD15_apply, r50PreD14_apply, r50PreD13_apply, r50PreD12_apply, r50PreD11_apply, r50PreD10_apply, r50PreD9_apply, r50PreD8_apply, r50PreD7_apply, r50PreD6_apply, r50PreD5_apply, r50PreD4_apply, r50PreD3_apply, r50PreD2_apply, r50PreD1_apply, r50Pre0_apply]
  exact r34HeadGraphSync_shard N q q w.Wd w.bd _ _ s16 r

/-- With no site a block's masked sync graph is `ResNet50SyncB`'s. -/
theorem r50IdGraphSyncD_none (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {mid oc : Nat} (pw : R50IdW mid oc) (bf16 : Bool) (e : Fin R → SHlo (N * (oc * h * w))) :
    r50IdGraphSyncD p epsStr dpN R hR N h w pw bf16 none e
      = r50IdGraphSync p epsStr R hR N h w pw bf16 e := rfl

theorem r50ProjGraphSyncD_none (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (pw : R50ProjW ic mid oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * h * w))) :
    r50ProjGraphSyncD p epsStr dpN R hR N h w pw bf16 none e
      = r50ProjGraphSync p epsStr R hR N h w pw bf16 e := rfl

theorem r50DownGraphSyncD_none (p epsStr dpN : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {ic mid oc : Nat} (pw : R50ProjW ic mid oc) (bf16 : Bool)
    (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w)))) :
    r50DownGraphSyncD p epsStr dpN R hR N h w pw bf16 none e
      = r50DownGraphSync p epsStr R hR N h w pw bf16 e := rfl

/-- With no sites the masked sync graph is `ResNet50SyncB`'s. -/
theorem resnet50FwdGraphSyncFullDrop_none (R : Nat) (hR : 0 < R) (N q : Nat) (epsStr : String)
    {nCls : Nat} (w : R50BWeights nCls) (bf16 : Bool)
    (e : Fin R → SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    resnet50FwdGraphSyncFullDrop R hR N q epsStr w bf16 none e
      = resnet50FwdGraphSyncFull R hR N q epsStr w bf16 e := by
  simp only [resnet50FwdGraphSyncFullDrop, resnet50FwdGraphSyncFull, r50Sd_none,
    r50IdGraphSyncD_none, r50ProjGraphSyncD_none, r50DownGraphSyncD_none]

end StableHLO

end Proofs
