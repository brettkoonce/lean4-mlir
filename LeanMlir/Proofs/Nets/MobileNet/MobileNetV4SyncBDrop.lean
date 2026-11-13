module

public import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncB
public import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBDrop
public import LeanMlir.Proofs.Foundation.DropSites
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # MobileNetV4's data-parallel forward with both masks — replica `r` IS shard `r`

`MobileNetV4SyncB` says the sync-BN forward graph on replica `r` denotes shard `r` of the
single-device forward at the global batch, with no masks. This file is that statement for the
masked renders (`mnv4in_accdp8x128wxdropdowd01bf16` and its twins), at stochastic depth on the
eighteen skip rows and classifier dropout before the dense:

    den (mnv4FwdGraphSyncFullDrop R hR N f epsStr mName w bf16 SD M e r)
      = batchShard R N nCls (mobilenetv4ForwardBFullDrop (R * N) f w SD M X) r

`SD : Fin 18 → Vec (R * N)` and `M : Vec ((R * N) * 1280)` are the GLOBAL masks. Replica `r`'s
graph drops skip row `k`'s branch at its `exampleShard` of `SD k` (the DP render's per-replica
`%dp<k>`) and its classifier input at its `batchShard` of `M` (its `%do`). The right-hand side is
`MobileNetV4FullBDrop`'s forward at the global batch and the global masks, the forward
`MobileNetV4SyncTieB.mnv4_net_syncTiedB` reads its saved activations from at `some` masks.

## How it is proved

`MobileNetV4SyncB`'s induction, with a drop site on each skip row's branch
(`den_dropPathOptG_shard`) and one on the classifier's input (`den_dropoutOptG_shard`). The row
bodies, the stem and the fused stage are that file's shard lemmas unchanged.

## What is NOT claimed here

That the replicas' inputs and masks ARE the shards of one batch and one pair of masks is the
driver's. How the masks are drawn is not modelled. The unmasked statement is
`mnv4FwdGraphSyncFull_shard`; the renders with only one of the two masks are this statement at
the all-ones value of the other (`mobilenetv4ForwardBFullDrop_sdOnes`).
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § A skip row with its drop site, and the head with classifier dropout, over the replicas
-- ════════════════════════════════════════════════════════════════

/-- **One skip row at sync-BN with its drop site**: `mnv4SkipGraphSync` with a `dropPathB` on the
    body's output, replica `r` at its shard of the global per-example scale `S`. -/
noncomputable def mnv4SkipDropGraphSync {R N n : Nat} (mN : String) (S : Vec (R * N))
    (body : (Fin R → SHlo (N * n)) → Fin R → SHlo (N * n))
    (e : Fin R → SHlo (N * n)) : Fin R → SHlo (N * n) :=
  fun r => .addVB (.dropPathB mN (exampleShard R N S r) (body e r)) (e r)

/-- A dropped skip row at sync-BN is shard `r` of `mnv4SkipDrop` of whatever its body shards to,
    at the global scale. -/
theorem mnv4SkipDropGraphSync_shard {R N n : Nat} (mN : String) (S : Vec (R * N))
    (body : (Fin R → SHlo (N * n)) → Fin R → SHlo (N * n)) (f : Vec ((R * N) * n) → Vec ((R * N) * n))
    (hb : ∀ (e' : Fin R → SHlo (N * n)) (X' : Vec ((R * N) * n)),
      (∀ r, den (e' r) = batchShard R N n X' r) → ∀ r, den (body e' r) = batchShard R N n (f X') r)
    (e : Fin R → SHlo (N * n)) (X : Vec ((R * N) * n))
    (he : ∀ r, den (e r) = batchShard R N n X r) (r : Fin R) :
    den (mnv4SkipDropGraphSync mN S body e r) = batchShard R N n (mnv4SkipDrop f S X) r :=
  den_addVB_shard (fun r => .dropPathB mN (exampleShard R N S r) (body e r)) e
    (dropPath (R * N) n S (f X)) X
    (den_dropPathOptG_shard mN (some S) (body e) (f X) (hb e X he)) he r

/-- Head at sync-BN with classifier dropout, over the replica family: `mnv4HeadGraphSync` with a
    `dropoutB` on the dense's input, replica `r` at its shard of the global mask `M`. -/
noncomputable def mnv4HeadGraphSyncDo (epsStr mName : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool) (M : Vec ((R * N) * oc))
    (e : Fin R → SHlo (N * (c * h * w))) : Fin R → SHlo (N * nCls) :=
  fun r => .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (.dropoutB mName (batchShard R N oc M r)
      (castIdx (mnv4_pool11 N oc).symm
        (.batchOp (N := N) (.relu (n := oc * 1 * 1))
          (bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc] R hR ε2 γ2 β2
            (fun r => .batchOp (N := N) (.convAt bf16 id (h := 1) (w := 1) "%hW" s!"%zb{oc}" W2 b2)
              (castIdx (mnv4_pool11 N mid)
                (.batchOp (N := N) (.gap (c := mid) (h := h) (w := w))
                  (.batchOp (N := N) (.relu (n := mid * h * w))
                    (bnSyncSiteLA "%h1g" "%h1bt" epsStr "h1gmu" "h1gvar" [mid] [mid] R hR ε1 γ1 β1
                      (fun r => .batchOp (N := N) (.convAt bf16 id (h := h) (w := w) "%h1W" s!"%zb{mid}" W1 b1)
                        (e r))
                      r)))))
            r))))

theorem mnv4HeadGraphSyncDo_shard (epsStr mName : String) (R : Nat) (hR : 0 < R) (N h w : Nat)
    {c mid oc nCls : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool) (M : Vec ((R * N) * oc))
    (e : Fin R → SHlo (N * (c * h * w)))
    (X : Vec ((R * N) * (c * h * w))) (he : ∀ r, den (e r) = batchShard R N (c * h * w) X r)
    (r : Fin R) :
    den (mnv4HeadGraphSyncDo epsStr mName R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd bf16 M e r)
      = batchShard R N nCls
          (batchMap (R * N) (dense Wd bd)
            (dropout M (mnv4HeadFeatB (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 X))) r := by
  have hm := nhw_ne_zero hN hh hw
  have hm1 := nhw_ne_zero hN Nat.one_pos Nat.one_pos
  have hc1 := den_batchOp_shard (N := N) (.convAt bf16 id (h := h) (w := w) "%h1W" s!"%zb{mid}" W1 b1) e X he
  simp only [Bf16Fold.denOp_convAt_id] at hc1
  have hn1 := den_bnSyncSiteLA "%h1g" "%h1bt" epsStr "h1gmu" "h1gvar" [mid] [mid]
    R hR hm ε1 γ1 β1 _ _ hc1
  have hr1 := den_relu_shard _ _ hn1
  have hg := den_batchOp_shard (N := N) (.gap (c := mid) (h := h) (w := w)) _ _ hr1
  have hp := den_castIdx_shard (by rw [Nat.mul_one, Nat.mul_one] : mid = mid * 1 * 1)
    (mnv4_pool11 N mid) _ _ hg
  have hc2 := den_batchOp_shard (N := N) (.convAt bf16 id (h := 1) (w := 1) "%hW" s!"%zb{oc}" W2 b2) _ _ hp
  simp only [Bf16Fold.denOp_convAt_id] at hc2
  have hn2 := den_bnSyncSiteLA "%hg" "%hbt" epsStr "hgmu" "hgvar" [oc] [oc]
    R hR hm1 ε2 γ2 β2 _ _ hc2
  have hr2 := den_relu_shard _ _ hn2
  have hf := den_castIdx_shard (by rw [Nat.mul_one, Nat.mul_one] : oc * 1 * 1 = oc)
    (mnv4_pool11 N oc).symm _ _ hr2
  have hdo := den_dropoutOptG_shard mName (some M) _ _ hf
  exact den_batchOp_shard (N := N) (.dense "%Wd" "%bd" Wd bd) _ _ hdo r

-- ════════════════════════════════════════════════════════════════
-- § The five resolution groups with their drop sites
--   Site `k` is the `k`-th skip row (rows 2, 4–10, 12–21), `MobileNetV4FullBDrop`'s numbering.
-- ════════════════════════════════════════════════════════════════

/-- Trunk group **Res28** at sync-BN with its drop sites — rows 1–2. -/
noncomputable def mnv4Res28DropGraphSync (R : Nat) (hR : 0 < R) (N f : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (48 * (2 * (2 * (2 * f))) * (2 * (2 * (2 * f)))))) :
    Fin R → SHlo (N * (80 * (2 * (2 * f)) * (2 * (2 * f)))) :=
  mnv4SkipDropGraphSync (dpName 0) (SD 0)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row2 f) w.b2 bf16)
    (mnv4StridedGraphSync epsStr R hR N (mnv4Row1 f) w.b1 bf16 e)

theorem mnv4Res28DropGraphSync_shard (R : Nat) (hR : 0 < R) (N f : Nat) (hN : 0 < N) (hf : 0 < f) (epsStr : String)
    {nCls : Nat} (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (48 * (2 * (2 * (2 * f))) * (2 * (2 * (2 * f))))))
    (X : Vec ((R * N) * (48 * (2 * (2 * (2 * f))) * (2 * (2 * (2 * f)))))) (he : ∀ r, den (e r) = batchShard R N (48 * (2 * (2 * (2 * f))) * (2 * (2 * (2 * f)))) X r)
    (r : Fin R) :
    den (mnv4Res28DropGraphSync R hR N f epsStr w bf16 SD e r)
      = batchShard R N (80 * (2 * (2 * f)) * (2 * (2 * f))) (mnv4Res28Drop (R * N) f w SD X) r := by
  rw [mnv4Res28Drop]
  have b1 := mnv4StridedGraphSync_shard epsStr R hR N hN (mnv4Row1 f) (by simp only; omega) w.b1 bf16 (by simp)
    e X he
  exact mnv4SkipDropGraphSync_shard (dpName 0) (SD 0)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row2 f) w.b2 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row2 f) w.b2).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row2 f) (by simp only; omega) w.b2 bf16 (by simp)
      (by simp))
    _ _ b1 r

/-- Trunk group **Res14a** at sync-BN with its drop sites — rows 3–6. -/
noncomputable def mnv4Res14aDropGraphSync (R : Nat) (hR : 0 < R) (N f : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (80 * (2 * (2 * f)) * (2 * (2 * f))))) :
    Fin R → SHlo (N * (160 * (2 * f) * (2 * f))) :=
  mnv4SkipDropGraphSync (dpName 3) (SD 3)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row6 f) w.b6 bf16)
    (mnv4SkipDropGraphSync (dpName 2) (SD 2)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row5 f) w.b5 bf16)
    (mnv4SkipDropGraphSync (dpName 1) (SD 1)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row4 f) w.b4 bf16)
    (mnv4StridedGraphSync epsStr R hR N (mnv4Row3 f) w.b3 bf16 e)))

theorem mnv4Res14aDropGraphSync_shard (R : Nat) (hR : 0 < R) (N f : Nat) (hN : 0 < N) (hf : 0 < f)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (80 * (2 * (2 * f)) * (2 * (2 * f))))) (X : Vec ((R * N) * (80 * (2 * (2 * f)) * (2 * (2 * f)))))
    (he : ∀ r, den (e r) = batchShard R N (80 * (2 * (2 * f)) * (2 * (2 * f))) X r) (r : Fin R) :
    den (mnv4Res14aDropGraphSync R hR N f epsStr w bf16 SD e r)
      = batchShard R N (160 * (2 * f) * (2 * f)) (mnv4Res14aDrop (R * N) f w SD X) r := by
  rw [mnv4Res14aDrop]
  have b3 := mnv4StridedGraphSync_shard epsStr R hR N hN (mnv4Row3 f) (by simp only; omega) w.b3 bf16 (by simp)
    e X he
  have b4 := mnv4SkipDropGraphSync_shard (dpName 1) (SD 1)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row4 f) w.b4 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row4 f) w.b4).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row4 f) (by simp only; omega) w.b4 bf16 (by simp)
      (by simp)) _ _ b3
  have b5 := mnv4SkipDropGraphSync_shard (dpName 2) (SD 2)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row5 f) w.b5 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row5 f) w.b5).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row5 f) (by simp only; omega) w.b5 bf16 (by simp)
      (by simp)) _ _ b4
  exact mnv4SkipDropGraphSync_shard (dpName 3) (SD 3)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row6 f) w.b6 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row6 f) w.b6).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row6 f) (by simp only; omega) w.b6 bf16 (by simp)
      (by simp)) _ _ b5 r

/-- Trunk group **Res14b** at sync-BN with its drop sites — rows 7–10: ExtraDW, ConvNeXt, FFN, ConvNeXt. -/
noncomputable def mnv4Res14bDropGraphSync (R : Nat) (hR : 0 < R) (N f : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (160 * (2 * f) * (2 * f)))) :
    Fin R → SHlo (N * (160 * (2 * f) * (2 * f))) :=
  mnv4SkipDropGraphSync (dpName 7) (SD 7)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row10 f) w.b10 bf16)
    (mnv4SkipDropGraphSync (dpName 6) (SD 6)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row9 f) w.b9 bf16)
    (mnv4SkipDropGraphSync (dpName 5) (SD 5)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row8 f) w.b8 bf16)
    (mnv4SkipDropGraphSync (dpName 4) (SD 4)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row7 f) w.b7 bf16) e)))

theorem mnv4Res14bDropGraphSync_shard (R : Nat) (hR : 0 < R) (N f : Nat) (hN : 0 < N) (hf : 0 < f)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (160 * (2 * f) * (2 * f)))) (X : Vec ((R * N) * (160 * (2 * f) * (2 * f))))
    (he : ∀ r, den (e r) = batchShard R N (160 * (2 * f) * (2 * f)) X r) (r : Fin R) :
    den (mnv4Res14bDropGraphSync R hR N f epsStr w bf16 SD e r)
      = batchShard R N (160 * (2 * f) * (2 * f)) (mnv4Res14bDrop (R * N) f w SD X) r := by
  rw [mnv4Res14bDrop]
  have b7 := mnv4SkipDropGraphSync_shard (dpName 4) (SD 4)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row7 f) w.b7 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row7 f) w.b7).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row7 f) (by simp only; omega) w.b7 bf16 (by simp)
      (by simp)) e X he
  have b8 := mnv4SkipDropGraphSync_shard (dpName 5) (SD 5)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row8 f) w.b8 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row8 f) w.b8).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN (mnv4Row8 f) (by simp only; omega) w.b8 bf16 (by simp)
      (by simp)) _ _ b7
  have b9 := mnv4SkipDropGraphSync_shard (dpName 6) (SD 6)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row9 f) w.b9 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row9 f) w.b9).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN (mnv4Row9 f) (by simp only; omega) w.b9 bf16 (by simp)
      (by simp)) _ _ b8
  exact mnv4SkipDropGraphSync_shard (dpName 7) (SD 7)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row10 f) w.b10 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row10 f) w.b10).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN (mnv4Row10 f) (by simp only; omega) w.b10 bf16 (by simp)
      (by simp)) _ _ b9 r

/-- Trunk group **Res7a** at sync-BN with its drop sites — rows 11–15. -/
noncomputable def mnv4Res7aDropGraphSync (R : Nat) (hR : 0 < R) (N f : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (160 * (2 * f) * (2 * f)))) :
    Fin R → SHlo (N * (256 * f * f)) :=
  mnv4SkipDropGraphSync (dpName 11) (SD 11)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row15 f) w.b15 bf16)
    (mnv4SkipDropGraphSync (dpName 10) (SD 10)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row14 f) w.b14 bf16)
    (mnv4SkipDropGraphSync (dpName 9) (SD 9)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row13 f) w.b13 bf16)
    (mnv4SkipDropGraphSync (dpName 8) (SD 8)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row12 f) w.b12 bf16)
    (mnv4StridedGraphSync epsStr R hR N (mnv4Row11 f) w.b11 bf16 e))))

theorem mnv4Res7aDropGraphSync_shard (R : Nat) (hR : 0 < R) (N f : Nat) (hN : 0 < N) (hf : 0 < f)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (160 * (2 * f) * (2 * f)))) (X : Vec ((R * N) * (160 * (2 * f) * (2 * f))))
    (he : ∀ r, den (e r) = batchShard R N (160 * (2 * f) * (2 * f)) X r) (r : Fin R) :
    den (mnv4Res7aDropGraphSync R hR N f epsStr w bf16 SD e r)
      = batchShard R N (256 * f * f) (mnv4Res7aDrop (R * N) f w SD X) r := by
  rw [mnv4Res7aDrop]
  have b11 := mnv4StridedGraphSync_shard epsStr R hR N hN (mnv4Row11 f) (by simp only; omega) w.b11 bf16
    (by simp) e X he
  have b12 := mnv4SkipDropGraphSync_shard (dpName 8) (SD 8)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row12 f) w.b12 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row12 f) w.b12).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row12 f) (by simp only; omega) w.b12 bf16 (by simp)
      (by simp)) _ _ b11
  have b13 := mnv4SkipDropGraphSync_shard (dpName 9) (SD 9)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row13 f) w.b13 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row13 f) w.b13).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row13 f) (by simp only; omega) w.b13 bf16 (by simp)
      (by simp)) _ _ b12
  have b14 := mnv4SkipDropGraphSync_shard (dpName 10) (SD 10)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row14 f) w.b14 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row14 f) w.b14).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row14 f) (by simp only; omega) w.b14 bf16 (by simp)
      (by simp)) _ _ b13
  exact mnv4SkipDropGraphSync_shard (dpName 11) (SD 11)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row15 f) w.b15 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row15 f) w.b15).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN (mnv4Row15 f) (by simp only; omega) w.b15 bf16 (by simp)
      (by simp)) _ _ b14 r

/-- Trunk group **Res7b** at sync-BN with its drop sites — rows 16–21. -/
noncomputable def mnv4Res7bDropGraphSync (R : Nat) (hR : 0 < R) (N f : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (256 * f * f))) :
    Fin R → SHlo (N * (256 * f * f)) :=
  mnv4SkipDropGraphSync (dpName 17) (SD 17)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row21 f) w.b21 bf16)
    (mnv4SkipDropGraphSync (dpName 16) (SD 16)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row20 f) w.b20 bf16)
    (mnv4SkipDropGraphSync (dpName 15) (SD 15)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row19 f) w.b19 bf16)
    (mnv4SkipDropGraphSync (dpName 14) (SD 14)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row18 f) w.b18 bf16)
    (mnv4SkipDropGraphSync (dpName 13) (SD 13)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row17 f) w.b17 bf16)
    (mnv4SkipDropGraphSync (dpName 12) (SD 12)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row16 f) w.b16 bf16) e)))))

theorem mnv4Res7bDropGraphSync_shard (R : Nat) (hR : 0 < R) (N f : Nat) (hN : 0 < N) (hf : 0 < f)
    (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (e : Fin R → SHlo (N * (256 * f * f))) (X : Vec ((R * N) * (256 * f * f)))
    (he : ∀ r, den (e r) = batchShard R N (256 * f * f) X r) (r : Fin R) :
    den (mnv4Res7bDropGraphSync R hR N f epsStr w bf16 SD e r)
      = batchShard R N (256 * f * f) (mnv4Res7bDrop (R * N) f w SD X) r := by
  rw [mnv4Res7bDrop]
  have b16 := mnv4SkipDropGraphSync_shard (dpName 12) (SD 12)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row16 f) w.b16 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row16 f) w.b16).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN (mnv4Row16 f) (by simp only; omega) w.b16 bf16 (by simp)
      (by simp)) e X he
  have b17 := mnv4SkipDropGraphSync_shard (dpName 13) (SD 13)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row17 f) w.b17 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row17 f) w.b17).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row17 f) (by simp only; omega) w.b17 bf16 (by simp)
      (by simp)) _ _ b16
  have b18 := mnv4SkipDropGraphSync_shard (dpName 14) (SD 14)
    (mnv4ExtraDWBodyGraphSync epsStr R hR N (mnv4Row18 f) w.b18 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row18 f) w.b18).fwd
    (mnv4ExtraDWBodyGraphSync_shard epsStr R hR N hN (mnv4Row18 f) (by simp only; omega) w.b18 bf16 (by simp)
      (by simp)) _ _ b17
  have b19 := mnv4SkipDropGraphSync_shard (dpName 15) (SD 15)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row19 f) w.b19 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row19 f) w.b19).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN (mnv4Row19 f) (by simp only; omega) w.b19 bf16 (by simp)
      (by simp)) _ _ b18
  have b20 := mnv4SkipDropGraphSync_shard (dpName 16) (SD 16)
    (mnv4FfnBodyGraphSync epsStr R hR N (mnv4Row20 f) w.b20 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row20 f) w.b20).fwd
    (mnv4FfnBodyGraphSync_shard epsStr R hR N hN (mnv4Row20 f) (by simp only; omega) w.b20 bf16 (by simp)
      (by simp)) _ _ b19
  exact mnv4SkipDropGraphSync_shard (dpName 17) (SD 17)
    (mnv4ConvNeXtBodyGraphSync epsStr R hR N (mnv4Row21 f) w.b21 bf16)
    (mnv4BodyOfRow (R * N) (mnv4Row21 f) w.b21).fwd
    (mnv4ConvNeXtBodyGraphSync_shard epsStr R hR N hN (mnv4Row21 f) (by simp only; omega) w.b21 bf16 (by simp)
      (by simp)) _ _ b20 r

-- ════════════════════════════════════════════════════════════════
-- § The whole net
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN data-parallel MobileNetV4-Conv-M forward graph with both masks, over the
    replica family.** `mnv4FwdGraphSyncFull` with each skip row's branch through a `dropPathB` at
    replica `r`'s shard of `SD k` and the classifier's input through a `dropoutB` at its shard of
    `M`. -/
noncomputable def mnv4FwdGraphSyncFullDrop (R : Nat) (hR : 0 < R) (N f : Nat) (epsStr mName : String)
    {nCls : Nat} (w : Mnv4BWeights nCls) (bf16 : Bool) (SD : Fin 18 → Vec (R * N))
    (M : Vec ((R * N) * 1280)) (e : Fin R → SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    Fin R → SHlo (N * nCls) :=
  mnv4HeadGraphSyncDo epsStr mName R hR N f f w.h1W w.h1b w.h1E w.h1g w.h1bt
    w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd bf16 M
    (mnv4Res7bDropGraphSync R hR N f epsStr w bf16 SD
      (mnv4Res7aDropGraphSync R hR N f epsStr w bf16 SD
        (mnv4Res14bDropGraphSync R hR N f epsStr w bf16 SD
          (mnv4Res14aDropGraphSync R hR N f epsStr w bf16 SD
            (mnv4Res28DropGraphSync R hR N f epsStr w bf16 SD
              (mnv4FusedGraphSync epsStr R hR N (2 * (2 * (2 * f))) (2 * (2 * (2 * f))) w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt bf16
                (mnv4StemGraphSync epsStr R hR N (2 * (2 * (2 * (2 * f)))) (2 * (2 * (2 * (2 * f)))) w.sW w.sb w.sE w.sg w.sbt bf16 e)))))))

/-- **At synchronised BatchNorm with both masks, replica `r`'s forward is shard `r` of the
    global-batch forward at the global masks.** Given that the replicas' inputs are the shards of
    one batch `X` of `R·N` images, the graph on replica `r` denotes `batchShard r` of
    `mobilenetv4ForwardBFullDrop (R * N) f w SD M X`. -/
theorem mnv4FwdGraphSyncFullDrop_shard (R : Nat) (hR : 0 < R) (N f : Nat) (hN : 0 < N) (hf : 0 < f)
    (epsStr mName : String) {nCls : Nat} (w : Mnv4BWeights nCls) (bf16 : Bool)
    (SD : Fin 18 → Vec (R * N)) (M : Vec ((R * N) * 1280))
    (e : Fin R → SHlo (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) (X : Vec ((R * N) * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f))))))))
    (he : ∀ r, den (e r) = batchShard R N (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))) X r) (r : Fin R) :
    den (mnv4FwdGraphSyncFullDrop R hR N f epsStr mName w bf16 SD M e r)
      = batchShard R N nCls (mobilenetv4ForwardBFullDrop (R * N) f w SD M X) r := by
  have h112 : 0 < 2 * (2 * (2 * (2 * f))) := by omega
  have s0 := mnv4StemGraphSync_shard epsStr R hR N (2 * (2 * (2 * (2 * f)))) (2 * (2 * (2 * (2 * f)))) hN h112 h112
    w.sW w.sb w.sE w.sg w.sbt bf16 e X he
  have s1 := mnv4FusedGraphSync_shard epsStr R hR N (2 * (2 * (2 * f))) (2 * (2 * (2 * f))) hN (by omega) (by omega)
    w.f0cW w.f0cb w.f0cE w.hf0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE w.hf0pE w.f0pg w.f0pbt bf16 _ _ s0
  have s2 := mnv4Res28DropGraphSync_shard R hR N f hN hf epsStr w bf16 SD _ _ s1
  have s3 := mnv4Res14aDropGraphSync_shard R hR N f hN hf epsStr w bf16 SD _ _ s2
  have s4 := mnv4Res14bDropGraphSync_shard R hR N f hN hf epsStr w bf16 SD _ _ s3
  have s5 := mnv4Res7aDropGraphSync_shard R hR N f hN hf epsStr w bf16 SD _ _ s4
  have s6 := mnv4Res7bDropGraphSync_shard R hR N f hN hf epsStr w bf16 SD _ _ s5
  unfold mobilenetv4ForwardBFullDrop mnv4Pre1 mnv4Pre0
  exact mnv4HeadGraphSyncDo_shard epsStr mName R hR N f f hN (by omega) (by omega)
    w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd bf16 M _ _ s6 r

end StableHLO

end Proofs
