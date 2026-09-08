import LeanMlir.Proofs.Training.ResNet34LiveSeal

/-!
# Full-depth live ResNet-34 witness + level-3 seal (`[3,4,6,3]`)

`ResNet34LivePC.liveFwd2` and its seal (`ResNet34LiveSeal`) use **empty** identity-block
chains (`chainComp [] = id`). This file fills them with the real ResNet-34 depth —
`[3,4,6,3] = 16` basic blocks (3 strided downsamples + 13 identity blocks) — and carries
both the level-2 non-vacuity and the level-3 nonzero-Jacobian seal through to the full net.

The key fact that makes this a *clean* extension (rather than a re-derivation): a live
identity block `idBlk2` has a **zeroed body**, so on a nonnegative activation it is the
affine shift `idBlk2 a = a + 1` (`relu(a + 1) = a + 1`). The shift is channel-symmetric, so
it is **transparent to the channel-difference carrier** `cd` (`cd (a + c) = cd a`) and to the
order invariant `Dom2`; and as an affine map it contributes the identity to the Jacobian. So
a chain of `k` identity blocks is just `a + k`: it multiplies the seal's `cd`-carrier by `1`,
leaving the four-`istd` product (and hence the directional derivative `≠ 0`) intact. The
seal's only kink remains the maxpool — handled exactly as in `ResNet34LiveSeal`.
-/

namespace Proofs
namespace ResNet34LiveFull

open scoped BigOperators
open Finset Filter Topology
open Proofs ResNet34Live2 ResNet34LivePC ResNet34LiveSeal

-- ════════════════════════════════════════════════════════════════
-- § The 2-channel identity residual block (zeroed body, BN (1,0,1))
--   The c = 2 peer of `Proofs.idBlk`: `relu(x + bn₂(conv₂(relu(bn₁(conv₁ x)))))` with
--   every conv zeroed, so the body collapses to the constant 1 and the block is `relu(x+1)`.
-- ════════════════════════════════════════════════════════════════

noncomputable def idBlk2 (h w : Nat) : Vec (2 * h * w) → Vec (2 * h * w) :=
  relu (2 * h * w) ∘ residual
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2))

/-- The identity block's body is the constant `1`. -/
theorem idBlk2_body_const (h w : Nat) (hhw : 0 < 2 * h * w) (a : Vec (2 * h * w)) :
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2)) a = (fun _ => (1 : ℝ)) := by
  simp only [Function.comp_apply]
  rw [flatConv_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]

theorem idBlk2_nonneg (h w : Nat) (a : Vec (2 * h * w)) (k : Fin (2 * h * w)) :
    0 ≤ idBlk2 h w a k := relu_nonneg (2 * h * w) _ k

noncomputable def idBlk2_hasVJPAt (h w : Nat) (hhw : 0 < 2 * h * w)
    (a : Vec (2 * h * w)) (ha : ∀ k, 0 ≤ a k) : HasVJPAt (idBlk2 h w) a :=
  resblock_has_vjp_at (h := h) (w := w) Zk2 Zb2 Zk2 Zb2 1 0 1 1 0 1 (by norm_num) (by norm_num) a
    (fun k => by
      rw [flatConv_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]
      change (1 : ℝ) ≠ 0; norm_num)
    (fun k => by
      rw [idBlk2_body_const h w hhw a]; change (1 : ℝ) + a k ≠ 0
      exact ne_of_gt (by linarith [ha k]))

theorem idBlk2_diffAt (h w : Nat) (hhw : 0 < 2 * h * w)
    (a : Vec (2 * h * w)) (ha : ∀ k, 0 ≤ a k) : DifferentiableAt ℝ (idBlk2 h w) a := by
  have hsm₁ : ∀ k, bnForward (2 * h * w) 1 0 1 (flatConv Zk2 Zb2 a) k ≠ 0 := fun k => by
    rw [flatConv_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]
    change (1 : ℝ) ≠ 0; norm_num
  have hF_diff : DifferentiableAt ℝ
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2)) a :=
    resblock_body_differentiableAt (h := h) (w := w) Zk2 Zb2 Zk2 Zb2 1 0 1 1 0 1
      (by norm_num) (by norm_num) a hsm₁
  have hsm_res : ∀ k, residual
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2)) a k ≠ 0 := fun k => by
    show ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2)) a k + a k ≠ 0
    rw [idBlk2_body_const h w hhw a]; change (1 : ℝ) + a k ≠ 0
    exact ne_of_gt (by linarith [ha k])
  show DifferentiableAt ℝ (relu (2 * h * w) ∘ residual
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2))) a
  exact (relu_differentiableAt_of_smooth (2 * h * w) _ hsm_res).comp a
    (DifferentiableAt.add hF_diff (differentiableAt_id))

/-- **On a nonnegative activation, the identity block is the affine shift `a + 1`** — the
    body is the constant 1 and the post-add ReLU is off (`a + 1 ≥ 1 > 0`). -/
theorem idBlk2_eq (h w : Nat) (hhw : 0 < 2 * h * w) (a : Vec (2 * h * w)) (ha : ∀ k, 0 ≤ a k) :
    idBlk2 h w a = (fun k => a k + 1) := by
  funext k
  have hbody := congrFun (idBlk2_body_const h w hhw a) k
  have hpos : 0 < residual
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2)) a k := by
    simp only [residual, biPath]; rw [hbody]; linarith [ha k]
  show relu (2 * h * w) (residual
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2)) a) k = a k + 1
  rw [relu_pos_eq _ k hpos]
  simp only [residual, biPath]; rw [hbody]; ring

-- ════════════════════════════════════════════════════════════════
-- § A chain of `k` identity blocks is the shift `a + k`
-- ════════════════════════════════════════════════════════════════

/-- **A chain of `k` identity blocks shifts a nonnegative activation by `k`.** -/
theorem idBlk2_chain_eq (h w : Nat) (hhw : 0 < 2 * h * w) (k : Nat) (a : Vec (2 * h * w))
    (ha : ∀ i, 0 ≤ a i) :
    chainComp (List.replicate k (idBlk2 h w)) a = (fun i => a i + (k : ℝ)) := by
  induction k with
  | zero => funext i; simp
  | succ n ih =>
    rw [List.replicate_succ, chainComp_cons, Function.comp_apply, ih,
        idBlk2_eq h w hhw _ (fun i => by
          have h1 := ha i; have h2 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n; linarith)]
    funext i; push_cast; ring

/-- Every running activation in the chain is nonnegative. -/
theorem idBlk2_chain_nonneg (h w : Nat) (hhw : 0 < 2 * h * w) (k : Nat) (a : Vec (2 * h * w))
    (ha : ∀ i, 0 ≤ a i) (i : Fin (2 * h * w)) :
    0 ≤ chainComp (List.replicate k (idBlk2 h w)) a i := by
  rw [idBlk2_chain_eq h w hhw k a ha]
  have h1 := ha i; have h2 : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k; linarith

/-- The per-block VJP+differentiability data for a chain of `k` identity blocks. -/
noncomputable def idBlk2_chainData (h w : Nat) (hhw : 0 < 2 * h * w) (k : Nat)
    (x : Vec (2 * h * w)) (hx : ∀ i, 0 ≤ x i) :
    ChainData x (List.replicate k (idBlk2 h w)) := by
  induction k with
  | zero => exact PUnit.unit
  | succ n ih =>
    rw [List.replicate_succ]
    exact ⟨idBlk2_diffAt h w hhw _ (fun i => idBlk2_chain_nonneg h w hhw n x hx i),
           idBlk2_hasVJPAt h w hhw _ (fun i => idBlk2_chain_nonneg h w hhw n x hx i),
           ih⟩

/-- Adding a channel-symmetric scalar preserves the top-left channel difference. -/
theorem cd_add_const {h w : Nat} [NeZero h] [NeZero w] (a : Vec (2 * h * w)) (c : ℝ) :
    cd (fun i => a i + c) = cd a := by
  simp only [cd, Tensor3.unflatten]; ring

/-- **The identity-block chain is transparent to the channel-difference carrier.** -/
theorem cd_chain (h w : Nat) (hhw : 0 < 2 * h * w) [NeZero h] [NeZero w] (k : Nat)
    (a : Vec (2 * h * w)) (ha : ∀ i, 0 ≤ a i) :
    cd (chainComp (List.replicate k (idBlk2 h w)) a) = cd a := by
  rw [idBlk2_chain_eq h w hhw k a ha, cd_add_const]

-- ════════════════════════════════════════════════════════════════
-- § Activation nonnegativity (every layer ends in relu / maxpool / bn-positive)
-- ════════════════════════════════════════════════════════════════

theorem stem2_nonneg (x : Vec (2 * (2 * 16) * (2 * 16))) (k : Fin (2 * 16 * 16)) :
    0 ≤ stem2 x k := by
  simp only [stem2, Function.comp_apply]; exact relu_nonneg _ _ k

theorem maxPool2_nonneg {c h w : Nat} {x : Tensor3 c (2 * h) (2 * w)}
    (hx : ∀ ci r s, 0 ≤ x ci r s) (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    0 ≤ maxPool2 x ci hi wi := by
  unfold maxPool2
  exact le_trans (hx _ _ _) (le_trans (le_max_left _ _) (le_max_left _ _))

theorem maxPoolFlat_nonneg {c h w : Nat} (v : Vec (c * (2 * h) * (2 * w)))
    (hv : ∀ k, 0 ≤ v k) (k : Fin (c * h * w)) : 0 ≤ maxPoolFlat c h w v k := by
  show 0 ≤ Tensor3.flatten (maxPool2 (Tensor3.unflatten v)) k
  unfold Tensor3.flatten
  exact maxPool2_nonneg (fun _ _ _ => hv _) _ _ _

-- ════════════════════════════════════════════════════════════════
-- § The full-depth witness and its whole-net VJP
-- ════════════════════════════════════════════════════════════════

/-- **The full-depth 2-channel live ResNet-34** — the real `[3,4,6,3]` block counts
    (3 strided downsamples + 13 identity blocks), 2 channels. -/
noncomputable def liveFwd2Full : Vec (2 * (2 * 16) * (2 * 16)) → Vec 2 :=
  dense Wd2 bd2 ∘ globalAvgPoolFlat 2 1 1 ∘
    chainComp (List.replicate 2 (idBlk2 1 1)) ∘ liveDownPC 1 1 ∘
    chainComp (List.replicate 5 (idBlk2 2 2)) ∘ liveDownPC 2 2 ∘
    chainComp (List.replicate 3 (idBlk2 4 4)) ∘ liveDownPC 4 4 ∘
    chainComp (List.replicate 3 (idBlk2 8 8)) ∘ maxPoolFlat 2 8 8 ∘ stem2

/-- The whole-net VJP at a point, given the stem + maxpool data there. Everything else
    (the identity-block chains, the strided downsamples, GAP, dense) is generic. -/
noncomputable def liveFwd2Full_has_vjp_at_of (x : Vec (2 * (2 * 16) * (2 * 16)))
    (hstem : PProd (HasVJPAt stem2 x) (DifferentiableAt ℝ stem2 x))
    (hmp : PProd (HasVJPAt (maxPoolFlat 2 8 8) (stem2 x))
                 (DifferentiableAt ℝ (maxPoolFlat 2 8 8) (stem2 x))) :
    HasVJPAt liveFwd2Full x :=
  resnet34_has_vjp_at stem2 (maxPoolFlat 2 8 8)
    (List.replicate 3 (idBlk2 8 8)) (liveDownPC 4 4)
    (List.replicate 3 (idBlk2 4 4)) (liveDownPC 2 2)
    (List.replicate 5 (idBlk2 2 2)) (liveDownPC 1 1)
    (List.replicate 2 (idBlk2 1 1))
    (globalAvgPoolFlat 2 1 1) (dense Wd2 bd2) x
    hstem hmp
    (idBlk2_chainData 8 8 (by norm_num) 3 _
      (fun i => maxPoolFlat_nonneg (c := 2) (h := 8) (w := 8) (stem2 x) (fun k => stem2_nonneg x k) i))
    ⟨liveDownPC_vjp 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)) _,
     liveDownPC_diff 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)) _⟩
    (idBlk2_chainData 4 4 (by norm_num) 3 _ (fun i => liveDownPC_nonneg 4 4 _ i))
    ⟨liveDownPC_vjp 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)) _,
     liveDownPC_diff 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)) _⟩
    (idBlk2_chainData 2 2 (by norm_num) 5 _ (fun i => liveDownPC_nonneg 2 2 _ i))
    ⟨liveDownPC_vjp 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)) _,
     liveDownPC_diff 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)) _⟩
    (idBlk2_chainData 1 1 (by norm_num) 2 _ (fun i => liveDownPC_nonneg 1 1 _ i))
    ⟨(globalAvgPoolFlat_has_vjp 2 1 1).toHasVJPAt _, (globalAvgPoolFlat_differentiable 2 1 1) _⟩
    ⟨(dense_has_vjp Wd2 bd2).toHasVJPAt _, (dense_differentiable Wd2 bd2) _⟩

/-- Whole-net VJP at the asymmetric witness `X2` (level-2 / non-vacuity witness). -/
noncomputable def liveFwd2Full_has_vjp_at_X2 : HasVJPAt liveFwd2Full X2 :=
  liveFwd2Full_has_vjp_at_of X2 ⟨stem2_vjp, stem2_diff⟩ ⟨hmp_vjp2, hmp_diff2⟩

theorem liveFwd2Full_has_vjp_correct (dy : Vec 2) (i : Fin (2 * (2 * 16) * (2 * 16))) :
    liveFwd2Full_has_vjp_at_X2.backward dy i = ∑ j : Fin 2, pdiv liveFwd2Full X2 i j * dy j :=
  liveFwd2Full_has_vjp_at_X2.correct dy i

/-- Whole-net VJP at the channel-symmetric base `Y` (the seal witness). -/
noncomputable def liveFwd2Full_has_vjp_at_Y : HasVJPAt liveFwd2Full Y :=
  liveFwd2Full_has_vjp_at_of Y ⟨stem2_vjp_Y, stem2_diff_Y⟩ ⟨hmp_vjp_Y, hmp_diff_Y⟩

-- ════════════════════════════════════════════════════════════════
-- § Differentiability of the full net at the base `Y`
-- ════════════════════════════════════════════════════════════════

theorem chain_diffAt (h w : Nat) (hhw : 0 < 2 * h * w) (k : Nat) (p : Vec (2 * h * w))
    (hp : ∀ i, 0 ≤ p i) :
    DifferentiableAt ℝ (chainComp (List.replicate k (idBlk2 h w))) p :=
  (chain_vjp_diff_at p (List.replicate k (idBlk2 h w)) (idBlk2_chainData h w hhw k p hp)).snd

theorem liveFwd2Full_diff_Y : DifferentiableAt ℝ liveFwd2Full Y := by
  have e1 : DifferentiableAt ℝ (maxPoolFlat 2 8 8 ∘ stem2) Y := hmp_diff_Y.comp Y stem2_diff_Y
  have e2 := (chain_diffAt 8 8 (by norm_num) 3 _
    (fun i => maxPoolFlat_nonneg (c := 2) (h := 8) (w := 8) (stem2 Y) (fun k => stem2_nonneg Y k) i)).comp Y e1
  have e3 := (liveDownPC_diff 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)) _).comp Y e2
  have e4 := (chain_diffAt 4 4 (by norm_num) 3 _ (fun i => liveDownPC_nonneg 4 4 _ i)).comp Y e3
  have e5 := (liveDownPC_diff 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)) _).comp Y e4
  have e6 := (chain_diffAt 2 2 (by norm_num) 5 _ (fun i => liveDownPC_nonneg 2 2 _ i)).comp Y e5
  have e7 := (liveDownPC_diff 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)) _).comp Y e6
  have e8 := (chain_diffAt 1 1 (by norm_num) 2 _ (fun i => liveDownPC_nonneg 1 1 _ i)).comp Y e7
  have e9 := ((globalAvgPoolFlat_differentiable 2 1 1).differentiableAt).comp Y e8
  exact ((dense_differentiable Wd2 bd2).differentiableAt).comp Y e9

-- ════════════════════════════════════════════════════════════════
-- § BN washes out a channel-symmetric shift ⇒ the chains collapse
-- ════════════════════════════════════════════════════════════════

/-- **BN is invariant to a constant shift** (it subtracts the mean). -/
theorem bnForward_shift (n : Nat) (hn : 0 < n) (ε γ β c : ℝ) (z : Vec n) :
    bnForward n ε γ β (fun i => z i + c) = bnForward n ε γ β z := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hmean : bnMean n (fun i => z i + c) = bnMean n z + c := by
    unfold bnMean
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  have hvar : bnVar n (fun i => z i + c) = bnVar n z := by
    simp only [bnVar, hmean]
    congr 1
    apply Finset.sum_congr rfl
    intro i _; ring
  have histd : bnIstd n (fun i => z i + c) ε = bnIstd n z ε := by unfold bnIstd; rw [hvar]
  funext k
  simp only [bnForward, bnXhat, hmean, histd]
  have h : z k + c - (bnMean n z + c) = z k - bnMean n z := by ring
  rw [h]

/-- Decimation commutes with a constant shift. -/
theorem decimate_shift (oc h w : Nat) (z : Vec (oc * (2 * h) * (2 * w))) (c : ℝ) :
    decimateFlat oc h w (fun i => z i + c) = (fun j => decimateFlat oc h w z j + c) := by
  funext j; rfl

/-- **A downsample absorbs an incoming channel-symmetric shift** (its BN washes it out). -/
theorem ld_absorb (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w))) (c : ℝ) :
    liveDownPC h w (fun i => a i + c) = liveDownPC h w a := by
  rw [liveDownPC_eq_ldS h w hhw hn, liveDownPC_eq_ldS h w hhw hn]
  funext k
  simp only [ldS]
  rw [decimate_shift, bnForward_shift (2 * h * w) hhw]

/-- GAP commutes with a constant shift. -/
theorem gap_add_const (u : Vec (2 * 1 * 1)) (c : ℝ) :
    globalAvgPoolFlat 2 1 1 (fun i => u i + c) = (fun j => globalAvgPoolFlat 2 1 1 u j + c) := by
  funext j; rw [gap_1x1, gap_1x1]; rfl

-- ════════════════════════════════════════════════════════════════
-- § The chains collapse: `liveFwd2Full = liveFwd2 + 2`
-- ════════════════════════════════════════════════════════════════

/-- **The identity-block chains wash out**: the full-depth net equals the empty-chain
    witness plus the constant `2` (the only shift that survives — the final chain, after
    which there is no BN). It cancels in the channel difference. -/
theorem liveFwd2Full_eq_add2 (v : Vec (2 * (2 * 16) * (2 * 16))) :
    liveFwd2Full v = (fun c => liveFwd2 v c + 2) := by
  have a0 : ∀ i, 0 ≤ maxPoolFlat 2 8 8 (stem2 v) i :=
    fun i => maxPoolFlat_nonneg (c := 2) (h := 8) (w := 8) (stem2 v) (fun k => stem2_nonneg v k) i
  have a1 : ∀ i, 0 ≤ liveDownPC 4 4 (maxPoolFlat 2 8 8 (stem2 v)) i := fun i => liveDownPC_nonneg 4 4 _ i
  have a2 : ∀ i, 0 ≤ liveDownPC 2 2 (liveDownPC 4 4 (maxPoolFlat 2 8 8 (stem2 v))) i :=
    fun i => liveDownPC_nonneg 2 2 _ i
  have a3 : ∀ i, 0 ≤ liveDownPC 1 1
      (liveDownPC 2 2 (liveDownPC 4 4 (maxPoolFlat 2 8 8 (stem2 v)))) i := fun i => liveDownPC_nonneg 1 1 _ i
  funext c
  simp only [liveFwd2Full, liveFwd2, Function.comp_apply,
    idBlk2_chain_eq 8 8 (by norm_num) 3 _ a0,
    ld_absorb 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)),
    idBlk2_chain_eq 4 4 (by norm_num) 3 _ a1,
    ld_absorb 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)),
    idBlk2_chain_eq 2 2 (by norm_num) 5 _ a2,
    ld_absorb 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)),
    idBlk2_chain_eq 1 1 (by norm_num) 2 _ a3,
    gap_add_const, dense_Wd2_apply]
  push_cast; ring

theorem liveFwd2Full_diff_eq (v : Vec (2 * (2 * 16) * (2 * 16))) :
    liveFwd2Full v 0 - liveFwd2Full v 1 = liveFwd2 v 0 - liveFwd2 v 1 := by
  rw [liveFwd2Full_eq_add2 v]; ring

-- ════════════════════════════════════════════════════════════════
-- § The full-depth seal (reuses `Rr` / `gd_hasDerivAt` via the collapse)
-- ════════════════════════════════════════════════════════════════

/-- The full net's output difference along the ray has the same derivative as the
    empty-chain witness (the chains are transparent to the channel difference). -/
theorem gd_full_hasDerivAt :
    HasDerivAt (fun t : ℝ => liveFwd2Full (Y + t • V) 0 - liveFwd2Full (Y + t • V) 1) (Rr 0) 0 := by
  have heq : (fun t : ℝ => liveFwd2Full (Y + t • V) 0 - liveFwd2Full (Y + t • V) 1)
      = (fun t : ℝ => liveFwd2S (Y + t • V) 0 - liveFwd2S (Y + t • V) 1) := by
    funext t; rw [liveFwd2Full_diff_eq, liveFwd2_eq_S]
  rw [heq]; exact gd_hasDerivAt

/-- **`fderiv ℝ liveFwd2Full Y ≠ 0`** — the full-depth `[3,4,6,3]` live ResNet-34's whole-net
    Jacobian is genuinely non-trivial at the witness base `Y` (level-3 seal, full depth). -/
theorem liveFwd2Full_jacobian_nonzero : fderiv ℝ liveFwd2Full Y ≠ 0 := by
  intro hzero
  have hfd : HasFDerivAt liveFwd2Full (0 : Vec (2 * (2 * 16) * (2 * 16)) →L[ℝ] Vec 2) Y := by
    rw [← hzero]; exact liveFwd2Full_diff_Y.hasFDerivAt
  have hsmul : HasDerivAt (fun t : ℝ => Y + t • V) V 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const V).const_add Y
  have hcomp : HasDerivAt (fun t : ℝ => liveFwd2Full (Y + t • V)) (0 : Vec 2) 0 := by
    have := HasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hfd hsmul (by simp)
    exact this
  have hpi := hasDerivAt_pi.mp hcomp
  have hd : HasDerivAt (fun t : ℝ => liveFwd2Full (Y + t • V) 0 - liveFwd2Full (Y + t • V) 1) 0 0 := by
    have := (hpi 0).sub (hpi 1)
    simp only [Pi.zero_apply, sub_zero] at this
    exact this
  exact (Rr_pos 0).ne' (gd_full_hasDerivAt.unique hd)

/-- **The full-depth level-3 seal** (Item A, full `[3,4,6,3]` depth): the proven whole-network
    backward of the full-depth live ResNet-34 is **not the zero map** at the witness base `Y`. -/
theorem liveFwd2Full_backward_nontrivial :
    ∃ (j₀ : Fin 2) (i₀ : Fin (2 * (2 * 16) * (2 * 16))),
      liveFwd2Full_has_vjp_at_Y.backward (basisVec j₀) i₀ ≠ 0 :=
  liveFwd2Full_has_vjp_at_Y.backward_nontrivial_of_fderiv_ne liveFwd2Full_jacobian_nonzero

/-- **The full-depth witness is non-degenerate** (level 2): `liveFwd2Full X2 ≠ liveFwd2Full 0`. -/
theorem liveFwd2Full_nonconstant : liveFwd2Full X2 ≠ liveFwd2Full (fun _ => (0 : ℝ)) := by
  rw [liveFwd2Full_eq_add2 X2, liveFwd2Full_eq_add2 (fun _ => (0 : ℝ))]
  intro h
  apply liveFwd2_nonconstant
  funext c
  linarith [congrFun h c]

end ResNet34LiveFull
end Proofs
