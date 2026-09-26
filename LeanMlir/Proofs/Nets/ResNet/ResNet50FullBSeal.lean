import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBSeal

/-!
# ResNet-50's non-degeneracy seal, on the full-width batched net (levels 2 and 3)

`ResNet50FullBVJP.lean` proves
`resnet50ForwardBFullHasVJPAt` under 48 relu clauses — three per bottleneck — plus the stem's
and the stem pool's. This file exhibits the clause bundle's joint satisfiability, and a nonzero
Jacobian with it, on `resnet50ForwardBFull` itself.

## What is inherited and what is new

Almost everything is [`Training/BatchSealKit.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Training/BatchSealKit.lean)'s or ResNet-34's. The carrier (`EDiff` and its
per-op steps), the ray, the centre-tap kernel and the 7×7/s2 stem's no-tie are the kit's; the
strided projection (`R34FullBSeal.sealProj`), the zeroed-body collapse (`projB_zero_const`), the
relu-removal (`cbReluStridedB_eq`) and the stem's own collapse (`r34StemB_eq`) are ResNet-34's,
stated at variable shapes and instantiated here — the same reuse `ResNet50FullB.lean` already makes
of `r34StemB` and `r34HeadB` themselves.

Three things are genuinely new:

* **three convolutions per block, so three relu clauses.** The bottleneck's `hm2` sits after the
  3×3 — but with `W₂` zeroed it sees a constant channel, so it is `β₂ = 1 ≠ 0` and, like `hm1`,
  needs nothing of the activation. Only `hout` does, through `0 ≤ activation`;
* **a stride-1 projection.** Stage 1 block 0 changes channels (64 → 256) at unchanged resolution,
  so its skip is `projB`, not `projStridedB`, and the carrier crosses it through the kit's
  stride-1 `eDiff_conv`. That also makes the carrier's BN count **five**, not the four of
  ResNet-34: stem, `s1b0`, `s2b0`, `s3b0`, `s4b0`;
* **`q` stays a binder.** ResNet-50 ships at two resolutions and the VJP is stated at both, so the
  seal is too: `0 < q` and `q ≤ 7` are all the witness needs, and `q = 7` (224 px) and `q = 5`
  (160 px) are instances of one theorem. The bound is the `β = 160`
  margin against the widest BN, the stem's `2·(16q)² = 512q²`; a larger `q` wants a larger `β`,
  not a different argument.
-/

namespace Proofs
namespace R50FullBSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs BatchSeal

-- ════════════════════════════════════════════════════════════════
-- § 1. The structural weights
-- ════════════════════════════════════════════════════════════════

/-- The structural identity bottleneck: all three convs zeroed, every BN `(ε, γ, β) = (1, 1, 1)`.
    The body is then the constant `1` and the block is `a ↦ a + 1` on a nonnegative activation. -/
noncomputable def sealIdW (mid oc : Nat) : R50IdW mid oc where
  W₁ := zk mid oc 1 1
  b₁ := kv mid 0
  ε₁ := 1
  γ₁ := kv mid 1
  β₁ := kv mid 1
  W₂ := zk mid mid 3 3
  b₂ := kv mid 0
  ε₂ := 1
  γ₂ := kv mid 1
  β₂ := kv mid 1
  W₃ := zk oc mid 1 1
  b₃ := kv oc 0
  ε₃ := 1
  γ₃ := kv oc 1
  β₃ := kv oc 1

/-- The structural projection bottleneck — one record for both projection forms, as `R50ProjW` is.
    Zeroed body, centre-tap 1×1 skip, `β_p = 160` (the margin that keeps the post-residual relu off
    its kink at every input). -/
noncomputable def sealPrW (ic mid oc : Nat) : R50ProjW ic mid oc where
  W₁ := zk mid ic 1 1
  b₁ := kv mid 0
  ε₁ := 1
  γ₁ := kv mid 1
  β₁ := kv mid 1
  W₂ := zk mid mid 3 3
  b₂ := kv mid 0
  ε₂ := 1
  γ₂ := kv mid 1
  β₂ := kv mid 1
  W₃ := zk oc mid 1 1
  b₃ := kv oc 0
  ε₃ := 1
  γ₃ := kv oc 1
  β₃ := kv oc 1
  Wp := ctK oc ic 1 1 1
  bp := kv oc 0
  εp := 1
  γp := kv oc 1
  βp := kv oc 160

/-- **The witness weights**, generic in the class count. -/
noncomputable def sealW (nCls : Nat) : R50BWeights nCls where
  sW := ctK 64 3 7 7 1
  sb := kv 64 0
  sε := 1
  sγ := kv 64 1
  sβ := kv 64 160
  s1b0 := sealPrW 64 64 256
  s1b1 := sealIdW 64 256
  s1b2 := sealIdW 64 256
  s2b0 := sealPrW 256 128 512
  s2b1 := sealIdW 128 512
  s2b2 := sealIdW 128 512
  s2b3 := sealIdW 128 512
  s3b0 := sealPrW 512 256 1024
  s3b1 := sealIdW 256 1024
  s3b2 := sealIdW 256 1024
  s3b3 := sealIdW 256 1024
  s3b4 := sealIdW 256 1024
  s3b5 := sealIdW 256 1024
  s4b0 := sealPrW 1024 512 2048
  s4b1 := sealIdW 512 2048
  s4b2 := sealIdW 512 2048
  Wd := fun i j => if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0
  bd := kv nCls 0

theorem sealW_Wd (nCls : Nat) :
    (sealW nCls).Wd = fun (i : Fin 2048) (j : Fin nCls) =>
      if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0 := rfl

-- ════════════════════════════════════════════════════════════════
-- § 2. The margin at every BN width, from `q ≤ 7`
-- ════════════════════════════════════════════════════════════════

/-- Every BN width in the witness clears the `β = 160` margin once `q ≤ 7`. The widest is the
    stem's `2·(16q)² = 512q² ≤ 25088 < 25600`; the five carrier sites are `512q²`, `128q²`, `32q²`,
    `8q²` and `2q²`. -/
theorem margin_q (n q : Nat) (hq : q ≤ 7) (hn : n ≤ 512 * (q * q)) :
    |(1 : ℝ)| * Real.sqrt ((n : ℕ) : ℝ) < 160 := by
  refine margin160 n ?_
  have hq2 : q * q ≤ 49 := Nat.mul_le_mul hq hq
  have hb : n ≤ 25088 := le_trans hn (by omega)
  have : ((n : ℕ) : ℝ) ≤ 25088 := by exact_mod_cast hb
  linarith

/-- `0 < 2 * (a * a)` — the `0 < N * (h * w)` a zeroed body's BN needs. -/
theorem two_sq_pos (a : Nat) (ha : 0 < a) : 0 < 2 * (a * a) :=
  Nat.mul_pos (by norm_num) (Nat.mul_pos ha ha)

-- ════════════════════════════════════════════════════════════════
-- § 3. What the structural blocks are
--   ⚠⚠ As in ResNet-34: every lemma at VARIABLE shapes, instantiated afterwards. See
--   `ResNet34FullBSeal.lean`'s §2 banner for what happens otherwise.
-- ════════════════════════════════════════════════════════════════

/-- The bottleneck body is the constant `1` at every input — the third conv is zeroed, so
    `projB_zero_const` collapses it whatever the two stages beneath it do. -/
theorem seal_id_body (N h w mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (oc * h * w))) :
    (projB N (h := h) (w := w) (sealIdW mid oc).W₃ (sealIdW mid oc).b₃ (sealIdW mid oc).ε₃
        (sealIdW mid oc).γ₃ (sealIdW mid oc).β₃ ∘
      StableHLO.cbReluB N (h := h) (w := w) (sealIdW mid oc).W₂ (sealIdW mid oc).b₂
          (sealIdW mid oc).ε₂ (sealIdW mid oc).γ₂ (sealIdW mid oc).β₂ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealIdW mid oc).W₁ (sealIdW mid oc).b₁
          (sealIdW mid oc).ε₁ (sealIdW mid oc).γ₁ (sealIdW mid oc).β₁) v = fun _ => (1 : ℝ) :=
  R34FullBSeal.projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1
    (fun _ => rfl) _

/-- The stride-1 projection block's body is the constant `1`. -/
theorem seal_pr_body (N h w ic mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (ic * h * w))) :
    (projB N (h := h) (w := w) (sealPrW ic mid oc).W₃ (sealPrW ic mid oc).b₃
        (sealPrW ic mid oc).ε₃ (sealPrW ic mid oc).γ₃ (sealPrW ic mid oc).β₃ ∘
      StableHLO.cbReluB N (h := h) (w := w) (sealPrW ic mid oc).W₂ (sealPrW ic mid oc).b₂
          (sealPrW ic mid oc).ε₂ (sealPrW ic mid oc).γ₂ (sealPrW ic mid oc).β₂ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealPrW ic mid oc).W₁ (sealPrW ic mid oc).b₁
          (sealPrW ic mid oc).ε₁ (sealPrW ic mid oc).γ₁ (sealPrW ic mid oc).β₁) v
      = fun _ => (1 : ℝ) :=
  R34FullBSeal.projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1
    (fun _ => rfl) _

/-- The strided projection block's body is the constant `1`. v1.5: its middle stage is the
    STRIDED conv-bn-relu, so this is not `seal_pr_body` at other shapes. -/
theorem seal_dn_body (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    (projB N (h := h) (w := w) (sealPrW ic mid oc).W₃ (sealPrW ic mid oc).b₃
        (sealPrW ic mid oc).ε₃ (sealPrW ic mid oc).γ₃ (sealPrW ic mid oc).β₃ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) (sealPrW ic mid oc).W₂ (sealPrW ic mid oc).b₂
          (sealPrW ic mid oc).ε₂ (sealPrW ic mid oc).γ₂ (sealPrW ic mid oc).β₂ ∘
        StableHLO.cbReluB N (h := 2 * h) (w := 2 * w) (sealPrW ic mid oc).W₁
          (sealPrW ic mid oc).b₁ (sealPrW ic mid oc).ε₁ (sealPrW ic mid oc).γ₁
          (sealPrW ic mid oc).β₁) v = fun _ => (1 : ℝ) :=
  R34FullBSeal.projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1
    (fun _ => rfl) _

/-- **The structural bottleneck is the shift `a ↦ a + 1`** on a nonnegative activation. -/
theorem sealIdB_eq (N h w mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (oc * h * w)))
    (hv : ∀ k, 0 ≤ v k) :
    r50IdB N h w (sealIdW mid oc) v = fun k => v k + 1 := by
  have hbody := seal_id_body N h w mid oc hn v
  have hres : ∀ k, residual
      (projB N (h := h) (w := w) (sealIdW mid oc).W₃ (sealIdW mid oc).b₃ (sealIdW mid oc).ε₃
          (sealIdW mid oc).γ₃ (sealIdW mid oc).β₃ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealIdW mid oc).W₂ (sealIdW mid oc).b₂
            (sealIdW mid oc).ε₂ (sealIdW mid oc).γ₂ (sealIdW mid oc).β₂ ∘
          StableHLO.cbReluB N (h := h) (w := w) (sealIdW mid oc).W₁ (sealIdW mid oc).b₁
            (sealIdW mid oc).ε₁ (sealIdW mid oc).γ₁ (sealIdW mid oc).β₁) v k = v k + 1 := by
    intro k
    rw [residual_apply, hbody]
    ring
  funext k
  show relu (N * (oc * h * w)) (residual _ v) k = v k + 1
  rw [relu_id_of_pos (fun i => by rw [hres i]; linarith [hv i]), hres k]

/-- The structural STRIDE-1 projection: a 1×1 centre-tap conv-BN. -/
noncomputable def sealProj1 (N h w ic oc : Nat) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) (ctK oc ic 1 1 1) (kv oc 0) 1 (kv oc 1) (kv oc 160)

theorem sealProj1_apply (N h w ic oc : Nat) (v : Vec (N * (ic * h * w))) :
    sealProj1 N h w ic oc v
      = StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160)
          (StableHLO.batchMap N (flatConv (h := h) (w := w) (ctK oc ic 1 1 1) (kv oc 0)) v) := rfl

theorem sealProj1_pos (N h w ic oc : Nat)
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * h * w))) (k : Fin (N * (oc * h * w))) :
    0 < sealProj1 N h w ic oc v k :=
  bnBatchLA_pos 1 one_pos (kv oc 1) (kv oc 160) 1 160 (fun _ => rfl) (fun _ => rfl) hm _ k

/-- **The structural stride-1 projection block is its projection plus one.** -/
theorem sealPrB_eq (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160) (v : Vec (N * (ic * h * w))) :
    r50ProjB N h w (sealPrW ic mid oc) v = fun k => sealProj1 N h w ic oc v k + 1 := by
  have hbody := seal_pr_body N h w ic mid oc hn v
  have hres : ∀ k, residualProj
      (projB N (h := h) (w := w) (sealPrW ic mid oc).Wp (sealPrW ic mid oc).bp
        (sealPrW ic mid oc).εp (sealPrW ic mid oc).γp (sealPrW ic mid oc).βp)
      (projB N (h := h) (w := w) (sealPrW ic mid oc).W₃ (sealPrW ic mid oc).b₃
          (sealPrW ic mid oc).ε₃ (sealPrW ic mid oc).γ₃ (sealPrW ic mid oc).β₃ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealPrW ic mid oc).W₂ (sealPrW ic mid oc).b₂
            (sealPrW ic mid oc).ε₂ (sealPrW ic mid oc).γ₂ (sealPrW ic mid oc).β₂ ∘
          StableHLO.cbReluB N (h := h) (w := w) (sealPrW ic mid oc).W₁ (sealPrW ic mid oc).b₁
            (sealPrW ic mid oc).ε₁ (sealPrW ic mid oc).γ₁ (sealPrW ic mid oc).β₁) v k
      = sealProj1 N h w ic oc v k + 1 := by
    intro k
    show sealProj1 N h w ic oc v k + _ = _
    rw [hbody]
  funext k
  show relu (N * (oc * h * w)) (residualProj _ _ v) k = sealProj1 N h w ic oc v k + 1
  rw [relu_id_of_pos (fun i => by rw [hres i]; linarith [sealProj1_pos N h w ic oc hm v i]),
    hres k]

/-- **The structural strided projection block is ResNet-34's strided projection plus one.** -/
theorem sealDnB_eq (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    r50DownB N h w (sealPrW ic mid oc) v
      = fun k => R34FullBSeal.sealProj N h w ic oc v k + 1 := by
  have hbody := seal_dn_body N h w ic mid oc hn v
  have hres : ∀ k, residualProj
      (StableHLO.projStridedB N (h := h) (w := w) (sealPrW ic mid oc).Wp (sealPrW ic mid oc).bp
        (sealPrW ic mid oc).εp (sealPrW ic mid oc).γp (sealPrW ic mid oc).βp)
      (projB N (h := h) (w := w) (sealPrW ic mid oc).W₃ (sealPrW ic mid oc).b₃
          (sealPrW ic mid oc).ε₃ (sealPrW ic mid oc).γ₃ (sealPrW ic mid oc).β₃ ∘
        StableHLO.cbReluStridedB N (h := h) (w := w) (sealPrW ic mid oc).W₂
            (sealPrW ic mid oc).b₂ (sealPrW ic mid oc).ε₂ (sealPrW ic mid oc).γ₂
            (sealPrW ic mid oc).β₂ ∘
          StableHLO.cbReluB N (h := 2 * h) (w := 2 * w) (sealPrW ic mid oc).W₁
            (sealPrW ic mid oc).b₁ (sealPrW ic mid oc).ε₁ (sealPrW ic mid oc).γ₁
            (sealPrW ic mid oc).β₁) v k
      = R34FullBSeal.sealProj N h w ic oc v k + 1 := by
    intro k
    show R34FullBSeal.sealProj N h w ic oc v k + _ = _
    rw [hbody]
  funext k
  show relu (N * (oc * h * w)) (residualProj _ _ v) k
    = R34FullBSeal.sealProj N h w ic oc v k + 1
  rw [relu_id_of_pos (fun i => by
    rw [hres i]; linarith [R34FullBSeal.sealProj_pos N h w ic oc hm v i]), hres k]

-- ════════════════════════════════════════════════════════════════
-- § 4. Every running activation is nonnegative, and the clause bundles
-- ════════════════════════════════════════════════════════════════

theorem r50IdB_nonneg (N h w mid oc : Nat) (p : R50IdW mid oc) (v : Vec (N * (oc * h * w)))
    (k : Fin (N * (oc * h * w))) : 0 ≤ r50IdB N h w p v k := relu_nonneg _ _ k

theorem r50ProjB_nonneg (N h w ic mid oc : Nat) (p : R50ProjW ic mid oc)
    (v : Vec (N * (ic * h * w))) (k : Fin (N * (oc * h * w))) :
    0 ≤ r50ProjB N h w p v k := relu_nonneg _ _ k

theorem r50DownB_nonneg (N h w ic mid oc : Nat) (p : R50ProjW ic mid oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (k : Fin (N * (oc * h * w))) :
    0 ≤ r50DownB N h w p v k := relu_nonneg _ _ k

theorem seal_id_pos (mid oc : Nat) : R50IdPos (sealIdW mid oc) := ⟨one_pos, one_pos, one_pos⟩

theorem seal_pr_pos (ic mid oc : Nat) : R50ProjPos (sealPrW ic mid oc) :=
  ⟨one_pos, one_pos, one_pos, one_pos⟩

/-- **The bottleneck's three relu clauses.** Both interior ones see a constant channel (their
    convs are zeroed), so they are `β = 1 ≠ 0` and weight-only; only the post-residual one needs
    the activation, and only through `0 ≤ ·`. -/
theorem seal_id_smooth (N h w mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (oc * h * w)))
    (hv : ∀ k, 0 ≤ v k) : R50IdSmoothAt N h w (sealIdW mid oc) v where
  hm1 := by
    intro k
    show StableHLO.bnBatchLA N mid h w 1 (kv mid 1) (kv mid 1)
      (StableHLO.batchMap N (flatConv (zk mid oc 1 1) (kv mid 0)) v) k ≠ 0
    rw [batchMap_flatConv_zero (zk mid oc 1 1) (kv mid 0) (fun _ _ _ _ => rfl) (fun _ => rfl),
      bnBatchLA_const hn 1 (kv mid 1) (kv mid 1) 1 0 (fun _ => rfl) k]
    norm_num
  hm2 := by
    intro k
    show StableHLO.bnBatchLA N mid h w 1 (kv mid 1) (kv mid 1)
      (StableHLO.batchMap N (flatConv (zk mid mid 3 3) (kv mid 0)) _) k ≠ 0
    rw [batchMap_flatConv_zero (zk mid mid 3 3) (kv mid 0) (fun _ _ _ _ => rfl) (fun _ => rfl),
      bnBatchLA_const hn 1 (kv mid 1) (kv mid 1) 1 0 (fun _ => rfl) k]
    norm_num
  hout := by
    intro k
    show _ + v k ≠ 0
    rw [congrFun (seal_id_body N h w mid oc hn v) k]
    intro hc
    linarith [hv k]

/-- The stride-1 projection block's three relu clauses — all weight-only. -/
theorem seal_pr_smooth (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160) (v : Vec (N * (ic * h * w))) :
    R50ProjSmoothAt N h w (sealPrW ic mid oc) v where
  hm1 := by
    intro k
    show StableHLO.bnBatchLA N mid h w 1 (kv mid 1) (kv mid 1)
      (StableHLO.batchMap N (flatConv (zk mid ic 1 1) (kv mid 0)) v) k ≠ 0
    rw [batchMap_flatConv_zero (zk mid ic 1 1) (kv mid 0) (fun _ _ _ _ => rfl) (fun _ => rfl),
      bnBatchLA_const hn 1 (kv mid 1) (kv mid 1) 1 0 (fun _ => rfl) k]
    norm_num
  hm2 := by
    intro k
    show StableHLO.bnBatchLA N mid h w 1 (kv mid 1) (kv mid 1)
      (StableHLO.batchMap N (flatConv (zk mid mid 3 3) (kv mid 0)) _) k ≠ 0
    rw [batchMap_flatConv_zero (zk mid mid 3 3) (kv mid 0) (fun _ _ _ _ => rfl) (fun _ => rfl),
      bnBatchLA_const hn 1 (kv mid 1) (kv mid 1) 1 0 (fun _ => rfl) k]
    norm_num
  hout := by
    intro k
    show sealProj1 N h w ic oc v k + _ ≠ 0
    rw [congrFun (seal_pr_body N h w ic mid oc hn v) k]
    intro hc
    linarith [sealProj1_pos N h w ic oc hm v k]

/-- The strided projection block's three relu clauses — all weight-only. v1.5: `hm1` is at the
    INPUT resolution and only `hm2` is at the halved one. -/
theorem seal_dn_smooth (N h w ic mid oc : Nat) (hn2 : 0 < N * ((2 * h) * (2 * w)))
    (hn : 0 < N * (h * w)) (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : R50DownSmoothAt N h w (sealPrW ic mid oc) v where
  hm1 := by
    intro k
    show StableHLO.bnBatchLA N mid (2 * h) (2 * w) 1 (kv mid 1) (kv mid 1)
      (StableHLO.batchMap N (flatConv (zk mid ic 1 1) (kv mid 0)) v) k ≠ 0
    rw [batchMap_flatConv_zero (zk mid ic 1 1) (kv mid 0) (fun _ _ _ _ => rfl) (fun _ => rfl),
      bnBatchLA_const hn2 1 (kv mid 1) (kv mid 1) 1 0 (fun _ => rfl) k]
    norm_num
  hm2 := by
    intro k
    show StableHLO.bnBatchLA N mid h w 1 (kv mid 1) (kv mid 1)
      (StableHLO.batchMap N (flatConvStride2 (zk mid mid 3 3) (kv mid 0)) _) k ≠ 0
    rw [batchMap_flatConvStride2_zero (zk mid mid 3 3) (kv mid 0) (fun _ _ _ _ => rfl)
        (fun _ => rfl),
      bnBatchLA_const hn 1 (kv mid 1) (kv mid 1) 1 0 (fun _ => rfl) k]
    norm_num
  hout := by
    intro k
    show R34FullBSeal.sealProj N h w ic oc v k + _ ≠ 0
    rw [congrFun (seal_dn_body N h w ic mid oc hn v) k]
    intro hc
    linarith [R34FullBSeal.sealProj_pos N h w ic oc hm v k]

-- ════════════════════════════════════════════════════════════════
-- § 5. The witness input, the ray and the stem — all of it `BatchSeal`'s at this net's spelling
-- ════════════════════════════════════════════════════════════════

/-- The witness input: the shared ray at `32q × 32q` (224 px at `q = 7`, 160 at `q = 5`). -/
noncomputable def sealX (q : Nat) (t : ℝ) : Vec (2 * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) :=
  rayX (2 * (2 * (2 * (2 * (2 * q))))) (2 * (2 * (2 * (2 * (2 * q))))) t

/-- Its direction — all of example 0's channel 0. -/
noncomputable def sealV (q : Nat) : Vec (2 * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) :=
  rayV (2 * (2 * (2 * (2 * (2 * q))))) (2 * (2 * (2 * (2 * (2 * q)))))

theorem sealX_zero_add (q : Nat) (t : ℝ) : sealX q 0 + t • sealV q = sealX q t := by
  rw [sealX, sealX, sealV]
  exact rayX_zero_add _ _ t

theorem eDiff_sealX (q : Nat) (t : ℝ) :
    EDiff (fun ci => if ci.val = 0 then t else 0) (sealX q t) := by
  rw [sealX]
  exact eDiff_rayX _ _ t

/-- The stem's centre-tap conv output — the carrier's first stop. -/
noncomputable def Zs (q : Nat) (t : ℝ) : Vec (2 * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) :=
  ctConv 64 7 7 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) t

/-- `2·a² ≤ 512·q²` whenever `a ≤ 16q` — every BN width against the stem's. -/
theorem bnd (a q : Nat) (ha : a ≤ 16 * q) : 2 * (a * a) ≤ 512 * (q * q) := by
  calc 2 * (a * a) ≤ 2 * ((16 * q) * (16 * q)) :=
        Nat.mul_le_mul_left _ (Nat.mul_le_mul ha ha)
    _ = 512 * (q * q) := by ring

/-- The `β = 160` margin at the stem's `2·(16q)²`. -/
theorem margin_stem (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q))))) : ℕ) : ℝ) < 160 :=
  margin_q _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 1's `2·(8q)²`. -/
theorem margin_p1 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * (2 * (2 * q))) * (2 * (2 * (2 * q)))) : ℕ) : ℝ) < 160 :=
  margin_q _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 2's `2·(4q)²`. -/
theorem margin_p2 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * (2 * q)) * (2 * (2 * q))) : ℕ) : ℝ) < 160 :=
  margin_q _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 3's `2·(2q)²`. -/
theorem margin_p3 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * q) * (2 * q)) : ℕ) : ℝ) < 160 :=
  margin_q _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 4's `2·q²`. -/
theorem margin_p4 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * (q * q) : ℕ) : ℝ) < 160 :=
  margin_q _ q hq (bnd _ q (by omega))

/-- The stem BN is strictly positive at every point of the ray. -/
theorem Zs_bn_pos (q : Nat) (hq : q ≤ 7) (t : ℝ) (k : Fin (2 * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q))))))) :
    0 < StableHLO.bnBatchLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 1 (kv 64 1) (kv 64 160) (Zs q t) k := by
  rw [Zs]
  exact ctConv_bn_pos 64 7 7 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (margin_stem q hq) t k

/-- The stem's relu is off at the witness, so the pool's no-tie can be read on the BN output. -/
theorem stem_relu_off (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    StableHLO.cbReluStridedB 2 (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q)))))
      (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ (sealX q t)
      = StableHLO.bnBatchLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 1 (kv 64 1) (kv 64 160) (Zs q t) :=
  R34FullBSeal.cbReluStridedB_eq _ _ _ _ _ (sealX q t) (fun k => Zs_bn_pos q hq t k)

theorem seal_stem_clause (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R34StemSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
      (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ (sealX q t) :=
  R34FullBSeal.seal_stem_smooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 3 64 _ _ (margin_stem q hq) (sealX q t)

theorem seal_pool_clause (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R34PoolSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
      (StableHLO.cbReluStridedB 2 (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q)))))

        (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ (sealX q t)) := by
  rw [stem_relu_off q hq, Zs]
  exact ctConv_pool_smooth 64 7 7 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (by norm_num) (by norm_num) t

-- ════════════════════════════════════════════════════════════════
-- § 6. The running activations: nonnegative, and collapsed
-- ════════════════════════════════════════════════════════════════

/-- The stem, collapsed: its relu is off, so it is pool ∘ bn ∘ centre-tap conv. -/
theorem pc0 (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    r50Pre0 2 q (sealW nCls) (sealX q t)
      = StableHLO.batchMap 2 (maxPool3s2Flat 64 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))))
          (StableHLO.bnBatchLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 1 (kv 64 1) (kv 64 160) (Zs q t)) := by
  rw [r50Pre0_apply]
  refine R34FullBSeal.r34StemB_eq _ _ _ _ _ (sealX q t) (fun k => ?_)
  exact Zs_bn_pos q hq t k

theorem nn1 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre1 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre1_apply]
  exact r50ProjB_nonneg 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 64 256 _ _ k

theorem nn2 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre2 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre2_apply]
  exact r50IdB_nonneg 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 _ _ k

theorem nn4 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre4 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre4_apply]
  exact r50DownB_nonneg 2 (2 * (2 * q)) (2 * (2 * q)) 256 128 512 _ _ k

theorem nn5 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre5 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre5_apply]
  exact r50IdB_nonneg 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 _ _ k

theorem nn6 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre6 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre6_apply]
  exact r50IdB_nonneg 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 _ _ k

theorem nn8 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre8 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre8_apply]
  exact r50DownB_nonneg 2 (2 * q) (2 * q) 512 256 1024 _ _ k

theorem nn9 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre9 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre9_apply]
  exact r50IdB_nonneg 2 (2 * q) (2 * q) 256 1024 _ _ k

theorem nn10 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre10 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre10_apply]
  exact r50IdB_nonneg 2 (2 * q) (2 * q) 256 1024 _ _ k

theorem nn11 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre11 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre11_apply]
  exact r50IdB_nonneg 2 (2 * q) (2 * q) 256 1024 _ _ k

theorem nn12 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre12 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre12_apply]
  exact r50IdB_nonneg 2 (2 * q) (2 * q) 256 1024 _ _ k

theorem nn14 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre14 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre14_apply]
  exact r50DownB_nonneg 2 q q 1024 512 2048 _ _ k

theorem nn15 (q : Nat) (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r50Pre15 2 q (sealW nCls) (sealX q t) k := by
  intro k
  rw [r50Pre15_apply]
  exact r50IdB_nonneg 2 q q 512 2048 _ _ k

theorem pc1 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    r50Pre1 2 q (sealW nCls) (sealX q t)
      = fun k => sealProj1 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (r50Pre0 2 q (sealW nCls) (sealX q t)) k + 1 := by
  rw [r50Pre1_apply]
  exact sealPrB_eq 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) (margin_p1 q hq) _

theorem pc2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre2 2 q (sealW nCls) (sealX q t) = fun k => r50Pre1 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre2_apply]
  exact sealIdB_eq 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) _ (nn1 q nCls t)

theorem pc3 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre3 2 q (sealW nCls) (sealX q t) = fun k => r50Pre2 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre3_apply]
  exact sealIdB_eq 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) _ (nn2 q nCls t)

theorem pc4 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    r50Pre4 2 q (sealW nCls) (sealX q t)
      = fun k => R34FullBSeal.sealProj 2 (2 * (2 * q)) (2 * (2 * q)) 256 512 (r50Pre3 2 q (sealW nCls) (sealX q t)) k + 1 := by
  rw [r50Pre4_apply]
  exact sealDnB_eq 2 (2 * (2 * q)) (2 * (2 * q)) 256 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) (margin_p2 q hq) _

theorem pc5 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre5 2 q (sealW nCls) (sealX q t) = fun k => r50Pre4 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre5_apply]
  exact sealIdB_eq 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn4 q nCls t)

theorem pc6 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre6 2 q (sealW nCls) (sealX q t) = fun k => r50Pre5 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre6_apply]
  exact sealIdB_eq 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn5 q nCls t)

theorem pc7 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre7 2 q (sealW nCls) (sealX q t) = fun k => r50Pre6 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre7_apply]
  exact sealIdB_eq 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn6 q nCls t)

theorem pc8 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    r50Pre8 2 q (sealW nCls) (sealX q t)
      = fun k => R34FullBSeal.sealProj 2 (2 * q) (2 * q) 512 1024 (r50Pre7 2 q (sealW nCls) (sealX q t)) k + 1 := by
  rw [r50Pre8_apply]
  exact sealDnB_eq 2 (2 * q) (2 * q) 512 256 1024 (two_sq_pos (2 * q) (by omega)) (margin_p3 q hq) _

theorem pc9 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre9 2 q (sealW nCls) (sealX q t) = fun k => r50Pre8 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre9_apply]
  exact sealIdB_eq 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn8 q nCls t)

theorem pc10 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre10 2 q (sealW nCls) (sealX q t) = fun k => r50Pre9 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre10_apply]
  exact sealIdB_eq 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn9 q nCls t)

theorem pc11 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre11 2 q (sealW nCls) (sealX q t) = fun k => r50Pre10 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre11_apply]
  exact sealIdB_eq 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn10 q nCls t)

theorem pc12 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre12 2 q (sealW nCls) (sealX q t) = fun k => r50Pre11 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre12_apply]
  exact sealIdB_eq 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn11 q nCls t)

theorem pc13 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre13 2 q (sealW nCls) (sealX q t) = fun k => r50Pre12 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre13_apply]
  exact sealIdB_eq 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn12 q nCls t)

theorem pc14 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    r50Pre14 2 q (sealW nCls) (sealX q t)
      = fun k => R34FullBSeal.sealProj 2 q q 1024 2048 (r50Pre13 2 q (sealW nCls) (sealX q t)) k + 1 := by
  rw [r50Pre14_apply]
  exact sealDnB_eq 2 q q 1024 512 2048 (two_sq_pos q (by omega)) (margin_p4 q hq) _

theorem pc15 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre15 2 q (sealW nCls) (sealX q t) = fun k => r50Pre14 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre15_apply]
  exact sealIdB_eq 2 q q 512 2048 (two_sq_pos q (by omega)) _ (nn14 q nCls t)

theorem pc16 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    r50Pre16 2 q (sealW nCls) (sealX q t) = fun k => r50Pre15 2 q (sealW nCls) (sealX q t) k + 1 := by
  rw [r50Pre16_apply]
  exact sealIdB_eq 2 q q 512 2048 (two_sq_pos q (by omega)) _ (nn15 q nCls t)

-- ════════════════════════════════════════════════════════════════
-- § 7. The 48 relu clauses at the witness
-- ════════════════════════════════════════════════════════════════

theorem sc_s1b0 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50ProjSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b0 (r50Pre0 2 q (sealW nCls) (sealX q t)) :=
  seal_pr_smooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) (margin_p1 q hq) _

theorem sc_s1b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b1 (r50Pre1 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) _ (nn1 q nCls t)

theorem sc_s1b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b2 (r50Pre2 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) _ (nn2 q nCls t)

theorem sc_s2b0 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50DownSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b0 (r50Pre3 2 q (sealW nCls) (sealX q t)) :=
  seal_dn_smooth 2 (2 * (2 * q)) (2 * (2 * q)) 256 128 512 (two_sq_pos (2 * (2 * (2 * q))) (by omega))
    (two_sq_pos (2 * (2 * q)) (by omega)) (margin_p2 q hq) _

theorem sc_s2b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b1 (r50Pre4 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn4 q nCls t)

theorem sc_s2b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b2 (r50Pre5 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn5 q nCls t)

theorem sc_s2b3 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b3 (r50Pre6 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn6 q nCls t)

theorem sc_s3b0 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50DownSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b0 (r50Pre7 2 q (sealW nCls) (sealX q t)) :=
  seal_dn_smooth 2 (2 * q) (2 * q) 512 256 1024 (two_sq_pos (2 * (2 * q)) (by omega))
    (two_sq_pos (2 * q) (by omega)) (margin_p3 q hq) _

theorem sc_s3b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b1 (r50Pre8 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn8 q nCls t)

theorem sc_s3b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b2 (r50Pre9 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn9 q nCls t)

theorem sc_s3b3 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b3 (r50Pre10 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn10 q nCls t)

theorem sc_s3b4 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b4 (r50Pre11 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn11 q nCls t)

theorem sc_s3b5 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b5 (r50Pre12 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn12 q nCls t)

theorem sc_s4b0 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50DownSmoothAt 2 q q (sealW nCls).s4b0 (r50Pre13 2 q (sealW nCls) (sealX q t)) :=
  seal_dn_smooth 2 q q 1024 512 2048 (two_sq_pos (2 * q) (by omega))
    (two_sq_pos q (by omega)) (margin_p4 q hq) _

theorem sc_s4b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 q q (sealW nCls).s4b1 (r50Pre14 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 q q 512 2048 (two_sq_pos q (by omega)) _ (nn14 q nCls t)

theorem sc_s4b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 q q (sealW nCls).s4b2 (r50Pre15 2 q (sealW nCls) (sealX q t)) :=
  seal_id_smooth 2 q q 512 2048 (two_sq_pos q (by omega)) _ (nn15 q nCls t)


-- ════════════════════════════════════════════════════════════════
-- § 8. The whole-net VJP at the witness, and differentiability there
-- ════════════════════════════════════════════════════════════════

/-- Every BN `ε` of the witness is `1`. -/
theorem seal_pos (nCls : Nat) : R50PosB (sealW nCls) :=
  ⟨one_pos, seal_pr_pos 64 64 256, seal_id_pos 64 256, seal_id_pos 64 256,
    seal_pr_pos 256 128 512, seal_id_pos 128 512, seal_id_pos 128 512, seal_id_pos 128 512,
    seal_pr_pos 512 256 1024, seal_id_pos 256 1024, seal_id_pos 256 1024, seal_id_pos 256 1024,
    seal_id_pos 256 1024, seal_id_pos 256 1024,
    seal_pr_pos 1024 512 2048, seal_id_pos 512 2048, seal_id_pos 512 2048⟩

/-- The stem clause, the pool's no-tie and all 48 relu clauses at `(sealW nCls, sealX q t)`. -/
theorem seal_smooth (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50SmoothAtB 2 q (sealW nCls) (sealX q t) :=
  ⟨seal_stem_clause q hq nCls t, seal_pool_clause q hq nCls t,
    sc_s1b0 q hq0 hq nCls t, sc_s1b1 q hq0 nCls t, sc_s1b2 q hq0 nCls t, sc_s2b0 q hq0 hq nCls t,
    sc_s2b1 q hq0 nCls t, sc_s2b2 q hq0 nCls t, sc_s2b3 q hq0 nCls t, sc_s3b0 q hq0 hq nCls t,
    sc_s3b1 q hq0 nCls t, sc_s3b2 q hq0 nCls t, sc_s3b3 q hq0 nCls t, sc_s3b4 q hq0 nCls t,
    sc_s3b5 q hq0 nCls t, sc_s4b0 q hq0 hq nCls t, sc_s4b1 q hq0 nCls t, sc_s4b2 q hq0 nCls t⟩

/-- **The whole-net VJP at the witness** — all 48 relu clauses, the stem clause and the
    pool's no-tie discharged at `(sealW nCls, sealX q t)`, on `resnet50ForwardBFull` itself
    (transported through `resnet50ForwardBFull_eq_chain`), at BOTH shipped resolutions. -/
noncomputable def sealVJP (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    HasVJPAt (resnet50ForwardBFull 2 q (sealW nCls)) (sealX q t) := by
  rw [show resnet50ForwardBFull 2 q (sealW nCls)
      = r34HeadB 2 q q (sealW nCls).Wd (sealW nCls).bd ∘ r50Pre16 2 q (sealW nCls)
      from funext (resnet50ForwardBFull_eq_chain 2 q (sealW nCls))]
  exact resnet50ForwardBFullHasVJPAt 2 q hq0 (sealW nCls) (seal_pos nCls) (sealX q t)
    (seal_smooth q hq0 hq nCls t)

/-- The net is differentiable at the witness — `fderiv_ne_zero_of_ray`'s first hypothesis. -/
theorem seal_differentiableAt (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (resnet50ForwardBFull 2 q (sealW nCls)) (sealX q t) :=
  resnet50ForwardBFull_differentiableAt 2 q hq0 (sealW nCls) (seal_pos nCls) (sealX q t)
    (seal_smooth q hq0 hq nCls t)

-- ════════════════════════════════════════════════════════════════
-- § 9. The carrier along the ray
--   FIVE BN sites lie on the carrier's path — the stem and the FOUR projections, one per stage,
--   because stage 1 block 0 projects too (at stride 1). The twelve bottlenecks contribute a
--   batch-uniform `+1` each, which the carrier does not see.
-- ════════════════════════════════════════════════════════════════

/-- The carrier after the stem BN. -/
noncomputable def dS (q : Nat) (t : ℝ) : Fin 64 → ℝ :=
  fun ci => t * bnIstd (2 * ((2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q))))))
    (bnRowLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) (Zs q t) ci) 1

/-- The stage-1 projection's pre-BN activation. -/
noncomputable def Zp1 (q : Nat) (nCls : Nat) (t : ℝ) : Vec (2 * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  StableHLO.batchMap 2 (flatConv (ctK 256 64 1 1 1) (kv 256 0))
    (r50Pre0 2 q (sealW nCls) (sealX q t))

/-- The carrier after it. -/
noncomputable def dP1 (q : Nat) (nCls : Nat) (t : ℝ) : Fin 256 → ℝ :=
  fun ci => dS q t 0 * bnIstd (2 * ((2 * (2 * (2 * q))) * (2 * (2 * (2 * q)))))
    (bnRowLA 2 256 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (Zp1 q nCls t) ci) 1

/-- The stage-2 projection's pre-BN activation. -/
noncomputable def Zp2 (q : Nat) (nCls : Nat) (t : ℝ) : Vec (2 * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  StableHLO.batchMap 2 (flatConvStride2 (ctK 512 256 1 1 1) (kv 512 0))
    (r50Pre3 2 q (sealW nCls) (sealX q t))

/-- The carrier after it. -/
noncomputable def dP2 (q : Nat) (nCls : Nat) (t : ℝ) : Fin 512 → ℝ :=
  fun ci => dP1 q nCls t 0 * bnIstd (2 * ((2 * (2 * q)) * (2 * (2 * q))))
    (bnRowLA 2 512 (2 * (2 * q)) (2 * (2 * q)) (Zp2 q nCls t) ci) 1

/-- The stage-3 projection's pre-BN activation. -/
noncomputable def Zp3 (q : Nat) (nCls : Nat) (t : ℝ) : Vec (2 * (1024 * (2 * q) * (2 * q))) :=
  StableHLO.batchMap 2 (flatConvStride2 (ctK 1024 512 1 1 1) (kv 1024 0))
    (r50Pre7 2 q (sealW nCls) (sealX q t))

/-- The carrier after it. -/
noncomputable def dP3 (q : Nat) (nCls : Nat) (t : ℝ) : Fin 1024 → ℝ :=
  fun ci => dP2 q nCls t 0 * bnIstd (2 * ((2 * q) * (2 * q))) (bnRowLA 2 1024 (2 * q) (2 * q) (Zp3 q nCls t) ci) 1

/-- The stage-4 projection's pre-BN activation. -/
noncomputable def Zp4 (q : Nat) (nCls : Nat) (t : ℝ) : Vec (2 * (2048 * q * q)) :=
  StableHLO.batchMap 2 (flatConvStride2 (ctK 2048 1024 1 1 1) (kv 2048 0))
    (r50Pre13 2 q (sealW nCls) (sealX q t))

/-- The carrier after it. -/
noncomputable def dP4 (q : Nat) (nCls : Nat) (t : ℝ) : Fin 2048 → ℝ :=
  fun ci => dP3 q nCls t 0 * bnIstd (2 * (q * q)) (bnRowLA 2 2048 q q (Zp4 q nCls t) ci) 1

theorem ed0 (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dS q t) (r50Pre0 2 q (sealW nCls) (sealX q t)) := by
  rw [pc0 q hq nCls t]
  refine eDiff_pool 64 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (dS q t) _ ?_
  refine eDiff_bn 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 1 (kv 64 1) (kv 64 160) (fun _ => t) (dS q t) (Zs q t) ?_ ?_
  · rw [Zs, ctConv]
    exact eDiff_convS2 (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q)))))
      (kH := 7) (kW := 7) (0 : Fin 3) rfl (by norm_num) (by norm_num) 1
      (kv 64 0) _ (fun _ => t) _ (eDiff_rayX _ _ t) (fun o => by norm_num)
  · intro ci
    simp only [dS, kv_apply]
    ring

theorem ed1 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP1 q nCls t) (r50Pre1 2 q (sealW nCls) (sealX q t)) := by
  rw [pc1 q hq0 hq nCls t]
  refine eDiff_shift _ _ 1 ?_
  rw [sealProj1_apply]
  exact eDiff_convBn (kH := 1) (kW := 1) (0 : Fin 64) rfl (by norm_num) (by norm_num) 1 160
    (Zp1 q nCls t) (ed0 q hq nCls t) rfl (fun ci => by simp only [dP1]; ring)

theorem ed2 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP1 q nCls t) (r50Pre2 2 q (sealW nCls) (sealX q t)) := by
  rw [pc2 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed1 q hq0 hq nCls t)

theorem ed3 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP1 q nCls t) (r50Pre3 2 q (sealW nCls) (sealX q t)) := by
  rw [pc3 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed2 q hq0 hq nCls t)

theorem ed4 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre4 2 q (sealW nCls) (sealX q t)) := by
  rw [pc4 q hq0 hq nCls t]
  refine eDiff_shift _ _ 1 ?_
  rw [R34FullBSeal.sealProj_apply]
  exact eDiff_convS2Bn (h := (2 * (2 * q))) (w := (2 * (2 * q))) (kH := 1) (kW := 1) (0 : Fin 256)
    rfl (by norm_num) (by norm_num) 1 160 (Zp2 q nCls t) (ed3 q hq0 hq nCls t) rfl
    (fun ci => by simp only [dP2]; ring)

theorem ed5 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre5 2 q (sealW nCls) (sealX q t)) := by
  rw [pc5 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed4 q hq0 hq nCls t)

theorem ed6 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre6 2 q (sealW nCls) (sealX q t)) := by
  rw [pc6 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed5 q hq0 hq nCls t)

theorem ed7 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre7 2 q (sealW nCls) (sealX q t)) := by
  rw [pc7 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed6 q hq0 hq nCls t)

theorem ed8 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre8 2 q (sealW nCls) (sealX q t)) := by
  rw [pc8 q hq0 hq nCls t]
  refine eDiff_shift _ _ 1 ?_
  rw [R34FullBSeal.sealProj_apply]
  exact eDiff_convS2Bn (h := (2 * q)) (w := (2 * q)) (0 : Fin 512) rfl (by norm_num) (by norm_num) 1
    160 (Zp3 q nCls t) (ed7 q hq0 hq nCls t) rfl (fun ci => by simp only [dP3]; ring)

theorem ed9 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre9 2 q (sealW nCls) (sealX q t)) := by
  rw [pc9 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed8 q hq0 hq nCls t)

theorem ed10 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre10 2 q (sealW nCls) (sealX q t)) := by
  rw [pc10 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed9 q hq0 hq nCls t)

theorem ed11 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre11 2 q (sealW nCls) (sealX q t)) := by
  rw [pc11 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed10 q hq0 hq nCls t)

theorem ed12 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre12 2 q (sealW nCls) (sealX q t)) := by
  rw [pc12 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed11 q hq0 hq nCls t)

theorem ed13 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre13 2 q (sealW nCls) (sealX q t)) := by
  rw [pc13 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed12 q hq0 hq nCls t)

theorem ed14 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP4 q nCls t) (r50Pre14 2 q (sealW nCls) (sealX q t)) := by
  rw [pc14 q hq0 hq nCls t]
  refine eDiff_shift _ _ 1 ?_
  rw [R34FullBSeal.sealProj_apply]
  exact eDiff_convS2Bn (h := q) (w := q) (kH := 1) (kW := 1) (0 : Fin 1024) rfl (by norm_num)
    (by norm_num) 1 160 (Zp4 q nCls t) (ed13 q hq0 hq nCls t) rfl
    (fun ci => by simp only [dP4]; ring)

theorem ed15 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP4 q nCls t) (r50Pre15 2 q (sealW nCls) (sealX q t)) := by
  rw [pc15 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed14 q hq0 hq nCls t)

theorem ed16 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP4 q nCls t) (r50Pre16 2 q (sealW nCls) (sealX q t)) := by
  rw [pc16 q hq0 nCls t]
  exact eDiff_shift _ _ 1 (ed15 q hq0 hq nCls t)


-- ════════════════════════════════════════════════════════════════
-- § 10. The head, and the output difference along the ray
-- ════════════════════════════════════════════════════════════════

/-- The head reads channel 0 into class 0 — `BatchSeal.head_diff_ct` at this net's widths. -/
theorem head_diff (q : Nat) (hq0 : 0 < q) (nCls : Nat) (hn : 0 < nCls)
    (v : Vec (2 * (2048 * q * q))) (δ : Fin 2048 → ℝ) (hv : EDiff δ v) :
    r34HeadB 2 q q (sealW nCls).Wd (sealW nCls).bd v (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - r34HeadB 2 q q (sealW nCls).Wd (sealW nCls).bd v (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = δ 0 :=
  head_diff_ct hq0 hq0 (0 : Fin 2048) rfl ⟨0, hn⟩ _ _
    (fun ci => by rw [sealW_Wd]; simp) rfl v δ hv

/-- **The positive, continuous nonlinear factor**: one `istd` per BN on the carrier's path —
    the stem's and the FOUR projections'. -/
noncomputable def Rr (q : Nat) (nCls : Nat) (t : ℝ) : ℝ :=
  bnIstd (2 * ((2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q))))))
    (bnRowLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) (Zs q t) 0) 1
  * (bnIstd (2 * ((2 * (2 * (2 * q))) * (2 * (2 * (2 * q)))))
    (bnRowLA 2 256 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (Zp1 q nCls t) 0) 1
     * (bnIstd (2 * ((2 * (2 * q)) * (2 * (2 * q)))) (bnRowLA 2 512 (2 * (2 * q)) (2 * (2 * q)) (Zp2 q nCls t) 0) 1
        * (bnIstd (2 * ((2 * q) * (2 * q))) (bnRowLA 2 1024 (2 * q) (2 * q) (Zp3 q nCls t) 0) 1
           * bnIstd (2 * (q * q)) (bnRowLA 2 2048 q q (Zp4 q nCls t) 0) 1)))

theorem Rr_pos (q : Nat) (nCls : Nat) (t : ℝ) : 0 < Rr q nCls t :=
  mul_pos (bnIstd_pos _ 1 one_pos)
    (mul_pos (bnIstd_pos _ 1 one_pos)
      (mul_pos (bnIstd_pos _ 1 one_pos)
        (mul_pos (bnIstd_pos _ 1 one_pos) (bnIstd_pos _ 1 one_pos))))

/-- **The class-0 difference between the two examples, along the ray, is `t · R t`.** -/
theorem gd_ray (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (hn : 0 < nCls) (t : ℝ) :
    resnet50ForwardBFull 2 q (sealW nCls) (sealX q t)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - resnet50ForwardBFull 2 q (sealW nCls) (sealX q t)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = t * Rr q nCls t := by
  rw [resnet50ForwardBFull_eq_chain, Function.comp_apply,
    head_diff q hq0 nCls hn _ (dP4 q nCls t) (ed16 q hq0 hq nCls t)]
  simp only [dP4, dP3, dP2, dP1, dS, Rr]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 11. `R` is continuous
-- ════════════════════════════════════════════════════════════════

theorem sealX_continuous (q : Nat) : Continuous (sealX q) := rayX_continuous _ _

/-- `R` is continuous: every bottleneck is, and `fun_prop` composes the thirteen-block prefix once
    the chain is unfolded to its atoms. Every BN on the witness has `ε = 1`. -/
theorem Rr_continuous (q : Nat) (_hq0 : 0 < q) (nCls : Nat) : Continuous (Rr q nCls) := by
  unfold Rr Zs Zp1 Zp2 Zp3 Zp4 ctConv sealX r50Pre13 r50Pre12 r50Pre11 r50Pre10 r50Pre9 r50Pre8
    r50Pre7 r50Pre6 r50Pre5 r50Pre4 r50Pre3 r50Pre2 r50Pre1 r50Pre0
  unfold r50IdB r50ProjB r50DownB r34StemB projB StableHLO.cbReluB StableHLO.cbReluStridedB
    StableHLO.projStridedB
  fun_prop (disch := exact one_pos)

-- ════════════════════════════════════════════════════════════════
-- § 12. The seal
-- ════════════════════════════════════════════════════════════════

/-- **Level 2 — the witness is non-degenerate**: the full-width batch-BN ResNet-50 at the
    structural weights is NOT constant in its input, at either shipped resolution. -/
theorem sealX_nonconstant (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (hn : 0 < nCls) :
    resnet50ForwardBFull 2 q (sealW nCls) (sealX q 1)
      ≠ resnet50ForwardBFull 2 q (sealW nCls) (sealX q 0) :=
  ne_of_ray_readout _ (sealX q) _ _ (gd_ray q hq0 hq nCls hn) (by simpa using (Rr_pos q nCls 1).ne')

/-- **Level 3 — the whole-net Jacobian is nonzero at the witness.** -/
theorem sealX_jacobian_nonzero (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat)
    (hn : 0 < nCls) :
    fderiv ℝ (resnet50ForwardBFull 2 q (sealW nCls)) (sealX q 0) ≠ 0 :=
  fderiv_ne_zero_of_ray_readout _ (sealX q) (sealV q) (sealX_zero_add q) _ _
    (gd_ray q hq0 hq nCls hn) (seal_differentiableAt q hq0 hq nCls 0) (Rr_pos q nCls 0).ne'
    (hasDerivAt_mul_self_zero (Rr_continuous q hq0 nCls).continuousAt)

/-- **The seal**: the proven whole-network backward of the **full-width, batch-BatchNorm,
    [3,4,6,3]-bottleneck** ResNet-50 — `resnet50ForwardBFull`, at BOTH shipped resolutions —
    is **not the zero map** at the witness. ResNet-50's clause bundle (48 relu clauses, the stem
    and the pool) had no exhibited point at all before this. -/
theorem sealX_backward_nontrivial (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat)
    (hn : 0 < nCls) :
    ∃ (j₀ : Fin (2 * nCls)) (i₀ : Fin (2 * (3 * (2 * (2 * (2 * (2 * (2 * q))))) *
        (2 * (2 * (2 * (2 * (2 * q)))))))),
      (sealVJP q hq0 hq nCls 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (sealVJP q hq0 hq nCls 0).backward_nontrivial_of_fderiv_ne
    (sealX_jacobian_nonzero q hq0 hq nCls hn)

end R50FullBSeal
end Proofs
