import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBSeal

/-!
# ResNet-50's non-degeneracy seal, on the full-width batched net (levels 2 and 3)

`planning/full_width_seals.md` §4.2. `ResNet50FullBVJP.lean` proves
`resnet50ForwardB_full_has_vjp_at` under 48 relu clauses — three per bottleneck — plus the stem's
and the stem pool's. ResNet-50 had **no** witness at all: unlike ResNet-34 and MobileNetV2 it never
had a 2-channel proxy, so the clause bundle's joint satisfiability was never exhibited. This file
exhibits it, and a nonzero Jacobian with it, on `resnet50ForwardB_full` itself.

## What is inherited and what is new

⭐ Almost everything is [`Training/BatchSealKit.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Training/BatchSealKit.lean)'s or ResNet-34's. The carrier (`EDiff` and its
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
  stride-1 `EDiff_conv`. ⛔ That also makes the carrier's BN count **five**, not the four of
  ResNet-34: stem, `s1b0`, `s2b0`, `s3b0`, `s4b0`;
* **`q` stays a binder.** ResNet-50 ships at two resolutions and the tier is stated at both, so the
  seal is too: `0 < q` and `q ≤ 7` are all the witness needs, and `q = 7` (224 px) and `q = 5`
  (160 px, where the 76.66% run lives) are instances of one theorem. The bound is the `β = 160`
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
theorem marginQ (n q : Nat) (hq : q ≤ 7) (hn : n ≤ 512 * (q * q)) :
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
theorem sealIdBody (N h w mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (oc * h * w))) :
    (projB N (h := h) (w := w) (sealIdW mid oc).W₃ (sealIdW mid oc).b₃ (sealIdW mid oc).ε₃
        (sealIdW mid oc).γ₃ (sealIdW mid oc).β₃ ∘
      StableHLO.cbReluB N (h := h) (w := w) (sealIdW mid oc).W₂ (sealIdW mid oc).b₂
          (sealIdW mid oc).ε₂ (sealIdW mid oc).γ₂ (sealIdW mid oc).β₂ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealIdW mid oc).W₁ (sealIdW mid oc).b₁
          (sealIdW mid oc).ε₁ (sealIdW mid oc).γ₁ (sealIdW mid oc).β₁) v = fun _ => (1 : ℝ) :=
  R34FullBSeal.projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1
    (fun _ => rfl) _

/-- The stride-1 projection block's body is the constant `1`. -/
theorem sealPrBody (N h w ic mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (ic * h * w))) :
    (projB N (h := h) (w := w) (sealPrW ic mid oc).W₃ (sealPrW ic mid oc).b₃
        (sealPrW ic mid oc).ε₃ (sealPrW ic mid oc).γ₃ (sealPrW ic mid oc).β₃ ∘
      StableHLO.cbReluB N (h := h) (w := w) (sealPrW ic mid oc).W₂ (sealPrW ic mid oc).b₂
          (sealPrW ic mid oc).ε₂ (sealPrW ic mid oc).γ₂ (sealPrW ic mid oc).β₂ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealPrW ic mid oc).W₁ (sealPrW ic mid oc).b₁
          (sealPrW ic mid oc).ε₁ (sealPrW ic mid oc).γ₁ (sealPrW ic mid oc).β₁) v
      = fun _ => (1 : ℝ) :=
  R34FullBSeal.projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1
    (fun _ => rfl) _

/-- The strided projection block's body is the constant `1`. ⚠ v1.5: its middle stage is the
    STRIDED conv-bn-relu, so this is not `sealPrBody` at other shapes. -/
theorem sealDnBody (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
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

/-- ⭐ **The structural bottleneck is the shift `a ↦ a + 1`** on a nonnegative activation. -/
theorem sealIdB_eq (N h w mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (oc * h * w)))
    (hv : ∀ k, 0 ≤ v k) :
    r50IdB N h w (sealIdW mid oc) v = fun k => v k + 1 := by
  have hbody := sealIdBody N h w mid oc hn v
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

/-- ⭐ **The structural stride-1 projection block is its projection plus one.** -/
theorem sealPrB_eq (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160) (v : Vec (N * (ic * h * w))) :
    r50ProjB N h w (sealPrW ic mid oc) v = fun k => sealProj1 N h w ic oc v k + 1 := by
  have hbody := sealPrBody N h w ic mid oc hn v
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

/-- ⭐ **The structural strided projection block is ResNet-34's strided projection plus one.** -/
theorem sealDnB_eq (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    r50DownB N h w (sealPrW ic mid oc) v
      = fun k => R34FullBSeal.sealProj N h w ic oc v k + 1 := by
  have hbody := sealDnBody N h w ic mid oc hn v
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

theorem sealIdPos (mid oc : Nat) : R50IdPos (sealIdW mid oc) := ⟨one_pos, one_pos, one_pos⟩

theorem sealPrPos (ic mid oc : Nat) : R50ProjPos (sealPrW ic mid oc) :=
  ⟨one_pos, one_pos, one_pos, one_pos⟩

/-- ⭐ **The bottleneck's three relu clauses.** Both interior ones see a constant channel (their
    convs are zeroed), so they are `β = 1 ≠ 0` and weight-only; only the post-residual one needs
    the activation, and only through `0 ≤ ·`. -/
theorem sealIdSmooth (N h w mid oc : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (oc * h * w)))
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
    rw [congrFun (sealIdBody N h w mid oc hn v) k]
    intro hc
    linarith [hv k]

/-- The stride-1 projection block's three relu clauses — all weight-only. -/
theorem sealPrSmooth (N h w ic mid oc : Nat) (hn : 0 < N * (h * w))
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
    rw [congrFun (sealPrBody N h w ic mid oc hn v) k]
    intro hc
    linarith [sealProj1_pos N h w ic oc hm v k]

/-- The strided projection block's three relu clauses — all weight-only. ⚠ v1.5: `hm1` is at the
    INPUT resolution and only `hm2` is at the halved one. -/
theorem sealDnSmooth (N h w ic mid oc : Nat) (hn2 : 0 < N * ((2 * h) * (2 * w)))
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
    rw [congrFun (sealDnBody N h w ic mid oc hn v) k]
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

theorem EDiff_sealX (q : Nat) (t : ℝ) :
    EDiff (fun ci => if ci.val = 0 then t else 0) (sealX q t) := by
  rw [sealX]
  exact EDiff_rayX _ _ t

/-- The stem's centre-tap conv output — the carrier's first stop. -/
noncomputable def Zs (q : Nat) (t : ℝ) : Vec (2 * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) :=
  ctConv 64 7 7 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) t

/-- `2·a² ≤ 512·q²` whenever `a ≤ 16q` — every BN width against the stem's. -/
theorem bnd (a q : Nat) (ha : a ≤ 16 * q) : 2 * (a * a) ≤ 512 * (q * q) := by
  calc 2 * (a * a) ≤ 2 * ((16 * q) * (16 * q)) :=
        Nat.mul_le_mul_left _ (Nat.mul_le_mul ha ha)
    _ = 512 * (q * q) := by ring

/-- The `β = 160` margin at the stem's `2·(16q)²`. -/
theorem marginStem (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q))))) : ℕ) : ℝ) < 160 :=
  marginQ _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 1's `2·(8q)²`. -/
theorem marginP1 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * (2 * (2 * q))) * (2 * (2 * (2 * q)))) : ℕ) : ℝ) < 160 :=
  marginQ _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 2's `2·(4q)²`. -/
theorem marginP2 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * (2 * q)) * (2 * (2 * q))) : ℕ) : ℝ) < 160 :=
  marginQ _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 3's `2·(2q)²`. -/
theorem marginP3 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * ((2 * q) * (2 * q)) : ℕ) : ℝ) < 160 :=
  marginQ _ q hq (bnd _ q (by omega))

/-- The `β = 160` margin at stage 4's `2·q²`. -/
theorem marginP4 (q : Nat) (hq : q ≤ 7) :
    |(1 : ℝ)| * Real.sqrt ((2 * (q * q) : ℕ) : ℝ) < 160 :=
  marginQ _ q hq (bnd _ q (by omega))

/-- The stem BN is strictly positive at every point of the ray. -/
theorem Zs_bn_pos (q : Nat) (hq : q ≤ 7) (t : ℝ) (k : Fin (2 * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q))))))) :
    0 < StableHLO.bnBatchLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 1 (kv 64 1) (kv 64 160) (Zs q t) k := by
  rw [Zs]
  exact ctConv_bn_pos 64 7 7 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (marginStem q hq) t k

/-- The stem's relu is off at the witness, so the pool's no-tie can be read on the BN output. -/
theorem stem_relu_off (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    StableHLO.cbReluStridedB 2 (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q)))))
      (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ (sealX q t)
      = StableHLO.bnBatchLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 1 (kv 64 1) (kv 64 160) (Zs q t) :=
  R34FullBSeal.cbReluStridedB_eq _ _ _ _ _ (sealX q t) (fun k => Zs_bn_pos q hq t k)

theorem sealStemClause (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R34StemSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
      (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ (sealX q t) :=
  R34FullBSeal.sealStemSmooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 3 64 _ _ (marginStem q hq) (sealX q t)

theorem sealPoolClause (q : Nat) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
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
  exact sealPrB_eq 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) (marginP1 q hq) _

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
  exact sealDnB_eq 2 (2 * (2 * q)) (2 * (2 * q)) 256 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) (marginP2 q hq) _

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
  exact sealDnB_eq 2 (2 * q) (2 * q) 512 256 1024 (two_sq_pos (2 * q) (by omega)) (marginP3 q hq) _

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
  exact sealDnB_eq 2 q q 1024 512 2048 (two_sq_pos q (by omega)) (marginP4 q hq) _

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
  sealPrSmooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) (marginP1 q hq) _

theorem sc_s1b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b1 (r50Pre1 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) _ (nn1 q nCls t)

theorem sc_s1b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b2 (r50Pre2 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (two_sq_pos (2 * (2 * (2 * q))) (by omega)) _ (nn2 q nCls t)

theorem sc_s2b0 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50DownSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b0 (r50Pre3 2 q (sealW nCls) (sealX q t)) :=
  sealDnSmooth 2 (2 * (2 * q)) (2 * (2 * q)) 256 128 512 (two_sq_pos (2 * (2 * (2 * q))) (by omega))
    (two_sq_pos (2 * (2 * q)) (by omega)) (marginP2 q hq) _

theorem sc_s2b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b1 (r50Pre4 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn4 q nCls t)

theorem sc_s2b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b2 (r50Pre5 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn5 q nCls t)

theorem sc_s2b3 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b3 (r50Pre6 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (two_sq_pos (2 * (2 * q)) (by omega)) _ (nn6 q nCls t)

theorem sc_s3b0 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50DownSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b0 (r50Pre7 2 q (sealW nCls) (sealX q t)) :=
  sealDnSmooth 2 (2 * q) (2 * q) 512 256 1024 (two_sq_pos (2 * (2 * q)) (by omega))
    (two_sq_pos (2 * q) (by omega)) (marginP3 q hq) _

theorem sc_s3b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b1 (r50Pre8 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn8 q nCls t)

theorem sc_s3b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b2 (r50Pre9 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn9 q nCls t)

theorem sc_s3b3 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b3 (r50Pre10 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn10 q nCls t)

theorem sc_s3b4 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b4 (r50Pre11 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn11 q nCls t)

theorem sc_s3b5 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 (2 * q) (2 * q) (sealW nCls).s3b5 (r50Pre12 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 (2 * q) (2 * q) 256 1024 (two_sq_pos (2 * q) (by omega)) _ (nn12 q nCls t)

theorem sc_s4b0 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    R50DownSmoothAt 2 q q (sealW nCls).s4b0 (r50Pre13 2 q (sealW nCls) (sealX q t)) :=
  sealDnSmooth 2 q q 1024 512 2048 (two_sq_pos (2 * q) (by omega))
    (two_sq_pos q (by omega)) (marginP4 q hq) _

theorem sc_s4b1 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 q q (sealW nCls).s4b1 (r50Pre14 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 q q 512 2048 (two_sq_pos q (by omega)) _ (nn14 q nCls t)

theorem sc_s4b2 (q : Nat) (hq0 : 0 < q) (nCls : Nat) (t : ℝ) :
    R50IdSmoothAt 2 q q (sealW nCls).s4b2 (r50Pre15 2 q (sealW nCls) (sealX q t)) :=
  sealIdSmooth 2 q q 512 2048 (two_sq_pos q (by omega)) _ (nn15 q nCls t)


-- ════════════════════════════════════════════════════════════════
-- § 8. The whole-net VJP at the witness, and differentiability there
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **The whole-net VJP at the witness** — all 48 relu clauses, the stem clause and the
    pool's no-tie discharged at `(sealW nCls, sealX q t)`, on `resnet50ForwardB_full` itself
    (transported through `resnet50ForwardB_full_eq_chain`), at BOTH shipped resolutions. -/
noncomputable def sealVJP (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    HasVJPAt (resnet50ForwardB_full 2 q (sealW nCls)) (sealX q t) := by
  rw [show resnet50ForwardB_full 2 q (sealW nCls)
      = r34HeadB 2 q q (sealW nCls).Wd (sealW nCls).bd ∘ r50Pre16 2 q (sealW nCls)
      from funext (resnet50ForwardB_full_eq_chain 2 q (sealW nCls))]
  exact resnet50ForwardB_full_has_vjp_at 2 q hq0 (sealW nCls) one_pos
    (sealPrPos 64 64 256) (sealIdPos 64 256) (sealIdPos 64 256)
    (sealPrPos 256 128 512) (sealIdPos 128 512) (sealIdPos 128 512) (sealIdPos 128 512)
    (sealPrPos 512 256 1024) (sealIdPos 256 1024) (sealIdPos 256 1024) (sealIdPos 256 1024)
    (sealIdPos 256 1024) (sealIdPos 256 1024)
    (sealPrPos 1024 512 2048) (sealIdPos 512 2048) (sealIdPos 512 2048)
    (sealX q t) (sealStemClause q hq nCls t) (sealPoolClause q hq nCls t)
    (sc_s1b0 q hq0 hq nCls t)
    (sc_s1b1 q hq0 nCls t)
    (sc_s1b2 q hq0 nCls t)
    (sc_s2b0 q hq0 hq nCls t)
    (sc_s2b1 q hq0 nCls t)
    (sc_s2b2 q hq0 nCls t)
    (sc_s2b3 q hq0 nCls t)
    (sc_s3b0 q hq0 hq nCls t)
    (sc_s3b1 q hq0 nCls t)
    (sc_s3b2 q hq0 nCls t)
    (sc_s3b3 q hq0 nCls t)
    (sc_s3b4 q hq0 nCls t)
    (sc_s3b5 q hq0 nCls t)
    (sc_s4b0 q hq0 hq nCls t)
    (sc_s4b1 q hq0 nCls t)
    (sc_s4b2 q hq0 nCls t)

/-- The net is differentiable at the witness — `fderiv_ne_zero_of_ray`'s first hypothesis. -/
theorem sealDiffAt (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (resnet50ForwardB_full 2 q (sealW nCls)) (sealX q t) := by
  rw [show resnet50ForwardB_full 2 q (sealW nCls)
      = r34HeadB 2 q q (sealW nCls).Wd (sealW nCls).bd ∘ r50Pre16 2 q (sealW nCls)
      from funext (resnet50ForwardB_full_eq_chain 2 q (sealW nCls))]
  have f0 : DifferentiableAt ℝ (r50Pre0 2 q (sealW nCls)) (sealX q t) :=
    r34StemB_differentiableAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) _ _ _ one_pos _ _ (by norm_num) (by omega)
      (by omega) (sealX q t) (sealStemClause q hq nCls t) (sealPoolClause q hq nCls t)
  have f1 : DifferentiableAt ℝ (r50Pre1 2 q (sealW nCls)) (sealX q t) :=
    (r50ProjB_differentiableAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b0 (sealPrPos 64 64 256) _
      (sc_s1b0 q hq0 hq nCls t)).comp (sealX q t) f0
  have f2 : DifferentiableAt ℝ (r50Pre2 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b1 (sealIdPos 64 256) _
      (sc_s1b1 q hq0 nCls t)).comp (sealX q t) f1
  have f3 : DifferentiableAt ℝ (r50Pre3 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (sealW nCls).s1b2 (sealIdPos 64 256) _
      (sc_s1b2 q hq0 nCls t)).comp (sealX q t) f2
  have f4 : DifferentiableAt ℝ (r50Pre4 2 q (sealW nCls)) (sealX q t) :=
    (r50DownB_differentiableAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b0 (sealPrPos 256 128 512) _
      (sc_s2b0 q hq0 hq nCls t)).comp (sealX q t) f3
  have f5 : DifferentiableAt ℝ (r50Pre5 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b1 (sealIdPos 128 512) _
      (sc_s2b1 q hq0 nCls t)).comp (sealX q t) f4
  have f6 : DifferentiableAt ℝ (r50Pre6 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b2 (sealIdPos 128 512) _
      (sc_s2b2 q hq0 nCls t)).comp (sealX q t) f5
  have f7 : DifferentiableAt ℝ (r50Pre7 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * (2 * q)) (2 * (2 * q)) (sealW nCls).s2b3 (sealIdPos 128 512) _
      (sc_s2b3 q hq0 nCls t)).comp (sealX q t) f6
  have f8 : DifferentiableAt ℝ (r50Pre8 2 q (sealW nCls)) (sealX q t) :=
    (r50DownB_differentiableAt 2 (2 * q) (2 * q) (sealW nCls).s3b0 (sealPrPos 512 256 1024) _
      (sc_s3b0 q hq0 hq nCls t)).comp (sealX q t) f7
  have f9 : DifferentiableAt ℝ (r50Pre9 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * q) (2 * q) (sealW nCls).s3b1 (sealIdPos 256 1024) _
      (sc_s3b1 q hq0 nCls t)).comp (sealX q t) f8
  have f10 : DifferentiableAt ℝ (r50Pre10 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * q) (2 * q) (sealW nCls).s3b2 (sealIdPos 256 1024) _
      (sc_s3b2 q hq0 nCls t)).comp (sealX q t) f9
  have f11 : DifferentiableAt ℝ (r50Pre11 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * q) (2 * q) (sealW nCls).s3b3 (sealIdPos 256 1024) _
      (sc_s3b3 q hq0 nCls t)).comp (sealX q t) f10
  have f12 : DifferentiableAt ℝ (r50Pre12 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * q) (2 * q) (sealW nCls).s3b4 (sealIdPos 256 1024) _
      (sc_s3b4 q hq0 nCls t)).comp (sealX q t) f11
  have f13 : DifferentiableAt ℝ (r50Pre13 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 (2 * q) (2 * q) (sealW nCls).s3b5 (sealIdPos 256 1024) _
      (sc_s3b5 q hq0 nCls t)).comp (sealX q t) f12
  have f14 : DifferentiableAt ℝ (r50Pre14 2 q (sealW nCls)) (sealX q t) :=
    (r50DownB_differentiableAt 2 q q (sealW nCls).s4b0 (sealPrPos 1024 512 2048) _
      (sc_s4b0 q hq0 hq nCls t)).comp (sealX q t) f13
  have f15 : DifferentiableAt ℝ (r50Pre15 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 q q (sealW nCls).s4b1 (sealIdPos 512 2048) _
      (sc_s4b1 q hq0 nCls t)).comp (sealX q t) f14
  have f16 : DifferentiableAt ℝ (r50Pre16 2 q (sealW nCls)) (sealX q t) :=
    (r50IdB_differentiableAt 2 q q (sealW nCls).s4b2 (sealIdPos 512 2048) _
      (sc_s4b2 q hq0 nCls t)).comp (sealX q t) f15
  exact (r34HeadB_differentiable 2 q q (sealW nCls).Wd (sealW nCls).bd _).comp (sealX q t) f16

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
  refine EDiff_pool 64 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (dS q t) _ ?_
  refine EDiff_bn 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 1 (kv 64 1) (kv 64 160) (fun _ => t) (dS q t) (Zs q t) ?_ ?_
  · rw [Zs, ctConv]
    exact EDiff_convS2 (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q)))))
      (kH := 7) (kW := 7) (0 : Fin 3) rfl (by norm_num) (by norm_num) 1
      (kv 64 0) _ (fun _ => t) _ (EDiff_rayX _ _ t) (fun o => by norm_num)
  · intro ci
    simp only [dS, kv_apply]
    ring

theorem ed1 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP1 q nCls t) (r50Pre1 2 q (sealW nCls) (sealX q t)) := by
  rw [pc1 q hq0 hq nCls t]
  refine EDiff_shift _ _ 1 ?_
  rw [sealProj1_apply]
  refine EDiff_bn 256 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 1 (kv 256 1) (kv 256 160) (fun _ => 1 * dS q t 0) (dP1 q nCls t)
    (Zp1 q nCls t) ?_ ?_
  · exact EDiff_conv (kH := 1) (kW := 1) (0 : Fin 64) rfl (by norm_num) (by norm_num) 1 (kv 256 0)
      (dS q t) _ (r50Pre0 2 q (sealW nCls) (sealX q t)) (ed0 q hq nCls t) (fun o => rfl)
  · intro ci
    simp only [dP1, kv_apply]
    ring

theorem ed2 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP1 q nCls t) (r50Pre2 2 q (sealW nCls) (sealX q t)) := by
  rw [pc2 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed1 q hq0 hq nCls t)

theorem ed3 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP1 q nCls t) (r50Pre3 2 q (sealW nCls) (sealX q t)) := by
  rw [pc3 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed2 q hq0 hq nCls t)

theorem ed4 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre4 2 q (sealW nCls) (sealX q t)) := by
  rw [pc4 q hq0 hq nCls t]
  refine EDiff_shift _ _ 1 ?_
  rw [R34FullBSeal.sealProj_apply]
  refine EDiff_bn 512 (2 * (2 * q)) (2 * (2 * q)) 1 (kv 512 1) (kv 512 160) (fun _ => 1 * dP1 q nCls t 0) (dP2 q nCls t)
    (Zp2 q nCls t) ?_ ?_
  · exact EDiff_convS2 (h := (2 * (2 * q))) (w := (2 * (2 * q))) (kH := 1) (kW := 1) (0 : Fin 256) rfl (by norm_num) (by norm_num) 1
      (kv 512 0)
      (dP1 q nCls t) _ (r50Pre3 2 q (sealW nCls) (sealX q t)) (ed3 q hq0 hq nCls t) (fun o => rfl)
  · intro ci
    simp only [dP2, kv_apply]
    ring

theorem ed5 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre5 2 q (sealW nCls) (sealX q t)) := by
  rw [pc5 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed4 q hq0 hq nCls t)

theorem ed6 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre6 2 q (sealW nCls) (sealX q t)) := by
  rw [pc6 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed5 q hq0 hq nCls t)

theorem ed7 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP2 q nCls t) (r50Pre7 2 q (sealW nCls) (sealX q t)) := by
  rw [pc7 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed6 q hq0 hq nCls t)

theorem ed8 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre8 2 q (sealW nCls) (sealX q t)) := by
  rw [pc8 q hq0 hq nCls t]
  refine EDiff_shift _ _ 1 ?_
  rw [R34FullBSeal.sealProj_apply]
  refine EDiff_bn 1024 (2 * q) (2 * q) 1 (kv 1024 1) (kv 1024 160) (fun _ => 1 * dP2 q nCls t 0) (dP3 q nCls t)
    (Zp3 q nCls t) ?_ ?_
  · exact EDiff_convS2 (h := (2 * q)) (w := (2 * q)) (0 : Fin 512) rfl (by norm_num) (by norm_num) 1
      (kv 1024 0)
      (dP2 q nCls t) _ (r50Pre7 2 q (sealW nCls) (sealX q t)) (ed7 q hq0 hq nCls t) (fun o => rfl)
  · intro ci
    simp only [dP3, kv_apply]
    ring

theorem ed9 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre9 2 q (sealW nCls) (sealX q t)) := by
  rw [pc9 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed8 q hq0 hq nCls t)

theorem ed10 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre10 2 q (sealW nCls) (sealX q t)) := by
  rw [pc10 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed9 q hq0 hq nCls t)

theorem ed11 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre11 2 q (sealW nCls) (sealX q t)) := by
  rw [pc11 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed10 q hq0 hq nCls t)

theorem ed12 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre12 2 q (sealW nCls) (sealX q t)) := by
  rw [pc12 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed11 q hq0 hq nCls t)

theorem ed13 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP3 q nCls t) (r50Pre13 2 q (sealW nCls) (sealX q t)) := by
  rw [pc13 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed12 q hq0 hq nCls t)

theorem ed14 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP4 q nCls t) (r50Pre14 2 q (sealW nCls) (sealX q t)) := by
  rw [pc14 q hq0 hq nCls t]
  refine EDiff_shift _ _ 1 ?_
  rw [R34FullBSeal.sealProj_apply]
  refine EDiff_bn 2048 q q 1 (kv 2048 1) (kv 2048 160) (fun _ => 1 * dP3 q nCls t 0) (dP4 q nCls t)
    (Zp4 q nCls t) ?_ ?_
  · exact EDiff_convS2 (h := q) (w := q) (kH := 1) (kW := 1) (0 : Fin 1024) rfl (by norm_num) (by norm_num) 1
      (kv 2048 0)
      (dP3 q nCls t) _ (r50Pre13 2 q (sealW nCls) (sealX q t)) (ed13 q hq0 hq nCls t) (fun o => rfl)
  · intro ci
    simp only [dP4, kv_apply]
    ring

theorem ed15 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP4 q nCls t) (r50Pre15 2 q (sealW nCls) (sealX q t)) := by
  rw [pc15 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed14 q hq0 hq nCls t)

theorem ed16 (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (t : ℝ) :
    EDiff (dP4 q nCls t) (r50Pre16 2 q (sealW nCls) (sealX q t)) := by
  rw [pc16 q hq0 nCls t]
  exact EDiff_shift _ _ 1 (ed15 q hq0 hq nCls t)


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

/-- ⭐⭐ **The positive, continuous nonlinear factor**: one `istd` per BN on the carrier's path —
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

/-- ⭐⭐ **The class-0 difference between the two examples, along the ray, is `t · R t`.** -/
theorem gd_ray (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (hn : 0 < nCls) (t : ℝ) :
    resnet50ForwardB_full 2 q (sealW nCls) (sealX q t)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - resnet50ForwardB_full 2 q (sealW nCls) (sealX q t)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = t * Rr q nCls t := by
  rw [resnet50ForwardB_full_eq_chain, Function.comp_apply,
    head_diff q hq0 nCls hn _ (dP4 q nCls t) (ed16 q hq0 hq nCls t)]
  simp only [dP4, dP3, dP2, dP1, dS, Rr]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 11. `R` is continuous
-- ════════════════════════════════════════════════════════════════

theorem sealX_continuous (q : Nat) : Continuous (sealX q) := rayX_continuous _ _

theorem r50IdB_continuous (N h w mid oc : Nat) (p : R50IdW mid oc) (h1 : 0 < p.ε₁)
    (h2 : 0 < p.ε₂) (h3 : 0 < p.ε₃) : Continuous (r50IdB N h w p) :=
  (relu_continuous _).comp (residual_continuous _
    ((R34FullBSeal.projB_continuous N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ h3 p.γ₃ p.β₃).comp
      ((R34FullBSeal.cbReluB_continuous N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ h2 p.γ₂ p.β₂).comp
        (R34FullBSeal.cbReluB_continuous N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ h1 p.γ₁ p.β₁))))

theorem r50ProjB_continuous (N h w ic mid oc : Nat) (p : R50ProjW ic mid oc) (h1 : 0 < p.ε₁)
    (h2 : 0 < p.ε₂) (h3 : 0 < p.ε₃) (hp : 0 < p.εp) : Continuous (r50ProjB N h w p) :=
  (relu_continuous _).comp (residualProj_continuous _ _
    (R34FullBSeal.projB_continuous N (h := h) (w := w) p.Wp p.bp p.εp hp p.γp p.βp)
    ((R34FullBSeal.projB_continuous N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ h3 p.γ₃ p.β₃).comp
      ((R34FullBSeal.cbReluB_continuous N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ h2 p.γ₂ p.β₂).comp
        (R34FullBSeal.cbReluB_continuous N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ h1 p.γ₁ p.β₁))))

theorem r50DownB_continuous (N h w ic mid oc : Nat) (p : R50ProjW ic mid oc) (h1 : 0 < p.ε₁)
    (h2 : 0 < p.ε₂) (h3 : 0 < p.ε₃) (hp : 0 < p.εp) : Continuous (r50DownB N h w p) :=
  (relu_continuous _).comp (residualProj_continuous _ _
    (R34FullBSeal.projStridedB_continuous N (h := h) (w := w) p.Wp p.bp p.εp hp p.γp p.βp)
    ((R34FullBSeal.projB_continuous N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ h3 p.γ₃ p.β₃).comp
      ((R34FullBSeal.cbReluStridedB_continuous N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ h2 p.γ₂
          p.β₂).comp
        (R34FullBSeal.cbReluB_continuous N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ h1 p.γ₁
          p.β₁))))

theorem cn0 (q : Nat) (nCls : Nat) : Continuous (r50Pre0 2 q (sealW nCls)) :=
  R34FullBSeal.r34StemB_continuous 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) _ _ _ one_pos _ _

theorem cn1 (q : Nat) (nCls : Nat) : Continuous (r50Pre1 2 q (sealW nCls)) :=
  (r50ProjB_continuous 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 64 256 (sealW nCls).s1b0 one_pos one_pos one_pos one_pos).comp
    (cn0 q nCls)

theorem cn2 (q : Nat) (nCls : Nat) : Continuous (r50Pre2 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (sealW nCls).s1b1 one_pos one_pos one_pos).comp (cn1 q nCls)

theorem cn3 (q : Nat) (nCls : Nat) : Continuous (r50Pre3 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 256 (sealW nCls).s1b2 one_pos one_pos one_pos).comp (cn2 q nCls)

theorem cn4 (q : Nat) (nCls : Nat) : Continuous (r50Pre4 2 q (sealW nCls)) :=
  (r50DownB_continuous 2 (2 * (2 * q)) (2 * (2 * q)) 256 128 512 (sealW nCls).s2b0 one_pos one_pos one_pos one_pos).comp
    (cn3 q nCls)

theorem cn5 (q : Nat) (nCls : Nat) : Continuous (r50Pre5 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (sealW nCls).s2b1 one_pos one_pos one_pos).comp (cn4 q nCls)

theorem cn6 (q : Nat) (nCls : Nat) : Continuous (r50Pre6 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (sealW nCls).s2b2 one_pos one_pos one_pos).comp (cn5 q nCls)

theorem cn7 (q : Nat) (nCls : Nat) : Continuous (r50Pre7 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * (2 * q)) (2 * (2 * q)) 128 512 (sealW nCls).s2b3 one_pos one_pos one_pos).comp (cn6 q nCls)

theorem cn8 (q : Nat) (nCls : Nat) : Continuous (r50Pre8 2 q (sealW nCls)) :=
  (r50DownB_continuous 2 (2 * q) (2 * q) 512 256 1024 (sealW nCls).s3b0 one_pos one_pos one_pos one_pos).comp
    (cn7 q nCls)

theorem cn9 (q : Nat) (nCls : Nat) : Continuous (r50Pre9 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * q) (2 * q) 256 1024 (sealW nCls).s3b1 one_pos one_pos one_pos).comp (cn8 q nCls)

theorem cn10 (q : Nat) (nCls : Nat) : Continuous (r50Pre10 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * q) (2 * q) 256 1024 (sealW nCls).s3b2 one_pos one_pos one_pos).comp (cn9 q nCls)

theorem cn11 (q : Nat) (nCls : Nat) : Continuous (r50Pre11 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * q) (2 * q) 256 1024 (sealW nCls).s3b3 one_pos one_pos one_pos).comp (cn10 q nCls)

theorem cn12 (q : Nat) (nCls : Nat) : Continuous (r50Pre12 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * q) (2 * q) 256 1024 (sealW nCls).s3b4 one_pos one_pos one_pos).comp (cn11 q nCls)

theorem cn13 (q : Nat) (nCls : Nat) : Continuous (r50Pre13 2 q (sealW nCls)) :=
  (r50IdB_continuous 2 (2 * q) (2 * q) 256 1024 (sealW nCls).s3b5 one_pos one_pos one_pos).comp (cn12 q nCls)

theorem Zs_continuous (q : Nat) : Continuous (Zs q) :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp
    (rayX_continuous _ _)

theorem Zp1_continuous (q : Nat) (nCls : Nat) : Continuous (Zp1 q nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn0 q nCls).comp (sealX_continuous q))

theorem Zp2_continuous (q : Nat) (nCls : Nat) : Continuous (Zp2 q nCls) :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp
    ((cn3 q nCls).comp (sealX_continuous q))

theorem Zp3_continuous (q : Nat) (nCls : Nat) : Continuous (Zp3 q nCls) :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp
    ((cn7 q nCls).comp (sealX_continuous q))

theorem Zp4_continuous (q : Nat) (nCls : Nat) : Continuous (Zp4 q nCls) :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp
    ((cn13 q nCls).comp (sealX_continuous q))

theorem Rr_continuous (q : Nat) (hq0 : 0 < q) (nCls : Nat) : Continuous (Rr q nCls) := by
  have c0 : Continuous (fun t : ℝ =>
      bnIstd (2 * ((2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q))))))
        (bnRowLA 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) (Zs q t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by positivity⟩ : Fin (2 * ((2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))).comp
      ((bnRowLA_continuous 2 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))) 0).comp (Zs_continuous q))
  have c1 : Continuous (fun t : ℝ =>
      bnIstd (2 * ((2 * (2 * (2 * q))) * (2 * (2 * (2 * q)))))
        (bnRowLA 2 256 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (Zp1 q nCls t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by positivity⟩ : Fin (2 * ((2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))).comp
      ((bnRowLA_continuous 2 256 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 0).comp (Zp1_continuous q nCls))
  have c2 : Continuous (fun t : ℝ =>
      bnIstd (2 * ((2 * (2 * q)) * (2 * (2 * q)))) (bnRowLA 2 512 (2 * (2 * q)) (2 * (2 * q)) (Zp2 q nCls t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by positivity⟩ : Fin (2 * ((2 * (2 * q)) * (2 * (2 * q)))))).comp
      ((bnRowLA_continuous 2 512 (2 * (2 * q)) (2 * (2 * q)) 0).comp (Zp2_continuous q nCls))
  have c3 : Continuous (fun t : ℝ =>
      bnIstd (2 * ((2 * q) * (2 * q))) (bnRowLA 2 1024 (2 * q) (2 * q) (Zp3 q nCls t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by positivity⟩ : Fin (2 * ((2 * q) * (2 * q))))).comp
      ((bnRowLA_continuous 2 1024 (2 * q) (2 * q) 0).comp (Zp3_continuous q nCls))
  have c4 : Continuous (fun t : ℝ =>
      bnIstd (2 * (q * q)) (bnRowLA 2 2048 q q (Zp4 q nCls t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by positivity⟩ : Fin (2 * (q * q)))).comp
      ((bnRowLA_continuous 2 2048 q q 0).comp (Zp4_continuous q nCls))
  exact c0.mul (c1.mul (c2.mul (c3.mul c4)))

-- ════════════════════════════════════════════════════════════════
-- § 12. The seal
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **Level 2 — the witness is non-degenerate**: the full-width batch-BN ResNet-50 at the
    structural weights is NOT constant in its input, at either shipped resolution. -/
theorem sealX_nonconstant (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat) (hn : 0 < nCls) :
    resnet50ForwardB_full 2 q (sealW nCls) (sealX q 1)
      ≠ resnet50ForwardB_full 2 q (sealW nCls) (sealX q 0) := by
  intro heq
  have h1 := gd_ray q hq0 hq nCls hn 1
  have h0 := gd_ray q hq0 hq nCls hn 0
  rw [heq] at h1
  have hz : (1 : ℝ) * Rr q nCls 1 = 0 * Rr q nCls 0 := by rw [← h1, ← h0]
  rw [one_mul, zero_mul] at hz
  linarith [Rr_pos q nCls 1]

/-- ⭐⭐ **Level 3 — the whole-net Jacobian is nonzero at the witness.** -/
theorem sealX_jacobian_nonzero (q : Nat) (hq0 : 0 < q) (hq : q ≤ 7) (nCls : Nat)
    (hn : 0 < nCls) :
    fderiv ℝ (resnet50ForwardB_full 2 q (sealW nCls)) (sealX q 0) ≠ 0 := by
  refine fderiv_ne_zero_of_ray (sealV q) (sealDiffAt q hq0 hq nCls 0)
    (fun y => y (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - y (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))) (by fun_prop)
    (Rr_pos q nCls 0).ne' ?_
  have heq : (fun t : ℝ => resnet50ForwardB_full 2 q (sealW nCls) (sealX q 0 + t • sealV q)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - resnet50ForwardB_full 2 q (sealW nCls) (sealX q 0 + t • sealV q)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls))))
      = fun t : ℝ => t * Rr q nCls t := by
    funext t
    rw [sealX_zero_add]
    exact gd_ray q hq0 hq nCls hn t
  rw [heq]
  exact hasDerivAt_mul_self_zero (Rr_continuous q hq0 nCls).continuousAt

/-- ⭐⭐ **The seal**: the proven whole-network backward of the **full-width, batch-BatchNorm,
    [3,4,6,3]-bottleneck** ResNet-50 — `resnet50ForwardB_full`, at BOTH shipped resolutions —
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
