import LeanMlir.Proofs.Architectures.ViTMultiHead

/-!
# ViT scaling pass — depth-k (general-depth tower, distinct per-block params)

The proven `transformerTower_has_vjp_mat` shares ONE param tuple across blocks;
the 2-block `vitForward2(V)` carried distinct params but fixed the depth. This
file closes general depth at the production form (vector-[D] LN + multi-head):

1. **`BlockParamsV`** — the 16-field per-block param structure, and
   **`vitBodyKVFlat`** — the depth-`k` block fold (head recursion: block 0
   first), with **`vitBodyKVFlat_has_vjp`** by induction on `k` (the chain step
   is `vjp_comp` + the bridged `transformerBlockV_has_vjp_mat`, exactly
   `vitForward2V_has_vjp`'s step with a `Fin k` param function).
   **`vitForwardKV(_has_vjp[_correct])`** — the whole net at depth `k`,
   UNCONDITIONAL except `0 < ε`. `vitForwardKV_two_eq`: at `k = 2` it IS
   `vitForward2V` (definitional).

2. **`vitBodyGraphKMHV`** — the token-level fold of `vitBlockGraphMHV` with
   per-block SSA prefixes `b{base+i}_`, and **`vitFwdGraphKMHV_faithful`**:
   the depth-`k` multi-head vector-LN forward graph denotes `vitForwardKV` at
   `heads := hm1 + 1` — by induction on `k` chaining
   `vitBlockGraphMHV_den_aux` + `vitBlockSpelledMHV_eq` per block (the
   per-block den_aux was designed for exactly this).

Depth-12 ViT-Tiny shapes are now a config change away (the production capstone
needs only the P=16/D=192/heads=3 instantiation of these).
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 1. Per-block params + the depth-k body fold (flat level) + VJP
-- ════════════════════════════════════════════════════════════════

/-- The 16 per-block ViT parameters (vector-LN form), bundled so depth-`k`
    signatures stay sane (`params : Fin k → BlockParamsV D mlpDim`). -/
structure BlockParamsV (D mlpDim : Nat) where
  γ1 : Vec D
  β1 : Vec D
  Wq : Mat D D
  Wk : Mat D D
  Wv : Mat D D
  Wo : Mat D D
  bq : Vec D
  bk : Vec D
  bv : Vec D
  bo : Vec D
  γ2 : Vec D
  β2 : Vec D
  Wfc1 : Mat D mlpDim
  bfc1 : Vec mlpDim
  Wfc2 : Mat mlpDim D
  bfc2 : Vec D

/-- `transformerBlockV` at a bundled param block. -/
noncomputable def blockV (Np1 heads d_head mlpDim : Nat) (ε : ℝ)
    (p : BlockParamsV (heads * d_head) mlpDim) :
    Mat Np1 (heads * d_head) → Mat Np1 (heads * d_head) :=
  transformerBlockV Np1 heads d_head mlpDim ε p.γ1 p.β1 p.Wq p.Wk p.Wv p.Wo
    p.bq p.bk p.bv p.bo p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2

/-- One block at the flat index (the `vitForward2V` per-block spelling). -/
noncomputable def blockVFlat (Np1 heads d_head mlpDim : Nat) (ε : ℝ)
    (p : BlockParamsV (heads * d_head) mlpDim) :
    Vec (Np1 * (heads * d_head)) → Vec (Np1 * (heads * d_head)) :=
  fun v => Mat.flatten (blockV Np1 heads d_head mlpDim ε p (Mat.unflatten v))

/-- **Depth-`k` block fold** (Mat level, head recursion — block `0` runs
    first): `body (k+1) ps = body k (ps ∘ succ) ∘ block (ps 0)`. -/
noncomputable def vitBodyKV (Np1 heads d_head mlpDim : Nat) (ε : ℝ) :
    (k : Nat) → (Fin k → BlockParamsV (heads * d_head) mlpDim) →
    Mat Np1 (heads * d_head) → Mat Np1 (heads * d_head)
  | 0, _ => fun A => A
  | k + 1, ps =>
      (vitBodyKV Np1 heads d_head mlpDim ε k (fun i => ps i.succ)) ∘
      (blockV Np1 heads d_head mlpDim ε (ps 0))

/-- **Depth-`k` block fold at the flat index** — per-block flat stages (the
    `vitForward2V` spelling, so the VJP composes block-at-a-time). -/
noncomputable def vitBodyKVFlat (Np1 heads d_head mlpDim : Nat) (ε : ℝ) :
    (k : Nat) → (Fin k → BlockParamsV (heads * d_head) mlpDim) →
    Vec (Np1 * (heads * d_head)) → Vec (Np1 * (heads * d_head))
  | 0, _ => fun v => v
  | k + 1, ps =>
      (vitBodyKVFlat Np1 heads d_head mlpDim ε k (fun i => ps i.succ)) ∘
      (blockVFlat Np1 heads d_head mlpDim ε (ps 0))

/-- The flat fold on a flattened input is the flatten of the Mat fold (the
    per-block `unflatten ∘ flatten` round-trips cancel, inductively). -/
lemma vitBodyKVFlat_eq_flatten (Np1 heads d_head mlpDim : Nat) (ε : ℝ) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
      (A : Mat Np1 (heads * d_head)),
      vitBodyKVFlat Np1 heads d_head mlpDim ε k ps (Mat.flatten A) =
        Mat.flatten (vitBodyKV Np1 heads d_head mlpDim ε k ps A)
  | 0, _, _ => rfl
  | k + 1, ps, A => by
      have hb : blockVFlat Np1 heads d_head mlpDim ε (ps 0) (Mat.flatten A) =
          Mat.flatten (blockV Np1 heads d_head mlpDim ε (ps 0) A) := by
        unfold blockVFlat
        rw [Mat.unflatten_flatten]
      show vitBodyKVFlat Np1 heads d_head mlpDim ε k (fun i => ps i.succ)
          (blockVFlat Np1 heads d_head mlpDim ε (ps 0) (Mat.flatten A)) = _
      rw [hb]
      exact vitBodyKVFlat_eq_flatten Np1 heads d_head mlpDim ε k
        (fun i => ps i.succ) (blockV Np1 heads d_head mlpDim ε (ps 0) A)

/-- Flat differentiability of the depth-`k` body, by induction on `k`. -/
lemma vitBodyKVFlat_diff (Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim),
      Differentiable ℝ (vitBodyKVFlat Np1 heads d_head mlpDim ε k ps)
  | 0, _ => differentiable_id
  | k + 1, ps =>
      Differentiable.comp
        (vitBodyKVFlat_diff Np1 heads d_head mlpDim ε hε k (fun i => ps i.succ))
        (transformerBlockV_flat_diff Np1 heads d_head mlpDim ε
          (ps 0).γ1 (ps 0).β1 hε (ps 0).Wq (ps 0).Wk (ps 0).Wv (ps 0).Wo
          (ps 0).bq (ps 0).bk (ps 0).bv (ps 0).bo (ps 0).γ2 (ps 0).β2
          (ps 0).Wfc1 (ps 0).bfc1 (ps 0).Wfc2 (ps 0).bfc2)

/-- **Depth-`k` body VJP** — the tower induction at distinct per-block params:
    the chain step is `vjp_comp` gluing the bridged
    `transformerBlockV_has_vjp_mat` onto the depth-`k` tail. Only `0 < ε`. -/
noncomputable def vitBodyKVFlat_has_vjp (Np1 heads d_head mlpDim : Nat)
    (ε : ℝ) (hε : 0 < ε) :
    (k : Nat) → (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) →
    HasVJP (vitBodyKVFlat Np1 heads d_head mlpDim ε k ps)
  | 0, _ => identity_has_vjp _
  | k + 1, ps =>
      vjp_comp
        (blockVFlat Np1 heads d_head mlpDim ε (ps 0))
        (vitBodyKVFlat Np1 heads d_head mlpDim ε k (fun i => ps i.succ))
        (transformerBlockV_flat_diff Np1 heads d_head mlpDim ε
          (ps 0).γ1 (ps 0).β1 hε (ps 0).Wq (ps 0).Wk (ps 0).Wv (ps 0).Wo
          (ps 0).bq (ps 0).bk (ps 0).bv (ps 0).bo (ps 0).γ2 (ps 0).β2
          (ps 0).Wfc1 (ps 0).bfc1 (ps 0).Wfc2 (ps 0).bfc2)
        (vitBodyKVFlat_diff Np1 heads d_head mlpDim ε hε k (fun i => ps i.succ))
        (hasVJPMat_to_hasVJP (transformerBlockV_has_vjp_mat Np1 heads d_head mlpDim ε
          (ps 0).γ1 (ps 0).β1 hε (ps 0).Wq (ps 0).Wk (ps 0).Wv (ps 0).Wo
          (ps 0).bq (ps 0).bk (ps 0).bv (ps 0).bo (ps 0).γ2 (ps 0).β2
          (ps 0).Wfc1 (ps 0).bfc1 (ps 0).Wfc2 (ps 0).bfc2))
        (vitBodyKVFlat_has_vjp Np1 heads d_head mlpDim ε hε k (fun i => ps i.succ))

-- ════════════════════════════════════════════════════════════════
-- § 2. The depth-k ViT forward + whole-net VJP
-- ════════════════════════════════════════════════════════════════

/-- **Depth-`k` distinct-param ViT forward** (vector-LN): patch embed →
    `k` blocks (`Fin k → BlockParamsV`) → final vector-LN → CLS slice →
    dense head. `vitForward2V` generalized over depth. -/
noncomputable def vitForwardKV
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  (classifier_flat N (heads * d_head) nClasses Wcls bcls) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (fun n => layerNormVec (heads * d_head) ε γF βF
      ((Mat.unflatten v) n))) ∘
  (vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps) ∘
  (patchEmbed_flat ic H W patchSize N (heads * d_head)
    W_conv b_conv cls_token pos_embed)

/-- **At `k = 2` the depth-`k` net IS `vitForward2V`** (definitional — the
    fold unrolls to exactly the 2-block composition). -/
theorem vitForwardKV_two_eq
    (ic H W patchSize N mlpDim heads d_head nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (ps : Fin 2 → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses 2
      W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls =
    vitForward2V ic H W patchSize N mlpDim heads d_head nClasses
      W_conv b_conv cls_token pos_embed ε
      (ps 0).γ1 (ps 0).β1 (ps 0).Wq (ps 0).Wk (ps 0).Wv (ps 0).Wo
      (ps 0).bq (ps 0).bk (ps 0).bv (ps 0).bo (ps 0).γ2 (ps 0).β2
      (ps 0).Wfc1 (ps 0).bfc1 (ps 0).Wfc2 (ps 0).bfc2
      (ps 1).γ1 (ps 1).β1 (ps 1).Wq (ps 1).Wk (ps 1).Wv (ps 1).Wo
      (ps 1).bq (ps 1).bk (ps 1).bv (ps 1).bo (ps 1).γ2 (ps 1).β2
      (ps 1).Wfc1 (ps 1).bfc1 (ps 1).Wfc2 (ps 1).bfc2
      γF βF Wcls bcls := rfl

/-- **Whole-net VJP for the depth-`k` ViT (global).** All-smooth, so the only
    hypothesis is `0 < ε` — at EVERY depth. Three `vjp_comp` steps gluing
    `patchEmbed_flat_has_vjp`, the inductive `vitBodyKVFlat_has_vjp`, the
    bridged per-token vector-LN, and `classifier_flat_has_vjp`. -/
noncomputable def vitForwardKV_has_vjp
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    HasVJP (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls) := by
  unfold vitForwardKV
  set PE := patchEmbed_flat ic H W patchSize N (heads * d_head)
              W_conv b_conv cls_token pos_embed with hPE
  have pe_diff := patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
                    W_conv b_conv cls_token pos_embed
  have pe_vjp : HasVJP PE := patchEmbed_flat_has_vjp ic H W patchSize N
                    (heads * d_head) W_conv b_conv cls_token pos_embed
  set BODY := vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps with hBODY
  have body_diff : Differentiable ℝ BODY :=
    vitBodyKVFlat_diff (N + 1) heads d_head mlpDim ε hε k ps
  have body_vjp : HasVJP BODY :=
    vitBodyKVFlat_has_vjp (N + 1) heads d_head mlpDim ε hε k ps
  have s1_vjp : HasVJP (BODY ∘ PE) := vjp_comp PE BODY pe_diff body_diff pe_vjp body_vjp
  have s1_diff : Differentiable ℝ (BODY ∘ PE) := body_diff.comp pe_diff
  set LNF := (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (fun n => layerNormVec (heads * d_head) ε γF βF
      ((Mat.unflatten v) n))) with hLNF
  have lnf_diff : Differentiable ℝ LNF :=
    layerNormVec_per_token_flat_diff (N + 1) (heads * d_head) ε γF βF hε
  have lnf_vjp : HasVJP LNF :=
    hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat (N + 1) (heads * d_head)
      ε γF βF hε)
  have s2_vjp : HasVJP (LNF ∘ (BODY ∘ PE)) :=
    vjp_comp (BODY ∘ PE) LNF s1_diff lnf_diff s1_vjp lnf_vjp
  have s2_diff : Differentiable ℝ (LNF ∘ (BODY ∘ PE)) := lnf_diff.comp s1_diff
  exact vjp_comp (LNF ∘ (BODY ∘ PE))
    (classifier_flat N (heads * d_head) nClasses Wcls bcls)
    s2_diff (classifier_flat_diff N (heads * d_head) nClasses Wcls bcls)
    s2_vjp (classifier_flat_has_vjp N (heads * d_head) nClasses Wcls bcls)

/-- **Public correctness theorem for `vitForwardKV_has_vjp`** — the depth-`k`
    ViT's backward equals the `pdiv`-contracted Jacobian at every input. -/
theorem vitForwardKV_has_vjp_correct
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (dy : Vec nClasses) (i : Fin (ic * H * W)) :
    (vitForwardKV_has_vjp ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls) x i j * dy j :=
  (vitForwardKV_has_vjp ic H W patchSize N mlpDim heads d_head nClasses k
    W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § 3. The production capstone — ViT-Tiny at its real dimensions
-- ════════════════════════════════════════════════════════════════

/-- **ViT-Tiny whole-network VJP — the production capstone.**

    `vitForwardKV_has_vjp_correct` instantiated at the exact `MainVitTrain.lean`
    `vitTiny` spec: a `3×224×224` image, `16×16` patches (`N = 196` patch tokens
    + the CLS token), embedding dim `D = 192 = 3 heads × 64`, MLP dim `768`,
    **12 transformer blocks with DISTINCT per-block parameters**
    (`ps : Fin 12 → BlockParamsV 192 768`), and Imagenette's `10` classes.

    The full 12-block / 3-head ViT-Tiny's backward pass equals its Mathlib-`fderiv`
    Jacobian-transpose contracted with the cotangent, at **every** input image —
    UNCONDITIONAL except `0 < ε` (softmax / GELU / vector-LN are kink-free, so no
    smoothness witness is needed, and the statement is generic in the weights, so
    it is non-degenerate by construction). The ViT peer of `convNextForwardT_has_vjp`
    (18-block ConvNeXt-T) and `efficientnetForwardB_full_has_vjp` (16-block
    EfficientNet-B0): a full-spec, real-architecture whole-network backward. -/
theorem vitTiny_has_vjp_correct
    (W_conv : Kernel4 (3 * 64) 3 16 16)
    (b_conv : Vec (3 * 64))
    (cls_token : Vec (3 * 64))
    (pos_embed : Mat (196 + 1) (3 * 64))
    (ε : ℝ) (hε : 0 < ε)
    (ps : Fin 12 → BlockParamsV (3 * 64) 768)
    (γF βF : Vec (3 * 64))
    (Wcls : Mat (3 * 64) 10) (bcls : Vec 10)
    (x : Vec (3 * 224 * 224)) (dy : Vec 10) (i : Fin (3 * 224 * 224)) :
    (vitForwardKV_has_vjp 3 224 224 16 196 768 3 64 10 12
      W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward x dy i =
      ∑ j : Fin 10,
        pdiv (vitForwardKV 3 224 224 16 196 768 3 64 10 12
          W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls) x i j * dy j :=
  vitForwardKV_has_vjp_correct 3 224 224 16 196 768 3 64 10 12
    W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls x dy i

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 3. The depth-k multi-head graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- `vitBlockGraphMHV` at a bundled param block. -/
def vitBlockGraphMHVP {Np1 hm1 d mlpDim : Nat}
    (pfx epsStr sStr oneStr zeroStr : String) (ε s : ℝ)
    (p : BlockParamsV ((hm1 + 1) * d) mlpDim)
    (x : SHlo (Np1 * ((hm1 + 1) * d))) : SHlo (Np1 * ((hm1 + 1) * d)) :=
  vitBlockGraphMHV pfx epsStr sStr oneStr zeroStr ε s p.γ1 p.β1
    p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.γ2 p.β2
    p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 x

/-- **Depth-`k` token-level block fold** — block `base` first, SSA prefixes
    `b{base+1}_`, `b{base+2}_`, … (distinct per block). -/
def vitBodyGraphKMHV {Np1 hm1 d mlpDim : Nat}
    (epsStr sStr oneStr zeroStr : String) (ε s : ℝ) :
    (base k : Nat) → (Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim) →
    SHlo (Np1 * ((hm1 + 1) * d)) → SHlo (Np1 * ((hm1 + 1) * d))
  | _, 0, _, e => e
  | base, k + 1, ps, e =>
      vitBodyGraphKMHV epsStr sStr oneStr zeroStr ε s (base + 1) k
        (fun i => ps i.succ)
        (vitBlockGraphMHVP s!"b{base + 1}_" epsStr sStr oneStr zeroStr ε s
          (ps 0) e)

/-- **Depth-`k` body denotation** — by induction on `k`, chaining
    `vitBlockGraphMHV_den_aux` + `vitBlockSpelledMHV_eq` per block: the token
    fold denotes the flatten of the Mat block fold at `heads := hm1 + 1`. -/
lemma vitBodyGraphKMHV_den {Np1 hm1 d mlpDim : Nat}
    (epsStr sStr oneStr zeroStr : String) (ε : ℝ) :
    ∀ (base k : Nat) (ps : Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim)
      (e : SHlo (Np1 * ((hm1 + 1) * d))) (A : Mat Np1 ((hm1 + 1) * d)),
      den e = Mat.flatten A →
      den (vitBodyGraphKMHV epsStr sStr oneStr zeroStr ε (sdpa_scale d)
            base k ps e) =
        Mat.flatten (vitBodyKV Np1 (hm1 + 1) d mlpDim ε k ps A)
  | _, 0, _, _, _, hA => hA
  | base, k + 1, ps, e, A, hA => by
      have hb := vitBlockGraphMHV_den_aux s!"b{base + 1}_" epsStr sStr oneStr zeroStr
        ε (ps 0).γ1 (ps 0).β1 (ps 0).Wq (ps 0).Wk (ps 0).Wv (ps 0).Wo
        (ps 0).bq (ps 0).bk (ps 0).bv (ps 0).bo (ps 0).γ2 (ps 0).β2
        (ps 0).Wfc1 (ps 0).bfc1 (ps 0).Wfc2 (ps 0).bfc2 e A hA
      have ih := vitBodyGraphKMHV_den epsStr sStr oneStr zeroStr ε
        (base + 1) k (fun i => ps i.succ) _ _ hb
      -- ih lands at the spelled block; tie it to `blockV` and refold the body.
      rw [vitBlockSpelledMHV_eq] at ih
      exact ih

/-- Whole **depth-`k` multi-head vector-LN ViT forward** graph: patch embed →
    `k` spelled multi-head vector-LN blocks (`b1_`…`b{k}_`, distinct params) →
    final vector-LN → CLS slice → dense head. -/
def vitFwdGraphKMHV {ic H W P N hm1 d mlpDim nClasses : Nat}
    (epsStr sStr oneStr zeroStr : String) (ε s : ℝ)
    (Wc : Kernel4 ((hm1 + 1) * d) ic P P) (bc cls : Vec ((hm1 + 1) * d))
    (pos : Mat (N + 1) ((hm1 + 1) * d))
    (k : Nat) (ps : Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim)
    (γF βF : Vec ((hm1 + 1) * d))
    (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) : SHlo nClasses :=
  let embed : SHlo ((N + 1) * ((hm1 + 1) * d)) :=
    .patchEmbedF "%Wp" "%bp" "%cls" "%pos" Wc bc cls pos (.operand "%x" x)
  let body := vitBodyGraphKMHV epsStr sStr oneStr zeroStr ε s 0 k ps embed
  let fl := SHlo.rowBiasF "%btF" βF
    (SHlo.rowScaleF "%gF" γF
      (SHlo.lnRowF oneStr zeroStr epsStr ε 1 0 body))
  denseF "%Wcls" "%bcls" Wcls bcls (.clsSliceF fl)

/-- **Depth-`k` multi-head vector-LN ViT forward faithfulness** — the
    general-depth graph denotes `vitForwardKV` at `heads := hm1 + 1`, for
    EVERY depth `k`. The depth analogue of `vitFwdGraphMHV_faithful`. -/
theorem vitFwdGraphKMHV_faithful
    (ic H W patchSize N hm1 d mlpDim nClasses : Nat)
    (epsStr sStr oneStr zeroStr : String)
    (Wc : Kernel4 ((hm1 + 1) * d) ic patchSize patchSize)
    (bc cls : Vec ((hm1 + 1) * d)) (pos : Mat (N + 1) ((hm1 + 1) * d))
    (ε : ℝ)
    (k : Nat) (ps : Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim)
    (γF βF : Vec ((hm1 + 1) * d))
    (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    den (vitFwdGraphKMHV epsStr sStr oneStr zeroStr ε (sdpa_scale d)
          Wc bc cls pos k ps γF βF Wcls bcls x)
      = vitForwardKV ic H W patchSize N mlpDim (hm1 + 1) d nClasses k
          Wc bc cls pos ε ps γF βF Wcls bcls x := by
  have h0 : den (SHlo.patchEmbedF (P := patchSize) "%Wp" "%bp" "%cls" "%pos"
        Wc bc cls pos (.operand "%x" x))
      = Mat.flatten (Mat.unflatten
          (patchEmbed_flat ic H W patchSize N ((hm1 + 1) * d) Wc bc cls pos x)) := by
    simp only [patchEmbedF_faithful, den_operand]
    rw [Mat.flatten_unflatten]
    rfl
  have hbody := vitBodyGraphKMHV_den epsStr sStr oneStr zeroStr ε 0 k ps _ _ h0
  simp only [vitFwdGraphKMHV, denseF_faithful, clsSliceF_faithful, rowBiasF_faithful,
             rowScaleF_faithful, lnRowF_faithful, hbody]
  simp only [rowLNFlat_flat, rowScaleFlat_flat, rowBiasFlat_flat]
  unfold vitForwardKV classifier_flat
  simp only [Function.comp_apply]
  rw [← Mat.flatten_unflatten
        (patchEmbed_flat ic H W patchSize N ((hm1 + 1) * d) Wc bc cls pos x),
      vitBodyKVFlat_eq_flatten]
  simp only [Mat.unflatten_flatten]
  rfl

end Proofs.StableHLO
