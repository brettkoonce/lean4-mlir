import LeanMlir.Proofs.Nets.ViT.ViTDepthK
import LeanMlir.Proofs.Foundation.ParamGrad
import LeanMlir.Proofs.Codegen.StableHLO.Pretty

/-! # ViT with stochastic depth — the batched forward graph and its faithfulness

`ViTDepthK` states the ViT forward one example at a time (`vitFwdGraphKMHV_faithful`). The
`*drop*` artifacts — `vit_drop_fwd`, `vitin_drop_fwd`, `vitsin_drop_fwd` and the forward half of
every `*drop*` train step, the book's `vitin_emadp128x4wxclipdropbf16` among them — add
**stochastic depth**: two `%dp<i>` mask inputs per block, a per-example scale `dropPath` on the
attention branch (after the out-dense, before the first skip add) and on the MLP branch (after
fc2, before the second), as `ViTRenderB.vBlockFwdB` emits them. Block `i`'s attention site is
`%dp<2i>` and its MLP site `%dp<2i+1>` (the render's `vitSiteIdx`).

**Why this graph is batched.** A drop mask is per EXAMPLE, so no per-example node can carry it:
in the per-example graph a node denotes one example and the batch is lifted outside the AST,
which is why the render that writes these artifacts is the batched one. So the statement here
is at the batched index `B`, over the batched tokens the render emits (`.batchOp` of the row
forms, `.matmulFB`, `.scaleB`, `.addVB`, `.dropPathB`), and it says what the per-example graph
says one level up: **example `t` of the batched graph's output is the per-example forward at
example `t`'s input, with example `t`'s mask entries as its drop scalars.**

* `vitFwdGraphBDrop_slice` / `vitFwdGraphBDrop_faithful` — the graph denotes
  `vitForwardKVDropB`, at every depth, every mask and every example;
* `vitForwardKVDrop_ones` / `vitForwardKVDropB_ones` — at all-ones masks (the driver's at eval)
  the forward is `vitForwardKV`, per example and lifted, exactly: the keep probability is folded
  into the mask (`Training/DropPath`).

The graph uses the render's SSA names (`%wConv`, `b<i>_`, `%gF`, `%Wc`, …). Like
`vitFwdGraphKMHV`, it is not tied to the artifact text: the render names each shared intermediate
once (LN1's output feeds Q, K and V), where a graph term repeats the subterm. These artifacts are
f32 where they are forwards; the bf16 train steps' forward differs by the matmul roundings, and
their backward through the drop sites is outside this statement, as it is outside
`ViTStepTieGB`.

## References

- Huang et al. 2016, *Deep Networks with Stochastic Depth*. <https://arxiv.org/abs/1603.09382>
- Touvron et al. 2021, *Training data-efficient image transformers & distillation through attention* (DeiT). <https://arxiv.org/abs/2012.12877>
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 1. One example: the block, the tower and the net with drop scalars
-- ════════════════════════════════════════════════════════════════

/-- **One vector-LN block with its two drop scalars**: `transformerBlockV` with the attention
    branch scaled by `a` and the MLP branch by `m` before their skip adds. -/
noncomputable def blockVDrop (Np1 heads d_head mlpDim : Nat) (ε a m : ℝ)
    (p : BlockParamsV (heads * d_head) mlpDim) :
    Mat Np1 (heads * d_head) → Mat Np1 (heads * d_head) :=
  biPathMat (fun X => X)
    (fun X r s => m * ((transformerMlp Np1 (heads * d_head) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2) ∘
      (fun X : Mat Np1 (heads * d_head) => fun n =>
        layerNormVec (heads * d_head) ε p.γ2 p.β2 (X n))) X r s) ∘
  biPathMat (fun X => X)
    (fun X r s => a * ((mhsaLayer Np1 heads d_head p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo) ∘
      (fun X : Mat Np1 (heads * d_head) => fun n =>
        layerNormVec (heads * d_head) ε p.γ1 p.β1 (X n))) X r s)

/-- At unit scalars the drop block is `blockV`. -/
theorem blockVDrop_one (Np1 heads d_head mlpDim : Nat) (ε : ℝ)
    (p : BlockParamsV (heads * d_head) mlpDim) :
    blockVDrop Np1 heads d_head mlpDim ε 1 1 p = blockV Np1 heads d_head mlpDim ε p := by
  unfold blockVDrop blockV transformerBlockV transformerAttnSublayerV transformerMlpSublayerV
  simp only [one_mul]

/-- The drop block spelled as the graph emits it — `vitBlockSpelledMHV` with the two scalars on
    the branches. -/
noncomputable def vitBlockSpelledMHVDrop (Np1 heads d mlpDim : Nat) (ε a mk : ℝ)
    (p : BlockParamsV (heads * d) mlpDim) (X : Mat Np1 (heads * d)) : Mat Np1 (heads * d) :=
  let xh1 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε 1 0 (X r)
  let sc1 : Mat Np1 (heads * d) := fun r => layerScale p.γ1 (xh1 r)
  let ln1 : Mat Np1 (heads * d) := fun r k => sc1 r k + p.β1 k
  let Q : Mat Np1 (heads * d) := fun r => dense p.Wq p.bq (ln1 r)
  let K : Mat Np1 (heads * d) := fun r => dense p.Wk p.bk (ln1 r)
  let V : Mat Np1 (heads * d) := fun r => dense p.Wv p.bv (ln1 r)
  let att : Mat Np1 (heads * d) := ∑ h : Fin heads, headPadMat Np1 heads d h
    (Mat.mul
      (rowSoftmax (fun i j => sdpaScale d *
        Mat.mul (headSliceMat Np1 heads d h Q)
          (Mat.transpose (headSliceMat Np1 heads d h K)) i j))
      (headSliceMat Np1 heads d h V))
  let O : Mat Np1 (heads * d) := fun r => dense p.Wo p.bo (att r)
  let hres : Mat Np1 (heads * d) := fun r s => X r s + a * O r s
  let xh2 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε 1 0 (hres r)
  let sc2 : Mat Np1 (heads * d) := fun r => layerScale p.γ2 (xh2 r)
  let ln2 : Mat Np1 (heads * d) := fun r k => sc2 r k + p.β2 k
  let m1 : Mat Np1 mlpDim := fun r => dense p.Wfc1 p.bfc1 (ln2 r)
  let g : Mat Np1 mlpDim := fun r => gelu mlpDim (m1 r)
  let m2 : Mat Np1 (heads * d) := fun r => dense p.Wfc2 p.bfc2 (g r)
  fun r s => hres r s + mk * m2 r s

/-- The spelled drop block IS `blockVDrop` (`vitBlockSpelledMHV_eq`'s proof, at the scalars). -/
lemma vitBlockSpelledMHVDrop_eq (Np1 heads d mlpDim : Nat) (ε a mk : ℝ)
    (p : BlockParamsV (heads * d) mlpDim) (X : Mat Np1 (heads * d)) :
    vitBlockSpelledMHVDrop Np1 heads d mlpDim ε a mk p X
      = blockVDrop Np1 heads d mlpDim ε a mk p X := by
  unfold blockVDrop transformerMlp biPathMat vitBlockSpelledMHVDrop
  simp only [Function.comp_apply]
  rw [mhsaLayer_spelled]
  rfl

/-- **The depth-`k` tower with drop scalars** — `vitBodyKV` with block `i` at `sd i`
    (attention, MLP). -/
noncomputable def vitBodyKVDrop (Np1 heads d_head mlpDim : Nat) (ε : ℝ) :
    (k : Nat) → (Fin k → BlockParamsV (heads * d_head) mlpDim) → (Fin k → ℝ × ℝ) →
    Mat Np1 (heads * d_head) → Mat Np1 (heads * d_head)
  | 0, _, _ => fun A => A
  | k + 1, ps, sd =>
      (vitBodyKVDrop Np1 heads d_head mlpDim ε k (fun i => ps i.succ) (fun i => sd i.succ)) ∘
      (blockVDrop Np1 heads d_head mlpDim ε (sd 0).1 (sd 0).2 (ps 0))

/-- At unit scalars the tower is `vitBodyKV`. -/
lemma vitBodyKVDrop_ones (Np1 heads d_head mlpDim : Nat) (ε : ℝ) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim),
      vitBodyKVDrop Np1 heads d_head mlpDim ε k ps (fun _ => (1, 1))
        = vitBodyKV Np1 heads d_head mlpDim ε k ps
  | 0, _ => rfl
  | k + 1, ps => by
      rw [vitBodyKVDrop, vitBodyKV, vitBodyKVDrop_ones Np1 heads d_head mlpDim ε k]
      exact congrArg _ (blockVDrop_one Np1 heads d_head mlpDim ε (ps 0))

/-- **The depth-`k` ViT forward with drop scalars, one example**: `vitForwardKV` with the tower
    at `sd`. -/
noncomputable def vitForwardKVDrop
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) (sd : Fin k → ℝ × ℝ)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  (classifierFlat N (heads * d_head) nClasses Wcls bcls) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (fun n => layerNormVec (heads * d_head) ε γF βF
      ((Mat.unflatten v) n))) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (vitBodyKVDrop (N + 1) heads d_head mlpDim ε k ps sd (Mat.unflatten v))) ∘
  (patchEmbedFlat ic H W patchSize N (heads * d_head)
    W_conv b_conv cls_token pos_embed)

/-- At unit scalars the forward is `vitForwardKV`. -/
theorem vitForwardKVDrop_ones
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    vitForwardKVDrop ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps (fun _ => (1, 1)) γF βF Wcls bcls
      = vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls := by
  funext x
  unfold vitForwardKVDrop vitForwardKV
  simp only [Function.comp_apply, vitBodyKVDrop_ones]
  rw [← Mat.flatten_unflatten
        (patchEmbedFlat ic H W patchSize N (heads * d_head) W_conv b_conv cls_token pos_embed x),
      vitBodyKVFlat_eq_flatten]
  simp only [Mat.unflatten_flatten]

-- ════════════════════════════════════════════════════════════════
-- § 2. The batch: each example with its own mask entries
-- ════════════════════════════════════════════════════════════════

/-- **The batched ViT forward with stochastic depth**: example `t` is `vitForwardKVDrop` at
    example `t`'s input, with `(sdA i t, sdM i t)` as block `i`'s drop scalars. `sdA i` / `sdM i`
    are the render's per-example masks `%dp<2i>` / `%dp<2i+1>`. -/
noncomputable def vitForwardKVDropB (B : Nat)
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim) (sdA sdM : Fin k → Vec B)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (B * (ic * H * W)) → Vec (B * nClasses) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    vitForwardKVDrop ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε ps (fun i => (sdA i p.1, sdM i p.1)) γF βF Wcls bcls
      (StableHLO.batchSlice B (ic * H * W) x p.1) p.2

/-- **At the all-ones masks the batched forward is `vitForwardKV` lifted**, exactly — the masks
    the driver passes to the forward artifacts at eval. -/
theorem vitForwardKVDropB_ones (B : Nat)
    (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    vitForwardKVDropB B ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps (fun _ _ => 1) (fun _ _ => 1) γF βF Wcls bcls
      = StableHLO.batchMap B (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
          W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls) := by
  funext x idx
  simp only [vitForwardKVDropB, vitForwardKVDrop_ones]
  rfl

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 3. Slicing one example out of the batched tokens
-- ════════════════════════════════════════════════════════════════

/-- A batched descriptor token, sliced at example `t`, is its per-example map at the slice. -/
lemma batchSlice_den_batchOp {B a b : Nat} (op : BatchableOp a b) (e : SHlo (B * a)) (t : Fin B) :
    batchSlice B b (den (.batchOp (N := B) op e)) t = denOp op (batchSlice B a (den e) t) := by
  rw [den_batchOp, batchSlice_batchMap]

/-- `matmulFB` sliced at example `t` multiplies example `t`'s two operands
    (`den_matmulFB_per_example`, as a slice). -/
lemma batchSlice_den_matmulFB {B m k n : Nat} (a : SHlo (B * (m * k))) (b : SHlo (B * (k * n)))
    (t : Fin B) :
    batchSlice B (m * n) (den (.matmulFB a b)) t
      = matMulFlat m k n (batchSlice B (m * k) (den a) t) (batchSlice B (k * n) (den b) t) := by
  rw [den_matmulFB, batchSlice_batchMapAux]

lemma batchSlice_den_scaleB {B n : Nat} (sS : String) (s : ℝ) (e : SHlo (B * n)) (t : Fin B) :
    batchSlice B n (den (.scaleB sS s e)) t = fun i => batchSlice B n (den e) t i * s := rfl

lemma batchSlice_den_addVB {B n : Nat} (a b : SHlo (B * n)) (t : Fin B) :
    batchSlice B n (den (.addVB a b)) t
      = fun i => batchSlice B n (den a) t i + batchSlice B n (den b) t i := rfl

/-- **A drop site, sliced at example `t`, scales by example `t`'s mask entry** — the per-example
    content of `dropPathB`. -/
lemma batchSlice_den_dropPathB {B n : Nat} (mN : String) (s : Vec B) (e : SHlo (B * n))
    (t : Fin B) :
    batchSlice B n (den (.dropPathB mN s e)) t = fun i => s t * batchSlice B n (den e) t i := by
  funext i
  simp only [batchSlice, den_dropPathB, dropPath_apply, Equiv.symm_apply_apply]

/-- Left-assoc `addVB` fold of one batched graph per head — `headsSumG` at the batched index, in
    the render's order (`acc := pd₀`, then `addVB acc pd_h`). -/
def headsSumGB {B n : Nat} : {hm1 : Nat} → (Fin (hm1 + 1) → SHlo (B * n)) → SHlo (B * n)
  | 0, f => f 0
  | hm1 + 1, f => .addVB (headsSumGB (fun i => f i.castSucc)) (f (Fin.last (hm1 + 1)))

/-- The batched head fold, sliced at example `t`, is the sum over heads of the slices. -/
lemma batchSlice_den_headsSumGB {B n : Nat} {hm1 : Nat} (f : Fin (hm1 + 1) → SHlo (B * n))
    (t : Fin B) :
    batchSlice B n (den (headsSumGB f)) t = fun j => ∑ h, batchSlice B n (den (f h)) t j := by
  induction hm1 with
  | zero =>
      funext j
      simp [headsSumGB]
  | succ k ih =>
      funext j
      rw [headsSumGB, batchSlice_den_addVB, ih (fun i => f i.castSucc), Fin.sum_univ_castSucc]

/-- The right-multiplied scale commutes with flattening (`scale_flat`, operands swapped — the
    batched `scaleB` multiplies on the right). -/
lemma scale_flat_right {m n : Nat} (s : ℝ) (A : Mat m n) :
    (fun i => Mat.flatten A i * s) = Mat.flatten (fun r c => s * A r c) :=
  funext fun i => mul_comm (Mat.flatten A i) s

/-- A drop site's per-example scale commutes with flattening, pointwise. -/
lemma scale_flat_pt {m n : Nat} (s : ℝ) (A : Mat m n) (j : Fin (m * n)) :
    s * Mat.flatten A j = Mat.flatten (fun r c => s * A r c) j := rfl

-- ════════════════════════════════════════════════════════════════
-- § 4. The batched block, tower and net graphs + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **One batched ViT block with its two drop sites**, node for node `ViTRenderB.vBlockFwdB`:
    vector-LN 1 (`lnRow` → `rowScale` → `rowBias`), Q/K/V `denseRow`, per head `headSlice` →
    `transpose` → `matmulFB` → `scaleB` → `softmaxRow` → `matmulFB` → `headPad`, summed by
    `headsSumGB`; out `denseRow`, `dropPathB` at `mA`, skip `addVB`; vector-LN 2, fc1, GELU, fc2,
    `dropPathB` at `mM`, skip `addVB`. -/
def vitBlockGraphBDrop {B Np1 hm1 d mlpDim : Nat}
    (pfx epsStr sStr mA mM : String) (ε s : ℝ)
    (p : BlockParamsV ((hm1 + 1) * d) mlpDim) (a m : Vec B)
    (x : SHlo (B * (Np1 * ((hm1 + 1) * d)))) : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
  let ln1 : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    .batchOp (N := B) (.rowBias (m := Np1) s!"%{pfx}bt1" p.β1)
      (.batchOp (N := B) (.rowScale (m := Np1) s!"%{pfx}g1" p.γ1)
        (.batchOp (N := B) (.lnRow (m := Np1) (n := (hm1 + 1) * d) "%one" "%zero" epsStr ε 1 0) x))
  let q : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    .batchOp (N := B) (.denseRow (N := Np1) s!"%{pfx}Wq" s!"%{pfx}bq" p.Wq p.bq) ln1
  let k : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    .batchOp (N := B) (.denseRow (N := Np1) s!"%{pfx}Wk" s!"%{pfx}bk" p.Wk p.bk) ln1
  let v : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    .batchOp (N := B) (.denseRow (N := Np1) s!"%{pfx}Wv" s!"%{pfx}bv" p.Wv p.bv) ln1
  let att : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    headsSumGB (fun h : Fin (hm1 + 1) =>
      .batchOp (N := B) (.headPad (N := Np1) (heads := hm1 + 1) (d := d) h)
        (.matmulFB (m := Np1) (k := Np1) (n := d)
          (.batchOp (N := B) (.softmaxRow (m := Np1) (n := Np1))
            (.scaleB sStr s
              (.matmulFB (m := Np1) (k := d) (n := Np1)
                (.batchOp (N := B) (.headSlice (N := Np1) (heads := hm1 + 1) (d := d) h) q)
                (.batchOp (N := B) (.transpose (m := Np1) (n := d))
                  (.batchOp (N := B) (.headSlice (N := Np1) (heads := hm1 + 1) (d := d) h) k)))))
          (.batchOp (N := B) (.headSlice (N := Np1) (heads := hm1 + 1) (d := d) h) v)))
  let o : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    .batchOp (N := B) (.denseRow (N := Np1) s!"%{pfx}Wo" s!"%{pfx}bo" p.Wo p.bo) att
  let hres : SHlo (B * (Np1 * ((hm1 + 1) * d))) := .addVB x (.dropPathB mA a o)
  let ln2 : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    .batchOp (N := B) (.rowBias (m := Np1) s!"%{pfx}bt2" p.β2)
      (.batchOp (N := B) (.rowScale (m := Np1) s!"%{pfx}g2" p.γ2)
        (.batchOp (N := B) (.lnRow (m := Np1) (n := (hm1 + 1) * d) "%one" "%zero" epsStr ε 1 0)
          hres))
  let m2 : SHlo (B * (Np1 * ((hm1 + 1) * d))) :=
    .batchOp (N := B) (.denseRow (N := Np1) s!"%{pfx}Wfc2" s!"%{pfx}bfc2" p.Wfc2 p.bfc2)
      (.batchOp (N := B) (.gelu (n := Np1 * mlpDim))
        (.batchOp (N := B) (.denseRow (N := Np1) s!"%{pfx}Wfc1" s!"%{pfx}bfc1" p.Wfc1 p.bfc1) ln2))
  .addVB hres (.dropPathB mM m m2)

/-- **Example `t` of the batched drop block is the spelled drop block at example `t`'s input and
    mask entries.** -/
lemma vitBlockGraphBDrop_slice {B Np1 hm1 d mlpDim : Nat}
    (pfx epsStr sStr mA mM : String) (ε : ℝ)
    (p : BlockParamsV ((hm1 + 1) * d) mlpDim) (a m : Vec B)
    (e : SHlo (B * (Np1 * ((hm1 + 1) * d)))) (t : Fin B) (A : Mat Np1 ((hm1 + 1) * d))
    (hA : batchSlice B (Np1 * ((hm1 + 1) * d)) (den e) t = Mat.flatten A) :
    batchSlice B (Np1 * ((hm1 + 1) * d))
        (den (vitBlockGraphBDrop pfx epsStr sStr mA mM ε (sdpaScale d) p a m e)) t
      = Mat.flatten (vitBlockSpelledMHVDrop Np1 (hm1 + 1) d mlpDim ε (a t) (m t) p A) := by
  simp only [vitBlockGraphBDrop, batchSlice_den_addVB, batchSlice_den_dropPathB,
    batchSlice_den_batchOp, batchSlice_den_matmulFB, batchSlice_den_scaleB,
    batchSlice_den_headsSumGB, denOp, hA]
  simp only [rowLNFlat_flat, rowScaleFlat_flat, rowBiasFlat_flat, rowDenseFlat_flat,
    headSliceFlat_flat, transposeFlat_flat, matMulFlat_flat, scale_flat_right,
    rowSoftmaxFlat_flat, headPadFlat_flat, flatten_sum, gelu_flat, scale_flat_pt, add_flat_pt]
  rfl

/-- **The batched depth-`k` tower with its drop sites** — block `base + i` carries the prefix
    `b{base+i}_` and reads `%dp<2(base+i)>` / `%dp<2(base+i)+1>`, as `ViTRenderB.vitFwd12B`
    names them (`vitSiteIdx`). -/
def vitBodyGraphBDrop {B Np1 hm1 d mlpDim : Nat}
    (epsStr sStr : String) (ε s : ℝ) :
    (base k : Nat) → (Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim) →
    (Fin k → Vec B) → (Fin k → Vec B) →
    SHlo (B * (Np1 * ((hm1 + 1) * d))) → SHlo (B * (Np1 * ((hm1 + 1) * d)))
  | _, 0, _, _, _, e => e
  | base, k + 1, ps, sdA, sdM, e =>
      vitBodyGraphBDrop epsStr sStr ε s (base + 1) k
        (fun i => ps i.succ) (fun i => sdA i.succ) (fun i => sdM i.succ)
        (vitBlockGraphBDrop s!"b{base}_" epsStr sStr (dpName (2 * base)) (dpName (2 * base + 1))
          ε s (ps 0) (sdA 0) (sdM 0) e)

/-- Example `t` of the batched tower is the drop tower at example `t`'s input and mask entries —
    by induction on `k`, one `vitBlockGraphBDrop_slice` per block. -/
lemma vitBodyGraphBDrop_slice {B Np1 hm1 d mlpDim : Nat} (epsStr sStr : String) (ε : ℝ) :
    ∀ (base k : Nat) (ps : Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim)
      (sdA sdM : Fin k → Vec B) (e : SHlo (B * (Np1 * ((hm1 + 1) * d)))) (t : Fin B)
      (A : Mat Np1 ((hm1 + 1) * d)),
      batchSlice B (Np1 * ((hm1 + 1) * d)) (den e) t = Mat.flatten A →
      batchSlice B (Np1 * ((hm1 + 1) * d))
          (den (vitBodyGraphBDrop epsStr sStr ε (sdpaScale d) base k ps sdA sdM e)) t =
        Mat.flatten (vitBodyKVDrop Np1 (hm1 + 1) d mlpDim ε k ps (fun i => (sdA i t, sdM i t)) A)
  | _, 0, _, _, _, _, _, _, hA => hA
  | base, k + 1, ps, sdA, sdM, e, t, A, hA => by
      have hb := vitBlockGraphBDrop_slice s!"b{base}_" epsStr sStr (dpName (2 * base))
        (dpName (2 * base + 1)) ε (ps 0) (sdA 0) (sdM 0) e t A hA
      have ih := vitBodyGraphBDrop_slice epsStr sStr ε (base + 1) k
        (fun i => ps i.succ) (fun i => sdA i.succ) (fun i => sdM i.succ) _ t _ hb
      rw [vitBlockSpelledMHVDrop_eq] at ih
      exact ih

/-- **The batched ViT forward graph with stochastic depth** — the typed form of
    `ViTRenderB.vitFwd12B … (sd := true)` at depth `k`: batched patch embed over `%x`, the
    drop tower, final vector-LN, CLS slice, dense head. -/
def vitFwdGraphBDrop {B ic H W P N hm1 d mlpDim nClasses : Nat}
    (epsStr sStr : String) (ε s : ℝ)
    (Wc : Kernel4 ((hm1 + 1) * d) ic P P) (bc cls : Vec ((hm1 + 1) * d))
    (pos : Mat (N + 1) ((hm1 + 1) * d))
    (k : Nat) (ps : Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim) (sdA sdM : Fin k → Vec B)
    (γF βF : Vec ((hm1 + 1) * d))
    (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) : SHlo (B * nClasses) :=
  .batchOp (N := B) (.dense "%Wc" "%bc" Wcls bcls)
    (.batchOp (N := B) (.clsSlice (N := N) (D := (hm1 + 1) * d))
      (.batchOp (N := B) (.rowBias (m := N + 1) "%btF" βF)
        (.batchOp (N := B) (.rowScale (m := N + 1) "%gF" γF)
          (.batchOp (N := B) (.lnRow (m := N + 1) (n := (hm1 + 1) * d) "%one" "%zero" epsStr ε 1 0)
            (vitBodyGraphBDrop epsStr sStr ε s 0 k ps sdA sdM
              (.batchOp (N := B) (.patchEmbed (N := N) "%wConv" "%bConv" "%cls" "%pos"
                  Wc bc cls pos)
                (.operand "%x" x)))))))

/-- **Example `t` of the batched drop graph is the per-example drop forward at example `t`'s
    input, with example `t`'s mask entries** — for every depth `k`. -/
theorem vitFwdGraphBDrop_slice {B ic H W patchSize N hm1 d mlpDim nClasses : Nat}
    (epsStr sStr : String)
    (Wc : Kernel4 ((hm1 + 1) * d) ic patchSize patchSize)
    (bc cls : Vec ((hm1 + 1) * d)) (pos : Mat (N + 1) ((hm1 + 1) * d))
    (ε : ℝ)
    (k : Nat) (ps : Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim) (sdA sdM : Fin k → Vec B)
    (γF βF : Vec ((hm1 + 1) * d))
    (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) (t : Fin B) :
    batchSlice B nClasses (den (vitFwdGraphBDrop epsStr sStr ε (sdpaScale d)
        Wc bc cls pos k ps sdA sdM γF βF Wcls bcls x)) t
      = vitForwardKVDrop ic H W patchSize N mlpDim (hm1 + 1) d nClasses k
          Wc bc cls pos ε ps (fun i => (sdA i t, sdM i t)) γF βF Wcls bcls
          (batchSlice B (ic * H * W) x t) := by
  have h0 : batchSlice B ((N + 1) * ((hm1 + 1) * d))
      (den (.batchOp (N := B) (.patchEmbed (N := N) (P := patchSize) "%wConv" "%bConv" "%cls" "%pos"
          Wc bc cls pos) (.operand "%x" x))) t
      = Mat.flatten (Mat.unflatten (patchEmbedFlat ic H W patchSize N ((hm1 + 1) * d)
          Wc bc cls pos (batchSlice B (ic * H * W) x t))) := by
    rw [batchSlice_den_batchOp, Mat.flatten_unflatten, den_operand]
    rfl
  have hbody := vitBodyGraphBDrop_slice epsStr sStr ε 0 k ps sdA sdM _ t _ h0
  simp only [vitFwdGraphBDrop, batchSlice_den_batchOp, denOp, hbody]
  simp only [rowLNFlat_flat, rowScaleFlat_flat, rowBiasFlat_flat]
  unfold vitForwardKVDrop classifierFlat
  simp only [Function.comp_apply, Mat.unflatten_flatten]
  rfl

/-- **The batched ViT forward graph with stochastic depth denotes `vitForwardKVDropB`** — at every
    depth, every pair of mask families and every input. -/
theorem vitFwdGraphBDrop_faithful {B ic H W patchSize N hm1 d mlpDim nClasses : Nat}
    (epsStr sStr : String)
    (Wc : Kernel4 ((hm1 + 1) * d) ic patchSize patchSize)
    (bc cls : Vec ((hm1 + 1) * d)) (pos : Mat (N + 1) ((hm1 + 1) * d))
    (ε : ℝ)
    (k : Nat) (ps : Fin k → BlockParamsV ((hm1 + 1) * d) mlpDim) (sdA sdM : Fin k → Vec B)
    (γF βF : Vec ((hm1 + 1) * d))
    (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (B * (ic * H * W))) :
    den (vitFwdGraphBDrop epsStr sStr ε (sdpaScale d) Wc bc cls pos k ps sdA sdM γF βF Wcls bcls x)
      = vitForwardKVDropB B ic H W patchSize N mlpDim (hm1 + 1) d nClasses k
          Wc bc cls pos ε ps sdA sdM γF βF Wcls bcls x := by
  funext idx
  have h := congrFun (vitFwdGraphBDrop_slice epsStr sStr Wc bc cls pos ε k ps sdA sdM γF βF
    Wcls bcls x (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2
  simp only [batchSlice, Prod.mk.eta, Equiv.apply_symm_apply] at h
  exact h

end Proofs.StableHLO
