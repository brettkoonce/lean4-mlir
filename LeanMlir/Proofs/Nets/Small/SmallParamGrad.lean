import LeanMlir.Proofs.Foundation.ParamGrad
import LeanMlir.Proofs.Foundation.SgdNodes
import LeanMlir.Proofs.Foundation.SmoothedLossCot

/-! # SmallParamGrad — the per-example kit for the chapter nets' loss gradients

The seven ImageNet nets state their parameter gradients through `ParamGradNodes`, at the batched
`*GradB` nodes. The chapter nets (linear, MLP, MNIST CNN, the CIFAR CNNs) run one example at a time
and emit the per-example `GradNode` ops (`SgdNodes`). This file is their kit:

* **Per node kind.** `convW_hasGradAt`, `convB_hasGradAt`, `denseW_hasGradAt`, `denseB_hasGradAt`:
  a node fed the gradient of `G` at its op's output is `∂G/∂θ` with that one parameter varied.
  The fused `*Sgd` ops the SGD renders emit are `θ − lr·` these nodes (`convWeightSgd_eq_grad`,
  `convBiasSgd_eq_grad`, `weightSgd_eq_grad`, `biasSgd_eq_grad`).
* **Per stage.** `hasGradAt_dense`, `hasGradAt_relu`, `hasGradAt_conv`: the loss read at a stage's
  output, pulled back through the stage's certified VJP, lands on the emitted chain's own cotangent.
* **The 2×2 pool, through a fixed selection.** `σ` names one cell of each window
  (`poolSelIdx`); the pool with its routing frozen at `σ` is a gather, linear and differentiable
  everywhere (`hasGradAt_gatherRelu`), and its backward routes each window's cotangent to that ONE
  cell (`selScatter`) — what the rendered `select_and_scatter` (select = `GE`) does.
  `maxPool_relu_eventuallyEq_sel` is why that is the loss's gradient in the parameters at a tied
  window: if every window is dead, or tied only between cells that are the same function of the
  moving parameter (`MaxPool2SmoothUpTo`), the pooled ReLU IS the gather along the parameter, near
  the point.
* **The loss.** `hasGradAt_crossEntropy`: softmax cross-entropy at a hard label has gradient
  `softmax − onehot` in the logits.

Which cells are twins is per net: each net file names them (cells equal at every value of the
weights upstream of the pool) and discharges `maxPool_relu_eventuallyEq_sel` along each
parameter.
-/

namespace Proofs.SmallParamGrad

open Proofs Proofs.StableHLO Proofs.IR
open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The loss
-- ════════════════════════════════════════════════════════════════

/-- **Softmax cross-entropy at a hard label has gradient `softmax − onehot`** in the logits. -/
theorem hasGradAt_crossEntropy {n : Nat} (label : Fin n) (z : Vec n) :
    HasGradAt (fun z' (_ : Fin 1) => crossEntropy n z' label) z
      (fun j => softmax n z j - oneHot n label j) :=
  ⟨differentiable_pi.mpr (fun _ => crossEntropy_differentiable n label) z,
    fun j => softmaxCE_grad n z label j⟩

-- ════════════════════════════════════════════════════════════════
-- § Per node kind: a gradient node fed `∇G` at its output is `∇_θ G`
-- ════════════════════════════════════════════════════════════════

/-- **Conv weight node = `∇_W G`.** -/
theorem convW_hasGradAt {ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (ic * h * w)) (W : Kernel4 oc ic kH kW) {G : Vec (oc * h * w) → Vec 1}
    {c : Vec (oc * h * w)} (hG : HasGradAt G (flatConv W b x) c) :
    HasGradAt (fun θ => G (flatConv (Kernel4.unflatten θ) b x)) (Kernel4.flatten W)
      (den (SHlo.convWeightGrad xN b (Tensor3.unflatten x) W (.operand cotN c))) := by
  have hG' : HasGradAt G (flatConv (Kernel4.unflatten (Kernel4.flatten W)) b x) c := by
    rwa [Kernel4.unflatten_flatten]
  have hP := hG'.param (layer := fun θ => flatConv (Kernel4.unflatten θ) b x)
    ((conv2d_weight_differentiable b (Tensor3.unflatten x)) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  rw [hP.pdiv_eq idx, GradNode.convWGrad_den]
  rfl

/-- **Conv bias node = `∇_b G`.** -/
theorem convB_hasGradAt {ic oc h w kH kW : Nat} (cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Vec (ic * h * w)) (b : Vec oc) {G : Vec (oc * h * w) → Vec 1}
    {c : Vec (oc * h * w)} (hG : HasGradAt G (flatConv W b x) c) :
    HasGradAt (fun θ => G (flatConv W θ x)) b
      (den (SHlo.convBiasGrad W (Tensor3.unflatten x) b (.operand cotN c))) := by
  have hP := hG.param (layer := fun θ => flatConv W θ x)
    ((conv2d_bias_differentiable W (Tensor3.unflatten x)) _)
  refine ⟨hP.differentiableAt, fun o => ?_⟩
  rw [hP.pdiv_eq o, GradNode.convBGrad_den]
  rfl

/-- **Dense weight node = `∇_W G`.** -/
theorem denseW_hasGradAt {m n : Nat} (aN cotN : String) (a : Vec m) (W : Mat m n) (b : Vec n)
    {G : Vec n → Vec 1} {c : Vec n} (hG : HasGradAt G (dense W b a) c) :
    HasGradAt (fun θ => G (dense (Mat.unflatten θ) b a)) (Mat.flatten W)
      (den (SHlo.weightGrad aN a (.operand cotN c))) := by
  have hG' : HasGradAt G (dense (Mat.unflatten (Mat.flatten W)) b a) c := by
    rwa [Mat.unflatten_flatten]
  have hP := hG'.param (layer := fun θ => dense (Mat.unflatten θ) b a)
    ((StableHLO.denseWeightMap_differentiable b a) _)
  refine ⟨hP.differentiableAt, fun idx => ?_⟩
  obtain ⟨ij, rfl⟩ := finProdFinEquiv.surjective idx
  rw [hP.pdiv_eq, GradNode.denseWGrad_den aN cotN a W b c ij.1 ij.2]

/-- **Dense bias node = `∇_b G`.** -/
theorem denseB_hasGradAt {m n : Nat} (cotN : String) (W : Mat m n) (a : Vec m) (b : Vec n)
    {G : Vec n → Vec 1} {c : Vec n} (hG : HasGradAt G (dense W b a) c) :
    HasGradAt (fun θ => G (dense W θ a)) b (den (SHlo.biasGrad (.operand cotN c))) := by
  have hP := hG.param (layer := fun θ => dense W θ a) (by unfold dense; fun_prop)
  refine ⟨hP.differentiableAt, fun i => ?_⟩
  rw [hP.pdiv_eq i, GradNode.denseBGrad_den cotN W a b c i]

/-! The fused `*Sgd` op is `θ − lr·` its un-fused `*Grad` peer, at any cotangent: the SGD renders
(the linear, MLP, MNIST-CNN and CIFAR arms) step by exactly the node the lemmas above identify. -/

theorem convWeightSgd_eq_grad {ic oc h w kH kW : Nat} (xN wN lrStr cotN : String) (b : Vec oc)
    (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (c : Vec (oc * h * w)) (lr : ℝ)
    (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * den (SHlo.convWeightGrad xN b x W (.operand cotN c)) idx := by
  rw [SgdNode.convW_den, GradNode.convWGrad_den]

theorem convBiasSgd_eq_grad {ic oc h w kH kW : Nat} (bN lrStr cotN : String)
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) (c : Vec (oc * h * w)) (lr : ℝ)
    (o : Fin oc) :
    den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * den (SHlo.convBiasGrad W x b (.operand cotN c)) o := by
  rw [SgdNode.convB_den, GradNode.convBGrad_den]

theorem weightSgd_eq_grad {m n : Nat} (aN wN lrStr cotN : String) (a : Vec m) (W : Mat m n)
    (b : Vec n) (c : Vec n) (lr : ℝ) (i : Fin m) (j : Fin n) :
    den (SHlo.weightSgd aN wN lrStr a W lr (.operand cotN c)) (finProdFinEquiv (i, j))
      = W i j - lr * den (SHlo.weightGrad aN a (.operand cotN c)) (finProdFinEquiv (i, j)) := by
  rw [SgdNode.denseW_den aN wN lrStr cotN a W b c lr i j, GradNode.denseWGrad_den aN cotN a W b c i j]

theorem biasSgd_eq_grad {m n : Nat} (bN lrStr cotN : String) (W : Mat m n) (a : Vec m)
    (b : Vec n) (c : Vec n) (lr : ℝ) (i : Fin n) :
    den (SHlo.biasSgd bN lrStr b lr (.operand cotN c)) i
      = b i - lr * den (SHlo.biasGrad (.operand cotN c)) i := by
  rw [SgdNode.denseB_den bN lrStr cotN W a b c lr i, GradNode.denseBGrad_den cotN W a b c i]

-- ════════════════════════════════════════════════════════════════
-- § Per stage: pull the loss gradient back onto the emitted chain's cotangent
-- ════════════════════════════════════════════════════════════════

/-- Through a dense layer: the backward is `W · dy` (`emitDenseBack`). -/
theorem hasGradAt_dense {m n : Nat} (W : Mat m n) (b : Vec n) (u : Vec m)
    {G : Vec n → Vec 1} {dy : Vec n} (hG : HasGradAt G (dense W b u) dy) :
    HasGradAt (fun y => G (dense W b y)) u ((emitDenseBack W).denote dy) :=
  hG.comp ((dense_differentiable W b) u) ((denseHasVJP W b).toHasVJPAt u)

/-- Through a ReLU off its kink: the backward is the mask `relu'(z) ⊙ dy` (`emitReluBack`). -/
theorem hasGradAt_relu {n : Nat} (z : Vec n) (hz : ∀ k, z k ≠ 0) {G : Vec n → Vec 1}
    {dy : Vec n} (hG : HasGradAt G (relu n z) dy) :
    HasGradAt (fun y => G (relu n y)) z ((emitReluBack z).denote dy) :=
  hG.comp (relu_differentiableAt_of_smooth n z hz) (reluHasVJPAt n z hz)

/-- Through a stride-1 conv with odd kernels: the backward is the rendered reversed-kernel conv
    (`Back3.conv`, `conv_flatten_bridge`). -/
theorem hasGradAt_conv {ic oc h w kH kW : Nat} (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW) (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (v : Vec (ic * h * w)) {G : Vec (oc * h * w) → Vec 1} {dy : Vec (oc * h * w)}
    (hG : HasGradAt G (flatConv W b v) dy) :
    HasGradAt (fun y => G (flatConv W b y)) v
      ((Back3.conv (c₁ := oc) (h₁ := h) (w₁ := w) W Back3.cot).flatDenote dy) :=
  (hG.comp ((flatConv_differentiable W b) v)
    ((HasVJP3.toHasVJP (conv2dHasVJP3 W b)).toHasVJPAt v)).of_eq
    (conv_flatten_bridge hkH hkW W b v dy).symm

-- ════════════════════════════════════════════════════════════════
-- § The 2×2 pool at a fixed selection
-- ════════════════════════════════════════════════════════════════

/-- The flat index of the cell `σ` selects in pooled entry `k`'s window. -/
def poolSelIdx {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) (k : Fin (c * h * w)) :
    Fin (c * (2 * h) * (2 * w)) :=
  t3Idx (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1
    (winRowInv (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2
      (σ (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1
        (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2 (finProdFinEquiv.symm k).2).1)
    (winColInv (finProdFinEquiv.symm k).2
      (σ (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1
        (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2 (finProdFinEquiv.symm k).2).2)

theorem poolSelIdx_t3Idx {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) (ci : Fin c)
    (ho : Fin h) (wo : Fin w) :
    poolSelIdx σ (t3Idx ci ho wo)
      = t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2) := by
  simp only [poolSelIdx, t3Idx, Equiv.symm_apply_apply]

/-- `poolGatherFlat` is the reindex along `poolSelIdx`. -/
theorem poolGatherFlat_eq_sel {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (u : Vec (c * (2 * h) * (2 * w))) : poolGatherFlat σ u = fun k => u (poolSelIdx σ k) := by
  funext k
  obtain ⟨ci, ho, wo, rfl⟩ := t3Idx_surj k
  rw [poolGatherFlat_apply, poolSelIdx_t3Idx]

/-- **The selection names a maximum of every window** of `u`. The rendered `select_and_scatter`
    (select = `GE`) makes such a choice; the canonical argmax is one (`poolSelDom_argmax`). -/
def PoolSelDom {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (u : Vec (c * (2 * h) * (2 * w))) : Prop :=
  ∀ (ci : Fin c) (ho : Fin h) (wo : Fin w) (cd : Fin 2 × Fin 2),
    u (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2))
      ≤ u (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2))

theorem poolSelDom_argmax {c h w : Nat} (u : Vec (c * (2 * h) * (2 * w))) :
    PoolSelDom (maxPool2Argmax (Tensor3.unflatten u : Tensor3 c (2 * h) (2 * w))) u :=
  fun ci ho wo cd => maxPool2Argmax_max (Tensor3.unflatten u) ci ho wo cd

/-- **Smooth, dead, or tied only between twins** at the 2×2 windows (`WindowSmoothUpTo`), on the
    pool's PRE-activation: a window is dead when its cells are all `≤ 0`. The pre-activation margin
    `MaxPool2MarginQUpTo δ T` implies it at any `δ ≥ 0` (`windowSmoothUpTo_of_margin`). -/
abbrev MaxPool2SmoothUpTo {c h w : Nat}
    (T : Fin (2 * h) × Fin (2 * w) → Fin (2 * h) × Fin (2 * w) → Prop)
    (x : Tensor3 c (2 * h) (2 * w)) : Prop :=
  WindowSmoothUpTo winRowInv winColInv T x

/-- The 2×2 pool is continuous (a max of coordinates), so a pre-activation computed through
    earlier pools moves continuously with the parameters. -/
theorem maxPoolFlat_continuous (c h w : Nat) : Continuous (maxPoolFlat c h w) := by
  rw [maxPoolFlat_eq_windowMaxFlat]; exact windowMaxFlat_continuous _ _

/-- The scatter along a selection: each pooled cotangent lands on the one cell it read. -/
def selScatter {m n : Nat} (σ : Fin n → Fin m) (dy : Vec n) : Vec m :=
  fun i => ∑ k : Fin n, if i = σ k then dy k else 0

/-- Through ReLU then a gather (`y ↦ relu y ∘ σ`), off the ReLU kinks: the backward is the scatter
    along `σ`, then the ReLU mask. No pool hypothesis: the gather is linear. -/
theorem hasGradAt_gatherRelu {m n : Nat} (σ : Fin n → Fin m) (z : Vec m) (hz : ∀ k, z k ≠ 0)
    {G : Vec n → Vec 1} {dy : Vec n} (hG : HasGradAt G (fun k => relu m z (σ k)) dy) :
    HasGradAt (fun y => G (fun k => relu m y (σ k))) z
      ((emitReluBack z).denote (selScatter σ dy)) := by
  have hR : HasGradAt (fun u => G (fun k => u (σ k))) (relu m z) (selScatter σ dy) :=
    (hG.comp (f := fun u k => u (σ k)) (x := relu m z) (reindexCLM σ).differentiableAt
      ((reindexVJP σ).toHasVJPAt (relu m z))).of_eq (reindexVJP_backward σ (relu m z) dy)
  exact hasGradAt_relu z hz hR

/-- **Along a parameter, ReLU then the 2×2 pool is the gather at a fixed selection.** `Z θ` is the
    pool's pre-activation as the parameter moves; at `θ₀` it has no zero entry, every window is
    dead or tied only between `T`-twins, `σ` names a maximum of every window of its ReLU, and
    `T`-twins are equal at EVERY `θ`. Then near `θ₀` the pooled ReLU reads each window at `σ`: a
    dead window stays negative, a strict maximum stays strict, and a twin stays tied with it. -/
theorem maxPool_relu_eventuallyEq_sel {P c h w : Nat} (Z : Vec P → Vec (c * (2 * h) * (2 * w)))
    (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (T : Fin (2 * h) × Fin (2 * w) → Fin (2 * h) × Fin (2 * w) → Prop)
    (hT : ∀ θ (ci : Fin c) (p q : Fin (2 * h) × Fin (2 * w)), T p q →
      Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2))
    (θ₀ : Vec P) (hZc : ContinuousAt Z θ₀) (hz : ∀ k, Z θ₀ k ≠ 0)
    (hs : MaxPool2SmoothUpTo T (Tensor3.unflatten (Z θ₀) : Tensor3 c (2 * h) (2 * w)))
    (hσ : PoolSelDom σ (relu _ (Z θ₀))) :
    ∀ᶠ θ in nhds θ₀,
      maxPoolFlat c h w (relu _ (Z θ)) = fun k => relu _ (Z θ) (poolSelIdx σ k) := by
  have hc : ∀ k, ContinuousAt (fun θ => Z θ k) θ₀ :=
    fun k => (continuous_apply k).continuousAt.comp hZc
  have hev : ∀ᶠ θ in nhds θ₀, ∀ (ci : Fin c) (ho : Fin h) (wo : Fin w) (cd : Fin 2 × Fin 2),
      relu _ (Z θ) (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) ≤
        relu _ (Z θ) (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2)) := by
    simp only [Filter.eventually_all, relu_apply_eq_max]
    intro ci ho wo cd
    by_cases hdead : ∀ cd' : Fin 2 × Fin 2,
        Z θ₀ (t3Idx ci (winRowInv ho cd'.1) (winColInv wo cd'.2)) ≤ 0
    · -- a dead window: the cell is strictly negative, and stays so
      have hlt := lt_of_le_of_ne (hdead cd) (hz _)
      filter_upwards [(hc _).eventually_lt continuousAt_const hlt] with θ h1
      exact max_le (h1.le.trans (le_max_right _ _)) (le_max_right _ _)
    obtain ⟨cd0, hcd0⟩ : ∃ cd0 : Fin 2 × Fin 2,
        0 < Z θ₀ (t3Idx ci (winRowInv ho cd0.1) (winColInv wo cd0.2)) := by
      push Not at hdead; exact hdead
    -- a live window: `σ` dominates the pre-activation too
    have hdom : ∀ cd' : Fin 2 × Fin 2,
        Z θ₀ (t3Idx ci (winRowInv ho cd'.1) (winColInv wo cd'.2)) ≤
          Z θ₀ (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2)) := by
      have h0 := hσ ci ho wo cd0
      rw [relu_apply_eq_max, relu_apply_eq_max, max_eq_left hcd0.le] at h0
      have hpos : 0 < Z θ₀ (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2)) := by
        have h0' := lt_of_lt_of_le hcd0 h0
        rcases le_or_gt (Z θ₀ (t3Idx ci (winRowInv ho (σ ci ho wo).1)
            (winColInv wo (σ ci ho wo).2))) 0 with hle | hgt
        · rw [max_eq_right hle] at h0'; exact absurd h0' (lt_irrefl 0)
        · exact hgt
      intro cd'
      have h1 := hσ ci ho wo cd'
      rw [relu_apply_eq_max, relu_apply_eq_max, max_eq_left hpos.le] at h1
      exact (le_max_left _ _).trans h1
    by_cases hpos : (winRowInv ho cd.1, winColInv wo cd.2) =
        (winRowInv ho (σ ci ho wo).1, winColInv wo (σ ci ho wo).2)
    · obtain ⟨hr, hw⟩ := Prod.mk.inj hpos
      exact Filter.Eventually.of_forall fun _ => by rw [hr, hw]
    rcases hs ci ho wo with hd | hsm
    · exact absurd (hd cd0) (not_le.mpr hcd0)
    rcases hsm (σ ci ho wo) cd (Ne.symm hpos) hdom with hlt | htw
    · filter_upwards [(hc _).eventually_lt (hc _) hlt] with θ h
      exact max_le_max h.le le_rfl
    · exact Filter.Eventually.of_forall fun θ => by rw [hT θ ci _ _ htw]
  filter_upwards [hev] with θ hθ
  rw [maxPoolFlat_eq_poolGatherFlat σ _ hθ, poolGatherFlat_eq_sel]

/-- **The step ties' pool backward is the scatter at the first argmax.** The `Back3` maxpool node
    (`maxPoolBackDenote`, the `den` of the rendered `maxPoolBack`) routes each window's cotangent
    to `maxPool2Argmax`'s cell, the window's first maximum, so through the flatten it is
    `selScatter` along that selection, at every point, ties included. With `poolSelDom_argmax`,
    the chain the step ties read is the loss-gradient chain at `σ = maxPool2Argmax`. -/
theorem maxpool_flatDenote_eq_selScatter {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (dy : Vec (c * h * w)) :
    (Back3.maxpool (c₁ := c) (h₁ := h) (w₁ := w) x Back3.cot).flatDenote dy
      = selScatter (poolSelIdx (maxPool2Argmax x)) dy := by
  funext i
  obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj i
  have key : ∀ A : Fin 2 × Fin 2,
      t3Idx ci hi wi = t3Idx ci (winRowInv (winRow hi) A.1) (winColInv (winCol wi) A.2)
        ↔ A = (winRowMod hi, winColMod wi) := by
    rintro ⟨a, b⟩
    simp only [t3Idx, EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq, true_and]
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨?_, ?_⟩
      · rw [h1, winRowMod_winRowInv]
      · rw [h2, winColMod_winColInv]
    · rintro ⟨rfl, rfl⟩
      exact ⟨(winRowInv_winRow hi).symm, (winColInv_winCol wi).symm⟩
  simp only [Back3.flatDenote, Back3.denote]
  rw [flatten_t3Idx, selScatter, Finset.sum_eq_single (t3Idx ci (winRow hi) (winCol wi))]
  · rw [poolSelIdx_t3Idx]
    exact if_congr (key _).symm rfl rfl
  · intro k _ hk
    obtain ⟨co, ho, wo, rfl⟩ := t3Idx_surj k
    rw [poolSelIdx_t3Idx]
    refine ite_eq_right_iff.mpr fun heq => ?_
    simp only [t3Idx, EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq] at heq
    obtain ⟨⟨hc, h1⟩, h2⟩ := heq
    exact absurd (by rw [h1, h2, winRow_winRowInv, winCol_winColInv, hc]) hk
  · exact fun h => absurd (Finset.mem_univ _) h

end Proofs.SmallParamGrad
