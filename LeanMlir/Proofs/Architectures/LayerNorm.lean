import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Architectures.Activations

/-!
# LayerNorm

A quick chapter that extends the normalization family to what ViT needs: a
structural footnote to BatchNorm, not new territory — which is itself the point.
The smooth activations ViT and EfficientNet use (GELU, Swish, sigmoid) are in
`Architectures.Activations`, which this file re-exports.

## LayerNorm: BatchNorm on a different axis

BatchNorm reduces over `(batch, H, W)` for each channel. LayerNorm
reduces over the **feature** dimension for each `(batch, token)`.
**The 1D normalization primitive is literally the same function.** What
differs is the axis you slice along before applying it.

Concretely, for a 4D activation `x : Tensor4 B C H W`:
- BN computes `C` means/variances, each over `B · H · W` elements.
- LN computes `B · H · W` means/variances, each over `C` elements.

The mean/var/istd/xhat/affine math is identical. The consolidated
three-term backward is identical. Only the index being summed over
changes. In our `Vec n` formalism, BN and LN collapse to the same
function. This file just renames it to tell the reader "yes, really,
it's the same thing."

## Also in this file

- Layer scale: `layerScale`, `layerScaleHasVJP`.
- Vector-`[D]` LayerNorm (per-feature affine): `layerNormVec`, `layerNormVecHasVJP`,
  and its per-token lift `layerNormVecPerTokenHasVJPMat`.
- Its γ/β parameter gradients: `vecLNGradGamma`, `vecLNGradBeta`, the bridges
  `layerNormVec_gamma_grad_bridge` / `layerNormVec_beta_grad_bridge`, and their SGD forms
  `layerNormVec_gamma_sgd_certified` / `layerNormVec_beta_sgd_certified`.

Imported directly by SE, Attention, DropPath and Nets/ConvNeXt/ConvNeXt; the rest of the corpus
reaches it through them.

## References

- Ba, Kiros, Hinton 2016, *Layer Normalization*. <https://arxiv.org/abs/1607.06450>
- Touvron et al. 2021, *Going deeper with Image Transformers* (CaiT, layer-scale). <https://arxiv.org/abs/2103.17239>
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § LayerNorm
-- ════════════════════════════════════════════════════════════════

/-- **LayerNorm forward** — renamed `bnForward` to make the book's
    claim unambiguous: this is the same function, operating on a
    different slice of the tensor.

    For a single "token's feature vector" `x : Vec n`:
    1. `mu = (1/n) sum_i x_i`                         — mean across features
    2. `sigma^2 = (1/n) sum_i (x_i - mu)^2`            — variance across features
    3. `istd = 1/sqrt(sigma^2 + eps)`
    4. `xhat_i = (x_i - mu) * istd`                    — normalized
    5. `y_i = gamma * xhat_i + beta`                    — affine

    Here `γ` and `β` are scalars, exactly as in `bnForward`. The
    per-feature `[D]` affine that LN uses in practice is `layerNormVec`
    below, built as `γ ⊙ layerNormForward D ε 1 0 x + β`.

    MLIR (`MlirCodegen.emitLayerNormForward`):
    identical reduction structure to BN, just across a different axis. -/
noncomputable def layerNormForward (n : Nat) (ε : ℝ) (γ β : ℝ)
    (x : Vec n) : Vec n :=
  bnForward n ε γ β x

/-- **LayerNorm input gradient** — identical closed form to BN.

    `dx_i = (1/n) * istd * (n * dxhat_i - sum_j dxhat_j - xhat_i * sum_j xhat_j * dxhat_j)`

    where `dxhat_i = gamma * dy_i`.

    `layerNormForward` is `bnForward` by definition, so `layerNormHasVJP`
    is `bnHasVJP`.
-/
noncomputable def layerNormHasVJP (n : Nat) (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (layerNormForward n ε γ β) := by
  -- layerNormForward is definitionally bnForward, so the BN VJP works as-is.
  show HasVJP (bnForward n ε γ β)
  exact bnHasVJP n ε γ β hε

/-! ## Why this isn't a new chapter

The *practical* differences between BN and LN (batch dependence,
inference vs training, running statistics) are engineering concerns,
not VJP concerns. The backward pass is the same three-term formula
either way. This is a general lesson about formal work: engineering
distinctions often dissolve at the math level, and that's worth
making explicit. A reader who assumed BN and LN needed separate
proofs learns that the separation was an implementation artifact.

BN and LN share one `HasVJP` (`layerNormForward` is `bnForward`).
GroupNorm and InstanceNorm apply the same 1D primitive to other slices
of the tensor; RMSNorm drops the mean centering and is a different
function. None of those three is formalized here.
-/

/-- **Public correctness theorem for `layerNormHasVJP`**: LayerNorm
reuses the BN proof template (LayerNorm is BN on a different axis), so
the contract is identical — backward equals the `pdiv`-contracted
Jacobian of `layerNormForward`. -/
theorem layerNormHasVJP_correct (n : Nat) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (layerNormHasVJP n ε γ β hε).backward x dy i =
    ∑ j : Fin n, pdiv (layerNormForward n ε γ β) x i j * dy j :=
  (layerNormHasVJP n ε γ β hε).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § Layer scale (per-channel learnable elementwise scale)
-- ════════════════════════════════════════════════════════════════

/-- **Layer scale** — per-channel learnable elementwise multiply by `γ`.
    `layerScale γ x i = γ i * x i`. A diagonal linear map. -/
noncomputable def layerScale {n : Nat} (γ : Vec n) (x : Vec n) : Vec n :=
  fun i => γ i * x i

/-- `layerScale γ` is differentiable everywhere (diagonal linear). -/
theorem layerScale_differentiable {n : Nat} (γ : Vec n) :
    Differentiable ℝ (layerScale γ) := by
  unfold layerScale; fun_prop

/-- **Jacobian of `layerScale`** — `∂(γ_j x_j)/∂x_i = γ_i δ_{ij}`. -/
theorem pdiv_layerScale {n : Nat} (γ : Vec n) (x : Vec n) (i j : Fin n) :
    pdiv (layerScale γ) x i j = if i = j then γ i else 0 := by
  rw [pdiv_of_linear _ (fun _ _ => by funext; simp [layerScale, mul_add])
    (fun _ _ => by funext; simp [layerScale, mul_left_comm])]
  rcases eq_or_ne i j with rfl | h
  · simp [layerScale]
  · simp [layerScale, h, Ne.symm h]

/-- **Layer scale VJP**: `back(x, dy)_i = γ i * dy i`. -/
noncomputable def layerScaleHasVJP {n : Nat} (γ : Vec n) :
    HasVJP (layerScale γ) where
  backward := fun _x dy i => γ i * dy i
  correct := by
    intro x dy i
    simp [pdiv_layerScale]

-- ════════════════════════════════════════════════════════════════
-- § Vector-[D] LayerNorm — per-token forward + VJP, and its per-token (rowwise) lift
--   (ViT's LN sites and ConvNeXt's channel-LN both read it)
-- ════════════════════════════════════════════════════════════════

/-- **Vector-[D] LayerNorm**: per-token normalize (the scalar LN at γ=1, β=0 — pure
    x̂), then the per-channel affine `γ ⊙ x̂ + β`. The committed `ViTRender` LN form. -/
noncomputable def layerNormVec (D : Nat) (ε : ℝ) (γv βv : Vec D) (x : Vec D) : Vec D :=
  fun k => γv k * layerNormForward D ε 1 0 x k + βv k

lemma layerNormVec_differentiable (D : Nat) (ε : ℝ) (γv βv : Vec D) (hε : 0 < ε) :
    Differentiable ℝ (layerNormVec D ε γv βv) := by
  have h : Differentiable ℝ (layerNormForward D ε 1 0) := bnForward_differentiable D ε 1 0 hε
  unfold layerNormVec; fun_prop

/-- Identity-plus-constant Jacobian: `∂(p_k + C_k)/∂p_i = δ_(i,k)`. -/
theorem pdiv_id_add_const {m : Nat} (C : Vec m) (x : Vec m) (i j : Fin m) :
    pdiv (fun p : Vec m => fun k => p k + C k) x i j = if i = j then 1 else 0 := by
  rw [show (fun p : Vec m => fun k => p k + C k) = fun p => p + C from rfl,
    pdiv_of_affine _ _ (fun _ _ => rfl) (fun _ _ => rfl)]
  simp only [basisVec_apply, @eq_comm _ j i]

/-- Masked-gather-plus-constant Jacobian:
    `∂(mask_k·cl_(σ k) + C_k)/∂cl_i = mask_k·δ_(i,σ k)`. -/
theorem pdiv_maskGather_add_const {m D : Nat} (mask : Vec m) (σ : Fin m → Fin D)
    (C : Vec m) (x : Vec D) (i : Fin D) (j : Fin m) :
    pdiv (fun cl : Vec D => fun k => mask k * cl (σ k) + C k) x i j
      = mask j * (if i = σ j then 1 else 0) := by
  rw [show (fun cl : Vec D => fun k => mask k * cl (σ k) + C k)
      = fun cl => (fun k => mask k * cl (σ k)) + C from rfl,
    pdiv_of_affine _ _ (fun _ _ => by funext; simp [mul_add])
      (fun _ _ => by funext; simp [mul_left_comm])]
  simp only [basisVec_apply, @eq_comm _ (σ j) i]

/-- The bias translation's VJP — backward is the identity (`dx = dy`). -/
noncomputable def biasAddHasVJP {n : Nat} (βv : Vec n) :
    HasVJP (fun z : Vec n => fun k => z k + βv k) where
  backward := fun _z dy => dy
  correct := by
    intro z dy i
    simp [pdiv_id_add_const βv z]

/-- **Vector-LN VJP** — `(+β) ∘ layerScale γ ∘ LN(1,0)`, three proven pieces glued
    by `vjpComp`. Only `0 < ε`. -/
noncomputable def layerNormVecHasVJP (D : Nat) (ε : ℝ) (γv βv : Vec D)
    (hε : 0 < ε) : HasVJP (layerNormVec D ε γv βv) :=
  have h1 : Differentiable ℝ (layerNormForward D ε 1 0) :=
    bnForward_differentiable D ε 1 0 hε
  have h2 : Differentiable ℝ (layerScale γv) := layerScale_differentiable γv
  have h3 : Differentiable ℝ (fun z : Vec D => fun k => z k + βv k) := by fun_prop
  vjpComp _ (fun z : Vec D => fun k => z k + βv k) (h2.comp h1) h3
    (vjpComp (layerNormForward D ε 1 0) (layerScale γv) h1 h2
      (layerNormHasVJP D ε 1 0 hε) (layerScaleHasVJP γv))
    (biasAddHasVJP βv)

/-- Per-token vector-LN across a sequence — the rowwise lift. -/
noncomputable def layerNormVecPerTokenHasVJPMat (N D : Nat) (ε : ℝ)
    (γv βv : Vec D) (hε : 0 < ε) :
    HasVJPMat (fun X : Mat N D => fun r => layerNormVec D ε γv βv (X r)) :=
  rowwiseHasVJPMat (layerNormVecHasVJP D ε γv βv hε)
    (layerNormVec_differentiable D ε γv βv hε)

/-- Generic flat differentiability of a rowwise lift — each output coordinate
    is a coordinate of the per-row map applied to one row of the input. -/
lemma rowwise_flat_differentiable {N D P : Nat} (g : Vec D → Vec P)
    (hg : Differentiable ℝ g) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => g (X n)) (Mat.unflatten v))) := by
  fun_prop

lemma layerNormVec_per_token_flat_differentiable (N D : Nat) (ε : ℝ) (γv βv : Vec D)
    (hε : 0 < ε) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => layerNormVec D ε γv βv (X n))
                   (Mat.unflatten v))) :=
  rowwise_flat_differentiable _ (layerNormVec_differentiable D ε γv βv hε)

-- ════════════════════════════════════════════════════════════════
-- § Vector-LN γ/β parameter gradients
--
-- As a function of `γv : Vec D`, the rowwise vector-LN site is a coefficient-gather:
-- `y_(r,k) = x̂_r(k)·γv(k) + βv(k)` — the masked-gather Jacobian recipe
-- (`pdiv_maskGather_add_const`) with the per-row x̂ as the coefficient. The
-- per-channel grads keep the channel axis: `dγ_k = Σ_tokens dy_(r,k)·x̂_r(k)`,
-- `dβ_k = Σ_tokens dy_(r,k)` — `ViTRender`'s LN param-grad reduces.
-- ════════════════════════════════════════════════════════════════

/-- **Jacobian of the rowwise vector-LN site w.r.t. γv** —
    `∂y_(r,k)/∂γv_i = δ_(i,k)·x̂_r(k)`. -/
theorem pdiv_vecLN_gamma {N D : Nat} (ε : ℝ) (βv : Vec D) (X : Mat N D)
    (γ : Vec D) (i : Fin D) (o : Fin (N * D)) :
    pdiv (fun gv : Vec D =>
            Mat.flatten (fun r => layerNormVec D ε gv βv (X r))) γ i o
      = layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o).1)
          (finProdFinEquiv.symm o).2 *
        (if i = (finProdFinEquiv.symm o).2 then 1 else 0) := by
  rw [show (fun gv : Vec D => Mat.flatten (fun r => layerNormVec D ε gv βv (X r)))
        = (fun gv : Vec D => fun o' : Fin (N * D) =>
            (fun o'' : Fin (N * D) =>
              layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o'').1)
                (finProdFinEquiv.symm o'').2) o' *
              gv ((fun o'' : Fin (N * D) => (finProdFinEquiv.symm o'').2) o') +
            (fun o'' : Fin (N * D) =>
              βv (finProdFinEquiv.symm o'').2) o') from by
      funext gv o'
      unfold layerNormVec Mat.flatten
      ring]
  exact pdiv_maskGather_add_const _ _ _ γ i o

/-- **Jacobian of the rowwise vector-LN site w.r.t. βv** — `∂y_(r,k)/∂βv_i = δ_(i,k)`. -/
theorem pdiv_vecLN_beta {N D : Nat} (ε : ℝ) (γv : Vec D) (X : Mat N D)
    (β : Vec D) (i : Fin D) (o : Fin (N * D)) :
    pdiv (fun bv : Vec D =>
            Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o
      = if i = (finProdFinEquiv.symm o).2 then 1 else 0 := by
  rw [show (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γv bv (X r)))
        = fun bv => (fun o' : Fin (N * D) => bv (finProdFinEquiv.symm o').2) +
            fun o' => γv (finProdFinEquiv.symm o').2 *
              layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o').1)
                (finProdFinEquiv.symm o').2 from by
      funext bv o'
      unfold layerNormVec Mat.flatten
      exact add_comm _ _,
    pdiv_of_affine _ _ (fun _ _ => rfl) (fun _ _ => rfl)]
  simp [@eq_comm _ i]

/-- The rendered **vector-LN γ gradient**: per-channel, the batch+token reduce
    `dγ_k = Σ_r dY_(r,k)·x̂_r(k)` (KEEPS the channel axis — `ViTRender`'s form). -/
noncomputable def vecLNGradGamma (N D : Nat) (ε : ℝ) (X dY : Mat N D) : Vec D :=
  fun i => ∑ r : Fin N, dY r i * layerNormForward D ε 1 0 (X r) i

/-- The rendered **vector-LN β gradient**: `dβ_k = Σ_r dY_(r,k)`. -/
noncomputable def vecLNGradBeta (N D : Nat) (dY : Mat N D) : Vec D :=
  fun i => ∑ r : Fin N, dY r i

/-- **Vector-LN γ-gradient bridge.** -/
theorem layerNormVec_gamma_grad_bridge {N D : Nat} (ε : ℝ) (βv : Vec D) (γ : Vec D)
    (X : Mat N D) (dy : Vec (N * D)) (i : Fin D) :
    vecLNGradGamma N D ε X (Mat.unflatten dy) i
      = ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (X r))) γ i o
            * dy o := by
  simp_rw [pdiv_vecLN_gamma]
  rw [sum_finProdFinEquiv (m := N) (n := D)]
  simp [vecLNGradGamma, Mat.unflatten, mul_comm]

/-- **Vector-LN β-gradient bridge.** -/
theorem layerNormVec_beta_grad_bridge {N D : Nat} (ε : ℝ) (γv : Vec D) (β : Vec D)
    (X : Mat N D) (dy : Vec (N * D)) (i : Fin D) :
    vecLNGradBeta N D (Mat.unflatten dy) i
      = ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o
            * dy o := by
  simp_rw [pdiv_vecLN_beta]
  rw [sum_finProdFinEquiv (m := N) (n := D)]
  simp [vecLNGradBeta, Mat.unflatten]

/-- **Vector-LN γ output, certified.** `γvⁿ_k = γv_k − lr·(Σ_tokens dy·x̂)_k` denotes
    the certified rowwise vector-LN ∂/∂γv contraction: a rewrite by
    `layerNormVec_gamma_grad_bridge`, stated at one LN site with generic `N`, `D`. -/
theorem layerNormVec_gamma_sgd_certified {N D : Nat} (ε : ℝ) (βv : Vec D)
    (γ : Vec D) (X : Mat N D) (dy : Vec (N * D)) (lr : ℝ) (i : Fin D) :
    γ i - lr * vecLNGradGamma N D ε X (Mat.unflatten dy) i
      = γ i - lr * ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (X r))) γ i o
            * dy o := by
  rw [layerNormVec_gamma_grad_bridge ε βv γ X dy i]

/-- **Vector-LN β output, certified.** -/
theorem layerNormVec_beta_sgd_certified {N D : Nat} (ε : ℝ) (γv : Vec D)
    (β : Vec D) (X : Mat N D) (dy : Vec (N * D)) (lr : ℝ) (i : Fin D) :
    β i - lr * vecLNGradBeta N D (Mat.unflatten dy) i
      = β i - lr * ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o
            * dy o := by
  rw [layerNormVec_beta_grad_bridge ε γv β X dy i]

end Proofs
