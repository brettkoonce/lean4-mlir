import LeanMlir.Proofs.MobileNetV2Close
import LeanMlir.Proofs.ConvNeXt

/-! # Closing the ConvNeXt render — the parameter-gradient close (two small new families)

`planning/convnext_close.md` Item C, applied to the representative 2-block ConvNeXt
(`tests/TestConvNeXtTrain.lean`; `convNextForward`, the proven whole-net VJP `convnext_has_vjp`).
The close is generic in the cotangent `dy` the backward chain delivers at each layer's output
(pinning that cotangent to the actual block chain is the optional Item D), batch-1 — LayerNorm is
per-example separable, so no batched apparatus (the EfficientNet contrast).

| family (render SSA)                       | forward fn            | certified by                                   |
|-------------------------------------------|-----------------------|------------------------------------------------|
| stem/expand/project 1×1 conv W/b          | `conv2d` (stride 1)   | `cnn_render_conv{W,b}_certified` (M3, **reuse**) |
| depthwise **7×7** W/b                     | `depthwiseConv2d`     | `cnx_render_dw7{W,b}_certified` (kernel-general — **pinned below**) |
| dense head `Wd`/`bd`                      | matmul / +bias        | M2 `weight/bias_grad_bridge` (**reuse**)       |
| **layer-scale `γ`** (`layerScaleF`)       | `γ ⊙ x`               | `cnx_render_lsgamma_certified` (**new**): `∂(γⱼxⱼ)/∂γᵢ = xᵢδᵢⱼ` ⇒ `dγ = x ⊙ dy` |
| **scalar-LN `γ/β`** (the `bnF` sites)     | `layerNormForward`    | `cnx_render_ln{gamma,beta}_certified` (**new**): the `Vec 1` embedding |
| gelu                                      | —                     | no parameters                                  |

Two genuinely-new bridge families:

* **Layer-scale `γ`** — `layerScale γ x = γ ⊙ x` is *symmetric* in `(γ, x)`, so the γ-Jacobian is
  the same diagonal as the input-Jacobian with the roles swapped: `∂(γⱼxⱼ)/∂γᵢ = xᵢ·δᵢⱼ`
  (`pdiv_layerScale_gamma`, the mirror of `pdiv_layerScale`), giving the rendered `dγ = x ⊙ dy`.
* **Scalar-LN `γ/β`** — the proof's LayerNorm (`layerNormForward = bnForward`) has *scalar* `γ β : ℝ`,
  and `BatchNorm.lean` deliberately left `bn_grad_gamma`/`bn_grad_beta` as definitions ("scalar
  params don't fit the `pdiv`/`HasVJP` framework cleanly"). This file closes that gap by embedding
  the scalar as `Vec 1`: as a function of `γ' : Vec 1`, LN is affine —
  `γ' ↦ fun k => x̂ₖ · γ'(0) + β` — so its Jacobian collapses through
  `pdiv_add`/`pdiv_mul`/`pdiv_const`/`pdiv_reindex` (the `CifarBnClose` recipe with the constant
  channel map `Fin n → Fin 1`) to `∂yₖ/∂γ = x̂ₖ` (resp. `1` for β), certifying the rendered whole-`n`
  reduces `dγ = Σ dy·x̂`, `dβ = Σ dy` (`bn_grad_gamma`/`bn_grad_beta` — now bridged, not just defined).
  Affine in the params, so no `0 < ε` needed (ε only enters the constant x̂).

The 7×7 depthwise is the only kernel size to pin (prior nets used 3×3/5×5; the generic
`mnv2_render_depthwise{W,b}_certified` is kernel-general — stride-1 only, ConvNeXt blocks don't
downsample). The 1×1 convs (stem/expand/project) and the dense head are verbatim M3/M2 reuse at the
ConvNeXt shapes. 3-axiom clean by inheritance.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § A. 7×7 depthwise pins (the kernel size no prior net exercised; stride-1)
-- ════════════════════════════════════════════════════════════════

/-- **7×7 depthwise weight output, certified.** The generic depthwise weight bridge at `kH=kW=7`;
    covers every ConvNeXt block's depthwise (stride-1 — ConvNeXt blocks keep resolution). -/
theorem cnx_render_dw7W_certified {c h w : Nat}
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c 7 7) (dy : Tensor3 c h w) (lr : ℝ)
    (ci : Fin c) (hi : Fin 7) (wi : Fin 7) :
    W ci hi wi - lr * (depthwise_weight_grad_has_vjp3 b x).backward W dy ci hi wi
      = W ci hi wi - lr * ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          pdiv3 (fun W' : DepthwiseKernel c 7 7 => depthwiseConv2d W' b x) W ci hi wi co ho wo
            * dy co ho wo :=
  mnv2_render_depthwiseW_certified b x W dy lr ci hi wi

/-- **7×7 depthwise bias output, certified.** -/
theorem cnx_render_dw7b_certified {c h w : Nat}
    (W : DepthwiseKernel c 7 7) (x : Tensor3 c h w) (b : Vec c) (dy : Vec (c * h * w)) (lr : ℝ)
    (cc : Fin c) :
    b cc - lr * (depthwise_bias_grad_has_vjp W x).backward b dy cc
      = b cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc j * dy j :=
  mnv2_render_depthwiseb_certified W x b dy lr cc

-- ════════════════════════════════════════════════════════════════
-- § B. Layer-scale γ — the multiplicative-bias bridge (genuinely new)
--
-- `layerScale γ x = γ ⊙ x` is symmetric in (γ, x): viewed as a function of γ (x fixed), it is
-- the same diagonal linear map with the roles swapped. `pdiv_layerScale` (ConvNeXt.lean) gives
-- the input-Jacobian γᵢδᵢⱼ; this gives the parameter-Jacobian xᵢδᵢⱼ, hence the rendered
-- `dγ = x ⊙ dy`.
-- ════════════════════════════════════════════════════════════════

/-- **Jacobian of `layerScale` w.r.t. γ** — `∂(γ_j x_j)/∂γ_i = x_i δ_{ij}`. The parameter mirror
    of `pdiv_layerScale`. -/
theorem pdiv_layerScale_gamma {n : Nat} (x : Vec n) (γ : Vec n) (i j : Fin n) :
    pdiv (fun γ' : Vec n => layerScale γ' x) γ i j = if i = j then x i else 0 := by
  have h_eq : (fun γ' : Vec n => layerScale γ' x) =
      (fun y : Vec n => fun k => (fun w : Vec n => w) y k * (fun _ : Vec n => x) y k) := by
    funext y k; rfl
  rw [h_eq]
  rw [pdiv_mul (fun w : Vec n => w) (fun _ : Vec n => x) γ
        differentiableAt_id (differentiableAt_const x) i j]
  rw [pdiv_id, pdiv_const]
  by_cases hij : i = j
  · subst hij; simp
  · rw [if_neg hij, if_neg hij]; ring

/-- The rendered **layer-scale γ gradient**: `dγ_i = x_i · dy_i` (elementwise multiply of the
    saved layer input with the cotangent — the `layerScaleF`-shaped backward). -/
noncomputable def layerScale_grad_gamma {n : Nat} (x dy : Vec n) : Vec n :=
  fun i => x i * dy i

/-- **Layer-scale γ-gradient bridge.** The rendered `dγ = x ⊙ dy` equals the certified Jacobian
    of `layerScale` (as a function of γ) contracted with the cotangent `dy`. -/
theorem layerScale_gamma_grad_bridge {n : Nat} (x : Vec n) (γ : Vec n) (dy : Vec n) (i : Fin n) :
    layerScale_grad_gamma x dy i
      = ∑ j : Fin n, pdiv (fun γ' : Vec n => layerScale γ' x) γ i j * dy j := by
  simp_rw [pdiv_layerScale_gamma]
  rw [Finset.sum_eq_single i]
  · rw [if_pos rfl]; rfl
  · intro b _ hne; rw [if_neg (fun h => hne h.symm)]; ring
  · intro hp; exact absurd (Finset.mem_univ _) hp

/-- **Layer-scale γ output, certified.** `γⁿ = γ − lr·(x ⊙ dy)` denotes
    `γ − lr·(certified ∂(layerScale)/∂γ · cotangent)`. The multiplicative-bias peer of
    `cnn_render_convb_certified`. -/
theorem cnx_render_lsgamma_certified {n : Nat} (x : Vec n) (γ : Vec n) (dy : Vec n) (lr : ℝ)
    (i : Fin n) :
    γ i - lr * layerScale_grad_gamma x dy i
      = γ i - lr * ∑ j : Fin n, pdiv (fun γ' : Vec n => layerScale γ' x) γ i j * dy j := by
  rw [layerScale_gamma_grad_bridge]

-- ════════════════════════════════════════════════════════════════
-- § C. Scalar-LN γ/β — the Vec-1 embedding of the scalar parameters (genuinely new)
--
-- `layerNormForward n ε γ β` has scalar `γ β : ℝ`; `bn_grad_gamma`/`bn_grad_beta`
-- (BatchNorm.lean) are the rendered whole-`n` reduces `Σ dy·x̂` / `Σ dy`, stated there as
-- *definitions only* because scalar params fall outside the `Vec`-indexed `pdiv` framework.
-- Embedding the scalar as `Vec 1` brings them inside: as a function of `γ' : Vec 1`, LN is
-- affine (`fun y k => x̂_k · y 0 + β`), so the CifarBnClose Jacobian recipe applies with the
-- constant channel map `Fin n → Fin 1`. Affine in the params ⇒ no `0 < ε` hypothesis.
-- ════════════════════════════════════════════════════════════════

/-- scalar-LN as a function of γ (β, x fixed), written affinely:
    `γ' ↦ fun k => x̂_k · γ'(0) + β`. -/
private theorem layerNorm_gamma_affine (n : Nat) (ε β : ℝ) (x : Vec n) :
    (fun γ' : Vec 1 => layerNormForward n ε (γ' 0) β x)
      = fun y k => bnXhat n ε x k * y ((fun _ : Fin n => (0 : Fin 1)) k) + β := by
  funext y k
  simp only [layerNormForward, bnForward]
  ring

/-- scalar-LN as a function of β (γ, x fixed):
    `β' ↦ fun k => γ·x̂_k + β'(0)`. -/
private theorem layerNorm_beta_affine (n : Nat) (ε γ : ℝ) (x : Vec n) :
    (fun β' : Vec 1 => layerNormForward n ε γ (β' 0) x)
      = fun y k => γ * bnXhat n ε x k + y ((fun _ : Fin n => (0 : Fin 1)) k) := by
  funext y k
  simp only [layerNormForward, bnForward]

/-- **Jacobian of scalar-LN w.r.t. γ** — `∂y_j/∂γ = x̂_j` (dense in `j`: the scalar γ scales
    every output). The scalar special-case of `pdiv_bnPerChannelFlat_gamma`. -/
private theorem pdiv_layerNorm_gamma (n : Nat) (ε β : ℝ) (x : Vec n) (γ : Vec 1)
    (i : Fin 1) (j : Fin n) :
    pdiv (fun γ' : Vec 1 => layerNormForward n ε (γ' 0) β x) γ i j = bnXhat n ε x j := by
  rw [layerNorm_gamma_affine]
  set XH : Vec n := bnXhat n ε x with hXH
  set BC : Vec n := fun _ => β with hBC
  have hmul_diff : DifferentiableAt ℝ
      (fun y : Vec 1 => fun k : Fin n => XH k * y ((fun _ : Fin n => (0 : Fin 1)) k)) γ :=
    (differentiableAt_const XH).mul
      ((reindexCLM (fun _ : Fin n => (0 : Fin 1))).differentiableAt)
  have hconst_diff : DifferentiableAt ℝ (fun _ : Vec 1 => BC) γ := differentiableAt_const _
  rw [show (fun (y : Vec 1) (k : Fin n) => XH k * y ((fun _ : Fin n => (0 : Fin 1)) k) + β)
        = (fun y k => (fun y' k' => XH k' * y' ((fun _ : Fin n => (0 : Fin 1)) k')) y k
                      + (fun _ k' => BC k') y k) from rfl,
      pdiv_add _ _ _ hmul_diff hconst_diff,
      pdiv_const BC γ i j, add_zero]
  have hconstXH_diff : DifferentiableAt ℝ (fun _ : Vec 1 => XH) γ := differentiableAt_const _
  have hgather_diff : DifferentiableAt ℝ
      (fun y : Vec 1 => fun k : Fin n => y ((fun _ : Fin n => (0 : Fin 1)) k)) γ :=
    (reindexCLM (fun _ : Fin n => (0 : Fin 1))).differentiableAt
  rw [show (fun (y : Vec 1) (k : Fin n) => XH k * y ((fun _ : Fin n => (0 : Fin 1)) k))
        = (fun y k => (fun _ k' => XH k') y k
                      * (fun y' k' => y' ((fun _ : Fin n => (0 : Fin 1)) k')) y k) from rfl,
      pdiv_mul _ _ _ hconstXH_diff hgather_diff,
      pdiv_const XH γ i j,
      pdiv_reindex (fun _ : Fin n => (0 : Fin 1)) γ i j]
  have hi0 : i = 0 := Fin.eq_zero i
  subst hi0
  simp

/-- **Jacobian of scalar-LN w.r.t. β** — `∂y_j/∂β = 1` (the scalar β shifts every output).
    The scalar special-case of `pdiv_bnPerChannelFlat_beta`. -/
private theorem pdiv_layerNorm_beta (n : Nat) (ε γ : ℝ) (x : Vec n) (β : Vec 1)
    (i : Fin 1) (j : Fin n) :
    pdiv (fun β' : Vec 1 => layerNormForward n ε γ (β' 0) x) β i j = 1 := by
  rw [layerNorm_beta_affine]
  set CC : Vec n := fun k => γ * bnXhat n ε x k with hCC
  have hconst_diff : DifferentiableAt ℝ (fun _ : Vec 1 => CC) β := differentiableAt_const _
  have hgather_diff : DifferentiableAt ℝ
      (fun y : Vec 1 => fun k : Fin n => y ((fun _ : Fin n => (0 : Fin 1)) k)) β :=
    (reindexCLM (fun _ : Fin n => (0 : Fin 1))).differentiableAt
  rw [show (fun (y : Vec 1) (k : Fin n) => γ * bnXhat n ε x k + y ((fun _ : Fin n => (0 : Fin 1)) k))
        = (fun y k => (fun _ k' => CC k') y k
                      + (fun y' k' => y' ((fun _ : Fin n => (0 : Fin 1)) k')) y k) from rfl,
      pdiv_add _ _ _ hconst_diff hgather_diff,
      pdiv_const CC β i j, zero_add,
      pdiv_reindex (fun _ : Fin n => (0 : Fin 1)) β i j]
  have hi0 : i = 0 := Fin.eq_zero i
  subst hi0
  simp

/-- **Scalar-LN γ-gradient bridge.** The rendered whole-`n` reduce `dγ = Σ_j dy_j·x̂_j`
    (`bn_grad_gamma`) equals the certified Jacobian of scalar-LN (as a function of γ, embedded
    `Vec 1`) contracted with the cotangent `dy`. Closes the gap `BatchNorm.lean` documented —
    `bn_grad_gamma` is now bridged, not just defined. -/
theorem cnx_lnGamma_grad_bridge (n : Nat) (ε β : ℝ) (γ : Vec 1) (x dy : Vec n) :
    bn_grad_gamma n ε x dy
      = ∑ j : Fin n, pdiv (fun γ' : Vec 1 => layerNormForward n ε (γ' 0) β x) γ 0 j * dy j := by
  simp only [pdiv_layerNorm_gamma]
  unfold bn_grad_gamma
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- **Scalar-LN β-gradient bridge.** Likewise the rendered `dβ = Σ_j dy_j` (`bn_grad_beta`) is
    the certified scalar-LN ∂/∂β contraction. -/
theorem cnx_lnBeta_grad_bridge (n : Nat) (ε γ : ℝ) (β : Vec 1) (x dy : Vec n) :
    bn_grad_beta n dy
      = ∑ j : Fin n, pdiv (fun β' : Vec 1 => layerNormForward n ε γ (β' 0) x) β 0 j * dy j := by
  simp only [pdiv_layerNorm_beta]
  unfold bn_grad_beta
  simp

/-- **Scalar-LN γ output, certified.** `γⁿ = γ − lr·(Σ dy·x̂)` denotes
    `γ − lr·(certified ∂(layerNormForward)/∂γ · cotangent)`. Covers the stem-LN, both block-LNs,
    and the head-LN of the representative ConvNeXt (each at its own `n`). -/
theorem cnx_render_lngamma_certified (n : Nat) (ε β : ℝ) (γ : Vec 1) (x dy : Vec n) (lr : ℝ) :
    γ 0 - lr * bn_grad_gamma n ε x dy
      = γ 0 - lr * ∑ j : Fin n,
          pdiv (fun γ' : Vec 1 => layerNormForward n ε (γ' 0) β x) γ 0 j * dy j := by
  rw [cnx_lnGamma_grad_bridge n ε β γ x dy]

/-- **Scalar-LN β output, certified.** `βⁿ = β − lr·(Σ dy)` denotes the certified scalar-LN
    `∂/∂β` contraction. -/
theorem cnx_render_lnbeta_certified (n : Nat) (ε γ : ℝ) (β : Vec 1) (x dy : Vec n) (lr : ℝ) :
    β 0 - lr * bn_grad_beta n dy
      = β 0 - lr * ∑ j : Fin n,
          pdiv (fun β' : Vec 1 => layerNormForward n ε γ (β' 0) x) β 0 j * dy j := by
  rw [cnx_lnBeta_grad_bridge n ε γ β x dy]

-- The stem/expand/project 1×1 convs and the dense head are covered VERBATIM by the existing
-- `cnn_render_conv{W,b}_certified` (M3) / M2 `weight_grad_bridge`/`bias_grad_bridge` at the
-- ConvNeXt shapes — no kernel size to pin (1×1 already exercised; dense is dense). GELU carries
-- no parameters. With the 7×7 depthwise, layer-scale γ, and scalar-LN γ/β above, every
-- representative-ConvNeXt train-step parameter output is certified
-- `θ − lr·(certified Jacobian · the cotangent)`.

end Proofs
