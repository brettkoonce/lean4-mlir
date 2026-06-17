import LeanMlir.Proofs.PerChannelBN

/-! # Closing the CIFAR-BN render — the BN parameter-gradient bridges (dγ, dβ certified)

The non-BN closes (`cnn_render_conv{W,b}_certified` + the M2 dense bridges) and the BN
**input**-grad (`bnPerChannel_grad_input_correct`, under `0<ε`) already cover every
parameter of the CIFAR-BN train step except the per-channel BN scale/shift γ, β. This
file supplies their bridges — the BN analogue of `IR.bias_grad_bridge` / `conv_bias_grad`.

γ and β enter BN **affinely**: per channel `c`, `y_(c,s) = γ_c · x̂_(c,s) + β_c`, and x̂
does not depend on γ or β. So as a function of γ (resp. β), per-channel BN is
`x̂ ⊙ gather_channel(γ) + const` (resp. `const + gather_channel(β)`) — a constant scaled
by a channel-gather, plus a constant. Its Jacobian therefore collapses through
`pdiv_add`/`pdiv_mul`/`pdiv_const`/`pdiv_reindex` to the sparse indicator
`∂y_j/∂γ_idx = x̂_j·[chan j = idx]` (resp. `[chan j = idx]`), and contracting with the
cotangent `dy` gives exactly the rendered per-channel reduces
`dγ_c = Σ_s dy·x̂`, `dβ_c = Σ_s dy` (the `bnParamGradPC` block in
`cifarBnTrainStepStructured`). Unlike the BN input grad these need no `0<ε` (affine in the
params; ε only enters the constant x̂). See `planning/render_close_handoff.md` §2b.
-/

namespace Proofs

open scoped BigOperators

-- `bnPerChannel_grad_gamma` / `bnPerChannel_grad_beta` moved to `PerChannelBN.lean`
-- (so the `bnGammaSgd`/`bnBetaSgd` `SHlo` ops' `den` can reference them upstream).

/-- Channel index of a flat position `(c, s) ↦ c`. -/
private noncomputable def chanOf (oc m : Nat) (k : Fin (oc * m)) : Fin oc :=
  (finProdFinEquiv.symm k).1

/-- per-channel BN, as a function of γ (β, v fixed), written affinely:
    `γ' ↦ fun k => x̂_k · γ'(chan k) + β(chan k)`. -/
private theorem bnPerChannelFlat_gamma_affine (oc m : Nat) (ε : ℝ) (β : Vec oc) (v : Vec (oc * m)) :
    (fun γ' : Vec oc => bnPerChannelFlat oc m ε γ' β v)
      = fun y k => bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm k).1) (finProdFinEquiv.symm k).2
                     * y (finProdFinEquiv.symm k).1
                   + β (finProdFinEquiv.symm k).1 := by
  funext y k
  simp only [bnPerChannelFlat, bnPerChannelMat, Mat.flatten, bnForward]
  ring

/-- per-channel BN, as a function of β (γ, v fixed):
    `β' ↦ fun k => γ(chan k)·x̂_k + β'(chan k)`. -/
private theorem bnPerChannelFlat_beta_affine (oc m : Nat) (ε : ℝ) (γ : Vec oc) (v : Vec (oc * m)) :
    (fun β' : Vec oc => bnPerChannelFlat oc m ε γ β' v)
      = fun y k => γ (finProdFinEquiv.symm k).1
                     * bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm k).1) (finProdFinEquiv.symm k).2
                   + y (finProdFinEquiv.symm k).1 := by
  funext y k
  simp only [bnPerChannelFlat, bnPerChannelMat, Mat.flatten, bnForward]

/-- **Jacobian of per-channel BN w.r.t. γ** — the sparse indicator `x̂_j·[chan j = idx]`. -/
private theorem pdiv_bnPerChannelFlat_gamma (oc m : Nat) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * m)) (idx : Fin oc) (j : Fin (oc * m)) :
    pdiv (fun γ' : Vec oc => bnPerChannelFlat oc m ε γ' β v) γ idx j
      = if idx = (finProdFinEquiv.symm j).1
        then bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm j).1) (finProdFinEquiv.symm j).2
        else 0 := by
  rw [bnPerChannelFlat_gamma_affine]
  -- `fun y k => XH k * y (chan k) + BC k`, with XH const, gather linear, BC const.
  set XH : Vec (oc * m) :=
    fun k => bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm k).1) (finProdFinEquiv.symm k).2 with hXH
  set BC : Vec (oc * m) := fun k => β (finProdFinEquiv.symm k).1 with hBC
  have hmul_diff : DifferentiableAt ℝ
      (fun y : Vec oc => fun k => XH k * y (finProdFinEquiv.symm k).1) γ :=
    (differentiableAt_const XH).mul
      ((reindexCLM (fun k : Fin (oc * m) => (finProdFinEquiv.symm k).1)).differentiableAt)
  have hconst_diff : DifferentiableAt ℝ (fun _ : Vec oc => BC) γ := differentiableAt_const _
  rw [show (fun (y : Vec oc) (k : Fin (oc * m)) => XH k * y (finProdFinEquiv.symm k).1 + BC k)
        = (fun y k => (fun y' k' => XH k' * y' (finProdFinEquiv.symm k').1) y k
                      + (fun _ k' => BC k') y k) from rfl,
      pdiv_add _ _ _ hmul_diff hconst_diff,
      pdiv_const BC γ idx j, add_zero]
  -- the product: (const XH) * (gather γ').
  have hconstXH_diff : DifferentiableAt ℝ (fun _ : Vec oc => XH) γ := differentiableAt_const _
  have hgather_diff : DifferentiableAt ℝ
      (fun y : Vec oc => fun k => y (finProdFinEquiv.symm k).1) γ :=
    (reindexCLM (fun k : Fin (oc * m) => (finProdFinEquiv.symm k).1)).differentiableAt
  rw [show (fun (y : Vec oc) (k : Fin (oc * m)) => XH k * y (finProdFinEquiv.symm k).1)
        = (fun y k => (fun _ k' => XH k') y k * (fun y' k' => y' (finProdFinEquiv.symm k').1) y k)
        from rfl,
      pdiv_mul _ _ _ hconstXH_diff hgather_diff,
      pdiv_const XH γ idx j,
      pdiv_reindex (fun k : Fin (oc * m) => (finProdFinEquiv.symm k).1) γ idx j]
  simp [hXH]

/-- **Jacobian of per-channel BN w.r.t. β** — the channel indicator `[chan j = idx]`. -/
private theorem pdiv_bnPerChannelFlat_beta (oc m : Nat) (ε : ℝ) (γ β : Vec oc)
    (v : Vec (oc * m)) (idx : Fin oc) (j : Fin (oc * m)) :
    pdiv (fun β' : Vec oc => bnPerChannelFlat oc m ε γ β' v) β idx j
      = if idx = (finProdFinEquiv.symm j).1 then 1 else 0 := by
  rw [bnPerChannelFlat_beta_affine]
  set CC : Vec (oc * m) :=
    fun k => γ (finProdFinEquiv.symm k).1
               * bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm k).1) (finProdFinEquiv.symm k).2 with hCC
  have hconst_diff : DifferentiableAt ℝ (fun _ : Vec oc => CC) β := differentiableAt_const _
  have hgather_diff : DifferentiableAt ℝ
      (fun y : Vec oc => fun k => y (finProdFinEquiv.symm k).1) β :=
    (reindexCLM (fun k : Fin (oc * m) => (finProdFinEquiv.symm k).1)).differentiableAt
  rw [show (fun (y : Vec oc) (k : Fin (oc * m)) => CC k + y (finProdFinEquiv.symm k).1)
        = (fun y k => (fun _ k' => CC k') y k
                      + (fun y' k' => y' (finProdFinEquiv.symm k').1) y k) from rfl,
      pdiv_add _ _ _ hconst_diff hgather_diff,
      pdiv_const CC β idx j, zero_add,
      pdiv_reindex (fun k : Fin (oc * m) => (finProdFinEquiv.symm k).1) β idx j]

/-- Sum-over-the-channel-fibre: `Σ_j [idx = chan j]·g j = Σ_s g (idx, s)`. -/
private theorem sum_channel_fibre (oc m : Nat) (idx : Fin oc) (g : Fin (oc * m) → ℝ) :
    (∑ j : Fin (oc * m), (if idx = (finProdFinEquiv.symm j).1 then g j else 0))
      = ∑ s : Fin m, g (finProdFinEquiv (idx, s)) := by
  rw [← Equiv.sum_comp finProdFinEquiv
        (fun j => if idx = (finProdFinEquiv.symm j).1 then g j else 0)]
  rw [Fintype.sum_prod_type]
  simp only [Equiv.symm_apply_apply]
  have hpull : ∀ c : Fin oc,
      (∑ s : Fin m, if idx = c then g (finProdFinEquiv (c, s)) else 0)
        = if idx = c then ∑ s : Fin m, g (finProdFinEquiv (c, s)) else 0 := by
    intro c; by_cases h : idx = c <;> simp [h]
  simp only [hpull]
  rw [Finset.sum_ite_eq Finset.univ idx (fun c => ∑ s : Fin m, g (finProdFinEquiv (c, s)))]
  simp

/-- **BN γ-gradient bridge.** The rendered per-channel `dγ_idx = Σ_s dy·x̂` equals the
    certified Jacobian of per-channel BN (as a function of γ) contracted with the
    cotangent `dy`. The BN analogue of the conv/dense weight bridges; affine in γ, so no
    `0<ε`. -/
theorem bnPerChannel_grad_gamma_correct (oc m : Nat) (ε : ℝ) (γ β : Vec oc)
    (v dy : Vec (oc * m)) (idx : Fin oc) :
    bnPerChannel_grad_gamma oc m ε v dy idx
      = ∑ j : Fin (oc * m), pdiv (fun γ' : Vec oc => bnPerChannelFlat oc m ε γ' β v) γ idx j * dy j := by
  simp only [pdiv_bnPerChannelFlat_gamma]
  rw [show (∑ j : Fin (oc * m),
          (if idx = (finProdFinEquiv.symm j).1
           then bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm j).1) (finProdFinEquiv.symm j).2
           else 0) * dy j)
        = ∑ j : Fin (oc * m),
            (if idx = (finProdFinEquiv.symm j).1
             then bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm j).1) (finProdFinEquiv.symm j).2 * dy j
             else 0) by
      apply Finset.sum_congr rfl; intro j _; by_cases h : idx = (finProdFinEquiv.symm j).1 <;> simp [h]]
  rw [sum_channel_fibre oc m idx
        (fun j => bnXhat m ε (Mat.unflatten v (finProdFinEquiv.symm j).1) (finProdFinEquiv.symm j).2 * dy j)]
  simp only [bnPerChannel_grad_gamma, Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl; intro s _; ring

/-- **BN β-gradient bridge.** The rendered per-channel `dβ_idx = Σ_s dy` equals the
    certified Jacobian of per-channel BN (as a function of β) contracted with `dy`. The
    BN analogue of `bias_grad_bridge`; affine in β, so no `0<ε`. -/
theorem bnPerChannel_grad_beta_correct (oc m : Nat) (ε : ℝ) (γ β : Vec oc)
    (v dy : Vec (oc * m)) (idx : Fin oc) :
    bnPerChannel_grad_beta oc m dy idx
      = ∑ j : Fin (oc * m), pdiv (fun β' : Vec oc => bnPerChannelFlat oc m ε γ β' v) β idx j * dy j := by
  simp only [pdiv_bnPerChannelFlat_beta]
  rw [show (∑ j : Fin (oc * m), (if idx = (finProdFinEquiv.symm j).1 then (1 : ℝ) else 0) * dy j)
        = ∑ j : Fin (oc * m), (if idx = (finProdFinEquiv.symm j).1 then dy j else 0) by
      apply Finset.sum_congr rfl; intro j _; by_cases h : idx = (finProdFinEquiv.symm j).1 <;> simp [h]]
  rw [sum_channel_fibre oc m idx (fun j => dy j)]
  rfl

/-- **BN γ output certified.** `γ_c − lr·(rendered dγ_c)` denotes
    `γ_c − lr·(certified ∂(per-channel BN)/∂γ_c · cotangent)`. The γ peer of
    `cnn_render_convb_certified`. -/
theorem cifar_bn_render_gamma_certified (oc m : Nat) (ε : ℝ) (γ β : Vec oc)
    (v dy : Vec (oc * m)) (lr : ℝ) (idx : Fin oc) :
    γ idx - lr * bnPerChannel_grad_gamma oc m ε v dy idx
      = γ idx - lr * ∑ j : Fin (oc * m),
          pdiv (fun γ' : Vec oc => bnPerChannelFlat oc m ε γ' β v) γ idx j * dy j := by
  rw [bnPerChannel_grad_gamma_correct]

/-- **BN β output certified.** `β_c − lr·(rendered dβ_c)` denotes the certified BN
    `∂/∂β` contraction. The β peer. -/
theorem cifar_bn_render_beta_certified (oc m : Nat) (ε : ℝ) (γ β : Vec oc)
    (v dy : Vec (oc * m)) (lr : ℝ) (idx : Fin oc) :
    β idx - lr * bnPerChannel_grad_beta oc m dy idx
      = β idx - lr * ∑ j : Fin (oc * m),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc m ε γ β' v) β idx j * dy j := by
  rw [bnPerChannel_grad_beta_correct (γ := γ)]

end Proofs
