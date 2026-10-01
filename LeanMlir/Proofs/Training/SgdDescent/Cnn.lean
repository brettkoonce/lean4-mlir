import LeanMlir.Proofs.Float.ConvFloat
import LeanMlir.Proofs.Training.SgdDescent.MlpBias
import LeanMlir.Proofs.Architectures.ConvGrad
import LeanMlir.Proofs.Nets.Small.MnistCNN

/-! # Lipschitz constants for the CNN softmax-CE loss — descent through the pool

`SgdDescent.Mlp` discharged `sgd_descends`' smoothness hypothesis for every
MLP weight layer; this file extends the program to the Chapter-3 MNIST CNN
(`conv → relu → conv → relu → maxpool → dense → relu → dense → relu →
dense`). What's genuinely new versus the MLP:

* **The dense head is free.** Below the pool the CNN *is* an MLP at the
  pooled activation: the loss-of-`W₅`/`W₄`/`W₃` maps are literal instances
  of `linear_sgd_descends` / `mlp_hidden_sgd_descends` /
  `mlp_input_sgd_descends` at `x := maxPoolFlat (…)`, and the loss-of-`b₅`/`b₄`/`b₃`
  maps of `linear_bias_sgd_descends` / `mlp_hidden_bias_sgd_descends` /
  `mlp_input_bias_sgd_descends` (`SgdDescent.MlpBias`). No new theorems are
  needed (the MLP statements are generic in the fixed activation vector).

* **The max-pool needs a selection margin, up to twins.** At a tied window
  the pool has no derivative, and real MNIST ties windows in nearly every
  image: a constant background patch makes a window's conv outputs equal
  for EVERY kernel. Those cells are twins: they read identical zero-padded
  input patches (`ConvPatchEq`, and `ConvPatchEq2` two convs deep), so they
  stay equal along any step. The rungs take `MaxPool2MarginQUpTo δ T` on the
  conv2 pre-activation: each window is dead (all cells `≤ 0`, kept dead by
  the relu₂ margin), or the cell dominating it is more than `2δ` above every
  cell that is not its twin. Then near every point of the step segment the
  relu'd pool IS the gather at the base point's argmax
  (`Conv2Slot.maxPool_relu_eventuallyEq_gather`), and the gather is linear
  and `ℓ1`-contractive (`poolGatherFlat_l1_contract`: the 2×2 stride-2
  windows partition the input), so the descent argument runs on it
  (`Conv2Slot.gather_grad_lipschitz`) and transfers back
  (`Conv2Slot.sgd_descends`). The probe
  scripts/probes/mnist_pool_twin_probe.py evaluates the condition on the
  MNIST test set at the trained Chapter-3 weights.

* **Conv layers are dense layers with weight sharing.** The conv output is
  affine in the kernel; each output entry reads one kernel slab against
  bounded input values (`conv2d_flat_kernel_drift_total`), and the `ℓ1` drift picks
  up the spatial multiplicity `h·w` — each kernel entry touches every
  spatial position (`conv2d_flat_kernel_drift_sum`).

The capstone `cnn_conv2_sgd_descends` mirrors `mlp_input_sgd_descends`:
under the four margins (relu₂, pool selection up to twins, relu₃, relu₄) at the step
radius and the small-step condition, one inexact SGD step on the second
conv kernel (one example, every other parameter fixed) decreases that example's
cross-entropy loss by ≥ `lr·‖∇L‖₂²/2`, with the segment-Lipschitz constant
explicit.

`cnn_conv1_sgd_descends` extends the program one layer deeper: the step
now crosses conv2 AS A FUNCTION OF ITS INPUT. Conv is linear there, its
Jacobian entry a single kernel tap (`convTap`, extracted point-free from
the certified input-VJP), and its `ℓ1` operator factor is LOCALITY —
`(channels)·kH·kW·w₂`, not a spatial count. Under FIVE margins (relu₁ +
the conv2 four, at conv1 radii) every routing decision freezes and the
loss provably drops.

`cnn_conv2_bias_sgd_descends` / `cnn_conv1_bias_sgd_descends` close the
biases: the bias-map Jacobian is a Kronecker channel indicator
(`conv2d_bias_pdiv`, extracted from the certified bias VJP), the
per-entry drift is exactly `|e o|` (no input bound `a`). Each conv layer's
drift chain, margins, segment-Lipschitz gradient and descent step are stated once, for any
parameter map with per-entry drift `ρ·‖e‖₁` (`Conv2Slot`, `Conv1Slot`): the
kernel rungs are `ρ = a`, the bias rungs `ρ = 1` — the bare `D` radii and
`a² ↦ 1` in the constants. Both conv kernels, both conv biases and the
dense-head weights and biases of the Chapter-3 CNN (the latter via the MLP
rungs, `SgdDescent.Mlp` and `SgdDescent.MlpBias`) each have a single-layer, single-example descent
statement, conditional on the margins above and the oracle-accuracy,
small-step and dominance hypotheses. The rungs that replace the oracle accuracy by
the proven accuracy of the FloatModel binary32 gradient are in `SgdDescent.CnnFloat`.

The index plumbing, the 2×2 max-pool window facts and the twin relations (`ConvPatchEq`,
`ConvPatchEq2`) it reads tensors through are in `ConvIndex`; the conv's kernel, input and bias
Jacobians and drifts (`conv2d_weight_pdiv`, `convTap`, `conv2d_input_l1_drift`,
`conv2d_bias_pdiv`) in `ConvGrad`. -/

namespace Proofs

open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The 3-dense head above the pool: input-gradient closed form
-- ════════════════════════════════════════════════════════════════

/-- The 3-dense head `CE ∘ d₅ ∘ relu ∘ d₄ ∘ relu ∘ d₃` is differentiable
    at any point whose two ReLU pre-activations are off the kinks. -/
private theorem ce_head3_differentiableAt {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃)
    (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC)
    (b₅ : Vec nC) (label : Fin nC) (u : Vec p)
    (hz3 : ∀ l, dense W₃ b₃ u l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q ≠ 0) :
    DifferentiableAt ℝ
      (fun y : Vec p => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y)))))
        label) u := by
  fun_prop (disch := assumption)

/-- **Loss input-gradient of the 3-dense head** `CE∘d₅∘relu∘d₄∘relu∘d₃`
    at the pooled vector — one `pdiv_comp` hop (peel `dense W₃`) on top of
    `ce_head2_input_grad`, exactly as `ce_head2` was one hop on
    `ce_head_relu`. Note there is NO leading mask: the pool output feeds
    `dense W₃` directly. -/
theorem ce_head3_input_grad {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃)
    (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC)
    (b₅ : Vec nC) (label : Fin nC) (u : Vec p)
    (hz3 : ∀ l, dense W₃ b₃ u l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q ≠ 0) (j : Fin p) :
    pdiv (fun y : Vec p => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y)))))
        label) u j 0
      = ∑ l, W₃ j l *
          ((if dense W₃ b₃ u l > 0 then (1:ℝ) else 0) *
            ∑ q, W₄ l q *
              ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q > 0
                  then (1:ℝ) else 0) *
                ∑ k, W₅ q k *
                  (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                      (relu d₃ (dense W₃ b₃ u))))) k -
                    oneHot nC label k))) := by
  have hH : DifferentiableAt ℝ
      (fun z : Vec d₃ => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ z)))) label)
      (dense W₃ b₃ u) := by
    fun_prop (disch := assumption)
  rw [show (fun y : Vec p => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y)))))
          label)
        = (fun z : Vec d₃ => fun _ : Fin 1 => crossEntropy nC
            (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ z)))) label)
          ∘ (dense W₃ b₃) from rfl,
      pdiv_comp _ _ _ ((dense_differentiable W₃ b₃) u) hH]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [pdiv_dense, ce_head2_input_grad W₄ b₄ W₅ b₅ label _ hz3 hz4 l]

-- ════════════════════════════════════════════════════════════════
-- § Through the pool: the loss gradient at the conv output
-- ════════════════════════════════════════════════════════════════

/-- The whole head above the conv output — `CE∘head3∘maxPoolFlat∘relu` —
    is differentiable at any point with the relu₂ pre-activation off the
    kinks, no pool ties (POST-relu), and the two head masks off the
    kinks. -/
private theorem pool_head_differentiableAt {c h w d₃ d₄ nC : Nat}
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₂ : Vec (c * (2*h) * (2*w))) (hz2 : ∀ k, z₂ k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₂) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
      z₂ := by
  have hmp_d : DifferentiableAt ℝ (maxPoolFlat c h w) (relu (c * (2*h) * (2*w)) z₂) := by
    rw [← Tensor3.flatten_unflatten (relu _ z₂)]
    exact maxPoolFlat_differentiableAt _ hmp
  fun_prop (disch := assumption)

/-- **Loss input-gradient at the conv output** — the key glue of the conv
    rung. The chain `pdiv`s through the relu (mask) and the pool (frozen
    selector): at a smooth point the sum over pooled coordinates collapses
    to the single argmax term, so

    `∂(CE∘head3∘pool∘relu)/∂z₂[ci,hi,wi] = relu'(z₂[ci,hi,wi]) ·
       𝟙[(ci,hi,wi) is its window's argmax] · head3grad(window(ci,hi,wi))`.

    NB the pool acts on the POST-relu activation, so the smoothness and
    argmax conditions are stated on `relu z₂`, not `z₂`. -/
theorem pool_relu_input_grad {c h w d₃ d₄ nC : Nat}
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₂ : Vec (c * (2*h) * (2*w))) (hz2 : ∀ k, z₂ k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₂) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)))) q ≠ 0)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
        z₂ (t3Idx ci hi wi) 0
      = (if z₂ (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          (if MaxPool2IsArgmax
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₂)) ci hi wi
            then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
              ((if dense W₃ b₃ (maxPoolFlat c h w
                    (relu (c * (2*h) * (2*w)) z₂)) l > 0
                  then (1:ℝ) else 0) *
                ∑ q, W₄ l q *
                  ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                        (relu (c * (2*h) * (2*w)) z₂)))) q > 0
                      then (1:ℝ) else 0) *
                    ∑ k, W₅ q k *
                      (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                          (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) z₂))))))) k -
                        oneHot nC label k)))
            else 0) := by
  have hHd := ce_head3_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label
    (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) z₂)) hz3 hz4
  have hpt : Tensor3.flatten (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₂) : Tensor3 c (2*h) (2*w)) =
      relu (c * (2*h) * (2*w)) z₂ := Tensor3.flatten_unflatten _
  have hmp_d : DifferentiableAt ℝ (maxPoolFlat c h w)
      (relu (c * (2*h) * (2*w)) z₂) := by
    rw [← hpt]
    exact maxPoolFlat_differentiableAt _ hmp
  have hG : DifferentiableAt ℝ
      ((fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
          label) ∘ (maxPoolFlat c h w))
      (relu (c * (2*h) * (2*w)) z₂) :=
    hHd.comp _ hmp_d
  -- hop 1: peel the relu; the chain picks up the mask
  rw [show (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
          crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
        = ((fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
            (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
            label) ∘ (maxPoolFlat c h w)) ∘ (relu (c * (2*h) * (2*w)))
        from rfl,
      pdiv_comp _ _ _
        (relu_differentiableAt_of_smooth (c * (2*h) * (2*w)) z₂ hz2) hG]
  simp_rw [pdiv_relu (c * (2*h) * (2*w)) z₂ hz2 (t3Idx ci hi wi), ite_mul,
    zero_mul]
  rw [Finset.sum_ite_eq]
  simp only [Finset.mem_univ, ite_true]
  congr 1
  -- hop 2: through the pool; the routing collapses to the argmax cell
  rw [pdiv_comp (maxPoolFlat c h w) _ _ hmp_d hHd (t3Idx ci hi wi) 0,
    sum_t3 (fun q : Fin (c * h * w) =>
      pdiv (maxPoolFlat c h w) (relu (c * (2*h) * (2*w)) z₂)
        (t3Idx ci hi wi) q *
      pdiv (fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
          label) (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) z₂)) q 0)]
  -- the pool pdiv IS pdiv3 of maxPool2, which the margin collapses
  have hglue : ∀ (co : Fin c) (ho : Fin h) (wo : Fin w),
      pdiv (maxPoolFlat c h w) (relu (c * (2*h) * (2*w)) z₂)
        (t3Idx ci hi wi) (t3Idx co ho wo) =
      (if co = ci ∧ ho = winRow hi ∧ wo = winCol wi ∧
          MaxPool2IsArgmax (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) z₂)) ci hi wi
        then (1:ℝ) else 0) := by
    intro co ho wo
    have h1 : pdiv (maxPoolFlat c h w) (relu (c * (2*h) * (2*w)) z₂)
        (t3Idx ci hi wi) (t3Idx co ho wo) =
        pdiv3 maxPool2 (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₂))
          ci hi wi co ho wo := by
      unfold pdiv3
      rw [hpt]
      rfl
    rw [h1, pdiv3_maxPool2_smooth _ hmp ci hi wi co ho wo]
  simp_rw [hglue]
  simp only [ite_and, ite_mul, one_mul, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true,
    ce_head3_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz3 hz4]

/-- **Loss input-gradient at the conv output, through a fixed gather** — the gather peer of
    `pool_relu_input_grad`. With the pool replaced by `poolGatherFlat σ`, the route is `σ`'s
    selection rather than an argmax, so no pool hypothesis is needed:

    `∂(CE∘head3∘gather∘relu)/∂z₂[ci,hi,wi] = relu'(z₂[ci,hi,wi]) ·
       𝟙[(hi,wi) is σ's cell of its window] · head3grad(window(ci,hi,wi))`. -/
theorem gather_relu_input_grad {c h w d₃ d₄ nC : Nat}
    (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₂ : Vec (c * (2*h) * (2*w))) (hz2 : ∀ k, z₂ k ≠ 0)
    (hz3 : ∀ l, dense W₃ b₃ (poolGatherFlat σ
      (relu (c * (2*h) * (2*w)) z₂)) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
      (relu (c * (2*h) * (2*w)) z₂)))) q ≠ 0)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (poolGatherFlat σ (relu (c * (2*h) * (2*w)) y))))))) label)
        z₂ (t3Idx ci hi wi) 0
      = (if z₂ (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          (if σ ci (winRow hi) (winCol wi) = (winRowMod hi, winColMod wi)
            then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
              ((if dense W₃ b₃ (poolGatherFlat σ
                    (relu (c * (2*h) * (2*w)) z₂)) l > 0
                  then (1:ℝ) else 0) *
                ∑ q, W₄ l q *
                  ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
                        (relu (c * (2*h) * (2*w)) z₂)))) q > 0
                      then (1:ℝ) else 0) *
                    ∑ k, W₅ q k *
                      (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                          (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
                            (relu (c * (2*h) * (2*w)) z₂))))))) k -
                        oneHot nC label k)))
            else 0) := by
  have hHd := ce_head3_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label
    (poolGatherFlat σ (relu (c * (2*h) * (2*w)) z₂)) hz3 hz4
  have hG : DifferentiableAt ℝ
      ((fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
          label) ∘ (poolGatherFlat σ))
      (relu (c * (2*h) * (2*w)) z₂) :=
    hHd.comp _ ((poolGatherFlat_differentiable σ) _)
  -- hop 1: peel the relu; the chain picks up the mask
  rw [show (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
          crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (poolGatherFlat σ (relu (c * (2*h) * (2*w)) y))))))) label)
        = ((fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
            (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
            label) ∘ (poolGatherFlat σ)) ∘ (relu (c * (2*h) * (2*w)))
        from rfl,
      pdiv_comp _ _ _
        (relu_differentiableAt_of_smooth (c * (2*h) * (2*w)) z₂ hz2) hG]
  simp_rw [pdiv_relu (c * (2*h) * (2*w)) z₂ hz2 (t3Idx ci hi wi), ite_mul,
    zero_mul]
  rw [Finset.sum_ite_eq]
  simp only [Finset.mem_univ, ite_true]
  congr 1
  -- hop 2: through the gather; the routing is σ's fixed selection
  rw [pdiv_comp (poolGatherFlat σ) _ _ ((poolGatherFlat_differentiable σ) _) hHd
      (t3Idx ci hi wi) 0,
    sum_t3 (fun q : Fin (c * h * w) =>
      pdiv (poolGatherFlat σ) (relu (c * (2*h) * (2*w)) z₂)
        (t3Idx ci hi wi) q *
      pdiv (fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
          label) (poolGatherFlat σ (relu (c * (2*h) * (2*w)) z₂)) q 0)]
  simp_rw [pdiv_poolGatherFlat]
  simp only [ite_and, ite_mul, one_mul, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true,
    ce_head3_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz3 hz4]

-- ════════════════════════════════════════════════════════════════
-- § The conv2 loss-of-kernel map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- **The loss gradient through a parameter map into a `c×h×w` activation** — the chain rule
    (`pdiv_comp`) with the flat activation index split into its triple: `Z`'s Jacobian row
    contracted with the head's input gradient. Each conv rung's `gradAt` closed form is this,
    the conv Jacobian (`conv2d_weight_pdiv` / `conv2d_bias_pdiv`) and the head gradient. -/
private theorem gradAt_comp_t3 {P c h w : Nat} (Z : Vec P → Vec (c * h * w))
    (G : Vec (c * h * w) → ℝ) (v : Vec P) (hZ : DifferentiableAt ℝ Z v)
    (hG : DifferentiableAt ℝ (fun y => fun _ : Fin 1 => G y) (Z v)) (idx : Fin P) :
    gradAt (fun v' => G (Z v')) v idx =
      ∑ ci : Fin c, ∑ hi : Fin h, ∑ wi : Fin w, pdiv Z v idx (t3Idx ci hi wi) *
        pdiv (fun y => fun _ : Fin 1 => G y) (Z v) (t3Idx ci hi wi) 0 := by
  rw [gradAt_eq_pdiv (fun v' => G (Z v')) v ((differentiableAt_pi.mp hG 0).comp v hZ) idx]
  exact (pdiv_comp Z (fun y => fun _ : Fin 1 => G y) v hZ hG idx 0).trans (sum_t3 _)

/-- **Closed form of the conv2 loss gradient** at any four-margin point —
    the chain rule through the conv weight map (`gradAt_comp_t3`)
    with the pool-collapsed head gradient
    (`pool_relu_input_grad`) and the point-free conv weight Jacobian
    (`conv2d_weight_pdiv`). The conv-layer peer of
    `mlp_input_loss_gradAt`; the spatial triple sum (vs the MLP's
    Kronecker collapse) is weight sharing. -/
theorem cnn_conv2_loss_gradAt {c h w d₃ d₄ nC kH kW : Nat}
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (v : Vec (c * c * kH * kW))
    (hz2 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q ≠ 0)
    (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label)
        v (k4Idx o cc kh kw)
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0) *
            ((if Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                  ci hi wi
                then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
                  ((if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                        l > 0 then (1:ℝ) else 0) *
                    ∑ q, W₄ l q *
                      ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q > 0
                          then (1:ℝ) else 0) *
                        ∑ k, W₅ q k *
                          (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                              (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d (Kernel4.unflatten v) b₂ x₁))))))))) k -
                            oneHot nC label k)))
                else 0)) := by
  refine (gradAt_comp_t3 (fun v' => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁))
    (fun y => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
      (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label) v
    (conv2d_weight_differentiable b₂ x₁ v)
    (pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label _
      hz2 hmp hz3 hz4) _).trans
    (Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl fun hi _ =>
      Finset.sum_congr rfl fun wi _ => ?_)
  rw [conv2d_weight_pdiv b₂ x₁ _ o cc kh kw ci hi wi,
    pool_relu_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hmp hz3 hz4 ci hi wi]

/-- The unmasked peer of `reluMask_dense_transpose_eq`: a bare `Wᵀ`
    contraction `∑ₖ Wₗₖ·cₖ = dense (transpose W) 0 c l`. The pool feeds
    `dense W₃` with **no** leading ReLU mask, so the W₃ contraction in the
    certified conv-2 gradient collapses through this, where the masked W₄/W₅
    contractions collapse through `reluMask_dense_transpose_eq`. NB this is
    generic in `(W, c)`, so fire it only where the goal has no *other* matrix
    contraction (e.g. the spatial `∑ convPad·cot`) — see `head3_cot_reluMask`. -/
theorem dense_transpose_eq {p n : Nat} (W : Mat p n) (c : Vec n) (l : Fin p) :
    (∑ k, W l k * c k) = dense (fun j i' => W i' j) (fun _ => 0) c l := by
  show (∑ k, W l k * c k) = (∑ k, c k * W l k) + (0:ℝ)
  rw [add_zero]
  exact Finset.sum_congr rfl fun k _ => mul_comm _ _

/-- **The 3-dense head cotangent in `dense`/`reluMask` form.** The raw nested
    `∑ₗ W₃·(𝟙[z₃]·∑_q W₄·(𝟙[z₄]·∑_k W₅·(softmax−onehot)))` that
    `pool_relu_input_grad` / `cnn_conv2_loss_gradAt` leave at the pooled vector
    `u` equals `dense W₃ᵀ 0 (mask z₃ (dense W₄ᵀ 0 (mask z₄ (dense W₅ᵀ 0
    (softmax−onehot)))))` — the two masked contractions via
    `reluMask_dense_transpose_eq`, the unmasked W₃ via `dense_transpose_eq`.
    Stated head-locally (no spatial sum) so the generic `dense_transpose_eq`
    fires only on the W₃ row. The head peer the conv grad-close bounds against
    via `dense_close` (W₃) and `cot_step_close` (W₄/W₅). -/
theorem head3_cot_reluMask {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (u : Vec p) (j : Fin p) :
    (∑ l, W₃ j l *
        ((if dense W₃ b₃ u l > 0 then (1:ℝ) else 0) *
          ∑ q, W₄ l q *
            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q > 0 then (1:ℝ) else 0) *
              ∑ k, W₅ q k *
                (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                    (relu d₃ (dense W₃ b₃ u))))) k - oneHot nC label k))))
      = dense (fun j' i' => W₃ i' j') (fun _ => 0)
          (FloatModel.reluMask (dense W₃ b₃ u)
            (dense (fun j' i' => W₄ i' j') (fun _ => 0)
              (FloatModel.reluMask (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))
                (dense (fun j' i' => W₅ i' j') (fun _ => 0)
                  (fun k => softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                      (relu d₃ (dense W₃ b₃ u))))) k - oneHot nC label k)))))
          j := by
  simp_rw [reluMask_dense_transpose_eq]
  rw [dense_transpose_eq]

/-- **The certified conv-2 loss gradient, head restated in `dense`/`reluMask`
    form** — the conv peer of `mlp_input_loss_gradAt_reluMask`. The two head `Wᵀ` contractions (under the d₄/d₃ ReLU masks)
    collapse via `reluMask_dense_transpose_eq`, the unmasked W₃ contraction via
    `dense_transpose_eq`; the conv-output ReLU mask `𝟙[z₂>0]` and the pool
    argmax selector are kept explicit (their float closeness is handled by
    `reluMask_close` and `MaxPool2MarginQ.poolBack_close`). The whole conv
    gradient is then packaged as the spatial dot `∑ₛ convPadWin·cotWin`
    (`convWeightGrad_eq_dot`) — the exact quantity the FloatModel gradient's
    conv-weight dot rounds, so the conv grad-close bounds against this. -/
theorem cnn_conv2_loss_gradAt_reluMask {c h w d₃ d₄ nC kH kW : Nat}
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (v : Vec (c * c * kH * kW))
    (hz2 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q ≠ 0)
    (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label)
        v (k4Idx o cc kh kw)
      = ∑ s, convPadWin kH kW x₁ cc kh kw s *
          cotWin (fun ci hi wi =>
            (if Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                  ci hi wi
                then dense (fun j i' => W₃ i' j) (fun _ => 0)
                  (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                        (conv2d (Kernel4.unflatten v) b₂ x₁)))))
                    (dense (fun j i' => W₄ i' j) (fun _ => 0)
                      (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                          (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d (Kernel4.unflatten v) b₂ x₁)))))))
                        (dense (fun j i' => W₅ i' j) (fun _ => 0)
                          (fun k => softmax nC (dense W₅ b₅ (relu d₄
                              (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                  (Tensor3.flatten (conv2d (Kernel4.unflatten v)
                                    b₂ x₁))))))))) k - oneHot nC label k)))))
                  (t3Idx ci (winRow hi) (winCol wi))
                else 0)) o s := by
  rw [cnn_conv2_loss_gradAt b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label v hz2 hmp hz3
      hz4 o cc kh kw]
  -- restate the head into dense/reluMask form (head-local lemma — does not
  -- touch the spatial `∑ convPad·cot` sum), then package as the spatial dot
  simp_rw [head3_cot_reluMask]
  rw [convWeightGrad_eq_dot x₁ _ o cc kh kw]
  -- collapse the `if ci = o` conv-channel selector
  simp only [ite_mul, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq',
    Finset.mem_univ, ite_true]

-- ════════════════════════════════════════════════════════════════
-- § Drift transport: conv → relu → pool → dense → relu → dense → logits
-- ════════════════════════════════════════════════════════════════

/-- Per-entry conv drift, flat-index form of `conv2d_kernel_drift_total`. -/
private theorem conv2d_flat_kernel_drift_total {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW))
    (k : Fin (oc * h * w)) :
    |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x) k -
      Tensor3.flatten (conv2d (Kernel4.unflatten v) b x) k| ≤
      a * ∑ idx, |e idx| := by
  obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
  rw [flatten_t3Idx, flatten_t3Idx]
  exact conv2d_kernel_drift_total b x ha hx v e o hi wi

/-- `ℓ1` conv drift, flat-index form of `conv2d_kernel_drift_sum`. -/
private theorem conv2d_flat_kernel_drift_sum {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW)) :
    ∑ k, |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x) k -
        Tensor3.flatten (conv2d (Kernel4.unflatten v) b x) k| ≤
      ((h * w : ℕ) : ℝ) * (a * ∑ idx, |e idx|) := by
  rw [sum_t3 (fun k : Fin (oc * h * w) =>
    |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x) k -
      Tensor3.flatten (conv2d (Kernel4.unflatten v) b x) k|)]
  calc ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
        |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x)
            (t3Idx o hi wi) -
          Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)
            (t3Idx o hi wi)|
      = ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
          |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
            conv2d (Kernel4.unflatten v) b x o hi wi| := by
        refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [flatten_t3Idx, flatten_t3Idx]
    _ ≤ ((h * w : ℕ) : ℝ) * (a * ∑ idx, |e idx|) :=
        conv2d_kernel_drift_sum b x ha hx v e

-- The chain below is stated for any parameter map `Z` into conv2's pre-activation that
-- moves it by at most `ρ·‖e‖₁` per entry and `(2h)·(2w)·ρ·‖e‖₁` in `ℓ1`: the conv2 kernel
-- (`ρ = a`, the input bound) and the conv2 bias (`ρ = 1`) are the two instances.

/-- Row mass of the conv kernel Jacobian: kernel tap `(o,cc,kh,kw)` reads output channel `o`
    only, through one bounded input read per output position. -/
private theorem convPad_row_l1 {ic oc h w kH kW : Nat} (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (o : Fin oc) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    ∑ ci : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
      |if ci = o then convPad kH kW x cc kh kw hi wi else 0| ≤ ((h * w : ℕ) : ℝ) * a := by
  simp only [apply_ite (fun t : ℝ => |t|), abs_zero, Finset.sum_ite_irrel, Finset.sum_const_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  exact (Finset.sum_le_sum fun hi _ => Finset.sum_le_sum fun wi _ =>
    abs_convPad_le x ha hx cc kh kw hi wi).trans_eq (by simp [mul_assoc])

/-- Row mass of the conv bias Jacobian: bias entry `o` feeds output channel `o` at every
    position. -/
private theorem biasRow_l1 {oc h w : Nat} (o : Fin oc) :
    ∑ ci : Fin oc, ∑ _hi : Fin h, ∑ _wi : Fin w, |if ci = o then (1:ℝ) else 0| ≤
      ((h * w : ℕ) : ℝ) * 1 := by
  simp [apply_ite (fun t : ℝ => |t|), Finset.sum_ite_irrel]

namespace Conv2Slot

/-- **Pooled `ℓ1` drift**: the conv2 output moves by `(2h)·(2w)·ρ·‖e‖₁` in `ℓ1` (`hZ1`);
    relu and the pool are `ℓ1` contractions. -/
private theorem pool_l1_drift {P c h w : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (S : Vec (c * (2*h) * (2*w)) → Vec (c * h * w))
    (hS : ∀ u u', ∑ q, |S u q - S u' q| ≤ ∑ k, |u k - u' k|)
    {ρ : ℝ}
    (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (v e : Vec P) :
    ∑ q, |S (relu (c * (2*h) * (2*w))
          (Z (v + e))) q -
        S (relu (c * (2*h) * (2*w))
          (Z v)) q| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|) :=
  le_trans (hS _ _)
    (le_trans (Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k)
      (hZ1 v e))

/-- Per-entry POST-relu tensor drift — what crossing conv2 as a function of its input
    consumes (`Conv1Slot`). -/
private theorem postrelu_close {P c h w : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w))) {ρ : ℝ}
    (hZ : ∀ v e k, |Z (v + e) k - Z v k| ≤ ρ * ∑ idx, |e idx|) (v e : Vec P)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Z (v + e))) :
          Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Z v)) :
          Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      ρ * ∑ idx, |e idx| := by
  rw [unflatten_t3Idx, unflatten_t3Idx]
  exact le_trans (relu_entry_lipschitz _ _ _ _)
    (hZ v e _)

/-- Per-entry drift of the relu₃ pre-activation. -/
private theorem z3_drift {P c h w d₃ : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (S : Vec (c * (2*h) * (2*w)) → Vec (c * h * w))
    (hS : ∀ u u', ∑ q, |S u q - S u' q| ≤ ∑ k, |u k - u' k|)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {ρ w₃ : ℝ} (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (v e : Vec P) (l : Fin d₃) :
    |dense W₃ b₃ (S (relu (c * (2*h) * (2*w))
        (Z (v + e)))) l -
      dense W₃ b₃ (S (relu (c * (2*h) * (2*w))
        (Z v))) l| ≤
      w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|)) :=
  le_trans (dense_input_drift W₃ b₃ hW₃ _ _ l)
    (mul_le_mul_of_nonneg_left (pool_l1_drift Z S hS hZ1 v e) hw₃)

/-- Per-entry drift of the relu₄ pre-activation. -/
private theorem z4_drift {P c h w d₃ d₄ : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (S : Vec (c * (2*h) * (2*w)) → Vec (c * h * w))
    (hS : ∀ u u', ∑ q, |S u q - S u' q| ≤ ∑ k, |u k - u' k|)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {ρ w₃ w₄ : ℝ} (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (v e : Vec P) (q : Fin d₄) :
    |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
        (relu (c * (2*h) * (2*w)) (Z (v + e)))))) q -
      dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
        (relu (c * (2*h) * (2*w)) (Z v))))) q| ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (ρ * ∑ idx, |e idx|)))) := by
  refine le_trans (dense_input_drift W₄ b₄ hW₄ _ _ q)
    (mul_le_mul_of_nonneg_left ?_ hw₄)
  calc ∑ l, |relu d₃ (dense W₃ b₃ (S
          (relu (c * (2*h) * (2*w)) (Z (v + e))))) l -
        relu d₃ (dense W₃ b₃ (S
          (relu (c * (2*h) * (2*w)) (Z v)))) l|
      ≤ ∑ l, |dense W₃ b₃ (S
            (relu (c * (2*h) * (2*w)) (Z (v + e)))) l -
          dense W₃ b₃ (S
            (relu (c * (2*h) * (2*w)) (Z v))) l| :=
        Finset.sum_le_sum fun l _ => relu_entry_lipschitz _ _ _ l
    _ ≤ (d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))) :=
        (Finset.sum_le_card_nsmul _ _ _ fun l _ =>
          z3_drift Z S hS W₃ b₃ hZ1 hw₃ hW₃ v e l).trans_eq (by simp)

/-- **Logit drift through the whole conv2 chain**: parameter perturbation →
    conv2 output → relu → pool → d₃ → relu → d₄ → relu → d₅. Each dense crossing
    contributes its `ℓ1→ℓ1` operator factor `dᵢ·wᵢ`; the conv output contributes
    the weight-sharing multiplicity `(2h)·(2w)`. -/
private theorem logit_drift {P c h w d₃ d₄ nC : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (S : Vec (c * (2*h) * (2*w)) → Vec (c * h * w))
    (hS : ∀ u u', ∑ q, |S u q - S u' q| ≤ ∑ k, |u k - u' k|)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    {ρ w₃ w₄ w₅ : ℝ} (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (v e : Vec P) (k : Fin nC) :
    |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (S (relu (c * (2*h) * (2*w)) (Z (v + e)))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (S (relu (c * (2*h) * (2*w)) (Z v))))))) k| ≤
      w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|)))))) := by
  refine le_trans (dense_input_drift W₅ b₅ hW₅ _ _ k)
    (mul_le_mul_of_nonneg_left ?_ hw₅)
  calc ∑ q, |relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
          (relu (c * (2*h) * (2*w)) (Z (v + e))))))) q -
        relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
          (relu (c * (2*h) * (2*w)) (Z v)))))) q|
      ≤ ∑ q, |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
            (relu (c * (2*h) * (2*w)) (Z (v + e)))))) q -
          dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
            (relu (c * (2*h) * (2*w)) (Z v))))) q| :=
        Finset.sum_le_sum fun q _ => relu_entry_lipschitz _ _ _ q
    _ ≤ (d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (ρ * ∑ idx, |e idx|))))) :=
        (Finset.sum_le_card_nsmul _ _ _ fun q _ =>
          z4_drift Z S hS W₃ b₃ W₄ b₄ hZ1 hw₃ hW₃ hw₄ hW₄ v e q).trans_eq (by simp)

/-- The relu₃ margin keeps the first head pre-activation off the kink,
    same sign, along the whole step segment. -/
private theorem margin3_keeps_offkink {P c h w d₃ : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (S : Vec (c * (2*h) * (2*w)) → Vec (c * h * w))
    (hS : ∀ u u', ∑ q, |S u q - S u' q| ≤ ∑ k, |u k - u' k|)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {ρ w₃ D : ℝ} (hρ : 0 ≤ ρ) (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (v e : Vec P) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)) <
      |dense W₃ b₃ (S (relu (c * (2*h) * (2*w))
        (Z v))) l|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (l : Fin d₃) :
    dense W₃ b₃ (S (relu (c * (2*h) * (2*w))
        (Z (v + t • e))))
        l ≠ 0 ∧
      (0 < dense W₃ b₃ (S (relu (c * (2*h) * (2*w))
          (Z (v + t • e))))
          l ↔
        0 < dense W₃ b₃ (S (relu (c * (2*h) * (2*w))
          (Z v))) l) := by
  refine sign_stable_of_close ?_ (hm l)
  have h1 := z3_drift Z S hS W₃ b₃ hZ1 hw₃ hW₃ v (t • e) l
  have h2 : w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |(t • e) idx|)) ≤
      w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)) :=
    by gcongr; exact smul_l1_mass_le e ht0 ht1 he
  linarith

/-- The relu₄ margin keeps the second head pre-activation off the kink,
    same sign, along the whole step segment. -/
private theorem margin4_keeps_offkink {P c h w d₃ d₄ : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (S : Vec (c * (2*h) * (2*w)) → Vec (c * h * w))
    (hS : ∀ u u', ∑ q, |S u q - S u' q| ≤ ∑ k, |u k - u' k|)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {ρ w₃ w₄ D : ℝ} (hρ : 0 ≤ ρ) (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (v e : Vec P) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (ρ * D)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
        (relu (c * (2*h) * (2*w)) (Z v))))) q|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (q : Fin d₄) :
    dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
        (relu (c * (2*h) * (2*w)) (Z (v + t • e)))))) q ≠ 0 ∧
      (0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
          (relu (c * (2*h) * (2*w)) (Z (v + t • e)))))) q ↔
        0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (S
          (relu (c * (2*h) * (2*w)) (Z v))))) q) := by
  refine sign_stable_of_close ?_ (hm q)
  have h1 := z4_drift Z S hS W₃ b₃ W₄ b₄ hZ1 hw₃ hW₃ hw₄ hW₄
    v (t • e) q
  have h2 : w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
      (ρ * ∑ idx, |(t • e) idx|)))) ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)))) :=
    by gcongr; exact smul_l1_mass_le e ht0 ht1 he
  linarith

end Conv2Slot

-- ════════════════════════════════════════════════════════════════
-- § The head-gradient drift under frozen masks
-- ════════════════════════════════════════════════════════════════

/-- **Frozen-mask head-gradient drift**: with the two head masks frozen
    (0/1-valued, shared between the two points) and the softmax drifting
    by at most `Δ`, the head3 gradient closed form drifts by at most
    `d₃·w₃·d₄·w₄·nC·w₅·Δ` — the oneHot cancels in the difference. -/
theorem head3_sum_drift {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃)
    (W₄ : Mat d₃ d₄) (W₅ : Mat d₄ nC) {w₃ w₄ w₅ Δ : ℝ}
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (m₃ : Fin d₃ → ℝ) (hm₃ : ∀ l, |m₃ l| ≤ 1)
    (m₄ : Fin d₄ → ℝ) (hm₄ : ∀ r, |m₄ r| ≤ 1)
    (s s' oh : Vec nC) (hs : ∀ k, |s' k - s k| ≤ Δ) (q : Fin p) :
    |(∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - oh k)))) -
      ∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s k - oh k)))| ≤
      (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))))) := by
  have hcoll : (∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - oh k)))) -
      (∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s k - oh k)))) =
      ∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - s k))) := by
    simp only [← Finset.sum_sub_distrib, ← mul_sub, sub_sub_sub_cancel_right]
  rw [hcoll]
  have hinner : ∀ r, |∑ k, W₅ r k * (s' k - s k)| ≤
      (nC : ℝ) * (w₅ * Δ) := by
    intro r
    calc |∑ k, W₅ r k * (s' k - s k)|
        ≤ ∑ k, |W₅ r k * (s' k - s k)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ (nC : ℝ) * (w₅ * Δ) :=
          (Finset.sum_le_card_nsmul _ _ _ fun k _ => (abs_mul _ _).trans_le
            (mul_le_mul (hW₅ r k) (hs k) (abs_nonneg _) hw₅)).trans_eq (by simp)
  have hmid : ∀ l, |∑ r, W₄ l r *
      (m₄ r * ∑ k, W₅ r k * (s' k - s k))| ≤
      (d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))) := by
    intro l
    calc |∑ r, W₄ l r * (m₄ r * ∑ k, W₅ r k * (s' k - s k))|
        ≤ ∑ r, |W₄ l r * (m₄ r * ∑ k, W₅ r k * (s' k - s k))| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ (d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))) :=
          (Finset.sum_le_card_nsmul _ _ _ fun r _ => (abs_mul _ _).trans_le
            (mul_le_mul (hW₄ l r) ((abs_mul _ _).trans_le ((mul_le_of_le_one_left
              (abs_nonneg _) (hm₄ r)).trans (hinner r))) (abs_nonneg _) hw₄)).trans_eq (by simp)
  calc |∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - s k)))|
      ≤ ∑ l, |W₃ q l * (m₃ l * ∑ r, W₄ l r *
          (m₄ r * ∑ k, W₅ r k * (s' k - s k)))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))))) :=
        (Finset.sum_le_card_nsmul _ _ _ fun l _ => (abs_mul _ _).trans_le
          (mul_le_mul (hW₃ q l) ((abs_mul _ _).trans_le ((mul_le_of_le_one_left
            (abs_nonneg _) (hm₃ l)).trans (hmid l))) (abs_nonneg _) hw₃)).trans_eq (by simp)

-- ════════════════════════════════════════════════════════════════
-- § Segment-Lipschitz gradient for the conv2 loss, explicit constant
-- ════════════════════════════════════════════════════════════════

namespace Conv2Slot

/-- **The loss of a conv2-slot parameter map**: a parameter map `Z` into conv2's pre-activation,
    then relu, the 2×2 max-pool and the 3-dense head. Every CNN descent rung's loss has this
    shape: the conv2 kernel and bias (`Z` affine in the parameter), and through `Conv1Slot` the
    conv1 kernel and bias. -/
noncomputable def loss {P c h w d₃ d₄ nC : Nat} (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) : Vec P → ℝ :=
  fun v' => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
    (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Z v')))))))) label

/-- **Near a point, the pooled loss is the gather loss.** If at `p` every window of `Z p` is
    strictly negative, or has the cell `σ` selects strictly above every other cell except its
    `T`-twins (cells equal at every parameter value, `hT`), the strict inequalities persist on a
    neighbourhood and the twins stay equal, so the relu'd pool and the relu'd gather agree near
    `p`. This is where a tied window stops mattering: the pool has no derivative at a tie, but
    a tie between twins is a tie along every parameter direction, and the gather is linear. -/
theorem maxPool_relu_eventuallyEq_gather {P c h w : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w))) (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (hT : ∀ a b, T a b → ∀ v' ci, Z v' (t3Idx ci a.1 a.2) = Z v' (t3Idx ci b.1 b.2))
    (p : Vec P) (hZc : ContinuousAt Z p)
    (hp : ∀ ci ho wo,
      (∀ cd : Fin 2 × Fin 2, Z p (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) < 0) ∨
      ∀ cd : Fin 2 × Fin 2, cd = σ ci ho wo ∨
        T (winRowInv ho (σ ci ho wo).1, winColInv wo (σ ci ho wo).2)
          (winRowInv ho cd.1, winColInv wo cd.2) ∨
        Z p (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) <
          Z p (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2))) :
    ∀ᶠ v' in nhds p, maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Z v')) =
      poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v')) := by
  have hc : ∀ k, ContinuousAt (fun v' => Z v' k) p :=
    fun k => (continuous_apply k).continuousAt.comp hZc
  have hev : ∀ᶠ v' in nhds p, ∀ (ci : Fin c) (ho : Fin h) (wo : Fin w) (cd : Fin 2 × Fin 2),
      relu (c * (2*h) * (2*w)) (Z v') (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) ≤
        relu (c * (2*h) * (2*w)) (Z v')
          (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2)) := by
    simp only [Filter.eventually_all, relu_apply_eq_max]
    intro ci ho wo cd
    rcases hp ci ho wo with hdead | hlive
    · filter_upwards [(hc _).eventually_lt continuousAt_const (hdead cd)] with v' h1
      exact max_le (h1.le.trans (le_max_right _ _)) (le_max_right _ _)
    · rcases hlive cd with hcd | htw | hlt
      · exact Filter.Eventually.of_forall fun _ => by rw [hcd]
      · exact Filter.Eventually.of_forall fun v' => by rw [hT _ _ htw v' ci]
      · filter_upwards [(hc _).eventually_lt (hc _) hlt] with v' h
        exact max_le_max h.le le_rfl
  filter_upwards [hev] with v' hv'
  exact maxPoolFlat_eq_poolGatherFlat σ _ hv'

/-- **The margin up to twins holds along the whole step segment, in strict form**, at the
    selection `σ` = the base point's window argmax: a dead window stays strictly negative (the
    relu₂ margin), and a non-twin cell stays strictly below the selected one (the `2ρD` gap
    against a per-entry drift of at most `ρD`). -/
theorem marginUpTo_seg {P c h w : Nat} (Z : Vec P → Vec (c * (2*h) * (2*w))) {ρ D : ℝ}
    (hρ : 0 ≤ ρ) (hZ : ∀ v e k, |Z (v + e) k - Z v k| ≤ ρ * ∑ idx, |e idx|)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (v d : Vec P) (hd : (∑ idx, |d idx|) ≤ D) (hm2 : ∀ k, ρ * D < |Z v k|)
    (hmq : MaxPool2MarginQUpTo (ρ * D) T (Tensor3.unflatten (Z v)))
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (ci : Fin c) (ho : Fin h) (wo : Fin w) :
    (∀ cd : Fin 2 × Fin 2,
        Z (v + t • d) (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) < 0) ∨
      ∀ cd : Fin 2 × Fin 2, cd = maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo ∨
        T (winRowInv ho (maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo).1,
            winColInv wo (maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo).2)
          (winRowInv ho cd.1, winColInv wo cd.2) ∨
        Z (v + t • d) (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) <
          Z (v + t • d) (t3Idx ci
            (winRowInv ho (maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo).1)
            (winColInv wo (maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo).2)) := by
  have hρD0 : 0 ≤ ρ * D := mul_nonneg hρ (le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd)
  have hclose : ∀ k, |Z (v + t • d) k - Z v k| ≤ ρ * D := fun k =>
    (hZ v (t • d) k).trans (mul_le_mul_of_nonneg_left (smul_l1_mass_le d ht0 ht1 hd) hρ)
  rcases hmq ci ho wo with hdead | hlive
  · refine Or.inl fun cd => ?_
    have hle : Z v (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) ≤ 0 := hdead cd
    have hst := margin_keeps_offkink_of_drift Z hρ hZ v d hd hm2 t ht0 ht1
      (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2))
    exact lt_of_le_of_ne (not_lt.mp fun h => not_lt.mpr hle (hst.2.mp h)) hst.1
  · refine Or.inr fun cd => ?_
    by_cases hcd : cd = maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo
    · exact Or.inl hcd
    refine Or.inr ?_
    have hpos : (winRowInv ho (maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo).1,
        winColInv wo (maxPool2Argmax (Tensor3.unflatten (Z v)) ci ho wo).2) ≠
        (winRowInv ho cd.1, winColInv wo cd.2) := fun h => hcd (by
      obtain ⟨h1, h2⟩ := Prod.mk.inj h
      have e1 := congrArg winRowMod h1
      have e2 := congrArg winColMod h2
      rw [winRowMod_winRowInv, winRowMod_winRowInv] at e1
      rw [winColMod_winColInv, winColMod_winColInv] at e2
      exact Prod.ext e1.symm e2.symm)
    rcases hlive _ cd hpos (maxPool2Argmax_max (Tensor3.unflatten (Z v)) ci ho wo) with hgap | htw
    · exact Or.inr (lt_of_lt_gap_of_close hgap (hclose _) (hclose _))
    · exact Or.inl htw

/-- The head above the conv2 pre-activation through the gather, `CE∘head3∘gather∘relu`, is
    differentiable wherever the three ReLU stages are off their kinks. -/
private theorem gather_head_differentiableAt {c h w d₃ d₄ nC : Nat}
    (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₂ : Vec (c * (2*h) * (2*w))) (hz2 : ∀ k, z₂ k ≠ 0)
    (hz3 : ∀ l, dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) z₂)) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
      (relu (c * (2*h) * (2*w)) z₂)))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (poolGatherFlat σ (relu (c * (2*h) * (2*w)) y))))))) label)
      z₂ := by
  fun_prop (disch := assumption)

/-- **The gather-loss gradient through a parameter map**: `Z`'s Jacobian row contracted with the
    gather head's input gradient (`gradAt_comp_t3` and `gather_relu_input_grad`). -/
private theorem gather_loss_gradAt {P c h w d₃ d₄ nC : Nat}
    (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (v : Vec P)
    (hZd : DifferentiableAt ℝ Z v) (hz2 : ∀ k, Z v k ≠ 0)
    (hz3 : ∀ l, dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
      (relu (c * (2*h) * (2*w)) (Z v))))) q ≠ 0)
    (idx : Fin P) :
    gradAt (fun v' : Vec P => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v')))))))) label) v idx
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w), pdiv Z v idx (t3Idx ci hi wi) *
          ((if Z v (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
            (if σ ci (winRow hi) (winCol wi) = (winRowMod hi, winColMod wi)
              then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
                ((if dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l > 0
                    then (1:ℝ) else 0) *
                  ∑ q, W₄ l q *
                    ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
                          (relu (c * (2*h) * (2*w)) (Z v))))) q > 0
                        then (1:ℝ) else 0) *
                      ∑ k, W₅ q k *
                        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                            (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
                              (relu (c * (2*h) * (2*w)) (Z v)))))))) k -
                          oneHot nC label k)))
              else 0)) :=
  (gradAt_comp_t3 Z (fun y => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) y))))))) label) v hZd
    (gather_head_differentiableAt σ W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hz3 hz4) idx).trans
    (Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl fun hi _ =>
      Finset.sum_congr rfl fun wi _ => by
        rw [gather_relu_input_grad σ W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hz3 hz4 ci hi wi])

/-- **Segment-Lipschitz gradient for a conv2-slot loss through a fixed gather, explicit
    constant.** For a parameter map `Z` into conv2's pre-activation with per-entry drift
    `ρ·‖e‖₁` (`hZ`) and `ℓ1` drift `(2h)·(2w)·ρ·‖e‖₁` (`hZ1`), differentiable with Jacobian
    row `J` (row mass `≤ (2h)·(2w)·ρ`, `hJ`) wherever `Q` holds (`hpd`): with the pool replaced by
    the gather `σ`, under the relu₂, relu₃ and relu₄ margins at radius `ρ·D` every mask freezes
    along `[v, v+d]`, the route is fixed by construction, the Jacobian factors out, and the
    difference collapses to the softmax drift. `Q` is needed only at the two ends of the
    segment (`hQv`, `hQt`): the conv2 rungs take `Q := True`, the conv1 slot `Q` = relu₁'s
    signs frozen. `sgd_descends` moves this onto the pooled loss. -/
theorem gather_grad_lipschitz {P c h w d₃ d₄ nC : Nat}
    (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    {ρ w₃ w₄ w₅ D : ℝ} (hρ : 0 ≤ ρ) (hZ : ∀ v e k, |Z (v + e) k - Z v k| ≤ ρ * ∑ idx, |e idx|)
    (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (J : Fin c → Fin (2*h) → Fin (2*w) → ℝ)
    (hJ : ∑ ci, ∑ hi, ∑ wi, |J ci hi wi| ≤ ((2*h * (2*w) : ℕ) : ℝ) * ρ)
    (idx : Fin P) (Q : Vec P → Prop)
    (hpd : ∀ v' : Vec P, Q v' → DifferentiableAt ℝ Z v' ∧
      ∀ ci hi wi, pdiv Z v' idx (t3Idx ci hi wi) = J ci hi wi)
    (v d : Vec P) (hd : (∑ idx, |d idx|) ≤ D)
    (hm2 : ∀ k, ρ * D < |Z v k|)
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)) <
      |dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
        (relu (c * (2*h) * (2*w)) (Z v))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D))))))) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1) (hQv : Q v) (hQt : Q (v + t • d)) :
    |gradAt (fun v' : Vec P => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v')))))))) label)
        (v + t • d) idx -
      gradAt (fun v' : Vec P => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v')))))))) label)
        v idx| ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * ρ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D))))))))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have hρD0 : 0 ≤ ρ * D := mul_nonneg hρ hD0
  -- the logit-drift constant at the step radius, the object every bound below is written in
  set δ : ℝ := w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)))))) with hδ
  have hδ0 : (0:ℝ) ≤ δ :=
    mul_nonneg hw₅ (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) hρD0)))))
  have hden : (0:ℝ) < 1 - 2 * δ := by linarith
  have hS := poolGatherFlat_l1_contract σ
  -- base-point conditions from the margins
  have hz2_v : ∀ k, Z v k ≠ 0 := fun k => abs_pos.mp (hρD0.trans_lt (hm2 k))
  have hz3_v : ∀ l, dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l ≠ 0 :=
    fun l => abs_pos.mp ((mul_nonneg hw₃ (mul_nonneg (Nat.cast_nonneg _) hρD0)).trans_lt (hm3 l))
  have hz4_v : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
      (relu (c * (2*h) * (2*w)) (Z v))))) q ≠ 0 :=
    fun q =>
      abs_pos.mp ((mul_nonneg hw₄ (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) hρD0)))).trans_lt (hm4 q))
  -- segment-point conditions: every mask frozen
  have hstab2 := fun k =>
    margin_keeps_offkink_of_drift Z hρ hZ v d hd hm2 t ht0 ht1 k
  have hz2_t : ∀ k, Z (v + t • d) k ≠ 0 := fun k => (hstab2 k).1
  have hstab3 := fun l =>
    margin3_keeps_offkink Z (poolGatherFlat σ) hS W₃ b₃ hρ hZ1 hw₃ hW₃ v d hd hm3 t ht0 ht1 l
  have hz3_t : ∀ l, dense W₃ b₃ (poolGatherFlat σ
      (relu (c * (2*h) * (2*w)) (Z (v + t • d)))) l ≠ 0 := fun l => (hstab3 l).1
  have hstab4 := fun q =>
    margin4_keeps_offkink Z (poolGatherFlat σ) hS W₃ b₃ W₄ b₄ hρ hZ1 hw₃ hW₃ hw₄ hW₄
      v d hd hm4 t ht0 ht1 q
  have hz4_t : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
      (relu (c * (2*h) * (2*w)) (Z (v + t • d)))))) q ≠ 0 := fun q => (hstab4 q).1
  -- both gradients in closed form, the Jacobian rows replaced by the fixed `J`
  rw [gather_loss_gradAt σ Z W₃ b₃ W₄ b₄ W₅ b₅ label (v + t • d) (hpd _ hQt).1 hz2_t hz3_t hz4_t,
    gather_loss_gradAt σ Z W₃ b₃ W₄ b₄ W₅ b₅ label v (hpd _ hQv).1 hz2_v hz3_v hz4_v]
  simp only [(hpd _ hQt).2, (hpd _ hQv).2]
  -- the frozen masks
  have hmask2 : ∀ (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)),
      (if Z (v + t • d) (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) =
      (if Z v (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) :=
    fun ci hi wi => if_congr (hstab2 _).2 rfl rfl
  have hmask3 : ∀ l : Fin d₃,
      (if dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z (v + t • d)))) l > 0
        then (1:ℝ) else 0) =
      (if dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l > 0
        then (1:ℝ) else 0) :=
    fun l => if_congr (hstab3 l).2 rfl rfl
  have hmask4 : ∀ q : Fin d₄,
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
          (relu (c * (2*h) * (2*w)) (Z (v + t • d)))))) q > 0 then (1:ℝ) else 0) =
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
          (relu (c * (2*h) * (2*w)) (Z v))))) q > 0 then (1:ℝ) else 0) :=
    fun q => if_congr (hstab4 q).2 rfl rfl
  -- the softmax drift along the segment
  have hzdrift : ∀ k, |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w))
        (Z (v + t • d)))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))))))) k| ≤
      t * δ := by
    intro k
    have h1 := logit_drift Z (poolGatherFlat σ) hS W₃ b₃ W₄ b₄ W₅ b₅ hZ1
      hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ v (t • d) k
    rw [smul_l1_mass d ht0] at h1
    have h2 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (ρ * (t * ∑ idx, |d idx|))))))) =
        t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |d idx|))))))) := by
      ring
    rw [h2] at h1
    have h3 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |d idx|)))))) ≤ δ :=
      by rw [hδ]; gcongr
    have h4 := mul_le_mul_of_nonneg_left h3 ht0
    linarith
  have hS' := softmax_seg_drift _ _ ht0 ht1 hδ0 hsmall hzdrift
  have hΔ0 : (0:ℝ) ≤ 2 * (t * δ) / (1 - 2 * δ) :=
    div_nonneg (mul_nonneg (by norm_num) (mul_nonneg ht0 hδ0)) hden.le
  have hM0 : (0:ℝ) ≤ (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
      (w₅ * (2 * (t * δ) / (1 - 2 * δ))))))) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₅ hΔ0)))))
  -- the endgame: combine, freeze, collapse to the softmax drift
  have hfinal : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      (|J ci hi wi| *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
          (w₅ * (2 * (t * δ) / (1 - 2 * δ))))))))) ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * ρ ^ 2 /
        (1 - 2 * δ)) * (t * D) := by
    calc ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
        (|J ci hi wi| *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * δ) / (1 - 2 * δ)))))))))
        = (∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w), |J ci hi wi|) *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * δ) / (1 - 2 * δ)))))))) := by
          simp only [← Finset.sum_mul]
      _ ≤ (((2*h * (2*w) : ℕ) : ℝ) * ρ) *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * δ) / (1 - 2 * δ)))))))) :=
          mul_le_mul_of_nonneg_right hJ hM0
      _ = (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
            (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * ρ ^ 2 /
            (1 - 2 * δ)) * (t * D) := by
          ring
  refine le_trans (le_trans (by
    rw [← Finset.sum_sub_distrib]
    refine le_trans (le_of_eq (congrArg abs (Finset.sum_congr rfl
      fun ci _ => by rw [← Finset.sum_sub_distrib]))) ?_
    refine le_trans (le_of_eq (congrArg abs (Finset.sum_congr rfl
      fun ci _ => Finset.sum_congr rfl fun hi _ => by
        rw [← Finset.sum_sub_distrib]))) ?_
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
    exact Finset.sum_le_sum fun ci _ => le_trans
      (Finset.abs_sum_le_sum_abs _ _)
      (Finset.sum_le_sum fun hi _ => Finset.abs_sum_le_sum_abs _ _))
    (Finset.sum_le_sum fun ci _ => Finset.sum_le_sum fun hi _ =>
      Finset.sum_le_sum fun wi _ => ?_)) hfinal
  -- per-term: freeze the masks, the route is fixed; collapse to the drift
  rw [hmask2 ci hi wi]
  simp only [hmask3, hmask4]
  by_cases hA : σ ci (winRow hi) (winCol wi) = (winRowMod hi, winColMod wi)
  · rw [ite_eq_left hA, ite_eq_left hA, ← mul_sub, abs_mul, ← mul_sub, abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
    refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_) ?_
    · split_ifs <;> simp
    · exact head3_sum_drift W₃ W₄ W₅ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
        (fun l => if dense W₃ b₃ (poolGatherFlat σ
          (relu (c * (2*h) * (2*w)) (Z v))) l > 0 then (1:ℝ) else 0)
        (fun l => by split_ifs <;> simp)
        (fun q => if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
          (relu (c * (2*h) * (2*w)) (Z v))))) q > 0 then (1:ℝ) else 0)
        (fun q => by split_ifs <;> simp)
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v)))))))))
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w))
            (Z (v + t • d))))))))))
        (oneHot nC label) hS' (t3Idx ci (winRow hi) (winCol wi))
  · rw [ite_eq_right hA, ite_eq_right hA]
    simp only [mul_zero, sub_self, abs_zero]
    exact mul_nonneg (abs_nonneg _) hM0

/-- **One inexact SGD step through a conv2 slot decreases the loss — margin up to twins.**
    For a parameter map `Z` into conv2's pre-activation with per-entry drift `ρ·‖e‖₁` and `ℓ1`
    drift `(2h)·(2w)·ρ·‖e‖₁`, differentiable with Jacobian rows `J` (row mass `≤ (2h)·(2w)·ρ`)
    wherever `Q` holds along the step, and twins `T` (cells equal at every parameter value,
    `hT`): under the relu₂ margin, the pool margin up to twins `MaxPool2MarginQUpTo`, the relu₃
    and relu₄ margins (all at radius `ρ·D`, `D` the step radius), the small-step condition and
    the two dominance conditions, one step with an `η`-accurate gradient oracle decreases `loss`
    by `≥ lr·‖∇loss‖₂²/2`.

    The pool needs no derivative of its own. Fix `σ` = the base point's window argmax. Along
    the segment every window is dead or has `σ`'s cell strictly above its non-twins
    (`marginUpTo_seg`), so near every segment point the loss IS the gather loss
    (`maxPool_relu_eventuallyEq_gather`); the gather loss is differentiable there and its
    gradient is segment-Lipschitz (`gather_grad_lipschitz`), and both facts transfer to `loss`
    through the local equality. Every CNN rung is this lemma at its `Z`, `ρ`, `J` and `Q`. -/
theorem sgd_descends {P c h w d₃ d₄ nC : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (v gh : Vec P) {ρ lr η w₃ w₄ w₅ : ℝ} (hρ : 0 ≤ ρ)
    (hZ : ∀ v e k, |Z (v + e) k - Z v k| ≤ ρ * ∑ idx, |e idx|)
    (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (J : Fin P → Fin c → Fin (2*h) → Fin (2*w) → ℝ)
    (hJ : ∀ idx, ∑ ci, ∑ hi, ∑ wi, |J idx ci hi wi| ≤ ((2*h * (2*w) : ℕ) : ℝ) * ρ)
    (Q : Vec P → Prop)
    (hpd : ∀ v' : Vec P, Q v' → DifferentiableAt ℝ Z v' ∧
      ∀ idx ci hi wi, pdiv Z v' idx (t3Idx ci hi wi) = J idx ci hi wi)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (hT : ∀ a b, T a b → ∀ v' ci, Z v' (t3Idx ci a.1 a.2) = Z v' (t3Idx ci b.1 b.2))
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - gradAt (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v idx| ≤ η)
    (hQ : ∀ t ∈ Set.Icc (0:ℝ) 1, Q (v + t • (-(lr • gh))))
    (hm2 : ∀ k, ρ * stepRadius (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η < |Z v k|)
    (hmq : MaxPool2MarginQUpTo (ρ * stepRadius (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η) T
      (Tensor3.unflatten (Z v)))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (ρ * stepRadius (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Z v))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (ρ * stepRadius (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Z v))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
      (ρ * stepRadius (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v idx|) ≤
      lr * (∑ idx, gradAt (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * ρ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (ρ * stepRadius (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η))))))))) *
        stepRadius (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η ^ 2 ≤
      lr * (∑ idx, gradAt (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v idx ^ 2) / 4) :
    loss Z W₃ b₃ W₄ b₄ W₅ b₅ label (v - lr • gh) ≤
      loss Z W₃ b₃ W₄ b₄ W₅ b₅ label v -
        lr * (∑ idx, gradAt (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v idx ^ 2) / 2 := by
  set L := loss Z W₃ b₃ W₄ b₄ W₅ b₅ label with hL
  set D := stepRadius L v lr η with hDdef
  have hD : (∑ idx, |(-(lr • gh)) idx|) ≤ D := sgd_step_l1_le _ gh hlr hgh
  have hD0 : 0 ≤ D := le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hD
  have hρD0 : 0 ≤ ρ * D := mul_nonneg hρ hD0
  set σ := maxPool2Argmax (Tensor3.unflatten (Z v)) with hσ
  have hQ0 : Q v := by simpa using hQ 0 ⟨le_rfl, zero_le_one⟩
  -- near every segment point the pooled loss is the gather loss
  set Lg : Vec P → ℝ := fun v' => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
    (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v')))))))) label with hLg
  have hev : ∀ t ∈ Set.Icc (0:ℝ) 1, L =ᶠ[nhds (v + t • (-(lr • gh)))] Lg := fun t ht =>
    (maxPool_relu_eventuallyEq_gather Z σ T hT _ (hpd _ (hQ t ht)).1.continuousAt
      (marginUpTo_seg Z hρ hZ T v _ hD hm2 hmq t ht.1 ht.2)).mono fun v' hv' => by
      simp only [hL, hLg, loss, hv']
  have hev0 : L =ᶠ[nhds v] Lg := by simpa using hev 0 ⟨le_rfl, zero_le_one⟩
  have hpool_v : maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Z v)) =
      poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v)) := by
    have h0 := marginUpTo_seg Z hρ hZ T v _ hD hm2 hmq 0 le_rfl zero_le_one
    simp only [zero_smul, add_zero] at h0
    exact (maxPool_relu_eventuallyEq_gather Z σ T hT _ (hpd _ hQ0).1.continuousAt h0).self_of_nhds
  have hm3g : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)) <
      |dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l| := by
    rw [← hpool_v]; exact hm3
  have hm4g : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
        (relu (c * (2*h) * (2*w)) (Z v))))) q| := by
    rw [← hpool_v]; exact hm4
  -- the gather loss is differentiable along the segment
  have hdiffg : ∀ t ∈ Set.Icc (0:ℝ) 1, DifferentiableAt ℝ Lg (v + t • (-(lr • gh))) := by
    intro t ht
    have hz2 := fun k =>
      (margin_keeps_offkink_of_drift Z hρ hZ v _ hD hm2 t ht.1 ht.2 k).1
    have hz3 := fun l => (margin3_keeps_offkink Z (poolGatherFlat σ)
      (poolGatherFlat_l1_contract σ) W₃ b₃ hρ hZ1 hw₃ hW₃ v _ hD hm3g t ht.1 ht.2 l).1
    have hz4 := fun q => (margin4_keeps_offkink Z (poolGatherFlat σ)
      (poolGatherFlat_l1_contract σ) W₃ b₃ W₄ b₄ hρ hZ1 hw₃ hW₃ hw₄ hW₄ v _ hD hm4g
        t ht.1 ht.2 q).1
    exact (differentiableAt_pi.mp (gather_head_differentiableAt σ W₃ b₃ W₄ b₄ W₅ b₅ label _
      hz2 hz3 hz4) 0).comp _ (hpd _ (hQ t ht)).1
  have hgrad_eq : ∀ t ∈ Set.Icc (0:ℝ) 1, ∀ i,
      gradAt L (v + t • (-(lr • gh))) i = gradAt Lg (v + t • (-(lr • gh))) i :=
    fun t ht i => by unfold gradAt; rw [(hev t ht).fderiv_eq]
  have hgrad_eq0 : ∀ i, gradAt L v i = gradAt Lg v i :=
    fun i => by unfold gradAt; rw [hev0.fderiv_eq]
  -- the curvature constant
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D))))))) := by linarith
  have hC0 : (0:ℝ) ≤ 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
      (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * ρ ^ 2 /
      (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)))))))) :=
    div_nonneg (by positivity) hden.le
  refine _root_.Proofs.sgd_descends L v gh hlr hη hC0 hgh
    (fun t ht => (hdiffg t ht).congr_of_eventuallyEq (hev t ht)) (fun t ht i => ?_) h1 h2
  rw [hgrad_eq t ht i, hgrad_eq0 i]
  exact gather_grad_lipschitz σ Z W₃ b₃ W₄ b₄ W₅ b₅ label hρ hZ hZ1 hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
    (J i) (hJ i) i Q (fun v' hq => ⟨(hpd v' hq).1, (hpd v' hq).2 i⟩) v _ hD hm2 hm3g hm4g
    hsmall t ht hQ0 (hQ t ht)

/-- **The margin up to twins, strict at its own point**: with every cell nonzero and `δ ≥ 0`, a
    dead window is strictly negative and every non-twin cell is strictly below the window
    argmax. `marginUpTo_seg` at the base point, without the drift. -/
theorem marginUpTo_strict {c h w : Nat} {δ : ℝ} (hδ : 0 ≤ δ) (x : Vec (c * (2*h) * (2*w)))
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop) (hne : ∀ k, x k ≠ 0)
    (hmq : MaxPool2MarginQUpTo δ T (Tensor3.unflatten x)) (ci : Fin c) (ho : Fin h)
    (wo : Fin w) :
    (∀ cd : Fin 2 × Fin 2, x (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) < 0) ∨
      ∀ cd : Fin 2 × Fin 2, cd = maxPool2Argmax (Tensor3.unflatten x) ci ho wo ∨
        T (winRowInv ho (maxPool2Argmax (Tensor3.unflatten x) ci ho wo).1,
            winColInv wo (maxPool2Argmax (Tensor3.unflatten x) ci ho wo).2)
          (winRowInv ho cd.1, winColInv wo cd.2) ∨
        x (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) <
          x (t3Idx ci (winRowInv ho (maxPool2Argmax (Tensor3.unflatten x) ci ho wo).1)
            (winColInv wo (maxPool2Argmax (Tensor3.unflatten x) ci ho wo).2)) := by
  rcases hmq ci ho wo with hdead | hlive
  · exact Or.inl fun cd => lt_of_le_of_ne (hdead cd) (hne _)
  · refine Or.inr fun cd => ?_
    by_cases hcd : cd = maxPool2Argmax (Tensor3.unflatten x) ci ho wo
    · exact Or.inl hcd
    refine Or.inr ?_
    have hpos : (winRowInv ho (maxPool2Argmax (Tensor3.unflatten x) ci ho wo).1,
        winColInv wo (maxPool2Argmax (Tensor3.unflatten x) ci ho wo).2) ≠
        (winRowInv ho cd.1, winColInv wo cd.2) := fun h => hcd (by
      obtain ⟨h1, h2⟩ := Prod.mk.inj h
      have e1 := congrArg winRowMod h1
      have e2 := congrArg winColMod h2
      rw [winRowMod_winRowInv, winRowMod_winRowInv] at e1
      rw [winColMod_winColInv, winColMod_winColInv] at e2
      exact Prod.ext e1.symm e2.symm)
    rcases hlive _ cd hpos (maxPool2Argmax_max (Tensor3.unflatten x) ci ho wo) with hgap | htw
    · exact Or.inr (by have : (0:ℝ) ≤ 2 * δ := by linarith
                       exact lt_of_sub_pos (lt_of_le_of_lt this hgap))
    · exact Or.inl htw

/-- **The loss gradient is at most the Jacobian row mass times the head's operator norm.** At a
    point where the pool margin up to twins holds (at any `δ ≥ 0`) and every ReLU stage is off
    its kink, the loss is the gather loss near the point, so its gradient is `Z`'s Jacobian row
    contracted with the gather head's gradient (`gather_loss_gradAt`); `|softmax − onehot| ≤ 1`
    leaves at most `d₃·w₃·d₄·w₄·nC·w₅` per cell of the row. The bound a concrete instance uses
    for the step radius, which it cannot compute (the softmax is transcendental). -/
theorem gradAt_abs_le {P c h w d₃ d₄ nC : Nat} (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) {δ ρ w₃ w₄ w₅ : ℝ} (hδ : 0 ≤ δ)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (J : Fin c → Fin (2*h) → Fin (2*w) → ℝ)
    (hJ : ∑ ci, ∑ hi, ∑ wi, |J ci hi wi| ≤ ((2*h * (2*w) : ℕ) : ℝ) * ρ) (idx : Fin P)
    (v : Vec P) (hZd : DifferentiableAt ℝ Z v)
    (hpdJ : ∀ ci hi wi, pdiv Z v idx (t3Idx ci hi wi) = J ci hi wi)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (hT : ∀ a b, T a b → ∀ v' ci, Z v' (t3Idx ci a.1 a.2) = Z v' (t3Idx ci b.1 b.2))
    (hz2 : ∀ k, Z v k ≠ 0) (hmq : MaxPool2MarginQUpTo δ T (Tensor3.unflatten (Z v)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Z v))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Z v))))) q ≠ 0) :
    |gradAt (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v idx| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * ρ * ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) := by
  set σ := maxPool2Argmax (Tensor3.unflatten (Z v))
  have hev := maxPool_relu_eventuallyEq_gather Z σ T hT v hZd.continuousAt
    (marginUpTo_strict hδ (Z v) T hz2 hmq)
  have hpool := hev.self_of_nhds
  have hLev : loss Z W₃ b₃ W₄ b₄ W₅ b₅ label =ᶠ[nhds v] fun v' => crossEntropy nC (dense W₅ b₅
      (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
        (relu (c * (2*h) * (2*w)) (Z v')))))))) label :=
    hev.mono fun v' hv' => by simp only [loss, hv']
  rw [hpool] at hz3 hz4
  have hgr : gradAt (loss Z W₃ b₃ W₄ b₄ W₅ b₅ label) v idx =
      gradAt (fun v' => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v')))))))) label) v idx := by
    unfold gradAt; rw [hLev.fderiv_eq]
  rw [hgr, gather_loss_gradAt σ Z W₃ b₃ W₄ b₄ W₅ b₅ label v hZd hz2 hz3 hz4 idx]
  simp only [hpdJ]
  -- the head factor at one cell is at most the head's operator norm
  have hH : ∀ (j : Fin (c * h * w)),
      |∑ l, W₃ j l * ((if dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l > 0
          then (1:ℝ) else 0) *
        ∑ q, W₄ l q * ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
            (relu (c * (2*h) * (2*w)) (Z v))))) q > 0 then (1:ℝ) else 0) *
          ∑ k, W₅ q k * (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v)))))))) k - oneHot nC label k)))| ≤
        (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1))))) := fun j => by
    have h := head3_sum_drift (Δ := 1) W₃ W₄ W₅ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
      (fun l => if dense W₃ b₃ (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))) l > 0
        then (1:ℝ) else 0) (fun l => by split_ifs <;> simp)
      (fun q => if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (poolGatherFlat σ
        (relu (c * (2*h) * (2*w)) (Z v))))) q > 0 then (1:ℝ) else 0) (fun q => by split_ifs <;> simp)
      (oneHot nC label) (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v))))))))) (oneHot nC label)
      (fun k => by
        rw [oneHot_apply]
        have h0 := softmax_nonneg (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v)))))))) k
        have h1 := softmax_le_one (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (poolGatherFlat σ (relu (c * (2*h) * (2*w)) (Z v)))))))) k
        rw [abs_le]; split_ifs <;> constructor <;> linarith) j
    simpa only [sub_self, mul_zero, Finset.sum_const_zero, sub_zero] using h
  have hB : (0:ℝ) ≤ (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1))))) := by positivity
  calc _ ≤ ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
        |J ci hi wi| * ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) := by
        refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun ci _ =>
          (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun hi _ =>
            (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun wi _ => ?_)))
        rw [abs_mul]
        refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
        rw [abs_mul]
        refine (mul_le_of_le_one_left (abs_nonneg _) (by split_ifs <;> simp)).trans ?_
        split_ifs
        · exact hH _
        · simpa using hB
    _ = (∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w), |J ci hi wi|) *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) := by
        simp only [Finset.sum_mul]
    _ ≤ _ := mul_le_mul_of_nonneg_right hJ hB

end Conv2Slot

-- ════════════════════════════════════════════════════════════════
-- § The conv2 capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- The loss as a function of the flattened second-conv kernel. -/
noncomputable def cnnConv2KernelLoss {c h w d₃ d₄ nC kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄)
    (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) : Vec (c * c * kH * kW) → ℝ :=
  fun v' => crossEntropy nC
    (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c *
      (2*h) * (2*w)) (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
    label

/-- **One inexact SGD step on the CNN's second conv kernel decreases one
    example's cross-entropy loss** (example `(x₁, label)` at the frozen conv-2
    input, `W₂` moving, every other parameter fixed). `Conv2Slot.sgd_descends` at the conv2
    kernel map (`ρ = a`, the point-free Jacobian `conv2d_weight_pdiv`). The pool hypothesis is
    the margin up to twins `MaxPool2MarginQUpTo` on the pre-activation `conv2d W₂ b₂ x₁`, twins
    being cells whose zero-padded input patches are identical (`hT`, `ConvPatchEq`): those are
    equal for every kernel, so a window of them stays tied along the step and costs nothing.
    That is the condition real MNIST meets at the trained Chapter-3 weights, where a constant
    background patch ties a window's cells (scripts/probes/mnist_pool_twin_probe.py). The
    remaining hypotheses: the oracle accuracy `η`, the relu₂/relu₃/relu₄ margins at the step
    radius `D = lr·(‖∇L‖₁ + |kernel|·η)`, the small-step condition and the two dominance
    conditions. Conclusion: the loss drops by ≥ `lr·‖∇L‖₂²/2`. The conv-layer peer of
    `mlp_input_sgd_descends`. -/
theorem cnn_conv2_sgd_descends {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec (c * c * kH * kW)) (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr η a w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hT : ∀ p q, T p q → ConvPatchEq kH kW x₁ p q)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) idx| ≤ η)
    (hm2 : ∀ k, a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₂) lr η) <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQUpTo (a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₂) lr η)) T (conv2d W₂ b₂ x₁))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius
      (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr η))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
          (Kernel4.flatten W₂) lr η))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius
        (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr η)))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) idx|) ≤
      lr * (∑ idx, gradAt
        (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius
            (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr η)))))))))) *
        (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
          (Kernel4.flatten W₂) lr η) ^ 2 ≤
      lr * (∑ idx, gradAt
        (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) idx ^ 2) / 4) :
    (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂ - lr • gh) ≤
      (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) -
        lr * (∑ idx, gradAt
          (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
          (Kernel4.flatten W₂) idx ^ 2) / 2 := by
  -- the conv2 kernel map, its drift and its point-free Jacobian
  set Z : Vec (c * c * kH * kW) → Vec (c * (2*h) * (2*w)) :=
    fun v' => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁) with hZdef
  have hZW : Z (Kernel4.flatten W₂) = Tensor3.flatten (conv2d W₂ b₂ x₁) := by
    rw [hZdef]; dsimp only; rw [Kernel4.unflatten_flatten]
  have hpd : ∀ v' : Vec (c * c * kH * kW), True → DifferentiableAt ℝ Z v' ∧
      ∀ idx ci hi wi, pdiv Z v' idx (t3Idx ci hi wi) = pdiv Z 0 idx (t3Idx ci hi wi) :=
    fun v' _ => ⟨conv2d_weight_differentiable b₂ x₁ v', fun idx ci hi wi => by
      obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
      rw [conv2d_weight_pdiv, conv2d_weight_pdiv]⟩
  have hJ : ∀ idx, ∑ ci, ∑ hi, ∑ wi, |pdiv Z 0 idx (t3Idx ci hi wi)| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * a := fun idx => by
    obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
    simp only [hZdef, conv2d_weight_pdiv]
    exact convPad_row_l1 x₁ ha hx o cc kh kw
  have hT' : ∀ p q, T p q → ∀ v' ci, Z v' (t3Idx ci p.1 p.2) = Z v' (t3Idx ci q.1 q.2) :=
    fun p q hpq v' ci => by
      simp only [hZdef, flatten_t3Idx]
      exact conv2d_eq_of_convPatchEq (hT p q hpq) _ _ ci
  exact Conv2Slot.sgd_descends Z W₃ b₃ W₄ b₄ W₅ b₅ label (Kernel4.flatten W₂) gh ha
    (conv2d_flat_kernel_drift_total b₂ x₁ ha hx) (conv2d_flat_kernel_drift_sum b₂ x₁ ha hx)
    hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (fun idx ci hi wi => pdiv Z 0 idx (t3Idx ci hi wi)) hJ
    (fun _ => True) hpd T hT' hlr hη hgh (fun _ _ => trivial)
    (by rw [hZW]; exact hm2) (by rw [hZW, Tensor3.unflatten_flatten]; exact hmq)
    (by rw [hZW]; exact hm3) (by rw [hZW]; exact hm4) hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § Exact-gradient corollaries: every hypothesis at an explicit radius
-- ════════════════════════════════════════════════════════════════

/-- **The exact step's radius, from an `ℓ1` bound on the gradient.** At `η = 0` the step radius
    is `lr·‖∇L‖₁`: nonnegative, at most `lr·G`, and (Cauchy–Schwarz, `(Σ|g|)² ≤ m·Σg²`) its
    square is at most `lr²·m·‖∇L‖₂²`. -/
private theorem stepRadius_exact_le {m : Nat} (L : Vec m → ℝ) (v : Vec m) {lr G : ℝ}
    (hlr : 0 ≤ lr) (hG : (∑ i, |gradAt L v i|) ≤ G) :
    0 ≤ stepRadius L v lr 0 ∧ stepRadius L v lr 0 ≤ lr * G ∧
      stepRadius L v lr 0 ^ 2 ≤ lr ^ 2 * ((m : ℝ) * ∑ i, gradAt L v i ^ 2) := by
  have hCS : (∑ i, |gradAt L v i|) ^ 2 ≤ (m : ℝ) * ∑ i, gradAt L v i ^ 2 := by
    have := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ => (1:ℝ))
      (fun i => |gradAt L v i|)
    simpa [sq_abs, Finset.card_univ] using this
  simp only [stepRadius, mul_zero, add_zero]
  exact ⟨mul_nonneg hlr (Finset.sum_nonneg fun _ _ => abs_nonneg _),
    mul_le_mul_of_nonneg_left hG hlr,
    by rw [mul_pow]; exact mul_le_mul_of_nonneg_left hCS (sq_nonneg _)⟩

/-- `m` entries of size at most `B` have `ℓ1` mass at most `m·B`. -/
private theorem gradAt_l1_le_card {m : Nat} (L : Vec m → ℝ) (v : Vec m) {B : ℝ}
    (hB : ∀ i, |gradAt L v i| ≤ B) : (∑ i, |gradAt L v i|) ≤ (m : ℝ) * B :=
  (Finset.sum_le_card_nsmul _ _ _ fun i _ => hB i).trans_eq
    (by rw [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul])

/-- **The second dominance condition at the exact step, from one at the explicit radius.** With
    the logit-drift constant `δ` no larger at the step radius `s` than at `R`, `s² ≤ lr²·N·S` and
    `C(R)·lr·N ≤ 1/4`: `C(s)·s² ≤ C(R)·lr²·N·S ≤ lr·S/4`. -/
private theorem curvature_exact_le (δ : ℝ → ℝ) {num lr R s N S : ℝ} (hnum : 0 ≤ num)
    (hlr : 0 ≤ lr) (hS : 0 ≤ S) (hδ : δ s ≤ δ R) (hden : 2 * δ R < 1)
    (hs : s ^ 2 ≤ lr ^ 2 * (N * S)) (hC : num / (1 - 2 * δ R) * lr * N ≤ 1 / 4) :
    num / (1 - 2 * δ s) * s ^ 2 ≤ lr * S / 4 := by
  have hdenR : 0 < 1 - 2 * δ R := by linarith
  have hCmono : num / (1 - 2 * δ s) ≤ num / (1 - 2 * δ R) :=
    div_le_div_of_nonneg_left hnum hdenR (by linarith)
  calc num / (1 - 2 * δ s) * s ^ 2
      ≤ num / (1 - 2 * δ R) * (lr ^ 2 * (N * S)) :=
        mul_le_mul hCmono hs (sq_nonneg _) (div_nonneg hnum hdenR.le)
    _ = (num / (1 - 2 * δ R) * lr * N) * (lr * S) := by ring
    _ ≤ 1 / 4 * (lr * S) := mul_le_mul_of_nonneg_right hC (mul_nonneg hlr hS)
    _ = lr * S / 4 := by ring

/-- The explicit `ℓ1` bound on the conv2-kernel loss gradient, `(c·c·kH·kW)·((2h)·(2w)·a)·
    (d₃·w₃·d₄·w₄·nC·w₅)`: `Conv2Slot.gradAt_abs_le` per kernel entry, times the number of
    entries. Named so `cnn_conv2_exact_sgd_descends` can state its radius once. -/
noncomputable def cnnConv2GradBound (c h w d₃ d₄ nC kH kW : ℕ) (a w₃ w₄ w₅ : ℝ) : ℝ :=
  ((c * c * kH * kW : ℕ) : ℝ) * (((2*h * (2*w) : ℕ) : ℝ) * a *
    ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))))

/-- **One exact-gradient SGD step on the conv2 kernel decreases the loss, every hypothesis at an
    explicit radius.** `cnn_conv2_sgd_descends` at the exact gradient (`η = 0`), with the step
    radius `lr·‖∇L‖₁` replaced by its upper bound `lr·cnnConv2GradBound` and the second dominance
    condition by `C·lr·(c·c·kH·kW) ≤ 1/4` (Cauchy–Schwarz, `(Σ|g|)² ≤ (c·c·kH·kW)·Σg²`). No
    hypothesis mentions the gradient: each is a statement about the weights, the input and `lr`,
    which a concrete instance checks by exact arithmetic (`Trained.CnnDescent`). -/
theorem cnn_conv2_exact_sgd_descends {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr a w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hT : ∀ p q, T p q → ConvPatchEq kH kW x₁ p q)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr)
    (hm2 : ∀ k, a * (lr * cnnConv2GradBound c h w d₃ d₄ nC kH kW a w₃ w₄ w₅) <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQUpTo (a * (lr * cnnConv2GradBound c h w d₃ d₄ nC kH kW a w₃ w₄ w₅)) T
      (conv2d W₂ b₂ x₁))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (lr * cnnConv2GradBound c h w d₃ d₄ nC kH kW a w₃ w₄ w₅))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (lr * cnnConv2GradBound c h w d₃ d₄ nC kH kW a w₃ w₄ w₅))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
      (a * (lr * cnnConv2GradBound c h w d₃ d₄ nC kH kW a w₃ w₄ w₅)))))))) < 1)
    (hC : 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * (lr * cnnConv2GradBound c h w d₃ d₄ nC kH kW a w₃ w₄ w₅))))))))) *
        lr * ((c * c * kH * kW : ℕ) : ℝ) ≤ 1 / 4) :
    (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂ -
        lr • gradAt (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂)) ≤
      (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) -
        lr * (∑ idx, gradAt
          (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
          (Kernel4.flatten W₂) idx ^ 2) / 2 := by
  set L := cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label with hL
  set G := cnnConv2GradBound c h w d₃ d₄ nC kH kW a w₃ w₄ w₅ with hG
  set K : ℝ := ((2*h * (2*w) : ℕ) : ℝ) with hK
  have hG0 : 0 ≤ G := by rw [hG, cnnConv2GradBound]; positivity
  have hR0 : 0 ≤ a * (lr * G) := mul_nonneg ha (mul_nonneg hlr hG0)
  -- the gradient's `ℓ1` mass is at most `G`
  set Z : Vec (c * c * kH * kW) → Vec (c * (2*h) * (2*w)) :=
    fun v' => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁) with hZdef
  have hZW : Z (Kernel4.flatten W₂) = Tensor3.flatten (conv2d W₂ b₂ x₁) := by
    rw [hZdef]; dsimp only; rw [Kernel4.unflatten_flatten]
  have hentry : ∀ idx, |gradAt L (Kernel4.flatten W₂) idx| ≤
      K * a * ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) := fun idx => by
    refine Conv2Slot.gradAt_abs_le Z W₃ b₃ W₄ b₄ W₅ b₅ label hR0 hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
      (fun ci hi wi => pdiv Z 0 idx (t3Idx ci hi wi)) ?_ idx _
      (conv2d_weight_differentiable b₂ x₁ _) (fun ci hi wi => ?_) T
      (fun p q hpq v' ci => by
        simp only [hZdef, flatten_t3Idx]
        exact conv2d_eq_of_convPatchEq (hT p q hpq) _ _ ci)
      (fun k => by rw [hZW]; exact abs_pos.mp (hR0.trans_lt (hm2 k)))
      (by rw [hZW, Tensor3.unflatten_flatten]; exact hmq)
      (fun l => by
        rw [hZW]
        have h0 : (0:ℝ) ≤ w₃ * (K * (a * (lr * G))) := by positivity
        exact abs_pos.mp (h0.trans_lt (hm3 l)))
      (fun q => by
        rw [hZW]
        have h0 : (0:ℝ) ≤ w₄ * ((d₃ : ℝ) * (w₃ * (K * (a * (lr * G))))) := by positivity
        exact abs_pos.mp (h0.trans_lt (hm4 q)))
    · obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
      simp only [hZdef, conv2d_weight_pdiv]
      exact convPad_row_l1 x₁ ha hx o cc kh kw
    · obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
      simp only [hZdef]
      rw [conv2d_weight_pdiv, conv2d_weight_pdiv]
  have hsum : (∑ idx, |gradAt L (Kernel4.flatten W₂) idx|) ≤ G :=
    (gradAt_l1_le_card L _ hentry).trans_eq (by rw [hG, cnnConv2GradBound])
  obtain ⟨-, hSR, hSR2⟩ := stepRadius_exact_le L (Kernel4.flatten W₂) hlr hsum
  have haSR := mul_le_mul_of_nonneg_left hSR ha
  refine cnn_conv2_sgd_descends W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label _ T ha hx hT hw₃ hW₃ hw₄ hW₄
    hw₅ hW₅ hlr le_rfl (fun idx => by simp [hL]) (fun k => lt_of_le_of_lt haSR (hm2 k))
    (WindowMarginUpTo.mono winRowInv winColInv haSR hmq)
    (fun l => lt_of_le_of_lt (by gcongr) (hm3 l)) (fun q => lt_of_le_of_lt (by gcongr) (hm4 q))
    (lt_of_le_of_lt (by gcongr) hsmall)
    (by simp only [mul_zero, zero_mul]; try positivity) ?_
  -- the curvature condition: `C(SR)·SR² ≤ C(lr·G)·lr²·(c·c·kH·kW)·Σg² ≤ lr·Σg²/4`
  exact curvature_exact_le (fun r => w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (K * (a * r)))))))
    (by positivity) hlr (Finset.sum_nonneg fun _ _ => sq_nonneg _) (by gcongr)
    hsmall hSR2 hC

/-- Swap the two index triples of a six-fold sum. -/
private theorem sum_swap_triple_triple {α β γ δ ε ζ : Type*} [Fintype α] [Fintype β] [Fintype γ]
    [Fintype δ] [Fintype ε] [Fintype ζ] (f : α → β → γ → δ → ε → ζ → ℝ) :
    ∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ, ∑ e : ε, ∑ g : ζ, f a b c d e g =
      ∑ d : δ, ∑ e : ε, ∑ g : ζ, ∑ a : α, ∑ b : β, ∑ c : γ, f a b c d e g :=
  calc _ = ∑ p : α × β × γ, ∑ q : δ × ε × ζ, f p.1 p.2.1 p.2.2 q.1 q.2.1 q.2.2 := by
          simp only [Fintype.sum_prod_type]
    _ = ∑ q : δ × ε × ζ, ∑ p : α × β × γ, f p.1 p.2.1 p.2.2 q.1 q.2.1 q.2.2 := Finset.sum_comm
    _ = _ := by simp only [Fintype.sum_prod_type]

-- ════════════════════════════════════════════════════════════════
-- § The conv1 drift chain: through BOTH convs to the logits
-- ════════════════════════════════════════════════════════════════

-- The chain below is stated for any parameter map `Z` into conv1's pre-activation that
-- moves it by at most `ρ·‖e‖₁` per entry and `(2h)·(2w)·ρ·‖e‖₁` in `ℓ1`: the conv1 kernel
-- (`ρ = a`) and the conv1 bias (`ρ = 1`) are the two instances. From conv2's pre-activation
-- on it is the `Conv2Slot` chain at radius `c·kH·kW·w₂·ρ`.

namespace Conv1Slot

/-- Per-entry conv2-preactivation drift: the conv1 pre-activation `Z` moves by `ρ·‖e‖₁`
    per entry and crosses conv2 as a function of its INPUT, picking up the locality
    factor `c·kH·kW·w₂`. -/
private theorem z2_entry_drift {P c h w kH kW : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w))) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {ρ w₂ : ℝ} (hρ : 0 ≤ ρ) (hZ : ∀ v e k, |Z (v + e) k - Z v k| ≤ ρ * ∑ idx, |e idx|)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (v e : Vec P) (k : Fin (c * (2*h) * (2*w))) :
    |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z (v + e))))) k -
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v)))) k| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (ρ * ∑ idx, |e idx|)) := by
  obtain ⟨o, ho, wo, rfl⟩ := t3Idx_surj k
  rw [flatten_t3Idx, flatten_t3Idx]
  exact conv2d_input_entry_drift W₂ b₂ _ _ hw₂ hW₂
    (mul_nonneg hρ (Finset.sum_nonneg fun _ _ => abs_nonneg _))
    (fun cc i j => Conv2Slot.postrelu_close Z hZ v e cc i j) o ho wo

/-- `ℓ1` conv2-preactivation drift: conv1 (`ℓ1`, `hZ1`) → relu → conv2-as-input (`ℓ1`,
    locality multiplicity `c·kH·kW`). -/
private theorem z2_l1_drift {P c h w kH kW : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w))) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {ρ w₂ : ℝ} (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (v e : Vec P) :
    ∑ k, |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Z (v + e))))) k -
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Z v)))) k| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (ρ * ∑ idx, |e idx|))) := by
  rw [sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
    |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z (v + e))))) k -
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v)))) k|)]
  simp only [flatten_t3Idx]
  refine le_trans (conv2d_input_l1_drift W₂ b₂ _ _ hw₂ hW₂)
    (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left ?_ hw₂) (Nat.cast_nonneg _))
  simp only [unflatten_t3Idx]
  rw [← sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
    |relu (c * (2*h) * (2*w)) (Z (v + e)) k - relu (c * (2*h) * (2*w)) (Z v) k|)]
  exact le_trans (Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k) (hZ1 v e)

/-- **conv2's Jacobian row through relu₁ has row mass `(2h)·(2w)·c·kH·kW·w₂·ρ`.** The row is
    `J`, relu₁'s mask at `v` and conv2's taps (`convTap`, at most `c·kH·kW·w₂` per input cell). -/
private theorem z2_row_l1 {P c h w kH kW : Nat} (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (W₂ : Kernel4 c c kH kW) {ρ w₂ : ℝ} (hw₂ : 0 ≤ w₂)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (J : Fin P → Fin c → Fin (2*h) → Fin (2*w) → ℝ)
    (hJ : ∀ idx, ∑ ci, ∑ hi, ∑ wi, |J idx ci hi wi| ≤ ((2*h * (2*w) : ℕ) : ℝ) * ρ)
    (v : Vec P) (idx : Fin P) :
    ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
      |∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w), J idx ci hi wi *
        ((if Z v (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) * convTap W₂ ci hi wi co ho wo)| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (((c * kH * kW : ℕ) : ℝ) * (w₂ * ρ)) :=
  calc _ ≤ ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |J idx ci hi wi| * |convTap W₂ ci hi wi co ho wo| := by
        refine Finset.sum_le_sum fun co _ => Finset.sum_le_sum fun ho _ =>
          Finset.sum_le_sum fun wo _ => (Finset.abs_sum_le_sum_abs _ _).trans
            (Finset.sum_le_sum fun ci _ => (Finset.abs_sum_le_sum_abs _ _).trans
              (Finset.sum_le_sum fun hi _ => (Finset.abs_sum_le_sum_abs _ _).trans
                (Finset.sum_le_sum fun wi _ => ?_)))
        rw [abs_mul, abs_mul]
        exact mul_le_mul_of_nonneg_left (mul_le_of_le_one_left (abs_nonneg _)
          (by split_ifs <;> simp)) (abs_nonneg _)
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w), |J idx ci hi wi| *
          ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
            |convTap W₂ ci hi wi co ho wo| := by
        rw [sum_swap_triple_triple]
        simp only [Finset.mul_sum]
    _ ≤ ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |J idx ci hi wi| * (((c * kH * kW : ℕ) : ℝ) * w₂) :=
        Finset.sum_le_sum fun ci _ => Finset.sum_le_sum fun hi _ =>
          Finset.sum_le_sum fun wi _ => mul_le_mul_of_nonneg_left
            (convTap_out_l1 W₂ hW₂ ci hi wi) (abs_nonneg _)
    _ ≤ ((2*h * (2*w) : ℕ) : ℝ) * ρ * (((c * kH * kW : ℕ) : ℝ) * w₂) := by
        simp only [← Finset.sum_mul]
        exact mul_le_mul_of_nonneg_right (hJ idx) (mul_nonneg (Nat.cast_nonneg _) hw₂)
    _ = _ := by ring

/-- **At a frozen-mask point, conv2's pre-activation through relu₁ has that row as its
    Jacobian**: the chain rule through `Z`, relu₁ (off its kink, signs as at `v`) and conv2 as a
    function of its input (`conv2d_flat_input_pdiv`). -/
private theorem z2_pdiv {P c h w kH kW : Nat} (Z : Vec P → Vec (c * (2*h) * (2*w)))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (J : Fin P → Fin c → Fin (2*h) → Fin (2*w) → ℝ)
    (hpd : ∀ v' : Vec P, DifferentiableAt ℝ Z v' ∧
      ∀ idx ci hi wi, pdiv Z v' idx (t3Idx ci hi wi) = J idx ci hi wi)
    (v v' : Vec P) (hq : ∀ k, Z v' k ≠ 0 ∧ (0 < Z v' k ↔ 0 < Z v k)) :
    DifferentiableAt ℝ (fun v'' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Z v''))))) v' ∧
    ∀ idx co ho wo, pdiv (fun v'' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v''))))) v' idx (t3Idx co ho wo) =
      ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w), J idx ci hi wi *
        ((if Z v (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) * convTap W₂ ci hi wi co ho wo) := by
  have hrelu := relu_differentiableAt_of_smooth (c * (2*h) * (2*w)) (Z v') fun k => (hq k).1
  have hrz : DifferentiableAt ℝ (relu (c * (2*h) * (2*w)) ∘ Z) v' := hrelu.comp v' (hpd v').1
  refine ⟨(flatConv_differentiable W₂ b₂ _).comp v' hrz, fun idx co ho wo => ?_⟩
  rw [show (fun v'' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v''))))) =
      (fun u : Vec (c * (2*h) * (2*w)) => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten u)))
        ∘ (relu (c * (2*h) * (2*w)) ∘ Z) from rfl,
    pdiv_comp (relu (c * (2*h) * (2*w)) ∘ Z)
      (fun u : Vec (c * (2*h) * (2*w)) => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten u)))
      v' hrz ((flatConv_differentiable W₂ b₂) _), sum_t3]
  refine Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl fun hi _ =>
    Finset.sum_congr rfl fun wi _ => ?_
  rw [conv2d_flat_input_pdiv, pdiv_comp Z (relu (c * (2*h) * (2*w))) v' (hpd v').1 hrelu]
  simp_rw [pdiv_relu (c * (2*h) * (2*w)) (Z v') (fun k => (hq k).1)]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true, (hpd v').2,
    gt_iff_lt, (hq _).2]
  split_ifs <;> ring

/-- **One inexact SGD step through a conv1 slot decreases the loss.** For a parameter map `Z`
    into conv1's pre-activation with per-entry drift `ρ·‖e‖₁` and `ℓ1` drift
    `(2h)·(2w)·ρ·‖e‖₁`, differentiable everywhere with a point-free Jacobian `J` (row mass
    `≤ (2h)·(2w)·ρ`): the relu₁ margin freezes relu₁'s mask along the step, so conv2's
    pre-activation `conv2 ∘ relu ∘ Z` moves by `c·kH·kW·w₂·ρ·‖e‖₁` per entry
    (`z2_entry_drift`), and at every point of the step its Jacobian is one fixed row: `J`, the
    base mask and conv2's taps (`convTap`, locality `c·kH·kW·w₂`). The rest is
    `Conv2Slot.sgd_descends` at that map, radius `c·kH·kW·w₂·ρ`, with `Q` = relu₁'s signs
    frozen. The conv1-kernel rung is the instance `ρ = a`, the conv1-bias rung `ρ = 1`. -/
theorem sgd_descends {P c h w d₃ d₄ nC kH kW : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w))) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (v gh : Vec P) {ρ lr η w₂ w₃ w₄ w₅ : ℝ} (hρ : 0 ≤ ρ)
    (hZ : ∀ v e k, |Z (v + e) k - Z v k| ≤ ρ * ∑ idx, |e idx|)
    (hZ1 : ∀ v e, ∑ k, |Z (v + e) k - Z v k| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |e idx|))
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (J : Fin P → Fin c → Fin (2*h) → Fin (2*w) → ℝ)
    (hJ : ∀ idx, ∑ ci, ∑ hi, ∑ wi, |J idx ci hi wi| ≤ ((2*h * (2*w) : ℕ) : ℝ) * ρ)
    (hpd : ∀ v' : Vec P, DifferentiableAt ℝ Z v' ∧
      ∀ idx ci hi wi, pdiv Z v' idx (t3Idx ci hi wi) = J idx ci hi wi)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (hT : ∀ a b, T a b → ∀ v' ci,
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))
          (t3Idx ci a.1 a.2) =
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))
          (t3Idx ci b.1 b.2))
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - gradAt (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂
      (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v idx| ≤ η)
    (hm1 : ∀ k, ρ * stepRadius (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂
      (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η <
      |Z v k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (ρ * stepRadius (Conv2Slot.loss
      (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η)) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v)))) k|)
    (hmq : MaxPool2MarginQUpTo (((c * kH * kW : ℕ) : ℝ) * (w₂ * (ρ * stepRadius (Conv2Slot.loss
      (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η))) T
      (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v)))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (ρ * stepRadius (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label)
          v lr η)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (ρ * stepRadius (Conv2Slot.loss
          (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Z v))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * (ρ * stepRadius (Conv2Slot.loss
        (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η))))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v idx|) ≤
      lr * (∑ idx, gradAt (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label)
          v idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 *
        (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * ρ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) * (ρ * stepRadius (Conv2Slot.loss
            (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η))))))))))) *
        stepRadius (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η ^ 2 ≤
      lr * (∑ idx, gradAt (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label)
          v idx ^ 2) / 4) :
    Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label (v - lr • gh) ≤
      Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label v -
        lr * (∑ idx, gradAt (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label)
            v idx ^ 2) / 2 := by
  have hD := sgd_step_l1_le _ gh hlr hgh
  -- relu₁'s mask, conv1's rows and conv2's taps: one fixed row at the conv2 pre-activation,
  -- which at a frozen-mask point is conv2's pre-activation's Jacobian
  have hJ₂ := z2_row_l1 Z W₂ hw₂ hW₂ J hJ v
  have hpd₂ := z2_pdiv Z W₂ b₂ J hpd v
  have hmq' : MaxPool2MarginQUpTo (((c * kH * kW : ℕ) : ℝ) * (w₂ * ρ) *
      stepRadius (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v lr η) T
      (Tensor3.unflatten (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v)))))) := by
    have e : ∀ x : ℝ, ((c * kH * kW : ℕ) : ℝ) * (w₂ * ρ) * x =
        ((c * kH * kW : ℕ) : ℝ) * (w₂ * (ρ * x)) := fun x => by ring
    rw [Tensor3.unflatten_flatten, e]; exact hmq
  refine (Conv2Slot.sgd_descends (ρ := ((c * kH * kW : ℕ) : ℝ) * (w₂ * ρ))
    (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label v gh
    (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₂ hρ))
    (fun v e k => (z2_entry_drift Z W₂ b₂ hρ hZ hw₂ hW₂ v e k).trans_eq (by ring))
    (fun v e => (z2_l1_drift Z W₂ b₂ hZ1 hw₂ hW₂ v e).trans_eq (by ring))
    hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ _ hJ₂ (fun v' => ∀ k, Z v' k ≠ 0 ∧ (0 < Z v' k ↔ 0 < Z v k))
    hpd₂ T hT hlr hη hgh
    (fun t ht k => margin_keeps_offkink_of_drift Z hρ hZ v _ hD hm1 t ht.1 ht.2 k)
    (fun k => lt_of_eq_of_lt (by ring) (hm2 k))
    hmq'
    (fun l => lt_of_eq_of_lt (by ring) (hm3 l)) (fun q => lt_of_eq_of_lt (by ring) (hm4 q))
    (lt_of_eq_of_lt (by ring) hsmall) h1 ((le_of_eq (by ring)).trans h2))

/-- **The conv1-slot loss gradient is at most conv2's row mass times the head's operator norm.**
    `Conv2Slot.gradAt_abs_le` at the map `conv2 ∘ relu ∘ Z`: at a point where relu₁ is off its
    kink, that map's Jacobian is the fixed row of `z2_pdiv`, of mass at most
    `(2h)·(2w)·c·kH·kW·w₂·ρ` (`z2_row_l1`). The bound the conv1 exact-gradient corollaries use
    for their step radius. -/
theorem gradAt_abs_le {P c h w d₃ d₄ nC kH kW : Nat}
    (Z : Vec P → Vec (c * (2*h) * (2*w))) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) {δ ρ w₂ w₃ w₄ w₅ : ℝ} (hδ : 0 ≤ δ)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (J : Fin P → Fin c → Fin (2*h) → Fin (2*w) → ℝ)
    (hJ : ∀ idx, ∑ ci, ∑ hi, ∑ wi, |J idx ci hi wi| ≤ ((2*h * (2*w) : ℕ) : ℝ) * ρ)
    (hpd : ∀ v' : Vec P, DifferentiableAt ℝ Z v' ∧
      ∀ idx ci hi wi, pdiv Z v' idx (t3Idx ci hi wi) = J idx ci hi wi)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (hT : ∀ a b, T a b → ∀ v' ci,
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))
          (t3Idx ci a.1 a.2) =
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v'))))
          (t3Idx ci b.1 b.2))
    (idx : Fin P) (v : Vec P) (hz1 : ∀ k, Z v k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Z v)))) k ≠ 0)
    (hmq : MaxPool2MarginQUpTo δ T (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Z v)))))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
      (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z v))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v))))))))) q ≠ 0) :
    |gradAt (Conv2Slot.loss (fun v' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Z v'))))) W₃ b₃ W₄ b₄ W₅ b₅ label) v idx| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (((c * kH * kW : ℕ) : ℝ) * (w₂ * ρ)) *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) := by
  have hp := z2_pdiv Z W₂ b₂ J hpd v v fun k => ⟨hz1 k, Iff.rfl⟩
  exact Conv2Slot.gradAt_abs_le _ W₃ b₃ W₄ b₄ W₅ b₅ label hδ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ _
    (z2_row_l1 Z W₂ hw₂ hW₂ J hJ v idx) idx v hp.1 (hp.2 idx) T hT hz2
    (by rw [Tensor3.unflatten_flatten]; exact hmq) hz3 hz4

end Conv1Slot

-- ════════════════════════════════════════════════════════════════
-- § The conv1 head gradient: through relu₁, conv2-as-input, and the
--   pool to the 3-dense head
-- ════════════════════════════════════════════════════════════════

/-- The whole head above the conv1 output — `CE∘head3∘pool∘relu∘
    (flatConv W₂ b₂)∘relu` — is differentiable at any five-condition
    point. -/
private theorem cnn1_pool_head_differentiableAt {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₁ : Vec (c * (2*h) * (2*w))) (hz1 : ∀ k, z₁ k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₁))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁))))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁)))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₁)))))))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) y))))))))))) label) z₁ := by
  exact (pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hmp hz3 hz4).comp
    (f := fun y : Vec (c * (2*h) * (2*w)) => Tensor3.flatten (conv2d W₂ b₂
      (Tensor3.unflatten (relu (c * (2*h) * (2*w)) y)))) z₁ (by fun_prop (disch := assumption))

/-- **Loss input-gradient at the conv1 output** — the conv1 peer of
    `pool_relu_input_grad`. One more relu mask and one conv-as-input
    crossing: the chain picks up `relu'(z₁)` and contracts the point-free
    tap Jacobian of conv2 with the pool-collapsed conv2-rung gradient. -/
theorem cnn1_pool_head_input_grad {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₁ : Vec (c * (2*h) * (2*w))) (hz1 : ∀ k, z₁ k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₁))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁))))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁)))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₁)))))))) q ≠ 0)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) y))))))))))) label)
        z₁ (t3Idx ci hi wi) 0
      = (if z₁ (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
            convTap W₂ ci hi wi co ho wo *
              ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) z₁))) (t3Idx co ho wo) > 0
                  then (1:ℝ) else 0) *
                (if MaxPool2IsArgmax (Tensor3.unflatten
                      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                        (conv2d W₂ b₂ (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) z₁)))))) co ho wo
                  then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                    ((if dense W₃ b₃ (maxPoolFlat c h w
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) z₁)))))) l > 0
                        then (1:ℝ) else 0) *
                      ∑ q, W₄ l q *
                        ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                              (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                (Tensor3.flatten (conv2d W₂ b₂
                                  (Tensor3.unflatten (relu
                                    (c * (2*h) * (2*w)) z₁)))))))) q > 0
                            then (1:ℝ) else 0) *
                          ∑ k, W₅ q k *
                            (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                                (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d W₂ b₂
                                      (Tensor3.unflatten (relu
                                        (c * (2*h) * (2*w))
                                        z₁))))))))))) k -
                              oneHot nC label k)))
                  else 0)) := by
  have hG2 := pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label
    (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₁)))) hz2 hmp hz3 hz4
  have hflat : DifferentiableAt ℝ
      (fun v : Vec (c * (2*h) * (2*w)) =>
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))
      (relu (c * (2*h) * (2*w)) z₁) :=
    (flatConv_differentiable (h := 2*h) (w := 2*w) W₂ b₂) _
  have hGF : DifferentiableAt ℝ
      ((fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label) ∘
        (fun v : Vec (c * (2*h) * (2*w)) =>
          Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v))))
      (relu (c * (2*h) * (2*w)) z₁) :=
    hG2.comp (relu (c * (2*h) * (2*w)) z₁) hflat
  -- hop 1: peel relu₁; the chain picks up the mask
  rw [show (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
          crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) y))))))))))) label)
        = ((fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
            crossEntropy nC
            (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
              (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y)))))))
            label) ∘
          (fun v : Vec (c * (2*h) * (2*w)) =>
            Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))) ∘
          (relu (c * (2*h) * (2*w)))
        from rfl,
      pdiv_comp _ _ _
        (relu_differentiableAt_of_smooth (c * (2*h) * (2*w)) z₁ hz1) hGF]
  simp_rw [pdiv_relu (c * (2*h) * (2*w)) z₁ hz1 (t3Idx ci hi wi)]
  rw [Fintype.sum_eq_single (t3Idx ci hi wi)
    (fun j hne => by rw [ite_eq_right (fun heq => hne heq.symm), zero_mul]), ite_eq_left rfl]
  congr 1
  -- hop 2: through conv2 as a function of its input
  have hop2 : pdiv ((fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
        crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label) ∘
        (fun v : Vec (c * (2*h) * (2*w)) =>
          Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v))))
      (relu (c * (2*h) * (2*w)) z₁) (t3Idx ci hi wi) 0
      = ∑ k : Fin (c * (2*h) * (2*w)),
          pdiv (fun v : Vec (c * (2*h) * (2*w)) =>
              Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))
            (relu (c * (2*h) * (2*w)) z₁) (t3Idx ci hi wi) k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC
              (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y)))))))
              label)
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) z₁)))) k 0 :=
    pdiv_comp _ _ _ hflat hG2 _ _
  rw [hop2, sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
    pdiv (fun v : Vec (c * (2*h) * (2*w)) =>
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))
      (relu (c * (2*h) * (2*w)) z₁) (t3Idx ci hi wi) k *
    pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
        crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁)))) k 0)]
  refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl
    fun ho _ => Finset.sum_congr rfl fun wo _ => ?_
  rw [conv2d_flat_input_pdiv W₂ b₂ _ ci hi wi co ho wo,
    pool_relu_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hmp hz3 hz4
      co ho wo]

-- ════════════════════════════════════════════════════════════════
-- § The conv1 loss-of-kernel map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- **Closed form of the conv1 loss gradient** at any five-margin point —
    the same chain rule, contracted with the conv1 head gradient
    (`cnn1_pool_head_input_grad`): the conv1 weight Jacobian
    (`convPad` reads of the IMAGE) times relu₁'s mask times the
    point-free conv2 tap Jacobian times the pool-collapsed head. Two
    spatial triple-sums: weight sharing at conv1, locality at conv2. -/
theorem cnn_conv1_loss_gradAt {ic c h w d₃ d₄ nC kH kW : Nat}
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC)
    (u : Vec (c * ic * kH * kW))
    (hz1 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
      k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q ≠ 0)
    (o : Fin c) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label)
        u (k4Idx o cc kh kw)
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then convPad kH kW x₀ cc kh kw hi wi else 0) *
            ((if Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d (Kernel4.unflatten u) b₁ x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d (Kernel4.unflatten u)
                                  b₁ x₀)))))))) co ho wo
                      then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                        ((if dense W₃ b₃ (maxPoolFlat c h w
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₂ b₂ (Tensor3.unflatten
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d
                                      (Kernel4.unflatten u)
                                      b₁ x₀)))))))) l > 0
                            then (1:ℝ) else 0) *
                          ∑ q, W₄ l q *
                            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                  (maxPoolFlat c h w (relu
                                    (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten
                                      (relu (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d
                                          (Kernel4.unflatten u)
                                          b₁ x₀)))))))))) q > 0
                                then (1:ℝ) else 0) *
                              ∑ k, W₅ q k *
                                (softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d W₂ b₂
                                          (Tensor3.unflatten (relu
                                            (c * (2*h) * (2*w))
                                            (Tensor3.flatten (conv2d
                                              (Kernel4.unflatten u)
                                              b₁ x₀))))))))))))) k -
                                  oneHot nC label k)))
                      else 0))) := by
  refine (gradAt_comp_t3 (fun u' => Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀))
    (fun y => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
      (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) y))))))))))) label) u
    (conv2d_weight_differentiable b₁ x₀ u)
    (cnn1_pool_head_differentiableAt W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _
      hz1 hz2 hmp hz3 hz4) _).trans
    (Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl fun hi _ =>
      Finset.sum_congr rfl fun wi _ => ?_)
  rw [conv2d_weight_pdiv b₁ x₀ _ o cc kh kw ci hi wi,
    cnn1_pool_head_input_grad W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _ hz1 hz2 hmp hz3 hz4 ci hi wi]

/-- **The certified conv-1 loss gradient, head restated in `dense`/`reluMask`
    form** — the conv-1 peer of `cnn_conv2_loss_gradAt_reluMask`. One conv-backward deeper than conv-2: the conv-1-output
    cotangent is `𝟙[z₁>0] · ∑_{co,ho,wo} convTap·(conv-2-output cotangent)`,
    with the 3-dense head collapsed by `head3_cot_reluMask` exactly as in
    conv-2. The conv-1 ReLU mask, the conv-2 backward tap (`convTap`, the
    point-free conv-2 input Jacobian), the conv-2 ReLU mask and the pool
    selector all stay explicit (their float closeness is `mask_scalar_close` /
    `dot_perturbed_close` / `poolBack_close`). Packaged as the spatial dot
    `∑ₛ convPadWin x₀·cotWin` (`convWeightGrad_eq_dot`) the float conv-1 weight
    dot rounds. -/
theorem cnn_conv1_loss_gradAt_reluMask {ic c h w d₃ d₄ nC kH kW : Nat}
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC)
    (u : Vec (c * ic * kH * kW))
    (hz1 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q ≠ 0)
    (o : Fin c) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label)
        u (k4Idx o cc kh kw)
      = ∑ s, convPadWin kH kW x₀ cc kh kw s *
          cotWin (fun ci hi wi =>
            (if Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d (Kernel4.unflatten u) b₁ x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d (Kernel4.unflatten u) b₁ x₀))))))))
                          co ho wo
                      then dense (fun j i' => W₃ i' j) (fun _ => 0)
                        (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d W₂ b₂ (Tensor3.unflatten
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))
                          (dense (fun j i' => W₄ i' j) (fun _ => 0)
                            (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                                (dense W₃ b₃ (maxPoolFlat c h w (relu
                                  (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten (relu
                                      (c * (2*h) * (2*w)) (Tensor3.flatten
                                        (conv2d (Kernel4.unflatten u)
                                          b₁ x₀)))))))))))
                              (dense (fun j i' => W₅ i' j) (fun _ => 0)
                                (fun k => softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w)) (Tensor3.flatten
                                          (conv2d W₂ b₂ (Tensor3.unflatten
                                            (relu (c * (2*h) * (2*w))
                                              (Tensor3.flatten (conv2d
                                                (Kernel4.unflatten u)
                                                b₁ x₀))))))))))))) k -
                                  oneHot nC label k)))))
                        (t3Idx co (winRow ho) (winCol wo))
                      else 0))) o s := by
  rw [cnn_conv1_loss_gradAt b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label u
      hz1 hz2 hmp hz3 hz4 o cc kh kw]
  simp_rw [head3_cot_reluMask]
  rw [convWeightGrad_eq_dot x₀ _ o cc kh kw]
  simp only [ite_mul, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq',
    Finset.mem_univ, ite_true]

-- ════════════════════════════════════════════════════════════════
-- § The conv1 capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- Two-layer twins are symmetric. -/
theorem ConvPatchEq2.symm {ic h w kH kW : Nat} {x : Tensor3 ic h w} {p q : Fin h × Fin w}
    (hpq : ConvPatchEq2 kH kW x p q) : ConvPatchEq2 kH kW x q p :=
  fun kh kw => ⟨(hpq kh kw).1.symm, fun hq hp => ((hpq kh kw).2 hp hq).symm⟩

/-- Two-layer twins are transitive: the middle cell's outer reads are in bounds exactly when the
    ends' are. With `ConvPatchEq2.symm`, the equivalence `windowMarginUpTo_of_cert` asks of the
    twin relation. -/
theorem ConvPatchEq2.trans {ic h w kH kW : Nat} {x : Tensor3 ic h w} {p q u : Fin h × Fin w}
    (hpq : ConvPatchEq2 kH kW x p q) (hqu : ConvPatchEq2 kH kW x q u) :
    ConvPatchEq2 kH kW x p u :=
  fun kh kw => ⟨(hpq kh kw).1.trans (hqu kh kw).1, fun hp hu =>
    ((hpq kh kw).2 hp ((hpq kh kw).1.mp hp)).trans ((hqu kh kw).2 ((hpq kh kw).1.mp hp) hu)⟩

/-- The loss as a function of the flattened first-conv kernel. -/
noncomputable def cnnConv1KernelLoss {ic c h w d₃ d₄ nC kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃)
    (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) :
    Vec (c * ic * kH * kW) → ℝ :=
  fun u' => crossEntropy nC
    (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c *
      (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
    label

/-- **One inexact SGD step on the CNN's FIRST conv kernel decreases one
    example's cross-entropy loss** (`W₁` moving, every other parameter fixed).
    The deepest rung: the step
    crosses relu₁, conv2 (as a function of its input — the point-free
    tap Jacobian with locality factor `c·kH·kW·w₂`), relu₂, the pool,
    and the 3-dense head. `Conv1Slot.sgd_descends` at the conv1 kernel map (`ρ = a`). Under the
    FIVE margins at the step radius `D = lr·(‖∇L‖₁ + |kernel|·η)`, the pool's taken up to
    two-layer twins (`ConvPatchEq2`: cells whose two-conv receptive fields in `x₀` are
    identical, equal for every `W₁`), every mask freezes along the step, the pool agrees with a
    fixed gather near every point of it, and the loss drops by ≥ `lr·‖∇L‖₂²/2`. -/
theorem cnn_conv1_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec (c * ic * kH * kW)) (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr η a w₂ w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₀ cc i j| ≤ a) (hT : ∀ p q, T p q → ConvPatchEq2 kH kW x₀ p q)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - (gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁)) idx| ≤ η)
    (hm1 : ∀ k, a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₁) lr η) < |(Tensor3.flatten (conv2d W₁ b₁ x₀)) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (stepRadius
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₁) lr η))) < |(Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu
      (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))) k|)
    (hmq : MaxPool2MarginQUpTo (((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (stepRadius
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr η)))) T
      (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₁) lr η))))) < |(dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h *
      (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₁) lr η)))))))
      < |(dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₁) lr η)))))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁) idx|) ≤
      lr * (∑ idx, (gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 *
      (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 / (1 - 2 * (w₅ *
      ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h *
      (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (Kernel4.flatten W₁) lr η)))))))))))) * (stepRadius
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr η) ^ 2 ≤
      lr * (∑ idx, (gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4) :
    (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁ - lr • gh) ≤
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) -
        lr * (∑ idx, (gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁)) idx ^ 2) / 2 := by
  -- the conv1 kernel map, its drift and its point-free Jacobian
  set Z : Vec (c * ic * kH * kW) → Vec (c * (2*h) * (2*w)) :=
    fun u' => Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀) with hZdef
  have hZW : Z (Kernel4.flatten W₁) = Tensor3.flatten (conv2d W₁ b₁ x₀) := by
    rw [hZdef]; dsimp only; rw [Kernel4.unflatten_flatten]
  have hpd : ∀ u' : Vec (c * ic * kH * kW), DifferentiableAt ℝ Z u' ∧
      ∀ idx ci hi wi, pdiv Z u' idx (t3Idx ci hi wi) = pdiv Z 0 idx (t3Idx ci hi wi) :=
    fun u' => ⟨conv2d_weight_differentiable b₁ x₀ u', fun idx ci hi wi => by
      obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
      rw [conv2d_weight_pdiv, conv2d_weight_pdiv]⟩
  have hJ : ∀ idx, ∑ ci, ∑ hi, ∑ wi, |pdiv Z 0 idx (t3Idx ci hi wi)| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * a := fun idx => by
    obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
    simp only [hZdef, conv2d_weight_pdiv]
    exact convPad_row_l1 x₀ ha hx o cc kh kw
  have hT' : ∀ p q, T p q → ∀ u' ci,
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z u'))))
          (t3Idx ci p.1 p.2) =
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Z u'))))
          (t3Idx ci q.1 q.2) := fun p q hpq u' ci => by
    simp only [hZdef, flatten_t3Idx]
    exact conv2d_eq_of_convPatchEq (convPatchEq_relu_conv (hT p q hpq) _ _) _ _ ci
  exact Conv1Slot.sgd_descends Z W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label (Kernel4.flatten W₁) gh ha
    (conv2d_flat_kernel_drift_total b₁ x₀ ha hx) (conv2d_flat_kernel_drift_sum b₁ x₀ ha hx)
    hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (fun idx ci hi wi => pdiv Z 0 idx (t3Idx ci hi wi)) hJ hpd
    T hT' hlr hη hgh (by rw [hZW]; exact hm1) (by rw [hZW]; exact hm2) (by rw [hZW]; exact hmq)
    (by rw [hZW]; exact hm3) (by rw [hZW]; exact hm4) hsmall h1 h2

/-- The explicit `ℓ1` bound on the conv1-kernel loss gradient, `(c·ic·kH·kW)·((2h)·(2w)·
    (c·kH·kW·w₂·a))·(d₃·w₃·d₄·w₄·nC·w₅)`: `Conv1Slot.gradAt_abs_le` at `ρ = a` per kernel entry,
    times the number of entries. -/
noncomputable def cnnConv1GradBound (ic c h w d₃ d₄ nC kH kW : ℕ) (a w₂ w₃ w₄ w₅ : ℝ) : ℝ :=
  ((c * ic * kH * kW : ℕ) : ℝ) * (((2*h * (2*w) : ℕ) : ℝ) * (((c * kH * kW : ℕ) : ℝ) * (w₂ * a)) *
    ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))))

/-- **One exact-gradient SGD step on the conv1 kernel decreases the loss, every hypothesis at an
    explicit radius.** `cnn_conv1_sgd_descends` at `η = 0`, the step radius bounded by
    `lr·cnnConv1GradBound` and the second dominance condition by `C·lr·(c·ic·kH·kW) ≤ 1/4`, as in
    `cnn_conv2_exact_sgd_descends`. The twins are two-layer (`ConvPatchEq2`). -/
theorem cnn_conv1_exact_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr a w₂ w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₀ cc i j| ≤ a) (hT : ∀ p q, T p q → ConvPatchEq2 kH kW x₀ p q)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr)
    (hm1 : ∀ k, a * (lr * cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅) <
      |Tensor3.flatten (conv2d W₁ b₁ x₀) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a *
        (lr * cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅))) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k|)
    (hmq : MaxPool2MarginQUpTo (((c * kH * kW : ℕ) : ℝ) * (w₂ * (a *
        (lr * cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅)))) T
      (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a *
        (lr * cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅))))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a *
          (lr * cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅))))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a *
        (lr * cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅)))))))))) < 1)
    (hC : 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 *
        (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a *
            (lr * cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅))))))))))) *
        lr * ((c * ic * kH * kW : ℕ) : ℝ) ≤ 1 / 4) :
    (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁ -
        lr • gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
          (Kernel4.flatten W₁)) ≤
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) -
        lr * (∑ idx, gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
          (Kernel4.flatten W₁) idx ^ 2) / 2 := by
  set L := cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label with hL
  set G := cnnConv1GradBound ic c h w d₃ d₄ nC kH kW a w₂ w₃ w₄ w₅ with hG
  set K : ℝ := ((2*h * (2*w) : ℕ) : ℝ) with hK
  set M : ℝ := ((c * kH * kW : ℕ) : ℝ) with hM
  have hG0 : 0 ≤ G := by rw [hG, cnnConv1GradBound]; positivity
  have hR0 : 0 ≤ a * (lr * G) := mul_nonneg ha (mul_nonneg hlr hG0)
  -- the conv1 kernel map, its point-free Jacobian and the gradient's `ℓ1` mass
  set Z : Vec (c * ic * kH * kW) → Vec (c * (2*h) * (2*w)) :=
    fun u' => Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀) with hZdef
  have hZW : Z (Kernel4.flatten W₁) = Tensor3.flatten (conv2d W₁ b₁ x₀) := by
    rw [hZdef]; dsimp only; rw [Kernel4.unflatten_flatten]
  have hpd : ∀ u' : Vec (c * ic * kH * kW), DifferentiableAt ℝ Z u' ∧
      ∀ idx ci hi wi, pdiv Z u' idx (t3Idx ci hi wi) = pdiv Z 0 idx (t3Idx ci hi wi) :=
    fun u' => ⟨conv2d_weight_differentiable b₁ x₀ u', fun idx ci hi wi => by
      obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
      rw [conv2d_weight_pdiv, conv2d_weight_pdiv]⟩
  have hJ : ∀ idx, ∑ ci, ∑ hi, ∑ wi, |pdiv Z 0 idx (t3Idx ci hi wi)| ≤ K * a := fun idx => by
    obtain ⟨o, cc, kh, kw, rfl⟩ := k4Idx_surj idx
    simp only [hZdef, conv2d_weight_pdiv]
    exact convPad_row_l1 x₀ ha hx o cc kh kw
  have hentry : ∀ idx, |gradAt L (Kernel4.flatten W₁) idx| ≤
      K * (M * (w₂ * a)) * ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) :=
    fun idx => Conv1Slot.gradAt_abs_le Z W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
      (by positivity : (0:ℝ) ≤ M * (w₂ * (a * (lr * G)))) hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
      (fun idx ci hi wi => pdiv Z 0 idx (t3Idx ci hi wi)) hJ hpd T
      (fun p q hpq u' ci => by
        simp only [hZdef, flatten_t3Idx]
        exact conv2d_eq_of_convPatchEq (convPatchEq_relu_conv (hT p q hpq) _ _) _ _ ci)
      idx _ (fun k => by rw [hZW]; exact abs_pos.mp (hR0.trans_lt (hm1 k)))
      (fun k => by
        rw [hZW]
        exact abs_pos.mp ((by positivity : (0:ℝ) ≤ M * (w₂ * (a * (lr * G)))).trans_lt (hm2 k)))
      (by rw [hZW]; exact hmq)
      (fun l => by
        rw [hZW]
        exact abs_pos.mp ((by positivity :
          (0:ℝ) ≤ w₃ * (M * (w₂ * (K * (a * (lr * G)))))).trans_lt (hm3 l)))
      (fun q => by
        rw [hZW]
        exact abs_pos.mp ((by positivity :
          (0:ℝ) ≤ w₄ * ((d₃ : ℝ) * (w₃ * (M * (w₂ * (K * (a * (lr * G)))))))).trans_lt (hm4 q)))
  have hsum : (∑ idx, |gradAt L (Kernel4.flatten W₁) idx|) ≤ G :=
    (gradAt_l1_le_card L _ hentry).trans_eq (by rw [hG, cnnConv1GradBound])
  obtain ⟨-, hSR, hSR2⟩ := stepRadius_exact_le L (Kernel4.flatten W₁) hlr hsum
  have haSR := mul_le_mul_of_nonneg_left hSR ha
  refine cnn_conv1_sgd_descends W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _ T ha hx hT hw₂ hW₂
    hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr le_rfl (fun idx => by simp [hL])
    (fun k => lt_of_le_of_lt haSR (hm1 k)) (fun k => lt_of_le_of_lt (by gcongr) (hm2 k))
    (WindowMarginUpTo.mono winRowInv winColInv (by gcongr) hmq)
    (fun l => lt_of_le_of_lt (by gcongr) (hm3 l)) (fun q => lt_of_le_of_lt (by gcongr) (hm4 q))
    (lt_of_le_of_lt (by gcongr) hsmall)
    (by simp only [mul_zero, zero_mul]; try positivity) ?_
  exact curvature_exact_le (fun r => w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (M * (w₂ *
    (K * (a * r)))))))))
    (by positivity) hlr (Finset.sum_nonneg fun _ _ => sq_nonneg _) (by gcongr) hsmall hSR2 hC

-- ════════════════════════════════════════════════════════════════
-- § The conv2 loss-of-bias map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- **Closed form of the conv2 bias loss gradient** at any four-margin
    point — the chain rule through the conv bias map (`gradAt_comp_t3`)
    with the pool-collapsed head gradient (`pool_relu_input_grad`, reused
    verbatim) and the Kronecker bias Jacobian (`conv2d_bias_pdiv`). -/
theorem cnn_conv2_bias_loss_gradAt {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (b : Vec c)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))))) q ≠ 0)
    (o : Fin c) :
    gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
        b o
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then (1:ℝ) else 0) *
            ((if Tensor3.flatten (conv2d W₂ b x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d W₂ b x₁))))
                  ci hi wi
                then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
                  ((if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                        (Tensor3.flatten (conv2d W₂ b x₁))))
                        l > 0 then (1:ℝ) else 0) *
                    ∑ q, W₄ l q *
                      ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d W₂ b x₁)))))) q > 0
                          then (1:ℝ) else 0) *
                        ∑ k, W₅ q k *
                          (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                              (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d W₂ b x₁))))))))) k -
                            oneHot nC label k)))
                else 0)) := by
  refine (gradAt_comp_t3 (fun b' => Tensor3.flatten (conv2d W₂ b' x₁))
    (fun y => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
      (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label) b
    (conv2d_bias_differentiable W₂ x₁ b)
    (pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label _
      hz2 hmp hz3 hz4) _).trans
    (Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl fun hi _ =>
      Finset.sum_congr rfl fun wi _ => ?_)
  rw [conv2d_bias_pdiv W₂ x₁ b o ci hi wi,
    pool_relu_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hmp hz3 hz4 ci hi wi]

/-- **The certified conv-2 BIAS loss gradient, restated as the spatial SUM of
    the `reluMask`-form cotangent** — the bias peer of
    `cnn_conv2_loss_gradAt_reluMask`. The 3-dense head collapses via
    `head3_cot_reluMask`; the channel-Kronecker Jacobian `if ci = o` collapses
    the `∑ ci` to `ci = o` (`Finset.sum_ite_eq'`); the remaining spatial
    `∑ hi wi` is packaged as `∑ s, cotWin c o s` (`convBiasGrad_eq_sum`) — the
    quantity the FloatModel gradient's bias SUM rounds. -/
theorem cnn_conv2_bias_loss_gradAt_reluMask {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (b : Vec c)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))))) q ≠ 0)
    (o : Fin c) :
    gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
        b o
      = ∑ s, cotWin (fun ci hi wi =>
          (if Tensor3.flatten (conv2d W₂ b x₁) (t3Idx ci hi wi) > 0
                then (1:ℝ) else 0) *
            (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b x₁)))) ci hi wi
              then dense (fun j i' => W₃ i' j) (fun _ => 0)
                (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b x₁)))))
                  (dense (fun j i' => W₄ i' j) (fun _ => 0)
                    (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                          (Tensor3.flatten (conv2d W₂ b x₁)))))))
                      (dense (fun j i' => W₅ i' j) (fun _ => 0)
                        (fun k => softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                            (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu
                              (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₂ b x₁))))))))) k - oneHot nC label k)))))
                (t3Idx ci (winRow hi) (winCol wi))
              else 0)) o s := by
  rw [cnn_conv2_bias_loss_gradAt W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label b hz2 hmp hz3
      hz4 o]
  simp_rw [head3_cot_reluMask]
  rw [convBiasGrad_eq_sum _ o]
  simp only [ite_mul, one_mul, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]

-- ════════════════════════════════════════════════════════════════
-- § The conv2-bias capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- The loss as a function of the second-conv bias. -/
noncomputable def cnnConv2BiasLoss {c h w d₃ d₄ nC kH kW : Nat} (W₂ : Kernel4 c c kH kW)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄)
    (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) : Vec c → ℝ :=
  fun b' => crossEntropy nC
    (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c *
      (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b' x₁)))))))))
    label

/-- **One inexact SGD step on the CNN's second conv BIAS decreases one
    example's cross-entropy loss** (`b₂` moving, every other parameter fixed).
    `Conv2Slot.sgd_descends` at the conv2 bias map (`ρ = 1`): the four margins at the step
    radius `D = lr·(‖∇L‖₁ + c·η)` carry no input bound `a` (the bias Jacobian is a Kronecker
    indicator), the pool's taken up to the same twins as the kernel rung (`ConvPatchEq`; twin
    cells are equal for every bias too), and the parameter needs no flatten/unflatten
    plumbing. -/
theorem cnn_conv2_bias_sgd_descends {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec c) (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr η w₃ w₄ w₅ : ℝ} (hT : ∀ p q, T p q → ConvPatchEq kH kW x₁ p q)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ o, |gh o -
      gradAt (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ o| ≤ η)
    (hm2 : ∀ k, stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr η <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQUpTo
      (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr η) T (conv2d W₂ b₂ x₁))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr η)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr η)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr η))))))) < 1)
    (h1 : lr * η * (∑ o, |gradAt
        (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ o|) ≤
      lr * (∑ o, gradAt
        (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ o ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr η))))))))) *
        (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr η) ^ 2 ≤
      lr * (∑ o, gradAt
        (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
          b₂ o ^ 2) / 4) :
    (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (b₂ - lr • gh) ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁))))))))) label -
        lr * (∑ o, gradAt
          (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
            b₂ o ^ 2) / 2 := by
  -- the conv2 bias map, its drift and its point-free Jacobian (`ρ = 1`)
  have hL : cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label =
      Conv2Slot.loss (fun b' => Tensor3.flatten (conv2d W₂ b' x₁)) W₃ b₃ W₄ b₄ W₅ b₅ label := rfl
  simp only [hL] at hgh hm2 hmq hm3 hm4 hsmall h1 h2 ⊢
  simp only [stepRadius] at *
  have hpd : ∀ b' : Vec c, True →
      DifferentiableAt ℝ (fun b'' : Vec c => Tensor3.flatten (conv2d W₂ b'' x₁)) b' ∧
      ∀ idx ci hi wi, pdiv (fun b'' : Vec c => Tensor3.flatten (conv2d W₂ b'' x₁)) b' idx
          (t3Idx ci hi wi) = if ci = idx then (1:ℝ) else 0 :=
    fun b' _ => ⟨conv2d_bias_differentiable W₂ x₁ b', fun idx ci hi wi => by
      rw [conv2d_bias_pdiv]⟩
  have hT' : ∀ p q, T p q → ∀ b' ci, Tensor3.flatten (conv2d W₂ b' x₁) (t3Idx ci p.1 p.2) =
      Tensor3.flatten (conv2d W₂ b' x₁) (t3Idx ci q.1 q.2) := fun p q hpq b' ci => by
    simp only [flatten_t3Idx]
    exact conv2d_eq_of_convPatchEq (hT p q hpq) _ _ ci
  exact Conv2Slot.sgd_descends (ρ := 1) (fun b' => Tensor3.flatten (conv2d W₂ b' x₁))
    W₃ b₃ W₄ b₄ W₅ b₅ label b₂ gh zero_le_one
    (fun v e k => by simpa only [one_mul] using conv2d_flat_bias_drift_total W₂ x₁ v e k)
    (fun v e => by simpa only [one_mul] using conv2d_flat_bias_drift_sum W₂ x₁ v e)
    hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (fun idx ci _ _ => if ci = idx then (1:ℝ) else 0)
    (fun idx => biasRow_l1 idx) (fun _ => True) hpd T hT' hlr hη hgh (fun _ _ => trivial)
    (by simpa only [one_mul, stepRadius] using hm2)
    (by simpa only [one_mul, stepRadius, Tensor3.unflatten_flatten] using hmq)
    (by simpa only [one_mul, stepRadius] using hm3) (by simpa only [one_mul, stepRadius] using hm4)
    (by simpa only [one_mul, stepRadius] using hsmall) h1
    (by simpa only [one_mul, one_pow, mul_one, stepRadius] using h2)

/-- The explicit `ℓ1` bound on the conv2-bias loss gradient, `c·((2h)·(2w)·1)·
    (d₃·w₃·d₄·w₄·nC·w₅)`: `Conv2Slot.gradAt_abs_le` at `ρ = 1` per bias entry, times the number of
    entries. -/
noncomputable def cnnConv2BiasGradBound (c h w d₃ d₄ nC : ℕ) (w₃ w₄ w₅ : ℝ) : ℝ :=
  (c : ℝ) * (((2*h * (2*w) : ℕ) : ℝ) * 1 *
    ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))))

/-- **One exact-gradient SGD step on the conv2 bias decreases the loss, every hypothesis at an
    explicit radius.** `cnn_conv2_bias_sgd_descends` at `η = 0`, the step radius bounded by
    `lr·cnnConv2BiasGradBound` and the second dominance condition by `C·lr·c ≤ 1/4`, as in
    `cnn_conv2_exact_sgd_descends`. -/
theorem cnn_conv2_bias_exact_sgd_descends {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr w₃ w₄ w₅ : ℝ} (hT : ∀ p q, T p q → ConvPatchEq kH kW x₁ p q)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr)
    (hm2 : ∀ k, lr * cnnConv2BiasGradBound c h w d₃ d₄ nC w₃ w₄ w₅ <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQUpTo (lr * cnnConv2BiasGradBound c h w d₃ d₄ nC w₃ w₄ w₅) T
      (conv2d W₂ b₂ x₁))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (lr * cnnConv2BiasGradBound c h w d₃ d₄ nC w₃ w₄ w₅)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (lr * cnnConv2BiasGradBound c h w d₃ d₄ nC w₃ w₄ w₅)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
      (lr * cnnConv2BiasGradBound c h w d₃ d₄ nC w₃ w₄ w₅))))))) < 1)
    (hC : 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (lr * cnnConv2BiasGradBound c h w d₃ d₄ nC w₃ w₄ w₅)))))))) * lr * (c : ℝ) ≤ 1 / 4) :
    (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (b₂ -
        lr • gradAt (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂) ≤
      (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ -
        lr * (∑ o, gradAt (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ o ^ 2) / 2 := by
  set L := cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label with hL
  set G := cnnConv2BiasGradBound c h w d₃ d₄ nC w₃ w₄ w₅ with hG
  set K : ℝ := ((2*h * (2*w) : ℕ) : ℝ) with hK
  have hG0 : 0 ≤ G := by rw [hG, cnnConv2BiasGradBound]; positivity
  have hR0 : 0 ≤ lr * G := mul_nonneg hlr hG0
  -- the gradient's `ℓ1` mass is at most `G`
  have hentry : ∀ o, |gradAt L b₂ o| ≤
      K * 1 * ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) := fun o =>
    Conv2Slot.gradAt_abs_le (fun b' => Tensor3.flatten (conv2d W₂ b' x₁)) W₃ b₃ W₄ b₄ W₅ b₅
      label hR0 hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (fun ci _ _ => if ci = o then (1:ℝ) else 0)
      (biasRow_l1 o) o b₂ (conv2d_bias_differentiable W₂ x₁ b₂)
      (fun ci hi wi => conv2d_bias_pdiv W₂ x₁ b₂ o ci hi wi) T
      (fun p q hpq b' ci => by
        simp only [flatten_t3Idx]
        exact conv2d_eq_of_convPatchEq (hT p q hpq) _ _ ci)
      (fun k => abs_pos.mp (hR0.trans_lt (hm2 k)))
      (by rw [Tensor3.unflatten_flatten]; exact hmq)
      (fun l => abs_pos.mp ((by positivity : (0:ℝ) ≤ w₃ * (K * (lr * G))).trans_lt (hm3 l)))
      (fun q => abs_pos.mp ((by positivity :
        (0:ℝ) ≤ w₄ * ((d₃ : ℝ) * (w₃ * (K * (lr * G))))).trans_lt (hm4 q)))
  have hsum : (∑ o, |gradAt L b₂ o|) ≤ G :=
    (gradAt_l1_le_card L _ hentry).trans_eq (by rw [hG, cnnConv2BiasGradBound])
  obtain ⟨-, hSR, hSR2⟩ := stepRadius_exact_le L b₂ hlr hsum
  refine cnn_conv2_bias_sgd_descends W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label _ T hT hw₃ hW₃ hw₄ hW₄
    hw₅ hW₅ hlr le_rfl (fun o => by simp [hL]) (fun k => lt_of_le_of_lt hSR (hm2 k))
    (WindowMarginUpTo.mono winRowInv winColInv hSR hmq)
    (fun l => lt_of_le_of_lt (by gcongr) (hm3 l)) (fun q => lt_of_le_of_lt (by gcongr) (hm4 q))
    (lt_of_le_of_lt (by gcongr) hsmall)
    (by simp only [mul_zero, zero_mul]; try positivity) ?_
  exact curvature_exact_le (fun r => w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (K * r))))))
    (by positivity) hlr (Finset.sum_nonneg fun _ _ => sq_nonneg _) (by gcongr) hsmall hSR2 hC

-- ════════════════════════════════════════════════════════════════
-- § The conv1 loss-of-bias map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- **Closed form of the conv1 bias loss gradient** at any five-margin
    point — the chain rule through conv1's bias map, contracted with the conv1 head
    gradient (`cnn1_pool_head_input_grad`, reused verbatim): the
    Kronecker bias Jacobian times relu₁'s mask times the point-free
    conv2 tap Jacobian times the pool-collapsed head. -/
theorem cnn_conv1_bias_loss_gradAt {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC)
    (b : Vec c)
    (hz1 : ∀ k, Tensor3.flatten (conv2d W₁ b x₀) k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₁ b x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))))) q ≠ 0)
    (o : Fin c) :
    gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀)))))))))))))
          label)
        b o
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then (1:ℝ) else 0) *
            ((if Tensor3.flatten (conv2d W₁ b x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d W₁ b x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₁ b
                                  x₀)))))))) co ho wo
                      then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                        ((if dense W₃ b₃ (maxPoolFlat c h w
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₂ b₂ (Tensor3.unflatten
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d
                                      W₁ b
                                      x₀)))))))) l > 0
                            then (1:ℝ) else 0) *
                          ∑ q, W₄ l q *
                            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                  (maxPoolFlat c h w (relu
                                    (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten
                                      (relu (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d
                                          W₁ b
                                          x₀)))))))))) q > 0
                                then (1:ℝ) else 0) *
                              ∑ k, W₅ q k *
                                (softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d W₂ b₂
                                          (Tensor3.unflatten (relu
                                            (c * (2*h) * (2*w))
                                            (Tensor3.flatten (conv2d
                                              W₁ b
                                              x₀))))))))))))) k -
                                  oneHot nC label k)))
                      else 0))) := by
  refine (gradAt_comp_t3 (fun b' => Tensor3.flatten (conv2d W₁ b' x₀))
    (fun y => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
      (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) y))))))))))) label) b
    (conv2d_bias_differentiable W₁ x₀ b)
    (cnn1_pool_head_differentiableAt W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _
      hz1 hz2 hmp hz3 hz4) _).trans
    (Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl fun hi _ =>
      Finset.sum_congr rfl fun wi _ => ?_)
  rw [conv2d_bias_pdiv W₁ x₀ b o ci hi wi,
    cnn1_pool_head_input_grad W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _ hz1 hz2 hmp hz3 hz4 ci hi wi]

/-- **The certified conv-1 BIAS loss gradient, restated as the spatial SUM of
    the `reluMask`-form cotangent** — the bias peer of
    `cnn_conv1_loss_gradAt_reluMask`, one conv-backward deeper. The head
    collapses via `head3_cot_reluMask`, the conv-2 backward stays the explicit
    `∑ convTap·c₂`, and the channel-Kronecker conv-1 Jacobian collapses the
    `∑ ci` (`Finset.sum_ite_eq'` + `convBiasGrad_eq_sum`) to the spatial SUM
    the float bias gradient rounds. -/
theorem cnn_conv1_bias_loss_gradAt_reluMask {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (b : Vec c)
    (hz1 : ∀ k, Tensor3.flatten (conv2d W₁ b x₀) k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))))) q ≠ 0)
    (o : Fin c) :
    gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀)))))))))))))
          label)
        b o
      = ∑ s,
          cotWin (fun ci hi wi =>
            (if Tensor3.flatten (conv2d W₁ b x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d W₁ b x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₁ b x₀))))))))
                          co ho wo
                      then dense (fun j i' => W₃ i' j) (fun _ => 0)
                        (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d W₂ b₂ (Tensor3.unflatten
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d W₁ b x₀)))))))))
                          (dense (fun j i' => W₄ i' j) (fun _ => 0)
                            (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                                (dense W₃ b₃ (maxPoolFlat c h w (relu
                                  (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten (relu
                                      (c * (2*h) * (2*w)) (Tensor3.flatten
                                        (conv2d W₁
                                          b x₀)))))))))))
                              (dense (fun j i' => W₅ i' j) (fun _ => 0)
                                (fun k => softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w)) (Tensor3.flatten
                                          (conv2d W₂ b₂ (Tensor3.unflatten
                                            (relu (c * (2*h) * (2*w))
                                              (Tensor3.flatten (conv2d
                                                W₁
                                                b x₀))))))))))))) k -
                                  oneHot nC label k)))))
                        (t3Idx co (winRow ho) (winCol wo))
                      else 0))) o s := by
  rw [cnn_conv1_bias_loss_gradAt W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label b
      hz1 hz2 hmp hz3 hz4 o]
  simp_rw [head3_cot_reluMask]
  rw [convBiasGrad_eq_sum _ o]
  simp only [ite_mul, one_mul, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero,
    Finset.sum_ite_eq', Finset.mem_univ, ite_true]

-- ════════════════════════════════════════════════════════════════
-- § The conv1-bias capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- The loss as a function of the first-conv bias. -/
noncomputable def cnnConv1BiasLoss {ic c h w d₃ d₄ nC kH kW : Nat} (W₁ : Kernel4 c ic kH kW)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃)
    (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) :
    Vec c → ℝ :=
  fun b' => crossEntropy nC
    (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c *
      (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₁ b' x₀)))))))))))))
    label

/-- **One inexact SGD step on the CNN's FIRST conv BIAS decreases one
    example's cross-entropy loss** (`b₁` moving, every other parameter fixed).
    `Conv1Slot.sgd_descends` at the conv1 bias map (`ρ = 1`): the FIVE margins at the step
    radius `D = lr·(‖∇L‖₁ + c·η)` carry no input bound `a` (the bias Jacobian is a Kronecker
    indicator), the pool's taken up to two-layer twins (`ConvPatchEq2`), and the parameter
    needs no flatten/unflatten plumbing. With this theorem both conv kernels,
    both conv biases and the three dense layers' weights and biases (via the
    MLP rungs, `SgdDescent.Mlp` and `SgdDescent.MlpBias`) of the Chapter-3 CNN
    each have a single-layer, single-example descent statement. -/
theorem cnn_conv1_bias_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec c) (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr η w₂ w₃ w₄ w₅ : ℝ} (hT : ∀ p q, T p q → ConvPatchEq2 kH kW x₀ p q)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - (gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁) idx| ≤ η)
    (hm1 : ∀ k, lr * (((∑ idx, |gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * η)) < |(Tensor3.flatten (conv2d W₁ b₁ x₀)) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * η)))) < |(Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))) k|)
    (hmq : MaxPool2MarginQUpTo (((c * kH * kW : ℕ) : ℝ) * (w₂ * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * η))))) T
      (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr *
      (((∑ idx, |gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * η)))))) < |(dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h *
      (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * η))))))))
      < |(dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * η))))))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) ≤
      lr * (∑ idx, (gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 *
      (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 / (1 - 2 * (w₅ *
      ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h *
      (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * η))))))))))))) * (stepRadius (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) b₁ lr η) ^ 2 ≤
      lr * (∑ idx, (gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁) idx ^ 2) / 4) :
    (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (b₁ - lr • gh) ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))))) label -
        lr * (∑ idx, (gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁) idx ^ 2) / 2 := by
  -- the conv1 bias map, its drift and its point-free Jacobian (`ρ = 1`)
  have hL : cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label =
      Conv2Slot.loss (fun b' => Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) ((fun b'' : Vec c => Tensor3.flatten (conv2d W₁ b'' x₀)) b')))))
        W₃ b₃ W₄ b₄ W₅ b₅ label := rfl
  simp only [hL] at hgh hm1 hm2 hmq hm3 hm4 hsmall h1 h2 ⊢
  simp only [stepRadius] at *
  have hpd : ∀ b' : Vec c,
      DifferentiableAt ℝ (fun b'' : Vec c => Tensor3.flatten (conv2d W₁ b'' x₀)) b' ∧
      ∀ idx ci hi wi, pdiv (fun b'' : Vec c => Tensor3.flatten (conv2d W₁ b'' x₀)) b' idx
          (t3Idx ci hi wi) = if ci = idx then (1:ℝ) else 0 :=
    fun b' => ⟨conv2d_bias_differentiable W₁ x₀ b', fun idx ci hi wi => by
      rw [conv2d_bias_pdiv]⟩
  have hT' : ∀ p q, T p q → ∀ b' ci,
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          ((fun b'' : Vec c => Tensor3.flatten (conv2d W₁ b'' x₀)) b')))) (t3Idx ci p.1 p.2) =
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          ((fun b'' : Vec c => Tensor3.flatten (conv2d W₁ b'' x₀)) b')))) (t3Idx ci q.1 q.2) :=
    fun p q hpq b' ci => by
      simp only [flatten_t3Idx]
      exact conv2d_eq_of_convPatchEq (convPatchEq_relu_conv (hT p q hpq) _ _) _ _ ci
  exact Conv1Slot.sgd_descends (ρ := 1) (fun b'' : Vec c => Tensor3.flatten (conv2d W₁ b'' x₀))
    W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label b₁ gh zero_le_one
    (fun v e k => by simpa only [one_mul] using conv2d_flat_bias_drift_total W₁ x₀ v e k)
    (fun v e => by simpa only [one_mul] using conv2d_flat_bias_drift_sum W₁ x₀ v e)
    hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (fun idx ci _ _ => if ci = idx then (1:ℝ) else 0)
    (fun idx => biasRow_l1 idx) hpd T hT' hlr hη hgh
    (by simpa only [one_mul, stepRadius] using hm1) (by simpa only [one_mul, stepRadius] using hm2)
    (by simpa only [one_mul, stepRadius] using hmq)
    (by simpa only [one_mul, stepRadius] using hm3) (by simpa only [one_mul, stepRadius] using hm4)
    (by simpa only [one_mul, stepRadius] using hsmall) h1
    (by simpa only [one_mul, one_pow, mul_one, stepRadius] using h2)

/-- The explicit `ℓ1` bound on the conv1-bias loss gradient, `c·((2h)·(2w)·(c·kH·kW·w₂·1))·
    (d₃·w₃·d₄·w₄·nC·w₅)`: `Conv1Slot.gradAt_abs_le` at `ρ = 1` per bias entry, times the number of
    entries. -/
noncomputable def cnnConv1BiasGradBound (c h w d₃ d₄ nC kH kW : ℕ) (w₂ w₃ w₄ w₅ : ℝ) : ℝ :=
  (c : ℝ) * (((2*h * (2*w) : ℕ) : ℝ) * (((c * kH * kW : ℕ) : ℝ) * (w₂ * 1)) *
    ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))))

/-- **One exact-gradient SGD step on the conv1 bias decreases the loss, every hypothesis at an
    explicit radius.** `cnn_conv1_bias_sgd_descends` at `η = 0`, the step radius bounded by
    `lr·cnnConv1BiasGradBound` and the second dominance condition by `C·lr·c ≤ 1/4`, as in
    `cnn_conv2_exact_sgd_descends`. -/
theorem cnn_conv1_bias_exact_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    {lr w₂ w₃ w₄ w₅ : ℝ} (hT : ∀ p q, T p q → ConvPatchEq2 kH kW x₀ p q)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr)
    (hm1 : ∀ k, lr * cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅ <
      |Tensor3.flatten (conv2d W₁ b₁ x₀) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (lr * cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅)) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k|)
    (hmq : MaxPool2MarginQUpTo (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (lr * cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅))) T
      (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (lr * cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) *
          (lr * cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) *
        (lr * cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅))))))))) < 1)
    (hC : 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 *
        (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) *
            (lr * cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅)))))))))) *
        lr * (c : ℝ) ≤ 1 / 4) :
    (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (b₁ -
        lr • gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) b₁) ≤
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) b₁ -
        lr * (∑ o, gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) b₁ o ^ 2) /
          2 := by
  set L := cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label with hL
  set G := cnnConv1BiasGradBound c h w d₃ d₄ nC kH kW w₂ w₃ w₄ w₅ with hG
  set K : ℝ := ((2*h * (2*w) : ℕ) : ℝ) with hK
  set M : ℝ := ((c * kH * kW : ℕ) : ℝ) with hM
  have hG0 : 0 ≤ G := by rw [hG, cnnConv1BiasGradBound]; positivity
  have hR0 : 0 ≤ lr * G := mul_nonneg hlr hG0
  -- the conv1 bias map (`ρ = 1`) and the gradient's `ℓ1` mass
  have hentry : ∀ o, |gradAt L b₁ o| ≤
      K * (M * (w₂ * 1)) * ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * 1)))))) :=
    fun o => Conv1Slot.gradAt_abs_le (fun b'' : Vec c => Tensor3.flatten (conv2d W₁ b'' x₀))
      W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label (by positivity : (0:ℝ) ≤ M * (w₂ * (lr * G))) hw₂ hW₂
      hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (fun idx ci _ _ => if ci = idx then (1:ℝ) else 0)
      (fun idx => biasRow_l1 idx)
      (fun b' => ⟨conv2d_bias_differentiable W₁ x₀ b', fun idx ci hi wi => by
        rw [conv2d_bias_pdiv]⟩) T
      (fun p q hpq b' ci => by
        simp only [flatten_t3Idx]
        exact conv2d_eq_of_convPatchEq (convPatchEq_relu_conv (hT p q hpq) _ _) _ _ ci)
      o b₁ (fun k => abs_pos.mp (hR0.trans_lt (hm1 k)))
      (fun k => abs_pos.mp ((by positivity : (0:ℝ) ≤ M * (w₂ * (lr * G))).trans_lt (hm2 k)))
      hmq
      (fun l => abs_pos.mp ((by positivity :
        (0:ℝ) ≤ w₃ * (M * (w₂ * (K * (lr * G))))).trans_lt (hm3 l)))
      (fun q => abs_pos.mp ((by positivity :
        (0:ℝ) ≤ w₄ * ((d₃ : ℝ) * (w₃ * (M * (w₂ * (K * (lr * G))))))).trans_lt (hm4 q)))
  have hsum : (∑ o, |gradAt L b₁ o|) ≤ G :=
    (gradAt_l1_le_card L _ hentry).trans_eq (by rw [hG, cnnConv1BiasGradBound])
  obtain ⟨-, hSR, hSR2⟩ := stepRadius_exact_le L b₁ hlr hsum
  -- the conv1-bias rung spells its radius out: `lr·(‖∇L‖₁ + c·η)`
  have hsum0 : (∑ o, |gradAt L b₁ o|) + (c : ℝ) * 0 ≤ G := by
    rw [mul_zero, add_zero]; exact hsum
  refine cnn_conv1_bias_sgd_descends W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _ T hT hw₂ hW₂
    hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr le_rfl (fun o => by simp [hL])
    (fun k => lt_of_le_of_lt hSR (hm1 k)) (fun k => lt_of_le_of_lt (by gcongr) (hm2 k))
    (WindowMarginUpTo.mono winRowInv winColInv (by gcongr) hmq)
    (fun l => lt_of_le_of_lt (by gcongr) (hm3 l)) (fun q => lt_of_le_of_lt (by gcongr) (hm4 q))
    (lt_of_le_of_lt (by gcongr) hsmall)
    (by simp only [mul_zero, zero_mul]; try positivity) ?_
  exact curvature_exact_le (fun r => w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (M * (w₂ *
    (K * r))))))))
    (by positivity) hlr (Finset.sum_nonneg fun _ _ => sq_nonneg _) (by gcongr) hsmall hSR2 hC

end Proofs
