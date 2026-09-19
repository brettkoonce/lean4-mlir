import LeanMlir.Proofs.Nets.Small.CnnChainClose
import LeanMlir.Proofs.Codegen.CnnRender
import LeanMlir.Proofs.Nets.Small.CifarCNN

/-! # PoC: the CIFAR-CNN (Chapter 4, no-BN) train step, proof-tied to the certified SGD step

The Chapter-4 peer of `CnnFold` — a deeper, two-spatial-scale conv net
(`(conv→relu)×2 → pool → (conv→relu)×2 → pool → (dense→relu)×2 → dense`; 14 params:
4 conv kernels/biases + 3 dense layers). `MainCifarVerified` trains on
`verified_mlir/cifar_train_step.mlir`; this file makes its parameter updates
`den`-faithful — each emitted SGD op denotes the certified loss-descent step.

**Zero new core ops.** The conv layers reuse the `convWeightSgd`/`convBiasSgd` ops
added for cnn (CnnFold); the dense head reuses `weightSgd`/`biasSgd`. The
only new content is the per-net `den = certified` capstones below.

* **Conv layers (all four):** `convW_den`/`convB_den` are *generic* in the conv dims
  and the cotangent `c` — `den (convWeightSgd … (.operand _ c)) = θ − lr·(certified
  ∂conv/∂θ · c)`, the emitted op's `den` reduced (`rfl`) to the LHS of the generic
  `cnn_render_conv{W,b}_certified`. Instantiated at each conv layer's `(b,x,W)` and the
  cotangent the renderer feeds there, they certify W₁/b₁ … W₄/b₄ (one lemma each, all
  four layers — conv2d's weight/bias VJP is dim-generic).
* **Dense head (W₅/W₆/W₇):** the classifier head is a 3-layer MLP over the flattened
  pool output, so its cotangents are the IR `mlpCotOut0/1` and its `den`s close via the
  M2 `weight_grad_bridge`/`bias_grad_bridge` — verbatim `CnnFold` (`dW3..5`).

## Honest residual (same boundary as cnn/mlp/linear)
* The conv cotangents here are free variables `c` (the `convW_den`/`convB_den` statement
  is ∀ c) — so the lemmas hold at the actual backward-chain cotangent the renderer
  feeds, without naming it. Pinning each `c` to the exact emitted backward subgraph
  (the `CnnChainClose` recipe, scaled to two stages) is the remaining polish.
* Per-op `pretty` lexing + ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CifarPoC

/-! ## Conv layers — generic `den = certified` (covers all four conv layers) -/

/-- **Any emitted conv weight op = certified.** Generic in the conv dims and the
    cotangent `c`: the `convWeightSgd` op denotes `flatten W − lr·(certified
    ∂conv/∂W · c)`. Instantiated at each layer's `(b,x,W,c)` it certifies W₁…W₄. -/
theorem convW_den {ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Tensor3 ic h w)
    (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*kH*kW)) :
    den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               (Kernel4.flatten W) idx j * c j :=
  cnn_render_convW_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted conv bias op = certified.** Generic peer of `convW_den`. -/
theorem convB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  cnn_render_convb_certified W x b c lr o

/-! ## Dense classifier head (W₅/W₆/W₇) — `weightSgd`/`biasSgd`, mirrors `CnnPoC`

The head `pool2 → W₅→relu→W₆→relu→W₇` is a 3-layer MLP; per-layer cotangents are the
IR `mlpCotOut0/1` (with `(W₇,W₆,W₅)` playing the MLP's `(W₂,W₁,W₀)`). -/

/-- Output-layer weight op `W₇` = certified step (cotangent = the loss cotangent `dy`). -/
theorem dW7_den {c2 h w d1 nClasses : Nat}
    (aN lrStr dyN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin d1) (j : Fin nClasses) :
    den (SHlo.weightSgd aN "%W7" lrStr (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) W₇ lr
          (.operand dyN dy)) (finProdFinEquiv (i, j))
      = W₇ i j - lr * ∑ k : Fin nClasses,
          pdiv (fun v : Vec (d1 * nClasses) =>
                  dense (Mat.unflatten v) b₇ (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))))
               (Mat.flatten W₇) (finProdFinEquiv (i, j)) k * dy k :=
  Cifar8PoC.denseW_den aN "%W7" lrStr dyN _ W₇ b₇ _ lr i j

/-- Hidden-layer weight op `W₆` = certified step (cotangent = `mlpCotOut1 W₇ h6`). -/
theorem dW6_den {c2 h w d1 nClasses : Nat}
    (aN lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses) (lr : ℝ) (i j : Fin d1) :
    den (SHlo.weightSgd aN "%W6" lrStr (relu d1 (dense W₅ b₅ pool)) W₆ lr
          (.operand cN ((mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy)))
        (finProdFinEquiv (i, j))
      = W₆ i j - lr * ∑ k : Fin d1,
          pdiv (fun v : Vec (d1 * d1) => dense (Mat.unflatten v) b₆ (relu d1 (dense W₅ b₅ pool)))
               (Mat.flatten W₆) (finProdFinEquiv (i, j)) k
            * (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy k :=
  Cifar8PoC.denseW_den aN "%W6" lrStr cN _ W₆ b₆ _ lr i j

/-- Input-layer (pool) weight op `W₅` = certified step (cotangent = `mlpCotOut0 W₆ W₇ h5 h6`). -/
theorem dW5_den {c2 h w d1 nClasses : Nat}
    (lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin (c2*h*w)) (j : Fin d1) :
    den (SHlo.weightSgd "%pool2" "%W5" lrStr pool W₅ lr
          (.operand cN ((mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                          (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy)))
        (finProdFinEquiv (i, j))
      = W₅ i j - lr * ∑ k : Fin d1,
          pdiv (fun v : Vec ((c2*h*w) * d1) => dense (Mat.unflatten v) b₅ pool)
               (Mat.flatten W₅) (finProdFinEquiv (i, j)) k
            * (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy k :=
  Cifar8PoC.denseW_den "%pool2" "%W5" lrStr cN _ W₅ b₅ _ lr i j

/-- Output-layer bias op `b₇` = certified step. -/
theorem db7_den {c2 h w d1 nClasses : Nat}
    (lrStr dyN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin nClasses) :
    den (SHlo.biasSgd "%b7" lrStr b₇ lr (.operand dyN dy)) i
      = b₇ i - lr * ∑ j : Fin nClasses,
          pdiv (fun b' : Vec nClasses =>
                  dense W₇ b' (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool))))) b₇ i j * dy j :=
  Cifar8PoC.denseB_den "%b7" lrStr dyN W₇ _ b₇ _ lr i

/-- Hidden-layer bias op `b₆` = certified step. -/
theorem db6_den {c2 h w d1 nClasses : Nat}
    (lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses) (lr : ℝ) (i : Fin d1) :
    den (SHlo.biasSgd "%b6" lrStr b₆ lr
          (.operand cN ((mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy))) i
      = b₆ i - lr * ∑ j : Fin d1,
          pdiv (fun b' : Vec d1 => dense W₆ b' (relu d1 (dense W₅ b₅ pool))) b₆ i j
            * (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy j :=
  Cifar8PoC.denseB_den "%b6" lrStr cN W₆ _ b₆ _ lr i

/-- Input-layer (pool) bias op `b₅` = certified step. -/
theorem db5_den {c2 h w d1 nClasses : Nat}
    (lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses) (lr : ℝ) (i : Fin d1) :
    den (SHlo.biasSgd "%b5" lrStr b₅ lr
          (.operand cN ((mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                          (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy))) i
      = b₅ i - lr * ∑ j : Fin d1,
          pdiv (fun b' : Vec d1 => dense W₅ b' pool) b₅ i j
            * (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy j :=
  Cifar8PoC.denseB_den "%b5" lrStr cN W₅ _ b₅ _ lr i

/-! ## The §1a tie — the conv layers/dense head, tied through the REAL cifar forward

The conv/dense `*_den` theorems above hold for a FREE cotangent (`convW_den`/`convB_den` are `∀ c`;
the dense head's `mlpCotOut0/1` are `∀ dy`). The capstones below pin those cotangents to the ones the
**real cifar forward + softmax-CE loss** actually drives — the cifar peer of `CnnFold`'s last
three theorems (`cnnLossCot_den` / `cnn_W5_tied_totalloss` / `cnn_conv_tied_certified`).

cifar is the cnn shape with **two** conv→conv→pool stages instead of one, so its conv backward chain
crosses an extra pool boundary. Three of the four conv-layer cotangents reuse the cnn chain cots
verbatim (every one is generic in its downstream cotangent):

* **W₄** (last conv before pool₂): `cnnChainCotW2 W₅ W₆ W₇ h5 h6 ac4 hc4 g` — relu₄ mask on the pool₂
  input-VJP of the dense-head cotangent (the cnn conv₂ pattern, at the cifar head dims).
* **W₃**: `cnnChainCotW1 W₄ hc3 cotW4` — relu₃ mask on conv₄'s input-VJP (the cnn conv₁ pattern).
* **W₁**: `cnnChainCotW1 W₂ hc1 cotW2` — relu₁ mask on conv₂'s input-VJP (same pattern).

Only **W₂** needs a new constructor `cifarChainCotW2`: its cotangent crosses pool₁ at the *relu-free*
conv₃-input boundary, so it is relu₂ mask on `maxpool₁-back(conv₃-back(W₃, cotW3))` — a conv input-VJP
*then* a maxpool input-VJP, the step cnn (one pool) never had. -/

/-- Cotangent the cifar backward chain delivers at **conv₂'s output** (`c1` ch @ `2(2h)`): the relu₂
    mask on `maxpool₁-back(conv₃-back(W₃, cotW3))`. `conv₃-back` (the `Back3.conv` input-VJP via
    `flatDenote`) carries `cotW3` from conv₃'s output to pool₁'s output (`c1` @ `2h`); `maxpool₁-back`
    (the `Back3.maxpool` input-VJP) lifts that to conv₂'s output (`c1` @ `2(2h)`). `ac2` is the pool₁
    input (= relu₂ output), `hc2` the conv₂ pre-activation (the relu₂ mask). -/
noncomputable def cifarChainCotW2 {c1 c2 h w kH kW : Nat}
    (W₃ : Kernel4 c2 c1 kH kW)
    (ac2 : Tensor3 c1 (2*(2*h)) (2*(2*w))) (hc2 : Vec (c1 * (2*(2*h)) * (2*(2*w))))
    (cotW3 : Vec (c2 * (2*h) * (2*w))) : Vec (c1 * (2*(2*h)) * (2*(2*w))) :=
  fun i => if hc2 i > 0
    then (Back3.maxpool (c₁ := c1) (h₁ := 2*h) (w₁ := 2*w) ac2 Back3.cot).flatDenote
           ((Back3.conv (c₁ := c2) (h₁ := 2*h) (w₁ := 2*w) W₃ Back3.cot).flatDenote cotW3) i
    else 0

/-- **The emitted loss-cotangent graph denotes the composed softmax-CE gradient of the cifar forward**
    (`= softmax(cifarCnnForward … x) − onehot = ∂CE/∂logits` at the real forward logits). The cifar
    peer of `CnnPoC.cnnLossCot_den` (same proof, `cifarCnnForward` for the logits operand). -/
theorem cifarLossCot_den {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (nlogN ohN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) (label : Fin nClasses) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe
            (.operand nlogN (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x))))
          (.operand ohN (oneHot nClasses label)))
      = fun j => softmax nClasses (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x) j
                  - oneHot nClasses label j := by
  funext j; simp only [den, softmax]

/-- **Dense output weight `W₇`, tied to the WHOLE softmax-CE loss through the cifar forward.** With the
    dense-head input = the real cifar forward pool₂ output and the cotangent the emitted loss graph
    denotes (`cifarLossCot_den`), the `weightSgd` for `W₇` denotes `W₇ − lr·∂(crossEntropy ∘ forward)/∂W₇`.
    The cifar peer of `CnnPoC.cnn_W5_tied_totalloss`. -/
theorem cifar_W7_tied_totalloss {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (aN lrStr dyN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) (label : Fin nClasses)
    (lr : ℝ) (i : Fin d1) (j : Fin nClasses) :
    den (SHlo.weightSgd aN "%W7" lrStr
          (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅
            (maxPoolFlat c2 h w (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₄ b₄
              (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₃ b₃
                (maxPoolFlat c1 (2*h) (2*w) (relu (c1*(2*(2*h))*(2*(2*w)))
                  (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂
                    (relu (c1*(2*(2*h))*(2*(2*w)))
                      (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ x)))))))))))))) W₇ lr
          (.operand dyN (fun k => softmax nClasses
              (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x) k - oneHot nClasses label k)))
        (finProdFinEquiv (i, j))
      = W₇ i j - lr * pdiv (fun v : Vec (d1 * nClasses) => fun _ : Fin 1 =>
            crossEntropy nClasses (dense (Mat.unflatten v) b₇
              (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅
                (maxPoolFlat c2 h w (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₄ b₄
                  (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₃ b₃
                    (maxPoolFlat c1 (2*h) (2*w) (relu (c1*(2*(2*h))*(2*(2*w)))
                      (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂
                        (relu (c1*(2*(2*h))*(2*(2*w)))
                          (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ x))))))))))))))) label)
          (Mat.flatten W₇) (finProdFinEquiv (i, j)) 0 := by
  rw [dW7_den aN lrStr dyN W₅ b₅ W₆ b₆ W₇ b₇
        (maxPoolFlat c2 h w (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₄ b₄
          (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₃ b₃
            (maxPoolFlat c1 (2*h) (2*w) (relu (c1*(2*(2*h))*(2*(2*w)))
              (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂
                (relu (c1*(2*(2*h))*(2*(2*w)))
                  (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ x))))))))))
        (fun k => softmax nClasses (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x) k
          - oneHot nClasses label k) lr i j,
      mlp_output_total_loss_grad W₇ b₇
        (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅
          (maxPoolFlat c2 h w (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₄ b₄
            (relu (c2*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₃ b₃
              (maxPoolFlat c1 (2*h) (2*w) (relu (c1*(2*(2*h))*(2*(2*w)))
                (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂
                  (relu (c1*(2*(2*h))*(2*(2*w)))
                    (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ x)))))))))))))) label i j]
  -- both the threaded loss cotangent (`cifarCnnForward`) and the fold's `mnistLinear W₇ b₇ a₆`
  -- are `dense W₇ b₇ (relu … pool₂)` — unfold both to match.
  simp only [cifarCnnForward, mnistLinear, Function.comp_apply]

set_option maxRecDepth 8000 in
/-- **Whole cifar conv tail, tied.** All four conv kernel/bias ops, at the real cifar forward and the
    composed softmax-CE cotangent `g = softmax(cifarCnnForward … xv) − onehot` (`cifarLossCot_den`),
    denote the certified loss-descent step. Each `den = certified` is the generic `convW_den`/`convB_den`
    instantiated at the cotangent the backward chain delivers: `cnnChainCotW2` for conv₄ (relu mask on
    pool₂-back of the dense head), `cnnChainCotW1` for conv₃/conv₁ (relu mask on the next conv's
    input-VJP), and `cifarChainCotW2` for conv₂ (relu mask on pool₁-back of conv₃'s input-VJP). Together
    with the dense head (`cifar_W7_tied_totalloss` + `dW5`/`dW6`/`db5`/`db6`/`db7` at `g`) the WHOLE
    cifar train step is den-composed forward→loss→backward — no free activations, no symbolic cotangent.
    (Residual: the conv backward is rendered hand-written, so the cotangent SSA ↔ chain-cot
    correspondence is the per-op trust the whole suite carries — the cnn `cnn_conv_tied_certified`
    residual verbatim.) -/
theorem cifar_conv_tied_certified {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (xN wN bN lrStr cotN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Tensor3 ic (2*(2*h)) (2*(2*w))) (label : Fin nClasses) (lr : ℝ) :
    -- the forward runs in flat `Vec` space (`flatConv`/`maxPoolFlat`); the backward/SGD read `Tensor3`
    -- activations (`conv2d`), so each conv activation appears as its `Vec` form and the
    -- `Tensor3.unflatten` of it — bridged in the statement (the `*_den` hold for any activation).
    let xv : Vec (ic*(2*(2*h))*(2*(2*w))) := Tensor3.flatten x
    let hc1 : Vec (c1*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ xv
    let ac1v : Vec (c1*(2*(2*h))*(2*(2*w))) := relu (c1*(2*(2*h))*(2*(2*w))) hc1
    let ac1 : Tensor3 c1 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten ac1v
    let hc2 : Vec (c1*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂ ac1v
    let ac2v : Vec (c1*(2*(2*h))*(2*(2*w))) := relu (c1*(2*(2*h))*(2*(2*w))) hc2
    let ac2 : Tensor3 c1 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten ac2v
    let zp1 : Vec (c1*(2*h)*(2*w)) := maxPoolFlat c1 (2*h) (2*w) ac2v
    let zp1t : Tensor3 c1 (2*h) (2*w) := Tensor3.unflatten zp1
    let hc3 : Vec (c2*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₃ b₃ zp1
    let ac3v : Vec (c2*(2*h)*(2*w)) := relu (c2*(2*h)*(2*w)) hc3
    let ac3 : Tensor3 c2 (2*h) (2*w) := Tensor3.unflatten ac3v
    let hc4 : Vec (c2*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₄ b₄ ac3v
    let ac4v : Vec (c2*(2*h)*(2*w)) := relu (c2*(2*h)*(2*w)) hc4
    let ac4 : Tensor3 c2 (2*h) (2*w) := Tensor3.unflatten ac4v
    let zp2 : Vec (c2*h*w) := maxPoolFlat c2 h w ac4v
    let h5 : Vec d1 := dense W₅ b₅ zp2
    let h6 : Vec d1 := dense W₆ b₆ (relu d1 h5)
    let g : Vec nClasses := fun k =>
      softmax nClasses (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ xv) k
        - oneHot nClasses label k
    let cotW4 : Vec (c2*(2*h)*(2*w)) := cnnChainCotW2 W₅ W₆ W₇ h5 h6 ac4 hc4 g
    let cotW3 : Vec (c2*(2*h)*(2*w)) := cnnChainCotW1 W₄ hc3 cotW4
    let cotW2 : Vec (c1*(2*(2*h))*(2*(2*w))) := cifarChainCotW2 W₃ ac2 hc2 cotW3
    let cotW1 : Vec (c1*(2*(2*h))*(2*(2*w))) := cnnChainCotW1 W₂ hc1 cotW2
    -- conv₄ (last conv before pool₂)
    ConvWSgdTied xN wN lrStr cotN b₄ ac3 W₄ cotW4 lr
  ∧ ConvBSgdTied bN lrStr cotN W₄ ac3 b₄ cotW4 lr
  -- conv₃
  ∧ ConvWSgdTied xN wN lrStr cotN b₃ zp1t W₃ cotW3 lr
  ∧ ConvBSgdTied bN lrStr cotN W₃ zp1t b₃ cotW3 lr
  -- conv₂ (across pool₁ — the new `cifarChainCotW2` cotangent)
  ∧ ConvWSgdTied xN wN lrStr cotN b₂ ac1 W₂ cotW2 lr
  ∧ ConvBSgdTied bN lrStr cotN W₂ ac1 b₂ cotW2 lr
  -- conv₁ (input layer)
  ∧ ConvWSgdTied xN wN lrStr cotN b₁ x W₁ cotW1 lr
  ∧ ConvBSgdTied bN lrStr cotN W₁ x b₁ cotW1 lr := by
  intro xv hc1 ac1v ac1 hc2 ac2v ac2 zp1 zp1t hc3 ac3v ac3 hc4 ac4v ac4 zp2 h5 h6 g cotW4 cotW3 cotW2 cotW1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact convW_den xN wN lrStr cotN b₄ ac3 W₄ cotW4 lr idx
  · intro o;   exact convB_den bN lrStr cotN W₄ ac3 b₄ cotW4 lr o
  · intro idx; exact convW_den xN wN lrStr cotN b₃ zp1t W₃ cotW3 lr idx
  · intro o;   exact convB_den bN lrStr cotN W₃ zp1t b₃ cotW3 lr o
  · intro idx; exact convW_den xN wN lrStr cotN b₂ ac1 W₂ cotW2 lr idx
  · intro o;   exact convB_den bN lrStr cotN W₂ ac1 b₂ cotW2 lr o
  · intro idx; exact convW_den xN wN lrStr cotN b₁ x W₁ cotW1 lr idx
  · intro o;   exact convB_den bN lrStr cotN W₁ x b₁ cotW1 lr o

end Proofs.CifarPoC
