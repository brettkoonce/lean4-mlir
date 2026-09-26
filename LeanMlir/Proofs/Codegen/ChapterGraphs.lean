import LeanMlir.Proofs.Codegen.StableHLOPretty

/-! # The chapter-net graphs and their printers (chapters 1–4)

The whole-net forward (and, for the MLP and the MNIST CNN, backward) `SHlo` graphs of the
book's small nets, and the `renderModule` printers that write the committed
`verified_mlir/` forwards from them (`ChapterArtifacts`' `#eval`s):

| net | graph | printer |
|---|---|---|
| linear (ch 1) | `fwdGraph`, `backGraph`, `lossCotGraph` (in `StableHLO`) | `linearFwdModuleV`, `linearBackModuleV`, `linearTrainStepModuleV` |
| MLP (ch 2) | `mlpFwdGraph`, `mlpBackGraph` | `mlpFwdModuleV` |
| MNIST CNN (ch 3) | `cnnFwdGraph`, `cnnBackGraph` | `cnnFwdModuleV` |
| CIFAR CNN (ch 4) | `cifarFwdGraph`, `cifar8FwdGraph`, `cifar8BnFwdGraph` | `cifarFwdModuleV`, `cifar8FwdModuleV`, `cifar8BnFwdModuleV` |

The MLP graphs' faithfulness theorems are here; that the chapter 3–4 graphs denote their nets is
`ChapterGraphTies`, since the IR imports no net. The op vocabulary, `den`, `pretty`
and `renderModule` are `StableHLO` / `StableHLOPretty`, which the batched ImageNet nets share,
so an edit to a chapter net stays out of their rebuild.
-/

namespace Proofs
namespace StableHLO

variable {e₀ e₁ e₂ e₃ : Nat}

-- ════════════════════════════════════════════════════════════════
-- § Chapter 2 — the MLP
-- ════════════════════════════════════════════════════════════════

/-- Whole-MLP **forward** graph `dense W₂ ∘ relu ∘ dense W₁ ∘ relu ∘ dense W₀`. -/
def mlpFwdGraph (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀) : SHlo e₃ :=
  denseF "%W2" "%b2" W₂ b₂ (.reluF (denseF "%W1" "%b1" W₁ b₁
    (.reluF (denseF "%W0" "%b0" W₀ b₀ (.operand "%x" x)))))

/-- **MLP forward faithfulness.** The forward graph denotes `mlpForward`. -/
theorem mlpFwdGraph_faithful (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀) :
    den (mlpFwdGraph W₀ b₀ W₁ b₁ W₂ b₂ x) = mlpForward W₀ b₀ W₁ b₁ W₂ b₂ x := by
  simp only [mlpFwdGraph, mlpForward, Function.comp_apply, denseF_faithful, reluF_faithful,
             den_operand]

/-- Whole-MLP **backward** (input-VJP) graph: `dotOut W₀ ∘ select(p₀) ∘
    dotOut W₁ ∘ select(p₁) ∘ dotOut W₂`, `pᵢ` the ReLU pre-activations. -/
def mlpBackGraph (W₀ : Mat e₀ e₁) (W₁ : Mat e₁ e₂) (W₂ : Mat e₂ e₃)
    (p₀ : Vec e₁) (p₁ : Vec e₂) (dy : Vec e₃) : SHlo e₀ :=
  .dotOut "%W0" W₀ (.selectPos "%h0" p₀ (.dotOut "%W1" W₁
    (.selectPos "%h1" p₁ (.dotOut "%W2" W₂ (.operand "%dy" dy)))))

/-- **MLP backward faithfulness (smooth point).** The backward graph denotes
    the proven `mlpHasVJPAt.backward` — the per-op `dot_general`/`select`
    ops assembled into the proven whole-network VJP (cf. `IR.mlp_whole_bridge`). -/
theorem mlpBackGraph_faithful (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀)
    (h0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h1 : ∀ k, dense W₁ b₁ (relu e₁ (dense W₀ b₀ x)) k ≠ 0) (dy : Vec e₃) :
    den (mlpBackGraph W₀ W₁ W₂ (dense W₀ b₀ x)
          (dense W₁ b₁ (relu e₁ (dense W₀ b₀ x))) dy)
      = (mlpHasVJPAt W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1).backward dy := by
  simp only [mlpBackGraph, denStep, denStepApp, mlpHasVJPAt, denseHasVJP, reluHasVJPAt,
             HasVJP.toHasVJPAt, Function.comp_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Chapters 3–4 — the MNIST and CIFAR CNNs
-- ════════════════════════════════════════════════════════════════

/-- Whole MNIST-CNN **forward** graph:
    `dense ∘ relu ∘ dense ∘ relu ∘ dense ∘ maxPool ∘ relu ∘ conv ∘ relu ∘ conv`. -/
def cnnFwdGraph {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (x : Vec (ic*(2*h)*(2*w))) : SHlo nClasses :=
  denseF "%W5" "%b5" W₅ b₅
    (.reluF (denseF "%W4" "%b4" W₄ b₄
      (.reluF (denseF "%W3" "%b3" W₃ b₃
        (.maxPoolF (c := c) (h := h) (w := w)
          (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W2" "%b2" W₂ b₂
            (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W1" "%b1" W₁ b₁
              (.operand "%x" x))))))))))

/-- Whole **CIFAR-CNN forward** graph (Chapter 4): two conv→relu→conv→relu→maxPool
    stages (channels `ic→c1→c1`, then `c1→c2→c2`) then `dense→relu→dense→relu→dense`.
    The Chapter-4 peer of `cnnFwdGraph`. -/
def cifarFwdGraph {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : SHlo nClasses :=
  denseF "%W7" "%b7" W₇ b₇
    (.reluF (denseF "%W6" "%b6" W₆ b₆
      (.reluF (denseF "%W5" "%b5" W₅ b₅
        (.maxPoolF (c := c2) (h := h) (w := w)
          (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W4" "%b4" W₄ b₄
            (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W3" "%b3" W₃ b₃
              (.maxPoolF (c := c1) (h := 2*h) (w := 2*w)
                (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W2" "%b2" W₂ b₂
                  (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W1" "%b1" W₁ b₁
                    (.operand "%x" x)))))))))))))))

/-- Whole **deeper (8-conv) CIFAR-CNN forward** graph: four conv→relu→conv→relu→maxPool
    stages (channels `ic→c1→c1`, `c1→c2→c2`, `c2→c3→c3`, `c3→c4→c4`) then
    `dense→relu→dense→relu→dense`. The 4-stage peer of `cifarFwdGraph`. -/
def cifar8FwdGraph {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : SHlo nClasses :=
  denseF "%Wb" "%bb" Wb bb
    (.reluF (denseF "%Wa" "%ba" Wa ba
      (.reluF (denseF "%W9" "%b9" W₉ b₉
        (.maxPoolF (c := c4) (h := h) (w := w)
          (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W8" "%b8" W₈ b₈
            (.reluF (.flatConvF (h := 2*h) (w := 2*w) "%W7" "%b7" W₇ b₇
              (.maxPoolF (c := c3) (h := 2*h) (w := 2*w)
                (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W6" "%b6" W₆ b₆
                  (.reluF (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W5" "%b5" W₅ b₅
                    (.maxPoolF (c := c2) (h := 2*(2*h)) (w := 2*(2*w))
                      (.reluF (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W4" "%b4" W₄ b₄
                        (.reluF (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W3" "%b3" W₃ b₃
                          (.maxPoolF (c := c1) (h := 2*(2*(2*h))) (w := 2*(2*(2*w)))
                            (.reluF (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W2" "%b2" W₂ b₂
                              (.reluF (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W1" "%b1" W₁ b₁
                                (.operand "%x" x)))))))))))))))))))))))))

/-- Whole **deeper (8-conv) BN-CIFAR forward** graph: each of the eight convs is followed
    by a per-channel `bnPerChannelF` before its ReLU. `epsStr` is the shared ε literal; the
    eight BN layers carry per-channel γ/β inputs `%g{i}`/`%bt{i}`. The BN peer of
    `cifar8FwdGraph`. -/
def cifar8BnFwdGraph {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : SHlo nClasses :=
  denseF "%Wb" "%bb" Wb bb
    (.reluF (denseF "%Wa" "%ba" Wa ba
      (.reluF (denseF "%W9" "%b9" W₉ b₉
        (.maxPoolF (c := c4) (h := h) (w := w)
          (.reluF (.bnPerChannelF (oc := c4) (h := 2*h) (w := 2*w) "%g8" "%bt8" epsStr ε₈ γ₈ β₈
            (.flatConvF (h := 2*h) (w := 2*w) "%W8" "%b8" W₈ b₈
            (.reluF (.bnPerChannelF (oc := c4) (h := 2*h) (w := 2*w) "%g7" "%bt7" epsStr ε₇ γ₇ β₇
              (.flatConvF (h := 2*h) (w := 2*w) "%W7" "%b7" W₇ b₇
              (.maxPoolF (c := c3) (h := 2*h) (w := 2*w)
                (.reluF (.bnPerChannelF (oc := c3) (h := 2*(2*h)) (w := 2*(2*w)) "%g6" "%bt6" epsStr ε₆ γ₆ β₆
                  (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W6" "%b6" W₆ b₆
                  (.reluF (.bnPerChannelF (oc := c3) (h := 2*(2*h)) (w := 2*(2*w)) "%g5" "%bt5" epsStr ε₅ γ₅ β₅
                    (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W5" "%b5" W₅ b₅
                    (.maxPoolF (c := c2) (h := 2*(2*h)) (w := 2*(2*w))
                      (.reluF (.bnPerChannelF (oc := c2) (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%g4" "%bt4" epsStr ε₄ γ₄ β₄
                        (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W4" "%b4" W₄ b₄
                        (.reluF (.bnPerChannelF (oc := c2) (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%g3" "%bt3" epsStr ε₃ γ₃ β₃
                          (.flatConvF (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) "%W3" "%b3" W₃ b₃
                          (.maxPoolF (c := c1) (h := 2*(2*(2*h))) (w := 2*(2*(2*w)))
                            (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%g2" "%bt2" epsStr ε₂ γ₂ β₂
                              (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W2" "%b2" W₂ b₂
                              (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%g1" "%bt1" epsStr ε₁ γ₁ β₁
                                (.flatConvF (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) "%W1" "%b1" W₁ b₁
                                (.operand "%x" x)))))))))))))))))))))))))))))))))

/-- Whole MNIST-CNN **backward** (input-VJP) graph, reversing `cnnFwdGraph`:
    `convBack W₁ ∘ select(a₁) ∘ convBack W₂ ∘ select(a₂) ∘ maxPoolBack ∘
     dotOut W₃ ∘ select(a₃) ∘ dotOut W₄ ∘ select(a₄) ∘ dotOut W₅`, with `aᵢ` the
    ReLU pre-activations and the conv/maxpool saved inputs threaded by name. -/
noncomputable def cnnBackGraph
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses)
    (x : Vec (ic * (2*h) * (2*w))) (dy : Vec nClasses) :
    SHlo (ic * (2*h) * (2*w)) :=
  let z1 := (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x
  let zmp := (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂) z1
  let zd3 := maxPoolFlat c h w zmp
  let zd4 := (relu d1 ∘ dense W₃ b₃) zd3
  .convBack "%W1" W₁ b₁ x
    (.selectPos "%a1" (flatConv (h := 2*h) (w := 2*w) W₁ b₁ x)
      (.convBack "%W2" W₂ b₂ z1
        (.selectPos "%a2" (flatConv (h := 2*h) (w := 2*w) W₂ b₂ z1)
          (.maxPoolBack "%z2" zmp
            (.dotOut "%W3" W₃
              (.selectPos "%a3" (dense W₃ b₃ zd3)
                (.dotOut "%W4" W₄
                  (.selectPos "%a4" (dense W₄ b₄ zd4)
                    (.dotOut "%W5" W₅ (.operand "%dy" dy))))))))))

-- ════════════════════════════════════════════════════════════════
-- § Printers — each writes one committed chapter artifact from its graph
-- ════════════════════════════════════════════════════════════════

/-- `@linear_fwd` rendered **from the verified AST**. -/
def linearFwdModuleV (B d₀ d₁ : Nat) (W : Mat d₀ d₁) (b : Vec d₁) (x : Vec d₀) : String :=
  renderModule "linear_fwd" s!"%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}" B d₁ (fwdGraph W b x)

/-- `@linear_back` rendered **from the verified AST**. -/
def linearBackModuleV (B d₀ d₁ : Nat) (W : Mat d₀ d₁) (dy : Vec d₁) : String :=
  renderModule "linear_back" s!"%dy: {ty [B,d₁]}, %W0: {ty [d₀,d₁]}" B d₀ (backGraph W dy)

/-- The full **`@linear_train_step`** rendered from the verified AST: forward +
    softmax-CE cotangent come from `pretty (lossCotGraph …)` (the `%onehot`
    operand value is `pretty`-irrelevant, so any placeholder renders the same
    text — at runtime `%onehot` is a graph input); the weight grad
    (`dot_general` over the batch axis), bias grad (`reduce`), and the SGD
    `multiply`/`subtract` updates are appended. Returns the two updated params.
    The verified-AST peer of `IRPrint.linearTrainStepModule`. -/
def linearTrainStepModuleV (B d₀ d₁ : Nat) (lr : String)
    (W : Mat d₀ d₁) (b : Vec d₁) (x : Vec d₀) : String :=
  let (body, dy) := (pretty B (lossCotGraph W b x (fun _ => 0))).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @linear_train_step(%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, " ++
  s!"%onehot: {ty [B,d₁]}) -> ({ty [d₀,d₁]}, {ty [d₁]}) " ++ "{\n" ++
  "    // ── forward + softmax-CE cotangent — rendered from the verified AST (lossCotGraph) ──\n" ++
  body ++
  s!"    // dy = {dy} = ⟦lossCotGraph⟧ = ∂CE/∂logits (lossCotGraph_isCEgrad)\n" ++
  "    // ── param grads: dW0 = x⊗dy, db0 = Σ_batch dy (wGrad/bGrad_is*Jacobian) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %dW0 = stablehlo.dot_general %x, {dy}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,d₀]}, {ty [B,d₁]}) -> {ty [d₀,d₁]}\n" ++
  s!"    %db0 = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,d₁]}, tensor<f32>) -> {ty [d₁]}\n" ++
  "    // ── SGD update θ' = θ − lr·∇ (sgdW/sgdB_isCertifiedGradStep) ──\n" ++
  s!"    %lW0 = stablehlo.constant dense<{lr}> : {ty [d₀,d₁]}\n" ++
  s!"    %sW0 = stablehlo.multiply %dW0, %lW0 : {ty [d₀,d₁]}\n" ++
  s!"    %W0n = stablehlo.subtract %W0, %sW0 : {ty [d₀,d₁]}\n" ++
  s!"    %lb0 = stablehlo.constant dense<{lr}> : {ty [d₁]}\n" ++
  s!"    %sb0 = stablehlo.multiply %db0, %lb0 : {ty [d₁]}\n" ++
  s!"    %b0n = stablehlo.subtract %b0, %sb0 : {ty [d₁]}\n" ++
  s!"    return %W0n, %b0n : {ty [d₀,d₁]}, {ty [d₁]}\n" ++
  "  }\n}\n"

/-- `@mlp_fwd` rendered from the verified forward AST `mlpFwdGraph`. -/
def mlpFwdModuleV (B d₀ d₁ d₂ d₃ : Nat)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) : String :=
  renderModule "mlp_fwd"
    s!"%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, %W1: {ty [d₁,d₂]}, %b1: {ty [d₂]}, %W2: {ty [d₂,d₃]}, %b2: {ty [d₃]}"
    B d₃ (mlpFwdGraph W₀ b₀ W₁ b₁ W₂ b₂ x)

/-- `@cnn_fwd` rendered from the verified CNN forward AST `cnnFwdGraph`. -/
def cnnFwdModuleV (B ic c h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) : String :=
  renderModule "cnn_fwd"
    s!"%x: {ty [B,ic*(2*h)*(2*w)]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [c*h*w,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}"
    B nClasses (cnnFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)

/-- `@cifar_fwd` rendered from the verified CIFAR forward AST `cifarFwdGraph`. -/
def cifarFwdModuleV (B ic c1 c2 h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  renderModule "cifar_fwd"
    s!"%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c2*h*w,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}"
    B nClasses (cifarFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)

/-- `@cifar8_fwd` rendered from the verified 8-conv CIFAR forward AST `cifar8FwdGraph`
    (`cifar8FwdGraph_faithful` proves it denotes `cifarCnn8Forward`). The 4-stage peer of
    `cifarFwdModuleV` — the committed `verified_mlir/cifar8_fwd.mlir` is
    `renderModule(provenGraph)`. -/
def cifar8FwdModuleV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  renderModule "cifar8_fwd"
    s!"%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [c4*h*w,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
    B nClasses (cifar8FwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x)

/-- `@cifar8_bn_fwd` rendered from the verified 8-conv per-channel-BN CIFAR forward AST
    `cifar8BnFwdGraph` (`cifar8BnFwdGraph_faithful` proves it denotes `cifarCnnBn8Forward`).
    The BN peer of `cifar8FwdModuleV`. -/
def cifar8BnFwdModuleV (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) : String :=
  renderModule "cifar8_bn_fwd"
    s!"%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [c4*h*w,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}"
    B nClasses (cifar8BnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
      W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x)

end StableHLO
end Proofs
