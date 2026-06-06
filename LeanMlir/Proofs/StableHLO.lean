import LeanMlir.Proofs.IR

/-! # R4 — printer faithfulness, Stage A (Chapter 2: the linear classifier)

The seed of `planning/validated_codegen_book.md`'s `Proofs/Hlo/{Syntax,Denote}`.

`IR.lean` gives the backward/forward IR a denotation in `ℝ` and proves it equals
the Mathlib-`fderiv` math. The remaining trusted link — **R4** — is that the
StableHLO **text** the printer emits means the same function. This file closes
R4 for Chapter 2, *both halves*, over a single typed AST `SHlo`:

* **Semantic half** (`den`, load-bearing): a denotation in StableHLO-spec terms
  (explicit contraction / reduce / divide), and faithfulness theorems
  `den (emit …) = <proven math>` for every piece of the linear train step —
  forward logits, dense input-VJP, softmax-CE cotangent (to the proven
  ∂CE/∂logits), the weight/bias parameter Jacobians, **and the SGD update**
  (`θ' = θ − lr·∇`, now proven rather than trusted).

* **Syntactic half** (`pretty`): the same `SHlo` carries SSA-name annotations
  (denotation-irrelevant — `den` ignores them) so it renders to real StableHLO
  text. The emitted modules — including the **whole `@linear_train_step`** —
  are `pretty (emit g)` (the doc's "Step 0 consolidation": one AST, both
  denotable and renderable).

**All together (the R4 chain for ch 2):**
`render text = pretty (emit g)` (syntactic, by construction);
`den (emit g) = Mathlib fderiv` (semantic, the theorems below).

**Scope / residue.** Per-example semantics (`Vec`/`Mat`): the batch axis is an
outer map, a printer concern (the doc's "D1 shortcut"). `pretty`'s lexical
conformance to the StableHLO spec is the audited/validated residue (the doc's
"4b": cross-checked by `iree-compile` + execution — the verified-rendered train
step trains MNIST to ~92%), not a verified `parse` round-trip ("4a"). Everything
here closes under `[propext, Classical.choice, Quot.sound]` (`tests/AuditAxioms.lean`).
-/

open Finset BigOperators

namespace Proofs
namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § StableHLO-subset AST — denotable AND renderable
-- ════════════════════════════════════════════════════════════════

/-- A StableHLO-subset expression, shape-indexed by result length. Leaves carry
    both a value (for `den`) and an SSA name (for `pretty`); the name is
    denotation-irrelevant. One constructor per emitted op. -/
inductive SHlo : Nat → Type where
  | operand    {n : Nat} (name : String) (v : Vec n)            : SHlo n
  | dotIn      {m n : Nat} (wName : String) (W : Mat m n)       : SHlo m → SHlo n
  | dotOut     {m n : Nat} (wName : String) (W : Mat m n)       : SHlo n → SHlo m
  | addBcast   {n : Nat} (bName : String) (b : Vec n)           : SHlo n → SHlo n
  | expe       {n : Nat}                                        : SHlo n → SHlo n
  | softmaxDiv {n : Nat}                                        : SHlo n → SHlo n
  | sub        {n : Nat}                                        : SHlo n → SHlo n → SHlo n
  -- Chapter 3 (MLP): ReLU forward (`maximum(·,0)`) and its backward mask
  -- (`select(x>0,·,0)`); `xName`/`x` is the saved pre-activation.
  | reluF      {n : Nat}                                        : SHlo n → SHlo n
  | selectPos  {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n

/-- **AST denotation `⟦·⟧ₐ`** — our reading of each StableHLO op's spec, over
    `ℝ`, per-example, in primitive terms — independent of `dense`/`Mat.mulVec`.
    SSA names are ignored. -/
noncomputable def den : {n : Nat} → SHlo n → Vec n
  | _, .operand _ v    => v
  | _, .dotIn _ W e    => fun j => ∑ i, den e i * W i j
  | _, .dotOut _ W e   => fun i => ∑ j, W i j * den e j
  | _, .addBcast _ b e => fun j => den e j + b j
  | _, .expe e         => fun j => Real.exp (den e j)
  | _, .softmaxDiv e   => fun j => den e j / ∑ k, den e k
  | _, .sub a b        => fun j => den a j - den b j
  | _, .reluF e        => fun i => max (den e i) 0
  | _, .selectPos _ x e => fun i => if x i > 0 then den e i else 0

@[simp] theorem den_operand {n : Nat} (s : String) (v : Vec n) :
    den (.operand s v) = v := rfl
@[simp] theorem den_dotIn {m n : Nat} (s : String) (W : Mat m n) (e : SHlo m) :
    den (.dotIn s W e) = fun j => ∑ i, den e i * W i j := rfl
@[simp] theorem den_dotOut {m n : Nat} (s : String) (W : Mat m n) (e : SHlo n) :
    den (.dotOut s W e) = fun i => ∑ j, W i j * den e j := rfl
@[simp] theorem den_addBcast {n : Nat} (s : String) (b : Vec n) (e : SHlo n) :
    den (.addBcast s b e) = fun j => den e j + b j := rfl
@[simp] theorem den_expe {n : Nat} (e : SHlo n) :
    den (.expe e) = fun j => Real.exp (den e j) := rfl
@[simp] theorem den_softmaxDiv {n : Nat} (e : SHlo n) :
    den (.softmaxDiv e) = fun j => den e j / ∑ k, den e k := rfl
@[simp] theorem den_sub {n : Nat} (a b : SHlo n) :
    den (.sub a b) = fun j => den a j - den b j := rfl
@[simp] theorem den_reluF {n : Nat} (e : SHlo n) :
    den (.reluF e) = fun i => max (den e i) 0 := rfl
@[simp] theorem den_selectPos {n : Nat} (s : String) (x : Vec n) (e : SHlo n) :
    den (.selectPos s x e) = fun i => if x i > 0 then den e i else 0 := rfl

-- ════════════════════════════════════════════════════════════════
-- § `emit`: the linear (Chapter-2) train-step graphs
-- ════════════════════════════════════════════════════════════════

variable {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)

/-- Forward logits graph `@linear_fwd`: `broadcast(b) + dot_general(x, W)`. -/
def fwdGraph : SHlo n := .addBcast "%b0" b (.dotIn "%W0" W (.operand "%x" x))

/-- Dense input-VJP graph (`@linear_back`): `dot_general(dy, W)`. -/
def backGraph (dy : Vec n) : SHlo m := .dotOut "%W0" W (.operand "%dy" dy)

/-- Softmax-CE loss-cotangent graph `softmax(logits) − onehot`. The one-hot is
    a parameter (a graph input `%onehot`); `den` reads it, `pretty` ignores it. -/
def lossCotGraph (oh : Vec n) : SHlo n :=
  .sub (.softmaxDiv (.expe (fwdGraph W b x))) (.operand "%onehot" oh)

-- ════════════════════════════════════════════════════════════════
-- § Semantic half: each emitted graph denotes the proven math
-- ════════════════════════════════════════════════════════════════

/-- **Forward faithfulness.** The forward graph denotes `mnistLinear W b`. -/
theorem fwdGraph_faithful : den (fwdGraph W b x) = mnistLinear W b x := by
  funext j; simp only [fwdGraph, den, mnistLinear, dense]

/-- **Dense input-VJP faithfulness.** The backward graph denotes the proven
    dense VJP backward `(dense_has_vjp W b).backward x = Mat.mulVec W`. -/
theorem backGraph_faithful (dy : Vec n) :
    den (backGraph W dy) = (dense_has_vjp W b).backward x dy := by
  funext i; simp only [backGraph, den, dense_has_vjp, Mat.mulVec]

/-- The softmax sub-graph denotes the proven `softmax`. -/
theorem softmaxDiv_expe_faithful (z : Vec n) :
    den (.softmaxDiv (.expe (.operand "%logits" z))) = softmax n z := by
  funext j; simp only [den, softmax]

/-- **Loss-cotangent faithfulness (spec level).** -/
theorem lossCotGraph_faithful (label : Fin n) :
    den (lossCotGraph W b x (oneHot n label)) = IR.emitLossCot n (mnistLinear W b x) label := by
  funext j
  simp only [lossCotGraph, IR.emitLossCot, den, oneHot, softmax, fwdGraph_faithful,
             mnistLinear, dense]

/-- **Loss-cotangent faithfulness (to the proven gradient).** Via
    `IR.lossCot_bridge`: the cotangent graph denotes `∂(crossEntropy)/∂logits`
    at the linear logits. -/
theorem lossCotGraph_isCEgrad (label : Fin n) (j : Fin n) :
    den (lossCotGraph W b x (oneHot n label)) j
      = pdiv (fun (z : Vec n) (_ : Fin 1) => crossEntropy n z label)
             (mnistLinear W b x) j 0 := by
  rw [lossCotGraph_faithful]; exact IR.lossCot_bridge n (mnistLinear W b x) label j

-- ── Parameter gradients (per-example; the batch `dot_general`/`reduce`
--    reduce, per the D1 shortcut, to the outer product / the cotangent). ──

/-- Weight-gradient (per-example): the batch-contracting `dot_general`, i.e.
    the outer product `x ⊗ dy`. -/
def wGrad (x : Vec m) (dy : Vec n) : Mat m n := Mat.outer x dy

/-- Bias-gradient (per-example): the batch `reduce`-add is the cotangent. -/
def bGrad (dy : Vec n) : Vec n := dy

theorem wGrad_faithful (dy : Vec n) :
    wGrad x dy = IR.emitWeightGrad x .cotangent dy := rfl

/-- **Weight-grad faithfulness** to the certified ∂/∂W Jacobian. -/
theorem wGrad_isWeightJacobian (dy : Vec n) (i : Fin m) (j : Fin n) :
    wGrad x dy i j
      = ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k :=
  IR.weight_grad_bridge W b x .cotangent dy i j

theorem bGrad_faithful (dy : Vec n) : bGrad dy = IR.emitBiasGrad (.cotangent) dy := rfl

/-- **Bias-grad faithfulness** to the certified ∂/∂b Jacobian. -/
theorem bGrad_isBiasJacobian (dy : Vec n) (i : Fin n) :
    bGrad dy i = ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j :=
  IR.bias_grad_bridge W b x .cotangent dy i

-- ════════════════════════════════════════════════════════════════
-- § SGD update — proven (not trusted) for plain SGD on the linear net
-- ════════════════════════════════════════════════════════════════

/-- The emitted **weight** SGD update `W − lr·(x⊗dy)`, with `dy` the proven
    softmax-CE cotangent. -/
noncomputable def sgdW (lr : ℝ) (label : Fin n) : Mat m n :=
  fun i j => W i j - lr * wGrad x (den (lossCotGraph W b x (oneHot n label))) i j

/-- The emitted **bias** SGD update `b − lr·dy`. -/
noncomputable def sgdB (lr : ℝ) (label : Fin n) : Vec n :=
  fun j => b j - lr * bGrad (den (lossCotGraph W b x (oneHot n label))) j

/-- **SGD weight-step faithfulness.** The emitted update subtracts `lr` times
    the *certified* ∂/∂W Jacobian contracted with the proven loss cotangent —
    plain-SGD optimizer promoted from trusted to proven. -/
theorem sgdW_descends_certified_grad (lr : ℝ) (label : Fin n) (i : Fin m) (j : Fin n) :
    sgdW W b x lr label i j
      = W i j - lr * ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k
            * den (lossCotGraph W b x (oneHot n label)) k := by
  unfold sgdW
  rw [wGrad_isWeightJacobian W b x (den (lossCotGraph W b x (oneHot n label))) i j]

/-- **SGD bias-step faithfulness.** Likewise for `b`. -/
theorem sgdB_descends_certified_grad (lr : ℝ) (label : Fin n) (j : Fin n) :
    sgdB W b x lr label j
      = b j - lr * ∑ i : Fin n,
          pdiv (fun b' : Vec n => dense W b' x) b j i
            * den (lossCotGraph W b x (oneHot n label)) i := by
  unfold sgdB
  rw [bGrad_isBiasJacobian W b x (den (lossCotGraph W b x (oneHot n label))) j]

-- ════════════════════════════════════════════════════════════════
-- § Chapter 3 — MLP: ReLU + multi-layer composition (semantic)
--
-- The forward adds ReLU (`maximum(·,0)`); the backward chains the proven
-- per-layer VJPs through `select(x>0,·,0)` ReLU masks. ReLU has a kink, so the
-- whole-MLP VJP is *conditional* (`mlp_has_vjp_at`, off the kink) — exactly the
-- regime the codegen's subgradient (`relu'(0)=0`) targets. The parameter grads
-- and SGD update reuse the layer-agnostic `wGrad`/`bGrad`/`sgd*` theorems above.
-- ════════════════════════════════════════════════════════════════

/-- `maximum(a,0)` equals ReLU's pointwise `if a>0 then a else 0`. -/
private theorem max_zero_eq (a : ℝ) : max a 0 = if a > 0 then a else 0 := by
  by_cases h : (0 : ℝ) < a
  · rw [if_pos h, max_eq_left h.le]
  · rw [if_neg h, max_eq_right (not_lt.1 h)]

/-- **ReLU forward faithfulness.** `maximum(·,0)` denotes the proven `relu`. -/
theorem reluF_faithful {k : Nat} (e : SHlo k) : den (.reluF e) = relu k (den e) := by
  funext i; simp only [den, relu]; exact max_zero_eq _

/-- **ReLU backward faithfulness (smooth point).** `select(x>0,·,0)` denotes the
    proven `relu_has_vjp_at` backward — the codegen's `relu'(0)=0` convention. -/
theorem selectPos_faithful {k : Nat} (s : String) (x : Vec k) (hx : ∀ i, x i ≠ 0)
    (e : SHlo k) :
    den (.selectPos s x e) = (relu_has_vjp_at k x hx).backward (den e) := rfl

/-- A dense forward layer graph: `broadcast(bias) + dot_general(·, W)`. -/
def denseF {a c : Nat} (wN bN : String) (W : Mat a c) (bias : Vec c) (e : SHlo a) : SHlo c :=
  .addBcast bN bias (.dotIn wN W e)

theorem denseF_faithful {a c : Nat} (wN bN : String) (W : Mat a c) (bias : Vec c) (e : SHlo a) :
    den (denseF wN bN W bias e) = dense W bias (den e) := by
  funext j; simp only [denseF, den, dense]

variable {e₀ e₁ e₂ e₃ : Nat}

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
    the proven `mlp_has_vjp_at.backward` — the per-op `dot_general`/`select`
    ops assembled into the proven whole-network VJP (cf. `IR.mlp_whole_bridge`). -/
theorem mlpBackGraph_faithful (W₀ : Mat e₀ e₁) (b₀ : Vec e₁) (W₁ : Mat e₁ e₂) (b₁ : Vec e₂)
    (W₂ : Mat e₂ e₃) (b₂ : Vec e₃) (x : Vec e₀)
    (h0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h1 : ∀ k, dense W₁ b₁ (relu e₁ (dense W₀ b₀ x)) k ≠ 0) (dy : Vec e₃) :
    den (mlpBackGraph W₀ W₁ W₂ (dense W₀ b₀ x)
          (dense W₁ b₁ (relu e₁ (dense W₀ b₀ x))) dy)
      = (mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1).backward dy := by
  simp only [mlpBackGraph, den, mlp_has_vjp_at, vjp_comp_at, dense_has_vjp, relu_has_vjp_at,
             HasVJP.toHasVJPAt, Mat.mulVec, id_eq, Function.comp_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Syntactic half: `pretty` renders the AST to real StableHLO text
-- ════════════════════════════════════════════════════════════════

/-- Tensor-type string `tensor<d₀x…xf32>`. -/
def ty (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["f32"]) ++ ">"

/-- Boolean (i1) tensor-type string, for `compare`/`select` masks. -/
def tyI1 (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString ++ ["i1"]) ++ ">"

/-- Fresh SSA name `%v{k}`. -/
def fresh : StateM Nat String := do
  let k ← get; set (k + 1); pure s!"%v{k}"

/-- **`pretty`** — render an `SHlo` graph to a StableHLO op sequence, batch dim
    `B`. Returns `(emitted code, result SSA)`. Op forms mirror `IRPrint.lean`. -/
def pretty (B : Nat) : {k : Nat} → SHlo k → StateM Nat (String × String)
  | _, .operand name _ => pure ("", name)
  | _, .dotIn (m := m) (n := n) wName _ e => do
      let (c, r) ← pretty B e; let o ← fresh
      pure (c ++ s!"    {o} = stablehlo.dot_general {r}, {wName}, contracting_dims = [1] x [0], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [m,n]}) -> {ty [B,n]}\n", o)
  | _, .dotOut (m := m) (n := n) wName _ e => do
      let (c, r) ← pretty B e; let o ← fresh
      pure (c ++ s!"    {o} = stablehlo.dot_general {r}, {wName}, contracting_dims = [1] x [1], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({ty [B,n]}, {ty [m,n]}) -> {ty [B,m]}\n", o)
  | k, .addBcast bName _ e => do
      let (c, r) ← pretty B e; let bb ← fresh; let o ← fresh
      pure (c ++ s!"    {bb} = stablehlo.broadcast_in_dim {bName}, dims = [1] : ({ty [k]}) -> {ty [B,k]}\n" ++
            s!"    {o} = stablehlo.add {r}, {bb} : {ty [B,k]}\n", o)
  | k, .expe e => do
      let (c, r) ← pretty B e; let o ← fresh
      pure (c ++ s!"    {o} = stablehlo.exponential {r} : {ty [B,k]}\n", o)
  | k, .softmaxDiv e => do
      let (c, r) ← pretty B e; let z ← fresh; let s ← fresh; let sb ← fresh; let o ← fresh
      pure (c ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {s} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,k]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0] : ({ty [B]}) -> {ty [B,k]}\n" ++
        s!"    {o} = stablehlo.divide {r}, {sb} : {ty [B,k]}\n", o)
  | k, .sub a b => do
      let (ca, ra) ← pretty B a; let (cb, rb) ← pretty B b; let o ← fresh
      pure (ca ++ cb ++ s!"    {o} = stablehlo.subtract {ra}, {rb} : {ty [B,k]}\n", o)
  | k, .reluF e => do
      let (c, r) ← pretty B e; let z ← fresh; let o ← fresh
      pure (c ++ s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,k]}\n" ++
            s!"    {o} = stablehlo.maximum {r}, {z} : {ty [B,k]}\n", o)
  | k, .selectPos xName _ e => do
      let (c, r) ← pretty B e; let z ← fresh; let msk ← fresh; let o ← fresh
      pure (c ++
        s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,k]}\n" ++
        s!"    {msk} = stablehlo.compare GT, {xName}, {z} : ({ty [B,k]}, {ty [B,k]}) -> {tyI1 [B,k]}\n" ++
        s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 [B,k]}, {ty [B,k]}\n", o)

/-- Wrap a rendered single-result graph as a `func.func` module. -/
def renderModule (name argSig : String) (B retLen : Nat) (g : SHlo retLen) : String :=
  let (body, res) := (pretty B g).run' 0
  "module @m {\n" ++ s!"  func.func @{name}({argSig}) -> {ty [B, retLen]} " ++ "{\n" ++
  body ++ s!"    return {res} : {ty [B, retLen]}\n" ++ "  }\n}\n"

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
  let (body, dy) := (pretty B (lossCotGraph W b x (fun _ => 0))).run' 0
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
  "    // ── SGD update θ' = θ − lr·∇ (sgdW/sgdB_descends_certified_grad) ──\n" ++
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

/-- Full **MLP** SGD train step. The forward layers emit exactly `mlpFwdGraph`'s
    ops (`dot_general`+`add`, `maximum`), saving the pre-activations `%h0,%h1`;
    the backward emits `mlpBackGraph`'s ops (`dot_general`, `compare GT`+`select`
    masks reading `%h0,%h1`); param grads + SGD as in the linear step. Each piece
    is proven faithful above (`mlpFwdGraph_faithful`, `mlpBackGraph_faithful`,
    `reluF_faithful`, `selectPos_faithful`, `wGrad/bGrad_is*Jacobian`,
    `lossCotGraph_isCEgrad`, `sgd*_descends_certified_grad`); the assembly/naming
    is the renderer (validated by `iree-compile` + the GPU run). -/
def mlpTrainStepText (B d₀ d₁ d₂ d₃ : Nat) (lr : String) : String :=
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let reduce (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @mlp_train_step(%x: {ty [B,d₀]}, %W0: {ty [d₀,d₁]}, %b0: {ty [d₁]}, %W1: {ty [d₁,d₂]}, %b1: {ty [d₂]}, %W2: {ty [d₂,d₃]}, %b2: {ty [d₃]}, %onehot: {ty [B,d₃]}) -> ({ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}) " ++ "{\n" ++
  "    // ── forward (mlpFwdGraph): %h0,%h1 pre-acts, %a0,%a1 activations, %logits ──\n" ++
  dense "%h0" "%x" "%W0" "%b0" d₀ d₁ ++ relu "%a0" "%h0" d₁ ++
  dense "%h1" "%a0" "%W1" "%b1" d₁ d₂ ++ relu "%a1" "%h1" d₂ ++
  dense "%logits" "%a1" "%W2" "%b2" d₂ d₃ ++
  "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,d₃]}\n" ++
  "    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B,d₃]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,d₃]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,d₃]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,d₃]}\n" ++
  "    // ── backward (mlpBackGraph): dotOut + select masks reading %h1,%h0 ──\n" ++
  dg "%dx2" "%dy" "%W2" "1" "1" (ty [B,d₃]) (ty [d₂,d₃]) (ty [B,d₂]) ++
  s!"    %bz1 = stablehlo.constant dense<0.0> : {ty [B,d₂]}\n" ++
  s!"    %bm1 = stablehlo.compare GT, %h1, %bz1 : ({ty [B,d₂]}, {ty [B,d₂]}) -> {tyI1 [B,d₂]}\n" ++
  s!"    %dy1 = stablehlo.select %bm1, %dx2, %bz1 : {tyI1 [B,d₂]}, {ty [B,d₂]}\n" ++
  dg "%dx1" "%dy1" "%W1" "1" "1" (ty [B,d₂]) (ty [d₁,d₂]) (ty [B,d₁]) ++
  s!"    %bz0 = stablehlo.constant dense<0.0> : {ty [B,d₁]}\n" ++
  s!"    %bm0 = stablehlo.compare GT, %h0, %bz0 : ({ty [B,d₁]}, {ty [B,d₁]}) -> {tyI1 [B,d₁]}\n" ++
  s!"    %dy0 = stablehlo.select %bm0, %dx1, %bz0 : {tyI1 [B,d₁]}, {ty [B,d₁]}\n" ++
  "    // ── param grads (wGrad/bGrad) ──\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  dg "%dW2" "%a1" "%dy" "0" "0" (ty [B,d₂]) (ty [B,d₃]) (ty [d₂,d₃]) ++ reduce "%db2" "%dy" d₃ ++
  dg "%dW1" "%a0" "%dy1" "0" "0" (ty [B,d₁]) (ty [B,d₂]) (ty [d₁,d₂]) ++ reduce "%db1" "%dy1" d₂ ++
  dg "%dW0" "%x" "%dy0" "0" "0" (ty [B,d₀]) (ty [B,d₁]) (ty [d₀,d₁]) ++ reduce "%db0" "%dy0" d₁ ++
  "    // ── SGD θ' = θ − lr·∇ ──\n" ++
  sgd "%W0" "%dW0" (ty [d₀,d₁]) ++ sgd "%b0" "%db0" (ty [d₁]) ++
  sgd "%W1" "%dW1" (ty [d₁,d₂]) ++ sgd "%b1" "%db1" (ty [d₂]) ++
  sgd "%W2" "%dW2" (ty [d₂,d₃]) ++ sgd "%b2" "%db2" (ty [d₃]) ++
  s!"    return %W0n, %b0n, %W1n, %b1n, %W2n, %b2n : {ty [d₀,d₁]}, {ty [d₁]}, {ty [d₁,d₂]}, {ty [d₂]}, {ty [d₂,d₃]}, {ty [d₃]}\n" ++
  "  }\n}\n"

end StableHLO
end Proofs

-- Emit the verified-renderer modules at the real ch-2 shapes (784→10, B=128).
-- `pretty` ignores operand/param *values* (only names/shapes reach the text),
-- so the constant placeholders below render exactly the text `den` is faithful
-- to. The train-step lr literal is 0.1/128 (mean-loss equiv of the book's 0.1).
#eval IO.FS.writeFile "/tmp/linear_fwd_v.mlir"
  (Proofs.StableHLO.linearFwdModuleV 128 784 10 (fun _ _ => 0) (fun _ => 0) (fun _ => 0))
#eval IO.FS.writeFile "/tmp/linear_back_v.mlir"
  (Proofs.StableHLO.linearBackModuleV 128 784 10 (fun _ _ => 0) (fun _ => 0))
#eval IO.FS.writeFile "/tmp/linear_train_step_v.mlir"
  (Proofs.StableHLO.linearTrainStepModuleV 128 784 10 "0.00078125" (fun _ _ => 0) (fun _ => 0) (fun _ => 0))

-- Committed verified-rendered artifacts (the exact `pretty (emit g)` text the
-- `mnist-linear-verified` trainer compiles + runs through the real Lean/IREE
-- FFI on GPU). Regenerate with `lake env lean LeanMlir/Proofs/StableHLO.lean`.
#eval (do
  IO.FS.createDirAll "verified_mlir"
  IO.FS.writeFile "verified_mlir/linear_fwd.mlir"
    (Proofs.StableHLO.linearFwdModuleV 128 784 10 (fun _ _ => 0) (fun _ => 0) (fun _ => 0))
  IO.FS.writeFile "verified_mlir/linear_train_step.mlir"
    (Proofs.StableHLO.linearTrainStepModuleV 128 784 10 "0.00078125"
       (fun _ _ => 0) (fun _ => 0) (fun _ => 0))
  -- Chapter 3 MLP (784→512→512→10): forward + full SGD train step.
  IO.FS.writeFile "verified_mlir/mlp_fwd.mlir"
    (Proofs.StableHLO.mlpFwdModuleV 128 784 512 512 10
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  IO.FS.writeFile "verified_mlir/mlp_train_step.mlir"
    (Proofs.StableHLO.mlpTrainStepText 128 784 512 512 10 "0.00078125") : IO Unit)
