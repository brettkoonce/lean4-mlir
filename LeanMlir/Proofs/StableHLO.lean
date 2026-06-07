import LeanMlir.Proofs.IR
import LeanMlir.Proofs.CifarCNN
import LeanMlir.Proofs.StridedConv
import LeanMlir.Proofs.PerChannelBN
import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.MobileNetV2
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.EfficientNet

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
  -- Chapter 7 (MobileNetV2): ReLU6 forward (`clamp(·,0,6) = min(max(·,0),6)`) and
  -- its backward mask (`select(0<x<6,·,0)` — the TWO-SIDED kink, smooth iff
  -- `x≠0 ∧ x≠6`). `selectMid`'s `xName`/`x` is the saved pre-activation.
  | relu6F     {n : Nat}                                        : SHlo n → SHlo n
  | selectMid  {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 4 (CNN): flattened conv forward (`stablehlo.convolution`) and
  -- 2×2 max-pool forward (`reduce_window`). Vec-indexed via the proofs'
  -- flattened forms `flatConv`/`maxPoolFlat`.
  | flatConvF  {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)                    : SHlo (ic*h*w) → SHlo (oc*h*w)
  | maxPoolF   {c h w : Nat}                                    : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  -- Conv input-VJP backward (reversed-kernel `stablehlo.convolution`); `v` is
  -- the saved conv input. Conv is linear, so this is a global VJP.
  | convBack   {ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*h*w)) : SHlo (oc*h*w) → SHlo (ic*h*w)
  -- Max-pool backward (`select_and_scatter`, route dy to the window argmax);
  -- `x` is the saved pre-pool input. Conditional (no-ties) like the ReLU kink.
  | maxPoolBack {c h w : Nat} (xName : String) (x : Vec (c*(2*h)*(2*w))) : SHlo (c*h*w) → SHlo (c*(2*h)*(2*w))
  -- Chapter 5 (BatchNorm): per-example normalization over the whole feature
  -- vec (reduce mean/var over axis [1], scalar γ/β). `gName,bName` are the γ,β
  -- scalar SSA inputs, `epsStr` the rendered ε literal; ε,γ,β carry the den.
  | bnF        {n : Nat} (gName bName epsStr : String) (ε γ β : ℝ)   : SHlo n → SHlo n
  -- BN input-VJP — the consolidated O(N) three-term gradient (`bn_grad_input`),
  -- recomputing x̂/istd from the saved BN input `x` (`xName`). Total in `x`;
  -- faithful (= pdiv-Jacobian) under `0 < ε` (`bn_input_grad_correct`).
  | bnBack     {n : Nat} (gName xName epsStr : String) (ε γ : ℝ) (x : Vec n) : SHlo n → SHlo n
  -- Chapter 6 (ResNet): residual add (`stablehlo.add`) and global-average-pool.
  -- `addV` is binary (mirrors `.sub`); the residual skip reuses the block-input
  -- subtree in BOTH operands, so the graph stays a tree. `gapF` reduces the
  -- spatial axes (`reduce add over [2,3]`, ÷h·w), `Vec (c*h*w) → Vec c`.
  | addV       {n : Nat}                                        : SHlo n → SHlo n → SHlo n
  | gapF       {c h w : Nat}                                    : SHlo (c*h*w) → SHlo c
  -- Chapter 6 Milestone B (ResNet-34 downsampling): stride-2 SAME conv forward
  -- (`stablehlo.convolution` with `window_strides=[2,2]`) and its input-VJP
  -- (zero-upsample the cotangent — `lhs_dilation` — then the reversed-kernel
  -- conv). `den` via the proven `flatConvStride2` / `flatConvStride2_has_vjp`.
  | flatConvStridedF {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)              : SHlo (ic*(2*h)*(2*w)) → SHlo (oc*h*w)
  | convStridedBack  {ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*(2*h)*(2*w))) : SHlo (oc*h*w) → SHlo (ic*(2*h)*(2*w))
  -- Chapter 6 Milestone B8 (real-ResNet PER-CHANNEL BatchNorm): normalize each
  -- channel-slice over its h·w spatial cells with its OWN `(γ_c, β_c)`, γ/β : `Vec oc`
  -- (rank-1, `broadcast dims=[1]` — vs `bnF`'s rank-0 scalars). `den` via the proven
  -- `bnPerChannelTensor3` (the Mat-split block-diagonal BN bridged into the `(oc*h)*w`
  -- activation layout) / its renderable backward `bnPerChannelTensor3_grad_input`.
  | bnPerChannelF    {oc h w : Nat} (gName bName epsStr : String) (ε : ℝ) (γ β : Vec oc)
                                                           : SHlo (oc*h*w) → SHlo (oc*h*w)
  | bnPerChannelBack {oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (oc*h*w))                                   : SHlo (oc*h*w) → SHlo (oc*h*w)
  -- Chapter 7 (MobileNetV2): depthwise conv forward (`stablehlo.convolution` with
  -- `feature_group_count = c` and a `[c, 1, kH, kW]` kernel — one filter per channel,
  -- no cross-channel mixing) and its input-VJP (the SAME-pad reversed-kernel depthwise
  -- conv — spatial flip only, since the per-channel groups are 1×1; same
  -- `feature_group_count`). `den` via the proven `depthwiseFlat` / `depthwiseFlat_has_vjp`.
  | depthwiseF    {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*h*w) → SHlo (c*h*w)
  | depthwiseBack {c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*h*w)) : SHlo (c*h*w) → SHlo (c*h*w)
  -- Chapter 7 C3: STRIDE-2 depthwise conv forward (`window_strides=[2,2]`,
  -- `feature_group_count = c`, `[c,1,kH,kW]` kernel — halves spatial, the MNv2
  -- downsampling op) and its input-VJP (zero-upsample the cotangent via
  -- `stablehlo.pad` interior=1 then the reversed-kernel stride-1 depthwise — the
  -- `convStridedBack` shape, per-channel). `den` via the proven `depthwiseStride2Flat`
  -- / `depthwiseStride2Flat_has_vjp` (= decimate ∘ depthwise).
  | depthwiseStridedF    {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c)            : SHlo (c*(2*h)*(2*w)) → SHlo (c*h*w)
  | depthwiseStridedBack {c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*(2*h)*(2*w))) : SHlo (c*h*w) → SHlo (c*(2*h)*(2*w))
  -- Chapter 8 (EfficientNet): swish forward (`x · σ(x)`, σ = `stablehlo.logistic`)
  -- and its input-VJP (`dy · swish'(x)`, closed form `σ(x)·(1 + x·(1−σ(x)))`).
  -- Swish is SMOOTH everywhere (no kink, NO smoothness hyp — unlike relu6); the
  -- VJP is the GLOBAL `swish_has_vjp` (no `_at`). `swishBack`'s `xName`/`x` is the
  -- saved pre-activation. `den` via the proven `swish` / `swish_has_vjp` (LayerNorm.lean).
  | swishF     {n : Nat}                                        : SHlo n → SHlo n
  | swishBack  {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 8 (EfficientNet): sigmoid forward (`σ(x) = stablehlo.logistic`, the SE
  -- gate's output nonlinearity) and its input-VJP (`dy · σ(x)·(1−σ(x))`). Like swish,
  -- SMOOTH everywhere (no kink, NO smoothness hyp — GLOBAL `sigmoid_has_vjp`, not `_at`).
  -- `sigmoidBack`'s `xName`/`x` is the saved pre-activation. `den` via the proven
  -- `sigmoid` / `sigmoid_has_vjp` (EfficientNet.lean).
  | sigmoidF     {n : Nat}                                      : SHlo n → SHlo n
  | sigmoidBack  {n : Nat} (xName : String) (x : Vec n)         : SHlo n → SHlo n

-- Total argmax-routing max-pool backward (the `select_and_scatter` formula),
-- matching `maxPool2_has_vjp_at3.backward` lifted through the flatten bridge.
-- Total in the saved input `xv` (the no-ties proof lives only in `.correct`).
open Classical in
noncomputable def maxPoolBackFlat (c h w : Nat)
    (xv : Vec (c*(2*h)*(2*w))) (dyv : Vec (c*h*w)) : Vec (c*(2*h)*(2*w)) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    if MaxPool2IsArgmax (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w)) q.1 q.2 p.2
    then (Tensor3.unflatten dyv : Tensor3 c h w) q.1 (winRow q.2) (winCol p.2) else 0

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
  | _, .relu6F e       => fun i => min (max (den e i) 0) 6
  | _, .selectMid _ x e => fun i => if 0 < x i ∧ x i < 6 then den e i else 0
  | _, .flatConvF _ _ W b e => flatConv W b (den e)
  | _, .maxPoolF (c := c) (h := h) (w := w) e => maxPoolFlat c h w (den e)
  | _, .convBack _ W b v e => (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward v (den e)
  | _, .maxPoolBack (c := c) (h := h) (w := w) _ x e => maxPoolBackFlat c h w x (den e)
  | _, .bnF (n := n) _ _ _ ε γ β e => bnForward n ε γ β (den e)
  | _, .bnBack (n := n) _ _ _ ε γ x e => bn_grad_input n ε γ x (den e)
  | _, .addV a b       => fun j => den a j + den b j
  | _, .gapF (c := c) (h := h) (w := w) e => globalAvgPoolFlat c h w (den e)
  | _, .flatConvStridedF _ _ W b e => flatConvStride2 W b (den e)
  | _, .convStridedBack _ W b v e => (flatConvStride2_has_vjp W b).backward v (den e)
  | _, .bnPerChannelF (oc := oc) (h := h) (w := w) _ _ _ ε γ β e =>
      bnPerChannelTensor3 oc h w ε γ β (den e)
  | _, .bnPerChannelBack (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      bnPerChannelTensor3_grad_input oc h w ε γ x (den e)
  | _, .depthwiseF _ _ W b e => depthwiseFlat W b (den e)
  | _, .depthwiseBack _ W b v e => (depthwiseFlat_has_vjp W b).backward v (den e)
  | _, .depthwiseStridedF _ _ W b e => depthwiseStride2Flat W b (den e)
  | _, .depthwiseStridedBack _ W b v e => (depthwiseStride2Flat_has_vjp W b).backward v (den e)
  | _, .swishF (n := n) e => swish n (den e)
  | _, .swishBack (n := n) _ x e => (swish_has_vjp n).backward x (den e)
  | _, .sigmoidF (n := n) e => sigmoid n (den e)
  | _, .sigmoidBack (n := n) _ x e => (sigmoid_has_vjp n).backward x (den e)

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
@[simp] theorem den_addV {n : Nat} (a b : SHlo n) :
    den (.addV a b) = fun j => den a j + den b j := rfl
@[simp] theorem den_reluF {n : Nat} (e : SHlo n) :
    den (.reluF e) = fun i => max (den e i) 0 := rfl
@[simp] theorem den_selectPos {n : Nat} (s : String) (x : Vec n) (e : SHlo n) :
    den (.selectPos s x e) = fun i => if x i > 0 then den e i else 0 := rfl
@[simp] theorem den_relu6F {n : Nat} (e : SHlo n) :
    den (.relu6F e) = fun i => min (max (den e i) 0) 6 := rfl
@[simp] theorem den_selectMid {n : Nat} (s : String) (x : Vec n) (e : SHlo n) :
    den (.selectMid s x e) = fun i => if 0 < x i ∧ x i < 6 then den e i else 0 := rfl

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

/-- **ReLU6 forward faithfulness.** `min(max(·,0),6)` denotes the proven `relu6`
    (MobileNetV2.lean). (`rfl` — `relu6` is defined as exactly this clamp.) -/
@[simp] theorem relu6F_faithful {k : Nat} (e : SHlo k) :
    den (.relu6F e) = relu6 k (den e) := rfl

/-- **ReLU6 backward faithfulness (smooth point).** `select(0<x<6,·,0)` denotes the
    proven `relu6_has_vjp_at` backward — the two-sided kink's mask, smooth iff
    `x≠0 ∧ x≠6` (both bounds, unlike ReLU's one-sided `x≠0`). -/
theorem selectMid_faithful {k : Nat} (s : String) (x : Vec k)
    (h_smooth : ∀ i, x i ≠ 0 ∧ x i ≠ 6) (e : SHlo k) :
    den (.selectMid s x e) = (relu6_has_vjp_at k x h_smooth).backward (den e) := rfl

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
-- § Chapter 4 — CNN: conv + maxpool (forward, semantic)
--
-- The conv/maxpool *forward* ops, denoted by the proofs' flattened forms
-- `flatConv`/`maxPoolFlat`. The whole MNIST-CNN forward graph denotes the
-- proven `mnistCnnNoBnForward`. (The backward VJP — conv input-grad via the
-- reversed kernel + maxpool select_and_scatter, = `mnistCnnNoBn_has_vjp_at` —
-- is the next phase.)
-- ════════════════════════════════════════════════════════════════

/-- **Conv forward faithfulness.** The (flattened) `stablehlo.convolution` op
    denotes the proven `flatConv`. -/
theorem flatConvF_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*h*w)) :
    den (.flatConvF wN bN W b e) = flatConv W b (den e) := rfl

/-- **Max-pool forward faithfulness.** The (flattened) `reduce_window(max)` op
    denotes the proven `maxPoolFlat`. -/
theorem maxPoolF_faithful {c h w : Nat} (e : SHlo (c*(2*h)*(2*w))) :
    den (.maxPoolF e) = maxPoolFlat c h w (den e) := rfl

/-- **Conv backward faithfulness.** The reversed-kernel `stablehlo.convolution`
    (transpose+reverse+conv) denotes the proven conv input-VJP — the flattened
    `conv2d_has_vjp3` backward (conv is linear, so this is a global VJP). -/
theorem convBack_faithful {ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*h*w)) (e : SHlo (oc*h*w)) :
    den (.convBack wN W b v e)
      = (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward v (den e) := rfl

/-- **Max-pool backward faithfulness (smooth point).** The emitted
    `select_and_scatter` graph denotes the proven `maxPoolFlat_has_vjp_at`
    backward — routing the cotangent to each window's argmax (the codegen's
    no-ties convention), under the MaxPool smoothness hypothesis. -/
theorem maxPoolBack_faithful {c h w : Nat} (xN : String) (x : Vec (c*(2*h)*(2*w)))
    (h_smooth : MaxPool2Smooth (Tensor3.unflatten x : Tensor3 c (2*h) (2*w)))
    (e : SHlo (c*h*w)) :
    den (.maxPoolBack xN x e)
      = (maxPoolFlat_has_vjp_at (Tensor3.unflatten x) h_smooth).backward (den e) := by
  funext idx
  simp only [den, maxPoolBackFlat, maxPoolFlat_has_vjp_at, hasVJPAt3_to_hasVJPAt,
             maxPool2_has_vjp_at3]

/-- **BN forward faithfulness.** The per-example reduce/normalize/affine graph
    (γ·(x−μ)·istd + β, μ/var over the feature axis) denotes the proven
    `bnForward` (BatchNorm.lean). -/
@[simp] theorem bnF_faithful {n : Nat} (gN bN es : String) (ε γ β : ℝ) (e : SHlo n) :
    den (.bnF gN bN es ε γ β e) = bnForward n ε γ β (den e) := rfl

/-- **Residual-add faithfulness** (= `den_addV`). The binary `stablehlo.add`
    denotes pointwise vector addition — the fan-in of a residual/skip
    connection. (`rfl`, so kept out of the axiom audit.) -/
theorem addV_faithful {n : Nat} (a b : SHlo n) :
    den (.addV a b) = fun j => den a j + den b j := rfl

/-- **Global-average-pool faithfulness.** The reduce-over-spatial / ÷h·w graph
    denotes the proven `globalAvgPoolFlat` (CNN.lean). -/
@[simp] theorem gapF_faithful {c h w : Nat} (e : SHlo (c*h*w)) :
    den (.gapF e) = globalAvgPoolFlat c h w (den e) := rfl

/-- **Strided-conv forward faithfulness.** The `window_strides=[2,2]`
    `stablehlo.convolution` denotes the proven `flatConvStride2`
    (= decimate ∘ stride-1 conv, StridedConv.lean). -/
@[simp] theorem flatConvStridedF_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*(2*h)*(2*w))) :
    den (.flatConvStridedF wN bN W b e) = flatConvStride2 W b (den e) := rfl

/-- **Strided-conv input-VJP faithfulness.** The zero-upsample (`lhs_dilation`)
    + reversed-kernel conv denotes the proven `flatConvStride2_has_vjp` backward. -/
theorem convStridedBack_faithful {ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*(2*h)*(2*w))) (e : SHlo (oc*h*w)) :
    den (.convStridedBack wN W b v e) = (flatConvStride2_has_vjp W b).backward v (den e) := rfl

/-- **BN backward faithfulness.** The consolidated three-term graph denotes the
    proven BN input-VJP — equal to the `pdiv`-contracted Jacobian of `bnForward`
    (`bn_input_grad_correct`), under `0 < ε`. β-independent (a constant shift
    does not enter the Jacobian). -/
theorem bnBack_faithful {n : Nat} (gN xN es : String) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (e : SHlo n) (i : Fin n) :
    den (.bnBack gN xN es ε γ x e) i
      = ∑ j : Fin n, pdiv (bnForward n ε γ β) x i j * den e j := by
  show bn_grad_input n ε γ x (den e) i = _
  exact bn_input_grad_correct n ε γ β hε x (den e) i

/-- **Per-channel BN forward faithfulness.** The 4-D reshape + per-channel
    reduce/normalize (μ/var over the spatial axes `[2,3]`, rank-1 γ/β `dims=[1]`)
    denotes the proven `bnPerChannelTensor3` (PerChannelBN.lean). (`rfl`, so kept
    out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem bnPerChannelF_faithful {oc h w : Nat} (gN bN es : String) (ε : ℝ)
    (γ β : Vec oc) (e : SHlo (oc*h*w)) :
    den (.bnPerChannelF gN bN es ε γ β e) = bnPerChannelTensor3 oc h w ε γ β (den e) := rfl

/-- **Per-channel BN backward faithfulness.** The block-diagonal three-term graph
    (per-channel, reducing over the spatial axes) denotes the proven per-channel BN
    input-VJP — equal to the `pdiv`-contracted (block-diagonal) Jacobian of
    `bnPerChannelTensor3` (`bnPerChannelTensor3_grad_input_correct`), under `0 < ε`. -/
theorem bnPerChannelBack_faithful {oc h w : Nat} (gN xN es : String) (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec oc) (x : Vec (oc*h*w)) (e : SHlo (oc*h*w)) (i : Fin (oc*h*w)) :
    den (.bnPerChannelBack gN xN es ε γ x e) i
      = ∑ j : Fin (oc*h*w), pdiv (bnPerChannelTensor3 oc h w ε γ β) x i j * den e j := by
  show bnPerChannelTensor3_grad_input oc h w ε γ x (den e) i = _
  exact bnPerChannelTensor3_grad_input_correct oc h w ε hε γ β x (den e) i

/-- **Depthwise-conv forward faithfulness.** The `feature_group_count = c`
    `stablehlo.convolution` (with a `[c,1,kH,kW]` kernel, one filter per channel)
    denotes the proven `depthwiseFlat` (= flatten ∘ depthwiseConv2d ∘ unflatten,
    Depthwise.lean). (`rfl`, so kept out of the axiom audit — `roundtrip` covers it
    structurally.) -/
@[simp] theorem depthwiseF_faithful {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (c*h*w)) :
    den (.depthwiseF wN bN W b e) = depthwiseFlat W b (den e) := rfl

/-- **Depthwise-conv input-VJP faithfulness.** The reversed-kernel depthwise
    `stablehlo.convolution` (reverse the per-channel filters over the spatial axes
    `[2,3]`; the channel groups are 1×1 so no o↔i transpose, same
    `feature_group_count = c`) denotes the proven `depthwiseFlat_has_vjp` backward
    (depthwise is linear, so this is a global VJP). -/
theorem depthwiseBack_faithful {c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*h*w)) (e : SHlo (c*h*w)) :
    den (.depthwiseBack wN W b v e) = (depthwiseFlat_has_vjp W b).backward v (den e) := rfl

/-- **Strided-depthwise forward faithfulness.** The `window_strides=[2,2]`,
    `feature_group_count = c` `stablehlo.convolution` denotes the proven
    `depthwiseStride2Flat` (= decimate ∘ stride-1 depthwise, Depthwise.lean). -/
@[simp] theorem depthwiseStridedF_faithful {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (c*(2*h)*(2*w))) :
    den (.depthwiseStridedF wN bN W b e) = depthwiseStride2Flat W b (den e) := rfl

/-- **Strided-depthwise input-VJP faithfulness.** The zero-upsample (`stablehlo.pad`
    interior=1) + reversed-kernel stride-1 depthwise denotes the proven
    `depthwiseStride2Flat_has_vjp` backward. -/
theorem depthwiseStridedBack_faithful {c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (v : Vec (c*(2*h)*(2*w))) (e : SHlo (c*h*w)) :
    den (.depthwiseStridedBack wN W b v e) = (depthwiseStride2Flat_has_vjp W b).backward v (den e) := rfl

/-- **Swish forward faithfulness.** The `multiply(x, logistic(x))` graph denotes
    the proven `swish` (= `x · σ(x)`, LayerNorm.lean). Smooth everywhere; no kink,
    no smoothness hypothesis. (`rfl`, so kept out of the axiom audit — `roundtrip`
    covers it structurally.) -/
@[simp] theorem swishF_faithful {n : Nat} (e : SHlo n) :
    den (.swishF e) = swish n (den e) := rfl

/-- **Swish input-VJP faithfulness.** The closed-form `dy ⊙ σ(x)·(1 + x·(1−σ(x)))`
    graph (recomputing σ from the saved pre-activation `x`) denotes the proven GLOBAL
    `swish_has_vjp` backward (`dy ⊙ swishScalarDeriv x`; swish is smooth everywhere, so
    this is a global VJP — no smoothness hypothesis). -/
theorem swishBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.swishBack xN x e) = (swish_has_vjp n).backward x (den e) := rfl

/-- **Sigmoid forward faithfulness.** The `stablehlo.logistic(x)` graph denotes the
    proven `sigmoid` (= σ(x), EfficientNet.lean) — the SE gate's output nonlinearity.
    Smooth everywhere. (`rfl`, so kept out of the axiom audit — `roundtrip` covers it.) -/
@[simp] theorem sigmoidF_faithful {n : Nat} (e : SHlo n) :
    den (.sigmoidF e) = sigmoid n (den e) := rfl

/-- **Sigmoid input-VJP faithfulness.** The closed-form `dy ⊙ σ(x)·(1−σ(x))` graph
    (recomputing σ from the saved pre-activation `x`) denotes the proven GLOBAL
    `sigmoid_has_vjp` backward (`dy ⊙ sigmoidScalarDeriv x`; sigmoid is smooth
    everywhere, so this is a global VJP — no smoothness hypothesis). -/
theorem sigmoidBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.sigmoidBack xN x e) = (sigmoid_has_vjp n).backward x (den e) := rfl

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

/-- **CNN forward faithfulness.** The forward graph denotes the proven
    `mnistCnnNoBnForward`. -/
theorem cnnFwdGraph_faithful {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) :
    den (cnnFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)
      = mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x := by
  simp only [cnnFwdGraph, mnistCnnNoBnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- Whole **CIFAR-CNN forward** graph (Chapter 5): two conv→relu→conv→relu→maxPool
    stages (channels `ic→c1→c1`, then `c1→c2→c2`) then `dense→relu→dense→relu→dense`.
    The Chapter-5 peer of `cnnFwdGraph`. -/
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

/-- **CIFAR-CNN forward faithfulness.** The forward graph denotes the proven
    `cifarCnnForward`. -/
theorem cifarFwdGraph_faithful {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) :
    den (cifarFwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)
      = cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x := by
  simp only [cifarFwdGraph, cifarCnnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- Whole **BN-CIFAR forward** graph (Chapter 5, BatchNorm variant): each conv is
    followed by a per-example `bnF` before its ReLU. `epsStr` is the shared ε
    literal; the four BN layers carry scalar γ/β inputs `%g{i}`/`%bt{i}`. -/
def cifarBnFwdGraph {ic c1 c2 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ γ₁ β₁ : ℝ)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ γ₂ β₂ : ℝ)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ γ₃ β₃ : ℝ)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ γ₄ β₄ : ℝ)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : SHlo nClasses :=
  denseF "%W7" "%b7" W₇ b₇
    (.reluF (denseF "%W6" "%b6" W₆ b₆
      (.reluF (denseF "%W5" "%b5" W₅ b₅
        (.maxPoolF (c := c2) (h := h) (w := w)
          (.reluF (.bnF "%g4" "%bt4" epsStr ε₄ γ₄ β₄
            (.flatConvF (h := 2*h) (w := 2*w) "%W4" "%b4" W₄ b₄
            (.reluF (.bnF "%g3" "%bt3" epsStr ε₃ γ₃ β₃
              (.flatConvF (h := 2*h) (w := 2*w) "%W3" "%b3" W₃ b₃
              (.maxPoolF (c := c1) (h := 2*h) (w := 2*w)
                (.reluF (.bnF "%g2" "%bt2" epsStr ε₂ γ₂ β₂
                  (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W2" "%b2" W₂ b₂
                  (.reluF (.bnF "%g1" "%bt1" epsStr ε₁ γ₁ β₁
                    (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W1" "%b1" W₁ b₁
                    (.operand "%x" x)))))))))))))))))))

/-- **BN-CIFAR forward faithfulness.** The forward graph denotes the proven
    `cifarCnnBnForward`. -/
theorem cifarBnFwdGraph_faithful {ic c1 c2 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ γ₁ β₁ : ℝ)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ γ₂ β₂ : ℝ)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ γ₃ β₃ : ℝ)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ γ₄ β₄ : ℝ)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) :
    den (cifarBnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ W₆ b₆ W₇ b₇ x)
      = cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ W₆ b₆ W₇ b₇ x := by
  simp only [cifarBnFwdGraph, cifarCnnBnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful,
             bnF_faithful, den_operand]

/-- Whole **ResNet-style forward** graph (Chapter 6): the structure the proven
    whole-net VJP `cnn_has_vjp_at` already covers —
    `dense ∘ GAP ∘ rblkP ∘ rblk ∘ maxPool ∘ cbr(stem)`. The stem is `convBnRelu`
    (SAME conv on the `2h×2w` input), one maxpool to `h×w`, an identity basic
    block (`rblk`: `relu(F(y)+y)`), a projection basic block (`rblkP`:
    `relu(proj(y)+F(y))`, `c→oc`), global-average-pool, then dense. Each block's
    skip reuses the block-input **subtree** in BOTH `addV` operands, so the graph
    stays a tree (the §7 "tree-safe via operand leaves" trick, generalized to a
    computed input). `epsStr` is the shared ε literal; each BN carries scalar γ/β
    SSA inputs (`%g*`/`%bt*`). The Chapter-6 peer of `cifarBnFwdGraph`. -/
def resnetFwdGraph
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip : ℝ)
    (Wd : Mat oc nClasses) (bd : Vec nClasses)
    (x : Vec (ic*(2*h)*(2*w))) : SHlo nClasses :=
  -- stem (convBnRelu on the 2h×2w input) → maxpool to h×w
  let pooled : SHlo (c*h*w) :=
    .maxPoolF (c := c) (h := h) (w := w)
      (.reluF (.bnF "%gs" "%bts" epsStr εs γs βs
        (.flatConvF (h := 2*h) (w := 2*w) "%Ws" "%bs" Ws bs (.operand "%x" x))))
  -- identity basic block: relu(F(pooled) + pooled),  F = bn∘conv ∘ relu∘bn∘conv
  let rblkOut : SHlo (c*h*w) :=
    .reluF (.addV
      (.bnF "%g2" "%bt2" epsStr f₂ h₂ i₂
        (.flatConvF (h := h) (w := w) "%W2" "%b2" W₂ b₂
          (.reluF (.bnF "%g1" "%bt1" epsStr f₁ h₁ i₁
            (.flatConvF (h := h) (w := w) "%W1" "%b1" W₁ b₁ pooled)))))
      pooled)
  -- projection basic block: relu(proj(rblkOut) + F'(rblkOut)),  c→oc
  let rblkPOut : SHlo (oc*h*w) :=
    .reluF (.addV
      (.bnF "%gp" "%btp" epsStr fp hp ip
        (.flatConvF (h := h) (w := w) "%Wp" "%bp" Wp bp rblkOut))
      (.bnF "%g2p" "%bt2p" epsStr e₂ g₂ bb₂
        (.flatConvF (h := h) (w := w) "%W2p" "%b2p" W₂' b₂'
          (.reluF (.bnF "%g1p" "%bt1p" epsStr e₁ g₁ bb₁
            (.flatConvF (h := h) (w := w) "%W1p" "%b1p" W₁' b₁' rblkOut))))))
  denseF "%Wd" "%bd" Wd bd (.gapF (c := oc) (h := h) (w := w) rblkPOut)

/-- **ResNet-style forward faithfulness.** The forward graph denotes the proven
    `cnnForward` — the net whose whole-network VJP is `cnn_has_vjp_at` (discharged
    unconditionally by `CnnConcrete.cnnConcrete_has_vjp_correct`). The residual
    `addV`s denote the `+` of `residual`/`residualProj` (`biPath`); each skip's
    duplicated subtree denotes the same block-input value, so `den` reads it
    twice and the fan-in is exact. -/
theorem resnetFwdGraph_faithful
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip : ℝ)
    (Wd : Mat oc nClasses) (bd : Vec nClasses)
    (x : Vec (ic*(2*h)*(2*w))) :
    den (resnetFwdGraph epsStr Ws bs εs γs βs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂
          W₁' b₁' W₂' b₂' Wp bp f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip Wd bd x)
      = cnnForward Ws bs εs γs βs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂
          W₁' b₁' W₂' b₂' Wp bp f₁ h₁ i₁ f₂ h₂ i₂ fp hp ip Wd bd x := by
  -- LHS: collapse the graph denotation to its explicit nested form.
  simp only [resnetFwdGraph, denseF_faithful, gapF_faithful, reluF_faithful,
             bnF_faithful, flatConvF_faithful, maxPoolF_faithful, den_addV, den_operand]
  -- RHS: unfold the abbreviations (incl. `biPath`, which `simp` can't unfold below
  -- its arity), then peel the `∘`s. Both sides land on the same `+`-nested form.
  unfold cnnForward cbr rblk rblkP residual residualProj biPath
  simp only [Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § Chapter 4 — CNN: whole-chain backward (A2c, the MLP-analog of
--   `mlpBackGraph_faithful`). The full backward graph denotes the proven
--   conditional whole-network VJP `mnistCnnNoBn_has_vjp_at.backward`.
-- ════════════════════════════════════════════════════════════════

/-- Pointwise-VJP backwards are unique: `.correct` pins `backward` to the
    `pdiv`-contracted Jacobian, so any two `HasVJPAt f x` agree on `backward`.
    Lets us swap the maxpool's `flatten∘unflatten` transport (built into
    `mnistCnnNoBn_has_vjp_at`) for the cast-free witness below. -/
theorem hasVJPAt_backward_det {m n : Nat} {f : Vec m → Vec n} {x : Vec m}
    (v v' : HasVJPAt f x) (dy : Vec n) : v.backward dy = v'.backward dy := by
  funext i; rw [v.correct, v'.correct]

/-- Max-pool VJP at a *raw* flattened point (no `flatten ∘ unflatten` index), so
    it composes without a transport cast; `backward` is `maxPoolBackFlat`. The
    `correct` field reuses `maxPoolFlat_has_vjp_at.correct`, aligning the point
    via `Tensor3.flatten_unflatten`. -/
noncomputable def maxPoolFlat_has_vjp_at' {c h w : Nat} (v : Vec (c*(2*h)*(2*w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    HasVJPAt (maxPoolFlat c h w) v where
  backward := maxPoolBackFlat c h w v
  correct := fun dy i => by
    have hbk : maxPoolBackFlat c h w v dy i
                = (maxPoolFlat_has_vjp_at (Tensor3.unflatten v) hs).backward dy i := by
      simp only [maxPoolFlat_has_vjp_at, hasVJPAt3_to_hasVJPAt, maxPool2_has_vjp_at3, maxPoolBackFlat]
    rw [hbk, (maxPoolFlat_has_vjp_at (Tensor3.unflatten v) hs).correct dy i,
        Tensor3.flatten_unflatten]

@[simp] theorem maxPoolFlat_has_vjp_at'_backward {c h w : Nat} (v : Vec (c*(2*h)*(2*w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    (maxPoolFlat_has_vjp_at' v hs).backward = maxPoolBackFlat c h w v := rfl

/-- Whole MNIST-CNN **backward** (input-VJP) graph, reversing `cnnFwdGraph`:
    `convBack W₁ ∘ select(a₁) ∘ convBack W₂ ∘ select(a₂) ∘ maxPoolBack ∘
     dotOut W₃ ∘ select(a₃) ∘ dotOut W₄ ∘ select(a₄) ∘ dotOut W₅`, with `aᵢ` the
    ReLU pre-activations and the conv/maxpool saved inputs threaded as in §4. -/
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

-- **CNN backward faithfulness (smooth point) — A2c.** The whole-chain backward
-- graph denotes the proven conditional whole-network VJP
-- `mnistCnnNoBn_has_vjp_at.backward` (the Chapter-4 peer of
-- `mlpBackGraph_faithful`). The per-op `convBack`/`selectPos`/`dotOut` ops
-- assemble through `vjp_comp_at`; the one `maxPoolBack` matches via VJP
-- uniqueness (`hasVJPAt_backward_det`) — sidestepping the `flatten∘unflatten`
-- transport in `mnistCnnNoBn_has_vjp_at`'s maxpool step.
set_option maxHeartbeats 2000000 in
theorem cnnBackGraph_faithful
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    (h1 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₁ b₁ x k ≠ 0)
    (h2 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₂ b₂
            ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x) k ≠ 0)
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)
            : Tensor3 c (2*h) (2*w)))
    (h3 : ∀ k, dense W₃ b₃ (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)) k ≠ 0)
    (h4 : ∀ k, dense W₄ b₄ ((relu d1 ∘ dense W₃ b₃) (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x))) k ≠ 0)
    (dy : Vec nClasses) :
    den (cnnBackGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ x dy)
      = (mnistCnnNoBn_has_vjp_at W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
          hc hh hw x h1 h2 h_mp h3 h4).backward dy := by
  simp only [cnnBackGraph, den, mnistCnnNoBn_has_vjp_at, convRelu_has_vjp_at,
    denseRelu_has_vjp_at, vjp_comp_at, dense_has_vjp, relu_has_vjp_at,
    hasVJP3_to_hasVJP, HasVJP.toHasVJPAt, Mat.mulVec, id_eq, Function.comp_apply]
  rw [hasVJPAt_backward_det _ (maxPoolFlat_has_vjp_at'
        ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
          ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x)) h_mp)]
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

-- ── Renderable skeleton + postorder tokenization (one form, shared with the
--    parser in StableHLOParse.lean) ──

/-- The renderable skeleton of an `SHlo` graph: opcodes + shapes + leaf SSA
    names, with `ℝ` operand values and the shape index erased — exactly what
    reaches the emitted text. -/
inductive Raw where
  | operand    (name : String) (n : Nat)  : Raw
  | dotIn      (w : String) (m n : Nat)    : Raw → Raw
  | dotOut     (w : String) (m n : Nat)    : Raw → Raw
  | addBcast   (b : String) (n : Nat)      : Raw → Raw
  | expe       (n : Nat)                   : Raw → Raw
  | softmaxDiv (n : Nat)                   : Raw → Raw
  | sub        (n : Nat)                   : Raw → Raw → Raw
  | reluF      (n : Nat)                   : Raw → Raw
  | selectPos  (x : String) (n : Nat)      : Raw → Raw
  | relu6F     (n : Nat)                   : Raw → Raw
  | selectMid  (x : String) (n : Nat)      : Raw → Raw
  | flatConvF  (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | maxPoolF   (c h w : Nat)               : Raw → Raw
  | convBack   (w : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | maxPoolBack (x : String) (c h w : Nat) : Raw → Raw
  | bnF        (g b eps : String) (n : Nat) : Raw → Raw
  | bnBack     (g x eps : String) (n : Nat) : Raw → Raw
  | addV       (n : Nat)                   : Raw → Raw → Raw
  | gapF       (c h w : Nat)               : Raw → Raw
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | bnPerChannelF    (g b eps : String) (oc h w : Nat) : Raw → Raw
  | bnPerChannelBack (g x eps : String) (oc h w : Nat) : Raw → Raw
  | depthwiseF    (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseBack (w : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedF    (w b : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedBack (w : String) (c h w' kH kW : Nat) : Raw → Raw
  | swishF     (n : Nat)                   : Raw → Raw
  | swishBack  (x : String) (n : Nat)      : Raw → Raw
  | sigmoidF   (n : Nat)                   : Raw → Raw
  | sigmoidBack (x : String) (n : Nat)     : Raw → Raw
deriving DecidableEq, Repr, Inhabited

/-- Erase an `SHlo` graph to its renderable skeleton (drops `ℝ` values + shape
    index; keeps op structure, shapes, leaf names). -/
def skel : {k : Nat} → SHlo k → Raw
  | k, .operand name _        => .operand name k
  | k, .dotIn (m := m) w _ e  => .dotIn w m k (skel e)
  | k, .dotOut (n := n) w _ e => .dotOut w k n (skel e)
  | k, .addBcast b _ e        => .addBcast b k (skel e)
  | k, .expe e                => .expe k (skel e)
  | k, .softmaxDiv e          => .softmaxDiv k (skel e)
  | k, .sub a b               => .sub k (skel a) (skel b)
  | k, .reluF e               => .reluF k (skel e)
  | k, .selectPos x _ e       => .selectPos x k (skel e)
  | k, .relu6F e              => .relu6F k (skel e)
  | k, .selectMid x _ e       => .selectMid x k (skel e)
  | _, .flatConvF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvF wN bN ic oc h w kH kW (skel e)
  | _, .maxPoolF (c := c) (h := h) (w := w) e => .maxPoolF c h w (skel e)
  | _, .convBack (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .convBack wN ic oc h w kH kW (skel e)
  | _, .maxPoolBack (c := c) (h := h) (w := w) xN _ e => .maxPoolBack xN c h w (skel e)
  | k, .bnF gN bN es _ _ _ e => .bnF gN bN es k (skel e)
  | k, .bnBack gN xN es _ _ _ e => .bnBack gN xN es k (skel e)
  | k, .addV a b              => .addV k (skel a) (skel b)
  | _, .gapF (c := c) (h := h) (w := w) e => .gapF c h w (skel e)
  | _, .flatConvStridedF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStridedF wN bN ic oc h w kH kW (skel e)
  | _, .convStridedBack (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .convStridedBack wN ic oc h w kH kW (skel e)
  | _, .bnPerChannelF (oc := oc) (h := h) (w := w) gN bN es _ _ _ e =>
      .bnPerChannelF gN bN es oc h w (skel e)
  | _, .bnPerChannelBack (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .bnPerChannelBack gN xN es oc h w (skel e)
  | _, .depthwiseF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseF wN bN c h w kH kW (skel e)
  | _, .depthwiseBack (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .depthwiseBack wN c h w kH kW (skel e)
  | _, .depthwiseStridedF (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .depthwiseStridedF wN bN c h w kH kW (skel e)
  | _, .depthwiseStridedBack (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .depthwiseStridedBack wN c h w kH kW (skel e)
  | k, .swishF e             => .swishF k (skel e)
  | k, .swishBack x _ e      => .swishBack x k (skel e)
  | k, .sigmoidF e           => .sigmoidF k (skel e)
  | k, .sigmoidBack x _ e    => .sigmoidBack x k (skel e)

/-- One serialized token: an opcode with shapes/names; operands are positional. -/
inductive Tok where
  | operand    (name : String) (n : Nat)  : Tok
  | dotIn      (w : String) (m n : Nat)    : Tok
  | dotOut     (w : String) (m n : Nat)    : Tok
  | addBcast   (b : String) (n : Nat)      : Tok
  | expe       (n : Nat)                   : Tok
  | softmaxDiv (n : Nat)                   : Tok
  | sub        (n : Nat)                   : Tok
  | reluF      (n : Nat)                   : Tok
  | selectPos  (x : String) (n : Nat)      : Tok
  | relu6F     (n : Nat)                   : Tok
  | selectMid  (x : String) (n : Nat)      : Tok
  | flatConvF  (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | maxPoolF   (c h w : Nat)               : Tok
  | convBack   (w : String) (ic oc h w' kH kW : Nat) : Tok
  | maxPoolBack (x : String) (c h w : Nat) : Tok
  | bnF        (g b eps : String) (n : Nat) : Tok
  | bnBack     (g x eps : String) (n : Nat) : Tok
  | addV       (n : Nat)                   : Tok
  | gapF       (c h w : Nat)               : Tok
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Tok
  | bnPerChannelF    (g b eps : String) (oc h w : Nat) : Tok
  | bnPerChannelBack (g x eps : String) (oc h w : Nat) : Tok
  | depthwiseF    (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseBack (w : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedF    (w b : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedBack (w : String) (c h w' kH kW : Nat) : Tok
  | swishF     (n : Nat)                   : Tok
  | swishBack  (x : String) (n : Nat)      : Tok
  | sigmoidF   (n : Nat)                   : Tok
  | sigmoidBack (x : String) (n : Nat)     : Tok
deriving DecidableEq, Repr

/-- Postorder serialization: children, then the node's opcode token. -/
def toToks : Raw → List Tok
  | .operand nm n    => [.operand nm n]
  | .dotIn w m n e   => toToks e ++ [.dotIn w m n]
  | .dotOut w m n e  => toToks e ++ [.dotOut w m n]
  | .addBcast b n e  => toToks e ++ [.addBcast b n]
  | .expe n e        => toToks e ++ [.expe n]
  | .softmaxDiv n e  => toToks e ++ [.softmaxDiv n]
  | .sub n a b       => toToks a ++ toToks b ++ [.sub n]
  | .reluF n e       => toToks e ++ [.reluF n]
  | .selectPos x n e => toToks e ++ [.selectPos x n]
  | .relu6F n e      => toToks e ++ [.relu6F n]
  | .selectMid x n e => toToks e ++ [.selectMid x n]
  | .flatConvF w b ic oc h w' kH kW e => toToks e ++ [.flatConvF w b ic oc h w' kH kW]
  | .maxPoolF c h w e => toToks e ++ [.maxPoolF c h w]
  | .convBack w ic oc h w' kH kW e => toToks e ++ [.convBack w ic oc h w' kH kW]
  | .maxPoolBack x c h w e => toToks e ++ [.maxPoolBack x c h w]
  | .bnF g b eps n e => toToks e ++ [.bnF g b eps n]
  | .bnBack g x eps n e => toToks e ++ [.bnBack g x eps n]
  | .addV n a b      => toToks a ++ toToks b ++ [.addV n]
  | .gapF c h w e    => toToks e ++ [.gapF c h w]
  | .flatConvStridedF w b ic oc h w' kH kW e => toToks e ++ [.flatConvStridedF w b ic oc h w' kH kW]
  | .convStridedBack w ic oc h w' kH kW e => toToks e ++ [.convStridedBack w ic oc h w' kH kW]
  | .bnPerChannelF g b eps oc h w e => toToks e ++ [.bnPerChannelF g b eps oc h w]
  | .bnPerChannelBack g x eps oc h w e => toToks e ++ [.bnPerChannelBack g x eps oc h w]
  | .depthwiseF w b c h w' kH kW e => toToks e ++ [.depthwiseF w b c h w' kH kW]
  | .depthwiseBack w c h w' kH kW e => toToks e ++ [.depthwiseBack w c h w' kH kW]
  | .depthwiseStridedF w b c h w' kH kW e => toToks e ++ [.depthwiseStridedF w b c h w' kH kW]
  | .depthwiseStridedBack w c h w' kH kW e => toToks e ++ [.depthwiseStridedBack w c h w' kH kW]
  | .swishF n e      => toToks e ++ [.swishF n]
  | .swishBack x n e => toToks e ++ [.swishBack x n]
  | .sigmoidF n e    => toToks e ++ [.sigmoidF n]
  | .sigmoidBack x n e => toToks e ++ [.sigmoidBack x n]

/-- Render one token: pop its operands' result-names off the stack, emit its
    StableHLO line(s), push its fresh result name. The per-op StableHLO *syntax*
    here is the audited lexical boundary (validated by `iree-compile` + GPU run);
    the *structure* it consumes is the proven-faithful token stream. -/
def emitTok (B : Nat) : Tok → List String → StateM Nat (String × List String)
  | .operand nm _, st => pure ("", nm :: st)
  | .dotIn w m n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.dot_general {r}, {w}, contracting_dims = [1] x [0], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [m,n]}) -> {ty [B,n]}\n", o :: st)
  | .dotOut w m n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.dot_general {r}, {w}, contracting_dims = [1] x [1], " ++
            s!"precision = [DEFAULT, DEFAULT] : ({ty [B,n]}, {ty [m,n]}) -> {ty [B,m]}\n", o :: st)
  | .addBcast b n, r :: st => do
      let bb ← fresh; let o ← fresh
      pure (s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [n]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.add {r}, {bb} : {ty [B,n]}\n", o :: st)
  | .expe n, r :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.exponential {r} : {ty [B,n]}\n", o :: st)
  | .softmaxDiv n, r :: st => do
      let z ← fresh; let s ← fresh; let sb ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {s} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.divide {r}, {sb} : {ty [B,n]}\n", o :: st)
  | .sub n, b :: a :: st => do
      let o ← fresh
      pure (s!"    {o} = stablehlo.subtract {a}, {b} : {ty [B,n]}\n", o :: st)
  | .reluF n, r :: st => do
      let z ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.maximum {r}, {z} : {ty [B,n]}\n", o :: st)
  | .selectPos x n, r :: st => do
      let z ← fresh; let msk ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
        s!"    {msk} = stablehlo.compare GT, {x}, {z} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
        s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 [B,n]}, {ty [B,n]}\n", o :: st)
  | .relu6F n, r :: st => do
      -- ReLU6 forward: clamp to [0,6] as `min(max(x,0),6)` (matches `relu6`'s def).
      let z ← fresh; let six ← fresh; let mx ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
            s!"    {six} = stablehlo.constant dense<6.0> : {ty [B,n]}\n" ++
            s!"    {mx} = stablehlo.maximum {r}, {z} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.minimum {mx}, {six} : {ty [B,n]}\n", o :: st)
  | .selectMid x n, r :: st => do
      -- ReLU6 backward mask: route dy where `0 < x < 6`, else 0 (the two-sided kink).
      let z ← fresh; let six ← fresh; let g0 ← fresh; let l6 ← fresh; let msk ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : {ty [B,n]}\n" ++
        s!"    {six} = stablehlo.constant dense<6.0> : {ty [B,n]}\n" ++
        s!"    {g0} = stablehlo.compare GT, {x}, {z} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
        s!"    {l6} = stablehlo.compare LT, {x}, {six} : ({ty [B,n]}, {ty [B,n]}) -> {tyI1 [B,n]}\n" ++
        s!"    {msk} = stablehlo.and {g0}, {l6} : {tyI1 [B,n]}\n" ++
        s!"    {o} = stablehlo.select {msk}, {r}, {z} : {tyI1 [B,n]}, {ty [B,n]}\n", o :: st)
  | .flatConvF w b ic oc h w' kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*h*w']}) -> {ty [B,ic,h,w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,h,w']}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .maxPoolF c h w, r :: st => do
      let xn ← fresh; let ninf ← fresh; let p ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {ninf} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
        s!"    {p} = \"stablehlo.reduce_window\"({xn}, {ninf}) (" ++ "{\n" ++
        "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
        "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
        "        stablehlo.return %pm : tensor<f32>\n" ++
        "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
        s!" : ({ty [B,c,2*h,2*w]}, tensor<f32>) -> {ty [B,c,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {p} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
  | .convBack w ic oc h w' kH kW, r :: st => do
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let wt ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w']}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {wt} = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {wr} = stablehlo.reverse {wt}, dims = [2, 3] : {ty [ic,oc,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({dn}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,oc,h,w']}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,h,w']}) -> {ty [B, ic*h*w']}\n", o :: st)
  | .maxPoolBack xN c h w, r :: st => do
      let xr ← fresh; let dr ← fresh; let z ← fresh; let scn ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {scn} = \"stablehlo.select_and_scatter\"({xr}, {dr}, {z}) (" ++ "{\n" ++
        "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
        "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
        "        stablehlo.return %sge : tensor<i1>\n" ++
        "    }, " ++ "{\n" ++
        "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
        "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
        "        stablehlo.return %ss : tensor<f32>\n" ++
        "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
        s!" : ({ty [B,c,2*h,2*w]}, {ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {o} = stablehlo.reshape {scn} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
  | .bnF gN bN epsStr n, r :: st => do
      -- per-example BatchNorm forward `γ·(x−μ)·istd + β` (reduce μ/var over [1])
      let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {r}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.add {gx}, {bb} : {ty [B,n]}\n", o :: st)
  | .bnBack gN xN epsStr n, r :: st => do
      -- BN input-VJP: recompute x̂/istd from saved input {xN}, then the
      -- consolidated three-term `(istd/N)·(N·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`, dx̂ = γ·dy.
      let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xN} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xN}, {mu} : {ty [B,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,n]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {r} : {ty [B,n]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,n]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [1] : ({ty [B,n]}, tensor<f32>) -> {ty [B]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0] : ({ty [B]}) -> {ty [B,n]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,n]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,n]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,n]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,n]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,n]}\n" ++
        s!"    {o} = stablehlo.multiply {sN}, {i2} : {ty [B,n]}\n", o :: st)
  | .addV n, b :: a :: st => do
      -- residual fan-in: dy of the two operands summed (`F(x) + skip`)
      let o ← fresh
      pure (s!"    {o} = stablehlo.add {a}, {b} : {ty [B,n]}\n", o :: st)
  | .gapF c h w, r :: st => do
      -- global average pool: reshape to [B,c,h,w], reduce-add over the spatial
      -- axes [2,3], divide by h·w. Denotes `globalAvgPoolFlat` (mean over H×W).
      let xn ← fresh; let z ← fresh; let sm ← fresh; let nf ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {sm} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
        s!"    {o} = stablehlo.divide {sm}, {nf} : {ty [B,c]}\n", o :: st)
  | .flatConvStridedF w b ic oc h w' kH kW, r :: st => do
      -- stride-2 SAME conv: reshape, convolution with window_strides=[2,2], +bias
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w')]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*h,2*w']}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
  | .convStridedBack w ic oc h w' kH kW, r :: st => do
      -- stride-2 conv input-VJP: zero-upsample dy (pad with interior=1, high=1) to
      -- the 2h×2w grid, then the reversed-kernel stride-1 conv (= decimate.back ▸ conv.back)
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let z ← fresh; let up ← fresh; let wt ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w']}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {up} = stablehlo.pad {dn}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w']}, tensor<f32>) -> {ty [B,oc,2*h,2*w']}\n" ++
        s!"    {wt} = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {wr} = stablehlo.reverse {wt}, dims = [2, 3] : {ty [ic,oc,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({up}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,oc,2*h,2*w']}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,2*h,2*w']}) -> {ty [B, ic*(2*h)*(2*w')]}\n", o :: st)
  | .bnPerChannelF gN bN epsStr oc h w, r :: st => do
      -- PER-CHANNEL BatchNorm forward: reshape to [B,oc,h,w], reduce μ/var over the
      -- spatial axes [2,3] (per channel), normalize, then γ·x̂+β with rank-1 γ/β
      -- (broadcast dims=[1]). Mirrors `bnF` but 4-D + per-channel.
      let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,oc,h,w]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  | .bnPerChannelBack gN xN epsStr oc h w, r :: st => do
      -- PER-CHANNEL BN input-VJP: recompute x̂/istd per channel from saved input {xN},
      -- then the block-diagonal three-term `(istd/m)·(m·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`,
      -- dx̂ = γ·dy, with all Σ reductions over the spatial axes [2,3] (m = h·w).
      let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,oc,h,w]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,oc,h,w]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,oc,h,w]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,oc,h,w]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,oc,h,w]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,oc,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {o0} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
  | .depthwiseF w b c h w' kH kW, r :: st => do
      -- depthwise conv forward: reshape to [B,c,h,w'], grouped `stablehlo.convolution`
      -- (feature_group_count = c, [c,1,kH,kW] kernel — one filter per channel, no
      -- cross-channel mixing), SAME pad, + per-channel bias, reshape back.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,h,w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseBack w c h w' kH kW, r :: st => do
      -- depthwise conv input-VJP: reshape dy, reverse the per-channel filters over the
      -- spatial axes [2,3] (the channel groups are 1×1, so no o↔i transpose), then the
      -- reversed-kernel SAME-pad depthwise conv (feature_group_count = c).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {wr} = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({dn}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,h,w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedF w b c h w' kH kW, r :: st => do
      -- stride-2 depthwise conv: reshape, grouped convolution with window_strides=[2,2]
      -- (feature_group_count = c, [c,1,kH,kW] kernel), SAME pad, + bias. Halves spatial.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w')]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,c,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .depthwiseStridedBack w c h w' kH kW, r :: st => do
      -- stride-2 depthwise input-VJP: zero-upsample dy (pad interior/high=1) back to
      -- 2h×2w', reverse the per-channel filters over [2,3] (no transpose, 1×1 groups),
      -- then the reversed-kernel stride-1 depthwise conv (feature_group_count = c).
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let dn ← fresh; let z ← fresh; let up ← fresh; let wr ← fresh; let dx ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {up} = stablehlo.pad {dn}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w']}, tensor<f32>) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {wr} = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
        s!"    {dx} = stablehlo.convolution({up}, {wr})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
        s!" : ({ty [B,c,2*h,2*w']}, {ty [c,1,kH,kW]}) -> {ty [B,c,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w']}) -> {ty [B, c*(2*h)*(2*w')]}\n", o :: st)
  | .swishF n, r :: st => do
      -- swish forward: y = x · σ(x), σ = logistic (smooth everywhere, no kink/mask).
      let s ← fresh; let o ← fresh
      pure (s!"    {s} = stablehlo.logistic {r} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {s} : {ty [B,n]}\n", o :: st)
  | .swishBack x n, r :: st => do
      -- swish input-VJP: dy ⊙ σ(x)·(1 + x·(1−σ(x))), recomputing σ from the saved
      -- pre-activation {x} (matches `swishScalarDeriv`'s closed form, IRPrint `swishB`).
      let s ← fresh; let one ← fresh; let om ← fresh; let xom ← fresh
      let inr ← fresh; let sp ← fresh; let o ← fresh
      pure (s!"    {s} = stablehlo.logistic {x} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {om} = stablehlo.subtract {one}, {s} : {ty [B,n]}\n" ++
            s!"    {xom} = stablehlo.multiply {x}, {om} : {ty [B,n]}\n" ++
            s!"    {inr} = stablehlo.add {one}, {xom} : {ty [B,n]}\n" ++
            s!"    {sp} = stablehlo.multiply {s}, {inr} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {sp} : {ty [B,n]}\n", o :: st)
  | .sigmoidF n, r :: st => do
      -- sigmoid forward: σ(x) = logistic(x) (smooth, the SE gate's output nonlinearity).
      let o ← fresh
      pure (s!"    {o} = stablehlo.logistic {r} : {ty [B,n]}\n", o :: st)
  | .sigmoidBack x n, r :: st => do
      -- sigmoid input-VJP: dy ⊙ σ(x)·(1−σ(x)), recomputing σ from the saved
      -- pre-activation {x} (matches `sigmoidScalarDeriv`'s closed form, IRPrint `sigmoidBackM`).
      let s ← fresh; let one ← fresh; let om ← fresh; let sp ← fresh; let o ← fresh
      pure (s!"    {s} = stablehlo.logistic {x} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {om} = stablehlo.subtract {one}, {s} : {ty [B,n]}\n" ++
            s!"    {sp} = stablehlo.multiply {s}, {om} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {sp} : {ty [B,n]}\n", o :: st)
  | _, st => pure ("    // MALFORMED token stream\n", st)

/-- Fold a token stream to accumulated `(code, result-name-stack)`. -/
def serializeToks (B : Nat) : List Tok → (String × List String) → StateM Nat (String × List String)
  | [], acc           => pure acc
  | t :: ts, (code, st) => do
      let (c, st') ← emitTok B t st
      serializeToks B ts (code ++ c, st')

/-- **`pretty`** — render an `SHlo` graph to StableHLO, now defined as
    `serialize ∘ toToks ∘ skel`: tokenize the graph (postorder), then print the
    tokens. The emitter shares ONE structured form with the parser, so the
    round-trip `parse (toToks (skel a)) = skel a` (StableHLOParse.lean) is about
    the very tokens this prints — the printer can't structurally drift. -/
def pretty (B : Nat) {k : Nat} (g : SHlo k) : StateM Nat (String × String) := do
  let (code, st) ← serializeToks B (toToks (skel g)) ("", [])
  match st with
  | [r] => pure (code, r)
  | _   => pure (code, "%MALFORMED")

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

/-- Full **CNN** SGD train step (`@cnn_train_step`), the ch4 peer of
    `mlpTrainStepText`. Architecture (= `mnistCnnNoBnForward`):
    `conv W₁ → relu → conv W₂ → relu → maxpool → flatten → dense W₃ → relu →
     dense W₄ → relu → dense W₅`. Each mathematical op is a rendering of a
    proof-backed piece:
    * forward conv/maxpool/dense/relu — `flatConvF_faithful`, `maxPoolF_faithful`,
      `denseF_faithful`, `reluF_faithful` (and `cnnFwdGraph_faithful` for the whole);
    * loss cotangent `%dy = softmax(logits) − onehot` — `lossCotGraph_isCEgrad`;
    * backward dense (`dot_general`, contract output axis) + relu masks
      (`compare GT`+`select`) — `mlpBackGraph_faithful`/`selectPos_faithful`;
    * maxpool backward (`select_and_scatter`, GE/add, route dy to the window
      argmax) — `maxPoolBack_faithful`; conv input-VJP (transpose+reverse+conv)
      — `convBack_faithful`;
    * dense W/b grads (`dot_general` over batch / `reduce`) — `wGrad/bGrad`;
    * conv weight grad — the **transpose trick** (`conv2d_weight_grad_has_vjp`):
      the SAME `stablehlo.convolution` with the batch axis as the contraction
      feature; rendered here, validated by the GPU run (a `convWGrad_faithful`
      theorem is optional polish, see §B2 of the handoff);
    * SGD `θ' = θ − lr·∇` — `sgd*_descends_certified_grad`.
    The op text mirrors the GPU-validated emitter (`emitTok`) byte-for-byte for
    conv/maxpool/convBack/select_and_scatter; assembly + SSA naming is the
    renderer. `lr = 0.1/B` (grads sum over the batch). -/
def cnnTrainStepText (B ic c H W kH kW d1 nClasses : Nat) (lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2; let flat := c * H2 * W2
  -- dense dot_general (explicit contraction dims), as in mlpTrainStepText
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let relu4 (o h : String) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,c,H,W]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,c,H,W]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  -- relu-backward masks (`select(pre>0, dy, 0)`), 2-D and 4-D forms
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let selMask4 (o pre dgrad : String) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,c,H,W]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,c,H,W]}, {ty [B,c,H,W]}) -> {tyI1 [B,c,H,W]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,c,H,W]}, {ty [B,c,H,W]}\n"
  -- conv forward (SAME pad, stride 1) + bias bcast over channel dim 1
  let convFwd (o lhs w bnm : String) (oc icc : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,H,W]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,H,W]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,H,W]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,H,W]}\n"
  -- conv input-VJP: transpose[1,0,2,3] + reverse[2,3] + convolution (= emitTok convBack)
  let convBack (o dh w : String) (icc oc : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,H,W]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,H,W]}\n"
  -- conv weight grad (transpose trick): dW[o,i,·] = Σ_{b,y,x} x[b,i,·]·dh[b,o,·];
  -- realized as a convolution with the batch axis as the contraction feature.
  let convWGrad (o inp grad : String) (icc oc : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,H,W]}) -> {ty [icc,B,H,W]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,H,W]}) -> {ty [oc,B,H,W]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,H,W]}, {ty [oc,B,H,W]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,H,W]}, tensor<f32>) -> {ty [oc]}\n"
  -- maxpool forward (`reduce_window` max) and backward (`select_and_scatter`)
  let maxpoolFwd (o a : String) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,c,H,W]}, tensor<f32>) -> {ty [B,c,H2,W2]}\n"
  let scatter (o src dgrad : String) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,c,H,W]}, {ty [B,c,H2,W2]}, tensor<f32>) -> {ty [B,c,H,W]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cnn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c,ic,kH,kW]}, %b1: {ty [c]}, %W2: {ty [c,c,kH,kW]}, %b2: {ty [c]}, %W3: {ty [flat,d1]}, %b3: {ty [d1]}, %W4: {ty [d1,d1]}, %b4: {ty [d1]}, %W5: {ty [d1,nClasses]}, %b5: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: conv→relu→conv→relu→maxpool→flatten→dense→relu→dense→relu→dense ──\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c ic ++ relu4 "%ac1" "%hc1" ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c c ++ relu4 "%ac2" "%hc2" ++
  maxpoolFwd "%pool" "%ac2" ++
  s!"    %flat = stablehlo.reshape %pool : ({ty [B,c,H2,W2]}) -> {ty [B,flat]}\n" ++
  dense "%h3" "%flat" "%W3" "%b3" flat d1 ++ relu "%a3" "%h3" d1 ++
  dense "%h4" "%a3" "%W4" "%b4" d1 d1 ++ relu "%a4" "%h4" d1 ++
  dense "%logits" "%a4" "%W5" "%b5" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut) + relu masks → reshape → select_and_scatter → convBack ──\n" ++
  dg "%dx5" "%dy" "%W5" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dy4" "%h4" "%dx5" d1 ++
  dg "%dx4" "%dy4" "%W4" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy3" "%h3" "%dx4" d1 ++
  dg "%dx3" "%dy3" "%W3" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  s!"    %dpool = stablehlo.reshape %dx3 : ({ty [B,flat]}) -> {ty [B,c,H2,W2]}\n" ++
  scatter "%dac2" "%ac2" "%dpool" ++
  selMask4 "%dhc2" "%hc2" "%dac2" ++
  convBack "%dac1" "%dhc2" "%W2" c c ++
  selMask4 "%dhc1" "%hc1" "%dac1" ++
  "    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dW5" "%a4" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db5" "%dy" nClasses ++
  dg "%dW4" "%a3" "%dy4" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db4" "%dy4" d1 ++
  dg "%dW3" "%flat" "%dy3" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db3" "%dy3" d1 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c c ++ convBiasGrad "%db2" "%dhc2" c ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c ++ convBiasGrad "%db1" "%dhc1" c ++
  "    // ── SGD θ' = θ − lr·∇ (all 10 params) ──\n" ++
  sgd "%W1" "%dW1" (ty [c,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c]) ++
  sgd "%W2" "%dW2" (ty [c,c,kH,kW]) ++ sgd "%b2" "%db2" (ty [c]) ++
  sgd "%W3" "%dW3" (ty [flat,d1]) ++ sgd "%b3" "%db3" (ty [d1]) ++
  sgd "%W4" "%dW4" (ty [d1,d1]) ++ sgd "%b4" "%db4" (ty [d1]) ++
  sgd "%W5" "%dW5" (ty [d1,nClasses]) ++ sgd "%b5" "%db5" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n : {ty [c,ic,kH,kW]}, {ty [c]}, {ty [c,c,kH,kW]}, {ty [c]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- Full **CIFAR CNN** SGD train step (`@cifar_train_step`), the Chapter-5 peer of
    `cnnTrainStepText`. Architecture (= `cifarCnnForward`):
    `conv 3→32 → relu → conv 32→32 → relu → maxpool → conv 32→64 → relu →
     conv 64→64 → relu → maxpool → flatten → dense 4096→512 → relu →
     dense 512→512 → relu → dense 512→10` + softmax-CE. Two conv→conv→pool
    stages at two spatial sizes (`H×W` then `H/2×W/2`), with channel changes.

    Every mathematical op is the SAME proof-backed render as `cnnTrainStepText`,
    just instantiated at more layers / two spatial scales — forward
    conv/maxpool/dense/relu (`cifarFwdGraph_faithful`); loss cotangent
    (`lossCotGraph_isCEgrad`); backward dense (`dot_general`) + relu masks
    (`selectPos_faithful`); maxpool backward (`select_and_scatter`,
    `maxPoolBack_faithful`); conv input-VJP (transpose+reverse+conv,
    `convBack_faithful`); dense W/b grads; conv weight grad (transpose trick);
    SGD `θ' = θ − lr·∇`. The per-op text mirrors the GPU-validated `emitTok`
    byte-for-byte; assembly + SSA naming is the renderer (validated by
    `iree-compile` + the GPU run). `lr = 0.1/B`. -/
def cifarTrainStepText (B ic c1 c2 H W kH kW d1 nClasses : Nat) (lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2          -- stage-2 spatial (16)
  let Hp := H2 / 2; let Wp := W2 / 2        -- final pooled (8)
  let flat := c2 * Hp * Wp
  -- dense dot_general (explicit contraction dims), as in cnnTrainStepText
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let relu4 (o h : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,C,Hh,Ww]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let selMask4 (o pre dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}) -> {tyI1 [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}\n"
  -- conv forward (SAME pad, stride 1) + bias bcast over channel dim 1
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  -- conv input-VJP: transpose[1,0,2,3] + reverse[2,3] + convolution (= emitTok convBack)
  let convBack (o dh w : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  -- conv weight grad (transpose trick): dW[o,i,·] = Σ_{b,y,x} x[b,i,·]·dh[b,o,·]
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  -- maxpool forward (`reduce_window` max) and backward (`select_and_scatter`)
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→relu)×2→pool →(conv→relu)×2→pool →flatten→(dense→relu)×2→dense ──\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ relu4 "%ac1" "%hc1" c1 H W ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ relu4 "%ac2" "%hc2" c1 H W ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ relu4 "%ac3" "%hc3" c2 H2 W2 ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ relu4 "%ac4" "%hc4" c2 H2 W2 ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  s!"    %flat = stablehlo.reshape %pool2 : ({ty [B,c2,Hp,Wp]}) -> {ty [B,flat]}\n" ++
  dense "%h5" "%flat" "%W5" "%b5" flat d1 ++ relu2 "%a5" "%h5" d1 ++
  dense "%h6" "%a5" "%W6" "%b6" d1 d1 ++ relu2 "%a6" "%h6" d1 ++
  dense "%logits" "%a6" "%W7" "%b7" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot (lossCotGraph_isCEgrad) ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut)+relu masks → scatter → convBack, twice through ──\n" ++
  dg "%dx7" "%dy" "%W7" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dy6" "%h6" "%dx7" d1 ++
  dg "%dx6" "%dy6" "%W6" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy5" "%h5" "%dx6" d1 ++
  dg "%dx5" "%dy5" "%W5" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  s!"    %dpool2 = stablehlo.reshape %dx5 : ({ty [B,flat]}) -> {ty [B,c2,Hp,Wp]}\n" ++
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++
  selMask4 "%dhc4" "%hc4" "%dac4" c2 H2 W2 ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++
  selMask4 "%dhc3" "%hc3" "%dac3" c2 H2 W2 ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++
  selMask4 "%dhc2" "%hc2" "%dac2" c1 H W ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++
  selMask4 "%dhc1" "%hc1" "%dac1" c1 H W ++
  "    // ── param grads: dense W/b (dot_general/reduce); conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dW7" "%a6" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db7" "%dy" nClasses ++
  dg "%dW6" "%a5" "%dy6" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db6" "%dy6" d1 ++
  dg "%dW5" "%flat" "%dy5" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db5" "%dy5" d1 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 14 params) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [flat,d1]) ++ sgd "%b5" "%db5" (ty [d1]) ++
  sgd "%W6" "%dW6" (ty [d1,d1]) ++ sgd "%b6" "%db6" (ty [d1]) ++
  sgd "%W7" "%dW7" (ty [d1,nClasses]) ++ sgd "%b7" "%db7" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- `@cifar_bn_fwd` rendered from the verified BN-CIFAR forward AST. γ/β are
    scalar `tensor<f32>` inputs (`%g{i}`/`%bt{i}`); `epsStr` the ε literal. -/
def cifarBnFwdModuleV (B ic c1 c2 h w d1 nClasses kH kW : Nat) (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ γ₁ β₁ : ℝ)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ γ₂ β₂ : ℝ)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ γ₃ β₃ : ℝ)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ γ₄ β₄ : ℝ)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  renderModule "cifar_bn_fwd"
    s!"%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: tensor<f32>, %bt1: tensor<f32>, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: tensor<f32>, %bt2: tensor<f32>, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: tensor<f32>, %bt3: tensor<f32>, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: tensor<f32>, %bt4: tensor<f32>, %W5: {ty [c2*h*w,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}"
    B nClasses (cifarBnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
      W₅ b₅ W₆ b₆ W₇ b₇ x)

/-- Full **BN-CIFAR** SGD train step (`@cifar_bn_train_step`). The Chapter-5
    BatchNorm peer of `cifarTrainStepText`: each conv→relu block becomes
    conv→BN→relu. The per-example BN forward (`bnFwd` = `renderLN`: reduce μ/var
    over the feature axis, normalize, scalar-affine — denotes `bnForward`), its
    consolidated three-term input-VJP (`bnBack` = `renderLNBack` — the proven
    `bn_grad_input`, `bnBack_faithful`), and the scalar param grads
    `dγ = Σ dy·x̂`, `dβ = Σ dy` are inserted. BN runs on the flattened
    `[B, oc·H·W]` per-example feature vec (reshape around the 4-D conv). 22
    params (4×{W,b,γ,β} + 3×{W,b}). The whole-net backward is
    `cifarCnnBn_has_vjp_at`. `lr = 0.1/B`. -/
def cifarBnTrainStepText (B ic c1 c2 H W kH kW d1 nClasses : Nat) (epsStr lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2
  let Hp := H2 / 2; let Wp := W2 / 2
  let flat := c2 * Hp * Wp
  let M1 := c1 * H * W            -- stage-1 BN feature size
  let M2 := c2 * H2 * W2          -- stage-2 BN feature size
  let dg (o a w cA cB tA tB tO : String) : String :=
    s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let reduce0 (o dyk : String) (nn : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"
  let selMask2 (o pre dgrad : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
    s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let convBack (o dh w : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,kH,kW]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,kH,kW]}) -> {ty [B,icc,Hh,Ww]}\n"
  let convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
    s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
    s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
    s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,kH,kW]}\n" ++
    s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,kH,kW]}) -> {ty [oc,icc,kH,kW]}\n"
  let convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
    s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
    s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, " ++ "{\n" ++
    "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"
  -- BN forward (per-example, reduce over feature axis [1]); saves {o}_xhat,_istd,_nf.
  let bnFwd (o x g bt : String) (Mn : Nat) : String :=
    s!"    {o}_nf = stablehlo.constant dense<{Mn}.0> : {ty [B,Mn]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,Mn]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,Mn]}, tensor<f32>) -> {ty [B]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0] : ({ty [B]}) -> {ty [B,Mn]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,Mn]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {x}, {o}_mu : {ty [B,Mn]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,Mn]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,Mn]}, tensor<f32>) -> {ty [B]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0] : ({ty [B]}) -> {ty [B,Mn]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,Mn]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,Mn]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,Mn]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,Mn]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {ty [B,Mn]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [] : (tensor<f32>) -> {ty [B,Mn]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,Mn]}\n" ++
    s!"    {o} = stablehlo.add {o}_gx, {o}_bb : {ty [B,Mn]}\n"
  -- BN input-VJP (the consolidated three-term form); reuses {bn}_xhat/_istd/_nf.
  let bnBack (o bn g dyf : String) (Mn : Nat) : String :=
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {ty [B,Mn]}\n" ++
    s!"    {o}_dxh = stablehlo.multiply {o}_gb, {dyf} : {ty [B,Mn]}\n" ++
    s!"    {o}_sdxr = stablehlo.reduce({o}_dxh init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,Mn]}, tensor<f32>) -> {ty [B]}\n" ++
    s!"    {o}_sdx = stablehlo.broadcast_in_dim {o}_sdxr, dims = [0] : ({ty [B]}) -> {ty [B,Mn]}\n" ++
    s!"    {o}_xd = stablehlo.multiply {bn}_xhat, {o}_dxh : {ty [B,Mn]}\n" ++
    s!"    {o}_sxdr = stablehlo.reduce({o}_xd init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,Mn]}, tensor<f32>) -> {ty [B]}\n" ++
    s!"    {o}_sxd = stablehlo.broadcast_in_dim {o}_sxdr, dims = [0] : ({ty [B]}) -> {ty [B,Mn]}\n" ++
    s!"    {o}_t1 = stablehlo.multiply {o}_dxh, {bn}_nf : {ty [B,Mn]}\n" ++
    s!"    {o}_i1 = stablehlo.subtract {o}_t1, {o}_sdx : {ty [B,Mn]}\n" ++
    s!"    {o}_xs = stablehlo.multiply {bn}_xhat, {o}_sxd : {ty [B,Mn]}\n" ++
    s!"    {o}_i2 = stablehlo.subtract {o}_i1, {o}_xs : {ty [B,Mn]}\n" ++
    s!"    {o}_s = stablehlo.divide {bn}_istd, {bn}_nf : {ty [B,Mn]}\n" ++
    s!"    {o} = stablehlo.multiply {o}_s, {o}_i2 : {ty [B,Mn]}\n"
  -- BN scalar param grads dγ = Σ dy·x̂, dβ = Σ dy (over batch+feature → scalar).
  let bnParamGrad (dgr dbe bn dyf : String) (Mn : Nat) : String :=
    s!"    {dgr}_p = stablehlo.multiply {dyf}, {bn}_xhat : {ty [B,Mn]}\n" ++
    s!"    {dgr} = stablehlo.reduce({dgr}_p init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,Mn]}, tensor<f32>) -> tensor<f32>\n" ++
    s!"    {dbe} = stablehlo.reduce({dyf} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,Mn]}, tensor<f32>) -> tensor<f32>\n"
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar_bn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: tensor<f32>, %bt1: tensor<f32>, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: tensor<f32>, %bt2: tensor<f32>, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: tensor<f32>, %bt3: tensor<f32>, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: tensor<f32>, %bt4: tensor<f32>, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, tensor<f32>, tensor<f32>, {ty [c1,c1,kH,kW]}, {ty [c1]}, tensor<f32>, tensor<f32>, {ty [c2,c1,kH,kW]}, {ty [c2]}, tensor<f32>, tensor<f32>, {ty [c2,c2,kH,kW]}, {ty [c2]}, tensor<f32>, tensor<f32>, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→BN→relu)×2→pool →(conv→BN→relu)×2→pool →flatten→(dense→relu)×2→dense ──\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" M1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" M1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" M2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" M2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  rs "%flat" "%pool2" [B,c2,Hp,Wp] [B,flat] ++
  dense "%h5" "%flat" "%W5" "%b5" flat d1 ++ relu2 "%a5" "%h5" d1 ++
  dense "%h6" "%a5" "%W6" "%b6" d1 d1 ++ relu2 "%a6" "%h6" d1 ++
  dense "%logits" "%a6" "%W7" "%b7" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut)+relu → scatter → (relu→BN-back→convBack)×stage, twice ──\n" ++
  dg "%dx7" "%dy" "%W7" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dy6" "%h6" "%dx7" d1 ++
  dg "%dx6" "%dy6" "%W6" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy5" "%h5" "%dx6" d1 ++
  dg "%dx5" "%dy5" "%W5" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  rs "%dpool2" "%dx5" [B,flat] [B,c2,Hp,Wp] ++
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++ rs "%dac4f" "%dac4" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn4" "%bn4" "%dac4f" M2 ++
  bnBack "%dhc4f" "%bn4" "%g4" "%dbn4" M2 ++ bnParamGrad "%dg4" "%dbt4" "%bn4" "%dbn4" M2 ++
  rs "%dhc4" "%dhc4f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++ rs "%dac3f" "%dac3" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn3" "%bn3" "%dac3f" M2 ++
  bnBack "%dhc3f" "%bn3" "%g3" "%dbn3" M2 ++ bnParamGrad "%dg3" "%dbt3" "%bn3" "%dbn3" M2 ++
  rs "%dhc3" "%dhc3f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++ rs "%dac2f" "%dac2" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn2" "%bn2" "%dac2f" M1 ++
  bnBack "%dhc2f" "%bn2" "%g2" "%dbn2" M1 ++ bnParamGrad "%dg2" "%dbt2" "%bn2" "%dbn2" M1 ++
  rs "%dhc2" "%dhc2f" [B,M1] [B,c1,H,W] ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++ rs "%dac1f" "%dac1" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn1" "%bn1" "%dac1f" M1 ++
  bnBack "%dhc1f" "%bn1" "%g1" "%dbn1" M1 ++ bnParamGrad "%dg1" "%dbt1" "%bn1" "%dbn1" M1 ++
  rs "%dhc1" "%dhc1f" [B,M1] [B,c1,H,W] ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dW7" "%a6" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%db7" "%dy" nClasses ++
  dg "%dW6" "%a5" "%dy6" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%db6" "%dy6" d1 ++
  dg "%dW5" "%flat" "%dy5" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db5" "%dy5" d1 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 22 params, incl. scalar γ/β) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++ sgd "%g1" "%dg1" "tensor<f32>" ++ sgd "%bt1" "%dbt1" "tensor<f32>" ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++ sgd "%g2" "%dg2" "tensor<f32>" ++ sgd "%bt2" "%dbt2" "tensor<f32>" ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++ sgd "%g3" "%dg3" "tensor<f32>" ++ sgd "%bt3" "%dbt3" "tensor<f32>" ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++ sgd "%g4" "%dg4" "tensor<f32>" ++ sgd "%bt4" "%dbt4" "tensor<f32>" ++
  sgd "%W5" "%dW5" (ty [flat,d1]) ++ sgd "%b5" "%db5" (ty [d1]) ++
  sgd "%W6" "%dW6" (ty [d1,d1]) ++ sgd "%b6" "%db6" (ty [d1]) ++
  sgd "%W7" "%dW7" (ty [d1,nClasses]) ++ sgd "%b7" "%db7" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, tensor<f32>, tensor<f32>, {ty [c1,c1,kH,kW]}, {ty [c1]}, tensor<f32>, tensor<f32>, {ty [c2,c1,kH,kW]}, {ty [c2]}, tensor<f32>, tensor<f32>, {ty [c2,c2,kH,kW]}, {ty [c2]}, tensor<f32>, tensor<f32>, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
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
    (Proofs.StableHLO.mlpTrainStepText 128 784 512 512 10 "0.00078125")
  -- Chapter 4 CNN forward (1→32→32 conv, 28×28→14×14 maxpool, 6272→512→512→10).
  IO.FS.writeFile "verified_mlir/cnn_fwd.mlir"
    (Proofs.StableHLO.cnnFwdModuleV 128 1 32 14 14 512 10 3 3
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- Chapter 4 CNN full SGD train step (same dims; lr = 0.1/128 = mean-loss equiv).
  IO.FS.writeFile "verified_mlir/cnn_train_step.mlir"
    (Proofs.StableHLO.cnnTrainStepText 128 1 32 28 28 3 3 512 10 "0.00078125")
  -- Chapter 5 CIFAR forward (3→32→32 conv, 32×32→16×16 pool, 32→64→64 conv,
  -- 16×16→8×8 pool, flatten 4096→512→512→10). h=w=8 ⇒ input 3·32·32 = 3072.
  IO.FS.writeFile "verified_mlir/cifar_fwd.mlir"
    (Proofs.StableHLO.cifarFwdModuleV 128 3 32 64 8 8 512 10 3 3
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- Chapter 5 CIFAR full SGD train step (32×32 stage-1 spatial; lr = 0.1/128).
  IO.FS.writeFile "verified_mlir/cifar_train_step.mlir"
    (Proofs.StableHLO.cifarTrainStepText 128 3 32 64 32 32 3 3 512 10 "0.00078125")
  -- Chapter 5 CIFAR **BatchNorm** forward (per-example BN after each conv; ε=1e-5).
  IO.FS.writeFile "verified_mlir/cifar_bn_fwd.mlir"
    (Proofs.StableHLO.cifarBnFwdModuleV 128 3 32 64 8 8 512 10 3 3 "1.0e-05"
       (fun _ _ _ _ => 0) (fun _ => 0) 1 0 0 (fun _ _ _ _ => 0) (fun _ => 0) 1 0 0
       (fun _ _ _ _ => 0) (fun _ => 0) 1 0 0 (fun _ _ _ _ => 0) (fun _ => 0) 1 0 0
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- Chapter 5 CIFAR **BatchNorm** full SGD train step (ε=1e-5; lr = 0.1/128).
  IO.FS.writeFile "verified_mlir/cifar_bn_train_step.mlir"
    (Proofs.StableHLO.cifarBnTrainStepText 128 3 32 64 32 32 3 3 512 10 "1.0e-05" "0.00078125") : IO Unit)
