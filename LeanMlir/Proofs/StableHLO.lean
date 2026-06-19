import LeanMlir.Proofs.IR
import LeanMlir.Proofs.CifarCNN
import LeanMlir.Proofs.StridedConv
import LeanMlir.Proofs.PerChannelBN
import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.MobileNetV2
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.ConvNeXt

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

-- Each extension of the `SHlo` constructor list raises the per-`whnf` cost of unfolding the
-- `den` match (the `brecOn`/`below` packaging scales with the constructor count), so the
-- `den (emit …) = <math>` faithfulness proofs below — even the trivial ones — sit closer to the
-- heartbeat ceiling with every added op. The MobileNetV2 depthwise-SGD ops (the 4 `depthwise*Sgd`
-- constructors) pushed several of these over the 200000 default; raise the file floor (the heavy
-- `cnnBackGraph_faithful` keeps its own larger `2000000` bump below).
set_option maxHeartbeats 1000000

-- ════════════════════════════════════════════════════════════════
-- § Batched lift (EfficientNet) — per-example block-apply over N examples,
--   plus the one genuinely batch-coupled op (true batch-norm).
-- ════════════════════════════════════════════════════════════════

/-- **Per-example block-apply.** Lift a per-example map `f : Vec a → Vec b` to a
    batch of `N` examples laid out row-major `[N, a] ↦ [N, b]` (the network's
    `[N,C,H,W]`-style flattening): example `n` occupies the `finProdFinEquiv`
    block `{(n, ·)}`. Every spatial/channel op in EfficientNet is batch-separable
    and lifts this way; only true batch-norm (`bnBatchTensor4`) couples the batch. -/
noncomputable def batchMap (N : Nat) {a b : Nat} (f : Vec a → Vec b) :
    Vec (N * a) → Vec (N * b) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    f (fun i : Fin a => x (finProdFinEquiv (p.1, i))) p.2

/-- The `n`-th example's slice of a batch laid out row-major `[N, a]`. A shared
    weight's batched gradient is the sum over `n` of the per-example gradient on
    `batchSlice n` — the form the batched param-SGD dens take (so the §1 fold closes
    via the per-example cert + sum-linearity). -/
def batchSlice (N a : Nat) (v : Vec (N * a)) (n : Fin N) : Vec a :=
  fun i => v (finProdFinEquiv (n, i))

/-- **A batch-separable EfficientNet op**, shape-indexed by per-example in/out
    length. The descriptor carried by `SHlo.batchOp`; its `denOp` is the proven
    per-example forward, lifted by `batchMap`. (swish/sigmoid/relu/addV are
    pointwise, so they need no descriptor — the existing tokens already denote
    them block-diagonally at the batched index `N·(c·h·w)`.) -/
inductive BatchableOp : Nat → Nat → Type where
  | conv {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*h*w) (oc*h*w)
  | convStrided {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (bias : Vec oc)            : BatchableOp (ic*(2*h)*(2*w)) (oc*h*w)
  | depthwise {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*h*w) (c*h*w)
  | depthwiseStrided {c h w kH kW : Nat} (wName bName : String)
      (W : DepthwiseKernel c kH kW) (bias : Vec c)         : BatchableOp (c*(2*h)*(2*w)) (c*h*w)
  | dense {a c : Nat} (wName bName : String)
      (W : Mat a c) (bias : Vec c)                         : BatchableOp a c
  | gap {c h w : Nat}                                      : BatchableOp (c*h*w) c
  | seBlock {c h w r : Nat} (w1Name b1Name w2Name b2Name : String)
      (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) : BatchableOp (c*h*w) (c*h*w)

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
  -- Chapter-2 SGD tail (the linear train step, folded into the AST): the two
  -- fused parameter-update ops that take the loss cotangent and emit the
  -- weight/bias SGD step. `weightSgd`: `W − lr·(x⊗dy)` (`dot_general` batch-
  -- contract → const → multiply → subtract), `den` = the certified `sgdW` step
  -- at B=1. `biasSgd`: `b − lr·(Σ_batch dy)` (`reduce` → const → mul → sub).
  -- LinearFaithfulPoC proves both `den`s = the certified loss-descent step.
  | weightSgd  {m n : Nat} (xName wName lrStr : String) (x : Vec m) (W : Mat m n) (lr : ℝ) : SHlo n → SHlo (m*n)
  | biasSgd    {n : Nat} (bName lrStr : String) (b : Vec n) (lr : ℝ)                        : SHlo n → SHlo n
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
  -- Chapter 4 (CNN) param-SGD tail (the conv train step, folded into the AST):
  -- the fused conv kernel/bias update ops — the conv analogue of `weightSgd`/`biasSgd`.
  -- `convWeightSgd`: `W − lr·(conv2d_weight_grad(b,x)·dy)` via the transpose-trick conv
  -- (transpose→transpose→convolution→transpose, then const→multiply→subtract), `den`
  -- = `cnn_render_convW_certified`. `convBiasSgd`: `b − lr·(conv2d_bias_grad(W,x)·dy)`
  -- (reduce over batch+spatial [0,2,3], then SGD). `xName`/`wName`/`bName` are the saved
  -- activation/kernel/bias SSA names; `W,x,b,lr` carry the den. CnnFaithfulPoC proves
  -- both `den`s = the certified loss-descent step (via the conv VJP bridges).
  | convWeightSgd {ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convBiasSgd   {ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- Chapter 5 (per-channel BatchNorm) param-SGD tail (the BN train step, folded into
  -- the AST): the fused per-channel γ/β update ops. `bnGammaSgd`: `γ − lr·dγ`,
  -- `dγ_c = Σ_{b,h,w} dy·x̂` (x̂ recomputed from the saved BN input `v` = conv output,
  -- `den` = `cifar_bn_render_gamma_certified` via `reassocFwd`); `bnBetaSgd`: `β − lr·dβ`,
  -- `dβ_c = Σ_{b,h,w} dy`. `gName`/`bName`/`vName` are the γ/β/conv-output SSA names;
  -- `epsStr` the ε literal. CifarBnFaithfulPoC proves both `den`s = the certified step.
  | bnGammaSgd {oc h w : Nat} (gName vName epsStr lrStr : String) (ε : ℝ) (γ : Vec oc)
      (v : Vec (oc*h*w)) (lr : ℝ)                          : SHlo (oc*h*w) → SHlo oc
  | bnBetaSgd  {oc h w : Nat} (bName lrStr : String) (β : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
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
  -- GAP backward (VJP): per-channel cotangent broadcast over H×W, /(h·w).
  | gapBack    {c h w : Nat}                                    : SHlo c → SHlo (c*h*w)
  -- Broadcast backward (VJP = sum-over-spatial): the adjoint of `broadcastFlat`.
  | broadcastBack {c h w : Nat}                                 : SHlo (c*h*w) → SHlo c
  -- Chapter 6 Milestone B (ResNet-34 downsampling): stride-2 SAME conv forward
  -- (`stablehlo.convolution` with `window_strides=[2,2]`) and its input-VJP
  -- (zero-upsample the cotangent — `lhs_dilation` — then the reversed-kernel
  -- conv). `den` via the proven `flatConvStride2` / `flatConvStride2_has_vjp`.
  | flatConvStridedF {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc)              : SHlo (ic*(2*h)*(2*w)) → SHlo (oc*h*w)
  | convStridedBack  {ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic*(2*h)*(2*w))) : SHlo (oc*h*w) → SHlo (ic*(2*h)*(2*w))
  -- Chapter 6 Milestone B (ResNet-34 downsampling) param-SGD tail: the strided conv
  -- kernel/bias update ops — the stride-2 analogues of `convWeightSgd`/`convBiasSgd`.
  -- `convStridedWeightSgd`: `W − lr·(flatConvStride2_weight_grad(b,x)·dy)` — zero-upsample
  -- the cotangent (the decimate-backward) then the SAME transpose-trick stride-1 weight-grad
  -- conv on the 2h×2w grid; `den` = the generic strided weight bridge (covers the 3×3
  -- downsample/projection AND the 7×7 stem, kH/kW-generic). `convStridedBiasSgd`: the bias
  -- grad is stride-INDEPENDENT (`Σ_{batch,spatial} dy`), so it emits the SAME `reduce` text
  -- as `convBiasSgd` (its `skel` aliases that op's Raw); only its `den` differs (the strided
  -- VJP). ResNet34FaithfulPoC proves both `den`s = the certified loss-descent step.
  | convStridedWeightSgd {ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (ic*(2*h)*(2*w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo (oc*ic*kH*kW)
  | convStridedBiasSgd   {ic oc h w kH kW : Nat} (bName lrStr : String)
      (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w))) (b : Vec oc) (lr : ℝ)
                                                           : SHlo (oc*h*w) → SHlo oc
  -- MobileNetV2 (inverted-residual) param-SGD tail: the depthwise kernel/bias update ops,
  -- the depthwise analogues of `convWeightSgd`/`convBiasSgd`. `depthwiseWeightSgd` (stride-1,
  -- blocks b2/b4): `W − lr·(depthwise_weight_grad(b,x)·dy)` via the per-channel transpose-trick
  -- conv (`batch_group_count = c`, output [1,c,kH,kW]→[c,1,kH,kW]); `den` =
  -- `mnv2_render_depthwiseW_certified`. `depthwiseStridedWeightSgd` (stride-2, blocks b1/b3/b5/b6):
  -- zero-upsample dy (interior=1 → 2h×2w) then the SAME per-channel weight-grad on the 2h×2w grid;
  -- `den` = `mnv2_render_depthwiseW_strided_certified`. The depthwise bias grad is stride-INDEPENDENT
  -- (`Σ_{batch,spatial} dy`), so both bias ops emit the SAME `reduce` text as `convBiasSgd` (their
  -- `skel` aliases that op's Raw); only their `den` differs. MobileNetV2FaithfulPoC proves all four
  -- `den`s = the certified loss-descent step.
  | depthwiseWeightSgd {c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo (c*kH*kW)
  | depthwiseBiasSgd   {c h w kH kW : Nat} (bName lrStr : String)
      (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) (b : Vec c) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo c
  | depthwiseStridedWeightSgd {c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Vec (c*(2*h)*(2*w))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo (c*kH*kW)
  | depthwiseStridedBiasSgd   {c h w kH kW : Nat} (bName lrStr : String)
      (W : DepthwiseKernel c kH kW) (x : Vec (c*(2*h)*(2*w))) (b : Vec c) (lr : ℝ)
                                                           : SHlo (c*h*w) → SHlo c
  -- Chapter 9 (ConvNeXt-T) param-SGD tail. `layerScaleChGammaSgd`: the PER-CHANNEL layer-scale γ
  -- update `γ_c − lr·dγ_c`, `dγ_c = Σ_{b,h,w} x⊙dy` (the saved layer input `x` ⊙ the cotangent,
  -- reduced over batch+spatial per channel — `lsGradCh`). `γ : Vec c`, broadcast over spatial via
  -- `chanIdx` by the `layerScaleChF` forward; `den` = ConvNeXtFaithfulPoC's `cnx_render_lsgammaCh`.
  | layerScaleChGammaSgd {c h w : Nat} (gName xName lrStr : String)
      (x : Vec (c*h*w)) (γ : Vec c) (lr : ℝ)               : SHlo (c*h*w) → SHlo c
  -- `lnGammaSgd`/`lnBetaSgd`: the SCALAR LayerNorm γ/β updates (the `bnF` sites — scalar LN over the
  -- whole `n = c·h·w`, `γ β : Vec 1` ≅ `tensor<f32>`). `dγ = Σ_{b,k} dy·x̂` (x̂ recomputed from the
  -- saved LN input `x`, `lnParamGrad`'s dγ half), `dβ = Σ_{b,k} dy`. `den` = the certified scalar-LN
  -- grad (`cnx_render_ln{gamma,beta}_certified`, the Vec-1 embedding). Output `SHlo 1`.
  | lnGammaSgd {n : Nat} (gName xName epsStr lrStr : String) (ε : ℝ) (x : Vec n) (γ : Vec 1) (lr : ℝ)
                                                           : SHlo n → SHlo 1
  | lnBetaSgd  {n : Nat} (bName lrStr : String) (β : Vec 1) (lr : ℝ)
                                                           : SHlo n → SHlo 1
  -- Chapter 9 scaling pass (full ConvNeXt-T): stride-4 SAME conv forward — the
  -- 4×4/s4 patchify stem (`stablehlo.convolution` with `window_strides=[4,4]`).
  -- `den` via the proven `flatConvStride4` (= decimate ∘ decimate ∘ stride-1 conv).
  | flatConvStride4F {ic oc h w kH kW : Nat} (wName bName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) : SHlo (ic*(2*(2*h))*(2*(2*w))) → SHlo (oc*h*w)
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
  -- Chapter 9 (ConvNeXt): GELU forward (tanh approximation,
  -- `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, via `stablehlo.tanh`) and its
  -- input-VJP (`dy · gelu'(x)`, closed form from the tanh-approx derivative).
  -- Like swish/sigmoid, SMOOTH everywhere (no kink, NO smoothness hyp — the VJP is
  -- the GLOBAL `gelu_has_vjp`, not `_at`). `geluBack`'s `xName`/`x` is the saved
  -- pre-activation. `den` via the proven `gelu` / `gelu_has_vjp` (LayerNorm.lean).
  | geluF      {n : Nat}                                        : SHlo n → SHlo n
  | geluBack   {n : Nat} (xName : String) (x : Vec n)           : SHlo n → SHlo n
  -- Chapter 9 (ConvNeXt): per-element layer-scale `γ ⊙ x` (diagonal linear, `γ : Vec n`
  -- over the flattened `c·h·w` map). `den` via the proven `layerScale` (ConvNeXt.lean).
  | layerScaleF {n : Nat} (γName : String) (γ : Vec n)          : SHlo n → SHlo n
  -- Per-CHANNEL layer-scale (the paper's form, the committed full-T render's
  -- `tensor<c>` γ): `den` = the proven `layerScale` at the channel-expanded
  -- vector `γ ∘ chanIdx` (a constant reindex of the parameter).
  | layerScaleChF {c h w : Nat} (γName : String) (γ : Vec c)    : SHlo (c*h*w) → SHlo (c*h*w)
  -- Chapter 10 (ViT): ROW-softmax forward — each of the `m` rows of an `[m,n]`
  -- matrix (flattened to `Vec (m*n)`, row-major) gets the 1-D `softmax` over its
  -- `n` columns (`reduce add` over the LAST axis, broadcast, divide — NO max-shift,
  -- matching the proven plain exp/sum `softmax`). `den` via `rowSoftmaxFlat` (=
  -- `Mat.flatten ∘ rowSoftmax ∘ Mat.unflatten`, the proven `rowSoftmax`).
  | softmaxRowF    {m n : Nat}                                  : SHlo (m*n) → SHlo (m*n)
  -- ROW-softmax input-VJP — per row the proven closed form `pᵢ⊙(dyᵢ − ⟨pᵢ,dyᵢ⟩)`
  -- with `p = softmax(preActᵢ)` recomputed from the saved pre-softmax scores
  -- (`xName`/`preAct`). SMOOTH everywhere (softmax has no kink). `den` via
  -- `rowSoftmaxBackFlat` (= `Mat.flatten ∘ rowSoftmax_has_vjp_mat.backward ∘ Mat.unflatten`).
  | softmaxRowBack {m n : Nat} (xName : String) (preAct : Vec (m*n)) : SHlo (m*n) → SHlo (m*n)
  -- Chapter 10 (ViT): matrix multiply `C = A·B` on row-major flattened operands
  -- (reshape both to rank-3, `stablehlo.dot_general` batching dim 0, contract A's
  -- last axis with B's middle, reshape back). Binary like `.sub`/`.addV`. `den` via
  -- `matMulFlat` (= flatten ∘ `Mat.mul` ∘ unflatten). The attention BACKWARDS reuse
  -- this same token — matmul's VJP IS matmul (`dA = dC·Bᵀ`, `dB = Aᵀ·dC`).
  | matmulF    {m k n : Nat}                                    : SHlo (m*k) → SHlo (k*n) → SHlo (m*n)
  -- Matrix transpose on the row-major flat layout (`stablehlo.transpose
  -- dims=[0,2,1]` at rank 3). `den` via `transposeFlat` (= flatten ∘ `Mat.transpose`
  -- ∘ unflatten). Pairs with `matmulF` to spell the attention backward matmuls.
  | transposeF {m n : Nat}                                      : SHlo (m*n) → SHlo (n*m)
  -- Scalar multiply `s · x` (`stablehlo.multiply` against a splat constant) — the
  -- `1/√d` of SDPA. `sStr` is the rendered literal (denotation-irrelevant); `s`
  -- carries the den. Linear, so it is its own VJP.
  | scaleF     {n : Nat} (sStr : String) (s : ℝ)                : SHlo n → SHlo n
  -- ROW-wise LayerNorm forward over an `[m,n]` row-major flat: each token row gets
  -- `bnF`'s normalize/affine graph with μ/var reduced over the LAST axis (scalar
  -- γ/β — LayerNorm IS per-example BN, `layerNormForward := bnForward` defeq).
  -- `den` via `rowLNFlat` (rowwise `bnForward`).
  | lnRowF     {m n : Nat} (gName bName epsStr : String) (ε γ β : ℝ) : SHlo (m*n) → SHlo (m*n)
  -- ROW-wise LayerNorm input-VJP — per row `bnBack`'s consolidated three-term
  -- gradient, recomputing x̂/istd from the saved flat pre-LN input `x` (`xName`).
  -- Total in `x`; faithful (= pdiv-Jacobian per row) under `0 < ε`.
  | lnRowBack  {m n : Nat} (gName xName epsStr : String) (ε γ : ℝ) (x : Vec (m*n)) : SHlo (m*n) → SHlo (m*n)
  -- PER-TOKEN dense forward: every row of the `[N,a]` flat through the same
  -- `W:[a,c]` + bias (`dot_general` contracting the feature axis `[2] x [0]`,
  -- bias broadcast `dims=[2]`). `den` via `rowDenseFlat` (rowwise `dense`).
  | denseRowF  {N a c : Nat} (wName bName : String) (W : Mat a c) (b : Vec c) : SHlo (N*a) → SHlo (N*c)
  -- PER-TOKEN dense input-VJP `dX = dY·Wᵀ` (`dot_general` contracting dy's feature
  -- axis with W's OUTPUT axis `[2] x [1]`). `den` via `rowDenseBackFlat` (rowwise
  -- `Mat.mulVec W` = the proven `dense_has_vjp` backward). Linear — global VJP.
  | denseRowBack {N a c : Nat} (wName : String) (W : Mat a c)   : SHlo (N*c) → SHlo (N*a)
  -- ViT patch embedding (one coarse token, like `seBlock`): stride-P VALID conv
  -- (kernel `[D,ic,P,P]`, the non-overlapping patch projection) + bias, channels-
  -- last transpose + flatten to `[N,D]` tokens, prepend the CLS token, add the
  -- position embedding. `den` via `patchEmbedFlat` (a local re-spelling of the
  -- proven `patchEmbed_flat`, Attention.lean — the tie is `rfl` in ViTFwdGraph).
  | patchEmbedF {ic H W P N D : Nat} (wName bName clsName posName : String)
      (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N+1) D) :
      SHlo (ic*H*W) → SHlo ((N+1)*D)
  -- ViT patch-embedding input-VJP: the strided-P patchify conv's input gradient
  -- (reversed-kernel `conv_transpose` on the patch-token rows of the `[N+1,D]`
  -- cotangent; the CLS row and position-add contribute nothing — input-VJP = id
  -- on a +constant). `den` via `patchEmbedBackFlat` (= the proven
  -- `patchEmbed_input_grad_formula` = `patchEmbed_flat_has_vjp.backward`, the tie
  -- is `rfl` in ViTBackB0). Linear in the cotangent — activation-independent, so
  -- it routes through the generic `batched` Raw/Tok tag (like the strided-conv
  -- backward batched ops) rather than a bespoke top-level Raw/Tok constructor.
  | patchEmbedBack {ic H W P N D : Nat} (wName : String)
      (Wc : Kernel4 D ic P P) :
      SHlo ((N+1)*D) → SHlo (ic*H*W)
  -- CLS-token gather: row 0 of the `[N+1,D]` flat (`stablehlo.slice` after
  -- reshape) — the classifier head's input. `den` via `clsSliceFlat` (= the
  -- proven `cls_slice_flat`, Attention.lean).
  | clsSliceF  {N D : Nat}                                      : SHlo ((N+1)*D) → SHlo D
  -- CLS-slice VJP: scatter `dy` to row 0, zeros elsewhere (`stablehlo.pad` with
  -- `high = [0, N, 0]`). `den` via `clsPadFlat` (= the proven
  -- `cls_slice_flat_has_vjp.backward`). Linear — global VJP.
  | clsPadF    {N D : Nat}                                      : SHlo D → SHlo ((N+1)*D)
  -- Multi-head (ch10 scaling pass): per-head column slice — head `h`'s `[N,d]`
  -- block of the `[N,heads·d]` flat (columns `[h·d,(h+1)·d)` are contiguous in the
  -- row-major layout: `stablehlo.slice` on the feature axis after reshape).
  -- `den` via `headSliceFlat` (= `mhsa_layer`'s `finProdFinEquiv (h, ·)` column
  -- gather). Linear reindex.
  | headSliceF {N heads d : Nat} (h : Fin heads)                : SHlo (N*(heads*d)) → SHlo (N*d)
  -- Multi-head: per-head column scatter — pad an `[N,d]` head block into head `h`'s
  -- columns of a zero `[N,heads·d]` (`stablehlo.pad` on the feature axis). Both the
  -- slice's VJP AND the forward concat (`concat = Σ_h headPadF h ∘ head h` — every
  -- column hits exactly one head, and the sum stays at the ONE index `N·(heads·d)`,
  -- dodging the `(N·a)+(N·b)` Nat-cast trap a binary concat token would hit). Linear.
  | headPadF   {N heads d : Nat} (h : Fin heads)                : SHlo (N*d) → SHlo (N*(heads*d))
  -- ViT vector-LN affine (the ch10 scaling pass): per-token broadcast scale — every
  -- row of an `[m,n]` flat elementwise-scaled by the SHARED `γ : [n]` (broadcast over
  -- the row axis; contrast `layerScaleF`, which has a distinct γ per position).
  -- Diagonal-linear, so it is its own input-VJP (the layer-scale trick, row-lifted).
  -- `den` via `rowScaleFlat`.
  | rowScaleF  {m n : Nat} (gName : String) (γ : Vec n)         : SHlo (m*n) → SHlo (m*n)
  -- Per-token broadcast bias `+ β` (`β : [n]` shared across rows). Translation —
  -- the input-VJP is the identity (cotangent passthrough). `den` via `rowBiasFlat`.
  | rowBiasF   {m n : Nat} (bName : String) (β : Vec n)         : SHlo (m*n) → SHlo (m*n)
  -- Chapter 8 (EfficientNet, BATCHED): a batch-separable op (conv/depthwise/dense/
  -- GAP/SE) lifted to `N` examples by `batchMap`; `den` is `batchMap N (denOp op)`.
  -- The whole EfficientNet forward graph lives at the batched index `N·(c·h·w)`;
  -- pointwise swish/sigmoid/relu/addV reuse their existing tokens at that index.
  | batchOp {N a b : Nat} (op : BatchableOp a b)               : SHlo (N * a) → SHlo (N * b)
  -- Chapter 8 (EfficientNet, BATCHED): TRUE batch-norm — reduce μ/var over the
  -- batch+spatial axes [0,2,3] per channel (NOT per-example). The one op that
  -- couples the batch; `den` is `bnBatchLA` (= the proven `bnBatchTensor4`,
  -- conjugated to the network's left-assoc `N·(oc·h·w)` flat index).
  | bnBatchF {N oc h w : Nat} (gName bName epsStr : String) (ε : ℝ) (γ β : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (oc * h * w))
  -- True batch-norm BACKWARD (VJP), `[N,C,H,W]` layout: the renderable three-term
  -- `bnBatchTensor4_grad_input` (reduce over [0,2,3] per channel). `den` is the
  -- proven `bnBatchTensor4` VJP backward (batch-coupled). Routes through the
  -- generic `batched` Raw/Tok tag like the forward batched ops.
  | bnBatchBack {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (N * (oc * (h * w)))) :
      SHlo (N * (oc * (h * w))) → SHlo (N * (oc * (h * w)))
  -- Batched conv input-VJP: `batchMap N` of the proven per-example conv
  -- input-grad (activation-independent — conv is linear). Routes through the
  -- generic `batched` tag like the forward batched ops.
  | convBackBatched {N ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * h * w))
  -- Batched STRIDE-2 conv input-VJP: `batchMap N` of the proven per-example
  -- strided-conv input-grad (`flatConvStride2_has_vjp` — activation-independent,
  -- strided conv = `decimate ∘ conv` is linear). The downsample basic-block's
  -- stride-2 conv1 backward; halves spatial vs `convBackBatched`. Routes through
  -- the generic `batched` tag like the stride-1 batched ops.
  | convStridedBackBatched {N ic oc h w kH kW : Nat} (wName : String)
      (W : Kernel4 oc ic kH kW) (b : Vec oc) :
      SHlo (N * (oc * h * w)) → SHlo (N * (ic * (2 * h) * (2 * w)))
  -- Batched depthwise input-VJP: `batchMap N` of the proven per-example
  -- depthwise input-grad (activation-independent — depthwise conv is linear).
  | depthwiseBackBatched {N c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * h * w))
  -- Batched STRIDE-2 depthwise input-VJP: `batchMap N` of the proven per-example
  -- strided-depthwise input-grad (`depthwiseStride2Flat_has_vjp` — activation-
  -- independent, strided depthwise = `decimate ∘ depthwise` is linear). The
  -- EfficientNet downsample MBConv's stride-2 depthwise backward; halves spatial
  -- vs `depthwiseBackBatched` (the depthwise analog of `convStridedBackBatched`).
  -- Routes through the generic `batched` tag like the stride-1 batched ops.
  | depthwiseStridedBackBatched {N c h w kH kW : Nat} (wName : String)
      (W : DepthwiseKernel c kH kW) (b : Vec c) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w)))
  -- True batch-norm backward on the NETWORK layout `N·(oc·h·w)` (what
  -- renderBody's `bnBatch` emits): the `bnBatchTensor4` backward reindex-
  -- conjugated to the left-assoc index (`bnBatchLA_eq_comp`).
  | bnBatchLABack {N oc h w : Nat} (gName xName epsStr : String) (ε : ℝ) (γ : Vec oc)
      (x : Vec (N * (oc * h * w))) :
      SHlo (N * (oc * h * w)) → SHlo (N * (oc * h * w))
  -- Batched squeeze-excite backward: rowwise application of the proven per-example
  -- `seBlockFull` VJP. SE is non-linear, so the backward uses each example's forward
  -- activation `v` (unlike the linear conv/depthwise). `den` references the proven
  -- witness rowwise; renderable emission (batchMap-of-SE-subgraph) is deferred.
  | seBackBatched {N c h w r : Nat} (w1Name b1Name w2Name b2Name vName : String)
      (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
      (v : Vec (N * (c * h * w))) :
      SHlo (N * (c * h * w)) → SHlo (N * (c * h * w))
  -- Batched SE GATE COTANGENT: `dgate[n,c] = Σ_{h,w} x[n,c,h,w]·dy[n,c,h,w]` — the
  -- broadcast-adjoint of the Hadamard `x ⊙ dy`, i.e. the FIRST step of the SE gate
  -- backward (the cotangent at the gate's sigmoid output). The un-fused-SE param-grad
  -- ENTRY POINT: feeds `sigmoidBack → denseWeightSgdB(W₂)/denseBiasSgdB + denseRowBack(W₂)
  -- → swishBack → denseWeightSgdB(W₁)/denseBiasSgdB`, exposing the SE dense param grads the
  -- fused `seBackBatched` (input-cotangent only) cannot. `x` = the SE input (saved by name),
  -- `e` = the SE-output cotangent. `den` = batched `broadcastFlat_has_vjp.backward (x⊙dy)`.
  | seReduceB {N c h w : Nat} (xName : String) (x : Vec (N * (c * h * w))) :
      SHlo (N * (c * h * w)) → SHlo (N * c)
  -- Batched GLOBAL-AVERAGE-POOL backward (VJP): `dx[n,c,h,w] = dgap[n,c]/(h·w)` — the
  -- per-example `globalAvgPoolFlat_has_vjp` backward (broadcast over spatial, ÷h·w),
  -- lifted by `batchMap N`. The head's GAP backward (`gapBack` is per-example, not a
  -- `BatchableOp`, so it needs its own batched ctor). `den` = `batchMap N (gap-adjoint)`.
  | gapBackBatched {N c h w : Nat} : SHlo (N * c) → SHlo (N * (c * h * w))
  -- Chapter 8 (EfficientNet, BATCHED) param-SGD tail: the fused per-channel BN
  -- γ/β updates over the network layout `N·(oc·(h·w))`. `den` is the per-channel BN
  -- grad at the merged batch+spatial axis `m = N·(h·w)` (via `bnchwFwd`, the
  -- network→oc-major reindex), so it is *exactly* `enet_render_bn{gamma,beta}_certified`'s
  -- LHS — the §1 fold is a one-line delegation. Emit recomputes x̂ from the saved BN
  -- input `vName` then `reduce[0,2,3]` (the dγ/dβ in `bnBatchBack`). Output is `Vec oc`.
  | bnGammaSgdB {N oc h w : Nat} (gName vName epsStr lrStr : String) (ε : ℝ) (γ : Vec oc)
      (v : Vec (N * (oc * (h * w)))) (lr : ℝ)             : SHlo (N * (oc * (h * w))) → SHlo oc
  | bnBetaSgdB  {N oc h w : Nat} (bName lrStr : String) (β : Vec oc) (lr : ℝ)
                                                          : SHlo (N * (oc * (h * w))) → SHlo oc
  -- Batched dense weight/bias SGD (SE squeeze/excite convs as `dot_general`, head dense).
  -- `den` = θ − lr·(Σ_n per-example grad on `batchSlice n`); the shared-weight batch sum.
  -- Emit reuses the `weightSgd`/`biasSgd` text (already batch-contracts over `B`).
  | denseWeightSgdB {N a c : Nat} (xName wName lrStr : String) (x : Vec (N * a)) (W : Mat a c) (lr : ℝ)
                                                          : SHlo (N * c) → SHlo (a * c)
  | denseBiasSgdB   {N c : Nat} (bName lrStr : String) (b : Vec c) (lr : ℝ)
                                                          : SHlo (N * c) → SHlo c
  -- Batched conv weight SGD (1×1 expand/project/head; the transpose-trick wgrad).
  -- `den` = flatten W − lr·(Σ_n per-example `conv2d_weight_grad` on `batchSlice n`).
  | convWeightSgdB {N ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  -- Batched STEM 3×3-strided conv weight + DEPTHWISE weight (stride 1/2) SGD. Same
  -- Σ_n shared-weight batch sum; the depthwise grad is HasVJP3 (flatten-bridged).
  | convStridedWeightSgdB {N ic oc h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW) (lr : ℝ)
                                                          : SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW)
  | depthwiseWeightSgdB {N c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                          : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)
  | depthwiseStridedWeightSgdB {N c h w kH kW : Nat} (xName wName lrStr : String)
      (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW) (lr : ℝ)
                                                          : SHlo (N * (c * h * w)) → SHlo (c * kH * kW)

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

/-- **Row-softmax (flattened)** — apply the 1-D `softmax` (MLP.lean) to each of
    the `m` rows of the row-major `Vec (m*n)`. Definitionally equal to
    `Mat.flatten ∘ rowSoftmax ∘ Mat.unflatten` (Attention.lean's `rowSoftmax`);
    spelled with MLP's `softmax` so `StableHLO` needn't import `Attention`
    (the tie to `rowSoftmax` is an `rfl` faithfulness lemma in `TestSoftmaxRow`). -/
noncomputable def rowSoftmaxFlat (m n : Nat) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => softmax n ((Mat.unflatten v) i))

/-- **Row-softmax backward (flattened)** — per row, the proven closed form
    `pᵢ⊙(dyᵢ − ⟨pᵢ,dyᵢ⟩)` with `pᵢ = softmax(preActᵢ)`. Definitionally equal to
    `Mat.flatten ∘ rowSoftmax_has_vjp_mat.backward (Mat.unflatten preAct) ∘ Mat.unflatten`
    (since `softmax_has_vjp.backward z dy i = let p := softmax z; p i·(dy i − ⟨p,dy⟩)`);
    spelled with MLP's `softmax` to keep `Attention` out of `StableHLO`'s imports. -/
noncomputable def rowSoftmaxBackFlat (m n : Nat) (preAct dy : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i =>
    let p := softmax n ((Mat.unflatten preAct) i)
    let dyi := (Mat.unflatten dy) i
    let s := ∑ j, p j * dyi j
    fun c => p c * (dyi c - s))

-- ── Chapter 10 (ViT) den helpers — flattened matrix/row-wise forms, spelled
--    with `Mat`/`bnForward`/`dense` so `StableHLO` needn't import `Attention`
--    (the rfl ties to `rowSoftmax`-style Attention forms live in ViTFwdGraph). ──

/-- **Flattened matrix multiply** `C = A·B` on row-major flat operands.
    Definitionally `Mat.flatten ∘ Mat.mul ∘ Mat.unflatten²`. -/
noncomputable def matMulFlat (m k n : Nat) (a : Vec (m*k)) (b : Vec (k*n)) : Vec (m*n) :=
  Mat.flatten (Mat.mul (Mat.unflatten a) (Mat.unflatten b))

/-- **Flattened transpose** — `Mat.transpose` conjugated by row-major flattening. -/
noncomputable def transposeFlat (m n : Nat) (v : Vec (m*n)) : Vec (n*m) :=
  Mat.flatten (Mat.transpose (Mat.unflatten v))

/-- **Row-wise LayerNorm (flattened)** — each of the `m` token rows gets the 1-D
    `bnForward` over its `n` features (LayerNorm IS per-example BN:
    `layerNormForward := bnForward` definitionally, LayerNorm.lean). -/
noncomputable def rowLNFlat (m n : Nat) (ε γ β : ℝ) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => bnForward n ε γ β ((Mat.unflatten v) i))

/-- **Row-wise LayerNorm input-VJP (flattened)** — per row the consolidated
    three-term `bn_grad_input`, recomputing x̂/istd from the saved pre-LN input. -/
noncomputable def rowLNBackFlat (m n : Nat) (ε γ : ℝ) (x dy : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun i => bn_grad_input n ε γ ((Mat.unflatten x) i) ((Mat.unflatten dy) i))

/-- **Per-token dense (flattened)** — every row of the `[N,a]` flat through the
    same `dense W b`. -/
noncomputable def rowDenseFlat (N a c : Nat) (W : Mat a c) (b : Vec c) (v : Vec (N*a)) :
    Vec (N*c) :=
  Mat.flatten (fun i => dense W b ((Mat.unflatten v) i))

/-- **Per-token dense input-VJP (flattened)** — per row `dX = W·dy` (=
    `(dense_has_vjp W b).backward`'s `Mat.mulVec W`, MLP.lean). -/
noncomputable def rowDenseBackFlat (N a c : Nat) (W : Mat a c) (dy : Vec (N*c)) :
    Vec (N*a) :=
  Mat.flatten (fun i => Mat.mulVec W ((Mat.unflatten dy) i))

/-- **ViT patch embedding (flattened)** — a LOCAL re-spelling of the proven
    `patchEmbed_flat` (Attention.lean), kept here so `StableHLO` needn't import
    `Attention` (the tie is an `rfl` lemma in ViTFwdGraph). Output row `n`:
    CLS token at `n = 0`, else conv-projection of patch `n−1` + bias; plus the
    position embedding everywhere. -/
noncomputable def patchEmbedFlat
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) :
    Vec (ic * H * W) → Vec ((N + 1) * D) :=
  fun img =>
    fun idx_out =>
      let n := (finProdFinEquiv.symm idx_out).1
      let d := (finProdFinEquiv.symm idx_out).2
      pos_embed n d +
        (if n.val = 0 then
          cls_token d
         else
          b_conv d +
          ∑ c : Fin ic, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
            W_conv d c kh kw *
              (let W' := W / patchSize
               let p := n.val - 1
               let h' := p / W'
               let w' := p % W'
               let hh := h' * patchSize + kh.val
               let ww := w' * patchSize + kw.val
               if hpad : hh < H ∧ ww < W then
                 img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
               else 0))

/-- **ViT patch-embedding input-VJP (flattened)** — a LOCAL re-spelling of the
    proven `patchEmbed_input_grad_formula` (Attention.lean), kept here so
    `StableHLO` needn't import `Attention` (the tie is an `rfl` lemma in
    ViTBackB0). The closed-form image cotangent: a sum over patches `p : Fin N`
    with reconstructed kernel offsets `(kh, kw)` matching the decoded input
    position `(c, hh, ww)`. The CLS row (`n = 0`) and the position-add (a
    +constant, input-VJP = id) contribute nothing — `idx_in` only flows through
    the conv-projection branch (`n = p+1`), so this is purely the strided 16×16
    patchify conv's input-VJP on the patch-token part of the cotangent. -/
noncomputable def patchEmbedBackFlat
    (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize)
    (dy : Vec ((N + 1) * D)) : Vec (ic * H * W) :=
  fun idx_in =>
    let c  := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1
    let hh := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let ww := (finProdFinEquiv.symm idx_in).2
    ∑ p : Fin N, ∑ kh : Fin patchSize, ∑ kw : Fin patchSize,
      let W' := W / patchSize
      let h' := p.val / W'
      let w' := p.val % W'
      if _h_match : h' * patchSize + kh.val = hh.val ∧
                    w' * patchSize + kw.val = ww.val then
        ∑ d : Fin D, W_conv d c kh kw *
          dy (finProdFinEquiv (p.succ, d))
      else 0

/-- **CLS slice (flattened)** — gather row 0 of the `[N+1,D]` flat (= the proven
    `cls_slice_flat`, Attention.lean; tie is `rfl` in ViTFwdGraph). -/
noncomputable def clsSliceFlat (N D : Nat) (v : Vec ((N+1)*D)) : Vec D :=
  fun k => v (finProdFinEquiv ((0 : Fin (N + 1)), k))

/-- **CLS pad (flattened)** — scatter `dy` to row 0, zeros elsewhere (= the proven
    `cls_slice_flat_has_vjp.backward`; tie is `rfl` in ViTFwdGraph). -/
noncomputable def clsPadFlat (N D : Nat) (dy : Vec D) : Vec ((N+1)*D) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    if p.1 = (0 : Fin (N + 1)) then dy p.2 else 0

/-- **Per-head column slice (flattened)** — head `h`'s `[N,d]` block of the
    `[N,heads·d]` flat: the `finProdFinEquiv (h, ·)` column gather `mhsa_layer`
    uses to feed each head's SDPA. -/
noncomputable def headSliceFlat (N heads d : Nat) (h : Fin heads)
    (v : Vec (N*(heads*d))) : Vec (N*d) :=
  Mat.flatten (fun (r : Fin N) (j : Fin d) =>
    (Mat.unflatten v) r (finProdFinEquiv (h, j)))

/-- **Per-head column pad (flattened)** — scatter an `[N,d]` head block into head
    `h`'s columns of a zero `[N,heads·d]`. `mhsa_layer`'s concat is the sum of
    these over heads; it is also `headSliceFlat`'s VJP. -/
noncomputable def headPadFlat (N heads d : Nat) (h : Fin heads)
    (v : Vec (N*d)) : Vec (N*(heads*d)) :=
  Mat.flatten (fun (r : Fin N) (hj : Fin (heads*d)) =>
    let p := finProdFinEquiv.symm hj
    if p.1 = h then (Mat.unflatten v) r p.2 else 0)

/-- **Row-broadcast scale (flattened)** — every token row elementwise-scaled by the
    shared `γ : Vec n` (= rowwise `layerScale γ`). -/
noncomputable def rowScaleFlat (m n : Nat) (γ : Vec n) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun r => layerScale γ ((Mat.unflatten v) r))

/-- **Row-broadcast bias (flattened)** — `+ β` on every token row. -/
noncomputable def rowBiasFlat (m n : Nat) (β : Vec n) (v : Vec (m*n)) : Vec (m*n) :=
  Mat.flatten (fun r k => (Mat.unflatten v) r k + β k)

/-- **The proven per-example forward of a `BatchableOp`** — exactly the existing
    batch-1 op (`flatConv`/`depthwiseFlat`/`dense`/`globalAvgPoolFlat`/`seBlockFull`/…).
    `SHlo.batchOp`'s `den` is `batchMap N (denOp op)`. -/
noncomputable def denOp : {a b : Nat} → BatchableOp a b → (Vec a → Vec b)
  | _, _, .conv _ _ W bias => flatConv W bias
  | _, _, .convStrided _ _ W bias => flatConvStride2 W bias
  | _, _, .depthwise _ _ W bias => depthwiseFlat W bias
  | _, _, .depthwiseStrided _ _ W bias => depthwiseStride2Flat W bias
  | _, _, .dense _ _ W bias => dense W bias
  | _, _, .gap (c := c) (h := h) (w := w) => globalAvgPoolFlat c h w
  | _, _, .seBlock (h := h) (w := w) _ _ _ _ W₁ b₁ W₂ b₂ => seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂

/-- **True batch-norm at the network's left-assoc `[N,C,H,W]` flat index.** The
    proven `bnBatchTensor4` (typed at `N·(oc·(h·w))`) conjugated by the `mul_assoc`
    reindex so it slots into the `N·(oc·h·w)` batched composition (where conv/etc.
    produce `oc·h·w = (oc·h)·w`). Reindex only — the function IS `bnBatchTensor4`. -/
noncomputable def bnBatchLA (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (oc * h * w)) → Vec (N * (oc * h * w)) :=
  fun v =>
    (fun y => y ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))
      (bnBatchTensor4 N oc h w ε γ β
        (v ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm))

/-- Channel index of a flat `c·h·w` position (the repo's left-assoc
    `finProdFinEquiv` convention: `k ↔ ((chan, row), col)`). Used to expand a
    per-channel parameter (`Vec c`) to the flat per-element map. -/
def chanIdx (c h w : Nat) (k : Fin (c * h * w)) : Fin c :=
  (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1

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
  | _, .weightSgd _ _ _ x W lr e => Mat.flatten (fun i j => W i j - lr * (x i * den e j))
  | _, .biasSgd _ _ b lr e       => fun j => b j - lr * den e j
  | _, .convWeightSgd _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx
        - lr * (conv2d_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e) idx
  | _, .convBiasSgd _ _ W x b lr e =>
      fun o => b o - lr * (conv2d_bias_grad_has_vjp W x).backward b (den e) o
  | _, .bnGammaSgd (oc := oc) (h := h) (w := w) _ _ _ _ ε γ v lr e =>
      fun c => γ c - lr *
        bnPerChannel_grad_gamma oc (h*w) ε (reassocFwd oc h w v) (reassocFwd oc h w (den e)) c
  | _, .bnBetaSgd (oc := oc) (h := h) (w := w) _ _ β lr e =>
      fun c => β c - lr * bnPerChannel_grad_beta oc (h*w) (reassocFwd oc h w (den e)) c
  | _, .layerScaleChGammaSgd (c := c) (h := h) (w := w) _ _ _ x γ lr e =>
      fun cc => γ cc - lr * ∑ k : Fin (c*h*w), (if chanIdx c h w k = cc then x k * den e k else 0)
  | _, .lnGammaSgd (n := n) _ _ _ _ ε x γ lr e =>
      fun _ => γ 0 - lr * bn_grad_gamma n ε x (den e)
  | _, .lnBetaSgd (n := n) _ _ β lr e =>
      fun _ => β 0 - lr * bn_grad_beta n (den e)
  | _, .bnGammaSgdB (N := N) (oc := oc) (h := h) (w := w) _ _ _ _ ε γ v lr e =>
      fun c => γ c - lr *
        bnPerChannel_grad_gamma oc (N*(h*w)) ε (bnchwFwd N oc h w v) (bnchwFwd N oc h w (den e)) c
  | _, .bnBetaSgdB (N := N) (oc := oc) (h := h) (w := w) _ _ β lr e =>
      fun c => β c - lr * bnPerChannel_grad_beta oc (N*(h*w)) (bnchwFwd N oc h w (den e)) c
  | _, .denseWeightSgdB (N := N) (a := a) (c := c) _ _ _ x W lr e =>
      Mat.flatten (fun i j => W i j - lr * ∑ n : Fin N, batchSlice N a x n i * batchSlice N c (den e) n j)
  | _, .denseBiasSgdB (N := N) (c := c) _ _ b lr e =>
      fun j => b j - lr * ∑ n : Fin N, batchSlice N c (den e) n j
  | _, .convWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx - lr * ∑ n : Fin N,
        (conv2d_weight_grad_has_vjp b (Tensor3.unflatten (batchSlice N (ic*h*w) x n))).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .convStridedWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx - lr * ∑ n : Fin N,
        (flatConvStride2_weight_grad_has_vjp b (batchSlice N (ic*(2*h)*(2*w)) x n)).backward
          (Kernel4.flatten W) (batchSlice N (oc*h*w) (den e) n) idx
  | _, .depthwiseWeightSgdB (N := N) (c := c) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Tensor3.flatten W idx - lr * ∑ n : Fin N,
        Tensor3.flatten ((depthwise_weight_grad_has_vjp3 b (Tensor3.unflatten (batchSlice N (c*h*w) x n))).backward
          W (Tensor3.unflatten (batchSlice N (c*h*w) (den e) n))) idx
  | _, .depthwiseStridedWeightSgdB (N := N) (c := c) (h := h) (w := w) _ _ _ b x W lr e =>
      fun idx => Tensor3.flatten W idx - lr * ∑ n : Fin N,
        (depthwiseStride2_weight_grad_has_vjp b (batchSlice N (c*(2*h)*(2*w)) x n)).backward
          (Tensor3.flatten W) (batchSlice N (c*h*w) (den e) n) idx
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
  | _, .gapBack (c := c) (h := h) (w := w) e =>
      (globalAvgPoolFlat_has_vjp c h w).backward (fun _ => 0) (den e)
  | _, .broadcastBack (c := c) (h := h) (w := w) e =>
      fun k => ∑ idx : Fin (c * h * w), if flatChannel c h w idx = k then den e idx else 0
  | _, .flatConvStridedF _ _ W b e => flatConvStride2 W b (den e)
  | _, .flatConvStride4F _ _ W b e => flatConvStride4 W b (den e)
  | _, .convStridedBack _ W b v e => (flatConvStride2_has_vjp W b).backward v (den e)
  | _, .convStridedWeightSgd _ _ _ b x W lr e =>
      fun idx => Kernel4.flatten W idx
        - lr * (flatConvStride2_weight_grad_has_vjp b x).backward (Kernel4.flatten W) (den e) idx
  | _, .convStridedBiasSgd _ _ W x b lr e =>
      fun o => b o - lr * (flatConvStride2_bias_grad_has_vjp W x).backward b (den e) o
  | _, .depthwiseWeightSgd _ _ _ b x W lr e => depthwiseWeightSgdDen b x W lr (den e)
  | _, .depthwiseBiasSgd _ _ W x b lr e => depthwiseBiasSgdDen W x b lr (den e)
  | _, .depthwiseStridedWeightSgd _ _ _ b x W lr e => depthwiseStridedWeightSgdDen b x W lr (den e)
  | _, .depthwiseStridedBiasSgd _ _ W x b lr e => depthwiseStridedBiasSgdDen W x b lr (den e)
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
  | _, .geluF (n := n) e => gelu n (den e)
  | _, .geluBack (n := n) _ x e => (gelu_has_vjp n).backward x (den e)
  | _, .layerScaleF (n := n) _ γ e => layerScale γ (den e)
  | _, .layerScaleChF (c := c) (h := h) (w := w) _ γ e =>
      layerScale (fun k => γ (chanIdx c h w k)) (den e)
  | _, .softmaxRowF (m := m) (n := n) e => rowSoftmaxFlat m n (den e)
  | _, .softmaxRowBack (m := m) (n := n) _ preAct e => rowSoftmaxBackFlat m n preAct (den e)
  | _, .matmulF (m := m) (k := k) (n := n) a b => matMulFlat m k n (den a) (den b)
  | _, .transposeF (m := m) (n := n) e => transposeFlat m n (den e)
  | _, .scaleF _ s e => fun i => s * den e i
  | _, .lnRowF (m := m) (n := n) _ _ _ ε γ β e => rowLNFlat m n ε γ β (den e)
  | _, .lnRowBack (m := m) (n := n) _ _ _ ε γ x e => rowLNBackFlat m n ε γ x (den e)
  | _, .denseRowF (N := N) (a := a) (c := c) _ _ W b e => rowDenseFlat N a c W b (den e)
  | _, .denseRowBack (N := N) (a := a) (c := c) _ W e => rowDenseBackFlat N a c W (den e)
  | _, .patchEmbedF (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ _ _ _ Wc bc cls pos e =>
      patchEmbedFlat ic H W P N D Wc bc cls pos (den e)
  | _, .patchEmbedBack (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ Wc e =>
      patchEmbedBackFlat ic H W P N D Wc (den e)
  | _, .clsSliceF (N := N) (D := D) e => clsSliceFlat N D (den e)
  | _, .clsPadF (N := N) (D := D) e => clsPadFlat N D (den e)
  | _, .headSliceF (N := N) (heads := heads) (d := d) h e => headSliceFlat N heads d h (den e)
  | _, .headPadF (N := N) (heads := heads) (d := d) h e => headPadFlat N heads d h (den e)
  | _, .rowScaleF (m := m) (n := n) _ γ e => rowScaleFlat m n γ (den e)
  | _, .rowBiasF (m := m) (n := n) _ β e => rowBiasFlat m n β (den e)
  | _, .batchOp (N := N) op e => batchMap N (denOp op) (den e)
  | _, .bnBatchF (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ β e =>
      bnBatchLA N oc h w ε γ β (den e)
  | _, .bnBatchBack (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      bnBatchTensor4_grad_input N oc h w ε γ x (den e)
  | _, .convBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).backward (fun _ => 0) dy) (den e)
  | _, .convStridedBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (flatConvStride2_has_vjp W b).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (hasVJP3_to_hasVJP (depthwise_has_vjp3 W b)).backward (fun _ => 0) dy) (den e)
  | _, .depthwiseStridedBackBatched (N := N) (c := c) (h := h) (w := w) _ W b e =>
      batchMap N (fun dy => (depthwiseStride2Flat_has_vjp W b).backward (fun _ => 0) dy) (den e)
  | _, .bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) _ _ _ ε γ x e =>
      fun i => ∑ k, if i = (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) k then
        bnBatchTensor4_grad_input N oc h w ε γ
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm) x)
          (fun i' => ∑ k', if i' = (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w))) k'
                           then den e k' else 0) k
        else 0
  | _, .seBackBatched (h := h) (w := w) _ _ _ _ _ W₁ b₁ W₂ b₂ v e =>
      fun idx =>
        (seBlockFull_has_vjp (h := h) (w := w) W₁ b₁ W₂ b₂).backward
          (Mat.unflatten v (finProdFinEquiv.symm idx).1)
          (Mat.unflatten (den e) (finProdFinEquiv.symm idx).1)
          (finProdFinEquiv.symm idx).2
  | _, .seReduceB (N := N) (c := c) (h := h) (w := w) _ x e =>
      -- the SE gate cotangent: per example, the broadcast-adjoint of `x ⊙ dy`
      -- (`broadcastFlat_has_vjp.backward` = sum each channel's spatial Hadamard).
      fun idx =>
        ∑ q : Fin (c * h * w),
          if flatChannel c h w q = (finProdFinEquiv.symm idx).2 then
            batchSlice N (c * h * w) x (finProdFinEquiv.symm idx).1 q
              * batchSlice N (c * h * w) (den e) (finProdFinEquiv.symm idx).1 q
          else 0
  | _, .gapBackBatched (N := N) (c := c) (h := h) (w := w) e =>
      batchMap N (fun dgap => (globalAvgPoolFlat_has_vjp c h w).backward (fun _ => 0) dgap) (den e)

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

-- Batched-lift faithfulness: `den` of each batched token = `batchMap N` of the
-- proven per-example op (rfl, since `denOp` returns that op directly), and the
-- true-batch-norm token denotes `bnBatchLA` (= the proven `bnBatchTensor4`).
@[simp] theorem den_batchOp_conv {N ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (bias : Vec oc) (e : SHlo (N * (ic*h*w))) :
    den (.batchOp (N := N) (.conv (h := h) (w := w) wN bN W bias) e)
      = batchMap N (flatConv W bias) (den e) := rfl
@[simp] theorem den_batchOp_convStrided {N ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (bias : Vec oc) (e : SHlo (N * (ic*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.convStrided (h := h) (w := w) wN bN W bias) e)
      = batchMap N (flatConvStride2 W bias) (den e) := rfl
@[simp] theorem den_batchOp_depthwise {N c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (bias : Vec c) (e : SHlo (N * (c*h*w))) :
    den (.batchOp (N := N) (.depthwise (h := h) (w := w) wN bN W bias) e)
      = batchMap N (depthwiseFlat W bias) (den e) := rfl
@[simp] theorem den_batchOp_depthwiseStrided {N c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (bias : Vec c) (e : SHlo (N * (c*(2*h)*(2*w)))) :
    den (.batchOp (N := N) (.depthwiseStrided (h := h) (w := w) wN bN W bias) e)
      = batchMap N (depthwiseStride2Flat W bias) (den e) := rfl
@[simp] theorem den_batchOp_dense {N a c : Nat} (wN bN : String)
    (W : Mat a c) (bias : Vec c) (e : SHlo (N * a)) :
    den (.batchOp (N := N) (.dense wN bN W bias) e)
      = batchMap N (dense W bias) (den e) := rfl
@[simp] theorem den_batchOp_gap {N c h w : Nat} (e : SHlo (N * (c*h*w))) :
    den (.batchOp (N := N) (.gap (c := c) (h := h) (w := w)) e)
      = batchMap N (globalAvgPoolFlat c h w) (den e) := rfl
@[simp] theorem den_batchOp_seBlock {N c h w r : Nat} (w1 b1 w2 b2 : String)
    (W₁ : Mat c r) (β₁ : Vec r) (W₂ : Mat r c) (β₂ : Vec c) (e : SHlo (N * (c*h*w))) :
    den (.batchOp (N := N) (.seBlock (h := h) (w := w) w1 b1 w2 b2 W₁ β₁ W₂ β₂) e)
      = batchMap N (seBlockFull (h := h) (w := w) W₁ β₁ W₂ β₂) (den e) := rfl
@[simp] theorem den_bnBatchF {N oc h w : Nat} (gN bN es : String) (ε : ℝ) (γ β : Vec oc)
    (e : SHlo (N * (oc*h*w))) :
    den (.bnBatchF gN bN es ε γ β e) = bnBatchLA N oc h w ε γ β (den e) := rfl

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

/-- **Stride-4 conv forward faithfulness.** The `window_strides=[4,4]`
    `stablehlo.convolution` (the ConvNeXt 4×4/s4 patchify stem) denotes the proven
    `flatConvStride4` (= decimate ∘ decimate ∘ stride-1 conv, StridedConv.lean). -/
@[simp] theorem flatConvStride4F_faithful {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic*(2*(2*h))*(2*(2*w)))) :
    den (.flatConvStride4F wN bN W b e) = flatConvStride4 W b (den e) := rfl

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

/-- **GELU forward faithfulness.** The tanh-approximation graph
    `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` denotes the proven `gelu`
    (LayerNorm.lean). Smooth everywhere; no kink, no smoothness hypothesis.
    (`rfl`, so kept out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem geluF_faithful {n : Nat} (e : SHlo n) :
    den (.geluF e) = gelu n (den e) := rfl

/-- **Layer-scale faithfulness.** The per-element multiply `γ ⊙ x` denotes the proven
    `layerScale` (ConvNeXt.lean). (`rfl`.) -/
@[simp] theorem layerScaleF_faithful {n : Nat} (γN : String) (γ : Vec n) (e : SHlo n) :
    den (.layerScaleF γN γ e) = layerScale γ (den e) := rfl

/-- **Per-channel layer-scale faithfulness.** The `[c]`-broadcast multiply denotes
    the proven `layerScale` at the channel-expanded vector. (`rfl`.) -/
@[simp] theorem layerScaleChF_faithful {c h w : Nat} (γN : String) (γ : Vec c)
    (e : SHlo (c*h*w)) :
    den (.layerScaleChF γN γ e) = layerScale (fun k => γ (chanIdx c h w k)) (den e) := rfl

/-- **GELU input-VJP faithfulness.** The closed-form `dy ⊙ gelu'(x)` graph
    (recomputing `tanh(u(x))` from the saved pre-activation `x`) denotes the proven
    GLOBAL `gelu_has_vjp` backward (`dy ⊙ geluScalarDeriv x`; GELU is smooth
    everywhere, so this is a global VJP — no smoothness hypothesis). -/
theorem geluBack_faithful {n : Nat} (xN : String) (x : Vec n) (e : SHlo n) :
    den (.geluBack xN x e) = (gelu_has_vjp n).backward x (den e) := rfl

/-- **Row-softmax forward faithfulness.** The per-row `exp / reduce[last] / divide`
    graph denotes `rowSoftmaxFlat` (= flattened `rowSoftmax`, Attention.lean). Plain
    exp/sum, no max-shift (matches the proven `softmax`). Smooth everywhere.
    (`rfl`, so kept out of the axiom audit — `roundtrip` covers it structurally.) -/
@[simp] theorem softmaxRowF_faithful {m n : Nat} (e : SHlo (m*n)) :
    den (.softmaxRowF e) = rowSoftmaxFlat m n (den e) := rfl

/-- **Row-softmax input-VJP faithfulness.** The per-row closed-form
    `p ⊙ (dy − ⟨p,dy⟩)` graph (recomputing `p` from the saved pre-softmax scores)
    denotes `rowSoftmaxBackFlat` (= flattened `rowSoftmax_has_vjp_mat.backward`).
    Softmax is smooth, so this is a global VJP — no smoothness hypothesis. -/
theorem softmaxRowBack_faithful {m n : Nat} (xN : String) (preAct : Vec (m*n)) (e : SHlo (m*n)) :
    den (.softmaxRowBack xN preAct e) = rowSoftmaxBackFlat m n preAct (den e) := rfl

/-- **Matrix-multiply faithfulness.** The reshape + batching-dim-0 `dot_general`
    (contracting `[2] x [1]`) + reshape graph denotes `matMulFlat` (= the flattened
    `Mat.mul`). Bilinear; the attention backwards reuse this token (`dA = dC·Bᵀ`,
    `dB = Aᵀ·dC`). (`rfl`, so kept out of the axiom audit — `roundtrip` covers it
    structurally.) -/
@[simp] theorem matmulF_faithful {m k n : Nat} (a : SHlo (m*k)) (b : SHlo (k*n)) :
    den (.matmulF a b) = matMulFlat m k n (den a) (den b) := rfl

/-- **Transpose faithfulness.** `stablehlo.transpose dims=[0,2,1]` (after reshape
    to rank 3) denotes `transposeFlat` (= the flattened `Mat.transpose`). (`rfl`.) -/
@[simp] theorem transposeF_faithful {m n : Nat} (e : SHlo (m*n)) :
    den (.transposeF e) = transposeFlat m n (den e) := rfl

/-- **Scalar-scale faithfulness.** The splat-constant `stablehlo.multiply` denotes
    pointwise `s · x` — SDPA's `1/√d`. (`rfl`; the `sStr ↔ s` literal agreement is
    the audited lexical boundary, like `bnF`'s `epsStr`.) -/
@[simp] theorem scaleF_faithful {n : Nat} (sN : String) (s : ℝ) (e : SHlo n) :
    den (.scaleF sN s e) = fun i => s * den e i := rfl

/-- **Row-LayerNorm forward faithfulness.** The rank-3 reduce[2]/normalize/affine
    graph (per token row, scalar γ/β) denotes `rowLNFlat` (rowwise `bnForward` =
    rowwise `layerNormForward`, definitionally). (`rfl`.) -/
@[simp] theorem lnRowF_faithful {m n : Nat} (gN bN es : String) (ε γ β : ℝ) (e : SHlo (m*n)) :
    den (.lnRowF gN bN es ε γ β e) = rowLNFlat m n ε γ β (den e) := rfl

/-- **Row-LayerNorm input-VJP faithfulness.** The per-row consolidated three-term
    graph (recomputing x̂/istd from the saved pre-LN input, reductions over the row
    axis) denotes `rowLNBackFlat` (rowwise `bn_grad_input` — faithful to the
    pdiv-Jacobian per row under `0 < ε`, `bn_input_grad_correct`). -/
theorem lnRowBack_faithful {m n : Nat} (gN xN es : String) (ε γ : ℝ) (x : Vec (m*n))
    (e : SHlo (m*n)) :
    den (.lnRowBack gN xN es ε γ x e) = rowLNBackFlat m n ε γ x (den e) := rfl

/-- **Per-token dense forward faithfulness.** The `dot_general [2] x [0]` + bias
    broadcast `dims=[2]` graph denotes `rowDenseFlat` (rowwise `dense W b`). (`rfl`.) -/
@[simp] theorem denseRowF_faithful {N a c : Nat} (wN bN : String) (W : Mat a c) (b : Vec c)
    (e : SHlo (N*a)) :
    den (.denseRowF wN bN W b e) = rowDenseFlat N a c W b (den e) := rfl

/-- **Per-token dense input-VJP faithfulness.** The `dot_general [2] x [1]` graph
    (dy against W's output axis) denotes `rowDenseBackFlat` (rowwise `Mat.mulVec W`
    = the proven `dense_has_vjp` backward; dense is affine — global VJP). -/
theorem denseRowBack_faithful {N a c : Nat} (wN : String) (W : Mat a c) (e : SHlo (N*c)) :
    den (.denseRowBack wN W e) = rowDenseBackFlat N a c W (den e) := rfl

/-- **Patch-embedding faithfulness.** The stride-P VALID conv + channels-last
    flatten + CLS concatenate + position-embed add graph denotes `patchEmbedFlat`
    (the local re-spelling of the proven `patchEmbed_flat`; the tie is `rfl` in
    ViTFwdGraph). (`rfl`, coarse-token like `seBlock`.) -/
@[simp] theorem patchEmbedF_faithful {ic H W P N D : Nat} (wN bN cN pN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N+1) D) (e : SHlo (ic*H*W)) :
    den (.patchEmbedF wN bN cN pN Wc bc cls pos e)
      = patchEmbedFlat ic H W P N D Wc bc cls pos (den e) := rfl

/-- **Patch-embedding input-VJP faithfulness.** The reversed-kernel strided
    `conv_transpose` (on the patch-token rows of the `[N+1,D]` cotangent) denotes
    `patchEmbedBackFlat` (= the proven `patchEmbed_input_grad_formula`; the tie to
    `patchEmbed_flat_has_vjp.backward` is `rfl` in ViTBackB0). (`rfl`.) -/
@[simp] theorem patchEmbedBack_faithful {ic H W P N D : Nat} (wN : String)
    (Wc : Kernel4 D ic P P) (e : SHlo ((N+1)*D)) :
    den (.patchEmbedBack wN Wc e) = patchEmbedBackFlat ic H W P N D Wc (den e) := rfl

/-- **CLS-slice faithfulness.** The row-0 `stablehlo.slice` denotes `clsSliceFlat`
    (= the proven `cls_slice_flat`). (`rfl`.) -/
@[simp] theorem clsSliceF_faithful {N D : Nat} (e : SHlo ((N+1)*D)) :
    den (.clsSliceF e) = clsSliceFlat N D (den e) := rfl

/-- **CLS-pad faithfulness.** The zero-pad scatter-to-row-0 denotes `clsPadFlat`
    (= the proven `cls_slice_flat_has_vjp.backward`; linear — global VJP). (`rfl`.) -/
@[simp] theorem clsPadF_faithful {N D : Nat} (e : SHlo D) :
    den (.clsPadF (N := N) e) = clsPadFlat N D (den e) := rfl

/-- **Per-head slice faithfulness.** The feature-axis `stablehlo.slice` of head `h`'s
    contiguous column block denotes `headSliceFlat` (= `mhsa_layer`'s per-head column
    gather). Linear reindex. (`rfl`.) -/
@[simp] theorem headSliceF_faithful {N heads d : Nat} (h : Fin heads)
    (e : SHlo (N*(heads*d))) :
    den (.headSliceF h e) = headSliceFlat N heads d h (den e) := rfl

/-- **Per-head pad faithfulness.** The feature-axis zero-pad into head `h`'s column
    block denotes `headPadFlat` (the slice's VJP; summed over heads it is
    `mhsa_layer`'s concat). Linear. (`rfl`.) -/
@[simp] theorem headPadF_faithful {N heads d : Nat} (h : Fin heads) (e : SHlo (N*d)) :
    den (.headPadF h e) = headPadFlat N heads d h (den e) := rfl

/-- **Row-broadcast scale faithfulness.** The reshape + broadcast-γ-over-rows +
    multiply graph denotes `rowScaleFlat` (rowwise `layerScale γ`). Diagonal-linear —
    its own input-VJP, so the backward reuses this token on the cotangent. (`rfl`.) -/
@[simp] theorem rowScaleF_faithful {m n : Nat} (gN : String) (γ : Vec n) (e : SHlo (m*n)) :
    den (.rowScaleF gN γ e) = rowScaleFlat m n γ (den e) := rfl

/-- **Row-broadcast bias faithfulness.** The broadcast-β-over-rows + add graph denotes
    `rowBiasFlat`. Translation — identity input-VJP. (`rfl`.) -/
@[simp] theorem rowBiasF_faithful {m n : Nat} (bN : String) (β : Vec n) (e : SHlo (m*n)) :
    den (.rowBiasF bN β e) = rowBiasFlat m n β (den e) := rfl

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
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : SHlo nClasses :=
  denseF "%W7" "%b7" W₇ b₇
    (.reluF (denseF "%W6" "%b6" W₆ b₆
      (.reluF (denseF "%W5" "%b5" W₅ b₅
        (.maxPoolF (c := c2) (h := h) (w := w)
          (.reluF (.bnPerChannelF (oc := c2) (h := 2*h) (w := 2*w) "%g4" "%bt4" epsStr ε₄ γ₄ β₄
            (.flatConvF (h := 2*h) (w := 2*w) "%W4" "%b4" W₄ b₄
            (.reluF (.bnPerChannelF (oc := c2) (h := 2*h) (w := 2*w) "%g3" "%bt3" epsStr ε₃ γ₃ β₃
              (.flatConvF (h := 2*h) (w := 2*w) "%W3" "%b3" W₃ b₃
              (.maxPoolF (c := c1) (h := 2*h) (w := 2*w)
                (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g2" "%bt2" epsStr ε₂ γ₂ β₂
                  (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W2" "%b2" W₂ b₂
                  (.reluF (.bnPerChannelF (oc := c1) (h := 2*(2*h)) (w := 2*(2*w)) "%g1" "%bt1" epsStr ε₁ γ₁ β₁
                    (.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) "%W1" "%b1" W₁ b₁
                    (.operand "%x" x)))))))))))))))))))

/-- **BN-CIFAR forward faithfulness.** The forward graph denotes the proven
    `cifarCnnBnForward`. -/
theorem cifarBnFwdGraph_faithful {ic c1 c2 h w d1 nClasses kH kW : Nat} (epsStr : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) :
    den (cifarBnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ W₆ b₆ W₇ b₇ x)
      = cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ W₆ b₆ W₇ b₇ x := by
  simp only [cifarBnFwdGraph, cifarCnnBnForward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful,
             bnPerChannelF_faithful, den_operand]

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

/-- **Deeper (8-conv) CIFAR-CNN forward faithfulness.** The forward graph denotes the
    proven `cifarCnn8Forward`. -/
theorem cifar8FwdGraph_faithful {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) :
    den (cifar8FwdGraph W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
          W₉ b₉ Wa ba Wb bb x)
      = cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
          W₉ b₉ Wa ba Wb bb x := by
  simp only [cifar8FwdGraph, cifarCnn8Forward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful, den_operand]

/-- Whole **deeper (8-conv) BN-CIFAR forward** graph: each of the eight convs is followed
    by a per-channel `bnPerChannelF` before its ReLU. `epsStr` is the shared ε literal; the
    eight BN layers carry per-channel γ/β inputs `%g{i}`/`%bt{i}`. The 4-stage peer of
    `cifarBnFwdGraph`. -/
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

/-- **Deeper (8-conv) BN-CIFAR forward faithfulness.** The forward graph denotes the
    proven `cifarCnnBn8Forward`. -/
theorem cifar8BnFwdGraph_faithful {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat} (epsStr : String)
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
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) :
    den (cifar8BnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          W₉ b₉ Wa ba Wb bb x)
      = cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
          W₉ b₉ Wa ba Wb bb x := by
  simp only [cifar8BnFwdGraph, cifarCnnBn8Forward, Function.comp_apply,
             denseF_faithful, reluF_faithful, flatConvF_faithful, maxPoolF_faithful,
             bnPerChannelF_faithful, den_operand]

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

/-- Whole **MobileNetV2 forward** graph (representative, ch7 peer of `resnetFwdGraph`):
    stem (conv→bn→relu6) → skip inverted-residual `addV(invresBody, stem)` → no-skip
    inverted-residual → global-average-pool → dense. Each inverted-residual body is
    `bn∘conv(project) ∘ relu6∘bn∘depthwise ∘ relu6∘bn∘conv(expand)`; the skip's `addV`
    reuses the block-input subtree (linear bottleneck — no relu6 after the add). Uses the
    MobileNetV2 ops `relu6F`/`depthwiseF` (SAME-spatial representative; the stride-2
    `depthwiseStridedF`/`flatConvStridedF` of the full render are exercised at the op level,
    not assembled here — full strided graph deferred, see planning doc). `epsStr` = shared ε
    literal; each scalar BN carries γ/β SSA inputs `%g*`/`%bt*`. -/
def mobilenetv2FwdGraph
    {ic c mid₁ oc mid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ)
    (Wh : Mat oc nClasses) (bh : Vec nClasses)
    (x : Vec (ic*h*w)) : SHlo nClasses :=
  -- stem: relu6(bn(conv x))
  let stemOut : SHlo (c*h*w) :=
    .relu6F (.bnF "%gs" "%bts" epsStr εs γs βs
      (.flatConvF (h := h) (w := w) "%Ws" "%bs" Ws bs (.operand "%x" x)))
  -- block1 body (inverted residual, c→mid₁→c): project ∘ depthwise ∘ expand
  let b1Body : SHlo (c*h*w) :=
    .bnF "%gp1" "%btp1" epsStr p₁ gp₁ bp1
      (.flatConvF (h := h) (w := w) "%Wp1" "%bp1" Wp₁ bp₁
        (.relu6F (.bnF "%gd1" "%btd1" epsStr d₁ gd₁ bd1
          (.depthwiseF (h := h) (w := w) "%Wd1" "%bd1" Wd₁ bd₁
            (.relu6F (.bnF "%ge1" "%bte1" epsStr e₁ ge₁ be1
              (.flatConvF (h := h) (w := w) "%We1" "%be1" We₁ be₁ stemOut)))))))
  -- block1 (skip): linear-bottleneck residual, no relu6 after the add
  let b1Out : SHlo (c*h*w) := .addV b1Body stemOut
  -- block2 body (inverted residual, c→mid₂→oc, no skip)
  let b2Out : SHlo (oc*h*w) :=
    .bnF "%gp2" "%btp2" epsStr p₂ gp₂ bp2
      (.flatConvF (h := h) (w := w) "%Wp2" "%bp2" Wp₂ bp₂
        (.relu6F (.bnF "%gd2" "%btd2" epsStr d₂ gd₂ bd2
          (.depthwiseF (h := h) (w := w) "%Wd2" "%bd2" Wd₂ bd₂
            (.relu6F (.bnF "%ge2" "%bte2" epsStr e₂ ge₂ be2
              (.flatConvF (h := h) (w := w) "%We2" "%be2" We₂ be₂ b1Out)))))))
  denseF "%Wh" "%bh" Wh bh (.gapF (c := oc) (h := h) (w := w) b2Out)

/-- **MobileNetV2 forward faithfulness.** The representative forward graph denotes the
    proven `mobilenetv2Forward` (whose end-to-end VJP at a smooth point is
    `mobilenetv2_has_vjp_at`). The skip `addV` denotes the `+` of `residual`/`biPath`;
    the inverted-residual body's `bn/conv/depthwise/relu6` ops denote
    `invresBody = ivProject ∘ ivDepthwise ∘ ivExpand`. ch7 peer of `resnetFwdGraph_faithful`. -/
theorem mobilenetv2FwdGraph_faithful
    {ic c mid₁ oc mid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    (epsStr : String)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ)
    (Wh : Mat oc nClasses) (bh : Vec nClasses)
    (x : Vec (ic*h*w)) :
    den (mobilenetv2FwdGraph epsStr Ws bs εs γs βs
          We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1
          We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 Wh bh x)
      = mobilenetv2Forward Ws bs εs γs βs
          We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1
          We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 Wh bh x := by
  simp only [mobilenetv2FwdGraph, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnF_faithful, flatConvF_faithful, depthwiseF_faithful, den_addV, den_operand]
  unfold mobilenetv2Forward invresBody ivExpand ivDepthwise ivProject residual biPath
  simp only [Function.comp_apply]

/-- Whole **MobileNetV2 forward** graph at the FULL ch7 render dims (3×224² → 7×7×64):
    strided stem (`flatConvStridedF`, 224→112) → 6 inverted-residual blocks (`b1/b3/b5/b6`
    stride-2 downsample via `depthwiseStridedF`, `b2/b4` stride-1 SAME with an `addV` skip)
    → 1×1 conv-bn-relu6 head → global-avg-pool → dense. Concrete (not symbolic) peer of
    `mobilenetv2FwdGraph`, tied to the *full* forward `mobilenetv2Forward_full`. Scalar BN. -/
def mobilenetv2FwdGraphFull
    (epsStr : String)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) : SHlo 10 :=
  let stemOut : SHlo (16 * 112 * 112) :=
    .relu6F (.bnF "%gs" "%bts" epsStr εs γs βs
      (.flatConvStridedF (h := 112) (w := 112) "%Ws" "%bs" Ws bs (.operand "%x" x)))
  let b1Out : SHlo (24 * 56 * 56) :=
    .bnF "%gp1" "%btp1" epsStr εp1 γp1 βp1
      (.flatConvF (h := 56) (w := 56) "%Wp1" "%bp1" Wp1 bp1
        (.relu6F (.bnF "%gd1" "%btd1" epsStr εd1 γd1 βd1
          (.depthwiseStridedF (h := 56) (w := 56) "%Wd1" "%bd1" Wd1 bd1
            (.relu6F (.bnF "%ge1" "%bte1" epsStr εe1 γe1 βe1
              (.flatConvF (h := 112) (w := 112) "%We1" "%be1" We1 be1 stemOut)))))))
  let b2Out : SHlo (24 * 56 * 56) :=
    .addV (.bnF "%gp2" "%btp2" epsStr εp2 γp2 βp2
      (.flatConvF (h := 56) (w := 56) "%Wp2" "%bp2" Wp2 bp2
        (.relu6F (.bnF "%gd2" "%btd2" epsStr εd2 γd2 βd2
          (.depthwiseF (h := 56) (w := 56) "%Wd2" "%bd2" Wd2 bd2
            (.relu6F (.bnF "%ge2" "%bte2" epsStr εe2 γe2 βe2
              (.flatConvF (h := 56) (w := 56) "%We2" "%be2" We2 be2 b1Out)))))))) b1Out
  let b3Out : SHlo (32 * 28 * 28) :=
    .bnF "%gp3" "%btp3" epsStr εp3 γp3 βp3
      (.flatConvF (h := 28) (w := 28) "%Wp3" "%bp3" Wp3 bp3
        (.relu6F (.bnF "%gd3" "%btd3" epsStr εd3 γd3 βd3
          (.depthwiseStridedF (h := 28) (w := 28) "%Wd3" "%bd3" Wd3 bd3
            (.relu6F (.bnF "%ge3" "%bte3" epsStr εe3 γe3 βe3
              (.flatConvF (h := 56) (w := 56) "%We3" "%be3" We3 be3 b2Out)))))))
  let b4Out : SHlo (32 * 28 * 28) :=
    .addV (.bnF "%gp4" "%btp4" epsStr εp4 γp4 βp4
      (.flatConvF (h := 28) (w := 28) "%Wp4" "%bp4" Wp4 bp4
        (.relu6F (.bnF "%gd4" "%btd4" epsStr εd4 γd4 βd4
          (.depthwiseF (h := 28) (w := 28) "%Wd4" "%bd4" Wd4 bd4
            (.relu6F (.bnF "%ge4" "%bte4" epsStr εe4 γe4 βe4
              (.flatConvF (h := 28) (w := 28) "%We4" "%be4" We4 be4 b3Out)))))))) b3Out
  let b5Out : SHlo (64 * 14 * 14) :=
    .bnF "%gp5" "%btp5" epsStr εp5 γp5 βp5
      (.flatConvF (h := 14) (w := 14) "%Wp5" "%bp5" Wp5 bp5
        (.relu6F (.bnF "%gd5" "%btd5" epsStr εd5 γd5 βd5
          (.depthwiseStridedF (h := 14) (w := 14) "%Wd5" "%bd5" Wd5 bd5
            (.relu6F (.bnF "%ge5" "%bte5" epsStr εe5 γe5 βe5
              (.flatConvF (h := 28) (w := 28) "%We5" "%be5" We5 be5 b4Out)))))))
  let b6Out : SHlo (64 * 7 * 7) :=
    .bnF "%gp6" "%btp6" epsStr εp6 γp6 βp6
      (.flatConvF (h := 7) (w := 7) "%Wp6" "%bp6" Wp6 bp6
        (.relu6F (.bnF "%gd6" "%btd6" epsStr εd6 γd6 βd6
          (.depthwiseStridedF (h := 7) (w := 7) "%Wd6" "%bd6" Wd6 bd6
            (.relu6F (.bnF "%ge6" "%bte6" epsStr εe6 γe6 βe6
              (.flatConvF (h := 14) (w := 14) "%We6" "%be6" We6 be6 b5Out)))))))
  let headOut : SHlo (128 * 7 * 7) :=
    .relu6F (.bnF "%gh" "%bth" epsStr εh γh βh
      (.flatConvF (h := 7) (w := 7) "%Wh" "%bh" Wh bh b6Out))
  denseF "%Wfc" "%bfc" Wfc bfc (.gapF (c := 128) (h := 7) (w := 7) headOut)

/-- **Full MobileNetV2 forward faithfulness.** The full strided render graph denotes the
    proven `mobilenetv2Forward_full` (the spec's real net, tied by `mobilenetv2Verified_denote_eq`).
    `simp`-based — so unlike the VJP fold it does not hit the concrete-dim `isDefEq` wall. -/
theorem mobilenetv2FwdGraphFull_faithful
    (epsStr : String)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphFull epsStr Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x)
      = mobilenetv2Forward_full Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x := by
  simp only [mobilenetv2FwdGraphFull, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnF_faithful, flatConvF_faithful, flatConvStridedF_faithful, depthwiseF_faithful,
             depthwiseStridedF_faithful, den_addV, den_operand]
  unfold mobilenetv2Forward_full invresBodyStrided invresBody ivExpand ivDepthwiseStrided
         ivDepthwise ivProject residual biPath
  simp only [Function.comp_apply]


/-- Whole **ConvNeXt forward** graph (representative, ch9 peer of `resnetFwdGraph`): 1×1
    patchify conv → stem-LN → 2 residual ConvNeXt blocks (depthwise → LN → 1×1 expand →
    GELU → 1×1 project → layerScale, then `addV` skip) → GAP → head-LN → dense. Scalar LN
    (`= bnForward`, via `bnF`); uses `geluF` + the new `layerScaleF`. Denotes the proven
    `convNextForward`. -/
def convNextFwdGraph {ic c cExp h w kH kW nClasses : Nat}
    (epsStr : String)
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) : SHlo nClasses :=
  let patchOut : SHlo (c * h * w) :=
    .flatConvF (h := h) (w := w) "%Wst" "%bst" Wst bst (.operand "%x" x)
  let stemLn : SHlo (c * h * w) :=
    .bnF "%gst" "%btst" epsStr εst γst βst patchOut
  let b1Body : SHlo (c * h * w) :=
    .layerScaleF "%gls1" γls₁
      (.flatConvF (h := h) (w := w) "%Wpr1" "%bpr1" Wpr₁ bpr₁
        (.geluF (.flatConvF (h := h) (w := w) "%Wex1" "%bex1" Wex₁ bex₁
          (.bnF "%gn1" "%btn1" epsStr εn₁ γn₁ βn₁
            (.depthwiseF (h := h) (w := w) "%Wdw1" "%bdw1" Wdw₁ bdw₁ stemLn)))))
  let b1Out : SHlo (c * h * w) := .addV b1Body stemLn
  let b2Body : SHlo (c * h * w) :=
    .layerScaleF "%gls2" γls₂
      (.flatConvF (h := h) (w := w) "%Wpr2" "%bpr2" Wpr₂ bpr₂
        (.geluF (.flatConvF (h := h) (w := w) "%Wex2" "%bex2" Wex₂ bex₂
          (.bnF "%gn2" "%btn2" epsStr εn₂ γn₂ βn₂
            (.depthwiseF (h := h) (w := w) "%Wdw2" "%bdw2" Wdw₂ bdw₂ b1Out)))))
  let b2Out : SHlo (c * h * w) := .addV b2Body b1Out
  let headLn : SHlo c :=
    .bnF "%ghd" "%bthd" epsStr εhd γhd βhd (.gapF (c := c) (h := h) (w := w) b2Out)
  denseF "%Wd" "%bd" Wd bd headLn

/-- **ConvNeXt forward faithfulness.** The representative forward graph denotes the proven
    `convNextForward`. Scalar LN (`layerNormForward = bnForward`); `simp`-based. -/
theorem convNextFwdGraph_faithful {ic c cExp h w kH kW nClasses : Nat}
    (epsStr : String)
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) :
    den (convNextFwdGraph epsStr Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd x) = convNextForward Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd x := by
  simp only [convNextFwdGraph, denseF_faithful, gapF_faithful, geluF_faithful, bnF_faithful,
             flatConvF_faithful, depthwiseF_faithful, layerScaleF_faithful, den_addV, den_operand]
  unfold convNextForward convNextBlock convNextBlockBody residual biPath layerNormForward
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
  | weightSgd  (xName wName lrStr : String) (m n : Nat) : Raw → Raw
  | biasSgd    (bName lrStr : String) (n : Nat)         : Raw → Raw
  | convWeightSgd (xName wName lrStr : String) (ic oc h w kH kW : Nat) : Raw → Raw
  | convBiasSgd   (bName lrStr : String) (oc h w : Nat)               : Raw → Raw
  | bnGammaSgd    (gName vName epsStr lrStr : String) (oc h w : Nat)  : Raw → Raw
  | bnBetaSgd     (bName lrStr : String) (oc h w : Nat)               : Raw → Raw
  | layerScaleChGammaSgd (gName xName lrStr : String) (c h w : Nat)   : Raw → Raw
  | lnGammaSgd    (gName xName epsStr lrStr : String) (n : Nat)       : Raw → Raw
  | lnBetaSgd     (bName lrStr : String) (n : Nat)                    : Raw → Raw
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
  | gapBack    (c h w : Nat)               : Raw → Raw
  | broadcastBack (c h w : Nat)            : Raw → Raw
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | convStridedWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Raw → Raw
  | depthwiseWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Raw → Raw
  | depthwiseStridedWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Raw → Raw
  | flatConvStride4F (w b : String) (ic oc h w' kH kW : Nat) : Raw → Raw
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
  | geluF      (n : Nat)                   : Raw → Raw
  | geluBack   (x : String) (n : Nat)      : Raw → Raw
  | layerScaleF (γ : String) (n : Nat)     : Raw → Raw
  | layerScaleChF (γ : String) (c h w : Nat) : Raw → Raw
  | softmaxRowF    (m n : Nat)             : Raw → Raw
  | softmaxRowBack (x : String) (m n : Nat) : Raw → Raw
  | matmulF    (m k n : Nat)               : Raw → Raw → Raw
  | transposeF (m n : Nat)                 : Raw → Raw
  | scaleF     (s : String) (n : Nat)      : Raw → Raw
  | lnRowF     (g b eps : String) (m n : Nat) : Raw → Raw
  | lnRowBack  (g x eps : String) (m n : Nat) : Raw → Raw
  | denseRowF  (w b : String) (N a c : Nat) : Raw → Raw
  | denseRowBack (w : String) (N a c : Nat) : Raw → Raw
  | patchEmbedF (w b cls pos : String) (ic H W P N D : Nat) : Raw → Raw
  | clsSliceF  (N D : Nat)                 : Raw → Raw
  | clsPadF    (N D : Nat)                 : Raw → Raw
  | headSliceF (N heads d hIdx : Nat)      : Raw → Raw
  | headPadF   (N heads d hIdx : Nat)      : Raw → Raw
  | rowScaleF  (g : String) (m n : Nat)    : Raw → Raw
  | rowBiasF   (b : String) (m n : Nat)    : Raw → Raw
  -- EfficientNet batched ops (`batchOp`/`bnBatchF`/the batched backward ops): the
  -- renderable skeleton keeps a tag discriminating the op, the SSA names the emit
  -- references (weight/bias/BN-input/γ/ε/SE-input names), and shape info. The tag
  -- is the BatchableOp variant ("conv"/"depthwise"/"seBlock"/…) for forward ops or
  -- the backward op name; this is what lets `emitTok` reconstruct real StableHLO.
  | batched    (tag : String) (names : List String) (info : List Nat) : Raw → Raw
deriving DecidableEq, Repr, Inhabited

/-- The `(tag, names, info)` skeleton descriptor of a batched per-example op — the
    discriminator + the SSA names the emit references + the shape dims. Keeps the
    `batchOp` skel one line and isolates the 7-variant match into a pure function. -/
def batchOpDescr {a b : Nat} (N : Nat) : BatchableOp a b → (String × List String × List Nat)
  | .conv (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("conv", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .convStrided (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("convStrided", [wN, bN], [N, ic, oc, h, w, kH, kW])
  | .depthwise (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwise", [wN, bN], [N, c, h, w, kH, kW])
  | .depthwiseStrided (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ =>
      ("depthwiseStrided", [wN, bN], [N, c, h, w, kH, kW])
  | .dense (c := c) wN bN _ _ => ("dense", [wN, bN], [N, a, c])
  | .gap (c := c) (h := h) (w := w) => ("gap", [], [N, c, h, w])
  | .seBlock (c := c) (h := h) (w := w) (r := r) w1 b1 w2 b2 _ _ _ _ =>
      ("seBlock", [w1, b1, w2, b2], [N, c, h, w, r])

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
  | _, .weightSgd (m := m) (n := n) xN wN lrS _ _ _ e => .weightSgd xN wN lrS m n (skel e)
  | k, .biasSgd bN lrS _ _ e  => .biasSgd bN lrS k (skel e)
  | _, .convWeightSgd (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .convWeightSgd xN wN lrS ic oc h w kH kW (skel e)
  | _, .convBiasSgd (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS oc h w (skel e)
  | _, .bnGammaSgd (oc := oc) (h := h) (w := w) gN vN es lrS _ _ _ _ e =>
      .bnGammaSgd gN vN es lrS oc h w (skel e)
  | _, .bnBetaSgd (oc := oc) (h := h) (w := w) bN lrS _ _ e =>
      .bnBetaSgd bN lrS oc h w (skel e)
  | _, .layerScaleChGammaSgd (c := c) (h := h) (w := w) gN xN lrS _ _ _ e =>
      .layerScaleChGammaSgd gN xN lrS c h w (skel e)
  | _, .lnGammaSgd (n := n) gN xN es lrS _ _ _ _ e =>
      .lnGammaSgd gN xN es lrS n (skel e)
  | _, .lnBetaSgd (n := n) bN lrS _ _ e =>
      .lnBetaSgd bN lrS n (skel e)
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
  | _, .gapBack (c := c) (h := h) (w := w) e => .gapBack c h w (skel e)
  | _, .broadcastBack (c := c) (h := h) (w := w) e => .broadcastBack c h w (skel e)
  | _, .flatConvStridedF (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStridedF wN bN ic oc h w kH kW (skel e)
  | _, .convStridedBack (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ _ e =>
      .convStridedBack wN ic oc h w kH kW (skel e)
  | _, .convStridedWeightSgd (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .convStridedWeightSgd xN wN lrS ic oc h w kH kW (skel e)
  | _, .convStridedBiasSgd (oc := oc) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS oc h w (skel e)
  | _, .depthwiseWeightSgd (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .depthwiseWeightSgd xN wN lrS c h w kH kW (skel e)
  | _, .depthwiseBiasSgd (c := c) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS c h w (skel e)
  | _, .depthwiseStridedWeightSgd (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .depthwiseStridedWeightSgd xN wN lrS c h w kH kW (skel e)
  | _, .depthwiseStridedBiasSgd (c := c) (h := h) (w := w) bN lrS _ _ _ _ e =>
      .convBiasSgd bN lrS c h w (skel e)
  | _, .flatConvStride4F (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN bN _ _ e =>
      .flatConvStride4F wN bN ic oc h w kH kW (skel e)
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
  | k, .geluF e              => .geluF k (skel e)
  | k, .geluBack x _ e       => .geluBack x k (skel e)
  | k, .layerScaleF γN _ e   => .layerScaleF γN k (skel e)
  | _, .layerScaleChF (c := c) (h := h) (w := w) γN _ e => .layerScaleChF γN c h w (skel e)
  | _, .softmaxRowF (m := m) (n := n) e => .softmaxRowF m n (skel e)
  | _, .softmaxRowBack (m := m) (n := n) x _ e => .softmaxRowBack x m n (skel e)
  | _, .matmulF (m := m) (k := k) (n := n) a b => .matmulF m k n (skel a) (skel b)
  | _, .transposeF (m := m) (n := n) e => .transposeF m n (skel e)
  | k, .scaleF sStr _ e => .scaleF sStr k (skel e)
  | _, .lnRowF (m := m) (n := n) gN bN es _ _ _ e => .lnRowF gN bN es m n (skel e)
  | _, .lnRowBack (m := m) (n := n) gN xN es _ _ _ e => .lnRowBack gN xN es m n (skel e)
  | _, .denseRowF (N := N) (a := a) (c := c) wN bN _ _ e => .denseRowF wN bN N a c (skel e)
  | _, .denseRowBack (N := N) (a := a) (c := c) wN _ e => .denseRowBack wN N a c (skel e)
  | _, .patchEmbedF (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) wN bN cN pN _ _ _ _ e =>
      .patchEmbedF wN bN cN pN ic H W P N D (skel e)
  | _, .patchEmbedBack (ic := ic) (H := H) (W := W) (P := P) (N := N) (D := D) _ _ e =>
      .batched "patchEmbedBack" [] [ic, H, W, P, N, D] (skel e)
  | _, .clsSliceF (N := N) (D := D) e => .clsSliceF N D (skel e)
  | _, .clsPadF (N := N) (D := D) e => .clsPadF N D (skel e)
  | _, .headSliceF (N := N) (heads := heads) (d := d) h e => .headSliceF N heads d h.val (skel e)
  | _, .headPadF (N := N) (heads := heads) (d := d) h e => .headPadF N heads d h.val (skel e)
  | _, .rowScaleF (m := m) (n := n) gN _ e => .rowScaleF gN m n (skel e)
  | _, .rowBiasF (m := m) (n := n) bN _ e => .rowBiasF bN m n (skel e)
  | _, .batchOp (N := N) op e =>
      let (tag, nms, inf) := batchOpDescr N op; .batched tag nms inf (skel e)
  | _, .bnBatchF (N := N) (oc := oc) (h := h) (w := w) gN bN es _ _ _ e =>
      .batched "bnBatch" [gN, bN, es] [N, oc, h, w] (skel e)
  | _, .bnBatchBack (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .batched "bnBatchBack" [gN, xN, es] [N, oc, h, w] (skel e)
  | _, .convBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "convBackBatched" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedBackBatched (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "convStridedBackBatched" [wN] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .depthwiseBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedBackBatched (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) wN _ _ e =>
      .batched "depthwiseStridedBackBatched" [wN] [N, c, h, w, kH, kW] (skel e)
  | _, .bnBatchLABack (N := N) (oc := oc) (h := h) (w := w) gN xN es _ _ _ e =>
      .batched "bnBatchLABack" [gN, xN, es] [N, oc, h, w] (skel e)
  | _, .seBackBatched (N := N) (c := c) (h := h) (w := w) (r := r) w1 b1 w2 b2 vN _ _ _ _ _ e =>
      .batched "seBackBatched" [w1, b1, w2, b2, vN] [N, c, h, w, r] (skel e)
  | _, .seReduceB (N := N) (c := c) (h := h) (w := w) xN _ e =>
      .batched "seReduceB" [xN] [N, c, h, w] (skel e)
  | _, .gapBackBatched (N := N) (c := c) (h := h) (w := w) e =>
      .batched "gapBackBatched" [] [N, c, h, w] (skel e)
  | _, .bnGammaSgdB (N := N) (oc := oc) (h := h) (w := w) gN vN es lrS _ _ _ _ e =>
      .batched "bnGammaSgd" [gN, vN, es, lrS] [N, oc, h, w] (skel e)
  | _, .bnBetaSgdB (N := N) (oc := oc) (h := h) (w := w) bN lrS _ _ e =>
      .batched "bnBetaSgd" [bN, lrS] [N, oc, h, w] (skel e)
  | _, .denseWeightSgdB (N := N) (a := a) (c := c) xN wN lrS _ _ _ e =>
      .batched "denseWeightSgd" [xN, wN, lrS] [N, a, c] (skel e)
  | _, .denseBiasSgdB (N := N) (c := c) bN lrS _ _ e =>
      .batched "denseBiasSgd" [bN, lrS] [N, c] (skel e)
  | _, .convWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .convStridedWeightSgdB (N := N) (ic := ic) (oc := oc) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "convStridedWeightSgd" [xN, wN, lrS] [N, ic, oc, h, w, kH, kW] (skel e)
  | _, .depthwiseWeightSgdB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "depthwiseWeightSgd" [xN, wN, lrS] [N, c, h, w, kH, kW] (skel e)
  | _, .depthwiseStridedWeightSgdB (N := N) (c := c) (h := h) (w := w) (kH := kH) (kW := kW) xN wN lrS _ _ _ _ e =>
      .batched "depthwiseStridedWeightSgd" [xN, wN, lrS] [N, c, h, w, kH, kW] (skel e)

/-- One serialized token: an opcode with shapes/names; operands are positional. -/
inductive Tok where
  | operand    (name : String) (n : Nat)  : Tok
  | dotIn      (w : String) (m n : Nat)    : Tok
  | dotOut     (w : String) (m n : Nat)    : Tok
  | addBcast   (b : String) (n : Nat)      : Tok
  | expe       (n : Nat)                   : Tok
  | softmaxDiv (n : Nat)                   : Tok
  | sub        (n : Nat)                   : Tok
  | weightSgd  (xName wName lrStr : String) (m n : Nat) : Tok
  | biasSgd    (bName lrStr : String) (n : Nat)         : Tok
  | convWeightSgd (xName wName lrStr : String) (ic oc h w kH kW : Nat) : Tok
  | convBiasSgd   (bName lrStr : String) (oc h w : Nat)               : Tok
  | bnGammaSgd    (gName vName epsStr lrStr : String) (oc h w : Nat)  : Tok
  | bnBetaSgd     (bName lrStr : String) (oc h w : Nat)               : Tok
  | layerScaleChGammaSgd (gName xName lrStr : String) (c h w : Nat)   : Tok
  | lnGammaSgd    (gName xName epsStr lrStr : String) (n : Nat)       : Tok
  | lnBetaSgd     (bName lrStr : String) (n : Nat)                    : Tok
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
  | gapBack    (c h w : Nat)               : Tok
  | broadcastBack (c h w : Nat)            : Tok
  | flatConvStridedF (w b : String) (ic oc h w' kH kW : Nat) : Tok
  | convStridedBack  (w : String) (ic oc h w' kH kW : Nat) : Tok
  | convStridedWeightSgd (xName wName lrStr : String) (ic oc h w' kH kW : Nat) : Tok
  | depthwiseWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Tok
  | depthwiseStridedWeightSgd (xName wName lrStr : String) (c h w' kH kW : Nat) : Tok
  | flatConvStride4F (w b : String) (ic oc h w' kH kW : Nat) : Tok
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
  | geluF      (n : Nat)                   : Tok
  | geluBack   (x : String) (n : Nat)      : Tok
  | layerScaleF (γ : String) (n : Nat)     : Tok
  | layerScaleChF (γ : String) (c h w : Nat) : Tok
  | softmaxRowF    (m n : Nat)             : Tok
  | softmaxRowBack (x : String) (m n : Nat) : Tok
  | matmulF    (m k n : Nat)               : Tok
  | transposeF (m n : Nat)                 : Tok
  | scaleF     (s : String) (n : Nat)      : Tok
  | lnRowF     (g b eps : String) (m n : Nat) : Tok
  | lnRowBack  (g x eps : String) (m n : Nat) : Tok
  | denseRowF  (w b : String) (N a c : Nat) : Tok
  | denseRowBack (w : String) (N a c : Nat) : Tok
  | patchEmbedF (w b cls pos : String) (ic H W P N D : Nat) : Tok
  | clsSliceF  (N D : Nat)                 : Tok
  | clsPadF    (N D : Nat)                 : Tok
  | headSliceF (N heads d hIdx : Nat)      : Tok
  | headPadF   (N heads d hIdx : Nat)      : Tok
  | rowScaleF  (g : String) (m n : Nat)    : Tok
  | rowBiasF   (b : String) (m n : Nat)    : Tok
  | batched    (tag : String) (names : List String) (info : List Nat) : Tok
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
  | .weightSgd xN wN lrS m n e => toToks e ++ [.weightSgd xN wN lrS m n]
  | .biasSgd bN lrS n e        => toToks e ++ [.biasSgd bN lrS n]
  | .convWeightSgd xN wN lrS ic oc h w kH kW e => toToks e ++ [.convWeightSgd xN wN lrS ic oc h w kH kW]
  | .convBiasSgd bN lrS oc h w e               => toToks e ++ [.convBiasSgd bN lrS oc h w]
  | .bnGammaSgd gN vN es lrS oc h w e          => toToks e ++ [.bnGammaSgd gN vN es lrS oc h w]
  | .bnBetaSgd bN lrS oc h w e                 => toToks e ++ [.bnBetaSgd bN lrS oc h w]
  | .layerScaleChGammaSgd gN xN lrS c h w e    => toToks e ++ [.layerScaleChGammaSgd gN xN lrS c h w]
  | .lnGammaSgd gN xN es lrS n e               => toToks e ++ [.lnGammaSgd gN xN es lrS n]
  | .lnBetaSgd bN lrS n e                      => toToks e ++ [.lnBetaSgd bN lrS n]
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
  | .gapBack c h w e => toToks e ++ [.gapBack c h w]
  | .broadcastBack c h w e => toToks e ++ [.broadcastBack c h w]
  | .flatConvStridedF w b ic oc h w' kH kW e => toToks e ++ [.flatConvStridedF w b ic oc h w' kH kW]
  | .convStridedBack w ic oc h w' kH kW e => toToks e ++ [.convStridedBack w ic oc h w' kH kW]
  | .convStridedWeightSgd xN wN lrS ic oc h w' kH kW e => toToks e ++ [.convStridedWeightSgd xN wN lrS ic oc h w' kH kW]
  | .depthwiseWeightSgd xN wN lrS c h w' kH kW e => toToks e ++ [.depthwiseWeightSgd xN wN lrS c h w' kH kW]
  | .depthwiseStridedWeightSgd xN wN lrS c h w' kH kW e => toToks e ++ [.depthwiseStridedWeightSgd xN wN lrS c h w' kH kW]
  | .flatConvStride4F w b ic oc h w' kH kW e => toToks e ++ [.flatConvStride4F w b ic oc h w' kH kW]
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
  | .geluF n e       => toToks e ++ [.geluF n]
  | .geluBack x n e  => toToks e ++ [.geluBack x n]
  | .layerScaleF γN n e => toToks e ++ [.layerScaleF γN n]
  | .layerScaleChF γN c h w e => toToks e ++ [.layerScaleChF γN c h w]
  | .softmaxRowF m n e    => toToks e ++ [.softmaxRowF m n]
  | .softmaxRowBack x m n e => toToks e ++ [.softmaxRowBack x m n]
  | .matmulF m k n a b    => toToks a ++ toToks b ++ [.matmulF m k n]
  | .transposeF m n e     => toToks e ++ [.transposeF m n]
  | .scaleF s n e         => toToks e ++ [.scaleF s n]
  | .lnRowF g b eps m n e => toToks e ++ [.lnRowF g b eps m n]
  | .lnRowBack g x eps m n e => toToks e ++ [.lnRowBack g x eps m n]
  | .denseRowF w b N a c e => toToks e ++ [.denseRowF w b N a c]
  | .denseRowBack w N a c e => toToks e ++ [.denseRowBack w N a c]
  | .patchEmbedF w b cls pos ic H W P N D e => toToks e ++ [.patchEmbedF w b cls pos ic H W P N D]
  | .clsSliceF N D e      => toToks e ++ [.clsSliceF N D]
  | .clsPadF N D e        => toToks e ++ [.clsPadF N D]
  | .headSliceF N heads d hIdx e => toToks e ++ [.headSliceF N heads d hIdx]
  | .headPadF N heads d hIdx e   => toToks e ++ [.headPadF N heads d hIdx]
  | .rowScaleF g m n e    => toToks e ++ [.rowScaleF g m n]
  | .rowBiasF b m n e     => toToks e ++ [.rowBiasF b m n]
  | .batched tag names info e   => toToks e ++ [.batched tag names info]

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
  | .weightSgd xN wN lrS m n, r :: st => do
      let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (s!"    {dW} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [B,n]}) -> {ty [m,n]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [m,n]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [m,n]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [m,n]}\n", o :: st)
  | .biasSgd bN lrS n, r :: st => do
      let z ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dB} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,n]}, tensor<f32>) -> {ty [n]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [n]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [n]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [n]}\n", o :: st)
  | .convWeightSgd xN wN lrS ic oc h w kH kW, r :: st => do
      -- conv weight grad (transpose trick) then SGD: reshape flat acts/cotangent to
      -- 4-D, transpose batch↔feature, convolve (batch as contraction), transpose back
      -- to [oc,ic,kH,kW], then θ' = θ − lr·dW. Same op text as `CnnRender.convWGrad`+`sgd`.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
  | .convBiasSgd bN lrS oc h w, r :: st => do
      -- conv bias grad (reduce over batch+spatial [0,2,3]) then SGD. Same op text as
      -- `CnnRender.convBiasGrad`+`sgd`.
      let dr ← fresh; let z ← fresh; let g ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {g} = stablehlo.reduce({dr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sB} = stablehlo.multiply {g}, {lB} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
  | .bnGammaSgd gN vN epsStr lrS oc h w, r :: st => do
      -- BN per-channel γ grad: recompute x̂ from the saved conv output {vN} (reduce μ/var
      -- over spatial [2,3]), dγ_c = Σ_{b,h,w} dy·x̂, then SGD. Same op text as the dγ half
      -- of `CnnRender.bnParamGradPC`+`sgd`.
      let z ← fresh; let xr ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let dyr ← fresh; let p ← fresh; let dg ← fresh
      let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,oc,h,w]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,oc,h,w]}\n" ++
        s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,oc]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {dyr}, {xhat} : {ty [B,oc,h,w]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [oc]}\n", o :: st)
  | .bnBetaSgd bN lrS oc h w, r :: st => do
      -- BN per-channel β grad: dβ_c = Σ_{b,h,w} dy, then SGD (β grad needs no x̂).
      let z ← fresh; let dyr ← fresh; let db ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {db} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
        s!"    {sB} = stablehlo.multiply {db}, {lB} : {ty [oc]}\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [oc]}\n", o :: st)
  | .layerScaleChGammaSgd gN xN lrS c h w, r :: st => do
      -- per-channel layer-scale γ grad: dγ_c = reduce[0,2,3](x⊙dy), then SGD (`lsGradCh` + wrap).
      let z ← fresh; let xr ← fresh; let dr ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {p} = stablehlo.multiply {xr}, {dr} : {ty [B,c,h,w]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [c]}\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : {ty [c]}\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : {ty [c]}\n", o :: st)
  | .lnGammaSgd gN xN epsStr lrS n, r :: st => do
      -- scalar-LN γ grad: recompute x̂ from the saved LN input {xN} (μ/var over [1]),
      -- dγ = Σ_{b,k} dy·x̂ → tensor<f32>, reshape to the Vec-1 param, SGD (`lnParamGrad` dγ half + wrap).
      let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh; let sm ← fresh
      let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
      let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh; let p ← fresh
      let dg ← fresh; let lG ← fresh; let sG ← fresh; let o ← fresh
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
        s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,n]}\n" ++
        s!"    {p} = stablehlo.multiply {r}, {xh} : {ty [B,n]}\n" ++
        s!"    {dg} = stablehlo.reduce({p} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {lG} = stablehlo.constant dense<{lrS}> : tensor<f32>\n" ++
        s!"    {sG} = stablehlo.multiply {dg}, {lG} : tensor<f32>\n" ++
        s!"    {o} = stablehlo.subtract {gN}, {sG} : tensor<f32>\n", o :: st)
  | .lnBetaSgd bN lrS n, r :: st => do
      -- scalar-LN β grad: dβ = Σ_{b,k} dy → tensor<f32> (rank-0, matches the scalar-LN bnF param), SGD.
      let z ← fresh; let db ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
      pure (
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {db} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,n]}, tensor<f32>) -> tensor<f32>\n" ++
        s!"    {lB} = stablehlo.constant dense<{lrS}> : tensor<f32>\n" ++
        s!"    {sB} = stablehlo.multiply {db}, {lB} : tensor<f32>\n" ++
        s!"    {o} = stablehlo.subtract {bN}, {sB} : tensor<f32>\n", o :: st)
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
  | .gapBack c h w, r :: st => do
      -- GAP backward (VJP): divide the per-channel cotangent by h·w, broadcast
      -- it back over the H×W spatial grid, reshape to flat. Reverse of `.gapF`.
      -- Denotes `globalAvgPoolFlat`'s VJP backward `dy[chan idx] / (h·w)`.
      -- (Text emission best-effort/unverified-vs-IREE; the `den` is proven.)
      let nf ← fresh; let dv ← fresh; let bb ← fresh; let o ← fresh
      pure (
        s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
        s!"    {dv} = stablehlo.divide {r}, {nf} : {ty [B,c]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {dv}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {o} = stablehlo.reshape {bb} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
  | .broadcastBack c h w, r :: st => do
      -- broadcast backward (VJP) = sum over H×W per channel (adjoint of broadcast):
      -- reshape to [B,c,h,w], reduce-add over spatial axes [2,3] → [B,c]. No divide.
      -- (Text emission best-effort/unverified-vs-IREE; the `den` is proven.)
      let xn ← fresh; let z ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {o} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n", o :: st)
  | .flatConvStride4F w b ic oc h w' kH kW, r :: st => do
      -- stride-4 patchify conv (the ConvNeXt 4×4/s4 stem): reshape, convolution
      -- with window_strides=[4,4], +bias. The denotation reads the SAME conv
      -- (pad (k-1)/2) at the offset-1 positions 4i+1 (decimate ∘ decimateOdd),
      -- so the emitted pad is one less: (k-1)/2 − 1 — for the 4×4 stem pad 0,
      -- the left-aligned window x[4i..4i+3] of the paper's pad-0 Conv2d(4, s=4).
      let pH := (kH - 1) / 2 - 1; let pW := (kW - 1) / 2 - 1
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*(2*(2*h))*(2*(2*w'))]}) -> {ty [B,ic,2*(2*h),2*(2*w')]}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {w})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [4, 4], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,2*(2*h),2*(2*w')]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {b}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w']}\n" ++
        s!"    {ob} = stablehlo.add {cv}, {bb} : {ty [B,oc,h,w']}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,oc,h,w']}) -> {ty [B, oc*h*w']}\n", o :: st)
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
      -- the 2h×2w grid, then the reversed-kernel stride-1 conv (= decimate.back ▸ conv.back).
      -- Transpose-conv pad: low = k−1−p, high = p (p = the forward pad (k−1)/2) —
      -- symmetric (k−1)/2 for odd k (3×3 MNV2/r34, unchanged), [[1,0]] for the
      -- even 2×2 ConvNeXt downsample (the left-aligned forward window).
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
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{kH - 1 - pH}, {pH}], [{kW - 1 - pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,oc,2*h,2*w']}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,2*h,2*w']}\n" ++
        s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,2*h,2*w']}) -> {ty [B, ic*(2*h)*(2*w')]}\n", o :: st)
  | .convStridedWeightSgd xN wN lrS ic oc h w kH kW, r :: st => do
      -- strided (stride-2) conv weight grad then SGD: reshape x to the 2h×2w grid and dy
      -- to h×w, zero-upsample dy (interior+high=1 → 2h×2w, the decimate-backward), then the
      -- SAME transpose-trick stride-1 weight-grad conv as `convWeightSgd` on the 2h×2w grid →
      -- [oc,ic,kH,kW], then θ' = θ − lr·dW. Same op text as `TestResnet34Train.convWGradStrided`.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,2*h,2*w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,2*h,2*w]}) -> {ty [oc,B,2*h,2*w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,2*h,2*w]}) -> {ty [ic,oc,kH,kW]}\n" ++
        s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
  | .depthwiseWeightSgd xN wN lrS c h w kH kW, r :: st => do
      -- depthwise (grouped) weight grad: per-channel transpose-trick conv with
      -- `batch_group_count = c` (each output kernel reads only its own channel) → [1,c,kH,kW],
      -- reshape to the depthwise kernel layout [c,1,kH,kW], then θ' = θ − lr·dW. Same op text as
      -- `TestMobilenetV2Train.dwconvWGrad` + sgd.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
        s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
        s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
  | .depthwiseStridedWeightSgd xN wN lrS c h w kH kW, r :: st => do
      -- strided depthwise weight grad: reshape x to the 2h×2w grid and dy to h×w, zero-upsample dy
      -- (interior+high=1 → 2h×2w, the decimate-backward), then the SAME per-channel transpose-trick
      -- weight-grad conv (`batch_group_count = c`) on the 2h×2w grid → [1,c,kH,kW], reshape to the
      -- depthwise layout [c,1,kH,kW], then θ' = θ − lr·dW. Same op text as
      -- `TestMobilenetV2Train.dwconvWGradStrided` + sgd.
      let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
      let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
      let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
      pure (
        s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
        s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
        s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
        s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
        s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
        s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
        s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
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
  | .layerScaleF gN n, r :: st => do
      -- per-element layer-scale `γ ⊙ x`: broadcast γ:[n] over the batch, then multiply.
      let gb ← fresh; let o ← fresh
      pure (s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [n]}) -> {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {gb} : {ty [B,n]}\n", o :: st)
  | .layerScaleChF gN c h w', r :: st => do
      -- per-channel layer-scale: reshape flat→NCHW, broadcast γ:[c] over
      -- batch+spatial (dims=[1]), multiply, reshape back.
      let xn ← fresh; let gb ← fresh; let m ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, c*h*w']}) -> {ty [B,c,h,w']}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w']}\n" ++
            s!"    {m} = stablehlo.multiply {xn}, {gb} : {ty [B,c,h,w']}\n" ++
            s!"    {o} = stablehlo.reshape {m} : ({ty [B,c,h,w']}) -> {ty [B, c*h*w']}\n", o :: st)
  | .geluF n, r :: st => do
      -- gelu forward (tanh approximation): y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
      -- Smooth everywhere (no kink/mask); `stablehlo.tanh` is the only non-arith op.
      let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
      let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
      let chalf ← fresh; let hx ← fresh; let o ← fresh
      pure (s!"    {x2} = stablehlo.multiply {r}, {r} : {ty [B,n]}\n" ++
            s!"    {x3} = stablehlo.multiply {x2}, {r} : {ty [B,n]}\n" ++
            s!"    {ck} = stablehlo.constant dense<0.044715> : {ty [B,n]}\n" ++
            s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty [B,n]}\n" ++
            s!"    {inn} = stablehlo.add {r}, {kx3} : {ty [B,n]}\n" ++
            s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty [B,n]}\n" ++
            s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty [B,n]}\n" ++
            s!"    {t} = stablehlo.tanh {u} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {opt} = stablehlo.add {one}, {t} : {ty [B,n]}\n" ++
            s!"    {chalf} = stablehlo.constant dense<0.5> : {ty [B,n]}\n" ++
            s!"    {hx} = stablehlo.multiply {chalf}, {r} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {hx}, {opt} : {ty [B,n]}\n", o :: st)
  | .geluBack x n, r :: st => do
      -- gelu input-VJP: dy ⊙ gelu'(x), recomputing tanh(u(x)) from the saved
      -- pre-activation {x}. gelu'(x) = 0.5·(1+t) + 0.5·x·(1−t²)·√(2/π)·(1+3·0.044715·x²),
      -- t = tanh(√(2/π)·(x+0.044715·x³)). (Matches IRPrint `renderGeluB`.)
      let x2 ← fresh; let x3 ← fresh; let ck ← fresh; let kx3 ← fresh; let inn ← fresh
      let csqrt ← fresh; let u ← fresh; let t ← fresh; let one ← fresh; let opt ← fresh
      let chalf ← fresh; let term1 ← fresh; let t2 ← fresh; let omt2 ← fresh
      let hx ← fresh; let hxo ← fresh; let c3b ← fresh; let a3x2 ← fresh
      let in2 ← fresh; let up ← fresh; let term2 ← fresh; let gp ← fresh; let o ← fresh
      pure (s!"    {x2} = stablehlo.multiply {x}, {x} : {ty [B,n]}\n" ++
            s!"    {x3} = stablehlo.multiply {x2}, {x} : {ty [B,n]}\n" ++
            s!"    {ck} = stablehlo.constant dense<0.044715> : {ty [B,n]}\n" ++
            s!"    {kx3} = stablehlo.multiply {ck}, {x3} : {ty [B,n]}\n" ++
            s!"    {inn} = stablehlo.add {x}, {kx3} : {ty [B,n]}\n" ++
            s!"    {csqrt} = stablehlo.constant dense<0.7978845608028654> : {ty [B,n]}\n" ++
            s!"    {u} = stablehlo.multiply {csqrt}, {inn} : {ty [B,n]}\n" ++
            s!"    {t} = stablehlo.tanh {u} : {ty [B,n]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,n]}\n" ++
            s!"    {opt} = stablehlo.add {one}, {t} : {ty [B,n]}\n" ++
            s!"    {chalf} = stablehlo.constant dense<0.5> : {ty [B,n]}\n" ++
            s!"    {term1} = stablehlo.multiply {chalf}, {opt} : {ty [B,n]}\n" ++
            s!"    {t2} = stablehlo.multiply {t}, {t} : {ty [B,n]}\n" ++
            s!"    {omt2} = stablehlo.subtract {one}, {t2} : {ty [B,n]}\n" ++
            s!"    {hx} = stablehlo.multiply {chalf}, {x} : {ty [B,n]}\n" ++
            s!"    {hxo} = stablehlo.multiply {hx}, {omt2} : {ty [B,n]}\n" ++
            s!"    {c3b} = stablehlo.constant dense<0.134145> : {ty [B,n]}\n" ++
            s!"    {a3x2} = stablehlo.multiply {c3b}, {x2} : {ty [B,n]}\n" ++
            s!"    {in2} = stablehlo.add {one}, {a3x2} : {ty [B,n]}\n" ++
            s!"    {up} = stablehlo.multiply {csqrt}, {in2} : {ty [B,n]}\n" ++
            s!"    {term2} = stablehlo.multiply {hxo}, {up} : {ty [B,n]}\n" ++
            s!"    {gp} = stablehlo.add {term1}, {term2} : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {gp} : {ty [B,n]}\n", o :: st)
  | .softmaxRowF m n, r :: st => do
      -- ROW-softmax: reshape flat `[B,m*n]` → `[B,m,n]`, exp, reduce add over the
      -- LAST axis [2] (per row), broadcast back over dims [0,1], divide, reshape to
      -- flat. Plain exp/sum (no max-shift), matching the proven `softmax` (3-D
      -- analogue of `.softmaxDiv`).
      let xn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh; let sb ← fresh
      let dv ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
        s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {dv} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {dv} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .softmaxRowBack x m n, r :: st => do
      -- ROW-softmax input-VJP `p ⊙ (dy − ⟨p,dy⟩)` per row: reshape flat→`[B,m,n]`,
      -- recompute `p` from the saved pre-softmax scores {x} (exp/reduce[2]/broadcast/
      -- divide), then the rank-1 correction (`pdy`, reduce[2], subtract, multiply),
      -- reshape to flat. {r} is dy.
      let xn ← fresh; let dn ← fresh; let z ← fresh; let e ← fresh; let s ← fresh
      let sb ← fresh; let p ← fresh; let pdy ← fresh; let sr ← fresh; let srb ← fresh
      let d ← fresh; let dz ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {x} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {e} = stablehlo.exponential {xn} : {ty [B,m,n]}\n" ++
        s!"    {s} = stablehlo.reduce({e} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sb} = stablehlo.broadcast_in_dim {s}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {p} = stablehlo.divide {e}, {sb} : {ty [B,m,n]}\n" ++
        s!"    {pdy} = stablehlo.multiply {p}, {dn} : {ty [B,m,n]}\n" ++
        s!"    {sr} = stablehlo.reduce({pdy} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {srb} = stablehlo.broadcast_in_dim {sr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {d} = stablehlo.subtract {dn}, {srb} : {ty [B,m,n]}\n" ++
        s!"    {dz} = stablehlo.multiply {p}, {d} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {dz} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .matmulF m k n, b :: a :: st => do
      -- flattened matrix multiply C = A·B: reshape both operands to rank 3,
      -- dot_general with batching dim 0 (contract A's last axis with B's middle),
      -- reshape back to flat. (Postorder pushes a then b, so b is on top.)
      let an ← fresh; let bn ← fresh; let mm ← fresh; let o ← fresh
      pure (s!"    {an} = stablehlo.reshape {a} : ({ty [B, m*k]}) -> {ty [B,m,k]}\n" ++
        s!"    {bn} = stablehlo.reshape {b} : ({ty [B, k*n]}) -> {ty [B,k,n]}\n" ++
        s!"    {mm} = stablehlo.dot_general {an}, {bn}, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,m,k]}, {ty [B,k,n]}) -> {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {mm} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .transposeF m n, r :: st => do
      -- flattened matrix transpose: reshape to rank 3, swap the matrix axes
      -- (dims = [0, 2, 1], batch axis fixed), reshape back.
      let xn ← fresh; let t ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {t} = stablehlo.transpose {xn}, dims = [0, 2, 1] : ({ty [B,m,n]}) -> {ty [B,n,m]}\n" ++
        s!"    {o} = stablehlo.reshape {t} : ({ty [B,n,m]}) -> {ty [B, n*m]}\n", o :: st)
  | .scaleF sStr n, r :: st => do
      -- scalar multiply s·x against a splat constant (SDPA's 1/√d).
      let c ← fresh; let o ← fresh
      pure (s!"    {c} = stablehlo.constant dense<{sStr}> : {ty [B,n]}\n" ++
            s!"    {o} = stablehlo.multiply {r}, {c} : {ty [B,n]}\n", o :: st)
  | .lnRowF gN bN epsStr m n, r :: st => do
      -- ROW-wise LayerNorm forward: reshape flat [B,m*n] → [B,m,n], then `bnF`'s
      -- normalize/affine graph at rank 3 — μ/var reduced over the LAST axis [2]
      -- (per token row), broadcast back over dims [0,1], scalar γ/β (dims = []),
      -- reshape to flat. LayerNorm IS per-example BN per row.
      let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let bb ← fresh; let gx ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {gx} = stablehlo.multiply {xhat}, {gb} : {ty [B,m,n]}\n" ++
        s!"    {ob} = stablehlo.add {gx}, {bb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .lnRowBack gN xN epsStr m n, r :: st => do
      -- ROW-wise LN input-VJP: recompute x̂/istd per row from the saved flat
      -- pre-LN input {xN}, then `bnBack`'s consolidated three-term
      -- `(istd/n)·(n·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))` (dx̂ = γ·dy) at rank 3, all Σ
      -- reductions over the row axis [2], reshape to flat. {r} is dy.
      let dn ← fresh; let xn ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh
      let smr ← fresh; let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh
      let vsr ← fresh; let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh
      let xhat ← fresh; let gb ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
      let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
      let xs ← fresh; let i2 ← fresh; let sN ← fresh; let o0 ← fresh; let o ← fresh
      pure (
        s!"    {dn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {xn} = stablehlo.reshape {xN} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {nf} = stablehlo.constant dense<{n}.0> : {ty [B,m,n]}\n" ++
        s!"    {ep} = stablehlo.constant dense<{epsStr}> : {ty [B,m,n]}\n" ++
        s!"    {smr} = stablehlo.reduce({xn} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {xc} = stablehlo.subtract {xn}, {mu} : {ty [B,m,n]}\n" ++
        s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,m,n]}\n" ++
        s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,m,n]}\n" ++
        s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,m,n]}\n" ++
        s!"    {xhat} = stablehlo.multiply {xc}, {istd} : {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [] : (tensor<f32>) -> {ty [B,m,n]}\n" ++
        s!"    {dxh} = stablehlo.multiply {gb}, {dn} : {ty [B,m,n]}\n" ++
        s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {xd} = stablehlo.multiply {xhat}, {dxh} : {ty [B,m,n]}\n" ++
        s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2] : ({ty [B,m,n]}, tensor<f32>) -> {ty [B,m]}\n" ++
        s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [0, 1] : ({ty [B,m]}) -> {ty [B,m,n]}\n" ++
        s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,m,n]}\n" ++
        s!"    {xs} = stablehlo.multiply {xhat}, {sxd} : {ty [B,m,n]}\n" ++
        s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,m,n]}\n" ++
        s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,m,n]}\n" ++
        s!"    {o0} = stablehlo.multiply {sN}, {i2} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {o0} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .denseRowF wN bN N a c, r :: st => do
      -- per-token dense: reshape [B,N*a] → [B,N,a], dot_general contracting the
      -- feature axis with W:[a,c] ([2] x [0] — every token row through the same W),
      -- bias broadcast dims = [2], reshape back. (ViTRender `mlpRowFwd` form.)
      let xn ← fresh; let dg ← fresh; let bb ← fresh; let ob ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, N*a]}) -> {ty [B,N,a]}\n" ++
        s!"    {dg} = stablehlo.dot_general {xn}, {wN}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,N,a]}, {ty [a,c]}) -> {ty [B,N,c]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [c]}) -> {ty [B,N,c]}\n" ++
        s!"    {ob} = stablehlo.add {dg}, {bb} : {ty [B,N,c]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,N,c]}) -> {ty [B, N*c]}\n", o :: st)
  | .denseRowBack wN N a c, r :: st => do
      -- per-token dense input-VJP dX = dY·Wᵀ: contract dy's feature axis with W's
      -- OUTPUT axis ([2] x [1] — the GPU-validated ViTRender backward form).
      let dn ← fresh; let dg ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*c]}) -> {ty [B,N,c]}\n" ++
        s!"    {dg} = stablehlo.dot_general {dn}, {wN}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,N,c]}, {ty [a,c]}) -> {ty [B,N,a]}\n" ++
        s!"    {o} = stablehlo.reshape {dg} : ({ty [B,N,a]}) -> {ty [B, N*a]}\n", o :: st)
  | .patchEmbedF wN bN clsN posN ic H W P N D, r :: st => do
      -- ViT patch embedding: reshape image to [B,ic,H,W], stride-P VALID conv
      -- (kernel [D,ic,P,P] — the non-overlapping patch projection) + bias, move
      -- channels last (transpose [0,2,3,1]) and flatten the patch grid to [B,N,D],
      -- prepend the broadcast CLS token (concatenate at dim 1), add the position
      -- embedding (broadcast dims = [1,2]), reshape to flat [B,(N+1)*D].
      let hp := H / P; let wp := W / P
      let xn ← fresh; let cv ← fresh; let bb ← fresh; let cb ← fresh
      let tr ← fresh; let tk ← fresh; let clsb ← fresh; let cat ← fresh
      let pb ← fresh; let ob ← fresh; let o ← fresh
      pure (
        s!"    {xn} = stablehlo.reshape {r} : ({ty [B, ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
        s!"    {cv} = stablehlo.convolution({xn}, {wN})\n" ++
        "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
        s!"      window = " ++ "{" ++ s!"stride = [{P}, {P}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
        "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
        s!" : ({ty [B,ic,H,W]}, {ty [D,ic,P,P]}) -> {ty [B,D,hp,wp]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [D]}) -> {ty [B,D,hp,wp]}\n" ++
        s!"    {cb} = stablehlo.add {cv}, {bb} : {ty [B,D,hp,wp]}\n" ++
        s!"    {tr} = stablehlo.transpose {cb}, dims = [0, 2, 3, 1] : ({ty [B,D,hp,wp]}) -> {ty [B,hp,wp,D]}\n" ++
        s!"    {tk} = stablehlo.reshape {tr} : ({ty [B,hp,wp,D]}) -> {ty [B,N,D]}\n" ++
        s!"    {clsb} = stablehlo.broadcast_in_dim {clsN}, dims = [2] : ({ty [D]}) -> {ty [B,1,D]}\n" ++
        s!"    {cat} = stablehlo.concatenate {clsb}, {tk}, dim = 1 : ({ty [B,1,D]}, {ty [B,N,D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {pb} = stablehlo.broadcast_in_dim {posN}, dims = [1, 2] : ({ty [N+1,D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {ob} = stablehlo.add {cat}, {pb} : {ty [B,N+1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {ob} : ({ty [B,N+1,D]}) -> {ty [B, (N+1)*D]}\n", o :: st)
  | .clsSliceF N D, r :: st => do
      -- CLS-token gather (row 0): reshape [B,(N+1)*D] → [B,N+1,D], slice the
      -- first token row, reshape to [B,D]. (ViTRender `headFwd` slice form.)
      let xn ← fresh; let sl ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, (N+1)*D]}) -> {ty [B,N+1,D]}\n" ++
        s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:1, 0:{D}] : ({ty [B,N+1,D]}) -> {ty [B,1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {sl} : ({ty [B,1,D]}) -> {ty [B,D]}\n", o :: st)
  | .clsPadF N D, r :: st => do
      -- CLS-slice VJP (scatter dy to row 0): reshape [B,D] → [B,1,D], zero-pad
      -- N token rows below (high = [0, N, 0]), reshape to flat [B,(N+1)*D].
      -- (ViTRender `headBack` pad form.)
      let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B,D]}) -> {ty [B,1,D]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, 0], high = [0, {N}, 0], interior = [0, 0, 0] : ({ty [B,1,D]}, tensor<f32>) -> {ty [B,N+1,D]}\n" ++
        s!"    {o} = stablehlo.reshape {pd} : ({ty [B,N+1,D]}) -> {ty [B, (N+1)*D]}\n", o :: st)
  | .headSliceF N heads d hIdx, r :: st => do
      -- per-head column slice: reshape [B,N*(H*d)] → [B,N,H*d], slice head h's
      -- contiguous feature block [h*d:(h+1)*d] (row-major layout), reshape to flat.
      let xn ← fresh; let sl ← fresh; let o ← fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, N*(heads*d)]}) -> {ty [B,N,heads*d]}\n" ++
        s!"    {sl} = stablehlo.slice {xn} [0:{B}, 0:{N}, {hIdx*d}:{(hIdx+1)*d}] : ({ty [B,N,heads*d]}) -> {ty [B,N,d]}\n" ++
        s!"    {o} = stablehlo.reshape {sl} : ({ty [B,N,d]}) -> {ty [B, N*d]}\n", o :: st)
  | .headPadF N heads d hIdx, r :: st => do
      -- per-head column scatter: reshape [B,N*d] → [B,N,d], zero-pad the feature
      -- axis into head h's block (low = h*d, high = (heads-1-h)*d), reshape to flat.
      let dn ← fresh; let z ← fresh; let pd ← fresh; let o ← fresh
      pure (s!"    {dn} = stablehlo.reshape {r} : ({ty [B, N*d]}) -> {ty [B,N,d]}\n" ++
        s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
        s!"    {pd} = stablehlo.pad {dn}, {z}, low = [0, 0, {hIdx*d}], high = [0, 0, {(heads-1-hIdx)*d}], interior = [0, 0, 0] : ({ty [B,N,d]}, tensor<f32>) -> {ty [B,N,heads*d]}\n" ++
        s!"    {o} = stablehlo.reshape {pd} : ({ty [B,N,heads*d]}) -> {ty [B, N*(heads*d)]}\n", o :: st)
  | .rowScaleF gN m n, r :: st => do
      -- per-token broadcast scale: reshape [B,m*n] -> [B,m,n], broadcast the shared
      -- gamma:[n] over batch+rows (dims = [2]), multiply, reshape back.
      let xn <- fresh; let gb <- fresh; let mu <- fresh; let o <- fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
        s!"    {mu} = stablehlo.multiply {xn}, {gb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {mu} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .rowBiasF bN m n, r :: st => do
      -- per-token broadcast bias: same bracket, broadcast beta:[n] dims = [2], add.
      let xn <- fresh; let bb <- fresh; let ad <- fresh; let o <- fresh
      pure (s!"    {xn} = stablehlo.reshape {r} : ({ty [B, m*n]}) -> {ty [B,m,n]}\n" ++
        s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [2] : ({ty [n]}) -> {ty [B,m,n]}\n" ++
        s!"    {ad} = stablehlo.add {xn}, {bb} : {ty [B,m,n]}\n" ++
        s!"    {o} = stablehlo.reshape {ad} : ({ty [B,m,n]}) -> {ty [B, m*n]}\n", o :: st)
  | .batched tag names info, r :: st =>
      -- EfficientNet batched op: emit the concrete `[N,C,H,W]` StableHLO from the
      -- tag (which op) + names (weight/bias/BN-input/SE-input/γ/ε SSA names) + info
      -- (shape dims). Batched values flow as 2-D `[B, c·h·w]` (B = batch); each op
      -- reshapes its operand to 4-D, computes, reshapes back — uniform with the
      -- per-example ops (`convWeightSgd`/`denseRowF` do the same). Backward ops are
      -- self-contained: they recompute forward intermediates from the carried
      -- input/weight names (the mnv2 pattern). `den` never calls `emit`; this text
      -- is iree-validated, not theorem-tied (the per-op lexing trust the whole
      -- suite carries). Backward tags are filled in the next pass.
      match tag, names, info with
      | "conv", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,ic,h,w]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "convStrided", [wN, bN], [_N, ic, oc, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,ic,2*h,2*w]}, {ty [oc,ic,kH,kW]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | "depthwise", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,h,w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "depthwiseStrided", [wN, bN], [_N, c, h, w, kH, kW] => do
          let p := (kH - 1) / 2
          let xr ← fresh; let cc ← fresh; let bb ← fresh; let ca ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {cc} = stablehlo.convolution({xr}, {wN})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [2, 2], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {ca} = stablehlo.add {cc}, {bb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {ca} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "dense", [wN, bN], [_N, a, c] => do
          let dg ← fresh; let bb ← fresh; let o ← fresh
          pure (
            s!"    {dg} = stablehlo.dot_general {r}, {wN}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [a,c]}) -> {ty [B,c]}\n" ++
            s!"    {bb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {o} = stablehlo.add {dg}, {bb} : {ty [B,c]}\n", o :: st)
      | "gap", [], [_N, c, h, w] => do
          let xr ← fresh; let z ← fresh; let sr ← fresh; let nf ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {o} = stablehlo.divide {sr}, {nf} : {ty [B,c]}\n", o :: st)
      | "seBlock", [w1, b1, w2, b2], [_N, c, h, w, rr] => do
          let xr ← fresh; let z ← fresh; let sqs ← fresh; let sqnf ← fresh; let sq ← fresh
          let exd ← fresh; let exbb ← fresh; let ex ← fresh; let a1s ← fresh; let a1 ← fresh
          let h2d ← fresh; let h2bb ← fresh; let h2 ← fresh; let gate ← fresh; let gb ← fresh
          let se ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sqs} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {sqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {sq} = stablehlo.divide {sqs}, {sqnf} : {ty [B,c]}\n" ++
            s!"    {exd} = stablehlo.dot_general {sq}, {w1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [c,rr]}) -> {ty [B,rr]}\n" ++
            s!"    {exbb} = stablehlo.broadcast_in_dim {b1}, dims = [1] : ({ty [rr]}) -> {ty [B,rr]}\n" ++
            s!"    {ex} = stablehlo.add {exd}, {exbb} : {ty [B,rr]}\n" ++
            s!"    {a1s} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {a1} = stablehlo.multiply {ex}, {a1s} : {ty [B,rr]}\n" ++
            s!"    {h2d} = stablehlo.dot_general {a1}, {w2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [rr,c]}) -> {ty [B,c]}\n" ++
            s!"    {h2bb} = stablehlo.broadcast_in_dim {b2}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {h2} = stablehlo.add {h2d}, {h2bb} : {ty [B,c]}\n" ++
            s!"    {gate} = stablehlo.logistic {h2} : {ty [B,c]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gate}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {se} = stablehlo.multiply {xr}, {gb} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {se} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "bnBatch", [gN, bN, es], [_N, oc, h, w] => do
          let xr ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh
          let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh
          let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let gb ← fresh; let btb ← fresh; let gx ← fresh; let o4 ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {btb} = stablehlo.broadcast_in_dim {bN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {gx} = stablehlo.multiply {xh}, {gb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o4} = stablehlo.add {gx}, {btb} : {ty [B,oc,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {o4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
      | t, [gN, xN, es], [_N, oc, h, w] =>
          -- bnBatchBack / bnBatchLABack: the 3-term true-BN input-VJP. Self-contained
          -- recompute of x̂/istd from the saved BN input `xN` + γ `gN` + ε `es`
          -- (mnv2 pattern), then dx = (istd/nf)·(nf·(γ⊙dy) − Σ(γ⊙dy) − x̂·Σ(x̂·γ⊙dy)).
          -- `r` is the upstream cotangent dy. (dγ/dβ are param grads, not here.)
          if t == "bnBatchBack" || t == "bnBatchLABack" then do
            let xr ← fresh; let z ← fresh; let nf ← fresh; let ep ← fresh; let smr ← fresh
            let sm ← fresh; let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh
            let vs ← fresh; let vr ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
            let gb ← fresh; let dyr ← fresh; let dxh ← fresh; let sdxr ← fresh; let sdx ← fresh
            let xd ← fresh; let sxdr ← fresh; let sxd ← fresh; let t1 ← fresh; let i1 ← fresh
            let xs ← fresh; let i2 ← fresh; let sN ← fresh; let dx4 ← fresh; let o ← fresh
            pure (
              s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
              s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
              s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
              s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
              s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
              s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
              s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
              s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
              s!"    {gb} = stablehlo.broadcast_in_dim {gN}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {dxh} = stablehlo.multiply {gb}, {dyr} : {ty [B,oc,h,w]}\n" ++
              s!"    {sdxr} = stablehlo.reduce({dxh} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sdx} = stablehlo.broadcast_in_dim {sdxr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {xd} = stablehlo.multiply {xh}, {dxh} : {ty [B,oc,h,w]}\n" ++
              s!"    {sxdr} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
              s!"    {sxd} = stablehlo.broadcast_in_dim {sxdr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
              s!"    {t1} = stablehlo.multiply {dxh}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {i1} = stablehlo.subtract {t1}, {sdx} : {ty [B,oc,h,w]}\n" ++
              s!"    {xs} = stablehlo.multiply {xh}, {sxd} : {ty [B,oc,h,w]}\n" ++
              s!"    {i2} = stablehlo.subtract {i1}, {xs} : {ty [B,oc,h,w]}\n" ++
              s!"    {sN} = stablehlo.divide {istd}, {nf} : {ty [B,oc,h,w]}\n" ++
              s!"    {dx4} = stablehlo.multiply {sN}, {i2} : {ty [B,oc,h,w]}\n" ++
              s!"    {o} = stablehlo.reshape {dx4} : ({ty [B,oc,h,w]}) -> {ty [B, oc*h*w]}\n", o :: st)
          else
            pure (s!"    // [EfficientNet Item B] batched {tag} {names} {info} — render TODO\n", r :: st)
      | "convBackBatched", [wN], [_N, ic, oc, h, w, kH, kW] => do
          -- conv input-VJP: dx = conv(dy, reverse(W,[2,3])ᵀ), reversed+transposed
          -- kernel, stride 1, same-pad p. (1×1 in enet ⇒ p=0, reverse a no-op.)
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh; let wt ← fresh; let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({dyr}, {wt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,oc,h,w]}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,h,w]}) -> {ty [B, ic*h*w]}\n", o :: st)
      | "convStridedBackBatched", [wN], [_N, ic, oc, h, w, kH, kW] => do
          -- stride-2 conv input-VJP: upsample dy (zero-interleave to 2h×2w) then the
          -- stride-1 conv input-VJP. Produces dx at the 2h×2w input resolution.
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh; let wt ← fresh
          let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {wt} = stablehlo.transpose {rev}, dims = [1, 0, 2, 3] : ({ty [oc,ic,kH,kW]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({up}, {wt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [B,oc,2*h,2*w]}, {ty [ic,oc,kH,kW]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,ic,2*h,2*w]}) -> {ty [B, ic*(2*h)*(2*w)]}\n", o :: st)
      | "depthwiseBackBatched", [wN], [_N, c, h, w, kH, kW] => do
          -- depthwise input-VJP: dx = depthwise_conv(dy, reverse(W,[2,3])), fgc=c,
          -- same-pad p (no transpose — one input channel per group).
          let p := (kH - 1) / 2
          let dyr ← fresh; let rev ← fresh; let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({dyr}, {rev})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,h,w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "depthwiseStridedBackBatched", [wN], [_N, c, h, w, kH, kW] => do
          -- stride-2 depthwise input-VJP: upsample dy then the stride-1 depthwise
          -- input-VJP. dx at the 2h×2w input resolution.
          let p := (kH - 1) / 2
          let dyr ← fresh; let z ← fresh; let up ← fresh; let rev ← fresh; let dx ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {up} = stablehlo.pad {dyr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {rev} = stablehlo.reverse {wN}, dims = [2, 3] : {ty [c,1,kH,kW]}\n" ++
            s!"    {dx} = stablehlo.convolution({up}, {rev})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      " ++ "{" ++ s!"batch_group_count = 1 : i64, feature_group_count = {c} : i64" ++ "}" ++
            s!" : ({ty [B,c,2*h,2*w]}, {ty [c,1,kH,kW]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {o} = stablehlo.reshape {dx} : ({ty [B,c,2*h,2*w]}) -> {ty [B, c*(2*h)*(2*w)]}\n", o :: st)
      | "seBackBatched", [w1, b1, w2, b2, vN], [_N, c, h, w, rr] => do
          -- SE backward: recompute the SE forward (GAP → dense W₁ b₁ → swish → dense
          -- W₂ b₂ → sigmoid gate) from the SE input `vN`, then the SE-input cotangent
          -- dx = gate⊙dse + GAP-adjoint(W₁ᵀ·swish'·W₂ᵀ·(gate·(1−gate))·Σ(x⊙dse)).
          -- `r` is the SE-output cotangent dse.
          let xr ← fresh; let z ← fresh; let sqs ← fresh; let sqnf ← fresh; let sq ← fresh
          let exd ← fresh; let exbb ← fresh; let ex ← fresh; let a1s ← fresh; let a1 ← fresh
          let h2d ← fresh; let h2bb ← fresh; let h2 ← fresh; let gate ← fresh
          let dser ← fresh; let gb2 ← fresh; let dleft ← fresh; let xdse ← fresh; let dgate ← fresh
          let one ← fresh; let omg ← fresh; let sg ← fresh; let dh2 ← fresh; let da1 ← fresh
          let dexs ← fresh; let dexone ← fresh; let dexom ← fresh; let dexxom ← fresh; let dexin ← fresh
          let dexsp ← fresh; let dex ← fresh; let dsq ← fresh; let dsqnf ← fresh; let dsqd ← fresh
          let dgsp ← fresh; let dds ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {sqs} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {sqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {sq} = stablehlo.divide {sqs}, {sqnf} : {ty [B,c]}\n" ++
            s!"    {exd} = stablehlo.dot_general {sq}, {w1}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [c,rr]}) -> {ty [B,rr]}\n" ++
            s!"    {exbb} = stablehlo.broadcast_in_dim {b1}, dims = [1] : ({ty [rr]}) -> {ty [B,rr]}\n" ++
            s!"    {ex} = stablehlo.add {exd}, {exbb} : {ty [B,rr]}\n" ++
            s!"    {a1s} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {a1} = stablehlo.multiply {ex}, {a1s} : {ty [B,rr]}\n" ++
            s!"    {h2d} = stablehlo.dot_general {a1}, {w2}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [rr,c]}) -> {ty [B,c]}\n" ++
            s!"    {h2bb} = stablehlo.broadcast_in_dim {b2}, dims = [1] : ({ty [c]}) -> {ty [B,c]}\n" ++
            s!"    {h2} = stablehlo.add {h2d}, {h2bb} : {ty [B,c]}\n" ++
            s!"    {gate} = stablehlo.logistic {h2} : {ty [B,c]}\n" ++
            s!"    {dser} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {gb2} = stablehlo.broadcast_in_dim {gate}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dleft} = stablehlo.multiply {gb2}, {dser} : {ty [B,c,h,w]}\n" ++
            s!"    {xdse} = stablehlo.multiply {xr}, {dser} : {ty [B,c,h,w]}\n" ++
            s!"    {dgate} = stablehlo.reduce({xdse} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n" ++
            s!"    {one} = stablehlo.constant dense<1.0> : {ty [B,c]}\n" ++
            s!"    {omg} = stablehlo.subtract {one}, {gate} : {ty [B,c]}\n" ++
            s!"    {sg} = stablehlo.multiply {gate}, {omg} : {ty [B,c]}\n" ++
            s!"    {dh2} = stablehlo.multiply {dgate}, {sg} : {ty [B,c]}\n" ++
            s!"    {da1} = stablehlo.dot_general {dh2}, {w2}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,c]}, {ty [rr,c]}) -> {ty [B,rr]}\n" ++
            s!"    {dexs} = stablehlo.logistic {ex} : {ty [B,rr]}\n" ++
            s!"    {dexone} = stablehlo.constant dense<1.0> : {ty [B,rr]}\n" ++
            s!"    {dexom} = stablehlo.subtract {dexone}, {dexs} : {ty [B,rr]}\n" ++
            s!"    {dexxom} = stablehlo.multiply {ex}, {dexom} : {ty [B,rr]}\n" ++
            s!"    {dexin} = stablehlo.add {dexone}, {dexxom} : {ty [B,rr]}\n" ++
            s!"    {dexsp} = stablehlo.multiply {dexs}, {dexin} : {ty [B,rr]}\n" ++
            s!"    {dex} = stablehlo.multiply {da1}, {dexsp} : {ty [B,rr]}\n" ++
            s!"    {dsq} = stablehlo.dot_general {dex}, {w1}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [B,rr]}, {ty [c,rr]}) -> {ty [B,c]}\n" ++
            s!"    {dsqnf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c]}\n" ++
            s!"    {dsqd} = stablehlo.divide {dsq}, {dsqnf} : {ty [B,c]}\n" ++
            s!"    {dgsp} = stablehlo.broadcast_in_dim {dsqd}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dds} = stablehlo.add {dleft}, {dgsp} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dds} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | "bnGammaSgd", [gN, vN, es, lrS], [_N, oc, h, w] => do
          -- BN γ update: recompute x̂ from the BN input `vN`, dγ = reduce[0,2,3](dy⊙x̂),
          -- γ' = γ − lr·dγ. Output is the channel-shaped updated γ.
          let xr ← fresh; let z ← fresh; let nf ← fresh; let smr ← fresh; let sm ← fresh
          let mu ← fresh; let xc ← fresh; let sq ← fresh; let vsr ← fresh; let vs ← fresh
          let vr ← fresh; let ep ← fresh; let ve ← fresh; let istd ← fresh; let xh ← fresh
          let dyr ← fresh; let dgp ← fresh; let dg ← fresh; let lc ← fresh; let sc ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {vN} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {nf} = stablehlo.constant dense<{B*h*w}.0> : {ty [B,oc,h,w]}\n" ++
            s!"    {smr} = stablehlo.reduce({xr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {sm} = stablehlo.broadcast_in_dim {smr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {mu} = stablehlo.divide {sm}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {xc} = stablehlo.subtract {xr}, {mu} : {ty [B,oc,h,w]}\n" ++
            s!"    {sq} = stablehlo.multiply {xc}, {xc} : {ty [B,oc,h,w]}\n" ++
            s!"    {vsr} = stablehlo.reduce({sq} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {vs} = stablehlo.broadcast_in_dim {vsr}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {vr} = stablehlo.divide {vs}, {nf} : {ty [B,oc,h,w]}\n" ++
            s!"    {ep} = stablehlo.constant dense<{es}> : {ty [B,oc,h,w]}\n" ++
            s!"    {ve} = stablehlo.add {vr}, {ep} : {ty [B,oc,h,w]}\n" ++
            s!"    {istd} = stablehlo.rsqrt {ve} : {ty [B,oc,h,w]}\n" ++
            s!"    {xh} = stablehlo.multiply {xc}, {istd} : {ty [B,oc,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {dgp} = stablehlo.multiply {dyr}, {xh} : {ty [B,oc,h,w]}\n" ++
            s!"    {dg} = stablehlo.reduce({dgp} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lc} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sc} = stablehlo.multiply {dg}, {lc} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {gN}, {sc} : {ty [oc]}\n", o :: st)
      | "bnBetaSgd", [bN, lrS], [_N, oc, h, w] => do
          -- BN β update: dβ = reduce[0,2,3](dy), β' = β − lr·dβ.
          let dyr ← fresh; let z ← fresh; let db ← fresh; let lc ← fresh; let sc ← fresh; let o ← fresh
          pure (
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {db} = stablehlo.reduce({dyr} init: {z}) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [oc]}\n" ++
            s!"    {lc} = stablehlo.constant dense<{lrS}> : {ty [oc]}\n" ++
            s!"    {sc} = stablehlo.multiply {db}, {lc} : {ty [oc]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sc} : {ty [oc]}\n", o :: st)
      | "denseWeightSgd", [xN, wN, lrS], [_N, a, c] => do
          -- dense weight update: dW = aᵀ·dy (dot_general contracts the batch), W' = W − lr·dW.
          let dW ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {dW} = stablehlo.dot_general {xN}, {r}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,a]}, {ty [B,c]}) -> {ty [a,c]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [a,c]}\n" ++
            s!"    {sW} = stablehlo.multiply {dW}, {lW} : {ty [a,c]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [a,c]}\n", o :: st)
      | "denseBiasSgd", [bN, lrS], [_N, c] => do
          -- dense bias update: dβ = reduce[0](dy) (sum over batch), β' = β − lr·dβ.
          let z ← fresh; let dB ← fresh; let lB ← fresh; let sB ← fresh; let o ← fresh
          pure (
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {dB} = stablehlo.reduce({r} init: {z}) applies stablehlo.add across dimensions = [0] : ({ty [B,c]}, tensor<f32>) -> {ty [c]}\n" ++
            s!"    {lB} = stablehlo.constant dense<{lrS}> : {ty [c]}\n" ++
            s!"    {sB} = stablehlo.multiply {dB}, {lB} : {ty [c]}\n" ++
            s!"    {o} = stablehlo.subtract {bN}, {sB} : {ty [c]}\n", o :: st)
      | "convWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          -- conv weight update via the transpose-trick wgrad (batch as the conv
          -- contraction), then W' = W − lr·dW. Same text as the per-example convWeightSgd.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*h*w]}) -> {ty [B,ic,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,h,w]}) -> {ty [ic,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,oc,h,w]}) -> {ty [oc,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,h,w]}, {ty [oc,B,h,w]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      | "convStridedWeightSgd", [xN, wN, lrS], [_N, ic, oc, h, w, kH, kW] => do
          -- stem 3×3 s2 weight: zero-upsample dy to 2h×2w then the transpose-trick wgrad.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, ic*(2*h)*(2*w)]}) -> {ty [B,ic,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, oc*h*w]}) -> {ty [B,oc,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,oc,h,w]}, tensor<f32>) -> {ty [B,oc,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,ic,2*h,2*w]}) -> {ty [ic,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,oc,2*h,2*w]}) -> {ty [oc,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [ic,B,2*h,2*w]}, {ty [oc,B,2*h,2*w]}) -> {ty [ic,oc,kH,kW]}\n" ++
            s!"    {g} = stablehlo.transpose {raw}, dims = [1, 0, 2, 3] : ({ty [ic,oc,kH,kW]}) -> {ty [oc,ic,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [oc,ic,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [oc,ic,kH,kW]}\n", o :: st)
      | "depthwiseWeightSgd", [xN, wN, lrS], [_N, c, h, w, kH, kW] => do
          -- depthwise weight: per-channel transpose-trick wgrad (batch_group_count=c).
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {dt} = stablehlo.transpose {dr}, dims = [1, 0, 2, 3] : ({ty [B,c,h,w]}) -> {ty [c,B,h,w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,h,w]}, {ty [c,B,h,w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
      | "depthwiseStridedWeightSgd", [xN, wN, lrS], [_N, c, h, w, kH, kW] => do
          -- strided depthwise weight: upsample dy to 2h×2w then the per-channel wgrad.
          let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
          let xr ← fresh; let dr ← fresh; let z ← fresh; let du ← fresh; let xt ← fresh; let dt ← fresh
          let raw ← fresh; let g ← fresh; let lW ← fresh; let sW ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*(2*h)*(2*w)]}) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {dr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {du} = stablehlo.pad {dr}, {z}, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c,2*h,2*w]}\n" ++
            s!"    {xt} = stablehlo.transpose {xr}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {dt} = stablehlo.transpose {du}, dims = [1, 0, 2, 3] : ({ty [B,c,2*h,2*w]}) -> {ty [c,B,2*h,2*w]}\n" ++
            s!"    {raw} = stablehlo.convolution({xt}, {dt})\n" ++
            "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
            s!"      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
            "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
            s!" : ({ty [c,B,2*h,2*w]}, {ty [c,B,2*h,2*w]}) -> {ty [1,c,kH,kW]}\n" ++
            s!"    {g} = stablehlo.reshape {raw} : ({ty [1,c,kH,kW]}) -> {ty [c,1,kH,kW]}\n" ++
            s!"    {lW} = stablehlo.constant dense<{lrS}> : {ty [c,1,kH,kW]}\n" ++
            s!"    {sW} = stablehlo.multiply {g}, {lW} : {ty [c,1,kH,kW]}\n" ++
            s!"    {o} = stablehlo.subtract {wN}, {sW} : {ty [c,1,kH,kW]}\n", o :: st)
      | "seReduceB", [xN], [_N, c, h, w] => do
          -- SE gate cotangent: dgate = reduce[2,3](x ⊙ dy). `xN` = SE input, `r` = the
          -- SE-output cotangent dy. Output is the per-example per-channel gate cotangent
          -- [B,c] (= the broadcast-adjoint of the Hadamard x⊙dy). Feeds the SE param grads.
          let xr ← fresh; let dyr ← fresh; let z ← fresh; let xd ← fresh; let o ← fresh
          pure (
            s!"    {xr} = stablehlo.reshape {xN} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {dyr} = stablehlo.reshape {r} : ({ty [B, c*h*w]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {z} = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
            s!"    {xd} = stablehlo.multiply {xr}, {dyr} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reduce({xd} init: {z}) applies stablehlo.add across dimensions = [2, 3] : ({ty [B,c,h,w]}, tensor<f32>) -> {ty [B,c]}\n", o :: st)
      | "gapBackBatched", [], [_N, c, h, w] => do
          -- GAP backward: broadcast the per-channel cotangent `r` ([B,c]) over the h×w
          -- grid and scale by 1/(h·w) — the `globalAvgPoolFlat` adjoint, batched.
          let bb ← fresh; let nf ← fresh; let dv ← fresh; let o ← fresh
          pure (
            s!"    {bb} = stablehlo.broadcast_in_dim {r}, dims = [0, 1] : ({ty [B,c]}) -> {ty [B,c,h,w]}\n" ++
            s!"    {nf} = stablehlo.constant dense<{h*w}.0> : {ty [B,c,h,w]}\n" ++
            s!"    {dv} = stablehlo.divide {bb}, {nf} : {ty [B,c,h,w]}\n" ++
            s!"    {o} = stablehlo.reshape {dv} : ({ty [B,c,h,w]}) -> {ty [B, c*h*w]}\n", o :: st)
      | _, _, _ =>
          pure (s!"    // [EfficientNet Item B] batched {tag} {names} {info} — backward render TODO\n", r :: st)
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

/-- **The linear train step rendered ENTIRELY from the verified AST.** Unlike
    `linearTrainStepModuleV` (forward via `pretty`, tail hand-written), here the
    *whole* module is `pretty` of denoted nodes: the cotangent (`lossCotGraph`,
    rendered once → shared `%dy`), then the two fused SGD ops `weightSgd`/`biasSgd`
    that consume `%dy`. So every emitted line is `pretty(provenNode)` and
    `LinearFaithfulPoC` proves the two outputs' `den` = the certified loss-descent
    SGD step. The `lr` ℝ / operand values are `skel`-erased (render is
    value-independent), so placeholders here render identically to the live graph
    the `den` theorems use. -/
def linTrainStepFaithfulV (B m n : Nat) (lrStr : String)
    (W : Mat m n) (b : Vec n) (x : Vec m) : String :=
  -- FULLY TIED: each SGD op consumes the proven `lossCotGraph` node DIRECTLY (not a
  -- name-pinned `.operand %dy <placeholder>`), so `den(output) = certified` is one composed
  -- theorem with the forward = the proven `fwdGraph` (nested inside `lossCotGraph`) — no
  -- SSA-name pin. The shared cotangent is rendered once per output (2× here); iree CSEs it.
  let act : StateM Nat (String × String × String) := do
    let (wBody, wRes) ← pretty B (SHlo.weightSgd "%x" "%W0" lrStr x W 0 (lossCotGraph W b x (fun _ => 0)))
    let (bBody, bRes) ← pretty B (SHlo.biasSgd "%b0" lrStr b 0 (lossCotGraph W b x (fun _ => 0)))
    pure (wBody ++ bBody, wRes, bRes)
  let (body, wRes, bRes) := act.run' 0
  "module @m {\n" ++
  s!"  func.func @linear_train_step(%x: {ty [B,m]}, %W0: {ty [m,n]}, %b0: {ty [n]}, " ++
  s!"%onehot: {ty [B,n]}) -> ({ty [m,n]}, {ty [n]}) " ++ "{\n" ++
  "    // ── linear train step: every line is pretty(verified AST node) ──\n" ++
  body ++
  s!"    return {wRes}, {bRes} : {ty [m,n]}, {ty [n]}\n" ++
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
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (x : Vec (ic*(2*(2*h))*(2*(2*w)))) : String :=
  renderModule "cifar_bn_fwd"
    s!"%x: {ty [B,ic*(2*(2*h))*(2*(2*w))]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c2*h*w,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}"
    B nClasses (cifarBnFwdGraph epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
      W₅ b₅ W₆ b₆ W₇ b₇ x)

/-- `@cifar8_fwd` rendered from the verified 8-conv CIFAR forward AST `cifar8FwdGraph`
    (`cifar8FwdGraph_faithful` proves it denotes `cifarCnn8Forward`). The 4-stage peer of
    `cifarFwdModuleV` — closes the cifar8 `_fwd` bytes (committed `verified_mlir/cifar8_fwd.mlir`
    is now `renderModule(provenGraph)`, replacing the hand-written `cifar8FwdText`). -/
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
    The BN peer of `cifar8FwdModuleV` — closes the cifar8-bn `_fwd` bytes, replacing the
    hand-written `cifar8BnFwdTextPC`. -/
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
  let M1 := c1 * H * W            -- stage-1 flattened feature size (= c1·S1)
  let M2 := c2 * H2 * W2          -- stage-2 flattened feature size (= c2·S2)
  let S1 := H * W                 -- stage-1 per-channel spatial size
  let S2 := H2 * W2               -- stage-2 per-channel spatial size
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
  -- Per-channel BN forward: reshape [B,C·S]→[B,C,S], reduce μ/var over the spatial
  -- axis [2] per channel, normalize, per-channel affine (γ,β : [C]), reshape back to
  -- [B,C·S]. Saves {o}_xhat,_istd,_nf in [B,C,S] for the backward. (= bnPerChannelFlat.)
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  -- Per-channel BN input-VJP (consolidated three-term form, reductions over spatial
  -- axis [2]); reuses {bn}_xhat/_istd/_nf ([B,C,S]). Output reshaped back to [B,C·S].
  let bnBack (o bn g dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_dxh = stablehlo.multiply {o}_gb, {o}_dyr : {ty [B,C,S]}\n" ++
    s!"    {o}_sdxr = stablehlo.reduce({o}_dxh init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sdx = stablehlo.broadcast_in_dim {o}_sdxr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_xd = stablehlo.multiply {bn}_xhat, {o}_dxh : {ty [B,C,S]}\n" ++
    s!"    {o}_sxdr = stablehlo.reduce({o}_xd init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sxd = stablehlo.broadcast_in_dim {o}_sxdr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_t1 = stablehlo.multiply {o}_dxh, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_i1 = stablehlo.subtract {o}_t1, {o}_sdx : {ty [B,C,S]}\n" ++
    s!"    {o}_xs = stablehlo.multiply {bn}_xhat, {o}_sxd : {ty [B,C,S]}\n" ++
    s!"    {o}_i2 = stablehlo.subtract {o}_i1, {o}_xs : {ty [B,C,S]}\n" ++
    s!"    {o}_s = stablehlo.divide {bn}_istd, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_dx3 = stablehlo.multiply {o}_s, {o}_i2 : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_dx3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  -- Per-channel BN param grads dγ_c = Σ_{b,s} dy·x̂, dβ_c = Σ_{b,s} dy (reduce [0,2] → [C]).
  let bnParamGrad (dgr dbe bn dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {dgr}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {dgr}_p = stablehlo.multiply {dgr}_dyr, {bn}_xhat : {ty [B,C,S]}\n" ++
    s!"    {dgr} = stablehlo.reduce({dgr}_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n" ++
    s!"    {dbe} = stablehlo.reduce({dgr}_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n"
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar_bn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→BN→relu)×2→pool →(conv→BN→relu)×2→pool →flatten→(dense→relu)×2→dense ──\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
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
  bnBack "%dhc4f" "%bn4" "%g4" "%dbn4" c2 S2 ++ bnParamGrad "%dg4" "%dbt4" "%bn4" "%dbn4" c2 S2 ++
  rs "%dhc4" "%dhc4f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++ rs "%dac3f" "%dac3" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn3" "%bn3" "%dac3f" M2 ++
  bnBack "%dhc3f" "%bn3" "%g3" "%dbn3" c2 S2 ++ bnParamGrad "%dg3" "%dbt3" "%bn3" "%dbn3" c2 S2 ++
  rs "%dhc3" "%dhc3f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++ rs "%dac2f" "%dac2" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn2" "%bn2" "%dac2f" M1 ++
  bnBack "%dhc2f" "%bn2" "%g2" "%dbn2" c1 S1 ++ bnParamGrad "%dg2" "%dbt2" "%bn2" "%dbn2" c1 S1 ++
  rs "%dhc2" "%dhc2f" [B,M1] [B,c1,H,W] ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++ rs "%dac1f" "%dac1" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn1" "%bn1" "%dac1f" M1 ++
  bnBack "%dhc1f" "%bn1" "%g1" "%dbn1" c1 S1 ++ bnParamGrad "%dg1" "%dbt1" "%bn1" "%dbn1" c1 S1 ++
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
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++ sgd "%g1" "%dg1" (ty [c1]) ++ sgd "%bt1" "%dbt1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++ sgd "%g2" "%dg2" (ty [c1]) ++ sgd "%bt2" "%dbt2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++ sgd "%g3" "%dg3" (ty [c2]) ++ sgd "%bt3" "%dbt3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++ sgd "%g4" "%dg4" (ty [c2]) ++ sgd "%bt4" "%dbt4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [flat,d1]) ++ sgd "%b5" "%db5" (ty [d1]) ++
  sgd "%W6" "%dW6" (ty [d1,d1]) ++ sgd "%b6" "%db6" (ty [d1]) ++
  sgd "%W7" "%dW7" (ty [d1,nClasses]) ++ sgd "%b7" "%db7" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- Per-channel **BN-CIFAR** eval forward (`@cifar_bn_fwd`): the forward half of
    `cifarBnTrainStepText` (conv→per-channel-BN→relu ×4, 2 pools, 3 dense), returning
    logits `[B,nClasses]`. Per-channel BN (`m=H·W`) is per-example ⇒ train=eval (no
    running stats). String-rendered (peer of the train-step) until the typed
    `cifarBnFwdGraph` is reconciled to per-channel in the proof pass. -/
def cifarBnFwdTextPC (B ic c1 c2 H W kH kW d1 nClasses : Nat) (epsStr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2; let W2 := W / 2
  let Hp := H2 / 2; let Wp := W2 / 2
  let flat := c2 * Hp * Wp
  let M1 := c1 * H * W; let M2 := c2 * H2 * W2
  let S1 := H * W; let S2 := H2 * W2
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    s!"    {oh}d = stablehlo.dot_general {a}, {w}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,mm]}, {ty [mm,nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  -- per-channel BN forward (reshape [B,C·S]→[B,C,S], reduce spatial [2], affine γ,β:[C]).
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  "module @m {\n" ++
  s!"  func.func @cifar_bn_fwd(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [flat,d1]}, %b5: {ty [d1]}, %W6: {ty [d1,d1]}, %b6: {ty [d1]}, %W7: {ty [d1,nClasses]}, %b7: {ty [nClasses]}) -> {ty [B,nClasses]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  rs "%flat" "%pool2" [B,c2,Hp,Wp] [B,flat] ++
  dense "%h5" "%flat" "%W5" "%b5" flat d1 ++ relu2 "%a5" "%h5" d1 ++
  dense "%h6" "%a5" "%W6" "%b6" d1 d1 ++ relu2 "%a6" "%h6" d1 ++
  dense "%logits" "%a6" "%W7" "%b7" d1 nClasses ++
  s!"    return %logits : {ty [B,nClasses]}\n" ++
  "  }\n}\n"

/-! ### Deeper 8-conv CIFAR CNN (FOUR conv→conv→pool stages) train-step + fwd text

The 4-stage peers of `cifarTrainStepText` / `cifarBnTrainStepText` and their forwards.
Channels `c1 c2 c3 c4`; spatial `H → H/2 → H/4 → H/8 → H/16` (CIFAR 32→16→8→4→2). The
forward is `(conv→[BN→]relu)×2 → pool` four times → flatten `c4·Hp·Wp` → 3-dense head; the
backward is the exact transpose/reverse mirror (the same op templates as the 2-stage text).
The whole-net VJPs are `Proofs.cifarCnn8_has_vjp_at` / `cifarCnnBn8_has_vjp_at`. `lr = 0.1/B`. -/

/-- 8-conv CIFAR train step (`@cifar8_train_step`, no BN). 4 conv→conv→pool stages
    (channels `ic→c1→c1`, `c1→c2→c2`, `c2→c3→c3`, `c3→c4→c4`) + 3-dense head. -/
def cifar8TrainStepText (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) (lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2          -- after pool1 (16)
  let H3 := H2 / 2; let W3 := W2 / 2         -- after pool2 (8)
  let H4 := H3 / 2; let W4 := W3 / 2         -- after pool3 (4)
  let Hp := H4 / 2; let Wp := W4 / 2         -- after pool4 (2)
  let flat := c4 * Hp * Wp
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
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ relu4 "%ac1" "%hc1" c1 H W ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ relu4 "%ac2" "%hc2" c1 H W ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ relu4 "%ac3" "%hc3" c2 H2 W2 ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ relu4 "%ac4" "%hc4" c2 H2 W2 ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ relu4 "%ac5" "%hc5" c3 H3 W3 ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ relu4 "%ac6" "%hc6" c3 H3 W3 ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ relu4 "%ac7" "%hc7" c4 H4 W4 ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ relu4 "%ac8" "%hc8" c4 H4 W4 ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  s!"    %flat = stablehlo.reshape %pool4 : ({ty [B,c4,Hp,Wp]}) -> {ty [B,flat]}\n" ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense (dotOut)+relu masks → scatter → convBack, four stages ──\n" ++
  dg "%dxb" "%dy" "%Wb" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dya" "%ha" "%dxb" d1 ++
  dg "%dxa" "%dya" "%Wa" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy9" "%h9" "%dxa" d1 ++
  dg "%dx9" "%dy9" "%W9" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  s!"    %dpool4 = stablehlo.reshape %dx9 : ({ty [B,flat]}) -> {ty [B,c4,Hp,Wp]}\n" ++
  -- stage 4
  scatter "%dac8" "%ac8" "%dpool4" c4 H4 W4 ++
  selMask4 "%dhc8" "%hc8" "%dac8" c4 H4 W4 ++
  convBack "%dac7" "%dhc8" "%W8" c4 c4 H4 W4 ++
  selMask4 "%dhc7" "%hc7" "%dac7" c4 H4 W4 ++
  convBack "%dpool3" "%dhc7" "%W7" c3 c4 H4 W4 ++
  -- stage 3
  scatter "%dac6" "%ac6" "%dpool3" c3 H3 W3 ++
  selMask4 "%dhc6" "%hc6" "%dac6" c3 H3 W3 ++
  convBack "%dac5" "%dhc6" "%W6" c3 c3 H3 W3 ++
  selMask4 "%dhc5" "%hc5" "%dac5" c3 H3 W3 ++
  convBack "%dpool2" "%dhc5" "%W5" c2 c3 H3 W3 ++
  -- stage 2
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++
  selMask4 "%dhc4" "%hc4" "%dac4" c2 H2 W2 ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++
  selMask4 "%dhc3" "%hc3" "%dac3" c2 H2 W2 ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  -- stage 1
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++
  selMask4 "%dhc2" "%hc2" "%dac2" c1 H W ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++
  selMask4 "%dhc1" "%hc1" "%dac1" c1 H W ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dWb" "%aa" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%dbb" "%dy" nClasses ++
  dg "%dWa" "%a9" "%dya" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%dba" "%dya" d1 ++
  dg "%dW9" "%flat" "%dy9" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db9" "%dy9" d1 ++
  convWGrad "%dW8" "%ac7" "%dhc8" c4 c4 H4 W4 ++ convBiasGrad "%db8" "%dhc8" c4 H4 W4 ++
  convWGrad "%dW7" "%pool3" "%dhc7" c3 c4 H4 W4 ++ convBiasGrad "%db7" "%dhc7" c4 H4 W4 ++
  convWGrad "%dW6" "%ac5" "%dhc6" c3 c3 H3 W3 ++ convBiasGrad "%db6" "%dhc6" c3 H3 W3 ++
  convWGrad "%dW5" "%pool2" "%dhc5" c2 c3 H3 W3 ++ convBiasGrad "%db5" "%dhc5" c3 H3 W3 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 22 params) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [c3,c2,kH,kW]) ++ sgd "%b5" "%db5" (ty [c3]) ++
  sgd "%W6" "%dW6" (ty [c3,c3,kH,kW]) ++ sgd "%b6" "%db6" (ty [c3]) ++
  sgd "%W7" "%dW7" (ty [c4,c3,kH,kW]) ++ sgd "%b7" "%db7" (ty [c4]) ++
  sgd "%W8" "%dW8" (ty [c4,c4,kH,kW]) ++ sgd "%b8" "%db8" (ty [c4]) ++
  sgd "%W9" "%dW9" (ty [flat,d1]) ++ sgd "%b9" "%db9" (ty [d1]) ++
  sgd "%Wa" "%dWa" (ty [d1,d1]) ++ sgd "%ba" "%dba" (ty [d1]) ++
  sgd "%Wb" "%dWb" (ty [d1,nClasses]) ++ sgd "%bb" "%dbb" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %W2n, %b2n, %W3n, %b3n, %W4n, %b4n, %W5n, %b5n, %W6n, %b6n, %W7n, %b7n, %W8n, %b8n, %W9n, %b9n, %Wan, %ban, %Wbn, %bbn : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- 8-conv CIFAR eval forward (`@cifar8_fwd`, no BN), returning logits `[B,nClasses]`. -/
def cifar8FwdText (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := c4 * Hp * Wp
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    s!"    {oh}d = stablehlo.dot_general {a}, {w}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,mm]}, {ty [mm,nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let relu4 (o h : String) (C Hh Ww : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,C,Hh,Ww]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_fwd(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}) -> {ty [B,nClasses]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,ic*H*W]}) -> {ty [B,ic,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ relu4 "%ac1" "%hc1" c1 H W ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ relu4 "%ac2" "%hc2" c1 H W ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ relu4 "%ac3" "%hc3" c2 H2 W2 ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ relu4 "%ac4" "%hc4" c2 H2 W2 ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ relu4 "%ac5" "%hc5" c3 H3 W3 ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ relu4 "%ac6" "%hc6" c3 H3 W3 ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ relu4 "%ac7" "%hc7" c4 H4 W4 ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ relu4 "%ac8" "%hc8" c4 H4 W4 ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  s!"    %flat = stablehlo.reshape %pool4 : ({ty [B,c4,Hp,Wp]}) -> {ty [B,flat]}\n" ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  s!"    return %logits : {ty [B,nClasses]}\n" ++
  "  }\n}\n"

/-- 8-conv CIFAR **per-channel BN** train step (`@cifar8_bn_train_step`). Each of the 8
    convs is followed by `bnFwd` (per-channel BN, reduce spatial axis [2]); the backward
    inserts the relu-mask → BN input-VJP (`bnBack`) → conv-back per block + BN param grads
    (`dγ=Σ dy·x̂, dβ=Σ dy`). 38 params (8×{W,b,γ,β} + 3×{W,b}). Whole-net VJP:
    `Proofs.cifarCnnBn8_has_vjp_at`. `lr = 0.1/B`. -/
def cifar8BnTrainStepText (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) (epsStr lr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := c4 * Hp * Wp
  let M1 := c1 * H * W;   let S1 := H * W
  let M2 := c2 * H2 * W2; let S2 := H2 * W2
  let M3 := c3 * H3 * W3; let S3 := H3 * W3
  let M4 := c4 * H4 * W4; let S4 := H4 * W4
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
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  let bnBack (o bn g dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_dxh = stablehlo.multiply {o}_gb, {o}_dyr : {ty [B,C,S]}\n" ++
    s!"    {o}_sdxr = stablehlo.reduce({o}_dxh init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sdx = stablehlo.broadcast_in_dim {o}_sdxr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_xd = stablehlo.multiply {bn}_xhat, {o}_dxh : {ty [B,C,S]}\n" ++
    s!"    {o}_sxdr = stablehlo.reduce({o}_xd init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sxd = stablehlo.broadcast_in_dim {o}_sxdr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_t1 = stablehlo.multiply {o}_dxh, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_i1 = stablehlo.subtract {o}_t1, {o}_sdx : {ty [B,C,S]}\n" ++
    s!"    {o}_xs = stablehlo.multiply {bn}_xhat, {o}_sxd : {ty [B,C,S]}\n" ++
    s!"    {o}_i2 = stablehlo.subtract {o}_i1, {o}_xs : {ty [B,C,S]}\n" ++
    s!"    {o}_s = stablehlo.divide {bn}_istd, {bn}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_dx3 = stablehlo.multiply {o}_s, {o}_i2 : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_dx3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  let bnParamGrad (dgr dbe bn dyf : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {dgr}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {dgr}_p = stablehlo.multiply {dgr}_dyr, {bn}_xhat : {ty [B,C,S]}\n" ++
    s!"    {dgr} = stablehlo.reduce({dgr}_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n" ++
    s!"    {dbe} = stablehlo.reduce({dgr}_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n"
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let sgd (θ dθ ty' : String) : String :=
    s!"    {θ}l = stablehlo.constant dense<{lr}> : {ty'}\n" ++
    s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
    s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_bn_train_step(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}, %onehot: {ty [B,nClasses]}) -> ({ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}) " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→BN→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ rs "%hc5f" "%hc5" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn5" "%hc5f" "%g5" "%bt5" c3 S3 ++ relu2 "%ac5f" "%bn5" M3 ++ rs "%ac5" "%ac5f" [B,M3] [B,c3,H3,W3] ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ rs "%hc6f" "%hc6" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn6" "%hc6f" "%g6" "%bt6" c3 S3 ++ relu2 "%ac6f" "%bn6" M3 ++ rs "%ac6" "%ac6f" [B,M3] [B,c3,H3,W3] ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ rs "%hc7f" "%hc7" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn7" "%hc7f" "%g7" "%bt7" c4 S4 ++ relu2 "%ac7f" "%bn7" M4 ++ rs "%ac7" "%ac7f" [B,M4] [B,c4,H4,W4] ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ rs "%hc8f" "%hc8" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn8" "%hc8f" "%g8" "%bt8" c4 S4 ++ relu2 "%ac8f" "%bn8" M4 ++ rs "%ac8" "%ac8f" [B,M4] [B,c4,H4,W4] ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  rs "%flat" "%pool4" [B,c4,Hp,Wp] [B,flat] ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  "    // ── loss cotangent dy = softmax(logits) − onehot ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,nClasses]}\n" ++
  s!"    %dy = stablehlo.subtract %lsm, %onehot : {ty [B,nClasses]}\n" ++
  "    // ── backward: dense+relu → scatter → (relu→BN-back→convBack)×stage, four stages ──\n" ++
  dg "%dxb" "%dy" "%Wb" "1" "1" (ty [B,nClasses]) (ty [d1,nClasses]) (ty [B,d1]) ++
  selMask2 "%dya" "%ha" "%dxb" d1 ++
  dg "%dxa" "%dya" "%Wa" "1" "1" (ty [B,d1]) (ty [d1,d1]) (ty [B,d1]) ++
  selMask2 "%dy9" "%h9" "%dxa" d1 ++
  dg "%dx9" "%dy9" "%W9" "1" "1" (ty [B,d1]) (ty [flat,d1]) (ty [B,flat]) ++
  rs "%dpool4" "%dx9" [B,flat] [B,c4,Hp,Wp] ++
  -- stage 4
  scatter "%dac8" "%ac8" "%dpool4" c4 H4 W4 ++ rs "%dac8f" "%dac8" [B,c4,H4,W4] [B,M4] ++
  selMask2 "%dbn8" "%bn8" "%dac8f" M4 ++
  bnBack "%dhc8f" "%bn8" "%g8" "%dbn8" c4 S4 ++ bnParamGrad "%dg8" "%dbt8" "%bn8" "%dbn8" c4 S4 ++
  rs "%dhc8" "%dhc8f" [B,M4] [B,c4,H4,W4] ++
  convBack "%dac7" "%dhc8" "%W8" c4 c4 H4 W4 ++ rs "%dac7f" "%dac7" [B,c4,H4,W4] [B,M4] ++
  selMask2 "%dbn7" "%bn7" "%dac7f" M4 ++
  bnBack "%dhc7f" "%bn7" "%g7" "%dbn7" c4 S4 ++ bnParamGrad "%dg7" "%dbt7" "%bn7" "%dbn7" c4 S4 ++
  rs "%dhc7" "%dhc7f" [B,M4] [B,c4,H4,W4] ++
  convBack "%dpool3" "%dhc7" "%W7" c3 c4 H4 W4 ++
  -- stage 3
  scatter "%dac6" "%ac6" "%dpool3" c3 H3 W3 ++ rs "%dac6f" "%dac6" [B,c3,H3,W3] [B,M3] ++
  selMask2 "%dbn6" "%bn6" "%dac6f" M3 ++
  bnBack "%dhc6f" "%bn6" "%g6" "%dbn6" c3 S3 ++ bnParamGrad "%dg6" "%dbt6" "%bn6" "%dbn6" c3 S3 ++
  rs "%dhc6" "%dhc6f" [B,M3] [B,c3,H3,W3] ++
  convBack "%dac5" "%dhc6" "%W6" c3 c3 H3 W3 ++ rs "%dac5f" "%dac5" [B,c3,H3,W3] [B,M3] ++
  selMask2 "%dbn5" "%bn5" "%dac5f" M3 ++
  bnBack "%dhc5f" "%bn5" "%g5" "%dbn5" c3 S3 ++ bnParamGrad "%dg5" "%dbt5" "%bn5" "%dbn5" c3 S3 ++
  rs "%dhc5" "%dhc5f" [B,M3] [B,c3,H3,W3] ++
  convBack "%dpool2" "%dhc5" "%W5" c2 c3 H3 W3 ++
  -- stage 2
  scatter "%dac4" "%ac4" "%dpool2" c2 H2 W2 ++ rs "%dac4f" "%dac4" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn4" "%bn4" "%dac4f" M2 ++
  bnBack "%dhc4f" "%bn4" "%g4" "%dbn4" c2 S2 ++ bnParamGrad "%dg4" "%dbt4" "%bn4" "%dbn4" c2 S2 ++
  rs "%dhc4" "%dhc4f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dac3" "%dhc4" "%W4" c2 c2 H2 W2 ++ rs "%dac3f" "%dac3" [B,c2,H2,W2] [B,M2] ++
  selMask2 "%dbn3" "%bn3" "%dac3f" M2 ++
  bnBack "%dhc3f" "%bn3" "%g3" "%dbn3" c2 S2 ++ bnParamGrad "%dg3" "%dbt3" "%bn3" "%dbn3" c2 S2 ++
  rs "%dhc3" "%dhc3f" [B,M2] [B,c2,H2,W2] ++
  convBack "%dpool1" "%dhc3" "%W3" c1 c2 H2 W2 ++
  -- stage 1
  scatter "%dac2" "%ac2" "%dpool1" c1 H W ++ rs "%dac2f" "%dac2" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn2" "%bn2" "%dac2f" M1 ++
  bnBack "%dhc2f" "%bn2" "%g2" "%dbn2" c1 S1 ++ bnParamGrad "%dg2" "%dbt2" "%bn2" "%dbn2" c1 S1 ++
  rs "%dhc2" "%dhc2f" [B,M1] [B,c1,H,W] ++
  convBack "%dac1" "%dhc2" "%W2" c1 c1 H W ++ rs "%dac1f" "%dac1" [B,c1,H,W] [B,M1] ++
  selMask2 "%dbn1" "%bn1" "%dac1f" M1 ++
  bnBack "%dhc1f" "%bn1" "%g1" "%dbn1" c1 S1 ++ bnParamGrad "%dg1" "%dbt1" "%bn1" "%dbn1" c1 S1 ++
  rs "%dhc1" "%dhc1f" [B,M1] [B,c1,H,W] ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dWb" "%aa" "%dy" "0" "0" (ty [B,d1]) (ty [B,nClasses]) (ty [d1,nClasses]) ++ reduce0 "%dbb" "%dy" nClasses ++
  dg "%dWa" "%a9" "%dya" "0" "0" (ty [B,d1]) (ty [B,d1]) (ty [d1,d1]) ++ reduce0 "%dba" "%dya" d1 ++
  dg "%dW9" "%flat" "%dy9" "0" "0" (ty [B,flat]) (ty [B,d1]) (ty [flat,d1]) ++ reduce0 "%db9" "%dy9" d1 ++
  convWGrad "%dW8" "%ac7" "%dhc8" c4 c4 H4 W4 ++ convBiasGrad "%db8" "%dhc8" c4 H4 W4 ++
  convWGrad "%dW7" "%pool3" "%dhc7" c3 c4 H4 W4 ++ convBiasGrad "%db7" "%dhc7" c4 H4 W4 ++
  convWGrad "%dW6" "%ac5" "%dhc6" c3 c3 H3 W3 ++ convBiasGrad "%db6" "%dhc6" c3 H3 W3 ++
  convWGrad "%dW5" "%pool2" "%dhc5" c2 c3 H3 W3 ++ convBiasGrad "%db5" "%dhc5" c3 H3 W3 ++
  convWGrad "%dW4" "%ac3" "%dhc4" c2 c2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" c2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" c1 c2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" c2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" c1 c1 H W ++ convBiasGrad "%db2" "%dhc2" c1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" ic c1 H W ++ convBiasGrad "%db1" "%dhc1" c1 H W ++
  "    // ── SGD θ' = θ − lr·∇ (all 38 params, incl. per-channel γ/β) ──\n" ++
  sgd "%W1" "%dW1" (ty [c1,ic,kH,kW]) ++ sgd "%b1" "%db1" (ty [c1]) ++ sgd "%g1" "%dg1" (ty [c1]) ++ sgd "%bt1" "%dbt1" (ty [c1]) ++
  sgd "%W2" "%dW2" (ty [c1,c1,kH,kW]) ++ sgd "%b2" "%db2" (ty [c1]) ++ sgd "%g2" "%dg2" (ty [c1]) ++ sgd "%bt2" "%dbt2" (ty [c1]) ++
  sgd "%W3" "%dW3" (ty [c2,c1,kH,kW]) ++ sgd "%b3" "%db3" (ty [c2]) ++ sgd "%g3" "%dg3" (ty [c2]) ++ sgd "%bt3" "%dbt3" (ty [c2]) ++
  sgd "%W4" "%dW4" (ty [c2,c2,kH,kW]) ++ sgd "%b4" "%db4" (ty [c2]) ++ sgd "%g4" "%dg4" (ty [c2]) ++ sgd "%bt4" "%dbt4" (ty [c2]) ++
  sgd "%W5" "%dW5" (ty [c3,c2,kH,kW]) ++ sgd "%b5" "%db5" (ty [c3]) ++ sgd "%g5" "%dg5" (ty [c3]) ++ sgd "%bt5" "%dbt5" (ty [c3]) ++
  sgd "%W6" "%dW6" (ty [c3,c3,kH,kW]) ++ sgd "%b6" "%db6" (ty [c3]) ++ sgd "%g6" "%dg6" (ty [c3]) ++ sgd "%bt6" "%dbt6" (ty [c3]) ++
  sgd "%W7" "%dW7" (ty [c4,c3,kH,kW]) ++ sgd "%b7" "%db7" (ty [c4]) ++ sgd "%g7" "%dg7" (ty [c4]) ++ sgd "%bt7" "%dbt7" (ty [c4]) ++
  sgd "%W8" "%dW8" (ty [c4,c4,kH,kW]) ++ sgd "%b8" "%db8" (ty [c4]) ++ sgd "%g8" "%dg8" (ty [c4]) ++ sgd "%bt8" "%dbt8" (ty [c4]) ++
  sgd "%W9" "%dW9" (ty [flat,d1]) ++ sgd "%b9" "%db9" (ty [d1]) ++
  sgd "%Wa" "%dWa" (ty [d1,d1]) ++ sgd "%ba" "%dba" (ty [d1]) ++
  sgd "%Wb" "%dWb" (ty [d1,nClasses]) ++ sgd "%bb" "%dbb" (ty [nClasses]) ++
  s!"    return %W1n, %b1n, %g1n, %bt1n, %W2n, %b2n, %g2n, %bt2n, %W3n, %b3n, %g3n, %bt3n, %W4n, %b4n, %g4n, %bt4n, %W5n, %b5n, %g5n, %bt5n, %W6n, %b6n, %g6n, %bt6n, %W7n, %b7n, %g7n, %bt7n, %W8n, %b8n, %g8n, %bt8n, %W9n, %b9n, %Wan, %ban, %Wbn, %bbn : {ty [c1,ic,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c1,c1,kH,kW]}, {ty [c1]}, {ty [c1]}, {ty [c1]}, {ty [c2,c1,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c2,c2,kH,kW]}, {ty [c2]}, {ty [c2]}, {ty [c2]}, {ty [c3,c2,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c3,c3,kH,kW]}, {ty [c3]}, {ty [c3]}, {ty [c3]}, {ty [c4,c3,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [c4,c4,kH,kW]}, {ty [c4]}, {ty [c4]}, {ty [c4]}, {ty [flat,d1]}, {ty [d1]}, {ty [d1,d1]}, {ty [d1]}, {ty [d1,nClasses]}, {ty [nClasses]}\n" ++
  "  }\n}\n"

/-- 8-conv CIFAR **per-channel BN** eval forward (`@cifar8_bn_fwd`), returning logits. -/
def cifar8BnFwdTextPC (B ic c1 c2 c3 c4 H W kH kW d1 nClasses : Nat) (epsStr : String) : String :=
  let pH := (kH - 1) / 2; let pW := (kW - 1) / 2
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := c4 * Hp * Wp
  let M1 := c1 * H * W;   let S1 := H * W
  let M2 := c2 * H2 * W2; let S2 := H2 * W2
  let M3 := c3 * H3 * W3; let S3 := H3 * W3
  let M4 := c4 * H4 * W4; let S4 := H4 * W4
  let rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
    s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"
  let dense (oh a w bnm : String) (mm nn : Nat) : String :=
    s!"    {oh}d = stablehlo.dot_general {a}, {w}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,mm]}, {ty [mm,nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
    s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"
  let relu2 (o h : String) (nn : Nat) : String :=
    s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
    s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"
  let convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
    s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
    "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
    s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,kH,kW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
    s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"
  let maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
    s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
    s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"
  let bnFwd (o x g bt : String) (C S : Nat) : String :=
    let Mn := C * S
    s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
    s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
    s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
    s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
    s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
    s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
    s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
    s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
    s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
    s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
    s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
    s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
    s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"
  "module @m {\n" ++
  s!"  func.func @cifar8_bn_fwd(%x: {ty [B,ic*H*W]}, %W1: {ty [c1,ic,kH,kW]}, %b1: {ty [c1]}, %g1: {ty [c1]}, %bt1: {ty [c1]}, %W2: {ty [c1,c1,kH,kW]}, %b2: {ty [c1]}, %g2: {ty [c1]}, %bt2: {ty [c1]}, %W3: {ty [c2,c1,kH,kW]}, %b3: {ty [c2]}, %g3: {ty [c2]}, %bt3: {ty [c2]}, %W4: {ty [c2,c2,kH,kW]}, %b4: {ty [c2]}, %g4: {ty [c2]}, %bt4: {ty [c2]}, %W5: {ty [c3,c2,kH,kW]}, %b5: {ty [c3]}, %g5: {ty [c3]}, %bt5: {ty [c3]}, %W6: {ty [c3,c3,kH,kW]}, %b6: {ty [c3]}, %g6: {ty [c3]}, %bt6: {ty [c3]}, %W7: {ty [c4,c3,kH,kW]}, %b7: {ty [c4]}, %g7: {ty [c4]}, %bt7: {ty [c4]}, %W8: {ty [c4,c4,kH,kW]}, %b8: {ty [c4]}, %g8: {ty [c4]}, %bt8: {ty [c4]}, %W9: {ty [flat,d1]}, %b9: {ty [d1]}, %Wa: {ty [d1,d1]}, %ba: {ty [d1]}, %Wb: {ty [d1,nClasses]}, %bb: {ty [nClasses]}) -> {ty [B,nClasses]} " ++ "{\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  rs "%xr" "%x" [B,ic*H*W] [B,ic,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%b1" c1 ic H W ++ rs "%hc1f" "%hc1" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" c1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,c1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%b2" c1 c1 H W ++ rs "%hc2f" "%hc2" [B,c1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" c1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,c1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" c1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%b3" c2 c1 H2 W2 ++ rs "%hc3f" "%hc3" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" c2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,c2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%b4" c2 c2 H2 W2 ++ rs "%hc4f" "%hc4" [B,c2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" c2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,c2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" c2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%b5" c3 c2 H3 W3 ++ rs "%hc5f" "%hc5" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn5" "%hc5f" "%g5" "%bt5" c3 S3 ++ relu2 "%ac5f" "%bn5" M3 ++ rs "%ac5" "%ac5f" [B,M3] [B,c3,H3,W3] ++
  convFwd "%hc6" "%ac5" "%W6" "%b6" c3 c3 H3 W3 ++ rs "%hc6f" "%hc6" [B,c3,H3,W3] [B,M3] ++
  bnFwd "%bn6" "%hc6f" "%g6" "%bt6" c3 S3 ++ relu2 "%ac6f" "%bn6" M3 ++ rs "%ac6" "%ac6f" [B,M3] [B,c3,H3,W3] ++
  maxpoolFwd "%pool3" "%ac6" c3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%b7" c4 c3 H4 W4 ++ rs "%hc7f" "%hc7" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn7" "%hc7f" "%g7" "%bt7" c4 S4 ++ relu2 "%ac7f" "%bn7" M4 ++ rs "%ac7" "%ac7f" [B,M4] [B,c4,H4,W4] ++
  convFwd "%hc8" "%ac7" "%W8" "%b8" c4 c4 H4 W4 ++ rs "%hc8f" "%hc8" [B,c4,H4,W4] [B,M4] ++
  bnFwd "%bn8" "%hc8f" "%g8" "%bt8" c4 S4 ++ relu2 "%ac8f" "%bn8" M4 ++ rs "%ac8" "%ac8f" [B,M4] [B,c4,H4,W4] ++
  maxpoolFwd "%pool4" "%ac8" c4 H4 W4 ++
  rs "%flat" "%pool4" [B,c4,Hp,Wp] [B,flat] ++
  dense "%h9" "%flat" "%W9" "%b9" flat d1 ++ relu2 "%a9" "%h9" d1 ++
  dense "%ha" "%a9" "%Wa" "%ba" d1 d1 ++ relu2 "%aa" "%ha" d1 ++
  dense "%logits" "%aa" "%Wb" "%bb" d1 nClasses ++
  s!"    return %logits : {ty [B,nClasses]}\n" ++
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
  -- Whole train step rendered from the verified AST (cotangent + weightSgd/biasSgd
  -- nodes), the den-certified renderer LinearFaithfulPoC proves; see that file +
  -- planning/verified_faithful_sweep.md. (linearTrainStepModuleV — forward-AST +
  -- hand-written tail — is its structural predecessor, kept for reference.)
  IO.FS.writeFile "verified_mlir/linear_train_step.mlir"
    (Proofs.StableHLO.linTrainStepFaithfulV 128 784 10 "0.00078125"
       (fun _ _ => 0) (fun _ => 0) (fun _ => 0))
  -- Chapter 3 MLP (784→512→512→10): forward + full SGD train step.
  IO.FS.writeFile "verified_mlir/mlp_fwd.mlir"
    (Proofs.StableHLO.mlpFwdModuleV 128 784 512 512 10
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- mlp_train_step.mlir is now generated by the faithful renderer in MlpRender.lean
  -- (mlpTrainStepFaithfulV, den-certified in MlpFaithfulPoC.lean), not mlpTrainStepText.
  -- Chapter 4 CNN forward (1→32→32 conv, 28×28→14×14 maxpool, 6272→512→512→10).
  IO.FS.writeFile "verified_mlir/cnn_fwd.mlir"
    (Proofs.StableHLO.cnnFwdModuleV 128 1 32 14 14 512 10 3 3
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- cnn_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cnnTrainStepFaithfulV, den-certified in CnnFaithfulPoC.lean), not cnnTrainStepText.
  -- Chapter 5 CIFAR forward (3→32→32 conv, 32×32→16×16 pool, 32→64→64 conv,
  -- 16×16→8×8 pool, flatten 4096→512→512→10). h=w=8 ⇒ input 3·32·32 = 3072.
  IO.FS.writeFile "verified_mlir/cifar_fwd.mlir"
    (Proofs.StableHLO.cifarFwdModuleV 128 3 32 64 8 8 512 10 3 3
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
       (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
       (fun _ => 0))
  -- cifar_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifarTrainStepFaithfulV, den-certified in CifarFaithfulPoC.lean), not cifarTrainStepText.
  -- Chapter 5 CIFAR **per-channel BatchNorm** forward (per-example per-channel BN
  -- after each conv; ε=1e-5; H=W=32 full input spatial). String renderer (peer of
  -- the train-step) until the typed cifarBnFwdGraph is reconciled to per-channel.
  IO.FS.writeFile "verified_mlir/cifar_bn_fwd.mlir"
    (Proofs.StableHLO.cifarBnFwdTextPC 128 3 32 64 32 32 3 3 512 10 "1.0e-05")
  -- cifar_bn_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifarBnTrainStepFaithfulV, den-certified in CifarBnFaithfulPoC.lean), not cifarBnTrainStepText.
  -- Deeper 8-conv CIFAR (no BN): 4 conv→conv→pool stages, channels [16,16,32,32],
  -- 32→16→8→4→2 spatial, flat 32·2·2 = 128 → 64 → 64 → 10. lr = 0.1/128.
  -- cifar8_fwd.mlir is now rendered from the verified cifar8FwdGraph (cifar8FwdGraph_faithful),
  -- not the hand-written cifar8FwdText. Dims: h=w=2 final pooled (image 32×32, 4 pools).
  IO.FS.writeFile "verified_mlir/cifar8_fwd.mlir"
    (Proofs.StableHLO.cifar8FwdModuleV 128 3 16 16 32 32 2 2 64 10 3 3
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) (fun _ _ _ _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0))
  -- cifar8_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifar8TrainStepFaithfulV, den-certified in Cifar8FaithfulPoC.lean), not cifar8TrainStepText.
  -- Deeper 8-conv CIFAR **per-channel BatchNorm** (ε=1e-5; lr = 0.1/128).
  -- cifar8_bn_fwd.mlir is now rendered from the verified cifar8BnFwdGraph (cifar8BnFwdGraph_faithful),
  -- not the hand-written cifar8BnFwdTextPC. Per conv layer: W b ε γ β (ε render-erased; epsStr carries it).
  IO.FS.writeFile "verified_mlir/cifar8_bn_fwd.mlir"
    (Proofs.StableHLO.cifar8BnFwdModuleV 128 3 16 16 32 32 2 2 64 10 3 3 "1.0e-05"
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
      (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
      (fun _ => 0))
  -- cifar8_bn_train_step.mlir is now generated by the faithful renderer in CnnRender.lean
  -- (cifar8BnTrainStepFaithfulV, den-certified by the existing generics), not cifar8BnTrainStepText.
  pure () : IO Unit)

