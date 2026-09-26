/-! # The reference path's specification types

`Layer` (one architecture primitive), `NetSpec` (a named layer list plus input size and per-net
conventions), `TrainConfig` (optimizer, schedule, loss and augmentation settings), `LossKind`,
`OptimizerKind` and `DatasetKind`. `Spec` counts and validates specs; `MlirCodegen` (NetSpec →
StableHLO at run time) and the JAX emitter (jax/Jax/Codegen.lean) lower them; `Train` runs them.
None of this is the verified path, which trains from the committed `verified_mlir/` renders
(`VerifiedSpec`, `VerifiedTrain`). Also here: the `iree-compile` argument builders. -/

/-- Pointwise activation carried by a layer: ReLU, ReLU6, identity, swish (SiLU), hard-swish,
    GELU. -/
inductive Activation where
  | relu
  | relu6
  | identity
  | swish
  | hSwish
  | gelu
deriving Repr, BEq

/-- Normalization choice — picked at the block level by primitives that
    can run with either LayerNorm or BatchNorm (e.g. ConvNeXt, in its
    original LN form or a hypothetical BN variant for ablation). -/
inductive Normalization where
  | bn
  | ln
deriving Repr, BEq

/-- Convolution padding intent: `same` keeps the spatial size (at stride 1), `valid` pads
    nothing. Read by the JAX emitter; see `PadStyle` for how `same` is spelled when strided. -/
inductive Padding where
  | same
  | valid
deriving Repr, BEq

/-- How a layer's `.same` padding is **spelled** when it is STRIDED.

    `.same` states an intent ("keep the output size"), and at stride 1 with an odd kernel the two
    spellings below are bit-identical — which is every conv in the kit except the strided ones.
    They differ only when strided on an even input, where the two grids sit one input position
    apart:

    * `.xlaSame`   — XLA `'SAME'`, extra row on the HIGH side. 7×7/s2 on 224 pads (2,3);
      3×3/s2 on an even input pads (0,1). This is the reference for the TF-origin ports
      (MobileNetV2/V4, EfficientNet), where asymmetric `'SAME'` is the published net.
    * `.symmetric` — torchvision / He et al., `(k-1)//2` on both sides, i.e.
      `nn.Conv2d(padding=k//2)`. This is the reference for the ResNet family.

    `NetSpec.convPadStyle` picks one per net. Neither shape, op count nor arity distinguishes
    the two, so a net whose reference and render disagree here shows no structural difference.
    In the JAX emitter `.symmetric` passes no padding argument, so the Python helpers
    `conv2d`/`conv_bn`'s own symmetric default applies and the convention is stated once. -/
inductive PadStyle where
  | xlaSame
  | symmetric
deriving Repr, BEq

/-- One architecture primitive of a `NetSpec`. A constructor is either trained by the reference
    codegen (`Layer.paramSlots` gives its parameter tensors) or shape-only (`paramSlots` is
    `none`; `Layer.nParamsUntrained` gives its displayed parameter count, and neither emitter
    lowers it unless its docstring says otherwise). -/
inductive Layer where
  /-- `ic → oc` convolution, `kSize × kSize`, stride 1, with bias, followed by `act`. The JAX
      emitter reads `pad`; `MlirCodegen` pads to keep the spatial size whatever `pad` says. -/
  | conv2d  (ic oc kSize : Nat) (pad : Padding) (act : Activation)
  /-- `ic → oc` convolution without bias, `kSize × kSize` at `stride`, then BatchNorm and an
      activation: the net's `NetSpec.convBnAct` in the JAX emitter, ReLU in `MlirCodegen`. `pad`
      as for `conv2d`. -/
  | convBn  (ic oc kSize stride : Nat) (pad : Padding)
  /-- `size × size` max-pool at `stride` (SAME-padded in `MlirCodegen`). -/
  | maxPool (size stride : Nat)
  /-- Global average pool over the spatial axes, `[B, C, H, W] → [B, C]`. -/
  | globalAvgPool
  /-- Flatten every non-batch axis. -/
  | flatten
  /-- Bare LayerNorm over a `[B, dim]` feature vector: normalize over `dim`, then the
      per-feature affine `γ ⊙ x̂ + β` (2·dim params). Not the channels-first LN inside
      `convNextStage`/`convNextDownsample` — this is the rank-2 head-side one, which sits
      between two layers and so cannot be a `norm :=` field on either of them; ConvNeXt's
      `GAP → LN → Linear` head uses it. Lowered by the JAX emitter; `MlirCodegen` has no
      lowering for it. -/
  | layerNorm (dim : Nat)
  /-- Fully-connected `fanIn → fanOut` with bias, followed by `act`. -/
  | dense   (fanIn fanOut : Nat) (act : Activation)
  /-- `nBlocks` ResNet basic blocks `ic → oc` (two 3×3 conv-BN each); the first runs at
      `firstStride` and carries a 1×1 conv-BN projection when `ic ≠ oc` or the stride is not 1. -/
  | residualBlock (ic oc nBlocks firstStride : Nat)
  /-- `nBlocks` ResNet bottleneck blocks `ic → oc` (1×1, 3×3, 1×1 conv-BN at width `oc / 4`);
      the first runs at `firstStride` and carries a 1×1 conv-BN projection when `ic ≠ oc` or the
      stride is not 1. -/
  | bottleneckBlock (ic oc nBlocks firstStride : Nat)
  /-- Depthwise-separable convolution: 3×3 depthwise then 1×1 pointwise `ic → oc`, each with BN.
      Lowered by the JAX emitter only; `MlirCodegen` has no case for it. -/
  | separableConv (ic oc stride : Nat)
  /-- `nBlocks` MobileNetV2 inverted residuals `ic → oc`: 1×1 expand conv-BN to `expand ×` the
      block input (omitted when `expand = 1`), 3×3 depthwise conv-BN, 1×1 project conv-BN; the
      first block runs at `stride`. -/
  | invertedResidual (ic oc expand stride nBlocks : Nat)
  /-- `nBlocks` EfficientNet MBConv blocks `ic → oc`: 1×1 expand (omitted when `expand = 1`),
      `kSize × kSize` depthwise, optional squeeze-excitation (squeeze width `mbConvSeMid` of the
      block input), 1×1 project, each conv with BN; activation `act`, the first block at
      `stride`. -/
  | mbConv (ic oc expand kSize stride nBlocks : Nat) (useSE : Bool) (act : Activation := .swish)
  /-- One MobileNetV3 block `ic → oc` with an explicit expanded width `expandCh` (the expand
      conv is omitted when `expandCh = ic`), `kSize × kSize` depthwise, optional SE, 1×1 project;
      activation `act`. -/
  | mbConvV3 (ic oc expandCh kSize stride : Nat) (useSE : Bool) (act : Activation := .relu)
  /-- `nBlocks` Fused-MBConv blocks `ic → oc`: a full `kSize × kSize` expand conv-BN (straight
      to `oc` when `expand = 1`), optional SE, and a 1×1 project conv-BN when `expand ≠ 1`. -/
  | fusedMbConv (ic oc expand kSize stride nBlocks : Nat) (useSE : Bool) (act : Activation := .swish)
  /-- MobileNetV4 Universal Inverted Bottleneck `ic → oc`: optional `preDWk × preDWk` depthwise,
      1×1 expand to `ic · expand`, optional `postDWk × postDWk` depthwise, 1×1 project, each conv
      with BN. `k = 0` omits that depthwise. -/
  | uib (ic oc expand stride : Nat) (preDWk postDWk : Nat)
  /-- SqueezeNet fire module: 1×1 squeeze `ic → squeeze`, then 1×1 and 3×3 expands concatenated.
      Lowered by the JAX emitter only; `MlirCodegen` has no case for it. -/
  | fireModule (ic squeeze expand1x1 expand3x3 : Nat)
  /-- ViT patch embedding: a `patchSize × patchSize` stride-`patchSize` conv `ic → dim` with
      bias, a CLS token, and a learned `[nPatches + 1, dim]` position embedding. -/
  | patchEmbed (ic dim patchSize nPatches : Nat)
  /-- `nBlocks` pre-LN transformer encoder blocks (LN, multi-head attention with Q/K/V/O
      denses, LN, `dim → mlpDim → dim` MLP), plus a final LN.

      * `causalMask` adds a triangular −∞ mask to the `QKᵀ` scores before softmax (TinyGPT sets
        it; vision transformers leave it false).
      * `keepSequence` skips the ViT-style CLS-token slice at the end, keeping the `[B, N, D]`
        token shape (DDPM's bottleneck attention sets it so the tokens can flow into a
        `spatialUnflatten`).
      * `flashAttn` emits the attention as a tiled online-softmax `stablehlo.while` instead of
        the dense `[B,H,T,T]` scores — `O(T·blk)` rather than `O(T²)` memory, same math. Used by
        the train step's forward and backward; the eval forward stays dense.
      * `rope` applies Rotary Position Embedding to Q/K inside attention (forward rotation and
        its VJP), a relative position encoding that extends past the trained length; pair it
        with `tokenPositionEmbed`'s `posEmb := false`. -/
  | transformerEncoder (dim heads mlpDim nBlocks : Nat)
                       (causalMask : Bool := false)
                       (keepSequence : Bool := false)
                       (flashAttn : Bool := false)
                       (rope : Bool := false)
  /-- Selective state-space block (Mamba / S6); `dim` hidden, `stateSize` = N, `expand` the
      inner-dim multiplier. Shape-only (its count in `Layer.nParamsUntrained` is approximate). -/
  | mambaBlock (dim stateSize expand nBlocks : Nat)
  /-- Swin Transformer stage: `nBlocks` blocks of (W-MSA + MLP) with residuals, alternating with
      the shifted-window variant; window-local attention at fixed resolution. Shape-only. -/
  | swinStage (dim heads mlpDim windowSize nBlocks : Nat)
  /-- Swin patch merging: 2×2 spatial downsample, channels `inDim → outDim` (stated explicitly
      to match any hierarchy). Shape-only. -/
  | patchMerging (inDim outDim : Nat)
  /-- UNet encoder stage: two 3×3 conv-BN-ReLU (`ic → oc → oc`), then a 2×2 max-pool. The
      pre-pool feature map is saved as the skip for the matching `unetUp`. -/
  | unetDown (ic oc : Nat)
  /-- UNet decoder stage: ×2 bilinear upsample, concatenation with the matching encoder skip
      (`oc` channels), then two 3×3 conv-BN-ReLU (`ic + oc → oc → oc`). `ic` is the channel
      count coming in from the previous decoder stage or bottleneck. -/
  | unetUp (ic oc : Nat)
  /-- Parameter-free bilinear upsample by the integer factor `scale`: channels unchanged,
      `(H, W) → (scale·H, scale·W)`. Lowered by `MlirCodegen` (forward and VJP). -/
  | bilinearUpsample (scale : Nat)
  /-- DETR-style transformer decoder: `nBlocks` blocks of self-attention over `nQueries` learned
      object queries, cross-attention with the encoder output, and an FFN; the object-query
      embedding is part of this layer's parameters. Shape-only. -/
  | transformerDecoder (dim heads mlpDim nBlocks nQueries : Nat)
  /-- DETR prediction heads on each of the decoder's output tokens: a class head
      (`dim → nClasses + 1`, the extra slot "no object") and a 3-layer box MLP
      (`dim → dim → dim → 4`, predicting `(cx, cy, w, h)`). Shape-only. -/
  | detrHeads (dim nClasses : Nat)
  /-- ShuffleNet v1 stage: `nUnits` units `ic → oc`, the first a stride-2 downsampling unit
      (avg-pool skip), the rest residual; each unit is 1×1 grouped conv → channel shuffle →
      3×3 depthwise → 1×1 grouped conv, with `groups` channel partitions. Shape-only. -/
  | shuffleBlock (ic oc groups nUnits : Nat)
  /-- ShuffleNet v2 stage (Ma et al. 2018): `nUnits` units `ic → oc`, the first a stride-2
      downsampling unit (both branches see the full input and are concatenated), the rest
      stride-1 basic units (channel split, identity on one half, 1×1 → 3×3 DW → 1×1 on the
      other, concatenate, shuffle). No grouped 1×1 convs and no element-wise adds.
      Shape-only. -/
  | shuffleV2Block (ic oc nUnits : Nat)
  /-- AlphaFold-2 Evoformer stack over the dual MSA (`s × r × msaChannels`) and pair
      (`r × r × pairChannels`) representations: MSA row attention with pair bias, column
      attention, transition, outer-product mean into the pair, triangle multiplicative updates,
      triangle attention and pair transition, bundled because the dual representation does not
      fit a linear `NetSpec`. Shape-only. -/
  | evoformerBlock (msaChannels pairChannels nBlocks : Nat)
  /-- AlphaFold-2 Structure Module: `nBlocks` rounds of Invariant Point Attention over residues,
      backbone frame updates and side-chain χ-angle prediction, with weights shared across the
      rounds. Shape-only. -/
  | structureModule (singleChannels pairChannels nBlocks : Nat)
  /-- MobileViT block (Mehta & Rastegari 2022), `ic → ic`: local 3×3 conv and 1×1 projection to
      the transformer width `dim`, unfold into patches, `nTxBlocks` transformer blocks (`heads`
      heads, FFN width `mlpDim`), fold back, 1×1 projection to `ic`, concatenation with the input
      and a 3×3 fusion conv. Shape-only. -/
  | mobileVitBlock (ic dim heads mlpDim nTxBlocks : Nat)
  /-- ConvNeXt stage (Liu et al. 2022): `nBlocks` residual blocks at fixed `channels`, each a
      7×7 depthwise conv, LN, 1×1 expand (×4), GELU, 1×1 project and layer scale. No
      downsampling (see `convNextDownsample`). `norm` selects LayerNorm (the paper) or BatchNorm,
      `act` GELU (the paper) or another activation. -/
  | convNextStage (channels nBlocks : Nat)
                  (norm : Normalization := .ln) (act : Activation := .gelu)
  /-- ConvNeXt inter-stage downsample: LayerNorm then a 2×2 stride-2 conv `ic → oc`. `norm` as
      in `convNextStage`, so a stage and its downsample stay in sync. -/
  | convNextDownsample (ic oc : Nat) (norm : Normalization := .ln)
  /-- ConvNeXt patchify stem: a `patch × patch` stride-`patch` conv `ic → oc` with bias, then a
      channels-first LayerNorm and no activation. Lowered by the JAX emitter; `MlirCodegen` has
      no lowering for it. -/
  | convNextStem (ic oc patch : Nat)
  /-- WaveNet (van den Oord 2016) residual stack: `nLayers` blocks at dilations
      `2⁰, …, 2^(nLayers−1)`, each a dilated causal conv, gated activation (tanh ⊙ sigmoid),
      1×1 projection back to `residualCh`, residual skip, and a separate 1×1 conv producing a
      `skipCh` output summed into the head. Shape-only. -/
  | waveNetBlock (residualCh skipCh nLayers : Nat)
  /-- Sinusoidal positional encoding (Vaswani 2017 / NeRF 2020),
      `γ(p) = (sin(2⁰πp), cos(2⁰πp), …, sin(2^(L−1)πp), cos(2^(L−1)πp))`. Parameter-free; output
      dim `inputDim · 2 · numFrequencies`. Shape-only. -/
  | positionalEncoding (inputDim numFrequencies : Nat)
  /-- NeRF MLP core (Mildenhall et al. 2020): 8 hidden layers of `hiddenDim` with ReLU, a skip
      re-concatenating the position encoding at layer 5, and two heads (density σ, and RGB from
      a direction-conditioned branch). `encodedPosDim` is the dim of γ(x) (typically 60),
      `encodedDirDim` of γ(d) (typically 24). Shape-only. -/
  | nerfMLP (encodedPosDim encodedDirDim hiddenDim : Nat)
  /-- Darknet residual stack (YOLOv3): `nBlocks` blocks at fixed `channels`, each 1×1 conv
      (`c → c/2`), 3×3 conv (`c/2 → c`) and residual add. No downsample. Shape-only. -/
  | darknetBlock (channels nBlocks : Nat)
  /-- Cross-Stage Partial block (Wang et al. 2019; the YOLOv4–v11 family): split the channels,
      run one half through `nBlocks` bottleneck residual blocks, concatenate with the other half
      and 1×1-project to `oc`. One primitive for the C3 / C2f / C3k2 variants. Shape-only. -/
  | cspBlock (ic oc nBlocks : Nat)
  /-- Inception module (GoogLeNet): four branches concatenated on channels — 1×1 (`b1out`);
      1×1 → 3×3 (`b2reduce`, `b2out`); 1×1 → 5×5 (`b3reduce`, `b3out`); 3×3 max-pool → 1×1
      (`b4out`). Output channels `b1out + b2out + b3out + b4out`. Shape-only. -/
  | inceptionModule (ic b1out b2reduce b2out b3reduce b3out b4out : Nat)
  /-- Atrous Spatial Pyramid Pooling (DeepLab v3+): five branches of `oc` channels — 1×1 conv,
      3×3 atrous convs at rates 6, 12 and 18, and global-avg-pool → 1×1 conv → upsample — each
      with BN and ReLU, concatenated and fused by a 1×1 conv back to `oc`. Shape-only. -/
  | asppModule (ic oc : Nat)
  /-- Feature Pyramid Network (Lin et al. 2017): 1×1 lateral convs from the four backbone stages
      (`c2`–`c5`) to `target` channels, top-down ×2 upsample-and-add, and a 3×3 smoothing conv
      per level; four `target`-channel outputs. Shape-only (see `fpnDetect` for the trained
      detector head). -/
  | fpnModule (c2 c3 c4 c5 target : Nat)
  /-- FPN multi-scale detector head. Taps the three preceding backbone stage outputs C3/C4/C5
      (channels `c3`/`c4`/`c5`, strides 8/16/32), runs a top-down neck (all → `oc` channels), an
      optional RetinaNet tower of `tower` 3×3 convs per level (bias, ReLU, channel-preserving, not
      shared across levels), then a per-scale 1×1 conv head (`oc → A·15`), and flattens and
      concatenates the three heads into one `[B, Ntot]` output
      (`Ntot = A·15·(g3² + g4² + g5²)`, `g5` the coarsest grid, `imageH / 32`). Parameter order
      is `fpnDetectParamShapes`, head biases last. `tower = 0` emits no tower ops. The loss is the
      FPN multi-scale YOLO loss, routed by `TrainConfig.fpnScales`. -/
  | fpnDetect (oc c3 c4 c5 g5 A tower : Nat)
  /-- DenseNet dense block: `nLayers` BN-ReLU-1×1 (to `4·growthRate`)-BN-ReLU-3×3 sub-layers,
      each adding `growthRate` channels to the running concatenation; output
      `ic + nLayers · growthRate` channels. Shape-only. -/
  | denseBlock (ic growthRate nLayers : Nat)
  /-- DenseNet transition: BN, 1×1 conv `ic → oc`, 2×2 stride-2 average pool. Shape-only. -/
  | transitionLayer (ic oc : Nat)
  /-- `[B, C, H, W] → [B, H·W, C]` (channel axis last), no parameters. Pair with
      `spatialUnflatten` around a token-stream op (typically `transformerEncoder`) in an NCHW
      pipeline. -/
  | spatialFlatten
  /-- `[B, H·W, C] → [B, C, H, W]`, no parameters. Carries `(C, H, W)` because the rank-3 token
      shape does not determine `(H, W)`. -/
  | spatialUnflatten (channels height width : Nat)
  /-- Token plus learned position embedding for autoregressive language models (TinyGPT).
      Input: flat `[B, seqLen · vocabSize]` one-hot; output `[B, seqLen, dModel]`. Parameters:
      the token embedding `[vocabSize, dModel]` and, when `posEmb`, the position embedding
      `[seqLen, dModel]`.

      * `idsInput` takes `[B, seqLen]` f32 token ids and builds the one-hot in-graph (iota,
        compare, select): same parameters and math, no `O(V·T)` host upload.
      * `gather` makes the forward a `stablehlo.gather` of the `[V, D]` table by the ids and the
        backward a `stablehlo.scatter`-add, so no `[B, T, V]` one-hot is materialized (`O(T·D)`
        rather than `O(T·V)` memory). Implies ids-shaped input.
      * `posEmb := false` drops the position table, for RoPE models: every weight is then
        independent of `seqLen`, so a model trained at one length runs at any length. -/
  | tokenPositionEmbed (vocabSize seqLen dModel : Nat) (idsInput : Bool := false)
      (gather : Bool := false) (posEmb : Bool := true)
  /-- Language-model head: per-position logits. Input `[B, seqLen, dModel]`, output flat
      `[B, seqLen · vocabSize]` so the per-pixel CE loss handles it by reshape. Parameters: dense
      `[dModel, vocabSize]` and bias. -/
  | lmHead (dModel vocabSize seqLen : Nat)
  /-- Diffusion time conditioning added onto a `[B, channels, H, W]` feature map: a learned
      dense projection of a `2·nFreq` sin/cos embedding of the per-image timestep, broadcast over
      `H, W`. The timestep is read in-graph from the last input channel (`Ddpm.prependTChannel`'s
      `t/Tmax` plane), so there is no extra input. Parameters `W [2·nFreq, channels]` and
      `b [channels]` start at zero, so conditioning starts as a no-op. -/
  | timeCondAdd (channels nFreq : Nat)
deriving Repr

/-- A reference-path architecture: a name, the layer list, the input size, and the per-net
    conventions (`NetSpec.convBnAct`, `NetSpec.convPadStyle`) the emitters read. `Spec` counts its parameters
    (`NetSpec.totalParams`) and validates it; `NetSpec.train` compiles and trains it. -/
structure NetSpec where
  name   : String
  layers : List Layer
  imageH : Nat := 28
  imageW : Nat := 28
  /-- Total downsampling stride from input to the detection feature map, for
      YOLO-style specs: the grid is `imageH / detStride`. 32 for the standard
      ResNet stride-32 backbone (grid = 224/32 = 7, 448/32 = 14); set to 16 for a
      stride-16 tap (last block stride 1) to double the grid (448/16 = 28). Only
      consulted on the detection path; irrelevant to classifiers/seg. -/
  detStride : Nat := 32
  /-- Optional suffix on `buildPrefix`, so two runs of the **same
      architecture** under different training configs get their own params,
      MLIR, and vmfb instead of overwriting each other.

      `buildPrefix` is derived from `name` alone, which is right — the name
      describes the architecture, and the loss is not part of the architecture.
      But it means a loss ablation is a set of runs that all claim the same
      `.lake/build/<net>_params.bin` and the same `<net>_train_step.vmfb`. Run
      them sequentially and each silently clobbers the last; run them concurrently,
      one per GPU, and they race on the vmfb mid-compile.

      Set this to the arm's name and the ablation parallelizes cleanly. Empty
      (the default) leaves `buildPrefix` as `name` alone. -/
  buildTag : String := ""
  /-- Activation applied after a `.convBn` layer by the JAX emitter, per net (default ReLU).
      MobileNetV2 is ReLU6 throughout (stem and head included) and EfficientNet-B0 SiLU/swish
      throughout, so their references set this to match the verified renders.

      A ReLU/ReLU6 mismatch is invisible on small activations (the two agree below 6), so a
      forward tie at small weight scale passes with it present; it shows only under He-scaled
      weights. A passing tie is not evidence that the activations agree.

      Scoped per net, not per layer, because every net in the kit uses one activation at all of
      its `.convBn` sites; a net that mixed them would need a `Layer` field instead. -/
  convBnAct : Activation := .relu
  /-- How `.same` is spelled at a strided top-level `.conv2d`/`.convBn`, per net, in the JAX
      emitter. See `PadStyle`. Defaults to `.xlaSame`. The ResNet-family mains set `.symmetric`
      because torchvision's `Conv2d(3,64,7,stride=2,padding=3)` is the net their verified render
      implements.

      It changes only a strided top-level conv, which in the kit is the stem. Block-internal
      convs (`basic_block_down`, `bottleneck_block_down`, …) call the helpers without a padding
      argument and so take the symmetric default; the mobile/TF-origin blocks pass `'SAME'`
      explicitly inside their own helpers. Both are untouched by this field.

      Scoped per net rather than per layer for the same reason as `convBnAct`: no net in the kit
      mixes the two conventions across its own top-level convs. -/
  convPadStyle : PadStyle := .xlaSame
deriving Repr

/-- The "kind" of supervised loss the train step computes. Picks the
    label-tensor shape + the forward/backward formula. Modifiers like
    `useFocal`, `labelSmoothing`, and the aug flags (`useMixup` etc.)
    layer on top — `LossKind` only captures the primary loss shape.

    Used by `compileVmfbs` for a single-match mutex check and to drive
    the codegen flag set. Defaults to `.classCE` for back-compat with
    every existing trainer; if left at the default, `compileVmfbs`
    derives the effective kind from the older booleans
    (`useYolov1`, `useSeg`, `useMixup`/`useCutmix`/`useKnnMixup`) so
    callers don't have to update. -/
inductive LossKind where
  /-- Default: int32 `[B]` class label, softmax cross-entropy. Compatible
      with `useFocal` (focal modifier) and `labelSmoothing`. -/
  | classCE
  /-- Float `[B, NC]` soft labels (mixup/cutmix/knn-mixup output).
      Compatible with `labelSmoothing` (already baked in by the caller). -/
  | softLabelCE
  /-- Int32 `[B, H, W]` per-pixel label tensor (segmentation), softmax cross-entropy per
      pixel. -/
  | perPixelCE
  /-- Soft Dice over the softmax probabilities, int32 `[B, H, W]` labels —
      same ABI as `perPixelCE`, different loss block. Dice is computed
      per-class over the whole batch and meaned:
      `1 - mean_c (2·Σ p_c·y_c + ε) / (Σ p_c + Σ y_c + ε)`.

      The point of it: per-pixel CE is a *mean over pixels*, so a class
      occupying 0.5% of pixels contributes 0.5% of the loss and the cheapest
      descent direction is to predict it away. Dice is a *ratio per class*,
      so every class carries equal weight no matter how few pixels it owns.

      Batch-Dice (reducing over B as well as H,W) rather than per-sample:
      with rare classes and small batches a sample may contain none of a
      class at all, which makes per-sample Dice for it degenerate. -/
  | perPixelDice
  /-- `perPixelDice + perPixelCE`, summed (loss and gradient both). The
      standard medical-segmentation default: Dice supplies the class-balanced
      signal, CE regularizes it (pure Dice has a noisy gradient early, when
      the softmax is near-uniform and every denominator is large). -/
  | perPixelDiceCE
  /-- Class-weighted per-pixel softmax CE: `weights[c]` scales the loss and the
      gradient of every pixel whose *true* class is `c`. Same ABI as
      `perPixelCE`; `weights.length` must equal the class count.

      This is the lever `perPixelDice` was supposed to be and isn't. Dice's
      gradient carries a factor of `p_i` from the softmax Jacobian, so it
      vanishes exactly where a collapsed class needs rescuing — measured on
      BraTS at 0.02% of CE's gradient once p₃ ≈ 2e-5
      (`scripts/probes/seg_dice_vanishing_grad_probe.py`). CE's seed is `(p - y)/N`,
      which is `-1/N` at `p = 0`: **flat, and wholly indifferent to the
      collapse.** Scaling that by `w_c` therefore keeps a live signal all the
      way down, which Dice cannot do.

      Reduction is the weighted mean `Σ_k w_{y_k}·CE_k / Σ_k w_{y_k}` (torch's
      `CrossEntropyLoss(weight=…, reduction='mean')` semantics), not `/N`. Two
      reasons, both practical: the loss stays on the same scale as unweighted
      CE so the arms of an ablation are readable against each other, and — since
      both sums are linear in `w` — the loss is **invariant to the overall
      scale of `weights`**. Only the ratios matter, so a caller cannot
      accidentally change the effective learning rate by normalizing its weight
      vector differently. `Σ_k w_{y_k}` depends only on the labels, so it is a
      constant w.r.t. the logits and contributes no gradient term. -/
  | perPixelWeightedCE (weights : List Float)
  /-- Per-pixel focal CE (Lin et al., RetinaNet): `-(1-p_t)^γ · log p_t`, meaned
      over pixels. Same ABI as `perPixelCE`. `γ = 0` is exactly `perPixelCE`.

      The third distinct answer to the imbalance, and mechanically the opposite
      of `perPixelWeightedCE` — worth stating, because "focal and class weights
      both reweight the loss" hides the whole point:

      * **weighted CE amplifies the rare class.** A static, per-*class* factor
        from the label frequencies.
      * **focal suppresses the easy class.** A dynamic, per-*pixel* factor from
        the current prediction. At `p_t → 1` (confident background — 97% of
        BraTS) the `(1-p_t)^γ` factor crushes the gradient toward 0. At
        `p_t → 0` it tends to 1 and the gradient tends to CE's: focal does
        **not** amplify the collapsed class, it defunds the majority drowning
        it out.

      So focal needs no frequency statistics and cannot be mis-tuned by a bad
      weight vector, but it also cannot help a class that is rare *and* easy.

      α is deliberately omitted. The paper's α_t is a per-class weight, i.e.
      exactly `perPixelWeightedCE`'s mechanism — folding it in here would
      confound the two arms of the very ablation this exists for. Compose them
      later, once each is understood alone.

      NB the gradient here is the **true** one, including the derivative of the
      `(1-p_t)^γ` factor. Contrast the YOLOv1 objectness path, which detaches
      that weight (`MlirCodegen.lean`, `%y1f_w0`) — a defensible approximation,
      but one whose `d_logits` is not the derivative of any loss it states, and
      so could not be FD-verified the way this is. -/
  | perPixelFocalCE (gamma : Float)
  /-- Float `[B, C, H, W]` target tensor with per-pixel MSE (DDPM,
      autoencoder regression). Caller passes `ddpmOutShape`. -/
  | floatTargetMse
  /-- YOLOv1: float `[B, perCell, gridH, gridW]` target + float
      `[B, gridH, gridW]` per-cell mask. 5-term masked MSE with √ ε-floor
      on the box-dim terms. -/
  | yolov1Masked
  /-- Binary cross-entropy with logits over multi-hot `[B, NC]` targets —
      timm "ResNet Strikes Back" RSB-A2's loss. Each class is an independent
      sigmoid; the mixup/cutmix soft-label path produces the `[B,NC]` target
      directly (hard labels are one-hot'd, with optional label smoothing).
      JAX-only: `compileVmfbs` rejects it on the MLIR path. Reduction is
      timm's `mean` over B×C. -/
  | bce
deriving Repr, BEq

/-- Which loss block the segmentation path emits. All three share one ABI
    (int32 `[B,H,W]` labels, the `trainStepAdamF32Seg` dispatch, the mIoU
    eval harness) — only the emitted loss + gradient differ, so this rides
    alongside `useSeg` rather than replacing it. -/
inductive SegLoss where
  /-- Per-pixel softmax cross-entropy (the original UNet-demo path). -/
  | ce
  /-- Soft Dice only. -/
  | dice
  /-- Dice + CE, summed. -/
  | diceCE
  /-- Per-pixel CE with a per-class weight on the true class. See
      `LossKind.perPixelWeightedCE` for the semantics and the argument. -/
  | weightedCE (weights : List Float)
  /-- Per-pixel focal CE, `-(1-p_t)^γ·log p_t`. See `LossKind.perPixelFocalCE`. -/
  | focalCE (gamma : Float)
deriving Repr, BEq, Inhabited

/-- Does this loss run on the segmentation path? Every per-pixel kind shares
    the int32 `[B,H,W]` label ABI and the seg train-step dispatch.

    Single source of truth: `compileVmfbs`, `NetSpec.train`, and `runTraining`
    each need this answer, and deriving it three times independently is how
    they drift apart. -/
def LossKind.isSeg : LossKind → Bool
  | .perPixelCE | .perPixelDice | .perPixelDiceCE
  | .perPixelWeightedCE _ | .perPixelFocalCE _ => true
  | _ => false

/-- Which seg loss block to emit. Non-seg kinds answer `.ce` and are never
    asked (the codegen only consults this under `useSeg`). -/
def LossKind.segLoss : LossKind → SegLoss
  | .perPixelDice          => .dice
  | .perPixelDiceCE        => .diceCE
  | .perPixelWeightedCE w  => .weightedCE w
  | .perPixelFocalCE g     => .focalCE g
  | _                      => .ce

/-- Optimizer selector for the training loop. Added additively over the legacy
    `TrainConfig.useAdam` bool (à la `LossKind` over the older loss booleans):
    the JAX backend derives the effective optimizer from `useAdam` when this is
    left at the `.sgd` default, so no existing config needs to change. -/
inductive OptimizerKind where
  /-- Plain SGD, or SGD + heavy-ball momentum when `TrainConfig.momentum > 0`. -/
  | sgd
  /-- Adam / AdamW (decoupled weight decay when `weightDecay > 0`). -/
  | adam
  /-- RMSprop with momentum — the native MobileNetV2 / EfficientNet optimizer.
      `v = ρ·v + (1-ρ)·g²;  buf = μ·buf + g/(√v+ε);  p -= lr·buf`, with ρ =
      `rmspropDecay`, μ = `momentum`, ε = `rmspropEps`. Weight decay stays
      coupled into the gradient (the form those papers use), unlike AdamW. -/
  | rmsprop
  /-- LAMB (You et al. 2019) — the large-batch optimizer in timm's "ResNet
      Strikes Back" RSB-A2 recipe. Adam moments `(m, v, t)` form the per-param
      direction `r = m̂/(√v̂+ε) + λ·θ` (DECOUPLED weight decay `λ = weightDecay`
      folded into the direction), then a layer-wise **trust ratio**
      `‖θ‖ / ‖r‖` rescales the step: `θ -= lr · (‖θ‖/‖r‖) · r`. The trust ratio
      is 1.0 wherever `‖θ‖` or `‖r‖` is 0 (timm convention). β1=0.9, β2=0.999,
      ε=1e-6. opt_state shape matches `.adam`: `(m, v, t)`. -/
  | lamb
deriving Repr, BEq, DecidableEq

/-- One reference-path training recipe: learning rate, batch size and epochs, the optimizer
    (`useAdam` / `optimizer`), the schedule, the loss (`lossKind` and the older booleans it
    subsumes), augmentation, weight averaging, precision and the detector knobs. Read by
    `NetSpec.train` / `NetSpec.runTraining` and by the JAX emitter; each field's docstring says
    which side reads it where only one does. Defaults leave a knob off. -/
structure TrainConfig where
  learningRate : Float
  batchSize    : Nat
  epochs       : Nat
  seed         : Nat := 314159
  momentum     : Float := 0.0
  useAdam      : Bool := false
  /-- Optimizer selector (additive over `useAdam`). Left at the `.sgd` default,
      the JAX backend derives the effective optimizer from `useAdam` (true →
      Adam) for back-compat; set explicitly to `.rmsprop` (or `.adam`) to
      override. The IREE/MLIR backend still reads `useAdam`. -/
  optimizer    : OptimizerKind := .sgd
  /-- RMSprop running-mean-square decay ρ (only used when `optimizer = .rmsprop`). -/
  rmspropDecay : Float := 0.9
  /-- RMSprop denominator ε — NOT 1e-8: MobileNetV2 uses 1.0, EfficientNet 1e-3;
      the large value is part of those recipes. -/
  rmspropEps   : Float := 1e-3
  weightDecay  : Float := 0.0
  cosineDecay  : Bool := false
  warmupEpochs : Nat := 0
  augment      : Bool := false
  labelSmoothing : Float := 0.0
  /-- Focal loss (Lin et al. 2017): replace CE loss with
      `-(1-p_y)^γ · log(p_y)`. Down-weights well-classified examples,
      up-weights hard ones. Restricted to the int-label path (no soft
      labels) and labelSmoothing must be 0 — focal mixes poorly with
      both. γ=2.0 is the paper default. -/
  useFocal     : Bool  := false
  focalGamma   : Float := 2.0
  /-- YOLOv1 box loss: `false` = the published √-MSE coord terms; `true` = a
      DIoU box loss on box0 with a positive box parameterization (cx=(j+σ(tx))/gW,
      w=exp(tw)). Only consulted on the `.yolov1Masked` path. NB: a DIoU-trained model must be
      decoded with the same σ/exp (scripts/demos/yolo_map_visdrone.py --box-param diou). -/
  useDiouBox   : Bool  := false
  /-- YOLO anchor priors (w_rel, h_rel) for the anchor-based detector.
      Empty = single-box YOLOv1. When non-empty, the yolo loss routes to the
      A-anchor path (perCell = A·15, box_a = anchor_a·exp(pred)); the target/head
      must use the matching `A·15`-channel layout (preprocess_visdrone --anchors). -/
  anchors      : List (Float × Float) := []
  /-- FPN multi-scale detector. Per-scale
      `(grid, anchors)` for P3/P4/P5 (e.g. `[(56, a3), (28, a4), (14, a5)]`).
      Empty = not an FPN detector. When non-empty AND the spec ends in a
      `.fpnDetect` layer, the loss routes to `emitMultiScaleYoloLoss` over the
      `[B, Ntot]` head concat, and the single target input is a flat
      `[B, Σ A·15·g_s²]` block (sliced per scale in the loss). -/
  fpnScales    : List (Nat × List (Float × Float)) := []
  /-- Box-aware affine augmentation for the FPN path: per-image scale gain
      `1 + U(-1,1)·fpnAffineScale` and translate `U(-1,1)·fpnAffineTranslate`
      (as a fraction of the frame), fired with probability `fpnAffineProb`.
      `0` scale gain and `0` probability (the defaults) leave the pipeline
      byte-identical to the HSV+hflip pack, so the existing arms stay reproducible.

      Separate from `augment` on purpose. `augment` is the committed and measured
      pack (HSV + hflip, worth 0.1243 → 0.1674 at 50 epochs); this is the arm that
      changes object SCALE, which on VisDrone is the axis the difficulty actually
      lives on — and therefore the one that can hurt as easily as help, since a
      2–5 px object scaled down goes under P3's stride-8 resolution entirely.
      It must be A/B'd under its own tag, on top of `augment`, not instead of it. -/
  fpnAffineScale     : Float := 0.0
  fpnAffineTranslate : Float := 0.0
  fpnAffineProb      : Float := 0.0
  /-- Drop a transformed box whose clipped side falls under `fpnAffineWhThrPx`
      pixels, or which keeps less than `fpnAffineAreaThr` of its area inside the
      frame. Ultralytics uses 2 px and 0.1; the 2 px default is wrong here,
      because a large share of VisDrone GT is 2–5 px to begin with and that
      threshold would silently delete the classes the detector is worst at. -/
  fpnAffineWhThrPx   : Float := 1.0
  fpnAffineAreaThr   : Float := 0.1
  /-- Per-class weights for the detector's classification term. `weights.length` must equal
      the detector class count
      (10 for VisDrone). Empty (the default) is the unweighted path and emits
      byte-identical MLIR.

      Motivated by measurement, not folklore: on the unweighted e12 checkpoint
      the class argmax collapsed onto the two most frequent classes (car 44% +
      pedestrian 21% of encoded positives), leaving 5/10 classes never predicted
      and per-class mAP pinned at ~0.0001 — see `scripts/probes/fpn_obj_separation.py`
      and `scripts/probes/fpn_class_freq.py`. Weights depend only on the target, so
      they are exactly constant w.r.t. the logits and the weighted gradient
      stays finite-difference checkable.

      Normalize so `Σ_c f_c·w_c = 1` (expected weight 1 under the GT class
      distribution) to keep this a pure redistribution — otherwise it silently
      rescales the class term against the box and objectness terms too. -/
  yoloClsWeights : List Float := []
  /-- Focal γ on the CLASS term. `0` = plain (weighted) softmax-CE and
      emits byte-identical MLIR. 2.0 is the RetinaNet value.

      FL = −w·(1−p_t)^γ·log p_t on assigned cells, p_t the softmax probability of
      the true class. Distinct from `focalGamma`, which is the OBJECTNESS focal
      and applies to every cell.

      Why this lever rather than more class weighting: the FPN_CLSW ladder
      measured static per-class weights as net harmful (mAP none 0.1774 / sqrt
      0.1771 / inv 0.1368) because a fixed constant raises rare-class recall by
      flooding those classes with false positives, and AP is precision-sensitive
      — tricycle detections went 7,419 → 31,322 against 1,045 GT. Focal
      down-weights EASY examples whatever their class, and the weight tracks p_t
      as it moves, so it cannot buy recall with a permanent precision tax. -/
  yoloClsFocalGamma : Float := 0.0
  /-- RetinaNet prior-bias init: initialize the **head's bias** to `log π_c`
      instead of zero, so the net starts predicting the class prior rather than
      a uniform distribution. Empty (the default) leaves the head at zero bias
      and changes nothing.

      Orthogonal to `lossKind` — it is an init, not a loss — and deliberately
      so: it is the natural partner of `.perPixelFocalCE`, whose `(1-p_t)^γ`
      factor is a **no-op at a uniform softmax** because there is no confidence
      to suppress. Prior-bias init manufactures that confidence at step 0.
      See `NetSpec.applyHeadPriorBias` for the measured size of the effect. -/
  headPriorBias : List Float := []
  /-- RetinaNet prior-bias init for the FPN **detector** head: initialize every
      objectness bias to `−log((1−π)/π)` so the head starts at `sigmoid = π`.
      `0.0` (the default) leaves the head biases at zero, which reproduces the
      biasless head exactly. Typical value 0.01.

      The sigmoid-head twin of `headPriorBias` above, and the lever the
      loss-breakdown measurements pointed at: objectness had signal but no
      dynamic range, because a bias-free 1×1 conv spends its weights
      manufacturing the constant background offset. See
      `NetSpec.applyDetPriorBias`. -/
  detPriorPi : Float := 0.0
  /-- DeiT-style data augmentation knobs. Setting `useMixup` or
      `useCutmix` switches the train-step to the soft-label codegen
      path; the dataloader produces a `[B, NC]` smoothed soft-label
      tensor instead of an int32 `[B]` vector. `mixupAlpha` and
      `cutmixAlpha` control the Beta-distribution shape; the paper
      defaults are 0.8 and 1.0 respectively. `randomErasing` operates
      on the int-label path (no soft-label conversion needed). -/
  useMixup       : Bool  := false
  mixupAlpha     : Float := 0.8
  useCutmix      : Bool  := false
  cutmixAlpha    : Float := 1.0
  /-- KNN-Mixup: pair each sample with its nearest neighbor in the batch
      (pixel-space L2) rather than a random partner. Closer manifold
      mixing → harder, more realistic intermediate samples. Mutually
      exclusive with `useMixup`/`useCutmix` (KNN takes precedence). -/
  useKnnMixup    : Bool  := false
  knnMixupAlpha  : Float := 1.0
  randomErasing  : Bool  := false
  randomErasingProb : Float := 0.25
  /-- timm's `RandomErasing(mode='pixel')` in place of the zero-filled box: the box filled with N(0, 1) per pixel and channel on the
      normalised image, aspect log-uniform in [0.3, 1/0.3], up to 10 draws for a box that fits, and
      the box sized from the image actually erased (so a `trainRes` recipe erases its own crop).
      Off keeps the zero-fill emit byte-identical. -/
  erasingPixel : Bool := false
  /-- PIL-bicubic geometric augmentation: timm's ShearX/Y, TranslateX/Y and Rotate run PIL
      BICUBIC (the model's data config), and TF has no bicubic projective warp, so the emitter
      writes PIL's affine sampler out in the TF graph (a = −1, border clamp, truncation) in place of
      `ImageProjectiveTransformV3(BILINEAR)`. Off keeps every generated file byte-identical. -/
  augBicubic : Bool := false
  /-- RandAugment-Color (Cubuk et al. 2019, color subset). Applied
      per-image before mixup/cutmix, after crop/hflip. `randAugmentN`
      ops drawn uniformly from {identity, brightness, contrast, color,
      autocontrast} per image, each at magnitude `randAugmentM` (0–10,
      paper default 9). No labels touched. -/
  useRandAugment : Bool  := false
  randAugmentN   : Nat   := 2
  randAugmentM   : Float := 9.0
  /-- Upgrade `useRandAugment` from the color-only "lite" path to the full
      RandAugment(N, M) sampler over the color+GEOMETRIC op set (shear/rotate/
      translate via `ImageProjectiveTransformV3`, shared with AutoAugment). This
      is what ConvNeXt's recipe wants. Only meaningful when `useRandAugment` is
      on; leaving it false keeps the back-compat color-lite path (e.g. ViT). -/
  randAugmentGeometric : Bool := false
  /-- DeiT/ConvNeXt RandAugment refinements, meaningful only with the
      geometric sampler on. `randAugmentMstd` (timm `mstd`, DeiT uses 0.5) draws
      each op's magnitude from N(M, mstd) clipped to [0,10] instead of a fixed M.
      `randAugmentInc` (timm `inc1`) uses the increasing-severity magnitude→arg
      mappings: solarize/posterize flip so higher M = more distortion, and the
      enhancement ops center at 1.0 ± sign·scaled (random direction). -/
  randAugmentMstd : Float := 0.0
  randAugmentInc  : Bool  := false
  /-- AutoAugment, ImageNet learned policy (Cubuk et al. 2018) — the full 25
      sub-policies, applied per-image after crop/hflip on the imagenet (tfds)
      path. Unlike `useRandAugment` (color subset only), this includes the
      GEOMETRIC ops (shear/rotate) via `tf.raw_ops.ImageProjectiveTransformV3`
      (core TF, no `tfa`) plus
      the full color set (posterize/solarize/equalize/autocontrast/etc).
      Subsumes the color RandAugment, so leave `useRandAugment` off when this
      is on. EfficientNet's original recipe; no labels touched. -/
  useAutoAugment : Bool  := false
  /-- Repeated Augmentation (Hoffer et al. 2020; timm RASampler), RSB-A2's `3×`.
      Each image contributes `repeatedAug` independently-augmented copies per
      epoch. On the tfds path this is a stream-level `flat_map(repeat K)` before
      the augment `_pp`, plus a re-shuffle so the copies spread across batches —
      an APPROXIMATION of timm's exact index-level RASampler. `steps_per_epoch`
      is unchanged, so an epoch sees ~1/K as many unique images ×K views, per
      the RSB recipe. 1 disables. -/
  repeatedAug    : Nat   := 1
  /-- Train/test resolution split (RSB-A3): TRAIN at `trainRes`×`trainRes`, EVAL at
      the spec's `imageH/imageW`. 0 = no split (train and eval same resolution). The
      generated `forward` infers the square resolution from the flat input length, so
      the conv stack + global-avg-pool run at either size (A3 trains @160, tests @224
      → ~2× cheaper per step). imagenet (tfds) path only. -/
  trainRes       : Nat   := 0
  /-- Explicit test-time center-crop ratio (RSB-A3 uses 0.95). 0 = the default
      `_IMG_SIZE/(_IMG_SIZE+32)` ≈ 0.875 ratio. imagenet (tfds) path only. -/
  testCropRatio  : Float := 0.0
  /-- Stochastic depth (Huang et al. 2016): drop each residual block's
      branch with a probability that ramps linearly from 0 to `dropPath`
      across the network's residual blocks; surviving branches are scaled
      by 1/keep (inverted, so inference is drop-free). 0 disables. JAX path:
      threads a per-step RNG through `forward`, and drops at ConvNeXt, bottleneck,
      MBConv, UIB and transformer-encoder blocks. -/
  dropPath       : Float := 0.0
  /-- AdamW `no_weight_decay` exclusion (timm/DeiT): when true, decoupled
      weight decay skips 1-D params (all biases, LayerNorm γ/β, the CLS
      token) and the positional embedding, decaying only ≥2-D weight
      matrices. Matches the ViT/DeiT reference; off keeps the legacy
      decay-everything behavior for the other nets. AdamW path only. -/
  wdExcludeNormBias : Bool := false
  /-- Validate every N epochs (plus always the final epoch) instead of every
      epoch. N ≤ 1 keeps every-epoch validation (byte-identical codegen).
      Cuts eval wall-time on large streaming datasets where the val pass is
      data-loading-bound (e.g. ImageNet: ~75s/epoch rebuilding the tfds val
      pipeline). ImageNet-streaming main only. -/
  valEveryEpochs : Nat := 1
  /-- Gradient accumulation: run `gradAccumSteps` micro-batches of `batchSize`
      before each optimizer update, giving an EFFECTIVE batch of
      `batchSize × gradAccumSteps` at the peak-activation cost of ONE micro-batch.
      The reproducibility lever for large-batch recipes (e.g. RSB LAMB @ bs2048)
      on small GPUs: batchSize=512 × gradAccumSteps=4 on 4×16GB instead of an
      8×A100 node. `learningRate` should target the EFFECTIVE batch. BatchNorm
      uses per-micro-batch (Ghost-BN) statistics — not identical to true
      large-batch BN, but a benign/beneficial variant at micro≥256. N ≤ 1 keeps
      the single-shot update (byte-identical codegen). ImageNet-streaming main. -/
  gradAccumSteps : Nat := 1
  /-- DeiT-style training-loop knobs that average weights for the eval
      checkpoint. Both can be on simultaneously; eval picks EMA when
      both are enabled. Storage cost: one extra `nParams`-sized buffer
      per knob; runtime cost is one `F32.ema` call per step (EMA) or
      per epoch (SWA), well below GPU step time. -/
  useEMA         : Bool  := false
  emaDecay       : Float := 0.9999
  useSWA         : Bool  := false
  /-- First epoch (zero-indexed) that contributes to the SWA average.
      Typical recipe: 0.75 × epochs, e.g. epoch 60 of 80. -/
  swaStartEpoch  : Nat   := 0
  /-- SWAG (Maddox et al. 2019): extends SWA with diagonal Σ_diag (via
      running mean of θ²) plus a low-rank component built from the last
      `swagK` per-epoch deviations from the SWA mean. At eval, sample
      `swagSamples` weight vectors from N(swaMean, ½Σ_diag + ½ Σ_low),
      run forward each, average logits. Requires useSWA=true. -/
  useSWAG        : Bool  := false
  swagK          : Nat   := 20
  swagSamples    : Nat   := 30
  /-- TTA (test-time augmentation): at periodic eval, run `ttaSamples`
      independently-augmented forwards per batch and average the logits.
      Augmentations are the same dataloader pipeline used for training
      (e.g. random crop + hflip for Imagenette), minus the soft-label
      ones (mixup/cutmix) which need the label. Eval-only — no train
      cost, M× eval cost. -/
  useTTA         : Bool  := false
  ttaSamples     : Nat   := 5
  /-- YOLOv1 5-term masked-MSE loss. Equivalent to `lossKind := .yolov1Masked`; the bool form predates
      LossKind and is retained for back-compat. -/
  useYolov1      : Bool  := false
  /-- Explicit loss-kind selector. If left at the default `.classCE`,
      `compileVmfbs` derives the effective kind from the older booleans
      (`useYolov1`, `useSeg`, soft-label augs). Set explicitly to skip
      the derivation path or to disambiguate borderline cases.
      See `LossKind`. -/
  lossKind       : LossKind := LossKind.classCE
  /-- Bootstrap from a pretrained backbone checkpoint. When set to
      `some (paramsPath, prefixFloats)`, `runTraining` overwrites the
      first `prefixFloats * 4` bytes of the He-init with bytes read
      from `paramsPath`. The companion `<basename>_bn_stats.bin` is
      auto-loaded too if present (the backbone's BN running stats must
      match the spec's BN layer count + sizes — true for YOLOv1 loading
      R34 weights since both have identical backbone layers).

      Example for YOLOv1+R34:
      `bootstrapBackbone := some (".lake/build/resnet_34_params.bin", 21284672)`. -/
  bootstrapBackbone : Option (String × Nat) := none
  /-- Offset-aware bootstrap, as `some (paramsPath, dstOffFloats, srcOffFloats,
      countFloats)`. Same job as `bootstrapBackbone` but for a spec whose FIRST
      layer differs from the checkpoint's, so the transferable weights are no
      longer a prefix — see `NetSpec.patchInitWithPretrainedRange`.

      The motivating case is `r34UnetBrats`: a 4-modality MRI stem (`[64,4,7,7]`,
      12,544 floats) in front of an R34 body pretrained on 3-channel RGB
      (`[64,3,7,7]`, 9,408 floats). The stem keeps its He-init — MRI is not RGB,
      so re-learning it is the correct behaviour, not a concession — and the
      remaining 21,275,264 floats of the backbone transfer intact.

      Takes precedence over `bootstrapBackbone` when both are set. BN running
      stats follow the same rule as the prefix path: loaded only on an exact
      size match, zeros otherwise. -/
  bootstrapBackboneRange : Option (String × Nat × Nat × Nat) := none
  /-- Save intermediate `{pfx}_params_e{N}.bin` and
      `{pfx}_bn_stats_e{N}.bin` snapshots every `checkpointEveryNEpochs`
      epochs. 0 disables. Default 10 = align with the eval cadence so
      training can be killed mid-run and resumed from the most recent
      checkpoint (or borrowed for downstream tasks like YOLOv1
      bootstrap — see `bootstrapBackbone`). -/
  checkpointEveryNEpochs : Nat := 10
  /-- Run the validation eval every N epochs (plus always on the final epoch).
      0 means final-epoch only.

      Default 10 suits a 100-epoch classifier; a 10-epoch ablation at the
      default gets one eval, at the very end. On the seg path the eval is the
      only instrument that can see a collapsed class — no scalar in the training
      log can — so set it to 1-2 for ablation arms; the eval is one forward pass
      over val. -/
  evalEveryNEpochs : Nat := 10
  /-- bf16 mixed precision on the JAX path: cast matmul operands (dense, attention
      QKV / scores / output, MLP, patch embed) to bfloat16, keeping master
      weights, LayerNorm, softmax, and conv in fp32 (convs follow `bf16Conv`). -/
  bf16 : Bool := false
  /-- bf16 conv compute: additionally cast the standard conv path
      (`conv2d` / `conv_bn`, hence the full ResNet/VGG/CIFAR-CNN conv stack)
      to bfloat16, returning fp32. Independent of `bf16` so a backend where
      bf16 convolution is slower (MIOpen) can keep convs in fp32 (default
      `false`) while CUDA/cuDNN can opt in. Only meaningful
      when `bf16 := true`. Depthwise/separable convs (MobileNet/EfficientNet)
      still stay fp32. -/
  bf16Conv : Bool := false
  /-- Running batch-norm statistics. When true, the JAX imagenet
      trainer tracks per-BN-layer running mean/var (EMA of batch stats,
      momentum `bnMomentum`) threaded through `forward` as `has_aux`, and EVAL
      normalizes with the running stats instead of the eval batch's own —
      the paper-faithful behaviour. Off (default) keeps the current
      batch-stats-at-eval path. Wired for convBn, residual, bottleneck,
      inverted-residual, MBConv, UIB and fused-MBConv blocks. -/
  runningBN : Bool := false
  /-- BatchNorm running-statistic **decay**, i.e. the weight on the OLD estimate:
      `running = bnMomentum·running + (1−bnMomentum)·batch`. Only consulted when
      `runningBN` is on (it is an eval-time statistic and touches no gradient).

      This is the TensorFlow convention, the reciprocal of PyTorch's.
      `torch.nn.BatchNorm2d(momentum=m)` weights the NEW batch by `m`, so timm's
      PyTorch default `momentum = 0.1` is `bnMomentum = 0.9` here — a 10-step-vs-100-step
      averaging window, not a 10× smaller one. Getting the sense backwards silently
      makes the running stats 10× noisier, and it is eval-only, so no loss curve moves.

      Per-net, because the nets chase different references (checked against the pinned
      timm 1.0.28, `create_model(...).modules()`):

      | net | reference | value |
      |---|---|---|
      | R50 / R34 | timm (RSB) — `momentum=0.1` | 0.9 |
      | EfficientNet-B0 | the TF paper (77.1/93.3) — TF's `decay=0.99` | 0.99 |
      | MobileNetV2 | the TF-slim paper (72.0) — `decay=0.997` | 0.997 |
      | MNv4 | the repo's own 100-epoch JAX run, not a paper | 0.99 |

      timm itself runs 0.9 on every one of those nets, `tf_efficientnet_b0`
      included: `BN_MOMENTUM_TF_DEFAULT` exists in `_efficientnet_builder.py` but
      `get_bn_args_tf()` has no caller in 1.0.28, so the `tf_*` ports inherit only
      `bn_eps = 1e-3`. So "0.99 is right for EfficientNet" is a statement about the
      original TF codebase, not about timm — flip a net to 0.9 only along with the
      reference it is being scored against.

      The default 0.99 keeps every net that does not set it byte-identical, on both
      the JAX emitter ([`Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean)'s `_bn`) and the verified host-side EMA
      (`VerifiedTrain.lean`'s `bnMom`). Under gradient accumulation both sides
      compensate to `bnMomentum^(1/K)` per micro-batch. -/
  bnMomentum : Float := 0.99
  /-- BatchNorm ε — `_bn`'s `eps`, on the running-BN path (`runningBN`). 1e-5 is PyTorch's and
      timm's default; the TF papers (MobileNetV2 in slim, EfficientNet) use 1e-3. The default
      keeps the emitted `eps=1e-5` literal, so every net that does not set it is byte-identical.
      The verified peer is the render's `epsStr`, a different artifact per ε (the `eps<d…>`
      variant marker and its own `_fwd_eval_eps<d…>` eval graph), not a host-side knob. -/
  bnEps : Float := 1e-5
  /-- Stochastic-depth ramp over `i/N` (TF EfficientNet's `drop_rate · idx / len(blocks)`) instead
      of `i/(N−1)` (timm's `linspace(0, rate, depth)`), on the MBConv path only. Off keeps every
      generated file byte-identical. -/
  dropPathOverN : Bool := false
  /-- timm/DeiT ViT weight init, replacing the generic Xavier-uniform for
      transformer-shaped nets. Off by default so every existing run is
      byte-identical; turn it on per-recipe.

      timm's `init_weights_vit_timm` gives every `nn.Linear` (QKV, attn-out,
      MLP fc1/fc2, and the classifier head) `trunc_normal_(std=0.02)` with
      zero bias, and leaves the patch-embed `nn.Conv2d` on PyTorch's default
      `kaiming_uniform_(a=sqrt(5))`, i.e. `U(±1/sqrt(fan_in))`.

      NB `trunc_normal_`'s default bounds a=-2, b=2 are ABSOLUTE, so at
      std=0.02 they sit at ±100σ and the truncation never fires — it is
      plain `normal(0, 0.02)`. Emitted as such, matching the CLS-token and
      positional-embedding init already in the emitter.

      Why it matters: Xavier scales as 1/sqrt(dim) against a fixed 0.02, so
      the generic path is 1.8× too wide at ViT-B, 2.6× at ViT-S and 3.6× at
      ViT-Ti, while the patch embed is ~6× too NARROW (it divides by the
      output fan `dim·p·p` rather than the input fan `ic·p·p`). -/
  vitInit : Bool := false
  /-- ConvNeXt paper weight init, replacing the generic Xavier-uniform for the
      ConvNeXt-shaped nets. Off by default; turn it on per-recipe.

      ConvNeXt's reference implementation applies `trunc_normal_(std=.02)` with
      zero bias to **every `nn.Conv2d` AND `nn.Linear`** (`_init_weights` in
      `facebookresearch/ConvNeXt`, and timm's `convnext.py` agrees), leaving the
      LayerNorms at (1, 0) and LayerScale at 1e-6 — both of which the emitter
      already gets right.

      This is a separate flag from `vitInit`, not a shared "timmInit". The two specs disagree on the conv path: ViT leaves its
      patch-embed `nn.Conv2d` on PyTorch's default `U(±1/sqrt(fan_in))`, while
      ConvNeXt trunc-normals its convs like everything else. One boolean cannot
      express both, and a flag named for the vendor rather than the
      distribution would invite exactly the misapplication below.

      Neither flag applies to the ResNet/MobileNet/EfficientNet family. "timm init" is not one thing: those nets use
      `kaiming_normal_(mode='fan_out', nonlinearity='relu')`, and the emitter's
      `emitConvBnInit` — uniform `±sqrt(6/(oc·k²))`, i.e. std `sqrt(2/fan_out)`
      — is ALREADY on that scale, differing only in distribution shape. Giving
      a ResNet conv `trunc_normal(0.02)` would be a regression, not a fix.

      Why it matters: the generic `emitConvBiasInit` is Xavier over
      `ic·kh·kw + oc`, so ConvNeXt-T's stem lands at std 0.118 against the
      paper's 0.02 — 5.9× too wide, the same failure `vitInit` fixes for
      ViT. -/
  cnxInit : Bool := false
  /-- Exponential LR decay, the EfficientNet/MobileNet schedule: after
      warmup, `lr = LR · rate^((epoch − warmup) / decayEpochs)`. 0 = off (use
      cosine). EfficientNet: rate 0.97, decayEpochs 2.4; MobileNetV2: rate 0.98,
      decayEpochs 1.0. Selected over cosine when `> 0`. -/
  expLRDecayRate   : Float := 0.0
  expLRDecayEpochs : Float := 1.0
  /-- The exponential schedule as TF builds it — `exponential_decay(staircase=True)` on the GLOBAL
      step, `rate^⌊epoch / decayEpochs⌋`, with any warmup overriding it only while it runs (TF
      EfficientNet's `build_learning_rate`) — instead of the continuous exponent counted from the end
      of warmup. Only read when `expLRDecayRate > 0`. Off keeps every net's generated file
      byte-identical. -/
  expLRStaircase : Bool := false
  /-- Classifier dropout: dropout rate applied before the final dense
      head during training (inverted, scaled by 1/keep so eval is drop-free).
      0 = off. EfficientNet-B0 / MobileNetV2 use 0.2. Threaded via the same
      drop_key as stochastic depth; requires the running-BN `training` flag (or
      drop_key≠None) to gate train-vs-eval. -/
  dropout : Float := 0.0
  /-- Clip gradients by global L2 norm before the optimizer step. 0 = off.
      DeiT default 1.0 — essential for stable ViT-from-scratch training: it
      lets you use the proper ~1e-3 LR without the collapse-to-chance seen at
      higher LR with no clipping. -/
  gradClipNorm : Float := 0.0
  /-- Per-group LR multiplier for the (from-scratch) dense head, relative to
      the base LR used by the pretrained conv backbone. 1.0 = uniform LR. Used
      for bootstrap fine-tuning where the He-init head must learn input-
      dependence far faster than the backbone should drift — a single LR can't
      do both (head under-trains → collapse-to-marginal; raise it globally and
      the backbone destabilizes). e.g. YOLOv1 detection uses ~10. -/
  headLrMult : Float := 1.0
deriving Repr

/-- Which on-disk dataset a reference-path run reads, and so its record layout, image size,
    normalization and label format. -/
inductive DatasetKind where
  | mnist
  | cifar10
  | imagenette
  | imagenet
  /-- Box detection (VisDrone, NEU-DET). Images are ImageNet-normalized on Lean read; the
      labels carry the detector target, whose layout the run's config picks: the YOLOv1
      target + per-cell mask + box tail at 224/7×7, the same at another grid
      (`loadDetBinDims`), per-anchor targets (`cfg.anchors`), or the flat FPN block
      (`cfg.fpnScales`). See `scripts/datasets/preprocess_visdrone.py` for the on-disk format. -/
  | detection
  /-- Brain-tumour segmentation on the Medical Segmentation Decathlon
      Task01_BrainTumour volumes (BraTS-derived). 2D axial slices: images are
      240×240×4 (FLAIR / T1w / T1gd / T2w modalities as channels, z-scored per
      volume over brain voxels — no ImageNet normalization), labels are 240×240
      uint8 per-pixel classes (0=background, 1=edema, 2=non-enhancing tumour,
      3=enhancing tumour). Segmentation kind: `labelBytesPerRecord = 240*240`
      selects `.perPixelCE` automatically. See `scripts/datasets/preprocess_brats.py` for the
      on-disk format. -/
  | brats
  /-- The same BraTS data at 224×224, produced by
      `scripts/datasets/preprocess_brats.py --size 224` (a **center crop**, not a resize — see
      `fit_plane`). Identical in every other respect: same 4 modalities, same
      patient split at seed 0, same slice selection, same 14,415/2,569 counts.

      224 exists for backbones with a /32 total stride, which 240 cannot serve
      (240/32 = 7.5). It is also ResNet-34's native ImageNet resolution. The
      crop is lossless for this dataset: across 4,000 sampled slices, the
      8-pixel border it removes contains zero tumour voxels and zero brain
      voxels — MSD's volumes are skull-stripped and centered, so the margin is
      pure background. Used by [`demos/MainUnetBratsR34.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/demos/MainUnetBratsR34.lean). -/
  | brats224
deriving Repr, BEq

/-- IREE compile flags from environment. Defaults to CUDA (sm_86).
    Set `IREE_BACKEND=rocm` and `IREE_CHIP=gfx1100` for AMD GPUs.
    Set `IREE_BACKEND=llvm-cpu` for CPU fallback (no chip needed). -/
def ireeCompileArgs (mlirPath outPath : String) : IO (Array String) := do
  let backend ← (IO.getEnv "IREE_BACKEND").map (·.getD "cuda")
  let baseArgs := #[mlirPath, s!"--iree-hal-target-backends={backend}"]
  let chipArgs ← if backend == "llvm-cpu" then
    pure #[]
  else
    let defaultChip := if backend == "rocm" then "gfx1100" else "sm_86"
    let chip ← (IO.getEnv "IREE_CHIP").map (·.getD defaultChip)
    pure #[s!"--iree-{backend}-target={chip}"]
  -- gfx1100 workaround: IREE's reduction vector-distribution pipeline
  -- fails to distribute full N-D→scalar reductions (the YOLOv1 masked
  -- loss at batch 16 emits `matvec_like_16x2x49` reductions that abort
  -- with "'func.func' op failed to distribute"). Disabling just the
  -- reduction pipeline routes those to the legacy lowering and compiles
  -- cleanly; conv/matmul keep vector distribution, so backbone perf is
  -- unaffected. Verified no regression on the ResNet-34 train step.
  let extraArgs := if backend == "rocm" then
    #["--iree-codegen-llvmgpu-use-reduction-vector-distribution=false"]
  else #[]
  -- IREE_EXTRA_FLAGS: space-split extra iree-compile args from the env, appended
  -- last (they win). Probe affordance for rebuild-free flag sweeps — see
  -- planning/archive/iree_trainstep_memory_scaling.md (2026-07-08 A100 session).
  let userArgs ← (IO.getEnv "IREE_EXTRA_FLAGS").map fun s =>
    ((s.getD "").splitOn " ").filter (· ≠ "") |>.toArray
  return baseArgs ++ chipArgs ++ extraArgs ++ userArgs ++ #["-o", outPath]

/-- The `iree-compile` to run: `.venv/bin/iree-compile` when present (local dev), else the one on
    `PATH` (Docker, system install). Every compile in the repo resolves its compiler here. -/
def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  return "iree-compile"

/-- **Compile one MLIR module and report.** Writes `body` to `.lake/build/{name}.mlir`, runs
    `iree-compile` on it, prints the verdict, and returns `true` on success. `stderrTake` bounds
    how much of `iree-compile`'s stderr is echoed. -/
def compileCheckB (name body : String) (stderrTake : Nat := 3000) : IO Bool := do
  IO.FS.createDirAll ".lake/build"
  let path := s!".lake/build/{name}.mlir"
  IO.FS.writeFile path body
  let cargs ← ireeCompileArgs path s!".lake/build/{name}.vmfb"
  let r ← IO.Process.output { cmd := (← findIreeCompile), args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[{name}] iree-compile FAILED:\n{r.stderr.take stderrTake}"; return false
  else
    IO.println s!"[{name}] iree-compile OK → .lake/build/{name}.vmfb"; return true

/-- `compileCheckB` with the verdict discarded — the shape most gates want. -/
def compileCheck (name body : String) (stderrTake : Nat := 3000) : IO Unit :=
  discard <| compileCheckB name body stderrTake

/-- **Compile `src` → `dst`, tolerating an absent compiler.** Unlike `compileCheck` this takes an
    MLIR file that already exists and never throws: a missing `iree-compile` is reported and
    stepped over, so a smoke gate still runs on a machine without IREE installed. -/
def tryCompile (src dst label : String) (stderrTake : Nat := 3000) : IO Unit := do
  try
    let cargs ← ireeCompileArgs src dst
    let r ← IO.Process.output { cmd := (← findIreeCompile), args := cargs }
    if r.exitCode != 0 then
      IO.eprintln s!"iree-compile ({label}) FAILED:\n{r.stderr.take stderrTake}"
    else IO.println s!"{label} iree-compile OK → {src}"
  catch e => IO.eprintln s!"iree-compile ({label}) skipped (compiler unavailable): {e}"

