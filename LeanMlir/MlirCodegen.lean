import LeanMlir.Types
import LeanMlir.Spec

set_option maxRecDepth 2000

/-! MLIR (StableHLO) code generator: emit MLIR modules from `NetSpec`.

    Supports MLPs (`.dense`) and CNNs (`.conv2d`, `.maxPool`, `.flatten`).
    Walks layers tracking the current activation shape. For CNNs with a flat
    input, emits a reshape to (batch, ic, imageH, imageW) at the head.

    Layout: NCHW tensors, OIHW kernels (matches the JAX codegen convention).
    Params are interleaved as (W0, b0, W1, b1, ...) in the function signature,
    with `idx` advancing for every conv or dense layer. -/

namespace MlirCodegen

/-- Total flat input size: for CNN, ic * imageH * imageW; for MLP, first
    dense layer's fanIn. -/
def inputFlatDim (spec : NetSpec) : Nat :=
  match spec.layers.head? with
  | some (.dense fanIn _ _) => fanIn
  | some (.conv2d ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.convBn ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.invertedResidual ic _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.mbConv ic _ _ _ _ _ _) => ic * spec.imageH * spec.imageW
  | some (.patchEmbed ic _ _ _) => ic * spec.imageH * spec.imageW
  | some (.unetDown ic _) => ic * spec.imageH * spec.imageW
  | some (.tokenPositionEmbed v t _ ids gather _) => if ids || gather then t else v * t
  | _ => spec.imageH * spec.imageW

/-- If the first layer is conv/convBn, returns the NCHW input channels. -/
def inputChannels (spec : NetSpec) : Option Nat :=
  match spec.layers.head? with
  | some (.conv2d ic _ _ _ _) => some ic
  | some (.convBn ic _ _ _ _) => some ic
  | some (.invertedResidual ic _ _ _ _) => some ic
  | some (.mbConv ic _ _ _ _ _ _) => some ic
  | some (.patchEmbed ic _ _ _) => some ic
  | some (.unetDown ic _) => some ic
  | _ => none

/-- Render tensor type: `tensor<128x1x28x28xf32>`. -/
private def tensorTy (dims : List Nat) : String :=
  "tensor<" ++ String.intercalate "x" (dims.map toString) ++ "xf32>"

/-- Sanitize a string for use as an MLIR identifier. Public so trainers
    can derive eval-call names from `spec.name` instead of hardcoding
    them (and getting them wrong — see e.g. the EffNetV2 eval-name bug). -/
def sanitize (s : String) : String :=
  s.toLower.map (fun c => if c.isAlphanum then c else '_')

/-- Compute SAME padding. Returns (padH0, padH1, padW0, padW1). -/
private def samePad (h w kSize stride : Nat) : Nat × Nat × Nat × Nat :=
  let oH := (h + stride - 1) / stride
  let oW := (w + stride - 1) / stride
  let tH := if (oH - 1) * stride + kSize > h then (oH - 1) * stride + kSize - h else 0
  let tW := if (oW - 1) * stride + kSize > w then (oW - 1) * stride + kSize - w else 0
  (tH / 2, tH - tH / 2, tW / 2, tW - tW / 2)

/-- Standard conv dimension numbers block (NCHW/OIHW). -/
private def convDimNumbers : String :=
  "        dimension_numbers = #stablehlo.conv<raw\n" ++
  "          input_batch_dimension = 0, input_feature_dimension = 1,\n" ++
  "          input_spatial_dimensions = [2, 3],\n" ++
  "          kernel_output_feature_dimension = 0, kernel_input_feature_dimension = 1,\n" ++
  "          kernel_spatial_dimensions = [2, 3],\n" ++
  "          output_batch_dimension = 0, output_feature_dimension = 1,\n" ++
  "          output_spatial_dimensions = [2, 3]>,\n"

/-- Standard conv attribute block (NCHW/OIHW), symmetric padding, stride 1. -/
private def convAttrBlock (pad : Nat) : String :=
  "        batch_group_count = 1 : i64,\n" ++
  convDimNumbers ++
  "        feature_group_count = 1 : i64,\n" ++
  s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n" ++
  "        rhs_dilation = array<i64: 1, 1>,\n" ++
  "        window_strides = array<i64: 1, 1>\n"

/-- General conv attribute block with explicit asymmetric padding and optional dilations. -/
private def convAttrBlockFull (pH0 pH1 pW0 pW1 : Nat)
    (sH sW : Nat := 1) (lhsH lhsW : Nat := 1) (rhsH rhsW : Nat := 1) : String :=
  "        batch_group_count = 1 : i64,\n" ++
  convDimNumbers ++
  "        feature_group_count = 1 : i64,\n" ++
  s!"        lhs_dilation = array<i64: {lhsH}, {lhsW}>,\n" ++
  s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n" ++
  s!"        rhs_dilation = array<i64: {rhsH}, {rhsW}>,\n" ++
  s!"        window_strides = array<i64: {sH}, {sW}>\n"

/-- Depthwise conv attribute block: feature_group_count = channels. -/
private def dwConvAttrBlock (pad : Nat) (channels : Nat) : String :=
  "        batch_group_count = 1 : i64,\n" ++
  convDimNumbers ++
  s!"        feature_group_count = {channels} : i64,\n" ++
  s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n" ++
  "        rhs_dilation = array<i64: 1, 1>,\n" ++
  "        window_strides = array<i64: 1, 1>\n"

/-- General depthwise conv attribute block with asymmetric padding, strides, dilations. -/
private def dwConvAttrBlockFull (pH0 pH1 pW0 pW1 : Nat) (channels : Nat)
    (sH sW : Nat := 1) (lhsH lhsW : Nat := 1) (rhsH rhsW : Nat := 1)
    (batchGroupCount : Nat := 1) : String :=
  s!"        batch_group_count = {batchGroupCount} : i64,\n" ++
  convDimNumbers ++
  s!"        feature_group_count = {channels} : i64,\n" ++
  s!"        lhs_dilation = array<i64: {lhsH}, {lhsW}>,\n" ++
  s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n" ++
  s!"        rhs_dilation = array<i64: {rhsH}, {rhsW}>,\n" ++
  s!"        window_strides = array<i64: {sH}, {sW}>\n"

/-- Emit an activation forward as inline StableHLO. Takes a uniqueness tag
    (e.g. `s!"_mb{p}"`), the input SSA, and the tensor shape. Returns
    (code, output_ssa). All five activation kinds handled inline so callers
    don't need to forward-reference helpers further down in this file. -/
private def emitActForward (act : Activation) (tag inSSA : String) (shape : List Nat)
    : String × String := Id.run do
  let ty := tensorTy shape
  match act with
  | .identity => return ("", inSSA)
  | .relu =>
    let s := s!"    %act_z{tag} = stablehlo.constant dense<0.0> : {ty}\n" ++
             s!"    %act_o{tag} = stablehlo.maximum {inSSA}, %act_z{tag} : {ty}\n"
    return (s, s!"%act_o{tag}")
  | .relu6 =>
    let s := s!"    %act_z{tag} = stablehlo.constant dense<0.0> : {ty}\n" ++
             s!"    %act_six{tag} = stablehlo.constant dense<6.0> : {ty}\n" ++
             s!"    %act_r{tag} = stablehlo.maximum {inSSA}, %act_z{tag} : {ty}\n" ++
             s!"    %act_o{tag} = stablehlo.minimum %act_r{tag}, %act_six{tag} : {ty}\n"
    return (s, s!"%act_o{tag}")
  | .swish =>
    let s := s!"    %act_sig{tag} = stablehlo.logistic {inSSA} : {ty}\n" ++
             s!"    %act_o{tag} = stablehlo.multiply {inSSA}, %act_sig{tag} : {ty}\n"
    return (s, s!"%act_o{tag}")
  | .hSwish =>
    let s := s!"    %act_three{tag} = stablehlo.constant dense<3.0> : {ty}\n" ++
             s!"    %act_six{tag} = stablehlo.constant dense<6.0> : {ty}\n" ++
             s!"    %act_z{tag} = stablehlo.constant dense<0.0> : {ty}\n" ++
             s!"    %act_xp3{tag} = stablehlo.add {inSSA}, %act_three{tag} : {ty}\n" ++
             s!"    %act_clip{tag} = stablehlo.minimum %act_xp3{tag}, %act_six{tag} : {ty}\n" ++
             s!"    %act_r6{tag} = stablehlo.maximum %act_clip{tag}, %act_z{tag} : {ty}\n" ++
             s!"    %act_div{tag} = stablehlo.divide %act_r6{tag}, %act_six{tag} : {ty}\n" ++
             s!"    %act_o{tag} = stablehlo.multiply {inSSA}, %act_div{tag} : {ty}\n"
    return (s, s!"%act_o{tag}")
  | .gelu =>
    -- GELU tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
    let s := s!"    %act_05{tag} = stablehlo.constant dense<0.5> : {ty}\n" ++
             s!"    %act_one{tag} = stablehlo.constant dense<1.0> : {ty}\n" ++
             s!"    %act_c{tag} = stablehlo.constant dense<0.7978845608028654> : {ty}\n" ++
             s!"    %act_k{tag} = stablehlo.constant dense<0.044715> : {ty}\n" ++
             s!"    %act_x2{tag} = stablehlo.multiply {inSSA}, {inSSA} : {ty}\n" ++
             s!"    %act_x3{tag} = stablehlo.multiply %act_x2{tag}, {inSSA} : {ty}\n" ++
             s!"    %act_kx3{tag} = stablehlo.multiply %act_x3{tag}, %act_k{tag} : {ty}\n" ++
             s!"    %act_inner{tag} = stablehlo.add {inSSA}, %act_kx3{tag} : {ty}\n" ++
             s!"    %act_u{tag} = stablehlo.multiply %act_inner{tag}, %act_c{tag} : {ty}\n" ++
             s!"    %act_t{tag} = stablehlo.tanh %act_u{tag} : {ty}\n" ++
             s!"    %act_op1{tag} = stablehlo.add %act_one{tag}, %act_t{tag} : {ty}\n" ++
             s!"    %act_hx{tag} = stablehlo.multiply {inSSA}, %act_05{tag} : {ty}\n" ++
             s!"    %act_o{tag} = stablehlo.multiply %act_hx{tag}, %act_op1{tag} : {ty}\n"
    return (s, s!"%act_o{tag}")

/-- Emit a conv2d + bias + activation block. Returns (code, newSSA, newShape).
    Assumes NCHW input [b, ic, h, w], produces [b, oc, h, w] for SAME padding,
    stride 1. -/
private def emitConv2d (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize : Nat) (act : Activation) : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let newShape := [b, oc, h, w]
    let pad := (kSize - 1) / 2
    let kShape := tensorTy [oc, ic, kSize, kSize]
    let mut s := ""
    s := s ++ s!"    %cv{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ "        dimension_numbers = #stablehlo.conv<raw\n"
    s := s ++ "          input_batch_dimension = 0,\n"
    s := s ++ "          input_feature_dimension = 1,\n"
    s := s ++ "          input_spatial_dimensions = [2, 3],\n"
    s := s ++ "          kernel_output_feature_dimension = 0,\n"
    s := s ++ "          kernel_input_feature_dimension = 1,\n"
    s := s ++ "          kernel_spatial_dimensions = [2, 3],\n"
    s := s ++ "          output_batch_dimension = 0,\n"
    s := s ++ "          output_feature_dimension = 1,\n"
    s := s ++ "          output_spatial_dimensions = [2, 3]\n"
    s := s ++ "        >,\n"
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {kShape}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cvb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1]" ++
         s!" : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cva{pidx} = stablehlo.add %cv{pidx}, %cvb{pidx} : {tensorTy newShape}\n"
    match act with
    | .relu =>
      s := s ++ s!"    %cvz{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
      s := s ++ s!"    %cvr{pidx} = stablehlo.maximum %cva{pidx}, %cvz{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cvr{pidx}", newShape)
    | .identity => return (s, s!"%cva{pidx}", newShape)
    | .relu6 => return (s, s!"%cva{pidx}", newShape)
    | .swish =>
      s := s ++ s!"    %cvsig{pidx} = stablehlo.logistic %cva{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cvsw{pidx} = stablehlo.multiply %cva{pidx}, %cvsig{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cvsw{pidx}", newShape)
    | .hSwish =>
      let ty := tensorTy newShape
      s := s ++ s!"    %ch3_{pidx} = stablehlo.constant dense<3.0> : {ty}\n"
      s := s ++ s!"    %ch6_{pidx} = stablehlo.constant dense<6.0> : {ty}\n"
      s := s ++ s!"    %ch0_{pidx} = stablehlo.constant dense<0.0> : {ty}\n"
      s := s ++ s!"    %chxp3_{pidx} = stablehlo.add %cva{pidx}, %ch3_{pidx} : {ty}\n"
      s := s ++ s!"    %chc_{pidx} = stablehlo.minimum %chxp3_{pidx}, %ch6_{pidx} : {ty}\n"
      s := s ++ s!"    %chr_{pidx} = stablehlo.maximum %chc_{pidx}, %ch0_{pidx} : {ty}\n"
      s := s ++ s!"    %chd_{pidx} = stablehlo.divide %chr_{pidx}, %ch6_{pidx} : {ty}\n"
      s := s ++ s!"    %chsw_{pidx} = stablehlo.multiply %cva{pidx}, %chd_{pidx} : {ty}\n"
      return (s, s!"%chsw_{pidx}", newShape)
    | .gelu =>
      let (sa, oa) := emitActForward .gelu s!"_cv{pidx}" s!"%cva{pidx}" newShape
      return (s ++ sa, oa, newShape)
  | _ => return ("    // conv2d error: expected rank-4 NCHW shape\n", curSSA, curShape)

/-- Emit convBn: conv (possibly strided) + instance norm + optional ReLU.
    Params: W{pidx} (kernel), g{pidx} (gamma/scale), bt{pidx} (beta/shift).
    Instance norm: per-sample, per-channel spatial statistics. -/
private def emitConvBn (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) (relu : Bool := true) (fixedBN : Bool := false)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let newShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy newShape}\n"
    if fixedBN then
      -- Use fixed running mean/var from inputs %bn_mean{pidx}, %bn_var{pidx}
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %bn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %bn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    else
      -- Batch norm: mean/var via 2-step reduction [2,3] then [0]
      let bnN := b * oH * oW
      s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
      s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
      s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    -- Normalize + affine (shared path)
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy newShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy newShape}\n"
    if relu then
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cbn_out{pidx}", newShape)
    else
      return (s, preSSA, newShape)
  | _ => return ("    // convBn error\n", curSSA, curShape)

/-- Emit depthwise convBn: depthwise conv + BN + optional ReLU6.
    Weight shape: (channels, 1, kSize, kSize), feature_group_count = channels.
    Output channels = input channels.
    If `noAct := true`, emits only convBn (no activation), returning pre-activation
    so the caller can attach its own (e.g., Swish for MBConv). -/
private def emitDepthwiseConvBn (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (channels kSize stride : Nat) (fixedBN : Bool := false) (noAct : Bool := false)
    (useRelu : Bool := false)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let newShape := [b, channels, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Depthwise conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy newShape}\n"
    if fixedBN then
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %bn_mean{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %bn_var{pidx}, %cbn_eps{pidx} : {tensorTy [channels]}\n"
    else
      let bnN := b * oH * oW
      s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
      s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({tensorTy newShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
      s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [channels]}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [channels]}\n"
    -- Normalize + affine
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy newShape}\n"
    if noAct then
      return (s, preSSA, newShape)
    if useRelu then
      -- Plain ReLU (used by UIB / MobileNet V4)
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy newShape}\n"
      return (s, s!"%cbn_out{pidx}", newShape)
    -- ReLU6 (default)
    s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_r{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_six{pidx} = stablehlo.constant dense<6.0> : {tensorTy newShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.minimum %cbn_r{pidx}, %cbn_six{pidx} : {tensorTy newShape}\n"
    return (s, s!"%cbn_out{pidx}", newShape)
  | _ => return ("    // depthwiseConvBn error\n", curSSA, curShape)

/-- Emit a raw depthwise conv (no BN, no activation). Used by ConvNeXt
    blocks where the DW output goes straight into LayerNorm.
    Weight shape (channels, 1, kSize, kSize), bias [channels]. -/
private def emitDepthwiseConvRaw (pidx : Nat) (curSSA : String) (curShape : List Nat)
    (channels kSize stride : Nat) : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let newShape := [b, channels, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    s := s ++ s!"    %dwr_cv{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %dwr_bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy newShape}\n"
    s := s ++ s!"    %dwr_out{pidx} = stablehlo.add %dwr_cv{pidx}, %dwr_bb{pidx} : {tensorTy newShape}\n"
    return (s, s!"%dwr_out{pidx}", newShape)
  | _ => return ("    // depthwiseConvRaw error\n", curSSA, curShape)

/-- Emit an inverted residual block stage for inference.
    Returns (code, newSSA, newShape, newPidx). -/
private def emitInvertedResidual (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand firstStride nBlocks : Nat) (fixedBN : Bool := false)
    : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    let mid := blockIc * expand
    let useSkip := stride == 1 && blockIc == oc
    -- 1. Expand: 1×1 convBn + ReLU6 (skip if expand == 1)
    if expand != 1 then
      let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid 1 1 false (fixedBN := fixedBN)
      code := code ++ s1
      -- Manual ReLU6 after convBn (which was called with relu=false)
      code := code ++ s!"    %ir_r6z{p} = stablehlo.constant dense<0.0> : {tensorTy sh1}\n"
      code := code ++ s!"    %ir_r6r{p} = stablehlo.maximum {out1}, %ir_r6z{p} : {tensorTy sh1}\n"
      code := code ++ s!"    %ir_r6s{p} = stablehlo.constant dense<6.0> : {tensorTy sh1}\n"
      code := code ++ s!"    %ir_r6o{p} = stablehlo.minimum %ir_r6r{p}, %ir_r6s{p} : {tensorTy sh1}\n"
      ssa := s!"%ir_r6o{p}"; shape := sh1; p := p + 1
    -- 2. Depthwise: 3×3 depthwise convBn + ReLU6
    let (s2, out2, sh2) := emitDepthwiseConvBn p ssa shape mid 3 stride (fixedBN := fixedBN)
    code := code ++ s2; ssa := out2; shape := sh2; p := p + 1
    -- 3. Project: 1×1 convBn, NO activation
    let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
    code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- 4. Skip connection: if stride==1 AND blockIc==oc, add (no ReLU)
    if useSkip then
      code := code ++ s!"    %ir_add{startPidx}_{bi} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
      ssa := s!"%ir_add{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit Squeeze-and-Excitation block.
    Input: x : (B, mid, H, W). Output: x * σ(W_exp · swish(W_red · GAP(x) + b_red) + b_exp).
    Uses dense (dot_general) in place of 1×1 convs on (B, mid) / (B, seMid) tensors.
    Params: reduce at `pRed` (W: [mid, seMid], b: [seMid]),
            expand at `pExp` (W: [seMid, mid], b: [mid]).
    NOTE: weight shapes in MLIR params are stored [seMid, mid, 1, 1] / [mid, seMid, 1, 1]
    (conv style) — we reshape them to dense layout here.
    Returns (code, outSSA, shape) where shape == input shape. -/
private def emitSEBlock (tag : String) (xSSA : String) (xShape : List Nat)
    (mid seMid pRed pExp : Nat) : String × String := Id.run do
  match xShape with
  | [b, _, h, w] =>
    let xTy := tensorTy xShape
    let mut s := ""
    -- 1. Squeeze: GAP over spatial dims → (B, mid)
    s := s ++ s!"    %se_gs{tag} = stablehlo.reduce({xSSA} init: %se_zf{tag}_init) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
    -- 2. Reduce (dense): W_red reshape [seMid, mid, 1, 1] → [seMid, mid], then matmul.
    --    out[b,s] = sum_m g[b,m] * W_red[s,m] + b_red[s]
    s := s ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
    s := s ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag},\n"
    s := s ++ "              contracting_dims = [1] x [1],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
    s := s ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
    s := s ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
    -- 3. Swish: rb * σ(rb)
    s := s ++ s!"    %se_rsig{tag} = stablehlo.logistic %se_rb{tag} : {tensorTy [b, seMid]}\n"
    s := s ++ s!"    %se_rsw{tag} = stablehlo.multiply %se_rb{tag}, %se_rsig{tag} : {tensorTy [b, seMid]}\n"
    -- 4. Expand (dense): W_exp reshape [mid, seMid, 1, 1] → [mid, seMid], then matmul.
    s := s ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
    s := s ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag},\n"
    s := s ++ "              contracting_dims = [1] x [1],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
    s := s ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
    -- 5. Sigmoid gate
    s := s ++ s!"    %se_sig{tag} = stablehlo.logistic %se_eb{tag} : {tensorTy [b, mid]}\n"
    -- 6. Excite: broadcast multiply. Reshape (B, mid) → (B, mid, 1, 1) → (B, mid, H, W)
    s := s ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
    s := s ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
    s := s ++ s!"    %se_out{tag} = stablehlo.multiply {xSSA}, %se_sigb{tag} : {xTy}\n"
    return (s, s!"%se_out{tag}")
  | _ => return ("    // SE error: expected rank-4 input\n", xSSA)

/-- Emit an MBConv (EfficientNet) block stage for inference.
    Same structure as InvertedResidual but:
      - uses the layer's `kSize` (not hardcoded 3)
      - uses Swish activation (x * sigmoid(x)) instead of ReLU6
      - optional Squeeze-and-Excitation between depthwise and project.
    Returns (code, newSSA, newShape, newPidx). -/
private def emitMbConv (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand kSize firstStride nBlocks : Nat) (useSE : Bool)
    (act : Activation := .swish)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    let mid := blockIc * expand
    let seMid := Nat.max 1 (mid / 4)
    let useSkip := stride == 1 && blockIc == oc
    -- 1. Expand: 1×1 convBn + activation (skip if expand == 1)
    if expand != 1 then
      let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid 1 1 false (fixedBN := fixedBN)
      code := code ++ s1
      let (sa, oa) := emitActForward act s!"_mb{p}" out1 sh1
      code := code ++ sa
      ssa := oa; shape := sh1; p := p + 1
    -- 2. Depthwise: k×k depthwise convBn + activation
    let (s2, out2, sh2) := emitDepthwiseConvBn p ssa shape mid kSize stride
                             (fixedBN := fixedBN) (noAct := true)
    code := code ++ s2
    let (sa2, oa2) := emitActForward act s!"_mbdw{p}" out2 sh2
    code := code ++ sa2
    ssa := oa2; shape := sh2; p := p + 1
    -- 3. SE block (optional)
    if useSE then
      let tag := s!"_i{startPidx}_{bi}"
      code := code ++ s!"    %se_zf{tag}_init = stablehlo.constant dense<0.0> : tensor<f32>\n"
      let (sse, outSE) := emitSEBlock tag ssa shape mid seMid p (p + 1)
      code := code ++ sse
      ssa := outSE
      p := p + 2
    -- 4. Project: 1×1 convBn, NO activation
    let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
    code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- 5. Skip connection: if stride==1 AND blockIc==oc, add (no post-add activation)
    if useSkip then
      code := code ++ s!"    %mb_add{startPidx}_{bi} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
      ssa := s!"%mb_add{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit a Fused-MBConv (EfficientNet V2) block stage for inference.
    Unlike regular MBConv, fused-MBConv replaces the 1×1 expand + k×k depthwise pair
    with a single regular k×k convolution.
    Structure per block:
      1. Fused expand: k×k regular convBn (blockIc → mid) + Swish
      2. Optional SE
      3. Project: 1×1 convBn (mid → oc), linear (SKIPPED if expand == 1)
      4. Skip if blockIc == oc && stride == 1
    Returns (code, newSSA, newShape, newPidx). -/
private def emitFusedMbConv (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand kSize firstStride nBlocks : Nat) (useSE : Bool)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    let mid := if expand == 1 then oc else blockIc * expand
    let seMid := Nat.max 1 (mid / 4)
    let useSkip := stride == 1 && blockIc == oc
    -- 1. Fused expand: regular k×k convBn, Swish activation.
    let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid kSize stride false (fixedBN := fixedBN)
    code := code ++ s1
    code := code ++ s!"    %fmb_sig{p} = stablehlo.logistic {out1} : {tensorTy sh1}\n"
    code := code ++ s!"    %fmb_sw{p} = stablehlo.multiply {out1}, %fmb_sig{p} : {tensorTy sh1}\n"
    ssa := s!"%fmb_sw{p}"; shape := sh1; p := p + 1
    -- 2. SE block (optional) — same as MBConv (swish reduce, sigmoid gate).
    if useSE then
      let tag := s!"_f{startPidx}_{bi}"
      code := code ++ s!"    %se_zf{tag}_init = stablehlo.constant dense<0.0> : tensor<f32>\n"
      let (sse, outSE) := emitSEBlock tag ssa shape mid seMid p (p + 1)
      code := code ++ sse
      ssa := outSE
      p := p + 2
    -- 3. Project: 1×1 convBn, NO activation — only when expand != 1.
    if expand != 1 then
      let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
      code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- 4. Skip connection: if stride==1 AND blockIc==oc, add (no post-add activation)
    if useSkip then
      code := code ++ s!"    %fmb_add{startPidx}_{bi} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
      ssa := s!"%fmb_add{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit h-swish forward: x * ReLU6(x + 3) / 6
    Returns (code, output_ssa). Caller provides input_ssa and shape. -/
private def emitHSwishForward (tag : String) (inSSA : String) (shape : List Nat)
    : String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %hs_three{tag} = stablehlo.constant dense<3.0> : {ty}\n"
  s := s ++ s!"    %hs_six{tag} = stablehlo.constant dense<6.0> : {ty}\n"
  s := s ++ s!"    %hs_zero{tag} = stablehlo.constant dense<0.0> : {ty}\n"
  s := s ++ s!"    %hs_xp3{tag} = stablehlo.add {inSSA}, %hs_three{tag} : {ty}\n"
  s := s ++ s!"    %hs_clip{tag} = stablehlo.minimum %hs_xp3{tag}, %hs_six{tag} : {ty}\n"
  s := s ++ s!"    %hs_r6{tag} = stablehlo.maximum %hs_clip{tag}, %hs_zero{tag} : {ty}\n"
  s := s ++ s!"    %hs_div{tag} = stablehlo.divide %hs_r6{tag}, %hs_six{tag} : {ty}\n"
  s := s ++ s!"    %hs_out{tag} = stablehlo.multiply {inSSA}, %hs_div{tag} : {ty}\n"
  return (s, s!"%hs_out{tag}")

/-- Emit h-sigmoid forward: ReLU6(x + 3) / 6 -/
private def emitHSigmoidForward (tag : String) (inSSA : String) (shape : List Nat)
    : String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %hsg_three{tag} = stablehlo.constant dense<3.0> : {ty}\n"
  s := s ++ s!"    %hsg_six{tag} = stablehlo.constant dense<6.0> : {ty}\n"
  s := s ++ s!"    %hsg_zero{tag} = stablehlo.constant dense<0.0> : {ty}\n"
  s := s ++ s!"    %hsg_xp3{tag} = stablehlo.add {inSSA}, %hsg_three{tag} : {ty}\n"
  s := s ++ s!"    %hsg_clip{tag} = stablehlo.minimum %hsg_xp3{tag}, %hsg_six{tag} : {ty}\n"
  s := s ++ s!"    %hsg_r6{tag} = stablehlo.maximum %hsg_clip{tag}, %hsg_zero{tag} : {ty}\n"
  s := s ++ s!"    %hsg_out{tag} = stablehlo.divide %hsg_r6{tag}, %hsg_six{tag} : {ty}\n"
  return (s, s!"%hsg_out{tag}")

/-- Emit ReLU forward (plain, not ReLU6) -/
private def emitReluForward (tag : String) (inSSA : String) (shape : List Nat)
    : String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %rel_z{tag} = stablehlo.constant dense<0.0> : {ty}\n"
  s := s ++ s!"    %rel_o{tag} = stablehlo.maximum {inSSA}, %rel_z{tag} : {ty}\n"
  return (s, s!"%rel_o{tag}")

/-- Emit an MBConvV3 (MobileNetV3) block for inference. Single block.
    - `expandCh` is the absolute mid channel count.
    - If `expandCh == ic`, skip the expand 1×1 convBn.
    - Activation: h-swish if `useHSwish`, otherwise plain ReLU.
    - SE: uses ReLU on reduce, h-sigmoid on gate.
    - Skip connection iff `stride == 1 && ic == oc`. -/
private def emitMbConvV3 (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expandCh kSize stride : Nat) (useSE useHSwish : Bool)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  let blockIn := ssa
  let mid := expandCh
  let seMid := Nat.max 1 (mid / 4)
  let useSkip := stride == 1 && ic == oc
  -- 1. Expand: 1×1 convBn + activation (skip if expandCh == ic)
  if expandCh != ic then
    let (s1, out1, sh1) := emitConvBn p ssa shape ic mid 1 1 false (fixedBN := fixedBN)
    code := code ++ s1
    if useHSwish then
      let (sa, oa) := emitHSwishForward s!"_v3e{p}" out1 sh1
      code := code ++ sa; ssa := oa
    else
      let (sa, oa) := emitReluForward s!"_v3e{p}" out1 sh1
      code := code ++ sa; ssa := oa
    shape := sh1; p := p + 1
  -- 2. Depthwise: k×k depthwise convBn + activation
  let (s2, out2, sh2) := emitDepthwiseConvBn p ssa shape mid kSize stride
                           (fixedBN := fixedBN) (noAct := true)
  code := code ++ s2
  if useHSwish then
    let (sa, oa) := emitHSwishForward s!"_v3d{p}" out2 sh2
    code := code ++ sa; ssa := oa
  else
    let (sa, oa) := emitReluForward s!"_v3d{p}" out2 sh2
    code := code ++ sa; ssa := oa
  shape := sh2; p := p + 1
  -- 3. SE block (optional) - uses ReLU on reduce, h-sigmoid on gate
  if useSE then
    -- Inline SE for V3 (different activations than emitSEBlock):
    -- Squeeze: GAP over spatial → (B, mid)
    -- Reduce: dense mid→seMid + ReLU
    -- Expand: dense seMid→mid + h-sigmoid
    -- Excite: broadcast multiply
    match shape with
    | [b, _, h, w] =>
      let xTy := tensorTy shape
      let bcTy := tensorTy [b, mid]
      let bsTy := tensorTy [b, seMid]
      let tag := s!"_v3se{p}"
      -- GAP
      code := code ++ s!"    %sez{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      code := code ++ s!"    %ses{tag} = stablehlo.reduce({ssa} init: %sez{tag}) applies stablehlo.add across dimensions = [2, 3]\n"
      code := code ++ s!"          : ({xTy}, tensor<f32>) -> {bcTy}\n"
      code := code ++ s!"    %sen{tag} = stablehlo.constant dense<{(h * w).toFloat}> : {bcTy}\n"
      code := code ++ s!"    %segap{tag} = stablehlo.divide %ses{tag}, %sen{tag} : {bcTy}\n"
      -- Reduce: dense mid → seMid (W stored as [seMid, mid, 1, 1], reshape to [seMid, mid])
      code := code ++ s!"    %sewr{tag} = stablehlo.reshape %W{p} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
      code := code ++ s!"    %sered{tag} = stablehlo.dot_general %segap{tag}, %sewr{tag}, contracting_dims = [1] x [1],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT] : ({bcTy}, {tensorTy [seMid, mid]}) -> {bsTy}\n"
      code := code ++ s!"    %sebr_bc{tag} = stablehlo.broadcast_in_dim %b{p}, dims = [1] : ({tensorTy [seMid]}) -> {bsTy}\n"
      code := code ++ s!"    %sered_b{tag} = stablehlo.add %sered{tag}, %sebr_bc{tag} : {bsTy}\n"
      -- ReLU on reduce
      code := code ++ s!"    %seredz{tag} = stablehlo.constant dense<0.0> : {bsTy}\n"
      code := code ++ s!"    %sered_a{tag} = stablehlo.maximum %sered_b{tag}, %seredz{tag} : {bsTy}\n"
      -- Expand: dense seMid → mid
      let pe := p + 1
      code := code ++ s!"    %sewe{tag} = stablehlo.reshape %W{pe} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
      code := code ++ s!"    %seexp{tag} = stablehlo.dot_general %sered_a{tag}, %sewe{tag}, contracting_dims = [1] x [1],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT] : ({bsTy}, {tensorTy [mid, seMid]}) -> {bcTy}\n"
      code := code ++ s!"    %sebe_bc{tag} = stablehlo.broadcast_in_dim %b{pe}, dims = [1] : ({tensorTy [mid]}) -> {bcTy}\n"
      code := code ++ s!"    %seexp_b{tag} = stablehlo.add %seexp{tag}, %sebe_bc{tag} : {bcTy}\n"
      -- h-sigmoid on expand
      let (sgcode, sgout) := emitHSigmoidForward s!"hsg{tag}" s!"%seexp_b{tag}" [b, mid]
      code := code ++ sgcode
      -- Broadcast multiply with input
      code := code ++ s!"    %sebc{tag} = stablehlo.reshape {sgout} : ({bcTy}) -> {tensorTy [b, mid, 1, 1]}\n"
      code := code ++ s!"    %sebcb{tag} = stablehlo.broadcast_in_dim %sebc{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
      code := code ++ s!"    %seout{tag} = stablehlo.multiply {ssa}, %sebcb{tag} : {xTy}\n"
      ssa := s!"%seout{tag}"
    | _ => pure ()
    p := p + 2
  -- 4. Project: 1×1 convBn, NO activation
  let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
  code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
  -- 5. Skip connection
  if useSkip then
    code := code ++ s!"    %mb3_add{startPidx} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
    ssa := s!"%mb3_add{startPidx}"
  return (code, ssa, shape, p)

/-- Emit a Universal Inverted Bottleneck (UIB) block for inference. Single block.
    MobileNet V4 building block. Structure:
      1. Optional preDW (depthwise convBn + plain ReLU), consumes stride if present
      2. Expand 1×1 convBn + plain ReLU
      3. Optional postDW (depthwise convBn + plain ReLU), uses remaining stride
      4. Project 1×1 convBn, NO activation
      5. Skip if stride == 1 && ic == oc
    All activations are plain ReLU (NOT ReLU6, NOT Swish, NOT h-swish). -/
private def emitUib (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc expand stride preDWk postDWk : Nat)
    (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  let blockIn := ssa
  let mid := ic * expand
  let useSkip := stride == 1 && ic == oc
  -- 1. Optional preDW: consumes stride, then effective stride becomes 1
  let mut effectiveStride := stride
  if preDWk > 0 then
    let (s1, out1, sh1) := emitDepthwiseConvBn p ssa shape ic preDWk stride
                             (fixedBN := fixedBN) (useRelu := true)
    code := code ++ s1
    ssa := out1; shape := sh1; p := p + 1
    effectiveStride := 1
  -- 2. Expand 1×1 convBn + plain ReLU
  let (s2, out2, sh2) := emitConvBn p ssa shape ic mid 1 1 true (fixedBN := fixedBN)
  code := code ++ s2
  ssa := out2; shape := sh2; p := p + 1
  -- 3. Optional postDW: uses remaining (effective) stride
  if postDWk > 0 then
    let (s3, out3, sh3) := emitDepthwiseConvBn p ssa shape mid postDWk effectiveStride
                             (fixedBN := fixedBN) (useRelu := true)
    code := code ++ s3
    ssa := out3; shape := sh3; p := p + 1
  -- 4. Project 1×1 convBn, NO activation
  let (s4, out4, sh4) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
  code := code ++ s4
  ssa := out4; shape := sh4; p := p + 1
  -- 5. Skip connection
  if useSkip then
    code := code ++ s!"    %uib_add{startPidx} = stablehlo.add {ssa}, {blockIn} : {tensorTy shape}\n"
    ssa := s!"%uib_add{startPidx}"
  return (code, ssa, shape, p)

/-- Emit global average pool: (b, c, h, w) -> (b, c). -/
private def emitGlobalAvgPool (pos : Nat) (curSSA : String) (curShape : List Nat)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let newShape := [b, c]
    let spatialN := h * w
    let mut s := ""
    s := s ++ s!"    %gap_zf{pos} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %gap_sum{pos} = stablehlo.reduce({curSSA} init: %gap_zf{pos}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy curShape}, tensor<f32>) -> {tensorTy newShape}\n"
    s := s ++ s!"    %gap_N{pos} = stablehlo.constant dense<{spatialN}.0> : {tensorTy newShape}\n"
    s := s ++ s!"    %gap{pos} = stablehlo.divide %gap_sum{pos}, %gap_N{pos} : {tensorTy newShape}\n"
    return (s, s!"%gap{pos}", newShape)
  | _ => return ("    // globalAvgPool error\n", curSSA, curShape)

/-- Emit max pool. Handles kSize != stride (e.g., 3/2 in ResNet). -/
private def emitMaxPool (pos : Nat) (curSSA : String) (curShape : List Nat)
    (size stride : Nat) : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let oH := (h + stride - 1) / stride  -- ceil div for SAME
    let oW := (w + stride - 1) / stride
    let newShape := [b, c, oH, oW]
    -- Padding for SAME pooling
    let padNeeded := (oH - 1) * stride + size
    let padH := if padNeeded > h then padNeeded - h else 0
    let padW := if padNeeded > w then padNeeded - w else 0
    let padH0 := padH / 2; let padH1 := padH - padH0
    let padW0 := padW / 2; let padW1 := padW - padW0
    let mut s := ""
    s := s ++ s!"    %pli{pos} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
    s := s ++ s!"    %pl{pos} = \"stablehlo.reduce_window\"({curSSA}, %pli{pos}) (" ++ "{\n"
    s := s ++ s!"      ^bb0(%rwa{pos}: tensor<f32>, %rwb{pos}: tensor<f32>):\n"
    s := s ++ s!"        %rwm{pos} = stablehlo.maximum %rwa{pos}, %rwb{pos} : tensor<f32>\n"
    s := s ++ s!"        \"stablehlo.return\"(%rwm{pos}) : (tensor<f32>) -> ()\n"
    s := s ++ "      }) " ++ "{\n"
    s := s ++ s!"        window_dimensions = array<i64: 1, 1, {size}, {size}>,\n"
    s := s ++ s!"        window_strides    = array<i64: 1, 1, {stride}, {stride}>"
    -- Add padding if needed (for pool kSize 3 stride 2)
    if padH > 0 || padW > 0 then
      s := s ++ s!",\n        padding = dense<[[0, 0], [0, 0], [{padH0}, {padH1}], [{padW0}, {padW1}]]> : tensor<4x2xi64>"
    s := s ++ "\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, tensor<f32>) -> {tensorTy newShape}\n"
    return (s, s!"%pl{pos}", newShape)
  | _ => return ("    // maxPool error: expected rank-4 input\n", curSSA, curShape)

/-- Emit flatten: (b, c, h, w) -> (b, c*h*w). -/
private def emitFlatten (pos : Nat) (curSSA : String) (curShape : List Nat)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let flat := c * h * w
    let newShape := [b, flat]
    let s := s!"    %fl{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy newShape}\n"
    return (s, s!"%fl{pos}", newShape)
  | _ => return ("    // flatten: already flat or unknown rank\n", curSSA, curShape)

/-- Channel-concatenate two NCHW tensors `A : (N, Ca, H, W)` and
    `B : (N, Cb, H, W)` along the channel axis (dim 1). Output:
    `(N, Ca + Cb, H, W)`. Used by UNet decoder skip-connection
    fusion (`emitUnetUp` will call this). The two inputs must
    share `(N, H, W)`. -/
private def emitChannelConcat (tag : String) (aSSA bSSA : String)
    (aShape bShape : List Nat) : String × String × List Nat := Id.run do
  match aShape, bShape with
  | [n, ca, h, w], [_, cb, _, _] =>
    let outShape := [n, ca + cb, h, w]
    let s := s!"    %cc_o{tag} = stablehlo.concatenate {aSSA}, {bSSA}, dim = 1 : ({tensorTy aShape}, {tensorTy bShape}) -> {tensorTy outShape}\n"
    return (s, s!"%cc_o{tag}", outShape)
  | _, _ => return ("    // channelConcat error: expected rank-4 NCHW inputs\n", aSSA, aShape)

/-- VJP of channel-concat: split the gradient `(N, Ca + Cb, H, W)`
    back into `dA : (N, Ca, H, W)` and `dB : (N, Cb, H, W)`. Returns
    `(code, dA_SSA, dB_SSA)`. -/
private def emitChannelSplitGrad (tag : String) (gSSA : String)
    (aShape bShape : List Nat) : String × String × String := Id.run do
  match aShape, bShape with
  | [n, ca, h, w], [_, cb, _, _] =>
    let gShape := [n, ca + cb, h, w]
    let gTy := tensorTy gShape
    let aTy := tensorTy aShape
    let bTy := tensorTy bShape
    let mut s := ""
    s := s ++ s!"    %cs_a{tag} = \"stablehlo.slice\"({gSSA}) " ++ "{" ++
      s!"start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {n}, {ca}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++
      "}" ++ s!" : ({gTy}) -> {aTy}\n"
    s := s ++ s!"    %cs_b{tag} = \"stablehlo.slice\"({gSSA}) " ++ "{" ++
      s!"start_indices = array<i64: 0, {ca}, 0, 0>, limit_indices = array<i64: {n}, {ca + cb}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++
      "}" ++ s!" : ({gTy}) -> {bTy}\n"
    return (s, s!"%cs_a{tag}", s!"%cs_b{tag}")
  | _, _ => return ("    // channelSplitGrad error: expected rank-4 NCHW shapes\n", gSSA, gSSA)

/-- 1-D bilinear-resampling weight matrix `Wy : (outLen × inLen)` for
    integer upsampling factor `scale`, where `outLen = inLen * scale`.
    Half-pixel centers, `align_corners = false` (PyTorch / JAX default):
    output index `i` samples input fractional coordinate
    `(i + 0.5) / scale - 0.5`, with edge values clamped (so the first
    and last output rows replicate the corresponding input rows).
    Each row has at most two nonzero entries. -/
private def bilinearWeights1D (inLen scale : Nat) : Array (Array Float) := Id.run do
  let outLen := inLen * scale
  let mut rows : Array (Array Float) := Array.replicate outLen (Array.replicate inLen 0.0)
  let den : Int := 2 * Int.ofNat scale
  let inLenI : Int := Int.ofNat inLen
  for i in [:outLen] do
    let num : Int := 2 * Int.ofNat i + 1 - Int.ofNat scale
    -- floor(num / den) for positive den, all signs of num
    let y0 : Int := if num ≥ 0 then num / den else -((-num + den - 1) / den)
    let wyNum : Int := num - y0 * den  -- in [0, den)
    let wy : Float := wyNum.toNat.toFloat / den.toNat.toFloat
    let cl := fun (x : Int) =>
      if x < 0 then (0 : Nat)
      else if x ≥ inLenI then inLen - 1
      else x.toNat
    let i0 := cl y0
    let i1 := cl (y0 + 1)
    let mut row := rows[i]!
    row := row.set! i0 (row[i0]! + (1.0 - wy))
    row := row.set! i1 (row[i1]! + wy)
    rows := rows.set! i row
  return rows

/-- Serialize a 2-D float matrix as MLIR `dense<[[...]]>` payload (no
    `dense<>` wrapper, no type — caller adds those). -/
private def floatMatrixToMlirDense (m : Array (Array Float)) : String :=
  let rowStrs := m.toList.map fun row =>
    "[" ++ ",".intercalate (row.toList.map fun v => s!"{v}") ++ "]"
  "[" ++ ",".intercalate rowStrs ++ "]"

/-- Emit bilinear upsample: (b, c, h, w) → (b, c, h*scale, w*scale).
    Factorizes as two `dot_general`s with precomputed weight matrices:
        Y = Wy · X (along H)        →  shape (b, c, w, oH)
        T = transpose Y to (b, c, oH, w)
        Z = T · Wxᵀ (along W)        →  shape (b, c, oH, oW)
    `Wy` and `Wx` are emitted as `stablehlo.constant dense<...>`. The
    matmul factorization keeps the cost at `O(b·c·(h·w·oH + oH·w·oW))`
    rather than naïve gather + 4-corner blend, and reuses the existing
    `dot_general` lowering paths. -/
private def emitBilinearUpsample (pos : Nat) (curSSA : String) (curShape : List Nat)
    (scale : Nat) : String × String × List Nat := Id.run do
  match curShape with
  | [b, c, h, w] =>
    let oH := h * scale
    let oW := w * scale
    let newShape := [b, c, oH, oW]
    let wyStr := floatMatrixToMlirDense (bilinearWeights1D h scale)
    let wxStr := floatMatrixToMlirDense (bilinearWeights1D w scale)
    let inTy := tensorTy curShape
    let interTy := tensorTy [b, c, w, oH]   -- after Wy contraction
    let inter2Ty := tensorTy [b, c, oH, w]  -- after transpose
    let outTy := tensorTy newShape
    let wyTy := s!"tensor<{oH}x{h}xf32>"
    let wxTy := s!"tensor<{oW}x{w}xf32>"
    let mut s := ""
    s := s ++ s!"    %bu_wy{pos} = stablehlo.constant dense<{wyStr}> : {wyTy}\n"
    s := s ++ s!"    %bu_wx{pos} = stablehlo.constant dense<{wxStr}> : {wxTy}\n"
    s := s ++ s!"    %bu_my{pos} = stablehlo.dot_general {curSSA}, %bu_wy{pos}, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : ({inTy}, {wyTy}) -> {interTy}\n"
    s := s ++ s!"    %bu_t{pos} = stablehlo.transpose %bu_my{pos}, dims = [0, 1, 3, 2] : ({interTy}) -> {inter2Ty}\n"
    s := s ++ s!"    %bu_out{pos} = stablehlo.dot_general %bu_t{pos}, %bu_wx{pos}, contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : ({inter2Ty}, {wxTy}) -> {outTy}\n"
    return (s, s!"%bu_out{pos}", newShape)
  | _ => return ("    // bilinearUpsample error: expected rank-4 NCHW\n", curSSA, curShape)

/-- Emit patch embedding forward. Input (B, ic, H, W) → (B, N+1, dim).
    Three params: W (dim, ic, p, p), b (dim,), cls (dim,), pos (N+1, dim).
    Returns (code, outSSA, outShape). -/
private def emitPatchEmbedForward (tag : String) (curSSA : String) (curShape : List Nat)
    (ic dim pSize nP : Nat) (pWSSA pBSSA pClsSSA pPosSSA : String)
    : String × String × List Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := h / pSize
    let oW := w / pSize
    let nT := nP + 1
    let convOut := [b, dim, oH, oW]
    let patchFlat := [b, dim, nP]
    let patchT := [b, nP, dim]
    let outShape := [b, nT, dim]
    let mut s := ""
    -- Conv stride=pSize, no padding
    s := s ++ s!"    %pe_cv{tag} = \"stablehlo.convolution\"({curSSA}, {pWSSA}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {pSize}, {pSize}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [dim, ic, pSize, pSize]}) -> {tensorTy convOut}\n"
    -- Add bias
    s := s ++ s!"    %pe_bb{tag} = stablehlo.broadcast_in_dim {pBSSA}, dims = [1] : ({tensorTy [dim]}) -> {tensorTy convOut}\n"
    s := s ++ s!"    %pe_cvb{tag} = stablehlo.add %pe_cv{tag}, %pe_bb{tag} : {tensorTy convOut}\n"
    -- Reshape (B, dim, oH, oW) → (B, dim, nP)
    s := s ++ s!"    %pe_rs{tag} = stablehlo.reshape %pe_cvb{tag} : ({tensorTy convOut}) -> {tensorTy patchFlat}\n"
    -- Transpose (B, dim, nP) → (B, nP, dim)
    s := s ++ s!"    %pe_tr{tag} = stablehlo.transpose %pe_rs{tag}, dims = [0, 2, 1] : ({tensorTy patchFlat}) -> {tensorTy patchT}\n"
    -- CLS token: broadcast (dim,) → (B, 1, dim)
    s := s ++ s!"    %pe_cls_bc{tag} = stablehlo.broadcast_in_dim {pClsSSA}, dims = [2] : ({tensorTy [dim]}) -> {tensorTy [b, 1, dim]}\n"
    -- Concat along axis=1
    s := s ++ s!"    %pe_cc{tag} = stablehlo.concatenate %pe_cls_bc{tag}, %pe_tr{tag}, dim = 1 : ({tensorTy [b, 1, dim]}, {tensorTy patchT}) -> {tensorTy outShape}\n"
    -- Positional embedding: broadcast (nT, dim) → (B, nT, dim), add
    s := s ++ s!"    %pe_pos_bc{tag} = stablehlo.broadcast_in_dim {pPosSSA}, dims = [1, 2] : ({tensorTy [nT, dim]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %pe_out{tag} = stablehlo.add %pe_cc{tag}, %pe_pos_bc{tag} : {tensorTy outShape}\n"
    return (s, s!"%pe_out{tag}", outShape)
  | _ => return ("    // patchEmbed error\n", curSSA, curShape)

/-- Emit LayerNorm forward for a (B, N, D) input, normalizing across D.
    Returns (code, outSSA, saved records: normSSA, istdSSA, meanSSA).
    gamma, beta are (D,). -/
private def emitLayerNormForward (tag : String) (xSSA : String) (shape : List Nat)
    (gammaSSA betaSSA : String)
    : String × String × String × String × String := Id.run do
  match shape with
  | [b, n, d] =>
    let ty := tensorTy shape
    let bnTy := tensorTy [b, n]
    let dTy := tensorTy [d]
    let mut s := ""
    -- sum over d
    s := s ++ s!"    %ln_zf{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %ln_sum{tag} = stablehlo.reduce({xSSA} init: %ln_zf{tag}) applies stablehlo.add across dimensions = [2]\n"
    s := s ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
    s := s ++ s!"    %ln_Nc{tag} = stablehlo.constant dense<{d.toFloat}> : {bnTy}\n"
    s := s ++ s!"    %ln_mean{tag} = stablehlo.divide %ln_sum{tag}, %ln_Nc{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_mean_bc{tag} = stablehlo.broadcast_in_dim %ln_mean{tag}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
    s := s ++ s!"    %ln_diff{tag} = stablehlo.subtract {xSSA}, %ln_mean_bc{tag} : {ty}\n"
    s := s ++ s!"    %ln_sq{tag} = stablehlo.multiply %ln_diff{tag}, %ln_diff{tag} : {ty}\n"
    s := s ++ s!"    %ln_vsum{tag} = stablehlo.reduce(%ln_sq{tag} init: %ln_zf{tag}) applies stablehlo.add across dimensions = [2]\n"
    s := s ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
    s := s ++ s!"    %ln_var{tag} = stablehlo.divide %ln_vsum{tag}, %ln_Nc{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_eps{tag} = stablehlo.constant dense<1.0e-5> : {bnTy}\n"
    s := s ++ s!"    %ln_ve{tag} = stablehlo.add %ln_var{tag}, %ln_eps{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_istd{tag} = stablehlo.rsqrt %ln_ve{tag} : {bnTy}\n"
    s := s ++ s!"    %ln_istd_bc{tag} = stablehlo.broadcast_in_dim %ln_istd{tag}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
    s := s ++ s!"    %ln_norm{tag} = stablehlo.multiply %ln_diff{tag}, %ln_istd_bc{tag} : {ty}\n"
    s := s ++ s!"    %ln_g_bc{tag} = stablehlo.broadcast_in_dim {gammaSSA}, dims = [2] : ({dTy}) -> {ty}\n"
    s := s ++ s!"    %ln_gn{tag} = stablehlo.multiply %ln_norm{tag}, %ln_g_bc{tag} : {ty}\n"
    s := s ++ s!"    %ln_b_bc{tag} = stablehlo.broadcast_in_dim {betaSSA}, dims = [2] : ({dTy}) -> {ty}\n"
    s := s ++ s!"    %ln_out{tag} = stablehlo.add %ln_gn{tag}, %ln_b_bc{tag} : {ty}\n"
    return (s, s!"%ln_out{tag}", s!"%ln_norm{tag}", s!"%ln_istd{tag}", s!"%ln_mean{tag}")
  | _ => return ("    // layerNorm error\n", xSSA, "", "", "")

/-- LayerNorm over the channel axis of an NCHW tensor `[b, c, h, w]`. For
    each `(b, h, w)` location, normalize across the `c` axis with γ, β
    broadcast from `[c]`.

    **Implementation note**: rather than reducing directly over axis 1 of
    the NCHW tensor (a `[B, C, H, W]` shape with the reduce-axis sandwiched
    in the middle, which IREE's GPU codegen `failed to distribute` on
    shapes like `Bx(HW)xC`), we transpose to NHWC and reshape to `[B, HW, C]`
    so the reduction lands on the inner axis — the easy case. The output
    is transposed back to NCHW so callers see no shape change. Saved
    `norm`/`istd`/`mean` stay in the inner-axis layout (`[B, HW, C]` and
    `[B, HW]`), which is what the matching backward consumes.

    Returns `(code, outSSA, normSSA, istdSSA, meanSSA)`. -/
private def emitLayerNormForwardNCHW (tag : String) (xSSA : String) (shape : List Nat)
    (gammaSSA betaSSA : String)
    : String × String × String × String × String := Id.run do
  match shape with
  | [b, c, h, w] =>
    let nchwTy := tensorTy shape
    let nhwcTy := tensorTy [b, h, w, c]
    let nhcTy  := tensorTy [b, h*w, c]
    let bnTy   := tensorTy [b, h*w]
    let cTy    := tensorTy [c]
    let mut s := ""
    -- Move channels to the innermost axis: [b, c, h, w] → [b, h, w, c] → [b, h*w, c]
    s := s ++ s!"    %lnc_t{tag} = stablehlo.transpose {xSSA}, dims = [0, 2, 3, 1] : ({nchwTy}) -> {nhwcTy}\n"
    s := s ++ s!"    %lnc_x{tag} = stablehlo.reshape %lnc_t{tag} : ({nhwcTy}) -> {nhcTy}\n"
    -- Standard LN over the inner axis (axis 2): reduce, mean, var, normalize, affine.
    s := s ++ s!"    %lnc_zf{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %lnc_sum{tag} = stablehlo.reduce(%lnc_x{tag} init: %lnc_zf{tag}) applies stablehlo.add across dimensions = [2]\n"
    s := s ++ s!"          : ({nhcTy}, tensor<f32>) -> {bnTy}\n"
    s := s ++ s!"    %lnc_Nc{tag} = stablehlo.constant dense<{c.toFloat}> : {bnTy}\n"
    s := s ++ s!"    %lnc_mean{tag} = stablehlo.divide %lnc_sum{tag}, %lnc_Nc{tag} : {bnTy}\n"
    s := s ++ s!"    %lnc_mean_bc{tag} = stablehlo.broadcast_in_dim %lnc_mean{tag}, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
    s := s ++ s!"    %lnc_diff{tag} = stablehlo.subtract %lnc_x{tag}, %lnc_mean_bc{tag} : {nhcTy}\n"
    s := s ++ s!"    %lnc_sq{tag} = stablehlo.multiply %lnc_diff{tag}, %lnc_diff{tag} : {nhcTy}\n"
    s := s ++ s!"    %lnc_vsum{tag} = stablehlo.reduce(%lnc_sq{tag} init: %lnc_zf{tag}) applies stablehlo.add across dimensions = [2]\n"
    s := s ++ s!"          : ({nhcTy}, tensor<f32>) -> {bnTy}\n"
    s := s ++ s!"    %lnc_var{tag} = stablehlo.divide %lnc_vsum{tag}, %lnc_Nc{tag} : {bnTy}\n"
    s := s ++ s!"    %lnc_eps{tag} = stablehlo.constant dense<1.0e-5> : {bnTy}\n"
    s := s ++ s!"    %lnc_ve{tag} = stablehlo.add %lnc_var{tag}, %lnc_eps{tag} : {bnTy}\n"
    s := s ++ s!"    %lnc_istd{tag} = stablehlo.rsqrt %lnc_ve{tag} : {bnTy}\n"
    s := s ++ s!"    %lnc_istd_bc{tag} = stablehlo.broadcast_in_dim %lnc_istd{tag}, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
    s := s ++ s!"    %lnc_norm{tag} = stablehlo.multiply %lnc_diff{tag}, %lnc_istd_bc{tag} : {nhcTy}\n"
    s := s ++ s!"    %lnc_g_bc{tag} = stablehlo.broadcast_in_dim {gammaSSA}, dims = [2] : ({cTy}) -> {nhcTy}\n"
    s := s ++ s!"    %lnc_gn{tag} = stablehlo.multiply %lnc_norm{tag}, %lnc_g_bc{tag} : {nhcTy}\n"
    s := s ++ s!"    %lnc_b_bc{tag} = stablehlo.broadcast_in_dim {betaSSA}, dims = [2] : ({cTy}) -> {nhcTy}\n"
    s := s ++ s!"    %lnc_yi{tag} = stablehlo.add %lnc_gn{tag}, %lnc_b_bc{tag} : {nhcTy}\n"
    -- Reshape + transpose back to NCHW for downstream layers.
    s := s ++ s!"    %lnc_yr{tag} = stablehlo.reshape %lnc_yi{tag} : ({nhcTy}) -> {nhwcTy}\n"
    s := s ++ s!"    %lnc_out{tag} = stablehlo.transpose %lnc_yr{tag}, dims = [0, 3, 1, 2] : ({nhwcTy}) -> {nchwTy}\n"
    return (s, s!"%lnc_out{tag}", s!"%lnc_norm{tag}", s!"%lnc_istd{tag}", s!"%lnc_mean{tag}")
  | _ => return ("    // layerNormNCHW error\n", xSSA, "", "", "")

/-- Standalone BatchNorm over an NCHW tensor — γ/β per channel, mean/var
    aggregated over batch+spatial. Uses the same `%cbn_*` SSA naming as
    `emitConvBn` so the existing BN-stats output machinery (collectBnLayers,
    train-step BN-stat returns, eval-forward fixed-BN params) all just
    work when this primitive is registered with the right pidx.

    Reductions are over `[0, 2, 3]` — the well-supported pattern that
    `convBn` already uses (and that IREE distributes cleanly on ROCm,
    unlike LN-NCHW's channel-axis reduction).

    Returns `(code, outSSA, normSSA, meanBcSSA, istdBcSSA)`. `meanBcSSA`
    and `istdBcSSA` are the broadcast-to-NCHW versions used by the
    matching backward. -/
private def emitBatchNormForwardNCHW (pidx : Nat) (xSSA : String) (shape : List Nat)
    (gammaSSA betaSSA : String) (fixedBN : Bool := false)
    : String × String × String × String × String := Id.run do
  match shape with
  | [b, c, h, w] =>
    let ty := tensorTy shape
    let cTy := tensorTy [c]
    let bcTy := tensorTy [b, c]
    let bnN := b * h * w
    let mut s := ""
    if fixedBN then
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %bn_mean{pidx}, dims = [1] : ({cTy}) -> {ty}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract {xSSA}, %cbn_mean_bc{pidx} : {ty}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {cTy}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %bn_var{pidx}, %cbn_eps{pidx} : {cTy}\n"
    else
      s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce({xSSA} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({ty}, tensor<f32>) -> {bcTy}\n"
      s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({bcTy}, tensor<f32>) -> {cTy}\n"
      s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {cTy}\n"
      s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {cTy}\n"
      s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({cTy}) -> {ty}\n"
      s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract {xSSA}, %cbn_mean_bc{pidx} : {ty}\n"
      s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {ty}\n"
      s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
      s := s ++ s!"          : ({ty}, tensor<f32>) -> {bcTy}\n"
      s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
      s := s ++ s!"          : ({bcTy}, tensor<f32>) -> {cTy}\n"
      s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {cTy}\n"
      s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {cTy}\n"
      s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {cTy}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {cTy}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({cTy}) -> {ty}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {ty}\n"
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim {gammaSSA}, dims = [1] : ({cTy}) -> {ty}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {ty}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim {betaSSA}, dims = [1] : ({cTy}) -> {ty}\n"
    s := s ++ s!"    %cbn_pre{pidx} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {ty}\n"
    return (s, s!"%cbn_pre{pidx}", s!"%cbn_norm{pidx}", s!"%cbn_mean_bc{pidx}", s!"%cbn_istd_bc{pidx}")
  | _ => return ("    // batchNormNCHW error\n", xSSA, "", "", "")

/-- Emit GELU forward (tanh form, exact) for a tensor of given shape.
    Returns (code, outSSA, tanhSSA) where tanhSSA is saved for backward. -/
private def emitGeluForward (tag : String) (xSSA : String) (shape : List Nat)
    : String × String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  s := s ++ s!"    %ge_c05{tag} = stablehlo.constant dense<0.5> : {ty}\n"
  s := s ++ s!"    %ge_ck{tag} = stablehlo.constant dense<0.044715> : {ty}\n"
  s := s ++ s!"    %ge_csqrt{tag} = stablehlo.constant dense<0.7978845608028654> : {ty}\n"
  s := s ++ s!"    %ge_x2{tag} = stablehlo.multiply {xSSA}, {xSSA} : {ty}\n"
  s := s ++ s!"    %ge_x3{tag} = stablehlo.multiply %ge_x2{tag}, {xSSA} : {ty}\n"
  s := s ++ s!"    %ge_kx3{tag} = stablehlo.multiply %ge_x3{tag}, %ge_ck{tag} : {ty}\n"
  s := s ++ s!"    %ge_xp{tag} = stablehlo.add {xSSA}, %ge_kx3{tag} : {ty}\n"
  s := s ++ s!"    %ge_u{tag} = stablehlo.multiply %ge_xp{tag}, %ge_csqrt{tag} : {ty}\n"
  s := s ++ s!"    %ge_t{tag} = stablehlo.tanh %ge_u{tag} : {ty}\n"
  s := s ++ s!"    %ge_one{tag} = stablehlo.constant dense<1.0> : {ty}\n"
  s := s ++ s!"    %ge_1pt{tag} = stablehlo.add %ge_one{tag}, %ge_t{tag} : {ty}\n"
  s := s ++ s!"    %ge_hx{tag} = stablehlo.multiply {xSSA}, %ge_c05{tag} : {ty}\n"
  s := s ++ s!"    %ge_out{tag} = stablehlo.multiply %ge_hx{tag}, %ge_1pt{tag} : {ty}\n"
  return (s, s!"%ge_out{tag}", s!"%ge_t{tag}")

/-- Emit dense layer for rank-3 (B, N, D) input → (B, N, fanOut).
    W shape (D, fanOut), b shape (fanOut,). -/
private def emitDense3D (tag : String) (xSSA : String) (shape : List Nat)
    (wSSA bSSA : String) (fanOut : Nat) : String × String × List Nat := Id.run do
  match shape with
  | [b, n, d] =>
    let outShape := [b, n, fanOut]
    let ty := tensorTy shape
    let outTy := tensorTy outShape
    let mut s := ""
    s := s ++ s!"    %d3_mm{tag} = stablehlo.dot_general {xSSA}, {wSSA},\n"
    s := s ++ "              contracting_dims = [2] x [0],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({ty}, {tensorTy [d, fanOut]}) -> {outTy}\n"
    s := s ++ s!"    %d3_bb{tag} = stablehlo.broadcast_in_dim {bSSA}, dims = [2] : ({tensorTy [fanOut]}) -> {outTy}\n"
    s := s ++ s!"    %d3_out{tag} = stablehlo.add %d3_mm{tag}, %d3_bb{tag} : {outTy}\n"
    return (s, s!"%d3_out{tag}", outShape)
  | _ => return ("    // dense3D error\n", xSSA, shape)

/-- Emit the FlashAttention forward SDPA core: `softmax(Q·Kᵀ/√dh + mask)·V`
    via a tiled online-softmax `stablehlo.while` over ⌈n/bk⌉ key blocks, so the
    only O(n·bk) tensor live per iteration is one score block `[b,h,n,bk]` — the
    full `[b,h,n,n]` matrix is never materialized. Inputs `q/k/v : [b,h,n,dh]`;
    returns `(code, oSSA [b,h,n,dh], lseSSA [b,h,n])`. `lse = m + log(l)` is the
    per-row logsumexp the backward needs. Requires `n % bk == 0`. Validated
    end-to-end against dense attention (jax/demos/flash_attention_ref.py +
    an IREE compile+run probe, ~1e-7 fp32). See planning/flash_attention.md. -/
private def emitFlashAttnSdpa (tag : String) (qSSA kSSA vSSA : String)
    (b heads n dh bk : Nat) (causal : Bool) : String × String × String := Id.run do
  let hTy := tensorTy [b, heads, n, dh]
  let blkTy := tensorTy [b, heads, bk, dh]
  let scTy := tensorTy [b, heads, n, bk]
  let rowTy := tensorTy [b, heads, n]
  let nkb := n / bk
  let invSqrtDh : Float := 1.0 / Float.sqrt dh.toFloat
  let mut s := ""
  -- loop-invariant constants
  s := s ++ s!"    %fa_c0{tag} = stablehlo.constant dense<0> : tensor<i32>\n"
  s := s ++ s!"    %fa_c1{tag} = stablehlo.constant dense<1> : tensor<i32>\n"
  s := s ++ s!"    %fa_cbk{tag} = stablehlo.constant dense<{bk}> : tensor<i32>\n"
  s := s ++ s!"    %fa_nkb{tag} = stablehlo.constant dense<{nkb}> : tensor<i32>\n"
  s := s ++ s!"    %fa_O0{tag} = stablehlo.constant dense<0.0> : {hTy}\n"
  s := s ++ s!"    %fa_m0{tag} = stablehlo.constant dense<0xFF800000> : {rowTy}\n"
  s := s ++ s!"    %fa_l0{tag} = stablehlo.constant dense<0.0> : {rowTy}\n"
  s := s ++ s!"    %fa_scale{tag} = stablehlo.constant dense<{invSqrtDh}> : {scTy}\n"
  s := s ++ s!"    %fa_ninf{tag} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  s := s ++ s!"    %fa_zf{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
  -- while(k_idx, O, m, l)
  s := s ++ s!"    %fa_r{tag}:4 = stablehlo.while(%fa_k{tag} = %fa_c0{tag}, %fa_O{tag} = %fa_O0{tag}, %fa_mm{tag} = %fa_m0{tag}, %fa_ll{tag} = %fa_l0{tag})\n"
  s := s ++ s!"        : tensor<i32>, {hTy}, {rowTy}, {rowTy}\n"
  s := s ++ s!"      cond " ++ "{\n"
  s := s ++ s!"        %fa_lt{tag} = stablehlo.compare LT, %fa_k{tag}, %fa_nkb{tag} : (tensor<i32>, tensor<i32>) -> tensor<i1>\n"
  s := s ++ s!"        stablehlo.return %fa_lt{tag} : tensor<i1>\n"
  s := s ++ "      } do {\n"
  s := s ++ s!"        %fa_off{tag} = stablehlo.multiply %fa_k{tag}, %fa_cbk{tag} : tensor<i32>\n"
  -- Kj, Vj = dynamic_slice at [0,0,off,0] size [b,heads,bk,dh]
  s := s ++ s!"        %fa_Kj{tag} = \"stablehlo.dynamic_slice\"({kSSA}, %fa_c0{tag}, %fa_c0{tag}, %fa_off{tag}, %fa_c0{tag}) " ++ "{" ++ s!" slice_sizes = array<i64: {b}, {heads}, {bk}, {dh}> " ++ "}" ++ s!" : ({hTy}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {blkTy}\n"
  s := s ++ s!"        %fa_Vj{tag} = \"stablehlo.dynamic_slice\"({vSSA}, %fa_c0{tag}, %fa_c0{tag}, %fa_off{tag}, %fa_c0{tag}) " ++ "{" ++ s!" slice_sizes = array<i64: {b}, {heads}, {bk}, {dh}> " ++ "}" ++ s!" : ({hTy}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {blkTy}\n"
  -- Sij = Q·Kjᵀ · scale : [b,h,n,dh]x[b,h,bk,dh] contract dh -> [b,h,n,bk]
  s := s ++ s!"        %fa_sc{tag} = stablehlo.dot_general {qSSA}, %fa_Kj{tag}, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({hTy}, {blkTy}) -> {scTy}\n"
  s := s ++ s!"        %fa_Sij{tag} = stablehlo.multiply %fa_sc{tag}, %fa_scale{tag} : {scTy}\n"
  let sijName ← if causal then do
    -- mask: keep where qidx >= off + kcol, else -inf. qidx = iota over query (dim2),
    -- kcol = off + iota over key block (dim3).
    let mut c := ""
    c := c ++ s!"        %fa_qi{tag} = stablehlo.iota dim = 2 : tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fa_ki0{tag} = stablehlo.iota dim = 3 : tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fa_offb{tag} = stablehlo.broadcast_in_dim %fa_off{tag}, dims = [] : (tensor<i32>) -> tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fa_ki{tag} = stablehlo.add %fa_ki0{tag}, %fa_offb{tag} : tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fa_cmp{tag} = stablehlo.compare GE, %fa_qi{tag}, %fa_ki{tag} : (tensor<{b}x{heads}x{n}x{bk}xi32>, tensor<{b}x{heads}x{n}x{bk}xi32>) -> tensor<{b}x{heads}x{n}x{bk}xi1>\n"
    c := c ++ s!"        %fa_ninfm{tag} = stablehlo.constant dense<0xFF800000> : {scTy}\n"
    c := c ++ s!"        %fa_Sm{tag} = stablehlo.select %fa_cmp{tag}, %fa_Sij{tag}, %fa_ninfm{tag} : (tensor<{b}x{heads}x{n}x{bk}xi1>, {scTy}, {scTy}) -> {scTy}\n"
    s := s ++ c
    pure s!"%fa_Sm{tag}"
  else pure s!"%fa_Sij{tag}"
  -- mij = rowmax over key axis (3); m_new = max(m, mij)
  s := s ++ s!"        %fa_mij{tag} = stablehlo.reduce({sijName} init: %fa_ninf{tag}) applies stablehlo.maximum across dimensions = [3]\n"
  s := s ++ s!"            : ({scTy}, tensor<f32>) -> {rowTy}\n"
  s := s ++ s!"        %fa_mnew{tag} = stablehlo.maximum %fa_mm{tag}, %fa_mij{tag} : {rowTy}\n"
  -- alpha = exp(m - m_new)  (guard: m starts -inf; -inf - -inf handled since m_new≥m so diff≤0, and -inf-(-inf) → nan, but first block m=-inf & mij finite ⇒ m_new finite ⇒ m-m_new = -inf ⇒ exp = 0, correct)
  s := s ++ s!"        %fa_mdiff{tag} = stablehlo.subtract %fa_mm{tag}, %fa_mnew{tag} : {rowTy}\n"
  s := s ++ s!"        %fa_alpha{tag} = stablehlo.exponential %fa_mdiff{tag} : {rowTy}\n"
  -- Pij = exp(Sij - m_new)
  s := s ++ s!"        %fa_mnewb{tag} = stablehlo.broadcast_in_dim %fa_mnew{tag}, dims = [0, 1, 2] : ({rowTy}) -> {scTy}\n"
  s := s ++ s!"        %fa_shift{tag} = stablehlo.subtract {sijName}, %fa_mnewb{tag} : {scTy}\n"
  s := s ++ s!"        %fa_Pij{tag} = stablehlo.exponential %fa_shift{tag} : {scTy}\n"
  -- l = alpha*l + rowsum(Pij)
  s := s ++ s!"        %fa_psum{tag} = stablehlo.reduce(%fa_Pij{tag} init: %fa_zf{tag}) applies stablehlo.add across dimensions = [3]\n"
  s := s ++ s!"            : ({scTy}, tensor<f32>) -> {rowTy}\n"
  s := s ++ s!"        %fa_al{tag} = stablehlo.multiply %fa_alpha{tag}, %fa_ll{tag} : {rowTy}\n"
  s := s ++ s!"        %fa_lnew{tag} = stablehlo.add %fa_al{tag}, %fa_psum{tag} : {rowTy}\n"
  -- O = alpha*O + Pij @ Vj
  s := s ++ s!"        %fa_alphab{tag} = stablehlo.broadcast_in_dim %fa_alpha{tag}, dims = [0, 1, 2] : ({rowTy}) -> {hTy}\n"
  s := s ++ s!"        %fa_aO{tag} = stablehlo.multiply %fa_alphab{tag}, %fa_O{tag} : {hTy}\n"
  s := s ++ s!"        %fa_pv{tag} = stablehlo.dot_general %fa_Pij{tag}, %fa_Vj{tag}, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : ({scTy}, {blkTy}) -> {hTy}\n"
  s := s ++ s!"        %fa_Onew{tag} = stablehlo.add %fa_aO{tag}, %fa_pv{tag} : {hTy}\n"
  s := s ++ s!"        %fa_knext{tag} = stablehlo.add %fa_k{tag}, %fa_c1{tag} : tensor<i32>\n"
  s := s ++ s!"        stablehlo.return %fa_knext{tag}, %fa_Onew{tag}, %fa_mnew{tag}, %fa_lnew{tag} : tensor<i32>, {hTy}, {rowTy}, {rowTy}\n"
  s := s ++ "      }\n"
  -- O / l  ;  lse = m + log(l)
  s := s ++ s!"    %fa_lbc{tag} = stablehlo.broadcast_in_dim %fa_r{tag}#3, dims = [0, 1, 2] : ({rowTy}) -> {hTy}\n"
  s := s ++ s!"    %fa_out{tag} = stablehlo.divide %fa_r{tag}#1, %fa_lbc{tag} : {hTy}\n"
  s := s ++ s!"    %fa_logl{tag} = stablehlo.log %fa_r{tag}#3 : {rowTy}\n"
  s := s ++ s!"    %fa_lse{tag} = stablehlo.add %fa_r{tag}#2, %fa_logl{tag} : {rowTy}\n"
  return (s, s!"%fa_out{tag}", s!"%fa_lse{tag}")

/-- Emit the FlashAttention backward: given `q/k/v [b,h,n,dh]`, the upstream
    grad `dO [b,h,n,dh]`, the forward output `O`, and the per-row logsumexp
    `lse [b,h,n]`, produce `(dQ, dK, dV)` — all `[b,h,n,dh]` — via a
    recompute `stablehlo.while` over key blocks. No `[b,h,n,n]` matrix is stored:
    each block recomputes `Pij = exp(Sij − lse)` and uses `D = rowsum(dO⊙O)` as
    the softmax-Jacobian term. dQ accumulates across ALL blocks; dK/dV are written
    once per (disjoint) key block via `dynamic_update_slice`. Matches
    `flash_attention_ref.py:flash_backward` (validated ~1e-15). -/
private def emitFlashAttnBwd (tag : String) (qSSA kSSA vSSA doSSA oSSA lseSSA : String)
    (b heads n dh bk : Nat) (causal : Bool) : String × String × String × String := Id.run do
  let hTy := tensorTy [b, heads, n, dh]
  let blkTy := tensorTy [b, heads, bk, dh]
  let scTy := tensorTy [b, heads, n, bk]
  let rowTy := tensorTy [b, heads, n]
  let i32 := "tensor<i32>"
  let nkb := n / bk
  let invSqrtDh : Float := 1.0 / Float.sqrt dh.toFloat
  let mut s := ""
  -- D = rowsum(dO ⊙ O)  [b,h,n]
  s := s ++ s!"    %fb_zf{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
  s := s ++ s!"    %fb_doo{tag} = stablehlo.multiply {doSSA}, {oSSA} : {hTy}\n"
  s := s ++ s!"    %fb_D{tag} = stablehlo.reduce(%fb_doo{tag} init: %fb_zf{tag}) applies stablehlo.add across dimensions = [3]\n"
  s := s ++ s!"        : ({hTy}, tensor<f32>) -> {rowTy}\n"
  -- constants + zero-init dQ/dK/dV
  s := s ++ s!"    %fb_c0{tag} = stablehlo.constant dense<0> : {i32}\n"
  s := s ++ s!"    %fb_c1{tag} = stablehlo.constant dense<1> : {i32}\n"
  s := s ++ s!"    %fb_cbk{tag} = stablehlo.constant dense<{bk}> : {i32}\n"
  s := s ++ s!"    %fb_nkb{tag} = stablehlo.constant dense<{nkb}> : {i32}\n"
  s := s ++ s!"    %fb_z{tag} = stablehlo.constant dense<0.0> : {hTy}\n"
  s := s ++ s!"    %fb_scale{tag} = stablehlo.constant dense<{invSqrtDh}> : {scTy}\n"
  s := s ++ s!"    %fb_ninf{tag} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  s := s ++ s!"    %fb_lseb{tag} = stablehlo.broadcast_in_dim {lseSSA}, dims = [0, 1, 2] : ({rowTy}) -> {scTy}\n"
  s := s ++ s!"    %fb_Db{tag} = stablehlo.broadcast_in_dim %fb_D{tag}, dims = [0, 1, 2] : ({rowTy}) -> {scTy}\n"
  -- while(k, dQ, dK, dV)
  s := s ++ s!"    %fb_r{tag}:4 = stablehlo.while(%fb_k{tag} = %fb_c0{tag}, %fb_dQ{tag} = %fb_z{tag}, %fb_dK{tag} = %fb_z{tag}, %fb_dV{tag} = %fb_z{tag})\n"
  s := s ++ s!"        : {i32}, {hTy}, {hTy}, {hTy}\n"
  s := s ++ s!"      cond " ++ "{\n"
  s := s ++ s!"        %fb_lt{tag} = stablehlo.compare LT, %fb_k{tag}, %fb_nkb{tag} : ({i32}, {i32}) -> tensor<i1>\n"
  s := s ++ s!"        stablehlo.return %fb_lt{tag} : tensor<i1>\n"
  s := s ++ "      } do {\n"
  s := s ++ s!"        %fb_off{tag} = stablehlo.multiply %fb_k{tag}, %fb_cbk{tag} : {i32}\n"
  s := s ++ s!"        %fb_Kj{tag} = \"stablehlo.dynamic_slice\"({kSSA}, %fb_c0{tag}, %fb_c0{tag}, %fb_off{tag}, %fb_c0{tag}) " ++ "{" ++ s!" slice_sizes = array<i64: {b}, {heads}, {bk}, {dh}> " ++ "}" ++ s!" : ({hTy}, {i32}, {i32}, {i32}, {i32}) -> {blkTy}\n"
  s := s ++ s!"        %fb_Vj{tag} = \"stablehlo.dynamic_slice\"({vSSA}, %fb_c0{tag}, %fb_c0{tag}, %fb_off{tag}, %fb_c0{tag}) " ++ "{" ++ s!" slice_sizes = array<i64: {b}, {heads}, {bk}, {dh}> " ++ "}" ++ s!" : ({hTy}, {i32}, {i32}, {i32}, {i32}) -> {blkTy}\n"
  -- Sij = Q·Kjᵀ·scale
  s := s ++ s!"        %fb_sc{tag} = stablehlo.dot_general {qSSA}, %fb_Kj{tag}, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({hTy}, {blkTy}) -> {scTy}\n"
  s := s ++ s!"        %fb_Sij{tag} = stablehlo.multiply %fb_sc{tag}, %fb_scale{tag} : {scTy}\n"
  let sijName ← if causal then do
    let mut c := ""
    c := c ++ s!"        %fb_qi{tag} = stablehlo.iota dim = 2 : tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fb_ki0{tag} = stablehlo.iota dim = 3 : tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fb_offb{tag} = stablehlo.broadcast_in_dim %fb_off{tag}, dims = [] : ({i32}) -> tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fb_ki{tag} = stablehlo.add %fb_ki0{tag}, %fb_offb{tag} : tensor<{b}x{heads}x{n}x{bk}xi32>\n"
    c := c ++ s!"        %fb_cmp{tag} = stablehlo.compare GE, %fb_qi{tag}, %fb_ki{tag} : (tensor<{b}x{heads}x{n}x{bk}xi32>, tensor<{b}x{heads}x{n}x{bk}xi32>) -> tensor<{b}x{heads}x{n}x{bk}xi1>\n"
    c := c ++ s!"        %fb_ninfm{tag} = stablehlo.constant dense<0xFF800000> : {scTy}\n"
    c := c ++ s!"        %fb_Sm{tag} = stablehlo.select %fb_cmp{tag}, %fb_Sij{tag}, %fb_ninfm{tag} : (tensor<{b}x{heads}x{n}x{bk}xi1>, {scTy}, {scTy}) -> {scTy}\n"
    s := s ++ c
    pure s!"%fb_Sm{tag}"
  else pure s!"%fb_Sij{tag}"
  -- Pij = exp(Sij - lse)
  s := s ++ s!"        %fb_shift{tag} = stablehlo.subtract {sijName}, %fb_lseb{tag} : {scTy}\n"
  s := s ++ s!"        %fb_Pij{tag} = stablehlo.exponential %fb_shift{tag} : {scTy}\n"
  -- dV_block = Pijᵀ·dO (contract query n) -> [b,h,bk,dh]; write into dV at off
  s := s ++ s!"        %fb_dVblk{tag} = stablehlo.dot_general %fb_Pij{tag}, {doSSA}, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : ({scTy}, {hTy}) -> {blkTy}\n"
  s := s ++ s!"        %fb_dVnew{tag} = \"stablehlo.dynamic_update_slice\"(%fb_dV{tag}, %fb_dVblk{tag}, %fb_c0{tag}, %fb_c0{tag}, %fb_off{tag}, %fb_c0{tag}) : ({hTy}, {blkTy}, {i32}, {i32}, {i32}, {i32}) -> {hTy}\n"
  -- dPij = dO·Vjᵀ (contract dh) -> [b,h,n,bk]
  s := s ++ s!"        %fb_dP{tag} = stablehlo.dot_general {doSSA}, %fb_Vj{tag}, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({hTy}, {blkTy}) -> {scTy}\n"
  -- dSij = Pij ⊙ (dPij - D) · scale
  s := s ++ s!"        %fb_dPmD{tag} = stablehlo.subtract %fb_dP{tag}, %fb_Db{tag} : {scTy}\n"
  s := s ++ s!"        %fb_PdP{tag} = stablehlo.multiply %fb_Pij{tag}, %fb_dPmD{tag} : {scTy}\n"
  s := s ++ s!"        %fb_dSij{tag} = stablehlo.multiply %fb_PdP{tag}, %fb_scale{tag} : {scTy}\n"
  -- dQ += dSij·Kj (contract bk) -> [b,h,n,dh]
  s := s ++ s!"        %fb_dQadd{tag} = stablehlo.dot_general %fb_dSij{tag}, %fb_Kj{tag}, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : ({scTy}, {blkTy}) -> {hTy}\n"
  s := s ++ s!"        %fb_dQnew{tag} = stablehlo.add %fb_dQ{tag}, %fb_dQadd{tag} : {hTy}\n"
  -- dK_block = dSijᵀ·Q (contract query n) -> [b,h,bk,dh]; write into dK at off
  s := s ++ s!"        %fb_dKblk{tag} = stablehlo.dot_general %fb_dSij{tag}, {qSSA}, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : ({scTy}, {hTy}) -> {blkTy}\n"
  s := s ++ s!"        %fb_dKnew{tag} = \"stablehlo.dynamic_update_slice\"(%fb_dK{tag}, %fb_dKblk{tag}, %fb_c0{tag}, %fb_c0{tag}, %fb_off{tag}, %fb_c0{tag}) : ({hTy}, {blkTy}, {i32}, {i32}, {i32}, {i32}) -> {hTy}\n"
  s := s ++ s!"        %fb_knext{tag} = stablehlo.add %fb_k{tag}, %fb_c1{tag} : {i32}\n"
  s := s ++ s!"        stablehlo.return %fb_knext{tag}, %fb_dQnew{tag}, %fb_dKnew{tag}, %fb_dVnew{tag} : {i32}, {hTy}, {hTy}, {hTy}\n"
  s := s ++ "      }\n"
  return (s, s!"%fb_r{tag}#1", s!"%fb_r{tag}#2", s!"%fb_r{tag}#3")

/-- Block size for FlashAttention: a divisor of `n` ≤ 128 (fewer blocks for
    short sequences, real memory win for long ones). -/
private def flashBlock (n : Nat) : Nat :=
  if n % 128 == 0 then 128 else if n % 64 == 0 then 64 else n

/-- Emit the RoPE cos/sin tables `[n, dh]` in-graph (scale-independent — no
    giant constant): `θ_i = base^(-2i/dh)`, `freqs[m,i] = m·θ_i`, tiled to dh.
    Returns `(code, cosSSA, sinSSA)`. See jax/demos/rope_ref.py. -/
private def emitRopeTables (tag : String) (n dh : Nat) (base : Float := 10000.0)
    : String × String × String := Id.run do
  let half := dh / 2
  let thTy := tensorTy [half]
  let thfTy := tensorTy [n, half]
  let dtTy := tensorTy [n, dh]
  -- θ_i = base^(-2i/dh) = exp( (-2i/dh)·ln base )
  let lnBase := Float.log base
  let thetas := (List.range half).map (fun i =>
    toString (Float.exp (lnBase * (-2.0 * i.toFloat / dh.toFloat))))
  let mut s := ""
  s := s ++ s!"    %rope_th{tag} = stablehlo.constant dense<[{String.intercalate ", " thetas}]> : {thTy}\n"
  s := s ++ s!"    %rope_ipos{tag} = stablehlo.iota dim = 0 : tensor<{n}xi32>\n"
  s := s ++ s!"    %rope_pos{tag} = stablehlo.convert %rope_ipos{tag} : (tensor<{n}xi32>) -> {tensorTy [n]}\n"
  s := s ++ s!"    %rope_posb{tag} = stablehlo.broadcast_in_dim %rope_pos{tag}, dims = [0] : ({tensorTy [n]}) -> {thfTy}\n"
  s := s ++ s!"    %rope_thb{tag} = stablehlo.broadcast_in_dim %rope_th{tag}, dims = [1] : ({thTy}) -> {thfTy}\n"
  s := s ++ s!"    %rope_fr{tag} = stablehlo.multiply %rope_posb{tag}, %rope_thb{tag} : {thfTy}\n"
  s := s ++ s!"    %rope_ch{tag} = stablehlo.cosine %rope_fr{tag} : {thfTy}\n"
  s := s ++ s!"    %rope_sh{tag} = stablehlo.sine %rope_fr{tag} : {thfTy}\n"
  s := s ++ s!"    %rope_cos{tag} = stablehlo.concatenate %rope_ch{tag}, %rope_ch{tag}, dim = 1 : ({thfTy}, {thfTy}) -> {dtTy}\n"
  s := s ++ s!"    %rope_sin{tag} = stablehlo.concatenate %rope_sh{tag}, %rope_sh{tag}, dim = 1 : ({thfTy}, {thfTy}) -> {dtTy}\n"
  return (s, s!"%rope_cos{tag}", s!"%rope_sin{tag}")

/-- Apply RoPE to `xSSA : [b,heads,n,dh]` given the `[n,dh]` cos/sin tables:
    `x⊙cos + rotate_half(x)⊙sin`, `rotate_half([a,b])=[-b,a]`. With `negSin`
    (the backward/VJP), sin is negated (rotation by −θ). Returns `(code, outSSA)`. -/
private def emitRopeApply (tag : String) (xSSA cosSSA sinSSA : String)
    (b heads n dh : Nat) (negSin : Bool) : String × String := Id.run do
  let half := dh / 2
  let hTy := tensorTy [b, heads, n, dh]
  let halfTy := tensorTy [b, heads, n, half]
  let mut s := ""
  -- x1 = x[..,:half], x2 = x[..,half:]; rot = concat(-x2, x1)
  s := s ++ s!"    %rope_x1{tag} = \"stablehlo.slice\"({xSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {b}, {heads}, {n}, {half}>, strides = array<i64: 1, 1, 1, 1>" ++ "}" ++ s!" : ({hTy}) -> {halfTy}\n"
  s := s ++ s!"    %rope_x2{tag} = \"stablehlo.slice\"({xSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0, {half}>, limit_indices = array<i64: {b}, {heads}, {n}, {dh}>, strides = array<i64: 1, 1, 1, 1>" ++ "}" ++ s!" : ({hTy}) -> {halfTy}\n"
  s := s ++ s!"    %rope_nx2{tag} = stablehlo.negate %rope_x2{tag} : {halfTy}\n"
  s := s ++ s!"    %rope_rot{tag} = stablehlo.concatenate %rope_nx2{tag}, %rope_x1{tag}, dim = 3 : ({halfTy}, {halfTy}) -> {hTy}\n"
  s := s ++ s!"    %rope_cosb{tag} = stablehlo.broadcast_in_dim {cosSSA}, dims = [2, 3] : ({tensorTy [n, dh]}) -> {hTy}\n"
  s := s ++ s!"    %rope_sinb0{tag} = stablehlo.broadcast_in_dim {sinSSA}, dims = [2, 3] : ({tensorTy [n, dh]}) -> {hTy}\n"
  let sinbName ← if negSin then do
    s := s ++ s!"    %rope_sinb{tag} = stablehlo.negate %rope_sinb0{tag} : {hTy}\n"
    pure s!"%rope_sinb{tag}"
  else pure s!"%rope_sinb0{tag}"
  s := s ++ s!"    %rope_xc{tag} = stablehlo.multiply {xSSA}, %rope_cosb{tag} : {hTy}\n"
  s := s ++ s!"    %rope_rs{tag} = stablehlo.multiply %rope_rot{tag}, {sinbName} : {hTy}\n"
  s := s ++ s!"    %rope_out{tag} = stablehlo.add %rope_xc{tag}, %rope_rs{tag} : {hTy}\n"
  return (s, s!"%rope_out{tag}")

/-- Emit multi-head self-attention forward.
    Input x (B, N, D). Params: Wq, bq, Wk, bk, Wv, bv, Wo, bo.
    Returns (code, outSSA, Q, K, V, softmax-or-Lse, preProj, sdpaOut SSAs). -/
private def emitMHSAForward (tag : String) (xSSA : String) (shape : List Nat)
    (heads : Nat) (wqSSA bqSSA wkSSA bkSSA wvSSA bvSSA woSSA boSSA : String)
    (causalMask : Bool := false) (flashAttn : Bool := false) (rope : Bool := false)
    : String × String × String × String × String × String × String × String := Id.run do
  match shape with
  | [b, n, d] =>
    let dh := d / heads
    let ty := tensorTy shape
    let heTy := tensorTy [b, n, heads, dh]
    let hTy := tensorTy [b, heads, n, dh]
    let sTy := tensorTy [b, heads, n, n]
    let mut s := ""
    -- Q, K, V projections (B, N, D) → (B, N, D)
    let (qCode, qSSA, _) := emitDense3D s!"{tag}_q" xSSA shape wqSSA bqSSA d
    s := s ++ qCode
    let (kCode, kSSA, _) := emitDense3D s!"{tag}_k" xSSA shape wkSSA bkSSA d
    s := s ++ kCode
    let (vCode, vSSA, _) := emitDense3D s!"{tag}_v" xSSA shape wvSSA bvSSA d
    s := s ++ vCode
    -- Reshape to (B, N, heads, dh), transpose to (B, heads, N, dh)
    s := s ++ s!"    %mh_qr{tag} = stablehlo.reshape {qSSA} : ({ty}) -> {heTy}\n"
    s := s ++ s!"    %mh_q{tag} = stablehlo.transpose %mh_qr{tag}, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
    s := s ++ s!"    %mh_kr{tag} = stablehlo.reshape {kSSA} : ({ty}) -> {heTy}\n"
    s := s ++ s!"    %mh_k{tag} = stablehlo.transpose %mh_kr{tag}, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
    s := s ++ s!"    %mh_vr{tag} = stablehlo.reshape {vSSA} : ({ty}) -> {heTy}\n"
    s := s ++ s!"    %mh_v{tag} = stablehlo.transpose %mh_vr{tag}, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
    -- RoPE: rotate Q,K per position (relative-position attention). V is not
    -- rotated. The roped Q,K become the "Q,K" the SDPA + its backward see, so
    -- dq/dk come back w.r.t. the roped vectors and get un-roped in the block bwd.
    let (ropeCode, qAttn, kAttn) := Id.run do
      if !rope then return ("", s!"%mh_q{tag}", s!"%mh_k{tag}")
      let (tcode, cosSSA, sinSSA) := emitRopeTables s!"{tag}q" n dh
      let (qc, qr) := emitRopeApply s!"{tag}q" s!"%mh_q{tag}" cosSSA sinSSA b heads n dh false
      let (kc, kr) := emitRopeApply s!"{tag}k" s!"%mh_k{tag}" cosSSA sinSSA b heads n dh false
      return (tcode ++ qc ++ kc, qr, kr)
    s := s ++ ropeCode
    -- SDPA core. `avSSA` = attention output [b,heads,n,dh]; `smLseSSA` = the
    -- saved backward state (dense softmax probs [b,h,n,n], or flash logsumexp
    -- [b,h,n]). Flash never materializes the [b,h,n,n] matrix.
    let bhnTy := tensorTy [b, heads, n]
    let (sdpaCode, avSSA, smLseSSA) := Id.run do
      let mut s2 := ""
      if flashAttn then
        let (fcode, foSSA, flseSSA) := emitFlashAttnSdpa s!"{tag}fa" qAttn kAttn s!"%mh_v{tag}" b heads n dh (flashBlock n) causalMask
        s2 := s2 ++ fcode
        return (s2, foSSA, flseSSA)
      -- Scores = Q @ K^T
      s2 := s2 ++ s!"    %mh_sc{tag} = stablehlo.dot_general {qAttn}, {kAttn},\n"
      s2 := s2 ++ "              batching_dims = [0, 1] x [0, 1],\n"
      s2 := s2 ++ "              contracting_dims = [3] x [3],\n"
      s2 := s2 ++ "              precision = [DEFAULT, DEFAULT]\n"
      s2 := s2 ++ s!"            : ({hTy}, {hTy}) -> {sTy}\n"
      let invSqrtDh : Float := 1.0 / Float.sqrt dh.toFloat
      s2 := s2 ++ s!"    %mh_scale{tag} = stablehlo.constant dense<{invSqrtDh}> : {sTy}\n"
      s2 := s2 ++ s!"    %mh_ss_pre{tag} = stablehlo.multiply %mh_sc{tag}, %mh_scale{tag} : {sTy}\n"
      if causalMask then
        let nnTy := tensorTy [n, n]
        s2 := s2 ++ s!"    %mh_iotaR{tag} = stablehlo.iota dim = 0 : {nnTy}\n"
        s2 := s2 ++ s!"    %mh_iotaC{tag} = stablehlo.iota dim = 1 : {nnTy}\n"
        s2 := s2 ++ s!"    %mh_cmp{tag} = stablehlo.compare GE, %mh_iotaR{tag}, %mh_iotaC{tag}, FLOAT : ({nnTy}, {nnTy}) -> tensor<{n}x{n}xi1>\n"
        s2 := s2 ++ s!"    %mh_zmask{tag} = stablehlo.constant dense<0.0> : {nnTy}\n"
        s2 := s2 ++ s!"    %mh_ninfmask{tag} = stablehlo.constant dense<0xFF800000> : {nnTy}\n"
        s2 := s2 ++ s!"    %mh_mask{tag} = stablehlo.select %mh_cmp{tag}, %mh_zmask{tag}, %mh_ninfmask{tag} : (tensor<{n}x{n}xi1>, {nnTy}, {nnTy}) -> {nnTy}\n"
        s2 := s2 ++ s!"    %mh_mask_bc{tag} = stablehlo.broadcast_in_dim %mh_mask{tag}, dims = [2, 3] : ({nnTy}) -> {sTy}\n"
        s2 := s2 ++ s!"    %mh_ss{tag} = stablehlo.add %mh_ss_pre{tag}, %mh_mask_bc{tag} : {sTy}\n"
      else
        s2 := s2 ++ s!"    %mh_ss_zc{tag} = stablehlo.constant dense<0.0> : {sTy}\n"
        s2 := s2 ++ s!"    %mh_ss{tag} = stablehlo.add %mh_ss_pre{tag}, %mh_ss_zc{tag} : {sTy}\n"
      s2 := s2 ++ s!"    %mh_neginf{tag} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
      s2 := s2 ++ s!"    %mh_max{tag} = stablehlo.reduce(%mh_ss{tag} init: %mh_neginf{tag}) applies stablehlo.maximum across dimensions = [3]\n"
      s2 := s2 ++ s!"          : ({sTy}, tensor<f32>) -> {bhnTy}\n"
      s2 := s2 ++ s!"    %mh_max_bc{tag} = stablehlo.broadcast_in_dim %mh_max{tag}, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
      s2 := s2 ++ s!"    %mh_shift{tag} = stablehlo.subtract %mh_ss{tag}, %mh_max_bc{tag} : {sTy}\n"
      s2 := s2 ++ s!"    %mh_exp{tag} = stablehlo.exponential %mh_shift{tag} : {sTy}\n"
      s2 := s2 ++ s!"    %mh_zf{tag} = stablehlo.constant dense<0.0> : tensor<f32>\n"
      s2 := s2 ++ s!"    %mh_sumE{tag} = stablehlo.reduce(%mh_exp{tag} init: %mh_zf{tag}) applies stablehlo.add across dimensions = [3]\n"
      s2 := s2 ++ s!"          : ({sTy}, tensor<f32>) -> {bhnTy}\n"
      s2 := s2 ++ s!"    %mh_sumE_bc{tag} = stablehlo.broadcast_in_dim %mh_sumE{tag}, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
      s2 := s2 ++ s!"    %mh_sm{tag} = stablehlo.divide %mh_exp{tag}, %mh_sumE_bc{tag} : {sTy}\n"
      s2 := s2 ++ s!"    %mh_av{tag} = stablehlo.dot_general %mh_sm{tag}, %mh_v{tag},\n"
      s2 := s2 ++ "              batching_dims = [0, 1] x [0, 1],\n"
      s2 := s2 ++ "              contracting_dims = [3] x [2],\n"
      s2 := s2 ++ "              precision = [DEFAULT, DEFAULT]\n"
      s2 := s2 ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
      return (s2, s!"%mh_av{tag}", s!"%mh_sm{tag}")
    s := s ++ sdpaCode
    -- Transpose back (B, heads, N, dh) → (B, N, heads, dh), reshape → (B, N, D)
    s := s ++ s!"    %mh_avT{tag} = stablehlo.transpose {avSSA}, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
    s := s ++ s!"    %mh_pp{tag} = stablehlo.reshape %mh_avT{tag} : ({heTy}) -> {ty}\n"
    -- Output projection
    let (oCode, oSSA, _) := emitDense3D s!"{tag}_o" s!"%mh_pp{tag}" shape woSSA boSSA d
    s := s ++ oCode
    -- Return the ROPED Q,K in the Q/K slots (what the SDPA saw) so the sdpa
    -- backward differentiates w.r.t. them; the block backward then un-ropes.
    return (s, oSSA, qAttn, kAttn, s!"%mh_v{tag}", smLseSSA, s!"%mh_pp{tag}", avSSA)
  | _ => return ("    // mhsa error\n", xSSA, "", "", "", "", "", "")

/-- Emit a transformer encoder block forward pass.
    Layout: LN1 → MHSA → + → LN2 → fc1 → GELU → fc2 → +
    Uses pidx starting at `startP` and 8 param pairs per block:
    LN1(g,b), Wq,bq, Wk,bk, Wv,bv, Wo,bo, LN2(g,b), Wfc1,bfc1, Wfc2,bfc2.
    Returns (code, outSSA, newPidx). -/
private def emitTransformerBlockForward (tag : String) (startP : Nat) (xSSA : String) (shape : List Nat)
    (heads mlpDim : Nat) (causalMask : Bool := false) (rope : Bool := false) : String × String × Nat := Id.run do
  match shape with
  | [_b, _n, d] =>
    let ty := tensorTy shape
    let mut s := ""
    let mut p := startP
    -- LN1
    let g1 := s!"%W{p}"; let b1 := s!"%b{p}"; p := p + 1
    let (ln1Code, ln1Out, _, _, _) := emitLayerNormForward s!"{tag}_ln1" xSSA shape g1 b1
    s := s ++ ln1Code
    -- MHSA
    let wq := s!"%W{p}"; let bq := s!"%b{p}"; p := p + 1
    let wk := s!"%W{p}"; let bk := s!"%b{p}"; p := p + 1
    let wv := s!"%W{p}"; let bv := s!"%b{p}"; p := p + 1
    let wo := s!"%W{p}"; let bo := s!"%b{p}"; p := p + 1
    let (mhsaCode, mhsaOut, _, _, _, _, _, _) := emitMHSAForward s!"{tag}_mh" ln1Out shape heads wq bq wk bk wv bv wo bo causalMask false rope
    s := s ++ mhsaCode
    -- Residual
    s := s ++ s!"    %tb_r1{tag} = stablehlo.add {xSSA}, {mhsaOut} : {ty}\n"
    let r1 := s!"%tb_r1{tag}"
    -- LN2
    let g2 := s!"%W{p}"; let b2 := s!"%b{p}"; p := p + 1
    let (ln2Code, ln2Out, _, _, _) := emitLayerNormForward s!"{tag}_ln2" r1 shape g2 b2
    s := s ++ ln2Code
    -- fc1 (D → mlpDim)
    let wf1 := s!"%W{p}"; let bf1 := s!"%b{p}"; p := p + 1
    let (fc1Code, fc1Out, fc1Shape) := emitDense3D s!"{tag}_fc1" ln2Out shape wf1 bf1 mlpDim
    s := s ++ fc1Code
    -- GELU
    let (geCode, geOut, _) := emitGeluForward s!"{tag}_ge" fc1Out fc1Shape
    s := s ++ geCode
    -- fc2 (mlpDim → D)
    let wf2 := s!"%W{p}"; let bf2 := s!"%b{p}"; p := p + 1
    let (fc2Code, fc2Out, _) := emitDense3D s!"{tag}_fc2" geOut fc1Shape wf2 bf2 d
    s := s ++ fc2Code
    -- Residual
    s := s ++ s!"    %tb_r2{tag} = stablehlo.add {r1}, {fc2Out} : {ty}\n"
    return (s, s!"%tb_r2{tag}", p)
  | _ => return ("    // transformerBlock error\n", xSSA, startP)

/-- Emit a residual block stage: nBlocks basic blocks, first may downsample.
    Returns (code, newSSA, newShape, newPidx). -/
private def emitResidualBlock (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc nBlocks firstStride : Nat) (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let needsProj := !(ic == oc && firstStride == 1)
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let blockInShape := shape
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    -- conv1: 3×3, stride, relu
    let (s1, out1, sh1) := emitConvBn p ssa shape blockIc oc 3 stride true (fixedBN := fixedBN)
    code := code ++ s1; ssa := out1; shape := sh1; p := p + 1
    -- conv2: 3×3, stride 1, NO relu
    let (s2, out2, _sh2) := emitConvBn p ssa shape oc oc 3 1 false (fixedBN := fixedBN)
    code := code ++ s2; ssa := out2; p := p + 1
    -- Skip connection
    let mut skipSSA := blockIn
    if bi == 0 && needsProj then
      -- 1×1 projection convBn (no relu)
      let (sp, outp, _) := emitConvBn p blockIn blockInShape ic oc 1 firstStride false (fixedBN := fixedBN)
      code := code ++ sp; skipSSA := outp; p := p + 1
    -- Add + ReLU
    code := code ++ s!"    %rb_add{startPidx}_{bi} = stablehlo.add {ssa}, {skipSSA} : {tensorTy shape}\n"
    code := code ++ s!"    %rb_rz{startPidx}_{bi} = stablehlo.constant dense<0.0> : {tensorTy shape}\n"
    code := code ++ s!"    %rb_out{startPidx}_{bi} = stablehlo.maximum %rb_add{startPidx}_{bi}, %rb_rz{startPidx}_{bi} : {tensorTy shape}\n"
    ssa := s!"%rb_out{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit a bottleneck block stage: nBlocks bottleneck blocks (1x1→3x3→1x1).
    mid = oc / 4 is the bottleneck channels. First block may downsample. -/
private def emitBottleneckBlock (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc nBlocks firstStride : Nat) (fixedBN : Bool := false) : String × String × List Nat × Nat := Id.run do
  let mid := oc / 4
  let needsProj := !(ic == oc && firstStride == 1)
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let blockInShape := shape
    let stride := if bi == 0 then firstStride else 1
    let blockIc := if bi == 0 then ic else oc
    -- conv1: 1×1, stride 1, relu (reduce channels)
    let (s1, out1, sh1) := emitConvBn p ssa shape blockIc mid 1 1 true (fixedBN := fixedBN)
    code := code ++ s1; ssa := out1; shape := sh1; p := p + 1
    -- conv2: 3×3, stride, relu
    let (s2, out2, sh2) := emitConvBn p ssa shape mid mid 3 stride true (fixedBN := fixedBN)
    code := code ++ s2; ssa := out2; shape := sh2; p := p + 1
    -- conv3: 1×1, stride 1, NO relu (expand channels)
    let (s3, out3, sh3) := emitConvBn p ssa shape mid oc 1 1 false (fixedBN := fixedBN)
    code := code ++ s3; ssa := out3; shape := sh3; p := p + 1
    -- Skip connection
    let mut skipSSA := blockIn
    if bi == 0 && needsProj then
      let (sp, outp, _) := emitConvBn p blockIn blockInShape ic oc 1 firstStride false (fixedBN := fixedBN)
      code := code ++ sp; skipSSA := outp; p := p + 1
    -- Add + ReLU
    code := code ++ s!"    %bn_add{startPidx}_{bi} = stablehlo.add {ssa}, {skipSSA} : {tensorTy shape}\n"
    code := code ++ s!"    %bn_rz{startPidx}_{bi} = stablehlo.constant dense<0.0> : {tensorTy shape}\n"
    code := code ++ s!"    %bn_out{startPidx}_{bi} = stablehlo.maximum %bn_add{startPidx}_{bi}, %bn_rz{startPidx}_{bi} : {tensorTy shape}\n"
    ssa := s!"%bn_out{startPidx}_{bi}"
  return (code, ssa, shape, p)

/-- Emit a dense layer. Assumes (batch, fanIn) input. -/
private def emitDense (pidx : Nat) (curSSA : String) (batchSize fanIn fanOut : Nat)
    (act : Activation) : String × String := Id.run do
  let bTy := tensorTy [batchSize, fanOut]
  let mut s := ""
  s := s ++ s!"    %mm{pidx} = stablehlo.dot_general {curSSA}, %W{pidx},\n"
  s := s ++ "              contracting_dims = [1] x [0],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({tensorTy [batchSize, fanIn]}, {tensorTy [fanIn, fanOut]}) -> {bTy}\n"
  s := s ++ s!"    %bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1]\n"
  s := s ++ s!"           : ({tensorTy [fanOut]}) -> {bTy}\n"
  s := s ++ s!"    %ab{pidx} = stablehlo.add %mm{pidx}, %bb{pidx} : {bTy}\n"
  match act with
  | .relu =>
    s := s ++ s!"    %z{pidx} = stablehlo.constant dense<0.0> : {bTy}\n"
    s := s ++ s!"    %h{pidx} = stablehlo.maximum %ab{pidx}, %z{pidx} : {bTy}\n"
    return (s, s!"%h{pidx}")
  | .identity => return (s, s!"%ab{pidx}")
  | .relu6 => return (s, s!"%ab{pidx}")
  | .swish =>
    s := s ++ s!"    %dsig{pidx} = stablehlo.logistic %ab{pidx} : {bTy}\n"
    s := s ++ s!"    %dsw{pidx} = stablehlo.multiply %ab{pidx}, %dsig{pidx} : {bTy}\n"
    return (s, s!"%dsw{pidx}")
  | .hSwish =>
    -- inline h-swish: x * max(0, min(6, x+3)) / 6
    s := s ++ s!"    %dh3_{pidx} = stablehlo.constant dense<3.0> : {bTy}\n"
    s := s ++ s!"    %dh6_{pidx} = stablehlo.constant dense<6.0> : {bTy}\n"
    s := s ++ s!"    %dh0_{pidx} = stablehlo.constant dense<0.0> : {bTy}\n"
    s := s ++ s!"    %dhxp3_{pidx} = stablehlo.add %ab{pidx}, %dh3_{pidx} : {bTy}\n"
    s := s ++ s!"    %dhc_{pidx} = stablehlo.minimum %dhxp3_{pidx}, %dh6_{pidx} : {bTy}\n"
    s := s ++ s!"    %dhr_{pidx} = stablehlo.maximum %dhc_{pidx}, %dh0_{pidx} : {bTy}\n"
    s := s ++ s!"    %dhd_{pidx} = stablehlo.divide %dhr_{pidx}, %dh6_{pidx} : {bTy}\n"
    s := s ++ s!"    %dhsw_{pidx} = stablehlo.multiply %ab{pidx}, %dhd_{pidx} : {bTy}\n"
    return (s, s!"%dhsw_{pidx}")
  | .gelu =>
    let (sa, oa) := emitActForward .gelu s!"_d{pidx}" s!"%ab{pidx}" [batchSize, fanOut]
    return (s ++ sa, oa)

/-- Emit a ConvNeXt stage (forward inference): `nBlocks` repeats of
    DW7×7-raw → norm(channel-axis) → 1×1 expand (×4) → activation → 1×1 project
    → LayerScale → residual add. Five pidx slots per block, in the order:
    DW (W, b), norm (γ, β), expand (W, b), project (W, b), LayerScale (γ).
    `norm` selects LN (default, paper) or BN (running stats fed in eval).
    Caller must already have emitted any preceding downsample. -/
private def emitConvNextStage (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (channels nBlocks : Nat) (norm : Normalization) (act : Activation) (fixedBN : Bool := false)
    : String × String × List Nat × Nat := Id.run do
  let mut code := ""
  let mut ssa := curSSA
  let mut shape := curShape
  let mut p := startPidx
  for bi in [:nBlocks] do
    let blockIn := ssa
    let blockShape := shape
    let tag := s!"{startPidx}_{bi}"
    -- 1) Depthwise 7×7 raw conv (pidx p)
    let (dwCode, dwOut, dwShape) := emitDepthwiseConvRaw p ssa shape channels 7 1
    code := code ++ dwCode
    ssa := dwOut; shape := dwShape; p := p + 1
    -- 2) Norm over channels (pidx p): LN over channel axis (per-spatial)
    --    or BN per-channel (over batch+spatial). Both consume %W{p}=γ, %b{p}=β.
    let (normCode, normOut) := match norm with
      | .ln =>
        let (c0, out, _, _, _) := emitLayerNormForwardNCHW tag ssa shape s!"%W{p}" s!"%b{p}"
        (c0, out)
      | .bn =>
        let (c0, out, _, _, _) := emitBatchNormForwardNCHW p ssa shape s!"%W{p}" s!"%b{p}" (fixedBN := fixedBN)
        (c0, out)
    code := code ++ normCode
    ssa := normOut; p := p + 1
    -- 3) 1×1 expand conv c → 4c (pidx p)
    let (exCode, exOut, exShape) := emitConv2d p ssa shape channels (4 * channels) 1 .identity
    code := code ++ exCode
    ssa := exOut; shape := exShape; p := p + 1
    -- 4) Activation on the 4c-channel feature map
    let (actCode, actOut) := emitActForward act s!"_cn{tag}" ssa shape
    code := code ++ actCode
    ssa := actOut
    -- 5) 1×1 project conv 4c → c (pidx p)
    let (pjCode, pjOut, pjShape) := emitConv2d p ssa shape (4 * channels) channels 1 .identity
    code := code ++ pjCode
    ssa := pjOut; shape := pjShape; p := p + 1
    -- 6) LayerScale: per-channel scalar multiply by %W{p} : [c]
    let cTy := tensorTy [channels]
    let ty := tensorTy shape
    code := code ++ s!"    %cn_ls_bc{tag} = stablehlo.broadcast_in_dim %W{p}, dims = [1] : ({cTy}) -> {ty}\n"
    code := code ++ s!"    %cn_ls{tag} = stablehlo.multiply {ssa}, %cn_ls_bc{tag} : {ty}\n"
    p := p + 1
    -- 7) Residual add: blockIn + LayerScale output
    code := code ++ s!"    %cn_res{tag} = stablehlo.add {blockIn}, %cn_ls{tag} : {tensorTy blockShape}\n"
    ssa := s!"%cn_res{tag}"
  return (code, ssa, shape, p)

/-- Emit a ConvNeXt downsample (forward inference): norm(channel-axis) on
    `ic`-channel input, then a 2×2 stride-2 valid-pad conv `ic → oc`.
    Two pidx slots: norm (γ, β), conv (W, b). `norm` selects LN/BN. -/
private def emitConvNextDownsample (startPidx : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc : Nat) (norm : Normalization) (fixedBN : Bool := false)
    : String × String × List Nat × Nat := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let mut code := ""
    let mut p := startPidx
    let tag := s!"{startPidx}"
    -- 1) Norm over channels (pidx p), γ = %W{p}, β = %b{p}
    let (normCode, normOut) := match norm with
      | .ln =>
        let (c0, out, _, _, _) := emitLayerNormForwardNCHW s!"ds{tag}" curSSA curShape s!"%W{p}" s!"%b{p}"
        (c0, out)
      | .bn =>
        let (c0, out, _, _, _) := emitBatchNormForwardNCHW p curSSA curShape s!"%W{p}" s!"%b{p}" (fixedBN := fixedBN)
        (c0, out)
    code := code ++ normCode
    let lnOut := normOut
    p := p + 1
    -- 2) 2×2 stride-2 valid-pad conv ic → oc (pidx p)
    let oH := h / 2
    let oW := w / 2
    let outShape := [b, oc, oH, oW]
    code := code ++ s!"    %cnds_cv{tag} = \"stablehlo.convolution\"({lnOut}, %W{p}) " ++ "{\n"
    code := code ++ "        batch_group_count = 1 : i64,\n"
    code := code ++ convDimNumbers
    code := code ++ "        feature_group_count = 1 : i64,\n"
    code := code ++ "        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,\n"
    code := code ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    code := code ++ "        window_strides = array<i64: 2, 2>\n"
    code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, 2, 2]}) -> {tensorTy outShape}\n"
    code := code ++ s!"    %cnds_bb{tag} = stablehlo.broadcast_in_dim %b{p}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    code := code ++ s!"    %cnds_out{tag} = stablehlo.add %cnds_cv{tag}, %cnds_bb{tag} : {tensorTy outShape}\n"
    p := p + 1
    return (code, s!"%cnds_out{tag}", outShape, p)
  | _ => return ("    // convNextDownsample error\n", curSSA, curShape, startPidx)

/-- The per-image timestep is carried by the last channel of the DDPM
    network input (`prependTChannel` fills it with `t/Tmax`, constant
    over the plane). Given the flat input `%x_flat : [B, inDim]` and the
    channel layout, the timestep column offset is `(cIn-1)·H·W`. -/
private def ddpmTStepOffset (spec : NetSpec) : Nat :=
  let cIn := match spec.layers.head? with
    | some (.unetDown ic _)      => ic
    | some (.convBn ic _ _ _ _)  => ic
    | some (.conv2d ic _ _ _ _)  => ic
    | _ => 1
  (cIn - 1) * spec.imageH * spec.imageW

/-- Forward emit for `.timeCondAdd channels nFreq`, shared by the eval
    and train-step forward walks. Reads the per-image timestep scalar
    `s = t/Tmax` from `%x_flat[:, tOff]`, builds a `2·nFreq` sin/cos
    embedding in-graph (frequencies `π·2^k`), projects it through the
    learned dense `pW : [2·nFreq, C]` + `pB : [C]`, and adds the result
    (broadcast over H,W) onto `featSSA : [B, C, H, W]`.
    Returns `(code, outSSA, embSSA)` — `embSSA` (the `[B, 2·nFreq]`
    sin/cos features) is stored by the train forward for the backward. -/
private def emitTimeCondForward (tag : String) (xSSA : String) (B inDim tOff nFreq c : Nat)
    (featSSA : String) (featShape : List Nat) (pW pB : String)
    : String × String × String := Id.run do
  let nf := nFreq
  let embDim := 2 * nf
  let bnf := tensorTy [B, nf]
  let bemb := tensorTy [B, embDim]
  let bc := tensorTy [B, c]
  -- Frequency constant ω_k = π·2^k, k = 0 .. nFreq-1.
  let freqs := (List.range nf).map (fun k => s!"{3.141592653589793 * (Float.ofNat (2 ^ k))}")
  let mut s : String := ""
  -- Slice the timestep column s : [B, 1] from the flat input : [B, inDim].
  s := s ++ s!"    %tc_s{tag} = \"stablehlo.slice\"({xSSA}) " ++ "{" ++
       s!" start_indices = array<i64: 0, {tOff}>, limit_indices = array<i64: {B}, {tOff+1}>, strides = array<i64: 1, 1>" ++
       "} : " ++ s!"({tensorTy [B, inDim]}) -> {tensorTy [B, 1]}\n"
  s := s ++ s!"    %tc_sb{tag} = stablehlo.broadcast_in_dim %tc_s{tag}, dims = [0, 1] : ({tensorTy [B, 1]}) -> {bnf}\n"
  s := s ++ s!"    %tc_freq{tag} = stablehlo.constant dense<[{String.intercalate ", " freqs}]> : {tensorTy [nf]}\n"
  s := s ++ s!"    %tc_freqb{tag} = stablehlo.broadcast_in_dim %tc_freq{tag}, dims = [1] : ({tensorTy [nf]}) -> {bnf}\n"
  s := s ++ s!"    %tc_ph{tag} = stablehlo.multiply %tc_sb{tag}, %tc_freqb{tag} : {bnf}\n"
  s := s ++ s!"    %tc_sin{tag} = stablehlo.sine %tc_ph{tag} : {bnf}\n"
  s := s ++ s!"    %tc_cos{tag} = stablehlo.cosine %tc_ph{tag} : {bnf}\n"
  s := s ++ s!"    %tc_emb{tag} = stablehlo.concatenate %tc_sin{tag}, %tc_cos{tag}, dim = 1 : ({bnf}, {bnf}) -> {bemb}\n"
  -- Project: [B, 2nFreq] · [2nFreq, C] -> [B, C], + bias.
  s := s ++ s!"    %tc_proj{tag} = stablehlo.dot_general %tc_emb{tag}, {pW},\n"
  s := s ++ s!"              contracting_dims = [1] x [0],\n"
  s := s ++ s!"              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({bemb}, {tensorTy [embDim, c]}) -> {bc}\n"
  s := s ++ s!"    %tc_bb{tag} = stablehlo.broadcast_in_dim {pB}, dims = [1] : ({tensorTy [c]}) -> {bc}\n"
  s := s ++ s!"    %tc_projb{tag} = stablehlo.add %tc_proj{tag}, %tc_bb{tag} : {bc}\n"
  -- Broadcast [B, C] -> [B, C, H, W] and add onto the feature map.
  s := s ++ s!"    %tc_bc{tag} = stablehlo.broadcast_in_dim %tc_projb{tag}, dims = [0, 1] : ({bc}) -> {tensorTy featShape}\n"
  s := s ++ s!"    %tc_out{tag} = stablehlo.add {featSSA}, %tc_bc{tag} : {tensorTy featShape}\n"
  return (s, s!"%tc_out{tag}", s!"%tc_emb{tag}")

/-- Token-embedding forward via a true `stablehlo.gather` (Part II Option 2).
    The `[B, T]` f32 ids index the `[V, D]` table directly: `emb[b,t,:] =
    W[ids[b,t], :]`, producing `%tpe_emb{pos}` `[B, T, D]` — no `[B, T, V]`
    one-hot. f32 ids convert to i32 exactly (vocab « 2²⁴). Literal attribute
    braces are concatenated (Lean s-strings reserve `{`). -/
private def emitTokenGatherFwd (pos : Nat) (idsSSA : String) (idsShape : List Nat)
    (pW : String) (b t d v : Nat) : String := Id.run do
  let btd := tensorTy [b, t, d]
  let bti := s!"tensor<{b}x{t}xi32>"
  let bt1i := s!"tensor<{b}x{t}x1xi32>"
  let mut s := ""
  s := s ++ s!"    %tpe_idsi{pos} = stablehlo.convert {idsSSA} : ({tensorTy idsShape}) -> {bti}\n"
  s := s ++ s!"    %tpe_idx{pos} = stablehlo.reshape %tpe_idsi{pos} : ({bti}) -> {bt1i}\n"
  s := s ++ s!"    %tpe_emb{pos} = \"stablehlo.gather\"({pW}, %tpe_idx{pos}) " ++ "{\n"
  s := s ++ "      dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,\n"
  s := s ++ s!"      slice_sizes = array<i64: 1, {d}>, indices_are_sorted = false\n"
  s := s ++ "    } : " ++ s!"({tensorTy [v, d]}, {bt1i}) -> {btd}\n"
  return s

/-- Emit the `@forward` function body walking layers in order. When
    `stopAtGAP` is true (GradCAM mode), the walk halts at the first
    `.globalAvgPool` it encounters: instead of emitting GAP+dense, we
    reshape the pre-GAP activation `[B, C, H, W]` to flat `[B, C*H*W]`
    and return that. CAM math is then done in Lean against the dense
    head's weights. -/
private def emitForwardBody (spec : NetSpec) (batchSize : Nat)
    (fixedBN : Bool := false) (stopAtGAP : Bool := false) : String := Id.run do
  let mut code : String := ""
  let mut curSSA : String := "%x"
  let mut curShape : List Nat := [batchSize, inputFlatDim spec]
  -- If first layer is conv, reshape flat input → NCHW
  match inputChannels spec with
  | some ic =>
    let nchw := [batchSize, ic, spec.imageH, spec.imageW]
    code := code ++ s!"    %xr = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy nchw}\n"
    curSSA := "%xr"
    curShape := nchw
  | none => pure ()
  let mut pidx : Nat := 0   -- param index (conv + dense)
  let mut pos  : Nat := 0   -- layer position (for unique SSA names on pool/flatten)
  -- UNet skip stack: each `unetDown` pushes the (preMaxPoolSSA, preMaxPoolShape)
  -- of its second convBn output; each `unetUp` pops the most-recent one to
  -- concat with its upsampled feature. Pairing is LIFO — i-th unetUp from the
  -- bottom matches i-th unetDown from the top, which is what unetPets / unet
  -- assume.
  let mut skipStack : List (String × List Nat) := []
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut act =>
      let (snip, newSSA) := emitDense pidx curSSA batchSize fanIn fanOut act
      code := code ++ snip
      curSSA := newSSA
      curShape := [batchSize, fanOut]
      pidx := pidx + 1
    | .conv2d ic oc kSize _ act =>
      let (snip, newSSA, newShape) := emitConv2d pidx curSSA curShape ic oc kSize act
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := pidx + 1
    | .maxPool size stride =>
      let (snip, newSSA, newShape) := emitMaxPool pos curSSA curShape size stride
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .flatten =>
      let (snip, newSSA, newShape) := emitFlatten pos curSSA curShape
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .bilinearUpsample scale =>
      let (snip, newSSA, newShape) := emitBilinearUpsample pos curSSA curShape scale
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .convBn ic oc kSize stride _ =>
      let (snip, newSSA, newShape) := emitConvBn pidx curSSA curShape ic oc kSize stride (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := pidx + 1
    | .globalAvgPool =>
      if stopAtGAP then
        match curShape with
        | [b, c, h, w] =>
          let flat := c * h * w
          let flatShape := [b, flat]
          code := code ++ s!"    %lc_flat{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy flatShape}\n"
          curSSA := s!"%lc_flat{pos}"
          curShape := flatShape
          break
        | _ =>
          code := code ++ "    // forward_cam: pre-GAP shape not rank-4\n"
          break
      else
        let (snip, newSSA, newShape) := emitGlobalAvgPool pos curSSA curShape
        code := code ++ snip
        curSSA := newSSA
        curShape := newShape
    | .residualBlock ic oc nBlocks firstStride =>
      let (snip, newSSA, newShape, newPidx) := emitResidualBlock pidx curSSA curShape ic oc nBlocks firstStride (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let (snip, newSSA, newShape, newPidx) := emitBottleneckBlock pidx curSSA curShape ic oc nBlocks firstStride (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .invertedResidual ic oc expand stride nBlocks =>
      let (snip, newSSA, newShape, newPidx) := emitInvertedResidual pidx curSSA curShape ic oc expand stride nBlocks (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .mbConv ic oc expand kSize stride nBlocks useSE act =>
      let (snip, newSSA, newShape, newPidx) := emitMbConv pidx curSSA curShape ic oc expand kSize stride nBlocks useSE (act := act) (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      let (snip, newSSA, newShape, newPidx) := emitFusedMbConv pidx curSSA curShape ic oc expand kSize stride nBlocks useSE (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .mbConvV3 ic oc expandCh kSize stride useSE act =>
      let (snip, newSSA, newShape, newPidx) := emitMbConvV3 pidx curSSA curShape ic oc expandCh kSize stride useSE (act == .hSwish) (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .uib ic oc expand stride preDWk postDWk =>
      let (snip, newSSA, newShape, newPidx) := emitUib pidx curSSA curShape ic oc expand stride preDWk postDWk (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .convNextStage channels nBlocks norm act =>
      let (snip, newSSA, newShape, newPidx) :=
        emitConvNextStage pidx curSSA curShape channels nBlocks norm act (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .convNextDownsample ic oc norm =>
      let (snip, newSSA, newShape, newPidx) :=
        emitConvNextDownsample pidx curSSA curShape ic oc norm (fixedBN := fixedBN)
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
      pidx := newPidx
    | .patchEmbed ic dim pSize nP =>
      let pW := s!"%W{pidx}"
      let pB := s!"%b{pidx}"; pidx := pidx + 1
      let pCls := s!"%W{pidx}"; pidx := pidx + 1
      let pPos := s!"%W{pidx}"; pidx := pidx + 1
      let (snip, newSSA, newShape) := emitPatchEmbedForward s!"{pos}" curSSA curShape ic dim pSize nP pW pB pCls pPos
      code := code ++ snip
      curSSA := newSSA
      curShape := newShape
    | .transformerEncoder dim heads mlpDim nBlocks causalMask keepSequence _ rope =>
      let mut cs := curSSA
      let mut p := pidx
      for bi in [:nBlocks] do
        let (snip, newSSA, newP) := emitTransformerBlockForward s!"{pos}_{bi}" p cs curShape heads mlpDim causalMask rope
        code := code ++ snip
        cs := newSSA
        p := newP
      -- Final LN
      let g := s!"%W{p}"; let bS := s!"%b{p}"; p := p + 1
      let (lnCode, lnOut, _, _, _) := emitLayerNormForward s!"{pos}_finalln" cs curShape g bS
      code := code ++ lnCode
      cs := lnOut
      if causalMask || keepSequence then
        -- Causal/sequence mode: keep the full [B, T, D] for autoregressive
        -- LM heads or for downstream `.spatialUnflatten` back to NCHW.
        curSSA := cs
      else
        -- ViT: slice the CLS token [B, 0, :] -> (B, D) for the classification head.
        match curShape with
        | [b, _, _] =>
          let clsShape := [b, 1, dim]
          let outShape := [b, dim]
          code := code ++ s!"    %te_cls{pos} = \"stablehlo.slice\"({cs}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, 1, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({tensorTy curShape}) -> {tensorTy clsShape}\n"
          code := code ++ s!"    %te_out{pos} = stablehlo.reshape %te_cls{pos} : ({tensorTy clsShape}) -> {tensorTy outShape}\n"
          curSSA := s!"%te_out{pos}"
          curShape := outShape
        | _ => pure ()
      let _ := dim
      pidx := p
    | .unetDown ic oc =>
      -- [convBn(ic→oc) + convBn(oc→oc)] → push pre-pool feature → maxPool 2.
      let (s1, ssa1, sh1) := emitConvBn pidx curSSA curShape ic oc 3 1 (fixedBN := fixedBN)
      code := code ++ s1
      pidx := pidx + 1
      let (s2, ssa2, sh2) := emitConvBn pidx ssa1 sh1 oc oc 3 1 (fixedBN := fixedBN)
      code := code ++ s2
      pidx := pidx + 1
      -- Save pre-pool feature for the matching `unetUp`.
      skipStack := (ssa2, sh2) :: skipStack
      let (s3, ssa3, sh3) := emitMaxPool pos ssa2 sh2 2 2
      code := code ++ s3
      curSSA := ssa3
      curShape := sh3
    | .unetUp ic oc =>
      -- bilinear×2 → concat with most-recent skip → 2× convBn ((ic+oc)→oc, oc→oc).
      let (s1, ssa1, sh1) := emitBilinearUpsample pos curSSA curShape 2
      code := code ++ s1
      match skipStack with
      | (skipSSA, skipShape) :: rest =>
        skipStack := rest
        let (s2, ssa2, sh2) := emitChannelConcat s!"{pos}_uu" ssa1 skipSSA sh1 skipShape
        code := code ++ s2
        let (s3, ssa3, sh3) := emitConvBn pidx ssa2 sh2 (ic + oc) oc 3 1 (fixedBN := fixedBN)
        code := code ++ s3
        pidx := pidx + 1
        let (s4, ssa4, sh4) := emitConvBn pidx ssa3 sh3 oc oc 3 1 (fixedBN := fixedBN)
        code := code ++ s4
        pidx := pidx + 1
        curSSA := ssa4
        curShape := sh4
      | [] =>
        code := code ++ "    // unetUp: skip stack empty (no matching unetDown above)\n"
    | .tokenPositionEmbed v t d ids gather posEmb =>
      -- Input is flat [B, V*T] one-hot (reshape to [B, T, V]) or, with
      -- idsInput, [B, T] f32 token ids (one-hot built in-graph via
      -- iota + compare + select). With `gather`, the [B, T] ids index the
      -- [V, D] table directly via stablehlo.gather — no [B, T, V] tensor.
      -- emb [B, T, D], then (unless posEmb off) add learnable position [T, D].
      match curShape with
      | [b, _] =>
        let btv := [b, t, v]
        let btd := [b, t, d]
        let pW := s!"%W{pidx}"            -- [V, D] embedding matrix (no bias)
        let pPos := s!"%W{pidx + 1}"      -- [T, D] positional embedding
        if gather then
          code := code ++ emitTokenGatherFwd pos curSSA curShape pW b t d v
        else
          if ids then
            let i1Ty := s!"tensor<{b}x{t}x{v}xi1>"
            code := code ++ s!"    %tpe_idb{pos} = stablehlo.broadcast_in_dim {curSSA}, dims = [0, 1] : ({tensorTy curShape}) -> {tensorTy btv}\n"
            code := code ++ s!"    %tpe_iota{pos} = stablehlo.iota dim = 2 : {tensorTy btv}\n"
            code := code ++ s!"    %tpe_cmp{pos} = stablehlo.compare EQ, %tpe_idb{pos}, %tpe_iota{pos}, FLOAT : ({tensorTy btv}, {tensorTy btv}) -> {i1Ty}\n"
            code := code ++ s!"    %tpe_onef{pos} = stablehlo.constant dense<1.0> : {tensorTy btv}\n"
            code := code ++ s!"    %tpe_zerof{pos} = stablehlo.constant dense<0.0> : {tensorTy btv}\n"
            code := code ++ s!"    %tpe_r{pos} = stablehlo.select %tpe_cmp{pos}, %tpe_onef{pos}, %tpe_zerof{pos} : {i1Ty}, {tensorTy btv}\n"
          else
            code := code ++ s!"    %tpe_r{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy btv}\n"
          code := code ++ s!"    %tpe_emb{pos} = stablehlo.dot_general %tpe_r{pos}, {pW},\n"
          code := code ++ "              contracting_dims = [2] x [0],\n"
          code := code ++ "              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy btv}, {tensorTy [v, d]}) -> {tensorTy btd}\n"
        if posEmb then
          code := code ++ s!"    %tpe_pbc{pos} = stablehlo.broadcast_in_dim {pPos}, dims = [1, 2] : ({tensorTy [t, d]}) -> {tensorTy btd}\n"
          code := code ++ s!"    %tpe_out{pos} = stablehlo.add %tpe_emb{pos}, %tpe_pbc{pos} : {tensorTy btd}\n"
          curSSA := s!"%tpe_out{pos}"
          pidx := pidx + 2
        else
          curSSA := s!"%tpe_emb{pos}"
          pidx := pidx + 1
        curShape := btd
      | _ => code := code ++ "    // tokenPositionEmbed error: input not [B, _]\n"
    | .lmHead _d v t =>
      -- Per-position dense [B, T, D] → [B, T, V] via dot_general + bias,
      -- then reshape to flat [B, T*V] so the existing CE machinery accepts it.
      match curShape with
      | [b, _, _] =>
        let btv := [b, t, v]
        let outShape := [b, t * v]
        let pW := s!"%W{pidx}"
        let pB := s!"%b{pidx}"
        let (snip, denseOut, _) := emitDense3D s!"_lmh{pos}" curSSA curShape pW pB v
        code := code ++ snip
        code := code ++ s!"    %lmh_r{pos} = stablehlo.reshape {denseOut} : ({tensorTy btv}) -> {tensorTy outShape}\n"
        curSSA := s!"%lmh_r{pos}"
        curShape := outShape
        pidx := pidx + 1
      | _ => code := code ++ "    // lmHead error: input not [B, T, D]\n"
    | .timeCondAdd c nFreq =>
      -- Per-block DDPM time conditioning; reads the timestep in-graph
      -- from %x_flat's t-channel. Feature map passes through with the
      -- projected sin/cos embedding added.
      match curShape with
      | [b, _, _, _] =>
        let (snip, outSSA, _) := emitTimeCondForward s!"e{pos}" "%x" b (inputFlatDim spec)
          (ddpmTStepOffset spec) nFreq c curSSA curShape s!"%W{pidx}" s!"%b{pidx}"
        code := code ++ snip
        curSSA := outSSA
        pidx := pidx + 1
      | _ => code := code ++ "    // timeCondAdd error: input not [B, C, H, W]\n"
    | .spatialFlatten =>
      -- [B, C, H, W] → transpose [0, 2, 3, 1] → [B, H, W, C] → reshape → [B, H*W, C]
      match curShape with
      | [b, c, h, w] =>
        let bhwc := [b, h, w, c]
        let bnc := [b, h * w, c]
        code := code ++ s!"    %sf_t{pos} = stablehlo.transpose {curSSA}, dims = [0, 2, 3, 1] : ({tensorTy curShape}) -> {tensorTy bhwc}\n"
        code := code ++ s!"    %sf_r{pos} = stablehlo.reshape %sf_t{pos} : ({tensorTy bhwc}) -> {tensorTy bnc}\n"
        curSSA := s!"%sf_r{pos}"
        curShape := bnc
      | _ => code := code ++ "    // spatialFlatten error: expected rank-4 NCHW input\n"
    | .spatialUnflatten c h w =>
      -- [B, H*W, C] → reshape → [B, H, W, C] → transpose [0, 3, 1, 2] → [B, C, H, W]
      match curShape with
      | [b, _, _] =>
        let bhwc := [b, h, w, c]
        let bchw := [b, c, h, w]
        code := code ++ s!"    %su_r{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy bhwc}\n"
        code := code ++ s!"    %su_t{pos} = stablehlo.transpose %su_r{pos}, dims = [0, 3, 1, 2] : ({tensorTy bhwc}) -> {tensorTy bchw}\n"
        curSSA := s!"%su_t{pos}"
        curShape := bchw
      | _ => code := code ++ "    // spatialUnflatten error: expected rank-3 input\n"
    | _ =>
      code := code ++ "    // UNSUPPORTED LAYER\n"
    pos := pos + 1
  code := code ++ s!"    return {curSSA} : {tensorTy curShape}\n"
  pure code

/-- Emit function signature: interleaved (W, b) per param-bearing layer. -/
private def emitForwardSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let inDim := inputFlatDim spec
  let mut params : String := s!"%x: {tensorTy [batchSize, inDim]}"
  let mut outShape : List Nat := [batchSize, inDim]
  let mut curShape : List Nat := [batchSize, inDim]
  -- Initial reshape if CNN
  match inputChannels spec with
  | some ic => curShape := [batchSize, ic, spec.imageH, spec.imageW]
  | none => pure ()
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [fanIn, fanOut]}, %b{pidx}: {tensorTy [fanOut]}"
      curShape := [batchSize, fanOut]
      outShape := curShape
      pidx := pidx + 1
    | .conv2d ic oc kSize _ _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %b{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .convBn ic oc kSize stride _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .unetDown ic oc =>
      -- 2× convBn (ic→oc, oc→oc) then maxPool 2 (no params).
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + 1) / 2, (w + 1) / 2]
      | _ => pure ()
      outShape := curShape
    | .unetUp ic oc =>
      -- bilinear ×2 (no params) + concat with skip(oc) → (ic+oc) channels +
      -- 2× convBn ((ic+oc)→oc, oc→oc).
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic + oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h * 2, w * 2]
      | _ => pure ()
      outShape := curShape
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, blockIc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          -- 1x1 reduce
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 3x3
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, mid, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 1x1 expand
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .invertedResidual ic oc expand stride nBlocks =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          -- Expand 1×1 (skip if expand==1)
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          -- Depthwise 3×3
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- Project 1×1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConv ic oc expand kSize stride nBlocks useSE _act =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            -- SE reduce (conv2d-style: W + b)
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            -- SE expand
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := if expand == 1 then oc else blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          -- Fused expand: regular k×k conv
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConvV3 ic oc expandCh kSize stride useSE _act =>
      match curShape with
      | [b, _, h, w] =>
        let mid := expandCh
        let seMid := Nat.max 1 (mid / 4)
        if expandCh != ic then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        if useSE then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
        pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .uib ic oc expand stride preDWk postDWk =>
      match curShape with
      | [b, _, h, w] =>
        let mid := ic * expand
        -- preDW (if preDWk > 0)
        if preDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [ic, 1, preDWk, preDWk]}, %g{pidx}: {tensorTy [ic]}, %bt{pidx}: {tensorTy [ic]}"
          pidx := pidx + 1
        -- Expand 1×1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        -- postDW (if postDWk > 0)
        if postDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, postDWk, postDWk]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        -- Project 1×1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
        pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .convNextStage channels nBlocks _norm _act =>
      -- Per block: DW (W, b) + LN (W=γ, b=β) + 1×1 expand + 1×1 project + LayerScale (W only).
      let c := channels
      for _ in [:nBlocks] do
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c, 1, 7, 7]}, %b{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c]}, %b{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [4*c, c, 1, 1]}, %b{pidx}: {tensorTy [4*c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c, 4*c, 1, 1]}, %b{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
      outShape := curShape
    | .convNextDownsample ic oc _norm =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [ic]}, %b{pidx}: {tensorTy [ic]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 2, 2]}, %b{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h / 2, w / 2]
      | _ => pure ()
      outShape := curShape
    | .maxPool _size stride =>
      match curShape with
      | [b, c, h, w] =>
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, c, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .globalAvgPool =>
      match curShape with
      | [b, c, _, _] => curShape := [b, c]
      | _ => pure ()
      outShape := curShape
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
      outShape := curShape
    | .bilinearUpsample scale =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c, h * scale, w * scale]
      | _ => pure ()
      outShape := curShape
    | .patchEmbed ic dim pSize nP =>
      -- 3 parameter slots: W (dim, ic, p, p), b (dim,) with name %b{pidx},
      -- then cls as %W{pidx+1} (dim,), and pos as %W{pidx+2} ((nP+1), dim)
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, ic, pSize, pSize]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [nP + 1, dim]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _, _] => curShape := [b, nP + 1, dim]
      | _ => pure ()
      outShape := curShape
    | .transformerEncoder dim _heads mlpDim nBlocks causalMask keepSequence _ _ =>
      for _bi in [:nBlocks] do
        -- LN1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wq, bq
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wk, bk
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wv, bv
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wo, bo
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- LN2
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        -- Wfc1 (dim, mlpDim), bfc1 (mlpDim,)
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, mlpDim]}, %b{pidx}: {tensorTy [mlpDim]}"
        pidx := pidx + 1
        -- Wfc2 (mlpDim, dim), bfc2 (dim,)
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mlpDim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
      -- Final LN
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      if !causalMask && !keepSequence then
        match curShape with
        | [b, _, _] => curShape := [b, dim]
        | _ => pure ()
      outShape := curShape
    | .tokenPositionEmbed v t d _ _ posEmb =>
      -- Embedding W [V, D] (named %W{pidx}, no bias) + positional [T, D] (%W{pidx+1}) unless posEmb off.
      if posEmb then
        params := params ++ s!",\n    %W{pidx}: {tensorTy [v, d]}, %W{pidx + 1}: {tensorTy [t, d]}"
        pidx := pidx + 2
      else
        params := params ++ s!",\n    %W{pidx}: {tensorTy [v, d]}"
        pidx := pidx + 1
      match curShape with
      | [b, _] => curShape := [b, t, d]
      | _ => pure ()
      outShape := curShape
    | .lmHead d v t =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [d, v]}, %b{pidx}: {tensorTy [v]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _] => curShape := [b, t * v]
      | _ => pure ()
      outShape := curShape
    | .timeCondAdd c nFreq =>
      -- Dense W [2·nFreq, C] + bias [C]. Shape passes through.
      params := params ++ s!",\n    %W{pidx}: {tensorTy [2 * nFreq, c]}, %b{pidx}: {tensorTy [c]}"
      pidx := pidx + 1
      outShape := curShape
    | .spatialFlatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, h * w, c]
      | _ => pure ()
      outShape := curShape
    | .spatialUnflatten c h w =>
      match curShape with
      | [b, _, _] => curShape := [b, c, h, w]
      | _ => pure ()
      outShape := curShape
    | _ => pure ()
  pure s!"func.func @forward(\n    {params}\n  ) -> {tensorTy outShape}"

/-- Generate a StableHLO MLIR module for forward-only inference. -/
def generate (spec : NetSpec) (batchSize : Nat) : String :=
  s!"// {spec.name} — Generated by Lean 4 → MLIR (StableHLO)\n" ++
  s!"// Batch size: {batchSize}\n\n" ++
  s!"module @{sanitize spec.name} " ++ "{\n" ++
  "  " ++ emitForwardSig spec batchSize ++ " {\n" ++
  emitForwardBody spec batchSize ++
  "  }\n" ++
  "}\n"

/-- Collect (pidx, oc) pairs for every convBn layer (including those inside residual/bottleneck blocks). -/
def collectBnLayers (spec : NetSpec) : Array (Nat × Nat) := Id.run do
  let mut result : Array (Nat × Nat) := #[]
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .dense _ _ _ => pidx := pidx + 1
    | .conv2d _ _ _ _ _ => pidx := pidx + 1
    | .convBn _ oc _ _ _ =>
      result := result.push (pidx, oc)
      pidx := pidx + 1
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        -- conv1
        result := result.push (pidx, oc)
        pidx := pidx + 1
        -- conv2
        result := result.push (pidx, oc)
        pidx := pidx + 1
        if bi == 0 && needsProj then
          -- projection
          result := result.push (pidx, oc)
          pidx := pidx + 1
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        -- 1x1 reduce
        result := result.push (pidx, mid)
        pidx := pidx + 1
        -- 3x3
        result := result.push (pidx, mid)
        pidx := pidx + 1
        -- 1x1 expand
        result := result.push (pidx, oc)
        pidx := pidx + 1
        if bi == 0 && needsProj then
          -- projection
          result := result.push (pidx, oc)
          pidx := pidx + 1
    | .invertedResidual ic oc expand _ nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        -- Expand 1×1 (skip if expand==1)
        if expand != 1 then
          result := result.push (pidx, mid)
          pidx := pidx + 1
        -- Depthwise 3×3
        result := result.push (pidx, mid)
        pidx := pidx + 1
        -- Project 1×1
        result := result.push (pidx, oc)
        pidx := pidx + 1
    | .mbConv ic oc expand _kSize _ nBlocks useSE _act =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        if expand != 1 then
          result := result.push (pidx, mid)
          pidx := pidx + 1
        result := result.push (pidx, mid)
        pidx := pidx + 1
        if useSE then
          -- SE params (reduce, expand): 2 conv2d-style params, no BN.
          pidx := pidx + 2
        result := result.push (pidx, oc)
        pidx := pidx + 1
    | .fusedMbConv ic oc expand _kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        -- Fused expand convBn
        result := result.push (pidx, mid)
        pidx := pidx + 1
        if useSE then
          pidx := pidx + 2
        if expand != 1 then
          result := result.push (pidx, oc)
          pidx := pidx + 1
    | .mbConvV3 ic oc expandCh _kSize _ useSE _ =>
      let mid := expandCh
      if expandCh != ic then
        result := result.push (pidx, mid)
        pidx := pidx + 1
      result := result.push (pidx, mid)
      pidx := pidx + 1
      if useSE then
        pidx := pidx + 2
      result := result.push (pidx, oc)
      pidx := pidx + 1
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      -- preDW BN (only if preDWk > 0): oc = ic
      if preDWk > 0 then
        result := result.push (pidx, ic)
        pidx := pidx + 1
      -- Expand BN: oc = mid
      result := result.push (pidx, mid)
      pidx := pidx + 1
      -- postDW BN (only if postDWk > 0): oc = mid
      if postDWk > 0 then
        result := result.push (pidx, mid)
        pidx := pidx + 1
      -- Project BN: oc = oc
      result := result.push (pidx, oc)
      pidx := pidx + 1
    | .convNextStage channels nBlocks norm _act =>
      -- 5 pidx slots per block: DW, norm, expand, project, LayerScale.
      -- BN slot is at basePidx + 1 (per block); LN variant has no BN.
      for _ in [:nBlocks] do
        if norm == .bn then
          result := result.push (pidx + 1, channels)
        pidx := pidx + 5
    | .convNextDownsample ic _oc norm =>
      -- 2 pidx slots: norm, conv. BN slot is at basePidx + 0.
      if norm == .bn then
        result := result.push (pidx, ic)
      pidx := pidx + 2
    | .unetDown _ic oc =>
      -- 2× convBn (ic→oc, oc→oc). Both BN, both `oc` channels.
      result := result.push (pidx, oc)
      pidx := pidx + 1
      result := result.push (pidx, oc)
      pidx := pidx + 1
    | .unetUp _ic oc =>
      -- 2× convBn ((ic+oc)→oc, oc→oc). Both BN, both `oc` channels.
      result := result.push (pidx, oc)
      pidx := pidx + 1
      result := result.push (pidx, oc)
      pidx := pidx + 1
    | .patchEmbed _ _ _ _ =>
      -- patchEmbed claims 3 pidx slots (W+b, cls, pos). No BN.
      pidx := pidx + 3
    | .transformerEncoder _dim _heads _mlpDim nBlocks _causal _keepSeq _ _ =>
      -- 8 param pairs per block + 1 for the final LN. No BN.
      pidx := pidx + 8 * nBlocks + 1
    | .tokenPositionEmbed _ _ _ _ _ posEmb =>
      -- W_emb (+ W_pos unless posEmb off). No BN.
      pidx := pidx + (if posEmb then 2 else 1)
    | .lmHead _ _ _ =>
      pidx := pidx + 1
    | .timeCondAdd _ _ =>
      -- 1 slot (W, b). No BN.
      pidx := pidx + 1
    -- spatialFlatten / spatialUnflatten: no params, no pidx advance.
    | _ => pure ()
  return result

/-- Emit function signature for forward_eval: same as forward but with
    bn_mean/bn_var params appended for each BN layer. -/
private def emitForwardEvalSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  let inDim := inputFlatDim spec
  let mut params : String := s!"%x: {tensorTy [batchSize, inDim]}"
  let mut outShape : List Nat := [batchSize, inDim]
  let mut curShape : List Nat := [batchSize, inDim]
  -- Initial reshape if CNN
  match inputChannels spec with
  | some ic => curShape := [batchSize, ic, spec.imageH, spec.imageW]
  | none => pure ()
  let mut pidx : Nat := 0
  for l in spec.layers do
    match l with
    | .dense fanIn fanOut _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [fanIn, fanOut]}, %b{pidx}: {tensorTy [fanOut]}"
      curShape := [batchSize, fanOut]
      outShape := curShape
      pidx := pidx + 1
    | .conv2d ic oc kSize _ _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %b{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .convBn ic oc kSize stride _ =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, kSize, kSize]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
      outShape := curShape
      pidx := pidx + 1
    | .unetDown ic oc =>
      -- 2× convBn (ic→oc, oc→oc) then maxPool 2 (no params).
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + 1) / 2, (w + 1) / 2]
      | _ => pure ()
      outShape := curShape
    | .unetUp ic oc =>
      -- bilinear ×2 (no params) + concat with skip(oc) → (ic+oc) channels +
      -- 2× convBn ((ic+oc)→oc, oc→oc).
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic + oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h * 2, w * 2]
      | _ => pure ()
      outShape := curShape
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, blockIc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, oc, 3, 3]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          -- 1x1 reduce
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 3x3
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, mid, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          -- 1x1 expand
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
          if bi == 0 && needsProj then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
      outShape := curShape
    | .invertedResidual ic oc expand stride nBlocks =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, 3, 3]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConv ic oc expand kSize stride nBlocks useSE _act =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
          pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      match curShape with
      | [b, _, h, w] =>
        for bi in [:nBlocks] do
          let blockIc := if bi == 0 then ic else oc
          let mid := if expand == 1 then oc else blockIc * expand
          let seMid := Nat.max 1 (mid / 4)
          -- Fused expand: regular k×k conv
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, blockIc, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
          if useSE then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
            pidx := pidx + 1
            params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
            pidx := pidx + 1
          if expand != 1 then
            params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
            pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .mbConvV3 ic oc expandCh kSize stride useSE _act =>
      match curShape with
      | [b, _, h, w] =>
        let mid := expandCh
        let seMid := Nat.max 1 (mid / 4)
        if expandCh != ic then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, kSize, kSize]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        if useSE then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [seMid, mid, 1, 1]}, %b{pidx}: {tensorTy [seMid]}"
          pidx := pidx + 1
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, seMid, 1, 1]}, %b{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
        pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .uib ic oc expand stride preDWk postDWk =>
      match curShape with
      | [b, _, h, w] =>
        let mid := ic * expand
        if preDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [ic, 1, preDWk, preDWk]}, %g{pidx}: {tensorTy [ic]}, %bt{pidx}: {tensorTy [ic]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, ic, 1, 1]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
        pidx := pidx + 1
        if postDWk > 0 then
          params := params ++ s!",\n    %W{pidx}: {tensorTy [mid, 1, postDWk, postDWk]}, %g{pidx}: {tensorTy [mid]}, %bt{pidx}: {tensorTy [mid]}"
          pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, mid, 1, 1]}, %g{pidx}: {tensorTy [oc]}, %bt{pidx}: {tensorTy [oc]}"
        pidx := pidx + 1
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, oc, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .convNextStage channels nBlocks _norm _act =>
      let c := channels
      for _ in [:nBlocks] do
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c, 1, 7, 7]}, %b{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c]}, %b{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [4*c, c, 1, 1]}, %b{pidx}: {tensorTy [4*c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c, 4*c, 1, 1]}, %b{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [c]}"
        pidx := pidx + 1
      outShape := curShape
    | .convNextDownsample ic oc _norm =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [ic]}, %b{pidx}: {tensorTy [ic]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [oc, ic, 2, 2]}, %b{pidx}: {tensorTy [oc]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h / 2, w / 2]
      | _ => pure ()
      outShape := curShape
    | .maxPool _ stride =>
      match curShape with
      | [b, c, h, w] =>
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        curShape := [b, c, oH, oW]
      | _ => pure ()
      outShape := curShape
    | .globalAvgPool =>
      match curShape with
      | [b, c, _, _] => curShape := [b, c]
      | _ => pure ()
      outShape := curShape
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
      outShape := curShape
    | .bilinearUpsample scale =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c, h * scale, w * scale]
      | _ => pure ()
      outShape := curShape
    | .patchEmbed ic dim pSize nP =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, ic, pSize, pSize]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [nP + 1, dim]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _, _] => curShape := [b, nP + 1, dim]
      | _ => pure ()
      outShape := curShape
    | .transformerEncoder dim _heads mlpDim nBlocks causalMask keepSequence _ _ =>
      for _bi in [:nBlocks] do
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [dim, mlpDim]}, %b{pidx}: {tensorTy [mlpDim]}"
        pidx := pidx + 1
        params := params ++ s!",\n    %W{pidx}: {tensorTy [mlpDim, dim]}, %b{pidx}: {tensorTy [dim]}"
        pidx := pidx + 1
      params := params ++ s!",\n    %W{pidx}: {tensorTy [dim]}, %b{pidx}: {tensorTy [dim]}"
      pidx := pidx + 1
      if !causalMask && !keepSequence then
        match curShape with
        | [b, _, _] => curShape := [b, dim]
        | _ => pure ()
      outShape := curShape
    | .tokenPositionEmbed v t d _ _ posEmb =>
      if posEmb then
        params := params ++ s!",\n    %W{pidx}: {tensorTy [v, d]}, %W{pidx + 1}: {tensorTy [t, d]}"
        pidx := pidx + 2
      else
        params := params ++ s!",\n    %W{pidx}: {tensorTy [v, d]}"
        pidx := pidx + 1
      match curShape with
      | [b, _] => curShape := [b, t, d]
      | _ => pure ()
      outShape := curShape
    | .lmHead d v t =>
      params := params ++ s!",\n    %W{pidx}: {tensorTy [d, v]}, %b{pidx}: {tensorTy [v]}"
      pidx := pidx + 1
      match curShape with
      | [b, _, _] => curShape := [b, t * v]
      | _ => pure ()
      outShape := curShape
    | .timeCondAdd c nFreq =>
      -- Dense W [2·nFreq, C] + bias [C]. Shape passes through.
      params := params ++ s!",\n    %W{pidx}: {tensorTy [2 * nFreq, c]}, %b{pidx}: {tensorTy [c]}"
      pidx := pidx + 1
      outShape := curShape
    | .spatialFlatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, h * w, c]
      | _ => pure ()
      outShape := curShape
    | .spatialUnflatten c h w =>
      match curShape with
      | [b, _, _] => curShape := [b, c, h, w]
      | _ => pure ()
      outShape := curShape
    | _ => pure ()
  -- Append bn_mean / bn_var params for each BN layer
  let bnLayers := collectBnLayers spec
  for (p, oc) in bnLayers do
    params := params ++ s!",\n    %bn_mean{p}: {tensorTy [oc]}, %bn_var{p}: {tensorTy [oc]}"
  pure s!"func.func @forward_eval(\n    {params}\n  ) -> {tensorTy outShape}"

/-- Generate a StableHLO MLIR module for eval (fixed BN using running mean/var). -/
def generateEval (spec : NetSpec) (batchSize : Nat) : String :=
  s!"// {spec.name} eval — Generated by Lean 4 → MLIR (StableHLO)\n" ++
  s!"// Batch size: {batchSize}, fixedBN: true\n\n" ++
  s!"module @{sanitize spec.name}_eval " ++ "{\n" ++
  "  " ++ emitForwardEvalSig spec batchSize ++ " {\n" ++
  emitForwardBody spec batchSize (fixedBN := true) ++
  "  }\n" ++
  "}\n"

/-- Shape walker: returns `(C, H, W)` of the activation immediately before
    the LAST `.globalAvgPool` in the spec — i.e. the last conv feature
    map, the input to the GAP+dense head. Returns `none` for specs
    without a `globalAvgPool` (those are not CAM-eligible). Mirrors the
    shape transitions in `emitForwardEvalSig`; kept focused (no param
    bookkeeping) so it stays readable. -/
def preGAPShape (spec : NetSpec) : Option (Nat × Nat × Nat) := Id.run do
  let mut curShape : List Nat := [1, inputFlatDim spec]
  match inputChannels spec with
  | some ic => curShape := [1, ic, spec.imageH, spec.imageW]
  | none => pure ()
  let mut found : Option (Nat × Nat × Nat) := none
  for l in spec.layers do
    -- Capture pre-GAP (the shape at the layer's INPUT, before GAP runs).
    match l with
    | .globalAvgPool =>
      match curShape with
      | [_, c, h, w] => found := some (c, h, w)
      | _ => pure ()
    | _ => pure ()
    -- Update curShape based on this layer.
    match l with
    | .dense _ fanOut _ =>
      match curShape with
      | [b, _] => curShape := [b, fanOut]
      | _ => pure ()
    | .conv2d _ oc _ _ _ =>
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
    | .convBn _ oc _ stride _ =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .maxPool _ stride =>
      match curShape with
      | [b, c, h, w] =>
        curShape := [b, c, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .globalAvgPool =>
      match curShape with
      | [b, c, _, _] => curShape := [b, c]
      | _ => pure ()
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
    | .residualBlock _ oc _ firstStride =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
    | .bottleneckBlock _ oc _ firstStride =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
    | .invertedResidual _ oc _ stride _ =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .mbConv _ oc _ _ stride _ _ _ =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .mbConvV3 _ oc _ _ stride _ _ =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .fusedMbConv _ oc _ _ stride _ _ =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .uib _ oc _ stride _ _ =>
      match curShape with
      | [b, _, h, w] =>
        curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .convNextStage _ _ _ _ => pure ()  -- channels & spatial preserved
    | .convNextDownsample _ oc _ =>
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h / 2, w / 2]
      | _ => pure ()
    | .bilinearUpsample scale =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c, h * scale, w * scale]
      | _ => pure ()
    | _ => pure ()
  return found

/-- Forward-with-capture signature: same params as `emitForwardEvalSig`
    but returns the flat pre-GAP feature map `[batch, C*H*W]` instead of
    the logits `[batch, NC]`. -/
private def emitForwardCamSig (spec : NetSpec) (batchSize : Nat) : String := Id.run do
  -- Reuse eval-sig generation, then rewrite the function name and the
  -- return type. Cheaper than duplicating ~250 lines of param emission.
  let evalSig := emitForwardEvalSig spec batchSize
  let (c, h, w) := (preGAPShape spec).getD (0, 0, 0)
  let flat := c * h * w
  let camOutTy := tensorTy [batchSize, flat]
  -- The eval sig ends with `) -> tensor<...x...xf32>`; replace the
  -- trailing return type with the flat lastConv shape.
  let renamed := evalSig.replace "func.func @forward_eval" "func.func @forward_cam"
  -- Strip trailing return-type substring. The eval sig builds it as
  -- `) -> {tensorTy outShape}` where outShape is `[batchSize, NC]`.
  let nc := match spec.layers.getLast? with
    | some (.dense _ fo _) => fo
    | _ => 0
  let evalOutTy := tensorTy [batchSize, nc]
  let needle := s!") -> {evalOutTy}"
  let replacement := s!") -> {camOutTy}"
  pure (renamed.replace needle replacement)

/-- Generate a StableHLO MLIR module for forward-with-capture (CAM mode).
    Module name `<sanitized>_cam`, function `@forward_cam`. Walks the
    network up through the last conv feature map, reshapes it to flat
    `[batch, C*H*W]`, and returns. The Lean side recomputes logits and
    the CAM heatmap from the dense-head weights. -/
def generateForwardCam (spec : NetSpec) (batchSize : Nat) : String :=
  s!"// {spec.name} forward_cam — Generated by Lean 4 → MLIR (StableHLO)\n" ++
  s!"// Batch size: {batchSize}, stops at globalAvgPool, returns flat last-conv\n\n" ++
  s!"module @{sanitize spec.name}_cam " ++ "{\n" ++
  "  " ++ emitForwardCamSig spec batchSize ++ " {\n" ++
  emitForwardBody spec batchSize (fixedBN := true) (stopAtGAP := true) ++
  "  }\n" ++
  "}\n"

-- ════════════════════════════════════════════════════════════════════
-- Train step generation: forward + softmax-CE loss + VJPs + SGD
-- ════════════════════════════════════════════════════════════════════

/-- Record saved during forward pass for use by backward. -/
private structure FwdRec where
  layer     : Layer
  pidx      : Option Nat   -- param index (conv/dense only)
  pos       : Nat          -- layer position
  inputSSA  : String       -- input to this layer
  preActSSA : String       -- pre-activation (before relu), "" if no activation
  outputSSA : String       -- output of this layer
  inShape   : List Nat     -- input shape
  outShape  : List Nat     -- output shape
  -- convBn intermediates for backward
  convOutSSA : String := ""   -- raw conv output before inst norm
  normSSA    : String := ""   -- (x - mean) * istd
  meanBcSSA  : String := ""   -- broadcast mean (B, OC, oH, oW)
  istdBcSSA  : String := ""   -- broadcast istd (B, OC, oH, oW)
  hasRelu    : Bool := true
  hasRelu6   : Bool := false  -- ReLU6 activation (MobileNet)
  hasSwish   : Bool := false  -- Swish activation (x * sigmoid(x)) (EfficientNet)
  hasHSwish  : Bool := false  -- h-swish activation: x * ReLU6(x+3) / 6 (MobileNetV3)
  isDepthwise : Bool := false -- depthwise conv (feature_group_count = channels)
  ic         : Nat := 0      -- input channels (for backward kernel shapes)
  kSize      : Nat := 3      -- kernel size
  stride     : Nat := 1      -- conv stride
  -- Residual skip: after this layer's backward, add this gradient SSA
  addSkipGrad : String := ""
  -- SE block (Squeeze-and-Excitation) intermediates
  isSE       : Bool := false
  sePidxRed  : Nat := 0      -- pidx of SE reduce conv (W, b)
  sePidxExp  : Nat := 0      -- pidx of SE expand conv (W, b)
  seMid      : Nat := 0      -- reduced channels in SE
  seMidFull  : Nat := 0      -- full channels (mid) in SE
  seInputSSA : String := ""  -- SSA of input x to SE (B, mid, H, W)
  seGapSSA   : String := ""  -- GAP result (B, mid) — post squeeze, pre reduce
  seRbSSA    : String := ""  -- reduce-conv + bias (B, seMid) — pre-swish
  seRswSSA   : String := ""  -- reduce swish out (B, seMid)
  seEbSSA    : String := ""  -- expand-conv + bias (B, mid) — pre-sigmoid
  seSigSSA   : String := ""  -- sigmoid result (B, mid)
  seOutSSA   : String := ""  -- x * broadcast(sig) (B, mid, H, W)
  -- SE variant: if true, SE block uses ReLU + h-sigmoid (MobileNetV3);
  -- otherwise swish + sigmoid (EfficientNet).
  seVariant  : Bool := false
  -- ═════════ ViT/Transformer intermediates ═════════
  -- patchEmbed fields (layer marked with isPatchEmbed)
  isPatchEmbed : Bool := false
  pePWPidx     : Nat := 0   -- base pidx for W (conv weight)
  pePBPidx     : Nat := 0   -- pidx for b (conv bias)
  peClsPidx    : Nat := 0   -- pidx for cls token (stored as W{pidx})
  pePosPidx    : Nat := 0   -- pidx for positional embedding
  pePSize      : Nat := 0
  pePIc        : Nat := 0
  pePDim       : Nat := 0
  pePNp        : Nat := 0
  -- Transformer block fields (layer marked with isTransformerBlock)
  isTransformerBlock : Bool := false
  tbBasePidx     : Nat := 0  -- starting pidx for this block's 8 param pairs
  tbHeads        : Nat := 0
  tbMlpDim       : Nat := 0
  tbDim          : Nat := 0
  -- Saved intermediates for each block's backward:
  tbLn1XSSA      : String := ""  -- input to LN1 (block input)
  tbLn1OutSSA    : String := ""
  tbLn1NormSSA   : String := ""
  tbLn1IstdSSA   : String := ""
  tbMhsaOutSSA   : String := ""
  tbR1SSA        : String := ""  -- after first residual (block input)
  tbLn2OutSSA    : String := ""
  tbLn2NormSSA   : String := ""
  tbLn2IstdSSA   : String := ""
  tbMhQSSA       : String := ""
  tbMhKSSA       : String := ""
  tbMhVSSA       : String := ""
  tbMhSmSSA      : String := ""  -- dense softmax probs [b,h,n,n], OR (flash) logsumexp [b,h,n]
  tbMhPpSSA      : String := ""  -- pre-projection MHSA output (B, N, D)
  tbMhFlash      : Bool   := false  -- attention emitted via FlashAttention (tbMhSmSSA is Lse, tbMhOSSA is O)
  tbMhOSSA       : String := ""  -- (flash) the sdpa output O [b,h,n,dh]
  tbMhRope       : Bool   := false  -- RoPE applied to Q/K (Q/K slots hold roped vecs; un-rope dq/dk in bwd)
  tbFc1OutSSA    : String := ""  -- pre-GELU (B, N, mlpDim)
  tbGeluTSSA     : String := ""  -- saved tanh value for GELU backward
  tbGeluOutSSA   : String := ""  -- post-GELU (B, N, mlpDim)
  -- Final LN (after all blocks) record
  isFinalLn       : Bool := false
  finalLnPidx     : Nat := 0
  finalLnNormSSA  : String := ""
  finalLnIstdSSA  : String := ""
  finalLnInSSA    : String := ""
  finalLnOutSSA   : String := ""
  -- CLS slice record (between final LN and classifier dense)
  isClsSlice      : Bool := false
  clsInShape      : List Nat := []
  clsInSSA        : String := ""
  -- ═════════ ConvNeXt block intermediates ═════════
  -- One FwdRec per ConvNeXt block with `isConvNextBlock := true`. Five
  -- pidx slots from `cnbBasePidx`: DW (W,b), norm (γ,β), expand (W,b),
  -- project (W,b), LayerScale (γ). `cnbNorm` selects LN vs BN — saved
  -- intermediates land in different fields per branch.
  isConvNextBlock : Bool := false
  cnbBasePidx     : Nat := 0
  cnbChannels     : Nat := 0
  cnbAct          : Activation := .gelu
  cnbNorm         : Normalization := .ln
  cnbBlockInSSA   : String := ""    -- input to the block (residual root)
  cnbDwOutSSA     : String := ""    -- DW + bias output, pre-norm
  -- LN-mode (norm-axis = channels, transposed view): see emitLayerNormForwardNCHW
  cnbLnNormSSA    : String := ""    -- normalized [b, h*w, c]
  cnbLnIstdSSA    : String := ""    -- 1/σ on shape [b, h*w]
  cnbLnOutSSA     : String := ""    -- post-LN, into 1×1 expand (NCHW)
  -- BN-mode (per-channel stats over batch+spatial; tensors stay NCHW)
  cnbBnNormSSA    : String := ""    -- normalized [b, c, h, w]
  cnbBnMeanBcSSA  : String := ""    -- broadcast mean [b, c, h, w]
  cnbBnIstdBcSSA  : String := ""    -- broadcast istd [b, c, h, w]
  cnbExpandOutSSA : String := ""    -- pre-activation (channels = 4c)
  cnbActOutSSA    : String := ""    -- post-activation, into project
  cnbActTanhSSA   : String := ""    -- saved tanh value for GELU backward
  cnbProjectOutSSA : String := ""   -- pre-LayerScale (channels = c)
  -- ═════════ ConvNeXt downsample intermediates ═════════
  isConvNextDs    : Bool := false
  cndBasePidx     : Nat := 0
  cndIc           : Nat := 0
  cndOc           : Nat := 0
  cndNorm         : Normalization := .ln
  cndInSSA        : String := ""    -- pre-norm
  cndLnNormSSA    : String := ""    -- LN-mode: [b, h*w, ic]
  cndLnIstdSSA    : String := ""    -- LN-mode: [b, h*w]
  cndLnOutSSA     : String := ""    -- post-norm, into 2×2 stride-2 conv (NCHW)
  cndBnNormSSA    : String := ""    -- BN-mode: [b, ic, h, w]
  cndBnMeanBcSSA  : String := ""    -- BN-mode: [b, ic, h, w]
  cndBnIstdBcSSA  : String := ""    -- BN-mode: [b, ic, h, w]
  -- ═════════ UNet decoder concat-split marker ═════════
  -- One per `unetUp`, pushed between bilinearUpsample and the two convBns.
  -- Carries enough info for the backward pass to channel-split the gradient
  -- at the concat output: `inShape` is the upsampled (decoder) side, the
  -- skip side is `unetSkipShape`. The decoder-half gradient flows back to
  -- bilinearUpsample; the skip-half is saved as `%unet_skip_g{unetEncoderIdx}`
  -- and consumed by the matching `unetDown`'s maxPool record (via
  -- `addSkipGrad`). Encoder index is assigned monotonically at forward
  -- emit time; pairing is LIFO (i-th unetUp from the bottom ↔ i-th
  -- unetDown from the top).
  isUnetUpConcat  : Bool := false
  unetSkipShape   : List Nat := []
  unetEncoderIdx  : Nat := 0
  -- ═════════ tinyGPT layers ═════════
  -- tokenPositionEmbed (sets these on its forward record):
  --   tpePidx = base param index (W_emb, then W_pos)
  --   tpeBtv  = SSA name of the [B, T, V] one-hot reshape (used for d_W backward)
  isTokenPosEmbed : Bool := false
  tpePidx         : Nat := 0
  tpeV            : Nat := 0
  tpeT            : Nat := 0
  tpeD            : Nat := 0
  -- One-hot path: SSA of the [B, T, V] one-hot (for dW = onehotᵀ·d_out).
  -- Gather path: SSA of the [B, T, 1] i32 index tensor (for scatter-add dW).
  tpeBtvSSA       : String := ""
  tpeGather       : Bool := false
  tpePosEmb       : Bool := true   -- learned absolute position table present (2 params vs 1)
  -- lmHead (sets these on its forward record):
  --   lmhPidx = param index for W [D, V] + b [V]
  --   lmhBtv  = SSA name of the post-dense [B, T, V] tensor (pre-transpose)
  isLmHead        : Bool := false
  lmhPidx         : Nat := 0
  lmhD            : Nat := 0
  lmhV            : Nat := 0
  lmhT            : Nat := 0
  lmhBtvSSA       : String := ""
instance : Inhabited FwdRec where
  default := { layer := .flatten, pidx := none, pos := 0,
               inputSSA := "", preActSSA := "", outputSSA := "",
               inShape := [], outShape := [] }

/-- Emit convBn forward for train step. Records intermediates for backward.
    Returns (code, FwdRec). -/
private def emitConvBnTrain (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) (relu : Bool) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm: 2-step reduction [2,3] then [0] (IREE can't distribute [0,2,3])
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    let outSSA := if relu then s!"%cbn_out{pidx}" else preSSA
    if relu then
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := outSSA
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := relu
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBn error\n", default)

/-- Emit depthwise convBn forward for train step. Records intermediates for backward.
    Weight shape: (channels, 1, kSize, kSize), feature_group_count = channels.
    Activation: ReLU6 by default, Swish if `useSwish := true`, or h-swish if
    `useHSwish := true`. -/
private def emitDepthwiseConvBnTrain (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (channels kSize stride : Nat) (useSwish : Bool := false)
    (useHSwish : Bool := false) (useRelu : Bool := false) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, channels, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Depthwise conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, channels]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [channels]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [channels]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    if useHSwish then
      -- h-swish: x * ReLU6(x + 3) / 6
      s := s ++ s!"    %cbn_hs3{pidx} = stablehlo.constant dense<3.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hs6{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsxp3{pidx} = stablehlo.add {preSSA}, %cbn_hs3{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsclip{pidx} = stablehlo.minimum %cbn_hsxp3{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsr6{pidx} = stablehlo.maximum %cbn_hsclip{pidx}, %cbn_hsz{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_hsdiv{pidx} = stablehlo.divide %cbn_hsr6{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_hsdiv{pidx} : {tensorTy outShape}\n"
    else if useSwish then
      -- Swish: x * sigmoid(x)
      s := s ++ s!"    %cbn_sig{pidx} = stablehlo.logistic {preSSA} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_sig{pidx} : {tensorTy outShape}\n"
    else if useRelu then
      -- Plain ReLU (used by UIB / MobileNet V4)
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
    else
      -- ReLU6 (default)
      s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_r6r{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_r6s{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
      s := s ++ s!"    %cbn_out{pidx} = stablehlo.minimum %cbn_r6r{pidx}, %cbn_r6s{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn channels channels kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := useRelu
      hasRelu6 := !useSwish && !useHSwish && !useRelu
      hasSwish := useSwish
      hasHSwish := useHSwish
      isDepthwise := true
      ic := channels, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // depthwiseConvBn error\n", default)

/-- Emit convBn forward for train step with ReLU6 (used for inverted residual expand).
    Same as emitConvBnTrain but with ReLU6 activation instead of ReLU. -/
private def emitConvBnTrainRelu6 (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    -- ReLU6
    s := s ++ s!"    %cbn_z{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_r6r{pidx} = stablehlo.maximum {preSSA}, %cbn_z{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_r6s{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.minimum %cbn_r6r{pidx}, %cbn_r6s{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := false, hasRelu6 := true, isDepthwise := false
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBnRelu6 error\n", default)

/-- Emit convBn forward for train step with Swish (x * sigmoid(x)) activation.
    Used by EfficientNet's MBConv. -/
private def emitConvBnTrainSwish (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    -- Swish: x * sigmoid(x)
    s := s ++ s!"    %cbn_sig{pidx} = stablehlo.logistic {preSSA} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_sig{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := false, hasRelu6 := false, hasSwish := true, isDepthwise := false
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBnSwish error\n", default)

/-- Emit convBn forward for train step with h-swish activation:
    h_swish(x) = x * ReLU6(x + 3) / 6. Used by MobileNetV3. -/
private def emitConvBnTrainHSwish (pidx pos : Nat) (curSSA : String) (curShape : List Nat)
    (ic oc kSize stride : Nat) : String × FwdRec := Id.run do
  match curShape with
  | [b, _, h, w] =>
    let oH := (h + stride - 1) / stride
    let oW := (w + stride - 1) / stride
    let outShape := [b, oc, oH, oW]
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    let mut s := ""
    -- Conv
    s := s ++ s!"    %cbn{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        window_strides = array<i64: {stride}, {stride}>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
    -- Batch norm
    let bnN := b * oH * oW
    s := s ++ s!"    %cbn_zf{pidx} = stablehlo.constant dense<0.0> : tensor<f32>\n"
    s := s ++ s!"    %cbn_ssp{pidx} = stablehlo.reduce(%cbn{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_sum{pidx} = stablehlo.reduce(%cbn_ssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_N{pidx} = stablehlo.constant dense<{bnN}.0> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean{pidx} = stablehlo.divide %cbn_sum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_mean_bc{pidx} = stablehlo.broadcast_in_dim %cbn_mean{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_diff{pidx} = stablehlo.subtract %cbn{pidx}, %cbn_mean_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_sq{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_diff{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_vssp{pidx} = stablehlo.reduce(%cbn_sq{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [2, 3]\n"
    s := s ++ s!"          : ({tensorTy outShape}, tensor<f32>) -> {tensorTy [b, oc]}\n"
    s := s ++ s!"    %cbn_vsum{pidx} = stablehlo.reduce(%cbn_vssp{pidx} init: %cbn_zf{pidx}) applies stablehlo.add across dimensions = [0]\n"
    s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_var{pidx} = stablehlo.divide %cbn_vsum{pidx}, %cbn_N{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_eps{pidx} = stablehlo.constant dense<1.0e-5> : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_ve{pidx} = stablehlo.add %cbn_var{pidx}, %cbn_eps{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd{pidx} = stablehlo.rsqrt %cbn_ve{pidx} : {tensorTy [oc]}\n"
    s := s ++ s!"    %cbn_istd_bc{pidx} = stablehlo.broadcast_in_dim %cbn_istd{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_norm{pidx} = stablehlo.multiply %cbn_diff{pidx}, %cbn_istd_bc{pidx} : {tensorTy outShape}\n"
    -- Affine
    s := s ++ s!"    %cbn_g_bc{pidx} = stablehlo.broadcast_in_dim %g{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_gn{pidx} = stablehlo.multiply %cbn_norm{pidx}, %cbn_g_bc{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_bt_bc{pidx} = stablehlo.broadcast_in_dim %bt{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
    let preSSA := s!"%cbn_pre{pidx}"
    s := s ++ s!"    {preSSA} = stablehlo.add %cbn_gn{pidx}, %cbn_bt_bc{pidx} : {tensorTy outShape}\n"
    -- h-swish: x * ReLU6(x + 3) / 6
    s := s ++ s!"    %cbn_hs3{pidx} = stablehlo.constant dense<3.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hs6{pidx} = stablehlo.constant dense<6.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsxp3{pidx} = stablehlo.add {preSSA}, %cbn_hs3{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsclip{pidx} = stablehlo.minimum %cbn_hsxp3{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsr6{pidx} = stablehlo.maximum %cbn_hsclip{pidx}, %cbn_hsz{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_hsdiv{pidx} = stablehlo.divide %cbn_hsr6{pidx}, %cbn_hs6{pidx} : {tensorTy outShape}\n"
    s := s ++ s!"    %cbn_out{pidx} = stablehlo.multiply {preSSA}, %cbn_hsdiv{pidx} : {tensorTy outShape}\n"
    let fwdRec : FwdRec := {
      layer := .convBn ic oc kSize stride .same
      pidx := some pidx, pos
      inputSSA := curSSA, preActSSA := preSSA, outputSSA := s!"%cbn_out{pidx}"
      inShape := curShape, outShape
      convOutSSA := s!"%cbn{pidx}"
      normSSA := s!"%cbn_norm{pidx}"
      meanBcSSA := s!"%cbn_mean_bc{pidx}"
      istdBcSSA := s!"%cbn_istd_bc{pidx}"
      hasRelu := false, hasRelu6 := false, hasSwish := false, hasHSwish := true
      isDepthwise := false
      ic := ic, kSize := kSize, stride := stride
    }
    return (s, fwdRec)
  | _ => return ("    // convBnHSwish error\n", default)

/-- Emit convBn backward: inst norm VJP + conv backward.
    Returns (code, gradient SSA name, gradient shape). -/
private def emitConvBnBackward (r : FwdRec) (gradSSA : String) : String × String × String := Id.run do
  let p := r.pidx.getD 0
  let oc := r.outShape[1]!
  let b := r.outShape[0]!
  let oH := r.outShape[2]!
  let oW := r.outShape[3]!
  let outTy := tensorTy r.outShape
  let _spatialN := oH * oW
  let mut s := ""
  -- Block header for the generated MLIR: cite the backing theorems so a
  -- reader of the .mlir file can jump straight from any op to its proof.
  s := s ++ s!"    // ════════════════════════════════════════════════════════════════\n"
  s := s ++ s!"    // ConvBn backward — see LeanMlir/Proofs/:\n"
  s := s ++ s!"    //   Activation VJP   MLP.lean: relu_has_vjp (ReLU); ReLU6/Swish/h-swish are analogous diagonal grads\n"
  s := s ++ s!"    //   BN VJP           BatchNorm.lean: bn_has_vjp + pdiv_bnNormalize (3-term consolidated formula)\n"
  s := s ++ s!"    //   Conv input grad  CNN.lean: conv2d_has_vjp3 (reversed kernel + transposed I/O channels)\n"
  s := s ++ s!"    //   Conv weight grad CNN.lean: conv2d_weight_grad_has_vjp (transpose trick; Phase 7)\n"
  s := s ++ s!"    // ════════════════════════════════════════════════════════════════\n"
  s := s ++ s!"    // ─── activation backward ───\n"
  -- ReLU / ReLU6 / Swish / h-swish backward
  -- NOTE: ReLU prefix is `cbg_rel` (not `cbg_relu`) to avoid collision with
  -- `cbg_relu6{p}` when a ReLU layer's pidx equals `6` followed by a ReLU6 layer's pidx.
  let effGrad := if r.hasRelu then s!"%cbg_rel{p}"
                 else if r.hasRelu6 then s!"%cbg_relu6{p}"
                 else if r.hasSwish then s!"%cbg_swish{p}"
                 else if r.hasHSwish then s!"%cbg_hswish{p}"
                 else gradSSA
  if r.hasRelu then
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_cmp{p} = stablehlo.compare GT, {r.preActSSA}, %cbn_z{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    {effGrad} = stablehlo.select %cbg_cmp{p}, {gradSSA}, %cbn_z{p} : {i1Ty}, {outTy}\n"
  else if r.hasRelu6 then
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_gt0{p} = stablehlo.compare GT, {r.preActSSA}, %cbn_z{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_lt6{p} = stablehlo.compare LT, {r.preActSSA}, %cbn_r6s{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_mask{p} = stablehlo.and %cbg_gt0{p}, %cbg_lt6{p} : {i1Ty}\n"
    s := s ++ s!"    {effGrad} = stablehlo.select %cbg_mask{p}, {gradSSA}, %cbn_z{p} : {i1Ty}, {outTy}\n"
  else if r.hasSwish then
    -- Swish backward: d/dx[x*σ(x)] = σ(x) + x*σ(x)*(1-σ(x)) = σ(x)*(1 + x*(1-σ(x)))
    -- Recompute sigmoid from pre-activation.
    s := s ++ s!"    %cbg_sig{p} = stablehlo.logistic {r.preActSSA} : {outTy}\n"
    s := s ++ s!"    %cbg_one{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_1ms{p} = stablehlo.subtract %cbg_one{p}, %cbg_sig{p} : {outTy}\n"
    s := s ++ s!"    %cbg_xt{p} = stablehlo.multiply {r.preActSSA}, %cbg_1ms{p} : {outTy}\n"
    s := s ++ s!"    %cbg_1px{p} = stablehlo.add %cbg_one{p}, %cbg_xt{p} : {outTy}\n"
    s := s ++ s!"    %cbg_dsw{p} = stablehlo.multiply %cbg_sig{p}, %cbg_1px{p} : {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_dsw{p} : {outTy}\n"
  else if r.hasHSwish then
    -- h-swish backward: d/dx[x * ReLU6(x+3)/6]
    --   = 0            if x ≤ -3
    --   = (2x + 3)/6   if -3 < x < 3
    --   = 1            if x ≥ 3
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_hsn3{p} = stablehlo.constant dense<-3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs3{p} = stablehlo.constant dense<3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs2{p} = stablehlo.constant dense<2.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs6{p} = stablehlo.constant dense<6.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs1{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs0{p} = stablehlo.constant dense<0.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hslt{p} = stablehlo.compare LT, {r.preActSSA}, %cbg_hsn3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hsgt{p} = stablehlo.compare GT, {r.preActSSA}, %cbg_hs3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hs2x{p} = stablehlo.multiply {r.preActSSA}, %cbg_hs2{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hs2xp3{p} = stablehlo.add %cbg_hs2x{p}, %cbg_hs3{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hsmid{p} = stablehlo.divide %cbg_hs2xp3{p}, %cbg_hs6{p} : {outTy}\n"
    -- If x < -3: 0 else mid. Then if x > 3: 1 else previous.
    s := s ++ s!"    %cbg_hsw1{p} = stablehlo.select %cbg_hslt{p}, %cbg_hs0{p}, %cbg_hsmid{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    %cbg_hsgrad{p} = stablehlo.select %cbg_hsgt{p}, %cbg_hs1{p}, %cbg_hsw1{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_hsgrad{p} : {outTy}\n"
  s := s ++ s!"    // ─── BN backward (BatchNorm.lean: bn_has_vjp) ───\n"
  -- d_gamma = reduce_sum(grad * norm, dims=[0,2,3])
  s := s ++ s!"    %cbg_gn{p} = stablehlo.multiply {effGrad}, {r.normSSA} : {outTy}\n"
  s := s ++ s!"    %d_g{p} = stablehlo.reduce(%cbg_gn{p} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [oc]}\n"
  -- d_beta = reduce_sum(grad, dims=[0,2,3])
  s := s ++ s!"    %d_bt{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [oc]}\n"
  -- d_norm = grad * gamma_broadcast
  s := s ++ s!"    %cbg_dnorm{p} = stablehlo.multiply {effGrad}, %cbn_g_bc{p} : {outTy}\n"
  -- Batch norm backward: d_conv_out
  -- d_xhat = d_norm; xhat = (x - mean) * istd; N = B * oH * oW
  -- d_x = (1/N) * istd * (N * d_xhat - sum(d_xhat, [0,2,3]) - xhat * sum(d_xhat * xhat, [0,2,3]))
  let bnN := b * oH * oW
  let Nf := s!"{bnN}.0"
  -- 2-step reduction: [2,3] then [0]
  s := s ++ s!"    %cbg_sdn_sp{p} = stablehlo.reduce(%cbg_dnorm{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, oc]}\n"
  s := s ++ s!"    %cbg_sdn{p} = stablehlo.reduce(%cbg_sdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
  s := s ++ s!"    %cbg_sdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sdn{p}, dims = [1] : ({tensorTy [oc]}) -> {outTy}\n"
  -- xhat * d_xhat
  s := s ++ s!"    %cbg_xdn{p} = stablehlo.multiply {r.normSSA}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_sxdn_sp{p} = stablehlo.reduce(%cbg_xdn{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, oc]}\n"
  s := s ++ s!"    %cbg_sxdn{p} = stablehlo.reduce(%cbg_sxdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, oc]}, tensor<f32>) -> {tensorTy [oc]}\n"
  s := s ++ s!"    %cbg_sxdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sxdn{p}, dims = [1] : ({tensorTy [oc]}) -> {outTy}\n"
  -- N * d_xhat - sum(d_xhat) - xhat * sum(d_xhat * xhat)
  s := s ++ s!"    %cbg_Nc{p} = stablehlo.constant dense<{Nf}> : {outTy}\n"
  s := s ++ s!"    %cbg_t1{p} = stablehlo.multiply %cbg_Nc{p}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t2{p} = stablehlo.subtract %cbg_t1{p}, %cbg_sdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t3{p} = stablehlo.multiply {r.normSSA}, %cbg_sxdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t4{p} = stablehlo.subtract %cbg_t2{p}, %cbg_t3{p} : {outTy}\n"
  -- d_conv_out = (1/N) * istd * result
  s := s ++ s!"    %cbg_t5{p} = stablehlo.multiply {r.istdBcSSA}, %cbg_t4{p} : {outTy}\n"
  s := s ++ s!"    %cbg_invN{p} = stablehlo.constant dense<{1.0 / bnN.toFloat}> : {outTy}\n"
  s := s ++ s!"    %cbg_dconv{p} = stablehlo.multiply %cbg_invN{p}, %cbg_t5{p} : {outTy}\n"
  s := s ++ s!"    // ─── conv weight grad (CNN.lean: conv2d_weight_grad_has_vjp) + input grad (conv2d_has_vjp3) ───\n"
  -- Conv backward: dW via transpose trick
  let ic := r.ic
  let kSize := r.kSize
  let stride := r.stride
  let h := r.inShape[2]!
  let w := r.inShape[3]!
  let inC := r.inShape[1]!
  if stride == 1 then
    -- Stride-1 case: simple transpose trick
    let pad := (kSize - 1) / 2
    s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [inC, b, h, w]}\n"
    s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [oc, b, oH, oW]}\n"
    s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
    s := s ++ convAttrBlock pad
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, h, w]}, {tensorTy [oc, b, oH, oW]}) -> {tensorTy [inC, oc, kSize, kSize]}\n"
    s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [inC, oc, kSize, kSize]}) -> {tensorTy [oc, inC, kSize, kSize]}\n"
    -- dx via reverse+transpose
    s := s ++ s!"    %cbg_Wt{p} = stablehlo.transpose %W{p}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %cbg_Wt{p}, dims = [2, 3] : {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ convAttrBlock pad
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [ic, oc, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  else
    -- Strided conv backward: dW computation
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    if kSize == 1 then
      -- 1×1 strided: subsample input at stride positions, then conv → 1×1
      s := s ++ s!"    %cbg_xsub{p} = \"stablehlo.slice\"({r.inputSSA}) " ++ "{"
      s := s ++ s!" start_indices = array<i64: 0, 0, 0, 0>,"
      s := s ++ s!" limit_indices = array<i64: {b}, {inC}, {h}, {w}>,"
      s := s ++ s!" strides = array<i64: 1, 1, {stride}, {stride}>"
      s := s ++ "}" ++ s!" : ({tensorTy r.inShape}) -> {tensorTy [b, inC, oH, oW]}\n"
      s := s ++ s!"    %cbg_bt_xs{p} = stablehlo.transpose %cbg_xsub{p}, dims = [1, 0, 2, 3] : ({tensorTy [b, inC, oH, oW]}) -> {tensorTy [inC, b, oH, oW]}\n"
      s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [oc, b, oH, oW]}\n"
      s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_xs{p}, %cbg_bt_g{p}) " ++ "{\n"
      s := s ++ convAttrBlock 0
      s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, oH, oW]}, {tensorTy [oc, b, oH, oW]}) -> {tensorTy [inC, oc, 1, 1]}\n"
      s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [inC, oc, 1, 1]}) -> {tensorTy [oc, inC, 1, 1]}\n"
    else
      -- kSize > 1 strided: use rhs_dilation
      s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [inC, b, h, w]}\n"
      s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [oc, b, oH, oW]}\n"
      s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
      s := s ++ convAttrBlockFull pH0 pH1 pW0 pW1 1 1 1 1 stride stride
      s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, h, w]}, {tensorTy [oc, b, oH, oW]}) -> {tensorTy [inC, oc, kSize, kSize]}\n"
      s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [inC, oc, kSize, kSize]}) -> {tensorTy [oc, inC, kSize, kSize]}\n"
    -- dx: conv(dy, W_rev, stride=1, lhs_dilation=stride, padding computed)
    let dxPH0 := kSize - 1 - pH0
    let dxPH1 := kSize - 1 - pH1 + (h + pH0 + pH1 - kSize) % stride
    let dxPW0 := kSize - 1 - pW0
    let dxPW1 := kSize - 1 - pW1 + (w + pW0 + pW1 - kSize) % stride
    s := s ++ s!"    %cbg_Wt{p} = stablehlo.transpose %W{p}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %cbg_Wt{p}, dims = [2, 3] : {tensorTy [ic, oc, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ convAttrBlockFull dxPH0 dxPH1 dxPW0 dxPW1 1 1 stride stride
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [ic, oc, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  return (s, s!"%cbg_dx{p}", s!"%d_W{p}")

/-- Emit depthwise convBn backward: BN VJP + depthwise conv backward.
    Weight shape: (channels, 1, kSize, kSize), feature_group_count = channels.
    Returns (code, gradient SSA name, dW SSA name). -/
private def emitDepthwiseConvBnBackward (r : FwdRec) (gradSSA : String) : String × String × String := Id.run do
  let p := r.pidx.getD 0
  let channels := r.outShape[1]!
  let b := r.outShape[0]!
  let oH := r.outShape[2]!
  let oW := r.outShape[3]!
  let outTy := tensorTy r.outShape
  let mut s := ""
  -- Block header for the generated MLIR: cite the backing theorems.
  s := s ++ s!"    // ════════════════════════════════════════════════════════════════\n"
  s := s ++ s!"    // Depthwise ConvBn backward — see LeanMlir/Proofs/:\n"
  s := s ++ s!"    //   Activation VJP        MLP.lean: relu_has_vjp (or ReLU6/Swish/h-swish variants)\n"
  s := s ++ s!"    //   BN VJP                BatchNorm.lean: bn_has_vjp + pdiv_bnNormalize\n"
  s := s ++ s!"    //   Depthwise input grad  Depthwise.lean: depthwise_has_vjp3 (per-channel reversed kernel)\n"
  s := s ++ s!"    //   Depthwise weight grad Depthwise.lean: depthwise_weight_grad_has_vjp3 (Phase 7; per-channel transpose trick)\n"
  s := s ++ s!"    // ════════════════════════════════════════════════════════════════\n"
  s := s ++ s!"    // ─── activation backward ───\n"
  -- ReLU / ReLU6 / Swish / h-swish backward
  let effGrad := if r.hasRelu then s!"%cbg_rel{p}"
                 else if r.hasRelu6 then s!"%cbg_relu6{p}"
                 else if r.hasSwish then s!"%cbg_swish{p}"
                 else if r.hasHSwish then s!"%cbg_hswish{p}"
                 else gradSSA
  if r.hasRelu then
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_cmp{p} = stablehlo.compare GT, {r.preActSSA}, %cbn_z{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    {effGrad} = stablehlo.select %cbg_cmp{p}, {gradSSA}, %cbn_z{p} : {i1Ty}, {outTy}\n"
  else if r.hasRelu6 then
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_gt0{p} = stablehlo.compare GT, {r.preActSSA}, %cbn_z{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_lt6{p} = stablehlo.compare LT, {r.preActSSA}, %cbn_r6s{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_mask{p} = stablehlo.and %cbg_gt0{p}, %cbg_lt6{p} : {i1Ty}\n"
    s := s ++ s!"    {effGrad} = stablehlo.select %cbg_mask{p}, {gradSSA}, %cbn_z{p} : {i1Ty}, {outTy}\n"
  else if r.hasSwish then
    -- Swish backward: see emitConvBnBackward comment.
    s := s ++ s!"    %cbg_sig{p} = stablehlo.logistic {r.preActSSA} : {outTy}\n"
    s := s ++ s!"    %cbg_one{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_1ms{p} = stablehlo.subtract %cbg_one{p}, %cbg_sig{p} : {outTy}\n"
    s := s ++ s!"    %cbg_xt{p} = stablehlo.multiply {r.preActSSA}, %cbg_1ms{p} : {outTy}\n"
    s := s ++ s!"    %cbg_1px{p} = stablehlo.add %cbg_one{p}, %cbg_xt{p} : {outTy}\n"
    s := s ++ s!"    %cbg_dsw{p} = stablehlo.multiply %cbg_sig{p}, %cbg_1px{p} : {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_dsw{p} : {outTy}\n"
  else if r.hasHSwish then
    -- h-swish backward: piecewise. See emitConvBnBackward.
    let i1Ty := outTy.replace "xf32>" "xi1>"
    s := s ++ s!"    %cbg_hsn3{p} = stablehlo.constant dense<-3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs3{p} = stablehlo.constant dense<3.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs2{p} = stablehlo.constant dense<2.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs6{p} = stablehlo.constant dense<6.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs1{p} = stablehlo.constant dense<1.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hs0{p} = stablehlo.constant dense<0.0> : {outTy}\n"
    s := s ++ s!"    %cbg_hslt{p} = stablehlo.compare LT, {r.preActSSA}, %cbg_hsn3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hsgt{p} = stablehlo.compare GT, {r.preActSSA}, %cbg_hs3{p} : ({outTy}, {outTy}) -> {i1Ty}\n"
    s := s ++ s!"    %cbg_hs2x{p} = stablehlo.multiply {r.preActSSA}, %cbg_hs2{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hs2xp3{p} = stablehlo.add %cbg_hs2x{p}, %cbg_hs3{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hsmid{p} = stablehlo.divide %cbg_hs2xp3{p}, %cbg_hs6{p} : {outTy}\n"
    s := s ++ s!"    %cbg_hsw1{p} = stablehlo.select %cbg_hslt{p}, %cbg_hs0{p}, %cbg_hsmid{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    %cbg_hsgrad{p} = stablehlo.select %cbg_hsgt{p}, %cbg_hs1{p}, %cbg_hsw1{p} : {i1Ty}, {outTy}\n"
    s := s ++ s!"    {effGrad} = stablehlo.multiply {gradSSA}, %cbg_hsgrad{p} : {outTy}\n"
  s := s ++ s!"    // ─── BN backward (BatchNorm.lean: bn_has_vjp) ───\n"
  -- d_gamma, d_beta (same as regular convBn)
  s := s ++ s!"    %cbg_gn{p} = stablehlo.multiply {effGrad}, {r.normSSA} : {outTy}\n"
  s := s ++ s!"    %d_g{p} = stablehlo.reduce(%cbg_gn{p} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %d_bt{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %cbg_dnorm{p} = stablehlo.multiply {effGrad}, %cbn_g_bc{p} : {outTy}\n"
  -- BN backward (same formula)
  let bnN := b * oH * oW
  let Nf := s!"{bnN}.0"
  s := s ++ s!"    %cbg_sdn_sp{p} = stablehlo.reduce(%cbg_dnorm{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, channels]}\n"
  s := s ++ s!"    %cbg_sdn{p} = stablehlo.reduce(%cbg_sdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %cbg_sdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sdn{p}, dims = [1] : ({tensorTy [channels]}) -> {outTy}\n"
  s := s ++ s!"    %cbg_xdn{p} = stablehlo.multiply {r.normSSA}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_sxdn_sp{p} = stablehlo.reduce(%cbg_xdn{p} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
  s := s ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [b, channels]}\n"
  s := s ++ s!"    %cbg_sxdn{p} = stablehlo.reduce(%cbg_sxdn_sp{p} init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({tensorTy [b, channels]}, tensor<f32>) -> {tensorTy [channels]}\n"
  s := s ++ s!"    %cbg_sxdn_bc{p} = stablehlo.broadcast_in_dim %cbg_sxdn{p}, dims = [1] : ({tensorTy [channels]}) -> {outTy}\n"
  s := s ++ s!"    %cbg_Nc{p} = stablehlo.constant dense<{Nf}> : {outTy}\n"
  s := s ++ s!"    %cbg_t1{p} = stablehlo.multiply %cbg_Nc{p}, %cbg_dnorm{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t2{p} = stablehlo.subtract %cbg_t1{p}, %cbg_sdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t3{p} = stablehlo.multiply {r.normSSA}, %cbg_sxdn_bc{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t4{p} = stablehlo.subtract %cbg_t2{p}, %cbg_t3{p} : {outTy}\n"
  s := s ++ s!"    %cbg_t5{p} = stablehlo.multiply {r.istdBcSSA}, %cbg_t4{p} : {outTy}\n"
  s := s ++ s!"    %cbg_invN{p} = stablehlo.constant dense<{1.0 / bnN.toFloat}> : {outTy}\n"
  s := s ++ s!"    %cbg_dconv{p} = stablehlo.multiply %cbg_invN{p}, %cbg_t5{p} : {outTy}\n"
  s := s ++ s!"    // ─── depthwise weight grad (Depthwise.lean: depthwise_weight_grad_has_vjp3) + input grad (depthwise_has_vjp3) ───\n"
  -- Depthwise conv backward
  let kSize := r.kSize
  let stride := r.stride
  let h := r.inShape[2]!
  let w := r.inShape[3]!
  -- dW: transpose input [B,C,H,W]->[C,B,H,W], transpose grad [B,C,oH,oW]->[C,B,oH,oW]
  --     conv with batch_group_count=channels (each channel independently)
  --     For depthwise: dW[c,1,kH,kW] = sum_b input[b,c,:,:] conv grad[b,c,:,:]
  if stride == 1 then
    let pad := (kSize - 1) / 2
    s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [channels, b, h, w]}\n"
    s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [channels, b, oH, oW]}\n"
    s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
    s := s ++ s!"        batch_group_count = {channels} : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [channels, b, h, w]}, {tensorTy [channels, b, oH, oW]}) -> {tensorTy [1, channels, kSize, kSize]}\n"
    s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [1, channels, kSize, kSize]}) -> {tensorTy [channels, 1, kSize, kSize]}\n"
    -- dx: reverse kernel, conv with feature_group_count=channels
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %W{p}, dims = [2, 3] : {tensorTy [channels, 1, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  else
    -- Strided depthwise backward
    let (pH0, pH1, pW0, pW1) := samePad h w kSize stride
    -- dW: transpose + conv with batch_group_count=channels, rhs_dilation=stride
    s := s ++ s!"    %cbg_bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [channels, b, h, w]}\n"
    s := s ++ s!"    %cbg_bt_g{p} = stablehlo.transpose %cbg_dconv{p}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [channels, b, oH, oW]}\n"
    s := s ++ s!"    %cbg_dWr{p} = \"stablehlo.convolution\"(%cbg_bt_in{p}, %cbg_bt_g{p}) " ++ "{\n"
    s := s ++ s!"        batch_group_count = {channels} : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ "        feature_group_count = 1 : i64,\n"
    s := s ++ s!"        lhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ s!"        rhs_dilation = array<i64: {stride}, {stride}>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({tensorTy [channels, b, h, w]}, {tensorTy [channels, b, oH, oW]}) -> {tensorTy [1, channels, kSize, kSize]}\n"
    s := s ++ s!"    %d_W{p} = stablehlo.transpose %cbg_dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy [1, channels, kSize, kSize]}) -> {tensorTy [channels, 1, kSize, kSize]}\n"
    -- dx: reverse kernel, conv with feature_group_count=channels, lhs_dilation=stride
    let dxPH0 := kSize - 1 - pH0
    let dxPH1 := kSize - 1 - pH1 + (h + pH0 + pH1 - kSize) % stride
    let dxPW0 := kSize - 1 - pW0
    let dxPW1 := kSize - 1 - pW1 + (w + pW0 + pW1 - kSize) % stride
    s := s ++ s!"    %cbg_Wrev{p} = stablehlo.reverse %W{p}, dims = [2, 3] : {tensorTy [channels, 1, kSize, kSize]}\n"
    s := s ++ s!"    %cbg_dx{p} = \"stablehlo.convolution\"(%cbg_dconv{p}, %cbg_Wrev{p}) " ++ "{\n"
    s := s ++ "        batch_group_count = 1 : i64,\n"
    s := s ++ convDimNumbers
    s := s ++ s!"        feature_group_count = {channels} : i64,\n"
    s := s ++ s!"        lhs_dilation = array<i64: {stride}, {stride}>,\n"
    s := s ++ s!"        padding = dense<[[{dxPH0}, {dxPH1}], [{dxPW0}, {dxPW1}]]> : tensor<2x2xi64>,\n"
    s := s ++ "        rhs_dilation = array<i64: 1, 1>,\n"
    s := s ++ "        window_strides = array<i64: 1, 1>\n"
    s := s ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [channels, 1, kSize, kSize]}) -> {tensorTy r.inShape}\n"
  return (s, s!"%cbg_dx{p}", s!"%d_W{p}")

/-- Emit Adam update for one param.
    m_new = β1*m + (1-β1)*grad
    v_new = β2*v + (1-β2)*grad²
    m_hat = m_new / bc1    (bc1 = 1 - β1^t, precomputed)
    v_hat = v_new / bc2    (bc2 = 1 - β2^t, precomputed)
    w_new = w - lr * m_hat / (sqrt(v_hat) + ε)
    + optional decoupled weight decay: w_new = w_new - wd*lr*w -/
private def emitAdamUpdate (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String)
    (applyWeightDecay : Bool := false) (clipScale : Option String := none) (lrSSA : String := "%lr")
    : String × String × String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  -- Optional global-norm gradient clipping: pre-scale the gradient by the
  -- precomputed clip factor. When clipScale = none the emitted IR is identical
  -- to the unclipped path (no behavior change for existing nets).
  let mut g := gradSSA
  match clipScale with
  | some sc =>
    s := s ++ s!"    %gclbc_{tag} = stablehlo.broadcast_in_dim {sc}, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %gcl_{tag} = stablehlo.multiply {gradSSA}, %gclbc_{tag} : {ty}\n"
    g := s!"%gcl_{tag}"
  | none => pure ()
  -- m_new = β1*m + (1-β1)*grad
  s := s ++ s!"    %b1_{tag} = stablehlo.broadcast_in_dim %beta1, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %ob1_{tag} = stablehlo.broadcast_in_dim %one_minus_b1, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %ms_{tag} = stablehlo.multiply %b1_{tag}, {mSSA} : {ty}\n"
  s := s ++ s!"    %mg_{tag} = stablehlo.multiply %ob1_{tag}, {g} : {ty}\n"
  s := s ++ s!"    %mn_{tag} = stablehlo.add %ms_{tag}, %mg_{tag} : {ty}\n"
  -- v_new = β2*v + (1-β2)*grad²
  s := s ++ s!"    %b2_{tag} = stablehlo.broadcast_in_dim %beta2, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %ob2_{tag} = stablehlo.broadcast_in_dim %one_minus_b2, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %vsc_{tag} = stablehlo.multiply %b2_{tag}, {vSSA} : {ty}\n"
  s := s ++ s!"    %g2_{tag} = stablehlo.multiply {g}, {g} : {ty}\n"
  s := s ++ s!"    %vg_{tag} = stablehlo.multiply %ob2_{tag}, %g2_{tag} : {ty}\n"
  s := s ++ s!"    %vn_{tag} = stablehlo.add %vsc_{tag}, %vg_{tag} : {ty}\n"
  -- Bias correction: m_hat = m_new / bc1, v_hat = v_new / bc2
  s := s ++ s!"    %bc1_{tag} = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %bc2_{tag} = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %mh_{tag} = stablehlo.divide %mn_{tag}, %bc1_{tag} : {ty}\n"
  s := s ++ s!"    %vh_{tag} = stablehlo.divide %vn_{tag}, %bc2_{tag} : {ty}\n"
  -- w_new = w - lr * m_hat / (sqrt(v_hat) + ε)
  s := s ++ s!"    %lr_{tag} = stablehlo.broadcast_in_dim {lrSSA}, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %eps_{tag} = stablehlo.broadcast_in_dim %adam_eps, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %sqv_{tag} = stablehlo.sqrt %vh_{tag} : {ty}\n"
  s := s ++ s!"    %den_{tag} = stablehlo.add %sqv_{tag}, %eps_{tag} : {ty}\n"
  s := s ++ s!"    %rat_{tag} = stablehlo.divide %mh_{tag}, %den_{tag} : {ty}\n"
  s := s ++ s!"    %upd_{tag} = stablehlo.multiply %lr_{tag}, %rat_{tag} : {ty}\n"
  s := s ++ s!"    %sub_{tag} = stablehlo.subtract {paramSSA}, %upd_{tag} : {ty}\n"
  if applyWeightDecay then
    -- Decoupled weight decay: w = w - lr*m_hat/(sqrt(v_hat)+ε) - wd*lr*w
    s := s ++ s!"    %wd_{tag} = stablehlo.broadcast_in_dim %wdecay, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %wdlr_{tag} = stablehlo.multiply %wd_{tag}, %lr_{tag} : {ty}\n"
    s := s ++ s!"    %wdp_{tag} = stablehlo.multiply %wdlr_{tag}, {paramSSA} : {ty}\n"
    s := s ++ s!"    %new_{tag} = stablehlo.subtract %sub_{tag}, %wdp_{tag} : {ty}\n"
  return (s, if applyWeightDecay then s!"%new_{tag}" else s!"%sub_{tag}", s!"%mn_{tag}", s!"%vn_{tag}")

/-- Emit SGD+momentum update for one param.
    v_new = mu * v + grad
    w_new = w - lr * v_new
    + optional decoupled weight decay: w_new = w_new - wd*lr*w -/
private def emitMomentumUpdate (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String)
    (applyWeightDecay : Bool := false) (clipScale : Option String := none) (lrSSA : String := "%lr")
    : String × String × String × String := Id.run do
  let ty := tensorTy shape
  let mut s := ""
  -- Optional global-norm gradient clipping (see emitAdamUpdate). none ⇒ identical IR.
  let mut g := gradSSA
  match clipScale with
  | some sc =>
    s := s ++ s!"    %gclbc_{tag} = stablehlo.broadcast_in_dim {sc}, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %gcl_{tag} = stablehlo.multiply {gradSSA}, %gclbc_{tag} : {ty}\n"
    g := s!"%gcl_{tag}"
  | none => pure ()
  -- v_new = mu * velocity + grad
  s := s ++ s!"    %mu_{tag} = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %vs_{tag} = stablehlo.multiply %mu_{tag}, {mSSA} : {ty}\n"
  s := s ++ s!"    %vn_{tag} = stablehlo.add %vs_{tag}, {g} : {ty}\n"
  -- w_new = w - lr * v_new
  s := s ++ s!"    %lr_{tag} = stablehlo.broadcast_in_dim {lrSSA}, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %up_{tag} = stablehlo.multiply %lr_{tag}, %vn_{tag} : {ty}\n"
  s := s ++ s!"    %sub_{tag} = stablehlo.subtract {paramSSA}, %up_{tag} : {ty}\n"
  if applyWeightDecay then
    -- Decoupled weight decay: w = w - lr*v_new - wd*lr*w
    s := s ++ s!"    %wd_{tag} = stablehlo.broadcast_in_dim %wdecay, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %wdlr_{tag} = stablehlo.multiply %wd_{tag}, %lr_{tag} : {ty}\n"
    s := s ++ s!"    %wdp_{tag} = stablehlo.multiply %wdlr_{tag}, {paramSSA} : {ty}\n"
    s := s ++ s!"    %new_{tag} = stablehlo.subtract %sub_{tag}, %wdp_{tag} : {ty}\n"
  -- mNew = velocity (stored in m_ slot), vPassthrough = original v_ unchanged
  return (s, if applyWeightDecay then s!"%new_{tag}" else s!"%sub_{tag}", s!"%vn_{tag}", vSSA)

/-- Emit a **Muon** update for one 2D weight matrix `W : [m, n]` (UNVERIFIED perf path).

    Momentum-Orthogonalized-by-Newton–Schulz (Jordan 2024): the heavy-ball momentum
    buffer is polar-projected onto the (semi-)orthogonal matrices — `G = UΣVᵀ ↦ UVᵀ`,
    the nearest orthogonal matrix — by a fixed 5-step Newton–Schulz iteration, so every
    singular direction gets an equal-size step.
    ```
      buf = μ·m + grad                                  -- heavy-ball momentum (m-slot)
      X   = buf / (‖buf‖_F + 1e-7)                       -- normalize (σ ≤ 1)
      repeat 5×:  A = X·Xᵀ;  X = a·X + b·(A·X) + c·(A·(A·X))   -- (a,b,c) = NS5 coeffs
      W   = W − lr·s·X                                   -- s = max(1, m/n)^½ (RMS match)
    ```
    `X·Xᵀ` is built with `dot_general contracting_dims = [1] x [1]` (no transpose op);
    the whole iteration is pure `dot_general` + scalar ops — same matmul vocabulary the
    forward/attention path already emits, which is exactly why it renders cleanly (and,
    per `planning/muon.md`, why a `den=` render-faithful tie is reachable). The momentum
    buffer lives in the m-slot and the v-slot is an unused passthrough (like
    `emitMomentumUpdate`), so the train-step signature/arity is unchanged. The NS
    constants `%ns_a/%ns_b/%ns_c/%ns_eps` and `%mu` are emitted once in the optimizer
    header. The `max(1,m/n)^½` shape-scale is a compile-time constant (`m`,`n` known here).
    NOTE: this is the standard Muon shape-scale; descent is NOT proved (Muon's guarantee is
    spectral-norm steepest descent, not the `lr·‖∇‖²/2` Frobenius bound) — render-faithful
    only, by design. -/
private def emitMuonUpdate (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String)
    (applyWeightDecay : Bool := false) (clipScale : Option String := none) (lrSSA : String := "%lr")
    : String × String × String × String := Id.run do
  let m := shape.headD 1
  let n := shape.getD 1 1
  let ty := tensorTy [m, n]
  let tyA := tensorTy [m, m]
  let ratio : Float := m.toFloat / n.toFloat
  let scale : Float := Float.sqrt (if ratio > 1.0 then ratio else 1.0)
  let mut s := ""
  -- optional global-norm gradient clipping (see emitAdamUpdate; none ⇒ identical IR)
  let mut g := gradSSA
  match clipScale with
  | some sc =>
    s := s ++ s!"    %gclbc_{tag} = stablehlo.broadcast_in_dim {sc}, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %gcl_{tag} = stablehlo.multiply {gradSSA}, %gclbc_{tag} : {ty}\n"
    g := s!"%gcl_{tag}"
  | none => pure ()
  -- heavy-ball momentum (stored in m-slot): buf = μ·m + grad
  s := s ++ s!"    %mmu_{tag} = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %mvs_{tag} = stablehlo.multiply %mmu_{tag}, {mSSA} : {ty}\n"
  s := s ++ s!"    %buf_{tag} = stablehlo.add %mvs_{tag}, {g} : {ty}\n"
  -- normalize: X0 = buf / (‖buf‖_F + ε)   (‖·‖_F ≥ spectral, so σ ≤ 1)
  s := s ++ s!"    %nsq_{tag} = stablehlo.multiply %buf_{tag}, %buf_{tag} : {ty}\n"
  s := s ++ s!"    %nsum_{tag} = stablehlo.reduce(%nsq_{tag} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
  s := s ++ s!"          : ({ty}, tensor<f32>) -> tensor<f32>\n"
  s := s ++ s!"    %nrm_{tag} = stablehlo.sqrt %nsum_{tag} : tensor<f32>\n"
  s := s ++ s!"    %nrme_{tag} = stablehlo.add %nrm_{tag}, %ns_eps : tensor<f32>\n"
  s := s ++ s!"    %nrmb_{tag} = stablehlo.broadcast_in_dim %nrme_{tag}, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %X0_{tag} = stablehlo.divide %buf_{tag}, %nrmb_{tag} : {ty}\n"
  -- broadcast the NS5 coefficients once
  s := s ++ s!"    %nsa_{tag} = stablehlo.broadcast_in_dim %ns_a, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %nsb_{tag} = stablehlo.broadcast_in_dim %ns_b, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %nsc_{tag} = stablehlo.broadcast_in_dim %ns_c, dims = [] : (tensor<f32>) -> {ty}\n"
  -- 5 Newton–Schulz iterations:  A = X·Xᵀ;  X ← a·X + b·(A·X) + c·(A·(A·X))
  let mut x := s!"%X0_{tag}"
  for it in [0,1,2,3,4] do
    let xn := s!"%X{it+1}_{tag}"
    s := s ++ s!"    %A{it}_{tag} = stablehlo.dot_general {x}, {x},\n"
    s := s ++ "              contracting_dims = [1] x [1],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({ty}, {ty}) -> {tyA}\n"
    s := s ++ s!"    %AX{it}_{tag} = stablehlo.dot_general %A{it}_{tag}, {x},\n"
    s := s ++ "              contracting_dims = [1] x [0],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({tyA}, {ty}) -> {ty}\n"
    s := s ++ s!"    %AAX{it}_{tag} = stablehlo.dot_general %A{it}_{tag}, %AX{it}_{tag},\n"
    s := s ++ "              contracting_dims = [1] x [0],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({tyA}, {ty}) -> {ty}\n"
    s := s ++ s!"    %aX{it}_{tag} = stablehlo.multiply %nsa_{tag}, {x} : {ty}\n"
    s := s ++ s!"    %bAX{it}_{tag} = stablehlo.multiply %nsb_{tag}, %AX{it}_{tag} : {ty}\n"
    s := s ++ s!"    %cAAX{it}_{tag} = stablehlo.multiply %nsc_{tag}, %AAX{it}_{tag} : {ty}\n"
    s := s ++ s!"    %xt{it}_{tag} = stablehlo.add %aX{it}_{tag}, %bAX{it}_{tag} : {ty}\n"
    s := s ++ s!"    {xn} = stablehlo.add %xt{it}_{tag}, %cAAX{it}_{tag} : {ty}\n"
    x := xn
  -- W_new = W − lr·s·X   (s = compile-time RMS shape-match scale)
  s := s ++ s!"    %scl_{tag} = stablehlo.constant dense<{scale}> : tensor<f32>\n"
  s := s ++ s!"    %lrs_{tag} = stablehlo.multiply {lrSSA}, %scl_{tag} : tensor<f32>\n"
  s := s ++ s!"    %lrsb_{tag} = stablehlo.broadcast_in_dim %lrs_{tag}, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %upd_{tag} = stablehlo.multiply %lrsb_{tag}, {x} : {ty}\n"
  s := s ++ s!"    %sub_{tag} = stablehlo.subtract {paramSSA}, %upd_{tag} : {ty}\n"
  if applyWeightDecay then
    -- decoupled weight decay: w = w − lr·s·X − wd·lr·w
    s := s ++ s!"    %lrb_{tag} = stablehlo.broadcast_in_dim {lrSSA}, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %wd_{tag} = stablehlo.broadcast_in_dim %wdecay, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %wdlr_{tag} = stablehlo.multiply %wd_{tag}, %lrb_{tag} : {ty}\n"
    s := s ++ s!"    %wdp_{tag} = stablehlo.multiply %wdlr_{tag}, {paramSSA} : {ty}\n"
    s := s ++ s!"    %new_{tag} = stablehlo.subtract %sub_{tag}, %wdp_{tag} : {ty}\n"
  -- buf in m-slot, v-slot passthrough (arity unchanged)
  return (s, if applyWeightDecay then s!"%new_{tag}" else s!"%sub_{tag}", s!"%buf_{tag}", vSSA)

/-- Matmul-only inverse square root of a symmetric PSD matrix `M : [n,n]`
    by the coupled-Newton iteration (Higham). Returns `(code, ssa)` where
    `ssa ≈ M^{-1/2}`. Trace-scaling `Ms = M/tr(M)` puts every eigenvalue in
    `(0,1] ⊂ (0,3)` — the iteration's convergence basin (tr ≥ λ_max):

        Y₀ = Ms,  Z₀ = I;   T = 1.5·I − 0.5·(Z·Y);   Y ← Y·T;  Z ← T·Z
        Y → Ms^{1/2},  Z → Ms^{-1/2};  then M^{-1/2} = Z / √tr.

    `identSSA` is a prebuilt `[n,n]` identity; `%one_scalar`/`%zf` are the
    module-level scalar 1.0/0.0. Pure `dot_general` + elementwise — the same
    class of ops as Muon's polar NS. -/
private def emitInvSqrtNS (tag mSSA identSSA : String) (n iters : Nat)
    : String × String := Id.run do
  let ty := tensorTy [n, n]
  let mut s := ""
  -- trace(M) = Σ (M ⊙ I)
  s := s ++ s!"    %shtrm_{tag} = stablehlo.multiply {mSSA}, {identSSA} : {ty}\n"
  s := s ++ s!"    %shtr_{tag} = stablehlo.reduce(%shtrm_{tag} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
  s := s ++ s!"          : ({ty}, tensor<f32>) -> tensor<f32>\n"
  -- Ms = M / tr
  s := s ++ s!"    %shtri_{tag} = stablehlo.divide %one_scalar, %shtr_{tag} : tensor<f32>\n"
  s := s ++ s!"    %shtrib_{tag} = stablehlo.broadcast_in_dim %shtri_{tag}, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %shMs_{tag} = stablehlo.multiply {mSSA}, %shtrib_{tag} : {ty}\n"
  -- NS constants broadcast to [n,n]: 1.5·I (fixed) and the scalar 0.5.
  s := s ++ s!"    %shhalf_{tag} = stablehlo.broadcast_in_dim %sh_half, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %sh1p5_{tag} = stablehlo.broadcast_in_dim %sh_1p5, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %sh1p5I_{tag} = stablehlo.multiply %sh1p5_{tag}, {identSSA} : {ty}\n"
  -- coupled-Newton iteration
  let mut y := s!"%shMs_{tag}"
  let mut z := identSSA
  for it in [:iters] do
    s := s ++ s!"    %shZY{it}_{tag} = stablehlo.dot_general {z}, {y},\n"
    s := s ++ "              contracting_dims = [1] x [0],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
    s := s ++ s!"    %shhZY{it}_{tag} = stablehlo.multiply %shhalf_{tag}, %shZY{it}_{tag} : {ty}\n"
    s := s ++ s!"    %shT{it}_{tag} = stablehlo.subtract %sh1p5I_{tag}, %shhZY{it}_{tag} : {ty}\n"
    s := s ++ s!"    %shY{it+1}_{tag} = stablehlo.dot_general {y}, %shT{it}_{tag},\n"
    s := s ++ "              contracting_dims = [1] x [0],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
    s := s ++ s!"    %shZ{it+1}_{tag} = stablehlo.dot_general %shT{it}_{tag}, {z},\n"
    s := s ++ "              contracting_dims = [1] x [0],\n"
    s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
    s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
    y := s!"%shY{it+1}_{tag}"
    z := s!"%shZ{it+1}_{tag}"
  -- result = Z / √tr
  s := s ++ s!"    %shsqtr_{tag} = stablehlo.sqrt %shtr_{tag} : tensor<f32>\n"
  s := s ++ s!"    %shisqtr_{tag} = stablehlo.divide %one_scalar, %shsqtr_{tag} : tensor<f32>\n"
  s := s ++ s!"    %shisqtrb_{tag} = stablehlo.broadcast_in_dim %shisqtr_{tag}, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %shInvSqrt_{tag} = stablehlo.multiply {z}, %shisqtrb_{tag} : {ty}\n"
  return (s, s!"%shInvSqrt_{tag}")

/-- Inverse fourth root `M^{-1/4}` by two `emitInvSqrtNS` passes:
    `A = M^{-1/2}`, `B = A^{-1/2} = M^{1/4}`, and `M^{-1/4} = A·B`. All matmul. -/
private def emitInvFourthRootNS (tag mSSA identSSA : String) (n iters : Nat)
    : String × String := Id.run do
  let ty := tensorTy [n, n]
  let (sA, aSSA) := emitInvSqrtNS s!"{tag}a" mSSA identSSA n iters
  let (sB, bSSA) := emitInvSqrtNS s!"{tag}b" aSSA identSSA n iters
  let mut s := sA ++ sB
  s := s ++ s!"    %shI4_{tag} = stablehlo.dot_general {aSSA}, {bSSA},\n"
  s := s ++ "              contracting_dims = [1] x [0],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
  return (s, s!"%shI4_{tag}")

/-- Emit a Shampoo update for a **square** 2D weight `W : [n,n]` (Gupta–Koren–
    Singer 2018). The L/R accumulators reuse the m-slot / v-slot (both `[n,n]`
    for a square weight), so the train-step signature is unchanged — the same
    trick Muon uses for its momentum. EMA-accumulate

        L ← β·L + (1−β)·G·Gᵀ,   R ← β·R + (1−β)·Gᵀ·G,

    regularize with a trace-relative `εI` (`ε·tr/n` — scale-invariant, and it
    conditions the ill-posed first few steps), form `W ← W − lr·L^{-1/4}·G·R^{-1/4}`
    via the matmul-only inverse-4th-root, and return `(code, W', L', R')` with
    `L'`/`R'` going back to the m/v slots. Single-step (β=0) Shampoo = Muon's
    `UVᵀ` — the jewel; accumulation is the memory knob. See `planning/shampoo.md`. -/
private def emitShampooUpdate (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String)
    (applyWeightDecay : Bool := false) (clipScale : Option String := none) (lrSSA : String := "%lr")
    (nsIters : Nat := 15) : String × String × String × String := Id.run do
  let n := shape.headD 1
  let ty := tensorTy [n, n]
  let mut s := ""
  -- optional global-norm gradient clipping (matches emitMuonUpdate)
  let mut g := gradSSA
  match clipScale with
  | some sc =>
    s := s ++ s!"    %shgclbc_{tag} = stablehlo.broadcast_in_dim {sc}, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %shgcl_{tag} = stablehlo.multiply {gradSSA}, %shgclbc_{tag} : {ty}\n"
    g := s!"%shgcl_{tag}"
  | none => pure ()
  -- identity [n,n] (via iota compare) — shared by trace, NS, and eps-reg
  s := s ++ s!"    %shir_{tag} = stablehlo.iota dim = 0 : tensor<{n}x{n}xi32>\n"
  s := s ++ s!"    %shic_{tag} = stablehlo.iota dim = 1 : tensor<{n}x{n}xi32>\n"
  s := s ++ s!"    %shieq_{tag} = stablehlo.compare EQ, %shir_{tag}, %shic_{tag} : (tensor<{n}x{n}xi32>, tensor<{n}x{n}xi32>) -> tensor<{n}x{n}xi1>\n"
  s := s ++ s!"    %shI1_{tag} = stablehlo.constant dense<1.0> : {ty}\n"
  s := s ++ s!"    %shI0_{tag} = stablehlo.constant dense<0.0> : {ty}\n"
  s := s ++ s!"    %shId_{tag} = stablehlo.select %shieq_{tag}, %shI1_{tag}, %shI0_{tag} : tensor<{n}x{n}xi1>, {ty}\n"
  -- G·Gᵀ  (contract dim1) and  Gᵀ·G  (contract dim0)
  s := s ++ s!"    %shGGt_{tag} = stablehlo.dot_general {g}, {g},\n"
  s := s ++ "              contracting_dims = [1] x [1],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
  s := s ++ s!"    %shGtG_{tag} = stablehlo.dot_general {g}, {g},\n"
  s := s ++ "              contracting_dims = [0] x [0],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
  -- EMA: L' = β·L + (1−β)·GGᵀ,  R' = β·R + (1−β)·GᵀG   (β from %sh_beta)
  s := s ++ s!"    %shbetab_{tag} = stablehlo.broadcast_in_dim %sh_beta, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %shombb_{tag} = stablehlo.broadcast_in_dim %sh_ombeta, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %shbL_{tag} = stablehlo.multiply %shbetab_{tag}, {mSSA} : {ty}\n"
  s := s ++ s!"    %shoGGt_{tag} = stablehlo.multiply %shombb_{tag}, %shGGt_{tag} : {ty}\n"
  s := s ++ s!"    %shLnew_{tag} = stablehlo.add %shbL_{tag}, %shoGGt_{tag} : {ty}\n"
  s := s ++ s!"    %shbR_{tag} = stablehlo.multiply %shbetab_{tag}, {vSSA} : {ty}\n"
  s := s ++ s!"    %shoGtG_{tag} = stablehlo.multiply %shombb_{tag}, %shGtG_{tag} : {ty}\n"
  s := s ++ s!"    %shRnew_{tag} = stablehlo.add %shbR_{tag}, %shoGtG_{tag} : {ty}\n"
  -- trace-relative εI regularization: M_reg = M' + (ε·tr(M')/n)·I
  let emitReg := fun (subtag mNewSSA : String) => Id.run do
    let nInv : Float := 1.0 / n.toFloat
    let mut r := ""
    r := r ++ s!"    %shrtm_{subtag} = stablehlo.multiply {mNewSSA}, %shId_{tag} : {ty}\n"
    r := r ++ s!"    %shrtr_{subtag} = stablehlo.reduce(%shrtm_{subtag} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
    r := r ++ s!"          : ({ty}, tensor<f32>) -> tensor<f32>\n"
    r := r ++ s!"    %shninv_{subtag} = stablehlo.constant dense<{nInv}> : tensor<f32>\n"
    r := r ++ s!"    %shrtn_{subtag} = stablehlo.multiply %shrtr_{subtag}, %shninv_{subtag} : tensor<f32>\n"
    r := r ++ s!"    %shepsc_{subtag} = stablehlo.multiply %shrtn_{subtag}, %sh_eps : tensor<f32>\n"
    r := r ++ s!"    %shepscb_{subtag} = stablehlo.broadcast_in_dim %shepsc_{subtag}, dims = [] : (tensor<f32>) -> {ty}\n"
    r := r ++ s!"    %shepsI_{subtag} = stablehlo.multiply %shepscb_{subtag}, %shId_{tag} : {ty}\n"
    r := r ++ s!"    %shreg_{subtag} = stablehlo.add {mNewSSA}, %shepsI_{subtag} : {ty}\n"
    (r, s!"%shreg_{subtag}")
  let (sLr, lRegSSA) := emitReg s!"L{tag}" s!"%shLnew_{tag}"
  let (sRr, rRegSSA) := emitReg s!"R{tag}" s!"%shRnew_{tag}"
  s := s ++ sLr ++ sRr
  -- L^{-1/4}, R^{-1/4}
  let (sLi, liSSA) := emitInvFourthRootNS s!"L{tag}" lRegSSA s!"%shId_{tag}" n nsIters
  let (sRi, riSSA) := emitInvFourthRootNS s!"R{tag}" rRegSSA s!"%shId_{tag}" n nsIters
  s := s ++ sLi ++ sRi
  -- P = L^{-1/4}·G·R^{-1/4}
  s := s ++ s!"    %shLiG_{tag} = stablehlo.dot_general {liSSA}, {g},\n"
  s := s ++ "              contracting_dims = [1] x [0],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
  s := s ++ s!"    %shP_{tag} = stablehlo.dot_general %shLiG_{tag}, {riSSA},\n"
  s := s ++ "              contracting_dims = [1] x [0],\n"
  s := s ++ "              precision = [DEFAULT, DEFAULT]\n"
  s := s ++ s!"            : ({ty}, {ty}) -> {ty}\n"
  -- W_new = W − lr·P   (the inverse roots normalize the step; no extra scale)
  s := s ++ s!"    %shlrb_{tag} = stablehlo.broadcast_in_dim {lrSSA}, dims = [] : (tensor<f32>) -> {ty}\n"
  s := s ++ s!"    %shupd_{tag} = stablehlo.multiply %shlrb_{tag}, %shP_{tag} : {ty}\n"
  s := s ++ s!"    %shsub_{tag} = stablehlo.subtract {paramSSA}, %shupd_{tag} : {ty}\n"
  if applyWeightDecay then
    s := s ++ s!"    %shwd_{tag} = stablehlo.broadcast_in_dim %wdecay, dims = [] : (tensor<f32>) -> {ty}\n"
    s := s ++ s!"    %shwdlr_{tag} = stablehlo.multiply %shwd_{tag}, %shlrb_{tag} : {ty}\n"
    s := s ++ s!"    %shwdp_{tag} = stablehlo.multiply %shwdlr_{tag}, {paramSSA} : {ty}\n"
    s := s ++ s!"    %shnew_{tag} = stablehlo.subtract %shsub_{tag}, %shwdp_{tag} : {ty}\n"
  -- L' → m-slot, R' → v-slot (both [n,n]; arity unchanged)
  return (s, if applyWeightDecay then s!"%shnew_{tag}" else s!"%shsub_{tag}", s!"%shLnew_{tag}", s!"%shRnew_{tag}")

/-- Emit Adam update for a convBn layer (W, gamma, beta). -/
private def emitConvBnAdam (p ic oc kSize : Nat) (applyWeightDecay : Bool := true)
    (clipScale : Option String := none) : String × Array String × Array String := Id.run do
  let wShape := [oc, ic, kSize, kSize]
  let bShape := [oc]
  let mut s := ""
  let (s1, wNew, mwNew, vwNew) := emitAdamUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"W{p}" (applyWeightDecay := applyWeightDecay) (clipScale := clipScale)
  s := s ++ s1
  let (s2, gNew, mgNew, vgNew) := emitAdamUpdate s!"%g{p}" s!"%d_g{p}" s!"%m_g{p}" s!"%v_g{p}" bShape s!"g{p}" (clipScale := clipScale)
  s := s ++ s2
  let (s3, btNew, mbtNew, vbtNew) := emitAdamUpdate s!"%bt{p}" s!"%d_bt{p}" s!"%m_bt{p}" s!"%v_bt{p}" bShape s!"bt{p}" (clipScale := clipScale)
  s := s ++ s3
  let retNames := #[wNew, gNew, btNew, mwNew, mgNew, mbtNew, vwNew, vgNew, vbtNew]
  let retTypes := #[tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape]
  return (s, retNames, retTypes)

/-- Emit SGD+momentum update for a convBn layer (W, gamma, beta). -/
private def emitConvBnMomentum (p ic oc kSize : Nat) (applyWeightDecay : Bool := false)
    (clipScale : Option String := none) : String × Array String × Array String := Id.run do
  let wShape := [oc, ic, kSize, kSize]
  let bShape := [oc]
  let mut s := ""
  let (s1, wNew, mwNew, vwNew) := emitMomentumUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"W{p}" (applyWeightDecay := applyWeightDecay) (clipScale := clipScale)
  s := s ++ s1
  let (s2, gNew, mgNew, vgNew) := emitMomentumUpdate s!"%g{p}" s!"%d_g{p}" s!"%m_g{p}" s!"%v_g{p}" bShape s!"g{p}" (clipScale := clipScale)
  s := s ++ s2
  let (s3, btNew, mbtNew, vbtNew) := emitMomentumUpdate s!"%bt{p}" s!"%d_bt{p}" s!"%m_bt{p}" s!"%v_bt{p}" bShape s!"bt{p}" (clipScale := clipScale)
  s := s ++ s3
  let retNames := #[wNew, gNew, btNew, mwNew, mgNew, mbtNew, vwNew, vgNew, vbtNew]
  let retTypes := #[tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape,
                    tensorTy wShape, tensorTy bShape, tensorTy bShape]
  return (s, retNames, retTypes)

/-- Emit the per-pixel softmax-CE block for segmentation. Logits
    are `(B, NC, H, W)` (curShape at the point of call), labels are
    `(B, H, W)` int32 (passed in as `%y_seg` by the caller).

    Forward:  loss = -mean over (n,h,w) of log softmax(logits)[n, label[n,h,w], h, w]
    Backward: %d_logits_seg = (softmax - onehot(labels)) / (B · H · W)

    Returns the MLIR text. The caller is responsible for setting
    `gradSSA := "%d_logits_seg"` and `gradShape := [B, NC, H, W]`
    after appending this block.

    `lossOut`/`gradOut` name the two results. They default to the plain
    `%loss` / `%d_logits_seg` the seg path consumes, and are overridden only
    by `.diceCE`, which emits this block and the Dice block side by side under
    private names and sums them. -/
private def emitPerPixelCEBlock (B NC H W : Nat) (logitsSSA labelSSA : String)
    (labelSmoothing : Float)
    (lossOut : String := "%loss") (gradOut : String := "%d_logits_seg")
    (weights : List Float := []) : String := Id.run do
  let bnhwTy := s!"tensor<{B}x{H}x{W}xi32>"
  let bnhwI1 := s!"tensor<{B}x{NC}x{H}x{W}xi1>"
  let bnhwfTy := tensorTy [B, NC, H, W]
  let bhwfTy := tensorTy [B, H, W]
  let ncTy := tensorTy [NC]
  -- `weights = []` is the unweighted path and emits byte-identical MLIR to
  -- before this parameter existed: the normalizer is the constant N and the
  -- per-pixel weight factor never appears.
  let wOn := !weights.isEmpty
  let denom : Float := (B * H * W).toFloat
  let smoothOn  : Float := if labelSmoothing > 0.0 then 1.0 - labelSmoothing + labelSmoothing / NC.toFloat else 1.0
  let smoothOff : Float := if labelSmoothing > 0.0 then labelSmoothing / NC.toFloat else 0.0
  let mut s := ""
  s := s ++ "\n    // ════════════ PER-PIXEL SOFTMAX-CE ════════════\n"
  -- 1. log_softmax along channel axis (axis 1) of (B, NC, H, W) logits.
  s := s ++ s!"    %seg_max = stablehlo.reduce({logitsSSA} init: %neginf) applies stablehlo.maximum across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %seg_max_b = stablehlo.broadcast_in_dim %seg_max, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %seg_shifted = stablehlo.subtract {logitsSSA}, %seg_max_b : {bnhwfTy}\n"
  s := s ++ s!"    %seg_exp = stablehlo.exponential %seg_shifted : {bnhwfTy}\n"
  s := s ++ s!"    %seg_sum = stablehlo.reduce(%seg_exp init: %zf) applies stablehlo.add across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %seg_logsum = stablehlo.log %seg_sum : {bhwfTy}\n"
  s := s ++ s!"    %seg_logsum_b = stablehlo.broadcast_in_dim %seg_logsum, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %seg_logp = stablehlo.subtract %seg_shifted, %seg_logsum_b : {bnhwfTy}\n"
  -- 2. Build (B, NC, H, W) onehot from (B, H, W) int32 labels with optional label smoothing.
  s := s ++ s!"    %seg_iota = stablehlo.iota dim = 1 : tensor<{B}x{NC}x{H}x{W}xi32>\n"
  s := s ++ s!"    %seg_y_b = stablehlo.broadcast_in_dim {labelSSA}, dims = [0, 2, 3] : ({bnhwTy}) -> tensor<{B}x{NC}x{H}x{W}xi32>\n"
  s := s ++ s!"    %seg_mask = stablehlo.compare EQ, %seg_iota, %seg_y_b : (tensor<{B}x{NC}x{H}x{W}xi32>, tensor<{B}x{NC}x{H}x{W}xi32>) -> {bnhwI1}\n"
  s := s ++ s!"    %seg_onef = stablehlo.constant dense<{smoothOn}> : {bnhwfTy}\n"
  s := s ++ s!"    %seg_zerof = stablehlo.constant dense<{smoothOff}> : {bnhwfTy}\n"
  s := s ++ s!"    %seg_onehot = stablehlo.select %seg_mask, %seg_onef, %seg_zerof : {bnhwI1}, {bnhwfTy}\n"
  -- 2b. (weighted only) per-pixel weight map w_k = weights[y_k], built by
  -- selecting the weight vector through the same one-hot mask and collapsing
  -- the channel axis. Reusing `%seg_mask` — the raw EQ comparison, which label
  -- smoothing does not touch — is what keeps this a weight on the *true* class
  -- rather than on the smoothed target.
  if wOn then
    let wlit := String.intercalate ", " (weights.map toString)
    s := s ++ s!"    %seg_wvec = stablehlo.constant dense<[{wlit}]> : {ncTy}\n"
    s := s ++ s!"    %seg_wvec_b = stablehlo.broadcast_in_dim %seg_wvec, dims = [1] : ({ncTy}) -> {bnhwfTy}\n"
    s := s ++ s!"    %seg_wz = stablehlo.constant dense<0.0> : {bnhwfTy}\n"
    s := s ++ s!"    %seg_wsel = stablehlo.select %seg_mask, %seg_wvec_b, %seg_wz : {bnhwI1}, {bnhwfTy}\n"
    s := s ++ s!"    %seg_wmap = stablehlo.reduce(%seg_wsel init: %zf) applies stablehlo.add across dimensions = [1]\n"
    s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
    s := s ++ s!"    %seg_wmap_b = stablehlo.broadcast_in_dim %seg_wmap, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  -- 3. Forward loss: -mean over (B, H, W) of softmax-CE per pixel, or the
  -- weighted mean Σ w_k·CE_k / Σ w_k when weights are given.
  --
  -- The weighted path collapses the channel axis FIRST (giving per-pixel CE at
  -- [B,H,W]) and only then applies the weight, rather than broadcasting the
  -- weight up to [B,NC,H,W] and reducing all four dims at once. The two are
  -- equal by distributivity — w_k doesn't depend on c — but the second form
  -- **miscompiles**: IREE fuses `reduce(multiply(x, broadcast(w)))` into a
  -- `vector.contract` and derives an invalid accumulator shape, failing in the
  -- llvm-cpu backend with "invalid accumulator/result vector shape". Same
  -- family as the `select_and_scatter` gap routed around at the maxPool
  -- backward. Reducing to [B,H,W] before the multiply keeps the weight out of
  -- any reduce's operand tree and compiles clean on both CPU and rocm — and is
  -- the more honest spelling of the math anyway, since the weight is a
  -- per-pixel quantity, not a per-channel one.
  if wOn then
    s := s ++ s!"    %seg_ce0 = stablehlo.multiply %seg_logp, %seg_onehot : {bnhwfTy}\n"
    s := s ++ s!"    %seg_cek = stablehlo.reduce(%seg_ce0 init: %zf) applies stablehlo.add across dimensions = [1]\n"
    s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
    s := s ++ s!"    %seg_wce = stablehlo.multiply %seg_cek, %seg_wmap : {bhwfTy}\n"
    s := s ++ s!"    %seg_total = stablehlo.reduce(%seg_wce init: %zf) applies stablehlo.add across dimensions = [0, 1, 2]\n"
    s := s ++ s!"           : ({bhwfTy}, tensor<f32>) -> tensor<f32>\n"
    -- Σ_k w_{y_k}. Labels-only, so it is a constant w.r.t. the logits and
    -- contributes no gradient term — which is the whole reason this
    -- normalization is affordable.
    s := s ++ s!"    %seg_denom = stablehlo.reduce(%seg_wmap init: %zf) applies stablehlo.add across dimensions = [0, 1, 2]\n"
    s := s ++ s!"           : ({bhwfTy}, tensor<f32>) -> tensor<f32>\n"
  else
    s := s ++ s!"    %seg_weighted = stablehlo.multiply %seg_logp, %seg_onehot : {bnhwfTy}\n"
    s := s ++ s!"    %seg_total = stablehlo.reduce(%seg_weighted init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
    s := s ++ s!"           : ({bnhwfTy}, tensor<f32>) -> tensor<f32>\n"
    s := s ++ s!"    %seg_denom = stablehlo.constant dense<{denom}> : tensor<f32>\n"
  s := s ++ s!"    %seg_mean = stablehlo.divide %seg_total, %seg_denom : tensor<f32>\n"
  s := s ++ s!"    {lossOut} = stablehlo.negate %seg_mean : tensor<f32>\n"
  -- 4. Backward seed: (softmax - onehot) / N, or (w_k/Σw)·(softmax - onehot).
  s := s ++ s!"    %seg_sum_b = stablehlo.broadcast_in_dim %seg_sum, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %seg_softmax = stablehlo.divide %seg_exp, %seg_sum_b : {bnhwfTy}\n"
  s := s ++ s!"    %seg_smmoh = stablehlo.subtract %seg_softmax, %seg_onehot : {bnhwfTy}\n"
  s := s ++ s!"    %seg_denom_b = stablehlo.broadcast_in_dim %seg_denom, dims = [] : (tensor<f32>) -> {bnhwfTy}\n"
  if wOn then
    s := s ++ s!"    %seg_scale = stablehlo.divide %seg_wmap_b, %seg_denom_b : {bnhwfTy}\n"
    s := s ++ s!"    {gradOut} = stablehlo.multiply %seg_smmoh, %seg_scale : {bnhwfTy}\n"
  else
    s := s ++ s!"    {gradOut} = stablehlo.divide %seg_smmoh, %seg_denom_b : {bnhwfTy}\n"
  return s

/-- Soft-Dice loss + `d_logits` for the segmentation path.

    Forward, per class `c`, reducing over the whole batch (dims 0,2,3):

      I_c = Σ p_c·y_c    P_c = Σ p_c    Y_c = Σ y_c
      D_c = (2·I_c + ε) / (P_c + Y_c + ε)
      loss = 1 - (1/NC)·Σ_c D_c

    ε (`smooth`) both regularizes the ratio and makes an absent-and-correctly-
    -unpredicted class score D_c = ε/ε = 1 (zero loss) rather than 0/0.

    Backward. `D_c` depends only on channel `c`, so dL/dp is elementwise with
    two per-class broadcast scalars:

      ∂D_c/∂p_{c,k} = [2·y_{c,k}·den_c - num_c] / den_c²  =  A_c·y_{c,k} - B_c
        where A_c = 2/den_c,  B_c = num_c/den_c²
      g_{c,k} := ∂L/∂p_{c,k} = (1/NC)·(B_c - A_c·y_{c,k})

    Then chain through the channel softmax. Unlike CE — whose `p - y` seed is
    what's left *after* the softmax Jacobian cancels the log — Dice gets no
    such cancellation, so the Jacobian-vector product is explicit:

      dz_i = p_i·(g_i - Σ_j g_j·p_j)

    which is elementwise ops plus one channel reduce. No new primitive.

    Labels are one-hot'd raw here: label smoothing is a CE notion (it softens
    a log-likelihood target) and has no meaning for a set-overlap ratio, so
    `.diceCE` applies smoothing to its CE half only. -/
private def emitSegDiceBlock (B NC H W : Nat) (logitsSSA labelSSA : String)
    (lossOut : String := "%loss") (gradOut : String := "%d_logits_seg")
    (smooth : Float := 1.0) : String := Id.run do
  let bhwI32  := s!"tensor<{B}x{H}x{W}xi32>"
  let bnhwI32 := s!"tensor<{B}x{NC}x{H}x{W}xi32>"
  let bnhwI1  := s!"tensor<{B}x{NC}x{H}x{W}xi1>"
  let bnhwfTy := tensorTy [B, NC, H, W]
  let bhwfTy  := tensorTy [B, H, W]
  let ncTy    := tensorTy [NC]
  let mut s := ""
  s := s ++ "\n    // ════════════ SOFT DICE (batch, per-class) ════════════\n"
  -- 1. softmax along the channel axis (max-shifted for stability).
  s := s ++ s!"    %dc_max = stablehlo.reduce({logitsSSA} init: %neginf) applies stablehlo.maximum across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %dc_max_b = stablehlo.broadcast_in_dim %dc_max, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %dc_shifted = stablehlo.subtract {logitsSSA}, %dc_max_b : {bnhwfTy}\n"
  s := s ++ s!"    %dc_exp = stablehlo.exponential %dc_shifted : {bnhwfTy}\n"
  s := s ++ s!"    %dc_esum = stablehlo.reduce(%dc_exp init: %zf) applies stablehlo.add across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %dc_esum_b = stablehlo.broadcast_in_dim %dc_esum, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %dc_p = stablehlo.divide %dc_exp, %dc_esum_b : {bnhwfTy}\n"
  -- 2. raw one-hot (no label smoothing — see the docstring).
  s := s ++ s!"    %dc_iota = stablehlo.iota dim = 1 : {bnhwI32}\n"
  s := s ++ s!"    %dc_y_b = stablehlo.broadcast_in_dim {labelSSA}, dims = [0, 2, 3] : ({bhwI32}) -> {bnhwI32}\n"
  s := s ++ s!"    %dc_mask = stablehlo.compare EQ, %dc_iota, %dc_y_b : ({bnhwI32}, {bnhwI32}) -> {bnhwI1}\n"
  s := s ++ s!"    %dc_ones = stablehlo.constant dense<1.0> : {bnhwfTy}\n"
  s := s ++ s!"    %dc_zeros = stablehlo.constant dense<0.0> : {bnhwfTy}\n"
  s := s ++ s!"    %dc_y = stablehlo.select %dc_mask, %dc_ones, %dc_zeros : {bnhwI1}, {bnhwfTy}\n"
  -- 3. per-class batch sums I_c, P_c, Y_c.
  s := s ++ s!"    %dc_py = stablehlo.multiply %dc_p, %dc_y : {bnhwfTy}\n"
  s := s ++ s!"    %dc_I = stablehlo.reduce(%dc_py init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {ncTy}\n"
  s := s ++ s!"    %dc_P = stablehlo.reduce(%dc_p init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {ncTy}\n"
  s := s ++ s!"    %dc_Y = stablehlo.reduce(%dc_y init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {ncTy}\n"
  -- 4. D_c = (2 I_c + ε) / (P_c + Y_c + ε);  loss = 1 - mean_c D_c.
  s := s ++ s!"    %dc_eps = stablehlo.constant dense<{smooth}> : {ncTy}\n"
  s := s ++ s!"    %dc_twov = stablehlo.constant dense<2.0> : {ncTy}\n"
  s := s ++ s!"    %dc_2I = stablehlo.multiply %dc_I, %dc_twov : {ncTy}\n"
  s := s ++ s!"    %dc_num = stablehlo.add %dc_2I, %dc_eps : {ncTy}\n"
  s := s ++ s!"    %dc_PY = stablehlo.add %dc_P, %dc_Y : {ncTy}\n"
  s := s ++ s!"    %dc_den = stablehlo.add %dc_PY, %dc_eps : {ncTy}\n"
  s := s ++ s!"    %dc_D = stablehlo.divide %dc_num, %dc_den : {ncTy}\n"
  s := s ++ s!"    %dc_Dsum = stablehlo.reduce(%dc_D init: %zf) applies stablehlo.add across dimensions = [0]\n"
  s := s ++ s!"          : ({ncTy}, tensor<f32>) -> tensor<f32>\n"
  s := s ++ s!"    %dc_ncf = stablehlo.constant dense<{NC.toFloat}> : tensor<f32>\n"
  s := s ++ s!"    %dc_Dmean = stablehlo.divide %dc_Dsum, %dc_ncf : tensor<f32>\n"
  s := s ++ s!"    %dc_one_s = stablehlo.constant dense<1.0> : tensor<f32>\n"
  s := s ++ s!"    {lossOut} = stablehlo.subtract %dc_one_s, %dc_Dmean : tensor<f32>\n"
  -- 5. g = ∂L/∂p = (1/NC)·(B_c - A_c·y),  A_c = 2/den,  B_c = num/den².
  s := s ++ s!"    %dc_den2 = stablehlo.multiply %dc_den, %dc_den : {ncTy}\n"
  s := s ++ s!"    %dc_A = stablehlo.divide %dc_twov, %dc_den : {ncTy}\n"
  s := s ++ s!"    %dc_Bc = stablehlo.divide %dc_num, %dc_den2 : {ncTy}\n"
  s := s ++ s!"    %dc_A_b = stablehlo.broadcast_in_dim %dc_A, dims = [1] : ({ncTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %dc_B_b = stablehlo.broadcast_in_dim %dc_Bc, dims = [1] : ({ncTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %dc_Ay = stablehlo.multiply %dc_A_b, %dc_y : {bnhwfTy}\n"
  s := s ++ s!"    %dc_BmAy = stablehlo.subtract %dc_B_b, %dc_Ay : {bnhwfTy}\n"
  s := s ++ s!"    %dc_ncf_b = stablehlo.constant dense<{NC.toFloat}> : {bnhwfTy}\n"
  s := s ++ s!"    %dc_g = stablehlo.divide %dc_BmAy, %dc_ncf_b : {bnhwfTy}\n"
  -- 6. softmax Jacobian-vector product: dz_i = p_i·(g_i - Σ_j g_j p_j).
  s := s ++ s!"    %dc_gp = stablehlo.multiply %dc_g, %dc_p : {bnhwfTy}\n"
  s := s ++ s!"    %dc_dot = stablehlo.reduce(%dc_gp init: %zf) applies stablehlo.add across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %dc_dot_b = stablehlo.broadcast_in_dim %dc_dot, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %dc_gmd = stablehlo.subtract %dc_g, %dc_dot_b : {bnhwfTy}\n"
  s := s ++ s!"    {gradOut} = stablehlo.multiply %dc_p, %dc_gmd : {bnhwfTy}\n"
  return s

/-- Per-pixel focal CE (Lin et al.) + `d_logits` for the segmentation path.

    Forward, per pixel `k`, with `p_t` the softmax probability of the true class:

      FL_k = -(1 - p_t)^γ · log p_t          loss = (1/N)·Σ_k FL_k

    Backward. Write `omp := 1 - p_t`. Differentiating through `p_t` and then the
    channel softmax (`∂p_t/∂z_c = p_t·(δ_tc - p_c)`) collapses to one per-pixel
    scalar times the familiar CE shape:

      A_k := γ·p_t·omp^(γ-1)·log p_t - omp^γ
      ∂FL_k/∂z_c = A_k · (y_c - p_c)

    Two readings of `A` that are the whole reason this loss is in the demo:

      * `γ = 0` ⇒ `A = -1` ⇒ `dz = (p - y)/N`, i.e. **exactly** `perPixelCE`.
        That is a free structural test, and `seg_loss_probe_check.py` runs it.
      * `p_t → 0` (the collapsed class) ⇒ `p_t·log p_t → 0` and `omp^γ → 1`, so
        `A → -1` and the gradient tends to **CE's**, not to zero. Focal does not
        rescue a collapsed class by amplifying it — contrast Dice, whose `p_i`
        factor sends the gradient to zero exactly there. It works, when it works,
        by driving `A → 0` at `p_t → 1` and defunding the easy majority.

    `omp^γ` is emitted as `exp(γ·log(max(omp, ε)))` rather than
    `stablehlo.power`, matching the YOLOv1 objectness path's idiom. The ε clamp
    only engages at `p_t > 1 - ε` (a saturated softmax), where `omp^γ` is ~0 and
    the pixel contributes nothing anyway.

    Unlike that YOLO path, the weight here is **not detached**: `A` carries the
    derivative of the `(1-p_t)^γ` factor itself. Detaching is a defensible
    approximation but yields a `d_logits` that is not the gradient of the loss
    being reported, which FD would (correctly) reject. -/
private def emitSegFocalBlock (B NC H W : Nat) (logitsSSA labelSSA : String)
    (gamma : Float)
    (lossOut : String := "%loss") (gradOut : String := "%d_logits_seg")
    : String := Id.run do
  let bhwI32  := s!"tensor<{B}x{H}x{W}xi32>"
  let bnhwI32 := s!"tensor<{B}x{NC}x{H}x{W}xi32>"
  let bnhwI1  := s!"tensor<{B}x{NC}x{H}x{W}xi1>"
  let bnhwfTy := tensorTy [B, NC, H, W]
  let bhwfTy  := tensorTy [B, H, W]
  let denom : Float := (B * H * W).toFloat
  let mut s := ""
  s := s ++ s!"\n    // ════════════ PER-PIXEL FOCAL CE (γ={gamma}) ════════════\n"
  -- 1. log_softmax + softmax along the channel axis (max-shifted).
  s := s ++ s!"    %fl_max = stablehlo.reduce({logitsSSA} init: %neginf) applies stablehlo.maximum across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %fl_max_b = stablehlo.broadcast_in_dim %fl_max, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %fl_shifted = stablehlo.subtract {logitsSSA}, %fl_max_b : {bnhwfTy}\n"
  s := s ++ s!"    %fl_exp = stablehlo.exponential %fl_shifted : {bnhwfTy}\n"
  s := s ++ s!"    %fl_esum = stablehlo.reduce(%fl_exp init: %zf) applies stablehlo.add across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %fl_logsum = stablehlo.log %fl_esum : {bhwfTy}\n"
  s := s ++ s!"    %fl_logsum_b = stablehlo.broadcast_in_dim %fl_logsum, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %fl_logp = stablehlo.subtract %fl_shifted, %fl_logsum_b : {bnhwfTy}\n"
  s := s ++ s!"    %fl_esum_b = stablehlo.broadcast_in_dim %fl_esum, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %fl_p = stablehlo.divide %fl_exp, %fl_esum_b : {bnhwfTy}\n"
  -- 2. Raw one-hot. Label smoothing is rejected host-side for this kind: p_t is
  -- the probability of *the* true class, which a softened target does not name.
  s := s ++ s!"    %fl_iota = stablehlo.iota dim = 1 : {bnhwI32}\n"
  s := s ++ s!"    %fl_y_b = stablehlo.broadcast_in_dim {labelSSA}, dims = [0, 2, 3] : ({bhwI32}) -> {bnhwI32}\n"
  s := s ++ s!"    %fl_mask = stablehlo.compare EQ, %fl_iota, %fl_y_b : ({bnhwI32}, {bnhwI32}) -> {bnhwI1}\n"
  s := s ++ s!"    %fl_ones = stablehlo.constant dense<1.0> : {bnhwfTy}\n"
  s := s ++ s!"    %fl_zeros = stablehlo.constant dense<0.0> : {bnhwfTy}\n"
  s := s ++ s!"    %fl_y = stablehlo.select %fl_mask, %fl_ones, %fl_zeros : {bnhwI1}, {bnhwfTy}\n"
  -- 3. Gather p_t and log p_t to [B,H,W] (the one-hot kills every other channel).
  s := s ++ s!"    %fl_py = stablehlo.multiply %fl_p, %fl_y : {bnhwfTy}\n"
  s := s ++ s!"    %fl_pt = stablehlo.reduce(%fl_py init: %zf) applies stablehlo.add across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  s := s ++ s!"    %fl_lpy = stablehlo.multiply %fl_logp, %fl_y : {bnhwfTy}\n"
  s := s ++ s!"    %fl_lpt = stablehlo.reduce(%fl_lpy init: %zf) applies stablehlo.add across dimensions = [1]\n"
  s := s ++ s!"          : ({bnhwfTy}, tensor<f32>) -> {bhwfTy}\n"
  -- 4. omp = max(1 - p_t, ε);  omp^γ = exp(γ·log omp).
  s := s ++ s!"    %fl_oneh = stablehlo.constant dense<1.0> : {bhwfTy}\n"
  s := s ++ s!"    %fl_epsh = stablehlo.constant dense<1.0e-12> : {bhwfTy}\n"
  s := s ++ s!"    %fl_omp0 = stablehlo.subtract %fl_oneh, %fl_pt : {bhwfTy}\n"
  s := s ++ s!"    %fl_omp = stablehlo.maximum %fl_omp0, %fl_epsh : {bhwfTy}\n"
  s := s ++ s!"    %fl_logomp = stablehlo.log %fl_omp : {bhwfTy}\n"
  s := s ++ s!"    %fl_gam = stablehlo.constant dense<{gamma}> : {bhwfTy}\n"
  s := s ++ s!"    %fl_glog = stablehlo.multiply %fl_gam, %fl_logomp : {bhwfTy}\n"
  s := s ++ s!"    %fl_ompg = stablehlo.exponential %fl_glog : {bhwfTy}\n"
  -- 5. loss = -(1/N)·Σ_k omp^γ · log p_t.
  s := s ++ s!"    %fl_flk = stablehlo.multiply %fl_ompg, %fl_lpt : {bhwfTy}\n"
  s := s ++ s!"    %fl_tot = stablehlo.reduce(%fl_flk init: %zf) applies stablehlo.add across dimensions = [0, 1, 2]\n"
  s := s ++ s!"           : ({bhwfTy}, tensor<f32>) -> tensor<f32>\n"
  s := s ++ s!"    %fl_den = stablehlo.constant dense<{denom}> : tensor<f32>\n"
  s := s ++ s!"    %fl_mean = stablehlo.divide %fl_tot, %fl_den : tensor<f32>\n"
  s := s ++ s!"    {lossOut} = stablehlo.negate %fl_mean : tensor<f32>\n"
  -- 6. A = γ·p_t·omp^(γ-1)·log p_t - omp^γ.
  if gamma == 0.0 then
    -- The γ factor kills the first term analytically, but emitting it would
    -- compute 0·(omp^-1)·log p_t — and at a saturated pixel omp^-1 is inf, so
    -- IEEE hands back 0·inf = NaN. Drop the term instead. This is what makes
    -- `focal0 ≡ ce` an exact, runnable check rather than a NaN.
    s := s ++ s!"    %fl_A = stablehlo.negate %fl_ompg : {bhwfTy}\n"
  else
    s := s ++ s!"    %fl_gm1 = stablehlo.constant dense<{gamma - 1.0}> : {bhwfTy}\n"
    s := s ++ s!"    %fl_gm1log = stablehlo.multiply %fl_gm1, %fl_logomp : {bhwfTy}\n"
    s := s ++ s!"    %fl_ompgm1 = stablehlo.exponential %fl_gm1log : {bhwfTy}\n"
    s := s ++ s!"    %fl_t1 = stablehlo.multiply %fl_gam, %fl_pt : {bhwfTy}\n"
    s := s ++ s!"    %fl_t2 = stablehlo.multiply %fl_t1, %fl_ompgm1 : {bhwfTy}\n"
    s := s ++ s!"    %fl_t3 = stablehlo.multiply %fl_t2, %fl_lpt : {bhwfTy}\n"
    s := s ++ s!"    %fl_A = stablehlo.subtract %fl_t3, %fl_ompg : {bhwfTy}\n"
  -- 7. dz = (A/N)·(y - p).
  s := s ++ s!"    %fl_denh = stablehlo.constant dense<{denom}> : {bhwfTy}\n"
  s := s ++ s!"    %fl_AoN = stablehlo.divide %fl_A, %fl_denh : {bhwfTy}\n"
  s := s ++ s!"    %fl_AoN_b = stablehlo.broadcast_in_dim %fl_AoN, dims = [0, 2, 3] : ({bhwfTy}) -> {bnhwfTy}\n"
  s := s ++ s!"    %fl_ymp = stablehlo.subtract %fl_y, %fl_p : {bnhwfTy}\n"
  s := s ++ s!"    {gradOut} = stablehlo.multiply %fl_AoN_b, %fl_ymp : {bnhwfTy}\n"
  return s

/-- Emit the seg loss block selected by `segLoss`, leaving `%loss` and
    `%d_logits_seg` defined either way. `.diceCE` emits both blocks under
    private names and sums them; the duplicated softmax prologue is identical
    SSA on identical inputs, so CSE folds it. -/
private def emitSegLossBlock (B NC H W : Nat) (logitsSSA labelSSA : String)
    (labelSmoothing : Float) (segLoss : SegLoss) : String := Id.run do
  let bnhwfTy := tensorTy [B, NC, H, W]
  match segLoss with
  | .ce   => emitPerPixelCEBlock B NC H W logitsSSA labelSSA labelSmoothing
  | .weightedCE w =>
    emitPerPixelCEBlock B NC H W logitsSSA labelSSA labelSmoothing (weights := w)
  | .focalCE g => emitSegFocalBlock B NC H W logitsSSA labelSSA g
  | .dice => emitSegDiceBlock B NC H W logitsSSA labelSSA
  | .diceCE =>
    let mut s := ""
    s := s ++ emitPerPixelCEBlock B NC H W logitsSSA labelSSA labelSmoothing "%loss_ce" "%dlog_ce"
    s := s ++ emitSegDiceBlock B NC H W logitsSSA labelSSA "%loss_dc" "%dlog_dc"
    s := s ++ "\n    // ──────── Dice + CE ────────\n"
    s := s ++ s!"    %loss = stablehlo.add %loss_ce, %loss_dc : tensor<f32>\n"
    s := s ++ s!"    %d_logits_seg = stablehlo.add %dlog_ce, %dlog_dc : {bnhwfTy}\n"
    s

/-- DIoU box-loss FORWARD block (brick #1, planning/yolo_drone.md WS-D — the
    IoU-family replacement for the fragile √-MSE box regression). `pred`/`tgt`
    are `[B,4,gH,gW]`: pred channels are the raw head outputs (tx,ty,tw,th), tgt
    channels are the YOLOv1 target layout (cell-offset x,y then w_rel,h_rel).
    `mask` is `[B,gH,gW]` (1 on object cells). Positive box parameterization by
    construction — `cx=(j+σ(tx))/gW`, `cy=(i+σ(ty))/gH`, `w=exp(tw)`, `h=exp(th)`
    — so boxes are always valid (fixes the negative-w/h gradient death). Emits
    `loss = Σ_cells mask·(1 - DIoU)` with `DIoU = IoU − ρ²(centers)/c²(enclosing)`.
    Requires `%zf` in scope. Numeric spec: scripts/diou_grad_check.py; the emitted
    forward is checked against numpy in scripts/diou_probe_check.py. -/
private def emitDiouForward (B gH gW : Nat) (predSSA tgtSSA maskSSA : String)
    (lossOut : String := "%dio_loss") (anchorW : Float := 1.0) (anchorH : Float := 1.0)
    (pfx : String := "dio") : String := Id.run do
  let c1 := tensorTy [B, 1, gH, gW]
  let p4 := tensorTy [B, 4, gH, gW]
  let maskTy := tensorTy [B, gH, gW]
  let invW := 1.0 / gW.toFloat
  let invH := 1.0 / gH.toFloat
  let slice := fun (dst src : String) (c : Nat) =>
    s!"    {dst} = \"stablehlo.slice\"({src}) " ++ "{" ++
      s!" start_indices = array<i64: 0, {c}, 0, 0>, limit_indices = array<i64: {B}, {c+1}, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++
      "} : " ++ s!"({p4}) -> {c1}\n"
  let mut s := "\n    // ════════════ DIoU box loss (forward) ════════════\n"
  -- constants
  s := s ++ s!"    %{pfx}_half = stablehlo.constant dense<0.5> : {c1}\n"
  s := s ++ s!"    %{pfx}_one = stablehlo.constant dense<1.0> : {c1}\n"
  s := s ++ s!"    %{pfx}_zero = stablehlo.constant dense<0.0> : {c1}\n"
  s := s ++ s!"    %{pfx}_eps = stablehlo.constant dense<1.0e-9> : {c1}\n"
  s := s ++ s!"    %{pfx}_invW = stablehlo.constant dense<{invW}> : {c1}\n"
  s := s ++ s!"    %{pfx}_invH = stablehlo.constant dense<{invH}> : {c1}\n"
  -- slice raw pred + target channels
  s := s ++ slice s!"%{pfx}_tx" predSSA 0
  s := s ++ slice s!"%{pfx}_ty" predSSA 1
  s := s ++ slice s!"%{pfx}_tw" predSSA 2
  s := s ++ slice s!"%{pfx}_th" predSSA 3
  s := s ++ slice s!"%{pfx}_gx" tgtSSA 0
  s := s ++ slice s!"%{pfx}_gy" tgtSSA 1
  s := s ++ slice s!"%{pfx}_gw" tgtSSA 2
  s := s ++ slice s!"%{pfx}_gh" tgtSSA 3
  -- cell indices
  s := s ++ s!"    %{pfx}_cj = stablehlo.iota dim = 3 : {c1}\n"
  s := s ++ s!"    %{pfx}_ci = stablehlo.iota dim = 2 : {c1}\n"
  -- pred box (center-size, normalized): cx=(j+σ(tx))/gW, w=exp(tw)
  s := s ++ s!"    %{pfx}_sx = stablehlo.logistic %{pfx}_tx : {c1}\n"
  s := s ++ s!"    %{pfx}_sy = stablehlo.logistic %{pfx}_ty : {c1}\n"
  s := s ++ s!"    %{pfx}_cjsx = stablehlo.add %{pfx}_cj, %{pfx}_sx : {c1}\n"
  s := s ++ s!"    %{pfx}_cx = stablehlo.multiply %{pfx}_cjsx, %{pfx}_invW : {c1}\n"
  s := s ++ s!"    %{pfx}_cisy = stablehlo.add %{pfx}_ci, %{pfx}_sy : {c1}\n"
  s := s ++ s!"    %{pfx}_cy = stablehlo.multiply %{pfx}_cisy, %{pfx}_invH : {c1}\n"
  -- w = anchorW · exp(tw): anchor-relative width prior (brick #2). anchorW=1
  -- reproduces the anchor-free path exactly. Backward is unchanged — it uses
  -- %{pfx}_w, and dw/dtw = anchorW·exp(tw) = w regardless of the prior.
  s := s ++ s!"    %{pfx}_expw = stablehlo.exponential %{pfx}_tw : {c1}\n"
  s := s ++ s!"    %{pfx}_aw = stablehlo.constant dense<{anchorW}> : {c1}\n"
  s := s ++ s!"    %{pfx}_w = stablehlo.multiply %{pfx}_expw, %{pfx}_aw : {c1}\n"
  s := s ++ s!"    %{pfx}_exph = stablehlo.exponential %{pfx}_th : {c1}\n"
  s := s ++ s!"    %{pfx}_ah = stablehlo.constant dense<{anchorH}> : {c1}\n"
  s := s ++ s!"    %{pfx}_h = stablehlo.multiply %{pfx}_exph, %{pfx}_ah : {c1}\n"
  s := s ++ s!"    %{pfx}_hw = stablehlo.multiply %{pfx}_w, %{pfx}_half : {c1}\n"
  s := s ++ s!"    %{pfx}_hh = stablehlo.multiply %{pfx}_h, %{pfx}_half : {c1}\n"
  s := s ++ s!"    %{pfx}_x0 = stablehlo.subtract %{pfx}_cx, %{pfx}_hw : {c1}\n"
  s := s ++ s!"    %{pfx}_x1 = stablehlo.add %{pfx}_cx, %{pfx}_hw : {c1}\n"
  s := s ++ s!"    %{pfx}_y0 = stablehlo.subtract %{pfx}_cy, %{pfx}_hh : {c1}\n"
  s := s ++ s!"    %{pfx}_y1 = stablehlo.add %{pfx}_cy, %{pfx}_hh : {c1}\n"
  -- target box
  s := s ++ s!"    %{pfx}_cjgx = stablehlo.add %{pfx}_cj, %{pfx}_gx : {c1}\n"
  s := s ++ s!"    %{pfx}_tcx = stablehlo.multiply %{pfx}_cjgx, %{pfx}_invW : {c1}\n"
  s := s ++ s!"    %{pfx}_cigy = stablehlo.add %{pfx}_ci, %{pfx}_gy : {c1}\n"
  s := s ++ s!"    %{pfx}_tcy = stablehlo.multiply %{pfx}_cigy, %{pfx}_invH : {c1}\n"
  s := s ++ s!"    %{pfx}_ghw = stablehlo.multiply %{pfx}_gw, %{pfx}_half : {c1}\n"
  s := s ++ s!"    %{pfx}_ghh = stablehlo.multiply %{pfx}_gh, %{pfx}_half : {c1}\n"
  s := s ++ s!"    %{pfx}_X0 = stablehlo.subtract %{pfx}_tcx, %{pfx}_ghw : {c1}\n"
  s := s ++ s!"    %{pfx}_X1 = stablehlo.add %{pfx}_tcx, %{pfx}_ghw : {c1}\n"
  s := s ++ s!"    %{pfx}_Y0 = stablehlo.subtract %{pfx}_tcy, %{pfx}_ghh : {c1}\n"
  s := s ++ s!"    %{pfx}_Y1 = stablehlo.add %{pfx}_tcy, %{pfx}_ghh : {c1}\n"
  -- intersection
  s := s ++ s!"    %{pfx}_ix1 = stablehlo.minimum %{pfx}_x1, %{pfx}_X1 : {c1}\n"
  s := s ++ s!"    %{pfx}_ix0 = stablehlo.maximum %{pfx}_x0, %{pfx}_X0 : {c1}\n"
  s := s ++ s!"    %{pfx}_iwd = stablehlo.subtract %{pfx}_ix1, %{pfx}_ix0 : {c1}\n"
  s := s ++ s!"    %{pfx}_iw = stablehlo.maximum %{pfx}_iwd, %{pfx}_zero : {c1}\n"
  s := s ++ s!"    %{pfx}_iy1 = stablehlo.minimum %{pfx}_y1, %{pfx}_Y1 : {c1}\n"
  s := s ++ s!"    %{pfx}_iy0 = stablehlo.maximum %{pfx}_y0, %{pfx}_Y0 : {c1}\n"
  s := s ++ s!"    %{pfx}_ihd = stablehlo.subtract %{pfx}_iy1, %{pfx}_iy0 : {c1}\n"
  s := s ++ s!"    %{pfx}_ih = stablehlo.maximum %{pfx}_ihd, %{pfx}_zero : {c1}\n"
  s := s ++ s!"    %{pfx}_inter = stablehlo.multiply %{pfx}_iw, %{pfx}_ih : {c1}\n"
  -- union + IoU
  s := s ++ s!"    %{pfx}_ap = stablehlo.multiply %{pfx}_w, %{pfx}_h : {c1}\n"
  s := s ++ s!"    %{pfx}_ag = stablehlo.multiply %{pfx}_gw, %{pfx}_gh : {c1}\n"
  s := s ++ s!"    %{pfx}_apag = stablehlo.add %{pfx}_ap, %{pfx}_ag : {c1}\n"
  s := s ++ s!"    %{pfx}_union = stablehlo.subtract %{pfx}_apag, %{pfx}_inter : {c1}\n"
  s := s ++ s!"    %{pfx}_unionc = stablehlo.maximum %{pfx}_union, %{pfx}_eps : {c1}\n"
  s := s ++ s!"    %{pfx}_iou = stablehlo.divide %{pfx}_inter, %{pfx}_unionc : {c1}\n"
  -- center distance ρ²
  s := s ++ s!"    %{pfx}_dcx = stablehlo.subtract %{pfx}_cx, %{pfx}_tcx : {c1}\n"
  s := s ++ s!"    %{pfx}_dcy = stablehlo.subtract %{pfx}_cy, %{pfx}_tcy : {c1}\n"
  s := s ++ s!"    %{pfx}_dcx2 = stablehlo.multiply %{pfx}_dcx, %{pfx}_dcx : {c1}\n"
  s := s ++ s!"    %{pfx}_dcy2 = stablehlo.multiply %{pfx}_dcy, %{pfx}_dcy : {c1}\n"
  s := s ++ s!"    %{pfx}_rho2 = stablehlo.add %{pfx}_dcx2, %{pfx}_dcy2 : {c1}\n"
  -- enclosing box diagonal² c²
  s := s ++ s!"    %{pfx}_ex1 = stablehlo.maximum %{pfx}_x1, %{pfx}_X1 : {c1}\n"
  s := s ++ s!"    %{pfx}_ex0 = stablehlo.minimum %{pfx}_x0, %{pfx}_X0 : {c1}\n"
  s := s ++ s!"    %{pfx}_cw = stablehlo.subtract %{pfx}_ex1, %{pfx}_ex0 : {c1}\n"
  s := s ++ s!"    %{pfx}_ey1 = stablehlo.maximum %{pfx}_y1, %{pfx}_Y1 : {c1}\n"
  s := s ++ s!"    %{pfx}_ey0 = stablehlo.minimum %{pfx}_y0, %{pfx}_Y0 : {c1}\n"
  s := s ++ s!"    %{pfx}_ch = stablehlo.subtract %{pfx}_ey1, %{pfx}_ey0 : {c1}\n"
  s := s ++ s!"    %{pfx}_cw2 = stablehlo.multiply %{pfx}_cw, %{pfx}_cw : {c1}\n"
  s := s ++ s!"    %{pfx}_ch2 = stablehlo.multiply %{pfx}_ch, %{pfx}_ch : {c1}\n"
  s := s ++ s!"    %{pfx}_c2 = stablehlo.add %{pfx}_cw2, %{pfx}_ch2 : {c1}\n"
  s := s ++ s!"    %{pfx}_c2c = stablehlo.maximum %{pfx}_c2, %{pfx}_eps : {c1}\n"
  s := s ++ s!"    %{pfx}_pen = stablehlo.divide %{pfx}_rho2, %{pfx}_c2c : {c1}\n"
  -- DIoU + masked loss
  s := s ++ s!"    %{pfx}_diou = stablehlo.subtract %{pfx}_iou, %{pfx}_pen : {c1}\n"
  s := s ++ s!"    %{pfx}_1md = stablehlo.subtract %{pfx}_one, %{pfx}_diou : {c1}\n"
  s := s ++ s!"    %{pfx}_mask = stablehlo.broadcast_in_dim {maskSSA}, dims = [0, 2, 3] : ({maskTy}) -> {c1}\n"
  s := s ++ s!"    %{pfx}_cell = stablehlo.multiply %{pfx}_1md, %{pfx}_mask : {c1}\n"
  s := s ++ s!"    {lossOut} = stablehlo.reduce(%{pfx}_cell init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
  s := s ++ s!"           : ({c1}, tensor<f32>) -> tensor<f32>\n"
  return s

/-- DIoU box-loss BACKWARD block. Assumes `emitDiouForward` ran just before (reuses
    its `%{pfx}_*` SSA values). Emits `{gradOut}` = `∂loss/∂pred` `[B,4,gH,gW]`
    (channels d/dtx, d/dty, d/dtw, d/dth), the hand-derived VJP of
    `Σ mask·(1 - DIoU)`. Analytic spec: `diou_loss_grad` in scripts/
    diou_grad_check.py; FD-verified against the emitted forward in
    scripts/diou_probe_check.py. Every piecewise min/max sub-gradient is an
    indicator (compare→select). -/
private def emitDiouBackward (B gH gW : Nat) (gradOut : String := "%dio_dpred") (pfx : String := "dio") : String := Id.run do
  let c1 := tensorTy [B, 1, gH, gW]
  let i1 := s!"tensor<{B}x1x{gH}x{gW}xi1>"
  let p4 := tensorTy [B, 4, gH, gW]
  let mut s := "\n    // ════════════ DIoU box loss (backward) ════════════\n"
  s := s ++ s!"    %{pfx}b_two = stablehlo.constant dense<2.0> : {c1}\n"
  s := s ++ s!"    %{pfx}b_nhalf = stablehlo.constant dense<-0.5> : {c1}\n"
  -- indicator helper: ind = 1.0 where (a REL b) else 0.0
  let ind := fun (dst rel a b : String) =>
    s!"    {dst}_i1 = stablehlo.compare {rel}, {a}, {b} : ({c1}, {c1}) -> {i1}\n" ++
    s!"    {dst} = stablehlo.select {dst}_i1, %{pfx}_one, %{pfx}_zero : {i1}, {c1}\n"
  -- intersection existence + which corner is active
  s := s ++ ind s!"%{pfx}b_iwp" "GT" s!"%{pfx}_iw" s!"%{pfx}_zero"
  s := s ++ ind s!"%{pfx}b_ihp" "GT" s!"%{pfx}_ih" s!"%{pfx}_zero"
  s := s ++ ind s!"%{pfx}b_a0" "GT" s!"%{pfx}_x0" s!"%{pfx}_X0"      -- max(x0,X0) picks x0
  s := s ++ ind s!"%{pfx}b_a1" "LT" s!"%{pfx}_x1" s!"%{pfx}_X1"      -- min(x1,X1) picks x1
  s := s ++ ind s!"%{pfx}b_b0" "GT" s!"%{pfx}_y0" s!"%{pfx}_Y0"
  s := s ++ ind s!"%{pfx}b_b1" "LT" s!"%{pfx}_y1" s!"%{pfx}_Y1"
  -- d inter / d corner   (diw_dx0 = -a0·iwp ; diw_dx1 = a1·iwp ; dI = ih·diw / iw·dih)
  s := s ++ s!"    %{pfx}b_a0p = stablehlo.multiply %{pfx}b_a0, %{pfx}b_iwp : {c1}\n"
  s := s ++ s!"    %{pfx}b_a1p = stablehlo.multiply %{pfx}b_a1, %{pfx}b_iwp : {c1}\n"
  s := s ++ s!"    %{pfx}b_b0p = stablehlo.multiply %{pfx}b_b0, %{pfx}b_ihp : {c1}\n"
  s := s ++ s!"    %{pfx}b_b1p = stablehlo.multiply %{pfx}b_b1, %{pfx}b_ihp : {c1}\n"
  -- dI_dx0 = ih·(-a0p) ; dI_dx1 = ih·a1p ; dI_dy0 = iw·(-b0p) ; dI_dy1 = iw·b1p
  s := s ++ s!"    %{pfx}b_dIdx1 = stablehlo.multiply %{pfx}_ih, %{pfx}b_a1p : {c1}\n"
  s := s ++ s!"    %{pfx}b_ih_a0 = stablehlo.multiply %{pfx}_ih, %{pfx}b_a0p : {c1}\n"
  s := s ++ s!"    %{pfx}b_dIdx0 = stablehlo.negate %{pfx}b_ih_a0 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dIdy1 = stablehlo.multiply %{pfx}_iw, %{pfx}b_b1p : {c1}\n"
  s := s ++ s!"    %{pfx}b_iw_b0 = stablehlo.multiply %{pfx}_iw, %{pfx}b_b0p : {c1}\n"
  s := s ++ s!"    %{pfx}b_dIdy0 = stablehlo.negate %{pfx}b_iw_b0 : {c1}\n"
  -- corner→center: dI_dcx = dIdx0+dIdx1 ; dI_dw = -0.5·dIdx0 + 0.5·dIdx1
  s := s ++ s!"    %{pfx}b_dIdcx = stablehlo.add %{pfx}b_dIdx0, %{pfx}b_dIdx1 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dIdcy = stablehlo.add %{pfx}b_dIdy0, %{pfx}b_dIdy1 : {c1}\n"
  s := s ++ s!"    %{pfx}b_x1mx0 = stablehlo.subtract %{pfx}b_dIdx1, %{pfx}b_dIdx0 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dIdw = stablehlo.multiply %{pfx}b_x1mx0, %{pfx}_half : {c1}\n"
  s := s ++ s!"    %{pfx}b_y1my0 = stablehlo.subtract %{pfx}b_dIdy1, %{pfx}b_dIdy0 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dIdh = stablehlo.multiply %{pfx}b_y1my0, %{pfx}_half : {c1}\n"
  -- dIoU = (dI·U - inter·(dAp - dI)) / U²   with U = %{pfx}_unionc
  s := s ++ s!"    %{pfx}b_U2 = stablehlo.multiply %{pfx}_unionc, %{pfx}_unionc : {c1}\n"
  -- helper: diou(dI, dAp) written inline per channel
  --   cx,cy: dAp = 0  ;  w: dAp = h  ;  h: dAp = w
  let diou := fun (dst dI dAp : String) =>
    s!"    {dst}_dUp = stablehlo.subtract {dAp}, {dI} : {c1}\n" ++
    s!"    {dst}_iU = stablehlo.multiply {dI}, %{pfx}_unionc : {c1}\n" ++
    s!"    {dst}_ip = stablehlo.multiply %{pfx}_inter, {dst}_dUp : {c1}\n" ++
    s!"    {dst}_num = stablehlo.subtract {dst}_iU, {dst}_ip : {c1}\n" ++
    s!"    {dst} = stablehlo.divide {dst}_num, %{pfx}b_U2 : {c1}\n"
  s := s ++ diou s!"%{pfx}b_diou_cx" s!"%{pfx}b_dIdcx" s!"%{pfx}_zero"
  s := s ++ diou s!"%{pfx}b_diou_cy" s!"%{pfx}b_dIdcy" s!"%{pfx}_zero"
  s := s ++ diou s!"%{pfx}b_diou_w" s!"%{pfx}b_dIdw" s!"%{pfx}_h"
  s := s ++ diou s!"%{pfx}b_diou_h" s!"%{pfx}b_dIdh" s!"%{pfx}_w"
  -- center penalty ρ²/c². drho2_dcx = 2·dcx (dio_dcx = cx-tcx). c2 = %{pfx}_c2c.
  s := s ++ s!"    %{pfx}b_drho_cx = stablehlo.multiply %{pfx}b_two, %{pfx}_dcx : {c1}\n"
  s := s ++ s!"    %{pfx}b_drho_cy = stablehlo.multiply %{pfx}b_two, %{pfx}_dcy : {c1}\n"
  -- enclosing indicators: e1=(x1>X1) picks x1 for max; e0=(x0<X0) picks x0 for min
  s := s ++ ind s!"%{pfx}b_e1" "GT" s!"%{pfx}_x1" s!"%{pfx}_X1"
  s := s ++ ind s!"%{pfx}b_e0" "LT" s!"%{pfx}_x0" s!"%{pfx}_X0"
  s := s ++ ind s!"%{pfx}b_f1" "GT" s!"%{pfx}_y1" s!"%{pfx}_Y1"
  s := s ++ ind s!"%{pfx}b_f0" "LT" s!"%{pfx}_y0" s!"%{pfx}_Y0"
  -- dcw_dx1 = e1 ; dcw_dx0 = -e0 ; dcw_dcx = dcw_dx0+dcw_dx1 ; dcw_dw = 0.5(dcw_dx1 - dcw_dx0)
  s := s ++ s!"    %{pfx}b_ncw0 = stablehlo.negate %{pfx}b_e0 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dcw_cx = stablehlo.add %{pfx}b_ncw0, %{pfx}b_e1 : {c1}\n"
  s := s ++ s!"    %{pfx}b_cw_diff = stablehlo.subtract %{pfx}b_e1, %{pfx}b_ncw0 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dcw_w = stablehlo.multiply %{pfx}b_cw_diff, %{pfx}_half : {c1}\n"
  s := s ++ s!"    %{pfx}b_nch0 = stablehlo.negate %{pfx}b_f0 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dch_cy = stablehlo.add %{pfx}b_nch0, %{pfx}b_f1 : {c1}\n"
  s := s ++ s!"    %{pfx}b_ch_diff = stablehlo.subtract %{pfx}b_f1, %{pfx}b_nch0 : {c1}\n"
  s := s ++ s!"    %{pfx}b_dch_h = stablehlo.multiply %{pfx}b_ch_diff, %{pfx}_half : {c1}\n"
  -- dc2 = 2·cw·dcw (+ 2·ch·dch for the y-affecting params). cw=%{pfx}_cw ch=%{pfx}_ch
  s := s ++ s!"    %{pfx}b_2cw = stablehlo.multiply %{pfx}b_two, %{pfx}_cw : {c1}\n"
  s := s ++ s!"    %{pfx}b_2ch = stablehlo.multiply %{pfx}b_two, %{pfx}_ch : {c1}\n"
  s := s ++ s!"    %{pfx}b_dc2_cx = stablehlo.multiply %{pfx}b_2cw, %{pfx}b_dcw_cx : {c1}\n"
  s := s ++ s!"    %{pfx}b_dc2_cy = stablehlo.multiply %{pfx}b_2ch, %{pfx}b_dch_cy : {c1}\n"
  s := s ++ s!"    %{pfx}b_dc2_w = stablehlo.multiply %{pfx}b_2cw, %{pfx}b_dcw_w : {c1}\n"
  s := s ++ s!"    %{pfx}b_dc2_h = stablehlo.multiply %{pfx}b_2ch, %{pfx}b_dch_h : {c1}\n"
  s := s ++ s!"    %{pfx}b_c2sq = stablehlo.multiply %{pfx}_c2c, %{pfx}_c2c : {c1}\n"
  -- dpen(drho2, dc2) = (drho2·c2 - rho2·dc2)/c2²  ; cx,cy have drho2≠0; w,h have drho2=0
  let dpen := fun (dst drho2 dc2 : String) =>
    s!"    {dst}_a = stablehlo.multiply {drho2}, %{pfx}_c2c : {c1}\n" ++
    s!"    {dst}_b = stablehlo.multiply %{pfx}_rho2, {dc2} : {c1}\n" ++
    s!"    {dst}_num = stablehlo.subtract {dst}_a, {dst}_b : {c1}\n" ++
    s!"    {dst} = stablehlo.divide {dst}_num, %{pfx}b_c2sq : {c1}\n"
  s := s ++ dpen s!"%{pfx}b_pen_cx" s!"%{pfx}b_drho_cx" s!"%{pfx}b_dc2_cx"
  s := s ++ dpen s!"%{pfx}b_pen_cy" s!"%{pfx}b_drho_cy" s!"%{pfx}b_dc2_cy"
  s := s ++ dpen s!"%{pfx}b_pen_w" s!"%{pfx}_zero" s!"%{pfx}b_dc2_w"
  s := s ++ dpen s!"%{pfx}b_pen_h" s!"%{pfx}_zero" s!"%{pfx}b_dc2_h"
  -- dL/dθ = -dIoU + dpen  (θ = cx,cy,w,h)
  s := s ++ s!"    %{pfx}b_dLcx = stablehlo.subtract %{pfx}b_pen_cx, %{pfx}b_diou_cx : {c1}\n"
  s := s ++ s!"    %{pfx}b_dLcy = stablehlo.subtract %{pfx}b_pen_cy, %{pfx}b_diou_cy : {c1}\n"
  s := s ++ s!"    %{pfx}b_dLw = stablehlo.subtract %{pfx}b_pen_w, %{pfx}b_diou_w : {c1}\n"
  s := s ++ s!"    %{pfx}b_dLh = stablehlo.subtract %{pfx}b_pen_h, %{pfx}b_diou_h : {c1}\n"
  -- chain to raw params: dcx/dtx = σ(1-σ)/gW ; dw/dtw = w ; then mask
  s := s ++ s!"    %{pfx}b_1msx = stablehlo.subtract %{pfx}_one, %{pfx}_sx : {c1}\n"
  s := s ++ s!"    %{pfx}b_sxg = stablehlo.multiply %{pfx}_sx, %{pfx}b_1msx : {c1}\n"
  s := s ++ s!"    %{pfx}b_sxgw = stablehlo.multiply %{pfx}b_sxg, %{pfx}_invW : {c1}\n"
  s := s ++ s!"    %{pfx}b_1msy = stablehlo.subtract %{pfx}_one, %{pfx}_sy : {c1}\n"
  s := s ++ s!"    %{pfx}b_syg = stablehlo.multiply %{pfx}_sy, %{pfx}b_1msy : {c1}\n"
  s := s ++ s!"    %{pfx}b_sygh = stablehlo.multiply %{pfx}b_syg, %{pfx}_invH : {c1}\n"
  s := s ++ s!"    %{pfx}b_gtx = stablehlo.multiply %{pfx}b_dLcx, %{pfx}b_sxgw : {c1}\n"
  s := s ++ s!"    %{pfx}b_gty = stablehlo.multiply %{pfx}b_dLcy, %{pfx}b_sygh : {c1}\n"
  s := s ++ s!"    %{pfx}b_gtw = stablehlo.multiply %{pfx}b_dLw, %{pfx}_w : {c1}\n"
  s := s ++ s!"    %{pfx}b_gth = stablehlo.multiply %{pfx}b_dLh, %{pfx}_h : {c1}\n"
  -- mask (loss sums only object cells) and assemble [B,4,gH,gW]
  s := s ++ s!"    %{pfx}b_mtx = stablehlo.multiply %{pfx}b_gtx, %{pfx}_mask : {c1}\n"
  s := s ++ s!"    %{pfx}b_mty = stablehlo.multiply %{pfx}b_gty, %{pfx}_mask : {c1}\n"
  s := s ++ s!"    %{pfx}b_mtw = stablehlo.multiply %{pfx}b_gtw, %{pfx}_mask : {c1}\n"
  s := s ++ s!"    %{pfx}b_mth = stablehlo.multiply %{pfx}b_gth, %{pfx}_mask : {c1}\n"
  s := s ++ s!"    {gradOut} = stablehlo.concatenate %{pfx}b_mtx, %{pfx}b_mty, %{pfx}b_mtw, %{pfx}b_mth, dim = 1 : ({c1}, {c1}, {c1}, {c1}) -> {p4}\n"
  return s

/-- Anchor YOLOv3-style loss (brick #2, WS-C). A anchors/cell, each slot is
    `[tx,ty,tw,th, obj, cls(NC=10)]`; `pred`/`tgt` are `[B, A·15, gH, gW]`, `mask`
    is `[B, A, gH, gW]` (per-anchor objectness assignment). Per anchor a: the
    FD-verified DIoU box loss at `pfx=a{a}` with prior `anchors[a]` (box_a =
    anchor_a·exp(pred)), focal-BCE objectness, and mask-weighted softmax-CE class.
    Loss = (1/B)·Σ_a [λ_box·Σmask_a(1-DIoU_a) + Σ focalBCE_a + Σ mask_a·CE_a]; the
    gradient is assembled per anchor to `[B,15,gH,gW]` and concatenated to
    `[B, A·15, gH, gW]`. Requires `%zf`,`%neginf` in scope. FD-verified in
    scripts/anchor_loss_probe_check.py. -/
private def emitAnchorYoloLoss (B gH gW : Nat) (anchors : List (Float × Float))
    (predSSA tgtSSA : String) (focalGamma : Float) (lambdaBox : Float)
    (lossOut gradOut : String) : String := Id.run do
  let A := anchors.length
  let NC := 10
  let P := 5 + NC                          -- 15
  let c1 := tensorTy [B, 1, gH, gW]
  let box4 := tensorTy [B, 4, gH, gW]
  let clsT := tensorTy [B, NC, gH, gW]
  let clsRed := tensorTy [B, gH, gW]
  let apT := tensorTy [B, A * P, gH, gW]
  let bhw := tensorTy [B, gH, gW]
  let slabTy := tensorTy [B, P, gH, gW]
  let ln := "0.5"                          -- λ_noobj
  let mut s := "\n    // ════════════ Anchor YOLO loss (A anchors) ════════════\n"
  s := s ++ s!"    %ay_Bf = stablehlo.constant dense<{B}.0> : tensor<f32>\n"
  s := s ++ s!"    %ay_lbox = stablehlo.constant dense<{lambdaBox}> : tensor<f32>\n"
  s := s ++ s!"    %ay_lboxg = stablehlo.constant dense<{lambdaBox / B.toFloat}> : {box4}\n"
  let mut lossParts : List String := []
  let mut gradParts : List String := []
  for a in [0:A] do
    let pa := s!"a{a}"
    let (aw, ah) := anchors.getD a (1.0, 1.0)
    let base := a * P
    let sl := fun (dst src : String) (c0 cn : Nat) (outTy : String) =>
      s!"    {dst} = \"stablehlo.slice\"({src}) " ++ "{" ++ s!" start_indices = array<i64: 0, {c0}, 0, 0>, limit_indices = array<i64: {B}, {cn}, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({apT}) -> {outTy}\n"
    s := s ++ sl s!"%{pa}_bp" predSSA base (base + 4) box4
    s := s ++ sl s!"%{pa}_op" predSSA (base + 4) (base + 5) c1
    s := s ++ sl s!"%{pa}_cp" predSSA (base + 5) (base + P) clsT
    s := s ++ sl s!"%{pa}_bt" tgtSSA base (base + 4) box4
    s := s ++ sl s!"%{pa}_ct" tgtSSA (base + 5) (base + P) clsT
    -- per-anchor mask = the target's objectness channel (encode_targets_anchor sets
    -- target[base+4]=1 exactly where anchor a is assigned), so no separate mask
    -- input / no FFI mask-widening is needed.
    s := s ++ sl s!"%{pa}_m4" tgtSSA (base + 4) (base + 5) c1
    s := s ++ s!"    %{pa}_m = stablehlo.reshape %{pa}_m4 : ({c1}) -> {bhw}\n"
    -- ── DIoU box loss (reuse the FD-verified block at pfx=a{a}, prior anchors[a]) ──
    s := s ++ emitDiouForward B gH gW s!"%{pa}_bp" s!"%{pa}_bt" s!"%{pa}_m" s!"%{pa}_dloss" aw ah pa
    s := s ++ emitDiouBackward B gH gW s!"%{pa}_ddp" pa
    s := s ++ s!"    %{pa}_boxloss = stablehlo.multiply %{pa}_dloss, %ay_lbox : tensor<f32>\n"
    s := s ++ s!"    %{pa}_bg = stablehlo.multiply %{pa}_ddp, %ay_lboxg : {box4}\n"
    -- ── Objectness focal-BCE (target t = mask channel a) ──
    s := s ++ s!"    %{pa}_one1 = stablehlo.constant dense<1.0> : {c1}\n"
    s := s ++ s!"    %{pa}_zero1 = stablehlo.constant dense<0.0> : {c1}\n"
    s := s ++ s!"    %{pa}_eps1 = stablehlo.constant dense<1.0e-12> : {c1}\n"
    s := s ++ s!"    %{pa}_gam1 = stablehlo.constant dense<{focalGamma}> : {c1}\n"
    s := s ++ s!"    %{pa}_ln1 = stablehlo.constant dense<{ln}> : {c1}\n"
    s := s ++ s!"    %{pa}_obj_p = stablehlo.logistic %{pa}_op : {c1}\n"
    s := s ++ s!"    %{pa}_obj_1mt = stablehlo.subtract %{pa}_one1, %{pa}_m4 : {c1}\n"
    s := s ++ s!"    %{pa}_obj_1mp = stablehlo.subtract %{pa}_one1, %{pa}_obj_p : {c1}\n"
    s := s ++ s!"    %{pa}_obj_tp = stablehlo.multiply %{pa}_m4, %{pa}_obj_p : {c1}\n"
    s := s ++ s!"    %{pa}_obj_btm = stablehlo.multiply %{pa}_obj_1mt, %{pa}_obj_1mp : {c1}\n"
    s := s ++ s!"    %{pa}_obj_pt = stablehlo.add %{pa}_obj_tp, %{pa}_obj_btm : {c1}\n"
    s := s ++ s!"    %{pa}_obj_1mpt = stablehlo.subtract %{pa}_one1, %{pa}_obj_pt : {c1}\n"
    s := s ++ s!"    %{pa}_obj_1mptc = stablehlo.maximum %{pa}_obj_1mpt, %{pa}_eps1 : {c1}\n"
    s := s ++ s!"    %{pa}_obj_log = stablehlo.log %{pa}_obj_1mptc : {c1}\n"
    s := s ++ s!"    %{pa}_obj_glog = stablehlo.multiply %{pa}_gam1, %{pa}_obj_log : {c1}\n"
    s := s ++ s!"    %{pa}_obj_w = stablehlo.exponential %{pa}_obj_glog : {c1}\n"
    s := s ++ s!"    %{pa}_obj_abg = stablehlo.multiply %{pa}_obj_1mt, %{pa}_ln1 : {c1}\n"
    s := s ++ s!"    %{pa}_obj_al = stablehlo.add %{pa}_m4, %{pa}_obj_abg : {c1}\n"
    s := s ++ s!"    %{pa}_obj_relu = stablehlo.maximum %{pa}_op, %{pa}_zero1 : {c1}\n"
    s := s ++ s!"    %{pa}_obj_zt = stablehlo.multiply %{pa}_op, %{pa}_m4 : {c1}\n"
    s := s ++ s!"    %{pa}_obj_abz = stablehlo.abs %{pa}_op : {c1}\n"
    s := s ++ s!"    %{pa}_obj_nabz = stablehlo.negate %{pa}_obj_abz : {c1}\n"
    s := s ++ s!"    %{pa}_obj_en = stablehlo.exponential %{pa}_obj_nabz : {c1}\n"
    s := s ++ s!"    %{pa}_obj_1pen = stablehlo.add %{pa}_one1, %{pa}_obj_en : {c1}\n"
    s := s ++ s!"    %{pa}_obj_sp = stablehlo.log %{pa}_obj_1pen : {c1}\n"
    s := s ++ s!"    %{pa}_obj_bce0 = stablehlo.subtract %{pa}_obj_relu, %{pa}_obj_zt : {c1}\n"
    s := s ++ s!"    %{pa}_obj_bce = stablehlo.add %{pa}_obj_bce0, %{pa}_obj_sp : {c1}\n"
    s := s ++ s!"    %{pa}_obj_aw = stablehlo.multiply %{pa}_obj_al, %{pa}_obj_w : {c1}\n"
    s := s ++ s!"    %{pa}_obj_cell = stablehlo.multiply %{pa}_obj_aw, %{pa}_obj_bce : {c1}\n"
    s := s ++ s!"    %{pa}_objloss = stablehlo.reduce(%{pa}_obj_cell init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
    s := s ++ s!"           : ({c1}, tensor<f32>) -> tensor<f32>\n"
    s := s ++ s!"    %{pa}_obj_pmt = stablehlo.subtract %{pa}_obj_p, %{pa}_m4 : {c1}\n"
    s := s ++ s!"    %{pa}_obj_gr0 = stablehlo.multiply %{pa}_obj_aw, %{pa}_obj_pmt : {c1}\n"
    s := s ++ s!"    %{pa}_Bf1 = stablehlo.broadcast_in_dim %ay_Bf, dims = [] : (tensor<f32>) -> {c1}\n"
    s := s ++ s!"    %{pa}_og = stablehlo.divide %{pa}_obj_gr0, %{pa}_Bf1 : {c1}\n"
    -- ── Class softmax-CE (masked by anchor a's assignment) ──
    s := s ++ s!"    %{pa}_cls_mx = stablehlo.reduce(%{pa}_cp init: %neginf) applies stablehlo.maximum across dimensions = [1]\n"
    s := s ++ s!"           : ({clsT}, tensor<f32>) -> {clsRed}\n"
    s := s ++ s!"    %{pa}_cls_mxb = stablehlo.broadcast_in_dim %{pa}_cls_mx, dims = [0, 2, 3] : ({clsRed}) -> {clsT}\n"
    s := s ++ s!"    %{pa}_cls_sh = stablehlo.subtract %{pa}_cp, %{pa}_cls_mxb : {clsT}\n"
    s := s ++ s!"    %{pa}_cls_ex = stablehlo.exponential %{pa}_cls_sh : {clsT}\n"
    s := s ++ s!"    %{pa}_cls_se = stablehlo.reduce(%{pa}_cls_ex init: %zf) applies stablehlo.add across dimensions = [1]\n"
    s := s ++ s!"           : ({clsT}, tensor<f32>) -> {clsRed}\n"
    s := s ++ s!"    %{pa}_cls_lse = stablehlo.log %{pa}_cls_se : {clsRed}\n"
    s := s ++ s!"    %{pa}_cls_lseb = stablehlo.broadcast_in_dim %{pa}_cls_lse, dims = [0, 2, 3] : ({clsRed}) -> {clsT}\n"
    s := s ++ s!"    %{pa}_cls_lsm = stablehlo.subtract %{pa}_cls_sh, %{pa}_cls_lseb : {clsT}\n"
    s := s ++ s!"    %{pa}_cls_maskb = stablehlo.broadcast_in_dim %{pa}_m4, dims = [0, 1, 2, 3] : ({c1}) -> {clsT}\n"
    s := s ++ s!"    %{pa}_cls_nll = stablehlo.multiply %{pa}_ct, %{pa}_cls_lsm : {clsT}\n"
    s := s ++ s!"    %{pa}_cls_nllm = stablehlo.multiply %{pa}_cls_nll, %{pa}_cls_maskb : {clsT}\n"
    -- two-step reduce (channel then spatial): the llvm-cpu backend can't lower a
    -- [B,NC,gH,gW]→scalar reduce (vector.contract), but ROCm can; this is
    -- equivalent and CPU-compilable so the FD probe runs on CPU.
    s := s ++ s!"    %{pa}_cls_ch = stablehlo.reduce(%{pa}_cls_nllm init: %zf) applies stablehlo.add across dimensions = [1]\n"
    s := s ++ s!"           : ({clsT}, tensor<f32>) -> {clsRed}\n"
    s := s ++ s!"    %{pa}_cls_ce = stablehlo.reduce(%{pa}_cls_ch init: %zf) applies stablehlo.add across dimensions = [0, 1, 2]\n"
    s := s ++ s!"           : ({clsRed}, tensor<f32>) -> tensor<f32>\n"
    s := s ++ s!"    %{pa}_clsloss = stablehlo.negate %{pa}_cls_ce : tensor<f32>\n"
    s := s ++ s!"    %{pa}_cls_seb = stablehlo.broadcast_in_dim %{pa}_cls_se, dims = [0, 2, 3] : ({clsRed}) -> {clsT}\n"
    s := s ++ s!"    %{pa}_cls_sm = stablehlo.divide %{pa}_cls_ex, %{pa}_cls_seb : {clsT}\n"
    s := s ++ s!"    %{pa}_cls_d = stablehlo.subtract %{pa}_cls_sm, %{pa}_ct : {clsT}\n"
    s := s ++ s!"    %{pa}_cls_dm = stablehlo.multiply %{pa}_cls_d, %{pa}_cls_maskb : {clsT}\n"
    s := s ++ s!"    %{pa}_BfC = stablehlo.broadcast_in_dim %ay_Bf, dims = [] : (tensor<f32>) -> {clsT}\n"
    s := s ++ s!"    %{pa}_cg = stablehlo.divide %{pa}_cls_dm, %{pa}_BfC : {clsT}\n"
    -- ── assemble anchor a's [B,15,gH,gW] gradient slab: box(4) ++ obj(1) ++ cls(10) ──
    s := s ++ s!"    %{pa}_g5 = stablehlo.concatenate %{pa}_bg, %{pa}_og, dim = 1 : ({box4}, {c1}) -> {tensorTy [B, 5, gH, gW]}\n"
    s := s ++ s!"    %{pa}_grad = stablehlo.concatenate %{pa}_g5, %{pa}_cg, dim = 1 : ({tensorTy [B, 5, gH, gW]}, {clsT}) -> {slabTy}\n"
    s := s ++ s!"    %{pa}_bo = stablehlo.add %{pa}_boxloss, %{pa}_objloss : tensor<f32>\n"
    s := s ++ s!"    %{pa}_anch = stablehlo.add %{pa}_bo, %{pa}_clsloss : tensor<f32>\n"
    lossParts := lossParts ++ [s!"%{pa}_anch"]
    gradParts := gradParts ++ [s!"%{pa}_grad"]
  -- total loss = Σ_a anchor_a / B
  let mut acc := lossParts.headD "%zf"
  for (lp, i) in (lossParts.drop 1).zipIdx do
    s := s ++ s!"    %ay_ls{i} = stablehlo.add {acc}, {lp} : tensor<f32>\n"
    acc := s!"%ay_ls{i}"
  s := s ++ s!"    {lossOut} = stablehlo.divide {acc}, %ay_Bf : tensor<f32>\n"
  -- total grad = concat anchor slabs along channel dim → [B, A·15, gH, gW]
  let mut gacc := gradParts.headD "%zf"
  let mut accCh := P
  for (gp, i) in (gradParts.drop 1).zipIdx do
    let nextCh := accCh + P
    s := s ++ s!"    %ay_gc{i} = stablehlo.concatenate {gacc}, {gp}, dim = 1 : ({tensorTy [B, accCh, gH, gW]}, {slabTy}) -> {tensorTy [B, nextCh, gH, gW]}\n"
    gacc := s!"%ay_gc{i}"
    accCh := nextCh
  s := s ++ s!"    {gradOut} = stablehlo.reshape {gacc} : ({apT}) -> {apT}\n"
  return s

/-! ### FPN neck (top-down multi-scale merge) — detection-infra brick #3

    Composes 1×1 lateral convs + the already-verified `bilinearUpsample` + adds
    into the RetinaNet top-down pyramid. No new primitives: the merge is a DAG of
    existing ops. Forward/backward FD-verified in numpy first
    (`scripts/fpn_neck_check.py`), then these emitters are checked against that
    oracle by `scripts/fpn_neck_probe_check.py` (the `fpn-neck-probe` exe). -/

/-- 1×1 conv (channel mix) forward: `einsum('oi,bihw->bohw', W, x)`.
    x `[b,ic,h,w]`, W `[oc,ic]` → `[b,oc,h,w]`. dot_general contracting the input
    channel dim, then transpose the trailing out-channel back to NCHW. -/
private def emitConv1x1Fwd (pfx xSSA wSSA : String) (b ic oc h w : Nat)
    : String × String := Id.run do
  let xTy := tensorTy [b, ic, h, w]
  let wTy := s!"tensor<{oc}x{ic}xf32>"
  let dgTy := tensorTy [b, h, w, oc]
  let outTy := tensorTy [b, oc, h, w]
  let mut s := ""
  s := s ++ s!"    %{pfx}_dg = stablehlo.dot_general {xSSA}, {wSSA}, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({xTy}, {wTy}) -> {dgTy}\n"
  s := s ++ s!"    %{pfx}_o = stablehlo.transpose %{pfx}_dg, dims = [0, 3, 1, 2] : ({dgTy}) -> {outTy}\n"
  return (s, s!"%{pfx}_o")

/-- 1×1 conv input-VJP: `einsum('oi,bohw->bihw', W, dP)` → `dX [b,ic,h,w]`.
    Contract the out-channel dim of `dP` against W's out dim. -/
private def emitConv1x1BwdDC (pfx wSSA dpSSA : String) (b ic oc h w : Nat)
    : String × String := Id.run do
  let wTy := s!"tensor<{oc}x{ic}xf32>"
  let dpTy := tensorTy [b, oc, h, w]
  let dgTy := tensorTy [b, h, w, ic]
  let outTy := tensorTy [b, ic, h, w]
  let mut s := ""
  s := s ++ s!"    %{pfx}_dcg = stablehlo.dot_general {dpSSA}, {wSSA}, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({dpTy}, {wTy}) -> {dgTy}\n"
  s := s ++ s!"    %{pfx}_dc = stablehlo.transpose %{pfx}_dcg, dims = [0, 3, 1, 2] : ({dgTy}) -> {outTy}\n"
  return (s, s!"%{pfx}_dc")

/-- 1×1 conv weight-VJP: `einsum('bohw,bihw->oi', dP, x)` → `dW [oc,ic]`.
    Contract the batch + spatial dims jointly. -/
private def emitConv1x1BwdDW (pfx dpSSA xSSA : String) (b ic oc h w : Nat)
    : String × String := Id.run do
  let dpTy := tensorTy [b, oc, h, w]
  let xTy := tensorTy [b, ic, h, w]
  let outTy := s!"tensor<{oc}x{ic}xf32>"
  let s := s!"    %{pfx}_dw = stablehlo.dot_general {dpSSA}, {xSSA}, contracting_dims = [0, 2, 3] x [0, 2, 3], precision = [DEFAULT, DEFAULT] : ({dpTy}, {xTy}) -> {outTy}\n"
  return (s, s!"%{pfx}_dw")

/-- Standalone VJP of `emitBilinearUpsample` (`[b,c,h,w]` → `[b,c,h·s,w·s]`):
    the transpose matmul-pair. Mirrors the reverse-record `.bilinearUpsample`
    backward exactly, but as a reusable emitter so the FPN DAG can route an
    upsample cotangent up the pyramid. `inShape` is the (small) pre-upsample
    shape; `gradSSA` carries the (large) post-upsample cotangent. -/
private def emitBilinearUpsampleBwd (pos : Nat) (gradSSA : String)
    (inShape : List Nat) (scale : Nat) : String × String := Id.run do
  match inShape with
  | [b, c, h, w] =>
    let oH := h * scale
    let oW := w * scale
    let outShape := [b, c, oH, oW]
    let wyStr := floatMatrixToMlirDense (bilinearWeights1D h scale)
    let wxStr := floatMatrixToMlirDense (bilinearWeights1D w scale)
    let wyTy := s!"tensor<{oH}x{h}xf32>"
    let wxTy := s!"tensor<{oW}x{w}xf32>"
    let dmShape := [b, c, oH, w]
    let dxtShape := [b, c, w, h]
    let mut s := ""
    s := s ++ s!"    %buT_wy{pos} = stablehlo.constant dense<{wyStr}> : {wyTy}\n"
    s := s ++ s!"    %buT_wx{pos} = stablehlo.constant dense<{wxStr}> : {wxTy}\n"
    s := s ++ s!"    %buT_dm{pos} = stablehlo.dot_general {gradSSA}, %buT_wx{pos}, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : ({tensorTy outShape}, {wxTy}) -> {tensorTy dmShape}\n"
    s := s ++ s!"    %buT_dt{pos} = stablehlo.dot_general %buT_dm{pos}, %buT_wy{pos}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({tensorTy dmShape}, {wyTy}) -> {tensorTy dxtShape}\n"
    s := s ++ s!"    %buT_dx{pos} = stablehlo.transpose %buT_dt{pos}, dims = [0, 1, 3, 2] : ({tensorTy dxtShape}) -> {tensorTy inShape}\n"
    return (s, s!"%buT_dx{pos}")
  | _ => return ("    // emitBilinearUpsampleBwd error: expected rank-4 NCHW\n", gradSSA)

/-- FPN top-down neck forward. Backbone taps `C3/C4/C5` at strides 8/16/32 (grids
    `4·g5 / 2·g5 / g5`), lateral 1×1 convs `W3/W4/W5 : [oc,c_n]`:

        P5 = conv1x1(C5,W5)                  [b,oc,g5,g5]
        P4 = conv1x1(C4,W4) + upsample2(P5)  [b,oc,2·g5,2·g5]
        P3 = conv1x1(C3,W3) + upsample2(P4)  [b,oc,4·g5,4·g5]

    Returns the code and the three pyramid SSAs `(P3,P4,P5)`. -/
private def emitFpnNeckForward (c3SSA c4SSA c5SSA w3SSA w4SSA w5SSA : String)
    (b oc c3 c4 c5 g5 : Nat) : String × (String × String × String) := Id.run do
  let g4 := 2 * g5
  let g3 := 4 * g5
  let mut s := ""
  let (s5, p5) := emitConv1x1Fwd "fpn_p5" c5SSA w5SSA b c5 oc g5 g5
  s := s ++ s5
  let (su5, up5, _) := emitBilinearUpsample 801 p5 [b, oc, g5, g5] 2
  s := s ++ su5
  let (s4, lat4) := emitConv1x1Fwd "fpn_l4" c4SSA w4SSA b c4 oc g4 g4
  s := s ++ s4
  s := s ++ s!"    %fpn_p4 = stablehlo.add {lat4}, {up5} : {tensorTy [b, oc, g4, g4]}\n"
  let (su4, up4, _) := emitBilinearUpsample 802 "%fpn_p4" [b, oc, g4, g4] 2
  s := s ++ su4
  let (s3, lat3) := emitConv1x1Fwd "fpn_l3" c3SSA w3SSA b c3 oc g3 g3
  s := s ++ s3
  s := s ++ s!"    %fpn_p3 = stablehlo.add {lat3}, {up4} : {tensorTy [b, oc, g3, g3]}\n"
  return (s, ("%fpn_p3", "%fpn_p4", p5))

/-- FPN neck VJP. Each `Pn` cotangent routes to its lateral conv AND up the
    pyramid (accumulating: `dP4_tot = dP4 + up^T(dP3)`, then
    `dP5_tot = dP5 + up^T(dP4_tot)`). Returns the code, the backbone-tap
    cotangents `(dC3,dC4,dC5)`, and the lateral-weight gradients `(dW3,dW4,dW5)`.
    Mirrors `fpn_grad` in `scripts/fpn_neck_check.py`. -/
private def emitFpnNeckBackward (dP3SSA dP4SSA dP5SSA : String)
    (c3SSA c4SSA c5SSA w3SSA w4SSA w5SSA : String) (b oc c3 c4 c5 g5 : Nat)
    : String × (String × String × String) × (String × String × String) := Id.run do
  let g4 := 2 * g5
  let g3 := 4 * g5
  let mut s := ""
  -- dP4_tot = dP4 + up^T(dP3)   (up^T maps the g3 cotangent down to g4)
  let (sbu3, upT3) := emitBilinearUpsampleBwd 811 dP3SSA [b, oc, g4, g4] 2
  s := s ++ sbu3
  s := s ++ s!"    %fpn_dp4t = stablehlo.add {dP4SSA}, {upT3} : {tensorTy [b, oc, g4, g4]}\n"
  -- dP5_tot = dP5 + up^T(dP4_tot)   (g4 → g5)
  let (sbu4, upT4) := emitBilinearUpsampleBwd 812 "%fpn_dp4t" [b, oc, g5, g5] 2
  s := s ++ sbu4
  s := s ++ s!"    %fpn_dp5t = stablehlo.add {dP5SSA}, {upT4} : {tensorTy [b, oc, g5, g5]}\n"
  -- lateral input grads
  let (sc3, dc3) := emitConv1x1BwdDC "fpn_dc3" w3SSA dP3SSA b c3 oc g3 g3
  let (sc4, dc4) := emitConv1x1BwdDC "fpn_dc4" w4SSA "%fpn_dp4t" b c4 oc g4 g4
  let (sc5, dc5) := emitConv1x1BwdDC "fpn_dc5" w5SSA "%fpn_dp5t" b c5 oc g5 g5
  s := s ++ sc3 ++ sc4 ++ sc5
  -- lateral weight grads
  let (sw3, dw3) := emitConv1x1BwdDW "fpn_dw3" dP3SSA c3SSA b c3 oc g3 g3
  let (sw4, dw4) := emitConv1x1BwdDW "fpn_dw4" "%fpn_dp4t" c4SSA b c4 oc g4 g4
  let (sw5, dw5) := emitConv1x1BwdDW "fpn_dw5" "%fpn_dp5t" c5SSA b c5 oc g5 g5
  s := s ++ sw3 ++ sw4 ++ sw5
  return (s, (dc3, dc4, dc5), (dw3, dw4, dw5))

/-- Emit the full train step (forward + loss + backward + SGD). -/
private def emitTrainStepBody (spec : NetSpec) (batchSize : Nat) (_moduleName : String)
    (labelSmoothing : Float := 0.1) (weightDecay : Float := 0.0001) (useAdam : Bool := true)
    (useSoftLabels : Bool := false)
    (useFocal : Bool := false) (focalGamma : Float := 2.0)
    (useSeg : Bool := false)
    (useDdpm : Bool := false)
    (useYolov1 : Bool := false)
    (yoloGridH : Nat := 7) (yoloGridW : Nat := 7)
    (yoloNumBoxes : Nat := 2) (yoloNumClasses : Nat := 20)
    (gradClipNorm : Float := 0.0) (headLrMult : Float := 1.0)
    (useMuon : Bool := false)
    (useShampoo : Bool := false)
    (segLoss : SegLoss := .ce)
    (useDiouBox : Bool := false)
    (yoloAnchors : List (Float × Float) := [])
    : String := Id.run do
  let B := batchSize
  let nClasses := spec.numClasses
  let mut code : String := ""
  code := code ++ "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  code := code ++ "    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  if useAdam || useMuon || useShampoo then
    -- Adam constants (Muon's non-2D params fall back to AdamW, so it needs these too)
    code := code ++ "    // Adam constants\n"
    code := code ++ "    %beta1 = stablehlo.constant dense<0.9> : tensor<f32>\n"
    code := code ++ "    %beta2 = stablehlo.constant dense<0.999> : tensor<f32>\n"
    code := code ++ "    %one_minus_b1 = stablehlo.constant dense<0.1> : tensor<f32>\n"
    code := code ++ "    %one_minus_b2 = stablehlo.constant dense<0.001> : tensor<f32>\n"
    code := code ++ "    %adam_eps = stablehlo.constant dense<1.0e-08> : tensor<f32>\n"
    code := code ++ "    // Bias correction: bc1 = 1 - β1^t, bc2 = 1 - β2^t\n"
    code := code ++ "    %one_scalar = stablehlo.constant dense<1.0> : tensor<f32>\n"
    code := code ++ "    %b1t = stablehlo.power %beta1, %t : tensor<f32>\n"
    code := code ++ "    %bc1 = stablehlo.subtract %one_scalar, %b1t : tensor<f32>\n"
    code := code ++ "    %b2t = stablehlo.power %beta2, %t : tensor<f32>\n"
    code := code ++ "    %bc2 = stablehlo.subtract %one_scalar, %b2t : tensor<f32>\n"
  else
    code := code ++ "    // SGD+momentum constants\n"
    code := code ++ "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n"
  if useMuon then
    -- Muon (Newton–Schulz polar projection) constants — see emitMuonUpdate
    code := code ++ "    // Muon constants\n"
    code := code ++ "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n"
    code := code ++ "    %ns_a = stablehlo.constant dense<3.4445> : tensor<f32>\n"
    code := code ++ "    %ns_b = stablehlo.constant dense<-4.775> : tensor<f32>\n"
    code := code ++ "    %ns_c = stablehlo.constant dense<2.0315> : tensor<f32>\n"
    code := code ++ "    %ns_eps = stablehlo.constant dense<1.0e-07> : tensor<f32>\n"
  if useShampoo then
    -- Shampoo constants (see emitShampooUpdate). EMA β, coupled-Newton 0.5/1.5,
    -- trace-relative regularization ε. Non-square / non-2D params fall back to AdamW.
    code := code ++ "    // Shampoo constants\n"
    code := code ++ "    %sh_beta = stablehlo.constant dense<0.95> : tensor<f32>\n"
    code := code ++ "    %sh_ombeta = stablehlo.constant dense<0.05> : tensor<f32>\n"
    code := code ++ "    %sh_half = stablehlo.constant dense<0.5> : tensor<f32>\n"
    code := code ++ "    %sh_1p5 = stablehlo.constant dense<1.5> : tensor<f32>\n"
    code := code ++ "    %sh_eps = stablehlo.constant dense<1.0e-04> : tensor<f32>\n"
  if weightDecay > 0.0 then
    code := code ++ s!"    %wdecay = stablehlo.constant dense<{weightDecay}> : tensor<f32>\n"
  code := code ++ "\n"

  -- ═══════════════ FORWARD PASS ═══════════════
  code := code ++ "    // ======================== FORWARD ========================\n"
  let mut curSSA : String := "%x_flat"
  let mut curShape : List Nat := [B, inputFlatDim spec]

  -- Reshape if CNN
  match inputChannels spec with
  | some ic =>
    let nchw := [B, ic, spec.imageH, spec.imageW]
    code := code ++ s!"    %x = stablehlo.reshape %x_flat : ({tensorTy curShape}) -> {tensorTy nchw}\n"
    curSSA := "%x"
    curShape := nchw
  | none => curSSA := "%x_flat"

  let mut pidx : Nat := 0
  let mut pos : Nat := 0
  let mut records : Array FwdRec := #[]
  -- UNet skip plumbing. `unetEncStack` holds encoder indices (assigned
  -- monotonically as `unetDown`s are seen) waiting for their matching
  -- `unetUp`. Pairing is LIFO. The encoder index also names the SSA the
  -- backward pass uses to thread the skip gradient: `%unet_skip_g{e}`.
  let mut unetEncStack : List Nat := []
  let mut nextUnetEncIdx : Nat := 0

  for l in spec.layers do
    let inSSA := curSSA
    let inShape := curShape
    match l with
    | .conv2d ic oc kSize _pad act =>
      match curShape with
      | [b, _, h, w] =>
        let outShape := [b, oc, h, w]
        let pad := (kSize - 1) / 2
        code := code ++ s!"    %cv{pidx} = \"stablehlo.convolution\"({curSSA}, %W{pidx}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy curShape}, {tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy outShape}\n"
        code := code ++ s!"    %cvb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
        let preSSA := s!"%cva{pidx}"
        code := code ++ s!"    {preSSA} = stablehlo.add %cv{pidx}, %cvb{pidx} : {tensorTy outShape}\n"
        if act == .relu then
          code := code ++ s!"    %cvz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
          code := code ++ s!"    %cvr{pidx} = stablehlo.maximum {preSSA}, %cvz{pidx} : {tensorTy outShape}\n"
          curSSA := s!"%cvr{pidx}"
        else
          curSSA := preSSA
        records := records.push { layer := l, pidx := some pidx, pos, inputSSA := inSSA, preActSSA := preSSA, outputSSA := curSSA, inShape, outShape }
        curShape := outShape
        pidx := pidx + 1
      | _ => pure ()

    | .dense fanIn fanOut act =>
      let outShape := [B, fanOut]
      code := code ++ s!"    %mm{pidx} = stablehlo.dot_general {curSSA}, %W{pidx}, contracting_dims = [1] x [0],\n"
      code := code ++ s!"             precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"           : ({tensorTy [B, fanIn]}, {tensorTy [fanIn, fanOut]}) -> {tensorTy outShape}\n"
      code := code ++ s!"    %bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [fanOut]}) -> {tensorTy outShape}\n"
      let preSSA := s!"%ab{pidx}"
      code := code ++ s!"    {preSSA} = stablehlo.add %mm{pidx}, %bb{pidx} : {tensorTy outShape}\n"
      if act == .relu then
        code := code ++ s!"    %dz{pidx} = stablehlo.constant dense<0.0> : {tensorTy outShape}\n"
        code := code ++ s!"    %dr{pidx} = stablehlo.maximum {preSSA}, %dz{pidx} : {tensorTy outShape}\n"
        curSSA := s!"%dr{pidx}"
      else
        curSSA := preSSA
      records := records.push { layer := l, pidx := some pidx, pos, inputSSA := inSSA, preActSSA := preSSA, outputSSA := curSSA, inShape, outShape }
      curShape := outShape
      pidx := pidx + 1

    | .maxPool size stride =>
      match curShape with
      | [b, c, h, w] =>
        let oH := (h + stride - 1) / stride
        let oW := (w + stride - 1) / stride
        let outShape := [b, c, oH, oW]
        let padNeeded := (oH - 1) * stride + size
        let padH := if padNeeded > h then padNeeded - h else 0
        let padW := if padNeeded > w then padNeeded - w else 0
        let padH0 := padH / 2; let padH1 := padH - padH0
        let padW0 := padW / 2; let padW1 := padW - padW0
        code := code ++ s!"    %pli{pos} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
        -- Explicit pad for SAME pooling (IREE doesn't support padded select_and_scatter)
        let hasPad := padH > 0 || padW > 0
        let paddedShape := [b, c, h + padH, w + padW]
        let mut poolInput := curSSA
        if hasPad then
          code := code ++ s!"    %plpad{pos} = stablehlo.pad {curSSA}, %pli{pos}, low = [0, 0, {padH0}, {padW0}], high = [0, 0, {padH1}, {padW1}], interior = [0, 0, 0, 0] : ({tensorTy curShape}, tensor<f32>) -> {tensorTy paddedShape}\n"
          poolInput := s!"%plpad{pos}"
        let poolInShape := if hasPad then paddedShape else curShape
        code := code ++ s!"    %pl{pos} = \"stablehlo.reduce_window\"({poolInput}, %pli{pos}) (" ++ "{\n"
        code := code ++ s!"      ^bb0(%rwa{pos}: tensor<f32>, %rwb{pos}: tensor<f32>):\n"
        code := code ++ s!"        %rwm{pos} = stablehlo.maximum %rwa{pos}, %rwb{pos} : tensor<f32>\n"
        code := code ++ s!"        \"stablehlo.return\"(%rwm{pos}) : (tensor<f32>) -> ()\n"
        code := code ++ "      }) " ++ "{" ++ s!"window_dimensions = array<i64: 1, 1, {size}, {size}>, "
        code := code ++ s!"window_strides = array<i64: 1, 1, {stride}, {stride}>" ++ "}\n"
        code := code ++ s!"      : ({tensorTy poolInShape}, tensor<f32>) -> {tensorTy outShape}\n"
        curSSA := s!"%pl{pos}"
        let savedInput := if hasPad then poolInput else inSSA
        let padInfo := s!"{padH0},{padH1},{padW0},{padW1}"
        let poolLayer := Layer.maxPool size stride
        let mut poolRec : FwdRec := default
        poolRec := { poolRec with layer := poolLayer, pidx := none, pos := pos }
        poolRec := { poolRec with inputSSA := savedInput, preActSSA := padInfo }
        poolRec := { poolRec with outputSSA := curSSA }
        poolRec := { poolRec with inShape := inShape, outShape := outShape }
        records := records.push poolRec
        curShape := outShape
      | _ => pure ()

    | .flatten =>
      match curShape with
      | [b, c, h, w] =>
        let flat := c * h * w
        let outShape := [b, flat]
        code := code ++ s!"    %fl{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy outShape}\n"
        curSSA := s!"%fl{pos}"
        records := records.push { layer := .flatten, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := curSSA, inShape, outShape }
        curShape := outShape
      | _ => pure ()

    | .bilinearUpsample scale =>
      let (snip, newSSA, outShape) := emitBilinearUpsample pos curSSA curShape scale
      code := code ++ snip
      curSSA := newSSA
      records := records.push { layer := .bilinearUpsample scale, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := curSSA, inShape, outShape }
      curShape := outShape

    | .convBn ic oc kSize stride _ =>
      let (snip, rec) := emitConvBnTrain pidx pos curSSA curShape ic oc kSize stride true
      code := code ++ snip
      curSSA := rec.outputSSA
      curShape := rec.outShape
      records := records.push rec
      pidx := pidx + 1

    | .globalAvgPool =>
      match curShape with
      | [b, c, h, w] =>
        let outShape := [b, c]
        let spatialN := h * w
        code := code ++ s!"    %gap_zf{pos} = stablehlo.constant dense<0.0> : tensor<f32>\n"
        code := code ++ s!"    %gap_sum{pos} = stablehlo.reduce({curSSA} init: %gap_zf{pos}) applies stablehlo.add across dimensions = [2, 3]\n"
        code := code ++ s!"          : ({tensorTy curShape}, tensor<f32>) -> {tensorTy outShape}\n"
        code := code ++ s!"    %gap_N{pos} = stablehlo.constant dense<{spatialN}.0> : {tensorTy outShape}\n"
        code := code ++ s!"    %gap{pos} = stablehlo.divide %gap_sum{pos}, %gap_N{pos} : {tensorTy outShape}\n"
        curSSA := s!"%gap{pos}"
        records := records.push { layer := .globalAvgPool, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := curSSA, inShape, outShape }
        curShape := outShape
      | _ => pure ()

    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        -- conv1: 3×3, stride, relu
        let (s1, rec1) := emitConvBnTrain pidx pos curSSA curShape blockIc oc 3 stride true
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        -- conv2: 3×3, stride 1, NO relu
        let (s2, rec2) := emitConvBnTrain pidx pos curSSA curShape oc oc 3 1 false
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- Projection (if needed)
        let mut skipSSA := blockIn
        if bi == 0 && needsProj then
          let (sp, recp) := emitConvBnTrain pidx pos blockIn blockInShape ic oc 1 firstStride false
          code := code ++ sp; skipSSA := recp.outputSSA
          records := records.push recp; pidx := pidx + 1
        -- Add + ReLU
        let addId := s!"{pidx}_{bi}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {skipSSA} : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_rz{addId} = stablehlo.constant dense<0.0> : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_out{addId} = stablehlo.maximum %rb_add{addId}, %rb_rz{addId} : {tensorTy curShape}\n"
        curSSA := s!"%rb_out{addId}"
        -- Record the skip-add-relu as a special FwdRec
        -- The first convBn of this block (at records.size - 2 or - 3) gets the skip grad
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Mark the first convBn of this block to accumulate skip grad
        let firstIdx := if bi == 0 && needsProj then records.size - 3 else records.size - 2
        let firstRec := records[firstIdx]!
        records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
        -- Record the skip-add for backward
        records := records.push {
          layer := .globalAvgPool  -- reuse as marker (no params)
          pidx := none, pos
          inputSSA := blockIn      -- block input (for skip backward)
          preActSSA := s!"%rb_add{addId}"  -- pre-relu sum
          outputSSA := curSSA
          inShape := blockInShape  -- block input shape
          outShape := curShape
          hasRelu := true
          -- Store projection info in addSkipGrad field: "proj:{projPidx}" or "identity"
          addSkipGrad := if bi == 0 && needsProj then s!"proj:{pidx - 1}" else "identity"
        }

    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        -- conv1: 1×1 reduce, relu
        let (s1, rec1) := emitConvBnTrain pidx pos curSSA curShape blockIc mid 1 1 true
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        -- conv2: 3×3, stride, relu
        let (s2, rec2) := emitConvBnTrain pidx pos curSSA curShape mid mid 3 stride true
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- conv3: 1×1 expand, NO relu
        let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
        -- Projection (if needed)
        let mut skipSSA := blockIn
        if bi == 0 && needsProj then
          let (sp, recp) := emitConvBnTrain pidx pos blockIn blockInShape ic oc 1 firstStride false
          code := code ++ sp; skipSSA := recp.outputSSA
          records := records.push recp; pidx := pidx + 1
        -- Add + ReLU
        let addId := s!"bn{pidx}_{bi}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {skipSSA} : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_rz{addId} = stablehlo.constant dense<0.0> : {tensorTy curShape}\n"
        code := code ++ s!"    %rb_out{addId} = stablehlo.maximum %rb_add{addId}, %rb_rz{addId} : {tensorTy curShape}\n"
        curSSA := s!"%rb_out{addId}"
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Mark the first convBn of this block to accumulate skip grad
        -- bottleneck: 3 convs + optional proj
        let firstIdx := if bi == 0 && needsProj then records.size - 4 else records.size - 3
        let firstRec := records[firstIdx]!
        records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
        records := records.push {
          layer := .globalAvgPool
          pidx := none, pos
          inputSSA := blockIn
          preActSSA := s!"%rb_add{addId}"
          outputSSA := curSSA
          inShape := blockInShape
          outShape := curShape
          hasRelu := true
          addSkipGrad := if bi == 0 && needsProj then s!"proj:{pidx - 1}" else "identity"
        }

    | .invertedResidual ic oc expand firstStride nBlocks =>
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let useSkip := stride == 1 && blockIc == oc
        -- Expand: 1×1, ReLU6 (skip if expand==1)
        if expand != 1 then
          let (s1, rec1) := emitConvBnTrainRelu6 pidx pos curSSA curShape blockIc mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1; pidx := pidx + 1
        -- Depthwise: 3×3, ReLU6
        let (s2, rec2) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid 3 stride
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- Project: 1×1, NO activation
        let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
        -- Skip connection: stride==1 AND blockIc==oc, NO ReLU after add
        if useSkip then
          let addId := s!"ir{pidx}_{bi}"
          code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
          curSSA := s!"%rb_add{addId}"
          let skipGradSSA := s!"%rb_dskip{addId}"
          -- Mark the first convBn of this block to accumulate skip grad
          -- invertedResidual with expand: expand + dw + proj = 3 records
          -- invertedResidual without expand: dw + proj = 2 records
          let nLayersInBlock := if expand != 1 then 3 else 2
          let firstIdx := records.size - nLayersInBlock
          let firstRec := records[firstIdx]!
          records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
          -- Record the skip-add for backward (NO ReLU, hasRelu := false)
          records := records.push {
            layer := .globalAvgPool  -- reuse as marker
            pidx := none, pos
            inputSSA := blockIn
            preActSSA := s!"%rb_add{addId}"  -- the add result (pre-act is same since no relu)
            outputSSA := curSSA
            inShape := blockInShape
            outShape := curShape
            hasRelu := false  -- NO ReLU after skip add in MobileNetV2
            addSkipGrad := "identity"
          }

    | .mbConv ic oc expand kSize firstStride nBlocks useSE act =>
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let useSkip := stride == 1 && blockIc == oc
        -- Expand: 1×1 + activation (skip if expand==1). Dispatch on `act`.
        if expand != 1 then
          let (s1, rec1) := match act with
            | .swish  => emitConvBnTrainSwish pidx pos curSSA curShape blockIc mid 1 1
            | .hSwish => emitConvBnTrainHSwish pidx pos curSSA curShape blockIc mid 1 1
            | _       => emitConvBnTrainRelu6 pidx pos curSSA curShape blockIc mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1; pidx := pidx + 1
        -- Depthwise: k×k + same activation
        let (s2, rec2) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid kSize stride
                            (useSwish := act == .swish)
                            (useHSwish := act == .hSwish)
                            (useRelu := act == .relu)
        code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
        records := records.push rec2; pidx := pidx + 1
        -- SE block (optional): insert between depthwise and project
        if useSE then
          match curShape with
          | [b, _, h, w] =>
            let tag := s!"_t{pidx}_{bi}"
            let seIn := curSSA
            let xTy := tensorTy curShape
            let pRed := pidx
            let pExp := pidx + 1
            -- 1. GAP (B, mid, H, W) → (B, mid)
            code := code ++ s!"    %se_gs{tag} = stablehlo.reduce({seIn} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
            -- 2. Reduce dense
            code := code ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
            code := code ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
            -- 3. Swish on reduce output
            code := code ++ s!"    %se_rsig{tag} = stablehlo.logistic %se_rb{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rsw{tag} = stablehlo.multiply %se_rb{tag}, %se_rsig{tag} : {tensorTy [b, seMid]}\n"
            -- 4. Expand dense
            code := code ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
            code := code ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
            -- 5. Sigmoid + broadcast multiply
            code := code ++ s!"    %se_sig{tag} = stablehlo.logistic %se_eb{tag} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
            code := code ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
            code := code ++ s!"    %se_out{tag} = stablehlo.multiply {seIn}, %se_sigb{tag} : {xTy}\n"
            curSSA := s!"%se_out{tag}"
            -- Record SE block
            records := records.push {
              layer := .globalAvgPool  -- abused marker; isSE discriminates
              pidx := none, pos
              inputSSA := seIn
              preActSSA := ""
              outputSSA := curSSA
              inShape := curShape
              outShape := curShape
              hasRelu := false
              isSE := true
              sePidxRed := pRed
              sePidxExp := pExp
              seMid := seMid
              seMidFull := mid
              seInputSSA := seIn
              seGapSSA := s!"%se_g{tag}"
              seRbSSA := s!"%se_rb{tag}"
              seRswSSA := s!"%se_rsw{tag}"
              seEbSSA := s!"%se_eb{tag}"
              seSigSSA := s!"%se_sig{tag}"
              seOutSSA := curSSA
            }
            pidx := pidx + 2
          | _ => pure ()
        -- Project: 1×1, NO activation
        let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
        -- Skip connection: stride==1 AND blockIc==oc, NO post-add activation
        if useSkip then
          let addId := s!"mb{pidx}_{bi}"
          code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
          curSSA := s!"%rb_add{addId}"
          let skipGradSSA := s!"%rb_dskip{addId}"
          -- Number of layer records in this block: expand?(1) + dw(1) + se?(1) + proj(1)
          let nLayersInBlock := (if expand != 1 then 1 else 0) + 1 + (if useSE then 1 else 0) + 1
          let firstIdx := records.size - nLayersInBlock
          let firstRec := records[firstIdx]!
          records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
          records := records.push {
            layer := .globalAvgPool
            pidx := none, pos
            inputSSA := blockIn
            preActSSA := s!"%rb_add{addId}"
            outputSSA := curSSA
            inShape := blockInShape
            outShape := curShape
            hasRelu := false
            addSkipGrad := "identity"
          }

    | .fusedMbConv ic oc expand kSize firstStride nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockInShape := curShape
        let stride := if bi == 0 then firstStride else 1
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let useSkip := stride == 1 && blockIc == oc
        -- 1. Fused expand: k×k regular convBn, Swish
        let (s1, rec1) := emitConvBnTrainSwish pidx pos curSSA curShape blockIc mid kSize stride
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        -- 2. SE block (optional): insert between expand and project
        if useSE then
          match curShape with
          | [b, _, h, w] =>
            let tag := s!"_tf{pidx}_{bi}"
            let seIn := curSSA
            let xTy := tensorTy curShape
            let pRed := pidx
            let pExp := pidx + 1
            -- 1. GAP (B, mid, H, W) → (B, mid)
            code := code ++ s!"    %se_gs{tag} = stablehlo.reduce({seIn} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
            -- 2. Reduce dense
            code := code ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
            code := code ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
            -- 3. Swish on reduce output
            code := code ++ s!"    %se_rsig{tag} = stablehlo.logistic %se_rb{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %se_rsw{tag} = stablehlo.multiply %se_rb{tag}, %se_rsig{tag} : {tensorTy [b, seMid]}\n"
            -- 4. Expand dense
            code := code ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
            code := code ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag}, contracting_dims = [1] x [1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
            -- 5. Sigmoid + broadcast multiply
            code := code ++ s!"    %se_sig{tag} = stablehlo.logistic %se_eb{tag} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
            code := code ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
            code := code ++ s!"    %se_out{tag} = stablehlo.multiply {seIn}, %se_sigb{tag} : {xTy}\n"
            curSSA := s!"%se_out{tag}"
            records := records.push {
              layer := .globalAvgPool
              pidx := none, pos
              inputSSA := seIn
              preActSSA := ""
              outputSSA := curSSA
              inShape := curShape
              outShape := curShape
              hasRelu := false
              isSE := true
              sePidxRed := pRed
              sePidxExp := pExp
              seMid := seMid
              seMidFull := mid
              seInputSSA := seIn
              seGapSSA := s!"%se_g{tag}"
              seRbSSA := s!"%se_rb{tag}"
              seRswSSA := s!"%se_rsw{tag}"
              seEbSSA := s!"%se_eb{tag}"
              seSigSSA := s!"%se_sig{tag}"
              seOutSSA := curSSA
            }
            pidx := pidx + 2
          | _ => pure ()
        -- 3. Project: 1×1, NO activation (only if expand != 1)
        if expand != 1 then
          let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
          code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
          records := records.push rec3; pidx := pidx + 1
        -- 4. Skip connection: stride==1 AND blockIc==oc, NO post-add activation
        if useSkip then
          let addId := s!"fmb{pidx}_{bi}"
          code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
          curSSA := s!"%rb_add{addId}"
          let skipGradSSA := s!"%rb_dskip{addId}"
          -- Number of layer records in this block: fused(1) + se?(1) + proj?(1)
          let nLayersInBlock := 1 + (if useSE then 1 else 0) + (if expand != 1 then 1 else 0)
          let firstIdx := records.size - nLayersInBlock
          let firstRec := records[firstIdx]!
          records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
          records := records.push {
            layer := .globalAvgPool
            pidx := none, pos
            inputSSA := blockIn
            preActSSA := s!"%rb_add{addId}"
            outputSSA := curSSA
            inShape := blockInShape
            outShape := curShape
            hasRelu := false
            addSkipGrad := "identity"
          }

    | .mbConvV3 ic oc expandCh kSize stride useSE act =>
      -- Single block variant of mbConv:
      -- expand (if expandCh != ic) → depthwise → optional SE → project → optional skip
      let useHSwish := act == .hSwish
      let blockIn := curSSA
      let blockInShape := curShape
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let useSkip := stride == 1 && ic == oc
      -- Expand: 1×1, h-swish (if useHSwish) or ReLU6
      if expandCh != ic then
        if useHSwish then
          let (s1, rec1) := emitConvBnTrainHSwish pidx pos curSSA curShape ic mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1
        else
          let (s1, rec1) := emitConvBnTrainRelu6 pidx pos curSSA curShape ic mid 1 1
          code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
          records := records.push rec1
        pidx := pidx + 1
      -- Depthwise: k×k, h-swish if useHSwish else ReLU6
      let (s2, rec2) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid kSize stride
                          (useHSwish := useHSwish)
      code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
      records := records.push rec2; pidx := pidx + 1
      -- SE block (optional)
      if useSE then
        match curShape with
        | [b, _, h, w] =>
          let tag := s!"_t3{pidx}"
          let seIn := curSSA
          let xTy := tensorTy curShape
          let pRed := pidx
          let pExp := pidx + 1
          code := code ++ s!"    %se_gs{tag} = stablehlo.reduce({seIn} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
          code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_gN{tag} = stablehlo.constant dense<{h * w}.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_g{tag} = stablehlo.divide %se_gs{tag}, %se_gN{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
          code := code ++ s!"    %se_rm{tag} = stablehlo.dot_general %se_g{tag}, %se_Wr{tag}, contracting_dims = [1] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_rbb{tag} = stablehlo.broadcast_in_dim %b{pRed}, dims = [1] : ({tensorTy [seMid]}) -> {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_rb{tag} = stablehlo.add %se_rm{tag}, %se_rbb{tag} : {tensorTy [b, seMid]}\n"
          -- Plain ReLU on reduce (MobileNetV3 SE variant)
          code := code ++ s!"    %se_rzero{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_rsw{tag} = stablehlo.maximum %se_rb{tag}, %se_rzero{tag} : {tensorTy [b, seMid]}\n"
          code := code ++ s!"    %se_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
          code := code ++ s!"    %se_em{tag} = stablehlo.dot_general %se_rsw{tag}, %se_We{tag}, contracting_dims = [1] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_ebb{tag} = stablehlo.broadcast_in_dim %b{pExp}, dims = [1] : ({tensorTy [mid]}) -> {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_eb{tag} = stablehlo.add %se_em{tag}, %se_ebb{tag} : {tensorTy [b, mid]}\n"
          -- h-sigmoid on expand: ReLU6(x + 3) / 6 (MobileNetV3)
          code := code ++ s!"    %se_sig3{tag} = stablehlo.constant dense<3.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sig6{tag} = stablehlo.constant dense<6.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigz{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigxp3{tag} = stablehlo.add %se_eb{tag}, %se_sig3{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigclip{tag} = stablehlo.minimum %se_sigxp3{tag}, %se_sig6{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigr6{tag} = stablehlo.maximum %se_sigclip{tag}, %se_sigz{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sig{tag} = stablehlo.divide %se_sigr6{tag}, %se_sig6{tag} : {tensorTy [b, mid]}\n"
          code := code ++ s!"    %se_sigr{tag} = stablehlo.reshape %se_sig{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
          code := code ++ s!"    %se_sigb{tag} = stablehlo.broadcast_in_dim %se_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
          code := code ++ s!"    %se_out{tag} = stablehlo.multiply {seIn}, %se_sigb{tag} : {xTy}\n"
          curSSA := s!"%se_out{tag}"
          records := records.push {
            layer := .globalAvgPool
            pidx := none, pos
            inputSSA := seIn
            preActSSA := ""
            outputSSA := curSSA
            inShape := curShape
            outShape := curShape
            hasRelu := false
            isSE := true
            sePidxRed := pRed
            sePidxExp := pExp
            seMid := seMid
            seMidFull := mid
            seInputSSA := seIn
            seGapSSA := s!"%se_g{tag}"
            seRbSSA := s!"%se_rb{tag}"
            seRswSSA := s!"%se_rsw{tag}"
            seEbSSA := s!"%se_eb{tag}"
            seSigSSA := s!"%se_sig{tag}"
            seOutSSA := curSSA
            seVariant := true
          }
          pidx := pidx + 2
        | _ => pure ()
      -- Project: 1×1, NO activation
      let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
      code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
      records := records.push rec3; pidx := pidx + 1
      -- Skip connection: stride==1 AND ic==oc, NO post-add activation
      if useSkip then
        let addId := s!"mb3{pidx}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
        curSSA := s!"%rb_add{addId}"
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Number of layer records in this block: expand?(1) + dw(1) + se?(1) + proj(1)
        let nLayersInBlock := (if expandCh != ic then 1 else 0) + 1 + (if useSE then 1 else 0) + 1
        let firstIdx := records.size - nLayersInBlock
        let firstRec := records[firstIdx]!
        records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
        records := records.push {
          layer := .globalAvgPool
          pidx := none, pos
          inputSSA := blockIn
          preActSSA := s!"%rb_add{addId}"
          outputSSA := curSSA
          inShape := blockInShape
          outShape := curShape
          hasRelu := false
          addSkipGrad := "identity"
        }

    | .uib ic oc expand stride preDWk postDWk =>
      -- UIB (MobileNet V4) block: optional preDW → expand 1×1 → optional postDW → project 1×1
      -- Plain ReLU throughout. Skip if stride==1 && ic==oc.
      let blockIn := curSSA
      let blockInShape := curShape
      let mid := ic * expand
      let useSkip := stride == 1 && ic == oc
      -- 1. Optional preDW: consumes stride, with plain ReLU
      let mut effectiveStride := stride
      if preDWk > 0 then
        let (s1, rec1) := emitDepthwiseConvBnTrain pidx pos curSSA curShape ic preDWk stride
                             (useRelu := true)
        code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
        records := records.push rec1; pidx := pidx + 1
        effectiveStride := 1
      -- 2. Expand 1×1 convBn + plain ReLU
      let (s2, rec2) := emitConvBnTrain pidx pos curSSA curShape ic mid 1 1 true
      code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
      records := records.push rec2; pidx := pidx + 1
      -- 3. Optional postDW with plain ReLU
      if postDWk > 0 then
        let (s3, rec3) := emitDepthwiseConvBnTrain pidx pos curSSA curShape mid postDWk effectiveStride
                             (useRelu := true)
        code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
        records := records.push rec3; pidx := pidx + 1
      -- 4. Project 1×1 convBn, NO activation
      let (s4, rec4) := emitConvBnTrain pidx pos curSSA curShape mid oc 1 1 false
      code := code ++ s4; curSSA := rec4.outputSSA; curShape := rec4.outShape
      records := records.push rec4; pidx := pidx + 1
      -- 5. Skip connection: stride==1 AND ic==oc, NO post-add activation
      if useSkip then
        let addId := s!"uib{pidx}"
        code := code ++ s!"    %rb_add{addId} = stablehlo.add {curSSA}, {blockIn} : {tensorTy curShape}\n"
        curSSA := s!"%rb_add{addId}"
        let skipGradSSA := s!"%rb_dskip{addId}"
        -- Number of layer records in this block: preDW?(1) + expand(1) + postDW?(1) + proj(1)
        let nLayersInBlock := (if preDWk > 0 then 1 else 0) + 1 + (if postDWk > 0 then 1 else 0) + 1
        let firstIdx := records.size - nLayersInBlock
        let firstRec := records[firstIdx]!
        records := records.set! firstIdx { firstRec with addSkipGrad := skipGradSSA }
        records := records.push {
          layer := .globalAvgPool
          pidx := none, pos
          inputSSA := blockIn
          preActSSA := s!"%rb_add{addId}"
          outputSSA := curSSA
          inShape := blockInShape
          outShape := curShape
          hasRelu := false
          addSkipGrad := "identity"
        }

    | .convNextStage channels nBlocks norm act =>
      -- Per block: DW raw → norm-NCHW → 1×1 expand → activation → 1×1 project → LayerScale → residual.
      -- One FwdRec per block with `isConvNextBlock := true`; backward uses saved intermediates.
      for bi in [:nBlocks] do
        match curShape with
        | [b, _, h, w] =>
          let blockIn := curSSA
          let blockShape := curShape
          let basePidx := pidx
          let tag := s!"cn{pos}_{bi}"
          let c := channels
          let blockTy := tensorTy blockShape
          let cTy := tensorTy [c]
          let expShape := [b, 4*c, h, w]
          let expTy := tensorTy expShape
          -- 1) DW raw 7×7 (pidx)
          let (pH0, pH1, pW0, pW1) := samePad h w 7 1
          code := code ++ s!"    %dwr_cv{pidx} = \"stablehlo.convolution\"({blockIn}, %W{pidx}) " ++ "{\n"
          code := code ++ "        batch_group_count = 1 : i64,\n"
          code := code ++ convDimNumbers
          code := code ++ s!"        feature_group_count = {c} : i64,\n"
          code := code ++ s!"        padding = dense<[[{pH0}, {pH1}], [{pW0}, {pW1}]]> : tensor<2x2xi64>,\n"
          code := code ++ "        rhs_dilation = array<i64: 1, 1>,\n"
          code := code ++ "        window_strides = array<i64: 1, 1>\n"
          code := code ++ s!"      " ++ "}" ++ s!" : ({blockTy}, {tensorTy [c, 1, 7, 7]}) -> {blockTy}\n"
          code := code ++ s!"    %dwr_bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({cTy}) -> {blockTy}\n"
          code := code ++ s!"    %dwr_out{pidx} = stablehlo.add %dwr_cv{pidx}, %dwr_bb{pidx} : {blockTy}\n"
          let dwOut := s!"%dwr_out{pidx}"
          pidx := pidx + 1
          -- 2) Norm (pidx): LN over channel axis OR BN per-channel.
          let mut lnOut : String := ""
          let mut lnNorm : String := ""    -- LN: [b, hw, c]; BN: [b, c, h, w]
          let mut lnIstd : String := ""    -- LN: [b, hw];     BN: [b, c, h, w] (broadcast)
          let mut bnMeanBc : String := ""
          let mut bnIstdBc : String := ""
          match norm with
          | .ln =>
            let (lc, out, nm, ist, _) := emitLayerNormForwardNCHW tag dwOut blockShape s!"%W{pidx}" s!"%b{pidx}"
            code := code ++ lc; lnOut := out; lnNorm := nm; lnIstd := ist
          | .bn =>
            let (bc, out, nm, mbc, ibc) := emitBatchNormForwardNCHW pidx dwOut blockShape s!"%W{pidx}" s!"%b{pidx}"
            code := code ++ bc; lnOut := out; lnNorm := nm; bnMeanBc := mbc; bnIstdBc := ibc
          pidx := pidx + 1
          -- 3) 1×1 expand c → 4c (pidx)
          code := code ++ s!"    %cnx_cv{pidx} = \"stablehlo.convolution\"({lnOut}, %W{pidx}) " ++ "{\n"
          code := code ++ convAttrBlock 0
          code := code ++ s!"      " ++ "}" ++ s!" : ({blockTy}, {tensorTy [4*c, c, 1, 1]}) -> {expTy}\n"
          code := code ++ s!"    %cnx_bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [4*c]}) -> {expTy}\n"
          code := code ++ s!"    %cnx_out{pidx} = stablehlo.add %cnx_cv{pidx}, %cnx_bb{pidx} : {expTy}\n"
          let expandOut := s!"%cnx_out{pidx}"
          pidx := pidx + 1
          -- 4) Activation
          let mut actOut := expandOut
          let mut actTanh := ""
          match act with
          | .gelu =>
            let (geCode, geOut, geT) := emitGeluForward tag expandOut expShape
            code := code ++ geCode
            actOut := geOut; actTanh := geT
          | .relu =>
            code := code ++ s!"    %cnar_z{tag} = stablehlo.constant dense<0.0> : {expTy}\n"
            code := code ++ s!"    %cnar_o{tag} = stablehlo.maximum {expandOut}, %cnar_z{tag} : {expTy}\n"
            actOut := s!"%cnar_o{tag}"
          | _ =>
            -- Fallback: identity (unsupported variants here)
            actOut := expandOut
          -- 5) 1×1 project 4c → c (pidx)
          code := code ++ s!"    %cnp_cv{pidx} = \"stablehlo.convolution\"({actOut}, %W{pidx}) " ++ "{\n"
          code := code ++ convAttrBlock 0
          code := code ++ s!"      " ++ "}" ++ s!" : ({expTy}, {tensorTy [c, 4*c, 1, 1]}) -> {blockTy}\n"
          code := code ++ s!"    %cnp_bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({cTy}) -> {blockTy}\n"
          code := code ++ s!"    %cnp_out{pidx} = stablehlo.add %cnp_cv{pidx}, %cnp_bb{pidx} : {blockTy}\n"
          let projectOut := s!"%cnp_out{pidx}"
          pidx := pidx + 1
          -- 6) LayerScale γ ⊙ projectOut (pidx)
          code := code ++ s!"    %cnls_bc{tag} = stablehlo.broadcast_in_dim %W{pidx}, dims = [1] : ({cTy}) -> {blockTy}\n"
          code := code ++ s!"    %cnls_o{tag} = stablehlo.multiply {projectOut}, %cnls_bc{tag} : {blockTy}\n"
          let lsOut := s!"%cnls_o{tag}"
          pidx := pidx + 1
          -- 7) Residual add
          code := code ++ s!"    %cnres_o{tag} = stablehlo.add {blockIn}, {lsOut} : {blockTy}\n"
          let blockOut := s!"%cnres_o{tag}"
          curSSA := blockOut
          curShape := blockShape
          -- Push fat record for backward
          let mut cnRec : FwdRec := default
          cnRec := { cnRec with layer := l, pidx := none, pos, inputSSA := blockIn, outputSSA := blockOut, inShape := blockShape, outShape := blockShape }
          cnRec := { cnRec with isConvNextBlock := true, cnbBasePidx := basePidx, cnbChannels := c, cnbAct := act, cnbNorm := norm }
          cnRec := { cnRec with cnbBlockInSSA := blockIn, cnbDwOutSSA := dwOut, cnbLnOutSSA := lnOut }
          cnRec := { cnRec with cnbLnNormSSA := lnNorm, cnbLnIstdSSA := lnIstd }
          cnRec := { cnRec with cnbBnNormSSA := lnNorm, cnbBnMeanBcSSA := bnMeanBc, cnbBnIstdBcSSA := bnIstdBc }
          cnRec := { cnRec with cnbExpandOutSSA := expandOut, cnbActOutSSA := actOut, cnbActTanhSSA := actTanh, cnbProjectOutSSA := projectOut }
          records := records.push cnRec
        | _ => pure ()

    | .convNextDownsample ic oc norm =>
      match curShape with
      | [b, _, h, w] =>
        let inSSA0 := curSSA
        let inSh := curShape
        let basePidx := pidx
        let tag := s!"cnds{pos}"
        let outShape := [b, oc, h / 2, w / 2]
        -- 1) Norm (pidx): LN OR BN
        let mut lnOut : String := ""
        let mut lnNorm : String := ""
        let mut lnIstd : String := ""
        let mut bnMeanBc : String := ""
        let mut bnIstdBc : String := ""
        match norm with
        | .ln =>
          let (lc, out, nm, ist, _) := emitLayerNormForwardNCHW tag inSSA0 inSh s!"%W{pidx}" s!"%b{pidx}"
          code := code ++ lc; lnOut := out; lnNorm := nm; lnIstd := ist
        | .bn =>
          let (bc, out, nm, mbc, ibc) := emitBatchNormForwardNCHW pidx inSSA0 inSh s!"%W{pidx}" s!"%b{pidx}"
          code := code ++ bc; lnOut := out; lnNorm := nm; bnMeanBc := mbc; bnIstdBc := ibc
        pidx := pidx + 1
        -- 2) 2×2 stride-2 valid-pad conv (pidx)
        code := code ++ s!"    %cnds_cv{pidx} = \"stablehlo.convolution\"({lnOut}, %W{pidx}) " ++ "{\n"
        code := code ++ "        batch_group_count = 1 : i64,\n"
        code := code ++ convDimNumbers
        code := code ++ "        feature_group_count = 1 : i64,\n"
        code := code ++ "        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,\n"
        code := code ++ "        rhs_dilation = array<i64: 1, 1>,\n"
        code := code ++ "        window_strides = array<i64: 2, 2>\n"
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy inSh}, {tensorTy [oc, ic, 2, 2]}) -> {tensorTy outShape}\n"
        code := code ++ s!"    %cnds_bb{pidx} = stablehlo.broadcast_in_dim %b{pidx}, dims = [1] : ({tensorTy [oc]}) -> {tensorTy outShape}\n"
        code := code ++ s!"    %cnds_o{pidx} = stablehlo.add %cnds_cv{pidx}, %cnds_bb{pidx} : {tensorTy outShape}\n"
        let dsOut := s!"%cnds_o{pidx}"
        pidx := pidx + 1
        curSSA := dsOut
        curShape := outShape
        let mut cnRec : FwdRec := default
        cnRec := { cnRec with layer := l, pidx := none, pos, inputSSA := inSSA0, outputSSA := dsOut, inShape := inSh, outShape := outShape }
        cnRec := { cnRec with isConvNextDs := true, cndBasePidx := basePidx, cndIc := ic, cndOc := oc, cndNorm := norm }
        cnRec := { cnRec with cndInSSA := inSSA0, cndLnOutSSA := lnOut }
        cnRec := { cnRec with cndLnNormSSA := lnNorm, cndLnIstdSSA := lnIstd }
        cnRec := { cnRec with cndBnNormSSA := lnNorm, cndBnMeanBcSSA := bnMeanBc, cndBnIstdBcSSA := bnIstdBc }
        records := records.push cnRec
      | _ => pure ()

    | .patchEmbed ic dim pSize nP =>
      let pWIdx := pidx
      let pBIdx := pidx  -- shared pidx (W + b of conv)
      let pClsIdx := pidx + 1
      let pPosIdx := pidx + 2
      let pWSSA := s!"%W{pWIdx}"
      let pBSSA := s!"%b{pBIdx}"
      let pClsSSA := s!"%W{pClsIdx}"
      let pPosSSA := s!"%W{pPosIdx}"
      let (snip, newSSA, newShape) := emitPatchEmbedForward s!"{pos}" curSSA curShape ic dim pSize nP pWSSA pBSSA pClsSSA pPosSSA
      code := code ++ snip
      let mut peRec : FwdRec := default
      peRec := { peRec with layer := l, pidx := none, pos, inputSSA := inSSA, outputSSA := newSSA, inShape := curShape, outShape := newShape }
      peRec := { peRec with isPatchEmbed := true, pePWPidx := pWIdx, pePBPidx := pBIdx, peClsPidx := pClsIdx, pePosPidx := pPosIdx }
      peRec := { peRec with pePSize := pSize, pePIc := ic, pePDim := dim, pePNp := nP }
      records := records.push peRec
      curSSA := newSSA
      curShape := newShape
      pidx := pidx + 3

    | .transformerEncoder dim heads mlpDim nBlocks causalMask keepSequence flashAttn rope =>
      for bi in [:nBlocks] do
        let blockIn := curSSA
        let blockShape := curShape
        let basePidx := pidx
        let tag := s!"{pos}_{bi}"
        let ty := tensorTy blockShape
        -- LN1
        let g1 := s!"%W{pidx}"; let b1 := s!"%b{pidx}"
        let (ln1Code, ln1Out, ln1Norm, ln1Istd, _ln1Mean) := emitLayerNormForward s!"{tag}_ln1" blockIn blockShape g1 b1
        code := code ++ ln1Code
        pidx := pidx + 1
        -- MHSA
        let wq := s!"%W{pidx}"; let bq := s!"%b{pidx}"; pidx := pidx + 1
        let wk := s!"%W{pidx}"; let bk := s!"%b{pidx}"; pidx := pidx + 1
        let wv := s!"%W{pidx}"; let bv := s!"%b{pidx}"; pidx := pidx + 1
        let wo := s!"%W{pidx}"; let bo := s!"%b{pidx}"; pidx := pidx + 1
        let (mhCode, mhOut, mhQ, mhK, mhV, mhSmLse, mhPp, mhAvO) := emitMHSAForward s!"{tag}_mh" ln1Out blockShape heads wq bq wk bk wv bv wo bo causalMask flashAttn rope
        code := code ++ mhCode
        -- Residual
        code := code ++ s!"    %tb_r1{tag} = stablehlo.add {blockIn}, {mhOut} : {ty}\n"
        let r1 := s!"%tb_r1{tag}"
        -- LN2
        let g2 := s!"%W{pidx}"; let b2 := s!"%b{pidx}"; pidx := pidx + 1
        let (ln2Code, ln2Out, ln2Norm, ln2Istd, _ln2Mean) := emitLayerNormForward s!"{tag}_ln2" r1 blockShape g2 b2
        code := code ++ ln2Code
        -- fc1
        let wf1 := s!"%W{pidx}"; let bf1 := s!"%b{pidx}"; pidx := pidx + 1
        let (fc1Code, fc1Out, fc1Shape) := emitDense3D s!"{tag}_fc1" ln2Out blockShape wf1 bf1 mlpDim
        code := code ++ fc1Code
        -- GELU
        let (geCode, geOut, geT) := emitGeluForward s!"{tag}_ge" fc1Out fc1Shape
        code := code ++ geCode
        -- fc2
        let wf2 := s!"%W{pidx}"; let bf2 := s!"%b{pidx}"; pidx := pidx + 1
        let (fc2Code, fc2Out, _) := emitDense3D s!"{tag}_fc2" geOut fc1Shape wf2 bf2 dim
        code := code ++ fc2Code
        -- Residual
        code := code ++ s!"    %tb_r2{tag} = stablehlo.add {r1}, {fc2Out} : {ty}\n"
        let blockOut := s!"%tb_r2{tag}"
        -- Record block
        let mut tbRec : FwdRec := default
        tbRec := { tbRec with layer := l, pidx := none, pos, inputSSA := blockIn, outputSSA := blockOut, inShape := blockShape, outShape := blockShape }
        tbRec := { tbRec with isTransformerBlock := true, tbBasePidx := basePidx, tbHeads := heads, tbMlpDim := mlpDim, tbDim := dim }
        tbRec := { tbRec with tbLn1XSSA := blockIn, tbLn1OutSSA := ln1Out, tbLn1NormSSA := ln1Norm, tbLn1IstdSSA := ln1Istd }
        tbRec := { tbRec with tbMhsaOutSSA := mhOut, tbR1SSA := r1 }
        tbRec := { tbRec with tbLn2OutSSA := ln2Out, tbLn2NormSSA := ln2Norm, tbLn2IstdSSA := ln2Istd }
        tbRec := { tbRec with tbMhQSSA := mhQ, tbMhKSSA := mhK, tbMhVSSA := mhV, tbMhSmSSA := mhSmLse, tbMhPpSSA := mhPp, tbMhFlash := flashAttn, tbMhOSSA := mhAvO, tbMhRope := rope }
        tbRec := { tbRec with tbFc1OutSSA := fc1Out, tbGeluTSSA := geT, tbGeluOutSSA := geOut }
        records := records.push tbRec
        curSSA := blockOut
      -- Final LN
      let gF := s!"%W{pidx}"; let bF := s!"%b{pidx}"
      let finalLnPidxV := pidx
      let (lnfCode, lnfOut, lnfNorm, lnfIstd, _) := emitLayerNormForward s!"{pos}_fln" curSSA curShape gF bF
      code := code ++ lnfCode
      pidx := pidx + 1
      let mut flnRec : FwdRec := default
      flnRec := { flnRec with layer := l, pidx := none, pos, inputSSA := curSSA, outputSSA := lnfOut, inShape := curShape, outShape := curShape }
      flnRec := { flnRec with isFinalLn := true, finalLnPidx := finalLnPidxV, finalLnNormSSA := lnfNorm, finalLnIstdSSA := lnfIstd, finalLnInSSA := curSSA, finalLnOutSSA := lnfOut }
      records := records.push flnRec
      curSSA := lnfOut
      if !causalMask && !keepSequence then
        -- ViT: slice the CLS token and reduce to [B, dim] for the classifier head.
        match curShape with
        | [b, _, _] =>
          let clsShape := [b, 1, dim]
          let outShape := [b, dim]
          code := code ++ s!"    %te_cls{pos} = \"stablehlo.slice\"({curSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, 1, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({tensorTy curShape}) -> {tensorTy clsShape}\n"
          code := code ++ s!"    %te_out{pos} = stablehlo.reshape %te_cls{pos} : ({tensorTy clsShape}) -> {tensorTy outShape}\n"
          let clsOut := s!"%te_out{pos}"
          let mut csRec : FwdRec := default
          csRec := { csRec with layer := l, pidx := none, pos, inputSSA := curSSA, outputSSA := clsOut, inShape := curShape, outShape := outShape }
          csRec := { csRec with isClsSlice := true, clsInShape := curShape, clsInSSA := curSSA }
          records := records.push csRec
          curSSA := clsOut
          curShape := outShape
        | _ => pure ()

    | .tokenPositionEmbed v t d ids gather posEmb =>
      -- Forward: flat [B, V*T] one-hot → reshape [B, T, V] (or, with
      -- idsInput, [B, T] f32 ids → in-graph one-hot) → matmul W [V, D]
      -- → [B, T, D] → (unless posEmb off) add learnable position [T, D].
      -- With `gather`, the [B, T] ids index W directly (stablehlo.gather); the
      -- backward is then a scatter-add. Either way the backward reads the SSA
      -- recorded in tpeBtvSSA (one-hot [B,T,V] or index [B,T,1]).
      match curShape with
      | [b, _] =>
        let btv := [b, t, v]
        let btd := [b, t, d]
        let inSSA := curSSA
        let pW := s!"%W{pidx}"          -- [V, D] embed (no bias)
        let pPos := s!"%W{pidx + 1}"    -- [T, D] position
        let bwdSSA := if gather then s!"%tpe_idx{pos}" else s!"%tpe_r{pos}"
        if gather then
          code := code ++ emitTokenGatherFwd pos curSSA curShape pW b t d v
        else
          if ids then
            let i1Ty := s!"tensor<{b}x{t}x{v}xi1>"
            code := code ++ s!"    %tpe_idb{pos} = stablehlo.broadcast_in_dim {curSSA}, dims = [0, 1] : ({tensorTy curShape}) -> {tensorTy btv}\n"
            code := code ++ s!"    %tpe_iota{pos} = stablehlo.iota dim = 2 : {tensorTy btv}\n"
            code := code ++ s!"    %tpe_cmp{pos} = stablehlo.compare EQ, %tpe_idb{pos}, %tpe_iota{pos}, FLOAT : ({tensorTy btv}, {tensorTy btv}) -> {i1Ty}\n"
            code := code ++ s!"    %tpe_onef{pos} = stablehlo.constant dense<1.0> : {tensorTy btv}\n"
            code := code ++ s!"    %tpe_zerof{pos} = stablehlo.constant dense<0.0> : {tensorTy btv}\n"
            code := code ++ s!"    %tpe_r{pos} = stablehlo.select %tpe_cmp{pos}, %tpe_onef{pos}, %tpe_zerof{pos} : {i1Ty}, {tensorTy btv}\n"
          else
            code := code ++ s!"    %tpe_r{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy btv}\n"
          code := code ++ s!"    %tpe_emb{pos} = stablehlo.dot_general %tpe_r{pos}, {pW},\n"
          code := code ++ "              contracting_dims = [2] x [0],\n"
          code := code ++ "              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy btv}, {tensorTy [v, d]}) -> {tensorTy btd}\n"
        let outSSA := if posEmb then s!"%tpe_out{pos}" else s!"%tpe_emb{pos}"
        if posEmb then
          code := code ++ s!"    %tpe_pbc{pos} = stablehlo.broadcast_in_dim {pPos}, dims = [1, 2] : ({tensorTy [t, d]}) -> {tensorTy btd}\n"
          code := code ++ s!"    %tpe_out{pos} = stablehlo.add %tpe_emb{pos}, %tpe_pbc{pos} : {tensorTy btd}\n"
        let mut tpeRec : FwdRec := default
        tpeRec := { tpeRec with layer := l, pidx := none, pos, inputSSA := inSSA, outputSSA := outSSA, inShape := curShape, outShape := btd }
        tpeRec := { tpeRec with isTokenPosEmbed := true, tpePidx := pidx, tpeV := v, tpeT := t, tpeD := d, tpeBtvSSA := bwdSSA, tpeGather := gather, tpePosEmb := posEmb }
        records := records.push tpeRec
        curSSA := outSSA
        curShape := btd
        pidx := pidx + (if posEmb then 2 else 1)
      | _ => code := code ++ "    // tokenPositionEmbed train: input not [B, _]\n"

    | .spatialFlatten =>
      match curShape with
      | [b, c, h, w] =>
        let bhwc := [b, h, w, c]
        let bnc := [b, h * w, c]
        let inSSA := curSSA
        code := code ++ s!"    %sf_t{pos} = stablehlo.transpose {curSSA}, dims = [0, 2, 3, 1] : ({tensorTy curShape}) -> {tensorTy bhwc}\n"
        code := code ++ s!"    %sf_r{pos} = stablehlo.reshape %sf_t{pos} : ({tensorTy bhwc}) -> {tensorTy bnc}\n"
        let outSSA := s!"%sf_r{pos}"
        records := records.push { layer := l, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := outSSA, inShape := curShape, outShape := bnc }
        curSSA := outSSA
        curShape := bnc
      | _ => code := code ++ "    // spatialFlatten train: input not [B, C, H, W]\n"
    | .spatialUnflatten c h w =>
      match curShape with
      | [b, _, _] =>
        let bhwc := [b, h, w, c]
        let bchw := [b, c, h, w]
        let inSSA := curSSA
        code := code ++ s!"    %su_r{pos} = stablehlo.reshape {curSSA} : ({tensorTy curShape}) -> {tensorTy bhwc}\n"
        code := code ++ s!"    %su_t{pos} = stablehlo.transpose %su_r{pos}, dims = [0, 3, 1, 2] : ({tensorTy bhwc}) -> {tensorTy bchw}\n"
        let outSSA := s!"%su_t{pos}"
        records := records.push { layer := l, pidx := none, pos, inputSSA := inSSA, preActSSA := "", outputSSA := outSSA, inShape := curShape, outShape := bchw }
        curSSA := outSSA
        curShape := bchw
      | _ => code := code ++ "    // spatialUnflatten train: input not [B, _, _]\n"

    | .lmHead d v t =>
      -- Forward: [B, T, D] → dense3D D→V → [B, T, V] → transpose [B, V, T] →
      -- reshape [B, V, T, 1].  The 4-D output drops into the existing useSeg
      -- per-pixel CE machinery directly (NCHW with H=T, W=1).
      match curShape with
      | [b, _, _] =>
        let inSSA := curSSA
        let inShape := curShape
        let btv := [b, t, v]
        let bvt := [b, v, t]
        let bvt1 := [b, v, t, 1]
        let pW := s!"%W{pidx}"
        let pB := s!"%b{pidx}"
        code := code ++ s!"    %lmh_mm{pos} = stablehlo.dot_general {curSSA}, {pW},\n"
        code := code ++ "              contracting_dims = [2] x [0],\n"
        code := code ++ "              precision = [DEFAULT, DEFAULT]\n"
        code := code ++ s!"            : ({tensorTy curShape}, {tensorTy [d, v]}) -> {tensorTy btv}\n"
        code := code ++ s!"    %lmh_bbc{pos} = stablehlo.broadcast_in_dim {pB}, dims = [2] : ({tensorTy [v]}) -> {tensorTy btv}\n"
        code := code ++ s!"    %lmh_btv{pos} = stablehlo.add %lmh_mm{pos}, %lmh_bbc{pos} : {tensorTy btv}\n"
        code := code ++ s!"    %lmh_t{pos} = stablehlo.transpose %lmh_btv{pos}, dims = [0, 2, 1] : ({tensorTy btv}) -> {tensorTy bvt}\n"
        code := code ++ s!"    %lmh_out{pos} = stablehlo.reshape %lmh_t{pos} : ({tensorTy bvt}) -> {tensorTy bvt1}\n"
        let outSSA := s!"%lmh_out{pos}"
        let mut lmhRec : FwdRec := default
        lmhRec := { lmhRec with layer := l, pidx := none, pos, inputSSA := inSSA, outputSSA := outSSA, inShape := inShape, outShape := bvt1 }
        lmhRec := { lmhRec with isLmHead := true, lmhPidx := pidx, lmhD := d, lmhV := v, lmhT := t, lmhBtvSSA := s!"%lmh_btv{pos}" }
        records := records.push lmhRec
        curSSA := outSSA
        curShape := bvt1
        pidx := pidx + 1
      | _ => code := code ++ "    // lmHead train: input not [B, T, D]\n"

    | .timeCondAdd c nFreq =>
      -- Per-block DDPM time conditioning. Adds a learned sin/cos time
      -- projection onto the feature map; timestep read in-graph from
      -- %x_flat's t-channel. Store the sin/cos emb SSA (preActSSA) for
      -- the backward's d_W. Residual add ⇒ gradient passes through.
      match curShape with
      | [b, _, _, _] =>
        let inSSA := curSSA
        let (snip, outSSA, embSSA) := emitTimeCondForward s!"t{pos}" "%x_flat" b (inputFlatDim spec)
          (ddpmTStepOffset spec) nFreq c curSSA curShape s!"%W{pidx}" s!"%b{pidx}"
        code := code ++ snip
        records := records.push
          { layer := l, pidx := some pidx, pos, inputSSA := inSSA,
            preActSSA := embSSA, outputSSA := outSSA,
            inShape := curShape, outShape := curShape }
        curSSA := outSSA
        pidx := pidx + 1
      | _ => code := code ++ "    // timeCondAdd train: input not [B, C, H, W]\n"

    | .unetDown ic oc =>
      -- 2× convBn (ic→oc, oc→oc) → push pre-pool feature on encoder stack →
      -- maxPool 2 with `addSkipGrad := %unet_skip_g{e}` so the matching
      -- unetUp's backward can route the skip-half gradient back here.
      let (s1, rec1) := emitConvBnTrain pidx pos curSSA curShape ic oc 3 1 true
      code := code ++ s1; curSSA := rec1.outputSSA; curShape := rec1.outShape
      records := records.push rec1; pidx := pidx + 1
      let (s2, rec2) := emitConvBnTrain pidx pos curSSA curShape oc oc 3 1 true
      code := code ++ s2; curSSA := rec2.outputSSA; curShape := rec2.outShape
      let preeMaxSSA := curSSA
      let preeMaxShape := curShape
      records := records.push rec2; pidx := pidx + 1
      -- Assign this encoder its slot index. The skip-grad SSA name is fixed:
      -- `%unet_skip_g{e}`. The matching unetUp will define it on backward.
      let encIdx := nextUnetEncIdx
      unetEncStack := encIdx :: unetEncStack
      nextUnetEncIdx := nextUnetEncIdx + 1
      -- maxPool 2 (size = stride = 2) — same emit as the standalone case.
      match curShape with
      | [b, c, h, w] =>
        let oH := (h + 1) / 2  -- ceil-div for SAME with stride 2
        let oW := (w + 1) / 2
        let outShape := [b, c, oH, oW]
        -- For stride 2 / size 2 we don't need extra padding (the standard
        -- maxPool case fully handles that, but here we know size==stride==2
        -- so no padding is required).
        code := code ++ s!"    %pli{pos} = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
        code := code ++ s!"    %pl{pos} = \"stablehlo.reduce_window\"({preeMaxSSA}, %pli{pos}) (" ++ "{\n"
        code := code ++ s!"      ^bb0(%rwa{pos}: tensor<f32>, %rwb{pos}: tensor<f32>):\n"
        code := code ++ s!"        %rwm{pos} = stablehlo.maximum %rwa{pos}, %rwb{pos} : tensor<f32>\n"
        code := code ++ s!"        \"stablehlo.return\"(%rwm{pos}) : (tensor<f32>) -> ()\n"
        code := code ++ "      }) " ++ "{" ++ s!"window_dimensions = array<i64: 1, 1, 2, 2>, "
        code := code ++ s!"window_strides = array<i64: 1, 1, 2, 2>" ++ "}\n"
        code := code ++ s!"      : ({tensorTy preeMaxShape}, tensor<f32>) -> {tensorTy outShape}\n"
        curSSA := s!"%pl{pos}"
        let mut poolRec : FwdRec := default
        poolRec := { poolRec with layer := .maxPool 2 2, pidx := none, pos := pos }
        poolRec := { poolRec with inputSSA := preeMaxSSA, preActSSA := "0,0,0,0" }
        poolRec := { poolRec with outputSSA := curSSA }
        poolRec := { poolRec with inShape := preeMaxShape, outShape := outShape }
        -- The matching unetUp's UnetUpConcat backward will define this SSA.
        poolRec := { poolRec with addSkipGrad := s!"%unet_skip_g{encIdx}" }
        records := records.push poolRec
        curShape := outShape
      | _ => pure ()

    | .unetUp ic oc =>
      -- bilinear ×2 (no params, recorded as a normal `.bilinearUpsample`) →
      -- UnetUpConcat marker (channel-concat + skip-grad-save point) →
      -- 2× convBn ((ic+oc)→oc, oc→oc).
      let (s1, ssa1, sh1) := emitBilinearUpsample pos curSSA curShape 2
      code := code ++ s1
      records := records.push { layer := .bilinearUpsample 2, pidx := none, pos
                              , inputSSA := inSSA, preActSSA := ""
                              , outputSSA := ssa1, inShape := curShape, outShape := sh1 }
      curSSA := ssa1; curShape := sh1
      -- Pop matching encoder from the LIFO stack.
      match unetEncStack with
      | encIdx :: rest =>
        unetEncStack := rest
        -- Concat with the skip feature pushed by the matching unetDown.
        -- The skip feature SSA is the second convBn output of that unetDown,
        -- which lives at `%cbn_out{pidx-of-that-convBn}`. We don't track
        -- that pidx here (would couple us to the encoder); instead use the
        -- existing emit helper which expects raw SSA + shapes. The skip
        -- feature SSA we want is exactly the input of the matching maxPool —
        -- we look it up via the records array.
        --
        -- Identify the matching unetDown's maxPool record (the one with
        -- `addSkipGrad = %unet_skip_g{encIdx}`) and use its `inputSSA` as
        -- the skip feature SSA. This avoids passing intermediate SSA
        -- through the layer-walk state.
        let skipMatch := records.findSomeRev? (fun r =>
          if r.addSkipGrad == s!"%unet_skip_g{encIdx}" then
            some (r.inputSSA, r.inShape) else none)
        match skipMatch with
        | some (skipSSA, skipShape) =>
          let (s2, ssa2, sh2) := emitChannelConcat s!"{pos}_uut" ssa1 skipSSA sh1 skipShape
          code := code ++ s2
          curSSA := ssa2; curShape := sh2
          -- Push UnetUpConcat marker. Backward will: split sh2 → (sh1, skipShape),
          -- save skipShape gradient as `%unet_skip_g{encIdx}`, set gradSSA to sh1.
          records := records.push {
            layer := .unetUp ic oc, pidx := none, pos
            inputSSA := ssa1, preActSSA := "", outputSSA := ssa2
            inShape := sh1, outShape := sh2
            isUnetUpConcat := true
            unetSkipShape := skipShape
            unetEncoderIdx := encIdx
          }
          let (s3, rec3) := emitConvBnTrain pidx pos curSSA curShape (ic + oc) oc 3 1 true
          code := code ++ s3; curSSA := rec3.outputSSA; curShape := rec3.outShape
          records := records.push rec3; pidx := pidx + 1
          let (s4, rec4) := emitConvBnTrain pidx pos curSSA curShape oc oc 3 1 true
          code := code ++ s4; curSSA := rec4.outputSSA; curShape := rec4.outShape
          records := records.push rec4; pidx := pidx + 1
        | none =>
          code := code ++ "    // unetUp: matching unetDown maxPool record not found\n"
      | [] =>
        code := code ++ "    // unetUp: encoder stack empty (no matching unetDown)\n"

    | _ => code := code ++ "    // UNSUPPORTED\n"
    pos := pos + 1

  let logitsSSA := curSSA
  let NC := nClasses

  -- Loss + backward seed: per-pixel CE for segmentation, classification CE
  -- otherwise. Both branches set %loss + the gradient seed; the seg branch
  -- also tells the dispatcher to use a 4-D gradShape (vs [B, NC]).
  let mut gradSSA : String := "%d_logits"
  let mut gradShape : List Nat := [B, NC]
  if useYolov1 then
    -- ═══════════════ YOLOv1: 5-term masked MSE ═══════════════
    -- See planning/yolo_demo_v2.md Phase 1 decisions D1-D11. Predictions
    -- arrive as flat [B, totalCh]; we reshape to [B, perCell, gH, gW]
    -- (NCHW), slice per-term, compute masked MSE for each, then concat
    -- gradient slabs back to [B, perCell, gH, gW] and reshape flat for
    -- the dense backward to consume.
    --
    -- Channel layout (perCell = numBoxes*5 + numClasses):
    --   [0..2)              box 0 (x, y)
    --   [2..4)              box 0 (w, h)     -- √ applied with ε floor
    --   [4..5)              box 0 confidence
    --   [5..9)              box 1 (x, y, w, h)  — never optimized (Option A)
    --   [9..10)             box 1 confidence — always penalized as no-object
    --   [10..perCell)       per-cell class one-hot (numClasses long)
    --
    -- Forward terms (each summed over its non-zero cells, then summed
    -- together and divided by B for per-image-mean gradient scaling):
    --   T1: λ_coord · Σ mask_obj · (pred_xy - tgt_xy)²
    --   T2: λ_coord · Σ mask_obj · (√pred_wh_ε - √tgt_wh_ε)²
    --   T3: Σ mask_obj · (pred_c0 - 1)²
    --   T4: λ_noobj · Σ (1 - mask_obj) · (pred_c0)²
    --   T5: λ_noobj · Σ (pred_c1)²
    --   T6: Σ mask_obj · Σ_c (pred_cls_c - tgt_cls_c)²
    --
    -- Backward: hand-derived per-term, concatenated along channel dim.
    -- The √ derivative `1 / (2·√pred)` is paired with the 2 from d(x²)/dx,
    -- so the prefactor is just `1 / √pred_clamp` (no 2). Where the input
    -- was clamped (pred < ε), gradient is zeroed via stablehlo.select.
    match curShape with
    | [_, totalCh] =>
      if yoloAnchors.isEmpty then
        let perCell := yoloNumBoxes * 5 + yoloNumClasses
        let gH := yoloGridH
        let gW := yoloGridW
        let numC := yoloNumClasses
        let expected := gH * gW * perCell
        if totalCh != expected then
          code := code ++ s!"    // useYolov1=true but flat dim {totalCh} ≠ gH*gW*perCell = {expected}\n"
        else
          let shape4 := [B, perCell, gH, gW]
          let shape4Ty := tensorTy shape4
          let maskTy := tensorTy [B, gH, gW]
          let shapeXY := [B, 2, gH, gW]
          let shapeXYTy := tensorTy shapeXY
          let shapeWH := shapeXY  -- same shape, different channels
          let shapeC1 := [B, 1, gH, gW]
          let shapeC1Ty := tensorTy shapeC1
          let shapeClass := [B, numC, gH, gW]
          let shapeClassTy := tensorTy shapeClass
          let shapeClsRed := [B, gH, gW]            -- class dim reduced (for softmax CE)
          let shapeClsRedTy := tensorTy shapeClsRed
          let shapeBox1XYWH := [B, 4, gH, gW]
          let shapeBox1XYWHTy := tensorTy shapeBox1XYWH
          let i1WHTy := s!"tensor<{B}x2x{gH}x{gW}xi1>"
          let lambdaCoord := "5.0"
          let lambdaNoobj := "0.5"
          let eps := "1.0e-06"
          let curTy := tensorTy curShape
          code := code ++ "\n    // ================ YOLOv1: 5-term masked MSE ================\n"
          -- Reshape predictions [B, totalCh] → [B, perCell, gH, gW]
          code := code ++ s!"    %y1_pred = stablehlo.reshape {logitsSSA} : ({curTy}) -> {shape4Ty}\n"
          -- Slice predictions + targets
          code := code ++ s!"    %y1_pred_xy = \"stablehlo.slice\"(%y1_pred) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {B}, 2, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {shapeXYTy}\n"
          code := code ++ s!"    %y1_tgt_xy = \"stablehlo.slice\"(%y_yolo) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {B}, 2, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {shapeXYTy}\n"
          code := code ++ s!"    %y1_pred_wh = \"stablehlo.slice\"(%y1_pred) " ++ "{" ++ s!" start_indices = array<i64: 0, 2, 0, 0>, limit_indices = array<i64: {B}, 4, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_tgt_wh = \"stablehlo.slice\"(%y_yolo) " ++ "{" ++ s!" start_indices = array<i64: 0, 2, 0, 0>, limit_indices = array<i64: {B}, 4, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_pred_c0 = \"stablehlo.slice\"(%y1_pred) " ++ "{" ++ s!" start_indices = array<i64: 0, 4, 0, 0>, limit_indices = array<i64: {B}, 5, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {shapeC1Ty}\n"
          code := code ++ s!"    %y1_pred_c1 = \"stablehlo.slice\"(%y1_pred) " ++ "{" ++ s!" start_indices = array<i64: 0, 9, 0, 0>, limit_indices = array<i64: {B}, 10, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {shapeC1Ty}\n"
          code := code ++ s!"    %y1_pred_cls = \"stablehlo.slice\"(%y1_pred) " ++ "{" ++ s!" start_indices = array<i64: 0, 10, 0, 0>, limit_indices = array<i64: {B}, {perCell}, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {shapeClassTy}\n"
          code := code ++ s!"    %y1_tgt_cls = \"stablehlo.slice\"(%y_yolo) " ++ "{" ++ s!" start_indices = array<i64: 0, 10, 0, 0>, limit_indices = array<i64: {B}, {perCell}, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {shapeClassTy}\n"
          -- Broadcast mask to per-term shapes
          code := code ++ s!"    %y1_mask_xy = stablehlo.broadcast_in_dim %m_yolo, dims = [0, 2, 3] : ({maskTy}) -> {shapeXYTy}\n"
          code := code ++ s!"    %y1_mask_c1 = stablehlo.broadcast_in_dim %m_yolo, dims = [0, 2, 3] : ({maskTy}) -> {shapeC1Ty}\n"
          code := code ++ s!"    %y1_mask_cls = stablehlo.broadcast_in_dim %m_yolo, dims = [0, 2, 3] : ({maskTy}) -> {shapeClassTy}\n"
          -- Constants
          code := code ++ s!"    %y1_oneC1 = stablehlo.constant dense<1.0> : {shapeC1Ty}\n"
          code := code ++ s!"    %y1_lcoord = stablehlo.constant dense<{lambdaCoord}> : tensor<f32>\n"
          code := code ++ s!"    %y1_lnoobj = stablehlo.constant dense<{lambdaNoobj}> : tensor<f32>\n"
          code := code ++ s!"    %y1_eps_wh = stablehlo.constant dense<{eps}> : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_Bf = stablehlo.constant dense<{B}.0> : tensor<f32>\n"
          -- T1: coord (x,y) box 0
          code := code ++ s!"    %y1_diff_xy = stablehlo.subtract %y1_pred_xy, %y1_tgt_xy : {shapeXYTy}\n"
          code := code ++ s!"    %y1_sq_xy = stablehlo.multiply %y1_diff_xy, %y1_diff_xy : {shapeXYTy}\n"
          code := code ++ s!"    %y1_msq_xy = stablehlo.multiply %y1_sq_xy, %y1_mask_xy : {shapeXYTy}\n"
          code := code ++ s!"    %y1_sum_xy = stablehlo.reduce(%y1_msq_xy init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
          code := code ++ s!"           : ({shapeXYTy}, tensor<f32>) -> tensor<f32>\n"
          code := code ++ s!"    %y1_t1 = stablehlo.multiply %y1_sum_xy, %y1_lcoord : tensor<f32>\n"
          -- T2: sqrt-coord (w,h) box 0 with ε floor
          code := code ++ s!"    %y1_pred_wh_clamp = stablehlo.maximum %y1_pred_wh, %y1_eps_wh : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_sqrt_pred_wh = stablehlo.sqrt %y1_pred_wh_clamp : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_tgt_wh_clamp = stablehlo.maximum %y1_tgt_wh, %y1_eps_wh : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_sqrt_tgt_wh = stablehlo.sqrt %y1_tgt_wh_clamp : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_diff_wh = stablehlo.subtract %y1_sqrt_pred_wh, %y1_sqrt_tgt_wh : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_sq_wh = stablehlo.multiply %y1_diff_wh, %y1_diff_wh : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_msq_wh = stablehlo.multiply %y1_sq_wh, %y1_mask_xy : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_sum_wh = stablehlo.reduce(%y1_msq_wh init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
          code := code ++ s!"           : ({tensorTy shapeWH}, tensor<f32>) -> tensor<f32>\n"
          code := code ++ s!"    %y1_t2 = stablehlo.multiply %y1_sum_wh, %y1_lcoord : tensor<f32>\n"
          -- ── T3 / T4 / T5: objectness confidence ──────────────────────────────
          -- Two paths.
          --   Non-focal (default): raw-MSE on raw conf — YOLOv1 as published.
          --   Focal (useFocal):    sigmoid + focal-BCE on the conf *logit*, with a
          --     DETACHED focal weight (1-p_t)^γ. This is the fix for the fg/bg
          --     objectness collapse (planning/yolo_v5.md §3): ~1-2 object cells vs
          --     ~47 background cells make "predict 0 everywhere" an MSE minimum, so
          --     the conv head localizes early then decays to a center-prior. Focal
          --     down-weights easy (well-classified) cells so the rare foreground keeps
          --     a gradient. α-balance = {1 on object cells, λ_noobj on background};
          --     box 1 is always background (target 0). The focal weight is computed
          --     once here and reused verbatim in the backward, so it is constant w.r.t.
          --     the gradient ("detached"):  d/dz = α · (1-p_t)^γ · (sigmoid(z) - t).
          if useFocal then
            code := code ++ s!"    // T3+T4 (box0) / T5 (box1): sigmoid focal-BCE objectness (γ={focalGamma})\n"
            code := code ++ s!"    %y1f_zero = stablehlo.constant dense<0.0> : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_eps = stablehlo.constant dense<1.0e-12> : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_gamma = stablehlo.constant dense<{focalGamma}> : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_lnoobj = stablehlo.constant dense<{lambdaNoobj}> : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_1mt = stablehlo.subtract %y1_oneC1, %y1_mask_c1 : {shapeC1Ty}\n"
            -- box 0: p0 = sigmoid(z0), target t0 = mask
            code := code ++ s!"    %y1f_p0 = stablehlo.logistic %y1_pred_c0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_1mp0 = stablehlo.subtract %y1_oneC1, %y1f_p0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_tp0 = stablehlo.multiply %y1_mask_c1, %y1f_p0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_1mt1mp0 = stablehlo.multiply %y1f_1mt, %y1f_1mp0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_pt0 = stablehlo.add %y1f_tp0, %y1f_1mt1mp0 : {shapeC1Ty}\n"
            -- detached focal weight w0 = (1 - p_t0)^γ = exp(γ·log(max(1-p_t0, ε)))
            code := code ++ s!"    %y1f_1mpt0 = stablehlo.subtract %y1_oneC1, %y1f_pt0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_1mpt0c = stablehlo.maximum %y1f_1mpt0, %y1f_eps : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_log0 = stablehlo.log %y1f_1mpt0c : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_glog0 = stablehlo.multiply %y1f_gamma, %y1f_log0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_w0 = stablehlo.exponential %y1f_glog0 : {shapeC1Ty}\n"
            -- α0 = mask·1 + (1-mask)·λ_noobj
            code := code ++ s!"    %y1f_a0bg = stablehlo.multiply %y1f_1mt, %y1f_lnoobj : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_a0 = stablehlo.add %y1_mask_c1, %y1f_a0bg : {shapeC1Ty}\n"
            -- stable BCE-with-logits: bce0 = max(z,0) - z·t + log(1 + exp(-|z|))
            code := code ++ s!"    %y1f_relu0 = stablehlo.maximum %y1_pred_c0, %y1f_zero : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_zt0 = stablehlo.multiply %y1_pred_c0, %y1_mask_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_absz0 = stablehlo.abs %y1_pred_c0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_nabsz0 = stablehlo.negate %y1f_absz0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_en0 = stablehlo.exponential %y1f_nabsz0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_1pen0 = stablehlo.add %y1_oneC1, %y1f_en0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_sp0 = stablehlo.log %y1f_1pen0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_bce0a = stablehlo.subtract %y1f_relu0, %y1f_zt0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_bce0 = stablehlo.add %y1f_bce0a, %y1f_sp0 : {shapeC1Ty}\n"
            -- T3 = Σ α0 · w0 · bce0 (covers both object and background cells); T4 folded in → 0
            code := code ++ s!"    %y1f_wb0 = stablehlo.multiply %y1f_a0, %y1f_w0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_t3cell = stablehlo.multiply %y1f_wb0, %y1f_bce0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_t3 = stablehlo.reduce(%y1f_t3cell init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
            code := code ++ s!"           : ({shapeC1Ty}, tensor<f32>) -> tensor<f32>\n"
            code := code ++ s!"    %y1_t4 = stablehlo.constant dense<0.0> : tensor<f32>\n"
            -- box 1: target t1 = 0 always → p_t1 = 1-p1, w1 = (1-p_t1)^γ = p1^γ
            code := code ++ s!"    %y1f_p1 = stablehlo.logistic %y1_pred_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_p1c = stablehlo.maximum %y1f_p1, %y1f_eps : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_log1 = stablehlo.log %y1f_p1c : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_glog1 = stablehlo.multiply %y1f_gamma, %y1f_log1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_w1 = stablehlo.exponential %y1f_glog1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_relu1 = stablehlo.maximum %y1_pred_c1, %y1f_zero : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_absz1 = stablehlo.abs %y1_pred_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_nabsz1 = stablehlo.negate %y1f_absz1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_en1 = stablehlo.exponential %y1f_nabsz1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_1pen1 = stablehlo.add %y1_oneC1, %y1f_en1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_sp1 = stablehlo.log %y1f_1pen1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_bce1 = stablehlo.add %y1f_relu1, %y1f_sp1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_wb1 = stablehlo.multiply %y1f_lnoobj, %y1f_w1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_t5cell = stablehlo.multiply %y1f_wb1, %y1f_bce1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_t5 = stablehlo.reduce(%y1f_t5cell init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
            code := code ++ s!"           : ({shapeC1Ty}, tensor<f32>) -> tensor<f32>\n"
          else
            -- T3: conf positive box 0
            code := code ++ s!"    %y1_diff_c0pos = stablehlo.subtract %y1_pred_c0, %y1_oneC1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_sq_c0pos = stablehlo.multiply %y1_diff_c0pos, %y1_diff_c0pos : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_msq_c0pos = stablehlo.multiply %y1_sq_c0pos, %y1_mask_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_t3 = stablehlo.reduce(%y1_msq_c0pos init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
            code := code ++ s!"           : ({shapeC1Ty}, tensor<f32>) -> tensor<f32>\n"
            -- T4: conf negative box 0
            code := code ++ s!"    %y1_inv_mask_c1 = stablehlo.subtract %y1_oneC1, %y1_mask_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_sq_c0neg = stablehlo.multiply %y1_pred_c0, %y1_pred_c0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_msq_c0neg = stablehlo.multiply %y1_sq_c0neg, %y1_inv_mask_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_sum_c0neg = stablehlo.reduce(%y1_msq_c0neg init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
            code := code ++ s!"           : ({shapeC1Ty}, tensor<f32>) -> tensor<f32>\n"
            code := code ++ s!"    %y1_t4 = stablehlo.multiply %y1_sum_c0neg, %y1_lnoobj : tensor<f32>\n"
            -- T5: conf negative box 1 (always)
            code := code ++ s!"    %y1_sq_c1neg = stablehlo.multiply %y1_pred_c1, %y1_pred_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_sum_c1neg = stablehlo.reduce(%y1_sq_c1neg init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
            code := code ++ s!"           : ({shapeC1Ty}, tensor<f32>) -> tensor<f32>\n"
            code := code ++ s!"    %y1_t5 = stablehlo.multiply %y1_sum_c1neg, %y1_lnoobj : tensor<f32>\n"
          -- T6: class — softmax cross-entropy over the numC class channels (dim 1),
          -- masked to object cells. (Replaces the original SSE class term, which was
          -- too weak to sharpen the 20-way class logits — see planning/yolo notes.)
          -- Numerically-stable: shift by per-cell max, then log-softmax.
          code := code ++ s!"    %y1_cls_ninf = stablehlo.constant dense<-3.0e38> : tensor<f32>\n"
          code := code ++ s!"    %y1_cls_max = stablehlo.reduce(%y1_pred_cls init: %y1_cls_ninf) applies stablehlo.maximum across dimensions = [1]\n"
          code := code ++ s!"           : ({shapeClassTy}, tensor<f32>) -> {shapeClsRedTy}\n"
          code := code ++ s!"    %y1_cls_maxb = stablehlo.broadcast_in_dim %y1_cls_max, dims = [0, 2, 3] : ({shapeClsRedTy}) -> {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_shift = stablehlo.subtract %y1_pred_cls, %y1_cls_maxb : {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_exp = stablehlo.exponential %y1_cls_shift : {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_sumexp = stablehlo.reduce(%y1_cls_exp init: %zf) applies stablehlo.add across dimensions = [1]\n"
          code := code ++ s!"           : ({shapeClassTy}, tensor<f32>) -> {shapeClsRedTy}\n"
          code := code ++ s!"    %y1_cls_logsum = stablehlo.log %y1_cls_sumexp : {shapeClsRedTy}\n"
          code := code ++ s!"    %y1_cls_logsumb = stablehlo.broadcast_in_dim %y1_cls_logsum, dims = [0, 2, 3] : ({shapeClsRedTy}) -> {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_logsm = stablehlo.subtract %y1_cls_shift, %y1_cls_logsumb : {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_nll = stablehlo.multiply %y1_tgt_cls, %y1_cls_logsm : {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_nll_m = stablehlo.multiply %y1_cls_nll, %y1_mask_cls : {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_ce_sum = stablehlo.reduce(%y1_cls_nll_m init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
          code := code ++ s!"           : ({shapeClassTy}, tensor<f32>) -> tensor<f32>\n"
          code := code ++ s!"    %y1_t6 = stablehlo.negate %y1_cls_ce_sum : tensor<f32>\n"
          -- DIoU box loss (brick #1, planning/yolo_drone.md WS-D): when enabled,
          -- replaces the √-MSE coord terms T1+T2 with an IoU-family loss on box0,
          -- using a positive box parameterization (cx=(j+σ(tx))/gW, w=exp(tw)).
          -- Emits %dio_loss (Σ mask·(1-DIoU)) and %dio_dpred [B,4,gH,gW]; both
          -- FD-verified in scripts/diou_probe_check.py. Scaled by λ_coord to keep
          -- the box-vs-objectness balance. NB: the decoder must apply the same σ/exp.
          let lambdaBox : Float := 5.0
          if useDiouBox then
            code := code ++ s!"    %y1_box_pred = \"stablehlo.slice\"(%y1_pred) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {B}, 4, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {tensorTy [B, 4, gH, gW]}\n"
            code := code ++ s!"    %y1_box_tgt = \"stablehlo.slice\"(%y_yolo) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {B}, 4, {gH}, {gW}>, strides = array<i64: 1, 1, 1, 1>" ++ "} : " ++ s!"({shape4Ty}) -> {tensorTy [B, 4, gH, gW]}\n"
            code := code ++ emitDiouForward B gH gW "%y1_box_pred" "%y1_box_tgt" "%m_yolo" "%dio_loss"
            code := code ++ emitDiouBackward B gH gW "%dio_dpred"
            code := code ++ s!"    %dio_gscale = stablehlo.constant dense<{lambdaBox / B.toFloat}> : {tensorTy [B, 4, gH, gW]}\n"
            code := code ++ s!"    %dio_box_grad = stablehlo.multiply %dio_dpred, %dio_gscale : {tensorTy [B, 4, gH, gW]}\n"
          -- Aggregate (box term = DIoU when enabled, else √-MSE T1+T2)
          if useDiouBox then
            code := code ++ s!"    %y1_s12 = stablehlo.multiply %dio_loss, %y1_lcoord : tensor<f32>\n"
          else
            code := code ++ s!"    %y1_s12 = stablehlo.add %y1_t1, %y1_t2 : tensor<f32>\n"
          code := code ++ s!"    %y1_s34 = stablehlo.add %y1_t3, %y1_t4 : tensor<f32>\n"
          code := code ++ s!"    %y1_s56 = stablehlo.add %y1_t5, %y1_t6 : tensor<f32>\n"
          code := code ++ s!"    %y1_s1234 = stablehlo.add %y1_s12, %y1_s34 : tensor<f32>\n"
          code := code ++ s!"    %y1_total = stablehlo.add %y1_s1234, %y1_s56 : tensor<f32>\n"
          code := code ++ s!"    %loss = stablehlo.divide %y1_total, %y1_Bf : tensor<f32>\n"
          -- ─── BACKWARD ───
          code := code ++ s!"    // ─── YOLOv1 backward (5+1 term, planning/yolo_demo_v2.md D4) ───\n"
          code := code ++ s!"    %y1_lcoord_xy = stablehlo.constant dense<{lambdaCoord}> : {shapeXYTy}\n"
          code := code ++ s!"    %y1_lnoobj_c1 = stablehlo.constant dense<{lambdaNoobj}> : {shapeC1Ty}\n"
          code := code ++ s!"    %y1_two_xy = stablehlo.constant dense<2.0> : {shapeXYTy}\n"
          code := code ++ s!"    %y1_two_c1 = stablehlo.constant dense<2.0> : {shapeC1Ty}\n"
          code := code ++ s!"    %y1_two_cls = stablehlo.constant dense<2.0> : {shapeClassTy}\n"
          code := code ++ s!"    %y1_Bf_xy = stablehlo.broadcast_in_dim %y1_Bf, dims = [] : (tensor<f32>) -> {shapeXYTy}\n"
          code := code ++ s!"    %y1_Bf_c1 = stablehlo.broadcast_in_dim %y1_Bf, dims = [] : (tensor<f32>) -> {shapeC1Ty}\n"
          code := code ++ s!"    %y1_Bf_cls = stablehlo.broadcast_in_dim %y1_Bf, dims = [] : (tensor<f32>) -> {shapeClassTy}\n"
          -- d/dpred_xy = 2 · λ_coord · mask · (pred - tgt) / B
          code := code ++ s!"    %y1_g_xy_a = stablehlo.multiply %y1_two_xy, %y1_lcoord_xy : {shapeXYTy}\n"
          code := code ++ s!"    %y1_g_xy_b = stablehlo.multiply %y1_g_xy_a, %y1_mask_xy : {shapeXYTy}\n"
          code := code ++ s!"    %y1_g_xy_c = stablehlo.multiply %y1_g_xy_b, %y1_diff_xy : {shapeXYTy}\n"
          code := code ++ s!"    %y1_g_xy = stablehlo.divide %y1_g_xy_c, %y1_Bf_xy : {shapeXYTy}\n"
          -- d/dpred_wh = mask · λ_coord · (√pred_clamp - √tgt) / √pred_clamp / B,
          -- zeroed where pred was clamped (pred < ε).
          code := code ++ s!"    %y1_g_wh_a = stablehlo.divide %y1_diff_wh, %y1_sqrt_pred_wh : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_g_wh_b = stablehlo.multiply %y1_g_wh_a, %y1_lcoord_xy : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_g_wh_c = stablehlo.multiply %y1_g_wh_b, %y1_mask_xy : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_active_wh = stablehlo.compare GT, %y1_pred_wh, %y1_eps_wh : ({tensorTy shapeWH}, {tensorTy shapeWH}) -> {i1WHTy}\n"
          code := code ++ s!"    %y1_zero_wh = stablehlo.constant dense<0.0> : {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_g_wh_m = \"stablehlo.select\"(%y1_active_wh, %y1_g_wh_c, %y1_zero_wh) : ({i1WHTy}, {tensorTy shapeWH}, {tensorTy shapeWH}) -> {tensorTy shapeWH}\n"
          code := code ++ s!"    %y1_g_wh = stablehlo.divide %y1_g_wh_m, %y1_Bf_xy : {tensorTy shapeWH}\n"
          if useFocal then
            -- focal-BCE objectness backward (detached weight): d/dz = α·w·(sigmoid(z) - t) / B.
            -- p0, w0, a0 (box0) and p1, w1 (box1) come from the focal forward above; the
            -- weights w0/w1 are reused as constants → "detached" focal weight.
            code := code ++ s!"    %y1f_pmt0 = stablehlo.subtract %y1f_p0, %y1_mask_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_g0a = stablehlo.multiply %y1f_a0, %y1f_w0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_g0b = stablehlo.multiply %y1f_g0a, %y1f_pmt0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c0 = stablehlo.divide %y1f_g0b, %y1_Bf_c1 : {shapeC1Ty}\n"
            -- box1: t1 = 0 → (p1 - t1) = p1
            code := code ++ s!"    %y1f_g1a = stablehlo.multiply %y1f_lnoobj, %y1f_w1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1f_g1b = stablehlo.multiply %y1f_g1a, %y1f_p1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c1 = stablehlo.divide %y1f_g1b, %y1_Bf_c1 : {shapeC1Ty}\n"
          else
            -- d/dpred_c0 = (2·mask·(pred-1) + 2·λ_noobj·(1-mask)·pred) / B
            code := code ++ s!"    %y1_g_c0pos_a = stablehlo.multiply %y1_two_c1, %y1_mask_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c0pos_b = stablehlo.multiply %y1_g_c0pos_a, %y1_diff_c0pos : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c0neg_a = stablehlo.multiply %y1_two_c1, %y1_lnoobj_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c0neg_b = stablehlo.multiply %y1_g_c0neg_a, %y1_inv_mask_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c0neg_c = stablehlo.multiply %y1_g_c0neg_b, %y1_pred_c0 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c0_sum = stablehlo.add %y1_g_c0pos_b, %y1_g_c0neg_c : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c0 = stablehlo.divide %y1_g_c0_sum, %y1_Bf_c1 : {shapeC1Ty}\n"
            -- d/dpred_c1 = 2 · λ_noobj · pred / B
            code := code ++ s!"    %y1_g_c1_a = stablehlo.multiply %y1_two_c1, %y1_lnoobj_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c1_b = stablehlo.multiply %y1_g_c1_a, %y1_pred_c1 : {shapeC1Ty}\n"
            code := code ++ s!"    %y1_g_c1 = stablehlo.divide %y1_g_c1_b, %y1_Bf_c1 : {shapeC1Ty}\n"
          -- d/dpred_cls = mask · (softmax(pred) - tgt) / B   [softmax-CE gradient,
          -- reusing exp/sumexp from the forward; no factor of 2 unlike the SSE term]
          code := code ++ s!"    %y1_cls_sumexpb = stablehlo.broadcast_in_dim %y1_cls_sumexp, dims = [0, 2, 3] : ({shapeClsRedTy}) -> {shapeClassTy}\n"
          code := code ++ s!"    %y1_cls_sm = stablehlo.divide %y1_cls_exp, %y1_cls_sumexpb : {shapeClassTy}\n"
          code := code ++ s!"    %y1_g_cls_d = stablehlo.subtract %y1_cls_sm, %y1_tgt_cls : {shapeClassTy}\n"
          code := code ++ s!"    %y1_g_cls_m = stablehlo.multiply %y1_g_cls_d, %y1_mask_cls : {shapeClassTy}\n"
          code := code ++ s!"    %y1_g_cls = stablehlo.divide %y1_g_cls_m, %y1_Bf_cls : {shapeClassTy}\n"
          -- Box 1 xywh: zero (never optimized under Option A)
          code := code ++ s!"    %y1_g_box1 = stablehlo.constant dense<0.0> : {shapeBox1XYWHTy}\n"
          -- Concat slabs along channel dim 1: xy(2) + wh(2) + c0(1) + box1(4) + c1(1) + cls(numC)
          -- Box0-xywh gradient slab: DIoU (%dio_box_grad) or √-MSE (concat g_xy,g_wh).
          let boxGradSSA := if useDiouBox then "%dio_box_grad" else "%y1_cc1"
          if !useDiouBox then
            code := code ++ s!"    %y1_cc1 = stablehlo.concatenate %y1_g_xy, %y1_g_wh, dim = 1 : ({shapeXYTy}, {tensorTy shapeWH}) -> {tensorTy [B, 4, gH, gW]}\n"
          code := code ++ s!"    %y1_cc2 = stablehlo.concatenate {boxGradSSA}, %y1_g_c0, dim = 1 : ({tensorTy [B, 4, gH, gW]}, {shapeC1Ty}) -> {tensorTy [B, 5, gH, gW]}\n"
          code := code ++ s!"    %y1_cc3 = stablehlo.concatenate %y1_cc2, %y1_g_box1, dim = 1 : ({tensorTy [B, 5, gH, gW]}, {shapeBox1XYWHTy}) -> {tensorTy [B, 9, gH, gW]}\n"
          code := code ++ s!"    %y1_cc4 = stablehlo.concatenate %y1_cc3, %y1_g_c1, dim = 1 : ({tensorTy [B, 9, gH, gW]}, {shapeC1Ty}) -> {tensorTy [B, 10, gH, gW]}\n"
          code := code ++ s!"    %y1_grad_4d = stablehlo.concatenate %y1_cc4, %y1_g_cls, dim = 1 : ({tensorTy [B, 10, gH, gW]}, {shapeClassTy}) -> {shape4Ty}\n"
          -- Reshape back to [B, totalCh] for the dense backward to consume
          code := code ++ s!"    %y1_grad_flat = stablehlo.reshape %y1_grad_4d : ({shape4Ty}) -> {curTy}\n"
          gradSSA := "%y1_grad_flat"
          gradShape := curShape
      else
        let gH := yoloGridH
        let gW := yoloGridW
        let anA := yoloAnchors.length
        let perCellA := anA * 15
        let curTyA := tensorTy curShape
        let apTA := tensorTy [B, perCellA, gH, gW]
        if totalCh != gH * gW * perCellA then
          code := code ++ s!"    // anchor yolo: flat {totalCh} != {gH * gW * perCellA}\n"
        else
          code := code ++ s!"    %ay_pred = stablehlo.reshape {logitsSSA} : ({curTyA}) -> {apTA}\n"
          code := code ++ emitAnchorYoloLoss B gH gW yoloAnchors "%ay_pred" "%y_yolo" focalGamma 5.0 "%loss" "%ay_grad4"
          code := code ++ s!"    %y1_grad_flat = stablehlo.reshape %ay_grad4 : ({apTA}) -> {curTyA}\n"
          gradSSA := "%y1_grad_flat"
          gradShape := curShape
    | _ =>
      code := code ++ s!"    // useYolov1=true but curShape is not [B, N] flat: {curShape}\n"
  else if useDdpm then
    -- ═══════════════ DDPM: per-pixel MSE between noise prediction and target ε ═══════════════
    -- Forward:   loss = mean_{B,C,H,W} (logits - y_ddpm)^2
    -- Backward:  d_logits = 2 (logits - y_ddpm) / N    where N = B·C·H·W
    match curShape with
    | [b4, c4, h4, w4] =>
      let outTy := tensorTy curShape
      let nElems := b4 * c4 * h4 * w4
      code := code ++ "\n    // ================ DDPM per-pixel MSE ================\n"
      code := code ++ s!"    %ddpm_diff = stablehlo.subtract {logitsSSA}, %y_ddpm : {outTy}\n"
      code := code ++ s!"    %ddpm_sq = stablehlo.multiply %ddpm_diff, %ddpm_diff : {outTy}\n"
      code := code ++ s!"    %ddpm_sum = stablehlo.reduce(%ddpm_sq init: %zf) applies stablehlo.add across dimensions = [0, 1, 2, 3]\n"
      code := code ++ s!"           : ({outTy}, tensor<f32>) -> tensor<f32>\n"
      code := code ++ s!"    %ddpm_N = stablehlo.constant dense<{nElems}.0> : tensor<f32>\n"
      code := code ++ s!"    %loss = stablehlo.divide %ddpm_sum, %ddpm_N : tensor<f32>\n"
      code := code ++ s!"    // ─── MSE backward: d_logits = 2 (logits - y) / N ───\n"
      code := code ++ s!"    %ddpm_two = stablehlo.constant dense<2.0> : {outTy}\n"
      code := code ++ s!"    %ddpm_2diff = stablehlo.multiply %ddpm_diff, %ddpm_two : {outTy}\n"
      code := code ++ s!"    %ddpm_Nb = stablehlo.broadcast_in_dim %ddpm_N, dims = [] : (tensor<f32>) -> {outTy}\n"
      code := code ++ s!"    %d_logits_ddpm = stablehlo.divide %ddpm_2diff, %ddpm_Nb : {outTy}\n"
      gradSSA := "%d_logits_ddpm"
      gradShape := curShape
    | _ =>
      code := code ++ s!"    // useDdpm=true but curShape is not 4D: {curShape}\n"
  else if useSeg then
    match curShape with
    | [_, segNC, segH, segW] =>
      code := code ++ emitSegLossBlock B segNC segH segW logitsSSA "%y_seg" labelSmoothing segLoss
      gradSSA := "%d_logits_seg"
      gradShape := [B, segNC, segH, segW]
    | _ =>
      code := code ++ s!"    // useSeg=true but curShape is not 4D: {curShape}\n"
  else
    -- ═══════════════ SOFTMAX CE LOSS ═══════════════
    code := code ++ "\n    // ================ SOFTMAX CROSS-ENTROPY ================\n"
    code := code ++ s!"    %maxv = stablehlo.reduce({logitsSSA} init: %neginf) applies stablehlo.maximum across dimensions = [1]\n"
    code := code ++ s!"          : ({tensorTy [B, NC]}, tensor<f32>) -> {tensorTy [B]}\n"
    code := code ++ s!"    %maxv_b = stablehlo.broadcast_in_dim %maxv, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, NC]}\n"
    code := code ++ s!"    %shifted = stablehlo.subtract {logitsSSA}, %maxv_b : {tensorTy [B, NC]}\n"
    code := code ++ s!"    %exp_s = stablehlo.exponential %shifted : {tensorTy [B, NC]}\n"
    code := code ++ s!"    %sum_e = stablehlo.reduce(%exp_s init: %zf) applies stablehlo.add across dimensions = [1]\n"
    code := code ++ s!"           : ({tensorTy [B, NC]}, tensor<f32>) -> {tensorTy [B]}\n"
    code := code ++ s!"    %log_s = stablehlo.log %sum_e : {tensorTy [B]}\n"
    code := code ++ s!"    %log_s_b = stablehlo.broadcast_in_dim %log_s, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, NC]}\n"
    code := code ++ s!"    %log_p = stablehlo.subtract %shifted, %log_s_b : {tensorTy [B, NC]}\n"
    -- `onehotSSA` is the smoothed/onehot label tensor used by both the loss
    -- forward and the d_logits backward. Two construction paths:
    --   * Soft-label path: caller passes a `%y_soft : [B, NC] f32` already
    --     containing label smoothing + mixup/cutmix mixing.
    --   * Int-label path (default): build %onehot from int32 %y on the fly,
    --     applying label smoothing inline.
    let onehotSSA : String := if useSoftLabels then "%y_soft" else "%onehot"
    if !useSoftLabels then
      code := code ++ s!"    %iota = stablehlo.iota dim = 1 : {tensorTy [B, NC]}".replace "xf32>" "xi32>"  ++ "\n"
      code := code ++ s!"    %y_b = stablehlo.broadcast_in_dim %y, dims = [0] : ({tensorTy [B]}".replace "xf32>" "xi32>" ++ s!") -> {tensorTy [B, NC]}".replace "xf32>" "xi32>" ++ "\n"
      let i1Ty := s!"tensor<{B}x{NC}xi1>"
      code := code ++ s!"    %mask = stablehlo.compare EQ, %iota, %y_b : ({tensorTy [B, NC]}".replace "xf32>" "xi32>" ++ s!", {tensorTy [B, NC]}".replace "xf32>" "xi32>" ++ s!") -> {i1Ty}\n"
      let smoothOn := if labelSmoothing > 0.0 then 1.0 - labelSmoothing + labelSmoothing / nClasses.toFloat else 1.0
      let smoothOff := if labelSmoothing > 0.0 then labelSmoothing / nClasses.toFloat else 0.0
      code := code ++ s!"    %onef = stablehlo.constant dense<{smoothOn}> : {tensorTy [B, NC]}\n"
      code := code ++ s!"    %zerof = stablehlo.constant dense<{smoothOff}> : {tensorTy [B, NC]}\n"
      code := code ++ s!"    %onehot = stablehlo.select %mask, %onef, %zerof : {i1Ty}, {tensorTy [B, NC]}\n"
    code := code ++ s!"    %weighted = stablehlo.multiply %log_p, {onehotSSA} : {tensorTy [B, NC]}\n"
    code := code ++ s!"    %Bc = stablehlo.constant dense<{B}.0> : tensor<f32>\n"
    if useFocal then
      -- Focal loss: -(1-p_y)^γ · log(p_y). Compute per-sample log_p_y first
      -- (via the existing %weighted = log_p · onehot reduced along NC), then
      -- p_y = exp(log_p_y), focal_factor = exp(γ · log(1-p_y)).
      code := code ++ s!"    %log_p_y = stablehlo.reduce(%weighted init: %zf) applies stablehlo.add across dimensions = [1]\n"
      code := code ++ s!"           : ({tensorTy [B, NC]}, tensor<f32>) -> {tensorTy [B]}\n"
      code := code ++ s!"    %p_y = stablehlo.exponential %log_p_y : {tensorTy [B]}\n"
      code := code ++ s!"    %onef_b = stablehlo.constant dense<1.0> : {tensorTy [B]}\n"
      code := code ++ s!"    %omp_y = stablehlo.subtract %onef_b, %p_y : {tensorTy [B]}\n"
      code := code ++ s!"    %eps_b = stablehlo.constant dense<1.0e-7> : {tensorTy [B]}\n"
      code := code ++ s!"    %omp_clamped = stablehlo.maximum %omp_y, %eps_b : {tensorTy [B]}\n"
      code := code ++ s!"    %log_omp = stablehlo.log %omp_clamped : {tensorTy [B]}\n"
      code := code ++ s!"    %gamma_b = stablehlo.constant dense<{focalGamma}> : {tensorTy [B]}\n"
      code := code ++ s!"    %g_log_omp = stablehlo.multiply %gamma_b, %log_omp : {tensorTy [B]}\n"
      code := code ++ s!"    %focal_factor = stablehlo.exponential %g_log_omp : {tensorTy [B]}\n"
      code := code ++ s!"    %focal_per = stablehlo.multiply %focal_factor, %log_p_y : {tensorTy [B]}\n"
      code := code ++ s!"    %total = stablehlo.reduce(%focal_per init: %zf) applies stablehlo.add across dimensions = [0]\n"
      code := code ++ s!"           : ({tensorTy [B]}, tensor<f32>) -> tensor<f32>\n"
      code := code ++ s!"    %mean = stablehlo.divide %total, %Bc : tensor<f32>\n"
      code := code ++ s!"    %loss = stablehlo.negate %mean : tensor<f32>\n"
    else
      code := code ++ s!"    %total = stablehlo.reduce(%weighted init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
      code := code ++ s!"           : ({tensorTy [B, NC]}, tensor<f32>) -> tensor<f32>\n"
      code := code ++ s!"    %mean = stablehlo.divide %total, %Bc : tensor<f32>\n"
      code := code ++ s!"    %loss = stablehlo.negate %mean : tensor<f32>\n"

    -- ═══════════════ BACKWARD ═══════════════
    code := code ++ "\n    // ════════════════════════════════════════════════════════════════\n"
    code := code ++ "    // BACKWARD PASS — each section cites its proof in LeanMlir/Proofs/.\n"
    code := code ++ "    // The walkthrough at the bottom of CNN.lean shows the full chain.\n"
    code := code ++ "    // ════════════════════════════════════════════════════════════════\n"
    code := code ++ "    // ─── softmax cross-entropy loss gradient (MLP.lean: softmaxCE_grad) ───\n"
    code := code ++ "    //     d_logits = (softmax(z) - onehot(y)) / B\n"
    code := code ++ s!"    %sum_e_b = stablehlo.broadcast_in_dim %sum_e, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, NC]}\n"
    code := code ++ s!"    %softmax = stablehlo.divide %exp_s, %sum_e_b : {tensorTy [B, NC]}\n"
    code := code ++ s!"    %sm_moh = stablehlo.subtract %softmax, {onehotSSA} : {tensorTy [B, NC]}\n"
    code := code ++ s!"    %Bc_nc = stablehlo.broadcast_in_dim %Bc, dims = [] : (tensor<f32>) -> {tensorTy [B, NC]}\n"
    if useFocal then
      -- Focal multiplier on standard CE backward:
      --   d_logits = focal_mult · (softmax - onehot) / B
      -- where focal_mult = (1-p_y)^γ - γ · p_y · (1-p_y)^(γ-1) · log(p_y)
      -- Derivation: dL/dp_y = γ·(1-p_y)^(γ-1)·log(p_y) - (1-p_y)^γ/p_y;
      -- chain through softmax to get the (softmax - onehot) factor.
      code := code ++ s!"    %onef_g1 = stablehlo.constant dense<1.0> : {tensorTy [B]}\n"
      code := code ++ s!"    %gm1 = stablehlo.constant dense<{focalGamma - 1.0}> : {tensorTy [B]}\n"
      code := code ++ s!"    %gm1_log_omp = stablehlo.multiply %gm1, %log_omp : {tensorTy [B]}\n"
      code := code ++ s!"    %ff_gm1 = stablehlo.exponential %gm1_log_omp : {tensorTy [B]}\n"
      code := code ++ s!"    %g_p_y = stablehlo.multiply %gamma_b, %p_y : {tensorTy [B]}\n"
      code := code ++ s!"    %g_p_y_lpy = stablehlo.multiply %g_p_y, %log_p_y : {tensorTy [B]}\n"
      code := code ++ s!"    %focal_mult_term2 = stablehlo.multiply %ff_gm1, %g_p_y_lpy : {tensorTy [B]}\n"
      code := code ++ s!"    %focal_mult = stablehlo.subtract %focal_factor, %focal_mult_term2 : {tensorTy [B]}\n"
      code := code ++ s!"    %focal_mult_b = stablehlo.broadcast_in_dim %focal_mult, dims = [0] : ({tensorTy [B]}) -> {tensorTy [B, NC]}\n"
      code := code ++ s!"    %fm_smmoh = stablehlo.multiply %focal_mult_b, %sm_moh : {tensorTy [B, NC]}\n"
      code := code ++ s!"    %d_logits = stablehlo.divide %fm_smmoh, %Bc_nc : {tensorTy [B, NC]}\n"
    else
      code := code ++ s!"    %d_logits = stablehlo.divide %sm_moh, %Bc_nc : {tensorTy [B, NC]}\n"
  let nRec := records.size
  -- Track projection pidx values whose backward was emitted inline during skip-add handling
  let mut bwdDone : Array Nat := #[]

  for ri in [:nRec] do
    let r : FwdRec := records[nRec - 1 - ri]!
    let p := r.pidx.getD 0
    -- Skip projection records whose backward was already emitted inline
    if r.pidx.isSome && bwdDone.contains p then
      pure ()
    else
    match r.layer with
    | .dense _fanIn fanOut act =>
      code := code ++ s!"    // ─── dense backward (MLP.lean: dense_has_vjp, dense_weight_grad_correct, dense_bias_grad_correct) ───\n"
      code := code ++ s!"    //     d_W = outer(x, effGrad)  |  d_b = effGrad  |  d_x = W · effGrad\n"
      let effGrad := if act == .relu then s!"%gp{p}" else gradSSA
      if act == .relu then
        let oTy := tensorTy r.outShape
        let i1Ty := oTy.replace "xf32>" "xi1>"
        code := code ++ s!"    %rm{p} = stablehlo.compare GT, {r.preActSSA}, %dz{p} : ({oTy}, {oTy}) -> {i1Ty}\n"
        code := code ++ s!"    {effGrad} = stablehlo.select %rm{p}, {gradSSA}, %dz{p} : {i1Ty}, {oTy}\n"
      code := code ++ s!"    %d_W{p} = stablehlo.dot_general {r.inputSSA}, {effGrad}, contracting_dims = [0] x [0],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"            : ({tensorTy r.inShape}, {tensorTy r.outShape}) -> {tensorTy [r.inShape[1]!, fanOut]}\n"
      code := code ++ s!"    %d_b{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0]\n"
      code := code ++ s!"          : ({tensorTy r.outShape}, tensor<f32>) -> {tensorTy [fanOut]}\n"
      code := code ++ s!"    %d_in{p} = stablehlo.dot_general {effGrad}, %W{p}, contracting_dims = [1] x [1],\n"
      code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
      code := code ++ s!"            : ({tensorTy r.outShape}, {tensorTy [r.inShape[1]!, fanOut]}) -> {tensorTy r.inShape}\n"
      gradSSA := s!"%d_in{p}"
      gradShape := r.inShape

    | .conv2d ic oc kSize _pad act =>
      match r.outShape with
      | [b, _oc, h, w] =>
        code := code ++ s!"    // ─── conv2d backward (CNN.lean: conv2d_has_vjp3 for dx, conv2d_weight_grad_has_vjp for dW) ───\n"
        code := code ++ s!"    //     dW via transpose trick; dx via reversed+transposed kernel convolution\n"
        let oTy := tensorTy r.outShape
        let i1Ty := oTy.replace "xf32>" "xi1>"
        let effGrad := if act == .relu then s!"%gpc{p}" else gradSSA
        if act == .relu then
          code := code ++ s!"    %rmc{p} = stablehlo.compare GT, {r.preActSSA}, %cvz{p} : ({oTy}, {oTy}) -> {i1Ty}\n"
          code := code ++ s!"    {effGrad} = stablehlo.select %rmc{p}, {gradSSA}, %cvz{p} : {i1Ty}, {oTy}\n"
        let inC := r.inShape[1]!
        code := code ++ s!"    %bt_in{p} = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [inC, b, h, w]}\n"
        code := code ++ s!"    %bt_g{p} = stablehlo.transpose {effGrad}, dims = [1, 0, 2, 3] : ({oTy}) -> {tensorTy [oc, b, h, w]}\n"
        let pad := (kSize - 1) / 2
        let dWrawShape := [inC, oc, kSize, kSize]
        code := code ++ s!"    %dWr{p} = \"stablehlo.convolution\"(%bt_in{p}, %bt_g{p}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [inC, b, h, w]}, {tensorTy [oc, b, h, w]}) -> {tensorTy dWrawShape}\n"
        code := code ++ s!"    %d_W{p} = stablehlo.transpose %dWr{p}, dims = [1, 0, 2, 3] : ({tensorTy dWrawShape}) -> {tensorTy [oc, inC, kSize, kSize]}\n"
        code := code ++ s!"    %d_b{p} = stablehlo.reduce({effGrad} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
        code := code ++ s!"          : ({oTy}, tensor<f32>) -> {tensorTy [oc]}\n"
        code := code ++ s!"    %Wt{p} = stablehlo.transpose %W{p}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, kSize, kSize]}) -> {tensorTy [ic, oc, kSize, kSize]}\n"
        code := code ++ s!"    %Wrev{p} = stablehlo.reverse %Wt{p}, dims = [2, 3] : {tensorTy [ic, oc, kSize, kSize]}\n"
        code := code ++ s!"    %d_x{p} = \"stablehlo.convolution\"({effGrad}, %Wrev{p}) " ++ "{\n"
        code := code ++ convAttrBlock pad
        code := code ++ s!"      " ++ "}" ++ s!" : ({oTy}, {tensorTy [ic, oc, kSize, kSize]}) -> {tensorTy r.inShape}\n"
        gradSSA := s!"%d_x{p}"
        gradShape := r.inShape
      | _ => pure ()

    | .convBn _ic _oc _kSize _stride _ =>
      if r.isDepthwise then
        let (bwdCode, dxSSA, _dWSSA) := emitDepthwiseConvBnBackward r gradSSA
        code := code ++ bwdCode
        gradSSA := dxSSA
        gradShape := r.inShape
      else
        let (bwdCode, dxSSA, _dWSSA) := emitConvBnBackward r gradSSA
        code := code ++ bwdCode
        gradSSA := dxSSA
        gradShape := r.inShape
      -- Accumulate skip gradient if tagged
      if r.addSkipGrad != "" then
        code := code ++ s!"    %cbg_skip_acc{p} = stablehlo.add {gradSSA}, {r.addSkipGrad} : {tensorTy r.inShape}\n"
        gradSSA := s!"%cbg_skip_acc{p}"

    | .globalAvgPool =>
      -- Check if this is an SE block record
      if r.isSE then
        match r.inShape with
        | [b, _mid, h, w] =>
          code := code ++ s!"    // ─── Squeeze-and-Excitation backward (SE.lean: seBlock_has_vjp) ───\n"
          code := code ++ s!"    //     dx = elemwiseProduct_has_vjp(x, gate): dx_direct + dx_gap\n"
          code := code ++ s!"    //     Gate path uses dense_has_vjp for two 1x1 convs + sigmoid/h-sigmoid gradient\n"
          let mid := r.seMidFull
          let seMid := r.seMid
          let pRed := r.sePidxRed
          let pExp := r.sePidxExp
          let xTy := tensorTy r.inShape
          let tag := s!"_b{pRed}"
          -- Input gradient: dL/d_y where y = x * sig_bc
          -- 1. dx_direct = grad * sig_bc  (same shape as x)
          code := code ++ s!"    %seb_sigr{tag} = stablehlo.reshape {r.seSigSSA} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
          code := code ++ s!"    %seb_sigb{tag} = stablehlo.broadcast_in_dim %seb_sigr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
          code := code ++ s!"    %seb_dx_dir{tag} = stablehlo.multiply {gradSSA}, %seb_sigb{tag} : {xTy}\n"
          -- 2. d_sig = reduce(grad * x, dims=[2,3]) → (B, mid)
          code := code ++ s!"    %seb_gx{tag} = stablehlo.multiply {gradSSA}, {r.seInputSSA} : {xTy}\n"
          code := code ++ s!"    %seb_dsig{tag} = stablehlo.reduce(%seb_gx{tag} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
          code := code ++ s!"          : ({xTy}, tensor<f32>) -> {tensorTy [b, mid]}\n"
          -- 3. d_se_eb: gate backward.
          if r.seVariant then
            -- h-sigmoid backward: d/dx h_sigmoid(x) = 1/6 if -3 < x < 3 else 0
            let i1Ty2 := (tensorTy [b, mid]).replace "xf32>" "xi1>"
            code := code ++ s!"    %seb_hsgn3{tag} = stablehlo.constant dense<-3.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsg3{tag} = stablehlo.constant dense<3.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsgs{tag} = stablehlo.constant dense<0.16666667> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsgz{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsglt{tag} = stablehlo.compare LT, {r.seEbSSA}, %seb_hsgn3{tag} : ({tensorTy [b, mid]}, {tensorTy [b, mid]}) -> {i1Ty2}\n"
            code := code ++ s!"    %seb_hsggt{tag} = stablehlo.compare GT, {r.seEbSSA}, %seb_hsg3{tag} : ({tensorTy [b, mid]}, {tensorTy [b, mid]}) -> {i1Ty2}\n"
            code := code ++ s!"    %seb_hsgw1{tag} = stablehlo.select %seb_hsglt{tag}, %seb_hsgz{tag}, %seb_hsgs{tag} : {i1Ty2}, {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_hsggrad{tag} = stablehlo.select %seb_hsggt{tag}, %seb_hsgz{tag}, %seb_hsgw1{tag} : {i1Ty2}, {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_deb{tag} = stablehlo.multiply %seb_dsig{tag}, %seb_hsggrad{tag} : {tensorTy [b, mid]}\n"
          else
            -- Sigmoid backward: d_eb = d_sig * sig * (1 - sig)
            code := code ++ s!"    %seb_one{tag} = stablehlo.constant dense<1.0> : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_1ms{tag} = stablehlo.subtract %seb_one{tag}, {r.seSigSSA} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_ssp{tag} = stablehlo.multiply {r.seSigSSA}, %seb_1ms{tag} : {tensorTy [b, mid]}\n"
            code := code ++ s!"    %seb_deb{tag} = stablehlo.multiply %seb_dsig{tag}, %seb_ssp{tag} : {tensorTy [b, mid]}\n"
          -- 4. d_b_se_exp = reduce(d_se_eb, dim=0) → (mid,)
          code := code ++ s!"    %d_b{pExp} = stablehlo.reduce(%seb_deb{tag} init: %zf) applies stablehlo.add across dimensions = [0]\n"
          code := code ++ s!"          : ({tensorTy [b, mid]}, tensor<f32>) -> {tensorTy [mid]}\n"
          -- 5. d_W_se_exp: dense backward. forward: eb[b,m] = sum_s rsw[b,s] * We[m,s]
          --    d_We[m, s] = sum_b d_eb[b, m] * rsw[b, s]
          code := code ++ s!"    %seb_dWe_flat{tag} = stablehlo.dot_general %seb_deb{tag}, {r.seRswSSA},\n"
          code := code ++ s!"              contracting_dims = [0] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [b, seMid]}) -> {tensorTy [mid, seMid]}\n"
          code := code ++ s!"    %d_W{pExp} = stablehlo.reshape %seb_dWe_flat{tag} : ({tensorTy [mid, seMid]}) -> {tensorTy [mid, seMid, 1, 1]}\n"
          -- 6. d_rsw: forward eb = rsw @ We^T; d_rsw[b, s] = sum_m d_eb[b, m] * We[m, s]
          code := code ++ s!"    %seb_We{tag} = stablehlo.reshape %W{pExp} : ({tensorTy [mid, seMid, 1, 1]}) -> {tensorTy [mid, seMid]}\n"
          code := code ++ s!"    %seb_drsw{tag} = stablehlo.dot_general %seb_deb{tag}, %seb_We{tag},\n"
          code := code ++ s!"              contracting_dims = [1] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, mid]}, {tensorTy [mid, seMid]}) -> {tensorTy [b, seMid]}\n"
          -- 7. d_rb: reduce activation backward.
          if r.seVariant then
            -- ReLU backward: d_rb = d_rsw if rb > 0 else 0
            let i1TyR := (tensorTy [b, seMid]).replace "xf32>" "xi1>"
            code := code ++ s!"    %seb_rz{tag} = stablehlo.constant dense<0.0> : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_rmask{tag} = stablehlo.compare GT, {r.seRbSSA}, %seb_rz{tag} : ({tensorTy [b, seMid]}, {tensorTy [b, seMid]}) -> {i1TyR}\n"
            code := code ++ s!"    %seb_drb{tag} = stablehlo.select %seb_rmask{tag}, %seb_drsw{tag}, %seb_rz{tag} : {i1TyR}, {tensorTy [b, seMid]}\n"
          else
            -- Swish backward: d_rb = d_rsw * (sig_r + rb * sig_r * (1 - sig_r))
            code := code ++ s!"    %seb_rsig{tag} = stablehlo.logistic {r.seRbSSA} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_one_s{tag} = stablehlo.constant dense<1.0> : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_1mrs{tag} = stablehlo.subtract %seb_one_s{tag}, %seb_rsig{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_rpart{tag} = stablehlo.multiply {r.seRbSSA}, %seb_1mrs{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_1pr{tag} = stablehlo.add %seb_one_s{tag}, %seb_rpart{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_dsw{tag} = stablehlo.multiply %seb_rsig{tag}, %seb_1pr{tag} : {tensorTy [b, seMid]}\n"
            code := code ++ s!"    %seb_drb{tag} = stablehlo.multiply %seb_drsw{tag}, %seb_dsw{tag} : {tensorTy [b, seMid]}\n"
          -- 8. d_b_se_red = reduce(d_rb, dim=0)
          code := code ++ s!"    %d_b{pRed} = stablehlo.reduce(%seb_drb{tag} init: %zf) applies stablehlo.add across dimensions = [0]\n"
          code := code ++ s!"          : ({tensorTy [b, seMid]}, tensor<f32>) -> {tensorTy [seMid]}\n"
          -- 9. d_W_se_red: forward rm[b, s] = sum_m g[b, m] * Wr[s, m]
          --    d_Wr[s, m] = sum_b d_rb[b, s] * g[b, m]
          code := code ++ s!"    %seb_dWr_flat{tag} = stablehlo.dot_general %seb_drb{tag}, {r.seGapSSA},\n"
          code := code ++ s!"              contracting_dims = [0] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [b, mid]}) -> {tensorTy [seMid, mid]}\n"
          code := code ++ s!"    %d_W{pRed} = stablehlo.reshape %seb_dWr_flat{tag} : ({tensorTy [seMid, mid]}) -> {tensorTy [seMid, mid, 1, 1]}\n"
          -- 10. d_gap: d_g[b, m] = sum_s d_rb[b, s] * Wr[s, m]
          code := code ++ s!"    %seb_Wr{tag} = stablehlo.reshape %W{pRed} : ({tensorTy [seMid, mid, 1, 1]}) -> {tensorTy [seMid, mid]}\n"
          code := code ++ s!"    %seb_dg{tag} = stablehlo.dot_general %seb_drb{tag}, %seb_Wr{tag},\n"
          code := code ++ s!"              contracting_dims = [1] x [0],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({tensorTy [b, seMid]}, {tensorTy [seMid, mid]}) -> {tensorTy [b, mid]}\n"
          -- 11. GAP backward: dx_gap = broadcast(d_g) / (H*W)
          code := code ++ s!"    %seb_dgr{tag} = stablehlo.reshape %seb_dg{tag} : ({tensorTy [b, mid]}) -> {tensorTy [b, mid, 1, 1]}\n"
          code := code ++ s!"    %seb_dgb{tag} = stablehlo.broadcast_in_dim %seb_dgr{tag}, dims = [0, 1, 2, 3] : ({tensorTy [b, mid, 1, 1]}) -> {xTy}\n"
          code := code ++ s!"    %seb_gN{tag} = stablehlo.constant dense<{h * w}.0> : {xTy}\n"
          code := code ++ s!"    %seb_dx_gap{tag} = stablehlo.divide %seb_dgb{tag}, %seb_gN{tag} : {xTy}\n"
          -- 12. Total dx = dx_direct + dx_gap
          code := code ++ s!"    %seb_dx{tag} = stablehlo.add %seb_dx_dir{tag}, %seb_dx_gap{tag} : {xTy}\n"
          gradSSA := s!"%seb_dx{tag}"
          gradShape := r.inShape
        | _ => pure ()
      else if r.addSkipGrad.startsWith "proj:" || r.addSkipGrad == "identity" then
        code := code ++ s!"    // ─── residual skip-add backward (Residual.lean: residual_has_vjp via biPath3_has_vjp) ───\n"
        code := code ++ s!"    //     dy flows to both branches: main path + skip (identity or 1x1-conv projection)\n"
        -- Residual skip-add backward
        let oTy := tensorTy r.outShape
        let addId := r.preActSSA.replace "%rb_add" ""
        if r.hasRelu then
          -- ReLU backward (ResNet-style)
          let i1Ty := oTy.replace "xf32>" "xi1>"
          code := code ++ s!"    %rb_rcmp{addId} = stablehlo.compare GT, {r.preActSSA}, %rb_rz{addId} : ({oTy}, {oTy}) -> {i1Ty}\n"
          code := code ++ s!"    %rb_dsum{addId} = stablehlo.select %rb_rcmp{addId}, {gradSSA}, %rb_rz{addId} : {i1Ty}, {oTy}\n"
          gradSSA := s!"%rb_dsum{addId}"
        else
          -- No ReLU after skip add (MobileNetV2 inverted residual)
          -- Gradient passes through the add directly
          pure ()
        let skipGradSrc := gradSSA  -- gradient flows to both branches
        if r.addSkipGrad.startsWith "proj:" then
          -- Projected skip: emit projection backward inline, mark as done
          let projPidx := (r.addSkipGrad.drop 5).toNat!
          bwdDone := bwdDone.push projPidx
          let mut projRec : FwdRec := default
          for rr in records do
            if rr.pidx == some projPidx then projRec := rr
          let (projBwd, projDx, _) := emitConvBnBackward projRec skipGradSrc
          code := code ++ projBwd
          code := code ++ s!"    %rb_dskip{addId} = stablehlo.reshape {projDx} : ({tensorTy r.inShape}) -> {tensorTy r.inShape}\n"
        else
          -- Identity skip: gradient passes through directly
          code := code ++ s!"    %rb_dskip{addId} = stablehlo.reshape {skipGradSrc} : ({oTy}) -> {oTy}\n"
        gradShape := r.outShape
      else
        -- Normal globalAvgPool backward
        match r.inShape with
        | [_b, _c, h, w] =>
          code := code ++ s!"    // ─── global avg pool backward (uniform broadcast of dy / (H·W), derivable from pdiv_reindex + sum rule) ───\n"
          let spatialN := h * w
          code := code ++ s!"    %dgap_bc{r.pos} = stablehlo.broadcast_in_dim {gradSSA}, dims = [0, 1] : ({tensorTy r.outShape}) -> {tensorTy r.inShape}\n"
          code := code ++ s!"    %dgap_N{r.pos} = stablehlo.constant dense<{spatialN}.0> : {tensorTy r.inShape}\n"
          code := code ++ s!"    %dgap{r.pos} = stablehlo.divide %dgap_bc{r.pos}, %dgap_N{r.pos} : {tensorTy r.inShape}\n"
          gradSSA := s!"%dgap{r.pos}"
          gradShape := r.inShape
        | _ => pure ()

    | .maxPool _size stride =>
      -- MaxPool backward via tile-compare-select (avoids select_and_scatter which
      -- IREE doesn't support). Works correctly for stride==size (non-overlapping).
      -- For overlapping pools (size>stride), use stride==size pooling instead.
      match r.inShape with
      | [b, c, h, w] =>
        code := code ++ s!"    // ─── max-pool backward (CNN.lean: maxPool2_has_vjp3 — gradient routes to argmax) ───\n"
        let oH := r.outShape[2]!
        let oW := r.outShape[3]!
        -- Tile gradient: (B,C,oH,oW) → (B,C,oH,1,oW,1) → (B,C,oH,S,oW,S) → (B,C,oH*S,oW*S)
        let tileShape := [b, c, oH, stride, oW, stride]
        let expandH := oH * stride
        let expandW := oW * stride
        code := code ++ s!"    %mp_gr{r.pos} = stablehlo.reshape {gradSSA} : ({tensorTy r.outShape}) -> {tensorTy [b, c, oH, 1, oW, 1]}\n"
        code := code ++ s!"    %mp_gt{r.pos} = stablehlo.broadcast_in_dim %mp_gr{r.pos}, dims = [0, 1, 2, 3, 4, 5] : ({tensorTy [b, c, oH, 1, oW, 1]}) -> {tensorTy tileShape}\n"
        code := code ++ s!"    %mp_ge{r.pos} = stablehlo.reshape %mp_gt{r.pos} : ({tensorTy tileShape}) -> {tensorTy [b, c, expandH, expandW]}\n"
        -- Tile pooled output similarly (for max comparison mask)
        code := code ++ s!"    %mp_pr{r.pos} = stablehlo.reshape {r.outputSSA} : ({tensorTy r.outShape}) -> {tensorTy [b, c, oH, 1, oW, 1]}\n"
        code := code ++ s!"    %mp_pt{r.pos} = stablehlo.broadcast_in_dim %mp_pr{r.pos}, dims = [0, 1, 2, 3, 4, 5] : ({tensorTy [b, c, oH, 1, oW, 1]}) -> {tensorTy tileShape}\n"
        code := code ++ s!"    %mp_pe{r.pos} = stablehlo.reshape %mp_pt{r.pos} : ({tensorTy tileShape}) -> {tensorTy [b, c, expandH, expandW]}\n"
        -- Slice to input size if expanded > input (can happen with non-exact division)
        let mut gradTile := s!"%mp_ge{r.pos}"
        let mut poolTile := s!"%mp_pe{r.pos}"
        if expandH > h || expandW > w then
          let eTy := tensorTy [b, c, expandH, expandW]
          let inTy := tensorTy [b, c, h, w]
          code := code ++ s!"    %mp_gs{r.pos} = \"stablehlo.slice\"({gradTile}) " ++ "{" ++ s!"start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {b}, {c}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++ "}" ++ s!" : ({eTy}) -> {inTy}\n"
          code := code ++ s!"    %mp_ps{r.pos} = \"stablehlo.slice\"({poolTile}) " ++ "{" ++ s!"start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {b}, {c}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++ "}" ++ s!" : ({eTy}) -> {inTy}\n"
          gradTile := s!"%mp_gs{r.pos}"
          poolTile := s!"%mp_ps{r.pos}"
        -- Compare original input with tiled max → mask where max was
        let inTy := tensorTy [b, c, h, w]
        let maskTy := inTy.replace "xf32>" "xi1>"
        code := code ++ s!"    %mp_mask{r.pos} = stablehlo.compare EQ, {poolTile}, {r.inputSSA} : ({inTy}, {inTy}) -> {maskTy}\n"
        code := code ++ s!"    %mp_zg{r.pos} = stablehlo.constant dense<0.0> : {inTy}\n"
        code := code ++ s!"    %mp_dx{r.pos} = stablehlo.select %mp_mask{r.pos}, {gradTile}, %mp_zg{r.pos} : {maskTy}, {inTy}\n"
        gradSSA := s!"%mp_dx{r.pos}"
        gradShape := r.inShape
        -- UNet skip plumbing: if this maxPool was emitted as part of a
        -- `unetDown` and tagged with a skip-grad slot, accumulate the
        -- skip-half gradient (defined earlier in the reverse walk by the
        -- matching `unetUp`'s UnetUpConcat backward).
        if r.addSkipGrad != "" then
          code := code ++ s!"    %mp_skip{r.pos} = stablehlo.add {gradSSA}, {r.addSkipGrad} : {inTy}\n"
          gradSSA := s!"%mp_skip{r.pos}"
      | _ => pure ()

    | .flatten =>
      code := code ++ s!"    // ─── flatten backward (inverse reshape; derivable from pdiv_reindex via Tensor3.flatten bijection) ───\n"
      code := code ++ s!"    %ufl{r.pos} = stablehlo.reshape {gradSSA} : ({tensorTy gradShape}) -> {tensorTy r.inShape}\n"
      gradSSA := s!"%ufl{r.pos}"
      gradShape := r.inShape

    | .bilinearUpsample scale =>
      -- VJP of `Y = Wy · X · Wxᵀ` (precomputed const Wy, Wx).
      --   dM = dY · Wx       (contract along W_out)
      --   dXᵀ = dM · Wy      (contract along H_out)
      --   dX  = transpose dXᵀ
      -- Wy / Wx are recomputed and re-emitted; IREE folds duplicate
      -- constants so this is purely a textual cost.
      match r.inShape with
      | [b, c, h, w] =>
        let oH := h * scale
        let oW := w * scale
        let wyStr := floatMatrixToMlirDense (bilinearWeights1D h scale)
        let wxStr := floatMatrixToMlirDense (bilinearWeights1D w scale)
        let wyTy := s!"tensor<{oH}x{h}xf32>"
        let wxTy := s!"tensor<{oW}x{w}xf32>"
        let dmShape := [b, c, oH, w]
        let dxtShape := [b, c, w, h]
        code := code ++ s!"    // ─── bilinear upsample backward (transpose of forward matmul-pair) ───\n"
        code := code ++ s!"    %bu_bwy{r.pos} = stablehlo.constant dense<{wyStr}> : {wyTy}\n"
        code := code ++ s!"    %bu_bwx{r.pos} = stablehlo.constant dense<{wxStr}> : {wxTy}\n"
        code := code ++ s!"    %bu_dm{r.pos} = stablehlo.dot_general {gradSSA}, %bu_bwx{r.pos}, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : ({tensorTy r.outShape}, {wxTy}) -> {tensorTy dmShape}\n"
        code := code ++ s!"    %bu_dt{r.pos} = stablehlo.dot_general %bu_dm{r.pos}, %bu_bwy{r.pos}, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : ({tensorTy dmShape}, {wyTy}) -> {tensorTy dxtShape}\n"
        code := code ++ s!"    %bu_dx{r.pos} = stablehlo.transpose %bu_dt{r.pos}, dims = [0, 1, 3, 2] : ({tensorTy dxtShape}) -> {tensorTy r.inShape}\n"
        gradSSA := s!"%bu_dx{r.pos}"
        gradShape := r.inShape
      | _ => pure ()

    | .unetUp _ic _oc =>
      if r.isUnetUpConcat then
        -- Concat-split backward at the UNet decoder skip junction.
        -- Incoming `gradSSA` has shape `r.outShape = [b, ic+oc, 2h, 2w]`.
        -- Split along the channel axis into:
        --   `g_upsampled : [b, ic, 2h, 2w]` (decoder-half) — flows out via gradSSA
        --   `g_skipFeat  : [b, oc, 2h, 2w]` (skip-half)    — saved as
        --                                                   `%unet_skip_g{e}`
        --                                                   for the matching
        --                                                   unetDown's maxPool.
        let e := r.unetEncoderIdx
        let aShape := r.inShape          -- decoder upsampled side
        let bShape := r.unetSkipShape    -- encoder skip side
        let outTy  := tensorTy r.outShape
        let aTy    := tensorTy aShape
        let bTy    := tensorTy bShape
        match aShape, bShape, r.outShape with
        | [n, ca, h, w], [_, cb, _, _], [_, _, _, _] =>
          code := code ++ s!"    // ─── UNet concat-split backward (channelSplit_has_vjp ↔ channelConcat) ───\n"
          code := code ++ s!"    //     dConcat[:, :ca]  → upsampled-half (decoder, flows to bilinearUpsample backward)\n"
          code := code ++ s!"    //     dConcat[:, ca:]  → skip-half (saved as %unet_skip_g{e}, accumulated at unetDown maxPool)\n"
          code := code ++ s!"    %uut_a{r.pos} = \"stablehlo.slice\"({gradSSA}) " ++ "{" ++
            s!"start_indices = array<i64: 0, 0, 0, 0>, limit_indices = array<i64: {n}, {ca}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++
            "}" ++ s!" : ({outTy}) -> {aTy}\n"
          code := code ++ s!"    %unet_skip_g{e} = \"stablehlo.slice\"({gradSSA}) " ++ "{" ++
            s!"start_indices = array<i64: 0, {ca}, 0, 0>, limit_indices = array<i64: {n}, {ca + cb}, {h}, {w}>, strides = array<i64: 1, 1, 1, 1>" ++
            "}" ++ s!" : ({outTy}) -> {bTy}\n"
          gradSSA := s!"%uut_a{r.pos}"
          gradShape := aShape
        | _, _, _ => pure ()

    | .patchEmbed _ic _dim _p _nP =>
      if r.isPatchEmbed then
        -- Patch embedding backward
        match r.inShape with
        | [b, ic, h, w] =>
          code := code ++ s!"    // ─── patch-embed backward (composite — no single theorem) ───\n"
          code := code ++ s!"    //     conv projection: CNN.lean: conv2d_has_vjp3 / conv2d_weight_grad_has_vjp\n"
          code := code ++ s!"    //     CLS token + positional embedding: pdiv_add + pdiv_reindex (Tensor.lean)\n"
          let dim := r.pePDim
          let pSize := r.pePSize
          let nP := r.pePNp
          let nT := nP + 1
          let pW := r.pePWPidx
          let pB := r.pePBPidx
          let pCls := r.peClsPidx
          let pPos := r.pePosPidx
          let outTy := tensorTy r.outShape
          let tag := s!"peb{r.pos}"
          code := code ++ s!"    %d_W{pPos} = stablehlo.reduce({gradSSA} init: %zf) applies stablehlo.add across dimensions = [0]\n"
          code := code ++ s!"          : ({outTy}, tensor<f32>) -> {tensorTy [nT, dim]}\n"
          let clsShape := [b, 1, dim]
          code := code ++ s!"    %{tag}_cls_sl = \"stablehlo.slice\"({gradSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 0, 0>, limit_indices = array<i64: {b}, 1, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({outTy}) -> {tensorTy clsShape}\n"
          code := code ++ s!"    %d_W{pCls} = stablehlo.reduce(%{tag}_cls_sl init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({tensorTy clsShape}, tensor<f32>) -> {tensorTy [dim]}\n"
          let patchShape := [b, nP, dim]
          code := code ++ s!"    %{tag}_pat_sl = \"stablehlo.slice\"({gradSSA}) " ++ "{" ++ s!" start_indices = array<i64: 0, 1, 0>, limit_indices = array<i64: {b}, {nT}, {dim}>, strides = array<i64: 1, 1, 1>" ++ "}" ++ s!" : ({outTy}) -> {tensorTy patchShape}\n"
          code := code ++ s!"    %{tag}_pat_tr = stablehlo.transpose %{tag}_pat_sl, dims = [0, 2, 1] : ({tensorTy patchShape}) -> {tensorTy [b, dim, nP]}\n"
          let oH := h / pSize
          let oW := w / pSize
          let convOut := [b, dim, oH, oW]
          code := code ++ s!"    %{tag}_grad = stablehlo.reshape %{tag}_pat_tr : ({tensorTy [b, dim, nP]}) -> {tensorTy convOut}\n"
          code := code ++ s!"    %d_b{pB} = stablehlo.reduce(%{tag}_grad init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
          code := code ++ s!"          : ({tensorTy convOut}, tensor<f32>) -> {tensorTy [dim]}\n"
          code := code ++ s!"    %{tag}_bt_in = stablehlo.transpose {r.inputSSA}, dims = [1, 0, 2, 3] : ({tensorTy r.inShape}) -> {tensorTy [ic, b, h, w]}\n"
          code := code ++ s!"    %{tag}_bt_g = stablehlo.transpose %{tag}_grad, dims = [1, 0, 2, 3] : ({tensorTy convOut}) -> {tensorTy [dim, b, oH, oW]}\n"
          code := code ++ s!"    %{tag}_dWr = \"stablehlo.convolution\"(%{tag}_bt_in, %{tag}_bt_g) " ++ "{\n"
          code := code ++ convAttrBlockFull 0 0 0 0 1 1 1 1 pSize pSize
          code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [ic, b, h, w]}, {tensorTy [dim, b, oH, oW]}) -> {tensorTy [ic, dim, pSize, pSize]}\n"
          code := code ++ s!"    %d_W{pW} = stablehlo.transpose %{tag}_dWr, dims = [1, 0, 2, 3] : ({tensorTy [ic, dim, pSize, pSize]}) -> {tensorTy [dim, ic, pSize, pSize]}\n"
          -- Input layer: no dx needed
          gradSSA := "// no_grad_needed"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()
    | .transformerEncoder _dim _heads _mlpDim _nBlocks _causal _keepSeq _ _ =>
      -- For records emitted during forward of a transformerEncoder layer:
      -- CLS slice, final LN, or transformer block (each discriminated by flags).
      if r.isClsSlice then
        match r.clsInShape with
        | [b, n, d] =>
          code := code ++ s!"    // ─── CLS slice backward (pdiv_reindex: scatter into position 0, zero elsewhere) ───\n"
          let preTy := tensorTy r.clsInShape
          let postTy := tensorTy r.outShape
          let slShape := [b, 1, d]
          code := code ++ s!"    %clsb_r{r.pos} = stablehlo.reshape {gradSSA} : ({postTy}) -> {tensorTy slShape}\n"
          let rest := n - 1
          let restShape := [b, rest, d]
          code := code ++ s!"    %clsb_z{r.pos} = stablehlo.constant dense<0.0> : {tensorTy restShape}\n"
          code := code ++ s!"    %clsb_cc{r.pos} = stablehlo.concatenate %clsb_r{r.pos}, %clsb_z{r.pos}, dim = 1 : ({tensorTy slShape}, {tensorTy restShape}) -> {preTy}\n"
          gradSSA := s!"%clsb_cc{r.pos}"
          gradShape := r.clsInShape
        | _ => pure ()
      else if r.isFinalLn then
        match r.inShape with
        | [b, n, d] =>
          code := code ++ s!"    // ─── final LN backward (LayerNorm.lean: layerNorm_has_vjp — applied to every token) ───\n"
          let p := r.finalLnPidx
          let ty := tensorTy r.inShape
          let bnTy := tensorTy [b, n]
          let dTy := tensorTy [d]
          let dF := d.toFloat
          code := code ++ s!"    %flnb_gn{p} = stablehlo.multiply {gradSSA}, {r.finalLnNormSSA} : {ty}\n"
          code := code ++ s!"    %d_b{p} = stablehlo.reduce({gradSSA} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %d_W{p} = stablehlo.reduce(%flnb_gn{p} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %flnb_g_bc{p} = stablehlo.broadcast_in_dim %W{p}, dims = [2] : ({dTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_dn{p} = stablehlo.multiply {gradSSA}, %flnb_g_bc{p} : {ty}\n"
          code := code ++ s!"    %flnb_sdn{p} = stablehlo.reduce(%flnb_dn{p} init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %flnb_dnn{p} = stablehlo.multiply %flnb_dn{p}, {r.finalLnNormSSA} : {ty}\n"
          code := code ++ s!"    %flnb_sdnn{p} = stablehlo.reduce(%flnb_dnn{p} init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %flnb_sdn_bc{p} = stablehlo.broadcast_in_dim %flnb_sdn{p}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_sdnn_bc{p} = stablehlo.broadcast_in_dim %flnb_sdnn{p}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_Nc{p} = stablehlo.constant dense<{dF}> : {ty}\n"
          code := code ++ s!"    %flnb_t1{p} = stablehlo.multiply %flnb_Nc{p}, %flnb_dn{p} : {ty}\n"
          code := code ++ s!"    %flnb_t2{p} = stablehlo.subtract %flnb_t1{p}, %flnb_sdn_bc{p} : {ty}\n"
          code := code ++ s!"    %flnb_t3{p} = stablehlo.multiply {r.finalLnNormSSA}, %flnb_sdnn_bc{p} : {ty}\n"
          code := code ++ s!"    %flnb_t4{p} = stablehlo.subtract %flnb_t2{p}, %flnb_t3{p} : {ty}\n"
          code := code ++ s!"    %flnb_istd_bc{p} = stablehlo.broadcast_in_dim {r.finalLnIstdSSA}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %flnb_invN{p} = stablehlo.constant dense<{1.0 / dF}> : {ty}\n"
          code := code ++ s!"    %flnb_scale{p} = stablehlo.multiply %flnb_istd_bc{p}, %flnb_invN{p} : {ty}\n"
          code := code ++ s!"    %flnb_dx{p} = stablehlo.multiply %flnb_scale{p}, %flnb_t4{p} : {ty}\n"
          gradSSA := s!"%flnb_dx{p}"
          gradShape := r.inShape
        | _ => pure ()
      else if r.isTransformerBlock then
        match r.inShape with
        | [b, n, d] =>
          code := code ++ s!"    // ════════════════════════════════════════════════════════════════\n"
          code := code ++ s!"    // Transformer block backward — see LeanMlir/Proofs/:\n"
          code := code ++ s!"    //   Residual skips (main + mlp)  Residual.lean: residual_has_vjp / biPath3_has_vjp\n"
          code := code ++ s!"    //   MLP (fc2, fc1)               MLP.lean: dense_has_vjp + dense_weight_grad_correct\n"
          code := code ++ s!"    //   GELU                         LayerNorm.lean: pdiv_gelu (tanh-form diagonal Jacobian)\n"
          code := code ++ s!"    //   LN2, LN1                     LayerNorm.lean: layerNorm_has_vjp\n"
          code := code ++ s!"    //   Scaled dot-product attention Attention.lean: sdpa_back_Q_correct / _K_ / _V_\n"
          code := code ++ s!"    //   Q/K/V/O projections          MLP.lean: dense_has_vjp (2D matmul generalization)\n"
          code := code ++ s!"    //   Row-softmax inside attention Attention.lean: rowSoftmax_has_vjp_mat\n"
          code := code ++ s!"    // ════════════════════════════════════════════════════════════════\n"
          let basePidx := r.tbBasePidx
          let pLn1   := basePidx
          let pWq    := basePidx + 1
          let pWk    := basePidx + 2
          let pWv    := basePidx + 3
          let pWo    := basePidx + 4
          let pLn2   := basePidx + 5
          let pFc1   := basePidx + 6
          let pFc2   := basePidx + 7
          let heads := r.tbHeads
          let mlpDim := r.tbMlpDim
          let dh := d / heads
          let ty := tensorTy r.inShape
          let mlpTy := tensorTy [b, n, mlpDim]
          let bnTy := tensorTy [b, n]
          let dTy := tensorTy [d]
          let mlpDTy := tensorTy [mlpDim]
          let dF := d.toFloat
          let tag := s!"tb{basePidx}"
          let dy := gradSSA
          code := code ++ s!"    // ─── fc2 backward (dense_has_vjp + dense_weight_grad_correct) ───\n"
          -- fc2 backward
          code := code ++ s!"    %{tag}_dwfc2 = stablehlo.dot_general {r.tbGeluOutSSA}, {dy},\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({mlpTy}, {ty}) -> {tensorTy [mlpDim, d]}\n"
          code := code ++ s!"    %d_W{pFc2} = stablehlo.reshape %{tag}_dwfc2 : ({tensorTy [mlpDim, d]}) -> {tensorTy [mlpDim, d]}\n"
          code := code ++ s!"    %d_b{pFc2} = stablehlo.reduce({dy} init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dge = stablehlo.dot_general {dy}, %W{pFc2},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [mlpDim, d]}) -> {mlpTy}\n"
          code := code ++ s!"    // ─── GELU backward (LayerNorm.lean: pdiv_gelu — tanh-form diagonal) ───\n"
          -- GELU backward
          code := code ++ s!"    %{tag}_t2 = stablehlo.multiply {r.tbGeluTSSA}, {r.tbGeluTSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_one = stablehlo.constant dense<1.0> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_1mt2 = stablehlo.subtract %{tag}_one, %{tag}_t2 : {mlpTy}\n"
          code := code ++ s!"    %{tag}_c134 = stablehlo.constant dense<0.134145> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_xsq = stablehlo.multiply {r.tbFc1OutSSA}, {r.tbFc1OutSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_cx2 = stablehlo.multiply %{tag}_c134, %{tag}_xsq : {mlpTy}\n"
          code := code ++ s!"    %{tag}_idu = stablehlo.add %{tag}_one, %{tag}_cx2 : {mlpTy}\n"
          code := code ++ s!"    %{tag}_csq = stablehlo.constant dense<0.7978845608028654> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_du = stablehlo.multiply %{tag}_idu, %{tag}_csq : {mlpTy}\n"
          code := code ++ s!"    %{tag}_term2a = stablehlo.multiply %{tag}_1mt2, %{tag}_du : {mlpTy}\n"
          code := code ++ s!"    %{tag}_c05 = stablehlo.constant dense<0.5> : {mlpTy}\n"
          code := code ++ s!"    %{tag}_hx = stablehlo.multiply %{tag}_c05, {r.tbFc1OutSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_term2 = stablehlo.multiply %{tag}_hx, %{tag}_term2a : {mlpTy}\n"
          code := code ++ s!"    %{tag}_1pt = stablehlo.add %{tag}_one, {r.tbGeluTSSA} : {mlpTy}\n"
          code := code ++ s!"    %{tag}_term1 = stablehlo.multiply %{tag}_c05, %{tag}_1pt : {mlpTy}\n"
          code := code ++ s!"    %{tag}_dgdx = stablehlo.add %{tag}_term1, %{tag}_term2 : {mlpTy}\n"
          code := code ++ s!"    %{tag}_dfc1 = stablehlo.multiply %{tag}_dge, %{tag}_dgdx : {mlpTy}\n"
          code := code ++ s!"    // ─── fc1 backward (dense_has_vjp + dense_weight_grad_correct) ───\n"
          -- fc1 backward
          code := code ++ s!"    %{tag}_dwfc1 = stablehlo.dot_general {r.tbLn2OutSSA}, %{tag}_dfc1,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {mlpTy}) -> {tensorTy [d, mlpDim]}\n"
          code := code ++ s!"    %d_W{pFc1} = stablehlo.reshape %{tag}_dwfc1 : ({tensorTy [d, mlpDim]}) -> {tensorTy [d, mlpDim]}\n"
          code := code ++ s!"    %d_b{pFc1} = stablehlo.reduce(%{tag}_dfc1 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({mlpTy}, tensor<f32>) -> {mlpDTy}\n"
          code := code ++ s!"    %{tag}_dln2 = stablehlo.dot_general %{tag}_dfc1, %W{pFc1},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({mlpTy}, {tensorTy [d, mlpDim]}) -> {ty}\n"
          code := code ++ s!"    // ─── LN2 backward (LayerNorm.lean: layerNorm_has_vjp — 3-term per-token formula) ───\n"
          -- LN2 backward
          code := code ++ s!"    %{tag}_ln2_gn = stablehlo.multiply %{tag}_dln2, {r.tbLn2NormSSA} : {ty}\n"
          code := code ++ s!"    %d_b{pLn2} = stablehlo.reduce(%{tag}_dln2 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %d_W{pLn2} = stablehlo.reduce(%{tag}_ln2_gn init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_ln2gbc = stablehlo.broadcast_in_dim %W{pLn2}, dims = [2] : ({dTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_dn = stablehlo.multiply %{tag}_dln2, %{tag}_ln2gbc : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_sdn = stablehlo.reduce(%{tag}_ln2_dn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln2_dnn = stablehlo.multiply %{tag}_ln2_dn, {r.tbLn2NormSSA} : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_sdnn = stablehlo.reduce(%{tag}_ln2_dnn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln2_sdn_bc = stablehlo.broadcast_in_dim %{tag}_ln2_sdn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_sdnn_bc = stablehlo.broadcast_in_dim %{tag}_ln2_sdnn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_Nc = stablehlo.constant dense<{dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t1 = stablehlo.multiply %{tag}_ln2_Nc, %{tag}_ln2_dn : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t2 = stablehlo.subtract %{tag}_ln2_t1, %{tag}_ln2_sdn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t3 = stablehlo.multiply {r.tbLn2NormSSA}, %{tag}_ln2_sdnn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_t4 = stablehlo.subtract %{tag}_ln2_t2, %{tag}_ln2_t3 : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_istdbc = stablehlo.broadcast_in_dim {r.tbLn2IstdSSA}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln2_invN = stablehlo.constant dense<{1.0 / dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln2_scale = stablehlo.multiply %{tag}_ln2_istdbc, %{tag}_ln2_invN : {ty}\n"
          code := code ++ s!"    %{tag}_dln2_in = stablehlo.multiply %{tag}_ln2_scale, %{tag}_ln2_t4 : {ty}\n"
          code := code ++ s!"    // ─── MLP-branch residual accumulate (Residual.lean: biPath3_has_vjp) ───\n"
          -- Residual 2 accumulate
          code := code ++ s!"    %{tag}_dr1 = stablehlo.add {dy}, %{tag}_dln2_in : {ty}\n"
          code := code ++ s!"    // ─── MHSA backward (Attention.lean: sdpa_back_Q/K/V_correct + rowSoftmax_has_vjp_mat) ───\n"
          code := code ++ s!"    //     output projection (Wo, bo), then per-head softmax VJP, then Q/K/V projections\n"
          -- MHSA backward
          code := code ++ s!"    %{tag}_dwo = stablehlo.dot_general {r.tbMhPpSSA}, %{tag}_dr1,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWo} = stablehlo.reshape %{tag}_dwo : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWo} = stablehlo.reduce(%{tag}_dr1 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dpp = stablehlo.dot_general %{tag}_dr1, %W{pWo},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          let heTy := tensorTy [b, n, heads, dh]
          let hTy := tensorTy [b, heads, n, dh]
          let sTy := tensorTy [b, heads, n, n]
          code := code ++ s!"    %{tag}_dppr = stablehlo.reshape %{tag}_dpp : ({ty}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dattn = stablehlo.transpose %{tag}_dppr, dims = [0, 2, 1, 3] : ({heTy}) -> {hTy}\n"
          let bhnTy := tensorTy [b, heads, n]
          -- SDPA backward → dq/dk/dv [b,h,n,dh]. Flash: recompute loop (no [b,h,n,n]);
          -- dense: the stored-softmax VJP. Downstream transpose/reshape is shared.
          let mut dqSSA := s!"%{tag}_dq"
          let mut dkSSA := s!"%{tag}_dk"
          let mut dvSSA := s!"%{tag}_dv"
          if r.tbMhFlash then
            let causal := match r.layer with
              | .transformerEncoder _ _ _ _ cm _ _ _ => cm | _ => false
            let (fbCode, fdq, fdk, fdv) := emitFlashAttnBwd s!"{tag}fb" r.tbMhQSSA r.tbMhKSSA r.tbMhVSSA
              s!"%{tag}_dattn" r.tbMhOSSA r.tbMhSmSSA b heads n dh (flashBlock n) causal
            code := code ++ fbCode
            dqSSA := fdq; dkSSA := fdk; dvSSA := fdv
          else
            code := code ++ s!"    %{tag}_dsm = stablehlo.dot_general %{tag}_dattn, {r.tbMhVSSA},\n"
            code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
            code := code ++ s!"              contracting_dims = [3] x [3],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({hTy}, {hTy}) -> {sTy}\n"
            code := code ++ s!"    %{tag}_dv = stablehlo.dot_general {r.tbMhSmSSA}, %{tag}_dattn,\n"
            code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
            code := code ++ s!"              contracting_dims = [2] x [2],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
            code := code ++ s!"    %{tag}_smdsm = stablehlo.multiply {r.tbMhSmSSA}, %{tag}_dsm : {sTy}\n"
            code := code ++ s!"    %{tag}_sumds = stablehlo.reduce(%{tag}_smdsm init: %zf) applies stablehlo.add across dimensions = [3]\n"
            code := code ++ s!"          : ({sTy}, tensor<f32>) -> {bhnTy}\n"
            code := code ++ s!"    %{tag}_sumds_bc = stablehlo.broadcast_in_dim %{tag}_sumds, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
            code := code ++ s!"    %{tag}_dsms = stablehlo.subtract %{tag}_dsm, %{tag}_sumds_bc : {sTy}\n"
            code := code ++ s!"    %{tag}_dscaled = stablehlo.multiply {r.tbMhSmSSA}, %{tag}_dsms : {sTy}\n"
            let invSqrtDh : Float := 1.0 / Float.sqrt dh.toFloat
            code := code ++ s!"    %{tag}_scale = stablehlo.constant dense<{invSqrtDh}> : {sTy}\n"
            code := code ++ s!"    %{tag}_dscores = stablehlo.multiply %{tag}_dscaled, %{tag}_scale : {sTy}\n"
            code := code ++ s!"    %{tag}_dq = stablehlo.dot_general %{tag}_dscores, {r.tbMhKSSA},\n"
            code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
            code := code ++ s!"              contracting_dims = [3] x [2],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
            code := code ++ s!"    %{tag}_dk = stablehlo.dot_general %{tag}_dscores, {r.tbMhQSSA},\n"
            code := code ++ s!"              batching_dims = [0, 1] x [0, 1],\n"
            code := code ++ s!"              contracting_dims = [2] x [2],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({sTy}, {hTy}) -> {hTy}\n"
          -- RoPE backward: dq/dk are w.r.t. the ROPED Q/K; un-rope (rotation by −θ)
          -- to get grads w.r.t. the projections. V (dv) is untouched.
          if r.tbMhRope then
            let (rtCode, cosSSA, sinSSA) := emitRopeTables s!"{tag}rb" n dh
            let (rqCode, rq) := emitRopeApply s!"{tag}rbq" dqSSA cosSSA sinSSA b heads n dh true
            let (rkCode, rk) := emitRopeApply s!"{tag}rbk" dkSSA cosSSA sinSSA b heads n dh true
            code := code ++ rtCode ++ rqCode ++ rkCode
            dqSSA := rq; dkSSA := rk
          code := code ++ s!"    %{tag}_dqt = stablehlo.transpose {dqSSA}, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dqr = stablehlo.reshape %{tag}_dqt : ({heTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dkt = stablehlo.transpose {dkSSA}, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dkr = stablehlo.reshape %{tag}_dkt : ({heTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dvt = stablehlo.transpose {dvSSA}, dims = [0, 2, 1, 3] : ({hTy}) -> {heTy}\n"
          code := code ++ s!"    %{tag}_dvr = stablehlo.reshape %{tag}_dvt : ({heTy}) -> {ty}\n"
          -- QKV projection backwards
          code := code ++ s!"    %{tag}_dwq = stablehlo.dot_general {r.tbLn1OutSSA}, %{tag}_dqr,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWq} = stablehlo.reshape %{tag}_dwq : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWq} = stablehlo.reduce(%{tag}_dqr init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dxq = stablehlo.dot_general %{tag}_dqr, %W{pWq},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dwk = stablehlo.dot_general {r.tbLn1OutSSA}, %{tag}_dkr,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWk} = stablehlo.reshape %{tag}_dwk : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWk} = stablehlo.reduce(%{tag}_dkr init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dxk = stablehlo.dot_general %{tag}_dkr, %W{pWk},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          code := code ++ s!"    %{tag}_dwv = stablehlo.dot_general {r.tbLn1OutSSA}, %{tag}_dvr,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {ty}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_W{pWv} = stablehlo.reshape %{tag}_dwv : ({tensorTy [d, d]}) -> {tensorTy [d, d]}\n"
          code := code ++ s!"    %d_b{pWv} = stablehlo.reduce(%{tag}_dvr init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_dxv = stablehlo.dot_general %{tag}_dvr, %W{pWv},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({ty}, {tensorTy [d, d]}) -> {ty}\n"
          code := code ++ s!"    // ─── Q/K/V backward fan-in into LN1 input (biPath3_has_vjp: three paths sum) ───\n"
          code := code ++ s!"    %{tag}_dln1a = stablehlo.add %{tag}_dxq, %{tag}_dxk : {ty}\n"
          code := code ++ s!"    %{tag}_dln1 = stablehlo.add %{tag}_dln1a, %{tag}_dxv : {ty}\n"
          code := code ++ s!"    // ─── LN1 backward (LayerNorm.lean: layerNorm_has_vjp) ───\n"
          -- LN1 backward
          code := code ++ s!"    %{tag}_ln1_gn = stablehlo.multiply %{tag}_dln1, {r.tbLn1NormSSA} : {ty}\n"
          code := code ++ s!"    %d_b{pLn1} = stablehlo.reduce(%{tag}_dln1 init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %d_W{pLn1} = stablehlo.reduce(%{tag}_ln1_gn init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {dTy}\n"
          code := code ++ s!"    %{tag}_ln1gbc = stablehlo.broadcast_in_dim %W{pLn1}, dims = [2] : ({dTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_dn = stablehlo.multiply %{tag}_dln1, %{tag}_ln1gbc : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_sdn = stablehlo.reduce(%{tag}_ln1_dn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln1_dnn = stablehlo.multiply %{tag}_ln1_dn, {r.tbLn1NormSSA} : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_sdnn = stablehlo.reduce(%{tag}_ln1_dnn init: %zf) applies stablehlo.add across dimensions = [2]\n"
          code := code ++ s!"          : ({ty}, tensor<f32>) -> {bnTy}\n"
          code := code ++ s!"    %{tag}_ln1_sdn_bc = stablehlo.broadcast_in_dim %{tag}_ln1_sdn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_sdnn_bc = stablehlo.broadcast_in_dim %{tag}_ln1_sdnn, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_Nc = stablehlo.constant dense<{dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t1 = stablehlo.multiply %{tag}_ln1_Nc, %{tag}_ln1_dn : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t2 = stablehlo.subtract %{tag}_ln1_t1, %{tag}_ln1_sdn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t3 = stablehlo.multiply {r.tbLn1NormSSA}, %{tag}_ln1_sdnn_bc : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_t4 = stablehlo.subtract %{tag}_ln1_t2, %{tag}_ln1_t3 : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_istdbc = stablehlo.broadcast_in_dim {r.tbLn1IstdSSA}, dims = [0, 1] : ({bnTy}) -> {ty}\n"
          code := code ++ s!"    %{tag}_ln1_invN = stablehlo.constant dense<{1.0 / dF}> : {ty}\n"
          code := code ++ s!"    %{tag}_ln1_scale = stablehlo.multiply %{tag}_ln1_istdbc, %{tag}_ln1_invN : {ty}\n"
          code := code ++ s!"    %{tag}_dln1_in = stablehlo.multiply %{tag}_ln1_scale, %{tag}_ln1_t4 : {ty}\n"
          code := code ++ s!"    // ─── MHSA-branch residual accumulate → block input gradient (biPath3_has_vjp) ───\n"
          code := code ++ s!"    %{tag}_dblockin = stablehlo.add %{tag}_dr1, %{tag}_dln1_in : {ty}\n"
          gradSSA := s!"%{tag}_dblockin"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()

    | .convNextStage _channels _nBlocks _norm _act =>
      if r.isConvNextBlock then
        match r.inShape with
        | [b, _, h, w] =>
          code := code ++ s!"    // ════════════════════════════════════════════════════════════════\n"
          code := code ++ s!"    // ConvNeXt block backward — see LeanMlir/Proofs/:\n"
          code := code ++ s!"    //   Residual fan-in            Residual.lean: residual_has_vjp\n"
          code := code ++ s!"    //   LayerScale (per-channel γ) Pointwise.lean: elemwiseProduct_has_vjp\n"
          code := code ++ s!"    //   1×1 project / expand convs CNN.lean: conv2d_has_vjp3 / conv2d_weight_grad_has_vjp\n"
          code := code ++ s!"    //   GELU                       LayerNorm.lean: pdiv_gelu (tanh-form diagonal)\n"
          code := code ++ s!"    //   LN over channel axis       LayerNorm.lean: layerNorm_has_vjp (axis-relabeled)\n"
          code := code ++ s!"    //   Depthwise 7×7 raw          Depthwise.lean: depthwise_has_vjp3 + depthwise_weight_grad_has_vjp3\n"
          code := code ++ s!"    // ════════════════════════════════════════════════════════════════\n"
          let basePidx := r.cnbBasePidx
          let pDw := basePidx
          let pLn := basePidx + 1
          let pEx := basePidx + 2
          let pPj := basePidx + 3
          let pLs := basePidx + 4
          let c := r.cnbChannels
          let act := r.cnbAct
          let blockTy := tensorTy r.inShape
          let cTy := tensorTy [c]
          let _bhwTy := tensorTy [b, h, w]
          let expShape := [b, 4*c, h, w]
          let expTy := tensorTy expShape
          let exB := tensorTy [4*c]
          let cF := c.toFloat
          let tag := s!"cnb{basePidx}"
          let dy := gradSSA
          -- 1) Residual fan-in: d_LSout = dy. d_blockIn_skip = dy (added at end).
          -- 2) LayerScale backward: d_γ = sum_{b,h,w}(projectOut · dy); d_projectOut = γ_bc ⊙ dy.
          code := code ++ s!"    %{tag}_lsgn = stablehlo.multiply {r.cnbProjectOutSSA}, {dy} : {blockTy}\n"
          code := code ++ s!"    %d_W{pLs} = stablehlo.reduce(%{tag}_lsgn init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
          code := code ++ s!"          : ({blockTy}, tensor<f32>) -> {cTy}\n"
          code := code ++ s!"    %{tag}_lsgbc = stablehlo.broadcast_in_dim %W{pLs}, dims = [1] : ({cTy}) -> {blockTy}\n"
          code := code ++ s!"    %{tag}_dpj = stablehlo.multiply {dy}, %{tag}_lsgbc : {blockTy}\n"
          -- 3) Project conv backward (1×1, stride 1, ic=4c, oc=c).
          code := code ++ s!"    %{tag}_pjbtin = stablehlo.transpose {r.cnbActOutSSA}, dims = [1, 0, 2, 3] : ({expTy}) -> {tensorTy [4*c, b, h, w]}\n"
          code := code ++ s!"    %{tag}_pjbtg = stablehlo.transpose %{tag}_dpj, dims = [1, 0, 2, 3] : ({blockTy}) -> {tensorTy [c, b, h, w]}\n"
          code := code ++ s!"    %{tag}_pjdWr = \"stablehlo.convolution\"(%{tag}_pjbtin, %{tag}_pjbtg) " ++ "{\n"
          code := code ++ convAttrBlock 0
          code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [4*c, b, h, w]}, {tensorTy [c, b, h, w]}) -> {tensorTy [4*c, c, 1, 1]}\n"
          code := code ++ s!"    %d_W{pPj} = stablehlo.transpose %{tag}_pjdWr, dims = [1, 0, 2, 3] : ({tensorTy [4*c, c, 1, 1]}) -> {tensorTy [c, 4*c, 1, 1]}\n"
          code := code ++ s!"    %d_b{pPj} = stablehlo.reduce(%{tag}_dpj init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
          code := code ++ s!"          : ({blockTy}, tensor<f32>) -> {cTy}\n"
          code := code ++ s!"    %{tag}_pjWt = stablehlo.transpose %W{pPj}, dims = [1, 0, 2, 3] : ({tensorTy [c, 4*c, 1, 1]}) -> {tensorTy [4*c, c, 1, 1]}\n"
          code := code ++ s!"    %{tag}_pjWrev = stablehlo.reverse %{tag}_pjWt, dims = [2, 3] : {tensorTy [4*c, c, 1, 1]}\n"
          code := code ++ s!"    %{tag}_dact = \"stablehlo.convolution\"(%{tag}_dpj, %{tag}_pjWrev) " ++ "{\n"
          code := code ++ convAttrBlock 0
          code := code ++ s!"      " ++ "}" ++ s!" : ({blockTy}, {tensorTy [4*c, c, 1, 1]}) -> {expTy}\n"
          -- 4) Activation backward: dy = %{tag}_dact, output = %{tag}_dexp.
          let mut dExpandSSA := s!"%{tag}_dact"
          match act with
          | .gelu =>
            code := code ++ s!"    // ─── GELU backward (tanh form) ───\n"
            code := code ++ s!"    %{tag}_t2 = stablehlo.multiply {r.cnbActTanhSSA}, {r.cnbActTanhSSA} : {expTy}\n"
            code := code ++ s!"    %{tag}_one = stablehlo.constant dense<1.0> : {expTy}\n"
            code := code ++ s!"    %{tag}_1mt2 = stablehlo.subtract %{tag}_one, %{tag}_t2 : {expTy}\n"
            code := code ++ s!"    %{tag}_c134 = stablehlo.constant dense<0.134145> : {expTy}\n"
            code := code ++ s!"    %{tag}_xsq = stablehlo.multiply {r.cnbExpandOutSSA}, {r.cnbExpandOutSSA} : {expTy}\n"
            code := code ++ s!"    %{tag}_cx2 = stablehlo.multiply %{tag}_c134, %{tag}_xsq : {expTy}\n"
            code := code ++ s!"    %{tag}_idu = stablehlo.add %{tag}_one, %{tag}_cx2 : {expTy}\n"
            code := code ++ s!"    %{tag}_csq = stablehlo.constant dense<0.7978845608028654> : {expTy}\n"
            code := code ++ s!"    %{tag}_du = stablehlo.multiply %{tag}_idu, %{tag}_csq : {expTy}\n"
            code := code ++ s!"    %{tag}_term2a = stablehlo.multiply %{tag}_1mt2, %{tag}_du : {expTy}\n"
            code := code ++ s!"    %{tag}_c05 = stablehlo.constant dense<0.5> : {expTy}\n"
            code := code ++ s!"    %{tag}_hx = stablehlo.multiply %{tag}_c05, {r.cnbExpandOutSSA} : {expTy}\n"
            code := code ++ s!"    %{tag}_term2 = stablehlo.multiply %{tag}_hx, %{tag}_term2a : {expTy}\n"
            code := code ++ s!"    %{tag}_1pt = stablehlo.add %{tag}_one, {r.cnbActTanhSSA} : {expTy}\n"
            code := code ++ s!"    %{tag}_term1 = stablehlo.multiply %{tag}_c05, %{tag}_1pt : {expTy}\n"
            code := code ++ s!"    %{tag}_dgdx = stablehlo.add %{tag}_term1, %{tag}_term2 : {expTy}\n"
            code := code ++ s!"    %{tag}_dexp = stablehlo.multiply %{tag}_dact, %{tag}_dgdx : {expTy}\n"
            dExpandSSA := s!"%{tag}_dexp"
          | .relu =>
            code := code ++ s!"    %{tag}_zero = stablehlo.constant dense<0.0> : {expTy}\n"
            let i1Ty := expTy.replace "xf32>" "xi1>"
            code := code ++ s!"    %{tag}_rmask = stablehlo.compare GT, {r.cnbExpandOutSSA}, %{tag}_zero : ({expTy}, {expTy}) -> {i1Ty}\n"
            code := code ++ s!"    %{tag}_dexp = stablehlo.select %{tag}_rmask, %{tag}_dact, %{tag}_zero : {i1Ty}, {expTy}\n"
            dExpandSSA := s!"%{tag}_dexp"
          | _ =>
            dExpandSSA := s!"%{tag}_dact"
          -- 5) Expand conv backward (1×1, stride 1, ic=c, oc=4c).
          code := code ++ s!"    %{tag}_exbtin = stablehlo.transpose {r.cnbLnOutSSA}, dims = [1, 0, 2, 3] : ({blockTy}) -> {tensorTy [c, b, h, w]}\n"
          code := code ++ s!"    %{tag}_exbtg = stablehlo.transpose {dExpandSSA}, dims = [1, 0, 2, 3] : ({expTy}) -> {tensorTy [4*c, b, h, w]}\n"
          code := code ++ s!"    %{tag}_exdWr = \"stablehlo.convolution\"(%{tag}_exbtin, %{tag}_exbtg) " ++ "{\n"
          code := code ++ convAttrBlock 0
          code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [c, b, h, w]}, {tensorTy [4*c, b, h, w]}) -> {tensorTy [c, 4*c, 1, 1]}\n"
          code := code ++ s!"    %d_W{pEx} = stablehlo.transpose %{tag}_exdWr, dims = [1, 0, 2, 3] : ({tensorTy [c, 4*c, 1, 1]}) -> {tensorTy [4*c, c, 1, 1]}\n"
          code := code ++ s!"    %d_b{pEx} = stablehlo.reduce({dExpandSSA} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
          code := code ++ s!"          : ({expTy}, tensor<f32>) -> {exB}\n"
          code := code ++ s!"    %{tag}_exWt = stablehlo.transpose %W{pEx}, dims = [1, 0, 2, 3] : ({tensorTy [4*c, c, 1, 1]}) -> {tensorTy [c, 4*c, 1, 1]}\n"
          code := code ++ s!"    %{tag}_exWrev = stablehlo.reverse %{tag}_exWt, dims = [2, 3] : {tensorTy [c, 4*c, 1, 1]}\n"
          code := code ++ s!"    %{tag}_dlnout = \"stablehlo.convolution\"({dExpandSSA}, %{tag}_exWrev) " ++ "{\n"
          code := code ++ convAttrBlock 0
          code := code ++ s!"      " ++ "}" ++ s!" : ({expTy}, {tensorTy [c, 4*c, 1, 1]}) -> {blockTy}\n"
          -- 6) Norm backward: branch on `cnbNorm`.
          match r.cnbNorm with
          | .ln =>
            -- Three-term LN over inner axis after transpose. Saved norm/istd
            -- are in [b, hw, c] / [b, hw]; transpose dy to NHWC, reshape, do
            -- the math on the inner axis (IREE-friendly), reshape+transpose
            -- the result back to NCHW for the DW backward.
            let nhwcTy := tensorTy [b, h, w, c]
            let nhcTy  := tensorTy [b, h*w, c]
            let bnTy   := tensorTy [b, h*w]
            code := code ++ s!"    %{tag}_dyt = stablehlo.transpose %{tag}_dlnout, dims = [0, 2, 3, 1] : ({blockTy}) -> {nhwcTy}\n"
            code := code ++ s!"    %{tag}_dyi = stablehlo.reshape %{tag}_dyt : ({nhwcTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lngn = stablehlo.multiply %{tag}_dyi, {r.cnbLnNormSSA} : {nhcTy}\n"
            code := code ++ s!"    %d_b{pLn} = stablehlo.reduce(%{tag}_dyi init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {cTy}\n"
            code := code ++ s!"    %d_W{pLn} = stablehlo.reduce(%{tag}_lngn init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {cTy}\n"
            code := code ++ s!"    %{tag}_lngbc = stablehlo.broadcast_in_dim %W{pLn}, dims = [2] : ({cTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lndn = stablehlo.multiply %{tag}_dyi, %{tag}_lngbc : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnsdn = stablehlo.reduce(%{tag}_lndn init: %zf) applies stablehlo.add across dimensions = [2]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {bnTy}\n"
            code := code ++ s!"    %{tag}_lndnn = stablehlo.multiply %{tag}_lndn, {r.cnbLnNormSSA} : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnsdnn = stablehlo.reduce(%{tag}_lndnn init: %zf) applies stablehlo.add across dimensions = [2]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {bnTy}\n"
            code := code ++ s!"    %{tag}_lnsdnbc = stablehlo.broadcast_in_dim %{tag}_lnsdn, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnsdnnbc = stablehlo.broadcast_in_dim %{tag}_lnsdnn, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnNc = stablehlo.constant dense<{cF}> : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt1 = stablehlo.multiply %{tag}_lnNc, %{tag}_lndn : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt2 = stablehlo.subtract %{tag}_lnt1, %{tag}_lnsdnbc : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt3 = stablehlo.multiply {r.cnbLnNormSSA}, %{tag}_lnsdnnbc : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt4 = stablehlo.subtract %{tag}_lnt2, %{tag}_lnt3 : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnistdbc = stablehlo.broadcast_in_dim {r.cnbLnIstdSSA}, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lninvN = stablehlo.constant dense<{1.0 / cF}> : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnscale = stablehlo.multiply %{tag}_lnistdbc, %{tag}_lninvN : {nhcTy}\n"
            code := code ++ s!"    %{tag}_ddwi = stablehlo.multiply %{tag}_lnscale, %{tag}_lnt4 : {nhcTy}\n"
            code := code ++ s!"    %{tag}_ddwr = stablehlo.reshape %{tag}_ddwi : ({nhcTy}) -> {nhwcTy}\n"
            code := code ++ s!"    %{tag}_ddwout = stablehlo.transpose %{tag}_ddwr, dims = [0, 3, 1, 2] : ({nhwcTy}) -> {blockTy}\n"
          | .bn =>
            -- BN-NCHW three-term backward. Reductions split into 2-step
            -- `[2, 3]` then `[0]` (mirroring `convBn`'s pattern) so the
            -- four BN reductions in this block don't fuse into a single
            -- IREE dispatch shape that fails to distribute on ROCm.
            let bnN := b * h * w
            let bnNF := bnN.toFloat
            let bcTy := tensorTy [b, c]
            code := code ++ s!"    %{tag}_bgn = stablehlo.multiply %{tag}_dlnout, {r.cnbBnNormSSA} : {blockTy}\n"
            code := code ++ s!"    %{tag}_dbsp = stablehlo.reduce(%{tag}_dlnout init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({blockTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %d_b{pLn} = stablehlo.reduce(%{tag}_dbsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {cTy}\n"
            code := code ++ s!"    %{tag}_dWsp = stablehlo.reduce(%{tag}_bgn init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({blockTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %d_W{pLn} = stablehlo.reduce(%{tag}_dWsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {cTy}\n"
            code := code ++ s!"    %{tag}_bgbc = stablehlo.broadcast_in_dim %W{pLn}, dims = [1] : ({cTy}) -> {blockTy}\n"
            code := code ++ s!"    %{tag}_bdn = stablehlo.multiply %{tag}_dlnout, %{tag}_bgbc : {blockTy}\n"
            code := code ++ s!"    %{tag}_bsdnsp = stablehlo.reduce(%{tag}_bdn init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({blockTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %{tag}_bsdn = stablehlo.reduce(%{tag}_bsdnsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {cTy}\n"
            code := code ++ s!"    %{tag}_bdnn = stablehlo.multiply %{tag}_bdn, {r.cnbBnNormSSA} : {blockTy}\n"
            code := code ++ s!"    %{tag}_bsdnnsp = stablehlo.reduce(%{tag}_bdnn init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({blockTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %{tag}_bsdnn = stablehlo.reduce(%{tag}_bsdnnsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {cTy}\n"
            code := code ++ s!"    %{tag}_bsdnbc = stablehlo.broadcast_in_dim %{tag}_bsdn, dims = [1] : ({cTy}) -> {blockTy}\n"
            code := code ++ s!"    %{tag}_bsdnnbc = stablehlo.broadcast_in_dim %{tag}_bsdnn, dims = [1] : ({cTy}) -> {blockTy}\n"
            code := code ++ s!"    %{tag}_bNc = stablehlo.constant dense<{bnNF}> : {blockTy}\n"
            code := code ++ s!"    %{tag}_bt1 = stablehlo.multiply %{tag}_bNc, %{tag}_bdn : {blockTy}\n"
            code := code ++ s!"    %{tag}_bt2 = stablehlo.subtract %{tag}_bt1, %{tag}_bsdnbc : {blockTy}\n"
            code := code ++ s!"    %{tag}_bt3 = stablehlo.multiply {r.cnbBnNormSSA}, %{tag}_bsdnnbc : {blockTy}\n"
            code := code ++ s!"    %{tag}_bt4 = stablehlo.subtract %{tag}_bt2, %{tag}_bt3 : {blockTy}\n"
            code := code ++ s!"    %{tag}_binvN = stablehlo.constant dense<{1.0 / bnNF}> : {blockTy}\n"
            code := code ++ s!"    %{tag}_bscale = stablehlo.multiply {r.cnbBnIstdBcSSA}, %{tag}_binvN : {blockTy}\n"
            code := code ++ s!"    %{tag}_ddwout = stablehlo.multiply %{tag}_bscale, %{tag}_bt4 : {blockTy}\n"
          -- 7) DW raw 7×7 stride-1 same-pad backward.
          let pad : Nat := 3
          code := code ++ s!"    %d_b{pDw} = stablehlo.reduce(%{tag}_ddwout init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
          code := code ++ s!"          : ({blockTy}, tensor<f32>) -> {cTy}\n"
          code := code ++ s!"    %{tag}_dwbtin = stablehlo.transpose {r.cnbBlockInSSA}, dims = [1, 0, 2, 3] : ({blockTy}) -> {tensorTy [c, b, h, w]}\n"
          code := code ++ s!"    %{tag}_dwbtg = stablehlo.transpose %{tag}_ddwout, dims = [1, 0, 2, 3] : ({blockTy}) -> {tensorTy [c, b, h, w]}\n"
          code := code ++ s!"    %{tag}_dwdWr = \"stablehlo.convolution\"(%{tag}_dwbtin, %{tag}_dwbtg) " ++ "{\n"
          code := code ++ s!"        batch_group_count = {c} : i64,\n"
          code := code ++ convDimNumbers
          code := code ++ "        feature_group_count = 1 : i64,\n"
          code := code ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
          code := code ++ "        rhs_dilation = array<i64: 1, 1>,\n"
          code := code ++ "        window_strides = array<i64: 1, 1>\n"
          code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [c, b, h, w]}, {tensorTy [c, b, h, w]}) -> {tensorTy [1, c, 7, 7]}\n"
          code := code ++ s!"    %d_W{pDw} = stablehlo.transpose %{tag}_dwdWr, dims = [1, 0, 2, 3] : ({tensorTy [1, c, 7, 7]}) -> {tensorTy [c, 1, 7, 7]}\n"
          code := code ++ s!"    %{tag}_dwWrev = stablehlo.reverse %W{pDw}, dims = [2, 3] : {tensorTy [c, 1, 7, 7]}\n"
          code := code ++ s!"    %{tag}_dxblk = \"stablehlo.convolution\"(%{tag}_ddwout, %{tag}_dwWrev) " ++ "{\n"
          code := code ++ "        batch_group_count = 1 : i64,\n"
          code := code ++ convDimNumbers
          code := code ++ s!"        feature_group_count = {c} : i64,\n"
          code := code ++ s!"        padding = dense<[[{pad}, {pad}], [{pad}, {pad}]]> : tensor<2x2xi64>,\n"
          code := code ++ "        rhs_dilation = array<i64: 1, 1>,\n"
          code := code ++ "        window_strides = array<i64: 1, 1>\n"
          code := code ++ s!"      " ++ "}" ++ s!" : ({blockTy}, {tensorTy [c, 1, 7, 7]}) -> {blockTy}\n"
          -- 8) Residual fan-in: dx_block = dy + dxblk
          code := code ++ s!"    %{tag}_dx = stablehlo.add {dy}, %{tag}_dxblk : {blockTy}\n"
          gradSSA := s!"%{tag}_dx"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()

    | .convNextDownsample _ic _oc _norm =>
      if r.isConvNextDs then
        match r.inShape with
        | [b, _, h, w] =>
          code := code ++ s!"    // ─── ConvNeXt downsample backward (LN over channels + 2×2 stride-2 conv) ───\n"
          let basePidx := r.cndBasePidx
          let pLn := basePidx
          let pCv := basePidx + 1
          let ic := r.cndIc
          let oc := r.cndOc
          let outShape := r.outShape
          let oH := outShape[2]!
          let oW := outShape[3]!
          let inTy := tensorTy r.inShape
          let outTy := tensorTy outShape
          let icTy := tensorTy [ic]
          let ocTy := tensorTy [oc]
          let _bhwTy := tensorTy [b, h, w]
          let icF := ic.toFloat
          let tag := s!"cnds{basePidx}"
          let dy := gradSSA
          -- 1) 2×2 stride-2 conv backward (input shape [b, ic, h, w], output [b, oc, h/2, w/2]).
          code := code ++ s!"    %{tag}_btin = stablehlo.transpose {r.cndLnOutSSA}, dims = [1, 0, 2, 3] : ({inTy}) -> {tensorTy [ic, b, h, w]}\n"
          code := code ++ s!"    %{tag}_btg = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({outTy}) -> {tensorTy [oc, b, oH, oW]}\n"
          code := code ++ s!"    %{tag}_dWr = \"stablehlo.convolution\"(%{tag}_btin, %{tag}_btg) " ++ "{\n"
          code := code ++ "        batch_group_count = 1 : i64,\n"
          code := code ++ convDimNumbers
          code := code ++ "        feature_group_count = 1 : i64,\n"
          code := code ++ "        lhs_dilation = array<i64: 1, 1>,\n"
          code := code ++ "        padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,\n"
          code := code ++ "        rhs_dilation = array<i64: 2, 2>,\n"
          code := code ++ "        window_strides = array<i64: 1, 1>\n"
          code := code ++ s!"      " ++ "}" ++ s!" : ({tensorTy [ic, b, h, w]}, {tensorTy [oc, b, oH, oW]}) -> {tensorTy [ic, oc, 2, 2]}\n"
          code := code ++ s!"    %d_W{pCv} = stablehlo.transpose %{tag}_dWr, dims = [1, 0, 2, 3] : ({tensorTy [ic, oc, 2, 2]}) -> {tensorTy [oc, ic, 2, 2]}\n"
          code := code ++ s!"    %d_b{pCv} = stablehlo.reduce({dy} init: %zf) applies stablehlo.add across dimensions = [0, 2, 3]\n"
          code := code ++ s!"          : ({outTy}, tensor<f32>) -> {ocTy}\n"
          code := code ++ s!"    %{tag}_Wt = stablehlo.transpose %W{pCv}, dims = [1, 0, 2, 3] : ({tensorTy [oc, ic, 2, 2]}) -> {tensorTy [ic, oc, 2, 2]}\n"
          code := code ++ s!"    %{tag}_Wrev = stablehlo.reverse %{tag}_Wt, dims = [2, 3] : {tensorTy [ic, oc, 2, 2]}\n"
          let dxPad : Nat := 1
          code := code ++ s!"    %{tag}_dlnout = \"stablehlo.convolution\"({dy}, %{tag}_Wrev) " ++ "{\n"
          code := code ++ "        batch_group_count = 1 : i64,\n"
          code := code ++ convDimNumbers
          code := code ++ "        feature_group_count = 1 : i64,\n"
          code := code ++ "        lhs_dilation = array<i64: 2, 2>,\n"
          code := code ++ s!"        padding = dense<[[{dxPad}, {dxPad}], [{dxPad}, {dxPad}]]> : tensor<2x2xi64>,\n"
          code := code ++ "        rhs_dilation = array<i64: 1, 1>,\n"
          code := code ++ "        window_strides = array<i64: 1, 1>\n"
          code := code ++ s!"      " ++ "}" ++ s!" : ({outTy}, {tensorTy [ic, oc, 2, 2]}) -> {inTy}\n"
          -- 2) Norm backward (channels = ic): branch on `cndNorm`.
          match r.cndNorm with
          | .ln =>
            -- Three-term LN over inner axis after transpose.
            let nhwcTy := tensorTy [b, h, w, ic]
            let nhcTy  := tensorTy [b, h*w, ic]
            let bnTy   := tensorTy [b, h*w]
            code := code ++ s!"    %{tag}_dyt = stablehlo.transpose %{tag}_dlnout, dims = [0, 2, 3, 1] : ({inTy}) -> {nhwcTy}\n"
            code := code ++ s!"    %{tag}_dyi = stablehlo.reshape %{tag}_dyt : ({nhwcTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lngn = stablehlo.multiply %{tag}_dyi, {r.cndLnNormSSA} : {nhcTy}\n"
            code := code ++ s!"    %d_b{pLn} = stablehlo.reduce(%{tag}_dyi init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {icTy}\n"
            code := code ++ s!"    %d_W{pLn} = stablehlo.reduce(%{tag}_lngn init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {icTy}\n"
            code := code ++ s!"    %{tag}_lngbc = stablehlo.broadcast_in_dim %W{pLn}, dims = [2] : ({icTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lndn = stablehlo.multiply %{tag}_dyi, %{tag}_lngbc : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnsdn = stablehlo.reduce(%{tag}_lndn init: %zf) applies stablehlo.add across dimensions = [2]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {bnTy}\n"
            code := code ++ s!"    %{tag}_lndnn = stablehlo.multiply %{tag}_lndn, {r.cndLnNormSSA} : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnsdnn = stablehlo.reduce(%{tag}_lndnn init: %zf) applies stablehlo.add across dimensions = [2]\n"
            code := code ++ s!"          : ({nhcTy}, tensor<f32>) -> {bnTy}\n"
            code := code ++ s!"    %{tag}_lnsdnbc = stablehlo.broadcast_in_dim %{tag}_lnsdn, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnsdnnbc = stablehlo.broadcast_in_dim %{tag}_lnsdnn, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnNc = stablehlo.constant dense<{icF}> : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt1 = stablehlo.multiply %{tag}_lnNc, %{tag}_lndn : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt2 = stablehlo.subtract %{tag}_lnt1, %{tag}_lnsdnbc : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt3 = stablehlo.multiply {r.cndLnNormSSA}, %{tag}_lnsdnnbc : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnt4 = stablehlo.subtract %{tag}_lnt2, %{tag}_lnt3 : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnistdbc = stablehlo.broadcast_in_dim {r.cndLnIstdSSA}, dims = [0, 1] : ({bnTy}) -> {nhcTy}\n"
            code := code ++ s!"    %{tag}_lninvN = stablehlo.constant dense<{1.0 / icF}> : {nhcTy}\n"
            code := code ++ s!"    %{tag}_lnscale = stablehlo.multiply %{tag}_lnistdbc, %{tag}_lninvN : {nhcTy}\n"
            code := code ++ s!"    %{tag}_dxi = stablehlo.multiply %{tag}_lnscale, %{tag}_lnt4 : {nhcTy}\n"
            code := code ++ s!"    %{tag}_dxr = stablehlo.reshape %{tag}_dxi : ({nhcTy}) -> {nhwcTy}\n"
            code := code ++ s!"    %{tag}_dx = stablehlo.transpose %{tag}_dxr, dims = [0, 3, 1, 2] : ({nhwcTy}) -> {inTy}\n"
          | .bn =>
            -- BN three-term backward — 2-step reductions `[2, 3]` then `[0]`
            -- to dodge IREE's distribute issue with stacked `[0, 2, 3]` reductions.
            let bnN := b * h * w
            let bnNF := bnN.toFloat
            let bcTy := tensorTy [b, ic]
            code := code ++ s!"    %{tag}_bgn = stablehlo.multiply %{tag}_dlnout, {r.cndBnNormSSA} : {inTy}\n"
            code := code ++ s!"    %{tag}_dbsp = stablehlo.reduce(%{tag}_dlnout init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({inTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %d_b{pLn} = stablehlo.reduce(%{tag}_dbsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {icTy}\n"
            code := code ++ s!"    %{tag}_dWsp = stablehlo.reduce(%{tag}_bgn init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({inTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %d_W{pLn} = stablehlo.reduce(%{tag}_dWsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {icTy}\n"
            code := code ++ s!"    %{tag}_bgbc = stablehlo.broadcast_in_dim %W{pLn}, dims = [1] : ({icTy}) -> {inTy}\n"
            code := code ++ s!"    %{tag}_bdn = stablehlo.multiply %{tag}_dlnout, %{tag}_bgbc : {inTy}\n"
            code := code ++ s!"    %{tag}_bsdnsp = stablehlo.reduce(%{tag}_bdn init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({inTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %{tag}_bsdn = stablehlo.reduce(%{tag}_bsdnsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {icTy}\n"
            code := code ++ s!"    %{tag}_bdnn = stablehlo.multiply %{tag}_bdn, {r.cndBnNormSSA} : {inTy}\n"
            code := code ++ s!"    %{tag}_bsdnnsp = stablehlo.reduce(%{tag}_bdnn init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
            code := code ++ s!"          : ({inTy}, tensor<f32>) -> {bcTy}\n"
            code := code ++ s!"    %{tag}_bsdnn = stablehlo.reduce(%{tag}_bsdnnsp init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({bcTy}, tensor<f32>) -> {icTy}\n"
            code := code ++ s!"    %{tag}_bsdnbc = stablehlo.broadcast_in_dim %{tag}_bsdn, dims = [1] : ({icTy}) -> {inTy}\n"
            code := code ++ s!"    %{tag}_bsdnnbc = stablehlo.broadcast_in_dim %{tag}_bsdnn, dims = [1] : ({icTy}) -> {inTy}\n"
            code := code ++ s!"    %{tag}_bNc = stablehlo.constant dense<{bnNF}> : {inTy}\n"
            code := code ++ s!"    %{tag}_bt1 = stablehlo.multiply %{tag}_bNc, %{tag}_bdn : {inTy}\n"
            code := code ++ s!"    %{tag}_bt2 = stablehlo.subtract %{tag}_bt1, %{tag}_bsdnbc : {inTy}\n"
            code := code ++ s!"    %{tag}_bt3 = stablehlo.multiply {r.cndBnNormSSA}, %{tag}_bsdnnbc : {inTy}\n"
            code := code ++ s!"    %{tag}_bt4 = stablehlo.subtract %{tag}_bt2, %{tag}_bt3 : {inTy}\n"
            code := code ++ s!"    %{tag}_binvN = stablehlo.constant dense<{1.0 / bnNF}> : {inTy}\n"
            code := code ++ s!"    %{tag}_bscale = stablehlo.multiply {r.cndBnIstdBcSSA}, %{tag}_binvN : {inTy}\n"
            code := code ++ s!"    %{tag}_dx = stablehlo.multiply %{tag}_bscale, %{tag}_bt4 : {inTy}\n"
          gradSSA := s!"%{tag}_dx"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()

    | .lmHead d v t =>
      if r.isLmHead then
        match r.inShape with
        | [b, _, _] =>
          let pIdx := r.lmhPidx
          let btv := [b, t, v]
          let bvt := [b, v, t]
          let bvt1 := [b, v, t, 1]
          let inBtdTy := tensorTy r.inShape
          let btvTy := tensorTy btv
          let bvtTy := tensorTy bvt
          let bvt1Ty := tensorTy bvt1
          let tag := s!"lmh{r.pos}"
          code := code ++ s!"    // ─── lmHead backward: reshape -> transpose -> dense3D backward ───\n"
          -- Undo reshape [B, V, T, 1] -> [B, V, T]
          code := code ++ s!"    %{tag}_g3 = stablehlo.reshape {gradSSA} : ({bvt1Ty}) -> {bvtTy}\n"
          -- Undo transpose [B, V, T] -> [B, T, V]
          code := code ++ s!"    %{tag}_gbtv = stablehlo.transpose %{tag}_g3, dims = [0, 2, 1] : ({bvtTy}) -> {btvTy}\n"
          -- d_b = sum d_btv over (B, T) -> [V]
          code := code ++ s!"    %d_b{pIdx} = stablehlo.reduce(%{tag}_gbtv init: %zf) applies stablehlo.add across dimensions = [0, 1]\n"
          code := code ++ s!"          : ({btvTy}, tensor<f32>) -> {tensorTy [v]}\n"
          -- d_W = input^T @ d_btv (contract over batch+time): [D, V]
          code := code ++ s!"    %d_W{pIdx} = stablehlo.dot_general {r.inputSSA}, %{tag}_gbtv,\n"
          code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({inBtdTy}, {btvTy}) -> {tensorTy [d, v]}\n"
          -- d_input = d_btv @ W^T -> [B, T, D]
          code := code ++ s!"    %{tag}_din = stablehlo.dot_general %{tag}_gbtv, %W{pIdx},\n"
          code := code ++ s!"              contracting_dims = [2] x [1],\n"
          code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
          code := code ++ s!"            : ({btvTy}, {tensorTy [d, v]}) -> {inBtdTy}\n"
          gradSSA := s!"%{tag}_din"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()

    | .timeCondAdd c nFreq =>
      -- Residual add ⇒ the gradient passes through to the pre-add feature
      -- UNCHANGED (d_feat = d_out); we only bolt on the dense param grads.
      match r.pidx, r.inShape with
      | some pIdx, [b, _, h, w] =>
        let embDim := 2 * nFreq
        let bchw := tensorTy [b, c, h, w]
        let bc := tensorTy [b, c]
        let tag := s!"tc{r.pos}"
        code := code ++ s!"    // ─── timeCondAdd backward: d_proj = sum_HW d_out; d_W = embᵀ·d_proj ───\n"
        -- d_proj [b, c] = reduce d_out over spatial dims [2, 3]
        code := code ++ s!"    %{tag}_dproj = stablehlo.reduce({gradSSA} init: %zf) applies stablehlo.add across dimensions = [2, 3]\n"
        code := code ++ s!"          : ({bchw}, tensor<f32>) -> {bc}\n"
        -- d_b [c] = sum d_proj over batch
        code := code ++ s!"    %d_b{pIdx} = stablehlo.reduce(%{tag}_dproj init: %zf) applies stablehlo.add across dimensions = [0]\n"
        code := code ++ s!"          : ({bc}, tensor<f32>) -> {tensorTy [c]}\n"
        -- d_W [2·nFreq, c] = embᵀ @ d_proj (contract over batch)
        code := code ++ s!"    %d_W{pIdx} = stablehlo.dot_general {r.preActSSA}, %{tag}_dproj,\n"
        code := code ++ s!"              contracting_dims = [0] x [0],\n"
        code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
        code := code ++ s!"            : ({tensorTy [b, embDim]}, {bc}) -> {tensorTy [embDim, c]}\n"
        -- gradSSA / gradShape unchanged: the add routes d_out to the feature.
      | _, _ => pure ()

    | .spatialFlatten =>
      -- Backward = inverse shape transformation. inShape is [B, C, H, W],
      -- outShape is [B, H*W, C]. The incoming gradSSA is on outShape.
      match r.inShape, r.outShape with
      | [b, c, h, w], _ =>
        let bhwc := [b, h, w, c]
        code := code ++ s!"    %sfb_r{r.pos} = stablehlo.reshape {gradSSA} : ({tensorTy r.outShape}) -> {tensorTy bhwc}\n"
        code := code ++ s!"    %sfb_t{r.pos} = stablehlo.transpose %sfb_r{r.pos}, dims = [0, 3, 1, 2] : ({tensorTy bhwc}) -> {tensorTy r.inShape}\n"
        gradSSA := s!"%sfb_t{r.pos}"
        gradShape := r.inShape
      | _, _ => pure ()

    | .spatialUnflatten _ _ _ =>
      -- Backward = inverse. inShape is [B, H*W, C], outShape is [B, C, H, W].
      match r.inShape, r.outShape with
      | _, [b, c, h, w] =>
        let bhwc := [b, h, w, c]
        code := code ++ s!"    %sub_t{r.pos} = stablehlo.transpose {gradSSA}, dims = [0, 2, 3, 1] : ({tensorTy r.outShape}) -> {tensorTy bhwc}\n"
        code := code ++ s!"    %sub_r{r.pos} = stablehlo.reshape %sub_t{r.pos} : ({tensorTy bhwc}) -> {tensorTy r.inShape}\n"
        gradSSA := s!"%sub_r{r.pos}"
        gradShape := r.inShape
      | _, _ => pure ()

    | .tokenPositionEmbed v t d _ _ _ =>
      if r.isTokenPosEmbed then
        match r.inShape with
        | [b, _] =>
          let pIdx := r.tpePidx          -- W_emb at pIdx, W_pos at pIdx + 1
          let btd := [b, t, d]
          let btv := [b, t, v]
          let btdTy := tensorTy btd
          let btvTy := tensorTy btv
          let _tag := s!"tpe{r.pos}"
          code := code ++ s!"    // ─── tokenPositionEmbed backward: {if r.tpePosEmb then "position grad = sum batch; " else ""}emb grad = {if r.tpeGather then "scatter-add over ids" else "btv^T @ d_out"} ───\n"
          -- d_position [T, D] = sum over batch of grad (only if the position table exists)
          if r.tpePosEmb then
            code := code ++ s!"    %d_W{pIdx + 1} = stablehlo.reduce({gradSSA} init: %zf) applies stablehlo.add across dimensions = [0]\n"
            code := code ++ s!"          : ({btdTy}, tensor<f32>) -> {tensorTy [t, d]}\n"
          if r.tpeGather then
            -- d_emb [V, D] = scatter-add of d_out[b,t,:] into row ids[b,t].
            -- tpeBtvSSA holds the [B, T, 1] i32 index tensor from the forward.
            let vd := tensorTy [v, d]
            let bt1i := s!"tensor<{b}x{t}x1xi32>"
            code := code ++ s!"    %tpe_dwz{r.pos} = stablehlo.constant dense<0.0> : {vd}\n"
            code := code ++ s!"    %d_W{pIdx} = \"stablehlo.scatter\"(%tpe_dwz{r.pos}, {r.tpeBtvSSA}, {gradSSA}) (" ++ "{\n"
            code := code ++ s!"    ^bb0(%tpe_lhs{r.pos}: tensor<f32>, %tpe_rhs{r.pos}: tensor<f32>):\n"
            code := code ++ s!"      %tpe_sum{r.pos} = stablehlo.add %tpe_lhs{r.pos}, %tpe_rhs{r.pos} : tensor<f32>\n"
            code := code ++ s!"      stablehlo.return %tpe_sum{r.pos} : tensor<f32>\n"
            code := code ++ "    }) {\n"
            code := code ++ "      scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>,\n"
            code := code ++ "      indices_are_sorted = false, unique_indices = false\n"
            code := code ++ "    } : " ++ s!"({vd}, {bt1i}, {btdTy}) -> {vd}\n"
          else
            -- d_emb [V, D] = btv^T @ d_out (contract over batch+time): [V, D]
            code := code ++ s!"    %d_W{pIdx} = stablehlo.dot_general {r.tpeBtvSSA}, {gradSSA},\n"
            code := code ++ s!"              contracting_dims = [0, 1] x [0, 1],\n"
            code := code ++ s!"              precision = [DEFAULT, DEFAULT]\n"
            code := code ++ s!"            : ({btvTy}, {btdTy}) -> {tensorTy [v, d]}\n"
          -- No d_input needed: input is the token ids, no gradient flows further.
          gradSSA := s!"// no_grad_needed_after_tokenPositionEmbed"
          gradShape := r.inShape
        | _ => pure ()
      else pure ()

    | _ => pure ()

  -- ═══════════════ OPTIMIZER UPDATES ═══════════════
  code := code ++ (if useAdam then "\n    // ================ ADAM UPDATES ================\n"
                   else "\n    // ================ SGD+MOMENTUM UPDATES ================\n")
  let wdActive := weightDecay > 0.0
  -- ─── Optional global-norm gradient clipping ───
  -- Compute scale = min(1, clipNorm / (‖g‖₂ + ε)) over ALL gradients, then each
  -- optimizer update pre-scales its gradient by it. ‖g‖₂ = sqrt(Σ_params Σ g²).
  -- Covers the standard param layers (conv2d/dense/convBn) — i.e. the CNN family
  -- that trains on the IREE path (ResNet/ConvNeXt + heads). gradClipNorm = 0 ⇒
  -- this whole block is skipped and clipScale stays none (no IR change).
  let mut clipScale : Option String := none
  if gradClipNorm > 0.0 then
    let mut gradList : Array (String × List Nat) := #[]
    for r in records do
      match r.pidx with
      | some p =>
        match r.layer with
        | .conv2d ic oc kSize _ _ =>
          gradList := gradList.push (s!"%d_W{p}", [oc, ic, kSize, kSize]) |>.push (s!"%d_b{p}", [oc])
        | .dense fanIn fanOut _ =>
          gradList := gradList.push (s!"%d_W{p}", [fanIn, fanOut]) |>.push (s!"%d_b{p}", [fanOut])
        | .convBn ic oc kSize _ _ =>
          let effIc := if r.isDepthwise then 1 else ic
          gradList := gradList.push (s!"%d_W{p}", [oc, effIc, kSize, kSize])
                              |>.push (s!"%d_g{p}", [oc]) |>.push (s!"%d_bt{p}", [oc])
        | _ => pure ()
      | none => pure ()
    if gradList.size > 0 then
      code := code ++ "\n    // ================ GRADIENT CLIP (global L2 norm) ================\n"
      let mut ssNames : Array String := #[]
      let mut i := 0
      for (gName, shape) in gradList do
        let ty := tensorTy shape
        let dims := String.intercalate ", " ((List.range shape.length).map toString)
        code := code ++ s!"    %gcsq_{i} = stablehlo.multiply {gName}, {gName} : {ty}\n"
        code := code ++ s!"    %gcss_{i} = stablehlo.reduce(%gcsq_{i} init: %zf) applies stablehlo.add across dimensions = [{dims}]\n"
        code := code ++ s!"           : ({ty}, tensor<f32>) -> tensor<f32>\n"
        ssNames := ssNames.push s!"%gcss_{i}"
        i := i + 1
      -- Tree-sum the per-gradient squared sums → total squared norm.
      let mut acc := ssNames[0]!
      for j in [1:ssNames.size] do
        code := code ++ s!"    %gcacc_{j} = stablehlo.add {acc}, {ssNames[j]!} : tensor<f32>\n"
        acc := s!"%gcacc_{j}"
      code := code ++ s!"    %gcnorm = stablehlo.sqrt {acc} : tensor<f32>\n"
      code := code ++ s!"    %gceps = stablehlo.constant dense<1.0e-06> : tensor<f32>\n"
      code := code ++ s!"    %gcnorme = stablehlo.add %gcnorm, %gceps : tensor<f32>\n"
      code := code ++ s!"    %gcthresh = stablehlo.constant dense<{gradClipNorm}> : tensor<f32>\n"
      code := code ++ s!"    %gcraw = stablehlo.divide %gcthresh, %gcnorme : tensor<f32>\n"
      code := code ++ s!"    %gcone = stablehlo.constant dense<1.0> : tensor<f32>\n"
      code := code ++ s!"    %gcscale = stablehlo.minimum %gcone, %gcraw : tensor<f32>\n"
      clipScale := some "%gcscale"
  -- ─── Per-group LR for the from-scratch dense head ───
  -- The head trains at headLrMult × base LR (backbone keeps the base LR). Only
  -- the .dense case below uses headLrSSA; everything else stays on %lr, so
  -- headLrMult = 1.0 emits identical IR. See TrainConfig.headLrMult.
  let headLrSSA : String := if headLrMult != 1.0 then "%lr_head" else "%lr"
  if headLrMult != 1.0 then
    code := code ++ s!"    %lr_headmult = stablehlo.constant dense<{headLrMult}> : tensor<f32>\n"
    code := code ++ s!"    %lr_head = stablehlo.multiply %lr, %lr_headmult : tensor<f32>\n"
  let mut paramRetNames : Array String := #[]
  let mut paramRetTypes : Array String := #[]
  let mut mRetNames : Array String := #[]
  let mut mRetTypes : Array String := #[]
  let mut vRetNames : Array String := #[]
  let mut vRetTypes : Array String := #[]
  let mut processedPidx : Array Nat := #[]
  -- Helper: choose the per-parameter update. With useMuon, every 2D weight matrix
  -- (both dims ≥ 16, so the small classifier head / embeddings stay on AdamW — Muon's
  -- canonical exclusion) is updated by Muon; all other params fall back to AdamW.
  -- Otherwise the original Adam/momentum dispatch is unchanged.
  let emitUpdate := fun (paramSSA gradSSA mSSA vSSA : String) (shape : List Nat) (tag : String) (applyWd : Bool) =>
    let is2DMuon : Bool := match shape with | [a, b] => decide (16 ≤ Nat.min a b) | _ => false
    -- Shampoo demo scope: only SQUARE 2D weights (L[n,n],R[n,n] reuse the m/v slots).
    let is2DSquareShampoo : Bool := match shape with | [a, b] => decide (a == b ∧ 16 ≤ a) | _ => false
    if useShampoo && is2DSquareShampoo then emitShampooUpdate paramSSA gradSSA mSSA vSSA shape tag (applyWeightDecay := applyWd) (clipScale := clipScale)
    else if useMuon && is2DMuon then emitMuonUpdate paramSSA gradSSA mSSA vSSA shape tag (applyWeightDecay := applyWd) (clipScale := clipScale)
    else if useAdam || useMuon || useShampoo then emitAdamUpdate paramSSA gradSSA mSSA vSSA shape tag (applyWeightDecay := applyWd) (clipScale := clipScale)
    else emitMomentumUpdate paramSSA gradSSA mSSA vSSA shape tag (applyWeightDecay := applyWd) (clipScale := clipScale)
  for r in records do
    match r.pidx with
    | some p =>
      if processedPidx.contains p then
        pure ()
      else
        processedPidx := processedPidx.push p
        match r.layer with
        | .conv2d ic oc kSize _ _ =>
          let wShape := [oc, ic, kSize, kSize]; let bShape := [oc]
          let (s1, wN, mwN, vwN) := emitUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"cW{p}" wdActive
          let (s2, bN, mbN, vbN) := emitUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" bShape s!"cb{p}" false
          code := code ++ s1 ++ s2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        | .dense fanIn fanOut _ =>
          let wShape := [fanIn, fanOut]; let bShape := [fanOut]
          -- Head dense layer: use the per-group head LR (headLrSSA). The W update
          -- also honours the matrix optimizers when engaged: a SQUARE dense weight
          -- routes to Shampoo, a large 2D one to Muon (both at the head LR). Small /
          -- non-square heads (e.g. ViT's 192×10 classifier) fall through to Adam, so
          -- default behaviour is unchanged.
          let is2DMuon : Bool := match wShape with | [a, b] => decide (16 ≤ Nat.min a b) | _ => false
          let is2DSquareShampoo : Bool := match wShape with | [a, b] => decide (a == b ∧ 16 ≤ a) | _ => false
          let headWUpd := fun (pS gS mS vS : String) (sh : List Nat) (tg : String) (wd : Bool) =>
            if useShampoo && is2DSquareShampoo then emitShampooUpdate pS gS mS vS sh tg (applyWeightDecay := wd) (clipScale := clipScale) (lrSSA := headLrSSA)
            else if useMuon && is2DMuon then emitMuonUpdate pS gS mS vS sh tg (applyWeightDecay := wd) (clipScale := clipScale) (lrSSA := headLrSSA)
            else if useAdam || useMuon || useShampoo then emitAdamUpdate pS gS mS vS sh tg (applyWeightDecay := wd) (clipScale := clipScale) (lrSSA := headLrSSA)
            else emitMomentumUpdate pS gS mS vS sh tg (applyWeightDecay := wd) (clipScale := clipScale) (lrSSA := headLrSSA)
          let headUpd := fun (pS gS mS vS : String) (sh : List Nat) (tg : String) (wd : Bool) =>
            if useAdam then emitAdamUpdate pS gS mS vS sh tg (applyWeightDecay := wd) (clipScale := clipScale) (lrSSA := headLrSSA)
            else emitMomentumUpdate pS gS mS vS sh tg (applyWeightDecay := wd) (clipScale := clipScale) (lrSSA := headLrSSA)
          let (s1, wN, mwN, vwN) := headWUpd s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"dW{p}" wdActive
          let (s2, bN, mbN, vbN) := headUpd s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" bShape s!"db{p}" false
          code := code ++ s1 ++ s2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        | .convBn ic oc kSize _ _ =>
          -- For depthwise convBn, weight shape is [channels, 1, kSize, kSize]
          let effIc := if r.isDepthwise then 1 else ic
          let (optCode, optNames, optTypes) :=
            if useAdam then emitConvBnAdam p effIc oc kSize (applyWeightDecay := wdActive) (clipScale := clipScale)
            else emitConvBnMomentum p effIc oc kSize (applyWeightDecay := wdActive) (clipScale := clipScale)
          code := code ++ optCode
          -- returns [wNew, gNew, btNew, mwNew, mgNew, mbtNew, vwNew, vgNew, vbtNew]
          paramRetNames := paramRetNames ++ optNames[:3]
          paramRetTypes := paramRetTypes ++ optTypes[:3]
          mRetNames := mRetNames ++ optNames[3:6]
          mRetTypes := mRetTypes ++ optTypes[3:6]
          vRetNames := vRetNames ++ optNames[6:]
          vRetTypes := vRetTypes ++ optTypes[6:]
        | .timeCondAdd c nFreq =>
          -- Plain dense update: W [2·nFreq, c] + b [c] (no head LR, wd on W).
          let wShape := [2 * nFreq, c]; let bShape := [c]
          let (s1, wN, mwN, vwN) := emitUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" wShape s!"tcW{p}" wdActive
          let (s2, bN, mbN, vbN) := emitUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" bShape s!"tcb{p}" false
          code := code ++ s1 ++ s2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        | _ => pure ()
    | none =>
      -- SE records (marker layer = globalAvgPool, isSE := true) carry 2 conv2d-style param
      -- pidxes (reduce, expand). Emit optimizer updates for them inline.
      if r.isSE then
        let mid := r.seMidFull
        let seMid := r.seMid
        let pRed := r.sePidxRed
        let pExp := r.sePidxExp
        -- Reduce: W shape [seMid, mid, 1, 1], b shape [seMid]
        let wShapeR := [seMid, mid, 1, 1]; let bShapeR := [seMid]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pRed}" s!"%d_W{pRed}" s!"%m_W{pRed}" s!"%v_W{pRed}" wShapeR s!"seRW{pRed}" wdActive
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pRed}" s!"%d_b{pRed}" s!"%m_b{pRed}" s!"%v_b{pRed}" bShapeR s!"seRb{pRed}" false
        code := code ++ sa1 ++ sa2
        paramRetNames := paramRetNames.push wN |>.push bN
        paramRetTypes := paramRetTypes.push (tensorTy wShapeR) |>.push (tensorTy bShapeR)
        mRetNames := mRetNames.push mwN |>.push mbN
        mRetTypes := mRetTypes.push (tensorTy wShapeR) |>.push (tensorTy bShapeR)
        vRetNames := vRetNames.push vwN |>.push vbN
        vRetTypes := vRetTypes.push (tensorTy wShapeR) |>.push (tensorTy bShapeR)
        -- Expand: W shape [mid, seMid, 1, 1], b shape [mid]
        let wShapeE := [mid, seMid, 1, 1]; let bShapeE := [mid]
        let (sa3, wN2, mwN2, vwN2) := emitUpdate s!"%W{pExp}" s!"%d_W{pExp}" s!"%m_W{pExp}" s!"%v_W{pExp}" wShapeE s!"seEW{pExp}" wdActive
        let (sa4, bN2, mbN2, vbN2) := emitUpdate s!"%b{pExp}" s!"%d_b{pExp}" s!"%m_b{pExp}" s!"%v_b{pExp}" bShapeE s!"seEb{pExp}" false
        code := code ++ sa3 ++ sa4
        paramRetNames := paramRetNames.push wN2 |>.push bN2
        paramRetTypes := paramRetTypes.push (tensorTy wShapeE) |>.push (tensorTy bShapeE)
        mRetNames := mRetNames.push mwN2 |>.push mbN2
        mRetTypes := mRetTypes.push (tensorTy wShapeE) |>.push (tensorTy bShapeE)
        vRetNames := vRetNames.push vwN2 |>.push vbN2
        vRetTypes := vRetTypes.push (tensorTy wShapeE) |>.push (tensorTy bShapeE)
      -- ViT patch embedding: 3 param tensors
      if r.isPatchEmbed then
        let dim := r.pePDim
        let ic := r.pePIc
        let pSize := r.pePSize
        let nP := r.pePNp
        let pW := r.pePWPidx
        let pB := r.pePBPidx
        let pCls := r.peClsPidx
        let pPos := r.pePosPidx
        let wShape := [dim, ic, pSize, pSize]
        let bShape := [dim]
        let clsShape := [dim]
        let posShape := [nP + 1, dim]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pW}" s!"%d_W{pW}" s!"%m_W{pW}" s!"%v_W{pW}" wShape s!"peW{pW}" wdActive
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pB}" s!"%d_b{pB}" s!"%m_b{pB}" s!"%v_b{pB}" bShape s!"peb{pB}" false
        let (sa3, clsN, mclsN, vclsN) := emitUpdate s!"%W{pCls}" s!"%d_W{pCls}" s!"%m_W{pCls}" s!"%v_W{pCls}" clsShape s!"peCls{pCls}" false
        let (sa4, posN, mposN, vposN) := emitUpdate s!"%W{pPos}" s!"%d_W{pPos}" s!"%m_W{pPos}" s!"%v_W{pPos}" posShape s!"pePos{pPos}" false
        code := code ++ sa1 ++ sa2 ++ sa3 ++ sa4
        paramRetNames := paramRetNames.push wN |>.push bN |>.push clsN |>.push posN
        paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape) |>.push (tensorTy clsShape) |>.push (tensorTy posShape)
        mRetNames := mRetNames.push mwN |>.push mbN |>.push mclsN |>.push mposN
        mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape) |>.push (tensorTy clsShape) |>.push (tensorTy posShape)
        vRetNames := vRetNames.push vwN |>.push vbN |>.push vclsN |>.push vposN
        vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape) |>.push (tensorTy clsShape) |>.push (tensorTy posShape)
      -- ViT transformer block: 8 param pairs (each W + b)
      if r.isTransformerBlock then
        let dim := r.tbDim
        let mlpDim := r.tbMlpDim
        let basePidx := r.tbBasePidx
        -- Shapes for each param pair:
        -- LN1: (dim,), (dim,); Wq,Wk,Wv,Wo: (dim, dim), (dim,); LN2: (dim,), (dim,); Wfc1: (dim, mlpDim), (mlpDim,); Wfc2: (mlpDim, dim), (dim,)
        let paramShapes : Array (List Nat × List Nat) := #[
          ([dim], [dim]),            -- LN1
          ([dim, dim], [dim]),       -- Wq
          ([dim, dim], [dim]),       -- Wk
          ([dim, dim], [dim]),       -- Wv
          ([dim, dim], [dim]),       -- Wo
          ([dim], [dim]),            -- LN2
          ([dim, mlpDim], [mlpDim]), -- Wfc1
          ([mlpDim, dim], [dim])     -- Wfc2
        ]
        let decayW : Array Bool := #[false, true, true, true, true, false, true, true]
        for i in [:8] do
          let pp := basePidx + i
          let (wShape, bShape) := paramShapes[i]!
          let decay := decayW[i]! && wdActive
          let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pp}" s!"%d_W{pp}" s!"%m_W{pp}" s!"%v_W{pp}" wShape s!"tb_w{pp}" decay
          let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pp}" s!"%d_b{pp}" s!"%m_b{pp}" s!"%v_b{pp}" bShape s!"tb_b{pp}" false
          code := code ++ sa1 ++ sa2
          paramRetNames := paramRetNames.push wN |>.push bN
          paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          mRetNames := mRetNames.push mwN |>.push mbN
          mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
          vRetNames := vRetNames.push vwN |>.push vbN
          vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
      -- ViT final LN (after all transformer blocks)
      if r.isFinalLn then
        let dim := r.inShape[2]!
        let p := r.finalLnPidx
        let shape := [dim]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{p}" s!"%d_W{p}" s!"%m_W{p}" s!"%v_W{p}" shape s!"fln_w{p}" false
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{p}" s!"%d_b{p}" s!"%m_b{p}" s!"%v_b{p}" shape s!"fln_b{p}" false
        code := code ++ sa1 ++ sa2
        paramRetNames := paramRetNames.push wN |>.push bN
        paramRetTypes := paramRetTypes.push (tensorTy shape) |>.push (tensorTy shape)
        mRetNames := mRetNames.push mwN |>.push mbN
        mRetTypes := mRetTypes.push (tensorTy shape) |>.push (tensorTy shape)
        vRetNames := vRetNames.push vwN |>.push vbN
        vRetTypes := vRetTypes.push (tensorTy shape) |>.push (tensorTy shape)
      -- ConvNeXt block: 5 pidx slots (DW, LN, expand, project, LayerScale)
      if r.isConvNextBlock then
        let basePidx := r.cnbBasePidx
        let c := r.cnbChannels
        let cShape : List Nat := [c]
        let dwShape : List Nat := [c, 1, 7, 7]
        let exShape : List Nat := [4*c, c, 1, 1]; let exB : List Nat := [4*c]
        let pjShape : List Nat := [c, 4*c, 1, 1]
        -- DW (W, b)
        let pDw := basePidx
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pDw}" s!"%d_W{pDw}" s!"%m_W{pDw}" s!"%v_W{pDw}" dwShape s!"cnbDw{pDw}" wdActive
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pDw}" s!"%d_b{pDw}" s!"%m_b{pDw}" s!"%v_b{pDw}" cShape s!"cnbDwb{pDw}" false
        code := code ++ sa1 ++ sa2
        paramRetNames := paramRetNames.push wN |>.push bN
        paramRetTypes := paramRetTypes.push (tensorTy dwShape) |>.push (tensorTy cShape)
        mRetNames := mRetNames.push mwN |>.push mbN
        mRetTypes := mRetTypes.push (tensorTy dwShape) |>.push (tensorTy cShape)
        vRetNames := vRetNames.push vwN |>.push vbN
        vRetTypes := vRetTypes.push (tensorTy dwShape) |>.push (tensorTy cShape)
        -- LN (γ, β) — no weight decay on γ/β
        let pLn := basePidx + 1
        let (sa3, gN, mgN, vgN) := emitUpdate s!"%W{pLn}" s!"%d_W{pLn}" s!"%m_W{pLn}" s!"%v_W{pLn}" cShape s!"cnbLng{pLn}" false
        let (sa4, betaN, mbetaN, vbetaN) := emitUpdate s!"%b{pLn}" s!"%d_b{pLn}" s!"%m_b{pLn}" s!"%v_b{pLn}" cShape s!"cnbLnb{pLn}" false
        code := code ++ sa3 ++ sa4
        paramRetNames := paramRetNames.push gN |>.push betaN
        paramRetTypes := paramRetTypes.push (tensorTy cShape) |>.push (tensorTy cShape)
        mRetNames := mRetNames.push mgN |>.push mbetaN
        mRetTypes := mRetTypes.push (tensorTy cShape) |>.push (tensorTy cShape)
        vRetNames := vRetNames.push vgN |>.push vbetaN
        vRetTypes := vRetTypes.push (tensorTy cShape) |>.push (tensorTy cShape)
        -- Expand (W, b)
        let pEx := basePidx + 2
        let (sa5, wEx, mwEx, vwEx) := emitUpdate s!"%W{pEx}" s!"%d_W{pEx}" s!"%m_W{pEx}" s!"%v_W{pEx}" exShape s!"cnbEx{pEx}" wdActive
        let (sa6, bEx, mbEx, vbEx) := emitUpdate s!"%b{pEx}" s!"%d_b{pEx}" s!"%m_b{pEx}" s!"%v_b{pEx}" exB s!"cnbExb{pEx}" false
        code := code ++ sa5 ++ sa6
        paramRetNames := paramRetNames.push wEx |>.push bEx
        paramRetTypes := paramRetTypes.push (tensorTy exShape) |>.push (tensorTy exB)
        mRetNames := mRetNames.push mwEx |>.push mbEx
        mRetTypes := mRetTypes.push (tensorTy exShape) |>.push (tensorTy exB)
        vRetNames := vRetNames.push vwEx |>.push vbEx
        vRetTypes := vRetTypes.push (tensorTy exShape) |>.push (tensorTy exB)
        -- Project (W, b)
        let pPj := basePidx + 3
        let (sa7, wPj, mwPj, vwPj) := emitUpdate s!"%W{pPj}" s!"%d_W{pPj}" s!"%m_W{pPj}" s!"%v_W{pPj}" pjShape s!"cnbPj{pPj}" wdActive
        let (sa8, bPj, mbPj, vbPj) := emitUpdate s!"%b{pPj}" s!"%d_b{pPj}" s!"%m_b{pPj}" s!"%v_b{pPj}" cShape s!"cnbPjb{pPj}" false
        code := code ++ sa7 ++ sa8
        paramRetNames := paramRetNames.push wPj |>.push bPj
        paramRetTypes := paramRetTypes.push (tensorTy pjShape) |>.push (tensorTy cShape)
        mRetNames := mRetNames.push mwPj |>.push mbPj
        mRetTypes := mRetTypes.push (tensorTy pjShape) |>.push (tensorTy cShape)
        vRetNames := vRetNames.push vwPj |>.push vbPj
        vRetTypes := vRetTypes.push (tensorTy pjShape) |>.push (tensorTy cShape)
        -- LayerScale γ — no weight decay
        let pLs := basePidx + 4
        let (sa9, lsN, mlsN, vlsN) := emitUpdate s!"%W{pLs}" s!"%d_W{pLs}" s!"%m_W{pLs}" s!"%v_W{pLs}" cShape s!"cnbLs{pLs}" false
        code := code ++ sa9
        paramRetNames := paramRetNames.push lsN
        paramRetTypes := paramRetTypes.push (tensorTy cShape)
        mRetNames := mRetNames.push mlsN
        mRetTypes := mRetTypes.push (tensorTy cShape)
        vRetNames := vRetNames.push vlsN
        vRetTypes := vRetTypes.push (tensorTy cShape)
      -- ConvNeXt downsample: 2 pidx slots (LN, conv)
      if r.isConvNextDs then
        let basePidx := r.cndBasePidx
        let ic := r.cndIc
        let oc := r.cndOc
        let icShape : List Nat := [ic]
        let ocShape : List Nat := [oc]
        let cvShape : List Nat := [oc, ic, 2, 2]
        let pLn := basePidx
        let pCv := basePidx + 1
        let (sa1, gN, mgN, vgN) := emitUpdate s!"%W{pLn}" s!"%d_W{pLn}" s!"%m_W{pLn}" s!"%v_W{pLn}" icShape s!"cndLng{pLn}" false
        let (sa2, betaN, mbetaN, vbetaN) := emitUpdate s!"%b{pLn}" s!"%d_b{pLn}" s!"%m_b{pLn}" s!"%v_b{pLn}" icShape s!"cndLnb{pLn}" false
        code := code ++ sa1 ++ sa2
        paramRetNames := paramRetNames.push gN |>.push betaN
        paramRetTypes := paramRetTypes.push (tensorTy icShape) |>.push (tensorTy icShape)
        mRetNames := mRetNames.push mgN |>.push mbetaN
        mRetTypes := mRetTypes.push (tensorTy icShape) |>.push (tensorTy icShape)
        vRetNames := vRetNames.push vgN |>.push vbetaN
        vRetTypes := vRetTypes.push (tensorTy icShape) |>.push (tensorTy icShape)
        let (sa3, wN, mwN, vwN) := emitUpdate s!"%W{pCv}" s!"%d_W{pCv}" s!"%m_W{pCv}" s!"%v_W{pCv}" cvShape s!"cndCv{pCv}" wdActive
        let (sa4, bN, mbN, vbN) := emitUpdate s!"%b{pCv}" s!"%d_b{pCv}" s!"%m_b{pCv}" s!"%v_b{pCv}" ocShape s!"cndCvb{pCv}" false
        code := code ++ sa3 ++ sa4
        paramRetNames := paramRetNames.push wN |>.push bN
        paramRetTypes := paramRetTypes.push (tensorTy cvShape) |>.push (tensorTy ocShape)
        mRetNames := mRetNames.push mwN |>.push mbN
        mRetTypes := mRetTypes.push (tensorTy cvShape) |>.push (tensorTy ocShape)
        vRetNames := vRetNames.push vwN |>.push vbN
        vRetTypes := vRetTypes.push (tensorTy cvShape) |>.push (tensorTy ocShape)
      -- tinyGPT token+position embedding: 2 param tensors, no biases.
      if r.isTokenPosEmbed then
        let pIdx := r.tpePidx
        let v := r.tpeV; let t := r.tpeT; let d := r.tpeD
        let wShape := [v, d]; let posShape := [t, d]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pIdx}" s!"%d_W{pIdx}" s!"%m_W{pIdx}" s!"%v_W{pIdx}" wShape s!"tpeW{pIdx}" wdActive
        code := code ++ sa1
        paramRetNames := paramRetNames.push wN
        paramRetTypes := paramRetTypes.push (tensorTy wShape)
        mRetNames := mRetNames.push mwN
        mRetTypes := mRetTypes.push (tensorTy wShape)
        vRetNames := vRetNames.push vwN
        vRetTypes := vRetTypes.push (tensorTy wShape)
        if r.tpePosEmb then
          let (sa2, posN, mposN, vposN) := emitUpdate s!"%W{pIdx + 1}" s!"%d_W{pIdx + 1}" s!"%m_W{pIdx + 1}" s!"%v_W{pIdx + 1}" posShape s!"tpePos{pIdx}" false
          code := code ++ sa2
          paramRetNames := paramRetNames.push posN
          paramRetTypes := paramRetTypes.push (tensorTy posShape)
          mRetNames := mRetNames.push mposN
          mRetTypes := mRetTypes.push (tensorTy posShape)
          vRetNames := vRetNames.push vposN
          vRetTypes := vRetTypes.push (tensorTy posShape)
      -- tinyGPT LM head: 1 dense pair (W [D, V], b [V]).
      if r.isLmHead then
        let pIdx := r.lmhPidx
        let d := r.lmhD; let v := r.lmhV
        let wShape := [d, v]; let bShape := [v]
        let (sa1, wN, mwN, vwN) := emitUpdate s!"%W{pIdx}" s!"%d_W{pIdx}" s!"%m_W{pIdx}" s!"%v_W{pIdx}" wShape s!"lmhW{pIdx}" wdActive
        let (sa2, bN, mbN, vbN) := emitUpdate s!"%b{pIdx}" s!"%d_b{pIdx}" s!"%m_b{pIdx}" s!"%v_b{pIdx}" bShape s!"lmhb{pIdx}" false
        code := code ++ sa1 ++ sa2
        paramRetNames := paramRetNames.push wN |>.push bN
        paramRetTypes := paramRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        mRetNames := mRetNames.push mwN |>.push mbN
        mRetTypes := mRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
        vRetNames := vRetNames.push vwN |>.push vbN
        vRetTypes := vRetTypes.push (tensorTy wShape) |>.push (tensorTy bShape)
  -- Return order: params, m, v, loss, then BN stats (mean0, var0, mean1, var1, ...)
  let mut retNames := paramRetNames ++ mRetNames ++ vRetNames |>.push "%loss"
  let mut retTypes := paramRetTypes ++ mRetTypes ++ vRetTypes |>.push "tensor<f32>"

  -- Append BN mean/var for each BN layer (computed during forward)
  let bnLayers := collectBnLayers spec
  for (p, oc) in bnLayers do
    retNames := retNames.push s!"%cbn_mean{p}" |>.push s!"%cbn_var{p}"
    retTypes := retTypes.push (tensorTy [oc]) |>.push (tensorTy [oc])

  code := code ++ s!"    return {String.intercalate ", " retNames.toList}\n"
  code := code ++ s!"      : {String.intercalate ", " retTypes.toList}\n"
  pure code

/-- Emit the train_step function signature. -/
private def emitTrainStepSig (spec : NetSpec) (batchSize : Nat)
    (useSoftLabels : Bool := false) (useSeg : Bool := false)
    (useDdpm : Bool := false) (ddpmOutShape : List Nat := [])
    (useYolov1 : Bool := false)
    (yoloGridH : Nat := 7) (yoloGridW : Nat := 7) (yoloPerCell : Nat := 30)
    : String := Id.run do
  let B := batchSize
  let NC := spec.numClasses
  let inDim := inputFlatDim spec
  let mut params : String := ""
  let mut paramRetTypes : Array String := #[]
  let mut mRetTypes : Array String := #[]
  let mut vRetTypes : Array String := #[]
  let mut pidx : Nat := 0
  let mut curShape : List Nat := [B, inDim]
  match inputChannels spec with
  | some ic => curShape := [B, ic, spec.imageH, spec.imageW]
  | none => pure ()
  -- Helper: emit param block for one layer (used for params, m_, and v_)
  let emitLayerParams := fun (pfx : String) (l : Layer) (idx : Nat) (retTypes : Array String) =>
    match l with
    | .conv2d ic oc kSize _ _ =>
      let wTy := tensorTy [oc, ic, kSize, kSize]; let bTy := tensorTy [oc]
      (s!"      %{pfx}W{idx}: {wTy}, %{pfx}b{idx}: {bTy},\n",
       retTypes.push wTy |>.push bTy)
    | .dense fanIn fanOut _ =>
      let wTy := tensorTy [fanIn, fanOut]; let bTy := tensorTy [fanOut]
      (s!"      %{pfx}W{idx}: {wTy}, %{pfx}b{idx}: {bTy},\n",
       retTypes.push wTy |>.push bTy)
    | .convBn ic oc kSize _ _ =>
      let wTy := tensorTy [oc, ic, kSize, kSize]; let gTy := tensorTy [oc]
      (s!"      %{pfx}W{idx}: {wTy}, %{pfx}g{idx}: {gTy}, %{pfx}bt{idx}: {gTy},\n",
       retTypes.push wTy |>.push gTy |>.push gTy)
    | _ => ("", retTypes)
  -- Walk layers to build param/m/v argument lists
  for l in spec.layers do
    match l with
    | .conv2d _ic oc _kSize _ _ =>
      let (pStr, pTys) := emitLayerParams "" l pidx paramRetTypes
      params := params ++ pStr; paramRetTypes := pTys
      mRetTypes := (emitLayerParams "" l pidx mRetTypes).2
      vRetTypes := (emitLayerParams "" l pidx vRetTypes).2
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h, w]
      | _ => pure ()
      pidx := pidx + 1
    | .dense _fanIn fanOut _ =>
      let (pStr, pTys) := emitLayerParams "" l pidx paramRetTypes
      params := params ++ pStr; paramRetTypes := pTys
      mRetTypes := (emitLayerParams "" l pidx mRetTypes).2
      vRetTypes := (emitLayerParams "" l pidx vRetTypes).2
      curShape := [B, fanOut]
      pidx := pidx + 1
    | .convBn _ic oc _kSize stride _ =>
      let (pStr, pTys) := emitLayerParams "" l pidx paramRetTypes
      params := params ++ pStr; paramRetTypes := pTys
      mRetTypes := (emitLayerParams "" l pidx mRetTypes).2
      vRetTypes := (emitLayerParams "" l pidx vRetTypes).2
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
      pidx := pidx + 1
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      let gTy := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let wTy1 := tensorTy [oc, blockIc, 3, 3]
        params := params ++ s!"      %W{pidx}: {wTy1}, %g{pidx}: {gTy}, %bt{pidx}: {gTy},\n"
        paramRetTypes := paramRetTypes.push wTy1 |>.push gTy |>.push gTy
        mRetTypes := mRetTypes.push wTy1 |>.push gTy |>.push gTy
        vRetTypes := vRetTypes.push wTy1 |>.push gTy |>.push gTy
        pidx := pidx + 1
        let wTy2 := tensorTy [oc, oc, 3, 3]
        params := params ++ s!"      %W{pidx}: {wTy2}, %g{pidx}: {gTy}, %bt{pidx}: {gTy},\n"
        paramRetTypes := paramRetTypes.push wTy2 |>.push gTy |>.push gTy
        mRetTypes := mRetTypes.push wTy2 |>.push gTy |>.push gTy
        vRetTypes := vRetTypes.push wTy2 |>.push gTy |>.push gTy
        pidx := pidx + 1
        if bi == 0 && needsProj then
          let pTy := tensorTy [oc, ic, 1, 1]
          params := params ++ s!"      %W{pidx}: {pTy}, %g{pidx}: {gTy}, %bt{pidx}: {gTy},\n"
          paramRetTypes := paramRetTypes.push pTy |>.push gTy |>.push gTy
          mRetTypes := mRetTypes.push pTy |>.push gTy |>.push gTy
          vRetTypes := vRetTypes.push pTy |>.push gTy |>.push gTy
          pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let wTy1 := tensorTy [mid, blockIc, 1, 1]
        params := params ++ s!"      %W{pidx}: {wTy1}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy1 |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy1 |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy1 |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        let wTy2 := tensorTy [mid, mid, 3, 3]
        params := params ++ s!"      %W{pidx}: {wTy2}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy2 |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy2 |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy2 |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        let wTy3 := tensorTy [oc, mid, 1, 1]
        params := params ++ s!"      %W{pidx}: {wTy3}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
        paramRetTypes := paramRetTypes.push wTy3 |>.push gTyO |>.push gTyO
        mRetTypes := mRetTypes.push wTy3 |>.push gTyO |>.push gTyO
        vRetTypes := vRetTypes.push wTy3 |>.push gTyO |>.push gTyO
        pidx := pidx + 1
        if bi == 0 && needsProj then
          let pTy := tensorTy [oc, ic, 1, 1]
          params := params ++ s!"      %W{pidx}: {pTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
          paramRetTypes := paramRetTypes.push pTy |>.push gTyO |>.push gTyO
          mRetTypes := mRetTypes.push pTy |>.push gTyO |>.push gTyO
          vRetTypes := vRetTypes.push pTy |>.push gTyO |>.push gTyO
          pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + firstStride - 1) / firstStride, (w + firstStride - 1) / firstStride]
      | _ => pure ()
    | .invertedResidual ic oc expand stride nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        -- Expand 1×1 (skip if expand==1)
        if expand != 1 then
          let wTy := tensorTy [mid, blockIc, 1, 1]
          params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
          paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
          mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
          vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
          pidx := pidx + 1
        -- Depthwise 3×3
        let dwTy := tensorTy [mid, 1, 3, 3]
        params := params ++ s!"      %W{pidx}: {dwTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push dwTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push dwTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push dwTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        -- Project 1×1
        let pjTy := tensorTy [oc, mid, 1, 1]
        params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
        paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
        mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
        vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
        pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .mbConv ic oc expand kSize stride nBlocks useSE _act =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          let wTy := tensorTy [mid, blockIc, 1, 1]
          params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
          paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
          mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
          vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
          pidx := pidx + 1
        let dwTy := tensorTy [mid, 1, kSize, kSize]
        params := params ++ s!"      %W{pidx}: {dwTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push dwTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push dwTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push dwTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        if useSE then
          let wRedTy := tensorTy [seMid, mid, 1, 1]
          let bRedTy := tensorTy [seMid]
          params := params ++ s!"      %W{pidx}: {wRedTy}, %b{pidx}: {bRedTy},\n"
          paramRetTypes := paramRetTypes.push wRedTy |>.push bRedTy
          mRetTypes := mRetTypes.push wRedTy |>.push bRedTy
          vRetTypes := vRetTypes.push wRedTy |>.push bRedTy
          pidx := pidx + 1
          let wExpTy := tensorTy [mid, seMid, 1, 1]
          let bExpTy := tensorTy [mid]
          params := params ++ s!"      %W{pidx}: {wExpTy}, %b{pidx}: {bExpTy},\n"
          paramRetTypes := paramRetTypes.push wExpTy |>.push bExpTy
          mRetTypes := mRetTypes.push wExpTy |>.push bExpTy
          vRetTypes := vRetTypes.push wExpTy |>.push bExpTy
          pidx := pidx + 1
        let pjTy := tensorTy [oc, mid, 1, 1]
        params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
        paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
        mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
        vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
        pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .fusedMbConv ic oc expand kSize stride nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        -- Fused expand: regular kxk convBn
        let wTy := tensorTy [mid, blockIc, kSize, kSize]
        params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
        if useSE then
          let wRedTy := tensorTy [seMid, mid, 1, 1]
          let bRedTy := tensorTy [seMid]
          params := params ++ s!"      %W{pidx}: {wRedTy}, %b{pidx}: {bRedTy},\n"
          paramRetTypes := paramRetTypes.push wRedTy |>.push bRedTy
          mRetTypes := mRetTypes.push wRedTy |>.push bRedTy
          vRetTypes := vRetTypes.push wRedTy |>.push bRedTy
          pidx := pidx + 1
          let wExpTy := tensorTy [mid, seMid, 1, 1]
          let bExpTy := tensorTy [mid]
          params := params ++ s!"      %W{pidx}: {wExpTy}, %b{pidx}: {bExpTy},\n"
          paramRetTypes := paramRetTypes.push wExpTy |>.push bExpTy
          mRetTypes := mRetTypes.push wExpTy |>.push bExpTy
          vRetTypes := vRetTypes.push wExpTy |>.push bExpTy
          pidx := pidx + 1
        if expand != 1 then
          let pjTy := tensorTy [oc, mid, 1, 1]
          params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
          paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
          mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
          vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
          pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .mbConvV3 ic oc expandCh kSize stride useSE _act =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if expandCh != ic then
        let wTy := tensorTy [mid, ic, 1, 1]
        params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
      let dwTy := tensorTy [mid, 1, kSize, kSize]
      params := params ++ s!"      %W{pidx}: {dwTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
      paramRetTypes := paramRetTypes.push dwTy |>.push gTyM |>.push gTyM
      mRetTypes := mRetTypes.push dwTy |>.push gTyM |>.push gTyM
      vRetTypes := vRetTypes.push dwTy |>.push gTyM |>.push gTyM
      pidx := pidx + 1
      if useSE then
        let wRedTy := tensorTy [seMid, mid, 1, 1]
        let bRedTy := tensorTy [seMid]
        params := params ++ s!"      %W{pidx}: {wRedTy}, %b{pidx}: {bRedTy},\n"
        paramRetTypes := paramRetTypes.push wRedTy |>.push bRedTy
        mRetTypes := mRetTypes.push wRedTy |>.push bRedTy
        vRetTypes := vRetTypes.push wRedTy |>.push bRedTy
        pidx := pidx + 1
        let wExpTy := tensorTy [mid, seMid, 1, 1]
        let bExpTy := tensorTy [mid]
        params := params ++ s!"      %W{pidx}: {wExpTy}, %b{pidx}: {bExpTy},\n"
        paramRetTypes := paramRetTypes.push wExpTy |>.push bExpTy
        mRetTypes := mRetTypes.push wExpTy |>.push bExpTy
        vRetTypes := vRetTypes.push wExpTy |>.push bExpTy
        pidx := pidx + 1
      let pjTy := tensorTy [oc, mid, 1, 1]
      params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
      paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
      mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
      vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .uib ic oc expand stride preDWk postDWk =>
      let mid := ic * expand
      let gTyI := tensorTy [ic]; let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      -- preDW (if preDWk > 0): W[ic, 1, k, k], gamma/beta [ic]
      if preDWk > 0 then
        let wTy := tensorTy [ic, 1, preDWk, preDWk]
        params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyI}, %bt{pidx}: {gTyI},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push gTyI |>.push gTyI
        mRetTypes := mRetTypes.push wTy |>.push gTyI |>.push gTyI
        vRetTypes := vRetTypes.push wTy |>.push gTyI |>.push gTyI
        pidx := pidx + 1
      -- Expand 1×1: W[mid, ic, 1, 1]
      let exTy := tensorTy [mid, ic, 1, 1]
      params := params ++ s!"      %W{pidx}: {exTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
      paramRetTypes := paramRetTypes.push exTy |>.push gTyM |>.push gTyM
      mRetTypes := mRetTypes.push exTy |>.push gTyM |>.push gTyM
      vRetTypes := vRetTypes.push exTy |>.push gTyM |>.push gTyM
      pidx := pidx + 1
      -- postDW (if postDWk > 0): W[mid, 1, k, k]
      if postDWk > 0 then
        let wTy := tensorTy [mid, 1, postDWk, postDWk]
        params := params ++ s!"      %W{pidx}: {wTy}, %g{pidx}: {gTyM}, %bt{pidx}: {gTyM},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push gTyM |>.push gTyM
        mRetTypes := mRetTypes.push wTy |>.push gTyM |>.push gTyM
        vRetTypes := vRetTypes.push wTy |>.push gTyM |>.push gTyM
        pidx := pidx + 1
      -- Project 1×1: W[oc, mid, 1, 1]
      let pjTy := tensorTy [oc, mid, 1, 1]
      params := params ++ s!"      %W{pidx}: {pjTy}, %g{pidx}: {gTyO}, %bt{pidx}: {gTyO},\n"
      paramRetTypes := paramRetTypes.push pjTy |>.push gTyO |>.push gTyO
      mRetTypes := mRetTypes.push pjTy |>.push gTyO |>.push gTyO
      vRetTypes := vRetTypes.push pjTy |>.push gTyO |>.push gTyO
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .convNextStage channels nBlocks _norm _act =>
      let c := channels
      let cTy := tensorTy [c]
      let dwTy := tensorTy [c, 1, 7, 7]
      let exTy := tensorTy [4*c, c, 1, 1]
      let exB := tensorTy [4*c]
      let pjTy := tensorTy [c, 4*c, 1, 1]
      for _ in [:nBlocks] do
        -- DW (W, b)
        params := params ++ s!"      %W{pidx}: {dwTy}, %b{pidx}: {cTy},\n"
        paramRetTypes := paramRetTypes.push dwTy |>.push cTy
        mRetTypes := mRetTypes.push dwTy |>.push cTy
        vRetTypes := vRetTypes.push dwTy |>.push cTy
        pidx := pidx + 1
        -- LN (γ, β)
        params := params ++ s!"      %W{pidx}: {cTy}, %b{pidx}: {cTy},\n"
        paramRetTypes := paramRetTypes.push cTy |>.push cTy
        mRetTypes := mRetTypes.push cTy |>.push cTy
        vRetTypes := vRetTypes.push cTy |>.push cTy
        pidx := pidx + 1
        -- 1×1 expand (W, b)
        params := params ++ s!"      %W{pidx}: {exTy}, %b{pidx}: {exB},\n"
        paramRetTypes := paramRetTypes.push exTy |>.push exB
        mRetTypes := mRetTypes.push exTy |>.push exB
        vRetTypes := vRetTypes.push exTy |>.push exB
        pidx := pidx + 1
        -- 1×1 project (W, b)
        params := params ++ s!"      %W{pidx}: {pjTy}, %b{pidx}: {cTy},\n"
        paramRetTypes := paramRetTypes.push pjTy |>.push cTy
        mRetTypes := mRetTypes.push pjTy |>.push cTy
        vRetTypes := vRetTypes.push pjTy |>.push cTy
        pidx := pidx + 1
        -- LayerScale (γ only)
        params := params ++ s!"      %W{pidx}: {cTy},\n"
        paramRetTypes := paramRetTypes.push cTy
        mRetTypes := mRetTypes.push cTy
        vRetTypes := vRetTypes.push cTy
        pidx := pidx + 1
    | .convNextDownsample ic oc _norm =>
      let icTy := tensorTy [ic]
      let ocTy := tensorTy [oc]
      let cvTy := tensorTy [oc, ic, 2, 2]
      -- LN (γ, β)
      params := params ++ s!"      %W{pidx}: {icTy}, %b{pidx}: {icTy},\n"
      paramRetTypes := paramRetTypes.push icTy |>.push icTy
      mRetTypes := mRetTypes.push icTy |>.push icTy
      vRetTypes := vRetTypes.push icTy |>.push icTy
      pidx := pidx + 1
      -- 2×2 stride-2 conv (W, b)
      params := params ++ s!"      %W{pidx}: {cvTy}, %b{pidx}: {ocTy},\n"
      paramRetTypes := paramRetTypes.push cvTy |>.push ocTy
      mRetTypes := mRetTypes.push cvTy |>.push ocTy
      vRetTypes := vRetTypes.push cvTy |>.push ocTy
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h / 2, w / 2]
      | _ => pure ()
    | .unetDown ic oc =>
      -- 2× convBn (ic→oc, oc→oc) then maxPool 2 (no params).
      let ocTy := tensorTy [oc]
      let cv1Ty := tensorTy [oc, ic, 3, 3]
      let cv2Ty := tensorTy [oc, oc, 3, 3]
      params := params ++ s!"      %W{pidx}: {cv1Ty}, %g{pidx}: {ocTy}, %bt{pidx}: {ocTy},\n"
      paramRetTypes := paramRetTypes.push cv1Ty |>.push ocTy |>.push ocTy
      mRetTypes := mRetTypes.push cv1Ty |>.push ocTy |>.push ocTy
      vRetTypes := vRetTypes.push cv1Ty |>.push ocTy |>.push ocTy
      pidx := pidx + 1
      params := params ++ s!"      %W{pidx}: {cv2Ty}, %g{pidx}: {ocTy}, %bt{pidx}: {ocTy},\n"
      paramRetTypes := paramRetTypes.push cv2Ty |>.push ocTy |>.push ocTy
      mRetTypes := mRetTypes.push cv2Ty |>.push ocTy |>.push ocTy
      vRetTypes := vRetTypes.push cv2Ty |>.push ocTy |>.push ocTy
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, (h + 1) / 2, (w + 1) / 2]
      | _ => pure ()
    | .unetUp ic oc =>
      -- bilinear ×2 (no params) + concat → 2× convBn ((ic+oc)→oc, oc→oc).
      let ocTy := tensorTy [oc]
      let cv1Ty := tensorTy [oc, ic + oc, 3, 3]
      let cv2Ty := tensorTy [oc, oc, 3, 3]
      params := params ++ s!"      %W{pidx}: {cv1Ty}, %g{pidx}: {ocTy}, %bt{pidx}: {ocTy},\n"
      paramRetTypes := paramRetTypes.push cv1Ty |>.push ocTy |>.push ocTy
      mRetTypes := mRetTypes.push cv1Ty |>.push ocTy |>.push ocTy
      vRetTypes := vRetTypes.push cv1Ty |>.push ocTy |>.push ocTy
      pidx := pidx + 1
      params := params ++ s!"      %W{pidx}: {cv2Ty}, %g{pidx}: {ocTy}, %bt{pidx}: {ocTy},\n"
      paramRetTypes := paramRetTypes.push cv2Ty |>.push ocTy |>.push ocTy
      mRetTypes := mRetTypes.push cv2Ty |>.push ocTy |>.push ocTy
      vRetTypes := vRetTypes.push cv2Ty |>.push ocTy |>.push ocTy
      pidx := pidx + 1
      match curShape with
      | [b, _, h, w] => curShape := [b, oc, h * 2, w * 2]
      | _ => pure ()
    | .maxPool _size stride =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c, (h + stride - 1) / stride, (w + stride - 1) / stride]
      | _ => pure ()
    | .globalAvgPool =>
      match curShape with
      | [b, c, _, _] => curShape := [b, c]
      | _ => pure ()
    | .flatten =>
      match curShape with
      | [b, c, h, w] => curShape := [b, c * h * w]
      | _ => pure ()
    | .patchEmbed ic dim pSize nP =>
      let wTy := tensorTy [dim, ic, pSize, pSize]; let bTy := tensorTy [dim]
      let clsTy := tensorTy [dim]; let posTy := tensorTy [nP + 1, dim]
      params := params ++ s!"      %W{pidx}: {wTy}, %b{pidx}: {bTy},\n"
      paramRetTypes := paramRetTypes.push wTy |>.push bTy
      mRetTypes := mRetTypes.push wTy |>.push bTy
      vRetTypes := vRetTypes.push wTy |>.push bTy
      pidx := pidx + 1
      params := params ++ s!"      %W{pidx}: {clsTy},\n"
      paramRetTypes := paramRetTypes.push clsTy
      mRetTypes := mRetTypes.push clsTy
      vRetTypes := vRetTypes.push clsTy
      pidx := pidx + 1
      params := params ++ s!"      %W{pidx}: {posTy},\n"
      paramRetTypes := paramRetTypes.push posTy
      mRetTypes := mRetTypes.push posTy
      vRetTypes := vRetTypes.push posTy
      pidx := pidx + 1
      match curShape with
      | [b, _, _, _] => curShape := [b, nP + 1, dim]
      | _ => pure ()
    | .transformerEncoder dim _heads mlpDim nBlocks _causal _keepSeq _ _ =>
      let dTy := tensorTy [dim]
      let ddTy := tensorTy [dim, dim]
      let fc1Ty := tensorTy [dim, mlpDim]
      let fc2Ty := tensorTy [mlpDim, dim]
      let mlpBTy := tensorTy [mlpDim]
      for _bi in [:nBlocks] do
        -- LN1
        params := params ++ s!"      %W{pidx}: {dTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push dTy |>.push dTy
        mRetTypes := mRetTypes.push dTy |>.push dTy
        vRetTypes := vRetTypes.push dTy |>.push dTy
        pidx := pidx + 1
        -- Wq, bq
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- Wk, bk
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- Wv, bv
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- Wo, bo
        params := params ++ s!"      %W{pidx}: {ddTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push ddTy |>.push dTy
        mRetTypes := mRetTypes.push ddTy |>.push dTy
        vRetTypes := vRetTypes.push ddTy |>.push dTy
        pidx := pidx + 1
        -- LN2
        params := params ++ s!"      %W{pidx}: {dTy}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push dTy |>.push dTy
        mRetTypes := mRetTypes.push dTy |>.push dTy
        vRetTypes := vRetTypes.push dTy |>.push dTy
        pidx := pidx + 1
        -- Wfc1, bfc1
        params := params ++ s!"      %W{pidx}: {fc1Ty}, %b{pidx}: {mlpBTy},\n"
        paramRetTypes := paramRetTypes.push fc1Ty |>.push mlpBTy
        mRetTypes := mRetTypes.push fc1Ty |>.push mlpBTy
        vRetTypes := vRetTypes.push fc1Ty |>.push mlpBTy
        pidx := pidx + 1
        -- Wfc2, bfc2
        params := params ++ s!"      %W{pidx}: {fc2Ty}, %b{pidx}: {dTy},\n"
        paramRetTypes := paramRetTypes.push fc2Ty |>.push dTy
        mRetTypes := mRetTypes.push fc2Ty |>.push dTy
        vRetTypes := vRetTypes.push fc2Ty |>.push dTy
        pidx := pidx + 1
      -- Final LN
      params := params ++ s!"      %W{pidx}: {dTy}, %b{pidx}: {dTy},\n"
      paramRetTypes := paramRetTypes.push dTy |>.push dTy
      mRetTypes := mRetTypes.push dTy |>.push dTy
      vRetTypes := vRetTypes.push dTy |>.push dTy
      pidx := pidx + 1
      match curShape with
      | [b, _, _] => curShape := [b, dim]
      | _ => pure ()
    | .tokenPositionEmbed v t d _ _ posEmb =>
      let wTy := tensorTy [v, d]
      let posTy := tensorTy [t, d]
      if posEmb then
        params := params ++ s!"      %W{pidx}: {wTy}, %W{pidx + 1}: {posTy},\n"
        paramRetTypes := paramRetTypes.push wTy |>.push posTy
        mRetTypes := mRetTypes.push wTy |>.push posTy
        vRetTypes := vRetTypes.push wTy |>.push posTy
        pidx := pidx + 2
      else
        params := params ++ s!"      %W{pidx}: {wTy},\n"
        paramRetTypes := paramRetTypes.push wTy
        mRetTypes := mRetTypes.push wTy
        vRetTypes := vRetTypes.push wTy
        pidx := pidx + 1
      match curShape with
      | [b, _] => curShape := [b, t, d]
      | _ => pure ()
    | .lmHead d v t =>
      let wTy := tensorTy [d, v]
      let bTy := tensorTy [v]
      params := params ++ s!"      %W{pidx}: {wTy}, %b{pidx}: {bTy},\n"
      paramRetTypes := paramRetTypes.push wTy |>.push bTy
      mRetTypes := mRetTypes.push wTy |>.push bTy
      vRetTypes := vRetTypes.push wTy |>.push bTy
      pidx := pidx + 1
      match curShape with
      | [b, _, _] => curShape := [b, v, t, 1]
      | _ => pure ()
    | .timeCondAdd c nFreq =>
      let wTy := tensorTy [2 * nFreq, c]
      let bTy := tensorTy [c]
      params := params ++ s!"      %W{pidx}: {wTy}, %b{pidx}: {bTy},\n"
      paramRetTypes := paramRetTypes.push wTy |>.push bTy
      mRetTypes := mRetTypes.push wTy |>.push bTy
      vRetTypes := vRetTypes.push wTy |>.push bTy
      pidx := pidx + 1
      -- curShape unchanged (added onto the feature map)
    | _ => pure ()
  -- m_ params (1st moment, same shapes)
  let mut mpidx : Nat := 0
  for l in spec.layers do
    let (pStr, _) := emitLayerParams "m_" l mpidx #[]
    if pStr != "" then
      params := params ++ pStr; mpidx := mpidx + 1
    -- Handle residual/bottleneck/invertedResidual blocks manually
    match l with
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      let gTy := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, blockIc, 3, 3]}, %m_g{mpidx}: {gTy}, %m_bt{mpidx}: {gTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, oc, 3, 3]}, %m_g{mpidx}: {gTy}, %m_bt{mpidx}: {gTy},\n"
        mpidx := mpidx + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, ic, 1, 1]}, %m_g{mpidx}: {gTy}, %m_bt{mpidx}: {gTy},\n"
          mpidx := mpidx + 1
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4; let needsProj := !(ic == oc && firstStride == 1)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, mid, 3, 3]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
        mpidx := mpidx + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, ic, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
          mpidx := mpidx + 1
    | .invertedResidual ic oc expand _ nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
          mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, 3, 3]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
        mpidx := mpidx + 1
    | .mbConv ic oc expand kSize _ nBlocks useSE _act =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
          mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, kSize, kSize]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        if useSE then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [seMid, mid, 1, 1]}, %m_b{mpidx}: {tensorTy [seMid]},\n"
          mpidx := mpidx + 1
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, seMid, 1, 1]}, %m_b{mpidx}: {tensorTy [mid]},\n"
          mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
        mpidx := mpidx + 1
    | .fusedMbConv ic oc expand kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, blockIc, kSize, kSize]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
        if useSE then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [seMid, mid, 1, 1]}, %m_b{mpidx}: {tensorTy [seMid]},\n"
          mpidx := mpidx + 1
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, seMid, 1, 1]}, %m_b{mpidx}: {tensorTy [mid]},\n"
          mpidx := mpidx + 1
        if expand != 1 then
          params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
          mpidx := mpidx + 1
    | .mbConvV3 ic oc expandCh kSize _ useSE _ =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if expandCh != ic then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, ic, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, kSize, kSize]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
      mpidx := mpidx + 1
      if useSE then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [seMid, mid, 1, 1]}, %m_b{mpidx}: {tensorTy [seMid]},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, seMid, 1, 1]}, %m_b{mpidx}: {tensorTy [mid]},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
      mpidx := mpidx + 1
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      let gTyI := tensorTy [ic]; let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if preDWk > 0 then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [ic, 1, preDWk, preDWk]}, %m_g{mpidx}: {gTyI}, %m_bt{mpidx}: {gTyI},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, ic, 1, 1]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
      mpidx := mpidx + 1
      if postDWk > 0 then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [mid, 1, postDWk, postDWk]}, %m_g{mpidx}: {gTyM}, %m_bt{mpidx}: {gTyM},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, mid, 1, 1]}, %m_g{mpidx}: {gTyO}, %m_bt{mpidx}: {gTyO},\n"
      mpidx := mpidx + 1
    | .convNextStage channels nBlocks _norm _act =>
      let c := channels
      let cTy := tensorTy [c]
      let dwTy := tensorTy [c, 1, 7, 7]
      let exTy := tensorTy [4*c, c, 1, 1]; let exB := tensorTy [4*c]
      let pjTy := tensorTy [c, 4*c, 1, 1]
      for _ in [:nBlocks] do
        params := params ++ s!"      %m_W{mpidx}: {dwTy}, %m_b{mpidx}: {cTy},\n"; mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {cTy}, %m_b{mpidx}: {cTy},\n"; mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {exTy}, %m_b{mpidx}: {exB},\n"; mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {pjTy}, %m_b{mpidx}: {cTy},\n"; mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {cTy},\n"; mpidx := mpidx + 1
    | .convNextDownsample ic oc _norm =>
      let icTy := tensorTy [ic]
      let ocTy := tensorTy [oc]
      let cvTy := tensorTy [oc, ic, 2, 2]
      params := params ++ s!"      %m_W{mpidx}: {icTy}, %m_b{mpidx}: {icTy},\n"; mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {cvTy}, %m_b{mpidx}: {ocTy},\n"; mpidx := mpidx + 1
    | .unetDown ic oc =>
      let ocTy := tensorTy [oc]
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, ic, 3, 3]}, %m_g{mpidx}: {ocTy}, %m_bt{mpidx}: {ocTy},\n"; mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, oc, 3, 3]}, %m_g{mpidx}: {ocTy}, %m_bt{mpidx}: {ocTy},\n"; mpidx := mpidx + 1
    | .unetUp ic oc =>
      let ocTy := tensorTy [oc]
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, ic + oc, 3, 3]}, %m_g{mpidx}: {ocTy}, %m_bt{mpidx}: {ocTy},\n"; mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [oc, oc, 3, 3]}, %m_g{mpidx}: {ocTy}, %m_bt{mpidx}: {ocTy},\n"; mpidx := mpidx + 1
    | .patchEmbed ic dim pSize nP =>
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [dim, ic, pSize, pSize]}, %m_b{mpidx}: {tensorTy [dim]},\n"
      mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [dim]},\n"
      mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [nP + 1, dim]},\n"
      mpidx := mpidx + 1
    | .transformerEncoder dim _heads mlpDim nBlocks _causal _keepSeq _ _ =>
      let dTy := tensorTy [dim]
      let ddTy := tensorTy [dim, dim]
      let fc1Ty := tensorTy [dim, mlpDim]
      let fc2Ty := tensorTy [mlpDim, dim]
      let mlpBTy := tensorTy [mlpDim]
      for _bi in [:nBlocks] do
        params := params ++ s!"      %m_W{mpidx}: {dTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {ddTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {dTy}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {fc1Ty}, %m_b{mpidx}: {mlpBTy},\n"
        mpidx := mpidx + 1
        params := params ++ s!"      %m_W{mpidx}: {fc2Ty}, %m_b{mpidx}: {dTy},\n"
        mpidx := mpidx + 1
      params := params ++ s!"      %m_W{mpidx}: {dTy}, %m_b{mpidx}: {dTy},\n"
      mpidx := mpidx + 1
    | .tokenPositionEmbed v t d _ _ posEmb =>
      if posEmb then
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [v, d]}, %m_W{mpidx + 1}: {tensorTy [t, d]},\n"
        mpidx := mpidx + 2
      else
        params := params ++ s!"      %m_W{mpidx}: {tensorTy [v, d]},\n"
        mpidx := mpidx + 1
    | .lmHead d v _ =>
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [d, v]}, %m_b{mpidx}: {tensorTy [v]},\n"
      mpidx := mpidx + 1
    | .timeCondAdd c nFreq =>
      params := params ++ s!"      %m_W{mpidx}: {tensorTy [2 * nFreq, c]}, %m_b{mpidx}: {tensorTy [c]},\n"
      mpidx := mpidx + 1
    | _ => pure ()
  -- v_ params (2nd moment, same shapes)
  let mut vpidx2 : Nat := 0
  for l in spec.layers do
    let (pStr, _) := emitLayerParams "v_" l vpidx2 #[]
    if pStr != "" then
      params := params ++ pStr; vpidx2 := vpidx2 + 1
    match l with
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      let gTy := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, blockIc, 3, 3]}, %v_g{vpidx2}: {gTy}, %v_bt{vpidx2}: {gTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, oc, 3, 3]}, %v_g{vpidx2}: {gTy}, %v_bt{vpidx2}: {gTy},\n"
        vpidx2 := vpidx2 + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, ic, 1, 1]}, %v_g{vpidx2}: {gTy}, %v_bt{vpidx2}: {gTy},\n"
          vpidx2 := vpidx2 + 1
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4; let needsProj := !(ic == oc && firstStride == 1)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, mid, 3, 3]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
        vpidx2 := vpidx2 + 1
        if bi == 0 && needsProj then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, ic, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
          vpidx2 := vpidx2 + 1
    | .invertedResidual ic oc expand _ nBlocks =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
          vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, 3, 3]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
        vpidx2 := vpidx2 + 1
    | .mbConv ic oc expand kSize _ nBlocks useSE _act =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        if expand != 1 then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
          vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, kSize, kSize]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        if useSE then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [seMid, mid, 1, 1]}, %v_b{vpidx2}: {tensorTy [seMid]},\n"
          vpidx2 := vpidx2 + 1
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, seMid, 1, 1]}, %v_b{vpidx2}: {tensorTy [mid]},\n"
          vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
        vpidx2 := vpidx2 + 1
    | .fusedMbConv ic oc expand kSize _ nBlocks useSE =>
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, blockIc, kSize, kSize]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
        if useSE then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [seMid, mid, 1, 1]}, %v_b{vpidx2}: {tensorTy [seMid]},\n"
          vpidx2 := vpidx2 + 1
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, seMid, 1, 1]}, %v_b{vpidx2}: {tensorTy [mid]},\n"
          vpidx2 := vpidx2 + 1
        if expand != 1 then
          params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
          vpidx2 := vpidx2 + 1
    | .mbConvV3 ic oc expandCh kSize _ useSE _ =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if expandCh != ic then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, ic, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, kSize, kSize]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
      vpidx2 := vpidx2 + 1
      if useSE then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [seMid, mid, 1, 1]}, %v_b{vpidx2}: {tensorTy [seMid]},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, seMid, 1, 1]}, %v_b{vpidx2}: {tensorTy [mid]},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
      vpidx2 := vpidx2 + 1
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      let gTyI := tensorTy [ic]; let gTyM := tensorTy [mid]; let gTyO := tensorTy [oc]
      if preDWk > 0 then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [ic, 1, preDWk, preDWk]}, %v_g{vpidx2}: {gTyI}, %v_bt{vpidx2}: {gTyI},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, ic, 1, 1]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
      vpidx2 := vpidx2 + 1
      if postDWk > 0 then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [mid, 1, postDWk, postDWk]}, %v_g{vpidx2}: {gTyM}, %v_bt{vpidx2}: {gTyM},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, mid, 1, 1]}, %v_g{vpidx2}: {gTyO}, %v_bt{vpidx2}: {gTyO},\n"
      vpidx2 := vpidx2 + 1
    | .convNextStage channels nBlocks _norm _act =>
      let c := channels
      let cTy := tensorTy [c]
      let dwTy := tensorTy [c, 1, 7, 7]
      let exTy := tensorTy [4*c, c, 1, 1]; let exB := tensorTy [4*c]
      let pjTy := tensorTy [c, 4*c, 1, 1]
      for _ in [:nBlocks] do
        params := params ++ s!"      %v_W{vpidx2}: {dwTy}, %v_b{vpidx2}: {cTy},\n"; vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {cTy}, %v_b{vpidx2}: {cTy},\n"; vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {exTy}, %v_b{vpidx2}: {exB},\n"; vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {pjTy}, %v_b{vpidx2}: {cTy},\n"; vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {cTy},\n"; vpidx2 := vpidx2 + 1
    | .convNextDownsample ic oc _norm =>
      let icTy := tensorTy [ic]
      let ocTy := tensorTy [oc]
      let cvTy := tensorTy [oc, ic, 2, 2]
      params := params ++ s!"      %v_W{vpidx2}: {icTy}, %v_b{vpidx2}: {icTy},\n"; vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {cvTy}, %v_b{vpidx2}: {ocTy},\n"; vpidx2 := vpidx2 + 1
    | .unetDown ic oc =>
      let ocTy := tensorTy [oc]
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, ic, 3, 3]}, %v_g{vpidx2}: {ocTy}, %v_bt{vpidx2}: {ocTy},\n"; vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, oc, 3, 3]}, %v_g{vpidx2}: {ocTy}, %v_bt{vpidx2}: {ocTy},\n"; vpidx2 := vpidx2 + 1
    | .unetUp ic oc =>
      let ocTy := tensorTy [oc]
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, ic + oc, 3, 3]}, %v_g{vpidx2}: {ocTy}, %v_bt{vpidx2}: {ocTy},\n"; vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [oc, oc, 3, 3]}, %v_g{vpidx2}: {ocTy}, %v_bt{vpidx2}: {ocTy},\n"; vpidx2 := vpidx2 + 1
    | .patchEmbed ic dim pSize nP =>
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [dim, ic, pSize, pSize]}, %v_b{vpidx2}: {tensorTy [dim]},\n"
      vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [dim]},\n"
      vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [nP + 1, dim]},\n"
      vpidx2 := vpidx2 + 1
    | .transformerEncoder dim _heads mlpDim nBlocks _causal _keepSeq _ _ =>
      let dTy := tensorTy [dim]
      let ddTy := tensorTy [dim, dim]
      let fc1Ty := tensorTy [dim, mlpDim]
      let fc2Ty := tensorTy [mlpDim, dim]
      let mlpBTy := tensorTy [mlpDim]
      for _bi in [:nBlocks] do
        params := params ++ s!"      %v_W{vpidx2}: {dTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {ddTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {dTy}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {fc1Ty}, %v_b{vpidx2}: {mlpBTy},\n"
        vpidx2 := vpidx2 + 1
        params := params ++ s!"      %v_W{vpidx2}: {fc2Ty}, %v_b{vpidx2}: {dTy},\n"
        vpidx2 := vpidx2 + 1
      params := params ++ s!"      %v_W{vpidx2}: {dTy}, %v_b{vpidx2}: {dTy},\n"
      vpidx2 := vpidx2 + 1
    | .tokenPositionEmbed v t d _ _ posEmb =>
      if posEmb then
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [v, d]}, %v_W{vpidx2 + 1}: {tensorTy [t, d]},\n"
        vpidx2 := vpidx2 + 2
      else
        params := params ++ s!"      %v_W{vpidx2}: {tensorTy [v, d]},\n"
        vpidx2 := vpidx2 + 1
    | .lmHead d v _ =>
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [d, v]}, %v_b{vpidx2}: {tensorTy [v]},\n"
      vpidx2 := vpidx2 + 1
    | .timeCondAdd c nFreq =>
      params := params ++ s!"      %v_W{vpidx2}: {tensorTy [2 * nFreq, c]}, %v_b{vpidx2}: {tensorTy [c]},\n"
      vpidx2 := vpidx2 + 1
    | _ => pure ()
  if useDdpm then
    -- DDPM: y_target is the noise ε the model learns to predict, same
    -- 4-D shape as the network output. Caller passes `ddpmOutShape`
    -- explicitly because we don't track the post-network shape here.
    let yTy := tensorTy ddpmOutShape
    params := params ++ s!"      %x_flat: {tensorTy [B, inDim]}, %y_ddpm: {yTy},\n"
  else if useYolov1 then
    -- YOLOv1: float target [B, perCell, gridH, gridW] + per-cell float mask
    -- [B, gridH, gridW]. See planning/yolo_demo_v2.md "Phase 1 decisions" D3
    -- (separate ByteArray arg for mask). Channel layout within perCell:
    --   [0..2)   box 0 (x, y)
    --   [2..4)   box 0 (w, h)
    --   [4..5)   box 0 confidence
    --   [5..9)   box 1 (x, y, w, h) — zeroed by "always predictor 0" rule
    --   [9..10)  box 1 confidence — always penalized as no-object
    --   [10..)   per-cell class one-hot (yoloNumClasses long)
    let tgtTy := tensorTy [B, yoloPerCell, yoloGridH, yoloGridW]
    let maskTy := tensorTy [B, yoloGridH, yoloGridW]
    params := params ++ s!"      %x_flat: {tensorTy [B, inDim]}, %y_yolo: {tgtTy}, %m_yolo: {maskTy},\n"
  else if useSeg then
    -- Segmentation: per-pixel int labels (B, H, W) i32. Uses spec.imageH/imageW
    -- assuming the network preserves spatial dims (encoder-decoder).
    params := params ++ s!"      %x_flat: {tensorTy [B, inDim]}, %y_seg: tensor<{B}x{spec.imageH}x{spec.imageW}xi32>,\n"
  else if useSoftLabels then
    -- Soft-label path: caller passes a smoothed/mixed [B, NC] f32 tensor.
    params := params ++ s!"      %x_flat: {tensorTy [B, inDim]}, %y_soft: {tensorTy [B, NC]},\n"
  else
    params := params ++ s!"      %x_flat: {tensorTy [B, inDim]}, %y: tensor<{B}xi32>,\n"
  params := params ++ "      %lr: tensor<f32>, %t: tensor<f32>"
  let mut retTypes := paramRetTypes ++ mRetTypes ++ vRetTypes |>.push "tensor<f32>"
  -- BN stats return types (mean + var per BN layer)
  let bnLayers := collectBnLayers spec
  for (_, oc) in bnLayers do
    retTypes := retTypes.push (tensorTy [oc]) |>.push (tensorTy [oc])
  pure s!"  func.func @main(\n{params}\n    ) -> ({String.intercalate ", " retTypes.toList})"

/-- Generate a full train_step MLIR module with VJPs. When `useSoftLabels`
    is true, the function takes a `%y_soft : [B, NC] f32` tensor (mixup/
    cutmix-style) instead of an int32 `%y : [B]` label vector; label
    smoothing is then expected to be applied by the caller. -/
def generateTrainStep (spec : NetSpec) (batchSize : Nat) (moduleName : String := "jit_train_step")
    (labelSmoothing : Float := 0.1) (weightDecay : Float := 0.0001) (useAdam : Bool := true)
    (useSoftLabels : Bool := false)
    (useFocal : Bool := false) (focalGamma : Float := 2.0)
    (useSeg : Bool := false)
    (useDdpm : Bool := false) (ddpmOutShape : List Nat := [])
    (useYolov1 : Bool := false)
    (yoloGridH : Nat := 7) (yoloGridW : Nat := 7)
    (yoloNumBoxes : Nat := 2) (yoloNumClasses : Nat := 20)
    (gradClipNorm : Float := 0.0) (headLrMult : Float := 1.0)
    (useMuon : Bool := false)
    (useShampoo : Bool := false)
    (segLoss : SegLoss := .ce)
    (useDiouBox : Bool := false)
    (yoloAnchors : List (Float × Float) := [])
    : String :=
  s!"// {spec.name} train_step — Generated by Lean 4 → MLIR (StableHLO + VJPs)\n" ++
  s!"// Batch size: {batchSize}, optimizer: {if useShampoo then "Shampoo (square 2D) + AdamW (rest)" else if useMuon then "Muon (2D) + AdamW (rest)" else if useAdam then "Adam" else "SGD+momentum"}\n" ++
  s!"// label_smoothing: {labelSmoothing}, weight_decay: {weightDecay}, soft_labels: {useSoftLabels}, focal: {useFocal} (γ={focalGamma}), seg: {useSeg}" ++
  (if useSeg then s!" ({repr segLoss})" else "") ++
  s!", ddpm: {useDdpm}, yolov1: {useYolov1}" ++
  (if useYolov1 then s!" (grid={yoloGridH}x{yoloGridW}, B={yoloNumBoxes}, C={yoloNumClasses})" else "") ++
  "\n\n" ++
  s!"module @{moduleName} " ++ "{\n" ++
  emitTrainStepSig spec batchSize useSoftLabels useSeg useDdpm ddpmOutShape
    useYolov1 yoloGridH yoloGridW (if yoloAnchors.isEmpty then yoloNumBoxes * 5 + yoloNumClasses else yoloAnchors.length * 15) ++ " {\n" ++
  emitTrainStepBody spec batchSize moduleName labelSmoothing weightDecay useAdam useSoftLabels useFocal focalGamma useSeg useDdpm
    useYolov1 yoloGridH yoloGridW yoloNumBoxes yoloNumClasses gradClipNorm headLrMult useMuon useShampoo segLoss useDiouBox yoloAnchors ++
  "  }\n" ++
  "}\n"

/-- Standalone segmentation-loss module — exercises `emitSegLossBlock` in
    isolation so the loss and its gradient can be checked against finite
    differences: `@main(logits : [B,NC,H,W], y : [B,H,W] i32) -> (loss, d_logits)`.

    This is the FD harness for `.dice` / `.diceCE`, whose gradient (unlike CE's
    `p - y`) is a real softmax Jacobian-vector product and therefore worth
    checking numerically rather than by inspection. Driven by
    `scripts/seg_loss_probe_check.py`; CPU is fine, no GPU needed. -/
def segLossProbeModule (B NC H W : Nat) (segLoss : SegLoss)
    (labelSmoothing : Float := 0.0) : String := Id.run do
  let logitsTy := tensorTy [B, NC, H, W]
  let yTy := s!"tensor<{B}x{H}x{W}xi32>"
  let mut s := s!"// seg-loss probe: {repr segLoss}, B={B} NC={NC} H={H} W={W}, ls={labelSmoothing}\n"
  s := s ++ "module @seg_loss_probe {\n"
  s := s ++ s!"  func.func @main(%logits: {logitsTy}, %y_seg: {yTy}) -> (tensor<f32>, {logitsTy}) " ++ "{\n"
  -- The loss blocks assume these two constants are already in scope (they are,
  -- inside emitTrainStepBody).
  s := s ++ "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  s := s ++ "    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  s := s ++ emitSegLossBlock B NC H W "%logits" "%y_seg" labelSmoothing segLoss
  s := s ++ s!"    return %loss, %d_logits_seg : tensor<f32>, {logitsTy}\n"
  s := s ++ "  }\n}\n"
  return s


/-- Standalone DIoU box-loss probe: `@main(pred,tgt : [B,4,gH,gW], mask : [B,gH,gW])
    -> (loss, d_pred)`. Compiled + run on CPU by scripts/diou_probe_check.py — the
    emitted forward is checked against numpy and the emitted backward against
    central finite differences, before wiring into the YOLOv1 train step. -/
def diouProbeModule (B gH gW : Nat) (anchorW : Float := 1.0) (anchorH : Float := 1.0) : String := Id.run do
  let p4 := tensorTy [B, 4, gH, gW]
  let maskTy := tensorTy [B, gH, gW]
  let mut s := s!"// diou-loss probe (fwd+bwd): B={B} gH={gH} gW={gW} anchorW={anchorW} anchorH={anchorH}\n"
  s := s ++ "module @diou_probe {\n"
  s := s ++ s!"  func.func @main(%pred: {p4}, %tgt: {p4}, %mask: {maskTy}) -> (tensor<f32>, {p4}) " ++ "{\n"
  s := s ++ "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  s := s ++ emitDiouForward B gH gW "%pred" "%tgt" "%mask" "%dio_loss" anchorW anchorH
  s := s ++ emitDiouBackward B gH gW "%dio_dpred"
  s := s ++ s!"    return %dio_loss, %dio_dpred : tensor<f32>, {p4}\n"
  s := s ++ "  }\n}\n"
  return s

/-- Standalone anchor-YOLO-loss probe: `@main(pred,tgt : [B,A·15,gH,gW],
    mask : [B,A,gH,gW]) -> (loss, d_pred)`. Deterministic test anchors
    (w=0.02+0.03·i, h=0.03+0.04·i), γ=2, λ_box=5 — mirrored in
    scripts/anchor_loss_probe_check.py, which checks the emitted forward against
    numpy and the emitted backward against finite differences. -/
def anchorLossProbeModule (B gH gW A : Nat) : String := Id.run do
  let P := 15
  let apT := tensorTy [B, A * P, gH, gW]
  let anchors : List (Float × Float) := (List.range A).map (fun i =>
    (0.02 + 0.03 * i.toFloat, 0.03 + 0.04 * i.toFloat))
  let mut s := s!"// anchor-loss probe: B={B} gH={gH} gW={gW} A={A}\n"
  s := s ++ "module @anchor_loss_probe {\n"
  s := s ++ s!"  func.func @main(%pred: {apT}, %tgt: {apT}) -> (tensor<f32>, {apT}) " ++ "{\n"
  s := s ++ "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  s := s ++ "    %neginf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  s := s ++ emitAnchorYoloLoss B gH gW anchors "%pred" "%tgt" 2.0 5.0 "%ay_loss" "%ay_grad"
  s := s ++ s!"    return %ay_loss, %ay_grad : tensor<f32>, {apT}\n"
  s := s ++ "  }\n}\n"
  return s

/-- Standalone FPN-neck probe: `@main(C3,C4,C5, W3,W4,W5, dP3,dP4,dP5) ->
    (P3,P4,P5, dC3,dC4,dC5, dW3,dW4,dW5)`. Compiled + run on CPU by
    `scripts/fpn_neck_probe_check.py`, which checks the emitted forward against
    the numpy `fpn_forward` and the emitted backward against the f64-FD-verified
    `fpn_grad` oracle (brick #3). Cotangents are explicit inputs — no scalar-loss
    reduce — so the probe mirrors the real train-step wiring (head backwards feed
    dPn tensors straight into the neck VJP). -/
def fpnNeckProbeModule (B oc c3 c4 c5 g5 : Nat) : String := Id.run do
  let g4 := 2 * g5
  let g3 := 4 * g5
  let c3Ty := tensorTy [B, c3, g3, g3]
  let c4Ty := tensorTy [B, c4, g4, g4]
  let c5Ty := tensorTy [B, c5, g5, g5]
  let w3Ty := s!"tensor<{oc}x{c3}xf32>"
  let w4Ty := s!"tensor<{oc}x{c4}xf32>"
  let w5Ty := s!"tensor<{oc}x{c5}xf32>"
  let p3Ty := tensorTy [B, oc, g3, g3]
  let p4Ty := tensorTy [B, oc, g4, g4]
  let p5Ty := tensorTy [B, oc, g5, g5]
  let (fwd, (p3, p4, p5)) :=
    emitFpnNeckForward "%C3" "%C4" "%C5" "%W3" "%W4" "%W5" B oc c3 c4 c5 g5
  let (bwd, (dc3, dc4, dc5), (dw3, dw4, dw5)) :=
    emitFpnNeckBackward "%dP3" "%dP4" "%dP5" "%C3" "%C4" "%C5" "%W3" "%W4" "%W5" B oc c3 c4 c5 g5
  let mut s := s!"// fpn-neck probe (fwd+bwd): B={B} oc={oc} c3={c3} c4={c4} c5={c5} g5={g5}\n"
  s := s ++ "module @fpn_neck_probe {\n"
  s := s ++ s!"  func.func @main(%C3: {c3Ty}, %C4: {c4Ty}, %C5: {c5Ty}, %W3: {w3Ty}, %W4: {w4Ty}, %W5: {w5Ty}, %dP3: {p3Ty}, %dP4: {p4Ty}, %dP5: {p5Ty}) -> ({p3Ty}, {p4Ty}, {p5Ty}, {c3Ty}, {c4Ty}, {c5Ty}, {w3Ty}, {w4Ty}, {w5Ty}) " ++ "{\n"
  s := s ++ fwd
  s := s ++ bwd
  s := s ++ s!"    return {p3}, {p4}, {p5}, {dc3}, {dc4}, {dc5}, {dw3}, {dw4}, {dw5} : {p3Ty}, {p4Ty}, {p5Ty}, {c3Ty}, {c4Ty}, {c5Ty}, {w3Ty}, {w4Ty}, {w5Ty}\n"
  s := s ++ "  }\n}\n"
  return s

/-- Standalone FlashAttention forward module — exercises `emitFlashAttnSdpa` in
    isolation for validation against dense attention: `@main(Q,K,V : [b,h,n,dh])
    -> O`. Compiled + run in the flash-attn probe (planning/flash_attention.md
    rung 2-3). -/
def flashProbeModule (b heads n dh bk : Nat) (causal : Bool) : String := Id.run do
  let hTy := tensorTy [b, heads, n, dh]
  let (code, oSSA, _) := emitFlashAttnSdpa "p" "%Q" "%K" "%V" b heads n dh bk causal
  let mut s := "module @flash_probe {\n"
  s := s ++ s!"  func.func @main(%Q: {hTy}, %K: {hTy}, %V: {hTy}) -> {hTy} " ++ "{\n"
  s := s ++ code
  s := s ++ s!"    return {oSSA} : {hTy}\n"
  s := s ++ "  }\n}\n"
  return s

/-- Standalone DENSE SDPA forward module `@main(Q,K,V) -> O` (materializes the
    full [b,h,n,n] scores) — the memory-comparison baseline for the flash probe.
    Compile both with `--iree-scheduling-dump-statistics-file` at increasing n to
    see dense peak allocation grow O(n²) while flash stays O(n·bk). -/
def denseSdpaProbeModule (b heads n dh : Nat) (causal : Bool) : String := Id.run do
  let hTy := tensorTy [b, heads, n, dh]
  let sTy := tensorTy [b, heads, n, n]
  let bhnTy := tensorTy [b, heads, n]
  let invSqrtDh : Float := 1.0 / Float.sqrt dh.toFloat
  let mut s := "module @flash_probe {\n"
  s := s ++ s!"  func.func @main(%Q: {hTy}, %K: {hTy}, %V: {hTy}) -> {hTy} " ++ "{\n"
  s := s ++ s!"    %sc = stablehlo.dot_general %Q, %K, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3], precision = [DEFAULT, DEFAULT] : ({hTy}, {hTy}) -> {sTy}\n"
  s := s ++ s!"    %scale = stablehlo.constant dense<{invSqrtDh}> : {sTy}\n"
  s := s ++ s!"    %ss0 = stablehlo.multiply %sc, %scale : {sTy}\n"
  if causal then
    let nnTy := tensorTy [n, n]
    s := s ++ s!"    %ir = stablehlo.iota dim = 0 : {nnTy}\n    %ic = stablehlo.iota dim = 1 : {nnTy}\n"
    s := s ++ s!"    %cmp = stablehlo.compare GE, %ir, %ic, FLOAT : ({nnTy}, {nnTy}) -> tensor<{n}x{n}xi1>\n"
    s := s ++ s!"    %zm = stablehlo.constant dense<0.0> : {nnTy}\n    %nm = stablehlo.constant dense<0xFF800000> : {nnTy}\n"
    s := s ++ s!"    %mask = stablehlo.select %cmp, %zm, %nm : (tensor<{n}x{n}xi1>, {nnTy}, {nnTy}) -> {nnTy}\n"
    s := s ++ s!"    %mbc = stablehlo.broadcast_in_dim %mask, dims = [2, 3] : ({nnTy}) -> {sTy}\n"
    s := s ++ s!"    %ss = stablehlo.add %ss0, %mbc : {sTy}\n"
  else
    s := s ++ s!"    %ss = stablehlo.add %ss0, %ss0 : {sTy}\n    %ssx = stablehlo.subtract %ss, %ss0 : {sTy}\n"
  let ssName := if causal then "%ss" else "%ssx"
  s := s ++ s!"    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
  s := s ++ s!"    %mx = stablehlo.reduce({ssName} init: %ninf) applies stablehlo.maximum across dimensions = [3] : ({sTy}, tensor<f32>) -> {bhnTy}\n"
  s := s ++ s!"    %mxb = stablehlo.broadcast_in_dim %mx, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
  s := s ++ s!"    %sh = stablehlo.subtract {ssName}, %mxb : {sTy}\n    %ex = stablehlo.exponential %sh : {sTy}\n"
  s := s ++ s!"    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n"
  s := s ++ s!"    %se = stablehlo.reduce(%ex init: %zf) applies stablehlo.add across dimensions = [3] : ({sTy}, tensor<f32>) -> {bhnTy}\n"
  s := s ++ s!"    %seb = stablehlo.broadcast_in_dim %se, dims = [0, 1, 2] : ({bhnTy}) -> {sTy}\n"
  s := s ++ s!"    %sm = stablehlo.divide %ex, %seb : {sTy}\n"
  s := s ++ s!"    %o = stablehlo.dot_general %sm, %V, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : ({sTy}, {hTy}) -> {hTy}\n"
  s := s ++ s!"    return %o : {hTy}\n  " ++ "}\n}\n"
  return s

/-- Standalone RoPE module `@main(x) -> rope(x)` (or the VJP with `backward`),
    for op-level validation against jax/demos/rope_ref.py. -/
def ropeProbeModule (b heads n dh : Nat) (backward : Bool) : String := Id.run do
  let hTy := tensorTy [b, heads, n, dh]
  let (tcode, cosSSA, sinSSA) := emitRopeTables "p" n dh
  let (acode, oSSA) := emitRopeApply "p" "%x" cosSSA sinSSA b heads n dh backward
  let mut s := "module @flash_probe {\n"
  s := s ++ s!"  func.func @main(%x: {hTy}) -> {hTy} " ++ "{\n"
  s := s ++ tcode ++ acode
  s := s ++ s!"    return {oSSA} : {hTy}\n"
  s := s ++ "  }\n}\n"
  return s

/-- Standalone FlashAttention forward+backward module: `@main(Q,K,V,dO) ->
    (dQ,dK,dV)`. Runs the forward emit to get O/Lse, then the backward emit;
    validated against the dense-attention VJP. -/
def flashBwdProbeModule (b heads n dh bk : Nat) (causal : Bool) : String := Id.run do
  let hTy := tensorTy [b, heads, n, dh]
  let (fcode, oSSA, lseSSA) := emitFlashAttnSdpa "p" "%Q" "%K" "%V" b heads n dh bk causal
  let (bcode, dqSSA, dkSSA, dvSSA) := emitFlashAttnBwd "p" "%Q" "%K" "%V" "%dO" oSSA lseSSA b heads n dh bk causal
  let mut s := "module @flash_probe {\n"
  s := s ++ s!"  func.func @main(%Q: {hTy}, %K: {hTy}, %V: {hTy}, %dO: {hTy}) -> ({hTy}, {hTy}, {hTy}) " ++ "{\n"
  s := s ++ fcode ++ bcode
  s := s ++ s!"    return {dqSSA}, {dkSSA}, {dvSSA} : {hTy}, {hTy}, {hTy}\n"
  s := s ++ "  }\n}\n"
  return s

end MlirCodegen
