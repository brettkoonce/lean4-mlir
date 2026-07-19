import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.F32Array

/-! Spec → tensor-shape / packed-bytes / parameter-init helpers.

These were copy-pasted into every `Main*Train.lean` file. Centralizing
them here means each trainer just imports `LeanMlir` and asks for
`spec.paramShapes`, `spec.bnShapesBA`, etc.

Anything new added here should be a pure function of `NetSpec` — no IO,
no architecture-specific specialization. Adding a new layer type means
adding a case here once and every trainer picks it up. -/

namespace NetSpec

/-- Per-parameter tensor shapes for the entire spec, in the same order
    `MlirCodegen.emitTrainStepSig` walks them. Used to pack/unpack the
    `params ++ m ++ v` ByteArray that goes into and out of every
    `trainStepAdamF32` call. -/
def paramShapes (spec : NetSpec) : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in spec.layers do
    match l with
    | .conv2d ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc]
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .fpnDetect oc c3 c4 c5 _ A tower =>
      -- Single source of truth for the ordering (Spec.lean) — shared with the
      -- codegen so a mismatch is impossible rather than merely unlikely.
      for sh in fpnDetectParamShapes oc c3 c4 c5 A tower do
        shapes := shapes.push (sh.toArray)
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[oc, blockIc, 3, 3] |>.push #[oc] |>.push #[oc]
        shapes := shapes.push #[oc, oc, 3, 3] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | .bottleneckBlock ic oc nBlocks firstStride =>
      let mid := oc / 4
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, mid, 3, 3] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | .invertedResidual ic oc expand _stride n =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        if expand != 1 then
          shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, 1, 3, 3] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .mbConv ic oc expand kSize _stride n useSE _act =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        if expand != 1 then
          shapes := shapes.push #[mid, blockIc, 1, 1] |>.push #[mid] |>.push #[mid]
        shapes := shapes.push #[mid, 1, kSize, kSize] |>.push #[mid] |>.push #[mid]
        if useSE then
          shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
          shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
        shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .mbConvV3 ic oc expandCh kSize _stride useSE _act =>
      let mid := expandCh
      let seMid := Nat.max 1 (mid / 4)
      if expandCh != ic then
        shapes := shapes.push #[mid, ic, 1, 1] |>.push #[mid] |>.push #[mid]
      shapes := shapes.push #[mid, 1, kSize, kSize] |>.push #[mid] |>.push #[mid]
      if useSE then
        shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
        shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
      shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .fusedMbConv ic oc expand kSize _stride n useSE =>
      for bi in [:n] do
        let blockIc := if bi == 0 then ic else oc
        let mid := if expand == 1 then oc else blockIc * expand
        let seMid := Nat.max 1 (mid / 4)
        shapes := shapes.push #[mid, blockIc, kSize, kSize] |>.push #[mid] |>.push #[mid]
        if useSE then
          shapes := shapes.push #[seMid, mid, 1, 1] |>.push #[seMid]
          shapes := shapes.push #[mid, seMid, 1, 1] |>.push #[mid]
        if expand != 1 then
          shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .uib ic oc expand _stride preDWk postDWk =>
      let mid := ic * expand
      if preDWk > 0 then
        shapes := shapes.push #[ic, 1, preDWk, preDWk] |>.push #[ic] |>.push #[ic]
      shapes := shapes.push #[mid, ic, 1, 1] |>.push #[mid] |>.push #[mid]
      if postDWk > 0 then
        shapes := shapes.push #[mid, 1, postDWk, postDWk] |>.push #[mid] |>.push #[mid]
      shapes := shapes.push #[oc, mid, 1, 1] |>.push #[oc] |>.push #[oc]
    | .convNextStage channels nBlocks _norm _act =>
      -- Per ConvNeXt block: DWConv 7×7 (W, b) + Norm (γ, β) +
      -- 1×1 expand (W, b, c→4c) + 1×1 project (W, b, 4c→c) + LayerScale (γ).
      let c := channels
      for _ in [:nBlocks] do
        shapes := shapes.push #[c, 1, 7, 7] |>.push #[c]
        shapes := shapes.push #[c] |>.push #[c]
        shapes := shapes.push #[4*c, c, 1, 1] |>.push #[4*c]
        shapes := shapes.push #[c, 4*c, 1, 1] |>.push #[c]
        shapes := shapes.push #[c]
    | .convNextDownsample ic oc _norm =>
      -- LN/BN (γ, β) on `ic` channels, then 2×2 conv stride 2 (W, b).
      shapes := shapes.push #[ic] |>.push #[ic]
      shapes := shapes.push #[oc, ic, 2, 2] |>.push #[oc]
    | .convNextStem ic oc p =>
      -- patchify conv p×p stride p (W, b), then channels-first LN (γ, β) on oc.
      shapes := shapes.push #[oc, ic, p, p] |>.push #[oc]
      shapes := shapes.push #[oc] |>.push #[oc]
    | .patchEmbed ic dim p nP =>
      shapes := shapes.push #[dim, ic, p, p] |>.push #[dim]    -- W, b
      shapes := shapes.push #[dim]                              -- cls token
      shapes := shapes.push #[nP + 1, dim]                      -- positional embedding
    | .transformerEncoder dim _heads mlpDim nBlocks _causal _keepSeq _ _ =>
      for _bi in [:nBlocks] do
        -- LN1 (gamma, beta)
        shapes := shapes.push #[dim] |>.push #[dim]
        -- Q, K, V projections (W, b each)
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- Output projection
        shapes := shapes.push #[dim, dim] |>.push #[dim]
        -- LN2
        shapes := shapes.push #[dim] |>.push #[dim]
        -- MLP fc1, fc2
        shapes := shapes.push #[dim, mlpDim] |>.push #[mlpDim]
        shapes := shapes.push #[mlpDim, dim] |>.push #[dim]
      -- Final LN after all blocks
      shapes := shapes.push #[dim] |>.push #[dim]
    | .unetDown ic oc =>
      -- 2× convBn (ic→oc, oc→oc). Maxpool no params.
      shapes := shapes.push #[oc, ic, 3, 3] |>.push #[oc] |>.push #[oc]
      shapes := shapes.push #[oc, oc, 3, 3] |>.push #[oc] |>.push #[oc]
    | .unetUp ic oc =>
      -- 2× convBn ((ic+oc)→oc, oc→oc). Bilinear/concat no params.
      shapes := shapes.push #[oc, ic + oc, 3, 3] |>.push #[oc] |>.push #[oc]
      shapes := shapes.push #[oc, oc, 3, 3] |>.push #[oc] |>.push #[oc]
    | .tokenPositionEmbed v t d _ _ posEmb =>
      -- Embedding W [V, D], then positional [T, D] unless posEmb off. No biases.
      shapes := if posEmb then shapes.push #[v, d] |>.push #[t, d] else shapes.push #[v, d]
    | .lmHead d v _ =>
      -- Dense W [D, V] + bias [V].
      shapes := shapes.push #[d, v] |>.push #[v]
    | .timeCondAdd c nFreq =>
      -- Time-conditioning dense: W [2·nFreq, C] + bias [C].
      shapes := shapes.push #[2 * nFreq, c] |>.push #[c]
    | _ => pure ()
  return shapes

/-- Packed `params ++ m ++ v` shape array (3× the param shapes), as int32 LE
    `ByteArray`. This is what gets passed to `trainStepAdamF32`. -/
def shapesBA (spec : NetSpec) : ByteArray :=
  packShapes (spec.paramShapes ++ spec.paramShapes ++ spec.paramShapes)

/-- BN-layer (pidx, oc) pairs as discovered by the codegen. ViT-style
    transformer specs return an empty array. -/
def bnLayers (spec : NetSpec) : Array (Nat × Nat) :=
  MlirCodegen.collectBnLayers spec

/-- Total float count needed to store running BN stats (mean + var per
    BN layer). 0 for ViT and any non-BN architecture. -/
def nBnStats (spec : NetSpec) : Nat :=
  spec.bnLayers.foldl (fun acc (_, oc) => acc + oc * 2) 0

/-- BN shapes packed for the FFI: `[n_bn_layers, oc0, oc1, ...]` as int32 LE.
    The `trainStepAdamF32` FFI uses this to know how many BN-stat outputs
    to pop after the params/loss. -/
def bnShapesBA (spec : NetSpec) : ByteArray := Id.run do
  let push := fun (ba : ByteArray) (v : Nat) =>
    let v32 : UInt32 := v.toUInt32
    ba.push (v32 &&& 0xFF).toUInt8
      |>.push ((v32 >>> 8) &&& 0xFF).toUInt8
      |>.push ((v32 >>> 16) &&& 0xFF).toUInt8
      |>.push ((v32 >>> 24) &&& 0xFF).toUInt8
  let bn := spec.bnLayers
  let mut ba := push .empty bn.size
  for (_, oc) in bn do ba := push ba oc
  return ba

/-- Param shapes for the eval forward pass: regular params followed by
    one `[oc]` mean and `[oc]` var per BN layer. ViT-style specs collapse
    this to just `paramShapes` because there are no BN layers. -/
def evalShapes (spec : NetSpec) : Array (Array Nat) := Id.run do
  let mut shapes := spec.paramShapes
  for (_, oc) in spec.bnLayers do
    shapes := shapes.push #[oc] |>.push #[oc]
  return shapes

/-- Packed `evalShapes` as int32 LE for the eval forward FFI. -/
def evalShapesBA (spec : NetSpec) : ByteArray :=
  packShapes spec.evalShapes

/-- Packed input-tensor shape for a flat-image batch (NCHW collapsed
    to `[batch, channels*H*W]`). Channel count comes from the first
    conv-style layer; defaults to 1 for pure-MLP specs. -/
def xShape (spec : NetSpec) (batch : Nat) : ByteArray :=
  packXShape #[batch, MlirCodegen.inputFlatDim spec]

/-- Sanitized base name for the spec — same transformation the codegen
    applies when generating MLIR module names. -/
def sanitizedName (spec : NetSpec) : String :=
  MlirCodegen.sanitize spec.name

/-- The eval forward function name to pass to `forwardF32`. The codegen
    emits modules of the form `@<sanitized_name>_eval` containing
    `func.func @forward_eval`, so the FFI call wants the qualified
    `"<sanitized_name>_eval.forward_eval"`. Computing this from the
    spec means trainers can never get the spelling wrong by hand. -/
def evalFnName (spec : NetSpec) : String :=
  spec.sanitizedName ++ "_eval.forward_eval"

/-- Emit He-initialized (W) + zero-bias for one conv+bias group. -/
private def heConvB (oc ic k : Nat) (seed : USize) : IO (ByteArray × ByteArray × USize) := do
  let W ← F32.heInit seed (oc*ic*k*k).toUSize (Float.sqrt (2.0 / (ic*k*k).toFloat))
  let b ← F32.const oc.toUSize 0.0
  return (W, b, seed + 1)

/-- Emit He-initialized (W) + γ=1 + β=0 for one convBn group. -/
private def heConvBn (oc ic k : Nat) (seed : USize) : IO (ByteArray × ByteArray × ByteArray × USize) := do
  let W ← F32.heInit seed (oc*ic*k*k).toUSize (Float.sqrt (2.0 / (ic*k*k).toFloat))
  let g ← F32.const oc.toUSize 1.0
  let b ← F32.const oc.toUSize 0.0
  return (W, g, b, seed + 1)

/-- Emit He-initialized (W) + zero-bias for one dense layer. -/
private def heDense (fi fo : Nat) (seed : USize) : IO (ByteArray × ByteArray × USize) := do
  let W ← F32.heInit seed (fi*fo).toUSize (Float.sqrt (2.0 / fi.toFloat))
  let b ← F32.const fo.toUSize 0.0
  return (W, b, seed + 1)

/-- Emit γ=1, β=0 for one LayerNorm. -/
private def heLN (dim : Nat) : IO (ByteArray × ByteArray) := do
  let g ← F32.const dim.toUSize 1.0
  let b ← F32.const dim.toUSize 0.0
  return (g, b)

/-- He-initialize one layer's parameters. Dispatches per layer constructor
    so semantics (bias vs γ, cls token vs LN start, etc.) are unambiguous —
    no more shape-peek heuristic that misfires at patchEmbed or
    transformer-block boundaries. -/
private def heInitLayer (l : Layer) (seed : USize) : IO (Array ByteArray × USize) := do
  match l with
  | .dense fi fo _ =>
    let (W, b, s') ← heDense fi fo seed
    return (#[W, b], s')
  | .conv2d ic oc k _ _ =>
    let (W, b, s') ← heConvB oc ic k seed
    return (#[W, b], s')
  | .fpnDetect oc c3 c4 c5 _ A tower =>
    -- Walk the canonical shape list (Spec.lean) so init can never disagree with
    -- the signature. Rank-2 [o,i] and rank-4 [o,i,k,k] are WEIGHTS → He-init
    -- (heConvB's W bytes are the [o,i] 2D weight when k = 1); rank-1 are BIASES
    -- → zero.
    --
    -- Zero head biases reproduce the biasless head exactly, so the bias codegen
    -- is a no-op until `applyDetPriorBias` installs the prior — that is what
    -- keeps "add a bias" and "start it at the RetinaNet prior" separable levers.
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for sh in fpnDetectParamShapes oc c3 c4 c5 A tower do
      match sh with
      | [n] => parts := parts.push (← F32.const (n.toUSize) 0.0)
      | [o, i] =>
        let (W, _b, s') ← heConvB o i 1 s
        parts := parts.push W
        s := s'
      | [o, i, k, _] =>
        let (W, _b, s') ← heConvB o i k s
        parts := parts.push W
        s := s'
      | _ => pure ()
    return (parts, s)
  | .convBn ic oc k _ _ =>
    let (W, g, b, s') ← heConvBn oc ic k seed
    return (#[W, g, b], s')
  | .residualBlock ic oc nBlocks firstStride =>
    let needsProj := !(ic == oc && firstStride == 1)
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:nBlocks] do
      let blockIc := if bi == 0 then ic else oc
      let (W1, g1, b1, s') ← heConvBn oc blockIc 3 s
      parts := parts.push W1 |>.push g1 |>.push b1
      let (W2, g2, b2, s'') ← heConvBn oc oc 3 s'
      parts := parts.push W2 |>.push g2 |>.push b2
      s := s''
      if bi == 0 && needsProj then
        let (Wp, gp, bp, s''') ← heConvBn oc ic 1 s
        parts := parts.push Wp |>.push gp |>.push bp
        s := s'''
    return (parts, s)
  | .bottleneckBlock ic oc nBlocks firstStride =>
    let mid := oc / 4
    let needsProj := !(ic == oc && firstStride == 1)
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:nBlocks] do
      let blockIc := if bi == 0 then ic else oc
      let (W1, g1, b1, s') ← heConvBn mid blockIc 1 s
      parts := parts.push W1 |>.push g1 |>.push b1
      let (W2, g2, b2, s'') ← heConvBn mid mid 3 s'
      parts := parts.push W2 |>.push g2 |>.push b2
      let (W3, g3, b3, s''') ← heConvBn oc mid 1 s''
      parts := parts.push W3 |>.push g3 |>.push b3
      s := s'''
      if bi == 0 && needsProj then
        let (Wp, gp, bp, sp) ← heConvBn oc ic 1 s
        parts := parts.push Wp |>.push gp |>.push bp
        s := sp
    return (parts, s)
  | .invertedResidual ic oc expand _stride n =>
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:n] do
      let blockIc := if bi == 0 then ic else oc
      let mid := blockIc * expand
      if expand != 1 then
        let (W, g, b, s') ← heConvBn mid blockIc 1 s
        parts := parts.push W |>.push g |>.push b
        s := s'
      let (Wdw, gdw, bdw, s') ← heConvBn mid 1 3 s
      parts := parts.push Wdw |>.push gdw |>.push bdw
      let (Wp, gp, bp, s'') ← heConvBn oc mid 1 s'
      parts := parts.push Wp |>.push gp |>.push bp
      s := s''
    return (parts, s)
  | .mbConv ic oc expand kSize _stride n useSE _act =>
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:n] do
      let blockIc := if bi == 0 then ic else oc
      let mid := blockIc * expand
      let seMid := Nat.max 1 (mid / 4)
      if expand != 1 then
        let (W, g, b, s') ← heConvBn mid blockIc 1 s
        parts := parts.push W |>.push g |>.push b
        s := s'
      let (Wdw, gdw, bdw, s') ← heConvBn mid 1 kSize s
      parts := parts.push Wdw |>.push gdw |>.push bdw
      s := s'
      if useSE then
        let (Wsq, bsq, s'') ← heConvB seMid mid 1 s
        parts := parts.push Wsq |>.push bsq
        s := s''
        let (Wex, bex, s''') ← heConvB mid seMid 1 s
        parts := parts.push Wex |>.push bex
        s := s'''
      let (Wp, gp, bp, sp) ← heConvBn oc mid 1 s
      parts := parts.push Wp |>.push gp |>.push bp
      s := sp
    return (parts, s)
  | .mbConvV3 ic oc expandCh kSize _stride useSE _act =>
    let mid := expandCh
    let seMid := Nat.max 1 (mid / 4)
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    if expandCh != ic then
      let (W, g, b, s') ← heConvBn mid ic 1 s
      parts := parts.push W |>.push g |>.push b
      s := s'
    let (Wdw, gdw, bdw, s') ← heConvBn mid 1 kSize s
    parts := parts.push Wdw |>.push gdw |>.push bdw
    s := s'
    if useSE then
      let (Wsq, bsq, s'') ← heConvB seMid mid 1 s
      parts := parts.push Wsq |>.push bsq
      s := s''
      let (Wex, bex, s''') ← heConvB mid seMid 1 s
      parts := parts.push Wex |>.push bex
      s := s'''
    let (Wp, gp, bp, sp) ← heConvBn oc mid 1 s
    parts := parts.push Wp |>.push gp |>.push bp
    return (parts, sp)
  | .fusedMbConv ic oc expand kSize _stride n useSE =>
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for bi in [:n] do
      let blockIc := if bi == 0 then ic else oc
      let mid := if expand == 1 then oc else blockIc * expand
      let seMid := Nat.max 1 (mid / 4)
      let (Wf, gf, bf, s') ← heConvBn mid blockIc kSize s
      parts := parts.push Wf |>.push gf |>.push bf
      s := s'
      if useSE then
        let (Wsq, bsq, s'') ← heConvB seMid mid 1 s
        parts := parts.push Wsq |>.push bsq
        s := s''
        let (Wex, bex, s''') ← heConvB mid seMid 1 s
        parts := parts.push Wex |>.push bex
        s := s'''
      if expand != 1 then
        let (Wp, gp, bp, sp) ← heConvBn oc mid 1 s
        parts := parts.push Wp |>.push gp |>.push bp
        s := sp
    return (parts, s)
  | .uib ic oc expand _stride preDWk postDWk =>
    let mid := ic * expand
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    if preDWk > 0 then
      let (Wp, gp, bp, s') ← heConvBn ic 1 preDWk s
      parts := parts.push Wp |>.push gp |>.push bp
      s := s'
    let (We, ge, be, s2) ← heConvBn mid ic 1 s
    parts := parts.push We |>.push ge |>.push be
    s := s2
    if postDWk > 0 then
      let (Wp2, gp2, bp2, s3) ← heConvBn mid 1 postDWk s
      parts := parts.push Wp2 |>.push gp2 |>.push bp2
      s := s3
    let (Wj, gj, bj, s4) ← heConvBn oc mid 1 s
    parts := parts.push Wj |>.push gj |>.push bj
    return (parts, s4)
  | .patchEmbed ic dim p nP =>
    -- Conv W, bias=0, cls=0, pos=He init (fan-in is nP+1).
    let W ← F32.heInit seed (dim*ic*p*p).toUSize (Float.sqrt (2.0 / (ic*p*p).toFloat))
    let b ← F32.const dim.toUSize 0.0
    let cls ← F32.const dim.toUSize 0.0
    let pos ← F32.heInit (seed + 1) ((nP+1)*dim).toUSize (Float.sqrt (2.0 / (nP+1).toFloat))
    return (#[W, b, cls, pos], seed + 2)
  | .convNextStage channels nBlocks _norm _act =>
    -- Per block: DW (W, b) + LN (γ, β) + 1×1 expand (W, b) + 1×1 project (W, b) + LayerScale (γ).
    -- LayerScale init: 1e-6 per the paper (small residual contribution at init).
    let c := channels
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for _ in [:nBlocks] do
      let (Wdw, bdw, s1) ← heConvB c 1 7 s
      parts := parts.push Wdw |>.push bdw
      let (gLN, bLN) ← heLN c
      parts := parts.push gLN |>.push bLN
      let (Wex, bex, s2) ← heConvB (4*c) c 1 s1
      parts := parts.push Wex |>.push bex
      let (Wpj, bpj, s3) ← heConvB c (4*c) 1 s2
      parts := parts.push Wpj |>.push bpj
      let lsGamma ← F32.const c.toUSize 0.000001
      parts := parts.push lsGamma
      s := s3
    return (parts, s)
  | .convNextDownsample ic oc _norm =>
    -- LN (γ, β) on `ic` channels + 2×2 stride-2 conv (W, b).
    let (gLN, bLN) ← heLN ic
    let (Wcv, bcv, s') ← heConvB oc ic 2 seed
    return (#[gLN, bLN, Wcv, bcv], s')
  | .convNextStem ic oc p =>
    -- patchify p×p conv (W, b) + channels-first LN (γ, β) on oc.
    let (Wcv, bcv, s') ← heConvB oc ic p seed
    let (gLN, bLN) ← heLN oc
    return (#[Wcv, bcv, gLN, bLN], s')
  | .transformerEncoder dim _heads mlpDim nBlocks _causal _keepSeq _ _ =>
    let mut parts : Array ByteArray := #[]
    let mut s := seed
    for _bi in [:nBlocks] do
      let (g1, b1) ← heLN dim
      parts := parts.push g1 |>.push b1
      let (Wq, bq, s') ← heDense dim dim s
      parts := parts.push Wq |>.push bq
      let (Wk, bk, s'') ← heDense dim dim s'
      parts := parts.push Wk |>.push bk
      let (Wv, bv, s''') ← heDense dim dim s''
      parts := parts.push Wv |>.push bv
      let (Wo, bo, s4) ← heDense dim dim s'''
      parts := parts.push Wo |>.push bo
      let (g2, b2) ← heLN dim
      parts := parts.push g2 |>.push b2
      let (Wfc1, bfc1, s5) ← heDense dim mlpDim s4
      parts := parts.push Wfc1 |>.push bfc1
      let (Wfc2, bfc2, s6) ← heDense mlpDim dim s5
      parts := parts.push Wfc2 |>.push bfc2
      s := s6
    -- Final LN after all blocks
    let (gf, bf) ← heLN dim
    parts := parts.push gf |>.push bf
    return (parts, s)
  | .unetDown ic oc =>
    -- 2× convBn (ic→oc, oc→oc). Maxpool no params.
    let (W1, g1, b1, s1) ← heConvBn oc ic 3 seed
    let (W2, g2, b2, s2) ← heConvBn oc oc 3 s1
    return (#[W1, g1, b1, W2, g2, b2], s2)
  | .unetUp ic oc =>
    -- 2× convBn ((ic+oc)→oc, oc→oc). Bilinear/concat no params.
    let (W1, g1, b1, s1) ← heConvBn oc (ic + oc) 3 seed
    let (W2, g2, b2, s2) ← heConvBn oc oc 3 s1
    return (#[W1, g1, b1, W2, g2, b2], s2)
  | .tokenPositionEmbed v t d _ _ posEmb =>
    -- Token embedding ~ N(0, 0.02²) (GPT-2 / nano-GPT convention),
    -- learned positional embedding (if present) initialized the same way.
    let emb ← F32.heInit seed (v * d).toUSize 0.02
    if posEmb then
      let pos ← F32.heInit (seed + 1) (t * d).toUSize 0.02
      return (#[emb, pos], seed + 2)
    else
      return (#[emb], seed + 1)
  | .timeCondAdd c nFreq =>
    -- Init W and b to ZERO so time conditioning starts as a no-op (pure
    -- residual) and grows in — the layer-scale trick the DDPM v2 plan
    -- wants (Workstream A / C). d_W = embᵀ·d_proj ≠ 0, so it trains.
    let w ← F32.const (2 * nFreq * c).toUSize 0.0
    let b ← F32.const c.toUSize 0.0
    return (#[w, b], seed + 1)
  | .lmHead d v _ =>
    -- Dense W [D, V] He-init; zero bias.
    let (W, b, s') ← heDense d v seed
    return (#[W, b], s')
  | _ =>
    -- Unsupported layer (no trainable params OR not in paramShapes).
    -- flatten / maxPool / globalAvgPool / reshape have no params → empty.
    return (#[], seed)

/-- He-initialize all parameters for a spec, walking layer-by-layer. -/
def heInitParams (spec : NetSpec) : IO ByteArray := do
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  for l in spec.layers do
    let (parts, s') ← heInitLayer l seed
    paramParts := paramParts ++ parts
    seed := s'
  return F32.concat paramParts

/-- Overwrite the head's bias with `log priors[c]` — RetinaNet's prior-bias
    init (Lin et al. §3.3, "prior"), for segmentation heads.

    `heInitParams` lays layers out in order and emits each conv's bias directly
    after its weights, so the head's bias is the **final `NC` floats** of the
    buffer. That is what this patches; it is a no-op on everything else.

    **What it buys, and why it is focal's other half.** A zero-bias head starts
    at a uniform softmax: every class at `1/NC`, background included. The net's
    first job is therefore to discover the class prior, and on BraTS it does
    that by walking straight into the trivial predictor — the collapse is
    decided in the first ~100 steps (`planning/brats_demo.md` Workstream A). A
    `log π_c` bias hands it the prior at step 0 instead, so the first gradient
    step is spent on the actual task.

    The quantitative version is in `scripts/seg_grad_scorecard.py`, whose sweep
    lands on this exact row. Prior-bias init starts the net at
    `z₀ - z₃ = log(π₀/π₃) = log(0.9746/0.0050) = 5.27` (verified against the
    emitted checkpoint: `softmax(head bias) == π` to 2e-09). Its measured
    balance ratio — rare-class gradient over majority gradient — reads:

    | (C) at | ce | dice | wce | focal |
    |---|---|---|---|---|
    | `z0 = 0` (uniform) | 5.09e-03 | 2.84e-02 | 9.96e-01 | **5.15e-03** |
    | `z0 = 5.27` (this) | 2.60e-01 | 1.12e-01 | 5.08e+01 | **9.90e+01** |

    **One bias vector is worth ~19,000× to focal, at step 0** — and flips it
    from the worst arm (tied with CE, a literal no-op) to the best (~2× wce).
    That is the whole content of "focal needs confidence to suppress": this
    manufactures the confidence up front instead of waiting for training to
    produce it too late. It is why Lin et al. ship the two together; they are
    one idea, and reading them as separate tricks is how the pairing gets lost.

    It helps every arm — wce's ratio rises 51× too — because starting at the
    prior is simply a better starting point than uniform. focal is the one that
    goes from *inert* to *leading*.

    `priors` need not be normalized: adding a constant to every logit is a
    no-op under softmax, so an overall scale on `priors` shifts every bias by
    `log k` and changes nothing. Only the ratios matter — the same
    scale-invariance that `perPixelWeightedCE`'s reduction enjoys, for the same
    reason.

    Returns a NEW ByteArray. -/
def applyHeadPriorBias (spec : NetSpec) (params : ByteArray)
    (priors : List Float) : IO ByteArray := do
  let nc := (spec.layers.getLast?.map (·.outChannels)).getD 0
  if priors.length != nc then
    throw <| IO.userError s!"headPriorBias: got {priors.length} priors for a {nc}-class head — they must match 1:1"
  if priors.any (· <= 0.0) then
    throw <| IO.userError "headPriorBias: priors must be strictly positive (log 0 = -inf would hard-zero the class, which is the collapse we are trying to prevent, installed by hand)"
  let total := params.size / 4
  if total < nc then
    throw <| IO.userError s!"headPriorBias: {total} params is smaller than the {nc}-float head bias"
  -- No scalar-store primitive on the F32 side, and none is worth adding for
  -- one vector written once per run: build the bias block and splice it onto
  -- the truncated buffer.
  let mut biasParts : Array ByteArray := #[]
  for pi in priors do
    biasParts := biasParts.push (← F32.const (1 : USize) (Float.log pi))
  return (params.extract 0 ((total - nc) * 4)).append (F32.concat biasParts)

/-- RetinaNet prior-bias init for the **FPN detector head** (planning/yolo_fpn.md
    Tier 2). Sets every objectness logit's bias to `−log((1−π)/π)` so the head
    starts predicting `sigmoid = π` (π ≈ 0.01) on every cell; box and class biases
    stay at zero.

    This is the classifier trick of `applyHeadPriorBias` transposed to a
    sigmoid/one-vs-all head, and it is aimed at a measured failure rather than a
    guess. On the e12 run every objectness logit sat in ≈[−2.7, −1.2] (p5..p95)
    with pos/neg means −1.549/−1.803: the head had real signal (AUC 0.742) but
    almost no dynamic range, because a bias-free 1×1 conv has to synthesize the
    constant background offset out of weights that also have to discriminate. The
    bias hands it that constant for free, which is the whole point — and per-class
    mAP is bounded by objectness *ranking*, so this is the lever that can move it
    (both Tier-1 levers were measured out; see the T1a/T1b write-ups).

    Assumes the `.fpnDetect` layer is last, so its 3 `[A·15]` biases are the final
    `3·A·15` floats of the buffer — the same tail-splice `applyHeadPriorBias` does.
    Within each `[A·15]` block, anchor `a`'s objectness is channel `a·15 + 4`
    (`emitAnchorYoloLoss` slices box at `base..base+4`, obj at `base+4`, class at
    `base+5..base+15`).

    Returns a NEW ByteArray. -/
def applyDetPriorBias (spec : NetSpec) (params : ByteArray) (pi : Float) : IO ByteArray := do
  if pi <= 0.0 || pi >= 1.0 then
    throw <| IO.userError s!"detPriorBias: π must be in (0,1), got {pi}"
  let A ← match spec.layers.getLast? with
    | some (.fpnDetect _ _ _ _ _ A _) => pure A
    | _ => throw <| IO.userError "detPriorBias: last layer is not .fpnDetect (the head this init targets)"
  let ap := A * 15
  let total := params.size / 4
  if total < 3 * ap then
    throw <| IO.userError s!"detPriorBias: {total} params is smaller than the {3 * ap}-float head-bias tail"
  let b0 := -(Float.log ((1.0 - pi) / pi))
  -- One [A·15] block: b0 on each anchor's objectness channel, 0 elsewhere.
  let mut blockParts : Array ByteArray := #[]
  for c in [0:ap] do
    blockParts := blockParts.push (← F32.const (1 : USize) (if c % 15 == 4 then b0 else 0.0))
  let block := F32.concat blockParts
  return (params.extract 0 ((total - 3 * ap) * 4)).append (F32.concat #[block, block, block])

/-- Patch the first `prefixBytes` of `initParams` with bytes read from
    `pretrainedPath`. Used to bootstrap a fresh init from a pretrained
    backbone — e.g. load R34-Imagenette weights into a YOLOv1 init,
    keeping the YOLOv1 head's He-init untouched.

    `prefixBytes` is computed by the caller (usually
    `4 * (spec.totalParams - <last layer fanOut + fanIn*fanOut>)` for a
    spec whose final dense layer differs from the pretrained source).

    Returns a NEW ByteArray; `initParams` is not modified. -/
def patchInitWithPretrainedPrefix (initParams : ByteArray) (pretrainedPath : String)
    (prefixBytes : Nat) : IO ByteArray := do
  let pre ← IO.FS.readBinFile pretrainedPath
  if pre.size < prefixBytes then
    throw <| IO.userError s!"pretrained checkpoint {pretrainedPath} has {pre.size} bytes; need at least {prefixBytes}"
  if initParams.size < prefixBytes then
    throw <| IO.userError s!"init params has {initParams.size} bytes; can't patch {prefixBytes}"
  let head := pre.extract 0 prefixBytes
  let tail := initParams.extract prefixBytes initParams.size
  return head ++ tail

end NetSpec
