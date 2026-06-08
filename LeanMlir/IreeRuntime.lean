/-! Lean FFI bindings for the IREE runtime.

    Links to `libiree_ffi.so` (thin wrapper) + IREE runtime via the Lean shim
    in `ffi/iree_lean_ffi.c`. Exposes:
      - `IreeSession.create` — load a .vmfb, bind to CUDA device
      - `IreeSession.mlpForward` — MLP-specific forward pass (MNIST shape) -/

/-- Opaque handle to an IREE runtime session (module + device). -/
private opaque IreeSessionPointed : NonemptyType
def IreeSession : Type := IreeSessionPointed.type
instance : Nonempty IreeSession := IreeSessionPointed.property

namespace IreeSession

/-- Load a `.vmfb` bytecode module onto the default CUDA device. -/
@[extern "lean_iree_session_create"]
opaque create (path : @& String) : IO IreeSession

/-- Run MNIST-MLP forward pass. Shapes are fixed:
    `x` is `batch×784`, `W0` is `784×512`, `b0` is `512`,
    `W1` is `512×512`, `b1` is `512`, `W2` is `512×10`, `b2` is `10`.
    Returns the logits as a `batch×10` flattened `FloatArray`. -/
@[extern "lean_iree_mlp_forward"]
opaque mlpForward
  (sess : @& IreeSession)
  (x : @& FloatArray)
  (W0 : @& FloatArray) (b0 : @& FloatArray)
  (W1 : @& FloatArray) (b1 : @& FloatArray)
  (W2 : @& FloatArray) (b2 : @& FloatArray)
  (batch : USize) : IO FloatArray

/-- Run one SGD training step. Params packed into a single FloatArray of
    length 669706 in order `W0|b0|W1|b1|W2|b2`. Labels are a ByteArray of
    `4*batch` bytes (int32 LE). Returns new params + loss as a single
    FloatArray of length 669707; `result[669706]` is the loss. -/
@[extern "lean_iree_mlp_train_step"]
opaque mlpTrainStep
  (sess : @& IreeSession)
  (params : @& FloatArray)
  (x : @& FloatArray)
  (y : @& ByteArray)
  (lr : Float)
  (batch : USize) : IO FloatArray

/-- Generic train step. Shapes are packed ByteArrays (see `packShapes`). -/
@[extern "lean_iree_train_step_packed"]
opaque trainStepPacked
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& FloatArray) (shapes : @& ByteArray)
  (x : @& FloatArray) (xShape : @& ByteArray)
  (y : @& ByteArray)
  (lr : Float) (batch : USize) : IO FloatArray

/-- Zero-copy f32 train step. All tensors are ByteArray (raw float32 bytes).
    No Float64↔Float32 conversion at the boundary. -/
@[extern "lean_iree_train_step_f32"]
opaque trainStepF32
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (y : @& ByteArray)
  (lr : Float) (batch : USize) : IO ByteArray

/-- Adam train step (f32). Passes step counter t for bias correction.
    Params = weights ++ m ++ v. Returns params ++ loss ++ BN stats.
    bnShapes: packed [n_bn_layers, oc0, oc1, ...] for BN stat output sizes. -/
@[extern "lean_iree_train_step_adam_f32"]
opaque trainStepAdamF32
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (y : @& ByteArray)
  (lr : Float) (t : Float)
  (bnShapes : @& ByteArray)
  (batch : USize) : IO ByteArray

/-- Soft-label variant: `ySoft` is a `[batch, nClasses]` f32 tensor
    (smoothed + mixed). Routes to the codegen produced with
    `useSoftLabels := true`. Used by the mixup/cutmix path. -/
@[extern "lean_iree_train_step_adam_f32_softlabel"]
opaque trainStepAdamF32Soft
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (ySoft : @& ByteArray)
  (lr : Float) (t : Float)
  (bnShapes : @& ByteArray)
  (batch : USize) (nClasses : USize) : IO ByteArray

/-- Per-pixel segmentation variant: `ySeg` is an int32 `[batch, H, W]`
    per-pixel label tensor. Routes to the codegen produced with
    `useSeg := true`. -/
@[extern "lean_iree_train_step_adam_f32_seg"]
opaque trainStepAdamF32Seg
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (ySeg : @& ByteArray)
  (lr : Float) (t : Float)
  (bnShapes : @& ByteArray)
  (batch : USize) (H : USize) (W : USize) : IO ByteArray

/-- DDPM variant: `yDdpm` is a `[batch, C, H, W]` f32 tensor — the
    target ε noise the model learns to predict. Routes to the codegen
    produced with `useDdpm := true`. Loss is per-pixel MSE. -/
@[extern "lean_iree_train_step_adam_f32_ddpm"]
opaque trainStepAdamF32Ddpm
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (yDdpm : @& ByteArray)
  (lr : Float) (t : Float)
  (bnShapes : @& ByteArray)
  (batch : USize) (outC : USize) (outH : USize) (outW : USize) : IO ByteArray

/-- YOLOv1 variant. `yYolo` is a `[batch, perCell, gridH, gridW]` f32
    target tensor (NCHW); `mYolo` is a `[batch, gridH, gridW]` f32
    per-cell objectness mask (1.0 where a GT box's center falls in
    the cell, 0.0 otherwise). Routes to the codegen produced with
    `useYolov1 := true`. Loss is the 5-term masked MSE described in
    `planning/yolo_demo_v2.md` Phase 1.

    `perCell = numBoxes * 5 + numClasses`. For VOC this is
    `2*5 + 20 = 30`; `gridH = gridW = 7`. -/
@[extern "lean_iree_train_step_adam_f32_yolov1"]
opaque trainStepAdamF32Yolov1
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (yYolo : @& ByteArray)
  (mYolo : @& ByteArray)
  (lr : Float) (t : Float)
  (bnShapes : @& ByteArray)
  (batch : USize) (gridH : USize) (gridW : USize) (perCell : USize) : IO ByteArray

/-- Zero-copy f32 forward pass. Pushes x then param tensors, returns logits.
    For inference/eval — no y, lr, or velocity inputs. -/
@[extern "lean_iree_forward_f32"]
opaque forwardF32
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (batch : USize) (nClasses : USize) : IO ByteArray

/-- Drive the **verified-renderer** `@linear_train_step`
    (`StableHLO.linearTrainStepModuleV`) through the generic IREE invoke.
    Inputs are raw f32 ByteArrays: `x` is `batch×d₀`, `W0` is `d₀×d₁`, `b0`
    is `d₁`; `y` is int32 `[batch]` (the one-hot is built in the C shim).
    Returns `W0n (d₀·d₁ f32) ++ b0n (d₁ f32)`. -/
@[extern "lean_iree_linear_train_step"]
opaque linearTrainStepV
  (sess : @& IreeSession) (fnName : @& String)
  (x : @& ByteArray) (W0 : @& ByteArray) (b0 : @& ByteArray) (y : @& ByteArray)
  (batch : USize) (d0 : USize) (d1 : USize) : IO ByteArray

/-- Drive the **verified-renderer** `@mlp_train_step`
    (`StableHLO.mlpTrainStepText`) through the generic IREE invoke. `params` is
    the packed f32 weights (sliced per `shapes`, same layout as `forwardF32`);
    `x` is `batch×d₀`; `y` is int32 `[batch]` (one-hot built in the C shim with
    `d₃` classes). Returns the updated params, packed in the same layout. -/
@[extern "lean_iree_mlp_train_step_v"]
opaque mlpTrainStepV
  (sess : @& IreeSession) (fnName : @& String)
  (x : @& ByteArray) (params : @& ByteArray) (shapes : @& ByteArray) (y : @& ByteArray)
  (batch : USize) (d0 : USize) (d3 : USize) : IO ByteArray

end IreeSession

/- Sizes for the packed-params layout. -/
namespace MlpLayout
def nW0 : Nat := 784 * 512  -- 401408
def nb0 : Nat := 512
def nW1 : Nat := 512 * 512  -- 262144
def nb1 : Nat := 512
def nW2 : Nat := 512 * 10   -- 5120
def nb2 : Nat := 10
def nParams : Nat := nW0 + nb0 + nW1 + nb1 + nW2 + nb2  -- 669706
def lossIdx : Nat := nParams
end MlpLayout

/-- Push a UInt32 as 4 bytes little-endian. -/
private def pushU32 (ba : ByteArray) (v : Nat) : ByteArray := Id.run do
  let mut b := ba
  b := b.push (v % 256).toUInt8
  b := b.push ((v / 256) % 256).toUInt8
  b := b.push ((v / 65536) % 256).toUInt8
  b := b.push ((v / 16777216) % 256).toUInt8
  return b

/-- Pack param shape descriptors: `[nParams, rank0, d0..., rank1, d1..., ...]` as int32 LE. -/
def packShapes (shapes : Array (Array Nat)) : ByteArray := Id.run do
  let mut ba := pushU32 .empty shapes.size
  for shape in shapes do
    ba := pushU32 ba shape.size
    for d in shape do ba := pushU32 ba d
  return ba

/-- Pack a single shape: `[rank, d0, d1, ...]` as int32 LE (for x input). -/
def packXShape (dims : Array Nat) : ByteArray := Id.run do
  let mut ba := pushU32 .empty dims.size
  for d in dims do ba := pushU32 ba d
  return ba

namespace CnnLayout
def paramShapes : Array (Array Nat) := #[
  #[32, 1, 3, 3], #[32],          -- conv0
  #[32, 32, 3, 3], #[32],         -- conv1
  #[6272, 512], #[512],           -- dense0
  #[512, 512], #[512],            -- dense1
  #[512, 10], #[10]               -- dense2
]
def nParams : Nat := 32*1*3*3 + 32 + 32*32*3*3 + 32 + 6272*512 + 512 + 512*512 + 512 + 512*10 + 10
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 784]
end CnnLayout

namespace CifarLayout
def paramShapes : Array (Array Nat) := #[
  #[32, 3, 3, 3], #[32],          -- conv0: 3→32
  #[32, 32, 3, 3], #[32],         -- conv1: 32→32
  #[64, 32, 3, 3], #[64],         -- conv2: 32→64
  #[64, 64, 3, 3], #[64],         -- conv3: 64→64
  #[4096, 512], #[512],           -- dense0
  #[512, 512], #[512],            -- dense1
  #[512, 10], #[10]               -- dense2
]
def nParams : Nat :=
  32*3*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 +
  4096*512 + 512 + 512*512 + 512 + 512*10 + 10  -- 2430018
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end CifarLayout

namespace CifarBnLayout
/-- BN-CIFAR params: each conv layer carries per-channel γ/β `[c]` after its
    bias, interleaved as `W|b|γ|β`. 22 params (4×{W,b,γ,β} + 3×{W,b}). Order MUST
    match `@cifar_bn_train_step`'s signature. -/
def paramShapes : Array (Array Nat) := #[
  #[32, 3, 3, 3], #[32], #[32], #[32],   -- conv0: 3→32  + γ1,β1 [32]
  #[32, 32, 3, 3], #[32], #[32], #[32],  -- conv1: 32→32 + γ2,β2 [32]
  #[64, 32, 3, 3], #[64], #[64], #[64],  -- conv2: 32→64 + γ3,β3 [64]
  #[64, 64, 3, 3], #[64], #[64], #[64],  -- conv3: 64→64 + γ4,β4 [64]
  #[4096, 512], #[512],                  -- dense0
  #[512, 512], #[512],                   -- dense1
  #[512, 10], #[10]                      -- dense2
]
def nParams : Nat :=
  (32*3*3*3 + 32 + 32 + 32) + (32*32*3*3 + 32 + 32 + 32) +
  (64*32*3*3 + 64 + 64 + 64) + (64*64*3*3 + 64 + 64 + 64) +
  4096*512 + 512 + 512*512 + 512 + 512*10 + 10
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end CifarBnLayout

namespace ResNet34Layout
/-- Chapter-6 **real ResNet-34** params (IMAGENETTE 3×224×224 — paper-native ImageNet
    resolution): **7×7 stride-2 stem** {W=`[64,3,7,7]`,b,γ,β} (224→112), then the
    16 basic blocks (3 strided downsample {W,b,γ,β}×2 + proj{W,b,γ,β}; 13 identity
    {W,b,γ,β}×2) at channels 64/128/256/512 (spatial 56/28/14/7), then dense {W,b}.
    Per-channel BN ⇒ γ/β are **rank-1 `[c]`** (not rank-0 scalars). 146 params. The
    `(dims, initKind)` order MUST match `@resnet34_train_step`'s signature (and
    `@resnet34_fwd`'s) — both rendered from the same `Blk` list (tests/TestResnet34*.lean
    `allParams`). `initKind`: 0 = He(fan-in) (stem fan-in = 3·7·7 = 147), 1 = ones (γ),
    2 = zeros (β / bias). -/
private def idBlk (c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2)]
private def downBlk (cin c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,cin,3,3],0),(#[c],2),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2),
    (#[c,cin,3,3],0),(#[c],2),(#[c],1),(#[c],2)]
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[64,3,7,7],0),(#[64],2),(#[64],1),(#[64],2)]  -- 7×7-s2 stem
  for _ in [0:3] do a := a ++ idBlk 64                                                     -- stage1
  a := a ++ downBlk 64 128;  for _ in [0:3] do a := a ++ idBlk 128                         -- stage2
  a := a ++ downBlk 128 256; for _ in [0:5] do a := a ++ idBlk 256                         -- stage3
  a := a ++ downBlk 256 512; for _ in [0:2] do a := a ++ idBlk 512                         -- stage4
  a := a ++ #[(#[512,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end ResNet34Layout

namespace MobileNetV2Layout
/-- Chapter-7 **MobileNetV2** params (IMAGENETTE 3×224×224 — paper-native ImageNet
    resolution, real downsampling `[t,c,n,s]`): stem {W,b,γ,β} (3×3 stride-2 conv
    3→16), then 6 inverted-residual blocks — each expand 1×1 {W,b,γ,β}, depthwise 3×3
    {W,b,γ,β} (a `[mid,1,3,3]` kernel, feature_group_count = mid; stride-2 for the 4
    downsampling blocks b1/b3/b5/b6), project 1×1 {W,b,γ,β} — then the head 1×1 conv
    {W,b,γ,β} (64→128, the MNv2 "features" layer: conv→BN→relu6 before GAP, so the
    pooled tensor isn't the constant β of an instance-normed BN) and dense {W,b}.
    Per-channel BN ⇒ γ/β are **rank-1 `[c]`**. 82 params. Spatial
    224→112(stem)→56(b1,s2)→28(b3,s2)→14(b5,s2)→7(b6,s2) — the real MobileNetV2 /32
    flow. The `(dims, initKind)` order MUST match `@mobilenetv2_train_step`'s signature
    (and `@mobilenetv2_fwd`'s) — both rendered from the same `blocks`/`allParams`
    (tests/TestMobilenetV2*.lean). Strides live only in the renderers (no param-shape
    effect). `initKind`: 0 = He(fan-in) (depthwise fan-in = 1·3·3 = 9), 1 = ones (γ),
    2 = zeros. -/
private def irBlk (ic mid oc : Nat) : Array (Array Nat × Nat) :=
  #[(#[mid,ic,1,1],0),(#[mid],2),(#[mid],1),(#[mid],2),    -- expand 1×1
    (#[mid,1,3,3],0),(#[mid],2),(#[mid],1),(#[mid],2),     -- depthwise 3×3 (stride 1 or 2)
    (#[oc,mid,1,1],0),(#[oc],2),(#[oc],1),(#[oc],2)]       -- project 1×1
/-- (ic, mid, oc) per block — MUST match tests/TestMobilenetV2*.lean `blocks`. -/
private def blocks : Array (Nat × Nat × Nat) :=
  #[(16,64,24),(24,96,24),(24,96,32),(32,128,32),(32,128,64),(64,256,64)]
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[16,3,3,3],0),(#[16],2),(#[16],1),(#[16],2)]  -- stem
  for (ic, mid, oc) in blocks do a := a ++ irBlk ic mid oc                                 -- 6 IR blocks
  a := a ++ #[(#[128,64,1,1],0),(#[128],2),(#[128],1),(#[128],2)]                          -- head 1×1 conv→BN→relu6
  a := a ++ #[(#[128,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end MobileNetV2Layout

namespace EfficientNetLayout
/-- Chapter-8 **EfficientNet-B0** params (CIFAR 3×32×32, E6 — faithful `[t,c,n,s,k]`
    config, all-swish + BATCH norm): stem {W,b,γ,β} (3×3 stride-1 conv 3→32, CIFAR
    adaptation), then 16 MBConv layers across 7 stages (channels [16,24,40,80,112,192,320],
    kernels [3,3,5,3,5,5,3], expand [1,6,6,6,6,6,6] — the MBConv1 stage-1 blocks have NO
    expand conv) — each (when expanded) expand 1×1 {W,b,γ,β}, depthwise k×k {W,b,γ,β}
    (`[mid,1,k,k]`, feature_group_count = mid), **squeeze-excite** {Ws₁`[mid,r]`,bs₁`[r]`,
    Ws₂`[r,mid]`,bs₂`[mid]`} (r = ic/4), project 1×1 {W,b,γ,β} — then head 1×1 conv {W,b,γ,β}
    (320→1280) and dense {W,b}. Batch-norm γ/β rank-1 `[c]`. 262 params. Spatial
    32→16→8→4→2 (4 strided stages, stem stride 1). The `(dims, initKind)` order MUST match
    `@efficientnet_train_step`'s signature — both rendered from the same `stages`/`blocks`
    generator (tests/TestEfficientNet*.lean). `initKind`: 0 = He(fan-in) (depthwise fan-in
    = k², SE dense = mid/r), 1 = ones (γ), 2 = zeros (β / bias). -/
private def stages : Array (Nat × Nat × Nat × Nat × Nat) :=
  #[(1,16,1,1,3),(6,24,2,2,3),(6,40,2,2,5),(6,80,3,2,3),(6,112,3,1,5),(6,192,4,2,5),(6,320,1,1,3)]
private def mbBlk (ic mid oc r k : Nat) : Array (Array Nat × Nat) :=
  (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],2),(#[mid],1),(#[mid],2)] else #[]) ++  -- expand (skip if t=1)
  #[(#[mid,1,k,k],0),(#[mid],2),(#[mid],1),(#[mid],2),    -- depthwise k×k (stride 1 or 2)
    (#[mid,r],0),(#[r],2),(#[r,mid],0),(#[mid],2),        -- squeeze-excite dense₁/dense₂
    (#[oc,mid,1,1],0),(#[oc],2),(#[oc],1),(#[oc],2)]      -- project 1×1
/-- `(dims, initKind)` for every param, in func-arg order — generated from the B0 stage
    spec exactly as tests/TestEfficientNet*.lean `blocks` (stem out 32, prev threading). -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[32,3,3,3],0),(#[32],2),(#[32],1),(#[32],2)]  -- stem 3→32
  let mut prev := 32
  for (t, c, n, _s, k) in stages do
    for j in [0:n] do
      let ic := if j == 0 then prev else c
      a := a ++ mbBlk ic (t*ic) c (max 1 (ic/4)) k
    prev := c
  a := a ++ #[(#[1280,320,1,1],0),(#[1280],2),(#[1280],1),(#[1280],2)]                     -- head 320→1280
  a := a ++ #[(#[1280,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end EfficientNetLayout

namespace ConvNeXtLayout
/-- Chapter-9 **ConvNeXt-T** params (IMAGENETTE 3×224×224 — paper-native resolution):
    4×4/s4 patchify stem {W=`[96,3,4,4]`,b} (224→56), then [3,3,9,3] blocks @ [96,192,
    384,768] (spatial 56/28/14/7) with 3 between-stage LN+2×2/s2 downsamples, then head
    GAP → LN(768) → dense {W,b}. ConvNeXt block (9 params): depthwise 7×7 {W=`[c,1,7,7]`,b}
    → **LN** (global per-example scalar γ/β, rank-0 `#[]`) → 1×1 expand {W=`[4c,c,1,1]`,b}
    → GELU → 1×1 project {W=`[c,4c,1,1]`,b} → **layerScale** (per-channel γ=`[c]`). Each
    downsample (4 params): LN scalar {γ,β} + 2×2 conv {W=`[2c,c,2,2]`,b}. 180 params. The
    `(dims, initKind)` order MUST match `@convnext_train_step`'s signature — both from the
    same [3,3,9,3] generator (tests/TestConvNeXt*.lean). `initKind`: 0 = He(fan-in)
    (depthwise 49, expand c, project 4c, patchify 48, downsample 4c, dense 768), 1 = ones
    (LN γ / layerScale γ), 2 = zeros (LN β / bias). -/
private def depths : Array Nat := #[3, 3, 9, 3]
private def dims   : Array Nat := #[96, 192, 384, 768]
private def blockSpec (c e : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,1,7,7],0),(#[c],2),(#[],1),(#[],2),     -- depthwise W,b ; LN γ,β (scalar)
    (#[e,c,1,1],0),(#[e],2),                      -- expand W,b
    (#[c,e,1,1],0),(#[c],2),                      -- project W,b
    (#[c],1)]                                     -- layerScale γ (per-channel)
private def downSpec (ci co : Nat) : Array (Array Nat × Nat) :=
  #[(#[],1),(#[],2),(#[co,ci,2,2],0),(#[co],2)]   -- LN γ,β (scalar) ; 2×2/s2 conv W,b
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[96,3,4,4],0),(#[96],2)]   -- patchify stem
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    for _ in [0:depths[si]!] do a := a ++ blockSpec c e
    if si < 3 then a := a ++ downSpec c dims[si+1]!
  a := a ++ #[(#[],1),(#[],2),(#[768,10],0),(#[10],2)]   -- head LN γ,β ; dense W,b
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end ConvNeXtLayout

namespace ViTLayout
/-- Chapter-10 **ViT-Tiny** params (IMAGENETTE 3×224×224, patch-16): a 16×16/s16 conv
    patch embed {W=`[192,3,16,16]`,b} (224→14×14=196 patches), a learned CLS token
    `[1,192]` + positional embed `[197,192]`, then 12 pre-norm transformer blocks
    (dim 192, 3 heads, MLP 768), final LayerNorm γ/β, CLS-slice dense head {W=`[192,10]`,b}.
    LayerNorm γ/β are **per-channel `[192]`** (the non-scalar form — beyond the scalar
    proof witness `vit_full`, faithful per-op: normalize ∘ per-channel affine). Each block
    (16 params): LN1 γ/β, Wq/bq/Wk/bk/Wv/bv/Wo/bo `[192,192]`/`[192]`, LN2 γ/β, MLP
    Wfc1`[192,768]`/bfc1/Wfc2`[768,192]`/bfc2. 4+12·16+4 = 200 params. The `(dims,initKind)`
    order MUST match `@vit_train_step`/`@vit_fwd` (tests/TestViT{Train,Fwd}.lean, from the
    same `ViTRender.vitParam*` generator). `initKind`: 0 = He(fan-in) (patch 3·16·16=768,
    QKV/out/head fan-in=192, fc1=192, fc2=768), 1 = ones (LN γ), 2 = zeros (LN β / bias /
    CLS / pos). -/
private def D : Nat := 192
private def M : Nat := 768
private def S : Nat := 16
private def NTOK : Nat := 197    -- 14·14 + 1 (CLS)
private def DEPTH : Nat := 12
private def NC : Nat := 10
private def blockSpec : Array (Array Nat × Nat) :=
  #[(#[D],1),(#[D],2),                                                       -- LN1 γ,β
    (#[D,D],0),(#[D],2),(#[D,D],0),(#[D],2),(#[D,D],0),(#[D],2),(#[D,D],0),(#[D],2),  -- Wq..bo
    (#[D],1),(#[D],2),                                                       -- LN2 γ,β
    (#[D,M],0),(#[M],2),(#[M,D],0),(#[D],2)]                                 -- MLP
/-- `(dims, initKind)` for every param, in `@vit_train_step` func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) :=
    #[(#[D,3,S,S],0),(#[D],2),(#[1,D],2),(#[NTOK,D],2)]   -- patch W,b ; CLS ; pos
  for _ in [0:DEPTH] do a := a ++ blockSpec
  a := a ++ #[(#[D],1),(#[D],2),(#[D,NC],0),(#[NC],2)]   -- final LN γ,β ; head W,b
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end ViTLayout

def MlpLayout.paramShapes : Array (Array Nat) := #[
  #[784, 512], #[512], #[512, 512], #[512], #[512, 10], #[10]
]
def MlpLayout.shapesBA : ByteArray := packShapes MlpLayout.paramShapes
