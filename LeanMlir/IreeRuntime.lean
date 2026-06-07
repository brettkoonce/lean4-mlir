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
/-- BN-CIFAR params: each conv layer carries scalar γ/β (rank-0 `#[]`) after its
    bias, interleaved as `W|b|γ|β`. 22 params (4×{W,b,γ,β} + 3×{W,b}). Order MUST
    match `@cifar_bn_train_step`'s signature. -/
def paramShapes : Array (Array Nat) := #[
  #[32, 3, 3, 3], #[32], #[], #[],     -- conv0: 3→32  + γ1,β1
  #[32, 32, 3, 3], #[32], #[], #[],    -- conv1: 32→32 + γ2,β2
  #[64, 32, 3, 3], #[64], #[], #[],    -- conv2: 32→64 + γ3,β3
  #[64, 64, 3, 3], #[64], #[], #[],    -- conv3: 64→64 + γ4,β4
  #[4096, 512], #[512],                -- dense0
  #[512, 512], #[512],                 -- dense1
  #[512, 10], #[10]                    -- dense2
]
def nParams : Nat :=
  (32*3*3*3 + 32 + 1 + 1) + (32*32*3*3 + 32 + 1 + 1) +
  (64*32*3*3 + 64 + 1 + 1) + (64*64*3*3 + 64 + 1 + 1) +
  4096*512 + 512 + 512*512 + 512 + 512*10 + 10
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end CifarBnLayout

namespace ResnetLayout
/-- Chapter-6 ResNet-style params (ic=3, c=32, oc=64, 3×3 convs): stem {W,b,γ,β},
    identity block 2×{W,b,γ,β}, projection block 2×{W,b,γ,β} + proj {W,b,γ,β},
    dense {W,b}. 26 params; γ/β are rank-0 `#[]` scalars. Order MUST match
    `@resnet_train_step`'s signature (and `@resnet_fwd`'s). -/
def paramShapes : Array (Array Nat) := #[
  #[32, 3, 3, 3], #[32], #[], #[],     -- stem conv 3→32   + γs,βs
  #[32, 32, 3, 3], #[32], #[], #[],    -- rblk  conv1 32→32 + γ1,β1
  #[32, 32, 3, 3], #[32], #[], #[],    -- rblk  conv2 32→32 + γ2,β2
  #[64, 32, 3, 3], #[64], #[], #[],    -- rblkP conv1 32→64 + γ1p,β1p
  #[64, 64, 3, 3], #[64], #[], #[],    -- rblkP conv2 64→64 + γ2p,β2p
  #[64, 32, 3, 3], #[64], #[], #[],    -- rblkP proj  32→64 + γp,βp
  #[64, 10], #[10]                     -- dense 64→10
]
def nParams : Nat :=
  (32*3*3*3 + 32 + 1 + 1) + (32*32*3*3 + 32 + 1 + 1) + (32*32*3*3 + 32 + 1 + 1) +
  (64*32*3*3 + 64 + 1 + 1) + (64*64*3*3 + 64 + 1 + 1) + (64*32*3*3 + 64 + 1 + 1) +
  64*10 + 10
def lossIdx : Nat := nParams
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end ResnetLayout

namespace ResNet34Layout
/-- Chapter-6 **real ResNet-34** params (CIFAR 3×32×32): stem {W,b,γ,β}, then the
    16 basic blocks (3 strided downsample {W,b,γ,β}×2 + proj{W,b,γ,β}; 13 identity
    {W,b,γ,β}×2) at channels 64/128/256/512, then dense {W,b}. Per-channel BN ⇒ γ/β
    are **rank-1 `[c]`** (not rank-0 scalars). 146 params. The `(dims, initKind)`
    order MUST match `@resnet34_train_step`'s signature (and `@resnet34_fwd`'s) —
    both rendered from the same `Blk` list (tests/TestResnet34*.lean `allParams`).
    `initKind`: 0 = He(fan-in), 1 = ones (γ), 2 = zeros (β / bias). -/
private def idBlk (c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2)]
private def downBlk (cin c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,cin,3,3],0),(#[c],2),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],2),(#[c],1),(#[c],2),
    (#[c,cin,3,3],0),(#[c],2),(#[c],1),(#[c],2)]
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[64,3,3,3],0),(#[64],2),(#[64],1),(#[64],2)]  -- stem
  for _ in [0:3] do a := a ++ idBlk 64                                                     -- stage1
  a := a ++ downBlk 64 128;  for _ in [0:3] do a := a ++ idBlk 128                         -- stage2
  a := a ++ downBlk 128 256; for _ in [0:5] do a := a ++ idBlk 256                         -- stage3
  a := a ++ downBlk 256 512; for _ in [0:2] do a := a ++ idBlk 512                         -- stage4
  a := a ++ #[(#[512,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end ResNet34Layout

namespace MobileNetV2Layout
/-- Chapter-7 small **MobileNetV2** params (CIFAR 3×32×32): stem {W,b,γ,β} (3×3
    stride-2 conv 3→32), then two inverted-residual blocks — each expand 1×1
    {W,b,γ,β}, depthwise 3×3 {W,b,γ,β} (a `[mid,1,3,3]` kernel, feature_group_count
    = mid), project 1×1 {W,b,γ,β} — then the head 1×1 conv {W,b,γ,β} (64→128, the
    MNv2 "features" layer: conv→BN→relu6 before GAP, so the pooled tensor isn't the
    constant β of an instance-normed BN) and dense {W,b}. Per-channel BN ⇒ γ/β are
    **rank-1 `[c]`**. 34 params. The `(dims, initKind)` order MUST match
    `@mobilenetv2_train_step`'s signature (and `@mobilenetv2_fwd`'s) — both rendered
    from the same `allParams` (tests/TestMobilenetV2*.lean). `initKind`: 0 =
    He(fan-in) (depthwise fan-in = 1·3·3 = 9), 1 = ones (γ), 2 = zeros (β / bias). -/
private def irBlk (ic mid oc : Nat) : Array (Array Nat × Nat) :=
  #[(#[mid,ic,1,1],0),(#[mid],2),(#[mid],1),(#[mid],2),    -- expand 1×1
    (#[mid,1,3,3],0),(#[mid],2),(#[mid],1),(#[mid],2),     -- depthwise 3×3
    (#[oc,mid,1,1],0),(#[oc],2),(#[oc],1),(#[oc],2)]       -- project 1×1
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[32,3,3,3],0),(#[32],2),(#[32],1),(#[32],2)]  -- stem
  a := a ++ irBlk 32 64 32                                                                 -- IR-A (skip)
  a := a ++ irBlk 32 64 64                                                                 -- IR-B (no-skip)
  a := a ++ #[(#[128,64,1,1],0),(#[128],2),(#[128],1),(#[128],2)]                          -- head 1×1 conv→BN→relu6
  a := a ++ #[(#[128,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3072]
end MobileNetV2Layout

def MlpLayout.paramShapes : Array (Array Nat) := #[
  #[784, 512], #[512], #[512, 512], #[512], #[512, 10], #[10]
]
def MlpLayout.shapesBA : ByteArray := packShapes MlpLayout.paramShapes
