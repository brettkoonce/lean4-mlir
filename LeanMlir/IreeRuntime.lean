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

/-- Load a `.vmfb` bytecode module onto the default CUDA device.

    On the **XLA backend** (`libpjrt_ffi.so`) the argument is instead the
    `.mlir` source — XLA compiles the StableHLO in-process, so there is no
    separate `iree-compile` step. Use `VerifiedNet.mkSession` rather than
    calling this directly; it picks the right path per `backendName`. -/
@[extern "lean_iree_session_create"]
opaque create (path : @& String) : IO IreeSession

/-- `"iree"` or `"xla"` — which shim this binary was linked against. Detected by
    probing for a symbol only `libpjrt_ffi.so` defines, so it cannot disagree
    with the linked library. See `planning/xla_pjrt_ladder.md`. -/
@[extern "lean_iree_backend_name"]
opaque backendName : IO String

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
    For inference/eval — no y, lr, or velocity inputs.

    `nResident` / `gen` — device residency in **HOLD** mode (§2d.3), and it is a
    different mechanism from the train step's. This graph returns *logits*, not
    parameters, so there is nothing to retain from the output; instead the whole
    parameter set is seeded once and reused across every eval batch, rather than
    pushed 79-123 times per epoch. Measured on the MNIST MLP, **73% of an eval
    step was the parameter push** (0.6 ms of 0.8 — compute is 0.1).

    ⚠ **`gen` is what makes holding safe, and it must change whenever `params`
    does.** Pass the epoch number. A held set that went stale would score the
    previous epoch's weights *silently*, which reads as a training plateau rather
    than as an error — a nastier failure than anything the update mode has. The
    shim re-seeds the moment the token differs.

    Defaults (`0`, `0`) = the copying path, so every inference demo that calls
    this is unaffected. -/
@[extern "lean_iree_forward_f32"]
opaque forwardF32
  (sess : @& IreeSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (batch : USize) (nClasses : USize)
  (nResident : USize := 0) (gen : USize := 0) : IO ByteArray

/-- Drive the **verified-renderer** `@linear_train_step`
    (`StableHLO.linearTrainStepModuleV`) through the generic IREE invoke.
    Inputs are raw f32 ByteArrays: `x` is `batch×d₀`, `W0` is `d₀×d₁`, `b0`
    is `d₁`; `y` is int32 `[batch]` (the one-hot is built in the C shim).
    Returns `W0n (d₀·d₁ f32) ++ b0n (d₁ f32)`.

    `nResident`: see `mlpTrainStepV`. Here it is **2** — `W0` and `b0`, i.e. the
    whole parameter set, since this graph returns exactly its two param inputs. -/
@[extern "lean_iree_linear_train_step"]
opaque linearTrainStepV
  (sess : @& IreeSession) (fnName : @& String)
  (x : @& ByteArray) (W0 : @& ByteArray) (b0 : @& ByteArray) (y : @& ByteArray)
  (batch : USize) (d0 : USize) (d1 : USize)
  (nResident : USize := 0) : IO ByteArray

/-- Data-parallel `@<slug>_train_step`: same packed-params protocol as
    `mlpTrainStepV`, but `batch` is the GLOBAL batch and the XLA shim splits x and
    the labels across `replicas` devices while replicating the parameters. The
    emitted graph all-reduces every gradient before the optimizer consumes it
    (`ViTRender.emitAdamVDP`), so all replicas produce identical parameters and
    the result is read back from replica 0.

    Only the XLA shim exports the underlying entry point; on the IREE build this
    raises rather than silently running single-device.

    `nResident`: see `mlpTrainStepV`. Each replica keeps its own retained set on
    its own device, which is where the bigger half of the win is — today the full
    `[θ|m|v]` is pushed to *every* replica every step, an O(N−1) cost against
    O(1) compute (§2d.3a: 4 GPUs currently buy 1.46×).

    `nShardTail`: how many TRAILING entries of the param list are PER-EXAMPLE and must be sharded
    like `x` rather than replicated like the parameters. Today that is exactly the stochastic-depth
    drop masks. ⚠ It is a COUNT supplied by the driver rather than something the shim infers: an
    index would be per-net and a shape test ("outer dim == batch") would sweep up any parameter
    that happens to be batch-sized. Default 0, so every existing call site is unchanged.

    ⚠ The extern is `_dp2`, not `_dp`, because this ADDED AN ARGUMENT — §4's rule for
    `pjrt_ffi_invoke_f32_resident_v2`: a stale `.so` against a new binary shifts every argument,
    which is garbage rather than a link error. A rename makes it a link error. -/
@[extern "lean_iree_mlp_train_step_v_dp2"]
opaque mlpTrainStepVDP
  (sess : @& IreeSession) (fnName : @& String)
  (x : @& ByteArray) (params : @& ByteArray) (shapes : @& ByteArray) (y : @& ByteArray)
  (batch : USize) (d0 : USize) (d3 : USize) (replicas : USize)
  (nResident : USize := 0) (nShardTail : USize := 0) : IO ByteArray

/-- Drive the **verified-renderer** `@mlp_train_step`
    (`StableHLO.mlpTrainStepText`) through the generic IREE invoke. `params` is
    the packed f32 weights (sliced per `shapes`, same layout as `forwardF32`);
    `x` is `batch×d₀`; `y` is int32 `[batch]` (one-hot built in the C shim with
    `d₃` classes). Returns the updated params, packed in the same layout.

    `nResident` — how many LEADING param tensors may stay on the device between
    steps (handoff §2d.3). The driver is the only place that knows the packed
    layout is `[θ|m|v | lr,bc₁,bc₂ | bn stats]`, and hence that the first `3×P`
    tensors are exactly the ones the host writes once and thereafter only feeds
    straight back; so it states the count and the shim checks that input `i+1`
    and output `i` really are the same tensor before retaining anything.

    **It is a request, not a mode.** The transport is chosen in C — residency
    engages only under `$PJRT_FFI_RESIDENT=1` on the XLA build, and is inert
    everywhere else — precisely so that this driver keeps no backend branch to
    drift (§2d.3, "the design decision that protects every existing gate").
    Default `0` = the copying path, which is what every tie and DP-check harness
    wants: those read the whole returned blob, and a retained prefix would leave
    it unwritten. -/
@[extern "lean_iree_mlp_train_step_v"]
opaque mlpTrainStepV
  (sess : @& IreeSession) (fnName : @& String)
  (x : @& ByteArray) (params : @& ByteArray) (shapes : @& ByteArray) (y : @& ByteArray)
  (batch : USize) (d0 : USize) (d3 : USize)
  (nResident : USize := 0) : IO ByteArray

/-- Read the **authoritative** leading `nBytes` of the packed parameter blob.

    Without residency this is `packed.extract 0 nBytes` and nothing more — which
    is what it must be on IREE, where the weak read-back symbol does not exist.
    With residency live the `[θ|m|v]` prefix of `packed` is unwritten (it never
    came back from the device), and this performs the one d2h that still happens.

    Either way it is a **per-epoch** call: the eval pass and the checkpoint are
    the only things that want the whole blob, and that call site was already
    once-per-epoch before any of this. -/
@[extern "lean_iree_read_params"]
opaque readParams
  (sess : @& IreeSession) (packed : @& ByteArray) (nBytes : USize) : IO ByteArray

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
/-- Chapter-5 **real ResNet-34** params (IMAGENETTE 3×224×224 — paper-native ImageNet
    resolution): **7×7 stride-2 stem** {W=`[64,3,7,7]`,γ,β} (224→112), then the
    16 basic blocks (3 strided downsample {W,γ,β}×2 + proj{W,γ,β}; 13 identity
    {W,γ,β}×2) at channels 64/128/256/512 (spatial 56/28/14/7), then dense {W,b}.
    Per-channel BN ⇒ γ/β are **rank-1 `[c]`** (not rank-0 scalars). **110 params** (§2l step B: no conv biases). The
    `(dims, initKind)` order MUST match `@resnet34_train_step`'s signature (and
    `@resnet34_fwd`'s) — both rendered from the same `Blk` list (tests/TestResnet34*.lean
    `allParams`). `initKind`: 0 = He(fan-in) (stem fan-in = 3·7·7 = 147), 1 = ones (γ),
    2 = zeros (β / bias). -/
-- §2l step B (2026-07-30): the conv BIASES are gone — `{W, γ, β}` per conv, not `{W, b, γ, β}`.
-- Every conv here is BN-followed and BN removes the bias, so it was 8,512 parameters that could
-- not affect the output; He et al.'s `.convBn` has none, and carrying them put this layout
-- 8,512 params away from the ImageNet reference it is supposed to be paired with (§2k).
-- MEASURED before the change, not argued: zeroing all 8,512 in the TRAINED net moves the logits
-- by rel 1e-6 (the same ablation on BN β moves them by 0.79), and the bias-free render ties the
-- biased one with every forward-only output BIT-EXACT. `tests/TestConvBiasZero.lean`.
private def idBlk (c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,c,3,3],0),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],1),(#[c],2)]
private def downBlk (cin c : Nat) : Array (Array Nat × Nat) :=
  #[(#[c,cin,3,3],0),(#[c],1),(#[c],2), (#[c,c,3,3],0),(#[c],1),(#[c],2),
    (#[c,cin,1,1],0),(#[c],1),(#[c],2)]   -- §2l step A: option-B 1×1 projection
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[64,3,7,7],0),(#[64],1),(#[64],2)]  -- 7×7-s2 stem
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
/-- Chapter-6 **MobileNetV2** params (IMAGENETTE 3×224×224 — paper-native ImageNet
    resolution, full-paper downsampling `[t,c,n,s]`): stem {W,γ,β} (3×3 stride-2 conv
    3→32), then 17 inverted-residual blocks — each expand 1×1 {W,γ,β}, depthwise 3×3
    {W,γ,β} (a `[mid,1,3,3]` kernel, feature_group_count = mid; stride-2 for the
    downsampling blocks), project 1×1 {W,γ,β} — then the head 1×1 conv
    {W,γ,β} (320→1280, the MNv2 "features" layer: conv→BN→relu6 before GAP, so the
    pooled tensor isn't the constant β of an instance-normed BN) and dense {W,b}.

    **158 params** (§2m: no conv biases). Every conv here is BN-followed and BN removes a bias,
    so the 52 this layout used to carry were parameters that could not affect the output — and
    they were the whole of the +17,056 gap to the JAX reference (3,521,928 vs 3,504,872 at
    K = 1000). Measured before the change, not argued: `conv-bias-zero --tie` puts every
    forward-only output of the AdamW step BIT-EXACT (`%loss` 3/3, all 34,112 BN running stats)
    with differences confined to the backward at 2.5e-7, and `--fwd`/`--fwd --eval` tie both
    forward artifacts BIT-EXACT on all 320 logits.
    Per-channel BN ⇒ γ/β are **rank-1 `[c]`**. Spatial
    224→112(stem)→56→28→14→7 — the real MobileNetV2 /32
    flow. The `(dims, initKind)` order MUST match `@mobilenetv2_train_step`'s signature
    (and `@mobilenetv2_fwd`'s) — both rendered from the same `blocks`/`allParams`
    (tests/TestMobilenetV2*.lean). Strides live only in the renderers (no param-shape
    effect). `initKind`: 0 = He(fan-in) (depthwise fan-in = 1·3·3 = 9), 1 = ones (γ),
    2 = zeros. -/
private def irBlk (ic mid oc : Nat) : Array (Array Nat × Nat) :=
  (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],1),(#[mid],2)] else #[]) ++  -- expand 1×1 (skip if t=1, mid=ic)
  #[(#[mid,1,3,3],0),(#[mid],1),(#[mid],2),                -- depthwise 3×3 (stride 1 or 2)
    (#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]                 -- project 1×1
/-- (ic, mid, oc) per block — MUST match tests/TestMobilenetV2*.lean `blocks`. Full paper net (17). -/
private def blocks : Array (Nat × Nat × Nat) :=
  #[(32,32,16),
    (16,96,24),(24,144,24),
    (24,144,32),(32,192,32),(32,192,32),
    (32,192,64),(64,384,64),(64,384,64),(64,384,64),
    (64,384,96),(96,576,96),(96,576,96),
    (96,576,160),(160,960,160),(160,960,160),
    (160,960,320)]
/-- `(dims, initKind)` for every param, in func-arg order. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[32,3,3,3],0),(#[32],1),(#[32],2)]             -- stem
  for (ic, mid, oc) in blocks do a := a ++ irBlk ic mid oc                                 -- 17 IR blocks
  a := a ++ #[(#[1280,320,1,1],0),(#[1280],1),(#[1280],2)]                                 -- head 1×1 conv→BN→relu6
  a := a ++ #[(#[1280,10],0),(#[10],2)]                                                    -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end MobileNetV2Layout

namespace EfficientNetLayout
/-- Chapter-7 **EfficientNet-B0** params (CIFAR 3×32×32, E6 — faithful `[t,c,n,s,k]`
    config, all-swish + BATCH norm): stem {W,γ,β} (3×3 stride-1 conv 3→32, CIFAR
    adaptation), then 16 MBConv layers across 7 stages (channels [16,24,40,80,112,192,320],
    kernels [3,3,5,3,5,5,3], expand [1,6,6,6,6,6,6] — the MBConv1 stage-1 blocks have NO
    expand conv) — each (when expanded) expand 1×1 {W,γ,β}, depthwise k×k {W,γ,β}
    (`[mid,1,k,k]`, feature_group_count = mid), **squeeze-excite** {Ws₁`[mid,r]`,bs₁`[r]`,
    Ws₂`[r,mid]`,bs₂`[mid]`} (r = ic/4), project 1×1 {W,γ,β} — then head 1×1 conv {W,γ,β}
    (320→1280) and dense {W,b}. Batch-norm γ/β rank-1 `[c]`. **213 params** (§2m: the 49
    BN-followed convs carry no bias; at K = 1000 this is 5,288,548 — the JAX reference's own
    count). ⚠ **SE's two biases STAY**: those 1×1 convs are followed by the sigmoid gate, not by
    BN, so nothing absorbs them and the reference carries them. Spatial
    32→16→8→4→2 (4 strided stages, stem stride 1). The `(dims, initKind)` order MUST match
    `@efficientnet_train_step`'s signature — both rendered from the same `stages`/`blocks`
    generator (tests/TestEfficientNet*.lean). `initKind`: 0 = He(fan-in) (depthwise fan-in
    = k², SE dense = mid/r), 1 = ones (γ), 2 = zeros (β / bias). -/
private def stages : Array (Nat × Nat × Nat × Nat × Nat) :=
  #[(1,16,1,1,3),(6,24,2,2,3),(6,40,2,2,5),(6,80,3,2,3),(6,112,3,1,5),(6,192,4,2,5),(6,320,1,1,3)]
private def mbBlk (ic mid oc r k : Nat) : Array (Array Nat × Nat) :=
  (if mid != ic then #[(#[mid,ic,1,1],0),(#[mid],1),(#[mid],2)] else #[]) ++  -- expand (skip if t=1)
  #[(#[mid,1,k,k],0),(#[mid],1),(#[mid],2),               -- depthwise k×k (stride 1 or 2)
    (#[mid,r],0),(#[r],2),(#[r,mid],0),(#[mid],2),        -- squeeze-excite dense₁/dense₂ — biases KEPT
    (#[oc,mid,1,1],0),(#[oc],1),(#[oc],2)]                -- project 1×1
/-- `(dims, initKind)` for every param, in func-arg order — generated from the B0 stage
    spec exactly as tests/TestEfficientNet*.lean `blocks` (stem out 32, prev threading). -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) := #[(#[32,3,3,3],0),(#[32],1),(#[32],2)]             -- stem 3→32
  let mut prev := 32
  for (t, c, n, _s, k) in stages do
    for j in [0:n] do
      let ic := if j == 0 then prev else c
      a := a ++ mbBlk ic (t*ic) c (max 1 (ic/4)) k
    prev := c
  a := a ++ #[(#[1280,320,1,1],0),(#[1280],1),(#[1280],2)]                                 -- head 320→1280
  a := a ++ #[(#[1280,10],0),(#[10],2)]                                                     -- dense
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end EfficientNetLayout

namespace ConvNeXtLayout
/-- Chapter-8 **ConvNeXt-T** params (IMAGENETTE 3×224×224 — paper-native resolution):
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
  #[(#[c,1,7,7],0),(#[c],2),(#[c],1),(#[c],2),   -- depthwise W,b ; LN γ,β (PER-CHANNEL, §2m)
    (#[e,c,1,1],0),(#[e],2),                      -- expand W,b
    (#[c,e,1,1],0),(#[c],2),                      -- project W,b
    (#[c],1)]                                     -- layerScale γ (per-channel)
private def downSpec (ci co : Nat) : Array (Array Nat × Nat) :=
  #[(#[ci],1),(#[ci],2),(#[co,ci,2,2],0),(#[co],2)]  -- LN γ,β at the PRE-conv width ; conv W,b
/-- `(dims, initKind)` for every param, in func-arg order.

    ⚠ §2m moved three things at once, and they are the whole difference from the retired
    180-param list: every LN affine went rank-0 `#[]` → per-channel `#[c]`, the **stem LN**
    appeared, and the **head LN** went away (the reference's `forward` is
    `patchify → channel_layer_norm → stages → GAP → dense`). The last two nearly cancel
    — `+2·768 − 2·96 = +1,344` of 28.6M — so a matching parameter count is a decomposition
    test, not an architecture check. The stem/head LN swap one-for-one, so this is still
    **180 param tensors**; the floats go 27,826,282 at K = 10, i.e. **28,587,592 at K = 1000**,
    which is the JAX reference's own reported count exactly. -/
def specs : Array (Array Nat × Nat) := Id.run do
  let mut a : Array (Array Nat × Nat) :=
    #[(#[96,3,4,4],0),(#[96],2),(#[96],1),(#[96],2)]   -- patchify stem + stem LN γ,β
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    for _ in [0:depths[si]!] do a := a ++ blockSpec c e
    if si < 3 then a := a ++ downSpec c dims[si+1]!
  a := a ++ #[(#[768,10],0),(#[10],2)]   -- head: dense W,b (NO head LN — §2m)
  return a
def paramShapes : Array (Array Nat) := specs.map (·.1)
def nParams : Nat := (specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
def shapesBA : ByteArray := packShapes paramShapes
def xShape (batch : Nat) : ByteArray := packXShape #[batch, 3 * 224 * 224]   -- Imagenette 224²
end ConvNeXtLayout

namespace ViTLayout
/-- Chapter-9 **ViT-Tiny** params (IMAGENETTE 3×224×224, patch-16): a 16×16/s16 conv
    patch embed {W=`[192,3,16,16]`,b} (224→14×14=196 patches), a learned CLS token
    `[192]` (1D, matching the proof-tied render) + positional embed `[197,192]`, then 12 pre-norm transformer blocks
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
    #[(#[D,3,S,S],0),(#[D],2),(#[D],2),(#[NTOK,D],2)]   -- patch W,b ; CLS [192] (1D) ; pos
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
