import LeanMlir.ParamLayouts
/-! Lean FFI bindings for the lowerer runtime — PJRT/XLA by default, IREE optionally
    (`LowererSession` is lowerer-agnostic).

    Links through the Lean shim in `ffi/`. Also holds the verified nets' parameter-layout
    tables (`*Layout`), which the trainers and `VerifiedSpec` read. -/

/-- Opaque handle to an IREE runtime session (module + device). -/
private opaque LowererSessionPointed : NonemptyType
def LowererSession : Type := LowererSessionPointed.type
instance : Nonempty LowererSession := LowererSessionPointed.property

namespace LowererSession

/-- Load a `.vmfb` bytecode module onto the default CUDA device.

    On the **XLA backend** (`libpjrt_ffi.so`) the argument is instead the
    `.mlir` source — XLA compiles the StableHLO in-process, so there is no
    separate `iree-compile` step. Use `VerifiedNet.mkSession` rather than
    calling this directly; it picks the right path per `backendName`. -/
@[extern "lean_iree_session_create"]
opaque create (path : @& String) : IO LowererSession

/-- A **sharded inference** session — XLA only. The `.mlir` is compiled for `replicas` devices
    and its outputs are GATHERED from all of them, so it is driven by `forwardF32Dp` at the same
    count. `create` compiles a graph with no cross-replica op for one device; this states the
    count instead. That graph computes the same function on every device, so data-parallel
    inference needs no new render.

    It REFUSES any graph with a cross-replica op: the train step's read-back takes replica 0
    only (correct there, because the `all_reduce` makes every replica's result identical), and
    the two contracts must not meet in one session. It is also a loud error, never a fall back
    to one device, when the loaded shim predates it or is IREE's. Use `VerifiedNet.mkSessionDp`
    rather than calling this directly. -/
@[extern "lean_iree_session_create_dp"]
opaque createDp (path : @& String) (replicas : USize) : IO LowererSession

/-- `"iree"` or `"xla"` — which shim this binary was linked against. Detected by
    probing for a symbol only `libpjrt_ffi.so` defines, so it cannot disagree
    with the linked library. See `planning/archive/xla_pjrt_ladder.md`. -/
@[extern "lean_iree_backend_name"]
opaque backendName : IO String

/-- Run MNIST-MLP forward pass. Shapes are fixed:
    `x` is `batch×784`, `W0` is `784×512`, `b0` is `512`,
    `W1` is `512×512`, `b1` is `512`, `W2` is `512×10`, `b2` is `10`.
    Returns the logits as a `batch×10` flattened `FloatArray`. -/
@[extern "lean_iree_mlp_forward"]
opaque mlpForward
  (sess : @& LowererSession)
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
  (sess : @& LowererSession)
  (params : @& FloatArray)
  (x : @& FloatArray)
  (y : @& ByteArray)
  (lr : Float)
  (batch : USize) : IO FloatArray

/-- Generic train step. Shapes are packed ByteArrays (see `packShapes`). -/
@[extern "lean_iree_train_step_packed"]
opaque trainStepPacked
  (sess : @& LowererSession) (fnName : @& String)
  (params : @& FloatArray) (shapes : @& ByteArray)
  (x : @& FloatArray) (xShape : @& ByteArray)
  (y : @& ByteArray)
  (lr : Float) (batch : USize) : IO FloatArray

/-- Zero-copy f32 train step. All tensors are ByteArray (raw float32 bytes).
    No Float64↔Float32 conversion at the boundary. -/
@[extern "lean_iree_train_step_f32"]
opaque trainStepF32
  (sess : @& LowererSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (y : @& ByteArray)
  (lr : Float) (batch : USize) : IO ByteArray

/-- Adam train step (f32). Passes step counter t for bias correction.
    Params = weights ++ m ++ v. Returns params ++ loss ++ BN stats.
    bnShapes: packed [n_bn_layers, oc0, oc1, ...] for BN stat output sizes. -/
@[extern "lean_iree_train_step_adam_f32"]
opaque trainStepAdamF32
  (sess : @& LowererSession) (fnName : @& String)
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
  (sess : @& LowererSession) (fnName : @& String)
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
  (sess : @& LowererSession) (fnName : @& String)
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
  (sess : @& LowererSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (yDdpm : @& ByteArray)
  (lr : Float) (t : Float)
  (bnShapes : @& ByteArray)
  (batch : USize) (outC : USize) (outH : USize) (outW : USize) : IO ByteArray

/-- `trainStepAdamF32Ddpm` with device residency (§2d.3): `nResident` = the number of
    param tensors in `[θ|m|v]` (all of them — the graph's params-in / params-out
    correspondence is index for index). With `PJRT_FFI_RESIDENT=1` on the XLA backend
    they stay on the device after the first call, the result's param region is left
    UNWRITTEN, and `readParams` / `readParamsPrefix` is the way back; `packed`'s param
    region is ignored after the seed. Unset, or on IREE, this is `trainStepAdamF32Ddpm`. -/
@[extern "lean_iree_train_step_adam_f32_ddpm_r"]
opaque trainStepAdamF32DdpmR
  (sess : @& LowererSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (yDdpm : @& ByteArray)
  (lr : Float) (t : Float)
  (bnShapes : @& ByteArray)
  (batch : USize) (outC : USize) (outH : USize) (outW : USize)
  (nResident : USize) : IO ByteArray

/-- YOLOv1 variant. `yYolo` is a `[batch, perCell, gridH, gridW]` f32
    target tensor (NCHW); `mYolo` is a `[batch, gridH, gridW]` f32
    per-cell objectness mask (1.0 where a GT box's center falls in
    the cell, 0.0 otherwise). Routes to the codegen produced with
    `useYolov1 := true`. Loss is the 5-term masked MSE described in
    `planning/archive/yolo_demo_v2.md` Phase 1.

    `perCell = numBoxes * 5 + numClasses`. For VOC this is
    `2*5 + 20 = 30`; `gridH = gridW = 7`. -/
@[extern "lean_iree_train_step_adam_f32_yolov1"]
opaque trainStepAdamF32Yolov1
  (sess : @& LowererSession) (fnName : @& String)
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
  (sess : @& LowererSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (batch : USize) (nClasses : USize)
  (nResident : USize := 0) (gen : USize := 0) : IO ByteArray

/-- `forwardF32` over `replicas` devices, for a session from `createDp` at that count.

    `batch` and `xShape`'s leading dim are the **GLOBAL** batch, `replicas` × the rendered one.
    Replica `r` gets rows `[r·b, (r+1)·b)` of `x`, the parameters go to every replica, and the
    logits come back as ONE `batch × nClasses` buffer in the original row order. So a caller
    indexes the result exactly as it indexes `x`, and the ragged-tail logic is unchanged: pad to
    the global batch and score only the real rows.

    `nResident`/`gen` are `forwardF32`'s hold mode. Each device keeps its own copy of the
    parameters, so the push is `replicas`× once per `gen`, not once per batch.

    At `replicas = 1` it makes exactly the calls `forwardF32` makes. -/
@[extern "lean_iree_forward_f32_dp"]
opaque forwardF32Dp
  (sess : @& LowererSession) (fnName : @& String)
  (params : @& ByteArray) (shapes : @& ByteArray)
  (x : @& ByteArray) (xShape : @& ByteArray)
  (batch : USize) (nClasses : USize) (replicas : USize)
  (nResident : USize := 0) (gen : USize := 0) : IO ByteArray

/-- Drive the **verified-renderer** `@linear_train_step`
    (`StableHLO.linTrainStepFaithfulV`) through the generic IREE invoke.
    Inputs are raw f32 ByteArrays: `x` is `batch×d₀`, `W0` is `d₀×d₁`, `b0`
    is `d₁`; `y` is int32 `[batch]` (the one-hot is built in the C shim).
    Returns `W0n (d₀·d₁ f32) ++ b0n (d₁ f32)`.

    `nResident`: see `mlpTrainStepV`. Here it is **2** — `W0` and `b0`, i.e. the
    whole parameter set, since this graph returns exactly its two param inputs. -/
@[extern "lean_iree_linear_train_step"]
opaque linearTrainStepV
  (sess : @& LowererSession) (fnName : @& String)
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
  (sess : @& LowererSession) (fnName : @& String)
  (x : @& ByteArray) (params : @& ByteArray) (shapes : @& ByteArray) (y : @& ByteArray)
  (batch : USize) (d0 : USize) (d3 : USize) (replicas : USize)
  (nResident : USize := 0) (nShardTail : USize := 0) : IO ByteArray

/-- Drive the **verified-renderer** `@mlp_train_step`
    (`StableHLO.mlpTrainStepFaithfulV`) through the generic IREE invoke. `params` is
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
  (sess : @& LowererSession) (fnName : @& String)
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
  (sess : @& LowererSession) (packed : @& ByteArray) (nBytes : USize) : IO ByteArray

/-- The leading `nBytes` of the parameter state — θ of `[θ|m|v]` — for a caller that
    needs θ every step (the DQN's online-Q forward), where `readParams` is per epoch.
    With residency live it moves θ alone off the device; otherwise it is
    `packed.extract 0 nBytes`. `nBytes` must end on a tensor boundary, or it throws. -/
@[extern "lean_iree_read_params_prefix"]
opaque readParamsPrefix
  (sess : @& LowererSession) (packed : @& ByteArray) (nBytes : USize) : IO ByteArray

end LowererSession
