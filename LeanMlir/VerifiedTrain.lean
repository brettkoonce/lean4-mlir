import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.E4M3Quant

/-! # Shared driver for the `*-verified` trainers

Every `Main*Verified.lean` trains a network on **pre-rendered, audited** StableHLO
(`verified_mlir/<slug>_{train_step,fwd}.mlir`, emitted offline by `tests/Test*` from
the proof stack) through the IREE FFI. Unlike the reference `NetSpec`/`Train.lean`
path — which *generates* the MLIR at runtime — the verified path consumes a fixed
codegen artifact, so a verified "model definition" is just:

  * `slug`   — which `verified_mlir/*.mlir` + which `m.*` functions to invoke,
  * `specs`  — the param layout (`(dims, initKind)`, = the matching `XLayout.specs`),
  * `d0`     — per-example input width, and
  * `data`   — which dataset/loader to feed it.

The architecture itself lives in the renderer + the audited VJP theorems; it is
deliberately NOT re-expressed here. This file factors the ~100 lines of identical
boilerplate (compile → sessions → load → init → train/eval loop) that every trainer
used to copy. A trainer is now a `VerifiedNet` value + a `VerifiedConfig` + a one-line
`main`, mirroring the shape of `MainResnetTrain.lean`.

NB the learning rate is **baked into the rendered train-step MLIR** — `VerifiedConfig.lr`
is for the banner only; changing it does not change training (re-render to change lr).
-/

/-- Which dataset a verified trainer runs on. Picks the loader, the eval-split name,
    and whether the training images need a 256²→224² center-crop per batch. -/
inductive VerifiedData where
  /-- MNIST idx files directly under `dataDir` (28×28×1, no crop). -/
  | mnist
  /-- CIFAR-10 `.bin` records under `dataDir/cifar-10` (32×32×3, no crop). -/
  | cifar
  /-- Imagenette under `dataDir/imagenette` — train stored at 256² (center-cropped
      to 224² per batch), val at 224². -/
  | imagenette
  /-- **Full 1000-class ImageNet, streamed from the generated tfds shim** (handoff §2k).
      1,281,167 train / 50,000 val at 224².

      Unlike every case above, this one is NOT preloaded: at f32 the train split is ~938 GiB of host
      RAM against a 188 GB box, so it cannot be. Batches arrive over a pipe from **that net's own**
      `jax/.lake/build/generated_*_imagenet_shim.py` (`VerifiedNet.shimScript`), already
      augmented, mean/std-normalized and flattened to `(B, 3·224·224)` — **so the Lean side does no
      augmentation at all** for this dataset, which is the point: there is exactly one definition of
      the transform and it is the one the JAX reference trainer uses.

      ⚠ *Per net*, and that is the part that was wrong until 2026-08-02: the script was hardcoded to
      ResNet-34's, so every net got RRC+hflip regardless of what its reference asked for. The
      transform is still single-definition — it is generated from the same `TrainConfig` the
      reference trainer runs — but WHICH definition is now a property of the net.

      The VAL split IS preloaded (49,920 imgs after tfds `drop_remainder` ⇒ 30 GB, which fits), so
      the eval loop is unchanged. 49,920 is the same count the reference run reported. -/
  | imagenet
deriving BEq, Repr

/-- A verified trainer: a pinned codegen artifact (`slug`) + its param layout
    (`specs`, `d0`, `nClasses`) + the dataset to run it on. See the module docstring. -/
structure VerifiedNet where
  /-- Display name, e.g. `"ResNet-34"`. -/
  name     : String
  /-- ⚠ **Which directory this net's artifacts live in.** Default `verified_mlir/` — the CERTIFIED
      corpus, whose contents are pinned by `scripts/regen_verified_mlir.sh check` to exactly the set
      with a literal `IO.FS.writeFile "verified_mlir/…"` writer in `Proofs/Codegen/`.

      ⚠⚠ **The width/batch SWEEP nets set `.lake/build` instead, and that is what keeps the pin
      possible.** `mlpG`, `cnnG` and `cifar8BnG` render their artifact at run time from argv and
      immediately train on it — so those files are BUILD PRODUCTS, not committed renders. They used
      to be written into `verified_mlir/` and 74 of them had been checked in: never loaded by
      anything, regenerated on every invocation, and invisible to the writer audit because that
      audit greps for a LITERAL path and these writers interpolate a slug. A directory that mixes a
      certified corpus with transients cannot be audited as either.

      ⚠ It is a field rather than a global because the read sites are per-net and there are 30 of
      them; one spelling in `VerifiedNet` beats 30 in the driver. -/
  mlirDir : String := "verified_mlir"
  /-- Codegen slug: drives `<mlirDir>/<slug>_{train_step,fwd}.mlir`,
      `.lake/build/<slug>_{ts,fwd}_v.vmfb`, and the `m.<slug>_{train_step,fwd}` funcs. -/
  slug     : String
  /-- `(dims, initKind)` per param, in func-arg order — the matching `XLayout.specs`.
      `initKind`: 0 = He(fan-in), 1 = ones (γ), 2 = zeros (β / bias). -/
  specs    : Array (Array Nat × Nat)
  /-- Per-example flattened input width (e.g. `3 * 224 * 224`). -/
  d0       : Nat
  /-- Number of output classes. -/
  nClasses : Nat := 10
  /-- Dataset / loader selector. -/
  data     : VerifiedData
  /-- One-line intro printed at startup (the prose banner). Carries the literal `%LOWERER%`
      where the transport belongs; print it with `printBlurb`, never with `IO.println` directly,
      so the banner names the lowerer that actually ran. -/
  blurb    : String
  /-- **Does `<slug>_train_step.mlir` return the trailing report-only `%loss` scalar?**

      A per-RENDER fact, not a driver-wide one. `VerifiedNet.train` used to append the slot
      unconditionally, which was true only of `mlp` and `cnn` (the two re-rendered for the
      chapter-2/3 loss carve-out) and wrong for every other net on this driver — `resnet34`,
      `cifar8`, `cifar8_bn`, `cifar`, `cifar_bn`, `mobilenetv2`, `efficientnet`, `convnext`,
      `vit`. Those all return parameters only, so the driver offered one destination too many
      and the G4 arity gate refused to run them.

      A wrong value here cannot corrupt anything: G4 compares the module's real output count
      against what the driver supplies and refuses on any mismatch. -/
  lossSlot : Bool := false
  /-- Per-BN-layer channel counts, in forward order (empty for LayerNorm / no-BN nets). When
      non-empty, `trainAdamSched` threads running BN stats: the adam train step carries per-layer
      batch mean/var out in passthrough slots, the driver EMAs them, and eval uses
      `<slug>_fwd_eval.mlir` (affine BN with the running stats) instead of `<slug>_fwd.mlir`. -/
  bnChannels : Array Nat := #[]
  /-- **Stochastic-depth keep probabilities**, one per drop site, in the render's signature order
      (`planning/stochastic_depth.md`). Empty = the net has no drop sites, which is every net today
      except EfficientNet's `*sd` variants.

      ⚠ THE DRIVER OWNS THE RAMP, and that is deliberate rather than a shortcut: the emitted op is a
      pure per-example multiply and `1/keep_i` is folded into the value supplied here, because a
      BAKED `1/keep_i` and "the forward emits the sites too" cannot both hold (a ones scale would
      then compute `x/keep_i`, and the reference returns the branch untouched at eval). It is the
      same place `%lr` lives, for the same reason — one graph, many schedules.

      ⚠ It is therefore a SECOND hand-list against the renderer's `enetDropIdxs`, exactly like
      `toSpecs == XLayout.specs`. `tests/TestDropPathRamp.lean` is the `#guard` that pins the two;
      `VerifiedSpec` sits downstream of this file, so the renderer cannot share the definition by
      import without inverting the dependency. -/
  dropKeeps : Array Float := #[]
  /-- ▶ **CLASSIFIER DROPOUT** (`recipe_gaps.md` gap C) — `(keep_prob, per-example width)`, or
      `none` when the net has none. EfficientNet-B0: `(0.8, 1280)` for the reference's
      `dropout := 0.2` (`jax/MainEfficientNetImagenet.lean:68`).

      ⚠⚠ **THE WIDTH IS HERE BECAUSE THE MASK IS PER-ELEMENT, WHICH IS THE WHOLE DIFFERENCE FROM
      `dropKeeps`.** Stochastic depth's masks are `tensor<Bxf32>` — one value per example, so the
      driver needs no width at all. Dropout's is `tensor<B×w×f32>`, drawn per (example, feature),
      because the reference draws `bernoulli(key, keep, x.shape)` rather than the `(B, 1, …, 1)`
      shape. Every downstream difference — the blob shape, the draw count, the shard split — falls
      out of that one number, which is why it is carried rather than assumed to be `net.d0` or
      `nClasses`. It is the CLASSIFIER'S INPUT width (EfficientNet's head channels), independent of
      the class count, so the Imagenette and ImageNet renders take the same mask shape.

      ⚠ `keep_prob`, not the drop rate: `1/keep` is folded into the supplied mask by the driver, so
      the graph bakes no constant and the ones-mask forward is the exact identity
      (`Proofs.dropout_ones_id`). Same convention as `dropKeeps`, for the same reason. -/
  dropoutKeep : Option (Float × Nat) := none
  /-- **The generated ImageNet batch shim this net streams**, as a bare filename under
      `jax/.lake/build/` — e.g. `"generated_vit_tiny_imagenet_shim.py"`. Required on every
      `.imagenet` net; ignored (and empty) on every other dataset, which loads from disk.

      ⚠⚠ **THIS FIELD EXISTS BECAUSE ITS DEFAULT USED TO BE R34's, FOR EVERY NET.** `spawnShim`
      hardcoded `generated_resnet34_imagenet_shim.py` and `$SHIM_SCRIPT` was set nowhere, so a
      "verified EfficientNet / ViT / ConvNeXt ImageNet run" streamed **ResNet-34's** augmentation —
      RandomResizedCrop + hflip and nothing else. Their references do not: EfficientNet's sets
      `useAutoAugment`, ViT's sets RandAugment m9/mstd0.5/inc1 + random erasing + repeated aug ×3,
      ConvNeXt's sets RandAugment + random erasing. The capability was there all along
      (`JaxCodegen.generateShim` honours every one of those flags); what was missing was the
      wiring, so the recipe matrix read ✅ on a capability rather than on the state.

      **There is deliberately no fallback.** An empty value on an `.imagenet` net REFUSES at spawn
      rather than substituting anything, because the failure this replaces was silent: the wrong
      augmentation compiles, streams, trains and descends. `scripts/gen_shims.sh` writes all five;
      `$SHIM_SCRIPT` still overrides with an explicit path, for a hand-placed or probe shim. -/
  shimScript : String := ""

/-- Training hyperparameters — the `TrainConfig` of the verified path. Mirrors the
    reference `TrainConfig`; kept as its own object so a net is a (spec, config) pair. -/
structure VerifiedConfig where
  /-- Number of training epochs. -/
  epochs    : Nat
  /-- Minibatch size (a free runtime param — the MLIR's batch dim is dynamic). -/
  batchSize : Nat := 32
  /-- Learning rate. DISPLAY ONLY — baked into `<slug>_train_step.mlir`; changing it
      here does not change training (re-render the MLIR to change lr). -/
  lr        : Float := 0.1

namespace VerifiedNet

/-- Param shapes in func-arg order (= `specs` dims). -/
def paramShapes (n : VerifiedNet) : Array (Array Nat) := n.specs.map (·.1)
/-- Packed shape descriptors for the FFI (see `packShapes`). -/
def shapesBA (n : VerifiedNet) : ByteArray := packShapes n.paramShapes
/-- Total float count across all params. -/
def nParams (n : VerifiedNet) : Nat := (n.specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
/-- Packed `x` input shape `[batch, d0]`. -/
def xShape (n : VerifiedNet) (batch : Nat) : ByteArray := packXShape #[batch, n.d0]

end VerifiedNet

/-! ## `LEAN_MLIR_VARIANT`'s axis predicates — ONE definition each

`variant` encodes five independent axes and every consumer recovers each with a string test on
the name. `tests/TestVariantPredicates.lean` is the table of what each must read, and its
docstring is the history: the naming has collided three times, each time between a PAIR of
markers meeting rather than between a new marker and an old one.

⚠⚠ **THEY LIVE HERE BECAUSE THE TEST USED TO PIN COPIES.** `trainAdamSched` computed all five
inline and `TestVariantPredicates` declared its own `private def` of each, so the table gated a
transcription of the driver rather than the driver: an edit to the real predicate could not turn
that file red. That is `next_session_verified_trainer_code.md` §5's lesson one level up — a gate
on *a* definition is not a gate on *the* definition — and `scoreCheckpoint` needing the same
region arithmetic is what made a third copy the alternative.

▶ The `&& !net.dropKeeps.isEmpty` / `&& net.dropoutKeep.isSome` conjuncts stay at the call sites:
those are facts about the NET, not about the name, and folding them in here would make the
predicate untestable from a string alone. -/
namespace VerifiedVariant

/-- EMA shadow — a FOURTH `[θ|m|v|ema]` blob region, 5 scalars not 3. -/
def emaOn (v : String) : Bool := v.startsWith "ema"

/-- RMSProp — the mean-square slot initialises to **1.0**, not 0.
    ⚠ SUBSTRING, not prefix: the RMSProp+EMA spelling is `emarms`, which does not start with
    "rms" (`planning/ema.md`'s defect). -/
def rmsOn (v : String) : Bool := (v.splitOn "rms").length > 1

/-- Stochastic depth — N extra `tensor<Bxf32>` scale inputs.
    ⚠ The marker is `drop` and not `sd` because `rms` ++ `dp` spells `rmsdp`, which contains
    "sd" (`planning/stochastic_depth.md`'s defect). -/
def sdOn (v : String) : Bool := (v.splitOn "drop").length > 1

/-- Classifier dropout — ONE extra `tensor<B×wxf32>` mask input.
    ⚠ The marker is `do` and not `dropout` because `dropout` contains `drop`, so a dropout-only
    variant would read as a stochastic-depth one (`recipe_gaps.md` gap C). -/
def cdOn (v : String) : Bool := (v.splitOn "do").length > 1

/-- Gradient accumulation — a FOURTH `[θ|m|v|G]` region, 5 scalars.
    ⚠⚠ SUBSTRING, not prefix: RSB-A3's composed optimizer is `lambaccdp8x64bce`, where `lamb` ++
    `acc` puts the marker in the MIDDLE. -/
def accOn (v : String) : Bool := (v.splitOn "acc").length > 1

/-- `k`, read back out of the name. The graph has `1/k` BAKED in and the driver decides the apply
    cadence; a disagreement does not fail, it trains at a silently wrong effective learning rate.
    Parsed from AFTER the marker, not from a fixed offset — see `accOn`. -/
def accK (v : String) : Nat :=
  if accOn v then
    let after := (v.splitOn "acc").getD 1 ""
    let after := if after.startsWith "dp" then after.drop 2 else after
    ((after.takeWhile (· != 'x')).toNat?).getD 0
  else 1

/-- Blob regions: `[θ|m|v]`, plus a fourth for the EMA shadow or the gradient accumulator.
    ⚠ A 3-region file loaded by a 4-region driver (or the reverse) misaligns EVERY parameter, so
    every consumer of a checkpoint sizes off this rather than off a literal. -/
def nRegions (v : String) : Nat := if emaOn v || accOn v then 4 else 3

/-- Rank-0 scalar slots in the blob tail: `lr,bc₁,bc₂`, plus `%emad,%oemad` (EMA) or
    `%aup,%akeep` (accumulation). -/
def nScalars (v : String) : Nat := if emaOn v || accOn v then 5 else 3

end VerifiedVariant

/-- iree-compile one `.mlir` → `.vmfb`, surfacing failures. Skips when the `.vmfb` is already
    newer than the `.mlir` (a content-stable cache): avoids the ~minutes-long 224² recompile, and
    lets two same-net runs share one GPU-pair safely — they only *read* the cached vmfb (concurrent
    reads are fine; it's the concurrent *writes* of an identical compile that would race). -/
private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  if (← System.FilePath.pathExists outPath) then
    let srcMd ← (System.FilePath.mk mlirPath).metadata
    let outMd ← (System.FilePath.mk outPath).metadata
    if outMd.modified.sec ≥ srcMd.modified.sec then
      IO.println s!"  (cached vmfb) {outPath}"
      return
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

/-- Open a session for one Lean-emitted graph, on whichever backend this binary
    dlopened (`planning/xla_pjrt_ladder.md`).

    * **XLA** — hand the `.mlir` straight to PJRT, which compiles it in-process.
      Nothing is written to disk.
    * **IREE** — `iree-compile` the `.mlir` to a cache file first, then load that.

    The cache path is *derived* from `mlirPath` rather than passed in. All 58
    call sites used to supply one and every one of them computed the same thing
    from the same slug, so the argument was a second place for the name to be
    wrong and no place for it to be right. Deriving it also makes collisions
    impossible: two graphs cannot land on one cache file, which is the failure
    the target scoping below exists to prevent from the other direction.

    Both backends consume the *same* `verified_mlir/*.mlir` — the emitter, the
    spec, and the §1a ties are identical. Only the trusted lowerer differs. -/
def mkSession (mlirPath : String) : IO LowererSession := do
  if (← LowererSession.backendName) == "xla" then
    IO.println s!"  xla/pjrt {mlirPath}"
    LowererSession.create mlirPath
  else
    -- Scope the cache by IREE target. `compileVmfb` reuses any existing file that
    -- is newer than the .mlir, so an unscoped path lets an `IREE_BACKEND=rocm`
    -- artifact be picked up by an `IREE_BACKEND=llvm-cpu` run (and vice versa).
    -- That matters now that llvm-cpu is used as an independent numerical
    -- reference — see planning/xla_pjrt_ladder.md §8, rung 3.
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    let base := (mlirPath.splitOn "/").getLastD mlirPath
    let stem := if base.endsWith ".mlir" then base.dropRight 5 else base
    compileVmfb mlirPath s!".lake/build/{stem}_{target}.vmfb"
    LowererSession.create s!".lake/build/{stem}_{target}.vmfb"

/-- Init one parameter from its `(dims, initKind)` spec, matching the JAX reference's
    initialisers — they are the oracle these nets are paired against:

      * rank-4 conv kernel `[oc, ic, kH, kW]` → He **fan-OUT**, variance `2/(oc·kH·kW)`
      * rank-2 dense matrix `[in, out]`       → **Glorot**, variance `2/(in + out)`
      * γ = 1 (kind 1), β / bias = 0 (kind 2)

    ⚠ **Both weight cases CHANGED 2026-08-04.** This used variance `2/fan_in` for BOTH, where
    `jax/Jax/Codegen.lean` emits `uniform(±√(6/fan_out))` for convs (variance `2/fan_out` —
    torchvision's `kaiming_normal_(mode='fan_out', nonlinearity='relu')` convention for ResNet,
    `emitConvBnInit`) and `uniform(±√(6/(fan_in+fan_out)))` for dense (Glorot, `emitDenseInit`).
    **The two paths had therefore never agreed on init, on any net.** It is identical wherever
    `ic == oc`; the gaps are the stem (R34: fan_in 147 vs fan_out 3136 — **4.6× in σ**), every
    stage-entry conv and 1×1 projection (2×), and every classifier (R34: `2/512` vs `2/1512`,
    1.7× in σ).

    ⚠⚠ This moves the init of **every verified net**, so no previously recorded accuracy is
    reproducible from its seed any more. It changes **no committed artifact** — init is host-side
    and no `verified_mlir/` file mentions it.

    ⚠ The DISTRIBUTION still differs and is left alone deliberately: `F32.heInit` sums three
    uniforms (Bates-3, ≈ normal) where JAX draws one uniform. **Variance is matched; shape is
    not.** torchvision itself uses a normal here, so neither side is canonical on that axis, and
    changing the sampler would move every net for a second-order reason. -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    -- `heInit`'s output variance is exactly `scale²` (three uniforms on [-½,½], summed, ×2·scale).
    let variance :=
      if dims.size == 4 then 2.0 / (dims[0]! * dims[2]! * dims[3]!).toFloat   -- He, fan-OUT
      else if dims.size == 2 then 2.0 / (dims[0]! + dims[1]!).toFloat         -- Glorot
      else 2.0 / (dims[0]!).toFloat                                           -- rank-1: unchanged
    F32.heInit seed.toUSize n.toUSize (Float.sqrt variance)

/-- Load CIFAR-10 `.bin` records (3073 bytes: 1 label byte + 3072 image bytes).
    Returns f32 images `[n×3072]` (normalized) and int32-LE labels `[n×4]`. -/
private def loadCifarSplit (paths : List String) : IO (ByteArray × ByteArray × Nat) := do
  let mut raw : ByteArray := .empty
  let mut labels : ByteArray := .empty
  let mut nTotal : Nat := 0
  for p in paths do
    let batchRaw ← IO.FS.readBinFile p
    let n := batchRaw.size / 3073
    for j in [:n] do
      labels := labels.push batchRaw[j * 3073]!
      labels := labels.push 0; labels := labels.push 0; labels := labels.push 0
    raw := raw.append batchRaw
    nTotal := nTotal + n
  let imgs ← F32.cifarBatch raw 0 nTotal.toUSize
  return (imgs, labels, nTotal)

/-! ### The ImageNet batch shim (handoff §2k)

Reads batches from `JaxCodegen.generateShim`'s stdout. The shim owns the whole transform; this side
only frames bytes, which is why there is no augmentation code here. -/

/-- Read EXACTLY `n` bytes, looping until they arrive. A pipe read returns what is *available*, not
    what was asked for — at 154 MB per batch a short read is the normal case, not the edge case, and
    treating one `read` as a batch silently misaligns the stream from then on. -/
def readExact (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut acc := ByteArray.empty
  while acc.size < n do
    let chunk ← h.read (USize.ofNat (n - acc.size))
    if chunk.size == 0 then
      throw <| IO.userError s!"imagenet shim closed the pipe after {acc.size} of {n} bytes \
(did it crash? its stderr is not captured — run it standalone with SHIM_HASH=1 to see)"
    acc := acc ++ chunk
  pure acc

/-- Resolve a net's generated shim to a path on disk, or `none`. The candidate list is
    `spawnShim`'s, factored out so the two callers cannot disagree about WHICH file they are
    reading — one of them decides the wire, the other spawns it. -/
def resolveShimScript (shimScript : String) : IO (Option System.FilePath) := do
  let candidates : List System.FilePath := match ← IO.getEnv "SHIM_SCRIPT" with
    | some p => [p]
    | none   => if shimScript.isEmpty then []
                else [s!"jax/.lake/build/{shimScript}", s!".lake/build/{shimScript}"]
  candidates.findM? (fun p => System.FilePath.pathExists p)

/-- ⭐ **The `SHIM_MIX` default a generated shim BAKES** — read out of the producer rather than
    restated here. `""` when the shim cannot be found or declares nothing.

    ⚠ This is the mixup-λ lesson one layer up (§0.4 finding 3): *recover a constant by READING it,
    not by fitting or re-declaring it.* The alternative was a `useMixup` field on `VerifiedNet`
    duplicating what `generateShim` already baked from the same config — a second definition of one
    fact, which is the failure this repo keeps paying for. The shim text is the single source. -/
def shimMixDefault (shimScript : String) : IO String := do
  match ← resolveShimScript shimScript with
  | none => pure ""
  | some script => do
    let txt ← IO.FS.readFile script
    let key := "os.environ.get('SHIM_MIX', '"
    match txt.splitOn key with
    | _ :: rest :: _ => pure ((rest.splitOn "'").headD "")
    | _              => pure ""

/-- Spawn the shim for one split and consume its preamble.

    The preamble (`LMSH` | version | batch | flat) is checked rather than skipped: a batch or
    resolution mismatch between the render and the shim would otherwise read as garbage pixels and
    look like a broken net. Same reasoning as the FFI's G4 arity guard. -/
def spawnShim (shimScript : String) (split : String) (batch flat seed : Nat)
    (shard : Option (Nat × Nat) := none) (nclasses : Nat := 0) : IO IO.FS.Handle := do
  -- `shimScript` is the NET'S OWN generated shim (`VerifiedNet.shimScript`), not a shared default.
  -- An empty one refuses here rather than falling back: R34's shim was the fallback for years and
  -- it silently gave every other net R34's augmentation. See that field's docstring.
  if shimScript.isEmpty && (← IO.getEnv "SHIM_SCRIPT").isNone then
    throw <| IO.userError "imagenet shim: this net has no `shimScript`. Every .imagenet net must \
name the shim generated from ITS OWN reference recipe — there is no default, because the default \
used to be ResNet-34's and it silently streamed RRC+hflip to nets whose references use \
AutoAugment / RandAugment / random erasing / repeated augmentation. Set it on the VerifiedNetSpec \
(see `VerifiedNet.shimScript`), or point $SHIM_SCRIPT at an explicit path."
  -- `jax/` is its own lake project, so `--shim` writes under ITS build dir; a run from the repo
  -- root finds it there. $SHIM_SCRIPT overrides with a full path, for a hand-placed or probe shim.
  let candidates : List System.FilePath := match ← IO.getEnv "SHIM_SCRIPT" with
    | some p => [p]
    | none   => [s!"jax/.lake/build/{shimScript}", s!".lake/build/{shimScript}"]
  let some script ← candidates.findM? (fun p => System.FilePath.pathExists p)
    | throw <| IO.userError s!"imagenet shim not found (looked in {candidates}) — generate ALL \
five with `scripts/gen_shims.sh`, or one with `(cd jax && lake exe <net>-imagenet default \
--shim)`, or set $SHIM_SCRIPT"
  -- The interpreter must be the PINNED env (jax + tfds), not whatever `python3` is on PATH:
  -- the shim imports tensorflow_datasets and jax. $SHIM_PYTHON overrides; otherwise prefer the
  -- repo-local `.venv` (built off requirements-cuda-lock.txt) over the historical absolute path,
  -- which points into a checkout that no longer exists on any box.
  --
  -- ⚠ Checked for existence BEFORE spawning, on purpose. `IO.Process.spawn` on a missing cmd does
  -- not fail here in a way this code can see, so a bad interpreter used to surface downstream as
  -- `bad preamble magic "ResN"` — an error that blames the shim's wire format for a missing
  -- Python. Naming the real cause is the whole point of this check.
  let pyCandidates : List System.FilePath := match ← IO.getEnv "SHIM_PYTHON" with
    | some p => [p]
    | none   => [".venv/bin/python3",
                 "/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3"]
  let some py ← pyCandidates.findM? (fun p => System.FilePath.pathExists p)
    | throw <| IO.userError s!"imagenet shim: no python interpreter found (looked in \
{pyCandidates}). The shim needs the PINNED env (jax + tensorflow_datasets), so point \
$SHIM_PYTHON at it — a bare `python3` off PATH will not have tfds."
  -- `shard = some (i, n)` sets SHIM_SHARD=i/n, i.e. this worker emits only elements ≡ i (mod n).
  -- Absent ⇒ the variable is not set at all and the shim takes its unsharded path, which is why
  -- the single-producer stream is byte-identical to before sharding existed (gated by SHIM_HASH).
  let shardEnv : Array (String × Option String) := match shard with
    | some (i, n) => #[("SHIM_SHARD", some s!"{i}/{n}")]
    | none        => #[]
  -- `nclasses > 0` requests WIRE v2: the label section becomes `float32[batch*nclasses]` target
  -- distributions instead of `int32[batch]` hard labels. 0 (the default) leaves the variable unset
  -- and the shim emits v1 byte-for-byte, which is why every existing run is untouched.
  let softEnv : Array (String × Option String) :=
    if nclasses > 0 then #[("SHIM_NCLASSES", some (toString nclasses))] else #[]
  -- ── SHIM_MIX, and this only became load-bearing when the shims went per-net ──────────────────
  --
  -- A shim BAKES its config's mixing as the `SHIM_MIX` default: `off` for R34/mnv2/EfficientNet,
  -- **`both`** for ViT and ConvNeXt, whose references run mixup+cutmix. And a mixed target is a
  -- distribution, so the shim REFUSES it on wire v1 (`int32[batch]` cannot carry one) — on the
  -- TRAIN split only, since `_MIX_ON` is `and training`.
  --
  -- ⚠ Before the per-net wiring every net ran R34's shim and this could not arise. With it, a
  -- plain (wire v1) ViT/ConvNeXt ImageNet run would die at spawn — and the symptom is the useless
  -- `shim closed the pipe after 0 of 16 bytes`, because the child's stderr is not captured. Found
  -- by `scripts/shim_wiring_gate.py --stream`, before any trainer ran.
  --
  -- So: at v1, pass `SHIM_MIX=off` explicitly and SAY SO when the net's own default was not off.
  -- At v2 pass nothing — the shim's baked default is that net's reference recipe, which is the
  -- state we want. ⚠ Announced rather than silent: dropping a declared augmentation without
  -- saying so is the same "matrix reads capability, not state" defect this whole thread fixes.
  -- ⚠ ONE reader, shared with the trainer's soft-target decision (`shimMixDefault`). They must
  -- agree: the trainer decides the WIRE from this value and this call site decides the
  -- ANNOUNCEMENT, so two readings could announce one thing and stream another.
  let mixDefault ← if nclasses > 0 then pure "" else shimMixDefault shimScript
  let mixEnv : Array (String × Option String) :=
    if nclasses > 0 then #[] else #[("SHIM_MIX", some "off")]
  if nclasses == 0 && split == "train" && mixDefault != "" && mixDefault != "off" then
    match ← IO.getEnv "SHIM_MIX" with
    | some m =>
      if m.toLower != "off" then
        throw <| IO.userError s!"SHIM_MIX={m} needs wire v2: a mixed target is a distribution and \
this stream is int32 hard labels. Set SHIM_SOFT=1 (and the shim mixes by this net's own recipe), \
or SHIM_MIX=off."
    | none =>
      IO.println s!"  ⚠ this net's recipe declares SHIM_MIX={mixDefault}; wire v1 cannot carry a \
mixed target, so it is OFF for this run. SHIM_SOFT=1 turns on soft targets AND its mixing."
  let child ← IO.Process.spawn {
    cmd := py.toString, args := #[script.toString], stdout := .piped, stdin := .null,
    env := #[("SHIM_BATCH", some (toString batch)), ("SHIM_SPLIT", some split),
             ("SHIM_SEED", some (toString seed))] ++ shardEnv ++ softEnv ++ mixEnv }
  let h := child.stdout
  let pre ← readExact h 16
  let magic := String.ofList ((List.range 4).map (fun i => Char.ofNat (pre.get! i).toNat))
  if magic != "LMSH" then
    throw <| IO.userError s!"imagenet shim: bad preamble magic {magic.quote} (expected \"LMSH\")"
  let rd32 (off : Nat) : Nat :=
    (pre.get! off).toNat ||| ((pre.get! (off+1)).toNat <<< 8) |||
    ((pre.get! (off+2)).toNat <<< 16) ||| ((pre.get! (off+3)).toNat <<< 24)
  let ver := rd32 4; let sBatch := rd32 8; let sFlat := rd32 12
  -- ⚠⚠ v3/v4, not v1/v2: every batch now carries an int32 ROW COUNT before its labels. v1/v2 had
  -- no way to express a short final batch — see `readShimBatchPartial`. Refusing an old shim here
  -- is the point: a v1 stream read as v3 would take the first four label bytes as a row count.
  let wantVer := if nclasses > 0 then 4 else 3
  if ver != wantVer then
    throw <| IO.userError s!"imagenet shim: wire version {ver}, expected {wantVer} \
(nclasses={nclasses} ⇒ v{wantVer}). A v3 shim cannot serve soft targets and a v4 record read as v3 \
slides off by a factor of nClasses on every batch, so this refuses rather than reading garbage. \
⚠ v1/v2 are the PRE-ROW-COUNT framing — regenerate with scripts/gen_shims.sh."
  -- v4 appends `nclasses` to the preamble, so it is 20 bytes rather than 16. Read the tail HERE,
  -- not at the first record: the alignment error a missed field causes is silent and cumulative.
  if ver == 4 then
    let pre2 ← readExact h 4
    let sNC := (pre2.get! 0).toNat ||| ((pre2.get! 1).toNat <<< 8) |||
               ((pre2.get! 2).toNat <<< 16) ||| ((pre2.get! 3).toNat <<< 24)
    if sNC != nclasses then
      throw <| IO.userError s!"imagenet shim MISMATCH: shim sends nclasses={sNC}, the render wants \
{nclasses} — refusing rather than reading misaligned targets"
  if sBatch != batch || sFlat != flat then
    throw <| IO.userError s!"imagenet shim MISMATCH: shim sends batch={sBatch} flat={sFlat}, \
the render wants batch={batch} flat={flat} — refusing rather than reading misaligned pixels"
  -- ⚠ The SCRIPT is printed, not just the shape. Every net used to resolve to R34's shim and the
  -- banner said nothing about which one — so a run streaming the wrong augmentation looked exactly
  -- like a run streaming the right one. This line is what makes the wiring readable from a log.
  IO.println s!"  imagenet shim: {script} — {split} split, batch {sBatch}, {sFlat} floats/img \
(seed {seed}){if nclasses > 0 then s!", wire v{ver} soft targets [{batch}x{nclasses}]" else ""}"
  pure h

/-- One batch off the wire: `int32[batch]` labels then `float32[batch*flat]` images, in that order
    (the shim writes labels first so a partial record is detectable at the smaller read). -/
def readShimBatch (h : IO.FS.Handle) (batch flat : Nat) (nclasses : Nat := 0)
    : IO (ByteArray × ByteArray) := do
  -- `nclasses = 0` ⇒ v1: `int32[batch]`. Otherwise v2: `float32[batch*nclasses]`. The FFI accepts
  -- either without a flag — `lean_fill_targets` dispatches on the buffer's SIZE — so nothing
  -- downstream of here changes shape.
  -- ⚠ The int32 row count precedes every batch (wire v3/v4). This reader wants FULL batches — the
  -- train stream repeats forever, so a short one here is a torn write, not a tail. Checked rather
  -- than skipped: reading past a wrong count is the silent reframing v3 exists to prevent.
  let pre ← readExact h 4
  let rows := (pre.get! 0).toNat ||| ((pre.get! 1).toNat <<< 8) |||
              ((pre.get! 2).toNat <<< 16) ||| ((pre.get! 3).toNat <<< 24)
  if rows != batch then
    throw <| IO.userError s!"imagenet shim: batch declares {rows} rows, this reader wants {batch}. \
A short batch on a repeating stream is a torn write; use `readShimBatchPartial` for a split that \
ends (the val drain)."
  let lbl ← readExact h (if nclasses > 0 then 4 * batch * nclasses else 4 * batch)
  let img ← readExact h (4 * batch * flat)
  pure (img, lbl)

/-- Read up to `n` bytes, returning **what actually arrived** instead of throwing at EOF.
    The peer of `readExact`, and the only difference is which of "short read" and "clean end of
    stream" it treats as the error. -/
def readUpTo (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let mut acc := ByteArray.empty
  while acc.size < n do
    let chunk ← h.read (USize.ofNat (n - acc.size))
    if chunk.size == 0 then break
    acc := acc ++ chunk
  pure acc

/-- **One shim batch, tolerating a SHORT FINAL BATCH** — the validation-split reader.

    ⚠⚠ **Why this exists (2026-08-14).** The val pipeline used `drop_remainder=True`, so ImageNet's
    50,000 images batched at 256 gave 195 full batches and **80 images were thrown away**. Every
    top-1 this repo has quoted for an ImageNet net is therefore over **49,920**, where timm's
    `validate.py` scores all 50,000 — a difference of 0.16% that is not an error bar, it is a
    different denominator. The shim now sets `drop_remainder=training`, which puts a partial batch
    on the wire that `readExact` refuses by construction ("shim closed the pipe after N of M
    bytes"). This reader accepts it.

    Returns `(img, lbl, rows)` where `rows ≤ batch`, and `rows = 0` means the stream ended cleanly.

    ⚠ It reads LABELS FIRST, matching the wire order, and infers `rows` from the label read — the
    label record is 4 bytes (or `4·nclasses`) against the image's `4·flat`, so a truncated stream
    is far more likely to be caught mid-image than mid-label. Inferring from the SMALLER record and
    then demanding exactly that many image bytes turns a torn write into a loud failure instead of
    a silently short batch.

    ⭐ **No MLIR changes.** The eval graph keeps its baked batch width: `F32.sliceImagesPad`
    zero-pads the tail up to it and the eval loop scores `min bs (nEval − bi·bs)` real rows, so the
    pad never reaches the accuracy count. That is safe because eval normalises PER EXAMPLE
    everywhere — running-stat BN through `@<slug>_fwd_eval`, LayerNorm through `@<slug>_fwd`.
    ⚠ The one exception is `LEAN_MLIR_EVAL_BATCHSTATS=1`, which scores through `@<slug>_fwd` with
    BATCH statistics: there the zero rows WOULD shift the real rows' normalisation. That flag is a
    declared diagnostic, and the drain refuses to keep the tail under it. -/
def readShimBatchPartial (h : IO.FS.Handle) (batch flat : Nat) (nclasses : Nat := 0)
    : IO (ByteArray × ByteArray × Nat) := do
  let lblRec := if nclasses > 0 then 4 * nclasses else 4
  -- ⚠⚠ THE ROW COUNT IS READ, NOT INFERRED — and that is the whole of the v3 framing.
  -- This used to do `readUpTo (lblRec * batch)` and divide the byte count by the record size. A
  -- pipe does not preserve write boundaries, so at a PARTIAL tail that read ran straight through
  -- the labels and into the images: ImageNet val's 80-row tail is 320 label bytes, the read took
  -- 320 + 704, inferred rows = 256, and then demanded a full batch that was 704 bytes short of
  -- arriving. The reported "closed the pipe after 48168256 of 154140672 bytes" was exactly that.
  -- ▶ A `readUpTo` of 4 bytes is unambiguous in a way one of `lblRec * batch` can never be: at a
  -- clean end it returns 0, and otherwise the count says how much follows.
  let pre ← readUpTo h 4
  if pre.size == 0 then pure (ByteArray.empty, ByteArray.empty, 0)
  else if pre.size != 4 then
    throw <| IO.userError s!"shim sent {pre.size} bytes of the 4-byte row count — the stream is \
torn, not merely short"
  else
    let rows := (pre.get! 0).toNat ||| ((pre.get! 1).toNat <<< 8) |||
                ((pre.get! 2).toNat <<< 16) ||| ((pre.get! 3).toNat <<< 24)
    if rows == 0 || rows > batch then
      throw <| IO.userError s!"shim declared {rows} rows, outside 1…{batch} — refusing rather than \
reading a misframed batch"
    -- Both sides EXACT now: the shim has committed to `rows`, so a short read of either block is a
    -- torn write and must be loud.
    let lbl ← readExact h (lblRec * rows)
    let img ← readExact h (4 * rows * flat)
    pure (img, lbl, rows)

/-- Spawn `n` shim processes over disjoint shards of one split, and read them round-robin.

    **Why this exists.** One shim process tops out at ~1,530 img/s (measured 2026-08-01, bs128,
    marginal so TF startup is out of it). A 4-replica ViT step consumes 512 images in 264 ms, i.e.
    ~1,940 img/s, so a single producer would make the GPUs wait — the first config in this repo
    where the loader, not the device, is the ceiling (R34/ImageNet at bs256 needs only ~380).
    Measured aggregate: 2 processes **1.71×**, 4 processes **2.36×** on this 32-core box, so two
    clear the requirement with margin.

    **What it does NOT do is add a second definition of the transform.** Each worker runs the same
    generated shim with `SHIM_SHARD=i/n`, which selects *which examples* it emits (`ds.shard`,
    before the map) and leaves `_pp` — the crop, flip and normalization — untouched. A hand-written
    loader here would be §2a's double-writer disease applied to the data path.

    ⚠ **Round-robin over BATCHES is not the unsharded stream.** `ds.shard` interleaves elements, so
    taking whole batches from each worker in turn gives a different batch *composition* than one
    producer would. Both are valid shuffled streams over the same epoch of data, and each worker
    shuffles its own slice with the pipeline's own seed — but the two are not byte-comparable, so a
    determinism hash from the unsharded config does not carry to a sharded one. Re-run `SHIM_HASH`
    per shard if you need that property.

    ⚠ Each worker gets a DISTINCT seed (`seed + i`). With one shared seed every worker draws the
    same augmentation sequence, and since the shards hold different images that is not a
    correctness bug — but it needlessly correlates the crops across workers. -/
def spawnShimSharded (shimScript : String) (split : String) (batch flat seed n : Nat)
    (nclasses : Nat := 0) : IO (Array IO.FS.Handle) := do
  if n <= 1 then
    pure #[← spawnShim shimScript split batch flat seed none nclasses]
  else
    let mut hs : Array IO.FS.Handle := #[]
    for i in [0:n] do
      hs := hs.push (← spawnShim shimScript split batch flat (seed + i) (some (i, n)) nclasses)
    IO.println s!"  imagenet shim: {n} sharded producers (round-robin over batches)"
    pure hs

/-- Round-robin read: batch `k` comes from worker `k % n`. `readExact` already blocks until a whole
    record has arrived, so a slow worker throttles rather than corrupting — the framing cannot slip. -/
def readShimBatchRR (hs : Array IO.FS.Handle) (k batch flat : Nat) (nclasses : Nat := 0)
    : IO (ByteArray × ByteArray) := do
  match hs[k % hs.size]? with
  | some h => readShimBatch h batch flat nclasses
  | none   => throw <| IO.userError "readShimBatchRR: no shim producers were spawned"


/-- Load the train + eval splits for a dataset. Returns
    `(trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop?)` where
    `trainPix` is the stored per-example width of the *training* images (256² for
    Imagenette, `d0` otherwise) and `crop?` requests the 256²→224² center-crop.

    `evalOnly` skips the TRAIN split entirely — for `scoreCheckpoint`, which never touches it.
    ⚠ It is INERT on `.imagenet`, whose train split was never preloaded (it streams off the shim),
    and that is exactly why it is worth having on the others: Imagenette's is 9,469 × 256² × 3 f32
    = **7.4 GB** read and held for a job that only scores 3,925 val images. `nTrain` comes back 0
    under it, so a caller that starts using it gets a division rather than a plausible epoch. -/
private def loadData (net : VerifiedNet) (dataDir : String) (evalD0 : Nat := 0)
    (evalOnly : Bool := false) :
    IO (ByteArray × ByteArray × Nat × ByteArray × ByteArray × Nat × Nat × Bool) := do
  let d0 := net.d0
  -- `evalD0` is the EVAL forward's rendered input width, read off the artifact by the caller. It is
  -- only consulted on the `.imagenet` path (the only one that drains a val split off a shim), and
  -- `0` means "not supplied" ⇒ fall back to `net.d0`, which is what every non-split net wants.
  let evalD0 := if evalD0 == 0 then d0 else evalD0
  match net.data with
  | .imagenette =>
    let idir := dataDir ++ "/imagenette"
    -- Train split ships at 256² → randomCrop 256→224 + hflip (the training recipe);
    -- val ships at 224² (center crop). DEFAULT is 256²/crop, matching the reference
    -- trainer (Train.lean `imagenetteIO` hardcodes 256). Some dirs store the train
    -- split at 224² already (records of [1 label byte + 224·224·3 uint8]); for those
    -- set LEAN_MLIR_IMAGENETTE_TRAIN=224 to load 224²/no-crop (else: "short read").
    -- px also feeds trainPix (3·px²) and crop := (px == 256).
    let px := ((← IO.getEnv "LEAN_MLIR_IMAGENETTE_TRAIN").bind (·.toNat?)).getD 256
    let (trI, trL, nTr) ← if evalOnly then pure (ByteArray.empty, ByteArray.empty, 0)
                          else F32.loadImagenetteSized (idir ++ "/train.bin") px.toUSize
    let (evI, evL, nEv) ← F32.loadImagenette (idir ++ "/val.bin")
    return (trI, trL, nTr, evI, evL, nEv, 3 * px * px, px == 256)
  | .mnist =>
    let (trI, nTr) ← if evalOnly then pure (ByteArray.empty, 0)
                     else F32.loadIdxImages (dataDir ++ "/train-images-idx3-ubyte")
    let (trL, _)   ← if evalOnly then pure (ByteArray.empty, 0)
                     else F32.loadIdxLabels (dataDir ++ "/train-labels-idx1-ubyte")
    let (evI, nEv) ← F32.loadIdxImages (dataDir ++ "/t10k-images-idx3-ubyte")
    let (evL, _)   ← F32.loadIdxLabels (dataDir ++ "/t10k-labels-idx1-ubyte")
    return (trI, trL, nTr, evI, evL, nEv, d0, false)
  | .cifar =>
    let cdir := dataDir ++ "/cifar-10"
    let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
    let (trI, trL, nTr) ← if evalOnly then pure (ByteArray.empty, ByteArray.empty, 0)
                          else loadCifarSplit trainPaths
    let (evI, evL, nEv) ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
    return (trI, trL, nTr, evI, evL, nEv, d0, false)
  | .imagenet =>
    -- Train is NOT loaded here — it is streamed per step (`trainAdamSched`). Only `nTrain` matters
    -- from this side, and it is the tfds count, which is what sets steps/epoch.
    --
    -- Val IS drained into RAM, once, so the eval loop below is untouched: 195 batches × 256 after
    -- tfds `drop_remainder` = 49,920 images = 30.0 GB. That is the SAME count
    -- `jax/runs/r34_imagenet_bf16_90ep/RESULTS.md` reports for its in-training val, so the two
    -- paths score the identical set.
    -- ⚠⚠ THIS IS THE **VAL** WIDTH, AND IT IS NOT ALWAYS THE TRAIN WIDTH. ImageNet val is drained
    -- at the EVAL forward's rendered width, which under RSB-A3 is 224² while TRAIN is 160²
    -- (`resnet50Imagenet160Verified`). Until 2026-08-06 this was the literal `3*224*224` AND was
    -- returned as `trainPix` too, so the train stream inherited the val resolution: the 160 net
    -- asked its shim for 150,528 floats/img while the shim correctly sent 76,800, and the wire
    -- guard refused ("shim sends batch=64 flat=76800, the render wants batch=64 flat=150528").
    -- ▶ Now READ OFF THE ARTIFACT by the caller and passed in, exactly like `evalBs` — so the val
    -- drain, the eval invoke's `xShape` and the eval slicer all size from one parse of one
    -- declaration, and a 224 net still gets 150,528 because that is what its eval render says.
    let evalFlat := evalD0
    let flat := evalFlat
    let vb := 256
    -- ⚠⚠ **NO LONGER A HARDCODED 195, 2026-08-14.** This read `nB := 195` — "what
    -- `drop_remainder` leaves of 50,000" — so the drain stopped at 49,920 and **80 val images were
    -- silently discarded**, on every ImageNet net, in every number this repo has quoted.
    -- `validate.py` in timm scores all 50,000, so ours were over a different denominator: not an
    -- error bar, a different measurement. `nB` is now an UPPER BOUND and the terminator is the
    -- closed pipe, which is what the comment below always claimed it was.
    --
    -- ⚠ The bound is deliberately loose (`196` = ⌈50000/256⌉, +1 of slack) and the loop exits on
    -- `rows = 0`. A tight equality here is the same fragility being removed: `.batch(256)` over
    -- 50,000 gives 196 batches only at THIS `vb`, and `vb` is a local constant that has changed
    -- before.
    let nB := 196
    -- ⚠ The VAL split takes the same per-net shim as train. It streams the center-crop path
    -- (`training=False` ⇒ no RRC, no AutoAugment/RandAugment, no erasing), so the two nets whose
    -- pipelines differ only in TRAIN augmentation drain an identical val set — but the crop rule
    -- itself is per-config (`testCropRatio`), so the script still has to be the net's own.
    let h ← spawnShim net.shimScript "validation" vb flat 0
    IO.println "  imagenet: draining the val split into RAM (~30 GB, one time)…"
    -- Reserve the final size and append each batch as it lands, rather than collecting all
    -- ~196 chunks and folding `(· ++ ·)` over them at the end. The fold cost ~45 GB of pure
    -- overhead on a 28 GiB result: the chunk array stayed live for the whole fold (30 GB) WHILE
    -- the accumulator grew beside it, and `++` is `copySlice … (exact := false)`, i.e. it DOUBLES
    -- capacity when it grows — so the last few appends allocated a 2× buffer before freeing the
    -- old one. Measured peak RSS was 75.5 GB. Pre-sizing removes both: `++` into a buffer that
    -- already has the capacity never reallocates, so peak is the result plus one 154 MB batch.
    let mut evI := ByteArray.emptyWithCapacity (nB * vb * flat * 4)
    let mut evL := ByteArray.emptyWithCapacity (nB * vb * 4)
    let mut n := 0
    -- The validation iterator neither shuffles nor repeats, so it ends; the closed pipe IS the
    -- terminator, and `readShimBatchPartial` reports it as `rows = 0` rather than throwing.
    -- ▶ `LEAN_MLIR_EVAL_BATCHSTATS=1` scores through `@<slug>_fwd` with BATCH statistics, where
    -- the zero-padded tail of a partial final batch would shift the real rows' normalisation. It
    -- is a declared diagnostic, so the tail is DROPPED under it rather than silently mis-scored —
    -- and the drop is announced, because a denominator that changes with a debug flag is exactly
    -- the kind of thing that gets quoted later without the flag.
    let dropTail := (← IO.getEnv "LEAN_MLIR_EVAL_BATCHSTATS").isSome
    let mut ended := false
    for _ in [0:nB] do
      if !ended then
        let (i, l, rows) ← readShimBatchPartial h vb flat
        if rows == 0 then ended := true
        else if rows < vb && dropTail then
          IO.println s!"  ⚠ LEAN_MLIR_EVAL_BATCHSTATS: dropping the {rows}-image tail — batch-stat \
scoring cannot see a zero-padded batch. This run's eval denominator is {n}, not 50,000."
          ended := true
        else
          evI := evI ++ i; evL := evL ++ l; n := n + rows
    IO.println s!"  imagenet: val ready — {n} images, {evI.size / 1048576} MB"
    -- ⭐ ANNOUNCED, because it is the denominator every top-1 from this run divides by, and it
    -- moved on 2026-08-14 (49,920 → 50,000). A number quoted against the old denominator is not
    -- comparable to one quoted against timm's, and nothing else in the output says which was used.
    if n != 50000 then
      IO.println s!"  ⚠ val is {n} of ImageNet's 50,000 — top-1 here is NOT over timm's denominator"
    else
      IO.println "  ▸ val = all 50,000 (timm's denominator; drop_remainder=False on the val split)"
    -- ▶ `trainPix` is `net.d0`, the width of the images the TRAIN stream carries — 150,528 for every
    -- 224 net (so this is INERT for all of them: `net.d0 == 3*224*224` exactly) and 76,800 for the
    -- 160 net. It is deliberately not `evalFlat`; see the warning above.
    return (ByteArray.empty, ByteArray.empty, 1281167, evI, evL, n, net.d0, false)

/-- Synthetic-input data for the `lake run benchmark` probes (`LEAN_MLIR_BENCH_SYNTH`):
    ONE constant batch, reused every step, but with the dataset's *real* `nTrain` so the
    per-epoch step count — and thus the per-epoch / per-step timing — matches the on-disk
    anchors (train-step throughput is value-independent). Lets the benchmark run with zero
    data downloaded. The per-step crop/hflip stays in the loop (so timing matches); eval is
    skipped in synth, so `nEval` is a placeholder. -/
private def mkSynthData (data : VerifiedData) (d0 bs : Nat) :
    IO (ByteArray × ByteArray × Nat × ByteArray × ByteArray × Nat × Nat × Bool) := do
  let (nTr, px, crop) := match data with
    | .imagenette => (9469, 3 * 256 * 256, true)   -- 256² pre-crop → 224² each step
    | .cifar      => (50000, d0, false)
    -- ⚠ ImageNet needs its OWN case, and until 2026-08-05 it fell through to mnist's 60,000 —
    -- so a synthetic ImageNet epoch was 234 steps where the real one is 5,004. Invisible to the
    -- `MAX_STEPS` probes (which cap far below either) and wrong for anything reading steps/epoch.
    -- The shim already delivers 224² pre-augmented, so there is no host-side crop here.
    | .imagenet   => (1281167, d0, false)
    | _           => (60000, d0, false)             -- mnist
  let img ← F32.const (bs * px).toUSize 0.1
  let lbl ← F32.const bs.toUSize 0.0               -- bs int32 zero labels (4 bytes each)
  pure (img, lbl, nTr, img, lbl, bs, px, crop)

/-- **Where this (net, variant) writes and resumes its checkpoint.**

    Scoped by BACKEND: without the suffix an XLA run would happily resume from an IREE checkpoint
    and vice versa, silently fusing two trajectories into one while looking completely normal on
    screen (`planning/xla_pjrt_ladder.md` §3). `$LEAN_MLIR_CKPT_TAG` appends a run-scoped suffix —
    without it every pass of the same (net, variant, backend) shares ONE path, so the parallel
    sweeps `planning/chapter_makeover.md` §3c mandates cannot be run: concurrent passes clobber
    each other's blob, and a later pass resumes from an earlier one's finished epoch 40.

    ⚠ **A function because `scoreCheckpoint` has to land on the SAME path the trainer wrote**, and
    "score the checkpoint the run just finished" is that tool's zero-argument case. Two spellings
    of this string would not fail — they would score a file that is not there, or worse, an older
    one from a different tag. -/
def VerifiedNet.ckptPathFor (net : VerifiedNet) (variant : String) : IO String := do
  let backend ← LowererSession.backendName
  let ckptTag := match ← IO.getEnv "LEAN_MLIR_CKPT_TAG" with
    | some t => if t.isEmpty then "" else "_" ++ t
    | none   => ""
  return s!".lake/build/{net.slug}_{variant}_ckpt{if backend == "xla" then "_xla" else ""}{ckptTag}.bin"

/-- Print the startup banner with `%LOWERER%` resolved to the lowerer that actually ran.

    Every net's `blurb` used to hard-code its transport, so the banner was a claim about the
    build rather than about the run. Three of the seven print sites patched it at run time with
    a `.replace "IREE FFI" "XLA/PJRT"` and the other four printed it raw, which meant a net whose
    blurb still said IREE announced IREE while training on XLA. The placeholder moves the decision
    to one place, and every caller is correct by construction. -/
def VerifiedNet.printBlurb (net : VerifiedNet) : IO Unit := do
  let transport := if (← LowererSession.backendName) == "xla" then "XLA/PJRT" else "IREE FFI"
  IO.println (net.blurb.replace "%LOWERER%" transport)

/-- Train a `VerifiedNet` end-to-end on its proof-rendered StableHLO: compile both
    MLIRs → lowerer sessions → load data → He/spec init → SGD train + eval loop. The
    SGD update (and lr) are baked into `<slug>_train_step.mlir`; we only feed batches. -/
def VerifiedNet.train (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  net.printBlurb
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let synth := (← IO.getEnv "LEAN_MLIR_BENCH_SYNTH").isSome
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    if synth then mkSynthData net.data d0 bs else loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} ({net.specs.size} params, {net.nParams} floats), mean-loss SGD lr={cfg.lr}, He init{if synth then " [SYNTH]" else ""}"
  (← IO.getStdout).flush
  -- LEAN_MLIR_MAX_STEPS caps batches per epoch. Needed to run gate G2 at small
  -- N: over a full run, ReLU branch flips amplify f32 noise, so a large final
  -- divergence is ambiguous between chaos and a plumbing bug. Diffing at 1 / 10
  -- / 100 steps separates them — see planning/xla_pjrt_ladder.md §8.
  let nbFull := nTrain / bs
  let nb := match (← IO.getEnv "LEAN_MLIR_MAX_STEPS").bind (·.toNat?) with
    | some n => min n nbFull
    | none   => nbFull
  let nbt := (nEval + bs - 1) / bs   -- ceil: the last partial batch is zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  -- Device-resident parameters (handoff §2d.3). **Every** param is resident here,
  -- not a prefix: this loop's step is `params ← trainStep(x, params, y)` and the
  -- host reads NOTHING out of the result per step — no loss slot, no BN stats, the
  -- whole blob is handed straight back. So the resident block is the entire tensor
  -- list, and `@<slug>_train_step` returns exactly those tensors in exactly that
  -- order (the packed-output walk in the shim already assumes it).
  --
  -- ⚠ This loop was explicitly OUT of §2d.3's original scope — *"`train`/`trainLinear`
  -- stay on the copying path, they are the demo loops, not the throughput ones"*.
  -- That was written before §2d.3's own measurement found the demo nets to be the
  -- MOST transfer-bound in the set (the dense probe at **75%**, against R34's 55%)
  -- and before residency measured **3.1×** on cifar8-bn. These loops are what a
  -- reader sits and watches, so this is an interactivity win rather than a
  -- throughput one — §2d.3's "the surprise worth carrying".
  --
  -- A REQUEST, not a mode: honoured only under `$PJRT_FFI_RESIDENT=1`, and the
  -- copying path stays the default and byte-identical.
  let nResident := net.paramShapes.size.toUSize
  -- init params in func-arg order from the layout specs (one seed per slot).
  -- Seed base is overridable via LEAN_MLIR_SEED (default 1) to probe how
  -- sensitive convergence is to the specific He-init draw.
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  -- LEAN_MLIR_PERTURB_R: displace the initial parameters along a random unit vector of exact L2
  -- norm r, in units of 1e-9 (no `String.toFloat?` in this toolchain), before any training. Same
  -- knob and same spelling as `trainAdamSched`; it was implemented ONLY there, which made
  -- `scripts/residency_gate.sh`'s init CONTROL a silent no-op for every net on this loop —
  -- the gate caught that itself and refused as VACUOUS rather than reporting a green.
  let params0 := F32.concat parts
  let mut params ← match (← IO.getEnv "LEAN_MLIR_PERTURB_R").bind (·.toNat?) with
    | some n => do
        let r := n.toFloat * 1e-9
        IO.println s!"  ▸ PERTURBED init: theta += r*u with ||r*u||_2 = {r}"
        F32.perturbUnit params0 0 net.nParams.toUSize r 12345
    | none   => pure params0
  -- LEAN_MLIR_MAX_EPOCHS caps the epoch count (opt-in; absent → full cfg.epochs).
  -- Used by `lake run benchmark` to probe steady-state per-epoch wall-clock with
  -- only a few epochs; harmless otherwise (timing per epoch is LR-independent).
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  -- ⭐ THE LOSS SLOT. The train-step render returns a trailing report-only `%loss`
  -- scalar (`MlpRender`, and the `%lslot` note there for why it is also an input).
  -- `tsShapes` therefore declares ONE more tensor than `shapes`: a rank-0 scalar,
  -- spelled `#[]`. `shapes` stays parameter-only because the eval forward below
  -- takes it and has no such slot.
  -- ⚠ The packed blob is `[θ | loss]`, so every parameter tensor still LEADS it and
  -- the resident prefix is unchanged — residency retains the first `nResident`
  -- tensors and the host now reads exactly one float off the tail.
  -- ⚠ Gated on `net.lossSlot`: only the renders that actually emit the scalar get the extra
  -- destination. See the field's docstring for the nine nets this was silently wrong for.
  let tsShapes := packShapes (if net.lossSlot then net.paramShapes ++ #[#[]] else net.paramShapes)
  if net.lossSlot then
    params := F32.concat #[params, ← F32.const 1 0.0]
  for ep in [0:nEpochs] do
    let tEp0 ← IO.monoMsNow
    let mut epochLossSum := 0.0
    for bi in [0:nb] do
      let xbRaw := if synth then trainImg else F32.sliceImages trainImg (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := if synth then trainLbl else F32.sliceLabels trainLbl (bi * bs) bs
      params ← LowererSession.mlpTrainStepV tsSess tsFn
                  xb params tsShapes yb bs.toUSize d0.toUSize nc.toUSize nResident
      if net.lossSlot then
        epochLossSum := epochLossSum + (F32.read params net.nParams.toUSize)
    -- Bring the parameters back to host for eval and for the G2 dump. Without
    -- residency this is the copy `params` already was, so the line is inert;
    -- with it, this is the ONE d2h per epoch that remains. It is placed outside
    -- the `if !synth` because the dump below reads `params` whether or not eval ran.
    params ← LowererSession.readParams tsSess params (net.nParams * 4).toUSize
    -- ⚠ `readParams` returns the PARAMETER prefix only, so the loss slot the step
    -- writes is dropped here. Put it back, or the next epoch feeds `tsShapes`
    -- (nParams+1 tensors) a blob holding nParams and the shim walks off the end.
    -- The value is irrelevant going in; the step overwrites it.
    params := F32.concat #[params, ← F32.const 1 0.0]
    let mut correct := 0
    if !synth then          -- synth probe: skip eval (no eval split on disk)
      for bi in [0:nbt] do
        let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
        -- Hold the parameters on device across the eval batches (§2d.3). `ep+1` is
        -- the generation token: `params` changes exactly once per epoch and this
        -- says so, so the held set cannot go stale.
        let logits ← LowererSession.forwardF32 fwdSess fwdFn params shapes
                        xb xShape bs.toUSize nc.toUSize
                        nResident (ep + 1).toUSize
        for j in [0:min bs (nEval - bi * bs)] do   -- score real rows only, not the pad
          let pred := (F32.argmaxN logits (j * nc).toUSize nc.toUSize).toNat
          let lbl  := F32.readLabel evalLbl (bi * bs + j)
          if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / nEval.toFloat * 100.0
    let epMs := (← IO.monoMsNow) - tEp0
    -- ⚠ Only nets whose render carries the `%loss` scalar have a loss to report. Printing
    -- `loss = 0.000000` for the others would put a fabricated number in a captured log, so
    -- the field is omitted instead. See `VerifiedNet.lossSlot`.
    let lossField := if net.lossSlot then s!"loss = {epochLossSum / nb.toFloat}, " else ""
    IO.println s!"  epoch {ep + 1}: {lossField}{evalName}_acc = {correct}/{nEval} = {acc}% ({epMs}ms)"
    (← IO.getStdout).flush
  -- Gate G2 (`planning/xla_pjrt_ladder.md` §3): dump the packed params so the IREE
  -- and XLA builds can be diffed tensor-for-tensor. He init runs in Lean from a
  -- fixed seed, so both backends start byte-identical without extra work.
  match ← IO.getEnv "LEAN_MLIR_DUMP_PARAMS" with
  | some path =>
      IO.FS.writeBinFile path params
      IO.println s!"  wrote final params ({params.size} bytes) → {path}"
  | none => pure ()
  IO.println s!"done (trained {net.name} via the proof-rendered StableHLO)."

/-- **AdamW training driver** — threads the first/second moment buffers as a single
    packed `[θ|m|v]` param blob through the generic FFI (`n_params = 3k`; the moments
    ride in the params slot, so the prebuilt `.so` is unchanged), against the
    baked-hyperparameter packed render `@<slug>_adam_train_step`
    (`ViTRender.vitTrainStepModuleAdamPacked`, optimizer = `Proofs.adamWParam`).
    Moments init to 0; eval reads the θ slice (first `nParams` floats). The Adam
    analogue of `VerifiedNet.train`. -/
def VerifiedNet.trainAdamPacked (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  net.printBlurb
  let tsVmfb  := s!".lake/build/{net.slug}_adam_ts.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_adam_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"             fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} AdamW (packed θ|m|v), He init"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: the last partial batch is zero-padded, not dropped
  -- θ|m|v packed: θ = He-init (one seed per slot, as `train`), m = v = 0. The
  -- shapes descriptor lists every tensor three times (θ, then m, then v).
  let adamShapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes)
  let fwdShapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_adam_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let theta := F32.concat parts
  let zeros ← F32.const net.nParams.toUSize 0.0
  let mut params := F32.concat #[theta, zeros, zeros]
  let pBytes := net.nParams * 4
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xbRaw := F32.sliceImages trainImg (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      params ← LowererSession.mlpTrainStepV tsSess tsFn
                  xb params adamShapes yb bs.toUSize d0.toUSize nc.toUSize
    let thetaCur := params.extract 0 pBytes
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn thetaCur fwdShapes
                      xb xShape bs.toUSize nc.toUSize
      for j in [0:min bs (nEval - bi * bs)] do   -- score real rows only, not the pad
        let pred := (F32.argmaxN logits (j * nc).toUSize nc.toUSize).toNat
        let lbl  := F32.readLabel evalLbl (bi * bs + j)
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / nEval.toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nEval} = {acc}%"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} with AdamW via packed θ|m|v threading)."

/-- The batch a forward artifact was **rendered at**, read out of its own `%x:` signature.

    Batch is baked into a render, not a runtime dimension, so the eval forward has a fixed width
    that need not equal the training batch — `LEAN_MLIR_BATCH=128` trains at 128 while every
    Imagenette `_fwd{,_eval}` is rendered at 32. Feeding a 128-wide slice to a 32-wide graph is a
    shape error at the first invoke, which is why `LEAN_MLIR_SKIP_EVAL` existed as the only way out.

    Reading the width off the artifact removes that trade-off, and it is **sound because eval is
    class-batch-independent by construction**: the BN nets score through `@<slug>_fwd_eval`, which
    is frozen-running-stat affine BN and performs *no* reduction over the batch (handoff §2g — the
    very property that made `mobilenetv2_fwd_eval` immune to the skew that hit `mobilenetv2_fwd`),
    and the others normalise per example (LayerNorm) or not at all. So the eval batch decides only
    how many rows ride per invoke; it cannot move a per-example logit.

    Returns `none` rather than guessing if the signature does not parse — the caller falls back to
    the training batch, i.e. exactly the old behaviour. -/
private def fwdRenderedBatch (path : String) : IO (Option Nat) := do
  if !(← System.FilePath.pathExists path) then return none
  let txt ← IO.FS.readFile path
  match txt.splitOn "%x: tensor<" with
  | _ :: rest :: _ => return (rest.takeWhile (· != 'x')).toNat?
  | _ => return none

/-- The eval forward's rendered input shape, `(batch, d0)`, off `%x: tensor<BxWxf32>`.

    ⭐ **BOTH numbers come from ONE parse of ONE declaration**, deliberately. `evalBs` was already
    read off the artifact rather than assumed; the WIDTH has to be too, and a second parser of the
    same text is the double-writer failure in miniature — the two could then disagree about the same
    tensor.

    ▶ Why the width is not `net.d0`: under RSB-A3 the eval resolution is **not** the train
    resolution. `resnet50in160_fwd_eval.mlir` declares `tensor<256x150528xf32>` (224² eval) while
    `resnet50in160_fwd.mlir` declares `tensor<64x76800xf32>` (160² train) — the split is already
    rendered into the artifacts, and this is what lets the driver honour it. For every 224 net the
    two coincide, so this returns `(bs, net.d0)` there and nothing downstream moves. -/
private def fwdRenderedShape (path : String) : IO (Option (Nat × Nat)) := do
  if !(← System.FilePath.pathExists path) then return none
  let txt ← IO.FS.readFile path
  match txt.splitOn "%x: tensor<" with
  | _ :: rest :: _ =>
    let b := (rest.takeWhile (· != 'x')).toNat?
    -- `rest` is "BxWxf32>…": drop "Bx", then take up to the next 'x'.
    let afterB := (rest.dropWhile (· != 'x')).drop 1
    let w := (afterB.takeWhile (· != 'x')).toNat?
    match b, w with
    | some b, some w => return some (b, w)
    | _, _ => return none
  | _ => return none

/-- **Scheduled AdamW driver** (Phase 2) — `trainAdamPacked` with a runtime LR and
    bias correction. `lr`/`bc₁`/`bc₂` ride as three rank-0 scalar params in the blob
    tail (`[θ|m|v|lr|bc₁|bc₂]`, the FFI takes no scalar slot) and are returned
    unchanged; the host recomputes them each step: cosine decay + linear warmup for
    `lr`, and `bc₁=1−β₁ᵗ`, `bc₂=1−β₂ᵗ` (proper bias correction). Drives
    `ViTRender.vitTrainStepModuleAdamSched`.

    **`expDecayRate > 0` selects the EfficientNet/MobileNetV2 exponential schedule**
    over cosine: after warmup, `lr = baseLR · rate^((epoch − warmupEpochs)/decayEpochs)`.
    Both references use it (mnv2 ×0.98 per epoch, EfficientNet ×0.97 every 2.4), and it
    is `recipe_gaps.md` Tier C — a driver item, not a render one, because `lr` is already
    a runtime operand. Default 0.0 keeps cosine, so every existing call site is unchanged.

    **RMSProp variants also need a different INITIAL STATE**, which is the other half of
    that gap and is handled below — see `rmsprop`. -/
def VerifiedNet.trainAdamSched (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (baseLR β1 β2 : Float) (warmupEpochs : Nat) (variant : String := "adam")
    (expDecayRate : Float := 0.0) (expDecayEpochs : Float := 1.0)
    (emaDecay : Float := 0.9999) : IO Unit := do
  -- `variant` selects the rendered train step `@<slug>_<variant>_train_step` (and its artifact /
  -- vmfb / checkpoint names). Default "adam" = the AdamW render; "mom" = the Nesterov-momentum SGD
  -- render (same packed [θ|m|v]+lr/bc1/bc2 signature; the momentum step ignores the m/bc slots and
  -- reads only lr + v, so this driver is shared verbatim). β1/β2 still drive the (unused-by-mom)
  -- bias-correction scalars; the cosine+warmup lr schedule is identical.
  --
  -- "rms" = the RMSProp-with-momentum render (`Proofs/Codegen/RmsPropStep.lean`), which reuses the
  -- SAME packed slots with `m` = the momentum BUFFER and `v` = the running MEAN-SQUARE — the
  -- signature is byte-identical to the net's AdamW peer apart from the entry name, and `%bc1`/`%bc2`
  -- ride through unread. So the only thing this driver owes it is the INITIAL STATE, below.
  --
  -- ⚠ The prefix test is the reverse of `{mnv2,enet}AdamVariant`, whose `.rmsprop` branch returns
  -- "rms"/"rmsdp" (+ the per-device batch): "rms", "rms64", "rmsdp64". That direction is pinned by
  -- the `#guard`s beside each renderer's `#eval`, so the two cannot drift apart silently.
  -- ⚠ SUBSTRING, not prefix, and this is a bug caught before it shipped. Optimizer and EMA are
  -- INDEPENDENT axes in EfficientNet's variant name, so the RMSProp+EMA spelling is `emarms` —
  -- which does NOT start with "rms". A prefix test silently classifies it as non-RMSProp, and the
  -- failure is not loud: the mean-square would initialise to 0 instead of 1.0, i.e. exactly the
  -- much-larger-first-step defect the RMSProp driver work exists to fix, reintroduced by a naming
  -- interaction. The variant strings are pinned by `#guard`s beside each renderer's `#eval`s.
  let rmsprop := VerifiedVariant.rmsOn variant
  -- "ema"/"emadp" = the EMA-shadow render (`planning/ema.md`), whose blob carries a FOURTH region:
  -- `[θ|m|v|ema]`, with the scalar tail 3 → 5 (`%emad`, `%oemad`). Everything below that indexes the
  -- blob is written against `nRegions`/`nScalars` rather than a literal 3, because a 4-region graph
  -- fed a 3-region blob is not a subtle numeric error — it is every parameter misaligned.
  --
  -- ⚠ Keyed off the variant PREFIX, the same reverse-of-`cnxAdamVariant` reading `rmsprop` uses,
  -- and pinned upstream by the `#guard`s beside that renderer's `#eval`s.
  let emaOn := VerifiedVariant.emaOn variant
  -- ⭐⭐ GRADIENT ACCUMULATION (`planning/next_session_pipeline_then_r50.md` §4). "acc<k>x<B>" /
  -- "accdp<k>x<B>" is the `.adamwAccum` render: a FOURTH region `G` holding the running gradient
  -- sum, and two extra scalars `%aup`/`%akeep` deciding, per micro-batch, whether this invoke
  -- accumulates or applies. Same blob SHAPE as the EMA render, so `nRegions`/`nScalars` carry it.
  --
  -- ⚠⚠ **`k` IS READ OFF THE VARIANT NAME, and that is the point.** The graph has `1/k` BAKED into
  -- `%ob1`/`%ob2` (`optConstsB`), and the driver decides the apply cadence. If those two disagree
  -- the run does not fail — it trains at a silently wrong effective learning rate. Reading `k` from
  -- the same string that names the artifact file makes them agree by construction;
  -- `ResNet50RenderB` pins the round trip with a `#guard` on the producing side.
  -- ⚠⚠ SUBSTRING, NOT PREFIX, and `k` parsed from AFTER the marker — changed 2026-08-06, defect #4
  -- in `tests/TestVariantPredicates.lean`. RSB-A3's composed optimizer is `lambaccdp8x64bce`, where
  -- `lamb` ++ `acc` puts the marker in the MIDDLE; `startsWith "acc"` is false there, so this
  -- driver would have packed THREE regions into a FOUR-region graph. ⭐ It also makes the
  -- `emaOn && accOn` refusal below reachable at all — under the prefix test `accOn "emaacc…"` was
  -- false, so that throw could never fire and the combination would have silently dropped
  -- accumulation. Both counterfactuals are pinned in `TestVariantPredicates`.
  let accOn := VerifiedVariant.accOn variant
  let accK := VerifiedVariant.accK variant
  if accOn && accK < 1 then
    throw <| IO.userError s!"variant '{variant}' contains 'acc' but no accumulation count could \
be read from it — the name must spell acc<k>x<B> or accdp<k>x<B> (optionally after an optimizer \
name, as in lambaccdp8x64bce), and <k> is what the graph's baked 1/k was rendered for"
  -- ⚠ Both want the fourth region. Silently letting one win would put the EMA shadow and the
  -- gradient accumulator in the same slots.
  if emaOn && accOn then
    throw <| IO.userError "variant selects BOTH the EMA shadow and gradient accumulation, and they \
occupy the same fourth region of [θ|m|v|·]. Render one or the other."
  let nRegions := VerifiedVariant.nRegions variant
  let nScalars := VerifiedVariant.nScalars variant
  -- "…drop" = the STOCHASTIC-DEPTH render (`planning/stochastic_depth.md`): the graph takes one
  -- extra `tensor<Bxf32>` per drop site, carrying `bernoulli(keep_i)/keep_i` per example.
  --
  -- ⚠⚠ THE MARKER IS `"drop"` BECAUSE `"sd"` COLLIDES, and the collision is between two OTHER
  -- markers meeting: `rms` ++ `dp` spells **`rmsdp`**, which contains "sd". A `"sd"` substring test
  -- therefore fires on `rmsdp64` and `emarmsdp64` — every RMSProp data-parallel variant, including
  -- the committed and gated `efficientnetin_rmsdp64` — and would have appended 9 drop scales to a graph
  -- that takes none. Caught by running the predicate table (`tests/TestVariantPredicates.lean`)
  -- rather than reading names one at a time; with three markers the collisions are between PAIRS.
  -- This is `planning/ema.md`'s `emarms` defect a second time, one axis further on.
  let sdOn := VerifiedVariant.sdOn variant && !net.dropKeeps.isEmpty
  let nDrop := if sdOn then net.dropKeeps.size else 0
  -- ▶ CLASSIFIER DROPOUT. ⚠⚠ The marker is `"do"` and NOT `"dropout"`, and that is forced by the
  -- line above: `"dropout"` contains `"drop"`, so a dropout-only variant would set `sdOn` and this
  -- driver would pack nine mask slots into a graph that has none. Collision #3 on this naming, and
  -- the first caught before it shipped — `tests/TestVariantPredicates.lean` runs the pairwise table
  -- rather than reasoning about it, and pins the counterfactual (`sdOn "adamdropout" == true`).
  let cdOn := VerifiedVariant.cdOn variant && net.dropoutKeep.isSome
  let doKeep := (net.dropoutKeep.map (·.1)).getD 1.0
  let doWidth := if cdOn then (net.dropoutKeep.map (·.2)).getD 0 else 0
  -- ⚠ The per-example TAIL the DP shim shards is BOTH mask families — nine `tensor<gbs>` scales
  -- followed by one `tensor<gbs × w>` mask — so the count it takes is their sum, not `nDrop`.
  -- Both are per-example along dim 0 and the shim splits by `elems / replicas`, so a rank-2 tail
  -- entry shards by ROWS exactly as a rank-1 one does. See the `mlpTrainStepVDP` call below.
  let nShardTail := nDrop + (if cdOn then 1 else 0)
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  net.printBlurb
  -- Running-stats BN: when `bnChannels` is non-empty the adam train step carries per-layer batch
  -- mean/var out in passthrough slots (so #out=#in), the driver EMAs them into `runningBnStats`,
  -- and eval uses `<slug>_fwd_eval.mlir` (affine BN with the running stats) — class-batch-independent
  -- eval parity, not the degenerate batch-BN-eval. LayerNorm / no-BN nets skip all of this.
  let hasBn := !net.bnChannels.isEmpty
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_{variant}_train_step.mlir"
  -- ⭐ PER-VARIANT forward resolution (`planning/mnv4_verified.md` §3d(b)).
  --
  -- The train step above is variant-resolved and this was NOT: every variant of a slug loaded the
  -- one `<slug>_fwd.mlir`. But a slug's variants do not all live in the same BN world — the SGD
  -- train step comes from the per-example renderer and everything from `*RenderB` is batch BN — so
  -- one forward artifact cannot be right for both. On MobileNetV2 and ResNet-34 it is the
  -- per-example one, i.e. correct for the SGD trainer and a DIFFERENT NET from the Adam graph that
  -- trains every quoted number.
  --
  -- `<slug>_<variant>_fwd.mlir` wins when it exists; `<slug>_fwd.mlir` is the fallback, which is
  -- correct for every net whose forward already matches its batched train step (efficientnet,
  -- convnext, vit, mnv4, resnet50). ⚠ The fallback is only safe because it is CHECKED below —
  -- a silent fallback to the wrong world is the defect, not the fix.
  let fwdVariant := s!"{net.mlirDir}/{net.slug}_{variant}_fwd.mlir"
  let fwdPath := if (← System.FilePath.pathExists fwdVariant) then fwdVariant
                 else s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let fwdSess ← mkSession fwdPath
  let fwdEvalSess ← if hasBn then
      mkSession s!"{net.mlirDir}/{net.slug}_fwd_eval.mlir"
    else pure fwdSess
  let synth := (← IO.getEnv "LEAN_MLIR_BENCH_SYNTH").isSome
  -- ⚠ `mkSynthData` must be sized at the GLOBAL batch, not `bs`. Under data
  -- parallelism one step consumes `bs * replicas` images (the shim shards them),
  -- so a `bs`-sized synthetic buffer is read past its end every step: silent at
  -- bs32×2, a `free(): invalid next size` abort at bs128×2. Found 2026-07-30
  -- while measuring the parameter transfer share (handoff §2d.3). This is why
  -- `replicas` is read here and not with the other knobs below.
  let replicas := ((← IO.getEnv "LEAN_MLIR_REPLICAS").bind (·.toNat?)).getD 1
  -- ▶ `LEAN_MLIR_EVAL_BATCHSTATS=1` — a DIAGNOSTIC, not a feature. Scores through `@<slug>_fwd`
  -- (BN over the EVAL BATCH's own statistics) instead of `@<slug>_fwd_eval` (the accumulated
  -- running buffers). It exists to separate "the weights are bad" from "the running statistics
  -- are bad", which the loss alone cannot do: the two paths read the SAME θ and differ only in
  -- what they normalise by. ⚠ Batch-stat scoring is transductive — it peeks at the eval batch —
  -- so it is NOT a reportable accuracy. It is an upper reference for what these weights can do.
  let batchStatEval := (← IO.getEnv "LEAN_MLIR_EVAL_BATCHSTATS").isSome
  let useRunning := hasBn && !batchStatEval
  if batchStatEval && hasBn then
    IO.println "  ⚠ LEAN_MLIR_EVAL_BATCHSTATS: scoring via @_fwd (EVAL-BATCH stats), not the \
running buffers — diagnostic only, transductive, not a reportable number."
  -- ⛔⛔ THE BN-WORLD INVARIANT, asserted exactly when the forward is about to be USED.
  --
  -- Batch-stat scoring only means anything if `@<slug>_fwd` normalises the way the train step
  -- does. Where it does not, this mode silently scores a DIFFERENT ARCHITECTURE — not merely
  -- different statistics — and reports a plausible number. That is the §3d(b) hazard in its live
  -- form, and until now nothing anywhere checked it: `regen_verified_mlir.sh` paired the forward
  -- only with the SGD train step, which shares its world by construction.
  --
  -- ⚠ Scoped to `!useRunning` ON PURPOSE. A normal run never invokes this forward (eval goes
  -- through `@<slug>_fwd_eval`), so failing there would break working trainers over an artifact
  -- they do not read. The check fires only on the path that actually reads it.
  if !useRunning && hasBn then
    let tsTxt ← IO.FS.readFile s!"{net.mlirDir}/{net.slug}_{variant}_train_step.mlir"
    let fwTxt ← IO.FS.readFile fwdPath
    let batchOf (t : String) : Bool := (t.splitOn "dimensions = [0, 2, 3]").length > 1
    if batchOf tsTxt != batchOf fwTxt then
      throw <| IO.userError s!"BN-WORLD MISMATCH — refusing to score through a different net.\n\
  train step {net.slug}_{variant}_train_step.mlir : \
{if batchOf tsTxt then "BATCH" else "PER-EXAMPLE"} BN\n\
  forward    {fwdPath} : {if batchOf fwTxt then "BATCH" else "PER-EXAMPLE"} BN\n\
  LEAN_MLIR_EVAL_BATCHSTATS would score a DIFFERENT ARCHITECTURE, not just different statistics.\n\
  Render {net.mlirDir}/{net.slug}_{variant}_fwd.mlir from the chain the train step \
differentiates (see r50FwdChainB for the pattern), or drop the env var and score through \
@{net.slug}_fwd_eval."
  -- The eval forward is rendered at ITS OWN batch AND ITS OWN INPUT WIDTH, neither of which need
  -- match training. Read both off the artifact rather than assuming (`fwdRenderedShape`); when they
  -- agree with `(bs, d0)` — every 224 net — nothing below changes.
  -- ⚠⚠ COMPUTED HERE, ABOVE `loadData`, and that ordering is load-bearing: the ImageNet val drain
  -- inside `loadData` has to allocate and read at the EVAL width, so it needs `evalD0` as an input.
  -- It used to hardcode `3*224*224`, which was right only while train and eval resolutions agreed.
  let (evalBs, evalD0) := (← fwdRenderedShape
    (if useRunning then s!"{net.mlirDir}/{net.slug}_fwd_eval.mlir"
     else fwdPath)).getD (bs, d0)
  -- ⚠ ANNOUNCED when it differs, because a train/eval resolution SPLIT is not visible anywhere else
  -- in the log, and a run that silently evaluated at the wrong resolution would still report a
  -- plausible accuracy. RSB-A3 is the case: train 160², eval 224².
  if evalD0 != d0 then
    IO.println s!"  ▸ EVAL RES SPLIT: train d0 {d0}, eval d0 {evalD0} (batch {evalBs}) — read off \
@{net.slug}_fwd{if useRunning then "_eval" else ""}"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    if synth then mkSynthData net.data d0 (bs * replicas)
    else loadData net dataDir evalD0
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  -- LEAN_MLIR_G2_STEPS caps batches per epoch for gate G2. Deliberately NOT
  -- LEAN_MLIR_MAX_STEPS: that name already means "time a step window then exit"
  -- in this driver (the benchmark's `attn` anchor), and it returns before the
  -- param dump. See planning/xla_pjrt_ladder.md §3.
  -- LEAN_MLIR_REPLICAS: data-parallel device count. The graph is rendered at the
  -- PER-REPLICA batch (cfg.batchSize), so one step consumes `bs * replicas`
  -- images and the shim splits them. Eval stays single-device at `bs`, because
  -- the forward graph is rendered at that batch. See planning/xla_pjrt_ladder.md §10.
  -- LEAN_MLIR_SKIP_EVAL: skip the per-epoch eval pass. It used to be REQUIRED whenever the train
  -- batch differed from the forward's baked one (bs256, bs128-DP); `evalBs` below removes that,
  -- so it is now just "don't spend the time".
  let skipEval := (← IO.getEnv "LEAN_MLIR_SKIP_EVAL").isSome
  let gbs := bs * replicas
  let nbFull := nTrain / gbs
  let nb := match (← IO.getEnv "LEAN_MLIR_G2_STEPS").bind (·.toNat?) with
    | some n => min n nbFull
    | none   => nbFull
  -- `evalBs`/`evalD0` are read off the eval artifact ABOVE, before `loadData` — see there.
  let nbt := (nEval + evalBs - 1) / evalBs  -- ceil: the last partial batch is zero-padded, not dropped
  -- The schedule label is part of the run's evidence, not decoration: an exponential-decay run and
  -- a cosine one are different experiments and the log has to say which it was. Spelled so the
  -- string is UNCHANGED at the default (`expDecayRate = 0`), i.e. every existing log line still reads
  -- "(cosine+warmup Nep, baseLR L)".
  let schedName := if expDecayRate > 0.0 then s!"exp x{expDecayRate}/{expDecayEpochs}ep" else "cosine"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} {variant} ({schedName}+warmup {warmupEpochs}ep, baseLR {baseLR}), He init"
  if rmsprop then
    IO.println s!"  ▸ RMSPROP: m = momentum buffer (init 0), v = running MEAN-SQUARE (init 1.0, \
TF convention — this optimizer is not bias-corrected)"
  if sdOn then
    IO.println s!"  ▸ STOCHASTIC DEPTH: {nDrop} drop sites, keeps {net.dropKeeps.map (fun k => (k * 1000.0).round / 1000.0)} — host-drawn per step, 1/keep folded in, NOT on the resident path. Eval is the identity (drop-free forward)."
  -- ⚠ It ANNOUNCES ITSELF, and that is §0.9's finding rather than politeness: a banner that names
  -- only the architecture makes a run with the wrong regulariser read exactly like a right one.
  -- The keep and the mask SHAPE are both printed, because the shape is the whole difference from
  -- the line above — `B × w` is per-element dropout, `B` alone would be stochastic depth.
  if cdOn then
    IO.println s!"  ▸ CLASSIFIER DROPOUT: keep {doKeep}, mask tensor<{gbs}x{doWidth}xf32> \
(PER-ELEMENT, one Bernoulli per example×feature — not per-example like the drop scales), \
host-drawn per step at seed+999983, 1/keep folded in. Eval is the identity (drop-free forward)."
  if emaOn then
    IO.println s!"  ▸ EMA: 4th blob region [θ|m|v|ema], shadow starts AT the weights, decay \
min({emaDecay}, (1+t)/(10+t)) — TF warmup-corrected. EVAL AND CHECKPOINT SCORE THE SHADOW."
  if accOn then
    -- ⚠⚠ IT ANNOUNCES ITSELF, and here that is not politeness either: a run with the wrong `k`
    -- prints an entirely normal loss curve at a silently wrong effective batch and learning rate.
    -- §6's rule — "a setting with no output and no gate is a setting that can be silently wrong".
    IO.println s!"  ▸ GRADIENT ACCUMULATION: k = {accK}, 4th blob region [θ|m|v|G]. Micro-batch \
{gbs} x {accK} = EFFECTIVE BATCH {gbs * accK}. {nb} micro-batches/epoch = {nb / accK} updates/epoch; \
the LR schedule and Adam's bias correction run on UPDATES, the augmentation and the prefetch on \
micro-batches."
    IO.println s!"     ⚠ This is AdamW at that batch, NOT rsb-faithful — LAMB and BCE-with-logits \
are still absent (planning/rsb_a3_r50_verified.md §2.3)."
    -- ⚠⚠ A cycle that straddles the epoch boundary applies with fewer than `k` micro-batches while
    -- the graph still divides by `k`, i.e. a short step at a wrong scale — once per epoch, invisible
    -- in the loss curve. Refuse rather than round.
    if nb % accK != 0 then
      throw <| IO.userError s!"{nb} micro-batches per epoch is not divisible by k = {accK}: the \
last cycle of every epoch would apply {nb % accK} micro-batches' gradient still divided by {accK}. \
Cap the steps to a multiple of {accK} (LEAN_MLIR_G2_STEPS) or render a different k."
  if evalBs != bs then
    IO.println s!"  eval batch {evalBs} (the batch @{net.slug}_fwd{if hasBn then "_eval" else ""} \
was RENDERED at) != train batch {bs} — sound because eval is class-batch-independent"
  if hasBn then IO.println s!"  running-stats BN: {net.bnChannels.size} layers, {nBnStats} stat floats → eval via @{net.slug}_fwd_eval"
  if replicas > 1 then
    IO.println s!"  DATA-PARALLEL: {replicas} replicas x bs {bs} = global batch {gbs}, {nb} steps/epoch"
  (← IO.getStdout).flush
  -- ⚠ The drop scales go LAST, after the BN stats, matching `enetFwdSig`/`inSig`'s placement.
  -- Anywhere else and they capture an existing positional slot — the mnv2 `convBias` failure
  -- (§2m), which is silent until the driver mis-walks the blob.
  -- ⚠⚠ `gbs`, NOT `bs`, AND THAT IS THE OTHER HALF OF §5b'S DEFECT. The mask is per-EXAMPLE, so
  -- the buffer the shim splits has to hold the GLOBAL batch: replica r takes rows
  -- [r*bs, (r+1)*bs) of each `tensor<gbs xf32>` mask, exactly as it does of `x`. Sized at `bs` the
  -- shim would have nothing to split — it would refuse on the outer dim, or (worse, before the
  -- shard flag existed) hand every replica the same `bs` rows. At `replicas = 1` this IS `bs`, so
  -- every existing run and every committed artifact is untouched.
  let dropShapes : Array (Array Nat) := Array.replicate nDrop #[gbs]
    -- ▶ CLASSIFIER DROPOUT's slot, LAST — after the stochastic-depth scales, matching
    -- `enetFwdSig`/`inSig`'s order. ⚠ It is the first mask slot that is not `#[gbs]`: rank 2, and
    -- `gbs` times WIDER. Everything above assumed a per-example mask was one float per example;
    -- this is one per (example, feature). `gbs` and not `bs` for `dropShapes`' reason exactly —
    -- the shim splits the GLOBAL buffer by rows, so a per-device sizing would leave it nothing to
    -- split (§5b's defect, the half that made replication type-check).
    ++ (if cdOn then #[#[gbs, doWidth]] else #[])
  let adamShapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                                -- the FOURTH region: the EMA shadow, or the gradient accumulator.
                                -- ⚠ The G4 gated interface counts destinations off THIS list, so
                                -- omitting it does not mis-walk the blob quietly — the shim refuses
                                -- ("returns 755 outputs, caller supplied 594 destinations").
                                ++ (if emaOn || accOn then net.paramShapes else #[])
                                ++ Array.replicate nScalars #[]
                                ++ (if hasBn then bnStatShapes else #[])
                                ++ dropShapes)
  -- Device-resident parameters (handoff §2d.3). The leading `3×P` tensors of
  -- `adamShapes` are `[θ|m|v]`, and they are exactly the part of the blob this
  -- loop writes ONCE and thereafter only hands straight back: below, the host
  -- touches the tail (`write3` the three scalars, `blit` the BN stats) and reads
  -- the tail (`read` the loss, `extract` the batch stats) — never the prefix,
  -- which is why `pbuf := out` is already a no-copy handover. So that prefix can
  -- live on the device across steps, and at R34 that is 260 MB each way per step
  -- that stops crossing PCIe (55% of a bs32 step, measured).
  --
  -- ⚠ This is a REQUEST, and nothing here selects a transport. The C boundary
  -- honours it only under `$PJRT_FFI_RESIDENT=1` on the XLA build, so IREE and
  -- XLA still run this identical body — the property every §2h cross-backend
  -- gate rests on. The gate is `scripts/residency_gate.sh`: bit-identical
  -- parameters, or it did not land.
  let nResident := (nRegions * net.paramShapes.size).toUSize
  let fwdShapes := net.shapesBA
  let fwdEvalShapes := packShapes (net.paramShapes ++ bnStatShapes)
  -- eval-only: the train step passes its dims directly. ⚠ `packXShape #[evalBs, evalD0]`, NOT
  -- `net.xShape evalBs` — the latter bakes `net.d0`, i.e. the TRAIN width, which is wrong the
  -- moment eval runs at a different resolution (RSB-A3: train 160², eval 224²).
  let xShape := packXShape #[evalBs, evalD0]
  let tsFn  := s!"m.{net.slug}_{variant}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  -- LEAN_MLIR_PERTURB_R: displace the initial parameters along a random unit
  -- vector of exact L2 norm r, before any training. This is the CONDITIONING
  -- probe for gate G2 (planning/xla_pjrt_ladder.md §8, rung 3): if an r that is
  -- f32-epsilon-sized relative to ||theta|| moves the resulting gradient about as
  -- much as the IREE/XLA disagreement does, then that disagreement is what
  -- ill-conditioning predicts, not evidence of a wrong backend.
  let theta0 := F32.concat parts
  -- Value is read in units of 1e-9 (no String.toFloat? in this toolchain), so
  -- LEAN_MLIR_PERTURB_R=15990 means an L2 displacement of 1.599e-5.
  let theta ← match (← IO.getEnv "LEAN_MLIR_PERTURB_R").bind (·.toNat?) with
    | some n => do
        let r := n.toFloat * 1e-9
        IO.println s!"  ▸ PERTURBED init: theta += r*u with ||r*u||_2 = {r}"
        F32.perturbUnit theta0 0 net.nParams.toUSize r 12345
    | none   => pure theta0
  let zeros ← F32.const net.nParams.toUSize 0.0
  -- ▶ THE MEAN-SQUARE SLOT, and it is a CORRECTNESS item rather than a tuning one.
  --
  -- AdamW/momentum start both moment slots at 0 and AdamW then bias-corrects, so its first step is
  -- scale-free. **TensorFlow's RMSProp — the one both these references train with — does neither:
  -- it starts the running mean-square at 1.0 and applies no bias correction.** At `s = 0` the first
  -- update is `gw/√((1−ρ)·gw² + ε)` where the reference computes `gw/√(ρ·1 + (1−ρ)·gw² + ε)`, i.e.
  -- a much larger first step with nothing downstream to absorb it. Both are "RMSProp"; only one is
  -- the optimizer `jax/MainMobilenetV2Imagenet.lean` and `jax/MainEfficientNetImagenet.lean` use,
  -- and `timm` ships a whole `RMSpropTF` class for exactly this distinction.
  --
  -- It lands in the DRIVER and not in the render for the same reason `lr` does: it is the initial
  -- value of a graph INPUT, and the graph is a step function that never sees step 0.
  -- `Proofs.rmsBufNext` is correct either way — this is what it gets fed.
  let msInit ← if rmsprop then F32.const net.nParams.toUSize 1.0 else pure zeros
  -- ⚠ THE EMA SHADOW STARTS AT THE WEIGHTS (`ema_params = params`, jax/Jax/Codegen.lean:2739), not
  -- at zeros. A zero-init shadow is a different filter; and it is the warmup-corrected decay below
  -- that stops even THIS init from poisoning the average early — see the `emaD` note.
  -- ⚠ The FOURTH region, when there is one, and the two features that use it seed it DIFFERENTLY.
  -- The EMA shadow starts AT the weights (starting it at the random init is the defect
  -- `planning/ema.md` records: a shadow evaluated at chance on short runs). The gradient
  -- ACCUMULATOR starts at ZERO — and it would be harmless at any value, because `%akeep = 0` on
  -- the first micro-batch of every cycle discards whatever is there. Zero anyway, so a checkpoint
  -- written mid-cycle resumes from something meaningful rather than from a stale partial sum.
  let mut thetamv := F32.concat (#[theta, zeros, msInit] ++
    (if emaOn then #[theta] else if accOn then #[zeros] else #[]))
  let mvBytes := nRegions * net.nParams * 4
  let pBytes := net.nParams * 4
  -- Running BN stats (EMA of per-layer batch mean/var; mom 1.0 on the first step to seed,
  -- then 0.1). Reset per process — washed out well before the per-epoch eval (mom 0.1).
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  -- The EMA shadow of those buffers (`ema_bn`). Starts where they start, as the reference does
  -- (`ema_bn = bn_state`). ⚠ Like `runningBnStats` it is NOT checkpointed — both are reset per
  -- process and rebuilt within an epoch, which is the pre-existing behaviour this does not change.
  let mut emaBnStats ← F32.const nBnStats.toUSize 0.0
  let mut bnFirst := true
  -- The reusable step buffer: [theta|m|v | lr,bc1,bc2 | bn stats]. Built once here
  -- and thereafter carried forward from each step's output (see the inner loop).
  let mut pbuf : ByteArray := .empty
  -- ⚠ Both are counted in OPTIMIZER steps. Under accumulation an epoch is `nb` micro-batches but
  -- only `nb / k` updates, so a schedule left in micro-batches would run the cosine (and the
  -- warmup) `k` times too fast and finish the run at a learning rate the recipe never reaches.
  let totalSteps := (cfg.epochs * nb / accK).toFloat
  let warmSteps := (warmupEpochs * nb / accK).toFloat
  -- Auto checkpoint/resume: each epoch writes [θ|m|v] + the next-epoch counter;
  -- on startup, resume from the latest checkpoint if present (survives reaps).
  -- Delete `.lake/build/<slug>_<variant>_ckpt*.bin{,.epoch}` to start fresh.
  -- ▶ The PATH — backend scoping and `$LEAN_MLIR_CKPT_TAG` — is `ckptPathFor`, shared with
  -- `scoreCheckpoint` so the tool that reads this file cannot spell its name differently.
  let ckptPath ← net.ckptPathFor variant
  let epPath := ckptPath ++ ".epoch"
  let mut startEpoch := 0
  if (← System.FilePath.pathExists ckptPath) && (← System.FilePath.pathExists epPath) then
    thetamv ← IO.FS.readBinFile ckptPath
    -- ⚠ SIZE GUARD. The checkpoint is the raw `[θ|m|v(|ema)]` blob — no header, no fingerprint, no
    -- region count — so a 3-region file loaded by the 4-region EMA driver (or the reverse) does not
    -- fail: it misaligns EVERY parameter and resumes silent garbage. §4 already records that a
    -- checkpoint outlives the artifact it was trained on; a layout change makes that one turn
    -- worse, and this is the two lines that make it loud.
    if thetamv.size != mvBytes then
      throw <| IO.userError s!"checkpoint {ckptPath} is {thetamv.size} bytes but this run wants \
{mvBytes} ({nRegions} regions x {net.nParams} params x 4). It was written by a different blob \
layout — most likely across the EMA boundary, since the `ema*` variants carry a 4th region. Move \
it and its .epoch marker aside and start fresh."
    startEpoch := ((← IO.FS.readFile epPath).toNat?).getD 0
    IO.println s!"  ▸ resuming from checkpoint at epoch {startEpoch}"
    (← IO.getStdout).flush
  -- Reuse ONE shuffle buffer across epochs (mirrors the reference trainer's
  -- curImg/curLbl). Shuffling the SAME mutable in place keeps it exclusive
  -- (rc 1) so F32.shuffle mutates it rather than allocating a fresh full-dataset
  -- copy each epoch. The old `F32.shuffle trainImg` kept the pristine trainImg
  -- alive (rc≥2), forcing the copy path every epoch and leaking ~one training
  -- set (5.3 GiB) per epoch → OOM after ~30 epochs on a 188 GB box.
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  -- The ImageNet train stream: spawned ONCE, not per epoch. The shim's train iterator is
  -- `.shuffle(seed=42, reshuffle_each_iteration=True).repeat()`, so it re-shuffles across the epoch
  -- boundary by itself and never ends — the per-epoch `F32.shuffle` below is skipped for it.
  -- $SHIM_WORKERS > 1 shards the stream across that many producer processes (default 1, i.e.
  -- byte-identical to before this knob existed). Needed once the step rate outruns one producer's
  -- ~1,530 img/s: a 4-replica ViT step wants ~1,940. See `spawnShimSharded`.
  let shimWorkers := ((← IO.getEnv "SHIM_WORKERS").bind (·.toNat?)).getD 1
  -- $SHIM_SOFT=1 asks the shim for WIRE v2 — `float32[batch*nClasses]` target distributions rather
  -- than `int32[batch]` labels. Today those are one-hots, i.e. the same information in the shape
  -- the graph already consumes, which is exactly what makes the transport gateable on its own:
  -- a one-hot sent as a soft target must train BIT-IDENTICALLY to the hard-label path. What it
  -- unlocks is mixup/cutmix, which need a target the label alphabet cannot express.
  --
  -- No render change is required for any of this: the committed renders are AFFINE in `%onehot`
  -- (measured, `lake build soft-target-tie`), so a mixed target yields the mixed gradient.
  -- ▶▶ **DEFAULT-ON as of 2026-08-03, for any net whose OWN shim declares mixing.** Until now
  -- `SHIM_SOFT` had to be set by hand, so a plain ViT/ConvNeXt ImageNet run streamed
  -- `SHIM_MIX=off` — i.e. trained WITHOUT the mixup/cutmix their references set — and merely
  -- announced that it had. An opt-in flag for a reference feature is the "matrix reads capability,
  -- not state" defect in the data path (§0.9 finding 3): the capability was there and no run used
  -- it. The default now follows the net's config.
  --
  -- ⚠ Derived from the shim's BAKED default, not from a new field: `generateShim` already wrote
  -- the config's `useMixup`/`useCutmix` into the script, and a `VerifiedNet.mixes` flag would be a
  -- second definition of that one fact. `shimMixDefault` is the single reader, shared with
  -- `spawnShim` so the wire and the announcement cannot disagree.
  --
  -- ⚠ `SHIM_SOFT=0` still forces wire v1 OFF — the escape hatch every gate needs, because several
  -- of them (`*-dp-check`, the known-answer ties) want hard labels and a deterministic stream.
  -- ⚠ Nets whose reference does NOT mix are untouched: R34, mnv2 and **EfficientNet** all bake
  -- `off`, so this is inert for them. Turning it on there would move them AWAY from their
  -- references, not toward them.
  let mixDecl ← shimMixDefault net.shimScript
  let declaresMix := mixDecl != "" && mixDecl != "off"
  let softTargets := match ← IO.getEnv "SHIM_SOFT" with
    | some v => v != "0" && v.toLower != "off" && v != "false"
    | none   => declaresMix
  let shimNC := if softTargets then net.nClasses else 0
  -- ⚠ ANNOUNCED, never silent — the whole point of the change is that the previous behaviour was
  -- announced-but-off, and an unannounced on would be worse.
  if declaresMix then
    IO.println s!"  ▸ MIXUP/CUTMIX: this net's recipe declares SHIM_MIX={mixDecl}; {if softTargets then "ON (wire v2, soft float32 targets — the reference recipe)"
  else "OFF (SHIM_SOFT explicitly disabled)"}. ⚠ λ is drawn from numpy's Generator, not jax.random — agreement with the reference is DISTRIBUTIONAL, never per-step."
  -- ⚠⚠ `!synth`, ADDED 2026-08-05, and its absence made `LEAN_MLIR_BENCH_SYNTH` INERT on the one
  -- dataset where the data path is the dominant term. The stream was spawned on `net.data ==
  -- .imagenet` alone and the per-step branch below prefers it whenever it is non-empty, so a
  -- "synthetic" ImageNet run still did the full 154 MB blocking pipe read every step. On every
  -- other dataset synth replaces a preloaded host array; ImageNet never had one, so the flag
  -- replaced nothing and said so nowhere. That is handoff §4's own lesson in a new place — *the
  -- synthetic path exists to remove a variable from a measurement, which makes it exactly the code
  -- least likely to be looked at when the measurement comes out clean.*
  --
  -- ▶ This is what splits `t_read` from `t_rest`: the same binary at the same step count, real vs
  -- synth, differs by exactly the shim read. That difference is the ceiling on what a prefetch can
  -- hide (planning/next_session_pipeline_then_r50.md §2).
  --
  -- ⚠ It changes what `scripts/residency_gate.sh` feeds an ImageNet net — from a seeded real
  -- stream to one constant batch. Both are deterministic, which is all that gate's bit-identity
  -- verdict needs, but a constant batch is less numerically varied, so re-confirm the FAULT
  -- control fires before trusting a green from it.
  let imgStreams : Array IO.FS.Handle ←
    if net.data == .imagenet && !synth then
      -- ⚠⚠ `net.d0`, NOT `3 * 224 * 224`. This is the width the TRAIN shim is *told* to emit, and
      -- it is the SECOND of two hardcoded 224s that had to fall for the 160 net — the other was
      -- `loadData`'s `trainPix` (which sizes the READ). Both had to agree with the render, and a
      -- literal here agreed with only the 224 ones: measured 2026-08-06 as
      -- "shim sends batch=64 flat=76800, the render wants batch=64 flat=150528".
      -- ▶ INERT for every incumbent — `LeanMlir/VerifiedNets.lean`'s closing `#guard` block proves
      -- `net.d0 == 3*224*224` for all six 224 ImageNet nets, so this substitutes equal for equal
      -- there and changes only `resnet50in160`.
      spawnShimSharded net.shimScript "train" gbs net.d0
        (((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1) shimWorkers shimNC
    else pure #[]
  if synth && net.data == .imagenet then
    -- ⚠ ANNOUNCED, because the previous behaviour was silent and that is the whole defect: a
    -- number measured this way is NOT a step time, and nothing else in the log would say so.
    IO.println s!"  [SYNTH] imagenet shim NOT spawned — one constant batch, zero pipe reads. \
This measures t_rest (compute + params + host blob patching), NOT a full step."
    -- Wire v2 sizes the target buffer at `gbs × nClasses`, not `gbs`, and `mkSynthData` cannot
    -- know that — it runs before the shim's declared mixing is read. Without this a mixing net
    -- (ViT, ConvNeXt) would read `gbs × nClasses` floats out of a `gbs`-float buffer the moment
    -- synth started supplying the labels, which is the bs128×2 overread of §2d.3 wearing new
    -- clothes. Uniform `1/nc` rather than zeros: a valid probability vector, and step timing is
    -- value-independent either way.
    if shimNC > 0 then
      curLbl ← F32.const (gbs * shimNC).toUSize (1.0 / shimNC.toFloat)
  -- LEAN_MLIR_MAX_STEPS: run a short steady-state ms/step probe then exit. This is
  -- the benchmark's `attn` anchor — ViT is matmul/attention-bound, so its per-step
  -- cost scales very differently from conv across GPUs and can't borrow the conv
  -- factor. A full ViT epoch is too slow to probe, so we time a step window.
  let probeSteps := (← IO.getEnv "LEAN_MLIR_MAX_STEPS").bind (·.toNat?)
  let probeWarm := 8
  let mut probePrev := 0
  let mut probeTimes : Array Nat := #[]
  -- LEAN_MLIR_MAX_EPOCHS: same opt-in cap as `VerifiedNet.train` (absent → full run).
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  -- Build the reusable step buffer once, AFTER any checkpoint resume has settled
  -- `thetamv`. The scalar slots are filled per step; the BN region per step too.
  let scalarSlots ← F32.const nScalars.toUSize 0.0
  -- The drop-scale slots are reserved here and refilled per step, exactly like the scalar and BN
  -- regions — a fresh `F32.concat` per step would cost two whole-blob host memcpys (the mistake
  -- `planning/xla_pjrt_ladder.md` §8 measured at 272 MB/step on R34).
  -- ⚠ Sized for BOTH families. The `1.0` fill is load-bearing and not a placeholder: a mask slot
  -- that is never refilled must be the exact identity, which `1.0` is and `0.0` emphatically is not
  -- (it would zero the classifier's input and train nothing).
  let dropSlots ← F32.const (nDrop * gbs + (if cdOn then gbs * doWidth else 0)).toUSize 1.0
  pbuf := if hasBn
          then F32.concat #[thetamv, scalarSlots, runningBnStats, dropSlots]
          else F32.concat #[thetamv, scalarSlots, dropSlots]
  -- ▶▶ DEPTH-1 PREFETCH of the shim read — planning/next_session_pipeline_then_r50.md §2.
  --
  -- The step was two blocking calls back to back: `readShimBatchRR` (154 MB off a pipe) and then
  -- the invoke, with NOTHING draining the pipe during compute. A batch is 154 MB and a pipe's
  -- buffer is 64 KB (this box caps `pipe-max-size` at 1 MB, still 0.6% of a batch), so the
  -- producer filled its buffer, blocked in `write()`, and slept through the entire compute. It
  -- measured 258% CPU on a 32-core box: not slow, throttled. ⚠ Which is why "one producer does
  -- ~1,530 img/s and R34 needs ~380" was never the relevant comparison — capacity is irrelevant
  -- when the consumer pulls one batch and walks away.
  --
  -- ⭐ MEASURED, `LEAN_MLIR_BENCH_SYNTH` real-vs-synth at 4×bs64 resident fp32: the step was
  -- **377 ms = 158 read + 219 rest**, so `max(158, 219)` = **219** and the read hides COMPLETELY
  -- behind compute. ✅ Delivered: **377 → 224 ms/step, 1.68×** (30 epochs 15.7 h → 9.3 h), 5 ms
  -- off that ceiling. Bit-identity gated by `tests/prefetch_tie.sh`.
  --
  -- ⭐⭐ **DEPTH n, ONE READ IN FLIGHT PER HANDLE (2026-08-11).** The correctness condition was never
  -- "one read outstanding" — it is **one read outstanding PER HANDLE**, because the hazard is two
  -- concurrent reads interleaving on ONE pipe (a pipe is a stream, not a message queue). Depth 1
  -- bought that by having one outstanding read globally, which is sufficient and, at
  -- `SHIM_WORKERS=n`, far stronger than necessary: it drains ONE producer while the other n−1 sit
  -- blocked in `write()` with 64 KB buffered — 0.08% of a batch — sleeping through the compute the
  -- prefetch exists to hide.
  --
  -- ⭐ MEASURED 2026-08-11, ViT/ImageNet 4×bs128, `SHIM_WORKERS=8`: the box ran **70% IDLE** (22 of
  -- 32 cores) at 783 ms/step against a 249 ms synthetic floor, with the eight producers drawing
  -- ~10 cores between them. Not slow, not contended — **throttled**, exactly the signature this
  -- comment recorded for R34 before depth 1 existed ("258% CPU on a 32-core box"). A zero-cost
  -- producer through the SAME pipes at the SAME depth ran 248 ms, so the plumbing and the 308 MB
  -- of transport were never the problem: 5 ms of the step. Capacity was not the problem either —
  -- making each producer 5.3× faster (`SHIM_DETERMINISM=0`) moved the step 0%.
  --
  -- ▶ So the generalisation this comment used to defer is the fix, and it is the ONLY lever the
  -- evidence points at. Step s reads handle `s % n`; the next step on that handle is `s + n`, so
  -- the refill is issued into the slot the wait just freed. Per handle the read SEQUENCE is
  -- unchanged (handle h still serves steps h, h+n, h+2n, … in that order, same bytes for the same
  -- step) — only *when* each read is issued moves earlier, which is precisely what
  -- `tests/prefetch_tie.sh` gates. The resident path's `res_gen` (`ffi/pjrt_ffi.c`) still sees
  -- strict step order because the INVOKES are still strictly ordered; only the reads overlap.
  --
  -- ⚠ It stays lock-free by CONSTRUCTION, not by guarding: each slot's task exclusively owns one
  -- handle, so "two reads on one pipe" is unrepresentable rather than checked for.
  -- ⚠ `LEAN_MLIR_PREFETCH_DEPTH=1` restores the old global-depth-1 behaviour exactly, as the A/B
  -- control. Absent ⇒ depth n = `SHIM_WORKERS`, i.e. depth 1 for the single-producer default, so
  -- every non-sharded net is byte- AND schedule-identical to before this change.
  --
  -- ⚠ Shim path only. Imagenette/CIFAR augment host-side off `augSeed` inside the loop and have
  -- no pipe to drain.
  -- ⚠ This introduces the FIRST concurrency primitive in the repo — `IO.asTask` appeared zero
  -- times before it. The reader thread touches nothing the step touches: it owns the handles and
  -- returns two fresh `ByteArray`s.
  --
  -- DEFAULT ON, with `LEAN_MLIR_PREFETCH=0` as the escape hatch — the same shape as `SHIM_SOFT=0`,
  -- and for the same reason: it is the control the gate needs. ⚠ The gate is that the read ORDER
  -- is unchanged (same handle, same sequence, same bytes; only *when* moves), so N steps with and
  -- without must give a BIT-IDENTICAL loss sequence. `tests/prefetch_tie.sh`.
  let prefetch := match ← IO.getEnv "LEAN_MLIR_PREFETCH" with
    | some v => v != "0" && v.toLower != "off" && v != "false"
    | none   => true
  -- The prefetch DEPTH: one read in flight per producer handle. Defaults to `SHIM_WORKERS`, which
  -- is 1 for every non-sharded net — so the default is byte- and schedule-identical to depth 1
  -- there, and only the sharded ImageNet jobs see a change. Clamped to [1, n] because a depth above
  -- n would need two reads on one handle, which is the one thing that is never allowed.
  let pfDepth := match (← IO.getEnv "LEAN_MLIR_PREFETCH_DEPTH").bind (·.toNat?) with
    | some d => max 1 (min d imgStreams.size)
    | none   => max 1 imgStreams.size
  if !imgStreams.isEmpty then
    -- ⚠ ANNOUNCED. §0.9's finding, and 2026-08-05's: a throughput setting that prints nothing when
    -- OFF is how `PJRT_FFI_RESIDENT` let a 16 h benchmark and a 26 h production config diverge for
    -- a week. Both states say so. ⚠ The DEPTH is announced too, and for the same reason: depth 1
    -- with 8 workers looks identical on screen to depth 8 and runs 3× slower.
    IO.println (if prefetch
      then s!"  ▸ SHIM PREFETCH: ON (depth {pfDepth} over {imgStreams.size} producer handle(s) — \
each step's read is issued before the previous step's invoke, one in flight PER HANDLE, so every \
producer drains during compute instead of blocking in write()). Measured 377 → 224 ms/step on \
R34/ImageNet 4×bs64 for depth 1."
      else "  ▸ SHIM PREFETCH: OFF (LEAN_MLIR_PREFETCH=0) — the read blocks the step. This is the \
gate's control, not a configuration.")
  -- One in-flight read per producer handle, indexed BY HANDLE (`step % nStreams`), so the slot a
  -- wait frees is exactly the slot its refill goes into. `none` = that handle has no read pending:
  -- true for every slot on the first step, and for the tail slots at the end of the run.
  let mut inflight : Array (Option (Task (Except IO.Error (ByteArray × ByteArray)))) :=
    Array.replicate (max 1 imgStreams.size) none
  for ep in [startEpoch:nEpochs] do
    let mut epochLossSum := 0.0
    let mut lastLr := 0.0
    -- Per-epoch Fisher-Yates shuffle (the reference does this; the data is
    -- class-sorted, so without it every batch is a single class — degenerate).
    -- Skipped when streaming: there is no resident array to shuffle, and tf.data already
    -- re-shuffles each iteration inside the shim.
    if !synth && imgStreams.isEmpty then
      let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPix.toUSize
                           4 -- classification: one f32 class id per record
                           (ep + 42).toUSize
      curImg := sImg; curLbl := sLbl
    for bi in [0:nb] do
      -- ▶▶ MICRO-STEP vs OPTIMIZER STEP. Without accumulation these are the same number and every
      -- expression below reads exactly as it did. With it, `mstep` counts micro-batches (it seeds
      -- the augmentation and the drop masks, and it is what the depth-1 prefetch indexes — §4.1's
      -- "the prefetch index must follow the MICRO-step") while `gstep` counts UPDATES and is what
      -- the LR schedule and Adam's bias correction read.
      let mstep := ep * nb + bi + 1
      -- `%akeep` is 0 on the first micro-batch of a cycle and 1 after: the accumulator RESETS by
      -- dropping the previous total (`Gt = akeep·G + g`), so there is no separate zeroing step that
      -- could be missed. `%aup` is 1 only on the last, where the optimizer actually moves.
      let applyNow := !accOn || mstep % accK == 0
      let keepAcc  := if accOn && (mstep - 1) % accK != 0 then 1.0 else 0.0
      let gstep := (if accOn then (mstep + accK - 1) / accK else mstep).toFloat
      -- Post-warmup decay: exponential when `expDecayRate > 0` (the EfficientNet/MobileNetV2
      -- schedule), cosine otherwise. The exponential branch reproduces the formula
      -- `jax/Jax/Codegen.lean` EMITS for these two references, line for line:
      --
      --     _ep = _global_step / steps_per_epoch
      --     lr  = LR * (rate ** ((_ep - warmup) / decayEpochs))
      --
      -- ⚠ `_global_step` there is 0-BASED at the point the LR is computed — its own warmup branch
      -- reads `(_global_step + 1) / warmup_steps`, which is this driver's `gstep / warmSteps` — so
      -- the epoch is `(gstep − 1) / nb`, NOT `gstep / nb`. One step of offset is invisible in a
      -- 5004-step epoch, which is exactly why it has to come off the reference rather than a guess.
      --
      -- Spelled `exp ∘ log` rather than with `^` because that is what the next two lines already do.
      let lrt := if gstep ≤ warmSteps then baseLR * gstep / warmSteps
                 else if expDecayRate > 0.0 then
                   baseLR * Float.exp (((gstep - 1.0) / nb.toFloat - warmupEpochs.toFloat)
                                       / expDecayEpochs * Float.log expDecayRate)
                 else baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (gstep - warmSteps) / (totalSteps - warmSteps)))
      let bc1 := 1.0 - Float.exp (gstep * Float.log β1)
      let bc2 := 1.0 - Float.exp (gstep * Float.log β2)
      -- Patch the reusable step buffer in place instead of rebuilding it. `pbuf`
      -- is [theta|m|v | lr,bc1,bc2 | bn stats] and the train step returns that
      -- exact layout, so the previous output IS the next input once the 3
      -- scalars and the BN region are refreshed. Rebuilding it with F32.concat
      -- (and slicing [theta|m|v] back out afterwards) cost two 272 MB host
      -- memcpys per step at R34 scale — see planning/xla_pjrt_ladder.md §8.
      -- ⚠ `lr = 0` ON AN ACCUMULATE MICRO-BATCH IS WHAT FREEZES θ, and it freezes it COMPLETELY:
      -- AdamW's decay is DECOUPLED (`θ' = θ − lr·m̂/(√v̂+ε) − lr·wd·θ`), so both terms vanish. A
      -- COUPLED-L2 optimizer would keep decaying k times per update and this would be wrong.
      pbuf ← F32.write3 pbuf (nRegions * net.nParams).toUSize
               (if applyNow then lrt else 0.0) bc1 bc2
      if accOn then
        let accPair ← F32.const 3 0.0
        let accPair ← F32.write3 accPair 0 (if applyNow then 1.0 else 0.0) keepAcc 0.0
        pbuf ← F32.blit pbuf (nRegions * net.nParams + 3).toUSize accPair 0 2
      -- ⚠ THE WARMUP-CORRECTED DECAY, required at our scale rather than optional.
      -- `d = min(decay, (1+t)/(10+t))` is TF's `ExponentialMovingAverage(decay, num_updates)`, the
      -- form the reference emits (`jax/Jax/Codegen.lean:2460`). Without it the shadow decays its own
      -- init away only as `decay^t`: the reference MEASURED a shadow still holding 12.8% init at
      -- 3.1 tau, scoring 0.00% top-1 while the live weights scored 70.48%. An 80-epoch Imagenette
      -- run is 23,600 steps = 2.4 tau at decay 0.9999 — squarely inside that regime.
      -- `t` is the reference's 0-BASED `_global_step`, i.e. `gstep - 1` here.
      let emaD := min emaDecay ((gstep - 1.0 + 1.0) / (gstep - 1.0 + 10.0))
      if emaOn then
        let emaPair ← F32.const 3 0.0
        let emaPair ← F32.write3 emaPair 0 emaD (1.0 - emaD) 0.0
        pbuf ← F32.blit pbuf (nRegions * net.nParams + 3).toUSize emaPair 0 2
      if hasBn then
        pbuf ← F32.blit pbuf (nRegions * net.nParams + nScalars).toUSize runningBnStats 0 nBnStats.toUSize
      -- ▶ STOCHASTIC DEPTH: draw this step's per-example keep scales and blit them into the
      -- trailing slots. ⚠ SEEDED FROM THE GLOBAL STEP, like `augSeed` below — an unseeded or
      -- wall-clock-seeded draw makes the run unreproducible and breaks every gate that replays a
      -- step. ⚠ These are ORDINARY inputs, deliberately NOT on the resident path (`nResident`
      -- covers only the leading `nRegions * P` tensors): they change every step, so retaining them
      -- would be wrong rather than merely wasteful.
      if sdOn then
        -- ⚠ drawn at the GLOBAL batch: `dropScales` is site-major (`bs` consecutive values per
        -- site), so a `gbs`-wide draw gives each mask input a contiguous global row block that the
        -- shim splits per replica. Drawing at `bs` and letting the shim replicate would give
        -- example i on replica 0 and example bs+i on replica 1 the SAME Bernoulli draw — masks
        -- correlated across the global batch, which is a weaker regulariser and is not what the
        -- reference computes.
        let sc ← F32.dropScales net.dropKeeps gbs (ep * nb + bi + 1).toUSize
        pbuf ← F32.blit pbuf (nRegions * net.nParams + nScalars + nBnStats).toUSize sc 0
                 (nDrop * gbs).toUSize
      -- ▶ CLASSIFIER DROPOUT: this step's per-ELEMENT mask, into the slot after the SD scales.
      -- ⚠⚠ A SEPARATE SEED STREAM, and that is the reference's own structure rather than caution:
      -- it draws the classifier mask at `fold_in(drop_key, 999983)` while stochastic depth uses
      -- `fold_in(drop_key, block_index)` — a distinct sub-key precisely so the two regularisers do
      -- not share draws. Handing both the same seed here would correlate the classifier mask with
      -- block 0's drop decision every step: it trains, it descends, and no gate in this feature's
      -- set compares the two streams. `999983` is carried verbatim so the divergence is the
      -- reference's constant and not an arbitrary one of ours.
      if cdOn then
        let dm ← F32.dropoutMask doKeep (gbs * doWidth) ((ep * nb + bi + 1) + 999983).toUSize
        pbuf ← F32.blit pbuf
                 (nRegions * net.nParams + nScalars + nBnStats + nDrop * gbs).toUSize dm 0
                 (gbs * doWidth).toUSize
      let augSeed := (ep * nb + bi + 1).toUSize
      -- ImageNet takes the whole batch off the wire, already augmented and normalized by the shim,
      -- so it bypasses BOTH the slice and the augmentation below. That is deliberate: the transform
      -- has exactly one definition (the generated shim, shared with the JAX reference), and a second
      -- copy here is the double-writer failure this repo keeps paying for.
      -- ⚠ Statement position, not `let (xb, yb) ← if …`, because the prefetch branch REASSIGNS
      -- `inflight`, and do-notation only threads a mutable variable through statements — inside a
      -- nested `do` used as an expression the assignment does not elaborate.
      let mut xb := ByteArray.empty
      let mut yb := ByteArray.empty
      if !imgStreams.isEmpty then
        -- Round-robin across the sharded producers; with SHIM_WORKERS=1 (the default) this is
        -- `imgStreams[0]` every step, i.e. exactly the single-producer path.
        -- ⚠⚠ THE READ WIDTH, and the THIRD of three independent hardcoded 224s the 160 net had to
        -- flush out (the others: the shim SPAWN width above, and `loadData`'s `trainPix`). All
        -- three describe the same buffer and every one of them had to agree with the render, so
        -- fixing them one at a time surfaced the same refusal three times over.
        -- ▶ INERT for the six 224 nets by `VerifiedNets.lean`'s closing `#guard` block.
        let flat := net.d0
        if prefetch then
          -- Step `bi`'s batch has been in flight since step `bi-1` issued it. `none` only on the
          -- very first step of the run — one step of no overlap in 150,120, not worth a case.
          -- ⚠ The index runs `ep * nb + bi` unbroken across the epoch boundary, which is what the
          -- round-robin needs: the train iterator `.repeat()`s inside the shim and never ends, so
          -- there is no per-epoch restart to resynchronise against.
          --
          -- ⭐ `Task.Priority.default` (the pool), NOT `.dedicated`, and it is worth **12 ms/step**
          -- — measured 2026-08-05, R34/ImageNet 4×bs64: **236 dedicated vs 224 pooled**. The usual
          -- advice for a blocking read is `.dedicated`, so that a long `read()` does not occupy a
          -- pool worker and starve other tasks. That reasoning does not apply here and its cost
          -- does: depth 1 means there is **exactly one outstanding task by construction**, so
          -- there is nothing to starve, while `.dedicated` spawns a fresh OS thread **every step**
          -- — 150,120 of them over a 30-epoch run. Pooled lands 5 ms above the 219 ms synth floor,
          -- and that residual is the real path allocating a fresh 154 MB `ByteArray` per step
          -- where synth reuses one buffer.
          -- ⚠ The step index runs unbroken across the epoch boundary — `ep * nb + bi + 1` at the
          -- end of epoch e is exactly `ep' * nb + 0` for e+1 — which is what keeps the round-robin
          -- continuous. The train iterator `.repeat()`s inside the shim and never ends, so there
          -- is no per-epoch restart to resynchronise against.
          let s := ep * nb + bi
          let nStr := inflight.size
          let slot := s % nStr
          let t ← match inflight[slot]! with
            | some t => pure t
            | none   => IO.asTask (readShimBatchRR imgStreams s gbs flat shimNC)
                          Task.Priority.default
          let r ← IO.wait t
          -- The wait FREES this handle's slot. Marking it before the refill loop is what makes
          -- "the slot I just consumed" the slot step `s + n` goes into, without special-casing it.
          inflight := inflight.set! slot none
          -- ⚠⚠ HERE, and the position is load-bearing at both ends. BEFORE the invoke below is the
          -- entire point — the readers drain the pipes during compute instead of sleeping through
          -- them. AFTER the wait above is the correctness condition: a handle's next read is issued
          -- only once its previous read has been consumed, so there is never more than one read on
          -- one pipe and per-handle issue order is preserved. Moving this above the wait would put
          -- two reads on one pipe and interleave them.
          -- ⚠ Not past the LAST step of the LAST epoch: such a read would never be consumed, and it
          -- would leave a pool worker blocked in `read()` on a live producer while `main` returns.
          -- ⚠ At depth n the loop below issues exactly ONE read on a steady step (into the slot the
          -- wait just freed, for step `s + n`); on the FIRST step every slot is free, so it issues
          -- n and primes all n producers at once. That is the whole priming story — no separate
          -- pre-loop pass, and the resume path (`startEpoch > 0`) primes identically on its first
          -- step rather than needing to know it resumed.
          -- ⚠ The `LEAN_MLIR_MAX_STEPS` probe still `return`s mid-loop with reads outstanding; that
          -- path is a measurement, not a training run, and it exits through the same reap.
          for j in [1:pfDepth+1] do
            let sj := s + j
            if sj < nEpochs * nb then
              let sl := sj % nStr
              if (inflight[sl]!).isNone then
                inflight := inflight.set! sl (some (← IO.asTask
                  (readShimBatchRR imgStreams sj gbs flat shimNC) Task.Priority.default))
          -- ⚠ Unwrapped AFTER the next reads are issued, so a mid-epoch read error still leaves the
          -- pipeline in a consistent state — it throws here with at most n orphaned tasks, which
          -- the process exit reaps. Unwrapping first would throw with nothing in flight and make
          -- the failure depend on where in the step it happened.
          let (i, l) ← IO.ofExcept r
          xb := i; yb := l
        else
          let (i, l) ← readShimBatchRR imgStreams (ep * nb + bi) gbs flat shimNC
          xb := i; yb := l
      else
        let xbRaw := if synth then curImg else F32.sliceImages curImg (bi * gbs) gbs trainPix
        -- Data-pipeline augmentation (the same FFI the unverified trainer uses;
        -- lives in the data pipeline, not the network): Imagenette = random crop
        -- 256→224 (when the source is 256²) + random hflip; CIFAR = hflip only;
        -- MNIST = none.
        let x ← match net.data with
          | .imagenette =>
              let c ← if crop then F32.randomCrop xbRaw gbs.toUSize 3 256 256 224 224 augSeed
                      else pure xbRaw
              F32.randomHFlip c gbs.toUSize 3 224 224 (augSeed + 7777)
          | .cifar => F32.randomHFlip xbRaw gbs.toUSize 3 32 32 augSeed
          | _ => pure xbRaw
        xb := x; yb := if synth then curLbl else F32.sliceLabels curLbl (bi * gbs) gbs
      let out ← if replicas > 1
        -- ⚠ `nDrop` is the SHARDED TAIL. The drop masks are per-EXAMPLE, so under data parallelism
        -- replica r must get mask rows [r*bs, (r+1)*bs) — the same split `x` gets — not a copy of
        -- replica 0's. They ride in the parameter blob (`dropShapes` above), which is exactly why
        -- they were being replicated: the DP shim's rule was "x and the labels shard, everything
        -- between them replicates". `planning/stochastic_depth.md` §5b predicted this; it was true
        -- of the shim before any DP drop render existed to expose it. At `nDrop = 0` the argument
        -- is inert and every non-SD DP run is byte-identical to before.
        then LowererSession.mlpTrainStepVDP tsSess tsFn xb pbuf adamShapes yb
               gbs.toUSize d0.toUSize nc.toUSize replicas.toUSize nResident nShardTail.toUSize
        else LowererSession.mlpTrainStepV tsSess tsFn xb pbuf adamShapes yb
               bs.toUSize d0.toUSize nc.toUSize nResident
      -- the train step emits the smoothed-CE loss in the slot after [θ'|m'|v']
      let stepLoss := F32.read out (nRegions * net.nParams).toUSize
      epochLossSum := epochLossSum + stepLoss
      lastLr := lrt
      if bi < 3 || bi % 100 == 0 then
        IO.println s!"  step {bi}/{nb}: loss={stepLoss}"
        (← IO.getStdout).flush
      -- EMA the batch BN stats (in the passthrough slots after [θ'|m'|v'|loss|bc1|bc2]).
      -- This slice is small (nBnStats floats), unlike the [θ|m|v] prefix.
      if hasBn then
        let batchBn := out.extract ((nRegions * net.nParams + nScalars) * 4)
                                   ((nRegions * net.nParams + nScalars + nBnStats) * 4)
        -- ⚠ 0.01, NOT 0.1 — corrected 2026-08-04. `F32.ema` computes
        -- `(1−m)·running + m·batch`, so this `m` is the weight on the NEW batch, and the
        -- reference's `momentum=0.99` (`_bn` in `jax/Jax/Codegen.lean`, which updates
        -- `momentum*rm + (1−momentum)*bm`) is `m = 0.01` here. At 0.1 the running stats
        -- averaged ~10 batches against the reference's ~100 — 10× noisier. It is EVAL-ONLY,
        -- so it depressed every reported top-1 without touching a single gradient, and it
        -- bit hardest early, when the activation statistics are still moving fast.
        --
        -- ⚠⚠ **AND COMPENSATED FOR GRADIENT ACCUMULATION, 2026-08-14** — the second half of that
        -- same 2026-08-04 fix, which was not made at the time (`a3_paper_fidelity.md` §2.3).
        -- This EMA fires once per MICRO-batch, so at `k` micro-batches per optimizer step the
        -- stats decay by `0.99^k` per step where the reference's decay by 0.99. At the A3 run's
        -- k = 8 that is 0.923 against 0.99 — our running estimates were ~8x fresher, and
        -- correspondingly noisier, PER OPTIMIZER STEP.
        --
        -- The reference compensates explicitly and its generated script says so: *"BN momentum
        -- compensated for gradient accumulation (K=4): per-micro momentum = 0.99**(1/K) -> K
        -- updates compose to ~one 0.99/step update"*. `m` here is the weight on the NEW batch,
        -- i.e. `1 - momentum`, so the compensated form is `1 - 0.99^(1/k)`: at k = 8 that is
        -- 0.001256, and at k = 1 it is EXACTLY 0.01 — so every non-accumulating run is
        -- bit-identical across this change, which is why the guard is `accOn` and not a version.
        --
        -- ⚠ EVAL-ONLY, on no gradient path. That is what makes it safe to change between runs and
        -- ALSO what let it hide for eight days: nothing about the loss curve moves. ▶ Do NOT apply
        -- it mid-run — the reported eval shifts, so a curve spanning the change develops a
        -- discontinuity that belongs to the metric rather than to the model.
        -- ▶ Direction: this delta plausibly made our reported top-1 UNDERSTATED, which matters
        -- because the A3 result (77.43%) is quoted as beating its JAX reference.
        let bnMom := if accOn then 1.0 - Float.pow 0.99 (1.0 / accK.toFloat) else 0.01
        runningBnStats ← F32.ema runningBnStats batchBn (if bnFirst then 1.0 else bnMom)
        -- ▶ `ema_bn` — the BN running buffers get their OWN shadow, and on a batch-BN net this is
        -- not optional decoration. The reference's own words: eval pairs EMA weights with
        -- EMA-LAGGED stats, "avoiding the weights/stats mismatch that blows up early eval". EMA
        -- weights are a average of many steps' parameters; the LIVE running stats describe only the
        -- most recent steps' activations, and the two do not describe the same network.
        -- ⚠ Same `emaD` as the parameter shadow — one definition of the decay per step. `F32.ema`
        -- takes the NEW-value weight, so it is `1 − d`.
        if emaOn then
          emaBnStats ← F32.ema emaBnStats runningBnStats (1.0 - emaD)
        bnFirst := false
      pbuf := out   -- no copy: the output buffer becomes the next step's input
      -- ms/step probe: start the clock past warmup, report + exit at the cap.
      match probeSteps with
      | some ps =>
        if bi == probeWarm then probePrev := (← IO.monoMsNow)
        else if bi > probeWarm && bi ≤ ps then
          let t ← IO.monoMsNow
          probeTimes := probeTimes.push (t - probePrev); probePrev := t
          if bi == ps then
            -- robust: median per-step time (drops the cold-cache / GC-blip outliers)
            let sorted := probeTimes.qsort Nat.blt
            IO.println s!"  PROBE: {sorted[sorted.size / 2]!} ms/step (median of {sorted.size} steps {probeWarm+1}..{ps}, {net.name})"
            (← IO.getStdout).flush
            return ()
      | none => pure ()
    IO.println s!"Epoch {ep + 1}/{cfg.epochs}: loss={epochLossSum / nb.toFloat} lr={lastLr}"
    -- One 272 MB copy per EPOCH (for eval + checkpoint), not per step. Under
    -- device residency (§2d.3) this is also the one d2h of `[θ|m|v]` that still
    -- happens at all — `readParams` is `pbuf.extract 0 mvBytes` whenever the
    -- parameters are host-resident, and the read-back otherwise, so the
    -- frequency is unchanged either way and this line reads the same.
    thetamv ← LowererSession.readParams tsSess pbuf mvBytes.toUSize
    -- ▶ EVAL AND THE CHECKPOINT SCORE THE SHADOW, not the live weights — which is what the
    -- reference does (`evalArgs`/`params_to_file` read `ema_params`) and the whole point of the
    -- feature: ConvNeXt's 75.93% IS the shadow's number. The shadow is region 4, so it starts at
    -- `3 * pBytes`.
    -- ⚠ Nothing in the `[θ|m|v]` residency gate can see this slice — eval-only state is
    -- structurally invisible to it, exactly as hold-mode is (§2d.3). Its gate is the accuracy
    -- trajectory: the shadow must TRACK THEN EXCEED the live weights, never start near chance.
    let thetaCur := thetamv.extract (if emaOn then 3 * pBytes else 0)
                                    (if emaOn then 4 * pBytes else pBytes)
    -- BN nets eval through `@<slug>_fwd_eval` with the running stats appended; others use `@<slug>_fwd`.
    let evalSess := if useRunning then fwdEvalSess else fwdSess
    let evalFn := if useRunning then s!"m.{net.slug}_fwd_eval" else fwdFn
    -- ⚠ EMA weights MUST be scored against the EMA-lagged stats, never the live ones — that
    -- pairing is the one the reference calls out as blowing up early eval.
    -- $LEAN_MLIR_EMA_BN=0 is a CONTROL, not a feature: it pairs the EMA weights with the LIVE
    -- running statistics, which is the configuration the reference says "blows up early eval". A
    -- claim like that should be measurable rather than asserted — the same reason every tie here
    -- ships with a control that makes it go red. Leave it unset.
    let emaLiveBn := (← IO.getEnv "LEAN_MLIR_EMA_BN") == some "0"
    let evalParams := if useRunning
                      then F32.concat #[thetaCur,
                             if emaOn && !emaLiveBn then emaBnStats else runningBnStats]
                      else thetaCur
    let evalShapes := if useRunning then fwdEvalShapes else fwdShapes
    let evalResident := (net.paramShapes.size + (if useRunning then 2 * net.bnChannels.size else 0)).toUSize
    let mut correct := 0
    let mut correct5 := 0
    for bi in [0:(if skipEval then 0 else nbt)] do
      -- ⚠ `evalD0`, not `d0`: `evalImg` was drained at the EVAL width, so slicing it at the TRAIN
      -- width would stride through the buffer wrongly from the second row on (RSB-A3: 224² val
      -- rows read as 160²). Silent — it would score a plausible-looking accuracy off garbage.
      let xb := F32.sliceImagesPad evalImg (bi * evalBs) evalBs evalD0 nEval
      -- Hold the eval parameters on device across the eval batches (§2d.3). The
      -- count is the tensor count of `evalShapes`, which for a BN net is the
      -- params PLUS the two running-stat slots per layer — all of them are inputs
      -- with no output counterpart, so all of them can be held.
      let logits ← LowererSession.forwardF32 evalSess evalFn evalParams evalShapes
                      xb xShape evalBs.toUSize nc.toUSize
                      evalResident (ep + 1).toUSize
      for j in [0:min evalBs (nEval - bi * evalBs)] do   -- score real rows only, not the pad
        let pred := (F32.argmaxN logits (j * nc).toUSize nc.toUSize).toNat
        let lbl  := F32.readLabel evalLbl (bi * evalBs + j)
        if pred == lbl then correct := correct + 1
        -- top-5 by the label's RANK, matching the reference's `sum(logits > true_logit) < 5`.
        -- Free: it reads the same logits row already on the host, and it is the metric the
        -- reference's headline is quoted by (72.02% top-1 / 90.62% top-5), which this side could
        -- not state at all until now.
        if (F32.rankOf logits (j * nc).toUSize nc.toUSize lbl.toUSize).toNat < 5 then
          correct5 := correct5 + 1
    let acc := correct.toFloat / nEval.toFloat * 100.0
    let acc5 := correct5.toFloat / nEval.toFloat * 100.0
    -- ⚠ Under `LEAN_MLIR_SKIP_EVAL` the loop above runs ZERO batches, so `correct` is 0 and this
    -- line printed `acc = 0/49920 = 0.000000%  top5 = 0/49920` — a number INDISTINGUISHABLE from a
    -- catastrophically broken net, on a run that scored nothing. Found 2026-08-05 on R50's first
    -- smoke, where it read as the new net being wrong. Exact zeros on BOTH top-1 and top-5 are the
    -- tell (chance at 1000 classes is ~50 and ~250), but a reader should not have to notice that.
    if skipEval then
      IO.println s!"  epoch {ep + 1}: eval SKIPPED (LEAN_MLIR_SKIP_EVAL) — no accuracy was measured"
    else
      IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nEval} = {acc}%  top5 = {correct5}/{nEval} = {acc5}%"
    (← IO.getStdout).flush
    IO.FS.writeBinFile ckptPath thetamv
    IO.FS.writeFile epPath (toString (ep + 1))
  -- Gate G2 (`planning/xla_pjrt_ladder.md` §3). Dumps the whole [θ|m|v] blob, so
  -- the Adam moments are compared too, not just the weights — a moment buffer
  -- that silently failed to thread would still let θ look plausible.
  match ← IO.getEnv "LEAN_MLIR_DUMP_PARAMS" with
  | some path =>
      IO.FS.writeBinFile path thetamv
      IO.println s!"  wrote final [θ|m|v] ({thetamv.size} bytes) → {path}"
  | none => pure ()
  IO.println s!"done (trained {net.name} {variant} + {schedName}/warmup via packed threading)."

/-- **Score a checkpoint, standalone** — the eval half of `trainAdamSched` with no training in
    front of it (`planning/next_session_verified_trainer_code.md` §2).

    Until this existed a verified accuracy could only be produced *in training*, and only for the
    weights that happened to be live at that moment. The JAX side has six `eval_*_full50k.py`; this
    is the verified peer, and it is a FACTORING job rather than new machinery — no new MLIR, no new
    ops, no renderer work. Every piece already existed inside the eval half:

    | need | reused |
    |---|---|
    | drain the val split | `loadData` (all 50,000 as of `ccca380`) |
    | the eval graph | `mkSession` on `<slug>_fwd_eval`, or the `_fwd` chain for the LN nets |
    | eval batch AND width | `fwdRenderedShape`, one parse of one declaration |
    | batching a short tail | `F32.sliceImagesPad` + `min evalBs (nEval − bi·evalBs)` |
    | forward | `LowererSession.forwardF32` |
    | metrics | `F32.argmaxN` (top-1), `F32.rankOf` (top-5) |

    ⭐ **THE GATE IS AN EQUALITY, NOT A SMOKE TEST.** For the same checkpoint at the same region,
    the number printed here must equal the one the training run printed for that epoch — same
    denominator, same batching, same graph. It is available today on ConvNeXt and ViT, which have
    `nBnStats = 0` and therefore carry their whole eval state in the checkpoint.

    ⚠⚠ **BN NETS ARE REFUSED, LOUDLY, AND THAT IS THE POINT.** The checkpoint is exactly
    `[θ|m|v(|ema)]`; the BN running mean/var are NOT in it — they are "reset per process and
    rebuilt within an epoch" (see `runningBnStats`). In-training eval works because the statistics
    have been accumulating all epoch. A fresh process reading a `.bin` has ZEROS, and
    `@<slug>_fwd_eval` then normalises by them: not a slightly-off number, garbage that still
    prints as a plausible-looking percentage. So R50/R34/MNv2/EfficientNet/MNv4 throw here rather
    than score, until §2b lands the stats in the checkpoint (format) plus `--recalibrate` (the
    fallback, and the only one of the two that can reach A3's finished checkpoint).

    ⭐ `region` is what one checkpoint cannot otherwise yield: the driver picks live-or-shadow at
    TRAIN time (`emaLiveBn`), so an EMA run reports one of the two numbers and discards the other.
    timm reports the shadow and RSB-A2 sets `emaDecay := 0.9999`, so without this an A2 result is
    not quotable the way its reference is. `"auto"` = the shadow when the variant has one, matching
    what the training run would have scored; `"live"` and `"ema"` name it explicitly. -/
def VerifiedNet.scoreCheckpoint (net : VerifiedNet) (dataDir : String) (variant : String)
    (ckptPath : String) (region : String := "auto") : IO Unit := do
  let emaOn := VerifiedVariant.emaOn variant
  let nRegions := VerifiedVariant.nRegions variant
  let hasBn := !net.bnChannels.isEmpty
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  net.printBlurb
  IO.println s!"  SCORING A CHECKPOINT — no training. {net.name} {variant}, {ckptPath}"
  -- ⛔⛔ THE BN BLOCKER, asserted before anything expensive happens (§2b). Refuse ahead of the
  -- ~30 GB val drain and the compile, not after: the whole failure being prevented is a number
  -- that looks like a number.
  if hasBn then
    throw <| IO.userError s!"{net.name} has {net.bnChannels.size} batch-norm layers \
({nBnStats} running-stat floats) and the checkpoint does not contain them — it is exactly \
[θ|m|v{if emaOn then "|ema" else ""}]. A fresh process would normalise @{net.slug}_fwd_eval by \
ZEROS and print a plausible-looking percentage off garbage.\n\
  Two exits, neither of them retroactive on its own (planning/next_session_verified_trainer_code.md \
§2b): (a) append the {nBnStats} stat floats to the checkpoint format — clean going forward, but A3's \
finished checkpoint does not contain them; (b) --recalibrate, ~100-200 training batches forward to \
re-accumulate the statistics, which DOES reach an existing checkpoint and is a different estimate \
from the run's own.\n\
  Scoring works today on the LayerNorm nets (ConvNeXt, ViT), which carry no running state."
  -- The region to score. ⚠ `"ema"` on a variant with no fourth region is a REFUSAL and not a
  -- fallback to live: the request and the artifact disagree, and quietly answering the other
  -- question is how a live-weight number gets quoted as a shadow one.
  let regIdx ← match region with
    | "auto" => pure (if emaOn then 3 else 0)
    | "live" => pure 0
    | "ema"  =>
      if !emaOn then
        throw <| IO.userError s!"region 'ema' asked of variant '{variant}', which has no EMA \
shadow — its blob is {nRegions} regions [θ|m|v] and there is nothing in slot 4 to score. Use \
'live', or score a checkpoint written by an ema* variant."
      else pure 3
    | r => throw <| IO.userError s!"unknown region '{r}' — one of auto | live | ema"
  -- Forward resolution, IDENTICAL to `trainAdamSched`'s: the per-variant `_fwd` wins when it
  -- exists, `<slug>_fwd.mlir` is the fallback. ⚠ The FUNCTION is `@<slug>_fwd` either way — the
  -- variant artifact re-renders the same entry name.
  let fwdVariant := s!"{net.mlirDir}/{net.slug}_{variant}_fwd.mlir"
  let fwdPath := if (← System.FilePath.pathExists fwdVariant) then fwdVariant
                 else s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  if !(← System.FilePath.pathExists fwdPath) then
    throw <| IO.userError s!"no forward artifact for {net.slug}: tried {fwdVariant} and {fwdPath}"
  -- ⚠ REFUSE rather than fall back to `(bs, net.d0)`. The training driver can default there
  -- because it has a `cfg.batchSize` the user chose; this tool has no such input, so a guess
  -- would be a silent mis-slice of the val buffer (RSB-A3: 224² rows read as 160²).
  let (evalBs, evalD0) ← match ← fwdRenderedShape fwdPath with
    | some s => pure s
    | none => throw <| IO.userError s!"could not read `%x: tensor<BxWxf32>` off {fwdPath} — the \
eval batch and the eval WIDTH both come from that one declaration, and neither is guessable here."
  if evalD0 != net.d0 then
    IO.println s!"  ▸ EVAL RES SPLIT: net d0 {net.d0}, eval d0 {evalD0} (batch {evalBs}) — read \
off @{net.slug}_fwd"
  -- The checkpoint, and its size guard — the same one `trainAdamSched` applies on resume, for the
  -- same reason: the blob has no header, no fingerprint and no region count, so a layout mismatch
  -- does not fail, it misaligns every parameter and scores silent garbage.
  if !(← System.FilePath.pathExists ckptPath) then
    throw <| IO.userError s!"no checkpoint at {ckptPath}"
  let thetamv ← IO.FS.readBinFile ckptPath
  let pBytes := net.nParams * 4
  let mvBytes := nRegions * pBytes
  if thetamv.size != mvBytes then
    throw <| IO.userError s!"checkpoint {ckptPath} is {thetamv.size} bytes but variant \
'{variant}' wants {mvBytes} ({nRegions} regions x {net.nParams} params x 4). It was written by a \
different blob layout — most likely across the EMA/accumulation boundary, since those variants \
carry a 4th region."
  let theta := thetamv.extract (regIdx * pBytes) ((regIdx + 1) * pBytes)
  IO.println s!"  region {regIdx} of {nRegions} \
({if regIdx == 3 then "the EMA SHADOW" else "the live weights"}), {net.nParams} params"
  (← IO.getStdout).flush
  let sess ← mkSession fwdPath
  -- ⚠ `evalOnly := true` — this tool never touches the train split, and on Imagenette reading it
  -- anyway is 7.4 GB held for nothing. Inert on `.imagenet`, which streams.
  let (_, _, _, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir evalD0 (evalOnly := true)
  let nc := net.nClasses
  let nbt := (nEval + evalBs - 1) / evalBs   -- ceil: the last partial batch is zero-padded
  let xShape := packXShape #[evalBs, evalD0]
  let fwdShapes := net.shapesBA
  let mut correct := 0
  let mut correct5 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * evalBs) evalBs evalD0 nEval
    -- Hold the parameters on device across every batch — one push, not `nbt` of them. `gen` is a
    -- constant because θ never changes here, which is the whole difference from the training loop.
    let logits ← LowererSession.forwardF32 sess s!"m.{net.slug}_fwd" theta fwdShapes
                    xb xShape evalBs.toUSize nc.toUSize
                    net.paramShapes.size.toUSize 1
    for j in [0:min evalBs (nEval - bi * evalBs)] do   -- score real rows only, not the pad
      let pred := (F32.argmaxN logits (j * nc).toUSize nc.toUSize).toNat
      let lbl  := F32.readLabel evalLbl (bi * evalBs + j)
      if pred == lbl then correct := correct + 1
      if (F32.rankOf logits (j * nc).toUSize nc.toUSize lbl.toUSize).toNat < 5 then
        correct5 := correct5 + 1
  let acc := correct.toFloat / nEval.toFloat * 100.0
  let acc5 := correct5.toFloat / nEval.toFloat * 100.0
  -- ⭐ Printed in the SAME shape as the in-training line, so the equality gate is a literal
  -- comparison of two strings rather than an arithmetic one.
  IO.println s!"  checkpoint: acc = {correct}/{nEval} = {acc}%  top5 = {correct5}/{nEval} = {acc5}%"
  if nEval != 50000 && net.data == .imagenet then
    IO.println s!"  ⚠ val is {nEval} of ImageNet's 50,000 — this is NOT over timm's denominator"
  (← IO.getStdout).flush

/-- Train driver for the **2-parameter linear** path (Chapter 1). The verified
    `@<slug>_train_step` takes `W0`/`b0` as *separate* arguments (`linearTrainStepV`),
    weights are zero-initialized, and the loss/lr are baked into the MLIR — distinct
    from the packed-params, He-init `train` above. Only the linear classifier uses this;
    shares `compileVmfb` / `loadData` / the eval pass with the main driver. -/
def VerifiedNet.trainLinear (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  net.printBlurb
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _trainPix, _crop) ←
    loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; dense {d0}->{d1}, bs {bs}, SGD"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: the last partial batch is zero-padded, not dropped
  let shapes := net.shapesBA          -- packed [W0|b0] layout for the verified forward
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut W0 ← F32.const (d0 * d1).toUSize 0.0
  let mut b0 ← F32.const d1.toUSize 0.0
  -- LEAN_MLIR_PERTURB_R, as in `train`/`trainAdamSched`. Without it this loop is
  -- the third for which `scripts/residency_gate.sh`'s init CONTROL is a silent
  -- no-op. Weights are ZERO-initialised here rather than He, so the displacement
  -- is off zero — if anything a cleaner control.
  match (← IO.getEnv "LEAN_MLIR_PERTURB_R").bind (·.toNat?) with
  | some n => do
      let r := n.toFloat * 1e-9
      IO.println s!"  ▸ PERTURBED init: theta += r*u with ||r*u||_2 = {r}"
      W0 ← F32.perturbUnit W0 0 (d0 * d1).toUSize r 12345
  | none   => pure ()
  -- Device-resident parameters (§2d.3): `W0` and `b0` — the WHOLE parameter set,
  -- since this graph is `(x, W0, b0, onehot) → (W0n, b0n)`.
  let nResident : USize := 2
  let pBytes := (d0 * d1 + d1) * 4
  -- The packed `[W0|b0]` the step returns, carried so the epoch boundary has ONE
  -- thing to make authoritative — the role `pbuf` plays in `trainAdamSched`.
  let mut packed := W0 ++ b0
  -- LEAN_MLIR_MAX_EPOCHS cap + per-epoch (Nms) timing, matching `train` (used by
  -- `lake run benchmark`); opt-in, full cfg.epochs otherwise.
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  for ep in [0:nEpochs] do
    let tEp0 ← IO.monoMsNow
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← LowererSession.linearTrainStepV tsSess tsFn
                  xb W0 b0 yb bs.toUSize d0.toUSize d1.toUSize nResident
      packed := out
      -- The per-step split is what the COPYING path needs: `W0`/`b0` are separate
      -- FFI arguments, so they have to be re-sliced every step. Under residency
      -- the shim ignores both operands and this slices an unwritten buffer — 31 KB
      -- of wasted memcpy on a net this size, and harmless, because the epoch
      -- boundary below makes `packed` authoritative before anything reads it.
      W0 := out.extract 0 (d0 * d1 * 4)
      b0 := out.extract (d0 * d1 * 4) pBytes
    -- Bring the parameters back for eval and the G2 dump. Inert without residency
    -- (it is the copy `packed` already was); with it, the one d2h per epoch.
    packed ← LowererSession.readParams tsSess packed pBytes.toUSize
    W0 := packed.extract 0 (d0 * d1 * 4)
    b0 := packed.extract (d0 * d1 * 4) pBytes
    let params := packed
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn params shapes
                      xb xShape bs.toUSize d1.toUSize
                      nResident (ep + 1).toUSize
      for j in [0:min bs (nEval - bi * bs)] do   -- score real rows only, not the pad
        let pred := (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat
        let lbl  := F32.readLabel evalLbl (bi * bs + j)
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / nEval.toFloat * 100.0
    let epMs := (← IO.monoMsNow) - tEp0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nEval} = {acc}% ({epMs}ms)"
    (← IO.getStdout).flush
  -- Gate G2 (`planning/xla_pjrt_ladder.md` §3): dump the final parameters so the
  -- IREE and XLA builds can be diffed tensor-for-tensor. Equal accuracy is a
  -- summary statistic, not a tie — this is the actual comparison.
  match ← IO.getEnv "LEAN_MLIR_DUMP_PARAMS" with
  | some path =>
      -- `packed` and not `W0 ++ b0`: under residency the two slices are only
      -- authoritative because the epoch boundary re-derived them from it, and if
      -- the loop ran zero epochs they never were.
      IO.FS.writeBinFile path packed
      IO.println s!"  wrote final params ({packed.size} bytes) → {path}"
  | none => pure ()
  IO.println s!"done (trained {net.name} via the proof-rendered StableHLO)."

/-- Phase-3 PGD-step kernel for the linear classifier (`planning/robustness.md`).
    `forward → softmax-CE input gradient dx = (softmax(xW+b) − onehot)·Wᵀ` (the proven
    linear input-VJP, `Proofs.mlpInputGrad`'s 1-layer case) → L∞ sign-step → project to the
    `eps`-ball around `x0` → clip to [0,1]. Returns the advanced adversarial input `x_adv`.
    `eps`/`alpha` baked as constants (recompiled per sweep point). Invoked via the generic
    `forwardF32` FFI with `onehot`+`x0` in the params blob and `nClasses := d0` (output size) —
    no new FFI/C shim. The whole PGD step runs on the GPU; the host just iterates. -/
private def genLinearPgdStep (bs d0 d1 : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let bxd0 := s!"tensor<{bs}x{d0}xf32>"
  let bxd1 := s!"tensor<{bs}x{d1}xf32>"
  let wty  := s!"tensor<{d0}x{d1}xf32>"
  let bty  := s!"tensor<{d1}xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  -- shared: forward → softmax-CE input gradient %dx, then the broadcast constants
  let header :=
    "module @m {\n" ++
    s!"  func.func @linear_pgd_step(%x: {bxd0}, %W0: {wty}, %b0: {bty}, %onehot: {bxd1}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %mm = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxd0}, {wty}) -> {bxd1}\n" ++
    s!"    %bb = stablehlo.broadcast_in_dim %b0, dims = [1] : ({bty}) -> {bxd1}\n" ++
    s!"    %logits = stablehlo.add %mm, %bb : {bxd1}\n" ++
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {bxd1}\n" ++
    s!"    %exp = stablehlo.exponential %shift : {bxd1}\n" ++
    s!"    %ssum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %softmax = stablehlo.divide %exp, %ssumb : {bxd1}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {bxd1}\n" ++
    s!"    %dx = stablehlo.dot_general %g, %W0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxd1}, {wty}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  -- step + projection: L∞ (sign, box-clip to x0±eps) or L2 (normalized grad, eps-ball)
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %step = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %step : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %c1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %c1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %step = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %step : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %c3 = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %c4 = stablehlo.minimum %c3, %oneb : {bxd0}\n" ++
  s!"    return %c4 : {bxd0}\n" ++
  "  }\n}\n"

/-- Build a one-hot `[bs, d1]` f32 batch from int32-LE labels (1.0 = bytes 00 00 80 3F). -/
private def oneHotBatch (labels : ByteArray) (start bs d1 : Nat) : IO ByteArray := do
  let mut oh ← F32.const (bs * d1).toUSize 0.0
  for j in [0:bs] do
    let lbl := (labels.get! (4 * (start + j))).toNat
    let fi := j * d1 + lbl
    oh := (((oh.set! (4*fi) 0).set! (4*fi+1) 0).set! (4*fi+2) 0x80).set! (4*fi+3) 0x3F
  return oh

/-- `oneHotBatch` for a possibly-partial final batch: rows past `total` records get an
    all-zero row (their padded images are never scored, so the gradient they induce is
    irrelevant — the row just has to exist because the batch dim is baked into the vmfb). -/
private def oneHotBatchPad (labels : ByteArray) (start bs d1 total : Nat) : IO ByteArray := do
  let mut oh ← F32.const (bs * d1).toUSize 0.0
  for j in [0:min bs (total - start)] do
    let lbl := (labels.get! (4 * (start + j))).toNat
    let fi := j * d1 + lbl
    oh := (((oh.set! (4*fi) 0).set! (4*fi+1) 0).set! (4*fi+2) 0x80).set! (4*fi+3) 0x3F
  return oh

/-- Spectral norm `‖W‖₂` of `W : [d0,d1]` (row-major) by power iteration on the small
    `WᵀW : [d1,d1]` Gram matrix. For the linear net this IS the global Lipschitz constant
    of the logit map (`logits = xW+b`, Jacobian `Wᵀ`). Host-side, pure. -/
private def specNormW (W : ByteArray) (d0 d1 : Nat) : Float := Id.run do
  let g := fun (i j : Nat) => Id.run do      -- WᵀW[i,j] = Σ_k W[k,i]·W[k,j]
    let mut s := 0.0
    for k in [0:d0] do
      s := s + (F32.read W (k*d1+i).toUSize) * (F32.read W (k*d1+j).toUSize)
    pure s
  let mut wtw : Array Float := Array.replicate (d1*d1) 0.0
  for i in [0:d1] do
    for j in [0:d1] do
      wtw := wtw.set! (i*d1+j) (g i j)
  let mv := fun (v : Array Float) => Id.run do  -- WᵀW · v
    let mut u : Array Float := Array.replicate d1 0.0
    for i in [0:d1] do
      let mut s := 0.0
      for j in [0:d1] do s := s + wtw[i*d1+j]! * v[j]!
      u := u.set! i s
    pure u
  let mut v : Array Float := Array.replicate d1 1.0
  for _ in [0:60] do
    let u := mv v
    let mut nrm := 0.0
    for i in [0:d1] do nrm := nrm + u[i]!*u[i]!
    nrm := Float.sqrt nrm
    if nrm > 1e-20 then
      for i in [0:d1] do v := v.set! i (u[i]!/nrm)
  let u := mv v
  let mut lam := 0.0
  for i in [0:d1] do lam := lam + v[i]! * u[i]!   -- Rayleigh quotient (‖v‖=1)
  pure (Float.sqrt lam)

/-- Spectral norm `‖M‖₂` of a `[rows, cols]` matrix given by an index function `get i j`
    (the same power iteration on the `cols×cols` Gram as `specNormW`, but reading via `get`
    so it works on strided sub-tensors — e.g. one tap-plane of a conv kernel). -/
private def specNormGet (get : Nat → Nat → Float) (rows cols : Nat) : Float := Id.run do
  let gram := fun (i j : Nat) => Id.run do        -- (MᵀM)[i,j] = Σ_k M[k,i]·M[k,j]
    let mut s := 0.0
    for k in [0:rows] do s := s + (get k i) * (get k j)
    pure s
  let mut wtw : Array Float := Array.replicate (cols*cols) 0.0
  for i in [0:cols] do
    for j in [0:cols] do
      wtw := wtw.set! (i*cols+j) (gram i j)
  let mv := fun (v : Array Float) => Id.run do
    let mut u : Array Float := Array.replicate cols 0.0
    for i in [0:cols] do
      let mut s := 0.0
      for j in [0:cols] do s := s + wtw[i*cols+j]! * v[j]!
      u := u.set! i s
    pure u
  let mut v : Array Float := Array.replicate cols 1.0
  for _ in [0:60] do
    let u := mv v
    let mut nrm := 0.0
    for i in [0:cols] do nrm := nrm + u[i]!*u[i]!
    nrm := Float.sqrt nrm
    if nrm > 1e-20 then
      for i in [0:cols] do v := v.set! i (u[i]!/nrm)
  let u := mv v
  let mut lam := 0.0
  for i in [0:cols] do lam := lam + v[i]! * u[i]!
  pure (Float.sqrt lam)

/-- A **sound** (loose) upper bound on the L2 operator norm of a zero-padded 2-D
    convolution with kernel `W : [outC, inC, kh, kw]` (row-major). Writing the conv as a
    sum over spatial taps `T = Σ_{ky,kx} S_{ky,kx} ∘ M_{ky,kx}` — each `S` a (norm ≤ 1)
    shift and each `M` the pointwise `[outC,inC]` channel-mixing matrix at that tap — the
    triangle inequality gives `‖T‖₂ ≤ Σ_{ky,kx} ‖W[:,:,ky,kx]‖₂`. Each tap-plane's spectral
    norm is the same power iteration as `specNormW`. Loose by up to `√(kh·kw)` vs the exact
    (Sedghi–Gupta–Long) value — which only sharpens the "depth ⇒ vacuous product" message. -/
private def specNormConvTapSum (W : ByteArray) (outC inC kh kw : Nat) : Float := Id.run do
  let mut s := 0.0
  for ky in [0:kh] do
    for kx in [0:kw] do
      s := s + specNormGet
        (fun o i => F32.read W (((o*inC+i)*kh+ky)*kw+kx).toUSize) outC inC
  pure s

/-- **Matrix-free** spectral norm `‖W‖₂` of `W : [d0,d1]` (row-major) — power iteration that
    applies `W` and `Wᵀ` as mat-vecs (`σ = ‖W v‖`, `v` the top right singular vector) instead
    of forming the `d1×d1` Gram. ~`2·d0·d1` per iteration vs `d0·d1²` for `specNormW`, so it's
    cheap enough to call **during** training (the spectral-norm projection below). Fewer iters
    (`iters`) trade a little precision for speed; `specNormW` stays the high-precision cert path. -/
private def specNormMV (W : ByteArray) (d0 d1 : Nat) (iters : Nat := 15) : Float := Id.run do
  let norm := fun (a : Array Float) (n : Nat) => Id.run do
    let mut s := 0.0
    for i in [0:n] do s := s + a[i]! * a[i]!
    pure (Float.sqrt s)
  let normalize := fun (a : Array Float) (n : Nat) => Id.run do
    let s := norm a n
    if s > 1e-20 then
      let mut b := a
      for i in [0:n] do b := b.set! i (a[i]! / s)
      pure b
    else pure a
  let mut v : Array Float := normalize (Array.replicate d1 1.0) d1
  let mut σ := 0.0
  for _ in [0:iters] do
    let mut u : Array Float := Array.replicate d0 0.0    -- u = W v
    for r in [0:d0] do
      let mut s := 0.0
      for cc in [0:d1] do s := s + F32.read W (r*d1+cc).toUSize * v[cc]!
      u := u.set! r s
    σ := norm u d0                                       -- σ = ‖W v‖ (‖v‖ = 1)
    let mut w : Array Float := Array.replicate d1 0.0    -- w = Wᵀ u
    for cc in [0:d1] do
      let mut s := 0.0
      for r in [0:d0] do s := s + F32.read W (r*d1+cc).toUSize * u[r]!
      w := w.set! cc s
    v := normalize w d1
  pure σ

/-- **Spectral-norm projection** (projected SGD onto the spectral ball): rescale every weight
    whose L2-Lipschitz bound exceeds `c` down to `c`, leaving biases untouched. Dense `[d0,d1]`:
    cap the spectral norm `‖W‖₂` (`specNormMV`). Conv `[o,i,kh,kw]`: cap the **same** tap-sum
    operator bound the CNN certificate uses (`specNormConvTapSum`, `‖T‖₂ ≤ Σ_tap‖W[:,:,ky,kx]‖₂`)
    by scaling the whole kernel — so the projection and the cert control the identical quantity.
    Caps each layer's Lipschitz constant at `c`, so the global `L = ∏ᵢ ≤ cᵏ` — the lever that
    turns the (vacuous) product certificate non-vacuous. `F32.scaleShift` does the rescale. -/
private def projectSpectral (theta : ByteArray) (specs : Array (Array Nat × Nat)) (c : Float)
    : IO ByteArray := do
  let mut parts : Array ByteArray := #[]
  let mut off := 0
  for spec in specs do
    let dims := spec.1
    let len := dims.foldl (·*·) 1
    let slice := theta.extract (off*4) ((off+len)*4)
    let slice' ← if dims.size == 2 then do
        let σ := specNormMV slice dims[0]! dims[1]!
        if σ > c then F32.scaleShift slice (c/σ) 0.0 else pure slice
      else if dims.size == 4 then do
        let s := specNormConvTapSum slice dims[0]! dims[1]! dims[2]! dims[3]!
        if s > c then F32.scaleShift slice (c/s) 0.0 else pure slice
      else pure slice
    parts := parts.push slice'
    off := off + len
  return F32.concat parts

/-- Phase-3 PGD-step kernel for the 2-hidden-layer MLP (`d0→h→h→d1`, ReLU). Forward
    (saving the pre-activations `z0,z1`) → the proven `mlpInputGrad` VJP
    `dx = ((g·W₂ᵀ ⊙ relu'(z₁))·W₁ᵀ ⊙ relu'(z₀))·W₀ᵀ` (ReLU masks via `compare GT`/`select`,
    the codegen's idiom) → L∞/L2 step + projection. Returns `x_adv`. -/
private def genMlpPgdStep (bs d0 h d1 : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let bxd0 := s!"tensor<{bs}x{d0}xf32>"
  let bxh  := s!"tensor<{bs}x{h}xf32>"
  let bxd1 := s!"tensor<{bs}x{d1}xf32>"
  let bxhi := s!"tensor<{bs}x{h}xi1>"
  let w0ty := s!"tensor<{d0}x{h}xf32>"
  let w1ty := s!"tensor<{h}x{h}xf32>"
  let w2ty := s!"tensor<{h}x{d1}xf32>"
  let hbty := s!"tensor<{h}xf32>"
  let d1bt := s!"tensor<{d1}xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  let header :=
    "module @m {\n" ++
    s!"  func.func @mlp_pgd_step(%x: {bxd0}, %W0: {w0ty}, %b0: {hbty}, %W1: {w1ty}, %b1: {hbty}, %W2: {w2ty}, %b2: {d1bt}, %onehot: {bxd1}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zh = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxh}\n" ++
    -- forward (save preacts z0, z1)
    s!"    %z0mm = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxd0}, {w0ty}) -> {bxh}\n" ++
    s!"    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1] : ({hbty}) -> {bxh}\n" ++
    s!"    %z0 = stablehlo.add %z0mm, %b0b : {bxh}\n" ++
    s!"    %h0 = stablehlo.maximum %z0, %zh : {bxh}\n" ++
    s!"    %z1mm = stablehlo.dot_general %h0, %W1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxh}, {w1ty}) -> {bxh}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : ({hbty}) -> {bxh}\n" ++
    s!"    %z1 = stablehlo.add %z1mm, %b1b : {bxh}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %zh : {bxh}\n" ++
    s!"    %lgmm = stablehlo.dot_general %h1, %W2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxh}, {w2ty}) -> {bxd1}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : ({d1bt}) -> {bxd1}\n" ++
    s!"    %logits = stablehlo.add %lgmm, %b2b : {bxd1}\n" ++
    -- softmax-CE gradient g
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {bxd1}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {bxd1}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {bxd1}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {bxd1}\n" ++
    -- backward: dx = ((g·W2ᵀ ⊙ relu'(z1))·W1ᵀ ⊙ relu'(z0))·W0ᵀ
    s!"    %dh1 = stablehlo.dot_general %g, %W2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxd1}, {w2ty}) -> {bxh}\n" ++
    s!"    %rm1 = stablehlo.compare GT, %z1, %zh : ({bxh}, {bxh}) -> {bxhi}\n" ++
    s!"    %dz1 = stablehlo.select %rm1, %dh1, %zh : {bxhi}, {bxh}\n" ++
    s!"    %dh0 = stablehlo.dot_general %dz1, %W1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxh}, {w1ty}) -> {bxh}\n" ++
    s!"    %rm0 = stablehlo.compare GT, %z0, %zh : ({bxh}, {bxh}) -> {bxhi}\n" ++
    s!"    %dz0 = stablehlo.select %rm0, %dh0, %zh : {bxhi}, {bxh}\n" ++
    s!"    %dx = stablehlo.dot_general %dz0, %W0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxh}, {w0ty}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD-step kernel for the verified MNIST CNN** (`conv 1→32 → relu → conv 32→32 →
    relu → maxpool 28→14 → flatten → dense 6272→512 → relu → 512→512 → relu → 512→10`).
    Forward (saving every pre-activation + the maxpool input) → softmax-CE seed → the full
    input-VJP `dx`, mirroring `verified_mlir/cnn_train_step.mlir`'s backward ops:
    `dot_general` adjoints + ReLU masks (`compare GT`/`select`), **maxpool-back**
    (`select_and_scatter`, scatter the pooled cotangent to the argmax cells), and the two
    **conv input-VJPs** (transpose-`o,i` + spatial `reverse` of the kernel, then the same
    padded conv). The train step stops at `dz1` (it only needs weight grads); here we add the
    final conv1 input-VJP to reach `dx` over the pixels. Then the L∞ sign-step / L2 projected
    step + ε-ball project + [0,1] clip. Architecture is fixed; only `bs`/`eps`/`alpha` vary. -/
private def genCnnPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let i4  := s!"tensor<{bs}x1x28x28xf32>"
  let c4  := s!"tensor<{bs}x32x28x28xf32>"
  let c4i := s!"tensor<{bs}x32x28x28xi1>"
  let p4  := s!"tensor<{bs}x32x14x14xf32>"
  let f2  := s!"tensor<{bs}x6272xf32>"
  let h2  := s!"tensor<{bs}x512xf32>"
  let h2i := s!"tensor<{bs}x512xi1>"
  let o2  := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x784xf32>"
  let rty := s!"tensor<{bs}xf32>"
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cnn_pgd_step(%x: {bxd0}, %W1: tensor<32x1x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<6272x512xf32>, %b3: tensor<512xf32>, %W4: tensor<512x512xf32>, %b4: tensor<512xf32>, %W5: tensor<512x10xf32>, %b5: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zc4 = stablehlo.constant dense<0.0> : {c4}\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    -- ── forward (save pre-acts z1,z2,z3,z4 + maxpool input h2c) ──
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x1x3x3xf32>) -> {c4}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> {c4}\n" ++
    s!"    %z1 = stablehlo.add %c1, %b1b : {c4}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %zc4 : {c4}\n" ++
    s!"    %c2 = stablehlo.convolution(%h1, %W2)\n      {convCfg} : ({c4}, tensor<32x32x3x3xf32>) -> {c4}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> {c4}\n" ++
    s!"    %z2 = stablehlo.add %c2, %b2b : {c4}\n" ++
    s!"    %h2c = stablehlo.maximum %z2, %zc4 : {c4}\n" ++
    s!"    %pool = \"stablehlo.reduce_window\"(%h2c, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) \{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : ({c4}, tensor<f32>) -> {p4}\n" ++
    s!"    %flat = stablehlo.reshape %pool : ({p4}) -> {f2}\n" ++
    s!"    %d3 = stablehlo.dot_general %flat, %W3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<6272x512xf32>) -> {h2}\n" ++
    s!"    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z3 = stablehlo.add %d3, %b3b : {h2}\n" ++
    s!"    %h3 = stablehlo.maximum %z3, %zh : {h2}\n" ++
    s!"    %d4 = stablehlo.dot_general %h3, %W4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z4 = stablehlo.add %d4, %b4b : {h2}\n" ++
    s!"    %h4 = stablehlo.maximum %z4, %zh : {h2}\n" ++
    s!"    %d5 = stablehlo.dot_general %h4, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d5, %b5b : {o2}\n" ++
    -- ── softmax-CE seed g = softmax(logits) − onehot ──
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- ── backward to dx ──
    s!"    %dh4 = stablehlo.dot_general %g, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm4 = stablehlo.compare GT, %z4, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz4 = stablehlo.select %rm4, %dh4, %zh : {h2i}, {h2}\n" ++
    s!"    %dh3 = stablehlo.dot_general %dz4, %W4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm3 = stablehlo.compare GT, %z3, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz3 = stablehlo.select %rm3, %dh3, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz3, %W3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<6272x512xf32>) -> {f2}\n" ++
    s!"    %dpool = stablehlo.reshape %dflat : ({f2}) -> {p4}\n" ++
    -- maxpool-back: scatter the pooled cotangent back to the argmax cells of the pool input
    s!"    %dpre2 = \"stablehlo.select_and_scatter\"(%h2c, %dpool, %zf) (\{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    s!"    }) \{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : ({c4}, {p4}, tensor<f32>) -> {c4}\n" ++
    s!"    %rmc2 = stablehlo.compare GT, %z2, %zc4 : ({c4}, {c4}) -> {c4i}\n" ++
    s!"    %dz2 = stablehlo.select %rmc2, %dpre2, %zc4 : {c4i}, {c4}\n" ++
    -- conv2 input-VJP: transpose o,i + spatial-reverse the kernel, conv with the cotangent
    s!"    %w2t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>\n" ++
    s!"    %w2r = stablehlo.reverse %w2t, dims = [2, 3] : tensor<32x32x3x3xf32>\n" ++
    s!"    %dpost1 = stablehlo.convolution(%dz2, %w2r)\n      {convCfg} : ({c4}, tensor<32x32x3x3xf32>) -> {c4}\n" ++
    s!"    %rmc1 = stablehlo.compare GT, %z1, %zc4 : ({c4}, {c4}) -> {c4i}\n" ++
    s!"    %dz1 = stablehlo.select %rmc1, %dpost1, %zc4 : {c4i}, {c4}\n" ++
    -- conv1 input-VJP → dx over the pixels (the step the train kernel omits; W1: 32x1x3x3 → 1x32x3x3)
    s!"    %w1t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x1x3x3xf32>) -> tensor<1x32x3x3xf32>\n" ++
    s!"    %w1r = stablehlo.reverse %w1t, dims = [2, 3] : tensor<1x32x3x3xf32>\n" ++
    s!"    %dxi = stablehlo.convolution(%dz1, %w1r)\n      {convCfg} : ({c4}, tensor<1x32x3x3xf32>) -> {i4}\n" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD-step kernel for the verified CIFAR-10 CNN** — the deeper sibling of
    `genCnnPgdStep` (`conv 3→32 → relu → conv 32→32 → relu → maxpool → conv 32→64 → relu →
    conv 64→64 → relu → maxpool → flatten(4096) → 512 → 512 → 10`). Same recipe — forward
    (saving every pre-activation + both maxpool inputs) → softmax-CE seed → the full input-VJP
    `dx`, mirroring `verified_mlir/cifar_train_step.mlir`'s backward (4 conv input-VJPs, 2
    `select_and_scatter` maxpool-backs, ReLU masks, dense adjoints) + the final conv1 input-VJP
    the train step omits — then the L∞/L2 step + ε-ball project + [0,1] clip. 3-channel 32×32,
    `bs`/`eps`/`alpha` vary. -/
private def genCifarPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let i4   := s!"tensor<{bs}x3x32x32xf32>"
  let m32  := s!"tensor<{bs}x32x32x32xf32>"
  let m32i := s!"tensor<{bs}x32x32x32xi1>"
  let p32  := s!"tensor<{bs}x32x16x16xf32>"
  let m64  := s!"tensor<{bs}x64x16x16xf32>"
  let m64i := s!"tensor<{bs}x64x16x16xi1>"
  let p64  := s!"tensor<{bs}x64x8x8xf32>"
  let f2   := s!"tensor<{bs}x4096xf32>"
  let h2   := s!"tensor<{bs}x512xf32>"
  let h2i  := s!"tensor<{bs}x512xi1>"
  let o2   := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x3072xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let poolAttr := "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cifar_pgd_step(%x: {bxd0}, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %z32 = stablehlo.constant dense<0.0> : {m32}\n" ++
    s!"    %z64 = stablehlo.constant dense<0.0> : {m64}\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    -- ── forward (save pre-acts z1,z2,z3,z4,z5,z6 + both maxpool inputs h2c,h4c) ──
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x3x3x3xf32>) -> {m32}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> {m32}\n" ++
    s!"    %z1 = stablehlo.add %c1, %b1b : {m32}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %z32 : {m32}\n" ++
    s!"    %c2 = stablehlo.convolution(%h1, %W2)\n      {convCfg} : ({m32}, tensor<32x32x3x3xf32>) -> {m32}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> {m32}\n" ++
    s!"    %z2 = stablehlo.add %c2, %b2b : {m32}\n" ++
    s!"    %h2c = stablehlo.maximum %z2, %z32 : {m32}\n" ++
    s!"    %pool1 = \"stablehlo.reduce_window\"(%h2c, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m32}, tensor<f32>) -> {p32}\n" ++
    s!"    %c3 = stablehlo.convolution(%pool1, %W3)\n      {convCfg} : ({p32}, tensor<64x32x3x3xf32>) -> {m64}\n" ++
    s!"    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> {m64}\n" ++
    s!"    %z3 = stablehlo.add %c3, %b3b : {m64}\n" ++
    s!"    %h3 = stablehlo.maximum %z3, %z64 : {m64}\n" ++
    s!"    %c4 = stablehlo.convolution(%h3, %W4)\n      {convCfg} : ({m64}, tensor<64x64x3x3xf32>) -> {m64}\n" ++
    s!"    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> {m64}\n" ++
    s!"    %z4 = stablehlo.add %c4, %b4b : {m64}\n" ++
    s!"    %h4c = stablehlo.maximum %z4, %z64 : {m64}\n" ++
    s!"    %pool2 = \"stablehlo.reduce_window\"(%h4c, %ninf) (\{\n" ++
    "      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):\n" ++
    "        %qm = stablehlo.maximum %qa, %qb : tensor<f32>\n" ++
    "        stablehlo.return %qm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m64}, tensor<f32>) -> {p64}\n" ++
    s!"    %flat = stablehlo.reshape %pool2 : ({p64}) -> {f2}\n" ++
    s!"    %d5 = stablehlo.dot_general %flat, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<4096x512xf32>) -> {h2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z5 = stablehlo.add %d5, %b5b : {h2}\n" ++
    s!"    %h5 = stablehlo.maximum %z5, %zh : {h2}\n" ++
    s!"    %d6 = stablehlo.dot_general %h5, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z6 = stablehlo.add %d6, %b6b : {h2}\n" ++
    s!"    %h6 = stablehlo.maximum %z6, %zh : {h2}\n" ++
    s!"    %d7 = stablehlo.dot_general %h6, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b7b = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d7, %b7b : {o2}\n" ++
    -- ── softmax-CE seed ──
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- ── backward to dx ──
    s!"    %dh6 = stablehlo.dot_general %g, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm6 = stablehlo.compare GT, %z6, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz6 = stablehlo.select %rm6, %dh6, %zh : {h2i}, {h2}\n" ++
    s!"    %dh5 = stablehlo.dot_general %dz6, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm5 = stablehlo.compare GT, %z5, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz5 = stablehlo.select %rm5, %dh5, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz5, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<4096x512xf32>) -> {f2}\n" ++
    s!"    %dpool2 = stablehlo.reshape %dflat : ({f2}) -> {p64}\n" ++
    -- maxpool2-back
    s!"    %dpre4 = \"stablehlo.select_and_scatter\"(%h4c, %dpool2, %zf) (\{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m64}, {p64}, tensor<f32>) -> {m64}\n" ++
    s!"    %rmc4 = stablehlo.compare GT, %z4, %z64 : ({m64}, {m64}) -> {m64i}\n" ++
    s!"    %dz4 = stablehlo.select %rmc4, %dpre4, %z64 : {m64i}, {m64}\n" ++
    -- conv4 input-VJP
    s!"    %w4t = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>\n" ++
    s!"    %w4r = stablehlo.reverse %w4t, dims = [2, 3] : tensor<64x64x3x3xf32>\n" ++
    s!"    %dpost3 = stablehlo.convolution(%dz4, %w4r)\n      {convCfg} : ({m64}, tensor<64x64x3x3xf32>) -> {m64}\n" ++
    s!"    %rmc3 = stablehlo.compare GT, %z3, %z64 : ({m64}, {m64}) -> {m64i}\n" ++
    s!"    %dz3 = stablehlo.select %rmc3, %dpost3, %z64 : {m64i}, {m64}\n" ++
    -- conv3 input-VJP (W3: 64x32x3x3 → 32x64x3x3): grad back to the pool1 output [bs,32,16,16]
    s!"    %w3t = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>\n" ++
    s!"    %w3r = stablehlo.reverse %w3t, dims = [2, 3] : tensor<32x64x3x3xf32>\n" ++
    s!"    %dpool1 = stablehlo.convolution(%dz3, %w3r)\n      {convCfg} : ({m64}, tensor<32x64x3x3xf32>) -> {p32}\n" ++
    -- maxpool1-back
    s!"    %dpre2 = \"stablehlo.select_and_scatter\"(%h2c, %dpool1, %zf) (\{\n" ++
    "      ^bb0(%ta: tensor<f32>, %tb: tensor<f32>):\n" ++
    "        %tge = stablehlo.compare GE, %ta, %tb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %tge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%tc: tensor<f32>, %td: tensor<f32>):\n" ++
    "        %ts = stablehlo.add %tc, %td : tensor<f32>\n" ++
    "        stablehlo.return %ts : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m32}, {p32}, tensor<f32>) -> {m32}\n" ++
    s!"    %rmc2 = stablehlo.compare GT, %z2, %z32 : ({m32}, {m32}) -> {m32i}\n" ++
    s!"    %dz2 = stablehlo.select %rmc2, %dpre2, %z32 : {m32i}, {m32}\n" ++
    -- conv2 input-VJP
    s!"    %w2t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>\n" ++
    s!"    %w2r = stablehlo.reverse %w2t, dims = [2, 3] : tensor<32x32x3x3xf32>\n" ++
    s!"    %dpost1 = stablehlo.convolution(%dz2, %w2r)\n      {convCfg} : ({m32}, tensor<32x32x3x3xf32>) -> {m32}\n" ++
    s!"    %rmc1 = stablehlo.compare GT, %z1, %z32 : ({m32}, {m32}) -> {m32i}\n" ++
    s!"    %dz1 = stablehlo.select %rmc1, %dpost1, %z32 : {m32i}, {m32}\n" ++
    -- conv1 input-VJP → dx (W1: 32x3x3x3 → 3x32x3x3)
    s!"    %w1t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x3x3x3xf32>) -> tensor<3x32x3x3xf32>\n" ++
    s!"    %w1r = stablehlo.reverse %w1t, dims = [2, 3] : tensor<3x32x3x3xf32>\n" ++
    s!"    %dxi = stablehlo.convolution(%dz1, %w1r)\n      {convCfg} : ({m32}, tensor<3x32x3x3xf32>) -> {i4}\n" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD-step kernel for the verified CIFAR-10 CNN + per-channel BatchNorm** (`cifar_bn`).
    The net's "BN" is **instance normalization** (`cifar_bn_fwd`: each image normalized over its
    spatial dims per channel, `nf=H·W`) — so the per-image gradient is clean, there's no train/eval
    split, and the deployed forward IS the attacked forward. Structure: `conv→+b→BN→relu` ×4 (2
    maxpools) → 3 denses. Forward saves every BN's `istd`/`xhat` + post-acts; backward runs the full
    input-VJP to `dx`: dense adjoints, ReLU masks, 2 maxpool `select_and_scatter`-backs, the BN
    grad-input 3-term formula `dx = istd·(dxhat − meanₛ dxhat − xhat·meanₛ(dxhat·xhat))` (per image,
    over spatial), and the 4 conv input-VJPs (transpose-`o,i`+`reverse`) + the conv1 VJP to the
    pixels. `bnBlock`/`bnBack` emit the repeated BN forward/backward. **No certificate** — instance
    norm absorbs the conv weight scale and its Lipschitz is data-dependent (`γ·istd`), a separate
    problem; this is the attack rung only. -/
private def genCifarBnPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let poolAttr := "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}"
  -- BN forward + bias + relu for block k (C channels, N spatial, H side): emits %hc{k} (conv+bias),
  -- saves %istd{k}/%xhat{k}/%gb{k}/%znk{k}/%nfk{k}/%mask{k}, post-relu reshaped to 4D as %h{k}4.
  let bnBlock := fun (k C N H : Nat) (convOut : String) =>
    let t4 := s!"tensor<{bs}x{C}x{H}x{H}xf32>"
    let tn := s!"tensor<{bs}x{C}x{N}xf32>"
    let tc := s!"tensor<{bs}x{C}xf32>"
    let tni := s!"tensor<{bs}x{C}x{N}xi1>"
    let cty := s!"tensor<{C}xf32>"
    s!"    %bbb{k} = stablehlo.broadcast_in_dim %b{k}, dims = [1] : ({cty}) -> {t4}\n" ++
    s!"    %hc{k} = stablehlo.add {convOut}, %bbb{k} : {t4}\n" ++
    s!"    %nfk{k} = stablehlo.constant dense<{N}.0> : {tn}\n" ++
    s!"    %epk{k} = stablehlo.constant dense<1.0e-05> : {tn}\n" ++
    s!"    %znk{k} = stablehlo.constant dense<0.0> : {tn}\n" ++
    s!"    %xr{k} = stablehlo.reshape %hc{k} : ({t4}) -> {tn}\n" ++
    s!"    %smr{k} = stablehlo.reduce(%xr{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %smb{k} = stablehlo.broadcast_in_dim %smr{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %mu{k} = stablehlo.divide %smb{k}, %nfk{k} : {tn}\n" ++
    s!"    %xc{k} = stablehlo.subtract %xr{k}, %mu{k} : {tn}\n" ++
    s!"    %sq{k} = stablehlo.multiply %xc{k}, %xc{k} : {tn}\n" ++
    s!"    %vsr{k} = stablehlo.reduce(%sq{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %vsb{k} = stablehlo.broadcast_in_dim %vsr{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %var{k} = stablehlo.divide %vsb{k}, %nfk{k} : {tn}\n" ++
    s!"    %ve{k} = stablehlo.add %var{k}, %epk{k} : {tn}\n" ++
    s!"    %istd{k} = stablehlo.rsqrt %ve{k} : {tn}\n" ++
    s!"    %xhat{k} = stablehlo.multiply %xc{k}, %istd{k} : {tn}\n" ++
    s!"    %gb{k} = stablehlo.broadcast_in_dim %g{k}, dims = [1] : ({cty}) -> {tn}\n" ++
    s!"    %btb{k} = stablehlo.broadcast_in_dim %bt{k}, dims = [1] : ({cty}) -> {tn}\n" ++
    s!"    %gx{k} = stablehlo.multiply %xhat{k}, %gb{k} : {tn}\n" ++
    s!"    %y{k} = stablehlo.add %gx{k}, %btb{k} : {tn}\n" ++
    s!"    %hn{k} = stablehlo.maximum %y{k}, %znk{k} : {tn}\n" ++
    s!"    %mask{k} = stablehlo.compare GT, %y{k}, %znk{k} : ({tn}, {tn}) -> {tni}\n" ++
    s!"    %h{k}4 = stablehlo.reshape %hn{k} : ({tn}) -> {t4}\n"
  -- BN backward for block k: grad w.r.t. post-relu (4D %dpost) → grad w.r.t. conv output %dco{k} (4D).
  let bnBack := fun (k C N H : Nat) (dpost : String) =>
    let t4 := s!"tensor<{bs}x{C}x{H}x{H}xf32>"
    let tn := s!"tensor<{bs}x{C}x{N}xf32>"
    let tc := s!"tensor<{bs}x{C}xf32>"
    let tni := s!"tensor<{bs}x{C}x{N}xi1>"
    s!"    %dpn{k} = stablehlo.reshape {dpost} : ({t4}) -> {tn}\n" ++
    s!"    %dy{k} = stablehlo.select %mask{k}, %dpn{k}, %znk{k} : {tni}, {tn}\n" ++
    s!"    %dxhat{k} = stablehlo.multiply %dy{k}, %gb{k} : {tn}\n" ++
    s!"    %ds1r{k} = stablehlo.reduce(%dxhat{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %ds1b{k} = stablehlo.broadcast_in_dim %ds1r{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %s1{k} = stablehlo.divide %ds1b{k}, %nfk{k} : {tn}\n" ++
    s!"    %dpr{k} = stablehlo.multiply %dxhat{k}, %xhat{k} : {tn}\n" ++
    s!"    %ds2r{k} = stablehlo.reduce(%dpr{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %ds2b{k} = stablehlo.broadcast_in_dim %ds2r{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %s2{k} = stablehlo.divide %ds2b{k}, %nfk{k} : {tn}\n" ++
    s!"    %xs2{k} = stablehlo.multiply %xhat{k}, %s2{k} : {tn}\n" ++
    s!"    %dsa{k} = stablehlo.subtract %dxhat{k}, %s1{k} : {tn}\n" ++
    s!"    %dsb{k} = stablehlo.subtract %dsa{k}, %xs2{k} : {tn}\n" ++
    s!"    %dxn{k} = stablehlo.multiply %istd{k}, %dsb{k} : {tn}\n" ++
    s!"    %dco{k} = stablehlo.reshape %dxn{k} : ({tn}) -> {t4}\n"
  -- conv input-VJP: transpose-(o,i) + spatial-reverse W{k} ([oC,iC,3,3]→[iC,oC,3,3]), conv with %dco{k}
  let convVjp := fun (k oC iC : Nat) (dco lhsTy outTy : String) (out : String) =>
    s!"    %wt{k} = stablehlo.transpose %W{k}, dims = [1, 0, 2, 3] : (tensor<{oC}x{iC}x3x3xf32>) -> tensor<{iC}x{oC}x3x3xf32>\n" ++
    s!"    %wr{k} = stablehlo.reverse %wt{k}, dims = [2, 3] : tensor<{iC}x{oC}x3x3xf32>\n" ++
    s!"    {out} = stablehlo.convolution({dco}, %wr{k})\n      {convCfg} : ({lhsTy}, tensor<{iC}x{oC}x3x3xf32>) -> {outTy}\n"
  let m32_4 := s!"tensor<{bs}x32x32x32xf32>"
  let m64_4 := s!"tensor<{bs}x64x16x16xf32>"
  let p32 := s!"tensor<{bs}x32x16x16xf32>"
  let p64 := s!"tensor<{bs}x64x8x8xf32>"
  let i4  := s!"tensor<{bs}x3x32x32xf32>"
  let f2  := s!"tensor<{bs}x4096xf32>"
  let h2  := s!"tensor<{bs}x512xf32>"
  let h2i := s!"tensor<{bs}x512xi1>"
  let o2  := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x3072xf32>"
  let rty := s!"tensor<{bs}xf32>"
  let poolFwd := fun (inTy outTy inp out : String) =>
    s!"    {out} = \"stablehlo.reduce_window\"({inp}, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({inTy}, tensor<f32>) -> {outTy}\n"
  let poolBack := fun (sfx : String) (srcTy gradTy inp grad out : String) =>
    s!"    {out} = \"stablehlo.select_and_scatter\"({inp}, {grad}, %zf) (\{\n" ++
    s!"      ^bb0(%sa{sfx}: tensor<f32>, %sb{sfx}: tensor<f32>):\n" ++
    s!"        %sge{sfx} = stablehlo.compare GE, %sa{sfx}, %sb{sfx} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    s!"        stablehlo.return %sge{sfx} : tensor<i1>\n" ++
    "    }, {\n" ++
    s!"      ^bb0(%sc{sfx}: tensor<f32>, %sd{sfx}: tensor<f32>):\n" ++
    s!"        %ss{sfx} = stablehlo.add %sc{sfx}, %sd{sfx} : tensor<f32>\n" ++
    s!"        stablehlo.return %ss{sfx} : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({srcTy}, {gradTy}, tensor<f32>) -> {srcTy}\n"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cifar_bn_pgd_step(%x: {bxd0}, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<32xf32>, %bt1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<32xf32>, %bt2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %g3: tensor<64xf32>, %bt3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %g4: tensor<64xf32>, %bt4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    -- forward: conv→BN→relu ×2, pool, ×2, pool, denses
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x3x3x3xf32>) -> {m32_4}\n" ++
    bnBlock 1 32 1024 32 "%c1" ++
    s!"    %c2 = stablehlo.convolution(%h14, %W2)\n      {convCfg} : ({m32_4}, tensor<32x32x3x3xf32>) -> {m32_4}\n" ++
    bnBlock 2 32 1024 32 "%c2" ++
    poolFwd m32_4 p32 "%h24" "%pool1" ++
    s!"    %c3 = stablehlo.convolution(%pool1, %W3)\n      {convCfg} : ({p32}, tensor<64x32x3x3xf32>) -> {m64_4}\n" ++
    bnBlock 3 64 256 16 "%c3" ++
    s!"    %c4 = stablehlo.convolution(%h34, %W4)\n      {convCfg} : ({m64_4}, tensor<64x64x3x3xf32>) -> {m64_4}\n" ++
    bnBlock 4 64 256 16 "%c4" ++
    poolFwd m64_4 p64 "%h44" "%pool2" ++
    s!"    %flat = stablehlo.reshape %pool2 : ({p64}) -> {f2}\n" ++
    s!"    %d5 = stablehlo.dot_general %flat, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<4096x512xf32>) -> {h2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z5 = stablehlo.add %d5, %b5b : {h2}\n" ++
    s!"    %h5 = stablehlo.maximum %z5, %zh : {h2}\n" ++
    s!"    %d6 = stablehlo.dot_general %h5, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z6 = stablehlo.add %d6, %b6b : {h2}\n" ++
    s!"    %h6 = stablehlo.maximum %z6, %zh : {h2}\n" ++
    s!"    %d7 = stablehlo.dot_general %h6, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b7b = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d7, %b7b : {o2}\n" ++
    -- softmax-CE
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- backward
    s!"    %dh6 = stablehlo.dot_general %g, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm6 = stablehlo.compare GT, %z6, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz6 = stablehlo.select %rm6, %dh6, %zh : {h2i}, {h2}\n" ++
    s!"    %dh5 = stablehlo.dot_general %dz6, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm5 = stablehlo.compare GT, %z5, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz5 = stablehlo.select %rm5, %dh5, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz5, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<4096x512xf32>) -> {f2}\n" ++
    s!"    %dpool2 = stablehlo.reshape %dflat : ({f2}) -> {p64}\n" ++
    poolBack "p2" m64_4 p64 "%h44" "%dpool2" "%dpre4" ++
    bnBack 4 64 256 16 "%dpre4" ++
    convVjp 4 64 64 "%dco4" m64_4 m64_4 "%dpre3" ++
    bnBack 3 64 256 16 "%dpre3" ++
    convVjp 3 64 32 "%dco3" m64_4 p32 "%dpool1" ++
    poolBack "p1" m32_4 p32 "%h24" "%dpool1" "%dpre2" ++
    bnBack 2 32 1024 32 "%dpre2" ++
    convVjp 2 32 32 "%dco2" m32_4 m32_4 "%dpre1" ++
    bnBack 1 32 1024 32 "%dpre1" ++
    convVjp 1 32 3 "%dco1" m32_4 i4 "%dxi" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxnv = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxnv, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD attack on the verified MNIST MLP** (`planning/robustness.md`). Trains the
    784→512→512→10 ReLU MLP on the proof-rendered SGD step, then attacks through IREE with the
    proven `mlpInputGrad` VJP kernel. The Lipschitz certificate is the **product** of the three
    layers' spectral norms — where the bound (and so the cert) goes loose. -/
def VerifiedNet.attackPgdMlp (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let hN := 512
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let nP := net.nParams
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut theta := F32.concat parts
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for _ in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
      theta := out.extract 0 (nP * 4)
  -- split θ (func-arg order: W0 b0 W1 b1 W2 b2)
  let W0 := theta.extract 0 (d0*hN*4)
  let W1 := theta.extract ((d0*hN + hN)*4) ((d0*hN + hN + hN*hN)*4)
  let W2 := theta.extract ((d0*hN + hN + hN*hN + hN)*4) ((d0*hN + hN + hN*hN + hN + hN*d1)*4)
  let mut clean := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
        clean := clean + 1
  IO.println s!"clean test acc = {clean}/{nEval} = {clean.toFloat/nEval.toFloat*100.0}%"
  let K := 40
  let pgdShapes := packShapes #[#[d0,hN], #[hN], #[hN,hN], #[hN], #[hN,d1], #[d1], #[bs,d1], #[bs,d0]]
  let runSweep := fun (linf : Bool) (epsList : List Float) => do
    for eps in epsList do
      let alpha := 2.5 * eps / K.toFloat
      IO.FS.writeFile ".lake/build/mlp_pgd_step.mlir" (genMlpPgdStep bs d0 hN d1 eps alpha linf)
      compileVmfb ".lake/build/mlp_pgd_step.mlir" ".lake/build/mlp_pgd_step.vmfb"
      let pgdSess ← LowererSession.create ".lake/build/mlp_pgd_step.vmfb"
      let mut correct := 0
      for bi in [0:nbt] do
        let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
        let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
        let pgdParams := F32.concat #[theta, oh, x0]
        let mut x := x0
        for _ in [0:K] do
          x ← LowererSession.forwardF32 pgdSess "m.mlp_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
        for j in [0:min bs (nEval - bi * bs)] do
          if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
            correct := correct + 1
      let lbl := if linf then "L∞" else "L2"
      IO.println s!"{lbl} PGD eps={eps}: adv acc = {correct.toFloat/nEval.toFloat*100.0}%"
  runSweep true [0.1, 0.2, 0.3]
  -- certificate: product of the three layers' spectral norms (ReLU is 1-Lipschitz)
  let L0 := specNormW W0 d0 hN
  let L1 := specNormW W1 hN hN
  let L2 := specNormW W2 hN d1
  let L := L0 * L1 * L2
  IO.println s!"\nspectral norms ‖W₀‖={L0}, ‖W₁‖={L1}, ‖W₂‖={L2}  →  global L = {L}  (PRODUCT over 3 layers — loose)"
  let tot := nEval.toFloat
  let mut cert05 := 0
  let mut cert10 := 0
  let mut cert15 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      let mut top := -1.0e30
      let mut sec := -1.0e30
      let mut topi := 0
      for c in [0:d1] do
        let v := F32.read logits (j * d1 + c).toUSize
        if v > top then
          sec := top
          top := v
          topi := c
        else if v > sec then
          sec := v
      if topi == F32.readLabel evalLbl (bi * bs + j) then
        let r := (top - sec) / (1.4142135623730951 * L)
        if r ≥ 0.5 then cert05 := cert05 + 1
        if r ≥ 1.0 then cert10 := cert10 + 1
        if r ≥ 1.5 then cert15 := cert15 + 1
  IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  runSweep false [0.5, 1.0, 1.5]
  IO.println "done (phase-3 MLP PGD: input gradient = the proven mlpInputGrad VJP via IREE)."

/-- **Spectral-norm-constrained training of the verified MNIST MLP** (`planning/robustness_ladder.md`,
    the research lever). Trains the 784→512→512→10 net with **projected SGD onto the spectral ball**
    — after every `K` proof-rendered steps (and once at the end) each weight `Wᵢ` is rescaled to
    `‖Wᵢ‖₂ ≤ c` (`projectSpectral`) — then runs the *same* `cert ≤ TRUE ≤ PGD` sandwich. Sweeps a
    few caps `c` (plus an unconstrained baseline) so the table shows the trade: shrinking `c` pulls
    the global `L = ∏‖Wᵢ‖₂` down (`L ≤ c³`), turning the **vacuous** product certificate
    **non-vacuous** — at the cost of clean accuracy. The empirical face of
    `lipschitz_margin_certified_radius` (`LeanMlir/Proofs/Certificates/LipschitzCert.lean`): smaller `L` ⇒ larger
    certified radius `m/(√2·L)`. The verified CE gradient stays in the proven kernel; the projection
    is host-side weight rescaling only. -/
def VerifiedNet.attackPgdSpectralMlp (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let hN := 512
  let d1 := net.nClasses
  let projEvery := 20            -- lazy projection: every 20 verified steps (+ once at the end)
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let K := 40
  let pgdShapes := packShapes #[#[d0,hN], #[hN], #[hN,hN], #[hN], #[hN,d1], #[d1], #[bs,d1], #[bs,d0]]
  -- run one PGD eps point, returning adversarial accuracy (%) on the verified net
  let pgdAcc := fun (theta : ByteArray) (linf : Bool) (eps : Float) => do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/mlp_pgd_step.mlir" (genMlpPgdStep bs d0 hN d1 eps alpha linf)
    compileVmfb ".lake/build/mlp_pgd_step.mlir" ".lake/build/mlp_pgd_step.vmfb"
    let pgdSess ← LowererSession.create ".lake/build/mlp_pgd_step.vmfb"
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[theta, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess "m.mlp_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    pure (correct.toFloat / nEval.toFloat * 100.0)
  let tot := nEval.toFloat
  let mut rows : Array String := #[]
  for cap in caps do
    let capStr := if cap ≥ 1.0e8 then "∞ (none)" else toString cap
    IO.println s!"\n── cap c = {capStr} ──"
    -- fresh He-init (same seeds ⇒ fair comparison across caps)
    let mut parts : Array ByteArray := #[]
    let mut seed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam seed spec.1 spec.2)
      seed := seed + 1
    let mut theta := F32.concat parts
    let mut step := 0
    for _ in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xb := F32.sliceImages trainImg (bi * bs) bs d0
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        theta ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
        step := step + 1
        if cap < 1.0e8 && step % projEvery == 0 then
          theta ← projectSpectral theta net.specs cap
    if cap < 1.0e8 then theta ← projectSpectral theta net.specs cap   -- enforce the cap on the final θ
    -- split θ for the certificate
    let W0 := theta.extract 0 (d0*hN*4)
    let W1 := theta.extract ((d0*hN + hN)*4) ((d0*hN + hN + hN*hN)*4)
    let W2 := theta.extract ((d0*hN + hN + hN*hN + hN)*4) ((d0*hN + hN + hN*hN + hN + hN*d1)*4)
    let L0 := specNormW W0 d0 hN
    let L1 := specNormW W1 hN hN
    let L2 := specNormW W2 hN d1
    let L := L0 * L1 * L2
    -- clean accuracy + certified-robust accuracy at L2 {0.25, 0.5, 1.0}
    let mut clean := 0
    let mut c025 := 0
    let mut c05 := 0
    let mut c10 := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for cidx in [0:d1] do
          let v := F32.read logits (j * d1 + cidx).toUSize
          if v > top then sec := top; top := v; topi := cidx
          else if v > sec then sec := v
        if topi == F32.readLabel evalLbl (bi * bs + j) then
          clean := clean + 1
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.25 then c025 := c025 + 1
          if r ≥ 0.5 then c05 := c05 + 1
          if r ≥ 1.0 then c10 := c10 + 1
    let cleanPct := clean.toFloat/tot*100.0
    IO.println s!"  ‖W₀‖={L0}  ‖W₁‖={L1}  ‖W₂‖={L2}  →  L = {L}"
    IO.println s!"  clean = {cleanPct}%   cert@L2 0.25/0.5/1.0 = {c025.toFloat/tot*100.0}% / {c05.toFloat/tot*100.0}% / {c10.toFloat/tot*100.0}%"
    let pinf ← pgdAcc theta true 0.1
    let pl2 ← pgdAcc theta false 0.5
    IO.println s!"  L∞ PGD ε=0.1 = {pinf}%   L2 PGD ε=0.5 = {pl2}%"
    (← IO.getStdout).flush
    rows := rows.push s!"  {capStr}\t{cleanPct}\t{L}\t{c05.toFloat/tot*100.0}\t{pl2}\t{pinf}"
  IO.println "\n══ spectral-norm training: the cert ≤ TRUE ≤ PGD trade ══"
  IO.println "  cap c\tclean%\tglobal L\tcert@L2 0.5\tL2 PGD 0.5\tL∞ PGD 0.1"
  for row in rows do IO.println row
  IO.println "\ndone (spectral-norm-constrained training: smaller c ⇒ smaller L ⇒ the product cert"
  IO.println "      goes non-vacuous, at the cost of clean accuracy — the gap-shrinking lever)."

/-- **Generic conv-net PGD attack** (`planning/robustness_ladder.md`). Trains any packed conv
    net on its proof-rendered SGD step, then attacks through IREE with `genKernel` — the full
    proven backward (conv input-VJPs + maxpool `select_and_scatter`-backs, mirroring the net's
    `<slug>_train_step.mlir`) run to `dx`. Certificate = the conv-aware spectral-norm **product**
    (`specNormConvTapSum` for convs × `specNormW` for denses; ReLU/maxpool are 1-Lipschitz) —
    astronomically loose, the depth-cliff. `genKernel` and `net.slug` select the architecture
    (`genCnnPgdStep`/MNIST-CNN, `genCifarPgdStep`/CIFAR-CNN). -/
def VerifiedNet.attackPgdConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (genKernel : Nat → Float → Float → Bool → String) (withCert : Bool := true) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut theta := F32.concat parts
  -- Best-checkpoint training: eval each epoch and keep the highest-accuracy θ. Plain SGD on the
  -- deeper nets (CIFAR) can diverge late; attacking the best checkpoint keeps the demo robust
  -- (and the cert finite). Monotone nets (MNIST CNN) → best = final, so numbers are unchanged.
  let mut bestTheta := theta
  let mut bestAcc := -1.0
  let evalAcc := fun (th : ByteArray) => do
    let mut c := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn th shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          c := c + 1
    pure (c.toFloat / nEval.toFloat * 100.0)
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      theta ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
    let acc ← evalAcc theta
    if acc > bestAcc then bestAcc := acc; bestTheta := theta
    IO.println s!"    epoch {ep + 1}/{cfg.epochs}: acc = {acc}%"
    (← IO.getStdout).flush
  theta := bestTheta                       -- attack the best checkpoint
  IO.println s!"clean test acc (best epoch) = {bestAcc}%"
  let K := 40
  let pgdShapes := packShapes (net.paramShapes ++ #[#[bs, d1], #[bs, d0]])
  let runSweep := fun (linf : Bool) (epsList : List Float) => do
    for eps in epsList do
      let alpha := 2.5 * eps / K.toFloat
      IO.FS.writeFile s!".lake/build/{net.slug}_pgd_step.mlir" (genKernel bs eps alpha linf)
      compileVmfb s!".lake/build/{net.slug}_pgd_step.mlir" s!".lake/build/{net.slug}_pgd_step.vmfb"
      let pgdSess ← LowererSession.create s!".lake/build/{net.slug}_pgd_step.vmfb"
      let mut correct := 0
      for bi in [0:nbt] do
        let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
        let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
        let pgdParams := F32.concat #[theta, oh, x0]
        let mut x := x0
        for _ in [0:K] do
          x ← LowererSession.forwardF32 pgdSess s!"m.{net.slug}_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
        for j in [0:min bs (nEval - bi * bs)] do
          if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
            correct := correct + 1
      let lbl := if linf then "L∞" else "L2"
      IO.println s!"{lbl} PGD eps={eps}: adv acc = {correct.toFloat/nEval.toFloat*100.0}%"
      (← IO.getStdout).flush
  runSweep true [0.1, 0.2, 0.3]
  -- ── certificate: conv-aware spectral-norm PRODUCT (ReLU/maxpool are 1-Lipschitz) ──
  -- Skipped for BN nets: instance norm absorbs the conv weight scale and its Lipschitz is
  -- data-dependent (γ·istd), so the conv-product cert is meaningless — a separate problem.
  if withCert then
    let mut L := 1.0
    let mut off := 0
    let mut msg := ""
    for spec in net.specs do
      let dims := spec.1
      let len := dims.foldl (·*·) 1
      let wslice := theta.extract (off*4) ((off+len)*4)
      if dims.size == 4 then
        let n := specNormConvTapSum wslice dims[0]! dims[1]! dims[2]! dims[3]!
        L := L * n
        msg := msg ++ s!"conv{dims[1]!}→{dims[0]!} Σtap‖·‖₂={n}  "
      else if dims.size == 2 then
        let n := specNormW wslice dims[0]! dims[1]!
        L := L * n
        msg := msg ++ s!"dense{dims[0]!}→{dims[1]!} ‖·‖₂={n}  "
      off := off + len
    IO.println s!"\nlayer norms: {msg}"
    IO.println s!"  →  global L = {L}  (PRODUCT over conv+dense layers — astronomically loose)"
    let tot := nEval.toFloat
    let mut cert05 := 0
    let mut cert10 := 0
    let mut cert15 := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for c in [0:d1] do
          let v := F32.read logits (j * d1 + c).toUSize
          if v > top then
            sec := top
            top := v
            topi := c
          else if v > sec then
            sec := v
        if topi == F32.readLabel evalLbl (bi * bs + j) then
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.5 then cert05 := cert05 + 1
          if r ≥ 1.0 then cert10 := cert10 + 1
          if r ≥ 1.5 then cert15 := cert15 + 1
    IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  else
    IO.println "\n(certificate N/A — instance-norm Lipschitz is data-dependent (γ·istd); deferred)"
  runSweep false [0.5, 1.0, 1.5]
  IO.println s!"done (phase-3 {net.name} PGD: input gradient = the proven conv/maxpool input-VJP via IREE)."

/-- PGD attack on the verified MNIST CNN (the first conv rung). -/
def VerifiedNet.attackPgdCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCnnPgdStep

/-- PGD attack on the verified CIFAR-10 CNN (the deeper conv rung: 4 conv + 2 pool + 3 dense). -/
def VerifiedNet.attackPgdCifar (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCifarPgdStep

/-- PGD attack on the verified CIFAR-10 CNN **+ per-channel (instance) BatchNorm** (`cifar_bn`).
    The BN input-VJP rung — `genCifarBnPgdStep` runs the proven backward through 4 instance-norm
    layers (the BN grad-input 3-term formula). Certificate skipped (`withCert := false`): instance
    norm's Lipschitz is data-dependent (`γ·istd`), a separate problem from the conv-product. -/
def VerifiedNet.attackPgdCifarBn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCifarBnPgdStep (withCert := false)

/-- **Spectral-norm-constrained training of the verified MNIST CNN** (`planning/robustness_ladder.md`,
    the gap-shrinking lever applied to the conv net). The CNN sibling of `attackPgdSpectralMlp`:
    projected SGD onto the spectral ball — every `K` proof-rendered steps (and once at the end)
    `projectSpectral` caps **both** the dense `‖Wᵢ‖₂` and the conv tap-sum bound at `c` — then the
    `cert ≤ TRUE ≤ PGD` sandwich (PGD via `genKernel`, cert = the conv-aware product). Harder than
    the MLP: it's a `k`-layer product (`L ≤ cᵏ`) and the conv tap-sum is a *loose* bound, so
    projection over-penalizes the convs — the cert needs a tighter `c` (and pays more clean accuracy)
    than the MLP did, and certifies only at *smaller* radii. The honest "depth + loose conv-norm ⇒
    certifying the conv net is harder." Generic over `genKernel`/`net.slug` (MNIST-CNN, CIFAR-CNN). -/
def VerifiedNet.attackPgdSpectralConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) (genKernel : Nat → Float → Float → Bool → String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  let projEvery := 20
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let K := 40
  let pgdShapes := packShapes (net.paramShapes ++ #[#[bs, d1], #[bs, d0]])
  let pgdAcc := fun (theta : ByteArray) (linf : Bool) (eps : Float) => do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile s!".lake/build/{net.slug}_pgd_step.mlir" (genKernel bs eps alpha linf)
    compileVmfb s!".lake/build/{net.slug}_pgd_step.mlir" s!".lake/build/{net.slug}_pgd_step.vmfb"
    let pgdSess ← LowererSession.create s!".lake/build/{net.slug}_pgd_step.vmfb"
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[theta, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess s!"m.{net.slug}_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    pure (correct.toFloat / nEval.toFloat * 100.0)
  let tot := nEval.toFloat
  let mut rows : Array String := #[]
  for cap in caps do
    let capStr := if cap ≥ 1.0e8 then "∞ (none)" else toString cap
    IO.println s!"\n── cap c = {capStr} ──"
    let mut parts : Array ByteArray := #[]
    let mut seed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam seed spec.1 spec.2)
      seed := seed + 1
    let mut theta := F32.concat parts
    let mut bestTheta := theta
    let mut bestAcc := -1.0
    let mut step := 0
    for _ in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xb := F32.sliceImages trainImg (bi * bs) bs d0
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        theta ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
        step := step + 1
        if cap < 1.0e8 && step % projEvery == 0 then
          theta ← projectSpectral theta net.specs cap
      -- best-checkpoint (the baseline ∞ cap can diverge late; constrained caps stay bounded)
      let mut c := 0
      for bi in [0:nbt] do
        let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
        for j in [0:min bs (nEval - bi * bs)] do
          if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
            c := c + 1
      let acc := c.toFloat / nEval.toFloat * 100.0
      if acc > bestAcc then bestAcc := acc; bestTheta := theta
    theta := bestTheta
    if cap < 1.0e8 then theta ← projectSpectral theta net.specs cap
    -- conv-aware certificate product (specNormConvTapSum convs × specNormW denses)
    let mut L := 1.0
    let mut off := 0
    let mut msg := ""
    for spec in net.specs do
      let dims := spec.1
      let len := dims.foldl (·*·) 1
      let wslice := theta.extract (off*4) ((off+len)*4)
      if dims.size == 4 then
        let n := specNormConvTapSum wslice dims[0]! dims[1]! dims[2]! dims[3]!
        L := L * n; msg := msg ++ s!"cv={n} "
      else if dims.size == 2 then
        let n := specNormW wslice dims[0]! dims[1]!
        L := L * n; msg := msg ++ s!"de={n} "
      off := off + len
    let mut clean := 0
    let mut cR1 := 0      -- certified @ L2 0.1
    let mut cR2 := 0      -- certified @ L2 0.25  (the CNN's visible band — it certifies at small radii)
    let mut cR3 := 0      -- certified @ L2 0.5
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for cidx in [0:d1] do
          let v := F32.read logits (j * d1 + cidx).toUSize
          if v > top then sec := top; top := v; topi := cidx
          else if v > sec then sec := v
        if topi == F32.readLabel evalLbl (bi * bs + j) then
          clean := clean + 1
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.1 then cR1 := cR1 + 1
          if r ≥ 0.25 then cR2 := cR2 + 1
          if r ≥ 0.5 then cR3 := cR3 + 1
    let cleanPct := clean.toFloat/tot*100.0
    IO.println s!"  {msg} →  L = {L}"
    IO.println s!"  clean = {cleanPct}%   cert@L2 0.1/0.25/0.5 = {cR1.toFloat/tot*100.0}% / {cR2.toFloat/tot*100.0}% / {cR3.toFloat/tot*100.0}%"
    let pinf ← pgdAcc theta true 0.1
    let pl2 ← pgdAcc theta false 0.5
    IO.println s!"  L∞ PGD ε=0.1 = {pinf}%   L2 PGD ε=0.5 = {pl2}%"
    (← IO.getStdout).flush
    rows := rows.push s!"  {capStr}\t{cleanPct}\t{L}\t{cR2.toFloat/tot*100.0}\t{pl2}\t{pinf}"
  IO.println s!"\n══ spectral-norm training ({net.name}): the cert ≤ TRUE ≤ PGD trade ══"
  IO.println "  cap c\tclean%\tglobal L\tcert@L2 0.25\tL2 PGD 0.5\tL∞ PGD 0.1"
  for row in rows do IO.println row
  IO.println "\ndone (spectral-norm-constrained conv training: the k-layer product + loose conv tap-sum"
  IO.println "      make certifying the conv net harder than the MLP — tighter c, more clean cost)."

/-- Spectral-norm-constrained training of the verified MNIST CNN. -/
def VerifiedNet.attackPgdSpectralCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  net.attackPgdSpectralConvNet cfg dataDir caps genCnnPgdStep

/-- Spectral-norm-constrained training of the verified CIFAR-10 CNN (7-layer product). -/
def VerifiedNet.attackPgdSpectralCifar (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  net.attackPgdSpectralConvNet cfg dataDir caps genCifarPgdStep

/-! ## Randomized-smoothing statistics (Cohen–Rosenfeld–Kolter 2019)

The pieces the smoothing certificate needs, in pure `Float` (no kernel, no Mathlib): the
probit `Φ⁻¹` for the radius `σ·Φ⁻¹(p_A)`, and a **sound** Clopper–Pearson lower confidence
bound on `p_A` (a genuine 1−α lower bound, not an approximation — a certificate must under-
estimate). CP is built bottom-up from the regularized incomplete beta `Iₓ(a,b)`. -/

/-- Inverse standard-normal CDF `Φ⁻¹` (probit), Peter Acklam's rational approximation
    (relative error < 1.15e-9 over `(0,1)`). `p_A` here lives in `(0.5, ~0.99)`, far from
    the tails, so this is orders of magnitude tighter than the Monte-Carlo sampling error. -/
private def invNormCdf (p : Float) : Float := Id.run do
  let a0 := -3.969683028665376e+01
  let a1 :=  2.209460984245205e+02
  let a2 := -2.759285104469687e+02
  let a3 :=  1.383577518672690e+02
  let a4 := -3.066479806614716e+01
  let a5 :=  2.506628277459239e+00
  let b1 := -5.447609879822406e+01
  let b2 :=  1.615858368580409e+02
  let b3 := -1.556989798598866e+02
  let b4 :=  6.680131188771972e+01
  let b5 := -1.328068155288572e+01
  let c0 := -7.784894002430293e-03
  let c1 := -3.223964580411365e-01
  let c2 := -2.400758277161838e+00
  let c3 := -2.549732539343734e+00
  let c4 :=  4.374664141464968e+00
  let c5 :=  2.938163982698783e+00
  let d1 :=  7.784695709041462e-03
  let d2 :=  3.224671290700398e-01
  let d3 :=  2.445134137142996e+00
  let d4 :=  3.754408661907416e+00
  let plow := 0.02425
  let phigh := 1.0 - plow
  if p < plow then
    let q := Float.sqrt (-2.0 * Float.log p)
    return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)
  else if p ≤ phigh then
    let q := p - 0.5
    let r := q*q
    return (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0)
  else
    let q := Float.sqrt (-2.0 * Float.log (1.0 - p))
    return 0.0 - (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)

/-- `log Γ(x)` for `x ≥ 0.5` (the only regime we hit: `a=k≥1`, `b=n−k+1≥1`), Lanczos `g=7`. -/
private def lgammaF (x : Float) : Float := Id.run do
  let g : Array Float := #[
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
  let xx := x - 1.0
  let mut a := g[0]!
  let t := xx + 7.5
  for i in [1:9] do
    a := a + g[i]! / (xx + i.toFloat)
  return 0.5 * Float.log (2.0 * 3.141592653589793) + (xx + 0.5) * Float.log t - t + Float.log a

/-- Lentz continued fraction for the incomplete beta (Numerical Recipes `betacf`). -/
private def betacf (a b x : Float) : Float := Id.run do
  let fpmin := 1.0e-300; let eps := 3.0e-12
  let qab := a + b; let qap := a + 1.0; let qam := a - 1.0
  let mut c := 1.0
  let mut d := 1.0 - qab * x / qap
  if Float.abs d < fpmin then d := fpmin
  d := 1.0 / d
  let mut h := d
  for m in [1:201] do
    let mf := m.toFloat; let m2 := 2.0 * mf
    let aa1 := mf * (b - mf) * x / ((qam + m2) * (a + m2))
    d := 1.0 + aa1 * d; if Float.abs d < fpmin then d := fpmin
    c := 1.0 + aa1 / c; if Float.abs c < fpmin then c := fpmin
    d := 1.0 / d; h := h * d * c
    let aa2 := -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
    d := 1.0 + aa2 * d; if Float.abs d < fpmin then d := fpmin
    c := 1.0 + aa2 / c; if Float.abs c < fpmin then c := fpmin
    d := 1.0 / d
    let del := d * c
    h := h * del
    if Float.abs (del - 1.0) < eps then break
  return h

/-- Regularized incomplete beta `Iₓ(a,b)` (Numerical Recipes `betai`); increasing in `x`. -/
private def betaiF (a b x : Float) : Float :=
  if x ≤ 0.0 then 0.0
  else if x ≥ 1.0 then 1.0
  else
    let lbt := lgammaF (a+b) - lgammaF a - lgammaF b + a * Float.log x + b * Float.log (1.0 - x)
    let bt := Float.exp lbt
    if x < (a + 1.0) / (a + b + 2.0) then bt * betacf a b x / a
    else 1.0 - bt * betacf b a (1.0 - x) / b

/-- Clopper–Pearson **exact** lower confidence bound for a Binomial proportion: the largest
    `p` with `P[Bin(n,p) ≥ k] ≤ α`, i.e. the `α`-quantile of `Beta(k, n−k+1)` — the `p` solving
    `I_p(k, n−k+1) = α`, found by bisection (`Iₓ` is monotone). The SOUND `1−α` lower bound on
    `p_A` that the certified radius `σ·Φ⁻¹(p_A)` rests on (Cohen 2019 uses the same CP bound).
    `k=0 ⇒ 0`. -/
private def clopperPearsonLower (k n : Nat) (alpha : Float) : Float := Id.run do
  if k == 0 then return 0.0
  let a := k.toFloat
  let b := (n - k + 1).toFloat
  let mut lo := 0.0
  let mut hi := 1.0
  for _ in [0:60] do
    let mid := 0.5 * (lo + hi)
    if betaiF a b mid < alpha then lo := mid else hi := mid
  return 0.5 * (lo + hi)

/-- **Randomized-smoothing certificate** (Cohen–Rosenfeld–Kolter 2019, `planning/robustness_ladder.md`
    §3) — the depth-INDEPENDENT cert, and the answer where the Lipschitz product is hopeless.
    The smoothed classifier `ĝ(x) = argmax_c P[f(x+η)=c]`, `η ~ N(0,σ²I)`, is certified robust at
    L2 radius `σ·Φ⁻¹(p_A)` where `p_A` is a lower bound on the top class's noise probability. It's
    **forward-only**: no new kernel, no input-VJP — just sample `n` noisy copies, run the existing
    proof-rendered `<slug>_fwd`, count argmax votes, Clopper–Pearson lower-bound `p_A`. The base
    classifier is trained with matched Gaussian augmentation (every batch corrupted with `N(0,σ²I)`
    host-side before the proof-rendered SGD step — the forward/backward graph is untouched), the
    Cohen recipe. Architecture-agnostic + depth-independent, so it certifies a *non-vacuous* radius
    on the very nets (CIFAR, deep) where `∏‖Wᵢ‖₂` is astronomically loose. Generic over any
    `VerifiedNet` (fwd + train-step only).

    `n` (`SMOOTH_N`, default 10000 — Cohen's large-`n` regime) is the estimation budget and the only
    honest tightening lever: the per-point radius is capped at `σ·Φ⁻¹(α^(1/n))` (a unanimous vote
    still only certifies `p_A ≤ α^(1/n)`), so larger `n` lifts the ceiling and tightens the CP bound
    toward the true noise-probability — bigger certified radii at the same `1−α` guarantee. -/
def VerifiedNet.smoothCertify (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (sigmas : List Float) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Randomized-smoothing certificate on {net.name} (verified codegen → IREE → GPU, forward-only)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  -- `trainPix`/`crop`: Imagenette train ships at 256² and is center-cropped to 224² per batch
  -- (the val/eval split is already 224² = d0, so certify reads it directly). For MNIST/CIFAR
  -- crop=false and trainPix=d0, so the crop below is a no-op.
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  -- knobs (env-overridable; defaults are Cohen's large-n estimation regime — drop SMOOTH_N /
  -- SMOOTH_MAXCERT for a cheap smoke).
  let n0      := ((← IO.getEnv "SMOOTH_N0").bind (·.toNat?)).getD 100
  let nSamp   := ((← IO.getEnv "SMOOTH_N").bind (·.toNat?)).getD 10000
  let maxCert := ((← IO.getEnv "SMOOTH_MAXCERT").bind (·.toNat?)).getD 200
  -- single-σ override (millis, same knob as the ConvNeXt exe) — e.g. the
  -- fixed-protocol scorecard runs σ=0.5 only via SMOOTH_SIGMA_MILLI=500
  let sigmas := match (← IO.getEnv "SMOOTH_SIGMA_MILLI") with
    | some s => match s.toNat? with
      | some m => [m.toFloat / 1000.0]
      | none => sigmas
    | none => sigmas
  -- per-epoch best-θ eval is a clean-acc proxy; cap it to a subsample on heavy 224² nets
  -- (`SMOOTH_EVAL_BATCHES`, default = full val set).
  let evalBatches := min nbt (((← IO.getEnv "SMOOTH_EVAL_BATCHES").bind (·.toNat?)).getD nbt)
  let alpha   := 0.001
  let radii : Array Float := #[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  let nCert   := min maxCert nEval
  -- SMOOTH_STRIDE=1 + SMOOTH_MAXCERT=100 = the fixed-first-100 scorecard
  -- protocol (mirrors the Lipschitz scorecard subset); default unchanged.
  let stride  := max 1 (((← IO.getEnv "SMOOTH_STRIDE").bind (·.toNat?)).getD (nEval / nCert))
  -- the rendered fwd has a STATIC batch = bs, so SampleUnderNoise feeds whole `bs`-batches;
  -- round the requested sample counts up to a multiple of bs (the effective n the CP bound uses).
  let n0Batches := max 1 ((n0 + bs - 1) / bs)
  let nBatches  := max 1 ((nSamp + bs - 1) / bs)
  let n0Eff := n0Batches * bs
  let nEff  := nBatches * bs
  -- the n→radius ceiling: even a UNANIMOUS vote gives only p_A ≤ α^(1/n) (the CP bound at n_A=n),
  -- so EVERY point's radius is capped at σ·Φ⁻¹(α^(1/n)) regardless of how robust it truly is.
  -- Larger n ⇒ p_A closer to the true noise-prob ⇒ higher ceiling — the only honest tightening
  -- lever (shrinking α would just weaken the 1−α guarantee, not tighten the estimate).
  let pMax := clopperPearsonLower nEff nEff alpha
  IO.println s!"  n0={n0Eff} (select)  n={nEff} (estimate)  α={alpha}  certifying {nCert} test imgs (every {stride}th)"
  IO.println s!"  p_A ceiling = {pMax} (= α^(1/n))  →  max certifiable radius = σ · {invNormCdf pMax}"
  let mut rows : Array String := #[]
  -- per-image certified radius, dumped to CSV for arbitrarily-fine frontier curves (cert-acc at
  -- radius r = fraction with correct ∧ radius ≥ r) + ACR + any threshold, all from one run.
  -- count,n = the raw CP inputs (k successes of n samples) — what the Lean
  -- kernel tail checks (SmoothingCP.binomTail_le_of_kernel_check) consume.
  let mut dumpRows : Array String := #["sigma,img_idx,label,pred,abstain,radius,count,n"]
  -- Lipschitz-hypothesis probe (SMOOTH_LIP_PROBE): measures |Φ⁻¹(p_ĉ(x+δ))−Φ⁻¹(p_ĉ(x))| vs the
  -- proven bound ‖δ‖/σ — the empirical grounding of `smoothing_certified_radius`'s (1/σ)-Lipschitz hyp.
  let lipProbe := (← IO.getEnv "SMOOTH_LIP_PROBE").isSome
  let mut lipRows : Array String := #["sigma,img_idx,delta,dg,bound,ratio"]
  let mut lipMax := 0.0
  let mut lipViol := 0
  let mut lipN := 0
  for sigma in sigmas do
    IO.println s!"\n── σ = {sigma}  (radius ceiling at this n = {sigma * invNormCdf pMax}) ──"
    -- (1) train a Gaussian-noise-augmented base classifier (the Cohen recipe): every batch is
    --     corrupted with N(0,σ²I) host-side before the proof-rendered SGD step — graph unchanged.
    --     Best-θ checkpoint on clean eval acc.
    let mut parts : Array ByteArray := #[]
    let mut iseed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam iseed spec.1 spec.2); iseed := iseed + 1
    let mut theta := F32.concat parts
    let mut bestTheta := theta
    let mut bestAcc := -1.0
    let mut nseed : USize := 1
    for ep in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xbRaw := F32.sliceImages trainImg (bi * bs) bs trainPix
        let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        let xbN ← F32.addGaussianTiled xb 0 (bs*d0).toUSize 1 sigma nseed
        nseed := nseed + 1
        theta ← LowererSession.mlpTrainStepV tsSess tsFn xbN theta shapes yb bs.toUSize d0.toUSize d1.toUSize
      let mut c := 0
      for bi in [0:evalBatches] do
        let xb := F32.sliceImages evalImg (bi * bs) bs d0
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb (net.xShape bs) bs.toUSize d1.toUSize
        for j in [0:bs] do
          if (F32.argmaxN logits (j*d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi*bs+j) then c := c+1
      let acc := c.toFloat/(evalBatches*bs).toFloat*100.0
      if acc > bestAcc then bestAcc := acc; bestTheta := theta
      IO.println s!"    epoch {ep+1}/{cfg.epochs}: clean acc = {acc}%"
      (← IO.getStdout).flush
    let th := bestTheta                       -- certify the best checkpoint (immutable snapshot)
    IO.println s!"  noise-trained clean acc = {bestAcc}%"
    -- (2) SampleUnderNoise: `nBatches·bs` noisy copies of ONE image (element offset `off`), each
    --     forward feeding exactly `bs` rows (the rendered fwd's static batch), argmax-vote → counts.
    let sampleCountsB := fun (base : ByteArray) (off nBat : Nat) (sd : USize) => do
      let mut counts : Array Nat := Array.replicate d1 0
      for ci in [0:nBat] do
        let xN ← F32.addGaussianTiled base off.toUSize d0.toUSize bs.toUSize sigma (sd + ci.toUSize)
        let logits ← LowererSession.forwardF32 fwdSess fwdFn th shapes xN (net.xShape bs) bs.toUSize d1.toUSize
        for j in [0:bs] do
          let a := (F32.argmaxN logits (j*d1).toUSize d1.toUSize).toNat
          counts := counts.set! a (counts[a]! + 1)
      pure counts
    let sampleCounts := fun (off nBat : Nat) (sd : USize) => sampleCountsB evalImg off nBat sd
    -- (3) certify each sampled test image: n0 select ĉ_A, n estimate p_A, radius σ·Φ⁻¹(p_A).
    let mut abstain := 0
    let mut natCorrect := 0
    let mut acr := 0.0
    let mut certCnt : Array Nat := Array.replicate radii.size 0
    for t in [0:nCert] do
      let imgIdx := t * stride
      let off := imgIdx * d0
      let label := F32.readLabel evalLbl imgIdx
      let base : USize := (imgIdx + 1).toUSize * 131 + 1
      let counts0 ← sampleCounts off n0Batches base
      let mut cHatA := 0
      for c in [1:d1] do
        if counts0[c]! > counts0[cHatA]! then cHatA := c
      if cHatA == label then natCorrect := natCorrect + 1
      let counts ← sampleCounts off nBatches (base + 524287)
      let pA := clopperPearsonLower counts[cHatA]! nEff alpha
      let certified := pA > 0.5
      let radius := if certified then sigma * invNormCdf pA else 0.0
      if certified then
        if cHatA == label then
          acr := acr + radius
          for ri in [0:radii.size] do
            if radius ≥ radii[ri]! then certCnt := certCnt.set! ri (certCnt[ri]! + 1)
      else
        abstain := abstain + 1
      dumpRows := dumpRows.push
        s!"{sigma},{imgIdx},{label},{cHatA},{if certified then 0 else 1},{radius},{counts[cHatA]!},{nEff}"
      if (t+1) % 100 == 0 then
        IO.println s!"    certified {t+1}/{nCert} ..."; (← IO.getStdout).flush
    -- (4) Lipschitz-hypothesis probe: shift x by r·(unit vector) and compare the probit-score change
    --     |Φ⁻¹(p_ĉ(x+δ))−Φ⁻¹(p_ĉ(x))| to the PROVEN bound ‖δ‖/σ (Salman et al. 2019 Lemma 2 — the
    --     hypothesis `smoothing_certified_radius` assumes). ratio = Δg·σ/‖δ‖ ≤ 1 ⟺ (1/σ)-Lipschitz holds.
    if lipProbe then
      let rGrid : Array Float := #[0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
      let mProbe := min 40 nEval
      let pstride := max 1 (nEval / mProbe)
      let clampP := fun (k : Nat) =>
        let lo := 1.0 / (2.0 * nEff.toFloat)
        max lo (min (1.0 - lo) (k.toFloat / nEff.toFloat))
      for t in [0:mProbe] do
        let imgIdx := t * pstride
        let off := imgIdx * d0
        let cnt0 ← sampleCountsB evalImg off nBatches ((imgIdx + 7).toUSize * 131 + 3)
        let mut chat := 0
        for c in [1:d1] do
          if cnt0[c]! > cnt0[chat]! then chat := c
        let gx := invNormCdf (clampP cnt0[chat]!)
        for ri in [0:rGrid.size] do
          let r := rGrid[ri]!
          let xp ← F32.perturbUnit evalImg off.toUSize d0.toUSize r (imgIdx.toUSize * 17 + ri.toUSize + 1)
          let cntp ← sampleCountsB xp 0 nBatches (imgIdx.toUSize * 53 + ri.toUSize + 11)
          let gxp := invNormCdf (clampP cntp[chat]!)
          let dg := Float.abs (gxp - gx)
          let ratio := dg * sigma / r
          if ratio > lipMax then lipMax := ratio
          if ratio > 1.05 then lipViol := lipViol + 1
          lipN := lipN + 1
          lipRows := lipRows.push s!"{sigma},{imgIdx},{r},{dg},{r/sigma},{ratio}"
      IO.println s!"  Lipschitz probe: {mProbe} imgs × {rGrid.size} shifts (Φ⁻¹∘p_ĉ vs ‖δ‖/σ)"
      (← IO.getStdout).flush
    let tot := nCert.toFloat
    let pct := fun (k : Nat) => k.toFloat / tot * 100.0
    let certStr := String.intercalate " / " (radii.toList.zipIdx.map
      (fun (r, ri) => s!"{r}→{pct certCnt[ri]!}%"))
    IO.println s!"  smoothed natural acc = {pct natCorrect}%   abstain = {pct abstain}%   ACR = {acr/tot}"
    IO.println s!"  certified-robust acc (L2): {certStr}"
    (← IO.getStdout).flush
    rows := rows.push s!"  {sigma}\t{bestAcc}\t{pct natCorrect}\t{pct abstain}\t{pct certCnt[1]!}\t{pct certCnt[3]!}\t{pct certCnt[5]!}\t{acr/tot}"
  IO.println s!"\n══ randomized smoothing ({net.name}): depth-independent certified L2 radius ══"
  IO.println "  σ\tclean%\tnat%\tabst%\tcert@.5\tcert@1.0\tcert@1.5\tACR"
  for row in rows do IO.println row
  let csvPath := s!"runs/smooth_{net.slug}_radii.csv"
  IO.FS.writeFile csvPath (String.intercalate "\n" dumpRows.toList ++ "\n")
  IO.println s!"\nper-image certified radii → {csvPath} ({dumpRows.size - 1} rows, all σ) — fine frontier curves + ACR"
  if lipProbe then
    let lipPath := s!"runs/smooth_{net.slug}_lipschitz.csv"
    IO.FS.writeFile lipPath (String.intercalate "\n" lipRows.toList ++ "\n")
    IO.println s!"Lipschitz-hypothesis probe → {lipPath} ({lipN} measurements): max ratio Δg·σ/‖δ‖ = {lipMax}"
    IO.println s!"  ⇒ Φ⁻¹∘p_ĉ is empirically (1/σ)-Lipschitz: {lipN - lipViol}/{lipN} measurements ≤ bound (violations>1.05: {lipViol})"
  IO.println "done (randomized smoothing: forward-only Monte-Carlo cert via the proof-rendered fwd —"
  IO.println "      architecture-agnostic + depth-independent, non-vacuous where ∏‖Wᵢ‖₂ is hopeless)."

/-- **Phase-3 PGD adversarial attack** on the verified linear classifier
    (`planning/robustness.md`). Trains via the proof-rendered train step, then attacks
    through the real IREE pipeline: each PGD step's input gradient is computed by the
    `genLinearPgdStep` StableHLO kernel (the proven `dx = (softmax−onehot)·Wᵀ` VJP) on the
    GPU. Reports clean vs L∞-PGD adversarial accuracy over an eps sweep. -/
def VerifiedNet.attackPgd (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut W0 ← F32.const (d0 * d1).toUSize 0.0
  let mut b0 ← F32.const d1.toUSize 0.0
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for _ in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← LowererSession.linearTrainStepV tsSess tsFn xb W0 b0 yb bs.toUSize d0.toUSize d1.toUSize
      W0 := out.extract 0 (d0 * d1 * 4)
      b0 := out.extract (d0 * d1 * 4) ((d0 * d1 + d1) * 4)
  -- clean accuracy
  let mut clean := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
        clean := clean + 1
  IO.println s!"clean test acc = {clean}/{nEval} = {clean.toFloat/nEval.toFloat*100.0}%"
  -- L∞ PGD sweep
  let K := 40
  for eps in ([0.1, 0.2, 0.3] : List Float) do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/linear_pgd_step.mlir" (genLinearPgdStep bs d0 d1 eps alpha true)
    compileVmfb ".lake/build/linear_pgd_step.mlir" ".lake/build/linear_pgd_step.vmfb"
    let pgdSess ← LowererSession.create ".lake/build/linear_pgd_step.vmfb"
    let pgdShapes := packShapes #[#[d0, d1], #[d1], #[bs, d1], #[bs, d0]]
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[W0, b0, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess "m.linear_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    IO.println s!"L∞ PGD eps={eps}: adv acc = {correct}/{nEval} = {correct.toFloat/nEval.toFloat*100.0}%"
  -- ── L2 sandwich: Lipschitz certificate (lower bound) vs L2 PGD (upper bound) ──
  let L := specNormW W0 d0 d1
  IO.println s!"\nglobal Lipschitz ‖W‖₂ = {L}  (linear: the logit map's exact L2 Lipschitz)"
  let tot := nEval.toFloat
  let mut cert05 := 0
  let mut cert10 := 0
  let mut cert15 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      let mut top := -1.0e30
      let mut sec := -1.0e30
      let mut topi := 0
      for c in [0:d1] do
        let v := F32.read logits (j * d1 + c).toUSize
        if v > top then
          sec := top
          top := v
          topi := c
        else if v > sec then
          sec := v
      if topi == F32.readLabel evalLbl (bi * bs + j) then
        let r := (top - sec) / (1.4142135623730951 * L)    -- certified L2 radius m(x)/(√2 L)
        if r ≥ 0.5 then cert05 := cert05 + 1
        if r ≥ 1.0 then cert10 := cert10 + 1
        if r ≥ 1.5 then cert15 := cert15 + 1
  IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  for eps in ([0.5, 1.0, 1.5] : List Float) do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/linear_pgd_step.mlir" (genLinearPgdStep bs d0 d1 eps alpha false)
    compileVmfb ".lake/build/linear_pgd_step.mlir" ".lake/build/linear_pgd_step.vmfb"
    let pgdSess ← LowererSession.create ".lake/build/linear_pgd_step.vmfb"
    let pgdShapes := packShapes #[#[d0, d1], #[d1], #[bs, d1], #[bs, d0]]
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[W0, b0, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess "m.linear_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    IO.println s!"L2 PGD eps={eps}: adv acc = {correct.toFloat/tot*100.0}%  (sandwich: cert ≤ true ≤ this)"
  IO.println "done (phase-3 PGD: gradient computed by the proven input-VJP kernel via IREE)."

/-- **fp8 (E4M3) Lean trainer** — the low-precision sibling of `trainLinear`.

    Keeps **fp32 master weights** and, each step, projects the weights
    (per-output-column) and the activations (per-tensor) onto the **E4M3** grid
    (`LeanMlir/E4M3Quant.lean`), runs the *same* verified `@<slug>_train_step`
    kernel (the matmul accumulates in fp32 — the `dotMixed` model: `u_leaf =
    E4M3`, `u_acc = fp32`), and applies the recovered gradient delta to the fp32
    master via `addDelta` (`master += Wout − Wq = master − lr·∇`). The MLIR and
    FFI are **unchanged**: fp8 here is host-side operand byte-prep, exactly the
    §3b render-tie model (`Proofs/E4M3FaithfulPoC.lean`). Eval runs the fp32
    master through `@<slug>_fwd` (the "fp32-infer" accuracy of the fp8-trained
    model, mirroring `scripts/mnist_e4m3_demo.py`).

    Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-linear-e4m3-verified data` -/
def VerifiedNet.trainLinearE4M3 (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  net.printBlurb
  IO.println "  [fp8 E4M3] fp32 master · per-column W / per-tensor x → E4M3 grid · fp32 accumulate"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _trainPix, _crop) ←
    loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; dense {d0}->{d1}, bs {bs}, fp8-SGD (E4M3 leaf / fp32 acc)"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  -- Static per-tensor activation scale ⇒ quantize the whole train set ONCE.
  let trainImgQ := F32E4M3.quantPerTensor trainImg
  let mut mW ← F32.const (d0 * d1).toUSize 0.0     -- fp32 master weights (zero-init)
  let mut mb ← F32.const d1.toUSize 0.0            -- fp32 master bias (unquantized)
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImgQ (bi * bs) bs d0     -- E4M3 activations
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let Wq := F32E4M3.quantPerColumn mW d0 d1               -- E4M3 weight operand
      let out ← LowererSession.linearTrainStepV tsSess tsFn
                  xb Wq mb yb bs.toUSize d0.toUSize d1.toUSize
      let Wout := out.extract 0 (d0 * d1 * 4)
      let bout := out.extract (d0 * d1 * 4) ((d0 * d1 + d1) * 4)
      mW := F32E4M3.addDelta mW Wout Wq                       -- master += (Wout − Wq)
      mb := bout                                              -- bias update is exact (unquantized)
    let params := mW ++ mb
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn params shapes
                      xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let pred := (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat
        let lbl  := F32.readLabel evalLbl (bi * bs + j)
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / nEval.toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nEval} = {acc}% (fp8 E4M3)"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} in fp8 E4M3 on the proof-rendered StableHLO)."

/-- **fp8 (E4M3) packed-params trainer** — the low-precision sibling of
    `VerifiedNet.train`, for the depth>1 nets (MLP, CNN). Keeps **fp32 master
    params** and, each step, projects every *weight* slot onto the E4M3 grid
    (dense per-output-column, conv per-output-channel; biases kept fp32 —
    `F32E4M3.quantPackedParams`) and the *input* per-tensor, runs the *same*
    verified `@<slug>_train_step` (fp32 accumulate inside), and folds the
    gradient delta back into the master with `addDelta` over the whole packed
    buffer (`master += out − paramsQ`: weight slots get `−lr·∇`, bias slots the
    exact update). MLIR/FFI unchanged.

    **Scope (honest):** host-side prep reaches weights + the *input* activation
    only. The intermediate activations (relu/pool/flatten outputs feeding the
    deeper matmuls) and the backward-chain cotangents are computed *inside* the
    fused kernel and stay fp32 — quantizing them needs in-graph E4M3 ops (the
    next, codegen-level step), not host byte-prep. So this is honest **fp8
    weights + fp8 input, fp32 intermediates**. Eval runs the fp32 master.

    Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-e4m3-verified data` -/
def VerifiedNet.trainE4M3 (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  net.printBlurb
  IO.println "  [fp8 E4M3] fp32 master · per-slot weight quant (dense per-col / conv per-channel) + per-tensor input · fp32 accumulate"
  IO.println "  note: depth>1 ⇒ intermediate activations & cotangents stay fp32 (inside the kernel); weights + input are E4M3"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} ({net.specs.size} params, {net.nParams} floats), fp8-SGD (E4M3 leaf / fp32 acc), He init"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts                       -- fp32 master params
  -- Static per-tensor input scale ⇒ quantize the train images ONCE (crop, if any,
  -- only selects grid-valued pixels, so quantize-then-crop stays on the grid).
  let trainImgQ := F32E4M3.quantPerTensor trainImg
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xbRaw := F32.sliceImages trainImgQ (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let paramsQ := F32E4M3.quantPackedParams params net.specs   -- E4M3 weight operands
      let out ← LowererSession.mlpTrainStepV tsSess tsFn
                  xb paramsQ shapes yb bs.toUSize d0.toUSize nc.toUSize
      params := F32E4M3.addDelta params out paramsQ              -- master += (out − paramsQ)
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn params shapes
                      xb xShape bs.toUSize nc.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let pred := (F32.argmaxN logits (j * nc).toUSize nc.toUSize).toNat
        let lbl  := F32.readLabel evalLbl (bi * bs + j)
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / nEval.toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nEval} = {acc}% (fp8 E4M3)"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} in fp8 E4M3 on the proof-rendered StableHLO)."

/-- **fp8 (E4M3) variant of `trainAdamSched`** — runs the Adam / Nesterov-momentum
    optimizer demos in fp8. Keeps an fp32 master `[θ|m|v]`; each step projects the
    *weight* third `θ` onto the E4M3 grid (`quantPackedParams`: dense per-column,
    conv per-channel; biases fp32) and the input per-tensor, runs the *same*
    verified `@<slug>_<variant>_train_step` (the optimizer is baked into the MLIR,
    so fp8 needs no new module — operand byte-prep only; fp32 accumulate), and folds
    the optimizer-step delta back into the fp32 master θ (`addDelta`), keeping the
    returned `m'/v'` moments in fp32. Distinct `_e4m3` checkpoint (won't resume an
    fp32 run); honors `LEAN_MLIR_MAX_EPOCHS`. Same scope as `trainE4M3`: fp8
    weights + input, fp32 intermediates / moments. -/
def VerifiedNet.trainAdamSchedE4M3 (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (baseLR β1 β2 : Float) (warmupEpochs : Nat) (variant : String := "adam") : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  net.printBlurb
  IO.println s!"  [fp8 E4M3] fp32 master [θ|m|v] · per-slot θ quant + per-tensor input · fp32 accumulate ({variant})"
  let hasBn := !net.bnChannels.isEmpty
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let tsVmfb  := s!".lake/build/{net.slug}_{variant}_ts.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  let fwdEvalVmfb := s!".lake/build/{net.slug}_fwd_eval_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_{variant}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"             fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let fwdEvalSess ← if hasBn then do
      compileVmfb s!"{net.mlirDir}/{net.slug}_fwd_eval.mlir" fwdEvalVmfb
      LowererSession.create fwdEvalVmfb
    else pure fwdSess
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} {variant} fp8 (cosine+warmup {warmupEpochs}ep, baseLR {baseLR}), He init"
  (← IO.getStdout).flush
  let adamShapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes ++ #[#[], #[], #[]]
                                ++ (if hasBn then bnStatShapes else #[]))
  let fwdShapes := net.shapesBA
  let fwdEvalShapes := packShapes (net.paramShapes ++ bnStatShapes)
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_{variant}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let theta := F32.concat parts
  let zeros ← F32.const net.nParams.toUSize 0.0
  let mut thetamv := F32.concat #[theta, zeros, zeros]
  let mvBytes := 3 * net.nParams * 4
  let pBytes := net.nParams * 4
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  let mut bnFirst := true
  let totalSteps := (cfg.epochs * nb).toFloat
  let warmSteps := (warmupEpochs * nb).toFloat
  let ckptPath := s!".lake/build/{net.slug}_{variant}_e4m3_ckpt.bin"   -- distinct from the fp32 runs
  let epPath := ckptPath ++ ".epoch"
  let mut startEpoch := 0
  if (← System.FilePath.pathExists ckptPath) && (← System.FilePath.pathExists epPath) then
    thetamv ← IO.FS.readBinFile ckptPath
    startEpoch := ((← IO.FS.readFile epPath).toNat?).getD 0
    IO.println s!"  ▸ resuming from fp8 checkpoint at epoch {startEpoch}"
    (← IO.getStdout).flush
  -- pre-quantize the train images ONCE (per-tensor E4M3); shuffle + hflip preserve the grid.
  let mut curImg := F32E4M3.quantPerTensor trainImg
  let mut curLbl := trainLbl
  for ep in [startEpoch:nEpochs] do
    let mut epochLossSum := 0.0
    let mut lastLr := 0.0
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPix.toUSize
                           4 -- classification: one f32 class id per record
                           (ep + 42).toUSize
    curImg := sImg; curLbl := sLbl
    for bi in [0:nb] do
      let gstep := (ep * nb + bi + 1).toFloat
      let lrt := if gstep ≤ warmSteps then baseLR * gstep / warmSteps
                 else baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (gstep - warmSteps) / (totalSteps - warmSteps)))
      let bc1 := 1.0 - Float.exp (gstep * Float.log β1)
      let bc2 := 1.0 - Float.exp (gstep * Float.log β2)
      let tail := F32.concat #[← F32.const (1 : USize) lrt, ← F32.const (1 : USize) bc1, ← F32.const (1 : USize) bc2]
      -- fp8: project the θ third onto the E4M3 grid (weights per-slot; biases + m/v stay fp32).
      let thetaMaster := thetamv.extract 0 pBytes
      let thetaQ := F32E4M3.quantPackedParams thetaMaster net.specs
      let thetamvQ := F32.concat #[thetaQ, thetamv.extract pBytes mvBytes]
      let params := if hasBn then F32.concat #[thetamvQ, tail, runningBnStats] else F32.concat #[thetamvQ, tail]
      let augSeed := (ep * nb + bi + 1).toUSize
      let xbRaw := F32.sliceImages curImg (bi * bs) bs trainPix
      let xb ← match net.data with
        | .imagenette =>
            let c ← if crop then F32.randomCrop xbRaw bs.toUSize 3 256 256 224 224 augSeed
                    else pure xbRaw
            F32.randomHFlip c bs.toUSize 3 224 224 (augSeed + 7777)
        | .cifar => F32.randomHFlip xbRaw bs.toUSize 3 32 32 augSeed
        | _ => pure xbRaw
      let yb := F32.sliceLabels curLbl (bi * bs) bs
      let out ← LowererSession.mlpTrainStepV tsSess tsFn xb params adamShapes yb bs.toUSize d0.toUSize nc.toUSize
      let stepLoss := F32.read out (3 * net.nParams).toUSize
      epochLossSum := epochLossSum + stepLoss
      lastLr := lrt
      if bi < 3 || bi % 100 == 0 then
        IO.println s!"  step {bi}/{nb}: loss={stepLoss}"
        (← IO.getStdout).flush
      -- fp8 master recovery: θ_master += (θ' − θ_q); keep the returned fp32 m'/v'.
      let thetaPrime := out.extract 0 pBytes
      let mvPrime := out.extract pBytes mvBytes
      let thetaMasterNew := F32E4M3.addDelta thetaMaster thetaPrime thetaQ
      thetamv := F32.concat #[thetaMasterNew, mvPrime]
      if hasBn then
        let batchBn := out.extract ((3 * net.nParams + 3) * 4) ((3 * net.nParams + 3 + nBnStats) * 4)
        -- ⚠ 0.01, NOT 0.1 — corrected 2026-08-04. `F32.ema` computes
        -- `(1−m)·running + m·batch`, so this `m` is the weight on the NEW batch, and the
        -- reference's `momentum=0.99` (`_bn` in `jax/Jax/Codegen.lean`, which updates
        -- `momentum*rm + (1−momentum)*bm`) is `m = 0.01` here. At 0.1 the running stats
        -- averaged ~10 batches against the reference's ~100 — 10× noisier. It is EVAL-ONLY,
        -- so it depressed every reported top-1 without touching a single gradient, and it
        -- bit hardest early, when the activation statistics are still moving fast.
        -- ⚠ NO accumulation compensation here, and that is correct rather than an omission:
        -- this fp8 trainer has no `accK` — it does not implement gradient accumulation at all —
        -- so k = 1 and the compensated form `1 − 0.99^(1/k)` is exactly this 0.01. If an
        -- accumulation path is ever added here, copy `bnMom` from `trainAdamSched`.
        runningBnStats ← F32.ema runningBnStats batchBn (if bnFirst then 1.0 else 0.01)
        bnFirst := false
    IO.println s!"Epoch {ep + 1}/{nEpochs}: loss={epochLossSum / nb.toFloat} lr={lastLr}"
    let thetaCur := thetamv.extract 0 pBytes
    let evalSess := if hasBn then fwdEvalSess else fwdSess
    let evalFn := if hasBn then s!"m.{net.slug}_fwd_eval" else fwdFn
    let evalParams := if hasBn then F32.concat #[thetaCur, runningBnStats] else thetaCur
    let evalShapes := if hasBn then fwdEvalShapes else fwdShapes
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 evalSess evalFn evalParams evalShapes
                      xb xShape bs.toUSize nc.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let pred := (F32.argmaxN logits (j * nc).toUSize nc.toUSize).toNat
        let lbl  := F32.readLabel evalLbl (bi * bs + j)
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / nEval.toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nEval} = {acc}% (fp8 E4M3, {variant})"
    (← IO.getStdout).flush
    IO.FS.writeBinFile ckptPath thetamv
    IO.FS.writeFile epPath (toString (ep + 1))
  IO.println s!"done (trained {net.name} {variant} in fp8 E4M3 on the proof-rendered StableHLO)."
