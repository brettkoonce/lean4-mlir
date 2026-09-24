import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.E4M3Quant
import LeanMlir.VerifiedSpec

/-! # Shared driver for the `*-verified` trainers

Every `Main*Verified.lean` trains a network on **pre-rendered, audited** StableHLO
(`verified_mlir/<slug>_{train_step,fwd}.mlir`, emitted offline by `tests/Test*` from
the proof stack) through the runtime FFI (PJRT by default, IREE optionally). Unlike the reference `NetSpec`/`Train.lean`
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

This is the training driver alone (plus its fp8 E4M3 variants). The PGD attacks and spectral-norm
studies are `VerifiedAttack` (on `VerifiedPgdGen`'s kernels), the smoothing certificate is
`VerifiedSmoothing`, and `VerifiedNets` imports all of them.

NB `VerifiedConfig.lr` is for the banner only. The SGD-inline train steps bake the learning rate
into the rendered MLIR (re-render to change it); the Adam-family steps take it as a runtime operand,
from `trainAdamSched`'s own `baseLR` argument and schedule.
-/

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
      `initKind`: 0 = He(fan-in), 1 = ones (γ), 2 = zeros (β / bias), 3 = 1e-6 (layer scale). -/
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
      `cifar8`, `cifar8_bn`, `cifar`, `mobilenetv2`, `efficientnet`, `convnext`,
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
      (`planning/archive/stochastic_depth.md`). Empty = the net has no drop sites, which is every net today
      except EfficientNet's `*sd` variants.

      ⚠ THE DRIVER OWNS THE RAMP, and that is deliberate rather than a shortcut: the emitted op is a
      pure per-example multiply and `1/keep_i` is folded into the value supplied here, because a
      BAKED `1/keep_i` and "the forward emits the sites too" cannot both hold (a ones scale would
      then compute `x/keep_i`, and the reference returns the branch untouched at eval). It is the
      same place `%lr` lives, for the same reason — one graph, many schedules.

      ⚠ It is therefore a SECOND hand-list against the renderer's `enetDropIdxs`, exactly like
      `toSpecs == XLayout.specs`. [`tests/TestDropPathRamp.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestDropPathRamp.lean) is the `#guard` that pins the two;
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
  /-- Validate every N epochs, plus always the last epoch the process runs — the verified peer
      of `TrainConfig.valEveryEpochs`, which the reference's S/B/ViT-S/B/MobileNetV4 ImageNet
      configs set to 5. N ≤ 1 keeps every-epoch validation. ImageNet path (`trainAdamSched`)
      only, and only the eval pass: the checkpoint is still written every epoch. Measured
      2026-09-21 on ConvNeXt-T at four replicas, the sharded val pass is ~25 s per epoch, so
      this buys ~2 h over 300 epochs — not the hours the reference's tfds pipeline rebuild cost.
      `LEAN_MLIR_VAL_EVERY=<n>` overrides it at launch, like `LEAN_MLIR_MAX_EPOCHS`. -/
  valEveryEpochs : Nat := 1
  /-- Learning rate. DISPLAY ONLY — SGD-inline steps bake it into `<slug>_train_step.mlir`;
      `trainAdamSched` takes its own `baseLR`. Changing it here does not change training. -/
  lr        : Float := 0.1
  /-- timm/DeiT ViT weight init — the verified peer of `TrainConfig.vitInit`, i.e. of the
      `deit-init` recipe the phase-2 reference run used (blueprint §9.6). Every weight at
      σ = 0.02 except the patch-embed conv on PyTorch's `U(±1/√fan_in)`. Unlike `lr`, this is
      NOT display-only: init is host-side, so the flag genuinely changes training and no
      re-render is needed. Off by default — every other net keeps its seed reproducibility. -/
  vitInit   : Bool := false
  /-- **ConvNeXt `_init_weights` — the verified peer of `TrainConfig.cnxInit`** (2026-09-18).
      `trunc_normal_(std=0.02)` on **every conv AND the head**; biases 0, LayerNorm γ 1,
      LayerScale γ 1e-6 — which the other three `kind`s already do, so this flag only has to
      reach the WEIGHTS.

      ⛔ **Why it exists.** The 2026-09-17 ConvNeXt/ImageNet pair run was killed at epoch 67
      because the two arms did not share an init: the reference sets `cnxInit := true`
      ([`jax/MainConvNeXtImagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainConvNeXtImagenet.lean)) while the verified path used the He fan-in default —
      **2.6x-10.2x wider**, worst at the 4x4 stem (0.2041 vs 0.02) and the 7x7 depthwise
      (0.2020 vs 0.02). That confounds exactly the lowering question the pair exists to answer.
      `runs/2026-09-17-cnx-verified-300ep/RESULTS.md` §7.0 carries the measurement.

      ⚠ **SEPARATE from `vitInit`, deliberately, and this is the same trap `TrainConfig` records
      on the JAX side**: the two specs DISAGREE on the conv path. timm's ViT leaves the patch
      embed on PyTorch's `U(±1/√fan_in)` while ConvNeXt trunc-normals its convs like everything
      else. One boolean cannot express both, so setting `vitInit` for a ConvNeXt would put the
      stem back at the wrong width.

      ⚠ **Variance-matched, not distribution-matched.** `F32.heInit` sums three uniforms
      (Bates-3, ≈normal) where the reference draws `trunc_normal`. At σ = 0.02 that truncation is
      INERT — `trunc_normal_`'s bounds are ABSOLUTE ±2, i.e. ±100σ, so nothing is ever truncated
      and the reference is effectively a plain normal. What remains is Bates-3's slightly lighter
      tail at equal σ, the same deliberate gap every other net carries.

      Like `vitInit`, NOT display-only and needs no re-render: init is host-side, so no committed
      artifact moves. Off by default — every other net keeps its seed reproducibility. -/
  cnxInit   : Bool := false
  /-- BatchNorm running-statistic **decay** — the verified peer of `TrainConfig.bnMomentum`,
      and the same TF sense: the weight on the OLD estimate, so timm's PyTorch
      `momentum = 0.1` is `0.9` here. That field's docstring carries the per-net table and
      the timm audit; keep the two in step, since a phase-2 ↔ phase-4 gap here is invisible
      in every loss curve.

      Like `vitInit` and unlike `lr`, this is NOT display-only and needs no re-render: the
      graph emits raw per-layer BATCH stats and the host EMAs them (`F32.ema` below), so
      the decay never enters the MLIR. Under gradient accumulation the driver compensates
      to `bnMomentum^(1/k)` per micro-batch, matching the reference's generated `_bn`. -/
  bnMomentum : Float := 0.99

/-- **The 95% Wilson score interval on an accuracy**, as `lo–hi` in percentage points.

    ⭐ Added 2026-08-30 because the eval line printed `3339/3925 = 85.070064%` — six significant
    figures on a quantity a 3,925-image validation set resolves to about **one**. At p ≈ 0.85 and
    n = 3925 the 95% half-width is ±1.11 pt, so all but the leading three digits were decoration,
    and the ~1.3 pt epoch-to-epoch swings that get read as "the model moved" are inside it.

    ⚠ **Wilson, not the normal approximation `p ± z·√(p(1−p)/n)`.** The normal form collapses to
    **±0** at `correct = 0` or `correct = n` — it would print a *perfectly certain* 0.000000% on a
    run that scored nothing, which is exactly the `LEAN_MLIR_SKIP_EVAL` failure the line below this
    one already had to be taught to say out loud. Wilson stays finite there.

    ⚠ It is the MEASUREMENT error only — how well 3,925 images pin this model's accuracy. It says
    nothing about seed-to-seed training variance, which needs n runs, and it is the WRONG test for
    comparing two models scored on the SAME set: that comparison is paired, so it wants McNemar
    over `LEAN_MLIR_DUMP_CORRECT`'s per-example bitmaps, which is far more powerful. -/
def wilson95 (correct nEval : Nat) : String :=
  if nEval == 0 then "n/a" else
  let n := nEval.toFloat
  let p := correct.toFloat / n
  let z := 1.959964
  let d := 1.0 + z * z / n
  let c := (p + z * z / (2.0 * n)) / d
  let h := (z / d) * Float.sqrt (p * (1.0 - p) / n + z * z / (4.0 * n * n))
  -- ⚠ Formatted from INTEGER hundredths, not `toString` on a Float: Lean prints a Float at six
  -- decimals, so `s!"{(x*10000.0).round/100.0}"` would emit `83.920000–86.150000` — the very
  -- false precision this function exists to remove, reintroduced by the printer.
  let pct := fun (x : Float) =>
    let y := if x < 0.0 then 0.0 else if x > 1.0 then 1.0 else x
    let r := (y * 10000.0).round.toUInt64.toNat
    s!"{r / 100}.{if r % 100 < 10 then "0" else ""}{r % 100}"
  s!"{pct (c - h)}–{pct (c + h)}"

/-- The weight `F32.ema` puts on the NEW batch, i.e. `1 − decay`, given the accumulation
    factor: `some k` on an `acc` variant (the EMA fires once per MICRO-batch, so the decay is
    k-th-rooted for k chained updates to compose to one `bnMomentum`/optimizer-step update),
    `none` otherwise.

    ⭐ **ONE definition, because it has three consumers** — `trainAdamSched`'s loop, its startup
    banner, and the fp8 trainer — and this file's own `VerifiedVariant` docstring is the record
    of what happens when a driver and its gate each keep a copy: an edit to the real expression
    cannot turn the copy red.

    ⚠ The `== 0.99` arm returns the historic `0.01` DOUBLE rather than `1.0 - 0.99`
    (= 0.010000000000000009 — different bits, 9e-16 relative), so every net that leaves
    `bnMomentum` at its default is bit-identical across the knob's introduction.
    ⚠ `some 0` is reachable — `VerifiedVariant.accK` returns 0 when it cannot parse a k out of
    the variant name — and lands on `1/0 = inf`, `pow → 0`, weight 1.0, i.e. the running stats
    become the latest batch. That is the pre-knob behaviour, preserved deliberately rather than
    quietly repaired: a variant name whose k does not parse is already training at a wrong
    effective LR (see `accK`), and this should not be the thing that hides it. -/
def VerifiedConfig.bnEmaWeight (cfg : VerifiedConfig) : Option Nat → Float
  | some k => 1.0 - Float.pow cfg.bnMomentum (1.0 / k.toFloat)
  | none   => if cfg.bnMomentum == 0.99 then 0.01 else 1.0 - cfg.bnMomentum

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
the name. [`tests/TestVariantPredicates.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestVariantPredicates.lean) is the table of what each must read, and its
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
    "rms" (`planning/archive/ema.md`'s defect). -/
def rmsOn (v : String) : Bool := v.contains "rms"

/-- Stochastic depth — N extra `tensor<Bxf32>` scale inputs.
    ⚠ The marker is `drop` and not `sd` because `rms` ++ `dp` spells `rmsdp`, which contains
    "sd" (`planning/archive/stochastic_depth.md`'s defect). -/
def sdOn (v : String) : Bool := v.contains "drop"

/-- Classifier dropout — ONE extra `tensor<B×wxf32>` mask input.
    ⚠ The marker is `do` and not `dropout` because `dropout` contains `drop`, so a dropout-only
    variant would read as a stochastic-depth one (`recipe_gaps.md` gap C). -/
def cdOn (v : String) : Bool := v.contains "do"

/-- Gradient accumulation — a FOURTH `[θ|m|v|G]` region, 5 scalars.
    ⚠⚠ SUBSTRING, not prefix: RSB-A3's composed optimizer is `lambaccdp8x64bce`, where `lamb` ++
    `acc` puts the marker in the MIDDLE. -/
def accOn (v : String) : Bool := v.contains "acc"

/-- LAMB — the per-tensor trust ratio (`R34Opt.lambAccum`), RSB-A3's optimizer.
    ⚠ SUBSTRING, not prefix, for `accOn`'s reason one spelling over: the EMA form is `emalamb…`,
    which does not start with "lamb". -/
def lambOn (v : String) : Bool := v.contains "lamb"

/-- BCE-with-logits (timm `BinaryCrossEntropy`), RSB's loss — not softmax CE.
    ⚠ SUBSTRING: the marker TRAILS the shape and is itself often trailed, by `wd001` or `bf16`
    (`lambaccdp8x64wxclipbcebf16`), so neither a prefix nor a suffix test finds it. -/
def bceOn (v : String) : Bool := v.contains "bce"

/-- `k`, read back out of the name. The graph has `1/k` BAKED in and the driver decides the apply
    cadence; a disagreement does not fail, it trains at a silently wrong effective learning rate.
    Parsed from AFTER the marker, not from a fixed offset — see `accOn`. -/
def accK (v : String) : Nat :=
  if accOn v then
    let after := (v.splitOn "acc").getD 1 ""
    let after := if after.startsWith "dp" then after.drop 2 else after
    ((after.takeWhile (· != 'x')).toNat?).getD 0
  else 1

/-- Blob regions: `[θ|m|v]`, plus `G` (gradient accumulation) and/or `E` (the EMA shadow) — so
    **3, 4 or 5**, and the two extras are INDEPENDENT.
    ⚠ A 3-region file loaded by a 4-region driver (or the reverse) misaligns EVERY parameter, so
    every consumer of a checkpoint sizes off this rather than off a literal.

    ⭐⭐ **THIS USED TO BE `if emaOn || accOn then 4 else 3`, and the two features were mutually
    exclusive because of it** — `trainAdamSched` threw on the pairing, and RSB-A2/A1 could not be
    rendered faithfully (their recipe sets BOTH `gradAccumSteps := 4` and `useEMA := true`;
    `planning/archive/verified_side_quest_counterparts.md` §4a). The fifth region is what lifts that.

    ⚠⚠ **`G` COMES BEFORE `E`, and that ordering is not free**: at `acc` alone `G` is region 3 and
    at `ema` alone `E` is region 3, so every checkpoint written before this change still reads at
    the index it was written at. The reverse order would have silently re-homed every committed
    `ema*` blob. -/
def nRegions (v : String) : Nat :=
  3 + (if accOn v then 1 else 0) + (if emaOn v then 1 else 0)

/-- Rank-0 scalar slots in the blob tail: `lr,bc₁,bc₂`, then `%aup,%akeep` (accumulation) and then
    `%emad,%oemad` (EMA) — so **3, 5 or 7**, in that order.
    ⚠ Same independence and same ordering rule as `nRegions`: the accumulation pair keeps slots
    3–4 and the EMA pair moves to 5–6 only when both are on, so neither single-axis layout moves. -/
def nScalars (v : String) : Nat :=
  3 + (if accOn v then 2 else 0) + (if emaOn v then 2 else 0)

/-- The blob index of the EMA shadow region, or `none` when the variant has no shadow.
    ⚠ It is **not the literal 3** any more: under accumulation `G` takes region 3 and the shadow is
    region 4. `scoreCheckpoint` and the per-epoch eval both slice θ out of the blob with this, and
    a stale literal there does not fail — it scores the gradient accumulator as if it were weights
    and prints a plausible percentage off it. -/
def emaRegion (v : String) : Option Nat :=
  if emaOn v then some (3 + (if accOn v then 1 else 0)) else none

/-- Offset of the `%emad,%oemad` pair inside the scalar tail — 3 alone, 5 behind `%aup,%akeep`. -/
def emaScalarOff (v : String) : Nat := 3 + (if accOn v then 2 else 0)

end VerifiedVariant

/-- iree-compile one `.mlir` → `.vmfb`, surfacing failures. Skips when the `.vmfb` is already
    newer than the `.mlir` (a content-stable cache): avoids the ~minutes-long 224² recompile, and
    lets two same-net runs share one GPU-pair safely — they only *read* the cached vmfb (concurrent
    reads are fine; it's the concurrent *writes* of an identical compile that would race). -/
def compileVmfb (mlirPath outPath : String) : IO Unit := do
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
    dlopened (`planning/archive/xla_pjrt_ladder.md`).

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
    -- reference — see planning/archive/xla_pjrt_ladder.md §8, rung 3.
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    let base := (mlirPath.splitOn "/").getLastD mlirPath
    let stem := if base.endsWith ".mlir" then (base.dropEnd 5).toString else base
    compileVmfb mlirPath s!".lake/build/{stem}_{target}.vmfb"
    LowererSession.create s!".lake/build/{stem}_{target}.vmfb"

/-- `mkSession` for the **eval forward** at `replicas` devices, the sharded-inference session
    (`LowererSession.createDp`, outputs gathered). `replicas ≤ 1` IS `mkSession`, so a single-GPU
    run and every IREE run are unchanged. Past 1 it is XLA-only and says so rather than quietly
    evaluating on one card, which is where this eval lived until 2026-09-18 (3 of 4 GPUs idle for
    ~100 s every ConvNeXt epoch). -/
def mkSessionDp (mlirPath : String) (replicas : Nat) : IO LowererSession := do
  if replicas ≤ 1 then return ← mkSession mlirPath
  if (← LowererSession.backendName) != "xla" then
    throw <| IO.userError s!"a {replicas}-replica eval needs the XLA lowerer; this binary loaded \
'{← LowererSession.backendName}'"
  IO.println s!"  xla/pjrt {mlirPath}  (eval, SHARDED over {replicas} replicas)"
  LowererSession.createDp mlirPath replicas.toUSize

/-- Init one parameter from its `(dims, initKind)` spec, matching the JAX reference's
    initialisers — they are the oracle these nets are paired against:

      * rank-4 conv kernel `[oc, ic, kH, kW]` → He **fan-OUT**, variance `2/(oc·kH·kW)`
      * rank-2 dense matrix `[in, out]`       → **Glorot**, variance `2/(in + out)`
      * γ = 1 (kind 1), β / bias = 0 (kind 2), layer scale = 1e-6 (kind 3)

    ⚠ **Kind 3 is new on 2026-09-13.** ConvNeXt's layer scale was kind 1 — ones — while the JAX
    reference (`emitLayerScaleInit`) and the paper use 1e-6, so the two paths never shared an
    init on that net; every ConvNeXt accuracy recorded before this date is from γ = 1. Kind 3
    must be matched here explicitly: the `_` branch below is He, which is the wrong answer for a
    per-channel scale.

    ⚠ **Both weight cases CHANGED 2026-08-04.** This used variance `2/fan_in` for BOTH, where
    [`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean) emits `uniform(±√(6/fan_out))` for convs (variance `2/fan_out` —
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
def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat)
    (vitInit : Bool := false) (biasSigma : Option Float := none)
    (heFanIn : Bool := false) (cnxInit : Bool := false) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 3 => F32.const n.toUSize 1e-6
  | 2 =>
    -- `biasSigma` is the gates' escape hatch, not a trainer knob: the weight-decay and grad-clip
    -- ties need a non-zero bias or their update is identically vacuous. `none` is the trainer's
    -- own zero-init and is what every net trains from.
    match biasSigma with
    | none   => F32.const n.toUSize 0.0
    | some s => F32.heInit seed.toUSize n.toUSize s
  | _ =>
    -- ⭐ **`vitInit` = timm/DeiT ViT init, the verified peer of `TrainConfig.vitInit`** on the JAX
    -- side (`jax/MainVitImagenet.lean`'s `deit-init` recipe). Off by default, so every non-ViT net
    -- is byte-identical and every recorded accuracy still reproduces from its seed.
    --
    -- The rule is uniform because timm's is: `init_weights_vit_timm` gives EVERY `nn.Linear`
    -- `trunc_normal_(std=0.02)`, and the CLS token and positional embedding are already 0.02 on the
    -- JAX side. So every weight lands at σ = 0.02 except the patch-embed `nn.Conv2d`, which timm
    -- leaves on PyTorch's default `U(±1/√fan_in)` with `fan_in = ic·kh·kw` — σ = 1/√(3·fan_in) =
    -- **0.02083** at ViT-Ti, 4% off the Linears rather than equal to them. Emitted exactly, not
    -- rounded to 0.02, because the whole point of the flag is to stop approximating this.
    --
    -- ⚠ Why it matters: the default branch below is Glorot for rank-2, which scales as 1/√d against
    -- timm's FIXED 0.02 and is therefore **3.6× too wide at ViT-Ti's d=192** (0.0722 vs 0.02), while
    -- rank-1 (CLS) comes out at 0.102 — 5× wide. Blueprint §9.6 carries the measurement.
    --
    -- ⚠ DISTRIBUTION, as ever, is matched in variance only: `F32.heInit` sums three uniforms
    -- (Bates-3, ≈normal) where the JAX side draws `random.normal`. Same σ, different shape — the
    -- same deliberate gap the 2026-08-04 note below records for every other net.
    let variance :=
      -- ⭐ **ConvNeXt `_init_weights`: σ = 0.02 on EVERY weight, whatever its rank.** Simpler
      -- than `vitInit` below, which has to special-case the patch embed — ConvNeXt trunc-normals
      -- its convs and its head alike. Measured on this net's 183 specs: 58 rank-4 (stem, 7x7
      -- depthwise, the 1x1s, the 2x2 downsamples) + 1 rank-2 (head) land here; the other 124 are
      -- `kind` 1/2/3 above (LayerNorm γ=1, biases 0, LayerScale γ=1e-6) and already match the
      -- reference, so this branch is the whole of the difference.
      -- ⚠ FIRST, so it cannot be silently overridden by a rank test below it.
      if cnxInit then 0.0004                                                      -- 0.02²
      else if vitInit then
        if dims.size == 4 then 1.0 / (3.0 * (dims[1]! * dims[2]! * dims[3]!).toFloat)  -- Conv2d dflt
        else 0.0004                                                                     -- 0.02²
      -- ⚠ `heFanIn` is a SEPARABILITY knob for one gate, not an initialisation opinion. It is the
      -- rule 25 gates had hand-copied before 2026-09-02; they all now use the fan-OUT default and
      -- their verdicts were unchanged by the move. The single holdout is `tests/TestRmsTie.lean`:
      -- its coupled-L2 control asks whether `wd·θ` is present, `wd = 4e-5` is baked into the
      -- committed `mobilenetv2_rms_train_step.mlir`, and the fan-OUT default makes mnv2's
      -- gradients 5.8× larger (|g|max 0.85 → 4.94), which drops `wd·θ/g` to ~2e-6 and below f32
      -- separability. The gate detects this itself and throws CONTROL DEAD rather than passing
      -- vacuously. Retiring this flag means re-rendering that artifact at a larger `wd`.
      else if heFanIn then
        (if dims.size == 4 then 2.0 / (dims[1]! * dims[2]! * dims[3]!).toFloat
         else 2.0 / (dims[0]!).toFloat)
      else if dims.size == 4 then 2.0 / (dims[0]! * dims[2]! * dims[3]!).toFloat   -- He, fan-OUT
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

/-- Read up to `len` bytes from `h` **into `buf`**, appending at `buf.size` and never touching its
    capacity — which is the reason it exists. `IO.FS.Handle.read` allocates its result on the
    CALLING thread, so a batch read on a pool thread and dropped on the main thread is freed by a
    thread that does not own it. The runtime's allocator (mimalloc) answers a cross-thread free of
    a huge block with `madvise(MADV_FREE)`, not a release: the pages stay in RSS as `LazyFree`
    until the kernel is under pressure. On ImageNet that was 79–136 MB/step, the box full inside
    one epoch, then continuous direct reclaim and a mean step 2.5× the median
    ([runs/2026-09-11-vit-leak-ab/README.md](https://github.com/brettkoonce/lean4-mlir/blob/main/runs/2026-09-11-vit-leak-ab/README.md)).
    With this primitive the main thread allocates AND frees every batch buffer; the pool thread
    only fills it, and the block is recycled in place step after step.
    ⚠ `buf` must be the ONLY reference (rc 1, or −1 once it has crossed a `Task` boundary). The C
    side refuses a shared buffer rather than copying it, because a silent copy would put the
    allocation straight back on the reading thread. Returns the buffer with its size advanced by
    the bytes read; 0 bytes means end of stream, as for `Handle.read`. -/
@[extern "lean_mlir_read_into"]
opaque readInto (h : @& IO.FS.Handle) (buf : ByteArray) (len : USize) : IO ByteArray

/-- Fill `buf` with EXACTLY `n` more bytes, looping until they arrive. A pipe read returns what is
    *available*, not what was asked for — at 154 MB per batch a short read is the normal case, not
    the edge case, and treating one `read` as a batch silently misaligns the stream from then on. -/
def readExactInto (h : IO.FS.Handle) (buf : ByteArray) (n : Nat) : IO ByteArray := do
  let target := buf.size + n
  let mut acc := buf
  while acc.size < target do
    let before := acc.size
    let want := USize.ofNat (target - before)
    acc ← readInto h acc want
    if acc.size == before then
      throw <| IO.userError s!"imagenet shim closed the pipe after {acc.size} of {target} bytes \
(did it crash? its stderr is not captured — run it standalone with SHIM_HASH=1 to see)"
  pure acc

/-- Read EXACTLY `n` bytes into a fresh buffer of exactly that capacity, allocated HERE — on the
    calling thread, which is what `readInto` is about. (This used to grow by `ByteArray.append`,
    which reallocates at double capacity: a 308 MB batch cost a 616 MB buffer and a full copy.) -/
def readExact (h : IO.FS.Handle) (n : Nat) : IO ByteArray :=
  readExactInto h (ByteArray.emptyWithCapacity n) n

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

/-- The stdio shape every shim child is spawned with. It has a name because `IO.Process.Child` is
    INDEXED by its config: without a concrete one the child cannot be stored in a structure, and
    without storing it the trainer holds a pipe it can neither kill nor reap. -/
abbrev ShimCfg : IO.Process.StdioConfig :=
  { stdin := .null, stdout := .piped, stderr := .inherit }

/-- A live shim producer: the child process AND its stdout pipe.

    ⚠ `spawnShim` used to return the handle alone and drop the child on the floor. Two things that
    costs: the validation shim left a `<defunct>` python for the life of every ImageNet run, and a
    producer could not be REPLACED mid-run — which is what a 350-epoch EfficientNet run needed after
    one of four loaders degraded and, under the round-robin read, paced the whole run at 1.8× its
    clean step time (`planning/shim_loader_health_and_resume_tests.md` §1). -/
structure ShimProc where
  child : IO.Process.Child ShimCfg
  h     : IO.FS.Handle

/-- Spawn the shim for one split and consume its preamble.

    The preamble (`LMSH` | version | batch | flat) is checked rather than skipped: a batch or
    resolution mismatch between the render and the shim would otherwise read as garbage pixels and
    look like a broken net. Same reasoning as the FFI's G4 arity guard. -/
def spawnShim (shimScript : String) (split : String) (batch flat seed : Nat)
    (shard : Option (Nat × Nat) := none) (nclasses : Nat := 0) : IO ShimProc := do
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
  -- ⚠ `stderr := .inherit` is the default and is spelled out anyway: `IO.Process.Child` is INDEXED
  -- by its stdio config, so the child `spawn` returns only has `ShimProc`'s field type when all
  -- three fields match `ShimCfg` syntactically.
  let child ← IO.Process.spawn {
    cmd := py.toString, args := #[script.toString],
    stdout := .piped, stdin := .null, stderr := .inherit,
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
  pure { child := child, h := h }

/-- One batch off the wire: `int32[batch]` labels then `float32[batch*flat]` images, in that order
    (the shim writes labels first so a partial record is detectable at the smaller read). -/
def readShimBatch (h : IO.FS.Handle) (batch flat : Nat) (nclasses : Nat := 0)
    (imgBuf : Option (IO.Ref ByteArray) := none) : IO (ByteArray × ByteArray) := do
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
  -- ⭐ The image buffer comes from the CALLER when there is one — the prefetch path allocates it on
  -- the main thread before spawning the task, see `readInto` for why that is the whole point. It
  -- is taken OUT of the ref (`swap`, not `get`) so this thread holds the only reference; `get`
  -- would leave a second one behind and `readInto` refuses a shared buffer. The labels stay a
  -- local allocation: at ≤ 4 MB they are below the size class the leak lives in (measured flat).
  let buf ← match imgBuf with
    | some r => r.swap ByteArray.empty
    | none   => pure (ByteArray.emptyWithCapacity (4 * batch * flat))
  let img ← readExactInto h buf (4 * batch * flat)
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
    (nclasses : Nat := 0) : IO (Array ShimProc) := do
  if n <= 1 then
    pure #[← spawnShim shimScript split batch flat seed none nclasses]
  else
    let mut hs : Array ShimProc := #[]
    for i in [0:n] do
      hs := hs.push (← spawnShim shimScript split batch flat (seed + i) (some (i, n)) nclasses)
    IO.println s!"  imagenet shim: {n} sharded producers (round-robin over batches)"
    pure hs

/-- Round-robin read: batch `k` comes from worker `k % n`. `readExact` already blocks until a whole
    record has arrived, so a slow worker throttles rather than corrupting — the framing cannot slip. -/
def readShimBatchRR (hs : Array ShimProc) (k batch flat : Nat) (nclasses : Nat := 0)
    (imgBuf : Option (IO.Ref ByteArray) := none) : IO (ByteArray × ByteArray) := do
  match hs[k % hs.size]? with
  | some p => readShimBatch p.h batch flat nclasses imgBuf
  | none   => throw <| IO.userError "readShimBatchRR: no shim producers were spawned"


/-- Where `evalScore` reads its rows from (planning/streaming_val.md §3.2). -/
inductive EvalRows where
  /-- The whole split held in RAM — MNIST, CIFAR, Imagenette: today's slicing, unchanged. -/
  | held (img lbl : ByteArray)
  /-- ImageNet: streamed per pass from `hs.size` batch-block producers (`spawnValStream`),
      `shimBatch` rows per shim batch, `flat` floats per image, read round-robin — global batch
      `k` from producer `(k + offset) % n`, where `offset` is 0 except under the gate's order
      fault. `dropTail` drops a partial final batch (`LEAN_MLIR_EVAL_BATCHSTATS`, or the gate's
      tail fault) instead of refusing the short pass. -/
  | stream (hs : Array ShimProc) (shimBatch flat offset : Nat) (dropTail : Bool)

/-- The streamed reader's carry. Shim batches arrive at `shimBatch` rows and an eval invoke wants
    `gB = R × evalBs` of them — one shim batch for ConvNeXt at 4 × 64, four for ViT at 4 × 256, a
    quarter of one at R = 1, bs 64 — so rows are carried across pulls. -/
structure ValCarry where
  img   : ByteArray := ByteArray.empty
  lbl   : ByteArray := ByteArray.empty
  rows  : Nat := 0
  next  : Nat := 0        -- the next global shim batch to read
  ended : Bool := false
  total : Nat := 0        -- rows delivered so far: the per-pass denominator check

/-- Pull the next `gB` rows off the stream: `(xb, lbl, real)` with `xb` zero-padded to `gB × flat`
    floats and `real ≤ gB` the rows to score; `real = 0` means the pass is over. The first `rows = 0`
    in round-robin order IS the end: every producer has emitted all of its blocks by then (producer
    `b % n` has exactly `⌊b / n⌋` of them when global batch `b` is the first past the end).
    Sequential by construction — `evalScore` issues the next pull only after awaiting this one — so
    the carry needs no lock. -/
def pullValRows (st : IO.Ref ValCarry) (hs : Array ShimProc) (shimBatch flat offset gB : Nat)
    (dropTail : Bool) : IO (ByteArray × ByteArray × Nat) := do
  let mut c ← st.get
  while c.rows < gB && !c.ended do
    let some p := hs[(c.next + offset) % hs.size]?
      | throw <| IO.userError "val stream: no producers were spawned"
    let (i, l, rows) ← readShimBatchPartial p.h shimBatch flat
    if rows == 0 then c := { c with ended := true }
    else if rows < shimBatch && dropTail then
      IO.println s!"  ⚠ dropping the {rows}-image val tail (LEAN_MLIR_EVAL_BATCHSTATS, or the gate's \
tail fault) — this pass's denominator is {c.total + c.rows}, not 50,000"
      c := { c with ended := true }
    else
      -- `++` into a pre-sized buffer never reallocates (loadData's 2026-08 lesson); size the carry
      -- for the rows one pull can hold before the remainder is carried.
      let img := if c.img.isEmpty then ByteArray.emptyWithCapacity (4 * (gB + shimBatch) * flat) ++ i
                 else c.img ++ i
      c := { c with img := img, lbl := c.lbl ++ l, rows := c.rows + rows, next := c.next + 1 }
  let real := min gB c.rows
  let xb ← if real == gB then pure (c.img.extract 0 (4 * gB * flat))
           else pure ((c.img.extract 0 (4 * real * flat)) ++ (← F32.const ((gB - real) * flat).toUSize 0.0))
  let lb := c.lbl.extract 0 (4 * real)
  st.set { c with img := c.img.extract (4 * real * flat) c.img.size,
                  lbl := c.lbl.extract (4 * real) c.lbl.size,
                  rows := c.rows - real, total := c.total + real }
  pure (xb, lb, real)

/-- Spawn the per-pass val producers (planning/streaming_val.md §3.2): `n` batch-block producers of
    the validation split at `shimBatch` rows — fresh for every pass and reaped by `reapValStream`, so
    nothing accumulates across epochs and the closed pipe stays the end-of-pass marker. `flat` is the
    EVAL width (RSB-A3 trains at 160² and evaluates at 224²), read off the artifact by the caller.

    ⚠ The gate-only fault knobs, in the `PJRT_FFI_FAULT` style — controls that must go red:
    `LEAN_MLIR_VAL_FAULT=order` starts the round-robin on producer 1 (every image scored against
    another image's label: same count, different bitmap), `=tail` drops the 80-row tail (the C4
    bug class: 49,920). `scripts/streamed_val_gate.sh`. -/
def spawnValStream (net : VerifiedNet) (flat : Nat) (n : Nat := 2) (shimBatch : Nat := 256) :
    IO EvalRows := do
  let hs ← spawnShimSharded net.shimScript "validation" shimBatch flat 0 n
  let fault ← IO.getEnv "LEAN_MLIR_VAL_FAULT"
  let offset := if fault == some "order" then 1 else 0
  let dropTail := (← IO.getEnv "LEAN_MLIR_EVAL_BATCHSTATS").isSome || fault == some "tail"
  if let some f := fault then
    IO.println s!"  ⚠ LEAN_MLIR_VAL_FAULT={f}: {if f == "order" then "the round-robin starts on producer 1" else if f == "tail" then "the partial last batch is dropped" else "unknown fault, no effect"} — the gate's CONTROL, not a configuration"
  IO.println s!"  ▸ val: STREAMED per pass — {hs.size} batch-block producer(s), {shimBatch}-row blocks \
k, k+{hs.size}, … read round-robin, nothing held (the 30 GB drain is gone; \
planning/streaming_val.md)"
  pure (.stream hs shimBatch flat offset dropTail)

/-- Kill then wait, the `<defunct>` lesson: the val stream ends by itself, but a dropped tail or a
    refusal can leave a child still writing. -/
def reapValStream : EvalRows → IO Unit
  | .stream hs .. => do
      for p in hs do
        try p.child.kill catch _ => pure ()
        let _ ← p.child.wait
  | _ => pure ()

/-- **The eval pass**: score `nEval` images through a forward session, `replicas × evalBs` per
    invoke. `(top-1 correct, top-5 correct, images scored, per-image top-1 bitmap)`. The bitmap is
    filled only when `wantBits`, in eval order, one byte per image. `rows` is where the images come
    from: the held split, or ImageNet's per-pass stream (`EvalRows`).

    ⭐ ONE copy, shared by the per-epoch eval in `trainAdamSched` and by `scoreCheckpoint`. That is
    not tidiness: `scripts/sharded_eval_gate.sh` compares 1 replica against N through
    `score-checkpoint`, and a gate on a COPY of the loop says nothing about the loop that runs.

    ⚠⚠ THE RAGGED TAIL is the first thing to get wrong. 50,000 is not a multiple of 4 × 64 = 256:
    it is 195 full invokes and an 80-image tail, which fills replica 0 and a quarter of replica 1,
    while replicas 2 and 3 score pure padding. `F32.sliceImagesPad` zero-pads the invoke to the
    global batch, the shim gathers the logits back in ROW ORDER, and only
    `min gB (nEval − bi·gB)` rows are scored. The pad rows are computed and never read. -/

def evalScore (sess : LowererSession) (fn : String) (params shapes : ByteArray) (rows : EvalRows)
    (nEval evalBs evalD0 nc replicas : Nat) (nResident gen : USize) (wantBits : Bool) :
    IO (Nat × Nat × Nat × ByteArray) := do
  let r := max replicas 1
  let gB := r * evalBs
  let xShape := packXShape #[gB, evalD0]
  let nbt := (nEval + gB - 1) / gB   -- ceil: the last partial invoke is zero-padded, not dropped
  let mut correct := 0
  let mut correct5 := 0
  let mut bits : ByteArray := ByteArray.empty
  -- The streamed source reads AHEAD: the pull for invoke k+1 is issued before invoke k runs, one in
  -- flight, so the ~30 GB of pipe reads per pass overlap compute instead of serialising with it
  -- (the train loop's depth-n prefetch, at depth 1). The task touches only the carry.
  let carry ← IO.mkRef ({} : ValCarry)
  let issue : BaseIO (Task (Except IO.Error (ByteArray × ByteArray × Nat))) :=
    match rows with
    | .stream hs sb flat off dt => IO.asTask (pullValRows carry hs sb flat off gB dt) Task.Priority.default
    | .held _ _ => IO.asTask (pure (ByteArray.empty, ByteArray.empty, 0)) Task.Priority.default
  let mut inflight : Option (Task (Except IO.Error (ByteArray × ByteArray × Nat))) := none
  if let .stream .. := rows then inflight := some (← issue)
  for bi in [0:nbt] do
    let mut xb := ByteArray.empty
    let mut lb := ByteArray.empty
    let mut real := 0
    let mut lblBase := 0
    match rows with
    | .held img lbl =>
        -- ⚠ `evalD0`, not the train width: the val buffer is at the EVAL width (RSB-A3 trains at
        -- 160², evaluates at 224²), and slicing it at the other one strides through it wrongly.
        xb := F32.sliceImagesPad img (bi * gB) gB evalD0 nEval
        lb := lbl; real := min gB (nEval - bi * gB); lblBase := bi * gB
    | .stream .. =>
        let some tk := inflight | throw <| IO.userError "val stream: no pull in flight"
        let (i, l, n) ← IO.ofExcept (← IO.wait tk)
        inflight ← if bi + 1 < nbt then pure (some (← issue)) else pure none
        xb := i; lb := l; real := n
    if real == 0 then break   -- the stream ended early; the count check below refuses
    let logits ← if r == 1
      then LowererSession.forwardF32 sess fn params shapes xb xShape gB.toUSize nc.toUSize
             nResident gen
      else LowererSession.forwardF32Dp sess fn params shapes xb xShape gB.toUSize nc.toUSize
             r.toUSize nResident gen
    for j in [0:real] do   -- score real rows only, not the pad
      let pred := (F32.argmaxN logits (j * nc).toUSize nc.toUSize).toNat
      let lbl  := F32.readLabel lb (lblBase + j)
      if pred == lbl then correct := correct + 1
      if wantBits then bits := bits.push (if pred == lbl then 1 else 0)
      -- top-5 by the label's RANK, matching the reference's `sum(logits > true_logit) < 5`.
      if (F32.rankOf logits (j * nc).toUSize nc.toUSize lbl.toUSize).toNat < 5 then
        correct5 := correct5 + 1
  -- ⚠⚠ `scored` IS NOT AN ACCUMULATOR. It was: `scored := scored + real` as the loop's last
  -- statement. Measured 2026-09-22 (Lean 4.34): on the R = 1 path that came back holding only the
  -- LAST invoke's rows (80 of 50,000; 256 with the tail dropped) while `correct`, mutated inside
  -- the inner loop, summed correctly — and it was right at R = 4, and right again on the same
  -- binary path the moment an `eprintln` read it after the assignment. That is the shape of a
  -- code-generation issue, not of this loop (a 20-line copy of the loop's shape sums correctly in
  -- isolation), so the denominator is read from what the pass delivered instead: the carry's
  -- `total` for a stream, `nEval` for a held split (the loop scores `min gB (nEval − bi·gB)` rows
  -- per invoke by construction). The bitmap's length cross-checks it whenever one is kept.
  let scored ← match rows with
    | .stream .. => do pure (← carry.get).total
    | .held _ _  => pure nEval
  if wantBits && bits.size != scored then
    throw <| IO.userError s!"eval scored {bits.size} rows but its source delivered {scored} — the two \
counts must agree, and neither is a number to report"
  -- ⭐ ASSERTED EVERY PASS, because it is the denominator every top-1 divides by and it moved once
  -- already (49,920 → 50,000, 2026-08-14). A short pass is a refusal, not a plausible number; an
  -- over-long one is refused too (one more pull must come back empty).
  match rows with
  | .stream hs sb flat off dt =>
      if scored == nEval then
        let (_, _, extra) ← pullValRows carry hs sb flat off 1 dt
        if extra != 0 then
          throw <| IO.userError s!"val stream delivered MORE than {nEval} images — the block sharding double-counted a batch"
        IO.println s!"  ▸ val = all {nEval} streamed (timm's denominator)"
      else if !dt then
        throw <| IO.userError s!"val stream delivered {scored} of {nEval} images — a short pass is a \
refusal, not a number (a producer died, or the block sharding lost a batch)"
  | _ => pure ()
  return (correct, correct5, scored, bits)

/-- Load the train + eval splits for a dataset. Returns
    `(trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop?)` where
    `trainPix` is the stored per-example width of the *training* images (256² for
    Imagenette, `d0` otherwise) and `crop?` requests the 256²→224² center-crop.

    `evalOnly` skips the TRAIN split entirely — for `scoreCheckpoint`, which never touches it.
    ⚠ It is INERT on `.imagenet`, whose train split was never preloaded (it streams off the shim),
    and that is exactly why it is worth having on the others: Imagenette's is 9,469 × 256² × 3 f32
    = **7.4 GB** read and held for a job that only scores 3,925 val images. `nTrain` comes back 0
    under it, so a caller that starts using it gets a division rather than a plausible epoch. -/
def loadData (net : VerifiedNet) (dataDir : String) (evalD0 : Nat := 0)
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
    -- ⭐ Val is NOT loaded either, since 2026-09-22. It used to be drained here into RAM once —
    -- 50,000 × 150,528 × 4 B = 30 GB held for the life of the run, nearly all of the trainer's RSS —
    -- and now streams per pass from batch-block producers (`spawnValStream` → `evalScore`), so those
    -- 28 GiB go back to the page cache the train split wants (planning/streaming_val.md §0).
    -- `nEval` is ImageNet's 50,000: the denominator every top-1 here divides by (timm's), and every
    -- pass asserts the stream delivered exactly that many.
    --
    -- ⚠⚠ THE VAL WIDTH IS NOT ALWAYS THE TRAIN WIDTH: RSB-A3 trains at 160² and evaluates at 224²,
    -- and `evalD0` — read off the eval artifact by the caller — is what the val producers are
    -- spawned at. `trainPix` stays `net.d0`, the width of the images the TRAIN stream carries.
    IO.println s!"  imagenet: val streams per pass (50,000 images at eval width {evalD0}) — nothing is held"
    return (ByteArray.empty, ByteArray.empty, 1281167, ByteArray.empty, ByteArray.empty, 50000, net.d0, false)

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

/-- Write `bytes` to `path` all-or-nothing: a sibling `.tmp`, then `rename(2)` over the target.
    A crash mid-write leaves the PREVIOUS file intact rather than a truncated one — the same
    guarantee the JAX reference's `save_train_state` gets from `os.replace`. -/
def writeBinAtomic (path : String) (bytes : ByteArray) : IO Unit := do
  let tmp := path ++ ".tmp"
  IO.FS.writeBinFile tmp bytes
  IO.FS.rename tmp path

/-- **Where this (net, variant) writes and resumes its checkpoint.**

    Scoped by BACKEND: without the suffix an XLA run would happily resume from an IREE checkpoint
    and vice versa, silently fusing two trajectories into one while looking completely normal on
    screen (`planning/archive/xla_pjrt_ladder.md` §3). `$LEAN_MLIR_CKPT_TAG` appends a run-scoped suffix —
    without it every pass of the same (net, variant, backend) shares ONE path, so the parallel
    sweeps `planning/archive/chapter_makeover.md` §3c mandates cannot be run: concurrent passes clobber
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
  -- / 100 steps separates them — see planning/archive/xla_pjrt_ladder.md §8.
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
    -- ▶ `wilson95` here too, not just in `trainAdamSched` (§2230): chapters 1-4 run through
    -- `VerifiedNet.train` and were the only tier printing a bare accuracy with no interval, so
    -- the book's MNIST/CIFAR rows could not be quoted the way its Imagenette rows are.
    -- ⚠ Placed after the percentage rather than at end-of-line because this print (unlike
    -- `trainAdamSched`'s) carries a trailing `(Nms)`; the statistic and its interval stay adjacent.
    IO.println s!"  epoch {ep + 1}: {lossField}{evalName}_acc = {correct}/{nEval} = {acc}%  [95% CI {wilson95 correct nEval}] ({epMs}ms)"
    (← IO.getStdout).flush
  -- Gate G2 (`planning/archive/xla_pjrt_ladder.md` §3): dump the packed params so the IREE
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
  -- "rms" = the RMSProp-with-momentum render (`Proofs/Training/Optim/RmsPropStep.lean`), which reuses the
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
  -- "ema"/"emadp" = the EMA-shadow render (`planning/archive/ema.md`), whose blob carries a FOURTH region:
  -- `[θ|m|v|ema]`, with the scalar tail 3 → 5 (`%emad`, `%oemad`). Everything below that indexes the
  -- blob is written against `nRegions`/`nScalars` rather than a literal 3, because a 4-region graph
  -- fed a 3-region blob is not a subtle numeric error — it is every parameter misaligned.
  --
  -- ⚠ Keyed off the variant PREFIX, the same reverse-of-`cnxAdamVariant` reading `rmsprop` uses,
  -- and pinned upstream by the `#guard`s beside that renderer's `#eval`s.
  let emaOn := VerifiedVariant.emaOn variant
  -- ⭐⭐ GRADIENT ACCUMULATION (`planning/archive/next_session_pipeline_then_r50.md` §4). "acc<k>x<B>" /
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
  -- ⭐⭐⭐ **THE REFUSAL IS GONE (2026-08-27), AND IT WAS THE LAST THING LIFTED.**
  --
  -- It read: *"variant selects BOTH the EMA shadow and gradient accumulation, and they occupy the
  -- same fourth region of [θ|m|v|·]. Render one or the other."* True when written, and it is what
  -- made RSB-A2 and RSB-A1 unrenderable — their recipe sets `useEMA := true` AND
  -- `gradAccumSteps := 4`, and accumulation is not optional at 224² on 16 GB cards. A3 met neither
  -- obstacle because A3's own recipe sets `useEMA := false`, which is exactly why the limitation
  -- was invisible from A3's success (`verified_side_quest_counterparts.md` §4a, §6a).
  --
  -- What replaced it: `G` and `E` are two INDEPENDENT regions in the order `[θ|m|v|G|E]`, so
  -- `nRegions` is 3, 4 or 5 and `nScalars` 3, 5 or 7. `VerifiedVariant.emaRegion` is where the
  -- shadow's index comes from — never the literal 3, which is what it was under the old layout.
  --
  -- ⚠⚠ **THE ORDER THIS WAS LIFTED IN IS LOAD-BEARING, and is recorded because the reverse is
  -- tempting.** While the throw stood, a wrong render failed LOUDLY at load. The moment it came
  -- off, an EMA-plus-accumulation graph whose regions are packed wrongly TRAINS and reports a
  -- number. So the fifth region landed in the renderer, in this driver's pack/unpack, in
  -- `TestVariantPredicates`' three-way partition and in `opt_step_tie.py`'s `emalambacc8wxclip`
  -- row — measured at 1.20e-07 against the reference's own `ema_update` — BEFORE this line was
  -- deleted, not after.
  let nRegions := VerifiedVariant.nRegions variant
  let nScalars := VerifiedVariant.nScalars variant
  -- "…drop" = the STOCHASTIC-DEPTH render (`planning/archive/stochastic_depth.md`): the graph takes one
  -- extra `tensor<Bxf32>` per drop site, carrying `bernoulli(keep_i)/keep_i` per example.
  --
  -- ⚠⚠ THE MARKER IS `"drop"` BECAUSE `"sd"` COLLIDES, and the collision is between two OTHER
  -- markers meeting: `rms` ++ `dp` spells **`rmsdp`**, which contains "sd". A `"sd"` substring test
  -- therefore fires on `rmsdp64` and `emarmsdp64` — every RMSProp data-parallel variant, including
  -- the committed and gated `efficientnetin_rmsdp64` — and would have appended 9 drop scales to a graph
  -- that takes none. Caught by running the predicate table (`tests/TestVariantPredicates.lean`)
  -- rather than reading names one at a time; with three markers the collisions are between PAIRS.
  -- This is `planning/archive/ema.md`'s `emarms` defect a second time, one axis further on.
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
  -- ⭐ PER-VARIANT forward resolution (`planning/archive/mnv4_verified.md` §3d(b)).
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
  -- ⚠ `mkSynthData` must be sized at the GLOBAL batch, not `bs`. Under data
  -- parallelism one step consumes `bs * replicas` images (the shim shards them),
  -- so a `bs`-sized synthetic buffer is read past its end every step: silent at
  -- bs32×2, a `free(): invalid next size` abort at bs128×2. Found 2026-07-30
  -- while measuring the parameter transfer share (handoff §2d.3). This is why
  -- `replicas` is read here and not with the other knobs below — and, since
  -- 2026-09-18, ABOVE the eval sessions, which are compiled at it.
  let replicas := ((← IO.getEnv "LEAN_MLIR_REPLICAS").bind (·.toNat?)).getD 1
  -- ⭐ THE EVAL IS SHARDED OVER THE SAME DEVICES AS THE TRAIN STEP (2026-09-18). It used to run on
  -- replica 0 alone. `grep -c all_reduce` is 0 on every `_fwd`/`_fwd_eval`, so the train steps
  -- were N-replica and the eval never was. Measured on the killed ConvNeXt run: ~100 s/epoch at
  -- 91% util on GPU 0 while GPUs 1-3 sat at 0%. Same artifacts, nothing re-rendered. The gate is
  -- `scripts/sharded_eval_gate.sh`: an identical correct count and bitmap at 1 and N replicas, with
  -- a control that must fail.
  let fwdSess ← mkSessionDp fwdPath replicas
  let fwdEvalSess ← if hasBn then
      mkSessionDp s!"{net.mlirDir}/{net.slug}_fwd_eval.mlir" replicas
    else pure fwdSess
  let synth := (← IO.getEnv "LEAN_MLIR_BENCH_SYNTH").isSome
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
    let batchOf (t : String) : Bool := t.contains "dimensions = [0, 2, 3]"
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
  -- param dump. See planning/archive/xla_pjrt_ladder.md §3.
  -- LEAN_MLIR_REPLICAS: data-parallel device count. The graph is rendered at the
  -- PER-REPLICA batch (cfg.batchSize), so one step consumes `bs * replicas`
  -- images and the shim splits them. Eval is sharded the same way since 2026-09-18 (`evalScore`):
  -- `replicas × evalBs` per invoke, logits gathered. See planning/archive/xla_pjrt_ladder.md §10.
  -- LEAN_MLIR_SKIP_EVAL: skip the per-epoch eval pass. It used to be REQUIRED whenever the train
  -- batch differed from the forward's baked one (bs256, bs128-DP); `evalBs` below removes that,
  -- so it is now just "don't spend the time".
  let skipEval := (← IO.getEnv "LEAN_MLIR_SKIP_EVAL").isSome
  -- LEAN_MLIR_VAL_EVERY: validate every n epochs (+ the last one this process runs). Opt-in
  -- override of `cfg.valEveryEpochs`; see the field. Unlike SKIP_EVAL this scores SOME epochs.
  let valEvery := match (← IO.getEnv "LEAN_MLIR_VAL_EVERY").bind (·.toNat?) with
    | some n => n
    | none   => cfg.valEveryEpochs
  let gbs := bs * replicas
  let nbFull := nTrain / gbs
  let nb := match (← IO.getEnv "LEAN_MLIR_G2_STEPS").bind (·.toNat?) with
    | some n => min n nbFull
    | none   => nbFull
  -- `evalBs`/`evalD0` are read off the eval artifact ABOVE, before `loadData` — see there.
  -- The schedule label is part of the run's evidence, not decoration: an exponential-decay run and
  -- a cosine one are different experiments and the log has to say which it was. Spelled so the
  -- string is UNCHANGED at the default (`expDecayRate = 0`), i.e. every existing log line still reads
  -- "(cosine+warmup Nep, baseLR L)".
  -- ⭐ `expDecayRate = 1.0` is an exactly CONSTANT rate: the decay branch computes
  -- `baseLR * exp(k * log 1.0) = baseLR * exp 0 = baseLR` at every step, and `warmupEpochs = 0`
  -- makes `warmSteps = 0 < gstep`, so the warmup branch never fires either. No new code path —
  -- but it needs its own NAME in the log, because "exp x1.000000/1.000000ep+warmup 0ep" is a
  -- true and unreadable description of a flat line. Chapter 4's optimizer levers run this way
  -- deliberately: comparing three update rules under a schedule compares four things.
  -- ⚠ The cosine and exp spellings are byte-identical to what they were, because every
  -- Imagenette and ImageNet transcript in the book quotes this line.
  let schedName := if expDecayRate == 1.0 then "constant lr"
    else if expDecayRate > 0.0 then s!"exp x{expDecayRate}/{expDecayEpochs}ep" else "cosine"
  let schedDesc := if expDecayRate == 1.0 then s!"constant lr {baseLR}"
    else s!"{schedName}+warmup {warmupEpochs}ep, baseLR {baseLR}"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} {variant} ({schedDesc}), He init"
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
    -- ⚠ It names the region INDEX, not "the 4th", because under accumulation it is the 5th — and a
    -- banner that says the wrong slot is worse than none when the failure mode is a mis-sliced blob.
    IO.println s!"  ▸ EMA: region {(VerifiedVariant.emaRegion variant).getD 3} of {nRegions} \
[θ|m|v{if accOn then "|G" else ""}|ema], shadow starts AT the weights, decay \
min({emaDecay}, (1+t)/(10+t)) — TF warmup-corrected. EVAL AND CHECKPOINT SCORE THE SHADOW.\
{if accOn then " ⚠ It advances on APPLY micro-batches only — once per optimizer step, as the reference does." else ""}"
  if accOn then
    -- ⚠⚠ IT ANNOUNCES ITSELF, and here that is not politeness either: a run with the wrong `k`
    -- prints an entirely normal loss curve at a silently wrong effective batch and learning rate.
    -- §6's rule — "a setting with no output and no gate is a setting that can be silently wrong".
    IO.println s!"  ▸ GRADIENT ACCUMULATION: k = {accK}, blob region 3 of {nRegions} \
[θ|m|v|G{if emaOn then "|ema" else ""}]. Micro-batch \
{gbs} x {accK} = EFFECTIVE BATCH {gbs * accK}. {nb} micro-batches/epoch = {nb / accK} updates/epoch; \
the LR schedule and Adam's bias correction run on UPDATES, the augmentation and the prefetch on \
micro-batches."
    -- ⚠⚠ CONDITIONAL, AND IT WAS NOT. This line fired on EVERY accumulation run, naming two
    -- absences unconditionally — including for `lambaccdp8x64wxclipbcebf16`, which ResNet50RenderB
    -- renders as `R34Opt.lambAccum 8` with `bce := true`. So on RSB-A3, the one recipe it exists to
    -- warn about, it asserted the exact opposite of the graph that was loaded. A warning that is
    -- always printed carries no information; one that is always printed AND sometimes false is
    -- worse, because the run log then reads as evidence for the wrong recipe.
    let missing := (if VerifiedVariant.lambOn variant then [] else ["LAMB"])
                ++ (if VerifiedVariant.bceOn  variant then [] else ["BCE-with-logits"])
    if missing.isEmpty then
      IO.println s!"     ▸ LAMB (per-tensor trust ratio) + BCE-with-logits: RSB-faithful optimizer \
and loss at this batch."
    else
      IO.println s!"     ⚠ This is AdamW at that batch, NOT rsb-faithful — \
{String.intercalate " and " missing} still absent (planning/archive/rsb_a3_r50_verified.md §2.3)."
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
  if hasBn then
    -- ▶ The decay is announced because it is otherwise INVISIBLE: it is eval-only, so a wrong
    -- value moves no loss curve and shows up only as a quietly depressed top-1. It was a hidden
    -- literal 0.99 until 2026-08-30 and disagreed with timm by 10× on R50 the whole time.
    -- ⚠ Prints the per-micro weight actually passed to `F32.ema` — the same `bnEmaWeight` call
    -- the loop makes, not a transcription — so an accumulation run says what it is really doing
    -- rather than what it was configured with.
    let bnMomShown := cfg.bnEmaWeight (if accOn then some accK else none)
    IO.println s!"  running-stats BN: {net.bnChannels.size} layers, {nBnStats} stat floats → eval via @{net.slug}_fwd_eval"
    IO.println s!"     decay {cfg.bnMomentum} (TF sense; = PyTorch/timm momentum {1.0 - cfg.bnMomentum}), \
new-batch weight {bnMomShown}{if accOn then s!" = 1 − {cfg.bnMomentum}^(1/{accK}), compensated for grad-accum" else ""}"
  if replicas > 1 then
    IO.println s!"  DATA-PARALLEL: {replicas} replicas x bs {bs} = global batch {gbs}, {nb} steps/epoch"
    -- ⚠ Announced with the TAIL, because the tail is where a sharded eval goes wrong.
    let egB := replicas * evalBs
    IO.println s!"  EVAL SHARDED: {replicas} replicas x {evalBs} = {egB} {evalName} images per invoke, \
{(nEval + egB - 1) / egB} invokes, last one {nEval - (nEval - 1) / egB * egB} real + \
{((nEval + egB - 1) / egB) * egB - nEval} pad"
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
                                -- the FOURTH and FIFTH regions: the gradient accumulator `G` and
                                -- the EMA shadow `E`, INDEPENDENT and in that order.
                                -- ⚠ The G4 gated interface counts destinations off THIS list, so
                                -- omitting one does not mis-walk the blob quietly — the shim refuses
                                -- ("returns 755 outputs, caller supplied 594 destinations").
                                ++ (if accOn then net.paramShapes else #[])
                                ++ (if emaOn then net.paramShapes else #[])
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
  let tsFn  := s!"m.{net.slug}_{variant}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  if cfg.vitInit then
    IO.println "  ▸ INIT: timm/DeiT (σ=0.02 weights, patch-embed on PyTorch Conv2d default)"
  -- ⚠ ANNOUNCED, because the run this flag exists for was killed over an init nobody could see
  -- from the log. The banner line further down says "He init" unconditionally; that is now a lie
  -- whenever either flag is set, so each says so here.
  if cfg.cnxInit then
    IO.println "  ▸ INIT: ConvNeXt _init_weights (σ=0.02 on every conv AND the head; biases 0, LN γ 1, LayerScale γ 1e-6)"
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2 cfg.vitInit (cnxInit := cfg.cnxInit))
    seed := seed + 1
  -- LEAN_MLIR_PERTURB_R: displace the initial parameters along a random unit
  -- vector of exact L2 norm r, before any training. This is the CONDITIONING
  -- probe for gate G2 (planning/archive/xla_pjrt_ladder.md §8, rung 3): if an r that is
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
  -- `planning/archive/ema.md` records: a shadow evaluated at chance on short runs). The gradient
  -- ACCUMULATOR starts at ZERO — and it would be harmless at any value, because `%akeep = 0` on
  -- the first micro-batch of every cycle discards whatever is there. Zero anyway, so a checkpoint
  -- written mid-cycle resumes from something meaningful rather than from a stale partial sum.
  -- ⚠⚠ **`if emaOn then … else if accOn then …` UNTIL 2026-08-27, i.e. an EITHER/OR** — which is
  -- what made RSB-A2/A1 unrenderable, since their recipe wants both. Now two independent regions in
  -- the order `[θ|m|v|G|E]`, and the order is what keeps every previously-written blob readable:
  -- at `acc` alone `G` is still region 3, at `ema` alone `E` is still region 3.
  let mut thetamv := F32.concat (#[theta, zeros, msInit] ++
    (if accOn then #[zeros] else #[]) ++ (if emaOn then #[theta] else #[]))
  let mvBytes := nRegions * net.nParams * 4
  let pBytes := net.nParams * 4
  -- Running BN stats (EMA of per-layer batch mean/var; mom 1.0 on the first step to seed,
  -- then 0.1). Rebuilt from nothing within ~1/mom steps, so for these alone a per-process reset
  -- would be harmless — their SHADOW below is not, and the two are checkpointed together.
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  -- The EMA shadow of those buffers (`ema_bn`). Starts where they start, as the reference does
  -- (`ema_bn = bn_state`). ▶ Both are CHECKPOINTED since 2026-09-12, as `<ckpt>.bn` beside the
  -- blob (see the epoch-end write). ⛔ Before that a resume restarted this shadow at zero under the
  -- mature decay, and that is NOT "rebuilt within an epoch": at 0.9999 it is a 10,000-step filter.
  let mut emaBnStats ← F32.const nBnStats.toUSize 0.0
  let mut bnFirst := true
  -- Steps on which `ema_bn` COPIES the running stats instead of averaging them. Zero except after
  -- resuming a checkpoint with no `.bn` companion (one written before it existed): the running
  -- stats rebuild in ~100 steps, and seeding the shadow from them is a far better estimate than
  -- blending in from zero at d = 0.9999.
  let mut emaBnSeedLeft : Nat := 0
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
    -- ▶ The BN companion (see the epoch-end write). ABSENT is a warning, not a refusal: every
    -- checkpoint written before 2026-09-12 lacks one, and for those the seed below is the fallback.
    -- A WRONG SIZE is a refusal, for the blob's reason — a companion from another layout would
    -- misalign every statistic and still print a plausible accuracy.
    if hasBn then
      let bnPath := ckptPath ++ ".bn"
      if ← System.FilePath.pathExists bnPath then
        let bn ← IO.FS.readBinFile bnPath
        if bn.size != 2 * nBnStats * 4 then
          throw <| IO.userError s!"BN companion {bnPath} is {bn.size} bytes but this run wants \
{2 * nBnStats * 4} (2 x {nBnStats} stat floats x 4). Move it aside with its checkpoint."
        runningBnStats := bn.extract 0 (nBnStats * 4)
        emaBnStats := bn.extract (nBnStats * 4) (2 * nBnStats * 4)
        bnFirst := false
        IO.println s!"  ▸ resumed BN running stats + ema_bn from {bnPath} (hash {bn.hash})"
      else
        emaBnSeedLeft := 100
        IO.println s!"  ⚠ no BN companion at {bnPath} (checkpoint predates 2026-09-12): ema_bn is \
re-seeded from the running stats over the first 100 steps, not restored"
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
  -- hide (planning/archive/next_session_pipeline_then_r50.md §2).
  --
  -- ⚠ It changes what `scripts/residency_gate.sh` feeds an ImageNet net — from a seeded real
  -- stream to one constant batch. Both are deterministic, which is all that gate's bit-identity
  -- verdict needs, but a constant batch is less numerically varied, so re-confirm the FAULT
  -- control fires before trusting a green from it.
  -- The base seed for the producers. Bound rather than read inline because a REPLACEMENT loader
  -- has to derive its own from it (`shimSeed + slot + generation × n`).
  let shimSeed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  let mut imgStreams : Array ShimProc ←
    if net.data == .imagenet && !synth then
      -- ⚠⚠ `net.d0`, NOT `3 * 224 * 224`. This is the width the TRAIN shim is *told* to emit, and
      -- it is the SECOND of two hardcoded 224s that had to fall for the 160 net — the other was
      -- `loadData`'s `trainPix` (which sizes the READ). Both had to agree with the render, and a
      -- literal here agreed with only the 224 ones: measured 2026-08-06 as
      -- "shim sends batch=64 flat=76800, the render wants batch=64 flat=150528".
      -- ▶ INERT for every incumbent — `LeanMlir/VerifiedNetsCore.lean`'s closing `#guard` block proves
      -- `net.d0 == 3*224*224` for all six 224 ImageNet nets, so this substitutes equal for equal
      -- there and changes only `resnet50in160`.
      spawnShimSharded net.shimScript "train" gbs net.d0 shimSeed shimWorkers shimNC
    else pure #[]
  -- ▶ `LEAN_MLIR_SHIM_RESPAWN_EPOCHS=E` (default 0 = off): every E epochs ONE producer is killed
  -- and replaced, cycling through the slots, so no loader lives past `E × n` epochs and at most one
  -- is ever cold. It exists because a single tf.data loader degrades after hours of uptime and the
  -- round-robin read then runs the whole job at its pace — 690 → 1,250 s/epoch on the 350-epoch
  -- EfficientNet run, cleared instantly by a restart (`planning/shim_loader_health_and_resume_tests.md`).
  -- ⚠ This is the BLOCKING form: the replacement is spawned at the epoch boundary and the trainer
  -- waits out its startup (~30 s, i.e. ~0.4% at E=10). §3d of that doc has the zero-downtime
  -- variant — spawn in the background, swap when its preamble lands — which is worth building only
  -- if this proves too coarse.
  -- ⚠ ANNOUNCED, like every other knob here: a run whose producers are being replaced under it and
  -- says so nowhere is indistinguishable in the log from one that is not.
  let respawnEvery := ((← IO.getEnv "LEAN_MLIR_SHIM_RESPAWN_EPOCHS").bind (·.toNat?)).getD 0
  if respawnEvery > 0 && !imgStreams.isEmpty then
    IO.println s!"  ▸ SHIM RESPAWN: one producer every {respawnEvery} epoch(s), round-robin over \
{imgStreams.size} — no loader lives past {respawnEvery * imgStreams.size} epochs."
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
  -- ⚠ ANNOUNCED, for the reason every other silent throughput/recipe flag here is: an ablation
  -- arm that trains without augmentation and says nothing is indistinguishable in the log from
  -- the full recipe, and the two differ by 7 points.
  let noAug := (← IO.getEnv "LEAN_MLIR_NO_AUG").isSome
  if noAug then
    IO.println "  ▸ NO AUGMENTATION (LEAN_MLIR_NO_AUG): centre crop, no random crop, no hflip \
— the ablation arm. The rendered train step is unchanged."
  let probeSteps := (← IO.getEnv "LEAN_MLIR_MAX_STEPS").bind (·.toNat?)
  -- LEAN_MLIR_PROBE_WARM: the step the probe clock STARTS at. Default 8 preserves every
  -- committed number. ⛔ 8 IS TOO EARLY TO BE A PRODUCTION RATE. `SHIM PREFETCH` keeps one
  -- read in flight PER HANDLE (depth = SHIM_WORKERS = 8), and the producers fill those while
  -- the graph compiles and while the ~90 s val drain runs — so the first ~8-16 steps are served
  -- from a queue nobody had to wait for. They measure BURST rate, not production rate. A window
  -- starting at 8 with `LEAN_MLIR_MAX_STEPS=40` is 32 samples of which ~16 are burst, which puts
  -- the median exactly on the boundary: that is how §9.6's ViT row got 159 ms/step where the
  -- steady state is 375 (2.4×). Benchmarks want PROBE_WARM=200 with MAX_STEPS=600.
  let probeWarm := ((← IO.getEnv "LEAN_MLIR_PROBE_WARM").bind (·.toNat?)).getD 8
  let mut probePrev := 0
  -- LEAN_MLIR_PROBE_DUMP also turns on a per-read trace to stderr (`READ h= step= issued= start=
  -- end=`), one line per prefetched batch, from the pool thread: issued→start is queueing in the
  -- task pool, start→end is the transfer off the pipe. Together with the dump's `wait_ms` this
  -- is enough to say which side a slow handle is slow on.
  let probeTrace := (← IO.getEnv "LEAN_MLIR_PROBE_DUMP").isSome
  let mut probeTimes : Array Nat := #[]
  let mut probeWaits : Array Nat := #[]   -- ms spent in `IO.wait` on the prefetched batch, per step
  let mut probeIssues : Array Nat := #[]  -- ms spent issuing the next reads (buffer alloc + spawn)
  let mut probeInvokes : Array Nat := #[] -- ms in the train-step invoke itself
  let mut lastWaitMs : Nat := 0
  let mut lastIssueMs : Nat := 0
  let mut lastInvokeMs : Nat := 0
  -- LEAN_MLIR_MAX_EPOCHS: same opt-in cap as `VerifiedNet.train` (absent → full run).
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  -- Build the reusable step buffer once, AFTER any checkpoint resume has settled
  -- `thetamv`. The scalar slots are filled per step; the BN region per step too.
  let scalarSlots ← F32.const nScalars.toUSize 0.0
  -- The drop-scale slots are reserved here and refilled per step, exactly like the scalar and BN
  -- regions — a fresh `F32.concat` per step would cost two whole-blob host memcpys (the mistake
  -- `planning/archive/xla_pjrt_ladder.md` §8 measured at 272 MB/step on R34).
  -- ⚠ Sized for BOTH families. The `1.0` fill is load-bearing and not a placeholder: a mask slot
  -- that is never refilled must be the exact identity, which `1.0` is and `0.0` emphatically is not
  -- (it would zero the classifier's input and train nothing).
  let dropSlots ← F32.const (nDrop * gbs + (if cdOn then gbs * doWidth else 0)).toUSize 1.0
  pbuf := if hasBn
          then F32.concat #[thetamv, scalarSlots, runningBnStats, dropSlots]
          else F32.concat #[thetamv, scalarSlots, dropSlots]
  -- ▶▶ DEPTH-1 PREFETCH of the shim read — planning/archive/next_session_pipeline_then_r50.md §2.
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
R34/ImageNet 4×bs64 for depth 1. Batch buffers are allocated and freed on the main thread; the \
pool only fills them."
      else "  ▸ SHIM PREFETCH: OFF (LEAN_MLIR_PREFETCH=0) — the read blocks the step. This is the \
gate's control, not a configuration.")
  -- One in-flight read per producer handle, indexed BY HANDLE (`step % nStreams`), so the slot a
  -- wait frees is exactly the slot its refill goes into. `none` = that handle has no read pending:
  -- true for every slot on the first step, and for the tail slots at the end of the run.
  let mut inflight : Array (Option (Task (Except IO.Error (ByteArray × ByteArray)))) :=
    Array.replicate (max 1 imgStreams.size) none
  -- Bumped once per respawn; it picks the slot (round-robin) and seeds the replacement, so two
  -- generations of the same shard never draw the same augmentation sequence.
  let mut shimGen : Nat := 0
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
      -- memcpys per step at R34 scale — see planning/archive/xla_pjrt_ladder.md §8.
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
        -- ⚠⚠ **THE SHADOW MOVES ONCE PER OPTIMIZER STEP, NOT ONCE PER MICRO-BATCH**, and under
        -- accumulation those differ by a factor of `k`. The reference EMAs after the `train_step`
        -- call (`jax/Jax/Codegen.lean:3017`) and JAX's accumulation lives INSIDE that call, so one
        -- `ema_update` covers all k micro-batches. This driver invokes the graph per micro-batch,
        -- so on an accumulate micro-batch it must hand the graph the IDENTITY: `%emad = 1`,
        -- `%oemad = 0` gives `e' = 1·e + 0·θ' = e` exactly.
        -- ▶ Not merely a k× faster filter if got wrong: θ is FROZEN on accumulate micro-batches
        -- (`%lr = 0`), so k−1 of every k updates would pull the shadow toward a weight that had not
        -- moved — a different, slower filter that still descends and still prints a curve.
        -- ⭐ Arithmetic, not a branch, and the same trick `%aup` plays on the moments: one graph,
        -- one compile, no way for an "accumulate" and an "apply" render to drift.
        let (ed, oed) := if applyNow then (emaD, 1.0 - emaD) else (1.0, 0.0)
        let emaPair ← F32.const 3 0.0
        let emaPair ← F32.write3 emaPair 0 ed oed 0.0
        -- ⚠ `emaScalarOff`, not the literal 3: behind `%aup`/`%akeep` the pair starts at slot 5.
        pbuf ← F32.blit pbuf
                 (nRegions * net.nParams + VerifiedVariant.emaScalarOff variant).toUSize emaPair 0 2
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
        -- ▶ INERT for the six 224 nets by `VerifiedNetsCore.lean`'s closing `#guard` block.
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
          -- — 150,120 of them over a 30-epoch run. Pooled lands 5 ms above the 219 ms synth floor.
          -- ⚠ Both numbers were taken with the leak below in place. `.dedicated` would also have
          -- hidden it by accident — a thread that exits abandons its heap, which the next free
          -- reclaims — at the price of refaulting the whole buffer every step (standalone repro,
          -- 2026-09-11).
          -- ⚠ The step index runs unbroken across the epoch boundary — `ep * nb + bi + 1` at the
          -- end of epoch e is exactly `ep' * nb + 0` for e+1 — which is what keeps the round-robin
          -- continuous. The train iterator `.repeat()`s inside the shim and never ends, so there
          -- is no per-epoch restart to resynchronise against.
          let s := ep * nb + bi
          let nStr := inflight.size
          let slot := s % nStr
          -- ⭐⭐ THE BATCH BUFFER IS ALLOCATED HERE, on the main thread, and handed to the task
          -- through a ref; the task only fills it (`readInto`). Letting the task allocate it —
          -- which is what `Handle.read` does — made a pool thread the owner of a 308 MB block the
          -- main thread then freed, and the runtime's allocator (mimalloc) answers a cross-thread
          -- free of a huge block with `madvise(MADV_FREE)`, not a release: the pages stay in RSS
          -- until the kernel is under pressure. 79–136 MB/step, the box full inside one epoch,
          -- then continuous direct reclaim and a mean step 2.5× the median
          -- (runs/2026-09-11-vit-leak-ab). Allocated and freed by the same thread, the block is
          -- recycled in place: no growth, and no fresh page faults after the first step.
          -- ⚠ None of the allocator's environment knobs reach this: `MALLOC_*` is glibc, which is
          -- not the allocator, and `MIMALLOC_PURGE_DELAY=0` only helps while the owning thread
          -- keeps allocating — at a 220 ms step cadence it changes nothing (measured).
          let issueRead (sj : Nat) : BaseIO (Task (Except IO.Error (ByteArray × ByteArray))) := do
            let buf ← IO.mkRef (ByteArray.emptyWithCapacity (4 * gbs * flat))
            let tIssue ← IO.monoMsNow
            IO.asTask (do
                let t0 ← IO.monoMsNow
                let r ← readShimBatchRR imgStreams sj gbs flat shimNC (some buf)
                if probeTrace then
                  let t1 ← IO.monoMsNow
                  IO.eprintln s!"READ h={sj % nStr} step={sj} issued={tIssue} start={t0} end={t1}"
                pure r)
              Task.Priority.default
          let t ← match inflight[slot]! with
            | some t => pure t
            | none   => issueRead s
          let w0 ← IO.monoMsNow
          let r ← IO.wait t
          lastWaitMs := (← IO.monoMsNow) - w0
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
          let is0 ← IO.monoMsNow
          for j in [1:pfDepth+1] do
            let sj := s + j
            if sj < nEpochs * nb then
              let sl := sj % nStr
              if (inflight[sl]!).isNone then
                inflight := inflight.set! sl (some (← issueRead sj))
          lastIssueMs := (← IO.monoMsNow) - is0
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
        -- ⭐ LEAN_MLIR_NO_AUG: the augmentation ABLATION arm, and the reason it is a flag rather
        -- than a second net is that augmentation lives in the data pipeline, not the graph —
        -- turning it off must leave the rendered train step byte-identical, or the arm would be
        -- measuring two things. It substitutes the DETERMINISTIC centre crop for the random one
        -- and drops the flip, which is exactly the eval-time pipeline; the images stay 224² so
        -- every downstream shape is unchanged.
        -- ⚠ Read ONCE per run, not per step: `IO.getEnv` in the batch loop would be a syscall
        -- 295 times an epoch, and the answer cannot change mid-run.
        let x ← match net.data with
          | .imagenette =>
              if noAug then
                if crop then F32.centerCrop xbRaw gbs.toUSize 3 256 256 224 224 else pure xbRaw
              else do
                let c ← if crop then F32.randomCrop xbRaw gbs.toUSize 3 256 256 224 224 augSeed
                        else pure xbRaw
                F32.randomHFlip c gbs.toUSize 3 224 224 (augSeed + 7777)
          | .cifar => if noAug then pure xbRaw
                      else F32.randomHFlip xbRaw gbs.toUSize 3 32 32 augSeed
          | _ => pure xbRaw
        xb := x; yb := if synth then curLbl else F32.sliceLabels curLbl (bi * gbs) gbs
      let inv0 ← IO.monoMsNow
      let out ← if replicas > 1
        -- ⚠ `nDrop` is the SHARDED TAIL. The drop masks are per-EXAMPLE, so under data parallelism
        -- replica r must get mask rows [r*bs, (r+1)*bs) — the same split `x` gets — not a copy of
        -- replica 0's. They ride in the parameter blob (`dropShapes` above), which is exactly why
        -- they were being replicated: the DP shim's rule was "x and the labels shard, everything
        -- between them replicates". `planning/archive/stochastic_depth.md` §5b predicted this; it was true
        -- of the shim before any DP drop render existed to expose it. At `nDrop = 0` the argument
        -- is inert and every non-SD DP run is byte-identical to before.
        then LowererSession.mlpTrainStepVDP tsSess tsFn xb pbuf adamShapes yb
               gbs.toUSize d0.toUSize nc.toUSize replicas.toUSize nResident nShardTail.toUSize
        else LowererSession.mlpTrainStepV tsSess tsFn xb pbuf adamShapes yb
               bs.toUSize d0.toUSize nc.toUSize nResident
      lastInvokeMs := (← IO.monoMsNow) - inv0
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
        --
        -- ⚠⚠ **AND THE DECAY IS NOW `cfg.bnMomentum`, NOT A LITERAL 0.99** (2026-08-30). 0.99 is
        -- TF's EfficientNet value; timm's PyTorch BN default gives R50/R34 a decay of 0.9, a
        -- 10-step averaging window against this 100-step one. Per-net table + the timm audit are
        -- in `TrainConfig.bnMomentum`'s docstring; `VerifiedConfig.bnMomentum` is the peer field,
        -- and `bnEmaWeight` — shared with the startup banner — is where both branches live.
        let bnMom := cfg.bnEmaWeight (if accOn then some accK else none)
        runningBnStats ← F32.ema runningBnStats batchBn (if bnFirst then 1.0 else bnMom)
        -- ▶ `ema_bn` — the BN running buffers get their OWN shadow, and on a batch-BN net this is
        -- not optional decoration. The reference's own words: eval pairs EMA weights with
        -- EMA-LAGGED stats, "avoiding the weights/stats mismatch that blows up early eval". EMA
        -- weights are a average of many steps' parameters; the LIVE running stats describe only the
        -- most recent steps' activations, and the two do not describe the same network.
        -- ⚠ Same `emaD` as the parameter shadow — one definition of the decay per step. `F32.ema`
        -- takes the NEW-value weight, so it is `1 − d`.
        -- ⚠ GATED ON `applyNow` for the parameter shadow's reason, one line up in the reference:
        -- `ema_bn = ema_update(ema_bn, bn_state, _global_step)` sits beside `ema_params`' update and
        -- fires on the same cadence. The running stats themselves DO move per micro-batch — that is
        -- what `bnMom`'s k-th-root compensation above is for — but their shadow does not.
        if emaOn && applyNow then
          -- Weight 1.0 = copy: only for the first 100 steps after resuming a checkpoint that has no
          -- `.bn` companion (see `emaBnSeedLeft`). Zero on every fresh run and every full resume.
          emaBnStats ← F32.ema emaBnStats runningBnStats
            (if emaBnSeedLeft > 0 then 1.0 else 1.0 - emaD)
          if emaBnSeedLeft > 0 then emaBnSeedLeft := emaBnSeedLeft - 1
        bnFirst := false
      pbuf := out   -- no copy: the output buffer becomes the next step's input
      -- ms/step probe: start the clock past warmup, report + exit at the cap.
      match probeSteps with
      | some ps =>
        if bi == probeWarm then probePrev := (← IO.monoMsNow)
        else if bi > probeWarm && bi ≤ ps then
          let t ← IO.monoMsNow
          probeTimes := probeTimes.push (t - probePrev); probePrev := t
          probeWaits := probeWaits.push lastWaitMs
          probeIssues := probeIssues.push lastIssueMs
          probeInvokes := probeInvokes.push lastInvokeMs
          if bi == ps then
            -- robust: median per-step time (drops the cold-cache / GC-blip outliers)
            let sorted := probeTimes.qsort Nat.blt
            -- ⭐ The SPREAD is the diagnostic, not the median. A compute-bound step is tight
            -- (min ≈ median); a SHIM-STARVED one is not — the min is what the step costs when the
            -- batch happened to be ready, so `median - min` is the wait. Print both, plus p90, so
            -- a slow kernel and a slow producer are distinguishable from ONE run instead of
            -- needing the synthetic arm to tell them apart.
            let pmin := sorted[0]!
            let pmed := sorted[sorted.size / 2]!
            let p90  := sorted[(sorted.size * 9) / 10]!
            let psum := probeTimes.foldl (· + ·) 0
            IO.println s!"  PROBE: {pmed} ms/step (median of {sorted.size} steps {probeWarm+1}..{ps}, {net.name})"
            IO.println s!"  PROBE-SPREAD: min={pmin} med={pmed} p90={p90} mean={psum / sorted.size} ms/step (starvation wait = med-min = {pmed - pmin} ms)"
            -- ⭐ LEAN_MLIR_PROBE_DUMP=<file>: the per-step series behind those four numbers, one
            -- `step<TAB>ms<TAB>wait_ms<TAB>issue_ms<TAB>invoke_ms` line each, in step order
            -- (`wait_ms` = blocked on the prefetched batch, `issue_ms` = allocating + spawning the
            -- next reads, `invoke_ms` = the train step itself; the first two are 0 off ImageNet). The summary cannot say WHICH steps are
            -- the slow ones, and with a round-robin over n producers "every n-th step" is the tell
            -- for one slow producer that no quantile shows.
            if let some path ← IO.getEnv "LEAN_MLIR_PROBE_DUMP" then
              let lines := probeTimes.mapIdx fun i ms =>
                s!"{probeWarm + 1 + i}\t{ms}\t{probeWaits[i]?.getD 0}\t{probeIssues[i]?.getD 0}\t{probeInvokes[i]?.getD 0}"
              IO.FS.writeFile path (String.intercalate "\n" lines.toList ++ "\n")
              IO.println s!"  PROBE-DUMP: {probeTimes.size} steps → {path}"
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
    -- ⚠⚠ **`emaRegion`, NOT THE LITERAL 3.** Under accumulation the shadow is region FOUR, because
    -- `G` takes three — and a stale literal here does not fail, it scores the GRADIENT ACCUMULATOR
    -- as if it were weights and prints a plausible-looking percentage off it.
    let emaReg := (VerifiedVariant.emaRegion variant).getD 0
    let thetaCur := thetamv.extract (emaReg * pBytes) ((emaReg + 1) * pBytes)
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
    -- ▶ `LEAN_MLIR_DUMP_CORRECT=<prefix>` writes one byte per validation image, 1 = top-1 correct,
    -- in eval order, to `<prefix>_e{N}.bin`. Unset ⇒ not accumulated and not written, so the
    -- default path is byte-identical.
    --
    -- ⭐⭐ **WHY A BITMAP AND NOT JUST THE SCALAR.** Two models scored on the SAME fixed validation
    -- set are a PAIRED comparison, and independent confidence intervals throw away almost all of
    -- the information in it. `wilson95` puts ±1.11 pt on a single Imagenette number, which makes
    -- most interesting gaps look like noise; McNemar's test over these bitmaps looks only at the
    -- images where the two models DISAGREE, and resolves the same gap comfortably. Worked example:
    -- R50-A3's headline 77.43 vs the reference's 77.22 is 0.21 pt — **0.79σ** as an unpaired
    -- comparison, i.e. not significant as stated — but it is 105 net label flips out of 50,000,
    -- which McNemar calls significant as long as the two models agree on ≳94% of images. The
    -- scalar cannot answer that question and the bitmap can, at 50 KB per eval.
    -- ⚠ It is a measurement about these two TRAINED MODELS, not about the recipe: "does this
    -- architecture change help" is a statement about the seed distribution and still needs n runs.
    let dumpCorrect := (← IO.getEnv "LEAN_MLIR_DUMP_CORRECT")
    -- Hold the eval parameters on device across the eval batches (§2d.3), one set per replica. The
    -- count is the tensor count of `evalShapes`, which for a BN net is the params PLUS the two
    -- running-stat slots per layer — all of them are inputs with no output counterpart, so all of
    -- them can be held. `gen := ep + 1` re-seeds them every epoch.
    -- `valEvery`: this epoch is scored if it is on the cadence or the last one this process
    -- runs (`nEpochs` is the MAX_EPOCHS-capped count, so a capped probe still gets its eval).
    let evalThisEpoch := valEvery ≤ 1 || (ep + 1) % valEvery == 0 || ep + 1 == nEpochs
    let (correct, correct5, nScored, correctBits) ← if skipEval || !evalThisEpoch then pure (0, 0, nEval, ByteArray.empty)
      else do
        -- ImageNet streams its val per pass (fresh producers, reaped after); the rest hold it.
        let rows ← if net.data == .imagenet then spawnValStream net evalD0
                   else pure (.held evalImg evalLbl)
        let r ← evalScore evalSess evalFn evalParams evalShapes rows
             nEval evalBs evalD0 nc replicas evalResident (ep + 1).toUSize dumpCorrect.isSome
        reapValStream rows
        pure r
    let acc := correct.toFloat / nScored.toFloat * 100.0
    let acc5 := correct5.toFloat / nScored.toFloat * 100.0
    -- ⚠ Under `LEAN_MLIR_SKIP_EVAL` the loop above runs ZERO batches, so `correct` is 0 and this
    -- line printed `acc = 0/49920 = 0.000000%  top5 = 0/49920` — a number INDISTINGUISHABLE from a
    -- catastrophically broken net, on a run that scored nothing. Found 2026-08-05 on R50's first
    -- smoke, where it read as the new net being wrong. Exact zeros on BOTH top-1 and top-5 are the
    -- tell (chance at 1000 classes is ~50 and ~250), but a reader should not have to notice that.
    if skipEval then
      IO.println s!"  epoch {ep + 1}: eval SKIPPED (LEAN_MLIR_SKIP_EVAL) — no accuracy was measured"
    else if !evalThisEpoch then
      -- Same shape as the SKIP_EVAL line so nobody reads a 0/50000 off a skipped epoch.
      IO.println s!"  epoch {ep + 1}: eval skipped (valEveryEpochs = {valEvery}; next scored epoch {min nEpochs (((ep + 1) / valEvery + 1) * valEvery)})"
    else
      -- ⚠ The CI is APPENDED, never woven into the existing fields: `blueprint/src/content.tex`
      -- quotes these lines verbatim and every `runs/*/` log is read by eye against that format.
      IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nScored} = {acc}%  top5 = {correct5}/{nScored} = {acc5}%  [95% CI {wilson95 correct nScored}]"
      -- ⚠⚠ `{variant}` is in the name, not just `{pfx}_e{N}`. `cifar8w-bn-ablation` and
      -- `cifar8w-ablation` each run THREE optimizer arms in one process (sgd/mom/adam), so a
      -- variant-less name has arm 2 overwrite arm 1 and arm 3 overwrite arm 2 — two thirds of
      -- the bitmaps silently lost, with the surviving file labelled as if it were the run.
      -- The checkpoint path has carried `variant` all along (`<slug>_<variant>_ckpt_xla.bin`);
      -- this brings the bitmap into line. ▶ Single-arm trainers pass "adam", so their files
      -- move `<pfx>_e80.bin` -> `<pfx>_adam_e80.bin`; `scripts/mcnemar.py` takes explicit
      -- paths and does not care, but bitmaps written before 2026-08-31 use the old name.
      match dumpCorrect with
      | some pfx =>
          IO.FS.writeBinFile s!"{pfx}_{variant}_e{ep + 1}.bin" correctBits
          IO.println s!"    per-example top-1 bitmap -> {pfx}_{variant}_e{ep + 1}.bin ({correctBits.size} bytes)"
      | none => pure ()
    (← IO.getStdout).flush
    -- ⛔ WRITE-THEN-RENAME, AND THE BN COMPANION (2026-09-12). This was two in-place writes, so a
    -- crash or power cut INSIDE the blob write left a truncated file that the size guard at resume
    -- then refuses on every restart — the supervisor burns all its attempts on a run that is not
    -- coming back — and a crash between the two writes left the marker one epoch behind the
    -- weights. `writeBinAtomic` makes each file all-or-nothing, and the order (companion, blob,
    -- marker) means the marker only advances once the state it names is on disk.
    -- ▶ `<ckpt>.bn` = [running BN stats | their EMA shadow]. Without it a resume restarted `ema_bn`
    -- at ZERO under the MATURE decay — `emaD` is keyed off the global step, so the warmup
    -- correction that rescues a fresh run is long spent — and at 0.9999 over 5,004 steps/epoch,
    -- 0.9999^5004 ≈ 61% of the eval's BN statistics were still that zero one epoch later, ~10
    -- epochs to wash out. The weights resumed exactly; the number scored off them did not. The JAX
    -- reference's `save_train_state` carries both (`ema_bn`, `bn_state`).
    if hasBn then
      let bn := F32.concat #[runningBnStats, emaBnStats]
      writeBinAtomic (ckptPath ++ ".bn") bn
      IO.println s!"    BN companion -> {ckptPath}.bn ({nBnStats} floats x 2, hash {bn.hash})"
      -- ⚠ Flushed HERE, not at the next step's print: stdout into supervise.sh's tee is
      -- block-buffered, and a process killed right after its checkpoint (a reap, a thermal rest)
      -- would otherwise lose the one line that says what the resume should read back.
      (← IO.getStdout).flush
    writeBinAtomic ckptPath thetamv
    writeBinAtomic epPath (toString (ep + 1)).toUTF8
    -- ▶ The staggered loader respawn (see `respawnEvery` above). Placed AFTER the checkpoint so a
    -- crash during a respawn costs nothing, and at an epoch boundary so the discarded batch below
    -- is the only data cost.
    -- ⚠ `ep + 1 < nEpochs`: without it the last epoch spawns a replacement and the process exits
    -- on top of it — a python that starts, builds a tf.data pipeline and is killed seconds later.
    -- Caught by the smoke test, which respawned generation 3 after its final epoch.
    if respawnEvery > 0 && !imgStreams.isEmpty && ep + 1 < nEpochs
        && (ep + 1) % respawnEvery == 0 then
      let slot := shimGen % imgStreams.size
      -- ⚠⚠ CONSUME THE OUTSTANDING READ FIRST. The prefetch keeps one read in flight per producer,
      -- and killing the child closes the pipe under it — the task would surface a torn read at the
      -- next step. Its batch is dropped on the floor; the step it was issued for is re-read from
      -- the replacement, because the refill loop issues into whichever slot is empty. One batch of
      -- data skipped per respawn, on a stream that reshuffles and never ends.
      if let some t := inflight[slot]! then
        let _ ← IO.wait t
      inflight := inflight.set! slot none
      -- ⚠ `arr[i]!` would want `Inhabited ShimProc`, and a live child process has no sensible
      -- default, so the slot is taken with `[i]?` and the replacement happens inside the `some`.
      if let some old := imgStreams[slot]? then
        try old.child.kill catch _ => pure ()
        let _ ← old.child.wait
        shimGen := shimGen + 1
        let newSeed := shimSeed + slot + shimGen * imgStreams.size
        let fresh ← spawnShim net.shimScript "train" gbs net.d0 newSeed
                      (some (slot, imgStreams.size)) shimNC
        imgStreams := imgStreams.set! slot fresh
        IO.println s!"  ▸ shim respawn: producer {slot} of {imgStreams.size} replaced after epoch \
{ep + 1} (generation {shimGen}, seed {newSeed})"
        (← IO.getStdout).flush
  -- Gate G2 (`planning/archive/xla_pjrt_ladder.md` §3). Dumps the whole [θ|m|v] blob, so
  -- the Adam moments are compared too, not just the weights — a moment buffer
  -- that silently failed to thread would still let θ look plausible.
  match ← IO.getEnv "LEAN_MLIR_DUMP_PARAMS" with
  | some path =>
      IO.FS.writeBinFile path thetamv
      IO.println s!"  wrote final [θ|m|v] ({thetamv.size} bytes) → {path}"
  | none => pure ()
  -- ⚠ "cosine/warmup" is quoted verbatim by every Imagenette and ImageNet transcript in the
  -- book; only the constant case gets a new spelling, and it drops the "/warmup" that would
  -- otherwise describe a warmup this configuration does not have.
  let doneSched := if expDecayRate == 1.0 then "constant lr" else s!"{schedName}/warmup"
  IO.println s!"done (trained {net.name} {variant} + {doneSched} via packed threading)."

/-- **Score a checkpoint, standalone** — the eval half of `trainAdamSched` with no training in
    front of it (`planning/archive/next_session_verified_trainer_code.md` §2).

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
    TRAIN time (`LEAN_MLIR_EMA_BN`), so an EMA run reports one of the two numbers and discards the other.
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
  Two exits, neither of them retroactive on its own (planning/archive/next_session_verified_trainer_code.md \
§2b): (a) append the {nBnStats} stat floats to the checkpoint format — clean going forward, but A3's \
finished checkpoint does not contain them; (b) --recalibrate, ~100-200 training batches forward to \
re-accumulate the statistics, which DOES reach an existing checkpoint and is a different estimate \
from the run's own.\n\
  Scoring works today on the LayerNorm nets (ConvNeXt, ViT), which carry no running state."
  -- The region to score. ⚠ `"ema"` on a variant with no fourth region is a REFUSAL and not a
  -- fallback to live: the request and the artifact disagree, and quietly answering the other
  -- question is how a live-weight number gets quoted as a shadow one.
  -- ⚠⚠ **`emaRegion`, NOT THE LITERAL 3** (2026-08-27). Under accumulation the gradient
  -- accumulator takes region 3 and the shadow is region 4, so a hardcoded index here scores `G`
  -- as if it were weights — a plausible-looking percentage off a running gradient sum, which is the
  -- exact failure class this function's BN blocker exists to prevent one line down.
  let regIdx ← match region with
    | "auto" => pure ((VerifiedVariant.emaRegion variant).getD 0)
    | "live" => pure 0
    | "ema"  =>
      match VerifiedVariant.emaRegion variant with
      | none =>
        throw <| IO.userError s!"region 'ema' asked of variant '{variant}', which has no EMA \
shadow — its blob is {nRegions} regions and there is no shadow slot to score. Use \
'live', or score a checkpoint written by an ema* variant."
      | some i => pure i
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
different blob layout — most likely across the EMA/accumulation boundary, since accumulation \
adds a 4th region and the EMA shadow a 5th."
  let theta := thetamv.extract (regIdx * pBytes) ((regIdx + 1) * pBytes)
  IO.println s!"  region {regIdx} of {nRegions} \
({if VerifiedVariant.emaRegion variant == some regIdx then "the EMA SHADOW" else "the live weights"}), \
{net.nParams} params"
  (← IO.getStdout).flush
  -- ▶ `LEAN_MLIR_REPLICAS=N` scores through the SHARDED eval — N devices, `N × evalBs` per invoke —
  -- read exactly as the trainers read it. ⭐ This is the knob `scripts/sharded_eval_gate.sh` turns:
  -- one checkpoint at 1 and at N replicas must give the same count AND the same bitmap, because
  -- the loop below is the per-epoch eval's own (`evalScore`), not a copy of it.
  let replicas := ((← IO.getEnv "LEAN_MLIR_REPLICAS").bind (·.toNat?)).getD 1
  let sess ← mkSessionDp fwdPath replicas
  -- ⚠ `evalOnly := true` — this tool never touches the train split, and on Imagenette reading it
  -- anyway is 7.4 GB held for nothing. Inert on `.imagenet`, which streams.
  let (_, _, _, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir evalD0 (evalOnly := true)
  let nc := net.nClasses
  let fwdShapes := net.shapesBA
  if replicas > 1 then
    let egB := replicas * evalBs
    IO.println s!"  EVAL SHARDED: {replicas} replicas x {evalBs} = {egB} images per invoke, \
{(nEval + egB - 1) / egB} invokes, last one {nEval - (nEval - 1) / egB * egB} real + \
{((nEval + egB - 1) / egB) * egB - nEval} pad"
  -- ▶ `LEAN_MLIR_DUMP_CORRECT=<prefix>` -> `<prefix>.bin`, one byte per val image (1 = top-1
  -- correct), in eval order. ⭐ THIS is the site McNemar wants: score two committed checkpoints,
  -- then compare their bitmaps. θ never changes here, so the bitmap is a pure function of the
  -- checkpoint and the val set — re-scoring gives the identical file.
  let dumpCorrect := (← IO.getEnv "LEAN_MLIR_DUMP_CORRECT")
  -- Hold the parameters on device across every batch — one push per replica, not one per invoke.
  -- `gen` is a constant because θ never changes here, which is the whole difference from the
  -- training loop.
  let rows ← if net.data == .imagenet then spawnValStream net evalD0 else pure (.held evalImg evalLbl)
  let (correct, correct5, nScored, correctBits) ← evalScore sess s!"m.{net.slug}_fwd" theta fwdShapes
    rows nEval evalBs evalD0 nc replicas net.paramShapes.size.toUSize 1
    dumpCorrect.isSome
  reapValStream rows
  let acc := correct.toFloat / nScored.toFloat * 100.0
  let acc5 := correct5.toFloat / nScored.toFloat * 100.0
  -- ⭐ Printed in the SAME shape as the in-training line, so the equality gate is a literal
  -- comparison of two strings rather than an arithmetic one.
  IO.println s!"  checkpoint: acc = {correct}/{nScored} = {acc}%  top5 = {correct5}/{nScored} = {acc5}%  [95% CI {wilson95 correct nScored}]"
  match dumpCorrect with
  | some pfx =>
      IO.FS.writeBinFile s!"{pfx}.bin" correctBits
      IO.println s!"    per-example top-1 bitmap -> {pfx}.bin ({correctBits.size} bytes) — pair two of these with scripts/mcnemar.py"
  | none => pure ()
  if nScored != 50000 && net.data == .imagenet then
    IO.println s!"  ⚠ val is {nScored} of ImageNet's 50,000 — this is NOT over timm's denominator"
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
    -- ▶ `wilson95` here too. ⚠ Chapter 1's linear model is the ONE book trainer that does not
    -- route through `VerifiedNet.train` — it keeps this bespoke entry point for the 2-argument
    -- `linearTrainStepV` FFI — so adding the interval there missed it, and ch.1 was the only
    -- chapter still printing a bare accuracy. Same placement rule as `VerifiedNet.train`: after
    -- the percentage, because this print carries a trailing `(Nms)`.
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nEval} = {acc}%  [95% CI {wilson95 correct nEval}] ({epMs}ms)"
    (← IO.getStdout).flush
  -- Gate G2 (`planning/archive/xla_pjrt_ladder.md` §3): dump the final parameters so the
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

/-- **fp8 (E4M3) Lean trainer** — the low-precision sibling of `trainLinear`.

    Keeps **fp32 master weights** and, each step, projects the weights
    (per-output-column) and the activations (per-tensor) onto the **E4M3** grid
    ([`LeanMlir/E4M3Quant.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/E4M3Quant.lean)), runs the *same* verified `@<slug>_train_step`
    kernel (the matmul accumulates in fp32 — the `dotMixed` model: `u_leaf =
    E4M3`, `u_acc = fp32`), and applies the recovered gradient delta to the fp32
    master via `addDelta` (`master += Wout − Wq = master − lr·∇`). The MLIR and
    FFI are **unchanged**: fp8 here is host-side operand byte-prep, exactly the
    §3b render-tie model ([`Proofs/E4M3Fold.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Float/E4M3Fold.lean)). Eval runs the fp32
    master through `@<slug>_fwd` (the "fp32-infer" accuracy of the fp8-trained
    model, mirroring `scripts/mnist_e4m3_demo.py`).

    Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-linear-e4m3-verified data` -/
def VerifiedNet.trainLinearE4M3 (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  net.printBlurb
  IO.println "  [fp8 E4M3] fp32 master · per-column W / per-tensor x → E4M3 grid · fp32 accumulate"
  -- ⭐ `mkSession` — the fp8 peer of `trainE4M3`/`trainAdamSchedE4M3` above, and IREE-only for
  -- the same reason until 2026-08-25. ⚠ The other `compileVmfb` call sites left in this file are
  -- the PGD / spectral / smoothing trainers, which are a separate (non-fp8) port.
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
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
  -- ⭐ `mkSession`, not `compileVmfb` + `LowererSession.create`. The fp8 arms were the last
  -- trainers still hardcoding a `.vmfb`, which made them IREE-ONLY: on XLA they printed the
  -- "XLA/PJRT" banner and then died in `iree-compile`. Nothing about that was fp8-specific —
  -- it is the same hardcoded-artifact bug class `planning/archive/demo_xla_port.md` §3 catalogues for
  -- the demos. `mkSession` hands the `.mlir` straight to PJRT and keeps the IREE compile path
  -- byte-identical, so both backends now serve the fp8 numerics.
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} ({net.specs.size} params, {net.nParams} floats), fp8-SGD (E4M3 leaf / fp32 acc), He init"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  -- ⭐ The TRAIN STEP declares one more tensor than the forward when the graph emits a loss:
  -- a rank-0 scalar after the params. Same construction as the fp32 `train` (see the `tsShapes`
  -- line in that function) — this path passed plain `shapes` to both, so on the two nets with
  -- `lossSlot := true` (`mlpVerified`, `cnnVerified`) it supplied N destinations for a graph
  -- returning N+1 and the PJRT shim's G4 arity gate refused to run:
  --     G4 VIOLATION: @mlp_train_step returns 7 outputs, caller supplied 6
  --     G4 VIOLATION: @cnn_train_step returns 11 outputs, caller supplied 10
  -- cifar8 leaves `lossSlot` false, which is the only reason the CIFAR fp8 arms ever ran.
  -- ⚠ `shapes` (no loss slot) stays correct for `forwardF32` below — the eval graph returns
  -- logits only. The two must NOT be unified.
  -- The extra trailing float is never read back: `F32E4M3.addDelta` iterates `F32.size master`.
  let tsShapes := packShapes (if net.lossSlot then net.paramShapes ++ #[#[]] else net.paramShapes)
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
                  xb paramsQ tsShapes yb bs.toUSize d0.toUSize nc.toUSize
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
    (baseLR β1 β2 : Float) (warmupEpochs : Nat) (variant : String := "adam")
    (expDecayRate : Float := 0.0) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  net.printBlurb
  IO.println s!"  [fp8 E4M3] fp32 master [θ|m|v] · per-slot θ quant + per-tensor input · fp32 accumulate ({variant})"
  let hasBn := !net.bnChannels.isEmpty
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  -- ⭐ `mkSession` — see `trainE4M3` above for why these were IREE-only until 2026-08-25.
  -- ⚠ Deliberately NOT adopting `trainAdamSched`'s per-variant forward resolution
  -- (`<slug>_<variant>_fwd.mlir` with a `<slug>_fwd.mlir` fallback): that would change WHICH
  -- graph the fp8 arms evaluate against, which is a numerics change, not a backend port.
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_{variant}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let fwdEvalSess ← if hasBn then
      mkSession s!"{net.mlirDir}/{net.slug}_fwd_eval.mlir"
    else pure fwdSess
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  let schedDesc := if expDecayRate == 1.0 then s!"constant lr {baseLR}"
    else s!"cosine+warmup {warmupEpochs}ep, baseLR {baseLR}"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} {variant} fp8 ({schedDesc}), He init"
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
  -- ⚠⚠ Route through `ckptPathFor`, do NOT hand-build this. It read
  --     s!".lake/build/{net.slug}_{variant}_e4m3_ckpt.bin"
  -- which is distinct from the fp32 runs (the point) but ALSO ignores `$LEAN_MLIR_CKPT_TAG`
  -- and the backend scoping that `ckptPathFor` applies. That silently broke an n=5 sweep on
  -- 2026-08-25: every seed of the fp8 arm resolved to ONE file, so seeds 2-5 printed
  -- "▸ resuming from fp8 checkpoint at epoch 40" and re-reported seed 1's result. The
  -- `_e4m3` suffix moves into the variant, which keeps fp32 and fp8 apart AND gains the tag.
  let ckptPath ← net.ckptPathFor s!"{variant}_e4m3"
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
      -- `expDecayRate = 1.0` ⇒ exactly constant, as in `trainAdamSched`: chapter 4's levers
      -- compare optimizers and precisions under ONE flat rate, because a schedule is a fourth
      -- variable and this driver's peer had been supplying one silently.
      let lrt := if expDecayRate == 1.0 then baseLR
                 else if gstep ≤ warmSteps then baseLR * gstep / warmSteps
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
        -- so k = 1 and the compensated form `1 − d^(1/k)` is exactly `1 − d`. If an
        -- accumulation path is ever added here, copy `bnMom` from `trainAdamSched`.
        -- ⚠ The decay is `cfg.bnMomentum` (2026-08-30), not a literal 0.99, and `none` is what
        -- "no accumulation here" is spelled as — `bnEmaWeight`'s `none` arm keeps the historic
        -- `0.01` double exactly.
        runningBnStats ← F32.ema runningBnStats batchBn
                           (if bnFirst then 1.0 else cfg.bnEmaWeight none)
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

-- ════════════════════════════════════════════════════════════════
-- § The spec → runtime lowering, and the spec-level entry points
--   (`VerifiedSpec` is the import-free DSL; these are the only places a spec meets the runtime)
-- ════════════════════════════════════════════════════════════════

namespace VerifiedNetSpec

/-- Lower to the runtime `VerifiedNet` the driver consumes. -/
def toNet (s : VerifiedNetSpec) : VerifiedNet :=
  { name := s.name, slug := s.slug, specs := s.toSpecs, d0 := s.d0,
    nClasses := s.nClasses, data := s.data, blurb := s.blurb, bnChannels := s.bnChannels,
    dropKeeps := s.dropKeeps, dropoutKeep := s.dropoutKeep, shimScript := s.shimScript,
    mlirDir := s.mlirDir, lossSlot := s.lossSlot }

/-- Train end-to-end (delegates to the shared `VerifiedNet.train` driver). -/
def train (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.train cfg dataDir

/-- Train the 2-parameter linear path (Chapter 1); see `VerifiedNet.trainLinear`. -/
def trainLinear (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.trainLinear cfg dataDir

end VerifiedNetSpec
