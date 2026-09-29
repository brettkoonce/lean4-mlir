import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Train

/-! # `resnet50-imagenet-verified` — ResNet-50 on full ImageNet-1k, verified renderer → XLA/PJRT

The renderer needs nothing ImageNet-specific — `nClasses`, `B`, `replicas`, `opt` and `slug` are
all parameters of `resnet50TrainStepFaithfulB`, so the four artifacts are four `#eval`s.

Before quoting anything from this: it is NOT RSB-A3 (no LAMB, no bs2048, no gradient
accumulation), and R50 has no incumbent render to tie against, so the swap license every other net
has does not exist here.

```bash
scripts/gen_shims.sh
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet50-imagenet-verified
lake env lean tests/TestR50Contract.lean
CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=1 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_G2_STEPS=40 \
  .lake/build/bin/resnet50-imagenet-verified data
```

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The target name has no
`-xla` suffix: the backend is a run-time choice about transport, not a different program.
-/

/-- 100 epochs — RSB-A3's own reference schedule at effective batch 2048, the length the
    composed A3 artifact (`lambaccdp8x64bce`) is specified for; the 4-GPU@160 probe measured 240 ms/step, so 100 epochs is ~33 h.

    **THIS FIELD IS THE LR SCHEDULE, NOT JUST A LOOP BOUND.**
    `totalSteps := cfg.epochs * nb / accK` (`Verified.Train` 1166) — the cosine anneals over
    exactly this many epochs. `LEAN_MLIR_MAX_EPOCHS` caps the LOOP (`min n cfg.epochs`) and does
    NOT touch the schedule, which is precisely what makes a capped run a resumable PREFIX of the
    full one rather than its own shorter experiment:

      LEAN_MLIR_MAX_EPOCHS=30   → epochs 0..29 of the 100-epoch cosine, checkpointed at 30.
      (then, unset)             → resumes at 30 and runs 30..99 on the SAME schedule.

    That is the intended way to take a look before committing the full ~33 h. It is NOT the same
    as `epochs := 30`, which anneals fully by epoch 30 and is a complete experiment.

    This is the config for EVERY variant of this driver, not just A3 — `adamdp64` and friends
    also schedule over 100 epochs. For the R34-comparable, fully-annealed 30-epoch tier
    you must set this field to 30, not pass `LEAN_MLIR_MAX_EPOCHS=30`. The run announces which
    it is every epoch (`Epoch {ep+1}/{cfg.epochs}`), so a log always says which schedule it ran. -/
def resnet50ImagenetConfig : VerifiedConfig where
  epochs    := 100
  batchSize := 64
  -- **0.9, NOT the driver's 0.99 default, and it MATCHES the JAX side.** `trainAdamSched`'s
  -- host-side BN EMA defaults to a decay of 0.99, as the JAX reference's `_bn` does: TF's
  -- EfficientNet value. R50's reference is timm, whose `BatchNorm2d` default `momentum = 0.1` is a
  -- decay of **0.9** (PyTorch's momentum weights the NEW batch). Read off
  -- `timm.create_model('resnet50')`, timm 1.0.28 — `momentum = 0.1, eps = 1e-5` on all 53 BN
  -- layers. `jax/MainResnet50Imagenet.lean` sets the same value, and the two must move together or
  -- the JAX ↔ verified comparison acquires a variable that no loss curve can show. At A3's `acc` k
  -- = 8 the driver compensates to `1 − 0.9^(1/8)` = 0.013084 per micro-batch, against `1 −
  -- 0.99^(1/8)` = 0.001256 at 0.99 — a 10× shorter window, as intended. EVAL-ONLY.
  bnMomentum := 0.9

/-- Entry point. Defaults to the 4-replica `adamdp64` artifact, since ImageNet-scale R50 on this
    box is a data-parallel job; `LEAN_MLIR_VARIANT=adam64` selects the single-device render. -/
def runResnet50Imagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adamdp64"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD resnet50ImagenetConfig.batchSize
  -- 0.001 is the AdamW rate and is the right default here — unlike R34/ImageNet, whose only
  -- render is heavy-ball, this net's artifacts are AdamW.
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.001
  -- `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it
  -- (`min n cfg.epochs`, and this file's own docstring above spells out why that distinction
  -- bites). `totalSteps := cfg.epochs * nb / accK` is what the cosine anneals over, so this is
  -- the knob that reaches the 90-epoch 2018 tier from a 100-epoch A3 default.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD resnet50ImagenetConfig.epochs
  -- `LEAN_MLIR_RES` picks the TRAIN resolution, which is not a knob but a choice of NET SPEC: it
  -- selects the slug (`resnet50in` vs `resnet50in160`), hence the artifact family, `d0`, and the
  -- shim. REFUSES on any other value rather than falling back to 224 — a silent fallback here is a
  -- run that looks correct and trains the wrong resolution.
  -- `LEAN_MLIR_RECIPE` picks the AUGMENTATION. Like `LEAN_MLIR_RES` it is not a knob but a
  -- choice of NET SPEC: it selects `shimScript`, hence what the producer actually streams.
  -- WHY IT EXISTS. `shimScript` is a field on the NET, and the `default` (RSB-A2) shim calls
  -- `_randaugment(img, 2, 7.0, 0.5)` unconditionally, so a 2018 run on the `default` spec trains
  -- 2018's optimizer on A2's augmentation — neither recipe, and not comparable to the JAX 2018
  -- number it exists to sit beside.
  -- REFUSES on any other value, for the same reason the resolution dispatch does: a silent
  -- fallback here is a run that looks correct and trains the wrong augmentation.
  let recipe := ((← IO.getEnv "LEAN_MLIR_RECIPE").getD "default").trimAscii.toString
  let net ← match (← IO.getEnv "LEAN_MLIR_RES") with
    | none | some "224" =>
        match recipe with
        | "default" => pure resnet50ImagenetVerified
        | "2018"    => pure resnet50Imagenet2018Verified
        -- RSB-A2 and A1. Each streams its own shim: the BCE target threshold 0.2 (both) and A1's
        -- Mixup α 0.2 and label smoothing 0.1 are target- and data-side, so they ride the shim.
        -- Each also needs its own TRAIN STEP, checked below: A1's weight decay 0.01 is baked
        -- (`…wd001`), A2's 0.02 is the unmarked default.
        | "a2"      => pure resnet50ImagenetA2Verified
        | "a1"      => pure resnet50ImagenetA1Verified
        | r => throw <| IO.userError s!"LEAN_MLIR_RECIPE={r}: at 224² only `default` (RSB-A2's \
            RandAugment), `2018` (random-resized-crop + hflip, no RandAugment), `a2` (RSB-A2 with \
            its BCE target threshold) and `a1` (A2's pack at Mixup α 0.2, ε 0.1) have shims. Naming \
            another needs a row in scripts/gen_shims.sh AND a VerifiedNetSpec carrying it."
    | some "160" =>
        if recipe == "default" then pure resnet50Imagenet160Verified
        else throw <| IO.userError s!"LEAN_MLIR_RECIPE={recipe} with LEAN_MLIR_RES=160: 160² IS \
            RSB-A3's train resolution and streams A3's own shim. Every other 224² recipe (`2018`, \
            `a2`, `a1`) is 224/224, so these two selectors cannot both be set away from their defaults."
    | some r     => throw <| IO.userError s!"LEAN_MLIR_RES={r}: only 160 and 224 are rendered. \
        160 is RSB-A3's train resolution (slug resnet50in160, d0 76800); 224 is the default \
        (slug resnet50in, d0 150528). Rendering another needs a new VerifiedNetSpec + artifacts."
  -- The recipe and the train step are two env vars naming one choice, and a mismatch trains and
  -- reports: A1's data on A2's decay, or an EMA shadow RSB's Table 2 does not have.
  if recipe == "a1" || recipe == "a2" then
    let wantWd := recipe == "a1"
    unless variant.startsWith "lambacc" && (variant.splitOn "wd001").length == (if wantWd then 2 else 1) do
      throw <| IO.userError s!"LEAN_MLIR_RECIPE={recipe} with LEAN_MLIR_VARIANT={variant}: RSB-{recipe.toUpper} \
        trains a no-EMA LAMB-accumulation step {if wantWd then "at the baked wd 0.01 (`…wd001`)" else "at wd 0.02 (no `wd001` marker)"}, \
        e.g. {if wantWd then "lambaccdp4x128wxclipdropbcewd001bf16" else "lambaccdp4x128wxclipdropbcebf16"}."
  -- ANNOUNCED, both states. Which resolution a run trained at is not recoverable from the loss
  -- curve.
  -- ANNOUNCED, shim included. Which AUGMENTATION a run trained on is no more recoverable from a
  -- loss curve than which resolution was.
  IO.println s!"  ▸ TRAIN RES: {net.imageH}×{net.imageW} (slug {net.slug}, d0 {net.d0}, shim {net.shimScript})"
  IO.println s!"  ▸ RECIPE: {recipe} — augmentation comes from {net.shimScript}"
  -- The 160 net is EVALUABLE. Its shim emits A3's split — 76,800 floats/img on train, 150,528 on
  -- val — and the driver reads the eval width off `@<slug>_fwd_eval` rather than reusing `net.d0`
  -- (`fwdRenderedShape`/`evalD0` in `Verified.Train`); the run announces "EVAL RES SPLIT".
  net.toNet.trainAdamSched
    { resnet50ImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runResnet50Imagenet argv
