import LeanMlir.VerifiedNets

/-! # `resnet50-imagenet-verified` — ResNet-50 on full ImageNet-1k, verified renderer → XLA/PJRT

R50 phase 3. Nothing new was needed in the renderer to reach ImageNet scale — `nClasses`, `B`,
`replicas`, `opt` and `slug` are all parameters of `resnet50TrainStepFaithfulB`, so the four
artifacts are four `#eval`s. What was needed was the three bottleneck block VJPs (phase 1) and the
renderer itself (phase 2).

⚠ Read the two warnings above before quoting anything from this: it is NOT
RSB-A3 (no LAMB, no bs2048, no gradient accumulation), and R50 has no incumbent render to tie
against, so the swap license every other net had does not exist here.

```bash
scripts/gen_shims.sh
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet50-imagenet-verified
lake env lean tests/TestR50Contract.lean
CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=1 LEAN_MLIR_SKIP_EVAL=1 LEAN_MLIR_G2_STEPS=40 \
  .lake/build/bin/resnet50-imagenet-verified data
```

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything: the
backend is a run-time choice about transport, not a different program.
-/

/-- 100 epochs — RSB-A3's own reference schedule at effective batch 2048. Raised from 30 on
    2026-08-06 to run the composed A3 artifact (`lambaccdp8x64bce`) at the length the recipe is
    specified for; the 4-GPU@160 probe measured 240 ms/step, so 100 epochs is ~33 h.

    ⚠⚠ **THIS FIELD IS THE LR SCHEDULE, NOT JUST A LOOP BOUND.**
    `totalSteps := cfg.epochs * nb / accK` (`VerifiedTrain.lean` 1166) — the cosine anneals over
    exactly this many epochs. `LEAN_MLIR_MAX_EPOCHS` caps the LOOP (`min n cfg.epochs`) and does
    NOT touch the schedule, which is precisely what makes a capped run a resumable PREFIX of the
    full one rather than its own shorter experiment:

      LEAN_MLIR_MAX_EPOCHS=30   → epochs 0..29 of the 100-epoch cosine, checkpointed at 30.
      (then, unset)             → resumes at 30 and runs 30..99 on the SAME schedule.

    ▶ That is the intended way to take a look before committing the full ~33 h. It is NOT the same
    as the old `epochs := 30`, which annealed fully by epoch 30 and was a complete experiment.

    ⚠ This is the config for EVERY variant of this driver, not just A3 — `adamdp64` and friends now
    also schedule over 100 epochs. To recover the old R34-comparable, fully-annealed 30-epoch tier
    you must set this field back to 30, not pass `LEAN_MLIR_MAX_EPOCHS=30`. The run announces which
    it is every epoch (`Epoch {ep+1}/{cfg.epochs}`), so a log always says which schedule it ran. -/
def resnet50ImagenetConfig : VerifiedConfig where
  epochs    := 100
  batchSize := 64

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
  -- ⚠ `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it
  -- (`min n cfg.epochs`, and this file's own docstring above spells out why that distinction
  -- bites). `totalSteps := cfg.epochs * nb / accK` is what the cosine anneals over, so this is
  -- the knob that reaches the 90-epoch 2018 tier from a 100-epoch A3 default.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD resnet50ImagenetConfig.epochs
  -- ▶ `LEAN_MLIR_RES` picks the TRAIN resolution, which is not a knob but a choice of NET SPEC:
  -- it selects the slug (`resnet50in` vs `resnet50in160`), hence the artifact family, `d0`, and
  -- the shim. `planning/next_session_rsb_a3.md` §2. ⚠ REFUSES on any other value rather than
  -- falling back to 224 — a silent fallback here is a run that looks correct and trains the wrong
  -- resolution, which is the §0.9 shim-fallback failure one layer up.
  let net ← match (← IO.getEnv "LEAN_MLIR_RES") with
    | none       => pure resnet50ImagenetVerified
    | some "224" => pure resnet50ImagenetVerified
    | some "160" => pure resnet50Imagenet160Verified
    | some r     => throw <| IO.userError s!"LEAN_MLIR_RES={r}: only 160 and 224 are rendered. \
        160 is RSB-A3's train resolution (slug resnet50in160, d0 76800); 224 is the default \
        (slug resnet50in, d0 150528). Rendering another needs a new VerifiedNetSpec + artifacts."
  -- ⚠ ANNOUNCED, both states. Which resolution a run trained at is not recoverable from the loss
  -- curve, and this repo has now twice paid for a throughput/shape setting that printed nothing.
  IO.println s!"  ▸ TRAIN RES: {net.imageH}×{net.imageW} (slug {net.slug}, d0 {net.d0}, shim {net.shimScript})"
  -- ✅ The 160 net is EVALUABLE as of 2026-08-06. Its shim emits A3's split — 76,800 floats/img on
  -- train, 150,528 on val — and the driver now reads the eval width off `@<slug>_fwd_eval` rather
  -- than reusing `net.d0` (`fwdRenderedShape`/`evalD0` in `VerifiedTrain.lean`). The refusal that
  -- stood here until then is gone; the run announces "EVAL RES SPLIT" instead.
  net.toNet.trainAdamSched
    { resnet50ImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runResnet50Imagenet argv
