import LeanMlir.VerifiedNets

/-! # Shared body of the verified ResNet-50 / **full ImageNet-1k** trainer

R50 phase 3 (`planning/next_session_pipeline_then_r50.md` §3.3). Same certified renderer as every
other net here, at `nClasses := 1000` — the bottleneck blocks are `ResNet50RenderB.lean`, their
VJPs are `Resnet50BlocksCertified.lean`, and the three artifacts are three `#eval`s of the same
renderer, exactly as R34's were.

Data comes from the generated tfds shim (`VerifiedData.imagenet`), R50's own
(`generated_resnet50_imagenet_shim.py`), so this side does no augmentation: one definition of the
transform and it is the reference's.

⚠⚠ **THIS IS NOT RSB-A3, AND MUST NOT BE DESCRIBED AS IT.** `jax/MainResnet50Imagenet.lean`'s
`rsb-faithful` recipe reached 76.66% top-1 with **LAMB at effective batch 2048** (512 micro × 4
gradient-accumulation steps), **BCE-with-logits**, mixup/cutmix, and 160-train/224-eval. This
driver has AdamW at bs256 and no gradient accumulation — `grep gradAccum LeanMlir/VerifiedTrain.lean`
still returns 0. `planning/rsb_a2_resnet50.md` records LAMB at bs512 giving **40.8% against 78.1%**,
so batch size is not a detail here. What this is: a verified R50 that exists, trains, and is gated
at the certified AdamW kit — §3.3's "real value even if §4 never runs".

⚠ **Claim ceiling**, unchanged from R34's: the proof-carrying tier stops at Imagenette. What is
inherited is *provenance* plus whatever a pair agreement shows. "One architecture, two independent
lowerings, agreeing" — never "proven".

⚠⚠ **AND R50 HAS NO INCUMBENT RENDER TO TIE AGAINST** (§3.2). Every other net's swap onto the
verified renderer was licensed by a bit-exact numeric tie against the hand-written artifact it
replaced; there is no such artifact here, so that license does not exist. Before any number is
quoted off this trainer, one of these has to be run AND NAMED:
  * `tests/vjp_oracle/run.sh bottleneck` — licenses the identity bottleneck's VJP against JAX's
    `value_and_grad`. ⚠ Its case is `.bottleneckBlock 8 8 1 1`, i.e. **no projection**, so it
    covers neither projection form and specifically not the stride-1 one.
  * a loss-sequence agreement against `jax/MainResnet50Imagenet.lean` at matched init — the only
    check that exercises the whole-net wiring, and the evidence class R34's ImageNet number rests
    on.
`tests/TestR50Contract.lean` already pins the parameter contract (161 tensors, 25,557,032 params,
shapes elementwise) but that is a *layout* gate, not a *gradient* one.

Prerequisites, in order:

```bash
scripts/gen_shims.sh                       # R50's OWN data shim
lake build resnet50-imagenet-verified-xla
lake env lean tests/TestR50Contract.lean   # the layout gate
```
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
    { resnet50ImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant
