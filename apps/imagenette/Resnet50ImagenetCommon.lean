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

/-- 30 epochs at global batch 256 — the same validation tier R34/ImageNet ran, chosen so the two
    are directly comparable rather than because it is R50's reference schedule (which is RSB-A3's
    100 epochs at bs2048; see the recipe warning above). -/
def resnet50ImagenetConfig : VerifiedConfig where
  epochs    := 30
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
  resnet50ImagenetVerified.toNet.trainAdamSched
    { resnet50ImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant
