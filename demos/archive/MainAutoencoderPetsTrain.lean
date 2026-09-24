import LeanMlir
import LeanMlir.ReferenceNets

/-! Skipless autoencoder on Oxford-IIIT Pets — UNet demo Phase 1 smoke test.

    Goal: prove the per-pixel CE codegen + seg train-step ABI work
    end-to-end on real data by training a small encoder/decoder with
    no skip connections. If train loss drops over a handful of epochs,
    the seg pipeline is good and we can move on to the real
    `unetDown` / `unetUp` skip-state plumbing.

    Architecture: 224×224 RGB → 14×14 (4×maxPool) → 224×224 (4×bilinear
    upsample) → 1×1 conv to 3 classes. ~5.5M params. Mirrors
    `ReferenceNets.autoencoderPets`.

    Usage:
      lake exe autoencoder-pets-train [data/pets]
-/

def autoencoderPetsConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 16
  epochs       := 3
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def main (args : List String) : IO Unit := do
  -- Optional 2nd arg overrides epochs (matched-budget skip ablation vs
  -- unet-pets-train). See planning/archive/unet_demo_v2.md Workstream B.
  let epochs := (args[1]?.bind String.toNat?).getD autoencoderPetsConfig.epochs
  ReferenceNets.autoencoderPets.train { autoencoderPetsConfig with epochs }
    (args.head?.getD "data/pets") .pets
