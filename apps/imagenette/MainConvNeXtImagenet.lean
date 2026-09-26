import LeanMlir.VerifiedNetsCore
import LeanMlir.VerifiedTrain

/-! # `convnext-imagenet-verified` — ConvNeXt-T on full ImageNet-1k, verified renderer → XLA

The ConvNeXt peer of `resnet34-imagenet-verified` and `vit-imagenet-verified` (§2p).
Unlike those two, this one needed a renderer change first: `nClasses` was a hardcoded literal and
`-α/K` was a caller-supplied string independent of it — the two-writers-for-one-fact shape that
produced R34's K=10 gradient bug. Both are fixed; and since 2026-09-17 `cBS`/`bB` are PARAMETERS, so this renders at
**batch 64 — global 256 on four replicas, which is its JAX reference's batch exactly**
(`ConvNeXtRenderB.cnxInBS`). It rendered at 32 (global 128) until then, which made the pair a
pair in architecture only: half the batch, twice the updates, and an LR off the linear-scaling
rule. ⭐ The rescope also made the LR correct for free — see `baseLR` below.

XLA-only by construction: collectives live on the PJRT path.

⚠ Does NOT move the verification tier, and is NOT the ConvNeXt paper recipe — see
`the net's Main file`'s claim-ceiling note before quoting anything from it.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
-/

/-- 300 epochs at 64 per device — the ConvNeXt paper's schedule length. `batchSize` is PER DEVICE
    and must match the batch the selected variant was rendered at (**64** since 2026-09-17; it is
    baked into the graph, so a mismatch is a shape error at the first invoke rather than a silent
    limp). ⚠ 64 × 4 replicas = global 256 = `SPE 5004`, which is what the JAX reference
    (`/home/skoonce/convnext_t300_3060/`) trains at; at the old 32 this job ran 10,009 steps/epoch
    against the reference's 5,004. -/
def convnextImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 64
  -- ⭐⭐ **ConvNeXt `_init_weights`, and it is the reason the 2026-09-17 run was killed at e67.**
  -- The JAX reference sets `cnxInit := true` (`jax/MainConvNeXtImagenet.lean:65`); this side used
  -- the He fan-in default, i.e. **2.6x-10.2x wider** — 0.2041 vs 0.02 at the 4x4 stem, 0.2020 vs
  -- 0.02 at the 7x7 depthwise. Two arms with different inits cannot isolate the lowerer, which is
  -- the ONE thing this BatchNorm-free net is in the book for.
  -- `runs/2026-09-17-cnx-verified-300ep/RESULTS.md` §7.0 has the per-layer table.
  -- ⚠ Host-side: no committed artifact moves, and no re-render is needed.
  -- ⚠ Set HERE and not on the ConvNeXt-S/-B ImageNet mains or the Imagenette `convnext-verified-adam`:
  --   S and B have never been trained or paired, and Imagenette HAS a landed number that this
  --   would invalidate. Neither should ride along silently on a flag flipped for T.
  -- ⛔ A checkpoint written before this flag existed must NOT be resumed into a run with it set —
  --   init is applied only on a FRESH start, so a resume silently keeps the old weights. The
  --   e67 blob was deleted for exactly this reason.
  cnxInit   := true

/-- Entry point. Defaults to the single-device `adam` variant rather than `adamdp`, matching the
    R34 and ViT ImageNet drivers: a DP default makes a plain invocation fail at the first step with
    a replica-count refusal, which reads as a broken build rather than a missing flag. -/
def runConvNeXtImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD convnextImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.00025   -- `convNeXtTinyImagenetConfig.learningRate`: 4e-3@bs4096 scaled to bs256.
                          -- ⭐ SINCE THE BATCH-64 RESCOPE THIS IS THE RIGHT NUMBER AND NEEDS NO
                          -- OVERRIDE: the run is at global 256 (4 × 64), so 2.5e-4 is both the
                          -- linear-scaling value AND the reference's own knob
                          -- (its banner: `lr=0.000250  batch_size=256 (4 devices x 64)`).
                          -- ⚠ Until 2026-09-17 this comment said the opposite — the run was at
                          -- global 128 where the rule wanted ~1.25e-4 and 2.5e-4 was a deliberate
                          -- mismatch left as "the thing to tune first". The rescope closed it.
  -- ⚠ `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it
  -- (`min n cfg.epochs`). `totalSteps := cfg.epochs * nb / accK` is what the cosine anneals over,
  -- so EPOCHS=80 is a complete 80-epoch experiment while MAX_EPOCHS=80 is a PREFIX of the committed
  -- 300-epoch decay stopped with the LR high. Without this knob the committed count is also
  -- unprobeable: at `epochs := 300` a short smoke run cannot be asked for at all. Spelled as in
  -- `MainResnet50Imagenet.lean`.
  -- ⚠ Clear checkpoints when switching schedules; resuming across them fuses two LR curves silently.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD convnextImagenetConfig.epochs
  convnextImagenetVerified.toNet.trainAdamSched
    { convnextImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 20 variant
    -- warmup 20 epochs, not 5: `convNeXtTinyImagenetConfig.warmupEpochs := 20` is a ConvNeXt-paper
    -- value and differs from every other net in this repo.

def main (argv : List String) : IO Unit := runConvNeXtImagenet argv
