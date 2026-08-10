import LeanMlir.VerifiedNets

/-! # Shared body of the verified MobileNetV4-Conv-S + AdamW Imagenette trainer

`mobilenetv4-verified-adam-xla` (XLA/PJRT) runs `mobilenetv4Verified` against
`verified_mlir/mnv4_<variant>_train_step.mlir`. Structurally the `Resnet50AdamCommon` twin:
identical harness, one net swapped.

Lake requires a distinct root module per executable, so the config and entry point live here rather
than being duplicated the moment an IREE peer is added; drift in `epochs`, `batchSize`, the seed or
any AdamW hyperparameter would quietly invalidate the comparison against the other Imagenette nets.

⚠ **XLA/PJRT only, and that is deliberate.** Every other Imagenette net has an IREE peer; this one
does not yet, because nothing has needed it. Adding one is a lakefile entry plus `ireeLink` — the
body here is backend-agnostic.

## What is and is not tied on this net (`planning/mnv4_verified.md`)

* ✅ The **forward** ties the JAX reference on shared weights at **1.423e-06**
  (`scripts/mnv4_forward_tie.py`), unpatched — which is what pins the pre/post-DW ORDER, invisible
  to every count and type.
* ✅ The **gradient** ties `jax.grad` of the reference per parameter
  (`scripts/grad_tie.py --net mnv4 --nokink`): 0/147 live parameters over 10× the control.
* ⛔ **There is no §1a proof-chain tie for this net.** Every op the render composes carries a proven
  `den`, but there is no MNv4 composed-backward theorem the way `MobileNetV2BackB0` /
  `ResNet34BackB0` exist for theirs. So a number off this trainer is **measured** correct, not
  proven — a tier below mnv2/R34. Name that when quoting it; the same caveat `Resnet50AdamCommon`
  carries, for the same reason.
-/

/-- Matches `resnet50AdamConfig` / `efficientnetAdamConfig` — 80 epochs, bs 32 — so the MNv4 row of
    the Imagenette tier is read against the other nets at the same schedule and any difference is
    the architecture rather than the recipe.

    ⚠ The target is `RESULTS.md`'s **84.58%**, which is the JAX-baseline path's number for this
    exact block table. Unlike MobileNetV2's, that number does **not** move when this render changes:
    the stem was built as `convStridedXla` precisely so the verified render and the baseline are the
    same net (`planning/mnv4_verified.md` §3e). -/
def mobilenetv4AdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine decay — the
    Imagenette tier's schedule exactly.

    `LEAN_MLIR_VARIANT` selects `verified_mlir/mnv4_<variant>_train_step.mlir`. Only **`adam`** (the
    default) is rendered today; `adamdp` is one `#eval` away in
    `Proofs/Codegen/MobileNetV4RenderB.lean` but has no artifact yet, so asking for it gets a
    missing-file error — the intended loud failure rather than a silent fallback.

    `LEAN_MLIR_BATCH` overrides the batch and **must match the batch the variant was rendered at**:
    the batch is baked into the graph, not a runtime dimension, so a mismatch is a shape error at
    the first invoke. Only bs32 exists, and the eval forward is rendered at bs32 too.

    `LEAN_MLIR_BASE_LR_U` — base LR in MICRO-units (1e-6), so `1000` is 0.001. Integer-encoded
    because this toolchain has no `String.toFloat?`; same knob and units as the R34/R50 trainers.

    ⚠⚠ **Delete the checkpoint when the render changes.** `.lake/build/mnv4_adam_ckpt_xla.bin` is
    size-guarded but NOT architecture-guarded, so a stale blob with the same parameter count resumes
    silently and the run prints `done` having trained nothing. That is not hypothetical: it is
    exactly what happened on the MobileNetV2 re-run (`planning/mnv4_verified.md` §3h trap 2), where
    an epoch-80 checkpoint from the OLD net made the new run "succeed" instantly and exit zero. -/
def runMobilenetV4Adam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD mobilenetv4AdamConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.001
  mobilenetv4Verified.toNet.trainAdamSched { mobilenetv4AdamConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 3 variant
