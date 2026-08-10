import LeanMlir.VerifiedNets

/-! # `mobilenetv4-verified-adam` — the MobileNetV4-Conv-S AdamW trainer on XLA/PJRT

Shared body in `apps/imagenette/MobilenetV4AdamCommon.lean`, linked against `ffi/libpjrt_ffi.so`.

Phase 4 of `planning/mnv4_verified.md`: 80 epochs, bs32, AdamW, target **84.58%** — the JAX-baseline
path's number for this block table. The forward and the gradient are both tied against that
reference (§3e, §3i), so the two paths are the same net and the number is a reproduction rather than
a fresh measurement.

⚠ **This does not move the verification tier.** Every op the render composes carries a proven `den`,
but MNv4 has no composed-backward theorem yet — see `MobilenetV4AdamCommon`'s header.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mobilenetv4-verified-adam

# ⚠ the data arg is the ROOT, not the dataset dir — `loadData` appends `/imagenette` itself.
#   Passing `data/imagenette` fails only at the loader, AFTER every artifact compiles and the
#   full header prints, which reads like a data problem and is an argv problem (§3h trap 1).
# ⚠ and move any stale checkpoint aside first, or the run resumes the OLD net and exits zero:
mv .lake/build/mnv4_adam_ckpt_xla.bin{,.bak} 2>/dev/null

HIP_VISIBLE_DEVICES=0 .lake/build/bin/mobilenetv4-verified-adam data
```

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
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

def main (argv : List String) : IO Unit := runMobilenetV4Adam argv
