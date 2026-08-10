import LeanMlir.VerifiedNets

/-! # `vit-verified-adam` — train ViT-Tiny with the VERIFIED-rendered **AdamW** step

Phase 3c of `planning/vit_train_to_vit_verified.md`: the SGD `vit-verified` with its optimizer
swapped for AdamW. The packed train step `@vit_adam_train_step` is `pretty(provenGraph)` out of
`LeanMlir/Proofs/Codegen/ViTRender.lean`'s `vitAdamTrainStepFaithful` — gradients un-fused and
handed to the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple — then driven by
`VerifiedNet.trainAdamSched`, which threads `[θ|m|v]` as a single packed param blob plus the runtime
`lr`/`bc₁`/`bc₂` scalars through the generic FFI (`n_params = 3k`; the moments ride in the params
slot, so the prebuilt `.so` is unchanged).

**The artifact became `pretty(provenGraph)` on 2026-07-28** (handoff §2a). Before that this driver
emitted the hand-written render itself at startup; the swap was licensed by `vit-adam-tie` —
gradient norm-rel 1e-6, `%loss` **bit-exact**, 0/200 params disagreeing, on all 16,579,041 returned
floats. `%loss` is the load-bearing check on this net rather than a footnote: ViT has no BN, so
nothing else in the output depends on the forward alone.

Recipe matches `MainVitTrain.lean`'s `vitTinyConfig`: AdamW lr 3e-4 / wd 1e-4, cosine + 5-epoch
warmup, label smoothing 0.1, augment, 80 epochs, bs 32 — the schedule differs from the other four

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/vit-verified-adam data`. ⛔ The XLA peer
`vit-verified-adam` exists but **does not run on this box** — the graph dies in MIOpen; see that

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
-/

-- Matches MainVitTrain.lean's `vitTinyConfig` (the reference): 80 epochs, bs 32,
-- AdamW lr 3e-4 / wd 1e-4, cosine + 5-epoch warmup, label smoothing 0.1, augment.
-- (vitTinyConfig sets NO EMA and gradClipNorm 0.0, so the DEFAULT verified path omits them too.)
--
-- ⚠ That parenthesis still holds and is why `LEAN_MLIR_VARIANT=ema` is opt-in here rather than the
-- default. EMA belongs to the **ImageNet** ViT recipe (`jax/MainVitImagenet.lean`'s
-- `vitTinyImagenetConfig.emaDecay := 0.99996`, the DeiT default), not to this Imagenette baseline.
-- So an `ema` run on Imagenette is a GATE VEHICLE — the cheap place to establish that the shadow is
-- wired correctly — not a matched pair with `vitTinyConfig`. Do not describe it as one.
def vitAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 3e-4, β₁ .9, β₂ .999, 5-epoch linear warmup then cosine
    decay (`vitTinyConfig`) — note ViT's schedule differs from the other four nets' 1e-3 / 3 epochs.

    `LEAN_MLIR_VARIANT` selects the rendered train step, i.e. which
    `verified_mlir/vit_<variant>_train_step.mlir` is loaded (and with it a distinct vmfb and
    checkpoint). All are `pretty(provenGraph)` out of `Proofs/Codegen/ViTRender.lean`, and
    `vitAdamVariant` there is the single description of the spelling:

    * **`adam`** (default) — the certified single-device render, licensed by `vit-adam-tie`:
      gradient norm-rel 1e-6, `%loss` bit-exact, 0/200 params disagreeing, on all 16,579,041
      returned floats (handoff §2a).
    * **`adamdp`** (and `adamdp64` / `adamdp32x4` / `adamdp128x4`) — the DATA-PARALLEL renders: the
      same graph plus one `all_reduce(add)/N` per parameter gradient between the certified gradient
      and the certified AdamW triple. That collective is a **declared trusted carve-out** (§5) and
      the render says so in its own output banner. ✅ Gated 2026-07-30 — `vit-dp-check` reproduces
      the single-device step **bit-exactly on all 16,579,041 floats** against a sum-not-mean control
      that fires at 0.996. (An earlier version of this docstring said the graph does not execute on
      this box; §2j retired that.) Pair it with `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N` and
      `HIP_VISIBLE_DEVICES` unset.
    * **`ema`** — the EMA weight-shadow render (`planning/ema.md`). ⚠ Its blob has **four** regions
      where every other variant has three, so it cannot share a checkpoint with them; the driver's
      size guard makes a crossed one throw rather than resume misaligned garbage.

    This driver no longer WRITES its artifact. Until 2026-07-28 it re-emitted the hand-written
    `ViTRender.vitTrainStepModuleAdamSched` on every startup, which meant the committed bytes were
    never authoritative and the writer audit could not see the writer at all. The artifact now comes
    from `Proofs/Codegen/ViTRender.lean`'s `vitAdamTrainStepFaithful` and that `#eval` is its sole
    writer, so a missing artifact **throws** rather than being quietly recreated. -/
def runViTAdam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let tsPath := s!"verified_mlir/vit_{variant}_train_step.mlir"
  if !(← System.FilePath.pathExists tsPath) then
    throw (IO.userError s!"{tsPath} missing — it is written by \
LeanMlir/Proofs/Codegen/ViTRender.lean; run `lake build LeanMlir.Proofs.Codegen.ViTRender` first")
  -- `LEAN_MLIR_BATCH` must MATCH the batch the selected variant was rendered at, because the batch
  -- is baked into the graph rather than being a runtime dimension. A mismatch is a shape error at
  -- the first invoke — loud, not a silent limp. The pairs that exist: `adam`/`adamdp` at 32,
  -- `adam64`/`adamdp64` at 64. Eval needs no flag: `trainAdamSched` reads the eval width off the
  -- forward artifact (`evalBs`), so it scores at 32 whatever the train batch is.
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD vitAdamConfig.batchSize
  -- ▶ `ema` selects the EMA-shadow render (`planning/ema.md`): the same AdamW graph plus a 4th
  -- `[θ|m|v|ema]` blob region updated by `adamMNextF` at `(β₁ := d)`, with EVAL AND THE CHECKPOINT
  -- scoring the shadow. ViT is the cheapest of the three EMA nets — LayerNorm means there is no
  -- `ema_bn` peer to carry, so it is the parameter shadow alone.
  --
  -- `LEAN_MLIR_EMA_DECAY_U` sets the decay in MICRO-units, because this toolchain has no
  -- `String.toFloat?` (the `LEAN_MLIR_BASE_LR_U` dodge). The default here is **0.99996** — the
  -- DeiT value `jax/MainVitImagenet.lean`'s `vitTinyImagenetConfig` carries, NOT the 0.9999
  -- ConvNeXt and EfficientNet use. `trainAdamSched`'s own default is 0.9999, so passing this
  -- explicitly is what stops the two references silently converging on one number.
  -- ⚠ It is the IMAGENET recipe's value, carried here so the knob's default is the one a matched
  -- pair would want; `vitTinyConfig` (this trainer's actual reference) has no EMA at all.
  -- ⚠ **`0` is meaningful and is the gate**: at decay 0 the shadow must be BIT-IDENTICAL to the
  -- live weights, since `d = min(0, ·) = 0 ⇒ ema' = θ'`. A free exact endpoint, and it is what pins
  -- the wiring — a shadow reading the wrong slot fails it immediately.
  --
  -- ⚠ At 0.99996 the time constant is 25,000 optimizer steps, and an 80-epoch Imagenette run is
  -- 23,600 — i.e. **under one τ**, tighter than ConvNeXt's 2.4 τ. That is exactly the regime
  -- `ema.md` §2 says the warmup correction exists for, and it is why the correction is applied in
  -- the driver rather than being left to the decay value.
  let emaDecay := match (← IO.getEnv "LEAN_MLIR_EMA_DECAY_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.99996
  -- `LEAN_MLIR_BASE_LR_U` — base LR in MICRO-units, the `MainResnet34VerifiedAdam` knob, added here for
  -- the EMA ratio gate rather than for training. Default 0.0003 (`vitTinyConfig`) is unchanged, so
  -- every existing run is bit-for-bit unaffected.
  --
  -- ⚠ It is an INSTRUMENT, and the reason is worth keeping. The gate's known answer is
  -- `ema₁ − θ₁ = d₀·(θ₀ − θ₁)`, so it measures a difference of two nearly-equal f32 numbers: at
  -- ViT's step 1 the warmup LR is 3e-4/(5·295) ≈ 2e-7, which puts `ema − θ` at roughly ONE ULP of
  -- θ and leaves the ratio dominated by cancellation rather than by the formula. Raising the LR
  -- moves θ far enough for the subtraction to be well-conditioned; it changes nothing about the
  -- EMA arithmetic under test. Same move as `r34-mom-tie` needing `BASE_LR_U=100000`, and the same
  -- §2j rule underneath: check the instrument can RESOLVE the thing you are measuring first.
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.0003
  -- ▶ `drop*` variants select the STOCHASTIC-DEPTH render (`planning/stochastic_depth.md`): the
  -- graph takes 24 extra `tensor<Bxf32>` inputs — TWO per block, the attention and MLP residual
  -- branches dropping independently at one shared keep — carrying `bernoulli(keep_i)/keep_i` per
  -- example, drawn on the host and seeded from the global step. ⚠ It is the ONE ViT render built on
  -- the BATCHED chain (`ViTRenderB`); a per-example mask is not expressible at the per-example
  -- index at all.
  --
  -- `LEAN_MLIR_DROP_RATE_U` is the rate in MICRO-units (`100000` = 0.1, the DeiT value and
  -- `vitTinyImagenetConfig.dropPath`). Unset ⇒ the spec's ramp.
  -- ⚠⚠ **`0` is meaningful and it is THE GATE**: every keep becomes 1.0, every supplied scale is
  -- exactly 1.0, and each drop op is the identity in IEEE (`Proofs.dropPath_ones_id`), so `adamdrop`
  -- must train bit-identically to plain `adam`. ⚠ But it is an ENDPOINT gate and endpoint gates are
  -- structurally BLIND TO PLACEMENT — `1 ⊙ (branch + x) = branch + x` exactly, so a site on the
  -- block OUTPUT passes it bit-for-bit. `scripts/misplace_drop_sites.py` is the control that makes a
  -- green run mean anything, and on ViT it needed fixing first: the branch is the SECOND operand of
  -- this net's residual add, which the script could not match.
  let dropNet := match (← IO.getEnv "LEAN_MLIR_DROP_RATE_U").bind (·.toNat?) with
    | some 0 => { vitVerified.toNet with
                    dropKeeps := vitVerified.dropKeeps.map (fun _ => 1.0) }
    | some u =>
        -- Re-derive from the SPEC's own keeps so the site→block pairing stays the renderer's:
        -- `i = (1 − keep)·11/0.1` recovers the BLOCK index the spec encoded, which is shared by
        -- each site pair. Deriving it from the site ordinal would unpair them.
        let rate := u.toFloat * 1e-6
        { vitVerified.toNet with
            dropKeeps := vitVerified.dropKeeps.map
              (fun k => 1.0 - rate * ((1.0 - k) * 11.0 / 0.1) / 11.0) }
    | none   => vitVerified.toNet
  dropNet.trainAdamSched { vitAdamConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant 0.0 1.0 emaDecay

def main (argv : List String) : IO Unit := runViTAdam argv
