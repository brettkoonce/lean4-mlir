import LeanMlir.VerifiedNets

/-! # `convnext-verified-adam` — train ConvNeXt-T with the VERIFIED-rendered **AdamW** step

The ConvNeXt-T peer of the `vit`/`mnv2`/`enet`/`r34` verified-adam trainers: the proof-rendered
train step (`LeanMlir/Proofs/Codegen/ConvNeXtRender.lean →
verified_mlir/convnext_adam_train_step.mlir`, `@convnext_adam_train_step`) — patchify stem →
[3,3,9,3] depthwise-7×7 blocks (LayerNorm + GELU + layerScale) + 3 between-stage downsamples → GAP →
dense — with the gradients un-fused and handed to the proven
`adamMNextF`/`adamVNextF`/`adamWParamF` triple, driven by the generic `VerifiedNet.trainAdamSched`:
`[θ|m|v]` (180 params) packed as one blob + runtime `lr`/`bc₁`/`bc₂` scalars through the unchanged
FFI.

**The artifact is `pretty(provenGraph)` since 2026-07-28** (handoff §2f-bis). It used to come from a
hand-written string emitter in `tests/TestConvNeXtTrain.lean`; the swap was licensed by
`convnext-adam-tie`, and after the stem-patchify and even-kernel-downsample weight-gradient gaps were
closed (`9bb00f5`) that tie passes on all **83,478,849** returned floats with spread 0/180 —
so all 180 params are `pretty(AST)` end to end. The driver itself needed no change: it resolves the
path from the net slug.

⚠ **Re-measured 2026-08-12: the float count here read 83,434,629 and the verdict read "bit-exact",
and both were stale.** 83,434,629 is 3 × 27,811,543, the retired SCALAR-LN net; the channel-LN net
returns 3 × 27,827,818 + 3 = 83,483,457. ⚠⚠ **The bit-exactness figures that used to be quoted here
are STALE** — they were taken at 27,826,282 params, i.e. before the head LN was restored
(2026-08-30, planning §7.1), and describe a net this driver no longer trains. Re-run
`convnext-adam-tie` to replace them; on 2026-08-30 it reported 0 of 182 params disagreeing at
gradient norm-rel 0.000000 against a 1e-4 threshold. The gate's own
note is why that is fine rather than a regression — the two renders agree BETTER than this render
agrees with ITSELF under a semantics-preserving batch reversal, so the residual is ConvNeXt's
conditioning (layer-scale γ is a cancelling reduce, §2f) and not a difference between emitters.

ConvNeXt is all-smooth (LayerNorm, not BN), so there's no running-stats / train-vs-eval BN gap —
this is **exact-parity** territory like ViT (eval matches train), and it is why there is no
`convnext_fwd_eval` artifact and `fwd-tie convnext --eval` refuses outright. Recipe matches the
reference (`MainConvNeXtTrain.lean`'s `convNextTinyConfig`): AdamW lr 1e-3 / wd 1e-4, cosine +
3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32. Weight decay uniform (incl.
LN/bias), matching the other verified paths.

Run (GPU): `CUDA_VISIBLE_DEVICES=0 PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8
.lake/build/bin/convnext-verified-adam data` (loader reads `data/imagenette`). That is the default
XLA/PJRT lowerer, which is measurably faster on this box and the only path the collectives exist
on; `LEAN_MLIR_LOWERER=iree` selects the other one.
⚠ Clear `.lake/build/convnext_adam_ckpt_xla*` between runs, or set `LEAN_MLIR_CKPT_TAG` — a stale
checkpoint resumes at the finished epoch, trains ZERO steps and still exits 0.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
-/

-- Matches MainConvNeXtTrain.lean's `convNextTinyConfig`: 80 epochs, bs 32, AdamW lr 1e-3 / wd 1e-4,
-- cosine + 3-epoch warmup, label smoothing 0.1, augment.
def convnextAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine
    decay (`convNextTinyConfig`).

    `LEAN_MLIR_VARIANT` selects the rendered train step, i.e. which
    `verified_mlir/convnext_<variant>_train_step.mlir` is loaded (and with it a distinct vmfb and
    checkpoint). ⚠ They come from **two** renderers now: `adam`/`adamdp` (and the `ema`/`wx`/`clip`
    spellings) from `Proofs/Codegen/ConvNeXtRender.lean` at the per-example index, and
    `adamdrop`/`adamdpdrop` from `Proofs/Codegen/ConvNeXtRenderB.lean` at the batched index `N := B`
    — which is the only place a per-EXAMPLE stochastic-depth mask is expressible at all.

    * **`adam`** (default) — `pretty(provenGraph)`, tied against the retired hand-written emitter
      on all 83,478,849 returned floats at norm-rel 3.7e-9, spread 0/180 (handoff §2f-bis, §5: all
      180 params are `pretty(AST)` since the two weight-gradient gaps were closed).
    * **`adamdp`** — the same graph plus one `all_reduce(add)/N` per parameter gradient, a declared
      trusted carve-out (§2h-quater). Gated by `convnext-dp-check`: on a duplicated batch the DP
      step reproduces the single-device one, `%loss` bit-exact and gradient norm-rel 1.1e-8, with
      the sum-not-mean control firing at 1.114. Pair it with `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N`
      and `HIP_VISIBLE_DEVICES` unset, and use the **XLA** build — collectives exist only on the
      PJRT path, and the IREE shim refuses a DP entry point outright rather than silently running
      single-device.

    * **`adamdrop`** / **`adamdpdrop`** — STOCHASTIC DEPTH (`planning/stochastic_depth.md`): the same
      AdamW graph plus 18 per-block residual-branch drop sites, taking 18 extra `tensor<32xf32>`
      inputs (and returning them unread, so the blob layout mirrors). `LEAN_MLIR_DROP_RATE_U` sets
      the rate; **`0` is the gate** — every keep is 1.0, every scale is exactly 1.0, and this must
      then train bit-identically to `adam`. Measured 2026-08-03: **0 of 83,478,846 floats** differ
      after 3 steps under `scripts/det_shim.sh`, with the real 0.1 ramp firing at norm-rel 0.399.
      ⚠ `adamdpdrop` is rendered at **2** replicas, not 4, because it exists for
      `drop-shard-check` — whose known answer is exact only at two (f32 addition is commutative;
      above two the collective is a tree and associativity does not hold).

    No `LEAN_MLIR_BATCH` here, unlike EfficientNet: the batch is baked into the graph and bs32 is
    the only ConvNeXt render that exists, so the knob could only ever produce a shape error. -/
def runConvNeXtAdam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  -- ▶ `ema`/`emadp` select the EMA-shadow render (`planning/ema.md`): same AdamW graph plus a 4th
  -- `[θ|m|v|ema]` blob region updated by `adamMNextF` at `(β₁ := d)`, with EVAL AND THE CHECKPOINT
  -- scoring the shadow. ConvNeXt is the right first net for it — LayerNorm means there is no
  -- `ema_bn` peer to carry, so it is the parameter shadow alone.
  --
  -- `LEAN_MLIR_EMA_DECAY_U` sets the decay in MICRO-units (`999900` = 0.9999, the reference's
  -- value), because this toolchain has no `String.toFloat?` — the `LEAN_MLIR_BASE_LR_U` dodge.
  -- ⚠ **`0` is meaningful and is the gate**: at decay 0 the shadow must be BIT-IDENTICAL to the
  -- live weights, since `d = min(0, ·) = 0 ⇒ ema' = θ'`. That is a free exact endpoint and it is
  -- what pins the wiring — a shadow reading the wrong slot fails it immediately.
  let emaDecay := match (← IO.getEnv "LEAN_MLIR_EMA_DECAY_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.9999
  -- ▶ `drop*` variants select the STOCHASTIC-DEPTH render (`planning/stochastic_depth.md`): the
  -- graph takes 18 extra `tensor<32xf32>` inputs, one per block, carrying `bernoulli(keep_i)/keep_i`
  -- per example, drawn on the host and seeded from the global step. ⚠ It is the ONE ConvNeXt render
  -- built on the BATCHED chain (`ConvNeXtRenderB`) — a per-example mask is not expressible at the
  -- per-example index at all — so `adamdrop` is not byte-comparable to `adam`; its keep = 1 tie is
  -- numeric.
  --
  -- `LEAN_MLIR_DROP_RATE_U` is the rate in MICRO-units (`100000` = 0.1, the ConvNeXt-T paper value
  -- and `convNeXtTinyImagenetConfig.dropPath`). Unset ⇒ the spec's ramp.
  -- ⚠⚠ **`0` is meaningful and it is THE GATE**: every keep becomes 1.0, so every supplied scale is
  -- exactly 1.0 and each drop op is the identity in IEEE (`Proofs.dropPath_ones_id`). The `adamdrop`
  -- render must then train the same parameters as the plain `adam` render — the peer of EMA's
  -- `decay = 0`. ⚠ But it is an ENDPOINT gate and endpoint gates are structurally BLIND TO
  -- PLACEMENT: `1 ⊙ (branch + x) = branch + x` exactly, so a site on the block OUTPUT passes it
  -- bit-for-bit (`stochastic_depth.md` §7b, measured). `scripts/misplace_drop_sites.py` is the
  -- control that makes a green run mean anything.
  let dropNet := match (← IO.getEnv "LEAN_MLIR_DROP_RATE_U").bind (·.toNat?) with
    | some 0 => { convnextVerified.toNet with
                    dropKeeps := convnextVerified.dropKeeps.map (fun _ => 1.0) }
    | some u =>
        -- Re-derive the ramp at a different rate from the SPEC's own keeps, so the block indices
        -- stay the renderer's: `i = (1 − keep)·17/0.1` recovers the index the spec encoded.
        let rate := u.toFloat * 1e-6
        { convnextVerified.toNet with
            dropKeeps := convnextVerified.dropKeeps.map
              (fun k => 1.0 - rate * ((1.0 - k) * 17.0 / 0.1) / 17.0) }
    | none   => convnextVerified.toNet
  dropNet.trainAdamSched convnextAdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3 variant 0.0 1.0 emaDecay

def main (argv : List String) : IO Unit := runConvNeXtAdam argv
