module

public import LeanMlir.Verified.NetsCore
public import LeanMlir.Verified.Train

@[expose] public section

/-! # ConvNeXt's `acc4x` — four micro-batches of 8 ARE one step at 32, on DIFFERENT data

`verified_mlir/convnext_acc4x_train_step.mlir` is AdamW over **k = 4 accumulated micro-batches of
8**, the Orin Nano's spelling of the batch-32 recipe (`ConvNeXtRenderB`, beside `ema`): a fourth
parameter region `G`, the `%aup` / `%akeep` scalars, `1/k` folded into `%ob1` and `1/k²` into `%ob2`.

## The construction — stronger than `r50-accum-tie`'s, because ConvNeXt has no BatchNorm

`r50-accum-tie` must feed one batch k times: R50 normalises per micro-batch, so k different
micro-batches are Ghost-BN and tie to nothing exactly. **ConvNeXt normalises per example
(LayerNorm), so the mean of four micro-batch mean-gradients over DIFFERENT rows IS the mean gradient
of their concatenation** — real arithmetic, summation order aside. The peer is therefore ONE step of
the committed batch-32 `convnext_adam_train_step.mlir` on the 32 rows the four micro-batches split,
from the same `(θ, m, v)`. That peer is a different artifact, rendered without accumulation, and
`convnext-adam-tie` ties it to the hand-written emitter. This one tie covers what `r50-accum-tie`'s
duplicated batch is blind to — the combination of different micro-batches — as well as the
accumulate / apply machinery, the reset, the `1/k` and the `1/k²`.

## What is checked

| | claim | the defect it catches |
|---|---|---|
| ① | after micro-steps 1..3, `θ`, `m`, `v` are **bit-identical** to their inputs | an accumulate micro-batch that moves the optimizer |
| ② | after micro-step 4, `(θ', m', v')` match the batch-32 step on the concatenation | the accumulator, the reset, the `1/k`, the `1/k²` on `v` |

## Calibrated, not asserted

Not bit-exact: the accumulated gradient is summed in a different order from the batch-32 reduce.
`convnext-adam-tie` measured that ConvNeXt has a gradient that is a cancelling reduce (`s3b2lg`,
|Σ|/Σ|·| ≈ 0.09) and reproduces to 1e-4 against NO reordering. So ② is gated against a control of
the same CLASS of difference measured where correctness is not in question: **the batch-32 render
on the same 32 rows in reversed order** (`TIE_REVERSE` in `convnext-adam-tie`). ② passes when the
tie is no worse than `ACC_TIE_SLACK` (default 10) times that control, on the worst region and on the
count of parameters off by more than 1e-4 norm-rel.

* **WRONG-k control.** Apply after 3 micro-batches instead of 4. The graph still divides by 4, and
  the fourth micro-batch's rows are missing — exactly the failure of a driver whose apply cadence
  disagrees with the artifact's baked `1/k`. It must MISS by far more than the tie.
* **Degeneracy.** `θ' ≠ θ` by a wide margin, or everything ties trivially.

**Not on the Orin Nano.** The peer is the batch-32 step the board cannot train at, and it cannot
run one bare step of it either. Tried 2026-10-27 at `LEAN_MLIR_MEM_FRACTION=0.70`: NvMap could not
reserve the pool from MemFree, fell back to a smaller one, and the peer's parameter transfer missed
(deploy/ORIN.md § Imagenette). A desktop GPU runs it:

    lake build convnext-accum-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/convnext-accum-tie

Knobs: `ACC_TIE_SLACK` (default 10), `ACC_TIE_XSEED` (default 555).
-/

/-- Max |a−b|, max(|a|,|b|) and the bit-exact count over a float range of two blobs. The magnitude
    is over BOTH buffers, so an `a` driven to zero cannot read as rel 0. -/
def cmp (a b : ByteArray) (offA offB n : Nat) : Float × Float × Nat := Id.run do
  let mut d := 0.0; let mut m := 0.0; let mut ex := 0
  for i in [0:n] do
    let x := F32.read a (offA + i).toUSize
    let y := F32.read b (offB + i).toUSize
    if x == y then ex := ex + 1
    if (x - y).abs > d then d := (x - y).abs
    if max x.abs y.abs > m then m := max x.abs y.abs
  return (d, m, ex)

/-- Per-parameter norm-relative error ‖a−b‖/‖b‖ over the θ' region, and how many parameters exceed
    1e-4 — the SPREAD, which a cotangent defect moves and a reordering does not. -/
def spread (a b : ByteArray) (shapes : Array (Array Nat)) : Float × Nat := Id.run do
  let mut off := 0; let mut worst := 0.0; let mut bad := 0
  for dims in shapes do
    let cnt := dims.foldl (· * ·) 1
    let mut dd := 0.0; let mut bb := 0.0
    for i in [off:off + cnt] do
      let x := F32.read a i.toUSize
      let y := F32.read b i.toUSize
      dd := dd + (x - y) * (x - y); bb := bb + y * y
    let r := dd.sqrt / max bb.sqrt 1e-30
    if r > worst then worst := r
    if r > 1e-4 then bad := bad + 1
    off := off + cnt
  return (worst, bad)

def main : IO Unit := do
  let net := convnextVerified.toNet
  let nP := net.nParams
  let k := 4
  let mb := 8
  let bs := k * mb
  let variant := "acc4x"
  -- `k` off the artifact NAME, the read the driver does (`VerifiedVariant.accK`)
  if VerifiedVariant.accK variant != k then
    throw <| IO.userError s!"accK {variant} = {VerifiedVariant.accK variant}, this harness is written for {k}"
  let slack := (((← IO.getEnv "ACC_TIE_SLACK").bind (·.toNat?)).getD 10).toFloat
  let xseed := ((← IO.getEnv "ACC_TIE_XSEED").bind (·.toNat?)).getD 555
  let lr := 0.01                    -- large enough that θ' ≠ θ by a wide margin
  let bc1 := 0.19                   -- 1−β₁ᵗ, 1−β₂ᵗ at a mid-training step, as `convnext-adam-tie`
  let bc2 := 0.002

  IO.println s!"convnext {variant}: {k} micro-batches of {mb} on DIFFERENT rows = one batch-{bs} step"
  IO.println s!"  under test  verified_mlir/convnext_{variant}_train_step.mlir   (4 regions [θ|m|v|G])"
  IO.println s!"  peer        verified_mlir/convnext_adam_train_step.mlir   (3 regions; tied by `convnext-adam-tie`)"
  IO.println s!"  {net.specs.size} params ({nP} floats), lr {lr}, xseed {xseed}, backend {← LowererSession.backendName}"

  -- ── one (θ, m, v, G, x, y) ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  -- m and v NON-ZERO and distinct: at m = v = 0 the β passthrough terms vanish
  let mIn ← F32.heInit 4242 nP.toUSize 0.02
  let vIn ← F32.scaleShift (← F32.heInit 8484 nP.toUSize 0.01) 1.0 0.05
  -- G's initial contents ARBITRARY: `%akeep = 0` on the first micro-batch must discard them
  let gIn ← F32.scaleShift (← F32.heInit 9191 nP.toUSize 0.5) 1.0 0.3
  let x ← F32.heInit xseed.toUSize (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    -- not periodic in 8, so the four micro-batches' label sets differ
    y := y.push (UInt8.ofNat ((i * 7 + i / 8) % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0
  let mut xrParts : Array ByteArray := #[]
  let mut yrParts : Array ByteArray := #[]
  for i in [0:bs] do
    xrParts := xrParts.push (F32.sliceImages x (bs - 1 - i) 1 net.d0)
    yrParts := yrParts.push (F32.sliceLabels y (bs - 1 - i) 1)
  let xR := F32.concat xrParts
  let yR := F32.concat yrParts

  let pShapes := net.paramShapes
  let accShapes := packShapes (pShapes ++ pShapes ++ pShapes ++ pShapes ++ Array.replicate 5 #[])
  let admShapes := packShapes (pShapes ++ pShapes ++ pShapes ++ #[#[], #[], #[]])

  -- ── the peer first: if the batch-32 step does not fit, nothing else is worth compiling ──
  let admSess ← mkSession "verified_mlir/convnext_adam_train_step.mlir"
  let admBuf := F32.concat #[θ, mIn, vIn, ← F32.write3 (← F32.const 3 0.0) 0 lr bc1 bc2]
  let runAdm (xb yb : ByteArray) : IO ByteArray :=
    LowererSession.mlpTrainStepV admSess "m.convnext_adam_train_step" xb admBuf admShapes yb
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println "  running the batch-32 peer…"; (← IO.getStdout).flush
  let admOut ← runAdm x y
  IO.println "  running the batch-32 peer on the reversed rows (the reordering control)…"; (← IO.getStdout).flush
  let revOut ← runAdm xR yR

  -- ── the accumulation cycle ──
  let accSess ← mkSession s!"verified_mlir/convnext_{variant}_train_step.mlir"
  let runAcc (buf xb yb : ByteArray) : IO ByteArray :=
    LowererSession.mlpTrainStepV accSess s!"m.convnext_{variant}_train_step" xb buf accShapes yb
      mb.toUSize net.d0.toUSize net.nClasses.toUSize
  /- `steps` micro-batches — rows [8i, 8i+8) — applying on the last. Returns the output blob and
     claim ①'s measurement, taken inside the cycle (after the apply everything has moved). -/
  let cycle (steps : Nat) : IO (ByteArray × Float × Nat) := do
    let mut buf := F32.concat #[θ, mIn, vIn, gIn, ← F32.const 5 0.0]
    let mut frozenΔ := 0.0
    let mut frozenExact := 3 * nP
    for i in [0:steps] do
      let applyNow := i + 1 == steps
      buf ← F32.write3 buf (4 * nP).toUSize (if applyNow then lr else 0.0) bc1 bc2
      buf ← F32.write3 buf (4 * nP + 2).toUSize bc2
              (if applyNow then 1.0 else 0.0) (if i == 0 then 0.0 else 1.0)
      let out ← runAcc buf (F32.sliceImages x (i * mb) mb net.d0) (F32.sliceLabels y (i * mb) mb)
      if !applyNow then
        let (dT, _, eT) := cmp out θ   0        0 nP
        let (dM, _, eM) := cmp out mIn nP       0 nP
        let (dV, _, eV) := cmp out vIn (2 * nP) 0 nP
        frozenΔ := max frozenΔ (max dT (max dM dV))
        frozenExact := min frozenExact (eT + eM + eV)
      buf := out
    return (buf, frozenΔ, frozenExact)
  IO.println s!"  running {k} accumulated micro-batches…"; (← IO.getStdout).flush
  let (accOut, frozenΔ, frozenExact) ← cycle k

  let rel (a b : ByteArray) (off : Nat) : Float :=
    let (d, m, _) := cmp a b off off nP; d / max m 1e-30
  let worstRel (a b : ByteArray) : Float := max (rel a b 0) (max (rel a b nP) (rel a b (2 * nP)))

  let (_, _, eT) := cmp accOut admOut 0 0 nP
  let tie := worstRel accOut admOut
  let ctl := worstRel revOut admOut
  let (tieP, tieBad) := spread accOut admOut pShapes
  let (ctlP, ctlBad) := spread revOut admOut pShapes
  let (dMove, _, _) := cmp accOut θ 0 0 nP
  IO.println ""
  IO.println s!"  ① accumulate micro-batches leave [θ|m|v] frozen"
  IO.println s!"      max abs Δ {frozenΔ}   bit-exact {frozenExact}/{3 * nP}"
  IO.println s!"  ② after {k} micro-steps == one batch-{bs} step on the concatenation"
  IO.println s!"      θ' rel {rel accOut admOut 0}   m' rel {rel accOut admOut nP}   v' rel {rel accOut admOut (2 * nP)}   θ' bit-exact {eT}/{nP}"
  IO.println s!"      θ' per-param worst norm-rel {tieP}, {tieBad}/{pShapes.size} params > 1e-4"
  IO.println s!"  ≈ reordering control: batch-{bs} on the reversed rows vs batch-{bs}"
  IO.println s!"      worst region rel {ctl}; θ' per-param worst {ctlP}, {ctlBad}/{pShapes.size} params > 1e-4"
  IO.println s!"      (θ' moved from θ by {dMove} — the degeneracy guard)"

  let (badOut, _, _) ← cycle (k - 1)
  let wrong := worstRel badOut admOut
  IO.println s!"  ⟂ WRONG-k control: apply after {k - 1} micro-batches, graph still divides by {k}"
  IO.println s!"      worst region rel {wrong}"

  IO.println ""
  if dMove < 1e-6 then
    throw <| IO.userError s!"DEGENERATE: θ' == θ (max Δ {dMove})"
  if frozenExact != 3 * nP then
    throw <| IO.userError s!"① FAILED: an accumulate micro-batch moved the optimizer state \
({frozenExact}/{3 * nP} bit-exact, max Δ {frozenΔ})"
  if ctl == 0.0 then
    throw <| IO.userError "CONTROL DEAD: the reversed batch reproduces bit-exactly, so it calibrates nothing"
  if tie > slack * ctl || tieBad > max (slack.toUInt64.toNat * ctlBad) 2 then
    throw <| IO.userError s!"② FAILED: {k} accumulated micro-batches != one batch-{bs} step \
(worst region rel {tie} vs reordering control {ctl}; {tieBad} vs {ctlBad} params off by > 1e-4). If \
v' is the outlier, suspect the 1/k² on %ob2"
  if wrong <= 10.0 * tie then
    throw <| IO.userError s!"CONTROL DEAD: applying after {k - 1} micro-batches fits as well as \
after {k} ({wrong} vs a tie of {tie})"
  IO.println s!"  ✅ TIED: {k} accumulated micro-batches of {mb} on different rows are ONE batch-{bs} \
step — worst region rel {tie} ({tie / ctl}x the reordering control's {ctl}), {tieBad}/{pShapes.size} \
params off by > 1e-4 (control {ctlBad}), [θ|m|v] bit-exact across every accumulate micro-batch, and a \
wrong-k control that misses by {wrong} ({wrong / max tie 1e-30}x the tie)."
