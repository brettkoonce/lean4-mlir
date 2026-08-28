import LeanMlir
import LeanMlir.VerifiedNets

/-! # `mnist-ddpm-score` — the image DDPM's first real number

    ⭐ **The 2-D demo's argument, moved onto images.** On a spiral the ground
    truth is a second point cloud, so `planning/diffusion_2d_demo.md` §4 can
    report cell recall and an energy distance instead of "does that look right
    to you". On MNIST the ground truth is a **classifier that already knows what
    a digit looks like** — and this repo has one whose math VJP is proven:
    Chapter 3's `cnnVerified` (`LeanMlir/VerifiedNets.lean`), 98.75 % at ten
    epochs, running from the committed `verified_mlir/cnn_fwd.mlir`.

    Push the generated samples through it and every metric of §4 comes back:

    * **class coverage** — how many of the ten digits hold ≥ 1 % of the mass.
      The direct analogue of the 8-gaussians' 8/8 mode recall, and the single
      most diagnostic number there.
    * **per-class mass** against a true 10 %. On 2-D this is where the entire
      residual turned out to live once recall saturated.
    * **confidence** — mean max-softmax, with real MNIST as the control.
    * **energy distance in the classifier's 10-d output**, generated against
      real, with a real-vs-real floor. Literally §4's statistic, computed in the
      space the classifier maps images into.

    ⚠ **No conditioning, and therefore no accuracy number.** This is the
    unconditional half: it says whether the samples are digits and whether all
    ten show up, not whether a requested digit came back. That needs a class
    embedding in the denoiser and is a separate piece of work.

    ⚠⚠ **Coverage and confidence are not enough on their own**, for exactly the
    reason `planning/diffusion_2d_demo.md` §5.7 records: a classifier scores
    confidently on things that are not from the data distribution, so a model
    emitting one canonical seven would post perfect confidence. That is the
    checkerboard mistake — 8/8 recall and a passing energy distance while a
    third of the mass sits in provably empty squares. The per-class mass and the
    energy distance are here so that failure has somewhere to show up.

    This driver only GENERATES and CLASSIFIES; `scripts/mnist_ddpm_score.py`
    does the statistics, the same split the 2-D demo uses.

    Run:
    ```
    lake exe mnist-ddpm-train data                                   # if no ckpt
    LEAN_MLIR_DUMP_PARAMS=.lake/build/cnn_verified_params.bin \
      lake exe mnist-cnn-verified data                               # 10 ep, ~50 s
    lake exe mnist-ddpm-score 1024 50
    python3 scripts/mnist_ddpm_score.py
    ```
-/

/-- ⚠ The THIRD copy of this spec (`MainMnistDdpmTrain`, `MainMnistDdpmSample`
    hold the other two). It is not shared because both of those are executables
    with their own `main`. The guard against drift is downstream and loud: the
    checkpoint is keyed by `spec.buildPrefix`, which is derived from `name`, and
    this driver checks the blob is exactly `spec.totalParams` floats. -/
def tinyDdpmUnet (centred : Bool := true) : NetSpec where
  name := if centred then "tiny DDPM UNet T-cond centered (MNIST 28x28x1)"
                     else "tiny DDPM UNet T-cond (MNIST 28x28x1)"
  imageH := 28
  imageW := 28
  layers := [
    .unetDown 2 16,
    .unetDown 16 32,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .unetUp 64 32,
    .unetUp 32 16,
    .conv2d 16 1 1 .same .identity
  ]

def main (args : List String) : IO Unit := do
  let nums   := args.filterMap String.toNat?
  let nGen   := (nums[0]?).getD 1024
  let nSteps := (nums[1]?).getD 50
  -- ⭐ The Score-SDE samplers, on the same weights, ported from the 2-D driver
  -- through the shared `Ddpm` schedule so the two cannot drift. The toy found
  -- the reverse SDE beating DDIM at every budget and saturating by NFE 50; this
  -- is where that either survives at image scale or does not.
  -- ⚠ `nSteps` is the NFE BUDGET. Heun spends two evaluations per step.
  let sampler := (args.find? fun a => Ddpm.samplerNfe.any (·.1 == a)).getD "ddim"
  let nfe := (Ddpm.samplerNfe.lookup sampler).getD 1
  let solverSteps := max 1 (nSteps / nfe)
  let dataDir := "data"
  let out := ".lake/build"

  -- ── the generator ────────────────────────────────────────────────────────
  -- `raw` scores the uncentred ablation arm; see the trainer's note.
  let raw  := args.any (· == "raw")
  let spec := tinyDdpmUnet (centred := !raw)
  let pfx  := spec.buildPrefix
  let B : Nat := 16                    -- the batch the cached eval graph holds
  let nPix : Nat := spec.imageH * spec.imageW
  let paramsPath := s!"{pfx}_params.bin"
  let bnPath     := s!"{pfx}_bn_stats.bin"
  for p in [paramsPath, bnPath] do
    unless ← System.FilePath.pathExists p do
      throw <| IO.userError s!"missing DDPM checkpoint {p} — run: lake exe mnist-ddpm-train data"
  let dParams ← IO.FS.readBinFile paramsPath
  unless F32.size dParams == spec.totalParams do
    throw <| IO.userError s!"DDPM checkpoint is {F32.size dParams} floats but this spec wants \
{spec.totalParams} — the three copies of `tinyDdpmUnet` have drifted"
  let bnStats ← IO.FS.readBinFile bnPath
  let evalMlirPath := s!"{pfx}_fwd_eval.mlir"
  IO.FS.writeFile evalMlirPath (MlirCodegen.generateEval spec B)
  let evalArt ← NetSpec.graphArtifact pfx "fwd_eval"
  let genSess ← LowererSession.create evalArt
  let evalParams := dParams.append bnStats
  let evalShapes := spec.evalShapesBA
  let xShape := spec.xShape B

  let T : Nat := 1000
  let alphaBar ← Ddpm.cosineSchedule T.toUSize
  let stride := T / nSteps
  let nBatch := (nGen + B - 1) / B
  IO.eprintln s!"sampling {nBatch * B} images, {nSteps} DDIM steps, batch {B}..."
  -- ⚠ Uniform-in-t, which the toy measured as the SDE's best grid (log-ᾱ made it
  -- 25× worse). The explicit ODE solvers want the opposite grid; that this
  -- driver offers only one is a limitation, and the reason `euler`/`heun` here
  -- are not a fair test of those methods.
  let tHi := (T - 1).toFloat / T.toFloat
  let tLo := 1.0 / T.toFloat
  let nAll := (B * nPix).toUSize
  -- ⚠ QUANTIZED to the 1000 training indices, exactly as in the 2-D driver: the
  -- t-channel encoder takes an integer because the model was trained on one.
  let epsAt := fun (xv : ByteArray) (t : Float) => do
    let r := Float.round (t * T.toFloat)
    let r := if r < 0.0 then 0.0
             else if r > (T - 1).toFloat then (T - 1).toFloat else r
    let xc ← Ddpm.prependTChannelScalar xv B.toUSize 1 spec.imageH.toUSize
               spec.imageW.toUSize r.toUInt64.toUSize T.toUSize
    LowererSession.forwardF32 genSess spec.evalFnName evalParams evalShapes
      xc xShape B.toUSize nPix.toUSize
  IO.eprintln s!"  sampler={sampler} ({nfe} eval/step), NFE={nSteps} \
=> {solverSteps} solver steps"
  let mut acc : Array ByteArray := #[]
  for bi in [:nBatch] do
    -- ⚠ One seed per BATCH, and they are consecutive. That is safe only because
    -- `Ddpm.sampleNoise` was fixed on 2026-08-28 — seeded the old way, nearby
    -- seeds shared a Box–Muller radius and every batch would have started from
    -- the same shell. `tests/TestSampleNoiseSeeding.lean` is the gate; this is
    -- the first caller written to depend on it.
    let mut x ← Ddpm.sampleNoise (B * nPix).toUSize (0xc0ffee + bi).toUSize
    if sampler != "ddim" then
     for k in [:solverSteps] do
      let t  := tHi + (tLo - tHi) * k.toFloat / solverSteps.toFloat
      let tn := tHi + (tLo - tHi) * (k + 1).toFloat / solverSteps.toFloat
      let h  := tn - t
      let b1 := Ddpm.betaC t
      let s1 := Ddpm.sigC t
      let e1 ← epsAt x t
      if sampler == "heun" then
        let xt ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * 0.5 * b1 / s1) nAll
        let b2 := Ddpm.betaC tn
        let s2 := Ddpm.sigC tn
        let e2 ← epsAt xt tn
        let mut a2 ← F32.scaleShift x (1.0 - h * 0.25 * b1) 0.0
        a2 ← F32.axpySlice a2 0 e1 0 nAll (h * 0.25 * b1 / s1)
        a2 ← F32.axpySlice a2 0 xt 0 nAll (-(h * 0.25 * b2))
        a2 ← F32.axpySlice a2 0 e2 0 nAll (h * 0.25 * b2 / s2)
        x := a2
      else if sampler == "sde" then
        x ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * b1 / s1) nAll
        let z ← Ddpm.sampleNoise nAll (bi * 131071 + k * 8191 + 17).toUSize
        x ← Ddpm.ddimStep x z 1.0 (Float.sqrt (b1 * (-h))) nAll
      else
        x ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * 0.5 * b1 / s1) nAll
    else
     for k in [:nSteps] do
      let t     := T - 1 - k * stride
      let tPrev := if k + 1 == nSteps then 0 else T - 1 - (k + 1) * stride
      let xc ← Ddpm.prependTChannelScalar x B.toUSize 1 spec.imageH.toUSize
                 spec.imageW.toUSize t.toUSize T.toUSize
      let eps ← LowererSession.forwardF32 genSess spec.evalFnName
                  evalParams evalShapes xc xShape B.toUSize nPix.toUSize
      let abT := F32.read alphaBar t.toUSize
      let abP := if tPrev == 0 then 0.9999 else F32.read alphaBar tPrev.toUSize
      let a := Float.sqrt abP / Float.sqrt abT
      let b := Float.sqrt (1.0 - abP) - a * Float.sqrt (1.0 - abT)
      x ← Ddpm.ddimStep x eps a b (B * nPix).toUSize
    acc := acc.push x
    if bi % 8 == 0 || bi + 1 == nBatch then
      IO.eprintln s!"  batch {bi+1}/{nBatch}"
  -- ⚠ Invert the trainer's [-1, 1] centring. `cnnVerified` was trained on the
  -- loader's [0, 1] images, so classifying without this measures the model
  -- against a scale it never saw — and the pixel-moment row would be comparing
  -- two different units.
  let gen ← if raw then pure (F32.concat acc)
            else F32.scaleShift (F32.concat acc) 0.5 0.5
  let nSamp := F32.size gen / nPix

  -- ── the Chapter-3 classifier, from its committed verified render ─────────
  let net := cnnVerified.toNet
  let cnnPath := ".lake/build/cnn_verified_params.bin"
  unless ← System.FilePath.pathExists cnnPath do
    throw <| IO.userError s!"missing {cnnPath} — run: LEAN_MLIR_DUMP_PARAMS={cnnPath} \
lake exe mnist-cnn-verified data"
  let raw ← IO.FS.readBinFile cnnPath
  -- ⚠ `VerifiedNet.train`'s dump carries a trailing report-only loss float
  -- (`cnnVerified.lossSlot`), so take the parameter PREFIX rather than the file.
  unless F32.size raw ≥ net.nParams do
    throw <| IO.userError s!"{cnnPath} holds {F32.size raw} floats, need ≥ {net.nParams}"
  let cnnParams := F32.slice raw 0 net.nParams
  let cbs : Nat := 128                 -- verified_mlir/cnn_fwd.mlir is batch-128
  let cnnSess ← LowererSession.create "verified_mlir/cnn_fwd.mlir"
  let cnnShapes := net.shapesBA
  let cnnXShape := net.xShape cbs
  let nc := net.nClasses

  -- Classify `n` images (flat `[n, 784]`, values in `[0,1]`), padding the last
  -- batch. Returns flat `[n, 10]` logits.
  let classify := fun (imgs : ByteArray) (n : Nat) => do
    let mut outs : Array ByteArray := #[]
    let nb := (n + cbs - 1) / cbs
    for bi in [:nb] do
      let xb := F32.sliceImagesPad imgs (bi * cbs) cbs nPix n
      let lg ← LowererSession.forwardF32 cnnSess "m.cnn_fwd" cnnParams cnnShapes
                 xb cnnXShape cbs.toUSize nc.toUSize
      outs := outs.push lg
    return F32.slice (F32.concat outs) 0 (n * nc)

  IO.eprintln s!"classifying {nSamp} generated samples..."
  let logitsGen ← classify gen nSamp

  -- ⭐ NEGATIVE CONTROL: unstructured pixels carrying MNIST's own first two
  -- moments (mean 0.1325, sd 0.3105). Without it, "169x the floor" says the
  -- samples are far from real without saying how far a model that learned
  -- NOTHING would be, and every reader has to guess the scale. The 2-D demo
  -- has the same shape of control in its true-vs-true floor at the other end.
  let noiseRaw ← Ddpm.sampleNoise (nSamp * nPix).toUSize 0x51ee7
  let noise ← F32.scaleShift noiseRaw 0.3105 0.1325
  IO.eprintln s!"classifying {nSamp} noise images (negative control)..."
  let logitsNoise ← classify noise nSamp

  -- The control: the same classifier on the real test split. Its accuracy is
  -- also the check that the weights loaded correctly — a mispacked blob scores
  -- at chance, and every number below would then be measuring nothing.
  let (testImg, nTest) ← F32.loadIdxImages s!"{dataDir}/t10k-images-idx3-ubyte"
  let (testLbl, _)     ← F32.loadIdxLabels s!"{dataDir}/t10k-labels-idx1-ubyte"
  IO.eprintln s!"classifying {nTest} real test images (control)..."
  let logitsReal ← classify testImg nTest

  IO.FS.writeBinFile s!"{out}/mnist_ddpm_samples.bin" gen
  IO.FS.writeBinFile s!"{out}/mnist_ddpm_logits_gen.bin" logitsGen
  IO.FS.writeBinFile s!"{out}/mnist_ddpm_logits_real.bin" logitsReal
  IO.FS.writeBinFile s!"{out}/mnist_ddpm_logits_noise.bin" logitsNoise
  -- The control's own pixels, so the scorer MEASURES its moments rather than
  -- restating the constants above.
  IO.FS.writeBinFile s!"{out}/mnist_ddpm_noise.bin" noise
  IO.FS.writeBinFile s!"{out}/mnist_ddpm_labels_real.bin" (F32.sliceLabels testLbl 0 nTest)
  IO.eprintln s!"wrote {nSamp} samples + logits ({nSamp}x{nc} gen, {nTest}x{nc} real)"
  IO.eprintln "▶ score it: python3 scripts/mnist_ddpm_score.py"
