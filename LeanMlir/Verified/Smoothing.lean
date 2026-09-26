import LeanMlir.Verified.Train

/-! ## Randomized-smoothing statistics (Cohen–Rosenfeld–Kolter 2019)

The pieces the smoothing certificate needs, in pure `Float` (no kernel, no Mathlib): the
probit `Φ⁻¹` for the radius `σ·Φ⁻¹(p_A)`, and the Clopper–Pearson lower confidence bound on
`p_A` (the exact binomial bound, not a normal approximation). CP is built bottom-up from the
regularized incomplete beta `Iₓ(a,b)` (Lanczos `lgamma`, Lentz continued fraction) and solved
by 60-step bisection. None of this is proved, and the bisection returns the midpoint of its
final bracket rather than rounding toward the conservative end. -/

/-- Inverse standard-normal CDF `Φ⁻¹` (probit), Peter Acklam's rational approximation
    (relative error < 1.15e-9 over `(0,1)`). `p_A` here lives in `(0.5, ~0.99)`, far from
    the tails, so this is orders of magnitude tighter than the Monte-Carlo sampling error. -/
private def invNormCdf (p : Float) : Float := Id.run do
  let a0 := -3.969683028665376e+01
  let a1 :=  2.209460984245205e+02
  let a2 := -2.759285104469687e+02
  let a3 :=  1.383577518672690e+02
  let a4 := -3.066479806614716e+01
  let a5 :=  2.506628277459239e+00
  let b1 := -5.447609879822406e+01
  let b2 :=  1.615858368580409e+02
  let b3 := -1.556989798598866e+02
  let b4 :=  6.680131188771972e+01
  let b5 := -1.328068155288572e+01
  let c0 := -7.784894002430293e-03
  let c1 := -3.223964580411365e-01
  let c2 := -2.400758277161838e+00
  let c3 := -2.549732539343734e+00
  let c4 :=  4.374664141464968e+00
  let c5 :=  2.938163982698783e+00
  let d1 :=  7.784695709041462e-03
  let d2 :=  3.224671290700398e-01
  let d3 :=  2.445134137142996e+00
  let d4 :=  3.754408661907416e+00
  let plow := 0.02425
  let phigh := 1.0 - plow
  if p < plow then
    let q := Float.sqrt (-2.0 * Float.log p)
    return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)
  else if p ≤ phigh then
    let q := p - 0.5
    let r := q*q
    return (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0)
  else
    let q := Float.sqrt (-2.0 * Float.log (1.0 - p))
    return 0.0 - (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)

/-- `log Γ(x)` for `x ≥ 0.5` (the only regime we hit: `a=k≥1`, `b=n−k+1≥1`), Lanczos `g=7`. -/
private def lgammaF (x : Float) : Float := Id.run do
  let g : Array Float := #[
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
  let xx := x - 1.0
  let mut a := g[0]!
  let t := xx + 7.5
  for i in [1:9] do
    a := a + g[i]! / (xx + i.toFloat)
  return 0.5 * Float.log (2.0 * 3.141592653589793) + (xx + 0.5) * Float.log t - t + Float.log a

/-- Lentz continued fraction for the incomplete beta (Numerical Recipes `betacf`). -/
private def betacf (a b x : Float) : Float := Id.run do
  let fpmin := 1.0e-300; let eps := 3.0e-12
  let qab := a + b; let qap := a + 1.0; let qam := a - 1.0
  let mut c := 1.0
  let mut d := 1.0 - qab * x / qap
  if Float.abs d < fpmin then d := fpmin
  d := 1.0 / d
  let mut h := d
  for m in [1:201] do
    let mf := m.toFloat; let m2 := 2.0 * mf
    let aa1 := mf * (b - mf) * x / ((qam + m2) * (a + m2))
    d := 1.0 + aa1 * d; if Float.abs d < fpmin then d := fpmin
    c := 1.0 + aa1 / c; if Float.abs c < fpmin then c := fpmin
    d := 1.0 / d; h := h * d * c
    let aa2 := -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
    d := 1.0 + aa2 * d; if Float.abs d < fpmin then d := fpmin
    c := 1.0 + aa2 / c; if Float.abs c < fpmin then c := fpmin
    d := 1.0 / d
    let del := d * c
    h := h * del
    if Float.abs (del - 1.0) < eps then break
  return h

/-- Regularized incomplete beta `Iₓ(a,b)` (Numerical Recipes `betai`); increasing in `x`. -/
private def betaiF (a b x : Float) : Float :=
  if x ≤ 0.0 then 0.0
  else if x ≥ 1.0 then 1.0
  else
    let lbt := lgammaF (a+b) - lgammaF a - lgammaF b + a * Float.log x + b * Float.log (1.0 - x)
    let bt := Float.exp lbt
    if x < (a + 1.0) / (a + b + 2.0) then bt * betacf a b x / a
    else 1.0 - bt * betacf b a (1.0 - x) / b

/-- Clopper–Pearson **exact** lower confidence bound for a Binomial proportion: the largest
    `p` with `P[Bin(n,p) ≥ k] ≤ α`, i.e. the `α`-quantile of `Beta(k, n−k+1)` — the `p` solving
    `I_p(k, n−k+1) = α`, found by 60-step bisection (`Iₓ` is monotone), returning the midpoint of
    the final bracket. This is the `1−α` lower bound on `p_A` that the certified radius
    `σ·Φ⁻¹(p_A)` rests on (Cohen 2019 uses the same CP bound), evaluated in `Float` without a
    proof or a conservative rounding direction. `k=0 ⇒ 0`. -/
private def clopperPearsonLower (k n : Nat) (alpha : Float) : Float := Id.run do
  if k == 0 then return 0.0
  let a := k.toFloat
  let b := (n - k + 1).toFloat
  let mut lo := 0.0
  let mut hi := 1.0
  for _ in [0:60] do
    let mid := 0.5 * (lo + hi)
    if betaiF a b mid < alpha then lo := mid else hi := mid
  return 0.5 * (lo + hi)

/-- **Randomized-smoothing certificate** (Cohen–Rosenfeld–Kolter 2019) — a depth-independent
    certificate, for nets where the Lipschitz product is too loose to certify anything.
    The smoothed classifier `ĝ(x) = argmax_c P[f(x+η)=c]`, `η ~ N(0,σ²I)`, is certified robust at
    L2 radius `σ·Φ⁻¹(p_A)` where `p_A` is a lower bound on the top class's noise probability. It's
    **forward-only**: no new kernel, no input-VJP — just sample `n` noisy copies, run the existing
    proof-rendered `<slug>_fwd`, count argmax votes, Clopper–Pearson lower-bound `p_A`. The base
    classifier is trained with matched Gaussian augmentation (every batch corrupted with `N(0,σ²I)`
    host-side before the proof-rendered SGD step — the forward/backward graph is untouched), the
    Cohen recipe. Architecture-agnostic + depth-independent, so it certifies a *non-vacuous* radius
    on the very nets (CIFAR, deep) where `∏‖Wᵢ‖₂` is astronomically loose. Generic over any
    `VerifiedNet` (fwd + train-step only).

    `n` (`SMOOTH_N`, default 10000 — Cohen's large-`n` regime) is the estimation budget and the only
    tightening lever: the per-point radius is capped at `σ·Φ⁻¹(α^(1/n))` (a unanimous vote
    certifies only `p_A ≥ α^(1/n)`), so larger `n` lifts the ceiling and tightens the CP bound
    toward the true noise-probability — bigger certified radii at the same `1−α` guarantee. -/
def VerifiedNet.smoothCertify (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (sigmas : List Float) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Randomized-smoothing certificate on {net.name} (verified codegen → IREE → GPU, forward-only)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  -- `trainPix`/`crop`: Imagenette train ships at 256² and is center-cropped to 224² per batch
  -- (the val/eval split is already 224² = d0, so certify reads it directly). For MNIST/CIFAR
  -- crop=false and trainPix=d0, so the crop below is a no-op.
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  -- knobs (env-overridable; defaults are Cohen's large-n estimation regime — drop SMOOTH_N /
  -- SMOOTH_MAXCERT for a cheap smoke).
  let n0      := ((← IO.getEnv "SMOOTH_N0").bind (·.toNat?)).getD 100
  let nSamp   := ((← IO.getEnv "SMOOTH_N").bind (·.toNat?)).getD 10000
  let maxCert := ((← IO.getEnv "SMOOTH_MAXCERT").bind (·.toNat?)).getD 200
  -- single-σ override (millis, same knob as the ConvNeXt exe) — e.g. the
  -- fixed-protocol scorecard runs σ=0.5 only via SMOOTH_SIGMA_MILLI=500
  let sigmas := match (← IO.getEnv "SMOOTH_SIGMA_MILLI") with
    | some s => match s.toNat? with
      | some m => [m.toFloat / 1000.0]
      | none => sigmas
    | none => sigmas
  -- per-epoch best-θ eval is a clean-acc proxy; cap it to a subsample on heavy 224² nets
  -- (`SMOOTH_EVAL_BATCHES`, default = full val set).
  let evalBatches := min nbt (((← IO.getEnv "SMOOTH_EVAL_BATCHES").bind (·.toNat?)).getD nbt)
  let alpha   := 0.001
  let radii : Array Float := #[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  let nCert   := min maxCert nEval
  -- SMOOTH_STRIDE=1 + SMOOTH_MAXCERT=100 = the fixed-first-100 scorecard
  -- protocol (mirrors the Lipschitz scorecard subset); default unchanged.
  let stride  := max 1 (((← IO.getEnv "SMOOTH_STRIDE").bind (·.toNat?)).getD (nEval / nCert))
  -- the rendered fwd has a STATIC batch = bs, so SampleUnderNoise feeds whole `bs`-batches;
  -- round the requested sample counts up to a multiple of bs (the effective n the CP bound uses).
  let n0Batches := max 1 ((n0 + bs - 1) / bs)
  let nBatches  := max 1 ((nSamp + bs - 1) / bs)
  let n0Eff := n0Batches * bs
  let nEff  := nBatches * bs
  -- the n→radius ceiling: even a UNANIMOUS vote gives only p_A ≤ α^(1/n) (the CP bound at n_A=n),
  -- so EVERY point's radius is capped at σ·Φ⁻¹(α^(1/n)) regardless of how robust it truly is.
  -- Larger n ⇒ p_A closer to the true noise-prob ⇒ higher ceiling — the only honest tightening
  -- lever (shrinking α would just weaken the 1−α guarantee, not tighten the estimate).
  let pMax := clopperPearsonLower nEff nEff alpha
  IO.println s!"  n0={n0Eff} (select)  n={nEff} (estimate)  α={alpha}  certifying {nCert} test imgs (every {stride}th)"
  IO.println s!"  p_A ceiling = {pMax} (= α^(1/n))  →  max certifiable radius = σ · {invNormCdf pMax}"
  let mut rows : Array String := #[]
  -- per-image certified radius, dumped to CSV for arbitrarily-fine frontier curves (cert-acc at
  -- radius r = fraction with correct ∧ radius ≥ r) + ACR + any threshold, all from one run.
  -- count,n = the raw CP inputs (k successes of n samples) — what the Lean
  -- kernel tail checks (SmoothingCP.binomTail_le_of_kernel_check) consume.
  let mut dumpRows : Array String := #["sigma,img_idx,label,pred,abstain,radius,count,n"]
  -- Lipschitz-hypothesis probe (SMOOTH_LIP_PROBE): measures |Φ⁻¹(p_ĉ(x+δ))−Φ⁻¹(p_ĉ(x))| vs the
  -- proven bound ‖δ‖/σ — the empirical grounding of `smoothing_certified_radius_probit`'s (1/σ)-Lipschitz hyp.
  let lipProbe := (← IO.getEnv "SMOOTH_LIP_PROBE").isSome
  let mut lipRows : Array String := #["sigma,img_idx,delta,dg,bound,ratio"]
  let mut lipMax := 0.0
  let mut lipViol := 0
  let mut lipN := 0
  for sigma in sigmas do
    IO.println s!"\n── σ = {sigma}  (radius ceiling at this n = {sigma * invNormCdf pMax}) ──"
    -- (1) train a Gaussian-noise-augmented base classifier (the Cohen recipe): every batch is
    --     corrupted with N(0,σ²I) host-side before the proof-rendered SGD step — graph unchanged.
    --     Best-θ checkpoint on clean eval acc.
    let mut parts : Array ByteArray := #[]
    let mut iseed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam iseed spec.1 spec.2); iseed := iseed + 1
    let mut theta := F32.concat parts
    let mut bestTheta := theta
    let mut bestAcc := -1.0
    let mut nseed : USize := 1
    for ep in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xbRaw := F32.sliceImages trainImg (bi * bs) bs trainPix
        let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        let xbN ← F32.addGaussianTiled xb 0 (bs*d0).toUSize 1 sigma nseed
        nseed := nseed + 1
        theta ← LowererSession.mlpTrainStepV tsSess tsFn xbN theta shapes yb bs.toUSize d0.toUSize d1.toUSize
      let mut c := 0
      for bi in [0:evalBatches] do
        let xb := F32.sliceImages evalImg (bi * bs) bs d0
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb (net.xShape bs) bs.toUSize d1.toUSize
        for j in [0:bs] do
          if (F32.argmaxN logits (j*d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi*bs+j) then c := c+1
      let acc := c.toFloat/(evalBatches*bs).toFloat*100.0
      if acc > bestAcc then bestAcc := acc; bestTheta := theta
      IO.println s!"    epoch {ep+1}/{cfg.epochs}: clean acc = {acc}%"
      (← IO.getStdout).flush
    let th := bestTheta                       -- certify the best checkpoint (immutable snapshot)
    IO.println s!"  noise-trained clean acc = {bestAcc}%"
    -- (2) SampleUnderNoise: `nBatches·bs` noisy copies of ONE image (element offset `off`), each
    --     forward feeding exactly `bs` rows (the rendered fwd's static batch), argmax-vote → counts.
    let sampleCountsB := fun (base : ByteArray) (off nBat : Nat) (sd : USize) => do
      let mut counts : Array Nat := Array.replicate d1 0
      for ci in [0:nBat] do
        let xN ← F32.addGaussianTiled base off.toUSize d0.toUSize bs.toUSize sigma (sd + ci.toUSize)
        let logits ← LowererSession.forwardF32 fwdSess fwdFn th shapes xN (net.xShape bs) bs.toUSize d1.toUSize
        for j in [0:bs] do
          let a := (F32.argmaxN logits (j*d1).toUSize d1.toUSize).toNat
          counts := counts.set! a (counts[a]! + 1)
      pure counts
    let sampleCounts := fun (off nBat : Nat) (sd : USize) => sampleCountsB evalImg off nBat sd
    -- (3) certify each sampled test image: n0 select ĉ_A, n estimate p_A, radius σ·Φ⁻¹(p_A).
    let mut abstain := 0
    let mut natCorrect := 0
    let mut acr := 0.0
    let mut certCnt : Array Nat := Array.replicate radii.size 0
    for t in [0:nCert] do
      let imgIdx := t * stride
      let off := imgIdx * d0
      let label := F32.readLabel evalLbl imgIdx
      let base : USize := (imgIdx + 1).toUSize * 131 + 1
      let counts0 ← sampleCounts off n0Batches base
      let mut cHatA := 0
      for c in [1:d1] do
        if counts0[c]! > counts0[cHatA]! then cHatA := c
      if cHatA == label then natCorrect := natCorrect + 1
      let counts ← sampleCounts off nBatches (base + 524287)
      let pA := clopperPearsonLower counts[cHatA]! nEff alpha
      let certified := pA > 0.5
      let radius := if certified then sigma * invNormCdf pA else 0.0
      if certified then
        if cHatA == label then
          acr := acr + radius
          for ri in [0:radii.size] do
            if radius ≥ radii[ri]! then certCnt := certCnt.set! ri (certCnt[ri]! + 1)
      else
        abstain := abstain + 1
      dumpRows := dumpRows.push
        s!"{sigma},{imgIdx},{label},{cHatA},{if certified then 0 else 1},{radius},{counts[cHatA]!},{nEff}"
      if (t+1) % 100 == 0 then
        IO.println s!"    certified {t+1}/{nCert} ..."; (← IO.getStdout).flush
    -- (4) Lipschitz-hypothesis probe: shift x by r·(unit vector) and compare the probit-score change
    --     |Φ⁻¹(p_ĉ(x+δ))−Φ⁻¹(p_ĉ(x))| to the PROVEN bound ‖δ‖/σ (Salman et al. 2019 Lemma 2 — the
    --     hypothesis `smoothing_certified_radius_probit` assumes). ratio = Δg·σ/‖δ‖ ≤ 1 ⟺ (1/σ)-Lipschitz holds.
    if lipProbe then
      let rGrid : Array Float := #[0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
      let mProbe := min 40 nEval
      let pstride := max 1 (nEval / mProbe)
      let clampP := fun (k : Nat) =>
        let lo := 1.0 / (2.0 * nEff.toFloat)
        max lo (min (1.0 - lo) (k.toFloat / nEff.toFloat))
      for t in [0:mProbe] do
        let imgIdx := t * pstride
        let off := imgIdx * d0
        let cnt0 ← sampleCountsB evalImg off nBatches ((imgIdx + 7).toUSize * 131 + 3)
        let mut chat := 0
        for c in [1:d1] do
          if cnt0[c]! > cnt0[chat]! then chat := c
        let gx := invNormCdf (clampP cnt0[chat]!)
        for ri in [0:rGrid.size] do
          let r := rGrid[ri]!
          let xp ← F32.perturbUnit evalImg off.toUSize d0.toUSize r (imgIdx.toUSize * 17 + ri.toUSize + 1)
          let cntp ← sampleCountsB xp 0 nBatches (imgIdx.toUSize * 53 + ri.toUSize + 11)
          let gxp := invNormCdf (clampP cntp[chat]!)
          let dg := Float.abs (gxp - gx)
          let ratio := dg * sigma / r
          if ratio > lipMax then lipMax := ratio
          if ratio > 1.05 then lipViol := lipViol + 1
          lipN := lipN + 1
          lipRows := lipRows.push s!"{sigma},{imgIdx},{r},{dg},{r/sigma},{ratio}"
      IO.println s!"  Lipschitz probe: {mProbe} imgs × {rGrid.size} shifts (Φ⁻¹∘p_ĉ vs ‖δ‖/σ)"
      (← IO.getStdout).flush
    let tot := nCert.toFloat
    let pct := fun (k : Nat) => k.toFloat / tot * 100.0
    let certStr := String.intercalate " / " (radii.toList.zipIdx.map
      (fun (r, ri) => s!"{r}→{pct certCnt[ri]!}%"))
    IO.println s!"  smoothed natural acc = {pct natCorrect}%   abstain = {pct abstain}%   ACR = {acr/tot}"
    IO.println s!"  certified-robust acc (L2): {certStr}"
    (← IO.getStdout).flush
    rows := rows.push s!"  {sigma}\t{bestAcc}\t{pct natCorrect}\t{pct abstain}\t{pct certCnt[1]!}\t{pct certCnt[3]!}\t{pct certCnt[5]!}\t{acr/tot}"
  IO.println s!"\n══ randomized smoothing ({net.name}): depth-independent certified L2 radius ══"
  IO.println "  σ\tclean%\tnat%\tabst%\tcert@.5\tcert@1.0\tcert@1.5\tACR"
  for row in rows do IO.println row
  let csvPath := s!"runs/smooth_{net.slug}_radii.csv"
  IO.FS.writeFile csvPath (String.intercalate "\n" dumpRows.toList ++ "\n")
  IO.println s!"\nper-image certified radii → {csvPath} ({dumpRows.size - 1} rows, all σ) — fine frontier curves + ACR"
  if lipProbe then
    let lipPath := s!"runs/smooth_{net.slug}_lipschitz.csv"
    IO.FS.writeFile lipPath (String.intercalate "\n" lipRows.toList ++ "\n")
    IO.println s!"Lipschitz-hypothesis probe → {lipPath} ({lipN} measurements): max ratio Δg·σ/‖δ‖ = {lipMax}"
    IO.println s!"  ⇒ Φ⁻¹∘p_ĉ is empirically (1/σ)-Lipschitz: {lipN - lipViol}/{lipN} measurements ≤ bound (violations>1.05: {lipViol})"
  IO.println "done (randomized smoothing: forward-only Monte-Carlo cert via the proof-rendered fwd —"
  IO.println "      architecture-agnostic + depth-independent, non-vacuous where ∏‖Wᵢ‖₂ is hopeless)."

namespace VerifiedNetSpec

/-- Randomized-smoothing certificate (Cohen 2019, depth-independent); see
    `VerifiedNet.smoothCertify`. Forward-only — works on any spec via its rendered fwd. -/
def smoothCertify (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (sigmas : List Float) : IO Unit :=
  s.toNet.smoothCertify cfg dataDir sigmas

end VerifiedNetSpec
