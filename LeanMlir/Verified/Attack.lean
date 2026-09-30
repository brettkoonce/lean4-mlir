import LeanMlir.Verified.Train
import LeanMlir.Verified.PgdGen

/-! # PGD attacks and the spectral-norm training studies

Each attack trains a verified net on its proof-rendered train step (`Verified.Train`'s driver),
then runs PGD through the runtime with a `Verified.PgdGen` kernel, and reports clean vs
adversarial accuracy over an ε sweep next to the Lipschitz-margin certificate: the radius
`m/(√2·L)` of `lipschitz_margin_certified_radius`, at `L` the product of per-layer upper bounds
(`denseLip`, `convLip`; ReLU and the disjoint 2×2 max-pools are 1-Lipschitz). The `Spectral`
variants rescale the weights toward a norm ball every few steps (`projectSpectral`).

The certificate is for the real-arithmetic net at the margin the float forward prints; the f32
evaluation error of the logits is not accounted for (`LipschitzCert.Float` does that for its
reduced model only). -/

/-- One-hot `[bs, d1]` f32 batch from int32-LE labels (1.0 = bytes 00 00 80 3F), for a possibly
    partial final batch: rows past `total` records get an all-zero row (their padded images are
    never scored, so the gradient they induce is irrelevant — the row just has to exist because
    the batch dim is baked into the compiled module). -/
private def oneHotBatchPad (labels : ByteArray) (start bs d1 total : Nat) : IO ByteArray := do
  let mut oh ← F32.const (bs * d1).toUSize 0.0
  for j in [0:min bs (total - start)] do
    let fi := j * d1 + F32.readLabel labels (start + j)
    oh := (((oh.set! (4*fi) 0).set! (4*fi+1) 0).set! (4*fi+2) 0x80).set! (4*fi+3) 0x3F
  return oh

/-! ## The certificate's Lipschitz constants

The printed certified radius is `lipschitz_margin_certified_radius` at `L = ∏ᵢ Lᵢ`, so each
layer's `Lᵢ` has to be an UPPER bound on its L2 Lipschitz constant. Power iteration cannot give
one: its Rayleigh quotient approaches `‖W‖₂` from below. The bound used here is the one
`denseE_lipschitzL2_gram2` proves: with `G = M·Mᵀ` and `H = Gᵀ·G`, `B = ‖H‖_F^{1/4}
= (Σσᵢ⁸)^{1/8} ≥ σ₁`. The power-iteration value is still computed from the same Gram and
printed beside it as the estimate, so the gap between the bound and the true norm stays
visible. -/

/-- A layer's two L2 numbers: `bound`, a Lipschitz constant a certificate-tier theorem gives,
    and `est`, the power-iteration estimate (≤ the true norm; never used as a constant). -/
private structure LipPair where
  bound : Float
  est   : Float

/-- Relative slack on every host-computed bound. The Gram sums are evaluated in `Float` (f64)
    from f32 weights, so the computed `‖H‖_F` carries accumulation error far below this factor at
    these layer sizes; the slack is the soundness margin for that rounding, not a proved one. -/
private def roundingSlack : Float := 1.000001

/-- The output-side Gram `G a b = Σⱼ M a j · M b j` (`k×k`, row-major) of the `k×n` matrix
    `M a j = get a j`, row `a` being output `a` — the orientation of `denseE`, so `G` is the
    `hG` data of `denseE_lipschitzL2_gram2`. `M` is copied into a contiguous buffer first. -/
private def gramOut (get : Nat → Nat → Float) (k n : Nat) : FloatArray := Id.run do
  let mut m : FloatArray := FloatArray.mk (Array.replicate (k*n) 0.0)
  for a in [0:k] do
    for j in [0:n] do m := m.set! (a*n+j) (get a j)
  let mut g : FloatArray := FloatArray.mk (Array.replicate (k*k) 0.0)
  for a in [0:k] do
    for b in [a:k] do
      let mut s := 0.0
      for j in [0:n] do s := s + m[a*n+j]! * m[b*n+j]!
      g := (g.set! (a*k+b) s).set! (b*k+a) s
  pure g

/-- The Schatten-8 bound from the Gram: `H = Gᵀ·G`, then `B = (Σ H²)^{1/8}` times
    `roundingSlack` — the `B` of `denseE_lipschitzL2_gram2`'s `hHF : Σ H² ≤ B⁸`. -/
private def schatten8OfGram (g : FloatArray) (k : Nat) : Float := Id.run do
  let mut s := 0.0
  for a in [0:k] do
    for b in [0:k] do
      let mut h := 0.0                       -- G is symmetric, so (GᵀG)[a,b] = Σ_c G[a,c]·G[b,c]
      for c in [0:k] do h := h + g[a*k+c]! * g[b*k+c]!
      s := s + h*h
  pure (Float.pow s 0.125 * roundingSlack)

/-- Power-iteration estimate of `σ₁ = √λ_max(G)` (a lower bound, up to convergence). -/
private def powerIterOfGram (g : FloatArray) (k : Nat) : Float := Id.run do
  let mv := fun (v : Array Float) => Id.run do
    let mut u : Array Float := Array.replicate k 0.0
    for i in [0:k] do
      let mut s := 0.0
      for j in [0:k] do s := s + g[i*k+j]! * v[j]!
      u := u.set! i s
    pure u
  let mut v : Array Float := Array.replicate k 1.0
  for _ in [0:60] do
    let u := mv v
    let mut nrm := 0.0
    for i in [0:k] do nrm := nrm + u[i]!*u[i]!
    nrm := Float.sqrt nrm
    if nrm > 1e-20 then
      for i in [0:k] do v := v.set! i (u[i]!/nrm)
  let u := mv v
  let mut lam := 0.0
  for i in [0:k] do lam := lam + v[i]! * u[i]!   -- Rayleigh quotient (‖v‖=1)
  pure (Float.sqrt lam)

/-- Both numbers for the `k×n` matrix `get` (row = output). -/
private def lipOfGet (get : Nat → Nat → Float) (k n : Nat) : LipPair :=
  let g := gramOut get k n
  { bound := schatten8OfGram g k, est := powerIterOfGram g k }

/-- A dense layer `W : [d0,d1]` (row-major, `logits = x·W + b`). Its `denseE` matrix is `Wᵀ`
    (`d1` outputs), so the Gram is the `d1×d1` `WᵀW`. The bias is a translation and leaves the
    Lipschitz constant unchanged. -/
private def denseLip (W : ByteArray) (d0 d1 : Nat) : LipPair :=
  lipOfGet (fun a j => F32.read W (j*d1+a).toUSize) d1 d0

/-- A zero-padded 2-D convolution with kernel `W : [outC, inC, kh, kw]` (row-major). Writing the
    conv as a sum over spatial taps `T = Σ_{ky,kx} S_{ky,kx} ∘ M_{ky,kx}`, each `S` a zero-filled
    shift (norm ≤ 1, at any stride) and each `M` the `[outC,inC]` channel-mixing matrix at that
    tap, the triangle inequality gives `‖T‖₂ ≤ Σ_tap ‖M_tap‖₂`; each tap's `‖M_tap‖₂` is bounded
    by its Schatten-8 bound. The per-tap bound is `denseE_lipschitzL2_gram2`; the tap-sum step
    has no Lean theorem. Both steps are loose against the exact (Sedghi–Gupta–Long) conv norm.
    `est` sums the taps' power-iteration values: the tap-sum at the true per-tap norms, not an
    estimate of `‖T‖₂`. -/
private def convLip (W : ByteArray) (outC inC kh kw : Nat) : LipPair := Id.run do
  let mut b := 0.0
  let mut e := 0.0
  for ky in [0:kh] do
    for kx in [0:kw] do
      let p := lipOfGet (fun o i => F32.read W (((o*inC+i)*kh+ky)*kw+kx).toUSize) outC inC
      b := b + p.bound
      e := e + p.est
  pure { bound := b, est := e }

/-- **Matrix-free** spectral norm `‖W‖₂` of `W : [d0,d1]` (row-major) — power iteration that
    applies `W` and `Wᵀ` as mat-vecs (`σ = ‖W v‖`, `v` the top right singular vector) instead
    of forming the `d1×d1` Gram. ~`2·d0·d1` per iteration vs `d0·d1²` for the Gram (`gramOut`), so it's
    cheap enough to call **during** training (the spectral-norm projection below). Fewer iters
    (`iters`) trade a little precision for speed. An estimate from below, so never a
    certificate constant (`denseLip` is). -/
private def specNormMV (W : ByteArray) (d0 d1 : Nat) (iters : Nat := 15) : Float := Id.run do
  let norm := fun (a : Array Float) (n : Nat) => Id.run do
    let mut s := 0.0
    for i in [0:n] do s := s + a[i]! * a[i]!
    pure (Float.sqrt s)
  let normalize := fun (a : Array Float) (n : Nat) => Id.run do
    let s := norm a n
    if s > 1e-20 then
      let mut b := a
      for i in [0:n] do b := b.set! i (a[i]! / s)
      pure b
    else pure a
  let mut v : Array Float := normalize (Array.replicate d1 1.0) d1
  let mut σ := 0.0
  for _ in [0:iters] do
    let mut u : Array Float := Array.replicate d0 0.0    -- u = W v
    for r in [0:d0] do
      let mut s := 0.0
      for cc in [0:d1] do s := s + F32.read W (r*d1+cc).toUSize * v[cc]!
      u := u.set! r s
    σ := norm u d0                                       -- σ = ‖W v‖ (‖v‖ = 1)
    let mut w : Array Float := Array.replicate d1 0.0    -- w = Wᵀ u
    for cc in [0:d1] do
      let mut s := 0.0
      for r in [0:d0] do s := s + F32.read W (r*d1+cc).toUSize * u[r]!
      w := w.set! cc s
    v := normalize w d1
  pure σ

/-- **Spectral-norm projection** (projected SGD toward the spectral ball): rescale every weight
    whose L2 norm exceeds `c` down to `c`, leaving biases untouched. Conv `[o,i,kh,kw]`: cap the
    **same** tap-sum bound the certificate uses (`convLip`) by scaling the whole kernel, so for
    the convs the projection and the certificate control the identical quantity. Dense `[d0,d1]`:
    cap the matrix-free power-iteration estimate of `‖W‖₂` (`specNormMV`) — the certificate's
    Schatten-8 bound costs a full Gram, too much to recompute every few steps — so a projected
    dense layer's certified `Lᵢ` (`denseLip`) sits somewhat above `c`. This is the lever that
    turns the (vacuous) product certificate non-vacuous. `F32.scaleShift` does the rescale. -/
private def projectSpectral (theta : ByteArray) (specs : Array (Array Nat × Nat)) (c : Float)
    : IO ByteArray := do
  let mut parts : Array ByteArray := #[]
  let mut off := 0
  for spec in specs do
    let dims := spec.1
    let len := dims.foldl (·*·) 1
    let slice := theta.extract (off*4) ((off+len)*4)
    let slice' ← if dims.size == 2 then do
        let σ := specNormMV slice dims[0]! dims[1]!
        if σ > c then F32.scaleShift slice (c/σ) 0.0 else pure slice
      else if dims.size == 4 then do
        let s := (convLip slice dims[0]! dims[1]! dims[2]! dims[3]!).bound
        if s > c then F32.scaleShift slice (c/s) 0.0 else pure slice
      else pure slice
    parts := parts.push slice'
    off := off + len
  return F32.concat parts

/-- **PGD attack on the verified MNIST MLP.** Trains the 784→512→512→10 ReLU MLP on the
    proof-rendered SGD step, then runs PGD with `genMlpPgdStep`, a hand-typed
    StableHLO kernel that follows the formula of `Proofs.mlpInputGrad` (no theorem ties the
    kernel's text). The Lipschitz certificate is the product of the three layers' spectral
    norms, which is where the bound, and so the certificate, goes loose. -/
def VerifiedNet.attackPgdMlp (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let hN := 512
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → GPU)"
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut theta := F32.concat parts
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for _ in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      theta ← net.sgdStep tsSess tsFn xb theta yb bs
  -- split θ (func-arg order: W0 b0 W1 b1 W2 b2)
  let W0 := theta.extract 0 (d0*hN*4)
  let W1 := theta.extract ((d0*hN + hN)*4) ((d0*hN + hN + hN*hN)*4)
  let W2 := theta.extract ((d0*hN + hN + hN*hN + hN)*4) ((d0*hN + hN + hN*hN + hN + hN*d1)*4)
  let mut clean := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
        clean := clean + 1
  IO.println s!"clean test acc = {clean}/{nEval} = {clean.toFloat/nEval.toFloat*100.0}%"
  let K := 40
  let pgdShapes := packShapes #[#[d0,hN], #[hN], #[hN,hN], #[hN], #[hN,d1], #[d1], #[bs,d1], #[bs,d0]]
  let runSweep := fun (linf : Bool) (epsList : List Float) => do
    for eps in epsList do
      let alpha := 2.5 * eps / K.toFloat
      IO.FS.writeFile ".lake/build/mlp_pgd_step.mlir" (genMlpPgdStep bs d0 hN d1 eps alpha linf)
      let pgdSess ← mkSession ".lake/build/mlp_pgd_step.mlir"
      let mut correct := 0
      for bi in [0:nbt] do
        let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
        let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
        let pgdParams := F32.concat #[theta, oh, x0]
        let mut x := x0
        for _ in [0:K] do
          x ← LowererSession.forwardF32 pgdSess "m.mlp_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
        for j in [0:min bs (nEval - bi * bs)] do
          if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
            correct := correct + 1
      let lbl := if linf then "L∞" else "L2"
      IO.println s!"{lbl} PGD eps={eps}: adv acc = {correct.toFloat/nEval.toFloat*100.0}%"
  runSweep true [0.1, 0.2, 0.3]
  -- certificate: product of the three layers' Schatten-8 bounds (ReLU is 1-Lipschitz)
  let L0 := denseLip W0 d0 hN
  let L1 := denseLip W1 hN hN
  let L2 := denseLip W2 hN d1
  let L := L0.bound * L1.bound * L2.bound
  IO.println s!"\nlayer bounds ‖W₀‖≤{L0.bound}, ‖W₁‖≤{L1.bound}, ‖W₂‖≤{L2.bound}  →  global L = {L}  (PRODUCT over 3 layers — loose)"
  IO.println s!"  (power-iteration estimates {L0.est}, {L1.est}, {L2.est}  →  product {L0.est * L1.est * L2.est}; not a bound)"
  let tot := nEval.toFloat
  let mut cert05 := 0
  let mut cert10 := 0
  let mut cert15 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      let mut top := -1.0e30
      let mut sec := -1.0e30
      let mut topi := 0
      for c in [0:d1] do
        let v := F32.read logits (j * d1 + c).toUSize
        if v > top then
          sec := top
          top := v
          topi := c
        else if v > sec then
          sec := v
      if topi == F32.readLabel evalLbl (bi * bs + j) then
        let r := (top - sec) / (1.4142135623730951 * L)
        if r ≥ 0.5 then cert05 := cert05 + 1
        if r ≥ 1.0 then cert10 := cert10 + 1
        if r ≥ 1.5 then cert15 := cert15 + 1
  IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  runSweep false [0.5, 1.0, 1.5]
  IO.println "done (MLP PGD: input gradient from the hand-typed PgdGen kernel, the mlpInputGrad formula)."

/-- **Spectral-norm-constrained training of the verified MNIST MLP.** Trains the 784→512→512→10 net with **projected SGD onto the spectral ball**
    — after every `projEvery` proof-rendered steps (and once at the end) each weight `Wᵢ` is
    rescaled so its power-iteration `‖Wᵢ‖₂` estimate is `≤ c` (`projectSpectral`) — then runs the
    *same* `cert ≤ TRUE ≤ PGD` sandwich. Sweeps a few caps `c` (plus an unconstrained baseline) so
    the table shows the trade: shrinking `c` pulls the certified `L = ∏ᵢ Lᵢ` (the Schatten-8
    bounds, each somewhat above `c`) down, turning the **vacuous** product certificate
    **non-vacuous** — at the cost of clean accuracy. The empirical face of
    `lipschitz_margin_certified_radius` ([`LeanMlir/Proofs/Certificates/LipschitzCert/Basic.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Certificates/LipschitzCert/Basic.lean)): smaller `L` ⇒ larger
    certified radius `m/(√2·L)`. The training gradient comes from the proof-rendered train step;
    the projection is host-side weight rescaling, and the PGD kernel is the hand-typed
    `genMlpPgdStep`. -/
def VerifiedNet.attackPgdSpectralMlp (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let hN := 512
  let d1 := net.nClasses
  let projEvery := 20            -- lazy projection: every 20 verified steps (+ once at the end)
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → GPU)"
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let K := 40
  let pgdShapes := packShapes #[#[d0,hN], #[hN], #[hN,hN], #[hN], #[hN,d1], #[d1], #[bs,d1], #[bs,d0]]
  -- run one PGD eps point, returning adversarial accuracy (%) on the verified net
  let pgdAcc := fun (theta : ByteArray) (linf : Bool) (eps : Float) => do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/mlp_pgd_step.mlir" (genMlpPgdStep bs d0 hN d1 eps alpha linf)
    let pgdSess ← mkSession ".lake/build/mlp_pgd_step.mlir"
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[theta, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess "m.mlp_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    pure (correct.toFloat / nEval.toFloat * 100.0)
  let tot := nEval.toFloat
  let mut rows : Array String := #[]
  for cap in caps do
    let capStr := if cap ≥ 1.0e8 then "∞ (none)" else toString cap
    IO.println s!"\n── cap c = {capStr} ──"
    -- fresh He-init (same seeds ⇒ fair comparison across caps)
    let mut parts : Array ByteArray := #[]
    let mut seed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam seed spec.1 spec.2)
      seed := seed + 1
    let mut theta := F32.concat parts
    let mut step := 0
    for _ in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xb := F32.sliceImages trainImg (bi * bs) bs d0
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        theta ← net.sgdStep tsSess tsFn xb theta yb bs
        step := step + 1
        if cap < 1.0e8 && step % projEvery == 0 then
          theta ← projectSpectral theta net.specs cap
    if cap < 1.0e8 then theta ← projectSpectral theta net.specs cap   -- enforce the cap on the final θ
    -- split θ for the certificate
    let W0 := theta.extract 0 (d0*hN*4)
    let W1 := theta.extract ((d0*hN + hN)*4) ((d0*hN + hN + hN*hN)*4)
    let W2 := theta.extract ((d0*hN + hN + hN*hN + hN)*4) ((d0*hN + hN + hN*hN + hN + hN*d1)*4)
    let L0 := denseLip W0 d0 hN
    let L1 := denseLip W1 hN hN
    let L2 := denseLip W2 hN d1
    let L := L0.bound * L1.bound * L2.bound
    -- clean accuracy + certified-robust accuracy at L2 {0.25, 0.5, 1.0}
    let mut clean := 0
    let mut c025 := 0
    let mut c05 := 0
    let mut c10 := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for cidx in [0:d1] do
          let v := F32.read logits (j * d1 + cidx).toUSize
          if v > top then sec := top; top := v; topi := cidx
          else if v > sec then sec := v
        if topi == F32.readLabel evalLbl (bi * bs + j) then
          clean := clean + 1
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.25 then c025 := c025 + 1
          if r ≥ 0.5 then c05 := c05 + 1
          if r ≥ 1.0 then c10 := c10 + 1
    let cleanPct := clean.toFloat/tot*100.0
    IO.println s!"  ‖W₀‖≤{L0.bound}  ‖W₁‖≤{L1.bound}  ‖W₂‖≤{L2.bound}  →  L = {L}   (estimates {L0.est} / {L1.est} / {L2.est})"
    IO.println s!"  clean = {cleanPct}%   cert@L2 0.25/0.5/1.0 = {c025.toFloat/tot*100.0}% / {c05.toFloat/tot*100.0}% / {c10.toFloat/tot*100.0}%"
    let pinf ← pgdAcc theta true 0.1
    let pl2 ← pgdAcc theta false 0.5
    IO.println s!"  L∞ PGD ε=0.1 = {pinf}%   L2 PGD ε=0.5 = {pl2}%"
    (← IO.getStdout).flush
    rows := rows.push s!"  {capStr}\t{cleanPct}\t{L}\t{c05.toFloat/tot*100.0}\t{pl2}\t{pinf}"
  IO.println "\n══ spectral-norm training: the cert ≤ TRUE ≤ PGD trade ══"
  IO.println "  cap c\tclean%\tglobal L\tcert@L2 0.5\tL2 PGD 0.5\tL∞ PGD 0.1"
  for row in rows do IO.println row
  IO.println "\ndone (spectral-norm-constrained training: smaller c ⇒ smaller L ⇒ the product cert"
  IO.println "      goes non-vacuous, at the cost of clean accuracy — the gap-shrinking lever)."

/-- The conv-aware certificate product over a packed parameter list: `convLip` for each
    `[o,i,kh,kw]` kernel, `denseLip` for each `[d0,d1]` weight, biases skipped. Returns the
    product of the bounds (the certificate's `L`), the product of the estimates, and a per-layer
    `bound (estimate)` line. -/
private def certProduct (theta : ByteArray) (specs : Array (Array Nat × Nat)) :
    Float × Float × String := Id.run do
  let mut L := 1.0
  let mut Lest := 1.0
  let mut msg := ""
  let mut off := 0
  for spec in specs do
    let dims := spec.1
    let len := dims.foldl (·*·) 1
    let wslice := theta.extract (off*4) ((off+len)*4)
    if dims.size == 4 then
      let p := convLip wslice dims[0]! dims[1]! dims[2]! dims[3]!
      L := L * p.bound; Lest := Lest * p.est
      msg := msg ++ s!"conv{dims[1]!}→{dims[0]!} {p.bound} ({p.est})  "
    else if dims.size == 2 then
      let p := denseLip wslice dims[0]! dims[1]!
      L := L * p.bound; Lest := Lest * p.est
      msg := msg ++ s!"dense{dims[0]!}→{dims[1]!} {p.bound} ({p.est})  "
    off := off + len
  pure (L, Lest, msg)

/-- **Generic conv-net PGD attack.** Trains any packed conv net on its proof-rendered SGD step,
    then runs PGD with `genKernel`: a hand-typed StableHLO kernel that computes the
    input gradient `dx` (conv input-VJPs and maxpool `select_and_scatter` backs, following the
    backward ops of the net's `<slug>_train_step.mlir`); no theorem ties the kernel's text.
    Certificate = the conv-aware **product** of per-layer upper bounds (`convLip` for convs ×
    `denseLip` for denses; ReLU and the disjoint 2×2 max-pools are 1-Lipschitz) —
    astronomically loose, the depth-cliff. `genKernel` and `net.slug` select the architecture
    (`genCnnPgdStep`/MNIST-CNN, `genCifarPgdStep`/CIFAR-CNN). -/
def VerifiedNet.attackPgdConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (genKernel : Nat → Float → Float → Bool → String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → GPU)"
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut theta := F32.concat parts
  -- Best-checkpoint training: eval each epoch and keep the highest-accuracy θ. Plain SGD on the
  -- deeper nets (CIFAR) can diverge late; attacking the best checkpoint keeps the demo robust
  -- (and the cert finite). Monotone nets (MNIST CNN) → best = final, so numbers are unchanged.
  let mut bestTheta := theta
  let mut bestAcc := -1.0
  let evalAcc := fun (th : ByteArray) => do
    let mut c := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn th shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          c := c + 1
    pure (c.toFloat / nEval.toFloat * 100.0)
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      theta ← net.sgdStep tsSess tsFn xb theta yb bs
    let acc ← evalAcc theta
    if acc > bestAcc then bestAcc := acc; bestTheta := theta
    IO.println s!"    epoch {ep + 1}/{cfg.epochs}: acc = {acc}%"
    (← IO.getStdout).flush
  theta := bestTheta                       -- attack the best checkpoint
  IO.println s!"clean test acc (best epoch) = {bestAcc}%"
  let K := 40
  let pgdShapes := packShapes (net.paramShapes ++ #[#[bs, d1], #[bs, d0]])
  let runSweep := fun (linf : Bool) (epsList : List Float) => do
    for eps in epsList do
      let alpha := 2.5 * eps / K.toFloat
      IO.FS.writeFile s!".lake/build/{net.slug}_pgd_step.mlir" (genKernel bs eps alpha linf)
      let pgdSess ← mkSession s!".lake/build/{net.slug}_pgd_step.mlir"
      let mut correct := 0
      for bi in [0:nbt] do
        let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
        let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
        let pgdParams := F32.concat #[theta, oh, x0]
        let mut x := x0
        for _ in [0:K] do
          x ← LowererSession.forwardF32 pgdSess s!"m.{net.slug}_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
        for j in [0:min bs (nEval - bi * bs)] do
          if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
            correct := correct + 1
      let lbl := if linf then "L∞" else "L2"
      IO.println s!"{lbl} PGD eps={eps}: adv acc = {correct.toFloat/nEval.toFloat*100.0}%"
      (← IO.getStdout).flush
  runSweep true [0.1, 0.2, 0.3]
  -- ── certificate: conv-aware PRODUCT of per-layer bounds (ReLU/maxpool are 1-Lipschitz) ──
  let (L, Lest, msg) := certProduct theta net.specs
  IO.println s!"\nlayer bounds (estimate): {msg}"
  IO.println s!"  →  global L = {L}  (PRODUCT over conv+dense layers — astronomically loose; estimates' product {Lest}, not a bound)"
  let tot := nEval.toFloat
  let mut cert05 := 0
  let mut cert10 := 0
  let mut cert15 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      let mut top := -1.0e30
      let mut sec := -1.0e30
      let mut topi := 0
      for c in [0:d1] do
        let v := F32.read logits (j * d1 + c).toUSize
        if v > top then
          sec := top
          top := v
          topi := c
        else if v > sec then
          sec := v
      if topi == F32.readLabel evalLbl (bi * bs + j) then
        let r := (top - sec) / (1.4142135623730951 * L)
        if r ≥ 0.5 then cert05 := cert05 + 1
        if r ≥ 1.0 then cert10 := cert10 + 1
        if r ≥ 1.5 then cert15 := cert15 + 1
  IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  runSweep false [0.5, 1.0, 1.5]
  IO.println s!"done ({net.name} PGD: input gradient from the hand-typed PgdGen kernel, the conv/maxpool input-VJP formula)."

/-- PGD attack on the verified MNIST CNN (the first conv rung). -/
def VerifiedNet.attackPgdCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCnnPgdStep

/-- PGD attack on the verified CIFAR-10 CNN (the deeper conv rung: 4 conv + 2 pool + 3 dense). -/
def VerifiedNet.attackPgdCifar (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCifarPgdStep

/-- **Spectral-norm-constrained training of the verified MNIST CNN.** The CNN sibling of `attackPgdSpectralMlp`:
    projected SGD toward the spectral ball — every `projEvery` proof-rendered steps (and once at
    the end) `projectSpectral` caps the dense `‖Wᵢ‖₂` estimate and the conv tap-sum bound at `c` —
    then the `cert ≤ TRUE ≤ PGD` sandwich (PGD via `genKernel`, cert = the conv-aware product).
    Harder than the MLP: it's a `k`-layer product and the conv tap-sum is a *loose* bound, so
    projection over-penalizes the convs — the cert needs a tighter `c` (and pays more clean accuracy)
    than the MLP did, and certifies only at *smaller* radii. The honest "depth + loose conv-norm ⇒
    certifying the conv net is harder." Generic over `genKernel`/`net.slug` (MNIST-CNN, CIFAR-CNN). -/
def VerifiedNet.attackPgdSpectralConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) (genKernel : Nat → Float → Float → Bool → String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  let projEvery := 20
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → GPU)"
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let K := 40
  let pgdShapes := packShapes (net.paramShapes ++ #[#[bs, d1], #[bs, d0]])
  let pgdAcc := fun (theta : ByteArray) (linf : Bool) (eps : Float) => do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile s!".lake/build/{net.slug}_pgd_step.mlir" (genKernel bs eps alpha linf)
    let pgdSess ← mkSession s!".lake/build/{net.slug}_pgd_step.mlir"
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[theta, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess s!"m.{net.slug}_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    pure (correct.toFloat / nEval.toFloat * 100.0)
  let tot := nEval.toFloat
  let mut rows : Array String := #[]
  for cap in caps do
    let capStr := if cap ≥ 1.0e8 then "∞ (none)" else toString cap
    IO.println s!"\n── cap c = {capStr} ──"
    let mut parts : Array ByteArray := #[]
    let mut seed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam seed spec.1 spec.2)
      seed := seed + 1
    let mut theta := F32.concat parts
    let mut bestTheta := theta
    let mut bestAcc := -1.0
    let mut step := 0
    for _ in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xb := F32.sliceImages trainImg (bi * bs) bs d0
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        theta ← net.sgdStep tsSess tsFn xb theta yb bs
        step := step + 1
        if cap < 1.0e8 && step % projEvery == 0 then
          theta ← projectSpectral theta net.specs cap
      -- best-checkpoint (the baseline ∞ cap can diverge late; constrained caps stay bounded)
      let mut c := 0
      for bi in [0:nbt] do
        let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
        let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
        for j in [0:min bs (nEval - bi * bs)] do
          if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
            c := c + 1
      let acc := c.toFloat / nEval.toFloat * 100.0
      if acc > bestAcc then bestAcc := acc; bestTheta := theta
    theta := bestTheta
    if cap < 1.0e8 then theta ← projectSpectral theta net.specs cap
    let (L, _, msg) := certProduct theta net.specs
    let mut clean := 0
    let mut cR1 := 0      -- certified @ L2 0.1
    let mut cR2 := 0      -- certified @ L2 0.25  (the CNN's visible band — it certifies at small radii)
    let mut cR3 := 0      -- certified @ L2 0.5
    for bi in [0:nbt] do
      let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let logits ← LowererSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for cidx in [0:d1] do
          let v := F32.read logits (j * d1 + cidx).toUSize
          if v > top then sec := top; top := v; topi := cidx
          else if v > sec then sec := v
        if topi == F32.readLabel evalLbl (bi * bs + j) then
          clean := clean + 1
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.1 then cR1 := cR1 + 1
          if r ≥ 0.25 then cR2 := cR2 + 1
          if r ≥ 0.5 then cR3 := cR3 + 1
    let cleanPct := clean.toFloat/tot*100.0
    IO.println s!"  {msg} →  L = {L}"
    IO.println s!"  clean = {cleanPct}%   cert@L2 0.1/0.25/0.5 = {cR1.toFloat/tot*100.0}% / {cR2.toFloat/tot*100.0}% / {cR3.toFloat/tot*100.0}%"
    let pinf ← pgdAcc theta true 0.1
    let pl2 ← pgdAcc theta false 0.5
    IO.println s!"  L∞ PGD ε=0.1 = {pinf}%   L2 PGD ε=0.5 = {pl2}%"
    (← IO.getStdout).flush
    rows := rows.push s!"  {capStr}\t{cleanPct}\t{L}\t{cR2.toFloat/tot*100.0}\t{pl2}\t{pinf}"
  IO.println s!"\n══ spectral-norm training ({net.name}): the cert ≤ TRUE ≤ PGD trade ══"
  IO.println "  cap c\tclean%\tglobal L\tcert@L2 0.25\tL2 PGD 0.5\tL∞ PGD 0.1"
  for row in rows do IO.println row
  IO.println "\ndone (spectral-norm-constrained conv training: the k-layer product + loose conv tap-sum"
  IO.println "      make certifying the conv net harder than the MLP — tighter c, more clean cost)."

/-- Spectral-norm-constrained training of the verified MNIST CNN. -/
def VerifiedNet.attackPgdSpectralCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  net.attackPgdSpectralConvNet cfg dataDir caps genCnnPgdStep

/-- Spectral-norm-constrained training of the verified CIFAR-10 CNN (7-layer product). -/
def VerifiedNet.attackPgdSpectralCifar (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  net.attackPgdSpectralConvNet cfg dataDir caps genCifarPgdStep

/-- **PGD adversarial attack** on the verified linear classifier. Trains via the proof-rendered
    train step, then runs PGD: each PGD step's input gradient is computed on the GPU
    by `genLinearPgdStep`, a hand-typed StableHLO kernel that follows the formula
    `dx = (softmax−onehot)·Wᵀ` (no theorem ties the kernel's text). Reports clean vs L∞-PGD adversarial accuracy over an eps sweep. -/
def VerifiedNet.attackPgd (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → GPU)"
  let tsSess  ← mkSession s!"{net.mlirDir}/{net.slug}_train_step.mlir"
  let fwdSess ← mkSession s!"{net.mlirDir}/{net.slug}_fwd.mlir"
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut W0 ← F32.const (d0 * d1).toUSize 0.0
  let mut b0 ← F32.const d1.toUSize 0.0
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for _ in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← LowererSession.linearTrainStepV tsSess tsFn xb W0 b0 yb bs.toUSize d0.toUSize d1.toUSize
      W0 := out.extract 0 (d0 * d1 * 4)
      b0 := out.extract (d0 * d1 * 4) ((d0 * d1 + d1) * 4)
  -- clean accuracy
  let mut clean := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
        clean := clean + 1
  IO.println s!"clean test acc = {clean}/{nEval} = {clean.toFloat/nEval.toFloat*100.0}%"
  -- L∞ PGD sweep
  let K := 40
  for eps in ([0.1, 0.2, 0.3] : List Float) do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/linear_pgd_step.mlir" (genLinearPgdStep bs d0 d1 eps alpha true)
    let pgdSess ← mkSession ".lake/build/linear_pgd_step.mlir"
    let pgdShapes := packShapes #[#[d0, d1], #[d1], #[bs, d1], #[bs, d0]]
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[W0, b0, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess "m.linear_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    IO.println s!"L∞ PGD eps={eps}: adv acc = {correct}/{nEval} = {correct.toFloat/nEval.toFloat*100.0}%"
  -- ── L2 sandwich: Lipschitz certificate (lower bound) vs L2 PGD (upper bound) ──
  let Lp := denseLip W0 d0 d1
  let L := Lp.bound
  IO.println s!"\nglobal Lipschitz bound ‖W‖₂ ≤ {L}  (Schatten-8; power-iteration estimate of the exact ‖W‖₂ = {Lp.est})"
  let tot := nEval.toFloat
  let mut cert05 := 0
  let mut cert10 := 0
  let mut cert15 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
    let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:min bs (nEval - bi * bs)] do
      let mut top := -1.0e30
      let mut sec := -1.0e30
      let mut topi := 0
      for c in [0:d1] do
        let v := F32.read logits (j * d1 + c).toUSize
        if v > top then
          sec := top
          top := v
          topi := c
        else if v > sec then
          sec := v
      if topi == F32.readLabel evalLbl (bi * bs + j) then
        let r := (top - sec) / (1.4142135623730951 * L)    -- certified L2 radius m(x)/(√2 L)
        if r ≥ 0.5 then cert05 := cert05 + 1
        if r ≥ 1.0 then cert10 := cert10 + 1
        if r ≥ 1.5 then cert15 := cert15 + 1
  IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  for eps in ([0.5, 1.0, 1.5] : List Float) do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/linear_pgd_step.mlir" (genLinearPgdStep bs d0 d1 eps alpha false)
    let pgdSess ← mkSession ".lake/build/linear_pgd_step.mlir"
    let pgdShapes := packShapes #[#[d0, d1], #[d1], #[bs, d1], #[bs, d0]]
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImagesPad evalImg (bi * bs) bs d0 nEval
      let oh ← oneHotBatchPad evalLbl (bi * bs) bs d1 nEval
      let pgdParams := F32.concat #[W0, b0, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← LowererSession.forwardF32 pgdSess "m.linear_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← LowererSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes x xShape bs.toUSize d1.toUSize
      for j in [0:min bs (nEval - bi * bs)] do
        if (F32.argmaxN logits (j * d1).toUSize d1.toUSize).toNat == F32.readLabel evalLbl (bi * bs + j) then
          correct := correct + 1
    IO.println s!"L2 PGD eps={eps}: adv acc = {correct.toFloat/tot*100.0}%  (sandwich: cert ≤ true ≤ this)"
  IO.println "done (PGD: input gradient from the hand-typed PgdGen kernel, the input-VJP formula)."

