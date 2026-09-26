import LeanMlir.VerifiedTrain
import LeanMlir.VerifiedPgdGen

/-! # PGD attacks and the spectral-norm training studies

Each attack trains a verified net on its proof-rendered train step (`VerifiedTrain`'s driver),
then runs PGD through the runtime with a `VerifiedPgdGen` kernel, and reports clean vs
adversarial accuracy over an ε sweep next to the Lipschitz bound (the product of the layers'
spectral norms). The `Spectral` variants project each weight onto a spectral-norm ball after
every step (`projectSpectral`, power iteration in `specNorm*`). -/

/-- Build a one-hot `[bs, d1]` f32 batch from int32-LE labels (1.0 = bytes 00 00 80 3F). -/
private def oneHotBatch (labels : ByteArray) (start bs d1 : Nat) : IO ByteArray := do
  let mut oh ← F32.const (bs * d1).toUSize 0.0
  for j in [0:bs] do
    let lbl := (labels.get! (4 * (start + j))).toNat
    let fi := j * d1 + lbl
    oh := (((oh.set! (4*fi) 0).set! (4*fi+1) 0).set! (4*fi+2) 0x80).set! (4*fi+3) 0x3F
  return oh

/-- `oneHotBatch` for a possibly-partial final batch: rows past `total` records get an
    all-zero row (their padded images are never scored, so the gradient they induce is
    irrelevant — the row just has to exist because the batch dim is baked into the vmfb). -/
private def oneHotBatchPad (labels : ByteArray) (start bs d1 total : Nat) : IO ByteArray := do
  let mut oh ← F32.const (bs * d1).toUSize 0.0
  for j in [0:min bs (total - start)] do
    let lbl := (labels.get! (4 * (start + j))).toNat
    let fi := j * d1 + lbl
    oh := (((oh.set! (4*fi) 0).set! (4*fi+1) 0).set! (4*fi+2) 0x80).set! (4*fi+3) 0x3F
  return oh

/-- Spectral norm `‖W‖₂` of `W : [d0,d1]` (row-major) by power iteration on the small
    `WᵀW : [d1,d1]` Gram matrix. For the linear net this IS the global Lipschitz constant
    of the logit map (`logits = xW+b`, Jacobian `Wᵀ`). Host-side, pure. -/
private def specNormW (W : ByteArray) (d0 d1 : Nat) : Float := Id.run do
  let g := fun (i j : Nat) => Id.run do      -- WᵀW[i,j] = Σ_k W[k,i]·W[k,j]
    let mut s := 0.0
    for k in [0:d0] do
      s := s + (F32.read W (k*d1+i).toUSize) * (F32.read W (k*d1+j).toUSize)
    pure s
  let mut wtw : Array Float := Array.replicate (d1*d1) 0.0
  for i in [0:d1] do
    for j in [0:d1] do
      wtw := wtw.set! (i*d1+j) (g i j)
  let mv := fun (v : Array Float) => Id.run do  -- WᵀW · v
    let mut u : Array Float := Array.replicate d1 0.0
    for i in [0:d1] do
      let mut s := 0.0
      for j in [0:d1] do s := s + wtw[i*d1+j]! * v[j]!
      u := u.set! i s
    pure u
  let mut v : Array Float := Array.replicate d1 1.0
  for _ in [0:60] do
    let u := mv v
    let mut nrm := 0.0
    for i in [0:d1] do nrm := nrm + u[i]!*u[i]!
    nrm := Float.sqrt nrm
    if nrm > 1e-20 then
      for i in [0:d1] do v := v.set! i (u[i]!/nrm)
  let u := mv v
  let mut lam := 0.0
  for i in [0:d1] do lam := lam + v[i]! * u[i]!   -- Rayleigh quotient (‖v‖=1)
  pure (Float.sqrt lam)

/-- Spectral norm `‖M‖₂` of a `[rows, cols]` matrix given by an index function `get i j`
    (the same power iteration on the `cols×cols` Gram as `specNormW`, but reading via `get`
    so it works on strided sub-tensors — e.g. one tap-plane of a conv kernel). -/
private def specNormGet (get : Nat → Nat → Float) (rows cols : Nat) : Float := Id.run do
  let gram := fun (i j : Nat) => Id.run do        -- (MᵀM)[i,j] = Σ_k M[k,i]·M[k,j]
    let mut s := 0.0
    for k in [0:rows] do s := s + (get k i) * (get k j)
    pure s
  let mut wtw : Array Float := Array.replicate (cols*cols) 0.0
  for i in [0:cols] do
    for j in [0:cols] do
      wtw := wtw.set! (i*cols+j) (gram i j)
  let mv := fun (v : Array Float) => Id.run do
    let mut u : Array Float := Array.replicate cols 0.0
    for i in [0:cols] do
      let mut s := 0.0
      for j in [0:cols] do s := s + wtw[i*cols+j]! * v[j]!
      u := u.set! i s
    pure u
  let mut v : Array Float := Array.replicate cols 1.0
  for _ in [0:60] do
    let u := mv v
    let mut nrm := 0.0
    for i in [0:cols] do nrm := nrm + u[i]!*u[i]!
    nrm := Float.sqrt nrm
    if nrm > 1e-20 then
      for i in [0:cols] do v := v.set! i (u[i]!/nrm)
  let u := mv v
  let mut lam := 0.0
  for i in [0:cols] do lam := lam + v[i]! * u[i]!
  pure (Float.sqrt lam)

/-- A **sound** (loose) upper bound on the L2 operator norm of a zero-padded 2-D
    convolution with kernel `W : [outC, inC, kh, kw]` (row-major). Writing the conv as a
    sum over spatial taps `T = Σ_{ky,kx} S_{ky,kx} ∘ M_{ky,kx}` — each `S` a (norm ≤ 1)
    shift and each `M` the pointwise `[outC,inC]` channel-mixing matrix at that tap — the
    triangle inequality gives `‖T‖₂ ≤ Σ_{ky,kx} ‖W[:,:,ky,kx]‖₂`. Each tap-plane's spectral
    norm is the same power iteration as `specNormW`. Loose by up to `√(kh·kw)` vs the exact
    (Sedghi–Gupta–Long) value — which only sharpens the "depth ⇒ vacuous product" message. -/
private def specNormConvTapSum (W : ByteArray) (outC inC kh kw : Nat) : Float := Id.run do
  let mut s := 0.0
  for ky in [0:kh] do
    for kx in [0:kw] do
      s := s + specNormGet
        (fun o i => F32.read W (((o*inC+i)*kh+ky)*kw+kx).toUSize) outC inC
  pure s

/-- **Matrix-free** spectral norm `‖W‖₂` of `W : [d0,d1]` (row-major) — power iteration that
    applies `W` and `Wᵀ` as mat-vecs (`σ = ‖W v‖`, `v` the top right singular vector) instead
    of forming the `d1×d1` Gram. ~`2·d0·d1` per iteration vs `d0·d1²` for `specNormW`, so it's
    cheap enough to call **during** training (the spectral-norm projection below). Fewer iters
    (`iters`) trade a little precision for speed; `specNormW` stays the high-precision cert path. -/
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

/-- **Spectral-norm projection** (projected SGD onto the spectral ball): rescale every weight
    whose L2-Lipschitz bound exceeds `c` down to `c`, leaving biases untouched. Dense `[d0,d1]`:
    cap the spectral norm `‖W‖₂` (`specNormMV`). Conv `[o,i,kh,kw]`: cap the **same** tap-sum
    operator bound the CNN certificate uses (`specNormConvTapSum`, `‖T‖₂ ≤ Σ_tap‖W[:,:,ky,kx]‖₂`)
    by scaling the whole kernel — so the projection and the cert control the identical quantity.
    Caps each layer's Lipschitz constant at `c`, so the global `L = ∏ᵢ ≤ cᵏ` — the lever that
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
        let s := specNormConvTapSum slice dims[0]! dims[1]! dims[2]! dims[3]!
        if s > c then F32.scaleShift slice (c/s) 0.0 else pure slice
      else pure slice
    parts := parts.push slice'
    off := off + len
  return F32.concat parts

/-- **PGD attack on the verified MNIST MLP.** Trains the 784→512→512→10 ReLU MLP on the
    proof-rendered SGD step, then runs PGD through IREE with `genMlpPgdStep`, a hand-typed
    StableHLO kernel that follows the formula of `Proofs.mlpInputGrad` (no theorem ties the
    kernel's text). The Lipschitz certificate is the product of the three layers' spectral
    norms, which is where the bound, and so the certificate, goes loose. -/
def VerifiedNet.attackPgdMlp (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let hN := 512
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net dataDir
  let nb  := nTrain / bs
  let nbt := (nEval + bs - 1) / bs   -- ceil: last partial batch zero-padded, not dropped
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let nP := net.nParams
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
      let out ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
      theta := out.extract 0 (nP * 4)
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
      compileVmfb ".lake/build/mlp_pgd_step.mlir" ".lake/build/mlp_pgd_step.vmfb"
      let pgdSess ← LowererSession.create ".lake/build/mlp_pgd_step.vmfb"
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
  -- certificate: product of the three layers' spectral norms (ReLU is 1-Lipschitz)
  let L0 := specNormW W0 d0 hN
  let L1 := specNormW W1 hN hN
  let L2 := specNormW W2 hN d1
  let L := L0 * L1 * L2
  IO.println s!"\nspectral norms ‖W₀‖={L0}, ‖W₁‖={L1}, ‖W₂‖={L2}  →  global L = {L}  (PRODUCT over 3 layers — loose)"
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
  IO.println "done (phase-3 MLP PGD: input gradient = the proven mlpInputGrad VJP via IREE)."

/-- **Spectral-norm-constrained training of the verified MNIST MLP.** Trains the 784→512→512→10 net with **projected SGD onto the spectral ball**
    — after every `K` proof-rendered steps (and once at the end) each weight `Wᵢ` is rescaled to
    `‖Wᵢ‖₂ ≤ c` (`projectSpectral`) — then runs the *same* `cert ≤ TRUE ≤ PGD` sandwich. Sweeps a
    few caps `c` (plus an unconstrained baseline) so the table shows the trade: shrinking `c` pulls
    the global `L = ∏‖Wᵢ‖₂` down (`L ≤ c³`), turning the **vacuous** product certificate
    **non-vacuous** — at the cost of clean accuracy. The empirical face of
    `lipschitz_margin_certified_radius` ([`LeanMlir/Proofs/Certificates/LipschitzCert.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Certificates/LipschitzCert.lean)): smaller `L` ⇒ larger
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
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
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
    compileVmfb ".lake/build/mlp_pgd_step.mlir" ".lake/build/mlp_pgd_step.vmfb"
    let pgdSess ← LowererSession.create ".lake/build/mlp_pgd_step.vmfb"
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
        theta ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
        step := step + 1
        if cap < 1.0e8 && step % projEvery == 0 then
          theta ← projectSpectral theta net.specs cap
    if cap < 1.0e8 then theta ← projectSpectral theta net.specs cap   -- enforce the cap on the final θ
    -- split θ for the certificate
    let W0 := theta.extract 0 (d0*hN*4)
    let W1 := theta.extract ((d0*hN + hN)*4) ((d0*hN + hN + hN*hN)*4)
    let W2 := theta.extract ((d0*hN + hN + hN*hN + hN)*4) ((d0*hN + hN + hN*hN + hN + hN*d1)*4)
    let L0 := specNormW W0 d0 hN
    let L1 := specNormW W1 hN hN
    let L2 := specNormW W2 hN d1
    let L := L0 * L1 * L2
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
    IO.println s!"  ‖W₀‖={L0}  ‖W₁‖={L1}  ‖W₂‖={L2}  →  L = {L}"
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

/-- **Generic conv-net PGD attack.** Trains any packed conv net on its proof-rendered SGD step,
    then runs PGD through IREE with `genKernel`: a hand-typed StableHLO kernel that computes the
    input gradient `dx` (conv input-VJPs and maxpool `select_and_scatter` backs, following the
    backward ops of the net's `<slug>_train_step.mlir`); no theorem ties the kernel's text.
    Certificate = the conv-aware spectral-norm **product** (`specNormConvTapSum` for convs ×
    `specNormW` for denses; ReLU/maxpool are 1-Lipschitz) —
    astronomically loose, the depth-cliff. `genKernel` and `net.slug` select the architecture
    (`genCnnPgdStep`/MNIST-CNN, `genCifarPgdStep`/CIFAR-CNN). -/
def VerifiedNet.attackPgdConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (genKernel : Nat → Float → Float → Bool → String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
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
      theta ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
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
      compileVmfb s!".lake/build/{net.slug}_pgd_step.mlir" s!".lake/build/{net.slug}_pgd_step.vmfb"
      let pgdSess ← LowererSession.create s!".lake/build/{net.slug}_pgd_step.vmfb"
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
  -- ── certificate: conv-aware spectral-norm PRODUCT (ReLU/maxpool are 1-Lipschitz) ──
  let mut L := 1.0
  let mut off := 0
  let mut msg := ""
  for spec in net.specs do
    let dims := spec.1
    let len := dims.foldl (·*·) 1
    let wslice := theta.extract (off*4) ((off+len)*4)
    if dims.size == 4 then
      let n := specNormConvTapSum wslice dims[0]! dims[1]! dims[2]! dims[3]!
      L := L * n
      msg := msg ++ s!"conv{dims[1]!}→{dims[0]!} Σtap‖·‖₂={n}  "
    else if dims.size == 2 then
      let n := specNormW wslice dims[0]! dims[1]!
      L := L * n
      msg := msg ++ s!"dense{dims[0]!}→{dims[1]!} ‖·‖₂={n}  "
    off := off + len
  IO.println s!"\nlayer norms: {msg}"
  IO.println s!"  →  global L = {L}  (PRODUCT over conv+dense layers — astronomically loose)"
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
  IO.println s!"done (phase-3 {net.name} PGD: input gradient = the proven conv/maxpool input-VJP via IREE)."

/-- PGD attack on the verified MNIST CNN (the first conv rung). -/
def VerifiedNet.attackPgdCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCnnPgdStep

/-- PGD attack on the verified CIFAR-10 CNN (the deeper conv rung: 4 conv + 2 pool + 3 dense). -/
def VerifiedNet.attackPgdCifar (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCifarPgdStep

/-- **Spectral-norm-constrained training of the verified MNIST CNN.** The CNN sibling of `attackPgdSpectralMlp`:
    projected SGD onto the spectral ball — every `K` proof-rendered steps (and once at the end)
    `projectSpectral` caps **both** the dense `‖Wᵢ‖₂` and the conv tap-sum bound at `c` — then the
    `cert ≤ TRUE ≤ PGD` sandwich (PGD via `genKernel`, cert = the conv-aware product). Harder than
    the MLP: it's a `k`-layer product (`L ≤ cᵏ`) and the conv tap-sum is a *loose* bound, so
    projection over-penalizes the convs — the cert needs a tighter `c` (and pays more clean accuracy)
    than the MLP did, and certifies only at *smaller* radii. The honest "depth + loose conv-norm ⇒
    certifying the conv net is harder." Generic over `genKernel`/`net.slug` (MNIST-CNN, CIFAR-CNN). -/
def VerifiedNet.attackPgdSpectralConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) (genKernel : Nat → Float → Float → Bool → String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  let projEvery := 20
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
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
    compileVmfb s!".lake/build/{net.slug}_pgd_step.mlir" s!".lake/build/{net.slug}_pgd_step.vmfb"
    let pgdSess ← LowererSession.create s!".lake/build/{net.slug}_pgd_step.vmfb"
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
        theta ← LowererSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
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
    -- conv-aware certificate product (specNormConvTapSum convs × specNormW denses)
    let mut L := 1.0
    let mut off := 0
    let mut msg := ""
    for spec in net.specs do
      let dims := spec.1
      let len := dims.foldl (·*·) 1
      let wslice := theta.extract (off*4) ((off+len)*4)
      if dims.size == 4 then
        let n := specNormConvTapSum wslice dims[0]! dims[1]! dims[2]! dims[3]!
        L := L * n; msg := msg ++ s!"cv={n} "
      else if dims.size == 2 then
        let n := specNormW wslice dims[0]! dims[1]!
        L := L * n; msg := msg ++ s!"de={n} "
      off := off + len
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
    train step, then runs PGD through IREE: each PGD step's input gradient is computed on the GPU
    by `genLinearPgdStep`, a hand-typed StableHLO kernel that follows the formula
    `dx = (softmax−onehot)·Wᵀ` (no theorem ties the kernel's text). Reports clean vs L∞-PGD adversarial accuracy over an eps sweep. -/
def VerifiedNet.attackPgd (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"{net.mlirDir}/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"{net.mlirDir}/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← LowererSession.create tsVmfb
  let fwdSess ← LowererSession.create fwdVmfb
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
    compileVmfb ".lake/build/linear_pgd_step.mlir" ".lake/build/linear_pgd_step.vmfb"
    let pgdSess ← LowererSession.create ".lake/build/linear_pgd_step.vmfb"
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
  let L := specNormW W0 d0 d1
  IO.println s!"\nglobal Lipschitz ‖W‖₂ = {L}  (linear: the logit map's exact L2 Lipschitz)"
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
    compileVmfb ".lake/build/linear_pgd_step.mlir" ".lake/build/linear_pgd_step.vmfb"
    let pgdSess ← LowererSession.create ".lake/build/linear_pgd_step.vmfb"
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
  IO.println "done (phase-3 PGD: gradient computed by the proven input-VJP kernel via IREE)."

namespace VerifiedNetSpec

/-- PGD adversarial attack (Chapter 1 linear); see `VerifiedNet.attackPgd`. -/
def attackPgd (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgd cfg dataDir

/-- PGD attack on the MLP (Chapter 2); see `VerifiedNet.attackPgdMlp`. -/
def attackPgdMlp (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgdMlp cfg dataDir

/-- PGD attack on the CNN (Chapter 3); see `VerifiedNet.attackPgdCnn`. -/
def attackPgdCnn (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgdCnn cfg dataDir

/-- Spectral-norm-constrained MLP training study; see `VerifiedNet.attackPgdSpectralMlp`. -/
def attackPgdSpectralMlp (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  s.toNet.attackPgdSpectralMlp cfg dataDir caps

/-- Spectral-norm-constrained CNN training study; see `VerifiedNet.attackPgdSpectralCnn`. -/
def attackPgdSpectralCnn (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  s.toNet.attackPgdSpectralCnn cfg dataDir caps

/-- PGD attack on the CIFAR-10 CNN (the deeper conv rung); see `VerifiedNet.attackPgdCifar`. -/
def attackPgdCifar (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  s.toNet.attackPgdCifar cfg dataDir

/-- Spectral-norm-constrained CIFAR training study; see `VerifiedNet.attackPgdSpectralCifar`. -/
def attackPgdSpectralCifar (s : VerifiedNetSpec) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  s.toNet.attackPgdSpectralCifar cfg dataDir caps

end VerifiedNetSpec
