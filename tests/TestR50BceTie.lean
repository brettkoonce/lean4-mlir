import LeanMlir.VerifiedNets

/-! # BCE-with-logits, numerically certified — at a point where it has a CLOSED FORM

`planning/next_session_pipeline_then_r50.md` §4's BCE row, built and gated. RSB-A2/A3 do not train
with softmax cross-entropy: every class is an independent sigmoid, and the loss is
`BinaryCrossEntropy` with **`reduction='mean'` over B×K**.

## ⭐ The trick that makes this exact rather than a tolerance argument

The logits are `z = Wd·gap + bd`. **Set `Wd = 0` and `z = bd` for every example** — a vector this
harness chose, known to the last bit, with no forward pass to reproduce. The whole loss and its
whole cotangent then collapse to closed forms in `bd` and the targets:

    L      = mean_{b,k} ( softplus(z_k) − t_{b,k}·z_k )       softplus(x) = max(x,0) + log(1+e^{−|x|})
    dL/dz  = (σ(z) − t) / (B·K)
    g_bd[k] = Σ_b (σ(z_k) − t_{b,k}) / (B·K)

and `g_bd` is recovered from the committed **AdamW** render's stored first moment at `m = v = 0`,
where `m' = (1−β₁)·g = 0.1·g` — §2k's construction, reused by `r34-mom-tie`, `rms-tie`,
`conv-bias-zero`, `r50-lamb-tie` and now this.

⚠ Zeroing `Wd` degenerates the CLASSIFIER, not the net: the 50 layers below still run, `%loss` is
still computed from their output through the real head, and both quantities checked here are
functions of `bd` alone **by construction of the graph**, not by approximation. The degeneracy is
the instrument.

## ▶▶ The controls — three wrong BCEs, and the middle one is the expensive mistake

| ⟂ | the wrong neighbour | what it costs |
|---|---|---|
| ① | **softmax-CE** — the loss this replaces | the whole swap |
| ② | **`reduction='mean'` over B only**, i.e. the mean of the per-example SUM over classes | ⚠ `K = 1000×` on the effective step. The reference's own comment: *"that would be NC× larger and need an NC× smaller lr — RSB-A2's lr 5e-3 is tuned to this form"* |
| ③ | **label smoothing α = 0.1 folded into `t`** | A3's arg string is `…-ls0.0-`: the soft targets come from mixup through `%onehot`, NOT from a constant in the loss |

⭐ ② is the one worth having a gate for. It changes no shape, no op and no arity — only a divisor —
and a run with it descends perfectly well at 1/1000 of the intended learning rate.

    lake build r50-bce-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-bce-tie
-/

/-- `softplus(x) = max(x,0) + log(1 + exp(−|x|))`, the same stable form the render emits. -/
private def softplus (x : Float) : Float := max x 0.0 + Float.log (1.0 + Float.exp (-(x.abs)))
private def sigmoidF (x : Float) : Float := 1.0 / (1.0 + Float.exp (-x))

def main : IO Unit := do
  let net  := resnet50ImagenetVerified.toNet
  let bs   := 64
  let nP   := net.nParams
  let nc   := net.nClasses
  let tol  := (((← IO.getEnv "R50_BCE_TOL_U").bind (·.toNat?)).map (fun u => u.toFloat * 1e-6)
                |>.getD 5.0e-4)
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]

  IO.println "§4's BCE-with-logits row, measured — the render against BCE's closed form at Wd = 0"
  IO.println s!"  under test  verified_mlir/{net.slug}_adam64bce_train_step.mlir"
  IO.println s!"  peer        verified_mlir/{net.slug}_adam64_train_step.mlir  (softmax-CE, ⟂①)"
  IO.println s!"  {net.specs.size} params ({nP} floats), bs {bs}, {nc} classes, \
mean over B·K = {bs * nc}, backend {← LowererSession.backendName}"

  -- ── θ, with the CLASSIFIER WEIGHT ZEROED so the logits are exactly `bd` ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  let nT := net.specs.size
  for t in [0:nT] do
    let (dims, kind) := net.specs[t]!
    -- the last two tensors are `Wd` (rank 2) and `bd` (rank 1)
    if t == nT - 2 then
      θparts := θparts.push (← F32.const (dims.foldl (· * ·) 1).toUSize 0.0)          -- Wd := 0
    else if t == nT - 1 then
      -- `bd` is the whole experiment: distinctive, both signs, and large enough that σ spans its
      -- range. A zero fill would put every σ at exactly ½ and make ⟂③ (smoothing) invisible.
      θparts := θparts.push (← F32.scaleShift (← F32.heInit 6060 dims[0]!.toUSize 1.5) 1.0 (-0.2))
    else
      θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let z ← F32.const nP.toUSize 0.0
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  let mut lbl : Array Nat := #[]
  for i in [0:bs] do
    let c := (7 * i + 3) % 251
    lbl := lbl.push c
    y := y.push (UInt8.ofNat c); y := y.push 0; y := y.push 0; y := y.push 0

  let run (variant : String) : IO ByteArray := do
    let vmfb := s!".lake/build/r50_bce_tie_{variant}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/r50_bce_tie_{variant}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession s!"verified_mlir/{net.slug}_{variant}_train_step.mlir"
    -- lr = 0 so θ' is untouched; only `m'` (the gradient) and `%loss` are read.
    let tl ← F32.write3 (← F32.const 3 0.0) 0 0.0 0.1 0.001
    LowererSession.mlpTrainStepV sess s!"m.{net.slug}_{variant}_train_step" x
      (F32.concat #[θ, z, z, tl, bnIn]) shapes y bs.toUSize net.d0.toUSize nc.toUSize
  let oB ← run "adam64bce"
  let oC ← run "adam64"

  -- ── the closed form: z = bd, t = one-hot ──
  let bdOff := nP - nc                       -- `bd` is the last tensor
  let bk := (bs * nc).toFloat
  let mut lossWant := 0.0                    -- ② the loss
  let mut lossCE := 0.0                      -- ⟂① softmax-CE's, at the same point
  let mut dG := 0.0; let mut mG := 0.0       -- ① the bd gradient
  let mut c2 := 0.0; let mut c3 := 0.0       -- ⟂②/③ on the gradient
  -- softmax over z = bd is the SAME row for every example (Wd = 0), so one pass computes it.
  let mut smDen := 0.0
  let mut zMax := -1e30
  for k in [0:nc] do
    let zk := F32.read θ (bdOff + k).toUSize
    if zk > zMax then zMax := zk
  for k in [0:nc] do
    smDen := smDen + Float.exp (F32.read θ (bdOff + k).toUSize - zMax)
  for k in [0:nc] do
    let zk := F32.read θ (bdOff + k).toUSize
    let sm := Float.exp (zk - zMax) / smDen
    let mut sumT := 0.0                      -- Σ_b t[b,k]
    for b in [0:bs] do
      if lbl[b]! == k then sumT := sumT + 1.0
    -- the loss, summed over b and k
    lossWant := lossWant + bs.toFloat * softplus zk - sumT * zk
    -- ⚠⚠ SMOOTHED CE, not plain CE, and this line is the reason the cross-check exists. The peer
    -- render emits `−mean_b[(1−α)·Σ_k t·log sm + (α/K)·Σ_k log sm]` with α = 0.1; a first draft
    -- here wrote plain CE and the ⟂① guard below CAUGHT IT (7.815 against 7.800). A control that
    -- can catch the harness is worth more than one that can only catch the render.
    lossCE := lossCE - (0.9 * sumT + 0.1 / nc.toFloat * bs.toFloat) * Float.log (max sm 1e-30)
    -- the gradient of `bd`
    let g := 10.0 * F32.read oB (nP + bdOff + k).toUSize
    let want := (bs.toFloat * sigmoidF zk - sumT) / bk
    if (g - want).abs > dG then dG := (g - want).abs
    if want.abs > mG then mG := want.abs
    -- ⟂② the divisor is B, not B·K
    let w2 := (bs.toFloat * sigmoidF zk - sumT) / bs.toFloat
    if (g - w2).abs > c2 then c2 := (g - w2).abs
    -- ⟂③ label smoothing α = 0.1 folded into t
    let tS := 0.9 * sumT + 0.1 / nc.toFloat * bs.toFloat
    let w3 := (bs.toFloat * sigmoidF zk - tS) / bk
    if (g - w3).abs > c3 then c3 := (g - w3).abs
  let lossMean := lossWant / bk
  let lossCEMean := lossCE / bs.toFloat
  let lossGot := F32.read oB (3 * nP).toUSize
  let lossGotCE := F32.read oC (3 * nP).toUSize

  let relG := dG / max mG 1e-30
  let relL := (lossGot - lossMean).abs / max lossMean.abs 1e-30
  IO.println ""
  IO.println s!"  ① g_bd = (σ(z) − t)/(B·K)   max abs Δ {dG}   rel {relG}   (|g|max {mG})"
  IO.println s!"  ② %loss = mean(softplus(z) − t·z)   render {lossGot}   closed form {lossMean}\
   rel {relL}"
  IO.println s!"  ⟂① softmax-CE at the same point: the CE render says {lossGotCE}, BCE's closed \
form is {lossMean}, CE's closed form is {lossCEMean}"
  IO.println s!"  ⟂② divisor B instead of B·K   rel {c2 / max mG 1e-30}"
  IO.println s!"  ⟂③ label smoothing α = 0.1    rel {c3 / max mG 1e-30}"

  IO.println ""
  if mG < 1e-12 then
    throw (IO.userError s!"DEGENERATE: the bd gradient is ~0 (|max| {mG}) — nothing is being checked")
  if relG > tol then
    throw (IO.userError s!"① FAILED: g_bd is not (σ(z) − t)/(B·K) (rel {relG} > {tol}). If it is \
off by exactly {nc}×, the reduction is mean-over-B rather than mean-over-B·K")
  if relL > tol then
    throw (IO.userError s!"② FAILED: %loss is not BCE-with-logits (render {lossGot}, closed form \
{lossMean}, rel {relL})")
  -- ⟂① — the CE render must NOT agree with BCE's closed form, or the swap did nothing.
  if (lossGotCE - lossMean).abs / max lossMean.abs 1e-30 <= 10.0 * relL then
    throw (IO.userError s!"CONTROL DEAD: the softmax-CE render reports the same loss as BCE's \
closed form — the two losses are indistinguishable here, so ② means nothing")
  -- and it must agree with CE's own closed form, which is what says the harness computes losses
  -- correctly at all rather than that BCE happens to match anything.
  if (lossGotCE - lossCEMean).abs / max lossCEMean.abs 1e-30 > tol then
    throw (IO.userError s!"HARNESS WRONG: the CE render ({lossGotCE}) does not match softmax-CE's \
own closed form ({lossCEMean}) at this point — the loss arithmetic here is not trustworthy, so ② is \
not either")
  for (nm, c) in [("divisor B not B·K", c2), ("label smoothing 0.1", c3)] do
    if c / max mG 1e-30 <= 10.0 * relG then
      throw (IO.userError s!"CONTROL DEAD: '{nm}' fits as well as the real thing \
({c / max mG 1e-30} vs a tie of {relG})")
  IO.println s!"  ✅ CERTIFIED: the render is BCE-with-logits, reduction mean over B·K — gradient \
rel {relG}, loss rel {relL}, against a softmax-CE peer that reports {lossGotCE} where BCE's closed \
form is {lossMean} (and which matches CE's OWN closed form {lossCEMean}, so the harness's arithmetic is \
not the thing being trusted), a wrong-divisor control missing by {c2 / max mG 1e-30} and a \
label-smoothing control missing by {c3 / max mG 1e-30}."
  IO.println "     ⚠ This certifies the LOSS. `lamb64bce` shares the same emitted loss block and \
the same cotangent, so it inherits this; nothing here says anything about the recipe's batch or \
resolution."
