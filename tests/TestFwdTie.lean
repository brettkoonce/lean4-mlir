import LeanMlir.VerifiedNets

/-! # `@<slug>_fwd` render tie — do two renders compute the same function?

Migration guard for `planning/xla_pjrt_handoff.md` §2a, the **forward** half. The five forward
artifacts `mobilenetv2_fwd{,_eval}`, `efficientnet_fwd{,_eval}` and `convnext_fwd` used to be
written by hand-written string emitters in `tests/Test*Fwd.lean`, independent of the proven graph;
they are now `pretty(provenGraph)` out of `LeanMlir/Proofs/Codegen/*Render.lean`, sharing their
forward chain with the train step. The two emitters produce *textually* different MLIR by
construction, so the only meaningful check is numeric: feed both the same parameters and the same
input, and compare logits.

    lake build fwd-tie
    .lake/build/bin/fwd-tie <slug> [--eval] [<refRender.mlir> [<newRender.mlir>]]

`slug` ∈ `convnext | efficientnet | mobilenetv2 | resnet34`. `--eval` ties `@<slug>_fwd_eval`
instead: the params plus 2 running-stat inputs per BN layer (μ/var interleaved in `bnChannels`
order) and frozen-stat affine BN. ConvNeXt has no `_fwd_eval` and must not grow one — LayerNorm
reduces within one example, so its forward is already class-batch-independent.

Defaults to comparing the artifact against itself (a self-tie smoke test). Exits non-zero if the
renders disagree, or if the comparison is **degenerate** — all-zero or non-finite logits agree
trivially and prove nothing.

**This replaces `tests/TestResnet34FwdTie.lean`** (`lake build resnet34-fwd-tie`), which was this
file with `resnet34Verified` hardcoded; `fwd-tie resnet34 [--eval]` is the same check. Recover it
from `git show 17413f0:tests/TestResnet34FwdTie.lean` if ever needed. Two near-identical tie
harnesses would be the double-writer disease one level down, in code.

Two things it does that the R34-specific version did not:

* **it deletes its `.vmfb` before every compile** (§4). `compileVmfb` reuses any existing output
  newer than the `.mlir` — the cache key is the output path and an mtime, never the source — so
  running a tie twice with different candidates under one tag silently reuses the FIRST candidate's
  binary and reports the second as a perfect match. Running a negative control looks exactly like
  that;
* **it counts bit-exact coordinates**, because `Float.toString` gives six decimals and a genuine
  3e-8 prints as `0.000000`, which reads as bit-exact when it is not (§2e-bis).
-/

/-- The driver's own init (`VerifiedTrain.mkParam`, which is private): He(fan-in) weights,
    γ = 1, β/bias = 0. Using the real init matters — a constant-splat parameter set makes BN see
    zero variance, and two wrong renders would agree on the resulting garbage. -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

private def netBySlug (slug : String) : IO VerifiedNetSpec :=
  match slug with
  | "convnext"     => pure convnextVerified
  | "efficientnet" => pure efficientnetVerified
  | "mobilenetv2"  => pure mobilenetv2Verified
  | "resnet34"     => pure resnet34Verified
  -- §2i: the cifar8 family. `cifar8w` is `cifar8` at d1=512 (the Chapter-5 wide-head "bridge" net),
  -- so all four ride the same two certified forward renderers at two widths.
  | "cifar8"       => pure cifar8Verified
  | "cifar8_bn"    => pure cifar8BnVerified
  | "cifar8w"      => pure cifar8wVerified
  | "cifar8w_bn"   => pure cifar8wBnVerified
  | _ => throw (IO.userError
      s!"unknown net slug '{slug}' — expected convnext | efficientnet | mobilenetv2 | resnet34 \
| cifar8 | cifar8_bn | cifar8w | cifar8w_bn")

def main (args : List String) : IO Unit := do
  let isEval := args.contains "--eval"
  let rest   := args.filter (· != "--eval")
  let (slug, paths) ← match rest with
    | s :: ps => pure (s, ps)
    | [] => throw (IO.userError
        ("usage: fwd-tie <slug> [--eval] [<pathA> [<pathB>]]\n" ++
         "  slug ∈ convnext | efficientnet | mobilenetv2 | resnet34 | cifar8[w][_bn]"))
  if isEval && slug == "convnext" then
    throw (IO.userError
      "convnext has no @convnext_fwd_eval — LayerNorm makes its forward class-batch-independent, \
       so train == eval and the frozen-stats peer would be the SAME graph. Drop --eval.")
  if isEval && slug.startsWith "cifar8" then
    throw (IO.userError
      "the cifar8 family has no @<slug>_fwd_eval — its BatchNorm is PER-CHANNEL PER-EXAMPLE, so it \
       keeps no running statistics (`bnChannels` is empty) and train == eval. Drop --eval.")
  let fn    := if isEval then s!"{slug}_fwd_eval" else s!"{slug}_fwd"
  let dflt  := s!"verified_mlir/{fn}.mlir"
  let (pathA, pathB) := match paths with
    | a :: b :: _ => (a, b)
    | [a]         => (a, dflt)
    | []          => (dflt, dflt)
  let net := (← netBySlug slug).toNet
  -- The batch is BAKED into each forward render, and the two families disagree: the four large nets
  -- are bs 32, the whole cifar8 family is bs 128. A wrong value here is a shape error on the first
  -- invoke, not a silent pass, but it is still a per-net fact rather than a constant.
  let bs  := if slug.startsWith "cifar8" then 128 else 32
  IO.println s!"@{fn} tie: A={pathA}  B={pathB}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), bs {bs}, backend {← IreeSession.backendName}"

  -- ── one deterministic (θ, x) both renders see ──
  let mut parts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    parts := parts.push (← mkParam sd dims kind)
    sd := sd + 1
  let mut params := F32.concat parts
  -- eval BN consumes frozen per-channel stats, appended after the params in `bnChannels` order,
  -- μ then var per layer. μ is centred noise; var must be POSITIVE (rsqrt) and must vary — a
  -- constant var would let a broadcast bug pass unnoticed.
  let mut shapes := net.shapesBA
  if isEval then
    let mut stats : Array ByteArray := #[]
    let mut ss := 5000
    for c in net.bnChannels do
      stats := stats.push (← F32.heInit ss.toUSize c.toUSize 0.30)                        -- μ
      stats := stats.push (← F32.scaleShift (← F32.heInit (ss+1).toUSize c.toUSize 0.20)
                               1.0 1.0)                                                   -- var ≈ 1 ± 0.2
      ss := ss + 2
    params := F32.concat (#[params] ++ stats)
    shapes := packShapes (net.paramShapes ++
                net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[])
    IO.println s!"  + {net.bnChannels.size} BN layers → {net.bnChannels.foldl (· + 2 * ·) 0} running-stat floats"
  let x ← F32.heInit 987654 (bs * net.d0).toUSize 1.0
  let xsh := net.xShape bs

  -- Delete both the bare and the `_$IREE_BACKEND`-scoped `.vmfb` before compiling — see the
  -- header. Without this a second run with a different candidate silently re-uses the first.
  let runOne (path tag : String) : IO ByteArray := do
    let vmfb := s!".lake/build/fwd_tie_{fn}_{tag}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/fwd_tie_{fn}_{tag}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession path
    IreeSession.forwardF32 sess s!"m.{fn}" params shapes x xsh
      bs.toUSize net.nClasses.toUSize
  let la ← runOne pathA "a"
  let lb ← runOne pathB "b"

  -- ── compare, and refuse a degenerate agreement ──
  let n := bs * net.nClasses
  let mut maxAbs    : Float := 0.0
  let mut maxRel    : Float := 0.0
  let mut maxMag    : Float := 0.0
  let mut nonFinite : Nat   := 0
  let mut exact     : Nat   := 0
  for i in [0:n] do
    let a := F32.read la i.toUSize
    let b := F32.read lb i.toUSize
    if !a.isFinite || !b.isFinite then nonFinite := nonFinite + 1
    if a == b then exact := exact + 1
    let d := (a - b).abs
    let m := max a.abs b.abs
    if m > maxMag then maxMag := m
    if d > maxAbs then maxAbs := d
    let r := if m > 1e-30 then d / m else 0.0
    if r > maxRel then maxRel := r

  IO.println s!"  logits |max| = {maxMag}   max abs diff = {maxAbs}   max rel diff = {maxRel}"
  IO.println s!"  bit-exact coordinates: {exact}/{n}"
  if nonFinite > 0 then
    throw (IO.userError s!"DEGENERATE: {nonFinite}/{n} non-finite logits — the tie proves nothing")
  if maxMag < 1e-6 then
    throw (IO.userError s!"DEGENERATE: logits are all ~0 (|max| = {maxMag}) — the tie proves nothing")
  if maxRel > 1e-4 then
    throw (IO.userError
      s!"TIE FAILED: max rel diff {maxRel} > 1e-4 — the renders compute different functions")
  IO.println s!"✓ renders TIE (max rel {maxRel} ≤ 1e-4) on non-degenerate logits"
