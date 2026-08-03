import LeanMlir.Proofs.Codegen.ConvNeXtRenderB

/-! # The batched-index ConvNeXt forward emits the committed forward, BYTE FOR BYTE

`lake build convnext-fwd-b-tie && .lake/build/bin/convnext-fwd-b-tie`

`LeanMlir/Proofs/Codegen/ConvNeXtRenderB.lean` renders ConvNeXt-T's forward at the batched index
`N := B` — every node a `batchOp`/`*B` form whose `den` is `batchMap N (…)`, rather than a
per-example node that `pretty B` lifts. The move exists so a per-EXAMPLE stochastic-depth mask is
expressible at all (handoff §0.2 ▶2); the claim this file gates is that it changed **nothing else**:

> the batched chain emits `verified_mlir/convnext_fwd.mlir` byte for byte.

⚠⚠ **Why a BYTE tie is available here, where §2b's R34 move needed a numeric one.** Every batched
form was built to emit its per-example peer's text byte-for-byte, and
`tests/TestBatchedEmitTie.lean` pins all 31 of them individually. So the whole-net statement is the
per-form statement composed — and when it fails, that file localises which form did it in one run,
which a numeric tie cannot do. §2b had no such per-form corpus at the time and paid for it with a
1e-6 tolerance argument.

⚠ **What this does NOT establish.** The `den` side. `skel` erases values, so a batched form with the
wrong denotation emits identical bytes and passes this file — which is precisely the trap the whole
thread is about (`softmaxDiv`'s batched `den` would have divided by the whole batch's sum while
emitting the same MLIR). That half is `den_batchOp_*` in `StableHLO.lean`, and neither half implies
the other.

⚠ The banner comment is passed in rather than compared modulo: a tie with a one-line hole is a tie
with a hole, and the hole would sit exactly where a renderer describes what it did.
-/

open Proofs.StableHLO

/-- Fail via `throw`, never `IO.Process.exit` — under `#eval` the elaborator buffers output and
    `exit` discards every diagnostic (§4). -/
def main : IO Unit := do
  let want ← IO.FS.readFile "verified_mlir/convnext_fwd.mlir"
  let got := convNextFwdRenderB "convnext_fwd" 10 cnxFwdPerExampleBanner
  IO.println "── ConvNeXt: the batched-index forward vs the committed per-example artifact ──"
  IO.println s!"  committed : {want.length} chars, {(want.splitOn "\n").length} lines"
  IO.println s!"  batched   : {got.length} chars, {(got.splitOn "\n").length} lines"
  if got == want then
    IO.println "  ✅ BYTE-IDENTICAL — the batched index changed the denotation, not the render"
  else
    let gl := (got.splitOn "\n").toArray
    let wl := (want.splitOn "\n").toArray
    let mut diffs : Nat := 0
    for i in [0:min gl.size wl.size] do
      if gl[i]! != wl[i]! then
        if diffs < 8 then
          IO.println s!"  L{i+1} batched  : {(gl[i]!).take 160}"
          IO.println s!"  L{i+1} committed: {(wl[i]!).take 160}"
        diffs := diffs + 1
    IO.println s!"  ✗ {diffs} differing line(s); lengths {gl.size} vs {wl.size}"
    throw <| IO.userError "MISMATCH: the batched ConvNeXt forward does not emit the committed \
artifact. Run `lake env lean tests/TestBatchedEmitTie.lean` FIRST — it localises which of the 31 \
batched forms diverged from its per-example peer, which this whole-net diff cannot."
