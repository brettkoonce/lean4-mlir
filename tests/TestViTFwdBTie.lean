import LeanMlir.Proofs.Codegen.ViTRenderB

/-! # The batched-index ViT forward emits the committed forward, BYTE FOR BYTE

`lake build vit-fwd-b-tie && .lake/build/bin/vit-fwd-b-tie`

`LeanMlir/Proofs/Codegen/ViTRenderB.lean` renders ViT-Tiny's depth-12 forward at the batched index
`N := B` — every node a `batchOp`/`*B` form whose `den` is `batchMap N (…)`, rather than a
per-example node that `pretty B` lifts. The move exists so a per-EXAMPLE stochastic-depth mask is
expressible at all (handoff §0.2 ▶3); the claim this file gates is that it changed **nothing else**:

> the batched chain emits `verified_mlir/vit_fwd.mlir` byte for byte.

⚠⚠ **WHY THE BYTE TIE IS THE RIGHT BAR HERE, AND SHARPER ON ViT THAN ON ConvNeXt.** ConvNeXt's
batched chain differs from its per-example one on 78 lines (two conv-VJP emitters that were never
tied to each other), so its train-step tie has to carry an allowance. ViT uses ONE emitter per op —
every batched form aliases or reproduces its per-example peer's text — so the bar here is exact
byte-identity with no allowance at all, and anything else is a defect rather than a known
divergence.

⚠ **What this does NOT establish.** The `den` side. `skel` erases values, so a batched form with the
wrong denotation emits identical bytes — which is precisely the trap this thread is about. On ViT
the sharp cases are `matmulFB` (example `k` must use its OWN `Q` against its OWN `K`;
`den_matmulFB_per_example`) and `clsSlice` (which CONTRACTS, so reading the batch as the token axis
keeps one example and drops the rest, and at `N = tk` it agrees — `den_batchOp_clsSlice_per_example`).
Both are in the axiom audit. Neither half implies the other.
-/

open Proofs.StableHLO

/-- Fail via `throw`, never `IO.Process.exit` — under `#eval` the elaborator buffers output and
    `exit` discards every diagnostic (§4). -/
def main : IO Unit := do
  let want ← IO.FS.readFile "verified_mlir/vit_fwd.mlir"
  let got := vitFwdRenderB "vit_fwd" 10
  IO.println "── ViT: the batched-index forward vs the committed per-example artifact ──"
  IO.println s!"  committed : {want.length} chars, {(want.splitOn "\n").length} lines"
  IO.println s!"  batched   : {got.length} chars, {(got.splitOn "\n").length} lines"
  if got == want then
    IO.println "  ✅ BYTE-IDENTICAL — the batched index changed the denotation, not the render"
    IO.println "  ⚠ The den side is NOT checked here: skel erases values, so a wrong denotation \
emits identical bytes. That half is den_matmulFB_per_example / den_batchOp_clsSlice_per_example."
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
    throw <| IO.userError "MISMATCH: the batched ViT forward does not emit the committed artifact. \
Run `lake env lean tests/TestBatchedEmitTie.lean` FIRST — it localises which of the 47 batched \
forms diverged from its per-example peer, which this whole-net diff cannot."
