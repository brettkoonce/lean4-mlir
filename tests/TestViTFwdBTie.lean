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

  -- ══════════════════════════════════════════════════════════════════════════════════════════
  --  The BACKWARD: the whole-net traversal, against its per-example peer.
  --
  --  ⚠⚠ TWO checks, and the second is the one a string diff cannot make. `vitBackAll` returns the
  --  200 gradient SSAs IN FUNC-ARG ORDER, and that list is what the AdamW tail zips against
  --  `vitParamSig`. Identical CODE with a permuted LIST is a render that computes every gradient
  --  correctly and hands them to the WRONG PARAMETERS — it passes a byte diff, passes every
  --  structural audit, trains, and descends. On ViT the permutation is a live hazard rather than a
  --  hypothetical: `blkNames` is pushed in REVERSE block order and re-indexed by
  --  `blkNames[vDEPTH-1-i]`, so an off-by-one there silently pairs block 3's gradients with block
  --  8's parameters.
  -- ══════════════════════════════════════════════════════════════════════════════════════════
  let smooth : Option (String × String × String) := some ("0.1", "-0.01", "32.0")
  let (wantCode, wantNames, wantSm) := (Proofs.StableHLO.vitBackAll 32 10 "0.003125" true smooth).run' 0
  -- ⚠⚠ `32` IS THE BATCH AND IT WAS MISSING. `vitBackAllB` gained a LEADING `vbB` when ViT stopped
  -- being batch-32-only, and this call was never updated — so this target has failed to BUILD ever
  -- since, and `.lake/build/bin/vit-fwd-b-tie` kept passing because the binary on disk was stale
  -- (dated 2026-08-03, i.e. older than the parameter). ▶ A tie that reports ✅ from a stale binary
  -- is worse than one that fails: `lake build vit-fwd-b-tie` exits 1 and the runner still prints
  -- three green lines. Run the BUILD, not just the binary.
  let (gotCode, gotNames, gotSm) := (vitBackAllB 32 10 smooth).run' 0
  IO.println "── ViT: the batched-index BACKWARD vs the per-example traversal ──"
  IO.println s!"  per-example : {wantCode.length} chars, {wantNames.length} gradients, softmax {wantSm}"
  IO.println s!"  batched     : {gotCode.length} chars, {gotNames.length} gradients, softmax {gotSm}"
  let mut bad : Nat := 0
  if gotCode != wantCode then
    let gl := (gotCode.splitOn "\n").toArray
    let wl := (wantCode.splitOn "\n").toArray
    if gl.size != wl.size then
      IO.println s!"  ✗ line counts differ: {gl.size} vs {wl.size}"
    let mut shown : Nat := 0
    let mut diffs : Nat := 0
    for i in [0:min gl.size wl.size] do
      if gl[i]! != wl[i]! then
        diffs := diffs + 1
        if shown < 6 then
          IO.println s!"  L{i+1} batched  : {(gl[i]!).take 160}"
          IO.println s!"  L{i+1} per-ex   : {(wl[i]!).take 160}"
          shown := shown + 1
    -- ⚠ NO ALLOWANCE, unlike ConvNeXt's tie. ConvNeXt has two independent emitters for the conv
    -- input-VJP that were never tied to each other (78 lines of commuting transpose/reverse); ViT
    -- has one emitter per op, so any differing line here is a defect.
    IO.println s!"  ✗ code: {diffs} differing line(s); ViT's tie carries NO allowance"
    bad := bad + diffs
  else
    IO.println s!"  ✅ code BYTE-IDENTICAL — {(gotCode.splitOn "\n").length} lines, SSA numbering unmoved"
  if gotNames != wantNames then
    let mut shown : Nat := 0
    for i in [0:min gotNames.length wantNames.length] do
      if gotNames[i]! != wantNames[i]! then
        if shown < 8 then
          IO.println s!"  #{i} batched {gotNames[i]!}   per-ex {wantNames[i]!}"
          shown := shown + 1
        bad := bad + 1
    IO.println s!"  ✗ gradient list differs ({gotNames.length} vs {wantNames.length} entries)"
  else
    IO.println s!"  ✅ gradient list IDENTICAL — {gotNames.length} parameters, same SSA, same ORDER"
  if gotSm != wantSm then
    IO.println s!"  ✗ softmax SSA {gotSm} vs {wantSm}"; bad := bad + 1
  if bad != 0 then
    throw <| IO.userError s!"MISMATCH in the batched ViT BACKWARD ({bad} difference(s))."
  IO.println "  ✅ the batched backward computes and routes what the per-example one does"
  IO.println "  ⚠ And the den side is still NOT checked: the CLS-token gradient changed from \
`denseBiasGradB (N := 1)` to `(N := vbB)` — sum-one-thing to sum-the-batch — with the SAME emitted \
text. That is the batched-index move in one line, and only den_rowDenseBiasGradB_at_one's argument \
covers it."

  -- ══════════════════════════════════════════════════════════════════════════════════════════
  --  The WHOLE TRAIN STEP, against the committed artifact — the bytes the trainer loads.
  --
  --  ⚠ The AdamW tail is rendered by the SAME function on both sides (`vitAdamTrainStepFaithful`
  --  with `traversal` swapped), because it is parameter-space and never sees the batch. So any
  --  difference here that the backward check above did not already report is in the seam between
  --  the traversal and the tail — the fresh-name counter, or the gradient list the tail zips.
  -- ══════════════════════════════════════════════════════════════════════════════════════════
  let wantTS ← IO.FS.readFile "verified_mlir/vit_adam_train_step.mlir"
  let gotTS := vitAdamTrainStepFaithfulB "vit_adam_train_step"
  IO.println "── ViT: the batched AdamW train step vs the committed artifact ──"
  IO.println s!"  committed : {wantTS.length} chars, {(wantTS.splitOn "\n").length} lines"
  IO.println s!"  batched   : {gotTS.length} chars, {(gotTS.splitOn "\n").length} lines"
  if gotTS == wantTS then
    IO.println "  ✅ BYTE-IDENTICAL — the whole train step, no allowance"
  else
    let gl := (gotTS.splitOn "\n").toArray
    let wl := (wantTS.splitOn "\n").toArray
    if gl.size != wl.size then IO.println s!"  ✗ line counts differ: {gl.size} vs {wl.size}"
    let mut diffs : Nat := 0
    let mut shown : Nat := 0
    for i in [0:min gl.size wl.size] do
      if gl[i]! != wl[i]! then
        diffs := diffs + 1
        if shown < 6 then
          IO.println s!"  L{i+1} batched  : {(gl[i]!).take 160}"
          IO.println s!"  L{i+1} committed: {(wl[i]!).take 160}"
          shown := shown + 1
    throw <| IO.userError s!"MISMATCH in the batched ViT TRAIN STEP ({diffs} differing line(s)). \
The gradient-list check above passed, so the routing is right — look at the fresh-name seam."
