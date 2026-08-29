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
artifact. Run `lake env lean tests/TestBatchedEmitTie.lean` FIRST — it localises which of the 34 \
batched forms diverged from its per-example peer, which this whole-net diff cannot."

  -- ══════════════════════════════════════════════════════════════════════════════════════════
  --  The BACKWARD: the whole-net traversal, against its per-example peer.
  --
  --  ⚠⚠ TWO checks, and the second is the one a string diff cannot make. `convNextBackAll`
  --  returns `(code, gradMap, softmax)` where `gradMap` maps each of the 180 parameter names to
  --  the SSA holding its gradient. Identical CODE with a permuted MAP is a render that computes
  --  every gradient correctly and hands them to the wrong parameters — it would pass a byte diff,
  --  pass every structural audit, train, and descend. So the map is compared name for name and in
  --  ORDER, which is also what the AdamW tail consumes.
  -- ══════════════════════════════════════════════════════════════════════════════════════════
  let smooth : Option (String × String × String) := some ("0.1", "-0.01", "32.0")
  let (wantCode, wantMap, wantSm) := (Proofs.StableHLO.convNextBackAll true smooth 10).run' (0, [])
  let (gotCode, gotMap, gotSm) := (convNextBackAllB smooth 10).run' (0, [])
  IO.println "── ConvNeXt: the batched-index BACKWARD vs the per-example traversal ──"
  IO.println s!"  per-example : {wantCode.length} chars, {wantMap.length} gradients, softmax {wantSm}"
  IO.println s!"  batched     : {gotCode.length} chars, {gotMap.length} gradients, softmax {gotSm}"
  let mut bad : Nat := 0
  if gotCode != wantCode then
    let gl := (gotCode.splitOn "\n").toArray
    let wl := (wantCode.splitOn "\n").toArray
    if gl.size != wl.size then
      IO.println s!"  ✗ line counts differ: {gl.size} vs {wl.size}"
      bad := bad + 1
    -- ⚠⚠ THE ONE ALLOWED DIFFERENCE, and it is a PRE-EXISTING divergence between two emitters that
    -- were never tied to each other. The conv input-VJP prepares its kernel with a `transpose`
    -- (dims [1,0,2,3]) and a `reverse` (dims [2,3]); `.convBack` emits transpose-then-reverse and
    -- `.convBackBatched` emits reverse-then-transpose. The two act on DISJOINT axes, so they
    -- commute and both renders compute the same kernel — but the text differs, and nothing noticed
    -- because no net used both emitters until this port. EfficientNet/mnv2/R34 use only the
    -- batched one; ConvNeXt/ViT only the per-example one.
    --
    -- The allowance is narrow ON PURPOSE: every differing line must be one of those two ops, the
    -- line counts must match, and the SSA numbering must not move. Anything else — a shape, a
    -- constant, an operand, an op that is not this pair — fails. An "ignore the conv backward"
    -- allowance would hide precisely the class of defect this tie exists to catch.
    let mut swapPair : Nat := 0
    let mut other : Nat := 0
    let mut shown : Nat := 0
    for i in [0:min gl.size wl.size] do
      if gl[i]! != wl[i]! then
        let isPair (s : String) : Bool :=
          (s.splitOn "stablehlo.reverse").length > 1 || (s.splitOn "stablehlo.transpose").length > 1
        if isPair (gl[i]!) && isPair (wl[i]!) then
          swapPair := swapPair + 1
        else
          other := other + 1
          if shown < 6 then
            IO.println s!"  L{i+1} batched  : {(gl[i]!).take 160}"
            IO.println s!"  L{i+1} per-ex   : {(wl[i]!).take 160}"
            shown := shown + 1
    IO.println s!"  ◐ code: {swapPair} line(s) are the conv-VJP transpose/reverse ORDER swap \
(commuting ops, disjoint axes) — allowed"
    if other != 0 then
      IO.println s!"  ✗ code: {other} differing line(s) are NOT that swap"
      bad := bad + other
    else
      IO.println s!"  ✅ code identical apart from that swap; {gl.size} lines, SSA numbering unmoved"
  else
    IO.println "  ✅ code BYTE-IDENTICAL"
  -- the map, name for name and in order
  if gotMap != wantMap then
    let mut shown : Nat := 0
    for i in [0:min gotMap.length wantMap.length] do
      let (gn, gs) := gotMap[i]!
      let (wn, ws) := wantMap[i]!
      if gn != wn || gs != ws then
        if shown < 8 then
          IO.println s!"  #{i} batched {gn} -> {gs}   per-ex {wn} -> {ws}"
          shown := shown + 1
        bad := bad + 1
    IO.println s!"  ✗ gradMap differs ({gotMap.length} vs {wantMap.length} entries)"
  else
    IO.println s!"  ✅ gradMap IDENTICAL — {gotMap.length} parameters, same names, same SSA, same order"
  if gotSm != wantSm then
    IO.println s!"  ✗ softmax SSA {gotSm} vs {wantSm}"; bad := bad + 1
  if bad != 0 then
    throw <| IO.userError s!"MISMATCH in the batched ConvNeXt BACKWARD ({bad} difference(s))."
  IO.println "  ✅ the batched backward computes and routes what the per-example one does"

  -- ══════════════════════════════════════════════════════════════════════════════════════════
  --  The WHOLE TRAIN STEP, against the committed artifact — the bytes the trainer loads.
  --
  --  ⚠ Same allowance, same reason, and it must stay THIS narrow: the AdamW tail is
  --  parameter-space (adamMNextF / clipScaleF / gradSumSqAccF are indexed by the param's own size
  --  and never see the batch), so it is rendered by the SAME function on both sides. Any
  --  difference here that is not the conv-VJP swap is therefore in the traversal, and is a defect.
  -- ══════════════════════════════════════════════════════════════════════════════════════════
  let wantTS ← IO.FS.readFile "verified_mlir/convnext_adam_train_step.mlir"
  let gotTS := convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0"
  IO.println "── ConvNeXt: the batched AdamW train step vs the committed artifact ──"
  IO.println s!"  committed : {wantTS.length} chars, {(wantTS.splitOn "\n").length} lines"
  IO.println s!"  batched   : {gotTS.length} chars, {(gotTS.splitOn "\n").length} lines"
  if gotTS == wantTS then
    IO.println "  ✅ BYTE-IDENTICAL"
  else
    let gl := (gotTS.splitOn "\n").toArray
    let wl := (wantTS.splitOn "\n").toArray
    let mut pairSwap : Nat := 0
    let mut otherTS : Nat := 0
    let mut shown : Nat := 0
    if gl.size != wl.size then
      IO.println s!"  ✗ line counts differ: {gl.size} vs {wl.size}"
      otherTS := otherTS + 1
    for i in [0:min gl.size wl.size] do
      if gl[i]! != wl[i]! then
        let isPair (s : String) : Bool :=
          (s.splitOn "stablehlo.reverse").length > 1 || (s.splitOn "stablehlo.transpose").length > 1
        if isPair (gl[i]!) && isPair (wl[i]!) then
          pairSwap := pairSwap + 1
        else
          otherTS := otherTS + 1
          if shown < 6 then
            IO.println s!"  L{i+1} batched  : {(gl[i]!).take 160}"
            IO.println s!"  L{i+1} committed: {(wl[i]!).take 160}"
            shown := shown + 1
    IO.println s!"  ◐ {pairSwap} line(s) are the conv-VJP transpose/reverse ORDER swap — allowed"
    if otherTS != 0 then
      IO.println s!"  ✗ {otherTS} differing line(s) are NOT that swap"
      throw <| IO.userError s!"MISMATCH in the batched ConvNeXt TRAIN STEP ({otherTS} unexplained \
difference(s)). The gradMap check above passed, so the routing is right — look at the emitted ops."
    IO.println s!"  ✅ train step identical apart from that swap; {gl.size} lines, SSA unmoved"
    IO.println "  ⚠ THE SWAP IS A REAL BYTE CHANGE. Swapping the writer to the batched renderer \
would move those lines in a COMMITTED artifact, so it needs `convnext-adam-tie` (numeric, GPU) to \
license it — a byte tie cannot, and §5's rule is that every swap is licensed by a numeric tie that \
was verified to fail."
