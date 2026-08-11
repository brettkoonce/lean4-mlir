import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.VerifiedTrain
import LeanMlir.Types

/-! # §2l step 1 — can the emitter spell a **1×1 strided** conv, and does it compute the right one?

The check `planning/xla_pjrt_handoff.md` §2l puts before everything else. The paper's ResNet-34
option-B shortcut is a **1×1** stride-2 projection; this repo's `downFwdB` builds it from
`Kernel4 c cin 3 3`, the same kernel as the block's first conv (§2k). Before re-instantiating the
render at `kHp = kWp = 1`, settle what the four strided-conv ops do there.

The worry is precedent, not speculation: §2f-bis found the symmetric-SAME formula `p = (k−1)/2`
**could not** spell an even kernel (k = 2 ⇒ p = 0 ⇒ a result one short of k, type-invalid MLIR), and
ConvNeXt's 2×2/s2 weight grad had to be hand-written. 1 is odd, so `p = 0` should fall out — but
"should" is the word that made that check necessary.

Two gates, and the second is the one that matters:

1. **It emits type-valid MLIR** — the whole strided family at `k = 1`, `iree-compile`d, with the
   committed `k = 3` rendered alongside as the control (if 3×3 fails, the harness is wrong rather
   than the kernel size). That is `main`'s render arm; it needs no GPU.
2. **It computes the function `den` says it does** — a known answer, on device. `den` is
   `flatConvStride2 = decimateFlat ∘ flatConv`, and `decimateIdx` reads the **even** positions
   `(2·ho, 2·wo)`, so at `k = 1` the whole family collapses to something writable in closed form:

   | op | known answer at `k = 1` |
   |---|---|
   | fwd | `y[b,o,i,j] = bias[o] + Σ_c W[o,c]·x[b,c,2i,2j]` |
   | input-VJP | `dx[b,c,2i,2j] = Σ_o W[o,c]·dy[b,o,i,j]`, **0 at odd positions** |
   | weight grad | `dW[o,c] = Σ_{b,i,j} x[b,c,2i,2j]·dy[b,o,i,j]` |
   | bias grad | `db[o] = Σ_{b,i,j} dy[b,o,i,j]` |

   This is the un-proven half by construction: `convStrided_faithful` et al are `rfl` and generic in
   `kH kW`, so the *denotation* at `k = 1` is already proven — what no proof covers is that
   `emitTok`'s text matches it (§5's audited lexical boundary). The odd-position zeros in the
   input-VJP are the load-bearing part: they are what distinguishes `decimate ∘ conv` from a conv
   that read the wrong pixel, and a stride/alignment slip would leave them non-zero.

⚠ The known answer is computed from the SAME buffers the device saw, read back through `F32.read`,
so it cannot drift from the inputs. It is a reference implementation, not a second render.

Run (render-only, no GPU):
  `lake build LeanMlir.Proofs.Codegen.StableHLO && lake env lean tests/TestStrided1x1.lean`
  (the `lake build` first is §4's `lake env lean` trap — it links committed `.olean`s.)
Run (the numeric gate):
  `lake build strided-1x1 && HIP_VISIBLE_DEVICES=0 .lake/build/bin/strided-1x1`
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 2
private def IC : Nat := 4
private def OC : Nat := 8
private def HH : Nat := 3          -- output spatial; input is 2·HH × 2·HH
private def HIN : Nat := 2 * HH

private def nX  : Nat := BS * (IC * HIN * HIN)
private def nY  : Nat := BS * (OC * HH * HH)
private def nW  : Nat := OC * IC     -- ×1×1
private def zk1 : Kernel4 OC IC 1 1 := fun _ _ _ _ => 0

-- ════════════════════════════════════════════════════════════════
-- § The renders — every node a real `SHlo` through `pretty`
-- ════════════════════════════════════════════════════════════════

private def zc   : Vec OC := fun _ => 0
private def zinS : Vec (BS*(IC*(2*HH)*(2*HH))) := fun _ => 0
private def zout : Vec (BS*(OC*HH*HH)) := fun _ => 0

/-- All four ops in one `func.func` — the compile-only shape check, at kernel `K×K`. -/
private def probeModule (K : Nat) (zk : Kernel4 OC IC K K) : String :=
  let go : StateM Nat String := do
    let (cF,  nF)  ← pretty BS (.batchOp (N := BS)
      (.convStrided (h := HH) (w := HH) "%W" "%b" zk zc) (.operand "%x" zinS))
    let (cDx, nDx) ← pretty BS (.convStridedBackBatched (N := BS) (ic := IC) (oc := OC)
      (h := HH) (w := HH) "%W" zk zc (.operand "%dy" zout))
    let (cW,  nWg) ← pretty BS (.convStridedWeightGradB "%x" zc zinS zk (.operand "%dy" zout))
    let (cB,  nBg) ← pretty BS (.convStridedBiasGradB (h := HH) (w := HH) zk zinS zc
      (.operand "%dy" zout))
    let retTys := String.intercalate ", " [ty [BS, OC*HH*HH], ty [BS, IC*HIN*HIN],
                                           ty [OC,IC,K,K], ty [OC]]
    pure (cF ++ cDx ++ cW ++ cB ++
          s!"    return {String.intercalate ", " [nF, nDx, nWg, nBg]} : {retTys}\n")
  let body : String := go.run' 0
  let sig := String.intercalate ", "
    [s!"%x: {ty [BS, IC*HIN*HIN]}", s!"%W: {ty [OC,IC,K,K]}", s!"%b: {ty [OC]}",
     s!"%dy: {ty [BS, OC*HH*HH]}"]
  let retTys := String.intercalate ", " [ty [BS, OC*HH*HH], ty [BS, IC*HIN*HIN],
                                         ty [OC,IC,K,K], ty [OC]]
  "module @m {\n" ++ s!"  func.func @strided_k{K}({sig}) -> ({retTys}) " ++ "{\n" ++
  body ++ "  }\n}\n"

/-- One op per module, so each can be driven through `forwardF32`'s single-output invoke.
    `x` is whichever buffer is pushed FIRST; the rest ride the packed-params slot. -/
private def oneOpModule (which : String) : String :=
  let go : StateM Nat (String × String) := do
    match which with
    | "fwd" => pretty BS (.batchOp (N := BS)
        (.convStrided (h := HH) (w := HH) "%W" "%b" zk1 zc) (.operand "%x" zinS))
    | "dx"  => pretty BS (.convStridedBackBatched (N := BS) (ic := IC) (oc := OC)
        (h := HH) (w := HH) "%W" zk1 zc (.operand "%dy" zout))
    | "dw"  => pretty BS (.convStridedWeightGradB "%x" zc zinS zk1 (.operand "%dy" zout))
    | _     => pretty BS (.convStridedBiasGradB (h := HH) (w := HH) zk1 zinS zc
        (.operand "%dy" zout))
  let (body, res) : String × String := go.run' 0
  -- (signature, return type) per op. The FIRST arg is the `x` slot of `forwardF32`.
  let (sig, retTy) :=
    match which with
    | "fwd" => (s!"%x: {ty [BS, IC*HIN*HIN]}, %W: {ty [OC,IC,1,1]}, %b: {ty [OC]}",
                ty [BS, OC*HH*HH])
    | "dx"  => (s!"%dy: {ty [BS, OC*HH*HH]}, %W: {ty [OC,IC,1,1]}", ty [BS, IC*HIN*HIN])
    | "dw"  => (s!"%dy: {ty [BS, OC*HH*HH]}, %x: {ty [BS, IC*HIN*HIN]}", ty [OC,IC,1,1])
    | _     => (s!"%dy: {ty [BS, OC*HH*HH]}", ty [OC])
  "module @m {\n" ++ s!"  func.func @k1_{which}({sig}) -> ({retTy}) " ++ "{\n" ++
  body ++ s!"    return {res} : {retTy}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The reference — closed form, from the buffers the device saw
-- ════════════════════════════════════════════════════════════════

/-- The negative controls, selected by `$K1_CONTROL`. Each perturbs the REFERENCE, never the
    render, so a control that leaves a gate green says that gate is blind to it — which is the
    thing worth knowing. `""` is the real check.

    * `align` — read the ODD spatial positions `(2i+1, 2j+1)` instead of the even ones. This is
      the failure a padding or stride slip would produce, and it is why the check exists at all.
    * `transpose` — mix channels with `W[c,o]` instead of `W[o,c]`. Same shape at `OC ≠ IC`? no —
      but at `1×1` the kernel is `[OC,IC]` and the transposed read is in-bounds, so nothing
      structural catches it.
    * `mean` — divide the batch-reducing gradients by `B`, the sum-vs-mean confusion §2a-quinquies
      found live in an emitter. -/
private def ctlAlign (ctl : String) : Bool := ctl == "align"
private def ctlTrans (ctl : String) : Bool := ctl == "transpose"
private def ctlMean  (ctl : String) : Bool := ctl == "mean"

/-- `x` is flat `[B, ic, 2h, 2w]`; read the even spatial position `(2i, 2j)` — `decimateIdx`. -/
private def xAt (ctl : String) (x : ByteArray) (b c i j : Nat) : Float :=
  let off := if ctlAlign ctl then 1 else 0
  F32.read x (b*(IC*HIN*HIN) + c*(HIN*HIN) + (2*i+off)*HIN + (2*j+off)).toUSize
private def xAtRaw (x : ByteArray) (b c y z : Nat) : Float :=
  F32.read x (b*(IC*HIN*HIN) + c*(HIN*HIN) + y*HIN + z).toUSize
private def dyAt (dy : ByteArray) (b o i j : Nat) : Float :=
  F32.read dy (b*(OC*HH*HH) + o*(HH*HH) + i*HH + j).toUSize
private def wAt (ctl : String) (W : ByteArray) (o c : Nat) : Float :=
  F32.read W (if ctlTrans ctl then c*OC + o else o*IC + c).toUSize

private def refFwd (ctl : String) (x W bias : ByteArray) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for b in [0:BS] do for o in [0:OC] do for i in [0:HH] do for j in [0:HH] do
    let mut s := F32.read bias o.toUSize
    for c in [0:IC] do s := s + wAt ctl W o c * xAt ctl x b c i j
    out := out.push s
  return out

private def refDx (ctl : String) (dy W : ByteArray) : Array Float := Id.run do
  let hit := if ctlAlign ctl then 1 else 0     -- which parity carries the scatter
  let mut out : Array Float := #[]
  for b in [0:BS] do for c in [0:IC] do for y in [0:HIN] do for z in [0:HIN] do
    if y % 2 != hit || z % 2 != hit then out := out.push 0.0
    else
      let mut s := 0.0
      for o in [0:OC] do s := s + wAt ctl W o c * dyAt dy b o (y/2) (z/2)
      out := out.push s
  return out

private def refDw (ctl : String) (x dy : ByteArray) : Array Float := Id.run do
  let den := if ctlMean ctl then BS.toFloat else 1.0
  let mut out : Array Float := #[]
  for o in [0:OC] do for c in [0:IC] do
    let mut s := 0.0
    for b in [0:BS] do for i in [0:HH] do for j in [0:HH] do
      s := s + xAt ctl x b c i j * dyAt dy b o i j
    out := out.push (s / den)
  return out

private def refDb (ctl : String) (dy : ByteArray) : Array Float := Id.run do
  let den := if ctlMean ctl then BS.toFloat else 1.0
  let mut out : Array Float := #[]
  for o in [0:OC] do
    let mut s := 0.0
    for b in [0:BS] do for i in [0:HH] do for j in [0:HH] do s := s + dyAt dy b o i j
    out := out.push (s / den)
  return out

-- ════════════════════════════════════════════════════════════════
-- § Drivers
-- ════════════════════════════════════════════════════════════════

/-- Print every `stablehlo.convolution` window line — the padding §2l asks to read. -/
private def windowLines (mlir : String) : List String :=
  (mlir.splitOn "\n").filter (fun l => (l.splitOn "window = {").length > 1)

private def renderProbe (K : Nat) (zk : Kernel4 OC IC K K) : IO Bool := do
  let mlir := probeModule K zk
  IO.println s!"\n══ k = {K}×{K} ══════════════════════════════════════════"
  for l in windowLines mlir do IO.println s!"  {l}"
  IO.FS.createDirAll ".lake/build"
  let path := s!".lake/build/strided_k{K}.mlir"
  IO.FS.writeFile path mlir
  let cargs ← ireeCompileArgs path s!".lake/build/strided_k{K}.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.println s!"  ⛔ iree-compile FAILED at k={K}:\n{r.stderr.take 2500}"
    pure false
  else
    IO.println s!"  ✅ iree-compile OK  ({path})"
    pure true

/-- Compare a device buffer against the closed form. Refuses a degenerate (all-~0) agreement:
    a reference that matches a zero buffer proves nothing, and three of these four ops CAN
    legitimately produce zeros (the odd positions of `dx`), so the check is on the max magnitude
    of the REFERENCE, which the odd zeros do not depress. -/
private def gate (name : String) (got : ByteArray) (want : Array Float) : IO Bool := do
  let mut maxAbs := 0.0
  let mut maxMag := 0.0
  let mut worst  := 0
  let mut nonFin := 0
  for i in [0:want.size] do
    let a := F32.read got i.toUSize
    let e := want[i]!
    if !a.isFinite then nonFin := nonFin + 1
    if e.abs > maxMag then maxMag := e.abs
    let d := (a - e).abs
    if d > maxAbs then maxAbs := d; worst := i
  let rel := if maxMag > 1e-30 then maxAbs / maxMag else 0.0
  IO.println s!"  {name}: {want.size} floats, |ref|max = {maxMag}, max abs diff = {maxAbs}, rel = {rel}"
  if nonFin > 0 then
    IO.println s!"  ⛔ {name}: {nonFin} non-finite outputs"; return false
  if maxMag < 1e-6 then
    IO.println s!"  ⛔ {name}: DEGENERATE — the reference is all ~0, so a match proves nothing"
    return false
  if rel > 1e-5 then
    IO.println (s!"  ⛔ {name}: MISMATCH at index {worst} " ++
      s!"(got {F32.read got worst.toUSize}, want {want[worst]!})")
    return false
  IO.println s!"  ✅ {name} matches the closed form (rel {rel} ≤ 1e-5)"
  return true

private def runOne (which : String) (params shapes xIn xSh : ByteArray) (nOut : Nat) :
    IO ByteArray := do
  let path := s!".lake/build/k1_{which}.mlir"
  IO.FS.writeFile path (oneOpModule which)
  -- §4: delete the .vmfb first, or a re-run silently reuses the previous op's binary.
  let vmfb := s!".lake/build/k1_{which}.vmfb"
  let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
  for p in [vmfb, s!".lake/build/k1_{which}_{target}.vmfb"] do
    if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let sess ← mkSession path
  IreeSession.forwardF32 sess s!"m.k1_{which}" params shapes xIn xSh 1 nOut.toUSize

def main : IO Unit := do
  -- ── gate 1: it renders, and the committed 3×3 renders beside it as the control ──
  let zk3 : Kernel4 OC IC 3 3 := fun _ _ _ _ => 0
  let ok3 ← renderProbe 3 zk3
  let ok1 ← renderProbe 1 zk1
  if !ok3 then throw (IO.userError "the 3×3 CONTROL failed — the harness is wrong, not the kernel")
  if !ok1 then throw (IO.userError "1×1 strided is NOT type-valid — §2l needs emitter work")

  -- ── gate 2: it computes what `den` says, on device ──
  let ctl := (← IO.getEnv "K1_CONTROL").getD ""
  IO.println s!"\n══ known answer, backend {← IreeSession.backendName} ═══════════"
  if ctl != "" then
    IO.println s!"  ⚠ K1_CONTROL={ctl} — the REFERENCE is perturbed; gates SHOULD go red"
  let x    ← F32.heInit 987654 nX.toUSize 1.0
  let W    ← F32.heInit 424242 nW.toUSize 1.0
  let bias ← F32.heInit 171717 OC.toUSize 1.0
  let dy   ← F32.heInit 313131 nY.toUSize 1.0

  let gF ← runOne "fwd" (F32.concat #[W, bias]) (packShapes #[#[OC,IC,1,1], #[OC]])
                  x (packXShape #[BS, IC*HIN*HIN]) nY
  let gX ← runOne "dx"  W (packShapes #[#[OC,IC,1,1]])
                  dy (packXShape #[BS, OC*HH*HH]) nX
  let gW ← runOne "dw"  x (packShapes #[#[BS, IC*HIN*HIN]])
                  dy (packXShape #[BS, OC*HH*HH]) nW
  let gB ← runOne "db"  .empty (packShapes #[])
                  dy (packXShape #[BS, OC*HH*HH]) OC

  let mut allOk := true
  for (nm, got, want) in [("fwd", gF, refFwd ctl x W bias), ("dx", gX, refDx ctl dy W),
                          ("dw", gW, refDw ctl x dy), ("db", gB, refDb ctl dy)] do
    if !(← gate nm got want) then allOk := false

  -- The load-bearing structural fact: `decimate ∘ conv` writes ZERO at odd input positions.
  -- A stride or alignment slip leaves them non-zero, and it would not move any norm above.
  let mut oddMax := 0.0
  for b in [0:BS] do for c in [0:IC] do for y in [0:HIN] do for z in [0:HIN] do
    if y % 2 == 1 || z % 2 == 1 then
      let v := (xAtRaw gX b c y z).abs
      if v > oddMax then oddMax := v
  IO.println s!"  dx odd-position |max| = {oddMax}   (must be 0 — the decimate signature)"
  if oddMax != 0.0 then
    IO.println "  ⛔ dx is non-zero at odd positions — this is NOT decimate ∘ conv"
    allOk := false

  IO.println ""
  if ctl != "" then
    -- A control run INVERTS the verdict: the perturbed reference must be rejected. If it is not,
    -- the harness is comparing something with itself and every green above is worthless.
    if allOk then
      throw (IO.userError s!"CONTROL {ctl} stayed GREEN — the harness is blind to it")
    IO.println s!"✅ control {ctl} fired — the gates above can go red"
    return
  if !allOk then throw (IO.userError "1×1 strided does not compute `den` — §2l needs op work")
  IO.println "✅ 1×1 strided: type-valid AND matches the closed form on all four ops"
  IO.println "   ⇒ §2l step A is re-instantiation (arguments), not emitter work"
  IO.println "   (run the controls: K1_CONTROL=align|transpose|mean — each must go red)"
