import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.VerifiedTrain

/-! # §2m ConvNeXt — can the existing ops spell a **channel** LayerNorm, and do they?

The check `planning/xla_pjrt_handoff.md` §2m puts before any of the ConvNeXt LN work, in the shape
§2l's `strided-1x1` used: settle the primitive on device before re-instantiating 21 sites around it.

**The defect being fixed.** ConvNeXt's render normalises with `.bnF` ⇒ `bnForward n ε γ β x`, one
mean and one variance over the WHOLE `C·H·W` feature map per example, with a **scalar** γ/β. The
reference is `channel_layer_norm`: `jnp.mean(x, axis=1)` over NCHW ⇒ `H·W` statistics per example,
each over the `C` channels at one spatial position, with a **per-channel** `[C]` affine. §2m first
recorded the axis as *"correct"* by matching the literal `across dimensions = [1]` against
`axis=1` — but the artifact's tensor is rank-2 `[B, C·H·W]` and the reference's is rank-4 NCHW.
Same literal, different function (§4's one-tensor-layout rule, hit a second time).

**Route A, which this probe is here to settle.** ConvNeXt's channel-LN *is* ViT's row-LN under a
transpose: view one example as `[C, S]` with `S = H·W`, transpose to `[S, C]`, and each row is one
spatial position holding its `C` channels — which is exactly what `rowLNFlat` normalises over.
Every op needed already exists and is proven, so if this composition is right the whole change
costs **no new `SHlo` op**:

    transposeF C S  →  lnRowF (γ=1, β=0)  →  rowScaleF γ:[C]  →  rowBiasF β:[C]  →  transposeF S C

Three gates, and the second is the one that matters:

1. **it emits type-valid MLIR** that XLA compiles;
2. **it computes channel-LN** — every output coordinate against the closed form
   `y[c,s] = γ[c]·(x[c,s] − μ[s]) / √(σ²[s] + ε) + β[c]`, recomputed here from the inputs, not
   from the render;
3. ⚠ **THE CONTROL: the incumbent `.bnF` chain must NOT match it.** A green gate 2 with a green
   `.bnF` would mean the probe is measuring something both paths satisfy — and would say the
   deviation §2m found does not exist. This gate is what makes the other two mean anything.

    lake build channel-ln && HIP_VISIBLE_DEVICES=0 .lake/build/bin/channel-ln
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 2      -- batch
private def C  : Nat := 4      -- channels (the axis LN must reduce over)
private def S  : Nat := 6      -- spatial positions H·W
private def EPS : String := "1.0e-6"
private def epsF : Float := 1.0e-6

private def nX : Nat := BS * (C * S)
private def zc : Vec C := fun _ => 0
private def zx : Vec (C*S) := fun _ => 0
private def zxT : Vec (S*C) := fun _ => 0

/-- **The Route-A chain.** `[C,S]ᵀ → row-LN over C → per-channel affine → transpose back.** -/
private def chainModule : String :=
  let go : StateM Proofs.StableHLO.EmitS String := do
    let (c1, t)  ← pretty BS (.transposeF (m := C) (n := S) (.operand "%x" zx))
    let (c2, n)  ← pretty BS (.lnRowF (m := S) (n := C) "%one" "%zero" EPS 0 1 0
                                (.operand t zxT))
    let (c3, sc) ← pretty BS (.rowScaleF (m := S) (n := C) "%g" zc (.operand n zxT))
    let (c4, bi) ← pretty BS (.rowBiasF (m := S) (n := C) "%bt" zc (.operand sc zxT))
    let (c5, o)  ← pretty BS (.transposeF (m := S) (n := C) (.operand bi zxT))
    pure (c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ s!"    return {o} : {ty [BS, C*S]}\n")
  let body : String := go.run' (0, [])
  "module @m {\n" ++
  s!"  func.func @chln(%x: {ty [BS, C*S]}, %g: {ty [C]}, %bt: {ty [C]}) -> {ty [BS, C*S]} " ++
  "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  body ++ "  }\n}\n"

/-- **The incumbent**, for the control: scalar-global `bnForward` over the whole `C·S` map. Its
    γ/β are the rank-0 scalars the committed ConvNeXt carries, so it is fed `%g0`/`%bt0`. -/
private def bnModule : String :=
  let go : StateM Proofs.StableHLO.EmitS String := do
    let (c1, o) ← pretty BS (.bnF "%g0" "%bt0" EPS 0 0 0 (.operand "%x" zx))
    pure (c1 ++ s!"    return {o} : {ty [BS, C*S]}\n")
  let body : String := go.run' (0, [])
  "module @m {\n" ++
  s!"  func.func @bnln(%x: {ty [BS, C*S]}, %g0: {ty []}, %bt0: {ty []}) -> {ty [BS, C*S]} " ++
  "{\n" ++ body ++ "  }\n}\n"

/-- The closed form, recomputed from the inputs — a reference implementation, not a second render.
    `y[b,c,s] = γ[c]·(x − μ[b,s])/√(σ²[b,s] + ε) + β[c]`, μ/σ² over the `C` axis. -/
private def channelLNRef (x g bt : ByteArray) : IO ByteArray := do
  let mut cells : Array ByteArray := #[]
  for b in [0:BS] do
    for c in [0:C] do
      for s in [0:S] do
        let mut μ := 0.0
        for c' in [0:C] do μ := μ + F32.read x (b*(C*S) + c'*S + s).toUSize
        μ := μ / C.toFloat
        let mut v := 0.0
        for c' in [0:C] do
          let d := F32.read x (b*(C*S) + c'*S + s).toUSize - μ
          v := v + d*d
        v := v / C.toFloat
        let xh := (F32.read x (b*(C*S) + c*S + s).toUSize - μ) / Float.sqrt (v + epsF)
        cells := cells.push (← F32.const 1 (F32.read g c.toUSize * xh + F32.read bt c.toUSize))
  pure (F32.concat cells)

/-- **The Route-A BACKWARD**, one op per module so each can be driven through the single-output
    invoke. Same wrapping as the forward: transpose in, ViT's proven LN-backward triple, transpose
    the input cotangent back out. `which ∈ dx | dg | db`.

    This is the half that decides the estimate. The forward composing correctly would still leave
    Route A needing new ops if the γ/β gradients or the input VJP had no row-layout peer — they do:
    `rowDenseBiasGrad` (dβ = Σ dy, and it contracts the BATCH as well as the rows, `dims = [0,1]`),
    `veclnGammaGrad` (dγ = Σ dy⊙x̂ recomputed from the saved LN input) and `lnRowBack` are all
    generic in their dims, ViT just instantiates them at 197×192. -/
private def backModule (which : String) : String :=
  let go : StateM Proofs.StableHLO.EmitS (String × String) := do
    let (cxT,  xT)  ← pretty BS (.transposeF (m := C) (n := S) (.operand "%x" zx))
    let (cdyT, dyT) ← pretty BS (.transposeF (m := C) (n := S) (.operand "%dy" zx))
    match which with
    | "db" => do
        let (c, o) ← pretty BS (.rowDenseBiasGrad (N := S) (c := C) (.operand dyT zxT))
        pure (cdyT ++ c, o)
    | "dg" => do
        let (c, o) ← pretty BS (.veclnGammaGrad (N := S) (D := C) xT EPS 0 zxT
                                  (.operand dyT zxT))
        pure (cxT ++ cdyT ++ c, o)
    | _ => do
        let (cs, da)  ← pretty BS (.rowScaleF (m := S) (n := C) "%g" zc (.operand dyT zxT))
        let (cn, dxT) ← pretty BS (.lnRowBack (m := S) (n := C) "%one" xT EPS 0 1 zxT
                                     (.operand da zxT))
        let (ct, o)   ← pretty BS (.transposeF (m := S) (n := C) (.operand dxT zxT))
        pure (cxT ++ cdyT ++ cs ++ cn ++ ct, o)
  let (body, res) : String × String := go.run' (0, [])
  let retTy := if which == "dx" then ty [BS, C*S] else ty [C]
  let sig := s!"%x: {ty [BS, C*S]}, %g: {ty [C]}, %dy: {ty [BS, C*S]}"
  "module @m {\n" ++
  s!"  func.func @chln_{which}({sig}) -> {retTy} " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  body ++ s!"    return {res} : {retTy}\n" ++ "  }\n}\n"

/-- Closed-form channel-LN backward: `dβ[c] = Σ_{b,s} dy`, `dγ[c] = Σ_{b,s} dy·x̂`, and
    `dx[c,s] = (g − mean_c g − x̂·mean_c(g⊙x̂)) / σ_s` with `g = γ⊙dy` — all recomputed from the
    inputs. Returns `(dx, dγ, dβ)`. -/
private def channelLNBackRef (x g dy : ByteArray) : IO (ByteArray × ByteArray × ByteArray) := do
  let mut dxCells : Array ByteArray := #[]
  let mut dgAcc : Array Float := Array.replicate C 0.0
  let mut dbAcc : Array Float := Array.replicate C 0.0
  -- pass 1: per (b,s) statistics, then dx and the parameter accumulators
  for b in [0:BS] do
    for c in [0:C] do
      for s in [0:S] do
        let idx := fun cc => (b*(C*S) + cc*S + s).toUSize
        let mut μ := 0.0
        for c' in [0:C] do μ := μ + F32.read x (idx c')
        μ := μ / C.toFloat
        let mut v := 0.0
        for c' in [0:C] do
          let d := F32.read x (idx c') - μ
          v := v + d*d
        v := v / C.toFloat
        let σ := Float.sqrt (v + epsF)
        let xh := fun cc => (F32.read x (idx cc) - μ) / σ
        let gg := fun cc => F32.read g cc.toUSize * F32.read dy (idx cc)
        let mut mg := 0.0
        let mut mgx := 0.0
        for c' in [0:C] do
          mg := mg + gg c'
          mgx := mgx + gg c' * xh c'
        mg := mg / C.toFloat
        mgx := mgx / C.toFloat
        dxCells := dxCells.push (← F32.const 1 ((gg c - mg - xh c * mgx) / σ))
        dgAcc := dgAcc.set! c (dgAcc[c]! + F32.read dy (idx c) * xh c)
        dbAcc := dbAcc.set! c (dbAcc[c]! + F32.read dy (idx c))
  let mut dgCells : Array ByteArray := #[]
  let mut dbCells : Array ByteArray := #[]
  for c in [0:C] do
    dgCells := dgCells.push (← F32.const 1 dgAcc[c]!)
    dbCells := dbCells.push (← F32.const 1 dbAcc[c]!)
  pure (F32.concat dxCells, F32.concat dgCells, F32.concat dbCells)

private def maxRel (a b : ByteArray) (n : Nat) : Float × Float := Id.run do
  let mut d := 0.0
  let mut m := 0.0
  for i in [0:n] do
    let u := F32.read a i.toUSize
    let w := F32.read b i.toUSize
    if (u-w).abs > d then d := (u-w).abs
    if u.abs > m then m := u.abs
  (d, m)

-- ════════════════════════════════════════════════════════════════
-- § `--bench` — what the transposes actually cost, at ConvNeXt-T's real shapes
-- ════════════════════════════════════════════════════════════════

/-- The four LN shapes ConvNeXt-T actually has, with their real site counts, read off the committed
    `convnext_fwd.mlir` (flat lengths 301056×4, 150528×4, 75264×10, 37632×3, plus the head 768×1).
    `(C, S, sites)`. The head is excluded: it is already correct and needs no transpose. -/
private def cnxSites : List (Nat × Nat × Nat) :=
  [(96, 3136, 4), (192, 784, 4), (384, 196, 10), (768, 49, 3)]

private def benchBS : Nat := 32   -- the batch the ConvNeXt artifacts are rendered at

/-- `reps` chained LN sites at one `(C,S)`, either the Route-A chain or the incumbent `.bnF`.
    Chaining is what keeps the measurement off the h2d/d2h path: one invoke, `reps` sites. -/
private def benchModule (C' S' reps : Nat) (routeA : Bool) : String :=
  let zx' : Vec (C'*S') := fun _ => 0
  let zxT' : Vec (S'*C') := fun _ => 0
  let zc' : Vec C' := fun _ => 0
  let go : StateM Proofs.StableHLO.EmitS String := do
    let mut code := ""
    let mut cur := "%x"
    for _ in [0:reps] do
      if routeA then
        let (c1, t)  ← pretty benchBS (.transposeF (m := C') (n := S') (.operand cur zx'))
        let (c2, n)  ← pretty benchBS (.lnRowF (m := S') (n := C') "%one" "%zero" EPS 0 1 0
                                          (.operand t zxT'))
        let (c3, sc) ← pretty benchBS (.rowScaleF (m := S') (n := C') "%g" zc' (.operand n zxT'))
        let (c4, bi) ← pretty benchBS (.rowBiasF (m := S') (n := C') "%bt" zc' (.operand sc zxT'))
        let (c5, o)  ← pretty benchBS (.transposeF (m := S') (n := C') (.operand bi zxT'))
        code := code ++ c1 ++ c2 ++ c3 ++ c4 ++ c5; cur := o
      else
        let (c1, o) ← pretty benchBS (.bnF "%g0" "%bt0" EPS 0 0 0 (.operand cur zx'))
        code := code ++ c1; cur := o
    pure (code ++ s!"    return {cur} : {ty [benchBS, C'*S']}\n")
  let body : String := go.run' (0, [])
  let sig := if routeA then s!"%x: {ty [benchBS, C'*S']}, %g: {ty [C']}, %bt: {ty [C']}"
             else s!"%x: {ty [benchBS, C'*S']}, %g0: {ty []}, %bt0: {ty []}"
  "module @m {\n" ++
  s!"  func.func @lnbench({sig}) -> {ty [benchBS, C'*S']} " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  body ++ "  }\n}\n"

private def stats (xs : Array Float) : Float × Float :=
  let s := xs.qsort (· < ·)
  (s[0]!, s[s.size / 2]!)

/-- ⚠ **What this measures and what it does not.** It is the FORWARD LN cost of the whole net at
    the real shapes and the real site counts, so no extrapolation is involved — but the sites are
    chained back-to-back here, where in the real graph each sits between a depthwise conv and a
    1×1 conv. XLA fuses across op boundaries, so the isolated number is an **upper bound** on the
    delta: in the real graph a transpose can be folded into the consumer's layout, and here it
    cannot be folded into anything but another LN.

    `.bnF` is the stand-in for **Route B**: both are reduce → normalize → affine with no transpose,
    so `A − bnF` is roughly what a fused rank-4 channel-LN op would save. It is a proxy, not
    Route B itself — Route B does not exist to be timed. -/
private def benchMode : IO Unit := do
  IO.println "§2m Route A — what the transposes cost, at ConvNeXt-T's real LN shapes"
  IO.println s!"  B {benchBS}, forward only, one invoke per (C,S) carrying that stage's REAL site count"
  IO.println "  (C, S)          sites   A ms   bnF ms   Δ ms   Δ/site"
  let rounds := 12
  let warm := 3
  let mut totA := 0.0
  let mut totB := 0.0
  for (C', S', reps) in cnxSites do
    let n := benchBS * (C' * S')
    let x  ← F32.heInit 4242 n.toUSize 1.0
    let g  ← F32.scaleShift (← F32.heInit 77 C'.toUSize 0.4) 1.0 1.0
    let bt ← F32.heInit 99 C'.toUSize 0.3
    let g0  ← F32.const 1 1.0
    let bt0 ← F32.const 1 0.0
    let time (routeA : Bool) : IO Float := do
      let tag := if routeA then "a" else "b"
      let path := s!"/tmp/chln_bench_{C'}_{tag}.mlir"
      IO.FS.writeFile path (benchModule C' S' reps routeA)
      let vmfb := s!".lake/build/chln_bench_{C'}_{tag}.vmfb"
      let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
      for p in [vmfb, s!".lake/build/chln_bench_{C'}_{tag}_{target}.vmfb"] do
        if ← System.FilePath.pathExists p then IO.FS.removeFile p
      let sess ← mkSession path
      let params := if routeA then F32.concat #[g, bt] else F32.concat #[g0, bt0]
      let shapes := if routeA then packShapes #[#[C'], #[C']] else packShapes #[#[], #[]]
      -- ⚠ `IO.monoMsNow` is INTEGER milliseconds, and one invoke of the small stages lands at
      -- 1-3 ms — a resolution comparable to the quantity being measured, which is the §2j lesson
      -- in a new place. Time `inner` invokes per sample and divide, so the tick is 1/inner ms.
      let inner := 20
      let one : IO Float := do
        let t0 ← IO.monoMsNow
        for _ in [0:inner] do
          let o ← LowererSession.forwardF32 sess "m.lnbench" params shapes x
                    (packXShape #[benchBS, C'*S']) benchBS.toUSize (C'*S').toUSize
          if o.size == 0 then throw (IO.userError "empty bench output")
        let t1 ← IO.monoMsNow
        pure ((t1 - t0).toFloat / inner.toFloat)
      for _ in [0:warm] do let _ ← one
      let mut xs : Array Float := #[]
      for _ in [0:rounds] do xs := xs.push (← one)
      pure (stats xs).2      -- median; §2j: one sample is not an anchor
    let a ← time true
    let b ← time false
    totA := totA + a; totB := totB + b
    IO.println s!"  ({C'}, {S'})   sites {reps}   A {a}   bnF {b}   Δ {a - b}   Δ/site {(a - b) / reps.toFloat}"
  IO.println ""
  IO.println s!"  whole-net FORWARD LN: Route A {totA} ms, `.bnF` {totB} ms, Δ {totA - totB} ms"
  IO.println s!"  ⇒ fwd+bwd ≈ 2× that ⇒ ~{2.0 * (totA - totB)} ms/step of added LN cost"
  IO.println "  Compare against ConvNeXt's measured step: 84.5 s/epoch over 295 steps ≈ 286 ms/step \
(§2h, train+eval), so read the Δ as a fraction of that — and remember it is an UPPER bound (the \
sites are chained here, so no transpose can fuse into a conv consumer)."

def main (args : List String) : IO Unit := do
  if args.contains "--bench" then benchMode; return
  IO.println "§2m ConvNeXt — does transpose ∘ row-LN ∘ per-channel affine = channel LayerNorm?"
  IO.println s!"  B {BS}, C {C}, S {S} (a site is Vec (C·S) = {C*S}), backend {← LowererSession.backendName}"
  -- inputs: x varies across BOTH axes (a constant channel would hide an axis mix-up), γ/β non-trivial
  let x  ← F32.heInit 4242 nX.toUSize 1.0
  let g  ← F32.scaleShift (← F32.heInit 77 C.toUSize 0.4) 1.0 1.0    -- γ ≈ 1 ± 0.4
  let bt ← F32.heInit 99 C.toUSize 0.3                                -- β centred noise
  let ref ← channelLNRef x g bt
  let run (src tag : String) (params : ByteArray) (shapes : ByteArray) (fn : String) : IO ByteArray := do
    let path := s!"/tmp/chln_{tag}.mlir"
    IO.FS.writeFile path src
    let vmfb := s!".lake/build/chln_{tag}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/chln_{tag}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession path
    LowererSession.forwardF32 sess s!"m.{fn}" params shapes x (packXShape #[BS, C*S])
      BS.toUSize (C*S).toUSize
  -- gate 1+2: the chain compiles, and it computes the closed form
  let yA ← run chainModule "chain" (F32.concat #[g, bt]) (packShapes #[#[C], #[C]]) "chln"
  let (dA, mA) := maxRel yA ref nX
  IO.println s!"  chain vs closed form : max abs {dA}   (|ref|max {mA}, rel {dA / mA})"
  -- gate 3, THE CONTROL: the incumbent scalar-global path must disagree
  let g0  ← F32.const 1 1.0
  let bt0 ← F32.const 1 0.0
  let yB ← run bnModule "bn" (F32.concat #[g0, bt0]) (packShapes #[#[], #[]]) "bnln"
  let (dB, _) := maxRel yB ref nX
  IO.println s!"  incumbent `.bnF` vs closed form : max abs {dB}   (rel {dB / mA})  ← control"
  if mA < 1e-6 then throw (IO.userError "DEGENERATE: the reference is ~0 — the probe proves nothing")
  if dB / mA < 1e-3 then
    throw (IO.userError s!"CONTROL FAILED: the incumbent scalar-global `.bnF` also matches channel-LN \
(rel {dB / mA}). Either the probe is not measuring the axis, or §2m's deviation does not exist — \
either way gate 2 means nothing.")
  if dA / mA > 1e-5 then
    IO.println s!"  ⛔ the chain does NOT compute channel-LN (rel {dA / mA})"
    throw (IO.userError "route A does not spell channel LayerNorm")
  -- ── gate 4: the BACKWARD, which is what decides whether Route A needs new ops ──
  IO.println "  ── backward ──"
  let dy ← F32.heInit 31337 nX.toUSize 1.0
  let (rdx, rdg, rdb) ← channelLNBackRef x g dy
  let runB (which : String) (nOut : Nat) : IO ByteArray := do
    let path := s!"/tmp/chln_b{which}.mlir"
    IO.FS.writeFile path (backModule which)
    let vmfb := s!".lake/build/chln_b{which}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/chln_b{which}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession path
    -- `%x` rides the x slot; `%g` and `%dy` ride the packed params, in signature order.
    LowererSession.forwardF32 sess s!"m.chln_{which}" (F32.concat #[g, dy])
      (packShapes #[#[C], #[BS, C*S]]) x (packXShape #[BS, C*S])
      (if which == "dx" then BS else 1).toUSize (if which == "dx" then C*S else nOut).toUSize
  let ydx ← runB "dx" nX
  let ydg ← runB "dg" C
  let ydb ← runB "db" C
  let (ddx, mdx) := maxRel ydx rdx nX
  let (ddg, mdg) := maxRel ydg rdg C
  let (ddb, mdb) := maxRel ydb rdb C
  IO.println s!"  dx vs closed form : max abs {ddx}   (|ref|max {mdx}, rel {ddx / mdx})"
  IO.println s!"  dγ vs closed form : max abs {ddg}   (|ref|max {mdg}, rel {ddg / mdg})"
  IO.println s!"  dβ vs closed form : max abs {ddb}   (|ref|max {mdb}, rel {ddb / mdb})"
  if mdx < 1e-6 || mdg < 1e-6 || mdb < 1e-6 then
    throw (IO.userError "DEGENERATE: a reference gradient is ~0 — that gate proves nothing")
  for (nm, d, m) in [("dx", ddx, mdx), ("dγ", ddg, mdg), ("dβ", ddb, mdb)] do
    if d / m > 1e-4 then
      IO.println s!"  ⛔ {nm} does NOT match the closed form (rel {d / m})"
      throw (IO.userError s!"route A's backward is wrong at {nm}")
  IO.println s!"  ✅ the BACKWARD composes too — dx {ddx / mdx}, dγ {ddg / mdg}, dβ {ddb / mdb}, \
all from `lnRowBack` / `veclnGammaGrad` / `rowDenseBiasGrad` wrapped in `transposeF`. \
**Route A needs no new `SHlo` op in either direction.**"
  IO.println s!"  ✅ transposeF ∘ lnRowF ∘ rowScaleF ∘ rowBiasF ∘ transposeF IS channel LayerNorm \
(rel {dA / mA}), while the incumbent `.bnF` is off by rel {dB / mA} — so the composition is right \
AND the deviation §2m found is real. No new `SHlo` op is needed."
