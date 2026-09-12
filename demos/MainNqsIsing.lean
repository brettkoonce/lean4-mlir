import LeanMlir

/-! Neural quantum states on the transverse-field Ising chain —
    `planning/transformer_wavefunction_demo.md`.

    The network is the wavefunction. H = -J Σ σᶻᵢσᶻᵢ₊₁ - h Σ σˣᵢ on a periodic
    chain of N spins; the ground state has positive amplitudes in the z basis,
    so log ψ is one real number per configuration and the whole thing rides on
    the f32 pipeline. The energy E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ is minimised by Adam through
    the stack with ZERO new codegen:

    ⭐ **Structure first, the network models the rest** (§1). The ansatz is
    ψ_θ(σ) = ψ_ref(σ) · exp f_θ(σ), where ψ_ref is the mean-field product state
    at the optimal angle (a closed form: log ψ_ref is LINEAR in σ) and f_θ is the
    network. f_θ = 0 is the floor row of the table and the network only has to
    model what mean field gets wrong. The host adds log ψ_ref; the network never
    sees it. `noref` drops it (uniform start) — the ablation of Table 2.

    ⭐ **The gradient is one host weight per configuration** (§4).
    ∂E/∂θ = 2 Σ_s p_s (E_loc(s) - E) ∂_θ log ψ(s). The train step is the rank-2
    DDPM MSE block, whose gradient on the output is 2(out - y)/(M·nOut); setting
    y = out - M·w/2 makes the block's output cotangent exactly w. The blackjack
    DQN did the same trick with a Bellman target; here the "target" is the
    model's own local energies.

    Three ansätze, one exe:
      mlp  `.dense N 64 .relu, .dense 64 64 .relu, .dense 64 1` on σ ∈ {±1}^N
      vit  patches of p spins as token ids → `tokenPositionEmbed` →
           `transformerEncoder (keepSequence)` → mean over tokens → `.dense d 1`
      gpt  the same tokens with a BOS, `causalMask`, `lmHead`: |ψ|² = Π_k p(patch_k | <k),
           log ψ = ½ Σ log p, the reference entering as a fixed bias on the logits
           so normalisation survives (§1). Sampling is exact and independent —
           the TinyGPT sampler batched over B chains, no Metropolis anywhere.

    Samples: enumeration at N ≤ 14 (energy and gradient exact, no Monte Carlo
    noise); above that, Metropolis single-spin flips for mlp/vit (B parallel
    chains, acceptance from ψ² ratios off the batched eval graph) and
    autoregressive draws for gpt. E_loc needs ψ at the N single-flip neighbours
    of every sample: one forward at batch B·N.

    XLA backend only. `lake exe nqs-ising <mlp|vit|gpt> [N=12] [h=1.0] [steps=1000]
    [seed=1] [p=2] [d=32] [heads=2] [blocks=2] [hidden=64] [B=1024] [sweeps=2]
    [lr=0.001] [cosine] [evalb=8] [tag=x] [noref] [symref] [check]`. Writes `<prefix>_metrics.json`,
    `_curve.csv`, `_samples.bin`, `_params.bin` under `.lake/build/`;
    `scripts/nqs_metrics.py score` brackets the JSON against enumeration
    (N ≤ 14) or Jordan-Wigner (any even N). -/

namespace NQS

def piF : Float := 3.14159265358979323846

/-- Fixed-point decimal; `Float.toString` prints 17 digits. -/
def fmt (x : Float) (d : Nat) : String :=
  let m := Float.pow 10.0 d.toFloat
  let y := Float.round (x * m)
  let neg := y < 0.0
  let yi := (Float.abs y).toUInt64.toNat
  let ip := yi / (10 ^ d)
  let fp := yi % (10 ^ d)
  let fs := toString fp
  let fs := String.ofList (List.replicate (d - fs.length) '0') ++ fs
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

/-- Scientific notation with three significant digits, for the variances. -/
def sci (x : Float) : String :=
  if x == 0.0 then "0" else
  let e := Float.floor (Float.log10 (Float.abs x))
  let m := x / Float.pow 10.0 e
  let es := if e < 0.0 then "-" ++ toString (Float.abs e).toUInt64.toNat
            else "+" ++ toString e.toUInt64.toNat
  s!"{fmt m 2}e{es}"

@[inline] def pushF32 (acc : ByteArray) (x : Float) : ByteArray :=
  let u : UInt32 := x.toFloat32.toBits
  (((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8).push ((u >>> 24) &&& 0xff).toUInt8

@[inline] def pushF64 (acc : ByteArray) (x : Float) : ByteArray := Id.run do
  let u := x.toBits
  let mut acc := acc
  for j in [0:8] do acc := acc.push ((u >>> (8 * j).toUInt64) &&& 0xff).toUInt8
  return acc

structure Cfg where
  arch   : String := "mlp"
  model  : String := "ising"   -- "ising" (rungs 1–3) | "j1j2" (rung 4, the sign-structure rung)
  J2     : Float := 0.0        -- j1j2: the next-nearest coupling, in units of J
  N      : Nat := 12
  h      : Float := 1.0
  J      : Float := 1.0
  steps  : Nat := 1000
  seed   : Nat := 1
  useRef : Bool := true
  symRef : Bool := false  -- Z2-symmetrised reference ψ_MF(σ) + ψ_MF(−σ), mlp/vit only
  p      : Nat := 2       -- spins per token (vit/gpt)
  d      : Nat := 32
  heads  : Nat := 2
  blocks : Nat := 2
  hidden : Nat := 64      -- mlp width
  B      : Nat := 1024    -- chains / autoregressive batch when N > 14
  sweeps : Nat := 2       -- Metropolis sweeps between training steps
  burn   : Nat := 20      -- Metropolis burn-in sweeps
  lr     : Float := 0.001
  cosine : Bool := false  -- cosine decay of lr to 5% of peak over the run
  check  : Bool := false  -- N ≤ 14: draw samples with the arch's sampler, compare to enumeration (§8)
  cseed  : Nat := 0       -- the check's own draw seed (0 = continue from `seed`)
  evalb  : Nat := 8       -- sample batches in the final measurement (N > 14)
  logEvery : Nat := 50
  tag    : String := ""

def Cfg.T (c : Cfg) : Nat := c.N / c.p
def Cfg.V (c : Cfg) : Nat := 1 <<< c.p
def Cfg.enumerate (c : Cfg) : Bool := c.N <= 14
/-- The training batch: every configuration when enumerating, else the chains. -/
def Cfg.M (c : Cfg) : Nat := if c.enumerate then 1 <<< c.N else c.B
/-- Output slots: the GPT's logits, the two-slot (log|ψ|, φ) head of the J1-J2 rung,
    or the single log-amplitude. -/
def Cfg.nOut (c : Cfg) : Nat :=
  if c.arch == "gpt" then c.T * c.V else if c.model == "j1j2" then 2 else 1

/-- The mean-field product state |φ*⟩^N: per-site energy −J cos²φ − h sin φ,
    minimised at sin φ = h/2J (h < 2J) and φ = π/2 beyond (the uniform state).
    Amplitude cos(φ/2) for σ = +1 and sin(φ/2) for −1, so log ψ_ref is a fixed
    visible bias, linear in σ. -/
structure Ref where
  phi   : Float
  logUp : Float
  logDn : Float
  eSite : Float

def mkRef (J h : Float) : Ref :=
  let phi := if h < 2.0 * J then Float.asin (h / (2.0 * J)) else piF / 2.0
  let c := Float.cos phi
  let s := Float.sin phi
  { phi, logUp := Float.log (Float.cos (phi / 2.0)), logDn := Float.log (Float.sin (phi / 2.0)),
    eSite := -J * c * c - h * s }

@[inline] def spin (c : UInt64) (i : Nat) : Float :=
  if (c >>> i.toUInt64) &&& 1 == 1 then 1.0 else -1.0

def popcnt (c : UInt64) (N : Nat) : Nat := Id.run do
  let mut n := 0
  for i in [0:N] do
    if (c >>> i.toUInt64) &&& 1 == 1 then n := n + 1
  return n

/-- log ψ_ref(σ), or 0 for the uniform start. With `symRef` the reference is the
    Z2-symmetrised product state ψ_MF(σ) + ψ_MF(−σ): the finite-N ground state is
    even under the global flip, and a broken reference leaves that to the network. -/
def refLogOf (cfg : Cfg) (ref : Ref) (nUp : Nat) : Float :=
  if !cfg.useRef then 0.0 else
  let a := nUp.toFloat * ref.logUp + (cfg.N - nUp).toFloat * ref.logDn
  if !cfg.symRef then a else
  let b := (cfg.N - nUp).toFloat * ref.logUp + nUp.toFloat * ref.logDn
  max a b + Float.log (1.0 + Float.exp (-(Float.abs (a - b))))

def refLog (cfg : Cfg) (ref : Ref) (c : UInt64) : Float := refLogOf cfg ref (popcnt c cfg.N)

/-- The diagonal part of H: −J Σ σᵢσᵢ₊₁ around the ring. -/
def eDiag (cfg : Cfg) (c : UInt64) : Float := Id.run do
  let mut s := 0.0
  for i in [0:cfg.N] do
    s := s + spin c i * spin c ((i + 1) % cfg.N)
  return -cfg.J * s

@[inline] def patchId (cfg : Cfg) (c : UInt64) (k : Nat) : Nat :=
  let mask : UInt64 := ((1 : UInt64) <<< cfg.p.toUInt64) - 1
  ((c >>> (k * cfg.p).toUInt64) &&& mask).toNat

/-- The reference as a per-patch log-probability, the GPT's fixed logit bias:
    log p_ref(patch) = Σ_j 2·log(amplitude of spin j). Zero for the uniform start. -/
def patchBias (cfg : Cfg) (ref : Ref) : Array Float := Id.run do
  let mut b : Array Float := Array.mkEmpty cfg.V
  for v in [0:cfg.V] do
    let mut acc := 0.0
    if cfg.useRef then
      for j in [0:cfg.p] do
        acc := acc + (if (v >>> j) &&& 1 == 1 then 2.0 * ref.logUp else 2.0 * ref.logDn)
    b := b.push acc
  return b

/-- 0 mlp (±1 floats), 1 vit (patch ids), 2 gpt (BOS then the first T−1 patch ids):
    the input layout `lean_nqs_inputs` writes. -/
def Cfg.mode (c : Cfg) : Nat := if c.arch == "mlp" then 0 else if c.arch == "vit" then 1 else 2

/-! Configurations cross into C as `[rows]` little-endian u64 (`cfgsBA`), and the
    hot paths — network inputs, up-spin counts, patch ids, the categorical draw —
    are `ffi/f32_helpers.c` helpers keyed by a `flipMode`: 0 the rows as given,
    1 row r with site `sites[r]` flipped (a Metropolis proposal per chain), 2 every
    row expanded into its N single-flip neighbours (the E_loc batch). In Lean the
    same loops cost ~1.5 µs per pushed float: 11 s per step for the MLP at N = 64. -/

@[inline] def pushU64 (acc : ByteArray) (c : UInt64) : ByteArray := Id.run do
  let mut acc := acc
  for j in [0:8] do acc := acc.push ((c >>> (8 * j).toUInt64) &&& 0xff).toUInt8
  return acc

def cfgsBA (cs : Array UInt64) : ByteArray :=
  cs.foldl pushU64 (ByteArray.emptyWithCapacity (cs.size * 8))

def setU64 (ba : ByteArray) (s : Nat) (c : UInt64) : ByteArray := Id.run do
  let mut ba := ba
  for j in [0:8] do ba := ba.set! (8 * s + j) ((c >>> (8 * j).toUInt64) &&& 0xff).toUInt8
  return ba

@[extern "lean_nqs_inputs"]
opaque nqsInputs (cfgs : @& ByteArray) (sites : @& ByteArray)
    (rows N p T mode flipMode : USize) : IO ByteArray
@[extern "lean_nqs_popcount"]
opaque nqsPopcount (cfgs : @& ByteArray) (sites : @& ByteArray)
    (rows N flipMode : USize) : IO ByteArray
@[extern "lean_nqs_patch_ids"]
opaque nqsPatchIds (cfgs : @& ByteArray) (sites : @& ByteArray)
    (rows N p T flipMode : USize) : IO ByteArray
@[extern "lean_nqs_gpt_draw"]
opaque nqsGptDraw (out : @& ByteArray) (bias : @& ByteArray)
    (B T V k seed : USize) : IO ByteArray

def inputsOf (cfg : Cfg) (cfgs sites : ByteArray) (rows flipMode : Nat) : IO ByteArray :=
  nqsInputs cfgs sites rows.toUSize cfg.N.toUSize cfg.p.toUSize cfg.T.toUSize
    cfg.mode.toUSize flipMode.toUSize

/-- log ψ of the GPT ansatz for every row of an eval output, in C
    ([ffi/f32_helpers.c](https://github.com/brettkoonce/lean4-mlir/blob/main/ffi/f32_helpers.c)
    `lean_nqs_gpt_logpsi`): ½ Σ_k log softmax(logits_k + bias)[id_k] per row.
    `ids` is [rows, T] u8. The Lean loop it replaces read every logit twice
    through `F32.read`, which at batch B·N = 65536 was the whole step. -/
@[extern "lean_nqs_gpt_logpsi"]
opaque gptLogPsiC (out : @& ByteArray) (ids : @& ByteArray) (bias : @& ByteArray)
    (rows T V : USize) : IO ByteArray

/-- log ψ_θ for every row the eval output `out` describes — the configurations
    `cfgs` under `flipMode` (row r of `out` is row r of that expansion),
    reference included. mlp/vit: log ψ_ref from the up-spin count (double
    precision on the host: at N = 64 and small h the reference is O(−200) and
    f32 would lose the 1e-5 we report) plus the network's scalar; gpt: the
    conditionals through `gptLogPsiC`. -/
def logPsiRows (cfg : Cfg) (ref : Ref) (biasBA : ByteArray) (out : ByteArray)
    (cfgs sites : ByteArray) (rows flipMode : Nat) : IO (Array Float) := do
  let outRows := if flipMode == 2 then rows * cfg.N else rows
  if cfg.arch != "gpt" then
    let pc ← nqsPopcount cfgs sites rows.toUSize cfg.N.toUSize flipMode.toUSize
    return (Array.range outRows).map fun s =>
      refLogOf cfg ref (pc.get! s).toNat + F32.read out s.toUSize
  else
    let ids ← nqsPatchIds cfgs sites rows.toUSize cfg.N.toUSize cfg.p.toUSize cfg.T.toUSize
                flipMode.toUSize
    let lp ← gptLogPsiC out ids biasBA outRows.toUSize cfg.T.toUSize cfg.V.toUSize
    return (Array.range outRows).map fun s => F32.read lp s.toUSize

/-- The MSE block's target that makes its output cotangent exactly `w`
    (§4): y = out − M·nOut·g/2 with g the cotangent on each output entry. For
    mlp/vit g_s = w_s; for the GPT the host chains the softmax Jacobian,
    g_{s,k,v} = w_s · ½ (δ_{v,patch_k} − q_{s,k,v}). -/
def targets (cfg : Cfg) (bias : Array Float) (out : ByteArray) (cs : Array UInt64)
    (w : Array Float) : ByteArray := Id.run do
  let M := cs.size
  let mut y : ByteArray := ByteArray.emptyWithCapacity (M * cfg.nOut * 4)
  if cfg.arch != "gpt" then
    for s in [0:M] do
      y := pushF32 y (F32.read out s.toUSize - M.toFloat * w[s]! / 2.0)
  else
    let T := cfg.T
    let V := cfg.V
    let scale := (M * T * V).toFloat / 2.0
    -- ⚠ the eval graph returns logits in (t, v) order but the TRAIN forward's
    -- lmHead output is [B, V, T, 1] (the per-pixel-CE layout, vocab first), so
    -- the target is written vocab-major.
    for s in [0:M] do
      let mut row : Array Float := Array.replicate (T * V) 0.0
      for k in [0:T] do
        let base := s * T * V + k * V
        let id := patchId cfg cs[s]! k
        let mut mx := -1.0e30
        for v in [0:V] do
          let z := F32.read out (base + v).toUSize + bias[v]!
          if z > mx then mx := z
        let mut se := 0.0
        for v in [0:V] do
          se := se + Float.exp (F32.read out (base + v).toUSize + bias[v]! - mx)
        for v in [0:V] do
          let l := F32.read out (base + v).toUSize
          let q := Float.exp (l + bias[v]! - mx) / se
          let g := w[s]! * 0.5 * ((if v == id then 1.0 else 0.0) - q)
          row := row.set! (v * T + k) (l - scale * g)
      for j in [0:T * V] do y := pushF32 y row[j]!
  return y

def mkSpec (cfg : Cfg) : NetSpec :=
  let name := s!"nqs-{cfg.model}-{cfg.arch}-n{cfg.N}" ++ (if cfg.tag == "" then "" else "-" ++ cfg.tag)
  if cfg.arch == "mlp" then
    { name, imageH := 1, imageW := 1, layers := [
        .dense cfg.N cfg.hidden .relu,
        .dense cfg.hidden cfg.hidden .relu,
        .dense cfg.hidden cfg.nOut .identity ] }
  else if cfg.arch == "vit" then
    { name, imageH := 1, imageW := 1, layers := [
        .tokenPositionEmbed cfg.V cfg.T cfg.d (idsInput := true),
        .transformerEncoder cfg.d cfg.heads (4 * cfg.d) cfg.blocks (keepSequence := true),
        .spatialUnflatten cfg.d cfg.T 1,
        .globalAvgPool,
        .dense cfg.d cfg.nOut .identity ] }
  else
    { name, imageH := 1, imageW := 1, layers := [
        .tokenPositionEmbed (cfg.V + 1) cfg.T cfg.d (idsInput := true),
        .transformerEncoder cfg.d cfg.heads (4 * cfg.d) cfg.blocks (causalMask := true),
        .lmHead cfg.d cfg.V cfg.T ] }

/-- Size of the head's parameters (W and b of the last layer), zeroed at init so
    f_θ = 0 exactly and the first step IS the reference (§8's R1 gate). -/
def headParams (cfg : Cfg) : Nat :=
  if cfg.arch == "mlp" then cfg.hidden * cfg.nOut + cfg.nOut
  else if cfg.arch == "vit" then cfg.d * cfg.nOut + cfg.nOut
  else cfg.d * cfg.V + cfg.V

/-- Everything the loop needs to run the eval graphs. -/
structure Net where
  cfg : Cfg
  ref : Ref
  bias : Array Float
  biasBA : ByteArray
  spec : NetSpec
  evalSess : LowererSession      -- batch M (all configs, or the chains)
  flipSess : Option LowererSession  -- batch B·N for E_loc when sampling
  evalShapes : ByteArray
  xShM : ByteArray
  xShBN : ByteArray

def Net.forward (net : Net) (sess : LowererSession) (params : ByteArray) (x : ByteArray)
    (xSh : ByteArray) (batch : Nat) : IO ByteArray :=
  LowererSession.forwardF32 sess net.spec.evalFnName params net.evalShapes x xSh
    batch.toUSize net.cfg.nOut.toUSize

/-- Local energies of sampled configurations: one forward at batch B·N on the
    single-flip neighbours. Returns (E_loc, Σᵢ ψ(flipᵢ s)/ψ(s)) per sample. -/
def Net.localEnergies (net : Net) (params : ByteArray) (cs : Array UInt64)
    (lps : Array Float) : IO (Array Float × Array Float) := do
  let cfg := net.cfg
  let B := cs.size
  let N := cfg.N
  let cfgs := cfgsBA cs
  let x ← inputsOf cfg cfgs ByteArray.empty B 2
  let some flipSess := net.flipSess | throw <| IO.userError "no flip session"
  let out ← net.forward flipSess params x net.xShBN (B * N)
  let lpF ← logPsiRows cfg net.ref net.biasBA out cfgs ByteArray.empty B 2
  let mut eloc : Array Float := Array.mkEmpty B
  let mut sx : Array Float := Array.mkEmpty B
  for s in [0:B] do
    let mut r := 0.0
    for i in [0:N] do
      r := r + Float.exp (lpF[s * N + i]! - lps[s]!)
    eloc := eloc.push (eDiag cfg cs[s]! - cfg.h * r)
    sx := sx.push r
  return (eloc, sx)

/-- Metropolis: `nSweeps` sweeps of N single-spin-flip proposals over all B
    chains, sites staggered across chains so a batch mixes them, acceptance
    min(1, ψ'²/ψ²) from the batched eval graph. Returns the acceptance rate. -/
def Net.metropolis (net : Net) (params : ByteArray) (cs : Array UInt64) (lps : Array Float)
    (nSweeps : Nat) (g : StdGen) : IO (Array UInt64 × Array Float × StdGen × Float) := do
  let cfg := net.cfg
  let B := cs.size
  let N := cfg.N
  let mut cs := cs
  let mut cfgs := cfgsBA cs
  let mut lps := lps
  let mut g := g
  let mut acc := 0
  for sw in [0:nSweeps] do
    for r in [0:N] do
      let mut sites : ByteArray := ByteArray.emptyWithCapacity B
      for s in [0:B] do sites := sites.push ((r + s + sw * 7) % N).toUInt8
      let x ← inputsOf cfg cfgs sites B 1
      let out ← net.forward net.evalSess params x net.xShM B
      let lp' ← logPsiRows cfg net.ref net.biasBA out cfgs sites B 1
      for s in [0:B] do
        let (u, g') := randNat g 1 1000000000
        g := g'
        if Float.log (u.toFloat / 1.0e9) < 2.0 * (lp'[s]! - lps[s]!) then
          let c' := cs[s]! ^^^ ((1 : UInt64) <<< (sites.get! s).toNat.toUInt64)
          cs := cs.set! s c'
          cfgs := setU64 cfgs s c'
          lps := lps.set! s lp'[s]!
          acc := acc + 1
  return (cs, lps, g, acc.toFloat / (nSweeps * N * B).toFloat)

/-- Exact autoregressive sampling for the GPT ansatz: T forwards at batch B,
    patch k drawn from softmax(logits_k + bias) and written into the context.
    The last forward's output holds every conditional, so log ψ comes free. -/
def Net.gptSample (net : Net) (params : ByteArray) (B : Nat) (g : StdGen)
    : IO (Array UInt64 × Array Float × StdGen) := do
  let cfg := net.cfg
  let T := cfg.T
  let V := cfg.V
  -- the context as [B, T] int32 ids (BOS = V everywhere to start), converted by
  -- `F32.idsToFloats`; one byte per id suffices since V + 1 ≤ 256
  let mut ctx : ByteArray := ByteArray.emptyWithCapacity (B * T * 4)
  for _ in [0:B * T] do ctx := (((ctx.push V.toUInt8).push 0).push 0).push 0
  let mut ids : Array Nat := Array.replicate (B * T) 0
  let mut g := g
  let mut out : ByteArray := ByteArray.empty
  for k in [0:T] do
    let x ← F32.idsToFloats ctx
    out ← net.forward net.evalSess params x net.xShM B
    let (seed, g') := randNat g 0 ((1 <<< 30) - 1)
    g := g'
    let draw ← nqsGptDraw out net.biasBA B.toUSize T.toUSize V.toUSize k.toUSize seed.toUSize
    for s in [0:B] do
      let id := (draw.get! s).toNat
      ids := ids.set! (s * T + k) id
      if k + 1 < T then ctx := ctx.set! ((s * T + k + 1) * 4) id.toUInt8
  let mut cs : Array UInt64 := Array.mkEmpty B
  for s in [0:B] do
    let mut c : UInt64 := 0
    for k in [0:T] do
      c := c ||| (ids[s * T + k]!.toUInt64 <<< (k * cfg.p).toUInt64)
    cs := cs.push c
  -- the last context was [BOS, id_0 … id_{T−2}], so its output holds every
  -- conditional of the finished configurations: log ψ comes free
  let lps ← logPsiRows cfg net.ref net.biasBA out (cfgsBA cs) ByteArray.empty B 0
  return (cs, lps, g)

/-- Correlation function C(r) = ⟨σᶻᵢσᶻᵢ₊ᵣ⟩ averaged over the ring and the weighted
    configurations, r = 0..N/2. -/
def correlations (cfg : Cfg) (cs : Array UInt64) (w : Array Float) : Array Float := Id.run do
  let N := cfg.N
  let mut out : Array Float := Array.mkEmpty (N / 2 + 1)
  for r in [0:N / 2 + 1] do
    let mut acc := 0.0
    for s in [0:cs.size] do
      let mut t := 0.0
      for i in [0:N] do
        t := t + spin cs[s]! i * spin cs[s]! ((i + r) % N)
      acc := acc + w[s]! * t / N.toFloat
    out := out.push acc
  return out

/-- Complex local energies of the J1-J2 chain over an enumerated S_z sector
    ([ffi/f32_helpers.c](https://github.com/brettkoonce/lean4-mlir/blob/main/ffi/f32_helpers.c)
    `lean_nqs_j1j2_eloc`): `cfgs` ascending u64, `a`/`phi` f64 [M] with the reference
    added, result f32 [M, 2] = (Re, Im). -/
@[extern "lean_nqs_j1j2_eloc"]
opaque nqsJ1J2Eloc (cfgs : @& ByteArray) (a : @& ByteArray) (phi : @& ByteArray)
    (M N : USize) (J1 J2 : Float) : IO ByteArray

/-- Rung R4 (§3): the J1-J2 Heisenberg chain, H = J1 Σ Sᵢ·Sᵢ₊₁ + J2 Σ Sᵢ·Sᵢ₊₂, in
    the S_z = 0 sector by enumeration (C(N, N/2) configurations, 12,870 at N = 16).
    Frustration gives the ground state a sign structure, so the head has two slots:
    log ψ = f_θ + i(φ_ref + φ_θ), with φ_ref the Marshall sign (−1)^(up spins on the
    even sublattice) — exact at J2 = 0 — and the amplitude reference uniform. The
    energy gradient for a complex ψ is 2 Re Σ p (E_loc − E) (∂ log ψ)*, i.e. two host
    weights per configuration, 2p·Re(E_loc − E) on the log|ψ| slot and
    2p·Im(E_loc − E) on the phase slot, through the same MSE block. `noref` drops
    the sign prior and the network has to find the signs itself. -/
def runJ1J2 (cfg : Cfg) : IO Unit := do
  unless cfg.N ≤ 16 && cfg.N % 2 == 0 do
    throw <| IO.userError "model=j1j2 enumerates the S_z = 0 sector: N even and ≤ 16"
  if cfg.arch == "gpt" then
    throw <| IO.userError "model=j1j2: the GPT needs a second (phase) head the spec language does not have; mlp/vit"
  let spec := mkSpec cfg
  match spec.validate with
  | some e => throw <| IO.userError s!"spec: {e}"
  | none => pure ()
  -- the sector, ascending (the C helper binary-searches it)
  let mut cs : Array UInt64 := #[]
  for c in [0:1 <<< cfg.N] do
    if popcnt c.toUInt64 cfg.N == cfg.N / 2 then cs := cs.push c.toUInt64
  let M := cs.size
  let nOut := 2
  IO.eprintln s!"{spec.name}: {spec.totalParams} params, J1-J2 chain N = {cfg.N}, J1 = {cfg.J}, \
J2 = {cfg.J2}, S_z = 0 sector of {M} configurations, {cfg.steps} steps, \
{if cfg.useRef then "Marshall sign prior" else "no sign prior (noref)"}, seed {cfg.seed}"
  -- the Marshall phase, 0 or π per configuration
  let mut maskA : UInt64 := 0
  for i in [0:cfg.N] do
    if i % 2 == 0 then maskA := maskA ||| ((1 : UInt64) <<< i.toUInt64)
  let phiRef : Array Float := cs.map fun c =>
    if cfg.useRef && popcnt (c &&& maskA) cfg.N % 2 == 1 then piF else 0.0

  -- ── graphs ──
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix
  let opfx := s!"{pfx}_j2{(Float.round (cfg.J2 * 100.0)).toUInt64.toNat}" ++
    (if cfg.useRef then "" else "_noref")
  let outShape : List Nat := [M, nOut, 1, 1]
  let trainMlir := MlirCodegen.generateTrainStep spec M
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := 0.0) (useAdam := true)
    (useDdpm := true) (ddpmOutShape := outShape)
  unless (trainMlir.splitOn "%loss =").length >= 2 do
    throw <| IO.userError "train step emitted without a loss"
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec M)
  let sess ← LowererSession.create (← NetSpec.graphArtifact pfx "train_step")
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
  IO.eprintln "  sessions loaded"
  let nP := spec.totalParams
  let nT := 3 * nP
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xShM := spec.xShape M
  let p0 ← spec.heInitParams
  let nHead := headParams cfg
  -- With the prior the head starts at zero, so step 0 IS the uniform × Marshall
  -- floor. Without it a zero head is the uniform POSITIVE state — the S = N/2
  -- eigenstate of the sector, where the gradient vanishes exactly — so the
  -- no-prior arm keeps its He-initialised head and starts from a random state.
  let zeroHead ← F32.const nHead.toUSize 0.0
  let mut p := if cfg.useRef then (F32.slice p0 0 (nP - nHead)).append zeroHead else p0
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let cfgsAll := cfgsBA cs
  let xAll ← inputsOf cfg cfgsAll ByteArray.empty M 0

  -- one evaluation of the state: the raw head, log|ψ|, arg ψ, weights, E_loc
  let evalState (params : ByteArray)
      : IO (ByteArray × Array Float × Array Float × Array Float × Array Float × Array Float) := do
    let out ← LowererSession.forwardF32 evalSess spec.evalFnName params evalShapes xAll xShM
                M.toUSize nOut.toUSize
    let a := (Array.range M).map fun s => F32.read out (2 * s).toUSize
    let ph := (Array.range M).map fun s => F32.read out (2 * s + 1).toUSize + phiRef[s]!
    let mut mx := -1.0e300
    for l in a do if l > mx then mx := l
    let mut z := 0.0
    let mut w : Array Float := Array.mkEmpty M
    for l in a do
      let e := Float.exp (2.0 * (l - mx))
      w := w.push e
      z := z + e
    w := w.map (· / z)
    let aBA := a.foldl pushF64 (ByteArray.emptyWithCapacity (M * 8))
    let phBA := ph.foldl pushF64 (ByteArray.emptyWithCapacity (M * 8))
    let el ← nqsJ1J2Eloc cfgsAll aBA phBA M.toUSize cfg.N.toUSize cfg.J cfg.J2
    let re := (Array.range M).map fun s => F32.read el (2 * s).toUSize
    let im := (Array.range M).map fun s => F32.read el (2 * s + 1).toUSize
    return (out, a, ph, w, re, im)

  let mut curve : Array (Nat × Float × Float) := #[]
  let t0 ← IO.monoMsNow
  let mut eFloor := 0.0
  for step in [0:cfg.steps + 1] do
    let (out, _, _, w, re, im) ← evalState p
    let mut er := 0.0
    let mut ei := 0.0
    for s in [0:M] do
      er := er + w[s]! * re[s]!
      ei := ei + w[s]! * im[s]!
    let mut var := 0.0
    for s in [0:M] do
      var := var + w[s]! * ((re[s]! - er) * (re[s]! - er) + (im[s]! - ei) * (im[s]! - ei))
    if step == 0 then eFloor := er
    if step % cfg.logEvery == 0 || step == cfg.steps then
      curve := curve.push (step, er, var)
      let t1 ← IO.monoMsNow
      IO.eprintln s!"  step {step}  E = {fmt er 6}  E/N = {fmt (er / cfg.N.toFloat) 6}  \
Im E = {sci ei}  Var(E_loc) = {sci var}  ({t1 - t0} ms)"
    if step == cfg.steps then break
    -- y = out − (M·nOut/2)·g with g the two cotangents per configuration
    let mut y : ByteArray := ByteArray.emptyWithCapacity (M * nOut * 4)
    for s in [0:M] do
      y := pushF32 y (F32.read out (2 * s).toUSize - M.toFloat * 2.0 * w[s]! * (re[s]! - er))
      y := pushF32 y (F32.read out (2 * s + 1).toUSize - M.toFloat * 2.0 * w[s]! * (im[s]! - ei))
    let packed := (p.append m).append v
    let lrNow := if cfg.cosine then
        let lrMin := 0.05 * cfg.lr
        lrMin + 0.5 * (1.0 + Float.cos (piF * step.toFloat / cfg.steps.toFloat)) * (cfg.lr - lrMin)
      else cfg.lr
    let res ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName
                packed allShapes xAll xShM y lrNow (step + 1).toFloat bnShapes
                M.toUSize nOut.toUSize 1 1
    p := F32.slice res 0 nP
    m := F32.slice res nP nP
    v := F32.slice res (2 * nP) nP
  let t1 ← IO.monoMsNow
  IO.eprintln s!"trained: {cfg.steps} steps, {t1 - t0} ms"

  -- ── final measurement, exact over the sector ──
  let (_, a, ph, w, re, im) ← evalState p
  let mut er := 0.0
  let mut ei := 0.0
  for s in [0:M] do
    er := er + w[s]! * re[s]!
    ei := ei + w[s]! * im[s]!
  let mut var := 0.0
  for s in [0:M] do
    var := var + w[s]! * ((re[s]! - er) * (re[s]! - er) + (im[s]! - ei) * (im[s]! - ei))
  let corr := correlations cfg cs w
  -- how well the phases follow the Marshall sign, up to a global phase:
  -- |Σ_s w_s exp(i(φ_s − φ_Marshall(s)))|, 1 when every phase agrees
  let mut mc := 0.0
  let mut ms := 0.0
  for s in [0:M] do
    let phM := if popcnt (cs[s]! &&& maskA) cfg.N % 2 == 1 then piF else 0.0
    mc := mc + w[s]! * Float.cos (ph[s]! - phM)
    ms := ms + w[s]! * Float.sin (ph[s]! - phM)
  let marshall := Float.sqrt (mc * mc + ms * ms)
  -- `_psi.bin`: [M, 2] f32 of (log|ψ|, φ) in the sector's ascending order, which the
  -- scorer regenerates; `_samples.bin` keeps the spins beside weight and E_loc
  let mut psi : ByteArray := ByteArray.emptyWithCapacity (M * 2 * 4)
  let mut rows : ByteArray := ByteArray.emptyWithCapacity (M * (cfg.N + 3) * 4)
  for s in [0:M] do
    for i in [0:cfg.N] do
      rows := pushF32 rows (spin cs[s]! i)
    psi := pushF32 psi a[s]!
    psi := pushF32 psi ph[s]!
    rows := pushF32 rows w[s]!
    rows := pushF32 rows re[s]!
    rows := pushF32 rows im[s]!
  let t2 ← IO.monoMsNow
  let corrStr := String.intercalate ", " (corr.toList.map fun c => fmt c 6)
  IO.println s!"{cfg.arch} j1j2 N={cfg.N} J2={cfg.J2}: E = {fmt er 6}  E/N = {fmt (er / cfg.N.toFloat) 6}  \
Im E = {sci ei}  Var(E_loc) = {sci var}  Marshall weight = {fmt marshall 5}  C(N/2) = {fmt corr[cfg.N / 2]! 5}  \
({M} configurations, {spec.totalParams} params, {(t2 - t0) / 1000} s)"
  let json := "{" ++
    s!"\"model\": \"j1j2\", \"arch\": \"{cfg.arch}\", \"N\": {cfg.N}, \"J\": {fmt cfg.J 4}, \"J2\": {fmt cfg.J2 4}, \"h\": 0, " ++
    s!"\"steps\": {cfg.steps}, \"seed\": {cfg.seed}, \"useRef\": {if cfg.useRef then "true" else "false"}, \"symRef\": false, " ++
    s!"\"params\": {spec.totalParams}, \"samples\": {M}, \"seconds\": {fmt ((t2 - t0).toFloat / 1000.0) 1}, " ++
    s!"\"E\": {fmt er 8}, \"E_im\": {sci ei}, \"var\": {sci var}, \"sx\": 0, \"corr\": [{corrStr}], " ++
    s!"\"marshall\": {fmt marshall 6}, \"E_floor\": {fmt eFloor 8}, " ++
    s!"\"config\": \"p={cfg.p} d={cfg.d} heads={cfg.heads} blocks={cfg.blocks} hidden={cfg.hidden} lr={fmt cfg.lr 5}{if cfg.cosine then " cosine" else ""}\"" ++
    "}\n"
  IO.FS.writeFile s!"{opfx}_metrics.json" json
  IO.FS.writeFile s!"{opfx}_curve.csv" ("step,E,var\n" ++
    String.join (curve.toList.map fun (s, e, v) => s!"{s},{fmt e 8},{sci v}\n"))
  IO.FS.writeBinFile s!"{opfx}_psi.bin" psi
  IO.FS.writeBinFile s!"{opfx}_samples.bin" rows
  IO.FS.writeBinFile s!"{opfx}_params.bin" p
  IO.eprintln s!"wrote {opfx}_metrics.json, _curve.csv, _psi.bin, _samples.bin, _params.bin"

def parseFloat (s : String) : Option Float :=
  let neg := s.startsWith "-"
  let s : String := if neg then String.ofList (s.toList.drop 1) else s
  match s.splitOn "." with
  | [a] => a.toNat?.map fun n => (if neg then -1.0 else 1.0) * n.toFloat
  | [a, b] =>
    match a.toNat?, b.toNat? with
    | some ia, some ib =>
      let f := ia.toFloat + ib.toFloat / Float.pow 10.0 b.length.toFloat
      some ((if neg then -1.0 else 1.0) * f)
    | _, _ => none
  | _ => none

def parseArgs (args : List String) : Cfg := Id.run do
  let mut c : Cfg := {}
  match args with
  | a :: _ => c := { c with arch := a }
  | [] => pure ()
  for a in args.drop 1 do
    if a == "noref" then c := { c with useRef := false }
    else if a == "cosine" then c := { c with cosine := true }
    else if a == "check" then c := { c with check := true }
    else if a == "symref" then c := { c with symRef := true }
    else match a.splitOn "=" with
    | [k, v] =>
      let n := v.toNat?
      let f := parseFloat v
      match k with
      | "N" => c := { c with N := n.getD c.N }
      | "h" => c := { c with h := f.getD c.h }
      | "J" => c := { c with J := f.getD c.J }
      | "steps" => c := { c with steps := n.getD c.steps }
      | "seed" => c := { c with seed := n.getD c.seed }
      | "p" => c := { c with p := n.getD c.p }
      | "d" => c := { c with d := n.getD c.d }
      | "heads" => c := { c with heads := n.getD c.heads }
      | "blocks" => c := { c with blocks := n.getD c.blocks }
      | "hidden" => c := { c with hidden := n.getD c.hidden }
      | "B" => c := { c with B := n.getD c.B }
      | "sweeps" => c := { c with sweeps := n.getD c.sweeps }
      | "burn" => c := { c with burn := n.getD c.burn }
      | "lr" => c := { c with lr := f.getD c.lr }
      | "evalb" => c := { c with evalb := n.getD c.evalb }
      | "log" => c := { c with logEvery := n.getD c.logEvery }
      | "tag" => c := { c with tag := v }
      | "model" => c := { c with model := v }
      | "J2" => c := { c with J2 := f.getD c.J2 }
      | "cseed" => c := { c with cseed := n.getD c.cseed }
      | _ => pure ()
    | _ => pure ()
  return c

end NQS

open NQS in
def main (args : List String) : IO Unit := do
  let cfg := parseArgs args
  unless cfg.arch == "mlp" || cfg.arch == "vit" || cfg.arch == "gpt" do
    throw <| IO.userError "usage: nqs-ising <mlp|vit|gpt> [N=12] [h=1.0] [steps=1000] [seed=1] \
[p=2] [d=32] [heads=2] [blocks=2] [hidden=64] [B=1024] [sweeps=2] [lr=0.001] [cosine] [evalb=8] [tag=x] [noref] [symref] [check] [model=ising|j1j2 J2=0.25]"
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "nqs-ising runs on the XLA backend only"
  if cfg.model == "j1j2" then
    runJ1J2 cfg
    return
  if cfg.model != "ising" then throw <| IO.userError s!"unknown model {cfg.model}: ising | j1j2"
  if cfg.arch != "mlp" && cfg.N % cfg.p != 0 then
    throw <| IO.userError s!"N = {cfg.N} is not a multiple of the patch size p = {cfg.p}"
  if cfg.N > 64 then throw <| IO.userError "N ≤ 64 (configurations are UInt64 bit strings)"
  if cfg.symRef && cfg.arch == "gpt" then
    throw <| IO.userError "symref: the symmetrised reference is not a product state, so it has no per-patch logit bias; mlp/vit only"
  let ref := mkRef cfg.J cfg.h
  let bias := patchBias cfg ref
  let biasBA := bias.foldl pushF32 ByteArray.empty
  let spec := mkSpec cfg
  match spec.validate with
  | some e => throw <| IO.userError s!"spec: {e}"
  | none => pure ()
  let M := cfg.M
  let nOut := cfg.nOut
  IO.eprintln s!"{spec.name}: {spec.totalParams} params, N = {cfg.N}, h = {cfg.h}, J = {cfg.J}, \
{cfg.steps} steps, {if cfg.enumerate then s!"enumeration of {M} configurations" else s!"{cfg.B} chains"}, \
{if !cfg.useRef then "uniform start (noref)" else if cfg.symRef then "Z2-symmetrised mean-field reference" else "mean-field reference"}, seed {cfg.seed}"
  IO.eprintln s!"  mean field: φ = {fmt ref.phi 4}, E_MF = {fmt (cfg.N.toFloat * ref.eSite) 6}"

  -- ── graphs ──
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix
  let opfx := s!"{pfx}_h{(Float.round (cfg.h * 100.0)).toUInt64.toNat}" ++
    (if !cfg.useRef then "_noref" else if cfg.symRef then "_symref" else "")
  -- the train forward's output: [M, 1, 1, 1] for a dense head, [M, V, T, 1] for the lmHead
  let outShape : List Nat := if cfg.arch == "gpt" then [M, cfg.V, cfg.T, 1] else [M, 1, 1, 1]
  let trainMlir := MlirCodegen.generateTrainStep spec M
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := 0.0) (useAdam := true)
    (useDdpm := true) (ddpmOutShape := outShape)
  unless (trainMlir.splitOn "%loss =").length >= 2 do
    throw <| IO.userError "train step emitted without a loss — the DDPM branch did not match the output shape"
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec M)
  let sess ← LowererSession.create (← NetSpec.graphArtifact pfx "train_step")
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
  let flipSess ← if cfg.enumerate then pure none else do
    IO.FS.writeFile s!"{pfx}_fwd_flip.mlir" (MlirCodegen.generateEval spec (cfg.B * cfg.N))
    pure (some (← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_flip")))
  IO.eprintln "  sessions loaded"

  let nP := spec.totalParams
  let nT := 3 * nP
  let allShapes := spec.shapesBA
  let bnShapes := spec.bnShapesBA
  let xShM := spec.xShape M
  let net : Net := { cfg, ref, bias, biasBA, spec, evalSess, flipSess, evalShapes := spec.evalShapesBA,
                     xShM, xShBN := spec.xShape (cfg.B * cfg.N) }
  -- He init everywhere but the head, which starts at zero: f_θ = 0, ψ = ψ_ref.
  let p0 ← spec.heInitParams
  let nHead := headParams cfg
  let mut p := (F32.slice p0 0 (nP - nHead)).append (← F32.const nHead.toUSize 0.0)
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let mut g := mkStdGen cfg.seed

  -- ── the configurations: all of them, or B chains ──
  let mut cs : Array UInt64 := #[]
  let mut lps : Array Float := #[]
  let mut xAll : ByteArray := ByteArray.empty
  let mut eDiagAll : Array Float := #[]
  if cfg.enumerate then
    for c in [0:M] do
      cs := cs.push c.toUInt64
      eDiagAll := eDiagAll.push (eDiag cfg c.toUInt64)
    xAll ← inputsOf cfg (cfgsBA cs) ByteArray.empty M 0
  else if cfg.arch != "gpt" then
    for s in [0:cfg.B] do
      let (u, g') := randNat g 0 ((1 <<< 30) - 1)
      let (u2, g'') := randNat g' 0 ((1 <<< 30) - 1)
      g := g''
      let c : UInt64 := (u.toUInt64 <<< 34) ^^^ (u2.toUInt64 <<< 4) ^^^ (u.toUInt64 >>> 3)
      let c := c &&& (if cfg.N == 64 then 0xffffffffffffffff else ((1 : UInt64) <<< cfg.N.toUInt64) - 1)
      cs := cs.push c
    -- log ψ of the initial chains, then burn in
    let cfgs0 := cfgsBA cs
    let x ← inputsOf cfg cfgs0 ByteArray.empty cfg.B 0
    let out ← net.forward evalSess p x xShM cfg.B
    lps ← logPsiRows cfg ref biasBA out cfgs0 ByteArray.empty cfg.B 0
    let (cs', lps', g', ar) ← net.metropolis p cs lps cfg.burn g
    cs := cs'; lps := lps'; g := g'
    IO.eprintln s!"  burn-in: {cfg.burn} sweeps, acceptance {fmt ar 3}"

  let cfgsAll := if cfg.enumerate then cfgsBA cs else ByteArray.empty
  let mut curve : Array (Nat × Float × Float) := #[]
  let t0 ← IO.monoMsNow
  let mut lastE := 0.0
  let mut lastVar := 0.0
  for step in [0:cfg.steps + 1] do
    -- ── samples and log ψ ──
    let mut out : ByteArray := ByteArray.empty
    let mut weights : Array Float := #[]
    let mut eloc : Array Float := #[]
    let mut x : ByteArray := ByteArray.empty
    if cfg.enumerate then
      x := xAll
      out ← net.forward evalSess p xAll xShM M
      lps ← logPsiRows cfg ref biasBA out cfgsAll ByteArray.empty M 0
      let mut mx := -1.0e300
      for l in lps do if l > mx then mx := l
      let mut z := 0.0
      weights := Array.mkEmpty M
      for l in lps do
        let w := Float.exp (2.0 * (l - mx))
        weights := weights.push w
        z := z + w
      weights := weights.map (· / z)
      eloc := Array.mkEmpty M
      for c in [0:M] do
        let mut r := 0.0
        for i in [0:cfg.N] do
          r := r + Float.exp (lps[c ^^^ (1 <<< i)]! - lps[c]!)
        eloc := eloc.push (eDiagAll[c]! - cfg.h * r)
    else
      if cfg.arch == "gpt" then
        let (cs', lps', g') ← net.gptSample p cfg.B g
        cs := cs'; lps := lps'; g := g'
      else
        let (cs', lps', g', _) ← net.metropolis p cs lps cfg.sweeps g
        cs := cs'; lps := lps'; g := g'
      x ← inputsOf cfg (cfgsBA cs) ByteArray.empty cfg.B 0
      out ← net.forward evalSess p x xShM cfg.B
      weights := Array.replicate cfg.B (1.0 / cfg.B.toFloat)
      let (e, _) ← net.localEnergies p cs lps
      eloc := e
    -- ── energy, variance, the gradient weights ──
    let mut e := 0.0
    for s in [0:M] do e := e + weights[s]! * eloc[s]!
    let mut var := 0.0
    for s in [0:M] do var := var + weights[s]! * (eloc[s]! - e) * (eloc[s]! - e)
    lastE := e; lastVar := var
    if step % cfg.logEvery == 0 || step == cfg.steps then
      curve := curve.push (step, e, var)
      let t1 ← IO.monoMsNow
      IO.eprintln s!"  step {step}  E = {fmt e 6}  E/N = {fmt (e / cfg.N.toFloat) 6}  \
Var(E_loc) = {sci var}  ({t1 - t0} ms)"
    if step == cfg.steps then break
    let w := (Array.range M).map fun s => 2.0 * weights[s]! * (eloc[s]! - e)
    let y := targets cfg bias out cs w
    let packed := (p.append m).append v
    let lrNow := if cfg.cosine then
        let lrMin := 0.05 * cfg.lr
        lrMin + 0.5 * (1.0 + Float.cos (piF * step.toFloat / cfg.steps.toFloat)) * (cfg.lr - lrMin)
      else cfg.lr
    let res ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName
                packed allShapes x xShM y lrNow (step + 1).toFloat bnShapes
                M.toUSize outShape[1]!.toUSize outShape[2]!.toUSize 1
    let _ := F32.extractLoss res nT
    p := F32.slice res 0 nP
    m := F32.slice res nP nP
    v := F32.slice res (2 * nP) nP
  let t1 ← IO.monoMsNow
  IO.eprintln s!"trained: {cfg.steps} steps, {t1 - t0} ms"

  -- ── final measurement: exact over all configurations, or evalb sample batches ──
  let mut mE := 0.0
  let mut mVar := 0.0
  let mut mSx := 0.0
  let mut corr : Array Float := Array.replicate (cfg.N / 2 + 1) 0.0
  let mut nSamples := 0
  let mut rows : ByteArray := ByteArray.empty
  let mut checkJson := ""
  if cfg.enumerate then
    let outF ← net.forward evalSess p xAll xShM M
    let lpsF ← logPsiRows cfg ref biasBA outF cfgsAll ByteArray.empty M 0
    let mut mx := -1.0e300
    for l in lpsF do if l > mx then mx := l
    let mut z := 0.0
    let mut weights : Array Float := Array.mkEmpty M
    for l in lpsF do
      let w := Float.exp (2.0 * (l - mx))
      weights := weights.push w
      z := z + w
    weights := weights.map (· / z)
    let mut eloc : Array Float := Array.mkEmpty M
    let mut sx : Array Float := Array.mkEmpty M
    for c in [0:M] do
      let mut r := 0.0
      for i in [0:cfg.N] do
        r := r + Float.exp (lpsF[c ^^^ (1 <<< i)]! - lpsF[c]!)
      eloc := eloc.push (eDiagAll[c]! - cfg.h * r)
      sx := sx.push (r / cfg.N.toFloat)
    for c in [0:M] do
      mE := mE + weights[c]! * eloc[c]!
      mSx := mSx + weights[c]! * sx[c]!
    for c in [0:M] do mVar := mVar + weights[c]! * (eloc[c]! - mE) * (eloc[c]! - mE)
    corr := correlations cfg cs weights
    nSamples := M
    for c in [0:M] do
      for i in [0:cfg.N] do rows := pushF32 rows (spin c.toUInt64 i)
      rows := pushF32 rows weights[c]!
      rows := pushF32 rows eloc[c]!
      rows := pushF32 rows sx[c]!
    if cfg.check then
      -- §8: the sampler the N > 14 runs rely on, tested where the answer is known.
      -- E_loc of every drawn configuration is a lookup in the enumerated table.
      IO.FS.writeFile s!"{pfx}_fwd_chk.mlir" (MlirCodegen.generateEval spec cfg.B)
      let chkSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_chk")
      let netB : Net := { net with evalSess := chkSess, xShM := spec.xShape cfg.B }
      if cfg.cseed != 0 then g := mkStdGen cfg.cseed
      let mut csB : Array UInt64 := #[]
      let mut lpsB : Array Float := #[]
      if cfg.arch != "gpt" then
        for _ in [0:cfg.B] do
          let (u, g') := randNat g 0 (M - 1)
          g := g'
          csB := csB.push u.toUInt64
        lpsB := csB.map fun c => lpsF[c.toNat]!
        let (cs', lps', g', ar) ← netB.metropolis p csB lpsB cfg.burn g
        csB := cs'; lpsB := lps'; g := g'
        IO.eprintln s!"  check: Metropolis burn-in {cfg.burn} sweeps, acceptance {fmt ar 3}"
      let mut sE := 0.0
      let mut sE2 := 0.0
      let mut sX := 0.0
      let mut sX2 := 0.0
      let mut n := 0
      for _ in [0:cfg.evalb] do
        if cfg.arch == "gpt" then
          let (cs', lps', g') ← netB.gptSample p cfg.B g
          csB := cs'; lpsB := lps'; g := g'
        else
          let (cs', lps', g', _) ← netB.metropolis p csB lpsB cfg.sweeps g
          csB := cs'; lpsB := lps'; g := g'
        for c in csB do
          let e := eloc[c.toNat]!
          let xv := sx[c.toNat]!
          sE := sE + e; sE2 := sE2 + e * e; sX := sX + xv; sX2 := sX2 + xv * xv
          n := n + 1
      let nf := n.toFloat
      let mcE := sE / nf
      let seE := Float.sqrt (max 0.0 (sE2 / nf - mcE * mcE) / nf)
      let mcX := sX / nf
      let seX := Float.sqrt (max 0.0 (sX2 / nf - mcX * mcX) / nf)
      IO.println s!"sampler check ({cfg.arch}, {n} draws{if cfg.arch == "gpt" then ", independent" else ", Metropolis chains — s.e. is a lower bound"}): \
E_MC = {fmt mcE 5} ± {fmt seE 5} vs exact {fmt mE 5} (z = {fmt ((mcE - mE) / seE) 2}); \
<sx>_MC = {fmt mcX 5} ± {fmt seX 5} vs exact {fmt mSx 5} (z = {fmt ((mcX - mSx) / seX) 2})"
      checkJson := s!", \"check\": \{\"n\": {n}, \"E_mc\": {fmt mcE 6}, \"E_se\": {fmt seE 6}, \
\"sx_mc\": {fmt mcX 6}, \"sx_se\": {fmt seX 6}}"
  else
    let mut allE : Array Float := #[]
    let mut allSx : Array Float := #[]
    let mut allC : Array UInt64 := #[]
    for _ in [0:cfg.evalb] do
      if cfg.arch == "gpt" then
        let (cs', lps', g') ← net.gptSample p cfg.B g
        cs := cs'; lps := lps'; g := g'
      else
        let (cs', lps', g', _) ← net.metropolis p cs lps cfg.sweeps g
        cs := cs'; lps := lps'; g := g'
      let (e, sx) ← net.localEnergies p cs lps
      allE := allE ++ e
      allSx := allSx ++ (sx.map (· / cfg.N.toFloat))
      allC := allC ++ cs
    nSamples := allE.size
    let wq := 1.0 / nSamples.toFloat
    for i in [0:nSamples] do
      mE := mE + wq * allE[i]!
      mSx := mSx + wq * allSx[i]!
    for i in [0:nSamples] do mVar := mVar + wq * (allE[i]! - mE) * (allE[i]! - mE)
    corr := correlations cfg allC (Array.replicate nSamples wq)
    for i in [0:nSamples] do
      for j in [0:cfg.N] do rows := pushF32 rows (spin allC[i]! j)
      rows := pushF32 rows wq
      rows := pushF32 rows allE[i]!
      rows := pushF32 rows allSx[i]!
  let t2 ← IO.monoMsNow
  let corrStr := String.intercalate ", " (corr.toList.map fun c => fmt c 6)
  IO.println s!"{cfg.arch} N={cfg.N} h={cfg.h}: E = {fmt mE 6}  E/N = {fmt (mE / cfg.N.toFloat) 6}  \
Var(E_loc) = {sci mVar}  <sx> = {fmt mSx 5}  C(N/2) = {fmt corr[cfg.N / 2]! 5}  \
({nSamples} samples, {spec.totalParams} params, {(t2 - t0) / 1000} s)"
  let json := "{" ++
    s!"\"arch\": \"{cfg.arch}\", \"N\": {cfg.N}, \"h\": {fmt cfg.h 4}, \"J\": {fmt cfg.J 4}, " ++
    s!"\"steps\": {cfg.steps}, \"seed\": {cfg.seed}, \"useRef\": {if cfg.useRef then "true" else "false"}, \"symRef\": {if cfg.symRef then "true" else "false"}, " ++
    s!"\"params\": {spec.totalParams}, \"samples\": {nSamples}, \"seconds\": {fmt ((t2 - t0).toFloat / 1000.0) 1}, " ++
    s!"\"E\": {fmt mE 8}, \"var\": {sci mVar}, \"sx\": {fmt mSx 8}, \"corr\": [{corrStr}], " ++
    s!"\"E_mf\": {fmt (cfg.N.toFloat * ref.eSite) 8}{checkJson}, " ++
    s!"\"config\": \"p={cfg.p} d={cfg.d} heads={cfg.heads} blocks={cfg.blocks} hidden={cfg.hidden} B={cfg.B} sweeps={cfg.sweeps} lr={fmt cfg.lr 5}{if cfg.cosine then " cosine" else ""}\"" ++
    "}\n"
  IO.FS.writeFile s!"{opfx}_metrics.json" json
  IO.FS.writeFile s!"{opfx}_curve.csv" ("step,E,var\n" ++
    String.join (curve.toList.map fun (s, e, v) => s!"{s},{fmt e 8},{sci v}\n"))
  IO.FS.writeBinFile s!"{opfx}_samples.bin" rows
  IO.FS.writeBinFile s!"{opfx}_params.bin" p
  IO.eprintln s!"wrote {opfx}_metrics.json, _curve.csv, _samples.bin, _params.bin"
