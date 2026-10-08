import LeanMlir.Leduc
import LeanMlir.F32Array
import LeanMlir.CliArgs

/-! Leduc hold'em: the exact instrument, the tabular solvers and the scripted arms, no stack,
    no GPU. `lake exe leduc-env [r=3] [iters=1000] [seeds=2] [budget=100000000]` prints the
    counts, solves the game with CFR+ and DCFR (exploitability at checkpoints, the game value),
    scores every scripted arm — uniform, honest, the equilibrium with its bluffs removed (the
    price of honesty) — by exploitability and head-to-head, reports the worst-hand raise
    frequencies as a range over solvers and seeds (Leduc's equilibria are not unique), runs
    ES-MCCFR's exploitability-against-nodes curve, and runs the gates the Deep CFR trainer
    relies on. `lake exe leduc-env show` deals and prints one hand under the CFR+ profile.

    Gates (planning/leduc_deep_cfr_demo.md §7, Gate 0): the Lean game's terminal payoffs equal
    the C tree's over every history; 6r + 30r² information sets; the uniform policy's
    exploitability is 2.373611 at r = 3 and the game value is −0.0856 to 1e-3; the CFR+ and
    DCFR tables, the terminal histories and the instrument's readings are written to
    .lake/build/ for `scripts/demos/leduc_gate0_openspiel.py`, which scores the same tables
    with OpenSpiel 2.0.2 and demands agreement to 1e-6 (passes). The game and the
    instrument are `LeanMlir/Leduc.lean`. -/

open FloatFmt Leduc

def say (msg : String) : IO Unit := do
  let out ← IO.getStdout
  out.putStrLn msg
  out.flush

/-- Raise frequency at the worst hands: round 1's open with the bottom rank, and round 2's
    first decision after check-check per (private, public) worst-hand cell. -/
def worstHandRaises (r : Nat) (keys worst tbl : ByteArray) : Array (String × Float) := Id.run do
  let mut out := #[]
  for i in [0:nInfo r] do
    if worst[i]! == 0 then continue
    let player := keys[i * 6]!
    let round := keys[i * 6 + 1]!
    let closing := keys[i * 6 + 2]!
    let state := keys[i * 6 + 3]!
    let a := keys[i * 6 + 4]!
    let pub := keys[i * 6 + 5]!
    if player != 0 || state != 0 then continue      -- P0's opening decision of a round
    if round == 1 && closing != 0 then continue     -- after check-check
    let name := if round == 0 then s!"round 1 open, rank {a}" else s!"rank {a} under public {pub}"
    out := out.push (name, F32.read tbl (i * 3 + 2).toUSize)
  return out

def main (args : List String) : IO Unit := do
  let natArg := CliArgs.natArg args
  let r := natArg "r" 3
  let iters := natArg "iters" 1000
  let seeds := natArg "seeds" 2
  let budget := natArg "budget" 100000000
  let t0 ← IO.monoMsNow
  let cnt ← counts r.toUSize
  let nI := readU64 cnt 0
  let F := readU64 cnt 1
  say s!"Leduc r = {r}: {nI} information sets (rank level), F = {F}, {readU64 cnt 2} suit-aware history nodes"
  let mut ok := true
  -- Gate 0a: the Lean game is the C tree, payoff for payoff
  let lean := allPayoffs r
  let c ← payoffs r.toUSize
  let nC := c.size / 4
  let mut bad := 0
  if lean.size != nC then bad := bad + 1
  else
    for i in [0:nC] do
      if (lean[i]! - F32.read c i.toUSize).abs > 1e-9 then bad := bad + 1
  say s!"  terminal histories: Lean {lean.size}, C {nC}, {bad} payoff mismatches"
  if bad != 0 then
    say "  GATE FAILED: the Lean game and the C tree disagree"
    ok := false
  if nI != nInfo r then
    say s!"  GATE FAILED: {nI} information sets, expected {nInfo r}"
    ok := false
  -- the instrument on the fixed arms
  let uni ← uniformTable r.toUSize
  let eU ← exploitabilityOf r.toUSize uni
  say s!"  uniform: exploitability {fmt (F32.read eU 0) 6} (best response as P0 {fmt (F32.read eU 1) 4}, as P1 {fmt (F32.read eU 2) 4})"
  if r == 3 && (F32.read eU 0 - 2.373611).abs > 1e-5 then
    say "  GATE FAILED: the uniform policy's exploitability is not OpenSpiel's 2.373611"
    ok := false
  -- the solvers
  let names := #["CFR+", "DCFR"]
  let mut tables : Array ByteArray := #[]
  for m in [0:2] do
    let arena ← cfrAlloc r.toUSize m.toUInt8 0 0.0
    let tS ← IO.monoMsNow
    let mut done := 0
    for cp in [10, 100, 300, 1000, 3000, 10000] do
      if cp > iters then break
      cfrIterate arena (cp - done).toUSize
      done := cp
      let tbl ← cfrAverage arena
      let e ← exploitability r tbl
      say s!"  {names[m]!} t={cp}: exploitability {e} value to P0 {fmt (headToHead r.toUSize tbl tbl) 6} ({(← IO.monoMsNow) - tS} ms)"
    if done < iters then
      cfrIterate arena (iters - done).toUSize
      let tbl ← cfrAverage arena
      let e ← exploitability r tbl
      say s!"  {names[m]!} t={iters}: exploitability {e} value to P0 {fmt (headToHead r.toUSize tbl tbl) 6} ({(← IO.monoMsNow) - tS} ms)"
    tables := tables.push (← cfrAverage arena)
  let eq := tables[0]!
  let value := headToHead r.toUSize eq eq
  -- the tables and the terminal histories for the OpenSpiel cross-check
  -- (`scripts/demos/leduc_gate0_openspiel.py` reads these from .lake/build/)
  let ks ← keys r.toUSize
  IO.FS.createDirAll ".lake/build"
  let dumpTable (name : String) (tbl : ByteArray) : IO Unit := do
    let col (i j : Nat) : Int := let v := ks[i * 6 + j]!; if v == 255 then -1 else v.toNat
    let mut s := ""
    for i in [0:nI] do
      s := s ++ s!"{i} {col i 1} {col i 2} {col i 3} {col i 4} {col i 5} \
{fmt (F32.read tbl (i * 3).toUSize) 9} {fmt (F32.read tbl (i * 3 + 1).toUSize) 9} \
{fmt (F32.read tbl (i * 3 + 2).toUSize) 9}\n"
    IO.FS.writeFile s!".lake/build/leduc_r{r}_{name}.txt" s
  dumpTable "cfrplus" eq
  dumpTable "dcfr" tables[1]!
  if r == 3 then
    IO.FS.writeFile ".lake/build/leduc_r3_payoffs.txt"
      ("\n".intercalate ((allTerminals r).map State.historyLine).toList ++ "\n")
  IO.FS.writeFile s!".lake/build/leduc_r{r}_gate0.txt"
    s!"uniform {fmt (F32.read eU 0) 10}\ncfrplus {fmt (← exploitability r eq) 10}\ndcfr {fmt (← exploitability r tables[1]!) 10}\n"
  if r == 3 && (value + 0.0856).abs > 1e-3 then
    say s!"  GATE FAILED: game value {value}, expected −0.0856"
    ok := false
  -- the arms table
  let honest ← honestTable r.toUSize
  let noBluff ← removeBluffs r.toUSize eq
  say "  arm                                  exploitability   vs CFR+ (seat-averaged)"
  for (name, tbl) in [("uniform random", uni), ("scripted honest", honest),
                       ("CFR+ with the bluffs removed", noBluff), ("DCFR", tables[1]!), ("CFR+", eq)] do
    let e ← exploitability r tbl
    let h := seatAveraged r tbl eq
    say s!"  {name.pushn ' ' (36 - name.length)} {fmt e 6}        {fmt h 5}"
  -- the bluff column as a range over solvers and inits
  let worst ← worstHands r.toUSize
  say s!"  worst-hand raise frequencies, {names.size} solvers × {seeds} random inits × {iters} iterations:"
  let mut rows : Array (String × Array Float) := #[]
  for m in [0:2] do
    for seed in [0:seeds + 1] do
      let tbl ← if seed == 0 then pure tables[m]! else do
        let arena ← cfrAlloc r.toUSize m.toUInt8 seed.toUInt64 1.0
        cfrIterate arena iters.toUSize
        cfrAverage arena
      for (name, f) in worstHandRaises r ks worst tbl do
        match rows.findIdx? (·.1 == name) with
        | some i => rows := rows.modify i fun (n, fs) => (n, fs.push f)
        | none => rows := rows.push (name, #[f])
  for (name, fs) in rows do
    let lo := fs.foldl min 1.0
    let hi := fs.foldl max 0.0
    say s!"    {name.pushn ' ' (28 - name.length)} {fmt (100.0 * lo) 1}% – {fmt (100.0 * hi) 1}%"
  -- ES-MCCFR: exploitability against nodes touched
  say "  ES-MCCFR (seed 1): nodes touched, exploitability, coverage"
  let es ← esAlloc r.toUSize 1
  let mut b := 1000
  while b <= budget do
    let st ← esRun es b.toUInt64
    let e ← exploitability r (← esAverage es)
    say s!"    {readU64 st 0} nodes ({readU64 st 1} iterations): {e}, coverage {fmt ((readU64 st 2).toFloat / nI.toFloat) 3}"
    b := b * 10
  if !ok then throw <| IO.userError "leduc gates failed"
  say s!"gates: the Lean game is the C tree, the information-set count is {nInfo r}\
{if r == 3 then ", uniform and the game value are OpenSpiel's" else ""} ({(← IO.monoMsNow) - t0} ms)"
