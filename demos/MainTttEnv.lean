import LeanMlir.TicTacToe
import LeanMlir.F32Array
import LeanMlir.CliArgs

/-! Tic-tac-toe on an n × n board, k in a row: the solved-game instrument and the
    scripted players, no stack, no GPU. `lake exe ttt-env [n=3] [k=n] [games=1000]
    [seed=1] [samples=20000] [sampleMin=6] [opening=0] [cap=22] [nocache]` builds the instrument — the dense table
    at n ≤ 4, the on-demand solver at every n — prints the position counts (or, above
    4×4, the root's solve and a random-play sample's expected agreements) and the empty
    board's exact value, plays every scripted pairing both ways, and runs the gates the
    AlphaZero trainer relies on: the solver reproduces the table on every reachable
    position where the table exists, perfect against perfect draws every game when the
    root is a draw, and perfect loses to nobody as either side. `lake exe ttt-env show
    <index> [n=3] [k=3]` prints one position with the exact value of every legal move.
    Above 4×4 the solver's cache is saved under `.lake/build/` and reloaded next time.
    The game and the instrument are `LeanMlir/TicTacToe.lean`. -/

open FloatFmt TTT

/-- `IO.println` with a flush: Lean's stdout is block-buffered under redirection, so a
    log of a long run would otherwise show nothing until exit. -/
def say (msg : String) : IO Unit := do
  let out ← IO.getStdout
  out.putStrLn msg
  out.flush

def showPos (t : Table) (idx : Nat) : IO Unit := do
  let n := t.n
  let mut p := Pos.empty n t.k
  let mut i := idx
  for c in [0:n * n] do
    let d := (i % 3).toUInt8
    if d != 0 then p := { p with cells := p.cells.set! c d, stones := p.stones + 1 }
    i := i / 3
  say s!"index {idx}, {if p.mover == 1 then "X" else "O"} to move, \
{if t.reachable p then s!"exact value {t.value p} for the mover" else "unreachable"}"
  IO.print p.render
  if t.reachable p && !t.isTerminal p then
    let opt := t.optimal p
    for c in p.legal do
      say s!"  cell {c} (r{c / n} c{c % n}): {-(t.value (p.play c))}\
{if opt.contains c then "  optimal" else ""}"

def main (args : List String) : IO Unit := do
  let natArg := CliArgs.natArg args
  let n := natArg "n" 3
  let k := natArg "k" n
  let games := natArg "games" 1000
  let seed := natArg "seed" 1
  let samples := natArg "samples" 20000
  -- the sample's shallowest positions: a fresh 3-stone subtree costs seconds to solve,
  -- a 6-stone one milliseconds, so the default sweep starts at six stones
  let sampleMin := natArg "sampleMin" 6
  let opening := natArg "opening" 0      -- > 0: every decision position with at most that many stones
  let cachePath := s!".lake/build/ttt_solver_{n}x{n}_k{k}.bin"
  let useCache := n > 4 && !args.contains "nocache"
  let t0 ← IO.monoMsNow
  let mut t ← Table.build n k (natArg "cap" 22)
  if useCache && (← System.FilePath.pathExists cachePath) then
    t := { t with solver := ← IO.FS.readBinFile cachePath }
    say s!"solver cache loaded from {cachePath}"
  let t1 ← IO.monoMsNow
  match args with
  | "show" :: idx :: _ =>
    showPos t (idx.toNat?.getD 0)
    return
  | _ => pure ()
  let root := Pos.empty n k
  -- above 4×4 the root is the max over its openings, each solved full-window (minutes at
  -- (5,5,4) the first time, `runs/2026-09-29-alphazero-ttt/n5_root.log`; the cache remembers)
  let rootV := t.value root
  let verdict := if rootV > 0 then "a first-player win" else if rootV < 0 then "a second-player win" else "a draw"
  let mut ok := true
  if t.dense then
    let (all, dec) ← t.counts
    say s!"{n}×{n}, {k} in a row: table {t.tbl.size} bytes in {t1 - t0} ms; \
{all} reachable positions, {dec} of them decisions; the empty board is {verdict}"
    let sa ← scriptedAgreement t.tbl n.toUSize k.toUSize
    say s!"expected agreement with the optimal set over the {dec} decision positions: \
random {fmt (100.0 * F32.read sa 1) 2}%, win-or-block {fmt (100.0 * F32.read sa 2) 2}%"
    -- the solver against the table, every reachable position
    let tc ← IO.monoMsNow
    let (checked, bad, badMove) ← t.check
    let st ← solverStats t.solver
    say s!"solver vs table: {checked} positions checked, {bad} value mismatches, {badMove} principal \
moves outside the optimal set, {readU64 st 0} entries after {readU64 st 1} nodes ({(← IO.monoMsNow) - tc} ms)"
    if bad != 0 || badMove != 0 then
      say "  GATE FAILED: the on-demand solver disagrees with the table"
      ok := false
  else
    let t2 ← IO.monoMsNow
    let st ← solverStats t.solver
    say s!"{n}×{n}, {k} in a row: no table (3^{n * n} entries); the on-demand solver says \
{verdict} in {t2 - t1} ms ({readU64 st 0} entries after {readU64 st 1} nodes\
, {readU64 st 2} replaced)"
    let idx ← samplePositions n.toUSize k.toUSize samples.toUSize 4242 sampleMin.toUSize
    let ta ← IO.monoMsNow
    let sa ← solverAgreement t.solver idx samples.toUSize
    let st ← solverStats t.solver
    say s!"expected agreement with the optimal set over {samples} random-play decision positions \
with at least {sampleMin} stones: \
random {fmt (100.0 * F32.read sa 1) 2}%, win-or-block {fmt (100.0 * F32.read sa 2) 2}% \
({(← IO.monoMsNow) - ta} ms; {readU64 st 0} entries after {readU64 st 1} nodes, {readU64 st 2} replaced)"
    if opening > 0 then
      let te ← IO.monoMsNow
      let op ← enumeratePositions n.toUSize k.toUSize opening.toUSize
      let nOp := op.size / 8
      let sa ← solverAgreement t.solver op nOp.toUSize
      let st ← solverStats t.solver
      say s!"the opening, every decision position with at most {opening} stones — {nOp} of them: \
random {fmt (100.0 * F32.read sa 1) 2}%, win-or-block {fmt (100.0 * F32.read sa 2) 2}% \
({(← IO.monoMsNow) - te} ms; {readU64 st 0} entries after {readU64 st 1} nodes, {readU64 st 2} replaced)"
  let perfect := perfectPlayer t
  let arms : Array (String × Player × Player) := #[
    ("random vs random", randomPlayer, randomPlayer),
    ("random vs perfect", randomPlayer, perfect),
    ("win-or-block vs perfect", heuristicPlayer, perfect),
    ("win-or-block vs random", heuristicPlayer, randomPlayer),
    ("perfect vs perfect", perfect, perfect)]
  say s!"{games} games each way, W/D/L for the first-named player:"
  say "  arm                        as X            as O"
  for (name, a, b) in arms do
    let tm ← IO.monoMsNow
    let (x, o) := matchUp a b root games seed
    say s!"  {name.pushn ' ' (26 - name.length)} {x.str.pushn ' ' (15 - x.str.length)} \
{o.str.pushn ' ' (15 - o.str.length)} {(← IO.monoMsNow) - tm} ms"
    if name == "perfect vs perfect" && rootV == 0 && (x.d != games || o.d != games) then
      say "  GATE FAILED: perfect vs perfect did not draw every game"
      ok := false
    if name.endsWith "vs perfect" && (x.w != 0 || o.w != 0) then
      say s!"  GATE FAILED: {name} won a game against perfect"
      ok := false
  if !ok then throw <| IO.userError "solver gates failed"
  say (if t.dense then "gates: the solver reproduces the table, perfect never loses, and draws itself when the root is a draw"
              else "gates: perfect never loses, and draws itself when the root is a draw")
  if useCache then
    IO.FS.writeBinFile cachePath t.solver
    let st ← solverStats t.solver
    say s!"solver cache saved to {cachePath} ({readU64 st 0} entries, {t.solver.size / 1048576} MB)"
