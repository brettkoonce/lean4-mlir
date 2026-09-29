import LeanMlir.TicTacToe
import LeanMlir.F32Array

/-! Tic-tac-toe on an n × n board, k in a row: the solved-game instrument and the
    scripted players, no stack, no GPU. `lake exe ttt-env [n=3] [k=3] [games=1000]
    [seed=1]` builds the table, prints the position counts and the empty board's exact
    value, plays every scripted pairing both ways, and runs the gates the AlphaZero
    trainer relies on: perfect against perfect draws every game when the root is a
    draw, and perfect loses to nobody as either side. `lake exe ttt-env show <index>
    [n=3] [k=3]` prints one position with the exact value of every legal move. The
    game and the table are `LeanMlir/TicTacToe.lean`. -/

open FloatFmt TTT

def showPos (t : Table) (idx : Nat) : IO Unit := do
  let n := t.n
  let mut p := Pos.empty n t.k
  let mut i := idx
  for c in [0:n * n] do
    let d := (i % 3).toUInt8
    if d != 0 then p := { p with cells := p.cells.set! c d, stones := p.stones + 1 }
    i := i / 3
  IO.println s!"index {idx}, {if p.mover == 1 then "X" else "O"} to move, \
{if t.reachable p then s!"exact value {t.value p} for the mover" else "unreachable"}"
  IO.print p.render
  if t.reachable p && !t.isTerminal p then
    let opt := t.optimal p
    for c in p.legal do
      IO.println s!"  cell {c} (r{c / n} c{c % n}): {-(t.value (p.play c))}\
{if opt.contains c then "  optimal" else ""}"

def main (args : List String) : IO Unit := do
  let kv (key : String) : Option String :=
    (args.find? (·.startsWith (key ++ "="))).map (·.drop (key.length + 1) |>.toString)
  let natArg (key : String) (d : Nat) : Nat := ((kv key) >>= String.toNat?).getD d
  let n := natArg "n" 3
  let k := natArg "k" n
  let games := natArg "games" 1000
  let seed := natArg "seed" 1
  let t0 ← IO.monoMsNow
  let t ← Table.build n k
  let t1 ← IO.monoMsNow
  match args with
  | "show" :: idx :: _ =>
    showPos t (idx.toNat?.getD 0)
    return
  | _ => pure ()
  let (all, dec) ← t.counts
  let root := Pos.empty n k
  let rootV := t.value root
  IO.println s!"{n}×{n}, {k} in a row: table {t.tbl.size} bytes in {t1 - t0} ms; \
{all} reachable positions, {dec} of them decisions; the empty board is \
{if rootV > 0 then "a first-player win" else if rootV < 0 then "a second-player win" else "a draw"}"
  let sa ← scriptedAgreement t.tbl n.toUSize k.toUSize
  IO.println s!"expected agreement with the optimal set over the {dec} decision positions: \
random {fmt (100.0 * F32.read sa 1) 2}%, win-or-block {fmt (100.0 * F32.read sa 2) 2}%"
  let perfect := perfectPlayer t
  let arms : Array (String × Player × Player) := #[
    ("random vs random", randomPlayer, randomPlayer),
    ("random vs perfect", randomPlayer, perfect),
    ("win-or-block vs perfect", heuristicPlayer, perfect),
    ("win-or-block vs random", heuristicPlayer, randomPlayer),
    ("perfect vs perfect", perfect, perfect)]
  IO.println s!"{games} games each way, W/D/L for the first-named player:"
  IO.println "  arm                        as X            as O"
  let mut ok := true
  for (name, a, b) in arms do
    let (x, o) := matchUp a b root games seed
    IO.println s!"  {name.pushn ' ' (26 - name.length)} {x.str.pushn ' ' (15 - x.str.length)} {o.str}"
    if name == "perfect vs perfect" && rootV == 0 && (x.d != games || o.d != games) then
      IO.println "  GATE FAILED: perfect vs perfect did not draw every game"
      ok := false
    if name.endsWith "vs perfect" && (x.w != 0 || o.w != 0) then
      IO.println s!"  GATE FAILED: {name} won a game against perfect"
      ok := false
  if !ok then throw <| IO.userError "solver gates failed"
  IO.println "gates: perfect never loses, and draws itself when the root is a draw"
