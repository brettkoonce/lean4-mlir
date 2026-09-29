import LeanMlir.LEBytes
import LeanMlir.FloatFmt

/-! Tic-tac-toe on an n × n board, k in a row to win: the game, the solved-game
    instrument and the scripted players. The game is pure Lean; the instrument is a
    minimax table over every reachable position, built in C (`ffi/f32_helpers.c`,
    `lean_ttt_*`) because it is 43 MB at 4×4. Shared by the `ttt-env` and
    `alphazero-ttt` demos.

    A position is a base-3 number over the cells in row-major order, digit 0 empty,
    1 X, 2 O, cell `c` weighted `3^c`; X moves first, so the side to move is the stone
    count's parity. Table entries are from the SIDE TO MOVE's view: bits 0–1 the value
    (0 loss, 1 draw, 2 win), bit 2 terminal, 255 unreached. -/

namespace TTT

open FloatFmt

/-- A position: the rules it is played under, the cells, the stone count and the
    cell just played (the only cell a win can run through). -/
structure Pos where
  n : Nat
  k : Nat
  cells : Array UInt8
  stones : Nat := 0
  last : Option Nat := none
deriving Inhabited, Repr

def Pos.empty (n k : Nat) : Pos := { n, k, cells := Array.replicate (n * n) 0 }

/-- 1 for X, 2 for O. -/
def Pos.mover (p : Pos) : UInt8 := (p.stones % 2 + 1).toUInt8

def Pos.index (p : Pos) : Nat := Id.run do
  let mut idx := 0
  let mut w := 1
  for c in [0:p.n * p.n] do
    idx := idx + p.cells[c]!.toNat * w
    w := w * 3
  return idx

/-- Did `who` complete `k` in a row through cell `c`? -/
def Pos.winsThrough (p : Pos) (c : Nat) (who : UInt8) : Bool := Id.run do
  let n := p.n
  let r0 : Int := c / n
  let c0 : Int := c % n
  let dirs : Array (Int × Int) := #[(0, 1), (1, 0), (1, 1), (1, -1)]
  for (dr, dc) in dirs do
    let mut run := 1
    for sgn in [(1 : Int), -1] do
      for s in [1:p.k] do
        let r := r0 + sgn * s * dr
        let cc := c0 + sgn * s * dc
        if r < 0 || r >= n || cc < 0 || cc >= n then break
        if p.cells[(r * n + cc).toNat]! != who then break
        run := run + 1
    if run >= p.k then return true
  return false

/-- The game is over: the last mover completed `k`, or the board is full. -/
def Pos.terminal (p : Pos) : Bool :=
  (match p.last with
    | some c => p.winsThrough c (3 - p.mover)
    | none => false) || p.stones == p.n * p.n

/-- The result of a terminal position from X's view: 1 X won, −1 O won, 0 draw. -/
def Pos.result (p : Pos) : Int :=
  match p.last with
  | some c => if p.winsThrough c (3 - p.mover) then (if p.mover == 2 then 1 else -1) else 0
  | none => 0

/-- The result from the mover's view: −1 at a lost terminal, 0 at a draw. -/
def Pos.resultForMover (p : Pos) : Int :=
  if p.mover == 1 then p.result else -p.result

def Pos.play (p : Pos) (c : Nat) : Pos :=
  { p with cells := p.cells.set! c p.mover, stones := p.stones + 1, last := some c }

def Pos.legal (p : Pos) : Array Nat := Id.run do
  let mut out := #[]
  for c in [0:p.n * p.n] do
    if p.cells[c]! == 0 then out := out.push c
  return out

def Pos.render (p : Pos) : String := Id.run do
  let mut s := ""
  for r in [0:p.n] do
    for c in [0:p.n] do
      let v := p.cells[r * p.n + c]!
      s := s ++ (if v == 1 then "X" else if v == 2 then "O" else ".")
      if c + 1 < p.n then s := s ++ " "
    s := s ++ "\n"
  return s

-- ── The solved game ──

@[extern "lean_ttt_solve"]
opaque solveTable (n k : USize) : IO ByteArray

/-- Reachable indices as u32 LE; decision positions only unless `includeTerminal = 1`. -/
@[extern "lean_ttt_reachable"]
opaque reachableIdx (tbl : @& ByteArray) (n : USize) (includeTerminal : UInt8) : IO ByteArray

/-- Canonical planes for an index list: f32 `[count, 2, n, n]`, plane 0 the mover's
    stones, plane 1 the opponent's. -/
@[extern "lean_ttt_planes"]
opaque planesOf (idx : @& ByteArray) (count n : USize) : IO ByteArray

/-- Score a logits block `[count, nOut]` against the table: f32
    `[argmax agreements, Σ(tanh v − z)², sign agreements, Σ|tanh v − z|]`. -/
@[extern "lean_ttt_score"]
opaque scoreLogits (tbl : @& ByteArray) (idx : @& ByteArray) (count n : USize)
    (out : @& ByteArray) (nOut : USize) : IO ByteArray

/-- The scripted players' expected agreement with the table over every decision
    position: f32 `[decision positions, random, win-or-block]`. -/
@[extern "lean_ttt_scripted_agreement"]
opaque scriptedAgreement (tbl : @& ByteArray) (n k : USize) : IO ByteArray

structure Table where
  n : Nat
  k : Nat
  tbl : ByteArray

def Table.build (n k : Nat) : IO Table := do
  return { n, k, tbl := ← solveTable n.toUSize k.toUSize }

def Table.entry (t : Table) (p : Pos) : UInt8 := t.tbl.get! p.index
def Table.reachable (t : Table) (p : Pos) : Bool := t.entry p != 255
def Table.isTerminal (t : Table) (p : Pos) : Bool := (t.entry p &&& 4) != 0

/-- The exact value from the mover's view: −1 loss, 0 draw, 1 win. -/
def Table.value (t : Table) (p : Pos) : Int := ((t.entry p &&& 3).toNat : Int) - 1

/-- The legal moves that keep the exact value: the child's value is the opponent's. -/
def Table.optimal (t : Table) (p : Pos) : Array Nat :=
  let v := t.value p
  p.legal.filter fun c => -(t.value (p.play c)) == v

def Table.counts (t : Table) : IO (Nat × Nat) := do
  let all ← reachableIdx t.tbl t.n.toUSize 1
  let dec ← reachableIdx t.tbl t.n.toUSize 0
  return (all.size / 4, dec.size / 4)

-- ── Players ──

/-- A player draws a legal move from a position. -/
abbrev Player := Pos → StdGen → Nat × StdGen

def pick (xs : Array Nat) (g : StdGen) : Nat × StdGen :=
  let (i, g) := randNat g 0 (xs.size - 1)
  (xs[i]!, g)

def randomPlayer : Player := fun p g => pick p.legal g

/-- Uniform over the optimal set, so a perfect opponent still varies its lines. -/
def perfectPlayer (t : Table) : Player := fun p g => pick (t.optimal p) g

/-- Win now if a move wins, else block an opponent's immediate win, else random. -/
def heuristicPlayer : Player := fun p g =>
  let wins := p.legal.filter fun c => (p.play c).winsThrough c p.mover
  if wins.size > 0 then pick wins g else
  let blocks := p.legal.filter fun c =>
    ({ p with cells := p.cells.set! c (3 - p.mover) } : Pos).winsThrough c (3 - p.mover)
  if blocks.size > 0 then pick blocks g else pick p.legal g

/-- Play one game out from `p0`; the result is from X's view. -/
partial def playGame (px po : Player) (p0 : Pos) (g : StdGen) : Int × StdGen :=
  if p0.terminal then (p0.result, g) else
  let (c, g) := (if p0.mover == 1 then px else po) p0 g
  playGame px po (p0.play c) g

structure WDL where
  w : Nat := 0
  d : Nat := 0
  l : Nat := 0
deriving Repr

def WDL.add (r : WDL) (x : Int) : WDL :=
  if x > 0 then { r with w := r.w + 1 } else if x < 0 then { r with l := r.l + 1 } else { r with d := r.d + 1 }

def WDL.str (r : WDL) : String := s!"{r.w}/{r.d}/{r.l}"

/-- `games` games with `a` as X against `b`, and `games` with `a` as O; both W/D/L
    from `a`'s view. -/
def matchUp (a b : Player) (start : Pos) (games seed : Nat) : WDL × WDL := Id.run do
  let mut g := mkStdGen seed
  let mut asX : WDL := {}
  let mut asO : WDL := {}
  for _ in [0:games] do
    let (r, g1) := playGame a b start g
    asX := asX.add r
    let (r, g2) := playGame b a start g1
    asO := asO.add (-r)
    g := g2
  return (asX, asO)

end TTT
