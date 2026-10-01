import LeanMlir.LEBytes
import LeanMlir.FloatFmt

/-! Tic-tac-toe on an n × n board, k in a row to win: the game, the solved-game
    instrument and the scripted players. The game is pure Lean; the instrument is a
    minimax table over every reachable position, built in C (`ffi/f32_helpers.c`,
    `lean_ttt_*`) because it is 43 MB at 4×4. Shared by the `ttt-env` and
    `alphazero-ttt` demos; the AlphaZero replay gather and loss targets (`gatherAug`,
    `targets`) bind here too, beside every other `lean_ttt_*` extern.

    A position is a base-3 number over the cells in row-major order, digit 0 empty,
    1 X, 2 O, cell `c` weighted `3^c`; X moves first, so the side to move is the stone
    count's parity. Table entries are from the SIDE TO MOVE's view: bits 0–1 the value
    (0 loss, 1 draw, 2 win), bit 2 terminal, 255 unreached. Index lists between Lean
    and C are u64 LE (3²⁵ > 2³²).

    The table exists at n ≤ 4 (3¹⁶ bytes). Above that the instrument is the on-demand
    solver (`lean_ttt_solver_*`): negamax with a symmetry-canonical transposition
    table, exact values only, gated against the dense table wherever both exist
    (`Table.check`). -/

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

/-- The uint64 at RECORD `i` of a packed little-endian u64 buffer. -/
def readU64 (ba : ByteArray) (i : Nat) : Nat := readU64LE ba (8 * i)

-- ── The solved game ──

@[extern "lean_ttt_solve"]
opaque solveTable (n k : USize) : IO ByteArray

/-- Reachable indices as u64 LE; decision positions only unless `includeTerminal = 1`. -/
@[extern "lean_ttt_reachable"]
opaque reachableIdx (tbl : @& ByteArray) (n : USize) (includeTerminal : UInt8) : IO ByteArray

/-- Canonical planes for an index list (u64 LE): f32 `[count, 2, n, n]`, plane 0 the
    mover's stones, plane 1 the opponent's. -/
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

-- ── The on-demand solver ──

/-- The solver's arena: board `n`, `k` in a row, `2^log2cap` transposition slots
    (10 bytes each). A file, when saved: `IO.FS.writeBinFile`. -/
@[extern "lean_ttt_solver_alloc"]
opaque solverAlloc (n k log2cap : USize) : IO ByteArray

/-- A position's table-style entry, solved on demand. Pure to Lean — the value of a
    position is a function of the position; the cache the call fills is invisible. -/
@[extern "lean_ttt_solver_entry_at"]
opaque solverEntry (arena : @& ByteArray) (idx : UInt64) : UInt8

/-- The principal move of a position whose entry is exact, in the position's own
    coordinates; 63 when the entry is missing or only a bound. Pure like `solverEntry`. -/
@[extern "lean_ttt_solver_best_at"]
opaque solverBest (arena : @& ByteArray) (idx : UInt64) : UInt8

/-- `[entries, nodes visited, entries replaced, capacity]` as u64. -/
@[extern "lean_ttt_solver_stats"]
opaque solverStats (arena : @& ByteArray) : IO ByteArray

/-- The gate: the solver against the dense table on every reachable position, and its
    principal move against the table's optimal set on every decision position —
    `[checked, value mismatches, bad principal moves]` as u64. -/
@[extern "lean_ttt_solver_check"]
opaque solverCheck (arena tbl : @& ByteArray) : IO ByteArray

/-- `count` decision positions with at least `minStones` stones, by random play from
    `seed`: u64 LE. -/
@[extern "lean_ttt_sample_positions"]
opaque samplePositions (n k count : USize) (seed : UInt64) (minStones : USize) : IO ByteArray

/-- Every decision position with at most `maxStones` stones (≤ 6), sorted and distinct:
    u64 LE. The exhaustive opening, where a sample would miss lines. -/
@[extern "lean_ttt_enumerate"]
opaque enumeratePositions (n k maxStones : USize) : IO ByteArray

/-- `scoreLogits` through the solver, over a position list. -/
@[extern "lean_ttt_solver_score"]
opaque solverScore (arena idx : @& ByteArray) (count : USize) (out : @& ByteArray) (nOut : USize) : IO ByteArray

/-- `scriptedAgreement` through the solver, over a position list. -/
@[extern "lean_ttt_solver_agreement"]
opaque solverAgreement (arena idx : @& ByteArray) (count : USize) : IO ByteArray

/-- Replay gather with one random dihedral view per sample, applied alike to the
    planes and the policy target: `(x [count, 2, n, n], π [count, n²])`. -/
@[extern "lean_ttt_gather_aug"]
opaque gatherAug (planes pi idx : @& ByteArray) (count n : USize) (seed : UInt64) :
    IO (ByteArray × ByteArray)

/-- The MSE block's target for `(z − tanh v)² − πᵀ log softmax(p)` on a logits block
    `[count, nOut]`, and the batch's mean AlphaZero loss. -/
@[extern "lean_ttt_targets"]
opaque targets (out pi z : @& ByteArray) (count nOut : USize) (scale : Float) :
    IO (ByteArray × ByteArray)

/-- The instrument: the dense table at n ≤ 4 (`tbl`, empty above), and the on-demand
    solver at every size. `entry` reads the table where it exists. -/
structure Table where
  n : Nat
  k : Nat
  tbl : ByteArray
  solver : ByteArray

/-- `log2cap` sizes the solver's transposition table (2²² slots = 42 MB). -/
def Table.build (n k : Nat) (log2cap : Nat := 22) : IO Table := do
  let tbl ← if n <= 4 then solveTable n.toUSize k.toUSize else pure ByteArray.empty
  let solver ← solverAlloc n.toUSize k.toUSize log2cap.toUSize
  return { n, k, tbl, solver }

def Table.dense (t : Table) : Bool := t.tbl.size > 0
def Table.entry (t : Table) (p : Pos) : UInt8 :=
  if t.dense then t.tbl.get! p.index else solverEntry t.solver p.index.toUInt64
def Table.reachable (t : Table) (p : Pos) : Bool := t.entry p != 255
def Table.isTerminal (t : Table) (p : Pos) : Bool := (t.entry p &&& 4) != 0

/-- The solver against the dense table on every reachable position:
    (checked, value mismatches, bad principal moves). -/
def Table.check (t : Table) : IO (Nat × Nat × Nat) := do
  let r ← solverCheck t.solver t.tbl
  return (readU64 r 0, readU64 r 1, readU64 r 2)

/-- Score a logits block over a position list through whichever instrument exists. -/
def Table.score (t : Table) (idx : ByteArray) (count : Nat) (out : ByteArray) (nOut : Nat) : IO ByteArray :=
  if t.dense then scoreLogits t.tbl idx count.toUSize t.n.toUSize out nOut.toUSize
  else solverScore t.solver idx count.toUSize out nOut.toUSize

/-- The exact value from the mover's view: −1 loss, 0 draw, 1 win. -/
def Table.value (t : Table) (p : Pos) : Int := ((t.entry p &&& 3).toNat : Int) - 1

/-- The legal moves that keep the exact value: the child's value is the opponent's. -/
def Table.optimal (t : Table) (p : Pos) : Array Nat :=
  let v := t.value p
  p.legal.filter fun c => -(t.value (p.play c)) == v

def Table.counts (t : Table) : IO (Nat × Nat) := do
  let all ← reachableIdx t.tbl t.n.toUSize 1
  let dec ← reachableIdx t.tbl t.n.toUSize 0
  return (all.size / 8, dec.size / 8)

-- ── Players ──

/-- A player draws a legal move from a position. -/
abbrev Player := Pos → StdGen → Nat × StdGen

def pick (xs : Array Nat) (g : StdGen) : Nat × StdGen :=
  let (i, g) := randNat g 0 (xs.size - 1)
  (xs[i]!, g)

def randomPlayer : Player := fun p g => pick p.legal g

/-- Uniform over the optimal set, so a perfect opponent still varies its lines — drawn as
    the first optimal move of a uniformly random order of the legal moves, which is the
    same distribution and values about two children a move instead of every one. On the
    solver path that is still a full solve per child, so from the third stone on the
    player takes the search's own principal move — one solve, the position's — and keeps
    the uniform draw for each side's first move, where the cached opening book makes it
    free and where the variety between games comes from. Exact either way. -/
def perfectPlayer (t : Table) : Player := fun p g => Id.run do
  let v := t.value p
  if !t.dense && p.stones >= 2 then
    let m := (solverBest t.solver p.index.toUInt64).toNat
    if m < p.n * p.n && p.cells[m]! == 0 then return (m, g)
  let mut legal := p.legal
  let mut g := g
  for i in [0:legal.size] do
    let (j, g') := randNat g i (legal.size - 1)
    g := g'
    let tmp := legal[i]!
    legal := (legal.set! i legal[j]!).set! j tmp
  for c in legal do
    if -(t.value (p.play c)) == v then return (c, g)
  return (legal[0]!, g)

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
