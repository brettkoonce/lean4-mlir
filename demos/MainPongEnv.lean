/-! Pong in Lean — Phase 0 of `planning/pong_dqn_demo.md`: the game, the renderer,
    the frame-skip wrapper, and the sanity baselines (random, tracker, self-play).
    No stack, no GPU: `lake exe pong-env [games=100]`. The DQN trainer of rung 3
    imports nothing from the stack that this file needs; it wraps `Game.step`. -/

structure Pong where
  ballX : Float
  ballY : Float
  vx : Float
  vy : Float
  padY : Float        -- player paddle centre, left side
  oppY : Float        -- opponent paddle centre, right side
  oppTarget : Float   -- the opponent's held target (reaction delay = sample-and-hold)
  frame : Nat
deriving Repr, Inhabited

inductive Act | stay | up | down
deriving Repr, Inhabited, BEq

def Act.dir : Act → Float
  | .stay => 0.0 | .up => -1.0 | .down => 1.0

def Act.ofNat : Nat → Act
  | 0 => .stay | 1 => .up | _ => .down

/-- Opponent knobs: tracking speed in px per frame and reaction delay in frames. -/
structure Opp where
  speed : Float := 1.5
  delay : Nat := 4

namespace Pong

def W : Float := 84.0
def H : Float := 84.0
def padHalf : Float := 6.0
def padSpeed : Float := 2.0
def playerX : Float := 4.0
def oppX : Float := 79.0
def serveSpeed : Float := 1.5
def speedInc : Float := 0.25
def speedMax : Float := 4.0
def spin : Float := 0.4

def clampF (x lo hi : Float) : Float := max lo (min hi x)

def uniform (g : StdGen) : Float × StdGen :=
  let (n, g) := randNat g 0 999999
  (n.toFloat / 1000000.0, g)

def serve (s : Pong) (g : StdGen) : Pong × StdGen :=
  let (u1, g) := uniform g
  let (u2, g) := uniform g
  let dir := if u1 < 0.5 then -1.0 else 1.0
  ({ s with ballX := 41.0, ballY := 41.0, vx := dir * serveSpeed, vy := (u2 - 0.5) * 2.0 }, g)

def init (g : StdGen) : Pong × StdGen :=
  serve { ballX := 41.0, ballY := 41.0, vx := 0.0, vy := 0.0, padY := 42.0, oppY := 42.0,
          oppTarget := 42.0, frame := 0 } g

def overlaps (bY padY : Float) : Bool :=
  bY + 2.0 > padY - padHalf && bY < padY + padHalf

/-- One raw frame. Returns the new state, the reward (±1 on a point, else 0), the RNG. -/
def step (s : Pong) (o : Opp) (a : Act) (g : StdGen) : Pong × Float × StdGen := Id.run do
  let padY := clampF (s.padY + padSpeed * a.dir) padHalf (H - padHalf)
  let oppTarget := if s.frame % o.delay == 0 then s.ballY + 1.0 else s.oppTarget
  let oppY := clampF (s.oppY + clampF (oppTarget - s.oppY) (-o.speed) o.speed) padHalf (H - padHalf)
  let mut bx := s.ballX + s.vx
  let mut bY := s.ballY + s.vy
  let mut vx := s.vx
  let mut vy := s.vy
  if bY < 0.0 then
    bY := -bY
    vy := -vy
  if bY > H - 2.0 then
    bY := 2.0 * (H - 2.0) - bY
    vy := -vy
  -- player paddle: ball moving left, its left edge inside [playerX - 2, playerX + 2]
  if vx < 0.0 && bx <= playerX + 2.0 && bx + 2.0 >= playerX && overlaps bY padY then
    vx := min speedMax (Float.abs vx + speedInc)
    vy := (bY + 1.0 - padY) * spin
    bx := playerX + 2.0
  if vx > 0.0 && bx + 2.0 >= oppX && bx <= oppX + 2.0 && overlaps bY oppY then
    vx := -(min speedMax (Float.abs vx + speedInc))
    vy := (bY + 1.0 - oppY) * spin
    bx := oppX - 2.0
  let s' : Pong := { s with ballX := bx, ballY := bY, vx := vx, vy := vy, padY := padY,
                            oppY := oppY, oppTarget := oppTarget, frame := s.frame + 1 }
  if bx + 2.0 < 0.0 then
    let (s2, g2) := serve s' g
    return (s2, -1.0, g2)
  if bx > W then
    let (s2, g2) := serve s' g
    return (s2, 1.0, g2)
  return (s', 0.0, g)

def rect (buf : ByteArray) (x y w h : Float) : ByteArray := Id.run do
  let mut b := buf
  let x0 := (clampF x 0.0 W).toUInt64.toNat
  let y0 := (clampF y 0.0 H).toUInt64.toNat
  let x1 := (clampF (x + w) 0.0 W).toUInt64.toNat
  let y1 := (clampF (y + h) 0.0 H).toUInt64.toNat
  for yy in [y0:y1] do
    for xx in [x0:x1] do
      b := b.set! (yy * 84 + xx) 255
  return b

/-- 84 × 84 u8: background 0, paddles and ball 255. -/
def render (s : Pong) : ByteArray :=
  let blank := ByteArray.mk (Array.replicate (84 * 84) 0)
  let b := rect blank playerX (s.padY - padHalf) 2.0 12.0
  let b := rect b oppX (s.oppY - padHalf) 2.0 12.0
  rect b s.ballX s.ballY 2.0 2.0

/-- The six-number state the state-vector DQN sees, scaled to about [-1, 1]. -/
def stateVec (s : Pong) : Array Float :=
  #[s.ballX / 42.0 - 1.0, s.ballY / 42.0 - 1.0, s.vx / speedMax, s.vy / speedMax,
    s.padY / 42.0 - 1.0, s.oppY / 42.0 - 1.0]

end Pong

/-- A game to 21 with frame skip 4. -/
structure Game where
  p : Pong
  scoreP : Nat
  scoreO : Nat
  frames : Nat
  g : StdGen

def frameCap : Nat := 30000

def Game.reset (seed : Nat) : Game :=
  let (p, g) := Pong.init (mkStdGen seed)
  { p := p, scoreP := 0, scoreO := 0, frames := 0, g := g }

def Game.step (gm : Game) (o : Opp) (a : Act) : Game × Float × Bool := Id.run do
  let mut p := gm.p
  let mut g := gm.g
  let mut r := 0.0
  let mut sp := gm.scoreP
  let mut so := gm.scoreO
  for _ in [0:4] do
    let (p', r', g') := Pong.step p o a g
    p := p'
    g := g'
    r := r + r'
    if r' > 0.0 then sp := sp + 1
    if r' < 0.0 then so := so + 1
  let frames := gm.frames + 4
  let done := sp >= 21 || so >= 21 || frames >= frameCap
  ({ p := p, scoreP := sp, scoreO := so, frames := frames, g := g }, r, done)

abbrev Policy := Pong → StdGen → Act × StdGen

def randomPol : Policy := fun _ g =>
  let (i, g) := randNat g 0 2
  (Act.ofNat i, g)

/-- Reactive tracker: move toward the ball's centre. No delay, 2 px per frame. -/
def trackerPol : Policy := fun p g =>
  let bY := p.ballY + 1.0
  (if bY < p.padY - 1.0 then .up else if bY > p.padY + 1.0 then .down else .stay, g)

/-- Play one game; returns (player points - opponent points, raw frames). -/
def playGame (seed : Nat) (o : Opp) (pol : Policy) : Int × Nat := Id.run do
  let mut gm := Game.reset seed
  let mut done := false
  for _ in [0:frameCap / 4 + 1] do
    if done then break
    let (a, g') := pol gm.p gm.g
    gm := { gm with g := g' }
    let (gm', _, d) := gm.step o a
    gm := gm'
    done := d
  return ((gm.scoreP : Int) - (gm.scoreO : Int), gm.frames)

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

def runGames (n seed0 : Nat) (o : Opp) (pol : Policy) : IO (Float × Float × Nat) := do
  let mut diffs : Array Float := #[]
  let mut frames := 0
  for i in [0:n] do
    let (d, f) := playGame (seed0 + i) o pol
    diffs := diffs.push (Float.ofInt d)
    frames := frames + f
  let mean := diffs.foldl (· + ·) 0.0 / n.toFloat
  let var := diffs.foldl (fun acc d => acc + (d - mean) * (d - mean)) 0.0 / n.toFloat
  return (mean, Float.sqrt var / Float.sqrt n.toFloat, frames)

def writePgm (path : String) (w h : Nat) (px : ByteArray) : IO Unit := do
  let hdr := s!"P5\n{w} {h}\n255\n"
  IO.FS.writeBinFile path (hdr.toUTF8 ++ px)

def main (args : List String) : IO Unit := do
  let n := (args.head? >>= String.toNat?).getD 100
  let o : Opp := {}
  IO.println s!"Pong: {n} games per arm, opponent speed {fmt o.speed 2} px/frame, delay {o.delay} frames"
  let t0 ← IO.monoMsNow
  let (m1, s1, f1) ← runGames n 1000 o randomPol
  let (m2, s2, f2) ← runGames n 2000 o trackerPol
  let (m3, s3, f3) ← runGames n 3000 { speed := 2.0, delay := 1 } trackerPol
  let (m4, s4, f4) ← runGames n 4000 { speed := 4.0, delay := 1 } trackerPol
  let t1 ← IO.monoMsNow
  let total := f1 + f2 + f3 + f4
  IO.println "arm                                        mean points/game   s.e.    frames/game"
  IO.println s!"random                                     {fmt m1 2}            {fmt s1 2}    {f1 / n}"
  IO.println s!"tracker vs default opponent                {fmt m2 2}            {fmt s2 2}    {f2 / n}"
  IO.println s!"tracker vs opponent (2.0, 1) ~ symmetric    {fmt m3 2}            {fmt s3 2}    {f3 / n}"
  IO.println s!"tracker vs opponent (4.0, 1) faster         {fmt m4 2}            {fmt s4 2}    {f4 / n}"
  IO.println s!"{total} raw frames in {t1 - t0} ms = {total * 1000 / (max 1 (t1 - t0))} frames/s (single thread, incl. RNG)"
  -- determinism
  let (d1, _) := playGame 7 o trackerPol
  let (d2, _) := playGame 7 o trackerPol
  IO.println s!"same seed twice: {d1} {d2} {if d1 == d2 then "(deterministic)" else "(NOT deterministic)"}"
  -- a four-frame stack from a rally, as one strip
  let mut gm := Game.reset 11
  let mut strip := ByteArray.mk (Array.replicate (84 * 84 * 4) 0)
  for k in [0:60] do
    let (a, g') := trackerPol gm.p gm.g
    gm := { gm with g := g' }
    let (gm', _, _) := gm.step o a
    gm := gm'
    if k >= 56 then
      let fr := Pong.render gm.p
      let col := k - 56
      for y in [0:84] do
        for x in [0:84] do
          strip := strip.set! (y * 336 + col * 84 + x) (fr.get! (y * 84 + x))
  IO.FS.createDirAll ".lake/build"
  writePgm ".lake/build/pong_stack.pgm" 336 84 strip
  IO.println "wrote .lake/build/pong_stack.pgm (four consecutive observations, 336x84)"
