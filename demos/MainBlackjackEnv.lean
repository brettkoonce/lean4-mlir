/-! Blackjack under Gymnasium Blackjack-v1 `sab=True` rules — Phase 0 and 1 of
    `planning/blackjack_dqn_demo.md`: the environment, the exact DP instrument,
    tabular Q, and the four comparison arms. No stack, no GPU:
    `lake exe blackjack-env [mcHands=1000000] [qHands=1000000]`. Rung 2's DQN
    trainer wraps `BJ.step` and scores its greedy policy with `BJ.gameValue`. -/

namespace BJ

/-- Cards 1..10, ten with probability 4/13 (infinite deck). -/
def draw (g : StdGen) : Nat × StdGen :=
  let (i, g) := randNat g 0 12
  (if i < 9 then i + 1 else 10, g)

def pCard (c : Nat) : Float := if c == 10 then 4.0 / 13.0 else 1.0 / 13.0

structure Hand where
  raw : Nat      -- sum with every ace as 1
  ace : Bool     -- holds at least one ace
  n : Nat        -- cards held
deriving Repr, Inhabited

def Hand.usable (h : Hand) : Bool := h.ace && h.raw + 10 <= 21
def Hand.total (h : Hand) : Nat := if h.usable then h.raw + 10 else h.raw
def Hand.bust (h : Hand) : Bool := h.total > 21
def Hand.natural (h : Hand) : Bool := h.n == 2 && h.total == 21
def Hand.add (h : Hand) (c : Nat) : Hand := { raw := h.raw + c, ace := h.ace || c == 1, n := h.n + 1 }
def Hand.ofCards (a b : Nat) : Hand := (Hand.add { raw := 0, ace := false, n := 0 } a).add b

/-- The observation: (player total, dealer showing, usable ace). -/
structure Obs where
  sum : Nat
  dealer : Nat
  usable : Bool
deriving Repr, BEq, Inhabited

structure State where
  player : Hand
  dealer : Hand
  up : Nat
deriving Repr, Inhabited

def State.obs (s : State) : Obs := { sum := s.player.total, dealer := s.up, usable := s.player.usable }

def reset (g : StdGen) : State × StdGen :=
  let (d1, g) := draw g
  let (d2, g) := draw g
  let (p1, g) := draw g
  let (p2, g) := draw g
  ({ player := Hand.ofCards p1 p2, dealer := Hand.ofCards d1 d2, up := d1 }, g)

partial def dealerPlay (h : Hand) (g : StdGen) : Hand × StdGen :=
  if h.total < 17 then
    let (c, g) := draw g
    dealerPlay (h.add c) g
  else (h, g)

def cmpF (a b : Nat) : Float := if a > b then 1.0 else if a < b then -1.0 else 0.0

/-- Gymnasium `sab=True` settlement after a stick. -/
def settle (player dealer : Hand) : Float :=
  let ps := if player.bust then 0 else player.total
  let ds := if dealer.bust then 0 else dealer.total
  if player.natural && !dealer.natural then 1.0 else cmpF ps ds

/-- `hit = true` hits, `false` sticks. Returns (state, reward, done, rng). -/
def step (s : State) (hit : Bool) (g : StdGen) : State × Float × Bool × StdGen :=
  if hit then
    let (c, g) := draw g
    let p := s.player.add c
    if p.bust then ({ s with player := p }, -1.0, true, g)
    else ({ s with player := p }, 0.0, false, g)
  else
    let (d, g) := dealerPlay s.dealer g
    ({ s with dealer := d }, settle s.player d, true, g)

-- ───────────────────────── the exact instrument ─────────────────────────

/-- Dealer final-outcome distribution from a hand: entries 0..4 = totals 17..21,
    5 = bust. Memoised over (raw, ace) by descending raw. -/
def dealerTable : Array (Array Float) := Id.run do
  -- index raw * 2 + ace, raw in 0..30
  let mut t : Array (Array Float) := Array.replicate 62 (Array.replicate 6 0.0)
  for k in [0:31] do
    let raw := 30 - k
    for a in [0:2] do
      let ace := a == 1
      let h : Hand := { raw := raw, ace := ace, n := 3 }
      let mut row := Array.replicate 6 0.0
      if h.total >= 17 then
        if h.bust then row := row.set! 5 1.0 else row := row.set! (h.total - 17) 1.0
      else
        for c in [1:11] do
          let h' := h.add c
          let nxt := t[h'.raw * 2 + (if h'.ace then 1 else 0)]!
          for j in [0:6] do
            row := row.set! j (row[j]! + pCard c * nxt[j]!)
      t := t.set! (raw * 2 + a) row
  return t

/-- Dealer outcome given the up card, over the hidden hole card: entries 0..4 =
    totals 17..21 (not natural), 5 = bust, 6 = natural. -/
def dealerDist (up : Nat) : Array Float := Id.run do
  let mut d := Array.replicate 7 0.0
  for hole in [1:11] do
    let h := Hand.ofCards up hole
    if h.natural then d := d.set! 6 (d[6]! + pCard hole)
    else
      let row := dealerTable[h.raw * 2 + (if h.ace then 1 else 0)]!
      for j in [0:6] do
        d := d.set! j (d[j]! + pCard hole * row[j]!)
  return d

def dealerDists : Array (Array Float) := (Array.range 11).map fun u => if u == 0 then #[] else dealerDist u

/-- Expected reward of sticking on total `s` against up card `up`. -/
def stickValue (s up : Nat) (natural : Bool) : Float := Id.run do
  let d := dealerDists[up]!
  let mut v := 0.0
  for j in [0:5] do
    let t := 17 + j
    v := v + d[j]! * (if natural then 1.0 else cmpF s t)
  v := v + d[5]! * 1.0
  v := v + d[6]! * (if s == 21 then 0.0 else -1.0)   -- dealer natural: a player 21 ties, else loses
  return v

/-- A policy is P(hit | obs). -/
abbrev Pol := Obs → Float

def sidx (s : Nat) (u : Bool) : Nat := s * 2 + (if u then 1 else 0)

/-- Q_hit for one dealer up card from a value table V[sidx]. -/
def hitValue (V : Array Float) (s : Nat) (u : Bool) : Float := Id.run do
  let mut q := 0.0
  for c in [1:11] do
    let (s', u', bust) :=
      if u then
        let raw' := s - 10 + c
        if raw' + 10 <= 21 then (raw' + 10, true, false) else (raw', false, false)
      else if c == 1 && s + 11 <= 21 then (s + 11, true, false)
      else if s + c > 21 then (0, false, true)
      else (s + c, false, false)
    q := q + pCard c * (if bust then -1.0 else V[sidx s' u']!)
  return q

/-- Value table of a policy against one up card, or of the optimum if `pi = none`.
    Order: non-usable 21..11, then usable 21..12, then non-usable 10..4. -/
def valueTable (up : Nat) (pi : Option Pol) : Array Float := Id.run do
  let mut V := Array.replicate 46 0.0
  let eval (V : Array Float) (s : Nat) (u : Bool) : Float :=
    let qh := hitValue V s u
    let qs := stickValue s up false
    match pi with
    | none => max qh qs
    | some p => let ph := p { sum := s, dealer := up, usable := u }
                ph * qh + (1.0 - ph) * qs
  for k in [0:11] do
    let s := 21 - k
    V := V.set! (sidx s false) (eval V s false)
  for k in [0:10] do
    let s := 21 - k
    V := V.set! (sidx s true) (eval V s true)
  for k in [0:7] do
    let s := 10 - k
    V := V.set! (sidx s false) (eval V s false)
  return V

/-- Exact value of the game (mean reward per hand) for a policy, or the optimum. -/
def gameValue (pi : Option Pol) : Float := Id.run do
  let mut v := 0.0
  for up in [1:11] do
    let V := valueTable up pi
    for a in [1:11] do
      for b in [1:11] do
        let h := Hand.ofCards a b
        let s := h.total
        let u := h.usable
        let qh := hitValue V s u
        let qs := stickValue s up h.natural
        let val := match pi with
          | none => max qh qs
          | some p => let ph := p { sum := s, dealer := up, usable := u }
                      ph * qh + (1.0 - ph) * qs
        v := v + pCard up * pCard a * pCard b * val
  return v

/-- The optimal action (hit?) at every decision state, from the DP. -/
def optimalHit : Obs → Bool := fun o =>
  let V := valueTable o.dealer none
  hitValue V o.sum o.usable > stickValue o.sum o.dealer false

def optimalPol : Pol := fun o => if optimalHit o then 1.0 else 0.0

/-- Agreement with the optimum over the 200 decision states (12..21 × 10 × 2). -/
def agreement (p : Pol) : Nat := Id.run do
  let mut n := 0
  for s in [12:22] do
    for up in [1:11] do
      for a in [0:2] do
        let o : Obs := { sum := s, dealer := up, usable := a == 1 }
        if (p o > 0.5) == optimalHit o then n := n + 1
  return n

def chart (p : Pol) (usable : Bool) : String := Id.run do
  let mut out := s!"  {if usable then "usable ace   " else "no usable ace"}  dealer: A  2  3  4  5  6  7  8  9  10\n"
  for k in [0:10] do
    let s := 21 - k
    let mut row := s!"  {if s < 10 then " " else ""}{s}                          "
    for up in [1:11] do
      row := row ++ (if p { sum := s, dealer := up, usable := usable } > 0.5 then "H  " else "S  ")
    out := out ++ row ++ "\n"
  return out

-- ───────────────────────── the arms ─────────────────────────

def randomPol : Pol := fun _ => 0.5

def thresholdPol : Pol := fun o => if o.sum < 18 then 0.8 else 0.2

/-- The old demo's published table, verbatim (IEEE 1299399 fig. 11), indexed by
    dealer card 1..10; it ignores the usable ace. -/
def tableRow (s : Nat) : String :=
  match s with
  | 10 => "HHHHHSSHHH" | 11 => "HHSSSSSSHH" | 12 => "HSHHHHHHHH" | 13 => "HSSHHHHHHH"
  | 14 => "HSHHHHHHHH" | 15 => "HSSHHHHHHH" | 16 => "HSSSSSHHHH" | 17 => "HSSSSHHHHH"
  | 18 => "SSSSSSSSSS" | 19 => "SSSSSSSSSS" | 20 => "SSSSSSSSSS" | 21 => "SSSSSSSSSS"
  | _ => "HHHHHHHHHH"

def publishedPol : Pol := fun o => if (tableRow o.sum).toList[o.dealer - 1]! == 'H' then 1.0 else 0.0

-- ───────────────────────── Monte Carlo ─────────────────────────

def mcEval (p : Pol) (n seed : Nat) : Float × Float := Id.run do
  let mut g := mkStdGen seed
  let mut sum := 0.0
  let mut sq := 0.0
  for _ in [0:n] do
    let (s0, g0) := reset g
    g := g0
    let mut s := s0
    let mut done := false
    let mut r := 0.0
    for _ in [0:30] do
      if done then break
      let (u, g1) := randNat g 0 999999
      g := g1
      let hit := u.toFloat / 1000000.0 < p s.obs
      let (s', r', d, g2) := step s hit g
      s := s'
      r := r'
      done := d
      g := g2
    sum := sum + r
    sq := sq + r * r
  let mean := sum / n.toFloat
  let var := sq / n.toFloat - mean * mean
  return (mean, Float.sqrt var / Float.sqrt n.toFloat)

-- ───────────────────────── tabular Q ─────────────────────────

def qidx (o : Obs) (a : Nat) : Nat := ((o.sum * 11 + o.dealer) * 2 + (if o.usable then 1 else 0)) * 2 + a

/-- ε-greedy tabular Q-learning, γ = 1, constant α. Returns the greedy policy's table. -/
def tabularQ (hands : Nat) (alpha eps : Float) (seed : Nat) : Array Float := Id.run do
  let mut Q := Array.replicate (32 * 11 * 2 * 2) 0.0
  -- visit counts; the step size is max(alpha, 1 / (1 + N)) so early updates average
  -- and late ones keep a floor that tracks the moving bootstrap target
  let mut N := Array.replicate (32 * 11 * 2 * 2) 0
  let mut g := mkStdGen seed
  for _ in [0:hands] do
    let (s0, g0) := reset g
    g := g0
    let mut s := s0
    let mut done := false
    for _ in [0:30] do
      if done then break
      let o := s.obs
      let (u, g1) := randNat g 0 999999
      g := g1
      let greedy := Q[qidx o 1]! > Q[qidx o 0]!
      let (u2, g2) := randNat g 0 1
      g := g2
      let hit := if u.toFloat / 1000000.0 < eps then u2 == 1 else greedy
      let (s', r, d, g3) := step s hit g
      g := g3
      let a := if hit then 1 else 0
      let target := if d then r else max Q[qidx s'.obs 0]! Q[qidx s'.obs 1]!
      let i := qidx o a
      N := N.set! i (N[i]! + 1)
      let a_eff := max alpha (1.0 / (1.0 + (N[i]!).toFloat))
      Q := Q.set! i (Q[i]! + a_eff * (target - Q[i]!))
      s := s'
      done := d
  return Q

def qPol (Q : Array Float) : Pol := fun o => if Q[qidx o 1]! > Q[qidx o 0]! then 1.0 else 0.0

end BJ

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

open BJ in
def main (args : List String) : IO Unit := do
  let nMC := (args.head? >>= String.toNat?).getD 1000000
  let nQ := ((args.drop 1).head? >>= String.toNat?).getD 1000000
  IO.println "Blackjack, Gymnasium Blackjack-v1 sab=True rules (Sutton & Barto Example 5.1)"
  IO.println ""
  IO.println "The exact optimum by dynamic programming (H = hit, S = stick), player total by dealer card:"
  IO.println (chart optimalPol false)
  IO.println (chart optimalPol true)
  let t0 ← IO.monoMsNow
  let Q := tabularQ nQ 0.001 0.1 42
  IO.println s!"  (Q table checksum {fmt (Q.foldl (· + ·) 0.0) 3})"
  let t1 ← IO.monoMsNow
  IO.println s!"Tabular Q, {nQ} hands, step max(0.001, 1/(1+N)), eps 0.1, gamma 1 ({t1 - t0} ms):"
  IO.println (chart (qPol Q) false)
  IO.println (chart (qPol Q) true)
  IO.println "The old demo's published table (casino rules, ignores the usable ace):"
  IO.println (chart publishedPol false)
  let arms : List (String × Pol) :=
    [("random", randomPol), ("threshold heuristic", thresholdPol),
     ("published table", publishedPol), ("tabular Q", qPol Q), ("exact optimum", optimalPol)]
  IO.println s!"arm                     exact value    Monte Carlo ({nMC} hands)    agreement /200"
  for (name, p) in arms do
    let ev := gameValue (some p)
    let (mc, se) := mcEval p nMC 7
    let ag := agreement p
    let pad := String.ofList (List.replicate (24 - name.length) ' ')
    IO.println s!"{name}{pad}{fmt ev 4}        {fmt mc 4} ± {fmt se 4}               {ag}"
  IO.println s!"optimum via max in the DP (should equal the exact-optimum row): {fmt (gameValue none) 4}"
  let t2 ← IO.monoMsNow
  IO.println s!"total {t2 - t0} ms"
