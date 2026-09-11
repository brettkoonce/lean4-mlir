import LeanMlir.Blackjack

/-! Blackjack under Gymnasium Blackjack-v1 `sab=True` rules — Phase 0 and 1 of
    `planning/blackjack_dqn_demo.md`: the environment, the exact DP instrument,
    tabular Q, and the four comparison arms. No stack, no GPU:
    `lake exe blackjack-env [mcHands=1000000] [qHands=1000000]`. Rung 2's DQN
    trainer wraps `BJ.step` and scores its greedy policy with `BJ.gameValue`.
    `lake exe blackjack-env play <seed> [hs...]` replays one hand from a seed
    with the DP's exact Q-values shown at every decision;
    `lake exe blackjack-env dump [qHands]` is one CSV row per state;
    `lake exe blackjack-env curve [qHands] [every]` is tabular Q's exact-value
    curve. The environment itself is `LeanMlir/Blackjack.lean`. -/

def cardStr (c : Nat) : String := if c == 1 then "A" else toString c

open BJ in
/-- Interactive replay: `blackjack-env play <seed> [moves]`, moves a string of `h`/`s`.
    The RNG is threaded, so re-running with one more letter continues the same hand.
    Every decision shows the exact Q-values of hit and stick from the DP. -/
def playHand (seed : Nat) (moves : String) : IO Unit := do
  let g0 := mkStdGen seed
  -- peek the four opening cards; `reset` draws them in this order
  let (d1, g1) := draw g0
  let (d2, g2) := draw g1
  let (p1, g3) := draw g2
  let (p2, _) := draw g3
  let (s0, gA) := reset g0
  let mut s := s0
  let mut g := gA
  let mut cards := [p1, p2]
  let handStr (cards : List Nat) (h : Hand) : String :=
    let cs := " + ".intercalate (cards.map cardStr)
    s!"{cs} = {h.total}{if h.usable then " (soft)" else ""}{if h.natural then " natural!" else ""}"
  let advise (s : State) : String :=
    let o := s.obs
    let V := valueTable o.dealer none
    let qh := hitValue V o.sum o.usable
    let qs := stickValue o.sum o.dealer s.player.natural
    s!"  exact: hit {fmt qh 4}  stick {fmt qs 4}  -> the DP says {if qh > qs then "HIT" else "STICK"}"
  IO.println s!"seed {seed}   dealer shows {cardStr d1}"
  IO.println s!"you: {handStr cards s.player}"
  IO.println (advise s)
  let mut done := false
  let mut r := 0.0
  for m in moves.toList do
    if done then break
    if m == 'h' then
      let (c, _) := draw g
      let (s', r', d, g') := step s true g
      cards := cards ++ [c]
      s := s'
      r := r'
      done := d
      g := g'
      IO.println s!"HIT   you draw {cardStr c}: {handStr cards s.player}{if s.player.bust then "  BUST" else ""}"
      if !done then IO.println (advise s)
    else if m == 's' then
      -- peek the dealer's draws; `step` replays exactly these
      let mut dc := [d1, d2]
      let mut dh := s.dealer
      let mut gg := g
      while dh.total < 17 do
        let (c, g'') := draw gg
        dc := dc ++ [c]
        dh := dh.add c
        gg := g''
      let (s', r', d, g') := step s false g
      s := s'
      r := r'
      done := d
      g := g'
      let ds := " + ".intercalate (dc.map cardStr)
      IO.println s!"STICK dealer: {ds} = {s.dealer.total}{if s.dealer.bust then "  BUST" else ""}{if s.dealer.natural then " natural" else ""}"
    else
      IO.println s!"ignored move '{m}' (use h or s)"
  if done then
    IO.println s!"result: {if r > 0.0 then "you WIN  +1" else if r < 0.0 then "you LOSE -1" else "push  0"}"
  else
    IO.println s!"your move: play {seed} {moves}h  or  play {seed} {moves}s"

open BJ in
/-- `blackjack-env dump [qHands]`: one CSV row per state (sum 4..21 × dealer × ace),
    with the DP's exact hit/stick values and the exact, tabular-Q and published actions. -/
def dumpStates (nQ : Nat) : IO Unit := do
  let Q := tabularQ nQ 0.001 0.1 42
  IO.println "usable,sum,dealer,qhit,qstick,opt,tabq,published"
  for a in [0:2] do
    let u := a == 1
    for s in [(if u then 12 else 4):22] do
      for up in [1:11] do
        let o : Obs := { sum := s, dealer := up, usable := u }
        let V := valueTable up none
        let qh := hitValue V s u
        let qs := stickValue s up false
        let act (p : Pol) : String := if p o > 0.5 then "H" else "S"
        IO.println s!"{a},{s},{up},{fmt qh 4},{fmt qs 4},{act optimalPol},{act (qPol Q)},{act publishedPol}"

open BJ in
def main (args : List String) : IO Unit := do
  if args.head? == some "play" then
    let seed := ((args.drop 1).head? >>= String.toNat?).getD 1
    let moves := ((args.drop 2).head?).getD ""
    playHand seed moves
    return
  if args.head? == some "dump" then
    dumpStates (((args.drop 1).head? >>= String.toNat?).getD 1000000)
    return
  if args.head? == some "curve" then
    let hands := ((args.drop 1).head? >>= String.toNat?).getD 1000000
    let every := ((args.drop 2).head? >>= String.toNat?).getD 10000
    IO.println "hands,exact,agreement"
    for (h, ev, ag) in tabularQCurve hands every 0.001 0.1 42 do
      IO.println s!"{h},{fmt ev 4},{ag}"
    return
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
